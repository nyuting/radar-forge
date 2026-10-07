"""Tests for radar_forge.core.tracking.coordinates: layouts, wrapping and state estimates."""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.tracking.coordinates import Coordinate, StateEstimate, StateLayout
from radar_forge.core.tracking.motion import CartesianMotion

ORIGIN = (36.00250, -78.94100, 60.0)

# A folded range with a period of 1000 m, beside an ordinary coordinate. The
# period is an integer so every wrap below is exact in float64.
PERIOD_M = 1000.0
FOLDED = StateLayout(
    (Coordinate("range_m", "m", period=PERIOD_M), Coordinate("range_rate_mps", "m/s"))
)
AZIMUTH = StateLayout((Coordinate("azimuth_deg", "deg", period=360.0),))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261005)


# --------------------------------------------------------------------------- #
# Coordinate and StateLayout construction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "unit", "period"),
    [("", "m", None), ("x_m", "", None), ("x_m", "m", 0.0), ("x_m", "m", np.inf)],
    ids=["empty-name", "empty-unit", "zero-period", "infinite-period"],
)
def test_a_coordinate_rejects_an_empty_name_or_a_bad_period(
    name: str, unit: str, period: float | None
) -> None:
    with pytest.raises(ValueError, match="coordinate"):
        Coordinate(name, unit, period)


def test_a_layout_rejects_a_repeated_name() -> None:
    with pytest.raises(ValueError, match="unique coordinates"):
        StateLayout((Coordinate("x_m", "m"), Coordinate("x_m", "m")))


@pytest.mark.parametrize(
    "origin", [(91.0, 0.0, 0.0), (0.0, 181.0, 0.0), (0.0, 0.0, np.nan)], ids=["lat", "lon", "nan"]
)
def test_a_layout_rejects_an_impossible_origin(origin: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError, match="ENU origin"):
        StateLayout((Coordinate("x_m", "m"),), "ENU", origin)


def test_a_layout_stores_its_coordinates_as_a_tuple_the_caller_cannot_change() -> None:
    """A list passed in is copied to a tuple, so changing the list later changes nothing."""
    coordinates = [Coordinate("x_m", "m")]
    layout = StateLayout(coordinates)  # type: ignore[arg-type]  # a list, on purpose
    coordinates.append(Coordinate("y_m", "m"))
    assert isinstance(layout.coordinates, tuple)
    assert layout.names == ("x_m",)


def test_indices_follow_the_order_asked_for() -> None:
    assert FOLDED.indices(("range_rate_mps", "range_m")) == (1, 0)


def test_indices_reject_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="not in the layout"):
        FOLDED.indices(("x_m",))


# --------------------------------------------------------------------------- #
# The wrap helper, at its boundaries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("range_m", "expected_m"),
    [(0.0, 0.0), (499.0, 499.0), (500.0, -500.0), (-500.0, -500.0), (1700.0, -300.0)],
    ids=["zero", "inside", "upper-edge-is-open", "lower-edge-is-closed", "two-folds"],
)
def test_a_centred_wrap_lands_in_the_half_open_centred_interval(
    range_m: float, expected_m: float
) -> None:
    wrapped = FOLDED.wrap(np.array([range_m, 0.0]))
    # Exact: the inputs and the period are small integers, so the mod is exact.
    np.testing.assert_allclose(wrapped[0], expected_m, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("range_m", "expected_m"),
    [(0.0, 0.0), (999.0, 999.0), (1000.0, 0.0), (-300.0, 700.0), (2500.0, 500.0)],
    ids=["zero", "inside", "upper-edge-is-open", "negative", "two-folds"],
)
def test_a_nonnegative_wrap_lands_in_zero_to_one_period(range_m: float, expected_m: float) -> None:
    wrapped = FOLDED.wrap(np.array([range_m, 0.0]), interval="nonnegative")
    # Exact: the inputs and the period are small integers, so the mod is exact.
    np.testing.assert_allclose(wrapped[0], expected_m, rtol=0, atol=0)


def test_a_tiny_negative_value_does_not_wrap_to_a_full_period() -> None:
    # np.mod(-1e-20, 1000.0) rounds to exactly 1000.0, outside [0, 1000).
    wrapped = FOLDED.wrap(np.array([-1e-20, 0.0]), interval="nonnegative")
    assert 0.0 <= wrapped[0] < PERIOD_M


def test_a_wrap_leaves_a_coordinate_without_a_period_alone() -> None:
    wrapped = FOLDED.wrap(np.array([0.0, 1.0e6]))
    assert wrapped[1] == 1.0e6


def test_a_stack_wraps_row_by_row() -> None:
    stack = np.array([[1700.0, 5.0], [-300.0, 6.0]])
    expected = np.array([FOLDED.wrap(row) for row in stack])
    # Exact: the same element-wise operations on the same numbers.
    np.testing.assert_allclose(FOLDED.wrap(stack), expected, rtol=0, atol=0)


def test_a_wrap_rejects_an_unknown_interval() -> None:
    with pytest.raises(ValueError, match="interval"):
        FOLDED.wrap(np.zeros(2), interval="positive")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [np.zeros(3), np.zeros((2, 2, 2)), np.array([np.nan, 0.0])],
    ids=["wrong-length", "three-dimensional", "nan"],
)
def test_a_wrap_rejects_a_wrong_shape_or_a_nan(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="must be finite with shape"):
        FOLDED.wrap(value)


def test_normalise_is_the_centred_wrap() -> None:
    # Exact: normalise calls wrap.
    np.testing.assert_allclose(
        FOLDED.normalise([1700.0, 0.0]), FOLDED.wrap([1700.0, 0.0]), rtol=0, atol=0
    )


# --------------------------------------------------------------------------- #
# Residuals and the circular mean
# --------------------------------------------------------------------------- #


def test_a_residual_across_due_north_takes_the_short_way_round() -> None:
    # 359 deg - 1 deg is -2 deg; the inputs are integers, so the result is exact.
    np.testing.assert_allclose(AZIMUTH.residual([359.0], [1.0]), [-2.0], rtol=0, atol=0)


def test_a_residual_of_a_stack_against_one_vector_matches_row_by_row(
    rng: np.random.Generator,
) -> None:
    stack = rng.uniform(-3000.0, 3000.0, (5, 2))
    reference = np.array([100.0, 1.0])
    expected = np.array([FOLDED.residual(row, reference) for row in stack])
    # Exact: the same element-wise operations on the same numbers.
    np.testing.assert_allclose(FOLDED.residual(stack, reference), expected, rtol=0, atol=0)


def test_a_residual_rejects_a_wrong_shape() -> None:
    with pytest.raises(ValueError, match="must be finite with shape"):
        FOLDED.residual(np.zeros(3), np.zeros(2))


def test_the_mean_of_179_and_minus_179_degrees_is_180() -> None:
    mean = AZIMUTH.weighted_mean([[179.0], [-179.0]], [0.5, 0.5])
    # The two sines cancel exactly, so atan2 returns pi to within one ulp.
    np.testing.assert_allclose(abs(mean[0]), 180.0, rtol=1e-15)


def test_an_ordinary_coordinate_uses_the_plain_weighted_sum() -> None:
    mean = FOLDED.weighted_mean([[0.0, 2.0], [0.0, 6.0]], [0.25, 0.75])
    # 0.25 * 2 + 0.75 * 6 = 5, a few exact float64 operations.
    np.testing.assert_allclose(mean[1], 5.0, rtol=1e-15)


def test_opposite_points_with_equal_weight_have_no_mean() -> None:
    with pytest.raises(ValueError, match="undefined"):
        AZIMUTH.weighted_mean([[0.0], [180.0]], [0.5, 0.5])


def test_mean_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        AZIMUTH.weighted_mean([[0.0], [10.0]], [0.5, 0.6])


@pytest.mark.parametrize(
    "points",
    [np.zeros(2), np.zeros((2, 2)), np.array([[np.inf], [0.0]])],
    ids=["one-dimensional", "wrong-width", "infinite"],
)
def test_mean_points_need_shape_n_points_by_n_state(points: np.ndarray) -> None:
    with pytest.raises(ValueError, match="points must be finite"):
        AZIMUTH.weighted_mean(points, [0.5, 0.5])


def test_mean_weights_need_one_per_point() -> None:
    with pytest.raises(ValueError, match="weights"):
        AZIMUTH.weighted_mean([[0.0], [10.0]], [1.0])


# --------------------------------------------------------------------------- #
# StateEstimate
# --------------------------------------------------------------------------- #


def test_a_covariance_with_a_negative_eigenvalue_is_rejected() -> None:
    motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    with pytest.raises(ValueError, match="semidefinite"):
        StateEstimate(np.zeros(2), np.diag([1.0, -1.0]), 0, motion.state_layout)


def test_an_estimate_wraps_its_mean() -> None:
    estimate = StateEstimate(np.array([1700.0, 0.0]), np.eye(2), 0.0, FOLDED)
    # Exact: an integer input and period.
    np.testing.assert_allclose(estimate.mean, [-300.0, 0.0], rtol=0, atol=0)


def test_an_estimate_is_detached_from_the_caller_arrays() -> None:
    mean, covariance = np.array([1.0, 2.0]), np.eye(2)
    estimate = StateEstimate(mean, covariance, 0.0, FOLDED)
    mean[:] = 0.0
    covariance[:] = 0.0
    np.testing.assert_array_equal(estimate.mean, [1.0, 2.0])
    np.testing.assert_array_equal(estimate.covariance, np.eye(2))


def test_an_estimate_is_read_only() -> None:
    estimate = StateEstimate(np.zeros(2), np.eye(2), 0.0, FOLDED)
    assert not estimate.mean.flags.writeable
    assert not estimate.covariance.flags.writeable


def test_an_estimate_rejects_a_mean_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="mean must be finite"):
        StateEstimate(np.zeros(3), np.eye(2), 0.0, FOLDED)


def test_an_estimate_rejects_a_nonfinite_time() -> None:
    with pytest.raises(ValueError, match="timestamp_s must be finite"):
        StateEstimate(np.zeros(2), np.eye(2), np.nan, FOLDED)
