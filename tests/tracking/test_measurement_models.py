"""Tests for radar_forge.tracking.measurement_models.

Ground truth is geometric: a 3-4-12-13 triangle, compass directions, a target moving along
its line of sight, and the monostatic limit of the bistatic model. Every analytic Jacobian is
checked against a central difference at many random states, all in one batched call.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from radar_forge.tracking.measurement_models import (
    BistaticRangeRateModel,
    LinearMeasurement,
    MarginalModel,
    RangeAzimuthElevationRangeRate,
    marginal,
)
from radar_forge.tracking.models import CartesianMotion, Coordinate, StateLayout

LAYOUT_3D = CartesianMotion().state_layout
SPHERICAL = ("range_m", "azimuth_rad", "elevation_rad", "range_rate_mps")


def _state(position_m: Any, velocity_mps: Any = (0.0, 0.0, 0.0)) -> NDArray[np.float64]:
    """A 3-D CV state in the interleaved order x, xdot, y, ydot, z, zdot."""
    return np.ravel(np.column_stack([position_m, velocity_mps]))


def _central_difference(model: Any, states: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerical ∂h/∂x for a stack of states, every column of every state in one call."""
    n_state = states.shape[-1]
    # A relative step: positions are kilometres and velocities tens of m/s.
    steps = 1e-5 * np.maximum(1.0, np.abs(states))[:, :, None] * np.eye(n_state)
    plus = model.predict(states[:, None, :] + steps)
    minus = model.predict(states[:, None, :] - steps)
    return np.swapaxes((plus - minus) / (2 * np.diagonal(steps, axis1=1, axis2=2))[..., None], 1, 2)


def _random_states(rng: np.random.Generator, layout: StateLayout) -> NDArray[np.float64]:
    """Fifty states with east in [1, 20] km, so azimuth stays away from its wrap at north."""
    states = rng.uniform(-100.0, 100.0, (50, layout.dimension))
    positions = list(layout.indices([n for n in ("x_m", "y_m", "z_m") if n in layout.names]))
    states[:, positions] = rng.uniform(-20_000.0, 20_000.0, (50, len(positions)))
    states[:, positions[0]] = rng.uniform(1_000.0, 20_000.0, 50)
    return states


def test_a_linear_measurement_reports_coordinates_in_the_order_named() -> None:
    model = LinearMeasurement(LAYOUT_3D, ("z_m", "x_m"))
    np.testing.assert_array_equal(model.predict(_state([1.0, 2.0, 3.0])), [3.0, 1.0])
    np.testing.assert_array_equal(model.jacobian(np.zeros((4, 6))).shape, (4, 2, 6))
    np.testing.assert_array_equal(model.jacobian(np.zeros(6))[0], [0, 0, 0, 0, 1, 0])


def test_a_measured_period_wraps_the_measurement_and_not_the_state() -> None:
    model = LinearMeasurement(LAYOUT_3D, ("x_m",), periods={"x_m": 1000.0})
    assert model.measurement_layout.coordinates[0].period == 1000.0
    assert LAYOUT_3D.coordinates[0].period is None


@pytest.mark.parametrize(
    ("names", "periods"), [((), None), (("x_m", "x_m"), None), (("x_m",), {"y_m": 1.0})]
)
def test_a_linear_measurement_rejects_bad_names_or_periods(names: Any, periods: Any) -> None:
    with pytest.raises(ValueError, match="names"):
        LinearMeasurement(LAYOUT_3D, names, periods=periods)


@pytest.mark.parametrize(
    ("position_m", "azimuth_rad"),
    [
        ([0.0, 1000.0, 0.0], 0.0),
        ([1000.0, 0.0, 0.0], np.pi / 2),
        ([-1000.0, 0.0, 0.0], 1.5 * np.pi),
    ],
)
def test_azimuth_is_clockwise_from_north(position_m: list[float], azimuth_rad: float) -> None:
    model = RangeAzimuthElevationRangeRate(LAYOUT_3D, np.zeros(3), ("azimuth_rad",))
    # atol 1e-12 rad: atan2 of exact inputs.
    np.testing.assert_allclose(model.predict(_state(position_m)), [azimuth_rad], atol=1e-12)


def test_elevation_is_up_from_the_local_horizontal_and_measured_from_the_site() -> None:
    site_m = np.array([100.0, 200.0, 50.0])
    model = RangeAzimuthElevationRangeRate(LAYOUT_3D, site_m, ("range_m", "elevation_rad"))
    z = model.predict(_state(site_m + np.array([300.0, 0.0, 300.0])))
    np.testing.assert_allclose(z, [300.0 * np.sqrt(2), np.pi / 4], rtol=1e-14)


@pytest.mark.parametrize(("speed_mps", "sign"), [(10.0, 1.0), (-10.0, -1.0)])
def test_range_rate_is_positive_receding(speed_mps: float, sign: float) -> None:
    """A target moving away along the line of sight has range rate +speed."""
    model = RangeAzimuthElevationRangeRate(LAYOUT_3D, np.zeros(3), ("range_rate_mps",))
    direction = np.array([3.0, 4.0, 12.0]) / 13.0
    z = model.predict(_state(1300.0 * direction, speed_mps * direction))
    np.testing.assert_allclose(z, [sign * 10.0], rtol=1e-14)


def test_only_azimuth_wraps() -> None:
    model = RangeAzimuthElevationRangeRate(LAYOUT_3D, np.zeros(3))
    periods = [c.period for c in model.measurement_layout.coordinates]
    assert periods == [None, 2 * np.pi, None, None]


def test_a_position_only_state_needs_no_velocity_without_range_rate() -> None:
    layout = StateLayout(tuple(Coordinate(n, "m") for n in ("x_m", "y_m", "z_m")))
    model = RangeAzimuthElevationRangeRate(layout, np.zeros(3), ("range_m", "azimuth_rad"))
    np.testing.assert_allclose(model.predict(np.array([3.0, 4.0, 12.0])), [13.0, np.arctan2(3, 4)])


@pytest.mark.parametrize(
    ("components", "fixed"),
    [
        ((), {"y_m": 0.0}),
        (("range_m", "range_m"), None),
        (("doppler_hz",), None),
        (SPHERICAL, {"w": 1.0}),
    ],
)
def test_a_spherical_model_rejects_bad_components_or_fixed_values(
    components: tuple[str, ...], fixed: Any
) -> None:
    with pytest.raises(ValueError, match=r"components|fixed"):
        RangeAzimuthElevationRangeRate(LAYOUT_3D, np.zeros(3), components, fixed)


@pytest.mark.parametrize(
    ("axes", "fixed", "components"),
    [
        ("xyz", None, SPHERICAL),
        ("xy", {"z_m": 3000.0, "zdot_mps": 0.0}, ("range_m", "azimuth_rad", "range_rate_mps")),
        ("xyz", None, ("elevation_rad", "range_m")),
    ],
    ids=["3d", "2d-fixed-height", "no-rate"],
)
def test_the_spherical_jacobian_matches_a_central_difference(
    axes: str, fixed: Any, components: tuple[str, ...], rng: np.random.Generator
) -> None:
    layout = CartesianMotion(axes).state_layout
    model = RangeAzimuthElevationRangeRate(layout, [10.0, -20.0, 5.0], components, fixed)
    states = _random_states(rng, layout)
    # rtol 1e-6, atol 1e-10: the best a central difference reaches on a norm, as in
    # tests/core/tracking/test_measurement_models.py.
    np.testing.assert_allclose(
        model.jacobian(states), _central_difference(model, states), rtol=1e-6, atol=1e-10
    )


@pytest.mark.parametrize(
    ("axes", "fixed"), [("xyz", None), ("xy", {"z_m": 3000.0, "zdot_mps": 0.0})]
)
def test_the_bistatic_jacobian_matches_a_central_difference(
    axes: str, fixed: Any, rng: np.random.Generator
) -> None:
    layout = CartesianMotion(axes).state_layout
    model = BistaticRangeRateModel(layout, np.zeros(3), [20_000.0, 1_000.0, 50.0], fixed)
    states = _random_states(rng, layout)
    # rtol 1e-6, atol 1e-10: as for the spherical model above.
    np.testing.assert_allclose(
        model.jacobian(states), _central_difference(model, states), rtol=1e-6, atol=1e-10
    )


def test_the_bistatic_range_is_the_path_from_transmitter_to_target_to_receiver() -> None:
    """3-4 from the transmitter, 12-13 on to a receiver straight above it: 5 + 13."""
    model = BistaticRangeRateModel(LAYOUT_3D, np.zeros(3), [0.0, 0.0, 12.0])
    np.testing.assert_allclose(model.predict(_state([3.0, 4.0, 0.0]))[0], 5.0 + 13.0, rtol=1e-15)


def test_the_bistatic_rate_is_positive_when_the_path_grows() -> None:
    model = BistaticRangeRateModel(LAYOUT_3D, np.zeros(3), [20_000.0, 0.0, 0.0])
    away = model.predict(_state([10_000.0, 5_000.0, 0.0], [0.0, 50.0, 0.0]))
    assert away[1] > 0.0


def test_a_monostatic_bistatic_model_is_twice_the_spherical_range_and_rate(
    rng: np.random.Generator,
) -> None:
    """tx == rx: the path is out and back, so z = (2 r, 2 r') with no special case."""
    site_m = np.array([500.0, -300.0, 20.0])
    bistatic = BistaticRangeRateModel(LAYOUT_3D, site_m, site_m)
    spherical = RangeAzimuthElevationRangeRate(LAYOUT_3D, site_m, ("range_m", "range_rate_mps"))
    states = _random_states(rng, LAYOUT_3D)
    # rtol 1e-12: the same geometry summed in a different order.
    np.testing.assert_allclose(bistatic.predict(states), 2 * spherical.predict(states), rtol=1e-12)
    np.testing.assert_allclose(
        bistatic.jacobian(states), 2 * spherical.jacobian(states), rtol=1e-12, atol=1e-15
    )


def test_a_batched_prediction_equals_one_state_at_a_time(rng: np.random.Generator) -> None:
    model = RangeAzimuthElevationRangeRate(LAYOUT_3D, np.zeros(3))
    states = _random_states(rng, LAYOUT_3D).reshape(5, 10, 6)
    one_by_one = np.array([[model.predict(s) for s in row] for row in states])
    np.testing.assert_array_equal(model.predict(states), one_by_one)


def test_missing_coordinates_must_be_fixed_exactly() -> None:
    layout = CartesianMotion("xy").state_layout
    with pytest.raises(ValueError, match="fixed"):
        BistaticRangeRateModel(layout, np.zeros(3), np.ones(3), {"z_m": 3000.0})
    with pytest.raises(ValueError, match="three finite"):
        BistaticRangeRateModel(layout, np.zeros(2), np.ones(3), {"z_m": 1.0, "zdot_mps": 0.0})


def test_a_marginal_keeps_the_named_components_rows_and_periods() -> None:
    base = RangeAzimuthElevationRangeRate(LAYOUT_3D, np.zeros(3))
    part = marginal(base, ("azimuth_rad", "range_m"))
    state = _state([3000.0, 4000.0, 1200.0], [10.0, 0.0, 0.0])
    assert isinstance(part, MarginalModel) and part.base is base
    np.testing.assert_array_equal(part.predict(state), base.predict(state)[[1, 0]])
    np.testing.assert_array_equal(part.jacobian(state), base.jacobian(state)[[1, 0]])
    assert part.measurement_layout.coordinates[0].period == 2 * np.pi


@pytest.mark.parametrize("names", [(), ("range_m", "range_m"), ("doppler_hz",)])
def test_a_marginal_needs_known_unique_components(names: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match=r"marginal|not in the layout"):
        marginal(LinearMeasurement(LAYOUT_3D, ("x_m",)), names)
