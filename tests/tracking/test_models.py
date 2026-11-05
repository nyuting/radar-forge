"""Tests for radar_forge.tracking.models: layouts, motion and measurement models.

Ground truth is analytic: the published continuous and discrete white-noise blocks
(Bar-Shalom, Li and Kirubarajan 2001, §6.2 and §6.3.2), exact dead reckoning, two
continuous half steps equal to one full step, a 3-4-12-13 triangle, compass directions and
the monostatic limit of the bistatic model. Every analytic Jacobian is checked against a
central difference at many random states, in one batched call.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from radar_forge.tracking.models import (
    BistaticRangeRateModel,
    CartesianMotion,
    Coordinate,
    LinearMeasurement,
    MarginalModel,
    RadialConstantVelocity,
    RangeAzimuthElevationRangeRate,
    StateLayout,
    marginal,
)

RANGE_PERIOD = StateLayout((Coordinate("range_m", "m", period=1000.0), Coordinate("v", "m/s")))


@pytest.mark.parametrize(
    ("name", "unit", "period"),
    [("", "m", None), ("x_m", "", None), ("x_m", "m", 0.0), ("x_m", "m", np.inf)],
)
def test_a_coordinate_rejects_an_empty_name_or_a_bad_period(
    name: str, unit: str, period: float | None
) -> None:
    with pytest.raises(ValueError, match="coordinate"):
        Coordinate(name, unit, period)


def test_a_layout_rejects_a_repeated_name() -> None:
    with pytest.raises(ValueError, match="unique"):
        StateLayout((Coordinate("x_m", "m"), Coordinate("x_m", "m")))


def test_indices_follow_the_order_asked_for_and_reject_an_unknown_name() -> None:
    assert RANGE_PERIOD.indices(("v", "range_m")) == (1, 0)
    with pytest.raises(ValueError, match="not in the layout"):
        RANGE_PERIOD.indices(("z_m",))


@pytest.mark.parametrize(
    ("range_m", "centred_m", "nonnegative_m"),
    [
        (700.0, -300.0, 700.0),
        (500.0, -500.0, 500.0),
        (-500.0, -500.0, 500.0),
        (1499.0, 499.0, 499.0),
    ],
)
def test_a_wrap_lands_in_its_half_open_interval(
    range_m: float, centred_m: float, nonnegative_m: float
) -> None:
    """Centred is [-p/2, p/2), nonnegative [0, p); a value one period on is the same value."""
    value = np.array([range_m, 7.0])
    np.testing.assert_array_equal(RANGE_PERIOD.wrap(value), [centred_m, 7.0])
    np.testing.assert_array_equal(
        RANGE_PERIOD.wrap(value, interval="nonnegative"), [nonnegative_m, 7.0]
    )


def test_a_tiny_negative_value_does_not_wrap_to_a_full_period() -> None:
    """np.mod(-1e-20, 1000) rounds to 1000, outside [0, 1000); the wrap folds it to 0."""
    assert RANGE_PERIOD.wrap([-1e-20, 0.0], interval="nonnegative")[0] == 0.0


def test_a_wrap_of_a_stack_wraps_every_row_and_rejects_an_unknown_interval() -> None:
    stack = np.array([[[700.0, 1.0], [-700.0, 2.0]], [[1200.0, 3.0], [0.0, 4.0]]])
    np.testing.assert_array_equal(RANGE_PERIOD.wrap(stack)[..., 0], [[-300.0, 300.0], [200.0, 0.0]])
    with pytest.raises(ValueError, match="interval"):
        RANGE_PERIOD.wrap(stack, interval="positive")  # type: ignore[arg-type]


def test_a_residual_across_due_north_takes_the_short_way_round() -> None:
    layout = StateLayout((Coordinate("azimuth_rad", "rad", period=2 * np.pi),))
    # rtol 1e-12: one subtraction and one wrap.
    np.testing.assert_allclose(layout.residual([0.01], [2 * np.pi - 0.01]), [0.02], rtol=1e-12)


def test_a_radial_transition_is_the_textbook_one_with_no_minus_sign() -> None:
    """Range rate is the range's derivative, positive receding: F = [[1, T], [0, 1]]."""
    motion = RadialConstantVelocity()
    np.testing.assert_array_equal(motion.jacobian(np.zeros(2), 2.5), [[1.0, 2.5], [0.0, 1.0]])
    np.testing.assert_array_equal(motion.transition(np.array([1000.0, 10.0]), 2.0), [1020.0, 10.0])
    names = motion.state_layout.names
    assert names == ("bistatic_range_m", "bistatic_range_rate_mps")


@pytest.mark.parametrize("order", [1, 2])
def test_dead_reckoning_follows_the_taylor_polynomial(order: int) -> None:
    """x + v T + a T²/2, exactly, on every axis."""
    motion = CartesianMotion("xy", order)  # type: ignore[arg-type]
    acceleration_mps2 = 2.0 if order == 2 else 0.0
    axis = [100.0, 10.0, acceleration_mps2][: order + 1]
    moved = motion.transition(np.array(axis + axis), 3.0)
    expected = [100.0 + 10.0 * 3.0 + acceleration_mps2 * 3.0**2 / 2, 10.0 + 3.0 * acceleration_mps2]
    np.testing.assert_allclose(moved[:2], expected, rtol=1e-15)
    np.testing.assert_array_equal(moved[: order + 1], moved[order + 1 :])


def test_the_continuous_noise_matches_the_published_blocks() -> None:
    """Bar-Shalom, Li and Kirubarajan (2001), §6.2: CV and CA blocks, q = 2 sigma² tau."""
    step_s, q = 1.5, 2 * 3.0**2 * 0.5
    cv = CartesianMotion("x", sigma_acceleration_mps2=3.0, acceleration_correlation_time_s=0.5)
    ca = CartesianMotion("x", 2, sigma_jerk_mps3=3.0, acceleration_correlation_time_s=0.5)
    t = step_s
    cv_block = q * np.array([[t**3 / 3, t**2 / 2], [t**2 / 2, t]])
    ca_block = q * np.array(
        [[t**5 / 20, t**4 / 8, t**3 / 6], [t**4 / 8, t**3 / 3, t**2 / 2], [t**3 / 6, t**2 / 2, t]]
    )
    # rtol 1e-14: a few products of exact coefficients.
    np.testing.assert_allclose(cv.process_noise(np.zeros(2), t), cv_block, rtol=1e-14)
    np.testing.assert_allclose(ca.process_noise(np.zeros(3), t), ca_block, rtol=1e-14)


@pytest.mark.parametrize(
    "motion",
    [
        CartesianMotion("x", process_noise="dwna", sigma_acceleration_mps2=3.0),
        RadialConstantVelocity(3.0, process_noise="dwna"),
    ],
)
def test_the_discrete_noise_matches_the_published_block(motion: Any) -> None:
    """Bar-Shalom, Li and Kirubarajan (2001), §6.3.2: sigma² Γ Γᵀ with Γ = (T²/2, T)."""
    t = 2.0
    expected = 9.0 * np.array([[t**4 / 4, t**3 / 2], [t**3 / 2, t**2]])
    np.testing.assert_allclose(motion.process_noise(np.zeros(2), t), expected, rtol=1e-14)


def test_the_defaults_give_a_noise_intensity_of_32() -> None:
    """sigma = 4 m/s², tau = 1 s: q = 2 · 16 · 1, the velocity variance added per second."""
    assert CartesianMotion("x").process_noise(np.zeros(2), 1.0)[1, 1] == 32.0


@pytest.mark.parametrize("order", [1, 2])
def test_two_continuous_half_steps_add_the_same_noise_as_one_full_step(order: int) -> None:
    """F(T/2) Q(T/2) F(T/2)ᵀ + Q(T/2) = Q(T): Q does not depend on how time is cut up."""
    motion = CartesianMotion("x", order)  # type: ignore[arg-type]
    state = np.zeros(order + 1)
    half_f, half_q = motion.jacobian(state, 0.7), motion.process_noise(state, 0.7)
    # rtol 1e-12: polynomial identities evaluated in float64.
    np.testing.assert_allclose(
        half_f @ half_q @ half_f.T + half_q, motion.process_noise(state, 1.4), rtol=1e-12
    )


def test_two_discrete_half_steps_add_less_velocity_noise_than_one_full_step() -> None:
    """DWNA adds sigma² T² per step: 2 · sigma² (T/2)² is half of sigma² T²."""
    motion = CartesianMotion("x", process_noise="dwna")
    half_f, half_q = motion.jacobian(np.zeros(2), 0.5), motion.process_noise(np.zeros(2), 0.5)
    split = half_f @ half_q @ half_f.T + half_q
    assert split[1, 1] == pytest.approx(motion.process_noise(np.zeros(2), 1.0)[1, 1] / 2)


def test_the_two_models_add_the_same_velocity_noise_when_tau_is_half_the_step() -> None:
    """2 sigma² tau T = sigma² T² exactly when tau = T/2 (module docstring)."""
    cwna = CartesianMotion("x", acceleration_correlation_time_s=0.75)
    dwna = CartesianMotion("x", process_noise="dwna")
    assert cwna.process_noise(np.zeros(2), 1.5)[1, 1] == pytest.approx(
        dwna.process_noise(np.zeros(2), 1.5)[1, 1], rel=1e-14
    )


def test_the_axes_are_interleaved_and_their_noise_independent() -> None:
    motion = CartesianMotion()
    assert motion.state_layout.names == ("x_m", "xdot_mps", "y_m", "ydot_mps", "z_m", "zdot_mps")
    noise = motion.process_noise(np.zeros(6), 1.0)
    assert np.all(noise[:2, 2:] == 0.0) and np.all(noise[2:4, 4:] == 0.0)


def test_a_batched_step_equals_one_state_at_a_time(rng: np.random.Generator) -> None:
    """Each row moves by its own dt, in one call, as the bank's rows do."""
    motion = CartesianMotion()
    states, steps_s = rng.normal(0.0, 100.0, (5, 6)), rng.uniform(0.1, 3.0, 5)
    one_by_one = [motion.transition(s, t) for s, t in zip(states, steps_s, strict=True)]
    noise = [motion.process_noise(s, t) for s, t in zip(states, steps_s, strict=True)]
    np.testing.assert_array_equal(motion.transition(states, steps_s), one_by_one)
    np.testing.assert_array_equal(motion.process_noise(states, steps_s), noise)


@pytest.mark.parametrize("motion", [CartesianMotion("xy", 2), RadialConstantVelocity()])
def test_every_motion_model_is_linear_as_the_taylor_prediction_assumes(
    motion: Any, rng: np.random.Generator
) -> None:
    """transition(x) is jacobian(x) @ x, so predicting the mean as F x needs one F."""
    state = rng.normal(0.0, 10.0, motion.state_layout.dimension)
    # rtol 1e-14: the same matrix applied two ways.
    np.testing.assert_allclose(
        motion.transition(state, 1.3), motion.jacobian(state, 1.3) @ state, rtol=1e-14
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"axes": ""},
        {"axes": "xw"},
        {"axes": "xx"},
        {"order": 3},
        {"sigma_acceleration_mps2": -1.0},
        {"sigma_acceleration_mps2": np.nan},
        {"acceleration_correlation_time_s": 0.0},
        {"process_noise": "singer"},
        {"order": 2, "process_noise": "dwna"},
    ],
)
def test_impossible_motion_settings_are_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"axes|process_noise|sigma"):
        CartesianMotion(**settings)


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
