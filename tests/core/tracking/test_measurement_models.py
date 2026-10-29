"""Tests for radar_forge.core.tracking.measurement_models: measurements, sensors, models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from radar_forge.core.geodesy import enu_to_range_azimuth_elevation
from radar_forge.core.radar import BistaticRadar, Radar, Receiver, Transmitter
from radar_forge.core.tracking.coordinates import Coordinate, StateEstimate, StateLayout
from radar_forge.core.tracking.measurement_models import (
    BistaticRangeDopplerModel,
    CartesianPosition,
    DifferentiableMeasurementModel,
    MarginalModel,
    Measurement,
    MeasurementBatch,
    RangeAzimuthElevationRangeRate,
    SensorPose,
    SensorRegistration,
    _Geometry,
    marginal,
)
from radar_forge.core.tracking.motion import CartesianMotion

ORIGIN = (36.00250, -78.94100, 60.0)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261005)


def _scene(kind: str = "CA") -> tuple[CartesianMotion, StateEstimate, SensorPose]:
    """A target at (3, 4, 12) m moving at (-3, -4, 0) m/s, and a site at the origin."""
    motion = CartesianMotion(dict.fromkeys("xyz", kind), origin_lla_deg_m=ORIGIN)
    values = {
        "x_m": 3,
        "y_m": 4,
        "z_m": 12,
        "xdot_mps": -3,
        "ydot_mps": -4,
        "zdot_mps": 0,
        "xddot_mps2": 3,
        "yddot_mps2": 4,
        "zddot_mps2": -12,
    }
    estimate = StateEstimate(
        np.array([values[n] for n in motion.state_layout.names], dtype=np.float64),
        np.eye(motion.state_layout.dimension),
        0,
        motion.state_layout,
    )
    pose = SensorPose(np.zeros(3), "ENU", ORIGIN)
    return motion, estimate, pose


def _pose(position_m: list[float]) -> SensorPose:
    return SensorPose(np.array(position_m), "ENU", ORIGIN)


def _transmitter() -> Transmitter:
    return Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3)


def _central_difference(model: DifferentiableMeasurementModel, state: NDArray[np.float64]) -> Any:
    """Numerical d h / d x, column by column, to check an analytic Jacobian against."""
    columns = []
    for index in range(state.size):
        # A loop over state elements: each column needs its own perturbed pair of states.
        # A relative step: positions are kilometres and velocities tens of m/s, so one
        # absolute step cannot suit both (as tests/core/test_tracking.py).
        step = 1e-5 * max(1.0, abs(float(state[index])))
        offset = np.zeros(state.size)
        offset[index] = step
        columns.append((model.predict(state + offset) - model.predict(state - offset)) / (2 * step))
    return np.stack(columns, axis=-1)


# --------------------------------------------------------------------------- #
# CartesianPosition
# --------------------------------------------------------------------------- #


def test_cartesian_position_reports_coordinates_in_measurement_order() -> None:
    _, estimate, _ = _scene("CV")
    model = CartesianPosition(estimate.state_layout, ("ydot_mps", "x_m"))
    # Exact: predict only copies two elements.
    np.testing.assert_allclose(model.predict(estimate.mean), [-4, 3], rtol=0, atol=0)


def test_cartesian_position_of_a_stack_equals_the_stacked_single_predictions(
    rng: np.random.Generator,
) -> None:
    _, estimate, _ = _scene("CV")
    model = CartesianPosition(estimate.state_layout, ("ydot_mps", "x_m"))
    states = rng.normal(0.0, 100.0, (7, estimate.state_layout.dimension))
    expected = np.array([model.predict(s) for s in states])
    # Exact: both paths copy the same elements.
    np.testing.assert_allclose(model.predict(states), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("names", "periods", "match"),
    [
        ((), None, "nonempty and unique"),
        (("x_m", "x_m"), None, "nonempty and unique"),
        (("q_m",), None, "not in the layout"),
        (("x_m",), {"y_m": 10.0}, "periods must refer"),
    ],
    ids=["empty", "repeated", "unknown", "period-for-unobserved"],
)
def test_cartesian_position_rejects_bad_names_or_periods(
    names: tuple[str, ...], periods: dict[str, float] | None, match: str
) -> None:
    _, estimate, _ = _scene("CV")
    with pytest.raises(ValueError, match=match):
        CartesianPosition(estimate.state_layout, names, periods)


def test_the_cartesian_jacobian_picks_the_reported_coordinates() -> None:
    motion, estimate, _ = _scene("CV")
    model = CartesianPosition(motion.state_layout, ("z_m", "xdot_mps"))
    expected = np.zeros((2, motion.state_layout.dimension))
    expected[0, motion.state_layout.names.index("z_m")] = 1.0
    expected[1, motion.state_layout.names.index("xdot_mps")] = 1.0
    np.testing.assert_array_equal(model.jacobian(estimate.mean), expected)


def test_cartesian_position_rejects_a_state_of_the_wrong_length() -> None:
    _, estimate, _ = _scene("CV")
    model = CartesianPosition(estimate.state_layout, ("x_m",))
    with pytest.raises(ValueError, match="must be finite with shape"):
        model.predict(np.zeros(3))


# --------------------------------------------------------------------------- #
# BistaticRangeDopplerModel
# --------------------------------------------------------------------------- #


def test_the_bistatic_path_range_is_out_and_back_for_co_sited_assets() -> None:
    _, estimate, pose = _scene("CV")
    model = BistaticRangeDopplerModel(estimate.state_layout, pose, pose)
    # A 3-4-12 triangle: 13 m each way. A norm and an add, so float64 round-off only.
    np.testing.assert_allclose(model.predict(estimate.mean)[0], 26.0, rtol=1e-12)


def test_the_bistatic_path_rate_is_positive_when_closing() -> None:
    _, estimate, pose = _scene("CV")
    model = BistaticRangeDopplerModel(estimate.state_layout, pose, pose)
    # Closing at 25/13 m/s each way: (3, 4, 12) / 13 . (-3, -4, 0) = -25/13.
    np.testing.assert_allclose(model.predict(estimate.mean)[1], 50 / 13, rtol=1e-12)


def test_excess_range_subtracts_the_baseline() -> None:
    _, estimate, _ = _scene("CV")
    layout = estimate.state_layout
    # Sites 100 m apart; the target 120 m above the midpoint is 130 m from each
    # (a 5-12-13 triangle scaled by 10), so the path is 260 m and the excess 160 m.
    state = np.zeros(layout.dimension)
    state[list(layout.indices(("x_m", "z_m")))] = [50.0, 120.0]
    tx, rx = _pose([0.0, 0.0, 0.0]), _pose([100.0, 0.0, 0.0])
    excess = BistaticRangeDopplerModel(layout, tx, rx).predict(state)[0]
    # Two norms and two adds of exact integers, so float64 round-off only.
    np.testing.assert_allclose(excess, 160.0, rtol=1e-12)


def test_without_excess_range_the_path_is_the_full_sum() -> None:
    _, estimate, _ = _scene("CV")
    layout = estimate.state_layout
    state = np.zeros(layout.dimension)
    state[list(layout.indices(("x_m", "z_m")))] = [50.0, 120.0]
    tx, rx = _pose([0.0, 0.0, 0.0]), _pose([100.0, 0.0, 0.0])
    total = BistaticRangeDopplerModel(layout, tx, rx, excess_range=False).predict(state)[0]
    # 130 m + 130 m; float64 round-off only.
    np.testing.assert_allclose(total, 260.0, rtol=1e-12)


def test_bistatic_predict_of_a_stack_equals_the_stacked_single_predictions(
    rng: np.random.Generator,
) -> None:
    _, estimate, _ = _scene("CA")
    model = BistaticRangeDopplerModel(
        estimate.state_layout, _pose([0.0, 0.0, 0.0]), _pose([2000.0, 500.0, 10.0])
    )
    states = rng.normal(0.0, 1000.0, (9, estimate.state_layout.dimension))
    expected = np.array([model.predict(s) for s in states])
    batched = model.predict(states)
    assert batched.shape == (9, 2)
    # The same operations in a different order of reduction; a few ulps at most.
    np.testing.assert_allclose(batched, expected, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("site", ["transmitter", "receiver"])
def test_a_target_on_a_site_is_a_singular_geometry(site: str) -> None:
    _, estimate, _ = _scene("CV")
    layout = estimate.state_layout
    tx, rx = _pose([0.0, 0.0, 0.0]), _pose([100.0, 0.0, 0.0])
    model = BistaticRangeDopplerModel(layout, tx, rx)
    on_site = np.zeros(layout.dimension)
    on_site[layout.indices(("x_m",))[0]] = 0.0 if site == "transmitter" else 100.0
    with pytest.raises(ValueError, match="singular"):
        model.predict(on_site)


def test_one_singular_point_in_a_stack_is_rejected() -> None:
    _, estimate, pose = _scene("CV")
    model = BistaticRangeDopplerModel(estimate.state_layout, pose, pose)
    stack = np.stack([estimate.mean, np.zeros(estimate.state_layout.dimension)])
    with pytest.raises(ValueError, match="singular"):
        model.predict(stack)


@pytest.mark.parametrize("excess_range", [True, False])
@pytest.mark.parametrize(
    ("axes", "fixed"),
    [("xyz", None), ("xy", {"z_m": 3000.0, "zdot_mps": 0.0})],
    ids=["3d", "2d-fixed-height"],
)
def test_the_bistatic_jacobian_matches_a_central_difference(
    axes: str, fixed: dict[str, float] | None, excess_range: bool, rng: np.random.Generator
) -> None:
    """The analytic Jacobian of the class Notes, checked at many states."""
    motion = CartesianMotion(dict.fromkeys(axes, "CV"), origin_lla_deg_m=ORIGIN)
    layout = motion.state_layout
    model = BistaticRangeDopplerModel(
        layout,
        _pose([0.0, 0.0, 0.0]),
        _pose([20_000.0, 1_000.0, 50.0]),
        excess_range=excess_range,
        fixed_coordinates=fixed,
    )
    positions = list(layout.indices(tuple(f"{a}_m" for a in axes)))
    for _ in range(50):
        # A loop over random states: each one is its own check.
        state = rng.uniform(-100.0, 100.0, layout.dimension)
        state[positions] = rng.uniform(-20_000.0, 20_000.0, len(positions))
        # rtol 1e-6, atol 1e-10: the best a central difference reaches on a norm, as in
        # tests/core/test_tracking.py's Jacobian test.
        np.testing.assert_allclose(
            model.jacobian(state), _central_difference(model, state), rtol=1e-6, atol=1e-10
        )


@pytest.mark.parametrize("site", ["transmitter", "receiver"])
def test_the_bistatic_jacobian_is_rejected_on_a_site(site: str) -> None:
    motion = CartesianMotion(dict.fromkeys("xyz", "CV"), origin_lla_deg_m=ORIGIN)
    position = [100.0, 0.0, 0.0] if site == "transmitter" else [0.0, 100.0, 0.0]
    model = BistaticRangeDopplerModel(
        motion.state_layout, _pose([100.0, 0.0, 0.0]), _pose([0.0, 100.0, 0.0])
    )
    state = np.zeros(6)
    state[list(motion.state_layout.indices(("x_m", "y_m", "z_m")))] = position
    with pytest.raises(ValueError, match="singular"):
        model.jacobian(state)


def test_bistatic_predict_rejects_a_state_of_the_wrong_length() -> None:
    _, estimate, pose = _scene("CV")
    model = BistaticRangeDopplerModel(estimate.state_layout, pose, pose)
    with pytest.raises(ValueError, match="must be finite with shape"):
        model.predict(np.zeros((2, 3)))


def test_missing_coordinates_must_be_fixed_explicitly() -> None:
    motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    pose = _pose([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="missing coordinates"):
        BistaticRangeDopplerModel(motion.state_layout, pose, pose)


def test_fixed_coordinates_fill_the_missing_axes() -> None:
    motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    pose = _pose([0.0, 0.0, 0.0])
    model = BistaticRangeDopplerModel(
        motion.state_layout,
        pose,
        pose,
        fixed_coordinates={"y_m": 4.0, "z_m": 0.0, "ydot_mps": 0.0, "zdot_mps": 0.0},
    )
    # A 3-4-5 triangle out and back: 10 m of path, closing at 2 * (3/5 * 5) m/s.
    np.testing.assert_allclose(model.predict(np.array([3.0, -5.0])), [10, 6], rtol=1e-12)


@pytest.mark.parametrize(
    ("fixed", "match"),
    [
        ({"x_m": 0.0, "y_m": 0.0, "z_m": 0.0, "ydot_mps": 0.0, "zdot_mps": 0.0}, "absent"),
        ({"y_m": 0.0, "z_m": 0.0, "ydot_mps": 0.0, "zdot_mps": 0.0, "q_m": 0.0}, "absent"),
        ({"y_m": np.nan, "z_m": 0.0, "ydot_mps": 0.0, "zdot_mps": 0.0}, "finite"),
    ],
    ids=["already-in-state", "not-required", "nan"],
)
def test_fixed_coordinates_must_be_needed_absent_and_finite(
    fixed: dict[str, float], match: str
) -> None:
    motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    pose = _pose([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=match):
        BistaticRangeDopplerModel(motion.state_layout, pose, pose, fixed_coordinates=fixed)


def test_a_site_in_another_frame_is_rejected() -> None:
    motion = CartesianMotion(dict.fromkeys("xyz", "CV"), origin_lla_deg_m=ORIGIN)
    pose = _pose([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="frame"):
        BistaticRangeDopplerModel(replace(motion.state_layout, origin_lla_deg_m=None), pose, pose)


def test_the_geometry_velocity_flag_is_keyword_only() -> None:
    motion = CartesianMotion(dict.fromkeys("xyz", "CV"), origin_lla_deg_m=ORIGIN)
    with pytest.raises(TypeError):
        _Geometry(motion.state_layout, _pose([0.0, 0.0, 0.0]), None, True)  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# SensorPose
# --------------------------------------------------------------------------- #


def test_a_radar_at_the_origin_has_a_zero_pose() -> None:
    radar = Radar(_transmitter(), Receiver(1.0e6, 30.0, 3.0), *ORIGIN)
    pose = SensorPose.from_radar(radar, ORIGIN)
    # The site is the origin, so geodesy returns zero up to round-off in the
    # ECEF difference of two equal positions.
    np.testing.assert_allclose(pose.position_m, np.zeros(3), rtol=0, atol=1e-9)


def test_a_radar_pose_keeps_the_tracking_origin() -> None:
    radar = Radar(_transmitter(), Receiver(1.0e6, 30.0, 3.0), *ORIGIN)
    assert SensorPose.from_radar(radar, ORIGIN).origin_lla_deg_m == ORIGIN


def test_a_bistatic_pair_places_the_receiver_at_its_baseline() -> None:
    pair = BistaticRadar(
        _transmitter(), Receiver(1.0e6, 30.0, 3.0), *ORIGIN, 36.0025, -78.741, 60.0
    )
    _, receiver = SensorPose.from_bistatic_radar(pair, ORIGIN)
    # With the transmitter at the origin, the receiver is the pair's own
    # receiver_enu_m: the same geodesy call, so float64 round-off only.
    np.testing.assert_allclose(receiver.position_m, pair.receiver_enu_m, rtol=1e-12)


def test_a_bistatic_pair_places_the_transmitter_at_the_origin() -> None:
    pair = BistaticRadar(
        _transmitter(), Receiver(1.0e6, 30.0, 3.0), *ORIGIN, 36.0025, -78.741, 60.0
    )
    transmitter, _ = SensorPose.from_bistatic_radar(pair, ORIGIN)
    # Zero up to round-off in the ECEF difference of two equal positions.
    np.testing.assert_allclose(transmitter.position_m, np.zeros(3), rtol=0, atol=1e-9)


def test_a_pose_rejects_a_position_that_is_not_three_numbers() -> None:
    with pytest.raises(ValueError, match="position_m"):
        SensorPose(np.zeros(2), "ENU", ORIGIN)


def test_a_pose_rejects_an_impossible_origin() -> None:
    with pytest.raises(ValueError, match="ENU origin"):
        SensorPose(np.zeros(3), "ENU", (95.0, 0.0, 0.0))


# --------------------------------------------------------------------------- #
# Measurement, MeasurementBatch and SensorRegistration
# --------------------------------------------------------------------------- #


def test_a_batch_copies_the_caller_arrays() -> None:
    value, covariance = np.array([1.0, 2.0]), np.eye(2)
    batch = SensorRegistration("sensor", ("position",)).batch(
        1.0, [("position", value, covariance)]
    )
    value[:] = 0
    covariance[:] = 0
    np.testing.assert_array_equal(batch.measurements[0].value, [1.0, 2.0])
    np.testing.assert_array_equal(batch.measurements[0].covariance, np.eye(2))


def test_a_measurement_is_read_only() -> None:
    measurement = Measurement(np.ones(2), np.eye(2), 1.0, "sensor", "position")
    assert not measurement.value.flags.writeable
    assert not measurement.covariance.flags.writeable


def test_a_batch_stamps_each_measurement_with_its_sensor_and_time() -> None:
    sensor = SensorRegistration("sensor", ("position",))
    batch = sensor.batch(1.0, [("position", np.ones(1), np.eye(1))])
    assert (batch.measurements[0].sensor_id, batch.measurements[0].timestamp_s) == ("sensor", 1.0)


def test_an_empty_scan_is_a_valid_batch() -> None:
    assert SensorRegistration("sensor", ("position",)).batch(2.0, []).measurements == ()


def test_a_batch_rejects_a_measurement_with_another_timestamp() -> None:
    observation = Measurement(np.ones(1), np.eye(1), 2.0, "sensor", "position")
    with pytest.raises(ValueError, match="batch sensor and timestamp"):
        MeasurementBatch(1.0, "sensor", (observation,))


def test_a_batch_rejects_a_measurement_from_another_sensor() -> None:
    observation = Measurement(np.ones(1), np.eye(1), 1.0, "other", "position")
    with pytest.raises(ValueError, match="batch sensor and timestamp"):
        MeasurementBatch(1.0, "sensor", (observation,))


def test_a_batch_needs_a_sensor() -> None:
    with pytest.raises(ValueError, match="sensor_id must be nonempty"):
        MeasurementBatch(1.0, "")


@pytest.mark.parametrize(
    ("value", "covariance"),
    [
        (np.array([]), np.empty((0, 0))),
        (np.ones((1, 1)), np.eye(1)),
        (np.ones(2), np.eye(1)),
        (np.array([np.nan]), np.eye(1)),
        (np.ones(1), np.array([[-1.0]])),
        (np.ones(1), np.array([[np.inf]])),
    ],
    ids=["empty", "matrix-value", "size-mismatch", "nan", "negative-variance", "inf"],
)
def test_a_measurement_rejects_invalid_shapes_or_covariance(
    value: NDArray[np.float64], covariance: NDArray[np.float64]
) -> None:
    with pytest.raises(ValueError):
        Measurement(value, covariance, 1.0, "sensor", "position")


@pytest.mark.parametrize(("sensor_id", "model_id"), [("", "position"), ("sensor", "")])
def test_a_measurement_needs_a_sensor_and_a_model(sensor_id: str, model_id: str) -> None:
    with pytest.raises(ValueError, match="are required"):
        Measurement(np.ones(1), np.eye(1), 1.0, sensor_id, model_id)


def test_a_scan_rejects_a_nonfinite_time() -> None:
    with pytest.raises(ValueError, match="timestamp_s must be finite"):
        SensorRegistration("sensor", ("position",)).batch(np.nan, [])


def test_a_scan_rejects_an_unregistered_model() -> None:
    with pytest.raises(ValueError, match="not registered"):
        SensorRegistration("sensor", ("position",)).batch(1.0, [("unknown", np.ones(1), np.eye(1))])


@pytest.mark.parametrize(
    ("sensor_id", "model_ids"),
    [("", ("position",)), ("sensor", ()), ("sensor", ("",)), ("sensor", ("a", "a"))],
    ids=["no-sensor-id", "no-models", "empty-model-id", "repeated-model-id"],
)
def test_a_sensor_needs_an_id_and_unique_model_names(
    sensor_id: str, model_ids: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="unique nonempty model IDs"):
        SensorRegistration(sensor_id, model_ids)


@pytest.mark.parametrize(
    ("frame", "origin", "match"),
    [("", ORIGIN, "frame"), ("ENU", (91.0, 0.0, 0.0), "ENU origin")],
    ids=["empty-frame", "impossible-origin"],
)
def test_a_sensor_pose_checks_its_frame_and_origin_as_a_layout_does(
    frame: str, origin: tuple[float, float, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        SensorPose(np.zeros(3), frame, origin)


# --------------------------------------------------------------------------- #
# RangeAzimuthElevationRangeRate
# --------------------------------------------------------------------------- #


def _spherical_state(
    layout: StateLayout, position_m: list[float], velocity_mps: list[float]
) -> NDArray[np.float64]:
    """A state with the given position and velocity, placed by name."""
    state = np.zeros(layout.dimension)
    state[list(layout.indices(("x_m", "y_m", "z_m")))] = position_m
    state[list(layout.indices(("xdot_mps", "ydot_mps", "zdot_mps")))] = velocity_mps
    return state


def test_spherical_azimuth_is_clockwise_from_north_and_elevation_is_up() -> None:
    """Catches the mathematical azimuth convention, a sign flip, or elevation from the zenith.

    Due north, east, south and west of a site 1 km up give azimuths 0, pi/2, pi and
    3 pi/2 and elevation 0. The 3-4-12 point gives range 13 and elevation asin(12/13).
    """
    _, estimate, _ = _scene("CV")
    layout = estimate.state_layout
    site = [100.0, -50.0, 1000.0]
    model = RangeAzimuthElevationRangeRate(layout, _pose(site))
    offsets = [[0, 500, 0], [500, 0, 0], [0, -500, 0], [-500, 0, 0], [3, 4, 12]]
    states = np.stack(
        [_spherical_state(layout, list(np.add(site, o)), [0.0, 0.0, 0.0]) for o in offsets]
    )
    z = model.predict(states)
    # atol 1e-12 rad: one atan2 or asin of exact inputs.
    np.testing.assert_allclose(z[:4, 1], [0.0, np.pi / 2, np.pi, 3 * np.pi / 2], atol=1e-12)
    np.testing.assert_allclose(z[:4, 2], 0.0, atol=1e-12)
    # rtol 1e-12: a norm and one inverse trig function.
    np.testing.assert_allclose(z[4, 0], 13.0, rtol=1e-12)
    np.testing.assert_allclose(z[4, 2], np.arcsin(12 / 13), rtol=1e-12)


@pytest.mark.parametrize(("velocity_sign", "rate_sign"), [(-1.0, 1.0), (1.0, -1.0)])
def test_spherical_range_rate_is_positive_when_closing(
    velocity_sign: float, rate_sign: float
) -> None:
    """Catches the receding-positive sign, which passes every test using one direction only.

    The target at (3, 4, 12) moves along its line of sight at 13 m/s, so the range rate
    is 13 m/s in size: positive inbound, negative outbound.
    """
    _, estimate, pose = _scene("CV")
    layout = estimate.state_layout
    model = RangeAzimuthElevationRangeRate(layout, pose)
    velocity = list(velocity_sign * np.array([3.0, 4.0, 12.0]))
    state = _spherical_state(layout, [3.0, 4.0, 12.0], velocity)
    np.testing.assert_allclose(model.predict(state)[3], rate_sign * 13.0, rtol=1e-12)


def test_spherical_model_agrees_with_geodesy_about_an_offset_site() -> None:
    """Catches the site offset left out: geodesy is given p - p_s by hand here."""
    _, estimate, _ = _scene("CV")
    layout = estimate.state_layout
    site = np.array([-700.0, 250.0, 30.0])
    position = np.array([4000.0, -2500.0, 900.0])
    model = RangeAzimuthElevationRangeRate(
        layout, _pose(list(site)), ("range_m", "azimuth_rad", "elevation_rad")
    )
    range_m, azimuth_deg, elevation_deg = enu_to_range_azimuth_elevation(position - site)
    z = model.predict(_spherical_state(layout, list(position), [0.0, 0.0, 0.0]))
    expected = np.array([range_m, np.radians(azimuth_deg), np.radians(elevation_deg)])
    np.testing.assert_allclose(z, expected, rtol=1e-12)


def test_spherical_components_are_chosen_by_name_in_order() -> None:
    """Catches rows picked by position: the reported order differs from the canonical one."""
    _, estimate, pose = _scene("CV")
    layout = estimate.state_layout
    full = RangeAzimuthElevationRangeRate(layout, pose)
    chosen = ("range_rate_mps", "azimuth_rad", "range_m")
    model = RangeAzimuthElevationRangeRate(layout, pose, chosen)
    assert model.measurement_layout.names == chosen
    np.testing.assert_array_equal(
        model.predict(estimate.mean), full.predict(estimate.mean)[[3, 1, 0]]
    )
    np.testing.assert_array_equal(
        model.jacobian(estimate.mean), full.jacobian(estimate.mean)[[3, 1, 0]]
    )


def test_only_spherical_azimuth_wraps() -> None:
    """Catches a missing period on azimuth (no wrap in the innovation) or one on elevation."""
    _, estimate, pose = _scene("CV")
    layout = RangeAzimuthElevationRangeRate(estimate.state_layout, pose).measurement_layout
    periods = {c.name: c.period for c in layout.coordinates}
    assert periods == {
        "range_m": None,
        "azimuth_rad": 2 * np.pi,
        "elevation_rad": None,
        "range_rate_mps": None,
    }


def test_a_spherical_model_without_range_rate_needs_no_velocity() -> None:
    """A radar without Doppler works on a state that holds no velocity."""
    layout = StateLayout(tuple(Coordinate(n, "m") for n in ("x_m", "y_m", "z_m")), "ENU", ORIGIN)
    model = RangeAzimuthElevationRangeRate(
        layout, _pose([0.0, 0.0, 0.0]), ("range_m", "azimuth_rad")
    )
    np.testing.assert_allclose(
        model.predict(np.array([3.0, 4.0, 12.0])), [13.0, np.arctan2(3.0, 4.0)], rtol=1e-12
    )


def test_spherical_predict_of_a_stack_equals_the_stacked_single_predictions(
    rng: np.random.Generator,
) -> None:
    _, estimate, _ = _scene("CA")
    model = RangeAzimuthElevationRangeRate(estimate.state_layout, _pose([20.0, -30.0, 5.0]))
    states = rng.normal(0.0, 1000.0, (9, estimate.state_layout.dimension))
    expected = np.array([model.predict(s) for s in states])
    batched = model.predict(states)
    assert batched.shape == (9, 4)
    # The same operations in a different order of reduction; a few ulps at most.
    np.testing.assert_allclose(batched, expected, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize(
    ("axes", "fixed", "components"),
    [
        ("xyz", None, ("range_m", "azimuth_rad", "elevation_rad", "range_rate_mps")),
        ("xy", {"z_m": 3000.0, "zdot_mps": 0.0}, ("range_m", "azimuth_rad", "range_rate_mps")),
        ("xyz", None, ("elevation_rad", "range_m")),
    ],
    ids=["3d", "2d-fixed-height", "no-rate"],
)
def test_the_spherical_jacobian_matches_a_central_difference(
    axes: str,
    fixed: dict[str, float] | None,
    components: tuple[str, ...],
    rng: np.random.Generator,
) -> None:
    """The analytic Jacobian of the jacobian Notes, checked at many states.

    East positions stay in [1, 20] km, so azimuth stays in (0, pi), away from its wrap
    at north, where a central difference straddling 0 and 2 pi would be meaningless.
    """
    motion = CartesianMotion(dict.fromkeys(axes, "CV"), origin_lla_deg_m=ORIGIN)
    layout = motion.state_layout
    model = RangeAzimuthElevationRangeRate(
        layout, _pose([0.0, 0.0, 0.0]), components, fixed_coordinates=fixed
    )
    positions = list(layout.indices(tuple(f"{a}_m" for a in axes)))
    for _ in range(50):
        # A loop over random states: each one is its own check.
        state = rng.uniform(-100.0, 100.0, layout.dimension)
        state[positions] = rng.uniform(-20_000.0, 20_000.0, len(positions))
        state[positions[0]] = rng.uniform(1_000.0, 20_000.0)
        # rtol 1e-6, atol 1e-10: the best a central difference reaches on a norm, as in
        # the Jacobian test of tests/core/test_tracking.py.
        np.testing.assert_allclose(
            model.jacobian(state), _central_difference(model, state), rtol=1e-6, atol=1e-10
        )


def test_the_spherical_model_is_differentiable() -> None:
    _, estimate, pose = _scene("CV")
    model = RangeAzimuthElevationRangeRate(estimate.state_layout, pose)
    assert isinstance(model, DifferentiableMeasurementModel)


@pytest.mark.parametrize("method", ["predict", "jacobian"])
def test_a_target_on_the_radar_is_a_singular_geometry(method: str) -> None:
    _, estimate, _ = _scene("CV")
    layout = estimate.state_layout
    model = RangeAzimuthElevationRangeRate(
        layout, _pose([10.0, 20.0, 30.0]), ("range_m", "range_rate_mps")
    )
    state = _spherical_state(layout, [10.0, 20.0, 30.0], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="singular at the site"):
        getattr(model, method)(state)


@pytest.mark.parametrize("method", ["predict", "jacobian"])
def test_the_zenith_is_singular_only_when_an_angle_is_reported(method: str) -> None:
    """Catches a guard that is missing, or one that also rejects the valid range-only case."""
    _, estimate, pose = _scene("CV")
    layout = estimate.state_layout
    overhead = _spherical_state(layout, [0.0, 0.0, 5000.0], [1.0, 0.0, -2.0])
    with pytest.raises(ValueError, match="zenith"):
        getattr(RangeAzimuthElevationRangeRate(layout, pose), method)(overhead)
    range_only = RangeAzimuthElevationRangeRate(layout, pose, ("range_m", "range_rate_mps"))
    assert np.all(np.isfinite(getattr(range_only, method)(overhead)))


@pytest.mark.parametrize(
    "components",
    [(), ("range_m", "range_m"), ("range_m", "bearing_rad")],
    ids=["empty", "repeated", "unknown"],
)
def test_spherical_components_must_be_known_and_unique(components: tuple[str, ...]) -> None:
    _, estimate, pose = _scene("CV")
    with pytest.raises(ValueError, match="components must be nonempty and unique"):
        RangeAzimuthElevationRangeRate(estimate.state_layout, pose, components)


# --------------------------------------------------------------------------- #
# Marginal models (spec/tracker-001.md §8.3)
# --------------------------------------------------------------------------- #


def test_a_marginal_predicts_the_kept_components_of_its_base() -> None:
    """The marginal over range rate and range, in that order, of the full spherical model."""
    _, estimate, pose = _scene("CV")
    full = RangeAzimuthElevationRangeRate(estimate.state_layout, pose)
    part = marginal(full, ("range_rate_mps", "range_m"))
    names = full.measurement_layout.names
    expected = full.predict(estimate.mean)[[names.index("range_rate_mps"), names.index("range_m")]]
    # Exact: the marginal slices the base model's own output.
    np.testing.assert_array_equal(part.predict(estimate.mean), expected)
    stacked = np.stack([estimate.mean, 2.0 * estimate.mean])
    assert part.predict(stacked).shape == (2, 2)
    assert part.measurement_layout.names == ("range_rate_mps", "range_m")
    assert part.state_layout == full.state_layout


def test_a_marginal_keeps_the_kept_rows_of_its_bases_jacobian() -> None:
    _, estimate, pose = _scene("CV")
    full = RangeAzimuthElevationRangeRate(estimate.state_layout, pose)
    part = marginal(full, ("azimuth_rad",))
    assert isinstance(part, DifferentiableMeasurementModel)
    row = full.measurement_layout.names.index("azimuth_rad")
    # Exact: the same row of the same matrix.
    np.testing.assert_array_equal(part.jacobian(estimate.mean), full.jacobian(estimate.mean)[[row]])


def test_a_marginal_of_a_model_without_a_jacobian_has_none() -> None:
    """So that the Taylor linearisation refuses it at its first check, as it does the base."""
    layout = StateLayout((Coordinate("x_m", "m"), Coordinate("xdot_mps", "m/s")))

    class NoJacobian:
        state_layout = layout
        measurement_layout = layout

        def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.asarray(state, dtype=np.float64)

    part = marginal(NoJacobian(), ("x_m",))
    assert isinstance(part, MarginalModel)
    assert not isinstance(part, DifferentiableMeasurementModel)


def test_a_marginal_keeps_a_components_period() -> None:
    """A folded range stays folded: the period goes with the kept coordinate."""
    layout = StateLayout((Coordinate("range_m", "m"), Coordinate("range_rate_mps", "m/s")))
    full = CartesianPosition(layout, ("range_m", "range_rate_mps"), {"range_m": 5996.0})
    (kept,) = marginal(full, ("range_m",)).measurement_layout.coordinates
    assert kept.period == 5996.0


def test_marginalise_keeps_the_values_and_their_covariance_block() -> None:
    layout = StateLayout((Coordinate("a_m", "m"), Coordinate("b_m", "m"), Coordinate("c_m", "m")))
    full = CartesianPosition(layout, ("a_m", "b_m", "c_m"))
    covariance = np.array([[4.0, 1.0, 0.5], [1.0, 9.0, 2.0], [0.5, 2.0, 16.0]])
    z = Measurement(np.array([1.0, 2.0, 3.0]), covariance, 7.0, "radar", "full")
    part = marginal(full, ("c_m", "a_m")).marginalise(z, "ca")
    np.testing.assert_array_equal(part.value, [3.0, 1.0])
    np.testing.assert_array_equal(part.covariance, [[16.0, 0.5], [0.5, 4.0]])
    assert (part.timestamp_s, part.sensor_id, part.measurement_model_id) == (7.0, "radar", "ca")


def test_marginalise_rejects_a_measurement_of_another_dimension() -> None:
    layout = StateLayout((Coordinate("a_m", "m"), Coordinate("b_m", "m")))
    part = marginal(CartesianPosition(layout, ("a_m", "b_m")), ("a_m",))
    z = Measurement(np.array([1.0]), np.eye(1), 0.0, "radar", "full")
    with pytest.raises(ValueError, match="2 components"):
        part.marginalise(z, "a")


@pytest.mark.parametrize(
    ("names", "match"),
    [((), "nonempty and unique"), (("a_m", "a_m"), "nonempty and unique"), (("z_m",), "z_m")],
    ids=["empty", "repeated", "unknown"],
)
def test_a_marginal_needs_known_unique_components(names: tuple[str, ...], match: str) -> None:
    layout = StateLayout((Coordinate("a_m", "m"), Coordinate("b_m", "m")))
    with pytest.raises(ValueError, match=match):
        marginal(CartesianPosition(layout, ("a_m", "b_m")), names)
