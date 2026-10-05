"""Tests for radar_forge.core.tracking.measurement_models: measurements, routes, models."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from radar_forge.core.radar import BistaticRadar, Radar, Receiver, Transmitter
from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.measurement_models import (
    BistaticRangeDopplerModel,
    CartesianPosition,
    Measurement,
    MeasurementBatch,
    SensorPose,
    SensorRoute,
    _Geometry,
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
# Measurement, MeasurementBatch and SensorRoute
# --------------------------------------------------------------------------- #


def test_a_batch_copies_the_caller_arrays() -> None:
    value, covariance = np.array([1.0, 2.0]), np.eye(2)
    batch = SensorRoute("sensor", ("position",)).batch(1.0, [("position", value, covariance)])
    value[:] = 0
    covariance[:] = 0
    np.testing.assert_array_equal(batch.measurements[0].value, [1.0, 2.0])
    np.testing.assert_array_equal(batch.measurements[0].covariance, np.eye(2))


def test_a_measurement_is_read_only() -> None:
    measurement = Measurement(np.ones(2), np.eye(2), 1.0, "sensor", "position")
    assert not measurement.value.flags.writeable
    assert not measurement.covariance.flags.writeable


def test_a_batch_stamps_each_measurement_with_its_sensor_and_time() -> None:
    sensor = SensorRoute("sensor", ("position",))
    batch = sensor.batch(1.0, [("position", np.ones(1), np.eye(1))])
    assert (batch.measurements[0].sensor_id, batch.measurements[0].timestamp_s) == ("sensor", 1.0)


def test_an_empty_scan_is_a_valid_batch() -> None:
    assert SensorRoute("sensor", ("position",)).batch(2.0, []).measurements == ()


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
        SensorRoute("sensor", ("position",)).batch(np.nan, [])


def test_a_scan_rejects_an_unregistered_model() -> None:
    with pytest.raises(ValueError, match="not registered"):
        SensorRoute("sensor", ("position",)).batch(1.0, [("unknown", np.ones(1), np.eye(1))])


@pytest.mark.parametrize(
    ("route_id", "model_ids"),
    [("", ("position",)), ("sensor", ()), ("sensor", ("",)), ("sensor", ("a", "a"))],
    ids=["no-route-id", "no-models", "empty-model-id", "repeated-model-id"],
)
def test_a_route_needs_an_id_and_unique_model_names(
    route_id: str, model_ids: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="unique nonempty model routes"):
        SensorRoute(route_id, model_ids)
