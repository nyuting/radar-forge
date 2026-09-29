from dataclasses import replace

import numpy as np
import pytest

from radar_forge.tracking import (
    UKF,
    BistaticRangeDoppler,
    CartesianMotion,
    CartesianPosition,
    CompositeMeasurementModel,
    Measurement,
    MonostaticRadar,
    RangeBearingInitiator,
    SensorPose,
    StateEstimate,
    derive_kinematics,
)

ORIGIN = (36.00250, -78.94100, 60.0)


def fixture(kind="CA"):
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
        np.array([values[n] for n in motion.state_space.names], dtype=np.float64),
        np.eye(motion.state_space.dimension),
        0,
        motion.state_space,
    )
    pose = SensorPose(np.zeros(3), "ENU", ORIGIN)
    return motion, estimate, pose


def test_radar_geometry_and_derived_values_use_north_clockwise_and_closing_positive():
    _, estimate, pose = fixture()
    model = MonostaticRadar(estimate.state_space, pose, include_velocity=True)
    predicted = model.predict(estimate.mean)
    expected = [13, np.arctan2(3, 4), np.arctan2(12, 5), 25 / 13]
    np.testing.assert_allclose(predicted, expected, rtol=1e-12)
    derived = derive_kinematics(estimate, sensor=pose)
    for name, value in zip(
        ("range_m", "azimuth_rad", "elevation_rad", "radial_velocity_mps"), expected, strict=True
    ):
        np.testing.assert_allclose(derived[name].value, value, rtol=1e-12)
    for name, value in {
        "abs_speed_mps": 5,
        "acceleration_mps2": 13,
        "accelxy_mps2": 5,
        "accelz_mps2": -12,
        "xdot_norm": -0.6,
        "ydot_norm": -0.8,
    }.items():
        np.testing.assert_allclose(derived[name].value, value, rtol=1e-12)


def test_missing_coordinates_require_explicit_fixed_assumptions():
    motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    pose = SensorPose(np.zeros(3), "ENU", ORIGIN)
    with pytest.raises(ValueError, match="missing coordinates"):
        MonostaticRadar(motion.state_space, pose)
    model = MonostaticRadar(
        motion.state_space, pose, True, {"y_m": 4.0, "z_m": 0.0, "ydot_mps": 0.0, "zdot_mps": 0.0}
    )
    np.testing.assert_allclose(
        model.predict(np.array([3.0, -5.0])), [5, np.arctan2(3, 4), 0, 3], rtol=1e-12, atol=1e-12
    )
    with pytest.raises(ValueError, match="frame"):
        MonostaticRadar(replace(motion.state_space, origin_lla_deg_m=None), pose)


def test_cv_and_zero_speed_do_not_fabricate_derived_acceleration_or_direction():
    _, estimate, pose = fixture("CV")
    mean = estimate.mean.copy()
    mean[list(estimate.state_space.indices(("xdot_mps", "ydot_mps", "zdot_mps")))] = 0
    estimate = StateEstimate(mean, estimate.covariance, 0, estimate.state_space)
    values = derive_kinematics(estimate, sensor=pose)
    for name in ("acceleration_mps2", "accelxy_mps2", "accelz_mps2", "xdot_norm", "ydot_norm"):
        assert values[name].value is None
        assert values[name].reason
    origin = StateEstimate(np.zeros(len(mean)), estimate.covariance, 0, estimate.state_space)
    values = derive_kinematics(origin, sensor=pose)
    assert values["azimuth_rad"].value is None
    assert values["radial_velocity_mps"].value is None


def test_bistatic_and_composite_models_preserve_measurement_semantics():
    _, estimate, pose = fixture("CV")
    model = BistaticRangeDoppler(estimate.state_space, pose, pose)
    np.testing.assert_allclose(model.predict(estimate.mean), [26, 50 / 13], rtol=1e-12)
    selection = CartesianPosition(estimate.state_space, ("ydot_mps", "x_m"))
    combined = CompositeMeasurementModel([selection, model])
    np.testing.assert_allclose(combined.predict(estimate.mean), [-4, 3, 26, 50 / 13], rtol=1e-12)


def test_spherical_birth_inverts_north_clockwise_geometry():
    motion, estimate, pose = fixture("CV")
    model = MonostaticRadar(estimate.state_space, pose)
    prior = StateEstimate(np.zeros(6), np.eye(6) * 100, 0, estimate.state_space)
    initiator = RangeBearingInitiator(lambda s: UKF(s, motion), prior)
    measurement = Measurement(
        model.predict(estimate.mean), np.diag([0.01, 1e-6, 1e-6]), 1, "s", "p"
    )
    born = initiator.initiate(measurement, model)
    assert born is not None
    indices = list(estimate.state_space.indices(("x_m", "y_m", "z_m")))
    np.testing.assert_allclose(born.state.mean[indices], estimate.mean[indices], rtol=1e-12)
    assert np.linalg.eigvalsh(born.state.covariance).min() > 0
