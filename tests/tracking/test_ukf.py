import numpy as np
import pytest

from radar_forge.tracking import (
    IMM,
    UKF,
    CartesianMotion,
    CartesianPosition,
    Coordinate,
    Measurement,
    StateEstimate,
    StateSpace,
)


def test_ukf_matches_linear_gaussian_equations_including_process_noise():
    motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=(36.00250, -78.94100, 60.0))
    prior = StateEstimate(
        np.array([1.0, 2.0]), np.array([[4.0, 1.0], [1.0, 2.0]]), 0, motion.state_space
    )
    ukf = UKF(prior, motion)
    model = CartesianPosition(motion.state_space, ("x_m",))
    measurement = Measurement(np.array([4.0]), np.array([[0.5]]), 1, "s", "p")
    f, q = motion.matrices(1)
    expected_mean, expected_cov = f @ prior.mean, f @ prior.covariance @ f.T + q
    ukf.predict_to(1)
    h = np.array([[1.0, 0.0]])
    gain = expected_cov @ h.T @ np.linalg.inv(h @ expected_cov @ h.T + measurement.covariance)
    expected_mean += gain @ (measurement.value - h @ expected_mean)
    expected_cov -= gain @ (h @ expected_cov @ h.T + measurement.covariance) @ gain.T
    ukf.update(measurement, model)
    # Sigma-point reconstruction and linear solves accumulate float64 roundoff.
    np.testing.assert_allclose(ukf.state.mean, expected_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ukf.state.covariance, expected_cov, rtol=1e-10, atol=1e-10)


def test_custom_coordinate_model_needs_no_filter_changes():
    class Drift:
        state_space = StateSpace((Coordinate("phase_rad", "rad", 2 * np.pi),))

        def transition(self, state, dt_s):
            return state + dt_s

        def process_noise(self, state, dt_s):
            return np.array([[0.01 * dt_s]])

    motion = Drift()
    ukf = UKF(StateEstimate(np.array([3.0]), np.array([[0.01]]), 0, motion.state_space), motion)
    ukf.predict_to(0.5)
    model = CartesianPosition(motion.state_space, ("phase_rad",))
    ukf.update(Measurement(np.array([3.5 - 2 * np.pi]), np.array([[0.01]]), 0.5, "s", "p"), model)
    np.testing.assert_allclose(ukf.state.mean, [3.5 - 2 * np.pi], rtol=1e-10, atol=1e-10)


def test_imm_mixture_and_equal_time_updates():
    motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=(36.00250, -78.94100, 60.0))
    modes = [
        UKF(StateEstimate(np.array([x, 0.0]), np.eye(2), 0, motion.state_space), motion)
        for x in (0.0, 2.0)
    ]
    imm = IMM(modes, np.eye(2), np.array([0.5, 0.5]))
    np.testing.assert_allclose(imm.state.mean, [1, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(imm.state.covariance, [[2, 0], [0, 1]], rtol=1e-12, atol=1e-12)
    imm.predict_to(0)
    model = CartesianPosition(motion.state_space, ("x_m",))
    imm.update(Measurement(np.array([0.0]), np.array([[0.1]]), 0, "s", "p"), model)
    assert imm.probabilities[0] > imm.probabilities[1]
    np.testing.assert_allclose(imm.probabilities.sum(), 1, rtol=1e-12)
    different = CartesianMotion({"x": "CA"}, origin_lla_deg_m=(36.00250, -78.94100, 60.0))
    incompatible = UKF(StateEstimate(np.zeros(3), np.eye(3), 0, different.state_space), different)
    with pytest.raises(ValueError, match="common StateSpace"):
        IMM([modes[0], incompatible], np.eye(2))


def test_covariance_validation_and_out_of_sequence():
    motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=(36.00250, -78.94100, 60.0))
    with pytest.raises(ValueError, match="semidefinite"):
        StateEstimate(np.zeros(2), np.diag([1.0, -1.0]), 0, motion.state_space)
    ukf = UKF(StateEstimate(np.zeros(2), np.eye(2), 1, motion.state_space), motion)
    with pytest.raises(ValueError, match="out-of-sequence"):
        ukf.predict_to(0)
