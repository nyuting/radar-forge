"""Tests for radar_forge.core.tracking.ukf: the unscented Kalman filter (UKF).

Ground truth is analytic wherever there is one. With linear models the UKF must give exactly
the Kalman filter's answer; a noiseless target must give a zero innovation; and the sigma points
must reproduce the estimate's mean and covariance. The consistency tests (NEES and NIS) are the
only ones that catch a filter that is confidently wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import chi2

from radar_forge.core.tracking.coordinates import Coordinate, StateEstimate, StateLayout
from radar_forge.core.tracking.measurement_models import (
    BistaticRangeDopplerModel,
    CartesianPosition,
    Measurement,
    SensorPose,
)
from radar_forge.core.tracking.motion import CartesianMotion, RadialMotion
from radar_forge.core.tracking.ukf import UKF

ORIGIN = (36.00250, -78.94100, 60.0)
SEED = 20261005

# Scenario 003's numbers (as in tests/core/test_tracking.py), so the filter runs at the six
# orders of magnitude between the range and range-rate variances that the tracker sees.
SIGMA_RANGE_M = 21.635652855125496
SIGMA_VELOCITY_MPS = 0.017247813329811023
V_MAX_MPS = 200.0
# The old tracker's sigma_accel = 2 m/s² at T = 1 s. Its velocity variance per step matches
# the white-noise model's when tau = T / 2 (motion module Notes), so q = 2 * 4 * 0.5 = 4 m²/s³.
SIGMA_ACCELERATION_MPS2 = 2.0
CORRELATION_TIME_S = 0.5


@pytest.fixture
def rng() -> np.random.Generator:
    """One seeded generator, so a failure is reproducible."""
    return np.random.default_rng(SEED)


def _measurement(value: Any, covariance: Any, time_s: float) -> Measurement:
    return Measurement(np.atleast_1d(value), np.atleast_2d(covariance), time_s, "s", "p")


def _initial_covariance() -> NDArray[np.float64]:
    """P0: range from R, the unmeasured rate at the largest speed expected."""
    return np.diag([SIGMA_RANGE_M**2, V_MAX_MPS**2])


# --------------------------------------------------------------------------- #
# Equivalence with the linear Kalman filter
# --------------------------------------------------------------------------- #


def _linear_case() -> tuple[UKF, NDArray[np.float64], NDArray[np.float64]]:
    """Run one predict and update, and return the filter with the Kalman filter's answer."""
    motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    prior = StateEstimate(
        np.array([1.0, 2.0]), np.array([[4.0, 1.0], [1.0, 2.0]]), 0, motion.state_layout
    )
    ukf = UKF(prior, motion)
    model = CartesianPosition(motion.state_layout, ("x_m",))
    measurement = _measurement(4.0, 0.5, 1)
    f, q = motion.matrices(1)
    expected_mean, expected_cov = f @ prior.mean, f @ prior.covariance @ f.T + q
    h = np.array([[1.0, 0.0]])
    s = h @ expected_cov @ h.T + measurement.covariance
    gain = expected_cov @ h.T @ np.linalg.inv(s)
    expected_mean = expected_mean + gain @ (measurement.value - h @ expected_mean)
    expected_cov = expected_cov - gain @ s @ gain.T
    ukf.predict_to(1)
    ukf.update(measurement, model)
    return ukf, expected_mean, expected_cov


def test_with_linear_models_the_mean_is_the_kalman_filters() -> None:
    ukf, expected_mean, _ = _linear_case()
    # rtol and atol 1e-10: sigma-point reconstruction and linear solves accumulate roundoff.
    np.testing.assert_allclose(ukf.state.mean, expected_mean, rtol=1e-10, atol=1e-10)


def test_with_linear_models_the_covariance_is_the_kalman_filters() -> None:
    ukf, _, expected_cov = _linear_case()
    # rtol and atol 1e-10: sigma-point reconstruction and linear solves accumulate roundoff.
    np.testing.assert_allclose(ukf.state.covariance, expected_cov, rtol=1e-10, atol=1e-10)


def test_a_custom_periodic_model_needs_no_filter_changes() -> None:
    class Drift:
        state_layout = StateLayout((Coordinate("phase_rad", "rad", 2 * np.pi),))

        def transition(self, state: NDArray[np.float64], dt_s: float) -> NDArray[np.float64]:
            return state + dt_s

        def process_noise(self, state: NDArray[np.float64], dt_s: float) -> NDArray[np.float64]:
            return np.array([[0.01 * dt_s]])

    motion = Drift()
    ukf = UKF(StateEstimate(np.array([3.0]), np.array([[0.01]]), 0, motion.state_layout), motion)
    ukf.predict_to(0.5)
    model = CartesianPosition(motion.state_layout, ("phase_rad",))
    ukf.update(_measurement(3.5 - 2 * np.pi, 0.01, 0.5), model)
    # rtol and atol 1e-10: a wrapped residual of zero, through a few float64 operations.
    np.testing.assert_allclose(ukf.state.mean, [3.5 - 2 * np.pi], rtol=1e-10, atol=1e-10)


def test_prediction_backwards_in_time_is_rejected() -> None:
    motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    ukf = UKF(StateEstimate(np.zeros(2), np.eye(2), 1, motion.state_layout), motion)
    with pytest.raises(ValueError, match="out-of-sequence"):
        ukf.predict_to(0)


@pytest.mark.parametrize(
    "kwargs", [{"alpha": 0.0}, {"beta": -1.0}, {"kappa": -2.0}, {"alpha": np.nan}]
)
def test_impossible_sigma_point_parameters_are_rejected(kwargs: dict[str, float]) -> None:
    motion = RadialMotion()
    prior = StateEstimate(np.zeros(2), np.eye(2), 0, motion.state_layout)
    with pytest.raises(ValueError, match="alpha"):
        UKF(prior, motion, **kwargs)


# --------------------------------------------------------------------------- #
# Sigma points and weights
# --------------------------------------------------------------------------- #


def _six_state_filter() -> UKF:
    motion = CartesianMotion({"x": "CV", "y": "CV", "z": "CV"}, origin_lla_deg_m=ORIGIN)
    root = np.arange(1.0, 37.0).reshape(6, 6) / 10
    covariance = root @ root.T + np.eye(6)
    prior = StateEstimate(np.arange(6.0), covariance, 0, motion.state_layout)
    return UKF(prior, motion)


def test_the_weighted_sigma_points_reproduce_the_mean() -> None:
    ukf = _six_state_filter()
    # rtol 1e-12, atol 1e-12: one weighted sum of 13 points, with weights that cancel.
    np.testing.assert_allclose(ukf.wm @ ukf._sigma_points(), ukf.state.mean, rtol=1e-12, atol=1e-12)


def test_the_weighted_sigma_points_reproduce_the_covariance() -> None:
    ukf = _six_state_filter()
    deviations = ukf._sigma_points() - ukf.state.mean
    # rtol 1e-12: a Cholesky factor and one weighted outer-product sum.
    np.testing.assert_allclose(
        (deviations.T * ukf.wc) @ deviations, ukf.state.covariance, rtol=1e-12
    )


def test_the_mean_weights_sum_to_one() -> None:
    # rel 1e-12: 13 float64 additions.
    assert _six_state_filter().wm.sum() == pytest.approx(1.0, rel=1e-12)


def _still_filter(n_state: int, **kwargs: float) -> UKF:
    """A UKF on an n-state layout with a motion model that never moves."""
    layout = StateLayout(tuple(Coordinate(f"s{i}_m", "m") for i in range(n_state)))

    class Still:
        state_layout = layout

        def transition(self, state: NDArray[np.float64], dt_s: float) -> NDArray[np.float64]:
            return state

        def process_noise(self, state: NDArray[np.float64], dt_s: float) -> NDArray[np.float64]:
            return np.zeros((n_state, n_state))

    return UKF(StateEstimate(np.zeros(n_state), np.eye(n_state), 0, layout), Still(), **kwargs)


@pytest.mark.parametrize("kappa", [0.0, 1.0])
@pytest.mark.parametrize("n_state", range(1, 10))
def test_with_alpha_one_no_covariance_weight_is_negative(n_state: int, kappa: float) -> None:
    """UKF Notes: alpha = 1 with kappa >= 0 gives non-negative weights for every n up to 3-D CA."""
    assert _still_filter(n_state, alpha=1.0, kappa=kappa).wc.min() >= 0.0


@pytest.mark.parametrize("n_state", [2, 6, 9])
def test_the_default_weights_are_the_unscaled_transform(n_state: int) -> None:
    """alpha = 1, beta = 2, kappa = 0: W0m = 0, W0c = beta and Wi = 1/(2n) (UKF Notes)."""
    ukf = _still_filter(n_state)
    expected = np.full(2 * n_state + 1, 1 / (2 * n_state))
    expected[0] = 0.0
    # rtol 1e-12: each weight is one or two float64 operations on exact inputs.
    np.testing.assert_allclose(ukf.wm, expected, rtol=1e-12, atol=1e-15)
    expected[0] = 2.0
    np.testing.assert_allclose(ukf.wc, expected, rtol=1e-12, atol=1e-15)


# --------------------------------------------------------------------------- #
# Range and range rate, at scenario 003's conditioning
# --------------------------------------------------------------------------- #


def test_a_noiseless_target_has_zero_innovation_at_every_step() -> None:
    """A perfect measurement of a perfectly predicted target; any sign error shows here."""
    motion = RadialMotion(0.0)
    model = CartesianPosition(motion.state_layout, ("range_m", "range_rate_mps"))
    noise = np.diag([SIGMA_RANGE_M**2, SIGMA_VELOCITY_MPS**2])
    range_m, closing_mps = 10_000.0, 80.0
    prior = StateEstimate(
        np.array([range_m, closing_mps]), _initial_covariance(), 0, motion.state_layout
    )
    ukf = UKF(prior, motion)
    residuals = []
    for step in range(1, 21):
        # A loop over time: each step starts from the one before.
        ukf.predict_to(step)
        measurement = _measurement([range_m - closing_mps * step, closing_mps], noise, step)
        residuals.append(ukf.innovation_statistics(measurement, model).residual)
        ukf.update(measurement, model)
    # atol, not rtol: the quantity being checked is zero, and 1e-6 m is roundoff at 1e4 m.
    np.testing.assert_allclose(residuals, 0.0, atol=1e-6)


def _long_run(rng: np.random.Generator) -> list[NDArray[np.float64]]:
    """Run 200 noisy updates at scenario 003's R, and return every posterior covariance."""
    motion = RadialMotion(SIGMA_ACCELERATION_MPS2, CORRELATION_TIME_S)
    model = CartesianPosition(motion.state_layout, ("range_m", "range_rate_mps"))
    noise = np.diag([SIGMA_RANGE_M**2, SIGMA_VELOCITY_MPS**2])
    ukf = UKF(
        StateEstimate(np.array([10_000.0, 80.0]), _initial_covariance(), 0, motion.state_layout),
        motion,
    )
    covariances = []
    for step in range(1, 201):
        # A loop over time: each step starts from the one before.
        ukf.predict_to(step)
        value = [10_000.0 + rng.normal(0.0, SIGMA_RANGE_M), 80.0]
        ukf.update(_measurement(value, noise, step), model)
        covariances.append(np.array(ukf.state.covariance))
    return covariances


def test_the_covariance_stays_symmetric_over_200_steps(rng: np.random.Generator) -> None:
    """The short-form update with symmetrisation, at this R's conditioning (update Notes)."""
    for covariance in _long_run(rng):
        # rtol 1e-12: the update takes the symmetric part, so any asymmetry is roundoff.
        np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12)


def test_the_covariance_stays_positive_definite_over_200_steps(rng: np.random.Generator) -> None:
    """What the Joseph form is for in a linear filter; update's Notes say why it holds here."""
    smallest = [np.linalg.eigvalsh(covariance).min() for covariance in _long_run(rng)]
    assert min(smallest) > 0.0


# A bistatic pair 20 km apart, with a target 3 km up flying across the baseline. The path
# range and path rate are nonlinear in the state, so here the centre sigma point's weight
# matters (UKF Notes, "Why alpha = 1 and kappa = 0").
BISTATIC_TRANSMITTER_M = (0.0, 0.0, 0.0)
BISTATIC_RECEIVER_M = (20_000.0, 0.0, 0.0)
BISTATIC_ALTITUDE_M = 3_000.0
SIGMA_PATH_RANGE_M = 20.0
SIGMA_PATH_RATE_MPS = 1.0


def _bistatic_long_run(rng: np.random.Generator) -> list[NDArray[np.float64]]:
    """Run 200 noisy bistatic updates on a 2-D CV state, and return every posterior covariance.

    The UKF uses its defaults, alpha = 1 and kappa = 0, which make every covariance weight
    non-negative: what the update's positive-definiteness argument needs.
    """
    motion = CartesianMotion({"x": "CV", "y": "CV"}, origin_lla_deg_m=ORIGIN)
    layout = motion.state_layout
    transmitter = SensorPose(np.array(BISTATIC_TRANSMITTER_M), "ENU", ORIGIN)
    receiver = SensorPose(np.array(BISTATIC_RECEIVER_M), "ENU", ORIGIN)
    # The state has no z, so the target's height and climb rate are fixed.
    model = BistaticRangeDopplerModel(
        layout,
        transmitter,
        receiver,
        fixed_coordinates={"z_m": BISTATIC_ALTITUDE_M, "zdot_mps": 0.0},
    )
    noise = np.diag([SIGMA_PATH_RANGE_M**2, SIGMA_PATH_RATE_MPS**2])
    # Default order is x_m, xdot_mps, y_m, ydot_mps.
    truth = np.array([5_000.0, 50.0, 15_000.0, -100.0])
    prior_covariance = np.diag([200.0**2, 30.0**2, 200.0**2, 30.0**2])
    start = truth + rng.multivariate_normal(np.zeros(4), prior_covariance)
    ukf = UKF(StateEstimate(start, prior_covariance, 0, layout), motion)
    covariances = []
    for step in range(1, 201):
        # A loop over time: each step starts from the one before.
        truth = motion.transition(truth, 1.0)
        ukf.predict_to(step)
        value = model.predict(truth) + rng.multivariate_normal(np.zeros(2), noise)
        ukf.update(_measurement(value, noise, step), model)
        covariances.append(np.array(ukf.state.covariance))
    return covariances


def test_with_the_bistatic_model_the_covariance_stays_symmetric_over_200_steps(
    rng: np.random.Generator,
) -> None:
    """A nonlinear model, where the centre weight matters, with the default alpha and kappa."""
    for covariance in _bistatic_long_run(rng):
        # rtol 1e-12: the update takes the symmetric part, so any asymmetry is roundoff.
        np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12)


def test_with_the_bistatic_model_the_covariance_stays_positive_definite_over_200_steps(
    rng: np.random.Generator,
) -> None:
    """The non-negative-weight argument of the UKF Notes, checked on a nonlinear model."""
    smallest = [np.linalg.eigvalsh(covariance).min() for covariance in _bistatic_long_run(rng)]
    assert min(smallest) > 0.0


def _range_only_update() -> tuple[StateEstimate, StateEstimate]:
    motion = RadialMotion(SIGMA_ACCELERATION_MPS2, CORRELATION_TIME_S)
    model = CartesianPosition(motion.state_layout, ("range_m",))
    prior = StateEstimate(np.array([10_000.0, 0.0]), _initial_covariance(), 0, motion.state_layout)
    ukf = UKF(prior, motion)
    ukf.update(_measurement(10_100.0, SIGMA_RANGE_M**2, 0), model)
    return prior, ukf.state


def test_a_range_only_update_leaves_the_rate_uncorrelated_at_first() -> None:
    """With a diagonal prior, one range measurement cannot inform the rate."""
    _, posterior = _range_only_update()
    # abs 1e-9 m²/s: the cross covariance is zero, and roundoff on 468 m² is far smaller.
    assert posterior.covariance[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_a_range_only_update_leaves_the_rate_estimate_unchanged() -> None:
    _, posterior = _range_only_update()
    # abs 1e-9 m/s: the prior rate is zero, and roundoff on a 100 m correction is far smaller.
    assert posterior.mean[1] == pytest.approx(0.0, abs=1e-9)


def test_a_range_only_update_leaves_the_rate_variance_unchanged() -> None:
    prior, posterior = _range_only_update()
    # rel 1e-12: the rate variance is untouched up to a few float64 operations.
    assert posterior.covariance[1, 1] == pytest.approx(prior.covariance[1, 1], rel=1e-12)


def test_with_almost_no_measurement_noise_the_measured_element_equals_the_measurement() -> None:
    """As R → 0, the gain on a measured element → 1."""
    motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    model = CartesianPosition(motion.state_layout, ("x_m",))
    prior = StateEstimate(np.array([0.0, 5.0]), np.diag([100.0, 25.0]), 0, motion.state_layout)
    ukf = UKF(prior, motion)
    ukf.update(_measurement(37.0, 1e-12, 0), model)
    # atol 1e-9 m: the posterior misses z by R / (P + R) times the 37 m innovation, about
    # 4e-13 m, plus roundoff.
    np.testing.assert_allclose(ukf.state.mean[0], 37.0, rtol=0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Reusing the innovation (contract C2)
# --------------------------------------------------------------------------- #


class _CountingPosition(CartesianPosition):
    """CartesianPosition that counts how often it is evaluated."""

    calls = 0

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        type(self).calls += 1
        return super().predict(state)


def _ready_filter() -> tuple[UKF, _CountingPosition, Measurement]:
    motion = RadialMotion(SIGMA_ACCELERATION_MPS2, CORRELATION_TIME_S)
    model = _CountingPosition(motion.state_layout, ("range_m",))
    prior = StateEstimate(np.array([10_000.0, 50.0]), _initial_covariance(), 0, motion.state_layout)
    ukf = UKF(prior, motion)
    ukf.predict_to(1.0)
    return ukf, model, _measurement(9_960.0, SIGMA_RANGE_M**2, 1.0)


def test_an_update_given_its_innovation_does_not_evaluate_the_model_again() -> None:
    ukf, model, measurement = _ready_filter()
    stats = ukf.innovation_statistics(measurement, model)
    _CountingPosition.calls = 0
    ukf.update(measurement, model, innovation=stats)
    assert _CountingPosition.calls == 0


def test_an_update_given_its_innovation_gives_the_same_estimate() -> None:
    reused, model, measurement = _ready_filter()
    reused.update(measurement, model, innovation=reused.innovation_statistics(measurement, model))
    fresh, model, measurement = _ready_filter()
    fresh.update(measurement, model)
    # rtol 1e-15: the same arithmetic, done once instead of twice.
    np.testing.assert_allclose(reused.state.covariance, fresh.state.covariance, rtol=1e-15)


def test_an_innovation_from_before_a_prediction_is_not_reused() -> None:
    ukf, model, measurement = _ready_filter()
    stale = ukf.innovation_statistics(measurement, model)
    ukf.predict_to(2.0)
    later = _measurement(9_910.0, SIGMA_RANGE_M**2, 2.0)
    _CountingPosition.calls = 0
    ukf.update(later, model, innovation=stale)
    assert _CountingPosition.calls == 1


# --------------------------------------------------------------------------- #
# Consistency over independent runs
# --------------------------------------------------------------------------- #

N_RUNS, N_STEPS = 200, 30


@pytest.fixture(scope="module")
def consistency_runs() -> tuple[list[float], list[float]]:
    """Simulate N_RUNS independent tracks, and return the final NEES and every NIS.

    Truth is simulated with the same F, Q and R the filter assumes, so any failure is the
    filter's. Seeded from SEED, the module's one seed, because a module-scoped fixture cannot
    use the function-scoped ``rng``.
    """
    rng = np.random.default_rng(SEED)
    motion = CartesianMotion(
        {"x": "CV"},
        origin_lla_deg_m=ORIGIN,
        # q = 2 sigma² tau = 1 m²/s³.
        sigma_acceleration_mps2=1.0,
        acceleration_correlation_time_s=0.5,
    )
    model = CartesianPosition(motion.state_layout, ("x_m",))
    f, q = motion.matrices(1.0)
    noise = np.array([[4.0]])
    p0 = np.diag([100.0, 25.0])
    nees_last, nis_all = [], []
    for _ in range(N_RUNS):
        # A loop over independent runs, and inside it over time: each filter step needs the
        # one before, and each run is its own filter.
        truth = rng.multivariate_normal([1000.0, 10.0], p0)
        ukf = UKF(StateEstimate(np.array([1000.0, 10.0]), p0, 0, motion.state_layout), motion)
        for k in range(1, N_STEPS + 1):
            truth = f @ truth + rng.multivariate_normal(np.zeros(2), q)
            measurement = _measurement(truth[0] + rng.normal(0.0, 2.0), noise, k)
            ukf.predict_to(k)
            stats = ukf.innovation_statistics(measurement, model)
            nis_all.append(stats.nis)
            ukf.update(measurement, model, innovation=stats)
        error = truth - ukf.state.mean
        nees_last.append(float(error @ np.linalg.solve(ukf.state.covariance, error)))
    return nees_last, nis_all


@pytest.mark.slow
def test_nees_is_chi_squared_over_independent_runs(
    consistency_runs: tuple[list[float], list[float]],
) -> None:
    """A consistent filter's NEES is chi2(n_state). Bar-Shalom, Li and Kirubarajan, §5.4.

    The sum over N_RUNS independent chi2(2) draws is chi2(2 N_RUNS); the check is its 99.9%
    interval, not a tolerance.
    """
    nees_last, _ = consistency_runs
    low, high = chi2.ppf([0.0005, 0.9995], 2 * N_RUNS)
    assert low <= np.sum(nees_last) <= high


@pytest.mark.slow
def test_nis_is_chi_squared_over_independent_runs(
    consistency_runs: tuple[list[float], list[float]],
) -> None:
    """A consistent filter's NIS is chi2(n_meas), and independent from step to step."""
    _, nis_all = consistency_runs
    low, high = chi2.ppf([0.0005, 0.9995], len(nis_all))
    assert low <= np.sum(nis_all) <= high


def test_a_measurement_within_roundoff_of_the_filter_time_is_accepted() -> None:
    """Times computed two ways (k * dt against a running sum) differ by roundoff."""
    ukf, model, _ = _ready_filter()
    running_sum_s = sum([0.1] * 10)
    assert running_sum_s != 1.0
    ukf.update(_measurement(9_960.0, SIGMA_RANGE_M**2, running_sum_s), model)
    assert ukf.state.timestamp_s == 1.0


def test_a_measurement_at_another_time_is_rejected() -> None:
    ukf, model, _ = _ready_filter()
    with pytest.raises(ValueError, match="predict the estimator"):
        ukf.update(_measurement(9_960.0, SIGMA_RANGE_M**2, 1.001), model)
