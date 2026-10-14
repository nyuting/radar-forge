"""Tests for radar_forge.core.tracking.estimation.GaussianFilter: the KF, EKF and UKF.

Ground truth is analytic wherever there is one. With linear models the UKF must give exactly
the Kalman filter's answer, and a noiseless target must give a zero innovation. The consistency
tests (NEES and NIS) are the only ones that catch a filter that is confidently wrong; they and
the long-run definiteness tests run once per linearisation (the ``linearisation`` fixture).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import chi2

from radar_forge.core.tracking.coordinates import Coordinate, StateEstimate, StateLayout
from radar_forge.core.tracking.estimation import GaussianFilter
from radar_forge.core.tracking.linearisation import (
    Linearisation,
    TaylorLinearisation,
    UnscentedLinearisation,
)
from radar_forge.core.tracking.measurement_models import (
    BistaticRangeDopplerModel,
    CartesianPosition,
    Measurement,
    SensorPose,
)
from radar_forge.core.tracking.motion import CartesianMotion, RadialMotion

ORIGIN = (36.00250, -78.94100, 60.0)
SEED = 20261005

# Scenario 003's numbers (as in tests/core/test_tracking.py), so the filter runs at the six
# orders of magnitude between the range and range-rate variances that the tracker sees.
SIGMA_RANGE_M = 21.635652855125496
SIGMA_VELOCITY_MPS = 0.017247813329811023
V_MAX_MPS = 200.0
# The old tracker's sigma_accel = 2 m/s² at T = 1 s. Its velocity variance per step matches
# the white-noise model's when tau = T / 2 (motion module docstring, "Process noise"), so
# q = 2 * 4 * 0.5 = 4 m²/s³.
SIGMA_ACCELERATION_MPS2 = 2.0
CORRELATION_TIME_S = 0.5


@pytest.fixture
def rng() -> np.random.Generator:
    """One seeded generator, so a failure is reproducible."""
    return np.random.default_rng(SEED)


def _measurement(value: Any, covariance: Any, time_s: float) -> Measurement:
    return Measurement(np.atleast_1d(value), np.atleast_2d(covariance), time_s, "s", "p")


def _ukf(prior: StateEstimate, motion: Any, **settings: float) -> GaussianFilter:
    """A GaussianFilter with the unscented linearisation: the UKF."""
    return GaussianFilter(prior, motion, UnscentedLinearisation(**settings))


def _initial_covariance() -> NDArray[np.float64]:
    """P0: range from R, the unmeasured rate at the largest speed expected."""
    return np.diag([SIGMA_RANGE_M**2, V_MAX_MPS**2])


# --------------------------------------------------------------------------- #
# Which models each linearisation accepts
# --------------------------------------------------------------------------- #
# Equivalence with the linear Kalman filter is test_linearisation.py's drift guards: the
# unscented and Taylor linearisations against each other, through the real GaussianFilter.


class _Drift:
    """A periodic phase that drifts at one radian per second, and has no ``matrices``."""

    state_layout = StateLayout((Coordinate("phase_rad", "rad", 2 * np.pi),))

    def transition(self, state: NDArray[np.float64], dt_s: float) -> NDArray[np.float64]:
        return state + dt_s

    def process_noise(self, state: NDArray[np.float64], dt_s: float) -> NDArray[np.float64]:
        return np.array([[0.01 * dt_s]])


def test_the_taylor_linearisation_rejects_a_motion_model_without_matrices() -> None:
    motion = _Drift()
    prior = StateEstimate(np.array([3.0]), np.array([[0.01]]), 0, motion.state_layout)
    with pytest.raises(TypeError, match="_Drift has none"):
        GaussianFilter(prior, motion, TaylorLinearisation())


class _RangeOnly:
    """A measurement model with ``predict`` but no ``jacobian``."""

    def __init__(self, state_layout: StateLayout) -> None:
        self.state_layout = state_layout
        self.measurement_layout = StateLayout((Coordinate("range_m", "m"),))

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(state, dtype=np.float64)[..., :1]


@pytest.mark.parametrize("method", ["innovation_statistics", "update"])
def test_the_taylor_linearisation_rejects_a_model_without_a_jacobian_before_any_change(
    method: str,
) -> None:
    motion = RadialMotion()
    prior = StateEstimate(np.array([1000.0, 10.0]), np.eye(2), 0, motion.state_layout)
    estimator = GaussianFilter(prior, motion, TaylorLinearisation())
    with pytest.raises(TypeError, match="_RangeOnly has none"):
        getattr(estimator, method)(_measurement(1000.0, 1.0, 0), _RangeOnly(motion.state_layout))
    assert estimator.state is prior


def test_a_custom_periodic_model_needs_no_filter_changes() -> None:
    motion = _Drift()
    ukf = _ukf(StateEstimate(np.array([3.0]), np.array([[0.01]]), 0, motion.state_layout), motion)
    ukf.predict_to(0.5)
    model = CartesianPosition(motion.state_layout, ("phase_rad",))
    ukf.update(_measurement(3.5 - 2 * np.pi, 0.01, 0.5), model)
    # rtol and atol 1e-10: a wrapped residual of zero, through a few float64 operations.
    np.testing.assert_allclose(ukf.state.mean, [3.5 - 2 * np.pi], rtol=1e-10, atol=1e-10)


def test_prediction_backwards_in_time_is_rejected() -> None:
    motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    ukf = _ukf(StateEstimate(np.zeros(2), np.eye(2), 1, motion.state_layout), motion)
    with pytest.raises(ValueError, match="out-of-sequence"):
        ukf.predict_to(0)


def test_prediction_to_a_nonfinite_time_is_rejected() -> None:
    motion = RadialMotion()
    ukf = _ukf(StateEstimate(np.zeros(2), np.eye(2), 0, motion.state_layout), motion)
    with pytest.raises(ValueError, match="timestamp_s must be finite"):
        ukf.predict_to(np.inf)


# A layout that no motion or measurement model in these tests uses: the same names as
# RadialMotion's, in another frame.
_OTHER_LAYOUT = StateLayout((Coordinate("range_m", "m"), Coordinate("range_rate_mps", "m/s")))


def test_a_filter_rejects_a_state_whose_layout_is_not_the_motion_models() -> None:
    with pytest.raises(ValueError, match="motion model's"):
        _ukf(StateEstimate(np.zeros(2), np.eye(2), 0, _OTHER_LAYOUT), RadialMotion())


def test_set_state_rejects_a_state_whose_layout_is_not_the_motion_models() -> None:
    motion = RadialMotion()
    ukf = _ukf(StateEstimate(np.zeros(2), np.eye(2), 0, motion.state_layout), motion)
    with pytest.raises(ValueError, match="motion model's"):
        ukf.set_state(StateEstimate(np.zeros(2), np.eye(2), 0, _OTHER_LAYOUT))


@pytest.mark.parametrize("method", ["innovation_statistics", "update"])
def test_a_measurement_model_in_another_layout_is_rejected(method: str) -> None:
    motion = RadialMotion()
    ukf = _ukf(StateEstimate(np.zeros(2), np.eye(2), 0, motion.state_layout), motion)
    model = CartesianPosition(_OTHER_LAYOUT, ("range_m",))
    with pytest.raises(ValueError, match="measurement model's StateLayout"):
        getattr(ukf, method)(_measurement(0.0, 1.0, 0), model)


@pytest.mark.parametrize(
    "kwargs", [{"alpha": 0.0}, {"beta": -1.0}, {"kappa": -2.0}, {"alpha": np.nan}]
)
def test_impossible_sigma_point_parameters_are_rejected(kwargs: dict[str, float]) -> None:
    motion = RadialMotion()
    prior = StateEstimate(np.zeros(2), np.eye(2), 0, motion.state_layout)
    with pytest.raises(ValueError, match="alpha"):
        _ukf(prior, motion, **kwargs)


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
    ukf = _ukf(prior, motion)
    residuals = []
    for step in range(1, 21):
        # A loop over time: each step starts from the one before.
        ukf.predict_to(step)
        measurement = _measurement([range_m - closing_mps * step, closing_mps], noise, step)
        residuals.append(ukf.innovation_statistics(measurement, model).innovation)
        ukf.update(measurement, model)
    # atol, not rtol: the quantity being checked is zero, and 1e-6 m is roundoff at 1e4 m.
    np.testing.assert_allclose(residuals, 0.0, atol=1e-6)


def _long_run(rng: np.random.Generator, linearisation: Linearisation) -> list[NDArray[np.float64]]:
    """Run 200 noisy updates at scenario 003's R, and return every posterior covariance."""
    motion = RadialMotion(SIGMA_ACCELERATION_MPS2, CORRELATION_TIME_S)
    model = CartesianPosition(motion.state_layout, ("range_m", "range_rate_mps"))
    noise = np.diag([SIGMA_RANGE_M**2, SIGMA_VELOCITY_MPS**2])
    estimator = GaussianFilter(
        StateEstimate(np.array([10_000.0, 80.0]), _initial_covariance(), 0, motion.state_layout),
        motion,
        linearisation,
    )
    covariances = []
    for step in range(1, 201):
        # A loop over time: each step starts from the one before.
        estimator.predict_to(step)
        value = [10_000.0 + rng.normal(0.0, SIGMA_RANGE_M), 80.0]
        estimator.update(_measurement(value, noise, step), model)
        covariances.append(np.array(estimator.state.covariance))
    return covariances


def test_the_covariance_stays_symmetric_over_200_steps(
    rng: np.random.Generator, linearisation: Linearisation
) -> None:
    """The update takes the symmetric part, at this R's conditioning (``_correct`` Notes)."""
    for covariance in _long_run(rng, linearisation):
        # rtol 1e-12: the update takes the symmetric part, so any asymmetry is roundoff.
        np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12)


def test_the_covariance_stays_positive_definite_over_200_steps(
    rng: np.random.Generator, linearisation: Linearisation
) -> None:
    """Joseph form with H, short form without; ``_correct``'s Notes say why each holds."""
    smallest = [
        np.linalg.eigvalsh(covariance).min() for covariance in _long_run(rng, linearisation)
    ]
    assert min(smallest) > 0.0


# A bistatic pair 20 km apart, with a target 3 km up flying across the baseline. The path
# range and path rate are nonlinear in the state, so here the centre sigma point's weight
# matters (UKF Notes, "Why alpha = 1 and kappa = 0").
BISTATIC_TRANSMITTER_M = (0.0, 0.0, 0.0)
BISTATIC_RECEIVER_M = (20_000.0, 0.0, 0.0)
BISTATIC_ALTITUDE_M = 3_000.0
SIGMA_PATH_RANGE_M = 20.0
SIGMA_PATH_RATE_MPS = 1.0


def _bistatic_long_run(
    rng: np.random.Generator, linearisation: Linearisation
) -> list[NDArray[np.float64]]:
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
    estimator = GaussianFilter(
        StateEstimate(start, prior_covariance, 0, layout), motion, linearisation
    )
    covariances = []
    for step in range(1, 201):
        # A loop over time: each step starts from the one before.
        truth = motion.transition(truth, 1.0)
        estimator.predict_to(step)
        value = model.predict(truth) + rng.multivariate_normal(np.zeros(2), noise)
        estimator.update(_measurement(value, noise, step), model)
        covariances.append(np.array(estimator.state.covariance))
    return covariances


def test_with_the_bistatic_model_the_covariance_stays_symmetric_over_200_steps(
    rng: np.random.Generator, linearisation: Linearisation
) -> None:
    """A nonlinear model, where the centre weight matters, with the default alpha and kappa."""
    for covariance in _bistatic_long_run(rng, linearisation):
        # rtol 1e-12: the update takes the symmetric part, so any asymmetry is roundoff.
        np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12)


def test_with_the_bistatic_model_the_covariance_stays_positive_definite_over_200_steps(
    rng: np.random.Generator, linearisation: Linearisation
) -> None:
    """The definiteness argument of ``_correct``'s Notes, checked on a nonlinear model."""
    smallest = [
        np.linalg.eigvalsh(covariance).min()
        for covariance in _bistatic_long_run(rng, linearisation)
    ]
    assert min(smallest) > 0.0


def _range_only_update() -> tuple[StateEstimate, StateEstimate]:
    motion = RadialMotion(SIGMA_ACCELERATION_MPS2, CORRELATION_TIME_S)
    model = CartesianPosition(motion.state_layout, ("range_m",))
    prior = StateEstimate(np.array([10_000.0, 0.0]), _initial_covariance(), 0, motion.state_layout)
    ukf = _ukf(prior, motion)
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
    ukf = _ukf(prior, motion)
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


def _ready_filter() -> tuple[GaussianFilter, _CountingPosition, Measurement]:
    motion = RadialMotion(SIGMA_ACCELERATION_MPS2, CORRELATION_TIME_S)
    model = _CountingPosition(motion.state_layout, ("range_m",))
    prior = StateEstimate(np.array([10_000.0, 50.0]), _initial_covariance(), 0, motion.state_layout)
    ukf = _ukf(prior, motion)
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
def consistency_runs(linearisation: Linearisation) -> tuple[list[float], list[float]]:
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
        ukf = GaussianFilter(
            StateEstimate(np.array([1000.0, 10.0]), p0, 0, motion.state_layout),
            motion,
            linearisation,
        )
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
    # A loop, not sum(): from Python 3.12 sum() compensates for roundoff, and
    # sum([0.1] * 10) is exactly 1.0. A tracker accumulates time step by step.
    running_sum_s = 0.0
    for _ in range(10):
        running_sum_s += 0.1
    assert running_sum_s != 1.0
    ukf.update(_measurement(9_960.0, SIGMA_RANGE_M**2, running_sum_s), model)
    assert ukf.state.timestamp_s == 1.0


def test_a_measurement_at_another_time_is_rejected() -> None:
    ukf, model, _ = _ready_filter()
    with pytest.raises(ValueError, match="predict the estimator"):
        ukf.update(_measurement(9_960.0, SIGMA_RANGE_M**2, 1.001), model)
