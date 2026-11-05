"""Tests for radar_forge.tracking.filters: the bank and the one Gaussian filter over it.

Ground truth is analytic wherever there is one: the Kalman update written out by hand, the
UKF equal to the KF on linear models (``spec/tracker-001.md`` §3), the batched
innovation equal to a per-pair loop (§7), and the steady state of the Riccati recursion.
NEES and NIS are checked against chi-square intervals over Monte Carlo runs, which here are
the rows of one bank.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are

from radar_forge.tracking.filters import (
    Linearisation,
    TaylorLinearisation,
    TrackBank,
    TrackStatus,
    UnscentedTransform,
    innovate,
    predict,
    update,
)
from radar_forge.tracking.metrics import consistency_interval, nees
from radar_forge.tracking.models import (
    BistaticRangeRateModel,
    CartesianMotion,
    LinearMeasurement,
    RadialConstantVelocity,
    RangeAzimuthElevationRangeRate,
    StateLayout,
)

SEED = 20261104


def _bank(layout: StateLayout, mean: Any, covariance: Any, time_s: float = 0.0) -> TrackBank:
    """A bank whose rows are ``mean`` ``(n, n_state)`` with ``covariance`` each."""
    mean = np.atleast_2d(np.asarray(mean, dtype=np.float64))
    covariance = np.zeros((len(mean), *np.shape(covariance)[-2:])) + covariance
    ids, status = np.arange(len(mean)), np.zeros(len(mean))
    return TrackBank.empty(layout).append(mean, covariance, time_s, ids, status)


def _step(
    bank: TrackBank, model: Any, linearisation: Linearisation, z: Any, noise: Any
) -> TrackBank:
    """Update every row with its own measurement: row i takes z[i]."""
    innovation = innovate(bank, model, linearisation, z, noise)
    return update(bank, innovation, np.arange(bank.n_tracks))


def test_append_validates_the_new_rows_and_keeps_the_old_ones_first() -> None:
    layout = RadialConstantVelocity().state_layout
    bank = _bank(layout, [[1.0, 2.0]], np.eye(2)).append([[3.0, 4.0]], [np.eye(2)], 5.0, [9], [1])
    np.testing.assert_array_equal(bank.track_id, [0, 9])
    np.testing.assert_array_equal(bank.time_s, [0.0, 5.0])
    assert bank.status[1] == TrackStatus.CONFIRMED and bank.n_hits.tolist() == [1, 1]


@pytest.mark.parametrize(
    ("mean", "covariance", "ids"),
    [
        ([[1.0, 2.0]], [[[1.0, 0.5], [0.0, 1.0]]], [1]),
        ([[1.0, 2.0]], [[[1.0, 0.0], [0.0, -1.0]]], [1]),
        ([[1.0, np.nan]], [np.eye(2)], [1]),
        ([[1.0, 2.0, 3.0]], [np.eye(2)], [1]),
        ([[1.0, 2.0]], [np.eye(2)], [1, 2]),
    ],
    ids=["asymmetric", "indefinite", "nan", "wrong-width", "wrong-ids"],
)
def test_append_rejects_rows_that_are_not_a_valid_state(
    mean: Any, covariance: Any, ids: Any
) -> None:
    empty = TrackBank.empty(RadialConstantVelocity().state_layout)
    with pytest.raises(ValueError, match=r"symmetric|definite|finite|shape"):
        empty.append(mean, covariance, 0.0, ids, [0] * len(ids))


def test_append_rejects_a_zero_variance_and_names_its_row() -> None:
    """Semidefinite is not enough: the UKF factorises every row, so the boundary refuses it."""
    empty = TrackBank.empty(RadialConstantVelocity().state_layout)
    covariance = np.stack([np.eye(2), np.diag([1.0, 0.0]), np.eye(2)])
    with pytest.raises(ValueError, match=r"positive definite; the matrices at rows \[1\]"):
        empty.append(np.zeros((3, 2)), covariance, 0.0, [1, 2, 3], [0, 0, 0])


@pytest.mark.parametrize("status", [[4], [-1], [7]])
def test_append_rejects_a_status_that_is_not_a_track_status(status: list[int]) -> None:
    """A stray int8 would never confirm or delete, so it is refused at the boundary."""
    empty = TrackBank.empty(RadialConstantVelocity().state_layout)
    with pytest.raises(ValueError, match="TrackStatus"):
        empty.append([[1.0, 2.0]], [np.eye(2)], 0.0, [1], status)


def test_compact_keeps_the_chosen_rows_in_order() -> None:
    bank = _bank(
        RadialConstantVelocity().state_layout, [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], np.eye(2)
    )
    np.testing.assert_array_equal(bank.compact([True, False, True]).mean[:, 0], [1.0, 3.0])


def test_a_snapshot_is_a_read_only_copy() -> None:
    bank = _bank(RadialConstantVelocity().state_layout, [[1.0, 2.0]], np.eye(2))
    snapshot = bank.snapshots()[0]
    bank.mean[0, 0] = 99.0
    assert snapshot.mean[0] == 1.0 and not snapshot.mean.flags.writeable
    assert snapshot.nis is None and snapshot.measurement_index is None
    assert snapshot.status is TrackStatus.TENTATIVE


def test_predict_moves_each_row_by_its_own_step() -> None:
    layout = RadialConstantVelocity().state_layout
    bank = _bank(layout, [[0.0, 10.0]], np.eye(2)).append([[0.0, 10.0]], [np.eye(2)], 1.0, [1], [0])
    moved = predict(bank, RadialConstantVelocity(), TaylorLinearisation(), 3.0)
    np.testing.assert_array_equal(moved.mean[:, 0], [30.0, 20.0])
    np.testing.assert_array_equal(moved.time_s, [3.0, 3.0])


# The unscented weights and sigma points.


@pytest.mark.parametrize("n_state", [1, 2, 4, 6, 9])
@pytest.mark.parametrize("kappa", [0.0, 1.0])
def test_the_mean_weights_sum_to_one(n_state: int, kappa: float) -> None:
    mean_weights, _ = UnscentedTransform(kappa=kappa).weights(n_state)
    assert mean_weights.sum() == pytest.approx(1.0, rel=1e-14)


def test_the_julier_uhlmann_kappa_is_accepted_up_to_nine_states_and_no_further() -> None:
    """With alpha = 1 and beta = 2, W0c = 3 - n/3 under kappa = 3 - n: nonnegative to n = 9."""
    assert UnscentedTransform(kappa=3.0 - 9).weights(9)[1][0] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="negative"):
        UnscentedTransform(kappa=3.0 - 10).weights(10)


@pytest.mark.parametrize("settings", [{"alpha": 1e-3}, {"alpha": 0.5}, {"kappa": -6.0}])
def test_settings_with_a_negative_centre_weight_or_no_spread_are_refused(
    settings: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match=r"negative|positive"):
        UnscentedTransform(**settings).weights(6)


@pytest.mark.parametrize("settings", [{"alpha": 0.0}, {"beta": -1.0}, {"kappa": np.nan}])
def test_impossible_sigma_point_settings_are_refused_at_construction(
    settings: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="alpha"):
        UnscentedTransform(**settings)


@pytest.mark.parametrize("n_state", [2, 6])
def test_sigma_points_refuse_a_spread_that_puts_every_point_on_the_mean(n_state: int) -> None:
    """kappa = -n makes c = alpha² (n + kappa) zero: every sigma point would be the mean."""
    layout = CartesianMotion("xyz"[: n_state // 2]).state_layout
    bank = _bank(layout, np.zeros((1, n_state)), np.eye(n_state))
    with pytest.raises(ValueError, match="kappa must be positive"):
        UnscentedTransform(kappa=-float(n_state)).sigma_points(bank)


def test_the_weights_are_computed_once_per_size() -> None:
    assert UnscentedTransform().weights(5)[0] is UnscentedTransform().weights(5)[0]


def test_the_weighted_sigma_points_reproduce_each_rows_mean_and_covariance(
    rng: np.random.Generator,
) -> None:
    layout = CartesianMotion().state_layout
    roots = rng.normal(0.0, 1.0, (4, 6, 6))
    bank = _bank(
        layout, rng.normal(0.0, 100.0, (4, 6)), roots @ np.swapaxes(roots, 1, 2) + np.eye(6)
    )
    transform = UnscentedTransform(alpha=0.9, kappa=1.0)
    mean_weights, covariance_weights = transform.weights(6)
    points = transform.sigma_points(bank)
    deviation = points - bank.mean[:, None, :]
    # rtol 1e-12: a Cholesky factor and weighted sums of a few terms.
    np.testing.assert_allclose(mean_weights @ points, bank.mean, rtol=1e-12)
    np.testing.assert_allclose(
        np.einsum("k,tki,tkj->tij", covariance_weights, deviation, deviation),
        bank.covariance,
        rtol=1e-12,
        atol=1e-12,
    )


# Drift guards: on linear models the UKF is the KF (spec/tracker-001.md §3).

# rtol 1e-10, and an atol of 1e-10 times each entry's own scale from the prior: the entries
# mix m² and (m/s)², so one absolute tolerance cannot suit them all.
GUARD = 1e-10
SIGMA_SETTINGS = [(1.0, 2.0, 0.0), (1.0, 2.0, 3.0 - 6), (1.0, 0.0, 0.0)]


def _guard_bank() -> TrackBank:
    """Two correlated priors whose positions (km) and velocities (10 m/s) differ in scale."""
    scale = np.array([1000.0, 30.0] * 3)
    root = np.arange(1.0, 37.0).reshape(6, 6) / 10
    correlation = root @ root.T + np.eye(6)
    correlation /= np.sqrt(np.outer(np.diag(correlation), np.diag(correlation)))
    means = [[5000.0, -20.0, 3000.0, 15.0, 1000.0, 2.0], [0.0, 5.0, -800.0, 0.0, 9.0, 1.0]]
    return _bank(CartesianMotion().state_layout, means, correlation * np.outer(scale, scale))


def _assert_scaled_close(actual: Any, expected: Any, scale: Any) -> None:
    """|actual - expected| <= GUARD (scale + |expected|), entry by entry."""
    allowed = GUARD * (np.asarray(scale) + np.abs(expected)) + np.finfo(float).tiny
    np.testing.assert_array_less(np.abs(np.asarray(actual) - expected), allowed)


def _sigma(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))


def _outer(sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    return sigma[..., :, None] * sigma[..., None, :]


@pytest.mark.parametrize(("alpha", "beta", "kappa"), SIGMA_SETTINGS)
def test_the_unscented_moments_are_the_kalman_moments_on_linear_models(
    alpha: float, beta: float, kappa: float
) -> None:
    """Spec §3: x⁻, P⁻, z, P_xz and S agree, whatever the accepted weights."""
    motion, bank = CartesianMotion(), _guard_bank()
    model = LinearMeasurement(motion.state_layout, ("x_m", "y_m", "z_m"))
    unscented, taylor, steps_s = (
        UnscentedTransform(alpha, beta, kappa),
        TaylorLinearisation(),
        np.ones(2),
    )
    mean_u, covariance_u = unscented.predict(motion, bank, steps_s)
    mean_t, covariance_t = taylor.predict(motion, bank, steps_s)
    sigma = _sigma(bank.covariance)
    _assert_scaled_close(mean_u, mean_t, sigma)
    _assert_scaled_close(covariance_u, covariance_t, _outer(sigma))

    predicted = replace(bank, mean=mean_t, covariance=covariance_t)
    moments_u, moments_t = unscented.measure(model, predicted), taylor.measure(model, predicted)
    sigma = _sigma(covariance_t)
    measured = sigma[:, list(model.indices)]
    _assert_scaled_close(moments_u.predicted_measurement, moments_t.predicted_measurement, measured)
    _assert_scaled_close(
        moments_u.cross_covariance,
        moments_t.cross_covariance,
        sigma[:, :, None] * measured[:, None, :],
    )
    _assert_scaled_close(
        moments_u.predicted_measurement_covariance,
        moments_t.predicted_measurement_covariance,
        _outer(measured),
    )
    assert moments_u.measurement_matrix is None and moments_t.measurement_matrix is not None


def test_the_ukf_and_the_kf_agree_over_200_steps(rng: np.random.Generator) -> None:
    """Spec §3: the same measurements through both filters, judged on the current spread."""
    motion = CartesianMotion()
    model = LinearMeasurement(motion.state_layout, ("x_m", "y_m", "z_m"))
    noise = np.diag([50.0**2] * 3)
    ukf = kf = _guard_bank()
    truth = kf.mean
    for step in range(1, 201):
        # A loop over time: each step starts from the one before.
        truth = motion.transition(truth, 1.0)
        z = model.predict(truth) + rng.normal(0.0, 50.0, (2, 3))
        ukf = _step(
            predict(ukf, motion, UnscentedTransform(), step), model, UnscentedTransform(), z, noise
        )
        kf = _step(
            predict(kf, motion, TaylorLinearisation(), step), model, TaylorLinearisation(), z, noise
        )
        sigma = _sigma(kf.covariance)
        _assert_scaled_close(ukf.mean, kf.mean, sigma)
        _assert_scaled_close(ukf.covariance, kf.covariance, _outer(sigma))


# The prior's spread is far wider than the radar's noise: 2 km against 20 m in range and
# 2 mrad (40 m) in angle at 20 km, so J P Jᵀ dominates R across the s range and the gain stays
# of order one. At much smaller s the gap falls as s⁴ and the slope bound would test nothing.
SECOND_ORDER_SCALES = np.geomspace(1 / 8, 1, 7)


def test_the_ekf_and_the_ukf_differ_at_second_order_on_the_spherical_model() -> None:
    """Spec §3: scale the prior covariance by s²; the gap in the updated mean shrinks as s².

    The UKF's predicted measurement is h(x) + ½ tr(∇²h P) to leading order (Särkkä 2013,
    ch. 5); the EKF's is h(x). A wrong Jacobian leaves a first-order gap, slope about 1. The
    seven scales are seven rows of one bank.
    """
    layout = CartesianMotion().state_layout
    model = RangeAzimuthElevationRangeRate(layout, np.zeros(3))
    mean = np.array([12_000.0, -150.0, 16_000.0, 20.0, 3_000.0, 5.0])
    sigma = np.array([2_000.0, 100.0] * 3)
    noise = np.diag([20.0**2, 2e-3**2, 2e-3**2, 1.0**2])
    z = model.predict(mean)[None] + np.array([[30.0, 2e-3, -2e-3, 1.0]])
    scaled = SECOND_ORDER_SCALES[:, None, None] ** 2 * np.diag(sigma**2)
    prior = _bank(layout, np.full((7, 6), mean), scaled)
    updated = [
        update(prior, innovate(prior, model, lin, z, noise), np.zeros(7)).mean
        for lin in (TaylorLinearisation(), UnscentedTransform())
    ]
    gaps = np.linalg.norm((updated[0] - updated[1]) / sigma, axis=1)
    slope = np.polyfit(np.log(SECOND_ORDER_SCALES), np.log(gaps), 1)[0]
    assert slope >= 1.8


def test_a_kalman_update_is_the_textbook_one(linearisation: Linearisation) -> None:
    """Predict and update written out with F, Q, H and an explicit inverse, independently."""
    motion = CartesianMotion("x")
    model = LinearMeasurement(motion.state_layout, ("x_m",))
    bank = _bank(motion.state_layout, [[1.0, 2.0]], [[4.0, 1.0], [1.0, 2.0]])
    f, q = motion.jacobian(np.zeros(2), 1.0), motion.process_noise(np.zeros(2), 1.0)
    h, r = np.array([[1.0, 0.0]]), np.array([[0.5]])
    mean, covariance = f @ bank.mean[0], f @ bank.covariance[0] @ f.T + q
    gain = covariance @ h.T @ np.linalg.inv(h @ covariance @ h.T + r)
    expected_mean = mean + gain @ (np.array([4.0]) - h @ mean)
    expected_covariance = (np.eye(2) - gain @ h) @ covariance
    result = _step(predict(bank, motion, linearisation, 1.0), model, linearisation, [[4.0]], r)
    # rtol and atol 1e-10: sigma points, the Joseph form and solves each add roundoff.
    np.testing.assert_allclose(result.mean[0], expected_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result.covariance[0], expected_covariance, rtol=1e-10, atol=1e-10)


def test_a_range_only_update_leaves_an_uncorrelated_rate_alone(
    linearisation: Linearisation,
) -> None:
    """With a diagonal prior, a range cannot inform the rate: P_xz has an exact zero there."""
    motion = RadialConstantVelocity()
    model = LinearMeasurement(motion.state_layout, ("bistatic_range_m",))
    prior = _bank(motion.state_layout, [[10_000.0, 0.0]], np.diag([400.0, 200.0**2]))
    posterior = _step(prior, model, linearisation, [[10_100.0]], [[400.0]])
    assert posterior.mean[0, 1] == 0.0 and posterior.covariance[0, 1, 1] == 200.0**2
    assert posterior.mean[0, 0] == pytest.approx(10_050.0, rel=1e-12)


def test_an_innovation_across_due_north_takes_the_short_way(linearisation: Linearisation) -> None:
    layout = CartesianMotion().state_layout
    model = RangeAzimuthElevationRangeRate(layout, np.zeros(3), ("range_m", "azimuth_rad"))
    bank = _bank(layout, [[-10.0, 0.0, 10_000.0, 0.0, 0.0, 0.0]], np.diag([1.0, 1.0] * 3))
    innovation = innovate(bank, model, linearisation, [[10_000.0, 0.001]], np.diag([1.0, 1e-6]))
    # atol 1e-6 rad: the predicted azimuth is 2π - 1e-3 to within the sigma-point spread.
    np.testing.assert_allclose(innovation.innovation[0, 0, 1], 0.002, atol=1e-6)


def test_the_batched_innovation_equals_a_per_pair_loop(
    linearisation: Linearisation, rng: np.random.Generator
) -> None:
    """Spec §7: every (track, measurement) pair scored at once equals each pair alone."""
    layout = CartesianMotion().state_layout
    model = RangeAzimuthElevationRangeRate(layout, [100.0, 50.0, 10.0])
    means = rng.uniform(-100.0, 100.0, (5, 6))
    means[:, [0, 2, 4]] = rng.uniform(2_000.0, 20_000.0, (5, 3))
    roots = rng.normal(0.0, 1.0, (5, 6, 6))
    bank = _bank(layout, means, 100.0 * roots @ np.swapaxes(roots, 1, 2) + 25.0 * np.eye(6))
    z = model.predict(means[rng.integers(0, 5, 7)]) + rng.normal(0.0, 1e-3, (7, 4))
    noise = np.diag([25.0, 1e-6, 1e-6, 0.25]) * rng.uniform(0.5, 2.0, (7, 1, 1))
    batched = innovate(bank, model, linearisation, z, noise)
    for i in range(5):
        # A loop over pairs, the reference the batch must equal.
        row = bank.compact(np.arange(5) == i)
        for j in range(7):
            alone = innovate(row, model, linearisation, z[j : j + 1], noise[j])
            # rtol 1e-12: the same operations on a stack of one.
            np.testing.assert_allclose(batched.nis[i, j], alone.nis[0, 0], rtol=1e-12)
            np.testing.assert_allclose(
                batched.log_likelihood[i, j], alone.log_likelihood[0, 0], rtol=1e-12
            )
            np.testing.assert_allclose(batched.innovation[i, j], alone.innovation[0, 0], rtol=1e-12)


def test_the_batched_nis_and_likelihood_are_the_textbook_formulas(rng: np.random.Generator) -> None:
    """The EKF innovation, S = H P Hᵀ + R, d² and log-likelihood per pair, by solve and slogdet."""
    layout = CartesianMotion().state_layout
    model = BistaticRangeRateModel(layout, np.zeros(3), [20_000.0, 0.0, 0.0])
    means = rng.uniform(2_000.0, 15_000.0, (4, 6)) * np.array([1.0, 0.01] * 3)
    bank = _bank(layout, means, np.diag([400.0, 25.0] * 3))
    z = model.predict(means)[[0, 1, 2]] + [[30.0, 1.0]]
    noise = np.diag([100.0, 1.0])
    batched = innovate(bank, model, TaylorLinearisation(), z, noise)
    for i in range(4):
        # A loop over pairs: the independent per-pair formulas.
        h = model.jacobian(means[i])
        s = h @ bank.covariance[i] @ h.T + noise
        for j in range(3):
            nu = z[j] - model.predict(means[i])
            nis = nu @ np.linalg.solve(s, nu)
            log_likelihood = -0.5 * (2 * np.log(2 * np.pi) + np.linalg.slogdet(s)[1] + nis)
            # rtol 1e-10: Cholesky against LU on a well-conditioned 2 x 2.
            np.testing.assert_allclose(batched.nis[i, j], nis, rtol=1e-10)
            np.testing.assert_allclose(batched.log_likelihood[i, j], log_likelihood, rtol=1e-10)


def _long_run(kind: str, linearisation: Linearisation, rng: np.random.Generator) -> list[Any]:
    """200 noisy updates of a crossing target; return every posterior covariance."""
    motion = CartesianMotion()
    layout = motion.state_layout
    if kind == "spherical":
        model: Any = RangeAzimuthElevationRangeRate(layout, np.zeros(3))
        noise = np.diag([20.0**2, 2e-3**2, 2e-3**2, 1.0])
    else:
        model = BistaticRangeRateModel(layout, np.zeros(3), [20_000.0, 0.0, 0.0])
        noise = np.diag([20.0**2, 1.0])
    truth = np.array([12_000.0, -150.0, 16_000.0, 0.0, 3_000.0, 0.0])
    prior_covariance = np.diag([200.0**2, 30.0**2] * 3)
    bank = _bank(
        layout, truth + rng.multivariate_normal(np.zeros(6), prior_covariance), prior_covariance
    )
    covariances = []
    for step in range(1, 201):
        # A loop over time: each step starts from the one before.
        truth = motion.transition(truth, 1.0)
        z = model.predict(truth)[None] + rng.multivariate_normal(np.zeros(len(noise)), noise)
        bank = _step(predict(bank, motion, linearisation, step), model, linearisation, z, noise)
        covariances.append(bank.covariance[0])
    return covariances


@pytest.mark.parametrize("kind", ["spherical", "bistatic"])
def test_the_covariance_stays_symmetric_and_positive_definite_over_200_steps(
    kind: str, linearisation: Linearisation, rng: np.random.Generator
) -> None:
    """Joseph form with H, the short form without; each ends by taking the symmetric part."""
    covariances = _long_run(kind, linearisation, rng)
    for covariance in covariances:
        # rtol 1e-12: the update takes the symmetric part, so any asymmetry is roundoff.
        np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12)
    assert min(np.linalg.eigvalsh(c)[0] for c in covariances) > 0.0


N_RUNS, N_STEPS = 200, 30


@pytest.fixture(scope="session")
def consistency_runs(linearisation: Linearisation) -> tuple[float, float]:
    """Run N_RUNS independent tracks, as the rows of one bank; return the mean NEES and NIS.

    Truth uses the same F, Q and R as the filter, so any failure is the filter's.
    """
    rng = np.random.default_rng(SEED)
    motion = CartesianMotion("x", sigma_acceleration_mps2=1.0, acceleration_correlation_time_s=0.5)
    model = LinearMeasurement(motion.state_layout, ("x_m",))
    f, q = motion.jacobian(np.zeros(2), 1.0), motion.process_noise(np.zeros(2), 1.0)
    p0, noise = np.diag([100.0, 25.0]), np.array([[4.0]])
    truth = rng.multivariate_normal([1000.0, 10.0], p0, N_RUNS)
    bank = _bank(motion.state_layout, np.full((N_RUNS, 2), [1000.0, 10.0]), p0)
    nis = []
    for step in range(1, N_STEPS + 1):
        # A loop over time; the runs are the bank's rows.
        truth = truth @ f.T + rng.multivariate_normal(np.zeros(2), q, N_RUNS)
        z = truth[:, :1] + rng.normal(0.0, 2.0, (N_RUNS, 1))
        bank = predict(bank, motion, linearisation, step)
        innovation = innovate(bank, model, linearisation, z, noise)
        nis.append(np.diagonal(innovation.nis))
        bank = update(bank, innovation, np.arange(N_RUNS))
    return float(np.mean(nees(bank.mean, bank.covariance, truth))), float(np.mean(nis))


@pytest.mark.slow
def test_nees_is_chi_square_over_independent_runs(consistency_runs: tuple[float, float]) -> None:
    """A consistent filter's NEES is chi2(n_state); Bar-Shalom, Li and Kirubarajan, §5.4.2.

    Slow: a Monte Carlo, shared with the NIS test through a session fixture.
    """
    low, high = consistency_interval(2, N_RUNS, 0.999)
    assert low <= consistency_runs[0] <= high


@pytest.mark.slow
def test_nis_is_chi_square_over_independent_runs(consistency_runs: tuple[float, float]) -> None:
    """A consistent filter's NIS is chi2(n_meas), independent from step to step. Slow."""
    low, high = consistency_interval(1, N_RUNS * N_STEPS, 0.999)
    assert low <= consistency_runs[1] <= high


SIGMA_RANGE_M = 21.635652855125496
BOOTSTRAP_THRESHOLD_MPS = 15.2955 / 6.0


def test_the_dwna_rate_variance_floors_above_the_bootstrap_threshold() -> None:
    """Why scenario 003's unfold test cannot be the filter's own rate covariance.

    Re-homed from ``tests/core/test_tracking.py``, which ``pipelines/tracking.py`` cites:
    with ``sigma_accel_mps2 = 2`` (one-way), DWNA adds sigma² T² to the rate variance every
    scan, so on range alone the rate standard deviation floors near 4.1 m/s, above the
    v_span / 6 = 2.55 m/s promotion threshold, and never unfolds. The state here is bistatic
    range, twice the one-way range, so sigma, R and the floor all double at the boundary. The
    floor is the steady state of the Riccati recursion, which 60 scans reach.
    """
    motion = RadialConstantVelocity(2 * 2.0, process_noise="dwna")
    model = LinearMeasurement(motion.state_layout, ("bistatic_range_m",))
    noise = np.array([[(2 * SIGMA_RANGE_M) ** 2]])
    bank = _bank(motion.state_layout, [[40_000.0, 0.0]], np.diag([noise[0, 0], (2 * 200.0) ** 2]))
    for step in range(1, 61):
        # A loop over scans: each starts from the one before.
        bank = predict(bank, motion, TaylorLinearisation(), step)
        bank = _step(bank, model, TaylorLinearisation(), [[2 * (20_000.0 - 80.0 * step)]], noise)
    f, q = motion.jacobian(np.zeros(2), 1.0), motion.process_noise(np.zeros(2), 1.0)
    h = model.measurement_matrix
    prior = solve_discrete_are(f.T, h.T, q, noise)
    steady = prior - prior @ h.T @ np.linalg.solve(h @ prior @ h.T + noise, h @ prior)
    # rtol 1e-9: the recursion contracts about 0.66 a scan, from 7.5e-6 at scan 30 to 2e-11.
    np.testing.assert_allclose(bank.covariance[0, 1, 1], steady[1, 1], rtol=1e-9)
    assert np.sqrt(steady[1, 1]) / 2 > BOOTSTRAP_THRESHOLD_MPS
