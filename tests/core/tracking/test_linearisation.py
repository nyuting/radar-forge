"""Tests for radar_forge.core.tracking.linearisation: how a mean and covariance cross a model.

The sigma points must reproduce the estimate's mean and covariance, and the weights must be the
ones the docstring derives. The drift guards of ``spec/tracker-001.md`` §4.5 check that on
linear models the unscented and Taylor linearisations agree, so the UKF is the KF; they replace
an older test that compared the UKF with a hand-coded third copy of the Kalman update.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.estimation import GaussianFilter
from radar_forge.core.tracking.linearisation import TaylorLinearisation, UnscentedLinearisation
from radar_forge.core.tracking.measurement_models import CartesianPosition, Measurement
from radar_forge.core.tracking.motion import CartesianMotion

ORIGIN = (36.00250, -78.94100, 60.0)
SEED = 20261014


@pytest.fixture
def rng() -> np.random.Generator:
    """One seeded generator, so a failure is reproducible."""
    return np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# Sigma points and weights
# --------------------------------------------------------------------------- #


def _six_state_estimate() -> StateEstimate:
    motion = CartesianMotion({"x": "CV", "y": "CV", "z": "CV"}, origin_lla_deg_m=ORIGIN)
    root = np.arange(1.0, 37.0).reshape(6, 6) / 10
    covariance = root @ root.T + np.eye(6)
    return StateEstimate(np.arange(6.0), covariance, 0, motion.state_layout)


def test_the_weighted_sigma_points_reproduce_the_mean() -> None:
    estimate = _six_state_estimate()
    linearisation = UnscentedLinearisation()
    wm, _ = linearisation.weights(6)
    # rtol 1e-12, atol 1e-12: one weighted sum of 13 points, with weights that cancel.
    np.testing.assert_allclose(
        wm @ linearisation.sigma_points(estimate), estimate.mean, rtol=1e-12, atol=1e-12
    )


def test_the_weighted_sigma_points_reproduce_the_covariance() -> None:
    estimate = _six_state_estimate()
    linearisation = UnscentedLinearisation()
    _, wc = linearisation.weights(6)
    deviations = linearisation.sigma_points(estimate) - estimate.mean
    # rtol 1e-12: a Cholesky factor and one weighted outer-product sum.
    np.testing.assert_allclose((deviations.T * wc) @ deviations, estimate.covariance, rtol=1e-12)


def test_the_mean_weights_sum_to_one() -> None:
    wm, _ = UnscentedLinearisation().weights(6)
    # rel 1e-12: 13 float64 additions.
    assert wm.sum() == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("kappa", [0.0, 1.0])
@pytest.mark.parametrize("n_state", range(1, 10))
def test_with_alpha_one_no_covariance_weight_is_negative(n_state: int, kappa: float) -> None:
    """Class Notes: alpha = 1 with kappa >= 0 gives non-negative weights up to 3-D CA."""
    _, wc = UnscentedLinearisation(alpha=1.0, kappa=kappa).weights(n_state)
    assert wc.min() >= 0.0


@pytest.mark.parametrize("n_state", [2, 6, 9])
def test_the_default_weights_are_the_unscaled_transform(n_state: int) -> None:
    """alpha = 1, beta = 2, kappa = 0: W0m = 0, W0c = beta and Wi = 1/(2n) (class Notes)."""
    wm, wc = UnscentedLinearisation().weights(n_state)
    expected = np.full(2 * n_state + 1, 1 / (2 * n_state))
    expected[0] = 0.0
    # rtol 1e-12: each weight is one or two float64 operations on exact inputs.
    np.testing.assert_allclose(wm, expected, rtol=1e-12, atol=1e-15)
    expected[0] = 2.0
    np.testing.assert_allclose(wc, expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("n_state", range(4, 10))
def test_the_julier_uhlmann_kappa_keeps_the_covariance_weights_non_negative(n_state: int) -> None:
    """kappa = 3 - n: W0m < 0 for n > 3, yet W0c >= 0 up to n = 9 (class Notes)."""
    wm, wc = UnscentedLinearisation(kappa=3.0 - n_state).weights(n_state)
    assert wm[0] < 0.0
    assert wc.min() >= 0.0


@pytest.mark.parametrize("alpha", [0.5, 1e-3])
def test_a_small_alpha_is_rejected_when_a_filter_is_built(alpha: float) -> None:
    """A small alpha makes W0c negative, which the short-form update cannot allow."""
    motion = CartesianMotion({"x": "CV", "y": "CV", "z": "CV"}, origin_lla_deg_m=ORIGIN)
    with pytest.raises(ValueError, match="negative covariance weight"):
        UnscentedLinearisation(alpha=alpha).check_motion_model(motion)


# --------------------------------------------------------------------------- #
# Drift guards: on linear models the UKF is the KF (spec/tracker-001.md §4.5)
# --------------------------------------------------------------------------- #

# rtol 1e-10, and an atol of 1e-10 times each entry's own scale from the prior: the entries
# mix m², m²/s and (m/s)², so one absolute tolerance cannot suit them all. The sigma points
# reproduce F P Fᵀ through a Cholesky factor and weighted sums, which costs roundoff of a few
# float64 epsilon times the entry's scale, far below 1e-10.
GUARD_RTOL = 1e-10
GUARD_SCALE = 1e-10

N_GUARD = 6
# (alpha, beta, kappa): the defaults, Julier and Uhlmann's kappa = 3 - n, and beta = 0. Each
# gives a different centre weight, which §4.4 says must not matter on a linear model.
SIGMA_SETTINGS = [(1.0, 2.0, 0.0), (1.0, 2.0, 3.0 - N_GUARD), (1.0, 0.0, 0.0)]


def _guard_motion() -> CartesianMotion:
    return CartesianMotion({"x": "CV", "y": "CV", "z": "CV"}, origin_lla_deg_m=ORIGIN)


def _guard_prior(motion: CartesianMotion) -> StateEstimate:
    """A correlated prior whose positions (km) and velocities (tens of m/s) differ in scale."""
    scale = np.array([1000.0, 30.0, 1000.0, 30.0, 1000.0, 30.0])
    root = np.arange(1.0, 37.0).reshape(6, 6) / 10
    correlation = root @ root.T + np.eye(6)
    correlation /= np.sqrt(np.outer(np.diag(correlation), np.diag(correlation)))
    covariance = correlation * np.outer(scale, scale)
    mean = np.array([5000.0, -20.0, 3000.0, 15.0, 1000.0, 2.0])
    return StateEstimate(mean, covariance, 0.0, motion.state_layout)


def _assert_scaled_close(actual: Any, expected: Any, scale: Any) -> None:
    """|actual - expected| <= GUARD_SCALE * scale + GUARD_RTOL * |expected|, entry by entry."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    allowed = GUARD_SCALE * np.asarray(scale) + GUARD_RTOL * np.abs(expected)
    np.testing.assert_array_less(np.abs(actual - expected), allowed + np.finfo(float).tiny)


def _sigma(covariance: Any) -> NDArray[np.float64]:
    """The standard deviation of each state element: the scale its entries are judged on."""
    return np.sqrt(np.diag(covariance))


@pytest.mark.parametrize(("alpha", "beta", "kappa"), SIGMA_SETTINGS)
def test_the_unscented_moments_are_the_kalman_moments_on_linear_models(
    alpha: float, beta: float, kappa: float
) -> None:
    """§4.4 steps 1-4: x̂⁻, P⁻, ẑ, P_xz and S agree, whatever the accepted weights."""
    motion = _guard_motion()
    model = CartesianPosition(motion.state_layout, ("x_m", "y_m", "z_m"))
    prior = _guard_prior(motion)
    unscented = UnscentedLinearisation(alpha, beta, kappa)
    unscented.check_motion_model(motion)
    taylor = TaylorLinearisation()

    mean_u, covariance_u = unscented.predict(motion, prior, 1.0)
    mean_t, covariance_t = taylor.predict(motion, prior, 1.0)
    sigma = _sigma(prior.covariance)
    _assert_scaled_close(mean_u, mean_t, sigma)
    _assert_scaled_close(covariance_u, covariance_t, np.outer(sigma, sigma))

    predicted = StateEstimate(mean_t, covariance_t, 1.0, motion.state_layout)
    moments_u = unscented.measure(model, predicted)
    moments_t = taylor.measure(model, predicted)
    sigma = _sigma(predicted.covariance)
    measured = list(model.indices)
    _assert_scaled_close(
        moments_u.predicted_measurement, moments_t.predicted_measurement, sigma[measured]
    )
    _assert_scaled_close(
        moments_u.cross_covariance, moments_t.cross_covariance, np.outer(sigma, sigma[measured])
    )
    _assert_scaled_close(
        moments_u.predicted_measurement_covariance,
        moments_t.predicted_measurement_covariance,
        np.outer(sigma[measured], sigma[measured]),
    )


def test_the_ukf_and_the_kf_agree_over_200_steps(rng: np.random.Generator) -> None:
    """The same prior and measurements through both filters: §4.4 step 5, step by step."""
    motion = _guard_motion()
    model = CartesianPosition(motion.state_layout, ("x_m", "y_m", "z_m"))
    prior = _guard_prior(motion)
    noise = np.diag([50.0**2] * 3)
    ukf = GaussianFilter(prior, motion, UnscentedLinearisation())
    kf = GaussianFilter(prior, motion, TaylorLinearisation())
    sigma = _sigma(prior.covariance)
    truth = prior.mean
    for step in range(1, 201):
        # A loop over time: each step starts from the one before.
        truth = motion.transition(truth, 1.0)
        value = model.predict(truth) + rng.normal(0.0, 50.0, 3)
        measurement = Measurement(value, noise, float(step), "sensor", "position")
        for estimator in (ukf, kf):
            estimator.predict_to(step)
            estimator.update(measurement, model)
        _assert_scaled_close(ukf.state.mean, kf.state.mean, sigma)
        _assert_scaled_close(ukf.state.covariance, kf.state.covariance, np.outer(sigma, sigma))


# --------------------------------------------------------------------------- #
# The Taylor linearisation is the textbook KF
# --------------------------------------------------------------------------- #


def test_the_taylor_prediction_is_f_x_and_f_p_ft_plus_q() -> None:
    motion = _guard_motion()
    prior = _guard_prior(motion)
    f, q = motion.matrices(2.0)
    mean, covariance = TaylorLinearisation().predict(motion, prior, 2.0)
    # rtol 1e-12: the same matrix products, then a symmetrisation that changes only roundoff.
    np.testing.assert_allclose(mean, f @ prior.mean, rtol=1e-12)
    np.testing.assert_allclose(covariance, f @ prior.covariance @ f.T + q, rtol=1e-12)


def test_the_taylor_measurement_supplies_h_and_the_unscented_does_not() -> None:
    """The one difference the correction core sees: Joseph form with H, short form without."""
    motion = _guard_motion()
    model = CartesianPosition(motion.state_layout, ("x_m",))
    prior = _guard_prior(motion)
    np.testing.assert_array_equal(
        TaylorLinearisation().measure(model, prior).measurement_matrix,
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    assert UnscentedLinearisation().measure(model, prior).measurement_matrix is None
