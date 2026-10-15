"""Tests for radar_forge.core.tracking.linearisation: how a mean and covariance cross a model.

The sigma points must reproduce the estimate's mean and covariance, and the weights must be the
ones the docstring derives. Whether a filter built on these is right is
``test_gaussian_filter.py``'s business.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.linearisation import UnscentedLinearisation
from radar_forge.core.tracking.motion import CartesianMotion

ORIGIN = (36.00250, -78.94100, 60.0)


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
