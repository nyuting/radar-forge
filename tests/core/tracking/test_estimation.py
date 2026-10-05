"""Tests for radar_forge.core.tracking.estimation: the NIS and the innovation log-likelihood.

The NIS is the statistic the gate tests and the consistency check averages. A filter with a
mis-scaled R or Q still follows the target; what it gets wrong is its own confidence, and the
NIS is what notices. Bar-Shalom, Li and Kirubarajan (2001), §5.4.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from radar_forge.core.tracking.estimation import innovation_stats

# Scenario 003's range and range-rate standard deviations, so the tests run at the six orders of
# magnitude between the two variances that the tracker sees.
SIGMA_RANGE_M = 21.635652855125496
SIGMA_VELOCITY_MPS = 0.017247813329811023
SEED = 20261005


@pytest.fixture
def rng() -> np.random.Generator:
    """One seeded generator, so a failure is reproducible."""
    return np.random.default_rng(SEED)


def test_the_scalar_nis_is_the_squared_residual_over_the_variance() -> None:
    """With one element, d² is simply (residual / sigma)²."""
    assert innovation_stats([0.0], [4.0], [[4.0]]).nis == pytest.approx(4.0, rel=1e-12)


def test_nis_is_invariant_to_the_units_of_each_element() -> None:
    """Scaling an element and its variance together cannot change d².

    This is what makes the spread between the range and range-rate variances harmless, and
    what a transposed R breaks.
    """
    residual = np.array([10.0, 0.02])
    covariance = np.diag([SIGMA_RANGE_M**2, SIGMA_VELOCITY_MPS**2])
    scale = np.diag([1e3, 1.0])
    scaled = innovation_stats(np.zeros(2), scale @ residual, scale @ covariance @ scale.T)
    # rel 1e-12: two triangular solves of a diagonal 2 x 2 system.
    assert scaled.nis == pytest.approx(
        innovation_stats(np.zeros(2), residual, covariance).nis, rel=1e-12
    )


def test_nis_is_chi_squared_distributed_under_a_correct_model(rng: np.random.Generator) -> None:
    """Residuals drawn from S itself give a mean NIS equal to the dimension.

    Asserted as a confidence interval on the mean of n_trials chi-squared(2) draws, whose
    variance is 2 * dim / n_trials.
    """
    dim, n_trials = 2, 5_000
    covariance = np.diag([SIGMA_RANGE_M**2, SIGMA_VELOCITY_MPS**2])
    residuals = rng.standard_normal((n_trials, dim)) @ np.linalg.cholesky(covariance).T
    # One call per draw: innovation_stats scores one innovation at a time.
    draws = [innovation_stats(np.zeros(dim), r, covariance).nis for r in residuals]
    standard_error = np.sqrt(2.0 * dim / n_trials)
    assert abs(float(np.mean(draws)) - dim) < 4.0 * standard_error


def test_the_log_likelihood_is_the_gaussian_log_density() -> None:
    covariance = np.array([[SIGMA_RANGE_M**2, 0.1], [0.1, SIGMA_VELOCITY_MPS**2]])
    residual = np.array([12.0, -0.01])
    expected = multivariate_normal(np.zeros(2), covariance).logpdf(residual)
    # rel 1e-10: two routes through a 2 x 2 factorisation whose condition number is about 1e6.
    assert innovation_stats(np.zeros(2), residual, covariance).log_likelihood == pytest.approx(
        expected, rel=1e-10
    )


def test_a_singular_innovation_covariance_is_rejected() -> None:
    with pytest.raises(ValueError, match="singular"):
        innovation_stats(np.zeros(2), [1.0, 1.0], np.zeros((2, 2)))


def test_an_indefinite_innovation_covariance_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        innovation_stats(np.zeros(2), [1.0, 1.0], np.diag([1.0, -1.0]))


@pytest.mark.parametrize(
    ("predicted", "residual", "covariance"),
    [
        ([0.0, 0.0], [1.0, 2.0], [[1.0]]),
        ([0.0], [1.0, 2.0], np.eye(2)),
        ([[0.0, 0.0]], [[1.0, 2.0]], np.eye(2)),
    ],
    ids=["covariance too small", "predicted too short", "not vectors"],
)
def test_mismatched_shapes_are_rejected(
    predicted: object, residual: object, covariance: object
) -> None:
    with pytest.raises(ValueError, match="do not agree"):
        innovation_stats(predicted, residual, covariance)  # type: ignore[arg-type]
