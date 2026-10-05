"""Tests for radar_forge.core.tracking.association: gating and assignment.

Ground truth is analytic wherever it can be: the gate thresholds are tabulated
chi-square quantiles, and the assignment tests use small hand-built cost
matrices with one known optimum. The one statistical test, the gate's
acceptance rate, uses a binomial confidence interval rather than a tolerance
(docs/conventions/testing.md §2).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import beta

from radar_forge.core.tracking.association import (
    ChiSquareGate,
    GlobalNearestNeighbour,
    NearestNeighbour,
)
from radar_forge.core.tracking.estimation import innovation_stats


@pytest.fixture
def rng() -> np.random.Generator:
    """One seeded generator, so a failure is reproducible."""
    return np.random.default_rng(20261005)


# --------------------------------------------------------------------------- #
# ChiSquareGate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("n_dimensions", "expected"),
    [(1, 6.635), (2, 9.210), (3, 11.345), (4, 13.277)],
)
def test_the_gate_threshold_matches_the_tabulated_quantiles(
    n_dimensions: int, expected: float
) -> None:
    """Chi-square quantiles at 0.99, as printed in any statistics table.

    abs 5e-4 because the table gives three decimals.
    """
    assert ChiSquareGate(0.99).threshold(n_dimensions) == pytest.approx(expected, abs=5e-4)


def test_the_gate_threshold_grows_with_dimension() -> None:
    """More measurement components means a larger admissible NIS."""
    thresholds = [ChiSquareGate(0.99).threshold(n) for n in (1, 2, 3, 4)]
    assert thresholds == sorted(thresholds)
    assert len(set(thresholds)) == len(thresholds)


def test_the_gate_accepts_the_fraction_of_true_measurements_it_claims(
    rng: np.random.Generator,
) -> None:
    """A 0.99 gate keeps 99 % of correctly modelled innovations.

    The innovations are drawn from the covariance the gate is told about, so
    their NIS is chi-square by construction, and the test goes through
    ``innovation_stats`` and ``accepts`` as the tracker does. The check is an
    exact Clopper-Pearson interval at 99.9 %, as for a Monte-Carlo statistic.
    """
    gate = ChiSquareGate(0.99)
    n_trials = 4_000
    covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    residuals = rng.multivariate_normal(np.zeros(2), covariance, size=n_trials)
    # Each trial is one call to the public gate, which takes one innovation at
    # a time, so the trials cannot be one array operation.
    accepted = sum(
        gate.accepts(innovation_stats(np.zeros(2), residual, covariance)) for residual in residuals
    )
    low = beta.ppf(0.0005, accepted, n_trials - accepted + 1)
    high = beta.ppf(0.9995, accepted + 1, n_trials - accepted)
    assert low <= gate.probability <= high


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.5])
def test_the_gate_rejects_a_probability_outside_zero_to_one(probability: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        ChiSquareGate(probability)


def test_the_gate_threshold_rejects_a_dimension_below_one() -> None:
    with pytest.raises(ValueError, match="dim"):
        ChiSquareGate(0.99).threshold(0)


# --------------------------------------------------------------------------- #
# GlobalNearestNeighbour
# --------------------------------------------------------------------------- #


def test_gnn_returns_the_known_optimum() -> None:
    """Greedy would take (0, 1) first and pay 1 + 8 = 9; the optimum is 2 + 2."""
    result = GlobalNearestNeighbour().associate(np.array([[2.0, 1.0], [8.0, 2.0]]))
    assert result.matches == ((0, 0), (1, 1))


def test_gnn_never_assigns_a_gated_out_pair() -> None:
    result = GlobalNearestNeighbour().associate(np.array([[1.0, np.inf], [np.inf, np.inf]]))
    assert result.matches == ((0, 0),)
    assert result.unassigned_tracks == (1,)
    assert result.unassigned_measurements == (1,)


def test_gnn_assigns_nothing_when_every_pair_is_gated_out() -> None:
    result = GlobalNearestNeighbour().associate(np.full((2, 3), np.inf))
    assert result.matches == ()
    assert result.unassigned_tracks == (0, 1)
    assert result.unassigned_measurements == (0, 1, 2)


@pytest.mark.parametrize("shape", [(0, 0), (2, 0), (0, 3)])
def test_gnn_handles_an_empty_cost_matrix(shape: tuple[int, int]) -> None:
    result = GlobalNearestNeighbour().associate(np.zeros(shape))
    assert result.matches == ()
    assert result.unassigned_tracks == tuple(range(shape[0]))
    assert result.unassigned_measurements == tuple(range(shape[1]))


def test_gnn_assigns_each_track_and_measurement_at_most_once() -> None:
    result = GlobalNearestNeighbour().associate(np.array([[1.0, 1.1, 1.2], [1.3, 1.0, 1.1]]))
    tracks = [i for i, _ in result.matches]
    measurements = [j for _, j in result.matches]
    assert len(set(tracks)) == len(tracks) == 2
    assert len(set(measurements)) == len(measurements) == 2


def test_gnn_leaves_extra_measurements_unassociated() -> None:
    """One target and two false alarms: the target takes its own measurement."""
    result = GlobalNearestNeighbour().associate(np.array([[0.5, 2.0, 3.0]]))
    assert result.matches == ((0, 0),)
    assert result.unassigned_measurements == (1, 2)


def test_gnn_maximises_the_number_of_matches_before_the_cost() -> None:
    """Two pairs at total cost -99 beat one pair at -100.

    Track 0's cheapest measurement is 0, but taking it would leave track 1,
    which can only use measurement 0, without one. Negative costs, as a
    negative log-likelihood gives, must not change that.
    """
    result = GlobalNearestNeighbour().associate(np.array([[-100.0, 0.0], [-99.0, np.inf]]))
    assert set(result.matches) == {(0, 1), (1, 0)}


@pytest.mark.parametrize("costs", [np.zeros(3), np.zeros((2, 2, 2))], ids=["1-D", "3-D"])
def test_gnn_rejects_a_cost_matrix_that_is_not_two_dimensional(costs: np.ndarray) -> None:
    with pytest.raises(ValueError, match="2-D"):
        GlobalNearestNeighbour().associate(costs)


@pytest.mark.parametrize("bad", [np.nan, -np.inf])
def test_gnn_rejects_nan_and_negative_infinity(bad: float) -> None:
    with pytest.raises(ValueError, match="2-D"):
        GlobalNearestNeighbour().associate(np.array([[1.0, bad]]))


# --------------------------------------------------------------------------- #
# NearestNeighbour
# --------------------------------------------------------------------------- #


def test_nn_takes_the_cheapest_pair_first() -> None:
    """On the matrix where greedy is not optimal, NN gives the greedy answer."""
    result = NearestNeighbour().associate(np.array([[2.0, 1.0], [8.0, 2.0]]))
    assert result.matches == ((0, 1), (1, 0))


def test_nn_never_assigns_a_gated_out_pair() -> None:
    result = NearestNeighbour().associate(np.array([[np.inf, 1.0], [np.inf, 0.5]]))
    assert result.matches == ((1, 1),)
    assert result.unassigned_tracks == (0,)
    assert result.unassigned_measurements == (0,)
