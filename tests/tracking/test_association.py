"""Tests for radar_forge.tracking.association: gating, costs and assignment.

Ground truth is analytic where it can be: tabulated chi-square quantiles, and small cost
matrices with one known optimum. The gate's acceptance rate is a Monte Carlo statistic, judged
by a Clopper-Pearson interval rather than a tolerance (docs/conventions/testing.md §2).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.stats import beta

from radar_forge.tracking.association import (
    GlobalNearestNeighbour,
    GreedyNearestNeighbour,
    association_cost,
    chi_square_gate_threshold,
    gate,
    global_nearest_neighbour,
    greedy_nearest_neighbour,
)
from radar_forge.tracking.filters import TaylorLinearisation, TrackBank, innovate
from radar_forge.tracking.measurement_models import (
    LinearMeasurement,
    RangeAzimuthElevationRangeRate,
)
from radar_forge.tracking.models import CartesianMotion

ALL = np.ones((2, 2), dtype=bool)


@pytest.mark.parametrize(("dim", "expected"), [(1, 6.635), (2, 9.210), (3, 11.345), (4, 13.277)])
def test_the_gate_threshold_matches_the_tabulated_quantiles(dim: int, expected: float) -> None:
    """Chi-square quantiles at 0.99, as printed in any table; abs 5e-4 for its three decimals."""
    assert chi_square_gate_threshold(0.99, dim) == pytest.approx(expected, abs=5e-4)


def test_the_gate_threshold_is_computed_once_per_setting() -> None:
    chi_square_gate_threshold(0.95, 3)
    hits = chi_square_gate_threshold.cache_info().hits
    chi_square_gate_threshold(0.95, 3)
    assert chi_square_gate_threshold.cache_info().hits == hits + 1


@pytest.mark.parametrize(("probability", "dim"), [(0.0, 1), (1.0, 1), (1.5, 1), (0.9, 0)])
def test_the_gate_threshold_rejects_an_impossible_setting(probability: float, dim: int) -> None:
    with pytest.raises(ValueError, match="probability"):
        chi_square_gate_threshold(probability, dim)


def test_the_gate_accepts_the_fraction_of_true_measurements_it_claims(
    rng: np.random.Generator,
) -> None:
    """A 0.99 gate keeps 99 % of correctly modelled measurements, through the batched path.

    20 000 measurements drawn from N(H x, H P Hᵀ + R) are scored against one track in one
    call; the acceptance count must sit in its exact Clopper-Pearson 99.9 % interval.
    """
    layout = CartesianMotion("xy").state_layout
    model = LinearMeasurement(layout, ("x_m", "y_m"))
    covariance = np.diag([4.0, 1.0, 9.0, 1.0])
    bank = TrackBank.empty(layout).append([[10.0, 1.0, -5.0, 0.0]], [covariance], 0.0, [1], [0])
    noise = np.array([[1.0, 0.5], [0.5, 2.0]])
    spread = model.measurement_matrix @ covariance @ model.measurement_matrix.T + noise
    n_trials = 20_000
    z = rng.multivariate_normal([10.0, -5.0], spread, n_trials)
    nis = innovate(bank, model, TaylorLinearisation(), z, noise).nis
    accepted = int(gate(nis, chi_square_gate_threshold(0.99, 2)).sum())
    low = beta.ppf(0.0005, accepted, n_trials - accepted + 1)
    high = beta.ppf(0.9995, accepted + 1, n_trials - accepted)
    assert low <= 0.99 <= high


@pytest.mark.benchmark
def test_a_bank_of_20_tracks_gates_4000_measurements_in_under_a_second(
    rng: np.random.Generator,
) -> None:
    """Every pair scored in one batched call: 80 000 innovations, each a 4 x 4 Cholesky.

    Timed over the fastest of three runs, so a busy machine's one slow run does not fail it.
    """
    layout = CartesianMotion().state_layout
    model = RangeAzimuthElevationRangeRate(layout, np.zeros(3))
    means = rng.uniform(-100.0, 100.0, (20, 6))
    means[:, [0, 2, 4]] = rng.uniform(2_000.0, 20_000.0, (20, 3))
    bank = TrackBank.empty(layout).append(
        means, np.full((20, 6, 6), np.diag([400.0, 25.0] * 3)), 0.0, np.arange(20), np.zeros(20)
    )
    z = model.predict(means[rng.integers(0, 20, 4_000)]) + rng.normal(0.0, 1.0, (4_000, 4))
    noise = np.diag([100.0, 1e-6, 1e-6, 1.0])
    elapsed_s = []
    for _ in range(3):
        # Three timings; the fastest is the one that measures the code, not the machine.
        start_s = time.perf_counter()
        innovation = innovate(bank, model, TaylorLinearisation(), z, noise)
        gate(innovation.nis, chi_square_gate_threshold(0.997, 4))
        elapsed_s.append(time.perf_counter() - start_s)
    assert innovation.nis.shape == (20, 4_000)
    assert min(elapsed_s) < 1.0


def test_gnn_returns_the_known_optimum_where_greedy_does_not() -> None:
    """On [[2, 1], [8, 2]] greedy takes (0, 1) then pays 8; GNN pays 2 + 2."""
    cost = np.array([[2.0, 1.0], [8.0, 2.0]])
    np.testing.assert_array_equal(global_nearest_neighbour(cost, ALL)[0], [0, 1])
    np.testing.assert_array_equal(greedy_nearest_neighbour(cost, ALL)[0], [1, 0])


@pytest.mark.parametrize("assign", [global_nearest_neighbour, greedy_nearest_neighbour])
def test_no_gated_out_pair_is_ever_assigned(assign: object) -> None:
    cost = np.array([[1.0, 50.0, 3.0], [50.0, 2.0, 1.0]])
    in_gate = np.array([[False, True, True], [False, False, False]])
    assigned, unassigned = assign(cost, in_gate)  # type: ignore[operator]
    np.testing.assert_array_equal(assigned, [2, -1])
    np.testing.assert_array_equal(unassigned, [0, 1])


@pytest.mark.parametrize("shape", [(0, 3), (2, 0), (0, 0)])
def test_an_empty_scan_or_bank_assigns_nothing(shape: tuple[int, int]) -> None:
    assigned, unassigned = global_nearest_neighbour(np.zeros(shape), np.ones(shape, dtype=bool))
    assert assigned.shape == (shape[0],) and np.all(assigned == -1)
    np.testing.assert_array_equal(unassigned, np.arange(shape[1]))


def test_gnn_makes_as_many_pairs_as_the_gate_allows_before_minimising_cost() -> None:
    """Pairing 0-1 alone costs 1; pairing both costs 20 + 30. GNN takes both."""
    cost = np.array([[20.0, 1.0], [np.inf, 30.0]])
    in_gate = np.isfinite(cost)
    np.testing.assert_array_equal(global_nearest_neighbour(cost, in_gate)[0], [0, 1])


def test_a_finite_non_assignment_cost_leaves_a_dear_pair_unmade() -> None:
    cost = np.array([[20.0, 1.0], [np.inf, 30.0]])
    assigned, _ = global_nearest_neighbour(cost, np.isfinite(cost), non_assignment_cost=10.0)
    np.testing.assert_array_equal(assigned, [1, -1])


def test_gnn_takes_negative_costs_such_as_a_log_likelihood_ratio() -> None:
    cost = np.array([[-5.0, -1.0], [-4.0, -6.0]])
    np.testing.assert_array_equal(global_nearest_neighbour(cost, ALL)[0], [0, 1])


def test_greedy_breaks_a_tie_row_by_row() -> None:
    np.testing.assert_array_equal(greedy_nearest_neighbour(np.ones((2, 2)), ALL)[0], [0, 1])


def test_each_track_and_measurement_is_used_at_most_once(rng: np.random.Generator) -> None:
    cost = rng.uniform(0.0, 10.0, (6, 4))
    for associator in (GlobalNearestNeighbour(), GreedyNearestNeighbour()):
        assigned, unassigned = associator.assign(cost, cost < 7.0)
        taken = assigned[assigned >= 0]
        assert len(set(taken.tolist())) == len(taken) and set(unassigned) & set(taken) == set()


def test_mismatched_cost_and_gate_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        global_nearest_neighbour(np.zeros((2, 2)), np.ones((2, 3), dtype=bool))


def test_the_likelihood_ratio_is_the_negative_log_likelihood_plus_the_log_density() -> None:
    layout = CartesianMotion("x").state_layout
    model = LinearMeasurement(layout, ("x_m",))
    bank = TrackBank.empty(layout).append([[0.0, 0.0]], [np.eye(2)], 0.0, [1], [0])
    innovation = innovate(bank, model, TaylorLinearisation(), [[1.0], [3.0]], [[1.0]])
    np.testing.assert_array_equal(association_cost(innovation, "nis"), innovation.nis)
    ratio = association_cost(innovation, "negative_log_likelihood_ratio", 1e-3)
    # rtol 1e-14: one addition.
    np.testing.assert_allclose(ratio, -innovation.log_likelihood + np.log(1e-3), rtol=1e-14)
    with pytest.raises(ValueError, match="clutter_density"):
        association_cost(innovation, "negative_log_likelihood_ratio")
