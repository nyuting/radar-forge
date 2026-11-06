"""Tests for radar_forge.tracking.metrics: NEES, its chi-square interval, RMSE and GOSPA.

Ground truth is closed form: hand-computed errors, and tabulated chi-square quantiles.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.tracking.metrics import consistency_interval, gospa, nees, rmse, trajectory_gospa
from radar_forge.tracking.models import Coordinate, StateLayout


def test_nees_is_the_error_weighted_by_the_inverse_covariance() -> None:
    """e = (1, 2), P = diag(1, 4): 1/1 + 4/4 = 2; a correlated P is solved, not inverted."""
    assert nees([0.0, 0.0], np.diag([1.0, 4.0]), [1.0, 2.0]) == pytest.approx(2.0, rel=1e-15)
    covariance = np.array([[2.0, 1.0], [1.0, 2.0]])
    error = np.array([1.0, -1.0])
    # Along the eigenvector (1, -1), whose eigenvalue is 1: eᵀ P⁻¹ e = |e|² / 1 = 2.
    assert nees(np.zeros(2), covariance, error) == pytest.approx(2.0, rel=1e-14)


def test_nees_is_batched_over_runs_and_steps() -> None:
    mean = np.zeros((3, 4, 2))
    values = nees(mean, np.eye(2), np.ones((3, 4, 2)))
    np.testing.assert_array_equal(values, np.full((3, 4), 2.0))


def test_nees_takes_the_short_way_round_a_periodic_coordinate() -> None:
    layout = StateLayout((Coordinate("azimuth_rad", "rad", period=2 * np.pi),))
    value = nees([2 * np.pi - 0.1], np.eye(1), [0.1], state_layout=layout)
    assert value == pytest.approx(0.04, rel=1e-12)


def test_rmse_is_per_coordinate_over_the_chosen_axes() -> None:
    estimate = np.array([[[1.0, 0.0], [3.0, 0.0]], [[0.0, 2.0], [0.0, 2.0]]])
    np.testing.assert_allclose(
        rmse(estimate, np.zeros_like(estimate), axis=(0, 1)), [np.sqrt(2.5), np.sqrt(2.0)]
    )
    np.testing.assert_allclose(rmse(estimate[0], np.zeros((2, 2))), [np.sqrt(5.0), 0.0])


def test_the_consistency_interval_matches_the_tabulated_quantiles() -> None:
    """One chi2(1) draw at 95 %: [0.000982, 5.024], as printed in any table."""
    low, high = consistency_interval(1, 1, 0.95)
    assert low == pytest.approx(0.000982, abs=5e-7) and high == pytest.approx(5.024, abs=5e-4)


def test_the_consistency_interval_narrows_about_the_dimension_as_samples_grow() -> None:
    widths = [np.subtract(*consistency_interval(2, n)[::-1]) for n in (10, 100, 1000)]
    assert widths == sorted(widths, reverse=True)
    low, high = consistency_interval(2, 100_000)
    assert low < 2.0 < high and high - low < 0.05


@pytest.mark.parametrize(("dim", "n", "p"), [(0, 1, 0.9), (1, 0, 0.9), (1, 1, 1.0)])
def test_the_consistency_interval_rejects_an_impossible_setting(dim: int, n: int, p: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        consistency_interval(dim, n, p)


# --- GOSPA and trajectory GOSPA (spec 011 criterion A9) ---


def test_gospa_of_one_missed_truth_is_c_over_root_two() -> None:
    """A9: no estimate, one truth: (c^p / 2)^(1/p) = 300 / sqrt(2) m, all of it missed."""
    score = gospa(np.zeros((0, 2)), [[1e3, 2e3]], cutoff_m=300.0)
    assert score.distance_m == pytest.approx(300.0 / np.sqrt(2.0), rel=1e-15)
    assert (score.n_missed, score.n_false, score.missed_m, score.false_m) == (
        1,
        0,
        score.distance_m,
        0.0,
    )


def test_gospa_of_one_false_estimate_is_c_over_root_two() -> None:
    """A9: one estimate, no truth: the same 212.13 m, all of it false."""
    score = gospa([[1e3, 2e3]], np.zeros((0, 2)), cutoff_m=300.0)
    assert score.distance_m == pytest.approx(300.0 / np.sqrt(2.0), rel=1e-15)
    assert (score.n_missed, score.n_false) == (0, 1)


def test_gospa_of_one_localisation_error_is_that_error() -> None:
    """A9: a 3-4-5 error of 50 m inside the cutoff is the distance itself; a 400 m one is c."""
    near = gospa([[30.0, 40.0]], [[0.0, 0.0]], cutoff_m=300.0)
    assert near.distance_m == pytest.approx(50.0, rel=1e-15)
    assert near.localisation_m == pytest.approx(50.0, rel=1e-15)
    far = gospa([[240.0, 320.0]], [[0.0, 0.0]], cutoff_m=300.0)
    # Beyond the cutoff the pair is one missed truth and one false estimate: c^p / 2 each.
    assert far.distance_m == pytest.approx(300.0, rel=1e-15)
    assert (far.n_missed, far.n_false, far.assignment.tolist()) == (1, 1, [-1])


def test_gospa_assigns_each_truth_its_nearest_estimate_and_adds_the_parts() -> None:
    """Two truths, three estimates: the parts' squares add to the distance's square."""
    score = gospa([[0.0, 10.0], [1e3, 0.0], [9e3, 9e3]], [[1e3, 20.0], [0.0, 0.0]])
    assert score.assignment.tolist() == [1, 0]
    parts = score.localisation_m**2 + score.missed_m**2 + score.false_m**2
    assert score.distance_m**2 == pytest.approx(parts, rel=1e-12)
    assert score.localisation_m == pytest.approx(np.sqrt(20.0**2 + 10.0**2), rel=1e-12)


def test_trajectory_gospa_charges_one_switch_per_track_that_changes_partner() -> None:
    """Two tracks swap truths half way: two switches, gamma^p each, no localisation error."""
    truth = np.array([[[0.0, 0.0], [1e3, 0.0]]] * 4)
    estimate = truth.copy()
    estimate[2:] = estimate[2:, ::-1]
    score = trajectory_gospa(estimate, truth, cutoff_m=300.0)
    assert score.n_switches == pytest.approx(2.0, abs=1e-9)
    assert score.distance_m == pytest.approx(np.sqrt(2.0) * 300.0, rel=1e-9)


def test_trajectory_gospa_without_switches_is_the_root_sum_of_per_scan_gospa() -> None:
    """No switch, no miss: the LP's value is the per-scan localisation errors' root sum square."""
    truth = np.array([[[0.0, 0.0], [1e3, 0.0]]] * 3)
    estimate = truth + np.array([[3.0, 4.0], [0.0, 0.0]])
    score = trajectory_gospa(estimate, truth, cutoff_m=300.0)
    # rtol 1e-9: HiGHS solves the program to about that.
    assert score.distance_m == pytest.approx(np.sqrt(3 * 25.0), rel=1e-9)
    assert score.n_switches == pytest.approx(0.0, abs=1e-9)


def test_trajectory_gospa_without_estimates_scores_every_truth_missed() -> None:
    """No estimate at all: each present truth at each step is missed, c^p / 2 each."""
    truth = np.array([[[0.0, 0.0], [1e3, 0.0]]] * 3)
    no_estimate = np.zeros((3, 0, 2))
    for at_estimates in (None, np.zeros((3, 2, 0, 2))):
        score = trajectory_gospa(no_estimate, truth, truths_at_estimates=at_estimates)
        # rtol 1e-9: HiGHS solves the program to about that; 6 misses of 45 000 m^2.
        assert score.distance_m == pytest.approx(np.sqrt(6 * 300.0**2 / 2), rel=1e-9)
        assert score.missed_m == pytest.approx(score.distance_m, rel=1e-9)
        assert score.n_switches == 0.0
