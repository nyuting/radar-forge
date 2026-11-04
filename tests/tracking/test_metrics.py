"""Tests for radar_forge.tracking.metrics: NEES, its chi-square interval, and RMSE.

Ground truth is closed form: hand-computed errors, and tabulated chi-square quantiles.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.tracking.metrics import consistency_interval, nees, rmse
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
