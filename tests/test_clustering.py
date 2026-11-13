"""Tests for radar_forge.clustering.

Every expected label and mode is worked by hand in its test's docstring, or is the output of
the grid labelling the point method must reduce to.
"""

import numpy as np
import pytest

from radar_forge.clustering import cluster_points, mean_shift
from radar_forge.detection import label_connected_cells


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261113)


# DBSCAN


@pytest.mark.parametrize(("connectivity", "epsilon"), [(1, 1.0), (2, 1.5)])
@pytest.mark.parametrize("wrap_axes", [(), (0,), (0, 1)])
def test_points_one_cell_apart_cluster_as_the_connected_cells_do(
    connectivity, epsilon, wrap_axes, rng
):
    """On cell indices a radius of 1 reaches the face neighbours and 1.5 the corners too."""
    mask = rng.random((12, 20)) < 0.3
    labels, _ = label_connected_cells(mask, connectivity=connectivity, wrap_axes=wrap_axes)
    period = [n if axis in wrap_axes else np.nan for axis, n in enumerate(mask.shape)]
    found = cluster_points(np.argwhere(mask), epsilon, period=period)
    np.testing.assert_array_equal(found, labels[mask] - 1)


def test_sparse_points_are_noise_and_dense_ones_cluster():
    """By hand, min_samples 3, radius 1: two triples cluster, the lone point is noise."""
    points = [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [9.0, 9.0], [5.0, 5.0], [5.5, 5.0], [5.0, 5.5]]
    np.testing.assert_array_equal(
        cluster_points(points, 1.0, min_samples=3), [0, 0, 0, -1, 1, 1, 1]
    )
    np.testing.assert_array_equal(cluster_points(points, 1.0), [0, 0, 0, 1, 2, 2, 2])


def test_a_border_point_joins_its_nearest_core_point():
    """By hand, min_samples 4, radius 1: four cores at 0 to 0.6, four at 2.5 to 3.1, one between.

    The point between reaches only 0.6 and 2.5, three points with itself, so it is a border
    point of both clusters: at 1.58 it is 0.92 from 2.5, at 1.52 it is 0.92 from 0.6.
    Reversing the points renumbers the clusters and moves nothing.
    """
    cores = [0.0, 0.2, 0.4, 0.6, 2.5, 2.7, 2.9, 3.1]
    points = np.array([*cores, 1.58])[:, None]
    labels = cluster_points(points, 1.0, min_samples=4)
    np.testing.assert_array_equal(labels, [0, 0, 0, 0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(
        cluster_points(points[::-1], 1.0, min_samples=4), 1 - labels[::-1]
    )
    points[-1] = 1.52
    np.testing.assert_array_equal(
        cluster_points(points, 1.0, min_samples=4), [0, 0, 0, 0, 1, 1, 1, 1, 0]
    )


def test_points_cluster_across_a_period_and_by_each_columns_radius():
    """Rates -49 and 49 m/s on a 100 m/s circle are 2 m/s apart; range does not wrap."""
    points = [[1000.0, -49.0], [1010.0, 49.0], [-1000.0, 49.0]]
    np.testing.assert_array_equal(cluster_points(points, [15.0, 3.0]), [0, 1, 2])
    period = [np.nan, 100.0]
    np.testing.assert_array_equal(cluster_points(points, [15.0, 3.0], period=period), [0, 0, 1])
    # (10 / 15)^2 + (2 / 3)^2 = 8 / 9 is inside the unit ball; (10 / 15)^2 + (2 / 2.5)^2 is not.
    np.testing.assert_array_equal(cluster_points(points, [15.0, 2.5], period=period), [0, 1, 2])


def test_clustering_no_points_returns_no_labels():
    assert cluster_points(np.empty((0, 2)), 1.0).shape == (0,)


# Mean shift


def test_each_point_climbs_to_the_mean_of_its_own_window():
    """By hand, bandwidth 1.5: 0, 1, 2 rest at 1 and 10, 11 at 10.5.

    From 0 the window holds 0 and 1, mean 0.5; from 0.5 it holds 0, 1, 2, mean 1, where it
    rests. From 10 and from 11 the window holds both, mean 10.5.
    """
    labels, modes = mean_shift(np.array([0.0, 1.0, 2.0, 10.0, 11.0])[:, None], 1.5)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1])
    np.testing.assert_allclose(modes, [[1.0], [10.5]], rtol=1e-14)


def test_weights_move_a_mode_to_the_weighted_mean():
    """Points 10 and 11 at weights 1 and 3: (10 + 33) / 4 = 10.75."""
    labels, modes = mean_shift([[10.0], [11.0]], 1.5, weights=[1.0, 3.0])
    np.testing.assert_array_equal(labels, [0, 0])
    np.testing.assert_allclose(modes, [[10.75]], rtol=1e-14)


def test_two_peaks_that_touch_are_two_modes():
    """Cells 0 to 6 at powers 1, 8, 1, 1, 1, 8, 1 touch; at bandwidth 1 they are two modes.

    At rest a window holds a peak and its two neighbours: (0 + 8 + 2) / 10 = 1 and
    (4 + 40 + 6) / 10 = 5. The middle cell's window is balanced, 2, 3, 4 at equal power, so
    it rests at 3, within a bandwidth of neither mode's seeds, and is a third group.
    """
    points = np.arange(7.0)[:, None]
    weights = [1.0, 8.0, 1.0, 1.0, 1.0, 8.0, 1.0]
    labels, modes = mean_shift(points, 1.0, weights=weights)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 2, 2, 2])
    np.testing.assert_allclose(modes, [[1.0], [3.0], [5.0]], rtol=1e-14)
    assert cluster_points(points, 1.0).max() == 0


def test_a_mode_forms_across_a_period():
    """Rates 49 and -49 m/s on a 100 m/s circle, bandwidth 3: one mode at the wrap, 50."""
    labels, modes = mean_shift([[49.0], [-49.0], [0.0]], 3.0, period=100.0)
    np.testing.assert_array_equal(labels, [0, 0, 1])
    np.testing.assert_allclose(modes % 100.0, [[50.0], [0.0]], rtol=1e-14)
    assert mean_shift([[49.0], [-49.0], [0.0]], 3.0)[0].max() == 2


def test_each_column_has_its_own_bandwidth():
    """(10 m, 2 m/s) apart: inside a (15, 3) window, (10 / 15)^2 + (2 / 3)^2 < 1; not (15, 2.5)."""
    points = [[1000.0, 5.0], [1010.0, 7.0]]
    labels, modes = mean_shift(points, [15.0, 3.0])
    np.testing.assert_array_equal(labels, [0, 0])
    np.testing.assert_allclose(modes, [[1005.0, 6.0]], rtol=1e-14)
    np.testing.assert_array_equal(mean_shift(points, [15.0, 2.5])[0], [0, 1])


def test_points_of_no_weight_stay_where_they_are():
    labels, modes = mean_shift([[0.0], [0.5]], 1.0, weights=[0.0, 0.0])
    np.testing.assert_array_equal(labels, [0, 0])
    np.testing.assert_array_equal(modes, [[0.0]])


def test_mean_shift_of_no_points_returns_nothing():
    labels, modes = mean_shift(np.empty((0, 2)), 1.0)
    assert labels.shape == (0,)
    assert modes.shape == (0, 2)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: cluster_points(np.zeros(4), 1.0), r"finite \(n, d\) array"),
        (lambda: cluster_points([[0.0, np.nan]], 1.0), r"finite \(n, d\) array"),
        (lambda: cluster_points(np.zeros((4, 2)), [1.0, 1.0, 1.0]), "one value per column"),
        (lambda: cluster_points(np.zeros((4, 2)), 0.0), "epsilon must be finite and positive"),
        (lambda: cluster_points(np.zeros((4, 2)), 1.0, period=[0.0, 1.0]), "period must be"),
        (lambda: cluster_points(np.zeros((4, 2)), 1.0, min_samples=0), "at least 1"),
        (lambda: cluster_points(np.zeros((4, 2)), 1.0, min_samples=1.5), "must be an integer"),
        (lambda: mean_shift(np.zeros((4, 2)), -1.0), "radius must be finite and positive"),
        (lambda: mean_shift(np.zeros((4, 2)), 1.0, weights=np.ones(3)), "weights must be 4"),
        (lambda: mean_shift(np.zeros((4, 2)), 1.0, weights=-np.ones(4)), "non-negative"),
    ],
)
def test_rejects_an_invalid_argument_and_says_which(call, match):
    with pytest.raises(ValueError, match=match):
        call()
