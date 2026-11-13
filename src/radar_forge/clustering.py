r"""Clustering of points: detections, plots or flagged cells given by their coordinates.

Two methods, each with one length scale per column and a period on each circular column, so
range in metres, rate in m/s and a wrapped Doppler axis can share a point. DBSCAN
(:func:`cluster_points`) joins points that lie within a radius of each other and calls sparse
ones noise [1]_. Mean shift (:func:`mean_shift`) moves every point uphill to a mode of the
weighted density [2]_ [3]_ [4]_, so it separates two peaks that touch, which joining
neighbours cannot. Cells on a map are grouped by :func:`radar_forge.detection.cluster_detections`.

Distances are measured in units of the length scale, :math:`d^2 = \sum_k (\Delta_k / h_k)^2`,
with :math:`\Delta_k` taken the short way round a circular column.

References
----------
.. [1] M. Ester, H.-P. Kriegel, J. Sander and X. Xu, "A density-based algorithm for
       discovering clusters in large spatial databases with noise," in *Proc. KDD-96*,
       pp. 226-231, 1996. (DBSCAN.)
.. [2] K. Fukunaga and L. D. Hostetler, "The estimation of the gradient of a density
       function, with applications in pattern recognition," *IEEE Trans. Inf. Theory*,
       vol. 21, no. 1, pp. 32-40, 1975. (Mean shift.)
.. [3] Y. Cheng, "Mean shift, mode seeking, and clustering," *IEEE Trans. PAMI*, vol. 17,
       no. 8, pp. 790-799, 1995. (Weighted points.)
.. [4] D. Comaniciu and P. Meer, "Mean shift: a robust approach toward feature space
       analysis," *IEEE Trans. PAMI*, vol. 24, no. 5, pp. 603-619, 2002. (Eq. 17; theorem 1,
       convergence.)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse, spatial
from scipy.sparse import csgraph

__all__ = ["cluster_points", "mean_shift"]

# A seed that moves less than this fraction of a bandwidth has reached its mode.
_STILL = 1.0e-9


def cluster_points(
    points: ArrayLike,
    epsilon: ArrayLike,
    *,
    min_samples: int = 1,
    period: ArrayLike | None = None,
) -> NDArray[np.intp]:
    """Return a DBSCAN label per point: 0, 1, ... by first core point, and -1 for noise.

    A point is a core point when at least ``min_samples`` points, itself included, lie within
    one radius of it. Core points within a radius of each other share a cluster; any other
    point within a radius of a core point joins its nearest one's cluster, and the rest are
    noise. With ``min_samples=1`` every point is a core point and the clusters are
    single-linkage groups: on cell indices with ``epsilon=1`` they are the groups of
    :func:`radar_forge.detection.label_connected_cells`.

    Parameters
    ----------
    points : array_like, shape (n, d)
        One row per point, e.g. columns of range in metres and range rate in m/s.
    epsilon : array_like
        Radius, one per column in that column's unit, or a scalar for all.
    period : array_like, optional
        Span of each circular column, in its unit; ``nan`` for a column that does not wrap.

    References
    ----------
    Ester et al. [1]_, definitions 1-5.
    """
    scaled, box = _scaled(points, epsilon, period, "epsilon")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int | np.integer):
        msg = f"min_samples must be an integer, got {min_samples!r}."
        raise ValueError(msg)
    if min_samples < 1:
        msg = f"min_samples must be at least 1, got {min_samples!r}."
        raise ValueError(msg)
    n_points = scaled.shape[0]
    if n_points == 0:
        return np.empty(0, dtype=np.intp)
    tree = spatial.cKDTree(scaled, boxsize=box)
    near = tree.sparse_distance_matrix(tree, 1.0, output_type="coo_matrix")
    first, second, distance = near.row, near.col, near.data
    core = np.bincount(first, minlength=n_points) >= min_samples
    linked = core[first] & core[second]
    graph = sparse.coo_array(
        (np.ones(int(linked.sum()), dtype=bool), (first[linked], second[linked])),
        shape=(n_points, n_points),
    )
    _, component = csgraph.connected_components(graph, directed=False)
    labels = np.full(n_points, -1, dtype=np.intp)
    # Components are numbered by their first point, so the core ones keep that order.
    labels[core] = np.unique(component[core], return_inverse=True)[1]
    # A border point takes its nearest core point's cluster; a tie goes to the lower index.
    reach = np.flatnonzero(~core[first] & core[second])
    reach = reach[np.lexsort((second[reach], distance[reach], first[reach]))]
    border, at = np.unique(first[reach], return_index=True)
    labels[border] = labels[second[reach[at]]]
    return labels


def mean_shift(
    points: ArrayLike,
    radius: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    period: ArrayLike | None = None,
    n_iterations: int = 100,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    r"""Return ``(labels, modes)``: the mode each point climbs to under a flat kernel.

    Every point starts a seed. Each iteration moves a seed :math:`y` to the weighted mean of
    the points within one bandwidth of it,
    :math:`y \leftarrow \sum_{i \in S(y)} w_i x_i / \sum_{i \in S(y)} w_i`, which is a step up
    the density estimated with an Epanechnikov kernel; a seed at rest is the centroid of its
    own window. Seeds stop when none moves, or after ``n_iterations``. Seeds that come to
    rest within a bandwidth of each other are one mode, so two peaks closer than that are
    not separated. There is no noise label: every point has a mode.

    Parameters
    ----------
    points : array_like, shape (n, d)
        One row per point, e.g. the indices of the cells that crossed a threshold.
    radius : array_like
        Window radius (the bandwidth), one per column in that column's unit, or a scalar for all.
    weights : array_like, shape (n,), optional
        Non-negative weight of each point, e.g. its power; default equal.
    period : array_like, optional
        Span of each circular column, in its unit; ``nan`` for a column that does not wrap.

    Returns
    -------
    labels : numpy.ndarray of int, shape (n,)
        Mode of each point, 0, 1, ... in order of first point.
    modes : numpy.ndarray, shape (n_modes, d)
        Where each mode's heaviest point came to rest, in the units of ``points``.

    References
    ----------
    Comaniciu and Meer [4]_, eq. 17 and theorem 1; Cheng [3]_ (weights).
    """
    scaled, box = _scaled(points, radius, period, "radius")
    n_points, n_columns = scaled.shape
    weight = np.ones(n_points) if weights is None else np.asarray(weights, dtype=np.float64)
    if weight.shape != (n_points,) or not np.all(np.isfinite(weight) & (weight >= 0.0)):
        msg = f"weights must be {n_points} finite non-negative values, got shape {weight.shape}."
        raise ValueError(msg)
    scale = np.broadcast_to(np.asarray(radius, dtype=np.float64), (n_columns,))
    if n_points == 0:
        return np.empty(0, dtype=np.intp), np.empty((0, n_columns))
    wraps = np.zeros(n_columns, dtype=bool) if box is None else box > 0.0
    tree = spatial.cKDTree(scaled, boxsize=box)
    seeds = scaled.copy()
    # Sequential by nature: each step starts where the last one ended.
    for _ in range(n_iterations):
        near = spatial.cKDTree(seeds, boxsize=box).sparse_distance_matrix(
            tree, 1.0, output_type="coo_matrix"
        )
        seed, point = near.row, near.col
        gap = scaled[point] - seeds[seed]
        if box is not None:
            gap[:, wraps] = _short_way(gap[:, wraps], box[wraps])
        total = np.bincount(seed, weights=weight[point], minlength=n_points)
        # One sum per column, of which there are one or two. A window of no weight stays put.
        moment = np.stack(
            [np.bincount(seed, weights=weight[point] * g, minlength=n_points) for g in gap.T],
            axis=1,
        )
        shift = np.divide(
            moment, total[:, None], out=np.zeros_like(moment), where=total[:, None] > 0
        )
        seeds = _wrapped(seeds + shift, box)
        if np.abs(shift).max() < _STILL:
            break
    labels = cluster_points(
        seeds, 1.0, period=None if box is None else np.where(wraps, box, np.nan)
    )
    # Each mode is reported where its heaviest point rests; a tie goes to the first point.
    order = np.lexsort((np.arange(n_points), -weight, labels))
    n_members = np.bincount(labels)
    modes = seeds[order[np.cumsum(n_members) - n_members]] * scale
    return labels, modes


def _scaled(
    points: ArrayLike, scale: ArrayLike, period: ArrayLike | None, name: str
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Return the points in units of ``scale`` and the k-d tree's box, ``None`` if none wraps.

    In those units every column weighs the same and a neighbourhood is a unit ball. Circular
    columns are wrapped into ``[0, period)``; a box size of zero leaves a column unwrapped.
    """
    rows = np.asarray(points, dtype=np.float64)
    if rows.ndim != 2 or not np.all(np.isfinite(rows)):
        msg = f"points must be a finite (n, d) array, got shape {rows.shape}."
        raise ValueError(msg)
    n_columns = rows.shape[1]
    radius = np.asarray(scale, dtype=np.float64)
    span = np.asarray(np.nan if period is None else period, dtype=np.float64)
    if radius.ndim > 1 or span.ndim > 1 or not {radius.size, span.size} <= {1, n_columns}:
        msg = f"{name} and period must be scalars or one value per column ({n_columns})."
        raise ValueError(msg)
    if not np.all(np.isfinite(radius) & (radius > 0.0)):
        msg = f"{name} must be finite and positive, got {scale!r}."
        raise ValueError(msg)
    if not np.all(np.isnan(span) | (np.isfinite(span) & (span > 0.0))):
        msg = f"period must be positive, or nan on a column that does not wrap, got {period!r}."
        raise ValueError(msg)
    ratio = np.broadcast_to(span / radius, (n_columns,))
    if not np.isfinite(ratio).any():
        return rows / radius, None
    box = np.where(np.isfinite(ratio), ratio, 0.0)
    return _wrapped(rows / radius, box), box


def _wrapped(scaled: NDArray[np.float64], box: NDArray[np.float64] | None) -> NDArray[np.float64]:
    """Return ``scaled`` with each circular column taken into ``[0, box)``."""
    if box is None:
        return scaled
    wraps = box > 0.0
    inside = scaled.copy()
    inside[:, wraps] %= box[wraps]
    # A tiny negative coordinate can round up to exactly the period.
    inside[:, wraps] = np.where(inside[:, wraps] >= box[wraps], 0.0, inside[:, wraps])
    return inside


def _short_way(gap: NDArray[np.float64], box: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a difference on a circle of circumference ``box`` in ``[-box / 2, box / 2)``."""
    result: NDArray[np.float64] = (gap + box / 2.0) % box - box / 2.0
    return result
