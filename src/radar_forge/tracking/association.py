r"""Gating and assignment: which measurement belongs to which track, for a whole scan at once.

Association pairs each track with at most one measurement of the scan, in three batched steps:

1. **Gate.** A pair passes when its NIS, :math:`d^2 = \nu^T S^{-1} \nu`, is at most the
   chi-square quantile :math:`\chi^2_m(P_G)`: a consistent filter's true measurement passes a
   fraction :math:`P_G` of the time [1]_ [2]_. :func:`gate` turns an ``(n_tracks, n_meas)``
   NIS matrix into a mask; :func:`chi_square_gate_threshold` is cached.
2. **Cost.** :func:`association_cost` scores each pair: ``"nis"``, or
   ``"negative_log_likelihood_ratio"``, :math:`-\ln(g / \lambda)` with :math:`g =
   \mathcal{N}(\nu; 0, S)` and λ the clutter density, which compares pairs scored at
   different measurement dimensions fairly [4]_.
3. **Assign.** :func:`global_nearest_neighbour` (GNN) solves the assignment for the whole scan
   with ``scipy.optimize.linear_sum_assignment`` [3]_; :func:`greedy_nearest_neighbour` takes
   the cheapest pair, then the cheapest of the rest. :class:`GlobalNearestNeighbour` and
   :class:`GreedyNearestNeighbour` hold their settings for :class:`~radar_forge.tracking.
   tracker.Tracker`.

JPDA association weights and Murty's k-best assignment (``murty_k_best``) are later additions
to this module, with scenario 011 (D28, D29).

References
----------
.. [1] Y. Bar-Shalom and X. R. Li, *Multitarget-Multisensor Tracking: Principles and
       Techniques*, YBS Publishing, 1995, §2.3.2 (the validation region) and §7.1.1 (data
       association as an assignment problem).
.. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001, §5.4.2 (the NIS is chi-square distributed).
.. [3] D. F. Crouse, "On implementing 2D rectangular assignment algorithms," *IEEE Trans.
       Aerosp. Electron. Syst.*, vol. 52, no. 4, pp. 1679-1696, 2016.
.. [4] S. S. Blackman, "Multiple hypothesis tracking for multiple target tracking," *IEEE
       Aerosp. Electron. Syst. Mag.*, vol. 19, no. 1, pp. 5-18, 2004 (the track score as a
       log-likelihood ratio against a false-target density).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from radar_forge.tracking.filters import Innovation

__all__ = [
    "Associator",
    "GlobalNearestNeighbour",
    "GreedyNearestNeighbour",
    "association_cost",
    "chi_square_gate_threshold",
    "gate",
    "global_nearest_neighbour",
    "greedy_nearest_neighbour",
]

Cost = Literal["nis", "negative_log_likelihood_ratio"]


@cache
def chi_square_gate_threshold(probability: float, dim: int) -> float:
    r"""Return the largest NIS a gate of probability ``probability`` accepts.

    Parameters
    ----------
    probability : float
        :math:`P_G`, strictly between 0 and 1.
    dim : int
        Measurement dimension: the chi-square degrees of freedom.

    Returns
    -------
    float
        :math:`\chi^2_{\mathrm{dim}}(P_G)`. Cached, so each pair of arguments costs one
        quantile evaluation per process.

    Raises
    ------
    ValueError
        If ``probability`` is outside (0, 1) or ``dim`` is below 1.

    References
    ----------
    .. [1] Bar-Shalom and Li (1995), §2.3.2; see the module References.

    Examples
    --------
    >>> round(chi_square_gate_threshold(0.99, 2), 3)
    9.21
    """
    if not 0.0 < probability < 1.0 or dim < 1:
        msg = f"need 0 < probability < 1 and dim >= 1; got {probability}, {dim}."
        raise ValueError(msg)
    return float(chi2.ppf(probability, dim))


def gate(nis: ArrayLike, threshold: float) -> NDArray[np.bool_]:
    """Return which pairs pass the gate: NIS at most the threshold.

    Parameters
    ----------
    nis : array_like
        NIS of each pair, shape ``(n_tracks, n_meas)``.
    threshold : float
        From :func:`chi_square_gate_threshold`.

    Returns
    -------
    numpy.ndarray
        Boolean mask, shape ``(n_tracks, n_meas)``.

    Examples
    --------
    >>> gate([[1.0, 12.0]], 9.21)
    array([[ True, False]])
    """
    return np.asarray(nis, dtype=np.float64) <= threshold


def association_cost(
    innovation: Innovation, cost: Cost, clutter_density: float | None = None
) -> NDArray[np.float64]:
    r"""Return the cost of every pair; smaller is better.

    ``"nis"`` is :math:`d^2`. Its expectation is the measurement dimension, so it compares
    pairs of one dimension only. ``"negative_log_likelihood_ratio"`` is :math:`-\ln g + \ln
    \lambda`: how much less likely the measurement is to be this track's than a false alarm
    [1]_. The ratio has no units, so it compares a range-only pair with a range-and-rate one.

    Parameters
    ----------
    innovation : Innovation
        The scored pairs, from :func:`~radar_forge.tracking.filters.innovate`.
    cost : {"nis", "negative_log_likelihood_ratio"}
        Which cost.
    clutter_density : float or None, optional
        λ: false measurements per scan per unit of the measurement space the pairs were
        scored in. Needed by the likelihood ratio, positive and finite.

    Returns
    -------
    numpy.ndarray
        Costs, shape ``(n_tracks, n_meas)``.

    Raises
    ------
    ValueError
        If ``cost`` is unknown, or the likelihood ratio has no valid density.

    References
    ----------
    .. [1] Blackman (2004); see the module References.
    """
    if cost == "nis":
        return innovation.nis
    if cost != "negative_log_likelihood_ratio" or not (
        clutter_density is not None and np.isfinite(clutter_density) and clutter_density > 0
    ):
        msg = (
            f"cost must be 'nis', or 'negative_log_likelihood_ratio' with a positive finite "
            f"clutter_density; got {cost!r}, {clutter_density}."
        )
        raise ValueError(msg)
    ratio: NDArray[np.float64] = -innovation.log_likelihood + np.log(clutter_density)
    return ratio


def _check(cost: ArrayLike, in_gate: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return the cost and gate as arrays of one 2-D shape, or raise."""
    costs, mask = np.asarray(cost, dtype=np.float64), np.asarray(in_gate, dtype=bool)
    if costs.ndim != 2 or costs.shape != mask.shape:
        msg = f"cost and gate must share one 2-D shape; got {costs.shape}, {mask.shape}."
        raise ValueError(msg)
    return costs, mask


def greedy_nearest_neighbour(
    cost: ArrayLike, in_gate: ArrayLike
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Assign greedily: the cheapest gated pair, then the cheapest of what is left.

    Equal costs are taken row by row, then left to right, so the result does not depend on
    the sort. Simpler than GNN, and it can miss the best total: on ``[[2, 1], [8, 2]]`` it
    takes (0, 1), then pays 8; GNN pays 2 + 2.

    Parameters
    ----------
    cost : array_like
        Shape ``(n_tracks, n_meas)``; smaller is better.
    in_gate : array_like of bool
        Which pairs passed the gate and so may be assigned, same shape.

    Returns
    -------
    assigned : numpy.ndarray
        The measurement each track takes, ``(n_tracks,)``, -1 for none.
    unassigned_measurements : numpy.ndarray
        The measurements no track took, in increasing order.

    Raises
    ------
    ValueError
        If the shapes differ or are not 2-D.

    Examples
    --------
    >>> greedy_nearest_neighbour([[2.0, 1.0], [8.0, 2.0]], [[True, True], [True, True]])[0]
    array([1, 0])
    """
    costs, mask = _check(cost, in_gate)
    assigned = np.full(costs.shape[0], -1, dtype=np.intp)
    taken = np.zeros(costs.shape[1], dtype=bool)
    rows, cols = np.nonzero(mask)
    order = np.lexsort((cols, rows, costs[rows, cols]))
    # Greedy choice is sequential by definition: whether a pair can be taken depends on every
    # pair taken before it. The loop runs over the gated pairs only.
    for row, col in zip(rows[order], cols[order], strict=True):
        if assigned[row] < 0 and not taken[col]:
            assigned[row], taken[col] = col, True
    return assigned, np.flatnonzero(~taken)


def global_nearest_neighbour(
    cost: ArrayLike, in_gate: ArrayLike, *, non_assignment_cost: float = np.inf
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    r"""Assign by global nearest neighbour: the best set of pairs for the whole scan.

    Each track gets a private "no measurement" column of cost ``non_assignment_cost``, and the
    gated-out pairs are forbidden, so the problem always has a solution [1]_ [2]_. With a
    finite non-assignment cost, a pair is made only when it is cheaper than leaving the track
    unassigned. With the default, infinite, GNN first makes as many pairs as the gate allows
    and only then minimises the cost: the non-assignment cost is replaced by
    :math:`D = n (c_{max} - c_{min}) + |c_{max}| + 1` over the gated costs, with n the
    smaller side. One more pair changes the total real cost by less than D, so a solution
    with more pairs is always cheaper.

    Parameters
    ----------
    cost : array_like
        Shape ``(n_tracks, n_meas)``; smaller is better, and may be negative.
    in_gate : array_like of bool
        Which pairs passed the gate and so may be assigned, same shape.
    non_assignment_cost : float, default inf
        What leaving one track unassigned costs.

    Returns
    -------
    assigned : numpy.ndarray
        The measurement each track takes, ``(n_tracks,)``, -1 for none.
    unassigned_measurements : numpy.ndarray
        The measurements no track took, in increasing order.

    Raises
    ------
    ValueError
        If the shapes differ or are not 2-D, or ``non_assignment_cost`` is NaN.

    References
    ----------
    .. [1] Bar-Shalom and Li (1995), §7.1.1; Crouse (2016). See the module References.
    .. [2] Stone Soup's ``GNNWith2DAssignment`` prices a miss the same way, finitely.

    Examples
    --------
    Greedy would pay 1 + 8; GNN pays 2 + 2:

    >>> global_nearest_neighbour([[2.0, 1.0], [8.0, 2.0]], [[True, True], [True, True]])[0]
    array([0, 1])
    """
    costs, mask = _check(cost, in_gate)
    if np.isnan(non_assignment_cost):
        msg = "non_assignment_cost must not be NaN."
        raise ValueError(msg)
    n_tracks, n_meas = costs.shape
    assigned = np.full(n_tracks, -1, dtype=np.intp)
    if mask.any():
        gated = costs[mask]
        if np.isinf(non_assignment_cost):
            span = gated.max() - gated.min()
            non_assignment_cost = min(n_tracks, n_meas) * span + abs(gated.max()) + 1.0
        augmented = np.full((n_tracks, n_meas + n_tracks), np.inf)
        augmented[:, :n_meas] = np.where(mask, costs, np.inf)
        augmented[np.arange(n_tracks), n_meas + np.arange(n_tracks)] = non_assignment_cost
        rows, cols = linear_sum_assignment(augmented)
        real = cols < n_meas
        assigned[rows[real]] = cols[real]
    taken = np.zeros(n_meas, dtype=bool)
    taken[assigned[assigned >= 0]] = True
    return assigned, np.flatnonzero(~taken)


class Associator(Protocol):
    """What :class:`~radar_forge.tracking.tracker.Tracker` asks of an assignment rule."""

    def assign(
        self, cost: ArrayLike, in_gate: ArrayLike
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Return the measurement each track takes (-1 for none) and the unassigned ones."""
        ...


@dataclass(frozen=True)
class GreedyNearestNeighbour:
    """The greedy rule, :func:`greedy_nearest_neighbour`, as an :class:`Associator`."""

    def assign(
        self, cost: ArrayLike, in_gate: ArrayLike
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Return :func:`greedy_nearest_neighbour` of the scan."""
        return greedy_nearest_neighbour(cost, in_gate)


@dataclass(frozen=True)
class GlobalNearestNeighbour:
    """Global nearest neighbour, :func:`global_nearest_neighbour`, as an :class:`Associator`.

    Parameters
    ----------
    non_assignment_cost : float, default inf
        What leaving one track unassigned costs; infinite maximises the number of pairs.
    """

    non_assignment_cost: float = np.inf

    def assign(
        self, cost: ArrayLike, in_gate: ArrayLike
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Return :func:`global_nearest_neighbour` of the scan."""
        return global_nearest_neighbour(cost, in_gate, non_assignment_cost=self.non_assignment_cost)
