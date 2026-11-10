r"""Gating and assignment: which measurement belongs to which track, for a whole scan at once.

A pair passes the gate when its normalised innovation squared (NIS) :math:`\nu^T S^{-1} \nu` is
at most :math:`\chi^2_m(P_G)`, which a consistent filter's true measurement passes a fraction
:math:`P_G` of the time [1] [2]. Each gated pair is costed by NIS, or by the negative
log-likelihood ratio against a clutter density λ, which compares pairs scored at different
dimensions [4]. Global nearest neighbour solves the scan's assignment with
``scipy.optimize.linear_sum_assignment`` [3]; greedy nearest neighbour takes the cheapest pair
first. Probabilistic association weights every gated measurement instead: PDA per track [7],
IPDA with a probability of existence [8], and JPDA over each cluster of tracks sharing
measurements, enumerated exactly [9], with JIPDA [10] and the coalescence fixes JPDA* [5] and
the set relabelling [6]; Murty's method ranks the k best assignments [11] [12], for MHT
and PMBM. Every function takes ``(n_tracks, n_meas)`` arrays.

References
----------
.. [1] Y. Bar-Shalom and X. R. Li, *Multitarget-Multisensor Tracking: Principles and
       Techniques*, YBS Publishing, 1995.
.. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
.. [3] D. F. Crouse, "On implementing 2D rectangular assignment algorithms," *IEEE Trans.
       Aerosp. Electron. Syst.*, vol. 52, no. 4, pp. 1679-1696, 2016.
.. [4] S. S. Blackman, "Multiple hypothesis tracking for multiple target tracking," *IEEE
       Aerosp. Electron. Syst. Mag.*, vol. 19, no. 1, pp. 5-18, 2004.
.. [5] H. A. P. Blom and E. A. Bloem, "Probabilistic data association avoiding track
       coalescence," *IEEE Trans. Autom. Control*, vol. 45, no. 2, pp. 247-259, 2000.
.. [6] L. Svensson, D. Svensson, M. Guerriero and P. Willett, "Set JPDA filter for
       multitarget tracking," *IEEE Trans. Signal Process.*, vol. 59, no. 10, pp. 4677-4691,
       2011.
.. [7] Y. Bar-Shalom, F. Daum and J. Huang, "The probabilistic data association filter,"
       *IEEE Control Syst. Mag.*, vol. 29, no. 6, pp. 82-100, 2009.
.. [8] D. Mušicki, R. Evans and S. Stanković, "Integrated probabilistic data association,"
       *IEEE Trans. Autom. Control*, vol. 39, no. 6, pp. 1237-1241, 1994.
.. [9] T. E. Fortmann, Y. Bar-Shalom and M. Scheffe, "Sonar tracking of multiple targets using
       joint probabilistic data association," *IEEE J. Oceanic Eng.*, vol. 8, no. 3,
       pp. 173-184, 1983.
.. [10] D. Mušicki and R. Evans, "Joint integrated probabilistic data association: JIPDA,"
        *IEEE Trans. Aerosp. Electron. Syst.*, vol. 40, no. 3, pp. 1093-1099, 2004.
.. [11] K. G. Murty, "An algorithm for ranking all the assignments in order of increasing
        cost," *Operations Research*, vol. 16, no. 3, pp. 682-687, 1968.
.. [12] I. J. Cox and M. L. Miller, "On finding ranked assignments with application to
        multitarget tracking and motion correspondence," *IEEE Trans. Aerosp. Electron.
        Syst.*, vol. 31, no. 1, pp. 486-489, 1995.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from functools import cache
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from scipy.special import logsumexp
from scipy.stats import chi2

from radar_forge.tracking.filters import Innovation

__all__ = [
    "AssociationWeights",
    "Associator",
    "GlobalNearestNeighbour",
    "GreedyNearestNeighbour",
    "JointProbabilisticDataAssociation",
    "association_cost",
    "chi_square_gate_threshold",
    "gate",
    "global_nearest_neighbour",
    "greedy_nearest_neighbour",
    "joint_probabilistic_data_association",
    "murty_k_best",
    "probabilistic_data_association",
]

Cost = Literal["nis", "negative_log_likelihood_ratio"]
Coalescence = Literal["none", "jpda_star", "set"]

Assignment = tuple[NDArray[np.intp], NDArray[np.intp]]


@cache
def chi_square_gate_threshold(probability: float, dim: int) -> float:
    r"""Return :math:`\chi^2_{dim}(P_G)`, the largest NIS the gate accepts; cached.

    References
    ----------
    Bar-Shalom and Li [1], §2.3.2; Bar-Shalom [2], §5.4.2.
    """
    if not 0.0 < probability < 1.0 or dim < 1:
        msg = f"need 0 < probability < 1 and dim >= 1; got {probability}, {dim}."
        raise ValueError(msg)
    return float(chi2.ppf(probability, dim))


def gate(nis: ArrayLike, threshold: float) -> NDArray[np.bool_]:
    """Return which pairs pass: NIS at most the threshold.

    References
    ----------
    Bar-Shalom and Li [1], §2.3.2 (the validation region).
    """
    return np.asarray(nis, dtype=np.float64) <= threshold


def association_cost(
    innovation: Innovation, cost: Cost, clutter_density: float | None = None
) -> NDArray[np.float64]:
    r"""Return each pair's cost: the NIS, or :math:`-\ln g + \ln \lambda`; smaller is better.

    NIS compares pairs of one dimension only, its expectation being the dimension. The ratio
    has no units, so it compares a range-only pair with a range-and-rate one.

    Parameters
    ----------
    clutter_density : float or None, optional
        λ, false measurements per scan per unit of the measurement space the pairs were scored
        in; required by the likelihood ratio.

    References
    ----------
    Blackman [4], the track score as a log-likelihood ratio.
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


def greedy_nearest_neighbour(cost: ArrayLike, in_gate: ArrayLike) -> Assignment:
    """Take the cheapest gated pair, then the cheapest of what is left; ties row by row.

    Returns the measurement each track takes (-1 for none) and the unassigned measurements.

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


def _augment(
    costs: NDArray[np.float64], mask: NDArray[np.bool_], non_assignment_cost: float
) -> NDArray[np.float64]:
    """Return the cost matrix with each row's private "none" column; gated-out pairs are inf."""
    n_rows, n_meas = costs.shape
    augmented = np.full((n_rows, n_meas + n_rows), np.inf)
    augmented[:, :n_meas] = np.where(mask, costs, np.inf)
    augmented[np.arange(n_rows), n_meas + np.arange(n_rows)] = non_assignment_cost
    return augmented


def global_nearest_neighbour(
    cost: ArrayLike, in_gate: ArrayLike, *, non_assignment_cost: float = np.inf
) -> Assignment:
    r"""Return the best set of pairs for the whole scan: global nearest neighbour (GNN).

    Each track gets a private "no measurement" column of cost ``non_assignment_cost``, so the
    problem always has a solution. With a finite cost, a pair is made only when cheaper than
    leaving the track unassigned. With the default, GNN makes as many pairs as the gate allows
    before minimising cost: the cost becomes :math:`n (c_{max} - c_{min}) + |c_{max}| + 1` over
    the gated costs, n the smaller side, which exceeds what one more pair can add to the real
    total. Below, greedy would pay 1 + 8; GNN pays 2 + 2.

    >>> global_nearest_neighbour([[2.0, 1.0], [8.0, 2.0]], [[True, True], [True, True]])[0]
    array([0, 1])

    References
    ----------
    Bar-Shalom and Li [1], §7.1.1; Crouse [3].
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
        rows, cols = linear_sum_assignment(_augment(costs, mask, non_assignment_cost))
        real = cols < n_meas
        assigned[rows[real]] = cols[real]
    taken = np.zeros(n_meas, dtype=bool)
    taken[assigned[assigned >= 0]] = True
    return assigned, np.flatnonzero(~taken)


def murty_k_best(
    cost: ArrayLike,
    k: int,
    *,
    in_gate: ArrayLike,
    non_assignment_cost: float,
    max_gap: float = np.inf,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    r"""Return the k cheapest assignments of a scan, cheapest first: Murty's ranking.

    Each row takes one gated column or its private "none" at the finite
    ``non_assignment_cost``, as in :func:`global_nearest_neighbour`, whose answer is the first.
    The pairs of a solution partition what is left: the i-th subproblem forbids its i-th pair
    and fixes the pairs before it, and the next solution is the cheapest subproblem's, each
    solved by ``linear_sum_assignment`` [3]. Returns ``(k', n_rows)`` measurements (-1 none)
    and their total costs; k' < k when fewer assignments exist, or fewer cost at most
    ``max_gap`` more than the first.

    >>> murty_k_best([[1.0, 2.0]], 3, in_gate=[[True, True]], non_assignment_cost=1.5)[0]
    array([[ 0],
           [-1],
           [ 1]])

    References
    ----------
    Murty [11], the partition of the assignment space; Cox and Miller [12].
    """
    costs, mask = _check(cost, in_gate)
    if not (math.isfinite(non_assignment_cost) and type(k) is int and k >= 1):
        msg = f"need a finite non_assignment_cost and int k >= 1; got {non_assignment_cost}, {k}."
        raise ValueError(msg)
    n_rows, n_meas = costs.shape
    found: list[tuple[float, NDArray[np.intp]]] = []
    queue: list[tuple[float, int, NDArray[np.intp], NDArray[np.float64], int]] = []
    counter = itertools.count()

    def push(matrix: NDArray[np.float64], fixed: int) -> None:
        """Solve one subproblem and queue its answer, unless it has none."""
        try:
            rows, cols = linear_sum_assignment(matrix)
        except ValueError:  # every choice of some row is barred: this part is empty
            return
        total = float(matrix[rows, cols].sum())
        heapq.heappush(queue, (total, next(counter), cols, matrix, fixed))

    push(_augment(costs, mask, non_assignment_cost), 0)
    # Murty's partition is sequential: each answer taken defines the next subproblems.
    while queue and len(found) < k:
        total, _, cols, matrix, fixed = heapq.heappop(queue)
        if found and total > found[0][0] + max_gap:
            break
        found.append((total, np.where(cols < n_meas, cols, -1).astype(np.intp)))
        if len(found) == k:
            break
        for row in range(fixed, n_rows):
            child = matrix.copy()
            child[row, cols[row]] = np.inf
            # Rows before this one keep their columns: every other entry of theirs is barred.
            kept, taken = np.arange(fixed, row), cols[fixed:row]
            child[kept] = np.inf
            child[:, taken] = np.inf
            child[kept, taken] = matrix[kept, taken]
            push(child, row)
    assignments = np.array([a for _, a in found], dtype=np.intp).reshape(len(found), n_rows)
    return assignments, np.array([c for c, _ in found], dtype=np.float64)


class Associator(Protocol):
    """An association rule: an assignment, and the probabilistic weights if it has them."""

    def assign(self, cost: ArrayLike, in_gate: ArrayLike) -> Assignment:
        """Return each track's measurement (-1 none) and the unassigned measurements."""
        ...

    def weights(
        self,
        log_likelihood: ArrayLike,
        in_gate: ArrayLike,
        *,
        clutter_intensity: ArrayLike | None,
        existence: ArrayLike | None = None,
        joint: bool = True,
    ) -> AssociationWeights | None:
        """Return :class:`AssociationWeights`, or None for a rule that only assigns."""
        ...


@dataclass(frozen=True)
class GreedyNearestNeighbour:
    """:func:`greedy_nearest_neighbour` as an :class:`Associator`."""

    def assign(self, cost: ArrayLike, in_gate: ArrayLike) -> Assignment:
        """Return :func:`greedy_nearest_neighbour` of the scan."""
        return greedy_nearest_neighbour(cost, in_gate)

    def weights(self, *args: object, **kwargs: object) -> None:
        """Return None: nearest neighbour only assigns."""
        return None


@dataclass(frozen=True)
class GlobalNearestNeighbour:
    """:func:`global_nearest_neighbour` as an :class:`Associator`; infinite cost maximises pairs."""

    non_assignment_cost: float = np.inf

    def assign(self, cost: ArrayLike, in_gate: ArrayLike) -> Assignment:
        """Return :func:`global_nearest_neighbour` of the scan."""
        return global_nearest_neighbour(cost, in_gate, non_assignment_cost=self.non_assignment_cost)

    def weights(self, *args: object, **kwargs: object) -> None:
        """Return None: nearest neighbour only assigns."""
        return None


@dataclass(frozen=True, eq=False)
class AssociationWeights:
    """A probabilistic association of one scan, and the clusters it was solved in.

    ``weights`` ``(n_tracks, n_meas + 1)``: each row's probability that measurement j is its
    target's, given the target exists, the last column that none is; rows sum to one.
    ``existence`` ``(n_tracks,)`` is each target's posterior probability of existing.
    ``cluster_tracks``, ``cluster_measurements`` and ``cluster_events`` give each cluster's
    size and its count of feasible joint events.
    """

    weights: NDArray[np.float64]
    existence: NDArray[np.float64]
    cluster_tracks: NDArray[np.int64]
    cluster_measurements: NDArray[np.int64]
    cluster_events: NDArray[np.int64]


def _ratios(
    log_likelihood: ArrayLike,
    in_gate: ArrayLike,
    clutter_intensity: ArrayLike | None,
    detection_probability: float,
    gate_probability: float,
    existence: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Return :math:`P_D r_t g_{tj} / \lambda_j` (zero outside the gate), the miss terms and r."""
    log_g, mask = _check(log_likelihood, in_gate)
    clutter = np.asarray(np.nan if clutter_intensity is None else clutter_intensity, dtype=float)
    prior = np.ones(len(log_g)) if existence is None else np.asarray(existence, dtype=np.float64)
    if clutter.shape != (log_g.shape[1],) or not np.all(np.isfinite(clutter) & (clutter > 0)):
        msg = f"clutter_intensity must be positive and finite, one per measurement; got {clutter}."
        raise ValueError(msg)
    if not (0 < detection_probability <= 1 and 0 < gate_probability <= 1):
        msg = f"need 0 < P_D, P_G <= 1; got {detection_probability}, {gate_probability}."
        raise ValueError(msg)
    if prior.shape != (len(log_g),) or not np.all((prior > 0) & (prior <= 1)):
        msg = f"existence must lie in (0, 1], one per track; got {prior}."
        raise ValueError(msg)
    detect = detection_probability * prior
    ratio = np.where(mask, detect[:, None] * np.exp(np.where(mask, log_g, 0.0)) / clutter, 0.0)
    return ratio, 1.0 - detect * gate_probability, prior


def _posterior(
    beta: NDArray[np.float64],
    prior: NDArray[np.float64],
    detection_probability: float,
    gate_probability: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Return the weights given existence, and the posterior existence, from event marginals.

    With no measurement, the target exists with probability :math:`(1 - P_D P_G) r / (1 - P_D
    P_G r)`; with one, surely. So :math:`r^+ = \sum_j \beta_j + \beta_0 (1 - P_D P_G) r / (1 -
    P_D P_G r)`, and r = 1 leaves the JPDA weights as they are.
    """
    seen = detection_probability * gate_probability
    # r = 1 is plain JPDA and leaves the weights alone, even where P_D P_G = 1 makes 0/0; a
    # target sure to be seen and not seen does not exist.
    sure = prior == 1.0
    room = np.where(sure, 1.0, 1 - seen * prior)
    unseen = np.where(sure, beta[:, -1], beta[:, -1] * (1 - seen) * prior / room)
    unseen = np.where(sure | (room > 0), unseen, 0.0)
    existence = beta[:, :-1].sum(axis=1) + unseen
    weights = np.column_stack([beta[:, :-1], unseen]) / existence[:, None]
    return weights, existence


def probabilistic_data_association(
    log_likelihood: ArrayLike,
    in_gate: ArrayLike,
    *,
    clutter_intensity: ArrayLike | None,
    detection_probability: float,
    gate_probability: float,
    existence: ArrayLike | None = None,
) -> AssociationWeights:
    r"""Return each track's PDA weights, every track alone, as if no other track existed.

    :math:`\beta_j = L_j / (b + \sum_k L_k)`, :math:`\beta_0 = b / (b + \sum_k L_k)`, with
    :math:`L_j = P_D r g_j / \lambda_j` and :math:`b = 1 - P_D P_G r`; with ``existence`` r
    this is IPDA, and r is updated too. ``log_likelihood`` is :math:`\ln g`, ``(n_tracks,
    n_meas)``; ``clutter_intensity`` λ per measurement, in its units.

    References
    ----------
    Bar-Shalom, Daum and Huang [7] (the PDAF); Mušicki, Evans and Stanković [8] (IPDA).
    """
    ratio, miss, prior = _ratios(
        log_likelihood,
        in_gate,
        clutter_intensity,
        detection_probability,
        gate_probability,
        existence,
    )
    total = miss + ratio.sum(axis=1)
    beta = np.column_stack([ratio, miss]) / total[:, None]
    weights, posterior = _posterior(beta, prior, detection_probability, gate_probability)
    sizes = np.ones(len(ratio), dtype=np.int64)
    gated = np.count_nonzero(ratio > 0, axis=1).astype(np.int64)
    return AssociationWeights(weights, posterior, sizes, gated, gated + 1)


def _clusters(mask: NDArray[np.bool_]) -> tuple[int, NDArray[np.intp]]:
    """Return the number of clusters and each track's and measurement's cluster label."""
    n_tracks, n_meas = mask.shape
    rows, cols = np.nonzero(mask)
    size = n_tracks + n_meas
    adjacency = coo_array((np.ones(rows.size), (rows, n_tracks + cols)), shape=(size, size))
    count, labels = connected_components(adjacency, directed=False)
    return int(count), np.asarray(labels, dtype=np.intp)


def _events(gate: NDArray[np.bool_], limit: int) -> NDArray[np.intp]:
    """Return every feasible joint event of one cluster, ``(n_events, n_tracks)``, -1 for none.

    A track takes at most one measurement of its gate and a measurement at most one track.
    """
    events = np.zeros((1, 0), dtype=np.intp)
    # A loop over the cluster's tracks: each extends every partial event at once.
    for row in gate:
        options = np.concatenate([[-1], np.flatnonzero(row)])
        used = (events[:, None, :] == options[None, :, None]).any(axis=2) & (options >= 0)
        keep_event, keep_option = np.nonzero(~used)
        events = np.column_stack([events[keep_event], options[keep_option]])
        if len(events) > limit:
            msg = (
                f"a cluster of {gate.shape[0]} tracks and {gate.shape[1]} measurements has "
                f"more than max_events_per_cluster = {limit} feasible events."
            )
            raise ValueError(msg)
    return events


def _coalescence(
    events: NDArray[np.intp], log_probability: NDArray[np.float64], coalescence: Coalescence
) -> NDArray[np.float64]:
    """Return the event probabilities, pruned (JPDA*) or relabelled (set) within each group.

    A group is every event detecting the same tracks with the same measurements: its members
    differ only in which detected track took which measurement.
    """
    probability: NDArray[np.float64] = np.exp(log_probability - logsumexp(log_probability))
    if coalescence == "none" or events.shape[1] < 2:
        return probability
    detected = events >= 0
    measurements = np.sort(np.where(detected, events, -1), axis=1)
    _, group = np.unique(np.column_stack([detected, measurements]), axis=0, return_inverse=True)
    group = group.reshape(-1)
    order = np.lexsort((-probability, group))
    first = np.ones(len(order), dtype=bool)
    first[1:] = group[order][1:] != group[order][:-1]
    best = np.zeros(len(order), dtype=bool)
    best[order[first]] = True
    if coalescence == "jpda_star":
        kept: NDArray[np.float64] = np.where(best, probability, 0.0)
        normalised: NDArray[np.float64] = kept / kept.sum()
        return normalised
    totals = np.bincount(group, weights=probability)
    relabelled: NDArray[np.float64] = np.where(best, totals[group], 0.0)
    return relabelled


def joint_probabilistic_data_association(
    log_likelihood: ArrayLike,
    in_gate: ArrayLike,
    *,
    clutter_intensity: ArrayLike | None,
    detection_probability: float,
    gate_probability: float,
    existence: ArrayLike | None = None,
    coalescence: Coalescence = "none",
    max_events_per_cluster: int = 100_000,
) -> AssociationWeights:
    r"""Return JPDA weights by exact enumeration of every cluster's joint events.

    Tracks sharing a gated measurement are clustered. An event θ, each track taking at most
    one gated measurement and each measurement at most one track, has :math:`P(\theta) \propto
    \prod_{t\,\mathrm{seen}} P_D r_t g_{t\theta_t}/\lambda_{\theta_t} \prod_{t\,\mathrm{missed}}
    (1 - P_D P_G r_t)`, and :math:`\beta_{tj}` sums the events giving j to t. With
    ``existence`` r this is JIPDA and r is updated. ``"jpda_star"`` keeps, of the events that
    differ only by which detected track took which measurement, the likeliest [5];
    ``"set"`` gives that likeliest one the group's whole probability, the relabelling of [6]
    restricted to weights a track's own estimate can carry. A cluster with more than
    ``max_events_per_cluster`` events raises, naming its size: it is never truncated.

    References
    ----------
    Fortmann, Bar-Shalom and Scheffe [9]; Bar-Shalom, Daum and Huang [7] (JPDA); Blom and
    Bloem [5]; Svensson et al. [6]; Mušicki and Evans [10] (JIPDA).
    """
    if coalescence not in ("none", "jpda_star", "set"):
        msg = f"coalescence must be 'none', 'jpda_star' or 'set'; got {coalescence!r}."
        raise ValueError(msg)
    ratio, miss, prior = _ratios(
        log_likelihood,
        in_gate,
        clutter_intensity,
        detection_probability,
        gate_probability,
        existence,
    )
    n_tracks, n_meas = ratio.shape
    count, labels = _clusters(ratio > 0)
    beta = np.zeros((n_tracks, n_meas + 1))
    sizes = np.zeros((3, count), dtype=np.int64)
    # A loop over the clusters: each is enumerated alone, and gating keeps them few and small.
    for cluster in range(count):
        tracks = np.flatnonzero(labels[:n_tracks] == cluster)
        if not tracks.size:
            continue
        measurements = np.flatnonzero(labels[n_tracks:] == cluster)
        local = ratio[np.ix_(tracks, measurements)]
        events = _events(local > 0, max_events_per_cluster)
        seen = events >= 0
        # A last column of ones stands for "none", so a cluster with no measurement indexes too.
        none = np.where(seen, events, measurements.size)
        taken = np.column_stack([local, np.ones(len(tracks))])[np.arange(len(tracks)), none]
        # A miss term of zero (P_D = P_G = r = 1) is floored, so its events weigh nothing.
        log_miss = np.log(np.maximum(miss[tracks], 1e-300))
        log_term = np.where(seen, np.log(np.where(seen, taken, 1.0)), log_miss)
        probability = _coalescence(events, log_term.sum(axis=1), coalescence)
        columns = np.append(measurements, n_meas)[none]
        np.add.at(beta, (np.broadcast_to(tracks, events.shape), columns), probability[:, None])
        sizes[:, cluster] = tracks.size, measurements.size, len(events)
    weights, posterior = _posterior(beta, prior, detection_probability, gate_probability)
    present = sizes[0] > 0
    return AssociationWeights(weights, posterior, *sizes[:, present])


@dataclass(frozen=True)
class JointProbabilisticDataAssociation:
    """:func:`joint_probabilistic_data_association` as an :class:`Associator` that weights.

    ``assign`` is :func:`global_nearest_neighbour`, for a caller that needs one measurement per
    track; ``weights`` with ``joint=False`` is :func:`probabilistic_data_association`, every
    track alone, which the tracker uses for tentative tracks so their wide gates never join a
    cluster.
    """

    detection_probability: float
    gate_probability: float
    coalescence: Coalescence = "none"
    max_events_per_cluster: int = 100_000

    def assign(self, cost: ArrayLike, in_gate: ArrayLike) -> Assignment:
        """Return :func:`global_nearest_neighbour` of the scan."""
        return global_nearest_neighbour(cost, in_gate)

    def weights(
        self,
        log_likelihood: ArrayLike,
        in_gate: ArrayLike,
        *,
        clutter_intensity: ArrayLike | None,
        existence: ArrayLike | None = None,
        joint: bool = True,
    ) -> AssociationWeights | None:
        """Return the scan's association weights, joint or per track."""
        if not joint:
            return probabilistic_data_association(
                log_likelihood,
                in_gate,
                clutter_intensity=clutter_intensity,
                detection_probability=self.detection_probability,
                gate_probability=self.gate_probability,
                existence=existence,
            )
        return joint_probabilistic_data_association(
            log_likelihood,
            in_gate,
            clutter_intensity=clutter_intensity,
            detection_probability=self.detection_probability,
            gate_probability=self.gate_probability,
            existence=existence,
            coalescence=self.coalescence,
            max_events_per_cluster=self.max_events_per_cluster,
        )
