r"""Track-oriented multiple hypothesis tracking (MHT): association deferred over trees of histories.

Each target is a tree whose leaves are its surviving association histories, rows of one
:class:`~radar_forge.tracking.filters.TrackBank`, each scored by the log-likelihood ratio of
target against false alarm [1]: :math:`\ln(P_D g / \lambda)` added on a hit,
:math:`\ln(1 - P_D P_G)` on a miss. A global hypothesis holds at most one leaf per tree and
uses no measurement twice; trees sharing a gated measurement or a hypothesis form a cluster,
whose ``k_best`` hypotheses each scan extends by Murty's ranked assignments [2] [3] and
N-scan pruning trims [4]. :class:`GlobalHypotheses`, :func:`extend`, :func:`collapse` and
:func:`split` are what MHT and PMBM (:mod:`~radar_forge.tracking.rfs`) share.
:class:`TrackOrientedMHT` is the associator a :class:`~radar_forge.tracking.tracker.Tracker`
asks for its :meth:`~TrackOrientedMHT.hypotheses`; the tracker updates the leaves and reports,
at each scan, the leaves of each cluster's best hypothesis, never rewritten later.

References
----------
.. [1] S. S. Blackman, "Multiple hypothesis tracking for multiple target tracking," *IEEE
       Aerosp. Electron. Syst. Mag.*, vol. 19, no. 1, pp. 5-18, 2004.
.. [2] K. G. Murty, "An algorithm for ranking all the assignments in order of increasing
       cost," *Operations Research*, vol. 16, no. 3, pp. 682-687, 1968.
.. [3] I. J. Cox and M. L. Miller, "On finding ranked assignments with application to
       multitarget tracking and motion correspondence," *IEEE Trans. Aerosp. Electron.
       Syst.*, vol. 31, no. 1, pp. 486-489, 1995.
.. [4] T. Kurien, "Issues in the design of practical multitarget tracking algorithms," in
       Y. Bar-Shalom (ed.), *Multitarget-Multisensor Tracking: Advanced Applications*, Artech
       House, 1990, pp. 43-83.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from radar_forge.tracking.association import Assignment, global_nearest_neighbour, murty_k_best
from radar_forge.tracking.filters import TrackBank, TrackStatus
from radar_forge.tracking.models import StateLayout

__all__ = [
    "CONFIRM_SCORE",
    "DELETE_DROP",
    "Branching",
    "Extension",
    "GlobalHypotheses",
    "Hypothesiser",
    "TrackOrientedMHT",
    "TrackTree",
    "collapse",
    "extend",
    "split",
]

# Wald's sequential test at alpha = 1e-4 (false track) and beta = 0.1 (lost target): confirm at
# ln((1 - beta)/alpha) = 9.1050, and delete a branch fallen -ln(beta/(1 - alpha)) = 2.3025.
CONFIRM_SCORE = math.log((1 - 0.1) / 1e-4)
DELETE_DROP = -math.log(0.1 / (1 - 1e-4))

# A new leaf's measurement code: -1 a miss, -2 an old leaf the scan did not cover.
_MISS, _CARRIED = -1, -2
_CONFIRMED = (TrackStatus.CONFIRMED, TrackStatus.COASTING)


@dataclass(frozen=True, eq=False)
class GlobalHypotheses:
    """Global hypotheses over leaves, held as pairs: hypothesis ``hypothesis[i]`` holds ``leaf[i]``.

    ``log_weight`` is each hypothesis's summed score (MHT) or log weight (PMBM), ``cluster``
    its cluster's label; every leaf lies in the hypotheses of one cluster. Clusters are
    independent, so the pairs grow with the leaves, not with leaves times hypotheses.
    """

    leaf: NDArray[np.intp]
    hypothesis: NDArray[np.intp]
    log_weight: NDArray[np.float64]
    cluster: NDArray[np.int64]

    @classmethod
    def empty(cls) -> GlobalHypotheses:
        """Return no hypotheses over no leaves."""
        nothing = np.zeros(0, dtype=np.intp)
        return cls(nothing, nothing, np.zeros(0), np.zeros(0, dtype=np.int64))

    def best(self) -> NDArray[np.intp]:
        """Return each cluster's likeliest hypothesis, the first on a tie, in cluster order."""
        order = np.lexsort((np.arange(self.cluster.size), -self.log_weight, self.cluster))
        first = np.ones(order.size, dtype=bool)
        first[1:] = self.cluster[order][1:] != self.cluster[order][:-1]
        return np.asarray(order[first], dtype=np.intp)

    def held(self, n_leaves: int, hypotheses: ArrayLike | None = None) -> NDArray[np.bool_]:
        """Return which leaves some hypothesis holds, of all or of ``hypotheses`` only."""
        pairs = self.leaf if hypotheses is None else self.leaf[np.isin(self.hypothesis, hypotheses)]
        result = np.zeros(n_leaves, dtype=bool)
        result[pairs] = True
        return result

    def leaf_cluster(self, n_leaves: int) -> NDArray[np.int64]:
        """Return each leaf's cluster label, -1 for a leaf no hypothesis holds."""
        result = np.full(n_leaves, -1, dtype=np.int64)
        result[self.leaf] = self.cluster[self.hypothesis]
        return result

    def sums(self, leaf_weight: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return each hypothesis's sum of ``leaf_weight`` over the leaves it holds."""
        total = np.bincount(
            self.hypothesis, weights=leaf_weight[self.leaf], minlength=self.cluster.size
        )
        return np.asarray(total, dtype=np.float64)

    def keep(self, hypotheses: NDArray[np.bool_]) -> GlobalHypotheses:
        """Return only the hypotheses marked, renumbered in order."""
        index = np.cumsum(hypotheses) - 1
        pairs = hypotheses[self.hypothesis]
        return GlobalHypotheses(
            self.leaf[pairs],
            index[self.hypothesis[pairs]],
            self.log_weight[hypotheses],
            self.cluster[hypotheses],
        )

    def drop(self, leaves: NDArray[np.bool_]) -> GlobalHypotheses:
        """Return the hypotheses without the leaves marked, and the rest renumbered in order."""
        index = np.cumsum(~leaves) - 1
        pairs = ~leaves[self.leaf]
        leaf = index[self.leaf[pairs]].astype(np.intp)
        return GlobalHypotheses(leaf, self.hypothesis[pairs], self.log_weight, self.cluster)

    def plant(self, n_leaves: int, n_new: int, log_weight: float) -> GlobalHypotheses:
        """Return these hypotheses with ``n_new`` leaves appended, each a cluster of its own."""
        new = np.arange(n_new, dtype=np.intp)
        start = int(self.cluster.max(initial=-1)) + 1
        return GlobalHypotheses(
            np.concatenate([self.leaf, n_leaves + new]),
            np.concatenate([self.hypothesis, self.cluster.size + new]),
            np.concatenate([self.log_weight, np.full(n_new, log_weight)]),
            np.concatenate([self.cluster, start + new.astype(np.int64)]),
        )


@dataclass(frozen=True, eq=False)
class Extension:
    """One scan's new leaves and global hypotheses, before pruning.

    New leaf i is old leaf ``parent[i]`` unchanged (``measurement`` -2: not covered), that
    leaf continued by ``measurement[i]`` (-1 a miss), or a new tree on ``measurement[i]``
    (``parent`` -1). ``n_parents`` and ``n_children`` count, per extended cluster, the prior
    hypotheses and the assignments Murty ranked from them.
    """

    parent: NDArray[np.intp]
    measurement: NDArray[np.intp]
    hypotheses: GlobalHypotheses
    n_parents: NDArray[np.int64]
    n_children: NDArray[np.int64]


def _top(weights: NDArray[np.float64], group: NDArray[np.int64], k: int) -> NDArray[np.bool_]:
    """Return which entries are among the k heaviest of their group, earlier first on a tie."""
    order = np.lexsort((np.arange(weights.size), -weights, group))
    start = np.searchsorted(group[order], group[order], side="left")
    keep = np.zeros(weights.size, dtype=bool)
    keep[order] = np.arange(weights.size) - start < k
    return keep


def _groups(labels: NDArray[np.intp], n_groups: int) -> list[NDArray[np.intp]]:
    """Return the indices holding each label 0 .. n_groups - 1, in order."""
    order = np.argsort(labels, kind="stable")
    return np.split(order, np.searchsorted(labels[order], np.arange(1, n_groups)))


def _cross(
    options: list[tuple[NDArray[np.intp], float]],
    more: list[tuple[NDArray[np.intp], float]],
    k: int,
) -> list[tuple[NDArray[np.intp], float]]:
    """Return the k heaviest unions of one hypothesis from each list; the k of all is exact."""
    pairs = [(np.concatenate([a, b]), wa + wb) for a, wa in options for b, wb in more]
    pairs.sort(key=lambda pair: -pair[1])
    return pairs[:k]


def extend(
    hypotheses: GlobalHypotheses,
    seen: NDArray[np.bool_],
    cost: NDArray[np.float64],
    in_gate: NDArray[np.bool_],
    *,
    non_assignment_cost: float,
    hit: NDArray[np.float64],
    miss: NDArray[np.float64],
    unassigned: NDArray[np.float64],
    born: NDArray[np.bool_],
    k_best: int,
    per_parent: Literal["k_best", "weighted"] = "k_best",
    linking: NDArray[np.bool_] | None = None,
    min_share: float = 0.0,
) -> Extension:
    r"""Extend every cluster the scan touches by its k best assignments; carry the rest.

    ``seen`` ``(n_leaves,)`` marks the covered leaves; ``cost``, ``in_gate`` and ``hit``
    are ``(n_seen, n_meas)`` over them, ``miss`` and ``linking`` ``(n_seen,)``. A child's log
    weight is its parent's plus each covered leaf's ``hit`` or ``miss`` plus ``unassigned``
    for each measurement no leaf took; ``cost`` must rank children as that weight does. Each
    prior hypothesis gives ``k_best`` assignments, or :math:`\lceil k w_j \rceil` with
    ``"weighted"`` (PMBM); the cluster keeps the ``k_best`` heaviest, none lighter than
    ``min_share`` of its parent's heaviest child, which no later pruning would keep. With
    ``linking`` None every gate joins clusters and an unassigned measurement with ``born``
    opens a tree in the children that leave it. Otherwise (confirmed first) only linking
    leaves' gates join clusters and block births; a cluster's other leaves also take the
    measurements in no linking gate, each cluster alone, and such a measurement no best
    hypothesis took opens a tree of its own, as a measurement in no gate does.
    """
    n_leaves, n_meas = len(seen), len(born)
    covered = np.flatnonzero(seen)
    padded = np.column_stack([hit, miss])
    row = np.full(n_leaves, -1, dtype=np.intp)
    row[covered] = np.arange(covered.size)
    labels, old = np.unique(hypotheses.cluster, return_inverse=True)
    leaf_old = np.searchsorted(labels, hypotheses.leaf_cluster(n_leaves))
    joins = in_gate if linking is None else in_gate & linking[:, None]
    linked = joins.any(axis=0)
    gated_rows, gated_meas = np.nonzero(joins)
    size = labels.size + n_meas
    graph = coo_array(
        (np.ones(gated_rows.size), (leaf_old[covered[gated_rows]], labels.size + gated_meas)),
        shape=(size, size),
    )
    _, component = connected_components(graph, directed=False)
    touched = np.zeros(labels.size, dtype=bool)
    touched[leaf_old[covered]] = True
    ranked = np.unique(component[: labels.size][touched])
    n_components = int(component.max(initial=-1)) + 1
    holds = [hypotheses.leaf[g] for g in _groups(hypotheses.hypothesis, hypotheses.cluster.size)]
    olds_of = _groups(component[: labels.size], n_components)
    measurements_of = _groups(component[labels.size :], n_components)
    hypotheses_of = _groups(old.reshape(-1), labels.size)
    rows_of = _groups(leaf_old[covered], labels.size)
    taken_best = np.zeros(n_meas, dtype=bool)
    keys: list[NDArray[np.int64]] = []
    weights: list[float] = []
    clusters: list[int] = []
    counts: list[tuple[int, int]] = []
    span = n_meas + 3  # a key packs (old leaf, measurement code + 2); a new tree follows them all
    gap = -math.log(min_share) if min_share > 0 else np.inf
    # One cluster at a time: each is ranked alone, and gating keeps them few and small.
    for label in ranked:
        olds = olds_of[label]
        measurements = measurements_of[label]
        if linking is not None:
            mine = np.concatenate([rows_of[o] for o in olds])
            loose = np.flatnonzero(in_gate[mine].any(axis=0) & ~linked)
            measurements = np.union1d(measurements, loose)
        parents: list[tuple[NDArray[np.intp], float]] = [(np.zeros(0, dtype=np.intp), 0.0)]
        for o in olds:
            options = [(holds[h], float(hypotheses.log_weight[h])) for h in hypotheses_of[o]]
            parents = _cross(parents, options, k_best)
        prior = np.array([w for _, w in parents])
        share = np.exp(prior - prior.max())
        share /= share.sum()
        children: list[tuple[int, NDArray[np.intp], float]] = []
        for p, (leaves, weight) in enumerate(parents):
            rows = row[leaves[seen[leaves]]]
            k = k_best if per_parent == "k_best" else max(1, math.ceil(k_best * share[p]))
            chosen, _ = murty_k_best(
                cost[np.ix_(rows, measurements)],
                k,
                in_gate=in_gate[np.ix_(rows, measurements)],
                non_assignment_cost=non_assignment_cost,
                max_gap=gap,
            )
            # Index -1, a miss, reads the last entry: the miss column and no measurement used.
            taken = np.append(measurements, _MISS)[chosen]
            gain = padded[rows[None, :], taken].sum(axis=1)
            used = np.append(unassigned, 0.0)[taken].sum(axis=1)
            total = weight + gain + unassigned[measurements].sum() - used
            children += [(p, t, float(w)) for t, w in zip(taken, total, strict=True)]
        children.sort(key=lambda child: -child[2])
        counts.append((len(parents), len(children)))
        best = children[0][1]
        taken_best[best[best >= 0]] = True
        opens = measurements[born[measurements]] if linking is None else measurements[:0]
        for p, taken, weight in children[:k_best]:
            leaves = parents[p][0]
            new = opens[~np.isin(opens, taken)] if opens.size else opens
            keys.append(
                np.concatenate(
                    [
                        leaves[seen[leaves]] * span + taken + 2,
                        leaves[~seen[leaves]] * span + _CARRIED + 2,
                        n_leaves * span + new,
                    ]
                )
            )
            weights.append(weight)
            clusters.append(int(label))
    # A measurement in no cluster ranked, and taken by no best hypothesis, opens a tree alone.
    outside = ~np.isin(component[labels.size :], ranked)
    alone = np.flatnonzero(outside & born & ~taken_best)
    keys += list(n_leaves * span + alone[:, None])
    weights += unassigned[alone].tolist()
    clusters += (2 * size + alone).tolist()
    # Clusters the scan did not touch keep their hypotheses and leaves as they were.
    for h in np.flatnonzero(~touched[old]):
        keys.append(holds[h] * span + _CARRIED + 2)
        weights.append(float(hypotheses.log_weight[h]))
        clusters.append(int(size + old[h]))
    owner = np.concatenate(
        [np.zeros(0, np.intp), *(np.full(len(k), h, np.intp) for h, k in enumerate(keys))]
    )
    unique, leaf = np.unique(np.concatenate([np.zeros(0, np.int64), *keys]), return_inverse=True)
    parent = np.where(unique < n_leaves * span, unique // span, -1).astype(np.intp)
    code = np.where(parent >= 0, unique % span - 2, unique - n_leaves * span).astype(np.intp)
    count = np.array(counts, dtype=np.int64).reshape(-1, 2)
    result = GlobalHypotheses(
        leaf.reshape(-1).astype(np.intp),
        owner,
        np.array(weights, dtype=np.float64),
        np.array(clusters, dtype=np.int64),
    )
    return Extension(parent, code, result, count[:, 0], count[:, 1])


def collapse(
    hypotheses: GlobalHypotheses, k_best: int, merge: Literal["max", "sum"]
) -> GlobalHypotheses:
    """Merge hypotheses holding the same leaves, keep each cluster's k best, drop empty clusters.

    Duplicates keep their best weight (``"max"``, a score) or add (``"sum"``, a probability).
    """
    if not hypotheses.cluster.size:
        return hypotheses
    order = np.lexsort((hypotheses.leaf, hypotheses.hypothesis))
    bounds = np.searchsorted(hypotheses.hypothesis[order], np.arange(1, hypotheses.cluster.size))
    holds = np.split(hypotheses.leaf[order], bounds)
    first: dict[tuple[int, bytes], int] = {}
    group = np.array(
        [
            first.setdefault((int(c), leaves.tobytes()), len(first))
            for c, leaves in zip(hypotheses.cluster, holds, strict=True)
        ],
        dtype=np.intp,
    ).reshape(-1)
    weight = np.full(len(first), -np.inf)
    if merge == "max":
        np.maximum.at(weight, group, hypotheses.log_weight)
    else:
        np.logaddexp.at(weight, group, hypotheses.log_weight)
    representative = np.zeros(len(first), dtype=np.intp)
    representative[group[::-1]] = np.arange(group.size)[::-1]
    cluster = hypotheses.cluster[representative]
    # A cluster whose hypotheses hold no leaf at all has nothing left to say.
    holding = np.isin(cluster, hypotheses.cluster[hypotheses.hypothesis])
    keep = np.zeros(group.size, dtype=bool)
    keep[representative[_top(weight, cluster, k_best) & holding]] = True
    unique = GlobalHypotheses(
        hypotheses.leaf, hypotheses.hypothesis, weight[group], hypotheses.cluster
    )
    return unique.keep(keep)


def split(
    hypotheses: GlobalHypotheses, tree: NDArray[np.int64], leaf_weight: NDArray[np.float64]
) -> GlobalHypotheses:
    """Move each tree whose one leaf is in every hypothesis of its cluster into its own cluster.

    ``leaf_weight`` is what each such leaf adds to every hypothesis holding it: its share
    moves with it.
    """
    n_leaves = len(tree)
    labels, cluster_index = np.unique(hypotheses.cluster, return_inverse=True)
    n_hypotheses = np.bincount(cluster_index.reshape(-1), minlength=labels.size)
    index = np.searchsorted(labels, hypotheses.leaf_cluster(n_leaves))
    _, tree_index, per_tree = np.unique(tree, return_inverse=True, return_counts=True)
    single = per_tree[tree_index.reshape(-1)] == 1
    holding = np.bincount(hypotheses.leaf, minlength=n_leaves)
    crowded = np.bincount(index, minlength=labels.size)[index] > 1
    settled = single & (holding == n_hypotheses[index]) & crowded
    if not settled.any():
        return hypotheses
    moved = np.flatnonzero(settled)
    pairs = ~settled[hypotheses.leaf]
    weight = hypotheses.log_weight - hypotheses.sums(np.where(settled, leaf_weight, 0.0))
    new = np.arange(moved.size, dtype=np.intp)
    start = int(hypotheses.cluster.max(initial=-1)) + 1
    return GlobalHypotheses(
        np.concatenate([hypotheses.leaf[pairs], moved]),
        np.concatenate([hypotheses.hypothesis[pairs], hypotheses.cluster.size + new]),
        np.concatenate([weight, leaf_weight[moved]]),
        np.concatenate([hypotheses.cluster, start + new.astype(np.int64)]),
    )


@dataclass(frozen=True, eq=False)
class TrackTree:
    """Every tree's leaves between scans: rows of one bank, one per surviving history.

    ``bank.track_id`` is a leaf's tree. ``score`` is its track score and ``peak`` the
    highest along its branch; ``node`` its own node's ID and ``ancestry`` ``(n_leaves,
    n_scan)`` its branch's nodes over the last ``n_scan`` scans, oldest first (-1 before the
    tree began); ``hypotheses`` the global hypotheses holding it.
    """

    bank: TrackBank
    score: NDArray[np.float64]
    peak: NDArray[np.float64]
    node: NDArray[np.int64]
    ancestry: NDArray[np.int64]
    hypotheses: GlobalHypotheses
    next_node: int = 0

    @classmethod
    def empty(cls, layout: StateLayout, n_modes: int = 0) -> TrackTree:
        """Return a tree with no leaves."""
        nothing = np.zeros(0)
        return cls(
            TrackBank.empty(layout, n_modes),
            nothing,
            nothing,
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 0), dtype=np.int64),
            GlobalHypotheses.empty(),
        )

    def best_leaves(self) -> NDArray[np.intp]:
        """Return the rows of the leaves in each cluster's best hypothesis: the reported ones."""
        best = self.hypotheses.best()
        return np.flatnonzero(self.hypotheses.held(self.bank.n_tracks, best))

    def leaf_count(self) -> NDArray[np.int64]:
        """Return how many leaves each tree holds, in tree-ID order."""
        return np.unique(self.bank.track_id, return_counts=True)[1].astype(np.int64)

    def hypothesis_count(self) -> NDArray[np.int64]:
        """Return how many global hypotheses each cluster holds, in cluster-label order."""
        return np.unique(self.hypotheses.cluster, return_counts=True)[1].astype(np.int64)

    def plant(self, roots: TrackBank, score: float) -> TrackTree:
        """Return the tree with ``roots`` added, each a tree of its own scored ``score``."""
        n_new, width = roots.n_tracks, self.ancestry.shape[1]
        return TrackTree(
            self.bank.join(roots),
            np.concatenate([self.score, np.full(n_new, score)]),
            np.concatenate([self.peak, np.full(n_new, score)]),
            np.concatenate([self.node, self.next_node + np.arange(n_new, dtype=np.int64)]),
            np.concatenate([self.ancestry, np.full((n_new, width), -1, dtype=np.int64)]),
            self.hypotheses.plant(self.bank.n_tracks, n_new, score),
            self.next_node + n_new,
        )


@dataclass(frozen=True, eq=False)
class Branching:
    """One scan of MHT: the surviving leaves, where each came from, and the new tree's columns.

    ``parent`` and ``measurement`` read as in :class:`Extension`; ``confirmed`` is whether a
    leaf's branch has reached the confirmation score; ``n_parents`` and ``n_children`` per
    extended cluster are what Murty ranked. :meth:`grow` takes the updated leaves.
    """

    parent: NDArray[np.intp]
    measurement: NDArray[np.intp]
    score: NDArray[np.float64]
    peak: NDArray[np.float64]
    node: NDArray[np.int64]
    ancestry: NDArray[np.int64]
    confirmed: NDArray[np.bool_]
    hypotheses: GlobalHypotheses
    next_node: int
    n_parents: NDArray[np.int64]
    n_children: NDArray[np.int64]

    def grow(self, bank: TrackBank) -> TrackTree:
        """Return the tree whose leaves are ``bank``'s rows, in this branching's order."""
        return TrackTree(
            bank, self.score, self.peak, self.node, self.ancestry, self.hypotheses, self.next_node
        )


def _n_scan(
    hypotheses: GlobalHypotheses, tree: NDArray[np.int64], decided: NDArray[np.int64]
) -> NDArray[np.bool_]:
    """Return the hypotheses that agree with their cluster's best on every decided branch.

    ``decided`` is each leaf's ancestor ``n_scan`` scans back, -1 for a tree younger than
    that. A tree the best hypothesis lacks may appear elsewhere only while undecided.
    """
    n_leaves = len(tree)
    _, tree_index = np.unique(tree, return_inverse=True)
    tree_index = tree_index.reshape(-1)
    absent = -2
    best_decided = np.full(int(tree_index.max(initial=-1)) + 1, absent, dtype=np.int64)
    in_best = hypotheses.held(n_leaves, hypotheses.best())
    best_decided[tree_index[in_best]] = decided[in_best]
    leaf, h = hypotheses.leaf, hypotheses.hypothesis
    want = best_decided[tree_index[leaf]]
    agree = (decided[leaf] == want) | ((decided[leaf] == -1) & (want == absent))
    n_hypotheses = hypotheses.cluster.size
    matched = np.bincount(h, weights=agree & (want >= 0), minlength=n_hypotheses)
    labels, cluster_index = np.unique(hypotheses.cluster, return_inverse=True)
    leaf_cluster = np.searchsorted(labels, hypotheses.leaf_cluster(n_leaves))
    required = np.bincount(leaf_cluster[in_best & (decided >= 0)], minlength=labels.size)
    disagree = np.bincount(h, weights=~agree, minlength=n_hypotheses)
    return np.asarray((disagree == 0) & (matched == required[cluster_index]), dtype=bool)


class Hypothesiser(Protocol):
    """What a :class:`~radar_forge.tracking.tracker.Tracker` asks of an associator that branches."""

    @property
    def confirm_score(self) -> float:
        """The score at which a branch is confirmed, and a cue starts."""
        ...

    def hypotheses(
        self,
        tree: TrackTree,
        seen: NDArray[np.bool_],
        log_likelihood: NDArray[np.float64],
        in_gate: NDArray[np.bool_],
        *,
        clutter_intensity: NDArray[np.float64],
        born: NDArray[np.bool_],
        linking: NDArray[np.bool_] | None = None,
    ) -> Branching:
        """Return one scan's branching of the covered leaves."""
        ...


@dataclass(frozen=True)
class TrackOrientedMHT:
    r"""Track-oriented MHT as an associator: each scan's k-best hypotheses per cluster.

    A hit adds :math:`\ln(P_D g / \lambda)`, a miss :math:`\ln(1 - P_D P_G)`, to a leaf's
    score [1]; the cost matrix is GNN's with the finite non-assignment cost
    :math:`\ln(P_D / (1 - P_D P_G))`, the score's break-even, so ``k_best = 1`` is GNN. A
    branch is confirmed at ``confirm_score`` and deleted when its score falls ``delete_drop``
    below the highest along it; after each scan every hypothesis that disagrees with its
    cluster's best on a branch ``n_scan`` scans back is dropped [4].

    References
    ----------
    Blackman [1] (the track score and its sequential thresholds); Murty [2]; Cox and
    Miller [3]; Kurien [4] (N-scan pruning).
    """

    detection_probability: float
    gate_probability: float
    k_best: int = 10
    n_scan: int = 3
    confirm_score: float = CONFIRM_SCORE
    delete_drop: float = DELETE_DROP

    def __post_init__(self) -> None:
        """Refuse settings that could not work."""
        p_d, p_g = self.detection_probability, self.gate_probability
        counts = (self.k_best, self.n_scan)
        if any(type(n) is not int or n < 1 for n in counts) or not (0 < p_d < 1 and 0 < p_g < 1):
            msg = f"need int k_best, n_scan >= 1 and 0 < P_D, P_G < 1; got {counts}, {p_d}, {p_g}."
            raise ValueError(msg)
        if not (math.isfinite(self.confirm_score) and self.delete_drop > 0):
            msg = f"need a finite confirm_score and delete_drop > 0; got {self}."
            raise ValueError(msg)

    @property
    def non_assignment_cost(self) -> float:
        r"""Return :math:`\ln(P_D / (1 - P_D P_G))`, where a hit and a miss score alike."""
        p_d = self.detection_probability
        return math.log(p_d / (1 - p_d * self.gate_probability))

    def assign(self, cost: ArrayLike, in_gate: ArrayLike) -> Assignment:
        """Return the one best assignment, GNN's, for a caller that needs one."""
        return global_nearest_neighbour(cost, in_gate, non_assignment_cost=self.non_assignment_cost)

    def weights(self, *args: object, **kwargs: object) -> None:
        """Return None: MHT branches rather than weights."""
        return None

    def hypotheses(
        self,
        tree: TrackTree,
        seen: NDArray[np.bool_],
        log_likelihood: NDArray[np.float64],
        in_gate: NDArray[np.bool_],
        *,
        clutter_intensity: NDArray[np.float64],
        born: NDArray[np.bool_],
        linking: NDArray[np.bool_] | None = None,
    ) -> Branching:
        """Branch every covered leaf, rank each cluster's hypotheses, and prune; never prints.

        ``log_likelihood`` and ``in_gate`` are ``(n_seen, n_meas)`` over the covered leaves,
        ``clutter_intensity`` λ per measurement, ``born`` where a new tree may start,
        ``linking`` the leaves whose gates join clusters (:func:`extend`; None: all).
        """
        p_d, p_g = self.detection_probability, self.gate_probability
        n_old, width = tree.ancestry.shape
        # Every old column gains a last entry, read by a new tree's parent index -1.
        ancestry = np.full((n_old + 1, self.n_scan), -1, dtype=np.int64)
        kept_width = min(width, self.n_scan)
        ancestry[:n_old, self.n_scan - kept_width :] = tree.ancestry[:, width - kept_width :]
        old_score, old_peak = np.append(tree.score, 0.0), np.append(tree.peak, 0.0)
        old_node = np.append(tree.node, -1)
        was_confirmed = np.append(np.isin(tree.bank.status, _CONFIRMED), False)
        ratio = np.where(in_gate, log_likelihood - np.log(clutter_intensity)[None, :], -np.inf)
        miss = math.log(1 - p_d * p_g)
        step = extend(
            tree.hypotheses,
            seen,
            np.where(in_gate, -ratio, np.inf),
            in_gate,
            non_assignment_cost=self.non_assignment_cost,
            hit=math.log(p_d) + ratio,
            miss=np.full(len(ratio), miss),
            unassigned=np.zeros(len(born)),
            born=born,
            k_best=self.k_best,
            linking=linking,
        )
        parent, code = step.parent, step.measurement
        row = np.append(np.cumsum(seen) - 1, -1)
        new, kept = parent < 0, code == _CARRIED
        hits = ~new & (code >= 0)
        increment = np.where(kept, 0.0, miss)
        increment[hits] = math.log(p_d) + ratio[row[parent[hits]], code[hits]]
        score = np.where(new, 0.0, old_score[parent] + increment)
        peak = np.where(new, 0.0, np.maximum(old_peak[parent], score))
        node = np.where(kept, old_node[parent], tree.next_node + np.arange(parent.size))
        lineage = np.where(
            kept[:, None],
            ancestry[parent],
            np.column_stack([ancestry[parent][:, 1:], old_node[parent]]),
        )
        lineage[new] = -1
        confirmed = was_confirmed[parent] | (score >= self.confirm_score)
        trees = np.where(new, -1 - np.maximum(code, 0), np.append(tree.bank.track_id, 0)[parent])
        # A branch fallen delete_drop below its peak leaves every hypothesis that held it.
        grown = step.hypotheses
        alive = score[grown.leaf] >= peak[grown.leaf] - self.delete_drop
        living = GlobalHypotheses(
            grown.leaf[alive], grown.hypothesis[alive], grown.log_weight, grown.cluster
        )
        living = GlobalHypotheses(
            living.leaf, living.hypothesis, living.sums(score), living.cluster
        )
        ranked = collapse(living, self.k_best, "max")
        ranked = ranked.keep(_n_scan(ranked, trees, lineage[:, 0]))
        used = ranked.held(parent.size)
        pruned = ranked.drop(~used)
        pruned = collapse(split(pruned, trees[used], score[used]), self.k_best, "max")
        return Branching(
            parent[used],
            code[used],
            score[used],
            peak[used],
            node[used],
            lineage[used],
            confirmed[used],
            pruned,
            tree.next_node + parent.size,
            step.n_parents,
            step.n_children,
        )
