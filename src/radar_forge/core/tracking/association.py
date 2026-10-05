r"""Gating and assignment: deciding which measurement belongs to which track.

Association is the step where a tracker pairs each track with at most one of
this scan's measurements. It has two parts, and this module holds both:

- **The gate.** :class:`ChiSquareGate` throws out pairs that are too far apart
  to be the same target. "Too far" is measured by the NIS (normalised
  innovation squared), :math:`d^2 = \nu^\top S^{-1} \nu`. Here :math:`\nu`
  is the innovation: the difference between the measurement and what the
  filter expected to see. :math:`S` is the innovation covariance: how
  uncertain that difference is expected to be.
- **The assignment.** An :class:`Associator` takes a cost matrix of shape
  ``(n_tracks, n_measurements)`` and picks pairs so that no track and no
  measurement is used twice. A pair the gate rejected has cost ``+inf`` and is
  never picked. :class:`GlobalNearestNeighbour` (GNN) finds the best set of
  pairs for the whole scan at once. :class:`NearestNeighbour` (NN) greedily
  takes the cheapest pair, then the cheapest of what is left, and so on.

Both associators make hard decisions: each measurement goes to one track or to
none. Methods that share a measurement between tracks (PDA, JPDA) or keep
several hypotheses open (MHT) are not implemented.

References
----------
.. [1] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking
       Systems*, Artech House, 1999, ch. 6 (gating and global nearest-neighbour
       assignment).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from radar_forge.core.tracking.estimation import InnovationStats
from radar_forge.core.tracking.kalman import gate_threshold

__all__ = [
    "AssociationResult",
    "Associator",
    "ChiSquareGate",
    "GlobalNearestNeighbour",
    "NearestNeighbour",
]


@dataclass(frozen=True)
class AssociationResult:
    """The pairs an associator chose, and what it left unpaired.

    All indices refer to the cost matrix the associator was given: a track
    index is a row and a measurement index is a column.

    Attributes
    ----------
    matches : tuple of (int, int)
        ``(track, measurement)`` pairs. ``((0, 1), (2, 0))`` means track 0 took
        measurement 1 and track 2 took measurement 0.
    unassigned_tracks : tuple of int
        Rows with no measurement, in increasing order.
    unassigned_measurements : tuple of int
        Columns with no track, in increasing order.
    """

    matches: tuple[tuple[int, int], ...]
    unassigned_tracks: tuple[int, ...]
    unassigned_measurements: tuple[int, ...]


class Associator(Protocol):
    """The interface every assignment rule follows.

    An associator sees only the cost matrix. It never sees a filter, so any
    filter can be paired with any associator.
    """

    def associate(self, costs: NDArray[np.float64]) -> AssociationResult:
        """Pair tracks with measurements, each used at most once.

        Parameters
        ----------
        costs : numpy.ndarray
            Shape ``(n_tracks, n_measurements)``. Smaller is better; ``+inf``
            forbids a pair.

        Returns
        -------
        AssociationResult
            The chosen pairs and the unpaired rows and columns.
        """
        ...


def _check_costs(costs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``costs`` as a float array, or raise if it is not a cost matrix."""
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 2 or np.any(np.isnan(costs)) or np.any(np.isneginf(costs)):
        msg = (
            "costs must be a 2-D (n_tracks, n_measurements) matrix of finite costs, "
            f"with +inf for gated-out pairs; got shape {costs.shape}."
        )
        raise ValueError(msg)
    return costs


def _result(costs: NDArray[np.float64], matches: list[tuple[int, int]]) -> AssociationResult:
    """Package the chosen pairs and list the rows and columns left unpaired."""
    rows = {i for i, _ in matches}
    cols = {j for _, j in matches}
    return AssociationResult(
        tuple(matches),
        tuple(i for i in range(costs.shape[0]) if i not in rows),
        tuple(j for j in range(costs.shape[1]) if j not in cols),
    )


class NearestNeighbour:
    """Greedy nearest neighbour (NN): take the cheapest pair, then repeat.

    Each step takes the smallest finite cost whose track and measurement are
    both still free. Equal costs are taken in row-major order: row by row,
    and left to right within a row. So the result does not depend on how the
    sort breaks ties. NN is simpler than GNN but can miss the best overall
    assignment. On ``[[2, 1], [8, 2]]`` it takes ``(0, 1)`` first, at cost 1.
    It is then left with the cost-8 pair, for a total of 9. GNN finds
    2 + 2 = 4.
    """

    def associate(self, costs: NDArray[np.float64]) -> AssociationResult:
        """Pair tracks with measurements greedily, cheapest first.

        Parameters
        ----------
        costs : numpy.ndarray
            Shape ``(n_tracks, n_measurements)``. Smaller is better; ``+inf``
            forbids a pair.

        Returns
        -------
        AssociationResult
            The chosen pairs and the unpaired rows and columns.

        Raises
        ------
        ValueError
            If ``costs`` is not 2-D, or holds NaN or ``-inf``.

        Examples
        --------
        >>> NearestNeighbour().associate(np.array([[2.0, 1.0], [8.0, 2.0]])).matches
        ((0, 1), (1, 0))
        """
        costs = _check_costs(costs)
        matches: list[tuple[int, int]] = []
        rows: set[int] = set()
        cols: set[int] = set()
        # Greedy choice is sequential by definition: whether a pair can be taken
        # depends on every pair taken before it, so this cannot be one array step.
        for flat in np.argsort(costs, axis=None, kind="stable"):
            i, j = (int(k) for k in np.unravel_index(flat, costs.shape))
            # The sort puts every +inf last, so the first one ends the search.
            if not np.isfinite(costs[i, j]):
                break
            if i not in rows and j not in cols:
                matches.append((i, j))
                rows.add(i)
                cols.add(j)
        return _result(costs, matches)


class GlobalNearestNeighbour:
    """Global nearest neighbour (GNN): the best set of pairs for the whole scan.

    GNN first makes as many pairs as the gate allows. Only then, among all
    assignments with that many pairs, does it pick the one with the smallest
    total cost. ``kalman.associate_gnn`` and Stone Soup's default GNN
    associator make the same choice. So a track is never left without a
    measurement just because leaving it out would lower the total cost.

    Notes
    -----
    The problem is solved with ``scipy.optimize.linear_sum_assignment`` on an
    augmented matrix. Each track gets one extra "no measurement" column of its
    own, a *dummy*. The finite costs are shifted and scaled into ``[0, 1]``,
    and every dummy entry costs ``n_tracks + 1``. The solver gives every track
    exactly one column, real or dummy. So an assignment with k real pairs pays
    for ``n_tracks - k`` dummies.

    Why the solver always prefers more pairs: each extra real pair removes one
    dummy, which saves ``n_tracks + 1``. All the real pairs together cost at
    most ``n_tracks``, because there are at most ``n_tracks`` of them and each
    costs at most 1. So an assignment with more real pairs always costs less,
    by at least 1, whatever its real costs are. Scaling keeps the order of the
    costs, so among the assignments with the most pairs the cheapest one still
    wins. Negative costs, such as a negative log-likelihood, are allowed for
    the same reason.

    The alternative is a finite non-assignment cost: leaving a track unpaired
    costs a fixed amount, often worked out from the detection probability and
    the clutter density. A pair is then made only if it is cheaper than that
    fixed cost, so the number of pairs is no longer maximised. That is the
    usual choice when clutter is dense. It is not implemented here.

    References
    ----------
    .. [1] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking
           Systems*, Artech House, 1999, ch. 6.
    .. [2] D. F. Crouse, "On implementing 2D rectangular assignment
           algorithms," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 52, no. 4,
           pp. 1679-1696, 2016. The algorithm behind
           ``scipy.optimize.linear_sum_assignment``.
    """

    def associate(self, costs: NDArray[np.float64]) -> AssociationResult:
        """Pair tracks with measurements: the most pairs, then the lowest cost.

        Parameters
        ----------
        costs : numpy.ndarray
            Shape ``(n_tracks, n_measurements)``. Smaller is better; ``+inf``
            forbids a pair. Finite costs may be negative.

        Returns
        -------
        AssociationResult
            The chosen pairs and the unpaired rows and columns.

        Raises
        ------
        ValueError
            If ``costs`` is not 2-D, or holds NaN or ``-inf``.

        Examples
        --------
        Greedy NN would take ``(0, 1)`` first and then pay 8. GNN pays 2 + 2:

        >>> GlobalNearestNeighbour().associate(np.array([[2.0, 1.0], [8.0, 2.0]])).matches
        ((0, 0), (1, 1))
        """
        costs = _check_costs(costs)
        n_tracks, n_measurements = costs.shape
        finite = np.isfinite(costs)
        if n_tracks == 0 or n_measurements == 0 or not finite.any():
            return _result(costs, [])
        values = costs[finite]
        scale = max(1.0, float(np.max(np.abs(values))))
        normalised = values / scale
        low, high = normalised.min(), normalised.max()
        augmented = np.full((n_tracks, n_measurements + n_tracks), float(n_tracks + 1))
        augmented[:, :n_measurements] = np.inf
        augmented[:, :n_measurements][finite] = (normalised - low) / max(1.0, float(high - low))
        rows, cols = linear_sum_assignment(augmented)
        matches = [
            (int(i), int(j))
            for i, j in zip(rows, cols, strict=True)
            if j < n_measurements and finite[i, j]
        ]
        return _result(costs, matches)


@dataclass(frozen=True)
class ChiSquareGate:
    r"""Accept a pair when its NIS is below a chi-square quantile.

    If the filter is right about its own uncertainty, the NIS of the true
    measurement follows a chi-square distribution, with as many degrees of
    freedom as the measurement has components. The gate accepts
    :math:`d^2 \le \chi^2_{n_z}(P_G)`: the value that a fraction :math:`P_G` of
    true measurements fall below. So :math:`P_G = 0.997` keeps the true
    measurement 99.7 % of the time. The gate is a statement about probability,
    not a fixed distance in metres: a track with a large covariance gets a
    large gate.

    Parameters
    ----------
    probability : float, default 0.997
        The gate probability :math:`P_G`, strictly between 0 and 1.

    Raises
    ------
    ValueError
        If ``probability`` is not strictly between 0 and 1.

    References
    ----------
    .. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with
           Applications to Tracking and Navigation*, Wiley, 2001, ch. 5 (the
           NIS of a consistent filter is chi-square distributed).
    .. [2] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking
           Systems*, Artech House, 1999, ch. 6 (gating).

    Examples
    --------
    >>> round(ChiSquareGate(0.99).threshold(2), 3)
    9.21
    """

    probability: float = 0.997

    def __post_init__(self) -> None:
        """Reject a gate probability outside the open interval (0, 1)."""
        if not 0 < self.probability < 1:
            msg = f"gate probability must lie strictly between 0 and 1; got {self.probability}."
            raise ValueError(msg)

    def threshold(self, n_dimensions: int) -> float:
        r"""Return the largest NIS the gate accepts.

        Parameters
        ----------
        n_dimensions : int
            Number of measurement components: the chi-square degrees of freedom.

        Returns
        -------
        float
            :math:`\chi^2_{n}(P_G)`, the ``probability`` quantile of the
            chi-square distribution with ``n_dimensions`` degrees of freedom.

        Raises
        ------
        ValueError
            If ``n_dimensions`` is less than 1.
        """
        return gate_threshold(self.probability, n_dimensions)

    def accepts(self, stats: InnovationStats) -> bool:
        """Return whether one track-measurement pair passes the gate.

        The NIS is already in ``stats``, so the gate only compares it with the
        threshold for the measurement's dimension. It does not change ``stats``.

        Parameters
        ----------
        stats : InnovationStats
            The innovation of one measurement against one track's prediction.

        Returns
        -------
        bool
            True if the pair may be associated, False if it is gated out.
        """
        return bool(stats.nis <= self.threshold(len(stats.residual)))
