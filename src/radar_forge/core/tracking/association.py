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
.. [1] Y. Bar-Shalom and X. R. Li, *Multitarget-Multisensor Tracking: Principles
       and Techniques*, YBS Publishing, 1995, §2.3.2 (the validation region),
       §7.1.1 (data association as an assignment problem).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from radar_forge.core.tracking.estimation import InnovationStats

__all__ = [
    "AssociationResult",
    "Associator",
    "ChiSquareGate",
    "GlobalNearestNeighbour",
    "NearestNeighbour",
    "associate_gnn",
    "gate_threshold",
]

# An unassociated pair must be impossible for the assignment solver to pick, but
# scipy raises "cost matrix is infeasible" whenever inf entries leave no complete
# matching -- which gating does routinely, as soon as one track has nothing in its
# gate or two tracks contend for one measurement. So the forbidden cost has to be
# a large finite number instead, and pairs carrying it are dropped from the
# result. Any value above the largest admissible d^2 works; this one is far
# above every chi-squared quantile and still squares without overflowing float64.
_FORBIDDEN_COST = 1e12


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


def gate_threshold(gate_probability: float, dim: int) -> float:
    r"""Return the chi-squared gate threshold for a measurement dimension.

    A measurement is validated when :math:`d^2 \le \chi^2_{\text{dim}}(P_G)`.
    The gate is the validation region of [1]_ §2.3.2, and :math:`d^2` is
    chi-squared distributed for a consistent filter ([2]_ §5.4.2). The gate is
    a statement about the filter's own uncertainty, not a fixed number of
    metres, and neither its width nor its *dimension* may be
    hard-coded: the dimension varies within a single run of scenario 003,
    because a track starts range-only and is promoted to range-and-rate once its
    Doppler fold becomes resolvable.

    Parameters
    ----------
    gate_probability : float
        The probability mass the gate admits, :math:`P_G`, in ``(0, 1)``.
    dim : int
        Measurement dimension, the chi-squared degrees of freedom.

    Returns
    -------
    float
        The threshold on :math:`d^2`.

    Raises
    ------
    ValueError
        If ``gate_probability`` is outside ``(0, 1)`` or ``dim`` is not
        positive.

    References
    ----------
    .. [1] Bar-Shalom and Li (1995), §2.3.2; see the module References.
    .. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with
           Applications to Tracking and Navigation*, Wiley, 2001, §5.4.2.

    Examples
    --------
    >>> bool(np.isclose(gate_threshold(0.99, 2), 9.210, atol=5e-4))
    True
    """
    if not 0.0 < gate_probability < 1.0:
        msg = f"gate_probability must lie strictly between 0 and 1; got {gate_probability}."
        raise ValueError(msg)
    if dim < 1:
        msg = f"dim must be a positive measurement dimension; got {dim}."
        raise ValueError(msg)
    return float(chi2.ppf(gate_probability, dim))


def associate_gnn(
    squared_distances: ArrayLike,
    gate: float,
) -> list[tuple[int, int]]:
    r"""Assign measurements to tracks by global nearest neighbour.

    Solves the rectangular assignment problem (data association posed as an
    assignment problem, [1]_ §7.1.1) over the matrix of normalised
    innovations, forbidding any pair that falls outside the gate, and
    returns the minimum-total-:math:`d^2` set of one-to-one pairings. The solver
    is ``scipy.optimize.linear_sum_assignment``, which implements the algorithm
    of [2]_; ``scipy`` is already a core dependency, so this adds none.

    Global nearest neighbour rather than JPDA is a deliberate limit
    (``spec/scenario-003-tracking.md`` §7). JPDA is an extension point,
    ``spec/tracker-001.md`` §7.4.

    Parameters
    ----------
    squared_distances : array_like
        Shape ``(n_tracks, n_measurements)``. Entry ``(i, j)`` is the
        :math:`d^2` of measurement ``j`` against track ``i``. Use ``inf`` or any
        value above ``gate`` for a pair that was never evaluated.
    gate : float
        The validation threshold on :math:`d^2`, from :func:`gate_threshold`. A
        pair scoring above it is never assigned.

    Returns
    -------
    list of (int, int)
        ``(track_index, measurement_index)`` pairs, sorted by track index. Every
        returned pair is inside the gate; tracks and measurements not named in
        the list went unassociated.

    Raises
    ------
    ValueError
        If ``squared_distances`` is not two-dimensional, or ``gate`` is not
        positive.

    Notes
    -----
    With one target and a handful of false alarms per frame, the assignment is
    almost always trivial and a greedy nearest neighbour would find the same
    answer. It is solved properly anyway because the cases where greedy differs
    -- two tracks contending for one measurement -- are exactly the cases worth
    getting right, and the solver costs nothing at this size.

    References
    ----------
    .. [1] Bar-Shalom and Li (1995), §7.1.1; see the module References.
    .. [2] D. F. Crouse, "On implementing 2D rectangular assignment
           algorithms," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 52, no. 4,
           pp. 1679-1696, 2016.

    Examples
    --------
    >>> associate_gnn([[1.0, 50.0], [50.0, 2.0]], 9.21)
    [(0, 0), (1, 1)]
    """
    cost = np.atleast_2d(np.asarray(squared_distances, dtype=np.float64))
    if cost.ndim != 2:
        msg = (
            "squared_distances must be two-dimensional, (n_tracks, n_measurements); "
            f"got {cost.shape}."
        )
        raise ValueError(msg)
    if gate <= 0.0:
        msg = f"gate must be a positive threshold on d^2; got {gate}."
        raise ValueError(msg)
    if cost.size == 0:
        return []

    gated = np.where(np.isfinite(cost) & (cost <= gate), cost, _FORBIDDEN_COST)
    track_indices, measurement_indices = linear_sum_assignment(gated)

    return [
        (int(track_index), int(measurement_index))
        for track_index, measurement_index in zip(track_indices, measurement_indices, strict=True)
        if gated[track_index, measurement_index] < _FORBIDDEN_COST
    ]


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
    total cost. So a track is never left without a measurement just because
    leaving it out would lower the total cost. Stone Soup's
    ``GNNWith2DAssignment`` differs: it prices a missed detection just above
    the largest gated distance and minimises the total cost, so it can leave a
    track unassigned to lower the total.

    Notes
    -----
    This is :func:`associate_gnn`, the assignment the scenario 003 tracker
    uses, with no gate of its own: the costs arrive already gated, with
    ``+inf`` for a rejected pair.
    ``associate_gnn`` gives each rejected pair a large finite cost,
    ``_FORBIDDEN_COST``, and solves with ``scipy.optimize.linear_sum_assignment``,
    which minimises the total cost. Using a rejected pair then costs that much,
    so the solver makes as many real pairs as it can, provided the real costs
    differ by far less than it, as NIS and log-likelihood costs always do. Negative costs, such as a
    negative log-likelihood, are allowed.

    The alternative is a finite non-assignment cost: leaving a track unpaired
    costs a fixed amount, often worked out from the detection probability and
    the clutter density. A pair is then made only if it is cheaper than that
    fixed cost, so the number of pairs is no longer maximised. That is the
    usual choice when clutter is dense. It is not implemented here.

    References
    ----------
    .. [1] Bar-Shalom and Li (1995), §7.1.1; see the module References.
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
        # The costs are already gated, so associate_gnn is given no gate: an
        # infinite threshold keeps every finite cost.
        return _result(costs, associate_gnn(costs, np.inf))


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
           Applications to Tracking and Navigation*, Wiley, 2001, §5.4.2 (the
           NIS of a consistent filter is chi-square distributed).
    .. [2] Bar-Shalom and Li (1995), §2.3.2 (the validation region); see the
           module References.

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
        return bool(stats.nis <= self.threshold(len(stats.innovation)))
