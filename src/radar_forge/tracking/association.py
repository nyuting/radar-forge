r"""Gating and assignment: which measurement belongs to which track, for a whole scan at once.

A pair passes the gate when its normalised innovation squared (NIS) :math:`\nu^T S^{-1} \nu` is
at most :math:`\chi^2_m(P_G)`, which a consistent filter's true measurement passes a fraction
:math:`P_G` of the time [1] [2]. Each gated pair is costed by NIS, or by the negative
log-likelihood ratio against a clutter density λ, which compares pairs scored at different
dimensions [4]. Global nearest neighbour solves the scan's assignment with
``scipy.optimize.linear_sum_assignment`` [3]; greedy nearest neighbour takes the cheapest pair
first. Every function takes ``(n_tracks, n_meas)`` arrays. JPDA association weights and Murty's
``murty_k_best`` are later additions here, with scenario 011 (D28, D29).

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
    """An assignment rule: each track's measurement (-1 none), and the unassigned ones."""

    def assign(self, cost: ArrayLike, in_gate: ArrayLike) -> Assignment:
        """Return the assignment of one scan's gated pairs."""
        ...


@dataclass(frozen=True)
class GreedyNearestNeighbour:
    """:func:`greedy_nearest_neighbour` as an :class:`Associator`."""

    def assign(self, cost: ArrayLike, in_gate: ArrayLike) -> Assignment:
        """Return :func:`greedy_nearest_neighbour` of the scan."""
        return greedy_nearest_neighbour(cost, in_gate)


@dataclass(frozen=True)
class GlobalNearestNeighbour:
    """:func:`global_nearest_neighbour` as an :class:`Associator`; infinite cost maximises pairs."""

    non_assignment_cost: float = np.inf

    def assign(self, cost: ArrayLike, in_gate: ArrayLike) -> Assignment:
        """Return :func:`global_nearest_neighbour` of the scan."""
        return global_nearest_neighbour(cost, in_gate, non_assignment_cost=self.non_assignment_cost)
