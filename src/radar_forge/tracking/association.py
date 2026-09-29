"""Hard assignment on gated costs; no estimator implementation dependencies.

Classes:
- AssociationResult: Hard assignment of measurements to tracks, with unassigned lists.
- Associator(Protocol): Template that ensures that all associators have an associate() method.
- NearestNeighbour: Greedy globally smallest remaining gated pair, with deterministic ties.
- GlobalNearestNeighbour: Maximum-cardinality gated assignment, then minimum total cost.
- ChiSquareGate: Accept Normalized Innovation Squared (NIS) below the observation-dimension chi-square quantile.

References
----------
.. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
.. [2] Adapted local tracker; spec/tracker-001-provenance.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from radar_forge.tracking._numerics import FloatArray
from radar_forge.tracking.estimation import InnovationStats

__all__ = [
    "AssociationResult",
    "Associator",
    "ChiSquareGate",
    "GlobalNearestNeighbour",
    "NearestNeighbour",
]


@dataclass(frozen=True)
class AssociationResult:
    """Integer indices refer to input track rows and measurement columns.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    matches: tuple[tuple[int, int], ...]        # e.g. ((0, 1), (2, 0)) means track 0 matched to measurement 1, and track 2 matched to measurement 0.
    unassigned_tracks: tuple[int, ...]          # e.g. (1, 3) means tracks 1 and 3 have no matched measurements.
    unassigned_measurements: tuple[int, ...]    # e.g. (2, 4) means measurements 2 and 4 have no matched tracks.


class Associator(Protocol):
    """Hard one-to-one assignment protocol, independent of estimator internals.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    ###################################################################################################
    # This function defines how a custom matching rule must assign readings to tracks without
    # reusing either of them (since association is one-to-one).
    # 
    # Inputs:
    # - costs (float64 NumPy array): Track-by-measurement scores; smaller is better, positive
    #   infinity forbids a pair.
    #
    # Outputs:
    # - AssociationResult: Matched index pairs and indices of tracks and measurements left
    #   unmatched. Required from the implementation.
    def associate(self, costs: FloatArray) -> AssociationResult:
        """Assign finite costs (n_tracks,n_measurements); positive infinity forbids a pair.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        ...
    ###################################################################################################

############################################## HELPER ##############################################
# The following helper function is used in the NearestNeighbor and GlobalNearestNeighbor classes (below) to validate 
# and process the cost matrix, ensuring it's in the correct format before a matching rule chooses reading/track pairs.
#
# Inputs:
# - costs (float64 NumPy array): Track-by-measurement scores; smaller is better, positive infinity
#   forbids a pair.
#
# Outputs:
# - float64 NumPy array: Checked score table with the same shape; may share the input storage.
def _validate(costs: FloatArray) -> FloatArray:
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 2 or np.any(np.isnan(costs)) or np.any(np.isneginf(costs)):
        raise ValueError("costs must be a 2D matrix of finite costs or +inf for gated-out pairs")
    return costs
###################################################################################################

############################################## HELPER ##############################################
# The following helper function is used in the NearestNeighbor and GlobalNearestNeighbor classes (below) 
# to package chosen pairs and find which tracks and readings still have no match.
#
# Inputs:
# - costs (float64 NumPy array): Track-by-measurement scores; smaller is better, positive infinity
#   forbids a pair.
# - matches (list of int pairs): Selected track-row and measurement-column indices.
#
# Outputs:
# - AssociationResult: Matched index pairs and indices of tracks and measurements left unmatched.
def _result(costs: FloatArray, matches: list[tuple[int, int]]) -> AssociationResult:
    rows, cols = {i for i, j in matches}, {j for i, j in matches}
    return AssociationResult(
        tuple(matches),
        tuple(i for i in range(costs.shape[0]) if i not in rows), # Find unassigned tracks
        tuple(j for j in range(costs.shape[1]) if j not in cols), # Find unassigned measurements
    )
###################################################################################################


class NearestNeighbour:
    """Greedy globally smallest remaining gated pair, with deterministic ties.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    ###################################################################################################
    # This function chooses the lowest-score available pair repeatedly, keeping each track 
    # and reading in at most one pair.
    #
    # Inputs:
    # - costs (float64 NumPy array): Track-by-measurement scores; smaller is better, positive
    #   infinity forbids a pair.
    #
    # Outputs:
    # - AssociationResult: Matched index pairs and indices of tracks and measurements left
    #   unmatched. Equal scores use array order.
    def associate(self, costs: FloatArray) -> AssociationResult:
        """Assign finite costs (n_tracks,n_measurements); positive infinity forbids a pair.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        costs = _validate(costs)
        matches: list[tuple[int, int]] = []
        rows: set[int] = set()
        cols: set[int] = set()
        # Components/assignments have independent mutable state and are processed in order.
        for flat in np.argsort(costs, axis=None, kind="stable"):
            i, j = np.unravel_index(flat, costs.shape)
            if not np.isfinite(costs[i, j]):
                break
            if i not in rows and j not in cols:
                matches.append((int(i), int(j)))
                rows.add(i)
                cols.add(j)
        return _result(costs, matches) # Returns an AssociationResult object with matched pairs and unassigned indices.
    ###################################################################################################


class GlobalNearestNeighbour:
    """Maximum-cardinality gated assignment, then minimum total cost.

    Dummy columns permit missed tracks. Normalizing finite costs bounds them in
    [0,1]; a penalty greater than the number of tracks prioritizes cardinality.
    This is a simple hard-assignment policy, not a clutter/detection likelihood model.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    ###################################################################################################
    # This function finds the most allowed pairs, then chooses the combination with the smallest
    # total score.
    #
    # Inputs:
    # - costs (float64 NumPy array): Track-by-measurement scores; smaller is better, positive
    #   infinity forbids a pair.
    #
    # Outputs:
    # - AssociationResult: Matched index pairs and indices of tracks and measurements left
    #   unmatched.
    def associate(self, costs: FloatArray) -> AssociationResult:
        """Assign finite costs (n_tracks,n_measurements); positive infinity forbids a pair.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        costs = _validate(costs)
        n, m = costs.shape
        finite = np.isfinite(costs)
        if n == 0 or m == 0 or not finite.any():
            return _result(costs, [])
        values = costs[finite]
        scale = max(1.0, float(np.max(np.abs(values))))
        normalized = values / scale
        low, high = normalized.min(), normalized.max()
        augmented = np.full((n, m + n), float(n + 1))
        augmented[:, :m] = np.inf
        augmented[:, :m][finite] = (normalized - low) / max(1.0, float(high - low))
        rows, cols = linear_sum_assignment(augmented)
        matches = [
            (int(i), int(j)) for i, j in zip(rows, cols, strict=True) if j < m and finite[i, j]
        ]
        return _result(costs, matches)
    ###################################################################################################

@dataclass(frozen=True)
class ChiSquareGate:
    """Accept Normalized Innovation Squared (NIS) below the observation-dimension chi-square quantile.
    This statistical gate uses the squared Mahalanobis distance to reject unlikely track/measurement 
    pairs, accounting for uncertainty.

    Parameters
    ----------
    probability : float
        Gate probability strictly between zero and one.

    References
    ----------
    .. [1] Bar-Shalom et al., 2001, statistical validation gates.
    """

    probability: float = 0.997

    ###################################################################################################
    # This function checks the stored acceptance probability before it is used to reject unlikely pairs.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - None; rejects probabilities outside the open interval from zero to one.
    def __post_init__(self) -> None:
        """Validate the documented constructor contract."""
        if not 0 < self.probability < 1:
            raise ValueError("gate probability must lie strictly between 0 and 1")
    ###################################################################################################

    ###################################################################################################
    # This function checks whether a reading is close enough to a predicted track, accounting for uncertainty. 
    # We make the abstraction that the innovation statistics are already computed, and we only need to compare 
    # the NIS to the chi-square quantile.
    #
    # Inputs:
    # - stats (InnovationStats): Reading differences, their uncertainty, and the resulting scaled
    #   difference score.
    #
    # Outputs:
    # - bool: True allows the pair to enter matching; False rejects it.
    def accepts(self, stats: InnovationStats) -> bool:
        """Test an innovation without changing it; see ChiSquareGate References.
        
        Algorithm: 
        1. Compute NIS = residual.T @ inverse(covariance) @ residual
        2. Compare NIS to the chi-square quantile: NIS <= chi2.ppf
        3. Return True if the pair is accepted, False otherwise.
        """
        return bool(stats.nis <= chi2.ppf(self.probability, len(stats.residual)))
    ###################################################################################################
