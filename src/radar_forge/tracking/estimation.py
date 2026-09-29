"""Estimator interfaces and Gaussian innovation statistics.

Classes:
- InnovationStats (contains information needed for gating and association)
- Estimator(Protocol): defines a "template" for all estimators to follow. You'll notice that imm.py
                       and ukf.py both implement the same methods defined by this Estimator.

References
----------
.. [1] Bar-Shalom et al., 2001; adapted local tracker (provenance manifest).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from radar_forge.tracking._numerics import FloatArray, cholesky
from radar_forge.tracking.measurements import MeasurementModel
from radar_forge.tracking.sensors import Measurement
from radar_forge.tracking.spaces import StateEstimate

__all__ = [
    "Estimator",
    "InnovationStats",
    "innovation_stats",
]


@dataclass(frozen=True)
class InnovationStats:
    """Measurement moments (m,), covariance (m,m), NIS and log density.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    predicted_measurement: FloatArray
    residual: FloatArray
    innovation_covariance: FloatArray
    nis: float
    log_likelihood: float

###################################################################################################
# This function scores a predicted-versus-measured difference, accounting for uncertainty, before
# track matching or updating.
#
# Inputs:
# - predicted (float64 NumPy array): Expected sensor readings, shape (m,), in measurement units.
# - residual (float64 NumPy array): Measured-minus-expected differences, shape (m,), in the same units.
# - cov (float64 NumPy array): Uncertainty of those differences, shape (m, m).
#
# Outputs:
# - InnovationStats: Expected readings, measured-minus-expected differences, uncertainty, and
#   match scores. Input reading and difference arrays are reused.
def innovation_stats(
    predicted: FloatArray, residual: FloatArray, cov: FloatArray
) -> InnovationStats:
    """Compute normalized innovation squared and Gaussian log density by Cholesky solves.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """
    factor = cholesky(cov)      # Use cholesky to find lower triangular matrix L s.t. cov = L @ L.T
    cov = factor @ factor.T     # For consistency, we'll use the same effective covariance for gating, likelihoods and gain.
    whitened = np.linalg.solve(factor, residual)
    nis = float(whitened @ whitened)
    log_likelihood = -0.5 * (
        len(residual) * np.log(2 * np.pi) + 2 * np.log(np.diag(factor)).sum() + nis
    )
    return InnovationStats(predicted, residual, cov, nis, float(log_likelihood))
###################################################################################################

class Estimator(Protocol):
    """Independent stateful Bayesian estimator; innovations must not mutate state.
    Think of this as a general template interface for Bayesian filters, utilised by
    the Track class in tracks.py to manage the lifecycle of a track's estimate.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    ###################################################################################################
    # This template function defines how a custom filter must expose its current estimate to track management.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - StateEstimate: Required tracked values, uncertainty, time, and coordinate definitions.
    @property
    def state(self) -> StateEstimate:
        """Return a detached semantic estimate; mean (n,), covariance (n,n).
        Detached = mutable copies of the mean and covariance (avoids accidentally mutating internal state).
        Semantic = consistent with the filter's state space and coordinate definitions.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        ...
    ###################################################################################################

    ###################################################################################################
    # This template function defines how a custom filter must accept a replacement estimate.
    #
    # Inputs:
    # - state (StateEstimate): Replacement values, uncertainty, time, and coordinate definitions.
    #
    # Outputs:
    # - None; an implementation replaces its stored estimate.
    def set_state(self, state: StateEstimate) -> None:
        """Replace the estimate after validating its semantic space.
        Usually called after track management has merged or split tracks, 
        or after a track has been initialised.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        ...
    ###################################################################################################

    ###################################################################################################
    # This template function defines how a custom filter must advance its estimate before comparing a new
    # scan.
    #
    # Inputs:
    # - timestamp_s (float): Requested time in seconds, no earlier than the current estimate.
    #
    # Outputs:
    # - None; an implementation updates its stored prediction and uncertainty.
    def predict_to(self, timestamp_s: float) -> None:
        """Predict to a nondecreasing timestamp in seconds.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        ...
    ###################################################################################################

    ###################################################################################################
    # This template function defines how a custom filter must use a reading to correct its current
    # prediction.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - None; an implementation updates its stored values and uncertainty.
    def update(self, measurement: Measurement, model: MeasurementModel) -> None:
        """Condition the estimate on one observation at its current timestamp.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        ...
    ###################################################################################################

    ###################################################################################################
    # This template function defines how a custom filter must score a reading without changing its
    # estimate. Used in imm.py to score each motion model before updating their probabilities.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - InnovationStats: Expected readings, measured-minus-expected differences, uncertainty, and
    #   match scores. Required from the implementation.
    def innovation_statistics(
        self, measurement: Measurement, model: MeasurementModel
    ) -> InnovationStats:
        """Compute residual (m,), covariance (m,m), NIS and log likelihood without mutation.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        ...
    ###################################################################################################
