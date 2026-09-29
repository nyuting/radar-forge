"""Interacting multiple-model estimator for a common semantic state space.

Classes:
- IMM: Owns several filters and their transition probabilities, mixing them at each prediction step.

References
----------
.. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
.. [2] Adapted local tracker; spec/tracker-001-provenance.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import logsumexp

from radar_forge.tracking._numerics import FloatArray, timestamp
from radar_forge.tracking.estimation import Estimator, InnovationStats, innovation_stats
from radar_forge.tracking.measurements import MeasurementModel
from radar_forge.tracking.sensors import Measurement
from radar_forge.tracking.spaces import StateEstimate

__all__ = [
    "IMM",
    "mixture",
]


###################################################################################################
# This function combines several model estimates, including the extra uncertainty caused by
# disagreement between them.
#
# Inputs:
# - states (sequence of StateEstimate): Estimates with the same coordinates and time.
# - weights (float64 NumPy array): Contribution of each estimate, shape (number of estimates,),
#   summing to one.
#
# Outputs:
# - StateEstimate: New combined values and uncertainty, using the first estimate's time and
#   coordinate definitions.
def mixture(states: Sequence[StateEstimate], weights: FloatArray) -> StateEstimate:
    """Moment-match compatible estimates with normalized weights, including mean spread.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """
    # Extract the state space - all elements of the input `states` should have the same state space, so we can
    # index into any random one to extract the state space; use the first (index 0) for convenience
    space = states[0].state_space 

    # Perform weighted mean because we are mixing models by the given `weights`
    mean = space.weighted_mean(np.array([s.mean for s in states]), weights)

    # Calculate how the covariance matrix changes due to combining model estimates
    cov = np.zeros_like(states[0].covariance)
    for s, w in zip(states, weights, strict=True):
        d = space.residual(s.mean, mean)
        # IMM covariance = within-mode uncertainty + between-mode spread.
        cov += w * (s.covariance + np.outer(d, d))

    # Return StateEstimate object containing new mean, new covariance, timestamp, and state space
    return StateEstimate(mean, cov, states[0].timestamp_s, space)
###################################################################################################

class IMM:
    """Own complete estimator modes. Transition[i,j]=P(next=j | current=i).

    Each positive-time prediction performs one transition/mixing step; transition
    probabilities are per event, not per second. Equal-time sensor updates do not
    remix modes. Innovation NIS uses moment matching; likelihood is the exact
    weighted mixture of mode Gaussian likelihoods.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    ###################################################################################################
    # This function sets up several filters for one target so their motion explanations can compete and 
    # be combined. I.e. the IMM is a meta-filter that owns several sub-filters. At each prediction step,
    # the sub-filters are mixed according to the transition matrix.
    #
    # Inputs:
    # - modes (sequence of Estimator): Distinct filters tracking the same quantities at the same time.
    # - transition (float64 NumPy array): Model-switching probabilities, shape (k, k); rows are
    #   current models, columns are next models.
    # - probabilities (float64 NumPy array or None): Starting model probabilities, shape (k,);
    #   None gives equal weight.
    #
    # Outputs:
    # - None; keeps the filter objects and stores checked copies of their switching settings.
    def __init__(
        self,
        modes: Sequence[Estimator],
        transition: FloatArray,
        probabilities: FloatArray | None = None,
    ) -> None:
        """ Initialise an Interacting Multiple Model:
        ---------------------------------------------------
        | E.g. A drone may fly:                           |
        |                                                 |
        | - Straight (CV)                                 |
        | - Accelerating (CA)                             |
        | - Turning (CT)                                  |
        |                                                 |
        | IMM creates one filter for each possibility     |
        | and assigns an initial belief to each model.    |
        ---------------------------------------------------
        """
    
        # Store the modes ("sub-filters") as a tuple and check that they are unique.
        self.modes = tuple(modes)
        if not self.modes or len({id(m) for m in self.modes}) != len(self.modes):
            raise ValueError("IMM requires distinct estimator instances")

        # Check that all modes have the same state space and timestamp; otherwise, the IMM cannot combine them.
        first = self.modes[0].state
        if any(
            m.state.state_space != first.state_space or m.state.timestamp_s != first.timestamp_s
            for m in self.modes
        ):
            raise ValueError(
                "IMM modes require a common StateSpace and timestamp; "
                "differing dimensions require explicit state transforms"
            )

        # Validate the transition matrix; the transition matrix tells us how to switch from model to model
        n = len(self.modes)
        self.transition = np.array(transition, dtype=np.float64, copy=True)
        if (
            self.transition.shape != (n, n)
            or not np.all(np.isfinite(self.transition))
            or np.any(self.transition < 0)
            or not np.allclose(self.transition.sum(axis=1), 1)
        ):
            raise ValueError("transition must be a row-stochastic (modes,modes) matrix")

        # Validate the initial mode probabilities, defaulting to uniform if not provided
        self.probabilities = (
            np.full(n, 1 / n)
            if probabilities is None
            else np.array(probabilities, dtype=np.float64, copy=True)
        )
        if (
            self.probabilities.shape != (n,)
            or not np.all(np.isfinite(self.probabilities))
            or np.any(self.probabilities < 0)
            or not np.isclose(self.probabilities.sum(), 1)
        ):
            raise ValueError("mode probabilities must be nonnegative and sum to one")
    ###################################################################################################

    ###################################################################################################
    # This function combines the current model estimates into one result for track management.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - StateEstimate: Fresh combined values and uncertainty, weighted by the stored model
    #   probabilities.
    @property
    def state(self) -> StateEstimate:
        """Return a detached semantic estimate; mean (n,), covariance (n,n).

        ---------------------------------------------------
        | E.g. The modes currently believe:               |
        |                                                 |
        | - CV: target at x=100                           |
        | - CA: target at x=105                           |
        | - CT: target at x=102                           |
        |                                                 |
        | state() returns one combined estimate using     |
        | the current model probabilities.                |
        ---------------------------------------------------

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        return mixture([m.state for m in self.modes], self.probabilities)
    ###################################################################################################
    
    ###################################################################################################
    # This function gives every stored model the same replacement estimate.
    #
    # Inputs:
    # - state (StateEstimate): New values, uncertainty, and time with compatible coordinate
    #   definitions.
    #
    # Outputs:
    # - None; replaces each model's estimate without changing model probabilities.
    def set_state(self, state: StateEstimate) -> None:
        """Replace the estimate after validating its semantic space.

        ---------------------------------------------------
        | E.g. A track is re-initialised at:              |
        |                                                 |
        |                 x=1000, y=500                   |
        |                                                 |
        | Rather than choosing one motion model, IMM      |
        | gives the same starting estimate to all modes.  |
        ---------------------------------------------------

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        if state.state_space != self.state.state_space:
            raise ValueError("IMM state space mismatch")

        # Set the same state for every IMM mode. 
        for mode in self.modes: # self.modes looks something like [UKF(<CV settings>), UKF(<CA settings>), ...]
            mode.set_state(state) # uses each UKF's set_state() method
    ###################################################################################################

    ###################################################################################################
    # This function shares information between motion models, then advances each one to the
    # requested time.
    #
    # Inputs:
    # - time_s (float): Requested time in seconds; cannot precede the stored estimate.
    #
    # Outputs:
    # - None; updates model estimates and probabilities once per positive time step; equal times
    #   leave them unchanged.
    def predict_to(self, time_s: float) -> None:
        """Predict to a nondecreasing timestamp in seconds.

        ---------------------------------------------------
        | E.g. Before prediction:                         |
        |                                                 |
        | - CV thinks straight flight is likely           |
        | - CT thinks a turn is possible                  |
        |                                                 |
        | IMM first shares information between models,    |
        | then lets each mode predict independently.      |
        ---------------------------------------------------

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        time_s = timestamp(time_s) # get current time

        # If the time to be predicted to is in the past, raise an error
        if time_s < self.state.timestamp_s:
            raise ValueError("out-of-sequence prediction is not supported")

        # If the time to be predicted to is the current timestamp, there is no prediction required
        if time_s == self.state.timestamp_s:
            return

        # Otherwise, we need to predict the state for each sub-filter
        # Compute the model probabilities at the next time step using the Markov transition matrix.
        prior = self.probabilities @ self.transition

        # Take a snapshot of the current state estimate from every mode.
        states = [m.state for m in self.modes] # E.g. [CV estimate, CA estimate, CT estimate]

        for j, mode in enumerate(self.modes):
            # If the j-th mode has non-zero probability
            if prior[j] > 0: 

                # compute mixing probabilities for that mode (i.e. if mode j is CV, how much of its starting state should come from CV, CA, CT)
                weights = self.probabilities * self.transition[:, j] / prior[j]

                # Build a mixed state estimate using information from all modes.
                mode.set_state(mixture(states, weights))

            # Once the mode has received its mixed starting state, predict it forward to the requested timestamp using its own motion model.
            mode.predict_to(time_s) # Each mode makes its own prediction

        # Store the predicted model probabilities, which will later be updated again when a measurement
        # arrives and each mode's likelihood is evaluated.
        self.probabilities = prior
    ###################################################################################################

    ############################################# HELPER #############################################
    # This function checks how well each stored motion model explains a reading without updating
    # any model.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - list of InnovationStats: One set of reading differences, uncertainty, and scores per
    #   model, in model order.
    def _mode_stats(
        self, measurement: Measurement, model: MeasurementModel
    ) -> list[InnovationStats]:
        """
        Compute each mode's residual, covariance, NIS and log likelihood without mutation.

        ---------------------------------------------------
        | E.g. A measurement arrives at x=109.            |
        |                                                 |
        | IMM asks each mode:                             |
        | - "How surprising is this measurement?"         |
        |                                                 |
        | This is a helper function. No mode is updated   | 
        | yet; only scores are collected for comparison.  |
        ---------------------------------------------------
        """
        return [m.innovation_statistics(measurement, model) for m in self.modes]
    ###################################################################################################

    ###################################################################################################
    # This function combines the models' reading comparisons into one score for track matching.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - InnovationStats: Expected readings, measured-minus-expected differences, uncertainty, and
    #   match scores. Model estimates and probabilities are unchanged.
    def innovation_statistics(
        self, measurement: Measurement, model: MeasurementModel
    ) -> InnovationStats:
        """Compute residual (m,), covariance (m,m), NIS and log likelihood without mutation.

        ---------------------------------------------------
        | E.g. The modes predict:                         |
        |                                                 |
        | - CV: 100                                       |
        | - CT: 110                                       |
        | - CA: 95                                        |
        |                                                 |
        | IMM combines these predictions into one         |
        | residual and covariance for gating/association. |
        ---------------------------------------------------

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        stats = self._mode_stats(measurement, model) # returns a list of InnovationStats for each mode
        space = model.measurement_space

        # IMM-predicted measurement = probability-weighted average of mode predictions.
        mean = space.weighted_mean(
            np.array([s.predicted_measurement for s in stats]), self.probabilities
        )

        ############## Combine mode innovation covariances into a single IMM covariance ##############
        cov = np.zeros_like(stats[0].innovation_covariance)

        for s, w in zip(stats, self.probabilities, strict=True):

            # Difference between this mode's prediction and the IMM average prediction
            d = space.residual(s.predicted_measurement, mean)

            # IMM covariance = within-mode uncertainty + between-mode spread.
            cov += w * (s.innovation_covariance + np.outer(d, d))

        # Residual and NIS for the overall IMM prediction
        result = innovation_stats(mean, space.residual(measurement.value, mean), cov)
        active = self.probabilities > 0

        # Combine mode likelihoods into a single IMM likelihood
        likelihood = logsumexp(
            np.log(self.probabilities[active]) + np.array([s.log_likelihood for s in stats])[active]
        )
        return InnovationStats(
            mean, result.residual, result.innovation_covariance, result.nis, float(likelihood)
        )
    ###################################################################################################

    ###################################################################################################
    # This function corrects every model with the same reading and gives more weight to the models
    # that explain it better.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - None; updates model estimates and their probabilities.
    def update(self, measurement: Measurement, model: MeasurementModel) -> None:
        """Condition the estimate on one observation at its current timestamp.

        ---------------------------------------------------
        | E.g. Let's say during prediction, each mode:    |
        |                                                 |
        | - CV predicts: measurement should be 100        |
        | - CT predicts: measurement should be 110        |
        | - CA predicts: measurement should be 95         |
        |                                                 |
        | Then the actual measurement arrives:            |
        | - measurement = 109                             |
        |                                                 |
        | CT model predicted best, so:                    |
        | - P(CV) decreases                               |
        | - P(CT) increases                               |
        | - P(CA) decreases                               |
        ---------------------------------------------------

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        # Innovation statistics and measurement likelihood for each mode
        stats = self._mode_stats(measurement, model)

        # logp[j] = log P(mode_j, measurement)
        logp = np.full(len(self.modes), -np.inf)

        active = self.probabilities > 0

        # Bayes update: (posterior) is proportional to (prior x likelihood) 
        # Taking log: posterior = prior mode probability + measurement likelihood
        logp[active] = (
            np.log(self.probabilities[active]) +
            np.array([s.log_likelihood for s in stats])[active]
        )

        # Update each mode's state estimate using the measurement
        for mode in self.modes:
            mode.update(measurement, model)

        # Normalise to obtain posterior mode probabilities
        self.probabilities = np.exp(logp - logsumexp(logp))
    ###################################################################################################
