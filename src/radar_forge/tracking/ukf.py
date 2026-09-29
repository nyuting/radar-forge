"""Scaled additive-noise unscented Kalman filter, independent of modality.

Classes:
- UKF: Full UKF estimator implementation, uses MotionModels defined in motion.py, StateEstimate 
       defined in spaces.py, uses MeasurementModel defined in measurement.py, and predicts state
       and covariance based on 2n+1 sigma points where n is the number of elements in the state vector

References
----------
.. [1] Julier, The scaled unscented transformation, ACC, 2002.
.. [3] FilterPy UKF, https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html.
.. [2] Adapted local tracker; spec/tracker-001-provenance.md.
"""

from __future__ import annotations

import numpy as np

from radar_forge.tracking._numerics import FloatArray, cholesky, covariance, timestamp, vector
from radar_forge.tracking.estimation import InnovationStats, innovation_stats
from radar_forge.tracking.measurements import MeasurementModel
from radar_forge.tracking.motion import MotionModel
from radar_forge.tracking.sensors import Measurement
from radar_forge.tracking.spaces import StateEstimate

__all__ = [
    "UKF",
]


class UKF:
    """2n+1 sigma points, lambda=alpha²(n+kappa)-n; beta=2 suits Gaussians.

    Parameters
    ----------
    state : StateEstimate
        Prior mean (n,) and covariance (n,n) in named coordinate units.
    motion_model : MotionModel
        Compatible transition and continuous-time process noise.
    alpha, beta, kappa : float
        Sigma-point spread, Gaussian-prior correction and secondary scaling.

    Notes
    -----
    Measurement sigma points are regenerated after prediction so additive Q is
    included in the measurement and cross covariances. Covariances are symmetrized;
    Cholesky uses the centralized roundoff-only jitter policy.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    ###################################################################################################
    # This function sets up one filter to predict a target and correct it using sensor readings.
    #
    # Inputs:
    # - state (StateEstimate): Initial values, uncertainty, time, and coordinate definitions.
    # - motion_model (MotionModel): Rule for predicting movement and added uncertainty.
    # - alpha (float): Controls the spread of test states used to represent uncertainty.
    # - beta (float): Adjusts uncertainty weights; 2 suits a bell-shaped error distribution.
    # - kappa (float): Additional test-state spacing adjustment.
    #
    # Outputs:
    # - None; copies the starting estimate and stores the motion model and calculation weights.
    def __init__(
        self,
        state: StateEstimate,
        motion_model: MotionModel,
        alpha: float = 0.5,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:

        self.motion_model = motion_model

        # Creates `_state`, which is an independent, non-public copy of the StateEstimate `state`
        self.set_state(state) 

        # Ensure dimensions and UKF hyperparameters are valid
        n = state.state_space.dimension
        if (
            not np.all(np.isfinite([alpha, beta, kappa]))
            or alpha <= 0
            or beta < 0
            or n + kappa <= 0
        ):
            raise ValueError("require alpha>0, beta>=0 and n+kappa>0")
        
        # Scaling parameter used when generating sigma points
        self.scale = alpha**2 * (n + kappa)

        # Mean weights for the 2n+1 sigma points
        self.wm = np.full(2 * n + 1, 1 / (2 * self.scale))  
        self.wm[0] = 1 - n / self.scale

        # Covariance weights (slightly different for the central sigma point)
        self.wc = self.wm.copy()
        self.wc[0] += 1 - alpha**2 + beta
    ###################################################################################################

    ###################################################################################################
    # This function provides the stored estimate for track management or other filter steps.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - StateEstimate: The current stored object, with read-only arrays; no fresh copy is made
    #   here.
    @property
    def state(self) -> StateEstimate:
        """Return a detached semantic estimate; mean (n,), covariance (n,n).

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        return self._state
    ###################################################################################################

    ###################################################################################################
    # This function replaces the stored estimate after checking that its coordinates match the
    # motion model.
    #
    # Inputs:
    # - state (StateEstimate): Replacement values, uncertainty, time, and coordinate definitions.
    #
    # Outputs:
    # - None; stores a new estimate with independent copies of its arrays.
    def set_state(self, state: StateEstimate) -> None:
        """Replace the estimate after validating its semantic space.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        if state.state_space != self.motion_model.state_space:
            raise ValueError("state and motion-model StateSpace must match")
        self._state = StateEstimate(
            state.mean, state.covariance, state.timestamp_s, state.state_space
        )
    ###################################################################################################

    ###################################################################################################
    # This function creates nearby test states from the stored estimate to carry uncertainty
    # through model calculations.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - float64 NumPy array: 2*n+1 test states for n tracked quantities, shape (2*n+1, n), in
    #   coordinate units.
    def _sigma_points(self) -> FloatArray:
        s = self.state

        # Sigma point offsets derived from the state covariance
        offsets = np.sqrt(self.scale) * cholesky(s.covariance).T

        return np.array(
            [
                # keep periodic coordinates on their valid branch
                s.state_space.normalize(p)
                # generate 2n+1 sigma points: 1 at mean, and 2 per principal axis at +/- offset_i
                for p in np.vstack([s.mean, s.mean + offsets, s.mean - offsets])
            ]
        )
    ###################################################################################################

    ###################################################################################################
    # This function advances the stored estimate and uncertainty using the motion model before a
    # sensor update.
    #
    # Inputs:
    # - time_s (float): Requested time in seconds; cannot precede the stored estimate.
    #
    # Outputs:
    # - None; stores the prediction at time_s, or leaves it unchanged if already at that time.
    def predict_to(self, time_s: float) -> None:
        """Predict to a nondecreasing timestamp in seconds.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        time_s = timestamp(time_s)

        # Prediction interval
        dt_s = time_s - self.state.timestamp_s

        if dt_s < 0:
            raise ValueError("out-of-sequence prediction is not supported")
        if dt_s == 0:
            return
        s = self.state

        # Propagate every sigma point through the motion model
        points = np.array(
            [
                vector(self.motion_model.transition(p, dt_s), len(s.mean), "transition")
                for p in self._sigma_points()
            ]
        )

        # Compute the predicted mean using the propagated sigma points
        mean = s.state_space.weighted_mean(points, self.wm)

        # Deviation of each sigma point from the predicted mean
        residuals = np.array([s.state_space.residual(p, mean) for p in points])

        # Predicted covariance = sigma-point spread + process noise
        cov = (residuals.T * self.wc) @ residuals + covariance(
            self.motion_model.process_noise(s.mean, dt_s), len(s.mean)
        )

        # Store the predicted estimate
        self._state = StateEstimate(mean, (cov + cov.T) / 2, time_s, s.state_space)
    ###################################################################################################

    ###################################################################################################
    # This function compares a reading with the prediction and prepares the information needed to
    # correct the track.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - InnovationStats: Expected readings, measured-minus-expected differences, uncertainty, and
    #   match scores.
    # - Cross-covariance float64 NumPy array: Table linking tracked-value errors to reading errors, 
    #   shape (n, m), used to calculate the correction.
    def _innovation(
        self, measurement: Measurement, model: MeasurementModel
    ) -> tuple[InnovationStats, FloatArray]:

        s = self.state

        # Updates must occur at the estimator's current timestamp
        if measurement.timestamp_s != s.timestamp_s:
            raise ValueError(
                "predict estimator to measurement timestamp_s before update/innovation"
            )
        
        # Measurement model must be compatible with the tracked state
        if model.state_space != s.state_space:
            raise ValueError("measurement model StateSpace is incompatible")
        
        m = model.measurement_space.dimension
        
        vector(measurement.value, m, "measurement")
        
        # Regenerate sigma points around the predicted state
        points = self._sigma_points()
        
        # Transform sigma points into measurement space
        zpoints = np.array([vector(model.predict(p), m, "predicted measurement") for p in points])
        
        # Predicted measurement
        zmean = model.measurement_space.weighted_mean(zpoints, self.wm)
        
        # Measurement-space residuals
        dz = np.array([model.measurement_space.residual(z, zmean) for z in zpoints])
        
        # State-space residuals
        dx = np.array([s.state_space.residual(p, s.mean) for p in points])
        
        # Innovation covariance = predicted measurement uncertainty + measurement noise
        # S = HPH^T + R
        cov = (dz.T * self.wc) @ dz + measurement.covariance

        # Calculate innovation statistics for scoring, gating, and association step
        stats = innovation_stats(
            zmean, model.measurement_space.residual(measurement.value, zmean), (cov + cov.T) / 2
        )

        # Cross-covariance links state errors to measurement errors
        return stats, (dx.T * self.wc) @ dz
    ###################################################################################################

    ###################################################################################################
    # This function scores a reading against the current prediction so track matching can choose a
    # suitable pair.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - InnovationStats: Expected readings, measured-minus-expected differences, uncertainty, and
    #   match scores. The stored estimate is unchanged.
    def innovation_statistics(
        self, measurement: Measurement, model: MeasurementModel
    ) -> InnovationStats:
        """Compute residual (m,), covariance (m,m), NIS and log likelihood without mutation.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        return self._innovation(measurement, model)[0]
    ###################################################################################################

    ###################################################################################################
    # This function corrects the stored prediction using a reading at the same time, accounting
    # for both uncertainties.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - None; replaces the tracked values and uncertainty with the corrected estimate.
    def update(self, measurement: Measurement, model: MeasurementModel) -> None:
        """Condition the estimate on one observation at its current timestamp.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        # Innovation statistics and state/measurement cross-covariance
        stats, cross = self._innovation(measurement, model)

        # Kalman gain: how strongly the measurement should correct the state
        gain = np.linalg.solve(stats.innovation_covariance, cross.T).T

        # Covariance reduction after incorporating the measurement
        # P = P - KSK^T
        cov = self.state.covariance - gain @ stats.innovation_covariance @ gain.T
        
        self._state = StateEstimate(
            self.state.mean + gain @ stats.residual, # posterior mean = predicted mean + correction
            (cov + cov.T) / 2,                       # maintain covariance symmetry
            self.state.timestamp_s,
            self.state.state_space,
        )
    ###################################################################################################
