"""Explicit observation-to-prior strategies, separate from lifecycle management.

Classes:
- TrackInitiator(Protocol)
- NoInitiation
- DirectStateInitiator
- RangeBearingInitiator
- RoutedInitiator

References
----------
.. [1] StoneSoup initiators; Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
.. [2] Local routed/direct initiation, spec/tracker-001-provenance.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np

from radar_forge.tracking.estimation import Estimator
from radar_forge.tracking.measurements import (
    CartesianPosition,
    MeasurementModel,
    MonostaticRadar,
)
from radar_forge.tracking.sensors import Measurement
from radar_forge.tracking.spaces import StateEstimate

__all__ = [
    "DirectStateInitiator",
    "NoInitiation",
    "RangeBearingInitiator",
    "RoutedInitiator",
    "TrackInitiator",
]


class TrackInitiator(Protocol):
    """Return an independent estimator or None for underdetermined observations.

    References
    ----------
    .. [1] StoneSoup initiator separation; local tracker initiation protocol.
    """
    ###################################################################################################
    # This function defines how a custom track starter must turn a reading into a new filter.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - Estimator or None: Required new, independent filter; None means the reading cannot start a
    #   track.
    def initiate(self, measurement: Measurement, model: MeasurementModel) -> Estimator | None:
        """Construct a prior at observation time; see TrackInitiator References."""
        ...
    ###################################################################################################

class NoInitiation:
    """Use external seeds only; no automatic births.

    References
    ----------
    .. [1] Local explicit-prior routing policy.
    """
    ###################################################################################################
    # This function prevents automatic track creation so tracks must be supplied through explicit
    # starting estimates.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - None; deliberately creates no filter and changes no tracks.
    def initiate(self, measurement: Measurement, model: MeasurementModel) -> Estimator | None:
        """Return None for every observation; see NoInitiation References."""
        return None
    ###################################################################################################


class DirectStateInitiator:
    """Replace observed prior coordinates using an independent prior for the rest.

    Parameters
    ----------
    estimator_factory : callable
        StateEstimate -> fresh estimator.
    prior : StateEstimate
        Mean (n,) and covariance (n,n); observed/unobserved blocks must be independent.

    References
    ----------
    .. [1] Local DirectStateInitiator; StoneSoup measurement-based initiation.
    """
    ###################################################################################################
    # This function stores a filter builder and starting assumptions for quantities a sensor does
    # not measure.
    #
    # Inputs:
    # - estimator_factory (callable): Builds a fresh filter from a StateEstimate.
    # - prior (StateEstimate): Starting values and uncertainty for quantities not supplied by a
    #   measurement.
    #
    # Outputs:
    # - None; retains the builder and starting estimate for later track creation.
    def __init__(
        self, estimator_factory: Callable[[StateEstimate], Estimator], prior: StateEstimate
    ) -> None:
        self.estimator_factory, self.prior = estimator_factory, prior
    ###################################################################################################

    ###################################################################################################
    # This function starts a filter from directly measured quantities and keeps the stored
    # assumptions for the others.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - Estimator or None: Fresh filter at the reading time; None for an unsupported model or
    #   incompatible coordinates.
    def initiate(self, measurement: Measurement, model: MeasurementModel) -> Estimator | None:
        """Initialize named coordinates, preserving full observation covariance.

        References
        ----------
        .. [1] DirectStateInitiator class references.
        """
        if not isinstance(model, CartesianPosition) or model.state_space != self.prior.state_space:
            return None

        # State elements directly observed by the measurement
        indices = list(model.indices)

        # State elements that are not observed
        other = [i for i in range(len(self.prior.mean)) if i not in indices]

        # Direct initiation assumes observed and unobserved priors are independent, so check that covariances are 0
        if np.any(self.prior.covariance[np.ix_(indices, other)] != 0):
            raise ValueError(
                "direct initiation requires independent observed/unobserved prior blocks"
            )

        # Start from the stored prior
        mean, cov = self.prior.mean.copy(), self.prior.covariance.copy()

        # Replace observed state values with the measurement
        mean[indices] = measurement.value

        # Replace observed-state uncertainty with measurement uncertainty
        cov[np.ix_(indices, indices)] = measurement.covariance

        # Create a fresh estimator at the measurement timestamp
        return self.estimator_factory(
            StateEstimate(mean, cov, measurement.timestamp_s, model.state_space)
        )
    ###################################################################################################

class RangeBearingInitiator:
    """Initialize 3D position from range, clockwise-north azimuth and elevation.

    Parameters
    ----------
    estimator_factory : callable
        StateEstimate -> fresh estimator.
    prior : StateEstimate
        Unobserved velocity/acceleration prior, shape (n,) and covariance (n,n).

    Notes
    -----
    Uses first-order spherical-to-Cartesian covariance propagation, suitable for
    small angular uncertainty. Position/prior cross terms are reset. Radial
    velocity is not sufficient to initialize a full Cartesian velocity vector.
    Missing position coordinates require a custom constrained initiator.

    References
    ----------
    .. [1] StoneSoup SimpleMeasurementInitiator; local range-bearing initiation,
           extended to three dimensions and Radar-Forge azimuth conventions.
    """
    ###################################################################################################
    # This function stores the builder and starting assumptions needed to create a position track
    # from radar readings.
    #
    # Inputs:
    # - estimator_factory (callable): Builds a fresh filter from a StateEstimate.
    # - prior (StateEstimate): Starting values and uncertainty for quantities not supplied by a
    #   measurement.
    #
    # Outputs:
    # - None; retains the builder and starting estimate for later track creation.
    def __init__(
        self, estimator_factory: Callable[[StateEstimate], Estimator], prior: StateEstimate
    ) -> None:
        self.estimator_factory, self.prior = estimator_factory, prior
    ###################################################################################################

    ###################################################################################################
    # This function turns radar distance and two angles into east/north/up position and estimates
    # its uncertainty.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - Estimator or None: Fresh position filter retaining unmeasured starting values; None if the
    #   reading or model cannot support this start.
    def initiate(self, measurement: Measurement, model: MeasurementModel) -> Estimator | None:
        """Propagate spherical uncertainty to ENU; see RangeBearingInitiator References."""
        if not isinstance(model, MonostaticRadar) or model.state_space != self.prior.state_space:
            return None

        # Require x/y/z position coordinates in the target state
        if not all(f"{a}_m" in model.state_space.names for a in "xyz"):
            return None
        
        distance_m, azimuth_rad, elevation_rad = measurement.value[:3]

        # Range-bearing initiation requires a valid positive range
        if distance_m <= 0:
            return None

        # Convert spherical angles into an ENU unit direction vector
        sa, ca = np.sin(azimuth_rad), np.cos(azimuth_rad)
        se, ce = np.sin(elevation_rad), np.cos(elevation_rad)

        direction = np.array([ce * sa, ce * ca, se], dtype=np.float64)

        # Jacobian for propagating measurement uncertainty into Cartesian space 
        # (P = FPF^T, since F not linear, need JPJ^T)
        jacobian = np.array(
            [
                [ce * sa, distance_m * ce * ca, -distance_m * se * sa],
                [ce * ca, -distance_m * ce * sa, -distance_m * se * ca],
                [se, 0, distance_m * ce],
            ],
            dtype=np.float64,
        )

        # Locate x/y/z within the tracker state vector
        indices = list(model.state_space.indices(tuple(f"{a}_m" for a in "xyz")))

        # Start from the stored prior
        mean, cov = self.prior.mean.copy(), self.prior.covariance.copy()

        # Initialise Cartesian position from radar range/azimuth/elevation
        mean[indices] = model.pose.position_m + distance_m * direction

        # Remove prior position uncertainty and cross-correlations
        cov[indices, :] = 0
        cov[:, indices] = 0

        # Insert propagated Cartesian position covariance
        cov[np.ix_(indices, indices)] = jacobian @ measurement.covariance[:3, :3] @ jacobian.T

        # Create a fresh estimator at the measurement timestamp
        return self.estimator_factory(
            StateEstimate(mean, cov, measurement.timestamp_s, model.state_space)
        )
    ###################################################################################################


# DONE: Create a BistaticRangeDoppler MeasurementModel
# TODO: Create a BistaticFSR MeasurementModel 
# TODO: Create a corresponding BistaticInitiator (need to discuss initiation strategy, especially for FSR)
# TODO: Register the new initiator with RoutedInitiator below


class RoutedInitiator:
    """Select a birth strategy by measurement model ID.

    Example:
    -------------------------------------------------------------
    | RoutedInitiator(                                          |
    |    {                                                      |
    |       "cartesian_position": DirectStateInitiator(...),    |
    |       "monostatic_radar": RangeBearingInitiator(...),     |
    |       "bistatic_rd": BistaticInitiator(...),              |
    |    }                                                      |
    | )                                                         |
    -------------------------------------------------------------

    Parameters
    ----------
    strategies : mapping
        Model IDs to initiators; unrouted models do not create tracks.

    References
    ----------
    .. [1] Local routed initiator, spec/tracker-001-provenance.md.
    """
    ###################################################################################################
    # This function sets which track-starting rule to use for each sensor reading type.
    #
    # Inputs:
    # - strategies (mapping of str to TrackInitiator): Measurement model IDs mapped to their track
    #   starters.
    #
    # Outputs:
    # - None; copies the mapping while retaining its track-starter objects.
    def __init__(self, strategies: Mapping[str, TrackInitiator]) -> None:
        self.strategies = dict(strategies)
    ###################################################################################################

    ###################################################################################################
    # This function chooses the stored track-starting rule using the reading's model ID.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - Estimator or None: Result from the chosen starter; None when no rule applies or no track
    #   can be started.
    def initiate(self, measurement: Measurement, model: MeasurementModel) -> Estimator | None:
        """Select the initiation strategy associated with this measurement type. 
        Used when tracker receives different measurement types that require different track-birth logic.
        
        For example, 
        - MonostaticRadar detection -> RangeBearingInitiator
        - CartesianPosition detection -> DirectStateInitiator

        RoutedInitiator.initiate() simply looks at the measurement's measurement_model_id and dispatches
        to the appropriate initiator automatically.
        """
        
        strategy = self.strategies.get(measurement.measurement_model_id)
        return None if strategy is None else strategy.initiate(measurement, model)
    ###################################################################################################