"""Mutable target hypotheses and detached read-only consumer snapshots.

Classes:
- TrackStatus: TENTATIVE / CONFIRMED / DELETED
- TrackSnapshot: this is a historical state and thus IMMUTABLE
- Track: this is the live tracker state and thus MUTABLE

Track
├─ mutable
├─ owned by the tracker
├─ contains the live estimator
└─ changes every predict/update cycle
 
TrackSnapshot
├─ read-only
├─ handed to outside consumers
└─ cannot affect the live tracker

References
----------
.. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
.. [2] Adapted local tracker; spec/tracker-001-provenance.md.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from radar_forge.tracking._numerics import FloatArray
from radar_forge.tracking.estimation import Estimator
from radar_forge.tracking.spaces import StateEstimate

__all__ = [
    "Track",
    "TrackSnapshot",
    "TrackStatus",
]


class TrackStatus(StrEnum):
    """Lifecycle labels: tentative, confirmed, and deleted.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    TENTATIVE = "tentative"     # newly created track, not yet confirmed
    CONFIRMED = "confirmed"     # track has satisfied track-initiation criteria
    DELETED = "deleted"         # track is scheduled for removal


@dataclass(frozen=True)
class TrackSnapshot:
    """Detached estimate, lifecycle status and source sensor identifiers.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    track_id: str
    estimate: StateEstimate
    status: TrackStatus
    existence_probability: float | None
    last_measurement_time_s: float
    source_sensor_ids: frozenset[str]

    ###################################################################################################
    # This function provides the saved estimate's time for consumers reading track results.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - float: Snapshot time in seconds.
    @property
    def timestamp_s(self) -> float:
        """Return the snapshot event time in seconds.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        # Snapshot time is the timestamp of the state estimate. May differ from last_measurement_time_s 
        # (e.g. if the state has been predicted forward since its most recent detection)
        return self.estimate.timestamp_s
    ###################################################################################################

    ###################################################################################################
    # This function provides the saved tracked values for display or export.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - float64 NumPy array: The snapshot's existing values, shape (n,), in coordinate order and
    #   units; no new copy.
    @property
    def state(self) -> FloatArray:
        """Return a detached semantic estimate; mean (n,), covariance (n,n).

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        return self.estimate.mean
    ###################################################################################################

    ###################################################################################################
    # This function provides the saved uncertainty for display or export.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - float64 NumPy array: The snapshot's existing uncertainty table, shape (n, n), in paired
    #   coordinate units; no new copy.
    @property
    def covariance(self) -> FloatArray:
        """Return the detached covariance (n,n) in coordinate-product units.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        return self.estimate.covariance
    ###################################################################################################

@dataclass
class Track:
    """One estimator per target; existence remains unset without a calibrated model.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    track_id: str
    estimator: Estimator                            # current estimator owned by this track
    status: TrackStatus = TrackStatus.TENTATIVE     # can be TENTATIVE / CONFIRMED / DELETED
    age: int = 1                                    # number of record() opportunities seen by this track
    score: float = 1.0                              # incremented on hits, decremented on misses
    existence_probability: float | None = None      # optional parameter, used if an existence model in use
    hit_count: int = 1                              # total successful associations
    miss_count: int = 0                             # current consecutive miss streak
    last_measurement_time_s: float = 0.0            # timestamp of the most recent associated measurement
    source_sensor_ids: set[str] = field(default_factory=set)    # sensors that have contributed measurements to this track
    recent_hits: deque[bool] = field(default_factory=deque)     # hit/miss history used for M-of-N confirmation logic
    history: deque[TrackSnapshot] = field(default_factory=lambda: deque(maxlen=0)) # optional bounded history of past snapshots

    ###################################################################################################
    # This function copies the current track result so consumers can inspect it without changing the
    # live filter.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - TrackSnapshot: Saved ID, status, time, sources, and an estimate with independent read-only arrays.
    def snapshot(self) -> TrackSnapshot:
        """Publish a detached read-only copy of the current target estimate.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """

        state = self.estimator.state

        # Never expose the estimator's live state directly. While the tracker continues to mutate its 
        # estimate during prediction and update, a snapshot should remain an immutable record of what 
        # was believed at the instant of "taking" the snapshot
        return TrackSnapshot(
            self.track_id,

            # Create detached StateEstimate so visualiser/logs cannot modify the live estimator
            StateEstimate(state.mean, state.covariance, state.timestamp_s, state.state_space),
           
            self.status,
            self.existence_probability,
            self.last_measurement_time_s,

            # Over time, the live Track may have new sensors contributing over time, which is why 
            # Track uses a mutable set. However, for this TrackSnapshot, sensor source is exposed
            # as an immutable set.
            frozenset(self.source_sensor_ids), # immutable
        )
    ###################################################################################################