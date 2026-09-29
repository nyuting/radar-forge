"""Configurable M-of-N lifecycle policy, independent of geometry and filtering.

Classes:
- LifecyclePolicy
- TrackManager

References
----------
.. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
.. [2] Adapted local tracker; spec/tracker-001-provenance.md.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from radar_forge.tracking.estimation import Estimator
from radar_forge.tracking.initiation import TrackInitiator
from radar_forge.tracking.measurements import MeasurementModel
from radar_forge.tracking.sensors import Measurement
from radar_forge.tracking.tracks import Track, TrackStatus

__all__ = [
    "LifecyclePolicy",
    "TrackManager",
]


@dataclass(frozen=True)
class LifecyclePolicy:
    """M-of-N confirmation, consecutive misses and event-time expiry settings.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    # Defaults:
    confirmation_hits: int = 3      # Track must be detected at least 3 times...
    confirmation_window: int = 5    # ...within the last 5 measurement opportunities
    deletion_misses: int = 5        # Delete a confirmed track after 5 consecutive misses
    max_coast_time_s: float = 30.0  # Delete a track if it has not been updated for 30 seconds
    history_size: int = 100         # Keep up to 100 past track snapshots for history / debugging

    ###################################################################################################
    # This function checks the stored rules for confirming, remembering, and removing tracks.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - None; rejects invalid hit/miss counts, history length, or maximum time without a reading.
    def __post_init__(self) -> None:
        """Validate the documented constructor contract."""
        import math

        if (
            any(
                type(v) is not int
                for v in (
                    self.confirmation_hits,
                    self.confirmation_window,
                    self.deletion_misses,
                    self.history_size,
                )
            )
            or not 1 <= self.confirmation_hits <= self.confirmation_window
            or self.deletion_misses < 1
            or self.history_size < 0
            or not math.isfinite(self.max_coast_time_s)
            or self.max_coast_time_s <= 0
        ):
            raise ValueError("invalid confirmation, deletion, coast or history policy")
    ###################################################################################################

class TrackManager:
    """Own initiation routing, bounded history and lifecycle counters.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """
    ###################################################################################################
    # This function sets up track creation and the rules for deciding when tracks are reliable or
    # should end.
    #
    # Inputs:
    # - initiator (TrackInitiator): Rule for creating a filter from a new reading.
    # - policy (LifecyclePolicy or None): Confirmation, removal, and history limits; None uses
    #   defaults.
    #
    # Outputs:
    # - None; stores these components and starts the track ID counter at one.
    def __init__(self, initiator: TrackInitiator, policy: LifecyclePolicy | None = None) -> None:
        self.initiator = initiator
        self.policy = policy or LifecyclePolicy()
        self._next_id = 1   # Track ID generator, monotonically incrementing by 1
    ###################################################################################################

    ###################################################################################################
    # This function asks the configured starter for a new filter and gives it a track ID when
    # successful.
    #
    # Inputs:
    # - measurement (Measurement): Sensor readings, their uncertainty, time, and source IDs.
    # - model (MeasurementModel): Rule for turning tracked values into expected sensor readings.
    #
    # Outputs:
    # - Track or None: New track ready for the engine to register; None if the reading cannot
    #   start one.
    def create(self, measurement: Measurement, model: MeasurementModel) -> Track | None:
        """Initiate a fresh track when the measurement route supports a birth.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        # Check with this TrackManager instance's configured TrackInitiator whether this measurement 
        # can start a track
        estimator = self.initiator.initiate(measurement, model)

        # Some measurements may not contain enough information for track birth
        if estimator is None:
            return None

        # If the initiator successfully creates an estimator from a measurement,
        # wrap the new estimator in a Track object (see seed function below)
        return self.seed(estimator, measurement.sensor_id)
    ###################################################################################################

    ###################################################################################################
    # This function wraps a supplied filter in a new track with an ID, initial hit, and limited history.
    #
    # Inputs:
    # - estimator (Estimator): Filter already holding the starting values, uncertainty, and time.
    # - sensor_id (str or None): Starting sensor ID to record, if supplied.
    #
    # Outputs:
    # - Track: New track holding the supplied filter; the manager also advances its ID counter.
    def seed(self, estimator: Estimator, sensor_id: str | None = None) -> Track:
        """Create a target hypothesis from an externally justified prior.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        # Create a new track around an already initialised estimator
        track = Track(
            str(self._next_id),  # track_id
            estimator,           # estimator
            last_measurement_time_s=estimator.state.timestamp_s
            # `recent_hits` initialised below
            # `history` initialised below
            # other fields will use defaults noted in tracks.py
        )

        # Increment to the next unique track ID
        self._next_id += 1

        # Newborn tracks start with 1 successful detection, so we use a queue with one element
        # E.g. recent_hits = [True, True, False, False, True] -> passes M-of-N (3 of 5) -> confirm
        track.recent_hits = deque([True], maxlen=self.policy.confirmation_window)

        # Limit stored history according to the lifecycle policy
        # E.g. history = [TrackSnapshot(..., TrackStatus="Confirmed",...), TrackSnapshot(...), ...]
        track.history = deque(maxlen=self.policy.history_size)

        # Record the originating sensor if known
        if sensor_id:
            track.source_sensor_ids.add(sensor_id)

        # Check whether the seeded track satisfies the confirmation rule (M-of-N)
        self._confirm(track) # any policy that is "1-of-N" (single hit confirmation) should be confirmed

        return track
    ###################################################################################################

    ############################################## HELPER ##############################################
    # This function marks a stored track as confirmed once enough recent sensor scans have matched
    # it.
    #
    # Inputs:
    # - track (Track): Track whose recent hits are checked against the stored policy.
    #
    # Outputs:
    # - None; changes the track status when the confirmation rule is met.
    def _confirm(self, track: Track) -> None:
        # E.g. sum([True, True, False, False, True]) = 3 >= 3, so the track is CONFIRMED
        if sum(track.recent_hits) >= self.policy.confirmation_hits:
            track.status = TrackStatus.CONFIRMED
    ###################################################################################################

    ###################################################################################################
    # This function records a matched reading or missed scan so track confirmation and removal can
    # be decided.
    #
    # Inputs:
    # - track (Track): Track whose counters and source information will change.
    # - measurement (Measurement or None): Matched reading; None records one missed scan.
    #
    # Outputs:
    # - None; updates hit/miss counters, score, and recent history; a hit may confirm the track.
    def record(self, track: Track, measurement: Measurement | None = None) -> None:
        """Record one eligible hit or miss and update confirmation counters.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        # Count another measurement opportunity
        track.age += 1

        # Store whether this opportunity resulted in a hit or miss
        track.recent_hits.append(measurement is not None)

        if measurement is not None:                                 # IF measurement received:
            track.hit_count += 1                                      # successful association
            track.miss_count = 0                                      # consecutive miss streak is broken
            track.score += 1                                          # simple quality score used by local track management
            track.last_measurement_time_s = measurement.timestamp_s   # remember when last measurement arrived
            track.source_sensor_ids.add(measurement.sensor_id)        # track which sensor it came from
            self._confirm(track)                                      # check whether the track can now be confirmed 
        else:                                                       # ELSE:
            track.miss_count += 1                                     # no measurement associated with this cycle
            track.score -= 1                                          # penalise track quality for missed detection
    ###################################################################################################

    ###################################################################################################
    # This function marks a track for removal after too many misses or too long without a reading.
    #
    # Inputs:
    # - track (Track): Track checked against the stored removal rules.
    # - time_s (float): Current event time in seconds, used to measure the gap since its last
    #   reading.
    #
    # Outputs:
    # - None; may mark the track deleted. The engine removes it from the active list.
    def expire(self, track: Track, time_s: float) -> None:
        """Delete after the miss limit or a gap greater than max_coast_time_s.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        # Delete tracks that have missed too many updates or coasted for too long
        if (
            track.miss_count >= self.policy.deletion_misses
            or time_s - track.last_measurement_time_s > self.policy.max_coast_time_s
        ):
            track.status = TrackStatus.DELETED
    ###################################################################################################