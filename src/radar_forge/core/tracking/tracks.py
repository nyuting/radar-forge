"""Tracks: the tracker's live record of each target, and read-only copies of it.

A track is the tracker's belief that one target exists, together with the
filter that estimates where it is. This module has three names:

- :data:`TrackStatus`: where a track is in its life: tentative, confirmed,
  coasting or deleted, the stages of a logic-based track formation
  procedure [1]_.
- :class:`Track`: the live, changeable record. The tracker owns it, and it
  changes every scan as the filter predicts and updates.
- :class:`TrackSnapshot`: a frozen copy of one track at one moment. This is
  what :class:`~radar_forge.core.tracking.tracker.Tracker` hands to callers.
  Changing it cannot change the tracker.

Both trackers in this package use the same :class:`Track`. Only the filter it
carries differs: a :class:`~radar_forge.core.tracking.tracker.Tracker` track
carries an :class:`~radar_forge.core.tracking.estimation.Estimator` such as the
UKF, and a :class:`~radar_forge.core.tracking.kalman.KalmanTracker` track
carries a :class:`~radar_forge.core.tracking.kalman.KalmanFilter`.

References
----------
.. [1] Y. Bar-Shalom and X. R. Li, *Multitarget-Multisensor Tracking: Principles
       and Techniques*, YBS Publishing, 1995, §2.6.1 (a logic-based track formation
       procedure).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar, get_args

import numpy as np
from numpy.typing import NDArray

from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.estimation import Estimator

__all__ = [
    "TRACK_STATUSES",
    "Track",
    "TrackSnapshot",
    "TrackStatus",
]

TrackStatus = Literal["tentative", "confirmed", "coasting", "deleted"]
"""Where a track is in its life.

``"tentative"``
    Newly started and not yet trusted. It becomes confirmed by the M-of-N
    rule: M hits (scans in which it got a measurement) within its first N
    scans. If it cannot reach M hits in time, it is deleted. See
    :mod:`~radar_forge.core.tracking.lifecycle`.
``"confirmed"``
    Has passed the M-of-N test, and got a measurement in its latest scan.
``"coasting"``
    Confirmed, but missed its latest scan. To *coast* is to carry a track
    through a scan with no measurement: the filter predicts it forward and
    there is nothing to update it with. The next hit makes it confirmed again.
``"deleted"``
    Removed. The tracker reports a track with this status once, in the scan
    it is deleted, and then forgets it.
"""

TRACK_STATUSES: tuple[TrackStatus, ...] = get_args(TrackStatus)
"""Every :data:`TrackStatus`, in the order a track moves through them."""

FilterT = TypeVar("FilterT")


@dataclass(frozen=True)
class TrackSnapshot:
    """A frozen copy of one track at one moment.

    Attributes
    ----------
    track_id : int
        The track's identifier, unique within one tracker and fixed for the
        track's life.
    estimate : StateEstimate
        The state estimate at the moment the snapshot was taken: mean
        ``(n_state,)`` and covariance ``(n_state, n_state)``.
    status : TrackStatus
        The track's status at that moment.
    last_measurement_time_s : float
        Time of the last measurement the track was updated with, seconds. It is
        earlier than :attr:`timestamp_s` when the track has been predicted
        forward through scans with no measurement.
    source_sensor_ids : frozenset of str
        Every sensor that has given the track a measurement, including the one
        it was born from.
    n_hits : int
        Scans in which the track was updated with a measurement, counting its
        birth.
    n_misses : int
        Scans in a row without a measurement.
    measurement_index : int or None
        The index, into the scan's ``MeasurementBatch.measurements``, of the
        measurement the track was updated with, or born from, in the scan the
        snapshot reports. ``None`` if it took none.
    nis : float or None
        The normalised innovation squared of that update, computed before it
        [1]_. ``None`` if the track took no measurement, or was born, since a
        birth has no prediction to compare against.

    Notes
    -----
    Stone Soup carries the same thing on the track's state instead: each
    ``GaussianStateUpdate`` holds the ``hypothesis`` that made it, with its
    ``measurement`` and ``measurement_prediction``. A snapshot holds the
    index and the NIS only, which is what ``tracks.csv`` needs (data-001
    §6.6).

    References
    ----------
    .. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with
           Applications to Tracking and Navigation*, Wiley, 2001, §5.4.2
           (the NIS test).
    """

    track_id: int
    estimate: StateEstimate
    status: TrackStatus
    last_measurement_time_s: float
    source_sensor_ids: frozenset[str]
    n_hits: int
    n_misses: int
    measurement_index: int | None
    nis: float | None

    @property
    def timestamp_s(self) -> float:
        """Time of the snapshot's estimate, seconds."""
        return self.estimate.timestamp_s

    @property
    def state(self) -> NDArray[np.float64]:
        """The estimate's mean, shape ``(n_state,)``.

        Only the mean: the covariance is :attr:`covariance`. The array is the
        snapshot's own read-only copy, in the state layout's order and units.
        """
        return self.estimate.mean

    @property
    def covariance(self) -> NDArray[np.float64]:
        """The estimate's covariance, shape ``(n_state, n_state)``.

        The snapshot's own read-only copy. Entry ``(i, j)`` has the unit of
        coordinate ``i`` times the unit of coordinate ``j``.
        """
        return self.estimate.covariance


@dataclass
class Track(Generic[FilterT]):
    """The live record of one target: its filter and its lifecycle counters.

    Each track owns its own filter. The counters are what
    :class:`~radar_forge.core.tracking.lifecycle.TrackManager` reads to decide
    whether to confirm or delete the track.

    ``Track`` is generic in its filter, written ``Track[FilterT]``. The
    lifecycle does not depend on how the state is estimated, so one record
    serves both trackers: ``Track[Estimator]`` in
    :class:`~radar_forge.core.tracking.tracker.Tracker` and
    ``Track[KalmanFilter]`` in
    :class:`~radar_forge.core.tracking.kalman.KalmanTracker`.

    Attributes
    ----------
    track_id : int
        The track's identifier, unique within one tracker and fixed for the
        track's life. A track whose identifier changes mid-run is a track that
        was lost and started again.
    estimator : FilterT
        The filter that holds this track's state estimate. No other track may
        share it.
    status : TrackStatus, default "tentative"
        Where the track is in its life.
    n_frames : int, default 1
        Scans this track has taken part in, counting the one it was born in. A
        scan counts only if the track could have been detected in it: a scan
        from a sensor that cannot see the track does not count. The M-of-N rule
        reads this to know how many of its first N scans are used up.
    n_hits : int, default 1
        Scans in which the track was updated with a measurement, counting its
        birth.
    n_misses : int, default 0
        Scans in a row without a measurement. A hit resets it to zero.
    last_measurement_time_s : float, default 0.0
        Time of the last measurement the track was updated with, seconds.
    source_sensor_ids : set of str
        Every sensor that has given the track a measurement.
    history : collections.deque of TrackSnapshot
        The most recent snapshots, one per scan, oldest first. Its length is
        capped by ``LifecyclePolicy.n_history``.
    """

    track_id: int
    estimator: FilterT
    status: TrackStatus = "tentative"
    n_frames: int = 1
    n_hits: int = 1
    n_misses: int = 0
    last_measurement_time_s: float = 0.0
    source_sensor_ids: set[str] = field(default_factory=set)
    history: deque[TrackSnapshot] = field(default_factory=lambda: deque(maxlen=0))

    @property
    def is_alive(self) -> bool:
        """Whether the track still takes part in association: not deleted."""
        return self.status != "deleted"

    @property
    def is_confirmed(self) -> bool:
        """Whether the track has passed the M-of-N test: confirmed or coasting."""
        return self.status in ("confirmed", "coasting")

    def snapshot(
        self: Track[Estimator],
        *,
        measurement_index: int | None = None,
        nis: float | None = None,
    ) -> TrackSnapshot:
        """Return a frozen copy of the track as it is now.

        Only a track whose filter is an
        :class:`~radar_forge.core.tracking.estimation.Estimator` has a
        snapshot, because only an estimator reports a timed
        :class:`~radar_forge.core.tracking.coordinates.StateEstimate`.

        Parameters
        ----------
        measurement_index : int or None, optional
            The index of the measurement the track took in this scan, if any.
            The track does not know it, so the tracker passes it in.
        nis : float or None, optional
            The NIS of that update, if it was one.

        Returns
        -------
        TrackSnapshot
            The track's identifier, status, counters, last measurement time and
            source sensors, the two arguments, and a copy of its current
            estimate. The copy's arrays are read-only and do not share memory
            with the filter.
        """
        state = self.estimator.state
        # Build a new StateEstimate, which copies the arrays. Then a caller who
        # changes the snapshot cannot reach the filter, and a later predict or
        # update cannot change a snapshot already handed out.
        return TrackSnapshot(
            self.track_id,
            StateEstimate(state.mean, state.covariance, state.timestamp_s, state.state_layout),
            self.status,
            self.last_measurement_time_s,
            frozenset(self.source_sensor_ids),
            self.n_hits,
            self.n_misses,
            measurement_index,
            nis,
        )
