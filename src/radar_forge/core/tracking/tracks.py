"""Tracks: the tracker's live record of each target, and read-only copies of it.

A track is the tracker's belief that one target exists, together with the
filter that estimates where it is. This module has three names:

- :class:`TrackStatus`: where a track is in its life: tentative, confirmed or
  deleted.
- :class:`Track`: the live, changeable record. The tracker owns it, and it
  changes every scan as the filter predicts and updates.
- :class:`TrackSnapshot`: a frozen copy of one track at one moment. This is
  what the tracker hands to callers. Changing it cannot change the tracker.

``kalman.py`` has its own ``Track`` and its own ``TrackStatus``, a ``Literal``
that also has a ``"coasting"`` value. The two sets live side by side until the
pipeline moves to this tracker and ``kalman.py`` is removed. Until then the
package ``__init__`` re-exports ``kalman``'s, so import these from this module.

References
----------
.. [1] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking
       Systems*, Artech House, 1999, ch. 6 (track initiation, confirmation and
       deletion).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.estimation import Estimator

__all__ = [
    "Track",
    "TrackSnapshot",
    "TrackStatus",
]


class TrackStatus(StrEnum):
    """Where a track is in its life.

    Attributes
    ----------
    TENTATIVE
        Newly started and not yet trusted. It becomes confirmed once it has
        enough hits (the M of M-of-N), or is deleted.
    CONFIRMED
        Has passed the M-of-N test. Callers usually display only these.
    DELETED
        Removed. The tracker reports a track with this status once, in the
        scan it is deleted, and then forgets it.
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


@dataclass(frozen=True)
class TrackSnapshot:
    """A frozen copy of one track at one moment.

    Attributes
    ----------
    track_id : str
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
    """

    track_id: str
    estimate: StateEstimate
    status: TrackStatus
    last_measurement_time_s: float
    source_sensor_ids: frozenset[str]

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
class Track:
    """The live record of one target: its filter and its lifecycle counters.

    Each track owns its own estimator. The counters are what
    :class:`~radar_forge.core.tracking.lifecycle.TrackManager` reads to decide
    whether to confirm or delete the track.

    Attributes
    ----------
    track_id : str
        The track's identifier, unique within one tracker.
    estimator : Estimator
        The filter that holds this track's state estimate. No other track may
        share it.
    status : TrackStatus, default TENTATIVE
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

    track_id: str
    estimator: Estimator
    status: TrackStatus = TrackStatus.TENTATIVE
    n_frames: int = 1
    n_hits: int = 1
    n_misses: int = 0
    last_measurement_time_s: float = 0.0
    source_sensor_ids: set[str] = field(default_factory=set)
    history: deque[TrackSnapshot] = field(default_factory=lambda: deque(maxlen=0))

    def snapshot(self) -> TrackSnapshot:
        """Return a frozen copy of the track as it is now.

        Returns
        -------
        TrackSnapshot
            The track's identifier, status, last measurement time and source
            sensors, and a copy of its current estimate. The copy's arrays are
            read-only and do not share memory with the filter.
        """
        state = self.estimator.state
        # Build a new StateEstimate, which copies the arrays, so that a caller
        # who changes the snapshot cannot reach the filter, and a later
        # predict or update cannot change a snapshot already handed out.
        return TrackSnapshot(
            self.track_id,
            StateEstimate(state.mean, state.covariance, state.timestamp_s, state.state_layout),
            self.status,
            self.last_measurement_time_s,
            frozenset(self.source_sensor_ids),
        )
