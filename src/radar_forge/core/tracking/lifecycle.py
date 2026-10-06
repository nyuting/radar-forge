"""Track lifecycle: when a track is started, confirmed and deleted.

A tracker cannot tell a real target from a false alarm (clutter) after one
measurement. So every new track starts *tentative*, and has to earn
confirmation. This module applies the M-of-N rule from Blackman & Popoli:

- **Confirm** a tentative track once it has M hits (``n_confirm_hits``) within
  its *first* N scans (``n_confirm_frames``). A hit is a scan in which the
  track was given a measurement. The scan that started the track counts as
  its first scan and its first hit.
- **Delete** a tentative track as soon as M hits can no longer be reached in
  those N scans, and at the latest when the N scans are used up. The window
  does not slide. A track that hasn't proved itself in its first N scans is
  treated as clutter, however often it is refreshed later.
- **Coast** a confirmed track through missed scans: keep it alive, with status
  ``"coasting"``, and predict it forward with no update. A hit makes it
  ``"confirmed"`` again.
- **Delete** any track after ``n_delete_misses`` misses in a row. A confirmed
  track may be given ``n_reacquire_frames`` more. Separately, delete any
  track, tentative or confirmed, once more than ``max_coast_time_s`` has
  passed since its last measurement.

The two classes split the work. :class:`LifecyclePolicy` holds the numbers.
:class:`TrackManager` applies them to :class:`~radar_forge.core.tracking.tracks.Track`
objects and hands out track identifiers. Neither looks at geometry or at the
filter's maths.

Examples
--------
A worked example of the M-of-N rule at the defaults, M = 3 hits in the first
N = 5 scans. Scan 1 is the birth, so it is always a hit.

==========  =====  =====  =====  =====  =====
scan          1      2      3      4      5
==========  =====  =====  =====  =====  =====
hit/miss     hit   miss   hit    miss   hit
n_hits        1      1      2      2      3
status       tent.  tent.  tent.  tent.  conf.
==========  =====  =====  =====  =====  =====

The third hit comes in scan 5, the last of the first N, so the track is
confirmed. Now take hit, miss, miss, miss. After scan 3 the track has 1 hit and
2 scans left, and 1 + 2 = 3 can still reach M, so it lives. After scan 4 it has
1 hit and 1 scan left, and 1 + 1 = 2 cannot, so it is deleted at scan 4,
without waiting for scan 5.

References
----------
.. [1] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking
       Systems*, Artech House, 1999, ch. 6 (M-of-N track initiation and track
       deletion).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TypeVar

from radar_forge.core.tracking.tracks import Track

__all__ = [
    "LifecyclePolicy",
    "TrackManager",
]

FilterT = TypeVar("FilterT")


@dataclass(frozen=True)
class LifecyclePolicy:
    """The numbers that decide when a track is confirmed or deleted.

    Parameters
    ----------
    n_confirm_hits : int, default 3
        M: hits a tentative track needs to be confirmed.
    n_confirm_frames : int, default 5
        N: the number of scans, counted from the track's birth, in which those
        M hits must come. A tentative track is deleted once it can no longer
        reach M hits within them.
    n_delete_misses : int, default 5
        Misses in a row after which a track is deleted. A tentative track
        meets the M-of-N deadline first unless this is smaller than N - M + 1.
    n_reacquire_frames : int, default 0
        Extra misses in a row that a confirmed track survives beyond
        ``n_delete_misses``. :class:`~radar_forge.core.tracking.kalman.KalmanTracker`
        uses them to re-acquire a target whose track has lost it. 0 deletes
        every track at ``n_delete_misses``.
    max_coast_time_s : float, default 30.0
        Seconds since a track's last measurement after which it is deleted,
        whatever its status. A gap of exactly this long is still allowed.
    n_history : int, default 100
        Number of past snapshots each track keeps in ``Track.history``, for
        plotting and debugging. 0 keeps none.

    Raises
    ------
    ValueError
        If a count is not an ``int``. If ``n_confirm_hits`` is less than 1 or
        more than ``n_confirm_frames``, so that no track could ever be
        confirmed. If ``n_delete_misses`` is less than 1,
        ``n_reacquire_frames`` or ``n_history`` is negative, or
        ``max_coast_time_s`` is not finite and positive.
    """

    n_confirm_hits: int = 3
    n_confirm_frames: int = 5
    n_delete_misses: int = 5
    n_reacquire_frames: int = 0
    max_coast_time_s: float = 30.0
    n_history: int = 100

    def __post_init__(self) -> None:
        """Reject a policy that could not work, naming the field at fault."""
        counts = {
            "n_confirm_hits": self.n_confirm_hits,
            "n_confirm_frames": self.n_confirm_frames,
            "n_delete_misses": self.n_delete_misses,
            "n_reacquire_frames": self.n_reacquire_frames,
            "n_history": self.n_history,
        }
        # A bool is an int in Python, so test the exact type to keep True out.
        not_int = {name: value for name, value in counts.items() if type(value) is not int}
        if not_int:
            msg = f"lifecycle counts must be int; got {not_int}."
            raise ValueError(msg)
        if not 1 <= self.n_confirm_hits <= self.n_confirm_frames:
            msg = (
                f"n_confirm_hits ({self.n_confirm_hits}) must be at least 1 and at most "
                f"n_confirm_frames ({self.n_confirm_frames}), or no track could be confirmed."
            )
            raise ValueError(msg)
        if self.n_delete_misses < 1:
            msg = f"n_delete_misses must be at least 1; got {self.n_delete_misses}."
            raise ValueError(msg)
        if self.n_reacquire_frames < 0:
            msg = f"n_reacquire_frames must be zero or more; got {self.n_reacquire_frames}."
            raise ValueError(msg)
        if self.n_history < 0:
            msg = f"n_history must be zero or more; got {self.n_history}."
            raise ValueError(msg)
        if not (math.isfinite(self.max_coast_time_s) and self.max_coast_time_s > 0):
            msg = f"max_coast_time_s must be finite and positive; got {self.max_coast_time_s}."
            raise ValueError(msg)


class TrackManager:
    """Hand out track identifiers, count hits and misses, and confirm or delete.

    The manager owns the lifecycle and nothing else. It never predicts, gates
    or updates a filter; the tracker that holds it does that, and tells the
    manager what happened with :meth:`record_hit` and :meth:`record_miss`.

    Parameters
    ----------
    policy : LifecyclePolicy or None, optional
        The confirmation and deletion numbers. None uses the defaults.

    Notes
    -----
    Track identifiers are 1, 2, … in order of creation, and are never reused
    within one manager.

    In Stone Soup's terms the manager plays the deleters: the
    ``n_delete_misses`` rule is ``UpdateTimeStepsDeleter`` and the
    ``max_coast_time_s`` rule is ``UpdateTimeDeleter``. Its M-of-N
    confirmation is what Stone Soup's ``MultiMeasurementInitiator`` does.

    References
    ----------
    .. [1] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking
           Systems*, Artech House, 1999, ch. 6.
    """

    def __init__(self, policy: LifecyclePolicy | None = None) -> None:
        self.policy = policy or LifecyclePolicy()
        self._next_id = 1

    def seed(
        self, estimator: FilterT, time_s: float, sensor_id: str | None = None
    ) -> Track[FilterT]:
        """Wrap a filter that already holds a starting estimate in a new track.

        The new track has one scan and one hit: the scan it was born in. With a
        1-of-N policy that is enough, and the track starts confirmed.

        Parameters
        ----------
        estimator : FilterT
            The track's own filter, holding its starting estimate.
        time_s : float
            Time of the measurement or cue the track starts from, seconds.
        sensor_id : str or None, optional
            The sensor the starting estimate came from, recorded in
            ``Track.source_sensor_ids``. None records no sensor.

        Returns
        -------
        Track
            The new track, with the next free identifier. The caller adds it
            to its own list of tracks.
        """
        track = Track(
            self._next_id,
            estimator,
            last_measurement_time_s=time_s,
            history=deque(maxlen=self.policy.n_history),
        )
        self._next_id += 1
        if sensor_id:
            track.source_sensor_ids.add(sensor_id)
        self._apply_m_of_n(track)
        return track

    def record_hit(
        self, track: Track[FilterT], time_s: float, sensor_id: str | None = None
    ) -> None:
        """Count a scan in which the track was updated with a measurement.

        A hit resets the miss count, records the measurement's time and
        sensor, makes a coasting track confirmed again, and may confirm a
        tentative one by the M-of-N rule.

        Parameters
        ----------
        track : Track
            The track to update. Its counters and status change in place.
        time_s : float
            Time of the measurement, seconds.
        sensor_id : str or None, optional
            The sensor the measurement came from. None records no sensor.
        """
        track.n_frames += 1
        track.n_hits += 1
        track.n_misses = 0
        track.last_measurement_time_s = time_s
        if sensor_id:
            track.source_sensor_ids.add(sensor_id)
        if track.status == "coasting":
            track.status = "confirmed"
        self._apply_m_of_n(track)

    def record_miss(self, track: Track[FilterT]) -> None:
        """Count a scan in which the track could have been detected but was not.

        Call it only for a scan from a sensor that can see the track. A miss
        makes a confirmed track coast, may delete a tentative one by the M-of-N
        rule, and deletes any track that has missed too many scans in a row:
        ``n_delete_misses`` for a tentative track, and ``n_reacquire_frames``
        more for a confirmed one.

        Parameters
        ----------
        track : Track
            The track to update. Its counters and status change in place.
        """
        track.n_frames += 1
        track.n_misses += 1
        if track.status == "confirmed":
            track.status = "coasting"
        self._apply_m_of_n(track)
        allowed = self.policy.n_delete_misses
        if track.is_confirmed:
            allowed += self.policy.n_reacquire_frames
        if track.n_misses >= allowed:
            track.status = "deleted"

    def expire(self, track: Track[FilterT], time_s: float) -> None:
        """Delete a track that has gone too long without a measurement.

        Parameters
        ----------
        track : Track
            The track to check. Its status may change to ``"deleted"``; the
            caller removes deleted tracks from its list.
        time_s : float
            The current scan time, seconds.

        Notes
        -----
        The track is deleted when ``time_s - track.last_measurement_time_s``
        is greater than ``policy.max_coast_time_s``. A gap of exactly the limit
        is allowed.
        """
        if time_s - track.last_measurement_time_s > self.policy.max_coast_time_s:
            track.status = "deleted"

    def _apply_m_of_n(self, track: Track[FilterT]) -> None:
        """Confirm or delete a tentative track by the M-of-N rule."""
        if track.status != "tentative":
            return
        m_hits = self.policy.n_confirm_hits
        n_frames = self.policy.n_confirm_frames
        if track.n_hits >= m_hits:
            track.status = "confirmed"
        # Even a hit in every remaining scan of the first N would leave it short
        # of M, so the track is already lost. Deleting it now, not at scan N,
        # stops a false alarm from competing for measurements it can never be
        # confirmed on. Once the N scans are used up, no scans remain, so this
        # also covers the deadline.
        elif track.n_hits + max(n_frames - track.n_frames, 0) < m_hits:
            track.status = "deleted"
