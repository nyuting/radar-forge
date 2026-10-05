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
- **Coast** a confirmed track through missed scans: keep it alive and predict
  it forward with no update. Delete it after ``n_delete_misses`` misses in a
  row, or once more than ``max_coast_time_s`` has passed since its last
  measurement. A tentative track gets no such grace: the M-of-N deadline above
  allows it at most N - M misses in all.

The two classes split the work. :class:`LifecyclePolicy` holds the numbers.
:class:`TrackManager` applies them to :class:`~radar_forge.core.tracking.tracks.Track`
objects and hands out track identifiers. Neither looks at geometry or at the
filter's maths.

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

from radar_forge.core.tracking.estimation import Estimator
from radar_forge.core.tracking.initiation import TrackInitiator
from radar_forge.core.tracking.measurement_models import Measurement, MeasurementModel
from radar_forge.core.tracking.tracks import Track, TrackStatus

__all__ = [
    "LifecyclePolicy",
    "TrackManager",
]


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
        Misses in a row after which a track is deleted. It is what limits how
        long a confirmed track coasts. A tentative track meets the M-of-N
        deadline first unless this is smaller than N - M + 1.
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
        confirmed. If ``n_delete_misses`` is less than 1, ``n_history`` is
        negative, or ``max_coast_time_s`` is not finite and positive.
    """

    n_confirm_hits: int = 3
    n_confirm_frames: int = 5
    n_delete_misses: int = 5
    max_coast_time_s: float = 30.0
    n_history: int = 100

    def __post_init__(self) -> None:
        """Reject a policy that could not work, naming the field at fault."""
        # A bool is an int in Python, so test the exact type to keep True out.
        if any(
            type(value) is not int
            for value in (
                self.n_confirm_hits,
                self.n_confirm_frames,
                self.n_delete_misses,
                self.n_history,
            )
        ):
            counts = {
                "n_confirm_hits": self.n_confirm_hits,
                "n_confirm_frames": self.n_confirm_frames,
                "n_delete_misses": self.n_delete_misses,
                "n_history": self.n_history,
            }
            msg = f"lifecycle counts must be int; got {counts}."
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
        if self.n_history < 0:
            msg = f"n_history must be zero or more; got {self.n_history}."
            raise ValueError(msg)
        if not (math.isfinite(self.max_coast_time_s) and self.max_coast_time_s > 0):
            msg = f"max_coast_time_s must be finite and positive; got {self.max_coast_time_s}."
            raise ValueError(msg)


class TrackManager:
    """Start tracks, count their hits and misses, and confirm or delete them.

    In Stone Soup's terms this one class plays two parts. It is the
    *initiator*: :meth:`create` turns a measurement no track took into a new
    tentative track, and M-of-N confirmation is what Stone Soup's
    ``MultiMeasurementInitiator`` does. It is also the *deleters*: the
    ``n_delete_misses`` rule is ``UpdateTimeStepsDeleter`` and the
    ``max_coast_time_s`` rule is ``UpdateTimeDeleter``.

    Parameters
    ----------
    initiator : TrackInitiator
        Builds a new filter from a measurement, or says that the measurement
        cannot start a track.
    policy : LifecyclePolicy or None, optional
        The confirmation and deletion numbers. None uses the defaults.

    Notes
    -----
    Track identifiers are ``"1"``, ``"2"``, … in order of creation, and are
    never reused within one manager.

    References
    ----------
    .. [1] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking
           Systems*, Artech House, 1999, ch. 6.
    """

    def __init__(self, initiator: TrackInitiator, policy: LifecyclePolicy | None = None) -> None:
        self.initiator = initiator
        self.policy = policy or LifecyclePolicy()
        self._next_id = 1

    def create(self, measurement: Measurement, model: MeasurementModel) -> Track | None:
        """Start a new tentative track from a measurement, if it can start one.

        Parameters
        ----------
        measurement : Measurement
            A measurement that no track took.
        model : MeasurementModel
            The model that measurement came through.

        Returns
        -------
        Track or None
            The new track, which the caller must add to its list. None when the
            initiator cannot build a filter from this measurement, for example
            because the measurement does not pin down enough of the state.
        """
        estimator = self.initiator.initiate(measurement, model)
        if estimator is None:
            return None
        return self.seed(estimator, measurement.sensor_id)

    def seed(self, estimator: Estimator, sensor_id: str | None = None) -> Track:
        """Wrap a filter that already holds a starting estimate in a new track.

        The new track has one scan and one hit: the scan it was born in. With a
        1-of-N policy that is enough, and the track starts confirmed.

        Parameters
        ----------
        estimator : Estimator
            The track's own filter, holding its starting state, covariance and
            time.
        sensor_id : str or None, optional
            The sensor the starting estimate came from, recorded in
            ``Track.source_sensor_ids``. None records no sensor.

        Returns
        -------
        Track
            The new track, with the next free identifier.
        """
        track = Track(
            str(self._next_id),
            estimator,
            last_measurement_time_s=estimator.state.timestamp_s,
            history=deque(maxlen=self.policy.n_history),
        )
        self._next_id += 1
        if sensor_id:
            track.source_sensor_ids.add(sensor_id)
        self._apply_m_of_n(track)
        return track

    def record(self, track: Track, measurement: Measurement | None = None) -> None:
        """Count one scan for a track, as a hit or a miss, and apply the rules.

        Call it once per scan in which the track could have been detected.
        Scans from a sensor that cannot see the track are not counted.

        Parameters
        ----------
        track : Track
            The track to update. Its counters and status change in place.
        measurement : Measurement or None, optional
            The measurement the track was updated with this scan, or None for a
            miss.

        Notes
        -----
        A hit resets the miss count and records the measurement's time and
        sensor. Then a tentative track is confirmed, deleted, or left
        tentative by the M-of-N rule (see the module docstring), and any track
        with ``n_delete_misses`` misses in a row is deleted.
        """
        track.n_frames += 1
        if measurement is None:
            track.n_misses += 1
        else:
            track.n_hits += 1
            track.n_misses = 0
            track.last_measurement_time_s = measurement.timestamp_s
            track.source_sensor_ids.add(measurement.sensor_id)
        self._apply_m_of_n(track)
        if track.n_misses >= self.policy.n_delete_misses:
            track.status = TrackStatus.DELETED

    def expire(self, track: Track, time_s: float) -> None:
        """Delete a track that has gone too long without a measurement.

        Parameters
        ----------
        track : Track
            The track to check. Its status may change to DELETED; the caller
            removes deleted tracks from its list.
        time_s : float
            The current scan time, seconds.

        Notes
        -----
        The track is deleted when ``time_s - track.last_measurement_time_s``
        is greater than ``policy.max_coast_time_s``. A gap of exactly the limit
        is allowed.
        """
        if time_s - track.last_measurement_time_s > self.policy.max_coast_time_s:
            track.status = TrackStatus.DELETED

    def _apply_m_of_n(self, track: Track) -> None:
        """Confirm or delete a tentative track by the M-of-N rule."""
        if track.status != TrackStatus.TENTATIVE:
            return
        m_hits = self.policy.n_confirm_hits
        n_frames = self.policy.n_confirm_frames
        if track.n_hits >= m_hits:
            track.status = TrackStatus.CONFIRMED
        # Even a hit in every remaining scan of the first N would leave it short
        # of M, so the track is already lost. Deleting it now, not at scan N,
        # stops a false alarm from competing for measurements it can never be
        # confirmed on. Once the N scans are used up, no scans remain, so this
        # also covers the deadline.
        elif track.n_hits + max(n_frames - track.n_frames, 0) < m_hits:
            track.status = TrackStatus.DELETED
