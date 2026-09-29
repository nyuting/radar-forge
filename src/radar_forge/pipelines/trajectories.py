"""Target trajectories: from a recorded track to what the radar measures.

A scenario begins with a list of positions and times, and the simulator needs
range, angle and radial velocity on a uniform frame grid. This module is that
bridge, and it is deliberately three separate steps rather than one function,
because each step loses something different and hiding that would teach the
wrong lesson:

:func:`load_flight_csv`
    Reads a trajectory CSV with the ``spec/data-001-formats.md`` §6.3 columns.
    The recorded track carries no altitude, so one is supplied as a scenario
    parameter and held constant.
:func:`resample`
    Linear interpolation onto the frame grid. The fixes arrive every 2-8
    seconds; the frames are wanted at 1 Hz, so most frames are interpolated
    rather than observed.
:func:`to_radar_frame`
    WGS-84 geometry against the radar site, plus a central difference for
    radial velocity.
:func:`to_bistatic_radar_frame`
    The same, for a transmitter and receiver at two different sites. There are
    then two ranges instead of one, and the rate that matters is the bisector
    rate of the *sum* of them.

The velocity is the weakest number in the chain and the one most worth
distrusting. It is a difference of an interpolated range, so it is smoothed
over the several seconds between real fixes: a good estimate of the mean
radial velocity across a frame, not of instantaneous Doppler. Nothing in the
input data cross-checks it, which is why the tests in this package use
synthetic tracks with closed-form answers and exercise the real CSV only for
loading and plumbing.

References
----------
.. [1] National Imagery and Mapping Agency, *Department of Defense World
       Geodetic System 1984*, NIMA TR8350.2, 3rd ed., 2000.
.. [2] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, S8.2 (radial velocity and the Doppler shift).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.core.geodesy import enu_to_range_azimuth_elevation, geodetic_to_enu_m
from radar_forge.core.radar import BistaticRadar, Radar

__all__ = [
    "BistaticTargetTrack",
    "TargetTrack",
    "Trajectory",
    "load_flight_csv",
    "resample",
    "to_bistatic_radar_frame",
    "to_radar_frame",
]

# Column names from spec/data-001-formats.md §6.3.
_TIME_COLUMNS = ("time_utc", "time_s")
_POSITION_COLUMNS = ("latitude_deg", "longitude_deg")
_LEGACY_COLUMNS = ("timestamp", "lat", "lon")


@dataclass(frozen=True)
class Trajectory:
    """A target's position over time, in geodetic coordinates.

    Attributes
    ----------
    time_s : numpy.ndarray
        Seconds since the first fix, shape ``(n_fixes,)``, strictly increasing.
        Relative rather than absolute so that a scenario window is expressed in
        the same units as its frame grid.
    latitude_deg, longitude_deg : numpy.ndarray
        Geodetic position, degrees, shape ``(n_fixes,)``.
    altitude_m : numpy.ndarray
        Height above the WGS-84 ellipsoid, metres, shape ``(n_fixes,)``.
    epoch : datetime.datetime or None
        Absolute time of the first fix, kept so that outputs can be labelled
        with real timestamps. ``None`` for a synthetic trajectory.
    """

    time_s: NDArray[np.float64]
    latitude_deg: NDArray[np.float64]
    longitude_deg: NDArray[np.float64]
    altitude_m: NDArray[np.float64]
    epoch: datetime | None = None

    def __post_init__(self) -> None:
        """Validate shapes and time ordering; see :func:`load_flight_csv`."""
        n_fixes = self.time_s.shape
        for name in ("latitude_deg", "longitude_deg", "altitude_m"):
            if getattr(self, name).shape != n_fixes:
                msg = (
                    f"{name} must have the same shape as time_s {n_fixes}; "
                    f"got {getattr(self, name).shape}."
                )
                raise ValueError(msg)
        if self.time_s.ndim != 1:
            msg = f"time_s must be one-dimensional; got shape {self.time_s.shape}."
            raise ValueError(msg)
        if self.time_s.size < 2:
            msg = f"a trajectory needs at least two fixes; got {self.time_s.size}."
            raise ValueError(msg)
        if not np.all(np.diff(self.time_s) > 0.0):
            msg = (
                "time_s must be strictly increasing; the track has a repeated or "
                "out-of-order timestamp."
            )
            raise ValueError(msg)

    @property
    def n_fixes(self) -> int:
        """Number of position fixes in the trajectory."""
        return int(self.time_s.size)

    @property
    def duration_s(self) -> float:
        """Span from the first fix to the last, in seconds."""
        return float(self.time_s[-1] - self.time_s[0])


@dataclass(frozen=True)
class TargetTrack:
    """What a radar at a fixed site measures of a trajectory.

    Attributes
    ----------
    time_s : numpy.ndarray
        Seconds since the trajectory's first fix, shape ``(n_frames,)``.
    range_m : numpy.ndarray
        Slant range from the radar, metres, shape ``(n_frames,)``.
    azimuth_deg, elevation_deg : numpy.ndarray
        Look angles, degrees, shape ``(n_frames,)``. Azimuth is zero at true
        north and increases clockwise.
    radial_velocity_mps : numpy.ndarray
        Rate of closure, metres/second, shape ``(n_frames,)``. **Positive
        closing**, per ``spec/structure.md`` D5, and smoothed -- see
        :func:`to_radar_frame`.
    """

    time_s: NDArray[np.float64]
    range_m: NDArray[np.float64]
    azimuth_deg: NDArray[np.float64]
    elevation_deg: NDArray[np.float64]
    radial_velocity_mps: NDArray[np.float64]

    @property
    def n_frames(self) -> int:
        """Number of frames in the track."""
        return int(self.time_s.size)


def load_flight_csv(path: Path | str, *, altitude_m: float | None = None) -> Trajectory:
    """Read a trajectory CSV into a :class:`Trajectory`.

    The columns are those of ``spec/data-001-formats.md`` §6.3: exactly one of
    ``time_utc`` and ``time_s``, then ``latitude_deg`` and ``longitude_deg``,
    and optionally ``target_id`` and ``altitude_m``.

    Parameters
    ----------
    path : pathlib.Path or str
        The CSV file, with a header row. ``time_utc`` is ISO 8601, and a
        trailing ``Z`` is accepted. ``time_s`` is seconds, for a synthetic track
        with no absolute epoch.
    altitude_m : float, optional
        Height above the ellipsoid to assign to every fix, metres. Used only
        when the file has no ``altitude_m`` column. The recorded track has
        none, so this is a scenario parameter rather than a measurement -- see
        the ``Notes``.

    Returns
    -------
    Trajectory
        Positions in fix order, with ``time_s`` measured from the first fix.
        ``epoch`` is the first fix's ``time_utc``, or ``None`` for a ``time_s``
        file.

    Raises
    ------
    ValueError
        If the header does not name exactly one time column and both position
        columns, if ``target_id`` names more than one target, if no altitude is
        given by either the file or ``altitude_m``, if the file holds fewer
        than two fixes, or if the times are not strictly increasing. A file
        with the old ``timestamp,lat,lon`` header is rejected with a message
        naming the new columns.

    Notes
    -----
    **Constant altitude is an approximation.** Elevation angle and the
    ground-range-to-slant-range correction are therefore approximate. For
    scenario 001 the correction is 1.6 % at the closest approach (8.26 km
    ground, 8.39 km slant) and shrinks with range.

    One file holds one target. ``target_id`` is accepted so that a file written
    for a later multi-target scenario still reads, but only while it names a
    single target.

    The file is opened with ``newline=""`` so that the :mod:`csv` module handles
    the line endings, which is what lets it read a CRLF file unchanged.

    References
    ----------
    .. [1] ``spec/data-001-formats.md`` §5 (time) and §6.3 (trajectory columns).
    """
    csv_path = Path(path)
    times: list[str] = []
    latitude_deg: list[float] = []
    longitude_deg: list[float] = []
    altitudes_m: list[float] = []
    target_ids: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if columns == _LEGACY_COLUMNS:
            msg = (
                f"{csv_path} has the old header {','.join(_LEGACY_COLUMNS)}; rename it to "
                "time_utc,latitude_deg,longitude_deg (spec/data-001-formats.md §6.3)."
            )
            raise ValueError(msg)
        missing = [name for name in _POSITION_COLUMNS if name not in columns]
        if missing:
            msg = (
                f"{csv_path} is missing required column(s) {missing}; "
                f"expected a header naming {list(_POSITION_COLUMNS)}."
            )
            raise ValueError(msg)
        time_columns = [name for name in _TIME_COLUMNS if name in columns]
        if len(time_columns) != 1:
            msg = (
                f"{csv_path} must name exactly one of {list(_TIME_COLUMNS)}; found {time_columns}."
            )
            raise ValueError(msg)
        time_column = time_columns[0]
        has_altitude = "altitude_m" in columns
        has_target_id = "target_id" in columns
        # A Python loop: csv.DictReader is a row-at-a-time stream, and the whole
        # point of streaming it is not to hold the text of 5422 rows at once.
        for row in reader:
            times.append(row[time_column])
            latitude_deg.append(float(row["latitude_deg"]))
            longitude_deg.append(float(row["longitude_deg"]))
            if has_altitude:
                altitudes_m.append(float(row["altitude_m"]))
            if has_target_id:
                target_ids.add(row["target_id"])

    if len(target_ids) > 1:
        msg = (
            f"{csv_path} names {len(target_ids)} targets {sorted(target_ids)}; "
            "a trajectory file may hold only one."
        )
        raise ValueError(msg)
    if len(times) < 2:
        msg = f"{csv_path} holds {len(times)} fix(es); a trajectory needs at least two."
        raise ValueError(msg)

    epoch: datetime | None
    if time_column == "time_utc":
        stamps = [datetime.fromisoformat(text) for text in times]
        epoch = stamps[0]
        time_s = np.array([(stamp - epoch).total_seconds() for stamp in stamps], dtype=np.float64)
    else:
        epoch = None
        seconds = np.asarray([float(text) for text in times], dtype=np.float64)
        time_s = seconds - seconds[0]

    if has_altitude:
        altitude_per_fix_m = np.asarray(altitudes_m, dtype=np.float64)
    elif altitude_m is None:
        msg = f"{csv_path} has no altitude_m column, so an altitude_m argument is required."
        raise ValueError(msg)
    else:
        altitude_per_fix_m = np.full(time_s.shape, float(altitude_m), dtype=np.float64)
    return Trajectory(
        time_s=time_s,
        latitude_deg=np.asarray(latitude_deg, dtype=np.float64),
        longitude_deg=np.asarray(longitude_deg, dtype=np.float64),
        altitude_m=altitude_per_fix_m,
        epoch=epoch,
    )


def resample(trajectory: Trajectory, times_s: ArrayLike) -> Trajectory:
    """Interpolate a trajectory onto a requested set of times.

    Parameters
    ----------
    trajectory : Trajectory
        The recorded track.
    times_s : array_like
        Times to sample at, seconds since the trajectory's first fix, shape
        ``(n_frames,)``. Must lie within the track's span and be strictly
        increasing.

    Returns
    -------
    Trajectory
        A trajectory on the requested grid, carrying the same ``epoch``.

    Raises
    ------
    ValueError
        If any requested time lies outside the track's span.

    Notes
    -----
    Interpolation is linear in latitude and longitude, which is a straight line
    in the local tangent plane rather than a great circle. Over the 2-8 second
    gaps of a real track that difference is millimetres and is neglected. It
    would not be safe over a long gap, and it does not handle a track crossing
    the antimeridian -- neither applies to scenario 001.

    Requesting a time outside the track raises rather than extrapolating:
    :func:`numpy.interp` would silently hold the endpoint value, producing a
    stationary target that looks like a real measurement.
    """
    requested_s = np.asarray(times_s, dtype=np.float64)
    first_s = float(trajectory.time_s[0])
    last_s = float(trajectory.time_s[-1])
    if requested_s.size and (requested_s.min() < first_s or requested_s.max() > last_s):
        msg = (
            f"requested times span [{requested_s.min()}, {requested_s.max()}] s, outside the "
            f"track's [{first_s}, {last_s}] s; extrapolating a trajectory is not supported."
        )
        raise ValueError(msg)

    return Trajectory(
        time_s=requested_s,
        latitude_deg=np.interp(requested_s, trajectory.time_s, trajectory.latitude_deg),
        longitude_deg=np.interp(requested_s, trajectory.time_s, trajectory.longitude_deg),
        altitude_m=np.interp(requested_s, trajectory.time_s, trajectory.altitude_m),
        epoch=trajectory.epoch,
    )


def to_radar_frame(trajectory: Trajectory, radar: Radar) -> TargetTrack:
    r"""Express a trajectory as the range, angles and closing rate a radar sees.

    Parameters
    ----------
    trajectory : Trajectory
        The target's positions, already on the frame grid.
    radar : radar_forge.core.radar.Radar
        The sited radar; its latitude, longitude and altitude are the origin of
        the local tangent plane.

    Returns
    -------
    TargetTrack
        Range, azimuth, elevation and radial velocity, shape ``(n_frames,)``
        each.

    Notes
    -----
    Radial velocity is minus the central difference of slant range,

    .. math:: v_r = -\frac{\mathrm{d}R}{\mathrm{d}t},

    the sign being what makes **closing positive**, per ``spec/structure.md``
    D5: a target approaching has a decreasing range and must report a positive
    velocity. :func:`numpy.gradient` supplies the central difference in the
    interior and a one-sided difference at each end.

    **The velocity is smoothed.** A central difference on a grid finer than the
    underlying fixes differences an *interpolated* range, so it recovers the
    mean rate of closure across the several seconds between real fixes, not the
    instantaneous one. For scenario 001's 2-8 second fix interval that is a
    smoothing window of the same order. It is the best estimate the input data
    supports, and it is why the truth labels are honest about being derived
    rather than observed.
    """
    enu_m = geodetic_to_enu_m(
        trajectory.latitude_deg,
        trajectory.longitude_deg,
        trajectory.altitude_m,
        radar.latitude_deg,
        radar.longitude_deg,
        radar.altitude_m,
    )
    range_m, azimuth_deg, elevation_deg = enu_to_range_azimuth_elevation(enu_m)
    radial_velocity_mps = -np.gradient(range_m, trajectory.time_s)

    return TargetTrack(
        time_s=trajectory.time_s,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        radial_velocity_mps=np.asarray(radial_velocity_mps, dtype=np.float64),
    )


@dataclass(frozen=True)
class BistaticTargetTrack:
    """What a two-site radar pair measures of a trajectory.

    The bistatic counterpart of :class:`TargetTrack`. Where a monostatic track
    carries one range and one radial velocity, this carries the two ranges of
    ``spec/scenario-002-bistatic.md`` §4 and the single bisector rate
    that follows from them.

    Attributes
    ----------
    time_s : numpy.ndarray
        Seconds since the trajectory's first fix, shape ``(n_frames,)``.
    range_tx_m, range_rx_m : numpy.ndarray
        Transmit and receive ranges, metres, shape ``(n_frames,)``. Named for
        the site each is measured to, not for a property of the target.
    bistatic_angle_rad : numpy.ndarray
        The angle subtended at the target by the two sites, radians, shape
        ``(n_frames,)``.
    transmit_azimuth_deg, transmit_elevation_deg : numpy.ndarray
        Look angles from the **transmitter** site, degrees -- the angle of
        departure. Azimuth is zero at true north and increases clockwise.
    receive_azimuth_deg, receive_elevation_deg : numpy.ndarray
        Look angles from the **receiver** site, degrees -- the angle of arrival.
        These genuinely differ from the transmit angles; that is what makes the
        geometry bistatic.
    bisector_velocity_mps : numpy.ndarray
        Half the rate of closure of the total path, metres/second, shape
        ``(n_frames,)``. **Positive closing**, and smoothed in exactly the way
        :func:`to_radar_frame`'s radial velocity is.

    Notes
    -----
    There is no ``radial_velocity_mps`` here, and the omission is deliberate.
    A bistatic target has two radial velocities, one towards each site, and
    neither of them is what the Doppler shift measures. Offering a field by
    that name would invite a caller to use the wrong one.
    """

    time_s: NDArray[np.float64]
    range_tx_m: NDArray[np.float64]
    range_rx_m: NDArray[np.float64]
    bistatic_angle_rad: NDArray[np.float64]
    transmit_azimuth_deg: NDArray[np.float64]
    transmit_elevation_deg: NDArray[np.float64]
    receive_azimuth_deg: NDArray[np.float64]
    receive_elevation_deg: NDArray[np.float64]
    bisector_velocity_mps: NDArray[np.float64]

    @property
    def n_frames(self) -> int:
        """Number of frames in the track."""
        return int(self.time_s.size)

    @property
    def range_m(self) -> NDArray[np.float64]:
        """Bistatic mean range :math:`(R_t + R_r)/2`, metres.

        The quantity a range-Doppler map's range axis actually carries in the
        bistatic case, per ``spec/scenario-002-bistatic.md`` D6, and
        the one that matches :attr:`TargetTrack.range_m` when the baseline
        shrinks to nothing.
        """
        result: NDArray[np.float64] = (self.range_tx_m + self.range_rx_m) / 2.0
        return result


def to_bistatic_radar_frame(trajectory: Trajectory, radar: BistaticRadar) -> BistaticTargetTrack:
    r"""Express a trajectory as the two ranges and the bisector rate a pair sees.

    The bistatic counterpart of :func:`to_radar_frame`.

    Parameters
    ----------
    trajectory : Trajectory
        The target's positions, already on the frame grid.
    radar : radar_forge.core.radar.BistaticRadar
        The sited pair; each site is the origin of its own local tangent plane.

    Returns
    -------
    BistaticTargetTrack
        Both ranges, both sets of look angles, the bistatic angle and the
        bisector rate, shape ``(n_frames,)`` each.

    Notes
    -----
    The bisector velocity is minus **half** the central difference of the total
    path length,

    .. math:: v_b = -\frac{1}{2}\frac{\mathrm{d}(R_t + R_r)}{\mathrm{d}t},

    the half being what makes it reduce to :func:`to_radar_frame`'s radial
    velocity when the two ranges coincide, so that the same
    :math:`f_d = 2v/\lambda` holds for both sitings [2]_.

    Every caveat on :func:`to_radar_frame`'s velocity applies here unchanged and
    for the same reason: this differences an *interpolated* range, so it is the
    mean rate across several seconds rather than the instantaneous one.

    See Also
    --------
    to_radar_frame : The monostatic form.
    radar_forge.core.signal.bistatic_line_of_sight_paths : Consumes
        ``bisector_velocity_mps`` directly.

    Examples
    --------
    >>> import numpy as np
    >>> from radar_forge.core.radar import BistaticRadar, Receiver, Transmitter
    >>> tx = Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3)
    >>> pair = BistaticRadar(
    ...     tx, Receiver(1.0e6, 30.0, 3.0), 36.0025, -78.941, 60.0, 36.0025, -78.741, 60.0
    ... )
    >>> track = to_bistatic_radar_frame(
    ...     Trajectory(
    ...         time_s=np.array([0.0, 1.0, 2.0]),
    ...         latitude_deg=np.array([35.9183, 35.9183, 35.9183]),
    ...         longitude_deg=np.array([-78.8969, -78.8969, -78.8969]),
    ...         altitude_m=np.array([1500.0, 1500.0, 1500.0]),
    ...     ),
    ...     pair,
    ... )
    >>> bool(np.all(track.bisector_velocity_mps == 0.0))  # a stationary target
    True
    """
    range_tx_m, transmit_azimuth_deg, transmit_elevation_deg = enu_to_range_azimuth_elevation(
        geodetic_to_enu_m(
            trajectory.latitude_deg,
            trajectory.longitude_deg,
            trajectory.altitude_m,
            radar.transmitter_latitude_deg,
            radar.transmitter_longitude_deg,
            radar.transmitter_altitude_m,
        )
    )
    range_rx_m, receive_azimuth_deg, receive_elevation_deg = enu_to_range_azimuth_elevation(
        geodetic_to_enu_m(
            trajectory.latitude_deg,
            trajectory.longitude_deg,
            trajectory.altitude_m,
            radar.receiver_latitude_deg,
            radar.receiver_longitude_deg,
            radar.receiver_altitude_m,
        )
    )
    bisector_velocity_mps = -np.gradient(range_tx_m + range_rx_m, trajectory.time_s) / 2.0

    return BistaticTargetTrack(
        time_s=trajectory.time_s,
        range_tx_m=range_tx_m,
        range_rx_m=range_rx_m,
        bistatic_angle_rad=radar.bistatic_angle_rad(range_tx_m, range_rx_m),
        transmit_azimuth_deg=transmit_azimuth_deg,
        transmit_elevation_deg=transmit_elevation_deg,
        receive_azimuth_deg=receive_azimuth_deg,
        receive_elevation_deg=receive_elevation_deg,
        bisector_velocity_mps=np.asarray(bisector_velocity_mps, dtype=np.float64),
    )
