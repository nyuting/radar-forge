"""Scenario configuration and the frame loop that turns it into IQ.

A scenario is a TOML file plus the code here that reads it and plays it back.
The split matters: the radar, the target and the window are *data*, so that
changing a waveform is an edit to a text file rather than to the library, and
the three variants of scenario 001 differ only in their ``[[leg]]`` tables.

Legs
----
A scenario has one or more **legs**, each a fully specified
:class:`~radar_forge.core.radar.Radar` transmitting its own coherent
processing interval within the frame. One leg is the ordinary case. Two legs
is how a dual-PRF radar resolves the ambiguity neither rate can resolve alone,
and they stay separate all the way through: two different sweep rates cannot
be coherently integrated together, so each leg gets its own cube, its own
range-Doppler map at its own scales, and the pair meets only at the level of
*measurements* in :func:`radar_forge.core.ambiguity.unfold_doppler_dual_prf`.
That is also how the hardware does it, and it is why :attr:`Frame.iq` is a
tuple rather than an array.

Frames
------
:func:`iterate_frames` is a generator, deliberately. Scenario 001's full track
is about 16,500 frames and S1's cube is 4.1 MB, so materialising the run would
cost 68 GB. Streaming it costs one frame.

Within each frame the radar transmits a short CPI -- 256 ms for S1 -- and is
idle for the rest of the second. That is not only a compute saving: over a
contiguous 1 s CPI an 80 m/s target would migrate 80 m, more than a full
74.95 m range bin, and the peak would smear. See the scenario specification
S5.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, S5.3 (pulse-Doppler ambiguity), S8.2.
.. [2] ``spec/scenario-001-singapore-xband.md``, S4-S5 (the three variants and
       the frame/CPI structure).
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.core.dsp import (
    doppler_bin_centers_mps,
    doppler_fft,
    matched_filter,
    range_bin_centers_m,
    range_doppler_map,
)
from radar_forge.core.radar import Radar, Receiver, Transmitter, WaveformKind
from radar_forge.core.signal import fmcw_deramp_baseband, line_of_sight_paths, pulsed_baseband
from radar_forge.core.targets import PointTarget
from radar_forge.core.waveforms import lfm_chirp
from radar_forge.pipelines.trajectories import (
    TargetTrack,
    Trajectory,
    load_flight_csv,
    resample,
    to_radar_frame,
)

__all__ = [
    "Frame",
    "RangeDopplerProduct",
    "Scenario",
    "iterate_frames",
    "leg_range_doppler",
    "load_scenario",
    "peak_range_velocity",
]

_RADAR_KEYS = frozenset({"latitude_deg", "longitude_deg", "altitude_m"})
_RECEIVER_KEYS = frozenset({"gain_rx_dbi", "noise_figure_db"})
_TARGET_KEYS = frozenset({"rcs_dbsm", "altitude_m", "name"})
_TRAJECTORY_KEYS = frozenset({"path", "start_time_s", "duration_s", "frame_rate_hz"})
_LEG_KEYS = frozenset(
    {
        "f0_hz",
        "bandwidth_hz",
        "transmit_power_w",
        "gain_tx_dbi",
        "chirp_time_s",
        "prf_hz",
        "waveform",
        "sample_rate_hz",
        "n_chirps",
    }
)
_SCENARIO_KEYS = frozenset({"name", "description", "seed"})


@dataclass(frozen=True)
class Scenario:
    """Everything needed to play a scenario back, resolved from its TOML.

    Attributes
    ----------
    name : str
        Short identifier, used to label outputs.
    description : str
        One line of prose about what the scenario demonstrates.
    legs : tuple of radar_forge.core.radar.Radar
        One fully specified radar per leg, in transmission order. Length 1 for
        an ordinary scenario, 2 for a dual-PRF one.
    n_chirps : tuple of int
        Chirps in each leg's coherent processing interval, same length and
        order as ``legs``.
    target : radar_forge.core.targets.PointTarget
        The illuminated target. Swerling 0, so its cross-section is constant.
    target_altitude_m : float
        Height above the ellipsoid assigned to the whole track, metres. The
        recorded track has no altitude column.
    trajectory_path : pathlib.Path
        The track CSV, resolved relative to the TOML file's directory.
    start_time_s, duration_s : float
        The window, in seconds from the track's first fix.
    frame_rate_hz : float
        Frames per second of scenario time.
    seed : int
        Seeds the one generator used for the whole run, so a scenario replays
        bit for bit.
    """

    name: str
    description: str
    legs: tuple[Radar, ...]
    n_chirps: tuple[int, ...]
    target: PointTarget
    target_altitude_m: float
    trajectory_path: Path
    start_time_s: float
    duration_s: float
    frame_rate_hz: float
    seed: int

    def __post_init__(self) -> None:
        """Validate the scenario as a whole; see :func:`load_scenario`."""
        if not self.legs:
            msg = "a scenario needs at least one [[leg]]."
            raise ValueError(msg)
        if len(self.legs) != len(self.n_chirps):
            msg = (
                f"legs and n_chirps must have the same length; got "
                f"{len(self.legs)} and {len(self.n_chirps)}."
            )
            raise ValueError(msg)
        if any(count <= 0 for count in self.n_chirps):
            msg = f"every leg needs a strictly positive n_chirps; got {self.n_chirps}."
            raise ValueError(msg)
        if self.duration_s <= 0.0:
            msg = f"duration_s must be strictly positive; got {self.duration_s!r}."
            raise ValueError(msg)
        if self.frame_rate_hz <= 0.0:
            msg = f"frame_rate_hz must be strictly positive; got {self.frame_rate_hz!r}."
            raise ValueError(msg)

    @property
    def n_frames(self) -> int:
        """Number of frames the window produces."""
        return round(self.duration_s * self.frame_rate_hz)

    @property
    def frame_times_s(self) -> NDArray[np.float64]:
        """Frame times, seconds from the track's first fix, shape ``(n_frames,)``."""
        return self.start_time_s + np.arange(self.n_frames, dtype=np.float64) / self.frame_rate_hz


@dataclass(frozen=True)
class Frame:
    """One frame: the IQ the radar collected, and the truth that produced it.

    The truth fields are the **true, unfolded** quantities in every variant.
    The whole point of scenario 001 is the discrepancy between them and what
    the range-Doppler map shows, so they are never folded to match it.

    Attributes
    ----------
    index : int
        Frame number from zero.
    time_s : float
        Scenario time, seconds from the track's first fix.
    iq : tuple of numpy.ndarray
        One complex128 cube per leg, each ``(n_chirps, n_samples)`` with slow
        time on axis 0. Legs may differ in both dimensions.
    range_m : float
        True slant range, metres.
    radial_velocity_mps : float
        True radial velocity, metres/second, positive closing and **not**
        folded into any leg's unambiguous interval.
    azimuth_deg, elevation_deg : float
        True look angles, degrees.
    """

    index: int
    time_s: float
    iq: tuple[NDArray[np.complex128], ...]
    range_m: float
    radial_velocity_mps: float
    azimuth_deg: float
    elevation_deg: float


def _resampled_track(trajectory: Trajectory, scenario: Scenario) -> tuple[TargetTrack, int]:
    """Resample onto the frame grid, padded by a frame each side where possible.

    Returns the track and the index of the window's first frame within it.
    """
    frame_times_s = scenario.frame_times_s
    frame_interval_s = 1.0 / scenario.frame_rate_hz
    track_first_s = float(trajectory.time_s[0])
    track_last_s = float(trajectory.time_s[-1])

    if frame_times_s[0] < track_first_s or frame_times_s[-1] > track_last_s:
        msg = (
            f"the scenario window [{frame_times_s[0]}, {frame_times_s[-1]}] s lies outside "
            f"the track's [{track_first_s}, {track_last_s}] s."
        )
        raise ValueError(msg)

    pad_before = frame_times_s[0] - frame_interval_s >= track_first_s
    pad_after = frame_times_s[-1] + frame_interval_s <= track_last_s
    grid_s = np.concatenate(
        [
            [frame_times_s[0] - frame_interval_s] if pad_before else [],
            frame_times_s,
            [frame_times_s[-1] + frame_interval_s] if pad_after else [],
        ]
    )
    if grid_s.size < 2:
        msg = (
            "a one-frame window at the very end of the track leaves nothing to "
            "difference; radial velocity needs at least two samples."
        )
        raise ValueError(msg)

    track = to_radar_frame(resample(trajectory, grid_s), scenario.legs[0])
    return track, 1 if pad_before else 0


def _require_keys(table: dict[str, Any], allowed: frozenset[str], name: str) -> None:
    """Reject keys a table does not define, so a typo is not silently ignored."""
    unknown = sorted(set(table) - allowed)
    if unknown:
        msg = f"[{name}] has unknown key(s) {unknown}; allowed keys are {sorted(allowed)}."
        raise ValueError(msg)


def _leg_radar(leg: dict[str, Any], radar: dict[str, Any], receiver: dict[str, Any]) -> Radar:
    """Build one leg's Radar, letting its own validation report bad physics."""
    waveform: WaveformKind = leg.get("waveform", "fmcw")
    transmitter = Transmitter(
        f0_hz=float(leg["f0_hz"]),
        bandwidth_hz=float(leg["bandwidth_hz"]),
        transmit_power_w=float(leg["transmit_power_w"]),
        gain_tx_dbi=float(leg["gain_tx_dbi"]),
        chirp_time_s=float(leg["chirp_time_s"]),
        prf_hz=float(leg["prf_hz"]),
        waveform=waveform,
    )
    return Radar(
        transmitter=transmitter,
        receiver=Receiver(
            sample_rate_hz=float(leg["sample_rate_hz"]),
            gain_rx_dbi=float(receiver["gain_rx_dbi"]),
            noise_figure_db=float(receiver["noise_figure_db"]),
        ),
        latitude_deg=float(radar["latitude_deg"]),
        longitude_deg=float(radar["longitude_deg"]),
        altitude_m=float(radar["altitude_m"]),
    )


def load_scenario(path: Path | str) -> Scenario:
    """Read a scenario TOML into a :class:`Scenario`.

    Parameters
    ----------
    path : pathlib.Path or str
        The TOML file. ``[trajectory].path`` is resolved relative to its
        directory, so a scenario is relocatable.

    Returns
    -------
    Scenario
        With every leg built as a :class:`~radar_forge.core.radar.Radar`.

    Raises
    ------
    ValueError
        If a required table or key is missing, if a table carries an unknown
        key, or if no ``[[leg]]`` is defined. Bad *physics* -- a duty cycle
        above one, a negative power -- is reported by ``Radar`` itself, so the
        message names the quantity rather than the file.

    Notes
    -----
    Parsing is :mod:`tomllib` from the standard library; scenario
    configuration adds no dependency.

    Unknown keys are an error rather than a warning. A misspelled
    ``sample_rate_hz`` that is silently ignored produces a scenario that runs,
    looks plausible, and is not the one that was asked for.

    Examples
    --------
    >>> scenario = load_scenario("scenarios/scenario_001_fmcw_low_prf.toml")
    >>> scenario.name
    'scenario-001-fmcw-low-prf'
    >>> len(scenario.legs)
    1
    >>> round(scenario.legs[0].unambiguous_velocity_mps, 3)
    7.648
    """
    toml_path = Path(path)
    with toml_path.open("rb") as handle:
        document = tomllib.load(handle)

    for table_name in ("scenario", "radar", "receiver", "target", "trajectory"):
        if table_name not in document:
            msg = f"{toml_path} is missing the required [{table_name}] table."
            raise ValueError(msg)
    if "leg" not in document or not document["leg"]:
        msg = f"{toml_path} defines no [[leg]]; a scenario needs at least one."
        raise ValueError(msg)

    scenario_table = document["scenario"]
    radar_table = document["radar"]
    receiver_table = document["receiver"]
    target_table = document["target"]
    trajectory_table = document["trajectory"]

    _require_keys(scenario_table, _SCENARIO_KEYS, "scenario")
    _require_keys(radar_table, _RADAR_KEYS, "radar")
    _require_keys(receiver_table, _RECEIVER_KEYS, "receiver")
    _require_keys(target_table, _TARGET_KEYS, "target")
    _require_keys(trajectory_table, _TRAJECTORY_KEYS, "trajectory")
    for leg in document["leg"]:
        _require_keys(leg, _LEG_KEYS, "leg")

    legs = tuple(_leg_radar(leg, radar_table, receiver_table) for leg in document["leg"])
    n_chirps = tuple(int(leg["n_chirps"]) for leg in document["leg"])

    return Scenario(
        name=str(scenario_table["name"]),
        description=str(scenario_table.get("description", "")),
        legs=legs,
        n_chirps=n_chirps,
        target=PointTarget.from_dbsm(
            float(target_table["rcs_dbsm"]), name=str(target_table.get("name", "target"))
        ),
        target_altitude_m=float(target_table["altitude_m"]),
        trajectory_path=(toml_path.parent / str(trajectory_table["path"])).resolve(),
        start_time_s=float(trajectory_table["start_time_s"]),
        duration_s=float(trajectory_table["duration_s"]),
        frame_rate_hz=float(trajectory_table["frame_rate_hz"]),
        seed=int(scenario_table["seed"]),
    )


def iterate_frames(scenario: Scenario) -> Iterator[Frame]:
    """Play a scenario back one frame at a time.

    Parameters
    ----------
    scenario : Scenario
        The resolved scenario.

    Yields
    ------
    Frame
        One per frame time, in order, carrying a cube per leg and the true
        geometry that produced it.

    Raises
    ------
    ValueError
        If the requested window falls outside the track's span. Raised on the
        first call rather than part-way through the run.

    Notes
    -----
    A generator, not a list: scenario 001's full track is about 16,500 frames
    and S1's cube is 4.1 MB, so the whole run would be 68 GB in memory.

    One :class:`numpy.random.Generator` is created from ``scenario.seed`` and
    shared by every frame and leg, so a run replays bit for bit -- but only if
    it is consumed in order. Skipping frames changes the noise in the frames
    that follow.

    The truth carried on each frame is unfolded. Folding it to match a
    particular leg's map is the *reader's* job, and the gap between the two is
    what the scenario exists to show.

    The trajectory is resampled one frame either side of the window wherever
    the track allows it, and those two samples are used for the velocity
    difference and then discarded. Without the padding the first and last
    reported frames would carry a one-sided difference -- a different, noisier
    estimator than every frame between them -- and a one-frame window would
    have no velocity at all.
    """
    trajectory = load_flight_csv(scenario.trajectory_path, altitude_m=scenario.target_altitude_m)
    track, first_frame = _resampled_track(trajectory, scenario)
    rng = np.random.default_rng(scenario.seed)
    rcs_m2 = scenario.target.rcs_m2

    # A Python loop over frames is the point: this is a generator, and each
    # frame's cube is built and handed out before the next one is allocated.
    for index in range(scenario.n_frames):
        sample = first_frame + index
        range_m = float(track.range_m[sample])
        radial_velocity_mps = float(track.radial_velocity_mps[sample])
        azimuth_deg = float(track.azimuth_deg[sample])
        elevation_deg = float(track.elevation_deg[sample])

        cubes: list[NDArray[np.complex128]] = []
        for leg_radar, leg_n_chirps in zip(scenario.legs, scenario.n_chirps, strict=True):
            paths = line_of_sight_paths(
                leg_radar,
                range_m,
                radial_velocity_mps,
                rcs_m2,
                azimuth_deg=azimuth_deg,
                elevation_deg=elevation_deg,
            )
            generate = (
                pulsed_baseband
                if leg_radar.transmitter.waveform == "pulsed"
                else fmcw_deramp_baseband
            )
            cubes.append(generate(paths, leg_radar, leg_n_chirps, rng=rng))

        yield Frame(
            index=index,
            time_s=float(track.time_s[sample]),
            iq=tuple(cubes),
            range_m=range_m,
            radial_velocity_mps=radial_velocity_mps,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
        )


@dataclass(frozen=True)
class RangeDopplerProduct:
    """One leg's range-Doppler map and the axes that label it.

    Attributes
    ----------
    rd_map : numpy.ndarray
        Complex map, shape ``(n_doppler_bins, n_range_bins)``.
    range_axis_m : numpy.ndarray
        Range bin centres, metres, shape ``(n_range_bins,)``. Unshifted: zero
        range is bin 0.
    velocity_axis_mps : numpy.ndarray
        Velocity bin centres, metres/second, shape ``(n_doppler_bins,)``.
        ``fftshift``-ed, zero in the middle, positive closing.
    """

    rd_map: NDArray[np.complex128]
    range_axis_m: NDArray[np.float64]
    velocity_axis_mps: NDArray[np.float64]


def leg_range_doppler(cube: NDArray[np.complex128], leg: Radar) -> RangeDopplerProduct:
    """Process one leg's IQ cube into a range-Doppler map with labelled axes.

    Parameters
    ----------
    cube : numpy.ndarray
        Baseband IQ, shape ``(n_chirps, n_samples)``, slow time on axis 0.
    leg : radar_forge.core.radar.Radar
        The leg that produced it; its waveform selects the receive chain.

    Returns
    -------
    RangeDopplerProduct
        The map and its two axes.

    Notes
    -----
    The two waveform families need different receive chains, and this is the
    one place that knows which:

    **FMCW** deramps to a beat frequency, so range *is* a spectrum and
    :func:`~radar_forge.core.dsp.range_doppler_map` applies directly.

    **Pulsed** needs pulse compression first. Matched filtering runs in
    ``full`` mode, so the output is longer than a repetition interval and a
    target at delay :math:`d` peaks at ``d + n_reference - 1``: the reference's
    own group delay. The window is trimmed by exactly that offset, which is
    what puts a target back at its true range rather than one pulse length
    beyond it. The trim also discards the convolution tail, whose energy
    belongs to the previous interval.

    Both paths return an unshifted range axis and an ``fftshift``-ed velocity
    axis, from the ``dsp`` helpers rather than re-derived -- see the scenario
    specification S7.1.
    """
    if leg.transmitter.waveform == "pulsed":
        reference = lfm_chirp(
            leg.transmitter.bandwidth_hz,
            leg.transmitter.chirp_time_s,
            leg.receiver.sample_rate_hz,
        )
        group_delay_samples = reference.size - 1
        compressed = matched_filter(cube, reference, axis=-1)
        n_samples = leg.n_samples_per_pri
        window = compressed[:, group_delay_samples : group_delay_samples + n_samples]
        rd_map = doppler_fft(window, axis=0)
        sample_index = np.arange(window.shape[1], dtype=np.float64)
        range_axis_m = sample_index * SPEED_OF_LIGHT_MPS / (2.0 * leg.receiver.sample_rate_hz)
    else:
        rd_map = range_doppler_map(cube)
        range_axis_m = range_bin_centers_m(
            rd_map.shape[1],
            leg.transmitter.bandwidth_hz,
            leg.transmitter.chirp_time_s,
            leg.receiver.sample_rate_hz,
        )

    velocity_axis_mps = doppler_bin_centers_mps(
        rd_map.shape[0],
        leg.transmitter.pulse_repetition_interval_s,
        leg.wavelength_m,
    )
    return RangeDopplerProduct(
        rd_map=rd_map,
        range_axis_m=np.asarray(range_axis_m, dtype=np.float64),
        velocity_axis_mps=np.asarray(velocity_axis_mps, dtype=np.float64),
    )


def peak_range_velocity(product: RangeDopplerProduct) -> tuple[float, float]:
    """Locate the brightest cell of a range-Doppler map.

    Parameters
    ----------
    product : RangeDopplerProduct
        The map and its axes.

    Returns
    -------
    range_m : float
        Range of the peak cell, metres. **As the map shows it** -- folded, if
        the waveform folds in range.
    velocity_mps : float
        Velocity of the peak cell, metres/second, positive closing. Likewise
        folded if the waveform folds in Doppler.

    Notes
    -----
    A single-target scenario, so the brightest cell is the target. With
    clutter or a second target this would need a detector; that is
    ``core/detection.py``'s job and it is not part of this slice.
    """
    doppler_bin, range_bin = np.unravel_index(np.abs(product.rd_map).argmax(), product.rd_map.shape)
    return (
        float(product.range_axis_m[range_bin]),
        float(product.velocity_axis_mps[doppler_bin]),
    )
