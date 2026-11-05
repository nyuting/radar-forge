r"""A scenario: what its TOML file says, and the true geometry of its targets at every frame.

A scenario is data. ``[site.tx]`` and an optional ``[site.rx]`` (absent: monostatic), one
``[[burst]]`` per coherent processing interval and ``[receiver]`` make the radar; each
``[[target]]`` is a recorded trajectory or a constant velocity, played over ``[frames]``;
``[detection]`` and ``[tracking]`` set the processing (:mod:`radar_forge.pipeline`); ``[verify]``
is read by :mod:`radar_forge.evaluate`. :func:`load_scenario` refuses an unknown table, key or
value type, so a typo is never silently ignored. :func:`truth_geometry` gives the unfolded
truth: bistatic range :math:`R_T + R_R` and its rate, positive receding.

References
----------
.. [1] National Imagery and Mapping Agency, *Department of Defense World Geodetic System
       1984*, NIMA TR8350.2, 3rd ed., Amendment 1, 2000.
.. [2] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005.
"""

from __future__ import annotations

import csv
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from radar_forge.detection import CfarVariant
from radar_forge.propagation import dbsm_to_m2
from radar_forge.radar import Radar, Receiver, Site, TargetGeometry, Transmitter, geodetic_to_enu_m

__all__ = [
    "DetectionSettings",
    "Scenario",
    "Target",
    "TrackingSettings",
    "load_scenario",
    "truth_geometry",
]

_SITE_KEYS = {"latitude_deg", "longitude_deg", "altitude_m"}
_BURST_KEYS = {f.name for f in fields(Transmitter)} | {"sample_rate_hz", "n_pulses"}
_TABLES = {"scenario", "target", "frames", "site", "burst", "receiver"}
_OPTIONAL_TABLES = {"detection", "tracking", "verify"}


@dataclass(frozen=True)
class Target:
    """One point target: a trajectory CSV at a fixed altitude, or a constant ENU velocity.

    ``position_enu_m`` is at the first frame, in the receiver site's east, north, up frame.
    """

    name: str
    rcs_dbsm: float
    trajectory_path: Path | None = None
    altitude_m: float = 0.0
    position_enu_m: tuple[float, float, float] | None = None
    velocity_enu_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """Require exactly one way of moving."""
        if (self.trajectory_path is None) == (self.position_enu_m is None):
            msg = f"target {self.name!r} needs one of trajectory_path and position_enu_m."
            raise ValueError(msg)

    @property
    def rcs_m2(self) -> float:
        """Cross-section, m² (Swerling 0)."""
        return float(dbsm_to_m2(self.rcs_dbsm))


@dataclass(frozen=True)
class DetectionSettings:
    """``[detection]``: CFAR along each map's range axis, which wraps, then sidelobe merging.

    A cluster within ``merge_range_bins`` range bins of a stronger one is taken for its Doppler
    sidelobe and dropped, a single-target simplification for an untapered map; 0 keeps every
    cluster. ``max_bistatic_range_rate_mps`` is the fastest a target may open or close: the
    dual-PRF search bound and a new track's prior rate spread.
    """

    pfa: float = 1e-5
    n_train: int = 16
    n_guard: int = 4
    variant: CfarVariant = "ca"
    merge_range_bins: float = 1.0
    max_bistatic_range_rate_mps: float = 400.0


@dataclass(frozen=True)
class TrackingSettings:
    """``[tracking]``: the filter, assignment, lifecycle and rate-unfolding policy.

    Every rate and acceleration is of bistatic range, twice the radial value when monostatic.
    ``range_only_retry`` offers each track range alone when range and rate fail the gate; it
    mixes measurement dimensions, so it needs ``cost = "negative_log_likelihood_ratio"``.
    """

    linearisation: Literal["taylor", "unscented"] = "unscented"
    process_noise: Literal["cwna", "dwna"] = "cwna"
    sigma_acceleration_mps2: float = 8.0
    acceleration_correlation_time_s: float = 1.0
    initiator: Literal["direct", "range_only"] = "direct"
    cost: Literal["nis", "negative_log_likelihood_ratio"] = "nis"
    assignment: Literal["confirmed_first", "joint"] = "confirmed_first"
    range_only_retry: bool = False
    gate_probability: float = 0.997
    n_confirm_hits: int = 3
    n_confirm_scans: int = 5
    n_delete_misses: int = 5
    n_reacquire_misses: int = 0
    reacquire_reset: tuple[str, ...] = ()
    rate_unfolding: Literal["none", "track_aided"] = "none"
    unfold_sigma_gate: float = 6.0
    n_slope_scans: int = 5


@dataclass(frozen=True)
class Scenario:
    """Everything a run needs, resolved from its TOML; ``radars`` holds one per burst."""

    name: str
    description: str
    seed: int
    radars: tuple[Radar, ...]
    n_pulses: tuple[int, ...]
    targets: tuple[Target, ...]
    start_time_s: float
    duration_s: float
    frame_rate_hz: float
    detection: DetectionSettings = DetectionSettings()
    tracking: TrackingSettings | None = None
    verify: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a scenario that could not run."""
        bursts = len(self.radars)
        if not (1 <= bursts <= 2 and len(self.n_pulses) == bursts and min(self.n_pulses) >= 1):
            msg = f"need one burst or a dual-PRF pair, each with n_pulses >= 1; got {bursts}."
            raise ValueError(msg)
        if not (self.targets and self.duration_s > 0.0 and self.frame_rate_hz > 0.0):
            msg = "need a [[target]], and duration_s and frame_rate_hz strictly positive."
            raise ValueError(msg)

    @property
    def radar(self) -> Radar:
        """The first burst's radar; every burst shares its sites."""
        return self.radars[0]

    @property
    def n_frames(self) -> int:
        """Frames in the window."""
        return round(self.duration_s * self.frame_rate_hz)

    @property
    def frame_times_s(self) -> NDArray[np.float64]:
        """Frame times, s since the trajectories' first fix, ``(n_frames,)``."""
        result: NDArray[np.float64] = (
            self.start_time_s + np.arange(self.n_frames, dtype=np.float64) / self.frame_rate_hz
        )
        return result


def _keys(table: Mapping[str, Any], allowed: set[str], name: str) -> dict[str, Any]:
    """Return ``table`` as a dict, or raise on a key ``[name]`` does not define."""
    unknown = sorted(set(table) - allowed)
    if unknown:
        msg = f"[{name}] has unknown key(s) {unknown}; allowed: {sorted(allowed)}."
        raise ValueError(msg)
    return dict(table)


def _settings(kind: Any, table: Mapping[str, Any], name: str) -> Any:
    """Build a settings dataclass, refusing an unknown key or a value of the wrong TOML type."""
    defaults = {f.name: f.default for f in fields(kind)}
    values = _keys(table, set(defaults), name)
    for key, value in values.items():
        default: Any = defaults[key]
        given: Any = tuple(value) if isinstance(default, tuple) else value
        # A bool is an int in Python, so the exact type is compared; a float takes an int.
        if not (type(given) is type(default) or (type(default) is float and type(given) is int)):
            msg = f"[{name}] {key} must be {type(default).__name__}; got {value!r}."
            raise TypeError(msg)
        values[key] = float(given) if type(default) is float else given
    return kind(**values)


def _radar(
    burst: Mapping[str, Any], receiver: Mapping[str, Any], sites: Mapping[str, Any]
) -> Radar:
    """Build one burst's radar; its own checks report bad physics by name."""
    burst = _keys(burst, _BURST_KEYS, "burst")
    receive = _keys(receiver, {"gain_rx_dbi", "noise_figure_db"}, "receiver")
    transmitter = Transmitter(
        **{
            k: float(v)
            for k, v in burst.items()
            if k not in ("waveform", "n_pulses", "sample_rate_hz")
        },
        waveform=burst.get("waveform", "fmcw"),
    )
    tx_site = Site(**_keys(sites["tx"], _SITE_KEYS, "site.tx"))
    rx_site = Site(**_keys(sites["rx"], _SITE_KEYS, "site.rx")) if "rx" in sites else None
    chain = Receiver(float(burst["sample_rate_hz"]), **receive)
    return Radar(transmitter, chain, tx_site, rx_site)


def _target(table: Mapping[str, Any], directory: Path) -> Target:
    """Build one ``[[target]]``; a trajectory path resolves against the TOML's directory."""
    allowed = {f.name for f in fields(Target)}
    values = _keys(table, allowed, "target")
    if "trajectory_path" in values:
        values["trajectory_path"] = (directory / values["trajectory_path"]).resolve()
    for key in ("position_enu_m", "velocity_enu_mps"):
        if key in values:
            values[key] = tuple(float(v) for v in values[key])
    return Target(**values)


def load_scenario(path: Path | str) -> Scenario:
    """Read a scenario TOML; an unknown table or key raises, so a typo is never ignored.

    Examples
    --------
    >>> scenario = load_scenario("config/scenario_001_fmcw_low_prf.toml")
    >>> scenario.radar.is_monostatic, scenario.n_frames, scenario.tracking is None
    (True, 120, True)
    """
    toml_path = Path(path)
    document = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    missing, unknown = _TABLES - set(document), set(document) - _TABLES - _OPTIONAL_TABLES
    if missing or unknown or "tx" not in document["site"]:
        msg = f"{toml_path}: missing {sorted(missing)} or [site.tx]; unknown {sorted(unknown)}."
        raise ValueError(msg)
    _keys(document["site"], {"tx", "rx"}, "site")
    scenario = _keys(document["scenario"], {"name", "description", "seed"}, "scenario")
    window = _keys(document["frames"], {"start_time_s", "duration_s", "frame_rate_hz"}, "frames")
    tracking = document.get("tracking")
    return Scenario(
        name=str(scenario["name"]),
        description=str(scenario.get("description", "")),
        seed=int(scenario["seed"]),
        radars=tuple(_radar(b, document["receiver"], document["site"]) for b in document["burst"]),
        n_pulses=tuple(int(burst["n_pulses"]) for burst in document["burst"]),
        targets=tuple(_target(t, toml_path.parent) for t in document["target"]),
        start_time_s=float(window["start_time_s"]),
        duration_s=float(window["duration_s"]),
        frame_rate_hz=float(window["frame_rate_hz"]),
        detection=_settings(DetectionSettings, document.get("detection", {}), "detection"),
        tracking=None if tracking is None else _settings(TrackingSettings, tracking, "tracking"),
        verify=dict(document.get("verify", {})),
    )


def _read_trajectory(path: Path, altitude_m: float) -> tuple[NDArray[np.float64], ...]:
    """Read a ``time_s, latitude_deg, longitude_deg[, altitude_m]`` CSV (data-001 §6.3)."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2 or not {"time_s", "latitude_deg", "longitude_deg"} <= set(rows[0]):
        msg = f"{path} needs time_s, latitude_deg and longitude_deg columns and two fixes."
        raise ValueError(msg)
    time_s, latitude_deg, longitude_deg = (
        np.array([float(row[name]) for row in rows])
        for name in ("time_s", "latitude_deg", "longitude_deg")
    )
    altitudes_m = [float(row.get("altitude_m") or altitude_m) for row in rows]
    if not np.all(np.diff(time_s) > 0.0):
        msg = f"{path}: time_s must be strictly increasing."
        raise ValueError(msg)
    return time_s - time_s[0], latitude_deg, longitude_deg, np.array(altitudes_m)


def _motion(target: Target, scenario: Scenario) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a target's ENU position and velocity at every frame, ``(n_frames, 3)`` each.

    A trajectory is interpolated linearly in latitude and longitude onto the frame grid, padded
    by a frame each side where the track allows, and its velocity is the central difference:
    the mean over the seconds between real fixes, not the instantaneous one.
    """
    times_s = scenario.frame_times_s
    if target.trajectory_path is None:
        velocity_mps = np.asarray(target.velocity_enu_mps, dtype=np.float64)
        elapsed_s = (times_s - times_s[0])[:, None]
        position_m = np.asarray(target.position_enu_m, dtype=np.float64) + elapsed_s * velocity_mps
        return position_m, np.zeros_like(position_m) + velocity_mps
    fix_time_s, *geodetic = _read_trajectory(target.trajectory_path, target.altitude_m)
    step_s = 1.0 / scenario.frame_rate_hz
    grid_s = np.concatenate([[times_s[0] - step_s], times_s, [times_s[-1] + step_s]])
    inside = (grid_s >= fix_time_s[0]) & (grid_s <= fix_time_s[-1])
    if not inside[1:-1].all():
        msg = f"the window [{times_s[0]}, {times_s[-1]}] s leaves {target.name!r}'s track."
        raise ValueError(msg)
    grid_s = grid_s[inside]
    origin = scenario.radar.rx_site or scenario.radar.tx_site
    latitude_deg, longitude_deg, altitude_m = (np.interp(grid_s, fix_time_s, c) for c in geodetic)
    position_m = geodetic_to_enu_m(
        latitude_deg,
        longitude_deg,
        altitude_m,
        origin.latitude_deg,
        origin.longitude_deg,
        origin.altitude_m,
    )
    velocity_mps = np.gradient(position_m, grid_s, axis=0)
    frame = slice(int(inside[0]), int(inside[0]) + times_s.size)
    return position_m[frame], velocity_mps[frame]


def truth_geometry(scenario: Scenario) -> list[TargetGeometry]:
    """Return every frame's true geometry, one row per target, unfolded."""
    motions = [_motion(target, scenario) for target in scenario.targets]
    position_m = np.stack([m[0] for m in motions], axis=1)
    velocity_mps = np.stack([m[1] for m in motions], axis=1)
    # One geometry per frame: TargetGeometry holds one frame's targets.
    return [scenario.radar.geometry(p, v) for p, v in zip(position_m, velocity_mps, strict=True)]
