r"""A scenario: what its TOML file says, and the true geometry of its targets at every frame.

A scenario is data. ``[site.tx]`` and an optional ``[site.rx]`` (absent: monostatic), one
``[[burst]]`` per coherent processing interval and ``[receiver]`` make the radar; each
``[[target]]`` is a recorded trajectory or a constant velocity, played over ``[frames]``;
``[detection]`` and ``[tracking]`` set the processing (:mod:`radar_forge.pipeline`); ``[verify]``
is read by :mod:`radar_forge.evaluate`. A pulsed ``[[burst]]`` may record a range gate, and
``[migration]`` corrects range migration (:mod:`radar_forge.migration`). A noise burst adds
``[leakage]``, ``[[clutter]]``, ``[transmitter_distortion]`` and ``[cancellation]``. A
plot-level scenario (scenario 011) has a ``[sensor]`` instead of the site, bursts and
receiver, ``[[target]] kind = "arc"`` trajectories and ``[clutter_plots]``. An FMCW radar may
receive on ``[receive_array]``, with scenario 006's ``[second_aircraft]`` and ``[doa]``. Each
table's settings class lives in :mod:`radar_forge.settings`. :func:`load_scenario` refuses an
unknown table, key or value type, so a typo is never silently ignored. :func:`truth_geometry`
gives the unfolded truth: bistatic range :math:`R_T + R_R` and its rate, positive receding.

References
----------
.. [1] National Imagery and Mapping Agency, *Department of Defense World Geodetic System
       1984*, NIMA TR8350.2, 3rd ed., Amendment 1, 2000.
.. [2] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005.
"""

from __future__ import annotations

import csv
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import numpy as np
from numpy.typing import NDArray

from radar_forge.array import (
    ArrayGeometry,
    offset_direction_cosine,
    rayleigh_width,
    uniform_linear_array,
)
from radar_forge.propagation import dbsm_to_m2
from radar_forge.radar import Radar, Receiver, Site, TargetGeometry, Transmitter, geodetic_to_enu_m
from radar_forge.settings import (
    ArcTarget,
    CancellationSettings,
    ClutterPlotSettings,
    ClutterSettings,
    DetectionSettings,
    DistortionSettings,
    DoaSettings,
    LeakageSettings,
    MigrationSettings,
    ReceiveArraySettings,
    SecondAircraftSettings,
    SensorSettings,
    Target,
    TrackingSettings,
)
from radar_forge.signal import RangeGate
from radar_forge.tracking.scene import Arc, ClutterComponent, FixedDiscretes, Leg

__all__ = [
    "RangeGate",
    "Scenario",
    "load_scenario",
    "scattering_phase_rad",
    "truth_geometry",
]

_SITE_KEYS = {"latitude_deg", "longitude_deg", "altitude_m"}
_GATE_KEYS = {"range_gate_start_m", "n_gate_samples"}
_BURST_KEYS = {f.name for f in fields(Transmitter)} | {"sample_rate_hz", "n_pulses"} | _GATE_KEYS
_TABLES = {"scenario", "target", "frames", "site", "burst", "receiver"}
_OPTIONAL_TABLES = {"detection", "tracking", "verify", "migration"}
# Scenario 005's continuous noise radar: a [[burst]] key and four tables of its own.
_BURST_KEYS |= {"n_range_lags"}
_NOISE_TABLES = {"leakage", "clutter", "transmitter_distortion", "cancellation"}
_OPTIONAL_TABLES |= _NOISE_TABLES
# Scenario 006's receive array: three tables of its own.
_OPTIONAL_TABLES |= {"receive_array", "second_aircraft", "doa"}


@dataclass(frozen=True)
class Scenario:
    """Everything a run needs, resolved from its TOML; ``radars`` holds one per burst.

    ``gates`` holds one :class:`RangeGate` per burst, the whole PRI when not given.
    """

    name: str
    description: str
    seed: int
    radars: tuple[Radar, ...]
    n_pulses: tuple[int, ...]
    targets: tuple[Target, ...]
    start_time_s: float
    duration_s: float
    frame_rate_hz: float
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    tracking: TrackingSettings | None = None
    verify: Mapping[str, Any] = field(default_factory=dict)
    gates: tuple[RangeGate, ...] = ()
    migration: MigrationSettings = field(default_factory=MigrationSettings)
    n_range_lags: tuple[int, ...] = ()
    leakage: LeakageSettings | None = None
    clutter: tuple[ClutterSettings, ...] = ()
    transmitter_distortion: DistortionSettings | None = None
    cancellation: CancellationSettings | None = None
    sensor: SensorSettings | None = None
    arcs: tuple[ArcTarget, ...] = ()
    clutter_plots: ClutterPlotSettings = field(default_factory=ClutterPlotSettings)
    receive_array: ReceiveArraySettings | None = None
    second_aircraft: SecondAircraftSettings | None = None
    doa: DoaSettings | None = None

    def __post_init__(self) -> None:
        """Reject a scenario that could not run."""
        bursts = len(self.radars)
        plot_level = self.sensor is not None and bursts == 0 and self.arcs and not self.targets
        if not plot_level and not (
            1 <= bursts <= 2 and len(self.n_pulses) == bursts and min(self.n_pulses) >= 1
        ):
            msg = f"need one burst or a dual-PRF pair, each with n_pulses >= 1; got {bursts}."
            raise ValueError(msg)
        if not self.gates:
            object.__setattr__(self, "gates", (RangeGate(),) * bursts)
        pulsed = all(radar.transmitter.waveform == "pulsed" for radar in self.radars)
        gated = any(gate != RangeGate() for gate in self.gates)
        if len(self.gates) != bursts or ((gated or self.migration.method != "none") and not pulsed):
            msg = "need one range gate per burst; a gate and [migration] need pulsed bursts."
            raise ValueError(msg)
        if not (
            (self.targets or plot_level) and self.duration_s > 0.0 and self.frame_rate_hz > 0.0
        ):
            msg = "need a [[target]], and duration_s and frame_rate_hz strictly positive."
            raise ValueError(msg)
        if self.n_frames < 1:
            msg = (
                f"[frames] duration_s = {self.duration_s} at frame_rate_hz = "
                f"{self.frame_rate_hz} holds no frame; it needs at least one."
            )
            raise ValueError(msg)
        if bursts == 2 and self.tracking and self.tracking.rate_unfolding != "none":
            msg = (
                '[tracking] rate_unfolding must be "none" with two bursts: the dual-PRF '
                "pair unfolds the rate itself, so a track has nothing to unfold."
            )
            raise ValueError(msg)
        self._check_noise_bursts()
        self._check_array()

    def _check_noise_bursts(self) -> None:
        """Give each burst its lag count; refuse noise tables without a noise burst to read them."""
        if not self.n_range_lags:
            object.__setattr__(self, "n_range_lags", (0,) * len(self.radars))
        noise = [radar.transmitter.waveform == "noise" for radar in self.radars]
        span = [n * r.n_samples_per_chirp for n, r in zip(self.n_pulses, self.radars, strict=True)]
        lags_ok = [
            (1 <= lags <= samples) if is_noise else lags == 0
            for lags, samples, is_noise in zip(self.n_range_lags, span, noise, strict=True)
        ]
        if not all(lags_ok) or (any(noise) and len(noise) > 1):
            msg = "a noise burst stands alone, its n_range_lags between 1 and its CPI samples."
            raise ValueError(msg)
        tables = (self.leakage, self.clutter, self.transmitter_distortion, self.cancellation)
        if not any(noise) and any(tables):
            msg = f"tables {sorted(_NOISE_TABLES)} describe a noise burst; this scenario has none."
            raise ValueError(msg)
        if self.cancellation is not None:
            self.cancellation.check_fits(span[0])

    def _check_array(self) -> None:
        """Refuse an array off FMCW, B or ``[doa]`` without one, a ``[doa]`` doa cannot run."""
        if self.receive_array is None:
            if self.second_aircraft is not None or self.doa is not None:
                msg = "[second_aircraft] and [doa] need a [receive_array]."
                raise ValueError(msg)
            return
        # migration.pulse_compress streams a whole-PRI cube as one row: it has no channel axis.
        if any(radar.transmitter.waveform != "fmcw" for radar in self.radars):
            msg = "[receive_array] needs FMCW bursts: the pulsed and noise chains are one channel."
            raise ValueError(msg)
        if self.second_aircraft is not None and not self.radar.is_monostatic:
            msg = "[second_aircraft] is placed at the first target's range: monostatic only."
            raise ValueError(msg)
        doa, n_rx = self.doa, self.receive_array.n_rx
        # The doa stage scores its estimates against aircraft A and B.
        if doa is not None and (self.second_aircraft is None or doa.n_sources != 2):
            msg = "[doa] scores aircraft A and B: it needs [second_aircraft] and n_sources = 2."
            raise ValueError(msg)
        if doa is None:
            return
        # Two subarrays at least, or spatial smoothing is forward-backward averaging alone.
        if not doa.n_sources < doa.n_subarray_elements < n_rx:
            msg = f"[doa] needs n_sources < n_subarray_elements < n_rx = {n_rx}."
            raise ValueError(msg)
        # A sample covariance of fewer cells than channels is singular: Capon cannot invert it.
        if (2 * doa.half_width_bins + 1) ** 2 <= n_rx:
            msg = (
                f"[doa] trains on (2 half_width_bins + 1)^2 cells: more than n_rx = {n_rx} needed."
            )
            raise ValueError(msg)
        scored = {
            *self.verify.get("resolved_methods", ()),
            *self.verify.get("baseline_methods", ()),
        }
        scored |= {
            k.removesuffix("_rmse_crb_max") for k in self.verify if k.endswith("_rmse_crb_max")
        }
        if not scored <= set(doa.methods):
            msg = (
                f"[verify] scores methods [doa] does not run: {sorted(scored - set(doa.methods))}."
            )
            raise ValueError(msg)

    @property
    def array(self) -> ArrayGeometry | None:
        """The receive array's geometry, or ``None`` for one channel."""
        settings = self.receive_array
        if settings is None:
            return None
        return uniform_linear_array(
            settings.n_rx,
            settings.element_spacing_wavelengths,
            self.radar.wavelength_m,
            boresight_azimuth_deg=settings.boresight_azimuth_deg,
            boresight_elevation_deg=settings.boresight_elevation_deg,
        )

    @property
    def truth_names(self) -> tuple[str, ...]:
        """Each truth row's name: the targets', then aircraft B's."""
        second = () if self.second_aircraft is None else (self.second_aircraft.name,)
        return tuple(target.name for target in self.targets) + second

    @property
    def rcs_m2(self) -> tuple[float, ...]:
        """Each truth row's cross-section, m²: the targets', then aircraft B's."""
        second = () if self.second_aircraft is None else (self.second_aircraft.rcs_dbsm,)
        return tuple(t.rcs_m2 for t in self.targets) + tuple(float(dbsm_to_m2(s)) for s in second)

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
        """Frame times, s, on the trajectory files' own ``time_s`` base, ``(n_frames,)``."""
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
    """Build a settings dataclass, refusing an unknown key, a wrong TOML type or choice."""
    defaults = {f.name: f.default for f in fields(kind)}
    hints = get_type_hints(kind)
    values = _keys(table, set(defaults), name)
    for key, value in values.items():
        choices = get_args(hints[key]) if get_origin(hints[key]) is Literal else ()
        if choices and value not in choices:
            msg = f"[{name}] {key} must be one of {list(choices)}; got {value!r}."
            raise ValueError(msg)
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
            if k not in ("waveform", "n_pulses", "sample_rate_hz", "n_range_lags", *_GATE_KEYS)
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
    for key in ("position_enu_m", "velocity_enu_mps", "acceleration_enu_mps2"):
        if key in values:
            values[key] = tuple(float(v) for v in values[key])
    return Target(**values)


def _gate(burst: Mapping[str, Any]) -> RangeGate:
    """Build one burst's range gate from its optional keys; the default is the whole PRI."""
    n_samples = burst.get("n_gate_samples")
    if n_samples is not None and (type(n_samples) is not int or n_samples < 1):
        msg = f"[burst] n_gate_samples must be a positive integer; got {n_samples!r}."
        raise TypeError(msg)
    return RangeGate(float(burst.get("range_gate_start_m", 0.0)), n_samples)


def _arc_target(table: Mapping[str, Any]) -> ArcTarget:
    """Build one ``[[target]] kind = "arc"``; each leg is straight, turning or accelerating."""
    keys = {"name", "kind", "class", "start_time_s", "position_enu_m", "heading_deg", "speed_mps"}
    values = _keys(table, keys | {"legs", "repeat"}, "target")
    if values.get("kind") != "arc" or values.get("class") not in ("aircraft", "bird"):
        msg = f'target {values.get("name")!r} needs kind = "arc" and class aircraft or bird.'
        raise ValueError(msg)
    legs = []
    for entry in values.get("legs", []):
        leg = _keys(entry, {"duration_s", "turn_rate_degps", "acceleration_mps2"}, "target.legs")
        turn_radps = math.radians(leg.get("turn_rate_degps", 0.0))
        legs.append(Leg(float(leg["duration_s"]), turn_radps, leg.get("acceleration_mps2", 0.0)))
    east_m, north_m = (float(v) for v in values["position_enu_m"])
    heading_rad, speed_mps = math.radians(values["heading_deg"]), float(values["speed_mps"])
    start_s, repeat = float(values.get("start_time_s", 0.0)), int(values.get("repeat", 1))
    arc = Arc(start_s, (east_m, north_m), heading_rad, speed_mps, tuple(legs) * repeat)
    return ArcTarget(str(values["name"]), values["class"], arc)


def _clutter_plots(table: Mapping[str, Any]) -> ClutterPlotSettings:
    """Build ``[clutter_plots]``: its ``[[component]]`` regions and optional ``[discretes]``."""
    table = _keys(table, {"component", "discretes"}, "clutter_plots")
    component_keys = {f.name for f in fields(ClutterComponent)} | {"name"}
    components = []
    for entry in table.get("component", []):
        values = _keys(entry, component_keys, "clutter_plots.component")
        values.pop("name", None)
        for key in ("centre_enu_m", "drift_enu_mps"):
            if key in values:
                values[key] = tuple(float(v) for v in values[key])
        components.append(ClutterComponent(**values))
    discretes = None
    if "discretes" in table:
        keys = {"count", "range_min_m", "range_max_m", "report_probability", "seed"}
        discretes = FixedDiscretes.uniform(**_keys(table["discretes"], keys, "discretes"))
    return ClutterPlotSettings(tuple(components), discretes)


def _plot_level(document: Mapping[str, Any], path: Path) -> Scenario:
    """Build a plot-level scenario: ``[sensor]``, arc targets and ``[clutter_plots]``."""
    allowed = {"scenario", "target", "frames", "sensor", "clutter_plots", "tracking", "verify"}
    if set(document) - allowed or {"scenario", "target", "frames"} - set(document):
        msg = f"{path}: a [sensor] scenario has tables {sorted(allowed)}; got {sorted(document)}."
        raise ValueError(msg)
    sensor = _settings(SensorSettings, document["sensor"], "sensor")
    head = _keys(document["scenario"], {"name", "description", "seed"}, "scenario")
    window = _keys(document["frames"], {"start_time_s", "duration_s"}, "frames")
    tracking = document.get("tracking")
    return Scenario(
        name=str(head["name"]),
        description=str(head.get("description", "")),
        seed=int(head["seed"]),
        radars=(),
        n_pulses=(),
        targets=(),
        start_time_s=float(window["start_time_s"]),
        duration_s=float(window["duration_s"]),
        frame_rate_hz=1.0 / sensor.scan_period_s,
        tracking=None if tracking is None else _settings(TrackingSettings, tracking, "tracking"),
        verify=dict(document.get("verify", {})),
        sensor=sensor,
        arcs=tuple(_arc_target(t) for t in document["target"]),
        clutter_plots=_clutter_plots(document.get("clutter_plots", {})),
    )


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
    if "sensor" in document:
        return _plot_level(document, toml_path)
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
        gates=tuple(_gate(burst) for burst in document["burst"]),
        migration=_settings(MigrationSettings, document.get("migration", {}), "migration"),
        n_range_lags=tuple(int(burst.get("n_range_lags", 0)) for burst in document["burst"]),
        leakage=_optional(LeakageSettings, document, "leakage"),
        clutter=tuple(
            _settings(ClutterSettings, t, "clutter") for t in document.get("clutter", [])
        ),
        transmitter_distortion=_optional(DistortionSettings, document, "transmitter_distortion"),
        cancellation=_optional(CancellationSettings, document, "cancellation"),
        receive_array=_optional(ReceiveArraySettings, document, "receive_array"),
        second_aircraft=_second_aircraft(document),
        doa=_optional(DoaSettings, document, "doa"),
    )


def _second_aircraft(document: Mapping[str, Any]) -> SecondAircraftSettings | None:
    """Build ``[second_aircraft]``, whose scattering phase is a word or a number."""
    if "second_aircraft" not in document:
        return None
    table = dict(document["second_aircraft"])
    phase = table.pop("scattering_phase", "uniform")
    settings: SecondAircraftSettings = _settings(SecondAircraftSettings, table, "second_aircraft")
    return replace(settings, scattering_phase=phase)


def _optional(kind: Any, document: Mapping[str, Any], name: str) -> Any:
    """Build a settings table if the document has it, else ``None``."""
    return _settings(kind, document[name], name) if name in document else None


def _read_trajectory(path: Path, altitude_m: float) -> tuple[NDArray[np.float64], ...]:
    """Read a ``time_s, latitude_deg, longitude_deg[, altitude_m]`` CSV (data-001 §6.3).

    ``time_s`` is kept as written, not rebased to the first fix, so every target's file and
    the frame times share one time base.
    """
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
    return time_s, latitude_deg, longitude_deg, np.array(altitudes_m)


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
        if target.acceleration_enu_mps2 is None:
            return position_m, np.zeros_like(position_m) + velocity_mps
        acceleration_mps2 = np.asarray(target.acceleration_enu_mps2, dtype=np.float64)
        return (
            position_m + acceleration_mps2 * elapsed_s**2 / 2.0,
            velocity_mps + acceleration_mps2 * elapsed_s,
        )
    fix_time_s, *geodetic = _read_trajectory(target.trajectory_path, target.altitude_m)
    step_s = 1.0 / scenario.frame_rate_hz
    grid_s = np.concatenate([[times_s[0] - step_s], times_s, [times_s[-1] + step_s]])
    inside = (grid_s >= fix_time_s[0]) & (grid_s <= fix_time_s[-1])
    if not inside[1:-1].all():
        msg = f"the window [{times_s[0]}, {times_s[-1]}] s leaves {target.name!r}'s track."
        raise ValueError(msg)
    grid_s = grid_s[inside]
    if grid_s.size < 2:
        msg = (
            f"the one-frame window at {times_s[0]} s has no frame of {target.name!r}'s "
            "track either side to difference for a velocity; widen it or move it inside."
        )
        raise ValueError(msg)
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
    """Return every frame's true geometry, one row per target, unfolded.

    A target with ``acceleration_enu_mps2`` carries its route's whole second derivative, the
    turning term included; every other target's is zero, as :class:`Target` says.
    """
    motions = [_motion(target, scenario) for target in scenario.targets]
    position_m = np.stack([m[0] for m in motions], axis=1)
    velocity_mps = np.stack([m[1] for m in motions], axis=1)
    given = [target.acceleration_enu_mps2 for target in scenario.targets]
    modelled = np.array([a is not None for a in given])
    acceleration_mps2 = np.array([a or (0.0, 0.0, 0.0) for a in given])
    frames = []
    # One geometry per frame: TargetGeometry holds one frame's targets.
    for p, v in zip(position_m, velocity_mps, strict=True):
        if not modelled.any():
            frames.append(scenario.radar.geometry(p, v))
            continue
        route = scenario.radar.geometry(p, v, acceleration_enu_mps2=acceleration_mps2)
        kept = np.where(modelled, route.bistatic_range_acceleration_mps2, 0.0)
        frames.append(replace(route, bistatic_range_acceleration_mps2=kept))
    if scenario.second_aircraft is not None:
        return [_with_second_aircraft(scenario, route) for route in frames]
    return frames


def _with_second_aircraft(scenario: Scenario, route: TargetGeometry) -> TargetGeometry:
    """Append aircraft B's row: the first target's, its arrival moved in direction cosine."""
    array, settings = scenario.array, scenario.second_aircraft
    if array is None or settings is None:
        msg = "aircraft B needs [receive_array] and [second_aircraft]."
        raise ValueError(msg)
    offset = settings.separation_rayleigh * rayleigh_width(array, scenario.radar.wavelength_m)
    arrival_rad = offset_direction_cosine(array, route.aoa_rad[:1], offset)
    second = {f.name: getattr(route, f.name)[:1] for f in fields(TargetGeometry)}
    # Monostatic: B departs along the direction it arrives from.
    second["aoa_rad"] = second["aod_rad"] = arrival_rad
    return TargetGeometry(
        **{name: np.concatenate([getattr(route, name), row]) for name, row in second.items()}
    )


def scattering_phase_rad(scenario: Scenario) -> NDArray[np.float64]:
    """Return each truth row's scattering phase at every frame, ``(n_frames, n_rows)``, rad.

    Zero for every target; aircraft B's is its fixed phase, or uniform on [0, 2 pi) per frame
    from ``default_rng([seed, 6])``, a generator of its own, so the noise does not move it.
    """
    n_rows = len(scenario.rcs_m2)
    phase_rad = np.zeros((scenario.n_frames, n_rows))
    settings = scenario.second_aircraft
    if settings is not None and settings.scattering_phase == "uniform":
        rng = np.random.default_rng([scenario.seed, 6])
        phase_rad[:, -1] = rng.uniform(0.0, 2.0 * np.pi, scenario.n_frames)
    elif settings is not None:
        phase_rad[:, -1] = float(settings.scattering_phase)
    return phase_rad
