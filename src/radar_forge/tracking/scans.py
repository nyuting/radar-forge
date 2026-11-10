r"""A plot-level scenario played rotation by rotation, and the ground-plane tracker it feeds.

:func:`scans` plays a scenario with a ``[sensor]`` table (scenario 011) one rotation at a time:
the beam crosses each target at its own time, the plot model of
:mod:`~radar_forge.tracking.scene` merges, detects and measures the echoes, and clutter and
fixed discretes are added, each plot with the clutter intensity at it. :class:`PlotTracker`
builds a :class:`~radar_forge.tracking.Tracker` (GNN, the JPDA family or MHT) or a
:class:`~radar_forge.tracking.rfs.PmbmTracker` from ``[tracking]`` and feeds it each rotation
as sector batches, keeping the clutter map as policy. Nothing here halves a range or does I/O.

References
----------
.. [1] W. Koch and G. van Keuk, "Multiple hypothesis track maintenance with possibly unresolved
       measurements," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 33, no. 3, pp. 883-892, 1997.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from radar_forge.tracking.association import (
    GlobalNearestNeighbour,
    JointProbabilisticDataAssociation,
)
from radar_forge.tracking.filters import (
    TaylorLinearisation,
    TrackBank,
    TrackStatus,
    UnscentedTransform,
)
from radar_forge.tracking.hypotheses import TrackOrientedMHT
from radar_forge.tracking.imm import InteractingMultipleModel
from radar_forge.tracking.initiation import DebiasedSphericalInitiator
from radar_forge.tracking.models import (
    CartesianMotion,
    CoordinatedTurn,
    RangeAzimuthElevationRangeRate,
)
from radar_forge.tracking.rfs import PmbmTracker
from radar_forge.tracking.scene import (
    LEAST_INTENSITY_PER_M_RAD,
    ClutterMap,
    FixedDiscretes,
    clutter_intensity,
    detect,
    detection_probability,
    measure,
    poisson_clutter_plots,
    polar,
    resolve,
)
from radar_forge.tracking.tracker import ExistencePolicy, LifecyclePolicy, Scan, Tracker

if TYPE_CHECKING:
    # Types only: scenario.py imports tracking.scene, so a run-time import would be a cycle.
    from radar_forge.scenario import Scenario
    from radar_forge.settings import TrackingSettings

__all__ = ["VARIANTS", "PlotScan", "PlotTracker", "scans", "tracker_name"]

# A plot-level scan's one model, and the state coordinates a sector's coverage reads.
_PLOTS = "plots"
_POSITION_VELOCITY = ("x_m", "xdot_mps", "y_m", "ydot_mps")

# The fixed-point iterations that find when the beam crosses a moving target: its azimuth moves
# under a degree a rotation, so each iteration shrinks the time error a hundredfold.
_BEAM_ITERATIONS = 3

# Each named tracker as [tracking]'s association, coalescence and existence; "pmbm" stands alone.
VARIANTS: dict[
    str, tuple[Literal["gnn", "jpda", "mht"], Literal["none", "jpda_star", "set"], bool]
] = {
    "gnn": ("gnn", "none", False),
    "jpda": ("jpda", "none", False),
    "jpda_star": ("jpda", "jpda_star", False),
    "set": ("jpda", "set", False),
    "jipda": ("jpda", "none", True),
    "mht": ("mht", "none", False),
}
# A cue's spread about the truth: 50 m in position, 5 m/s in velocity (011a's A16 template).
_CUE_SIGMA = {"x_m": 50.0, "y_m": 50.0, "xdot_mps": 5.0, "ydot_mps": 5.0}


@dataclass(frozen=True, eq=False)
class PlotScan:
    """One rotation of a plot-level scenario: its plots, and the truth behind them.

    Plots ``(n_plots,)``: the time the beam crossed each, horizontal range, azimuth, the
    sector it is batched in, its ``source`` (``target``, ``merged``, ``clutter``,
    ``discrete``) and ``source_ids`` (truth names joined by ``;``), and the clutter
    intensity there per (m rad), the oracle. Echoes ``(n_targets,)``: when the beam crossed
    each target, whether it was in coverage, in the MTI notch, and detected (alone or merged).
    The tracker never reads ``source`` or the echoes.
    """

    index: int
    start_time_s: float
    time_s: NDArray[np.float64]
    range_m: NDArray[np.float64]
    azimuth_rad: NDArray[np.float64]
    sector: NDArray[np.int64]
    source: tuple[str, ...]
    source_ids: tuple[str, ...]
    clutter_intensity_per_m_rad: NDArray[np.float64]
    echo_time_s: NDArray[np.float64]
    echo_in_coverage: NDArray[np.bool_]
    echo_in_notch: NDArray[np.bool_]
    echo_detected: NDArray[np.bool_]


def _beam_crossing(
    scenario: Scenario, start_s: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return when the beam crosses each target in a rotation, and its position and velocity."""
    period_s = scenario.sensor.scan_period_s if scenario.sensor else 1.0
    time_s = np.full(len(scenario.arcs), start_s)
    # A short fixed-point iteration per rotation: time from azimuth, azimuth from time.
    for _ in range(_BEAM_ITERATIONS):
        states = [target.arc.state(t) for target, t in zip(scenario.arcs, time_s, strict=True)]
        position_m = np.array([p for p, _ in states])
        # A target not yet flying is NaN; its azimuth counts as 0, the rotation's start.
        azimuth = np.nan_to_num(polar(position_m)[1])
        time_s = start_s + azimuth / (2 * np.pi) * period_s
    return time_s, position_m, np.array([v for _, v in states])


def scans(
    scenario: Scenario, *, n_frames: int | None = None, rng: np.random.Generator | None = None
) -> Iterator[PlotScan]:
    """Play a plot-level scenario back one rotation at a time, with the truth behind each plot.

    Each target is seen when the beam crosses it; echoes merge, are detected and measured by
    :mod:`~radar_forge.tracking.scene`'s plot model; clutter and fixed discretes are added.
    One generator (default: the scenario's seed) draws everything, so a run replays.
    """
    sensor = scenario.sensor
    if sensor is None:
        msg = f"scenario {scenario.name!r} has no [sensor]: it is not plot-level."
        raise ValueError(msg)
    rng = np.random.default_rng(scenario.seed) if rng is None else rng
    names = np.array([target.name for target in scenario.arcs])
    pd = np.array([getattr(sensor, f"pd_{target.target_class}") for target in scenario.arcs])
    sector_rad = 2 * np.pi / sensor.n_sectors
    sigma_az = math.radians(sensor.sigma_azimuth_deg)
    count = scenario.n_frames if n_frames is None else min(n_frames, scenario.n_frames)
    # A loop over rotations: each is drawn and handed out before the next.
    for index in range(count):
        start_s = scenario.start_time_s + index * sensor.scan_period_s
        echo_s, position_m, velocity_mps = _beam_crossing(scenario, start_s)
        ranges, azimuths = polar(position_m)
        rate_mps = np.sum(position_m * velocity_mps, axis=-1) / ranges
        seen = np.isfinite(ranges) & (ranges >= sensor.range_min_m) & (ranges < sensor.range_max_m)
        notch = np.abs(rate_mps) < sensor.mti_notch_mps
        rows = np.flatnonzero(seen)
        group = resolve(
            ranges[rows],
            azimuths[rows],
            alpha_range_m=sensor.range_cell_m,
            alpha_azimuth_rad=math.radians(sensor.beamwidth_azimuth_deg),
            rng=rng,
        )
        echo_pd = detection_probability(
            rate_mps[rows],
            pd[rows],
            mti_notch_mps=sensor.mti_notch_mps,
            notch_detection_probability=sensor.pd_notch,
        )
        found = detect(group, echo_pd, rng)
        plot_r, plot_az = measure(
            ranges[rows],
            azimuths[rows],
            group,
            sigma_range_m=sensor.sigma_range_m,
            sigma_azimuth_rad=sigma_az,
            rng=rng,
        )
        members = [names[rows][group == g] for g in range(found.size)]
        detected = np.zeros(len(names), dtype=bool)
        detected[rows] = found[group]
        clutter_r, clutter_az, _ = poisson_clutter_plots(
            scenario.clutter_plots.components, start_s, rng
        )
        discretes = scenario.clutter_plots.discretes or FixedDiscretes(
            np.zeros(0), np.zeros(0), 0.0
        )
        fixed_r, fixed_az = discretes.plots(sensor.sigma_range_m, sigma_az, rng)
        range_m = np.concatenate([plot_r[found], clutter_r, fixed_r])
        azimuth_rad = np.concatenate([plot_az[found], clutter_az, fixed_az])
        kinds = [("merged" if len(m) > 1 else "target", ";".join(m)) for m in members]
        source = [k for k, keep in zip(kinds, found, strict=True) if keep]
        source += [("clutter", "")] * clutter_r.size + [("discrete", "")] * fixed_r.size
        yield PlotScan(
            index=index,
            start_time_s=start_s,
            time_s=start_s + azimuth_rad / (2 * np.pi) * sensor.scan_period_s,
            range_m=range_m,
            azimuth_rad=azimuth_rad,
            sector=np.minimum(azimuth_rad // sector_rad, sensor.n_sectors - 1).astype(np.int64),
            source=tuple(s for s, _ in source),
            source_ids=tuple(i for _, i in source),
            clutter_intensity_per_m_rad=clutter_intensity(
                scenario.clutter_plots.components, range_m, azimuth_rad, start_s
            ),
            echo_time_s=echo_s,
            echo_in_coverage=seen,
            echo_in_notch=notch,
            echo_detected=detected,
        )


def tracker_name(settings: TrackingSettings) -> str:
    """Return the configured tracker's name, the key of :data:`VARIANTS` its settings match."""
    chosen = (settings.association, settings.coalescence, settings.existence)
    return next((name for name, value in VARIANTS.items() if value == chosen), "jpda")


class PlotTracker:
    """The ``[tracking]`` table as a ground-plane :class:`~radar_forge.tracking.Tracker`.

    The state is ENU (x, y) and velocity, plus the turn rate under ``motion = "imm"``, whose
    modes are CV quiet, coordinated turn and CV agile; z is (horizontal range, azimuth), the
    radar at the origin and z fixed at its height. A rotation reaches the tracker as
    ``n_sectors`` batches, each at its centre time, and a batch sees only the tracks whose
    extrapolated azimuth lies in its sector. GNN costs by the likelihood ratio against the
    clutter regions' mean intensity that rotation, with no assignment cheaper than
    ln(P_D / (1 - P_D P_G)), the track score's break-even; JPDA weighs
    confirmed tracks jointly and tentative ones alone, with the clutter intensity at each plot
    from the scene (``"oracle"``) or a :class:`~radar_forge.tracking.scene.ClutterMap`
    (``"map"``) fed every plot no confirmed track took. MHT scores by the same intensity.
    ``name`` (a key of :data:`VARIANTS`, or ``"pmbm"``) replaces the configured association;
    PMBM runs one CV model, the agile mode's under ``motion = "imm"``. ``seed_truth`` starts
    each aircraft's track at its truth, confirmed.
    """

    def __init__(self, scenario: Scenario, name: str | None = None) -> None:
        settings, sensor = scenario.tracking, scenario.sensor
        if settings is None or sensor is None:
            msg = f"scenario {scenario.name!r} needs [sensor] and [tracking] to track plots."
            raise ValueError(msg)
        if name is not None and name != "pmbm":
            association, coalescence, existence = VARIANTS[name]
            settings = dataclasses.replace(
                settings, association=association, coalescence=coalescence, existence=existence
            )
        self.sensor, self.settings = sensor, settings
        self.period_s = sensor.scan_period_s
        quiet, turn, agile = settings.mode_sigma_acceleration_mps2
        cv = CartesianMotion("xy", sigma_acceleration_mps2=settings.sigma_acceleration_mps2)
        motion: Any = cv
        if settings.motion == "imm":
            motion = InteractingMultipleModel(
                (
                    CartesianMotion("xy", sigma_acceleration_mps2=quiet),
                    CoordinatedTurn(turn, settings.turn_rate_noise_density_rad2ps3),
                    CartesianMotion("xy", sigma_acceleration_mps2=agile),
                ),
                settings.sojourn_time_s,
                {"turn_rate_radps": math.radians(settings.turn_rate_sigma_degps)},
            )
        layout = motion.state_layout
        model = RangeAzimuthElevationRangeRate(
            layout, np.zeros(3), ("range_m", "azimuth_rad"), fixed={"z_m": 0.0}
        )
        self.noise_covariance = np.diag(
            [sensor.sigma_range_m**2, math.radians(sensor.sigma_azimuth_deg) ** 2]
        )
        prior = np.diag([v**2 for v in self._prior_sigma(layout.names, settings)])
        p_d, p_g = settings.detection_probability, settings.gate_probability
        self.components = scenario.clutter_plots.components
        initiator = DebiasedSphericalInitiator(np.zeros(layout.dimension), prior)
        associator: Any = GlobalNearestNeighbour(math.log(p_d / (1 - p_d * p_g)))
        if settings.association == "jpda":
            associator = JointProbabilisticDataAssociation(p_d, p_g, settings.coalescence)
        elif settings.association == "mht":
            associator = TrackOrientedMHT(
                p_d,
                p_g,
                settings.k_best,
                settings.n_scan,
                settings.confirm_score,
                settings.delete_drop,
            )
        self.tracker: Tracker | PmbmTracker = Tracker(
            {_PLOTS: model},
            motion,
            {"taylor": TaylorLinearisation, "unscented": UnscentedTransform}[
                settings.linearisation
            ](),
            p_g,
            associator,
            initiator,
            LifecyclePolicy(
                settings.n_confirm_hits,
                settings.n_confirm_scans,
                settings.n_delete_misses,
                max_coast_time_s=(settings.n_delete_misses + 1) * self.period_s,
                max_position_sigma_m=min(settings.max_position_sigma_m, 1e12),
            ),
            "negative_log_likelihood_ratio",
            {_PLOTS: self.mean_intensity(scenario.start_time_s)},
            assignment=settings.assignment,
            coverage=self._covers,
            existence=ExistencePolicy(settings.survival_probability)
            if settings.existence
            else None,
            max_birth_clutter_intensity=settings.max_birth_clutter_intensity_per_m_rad,
        )
        if name == "pmbm":
            sigma = settings.sigma_acceleration_mps2 if settings.motion == "cv" else agile
            cv_prior = np.diag([v**2 for v in self._prior_sigma(cv.state_layout.names, settings)])
            self.tracker = PmbmTracker(
                CartesianMotion("xy", sigma_acceleration_mps2=sigma),
                RangeAzimuthElevationRangeRate(
                    cv.state_layout, np.zeros(3), ("range_m", "azimuth_rad"), fixed={"z_m": 0.0}
                ),
                DebiasedSphericalInitiator(np.zeros(cv.state_layout.dimension), cv_prior),
                detection_probability=p_d,
                gate_probability=p_g,
                coverage_volume=2 * np.pi * (sensor.range_max_m - sensor.range_min_m),
                birth_intensity_per_scan=settings.birth_intensity_per_scan,
                survival_probability=settings.survival_probability,
                prune_existence=settings.prune_existence,
                prune_weight=settings.prune_weight,
                max_birth_clutter_intensity=settings.max_birth_clutter_intensity_per_m_rad,
                coverage=self._covers,
                assignment=settings.assignment,
            )
        if settings.seed_truth:
            self._seed(scenario)
        # The map starts at twice the birth threshold: no track starts where it has not looked.
        limit = settings.max_birth_clutter_intensity_per_m_rad
        self.clutter_map = ClutterMap(range_max_m=sensor.range_max_m, prior_per_m_rad=2 * limit)
        self._sector = 0

    def _seed(self, scenario: Scenario) -> None:
        """Start a confirmed track on each aircraft at its truth at the start time: a cue."""
        start_s = scenario.start_time_s
        aircraft = [t for t in scenario.arcs if t.target_class == "aircraft"]
        # Position and velocity, east then north, per aircraft: (n, 4) in x, xdot, y, ydot order.
        truth = np.array([np.ravel(np.array(t.arc.state(start_s)).T) for t in aircraft])
        names = self.tracker.bank.layout.names
        mean = np.zeros((len(aircraft), len(names)))
        for k, name in enumerate(_POSITION_VELOCITY):
            mean[:, names.index(name)] = truth[:, k]
        sigma = self._prior_sigma(names, self.settings)
        spread = np.array([_CUE_SIGMA.get(n, sigma[k]) for k, n in enumerate(names)])
        covariance = np.zeros((len(aircraft), len(names), len(names))) + np.diag(spread**2)
        self.tracker.seed(mean, covariance, start_s)

    @property
    def bank(self) -> TrackBank:
        """Return the tracks this tracker reports now: every track, or MHT's and PMBM's best."""
        return self.tracker.bank

    @staticmethod
    def _prior_sigma(names: Sequence[str], settings: TrackingSettings) -> list[float]:
        """Return a new track's prior standard deviation per coordinate; positions are replaced."""
        speed, turn = settings.initial_speed_sigma_mps, math.radians(settings.turn_rate_sigma_degps)
        sigma = {"x_m": 1.0, "y_m": 1.0, "xdot_mps": speed, "ydot_mps": speed}
        return [sigma.get(name, turn) for name in names]

    def _covers(self, bank: TrackBank, scan: Scan) -> NDArray[np.bool_]:
        """Return the tracks whose azimuth, extrapolated to the batch, lies in its sector."""
        layout = bank.layout
        x, xdot, y, ydot = (bank.mean[:, layout.names.index(n)] for n in _POSITION_VELOCITY)
        dt_s = scan.time_s - bank.time_s
        _, azimuth = polar(np.column_stack([x + xdot * dt_s, y + ydot * dt_s]))
        sector = np.minimum(
            azimuth // (2 * np.pi / self.sensor.n_sectors), self.sensor.n_sectors - 1
        )
        return np.asarray(sector == self._sector, dtype=bool)

    def mean_intensity(self, time_s: float) -> float:
        """Return the clutter regions' mean intensity at ``time_s``, plots per (m rad): GNN's λ."""
        volumes = [c.volume_m_rad(time_s) for c in self.components]
        plots = sum(
            c.intensity_per_m_rad * v for c, v in zip(self.components, volumes, strict=True)
        )
        mean = plots / sum(volumes) if self.components else 0.0
        return max(mean, LEAST_INTENSITY_PER_M_RAD)

    def rotation(self, plots: PlotScan) -> list[tuple[TrackBank, Any]]:
        """Track one rotation, sector by sector; return each batch's bank and JPDA clusters."""
        n_sectors = self.sensor.n_sectors
        # A drifting disc changes the regions' mean, so GNN's density is this rotation's.
        if isinstance(self.tracker, Tracker):
            self.tracker.clutter_density = MappingProxyType(
                {_PLOTS: self.mean_intensity(plots.start_time_s)}
            )
        oracle = self.settings.clutter_intensity == "oracle"
        intensity = (
            plots.clutter_intensity_per_m_rad
            if oracle
            else self.clutter_map.intensity(plots.range_m, plots.azimuth_rad)
        )
        results, taken = [], np.zeros(plots.range_m.size, dtype=bool)
        # Sector by sector: each batch's tracks are the next one's prior.
        for sector in range(n_sectors):
            self._sector = sector
            mine = np.flatnonzero(plots.sector == sector)
            time_s = plots.start_time_s + (sector + 0.5) / n_sectors * self.period_s
            z = np.column_stack([plots.range_m[mine], plots.azimuth_rad[mine]])
            scan = Scan(time_s, z, self.noise_covariance, _PLOTS, intensity[mine])
            association = None
            if isinstance(self.tracker, Tracker):
                self.tracker.last_association = None
                bank = self.tracker.process(scan)
                association = self.tracker.last_association
            else:
                bank = self.tracker.process(scan)
            confirmed = np.isin(bank.status, (TrackStatus.CONFIRMED, TrackStatus.COASTING))
            used = bank.measurement_index[confirmed & (bank.measurement_index >= 0)]
            taken[mine[used]] = True
            results.append((bank, association))
        self.clutter_map.update(plots.range_m[~taken], plots.azimuth_rad[~taken])
        return results
