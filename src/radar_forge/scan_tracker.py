r"""A scenario's tracker: the ``[tracking]`` table as one tracker, fed one scan at a time.

:class:`ScanTracker` builds a :class:`~radar_forge.tracking.Tracker` (GNN, the JPDA family or
MHT) or a :class:`~radar_forge.tracking.pmbm.PmbmTracker` from ``[tracking]``, for either kind
of scenario, and feeds it each scan. An IQ scenario's radar stares: a scan's accepted plots
(:class:`~radar_forge.pipeline.Plots`) reach the tracker as one batch at the scan's time, the
state bistatic range and its rate. A plot-level scenario's radar rotates (``[sensor]``,
scenario 011): a scan's plots (:class:`~radar_forge.plot_simulator.PlotScan`) reach it as
``n_sectors`` batches, each at its sector's centre time, the state ENU position and velocity in
the ground plane. MATLAB delivers a rotating radar's scan the same way, as field-of-view
batches. Nothing here halves a range or does I/O.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from radar_forge.coordinates import east_north_to_range_azimuth_rad
from radar_forge.pipeline import Plots, map_axes
from radar_forge.plot_simulator import LEAST_DENSITY_PER_M_RAD, PlotScan, sector_index
from radar_forge.scenario import Scenario
from radar_forge.settings import TrackingSettings
from radar_forge.tracking.association import (
    AssociationWeights,
    GlobalNearestNeighbour,
    JointProbabilisticDataAssociation,
    non_assignment_cost,
)
from radar_forge.tracking.clutter_map import ClutterMap
from radar_forge.tracking.filters import (
    TaylorLinearisation,
    TrackBank,
    TrackStatus,
    UnscentedTransform,
    is_confirmed,
)
from radar_forge.tracking.hypotheses import TrackOrientedMHT
from radar_forge.tracking.imm import InteractingMultipleModel
from radar_forge.tracking.initiation import (
    DebiasedSphericalInitiator,
    DirectStateInitiator,
    Initiator,
)
from radar_forge.tracking.lifecycle import ExistencePolicy, LifecyclePolicy
from radar_forge.tracking.measurement import (
    LinearMeasurement,
    RangeAzimuthElevationRangeRate,
    marginal,
)
from radar_forge.tracking.motion import ConstantTurn, ConstantVelocity
from radar_forge.tracking.pmbm import PmbmTracker
from radar_forge.tracking.tracker import Scan, Tracker

__all__ = ["VARIANTS", "ScanTracker", "tracker_name"]

# The model IDs a staring scan is measured through: its own, and its range-only marginal; a
# rotating scan's one model.
_MEASURED, _RANGE, _PLOTS = "range_rate", "range", "plots"
# The state coordinates a rotating batch's coverage reads.
_POSITION_VELOCITY = ("x_m", "xdot_mps", "y_m", "ydot_mps")
_LINEARISATIONS = {"taylor": TaylorLinearisation, "unscented": UnscentedTransform}

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
# A cue's spread about the truth, 50 m in position and 5 m/s in velocity: what the coalescence
# test in tests/tracking/test_association.py seeds its tracks with.
_CUE_SIGMA = {"x_m": 50.0, "y_m": 50.0, "xdot_mps": 5.0, "ydot_mps": 5.0}

Batch = tuple[TrackBank, AssociationWeights | None]


def tracker_name(settings: TrackingSettings) -> str:
    """Return the configured tracker's name, the key of :data:`VARIANTS` its settings match."""
    chosen = (settings.association, settings.coalescence, settings.existence)
    return next((name for name, value in VARIANTS.items() if value == chosen), "jpda")


class ScanTracker:
    r"""A scenario's ``[tracking]`` table as a tracker, and the policy that feeds it each scan.

    Both kinds read alike the filter (``linearisation``, the motion model's
    ``sigma_acceleration_mps2``, ``white_noise`` and correlation time), ``initiator``, ``cost``,
    ``assignment``, ``gate_probability``, the association (``association``, ``coalescence``,
    ``existence`` and MHT's tree) and the lifecycle counts; ``name`` (a key of
    :data:`VARIANTS`, or ``"pmbm"`` when rotating) replaces the configured association. GNN
    takes any pair the gate allows, or, given ``detection_probability``, none cheaper than
    ln(P_D / (1 - P_D P_G)), the track score's break-even.

    Staring: the state is ``[bistatic_range_m, bistatic_range_rate_mps]``. The rate is measured
    where it cannot fold: after dual-PRF pairing, from one burst whose rate period exceeds twice
    the largest rate, or once a track can unfold it; otherwise range alone. Range is measured
    modulo the first map's period (the state stays continuous), with a bin over √12 of noise,
    and the clutter density is one false plot a scan over the map. ``rate_unfolding =
    "track_aided"`` is scenario 003's policy: a track may unfold after N associated ranges, the
    least N whose least-squares slope is certain to a ``unfold_sigma_gate``-th of the rate
    period, :math:`(\sigma_R/T)\sqrt{12/(N(N^2-1))} < v_{period}/g` (Bar-Shalom [1] §5.5). The
    track with the longest such history gives each plot the fold nearest its rate, unless that
    lies over half a period from the slope of its last ``n_slope_scans`` ranges, the fold
    monitor: a wrong fold agrees with the prediction that chose it, never with the range. One
    reference track serves a scan: single-target policy.

    Rotating: the state is ENU (x, y) and velocity, plus the turn rate under ``motion =
    "imm"``, whose modes are CV quiet, constant turn and CV agile; z is (horizontal range,
    azimuth), the radar at the origin and z fixed at its height. A batch sees only the tracks
    whose extrapolated azimuth lies in its sector. The likelihood ratio's density is the clutter
    regions' mean density that scan; JPDA and MHT take the density at each plot from the scene
    (``clutter_density_source = "oracle"``) or a
    :class:`~radar_forge.tracking.clutter_map.ClutterMap` (``"map"``) fed every plot no
    confirmed track took. PMBM runs one CV model, the agile mode's under ``motion = "imm"``.
    ``seed_truth`` starts each aircraft's track at its truth, confirmed.
    """

    def __init__(self, scenario: Scenario, name: str | None = None) -> None:
        settings = scenario.tracking
        if settings is None:
            msg = f"scenario {scenario.name!r} has no [tracking] table."
            raise ValueError(msg)
        if name is not None and name != "pmbm":
            association, coalescence, existence = VARIANTS[name]
            settings = dataclasses.replace(
                settings, association=association, coalescence=coalescence, existence=existence
            )
        self.settings, self.sensor = settings, scenario.sensor
        self.scan_period_s = scenario.scan_period_s
        if self.sensor is None and (name == "pmbm" or settings.seed_truth):
            msg = "PMBM and seed_truth need a plot-level scenario: its [sensor] and its arcs."
            raise ValueError(msg)
        if self.sensor is None:
            motion, models, initiator, density = self._staring(scenario, settings)
        else:
            motion, models, initiator, density = self._rotating(scenario, settings)
        lives = settings.n_delete_misses + settings.n_reacquire_misses
        sigma_m = settings.max_position_sigma_m
        self.tracker: Tracker | PmbmTracker = Tracker(
            models,
            motion,
            _LINEARISATIONS[settings.linearisation](),
            settings.gate_probability,
            self._associator(settings),
            initiator,
            LifecyclePolicy(
                settings.n_confirm_hits,
                settings.n_confirm_scans,
                settings.n_delete_misses,
                settings.n_reacquire_misses,
                # The miss count deletes; the coast-time limit is set past it, never to fire.
                max_coast_time_s=(lives + 1) * self.scan_period_s,
                reacquire_reset=settings.reacquire_reset,
                max_position_sigma_m=None if math.isinf(sigma_m) else sigma_m,
            ),
            settings.cost,
            density if settings.cost != "nis" else None,
            assignment=settings.assignment,
            model_selector=self._select if len(models) > 1 else None,
            coverage=None if self.sensor is None else self._covers,
            existence=ExistencePolicy(settings.survival_probability)
            if settings.existence
            else None,
            max_birth_clutter_density=None
            if self.sensor is None
            else settings.max_birth_clutter_density_per_m_rad,
        )
        if name == "pmbm":
            self.tracker = self._pmbm(settings)
        if settings.seed_truth:
            self._seed(scenario)

    @staticmethod
    def _associator(settings: TrackingSettings) -> Any:
        """Return the configured associator; GNN's non-assignment cost needs P_D to be finite."""
        p_d, p_g = settings.detection_probability, settings.gate_probability
        if settings.association == "gnn":
            if p_d is None:
                return GlobalNearestNeighbour()
            return GlobalNearestNeighbour(non_assignment_cost(p_d, p_g))
        if p_d is None:
            msg = f'[tracking] association = "{settings.association}" needs detection_probability.'
            raise ValueError(msg)
        if settings.association == "jpda":
            return JointProbabilisticDataAssociation(p_d, p_g, settings.coalescence)
        return TrackOrientedMHT(
            p_d,
            p_g,
            settings.k_best,
            settings.n_scan_depth,
            settings.confirm_score,
            settings.delete_drop,
        )

    # --- staring: an IQ scenario's accepted plots, one batch a scan ---

    def _staring(
        self, scenario: Scenario, settings: TrackingSettings
    ) -> tuple[ConstantVelocity, dict[str, Any], Initiator, dict[str, float]]:
        """Build the bistatic range-and-rate models, the scan policy and the density."""
        if settings.initiator == "debiased_spherical":
            msg = '[tracking] initiator = "debiased_spherical" needs a plot-level [sensor].'
            raise ValueError(msg)
        range_axis_m, rate_axis_mps = map_axes(scenario.radar, scenario.n_pulses_per_burst[0])
        range_bin_m = float(range_axis_m[1] - range_axis_m[0])
        rate_bin_mps = float(rate_axis_mps[0] - rate_axis_mps[1])
        self.range_period_m = range_axis_m.size * range_bin_m
        self.rate_period_mps = rate_axis_mps.size * rate_bin_mps
        self.n_slope_scans = settings.n_slope_scans
        sigma_range_m, sigma_rate_mps = (
            range_bin_m / math.sqrt(12.0),
            rate_bin_mps / math.sqrt(12.0),
        )
        max_rate_mps, dual = (
            scenario.detection.max_bistatic_range_rate_mps,
            len(scenario.radars) == 2,
        )
        track_aided = {"track_aided": True, "none": False}[settings.rate_unfolding]
        self.measures_rate = dual or track_aided or self.rate_period_mps / 2.0 > max_rate_mps
        names = ("bistatic_range_m", "bistatic_range_rate_mps")[: 1 + self.measures_rate]
        motion = ConstantVelocity.bistatic_range(
            settings.sigma_acceleration_mps2,
            acceleration_correlation_time_s=settings.acceleration_correlation_time_s,
            white_noise=settings.white_noise,
        )
        model = LinearMeasurement(
            motion.state_layout, names, periods={names[0]: self.range_period_m}
        )
        models: dict[str, Any] = {_MEASURED: model}
        if self.measures_rate and (settings.range_only_retry or track_aided):
            models[_RANGE] = marginal(model, names[:1])
        # One false alarm per scan over the range period and over the rate span a plot can take;
        # only their ratio decides an assignment, since GNN maximises the pairs first.
        rate_span_mps = 2.0 * max_rate_mps if dual else self.rate_period_mps
        density = {
            _MEASURED: 1.0 / self.range_period_m / (rate_span_mps if self.measures_rate else 1)
        }
        density[_RANGE] = 1.0 / self.range_period_m
        self._plot_density = density[_MEASURED]
        n_scans = np.arange(2, 1000)
        slope_sigma_mps = sigma_range_m / self.scan_period_s * np.sqrt(12 / (n_scans**3 - n_scans))
        enough = n_scans[slope_sigma_mps < self.rate_period_mps / settings.unfold_sigma_gate]
        if track_aided and not enough.size:
            msg = "the range slope never resolves the fold within 1000 scans; widen the gate."
            raise ValueError(msg)
        # Tracks unfold once they hold this many ranges; None without track-aided unfolding.
        self.n_unfold_scans = int(enough[0]) if track_aided else None
        # Each track's (scan time, unwrapped range) pairs, from its birth on.
        self.history: dict[int, list[tuple[float, float]]] = {}
        noise = np.diag([sigma_range_m**2, sigma_rate_mps**2])
        self.noise_covariance = noise[: len(names), : len(names)]
        initiator = DirectStateInitiator(
            np.zeros(2),
            np.diag([sigma_range_m**2, max_rate_mps**2]),
            seed_names={"range_only": names[:1], "direct": None}[settings.initiator],
        )
        return motion, models, initiator, {k: density[k] for k in models}

    def unfoldable(self, track_id: NDArray[np.int64]) -> NDArray[np.bool_]:
        """Return which tracks may unfold: all without track-aided unfolding, else by history."""
        if self.n_unfold_scans is None:
            return np.ones(track_id.shape, dtype=bool)
        lengths = [len(self.history.get(int(t), ())) for t in track_id]
        return np.asarray(lengths, dtype=np.int64) >= self.n_unfold_scans

    def _select(self, bank: TrackBank, model_id: str) -> list[tuple[str, NDArray[np.bool_]]]:
        """Offer range and rate to every track that may unfold, then range alone to all."""
        return [(_MEASURED, self.unfoldable(bank.track_id)), (_RANGE, np.ones(bank.n_tracks, bool))]

    def _fold(self, folded_mps: NDArray[np.float64], reference_mps: float) -> NDArray[np.float64]:
        """Return each folded rate moved by whole periods to the fold nearest ``reference_mps``."""
        period_mps = self.rate_period_mps
        result: NDArray[np.float64] = folded_mps + period_mps * np.round(
            (reference_mps - folded_mps) / period_mps
        )
        return result

    def unfold(self, folded_mps: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return rates unfolded by the reference track and the fold monitor, or as read."""
        bank = self.tracker.bank
        ready = np.flatnonzero(self.unfoldable(bank.track_id))
        if self.n_unfold_scans is None or not ready.size:
            return folded_mps
        histories = [self.history[int(bank.track_id[row])] for row in ready]
        longest = int(np.argmax([len(history) for history in histories]))
        unfolded_mps = self._fold(folded_mps, float(bank.mean[ready[longest], 1]))
        # Fit on the scans' own times: a coasted scan leaves a gap, not a shorter step.
        time_s, recent_m = np.array(histories[longest][-self.n_slope_scans :]).T
        slope_mps = float(np.polyfit(time_s, recent_m, 1)[0])
        disagree = np.abs(unfolded_mps - slope_mps) > self.rate_period_mps / 2.0
        result: NDArray[np.float64] = np.where(
            disagree, self._fold(folded_mps, slope_mps), unfolded_mps
        )
        return result

    def _record(self, bank: TrackBank, ranges_m: NDArray[np.float64], time_s: float) -> None:
        """Add each track that took a measurement, born or updated, to its range history.

        Ranges are unwrapped across the range period and kept with their scan time.
        """
        alive = bank.status != TrackStatus.DELETED
        living = set(bank.track_id[alive].tolist())
        self.history = {t: h for t, h in self.history.items() if t in living}
        period_m = self.range_period_m
        # One append per measured track: a handful a scan, each to its own list.
        for row in np.flatnonzero(alive & (bank.measurement_index >= 0)):
            history = self.history.setdefault(int(bank.track_id[row]), [])
            range_m = float(ranges_m[bank.measurement_index[row]])
            if history:
                last_m = history[-1][1]
                # Adds last_m after the mod: last_m + _periodic.fold(...) adds in another
                # order and moves the last bit of many ranges.
                range_m = last_m + (range_m - last_m + period_m / 2) % period_m - period_m / 2
            history.append((time_s, range_m))

    def _stare(self, plots: Plots, time_s: float) -> Batch:
        """Track a scan's accepted plots as one batch; record the ranges each track took."""
        accepted = plots.accepted
        rate_mps = plots.bistatic_range_rate_mps[accepted]
        if self.n_unfold_scans is not None:
            rate_mps = self.unfold(plots.folded_rate_mps[accepted])
        measurements = np.column_stack([plots.bistatic_range_m[accepted], rate_mps])
        measurements = measurements[:, : self.noise_covariance.shape[0]]
        density = np.full(len(measurements), self._plot_density)
        batch = self._batch(Scan(time_s, measurements, self.noise_covariance, _MEASURED, density))
        self._record(batch[0], measurements[:, 0], time_s)
        return batch

    # --- rotating: a plot-level scenario's plots, a batch a sector ---

    def _rotating(
        self, scenario: Scenario, settings: TrackingSettings
    ) -> tuple[Any, dict[str, Any], Initiator, dict[str, float]]:
        """Build the ground-plane models, the clutter map and the regions' mean density."""
        sensor = scenario.sensor
        if sensor is None or settings.initiator != "debiased_spherical":
            msg = '[tracking] of a plot-level scenario needs initiator = "debiased_spherical".'
            raise ValueError(msg)
        self.components = scenario.clutter_plots.components
        quiet, turn, agile = settings.mode_sigma_acceleration_mps2
        motion: Any = self._constant_velocity(settings, settings.sigma_acceleration_mps2)
        if settings.motion == "imm":
            motion = InteractingMultipleModel(
                (
                    self._constant_velocity(settings, quiet),
                    ConstantTurn(
                        turn,
                        settings.turn_rate_noise_density_rad2ps3,
                        acceleration_correlation_time_s=settings.acceleration_correlation_time_s,
                    ),
                    self._constant_velocity(settings, agile),
                ),
                settings.sojourn_time_s,
                {"turn_rate_radps": math.radians(settings.turn_rate_sigma_degps)},
            )
        layout = motion.state_layout
        self.noise_covariance = np.diag(
            [sensor.sigma_range_m**2, math.radians(sensor.sigma_azimuth_deg) ** 2]
        )
        prior = np.diag([v**2 for v in self._prior_sigma(layout.names, settings)])
        # The map starts at twice the birth threshold: no track starts where it has not looked.
        limit = settings.max_birth_clutter_density_per_m_rad
        self.clutter_map = ClutterMap(range_max_m=sensor.range_max_m, prior_per_m_rad=2 * limit)
        self._sector = 0
        return (
            motion,
            {_PLOTS: self._spherical(layout)},
            DebiasedSphericalInitiator(np.zeros(layout.dimension), prior),
            {_PLOTS: self.mean_density(scenario.start_time_s)},
        )

    @staticmethod
    def _constant_velocity(settings: TrackingSettings, sigma_mps2: float) -> ConstantVelocity:
        """Return the ground plane's CV model at one acceleration sigma."""
        return ConstantVelocity(
            "xy",
            sigma_acceleration_mps2=sigma_mps2,
            acceleration_correlation_time_s=settings.acceleration_correlation_time_s,
            white_noise=settings.white_noise,
        )

    @staticmethod
    def _spherical(layout: Any) -> RangeAzimuthElevationRangeRate:
        """Return (horizontal range, azimuth) from the radar at the origin, z fixed at 0."""
        return RangeAzimuthElevationRangeRate(
            layout, np.zeros(3), ("range_m", "azimuth_rad"), fixed={"z_m": 0.0}
        )

    def _pmbm(self, settings: TrackingSettings) -> PmbmTracker:
        """Return PMBM on one CV model, the agile mode's under the IMM."""
        sensor = self.sensor
        if sensor is None or settings.detection_probability is None:
            msg = "PMBM needs a plot-level [sensor] and [tracking] detection_probability."
            raise ValueError(msg)
        agile = settings.mode_sigma_acceleration_mps2[2]
        cv = self._constant_velocity(
            settings, settings.sigma_acceleration_mps2 if settings.motion == "cv" else agile
        )
        prior = np.diag([v**2 for v in self._prior_sigma(cv.state_layout.names, settings)])
        return PmbmTracker(
            cv,
            self._spherical(cv.state_layout),
            DebiasedSphericalInitiator(np.zeros(cv.state_layout.dimension), prior),
            detection_probability=settings.detection_probability,
            gate_probability=settings.gate_probability,
            coverage_volume=2 * np.pi * (sensor.range_max_m - sensor.range_min_m),
            birth_intensity_per_scan=settings.birth_intensity_per_scan,
            survival_probability=settings.survival_probability,
            prune_existence=settings.prune_existence,
            prune_weight=settings.prune_weight,
            max_birth_clutter_density=settings.max_birth_clutter_density_per_m_rad,
            coverage=self._covers,
            assignment=settings.assignment,
        )

    def _seed(self, scenario: Scenario) -> None:
        """Start a confirmed track on each aircraft at its truth at the start time: a cue."""
        start_s = scenario.start_time_s
        aircraft = [t for t in scenario.arcs if t.target_class == "aircraft"]
        # Position and velocity, east then north, per aircraft: (n, 4) in x, xdot, y, ydot order.
        truth = np.array([np.ravel(np.array(t.arc.state(start_s)).T) for t in aircraft])
        names = self.tracker.bank.state_layout.names
        mean = np.zeros((len(aircraft), len(names)))
        for k, name in enumerate(_POSITION_VELOCITY):
            mean[:, names.index(name)] = truth[:, k]
        sigma = self._prior_sigma(names, self.settings)
        spread = np.array([_CUE_SIGMA.get(n, sigma[k]) for k, n in enumerate(names)])
        covariance = np.zeros((len(aircraft), len(names), len(names))) + np.diag(spread**2)
        self.tracker.seed(mean, covariance, start_s)

    @staticmethod
    def _prior_sigma(names: Sequence[str], settings: TrackingSettings) -> list[float]:
        """Return a new track's prior standard deviation per coordinate; positions are replaced."""
        speed, turn = settings.initial_speed_sigma_mps, math.radians(settings.turn_rate_sigma_degps)
        sigma = {"x_m": 1.0, "y_m": 1.0, "xdot_mps": speed, "ydot_mps": speed}
        return [sigma.get(name, turn) for name in names]

    def _covers(self, bank: TrackBank, scan: Scan) -> NDArray[np.bool_]:
        """Return the tracks whose azimuth, extrapolated to the batch, lies in its sector."""
        if self.sensor is None:
            return np.ones(bank.n_tracks, dtype=bool)
        layout = bank.state_layout
        x, xdot, y, ydot = (bank.mean[:, layout.names.index(n)] for n in _POSITION_VELOCITY)
        dt_s = scan.time_s - bank.time_s
        _, azimuth = east_north_to_range_azimuth_rad(
            np.column_stack([x + xdot * dt_s, y + ydot * dt_s])
        )
        return np.asarray(sector_index(azimuth, self.sensor.n_sectors) == self._sector, dtype=bool)

    def mean_density(self, time_s: float) -> float:
        """Return the clutter regions' mean density at ``time_s``, plots per (m rad): GNN's λ."""
        volumes = [c.volume_m_rad(time_s) for c in self.components]
        plots = sum(c.density_per_m_rad * v for c, v in zip(self.components, volumes, strict=True))
        mean = plots / sum(volumes) if self.components else 0.0
        return max(mean, LEAST_DENSITY_PER_M_RAD)

    def _rotate(self, plots: PlotScan) -> list[Batch]:
        """Track one scan sector by sector; feed the clutter map what no confirmed track took."""
        sensor = self.sensor
        if sensor is None:
            msg = "a staring tracker takes a scan's Plots, not a PlotScan."
            raise ValueError(msg)
        n_sectors = sensor.n_sectors
        # A drifting disc changes the regions' mean, so the likelihood ratio's density is this
        # scan's.
        if isinstance(self.tracker, Tracker) and self.settings.cost != "nis":
            self.tracker.clutter_density = MappingProxyType(
                {_PLOTS: self.mean_density(plots.start_time_s)}
            )
        oracle = self.settings.clutter_density_source == "oracle"
        density = (
            plots.clutter_density_per_m_rad
            if oracle
            else self.clutter_map.density(plots.range_m, plots.azimuth_rad)
        )
        results, taken = [], np.zeros(plots.range_m.size, dtype=bool)
        # Sector by sector: each batch's tracks are the next one's prior.
        for sector in range(n_sectors):
            self._sector = sector
            mine = np.flatnonzero(plots.sector == sector)
            time_s = plots.start_time_s + (sector + 0.5) / n_sectors * self.scan_period_s
            z = np.column_stack([plots.range_m[mine], plots.azimuth_rad[mine]])
            bank, association = self._batch(
                Scan(time_s, z, self.noise_covariance, _PLOTS, density[mine])
            )
            confirmed = is_confirmed(bank.status)
            used = bank.measurement_index[confirmed & (bank.measurement_index >= 0)]
            taken[mine[used]] = True
            results.append((bank, association))
        self.clutter_map.update(plots.range_m[~taken], plots.azimuth_rad[~taken])
        return results

    # --- both ---

    def _batch(self, scan: Scan) -> Batch:
        """Run one batch; return the bank and a weighting associator's clusters, else None."""
        if isinstance(self.tracker, PmbmTracker):
            return self.tracker.process(scan), None
        self.tracker.last_association = None
        bank = self.tracker.process(scan)
        return bank, self.tracker.last_association

    @property
    def bank(self) -> TrackBank:
        """Return the tracks this tracker reports now: every track, or MHT's and PMBM's best."""
        return self.tracker.bank

    def process(self, scan_plots: Plots | PlotScan, time_s: float | None = None) -> list[Batch]:
        """Track one scan; return each batch's bank and association, in the order run.

        Staring, the scan's :class:`~radar_forge.pipeline.Plots` and its time ``time_s`` (plots
        carry none) give one batch; rotating, a
        :class:`~radar_forge.plot_simulator.PlotScan` gives one batch a sector.
        """
        if isinstance(scan_plots, PlotScan):
            return self._rotate(scan_plots)
        if self.sensor is not None or time_s is None:
            msg = (
                "a staring tracker takes a scan's Plots and its time_s; a rotating one a PlotScan."
            )
            raise ValueError(msg)
        return [self._stare(scan_plots, time_s)]
