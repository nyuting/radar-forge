r"""The tracker a plot-level scenario's scans feed, built from ``[tracking]``: :class:`PlotTracker`.

:class:`PlotTracker` builds a :class:`~radar_forge.tracking.Tracker` (GNN, the JPDA family or
MHT) or a :class:`~radar_forge.tracking.pmbm.PmbmTracker` from ``[tracking]`` and feeds it each
scan of :func:`~radar_forge.plot_simulator.plot_scans` as sector batches, keeping the clutter
map as policy. Nothing here halves a range or does I/O.
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
from radar_forge.plot_simulator import LEAST_DENSITY_PER_M_RAD, PlotScan, sector_index
from radar_forge.scenario import Scenario
from radar_forge.settings import TrackingSettings
from radar_forge.tracking.association import (
    GlobalNearestNeighbour,
    JointProbabilisticDataAssociation,
    non_assignment_cost,
)
from radar_forge.tracking.clutter_map import ClutterMap
from radar_forge.tracking.filters import (
    TaylorLinearisation,
    TrackBank,
    UnscentedTransform,
    is_confirmed,
)
from radar_forge.tracking.hypotheses import TrackOrientedMHT
from radar_forge.tracking.imm import InteractingMultipleModel
from radar_forge.tracking.initiation import DebiasedSphericalInitiator
from radar_forge.tracking.lifecycle import ExistencePolicy, LifecyclePolicy
from radar_forge.tracking.measurement import RangeAzimuthElevationRangeRate
from radar_forge.tracking.motion import ConstantTurn, ConstantVelocity
from radar_forge.tracking.pmbm import PmbmTracker
from radar_forge.tracking.tracker import Scan, Tracker

__all__ = ["VARIANTS", "PlotTracker", "tracker_name"]

# A plot-level scan's one model, and the state coordinates a sector's coverage reads.
_PLOTS = "plots"
_POSITION_VELOCITY = ("x_m", "xdot_mps", "y_m", "ydot_mps")

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


def tracker_name(settings: TrackingSettings) -> str:
    """Return the configured tracker's name, the key of :data:`VARIANTS` its settings match."""
    chosen = (settings.association, settings.coalescence, settings.existence)
    return next((name for name, value in VARIANTS.items() if value == chosen), "jpda")


class PlotTracker:
    """The ``[tracking]`` table as a ground-plane :class:`~radar_forge.tracking.Tracker`.

    The state is ENU (x, y) and velocity, plus the turn rate under ``motion = "imm"``, whose
    modes are CV quiet, coordinated turn and CV agile; z is (horizontal range, azimuth), the
    radar at the origin and z fixed at its height. A scan reaches the tracker as
    ``n_sectors`` batches, each at its centre time, and a batch sees only the tracks whose
    extrapolated azimuth lies in its sector. GNN costs by the likelihood ratio against the
    clutter regions' mean density that scan, with no assignment cheaper than
    ln(P_D / (1 - P_D P_G)), the track score's break-even; JPDA weighs
    confirmed tracks jointly and tentative ones alone, with the clutter density at each plot
    from the scene (``"oracle"``) or a :class:`~radar_forge.tracking.clutter_map.ClutterMap`
    (``"map"``) fed every plot no confirmed track took. MHT scores by the same density.
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
        self.period_s = scenario.scan_period_s
        quiet, turn, agile = settings.mode_sigma_acceleration_mps2
        cv = ConstantVelocity("xy", sigma_acceleration_mps2=settings.sigma_acceleration_mps2)
        motion: Any = cv
        if settings.motion == "imm":
            motion = InteractingMultipleModel(
                (
                    ConstantVelocity("xy", sigma_acceleration_mps2=quiet),
                    ConstantTurn(turn, settings.turn_rate_noise_density_rad2ps3),
                    ConstantVelocity("xy", sigma_acceleration_mps2=agile),
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
        associator: Any = GlobalNearestNeighbour(non_assignment_cost(p_d, p_g))
        if settings.association == "jpda":
            associator = JointProbabilisticDataAssociation(p_d, p_g, settings.coalescence)
        elif settings.association == "mht":
            associator = TrackOrientedMHT(
                p_d,
                p_g,
                settings.k_best,
                settings.n_scan_depth,
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
            {_PLOTS: self.mean_density(scenario.start_time_s)},
            assignment=settings.assignment,
            coverage=self._covers,
            existence=ExistencePolicy(settings.survival_probability)
            if settings.existence
            else None,
            max_birth_clutter_density=settings.max_birth_clutter_density_per_m_rad,
        )
        if name == "pmbm":
            sigma = settings.sigma_acceleration_mps2 if settings.motion == "cv" else agile
            cv_prior = np.diag([v**2 for v in self._prior_sigma(cv.state_layout.names, settings)])
            self.tracker = PmbmTracker(
                ConstantVelocity("xy", sigma_acceleration_mps2=sigma),
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
                max_birth_clutter_density=settings.max_birth_clutter_density_per_m_rad,
                coverage=self._covers,
                assignment=settings.assignment,
            )
        if settings.seed_truth:
            self._seed(scenario)
        # The map starts at twice the birth threshold: no track starts where it has not looked.
        limit = settings.max_birth_clutter_density_per_m_rad
        self.clutter_map = ClutterMap(range_max_m=sensor.range_max_m, prior_per_m_rad=2 * limit)
        self._sector = 0

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

    def rotation(self, plots: PlotScan) -> list[tuple[TrackBank, Any]]:
        """Track one scan, sector by sector; return each batch's bank and JPDA clusters."""
        n_sectors = self.sensor.n_sectors
        # A drifting disc changes the regions' mean, so GNN's density is this scan's.
        if isinstance(self.tracker, Tracker):
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
            time_s = plots.start_time_s + (sector + 0.5) / n_sectors * self.period_s
            z = np.column_stack([plots.range_m[mine], plots.azimuth_rad[mine]])
            scan = Scan(time_s, z, self.noise_covariance, _PLOTS, density[mine])
            association = None
            if isinstance(self.tracker, Tracker):
                self.tracker.last_association = None
                bank = self.tracker.process(scan)
                association = self.tracker.last_association
            else:
                bank = self.tracker.process(scan)
            confirmed = is_confirmed(bank.status)
            used = bank.measurement_index[confirmed & (bank.measurement_index >= 0)]
            taken[mine[used]] = True
            results.append((bank, association))
        self.clutter_map.update(plots.range_m[~taken], plots.azimuth_rad[~taken])
        return results
