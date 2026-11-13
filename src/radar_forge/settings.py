r"""The settings a scenario's TOML tables hold: one frozen dataclass a table, checked when built.

:mod:`radar_forge.scenario` reads a TOML file into these and assembles the
:class:`~radar_forge.scenario.Scenario`; each class names its table and validates its own keys'
values, so a bad value is refused by name before anything runs. :class:`Target` and
:class:`ArcTarget` are a ``[[target]]``'s two kinds; the rest set detection, tracking, a plot-level
sensor and its clutter, range-migration correction, and a noise radar's leakage, near clutter,
transmitter distortion and canceller; scenario 006's, an FMCW radar's receive array, the second
aircraft built from the first, and the DOA estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from radar_forge.detection import CfarVariant
from radar_forge.doa import label_parts
from radar_forge.propagation import dbsm_to_m2
from radar_forge.tracking.hypotheses import CONFIRM_SCORE, DELETE_DROP
from radar_forge.tracking.scene import Arc, ClutterComponent, FixedDiscretes

__all__ = [
    "ArcTarget",
    "CancellationSettings",
    "ClutterPlotSettings",
    "ClutterSettings",
    "DetectionSettings",
    "DistortionSettings",
    "DoaSettings",
    "LeakageSettings",
    "MigrationSettings",
    "ReceiveArraySettings",
    "SecondAircraftSettings",
    "SensorSettings",
    "Target",
    "TrackingSettings",
]


@dataclass(frozen=True)
class Target:
    r"""One point target: a trajectory CSV at a fixed altitude, or a constant ENU motion.

    ``position_enu_m`` is at the first frame, in the receiver site's east, north, up frame; the
    target then follows :math:`p + v t + a t^2/2`. Without ``acceleration_enu_mps2`` its
    route's :math:`\ddot R_b` is not modelled (zero), as for a trajectory.
    """

    name: str
    rcs_dbsm: float
    trajectory_path: Path | None = None
    altitude_m: float = 0.0
    position_enu_m: tuple[float, float, float] | None = None
    velocity_enu_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration_enu_mps2: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        """Require exactly one way of moving; an acceleration only with a position."""
        if (self.trajectory_path is None) == (self.position_enu_m is None):
            msg = f"target {self.name!r} needs one of trajectory_path and position_enu_m."
            raise ValueError(msg)
        if self.acceleration_enu_mps2 is not None and self.position_enu_m is None:
            msg = f"target {self.name!r}: acceleration_enu_mps2 needs position_enu_m."
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
    dual-PRF search bound and a new track's prior rate spread. Crossings are grouped as
    connected cells, or, with ``mean_shift_bandwidth_bins`` above 0, by the mode of the power
    each climbs to within that many bins, which separates two peaks that touch.
    """

    pfa: float = 1e-5
    n_train: int = 16
    n_guard: int = 4
    variant: CfarVariant = "ca"
    merge_range_bins: float = 1.0
    mean_shift_bandwidth_bins: float = 0.0
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
    # Plot-level tracking (scenario 011): ENU ground-plane states, z = (range, azimuth).
    motion: Literal["cv", "imm"] = "cv"
    association: Literal["gnn", "jpda", "mht"] = "gnn"
    coalescence: Literal["none", "jpda_star", "set"] = "none"
    existence: bool = False
    mode_sigma_acceleration_mps2: tuple[float, ...] = (0.5, 0.5, 3.1623)
    turn_rate_noise_density_rad2ps3: float = 1.6e-5
    sojourn_time_s: tuple[float, ...] = (120.0, 30.0, 20.0)
    turn_rate_sigma_degps: float = 3.0
    clutter_intensity: Literal["oracle", "map"] = "oracle"
    detection_probability: float = 0.9
    max_birth_clutter_intensity_per_m_rad: float = 0.01
    initial_speed_sigma_mps: float = 150.0
    max_position_sigma_m: float = np.inf
    # Scenario 011b: MHT's tree, PMBM's births and pruning, cues at the truth, and further
    # trackers to score beside the configured one (tracking.scans.VARIANTS, or "pmbm").
    k_best: int = 10
    n_scan: int = 3
    confirm_score: float = CONFIRM_SCORE
    delete_drop: float = DELETE_DROP
    birth_intensity_per_scan: float = 0.05
    survival_probability: float = 0.995
    prune_existence: float = 1e-5
    prune_weight: float = 1e-4
    seed_truth: bool = False
    trackers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse track-aided unfolding behind a folded seed, and existence without JPDA."""
        names = ("gnn", "jpda", "jpda_star", "set", "jipda", "mht", "pmbm")
        if not set(self.trackers) <= set(names) or len(set(self.trackers)) < len(self.trackers):
            msg = f"[tracking] trackers must be distinct names among {list(names)}."
            raise ValueError(msg)
        if self.rate_unfolding == "track_aided" and self.initiator != "range_only":
            msg = (
                '[tracking] rate_unfolding = "track_aided" needs initiator = "range_only": a '
                f"{self.initiator!r} seed takes the folded rate, at a bin of variance."
            )
            raise ValueError(msg)
        if self.existence and self.association != "jpda":
            msg = '[tracking] existence = true is JIPDA: it needs association = "jpda".'
            raise ValueError(msg)
        if len(self.mode_sigma_acceleration_mps2) != 3 or len(self.sojourn_time_s) != 3:
            msg = "[tracking] the IMM's modes are quiet, turn and agile: give three values each."
            raise ValueError(msg)


@dataclass(frozen=True)
class SensorSettings:
    """``[sensor]``: a rotating surveillance radar's plots, z = (horizontal range, azimuth).

    The beam crosses azimuth a at ``scan start + a / (2π) scan_period_s``; a rotation reaches
    the tracker in ``n_sectors`` batches, each at its centre time. Echoes merge within the
    resolution cell ``range_cell_m`` by ``beamwidth_azimuth_deg``.
    """

    carrier_frequency_hz: float = 2.8e9
    scan_period_s: float = 4.8
    beamwidth_azimuth_deg: float = 1.4
    range_cell_m: float = 115.75
    range_min_m: float = 1000.0
    range_max_m: float = 40000.0
    sigma_range_m: float = 33.414
    sigma_azimuth_deg: float = 0.15
    pd_aircraft: float = 0.9
    pd_bird: float = 0.3
    mti_notch_mps: float = 4.0
    pd_notch: float = 0.3
    n_sectors: int = 16


@dataclass(frozen=True)
class ArcTarget:
    """One ``[[target]] kind = "arc"``: its class (``aircraft`` or ``bird``) and trajectory."""

    name: str
    target_class: Literal["aircraft", "bird"]
    arc: Arc


@dataclass(frozen=True)
class ClutterPlotSettings:
    """``[clutter_plots]``: Poisson components and fixed discretes, in plots per (m rad)."""

    components: tuple[ClutterComponent, ...] = ()
    discretes: FixedDiscretes | None = None


@dataclass(frozen=True)
class LeakageSettings:
    """``[leakage]``: the transmitter's leakage at the ADC, one whole-sample lag per path.

    ``inr_db`` is each path's power over the noise in one sample, not over the matched band.
    """

    lags: tuple[int, ...] = ()
    inr_db: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Require one non-negative whole lag per level."""
        lags_ok = all(type(lag) is int and lag >= 0 for lag in self.lags)
        if not (self.lags and lags_ok and len(self.lags) == len(self.inr_db)):
            msg = "[leakage] needs lags, non-negative integers, and one inr_db for each."
            raise ValueError(msg)
        object.__setattr__(self, "inr_db", tuple(float(value) for value in self.inr_db))


@dataclass(frozen=True)
class ClutterSettings:
    """One ``[[clutter]]`` scatterer: its bistatic route, power over noise per sample, Doppler."""

    range_m: float = 0.0
    relative_power_db: float = 0.0
    doppler_hz: float = 0.0

    def __post_init__(self) -> None:
        """Require a route: a scatterer at zero range is leakage."""
        if not self.range_m > 0.0:
            msg = f"[[clutter]] range_m must be strictly positive; got {self.range_m!r}."
            raise ValueError(msg)


@dataclass(frozen=True)
class DistortionSettings:
    """``[transmitter_distortion]``: the third-order term of the transmit amplifier."""

    a3: float = 0.0


@dataclass(frozen=True)
class CancellationSettings:
    """``[cancellation]``: the canceller a noise burst's stream passes before correlation.

    ``least_squares`` fits the CPI, or each ``n_samples_per_batch`` block, on the block widened
    by ``n_samples_extension`` a side (:func:`~radar_forge.clutter.least_squares_cancel`);
    ``wiener`` is Wiener-SMI on the CPI, linear; ``none`` passes the stream through.
    """

    method: Literal["none", "least_squares", "wiener"] = "least_squares"
    n_taps: int = 1
    n_samples_per_batch: int = 0
    n_samples_extension: int = 0
    odd_orders: tuple[int, ...] = (1,)
    diagonal_loading: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a count below its floor, or a Wiener fit asked to batch or to go nonlinear."""
        counts = (self.n_taps - 1, self.n_samples_per_batch, self.n_samples_extension)
        if min(counts) < 0 or (self.n_samples_extension and not self.n_samples_per_batch):
            msg = "[cancellation] needs n_taps >= 1, and an extension only around a batch."
            raise ValueError(msg)
        if not self.diagonal_loading >= 0.0:
            msg = "[cancellation] diagonal_loading is a fraction of the mean diagonal, >= 0."
            raise ValueError(msg)
        orders = self.odd_orders
        if not orders or len(set(orders)) < len(orders) or any(p < 1 or p % 2 == 0 for p in orders):
            msg = (
                f"[cancellation] odd_orders must be distinct odd orders, such as [1, 3]; {orders}."
            )
            raise ValueError(msg)
        tuned = self.n_samples_per_batch or orders != (1,) or self.diagonal_loading
        if self.method == "wiener" and tuned:
            msg = '[cancellation] "wiener" fits the CPI on the linear basis, unloaded.'
            raise ValueError(msg)

    def check_fits(self, n_samples: int) -> None:
        """Refuse blocks that do not tile a CPI of ``n_samples``, or a window short of the basis."""
        if self.method == "none":
            return
        block = self.n_samples_per_batch or n_samples
        if n_samples % block or self.n_samples_extension % block:
            msg = (
                f"[cancellation] n_samples_per_batch must divide the CPI's {n_samples} samples, "
                "and n_samples_extension be whole batches."
            )
            raise ValueError(msg)
        n_blocks, reach = n_samples // block, self.n_samples_extension // block
        smallest = min(n_blocks, reach + 1) * block
        if smallest < self.n_taps * len(self.odd_orders):
            msg = f"[cancellation] a {smallest}-sample window holds fewer samples than its basis."
            raise ValueError(msg)


@dataclass(frozen=True)
class MigrationSettings:
    """``[migration]``: range-migration correction between pulse compression and Doppler FFT.

    ``"keystone"`` removes linear walk (:func:`~radar_forge.migration.keystone`, interpolating
    by ``keystone``); ``"keystone_doppler_rate"`` then searches ``acceleration_search_mps2 =
    [low, high, step]``, bistatic and inclusive, with the Doppler-rate filter. A positive
    ``fold_factor_max`` searches the keystone's fold factor over ``[-max, max]``.
    """

    method: Literal["none", "keystone", "keystone_doppler_rate"] = "none"
    keystone: Literal["czt", "sinc"] = "czt"
    fold_factor_max: int = 0
    acceleration_search_mps2: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a search the method does not run, or a grid that is not ``[low, high, step]``."""
        grid = tuple(float(value) for value in self.acceleration_search_mps2)
        object.__setattr__(self, "acceleration_search_mps2", grid)
        searches = self.method == "keystone_doppler_rate"
        if self.fold_factor_max < 0 or (self.method == "none" and self.fold_factor_max):
            msg = "[migration] fold_factor_max must be >= 0, and needs a keystone method."
            raise ValueError(msg)
        if searches != bool(grid) or (grid and not (len(grid) == 3 and grid[0] <= grid[1])):
            msg = (
                "[migration] acceleration_search_mps2 = [low, high, step], low <= high, goes "
                'with method = "keystone_doppler_rate" and only with it.'
            )
            raise ValueError(msg)
        if grid and not grid[2] > 0.0:
            msg = f"[migration] acceleration_search_mps2 step must be positive; got {grid[2]}."
            raise ValueError(msg)
        steps = (grid[1] - grid[0]) / grid[2] if grid else 0.0
        if abs(steps - round(steps)) > 1e-9:
            msg = (
                f"[migration] acceleration_search_mps2 = {list(grid)}: high - low is {steps:g} "
                "steps, not a whole number; set high to low plus a whole number of steps."
            )
            raise ValueError(msg)

    @property
    def accelerations_mps2(self) -> NDArray[np.float64]:
        """The acceleration grid, ``low`` to ``high`` inclusive; empty without a search."""
        if not self.acceleration_search_mps2:
            return np.zeros(0)
        low, high, step = self.acceleration_search_mps2
        steps = np.arange(round((high - low) / step) + 1, dtype=np.float64)
        result: NDArray[np.float64] = low + step * steps
        return result


@dataclass(frozen=True)
class ReceiveArraySettings:
    """``[receive_array]``: a ULA of ``n_rx`` channels, each with ``[receiver]``'s gain and NF."""

    n_rx: int = 16
    element_spacing_wavelengths: float = 0.5
    boresight_azimuth_deg: float = 0.0
    boresight_elevation_deg: float = 0.0

    def __post_init__(self) -> None:
        """Refuse fewer than two channels or a spacing that is not positive."""
        if self.n_rx < 2 or not self.element_spacing_wavelengths > 0.0:
            msg = "[receive_array] needs n_rx >= 2 and element_spacing_wavelengths > 0."
            raise ValueError(msg)


@dataclass(frozen=True)
class SecondAircraftSettings:
    """``[second_aircraft]``: aircraft B, the first target moved in the array's coordinates.

    B keeps the first target's ranges, rate and array-frame elevation; its direction cosine moves
    ``separation_rayleigh`` Rayleigh widths towards +x. ``scattering_phase``: ``"uniform"`` or rad.
    """

    separation_rayleigh: float = 1.0
    rcs_dbsm: float = 10.0
    scattering_phase: str | float = "uniform"
    name: str = "aircraft-b"

    def __post_init__(self) -> None:
        """Refuse a scattering phase that is neither ``"uniform"`` nor a number."""
        phase = self.scattering_phase
        if type(phase) is int:
            object.__setattr__(self, "scattering_phase", float(phase))
        elif phase != "uniform" and type(phase) is not float:
            msg = f'[second_aircraft] scattering_phase is "uniform" or radians; got {phase!r}.'
            raise ValueError(msg)


@dataclass(frozen=True)
class DoaSettings:
    """``[doa]``: what the ``doa`` stage estimates on each detection (:mod:`radar_forge.doa`).

    ``methods`` are :data:`~radar_forge.doa.DOA_LABELS`. Capon, MUSIC, root-MUSIC and ESPRIT
    (raw, ``fb_`` or ``fbss_``) train on the chip's ``(2 half_width_bins + 1)^2`` cells;
    Bartlett and the snapshot methods take its peak cell.
    """

    methods: tuple[str, ...] = ("bartlett", "music", "fbss_music", "iaa", "deterministic_ml")
    n_sources: int = 2
    n_subarray_elements: int = 12
    n_iaa_iterations: int = 15
    n_spice_iterations: int = 60
    n_grid: int = 4096
    half_width_bins: int = 2

    def __post_init__(self) -> None:
        """Refuse no method, an unknown one, or a count below its floor."""
        if not self.methods:
            msg = "[doa] methods must name at least one of radar_forge.doa.DOA_LABELS."
            raise ValueError(msg)
        for label in self.methods:
            label_parts(label)
        counts = (self.n_sources, self.n_iaa_iterations, self.n_spice_iterations)
        if min(counts) < 1 or self.n_grid < 3 or self.half_width_bins < 1:
            msg = "[doa] needs counts >= 1, n_grid >= 3 and half_width_bins >= 1."
            raise ValueError(msg)
