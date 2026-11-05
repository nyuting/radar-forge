r"""The chain from a scenario to tracks: frames of IQ, range-Doppler maps, plots and scans.

:func:`frames` plays a :class:`~radar_forge.scenario.Scenario` back one frame at a time, each
burst's cube synthesised from line-of-sight paths. :func:`range_doppler_product` makes each
cube a :class:`RangeDopplerProduct`; :func:`frame_plots` runs CFAR along range, clusters, and
pairs two bursts by dual PRF into :class:`Plots`; :class:`ScanTracker` feeds the accepted plots
to a :class:`~radar_forge.tracking.Tracker`. The map axes carry bistatic range
:math:`R_T + R_R` and its rate, positive receding; nothing here halves them. A plot-level
scenario skips the IQ: :func:`scans` plays its plots rotation by rotation and
:class:`PlotTracker` tracks them in the ground plane.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014.
.. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.constants import SPEED_OF_LIGHT_MPS
from radar_forge.detection import Detection, cfar_detect, cluster_detections
from radar_forge.dsp import (
    doppler_bin_centers_mps,
    doppler_fft,
    fold_velocity_mps,
    matched_filter,
    range_bin_centers_m,
    range_doppler_map,
    unfold_doppler_dual_prf,
    unfold_range_dual_prf,
)
from radar_forge.propagation import line_of_sight_paths
from radar_forge.radar import Radar, TargetGeometry
from radar_forge.scenario import DetectionSettings, Scenario, TrackingSettings, truth_geometry
from radar_forge.signal import fmcw_deramp_baseband, lfm_chirp, pulsed_baseband
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
from radar_forge.tracking.imm import InteractingMultipleModel
from radar_forge.tracking.initiation import DebiasedSphericalInitiator, DirectStateInitiator
from radar_forge.tracking.models import (
    CartesianMotion,
    CoordinatedTurn,
    LinearMeasurement,
    RadialConstantVelocity,
    RangeAzimuthElevationRangeRate,
    marginal,
)
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

__all__ = [
    "Frame",
    "PlotScan",
    "PlotTracker",
    "Plots",
    "RangeDopplerProduct",
    "ScanTracker",
    "frame_plots",
    "frames",
    "map_axes",
    "range_doppler_product",
    "scans",
    "synthesise",
]

# The model IDs a scan is measured through: the scan's own, and its range-only marginal.
_MEASURED, _RANGE = "range_rate", "range"
# A plot-level scan's one model, and the state coordinates a sector's coverage reads.
_PLOTS = "plots"
_POSITION_VELOCITY = ("x_m", "xdot_mps", "y_m", "ydot_mps")


def synthesise(
    radar: Radar,
    geometry: TargetGeometry,
    rcs_m2: ArrayLike,
    n_pulses: int,
    *,
    rng: np.random.Generator | None,
) -> NDArray[np.complex128]:
    """Return one burst's cube ``(n_pulses, n_samples_per_pri)`` for line-of-sight targets."""
    paths = line_of_sight_paths(radar, geometry, rcs_m2)
    generate = pulsed_baseband if radar.transmitter.waveform == "pulsed" else fmcw_deramp_baseband
    return generate(paths, radar, n_pulses, rng=rng)


@dataclass(frozen=True)
class Frame:
    """One frame: its time, the targets' true geometry, and one cube per burst."""

    index: int
    time_s: float
    truth: TargetGeometry
    cubes: tuple[NDArray[np.complex128], ...]


def frames(scenario: Scenario, *, n_frames: int | None = None) -> Iterator[Frame]:
    """Play a scenario back one frame at a time, every burst drawing noise from one generator.

    A generator, so a run holds one frame's cubes at a time. Consumed in order it replays bit
    for bit; skipping a frame changes the noise of every frame after it.
    """
    truths = truth_geometry(scenario)[:n_frames]
    rng = np.random.default_rng(scenario.seed)
    rcs_m2 = [target.rcs_m2 for target in scenario.targets]
    # A loop over frames is the point: each frame is built and handed out before the next.
    for index, truth in enumerate(truths):
        cubes = tuple(
            synthesise(radar, truth, rcs_m2, n_pulses, rng=rng)
            for radar, n_pulses in zip(scenario.radars, scenario.n_pulses, strict=True)
        )
        yield Frame(index, float(scenario.frame_times_s[index]), truth, cubes)


def map_axes(radar: Radar, n_pulses: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a burst's map axes: bistatic range of each range bin, and rate of each Doppler bin.

    FMCW range is the beat axis; a pulsed bin is one sample of delay, :math:`c n / f_s`.
    """
    transmitter, sample_rate_hz = radar.transmitter, radar.receiver.sample_rate_hz
    n_samples = radar.n_samples_per_pri
    if transmitter.waveform == "pulsed":
        range_axis_m = SPEED_OF_LIGHT_MPS * np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    else:
        range_axis_m = range_bin_centers_m(
            n_samples, transmitter.bandwidth_hz, transmitter.chirp_duration_s, sample_rate_hz
        )
    pri_s = transmitter.pulse_repetition_interval_s
    return range_axis_m, doppler_bin_centers_mps(n_pulses, pri_s, radar.wavelength_m)


@dataclass(frozen=True)
class RangeDopplerProduct:
    """One burst's map ``(n_doppler_bins, n_range_bins)`` and its axes (:func:`map_axes`).

    Both axes are circles: a fractional index converts by the axis's own uniform formula, so an
    index past the last bin lies across the wrap, and a rate folds into ``(-span/2, +span/2]``,
    span = :attr:`rate_period_mps`.
    The rate axis falls with index, its sign from :func:`~radar_forge.dsp.doppler_bin_centers_mps`.
    """

    rd_map: NDArray[np.complex128]
    range_axis_m: NDArray[np.float64]
    rate_axis_mps: NDArray[np.float64]

    @property
    def range_bin_m(self) -> float:
        """Range bin width, m."""
        return float(self.range_axis_m[1] - self.range_axis_m[0])

    @property
    def rate_bin_mps(self) -> float:
        """Rate step from one Doppler bin to the next, m/s: negative."""
        return float(self.rate_axis_mps[1] - self.rate_axis_mps[0])

    @property
    def range_period_m(self) -> float:
        """Span of the range axis, m."""
        return self.range_axis_m.size * self.range_bin_m

    @property
    def rate_period_mps(self) -> float:
        r"""Span of the rate axis, :math:`\lambda\,\mathrm{PRF}`, m/s."""
        return self.rate_axis_mps.size * abs(self.rate_bin_mps)

    def range_at_index_m(self, index: ArrayLike) -> NDArray[np.float64]:
        """Return the bistatic range at fractional range indices in ``[0, n_range_bins)``."""
        result: NDArray[np.float64] = self.range_axis_m[0] + np.asarray(index) * self.range_bin_m
        return result

    def rate_at_index_mps(self, index: ArrayLike) -> NDArray[np.float64]:
        """Return the folded rate at fractional Doppler indices, in ``(-span/2, +span/2]``."""
        rate_mps = self.rate_axis_mps[0] + np.asarray(index) * self.rate_bin_mps
        return fold_velocity_mps(rate_mps, self.rate_period_mps)

    def index_of(
        self, bistatic_range_m: ArrayLike, rate_mps: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the fractional ``(range, Doppler)`` indices where unfolded values appear."""
        range_index = np.mod(np.asarray(bistatic_range_m), self.range_period_m) / self.range_bin_m
        offset_mps = np.asarray(rate_mps) - self.rate_axis_mps[0]
        return range_index, np.mod(offset_mps / self.rate_bin_mps, self.rate_axis_mps.size)


def range_doppler_product(cube: ArrayLike, radar: Radar) -> RangeDopplerProduct:
    """Form one burst's map: FMCW, two FFTs; pulsed, matched filter then the Doppler FFT.

    The receiver keeps listening, so the pulsed chain compresses the rows as one stream and
    cuts it back into rows from the reference's group delay, ``n_reference - 1``: zero delay
    lands at bin 0 (Richards [1] §4.6), an echo that runs into the next row compresses whole,
    and every cell holds the same noise. No window is applied on either axis.
    """
    samples = np.asarray(cube, dtype=np.complex128)
    transmitter = radar.transmitter
    if transmitter.waveform == "pulsed":
        reference = lfm_chirp(
            transmitter.bandwidth_hz, transmitter.chirp_duration_s, radar.receiver.sample_rate_hz
        )
        stream = matched_filter(samples.reshape(-1), reference)
        start = reference.size - 1
        rd_map = doppler_fft(stream[start : start + samples.size].reshape(samples.shape), axis=0)
    else:
        rd_map = range_doppler_map(samples)
    return RangeDopplerProduct(rd_map, *map_axes(radar, samples.shape[0]))


@dataclass(frozen=True)
class Plots:
    """Every detection of one frame, every burst's, as columns ``(n_plots,)``.

    ``bistatic_range_m`` and ``folded_rate_mps`` are as the burst's map shows them;
    ``bistatic_range_rate_mps`` is the unfolded rate, nan until a fold is known. The sigmas are
    a bin over √12, the quantisation floor (Richards [1] §7.1). ``status`` says whether a plot
    reaches the tracker and, after dual-PRF pairing, why not; ``pair_id`` is -1 unpaired.
    """

    detections: tuple[Detection, ...]
    burst_index: NDArray[np.int64]
    bistatic_range_m: NDArray[np.float64]
    folded_rate_mps: NDArray[np.float64]
    bistatic_range_rate_mps: NDArray[np.float64]
    sigma_range_m: NDArray[np.float64]
    sigma_rate_mps: NDArray[np.float64]
    status: tuple[str, ...]
    pair_id: NDArray[np.int64]
    unfolded_range_m: NDArray[np.float64]

    @property
    def accepted(self) -> NDArray[np.intp]:
        """Indices of the plots that reach the tracker: the first burst's accepted ones."""
        accepted = np.array([s == "accepted" for s in self.status], dtype=bool)
        return np.flatnonzero(accepted & (self.burst_index == 0))


def _merge_sidelobes(
    clusters: list[Detection], merge_range_bins: float, n_range_bins: int
) -> list[Detection]:
    """Keep each cluster unless a stronger one lies within ``merge_range_bins`` round the circle."""
    if merge_range_bins <= 0.0:
        return clusters
    kept: list[Detection] = []
    # Each cluster is kept or dropped against those already kept, strongest first: sequential.
    for cluster in clusters:
        gaps = [abs(cluster.centroid_index[1] - k.centroid_index[1]) for k in kept]
        if all(min(gap, n_range_bins - gap) > merge_range_bins for gap in gaps):
            kept.append(cluster)
    return kept


def _detect(
    product: RangeDopplerProduct, settings: DetectionSettings, burst_index: int, max_rate_mps: float
) -> Plots:
    """Run CFAR along range, cluster on the circle of both axes, merge sidelobes, convert."""
    power_w = np.abs(product.rd_map) ** 2
    detected, threshold_w = cfar_detect(
        power_w,
        settings.pfa,
        n_train=settings.n_train,
        n_guard=settings.n_guard,
        variant=settings.variant,
        axis=-1,
        wrap=True,
    )
    clusters = cluster_detections(detected, power_w, threshold_w=threshold_w, wrap_axes=(0, 1))
    kept = tuple(_merge_sidelobes(clusters, settings.merge_range_bins, power_w.shape[1]))
    rate_index, range_index = np.array([d.centroid_index for d in kept]).reshape(-1, 2).T
    folded_mps = product.rate_at_index_mps(rate_index)
    # One burst wider than any rate the target can have cannot fold: its reading is the rate.
    unfolded_mps = folded_mps if product.rate_period_mps / 2.0 > max_rate_mps else np.nan
    ones = np.ones(len(kept))
    return Plots(
        kept,
        np.full(len(kept), burst_index, dtype=np.int64),
        product.range_at_index_m(range_index),
        folded_mps,
        unfolded_mps * ones,
        ones * abs(product.range_bin_m) / math.sqrt(12.0),
        ones * abs(product.rate_bin_mps) / math.sqrt(12.0),
        ("accepted",) * len(kept),
        np.full(len(kept), -1, dtype=np.int64),
        np.full(len(kept), np.nan),
    )


def _strictly_nearest(gap_m: NDArray[np.float64], axis: int) -> NDArray[np.bool_]:
    """Mark the one smallest entry of each line along ``axis``; a tie marks none."""
    if gap_m.shape[axis] == 0:
        return np.zeros(gap_m.shape, dtype=bool)
    smallest = gap_m == gap_m.min(axis=axis, keepdims=True)
    return np.asarray(smallest & (smallest.sum(axis=axis, keepdims=True) == 1), dtype=bool)


def _pair(
    first: Plots, second: Plots, products: Sequence[RangeDopplerProduct], max_rate_mps: float
) -> Plots:
    """Pair two bursts' plots by mutual nearest range, then unfold each pair's rate.

    Ranges are unfolded as rates are, so maps of different spans still pair; a pair must agree
    within two bins of the coarser map in each (Richards [1] §5.5.4). A plot with no partner in
    range is ``missing_pair``, one whose partner is nearer another ``ambiguous_pair``, and a pair
    whose rates agree on no value ``unresolved_rate``.
    """
    a, b = products
    range_tolerance_m = 2.0 * max(abs(a.range_bin_m), abs(b.range_bin_m))
    unfolded_m, distance_m = unfold_range_dual_prf(
        first.bistatic_range_m[:, None],
        second.bistatic_range_m[None, :],
        a.range_period_m,
        b.range_period_m,
        max_range_m=max(a.range_period_m, b.range_period_m),
        tolerance_m=range_tolerance_m,
    )
    compatible = distance_m <= range_tolerance_m
    mutual = compatible & _strictly_nearest(distance_m, 1) & _strictly_nearest(distance_m, 0)
    i, j = np.nonzero(mutual)
    rate_mps, _ = unfold_doppler_dual_prf(
        first.folded_rate_mps[i],
        second.folded_rate_mps[j],
        a.rate_period_mps,
        b.rate_period_mps,
        max_velocity_mps=max_rate_mps,
        tolerance_mps=2.0 * max(abs(a.rate_bin_mps), abs(b.rate_bin_mps)),
    )
    members = np.concatenate([i, first.burst_index.size + j])
    both_mps = np.concatenate([rate_mps, rate_mps])
    status = np.array(["missing_pair"] * (len(first.status) + len(second.status)), dtype=object)
    status[np.concatenate([compatible.any(axis=1), compatible.any(axis=0)])] = "ambiguous_pair"
    status[members] = np.where(np.isfinite(both_mps), "accepted", "unresolved_rate")
    columns: dict[str, Any] = {
        f.name: np.concatenate([getattr(first, f.name), getattr(second, f.name)])
        for f in fields(Plots)
        if f.name not in ("detections", "status")
    }
    columns["pair_id"][members] = np.concatenate([np.arange(i.size)] * 2)
    columns["bistatic_range_rate_mps"][members] = both_mps
    columns["unfolded_range_m"][members] = np.concatenate([unfolded_m[i, j]] * 2)
    return Plots(detections=first.detections + second.detections, status=tuple(status), **columns)


def frame_plots(products: Sequence[RangeDopplerProduct], settings: DetectionSettings) -> Plots:
    """Return one frame's plots from each burst's map, a dual-PRF pair paired and unfolded."""
    max_rate_mps = settings.max_bistatic_range_rate_mps
    plots = [_detect(p, settings, k, max_rate_mps) for k, p in enumerate(products)]
    return plots[0] if len(plots) == 1 else _pair(plots[0], plots[1], products, max_rate_mps)


class ScanTracker:
    r"""The ``[tracking]`` table as a :class:`~radar_forge.tracking.Tracker`, and its scan policy.

    The rate is measured where it cannot fold: after dual-PRF pairing, from one burst whose
    rate interval exceeds the largest rate, or once a track can unfold it; otherwise range
    alone. Range is measured modulo the first map's span (the state stays continuous), with a
    bin over √12 of noise. ``rate_unfolding = "track_aided"`` is scenario 003's policy: a
    track may unfold after N associated ranges, the least N whose least-squares slope is
    certain to a ``unfold_sigma_gate``-th of a fold span, :math:`(\sigma_R/T)\sqrt{12/(N(N^2-1))}
    < v_{span}/g` (Bar-Shalom [2] §5.5). The track with the longest such history gives each
    plot the fold nearest its rate, unless that lies over half a span from the slope of its last
    ``n_slope_scans`` ranges, the fold monitor: a wrong fold agrees with the prediction that
    chose it, never with the range. One reference track serves a scan: single-target policy.
    """

    def __init__(self, scenario: Scenario) -> None:
        settings = scenario.tracking
        if settings is None:
            msg = f"scenario {scenario.name!r} has no [tracking] table."
            raise ValueError(msg)
        range_axis_m, rate_axis_mps = map_axes(scenario.radar, scenario.n_pulses[0])
        range_bin_m = float(range_axis_m[1] - range_axis_m[0])
        rate_bin_mps = float(rate_axis_mps[0] - rate_axis_mps[1])
        self.range_period_m = range_axis_m.size * range_bin_m
        self.fold_span_mps = rate_axis_mps.size * rate_bin_mps
        self.scan_interval_s, self.n_slope_scans = (
            1.0 / scenario.frame_rate_hz,
            settings.n_slope_scans,
        )
        sigma_range_m, sigma_rate_mps = (
            range_bin_m / math.sqrt(12.0),
            rate_bin_mps / math.sqrt(12.0),
        )
        max_rate_mps, dual = (
            scenario.detection.max_bistatic_range_rate_mps,
            len(scenario.radars) == 2,
        )
        track_aided = {"track_aided": True, "none": False}[settings.rate_unfolding]
        self.measures_rate = dual or track_aided or self.fold_span_mps / 2.0 > max_rate_mps
        names = ("bistatic_range_m", "bistatic_range_rate_mps")[: 1 + self.measures_rate]
        motion = RadialConstantVelocity(
            settings.sigma_acceleration_mps2,
            acceleration_correlation_time_s=settings.acceleration_correlation_time_s,
            process_noise=settings.process_noise,
        )
        model = LinearMeasurement(
            motion.state_layout, names, periods={names[0]: self.range_period_m}
        )
        models: dict[str, Any] = {_MEASURED: model}
        if self.measures_rate and (settings.range_only_retry or track_aided):
            models[_RANGE] = marginal(model, names[:1])
        # One false alarm per scan over the range span and over the rate span a plot can take;
        # only their ratio decides an assignment, since GNN maximises the pairs first.
        rate_span_mps = 2.0 * max_rate_mps if dual else self.fold_span_mps
        density = {
            _MEASURED: 1.0 / self.range_period_m / (rate_span_mps if self.measures_rate else 1)
        }
        density[_RANGE] = 1.0 / self.range_period_m
        n_scans = np.arange(2, 1000)
        slope_sigma_mps = (
            sigma_range_m / self.scan_interval_s * np.sqrt(12 / (n_scans**3 - n_scans))
        )
        enough = n_scans[slope_sigma_mps < self.fold_span_mps / settings.unfold_sigma_gate]
        if track_aided and not enough.size:
            msg = "the range slope never resolves the fold within 1000 scans; widen the gate."
            raise ValueError(msg)
        # Tracks unfold once they hold this many ranges; None without track-aided unfolding.
        self.n_unfold_scans = int(enough[0]) if track_aided else None
        # Each track's (scan time, unwrapped range) pairs, from its birth on.
        self.history: dict[int, list[tuple[float, float]]] = {}
        n_lives = settings.n_delete_misses + settings.n_reacquire_misses
        self.tracker = Tracker(
            models,
            motion,
            {"taylor": TaylorLinearisation, "unscented": UnscentedTransform}[
                settings.linearisation
            ](),
            settings.gate_probability,
            GlobalNearestNeighbour(),
            DirectStateInitiator(
                np.zeros(2),
                np.diag([sigma_range_m**2, max_rate_mps**2]),
                seed_names={"range_only": names[:1], "direct": None}[settings.initiator],
            ),
            LifecyclePolicy(
                settings.n_confirm_hits,
                settings.n_confirm_scans,
                settings.n_delete_misses,
                settings.n_reacquire_misses,
                # The miss count deletes; the coast-time limit is set past it, never to fire.
                max_coast_time_s=(n_lives + 1) * self.scan_interval_s,
                reacquire_reset=settings.reacquire_reset,
            ),
            settings.cost,
            {k: density[k] for k in models} if settings.cost != "nis" else None,
            assignment=settings.assignment,
            model_selector=self._select if len(models) > 1 else None,
        )
        noise = np.diag([sigma_range_m**2, sigma_rate_mps**2])
        self.noise_covariance = noise[: len(names), : len(names)]

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
        """Return each folded rate moved by whole spans to the fold nearest ``reference_mps``."""
        span_mps = self.fold_span_mps
        result: NDArray[np.float64] = folded_mps + span_mps * np.round(
            (reference_mps - folded_mps) / span_mps
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
        disagree = np.abs(unfolded_mps - slope_mps) > self.fold_span_mps / 2.0
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
                range_m = last_m + (range_m - last_m + period_m / 2) % period_m - period_m / 2
            history.append((time_s, range_m))

    def scan(self, plots: Plots, time_s: float) -> tuple[TrackBank, NDArray[np.intp]]:
        """Track a frame's accepted plots; return the bank and the plot behind each measurement."""
        accepted = plots.accepted
        rate_mps = plots.bistatic_range_rate_mps[accepted]
        if self.n_unfold_scans is not None:
            rate_mps = self.unfold(plots.folded_rate_mps[accepted])
        measurements = np.column_stack([plots.bistatic_range_m[accepted], rate_mps])
        measurements = measurements[:, : self.noise_covariance.shape[0]]
        bank = self.tracker.process(Scan(time_s, measurements, self.noise_covariance, _MEASURED))
        self._record(bank, measurements[:, 0], time_s)
        return bank, accepted


# Plot-level scenarios (scenario 011): a rotating radar's plots, and the tracker they feed.

# The fixed-point iterations that find when the beam crosses a moving target: its azimuth moves
# under a degree a rotation, so each iteration shrinks the time error a hundredfold.
_BEAM_ITERATIONS = 3


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
        _, azimuth = polar(np.nan_to_num(position_m))
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


class PlotTracker:
    """The ``[tracking]`` table as a ground-plane :class:`~radar_forge.tracking.Tracker`.

    The state is ENU (x, y) and velocity, plus the turn rate under ``motion = "imm"``, whose
    modes are CV quiet, coordinated turn and CV agile; z is (horizontal range, azimuth), the
    radar at the origin and z fixed at its height. A rotation reaches the tracker as
    ``n_sectors`` batches, each at its centre time, and a batch sees only the tracks whose
    extrapolated azimuth lies in its sector. GNN costs by the likelihood ratio with no
    assignment cheaper than ln(P_D / (1 - P_D P_G)), the track score's break-even; JPDA weighs
    confirmed tracks jointly and tentative ones alone, with the clutter intensity at each plot
    from the scene (``"oracle"``) or a :class:`~radar_forge.tracking.scene.ClutterMap`
    (``"map"``) fed every plot no confirmed track took.
    """

    def __init__(self, scenario: Scenario) -> None:
        settings, sensor = scenario.tracking, scenario.sensor
        if settings is None or sensor is None:
            msg = f"scenario {scenario.name!r} needs [sensor] and [tracking] to track plots."
            raise ValueError(msg)
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
        components = scenario.clutter_plots.components
        volume = sum(c.volume_m_rad(scenario.start_time_s) for c in components)
        plots = sum(
            c.intensity_per_m_rad * c.volume_m_rad(scenario.start_time_s) for c in components
        )
        self.mean_intensity = max(plots / volume if volume else 0.0, LEAST_INTENSITY_PER_M_RAD)
        associator: Any = GlobalNearestNeighbour(math.log(p_d / (1 - p_d * p_g)))
        if settings.association == "jpda":
            associator = JointProbabilisticDataAssociation(p_d, p_g, settings.coalescence)
        self.tracker = Tracker(
            {_PLOTS: model},
            motion,
            {"taylor": TaylorLinearisation, "unscented": UnscentedTransform}[
                settings.linearisation
            ](),
            p_g,
            associator,
            DebiasedSphericalInitiator(np.zeros(layout.dimension), prior),
            LifecyclePolicy(
                settings.n_confirm_hits,
                settings.n_confirm_scans,
                settings.n_delete_misses,
                max_coast_time_s=(settings.n_delete_misses + 1) * self.period_s,
                max_position_sigma_m=min(settings.max_position_sigma_m, 1e12),
            ),
            "negative_log_likelihood_ratio",
            {_PLOTS: self.mean_intensity},
            assignment=settings.assignment,
            coverage=self._covers,
            existence=ExistencePolicy() if settings.existence else None,
            max_birth_clutter_intensity=settings.max_birth_clutter_intensity_per_m_rad,
        )
        # The map starts at twice the birth threshold: no track starts where it has not looked.
        limit = settings.max_birth_clutter_intensity_per_m_rad
        self.clutter_map = ClutterMap(range_max_m=sensor.range_max_m, prior_per_m_rad=2 * limit)
        self._sector = 0

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

    def rotation(self, plots: PlotScan) -> list[tuple[TrackBank, Any]]:
        """Track one rotation, sector by sector; return each batch's bank and JPDA clusters."""
        n_sectors = self.sensor.n_sectors
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
            self.tracker.last_association = None
            bank = self.tracker.process(scan)
            confirmed = np.isin(bank.status, (TrackStatus.CONFIRMED, TrackStatus.COASTING))
            used = bank.measurement_index[confirmed & (bank.measurement_index >= 0)]
            taken[mine[used]] = True
            results.append((bank, self.tracker.last_association))
        self.clutter_map.update(plots.range_m[~taken], plots.azimuth_rad[~taken])
        return results
