r"""The chain from a scenario to tracks: frames of IQ, range-Doppler maps, plots and scans.

:func:`frames` plays a :class:`~radar_forge.scenario.Scenario` back one frame at a time, each
burst's cube synthesised from line-of-sight paths. :func:`range_doppler_product` makes each
cube a :class:`RangeDopplerProduct`, a pulsed cube through its range gate and ``[migration]``
(:mod:`radar_forge.migration`); :func:`frame_plots` runs CFAR along range, clusters, and
pairs two bursts by dual PRF into :class:`Plots`; :class:`ScanTracker` feeds the accepted plots
to a :class:`~radar_forge.tracking.Tracker`. The map axes carry bistatic range
:math:`R_T + R_R` and its rate, positive receding; nothing here halves them.

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

from radar_forge.detection import Detection, cfar_detect, cluster_detections
from radar_forge.dsp import (
    delay_bin_centers_m,
    doppler_bin_centers_mps,
    doppler_fft,
    fold_velocity_mps,
    range_bin_centers_m,
    range_doppler_map,
    unfold_doppler_dual_prf,
    unfold_range_dual_prf,
)
from radar_forge.migration import compensate_migration, pulse_compress
from radar_forge.propagation import line_of_sight_paths
from radar_forge.radar import Radar, TargetGeometry
from radar_forge.scenario import (
    DetectionSettings,
    MigrationSettings,
    Scenario,
    truth_geometry,
)
from radar_forge.signal import RangeGate, fmcw_deramp_baseband, pulsed_baseband
from radar_forge.tracking.association import GlobalNearestNeighbour
from radar_forge.tracking.filters import (
    TaylorLinearisation,
    TrackBank,
    TrackStatus,
    UnscentedTransform,
)
from radar_forge.tracking.models import LinearMeasurement, RadialConstantVelocity, marginal
from radar_forge.tracking.tracker import DirectStateInitiator, LifecyclePolicy, Scan, Tracker

__all__ = [
    "Frame",
    "Plots",
    "RangeDopplerProduct",
    "ScanTracker",
    "frame_plots",
    "frame_products",
    "frames",
    "map_axes",
    "range_doppler_product",
    "synthesise",
]

# The model IDs a scan is measured through: the scan's own, and its range-only marginal.
_MEASURED, _RANGE = "range_rate", "range"


def synthesise(
    radar: Radar,
    geometry: TargetGeometry,
    rcs_m2: ArrayLike,
    n_pulses: int,
    *,
    rng: np.random.Generator | None,
    gate: RangeGate | None = None,
) -> NDArray[np.complex128]:
    """Return one burst's cube ``(n_pulses, n_samples)`` for line-of-sight targets.

    A pulsed burst records its ``gate`` (default: the whole PRI); FMCW records every sample.
    """
    paths = line_of_sight_paths(radar, geometry, rcs_m2)
    if radar.transmitter.waveform == "fmcw":
        return fmcw_deramp_baseband(paths, radar, n_pulses, rng=rng)
    gate = RangeGate() if gate is None else gate
    return pulsed_baseband(
        paths,
        radar,
        n_pulses,
        rng=rng,
        range_gate_start_m=gate.range_gate_start_m,
        n_gate_samples=gate.n_gate_samples,
    )


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
            synthesise(radar, truth, rcs_m2, n_pulses, rng=rng, gate=gate)
            for radar, n_pulses, gate in zip(
                scenario.radars, scenario.n_pulses, scenario.gates, strict=True
            )
        )
        yield Frame(index, float(scenario.frame_times_s[index]), truth, cubes)


def map_axes(
    radar: Radar, n_pulses: int, *, gate: RangeGate | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a burst's map axes: bistatic range of each range bin, and rate of each Doppler bin.

    FMCW range is the beat axis; a pulsed bin is one sample of delay from the range gate's
    first, :func:`~radar_forge.dsp.delay_bin_centers_m` (no gate: the whole PRI from zero).
    """
    transmitter, sample_rate_hz = radar.transmitter, radar.receiver.sample_rate_hz
    gate = RangeGate() if gate is None else gate
    if transmitter.waveform == "pulsed":
        range_axis_m = delay_bin_centers_m(
            gate.n_samples(radar), sample_rate_hz, range_gate_start_m=gate.range_gate_start_m
        )
    else:
        range_axis_m = range_bin_centers_m(
            radar.n_samples_per_pri,
            transmitter.bandwidth_hz,
            transmitter.chirp_duration_s,
            sample_rate_hz,
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
    A range-gated map is a window on its range circle, ``unambiguous_range_m`` round.
    """

    rd_map: NDArray[np.complex128]
    range_axis_m: NDArray[np.float64]
    rate_axis_mps: NDArray[np.float64]
    unambiguous_range_m: float | None = None

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
        """Circumference of the range circle, m: the axis's span unless the map is gated."""
        if self.unambiguous_range_m is not None:
            return self.unambiguous_range_m
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
        range_offset_m = np.asarray(bistatic_range_m) - self.range_axis_m[0]
        range_index = np.mod(range_offset_m, self.range_period_m) / self.range_bin_m
        offset_mps = np.asarray(rate_mps) - self.rate_axis_mps[0]
        return range_index, np.mod(offset_mps / self.rate_bin_mps, self.rate_axis_mps.size)


def range_doppler_product(
    cube: ArrayLike,
    radar: Radar,
    *,
    gate: RangeGate | None = None,
    migration: MigrationSettings | None = None,
) -> RangeDopplerProduct:
    """Form one burst's map: FMCW, two FFTs; pulsed, compression, migration, the Doppler FFT.

    A pulsed chain applies ``migration`` between the two, so detection and tracking see the
    compensated map. A gated map's range circle is the PRI's. No window on either axis.
    """
    samples = np.asarray(cube, dtype=np.complex128)
    if radar.transmitter.waveform == "pulsed":
        data_rfst = pulse_compress(samples, radar, gate=gate)
        if migration is not None:
            data_rfst = compensate_migration(
                data_rfst,
                radar,
                method=migration.method,
                keystone=migration.keystone,
                fold_factor_max=migration.fold_factor_max,
                accelerations_mps2=migration.accelerations_mps2,
            ).data_rfst
        rd_map = doppler_fft(data_rfst, axis=0)
    else:
        rd_map = range_doppler_map(samples)
    windowed = gate is not None and not gate.is_whole_pri(radar)
    circle_m = radar.unambiguous_bistatic_range_m if windowed else None
    return RangeDopplerProduct(rd_map, *map_axes(radar, samples.shape[0], gate=gate), circle_m)


def frame_products(scenario: Scenario, frame: Frame) -> list[RangeDopplerProduct]:
    """Return each burst's map of one frame, through its range gate and ``[migration]``."""
    return [
        range_doppler_product(cube, radar, gate=gate, migration=scenario.migration)
        for cube, radar, gate in zip(frame.cubes, scenario.radars, scenario.gates, strict=True)
    ]


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
