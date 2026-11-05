r"""The stepper: a scenario run as named stages, each printing, checking and writing one step.

``geometry``, ``iq``, ``rd``, ``detect`` and ``track`` run in that order. Each prints its
intermediate quantities and wall time, checks the thresholds the scenario's ``[verify]`` table
gives, and writes its files under the run directory (``spec/data-001-formats.md``). ``[verify]``
holds ``stages`` and thresholds: ``truth_in_map_fraction = [low, high]``, ``expected_snr_db`` with
``snr_tolerance_db``, ``noise_tolerance_db``, ``peak_tolerance_db``, ``truth_within_bins`` with
``detection_fraction``, ``pfa_tolerance_factor``, ``rate_tolerance_mps``, ``range_rmse_max_m``,
``range_rate_rmse_max_mps``, ``nees_fraction_min`` and ``confirmation_latency_max_scans``. A
stage not listed does not run, a threshold not given is not checked, and :func:`run` raises
:class:`VerificationError` naming every failure. CSVs show a monostatic radar's range and rate
halved (:func:`~radar_forge.plots.display_axes`); everything else is bistatic.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014.
.. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from radar_forge import plots
from radar_forge.detection import cfar_detect
from radar_forge.pipeline import (
    Plots,
    RangeDopplerProduct,
    ScanTracker,
    frame_plots,
    frames,
    map_axes,
    range_doppler_product,
    synthesise,
)
from radar_forge.propagation import received_power_w
from radar_forge.radar import Radar, TargetGeometry
from radar_forge.scenario import Scenario, truth_geometry
from radar_forge.signal import thermal_noise
from radar_forge.tracking.filters import TrackBank, TrackStatus
from radar_forge.tracking.metrics import nees

__all__ = ["STAGES", "Check", "Context", "Report", "VerificationError", "run"]

# The false alarms the rd stage's noise-only maps are expected to hold: a Poisson count whose
# 99.9 % interval, about 14 to 50, lies within a factor of two of it.
_FALSE_ALARMS = 30.0
# Their cells are capped, which only a design Pfa below about 1e-5 reaches: fewer expected.
_MAX_CELLS = 4_000_000


@dataclass(frozen=True)
class Check:
    """One verified quantity: what was measured, the closed interval it needs, and the verdict."""

    stage: str
    name: str
    measured: float
    low: float
    high: float

    @property
    def passed(self) -> bool:
        """Whether the measured value lies in ``[low, high]``; nan never does."""
        return bool(self.low <= self.measured <= self.high)


@dataclass
class Report:
    """Every check of a run, every scalar a stage measured, and each stage's wall time."""

    scenario: str
    checks: list[Check] = field(default_factory=list)
    values: dict[str, float] = field(default_factory=dict)
    wall_time_s: dict[str, float] = field(default_factory=dict)

    @property
    def failures(self) -> list[Check]:
        """The checks that failed."""
        return [check for check in self.checks if not check.passed]


class VerificationError(Exception):
    """Raised by :func:`run` when any check fails; the message lists every failure."""

    def __init__(self, report: Report) -> None:
        """Keep the report, and name each failed check and the interval it needed."""
        self.report = report
        lines = [
            f"{c.stage}: {c.name} = {c.measured:.6g}, not in [{c.low}, {c.high}]"
            for c in report.failures
        ]
        super().__init__(f"{report.scenario}: {len(lines)} check(s) failed\n" + "\n".join(lines))


@dataclass
class Context:
    """What the stages share: the scenario, the run's settings, and each stage's results."""

    scenario: Scenario
    out_dir: Path
    n_frames: int
    save_iq: bool
    report: Report
    render: bool = True
    truth: list[TargetGeometry] = field(default_factory=list)
    plots: list[Plots] = field(default_factory=list)
    banks: list[tuple[TrackBank, NDArray[np.intp]]] = field(default_factory=list)

    def record(self, name: str, value: float) -> float:
        """Print a measured scalar, keep it in the report, and return it."""
        self.report.values[name] = float(value)
        print(f"  {name} = {value:.6g}")
        return float(value)

    def check(self, stage: str, name: str, value: float, low: Any, high: Any) -> None:
        """Record a check that ``value`` lies in ``[low, high]``."""
        result = Check(stage, name, float(value), float(low), float(high))
        self.report.checks.append(result)
        verdict = "pass" if result.passed else "FAIL"
        print(f"  check {name} = {value:.6g} in [{low}, {high}]: {verdict}")


Stage = Callable[[Context], Context]
STAGES: dict[str, Stage] = {}
"""The registered stages, by name, in the order :func:`run` runs them."""


def _stage(name: str) -> Callable[[Stage], Stage]:
    """Register a stage under ``name``."""

    def register(function: Stage) -> Stage:
        STAGES[name] = function
        return function

    return register


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dict rows under the first row's keys; a nan is written as an empty cell."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["frame"])
        writer.writeheader()
        writer.writerows(
            {k: "" if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()}
            for row in rows
        )


def _db(value: ArrayLike) -> float:
    return float(10.0 * np.log10(value))


def _wrapped(value: ArrayLike, period: float) -> NDArray[np.float64]:
    """Return ``value`` folded into ``[-period/2, period/2)``: the short way round."""
    result: NDArray[np.float64] = (np.asarray(value) + period / 2.0) % period - period / 2.0
    return result


def _power_w(radar: Radar, truth: TargetGeometry, rcs_m2: ArrayLike) -> NDArray[np.float64]:
    """Return each target's received power by the radar equation, W."""
    return received_power_w(
        transmit_power_w=radar.transmitter.transmit_power_w,
        gain_tx_linear=10.0 ** (radar.transmitter.gain_tx_dbi / 10.0),
        gain_rx_linear=10.0 ** (radar.receiver.gain_rx_dbi / 10.0),
        wavelength_m=radar.wavelength_m,
        rcs_m2=rcs_m2,
        range_tx_m=truth.range_tx_m,
        range_rx_m=truth.range_rx_m,
    )


def _spans(scenario: Scenario, burst: int) -> tuple[float, float, float]:
    """Return a burst's range span, rate span and range bin, m and m/s."""
    range_axis_m, rate_axis_mps = map_axes(scenario.radars[burst], scenario.n_pulses[burst])
    range_bin_m = float(range_axis_m[1] - range_axis_m[0])
    rate_span_mps = float(rate_axis_mps.size * (rate_axis_mps[0] - rate_axis_mps[1]))
    return range_axis_m.size * range_bin_m, rate_span_mps, range_bin_m


def _truth(ctx: Context) -> list[TargetGeometry]:
    """Return every frame's truth, from the geometry stage or computed now."""
    if not ctx.truth:
        ctx.truth = truth_geometry(ctx.scenario)[: ctx.n_frames]
    return ctx.truth


@_stage("geometry")
def _geometry(ctx: Context) -> Context:
    """Truth for every frame; link budget, baseline and unambiguous limits; truth.csv."""
    scenario, radar, verify, truth = (
        ctx.scenario,
        ctx.scenario.radar,
        ctx.scenario.verify,
        _truth(ctx),
    )
    rcs_m2 = [target.rcs_m2 for target in scenario.targets]
    power_w = float(_power_w(radar, truth[0], rcs_m2)[0])
    # Integrated over one chirp's samples and every pulse: P T N / N0, on an exact bin.
    gain = radar.n_samples_per_chirp * scenario.n_pulses[0]
    snr_db = _db(power_w / radar.receiver.noise_power_per_sample_w * gain)
    beta_deg = np.degrees([frame.bistatic_angle_rad for frame in truth])
    for name, value in (
        ("received_power_dbw", _db(power_w)),
        ("noise_power_per_sample_dbw", _db(radar.receiver.noise_power_per_sample_w)),
        ("predicted_snr_db", snr_db),
        ("baseline_m", radar.baseline_m),
        ("bistatic_angle_min_deg", beta_deg.min()),
        ("bistatic_angle_max_deg", beta_deg.max()),
        ("unambiguous_bistatic_range_m", radar.unambiguous_bistatic_range_m),
        ("unambiguous_bistatic_range_rate_mps", radar.unambiguous_bistatic_range_rate_mps),
    ):
        ctx.record(name, value)
    # Inside the map unfolded: one burst's spans, or the larger span and the dual-PRF bound.
    spans = [_spans(scenario, k) for k in range(len(scenario.radars))]
    half_rate_mps = (
        scenario.detection.max_bistatic_range_rate_mps if len(spans) == 2 else spans[0][1] / 2
    )
    max_range_m = max(span[0] for span in spans)
    inside = [
        np.all(
            (t.bistatic_range_m < max_range_m) & (np.abs(t.bistatic_range_rate_mps) < half_rate_mps)
        )
        for t in truth
    ]
    fraction = ctx.record("truth_in_map_fraction", float(np.mean(inside)))
    if "truth_in_map_fraction" in verify:
        ctx.check("geometry", "truth_in_map_fraction", fraction, *verify["truth_in_map_fraction"])
    if "expected_snr_db" in verify:
        expected, tolerance = verify["expected_snr_db"], verify.get("snr_tolerance_db", 0.1)
        ctx.check(
            "geometry", "predicted_snr_db", snr_db, expected - tolerance, expected + tolerance
        )
    show, rows = plots.display_axes(radar), []
    # One row per target per frame: a few hundred.
    for index, frame in enumerate(truth):
        for k, target in enumerate(scenario.targets):
            row = {"frame": index, "time_s": scenario.frame_times_s[index], "target": target.name}
            row[show.range_name] = frame.bistatic_range_m[k] * show.scale
            row[show.rate_name] = frame.bistatic_range_rate_mps[k] * show.scale
            row["azimuth_deg"], row["elevation_deg"] = np.degrees(frame.aoa_rad[k])
            if not radar.is_monostatic:
                row["range_tx_m"], row["range_rx_m"] = frame.range_tx_m[k], frame.range_rx_m[k]
                row["bistatic_angle_deg"] = math.degrees(frame.bistatic_angle_rad[k])
            rows.append(row)
    _write_csv(ctx.out_dir / "truth.csv", rows)
    return ctx


@_stage("iq")
def _iq(ctx: Context) -> Context:
    """Noise per sample of a target-free cube, and frame 0's noiseless peak, every burst."""
    scenario, verify, first = ctx.scenario, ctx.scenario.verify, _truth(ctx)[0]
    rcs_m2 = np.array([target.rcs_m2 for target in scenario.targets])
    noise_errors_db, peak_errors_db = [], []
    for k, (radar, n_pulses) in enumerate(zip(scenario.radars, scenario.n_pulses, strict=True)):
        # A generator of its own, so the frames' noise does not depend on whether this ran.
        rng = np.random.default_rng([scenario.seed, 1, k])
        noise_db = _db(
            np.mean(np.abs(synthesise(radar, first, 0.0 * rcs_m2, n_pulses, rng=rng)) ** 2)
        )
        peak_db = _db(np.max(np.abs(synthesise(radar, first, rcs_m2, n_pulses, rng=None)) ** 2))
        noise_errors_db.append(abs(noise_db - _db(radar.receiver.noise_power_per_sample_w)))
        # The strongest target's power: exact for one target, a lower bound for several.
        peak_errors_db.append(abs(peak_db - _db(_power_w(radar, first, rcs_m2).max())))
        if k == 0:
            ctx.record("iq_noise_power_dbw", noise_db)
            ctx.record("iq_peak_power_dbw", peak_db)
    if "noise_tolerance_db" in verify:
        ctx.check("iq", "noise_error_db", max(noise_errors_db), 0.0, verify["noise_tolerance_db"])
    if "peak_tolerance_db" in verify:
        ctx.check("iq", "peak_error_db", max(peak_errors_db), 0.0, verify["peak_tolerance_db"])
    if ctx.save_iq:
        # One file per frame: the cubes stream through and none is held.
        for frame in frames(scenario, n_frames=ctx.n_frames):
            cubes = {f"burst{k}": cube for k, cube in enumerate(frame.cubes)}
            np.savez_compressed(ctx.out_dir / f"iq_{frame.index:05d}.npz", **cubes)  # type: ignore[arg-type]
    return ctx


def _hits(
    found: Plots, burst: int, product: RangeDopplerProduct, truth: TargetGeometry, bins: float
) -> list[bool]:
    """Return, per target, whether one of ``burst``'s detections lies within ``bins`` of it."""
    range_index, rate_index = product.index_of(
        truth.bistatic_range_m, truth.bistatic_range_rate_mps
    )
    cells = np.array(
        [
            d.centroid_index
            for d, b in zip(found.detections, found.burst_index, strict=True)
            if b == burst
        ]
    )
    n_rate, n_range = product.rd_map.shape
    cells = cells.reshape(-1, 2)
    near_rate = np.abs(_wrapped(cells[None, :, 0] - rate_index[:, None], n_rate)) <= bins
    near_range = np.abs(_wrapped(cells[None, :, 1] - range_index[:, None], n_range)) <= bins
    return list(np.any(near_rate & near_range, axis=1))


@_stage("rd")
def _rd(ctx: Context) -> Context:
    """Map, CFAR and cluster every frame; detections near truth; Pfa on noise; rd PNGs."""
    scenario, verify = ctx.scenario, ctx.scenario.verify
    bins, hits = float(verify.get("truth_within_bins", 1.0)), []
    for frame in frames(scenario, n_frames=ctx.n_frames):
        products = [
            range_doppler_product(c, r) for c, r in zip(frame.cubes, scenario.radars, strict=True)
        ]
        found = frame_plots(products, scenario.detection)
        ctx.plots.append(found)
        for k, product in enumerate(products):
            hits += _hits(found, k, product, frame.truth, bins)
            if ctx.render:
                mine = found.burst_index == k
                figure = plots.render_range_doppler(
                    product.rd_map,
                    product.range_axis_m,
                    product.rate_axis_mps,
                    scenario.radars[k],
                    detection_range_m=found.bistatic_range_m[mine],
                    detection_rate_mps=found.folded_rate_mps[mine],
                    truth_range_m=frame.truth.bistatic_range_m,
                    truth_rate_mps=frame.truth.bistatic_range_rate_mps,
                    title=f"{scenario.name}, frame {frame.index}, burst {k}",
                )
                suffix = "" if len(products) == 1 else f"_burst{k}"
                plots.save_figure(figure, ctx.out_dir / f"rd{suffix}_{frame.index:05d}.png")
        if frame.index == 0:
            power_w = np.abs(products[0].rd_map) ** 2
            rate_bin, range_bin = np.unravel_index(int(np.argmax(power_w)), power_w.shape)
            ctx.record("rd_peak_range_index", float(range_bin))
            ctx.record("rd_peak_rate_index", float(rate_bin))
            ctx.record("rd_peak_power_dbw", _db(power_w.max()))
            ctx.record("rd_noise_floor_dbw", _db(np.median(power_w)))
    ctx.record("n_detections", sum(len(p.status) for p in ctx.plots))
    fraction = ctx.record("detection_fraction", float(np.mean(hits)))
    if "detection_fraction" in verify:
        ctx.check("rd", "detection_fraction", fraction, verify["detection_fraction"], 1.0)
    if "pfa_tolerance_factor" in verify:
        # The burst furthest from its design rate, by ratio either way; zero crossings is worst.
        ratios = [_false_alarm_ratio(scenario, k) for k in range(len(scenario.radars))]
        ratio = ctx.record(
            "false_alarm_ratio", max(ratios, key=lambda r: abs(math.log(r + 1e-300)))
        )
        factor = verify["pfa_tolerance_factor"]
        ctx.check("rd", "false_alarm_ratio", ratio, 1.0 / factor, factor)
    return ctx


def _false_alarm_ratio(scenario: Scenario, burst: int) -> float:
    """Return measured over design Pfa on one burst's noise-only maps (Richards [1] §6.5)."""
    settings, radar, n_pulses = scenario.detection, scenario.radars[burst], scenario.n_pulses[burst]
    n_cells = n_pulses * radar.n_samples_per_pri
    n_maps = max(1, min(math.ceil(_FALSE_ALARMS / (settings.pfa * n_cells)), _MAX_CELLS // n_cells))
    rng = np.random.default_rng([scenario.seed, 2, burst])
    shape = (n_maps, n_pulses, radar.n_samples_per_pri)
    noise = thermal_noise(shape, radar.receiver.noise_power_per_sample_w, rng=rng)
    # One map at a time: the receive chain takes one cube.
    power_w = np.stack([np.abs(range_doppler_product(cube, radar).rd_map) ** 2 for cube in noise])
    detected, _ = cfar_detect(
        power_w,
        settings.pfa,
        n_train=settings.n_train,
        n_guard=settings.n_guard,
        variant=settings.variant,
        wrap=True,
    )
    return float(np.mean(detected) / settings.pfa)


@_stage("detect")
def _detect(ctx: Context) -> Context:
    """Rate of the plot nearest each target against its truth; plot counts; detections.csv."""
    scenario, verify, truth = ctx.scenario, ctx.scenario.verify, _truth(ctx)
    if len(ctx.plots) != ctx.n_frames:
        msg = "the detect stage needs the rd stage's plots; run rd first."
        raise ValueError(msg)
    range_span_m, rate_span_mps, range_bin_m = _spans(scenario, 0)
    within_m = float(verify.get("truth_within_bins", 1.0)) * range_bin_m
    errors_mps, show, rows = [], plots.display_axes(scenario.radar), []
    # A frame's plots against its targets: a handful of each.
    for index, (found, frame) in enumerate(zip(ctx.plots, truth, strict=True)):
        accepted = found.accepted
        for k, rate_mps in enumerate(frame.bistatic_range_rate_mps):
            gap_m = np.abs(
                _wrapped(found.bistatic_range_m[accepted] - frame.bistatic_range_m[k], range_span_m)
            )
            if np.any(gap_m <= within_m):
                plot = accepted[np.argmin(gap_m)]
                unfolded_mps = found.bistatic_range_rate_mps[plot]
                # Unfolded, against the truth; read off the map, against the truth folded too.
                folded_gap_mps = _wrapped(found.folded_rate_mps[plot] - rate_mps, rate_span_mps)
                errors_mps.append(
                    abs(unfolded_mps - rate_mps if unfolded_mps == unfolded_mps else folded_gap_mps)
                )
        for j, detection in enumerate(found.detections):
            row = {"frame": index, "time_s": scenario.frame_times_s[index], "detection_id": j}
            row["burst_index"], row["status"] = found.burst_index[j], found.status[j]
            row[show.range_name] = found.bistatic_range_m[j] * show.scale
            row[show.rate_name + "_folded"] = found.folded_rate_mps[j] * show.scale
            row[show.rate_name] = found.bistatic_range_rate_mps[j] * show.scale
            row["rate_index"], row["range_index"] = detection.centroid_index
            row["sigma_range_m"] = found.sigma_range_m[j] * show.scale
            row["sigma_rate_mps"] = found.sigma_rate_mps[j] * show.scale
            row["peak_power_w"], row["n_cells"] = detection.peak_power_w, detection.n_cells
            row["pair_id"] = found.pair_id[j] if found.pair_id[j] >= 0 else ""
            row["unfolded_" + show.range_name] = found.unfolded_range_m[j] * show.scale
            rows.append(row)
    ctx.record("n_accepted", sum(p.accepted.size for p in ctx.plots))
    error_mps = ctx.record("rate_error_max_mps", max(errors_mps, default=math.nan))
    if "rate_tolerance_mps" in verify:
        ctx.check("detect", "rate_error_max_mps", error_mps, 0.0, verify["rate_tolerance_mps"])
    _write_csv(ctx.out_dir / "detections.csv", rows)
    return ctx


@_stage("track")
def _track(ctx: Context) -> Context:
    """Track every frame; score each target's primary track; tracks, metrics and track.png."""
    scenario, verify, settings = ctx.scenario, ctx.scenario.verify, ctx.scenario.tracking
    if len(ctx.plots) != ctx.n_frames or settings is None:
        msg = "the track stage needs a [tracking] table and the rd stage's plots."
        raise ValueError(msg)
    tracker = ScanTracker(scenario)
    period_m, show, rows = tracker.range_period_m, plots.display_axes(scenario.radar), []
    # Scans are sequential: each one's tracks are the next one's prior.
    for index, found in enumerate(ctx.plots):
        bank, accepted = tracker.scan(found, float(scenario.frame_times_s[index]))
        ctx.banks.append((bank, accepted))
        for r in range(bank.n_tracks):
            row = {
                "frame": index,
                "time_s": scenario.frame_times_s[index],
                "track_id": bank.track_id[r],
            }
            row["status"] = TrackStatus(bank.status[r]).name.lower()
            row[show.range_name] = bank.mean[r, 0] % period_m * show.scale
            row[show.rate_name] = bank.mean[r, 1] * show.scale
            row["var_range_m2"], row["var_rate_m2ps2"] = (
                np.diagonal(bank.covariance[r]) * show.scale**2
            )
            row["cov_0_1"] = bank.covariance[r, 0, 1] * show.scale**2
            row["nis"], row["n_hits"], row["n_misses_in_a_row"] = (
                bank.nis[r],
                bank.n_hits[r],
                bank.n_misses[r],
            )
            measured = bank.measurement_index[r]
            row["detection_id"] = accepted[measured] if measured >= 0 else ""
            rows.append(row)
    _write_csv(ctx.out_dir / "tracks.csv", rows)
    scores, metric_rows, tracks = _score(ctx, period_m)
    for name, value in scores.items():
        ctx.record(name, value)
    # The file shows errors as tracks.csv shows the state: halved and renamed if monostatic.
    shown = {
        "range_rmse_m": show.range_name.removesuffix("_m") + "_rmse_m",
        "range_rate_rmse_mps": show.rate_name.removesuffix("_mps") + "_rmse_mps",
    }
    for row in metric_rows:
        if row["metric"] in shown:
            row["metric"], row["value"] = shown[row["metric"]], row["value"] * show.scale
    _write_csv(ctx.out_dir / "metrics.csv", metric_rows)
    # A run shorter than two confirmation windows scores too few frames to judge a tracker.
    if ctx.n_frames < 2 * settings.n_confirm_scans:
        print(f"  under {2 * settings.n_confirm_scans} frames: too short to check a tracker")
    else:
        for key, name, low, high in (
            ("range_rmse_max_m", "range_rmse_m", 0.0, verify.get("range_rmse_max_m")),
            (
                "range_rate_rmse_max_mps",
                "range_rate_rmse_mps",
                0.0,
                verify.get("range_rate_rmse_max_mps"),
            ),
            ("nees_fraction_min", "nees_fraction", verify.get("nees_fraction_min"), 1.0),
            (
                "confirmation_latency_max_scans",
                "confirmation_latency_scans",
                0.0,
                verify.get("confirmation_latency_max_scans"),
            ),
        ):
            if key in verify:
                ctx.check("track", name, scores[name], low, high)
    if ctx.render:
        truth = _truth(ctx)
        accepted_m = [found.bistatic_range_m[found.accepted] for found in ctx.plots]
        # broadcast-exempt: one frame time per accepted plot, for a scatter.
        plot_time_s = np.repeat(
            scenario.frame_times_s[: ctx.n_frames], [r.size for r in accepted_m]
        )
        figure = plots.render_range_time_history(
            scenario.frame_times_s[: ctx.n_frames],
            np.array([frame.bistatic_range_m for frame in truth]),
            np.array([frame.bistatic_range_rate_mps for frame in truth]),
            scenario.radar,
            tracks=tracks,
            detection_time_s=plot_time_s,
            detection_range_m=np.concatenate(accepted_m),
            range_period_m=period_m
            if max(np.max(f.bistatic_range_m) for f in truth) >= period_m
            else None,
            title=scenario.name,
        )
        plots.save_figure(figure, ctx.out_dir / "track.png")
    return ctx


def _score(
    ctx: Context, period_m: float
) -> tuple[dict[str, float], list[dict[str, Any]], list[Any]]:
    """Score each target's primary track, the one nearest it in the most confirmed frames.

    Errors are truth minus estimate, range the short way round the map's span; the NEES
    fraction counts frames inside chi-square(2)'s two-sided 95 % interval (Bar-Shalom [2]
    §5.4.2); latency is scans from a track's first row to its confirmation. Over targets the
    worst of each is reported. Returns the scores, the metrics.csv rows and the tracks drawn.
    """
    truth = _truth(ctx)
    confirmed = [
        np.isin(b.status, (TrackStatus.CONFIRMED, TrackStatus.COASTING)) for b, _ in ctx.banks
    ]
    votes: dict[tuple[int, int], int] = {}
    # Each frame's confirmed tracks vote for the target nearest in range: a few rows a frame.
    for (bank, _), mask, frame in zip(ctx.banks, confirmed, truth, strict=True):
        for row in np.flatnonzero(mask):
            gap_m = np.abs(_wrapped(frame.bistatic_range_m - bank.mean[row, 0], period_m))
            key = (int(np.argmin(gap_m)), int(bank.track_id[row]))
            votes[key] = votes.get(key, 0) + 1
    low, high = chi2.ppf([0.025, 0.975], 2)
    worst = dict.fromkeys(
        ("range_rmse_m", "range_rate_rmse_mps", "nees_fraction", "confirmation_latency_scans"),
        math.nan,
    )
    metric_rows, tracks = [], []
    for target, name in enumerate(target.name for target in ctx.scenario.targets):
        mine = {track_id: n for (k, track_id), n in votes.items() if k == target}
        if not mine:
            continue
        track_id = max(mine, key=mine.__getitem__)
        rows = [np.flatnonzero(bank.track_id == track_id) for bank, _ in ctx.banks]
        scored = [i for i, r in enumerate(rows) if r.size and confirmed[i][r[0]]]
        mean = np.array([ctx.banks[i][0].mean[rows[i][0]] for i in scored])
        covariance = np.array([ctx.banks[i][0].covariance[rows[i][0]] for i in scored])
        range_error_m = _wrapped(
            [truth[i].bistatic_range_m[target] for i in scored] - mean[:, 0], period_m
        )
        rate_error_mps = [truth[i].bistatic_range_rate_mps[target] for i in scored] - mean[:, 1]
        error = np.column_stack([range_error_m, rate_error_mps])
        inside = (low <= (scaled := nees(np.zeros(2), covariance, error))) & (scaled <= high)
        born = next(i for i, r in enumerate(rows) if r.size)
        scores = {
            "range_rmse_m": float(np.sqrt(np.mean(range_error_m**2))),
            "range_rate_rmse_mps": float(np.sqrt(np.mean(np.square(rate_error_mps)))),
            "nees_fraction": float(np.mean(inside)),
            "confirmation_latency_scans": float(scored[0] - born),
        }
        for metric, value in scores.items():
            metric_rows.append(
                {
                    "metric": metric,
                    "target": name,
                    "track_id": track_id,
                    "frame_start": scored[0],
                    "frame_end": scored[-1],
                    "value": value,
                }
            )
            pick = min if metric == "nees_fraction" else max
            worst[metric] = (
                value if math.isnan(worst[metric]) else float(pick(worst[metric], value))
            )
        tracks.append(
            (f"track {track_id}", ctx.scenario.frame_times_s[scored], mean[:, 0], mean[:, 1])
        )
    worst["n_confirmed_tracks"] = float(len({track_id for _, track_id in votes}))
    return worst, metric_rows, tracks


def run(
    scenario: Scenario,
    stages: Sequence[str] | None = None,
    *,
    n_frames: int | None = None,
    out_dir: Path | str,
    save_iq: bool = False,
    render: bool = True,
) -> Report:
    """Run the named stages in order (default ``[verify].stages``, else all) and check them.

    Raises
    ------
    VerificationError
        If any check fails, after every stage has run.
    """
    names = list(scenario.verify.get("stages", STAGES) if stages is None else stages)
    if set(names) - set(STAGES):
        msg = f"unknown stage(s) {sorted(set(names) - set(STAGES))}; known: {list(STAGES)}."
        raise ValueError(msg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = scenario.n_frames if n_frames is None else min(n_frames, scenario.n_frames)
    ctx = Context(scenario, out, count, save_iq, Report(scenario.name), render)
    print(f"{scenario.name}: {count} frame(s), {len(scenario.radars)} burst(s) -> {out}")
    for name, stage in STAGES.items():
        if name in names:
            print(f"[{name}]")
            start_s = time.perf_counter()
            ctx = stage(ctx)
            ctx.report.wall_time_s[name] = time.perf_counter() - start_s
            print(f"  wall time {ctx.report.wall_time_s[name]:.2f} s")
    if ctx.report.failures:
        raise VerificationError(ctx.report)
    return ctx.report
