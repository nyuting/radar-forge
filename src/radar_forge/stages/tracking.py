r"""The ``multitarget`` stage: many targets through the plot model and a configured tracker.

A plot-level scenario (``[sensor]``) is played rotation by rotation (:func:`~radar_forge.
pipeline.scans`) into :class:`~radar_forge.pipeline.PlotTracker`. Each rotation's confirmed
tracks are scored against the aircraft in coverage by GOSPA [1], each truth taken at its
track's own time; over the run the stage measures swaps at every crossing the arcs hold, the
IMM's latency to its turn mode after each turn onset, the position NEES, the measured Pd and
false-plot rate against the plot model's, and the JPDA cluster sizes, then trajectory GOSPA
[2]. An IQ scenario (scenario 011's V5) gives the plot model's rates and merging from
``ctx.plots`` instead. Writes ``plots.csv``, ``tracks.csv``, ``gospa.csv`` and ``scene.png``.

References
----------
.. [1] A. S. Rahmathullah, Á. F. García-Fernández and L. Svensson, "Generalized optimal
       sub-pattern assignment metric," *Proc. 20th Int. Conf. Information Fusion*, 2017.
.. [2] Á. F. García-Fernández, A. S. Rahmathullah and L. Svensson, "A metric on the space of
       finite sets of trajectories for evaluation of multi-target tracking algorithms," *IEEE
       Trans. Signal Process.*, vol. 68, pp. 3917-3928, 2020.
.. [3] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from radar_forge import evaluate, plots
from radar_forge.pipeline import PlotScan, PlotTracker, scans
from radar_forge.scenario import Scenario
from radar_forge.tracking.filters import TrackBank, TrackStatus
from radar_forge.tracking.metrics import gospa, nees, trajectory_gospa

__all__ = ["merged_counts", "multitarget", "pair_windows"]

# GOSPA's settings (spec 011 §3.6): p = 2, cutoff c = 300 m, alpha = 2.
_CUTOFF_M = 300.0
# A crossing is a pair of aircraft that come within c/2 of each other; a swap is judged from the
# scan before it to this many after (spec 011 §3.6).
_SWAP_SCANS = 5
# A turn shorter than this many scans ends before an IMM can be asked to have seen it.
_MIN_TURN_SCANS = 4
# The IMM's coordinated-turn mode, second of quiet, turn and agile.
_TURN_MODE = 1
# A rate is checked only on enough samples for its interval to say something: about 30
# expected events, and 100 echoes for a Pd.
_MIN_FALSE_PLOTS, _MIN_ECHOES = 30.0, 100
# An IQ scene's targets closer than this many range bins are one pair (V5 places its pairs
# at most 6 bins apart, and the pairs 20 bins or more apart).
_PAIR_BINS = 8.0

_CONFIRMED = (TrackStatus.CONFIRMED, TrackStatus.COASTING)


@dataclass
class _Run:
    """What the rotations accumulate for the scores."""

    gospa_rows: list[dict[str, Any]] = field(default_factory=list)
    assigned: list[NDArray[np.int64]] = field(default_factory=list)
    nees: list[float] = field(default_factory=list)
    turn_probability: list[NDArray[np.float64]] = field(default_factory=list)
    time_s: list[NDArray[np.float64]] = field(default_factory=list)
    track_paths: dict[int, list[NDArray[np.float64]]] = field(default_factory=dict)
    confirmed: list[tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]] = field(
        default_factory=list
    )
    present: list[NDArray[np.intp]] = field(default_factory=list)
    cluster_tracks: list[int] = field(default_factory=list)
    cluster_events: list[int] = field(default_factory=list)
    echoes: list[tuple[NDArray[np.bool_], NDArray[np.bool_]]] = field(default_factory=list)
    false_plots: int = 0
    expected_false: float = 0.0


def _truth(scenario: Scenario, time_s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return every target's position at each time, ``(n_targets, *time.shape, 2)``."""
    return np.stack([target.arc.state(time_s)[0] for target in scenario.arcs])


def _score_rotation(scenario: Scenario, plots_scan: PlotScan, bank: TrackBank, run: _Run) -> None:
    """Score one rotation's confirmed tracks: GOSPA, its assignment, NEES and mode probability."""
    aircraft = np.array([t.target_class == "aircraft" for t in scenario.arcs])
    present = np.flatnonzero(aircraft & plots_scan.echo_in_coverage)
    confirmed = np.flatnonzero(np.isin(bank.status, _CONFIRMED))
    names = bank.layout.names
    columns = [names.index("x_m"), names.index("y_m")]
    estimate = bank.mean[confirmed][:, columns]
    truth = _truth(scenario, bank.time_s[confirmed])[present]
    score = gospa(estimate, truth.reshape(len(present), len(confirmed), 2), cutoff_m=_CUTOFF_M)
    run.gospa_rows.append(
        {
            "scan": plots_scan.index,
            "time_s": plots_scan.start_time_s,
            "gospa_m": score.distance_m,
            "localisation_m": score.localisation_m,
            "missed_m": score.missed_m,
            "false_m": score.false_m,
            "n_truths": len(present),
            "n_tracks": len(confirmed),
        }
    )
    assigned = np.full(len(scenario.arcs), -1, dtype=np.int64)
    turn = np.full(len(scenario.arcs), np.nan)
    times = np.full(len(scenario.arcs), np.nan)
    for k, row in zip(present, score.assignment, strict=True):
        # A few aircraft a rotation; each scored against the one track GOSPA gave it.
        if row < 0:
            continue
        track = confirmed[row]
        assigned[k] = bank.track_id[track]
        times[k] = bank.time_s[track]
        error = truth[list(present).index(k), row] - estimate[row]
        block = bank.covariance[track][np.ix_(columns, columns)]
        run.nees.append(float(nees(np.zeros(2), block, error)))
        if bank.n_modes:
            turn[k] = bank.mode_probability[track, _TURN_MODE]
    run.assigned.append(assigned)
    run.turn_probability.append(turn)
    run.time_s.append(times)
    for row in confirmed:
        run.track_paths.setdefault(int(bank.track_id[row]), []).append(bank.mean[row, columns])
    run.confirmed.append((bank.track_id[confirmed], estimate, bank.time_s[confirmed]))
    run.present.append(present)


def _swap_fraction(scenario: Scenario, run: _Run, period_s: float) -> float:
    """Return the fraction of crossings after which either aircraft changed its GOSPA track."""
    n_scans = len(run.assigned)
    if n_scans < 2:
        return math.nan
    grid_s = np.arange(0.0, n_scans * period_s, 0.1) + scenario.start_time_s
    path = _truth(scenario, grid_s)
    swaps = []
    aircraft = [k for k, t in enumerate(scenario.arcs) if t.target_class == "aircraft"]
    # Every pair of aircraft, a few dozen: each pair's closest approach is one batched norm.
    for i, a in enumerate(aircraft):
        for b in aircraft[i + 1 :]:
            gap_m = np.linalg.norm(path[a] - path[b], axis=-1)
            if not np.nanmin(gap_m, initial=np.inf) < _CUTOFF_M / 2:
                continue
            scan = int((grid_s[np.nanargmin(gap_m)] - scenario.start_time_s) // period_s)
            before, after = scan - 1, scan + _SWAP_SCANS
            if before < 0 or after >= n_scans:
                continue
            first, last = run.assigned[before][[a, b]], run.assigned[after][[a, b]]
            swaps.append(bool(np.any(first != last)))
    return float(np.mean(swaps)) if swaps else math.nan


def _mode_latency(scenario: Scenario, run: _Run, period_s: float) -> float:
    """Return the median scans the IMM took to put its turn mode above 0.5 after a turn's onset.

    Only turns of at least four scans that an aircraft began with a confirmed track count; the
    count starts at the first scan after the onset, and a turn the mode never reaches, or whose
    track is lost, is inf. The median, because one run holds a handful of turns.
    """
    probability, times = np.array(run.turn_probability), np.array(run.time_s)
    if not probability.size or np.all(np.isnan(probability)):
        return math.nan
    worst = []
    for k, target in enumerate(scenario.arcs):
        durations = [leg.duration_s for leg in target.arc.legs]
        starts = target.arc.start_time_s + np.cumsum([0.0, *durations])
        tracked = np.flatnonzero(np.isfinite(times[:, k]))
        # Each aircraft's legs: a handful, each checked once.
        for leg, onset_s in zip(target.arc.legs, starts, strict=False):
            if leg.turn_rate_radps == 0 or leg.duration_s < _MIN_TURN_SCANS * period_s:
                continue
            before = tracked[times[tracked, k] < onset_s]
            if not before.size or before[-1] + 1 >= len(times) or onset_s > np.nanmax(times):
                continue
            # The scans after the onset while the turn lasts, tracked or not.
            span = np.arange(before[-1] + 1, len(times))
            during = span[(span - before[-1]) * period_s <= leg.duration_s + period_s]
            above = during[probability[during, k] > 0.5]
            latency = float(above[0] - before[-1]) if above.size else math.inf
            print(
                f"  turn of {target.name} at {onset_s:.1f} s: turn mode over 0.5 in {latency} scans"
            )
            worst.append(latency)
    return float(np.median(worst)) if worst else math.nan


def _plot_rows(plots_scan: PlotScan) -> list[dict[str, Any]]:
    """Return one plots.csv row per plot of a rotation."""
    return [
        {
            "scan": plots_scan.index,
            "time_s": plots_scan.time_s[j],
            "plot_id": j,
            "range_m": plots_scan.range_m[j],
            "azimuth_deg": math.degrees(plots_scan.azimuth_rad[j]),
            "source": plots_scan.source[j],
            "source_ids": plots_scan.source_ids[j],
        }
        for j in range(plots_scan.range_m.size)
    ]


def _track_rows(plots_scan: PlotScan, bank: TrackBank) -> list[dict[str, Any]]:
    """Return one tracks.csv row per live track at the end of a rotation."""
    rows = []
    names = bank.layout.names
    # One row per track: tens to a few thousand a rotation.
    for r in range(bank.n_tracks):
        row: dict[str, Any] = {"scan": plots_scan.index, "time_s": bank.time_s[r]}
        row["track_id"], row["status"] = bank.track_id[r], TrackStatus(bank.status[r]).name.lower()
        row.update({name: bank.mean[r, k] for k, name in enumerate(names)})
        row["existence_probability"] = bank.existence[r]
        for k, mode in enumerate(("quiet", "turn", "agile")[: bank.n_modes]):
            row[f"mode_probability_{mode}"] = bank.mode_probability[r, k]
        rows.append(row)
    return rows


def _expected_false_plots(scenario: Scenario, time_s: float) -> float:
    """Return the false plots a rotation should hold: Poisson regions and discretes."""
    clutter = scenario.clutter_plots
    regions = sum(c.intensity_per_m_rad * c.volume_m_rad(time_s) for c in clutter.components)
    discretes = clutter.discretes
    return regions + (
        0.0 if discretes is None else discretes.report_probability * discretes.range_m.size
    )


@evaluate.stage(
    "multitarget",
    "gospa_mean_max_m",
    "swap_fraction_max",
    "mode_latency_max_scans",
    "nees_fraction_min",
    "pd_min",
    "false_alarm_rate_max",
)
def multitarget(ctx: evaluate.Context) -> evaluate.Context:
    """Track a plot-level scene and score it, or measure an IQ chain's plot statistics.

    ``gospa_mean_max_m`` bounds the mean GOSPA over the rotations; ``swap_fraction_max`` the
    fraction of crossings followed by a swap; ``mode_latency_max_scans`` the median scans an
    IMM takes to its turn mode after the onset of a turn it tracks; ``nees_fraction_min`` the
    fraction of position NEES inside chi-square(2)'s 95 % interval; ``pd_min`` the measured
    Pd; ``false_alarm_rate_max`` ``false_plot_ratio``, false plots over the model's count.
    """
    if ctx.scenario.sensor is None:
        return _iq_statistics(ctx)
    scenario, verify = ctx.scenario, evaluate.thresholds(ctx, "multitarget")
    sensor, tracker, run = ctx.scenario.sensor, PlotTracker(scenario), _Run()
    plot_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    false_enu: list[NDArray[np.float64]] = []
    true_enu: list[NDArray[np.float64]] = []
    # Rotations are sequential: each one's tracks are the next one's prior.
    for plots_scan in scans(scenario, n_frames=ctx.n_frames):
        for _, association in tracker.rotation(plots_scan):
            if association is not None:
                run.cluster_tracks += association.cluster_tracks.tolist()
                run.cluster_events += association.cluster_events.tolist()
        bank = tracker.tracker.bank
        _score_rotation(scenario, plots_scan, bank, run)
        aircraft = np.array([t.target_class == "aircraft" for t in scenario.arcs])
        kept = aircraft & plots_scan.echo_in_coverage & ~plots_scan.echo_in_notch
        run.echoes.append((kept, plots_scan.echo_detected))
        false = np.isin(plots_scan.source, ("clutter", "discrete"))
        run.false_plots += int(false.sum())
        run.expected_false += _expected_false_plots(scenario, plots_scan.start_time_s)
        enu = plots_scan.range_m[:, None] * np.column_stack(
            [np.sin(plots_scan.azimuth_rad), np.cos(plots_scan.azimuth_rad)]
        )
        false_enu.append(enu[false])
        true_enu.append(enu[~false])
        plot_rows += _plot_rows(plots_scan)
        track_rows += _track_rows(plots_scan, bank)
    _record(ctx, verify, run, scenario, sensor.scan_period_s)
    evaluate.write_csv(ctx.out_dir / "plots.csv", plot_rows)
    evaluate.write_csv(ctx.out_dir / "tracks.csv", track_rows)
    evaluate.write_csv(ctx.out_dir / "gospa.csv", run.gospa_rows)
    if ctx.render:
        times_s = scenario.start_time_s + np.arange(0.0, ctx.n_frames * sensor.scan_period_s, 1.0)
        truth = _truth(scenario, times_s)
        figure = plots.render_scene(
            [(t.name, truth[k]) for k, t in enumerate(scenario.arcs)],
            plot_enu_m=np.concatenate([*false_enu, *true_enu]),
            plot_is_false=np.concatenate(
                [
                    np.ones(sum(len(f) for f in false_enu), bool),
                    np.zeros(sum(len(t) for t in true_enu), bool),
                ]
            ),
            tracks=[(str(k), np.array(path)) for k, path in run.track_paths.items()],
            title=scenario.name,
        )
        plots.save_figure(figure, ctx.out_dir / "scene.png")
    return ctx


def _record(
    ctx: evaluate.Context, verify: dict[str, Any], run: _Run, scenario: Scenario, period_s: float
) -> None:
    """Record and print every score, and check those the run is long enough to judge."""
    distances = np.array([row["gospa_m"] for row in run.gospa_rows])
    for part in ("gospa_m", "localisation_m", "missed_m", "false_m"):
        ctx.record(
            f"{part.removesuffix('_m')}_mean_m", float(np.mean([r[part] for r in run.gospa_rows]))
        )
    swap = ctx.record("swap_fraction", _swap_fraction(scenario, run, period_s))
    latency = ctx.record("mode_latency_median_scans", _mode_latency(scenario, run, period_s))
    low, high = chi2.ppf([0.025, 0.975], 2)
    inside = [low <= value <= high for value in run.nees]
    nees_fraction = ctx.record("nees_fraction", float(np.mean(inside)) if inside else math.nan)
    kept = np.concatenate([k for k, _ in run.echoes])
    detected = np.concatenate([d for _, d in run.echoes])
    pd = ctx.record("measured_pd", float(np.mean(detected[kept])) if kept.any() else math.nan)
    ctx.record("n_echoes", float(kept.sum()))
    ratio = ctx.record("false_plot_ratio", run.false_plots / max(run.expected_false, 1e-300))
    ctx.record("n_false_plots", float(run.false_plots))
    ctx.record("cluster_tracks_max", float(max(run.cluster_tracks, default=0)))
    ctx.record("cluster_events_max", float(max(run.cluster_events, default=0)))
    if run.cluster_tracks:
        sizes, counts = np.unique(run.cluster_tracks, return_counts=True)
        print(
            "  cluster sizes (tracks: clusters): "
            + ", ".join(f"{s}: {c}" for s, c in zip(sizes, counts, strict=True))
        )
    paths = _trajectory_score(scenario, run)
    if paths is not None:
        ctx.record("trajectory_gospa_m", paths.distance_m)
        ctx.record("n_switches", paths.n_switches)
    long_enough = (
        len(distances) >= 2 * scenario.tracking.n_confirm_scans if scenario.tracking else False
    )
    checks = [
        (
            "gospa_mean_max_m",
            "gospa_mean_m",
            float(np.mean(distances)),
            0.0,
            verify.get("gospa_mean_max_m"),
            long_enough,
        ),
        (
            "swap_fraction_max",
            "swap_fraction",
            swap,
            0.0,
            verify.get("swap_fraction_max"),
            long_enough,
        ),
        (
            "mode_latency_max_scans",
            "mode_latency_median_scans",
            latency,
            0.0,
            verify.get("mode_latency_max_scans"),
            long_enough,
        ),
        (
            "nees_fraction_min",
            "nees_fraction",
            nees_fraction,
            verify.get("nees_fraction_min"),
            1.0,
            long_enough,
        ),
        ("pd_min", "measured_pd", pd, verify.get("pd_min"), 1.0, kept.sum() >= _MIN_ECHOES),
        (
            "false_alarm_rate_max",
            "false_plot_ratio",
            ratio,
            0.0,
            verify.get("false_alarm_rate_max"),
            run.expected_false >= _MIN_FALSE_PLOTS,
        ),
    ]
    for key, name, value, low_bound, high_bound, judged in checks:
        if key in verify and judged:
            ctx.check("multitarget", name, value, low_bound, high_bound)
        elif key in verify:
            print(f"  {name}: too few scans or samples in this run to check")


def _trajectory_score(scenario: Scenario, run: _Run) -> Any:
    """Return trajectory GOSPA over the run's confirmed tracks, or None without any.

    Each confirmed track is one trajectory, present in the rotations it was confirmed; each
    aircraft present while in coverage, taken at each track's time.
    """
    ids = sorted({int(i) for track_id, _, _ in run.confirmed for i in track_id})
    if not ids:
        return None
    column = {track_id: k for k, track_id in enumerate(ids)}
    n_steps, n_targets = len(run.confirmed), len(scenario.arcs)
    estimates = np.full((n_steps, len(ids), 2), np.nan)
    times = np.full((n_steps, len(ids)), np.nan)
    truths = np.full((n_steps, n_targets, len(ids), 2), np.nan)
    # One rotation at a time: its tracks and the aircraft in coverage.
    for step, ((track_id, position, time_s), present) in enumerate(
        zip(run.confirmed, run.present, strict=True)
    ):
        where = [column[int(i)] for i in track_id]
        estimates[step, where], times[step, where] = position, time_s
        at = np.where(
            np.isnan(times[step]), np.nanmean(time_s) if time_s.size else 0.0, times[step]
        )
        truths[step, present] = _truth(scenario, at)[present]
    aircraft = [k for k, t in enumerate(scenario.arcs) if t.target_class == "aircraft"]
    return trajectory_gospa(estimates, truths[:, aircraft], cutoff_m=_CUTOFF_M)


def pair_windows(
    ctx: evaluate.Context, frame: int, bins: float
) -> list[tuple[list[int], float, float]]:
    """Return the IQ scene's targets grouped into pairs closer than ``bins`` range bins.

    Each group is its targets and, at ``frame``, the bistatic range window one bin either
    side of them in which its detections are counted; groups are fixed by frame 0.
    """
    truth = evaluate.truth(ctx)
    first, now = truth[0].bistatic_range_m, truth[frame].bistatic_range_m
    _, _, bin_m = evaluate.spans(ctx.scenario, 0)
    order = np.argsort(first)
    groups: list[list[int]] = []
    # Targets in range order: a handful, each joining the group before it or starting one.
    for k in order:
        if groups and first[k] - first[groups[-1][-1]] < bins * bin_m:
            groups[-1].append(int(k))
        else:
            groups.append([int(k)])
    return [(g, float(now[g].min() - bin_m), float(now[g].max() + bin_m)) for g in groups]


def merged_counts(ctx: evaluate.Context) -> tuple[NDArray[np.float64], NDArray[np.int64], int]:
    """Return each pair's bistatic separation, its frames with one detection, and the frames.

    A pair's detections are burst 0's clusters inside its window, whatever their pairing
    status: one is a merged group, two or more are resolved.
    """
    truth = evaluate.truth(ctx)
    pairs = [g for g, _, _ in pair_windows(ctx, 0, _PAIR_BINS) if len(g) == 2]
    separation = np.array([abs(np.diff(truth[0].bistatic_range_m[g])[0]) for g in pairs])
    merged = np.zeros(len(pairs), dtype=np.int64)
    # One frame at a time: its windows follow the pairs as they move.
    for frame, found in enumerate(ctx.plots):
        ranges = found.bistatic_range_m[found.burst_index == 0]
        windows = [(lo, hi) for g, lo, hi in pair_windows(ctx, frame, _PAIR_BINS) if len(g) == 2]
        counts = [np.count_nonzero((ranges >= lo) & (ranges <= hi)) for lo, hi in windows]
        merged += np.array(counts, dtype=np.int64) == 1
    return separation, merged, len(ctx.plots)


def _iq_statistics(ctx: evaluate.Context) -> evaluate.Context:
    """Measure an IQ chain's plot statistics: Pd, false-alarm rate, merged fraction per pair.

    A target is detected in a frame when a burst-0 cluster lies in its pair's window (alone or
    merged with its partner). False alarms are clusters of either burst outside every window
    widened by the CFAR's training and guard cells, counted against the design Pfa times the
    cells left; the ratio is measured over design.
    """
    scenario, verify = ctx.scenario, evaluate.thresholds(ctx, "multitarget")
    if len(ctx.plots) != ctx.n_frames:
        msg = "the multitarget stage on an IQ scenario needs the rd stage's plots; run rd first."
        raise ValueError(msg)
    settings = scenario.detection
    reach = settings.n_train + settings.n_guard
    detected, false_count, cells = [], 0, 0.0
    for frame, found in enumerate(ctx.plots):
        windows = pair_windows(ctx, frame, _PAIR_BINS)
        for burst, (radar, n_pulses) in enumerate(
            zip(scenario.radars, scenario.n_pulses, strict=True)
        ):
            _, _, bin_m = evaluate.spans(scenario, burst)
            ranges = found.bistatic_range_m[found.burst_index == burst]
            near = np.zeros(ranges.size, dtype=bool)
            excluded_bins = 0
            for group, low_m, high_m in windows:
                inside = (ranges >= low_m) & (ranges <= high_m)
                if burst == 0:
                    detected += [bool(inside.any())] * len(group)
                near |= (ranges >= low_m - reach * bin_m) & (ranges <= high_m + reach * bin_m)
                excluded_bins += round((high_m - low_m) / bin_m) + 2 * reach
            false_count += int(np.count_nonzero(~near))
            cells += n_pulses * (radar.n_samples_per_pri - excluded_bins)
    pd = ctx.record("measured_pd", float(np.mean(detected)) if detected else math.nan)
    ctx.record("n_echoes", float(len(detected)))
    expected = settings.pfa * cells
    ratio = ctx.record("false_plot_ratio", false_count / expected)
    ctx.record("n_false_alarms", float(false_count))
    separation, merged, n_frames = merged_counts(ctx)
    for gap_m, count in zip(separation, merged, strict=True):
        ctx.record(f"merged_fraction_{gap_m:.0f}m", count / n_frames)
    if "pd_min" in verify and len(detected) >= _MIN_ECHOES:
        ctx.check("multitarget", "measured_pd", pd, verify["pd_min"], 1.0)
    elif "pd_min" in verify:
        print("  measured_pd: too few echoes in this run to check")
    if "false_alarm_rate_max" in verify and expected >= _MIN_FALSE_PLOTS:
        ctx.check("multitarget", "false_plot_ratio", ratio, 0.0, verify["false_alarm_rate_max"])
    elif "false_alarm_rate_max" in verify:
        print("  false_plot_ratio: too few expected false alarms in this run to check")
    return ctx
