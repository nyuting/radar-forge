#!/usr/bin/env python3
"""Run a scenario end to end and write IQ, range-Doppler frames, and truth labels.

    uv run python scripts/run_scenario.py scenarios/scenario_001_fmcw_low_prf.toml --out out/s1

Thin by design: every decision about physics, processing and geometry lives in
``radar_forge.pipelines`` and ``radar_forge.core``, so this file only walks the
frame generator and writes files. Rendering needs the ``viz`` extra; pass
``--no-plots`` to run without it.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from radar_forge import __version__
from radar_forge.core.ambiguity import fold_velocity_mps
from radar_forge.core.radar import BistaticRadar
from radar_forge.pipelines.scenarios import (
    Frame,
    RangeDopplerProduct,
    Scenario,
    form_range_doppler_map,
    iterate_frames,
    load_scenario,
    peak_range_velocity,
)
from radar_forge.pipelines.tracking import (
    SENSOR_ID,
    STATE_FIELDS,
    FrameTracks,
    MetricRow,
    ScenarioTracker,
    score_primary_track,
)
from radar_forge.pipelines.trajectories import load_flight_csv

# A monostatic run writes the first six. A bistatic run appends the last three,
# so the bistatic column set is a superset of the monostatic one and the D5 COCO
# exporter can consume either unchanged -- see
# spec/scenario-002-bistatic.md §3.6. In a bistatic run `range_m` is the
# bistatic mean range and `radial_velocity_mps` the bisector rate.
TRUTH_COLUMNS = [
    "frame",
    "time_s",
    "range_m",
    "radial_velocity_mps",
    "azimuth_deg",
    "elevation_deg",
]
BISTATIC_TRUTH_COLUMNS = ["range_tx_m", "range_rx_m", "bistatic_angle_deg"]

# The data-001 files, each as one tuple of column names in the order
# spec/data-001-formats.md §6 lists them. Each row is built by name, so a
# column can only be written under its own header. When pipelines/io lands
# (data-001 §12), these tuples move into its schema registry unchanged.
#
# §6.5. `time_utc` is written only when the trajectory has an absolute epoch.
# The angle columns are optional and left out: this radar measures no angle.
# `associated_track_id` is empty for a detection no track claimed, which is how
# the plots tell a false alarm from a hit without rerunning the associator.
# `burst_index`, `status` and `pair_id` say what became of each detection on
# its way to the tracker, which matters for a dual-PRF pair.
DETECTION_COLUMNS = (
    "frame",
    "time_s",
    "time_utc",
    "detection_id",
    "sensor_id",
    "range_m",
    "velocity_folded_mps",
    "velocity_unfolded_mps",
    "fold_index",
    "range_index",
    "velocity_index",
    "peak_power_w",
    "total_power_w",
    "snr_db",
    "n_cells",
    "associated_track_id",
    "burst_index",
    "status",
    "pair_id",
)
# §6.6, for the range_1d state model: the state *is* `range_m` and
# `range_rate_mps`, and `cov_0_1` is their one off-diagonal covariance, in
# m²/s, indexed by `tracking.state_fields` in metadata.json. `measurement_dim`
# records the §5.3 bootstrap: the frame at which it steps from 1 to 2 is the
# frame the track became able to unfold its Doppler.
TRACK_COLUMNS = (
    "frame",
    "time_s",
    "time_utc",
    "track_id",
    "status",
    "state_model",
    "measurement_dim",
    "n_hits",
    "n_misses_in_a_row",
    "range_m",
    "range_rate_mps",
    "var_range_m2",
    "var_range_rate_m2ps2",
    "cov_0_1",
    "nis",
    "associated",
    "associated_detection_id",
)
# §6.9: tidy and long, one row per metric per scope.
METRIC_COLUMNS = ("metric", "track_id", "target_id", "frame_start", "frame_end", "value")

# Fixed strings restating data-001 §5, so a reader can check them rather than
# assume them (§6.1).
CONVENTIONS = {
    "frame": "enu",
    "azimuth_reference": "true_north_clockwise",
    "velocity_sign": "closing_positive",
    "cube_layout": "frame,pulse,sample[,rx]",
    "rd_layout": "frame,doppler,range",
}
SCHEMA_VERSION = "1.1.0"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line; ``None`` reads ``sys.argv``.

    Every switch is a ``--no-*`` opt-*out*, so the bare invocation produces
    everything the scenario can produce and a reader who omits a flag is never
    silently handed less than the run appears to promise.
    """
    parser = argparse.ArgumentParser(
        description="Run a radar-forge scenario and write its outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scenario", type=Path, help="Scenario TOML to run.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Run only this many frames, instead of the scenario's whole window.",
    )
    parser.add_argument("--no-iq", action="store_true", help="Skip writing the IQ cubes.")
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip rendering (avoids the viz extra)."
    )
    parser.add_argument("--no-movie", action="store_true", help="Skip assembling the movie.")
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Skip detection and tracking even if the TOML carries a [tracking] table.",
    )
    parser.add_argument(
        "--dynamic-range-db", type=float, default=60.0, help="Decibels shown below the peak."
    )
    return parser.parse_args(argv)


def git_commit() -> str | None:
    """The commit the run was made at, or None outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def burst_metadata(scenario: Scenario) -> list[dict[str, Any]]:
    """Everything about each burst a reader needs to interpret the cubes."""
    bursts: list[dict[str, Any]] = []
    for index, (burst, n_pulses) in enumerate(zip(scenario.bursts, scenario.n_pulses, strict=True)):
        bursts.append(
            {
                "index": index,
                "waveform": burst.transmitter.waveform,
                "f0_hz": burst.transmitter.f0_hz,
                "bandwidth_hz": burst.transmitter.bandwidth_hz,
                "chirp_duration_s": burst.transmitter.chirp_duration_s,
                "prf_hz": burst.transmitter.prf_hz,
                "sample_rate_hz": burst.receiver.sample_rate_hz,
                "n_pulses": n_pulses,
                "n_samples": burst.n_samples_per_pri,
                "range_resolution_m": burst.range_resolution_m,
                "unambiguous_range_m": burst.unambiguous_range_m,
                "unambiguous_velocity_mps": burst.unambiguous_velocity_mps,
                "noise_power_w": burst.noise_power_w,
            }
        )
    return bursts


def site_metadata(scenario: Scenario) -> dict[str, object]:
    """Describe where the radar stands, in whichever siting the scenario uses."""
    burst = scenario.bursts[0]
    if isinstance(burst, BistaticRadar):
        return {
            "siting": "bistatic",
            "transmitter_site": {
                "latitude_deg": burst.transmitter_latitude_deg,
                "longitude_deg": burst.transmitter_longitude_deg,
                "altitude_m": burst.transmitter_altitude_m,
            },
            "receiver_site": {
                "latitude_deg": burst.receiver_latitude_deg,
                "longitude_deg": burst.receiver_longitude_deg,
                "altitude_m": burst.receiver_altitude_m,
            },
            "baseline_m": burst.baseline_m,
        }
    return {
        "siting": "monostatic",
        "radar_site": {
            "latitude_deg": burst.latitude_deg,
            "longitude_deg": burst.longitude_deg,
            "altitude_m": burst.altitude_m,
        },
    }


def reference_site(scenario: Scenario) -> dict[str, object]:
    """The ENU origin of data-001 §5: the radar, or the receiver of a bistatic pair."""
    burst = scenario.bursts[0]
    if isinstance(burst, BistaticRadar):
        return {
            "latitude_deg": burst.receiver_latitude_deg,
            "longitude_deg": burst.receiver_longitude_deg,
            "altitude_m": burst.receiver_altitude_m,
            "role": "receiver",
        }
    return {
        "latitude_deg": burst.latitude_deg,
        "longitude_deg": burst.longitude_deg,
        "altitude_m": burst.altitude_m,
        "role": "monostatic",
    }


def tracking_metadata(scenario: Scenario, tracker: ScenarioTracker) -> dict[str, object]:
    """The resolved tracking settings, with what a reader of tracks.csv needs to know.

    The settings are the ones the run used, defaults included, under the
    scenario TOML's names, so every key of the ``[tracking]`` table appears
    with its value, as data-001 §6.1 asks. ``state_fields`` indexes the ``cov_<i>_<j>`` columns of
    tracks.csv (data-001 §6.1). ``range_period_m`` is the period of the
    exported range when range folds, and ``None`` when it does not, so that a
    reader knows once, here, whether ``range_m`` is absolute or modulo.
    """
    settings = asdict(tracker.tracking)
    # configs_from_scenario renames the TOML's velocity_unfolding to the field
    # unfolding_mode on the way in; this renames it back on the way out.
    settings["velocity_unfolding"] = settings.pop("unfolding_mode")
    return {
        **settings,
        # Recorded explicitly, so no stored run is ambiguous about whether its
        # angles were measured or synthesised (scenario 003 §6.3).
        "simulated_angles": bool((scenario.tracking_table or {}).get("simulated_angles", False)),
        "state_fields": list(STATE_FIELDS),
        "range_period_m": tracker.folding_layout.coordinates[0].period,
    }


def write_metadata(
    scenario: Scenario,
    out_dir: Path,
    n_frames: int,
    *,
    tracker: ScenarioTracker | None,
    created_utc: datetime,
    epoch_utc: datetime | None,
    command: Sequence[str],
) -> None:
    """Write ``metadata.json``: everything needed to interpret the run's files.

    The keys are data-001 §6.1's, except ``files``, the inventory of written
    files, which comes with the ``pipelines/io`` follow-up (§12). ``n_frames``
    is the count actually written, which ``--frames`` can make smaller than
    ``scenario.n_frames``. Recording the two separately is what lets a reader
    tell a truncated run from a short scenario.
    """
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"{scenario.name}-{created_utc:%Y%m%dT%H%M%SZ}",
        "created_utc": _iso_utc(created_utc),
        "epoch_utc": None if epoch_utc is None else _iso_utc(epoch_utc),
        "reference_site": reference_site(scenario),
        "conventions": CONVENTIONS,
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
            "seed": scenario.seed,
            "start_time_s": scenario.start_time_s,
            "duration_s": scenario.duration_s,
            "frame_rate_hz": scenario.frame_rate_hz,
            "n_frames": n_frames,
            "trajectory_path": str(scenario.trajectory_path),
            "target_altitude_m": scenario.target_altitude_m,
        },
        "sites": site_metadata(scenario),
        "target": asdict(scenario.target),
        "bursts": burst_metadata(scenario),
        "tracking": None if tracker is None else tracking_metadata(scenario, tracker),
        "detection": None if tracker is None else asdict(tracker.detection),
        "provenance": {
            "radar_forge_version": __version__,
            "git_commit": git_commit(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "command": list(command),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def trajectory_epoch(scenario: Scenario) -> datetime | None:
    """The absolute time of the trajectory's first fix, from which ``time_s`` counts."""
    epoch = load_flight_csv(scenario.trajectory_path, altitude_m=scenario.target_altitude_m).epoch
    if epoch is None:
        return None
    return epoch.replace(tzinfo=UTC) if epoch.tzinfo is None else epoch.astimezone(UTC)


def _iso_utc(instant: datetime) -> str:
    """ISO 8601 UTC with microseconds and a ``Z``, as data-001 §5 writes every instant."""
    return instant.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _cell(value: object) -> str:
    """Format one CSV cell as data-001 §5 asks.

    ``None`` is an empty cell (not applicable), a boolean is 0 or 1, and a
    float is written in the shortest form that reads back to the same value
    (Python's ``repr``), never at a fixed precision.
    """
    if value is None:
        return ""
    if isinstance(value, bool | np.bool_):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return repr(float(value))
    return str(value)


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    """Write rows under ``columns``, each row's cells looked up by column name."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row[column]) for column in columns])


def _columns(columns: Sequence[str], *, with_time_utc: bool) -> tuple[str, ...]:
    """Drop ``time_utc`` from a column list when the run has no absolute epoch."""
    return tuple(c for c in columns if c != "time_utc" or with_time_utc)


def detection_rows(
    frames: Iterable[FrameTracks], scenario: Scenario, epoch_utc: datetime | None
) -> Iterable[dict[str, object]]:
    """One detections.csv row per detection per frame (data-001 §6.5)."""
    for record in frames:
        claimed = {index: track_id for track_id, index in record.associations.items()}
        for detection_id, measurement in enumerate(record.measurements):
            # snr_db is against the burst's thermal noise power, not the CFAR
            # estimate of the local floor (data-001 §6.5).
            noise_power_w = scenario.bursts[measurement.burst_index].noise_power_w
            yield {
                "frame": record.frame_index,
                "time_s": record.time_s,
                "time_utc": _time_utc(epoch_utc, record.time_s),
                "detection_id": detection_id,
                "sensor_id": SENSOR_ID,
                "range_m": measurement.range_m,
                "velocity_folded_mps": measurement.velocity_folded_mps,
                "velocity_unfolded_mps": measurement.velocity_unfolded_mps,
                "fold_index": measurement.fold_index,
                "range_index": measurement.range_index,
                "velocity_index": measurement.velocity_index,
                "peak_power_w": measurement.peak_power_w,
                "total_power_w": measurement.total_power_w,
                "snr_db": float(10.0 * np.log10(measurement.peak_power_w / noise_power_w)),
                "n_cells": measurement.n_cells,
                "associated_track_id": claimed.get(detection_id),
                "burst_index": measurement.burst_index,
                "status": measurement.status,
                "pair_id": measurement.pair_id,
            }


def track_rows(
    frames: Iterable[FrameTracks], state_model: str, epoch_utc: datetime | None
) -> Iterable[dict[str, object]]:
    """One tracks.csv row per track per frame (data-001 §6.6)."""
    for record in frames:
        for track in record.tracks:
            yield {
                "frame": record.frame_index,
                "time_s": record.time_s,
                "time_utc": _time_utc(epoch_utc, record.time_s),
                "track_id": track.track_id,
                "status": track.status,
                "state_model": state_model,
                "measurement_dim": track.measurement_dim,
                "n_hits": track.n_hits,
                "n_misses_in_a_row": track.n_misses,
                "range_m": float(track.state[0]),
                "range_rate_mps": float(track.state[1]),
                "var_range_m2": float(track.covariance[0, 0]),
                "var_range_rate_m2ps2": float(track.covariance[1, 1]),
                "cov_0_1": float(track.covariance[0, 1]),
                "nis": track.nis,
                "associated": track.track_id in record.associations,
                "associated_detection_id": record.associations.get(track.track_id),
            }


def metric_rows(metrics: Iterable[MetricRow]) -> Iterable[dict[str, object]]:
    """One metrics.csv row per metric (data-001 §6.9)."""
    for row in metrics:
        yield {
            "metric": row.metric,
            "track_id": row.track_id,
            "target_id": row.target_id,
            "frame_start": row.frame_start,
            "frame_end": row.frame_end,
            "value": row.value,
        }


def _time_utc(epoch_utc: datetime | None, time_s: float) -> str | None:
    """``epoch_utc + time_s`` as data-001 §5 writes it, or None with no epoch."""
    return None if epoch_utc is None else _iso_utc(epoch_utc + timedelta(seconds=time_s))


def write_detections_csv(
    path: Path, rows: Iterable[Mapping[str, object]], *, with_time_utc: bool
) -> None:
    """Write detections.csv, columns in data-001 §6.5's order."""
    _write_csv(path, _columns(DETECTION_COLUMNS, with_time_utc=with_time_utc), rows)


def write_tracks_csv(
    path: Path, rows: Iterable[Mapping[str, object]], *, with_time_utc: bool
) -> None:
    """Write tracks.csv, columns in data-001 §6.6's order."""
    _write_csv(path, _columns(TRACK_COLUMNS, with_time_utc=with_time_utc), rows)


def write_metrics_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write metrics.csv, columns in data-001 §6.9's order."""
    _write_csv(path, METRIC_COLUMNS, rows)


def render_frame(
    frame: Frame,
    scenario: Scenario,
    products: Sequence[RangeDopplerProduct],
    out_dir: Path,
    dynamic_range_db: float,
    record: FrameTracks | None = None,
) -> None:
    """Draw one range-Doppler panel per burst and save it.

    Takes the already-processed ``products`` rather than processing the cubes
    itself, so the receive chain runs once per frame however many consumers a
    frame has.
    """
    from radar_forge.viz.plotting import save_figure
    from radar_forge.viz.scopes.rd_map import render_range_doppler

    for index, (product, burst) in enumerate(zip(products, scenario.bursts, strict=True)):
        suffix = "" if len(scenario.bursts) == 1 else f"_burst{index}"
        # Where this burst's ambiguities oblige the target to appear: Doppler
        # wraps into the unambiguous interval, range modulo the unambiguous
        # range. For a burst that folds in neither, both are the truth itself.
        folded_velocity_mps = float(
            fold_velocity_mps(frame.radial_velocity_mps, burst.unambiguous_velocity_mps)
        )
        folded_range_m = frame.range_m % burst.unambiguous_range_m
        title = (
            f"{scenario.name}{suffix}  t = {frame.time_s:.0f} s   "
            f"truth {frame.range_m / 1e3:.2f} km, {frame.radial_velocity_mps:+.1f} m/s"
        )
        # Detections and the track are drawn on the first burst's map only:
        # they are formed from it, or in the dual-PRF case from the pair, and
        # repeating them on burst B would imply they were measured there.
        detections = track_estimate = gate_extent = None
        if record is not None and index == 0:
            # ``velocity_folded_mps`` is already inside this burst's
            # unambiguous interval -- it was read off the map's own velocity
            # axis -- so it is what the marker goes on. The *unfolded* value
            # would land off-scale, or worse, somewhere plausible and wrong.
            detections = [
                (measurement.range_m, measurement.velocity_folded_mps)
                for measurement in record.measurements
            ]
            confirmed = [track for track in record.tracks if track.is_confirmed]
            if confirmed:
                track = max(confirmed, key=lambda candidate: candidate.n_hits)
                track_estimate = (
                    float(track.state[0]),
                    float(fold_velocity_mps(track.state[1], burst.unambiguous_velocity_mps)),
                )
                # Three standard deviations of the estimate in each component,
                # the extent of the track's uncertainty on the map.
                gate_extent = (
                    3.0 * float(np.sqrt(track.covariance[0, 0])),
                    3.0 * float(np.sqrt(track.covariance[1, 1])),
                )

        figure = render_range_doppler(
            product.rd_map,
            product.range_axis_m,
            product.velocity_axis_mps,
            truth_range_m=frame.range_m,
            truth_velocity_mps=frame.radial_velocity_mps,
            folded_range_m=folded_range_m,
            folded_velocity_mps=folded_velocity_mps,
            title=title,
            dynamic_range_db=dynamic_range_db,
            bistatic=scenario.is_bistatic,
            detections=detections,
            track_estimate=track_estimate,
            gate_extent=gate_extent,
        )
        save_figure(figure, out_dir / f"rd{suffix}_{frame.index:05d}.png")


def render_track_frame(
    out_dir: Path,
    tracker: ScenarioTracker,
    truth_time_s: Sequence[float],
    truth_range_m: Sequence[float],
    truth_range_rate_mps: Sequence[float],
) -> None:
    """Draw range and range rate against time, up to the most recent frame.

    Called once per frame so the series grow, which is what makes the assembled
    movie show a track being built rather than a finished plot appearing. When
    range folds, the truth is wrapped as the detections and the track are, so
    all three are compared modulo the same period.
    """
    from radar_forge.viz.plotting import save_figure
    from radar_forge.viz.scopes.track_plot import render_range_time_history

    frames = tracker.frames
    record = frames[-1]
    range_period_m = tracker.folding_layout.coordinates[0].period
    detection_time_s = [entry.time_s for entry in frames for _ in entry.measurements]
    detection_range_m = [
        measurement.range_m for entry in frames for measurement in entry.measurements
    ]

    # The longest-lived confirmed track, so the line does not jump between
    # rivals from frame to frame.
    counts: dict[int, int] = {}
    for entry in frames:
        for track in entry.tracks:
            if track.is_confirmed:
                counts[track.track_id] = counts.get(track.track_id, 0) + 1
    track_time_s: list[float] = []
    track_states: list[np.ndarray[Any, Any]] = []
    track_sigma_m: list[float] = []
    if counts:
        best_id = max(counts, key=lambda key: counts[key])
        for entry in frames:
            for track in entry.tracks:
                if track.track_id == best_id and track.is_confirmed:
                    track_time_s.append(entry.time_s)
                    track_states.append(track.state)
                    track_sigma_m.append(float(np.sqrt(track.covariance[0, 0])))

    truth_m = tracker.folding_layout.wrap(
        np.asarray(truth_range_m)[:, None], interval="nonnegative"
    )
    figure = render_range_time_history(
        truth_time_s,
        truth_m[:, 0],
        detection_time_s=detection_time_s,
        detection_range_m=detection_range_m,
        track_time_s=track_time_s or None,
        track_range_m=[float(state[0]) for state in track_states] or None,
        track_sigma_range_m=track_sigma_m or None,
        current_time_s=record.time_s,
        title=f"frame {record.frame_index}  t = {record.time_s:.1f} s",
        truth_range_rate_mps=truth_range_rate_mps,
        track_range_rate_mps=[float(state[1]) for state in track_states] or None,
        range_period_m=range_period_m,
    )
    save_figure(figure, out_dir / f"track_{record.frame_index:05d}.png")


def clear_previous_frames(out_dir: Path) -> int:
    """Delete the per-frame files an earlier run left in ``out_dir``.

    Returns the number removed.

    Without this a shorter re-run into the same directory leaves the tail of
    the previous run behind, and :func:`assemble_movie` globs whatever is
    present -- splicing stale frames onto the new ones with nothing to warn
    you. Only this script's own numbered outputs are touched; ``truth.csv``
    and ``metadata.json`` are overwritten in place, and anything else in the
    directory is left alone.
    """
    stale = sorted(out_dir.glob("rd*_[0-9][0-9][0-9][0-9][0-9].png"))
    stale += sorted(out_dir.glob("track_[0-9][0-9][0-9][0-9][0-9].png"))
    stale += sorted(out_dir.glob("iq_[0-9][0-9][0-9][0-9][0-9].npz"))
    for path in stale:
        path.unlink()
    return len(stale)


def assemble_movie(out_dir: Path, pattern: str, stem: str, frame_rate_hz: float) -> str | None:
    """Assemble the PNG frames into an MP4, falling back to an animated GIF."""
    # Match the five digits the pattern formats, so a single-burst run's
    # ``rd_%05d.png`` does not also sweep up a previous multi-burst run's
    # ``rd_burst0_00000.png``.
    frames = sorted(out_dir.glob(pattern.replace("%05d", "[0-9][0-9][0-9][0-9][0-9]")))
    if not frames:
        return None

    if shutil.which("ffmpeg") is not None:
        target = out_dir / f"{stem}.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(frame_rate_hz),
            "-i",
            str(out_dir / pattern),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(target),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return str(target)
        print(f"  ffmpeg failed ({result.returncode}); falling back to GIF.")

    try:
        from PIL import Image
    except ImportError:
        print("  neither ffmpeg nor Pillow is available; skipping the movie.")
        return None

    target = out_dir / f"{stem}.gif"
    # A generator, not a list: Pillow consumes append_images one at a time, so
    # only the frame being appended is decoded. Materialising them all is what
    # makes the fallback path -- the one taken by whoever lacks ffmpeg -- run
    # out of memory on a long scenario.
    # Image.Palette.ADAPTIVE, not the bare Image.ADAPTIVE: the latter is a
    # legacy alias that still works at runtime but is absent from Pillow's type
    # stubs, so it only fails the type gate once the viz extra is present.
    first = Image.open(frames[0]).convert("P", palette=Image.Palette.ADAPTIVE)
    rest = (Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frames[1:])
    first.save(
        target,
        save_all=True,
        append_images=rest,
        duration=int(1000.0 / frame_rate_hz),
        loop=0,
    )
    return str(target)


def main(argv: Sequence[str] | None = None) -> int:
    created_utc = datetime.now(UTC)
    command = sys.argv if argv is None else [sys.argv[0], *argv]
    args = parse_args(argv)
    scenario = load_scenario(args.scenario)
    epoch_utc = trajectory_epoch(scenario)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frames = scenario.n_frames if args.frames is None else min(args.frames, scenario.n_frames)
    print(f"{scenario.name}: {n_frames} frames, {len(scenario.bursts)} burst(s) -> {out_dir}")

    removed = clear_previous_frames(out_dir)
    if removed:
        print(f"  removed {removed} frame file(s) from an earlier run")

    tracking_enabled = scenario.tracking_table is not None and not args.no_tracking
    tracker: ScenarioTracker | None = None
    truth_time_s: list[float] = []
    truth_range_m: list[float] = []
    truth_range_rate_mps: list[float] = []
    if tracking_enabled:
        tracker = ScenarioTracker.from_scenario(scenario)
        bootstrap = (
            ""
            if tracker.min_unfold_frames is None
            else f", bootstrap {tracker.min_unfold_frames} frames"
        )
        print(
            f"  tracking: {tracker.tracking.estimator}, {tracker.tracking.state_model}, "
            f"unfolding {tracker.tracking.unfolding_mode}{bootstrap}"
        )
        if (scenario.tracking_table or {}).get("simulated_angles", False):
            # §6.3: the angles are a crutch and must be labelled as one
            # everywhere they appear, so that no stored run is ambiguous about
            # whether its angles were real.
            print(
                "  WARNING: simulated_angles is on -- the azimuth and elevation "
                "measurements are synthesised from truth, not measured. This "
                "radar has no array."
            )

    truth_path = out_dir / "truth.csv"
    written = 0
    with truth_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = list(TRUTH_COLUMNS)
        if scenario.is_bistatic:
            columns += BISTATIC_TRUTH_COLUMNS
        writer.writerow(columns)

        for frame in iterate_frames(scenario):
            if frame.index >= n_frames:
                break
            # Truth is the true, unfolded geometry in every variant; the gap
            # between it and the map is what the scenario is for.
            row = [
                frame.index,
                f"{frame.time_s:.3f}",
                f"{frame.range_m:.3f}",
                f"{frame.radial_velocity_mps:.6f}",
                f"{frame.azimuth_deg:.6f}",
                f"{frame.elevation_deg:.6f}",
            ]
            if (
                frame.range_tx_m is not None
                and frame.range_rx_m is not None
                and frame.bistatic_angle_deg is not None
            ):
                row += [
                    f"{frame.range_tx_m:.3f}",
                    f"{frame.range_rx_m:.3f}",
                    f"{frame.bistatic_angle_deg:.6f}",
                ]
            writer.writerow(row)

            if not args.no_iq:
                arrays = {f"burst{index}": cube for index, cube in enumerate(frame.iq)}
                # One named array per burst. The ignore is a stub limitation:
                # savez_compressed takes **kwds of arrays, but the stub's
                # `allow_pickle` keyword makes mypy read the unpacked dict as
                # a candidate for it.
                np.savez_compressed(
                    out_dir / f"iq_{frame.index:05d}.npz",
                    **arrays,  # type: ignore[arg-type]
                )

            written += 1
            show_progress = written % 10 == 0 or written == n_frames
            # The receive chain is the expensive part of a frame, and both the
            # render and the progress line want the same maps. Run it once,
            # and only when something is actually going to read the result --
            # a --no-plots run processes nothing but its progress frames.
            products: list[RangeDopplerProduct] = []
            # Tracking needs the maps every frame, not only the ones that are
            # rendered or reported on.
            if not args.no_plots or show_progress or tracking_enabled:
                products = [
                    form_range_doppler_map(cube, burst)
                    for cube, burst in zip(frame.iq, scenario.bursts, strict=True)
                ]

            record: FrameTracks | None = None
            if tracker is not None:
                record = tracker.step(
                    products,
                    frame_index=frame.index,
                    time_s=frame.time_s,
                    truth_velocity_mps=frame.radial_velocity_mps,
                )
                truth_time_s.append(frame.time_s)
                truth_range_m.append(frame.range_m)
                truth_range_rate_mps.append(frame.radial_velocity_mps)

            if not args.no_plots:
                render_frame(frame, scenario, products, out_dir, args.dynamic_range_db, record)
                if tracker is not None:
                    render_track_frame(
                        out_dir, tracker, truth_time_s, truth_range_m, truth_range_rate_mps
                    )
            if show_progress:
                peaks = [peak_range_velocity(product) for product in products]
                summary = "  ".join(f"[{r / 1e3:6.2f} km {v:+7.2f} m/s]" for r, v in peaks)
                print(f"  frame {written}/{n_frames}  peak {summary}")

    write_metadata(
        scenario,
        out_dir,
        written,
        tracker=tracker,
        created_utc=created_utc,
        epoch_utc=epoch_utc,
        command=command,
    )
    print(f"  wrote {truth_path}")

    if tracker is not None and tracker.frames:
        with_time_utc = epoch_utc is not None
        write_detections_csv(
            out_dir / "detections.csv",
            detection_rows(tracker.frames, scenario, epoch_utc),
            with_time_utc=with_time_utc,
        )
        write_tracks_csv(
            out_dir / "tracks.csv",
            track_rows(tracker.frames, tracker.tracking.state_model, epoch_utc),
            with_time_utc=with_time_utc,
        )
        metrics = score_primary_track(
            tracker.frames,
            truth_range_m,
            truth_range_rate_mps,
            folding_layout=tracker.folding_layout,
        )
        write_metrics_csv(out_dir / "metrics.csv", metric_rows(metrics))
        for name in ("detections.csv", "tracks.csv", "metrics.csv"):
            print(f"  wrote {out_dir / name}")
        values = {row.metric: row.value for row in metrics}
        print(
            f"  tracked {values['n_tracked_frames']:.0f}/{len(tracker.frames)} frames "
            f"under {values['n_confirmed_tracks']:.0f} confirmed track id(s)"
        )
        if "n_confirmed_frames" in values:
            print(
                f"  primary track: confirmed in {values['n_confirmed_frames']:.0f} frames, "
                f"range RMSE {values['range_rmse_m']:.1f} m, "
                f"range-rate RMSE {values['range_rate_rmse_mps']:.2f} m/s"
            )

    if not args.no_plots and not args.no_movie and tracker is not None and tracker.frames:
        movie = assemble_movie(out_dir, "track_%05d.png", "track", scenario.frame_rate_hz)
        if movie is not None:
            print(f"  wrote {movie}")

    if not args.no_plots and not args.no_movie:
        for index in range(len(scenario.bursts)):
            suffix = "" if len(scenario.bursts) == 1 else f"_burst{index}"
            movie = assemble_movie(
                out_dir, f"rd{suffix}_%05d.png", f"rd{suffix}", scenario.frame_rate_hz
            )
            if movie is not None:
                print(f"  wrote {movie}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
