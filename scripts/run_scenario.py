#!/usr/bin/env python3
"""Run a scenario end to end and write IQ, range-Doppler frames, and truth labels.

    uv run python scripts/run_scenario.py scenarios/scenario_001_fmcw_low_prf.toml --out out/s1

Thin by design: every decision about physics, processing and geometry lives in
``radar_forge.pipelines`` and ``radar_forge.core``, so this file only walks the
frame generator and writes files. Rendering needs the ``teaching`` extra; pass
``--no-plots`` to run without it.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
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
    FrameTracks,
    ScenarioTracker,
    configs_from_scenario,
    dual_prf_measurements,
)

# A monostatic run writes the first six. A bistatic run appends the last three,
# so the bistatic column set is a superset of the monostatic one and the D5 COCO
# exporter can consume either unchanged -- see
# spec/scenario-002-bistatic.md S8. In a bistatic run `range_m` is the
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

# Scenario 003 §9. `associated_track_id` is empty for a measurement no track
# claimed, which is how the plots tell a false alarm from a hit without
# rerunning the associator.
DETECTION_COLUMNS = [
    "frame",
    "time_s",
    "range_m",
    "velocity_folded_mps",
    "velocity_unfolded_mps",
    "fold_index",
    "peak_power_w",
    "n_cells",
    "associated_track_id",
]
# `measurement_dim` records the §5.3 bootstrap: the frame at which it steps
# from 1 to 2 is the frame the track became able to unfold its Doppler, and
# `velocity_unfolded_mps` in detections.csv is empty until then.
TRACK_COLUMNS = [
    "frame",
    "time_s",
    "track_id",
    "status",
    "measurement_dim",
    "range_m",
    "range_rate_mps",
    "var_range_m2",
    "var_range_rate_m2ps2",
    "nis",
    "associated",
]


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
        "--no-plots", action="store_true", help="Skip rendering (avoids the teaching extra)."
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


def write_metadata(scenario: Scenario, out_dir: Path, n_frames: int) -> None:
    """Write ``metadata.json``: everything needed to interpret the run's files.

    ``n_frames`` is the count actually written, which ``--frames`` can make
    smaller than ``scenario.n_frames``. Recording the two separately is what
    lets a reader tell a truncated run from a short scenario.
    """
    metadata = {
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
        "tracking": (
            None
            if scenario.tracking_table is None
            else {
                **scenario.tracking_table,
                # Recorded explicitly, so no stored run is ambiguous about
                # whether its angles were measured or synthesised (§6.3).
                "simulated_angles": bool(scenario.tracking_table.get("simulated_angles", False)),
            }
        ),
        "detection": scenario.detection_table,
        "provenance": {"radar_forge_version": __version__, "git_commit": git_commit()},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


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
    from radar_forge.teaching.plotting import save_figure
    from radar_forge.teaching.scopes.rd_map import render_range_doppler

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
                    float(track.estimate.state[0]),
                    float(
                        fold_velocity_mps(track.estimate.state[1], burst.unambiguous_velocity_mps)
                    ),
                )
                # One standard deviation of the innovation in each component, so
                # the ellipse is the gate the associator actually applied.
                gate_extent = (
                    3.0 * float(np.sqrt(track.estimate.covariance[0, 0])),
                    3.0 * float(np.sqrt(track.estimate.covariance[1, 1])),
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


def build_tracker(scenario: Scenario) -> tuple[ScenarioTracker, bool]:
    """Build the tracker a scenario's [tracking] table asks for.

    Returns the tracker and whether the scenario uses the simulated angle
    measurement of §6.3, which the caller warns about. A scenario with two
    bursts is tracked through the dual-PRF path, where the waveform resolves
    the Doppler ambiguity and no unfolding is needed.
    """
    detection, tracking = configs_from_scenario(scenario)
    simulated_angles = bool((scenario.tracking_table or {}).get("simulated_angles", False))
    tracker = ScenarioTracker(
        fold_span_mps=2.0 * scenario.bursts[0].unambiguous_velocity_mps,
        frame_time_s=1.0 / scenario.frame_rate_hz,
        detection=detection,
        tracking=tracking,
    )
    return tracker, simulated_angles


def track_frame(
    tracker: ScenarioTracker,
    scenario: Scenario,
    frame: Frame,
    products: list[RangeDopplerProduct],
) -> FrameTracks:
    """Advance the tracker one frame, by whichever path the waveform allows."""
    if len(products) == 2:
        spans_mps = [2.0 * burst.unambiguous_velocity_mps for burst in scenario.bursts]
        measurements = dual_prf_measurements(products, spans_mps, config=tracker.detection)
        return tracker.step_unfolded(measurements, frame_index=frame.index, time_s=frame.time_s)
    return tracker.step(
        products[0],
        frame_index=frame.index,
        time_s=frame.time_s,
        truth_velocity_mps=frame.radial_velocity_mps,
    )


def write_tracking_csvs(out_dir: Path, frames: Sequence[FrameTracks]) -> tuple[Path, Path]:
    """Write detections.csv and tracks.csv, per scenario 003 §9."""
    detections_path = out_dir / "detections.csv"
    with detections_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(DETECTION_COLUMNS)
        for record in frames:
            claimed = {index: track_id for track_id, index in record.associations.items()}
            for index, measurement in enumerate(record.measurements):
                writer.writerow(
                    [
                        record.frame_index,
                        f"{record.time_s:.3f}",
                        f"{measurement.range_m:.3f}",
                        f"{measurement.velocity_folded_mps:.6f}",
                        ""
                        if measurement.velocity_unfolded_mps is None
                        else f"{measurement.velocity_unfolded_mps:.6f}",
                        "" if measurement.fold_index is None else measurement.fold_index,
                        f"{measurement.peak_power_w:.6e}",
                        measurement.n_cells,
                        claimed.get(index, ""),
                    ]
                )

    tracks_path = out_dir / "tracks.csv"
    with tracks_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRACK_COLUMNS)
        for record in frames:
            for track in record.tracks:
                covariance = track.estimate.covariance
                writer.writerow(
                    [
                        record.frame_index,
                        f"{record.time_s:.3f}",
                        track.track_id,
                        track.status,
                        track.measurement_dim,
                        f"{track.estimate.state[0]:.3f}",
                        f"{track.estimate.state[1]:.6f}",
                        f"{covariance[0, 0]:.6e}",
                        f"{covariance[1, 1]:.6e}",
                        "" if track.last_nis is None else f"{track.last_nis:.6f}",
                        int(track.track_id in record.associations),
                    ]
                )
    return detections_path, tracks_path


def render_track_frame(
    out_dir: Path,
    frames: Sequence[FrameTracks],
    truth_time_s: Sequence[float],
    truth_range_m: Sequence[float],
) -> None:
    """Draw the range-against-time history up to the most recent frame.

    Called once per frame so the series grow, which is what makes the assembled
    movie show a track being built rather than a finished plot appearing.
    """
    from radar_forge.teaching.plotting import save_figure
    from radar_forge.teaching.scopes.track_plot import render_range_time_history

    record = frames[-1]
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
    track_range_m: list[float] = []
    track_sigma_m: list[float] = []
    if counts:
        best_id = max(counts, key=lambda key: counts[key])
        for entry in frames:
            for track in entry.tracks:
                if track.track_id == best_id and track.is_confirmed:
                    track_time_s.append(entry.time_s)
                    track_range_m.append(float(track.estimate.state[0]))
                    track_sigma_m.append(float(np.sqrt(track.estimate.covariance[0, 0])))

    figure = render_range_time_history(
        truth_time_s,
        truth_range_m,
        detection_time_s=detection_time_s,
        detection_range_m=detection_range_m,
        track_time_s=track_time_s or None,
        track_range_m=track_range_m or None,
        track_sigma_range_m=track_sigma_m or None,
        current_time_s=record.time_s,
        title=f"frame {record.frame_index}  t = {record.time_s:.1f} s",
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
    # stubs, so it only fails the type gate once the teaching extra is present.
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
    args = parse_args(argv)
    scenario = load_scenario(args.scenario)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frames = scenario.n_frames if args.frames is None else min(args.frames, scenario.n_frames)
    print(f"{scenario.name}: {n_frames} frames, {len(scenario.bursts)} burst(s) -> {out_dir}")

    removed = clear_previous_frames(out_dir)
    if removed:
        print(f"  removed {removed} frame file(s) from an earlier run")

    tracking_enabled = scenario.tracking_table is not None and not args.no_tracking
    tracker: ScenarioTracker | None = None
    tracked_frames: list[FrameTracks] = []
    truth_time_s: list[float] = []
    truth_range_m: list[float] = []
    if tracking_enabled:
        tracker, simulated_angles = build_tracker(scenario)
        print(
            f"  tracking: {tracker.tracking.state_model}, "
            f"unfolding {tracker.tracking.unfolding_mode}, "
            f"bootstrap {tracker.min_unfold_frames} frames"
        )
        if simulated_angles:
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
                record = track_frame(tracker, scenario, frame, products)
                tracked_frames.append(record)
                truth_time_s.append(frame.time_s)
                truth_range_m.append(frame.range_m)

            if not args.no_plots:
                render_frame(frame, scenario, products, out_dir, args.dynamic_range_db, record)
                if record is not None:
                    render_track_frame(out_dir, tracked_frames, truth_time_s, truth_range_m)
            if show_progress:
                peaks = [peak_range_velocity(product) for product in products]
                summary = "  ".join(f"[{r / 1e3:6.2f} km {v:+7.2f} m/s]" for r, v in peaks)
                print(f"  frame {written}/{n_frames}  peak {summary}")

    write_metadata(scenario, out_dir, written)
    print(f"  wrote {truth_path}")

    if tracked_frames:
        detections_path, tracks_path = write_tracking_csvs(out_dir, tracked_frames)
        print(f"  wrote {detections_path}")
        print(f"  wrote {tracks_path}")
        confirmed = {
            track.track_id
            for record in tracked_frames
            for track in record.tracks
            if track.is_confirmed
        }
        n_tracked = sum(
            1 for record in tracked_frames if any(t.is_confirmed for t in record.tracks)
        )
        print(
            f"  tracked {n_tracked}/{len(tracked_frames)} frames "
            f"under {len(confirmed)} confirmed track id(s)"
        )

    if not args.no_plots and not args.no_movie and tracked_frames:
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
