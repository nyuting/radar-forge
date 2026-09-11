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
from radar_forge.pipelines.scenarios import (
    Frame,
    RangeDopplerProduct,
    Scenario,
    iterate_frames,
    leg_range_doppler,
    load_scenario,
    peak_range_velocity,
)

TRUTH_COLUMNS = [
    "frame",
    "time_s",
    "range_m",
    "radial_velocity_mps",
    "azimuth_deg",
    "elevation_deg",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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


def leg_metadata(scenario: Scenario) -> list[dict[str, Any]]:
    """Everything about each leg a reader needs to interpret the cubes."""
    legs: list[dict[str, Any]] = []
    for index, (leg, n_chirps) in enumerate(zip(scenario.legs, scenario.n_chirps, strict=True)):
        legs.append(
            {
                "index": index,
                "waveform": leg.transmitter.waveform,
                "f0_hz": leg.transmitter.f0_hz,
                "bandwidth_hz": leg.transmitter.bandwidth_hz,
                "chirp_time_s": leg.transmitter.chirp_time_s,
                "prf_hz": leg.transmitter.prf_hz,
                "sample_rate_hz": leg.receiver.sample_rate_hz,
                "n_chirps": n_chirps,
                "n_samples": leg.n_samples_per_pri,
                "range_resolution_m": leg.range_resolution_m,
                "unambiguous_range_m": leg.unambiguous_range_m,
                "unambiguous_velocity_mps": leg.unambiguous_velocity_mps,
                "noise_power_w": leg.noise_power_w,
            }
        )
    return legs


def write_metadata(scenario: Scenario, out_dir: Path, n_frames: int) -> None:
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
        "radar_site": {
            "latitude_deg": scenario.legs[0].latitude_deg,
            "longitude_deg": scenario.legs[0].longitude_deg,
            "altitude_m": scenario.legs[0].altitude_m,
        },
        "target": asdict(scenario.target),
        "legs": leg_metadata(scenario),
        "provenance": {"radar_forge_version": __version__, "git_commit": git_commit()},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def render_frame(
    frame: Frame,
    scenario: Scenario,
    products: Sequence[RangeDopplerProduct],
    out_dir: Path,
    dynamic_range_db: float,
) -> None:
    """Draw one range-Doppler panel per leg and save it.

    Takes the already-processed ``products`` rather than processing the cubes
    itself, so the receive chain runs once per frame however many consumers a
    frame has.
    """
    from radar_forge.teaching.plotting import save_figure
    from radar_forge.teaching.scopes.rd_map import render_range_doppler

    for index, (product, leg) in enumerate(zip(products, scenario.legs, strict=True)):
        suffix = "" if len(scenario.legs) == 1 else f"_leg{index}"
        # Where this leg's ambiguities oblige the target to appear: Doppler
        # wraps into the unambiguous interval, range modulo the unambiguous
        # range. For a leg that folds in neither, both are the truth itself.
        folded_velocity_mps = float(
            fold_velocity_mps(frame.radial_velocity_mps, leg.unambiguous_velocity_mps)
        )
        folded_range_m = frame.range_m % leg.unambiguous_range_m
        title = (
            f"{scenario.name}{suffix}  t = {frame.time_s:.0f} s   "
            f"truth {frame.range_m / 1e3:.2f} km, {frame.radial_velocity_mps:+.1f} m/s"
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
        )
        save_figure(figure, out_dir / f"rd{suffix}_{frame.index:05d}.png")


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
    stale += sorted(out_dir.glob("iq_[0-9][0-9][0-9][0-9][0-9].npz"))
    for path in stale:
        path.unlink()
    return len(stale)


def assemble_movie(out_dir: Path, pattern: str, stem: str, frame_rate_hz: float) -> str | None:
    """Assemble the PNG frames into an MP4, falling back to an animated GIF."""
    # Match the five digits the pattern formats, so a single-leg run's
    # ``rd_%05d.png`` does not also sweep up a previous multi-leg run's
    # ``rd_leg0_00000.png``.
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
    first = Image.open(frames[0]).convert("P", palette=Image.ADAPTIVE)
    rest = (Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frames[1:])
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
    print(f"{scenario.name}: {n_frames} frames, {len(scenario.legs)} leg(s) -> {out_dir}")

    removed = clear_previous_frames(out_dir)
    if removed:
        print(f"  removed {removed} frame file(s) from an earlier run")

    truth_path = out_dir / "truth.csv"
    written = 0
    with truth_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRUTH_COLUMNS)

        for frame in iterate_frames(scenario):
            if frame.index >= n_frames:
                break
            # Truth is the true, unfolded geometry in every variant; the gap
            # between it and the map is what the scenario is for.
            writer.writerow(
                [
                    frame.index,
                    f"{frame.time_s:.3f}",
                    f"{frame.range_m:.3f}",
                    f"{frame.radial_velocity_mps:.6f}",
                    f"{frame.azimuth_deg:.6f}",
                    f"{frame.elevation_deg:.6f}",
                ]
            )

            if not args.no_iq:
                arrays = {f"leg{index}": cube for index, cube in enumerate(frame.iq)}
                # One named array per leg. The ignore is a stub limitation:
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
            if not args.no_plots or show_progress:
                products = [
                    leg_range_doppler(cube, leg)
                    for cube, leg in zip(frame.iq, scenario.legs, strict=True)
                ]

            if not args.no_plots:
                render_frame(frame, scenario, products, out_dir, args.dynamic_range_db)
            if show_progress:
                peaks = [peak_range_velocity(product) for product in products]
                summary = "  ".join(f"[{r / 1e3:6.2f} km {v:+7.2f} m/s]" for r, v in peaks)
                print(f"  frame {written}/{n_frames}  peak {summary}")

    write_metadata(scenario, out_dir, written)
    print(f"  wrote {truth_path}")

    if not args.no_plots and not args.no_movie:
        for index in range(len(scenario.legs)):
            suffix = "" if len(scenario.legs) == 1 else f"_leg{index}"
            movie = assemble_movie(
                out_dir, f"rd{suffix}_%05d.png", f"rd{suffix}", scenario.frame_rate_hz
            )
            if movie is not None:
                print(f"  wrote {movie}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
