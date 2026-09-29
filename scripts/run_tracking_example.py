#!/usr/bin/env python3
"""Seeded 1D/2D/3D ENU position-observation examples, independent of Duke IQ.

uv run python scripts/run_tracking_example.py --axes xyz --acceleration --out out/enu.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radar_forge.pipelines.tracking_config import build_enu_tracker
from radar_forge.pipelines.tracking_output import snapshot_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axes", choices=["x", "xy", "xyz"], default="xyz")
    parser.add_argument("--acceleration", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    kind = "CA" if args.acceleration else "CV"
    engine = build_enu_tracker(
        dict.fromkeys(args.axes, kind),
        origin_lla_deg_m=(36.00250, -78.94100, 60.0),
        noise_density=0.01,
    )
    sensor = engine.sensors["sensor"]
    rng = np.random.default_rng(20260915)
    dimensions = len(args.axes)
    initial_m = np.arange(1, dimensions + 1, dtype=np.float64) * 100
    velocity_mps = np.arange(1, dimensions + 1, dtype=np.float64) * 2
    acceleration_mps2 = np.full(dimensions, 0.1 if args.acceleration else 0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        # Filtering is a sequential recursion; only one synthetic event is retained.
        for time_s in range(120):
            truth_m = initial_m + velocity_mps * time_s + acceleration_mps2 * time_s**2 / 2
            observed_m = truth_m + rng.normal(0, 1, dimensions)
            batch = sensor.batch(time_s, [("measurement", observed_m, np.eye(dimensions))])
            for snapshot in engine.process(batch):
                handle.write(json.dumps(snapshot_record(snapshot), allow_nan=False) + "\n")
    print(f"Wrote {kind} {args.axes} ENU tracking example to {args.out}")


if __name__ == "__main__":
    main()
