#!/usr/bin/env python3
"""Write tests/data/golden/fingerprints.csv: a dozen scalars per config/*.toml, three frames.

    uv run python scripts/regen_golden.py

Each row is one scenario's three-frame run through every stage its ``[verify]`` table names
(``radar_forge.evaluate.run``): where the peak lands, how bright it and the floor are, how many
detections, and how well the tracker did. ``tests/test_scenarios.py`` compares a fresh run
against it at rtol 1e-9, so regenerating it is a behaviour change: review the diff, and commit
it on its own with the reason.
"""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path

from radar_forge.evaluate import run
from radar_forge.scenario import load_scenario

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "data" / "golden" / "fingerprints.csv"
N_FRAMES = 3
FINGERPRINT = (
    "predicted_snr_db",
    "iq_noise_power_dbw",
    "iq_peak_power_dbw",
    "rd_peak_range_index",
    "rd_peak_rate_index",
    "rd_peak_power_dbw",
    "rd_noise_floor_dbw",
    "n_detections",
    "detection_fraction",
    "n_accepted",
    "rate_error_max_mps",
    "range_rmse_m",
    "nees_fraction",
)


def fingerprint(path: Path) -> dict[str, float]:
    """Return the fingerprint scalars of a scenario's three-frame run; nan where none."""
    with tempfile.TemporaryDirectory() as out_dir:
        report = run(load_scenario(path), n_frames=N_FRAMES, out_dir=out_dir, render=False)
    return {name: report.values.get(name, math.nan) for name in FINGERPRINT}


def main() -> int:
    rows = [
        {"scenario": path.stem, **{k: repr(v) for k, v in fingerprint(path).items()}}
        for path in sorted((REPO / "config").glob("*.toml"))
    ]
    with GOLDEN.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", *FINGERPRINT])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} fingerprints to {GOLDEN.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
