"""Performance budgets: one frame's memory, and each stage's wall time.

Regressions, not correctness (``benchmark``). Memory is measured with ``tracemalloc`` over one
frame of the largest shipped scenario, from synthesis to plots: the chain may hold the cube,
its map and their working copies, but nothing that grows with the frame count or the paths.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import pytest

from radar_forge import evaluate
from radar_forge.pipeline import frame_plots, frames, range_doppler_product
from radar_forge.scenario import load_scenario

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "config").glob("*.toml"))


def _cube_bytes(path: Path) -> int:
    scenario = load_scenario(path)
    # complex128: 16 bytes a sample.
    return sum(
        16 * n * r.n_samples_per_pri
        for r, n in zip(scenario.radars, scenario.n_pulses, strict=True)
    )


@pytest.mark.benchmark
def test_one_frame_of_the_largest_scenario_holds_at_most_two_cubes_and_64_mib():
    """Synthesis to plots in one frame stays within twice the cube plus a fixed 64 MiB."""
    largest = max(CONFIGS, key=_cube_bytes)
    scenario = load_scenario(largest)
    tracemalloc.start()
    try:
        frame = next(frames(scenario, n_frames=1))
        products = [
            range_doppler_product(c, r) for c, r in zip(frame.cubes, scenario.radars, strict=True)
        ]
        frame_plots(products, scenario.detection)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak_bytes <= 2 * _cube_bytes(largest) + 64 * 2**20


# Budgets, s, for three frames of S1 without figures: about four times this machine's time.
STAGE_BUDGET_S = {"geometry": 2.0, "iq": 2.0, "rd": 8.0, "detect": 1.0, "track": 2.0}


@pytest.mark.benchmark
def test_each_stage_of_three_frames_runs_within_its_budget(tmp_path):
    """Three frames of scenario 003 over S1 through every stage, each inside its budget."""
    scenario = load_scenario(REPO / "config" / "scenario_003_tracking.toml")
    report = evaluate.run(scenario, n_frames=3, out_dir=tmp_path, render=False)
    over = {k: t for k, t in report.wall_time_s.items() if t > STAGE_BUDGET_S[k]}
    assert set(report.wall_time_s) == set(STAGE_BUDGET_S)
    assert not over, f"stages over budget (s): {over}"
