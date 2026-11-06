"""Performance budgets: one frame's memory, and each stage's wall time.

Regressions, not correctness (``benchmark``). Memory is measured with ``tracemalloc`` over one
frame, from synthesis to plots, of the largest shipped scenario and of every one that corrects
range migration: the chain may hold the cube (as its range gate records it), its map and their
working copies, and a keystone its chirp-z plan and a few range spectra, but nothing that grows
with the frame count or the paths.
"""

from __future__ import annotations

import math
import time
import tracemalloc
from pathlib import Path

import pytest

from radar_forge import evaluate, migration
from radar_forge.detection import cluster_detections
from radar_forge.pipeline import frame_plots, frame_products, frames
from radar_forge.scenario import load_scenario

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "config").glob("*.toml"))


def _cube_bytes(path: Path) -> int:
    scenario = load_scenario(path)
    # complex128: 16 bytes a sample, of the samples each burst's gate records. A noise burst
    # holds its code beside its stream, and its map is n_range_lags wide, not a batch: those
    # count with the cube. A receive array multiplies it by its channels (scenario 006).
    n_rx = 1 if scenario.receive_array is None else scenario.receive_array.n_rx
    return n_rx * sum(
        16 * n * (gate.n_samples(r) * (2 if lags else 1) + lags)
        for r, n, gate, lags in zip(
            scenario.radars, scenario.n_pulses, scenario.gates, scenario.n_range_lags, strict=True
        )
    )


def _migration_bytes(path: Path) -> int:
    """A keystone's working set: its chirp-z plan, 16 M (2N + L) bytes, and six range spectra.

    The spectra, (N, M) complex each: the range spectrum, its frequency-major copy, a search's
    trial and its return to range, the corrected cube and its map. M is the padded range
    length, L the chirp-z convolution length.
    """
    scenario = load_scenario(path)
    if scenario.migration.method == "none":
        return 0
    total = 0
    for radar, n_pulses, gate in zip(
        scenario.radars, scenario.n_pulses, scenario.gates, strict=True
    ):
        n_range_fft = 2 ** math.ceil(math.log2(2 * gate.n_samples(radar)))
        n_convolution = 2 ** math.ceil(math.log2(2 * n_pulses - 1))
        spectrum_bytes = 16 * n_pulses * n_range_fft
        total += 16 * n_range_fft * (2 * n_pulses + n_convolution) + 6 * spectrum_bytes
    return total


def _array_bytes(path: Path) -> int:
    """A receive array's map: the range profiles and the Doppler spectrum, a cube each.

    Each transform makes a copy of the multi-channel cube, 65.5 MB at scenario 006's 16
    channels, beyond the 64 MiB slack a one-channel chain's copies fit in.
    """
    return 0 if load_scenario(path).receive_array is None else 2 * _cube_bytes(path)


MEASURED = sorted(
    {max(CONFIGS, key=_cube_bytes)}
    | {path for path in CONFIGS if load_scenario(path).migration.method != "none"}
)


@pytest.mark.benchmark
@pytest.mark.parametrize("path", MEASURED, ids=[path.stem for path in MEASURED])
def test_one_frame_holds_at_most_two_cubes_a_keystones_working_set_and_64_mib(path):
    """Synthesis to plots in one frame, through the gate and [migration], within budget.

    The chirp-z plan cache (one plan, migration.py) is emptied first, so building the plan is
    measured too.
    """
    scenario = load_scenario(path)
    migration._chirp_z_plan.cache_clear()
    tracemalloc.start()
    try:
        frame = next(frames(scenario, n_frames=1))
        frame_plots(frame_products(scenario, frame), scenario.detection)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    budget_bytes = 2 * _cube_bytes(path) + _migration_bytes(path) + _array_bytes(path)
    assert peak_bytes <= budget_bytes + 64 * 2**20


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


@pytest.mark.benchmark
def test_clustering_a_frame_sized_map_runs_within_its_budget(rng):
    """A (256, 1000) map, ten 3 x 5 targets and about 30 false alarms, clustered inside 0.02 s.

    Both axes wrap. The whole-map ndimage summaries took 32 ms here; the labelled cells alone
    take about 3 ms.
    """
    power_w = rng.exponential(size=(256, 1000))
    # Ten blocks, one assignment each: a loop over the targets, not the cells.
    for k in range(10):
        power_w[25 * k + 3 : 25 * k + 6, 97 * k + 5 : 97 * k + 10] = 100.0
    # A CFAR-like threshold: unit exponential noise crosses it in about 30 cells.
    mask = power_w > math.log(power_w.size / 30.0)
    start_s = time.perf_counter()
    found = cluster_detections(mask, power_w, wrap_axes=(0, 1))
    elapsed_s = time.perf_counter() - start_s
    assert sum(d.n_cells >= 15 for d in found) == 10
    assert 20 <= len(found) <= 60
    assert elapsed_s < 0.02, f"clustering took {elapsed_s:.3f} s"
