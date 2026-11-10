"""Tests for radar_forge.tracking.scans: a plot-level scenario's rotations and its tracker.

One rotation of V4 is checked for what it must hold; the sector batching is scenario 011's A12,
a Monte Carlo over 20 seeds through the multitarget stage.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from radar_forge import evaluate
from radar_forge.scenario import load_scenario
from radar_forge.tracking.scans import scans

TERMINAL = Path(__file__).resolve().parents[2] / "config" / "scenario_011_terminal_dense.toml"


def test_a_plot_level_rotation_holds_its_echoes_clutter_and_discretes() -> None:
    """One V4 rotation: every plot timed within its rotation and batched by its azimuth."""
    scenario = load_scenario(TERMINAL)
    rotation = next(scans(scenario, n_frames=1))
    assert scenario.sensor is not None
    period_s = scenario.sensor.scan_period_s
    assert np.all((rotation.time_s >= 0.0) & (rotation.time_s < period_s))
    sector = np.floor(rotation.azimuth_rad / (2 * np.pi / scenario.sensor.n_sectors))
    np.testing.assert_array_equal(
        rotation.sector, np.minimum(sector, scenario.sensor.n_sectors - 1)
    )
    kinds = set(rotation.source)
    assert {"target", "clutter", "discrete"} <= kinds
    assert rotation.echo_in_coverage.sum() >= 10


# Slow: 20 seeds of V4 at 16 and at 32 sectors, 25 rotations each, about six minutes.
@pytest.mark.slow
def test_halving_the_sector_moves_the_terminal_scenes_gospa_less_than_its_seed_spread(tmp_path):
    """A12: mean GOSPA at 32 sectors minus at 16, under one standard deviation over 20 seeds.

    The window is the first 25 rotations (120 s), not 125, to hold the test to minutes; the
    stamp error the sectors trade is the same in every rotation.
    """
    scenario = load_scenario(TERMINAL)
    assert scenario.sensor is not None and scenario.tracking is not None
    # The configured JIPDA alone: the other trackers V4 scores (011b) do not enter A12.
    tracking = dataclasses.replace(scenario.tracking, trackers=())
    gospa_m = {16: [], 32: []}
    for seed in range(20):
        for n_sectors in (16, 32):
            sensor = dataclasses.replace(scenario.sensor, n_sectors=n_sectors)
            run = dataclasses.replace(
                scenario, seed=20261108 + seed, sensor=sensor, tracking=tracking, verify={}
            )
            report = evaluate.run(run, ["multitarget"], n_frames=25, out_dir=tmp_path, render=False)
            gospa_m[n_sectors].append(report.values["gospa_mean_m"])
    shift = abs(np.mean(gospa_m[32]) - np.mean(gospa_m[16]))
    spread = np.std(gospa_m[16], ddof=1)
    print(
        "mean GOSPA, 16 and 32 sectors:", np.mean(gospa_m[16]), np.mean(gospa_m[32]), "sd", spread
    )
    assert shift < spread


def test_gnn_clutter_density_follows_the_drifting_rain_cell() -> None:
    """The regions' mean intensity is set each rotation, at that rotation's time."""
    from radar_forge.tracking.scans import PlotTracker

    scenario = load_scenario(TERMINAL)
    tracker = PlotTracker(scenario)
    assert tracker.mean_intensity(0.0) != tracker.mean_intensity(600.0)
    rotations = scans(scenario, n_frames=2)
    next(rotations)
    second = next(rotations)
    tracker.rotation(second)
    assert tracker.tracker.clutter_density["plots"] == tracker.mean_intensity(second.start_time_s)
