"""Tests for radar_forge.tracking.scans: the tracker a plot-level scenario's scans feed.

The sector batching is checked by a Monte Carlo over 20 seeds through the multitarget stage,
and the likelihood ratio's density against the drifting rain cell.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from radar_forge import verify
from radar_forge.plot_simulator import plot_scans
from radar_forge.scenario import load_scenario

TERMINAL = Path(__file__).resolve().parents[2] / "config" / "scenario_011_terminal_dense.toml"


# Slow: 20 seeds of V4 at 16 and at 32 sectors, 25 scans each, about six minutes.
@pytest.mark.slow
def test_halving_the_sector_moves_the_terminal_scenes_gospa_less_than_its_seed_spread(tmp_path):
    """Mean GOSPA at 32 sectors minus at 16, under one standard deviation over 20 seeds.

    The window is the first 25 scans (120 s), not 125, to hold the test to minutes; the
    stamp error the sectors trade is the same in every scan.
    """
    scenario = load_scenario(TERMINAL)
    assert scenario.sensor is not None and scenario.tracking is not None
    # The configured JIPDA alone: the other trackers V4 scores (011b) are not in the batching check.
    tracking = dataclasses.replace(scenario.tracking, trackers=())
    gospa_m = {16: [], 32: []}
    for seed in range(20):
        for n_sectors in (16, 32):
            sensor = dataclasses.replace(scenario.sensor, n_sectors=n_sectors)
            run = dataclasses.replace(
                scenario, seed=20261108 + seed, sensor=sensor, tracking=tracking, verify={}
            )
            report = verify.run(run, ["multitarget"], n_scans=25, out_dir=tmp_path, render=False)
            gospa_m[n_sectors].append(report.values["gospa_mean_m"])
    shift = abs(np.mean(gospa_m[32]) - np.mean(gospa_m[16]))
    spread = np.std(gospa_m[16], ddof=1)
    print(
        "mean GOSPA, 16 and 32 sectors:", np.mean(gospa_m[16]), np.mean(gospa_m[32]), "sd", spread
    )
    assert shift < spread


def test_gnn_clutter_density_follows_the_drifting_rain_cell() -> None:
    """The regions' mean density is set each scan, at that scan's time."""
    from radar_forge.tracking.scans import PlotTracker

    scenario = load_scenario(TERMINAL)
    tracker = PlotTracker(scenario)
    assert tracker.mean_density(0.0) != tracker.mean_density(600.0)
    scans = plot_scans(scenario, n_scans=2)
    next(scans)
    second = next(scans)
    tracker.rotation(second)
    assert tracker.tracker.clutter_density["plots"] == tracker.mean_density(second.start_time_s)
