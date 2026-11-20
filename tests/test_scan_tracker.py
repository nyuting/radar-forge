"""Tests for radar_forge.scan_tracker: one tracker for a staring and a rotating scenario.

Staring (scenario 003): the fold monitor's slope, a birth's range history, and JPDA, which only
the one tracker can give an IQ scenario, against the target's truth, and refused where a scan
offers range alone beside range and rate. Rotating (scenario 011):
the sector batching by a Monte Carlo over 20 seeds through the multitarget stage, and the
likelihood ratio's density against the drifting rain cell.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from radar_forge import verify
from radar_forge.pipeline import Plots
from radar_forge.plot_simulator import plot_scans
from radar_forge.scan_tracker import ScanTracker
from radar_forge.scenario import load_scenario, truth_geometry
from radar_forge.tracking.filters import is_confirmed

REPO = Path(__file__).resolve().parents[1]
TERMINAL = REPO / "config" / "scenario_011_terminal_dense.toml"


def _one_plot(range_m: float, folded_rate_mps: float, rate_mps: float = float("nan")) -> Plots:
    """One accepted plot of burst 0, read as the map shows it; its rate nan until unfolded."""
    return Plots(
        (),
        np.array([0]),
        np.array([range_m]),
        np.array([folded_rate_mps]),
        np.array([rate_mps]),
        np.array([43.3]),
        np.array([0.0345]),
        ("accepted",),
        np.array([-1]),
        np.array([np.nan]),
    )


# Review fixes: each a behaviour the staring tracker's policy got wrong.


def test_the_fold_monitor_fits_the_slope_on_the_scans_own_times():
    """Regression: a coast squeezed the slope's time axis, so a four-scan gap read 80 m/s for 50.

    The prediction sits one fold high; the slope over the real times (50 m/s) disagrees by a
    whole span and refolds the plot to the truth. On scan indices the slope agreed with the
    wrong prediction and the wrong fold stood.
    """
    tracker = ScanTracker(load_scenario(REPO / "config" / "scenario_003_tracking.toml"))
    period_mps, rate_mps = tracker.rate_period_mps, 50.0
    ids = tracker.tracker.seed(
        [[20_600.0, rate_mps + period_mps]], np.diag([100.0, 1.0])[None], 12.0
    )
    times_s = [*range(9), 12.0]
    tracker.history[int(ids[0])] = [(float(t), 20_000.0 + rate_mps * t) for t in times_s]
    folded_mps = rate_mps - period_mps * round(rate_mps / period_mps)
    # abs 1e-9: whole spans added to a folded reading, a few float64 operations.
    np.testing.assert_allclose(tracker.unfold(np.array([folded_mps])), [rate_mps], atol=1e-9)


def test_a_track_birth_starts_its_range_history():
    """Regression: a birth has no NIS, so its range was dropped and unfolding came a scan late."""
    tracker = ScanTracker(load_scenario(REPO / "config" / "scenario_003_tracking.toml"))
    [(bank, _)] = tracker.process(_one_plot(20_000.0, 1.0), 0.0)
    assert tracker.history == {int(bank.track_id[0]): [(0.0, 20_000.0)]}


def test_an_iq_scenario_tracks_under_jpda_and_confirms_its_target() -> None:
    """Scenario 003's dual-PRF UKF with association = "jpda": the one tracker gives an IQ scene
    the JPDA family, which only the plot-level tracker had.

    One plot a scan at the target's true bistatic range, as the map folds it, and its rate,
    weighed against the map's uniform clutter density: three hits in the first three scans
    confirm the track (3 of 5), and it stays the one confirmed track, within a range bin.
    """
    scenario = load_scenario(REPO / "config" / "scenario_003_ukf_fmcw_dual_prf.toml")
    assert scenario.tracking is not None
    tracking = dataclasses.replace(scenario.tracking, association="jpda", detection_probability=0.9)
    tracker = ScanTracker(dataclasses.replace(scenario, tracking=tracking))
    times_s, truths = scenario.scan_times_s[:10], truth_geometry(scenario)[:10]
    confirmed = []
    # Scans are sequential: each one's tracks are the next one's prior.
    for time_s, truth in zip(times_s, truths, strict=True):
        range_m = float(truth.bistatic_range_m[0]) % tracker.range_period_m
        rate_mps = float(truth.bistatic_range_rate_mps[0])
        [(bank, association)] = tracker.process(_one_plot(range_m, rate_mps, rate_mps), time_s)
        assert association is not None
        confirmed.append(int(is_confirmed(bank.status).sum()))
    assert confirmed == [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    (row,) = np.flatnonzero(is_confirmed(tracker.bank.status))
    gap_m = (truths[-1].bistatic_range_m[0] - tracker.bank.mean[row, 0]) % tracker.range_period_m
    bin_m = np.sqrt(12.0 * tracker.noise_covariance[0, 0])
    assert min(gap_m, tracker.range_period_m - gap_m) < bin_m


def test_jpda_is_refused_when_built_where_range_alone_is_offered_beside_range_and_rate() -> None:
    """Regression: scenario 003's range-only retry built JPDA, which raised on the first scan."""
    scenario = load_scenario(REPO / "config" / "scenario_003_tracking.toml")
    assert scenario.tracking is not None
    tracking = dataclasses.replace(scenario.tracking, association="jpda", detection_probability=0.9)
    with pytest.raises(ValueError, match="needs one measurement model a scan"):
        ScanTracker(dataclasses.replace(scenario, tracking=tracking))


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
    scenario = load_scenario(TERMINAL)
    tracker = ScanTracker(scenario)
    assert tracker.mean_density(0.0) != tracker.mean_density(600.0)
    scans = plot_scans(scenario, n_scans=2)
    next(scans)
    second = next(scans)
    tracker.process(second)
    assert tracker.tracker.clutter_density["plots"] == tracker.mean_density(second.start_time_s)
