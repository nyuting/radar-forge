"""Spec 011b V3, the separation sweep, through the multitarget stage's own code.

Each trial is V3's scene (``config/scenario_011_cst_sweep.toml``) with the pair moved to one
point, a separation d in resolution cells across or along range and a clutter level n_gate,
and given its own seed; ``stages.tracking.play`` plays it once into all seven trackers and
``pair_scores`` judges each. The sweep is 30 points of 100 trials, ``slow``; two points run
unmarked, over one seeded trial of 20 rotations.
"""

from __future__ import annotations

import dataclasses
import math
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from radar_forge.scenario import ArcTarget, Scenario, load_scenario
from radar_forge.stages.tracking import pair_scores, play
from radar_forge.tracking.scene import Arc

V3 = load_scenario(Path(__file__).resolve().parents[2] / "config" / "scenario_011_cst_sweep.toml")
# The reference gate of spec 011 §3.2, m rad: n_gate expected false plots in it a scan.
REFERENCE_GATE_M_RAD = 4.6123
POINTS = [
    (d, n_gate, across)
    for d in (0.25, 0.5, 1.0, 2.0, 4.0)
    for n_gate in (0.1, 1.0, 3.0)
    for across in (True, False)
]


def _scene(d: float, n_gate: float, across: bool, seed: int, n_frames: int = 40) -> Scenario:
    """Return V3 with its pair d cells apart, across range (azimuth) or along it, at 20 km."""
    assert V3.sensor is not None
    if across:
        azimuth = math.pi + np.array([-0.5, 0.5]) * d * math.radians(
            V3.sensor.beamwidth_azimuth_deg
        )
        range_m = np.full(2, 20e3)
    else:
        azimuth = np.full(2, math.pi)
        range_m = 20e3 + np.array([-0.5, 0.5]) * d * V3.sensor.range_cell_m
    arcs = tuple(
        ArcTarget(f"P{k}", "aircraft", Arc(0.0, (r * math.sin(a), r * math.cos(a)), a, 72.0))
        for k, (r, a) in enumerate(zip(range_m, azimuth, strict=True))
    )
    disc = dataclasses.replace(
        V3.clutter_plots.components[0], intensity_per_m_rad=n_gate / REFERENCE_GATE_M_RAD
    )
    return dataclasses.replace(
        V3,
        arcs=arcs,
        clutter_plots=dataclasses.replace(V3.clutter_plots, components=(disc,)),
        duration_s=n_frames * V3.sensor.scan_period_s,
        seed=seed,
        verify={},
    )


@cache
def _resolved(d: float, n_gate: float, across: bool, trials: int) -> dict[str, NDArray[np.float64]]:
    """Return each tracker's resolved (1) or not (0) in each trial of one point."""
    resolved: dict[str, list[float]] = {}
    # One scene per trial: each is a 40-scan run of all seven trackers on the same plots.
    for trial in range(trials):
        scenario = _scene(d, n_gate, across, 20261109 + trial)
        for name, run in play(scenario)[0].items():
            resolved.setdefault(name, []).append(pair_scores(scenario, run)[0])
    return {name: np.array(values) for name, values in resolved.items()}


@pytest.mark.parametrize("across", [True, False], ids=["across", "along"])
def test_a_pair_four_cells_apart_in_light_clutter_is_resolved_by_every_tracker(
    across: bool,
) -> None:
    """A pair 4 cells apart merges in 0.03 % of scans (P_u(4) = e^-8) and n_gate = 0.1 clutter
    rarely draws a track off: every tracker, from its cue at the truth, holds both aircraft.
    One seeded trial of the 20 rotations a pair is judged over."""
    scenario = _scene(4.0, 0.1, across, 20261109, n_frames=20)
    resolved = {name: pair_scores(scenario, run)[0] for name, run in play(scenario)[0].items()}
    assert resolved == dict.fromkeys(resolved, 1.0)
    assert set(resolved) == {"jpda", "gnn", "jpda_star", "set", "jipda", "mht", "pmbm"}


# Slow: 30 points of 100 trials, each 40 rotations of seven trackers: about an hour.
@pytest.mark.slow
@pytest.mark.parametrize(
    ("d", "n_gate", "across"),
    POINTS,
    ids=[f"d{d}-n{n}-{'across' if a else 'along'}" for d, n, a in POINTS],
)
def test_pmbm_resolves_a_pair_as_often_as_any_tracker_less_0_05(
    d: float, n_gate: float, across: bool
) -> None:
    """A17: PMBM's P(resolved) is at least every other tracker's less 0.05, at every point.

    The trackers see the same plots, so their resolved counts are paired: PMBM's lead over
    each is the mean of the paired differences over 100 trials. [14]: the others approximate
    PMBM's posterior; one beating it materially is a bug in one of them.
    """
    resolved = _resolved(d, n_gate, across, 100)
    print({name: float(values.mean()) for name, values in resolved.items()})
    for name, values in resolved.items():
        assert resolved["pmbm"].mean() >= values.mean() - 0.05, name


# Slow: the 100 trials of A17's point d = 1 across range at n_gate = 1, shared with it.
@pytest.mark.slow
def test_mht_resolves_the_pair_one_cell_apart_as_often_as_jpda() -> None:
    """A16 (MHT clause): at d = 1 across range, n_gate = 1, P(resolved) for MHT >= JPDA's.

    Measured while building it: every tracker 0.00 of 60 trials; 61 % of scans merge the
    pair into one plot 244 m from each aircraft, outside both 99 % gates, so the clause
    holds without saying much.
    """
    resolved = _resolved(1.0, 1.0, True, 100)
    print({name: float(values.mean()) for name, values in resolved.items()})
    assert resolved["mht"].mean() >= resolved["jpda"].mean()
