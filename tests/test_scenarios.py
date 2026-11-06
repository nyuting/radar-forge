"""Every shipped scenario passes its own ``[verify]`` table and reproduces its fingerprint.

Three frames of each ``config/*.toml`` run through every stage it names, unmarked; the whole
window is ``slow``. A config in ``scripts/regen_golden.py``'s ``FAST_FRAMES`` runs that many
frames unmarked instead, and its three are ``slow`` (0: it runs only ``slow``). The checks are
the scenario's acceptance criteria, re-derived for bistatic range and rate in each file's
comments. The fingerprint is golden data (``docs/conventions/testing.md`` §6): it proves the
short run's numbers did not move, not that they are right.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from radar_forge import evaluate
from radar_forge.scenario import load_scenario

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "config").glob("*.toml"))
GOLDEN = REPO / "tests" / "data" / "golden" / "fingerprints.csv"


def _load_golden_script() -> ModuleType:
    # scripts/ is developer tooling, not an installed package, so it is loaded by path.
    spec = importlib.util.spec_from_file_location(
        "regen_golden", REPO / "scripts" / "regen_golden.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


regen_golden = _load_golden_script()
FAST_FRAMES: dict[str, int] = regen_golden.FAST_FRAMES
# Each config's first run: unmarked unless its fast frames are 0.
SHORT_RUNS = [
    pytest.param(
        path, id=path.stem, marks=[pytest.mark.slow] if FAST_FRAMES.get(path.stem) == 0 else []
    )
    for path in CONFIGS
]


# rtol 1e-9 holds a value computed the same way; an error is a difference of nearly equal
# numbers (a measured rate less the truth's), so the platform's FFT round-off, 1e-15 of each,
# can be 1e-8 of the difference. Each unit also gets an absolute tolerance far below anything
# physical: a micrometre a second against S1's 0.12 m/s rate bin, a millimetre against its
# 150 m range bin, a micro-decibel. Counts, indices and fractions stay exact.
ABSOLUTE_TOLERANCE = {"_mps": 1e-6, "_m": 1e-3, "_db": 1e-6, "_dbw": 1e-6}


def _atol(key: str) -> float:
    return next((atol for unit, atol in ABSOLUTE_TOLERANCE.items() if key.endswith(unit)), 0.0)


def _golden() -> dict[str, dict[str, float]]:
    with GOLDEN.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row.pop("scenario"): {k: float(v) for k, v in row.items()} for row in rows}


@pytest.fixture(scope="module", params=SHORT_RUNS)
def short_run(request, tmp_path_factory):
    """A config's fingerprinted frames (three, or its fast frames) through its stages."""
    path = request.param
    out_dir = tmp_path_factory.mktemp(path.stem)
    n_frames = regen_golden.fingerprint_frames(path.stem)
    return path.stem, evaluate.run(
        load_scenario(path), n_frames=n_frames, out_dir=out_dir, render=False
    )


def test_three_frames_pass_every_check_they_reach(short_run):
    """The stages run, and the checks a three-frame run can judge all pass."""
    _, report = short_run
    assert report.checks, "the scenario's [verify] table checks nothing"
    assert not report.failures


def test_three_frames_match_the_golden_fingerprint(short_run):
    """Regression: the three-frame numbers equal those of scripts/regen_golden.py."""
    name, report = short_run
    golden = _golden()
    assert name in golden, f"no fingerprint for {name}: run scripts/regen_golden.py"
    expected = golden[name]
    measured = {key: report.values.get(key, math.nan) for key in expected}
    mismatched = {
        key: (measured[key], value)
        for key, value in expected.items()
        if not np.isclose(measured[key], value, rtol=1e-9, atol=_atol(key), equal_nan=True)
    }
    assert not mismatched, (
        f"{name} moved (measured, golden): {mismatched}. If the change is intended, run "
        "`uv run python scripts/regen_golden.py`, review the diff and commit it on its own."
    )


def test_every_scenario_has_a_fingerprint():
    """A new config file gets a golden row, and a deleted one loses it."""
    assert set(_golden()) == {path.stem for path in CONFIGS}


# Slow: a config whose fast run is shorter runs its three frames here, 5 to 10 s each.
@pytest.mark.slow
@pytest.mark.parametrize(
    "path",
    [path for path in CONFIGS if FAST_FRAMES.get(path.stem)],
    ids=[path.stem for path in CONFIGS if FAST_FRAMES.get(path.stem)],
)
def test_three_frames_of_a_shortened_config_pass_every_check(path, tmp_path):
    """The three frames the fast suite skips for a config listed in FAST_FRAMES."""
    report = evaluate.run(load_scenario(path), n_frames=3, out_dir=tmp_path, render=False)
    assert report.checks
    assert not report.failures


# Slow: each window is 120 frames of synthesis, two FFTs and CFAR, 5 to 50 s a scenario.
@pytest.mark.slow
@pytest.mark.parametrize("path", CONFIGS, ids=[path.stem for path in CONFIGS])
def test_the_whole_window_passes_every_check(path, tmp_path):
    """The full window meets every criterion its [verify] table encodes."""
    report = evaluate.run(load_scenario(path), out_dir=tmp_path, render=False)
    assert report.checks
    assert not report.failures
