"""The stepper's command line: three frames of a shipped scenario, its files, its exit status.

Scenario 003 criterion 8 is that the pictures exist: this run draws them. S2's maps are the
smallest shipped, 256 x 100 cells, so it is the cheapest scenario to draw three frames of.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO = REPO_ROOT / "config" / "scenario_003_ukf_pulsed_medium_prf.toml"


def _load_script() -> ModuleType:
    # scripts/ is developer tooling, not an installed package, so it is loaded by path.
    spec = importlib.util.spec_from_file_location(
        "run_scenario", REPO_ROOT / "scripts" / "run_scenario.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_scenario = _load_script()


def test_three_frames_write_every_stage_file_and_exit_zero(tmp_path, capsys):
    """Every stage runs, prints its wall time, and writes its files; the checks pass.

    ``--no-plots``, so it runs without the visualisation extra, as the fast CI job does.
    """
    status = run_scenario.main(
        [str(SCENARIO), "--frames", "3", "--save-iq", "--no-plots", "--out", str(tmp_path)]
    )
    printed = capsys.readouterr().out
    assert status == 0
    assert printed.count("wall time") == 5
    expected = {"truth.csv", "detections.csv", "tracks.csv", "metrics.csv"}
    expected |= {f"iq_{k:05d}.npz" for k in range(3)}
    written = {path.name for path in tmp_path.iterdir()}
    assert expected <= written
    assert not any(name.endswith(".png") for name in written)
    with (tmp_path / "truth.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # Monostatic, so the file shows the slant range and radial velocity: half the map's values.
    assert [int(row["frame"]) for row in rows] == [0, 1, 2]
    assert {"range_m", "range_rate_mps"} <= set(rows[0])


def test_the_figures_are_drawn_with_the_visualisation_extra(tmp_path):
    """Scenario 003 criterion 8: an rd map per frame and the range-time history exist."""
    pytest.importorskip("matplotlib")
    status = run_scenario.main([str(SCENARIO), "--frames", "2", "--out", str(tmp_path)])
    assert status == 0
    expected = {"rd_00000.png", "rd_00001.png", "track.png"}
    assert expected <= {path.name for path in tmp_path.iterdir()}


def test_stop_after_runs_only_the_stages_up_to_it(tmp_path, capsys):
    """``--stop-after rd`` runs geometry, iq and rd, and nothing downstream of them."""
    status = run_scenario.main(
        [str(SCENARIO), "--frames", "1", "--stop-after", "rd", "--no-plots", "--out", str(tmp_path)]
    )
    printed = capsys.readouterr().out
    assert status == 0
    assert "[rd]" in printed and "[detect]" not in printed
    assert not (tmp_path / "detections.csv").exists()


def test_a_failed_check_exits_one_and_names_it(tmp_path, capsys):
    """A [verify] threshold the run misses makes the command fail, saying which and by how much."""
    text = SCENARIO.read_text(encoding="utf-8").replace(
        "expected_snr_db = 42.01", "expected_snr_db = 30.0"
    )
    text = text.replace(
        'trajectory_path = "../data/', f'trajectory_path = "{(REPO_ROOT / "data").as_posix()}/'
    )
    broken = tmp_path / "broken.toml"
    broken.write_text(text, encoding="utf-8")
    status = run_scenario.main(
        [str(broken), "--frames", "1", "--stop-after", "geometry", "--out", str(tmp_path / "run")]
    )
    assert status == 1
    assert "predicted_snr_db" in capsys.readouterr().err
