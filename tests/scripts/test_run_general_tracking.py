"""Exercise the port's CLI boundary and its separate configuration schema."""

import csv
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from radar_forge.core.detection_2d import DetectionConfig
from radar_forge.pipelines.scenarios import load_scenario
from radar_forge.pipelines.tracking_config import TrackingConfig

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_general_tracking.py"


@pytest.fixture(scope="module")
def runner():
    return runpy.run_path(str(RUNNER))


def test_default_and_partial_configuration(runner, tmp_path):
    assert runner["load_tracker_config"](None) == (DetectionConfig(), TrackingConfig())
    path = tmp_path / "settings.toml"
    path.write_text("[tracking]\nhistory_size = 7\n")
    detection, tracking = runner["load_tracker_config"](path)
    assert detection == DetectionConfig()
    assert tracking == TrackingConfig(history_size=7)


@pytest.mark.parametrize(
    "contents",
    [
        "[unknown]\n",
        "tracking = 1\n",
        "[tracking]\nhistory_size = 1.5\n",
        "[tracking]\nconfirmation_hits = true\n",
        "[tracking]\nunknown = 1\n",
        "[detection]\npfa = '0.01'\n",
        "[detection]\ntraining_cells = 0\n",
    ],
)
def test_invalid_configuration_rejected(runner, tmp_path, contents):
    path = tmp_path / "settings.toml"
    path.write_text(contents)
    with pytest.raises(ValueError):
        runner["load_tracker_config"](path)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [("fmcw_low_prf", "S1"), ("pulsed_medium_prf", "S2"), ("fmcw_dual_prf", "S3")],
)
def test_cli_streams_outputs_and_resolved_settings(tmp_path, variant, expected):
    settings = tmp_path / "settings.toml"
    settings.write_text("[tracking]\nconfirmation_hits = 2\nconfirmation_window = 3\n")
    output = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(ROOT / "scenarios" / f"scenario_001_{variant}.toml"),
            "--tracker-config",
            str(settings),
            "--frames",
            "3",
            "--out",
            str(output),
            "--no-iq",
            "--no-plots",
            "--no-movie",
        ],
        cwd=tmp_path,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["scenario"]["n_frames"] == 3
    assert metadata["tracking"]["variant"] == expected
    assert metadata["tracking"]["tracking"]["confirmation_hits"] == 2
    assert metadata["tracking"]["detection"]["pfa"] == DetectionConfig().pfa
    metrics = json.loads((output / "tracking_metrics.json").read_text())
    assert metrics["n_frames"] == 3
    assert metrics["confirmation_latency_s"] == 1
    for filename in ("truth.csv", "detections.csv", "tracks.csv"):
        with (output / filename).open(newline="") as handle:
            assert list(csv.DictReader(handle)), filename
    if expected == "S2":
        assert metrics["range_interpretation"] == "modulo_absolute_unresolved"
    assert not list(output.glob("*.npz"))
    assert not list(output.glob("*.png"))


def test_bistatic_iq_rejected_before_outputs(runner, tmp_path):
    output = tmp_path / "outputs"
    with pytest.raises(ValueError, match="requires monostatic"):
        runner["main"](
            [
                str(ROOT / "scenarios/scenario_002_bistatic_xband.toml"),
                "--out",
                str(output),
            ]
        )
    assert not output.exists()


def test_existing_scenario_tables_do_not_select_general_settings(runner):
    scenario = load_scenario(ROOT / "scenarios/scenario_003_tracking.toml")
    assert scenario.tracking_table is not None
    assert runner["load_tracker_config"](None) == (DetectionConfig(), TrackingConfig())
    assert runner["tracking_legs"](scenario) == scenario.bursts
