"""Tests for scripts/run_scenario.py: the files a run writes.

The column lists are checked against ``spec/data-001-formats.md`` itself, so
the writer and the spec cannot drift apart without a test failing. The one
end-to-end run is short and marked slow.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parent.parent.parent
SPEC = ROOT / "spec" / "data-001-formats.md"


def load_runner() -> ModuleType:
    """Import scripts/run_scenario.py, which is a script and not a package module."""
    spec = importlib.util.spec_from_file_location(
        "run_scenario", ROOT / "scripts" / "run_scenario.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def spec_columns(section: str) -> list[tuple[str, str]]:
    """The ``(column, requirement)`` rows of one data-001 table, in order.

    A row naming several columns (``range_index``, ``velocity_index``) gives
    one entry each. A pattern row such as ``cov_<i>_<j>`` gives the pattern,
    which :func:`position` matches.
    """
    text = SPEC.read_text(encoding="utf-8")
    start = text.index(f"### {section} ")
    end = text.index("\n### ", start + 1)
    rows = []
    for line in text[start:end].splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not cells[0].startswith("`"):
            continue
        requirement = cells[3] if section != "6.9" else cells[2]
        rows += [(name, requirement) for name in re.findall(r"`([a-z0-9_<>]+)`", cells[0])]
    return rows


def position(column: str, order: list[str]) -> int:
    """Index of the first spec row naming ``column``, by name or by pattern."""
    for index, name in enumerate(order):
        if re.fullmatch(re.sub(r"<[a-z]+>", "[a-z0-9_]+", name), column):
            return index
    raise AssertionError(f"{column} is in no row of the spec's table")


@pytest.mark.parametrize(
    ("section", "columns"),
    [
        ("6.5", runner.DETECTION_COLUMNS),
        ("6.6", runner.TRACK_COLUMNS),
        ("6.9", runner.METRIC_COLUMNS),
    ],
    ids=["detections", "tracks", "metrics"],
)
def test_columns_are_the_specs_in_the_specs_order(section, columns):
    """Every required column is written, and no column is out of the spec's order."""
    listed = spec_columns(section)
    order = [name for name, _ in listed]
    positions = [position(column, order) for column in columns]
    assert positions == sorted(positions)
    required = {name for name, requirement in listed if requirement in ("R", "T", "E")}
    assert required <= set(columns), required - set(columns)


class TestCell:
    """data-001 §5: empty for not applicable, 0/1 booleans, shortest round-trip floats."""

    def test_none_is_an_empty_cell(self):
        assert runner._cell(None) == ""

    @pytest.mark.parametrize(("value", "cell"), [(True, "1"), (False, "0")])
    def test_booleans_are_zero_or_one(self, value, cell):
        assert runner._cell(value) == cell

    @pytest.mark.parametrize("value", [0.1, 1.0 / 3.0, 21.635652855125496, 1e-300, -0.0])
    def test_a_float_reads_back_exactly(self, value):
        assert float(runner._cell(value)) == value

    def test_a_float_is_not_padded_to_a_fixed_precision(self):
        assert runner._cell(0.5) == "0.5"


@pytest.mark.parametrize(
    "name",
    ["scenario_003_tracking", "scenario_003_tracking_dual_prf", "scenario_003_ukf_fmcw_dual_prf"],
)
def test_metadata_tracking_keeps_every_key_of_the_toml_table(name):
    """data-001 §6.1: ``tracking`` is the [tracking] table as written, plus additions.

    Renaming a key is a major change (§8), so each key of the table must come
    back under its own name, with its own value.
    """
    scenario = runner.load_scenario(ROOT / "scenarios" / f"{name}.toml")
    tracker = runner.ScenarioTracker.from_scenario(scenario)
    tracking = runner.tracking_metadata(scenario, tracker)
    assert scenario.tracking_table is not None
    for key, value in scenario.tracking_table.items():
        assert tracking[key] == value, key


@pytest.mark.slow
def test_a_ukf_run_writes_the_data_001_files(tmp_path):
    """Three frames of the S3 UKF scenario: the headers, the metadata keys and the metrics."""
    out_dir = tmp_path / "run"
    status = runner.main(
        [
            str(ROOT / "scenarios" / "scenario_003_ukf_fmcw_dual_prf.toml"),
            "--out",
            str(out_dir),
            "--frames",
            "3",
            "--no-iq",
            "--no-plots",
        ]
    )
    assert status == 0

    def header(name):
        with (out_dir / name).open(newline="", encoding="utf-8") as handle:
            return tuple(next(csv.reader(handle)))

    # The trajectory has an absolute epoch, so time_utc is written.
    assert header("detections.csv") == runner.DETECTION_COLUMNS
    assert header("tracks.csv") == runner.TRACK_COLUMNS
    assert header("metrics.csv") == runner.METRIC_COLUMNS

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == runner.SCHEMA_VERSION
    assert metadata["conventions"]["velocity_sign"] == "closing_positive"
    assert len(metadata["bursts"]) == 2
    assert metadata["tracking"]["state_fields"] == ["range_m", "range_rate_mps"]
    assert metadata["tracking"]["estimator"] == "ukf"
    assert metadata["epoch_utc"].endswith("Z")
    assert metadata["reference_site"]["role"] == "monostatic"

    with (out_dir / "detections.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["sensor_id"] for row in rows} == {"rx0"}
    assert {row["burst_index"] for row in rows} == {"0", "1"}
