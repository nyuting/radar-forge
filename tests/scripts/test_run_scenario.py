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
from datetime import UTC, datetime, timedelta
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


def without_time_utc(columns):
    """A column tuple as written for a run with no absolute epoch."""
    return tuple(column for column in columns if column != "time_utc")


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


@pytest.mark.parametrize(
    ("name", "unused"),
    [
        (
            "scenario_003_ukf_fmcw_dual_prf",
            {
                "sigma_range_m",
                "sigma_velocity_mps",
                "sigma_azimuth_deg",
                "sigma_elevation_deg",
                "unfold_sigma_gate",
                "n_slope_frames",
            },
        ),
        (
            "scenario_003_tracking",
            {"acceleration_correlation_time_s", "sigma_azimuth_deg", "sigma_elevation_deg"},
        ),
    ],
)
def test_metadata_says_which_settings_the_estimator_did_not_read(name, unused):
    """A UKF run records S1's frozen sigma_range_m, and must say it went unused."""
    scenario = runner.load_scenario(ROOT / "scenarios" / f"{name}.toml")
    tracker = runner.ScenarioTracker.from_scenario(scenario)
    tracking = runner.tracking_metadata(scenario, tracker)
    assert set(tracking["unused"]) == unused
    # Every unused name is a recorded key, so a reader can find what it labels.
    assert set(tracking["unused"]) <= set(tracking)


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

    # The shipped track counts time_s from its first fix and has no absolute
    # epoch, so time_utc is left out of every table (data-001 §5).
    assert header("detections.csv") == without_time_utc(runner.DETECTION_COLUMNS)
    assert header("tracks.csv") == without_time_utc(runner.TRACK_COLUMNS)
    assert header("metrics.csv") == without_time_utc(runner.METRIC_COLUMNS)

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == runner.SCHEMA_VERSION
    assert metadata["conventions"]["velocity_sign"] == "closing_positive"
    assert len(metadata["bursts"]) == 2
    assert metadata["tracking"]["state_fields"] == ["range_m", "range_rate_mps"]
    assert metadata["tracking"]["estimator"] == "ukf"
    assert metadata["epoch_utc"] is None
    assert metadata["reference_site"]["role"] == "monostatic"

    with (out_dir / "detections.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["sensor_id"] for row in rows} == {"rx0"}
    assert {row["burst_index"] for row in rows} == {"0", "1"}
    detection_keys = {(row["frame"], row["detection_id"]) for row in rows}

    # A track's first row names the detection it was born from, which is not
    # an update, so associated is 0 and the detection is left unassociated.
    with (out_dir / "tracks.csv").open(newline="", encoding="utf-8") as handle:
        track_rows = list(csv.DictReader(handle))
    first_rows = {}
    for row in track_rows:
        first_rows.setdefault(row["track_id"], row)
    assert first_rows
    for row in first_rows.values():
        assert row["associated"] == "0"
        assert (row["frame"], row["associated_detection_id"]) in detection_keys
        seed = next(
            r
            for r in rows
            if (r["frame"], r["detection_id"]) == (row["frame"], row["associated_detection_id"])
        )
        assert seed["associated_track_id"] == ""


@pytest.mark.slow
def test_a_trajectory_with_an_epoch_writes_time_utc(tmp_path):
    """data-001 §5: with an absolute epoch, every table carries ``time_utc``.

    The shipped track has none, so this runs one frame on a copy of it whose
    first column is ``time_utc``. The epoch is data-001 §5's own example.
    """
    epoch_utc = datetime(2026, 9, 3, 0, 17, 56, tzinfo=UTC)
    with (ROOT / "data" / "flight_coordinates.csv").open(newline="", encoding="utf-8") as handle:
        fixes = list(csv.DictReader(handle))
    track_path = tmp_path / "track_utc.csv"
    with track_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_utc", "latitude_deg", "longitude_deg"])
        for fix in fixes:
            instant = epoch_utc + timedelta(seconds=float(fix["time_s"]))
            writer.writerow(
                [instant.strftime("%Y-%m-%dT%H:%M:%SZ"), fix["latitude_deg"], fix["longitude_deg"]]
            )
    toml_text = (ROOT / "scenarios" / "scenario_003_ukf_fmcw_dual_prf.toml").read_text(
        encoding="utf-8"
    )
    toml_text, n_replaced = re.subn(
        r'^path = "[^"]*"', f"path = {json.dumps(str(track_path))}", toml_text, flags=re.MULTILINE
    )
    assert n_replaced == 1
    toml_path = tmp_path / "scenario.toml"
    toml_path.write_text(toml_text, encoding="utf-8")

    out_dir = tmp_path / "run"
    status = runner.main(
        [str(toml_path), "--out", str(out_dir), "--frames", "1", "--no-iq", "--no-plots"]
    )
    assert status == 0

    # truth.csv is left out: its time_utc is data-001 §8's later, additive
    # migration, not yet written.
    for name, columns in [
        ("detections.csv", runner.DETECTION_COLUMNS),
        ("tracks.csv", runner.TRACK_COLUMNS),
        ("metrics.csv", runner.METRIC_COLUMNS),
    ]:
        with (out_dir / name).open(newline="", encoding="utf-8") as handle:
            assert tuple(next(csv.reader(handle))) == columns, name
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["epoch_utc"] == "2026-09-03T00:17:56.000000Z"
