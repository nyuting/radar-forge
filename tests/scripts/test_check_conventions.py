"""Tests for scripts/check_conventions.py: at least one per live rule, R1, R2 and R4-R12."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_conventions.py"


def _load_script() -> ModuleType:
    # scripts/ is developer tooling, not an installed package, so it is loaded by
    # path rather than imported.
    spec = importlib.util.spec_from_file_location("check_conventions", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_script()
LIBRARY_FILE = Path("src/radar_forge/x.py")


def _rules(problems: list) -> list[str]:
    return [problem.rule for problem in problems]


@pytest.mark.parametrize("name", ["setup.py", "requirements.txt", "tox.ini"])
def test_a_root_config_file_outside_pyproject_is_flagged(name: str) -> None:
    """Catches R1 letting a tool's own config file split the configuration."""
    assert _rules(checker.check_path_rules(Path(name))) == ["R1 toml"]


def test_a_nested_file_with_a_config_name_is_not_flagged() -> None:
    """Catches R1 rejecting a fixture that only shares a name with a config file."""
    assert checker.check_path_rules(Path("tests/data/setup.py")) == []


@pytest.mark.parametrize("name", ["plot.png", "model.pkl", "demo.ipynb"])
def test_a_non_python_file_in_the_package_is_flagged(name: str) -> None:
    """Catches R2 letting an image, pickle or notebook into the import package."""
    assert _rules(checker.check_path_rules(Path("src/radar_forge") / name)) == ["R2 layout"]


def test_package_data_is_not_flagged() -> None:
    """Catches R2 rejecting ``py.typed``, which marks the package as typed."""
    assert checker.check_path_rules(Path("src/radar_forge/py.typed")) == []


CANONICAL_CONSTANTS = {"SPEED_OF_LIGHT_MPS", "BOLTZMANN_JPK"}


@pytest.mark.parametrize(
    "line",
    ["delay_s = 2 * range_m / 3e8", "SPEED_OF_LIGHT_MPS = 299_792_458.0", "k = 1.38e-23"],
)
def test_a_constant_literal_or_rebinding_is_flagged(line: str) -> None:
    """Catches R4 missing an approximate c, a second definition, or a rounded Boltzmann."""
    problems = checker.check_constants(ast.parse(line), LIBRARY_FILE, CANONICAL_CONSTANTS)
    # A rebinding to the exact value is two findings: the name and the literal.
    assert set(_rules(problems)) == {"R4 constants"}


def test_the_constants_module_may_define_constants() -> None:
    """Catches R4 rejecting the one module the constants belong in."""
    tree = ast.parse("SPEED_OF_LIGHT_MPS = 299_792_458.0")
    assert checker.check_constants(tree, checker.CONSTANTS_MODULE, CANONICAL_CONSTANTS) == []


def _r5_rules(parameter: str) -> list[str]:
    tree = ast.parse(f"def f({parameter}: float) -> float:\n    return {parameter}\n")
    return _rules(checker.check_units(tree, LIBRARY_FILE))


@pytest.mark.parametrize("parameter", ["range", "gain_tx", "snr"])
def test_a_parameter_with_no_unit_is_flagged(parameter: str) -> None:
    """Catches R5 passing a qualified bare name such as ``gain_tx`` (linear or dB?)."""
    assert _r5_rules(parameter) == ["R5 units"]


@pytest.mark.parametrize("parameter", ["range_m", "gain_tx_dbi", "snr_linear", "n_pulses"])
def test_a_parameter_with_a_unit_or_a_count_is_not_flagged(parameter: str) -> None:
    """Catches R5 rejecting a suffixed quantity or an ``n_`` count."""
    assert _r5_rules(parameter) == []


def _r6_rules(source: str) -> list[str]:
    return _rules(checker.check_broadcast(ast.parse(source), LIBRARY_FILE, source.splitlines()))


@pytest.mark.parametrize("call", ["np.tile(a, 3)", "np.repeat(a, 3, axis=0)"])
def test_tile_and_repeat_are_flagged(call: str) -> None:
    """Catches R6 letting a materialised copy stand in for broadcasting."""
    assert _r6_rules(f"b = {call}\n") == ["R6 broadcast"]


def test_broadcast_to_is_not_flagged() -> None:
    """Catches R6 flagging ``np.broadcast_to``, which returns a view: the fix, not the fault."""
    assert _r6_rules("b = np.broadcast_to(a, (4, 3))\n") == []


def test_a_broadcast_exempt_comment_satisfies_r6() -> None:
    source = "# broadcast-exempt: a scatter needs one marker per point\nb = np.repeat(a, 3)\n"
    assert _r6_rules(source) == []


def test_a_dead_in_file_anchor_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches R7 passing a jump link whose heading was renamed, or checking a quoted example."""
    text = "# Title\n\nSee [ok](#title) and [gone](#old-heading).\n\n```\n[x](#quoted)\n```\n"
    (tmp_path / "doc.md").write_text(text, encoding="utf-8")
    monkeypatch.setattr(checker, "REPO", tmp_path)
    problems = checker.check_markdown_anchors(Path("doc.md"))
    assert [(problem.rule, problem.line) for problem in problems] == [("R7 anchor", 3)]


def _r8_lines(source: str) -> list[int]:
    problems = checker.check_comment_style(Path("src/radar_forge/x.py"), source.splitlines())
    assert all(problem.rule == "R8 comments" for problem in problems)
    return [problem.line for problem in problems]


def test_hash_banner_comment_is_flagged() -> None:
    source = "x = 1\n##########################\ny = 2\n"
    assert _r8_lines(source) == [2]


def test_indented_hash_banner_comment_is_flagged() -> None:
    source = "def f() -> None:\n    ############\n    pass\n"
    assert _r8_lines(source) == [2]


def test_inputs_comment_is_flagged() -> None:
    source = "def f(range_m: float) -> float:\n    # Inputs: range_m\n    return range_m\n"
    assert _r8_lines(source) == [2]


def test_outputs_comment_is_flagged() -> None:
    source = "def f(range_m: float) -> float:\n    #Outputs: the range\n    return range_m\n"
    assert _r8_lines(source) == [2]


def test_dashed_section_header_is_not_flagged() -> None:
    header = "# " + "-" * 75 + " #"
    source = f"{header}\n# R1 — configuration lives in TOML\n{header}\nx = 1\n"
    assert _r8_lines(source) == []


def test_ordinary_comment_is_not_flagged() -> None:
    source = "# asarray rather than array: callers pass an ndarray in a hot loop.\nx = 1\n"
    assert _r8_lines(source) == []


def test_hashes_inside_a_string_are_not_flagged() -> None:
    source = 'LINE = "##########"\nDOC = """\n# Inputs: not a comment\n"""\n'
    assert _r8_lines(source) == []


@pytest.mark.slow  # over a second: parses every tracked file
def test_every_tracked_python_file_passes_r8() -> None:
    problems = [
        problem
        for rel in checker.git_paths(staged=False)
        if rel.suffix == ".py" and (REPO_ROOT / rel).is_file()
        for problem in checker.check_comment_style(
            rel, (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()
        )
    ]
    assert problems == []


def _r9_lines(text: str) -> list[int]:
    problems = checker.check_control_characters(Path("src/radar_forge/x.py"), text)
    assert all(problem.rule == "R9 control" for problem in problems)
    return [problem.line for problem in problems]


def test_a_bel_written_in_place_of_backslash_alpha_is_flagged() -> None:
    """Catches the BEL that lands in a file when a tool writes ``\\alpha`` through an escape."""
    # The source as cd568eb found it: "\a" here is the BEL byte itself.
    source = 'def f() -> None:\n    r"""Weight W0 = \alpha^2."""\n'
    assert _r9_lines(source) == [2]


@pytest.mark.parametrize("character", ["\b", "\f", "\x00", "\x1b", "\x7f"])
def test_other_control_characters_are_flagged(character: str) -> None:
    """Catches a pattern that stops at BEL and lets backspace, form feed or ESC through."""
    assert _r9_lines(f"x = 1\ny = '{character}'\n") == [2]


def test_tab_and_crlf_are_not_flagged() -> None:
    """Catches R9 rejecting a CRLF working copy, or a tab in a Markdown code block."""
    assert _r9_lines("x = 1\r\n\ty = 2\r\n") == []


@pytest.mark.parametrize(
    ("heading", "slug"),
    [
        # Each space becomes a hyphen, so the two around a dropped dash give two hyphens.
        ("F1 — `FrameResult` was returned", "f1--frameresult-was-returned"),
        ("13. Migration from today's code", "13-migration-from-todays-code"),
    ],
)
def test_a_heading_slugs_the_way_github_does(heading: str, slug: str) -> None:
    """Catches a slug that collapses space runs, so R7 passes links GitHub cannot follow."""
    assert checker.heading_slug(heading) == slug


def test_a_repeated_heading_is_reached_with_a_numbered_suffix() -> None:
    """Catches R7 rejecting ``#notes-1``, GitHub's anchor for a file's second ``Notes``."""
    assert checker.heading_slugs(["Notes", "Notes", "Notes"]) == {"notes", "notes-1", "notes-2"}


LIBRARY_MODULE = Path("src/radar_forge/pipeline.py")
HOT_PATH_MODULE = Path("src/radar_forge/x.py")


def _blocking(problems: list) -> list[bool]:
    return [problem.blocking for problem in problems]


@pytest.mark.parametrize(
    ("n_lines", "expected"),
    [(500, []), (501, [False]), (800, [False]), (801, [True])],
)
def test_module_length_warns_past_500_and_fails_past_800(n_lines: int, expected: list) -> None:
    """Catches an off-by-one at either limit, or a warning that blocks the commit."""
    assert _blocking(checker.check_module_length(LIBRARY_MODULE, n_lines)) == expected


def test_a_grandfathered_module_only_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    rel = Path("src/radar_forge/x.py")
    monkeypatch.setattr(checker, "LONG_MODULE_GRANDFATHER", frozenset({rel}))
    assert _blocking(checker.check_module_length(rel, 5_000)) == [False]


def test_tests_and_markdown_have_no_length_limit() -> None:
    assert checker.check_module_length(Path("tests/tracking/test_x.py"), 5_000) == []
    assert checker.check_module_length(Path("src/radar_forge/x.md"), 5_000) == []


def test_every_grandfathered_module_still_needs_it() -> None:
    """Catches a split module left in the set, so the set can only shrink."""
    for rel in checker.LONG_MODULE_GRANDFATHER:
        n_lines = len((REPO_ROOT / rel).read_text(encoding="utf-8").splitlines())
        assert n_lines > checker.MODULE_MAX_LINES, f"{rel} is {n_lines} lines: drop it"


def _r11_lines(source: str, rel: Path = HOT_PATH_MODULE) -> list[int]:
    problems = checker.check_hot_path(ast.parse(source), rel)
    assert all(problem.rule == "R11 hot path" for problem in problems)
    return [problem.line for problem in problems]


@pytest.mark.parametrize(
    "line",
    [
        "print(x)",
        "import logging",
        "import logging.handlers",
        "from logging import getLogger",
        "t = time.perf_counter()",
        "t = time.time()",
        "from time import perf_counter",
        "f = open(path)",
        "path.write_text(x)",
        "Path(p).write_bytes(x)",
    ],
)
def test_the_hot_path_rejects_io_logging_and_timing(line: str) -> None:
    assert _r11_lines(f"x = 1\n{line}\n") == [2]


def test_the_hot_path_allows_warnings_and_non_clock_time_names() -> None:
    """Catches R11 rejecting warnings.warn, which style.md asks for, or any `.time` attribute."""
    source = "import warnings\nwarnings.warn(msg, stacklevel=2)\nt = state.time_s\nimport time\n"
    assert _r11_lines(source) == []


@pytest.mark.parametrize("name", ["scenario.py", "pipeline.py", "evaluate.py", "plots.py"])
def test_r11_spares_the_modules_that_run_a_scenario(name: str) -> None:
    assert _r11_lines("print(x)\n", Path("src/radar_forge") / name) == []


def test_r11_covers_the_tracking_package_and_spares_scripts() -> None:
    assert _r11_lines("print(x)\n", Path("src/radar_forge/tracking/x.py")) == [1]
    assert _r11_lines("print(x)\n", Path("scripts/x.py")) == []


def _r12_lines(source: str, rel: Path = LIBRARY_MODULE) -> list[int]:
    problems = checker.check_errstate(ast.parse(source), rel, source.splitlines())
    assert all(problem.rule == "R12 errstate" for problem in problems)
    return [problem.line for problem in problems]


@pytest.mark.parametrize(
    "line",
    ['with np.errstate(divide="ignore"):\n    y = 1 / x', 'np.seterr(all="ignore")'],
)
def test_silencing_floating_point_warnings_is_flagged(line: str) -> None:
    assert _r12_lines(f"x = 1\n{line}\n") == [2]


@pytest.mark.parametrize("on_line_above", [True, False])
def test_an_errstate_exempt_comment_satisfies_r12(on_line_above: bool) -> None:
    reason = "# errstate-exempt: forward scatter gives an infinite resolution"
    call = 'with np.errstate(divide="ignore"):'
    source = f"{reason}\n{call}\n" if on_line_above else f"{call}  {reason}\n"
    assert _r12_lines(source + "    y = 1 / x\n") == []


def test_an_empty_exempt_comment_does_not_satisfy_r12() -> None:
    assert _r12_lines('with np.errstate(all="ignore"):  # errstate-exempt:\n    y = 1\n') == [1]


def test_every_errstate_grandfathered_file_still_needs_it() -> None:
    """Catches a file whose calls were justified but which was left in the set."""
    for rel in checker.ERRSTATE_GRANDFATHER:
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert checker.check_errstate(tree, Path("src/x.py"), source.splitlines()), rel
