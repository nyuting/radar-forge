"""Tests for scripts/check_conventions.py, rule R8 (comment style)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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


def test_every_tracked_python_file_passes_r8() -> None:
    problems = [
        problem
        for rel in checker.git_paths(staged=False)
        if rel.suffix == ".py"
        for problem in checker.check_file(rel, set())
        if problem.rule == "R8 comments"
    ]
    assert problems == []
