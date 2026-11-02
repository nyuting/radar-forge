"""Tests for scripts/check_conventions.py: R7 slugs, R8 comment style, R9 control characters."""

from __future__ import annotations

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
