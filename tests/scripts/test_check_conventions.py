"""Tests for the R7 heading slugs in scripts/check_conventions.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parent.parent.parent


def load_checker() -> ModuleType:
    """Import scripts/check_conventions.py, which is a script and not a package module."""
    spec = importlib.util.spec_from_file_location(
        "check_conventions", ROOT / "scripts" / "check_conventions.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load_checker()


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
