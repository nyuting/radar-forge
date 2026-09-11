#!/usr/bin/env python3
"""Enforce repository-layout conventions that linters cannot express.

Pure standard library and Python 3.9+ compatible on purpose: this runs from a
git hook before `uv sync` has necessarily happened, and from CI.

Rules enforced
--------------
1. Configuration lives in TOML.  `pyproject.toml` is the only config file; no
   setup.py / setup.cfg / .flake8 / tox.ini / standalone ruff or mypy configs.
2. The library is Python under a src layout.  Everything imported as
   ``radar_forge`` lives in ``src/radar_forge/`` and is a ``.py`` file.
3. Every library module carries a module docstring.

Usage
-----
    python3 scripts/check_conventions.py            # whole repo
    python3 scripts/check_conventions.py --staged   # staged files only (hook)
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BANNED_CONFIG = {
    "setup.py": "declare the build in pyproject.toml [build-system] / [project]",
    "setup.cfg": "move the settings into pyproject.toml",
    ".flake8": "linting is ruff, configured in [tool.ruff]",
    "tox.ini": "use the Makefile targets and the CI matrix",
    ".ruff.toml": "ruff config belongs in pyproject.toml [tool.ruff]",
    "ruff.toml": "ruff config belongs in pyproject.toml [tool.ruff]",
    "mypy.ini": "mypy config belongs in pyproject.toml [tool.mypy]",
    ".mypy.ini": "mypy config belongs in pyproject.toml [tool.mypy]",
    "pytest.ini": "pytest config belongs in pyproject.toml [tool.pytest.ini_options]",
    "requirements.txt": "dependencies are declared in pyproject.toml [project]",
    "requirements-dev.txt": "dev deps go in [project.optional-dependencies].dev",
}

# Non-.py files that are legitimately part of the package payload.
ALLOWED_PACKAGE_SUFFIXES = {".py", ".pyi", ".typed", ".json", ".toml", ".csv"}


def staged_paths() -> list[Path]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    ).stdout
    return [Path(line) for line in out.splitlines() if line.strip()]


def tracked_paths() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True, cwd=REPO
    ).stdout
    return [Path(line) for line in out.splitlines() if line.strip()]


def check(paths: list[Path]) -> list[str]:
    errors: list[str] = []

    for rel in paths:
        name = rel.name
        parts = rel.parts

        # Rule 1 — config in TOML.
        if name in BANNED_CONFIG and len(parts) == 1:
            errors.append(
                f"{rel}: configuration must live in pyproject.toml — {BANNED_CONFIG[name]}."
            )

        if parts[:2] != ("src", "radar_forge"):
            continue

        # Rule 2 — the library is Python.
        if rel.suffix not in ALLOWED_PACKAGE_SUFFIXES:
            errors.append(
                f"{rel}: only {', '.join(sorted(ALLOWED_PACKAGE_SUFFIXES))} files belong "
                f"in the import package; data and notebooks live elsewhere."
            )
            continue

        if rel.suffix != ".py":
            continue

        # Rule 3 — module docstrings.
        abs_path = REPO / rel
        if not abs_path.is_file():
            continue
        try:
            tree = ast.parse(abs_path.read_text(encoding="utf-8"), filename=str(rel))
        except SyntaxError as exc:  # ruff reports this far better than we can
            errors.append(f"{rel}: syntax error at line {exc.lineno}.")
            continue
        if ast.get_docstring(tree) is None:
            errors.append(
                f"{rel}: missing a module docstring — say what the module models "
                f"and cite the reference it implements (docs/conventions/style.md)."
            )

    return errors


def main() -> int:
    paths = staged_paths() if "--staged" in sys.argv[1:] else tracked_paths()
    errors = check(paths)
    if errors:
        print("\n\033[0;31m\033[1m✗ blocked:\033[0m repository convention violations\n")
        for err in errors:
            print(f"  • {err}")
        print("\n  See docs/conventions/style.md\n")
        return 1
    print("✓ conventions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
