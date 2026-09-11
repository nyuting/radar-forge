#!/usr/bin/env python3
"""Enforce the conventions that ruff and mypy cannot express.

Pure standard library on purpose: this runs from a git hook, possibly before
``uv sync`` has happened, and from CI.

Rules
-----
R1  Configuration lives in TOML (``pyproject.toml`` only).
R2  The import package is Python under ``src/radar_forge/``.
R3  Every library module has a module docstring.
R4  Physical constants are defined once, in ``core/constants.py``. No other file
    may bind those names, and their literal values may not appear elsewhere.
R5  Physical quantities carry a unit suffix; bare names like ``range`` or
    ``power`` are rejected.
R6  ``np.tile``/``np.repeat`` used to fake broadcasting must carry a one-line
    justification.

Each failure prints the rule, the fix, and where the rule is documented.

Usage
-----
    python3 scripts/check_conventions.py            # every tracked file
    python3 scripts/check_conventions.py --staged   # staged files only (hook)
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent
PKG = Path("src") / "radar_forge"
CONSTANTS_MODULE = PKG / "core" / "constants.py"
STYLE = "docs/conventions/style.md"


class Problem(NamedTuple):
    """One convention violation, with the rule that caught it and the fix."""

    path: Path
    line: int
    rule: str
    message: str
    fix: str
    doc: str = STYLE

    def render(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else str(self.path)
        return (
            f"  • {loc}\n"
            f"      [{self.rule}] {self.message}\n"
            f"      fix: {self.fix}\n"
            f"      see: {self.doc}"
        )


# --------------------------------------------------------------------------- #
# R1 — configuration lives in TOML
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
# R2 — the import package is Python
# --------------------------------------------------------------------------- #

ALLOWED_PACKAGE_SUFFIXES = {".py", ".pyi", ".typed", ".json", ".toml", ".csv"}

# --------------------------------------------------------------------------- #
# R4 — constants are defined once
# --------------------------------------------------------------------------- #

# Literal values that must only ever appear in core/constants.py, mapped to the
# canonical name to import instead. Matched on the normalised numeric token, so
# 3e8, 3.0e8 and 300000000.0 are all caught.
# Literal values that must only ever appear in core/constants.py, paired with
# the canonical name to import instead. Written as pairs rather than a dict
# because several spellings collapse to the same float (6_371_000.0 is 6.371e6),
# and the near-miss approximations are the ones worth catching.
CONSTANT_LITERALS: tuple[tuple[float, str], ...] = (
    (299_792_458.0, "SPEED_OF_LIGHT_MPS"),
    (3e8, "SPEED_OF_LIGHT_MPS"),
    (2.998e8, "SPEED_OF_LIGHT_MPS"),
    (2.997e8, "SPEED_OF_LIGHT_MPS"),
    (1.380_649e-23, "BOLTZMANN_JPK"),
    (1.38e-23, "BOLTZMANN_JPK"),
    (1.381e-23, "BOLTZMANN_JPK"),
    (6_371_000.0, "EARTH_RADIUS_M"),
    (6_378_137.0, "EARTH_RADIUS_M"),
)


def canonical_constant_names() -> set[str]:
    """Return the names exported by core/constants.py, read from its ``__all__``."""
    source = REPO / CONSTANTS_MODULE
    if not source.is_file():
        return set()
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return {
                elt.value
                for elt in getattr(node.value, "elts", [])
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
    return set()


# --------------------------------------------------------------------------- #
# R5 — unit suffixes
# --------------------------------------------------------------------------- #

# Bare physical-quantity names, mapped to the suffixes that would make them
# unambiguous. The point is that `gain` does not say linear-or-dB and `range`
# does not say metres-or-bins; both mistakes are silent.
BARE_UNIT_NAMES: dict[str, str] = {
    "range": "range_m (or range_bins)",
    "ranges": "ranges_m",
    "distance": "distance_m",
    "altitude": "altitude_m",
    "height": "height_m",
    "spacing": "spacing_m (or spacing_wavelengths)",
    "wavelength": "wavelength_m",
    "power": "power_w (or power_dbm)",
    # Dimensionless ratios are the worst offenders: nothing in the value says
    # whether 30 means 30x or 30 dB, so the suffix must say it.
    "gain": "gain_linear (or gain_dbi)",
    "loss": "loss_linear (or loss_db)",
    "snr": "snr_linear (or snr_db)",
    "efficiency": "efficiency_linear",
    "ratio": "ratio_linear (or ratio_db)",
    "rcs": "rcs_m2 (or rcs_dbsm)",
    "freq": "freq_hz",
    "frequency": "frequency_hz",
    "bandwidth": "bandwidth_hz",
    "doppler": "doppler_hz (or doppler_mps)",
    "time": "time_s",
    "duration": "duration_s",
    "delay": "delay_s",
    "pri": "pri_s",
    "prf": "prf_hz",
    "angle": "angle_rad (or angle_deg)",
    "azimuth": "azimuth_rad",
    "elevation": "elevation_rad",
    "theta": "theta_rad",
    "phi": "phi_rad",
    "beamwidth": "beamwidth_rad",
    "velocity": "velocity_mps",
    "speed": "speed_mps",
    "temperature": "temperature_k",
    "temp": "temperature_k",
    "phase": "phase_rad",
    "amplitude": "amplitude_v (or amplitude, normalised)",
}

# A name is considered unit-bearing if it ends in one of these.
UNIT_SUFFIXES = (
    "_m",
    "_m2",
    "_km",
    "_cm",
    "_mm",
    "_s",
    "_us",
    "_ns",
    "_ms",
    "_hz",
    "_khz",
    "_mhz",
    "_ghz",
    "_rad",
    "_deg",
    "_w",
    "_kw",
    "_mw",
    "_dbm",
    "_dbw",
    "_db",
    "_dbi",
    "_dbsm",
    "_mps",
    "_kmh",
    "_k",
    "_j",
    "_v",
    "_bins",
    "_samples",
    "_wavelengths",
    "_linear",
    "_norm",
)

# Names that look bare but are dimensionless, a count, or an index.
UNIT_EXEMPT = {"n", "i", "j", "k", "idx", "axis", "out", "dtype", "rng", "seed", "self", "cls"}

# --------------------------------------------------------------------------- #
# R6 — broadcasting
# --------------------------------------------------------------------------- #

BROADCAST_EXEMPT = re.compile(r"#\s*broadcast-exempt:\s*\S")
TILE_FUNCS = {"tile", "repeat", "broadcast_to"}


def has_unit_suffix(name: str) -> bool:
    """Return True if *name* ends in a recognised unit suffix."""
    lowered = name.lower().rstrip("_")
    return any(lowered.endswith(suffix) for suffix in UNIT_SUFFIXES)


def bare_stem(name: str) -> str | None:
    """Return the physical-quantity stem of *name* if it lacks a unit suffix.

    Matches the leading token so that qualifiers do not launder a bare name:
    ``gain_tx`` is as ambiguous as ``gain`` (30 or 30 dB?), while ``gain_tx_dbi``
    and ``range_m`` are both fine.
    """
    lowered = name.lower().strip("_")
    if not lowered or lowered in UNIT_EXEMPT or lowered.startswith("n_"):
        return None
    if has_unit_suffix(lowered):
        return None
    stem = lowered.split("_", 1)[0]
    return stem if stem in BARE_UNIT_NAMES else None


def check_units(tree: ast.AST, rel: Path) -> list[Problem]:
    """R5: reject bare physical-quantity names in function signatures."""
    problems: list[Problem] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if args.vararg:
            every.append(args.vararg)
        if args.kwarg:
            every.append(args.kwarg)
        for arg in every:
            stem = bare_stem(arg.arg)
            if stem is None:
                continue
            problems.append(
                Problem(
                    rel,
                    arg.lineno,
                    "R5 units",
                    f"parameter '{arg.arg}' of {node.name}() is a physical quantity "
                    f"with no unit in its name.",
                    f"add the unit — for '{stem}' that is {BARE_UNIT_NAMES[stem]}. "
                    f"A bare name hides a dB/linear or metres/bins mix-up, and that "
                    f"fails silently: the array still has the right shape.",
                )
            )
    return problems


def check_constants(tree: ast.AST, rel: Path, canonical: set[str]) -> list[Problem]:
    """R4: constants are bound and valued only in core/constants.py."""
    if rel == CONSTANTS_MODULE:
        return []
    problems: list[Problem] = []

    for node in ast.walk(tree):
        # Rebinding a canonical constant name anywhere else.
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            name = None
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            if name and name in canonical:
                problems.append(
                    Problem(
                        rel,
                        node.lineno,
                        "R4 constants",
                        f"'{name}' is defined in {CONSTANTS_MODULE.as_posix()}; "
                        f"assigning it here creates a second, divergent definition.",
                        f"import it instead: from radar_forge.core.constants import {name}",
                    )
                )

        # Hardcoding a constant's literal value.
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = float(node.value)
            for literal, canonical_name in CONSTANT_LITERALS:
                if value != literal:
                    continue
                fix = f"import it: from radar_forge.core.constants import {canonical_name}"
                if canonical_name == "SPEED_OF_LIGHT_MPS" and value != 299_792_458.0:
                    fix += (
                        " — and note this approximation is 0.07% high, 70 cm of range error at 1 km"
                    )
                problems.append(
                    Problem(
                        rel,
                        node.lineno,
                        "R4 constants",
                        f"the literal {node.value!r} is a physical constant.",
                        fix,
                    )
                )
                break
    return problems


def check_broadcast(tree: ast.AST, rel: Path, lines: list[str]) -> list[Problem]:
    """R6: np.tile/np.repeat needs a justification, since broadcasting is free."""
    problems: list[Problem] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in TILE_FUNCS):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id in {"np", "numpy"}):
            continue
        # An explanation on the call line or the line above satisfies the rule.
        window = lines[max(0, node.lineno - 2) : node.lineno]
        if any(BROADCAST_EXEMPT.search(line) for line in window):
            continue
        problems.append(
            Problem(
                rel,
                node.lineno,
                "R6 broadcast",
                f"np.{func.attr}() materialises a copy that broadcasting usually "
                f"makes unnecessary.",
                "rewrite using broadcasting (e.g. arr[:, None] * other), or keep it "
                "and justify with a comment:  # broadcast-exempt: <reason>",
            )
        )
    return problems


def check_path_rules(rel: Path) -> list[Problem]:
    """R1 and R2: file placement and configuration format."""
    problems: list[Problem] = []
    if rel.name in BANNED_CONFIG and len(rel.parts) == 1:
        problems.append(
            Problem(
                rel,
                0,
                "R1 toml",
                "configuration must live in pyproject.toml.",
                BANNED_CONFIG[rel.name],
            )
        )
    if rel.parts[:2] == PKG.parts and rel.suffix not in ALLOWED_PACKAGE_SUFFIXES:
        problems.append(
            Problem(
                rel,
                0,
                "R2 layout",
                f"'{rel.suffix}' files do not belong in the import package.",
                "notebooks go in docs/, data in tests/data/golden/, scripts in scripts/.",
            )
        )
    return problems


def check_file(rel: Path, canonical: set[str]) -> list[Problem]:
    """Run every rule against one repository-relative path."""
    problems = check_path_rules(rel)

    if rel.parts[:2] != PKG.parts or rel.suffix != ".py":
        return problems

    abs_path = REPO / rel
    if not abs_path.is_file():
        return problems
    source = abs_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(rel))
    except SyntaxError as exc:  # ruff reports these far better than we can
        return [*problems, Problem(rel, exc.lineno or 0, "syntax", str(exc.msg), "fix the syntax")]

    if ast.get_docstring(tree) is None:
        problems.append(
            Problem(
                rel,
                1,
                "R3 docstring",
                "module has no docstring.",
                "say what the module models and cite the reference it implements.",
            )
        )

    lines = source.splitlines()
    problems += check_units(tree, rel)
    problems += check_constants(tree, rel, canonical)
    problems += check_broadcast(tree, rel, lines)
    return problems


def git_paths(staged: bool) -> list[Path]:
    """Return staged or tracked repository-relative paths."""
    cmd = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
        if staged
        else ["git", "ls-files"]
    )
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO).stdout
    return [Path(line) for line in out.splitlines() if line.strip()]


def main() -> int:
    """Check every relevant file and report violations."""
    canonical = canonical_constant_names()
    problems: list[Problem] = []
    for rel in git_paths(staged="--staged" in sys.argv[1:]):
        problems += check_file(rel, canonical)

    if problems:
        print("\n\033[0;31m\033[1m✗ blocked:\033[0m repository convention violations\n")
        for problem in problems:
            print(problem.render())
            print()
        return 1
    print("✓ conventions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
