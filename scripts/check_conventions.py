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
R7  Every in-file Markdown anchor (``](#section)``) resolves to a real heading
    in that same file, so a summary table's jump links cannot rot.
R8  Python comments carry no hash banners and no ``Inputs:``/``Outputs:``
    blocks; parameters and returns belong in the NumPy docstring.
R9  Python and Markdown files hold no control characters other than tab, line
    feed and carriage return, such as the BEL a tool writes for ``\\alpha``.
R10 A Python module under ``src/``, ``scripts/`` or ``tools/`` warns past 500
    lines and fails past 800, except the files grandfathered below.
R11 The hot path, ``src/radar_forge/core/``, does no I/O, logging or timing: no
    ``print``, ``open``, ``Path.write_*``, ``logging`` or ``time`` clocks.
R12 ``np.errstate``/``np.seterr`` in ``src/`` carry ``# errstate-exempt: <reason>``.

Each failure prints the rule, the fix, and where the rule is documented. A
warning (R10 between the two limits) is printed and does not block.

Usage
-----
    uv run python scripts/check_conventions.py            # every tracked file
    uv run python scripts/check_conventions.py --staged   # staged files only (hook)
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tokenize
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
    blocking: bool = True

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
    (6_378_137.0, "WGS84_SEMI_MAJOR_AXIS_M"),
    (298.257_223_563, "WGS84_FLATTENING"),
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
# fmt: off
UNIT_SUFFIXES = (
    "_m", "_m2", "_km", "_cm", "_mm", "_s", "_us", "_ns", "_ms", "_hz", "_khz", "_mhz", "_ghz",
    "_rad", "_deg", "_w", "_kw", "_mw", "_dbm", "_dbw", "_db", "_dbi", "_dbsm", "_mps", "_kmh",
    "_k", "_j", "_v", "_bins", "_samples", "_wavelengths", "_linear", "_norm",
)
# fmt: on

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
                        target.lineno,
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
        if is_justified(node.lineno, lines, BROADCAST_EXEMPT):
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


def is_justified(lineno: int, lines: list[str], pattern: re.Pattern[str]) -> bool:
    """Return True if line *lineno* (1-based) or the line above matches *pattern*."""
    window = lines[max(0, lineno - 2) : lineno]
    return any(pattern.search(line) for line in window)


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


# ---------------------------------------------------------------------------
# R7 — Markdown anchors resolve
# ---------------------------------------------------------------------------

ANCHOR_RE = re.compile(r"\]\(#([^)]+)\)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")


def heading_slug(heading: str) -> str:
    """Slugify a Markdown heading the way GitHub does.

    Lowercase, backticks and punctuation dropped, then **each** space becomes a
    hyphen. Runs are not collapsed: GitHub turns ``F1 — Title`` into
    ``f1--title``, because the dash goes and both spaces around it stay. This is
    what a reader's browser resolves, so it is what the rule has to model.
    """
    text = heading.strip().lower().replace("`", "")
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "-")


def heading_slugs(headings: list[str]) -> set[str]:
    """Every anchor GitHub gives a file's headings, in document order.

    A repeated heading gets ``-1``, ``-2`` and so on, so the second ``Notes``
    in a file is reached as ``#notes-1``.
    """
    seen: dict[str, int] = {}
    slugs: set[str] = set()
    for heading in headings:
        slug = heading_slug(heading)
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        slugs.add(slug if count == 0 else f"{slug}-{count}")
    return slugs


def check_markdown_anchors(rel: Path) -> list[Problem]:
    """R7: every ``](#anchor)`` points at a heading in the same file.

    Links inside fenced code blocks and inline code spans are skipped: a document
    that *documents* the convention quotes an example anchor, and an example is
    not a live link.
    """
    abs_path = REPO / rel
    if not abs_path.is_file():
        return []
    lines = abs_path.read_text(encoding="utf-8").splitlines()

    headings: list[str] = []
    in_fence = False
    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and line.startswith("#"):
            headings.append(re.sub(r"^#+\s*", "", line))
    slugs = heading_slugs(headings)

    problems: list[Problem] = []
    in_fence = False
    for number, line in enumerate(lines, start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # An anchor inside `backticks` is being shown, not followed.
        for match in ANCHOR_RE.finditer(INLINE_CODE_RE.sub("", line)):
            anchor = match.group(1)
            if anchor not in slugs:
                problems.append(
                    Problem(
                        rel,
                        number,
                        "R7 anchor",
                        f"link to #{anchor}, which is not a heading in this file.",
                        "point it at a real heading, or fix the heading it lost.",
                    )
                )
    return problems


# --------------------------------------------------------------------------- #
# R8 — comment style
# --------------------------------------------------------------------------- #

# Ten or more leading hashes: a decorative banner. The section header used in
# this file ("# ---- #") starts "# " and so never matches.
BANNER_RE = re.compile(r"^\s*#{10,}")
# An "Inputs:" / "Outputs:" comment block restates what the docstring's
# Parameters and Returns sections already say, and drifts from them.
IO_BLOCK_RE = re.compile(r"^\s*#\s*(Inputs|Outputs):")


def check_comment_style(rel: Path, lines: list[str]) -> list[Problem]:
    """R8: no hash banners and no Inputs/Outputs comment blocks.

    Uses ``tokenize`` so only real comments are inspected: a string or docstring
    that happens to contain ``##########`` is data, not a comment.
    """
    source = "\n".join(lines) + "\n"
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):  # ruff reports broken source
        return []

    problems: list[Problem] = []
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        line = token.start[0]
        if BANNER_RE.match(token.string):
            problems.append(
                Problem(
                    rel,
                    line,
                    "R8 comments",
                    "hash banner comment.",
                    "delete it; if the module needs sections, use a single "
                    "'# ---- #' style header (see this checker's own source).",
                )
            )
        elif IO_BLOCK_RE.match(token.string):
            problems.append(
                Problem(
                    rel,
                    line,
                    "R8 comments",
                    f"'{token.string.strip()}' comment documents parameters or returns.",
                    "move it into the NumPy docstring's Parameters/Returns sections "
                    "(style.md §4); comments say why, not what (style.md §10).",
                )
            )
    return problems


# --------------------------------------------------------------------------- #
# R9 — no control characters
# --------------------------------------------------------------------------- #

# C0 controls and DEL, except tab, line feed and carriage return: a CRLF working
# copy is the editor's business, and git normalises it on commit.
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def check_control_characters(rel: Path, text: str) -> list[Problem]:
    """R9: no control characters in a Python or Markdown file.

    They are invisible in an editor and in a diff, and no linter reports them. The
    usual source is a script that writes LaTeX through a string that reads escapes,
    such as Python in a shell heredoc, so ``\\alpha``, ``\\beta`` and ``\\frac``
    land in the file as BEL, BS and FF.
    """
    problems: list[Problem] = []
    for number, line in enumerate(text.split("\n"), start=1):
        match = CONTROL_RE.search(line)
        if match:
            problems.append(
                Problem(
                    rel,
                    number,
                    "R9 control",
                    f"control character U+{ord(match.group()):04X} at column {match.start() + 1}.",
                    "type the backslash the character replaced (\\a for BEL, \\b for BS, "
                    "\\f for FF), or delete it.",
                )
            )
    return problems


# --------------------------------------------------------------------------- #
# R10 — module length
# --------------------------------------------------------------------------- #

MODULE_WARN_LINES = 500
MODULE_MAX_LINES = 800
LENGTH_SCOPE = {"src", "scripts", "tools"}

# Modules already over MODULE_MAX_LINES when R10 was added. They warn instead of
# failing. The set only shrinks: once a module is split below the limit, delete
# its line (tests/scripts/test_check_conventions.py fails until you do).
LONG_MODULE_GRANDFATHER: frozenset[Path] = frozenset(
    {
        PKG / "core" / "detection.py",
        PKG / "core" / "radar.py",
        PKG / "core" / "tracking" / "kalman.py",
        PKG / "core" / "tracking" / "measurement_models.py",
        PKG / "core" / "tracking" / "tracker.py",
        PKG / "pipelines" / "scenarios.py",
        PKG / "pipelines" / "tracking.py",
        Path("scripts") / "run_scenario.py",
    }
)


def check_module_length(rel: Path, n_lines: int) -> list[Problem]:
    """R10: a long module is a prompt to split it; a very long one must be split."""
    if rel.suffix != ".py" or rel.parts[0] not in LENGTH_SCOPE or n_lines <= MODULE_WARN_LINES:
        return []
    over_max = n_lines > MODULE_MAX_LINES and rel not in LONG_MODULE_GRANDFATHER
    limit = MODULE_MAX_LINES if over_max else MODULE_WARN_LINES
    return [
        Problem(
            rel,
            0,
            "R10 length",
            f"{n_lines} lines, over the {limit}-line limit.",
            "split it along the concepts it holds: one coherent concept per module.",
            blocking=over_max,
        )
    ]


# --------------------------------------------------------------------------- #
# R11 — the hot path does no I/O, logging or timing
# --------------------------------------------------------------------------- #

HOT_PATH = PKG / "core"
# Modules under core/ that broke R11 when it was added. None did.
HOT_PATH_GRANDFATHER: frozenset[Path] = frozenset()
HOT_PATH_BUILTINS = {"print", "open"}
HOT_PATH_CLOCKS = {"time", "time_ns", "perf_counter", "perf_counter_ns", "monotonic"}
HOT_PATH_WRITES = {"write_text", "write_bytes"}


def hot_path_offence(node: ast.AST) -> str | None:
    """Name what *node* does that the hot path may not, or return None."""
    if isinstance(node, ast.Import):
        if any(alias.name.split(".")[0] == "logging" for alias in node.names):
            return "imports logging"
        return None
    if isinstance(node, ast.ImportFrom):
        root = (node.module or "").split(".")[0]
        if root == "logging":
            return "imports logging"
        if root == "time" and any(alias.name in HOT_PATH_CLOCKS for alias in node.names):
            return "imports a clock from time"
        return None
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id in HOT_PATH_BUILTINS:
        return f"calls {func.id}()"
    if isinstance(func, ast.Attribute) and func.attr in HOT_PATH_WRITES:
        return f"calls .{func.attr}()"
    if (
        isinstance(func, ast.Attribute)
        and func.attr in HOT_PATH_CLOCKS
        and isinstance(func.value, ast.Name)
        and func.value.id == "time"
    ):
        return f"calls time.{func.attr}()"
    return None


def check_hot_path(tree: ast.AST, rel: Path) -> list[Problem]:
    """R11: core/ computes; the layers around it log, time and write."""
    if rel.parts[: len(HOT_PATH.parts)] != HOT_PATH.parts or rel in HOT_PATH_GRANDFATHER:
        return []
    problems: list[Problem] = []
    for node in ast.walk(tree):
        offence = hot_path_offence(node)
        if offence is None:
            continue
        problems.append(
            Problem(
                rel,
                getattr(node, "lineno", 0),
                "R11 hot path",
                f"core/ {offence}; the hot path does no I/O, logging or timing.",
                "return the value and let pipelines/ or scripts/ print, log, time or "
                "write it. Tell the user about a recoverable problem with warnings.warn.",
            )
        )
    return problems


# --------------------------------------------------------------------------- #
# R12 — floating-point warnings are not silenced without a reason
# --------------------------------------------------------------------------- #

ERRSTATE_EXEMPT = re.compile(r"#\s*errstate-exempt:\s*\S")
ERRSTATE_FUNCS = {"errstate", "seterr"}
# Files that called np.errstate with no exempt comment when R12 was added. The
# change that added R12 could not edit src/; Phase 2 of the governance plan adds
# `# errstate-exempt: <reason>` to each call and empties this set.
ERRSTATE_GRANDFATHER: frozenset[Path] = frozenset(
    {
        PKG / "core" / "radar.py",
        PKG / "visualisation" / "plotting.py",
    }
)


def check_errstate(tree: ast.AST, rel: Path, lines: list[str]) -> list[Problem]:
    """R12: silencing a floating-point warning hides the next bug along with it."""
    if rel in ERRSTATE_GRANDFATHER:
        return []
    problems: list[Problem] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in ERRSTATE_FUNCS):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id in {"np", "numpy"}):
            continue
        if is_justified(node.lineno, lines, ERRSTATE_EXEMPT):
            continue
        problems.append(
            Problem(
                rel,
                node.lineno,
                "R12 errstate",
                f"np.{func.attr}() silences a floating-point warning with no stated reason.",
                "rearrange the maths, mask the case, or raise ValueError. If the NaN or inf "
                "is the documented answer, scope the narrowest category to one expression "
                "and add  # errstate-exempt: <reason>",
            )
        )
    return problems


def check_file(rel: Path, canonical: set[str]) -> list[Problem]:
    """Run every rule against one repository-relative path."""
    problems = check_path_rules(rel)

    if rel.suffix in {".md", ".py"} and (REPO / rel).is_file():
        problems += check_control_characters(rel, (REPO / rel).read_text(encoding="utf-8"))

    if rel.suffix == ".md":
        problems += check_markdown_anchors(rel)

    # R8 applies to every tracked Python file (src/, scripts/, tests/, tools/);
    # the remaining rules are about the library package only.
    if rel.suffix == ".py" and (REPO / rel).is_file():
        lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
        problems += check_comment_style(rel, lines)
        problems += check_module_length(rel, len(lines))

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
    problems += check_hot_path(tree, rel)
    problems += check_errstate(tree, rel, lines)
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

    for problem in problems:
        if not problem.blocking:
            print(f"  warning [{problem.rule}] {problem.path}: {problem.message}")
    blocking = [problem for problem in problems if problem.blocking]
    if blocking:
        print("\n\033[0;31m\033[1m✗ blocked:\033[0m repository convention violations\n")
        for problem in blocking:
            print(problem.render())
            print()
        return 1
    print("✓ conventions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
