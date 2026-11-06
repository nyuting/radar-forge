"""Package-wide conventions: the import is light, the API explicit, the docs run.

Four rules from ``CLAUDE.md`` and ``docs/conventions/style.md`` that hold for the whole
library and no module can test of itself:

* ``import radar_forge`` needs no extra: matplotlib, the ``visualisation`` extra, is imported
  only inside the functions that draw (scenario 001's acceptance A8, kept here);
* every module declares ``__all__``, exports no private or repeated name, and every type a
  public callable returns is on its surface; array arguments are ``ArrayLike`` in, float64 out;
* every docstring example runs and prints what it claims, and every export is documented;
* every ``python`` block of ``docs/user-guide/`` runs, and prints the ``text`` block after it
  (a last line ``...`` stands for any further lines).
"""

from __future__ import annotations

import ast
import contextlib
import doctest
import importlib
import inspect
import io
import pathlib
import pkgutil
import re
import subprocess
import sys

import numpy as np
import pytest

import radar_forge
from radar_forge.dsp import window
from radar_forge.signal import (
    bistatic_range_from_beat_frequency_m,
    range_resolution_m,
    sweep_rate_hzps,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
MODULE_NAMES = sorted(
    info.name for info in pkgutil.walk_packages(radar_forge.__path__, prefix="radar_forge.")
)
PUBLIC_MODULE_NAMES = [n for n in MODULE_NAMES if not n.rsplit(".", 1)[-1].startswith("_")]
PACKAGE_NAMES = ["radar_forge.stages", "radar_forge.tracking"]


def test_package_imports_and_has_version():
    assert isinstance(radar_forge.__version__, str)
    assert radar_forge.__version__
    for name in radar_forge.__all__:
        assert hasattr(radar_forge, name)


# Slow: a fresh interpreter pays the whole cold import of numpy and scipy.
@pytest.mark.slow
def test_importing_the_library_does_not_pull_in_matplotlib():
    """Regression: one convenience import of pyplot at module level costs every user the extra."""
    code = (
        "import sys, radar_forge, radar_forge.plots, radar_forge.evaluate; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# Slow: a fresh interpreter, so no earlier import of radar_forge.stages hides a missing one.
@pytest.mark.slow
@pytest.mark.parametrize(
    "first_import", ["radar_forge.evaluate", "radar_forge.stages", "radar_forge.stages.core"]
)
def test_the_core_stages_register_in_their_run_order(first_import: str) -> None:
    """Whichever module is imported first, the five core stages register first, in run order.

    Catches the stages import at the foot of evaluate.py going missing or going lazy, and
    either stages import breaking on the partly initialised module it passes through.
    """
    code = (
        f"import {first_import}; from radar_forge.evaluate import STAGES; "
        "assert list(STAGES)[:5] == ['geometry', 'iq', 'rd', 'detect', 'track'], list(STAGES); "
        "assert list(STAGES).index('keystone') > 4, list(STAGES); "
        "assert list(STAGES).index('leakage') > 4, list(STAGES)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module_name", PUBLIC_MODULE_NAMES)
def test_every_module_declares_its_public_surface(module_name: str) -> None:
    """A module without ``__all__`` exports its imports and its helpers alike."""
    module = importlib.import_module(module_name)
    assert getattr(module, "__all__", None), f"{module_name} declares no __all__"


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_no_name_is_exported_twice_or_private(module_name: str) -> None:
    """A repeated export hides one entry; a leading underscore is the marker a reader skips."""
    names = list(getattr(importlib.import_module(module_name), "__all__", ()))
    assert len(names) == len(set(names)), f"{module_name}.__all__ repeats a name"
    assert [n for n in names if n.startswith("_") and n != "__version__"] == []


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        pytest.param(range_resolution_m, ([1e9, 2e9],), id="range_resolution_m"),
        pytest.param(sweep_rate_hzps, ([1e9, 2e9], [1e-5, 2e-5]), id="sweep_rate_hzps"),
        pytest.param(bistatic_range_from_beat_frequency_m, ([1e6], 1e9, 4e-5), id="from_beat"),
    ],
)
def test_a_python_list_is_accepted_and_a_float64_array_returned(function, arguments):
    """``ArrayLike`` in, ``NDArray[float64]`` out: a caller with a list never converts."""
    result = function(*arguments)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_a_window_is_returned_as_a_fresh_writable_array():
    """Two calls must not share a buffer, or scaling one dwell's window scales every other."""
    first, second = window("hann", 16), window("hann", 16)
    first *= 2.0
    np.testing.assert_allclose(window("hann", 16), second, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_every_submodule_is_exported_and_imported_by_name(package_name: str) -> None:
    """A submodule in ``__all__`` is imported explicitly, not bound by a sibling's side effect."""
    package = importlib.import_module(package_name)
    names = {i.name.rsplit(".", 1)[-1] for i in pkgutil.iter_modules(package.__path__)}
    public = {name for name in names if not name.startswith("_")}
    assert public <= set(package.__all__)
    tree = ast.parse(pathlib.Path(str(package.__file__)).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == package_name
        for alias in node.names
    }
    assert public <= imported


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_every_public_return_type_is_exported(module_name: str) -> None:
    """A class a public callable returns is importable from the same module."""
    module = importlib.import_module(module_name)
    exported = set(getattr(module, "__all__", ()))
    callables = []
    for name in exported:
        attribute = getattr(module, name)
        if inspect.isfunction(attribute):
            callables.append(attribute)
        elif inspect.isclass(attribute):
            callables += [
                m
                for k, m in vars(attribute).items()
                if not k.startswith("_") and inspect.isfunction(m)
            ]
    for function in callables:
        annotation = function.__annotations__.get("return")
        for identifier in re.findall(
            r"[A-Za-z_]\w*", annotation if isinstance(annotation, str) else ""
        ):
            candidate = getattr(module, identifier, None)
            if (
                inspect.isclass(candidate)
                and candidate.__module__ == module_name
                and identifier[0] != "_"
            ):
                assert identifier in exported, f"{module_name} returns {identifier}, not exported"


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_every_docstring_example_evaluates_to_what_it_claims(module_name: str) -> None:
    """A worked example that no longer holds is a documented wrong answer."""
    results = doctest.testmod(importlib.import_module(module_name), verbose=False, report=True)
    assert results.failed == 0, f"{results.failed} failing doctest(s) in {module_name}"


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_every_public_name_is_documented(module_name: str) -> None:
    """Nothing exported reaches a reader without a docstring."""
    module = importlib.import_module(module_name)
    for name in getattr(module, "__all__", ()):
        obj = getattr(module, name)
        if isinstance(obj, str | int | float | tuple | dict) or type(obj).__module__ == "typing":
            continue  # A constant or a Literal alias documents itself by name.
        assert getattr(obj, "__doc__", None), f"{module_name}.{name} has no docstring"


_BLOCK_RE = re.compile(r"^```(\w*)\n(.*?)^```\n", re.MULTILINE | re.DOTALL)


def _guide_examples() -> list[tuple[str, str, str | None]]:
    """Every ``python`` block in the user guide, with the ``text`` block after it."""
    examples = []
    for page in sorted((REPO / "docs" / "user-guide").glob("*.md")):
        text = page.read_text(encoding="utf-8")
        blocks = list(_BLOCK_RE.finditer(text))
        for index, block in enumerate(blocks):
            if block.group(1) != "python":
                continue
            line = text.count("\n", 0, block.start()) + 1
            following = blocks[index + 1] if index + 1 < len(blocks) else None
            expected = following.group(2) if following and following.group(1) == "text" else None
            examples.append((f"{page.name}:{line}", block.group(2), expected))
    return examples


GUIDE_EXAMPLES = _guide_examples()


def test_the_guide_has_examples() -> None:
    assert GUIDE_EXAMPLES


@pytest.mark.parametrize(
    ("source", "expected"),
    [(code, expected) for _, code, expected in GUIDE_EXAMPLES],
    ids=[where for where, _, _ in GUIDE_EXAMPLES],
)
def test_guide_example_prints_what_the_guide_shows(source, expected, monkeypatch) -> None:
    monkeypatch.chdir(REPO)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(source, "<user-guide example>", "exec"), {"__name__": "__main__"})
    if expected is None:
        return
    printed = stdout.getvalue().rstrip("\n").splitlines()
    wanted = expected.rstrip("\n").splitlines()
    if wanted and wanted[-1] == "...":
        wanted = wanted[:-1]
        printed = printed[: len(wanted)]
    assert printed == wanted
