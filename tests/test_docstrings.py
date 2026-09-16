"""Package-wide docstring integrity: the examples run, and the prose exists.

Docstring examples are the first thing a student copies and the last thing a
change breaks, because `make check` does not run them. This module makes them
part of the gate.

Regression: ``core.tracking.predict`` shipped an example asserting that a
100 m target closing at 5 m/s reaches 105 m after one second. The code was
right -- ``range_1d`` carries a range rate that is positive *closing*, so its
transition is ``[[1, -T], [0, 1]]`` and the answer is 95 m -- and the *example*
was wrong, which is the worse of the two failures for a teaching library: the
narrative contradicted the sign convention the rest of the tracker depends on.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pytest

import radar_forge


def _module_names() -> list[str]:
    """Every importable module in the package, teaching extras excluded."""
    names = []
    for info in pkgutil.walk_packages(radar_forge.__path__, prefix="radar_forge."):
        # teaching/ needs matplotlib, which is an optional extra; importing it
        # here would turn a missing extra into a failure of the core suite.
        if info.name.startswith("radar_forge.teaching"):
            continue
        names.append(info.name)
    return sorted(names)


MODULE_NAMES = _module_names()


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_every_docstring_example_evaluates_to_what_it_claims(module_name: str) -> None:
    """Every ``>>>`` in the library produces the output written beside it.

    A worked example that no longer holds is a documented wrong answer. Run
    under `doctest` rather than by eye, because the failures are arithmetic
    and sign-convention errors, not typos.
    """
    module = importlib.import_module(module_name)
    results = doctest.testmod(module, verbose=False, report=True)
    assert results.failed == 0, f"{results.failed} failing doctest(s) in {module_name}"


def test_the_prediction_example_respects_the_closing_sign_convention() -> None:
    """Regression: the ``predict`` example must close range, not open it.

    Pins the property the broken example denied, in the module that owns the
    convention, so that a future edit to the docstring cannot quietly restore
    the opening-range reading.
    """
    source = doctest.DocTestFinder().find(radar_forge.core.tracking.predict)
    examples = [example for test in source for example in test.examples]
    assert examples, "predict must keep a worked example"
    assert not any("105.0" in example.source for example in examples)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_every_public_name_is_documented(module_name: str) -> None:
    """Nothing exported by a module reaches a reader without a docstring.

    `ruff`'s pydocstyle rules cover definitions in `src/`; this covers the
    *export*, which is what a reader meets first, including names re-exported
    into a package ``__init__``.
    """
    module = importlib.import_module(module_name)
    for name in getattr(module, "__all__", ()):
        obj = getattr(module, name)
        if isinstance(obj, str | int | float | tuple):
            continue  # A constant's documentation is its name and its unit.
        if type(obj).__module__ == "typing":
            continue  # A Literal alias carries no runtime __doc__ to inspect.
        assert getattr(obj, "__doc__", None), f"{module_name}.{name} has no docstring"
