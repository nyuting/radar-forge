"""Package-wide API contracts: explicit exports, and array_like in / ndarray out.

Two rules in `CLAUDE.md` and `docs/conventions/style.md` are stated for the
whole library but were only ever tested in one module:

* every module declares ``__all__``, so a reader can tell the surface from the
  scaffolding (`spec/refactor-001` §6.1);
* array arguments are ``ArrayLike`` in and ``NDArray`` out, so a caller may
  pass a Python list and always gets a float64 (or complex128) array back.

The second is a type-safety property that `mypy --strict` cannot check, because
``ArrayLike`` accepts a list at the type level and the failure is at runtime --
a function that forgets ``np.asarray`` type-checks perfectly and then raises on
the first list it is handed, or worse, silently returns a list.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import pkgutil
import re

import numpy as np
import pytest

import radar_forge
from radar_forge.core import (
    dbsm_to_m2,
    m2_to_dbsm,
    range_from_beat_frequency_m,
    range_resolution_m,
    sweep_rate_hzps,
    taper,
)


def _module_names() -> list[str]:
    """Every importable module in the package, teaching extras excluded."""
    return sorted(
        info.name
        for info in pkgutil.walk_packages(radar_forge.__path__, prefix="radar_forge.")
        if not info.name.startswith("radar_forge.teaching")
    )


MODULE_NAMES = _module_names()


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_every_module_declares_its_public_surface(module_name: str) -> None:
    """A module without ``__all__`` exports its imports and its helpers alike."""
    module = importlib.import_module(module_name)
    assert hasattr(module, "__all__"), f"{module_name} declares no __all__"
    assert module.__all__, f"{module_name}.__all__ is empty"


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_every_exported_name_exists(module_name: str) -> None:
    """``__all__`` is a promise; a stale entry breaks ``from module import *``."""
    module = importlib.import_module(module_name)
    for name in getattr(module, "__all__", ()):
        assert hasattr(module, name), f"{module_name}.__all__ names a missing {name!r}"


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_no_name_is_exported_twice(module_name: str) -> None:
    """A repeated export is a merge artefact and hides one of the two entries.

    Ordering is left to `ruff`'s RUF022, which sorts constants, classes and
    functions into separate runs; asserting plain alphabetical order here would
    contradict the linter.
    """
    names = list(getattr(importlib.import_module(module_name), "__all__", ()))
    assert len(names) == len(set(names)), f"{module_name}.__all__ repeats a name"


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_no_private_name_is_exported(module_name: str) -> None:
    """A leading underscore is the marker a reader skips; honour it."""
    module = importlib.import_module(module_name)
    leaked = [name for name in getattr(module, "__all__", ()) if name.startswith("_")]
    assert leaked == [] or leaked == ["__version__"], f"{module_name} exports {leaked}"


# One representative array-taking function per contract shape: a single
# array_like argument, two of them broadcast against each other, and a
# constructor of a fresh array. The point is the *convention*, not the physics,
# which each module's own tests already cover.
ARRAY_LIKE_CASES = [
    pytest.param(range_resolution_m, ([1e9, 2e9],), id="range_resolution_m"),
    pytest.param(sweep_rate_hzps, ([1e9, 2e9], [1e-5, 2e-5]), id="sweep_rate_hzps"),
    pytest.param(range_from_beat_frequency_m, ([1e6], 1e9, 4e-5), id="range_from_beat"),
    pytest.param(dbsm_to_m2, ([0.0, 10.0],), id="dbsm_to_m2"),
    pytest.param(m2_to_dbsm, ([1.0, 10.0],), id="m2_to_dbsm"),
]


@pytest.mark.parametrize(("function", "arguments"), ARRAY_LIKE_CASES)
def test_a_python_list_is_accepted_and_a_float64_array_returned(
    function: object,
    arguments: tuple[object, ...],
) -> None:
    """``array_like`` in, ``NDArray[float64]`` out, for every documented signature.

    Pins the calling convention itself: a caller who has a list, a tuple or a
    0-d value never has to convert, and every caller downstream may rely on
    ndarray semantics (``.shape``, broadcasting, float64 precision) on the way
    out.
    """
    result = function(*arguments)  # type: ignore[operator]
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


@pytest.mark.parametrize(("function", "arguments"), ARRAY_LIKE_CASES)
def test_a_list_and_an_array_give_bit_identical_results(
    function: object,
    arguments: tuple[object, ...],
) -> None:
    """Conversion at the boundary must not change the arithmetic.

    ``np.asarray`` of a list of Python floats is exactly the float64 array the
    caller would have built, so the two paths agree to the last bit -- an
    equality that is legitimate here precisely because nothing has been
    recomputed.
    """
    as_lists = function(*arguments)  # type: ignore[operator]
    as_arrays = function(*(np.asarray(a, dtype=np.float64) for a in arguments))  # type: ignore[operator]
    np.testing.assert_array_equal(as_lists, as_arrays)


def test_a_scalar_argument_returns_a_zero_dimensional_float64() -> None:
    """The scalar case is the 0-d corner of the same broadcasting contract.

    NumPy collapses a 0-d result to a ``np.float64`` scalar, which still carries
    ``.ndim``, ``.shape`` and float64 precision -- so a caller may index or
    broadcast the result of a scalar call exactly as for an array one. What must
    never happen is a bare Python ``float``, which would lose the dtype guarantee.
    """
    result = range_resolution_m(1e9)
    assert isinstance(result, np.floating)
    assert result.dtype == np.float64
    assert result.ndim == 0


def test_a_taper_is_returned_as_a_fresh_writable_array() -> None:
    """A window is data the caller owns; a shared or read-only buffer aliases.

    Two calls must not return the same object, or scaling one dwell's window in
    place would silently scale every other dwell that asked for the same taper.
    """
    first = taper("hann", 16)
    second = taper("hann", 16)
    assert first is not second
    first *= 2.0
    np.testing.assert_allclose(taper("hann", 16), second, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# A package's submodule surface is explicit, not accidental
# ---------------------------------------------------------------------------

PACKAGE_NAMES = ["radar_forge.core", "radar_forge.pipelines"]


def _submodule_files(package_name: str) -> set[str]:
    """Every ``.py`` submodule of a package, by bare name."""
    package = importlib.import_module(package_name)
    return {
        info.name.rsplit(".", 1)[-1]
        for info in pkgutil.iter_modules(package.__path__, prefix=f"{package_name}.")
    }


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_every_submodule_is_exported(package_name: str) -> None:
    """Every submodule that exists is named in the package's ``__all__``.

    A submodule left out of ``__all__`` is still reachable -- importing one name
    from it binds it on the parent package as a side effect -- so the omission is
    invisible until someone reorders the imports. Pinning the set makes the
    surface a decision rather than an accident.
    """
    package = importlib.import_module(package_name)
    exported = set(package.__all__)
    missing = _submodule_files(package_name) - exported
    assert not missing, f"{package_name} submodules exist but are not exported: {sorted(missing)}"


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_every_exported_submodule_is_imported_explicitly(package_name: str) -> None:
    """A submodule in ``__all__`` is imported by name, not bound by side effect.

    ``from radar_forge.core.radar_equation import received_power_w`` binds
    ``radar_equation`` on the parent package as a side effect. A submodule listed
    in ``__all__`` but never imported therefore resolves -- until the sibling
    import that was carrying it is removed or reordered, at which point
    ``from ... import *`` raises ``AttributeError`` on a name the module claims
    to export. Requiring the explicit import removes that dependence on order.
    """
    source = importlib.import_module(package_name).__file__
    assert source is not None
    text = pathlib.Path(source).read_text(encoding="utf-8")
    tree = ast.parse(text)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == package_name:
            imported.update(alias.name for alias in node.names)

    exported_submodules = _submodule_files(package_name) & set(
        importlib.import_module(package_name).__all__
    )
    missing = exported_submodules - imported
    assert not missing, (
        f"{package_name} exports {sorted(missing)} but never imports them by name -- "
        "they resolve only as a side effect of a sibling from-import."
    )


# ---------------------------------------------------------------------------
# A public return type is part of the public surface
# ---------------------------------------------------------------------------


def _public_callables(module: object) -> list[tuple[str, object]]:
    """Public functions, plus the public methods of exported classes."""
    found: list[tuple[str, object]] = []
    for name in getattr(module, "__all__", ()):
        attribute = getattr(module, name)
        if inspect.isfunction(attribute):
            found.append((name, attribute))
        elif inspect.isclass(attribute):
            found.extend(
                (f"{name}.{method_name}", method)
                for method_name, method in vars(attribute).items()
                if not method_name.startswith("_") and inspect.isfunction(method)
            )
    return found


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_every_public_return_type_is_exported(module_name: str) -> None:
    """A type a public callable returns must be importable from the same module.

    A function may return a class the module never exports -- the annotation
    reads fine, ``mypy`` is satisfied, and every existing test constructs the
    value rather than naming its type. The caller who wants to annotate a
    variable, or to write ``isinstance``, then finds the name missing from
    ``__all__`` and reaches into the module's internals for it. The returned
    type is part of the contract, so it belongs on the surface.
    """
    module = importlib.import_module(module_name)
    exported = set(getattr(module, "__all__", ()))
    for label, callable_object in _public_callables(module):
        annotation = callable_object.__annotations__.get("return")
        if not isinstance(annotation, str):
            continue
        for identifier in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", annotation):
            candidate = getattr(module, identifier, None)
            if (
                inspect.isclass(candidate)
                and getattr(candidate, "__module__", None) == module_name
                and not identifier.startswith("_")
            ):
                assert identifier in exported, (
                    f"{module_name}.{label} returns {identifier!r}, which "
                    f"{module_name}.__all__ does not export"
                )
