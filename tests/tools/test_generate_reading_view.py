"""Tests for tools/generate_reading_view.py, the minimised reading view generator."""

from __future__ import annotations

import ast
import html
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "generate_reading_view.py"
PACKAGE_ROOT = REPO_ROOT / "src" / "radar_forge"


def _load_tool() -> ModuleType:
    # tools/ is developer tooling, not an installed package, so it is loaded by
    # path rather than imported.
    spec = importlib.util.spec_from_file_location("generate_reading_view", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


SAMPLE = '''"""Module docstring that must not survive."""

from __future__ import annotations

import numpy as np

__all__ = ["scaled_range_m"]

_GAIN_LINEAR: float = 2.0


class Fix:
    """Class docstring that must not survive."""

    range_m: float
    label: str = "fix"


def scaled_range_m(range_m: float, weight_linear: float = 1.0) -> float:
    """Scale a range.

    Returns
    -------
    float
    """
    if range_m <= 0.0:
        msg = "range_m must be strictly positive."
        raise ValueError(msg)
    # An inline comment that must not survive.
    scaled = np.asarray(range_m) * weight_linear * _GAIN_LINEAR
    return float(scaled)
'''


@pytest.fixture(scope="module")
def minimised_sample() -> str:
    return tool.minimise_source(SAMPLE, filename="sample.py")


@pytest.fixture(scope="module")
def sample_tree(minimised_sample: str) -> ast.Module:
    return ast.parse(minimised_sample)


def test_sample_output_is_valid_python(sample_tree: ast.Module) -> None:
    assert isinstance(sample_tree, ast.Module)


def test_docstrings_are_stripped(sample_tree: ast.Module) -> None:
    for node in ast.walk(sample_tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
            assert ast.get_docstring(node) is None


def test_comments_are_stripped(minimised_sample: str) -> None:
    assert "#" not in minimised_sample


def test_annotations_are_stripped(sample_tree: ast.Module) -> None:
    assert not [n for n in ast.walk(sample_tree) if isinstance(n, ast.AnnAssign)]
    for node in ast.walk(sample_tree):
        if isinstance(node, ast.FunctionDef):
            assert node.returns is None
            assert all(arg.annotation is None for arg in node.args.args)


def test_guard_block_is_stripped(minimised_sample: str) -> None:
    assert "ValueError" not in minimised_sample
    assert "must be strictly positive" not in minimised_sample


def test_imports_signatures_and_returns_survive(
    minimised_sample: str, sample_tree: ast.Module
) -> None:
    assert "import numpy as np" in minimised_sample
    functions = [n for n in ast.walk(sample_tree) if isinstance(n, ast.FunctionDef)]
    assert [f.name for f in functions] == ["scaled_range_m"]
    assert [a.arg for a in functions[0].args.args] == ["range_m", "weight_linear"]
    assert functions[0].args.defaults, "the default value must survive"
    assert any(isinstance(n, ast.Return) for n in ast.walk(functions[0]))


def test_maths_and_broadcasting_survive(minimised_sample: str) -> None:
    assert "np.asarray(range_m) * weight_linear * _GAIN_LINEAR" in minimised_sample


def test_annotation_only_field_keeps_its_name(minimised_sample: str) -> None:
    # A dataclass-style declaration becomes `range_m = ...` so the field stays visible.
    assert "range_m = ..." in minimised_sample


def test_non_guard_raise_survives() -> None:
    source = (
        "def f(x):\n"
        "    if x:\n"
        "        raise ValueError('bad')\n"
        "    else:\n"
        "        return 1\n"
        "    return 2\n"
    )
    assert "ValueError" in tool.minimise_source(source)


def test_emptied_body_gets_a_pass() -> None:
    assert "pass" in tool.minimise_source('def f(x):\n    """Only a docstring."""\n')


def test_generate_runs_clean_over_the_package(tmp_path: Path) -> None:
    out = tmp_path / "radar_forge_reading"
    written = tool.generate(PACKAGE_ROOT, out)

    sources = sorted(p.relative_to(PACKAGE_ROOT) for p in PACKAGE_ROOT.rglob("*.py"))
    generated = sorted(p.relative_to(out) for p in out.rglob("*.py"))
    assert generated == sources, "the output must mirror the package tree"
    assert (out / "index.html") in written


def test_generated_package_is_valid_stripped_python(tmp_path: Path) -> None:
    out = tmp_path / "radar_forge_reading"
    tool.generate(PACKAGE_ROOT, out)

    for path in sorted(out.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
                assert ast.get_docstring(node) is None, path
            if isinstance(node, ast.AnnAssign):
                pytest.fail(f"annotation survived in {path}:{node.lineno}")
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                assert node.returns is None, path
                assert all(a.annotation is None for a in node.args.args), path
            if isinstance(node, ast.If) and not node.orelse and node.body:
                assert not isinstance(node.body[-1], ast.Raise), f"guard survived in {path}"


def test_generated_package_keeps_imports_and_returns(tmp_path: Path) -> None:
    out = tmp_path / "radar_forge_reading"
    tool.generate(PACKAGE_ROOT, out)

    def counts(root: Path) -> tuple[int, int]:
        imports = returns = 0
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                imports += isinstance(node, ast.Import | ast.ImportFrom)
                returns += isinstance(node, ast.Return)
        return imports, returns

    source_imports, source_returns = counts(PACKAGE_ROOT)
    reading_imports, reading_returns = counts(out)
    assert reading_imports == source_imports
    # Returns inside a stripped guard block do not exist, so none may be lost.
    assert reading_returns == source_returns


def test_index_links_every_generated_file(tmp_path: Path) -> None:
    out = tmp_path / "radar_forge_reading"
    tool.generate(PACKAGE_ROOT, out)

    index = (out / "index.html").read_text(encoding="utf-8")
    for path in sorted(out.rglob("*.py")):
        assert f'href="{path.relative_to(out).as_posix()}.html"' in index


def test_every_module_has_a_page_showing_its_code(tmp_path: Path) -> None:
    out = tmp_path / "radar_forge_reading"
    tool.generate(PACKAGE_ROOT, out)

    for path in sorted(out.rglob("*.py")):
        page = path.with_name(path.name + ".html").read_text(encoding="utf-8")
        body = path.read_text(encoding="utf-8").removeprefix(tool._HEADER)
        assert html.escape(body) in page, path


def test_module_page_escapes_html_and_links_home() -> None:
    page = tool.render_module("core/signal.py", "x = a < b\n")
    assert "x = a &lt; b" in page
    assert 'href="../index.html"' in page


def test_main_writes_the_tree(tmp_path: Path) -> None:
    out = tmp_path / "view"
    assert tool.main(["--src", str(PACKAGE_ROOT), "--out", str(out)]) == 0
    assert (out / "index.html").is_file()


def test_main_rejects_a_missing_source(tmp_path: Path) -> None:
    assert tool.main(["--src", str(tmp_path / "nope"), "--out", str(tmp_path / "out")]) == 1
