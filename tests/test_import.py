"""Smoke test: the package imports, reports a version, and needs no extras.

The zero-extras guarantee is spec/structure.md B.3 rule 2 and is easy to break
by accident: one convenience import of radar_forge.teaching at the top of a
package __init__ makes `import radar_forge` fail for anyone who installed the
library without matplotlib. It is checked in a subprocess because by the time
this module runs, the test session has usually imported teaching already.
"""

import subprocess
import sys

import radar_forge


def test_package_imports_and_has_version():
    assert isinstance(radar_forge.__version__, str)
    assert radar_forge.__version__


def test_the_public_api_is_what_it_re_exports():
    for name in radar_forge.__all__:
        assert hasattr(radar_forge, name)


def test_importing_the_package_does_not_pull_in_the_teaching_extra():
    """teaching needs matplotlib, so importing it at top level costs everyone."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, radar_forge; "
            "assert 'radar_forge.teaching' not in sys.modules, 'teaching imported'; "
            "assert 'matplotlib' not in sys.modules, 'matplotlib imported'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_the_pipelines_subpackage_is_available_without_extras():
    result = subprocess.run(
        [sys.executable, "-c", "import radar_forge; radar_forge.pipelines.load_scenario"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
