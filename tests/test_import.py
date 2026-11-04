"""Smoke test: the package imports, reports a version, and needs no extras.

The zero-extras guarantee is spec/structure.md B.2 rule 2 and is easy to break
by accident: one convenience import of radar_forge.visualisation at the top of a
package __init__ makes `import radar_forge` fail for anyone who installed the
library without matplotlib. It is checked in a subprocess because by the time
this module runs, the test session has usually imported visualisation already.
"""

import subprocess
import sys

import pytest

import radar_forge


def test_package_imports_and_has_version():
    assert isinstance(radar_forge.__version__, str)
    assert radar_forge.__version__


def test_the_public_api_is_what_it_re_exports():
    for name in radar_forge.__all__:
        assert hasattr(radar_forge, name)


@pytest.mark.slow
def test_importing_the_package_does_not_pull_in_the_visualisation_extra():
    """visualisation needs matplotlib, so importing it at top level costs everyone.

    The same fresh interpreter also checks that ``import radar_forge`` binds
    ``pipelines``: scenario-001 A8 and spec/structure.md B.2 rule 2 both ask
    for the pipelines without extras, and a second subprocess cost 0.7 s for
    that one attribute.

    Slow: a fresh interpreter pays the whole cold import of numpy and scipy.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            # pipelines is reached first, so it too is held to the no-visualisation rule.
            "import sys, radar_forge; "
            "radar_forge.pipelines.load_scenario; "
            "assert 'radar_forge.visualisation' not in sys.modules, 'visualisation imported'; "
            "assert 'matplotlib' not in sys.modules, 'matplotlib imported'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
