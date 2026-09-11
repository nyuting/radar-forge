"""Smoke test: the package imports and reports a version."""

import radar_forge


def test_package_imports_and_has_version():
    assert isinstance(radar_forge.__version__, str)
    assert radar_forge.__version__
