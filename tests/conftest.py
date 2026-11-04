"""Suite-wide pytest configuration.

Only one thing lives here: a workaround for a slow collection step in
Hypothesis on Windows. Shared fixtures belong in the conftest.py of the
directory that uses them (``tests/core/tracking/conftest.py``).
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    # At collection, Hypothesis scans every imported module for constants and
    # resolves the path of each one it cannot rule out as third-party. Its cheap
    # test for "third-party" looks for "/site-packages/", which never matches a
    # Windows path ("\site-packages\"), so every module gets a Path.resolve().
    # Where each filesystem call is slow, that was 12 s of every run. Handing it
    # forward slashes restores the cheap test; the modules it scans are the same.
    # If Hypothesis renames the function, this leaves it alone and the suite is
    # only slower.
    try:
        from hypothesis.internal.conjecture import providers as _providers
    except ImportError:  # pragma: no cover - hypothesis is a dev dependency
        _providers = None

    _original = getattr(_providers, "is_local_module_file", None)
    if _original is not None:

        def _is_local_module_file(path: str) -> bool:
            return bool(_original(path.replace("\\", "/")))

        _providers.is_local_module_file = _is_local_module_file
