"""The stepper's stages; each registers itself with :mod:`radar_forge.evaluate` on import.

The import order below is the run order, since the registry is an append-only dict. ``core``
holds the five stages every scenario runs. A later scenario adds ``stages/<library module>.py``
(``stages/migration.py``, ``stages/clutter.py``, ``stages/tracking.py``, ...), with one import
line and one ``__all__`` entry here.
"""

from radar_forge.stages import core

__all__ = ["core"]
