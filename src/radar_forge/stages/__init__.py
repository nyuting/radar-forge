"""The stepper's stages; each registers itself with :mod:`radar_forge.evaluate` on import.

The import order below is the run order, since the registry is an append-only dict. ``core``
holds the five stages every scenario runs. A later scenario adds ``stages/<library module>.py``
(``stages/migration.py``, ``stages/clutter.py``, ``stages/tracking.py``, ...), with one import
line and one ``__all__`` entry here.

A stage module imports ``from radar_forge import evaluate`` and calls the shared helpers
qualified, ``evaluate.stage``, ``evaluate.truth(ctx)``, ``evaluate.spans(...)``: stages bind
locals named ``truth`` and ``spans``, which would shadow a bare import and raise
``UnboundLocalError``.
"""

from radar_forge.stages import core, tracking

__all__ = ["core", "tracking"]
