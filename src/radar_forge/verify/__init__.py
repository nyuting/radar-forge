"""The stepper and its stages: a scenario run one step at a time, each step checked.

:func:`run` is the entry point: it runs the stages a scenario's ``[verify]`` table names and
raises :class:`VerificationError` naming every check that failed. :mod:`.stepper` holds the run,
the context the stages share and the registry. Each stage module registers its stages on
import, and the import order below is the run order, since the registry is an append-only dict:
``chain`` holds the five stages every scenario runs (``geometry``, ``iq``, ``rd``, ``detect``,
``track``); ``keystone`` (scenario 004), ``leakage`` (005), ``multitarget`` (011) and ``doa``
(006) one each. A later scenario adds ``verify/<stage>.py``, with one import line and one
``__all__`` entry here.

A stage module imports ``from radar_forge.verify import stepper`` and calls the shared helpers
qualified, ``stepper.stage``, ``stepper.truth(ctx)``, ``stepper.periods(...)``: stages bind
locals named ``truth`` and ``periods``, which would shadow a bare import and raise
``UnboundLocalError``.
"""

from radar_forge.verify import stepper
from radar_forge.verify.stepper import STAGES, Check, Report, VerificationError, run

# isort: split
from radar_forge.verify import chain, keystone

# isort: split
from radar_forge.verify import leakage

# isort: split
from radar_forge.verify import multitarget

# isort: split
from radar_forge.verify import doa

__all__ = [
    "STAGES",
    "Check",
    "Report",
    "VerificationError",
    "chain",
    "doa",
    "keystone",
    "leakage",
    "multitarget",
    "run",
    "stepper",
]
