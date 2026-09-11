"""Physical constants — the single source of truth for the whole library.

Every physical constant used anywhere in ``radar_forge`` is defined here, once,
and imported from here. Nothing else defines one, and nothing hardcodes a
literal for one.

Why this is a rule
------------------
A second definition of the speed of light is not a style problem, it is a
correctness problem: two modules silently disagreeing in the ninth digit
produces range errors that look like a calibration issue and take days to find.
Two mechanisms keep that from happening:

1. Every constant is annotated :data:`typing.Final`, so ``mypy`` rejects any
   reassignment — including ``radar_forge.core.constants.SPEED_OF_LIGHT_MPS = 3e8``
   from another module.
2. ``scripts/check_conventions.py`` (run by the pre-commit and pre-push hooks)
   rejects any *other* file that assigns one of these names, and rejects the
   literal values appearing anywhere outside this module.

Names follow the unit-suffix convention in ``docs/conventions/style.md``.

References
----------
.. [1] CODATA 2018 recommended values, https://physics.nist.gov/cuu/Constants/
.. [2] IEEE Std 686-2017, *IEEE Standard Radar Definitions*.
.. [3] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §1.4, §2.3.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "BOLTZMANN_JPK",
    "EARTH_RADIUS_M",
    "FOUR_THIRDS_EARTH_RADIUS_M",
    "SPEED_OF_LIGHT_MPS",
    "STANDARD_NOISE_TEMPERATURE_K",
]

# --------------------------------------------------------------------------- #
# Exact, by SI definition
# --------------------------------------------------------------------------- #

SPEED_OF_LIGHT_MPS: Final[float] = 299_792_458.0
"""Speed of light in vacuum, m/s. Exact by the 2019 SI definition of the metre.

Use this, never 3e8. The 0.07% error in 3e8 is 70 cm of range error at 1 km —
larger than the range resolution of a 1 GHz-bandwidth automotive radar.
"""

BOLTZMANN_JPK: Final[float] = 1.380_649e-23
"""Boltzmann constant, J/K. Exact by the 2019 SI definition of the kelvin.

Thermal noise power in a bandwidth B is ``BOLTZMANN_JPK * T * B`` watts.
"""

# --------------------------------------------------------------------------- #
# Radar conventions
# --------------------------------------------------------------------------- #

STANDARD_NOISE_TEMPERATURE_K: Final[float] = 290.0
"""Reference noise temperature :math:`T_0`, K.

The IEEE convention against which noise figure is defined [2]_. It is a
convention, not a measurement: do not substitute the actual ambient
temperature unless you are computing system noise temperature rather than
noise figure.
"""

EARTH_RADIUS_M: Final[float] = 6_371_000.0
"""Mean Earth radius, m (IUGG mean radius :math:`R_1`)."""

FOUR_THIRDS_EARTH_RADIUS_M: Final[float] = 4.0 / 3.0 * EARTH_RADIUS_M
"""Effective Earth radius for radar horizon calculations, m.

The standard 4/3 approximation models atmospheric refraction bending rays
towards the Earth, so the radar horizon sits further out than the geometric
one [3]_. Valid for a standard atmosphere only; it does not model ducting.
"""
