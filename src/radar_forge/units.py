r"""Decibels: a power or an amplitude to and from dB, each conversion written once.

A power ratio is :math:`10\log_{10}` of itself and an amplitude ratio (a voltage, a field, a
magnitude) :math:`20\log_{10}`, so a signal's power and its amplitude are the same number of
decibels [1]. MATLAB calls the pairs ``pow2db``/``db2pow`` and ``mag2db``/``db2mag``. A power in
W gives dBW. Zero is ``-inf`` dB and nan stays nan, an absent value; a negative value has no
decibel value and raises.

References
----------
.. [1] IEC 60027-3:2002, *Letter symbols to be used in electrical technology, Part 3:
       Logarithmic and related quantities, and their units*.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["amplitude_to_db", "db_to_amplitude", "db_to_power", "power_to_db"]


def power_to_db(value_linear: ArrayLike) -> NDArray[np.float64]:
    r"""Return a power, or a power ratio, in decibels: :math:`10\log_{10} x` [1]."""
    return _decibels(value_linear, 10.0)


def db_to_power(value_db: ArrayLike) -> NDArray[np.float64]:
    r"""Return the power, or power ratio, of a decibel value: :math:`10^{x/10}` [1]."""
    result: NDArray[np.float64] = 10.0 ** (np.asarray(value_db, dtype=np.float64) / 10.0)
    return result


def amplitude_to_db(value_linear: ArrayLike) -> NDArray[np.float64]:
    r"""Return an amplitude, or an amplitude ratio, in decibels: :math:`20\log_{10} x` [1]."""
    return _decibels(value_linear, 20.0)


def db_to_amplitude(value_db: ArrayLike) -> NDArray[np.float64]:
    r"""Return the amplitude, or amplitude ratio, of a decibel value: :math:`10^{x/20}` [1]."""
    result: NDArray[np.float64] = 10.0 ** (np.asarray(value_db, dtype=np.float64) / 20.0)
    return result


def _decibels(value_linear: ArrayLike, factor: float) -> NDArray[np.float64]:
    """Return ``factor`` times the base-10 logarithm of a non-negative value; zero is -inf."""
    value_arr = np.asarray(value_linear, dtype=np.float64)
    # x < 0 rather than not x >= 0: nan is no estimate, and stays nan.
    if np.any(value_arr < 0.0):
        msg = "value_linear must be non-negative: a negative value has no decibel value."
        raise ValueError(msg)
    with np.errstate(divide="ignore"):  # errstate-exempt: log10(0) is -inf, zero's decibels
        result: NDArray[np.float64] = factor * np.log10(value_arr)
    return result
