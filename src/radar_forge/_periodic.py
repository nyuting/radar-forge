r"""Folding a value into one period: private, the one wrap behind every periodic coordinate.

:func:`radar_forge.dsp.fold_into_period` is the public form, which checks the period. The
tracker's state layout, the clustering of circular columns, the azimuth of
:func:`radar_forge.coordinates.enu_to_range_azimuth_elevation_rad` and DOA's direction cosines
call :func:`fold` directly. It lives here, not in ``dsp``, so that ``tracking/`` need not import
the signal-processing chain (``spec/tracker-001.md`` §2).

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["fold"]


def fold(value: ArrayLike, period: ArrayLike, *, lower: ArrayLike) -> NDArray[np.float64]:
    """Return ``value`` folded into ``[lower, lower + period)`` (Richards [1]_ §5.5.4).

    The period is not checked. ``lower = -period / 2`` gives the short way round, ``lower = 0``
    a compass azimuth.
    """
    value_arr = np.asarray(value, dtype=np.float64)
    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = lower_arr + np.asarray(period, dtype=np.float64)
    wrapped = np.mod(value_arr - lower_arr, period) + lower_arr
    # np.mod of a tiny negative offset rounds up to the divisor itself, which would
    # land on the open top edge; that point is the bottom of the next fold.
    result: NDArray[np.float64] = np.where(wrapped >= upper_arr, lower_arr, wrapped)
    return result
