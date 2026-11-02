"""Shared argument checks for the array functions in ``radar_forge.core``.

References
----------
.. [1] NumPy Developers, "numpy.lib.array_utils.normalize_axis_index", *NumPy Reference*.
"""

from __future__ import annotations


def _normalize_axis(axis: int, ndim: int) -> int:
    """Return ``axis`` as a non-negative index into an ``ndim``-dimensional array.

    Negative axes count from the end, as in NumPy [1]_, so ``-1`` is ``ndim - 1``. The name
    mirrors NumPy's ``normalize_axis_index``, which needs NumPy 2.0.

    Parameters
    ----------
    axis : int
        Axis index, in ``-ndim .. ndim - 1``.
    ndim : int
        Number of dimensions of the array.

    Returns
    -------
    int
        ``axis`` in ``0 .. ndim - 1``.

    Raises
    ------
    ValueError
        If ``axis`` is out of bounds for ``ndim`` dimensions.

    References
    ----------
    .. [1] NumPy Developers, "numpy.lib.array_utils.normalize_axis_index", *NumPy Reference*.
    """
    if not -ndim <= axis < ndim:
        msg = f"axis {axis} is out of bounds for an array with {ndim} dimension(s)."
        raise ValueError(msg)
    return axis % ndim
