"""Shared checks for the vectors and covariances the tracker passes around.

A covariance is a square table that says how uncertain each value is (on its diagonal) and how
the errors of two values move together (off the diagonal). A valid covariance is symmetric and
positive semidefinite: no direction has a negative variance.

Every check here allows for float64 roundoff and nothing more. The allowance is the same
everywhere: ``_ROUNDOFF_TOLERANCE`` times the largest absolute element of the matrix.

References
----------
.. [1] N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002,
       ch. 10 (Cholesky factorisation and its roundoff behaviour).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# The roundoff a covariance may carry, relative to its largest element. A covariance built from
# a few matrix products of size n carries roundoff of about n times float64 epsilon (2.2e-16),
# so about 1e-15 for the state sizes here. 1e-10 leaves a margin of 1e5 over that, and is still
# far below any error a wrong model would cause.
_ROUNDOFF_TOLERANCE = 1e-10

# Two timestamps closer than this, in seconds, are treated as the same time. The same scan
# time can be computed two ways, such as k * dt against a running sum of dt, and the two then
# differ by float roundoff. Even after 100,000 additions that roundoff stays well under a
# microsecond, while a scan interval is milliseconds or more. A target at 300 m/s moves 0.3 mm
# in a microsecond, so treating the two times as equal costs nothing.
TIMESTAMP_TOLERANCE_S = 1e-6

# Fractions of the roundoff allowance tried, in turn, when a Cholesky factorisation fails. The
# largest is the allowance itself, so the retry never hides more than roundoff.
_JITTER_STEPS = (0.0, 1e-2, 1e-1, 1.0)


def as_vector(value: ArrayLike, dimension: int, name: str = "vector") -> NDArray[np.float64]:
    """Copy a vector and check that it is finite and has the expected length.

    Parameters
    ----------
    value : array_like
        The numbers to check, a list or an array of shape ``(dimension,)``.
    dimension : int
        The number of values required. Must be positive.
    name : str, optional
        What the values are, used in the error message.

    Returns
    -------
    numpy.ndarray
        An independent float64 copy, shape ``(dimension,)``.

    Raises
    ------
    ValueError
        If the shape is not ``(dimension,)`` or any value is NaN or infinite.

    Examples
    --------
    >>> as_vector([1, 2], 2)
    array([1., 2.])
    """
    result = np.array(value, dtype=np.float64, copy=True)
    if dimension < 1 or result.shape != (dimension,) or not np.all(np.isfinite(result)):
        msg = f"{name} must be finite with shape ({dimension},); got shape {result.shape}"
        raise ValueError(msg)
    return result


def as_points(
    value: ArrayLike, n_points: int, dimension: int, name: str = "points"
) -> NDArray[np.float64]:
    """Check a batch of vectors, such as sigma points, in one pass.

    Parameters
    ----------
    value : array_like
        The batch, shape ``(n_points, dimension)``: one vector per row.
    n_points : int
        The number of rows required.
    dimension : int
        The number of values in each row.
    name : str, optional
        What the values are, used in the error message.

    Returns
    -------
    numpy.ndarray
        The batch as float64, shape ``(n_points, dimension)``. It is not copied if it already
        is a float64 array.

    Raises
    ------
    ValueError
        If the shape is wrong or any value is NaN or infinite.

    Examples
    --------
    >>> as_points([[1, 2], [3, 4]], 2, 2).shape
    (2, 2)
    """
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (n_points, dimension) or not np.all(np.isfinite(result)):
        msg = (
            f"{name} must be finite with shape ({n_points}, {dimension}); got shape {result.shape}"
        )
        raise ValueError(msg)
    return result


def _roundoff_allowance(matrix: NDArray[np.float64]) -> float:
    """Return the roundoff a matrix may carry, in the units of its elements."""
    # The largest absolute element, not the largest diagonal element, so that the symmetry
    # check means something before we know the matrix is a covariance. For a real covariance
    # the two are the same, because no element can exceed the larger of its two variances.
    return _ROUNDOFF_TOLERANCE * float(np.max(np.abs(matrix)))


def _check_square(matrix: NDArray[np.float64], dimension: int, name: str) -> None:
    """Raise unless ``matrix`` is a finite, symmetric ``(dimension, dimension)`` array."""
    if dimension < 1 or matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        msg = (
            f"{name} must be finite with shape ({dimension}, {dimension}); got shape {matrix.shape}"
        )
        raise ValueError(msg)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=_roundoff_allowance(matrix)):
        msg = f"{name} must be symmetric to within roundoff"
        raise ValueError(msg)


def _is_semidefinite(matrix: NDArray[np.float64]) -> bool:
    """Return whether the smallest eigenvalue is above minus the roundoff allowance."""
    allowance = _roundoff_allowance(matrix)
    # Adding the allowance to the diagonal raises every eigenvalue by that amount. If the
    # factorisation then succeeds, no eigenvalue was below minus the allowance. A Cholesky
    # factorisation costs a fraction of a full eigendecomposition, and almost every covariance
    # passes this way.
    try:
        np.linalg.cholesky(matrix + allowance * np.eye(len(matrix)))
    except np.linalg.LinAlgError:
        # The quick test can fail on a matrix that is exactly singular but valid, such as an
        # all-zero process noise. Only then do we pay for the eigenvalues.
        return bool(np.linalg.eigvalsh(matrix)[0] >= -allowance)
    return True


def as_covariance(value: ArrayLike, dimension: int) -> NDArray[np.float64]:
    """Copy a covariance and check that it is symmetric and positive semidefinite.

    Parameters
    ----------
    value : array_like
        The covariance, shape ``(dimension, dimension)``. Each element has the product of the
        units of its row and its column, so a range variance is in m².
    dimension : int
        The number of values the covariance describes. Must be positive.

    Returns
    -------
    numpy.ndarray
        An independent float64 copy, shape ``(dimension, dimension)``, made exactly symmetric.

    Raises
    ------
    ValueError
        If the shape is wrong, an element is not finite, the matrix is not symmetric, or it has
        an eigenvalue below zero, each by more than roundoff.

    Notes
    -----
    The roundoff allowance is ``1e-10`` times the largest absolute element. It scales with the
    matrix, so the same covariance passes or fails whatever units it is written in.

    Examples
    --------
    >>> as_covariance([[4.0, 1.0], [1.0, 2.0]], 2)
    array([[4., 1.],
           [1., 2.]])
    """
    result = np.array(value, dtype=np.float64, copy=True)
    _check_square(result, dimension, "covariance")
    result = (result + result.T) / 2
    if not _is_semidefinite(result):
        msg = "covariance must be positive semidefinite; it has a negative eigenvalue"
        raise ValueError(msg)
    return result


def cholesky_factor(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the lower-triangular Cholesky factor L of a covariance, so that L Lᵀ = matrix.

    The UKF uses L to place its sigma points, and the innovation statistics use it to whiten a
    residual.

    Parameters
    ----------
    matrix : numpy.ndarray
        The covariance, shape ``(n, n)``.

    Returns
    -------
    numpy.ndarray
        The factor L, float64, shape ``(n, n)``.

    Raises
    ------
    ValueError
        If the matrix is not finite, square and symmetric, or is singular or not positive
        definite by more than roundoff.

    Notes
    -----
    A covariance that is positive definite in exact arithmetic can fail to factor because of
    roundoff. If it does, a small multiple of the identity is added to the diagonal and the
    factorisation is tried again. The amount grows in steps up to the same roundoff allowance
    that :func:`as_covariance` uses, ``1e-10`` times the largest absolute element. An
    all-zero matrix is rejected, because that allowance is then zero too.

    Examples
    --------
    >>> cholesky_factor(np.array([[4.0, 0.0], [0.0, 9.0]]))
    array([[2., 0.],
           [0., 3.]])
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    _check_square(matrix, matrix.shape[0] if matrix.ndim else 0, "covariance")
    allowance = _roundoff_allowance(matrix)
    identity = np.eye(len(matrix))
    # A loop, not an array operation: each attempt depends on whether the one before failed.
    for step in _JITTER_STEPS:
        try:
            return np.asarray(
                np.linalg.cholesky(matrix + step * allowance * identity), dtype=np.float64
            )
        except np.linalg.LinAlgError:
            pass
    msg = "covariance is singular or not positive definite, beyond roundoff"
    raise ValueError(msg)
