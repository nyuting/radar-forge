"""Shared float64 validation and roundoff policy.

Adapted from unified-extensible-tracker; see spec/tracker-001-provenance.md.
References: Bar-Shalom et al., Estimation with Applications to Tracking, 2001.


References
----------
.. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
.. [2] Adapted local tracker; spec/tracker-001-provenance.md.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "FloatArray",
    "cholesky",
    "covariance",
    "timestamp",
    "vector",
]

FloatArray = NDArray[np.float64]


###################################################################################################
# This function copies and checks a list of numbers before the tracker uses it. 
#                                                                               
# Inputs:                                                                       
# - value (ArrayLike): List or NumPy array containing the numbers to check.
# - dimension (int): Required number of values.
# - name (str): Label used if the error message needs to identify these values.
#
# Outputs:
# - float64 NumPy array: Independent copy, shape (dimension,), with no missing or 
#   infinite values.
def vector(value: ArrayLike, dimension: int, name: str = "vector") -> FloatArray:
    """Copy a finite vector with the required dimension.

    Parameters
    ----------
    value : array_like
        Input vector, shape (dimension,).
    dimension : int
        Positive number of coordinates.
    name : str, optional
        Label used in validation errors.

    Returns
    -------
    ndarray
        Independent float64 vector, shape (dimension,).

    References
    ----------
    .. [1] Tracker 001 numerical contracts, spec/tracker-001-integration.md.
    """
    result = np.array(value, dtype=np.float64, copy=True)
    if dimension < 1 or result.shape != (dimension,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape ({dimension},)")
    return result
###################################################################################################

###################################################################################################
# This function checks and copies a covariance matrix before a filter uses it.
#
# Inputs:
# - value (ArrayLike): Square covariance matrix (describes uncertainty and how errors vary together).
# - dimension (int): Number of tracked or measured quantities.
#
# Outputs:
# - float64 NumPy array: Checked, symmetric covariance matrix, shape (dimension, dimension).
def covariance(value: ArrayLike, dimension: int) -> FloatArray:
    """Copy and validate a symmetric positive-semidefinite covariance.

    Parameters
    ----------
    value : array_like
        Covariance, shape (dimension, dimension), in coordinate-product units.
    dimension : int
        Positive number of coordinates.

    Returns
    -------
    ndarray
        Symmetrized float64 covariance, shape (dimension, dimension).

    Notes
    -----
    Symmetry and eigenvalue checks allow only scale-relative roundoff of 1e-10.

    References
    ----------
    .. [1] Tracker 001 numerical contracts, spec/tracker-001-integration.md.
    """
    result = np.array(value, dtype=np.float64, copy=True)
    if dimension < 1 or result.shape != (dimension, dimension) or not np.all(np.isfinite(result)):
        raise ValueError(f"covariance must be finite with shape ({dimension}, {dimension})")
    scale = max(1.0, float(np.max(np.abs(result))))
    if not np.allclose(result, result.T, rtol=0, atol=1e-10 * scale):
        raise ValueError("covariance must be symmetric")
    result = (result + result.T) / 2
    if np.linalg.eigvalsh(result)[0] < -1e-10 * scale:
        raise ValueError("covariance must be positive semidefinite")
    return result
###################################################################################################

###################################################################################################
# This function puts a covariance matrix into the form needed to create test states and compare
# readings.
#
# Inputs:
# - matrix (float64 NumPy array): Square covariance matrix, shape (n, n).
#
# Outputs:
# - float64 NumPy array: Square factor used by the filter; only tiny numerical adjustments are
#   allowed.
def cholesky(matrix: FloatArray) -> FloatArray:
    """Factor a covariance using the bounded roundoff-only retry policy.

    Parameters
    ----------
    matrix : ndarray
        Covariance, shape (n, n), in coordinate-product units.

    Returns
    -------
    ndarray
        Lower-triangular float64 factor, shape (n, n).

    Raises
    ------
    ValueError
        If validation fails or the bounded diagonal adjustment is insufficient.

    References
    ----------
    .. [1] Tracker 001 numerical contracts, spec/tracker-001-integration.md.
    """
    matrix = covariance(matrix, len(matrix))
    scale = max(1.0, float(np.max(np.diag(matrix))))
    # Successive factorizations implement a bounded roundoff-only retry policy.
    for jitter in (0.0, 1e-12, 1e-11, 1e-10):
        try:
            return np.asarray(
                np.linalg.cholesky(matrix + np.eye(len(matrix)) * scale * jitter), dtype=np.float64
            )
        except np.linalg.LinAlgError:
            pass
    raise ValueError("covariance is not numerically positive definite")
###################################################################################################

###################################################################################################
# This function checks that an event time is usable before prediction or measurement handling.
#
# Inputs:
# - value_s (float): Event time in seconds.
#
# Outputs:
# - float: Checked time in seconds; missing or infinite values raise an error.
def timestamp(value_s: float) -> float:
    """Convert an event time to a finite number of seconds.

    Parameters
    ----------
    value_s : float
        Event time in seconds.

    Returns
    -------
    float
        Finite event time in seconds.

    References
    ----------
    .. [1] Tracker 001 event contracts, spec/tracker-001-integration.md.
    """
    result = float(value_s)
    if not np.isfinite(result):
        raise ValueError("timestamp_s must be finite")
    return result
###################################################################################################
