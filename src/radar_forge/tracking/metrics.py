r"""Scores for a tracker against the truth: filter consistency and accuracy.

- :func:`nees`, the normalised estimation error squared, :math:`e^T P^{-1} e` with
  :math:`e = x - \hat{x}`. For a consistent filter it is chi-square with n_state degrees of
  freedom; the NIS, its measurement-side twin that needs no truth, is
  :attr:`~radar_forge.tracking.filters.Innovation.nis`.
- :func:`consistency_interval`, the two-sided chi-square interval the *average* NEES or NIS
  over N independent samples must fall in [1]_.
- :func:`rmse`, the root-mean-square error of each state coordinate.

OSPA and GOSPA, single scores for a multi-target scene, land in this module with scenario 011.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001, §5.4.2 (the statistical tests for filter consistency).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from radar_forge.tracking.models import StateLayout

__all__ = ["consistency_interval", "nees", "rmse"]


def _error(
    estimate: ArrayLike, truth: ArrayLike, state_layout: StateLayout | None
) -> NDArray[np.float64]:
    """Return truth minus estimate, the short way round when a layout gives periods."""
    if state_layout is None:
        return np.subtract(truth, estimate, dtype=np.float64)
    return state_layout.residual(truth, estimate)


def nees(
    mean: ArrayLike,
    covariance: ArrayLike,
    truth: ArrayLike,
    *,
    state_layout: StateLayout | None = None,
) -> NDArray[np.float64]:
    r"""Return the normalised estimation error squared, :math:`e^T P^{-1} e`.

    Parameters
    ----------
    mean : array_like
        Estimates, ``(..., n_state)``.
    covariance : array_like
        Their covariances, ``(..., n_state, n_state)``.
    truth : array_like
        True states, ``(..., n_state)``.
    state_layout : StateLayout or None, optional
        If given, periodic coordinates are differenced the short way round.

    Returns
    -------
    numpy.ndarray
        NEES, shape ``(...)``; P is never inverted, the error is solved against it.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.4.2; see the module References.

    Examples
    --------
    >>> float(nees([0.0, 0.0], np.diag([1.0, 4.0]), [1.0, 2.0]))
    2.0
    """
    error = _error(mean, truth, state_layout)
    solved = np.linalg.solve(np.asarray(covariance, dtype=np.float64), error[..., None])
    result: NDArray[np.float64] = np.sum(error * solved[..., 0], axis=-1)
    return result


def consistency_interval(
    n_dimensions: int, n_samples: int, probability: float = 0.999
) -> tuple[float, float]:
    r"""Return the interval the average of ``n_samples`` chi-square draws falls in.

    N independent draws of :math:`\chi^2_n` sum to :math:`\chi^2_{nN}`, so their average lies
    in :math:`[\chi^2_{nN}((1-p)/2), \chi^2_{nN}((1+p)/2)] / N` with probability p [1]_.

    Parameters
    ----------
    n_dimensions : int
        Degrees of freedom of one draw: n_state for NEES, n_meas for NIS.
    n_samples : int
        Number of independent draws averaged.
    probability : float, default 0.999
        Two-sided coverage, in (0, 1).

    Returns
    -------
    tuple of float
        The lower and upper bounds on the average.

    Raises
    ------
    ValueError
        If a count is below 1 or ``probability`` is outside (0, 1).

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.4.2; see the module References.

    Examples
    --------
    >>> [round(bound, 4) for bound in consistency_interval(1, 1, 0.95)]
    [0.001, 5.0239]
    """
    if n_dimensions < 1 or n_samples < 1 or not 0.0 < probability < 1.0:
        msg = f"need counts >= 1 and 0 < probability < 1; got {n_dimensions}, {n_samples}."
        raise ValueError(msg)
    dof = n_dimensions * n_samples
    low, high = chi2.ppf([(1 - probability) / 2, (1 + probability) / 2], dof) / n_samples
    return float(low), float(high)


def rmse(
    mean: ArrayLike,
    truth: ArrayLike,
    *,
    axis: int | tuple[int, ...] = 0,
    state_layout: StateLayout | None = None,
) -> NDArray[np.float64]:
    """Return the root-mean-square error of each coordinate.

    Parameters
    ----------
    mean : array_like
        Estimates, ``(n_steps, n_state)`` or any shape ending in ``n_state``.
    truth : array_like
        True states, the same shape.
    axis : int or tuple of int, default 0
        The axes averaged over, such as time and Monte Carlo run; never the last.
    state_layout : StateLayout or None, optional
        If given, periodic coordinates are differenced the short way round.

    Returns
    -------
    numpy.ndarray
        RMSE per coordinate, in each coordinate's unit.

    Examples
    --------
    >>> rmse([[1.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
    array([2.23606798, 0.        ])
    """
    error = _error(mean, truth, state_layout)
    result: NDArray[np.float64] = np.sqrt(np.mean(error**2, axis=axis))
    return result
