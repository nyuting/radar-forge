r"""Scores of a tracker against the truth: filter consistency and accuracy.

:func:`nees`, the normalised estimation error squared (NEES) :math:`e^T P^{-1} e`, is chi-square
with n_state degrees of freedom for a consistent filter; the normalised innovation squared (NIS),
its truth-free twin, is ``Innovation.nis``. :func:`consistency_interval` bounds the average of
N such draws [1]; :func:`rmse` is per coordinate [2]. OSPA and GOSPA, single scores for a
multi-target scene, land here with scenario 011.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
.. [2] X. R. Li and Z. Zhao, "Evaluation of estimation algorithms part I: incomprehensive
       measures of performance," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 42, no. 4,
       pp. 1340-1358, 2006.
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
    """Return the NEES of estimates ``(..., n_state)``, solving rather than inverting P.

    >>> float(nees([0.0, 0.0], np.diag([1.0, 4.0]), [1.0, 2.0]))
    2.0

    References
    ----------
    Bar-Shalom [1], §5.4.2.
    """
    error = _error(mean, truth, state_layout)
    solved = np.linalg.solve(np.asarray(covariance, dtype=np.float64), error[..., None])
    result: NDArray[np.float64] = np.sum(error * solved[..., 0], axis=-1)
    return result


def consistency_interval(
    n_dimensions: int, n_samples: int, probability: float = 0.999
) -> tuple[float, float]:
    r"""Return the two-sided interval the average of ``n_samples`` chi-square draws falls in.

    N draws of :math:`\chi^2_n` sum to :math:`\chi^2_{nN}`, so the bounds are its quantiles
    over N.

    References
    ----------
    Bar-Shalom [1], §5.4.2.
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
    """Return the root-mean-square error of each coordinate, averaged over ``axis``.

    References
    ----------
    Li and Zhao [2] (the RMSE as a measure of estimation performance).
    """
    error = _error(mean, truth, state_layout)
    result: NDArray[np.float64] = np.sqrt(np.mean(error**2, axis=axis))
    return result
