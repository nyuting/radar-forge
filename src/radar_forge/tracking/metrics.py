r"""Scores of a tracker against the truth: filter consistency and accuracy.

:func:`nees`, the normalised estimation error squared (NEES) :math:`e^T P^{-1} e`, is chi-square
with n_state degrees of freedom for a consistent filter; the normalised innovation squared (NIS),
its truth-free twin, is ``Innovation.nis``. :func:`consistency_interval` bounds the average of
N such draws [1]; :func:`rmse` is per coordinate [2]. :func:`gospa` scores a multi-target
scene at one time, as localisation, missed and false parts [3]; :func:`trajectory_gospa` scores
a window of tracks against the truth's trajectories and charges each track switch [4].

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
.. [2] X. R. Li and Z. Zhao, "Evaluation of estimation algorithms part I: incomprehensive
       measures of performance," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 42, no. 4,
       pp. 1340-1358, 2006.
.. [3] A. S. Rahmathullah, Á. F. García-Fernández and L. Svensson, "Generalized optimal
       sub-pattern assignment metric," *Proc. 20th Int. Conf. Information Fusion*, 2017;
       arXiv:1601.05585.
.. [4] Á. F. García-Fernández, A. S. Rahmathullah and L. Svensson, "A metric on the space of
       finite sets of trajectories for evaluation of multi-target tracking algorithms," *IEEE
       Trans. Signal Process.*, vol. 68, pp. 3917-3928, 2020; arXiv:1605.01177.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.optimize import linear_sum_assignment, linprog
from scipy.stats import chi2

from radar_forge.tracking.models import StateLayout

__all__ = [
    "Gospa",
    "TrajectoryGospa",
    "consistency_interval",
    "gospa",
    "nees",
    "rmse",
    "trajectory_gospa",
]


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


@dataclass(frozen=True, eq=False)
class Gospa:
    """One GOSPA score and its parts, metres; with alpha = 2 the parts' p-th powers add up.

    ``assignment`` ``(n_truths,)`` is the estimate each truth is localised by, -1 for a missed
    truth: the pairing a swap test follows from scan to scan.
    """

    distance_m: float
    localisation_m: float
    missed_m: float
    false_m: float
    n_missed: int
    n_false: int
    assignment: NDArray[np.intp]


def _distances(estimates: ArrayLike, truths: ArrayLike) -> NDArray[np.float64]:
    """Return ``(n_truths, n_estimates)`` distances; truths ``(n, d)`` or ``(n, n_est, d)``."""
    estimate = np.asarray(estimates, dtype=np.float64)
    truth = np.asarray(truths, dtype=np.float64)
    if truth.ndim == 2:
        truth = truth[:, None, :]
    result: NDArray[np.float64] = np.linalg.norm(truth - estimate[None], axis=-1)
    return result


def _check_metric(p: float, cutoff_m: float, alpha: float) -> None:
    """Raise unless 1 <= p < inf, c > 0 and 0 < alpha <= 2."""
    if not (1.0 <= p < np.inf and 0.0 < cutoff_m < np.inf and 0.0 < alpha <= 2.0):
        msg = f"need 1 <= p, 0 < cutoff_m and 0 < alpha <= 2; got {p}, {cutoff_m}, {alpha}."
        raise ValueError(msg)


def gospa(
    estimates: ArrayLike,
    truths: ArrayLike,
    *,
    p: float = 2.0,
    cutoff_m: float = 300.0,
    alpha: float = 2.0,
) -> Gospa:
    r"""Return the GOSPA distance between estimated and true positions, ``(n, d)`` each.

    :math:`d = (\min_\pi \sum_{i \le |X|} \min(d_{i\pi(i)}, c)^p + (c^p/\alpha)(|Y| - |X|))^{1/p}`
    for :math:`|X| \le |Y|`. With alpha = 2 it is the best assignment's localisation error plus
    :math:`c^p/2` per missed truth and per false estimate, the three parts reported; a pair at
    c or more counts as one of each. ``truths`` may be ``(n_truths, n_estimates, d)``, each
    truth where it was at that estimate's time.

    >>> score = gospa([[0.0, 0.0], [5000.0, 0.0]], [[30.0, 40.0]], cutoff_m=300.0)
    >>> round(score.distance_m, 6), score.n_missed, score.n_false
    (217.944947, 0, 1)

    References
    ----------
    Rahmathullah, García-Fernández and Svensson [3], Definition 1, eq. (1), and Proposition 1.
    """
    _check_metric(p, cutoff_m, alpha)
    distance = _distances(estimates, truths)
    n_truths, n_estimates = distance.shape
    cost = np.minimum(distance, cutoff_m) ** p
    rows, cols = linear_sum_assignment(cost)
    paired = distance[rows, cols] < cutoff_m
    assignment = np.full(n_truths, -1, dtype=np.intp)
    assignment[rows[paired]] = cols[paired]
    n_missed, n_false = n_truths - int(paired.sum()), n_estimates - int(paired.sum())
    localisation = float(np.sum(distance[rows[paired], cols[paired]] ** p))
    unpaired = cutoff_m**p / alpha * abs(n_estimates - n_truths)
    total = float(np.sum(cost[rows, cols])) + unpaired
    half = cutoff_m**p / 2.0
    parts = (localisation, half * n_missed, half * n_false) if alpha == 2.0 else (np.nan,) * 3
    localisation_m, missed_m, false_m = (float(v ** (1 / p)) for v in parts)
    return Gospa(total ** (1 / p), localisation_m, missed_m, false_m, n_missed, n_false, assignment)


@dataclass(frozen=True)
class TrajectoryGospa:
    """A trajectory GOSPA score over a window and its parts, metres, and its switch count.

    The parts' p-th powers add up to the distance's. ``n_switches`` counts a change of
    partner as one and a pairing begun or ended mid-track as one half.
    """

    distance_m: float
    localisation_m: float
    missed_m: float
    false_m: float
    switch_m: float
    n_switches: float


def trajectory_gospa(
    estimates: ArrayLike,
    truths: ArrayLike,
    *,
    p: float = 2.0,
    cutoff_m: float = 300.0,
    switch_cost_m: float | None = None,
) -> TrajectoryGospa:
    r"""Return the trajectory GOSPA distance, as the linear program of [4], alpha = 2.

    ``estimates`` ``(n_steps, n_estimates, d)`` and ``truths`` ``(n_steps, n_truths, d)`` hold
    NaN where a trajectory is absent (truths may be ``(n_steps, n_truths, n_estimates, d)``).
    Each step k assigns with a matrix :math:`W^k` over the trajectories plus one dummy each,
    costed by :math:`D^k`: :math:`\min(d, c)^p` for two present, :math:`c^p/2` for one, zero
    for none. The program minimises :math:`\sum_k \mathrm{tr}(D^{kT} W^k) + (\gamma^p/2)
    \sum_k \sum_{i,j} |W^k_{ij} - W^{k+1}_{ij}|`, the switch term over the real entries only;
    the program is itself a metric and its optimum is usually binary. Cost grows as steps
    times truths times estimates.

    References
    ----------
    García-Fernández, Rahmathullah and Svensson [4], Proposition 2, eq. (23).
    """
    _check_metric(p, cutoff_m, 2.0)
    gamma = cutoff_m if switch_cost_m is None else switch_cost_m
    estimate = np.asarray(estimates, dtype=np.float64)
    truth = np.asarray(truths, dtype=np.float64)
    n_steps, n_estimates = estimate.shape[:2]
    n_truths = truth.shape[1]
    # One distance matrix per step: a handful of scans to a few hundred.
    distance = np.stack([_distances(estimate[k], truth[k]) for k in range(n_steps)])
    estimate_on = np.all(np.isfinite(estimate), axis=-1)
    truth_on = np.all(np.isfinite(truth[:, :, 0] if truth.ndim == 4 else truth), axis=-1)
    half = cutoff_m**p / 2.0
    real = np.where(
        truth_on[:, :, None] & estimate_on[:, None, :],
        np.minimum(np.nan_to_num(distance, nan=cutoff_m), cutoff_m) ** p,
        half * (truth_on[:, :, None] ^ estimate_on[:, None, :]),
    )
    shape = (n_steps, n_truths + 1, n_estimates + 1)
    cost = np.zeros(shape)
    cost[:, :n_truths, :n_estimates] = real
    cost[:, :n_truths, -1] = half * truth_on
    cost[:, -1, :n_estimates] = half * estimate_on
    n_w = cost.size
    w_index = np.arange(n_w).reshape(shape)
    n_e = (n_steps - 1) * n_truths * n_estimates
    # Each real row and column of every W^k sums to one; the dummy-dummy entry is held at 0.
    truth_row = np.arange(n_steps * n_truths).reshape(n_steps, n_truths, 1)
    estimate_row = n_steps * n_truths + np.arange(n_steps * n_estimates).reshape(n_steps, 1, -1)
    equality = sparse.coo_array(
        (
            np.ones(2 * n_steps * n_truths * n_estimates + n_steps * (n_truths + n_estimates)),
            (
                np.concatenate(
                    [
                        np.broadcast_to(truth_row, (n_steps, n_truths, n_estimates + 1)).ravel(),
                        np.broadcast_to(estimate_row, (n_steps, n_truths + 1, n_estimates)).ravel(),
                    ]
                ),
                np.concatenate(
                    [w_index[:, :n_truths, :].ravel(), w_index[:, :, :n_estimates].ravel()]
                ),
            ),
        ),
        shape=(n_steps * (n_truths + n_estimates), n_w + n_e),
    )
    # |W^k - W^(k+1)| <= e on every real entry, as two inequalities each.
    now = w_index[:-1, :n_truths, :n_estimates].ravel()
    later = w_index[1:, :n_truths, :n_estimates].ravel()
    slack = n_w + np.arange(n_e)
    count = np.arange(n_e)
    sign = np.concatenate([np.ones(n_e), -np.ones(n_e)])
    inequality = sparse.coo_array(
        (
            np.concatenate([sign, -sign, -np.ones(2 * n_e)]),
            (
                np.concatenate([count, n_e + count] * 3),
                np.concatenate([now, now, later, later, slack, slack]),
            ),
        ),
        shape=(2 * n_e, n_w + n_e),
    )
    bounds = np.zeros((n_w + n_e, 2))
    bounds[:, 1] = np.inf
    bounds[w_index[:, -1, -1], 1] = 0.0
    objective = np.concatenate([cost.ravel(), np.full(n_e, gamma**p / 2.0)])
    solution = linprog(
        objective,
        A_ub=inequality.tocsr(),
        b_ub=np.zeros(2 * n_e),
        A_eq=equality.tocsr(),
        b_eq=np.ones(n_steps * (n_truths + n_estimates)),
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        msg = f"the trajectory GOSPA program did not solve: {solution.message}"
        raise ValueError(msg)
    weight = solution.x[:n_w].reshape(shape)
    both = truth_on[:, :, None] & estimate_on[:, None, :]
    close = both & (np.nan_to_num(distance, nan=np.inf) < cutoff_m)
    localisation = float(np.sum(weight[:, :n_truths, :n_estimates] * np.where(close, real, 0.0)))
    on_truth = np.where(close, 0.0, cost[:, :n_truths, :n_estimates] * truth_on[:, :, None])
    missed = float(np.sum(weight[:, :n_truths, -1] * cost[:, :n_truths, -1]))
    missed += float(np.sum(weight[:, :n_truths, :n_estimates] * np.minimum(on_truth, half)))
    switches = float(np.sum(solution.x[n_w:])) / 2.0
    switch = gamma**p * switches
    false = max(float(solution.fun) - localisation - missed - switch, 0.0)
    localisation_m, missed_m, false_m, switch_m = (
        v ** (1 / p) for v in (localisation, missed, false, switch)
    )
    return TrajectoryGospa(
        float(solution.fun) ** (1 / p), localisation_m, missed_m, false_m, switch_m, switches
    )
