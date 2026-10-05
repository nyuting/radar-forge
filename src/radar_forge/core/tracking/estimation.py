r"""The estimator interface, and the statistics that score a measurement against a track.

An estimator is a filter that holds one track's estimate: its mean, its covariance and its time.
:class:`Estimator` is the interface every filter follows; :class:`radar_forge.core.tracking.ukf.UKF`
is the one implementation. The tracks in :mod:`radar_forge.core.tracking.tracks` hold an
estimator each and call only these methods, so another filter can be swapped in without
changing them.

Before a track takes a measurement, the tracker asks how well the two agree. The answer is the
*innovation*, the difference between the measurement and what the filter expected to see, and
*S*, the innovation covariance, which says how large that difference should be. Two scores
follow from them:

- NIS, the normalised innovation squared, :math:`d^2 = \nu^T S^{-1} \nu`. It is the squared
  distance in units of standard deviations. If the filter is right about its own uncertainty, it
  follows a chi-squared distribution with as many degrees of freedom as the measurement has
  elements, and that is what the gate tests.
- The Gaussian log-likelihood, :math:`-\tfrac12 (m \ln 2\pi + \ln|S| + d^2)`, for :math:`m`
  measured values.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001, §5.4 (the innovation, its covariance, and the NIS
       consistency test).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_triangular

from radar_forge.core.tracking._validation import cholesky_factor
from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.measurement_models import Measurement, MeasurementModel

__all__ = [
    "Estimator",
    "InnovationStats",
    "innovation_stats",
]


@dataclass(frozen=True)
class InnovationStats:
    """How well one measurement agrees with one track's prediction.

    Parameters
    ----------
    predicted_measurement : numpy.ndarray
        What the filter expected to measure, shape ``(n_meas,)``, in measurement units.
    residual : numpy.ndarray
        The innovation: the measurement minus ``predicted_measurement``, shape ``(n_meas,)``.
        Periodic elements, such as an angle, take the short way round.
    innovation_covariance : numpy.ndarray
        S, the innovation covariance, shape ``(n_meas, n_meas)``: the predicted measurement's
        uncertainty plus the measurement noise R.
    nis : float
        The normalised innovation squared, dimensionless and never negative.
    log_likelihood : float
        The natural log of the Gaussian density of ``residual`` under S.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.4.
    """

    predicted_measurement: NDArray[np.float64]
    residual: NDArray[np.float64]
    innovation_covariance: NDArray[np.float64]
    nis: float
    log_likelihood: float


def innovation_stats(predicted: ArrayLike, residual: ArrayLike, cov: ArrayLike) -> InnovationStats:
    r"""Score an innovation: its NIS and its Gaussian log-likelihood.

    Parameters
    ----------
    predicted : array_like
        The predicted measurement, shape ``(n_meas,)``, in measurement units.
    residual : array_like
        The innovation :math:`\nu`, the measurement minus ``predicted``, shape ``(n_meas,)``,
        in the same units.
    cov : array_like
        S, the innovation covariance, shape ``(n_meas, n_meas)``. Must be symmetric and
        positive definite.

    Returns
    -------
    InnovationStats
        The inputs as float64 arrays, with S replaced by :math:`L L^T` (see Notes), and the two
        scores. The arrays are not copied if they already are float64.

    Raises
    ------
    ValueError
        If the three shapes do not agree, or S is singular or not positive definite.

    Notes
    -----
    Both scores come from the Cholesky factor L of S, with :math:`L L^T = S`. Solving
    :math:`L w = \nu` gives the whitened innovation :math:`w`, so :math:`d^2 = w^T w`, and
    :math:`\ln|S| = 2 \sum_i \ln L_{ii}`. This never forms :math:`S^{-1}`, which loses accuracy
    when S is badly conditioned, as it is when range and range rate differ in variance by six
    orders of magnitude.

    Examples
    --------
    >>> stats = innovation_stats([0.0], [3.0], [[9.0]])
    >>> stats.nis
    1.0
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    if (
        residual.ndim != 1
        or predicted.shape != residual.shape
        or cov.shape != (residual.size, residual.size)
    ):
        msg = (
            f"predicted {predicted.shape}, residual {residual.shape} and covariance "
            f"{cov.shape} do not agree; expected (m,), (m,) and (m, m)"
        )
        raise ValueError(msg)

    factor = cholesky_factor(cov)
    # Gating, the likelihood and the Kalman gain all use the same S. If the factorisation had
    # to add roundoff jitter, L Lᵀ is the S that was actually scored, so that is the one kept.
    effective_cov = factor @ factor.T
    whitened = solve_triangular(factor, residual, lower=True)
    nis = float(whitened @ whitened)
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor))))
    log_likelihood = -0.5 * (residual.size * np.log(2.0 * np.pi) + log_determinant + nis)
    return InnovationStats(predicted, residual, effective_cov, nis, float(log_likelihood))


class Estimator(Protocol):
    """The interface every filter follows, so that a track can hold any of them.

    An estimator holds one track's estimate and changes it in place: :meth:`predict_to` moves it
    forward in time, and :meth:`update` corrects it with a measurement.
    :meth:`innovation_statistics` only looks; it must never change the estimate, because the
    tracker calls it for every measurement near the track before choosing one.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.2 (the predict and update cycle).
    """

    @property
    def state(self) -> StateEstimate:
        """The current estimate: mean ``(n_state,)``, covariance ``(n_state, n_state)``, time.

        Returns
        -------
        StateEstimate
            The estimate. Its arrays must be read-only, or copies, so that a caller cannot
            change the filter through them.
        """
        ...

    def set_state(self, state: StateEstimate) -> None:
        """Replace the estimate, for example when a track starts or is merged.

        Parameters
        ----------
        state : StateEstimate
            The new estimate. Its layout must match the filter's.

        Raises
        ------
        ValueError
            If the layouts do not match.
        """
        ...

    def predict_to(self, timestamp_s: float) -> None:
        """Move the estimate forward to ``timestamp_s``, growing its uncertainty.

        Parameters
        ----------
        timestamp_s : float
            The new time, in seconds. Must not be earlier than the current estimate.

        Raises
        ------
        ValueError
            If ``timestamp_s`` is not finite or is earlier than the current estimate.
        """
        ...

    def update(
        self,
        measurement: Measurement,
        model: MeasurementModel,
        *,
        innovation: InnovationStats | None = None,
    ) -> None:
        """Correct the estimate with one measurement taken at the estimate's own time.

        Parameters
        ----------
        measurement : Measurement
            The measurement, its noise covariance R and its time.
        model : MeasurementModel
            How the state maps to a measurement.
        innovation : InnovationStats, optional
            What :meth:`innovation_statistics` returned for this same measurement and model,
            against the current estimate. A filter may reuse it rather than recompute it.

        Raises
        ------
        ValueError
            If the measurement is not at the estimate's time, or the model's state layout is
            not the filter's.
        """
        ...

    def innovation_statistics(
        self, measurement: Measurement, model: MeasurementModel
    ) -> InnovationStats:
        """Score a measurement against the current estimate, without changing it.

        The tracker uses this to gate and to score each pairing of a track and a measurement.

        Parameters
        ----------
        measurement : Measurement
            The measurement, its noise covariance R and its time.
        model : MeasurementModel
            How the state maps to a measurement.

        Returns
        -------
        InnovationStats
            The predicted measurement, the innovation, S, the NIS and the log-likelihood.

        Raises
        ------
        ValueError
            If the measurement is not at the estimate's time, or the model's state layout is
            not the filter's.
        """
        ...
