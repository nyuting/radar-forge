r"""The estimator interface, and the statistics that score a measurement against a track.

An estimator is a filter that holds one track's estimate: its mean, its covariance and its time.
:class:`Estimator` is the interface every filter follows. :class:`GaussianFilter` implements
it, and it is the Kalman filter (KF), the extended Kalman filter (EKF) and the unscented Kalman
filter (UKF) in one class: which of them it is depends only on the
:mod:`~radar_forge.core.tracking.linearisation` it is given. The tracks in
:mod:`radar_forge.core.tracking.tracks` hold an estimator each and call only these methods, so
another filter can be swapped in without changing them.

Before a track takes a measurement, the tracker asks how well the two agree. The answer has two
parts. The *innovation* :math:`\nu` is the difference between the measurement and what the
filter expected to see. *S*, the innovation covariance, says how large that difference should
be. Two scores follow from them:

- NIS, the normalised innovation squared, :math:`d^2 = \nu^T S^{-1} \nu`. It is the squared
  distance in units of standard deviations, and d itself is the *Mahalanobis distance*.
  Suppose the filter is right about its own uncertainty. Then the NIS follows a chi-squared
  distribution, with as many degrees of freedom as the measurement has elements. That is
  what the gate tests.
- The Gaussian log-likelihood, :math:`-\tfrac12 (m \ln 2\pi + \ln|S| + d^2)`, for :math:`m`
  measured values.

After the score comes the correction, the same for every filter. It lives in one function,
``_correct``, which :meth:`GaussianFilter.update` calls.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001, §5.2.4 (the Kalman filter
       algorithm, with the innovation and its covariance), §5.2.6 (the innovations
       and the likelihood function) and §5.4.2 (the NIS consistency test).
.. [2] S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013, ch. 6
       (general Gaussian filtering: one filter, with the Taylor and unscented approximations
       as instances).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_triangular

from radar_forge.core.tracking._validation import (
    TIMESTAMP_TOLERANCE_S,
    cholesky_factor,
    symmetrise,
)
from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.linearisation import Linearisation, MeasurementMoments
from radar_forge.core.tracking.measurement_models import Measurement, MeasurementModel
from radar_forge.core.tracking.motion import MotionModel

__all__ = [
    "Estimator",
    "GaussianFilter",
    "InnovationStats",
    "innovation_stats",
]


@dataclass(frozen=True)
class InnovationStats:
    r"""How well one measurement agrees with one track's prediction.

    Parameters
    ----------
    predicted_measurement : numpy.ndarray
        What the filter expected to measure, shape ``(n_meas,)``, in measurement units.
    innovation : numpy.ndarray
        :math:`\nu`, the measurement minus ``predicted_measurement``, shape ``(n_meas,)``.
        Periodic elements, such as an angle, take the short way round.
    innovation_covariance : numpy.ndarray
        S, the innovation covariance, shape ``(n_meas, n_meas)``: the predicted measurement's
        uncertainty plus the measurement noise R.
    nis : float
        The normalised innovation squared, dimensionless and never negative.
    log_likelihood : float
        The natural log of the Gaussian density of ``innovation`` under S.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.2.4, §5.2.6 and §5.4.2.
    """

    predicted_measurement: NDArray[np.float64]
    innovation: NDArray[np.float64]
    innovation_covariance: NDArray[np.float64]
    nis: float
    log_likelihood: float


def innovation_stats(
    predicted: ArrayLike, innovation: ArrayLike, cov: ArrayLike
) -> InnovationStats:
    r"""Score an innovation: its NIS and its Gaussian log-likelihood.

    Parameters
    ----------
    predicted : array_like
        The predicted measurement, shape ``(n_meas,)``, in measurement units.
    innovation : array_like
        :math:`\nu`, the measurement minus ``predicted``, shape ``(n_meas,)``, in the same
        units.
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
    :math:`L w = \nu` gives the *whitened* innovation :math:`w`: the innovation rescaled so
    that its elements are independent with unit variance. So :math:`d^2 = w^T w`, and
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
    innovation = np.asarray(innovation, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    if (
        innovation.ndim != 1
        or predicted.shape != innovation.shape
        or cov.shape != (innovation.size, innovation.size)
    ):
        msg = (
            f"predicted {predicted.shape}, innovation {innovation.shape} and covariance "
            f"{cov.shape} do not agree; expected (m,), (m,) and (m, m)."
        )
        raise ValueError(msg)

    factor = cholesky_factor(cov)
    # Gating, the likelihood and the Kalman gain all use the same S. If the factorisation had
    # to add roundoff jitter, L Lᵀ is the S that was actually scored, so that is the one kept.
    effective_cov = factor @ factor.T
    whitened = solve_triangular(factor, innovation, lower=True)
    nis = float(whitened @ whitened)
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor))))
    log_likelihood = -0.5 * (innovation.size * np.log(2.0 * np.pi) + log_determinant + nis)
    return InnovationStats(predicted, innovation, effective_cov, nis, float(log_likelihood))


class Estimator(Protocol):
    """The interface every filter follows, so that a track can hold any of them.

    An estimator holds one track's estimate and changes it in place: :meth:`predict_to` moves it
    forward in time, and :meth:`update` corrects it with a measurement.
    :meth:`innovation_statistics` only looks; it must never change the estimate, because the
    tracker calls it for every measurement near the track before choosing one.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.2.4 (the predict and update cycle).
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


def _correct(
    prior: StateEstimate,
    stats: InnovationStats,
    cross_covariance: NDArray[np.float64],
    measurement_matrix: NDArray[np.float64] | None,
    measurement_noise: NDArray[np.float64],
) -> StateEstimate:
    r"""Correct a predicted estimate with one scored measurement: the Kalman update.

    This is the one correction every filter in this package uses. The KF, the EKF and the UKF
    differ only in how they found the predicted measurement, :math:`P_{xz}` and S (see
    :mod:`~radar_forge.core.tracking.linearisation`). Everything from here on is the same.

    Parameters
    ----------
    prior : StateEstimate
        The predicted estimate: mean :math:`\hat{x}^-` ``(n_state,)`` and covariance
        :math:`P^-` ``(n_state, n_state)``.
    stats : InnovationStats
        The innovation :math:`\nu` ``(n_meas,)`` and its covariance S ``(n_meas, n_meas)``,
        from :func:`innovation_stats`.
    cross_covariance : numpy.ndarray
        :math:`P_{xz}`, shape ``(n_state, n_meas)``.
    measurement_matrix : numpy.ndarray or None
        H, or the Jacobian of the measurement model, shape ``(n_meas, n_state)``, when the
        linearisation has one; None when it does not (the UKF).
    measurement_noise : numpy.ndarray
        R, shape ``(n_meas, n_meas)``. Used only with ``measurement_matrix``.

    Returns
    -------
    StateEstimate
        The corrected estimate, at the prior's time.

    Notes
    -----
    The gain and the corrected mean are [1]_

    .. math::

        K = P_{xz} S^{-1}, \qquad \hat{x} = \hat{x}^- + K \nu.

    K is found by solving :math:`S K^T = P_{xz}^T`, never by inverting S. The Kalman gain
    says how strongly the measurement corrects the state: a precise measurement (small S)
    against an uncertain prediction (large :math:`P_{xz}`) moves the estimate most of the way
    to the measurement. ``StateEstimate`` wraps the mean's periodic coordinates, through
    :meth:`StateLayout.wrap`, using the *state* layout's periods. The measurement layout's
    periods were used once already, for :math:`\nu`, and are not applied here: a folded
    range has its period on the measurement, while the range state stays continuous.

    **Two forms of the covariance update.** With H, the update is the Joseph form,

    .. math::

        P = (I - K H) P^- (I - K H)^T + K R K^T.

    Both terms have the form :math:`A M A^T` with M positive semidefinite, so P stays
    positive semidefinite for any K, and roundoff cannot make a variance negative. That is
    why the KF and the EKF use it. The UKF has no H, so it uses the short form,

    .. math::

        P = P^- - K S K^T.

    This is the Schur complement of S in the joint covariance of state and measurement. With
    every sigma-point covariance weight non-negative, which
    :class:`~radar_forge.core.tracking.linearisation.UnscentedLinearisation` enforces, that
    joint covariance is positive semidefinite, and so is its Schur complement, up to
    roundoff.

    The two forms agree in exact arithmetic at the optimal gain. With :math:`S = H P^- H^T +
    R` and :math:`P_{xz} = P^- H^T`, the Joseph form expands to :math:`P^- - K P_{xz}^T -
    P_{xz} K^T + K S K^T`. With :math:`K = P_{xz} S^{-1}`, each of the last three terms equals
    :math:`K S K^T`, so the sum is :math:`P^- - K S K^T`. They differ only in how roundoff
    enters. Either way, the symmetric part of P is taken last, which keeps any asymmetric
    roundoff from building up over a long track.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.2.4; Särkkä (2013), ch. 6. See the module
           References.
    """
    innovation_covariance = stats.innovation_covariance
    # K = Pxz S⁻¹, found by solving S Kᵀ = Pxzᵀ rather than inverting S.
    gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T
    mean = prior.mean + gain @ stats.innovation
    if measurement_matrix is None:
        covariance = prior.covariance - gain @ innovation_covariance @ gain.T
    else:
        spread = np.eye(len(prior.mean)) - gain @ measurement_matrix
        covariance = spread @ prior.covariance @ spread.T + gain @ measurement_noise @ gain.T
    return StateEstimate(mean, symmetrise(covariance), prior.timestamp_s, prior.state_layout)


class GaussianFilter:
    r"""The Gaussian filter: the KF, the EKF and the UKF, chosen by a linearisation.

    A Kalman filter keeps a Gaussian estimate of a target's state, a mean and a covariance,
    and repeats two steps. *Predict* moves the estimate forward in time, and *update*
    corrects it with a measurement. The linear Kalman filter (KF), the extended Kalman filter
    (EKF) and the unscented Kalman filter (UKF) all do this, and differ only in how they
    carry a mean and a covariance through a model. That is the ``linearisation``, and it is
    the one thing that changes between them:

    - :class:`~radar_forge.core.tracking.linearisation.UnscentedLinearisation` gives the UKF.

    The flow is the same for all of them, and this class never asks which linearisation it
    holds [1]_.

    Parameters
    ----------
    state : StateEstimate
        The starting estimate: mean ``(n_state,)``, covariance ``(n_state, n_state)`` and
        time. Its layout must be the motion model's.
    motion_model : MotionModel
        Moves states forward in time and gives the process noise Q.
    linearisation : Linearisation
        How means and covariances are carried through the models.

    Attributes
    ----------
    motion_model : MotionModel
        The motion model.
    linearisation : Linearisation
        The linearisation.

    Raises
    ------
    ValueError
        If the state's layout is not the motion model's, or the linearisation's settings do
        not suit the state (see its ``check_motion_model``).
    TypeError
        If the linearisation cannot use this motion model.

    Notes
    -----
    One object holds the estimate and changes it in place, as FilterPy's filters do. Stone
    Soup splits the same algorithm into a stateless predictor and updater, which take an
    estimate and return a new one. The maths is the same; only where the estimate is kept
    differs.

    References
    ----------
    .. [1] Särkkä (2013), ch. 6; see the module References.

    Examples
    --------
    >>> from radar_forge.core.tracking.linearisation import UnscentedLinearisation
    >>> from radar_forge.core.tracking.motion import RadialMotion
    >>> motion = RadialMotion()
    >>> prior = StateEstimate(np.array([1000.0, 10.0]), np.eye(2), 0.0, motion.state_layout)
    >>> ukf = GaussianFilter(prior, motion, UnscentedLinearisation())
    >>> ukf.predict_to(2.0)
    >>> ukf.state.mean
    array([980.,  10.])
    """

    def __init__(
        self,
        state: StateEstimate,
        motion_model: MotionModel,
        linearisation: Linearisation,
    ) -> None:
        linearisation.check_motion_model(motion_model)
        self.motion_model = motion_model
        self.linearisation = linearisation
        # The innovations handed out for the current estimate, by id, with what update() needs
        # to reuse each one. Keeping the objects themselves means no id is reused while listed.
        self._issued: dict[
            int, tuple[InnovationStats, Measurement, MeasurementModel, MeasurementMoments]
        ] = {}
        self.set_state(state)

    @property
    def state(self) -> StateEstimate:
        """The current estimate: mean ``(n_state,)``, covariance ``(n_state, n_state)``, time.

        Returns
        -------
        StateEstimate
            The filter's own estimate object, not a copy. Its arrays are read-only, so a
            caller cannot change the filter through it. The filter replaces the object, never
            changes it, so a reference kept from before a predict or update still holds the
            old estimate.
        """
        return self._state

    def set_state(self, state: StateEstimate) -> None:
        """Replace the estimate.

        Parameters
        ----------
        state : StateEstimate
            The new estimate. Its layout must be the motion model's. It is kept as it is: a
            ``StateEstimate`` already holds read-only copies of its arrays.

        Raises
        ------
        ValueError
            If the layouts do not match.
        """
        if state.state_layout != self.motion_model.state_layout:
            msg = "the state's StateLayout must match the motion model's."
            raise ValueError(msg)
        self._replace_state(state)

    def _replace_state(self, state: StateEstimate) -> None:
        """Store a new estimate and forget everything computed from the old one."""
        self._state = state
        self._issued = {}

    def predict_to(self, timestamp_s: float) -> None:
        """Move the estimate forward to ``timestamp_s``.

        Parameters
        ----------
        timestamp_s : float
            The new time, in seconds. If it is within 1e-6 s of the current time, nothing
            changes.

        Raises
        ------
        ValueError
            If ``timestamp_s`` is not finite, or is earlier than the current estimate
            ("out-of-sequence"). Also if the motion model returns arrays of the wrong shape,
            or a process noise that is not a covariance.
        """
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s):
            msg = f"timestamp_s must be finite; got {timestamp_s}."
            raise ValueError(msg)
        dt_s = timestamp_s - self._state.timestamp_s
        if abs(dt_s) <= TIMESTAMP_TOLERANCE_S:
            return
        if dt_s < 0:
            msg = (
                "out-of-sequence prediction is not supported: "
                f"{timestamp_s} s is before the estimate's {self._state.timestamp_s} s."
            )
            raise ValueError(msg)

        mean, covariance = self.linearisation.predict(self.motion_model, self._state, dt_s)
        self._replace_state(StateEstimate(mean, covariance, timestamp_s, self._state.state_layout))

    def _score(
        self, measurement: Measurement, model: MeasurementModel
    ) -> tuple[InnovationStats, MeasurementMoments]:
        """Return the innovation statistics of a measurement, and the moments behind them."""
        s = self._state
        if abs(measurement.timestamp_s - s.timestamp_s) > TIMESTAMP_TOLERANCE_S:
            msg = (
                "predict the estimator to the measurement's timestamp_s before an update or "
                f"innovation; the measurement is at {measurement.timestamp_s} s, the estimate "
                f"at {s.timestamp_s} s."
            )
            raise ValueError(msg)
        if model.state_layout != s.state_layout:
            msg = "the measurement model's StateLayout must match the filter's."
            raise ValueError(msg)

        moments = self.linearisation.measure(model, s)
        predicted = moments.predicted_measurement
        # S = the predicted measurement's covariance + R.
        innovation_covariance = moments.predicted_measurement_covariance + measurement.covariance
        # residual() takes the measurement layout's periods, and checks the dimension.
        innovation = model.measurement_layout.residual(measurement.value, predicted)
        stats = innovation_stats(predicted, innovation, symmetrise(innovation_covariance))
        return stats, moments

    def innovation_statistics(
        self, measurement: Measurement, model: MeasurementModel
    ) -> InnovationStats:
        """Score a measurement against the current estimate, without changing it.

        Parameters
        ----------
        measurement : Measurement
            The measurement, its noise covariance R and its time. Its time must be within
            1e-6 s of the estimate's: call :meth:`predict_to` with ``measurement.timestamp_s``
            first.
        model : MeasurementModel
            How the state maps to a measurement. Its state layout must be the filter's.

        Returns
        -------
        InnovationStats
            The predicted measurement ``(n_meas,)``, the innovation ``(n_meas,)``, S
            ``(n_meas, n_meas)``, the NIS and the log-likelihood. Pass it to :meth:`update`
            as ``innovation`` to save computing it again.

        Raises
        ------
        ValueError
            If the times differ by more than 1e-6 s, or the layouts do not match. Also if
            the measurement or the model's output has the wrong shape, or S is not positive
            definite.
        """
        stats, moments = self._score(measurement, model)
        self._issued[id(stats)] = (stats, measurement, model, moments)
        return stats

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
            The measurement, its noise covariance R and its time. Its time must be within
            1e-6 s of the estimate's.
        model : MeasurementModel
            How the state maps to a measurement. Its state layout must be the filter's.
        innovation : InnovationStats, optional
            What :meth:`innovation_statistics` returned for this same ``measurement`` and
            ``model`` (the same objects), since the last change to the estimate. It is then
            reused, and the model is not evaluated again. Anything else, including ``None``,
            is ignored and the innovation is computed afresh, which gives the same result.

        Raises
        ------
        ValueError
            If the times differ by more than 1e-6 s, or the layouts do not match. Also if
            the measurement or the model's output has the wrong shape, or S is not positive
            definite.

        Notes
        -----
        The correction itself, and why it has two covariance forms, is ``_correct``'s, in
        this module.
        """
        issued = self._issued.get(id(innovation)) if innovation is not None else None
        if issued is not None and issued[1] is measurement and issued[2] is model:
            stats, moments = issued[0], issued[3]
        else:
            stats, moments = self._score(measurement, model)
        self._replace_state(
            _correct(
                self._state,
                stats,
                moments.cross_covariance,
                moments.measurement_matrix,
                measurement.covariance,
            )
        )
