r"""The unscented Kalman filter (UKF), for any motion model and any measurement model.

A Kalman filter keeps a Gaussian estimate of a target's state, a mean and a covariance, and
repeats two steps: *predict* moves the estimate forward in time, and *update* corrects it with a
measurement. The classic filter needs both models to be linear. The UKF lifts that limit without
derivatives: it picks a small set of *sigma points*, states spread around the mean so that
their weighted mean and covariance are exactly the estimate's. It pushes each point through the
model, and reads the new mean and covariance off the moved points.

For :math:`n` state elements there are :math:`2n + 1` sigma points: the mean itself, and one
point either side of it along each column of the covariance's Cholesky factor. All of them are
handled together as one ``(n_points, n_state)`` array.

This class follows FilterPy's ``UnscentedKalmanFilter``: one object holds the estimate and
changes it in place. Stone Soup splits the same algorithm into a stateless
``UnscentedKalmanPredictor`` and ``UnscentedKalmanUpdater``, which take an estimate and return a
new one. The maths is the same; only where the estimate is kept differs.

References
----------
.. [1] E. A. Wan and R. van der Merwe, "The unscented Kalman filter for nonlinear estimation,"
       *Proc. IEEE Adaptive Systems for Signal Processing, Communications, and Control
       Symposium*, 2000, pp. 153-158, §3 (the UKF algorithm).
.. [2] S. J. Julier, "The scaled unscented transformation," *Proc. American Control
       Conference*, 2002, pp. 4555-4559 (the parameters alpha, beta and kappa).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from radar_forge.core.tracking._validation import (
    TIMESTAMP_TOLERANCE_S,
    as_covariance,
    as_points,
    cholesky_factor,
)
from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.estimation import InnovationStats, innovation_stats
from radar_forge.core.tracking.measurement_models import Measurement, MeasurementModel
from radar_forge.core.tracking.motion import MotionModel

__all__ = [
    "UKF",
]


def _symmetrise(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the symmetric part of a square matrix, removing asymmetric roundoff."""
    return (matrix + matrix.T) / 2


class UKF:
    r"""A scaled unscented Kalman filter with additive process and measurement noise.

    Parameters
    ----------
    state : StateEstimate
        The starting estimate: mean ``(n_state,)``, covariance ``(n_state, n_state)`` and time.
        Its layout must be the motion model's.
    motion_model : MotionModel
        Moves states forward in time and gives the process noise Q.
    alpha : float, default 0.5
        :math:`\alpha`, the standard sigma-point symbol for how far the points spread from the
        mean. Must be positive.
    beta : float, default 2.0
        :math:`\beta`, the standard sigma-point symbol for prior knowledge of the distribution.
        2 is the best choice for a Gaussian. Must not be negative.
    kappa : float, default 1.0
        :math:`\kappa`, the standard sigma-point symbol for a second spread adjustment.
        ``n_state + kappa`` must be positive. See Notes for why the default is 1.

    Attributes
    ----------
    motion_model : MotionModel
        The motion model.
    scale : float
        :math:`\alpha^2 (n + \kappa)`. The sigma points sit at :math:`\pm\sqrt{scale}` times
        each column of the covariance's Cholesky factor.
    wm, wc : numpy.ndarray
        The weights, shape ``(2 n_state + 1,)``, for the mean and for the covariance.

    Raises
    ------
    ValueError
        If a parameter is not finite, ``alpha`` is not positive, ``beta`` is negative,
        ``n_state + kappa`` is not positive, or the state's layout is not the motion model's.

    Notes
    -----
    With :math:`\lambda = \alpha^2 (n + \kappa) - n`, the weights are [2]_

    .. math::

        W^{(m)}_0 = \frac{\lambda}{n + \lambda}, \quad
        W^{(c)}_0 = W^{(m)}_0 + 1 - \alpha^2 + \beta, \quad
        W^{(m)}_i = W^{(c)}_i = \frac{1}{2 (n + \lambda)}, \; i = 1 \ldots 2n.

    **Why kappa = 1.** The covariances the filter computes are weighted sums of outer products,
    one per sigma point. If every covariance weight is zero or positive, each of those sums is
    positive semidefinite by construction, and so is the updated covariance (see
    :meth:`update`). All the weights after the first are positive. The first, :math:`W^{(c)}_0`,
    is not positive for every choice:

    - ``kappa = 0``, the value in Wan and van der Merwe [1]_, gives :math:`W^{(m)}_0 = 1 -
      1/\alpha^2 = -3` and :math:`W^{(c)}_0 = -0.25` for every :math:`n`.
    - ``kappa = 3 - n``, Julier's choice for a Gaussian and Stone Soup's default, keeps
      :math:`n + \kappa = 3`, but gives :math:`W^{(c)}_0 = 3.75 - 4n/3`, negative for
      :math:`n \ge 3`: -1.58 for a 2-D CV state, -4.25 for a 3-D one.
    - ``kappa = 1`` gives :math:`W^{(c)}_0 = 3.75 - 4n/(n + 1)`, zero or positive for every
      :math:`n` up to 15. The largest state here, 3-D CA, has :math:`n = 9`.

    With a linear model the first sigma point lands exactly on the mean, so its weight does not
    matter. It matters with a nonlinear model such as the bistatic range and Doppler model.

    The measurement sigma points are drawn again after each prediction, so the process noise Q
    is part of the measurement and cross covariances. The sigma points of the current estimate
    are computed once and reused until the estimate changes.

    References
    ----------
    .. [1] Wan and van der Merwe (2000), §3.
    .. [2] Julier (2002).

    Examples
    --------
    >>> from radar_forge.core.tracking.motion import RadialMotion
    >>> motion = RadialMotion()
    >>> prior = StateEstimate(np.array([1000.0, 10.0]), np.eye(2), 0.0, motion.state_layout)
    >>> ukf = UKF(prior, motion)
    >>> ukf.predict_to(2.0)
    >>> ukf.state.mean
    array([980.,  10.])
    """

    def __init__(
        self,
        state: StateEstimate,
        motion_model: MotionModel,
        alpha: float = 0.5,
        beta: float = 2.0,
        kappa: float = 1.0,
    ) -> None:
        n_state = state.state_layout.dimension
        if not np.all(np.isfinite([alpha, beta, kappa])):
            msg = f"alpha, beta and kappa must be finite; got {alpha}, {beta}, {kappa}"
            raise ValueError(msg)
        if alpha <= 0 or beta < 0 or n_state + kappa <= 0:
            msg = (
                "require alpha > 0, beta >= 0 and n_state + kappa > 0; "
                f"got alpha={alpha}, beta={beta}, n_state + kappa={n_state + kappa}"
            )
            raise ValueError(msg)

        self.motion_model = motion_model
        self._points: NDArray[np.float64] | None = None
        # The innovations handed out for the current estimate, by id, with what update() needs
        # to reuse each one. Keeping the objects themselves means no id is reused while listed.
        self._issued: dict[
            int, tuple[InnovationStats, Measurement, MeasurementModel, NDArray[np.float64]]
        ] = {}
        self.set_state(state)
        self.scale = alpha**2 * (n_state + kappa)
        self.wm = np.full(2 * n_state + 1, 1 / (2 * self.scale), dtype=np.float64)
        self.wm[0] = 1 - n_state / self.scale
        self.wc = self.wm.copy()
        self.wc[0] += 1 - alpha**2 + beta

    @property
    def state(self) -> StateEstimate:
        """The current estimate: mean ``(n_state,)``, covariance ``(n_state, n_state)``, time.

        Returns
        -------
        StateEstimate
            The filter's own estimate object, not a copy. Its arrays are read-only, so a caller
            cannot change the filter through it. The filter replaces the object, never changes
            it, so a reference kept from before a predict or update still holds the old
            estimate.
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
            msg = "the state's StateLayout must match the motion model's"
            raise ValueError(msg)
        self._replace_state(state)

    def _replace_state(self, state: StateEstimate) -> None:
        """Store a new estimate and forget everything computed from the old one."""
        self._state = state
        self._points = None
        self._issued = {}

    def _sigma_points(self) -> NDArray[np.float64]:
        r"""Return the sigma points of the current estimate, shape ``(2 n_state + 1, n_state)``.

        Row 0 is the mean. Rows 1 to n are the mean plus :math:`\sqrt{scale}` times each column
        of the covariance's Cholesky factor, and rows n + 1 to 2n the mean minus them. Periodic
        elements are wrapped onto their principal interval.
        """
        if self._points is None:
            s = self._state
            # Row i of Lᵀ is column i of L, so each row is one offset.
            offsets = np.sqrt(self.scale) * cholesky_factor(s.covariance).T
            deviations = np.concatenate([np.zeros((1, len(s.mean))), offsets, -offsets])
            self._points = s.state_layout.normalise(s.mean + deviations)
        return self._points

    def predict_to(self, time_s: float) -> None:
        r"""Move the estimate forward to ``time_s``.

        Parameters
        ----------
        time_s : float
            The new time, in seconds. If it is within 1e-6 s of the current time, nothing
            changes.

        Raises
        ------
        ValueError
            If ``time_s`` is not finite, or is earlier than the current estimate
            ("out-of-sequence"); or if the motion model returns arrays of the wrong shape, or a
            process noise that is not a covariance.

        Notes
        -----
        Each sigma point :math:`\mathcal{X}_i` is moved by the motion model to
        :math:`\mathcal{X}'_i`. Then

        .. math::

            \hat{x}^- = \sum_i W^{(m)}_i \mathcal{X}'_i, \qquad
            P^- = \sum_i W^{(c)}_i (\mathcal{X}'_i - \hat{x}^-)(\mathcal{X}'_i -
            \hat{x}^-)^T + Q.
        """
        time_s = float(time_s)
        if not np.isfinite(time_s):
            msg = f"timestamp_s must be finite; got {time_s}"
            raise ValueError(msg)
        dt_s = time_s - self._state.timestamp_s
        if abs(dt_s) <= TIMESTAMP_TOLERANCE_S:
            return
        if dt_s < 0:
            msg = (
                "out-of-sequence prediction is not supported: "
                f"{time_s} s is before the estimate's {self._state.timestamp_s} s"
            )
            raise ValueError(msg)

        s = self._state
        n_points, n_state = len(self.wm), len(s.mean)
        points = as_points(
            self.motion_model.transition(self._sigma_points(), dt_s),
            n_points,
            n_state,
            "transition",
        )
        noise = as_covariance(self.motion_model.process_noise(s.mean, dt_s), n_state)

        mean = s.state_layout.weighted_mean(points, self.wm)
        deviations = s.state_layout.residual(points, mean)
        covariance = (deviations.T * self.wc) @ deviations + noise
        self._replace_state(StateEstimate(mean, _symmetrise(covariance), time_s, s.state_layout))

    def _innovation(
        self, measurement: Measurement, model: MeasurementModel
    ) -> tuple[InnovationStats, NDArray[np.float64]]:
        """Return the innovation statistics and the cross covariance, shape ``(n_state, n_meas)``.

        The cross covariance :math:`P_{xz}` says how the state's errors move with the predicted
        measurement's errors. The Kalman gain is built from it.
        """
        s = self._state
        if abs(measurement.timestamp_s - s.timestamp_s) > TIMESTAMP_TOLERANCE_S:
            msg = (
                "predict the estimator to the measurement's timestamp_s before an update or "
                f"innovation; the measurement is at {measurement.timestamp_s} s, the estimate "
                f"at {s.timestamp_s} s"
            )
            raise ValueError(msg)
        if model.state_layout != s.state_layout:
            msg = "the measurement model's StateLayout must match the filter's"
            raise ValueError(msg)

        points = self._sigma_points()
        layout = model.measurement_layout
        predicted_points = as_points(
            model.predict(points), len(self.wm), layout.dimension, "predicted measurement"
        )
        predicted = layout.weighted_mean(predicted_points, self.wm)
        dz = layout.residual(predicted_points, predicted)
        dx = s.state_layout.residual(points, s.mean)
        # S = sum of W_c dz dzᵀ + R, the sigma-point form of H P Hᵀ + R.
        innovation_covariance = (dz.T * self.wc) @ dz + measurement.covariance
        # residual() also checks that the measurement has the model's dimension.
        residual = layout.residual(measurement.value, predicted)
        stats = innovation_stats(predicted, residual, _symmetrise(innovation_covariance))
        return stats, (dx.T * self.wc) @ dz

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
            ``(n_meas, n_meas)``, the NIS and the log-likelihood. Pass it to :meth:`update` as
            ``innovation`` to save computing it again.

        Raises
        ------
        ValueError
            If the times differ by more than 1e-6 s, the layouts do not match, the
            measurement or the model's output has the wrong shape, or S is not positive
            definite.
        """
        stats, cross = self._innovation(measurement, model)
        self._issued[id(stats)] = (stats, measurement, model, cross)
        return stats

    def update(
        self,
        measurement: Measurement,
        model: MeasurementModel,
        *,
        innovation: InnovationStats | None = None,
    ) -> None:
        r"""Correct the estimate with one measurement taken at the estimate's own time.

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
            reused, and the sigma points are not pushed through the model again. Anything
            else, including ``None``, is ignored and the innovation is computed afresh, which
            gives the same result.

        Raises
        ------
        ValueError
            If the times differ by more than 1e-6 s, the layouts do not match, the
            measurement or the model's output has the wrong shape, or S is not positive
            definite.

        Notes
        -----
        With S the innovation covariance, :math:`P_{xz}` the cross covariance and
        :math:`\nu` the innovation, the gain and the corrected estimate are [1]_

        .. math::

            K = P_{xz} S^{-1}, \qquad \hat{x} = \hat{x}^- + K \nu, \qquad
            P = P^- - K S K^T.

        **Why not the Joseph form.** The linear Kalman filter is often written in the Joseph
        form, :math:`(I - KH) P^- (I - KH)^T + K R K^T`, because it stays positive semidefinite
        under roundoff and even with a gain that is not optimal. It needs the measurement
        matrix H, which the UKF does not have. The short form is safe here for two reasons:

        - In exact arithmetic :math:`P^- - K S K^T = P^- - P_{xz} S^{-1} P_{xz}^T`. That is the
          Schur complement of S in the joint covariance of the state and the measurement,
          :math:`\begin{bmatrix} P^- & P_{xz} \\ P_{xz}^T & S \end{bmatrix}`. The joint
          covariance is a sum of outer products with weights :math:`W^{(c)}_i`, plus R. With
          every weight non-negative (the default ``kappa``) it is positive semidefinite, and
          so is any Schur complement of it.
        - In floating point the result is then off by roundoff only, about :math:`n` times
          float64 epsilon of the largest element. Taking the symmetric part removes the
          asymmetric part of that error. What is left is five orders of magnitude below the
          tolerance ``StateEstimate`` allows, so roundoff cannot stop a long run.

        ``tests/core/tracking/test_ukf.py`` runs 200 steps at a range variance of 468 m² and a
        range-rate variance of 3e-4 m²/s², and checks the covariance stays positive definite.

        References
        ----------
        .. [1] Wan and van der Merwe (2000), §3.
        """
        issued = self._issued.get(id(innovation)) if innovation is not None else None
        if issued is not None and issued[1] is measurement and issued[2] is model:
            stats, cross = issued[0], issued[3]
        else:
            stats, cross = self._innovation(measurement, model)

        s = self._state
        innovation_covariance = stats.innovation_covariance
        # K = Pxz S⁻¹, found by solving S Kᵀ = Pxzᵀ rather than inverting S.
        gain = np.linalg.solve(innovation_covariance, cross.T).T
        mean = s.mean + gain @ stats.residual
        covariance = s.covariance - gain @ innovation_covariance @ gain.T
        self._replace_state(
            StateEstimate(mean, _symmetrise(covariance), s.timestamp_s, s.state_layout)
        )
