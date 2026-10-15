r"""How a Gaussian filter carries a mean and a covariance through a function.

The Kalman filter (KF), the extended Kalman filter (EKF) and the unscented Kalman filter (UKF)
are one algorithm, the general Gaussian filter [1]_. Each step treats the estimate as a
Gaussian with mean :math:`\hat{x}` and covariance :math:`P`, pushes it through a function g
(the motion model to predict, the measurement model to update), and keeps only the mean and
covariance of the result. Then the same Kalman correction follows. The three filters differ in
one thing only: how they work out the mean and covariance of :math:`g(x)`. That is the
filter's *linearisation*, and this module holds them.

- :class:`UnscentedLinearisation`, the UKF: weighted sums over :math:`2n + 1` sigma points.

A linearisation is handed to
:class:`~radar_forge.core.tracking.estimation.GaussianFilter`, which never asks which one it
holds. A new one, such as cubature or Gauss-Hermite, is a new class here, never a new filter.

Each linearisation has the two methods of :class:`Linearisation`, one per step of the filter:

- ``predict(motion_model, state, dt_s)`` returns the predicted mean :math:`\hat{x}^-`,
  shape ``(n_state,)``, and covariance :math:`P^-`, shape ``(n_state, n_state)``, with the
  process noise Q already added.
- ``measure(model, state)`` returns :class:`MeasurementMoments`: the predicted measurement,
  the cross covariance :math:`P_{xz}` and the predicted measurement's covariance, which is
  the innovation covariance S without the measurement noise R. A linearisation that has a
  measurement matrix H also returns it, and the filter then uses the Joseph form of the
  covariance update (:mod:`~radar_forge.core.tracking.estimation`).

A third, ``check_motion_model``, lets a linearisation refuse a motion model it cannot use
when the filter is built, rather than at the first prediction.

References
----------
.. [1] S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013,
       ch. 5 (the extended and unscented Kalman filters) and ch. 6 (general Gaussian
       filtering, with the Taylor and unscented approximations as instances).
.. [2] E. A. Wan and R. van der Merwe, "The unscented Kalman filter for nonlinear estimation,"
       *Proc. IEEE Adaptive Systems for Signal Processing, Communications, and Control
       Symposium*, 2000, pp. 153-158, §3 (the UKF algorithm and its weights).
.. [3] S. J. Julier, "The scaled unscented transformation," *Proc. American Control
       Conference*, 2002, pp. 4555-4559 (the scaling parameter alpha).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from radar_forge.core.tracking._validation import (
    as_covariance,
    as_points,
    check_sigma_point_settings,
    cholesky_factor,
    symmetrise,
)
from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.measurement_models import MeasurementModel
from radar_forge.core.tracking.motion import MotionModel

__all__ = [
    "Linearisation",
    "MeasurementMoments",
    "UnscentedLinearisation",
]


@dataclass(frozen=True)
class MeasurementMoments:
    r"""What a linearisation says about the measurement a state should produce.

    Parameters
    ----------
    predicted_measurement : numpy.ndarray
        :math:`\hat{z}`, the expected measurement, shape ``(n_meas,)``, in measurement units.
    cross_covariance : numpy.ndarray
        :math:`P_{xz}`, how the state's errors move with the predicted measurement's errors,
        shape ``(n_state, n_meas)``. The Kalman gain is built from it.
    predicted_measurement_covariance : numpy.ndarray
        The covariance of the predicted measurement, shape ``(n_meas, n_meas)``: the
        innovation covariance S *without* the measurement noise R, which the filter adds.
    measurement_matrix : numpy.ndarray or None, optional
        H, or the Jacobian of the measurement model at the state, shape
        ``(n_meas, n_state)``, for a linearisation that has one. None (the default) for one
        that does not, such as the unscented linearisation.

    References
    ----------
    .. [1] Särkkä (2013), ch. 6; see the module References.
    """

    predicted_measurement: NDArray[np.float64]
    cross_covariance: NDArray[np.float64]
    predicted_measurement_covariance: NDArray[np.float64]
    measurement_matrix: NDArray[np.float64] | None = None


class Linearisation(Protocol):
    """The interface every linearisation follows: how to carry a Gaussian through a function.

    References
    ----------
    .. [1] Särkkä (2013), ch. 6; see the module References.
    """

    def check_motion_model(self, motion_model: MotionModel) -> None:
        """Raise unless this linearisation can predict with ``motion_model``.

        Parameters
        ----------
        motion_model : MotionModel
            The motion model the filter is being built with.

        Raises
        ------
        TypeError
            If the motion model lacks a method this linearisation needs.
        ValueError
            If the linearisation's own settings do not suit the model's state size.
        """
        ...

    def predict(
        self, motion_model: MotionModel, state: StateEstimate, dt_s: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Return the predicted mean and covariance after a step of ``dt_s`` seconds.

        Parameters
        ----------
        motion_model : MotionModel
            Moves states forward in time and gives the process noise Q.
        state : StateEstimate
            The estimate to move: mean ``(n_state,)`` and covariance ``(n_state, n_state)``.
        dt_s : float
            The time step, in seconds. Positive.

        Returns
        -------
        mean : numpy.ndarray
            :math:`\hat{x}^-`, shape ``(n_state,)``.
        covariance : numpy.ndarray
            :math:`P^-`, shape ``(n_state, n_state)``, with Q added.
        """
        ...

    def measure(self, model: MeasurementModel, state: StateEstimate) -> MeasurementMoments:
        """Return the moments of the measurement ``state`` should produce through ``model``.

        Parameters
        ----------
        model : MeasurementModel
            How the state maps to a measurement. Its state layout is the state's.
        state : StateEstimate
            The predicted estimate.

        Returns
        -------
        MeasurementMoments
            The predicted measurement, :math:`P_{xz}`, S without R, and H or None.
        """
        ...


@dataclass(frozen=True)
class UnscentedLinearisation:
    r"""The unscented transformation: the linearisation of the UKF.

    The unscented filter needs no derivatives [2]_. It picks :math:`2n + 1` *sigma points*:
    states spread around the mean, so that their weighted mean and covariance are exactly
    the estimate's. It pushes each point through the function, and reads the new mean and
    covariance off the moved points. One point is the mean itself. The others sit either
    side of it, along each column of the covariance's *Cholesky factor*: the
    lower-triangular L with :math:`L L^T = P`, a kind of square root of P. All the points are
    handled together as one ``(2 n_state + 1, n_state)`` array.

    Parameters
    ----------
    alpha : float, default 1.0
        :math:`\alpha`, the scaling parameter of the scaled unscented transformation [3]_:
        how far the points spread from the mean. Must be positive, and close enough to 1
        that :math:`W^{(c)}_0 \ge 0`: with :math:`\beta = 2` and :math:`\kappa = 0`, between
        about 0.52 and 1.93. See Notes for why the default is 1.
    beta : float, default 2.0
        :math:`\beta`, prior knowledge of the distribution. 2 is the best choice for a
        Gaussian. Must not be negative.
    kappa : float, default 0.0
        :math:`\kappa`, a second spread adjustment. ``n_state + kappa`` must be positive.

    Notes
    -----
    With :math:`c = \alpha^2 (n + \kappa)`, the weights are [2]_

    .. math::

        W^{(m)}_0 = 1 - \frac{n}{c}, \quad
        W^{(c)}_0 = W^{(m)}_0 + 1 - \alpha^2 + \beta, \quad
        W^{(m)}_i = W^{(c)}_i = \frac{1}{2c}, \; i = 1 \ldots 2n.

    **Why every covariance weight must be non-negative.** The filter's covariance update for
    this linearisation is the short form :math:`P^- - K S K^T`, which is guaranteed positive
    semidefinite when every *covariance* weight is non-negative. So settings with
    :math:`W^{(c)}_0 < 0` are rejected when a filter is built
    (:func:`~radar_forge.core.tracking._validation.check_sigma_point_settings`). The mean
    weight :math:`W^{(m)}_0` may be negative. With the defaults, :math:`W^{(m)}_0 = 0`,
    :math:`W^{(c)}_0 = 2` and :math:`W_i = 1/(2n)` for any n. The heuristic
    :math:`\kappa = 3 - n` of Julier and Uhlmann is accepted up to n = 9. A small
    :math:`\alpha`, such as the :math:`10^{-3}` Wan and van der Merwe call usual, makes
    :math:`W^{(c)}_0` large and negative, so it is rejected, on purpose.

    **Why alpha = 1 and kappa = 0.** With :math:`\alpha = 1` the transform is the plain,
    unscaled one: the points sit at :math:`\pm\sqrt{n + \kappa}` along each column of L. A
    small :math:`\alpha` pulls them in, which helps when :math:`n` is large enough that
    :math:`\sqrt{n}` standard deviations reaches far into a nonlinear model. Here
    :math:`n \le 9` (3-D CA), so the points are at most 3 standard deviations out.

    **When the UKF is the KF.** On a linear motion model and a linear measurement model,
    with no periodic state coordinate, this linearisation returns exactly the moments of the
    KF, for any accepted :math:`\alpha`, :math:`\beta` and :math:`\kappa`. Let :math:`L_i`
    be column i of L, so that :math:`\sum_i L_i L_i^T = P`.

    1. *The points.* :math:`\chi_0 = \hat{x}`, and :math:`\chi_{\pm i} = \hat{x} \pm
       \sqrt{c} L_i` for i = 1 … n.
    2. *The mean.* The mean weights sum to one: :math:`W^{(m)}_0 + 2n / (2c) = 1`. Through a
       linear map F the ± pairs cancel, so :math:`\sum W^{(m)} F \chi = F \hat{x}`, the KF's
       mean.
    3. *The covariance.* The centre point's deviation :math:`F \chi_0 - F \hat{x}` is zero,
       so :math:`W^{(c)}_0` multiplies zero. That removes the whole of :math:`\beta`. The
       other 2n points give :math:`\sum_i 2 \cdot \frac{1}{2c} \cdot c\, F L_i L_i^T F^T
       = F P F^T`. Adding Q gives the KF's :math:`P^-`.
    4. *The update.* The same steps with H in place of F give :math:`\hat{z} = H
       \hat{x}^-`, :math:`P_{xz} = P^- H^T` and :math:`S = H P^- H^T + R`.
    5. *The covariance form.* What is left is that this filter uses the short form and the
       KF the Joseph form. The two are equal at the optimal gain
       (:mod:`~radar_forge.core.tracking.estimation`).

    So on linear models the UKF *is* the KF, up to roundoff. The weights matter only once
    the function bends, as the bistatic range and Doppler model does.

    Periodic coordinates are averaged with :meth:`StateLayout.weighted_mean` and differenced
    with :meth:`StateLayout.residual`, so sigma points that straddle a wrap (an azimuth
    near ±π) are taken the short way round.

    References
    ----------
    .. [1] Särkkä (2013), ch. 5 and ch. 6; see the module References.
    .. [2] Wan and van der Merwe (2000), §3, Eq. (15).
    .. [3] Julier (2002).

    Examples
    --------
    >>> linearisation = UnscentedLinearisation()
    >>> wm, wc = linearisation.weights(2)
    >>> wm, wc
    (array([0.  , 0.25, 0.25, 0.25, 0.25]), array([2.  , 0.25, 0.25, 0.25, 0.25]))
    """

    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0

    def check_motion_model(self, motion_model: MotionModel) -> None:
        """Raise unless the sigma-point settings are valid for the model's state size.

        Parameters
        ----------
        motion_model : MotionModel
            The motion model the filter is being built with.

        Raises
        ------
        ValueError
            As :func:`~radar_forge.core.tracking._validation.check_sigma_point_settings`.
        """
        check_sigma_point_settings(
            self.alpha, self.beta, self.kappa, motion_model.state_layout.dimension
        )

    def _scale(self, n_state: int) -> float:
        r"""Return :math:`c = \alpha^2 (n + \kappa)`, the squared spread of the points."""
        return self.alpha**2 * (n_state + self.kappa)

    def weights(self, n_state: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the mean and covariance weights for a state of ``n_state`` coordinates.

        Parameters
        ----------
        n_state : int
            Number of state coordinates.

        Returns
        -------
        wm : numpy.ndarray
            Mean weights, shape ``(2 n_state + 1,)``. They sum to one.
        wc : numpy.ndarray
            Covariance weights, shape ``(2 n_state + 1,)``.
        """
        scale = self._scale(n_state)
        wm = np.full(2 * n_state + 1, 1 / (2 * scale), dtype=np.float64)
        wm[0] = 1 - n_state / scale
        wc = wm.copy()
        wc[0] += 1 - self.alpha**2 + self.beta
        return wm, wc

    def sigma_points(self, state: StateEstimate) -> NDArray[np.float64]:
        r"""Return the sigma points of an estimate, shape ``(2 n_state + 1, n_state)``.

        Row 0 is the mean. Rows 1 to n are the mean plus :math:`\sqrt{c}` times each column
        of the covariance's Cholesky factor, and rows n + 1 to 2n the mean minus them. Each
        periodic coordinate is wrapped into [-period/2, period/2).

        Parameters
        ----------
        state : StateEstimate
            The estimate: mean ``(n_state,)`` and covariance ``(n_state, n_state)``.

        Returns
        -------
        numpy.ndarray
            The points, one per row.
        """
        n_state = len(state.mean)
        # Row i of Lᵀ is column i of L, so each row is one offset.
        offsets = np.sqrt(self._scale(n_state)) * cholesky_factor(state.covariance).T
        deviations = np.concatenate([np.zeros((1, n_state)), offsets, -offsets])
        return state.state_layout.wrap(state.mean + deviations)

    def predict(
        self, motion_model: MotionModel, state: StateEstimate, dt_s: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Return the predicted mean and covariance, from the moved sigma points.

        Parameters
        ----------
        motion_model : MotionModel
            Moves states forward in time and gives the process noise Q.
        state : StateEstimate
            The estimate to move: mean ``(n_state,)`` and covariance ``(n_state, n_state)``.
        dt_s : float
            The time step, in seconds.

        Returns
        -------
        mean : numpy.ndarray
            :math:`\hat{x}^-`, shape ``(n_state,)``.
        covariance : numpy.ndarray
            :math:`P^-`, shape ``(n_state, n_state)``, with Q added.

        Raises
        ------
        ValueError
            If the motion model returns arrays of the wrong shape, or a process noise that
            is not a covariance.

        Notes
        -----
        Each sigma point :math:`\mathcal{X}_i` is moved by the motion model to
        :math:`\mathcal{X}'_i`. Then [2]_

        .. math::

            \hat{x}^- = \sum_i W^{(m)}_i \mathcal{X}'_i, \qquad
            P^- = \sum_i W^{(c)}_i (\mathcal{X}'_i - \hat{x}^-)(\mathcal{X}'_i -
            \hat{x}^-)^T + Q.

        References
        ----------
        .. [2] Wan and van der Merwe (2000), §3.
        """
        layout = state.state_layout
        wm, wc = self.weights(len(state.mean))
        points = as_points(
            motion_model.transition(self.sigma_points(state), dt_s),
            len(wm),
            len(state.mean),
            "transition",
        )
        noise = as_covariance(motion_model.process_noise(state.mean, dt_s), len(state.mean))

        mean = layout.weighted_mean(points, wm)
        deviations = layout.residual(points, mean)
        covariance = (deviations.T * wc) @ deviations + noise
        return mean, symmetrise(covariance)

    def measure(self, model: MeasurementModel, state: StateEstimate) -> MeasurementMoments:
        r"""Return the measurement moments, from the sigma points pushed through ``model``.

        Parameters
        ----------
        model : MeasurementModel
            How the state maps to a measurement. Its state layout is the state's.
        state : StateEstimate
            The predicted estimate.

        Returns
        -------
        MeasurementMoments
            :math:`\hat{z} = \sum W^{(m)}_i h(\mathcal{X}_i)`,
            :math:`P_{xz} = \sum W^{(c)}_i (\mathcal{X}_i - \hat{x})(h(\mathcal{X}_i) -
            \hat{z})^T`, the predicted measurement's covariance
            :math:`\sum W^{(c)}_i (h(\mathcal{X}_i) - \hat{z})(h(\mathcal{X}_i) -
            \hat{z})^T`, and no measurement matrix.

        Raises
        ------
        ValueError
            If the model returns an array of the wrong shape.

        Notes
        -----
        The sigma points are drawn from the predicted estimate, so the process noise Q is
        part of every moment here [2]_.

        References
        ----------
        .. [2] Wan and van der Merwe (2000), §3.
        """
        points = self.sigma_points(state)
        wm, wc = self.weights(len(state.mean))
        layout = model.measurement_layout
        predicted_points = as_points(
            model.predict(points), len(wm), layout.dimension, "predicted measurement"
        )
        predicted = layout.weighted_mean(predicted_points, wm)
        dz = layout.residual(predicted_points, predicted)
        dx = state.state_layout.residual(points, state.mean)
        return MeasurementMoments(
            predicted_measurement=predicted,
            cross_covariance=(dx.T * wc) @ dz,
            predicted_measurement_covariance=(dz.T * wc) @ dz,
        )
