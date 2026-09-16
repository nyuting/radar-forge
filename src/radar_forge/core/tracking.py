r"""Kalman filtering, validation gating, assignment and track management.

This module is the whole of the tracking stack specified by
``spec/scenario-003-tracking.md``: a linear Kalman filter over a
constant-velocity motion model, a chi-squared validation gate, global
nearest-neighbour assignment, and an M-of-N track manager. It is deliberately
one module -- ``spec/structure.md`` D2 keeps it unsplit until a *second*
association strategy (JPDA, MHT) or a fusion layer arrives, and three state
models are not that.

The decomposition follows Stone Soup's vocabulary at a fraction of its surface
area: :func:`predict` and :func:`update` are the predictor and updater,
:func:`normalised_innovation_squared` and :func:`gate_threshold` are the gater,
:func:`associate_gnn` is the data associator, and :class:`TrackManager` is the
initiator/deleter loop. The state models are *data* -- :func:`state_model_matrices`
returns the matrices and callables for a model rather than a subclass -- so
adding one is a table entry, not a class hierarchy.

Two conventions are load-bearing and easy to get wrong:

* **Range rate is positive closing**, per ``spec/structure.md`` D5 and the sign
  of :func:`radar_forge.core.dsp.doppler_bin_centers_mps`. A tracker that flips
  it produces closing targets that open, and the plot still looks plausible.
* **The filter's time step is the frame interval, not the CPI length.** See the
  ``Notes`` of :func:`state_model_matrices`.

This module knows nothing about range-Doppler maps, Doppler ambiguity or where
a measurement came from. Detection, velocity unfolding and the simulated angle
measurement are all :mod:`radar_forge.pipelines.tracking`'s business; the
contract here is "you give me measurements and a model".

References
----------
.. [1] Y. Bar-Shalom, P. K. Willett and X. Tian, *Tracking and Data Fusion: A
       Handbook of Algorithms*, YBS Publishing, 2011, ch. 2 (gating), ch. 3
       (assignment).
.. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with
       Applications to Tracking and Navigation*, Wiley, 2001, §5.2 (discrete
       white-noise acceleration), §6.3 (validation gating and nearest
       neighbour), §11.7 (track initiation).
.. [3] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking
       Systems*, Artech House, 1999, ch. 6 (M-of-N initiation, global nearest
       neighbour).
.. [4] D. F. Crouse, "On implementing 2D rectangular assignment algorithms,"
       *IEEE Trans. Aerosp. Electron. Syst.*, vol. 52, no. 4, pp. 1679-1696,
       2016. The algorithm behind ``scipy.optimize.linear_sum_assignment``.
.. [5] R. R. Labbe, *Kalman and Bayesian Filters in Python*, 2020. The FilterPy
       companion text; the source of the ``Q_discrete_white_noise`` formulation
       reimplemented in :func:`process_noise_dwna`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "STATE_MODELS",
    "TRACK_STATUSES",
    "FrameResult",
    "KalmanState",
    "StateModel",
    "Track",
    "TrackManager",
    "TrackModel",
    "TrackStatus",
    "UpdateResult",
    "associate_gnn",
    "gate_threshold",
    "innovation_of",
    "normalised_innovation_squared",
    "predict",
    "process_noise_dwna",
    "state_model_matrices",
    "update",
]

StateModel = Literal["range_1d", "enu_2d", "enu_3d"]
"""The state vectors this module can build."""

TrackStatus = Literal["tentative", "confirmed", "coasting", "deleted"]

STATE_MODELS: tuple[StateModel, ...] = get_args(StateModel)
TRACK_STATUSES: tuple[TrackStatus, ...] = get_args(TrackStatus)

# An unassociated pair must be impossible for the assignment solver to pick, but
# scipy rejects a cost matrix containing inf, so the forbidden cost has to be a
# large finite number instead. Any value above the largest admissible d^2 works;
# this one is far above every chi-squared quantile and still squares without
# overflowing float64.
_FORBIDDEN_COST = 1e12


@dataclass(frozen=True)
class KalmanState:
    """A Gaussian state estimate: a mean vector and its covariance.

    Attributes
    ----------
    state : ndarray
        Shape ``(n_state,)``. The state vector, in the order the
        :data:`StateModel` names -- positions first, then their rates.
    covariance : ndarray
        Shape ``(n_state, n_state)``. The estimate error covariance.
    """

    state: NDArray[np.float64]
    covariance: NDArray[np.float64]

    @property
    def n_state(self) -> int:
        """Length of the state vector."""
        return int(self.state.shape[0])


@dataclass(frozen=True)
class TrackModel:
    r"""Everything the filter needs for one state model at one time step.

    This is the "state models are data" of the specification's §6.2: a model is
    a record of matrices and callables, not a subclass, so :func:`predict` and
    :func:`update` have exactly one code path each.

    Attributes
    ----------
    transition : ndarray
        Shape ``(n_state, n_state)``. The state transition matrix :math:`F`.
    process_noise : ndarray
        Shape ``(n_state, n_state)``. The process noise covariance :math:`Q`.
    measurement_function : callable
        Maps a state of shape ``(n_state,)`` to a predicted measurement of shape
        ``(measurement_dim,)`` -- the :math:`h(x)` of the extended filter. For a
        linear model this is simply :math:`Hx`.
    measurement_jacobian : callable
        Maps a state of shape ``(n_state,)`` to the Jacobian
        :math:`\partial h/\partial x`, shape ``(measurement_dim, n_state)``. For
        a linear model it ignores its argument and returns :math:`H`.
    measurement_noise : ndarray
        Shape ``(measurement_dim, measurement_dim)``. The measurement noise
        covariance :math:`R`.
    measurement_dim : int
        Number of measurement components, ``z.size``.
    angle_rows : tuple of int
        Indices of the measurement components that are angles in radians, whose
        innovation must be wrapped to :math:`[-\pi, \pi)`. Empty for a model
        that measures no angle.
    """

    transition: NDArray[np.float64]
    process_noise: NDArray[np.float64]
    measurement_function: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    measurement_jacobian: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    measurement_noise: NDArray[np.float64]
    measurement_dim: int
    angle_rows: tuple[int, ...] = ()

    def restricted(self, n_rows: int) -> TrackModel:
        """Return this model measuring only its first ``n_rows`` components.

        This is how the specification's §5.3 bootstrap is expressed without a
        second code path. A track that cannot yet unfold its Doppler measures
        range alone, so it uses the same model with one measurement row; once
        its covariance says the fold is resolvable it uses the full one. The
        gate dimension follows automatically, because it is read from
        :attr:`measurement_dim`.

        Parameters
        ----------
        n_rows : int
            Number of leading measurement components to keep, ``1 <= n_rows <=
            measurement_dim``.

        Returns
        -------
        TrackModel
            The restricted model. Returns ``self`` when nothing is dropped.

        Raises
        ------
        ValueError
            If ``n_rows`` is outside ``1 .. measurement_dim``.
        """
        if not 1 <= n_rows <= self.measurement_dim:
            msg = (
                f"n_rows must be between 1 and measurement_dim "
                f"({self.measurement_dim}); got {n_rows}."
            )
            raise ValueError(msg)
        if n_rows == self.measurement_dim:
            return self

        rows = slice(0, n_rows)
        measurement_function = self.measurement_function
        measurement_jacobian = self.measurement_jacobian
        return replace(
            self,
            measurement_function=lambda x: measurement_function(x)[rows],
            measurement_jacobian=lambda x: measurement_jacobian(x)[rows, :],
            measurement_noise=self.measurement_noise[rows, rows],
            measurement_dim=n_rows,
            angle_rows=tuple(row for row in self.angle_rows if row < n_rows),
        )


@dataclass(frozen=True)
class UpdateResult:
    r"""The outcome of one measurement update.

    Attributes
    ----------
    posterior : KalmanState
        The corrected estimate.
    innovation : ndarray
        Shape ``(measurement_dim,)``. The measurement residual
        :math:`\nu = z - h(\hat{x})`, with any angle component wrapped.
    innovation_covariance : ndarray
        Shape ``(measurement_dim, measurement_dim)``. :math:`S = HPH^T + R`.
    nis : float
        The normalised innovation squared :math:`\nu^T S^{-1} \nu`, the
        statistic the gate tests and the consistency criterion averages.
    """

    posterior: KalmanState
    innovation: NDArray[np.float64]
    innovation_covariance: NDArray[np.float64]
    nis: float


def process_noise_dwna(
    frame_time_s: float,
    sigma_accel_mps2: float,
    n_axes: int = 1,
) -> NDArray[np.float64]:
    r"""Return the discrete white-noise acceleration process noise covariance.

    Implements the piecewise-constant white acceleration model of [2]_ §5.2, in
    which the acceleration is taken as constant over each sampling interval and
    independent between intervals. Per axis, with :math:`T` the step and
    :math:`\sigma_a` the acceleration standard deviation,

    .. math::

        Q_{\text{axis}} = \sigma_a^2
        \begin{bmatrix} T^4/4 & T^3/2 \\ T^3/2 & T^2 \end{bmatrix}

    Axes are independent, so the full matrix carries one such block per axis and
    no cross-axis terms. The state is ordered positions first and then rates --
    :math:`[e, n, \dot{e}, \dot{n}]`, not :math:`[e, \dot{e}, n, \dot{n}]` --
    so the four scalars above become four :math:`n_{\text{axes}}`-square blocks.

    Parameters
    ----------
    frame_time_s : float
        The time step :math:`T`, in seconds. This is the *frame* interval, not
        the CPI length; see :func:`state_model_matrices`.
    sigma_accel_mps2 : float
        Acceleration standard deviation :math:`\sigma_a`, in metres per second
        squared.
    n_axes : int, optional
        Number of independent spatial axes, default 1.

    Returns
    -------
    ndarray
        Shape ``(2 * n_axes, 2 * n_axes)``, symmetric positive semi-definite.

    Raises
    ------
    ValueError
        If ``frame_time_s`` is not positive, ``sigma_accel_mps2`` is negative,
        or ``n_axes`` is not positive.

    Notes
    -----
    DWNA is chosen over continuous white noise because it is the form every
    reference here states, and because a single acceleration standard deviation
    in m/s^2 is a quantity a reader can reason about physically. The matrix is
    singular by construction: one scalar noise per axis drives two states, so
    each block has rank 1.

    Examples
    --------
    >>> q = process_noise_dwna(1.0, 2.0)
    >>> bool(np.isclose(q[0, 0], 1.0))
    True
    >>> bool(np.isclose(q[1, 1], 4.0))
    True
    >>> bool(np.isclose(np.linalg.matrix_rank(q), 1))
    True
    """
    if frame_time_s <= 0.0:
        msg = f"frame_time_s must be positive, in seconds; got {frame_time_s}."
        raise ValueError(msg)
    if sigma_accel_mps2 < 0.0:
        msg = f"sigma_accel_mps2 must be non-negative; got {sigma_accel_mps2}."
        raise ValueError(msg)
    if n_axes < 1:
        msg = f"n_axes must be a positive number of spatial axes; got {n_axes}."
        raise ValueError(msg)

    step_s = float(frame_time_s)
    variance = float(sigma_accel_mps2) ** 2
    identity = np.eye(n_axes, dtype=np.float64)

    position_position = variance * step_s**4 / 4.0
    position_rate = variance * step_s**3 / 2.0
    rate_rate = variance * step_s**2

    result: NDArray[np.float64] = np.block(
        [
            [position_position * identity, position_rate * identity],
            [position_rate * identity, rate_rate * identity],
        ]
    )
    return result


def _constant_velocity_transition(frame_time_s: float, n_axes: int) -> NDArray[np.float64]:
    """Build the constant-velocity transition, positions first then rates."""
    identity = np.eye(n_axes, dtype=np.float64)
    zeros = np.zeros((n_axes, n_axes), dtype=np.float64)
    return np.block([[identity, float(frame_time_s) * identity], [zeros, identity]])


def state_model_matrices(
    state_model: StateModel,
    frame_time_s: float,
    *,
    sigma_accel_mps2: float,
    sigma_range_m: float,
    sigma_velocity_mps: float,
    sigma_azimuth_deg: float = 0.5,
    sigma_elevation_deg: float = 0.5,
) -> TrackModel:
    r"""Return the transition, process noise and measurement model for a state model.

    The returned :class:`TrackModel` is plain data, per the specification's
    §6.2: a state model is a record, not a subclass, and one :func:`predict` and
    one :func:`update` serve every model.

    ``range_1d`` carries the state :math:`[r, \dot{r}]` and measures it
    directly, so :math:`h(x) = x` and the Jacobian is the identity. Its range
    rate is positive **closing**, which is why its transition carries
    :math:`-T`; see the comment in the body. It is the
    only state model this radar's own measurements make observable -- with no
    angle, a Cartesian state lies somewhere on a circle of unknown bearing --
    which is why it is the default and the one the acceptance criteria are
    stated on.

    ``enu_2d`` and ``enu_3d`` carry Cartesian states in the local ENU frame of
    :mod:`radar_forge.core.geodesy`, positions first and then their rates, and
    measure

    .. math::

        h(x) = \bigl[\lVert p \rVert,\;
        \operatorname{atan2}(e, n),\;
        -\,p \cdot v / \lVert p \rVert \bigr]

    plus an elevation row for ``enu_3d``. Azimuth is zero at true north and
    increases clockwise -- :math:`\operatorname{atan2}(e, n)`, not the
    mathematical convention -- matching
    :func:`radar_forge.core.geodesy.enu_to_range_azimuth_elevation`, and the
    range-rate row carries a minus sign because the measurement is positive
    closing while :math:`p \cdot v / \lVert p \rVert` is positive opening.
    These models are what a real tracker uses, but they are **not observable**
    from this radar's own measurements; §6.3's simulated angle is what makes
    them run at all, and :mod:`radar_forge.pipelines.tracking` supplies it.

    Parameters
    ----------
    state_model : {'range_1d', 'enu_2d', 'enu_3d'}
        Which state vector to build; one of :data:`STATE_MODELS`. ``range_1d``
        is the default choice for this radar, for the observability reason
        above.
    frame_time_s : float
        The filter time step :math:`T`, in seconds.
    sigma_accel_mps2 : float
        Acceleration standard deviation for the process noise, in m/s^2.
    sigma_range_m : float
        Range measurement standard deviation, in metres.
    sigma_velocity_mps : float
        Range-rate measurement standard deviation, in metres per second.
    sigma_azimuth_deg, sigma_elevation_deg : float, optional
        Angle measurement standard deviations, in degrees, default 0.5 each.
        Used only by the ENU models. This radar measures no angle at all, so
        these describe the *simulated* angle measurement of §6.3 rather than any
        property of the radar; 0.5 degrees is a plausible monopulse accuracy for
        the 5.1 degree beam implied by 30 dBi, and far worse than the 0.008
        degrees the SNR alone would suggest, because the real error budget is
        boresight calibration and not thermal noise.

    Returns
    -------
    TrackModel
        The matrices and callables for this model at this time step.

    Raises
    ------
    ValueError
        If ``state_model`` is not a known model, or any standard deviation is
        negative.

    Notes
    -----
    **The time step is the frame interval, not the CPI length.** Scenario 003
    forms a 256 ms coherent processing interval once per 1 s frame and
    timestamps the resulting measurement at frame time. The filter therefore
    steps by 1.0 s, and the 744 ms in which the radar is not looking is absorbed
    into the process noise rather than modelled. This is the standard
    approximation and it is named here because it is invisible at the call site.

    The measurement noise is diagonal, and its two variances differ by five
    orders of magnitude: a range-Doppler radar has a coarse ranging measurement
    and an exquisite velocity one. That is not a mistake, and it is what makes a
    transposed :math:`R` so easy to spot in the consistency statistic.

    References
    ----------
    .. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with
           Applications to Tracking and Navigation*, Wiley, 2001, §5.2.
    """
    if state_model not in STATE_MODELS:
        msg = f"state_model must be one of {list(STATE_MODELS)}; got {state_model!r}."
        raise ValueError(msg)
    for name, value in (
        ("sigma_range_m", sigma_range_m),
        ("sigma_velocity_mps", sigma_velocity_mps),
    ):
        if value < 0.0:
            msg = f"{name} must be non-negative; got {value}."
            raise ValueError(msg)

    for name, value in (
        ("sigma_azimuth_deg", sigma_azimuth_deg),
        ("sigma_elevation_deg", sigma_elevation_deg),
    ):
        if value < 0.0:
            msg = f"{name} must be non-negative; got {value}."
            raise ValueError(msg)

    if state_model != "range_1d":
        return _enu_model(
            state_model,
            frame_time_s,
            sigma_accel_mps2=sigma_accel_mps2,
            sigma_range_m=sigma_range_m,
            sigma_velocity_mps=sigma_velocity_mps,
            sigma_azimuth_deg=sigma_azimuth_deg,
            sigma_elevation_deg=sigma_elevation_deg,
        )

    n_axes = 1
    # The state is [range, range rate] with range rate positive *closing*, per
    # spec/structure.md D5 and the sign of dsp.doppler_bin_centers_mps. Closing
    # therefore *reduces* range, so the transition carries -T rather than +T,
    # and the position-rate cross terms of Q change sign with it. Conjugating
    # the textbook constant-velocity pair by diag(1, -1) does both at once, and
    # leaves process_noise_dwna in the form the references state.
    #
    # Getting this wrong is not a tuning problem: the filter dead-reckons the
    # target the wrong way down the line of sight, every fold selection made
    # from its prediction is wrong, and the range track still looks roughly
    # right for a few frames, which is what makes it worth a comment.
    closing = np.diag(np.asarray([1.0, -1.0], dtype=np.float64))
    transition = closing @ _constant_velocity_transition(frame_time_s, n_axes) @ closing
    process_noise = closing @ process_noise_dwna(frame_time_s, sigma_accel_mps2, n_axes) @ closing

    # Slant range and range rate are measured directly, so h is the identity on
    # the whole state and the "extended" path collapses to the linear one.
    jacobian = np.eye(2, dtype=np.float64)
    measurement_noise = np.diag(
        np.asarray([sigma_range_m**2, sigma_velocity_mps**2], dtype=np.float64)
    )

    return TrackModel(
        transition=transition,
        process_noise=process_noise,
        measurement_function=lambda x: jacobian @ x,
        measurement_jacobian=lambda _state: jacobian,
        measurement_noise=measurement_noise,
        measurement_dim=2,
        angle_rows=(),
    )


def _enu_model(
    state_model: StateModel,
    frame_time_s: float,
    *,
    sigma_accel_mps2: float,
    sigma_range_m: float,
    sigma_velocity_mps: float,
    sigma_azimuth_deg: float,
    sigma_elevation_deg: float,
) -> TrackModel:
    """Build the ENU Cartesian models; see :func:`state_model_matrices`."""
    n_axes = 3 if state_model == "enu_3d" else 2
    measures_elevation = state_model == "enu_3d"

    def measurement_function(state: NDArray[np.float64]) -> NDArray[np.float64]:
        position = state[:n_axes]
        velocity = state[n_axes:]
        range_m = float(np.linalg.norm(position))
        # atan2(east, north): zero at true north, increasing clockwise, per
        # spec/structure.md D5 and core.geodesy. The mathematical convention
        # would put zero at east and turn the other way, which produces a track
        # that mirrors the truth and still looks like a track.
        azimuth_rad = float(np.arctan2(position[0], position[1]))
        # Negated: p.v / |p| is the *opening* rate, and the measurement is
        # positive closing.
        closing_mps = -float(position @ velocity) / range_m

        rows = [range_m, azimuth_rad]
        if measures_elevation:
            ground_m = float(np.hypot(position[0], position[1]))
            rows.append(float(np.arctan2(position[2], ground_m)))
        rows.append(closing_mps)
        return np.asarray(rows, dtype=np.float64)

    def measurement_jacobian(state: NDArray[np.float64]) -> NDArray[np.float64]:
        position = state[:n_axes]
        velocity = state[n_axes:]
        range_m = float(np.linalg.norm(position))
        ground_m = float(np.hypot(position[0], position[1]))
        opening_mps = float(position @ velocity) / range_m

        n_rows = 4 if measures_elevation else 3
        jacobian = np.zeros((n_rows, 2 * n_axes), dtype=np.float64)

        # Range: d|p|/dp = p / |p|.
        jacobian[0, :n_axes] = position / range_m

        # Azimuth = atan2(e, n): d/de = n / (e^2 + n^2), d/dn = -e / (e^2 + n^2).
        jacobian[1, 0] = position[1] / ground_m**2
        jacobian[1, 1] = -position[0] / ground_m**2

        rate_row = 2
        if measures_elevation:
            # Elevation = atan2(u, sqrt(e^2 + n^2)).
            jacobian[2, 0] = -position[2] * position[0] / (range_m**2 * ground_m)
            jacobian[2, 1] = -position[2] * position[1] / (range_m**2 * ground_m)
            jacobian[2, 2] = ground_m / range_m**2
            rate_row = 3

        # Closing rate = -(p.v)/|p|, so d/dp = -(v/|p| - (p.v) p / |p|^3) and
        # d/dv = -p/|p|.
        jacobian[rate_row, :n_axes] = -(velocity / range_m - opening_mps * position / range_m**2)
        jacobian[rate_row, n_axes:] = -position / range_m
        return jacobian

    variances = [sigma_range_m**2, np.deg2rad(sigma_azimuth_deg) ** 2]
    if measures_elevation:
        variances.append(np.deg2rad(sigma_elevation_deg) ** 2)
    variances.append(sigma_velocity_mps**2)

    return TrackModel(
        transition=_constant_velocity_transition(frame_time_s, n_axes),
        process_noise=process_noise_dwna(frame_time_s, sigma_accel_mps2, n_axes),
        measurement_function=measurement_function,
        measurement_jacobian=measurement_jacobian,
        measurement_noise=np.diag(np.asarray(variances, dtype=np.float64)),
        measurement_dim=len(variances),
        angle_rows=(1, 2) if measures_elevation else (1,),
    )


def predict(state: KalmanState, model: TrackModel) -> KalmanState:
    r"""Advance a state estimate one time step, with no measurement.

    The textbook Kalman prediction [2]_,

    .. math::

        \hat{x}^- = F\hat{x}, \qquad P^- = F P F^T + Q

    Parameters
    ----------
    state : KalmanState
        The estimate at the previous step.
    model : TrackModel
        Supplies :math:`F` and :math:`Q`.

    Returns
    -------
    KalmanState
        The predicted estimate. A coasting track is exactly this, applied
        repeatedly with no update in between, and its covariance grows without
        bound -- which is what eventually pushes a real measurement back inside
        its gate, and why deletion is counted in misses rather than in seconds.

    Examples
    --------
    Range rate is positive *closing*, so one second of dead reckoning at
    5 m/s takes a 100 m target to 95 m, not to 105 m -- the transition of
    ``range_1d`` carries :math:`-T`. See :func:`state_model_matrices`.

    >>> model = state_model_matrices(
    ...     "range_1d", 1.0, sigma_accel_mps2=0.0,
    ...     sigma_range_m=1.0, sigma_velocity_mps=1.0)
    >>> prior = KalmanState(np.array([100.0, 5.0]), np.eye(2))
    >>> bool(np.isclose(predict(prior, model).state[0], 95.0))
    True
    """
    transition = model.transition
    mean: NDArray[np.float64] = transition @ state.state
    covariance: NDArray[np.float64] = (
        transition @ state.covariance @ transition.T + model.process_noise
    )
    return KalmanState(state=mean, covariance=covariance)


def _wrap_to_pi(angle_rad: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap an angle in radians to [-pi, pi)."""
    wrapped: NDArray[np.float64] = (angle_rad + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped


def innovation_of(
    state: KalmanState,
    measurement: ArrayLike,
    model: TrackModel,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Return the innovation, its covariance and the Jacobian, for one measurement.

    Separated from :func:`update` because gating needs exactly this and nothing
    else: a measurement is tested against a track before it is used, and most
    tested pairs are never updated.

    Parameters
    ----------
    state : KalmanState
        The *predicted* estimate, from :func:`predict`.
    measurement : array_like
        Shape ``(measurement_dim,)``. The measurement :math:`z`.
    model : TrackModel
        Supplies :math:`h`, its Jacobian and :math:`R`.

    Returns
    -------
    innovation : ndarray
        Shape ``(measurement_dim,)``. :math:`\nu = z - h(\hat{x}^-)`, with any
        component listed in ``model.angle_rows`` wrapped to :math:`[-\pi, \pi)`
        so that a target near due north does not produce a residual of nearly
        :math:`2\pi`.
    innovation_covariance : ndarray
        Shape ``(measurement_dim, measurement_dim)``. :math:`S = HPH^T + R`.
    jacobian : ndarray
        Shape ``(measurement_dim, n_state)``. :math:`H` at the predicted state.

    Raises
    ------
    ValueError
        If ``measurement`` does not have shape ``(measurement_dim,)``.
    """
    z = np.asarray(measurement, dtype=np.float64)
    if z.shape != (model.measurement_dim,):
        msg = (
            f"measurement must have shape ({model.measurement_dim},) to match the "
            f"model's measurement_dim; got {z.shape}."
        )
        raise ValueError(msg)

    jacobian = model.measurement_jacobian(state.state)
    innovation = z - model.measurement_function(state.state)
    if model.angle_rows:
        rows = list(model.angle_rows)
        innovation[rows] = _wrap_to_pi(innovation[rows])
    innovation_covariance = jacobian @ state.covariance @ jacobian.T + model.measurement_noise
    return innovation, innovation_covariance, jacobian


def normalised_innovation_squared(
    innovation: ArrayLike,
    innovation_covariance: ArrayLike,
) -> float:
    r"""Return the normalised innovation squared, :math:`\nu^T S^{-1} \nu`.

    The statistic validation gating tests and the consistency criterion
    averages [2]_ §6.3. Under a correct filter it is chi-squared distributed
    with ``measurement_dim`` degrees of freedom, so its mean over many frames
    should be the measurement dimension itself -- a check that catches a
    mis-scaled :math:`R` or :math:`Q` that no single-frame assertion would.

    Parameters
    ----------
    innovation : array_like
        Shape ``(measurement_dim,)``. The residual :math:`\nu`.
    innovation_covariance : array_like
        Shape ``(measurement_dim, measurement_dim)``. The covariance :math:`S`.

    Returns
    -------
    float
        The scalar :math:`d^2`. Never negative for a positive-definite
        :math:`S`.

    Raises
    ------
    ValueError
        If the shapes disagree, or :math:`S` is singular.

    Examples
    --------
    >>> float(normalised_innovation_squared([3.0], [[9.0]]))
    1.0
    """
    nu = np.asarray(innovation, dtype=np.float64)
    covariance = np.atleast_2d(np.asarray(innovation_covariance, dtype=np.float64))
    if nu.ndim != 1 or covariance.shape != (nu.size, nu.size):
        msg = (
            f"innovation of shape {nu.shape} and innovation_covariance of shape "
            f"{covariance.shape} do not agree; expected (n,) and (n, n)."
        )
        raise ValueError(msg)

    try:
        solved = np.linalg.solve(covariance, nu)
    except np.linalg.LinAlgError as error:
        msg = (
            "innovation_covariance is singular, so the normalised innovation is "
            "undefined. This usually means a measurement noise of zero, or a "
            "state model whose Jacobian has a zero row."
        )
        raise ValueError(msg) from error
    return float(nu @ solved)


def gate_threshold(gate_probability: float, dim: int) -> float:
    r"""Return the chi-squared gate threshold for a measurement dimension.

    A measurement is validated when :math:`d^2 \le \chi^2_{\text{dim}}(P_G)`
    [2]_ §6.3. The gate is a statement about the filter's own uncertainty, not a
    fixed number of metres, and neither its width nor its *dimension* may be
    hard-coded: the dimension varies within a single run of scenario 003,
    because a track starts range-only and is promoted to range-and-rate once its
    Doppler fold becomes resolvable.

    Parameters
    ----------
    gate_probability : float
        The probability mass the gate admits, :math:`P_G`, in ``(0, 1)``.
    dim : int
        Measurement dimension, the chi-squared degrees of freedom.

    Returns
    -------
    float
        The threshold on :math:`d^2`.

    Raises
    ------
    ValueError
        If ``gate_probability`` is outside ``(0, 1)`` or ``dim`` is not
        positive.

    Examples
    --------
    >>> bool(np.isclose(gate_threshold(0.99, 2), 9.210, atol=5e-4))
    True
    """
    if not 0.0 < gate_probability < 1.0:
        msg = f"gate_probability must lie strictly between 0 and 1; got {gate_probability}."
        raise ValueError(msg)
    if dim < 1:
        msg = f"dim must be a positive measurement dimension; got {dim}."
        raise ValueError(msg)
    return float(chi2.ppf(gate_probability, dim))


def update(
    state: KalmanState,
    measurement: ArrayLike,
    model: TrackModel,
) -> UpdateResult:
    r"""Correct a predicted estimate with one measurement.

    The Kalman update in its Joseph-stabilised form [2]_,

    .. math::

        K = P^- H^T S^{-1}, \qquad \hat{x} = \hat{x}^- + K\nu, \qquad
        P = (I - KH) P^- (I - KH)^T + K R K^T

    Parameters
    ----------
    state : KalmanState
        The *predicted* estimate, from :func:`predict`.
    measurement : array_like
        Shape ``(measurement_dim,)``. The measurement :math:`z`.
    model : TrackModel
        Supplies :math:`h`, its Jacobian and :math:`R`. Pass a
        :meth:`TrackModel.restricted` model to update on a subset of the
        measurement components.

    Returns
    -------
    UpdateResult
        The corrected estimate and the innovation statistics that produced it.

    Raises
    ------
    ValueError
        If ``measurement`` has the wrong shape, or :math:`S` is singular.

    Notes
    -----
    The Joseph form is used rather than the shorter :math:`P = (I - KH)P^-`
    because it stays symmetric and positive semi-definite under finite
    precision even when the two covariance terms differ by the five orders of
    magnitude this scenario's :math:`R` does. The short form is cheaper and, at
    that conditioning, is the kind of thing that produces a negative variance
    after a few hundred frames.
    """
    innovation, innovation_covariance, jacobian = innovation_of(state, measurement, model)
    nis = normalised_innovation_squared(innovation, innovation_covariance)

    gain = np.linalg.solve(innovation_covariance.T, (state.covariance @ jacobian.T).T).T
    mean: NDArray[np.float64] = state.state + gain @ innovation

    spread = np.eye(state.n_state, dtype=np.float64) - gain @ jacobian
    covariance: NDArray[np.float64] = (
        spread @ state.covariance @ spread.T + gain @ model.measurement_noise @ gain.T
    )
    # Symmetrise: the Joseph form is symmetric in exact arithmetic, and forcing
    # it here stops a slow drift from accumulating over a long track.
    covariance = 0.5 * (covariance + covariance.T)

    return UpdateResult(
        posterior=KalmanState(state=mean, covariance=covariance),
        innovation=innovation,
        innovation_covariance=innovation_covariance,
        nis=nis,
    )


def associate_gnn(
    squared_distances: ArrayLike,
    gate: float,
) -> list[tuple[int, int]]:
    r"""Assign measurements to tracks by global nearest neighbour.

    Solves the rectangular assignment problem over the matrix of normalised
    innovations [1]_ ch. 3, forbidding any pair that falls outside the gate, and
    returns the minimum-total-:math:`d^2` set of one-to-one pairings. The solver
    is ``scipy.optimize.linear_sum_assignment``, which implements the algorithm
    of [4]_; ``scipy`` is already a core dependency, so this adds none.

    Global nearest neighbour rather than JPDA is a deliberate limit. Per
    ``spec/structure.md`` D2, a second association strategy is exactly the
    trigger that promotes this module to a subpackage, and whoever adds one does
    the promotion in the same change.

    Parameters
    ----------
    squared_distances : array_like
        Shape ``(n_tracks, n_measurements)``. Entry ``(i, j)`` is the
        :math:`d^2` of measurement ``j`` against track ``i``. Use ``inf`` or any
        value above ``gate`` for a pair that was never evaluated.
    gate : float
        The validation threshold on :math:`d^2`, from :func:`gate_threshold`. A
        pair scoring above it is never assigned.

    Returns
    -------
    list of (int, int)
        ``(track_index, measurement_index)`` pairs, sorted by track index. Every
        returned pair is inside the gate; tracks and measurements not named in
        the list went unassociated.

    Raises
    ------
    ValueError
        If ``squared_distances`` is not two-dimensional, or ``gate`` is not
        positive.

    Notes
    -----
    With one target and a handful of false alarms per frame, the assignment is
    almost always trivial and a greedy nearest neighbour would find the same
    answer. It is solved properly anyway because the cases where greedy differs
    -- two tracks contending for one measurement -- are exactly the cases worth
    getting right, and the solver costs nothing at this size.

    Examples
    --------
    >>> associate_gnn([[1.0, 50.0], [50.0, 2.0]], 9.21)
    [(0, 0), (1, 1)]
    """
    cost = np.atleast_2d(np.asarray(squared_distances, dtype=np.float64))
    if cost.ndim != 2:
        msg = (
            "squared_distances must be two-dimensional, (n_tracks, n_measurements); "
            f"got {cost.shape}."
        )
        raise ValueError(msg)
    if gate <= 0.0:
        msg = f"gate must be a positive threshold on d^2; got {gate}."
        raise ValueError(msg)
    if cost.size == 0:
        return []

    gated = np.where(np.isfinite(cost) & (cost <= gate), cost, _FORBIDDEN_COST)
    track_indices, measurement_indices = linear_sum_assignment(gated)

    return [
        (int(track_index), int(measurement_index))
        for track_index, measurement_index in zip(track_indices, measurement_indices, strict=True)
        if gated[track_index, measurement_index] < _FORBIDDEN_COST
    ]


@dataclass
class Track:
    """One target hypothesis, and the bookkeeping that decides whether it lives.

    Mutable, unlike the estimates it carries: a track *is* the thing that
    changes from frame to frame, and copying it each step would make the
    manager's loop harder to read for no benefit.

    Attributes
    ----------
    track_id : int
        Identity, unique within a :class:`TrackManager` and stable for the life
        of the track. A track whose id changes mid-run is a track that was lost
        and re-initiated, which the acceptance criteria forbid.
    estimate : KalmanState
        The current state estimate, posterior if this frame associated and
        prior if it coasted.
    status : str
        One of :data:`TRACK_STATUSES`.
    n_hits : int
        Frames in which this track associated a measurement.
    n_frames : int
        Frames this track has existed for, counting the one that created it.
    n_misses_in_a_row : int
        Consecutive frames without an association. Reset by every hit.
    measurement_dim : int
        Dimension of the measurement this track last used. Records the §5.3
        bootstrap: the frame at which it steps from 1 to 2 is the frame the
        track became able to unfold its Doppler, and is the single most
        informative diagnostic this scenario produces.
    last_nis : float or None
        The normalised innovation squared of the last association, or ``None``
        if the track has never associated or coasted this frame.
    """

    track_id: int
    estimate: KalmanState
    status: TrackStatus = "tentative"
    n_hits: int = 1
    n_frames: int = 1
    n_misses_in_a_row: int = 0
    measurement_dim: int = 1
    last_nis: float | None = None

    @property
    def is_alive(self) -> bool:
        """Whether this track still takes part in association."""
        return self.status != "deleted"

    @property
    def is_confirmed(self) -> bool:
        """Whether this track has passed the M-of-N initiation test."""
        return self.status in ("confirmed", "coasting")


@dataclass(frozen=True)
class FrameResult:
    """What one call to :meth:`TrackManager.step` did.

    Attributes
    ----------
    associations : dict
        Maps ``track_id`` to the index of the measurement it took this frame.
    unassociated : tuple of int
        Indices of measurements that no track claimed. Each of these seeded a
        new tentative track.
    tracks : tuple of Track
        Every track alive at the end of the frame, deleted ones excluded.
    """

    associations: dict[int, int]
    unassociated: tuple[int, ...]
    tracks: tuple[Track, ...]


@dataclass
class TrackManager:
    """Gate, assign, update, initiate and delete, one frame at a time.

    The loop is the shape motpy and Stone Soup both use, and it is small enough
    to read in one sitting:

    1. **Predict** every live track to the current frame.
    2. **Gate** every (track, measurement) pair, at the dimension *that track*
       is currently measuring -- which differs between tracks during the §5.3
       bootstrap, and is why the gate dimension is read per track rather than
       fixed for the frame.
    3. **Assign** by global nearest neighbour, :func:`associate_gnn`.
    4. **Update** the assigned tracks; **coast** the rest on their prediction.
    5. **Initiate** a tentative track from every unassociated measurement.
    6. **Promote** a tentative track that has ``n_confirm_hits`` hits in its
       first ``n_confirm_frames`` frames, and **delete** one that can no longer
       reach that count, or any track that has missed ``n_delete_misses`` frames
       in a row.

    Parameters
    ----------
    model : TrackModel
        The state model every track uses, from :func:`state_model_matrices`.
    initial_covariance : ndarray
        Shape ``(n_state, n_state)``. The covariance a new track starts with.
        Its measured components come from :math:`R`; its unmeasured rate
        components should carry the square of the largest plausible speed, since
        a new track has no velocity estimate at all.
    gate_probability : float, optional
        Probability mass the validation gate admits, default 0.99.
    n_confirm_hits, n_confirm_frames : int, optional
        The M and N of M-of-N initiation, default 4 and 5.
    n_delete_misses : int, optional
        Consecutive misses after which a confirmed track is deleted, default 3.
    n_initiation_rows : int, optional
        How many leading measurement components seed a new track's state,
        default 1. The rest start at zero carrying ``initial_covariance``'s
        variance. The default is what scenario 003 §7 requires: a brand-new
        track is still in its range-only bootstrap, so the velocity component
        of the measurement it was born from is a *folded* value and seeding the
        state with it is worse than seeding nothing. Raise it only for a
        measurement whose every component is unambiguous at birth.
    n_reacquire_frames : int, optional
        Extra frames a confirmed track may coast, with its rate forgotten and
        its gate widening, before it is finally deleted. Default 5; set to 0 to
        delete at ``n_delete_misses`` with no re-acquisition. See
        :meth:`_forget_rate`.

    Notes
    -----
    The M-of-N parameters are sized against the false-alarm rate in the
    specification's §3, not chosen for taste: at ``pfa = 1e-5`` they make a
    spurious confirmed track roughly a once-in-four-hundred-runs event. The
    deletion count is sized against the coast time a reader can see on the plot,
    and is not derived.

    References
    ----------
    .. [3] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking
           Systems*, Artech House, 1999, ch. 6.
    """

    model: TrackModel
    initial_covariance: NDArray[np.float64]
    gate_probability: float = 0.99
    n_confirm_hits: int = 4
    n_confirm_frames: int = 5
    n_delete_misses: int = 3
    n_initiation_rows: int = 1
    n_reacquire_frames: int = 5
    tracks: list[Track] = field(default_factory=list)
    _next_track_id: int = 0

    def __post_init__(self) -> None:
        """Validate the management parameters against each other."""
        if not 0.0 < self.gate_probability < 1.0:
            msg = (
                f"gate_probability must lie strictly between 0 and 1; got {self.gate_probability}."
            )
            raise ValueError(msg)
        if self.n_confirm_hits > self.n_confirm_frames:
            msg = (
                f"n_confirm_hits ({self.n_confirm_hits}) cannot exceed n_confirm_frames "
                f"({self.n_confirm_frames}); no track could ever be confirmed."
            )
            raise ValueError(msg)
        if self.n_delete_misses < 1:
            msg = f"n_delete_misses must be at least 1; got {self.n_delete_misses}."
            raise ValueError(msg)

    @property
    def lost_track_ids(self) -> tuple[int, ...]:
        """Ids of confirmed tracks currently coasting with a forgotten rate."""
        return tuple(
            track.track_id
            for track in self.tracks
            if track.is_confirmed and track.n_misses_in_a_row >= self.n_delete_misses
        )

    @property
    def confirmed_tracks(self) -> list[Track]:
        """The live tracks that have passed the M-of-N test."""
        return [track for track in self.tracks if track.is_confirmed]

    def step(
        self,
        measurements: Sequence[ArrayLike],
        *,
        measurement_dims: Sequence[int] | None = None,
    ) -> FrameResult:
        """Advance every track one frame against this frame's measurements.

        Parameters
        ----------
        measurements : sequence of array_like
            This frame's measurements. Every entry must have the full
            ``model.measurement_dim`` length; a track measuring fewer components
            takes the leading ones, per :meth:`TrackModel.restricted`.
        measurement_dims : sequence of int, optional
            Per-track measurement dimension for this frame, in the order of
            :attr:`tracks`. This is how the caller expresses the §5.3 bootstrap:
            a track that cannot yet unfold its Doppler is given 1, one that can
            is given the full dimension. Defaults to the full dimension for
            every track.

        Returns
        -------
        FrameResult
            What associated with what, and the surviving tracks.

        Raises
        ------
        ValueError
            If ``measurement_dims`` is given and does not have one entry per
            live track.
        """
        live = [track for track in self.tracks if track.is_alive]
        if measurement_dims is None:
            dims = [self.model.measurement_dim] * len(live)
        else:
            dims = list(measurement_dims)
            if len(dims) != len(live):
                msg = (
                    f"measurement_dims must have one entry per live track "
                    f"({len(live)}); got {len(dims)}."
                )
                raise ValueError(msg)

        stacked = [np.asarray(z, dtype=np.float64) for z in measurements]

        # 1. Predict, and remember each track's model for this frame.
        predictions = [predict(track.estimate, self.model) for track in live]
        models = [self.model.restricted(dim) for dim in dims]

        # 2. Gate. A pair outside its gate is forbidden rather than merely
        #    expensive, so the gate is applied here and not left to the solver.
        cost = np.full((len(live), len(stacked)), np.inf, dtype=np.float64)
        # Which measurement dimension each pair would be updated at. A pair that
        # fails the full gate may still pass a range-only one; see below.
        pair_dims = np.ones((len(live), len(stacked)), dtype=np.int_)
        for track_index, (prediction, model) in enumerate(zip(predictions, models, strict=True)):
            candidates = [model]
            if model.measurement_dim > 1:
                # The velocity-gate fallback. On this scenario a confirmed
                # track's gate rejection is almost never a missed detection --
                # at the link budget of §4 the target is found in every frame.
                # It is a measurement unfolded to the wrong Doppler fold, which
                # misses by a whole fold span in velocity while its *range* is
                # as good as any other frame's.
                #
                # Coasting on that, as §5.3 proposes, throws away a perfectly
                # good range measurement, lets the track drift, and leaves the
                # detection free to seed a parallel track that confirms and
                # competes -- which is how a single target ends up with several
                # track ids. Falling back to a range-only update instead keeps
                # the range lock, lets the rate be re-estimated from range
                # alone, and denies the competitor its measurement.
                candidates.append(model.restricted(1))

            for measurement_index, z in enumerate(stacked):
                for candidate in candidates:
                    threshold = gate_threshold(self.gate_probability, candidate.measurement_dim)
                    innovation, innovation_covariance, _ = innovation_of(
                        prediction, z[: candidate.measurement_dim], candidate
                    )
                    distance = normalised_innovation_squared(innovation, innovation_covariance)
                    if distance <= threshold:
                        cost[track_index, measurement_index] = distance
                        pair_dims[track_index, measurement_index] = candidate.measurement_dim
                        break

        # 3. Assign. The gate is per track, so the matrix already encodes it and
        #    the solver is handed the loosest threshold in play.
        loosest = max(
            (gate_threshold(self.gate_probability, model.measurement_dim) for model in models),
            default=1.0,
        )
        # One global solve over every live track. Assigning confirmed tracks
        # ahead of tentative ones was tried, on the theory that confirmation
        # should outrank a smaller residual when an established track has just
        # widened its gate to re-acquire its target. Measured over the default
        # window it was worse on every count -- range RMSE 25.3 m against
        # 17.5 m, fold selection 82% against 86% -- because a widened track
        # outbids tentative ones for false alarms too. Left as one solve.
        pairs = associate_gnn(cost, loosest) if live and stacked else []
        assigned = dict(pairs)

        # 4. Update the assigned; coast the rest.
        associations: dict[int, int] = {}
        for track_index, track in enumerate(live):
            track.n_frames += 1
            prediction = predictions[track_index]
            model = models[track_index]
            track.measurement_dim = model.measurement_dim

            matched_index = assigned.get(track_index)
            if matched_index is None:
                track.estimate = prediction
                track.n_misses_in_a_row += 1
                track.last_nis = None
                if track.status == "confirmed":
                    track.status = "coasting"
                continue

            # The pair may have gated at a lower dimension than the track's own.
            model = model.restricted(int(pair_dims[track_index, matched_index]))
            track.measurement_dim = model.measurement_dim
            result = update(prediction, stacked[matched_index][: model.measurement_dim], model)
            track.estimate = result.posterior
            track.n_hits += 1
            track.n_misses_in_a_row = 0
            track.last_nis = result.nis
            if track.status == "coasting":
                track.status = "confirmed"
            associations[track.track_id] = matched_index

        # 5. Initiate from what nothing claimed.
        claimed = set(assigned.values())
        unassociated = tuple(index for index in range(len(stacked)) if index not in claimed)
        for index in unassociated:
            self.tracks.append(self._initiate(stacked[index]))

        # 6. Promote and delete.
        self._manage(live)

        self.tracks = [track for track in self.tracks if track.is_alive]
        return FrameResult(
            associations=associations,
            unassociated=unassociated,
            tracks=tuple(self.tracks),
        )

    def _forget_rate(self, track: Track) -> Track:
        """Reset a coasting track's rate variance to its initiation value.

        A confirmed track stops associating when its measurements stop passing
        the gate, and on this scenario that means its velocity is wrong -- not
        that the target vanished. At the link budget of §4 the target is
        detected in every frame; what fails is the Doppler fold.

        Deleting the track then is the wrong move twice over. The target is
        still there, and the detection the track could not use goes on to seed a
        *parallel* tentative track that confirms a few frames later and competes
        with it, which is how one target acquires several track ids.

        So on the ``n_delete_misses``-th miss the rate is explicitly forgotten
        instead: its variance goes back to ``initial_covariance``'s and its
        correlation with position is dropped. The range gate then widens by
        ``T^2 v_max^2`` every frame, so within a frame or two it is hundreds of
        metres across and the redetection falls inside it -- and because the
        track is still live, ordinary assignment hands it the measurement ahead
        of any tentative rival. The track is deleted only if that fails for
        ``n_reacquire_frames`` more frames.
        """
        n_state = track.estimate.n_state
        n_axes = n_state // 2
        covariance = track.estimate.covariance.copy()
        covariance[n_axes:, n_axes:] = self.initial_covariance[n_axes:, n_axes:]
        covariance[:n_axes, n_axes:] = 0.0
        covariance[n_axes:, :n_axes] = 0.0
        track.estimate = KalmanState(state=track.estimate.state, covariance=covariance)
        return track

    def _initiate(self, measurement: NDArray[np.float64]) -> Track:
        """Seed a tentative track from an unassociated measurement.

        Single-point initiation: the first ``n_initiation_rows`` measurement
        components seed the state and everything else starts at zero, carrying
        the large variance of ``initial_covariance``. A new track is by §5.3
        still in its range-only bootstrap, so by default only range seeds it --
        the velocity component of the measurement is a folded value, and a
        track seeded with it starts out confidently wrong about which way the
        target is going.
        """
        state = np.zeros(self.initial_covariance.shape[0], dtype=np.float64)
        n_known = min(self.n_initiation_rows, measurement.size, state.size)
        state[:n_known] = measurement[:n_known]

        self._next_track_id += 1
        return Track(
            track_id=self._next_track_id,
            estimate=KalmanState(state=state, covariance=self.initial_covariance.copy()),
            status="tentative",
            measurement_dim=1,
        )

    def _manage(self, live: Sequence[Track]) -> None:
        """Apply M-of-N confirmation and the deletion rules."""
        for track in live:
            if track.n_misses_in_a_row >= self.n_delete_misses:
                if not track.is_confirmed:
                    track.status = "deleted"
                    continue
                # A confirmed track gets a grace period before deletion, during
                # which its rate is forgotten and its gate widens every frame.
                if track.n_misses_in_a_row == self.n_delete_misses:
                    self._forget_rate(track)
                if track.n_misses_in_a_row >= self.n_delete_misses + self.n_reacquire_frames:
                    track.status = "deleted"
                continue

            if track.status != "tentative":
                continue

            if track.n_hits >= self.n_confirm_hits:
                track.status = "confirmed"
            elif track.n_frames >= self.n_confirm_frames:
                # Out of frames to reach M hits: this was clutter.
                track.status = "deleted"
            else:
                # Delete early when the remaining frames cannot reach M hits,
                # rather than carrying a hypothesis that is already arithmetically
                # dead. Without this a false alarm survives the full N frames and
                # keeps competing for measurements it can never be confirmed on.
                remaining = self.n_confirm_frames - track.n_frames
                if track.n_hits + remaining < self.n_confirm_hits:
                    track.status = "deleted"
