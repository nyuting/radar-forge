r"""Motion models: how a target's state moves forward in time, and how unsure that makes us.

A motion model gives the filter two things for a time step of :math:`T` seconds:

- the transition, which moves a state forward, :math:`x_{k+1} = F x_k`;
- the process noise covariance Q, the uncertainty the step adds because real targets do not
  move exactly as the model says.

:class:`CartesianMotion` tracks east, north and up positions with a constant-velocity (CV) or
constant-acceleration (CA) model on each axis. :class:`RadialMotion` tracks only range and range
rate. Both are linear. Any other model that follows :class:`MotionModel` works with the filter
unchanged.

Cartesian axes are named x, y and z: x is east, y is north and z is up, in the local
east-north-up (ENU) frame about ``origin_lla_deg_m`` (see :mod:`radar_forge.core.geodesy`).

Process noise
-------------
A CV model says the velocity never changes. Real targets speed up, slow down and turn, so the
filter must allow for motion its model leaves out. It does this by treating the target's
acceleration as random. Two physical numbers describe that acceleration:

- ``sigma_acceleration_mps2``, :math:`\sigma_a`: its standard deviation, in m/s². If you know
  the largest acceleration :math:`a_{max}` and read it as a 3-sigma bound,
  :math:`\sigma_a = a_{max}/3`.
- ``acceleration_correlation_time_s``, :math:`\tau`: roughly how long one acceleration lasts
  before it changes, in seconds.

Singer [2]_ models such an acceleration as a random process with variance :math:`\sigma_a^2`
whose correlation dies away as :math:`e^{-|t|/\tau}`. Over times longer than :math:`\tau` it
acts like *white noise*: random, with no memory. The strength of white noise is its *noise
density* q, also called the power spectral density, and for this process it is

.. math::

    q = 2 \sigma_a^2 \tau \quad \text{(m²/s³)}.

Each model computes q once from :math:`\sigma_a` and :math:`\tau`, and then builds Q from the
actual time step T at every prediction, with the continuous white-noise acceleration model of
Bar-Shalom et al. [1]_ (the Notes of :class:`CartesianMotion` give the matrices). Two
consequences follow:

- **Q does not depend on how the time is cut up.** Two steps of T/2 add the same uncertainty
  as one step of T. That matters when two sensors report at different rates.
- **For steps shorter than** :math:`\tau` **it errs on the safe side.** A step then adds a
  velocity variance of :math:`2 \sigma_a^2 \tau T`, more than the :math:`\sigma_a^2 T^2` of an
  acceleration that really holds through the step. Too much noise widens the gate; too little
  loses the target. Singer's full model, which carries the acceleration in the state, is exact
  at every T. It is left for the first scenario that needs it.

**Why not the usual rule of thumb.** Textbooks often choose q so that :math:`\sqrt{q T}`, the
typical change of velocity in one step, is about :math:`a_{max} T` [1]_, which gives
:math:`q \approx a_{max}^2 T`. The sampling interval T is standing in for :math:`\tau` there,
so that q only means what was intended at one update rate. With two sensors there is no one T.

**The defaults**, :math:`\sigma_a = 4` m/s² and :math:`\tau = 1` s (so q = 32 m²/s³), were
measured from a scenario's truth. ``docs/tracking/README.md`` records the measurement, and how
well a 99.7 % gate held with them.

A CA model instead treats the *jerk*, the rate of change of acceleration, as random. It takes
``sigma_jerk_mps3`` in m/s³ and the same :math:`\tau`, and its density
:math:`q = 2 \sigma_j^2 \tau` is in m²/s⁵. No scenario uses CA yet, so its default,
1 m/s³, is not derived from data.

The older tracker's :func:`radar_forge.core.tracking.kalman.process_noise_dwna` also takes
:math:`\sigma_a`, but holds one random acceleration constant through each step. That makes its
Q depend on the step length: two steps of T/2 add less than one of T. Both models add the same
velocity variance per step only when :math:`\tau = T/2`.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001, §6.2 (the continuous white-noise acceleration and
       Wiener-process acceleration models).
.. [2] R. A. Singer, "Estimating optimal tracking filter performance for manned maneuvering
       targets," *IEEE Trans. Aerospace and Electronic Systems*, vol. AES-6, no. 4,
       pp. 473-483, 1970 (the exponentially correlated acceleration model).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial

from radar_forge.core.tracking.coordinates import Coordinate, StateLayout

__all__ = [
    "CartesianMotion",
    "MotionKind",
    "MotionModel",
    "RadialMotion",
]

# The motion model of one Cartesian axis: constant velocity (CV) or constant acceleration (CA).
MotionKind = Literal["CV", "CA"]


class MotionModel(Protocol):
    """The interface a motion model follows so that the filter can use it.

    Arrays follow the order of ``state_layout``, and each value is in its coordinate's unit.

    Attributes
    ----------
    state_layout : StateLayout
        The names, units and order of the state's elements.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §6.2.
    """

    state_layout: StateLayout

    def transition(self, state: NDArray[np.float64], dt_s: float) -> NDArray[np.float64]:
        """Move one state, or a batch of states, forward by ``dt_s`` seconds.

        Parameters
        ----------
        state : numpy.ndarray
            One state, shape ``(n_state,)``, or a batch, shape ``(n_points, n_state)`` with one
            state per row. The UKF passes all its sigma points as one batch.
        dt_s : float
            The time step, in seconds. Must be finite and not negative.

        Returns
        -------
        numpy.ndarray
            The moved state or states, the same shape as ``state``. The input is not changed.
        """
        ...

    def process_noise(self, state: NDArray[np.float64], dt_s: float) -> NDArray[np.float64]:
        """Return Q, the covariance a step of ``dt_s`` seconds adds.

        Parameters
        ----------
        state : numpy.ndarray
            The state at the start of the step, shape ``(n_state,)``. A model whose noise does
            not depend on the state ignores it.
        dt_s : float
            The time step, in seconds. Must be finite and not negative.

        Returns
        -------
        numpy.ndarray
            Q, shape ``(n_state, n_state)``, symmetric positive semidefinite. Each element has
            the product of the units of its row and its column.
        """
        ...


def _check_dt(dt_s: float) -> None:
    """Raise unless ``dt_s`` is a finite time step that does not go backwards."""
    if not np.isfinite(dt_s) or dt_s < 0:
        msg = f"dt_s must be finite and nonnegative, in seconds; got {dt_s}."
        raise ValueError(msg)


def _check_state_shape(state: NDArray[np.float64], n_state: int) -> None:
    """Raise unless ``state`` has shape ``(n_state,)`` or ``(n_points, n_state)``."""
    if state.ndim not in (1, 2) or state.shape[-1] != n_state:
        msg = f"state must have shape ({n_state},) or (n_points, {n_state}); got {state.shape}."
        raise ValueError(msg)


def _white_noise_density(sigma: float, correlation_time_s: float) -> float:
    """Return q = 2 sigma² tau, the noise density of a correlated random process.

    The module docstring's "Process noise" section derives it.
    """
    return 2.0 * sigma**2 * correlation_time_s


def _check_correlation_time(correlation_time_s: float) -> None:
    """Raise unless the correlation time is finite and positive."""
    if not np.isfinite(correlation_time_s) or correlation_time_s <= 0:
        msg = (
            "acceleration_correlation_time_s must be finite and positive, in seconds; "
            f"got {correlation_time_s}."
        )
        raise ValueError(msg)


def _check_sigma_mappings(
    axes: Mapping[str, MotionKind], given: Mapping[MotionKind, float | Mapping[str, float]]
) -> None:
    """Raise unless each standard deviation given as a mapping names exactly its model's axes."""
    if any(
        isinstance(sigma, Mapping)
        and set(sigma) != {axis for axis, axis_kind in axes.items() if axis_kind == kind}
        for kind, sigma in given.items()
    ):
        msg = (
            "a sigma_acceleration_mps2 or sigma_jerk_mps3 mapping must name exactly the axes "
            f"of its model (CV or CA); got axes {dict(axes)} and values {dict(given)}."
        )
        raise ValueError(msg)


def _sigma_of(axis: str, sigma: float | Mapping[str, float]) -> float:
    """Return one axis's standard deviation from a single value or a per-axis mapping."""
    return float(sigma[axis] if isinstance(sigma, Mapping) else sigma)


def _axis_names(axis: str, kind: MotionKind) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the coordinate names and units of one axis, position first."""
    names: tuple[str, ...] = (f"{axis}_m", f"{axis}dot_mps")
    units: tuple[str, ...] = ("m", "m/s")
    if kind == "CA":
        names += (f"{axis}ddot_mps2",)
        units += ("m/s^2",)
    return names, units


class CartesianMotion:
    r"""Constant-velocity or constant-acceleration motion on each east, north or up axis.

    Each axis moves on its own, with its own noise. An axis with the CV model holds position
    and velocity; one with the CA model also holds acceleration.

    Parameters
    ----------
    axes : mapping of str to {"CV", "CA"}
        The axes to track, each mapped to its model, ``"CV"`` or ``"CA"`` (see
        :data:`MotionKind`). The keys are a nonempty subset of x (east), y (north) and z (up):
        ``{"x": "CV", "y": "CA"}``.
    origin_lla_deg_m : tuple of float
        The ENU origin: latitude in degrees, longitude in degrees, height in metres.
    sigma_acceleration_mps2 : float or mapping of str to float, default 4.0
        :math:`\sigma_a`, the standard deviation of the target's acceleration on each CV axis,
        in m/s² (:math:`a_{max}/3` for a 3-sigma bound :math:`a_{max}`). One value for every
        CV axis, or a mapping that names exactly the CV axes. A mapping is for motion that is
        less predictable in one direction than another. For example, an aircraft usually turns
        more freely than it climbs, so z can take a smaller value than x and y. The module
        docstring's "Process noise" section derives the default. Must be finite and not
        negative.
    sigma_jerk_mps3 : float or mapping of str to float, default 1.0
        :math:`\sigma_j`, the standard deviation of the jerk on each CA axis, in m/s³. One
        value for every CA axis, or a mapping that names exactly the CA axes. Must be finite
        and not negative.
    acceleration_correlation_time_s : float, default 1.0
        :math:`\tau`, roughly how long one acceleration (or, on a CA axis, one jerk) lasts, in
        seconds. The noise density of each axis is :math:`q = 2 \sigma^2 \tau` (module
        docstring, "Process noise"). Must be finite and positive.
    order : tuple of str, optional
        The order of the state's elements, by name, for example ``("x_m", "xdot_mps")``. It
        must name every element exactly once. The default keeps each axis together, in the
        order x, y, z, and position, velocity, acceleration within it.
    frame : str, default "ENU"
        The name of the coordinate frame, which models compare to check they agree.

    Attributes
    ----------
    state_layout : StateLayout
        Names ``{a}_m`` (m), ``{a}dot_mps`` (m/s) and, on a CA axis, ``{a}ddot_mps2``
        (m/s²), for each axis ``a``.
    noise_density : dict of str to float
        q for each axis, :math:`2 \sigma^2 \tau`: in m²/s³ on a CV axis, m²/s⁵ on a CA axis.

    Raises
    ------
    ValueError
        If ``axes`` is empty, or names an axis other than x, y or z, or a model other than CV
        or CA. If a sigma mapping does not name exactly the right axes, a sigma is negative or
        not finite, or the correlation time is not finite and positive. If ``order`` is not an
        exact reordering of the element names.

    Notes
    -----
    On each axis, with :math:`T` the time step and the state ordered position first, the
    transition is the Taylor series of the motion,

    .. math::

        F_{CV} = \begin{bmatrix} 1 & T \\ 0 & 1 \end{bmatrix}, \qquad
        F_{CA} = \begin{bmatrix} 1 & T & T^2/2 \\ 0 & 1 & T \\ 0 & 0 & 1 \end{bmatrix},

    and the process noise is the white noise integrated over the step [1]_,

    .. math::

        Q_{CV} = q \begin{bmatrix} T^3/3 & T^2/2 \\ T^2/2 & T \end{bmatrix}, \qquad
        Q_{CA} = q \begin{bmatrix} T^5/20 & T^4/8 & T^3/6 \\ T^4/8 & T^3/3 & T^2/2 \\
                 T^3/6 & T^2/2 & T \end{bmatrix}.

    Both follow one formula. Number an axis's elements from the highest derivative down, so
    that the last element is position: :math:`a = 0` is the driven derivative (velocity for
    CV, acceleration for CA). Then :math:`F_{ij} = T^{j-i}/(j-i)!` for :math:`j \ge i` in the
    usual order, and :math:`Q_{ab} = q\,T^{a+b+1} / ((a+b+1)\,a!\,b!)`.

    The axes have independent noise, so Q has no terms between axes. A cross-axis covariance
    already in the estimate is kept, because the filter computes :math:`F P F^T`.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §6.2.

    Examples
    --------
    >>> motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=(36.0, -78.9, 60.0))
    >>> motion.transition(np.array([100.0, 10.0]), 2.0)
    array([120.,  10.])
    """

    def __init__(
        self,
        axes: Mapping[str, MotionKind],
        *,
        origin_lla_deg_m: tuple[float, float, float],
        sigma_acceleration_mps2: float | Mapping[str, float] = 4.0,
        sigma_jerk_mps3: float | Mapping[str, float] = 1.0,
        acceleration_correlation_time_s: float = 1.0,
        order: tuple[str, ...] | None = None,
        frame: str = "ENU",
    ) -> None:
        if not axes or set(axes) - set("xyz") or any(v not in ("CV", "CA") for v in axes.values()):
            msg = f"axes must map a nonempty subset of x/y/z to CV or CA; got {dict(axes)}."
            raise ValueError(msg)
        # The CV and CA standard deviations have different units (m/s² and m/s³), so each kind
        # of axis takes its own argument.
        given: dict[MotionKind, float | Mapping[str, float]] = {
            "CV": sigma_acceleration_mps2,
            "CA": sigma_jerk_mps3,
        }
        _check_sigma_mappings(axes, given)
        sigma = {axis: _sigma_of(axis, given[kind]) for axis, kind in sorted(axes.items())}
        if not all(np.isfinite(value) and value >= 0 for value in sigma.values()):
            msg = (
                "sigma_acceleration_mps2 and sigma_jerk_mps3 must be finite and nonnegative; "
                f"got {sigma}."
            )
            raise ValueError(msg)
        _check_correlation_time(acceleration_correlation_time_s)
        self.noise_density = {
            axis: _white_noise_density(value, acceleration_correlation_time_s)
            for axis, value in sigma.items()
        }
        names = {axis: _axis_names(axis, axes[axis]) for axis in sorted(axes)}
        default_order = [name for axis in sorted(axes) for name in names[axis][0]]
        if order is not None and sorted(order) != sorted(default_order):
            msg = f"order must name each of {default_order} exactly once; got {order}."
            raise ValueError(msg)

        unit = {
            name: unit_of_name
            for axis_names, axis_units in names.values()
            for name, unit_of_name in zip(axis_names, axis_units, strict=True)
        }
        self.state_layout = StateLayout(
            tuple(Coordinate(name, unit[name]) for name in (order or default_order)),
            frame,
            origin_lla_deg_m,
        )

        # F and Q are polynomials in T with fixed exponents and coefficients. Working those out
        # once here lets each be built with one array expression per call.
        n_state = self.state_layout.dimension
        self._f_exponent = np.zeros((n_state, n_state), dtype=np.int64)
        self._f_coefficient = np.eye(n_state, dtype=np.float64)
        self._q_exponent = np.zeros((n_state, n_state), dtype=np.int64)
        self._q_coefficient = np.zeros((n_state, n_state), dtype=np.float64)
        # A loop over at most three axes, run once per model: each axis's block lands at its own
        # scattered indices, which depend on ``order``.
        for axis, (axis_names, _) in names.items():
            index = np.array(self.state_layout.indices(axis_names))
            rows, cols = np.meshgrid(index, index, indexing="ij")
            i, j = np.meshgrid(np.arange(len(index)), np.arange(len(index)), indexing="ij")
            # a and b count them down from the highest one instead, as in the Notes formula.
            a, b = len(index) - 1 - i, len(index) - 1 - j
            power = np.maximum(j - i, 0)
            self._f_exponent[rows, cols] = power
            self._f_coefficient[rows, cols] = np.where(j >= i, 1.0 / factorial(power), 0.0)
            self._q_exponent[rows, cols] = a + b + 1
            self._q_coefficient[rows, cols] = self.noise_density[axis] / (
                (a + b + 1) * factorial(a) * factorial(b)
            )

    def matrices(self, dt_s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the transition matrix F and the process noise Q for a step of ``dt_s``.

        Parameters
        ----------
        dt_s : float
            The time step, in seconds. Must be finite and not negative.

        Returns
        -------
        transition : numpy.ndarray
            F, shape ``(n_state, n_state)``, in the order of ``state_layout``.
        noise : numpy.ndarray
            Q, shape ``(n_state, n_state)``. Each element has the product of the units of its
            row and its column.

        Raises
        ------
        ValueError
            If ``dt_s`` is negative or not finite.

        Examples
        --------
        >>> motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=(36.0, -78.9, 60.0))
        >>> f, q = motion.matrices(1.0)
        >>> f
        array([[1., 1.],
               [0., 1.]])
        """
        return self._transition_matrix(dt_s), self._noise_matrix(dt_s)

    def _transition_matrix(self, dt_s: float) -> NDArray[np.float64]:
        """Return F for a step of ``dt_s`` seconds, after checking the step."""
        _check_dt(dt_s)
        return self._f_coefficient * float(dt_s) ** self._f_exponent

    def _noise_matrix(self, dt_s: float) -> NDArray[np.float64]:
        """Return Q for a step of ``dt_s`` seconds, after checking the step."""
        _check_dt(dt_s)
        return self._q_coefficient * float(dt_s) ** self._q_exponent

    def transition(self, state: NDArray[np.float64], dt_s: float) -> NDArray[np.float64]:
        """Move one state, or a batch of states, forward by ``dt_s`` seconds.

        Parameters
        ----------
        state : numpy.ndarray
            One state, shape ``(n_state,)``, or one per row, shape ``(n_points, n_state)``.
        dt_s : float
            The time step, in seconds. Must be finite and not negative.

        Returns
        -------
        numpy.ndarray
            :math:`F x` for each state, the same shape as ``state``.

        Raises
        ------
        ValueError
            If ``dt_s`` is negative or not finite, or ``state`` has the wrong shape.
        """
        state = np.asarray(state, dtype=np.float64)
        _check_state_shape(state, self.state_layout.dimension)
        # With one state per row, x Fᵀ is F x for every row at once.
        return state @ self._transition_matrix(dt_s).T

    def process_noise(self, state: NDArray[np.float64], dt_s: float) -> NDArray[np.float64]:
        """Return Q, the covariance a step of ``dt_s`` seconds adds.

        Parameters
        ----------
        state : numpy.ndarray
            The state, shape ``(n_state,)``. Not used: Q here does not depend on the state.
        dt_s : float
            The time step, in seconds. Must be finite and not negative.

        Returns
        -------
        numpy.ndarray
            Q, shape ``(n_state, n_state)``.

        Raises
        ------
        ValueError
            If ``dt_s`` is negative or not finite.
        """
        return self._noise_matrix(dt_s)


class RadialMotion:
    r"""Constant range rate along the line of sight, with random radial acceleration.

    The state is range and range rate. Range rate is positive when the target is *closing*,
    so a positive range rate makes the range fall. This matches the sign of Doppler in
    :mod:`radar_forge.core.dsp`.

    Parameters
    ----------
    sigma_acceleration_mps2 : float, default 4.0
        :math:`\sigma_a`, the standard deviation of the target's radial acceleration, in m/s²
        (:math:`a_{max}/3` for a 3-sigma bound :math:`a_{max}`). The module docstring's
        "Process noise" section derives the default. Must be finite and not negative.
    acceleration_correlation_time_s : float, default 1.0
        :math:`\tau`, roughly how long one acceleration lasts, in seconds. Must be finite and
        positive.

    Attributes
    ----------
    noise_density_m2ps3 : float
        The noise density :math:`q = 2 \sigma_a^2 \tau`, in m²/s³ (module docstring,
        "Process noise").
    state_layout : StateLayout
        ``range_m`` (m) then ``range_rate_mps`` (m/s), in the frame ``"radial"``.

    Raises
    ------
    ValueError
        If ``sigma_acceleration_mps2`` is negative or not finite, or the correlation time is
        not finite and positive.

    Notes
    -----
    This is the CV model of :class:`CartesianMotion` with the sign of the rate flipped, so

    .. math::

        F = \begin{bmatrix} 1 & -T \\ 0 & 1 \end{bmatrix}, \qquad
        Q = q \begin{bmatrix} T^3/3 & -T^2/2 \\ -T^2/2 & T \end{bmatrix}.

    The cross terms of Q are negative because a closing acceleration makes the range fall.

    Range is kept as one continuous value, even when the radar measures it folded (modulo its
    unambiguous range).

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §6.2.

    Examples
    --------
    >>> RadialMotion().transition(np.array([100.0, 5.0]), 2.0)
    array([90.,  5.])
    """

    def __init__(
        self,
        sigma_acceleration_mps2: float = 4.0,
        acceleration_correlation_time_s: float = 1.0,
    ) -> None:
        if not np.isfinite(sigma_acceleration_mps2) or sigma_acceleration_mps2 < 0:
            msg = (
                "sigma_acceleration_mps2 must be finite and nonnegative; "
                f"got {sigma_acceleration_mps2}."
            )
            raise ValueError(msg)
        _check_correlation_time(acceleration_correlation_time_s)
        self.noise_density_m2ps3 = _white_noise_density(
            float(sigma_acceleration_mps2), float(acceleration_correlation_time_s)
        )
        self.state_layout = StateLayout(
            (Coordinate("range_m", "m"), Coordinate("range_rate_mps", "m/s")), "radial"
        )

    def transition(self, state: NDArray[np.float64], dt_s: float) -> NDArray[np.float64]:
        """Move one state, or a batch of states, forward by ``dt_s`` seconds.

        Parameters
        ----------
        state : numpy.ndarray
            Range in m and range rate in m/s (positive closing): shape ``(2,)``, or one state
            per row, shape ``(n_points, 2)``.
        dt_s : float
            The time step, in seconds. Must be finite and not negative.

        Returns
        -------
        numpy.ndarray
            The moved state or states, the same shape as ``state``. The range rate is
            unchanged.

        Raises
        ------
        ValueError
            If ``dt_s`` is negative or not finite, or ``state`` has the wrong shape.
        """
        _check_dt(dt_s)
        state = np.asarray(state, dtype=np.float64)
        _check_state_shape(state, 2)
        predicted = state.copy()
        predicted[..., 0] -= state[..., 1] * dt_s
        return predicted

    def process_noise(self, state: NDArray[np.float64], dt_s: float) -> NDArray[np.float64]:
        """Return Q, the covariance a step of ``dt_s`` seconds adds.

        Parameters
        ----------
        state : numpy.ndarray
            The state, shape ``(2,)``. Not used: Q here does not depend on the state.
        dt_s : float
            The time step, in seconds. Must be finite and not negative.

        Returns
        -------
        numpy.ndarray
            Q, shape ``(2, 2)``: m² for range, m²/s² for range rate, m²/s between them.

        Raises
        ------
        ValueError
            If ``dt_s`` is negative or not finite.
        """
        _check_dt(dt_s)
        return self.noise_density_m2ps3 * np.array(
            [[dt_s**3 / 3, -(dt_s**2) / 2], [-(dt_s**2) / 2, dt_s]], dtype=np.float64
        )
