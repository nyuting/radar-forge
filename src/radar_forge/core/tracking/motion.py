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
filter must allow for motion its model leaves out. It does this by pretending that an unknown
acceleration acts on the target: random, changing from instant to instant, with no memory (this
is called *white noise*). The *noise density* q, also called the power spectral density, says
how strong that random acceleration is. On a CV axis it is in m²/s³.

Over one step of :math:`T` seconds the noise adds a velocity variance of :math:`q T` (in m²/s²),
so :math:`\sqrt{q T}` is the typical change of velocity in one step. A rule of thumb [1]_ sets
that equal to the largest change of velocity in one step, :math:`a_{max} T`. Here
:math:`a_{max}` is the largest acceleration expected. That gives
:math:`q \approx a_{max}^2 T`. A target that pulls 4 m/s², tracked every second, needs
q ≈ 16 m²/s³. The default, 1 m²/s³, suits about 1 m/s² at one update a second.

A CA model instead pretends that the *jerk*, the rate of change of acceleration, is white
noise. Its density is in m²/s⁵. :math:`\sqrt{q T}` is then the typical change of acceleration
in one step. So :math:`q \approx \Delta a_{max}^2 / T`, where :math:`\Delta a_{max}` is the
largest change of acceleration expected in one step.

The older tracker's :func:`radar_forge.core.tracking.kalman.process_noise_dwna` uses a
discrete model instead: one random acceleration, held constant through each step, with
variance :math:`\sigma_a^2` in m²/s⁴. The two add the same velocity variance per step when
:math:`q = \sigma_a^2 T`. Their position terms then differ a little: :math:`\sigma_a^2 T^4/3`
here against :math:`\sigma_a^2 T^4/4` there.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001, §6.2 (the continuous white-noise acceleration and
       Wiener-process acceleration models, and how to choose their noise density).
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
        msg = f"dt_s must be finite and nonnegative, in seconds; got {dt_s}"
        raise ValueError(msg)


def _check_state_shape(state: NDArray[np.float64], n_state: int) -> None:
    """Raise unless ``state`` has shape ``(n_state,)`` or ``(n_points, n_state)``."""
    if state.ndim not in (1, 2) or state.shape[-1] != n_state:
        msg = f"state must have shape ({n_state},) or (n_points, {n_state}); got {state.shape}"
        raise ValueError(msg)


def _check_density_mappings(
    axes: Mapping[str, MotionKind], given: Mapping[MotionKind, float | Mapping[str, float]]
) -> None:
    """Raise unless each density given as a mapping names exactly the axes of its model."""
    if any(
        isinstance(density, Mapping)
        and set(density) != {axis for axis, axis_kind in axes.items() if axis_kind == kind}
        for kind, density in given.items()
    ):
        msg = (
            "a noise density mapping must name exactly the axes of its model (CV or CA); "
            f"got axes {dict(axes)} and densities {dict(given)}"
        )
        raise ValueError(msg)


def _density_of(axis: str, density: float | Mapping[str, float]) -> float:
    """Return one axis's noise density from a single value or a per-axis mapping."""
    return float(density[axis] if isinstance(density, Mapping) else density)


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
    acceleration_noise_density_m2ps3 : float or mapping of str to float, default 1.0
        The noise density q of each CV axis, in m²/s³: the strength of the random acceleration
        the model allows for. Choose :math:`q \approx a_{max}^2 T` (module Notes). One value
        for every CV axis, or a mapping that names exactly the CV axes. A mapping is for motion
        that is less predictable in one direction than another. For example, an aircraft
        usually turns more freely than it climbs, so z can take a smaller q than x and y. Must
        be finite and not negative.
    jerk_noise_density_m2ps5 : float or mapping of str to float, default 1.0
        The noise density q of each CA axis, in m²/s⁵: the strength of the random jerk the
        model allows for. Choose :math:`q \approx \Delta a_{max}^2 / T` (module Notes). One
        value for every CA axis, or a mapping that names exactly the CA axes. Must be finite
        and not negative.
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

    Raises
    ------
    ValueError
        If ``axes`` is empty, or names an axis other than x, y or z, or a model other than CV
        or CA. If a density mapping does not name exactly the right axes, or a density is
        negative or not finite. If ``order`` is not an exact reordering of the element names.

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
        acceleration_noise_density_m2ps3: float | Mapping[str, float] = 1.0,
        jerk_noise_density_m2ps5: float | Mapping[str, float] = 1.0,
        order: tuple[str, ...] | None = None,
        frame: str = "ENU",
    ) -> None:
        if not axes or set(axes) - set("xyz") or any(v not in ("CV", "CA") for v in axes.values()):
            msg = f"axes must map a nonempty subset of x/y/z to CV or CA; got {dict(axes)}"
            raise ValueError(msg)
        # The CV and CA densities have different units (m²/s³ and m²/s⁵), so each kind of axis
        # takes its own argument.
        given: dict[MotionKind, float | Mapping[str, float]] = {
            "CV": acceleration_noise_density_m2ps3,
            "CA": jerk_noise_density_m2ps5,
        }
        _check_density_mappings(axes, given)
        density = {axis: _density_of(axis, given[kind]) for axis, kind in sorted(axes.items())}
        if not all(np.isfinite(q) and q >= 0 for q in density.values()):
            msg = f"noise densities must be finite and nonnegative; got {density}"
            raise ValueError(msg)
        names = {axis: _axis_names(axis, axes[axis]) for axis in sorted(axes)}
        default_order = [name for axis in sorted(axes) for name in names[axis][0]]
        if order is not None and sorted(order) != sorted(default_order):
            msg = f"order must name each of {default_order} exactly once; got {order}"
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
        # once here lets matrices() build both with one array expression per call.
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
            # i and j count the axis's derivatives up from position (0) to the highest one.
            i, j = np.meshgrid(np.arange(len(index)), np.arange(len(index)), indexing="ij")
            # a and b count them down from the highest one instead, as in the Notes formula.
            a, b = len(index) - 1 - i, len(index) - 1 - j
            power = np.maximum(j - i, 0)
            self._f_exponent[rows, cols] = power
            self._f_coefficient[rows, cols] = np.where(j >= i, 1.0 / factorial(power), 0.0)
            self._q_exponent[rows, cols] = a + b + 1
            self._q_coefficient[rows, cols] = density[axis] / (
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
        _check_dt(dt_s)
        dt_s = float(dt_s)
        transition = self._f_coefficient * dt_s**self._f_exponent
        noise = self._q_coefficient * dt_s**self._q_exponent
        return transition, noise

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
        return state @ self.matrices(dt_s)[0].T

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
        return self.matrices(dt_s)[1]


class RadialMotion:
    r"""Constant range rate along the line of sight, with random radial acceleration.

    The state is range and range rate. Range rate is positive when the target is *closing*,
    so a positive range rate makes the range fall. This matches the sign of Doppler in
    :mod:`radar_forge.core.dsp`.

    Parameters
    ----------
    acceleration_noise_density_m2ps3 : float, default 1.0
        The noise density q, in m²/s³: the strength of the random radial acceleration the model
        allows for. Choose :math:`q \approx a_{max}^2 T` (module Notes). Must be finite and not
        negative.

    Attributes
    ----------
    density : float
        The noise density q, in m²/s³.
    state_layout : StateLayout
        ``range_m`` (m) then ``range_rate_mps`` (m/s), in the frame ``"radial"``.

    Raises
    ------
    ValueError
        If the noise density is negative or not finite.

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

    def __init__(self, acceleration_noise_density_m2ps3: float = 1.0) -> None:
        if (
            not np.isfinite(acceleration_noise_density_m2ps3)
            or acceleration_noise_density_m2ps3 < 0
        ):
            msg = (
                "acceleration_noise_density_m2ps3 must be finite and nonnegative; "
                f"got {acceleration_noise_density_m2ps3}"
            )
            raise ValueError(msg)
        self.density = float(acceleration_noise_density_m2ps3)
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
        return self.density * np.array(
            [[dt_s**3 / 3, -(dt_s**2) / 2], [-(dt_s**2) / 2, dt_s]], dtype=np.float64
        )
