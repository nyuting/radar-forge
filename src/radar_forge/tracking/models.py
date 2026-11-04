r"""State layouts, and models of how a target moves.

- :class:`Coordinate` and :class:`StateLayout` name each number of a state, give its unit and
  say which numbers repeat. The layout holds the package's one wrap of periodic values.
- A :class:`MotionModel` moves states ``(..., n_state)`` forward by ``dt_s`` and gives the
  process noise Q, batched, so a bank of tracks or all its sigma points go through one call:
  :class:`CartesianMotion` (constant velocity or acceleration per east-north-up axis) and
  :class:`RadialConstantVelocity` (bistatic range and its rate, the rate the range's
  derivative, positive receding). :class:`CoordinatedTurn` is reserved for scenario 011.

**Process noise** [1]_. ``"cwna"`` drives the highest derivative with continuous white noise
of intensity :math:`q = 2 \sigma^2 \tau`, Singer's model [2]_ seen over steps longer than its
correlation time :math:`\tau`; its Q is right for any step, so for asynchronous sensors.
``"dwna"`` holds one random acceleration of standard deviation :math:`\sigma` through each
step, so its Q depends on how time is cut up; scenario 003 specifies it. Both add the same
velocity variance per step only when :math:`\tau = T/2`.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001, §6.2 (continuous white-noise acceleration and
       Wiener-process acceleration models) and §6.3.2 (discrete white-noise acceleration).
.. [2] R. A. Singer, "Estimating optimal tracking filter performance for manned maneuvering
       targets," *IEEE Trans. Aerospace and Electronic Systems*, vol. AES-6, no. 4,
       pp. 473-483, 1970.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import block_diag
from scipy.special import factorial

__all__ = [
    "CartesianMotion",
    "Coordinate",
    "CoordinatedTurn",
    "MotionModel",
    "RadialConstantVelocity",
    "StateLayout",
]

ProcessNoise = Literal["cwna", "dwna"]

_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Coordinate:
    """One named number of a state or measurement: name, unit and period.

    Parameters
    ----------
    name : str
        Name with a unit suffix, such as ``"x_m"``.
    unit : str
        Physical unit, such as ``"m/s"``.
    period : float or None, optional
        The length of one repeat, in ``unit``, for a value that repeats: 2π for an azimuth,
        the unambiguous range for a folded range. None (the default) if it does not.

    Raises
    ------
    ValueError
        If ``name`` or ``unit`` is empty, or ``period`` is not finite and positive.

    Examples
    --------
    >>> Coordinate("azimuth_rad", "rad", period=2 * np.pi).unit
    'rad'
    """

    name: str
    unit: str
    period: float | None = None

    def __post_init__(self) -> None:
        """Check the name, unit and period."""
        if not self.name or not self.unit:
            msg = f"a coordinate needs a name and a unit; got {self.name!r}, {self.unit!r}."
            raise ValueError(msg)
        if self.period is not None and not (np.isfinite(self.period) and self.period > 0):
            msg = f"a coordinate period must be finite and positive; got {self.period!r}."
            raise ValueError(msg)


@dataclass(frozen=True)
class StateLayout:
    """The ordered, named coordinates of a vector, and the frame they live in.

    Code finds a coordinate by name, never by position. Two models work together only if
    their layouts are equal.

    Parameters
    ----------
    coordinates : tuple of Coordinate
        In vector order, with unique names.
    frame : str, optional
        Name of the frame, such as ``"ENU"``. Default ``"local"``.
    origin_lla_deg_m : tuple of float or None, optional
        The ENU origin (latitude and longitude in degrees, altitude in metres), kept for
        converting back to geodetic. None (the default) for a layout not tied to the Earth.

    Raises
    ------
    ValueError
        If ``coordinates`` is empty or repeats a name, or ``frame`` is empty.

    Examples
    --------
    >>> layout = StateLayout((Coordinate("x_m", "m"), Coordinate("xdot_mps", "m/s")))
    >>> layout.indices(("xdot_mps",))
    (1,)
    """

    coordinates: tuple[Coordinate, ...]
    frame: str = "local"
    origin_lla_deg_m: tuple[float, float, float] | None = None
    _periodic: NDArray[np.bool_] = field(init=False, repr=False, compare=False)
    _period: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the coordinates and find the periodic ones once."""
        coordinates = tuple(self.coordinates)
        names = [c.name for c in coordinates]
        if not coordinates or len(set(names)) != len(names) or not self.frame:
            msg = f"a layout needs unique names and a frame; got {names}, {self.frame!r}."
            raise ValueError(msg)
        periodic = np.array([c.period is not None for c in coordinates])
        period = np.array([c.period for c in coordinates if c.period is not None], dtype=float)
        # The dataclass is frozen, so the checked values are stored through object.
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "_periodic", periodic)
        object.__setattr__(self, "_period", period)

    @property
    def names(self) -> tuple[str, ...]:
        """Coordinate names, in vector order."""
        return tuple(c.name for c in self.coordinates)

    @property
    def dimension(self) -> int:
        """Number of coordinates."""
        return len(self.coordinates)

    def indices(self, names: Sequence[str]) -> tuple[int, ...]:
        """Return the positions of the named coordinates, in the order asked for.

        Raises
        ------
        ValueError
            If a name is not in this layout.
        """
        unknown = [name for name in names if name not in self.names]
        if unknown:
            msg = f"coordinates {unknown} are not in the layout {self.names}."
            raise ValueError(msg)
        return tuple(self.names.index(name) for name in names)

    def wrap(
        self, value: ArrayLike, *, interval: Literal["centred", "nonnegative"] = "centred"
    ) -> NDArray[np.float64]:
        """Return a copy with every periodic coordinate wrapped into one period.

        Parameters
        ----------
        value : array_like
            Vectors in this layout's order, shape ``(..., n_state)``.
        interval : {"centred", "nonnegative"}, optional
            [-period/2, period/2) (the default), for differences; or [0, period), for
            display.

        Returns
        -------
        numpy.ndarray
            Float64 copy, shape ``(..., n_state)``.

        Raises
        ------
        ValueError
            If ``interval`` is unknown.

        Examples
        --------
        >>> layout = StateLayout((Coordinate("range_m", "m", period=1000.0),))
        >>> layout.wrap([700.0]), layout.wrap([-300.0], interval="nonnegative")
        (array([-300.]), array([700.]))
        """
        if interval not in ("centred", "nonnegative"):
            msg = f"interval must be 'centred' or 'nonnegative'; got {interval!r}."
            raise ValueError(msg)
        result = np.array(value, dtype=np.float64)
        if not self._period.size:
            return result
        offset = self._period / 2 if interval == "centred" else np.zeros_like(self._period)
        folded = np.mod(result[..., self._periodic] + offset, self._period)
        # np.mod of a tiny negative number can round up to exactly one period.
        folded = np.where(folded >= self._period, folded - self._period, folded)
        result[..., self._periodic] = folded - offset
        return result

    def residual(self, a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
        """Return ``a - b``, broadcast, the short way round for periodic coordinates.

        Examples
        --------
        >>> layout = StateLayout((Coordinate("azimuth_deg", "deg", period=360.0),))
        >>> layout.residual([359.0], [1.0])
        array([-2.])
        """
        return self.wrap(np.subtract(a, b, dtype=np.float64))


class MotionModel(Protocol):
    """What a filter asks of a motion model, for states ``(..., n_state)`` at once.

    ``dt_s`` broadcasts against the states' leading axes ``(...)``.

    Attributes
    ----------
    state_layout : StateLayout
        The state's names, units and order.
    """

    state_layout: StateLayout

    def transition(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return the states moved forward by ``dt_s``, shape ``(..., n_state)``."""
        ...

    def jacobian(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return the derivative of the transition, shape ``(..., n_state, n_state)``."""
        ...

    def process_noise(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return Q for each step, shape ``(..., n_state, n_state)``."""
        ...


def _noise_intensity(sigma: float, tau_s: float, process_noise: str, order: int) -> float:
    """Return q = 2 sigma² tau for "cwna", or sigma² for "dwna", after checking the settings."""
    if process_noise not in ("cwna", "dwna") or (process_noise == "dwna" and order != 1):
        msg = f"process_noise must be 'cwna', or 'dwna' at order 1; got {process_noise!r}."
        raise ValueError(msg)
    if not (np.isfinite(sigma) and sigma >= 0 and np.isfinite(tau_s) and tau_s > 0):
        msg = f"need a finite sigma >= 0 and correlation time > 0; got {sigma}, {tau_s}."
        raise ValueError(msg)
    return 2.0 * sigma**2 * tau_s if process_noise == "cwna" else sigma**2


def _axis_tables(order: int, process_noise: str, intensity: float) -> list[NDArray[np.float64]]:
    """Return one axis's F and Q as coefficient and exponent tables in T, position first.

    With i, j counting up from position and a, b down from the driven derivative:
    F_ij = T^(j-i) / (j-i)! for j >= i; CWNA Q_ab = q T^(a+b+1) / ((a+b+1) a! b!); DWNA
    Q_ab = sigma² T^(a+b+2) / ((a+1)! (b+1)!), that is sigma² Γ Γᵀ with Γ = (T²/2, T).
    """
    i, j = np.meshgrid(np.arange(order + 1), np.arange(order + 1), indexing="ij")
    a, b = order - i, order - j
    f_exponent = np.maximum(j - i, 0)
    f_coefficient = np.where(j >= i, 1.0 / factorial(f_exponent), 0.0)
    if process_noise == "cwna":
        q_exponent = a + b + 1
        q_coefficient = intensity / (q_exponent * factorial(a) * factorial(b))
    else:
        q_exponent = a + b + 2
        q_coefficient = intensity / (factorial(a + 1) * factorial(b + 1))
    return [f_coefficient, f_exponent.astype(float), q_coefficient, q_exponent.astype(float)]


class _PolynomialMotion:
    """A linear motion model whose F and Q are fixed polynomials in the step T.

    The block-diagonal tables are built once, so each call is one array expression.
    """

    def __init__(self, layout: StateLayout, axis_tables: list[list[NDArray[np.float64]]]) -> None:
        self.state_layout = layout
        tables = [block_diag(*parts) for parts in zip(*axis_tables, strict=True)]
        self._f_coefficient, self._f_exponent, self._q_coefficient, self._q_exponent = tables

    def jacobian(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return F, which is also the Jacobian, shape ``(..., n_state, n_state)``."""
        step_s = np.asarray(dt_s, dtype=np.float64)[..., None, None]
        result: NDArray[np.float64] = self._f_coefficient * step_s**self._f_exponent
        return result

    def transition(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return F x, shape ``(..., n_state)``."""
        column = np.asarray(state, dtype=np.float64)[..., None]
        return (self.jacobian(state, dt_s) @ column)[..., 0]

    def process_noise(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return Q, shape ``(..., n_state, n_state)``; it does not depend on the state."""
        step_s = np.asarray(dt_s, dtype=np.float64)[..., None, None]
        result: NDArray[np.float64] = self._q_coefficient * step_s**self._q_exponent
        return result


class CartesianMotion(_PolynomialMotion):
    r"""Constant velocity (order 1) or constant acceleration (order 2) on each ENU axis.

    The state is interleaved per axis, ``[x_m, xdot_mps, (xddot_mps2,) y_m, ...]``. F and Q
    come from closed-form polynomial tables [1]_, not a matrix exponential:

    .. math::

        F = \begin{bmatrix} 1 & T \\ 0 & 1 \end{bmatrix}, \quad
        Q^{cwna} = q \begin{bmatrix} T^3/3 & T^2/2 \\ T^2/2 & T \end{bmatrix}, \quad
        Q^{dwna} = \sigma_a^2 \begin{bmatrix} T^4/4 & T^3/2 \\ T^3/2 & T^2 \end{bmatrix},

    and at order 2 :math:`Q = q\,[T^5/20, T^4/8, T^3/6;\ \cdot, T^3/3, T^2/2;\ \cdot, \cdot,
    T]`.

    Parameters
    ----------
    axes : str or sequence of str, default "xyz"
        The axes tracked, from x (east), y (north) and z (up).
    order : {1, 2}, default 1
        1 for constant velocity, 2 for constant acceleration.
    sigma_acceleration_mps2 : float, default 4.0
        Order 1: standard deviation of the target's acceleration.
    sigma_jerk_mps3 : float, default 1.0
        Order 2: standard deviation of its jerk.
    acceleration_correlation_time_s : float, default 1.0
        :math:`\tau` for ``"cwna"``. With the defaults, q = 32 m²/s³, measured on scenario
        001's truth (``docs/user-guide/06-tracking.md``).
    process_noise : {"cwna", "dwna"}, default "cwna"
        The noise model (module docstring); ``"dwna"`` needs order 1.
    frame : str, default "ENU"
        The layout's frame.
    origin_lla_deg_m : tuple of float or None, optional
        The ENU origin, kept on the layout.

    Raises
    ------
    ValueError
        If the axes, order or noise settings are invalid.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §6.2 and §6.3.2; see the module References.

    Examples
    --------
    >>> CartesianMotion("x").transition(np.array([100.0, 10.0]), 2.0)
    array([120.,  10.])
    """

    def __init__(
        self,
        axes: str | Sequence[str] = "xyz",
        order: Literal[1, 2] = 1,
        *,
        sigma_acceleration_mps2: float = 4.0,
        sigma_jerk_mps3: float = 1.0,
        acceleration_correlation_time_s: float = 1.0,
        process_noise: ProcessNoise = "cwna",
        frame: str = "ENU",
        origin_lla_deg_m: tuple[float, float, float] | None = None,
    ) -> None:
        axes = tuple(axes)
        if not axes or len(set(axes)) != len(axes) or set(axes) - set(_AXES) or order not in (1, 2):
            msg = f"axes must be unique, from x, y, z, and order 1 or 2; got {axes}, {order}."
            raise ValueError(msg)
        sigma = sigma_acceleration_mps2 if order == 1 else sigma_jerk_mps3
        q = _noise_intensity(sigma, acceleration_correlation_time_s, process_noise, order)
        kinds = (("_m", "m"), ("dot_mps", "m/s"), ("ddot_mps2", "m/s^2"))[: order + 1]
        layout = StateLayout(
            tuple(Coordinate(f"{a}{suffix}", unit) for a in axes for suffix, unit in kinds),
            frame,
            origin_lla_deg_m,
        )
        super().__init__(layout, [_axis_tables(order, process_noise, q) for _ in axes])


class RadialConstantVelocity(_PolynomialMotion):
    r"""Constant bistatic range rate, with random acceleration along the path.

    The state is ``[bistatic_range_m, bistatic_range_rate_mps]``; the rate is the range's
    derivative, so F is the textbook :math:`[[1, T], [0, 1]]`, with Q the constant-velocity
    block of :class:`CartesianMotion` [1]_. Monostatic, the bistatic range is twice the slant
    range, so its acceleration is twice the target's radial acceleration.

    Parameters
    ----------
    sigma_acceleration_mps2 : float, default 4.0
        Standard deviation of the bistatic range's acceleration.
    acceleration_correlation_time_s : float, default 1.0
        :math:`\tau` for ``"cwna"``.
    process_noise : {"cwna", "dwna"}, default "cwna"
        The noise model (module docstring).

    Raises
    ------
    ValueError
        If a setting is invalid.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §6.2 and §6.3.2; see the module References.

    Examples
    --------
    >>> RadialConstantVelocity().jacobian(np.zeros(2), 2.0)
    array([[1., 2.],
           [0., 1.]])
    """

    def __init__(
        self,
        sigma_acceleration_mps2: float = 4.0,
        *,
        acceleration_correlation_time_s: float = 1.0,
        process_noise: ProcessNoise = "cwna",
    ) -> None:
        tau_s = acceleration_correlation_time_s
        q = _noise_intensity(sigma_acceleration_mps2, tau_s, process_noise, 1)
        layout = StateLayout(
            (Coordinate("bistatic_range_m", "m"), Coordinate("bistatic_range_rate_mps", "m/s")),
            "radial",
        )
        super().__init__(layout, [_axis_tables(1, process_noise, q)])


class CoordinatedTurn:
    """Reserved: scenario 011's coordinated turn (D26), not yet implemented.

    The state is ``(x_m, xdot_mps, y_m, ydot_mps, turn_rate_radps)``. D26 adds the exact
    discretisation, its analytic Jacobian (the Taylor linearisation already predicts through
    ``jacobian``, so this is the EKF), the series form below ``|ωT| = 1e-6``, and noise of
    white acceleration on position plus a white turn-rate derivative.

    Parameters
    ----------
    sigma_acceleration_mps2 : float, default 4.0
        Standard deviation of the acceleration on position.
    turn_rate_noise_density_rad2ps3 : float, default 1e-4
        Intensity of the white turn-rate derivative.

    Raises
    ------
    NotImplementedError
        From every method, until scenario 011.
    """

    def __init__(
        self, sigma_acceleration_mps2: float = 4.0, turn_rate_noise_density_rad2ps3: float = 1e-4
    ) -> None:
        self.sigma_acceleration_mps2 = sigma_acceleration_mps2
        self.turn_rate_noise_density_rad2ps3 = turn_rate_noise_density_rad2ps3
        names = ("x_m", "xdot_mps", "y_m", "ydot_mps", "turn_rate_radps")
        units = ("m", "m/s", "m", "m/s", "rad/s")
        self.state_layout = StateLayout(tuple(map(Coordinate, names, units)), "ENU")

    def transition(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Not yet implemented."""
        raise NotImplementedError(_TURN_LATER)

    def jacobian(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Not yet implemented."""
        raise NotImplementedError(_TURN_LATER)

    def process_noise(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Not yet implemented."""
        raise NotImplementedError(_TURN_LATER)


_TURN_LATER = "CoordinatedTurn arrives with scenario 011 (D26)."
