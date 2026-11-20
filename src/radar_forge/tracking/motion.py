r"""Coordinate layouts and motion models, batched over leading axes.

Every model takes states ``(..., n_state)``, so a bank of tracks or all its sigma points go
through one call. :class:`CoordinateLayout` names each coordinate and wraps its periodic values;
:func:`check_covariances` is the one boundary check of a covariance. Motion:
:class:`ConstantVelocity` (on ENU axes, or on bistatic range by
:meth:`ConstantVelocity.bistatic_range`), :class:`ConstantAcceleration` and the nonlinear
:class:`ConstantTurn`; what a sensor reports of a state is in
:mod:`~radar_forge.tracking.measurement`.

Conventions: ENU metres (x east, y north, z up). A range rate is the range's time derivative,
positive receding, and ``doppler_hz = -bistatic_range_rate_mps / wavelength_m``. Bistatic range
is the path R_T + R_R, twice the slant range when monostatic. ``white_noise = "continuous"`` is
continuous white-noise acceleration, whose Q is exact for any step under that model; its
intensity q = 2 sigma² tau is Singer's [2], an approximation to a correlated acceleration that
holds only for steps long against tau. ``"discrete"`` holds one acceleration of standard
deviation sigma through each step, as scenario 003 specifies. The two add equal velocity
variance per step only when tau = T/2.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
.. [2] R. A. Singer, "Estimating optimal tracking filter performance for manned maneuvering
       targets," *IEEE Trans. Aerospace and Electronic Systems*, vol. AES-6, no. 4,
       pp. 473-483, 1970.
.. [3] X. R. Li and V. P. Jilkov, "Survey of maneuvering target tracking. Part I: Dynamic
       models," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 39, no. 4, pp. 1333-1364, 2003.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import block_diag
from scipy.special import factorial

from radar_forge._periodic import fold
from radar_forge.tracking._linalg import symmetrise

__all__ = [
    "ConstantAcceleration",
    "ConstantTurn",
    "ConstantVelocity",
    "Coordinate",
    "CoordinateLayout",
    "MotionModel",
    "check_covariances",
]

WhiteNoise = Literal["continuous", "discrete"]

_AXES = ("x", "y", "z")

# The roundoff a covariance may carry, relative to its largest entry: products of a few small
# matrices carry about n float64 epsilons, 1e-15, and 1e-10 leaves a wide margin over that.
_ROUNDOFF_TOLERANCE = 1e-10


def _is_positive_definite(matrix: NDArray[np.float64]) -> bool:
    """Return whether a Cholesky factorisation of ``matrix`` succeeds."""
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True


def check_covariances(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    """Return a float64 copy of covariances of ``shape``, checked symmetric and positive definite.

    Definite, not semidefinite: the unscented transform factorises every covariance and S must be
    invertible, so a zero variance is refused here, at the boundary, and never mid-scan.

    References
    ----------
    Bar-Shalom [1], §5.2.1 (the state estimate and its covariance).
    """
    covariance = np.array(value, dtype=np.float64)
    if covariance.shape != shape or not np.all(np.isfinite(covariance)):
        msg = f"{name} must be finite with shape {shape}; got shape {covariance.shape}."
        raise ValueError(msg)
    transposed = np.swapaxes(covariance, -1, -2)
    allowance = _ROUNDOFF_TOLERANCE * np.max(np.abs(covariance), axis=(-2, -1), initial=0.0)
    if np.any(np.max(np.abs(covariance - transposed), axis=(-2, -1), initial=0.0) > allowance):
        msg = f"{name} must be symmetric to within roundoff."
        raise ValueError(msg)
    covariance = symmetrise(covariance)
    if not _is_positive_definite(covariance):
        stack = covariance.reshape(-1, *shape[-2:])
        # Only on the error path: one factorisation per matrix names the ones that fail.
        rows = [i for i, matrix in enumerate(stack) if not _is_positive_definite(matrix)]
        msg = f"{name} must be positive definite; the matrices at rows {rows} are not."
        raise ValueError(msg)
    return covariance


@dataclass(frozen=True)
class Coordinate:
    """One named number of a state or measurement, with its unit and period.

    Parameters
    ----------
    period : float or None, optional
        Length of one repeat in ``unit`` (2π for an azimuth, the unambiguous range for a
        folded range); None for a value that does not repeat.
    """

    name: str
    unit: str
    period: float | None = None

    def __post_init__(self) -> None:
        """Check the name, unit and period."""
        if not (self.name and self.unit) or (
            self.period is not None and not (np.isfinite(self.period) and self.period > 0)
        ):
            msg = f"a coordinate needs a name, a unit, and a period > 0 or None; got {self}."
            raise ValueError(msg)


@dataclass(frozen=True)
class CoordinateLayout:
    """The ordered, named coordinates of a vector; code finds a coordinate by name only."""

    coordinates: tuple[Coordinate, ...]
    _periodic: NDArray[np.bool_] = field(init=False, repr=False, compare=False)
    _period: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the coordinates and find the periodic ones once."""
        coordinates = tuple(self.coordinates)
        names = [c.name for c in coordinates]
        if not coordinates or len(set(names)) != len(names):
            msg = f"a layout needs coordinates with unique names; got {names}."
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
        """Return the positions of the named coordinates, in the order asked for."""
        unknown = [name for name in names if name not in self.names]
        if unknown:
            msg = f"coordinates {unknown} are not in the layout {self.names}."
            raise ValueError(msg)
        return tuple(self.names.index(name) for name in names)

    def wrap(
        self, value: ArrayLike, *, interval: Literal["centred", "nonnegative"] = "centred"
    ) -> NDArray[np.float64]:
        """Return a copy with each periodic coordinate in [-p/2, p/2), or [0, p) if nonnegative.

        >>> layout = CoordinateLayout((Coordinate("range_m", "m", period=1000.0),))
        >>> layout.wrap([700.0]), layout.wrap([-300.0], interval="nonnegative")
        (array([-300.]), array([700.]))
        """
        if interval not in ("centred", "nonnegative"):
            msg = f"interval must be 'centred' or 'nonnegative'; got {interval!r}."
            raise ValueError(msg)
        result = np.array(value, dtype=np.float64)
        if not self._period.size:
            return result
        lower = -self._period / 2 if interval == "centred" else np.zeros_like(self._period)
        result[..., self._periodic] = fold(result[..., self._periodic], self._period, lower=lower)
        return result

    def residual(self, a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
        """Return ``a - b``, broadcast, the short way round for periodic coordinates."""
        return self.wrap(np.subtract(a, b, dtype=np.float64))


class MotionModel(Protocol):
    """A motion model: ``transition`` moves states and ``jacobian`` is its derivative.

    ``dt_s`` broadcasts against the states' leading axes ``(...)``. A linear model's
    ``transition(x)`` is ``jacobian(x) @ x``; the constant turn's is not, so the Taylor
    linearisation moves the mean through ``transition`` and the covariance through F.
    """

    state_layout: CoordinateLayout

    def transition(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return f(x), the states moved on by ``dt_s``, ``(..., n_state)``."""
        ...

    def jacobian(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return F, the transition's Jacobian, ``(..., n_state, n_state)``."""
        ...

    def process_noise(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return Q, ``(..., n_state, n_state)``."""
        ...


def _noise_intensity(sigma: float, tau_s: float, white_noise: str, order: int) -> float:
    """Return q = 2 sigma² tau if continuous, sigma² if discrete, after checking the settings."""
    if white_noise not in ("continuous", "discrete") or (white_noise == "discrete" and order != 1):
        msg = f"white_noise must be 'continuous', or 'discrete' at order 1; got {white_noise!r}."
        raise ValueError(msg)
    if not (np.isfinite(sigma) and sigma >= 0 and np.isfinite(tau_s) and tau_s > 0):
        msg = f"need a finite sigma >= 0 and correlation time > 0; got {sigma}, {tau_s}."
        raise ValueError(msg)
    return 2.0 * sigma**2 * tau_s if white_noise == "continuous" else sigma**2


def _axis_tables(order: int, white_noise: str, intensity: float) -> list[NDArray[np.float64]]:
    """Return one axis's F and Q as coefficient and exponent tables in T, position first.

    With i, j counting up from position and a, b down from the driven derivative:
    F_ij = T^(j-i) / (j-i)!; continuous Q_ab = q T^(a+b+1) / ((a+b+1) a! b!); discrete
    Q_ab = sigma² T^(a+b+2) / ((a+1)! (b+1)!), which is sigma² Γ Γᵀ with Γ = (T²/2, T).
    """
    i, j = np.meshgrid(np.arange(order + 1), np.arange(order + 1), indexing="ij")
    a, b = order - i, order - j
    f_exponent = np.maximum(j - i, 0)
    f_coefficient = np.where(j >= i, 1.0 / factorial(f_exponent), 0.0)
    if white_noise == "continuous":
        q_exponent = a + b + 1
        q_coefficient = intensity / (q_exponent * factorial(a) * factorial(b))
    else:
        q_exponent = a + b + 2
        q_coefficient = intensity / (factorial(a + 1) * factorial(b + 1))
    return [f_coefficient, f_exponent.astype(float), q_coefficient, q_exponent.astype(float)]


class _PolynomialMotion:
    """A linear motion model whose F and Q are polynomials in T, from tables built once."""

    def __init__(
        self, layout: CoordinateLayout, axis_tables: list[list[NDArray[np.float64]]]
    ) -> None:
        self.state_layout = layout
        tables = [block_diag(*parts) for parts in zip(*axis_tables, strict=True)]
        self._f_coefficient, self._f_exponent, self._q_coefficient, self._q_exponent = tables

    def jacobian(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return F, the Jacobian of a linear model, ``(..., n_state, n_state)``."""
        step_s = np.asarray(dt_s, dtype=np.float64)[..., None, None]
        result: NDArray[np.float64] = self._f_coefficient * step_s**self._f_exponent
        return result

    def transition(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return f(x) = F x, ``(..., n_state)``."""
        column = np.asarray(state, dtype=np.float64)[..., None]
        return (self.jacobian(state, dt_s) @ column)[..., 0]

    def process_noise(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return Q, ``(..., n_state, n_state)``; it does not depend on the state."""
        step_s = np.asarray(dt_s, dtype=np.float64)[..., None, None]
        result: NDArray[np.float64] = self._q_coefficient * step_s**self._q_exponent
        return result


def _per_axis(
    axes: str | Sequence[str], order: int, sigma: float, tau_s: float, white_noise: str
) -> tuple[CoordinateLayout, list[list[NDArray[np.float64]]]]:
    """Return the per-axis interleaved layout and each axis's tables, after checking the axes."""
    axes = tuple(axes)
    if not axes or len(set(axes)) != len(axes) or set(axes) - set(_AXES):
        msg = f"axes must be unique, from x, y, z; got {axes}."
        raise ValueError(msg)
    q = _noise_intensity(sigma, tau_s, white_noise, order)
    kinds = (("_m", "m"), ("dot_mps", "m/s"), ("ddot_mps2", "m/s^2"))[: order + 1]
    names = tuple(Coordinate(f"{a}{suffix}", unit) for a in axes for suffix, unit in kinds)
    return CoordinateLayout(names), [_axis_tables(order, white_noise, q) for _ in axes]


class ConstantVelocity(_PolynomialMotion):
    r"""Constant velocity on each ENU axis, or on bistatic range, driven by white acceleration.

    The state is interleaved per axis, ``[x_m, xdot_mps, y_m, ydot_mps, ...]``. F and Q come
    from closed-form tables, not a matrix exponential: F = [[1, T], [0, 1]], continuous
    :math:`Q = q [T^3/3, T^2/2; T^2/2, T]`, discrete :math:`Q = \sigma^2 [T^4/4, T^3/2; T^3/2,
    T^2]`. The defaults give q = 32 m²/s³, measured on scenario 001's truth.
    :meth:`bistatic_range` builds the same tables on bistatic range and its rate.

    References
    ----------
    Bar-Shalom [1], §6.2 (continuous) and §6.3.2 (discrete white-noise acceleration).
    """

    def __init__(
        self,
        axes: str | Sequence[str] = "xyz",
        *,
        sigma_acceleration_mps2: float = 4.0,
        acceleration_correlation_time_s: float = 1.0,
        white_noise: WhiteNoise = "continuous",
    ) -> None:
        tau_s = acceleration_correlation_time_s
        layout, tables = _per_axis(axes, 1, sigma_acceleration_mps2, tau_s, white_noise)
        super().__init__(layout, tables)

    @classmethod
    def bistatic_range(
        cls,
        sigma_acceleration_mps2: float = 4.0,
        *,
        acceleration_correlation_time_s: float = 1.0,
        white_noise: WhiteNoise = "continuous",
    ) -> ConstantVelocity:
        """Return constant velocity on ``[bistatic_range_m, bistatic_range_rate_mps]``.

        The rate is the range's derivative, positive receding, so F is the textbook one with no
        minus sign. Monostatic, the bistatic range is twice the slant range, so
        ``sigma_acceleration_mps2`` is twice the target's radial value.

        >>> ConstantVelocity.bistatic_range().jacobian(np.zeros(2), 2.0)
        array([[1., 2.],
               [0., 1.]])
        """
        tau_s = acceleration_correlation_time_s
        q = _noise_intensity(sigma_acceleration_mps2, tau_s, white_noise, 1)
        names = (Coordinate("bistatic_range_m", "m"), Coordinate("bistatic_range_rate_mps", "m/s"))
        # One axis whose names are not an ENU axis's: the tables are __init__'s, the layout not.
        model = cls.__new__(cls)
        _PolynomialMotion.__init__(
            model, CoordinateLayout(names), [_axis_tables(1, white_noise, q)]
        )
        return model


class ConstantAcceleration(_PolynomialMotion):
    r"""Constant acceleration on each ENU axis, driven by continuous white jerk.

    The state is interleaved per axis, ``[x_m, xdot_mps, xddot_mps2, y_m, ...]``, with
    :math:`Q = q [T^5/20, T^4/8, T^3/6; T^4/8, T^3/3, T^2/2; T^3/6, T^2/2, T]`,
    q = 2 sigma_jerk² tau. There is no discrete form.

    References
    ----------
    Bar-Shalom [1], §6.2 (continuous white-noise models).
    """

    def __init__(
        self,
        axes: str | Sequence[str] = "xyz",
        *,
        sigma_jerk_mps3: float = 1.0,
        acceleration_correlation_time_s: float = 1.0,
        white_noise: WhiteNoise = "continuous",
    ) -> None:
        tau_s = acceleration_correlation_time_s
        layout, tables = _per_axis(axes, 2, sigma_jerk_mps3, tau_s, white_noise)
        super().__init__(layout, tables)


# Below this |omega T| the turn's coefficients are their Taylor series to second order: the
# exact forms divide by omega, and the next series term is (omega T)^4, 1e-24, of the first.
_SERIES_TURN_RAD = 1e-6
# The one coefficient whose exact form cancels badly, the along-track term's rate derivative,
# is a series to a wider bound (see ConstantTurn._coefficients).
_SERIES_DERIVATIVE_RAD = 1e-2


class ConstantTurn:
    r"""Constant turn, also called the coordinated turn: speed and turn rate held, rate in state.

    State ``[x_m, xdot_mps, y_m, ydot_mps, turn_rate_radps]`` in the ground plane, the rate
    being the heading's (clockwise from north, positive to the right; Li and Jilkov's ω with its
    sign reversed). With :math:`s = \sin\omega T`, :math:`c = \cos\omega T`: :math:`x' = x +
    (s/\omega)\dot x + ((1-c)/\omega)\dot y`, :math:`\dot x' = c\dot x + s\dot y`, :math:`y' =
    y - ((1-c)/\omega)\dot x + (s/\omega)\dot y`, :math:`\dot y' = -s\dot x + c\dot y`, ω held;
    1 - c is written 2 sin²(ωT/2), free of cancellation. Below :math:`|\omega T| = 10^{-6}` the
    coefficients are their series, so F tends to the CV matrix. Q is CV's continuous one per axis
    (q = 2 sigma² tau) and ``turn_rate_noise_density_rad2ps3`` times T on the rate.

    References
    ----------
    Li and Jilkov [3], §V (the coordinated turn with unknown turn rate); Bar-Shalom [1], §6.2.2.
    """

    def __init__(
        self,
        sigma_acceleration_mps2: float = 0.5,
        turn_rate_noise_density_rad2ps3: float = 1.6e-5,
        *,
        acceleration_correlation_time_s: float = 1.0,
    ) -> None:
        tau_s = acceleration_correlation_time_s
        q = _noise_intensity(sigma_acceleration_mps2, tau_s, "continuous", 1)
        if not (
            np.isfinite(turn_rate_noise_density_rad2ps3) and turn_rate_noise_density_rad2ps3 >= 0
        ):
            msg = f"need a finite turn-rate density >= 0; got {turn_rate_noise_density_rad2ps3}."
            raise ValueError(msg)
        names = ("x_m", "xdot_mps", "y_m", "ydot_mps", "turn_rate_radps")
        units = ("m", "m/s", "m", "m/s", "rad/s")
        self.state_layout = CoordinateLayout(tuple(map(Coordinate, names, units)))
        tables = _axis_tables(1, "continuous", q)
        self._q_coefficient = block_diag(tables[2], tables[2], [[turn_rate_noise_density_rad2ps3]])
        self._q_exponent = block_diag(tables[3], tables[3], [[1.0]])

    @staticmethod
    def _coefficients(
        turn_rate_radps: NDArray[np.float64], dt_s: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], ...]:
        """Return sin ωT/ω, (1 - cos ωT)/ω, cos ωT, sin ωT and the first two's ω-derivatives."""
        angle = turn_rate_radps * dt_s
        series = np.abs(angle) < _SERIES_TURN_RAD
        rate = np.where(series, 1.0, turn_rate_radps)
        sine, cosine = np.sin(angle), np.cos(angle)
        along = np.where(series, dt_s * (1 - angle**2 / 6), sine / rate)
        across = np.where(series, angle * dt_s / 2, 2 * np.sin(angle / 2) ** 2 / rate)
        # (T cos ωT - sin ωT / ω) / ω loses (ωT)⁻² of its precision to cancellation; below
        # |ωT| = 1e-2 its series in x = ωT, T² (-x/3 + x³/30 - x⁵/840), is exact to roundoff.
        short = np.abs(angle) < _SERIES_DERIVATIVE_RAD
        d_along = np.where(
            short,
            dt_s**2 * (-angle / 3 + angle**3 / 30 - angle**5 / 840),
            (dt_s * cosine - along) / np.where(short, 1.0, rate),
        )
        d_across = np.where(series, dt_s**2 * (0.5 - angle**2 / 8), (dt_s * sine - across) / rate)
        return along, across, cosine, sine, d_along, d_across

    def _parts(
        self, state: NDArray[np.float64], dt_s: ArrayLike
    ) -> tuple[NDArray[np.float64], tuple[NDArray[np.float64], ...]]:
        """Return the states and their coefficients, broadcast to one leading shape."""
        state = np.asarray(state, dtype=np.float64)
        step_s = np.broadcast_to(np.asarray(dt_s, dtype=np.float64), state.shape[:-1])
        return state, self._coefficients(state[..., 4], step_s)

    def transition(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return f(x), the states moved on along their turns, ``(..., 5)``."""
        state, (along, across, cosine, sine, _, _) = self._parts(state, dt_s)
        x, xdot, y, ydot, rate = np.moveaxis(state, -1, 0)
        return np.stack(
            [
                x + along * xdot + across * ydot,
                cosine * xdot + sine * ydot,
                y - across * xdot + along * ydot,
                -sine * xdot + cosine * ydot,
                rate,
            ],
            axis=-1,
        )

    def jacobian(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return F, the transition's Jacobian at each state, ``(..., 5, 5)``."""
        state, (along, across, cosine, sine, d_along, d_across) = self._parts(state, dt_s)
        _, xdot, _, ydot, _ = np.moveaxis(state, -1, 0)
        step_s = np.broadcast_to(np.asarray(dt_s, dtype=np.float64), xdot.shape)
        one, zero = np.ones_like(xdot), np.zeros_like(xdot)
        rows = [
            [one, along, zero, across, d_along * xdot + d_across * ydot],
            [zero, cosine, zero, sine, step_s * (cosine * ydot - sine * xdot)],
            [zero, -across, one, along, -d_across * xdot + d_along * ydot],
            [zero, -sine, zero, cosine, -step_s * (cosine * xdot + sine * ydot)],
            [zero, zero, zero, zero, one],
        ]
        return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)

    def process_noise(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return Q, ``(..., 5, 5)``; it does not depend on the state."""
        step_s = np.asarray(dt_s, dtype=np.float64)[..., None, None]
        result: NDArray[np.float64] = self._q_coefficient * step_s**self._q_exponent
        return result
