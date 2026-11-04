r"""State layouts, motion models and measurement models, batched over leading axes.

Every model takes states ``(..., n_state)``, so a bank of tracks or all its sigma points go
through one call. :class:`StateLayout` names each coordinate and holds the package's one wrap
of periodic values. Motion: :class:`CartesianMotion` and :class:`RadialConstantVelocity`
(:class:`CoordinatedTurn` is reserved for scenario 011). Measurement: :class:`LinearMeasurement`,
:class:`RangeAzimuthElevationRangeRate`, :class:`BistaticRangeRateModel` and :func:`marginal`.

Conventions: ENU metres (x east, y north, z up). A range rate is the range's time derivative,
positive receding, and ``doppler_hz = -bistatic_range_rate_mps / wavelength_m``. Bistatic range
is the path R_T + R_R, twice the slant range when monostatic. Process noise ``"cwna"`` has
intensity q = 2 sigma² tau, Singer's model [2] over steps longer than tau, and is right for any
step; ``"dwna"`` holds one acceleration of standard deviation sigma through each step, as
scenario 003 specifies. The two add equal velocity variance per step only when tau = T/2.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
.. [2] R. A. Singer, "Estimating optimal tracking filter performance for manned maneuvering
       targets," *IEEE Trans. Aerospace and Electronic Systems*, vol. AES-6, no. 4,
       pp. 473-483, 1970.
.. [3] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005.
.. [4] S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, NoReturn, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import block_diag
from scipy.special import factorial

__all__ = [
    "BistaticRangeRateModel",
    "CartesianMotion",
    "Coordinate",
    "CoordinatedTurn",
    "LinearMeasurement",
    "MarginalModel",
    "MeasurementModel",
    "MotionModel",
    "RadialConstantVelocity",
    "RangeAzimuthElevationRangeRate",
    "StateLayout",
    "marginal",
]

ProcessNoise = Literal["cwna", "dwna"]

_AXES = ("x", "y", "z")


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
class StateLayout:
    """The ordered, named coordinates of a vector; code finds a coordinate by name only."""

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
        """Return ``a - b``, broadcast, the short way round for periodic coordinates."""
        return self.wrap(np.subtract(a, b, dtype=np.float64))


class MotionModel(Protocol):
    """A motion model; ``dt_s`` broadcasts against the states' leading axes ``(...)``."""

    state_layout: StateLayout

    def transition(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return the states moved on by ``dt_s``, ``(..., n_state)``."""
        ...

    def jacobian(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return the transition's Jacobian, ``(..., n_state, n_state)``."""
        ...

    def process_noise(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return Q, ``(..., n_state, n_state)``."""
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
    F_ij = T^(j-i) / (j-i)!; CWNA Q_ab = q T^(a+b+1) / ((a+b+1) a! b!); DWNA
    Q_ab = sigma² T^(a+b+2) / ((a+1)! (b+1)!), which is sigma² Γ Γᵀ with Γ = (T²/2, T).
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
    """A linear motion model whose F and Q are polynomials in T, from tables built once."""

    def __init__(self, layout: StateLayout, axis_tables: list[list[NDArray[np.float64]]]) -> None:
        self.state_layout = layout
        tables = [block_diag(*parts) for parts in zip(*axis_tables, strict=True)]
        self._f_coefficient, self._f_exponent, self._q_coefficient, self._q_exponent = tables

    def jacobian(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return F, the Jacobian of a linear model, ``(..., n_state, n_state)``."""
        step_s = np.asarray(dt_s, dtype=np.float64)[..., None, None]
        result: NDArray[np.float64] = self._f_coefficient * step_s**self._f_exponent
        return result

    def transition(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return F x, ``(..., n_state)``."""
        column = np.asarray(state, dtype=np.float64)[..., None]
        return (self.jacobian(state, dt_s) @ column)[..., 0]

    def process_noise(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return Q, ``(..., n_state, n_state)``; it does not depend on the state."""
        step_s = np.asarray(dt_s, dtype=np.float64)[..., None, None]
        result: NDArray[np.float64] = self._q_coefficient * step_s**self._q_exponent
        return result


class CartesianMotion(_PolynomialMotion):
    r"""Constant velocity (order 1) or acceleration (order 2) on each ENU axis.

    The state is interleaved per axis, ``[x_m, xdot_mps, (xddot_mps2), y_m, ...]``. F and Q
    come from closed-form tables, not a matrix exponential: F = [[1, T], [0, 1]],
    :math:`Q^{cwna} = q [T^3/3, T^2/2; T^2/2, T]`, :math:`Q^{dwna} = \sigma^2 [T^4/4, T^3/2;
    T^3/2, T^2]`. Order 1 reads ``sigma_acceleration_mps2``, order 2 ``sigma_jerk_mps3``; the
    defaults give q = 32 m²/s³, measured on scenario 001's truth.

    References
    ----------
    Bar-Shalom [1], §6.2 (continuous) and §6.3.2 (discrete white-noise acceleration).
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
        names = tuple(Coordinate(f"{a}{suffix}", unit) for a in axes for suffix, unit in kinds)
        layout = StateLayout(names, frame, origin_lla_deg_m)
        super().__init__(layout, [_axis_tables(order, process_noise, q) for _ in axes])


class RadialConstantVelocity(_PolynomialMotion):
    """Constant bistatic range rate: state ``[bistatic_range_m, bistatic_range_rate_mps]``.

    The rate is the range's derivative, positive receding, so F is the textbook one with no
    minus sign. Monostatic, the bistatic range is twice the slant range, so
    ``sigma_acceleration_mps2`` is twice the target's radial value.

    >>> RadialConstantVelocity().jacobian(np.zeros(2), 2.0)
    array([[1., 2.],
           [0., 1.]])

    References
    ----------
    Bar-Shalom [1], §6.2 and §6.3.2.
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
        names = (Coordinate("bistatic_range_m", "m"), Coordinate("bistatic_range_rate_mps", "m/s"))
        super().__init__(StateLayout(names, "radial"), [_axis_tables(1, process_noise, q)])


class CoordinatedTurn:
    """Reserved: scenario 011's coordinated turn (D26); every method raises until then.

    The state is ``(x_m, xdot_mps, y_m, ydot_mps, turn_rate_radps)``. D26 adds the exact
    discretisation and its Jacobian (the Taylor linearisation already predicts through
    ``jacobian``), the series form below ``|ωT| = 1e-6``, and white noise on acceleration and
    on the turn rate's derivative.
    """

    def __init__(
        self, sigma_acceleration_mps2: float = 4.0, turn_rate_noise_density_rad2ps3: float = 1e-4
    ) -> None:
        self.sigma_acceleration_mps2 = sigma_acceleration_mps2
        self.turn_rate_noise_density_rad2ps3 = turn_rate_noise_density_rad2ps3
        names = ("x_m", "xdot_mps", "y_m", "ydot_mps", "turn_rate_radps")
        units = ("m", "m/s", "m", "m/s", "rad/s")
        self.state_layout = StateLayout(tuple(map(Coordinate, names, units)), "ENU")

    def _not_yet(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NoReturn:
        msg = "CoordinatedTurn arrives with scenario 011 (D26)."
        raise NotImplementedError(msg)

    transition = jacobian = process_noise = _not_yet


class MeasurementModel(Protocol):
    """A measurement model: h(x) and its Jacobian, for states ``(..., n_state)``."""

    state_layout: StateLayout
    measurement_layout: StateLayout

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return h(x), ``(..., n_meas)``."""
        ...

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the Jacobian of h, ``(..., n_meas, n_state)``."""
        ...


class LinearMeasurement:
    """A sensor that reports some state coordinates directly: z = H x + w.

    Parameters
    ----------
    periods : mapping of str to float, optional
        A period for a reported coordinate whose *measurement* wraps, such as a range folded at
        the map's span; the state itself stays continuous.

    References
    ----------
    Bar-Shalom [1], §4.3.1.
    """

    def __init__(
        self,
        state_layout: StateLayout,
        names: Sequence[str],
        *,
        periods: Mapping[str, float] | None = None,
    ) -> None:
        names, periods = tuple(names), dict(periods or {})
        if not names or len(set(names)) != len(names) or set(periods) - set(names):
            msg = f"names must be unique, with periods only for them; got {names}, {periods}."
            raise ValueError(msg)
        self.state_layout = state_layout
        self.indices = state_layout.indices(names)
        reported = [state_layout.coordinates[i] for i in self.indices]
        coordinates = tuple(
            Coordinate(c.name, c.unit, periods.get(c.name, c.period)) for c in reported
        )
        frame, origin = state_layout.frame, state_layout.origin_lla_deg_m
        self.measurement_layout = StateLayout(coordinates, frame, origin)
        self.measurement_matrix = np.eye(state_layout.dimension)[list(self.indices)]

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the reported coordinates, ``(..., n_meas)``."""
        return np.asarray(state, dtype=np.float64)[..., list(self.indices)]

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return H for each state as one read-only view, ``(..., n_meas, n_state)``."""
        shape = (*np.shape(state)[:-1], *self.measurement_matrix.shape)
        # broadcast-exempt: H is the same for every state, so one view serves them all.
        return np.broadcast_to(self.measurement_matrix, shape)


def _site(position_m: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return a site's ENU position as a read-only (3,) array, or raise."""
    site = np.array(position_m, dtype=np.float64)
    if site.shape != (3,) or not np.all(np.isfinite(site)):
        msg = f"{name} must be three finite ENU metres; got {site!r}."
        raise ValueError(msg)
    site.setflags(write=False)
    return site


class _CartesianGeometry:
    """Read ENU position (and velocity) from a state by name; fixed values fill absent axes."""

    def __init__(
        self, layout: StateLayout, fixed: Mapping[str, float] | None, *, include_velocity: bool
    ) -> None:
        fixed = dict(fixed or {})
        suffixes = ("_m", "dot_mps") if include_velocity else ("_m",)
        absent = {f"{a}{s}" for s in suffixes for a in _AXES} - set(layout.names)
        # A missing coordinate is never guessed: each must be fixed, and only those may be.
        if set(fixed) != absent or not np.all(np.isfinite(list(fixed.values()))):
            msg = f"fixed must give a finite value for exactly {sorted(absent)}; got {fixed}."
            raise ValueError(msg)
        self.n_state = layout.dimension
        self._parts: dict[str, tuple[NDArray[np.bool_], NDArray[np.intp], NDArray[np.float64]]]
        self._parts = {}
        for suffix in suffixes:
            # Two lookups, position and velocity, done once per model.
            names = [f"{a}{suffix}" for a in _AXES]
            present = np.array([n in layout.names for n in names])
            columns = np.array([layout.names.index(n) if n in layout.names else 0 for n in names])
            self._parts[suffix] = (present, columns, np.array([fixed.get(n, 0.0) for n in names]))

    def vector(self, state: NDArray[np.float64], suffix: str) -> NDArray[np.float64]:
        """Return position (``"_m"``) or velocity (``"dot_mps"``), ``(..., 3)``."""
        present, columns, values = self._parts[suffix]
        result: NDArray[np.float64] = np.where(present, state[..., columns], values)
        return result

    def scatter(
        self, by_position: NDArray[np.float64], by_velocity: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        """Place derivatives by ENU position and velocity, ``(..., m, 3)``, in state columns."""
        jacobian = np.zeros((*by_position.shape[:-1], self.n_state))
        for suffix, derivative in (("_m", by_position), ("dot_mps", by_velocity)):
            # Two kinds of coordinate; a fixed axis has no state column to fill.
            if derivative is not None:
                present, columns, _ = self._parts[suffix]
                jacobian[..., columns[present]] = derivative[..., present]
        return jacobian


_SPHERICAL = (
    Coordinate("range_m", "m"),
    Coordinate("azimuth_rad", "rad", period=2 * np.pi),
    Coordinate("elevation_rad", "rad"),
    Coordinate("range_rate_mps", "m/s"),
)
_SPHERICAL_NAMES = tuple(c.name for c in _SPHERICAL)


class RangeAzimuthElevationRangeRate:
    r"""A monostatic radar's range, azimuth, elevation and range rate, chosen by name.

    With :math:`d = p - p_s`, :math:`r = |d|`, :math:`u = d / r` and ground range g: azimuth
    atan2(d_e, d_n) clockwise from north in [0, 2π), elevation atan2(d_u, g), and range rate
    :math:`u \cdot v`, positive receding. Jacobian rows on position: :math:`u^T`,
    :math:`(d_n, -d_e, 0)/g^2`, :math:`(-d_e d_u, -d_n d_u, g^2)/(r^2 g)` and
    :math:`(v - u (u \cdot v))^T / r`; the rate's row on velocity is :math:`u^T`. The site,
    and the zenith when an angle is reported, are singular.

    Parameters
    ----------
    sensor_enu_m : array_like
        The radar's site, ``(3,)``, in the state's ENU frame.
    fixed : mapping of str to float, optional
        A value for each position (and, with range rate, velocity) coordinate the state lacks.

    References
    ----------
    Bar-Shalom [1], §10.3.
    """

    def __init__(
        self,
        state_layout: StateLayout,
        sensor_enu_m: ArrayLike,
        components: Sequence[str] = _SPHERICAL_NAMES,
        fixed: Mapping[str, float] | None = None,
    ) -> None:
        components = tuple(components)
        if not components or len(set(components)) != len(components):
            msg = f"components must be nonempty and unique; got {components}."
            raise ValueError(msg)
        if set(components) - set(_SPHERICAL_NAMES):
            msg = f"components must come from {_SPHERICAL_NAMES}; got {components}."
            raise ValueError(msg)
        self.state_layout = state_layout
        self.sensor_enu_m = _site(sensor_enu_m, "sensor_enu_m")
        self.components = components
        self._rate = "range_rate_mps" in components
        self._geometry = _CartesianGeometry(state_layout, fixed, include_velocity=self._rate)
        reported = tuple(_SPHERICAL[_SPHERICAL_NAMES.index(c)] for c in components)
        self.measurement_layout = StateLayout(reported, "spherical")

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the reported components, ``(..., n_meas)``."""
        state = np.asarray(state, dtype=np.float64)
        offset_m = self._geometry.vector(state, "_m") - self.sensor_enu_m
        east_m, north_m, up_m = np.moveaxis(offset_m, -1, 0)
        range_m = np.linalg.norm(offset_m, axis=-1)
        values = {
            "range_m": range_m,
            "azimuth_rad": np.mod(np.arctan2(east_m, north_m), 2 * np.pi),
            "elevation_rad": np.arctan2(up_m, np.hypot(east_m, north_m)),
        }
        if self._rate:
            velocity_mps = self._geometry.vector(state, "dot_mps")
            values["range_rate_mps"] = np.sum(offset_m * velocity_mps, axis=-1) / range_m
        return np.stack([values[name] for name in self.components], axis=-1)

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the Jacobian, ``(..., n_meas, n_state)``."""
        state = np.asarray(state, dtype=np.float64)
        offset_m = self._geometry.vector(state, "_m") - self.sensor_enu_m
        east_m, north_m, up_m = np.moveaxis(offset_m, -1, 0)
        range_m: NDArray[np.float64] = np.linalg.norm(offset_m, axis=-1, keepdims=True)
        ground2_m2 = (east_m**2 + north_m**2)[..., None]
        unit = offset_m / range_m

        def by_position(name: str) -> NDArray[np.float64]:
            row: NDArray[np.float64] = unit
            if name == "azimuth_rad":
                row = np.stack([north_m, -east_m, 0 * up_m], axis=-1) / ground2_m2
            elif name == "elevation_rad":
                row = np.stack([-east_m * up_m, -north_m * up_m, ground2_m2[..., 0]], axis=-1)
                row = row / (range_m**2 * np.sqrt(ground2_m2))
            elif name == "range_rate_mps":
                velocity_mps = self._geometry.vector(state, "dot_mps")
                along_mps = np.sum(unit * velocity_mps, axis=-1, keepdims=True)
                row = (velocity_mps - unit * along_mps) / range_m
            return row

        # At most four rows, each built only if reported, so an unreported angle never divides
        # by a zero ground range.
        position_rows = np.stack([by_position(name) for name in self.components], axis=-2)
        velocity_rows = np.stack([unit * (c == "range_rate_mps") for c in self.components], -2)
        return self._geometry.scatter(position_rows, velocity_rows if self._rate else None)


class BistaticRangeRateModel:
    r"""Bistatic range :math:`R_T + R_R` and its rate, for a transmitter and a receiver.

    With :math:`u_k` the unit vector from site k to the target, the rate is :math:`(u_T + u_R)
    \cdot v`, positive receding. The Jacobian is :math:`(u_T + u_R)^T` for range on position
    and rate on velocity, and :math:`\sum_k (v - u_k (u_k \cdot v))^T / R_k` for rate on
    position. With the sites together it is monostatic, z = (2r, 2r'), with no special case.
    There is no excess-range option: the baseline is a constant a caller may subtract.

    Parameters
    ----------
    tx_enu_m, rx_enu_m : array_like
        The sites, ``(3,)`` each, in the state's ENU frame.
    fixed : mapping of str to float, optional
        A value for each position or velocity coordinate the state lacks.

    References
    ----------
    Willis [3], ch. 3.
    """

    def __init__(
        self,
        state_layout: StateLayout,
        tx_enu_m: ArrayLike,
        rx_enu_m: ArrayLike,
        fixed: Mapping[str, float] | None = None,
    ) -> None:
        self.state_layout = state_layout
        self.tx_enu_m = _site(tx_enu_m, "tx_enu_m")
        self.rx_enu_m = _site(rx_enu_m, "rx_enu_m")
        self._geometry = _CartesianGeometry(state_layout, fixed, include_velocity=True)
        names = (Coordinate("bistatic_range_m", "m"), Coordinate("bistatic_range_rate_mps", "m/s"))
        self.measurement_layout = StateLayout(names, "bistatic")

    def _legs(self, state: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        """Return each leg's unit vector ``(..., 2, 3)`` and length ``(..., 2)``, and v."""
        state = np.asarray(state, dtype=np.float64)
        sites_m = np.stack([self.tx_enu_m, self.rx_enu_m])
        offset_m = self._geometry.vector(state, "_m")[..., None, :] - sites_m
        leg_m: NDArray[np.float64] = np.linalg.norm(offset_m, axis=-1)
        return offset_m / leg_m[..., None], leg_m, self._geometry.vector(state, "dot_mps")

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``(bistatic_range_m, bistatic_range_rate_mps)``, ``(..., 2)``."""
        unit, leg_m, velocity_mps = self._legs(state)
        rate_mps = np.sum(unit.sum(axis=-2) * velocity_mps, axis=-1)
        return np.stack([leg_m.sum(axis=-1), rate_mps], axis=-1)

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the Jacobian, ``(..., 2, n_state)``."""
        unit, leg_m, velocity_mps = self._legs(state)
        unit_sum = unit.sum(axis=-2)
        along_mps = np.sum(unit * velocity_mps[..., None, :], axis=-1, keepdims=True)
        across = (velocity_mps[..., None, :] - unit * along_mps) / leg_m[..., None]
        by_position = np.stack([unit_sum, across.sum(axis=-2)], axis=-2)
        by_velocity = np.stack([0 * unit_sum, unit_sum], axis=-2)
        return self._geometry.scatter(by_position, by_velocity)


class MarginalModel:
    """The named components of ``base``: those rows of its prediction and Jacobian.

    References
    ----------
    Särkkä [4], Appendix A.1 (the marginal of a joint Gaussian).
    """

    def __init__(self, base: MeasurementModel, names: Sequence[str]) -> None:
        names = tuple(names)
        if not names or len(set(names)) != len(names):
            msg = f"marginal names must be nonempty and unique; got {names}."
            raise ValueError(msg)
        full = base.measurement_layout
        self.base = base
        self.indices = full.indices(names)
        self.state_layout = base.state_layout
        kept = tuple(full.coordinates[i] for i in self.indices)
        self.measurement_layout = StateLayout(kept, full.frame, full.origin_lla_deg_m)

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the kept components, ``(..., n_kept)``."""
        return self.base.predict(state)[..., list(self.indices)]

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the kept rows of the base Jacobian, ``(..., n_kept, n_state)``."""
        return self.base.jacobian(state)[..., list(self.indices), :]


def marginal(model: MeasurementModel, names: Sequence[str]) -> MarginalModel:
    """Return the marginal of ``model`` over the named components, such as range alone.

    A track that cannot yet trust a measured rate updates on its range through one
    (``spec/tracker-001.md`` §8.3).
    """
    return MarginalModel(model, names)
