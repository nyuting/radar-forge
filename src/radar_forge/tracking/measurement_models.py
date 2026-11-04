r"""Measurement models: h(x), the measurement a sensor would report without noise.

Each is batched over leading axes and gives its analytic Jacobian, so every linearisation can
use it: :class:`LinearMeasurement` (state coordinates directly),
:class:`RangeAzimuthElevationRangeRate` (a monostatic radar) and :class:`BistaticRangeRateModel`
(monostatic when its two sites coincide). :func:`marginal` keeps some components.

**Conventions.** Cartesian axes are local east-north-up (ENU) metres. A range rate is the time
derivative of a range, *positive receding*, and ``doppler_hz = -bistatic_range_rate_mps /
wavelength_m``. Bistatic range is the path :math:`R_T + R_R`, twice the slant range when
monostatic.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001, §5.2.1 (the linear measurement equation) and §10.3 (the
       Jacobian the extended Kalman filter needs).
.. [2] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005, ch. 3 (the bistatic
       range sum and its geometry).
.. [3] S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013,
       Appendix A.1 (the marginal of a joint Gaussian).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.tracking.models import Coordinate, StateLayout

__all__ = [
    "BistaticRangeRateModel",
    "LinearMeasurement",
    "MarginalModel",
    "MeasurementModel",
    "RangeAzimuthElevationRangeRate",
    "marginal",
]

_AXES = ("x", "y", "z")


class MeasurementModel(Protocol):
    """What a filter asks of a measurement model: h(x) and its Jacobian, batched.

    Attributes
    ----------
    state_layout : StateLayout
        Layout of the states it takes.
    measurement_layout : StateLayout
        Layout of the measurements it returns, with any periods.
    """

    state_layout: StateLayout
    measurement_layout: StateLayout

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return h(x), shape ``(..., n_meas)``, for states ``(..., n_state)``."""
        ...

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the Jacobian of h, shape ``(..., n_meas, n_state)``."""
        ...


class LinearMeasurement:
    """A sensor that reports some state coordinates directly: z = H x + w [1]_.

    Parameters
    ----------
    state_layout : StateLayout
        Layout of the state.
    names : sequence of str
        The coordinates reported, in measurement order.
    periods : mapping of str to float, optional
        A period for a reported coordinate whose *measurement* wraps, such as a range folded
        at the map's span; the state itself stays continuous.

    Attributes
    ----------
    indices : tuple of int
        Where each reported coordinate sits in the state.
    measurement_matrix : numpy.ndarray
        H, shape ``(n_meas, n_state)``.

    Raises
    ------
    ValueError
        If ``names`` is empty, repeats or is unknown, or ``periods`` names one not reported.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.2.1 (the linear measurement equation).

    Examples
    --------
    >>> from radar_forge.tracking.models import CartesianMotion
    >>> model = LinearMeasurement(CartesianMotion("x").state_layout, ("x_m",))
    >>> model.predict(np.array([[10.0, 2.0], [20.0, 3.0]]))
    array([[10.],
           [20.]])
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
        self.measurement_layout = StateLayout(
            tuple(Coordinate(c.name, c.unit, periods.get(c.name, c.period)) for c in reported),
            state_layout.frame,
            state_layout.origin_lla_deg_m,
        )
        self.measurement_matrix = np.eye(state_layout.dimension)[list(self.indices)]

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the reported coordinates, shape ``(..., n_meas)``."""
        return np.asarray(state, dtype=np.float64)[..., list(self.indices)]

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return H for each state, ``(..., n_meas, n_state)``, as a read-only view."""
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
        self._parts = {}
        for suffix in suffixes:
            # Two lookups, position and velocity, done once per model.
            names = [f"{a}{suffix}" for a in _AXES]
            present = np.array([n in layout.names for n in names])
            columns = np.array([layout.names.index(n) if n in layout.names else 0 for n in names])
            self._parts[suffix] = (present, columns, np.array([fixed.get(n, 0.0) for n in names]))

    def vector(self, state: NDArray[np.float64], suffix: str) -> NDArray[np.float64]:
        """Return position (``"_m"``) or velocity (``"dot_mps"``), shape ``(..., 3)``."""
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
    r"""What a monostatic radar reports: range, azimuth, elevation and range rate.

    With :math:`d = p - p_s`, :math:`r = |d|`, :math:`u = d / r` and ground range :math:`g`:
    range r; azimuth atan2(d_e, d_n), clockwise from north, in [0, 2π), period 2π; elevation
    atan2(d_u, g); range rate :math:`u \cdot v`, positive receding. The Jacobian rows on
    position are :math:`u^T`, :math:`(d_n, -d_e, 0) / g^2`, :math:`(-d_e d_u, -d_n d_u, g^2)
    / (r^2 g)` and :math:`(v - u (u \cdot v))^T / r`; the rate's row on velocity is
    :math:`u^T` [1]_. The site (r = 0) and, with an angle, the zenith (g = 0) are singular.

    Parameters
    ----------
    state_layout : StateLayout
        Layout of the Cartesian state.
    sensor_enu_m : array_like
        The radar's site, shape ``(3,)``, ENU metres in the state's frame.
    components : sequence of str, optional
        Those reported, in order, from ``range_m``, ``azimuth_rad``, ``elevation_rad`` and
        ``range_rate_mps``; default all four.
    fixed : mapping of str to float, optional
        A value for each position (and, with range rate, velocity) coordinate the state lacks.

    Raises
    ------
    ValueError
        If ``components`` is empty, repeats or is unknown, or ``fixed`` does not fill exactly
        the missing coordinates.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §10.3; see the module References.

    Examples
    --------
    A target 1 km east of the radar, receding at 10 m/s:

    >>> from radar_forge.tracking.models import CartesianMotion
    >>> model = RangeAzimuthElevationRangeRate(CartesianMotion().state_layout, np.zeros(3))
    >>> model.predict(np.array([1000.0, 10.0, 0.0, 0.0, 0.0, 0.0])).round(4).tolist()
    [1000.0, 1.5708, 0.0, 10.0]
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
        self.measurement_layout = StateLayout(
            tuple(_SPHERICAL[_SPHERICAL_NAMES.index(c)] for c in components), "spherical"
        )

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the reported components, shape ``(..., n_meas)``."""
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
        """Return the Jacobian, shape ``(..., n_meas, n_state)`` (class docstring)."""
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
    r"""Bistatic range and its rate, for a transmitter and a receiver at two sites.

    With :math:`u_k = (p - p_k) / R_k` the unit vector from site k to the target [1]_,
    :math:`z = (R_T + R_R,\ (u_T + u_R) \cdot v)`, the rate positive receding. The Jacobian is
    :math:`(u_T + u_R)^T` for range on position and rate on velocity, and :math:`\sum_k (v -
    u_k (u_k \cdot v))^T / R_k` for rate on position. With the sites together it is
    monostatic, :math:`z = (2r, 2\dot r)`, with no special case. There is no excess-range
    option: the baseline is a constant the caller may subtract. A target on a site is
    singular.

    Parameters
    ----------
    state_layout : StateLayout
        Layout of the Cartesian state.
    tx_enu_m, rx_enu_m : array_like
        The transmitter's and receiver's sites, shape ``(3,)``, ENU metres.
    fixed : mapping of str to float, optional
        A value for each position or velocity coordinate the state lacks.

    Raises
    ------
    ValueError
        If a site is not three finite numbers, or ``fixed`` does not fill exactly the missing
        coordinates.

    References
    ----------
    .. [1] Willis (2005), ch. 3; see the module References.

    Examples
    --------
    Co-sited, a target 13 m away: the path is 26 m.

    >>> from radar_forge.tracking.models import CartesianMotion
    >>> model = BistaticRangeRateModel(CartesianMotion().state_layout, np.zeros(3), np.zeros(3))
    >>> model.predict(np.array([3.0, 0.0, 4.0, 0.0, 12.0, 0.0]))
    array([26.,  0.])
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
        self.measurement_layout = StateLayout(
            (Coordinate("bistatic_range_m", "m"), Coordinate("bistatic_range_rate_mps", "m/s")),
            "bistatic",
        )

    def _legs(
        self, state: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return each leg's unit vector ``(..., 2, 3)`` and length ``(..., 2)``, and v."""
        state = np.asarray(state, dtype=np.float64)
        offset_m = self._geometry.vector(state, "_m")[..., None, :] - np.stack(
            [self.tx_enu_m, self.rx_enu_m]
        )
        leg_m: NDArray[np.float64] = np.linalg.norm(offset_m, axis=-1)
        return offset_m / leg_m[..., None], leg_m, self._geometry.vector(state, "dot_mps")

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``(bistatic_range_m, bistatic_range_rate_mps)``, shape ``(..., 2)``."""
        unit, leg_m, velocity_mps = self._legs(state)
        rate_mps = np.sum(unit.sum(axis=-2) * velocity_mps, axis=-1)
        return np.stack([leg_m.sum(axis=-1), rate_mps], axis=-1)

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the Jacobian, shape ``(..., 2, n_state)`` (class docstring)."""
        unit, leg_m, velocity_mps = self._legs(state)
        unit_sum = unit.sum(axis=-2)
        along_mps = np.sum(unit * velocity_mps[..., None, :], axis=-1, keepdims=True)
        across = (velocity_mps[..., None, :] - unit * along_mps) / leg_m[..., None]
        by_position = np.stack([unit_sum, across.sum(axis=-2)], axis=-2)
        by_velocity = np.stack([0 * unit_sum, unit_sum], axis=-2)
        return self._geometry.scatter(by_position, by_velocity)


class MarginalModel:
    """The named components of another measurement model; build one with :func:`marginal`.

    A Gaussian's marginal is those components of its mean and that block of its covariance
    [1]_, so this keeps the matching components of ``base.predict`` and rows of its Jacobian.

    Attributes
    ----------
    base : MeasurementModel
        The full model.
    indices : tuple of int
        Where each kept component sits in the base model's measurement.

    References
    ----------
    .. [1] Särkkä (2013), Appendix A.1; see the module References.
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
        self.measurement_layout = StateLayout(
            tuple(full.coordinates[i] for i in self.indices), full.frame, full.origin_lla_deg_m
        )

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the kept components, shape ``(..., n_kept)``."""
        return self.base.predict(state)[..., list(self.indices)]

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the kept rows of the base Jacobian, ``(..., n_kept, n_state)``."""
        return self.base.jacobian(state)[..., list(self.indices), :]


def marginal(model: MeasurementModel, names: Sequence[str]) -> MarginalModel:
    """Return the marginal of a measurement model over the named components.

    A track that cannot yet trust a measured rate updates on range alone through the
    marginal over the range (``spec/tracker-001.md`` §8.3).

    Parameters
    ----------
    model : MeasurementModel
        The full model.
    names : sequence of str
        The components kept, in order.

    Returns
    -------
    MarginalModel
        It has a Jacobian, so every linearisation can use it.

    Raises
    ------
    ValueError
        If ``names`` is empty, repeats, or names a component the model does not measure.

    Examples
    --------
    >>> from radar_forge.tracking.models import CartesianMotion
    >>> both = LinearMeasurement(CartesianMotion("x").state_layout, ("x_m", "xdot_mps"))
    >>> marginal(both, ("x_m",)).predict(np.array([10.0, 2.0]))
    array([10.])
    """
    return MarginalModel(model, names)
