r"""Measurement models: what a sensor reports of a state, batched over leading axes.

Every model takes states ``(..., n_state)`` in a :class:`~radar_forge.tracking.motion.StateLayout`
and reports in its own ``measurement_layout``, so a bank of tracks or all its sigma points go
through one call. :class:`LinearMeasurement` reports state coordinates directly;
:class:`RangeAzimuthElevationRangeRate` a monostatic radar's spherical measurement of an ENU
state; :class:`BistaticRangeRateModel` the path :math:`R_T + R_R` and its rate; :func:`marginal`
any of them over fewer components.

Conventions: ENU metres (x east, y north, z up), azimuth clockwise from north. A range rate is
the range's time derivative, positive receding, and ``doppler_hz = -bistatic_range_rate_mps /
wavelength_m``. Bistatic range is the path R_T + R_R, twice the slant range when monostatic.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
.. [2] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005.
.. [3] S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.tracking.motion import Coordinate, StateLayout

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
    atan2(d_e, d_n) clockwise from north in [0, 2π] (2π itself only when a tiny negative angle
    rounds up), elevation atan2(d_u, g), and range rate
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
        # Inline, not coordinates.enu_to_range_azimuth_elevation_rad: elevation here is
        # atan2(up, ground), where that function takes asin(up / r). About half of all
        # directions differ, by up to a few hundred ulp (1e-13 rad), and near the zenith
        # asin(up / r) is off by up to 2e-8 rad where atan2 is not.
        values = {
            "range_m": range_m,
            # Unguarded: a tiny negative angle rounds up to exactly 2 pi, where _periodic.fold
            # gives 0.
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
    Willis [2], ch. 3.
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
    Särkkä [3], Appendix A.1 (the marginal of a joint Gaussian).
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
    (``spec/tracker-001.md`` §5.4).

    References
    ----------
    Särkkä [3], Appendix A.1 (the marginal of a joint Gaussian).
    """
    return MarginalModel(model, names)
