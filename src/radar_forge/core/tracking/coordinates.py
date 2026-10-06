"""Named coordinates, wrapping of periodic values, and frozen state estimates.

A tracking state is a vector of numbers, such as a position and a velocity. This
module gives each number a name and a unit, so that models look coordinates up by
name instead of relying on a fixed order:

- :class:`Coordinate` is one named number, with its unit and, for a value that
  repeats (an angle, or a folded range), its period.
- :class:`StateLayout` is the ordered list of coordinates in a state vector. It
  also knows how to wrap periodic values, subtract two states, and average many.
- :class:`StateEstimate` is a frozen snapshot of a filter: the mean, the
  covariance, the time and the layout.

**Axis convention.** Cartesian layouts name their axes x, y and z: x is east, y is
north and z is up. They are metres in the local east-north-up (ENU) frame of
:mod:`radar_forge.core.geodesy`, centred on the layout's ``origin_lla_deg_m``.

A periodic coordinate is one whose values repeat every ``period``: an azimuth
repeats every 2π rad, and a range measured by a pulsed radar repeats every
unambiguous range. Two values that differ by a whole number of periods are the
same physical value, so subtraction and averaging must take the wrap into account.

References
----------
.. [1] K. V. Mardia and P. E. Jupp, *Directional Statistics*, Wiley, 2000, §2.2.1
       (the mean direction of circular data, used by
       :meth:`StateLayout.weighted_mean`).
.. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications
       to Tracking and Navigation*, Wiley, 2001, §5.2.1 (the state estimate and its
       covariance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.core.tracking._validation import (
    as_covariance,
    as_vector,
    check_frame_and_origin,
    check_points,
)

__all__ = [
    "Coordinate",
    "StateEstimate",
    "StateLayout",
]

# Below this length, the average of the unit vectors is treated as zero: the
# points are spread evenly round the circle, so they have no mean direction.
_MIN_RESULTANT_LENGTH = 1e-12

# How far the mean weights may sum from one. UKF weights are a few fractions such
# as 1/(2n), and their float64 sum is off from one by about n times machine epsilon,
# 1e-15. rtol 1e-10 and atol 1e-12 leave a wide margin over that, and still catch
# any weight that is wrong rather than rounded.
_WEIGHT_SUM_RTOL = 1e-10
_WEIGHT_SUM_ATOL = 1e-12


@dataclass(frozen=True)
class Coordinate:
    """One named number in a state or a measurement: its name, unit and period.

    Parameters
    ----------
    name : str
        Name with a unit suffix, such as ``"x_m"``. It must be unique within a
        :class:`StateLayout`.
    unit : str
        Physical unit, such as ``"m"`` or ``"m/s"``.
    period : float or None, optional
        For a value that repeats, the length of one repeat, in ``unit``. It is
        2π for an angle in radians such as ``azimuth_rad``. For a folded
        ``range_m`` it is the unambiguous range in metres. None (the default)
        means the value does not repeat.

    Raises
    ------
    ValueError
        If ``name`` or ``unit`` is empty, or ``period`` is not finite and
        positive.

    Examples
    --------
    >>> import numpy as np
    >>> Coordinate("azimuth_rad", "rad", period=2 * np.pi).unit
    'rad'
    """

    name: str
    unit: str
    period: float | None = None

    def __post_init__(self) -> None:
        """Check the name, unit and period."""
        if not self.name or not self.unit:
            msg = f"coordinate name and unit must be nonempty; got {self.name!r}, {self.unit!r}."
            raise ValueError(msg)
        if self.period is not None and not (np.isfinite(self.period) and self.period > 0):
            msg = f"coordinate period must be finite and positive; got {self.period!r}."
            raise ValueError(msg)


@dataclass(frozen=True)
class StateLayout:
    """The ordered, named coordinates of a vector, and the frame they live in.

    Models use a layout to find a coordinate by name, so that no model depends on
    the order of the state vector. Two models can work together only if their
    layouts are equal: the same coordinates in the same order, the same frame and
    the same origin.

    Parameters
    ----------
    coordinates : tuple of Coordinate
        The coordinates, in vector order. Names must be unique. Each covariance
        entry has the product of its two coordinates' units.
    frame : str, optional
        Name of the frame, such as ``"ENU"``. Default ``"local"``.
    origin_lla_deg_m : tuple of float or None, optional
        Origin of the east-north-up frame: latitude and longitude in degrees,
        altitude in metres. The built-in Cartesian models need it; a custom layout
        that is not tied to the Earth may leave it as None (the default).

    Raises
    ------
    ValueError
        If ``coordinates`` is empty or repeats a name, ``frame`` is empty, or
        ``origin_lla_deg_m`` is not three finite numbers with latitude in
        [-90, 90] and longitude in [-180, 180].

    Examples
    --------
    >>> layout = StateLayout((Coordinate("x_m", "m"), Coordinate("xdot_mps", "m/s")))
    >>> layout.names
    ('x_m', 'xdot_mps')
    >>> layout.indices(("xdot_mps",))
    (1,)
    """

    coordinates: tuple[Coordinate, ...]
    frame: str = "local"
    origin_lla_deg_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        """Check the coordinates and origin, and store both as tuples."""
        coordinates = tuple(self.coordinates)
        names = [c.name for c in coordinates]
        if not coordinates or len(set(names)) != len(names):
            msg = f"state layout requires unique coordinates; got names {names}."
            raise ValueError(msg)
        origin = check_frame_and_origin(self.frame, self.origin_lla_deg_m)

        # The coordinates are stored as a tuple, not the list a caller may pass, so that
        # they cannot change after construction. The dataclass is frozen, so the checked
        # tuples are stored through object.
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "origin_lla_deg_m", origin)

    @property
    def names(self) -> tuple[str, ...]:
        """Coordinate names, in vector order."""
        return tuple(c.name for c in self.coordinates)

    @property
    def dimension(self) -> int:
        """Number of coordinates, which is the length of the vector."""
        return len(self.coordinates)

    def indices(self, names: tuple[str, ...]) -> tuple[int, ...]:
        """Return the vector positions of the named coordinates.

        Parameters
        ----------
        names : tuple of str
            Coordinate names, in the order wanted.

        Returns
        -------
        tuple of int
            The position of each name in the vector, in the same order as ``names``.

        Raises
        ------
        ValueError
            If a name is not in this layout.
        """
        if not set(names) <= set(self.names):
            unknown = [name for name in names if name not in self.names]
            msg = f"coordinates {unknown} are not in the layout {self.names}."
            raise ValueError(msg)
        return tuple(self.names.index(name) for name in names)

    def wrap(
        self,
        value: ArrayLike,
        *,
        interval: Literal["centred", "nonnegative"] = "centred",
    ) -> NDArray[np.float64]:
        """Return a copy with every periodic coordinate wrapped into one period.

        Coordinates without a period are copied unchanged.

        Parameters
        ----------
        value : array_like
            One vector, shape ``(n_state,)``, or a stack of vectors,
            shape ``(n_points, n_state)``, in this layout's order and units.
        interval : {"centred", "nonnegative"}, optional
            Where to put each wrapped value. ``"centred"`` (the default) uses
            [-period/2, period/2). Use it for residuals, the difference between
            two values, so that the shorter way round is chosen. ``"nonnegative"``
            uses [0, period). Use it for values shown to a person or written to a
            file, such as a folded range.

        Returns
        -------
        numpy.ndarray
            Wrapped float64 copy, the same shape as ``value``.

        Raises
        ------
        ValueError
            If ``value`` has the wrong shape or is not finite, or ``interval`` is
            not one of the two names above.

        Examples
        --------
        >>> import numpy as np
        >>> layout = StateLayout((Coordinate("range_m", "m", period=1000.0),))
        >>> layout.wrap(np.array([700.0]))
        array([-300.])
        >>> layout.wrap(np.array([-300.0]), interval="nonnegative")
        array([700.])
        """
        result = np.array(value, dtype=np.float64, copy=True)
        check_points(result, self.dimension, "value")
        if interval not in ("centred", "nonnegative"):
            msg = f"interval must be 'centred' or 'nonnegative'; got {interval!r}."
            raise ValueError(msg)

        periodic = np.array([c.period is not None for c in self.coordinates])
        period = np.array([c.period for c in self.coordinates if c.period is not None])
        offset = period / 2 if interval == "centred" else np.zeros_like(period)
        wrapped = np.mod(result[..., periodic] + offset, period)
        # np.mod of a tiny negative number can round up to exactly one period,
        # which lies outside the half-open interval; fold it back to zero.
        wrapped = np.where(wrapped >= period, wrapped - period, wrapped)
        result[..., periodic] = wrapped - offset
        return result

    def normalise(self, value: ArrayLike) -> NDArray[np.float64]:
        """Return a copy with periodic coordinates wrapped to [-period/2, period/2).

        This is :meth:`wrap` with ``interval="centred"``.

        Parameters
        ----------
        value : array_like
            One vector, shape ``(n_state,)``, or a stack, shape ``(n_points, n_state)``.

        Returns
        -------
        numpy.ndarray
            Wrapped float64 copy, the same shape as ``value``.

        Raises
        ------
        ValueError
            If ``value`` has the wrong shape or is not finite.

        Examples
        --------
        >>> import numpy as np
        >>> layout = StateLayout((Coordinate("azimuth_rad", "rad", period=2 * np.pi),))
        >>> bool(np.isclose(layout.normalise([1.5 * np.pi])[0], -0.5 * np.pi))
        True
        """
        return self.wrap(value)

    def residual(self, a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
        """Return ``a - b``, taking the shorter way round for periodic coordinates.

        For an azimuth in degrees, 359° - 1° is -2°, not 358°.

        Parameters
        ----------
        a, b : array_like
            Vectors in this layout's order and units. Each is shape ``(n_state,)``
            or ``(n_points, n_state)``. If one is a single vector and the other a
            stack, the single vector is used against every row of the stack.

        Returns
        -------
        numpy.ndarray
            Differences, shape ``(n_state,)``, or ``(n_points, n_state)`` if either
            input is a stack. Periodic entries lie in [-period/2, period/2).

        Raises
        ------
        ValueError
            If either input has the wrong shape or is not finite.

        Examples
        --------
        >>> import numpy as np
        >>> layout = StateLayout((Coordinate("azimuth_deg", "deg", period=360.0),))
        >>> layout.residual([359.0], [1.0])
        array([-2.])
        """
        first = np.asarray(a, dtype=np.float64)
        second = np.asarray(b, dtype=np.float64)
        check_points(first, self.dimension, "a")
        check_points(second, self.dimension, "b")
        return self.wrap(first - second)

    def weighted_mean(self, points: ArrayLike, weights: ArrayLike) -> NDArray[np.float64]:
        """Return the weighted mean of many vectors, averaging periodic values on a circle.

        Ordinary coordinates use the usual weighted sum. A periodic coordinate is
        first turned into an angle on a circle, phase = 2π · value / period. The
        mean is then the direction of the weighted sum of the unit vectors
        (cos phase, sin phase), turned back into the coordinate's unit [1]_.
        This is why +179° and -179° average to 180°, not to 0°.

        Parameters
        ----------
        points : array_like
            The vectors, shape ``(n_points, n_state)``.
        weights : array_like
            One weight per vector, shape ``(n_points,)``. They must sum to one.
            They may be negative, as the centre weight of a UKF, an unscented
            Kalman filter, can be.

        Returns
        -------
        numpy.ndarray
            The mean, shape ``(n_state,)``. Periodic entries lie in
            (-period/2, period/2].

        Raises
        ------
        ValueError
            If ``points`` is not finite with shape ``(n_points, n_state)``,
            ``weights`` has the wrong shape or does not sum to one, or a periodic
            coordinate has no mean direction (see Notes).

        Notes
        -----
        If the weighted unit vectors cancel, for example equal weight at 0° and
        180°, there is no mean direction, and a ValueError is raised. This mean
        assumes the values gather around one direction. It cannot describe a
        density with two or more separate peaks on the circle.

        References
        ----------
        .. [1] Mardia and Jupp (2000), §2.2.1; see the module References.

        Examples
        --------
        >>> import numpy as np
        >>> layout = StateLayout((Coordinate("azimuth_deg", "deg", period=360.0),))
        >>> mean = layout.weighted_mean([[179.0], [-179.0]], [0.5, 0.5])
        >>> bool(np.isclose(abs(mean[0]), 180.0))
        True
        """
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.dimension or not np.all(np.isfinite(array)):
            msg = (
                f"points must be finite with shape (n_points, {self.dimension}); "
                f"got shape {array.shape}."
            )
            raise ValueError(msg)
        weight = as_vector(weights, len(array), "weights")
        if not np.isclose(weight.sum(), 1.0, rtol=_WEIGHT_SUM_RTOL, atol=_WEIGHT_SUM_ATOL):
            msg = f"mean weights must sum to one; got {weight.sum()!r}."
            raise ValueError(msg)

        periodic = np.array([c.period is not None for c in self.coordinates])
        period = np.array([c.period for c in self.coordinates if c.period is not None])
        phase_rad = array[:, periodic] * (2 * np.pi / period)
        sine = weight @ np.sin(phase_rad)
        cosine = weight @ np.cos(phase_rad)
        if np.any(np.hypot(sine, cosine) < _MIN_RESULTANT_LENGTH):
            msg = "periodic coordinate mean is undefined: the weighted points cancel on the circle."
            raise ValueError(msg)

        result = weight @ array
        result[periodic] = np.arctan2(sine, cosine) * period / (2 * np.pi)
        return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True)
class StateEstimate:
    """A frozen snapshot of a filter's state: mean, covariance, time and layout.

    The constructor copies the arrays, wraps periodic coordinates of the mean to
    [-period/2, period/2), and makes both arrays read-only. So the snapshot cannot
    change after it is made, even if the caller changes its own arrays.

    Parameters
    ----------
    mean : numpy.ndarray
        The state, shape ``(n_state,)``, in the layout's order and units.
    covariance : numpy.ndarray
        Its covariance, shape ``(n_state, n_state)``. It must be symmetric and
        positive semidefinite: no combination of the coordinates may have a
        negative variance. Each entry has the product of its two coordinates'
        units.
    timestamp_s : float
        Time of the estimate, in seconds.
    state_layout : StateLayout
        Names, units and frame of the coordinates.

    Raises
    ------
    ValueError
        If ``timestamp_s`` is not finite, ``mean`` is not finite with shape
        ``(n_state,)``, or ``covariance`` is not finite, square, symmetric and
        positive semidefinite.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.2.1; see the module References.

    Examples
    --------
    >>> import numpy as np
    >>> layout = StateLayout((Coordinate("x_m", "m"),))
    >>> estimate = StateEstimate(np.array([3.0]), np.eye(1), 0.0, layout)
    >>> estimate.mean.flags.writeable
    False
    """

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    timestamp_s: float
    state_layout: StateLayout

    def __post_init__(self) -> None:
        """Check, copy, wrap and freeze the mean and covariance."""
        timestamp_s = float(self.timestamp_s)
        if not np.isfinite(timestamp_s):
            msg = f"timestamp_s must be finite; got {timestamp_s!r}."
            raise ValueError(msg)
        dimension = self.state_layout.dimension
        covariance = as_covariance(self.covariance, dimension)
        mean = self.state_layout.normalise(as_vector(self.mean, dimension, "mean"))

        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "timestamp_s", timestamp_s)
