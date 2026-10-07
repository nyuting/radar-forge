"""Sensor measurements, how they reach the tracker, and models that predict them.

A measurement is what a sensor reports: some numbers, such as a position or a
range, with their covariance (how uncertain each number is, and how the errors
vary together). A measurement model is the rule that says what a sensor *should*
report if the target's state were known. The tracker compares the two.

The path of a measurement into the tracker::

    raw sensor output
            │
            ▼
    SensorRoute.batch()      checks the model names, copies the arrays
            │
            ▼
    MeasurementBatch         all measurements from one scan of one sensor
            │
            ▼
    Tracker.process()

The classes:

- :class:`Measurement` is one detection: its value, covariance, time, the sensor
  that made it, and the name of the measurement model that explains it.
- :class:`MeasurementBatch` is every measurement from one scan of one sensor,
  possibly none.
- :class:`SensorRoute` names a sensor, and the models its measurements may use.
- :class:`MeasurementModel` is the interface every measurement model follows.
- :class:`CartesianPosition` reports some of the state's coordinates directly.
- :class:`SensorPose` is where a sensor stands.
- :class:`BistaticRangeDopplerModel` predicts the path length and path rate seen
  by a transmitter and receiver at two places.

States use the x = east, y = north, z = up convention of
:mod:`radar_forge.core.tracking.coordinates`.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications
       to Tracking and Navigation*, Wiley, 2001, §4.3.1 (the
       discrete-time state-space model) and §10.2 (estimation in nonlinear
       stochastic systems). In z = Hx + w, w is the measurement noise: zero
       mean, with covariance R.
.. [2] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005, ch. 3
       (coordinate systems and geometry: the range sum and the baseline).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from radar_forge.core.geodesy import geodetic_to_enu_m
from radar_forge.core.tracking._validation import (
    as_covariance,
    as_vector,
    check_frame_and_origin,
    check_points,
)
from radar_forge.core.tracking.coordinates import Coordinate, StateLayout

if TYPE_CHECKING:
    from radar_forge.core.radar import BistaticRadar, Radar
    from radar_forge.core.tracking.tracks import TrackSnapshot

__all__ = [
    "BistaticRangeDopplerModel",
    "CartesianPosition",
    "Measurement",
    "MeasurementBatch",
    "MeasurementModel",
    "SensorPose",
    "SensorRoute",
]


@dataclass(frozen=True)
class Measurement:
    """One detection: value, covariance, time, sensor and measurement model.

    The constructor copies the arrays and makes the copies read-only, so a stored
    measurement cannot change after it is made.

    Parameters
    ----------
    value : numpy.ndarray
        The measured numbers, shape ``(n_meas,)``, in the units of the
        measurement model's ``measurement_layout``.
    covariance : numpy.ndarray
        Their covariance, R, shape ``(n_meas, n_meas)``. It must be symmetric and
        positive semidefinite.
    timestamp_s : float
        Time of the measurement, in seconds, on the same clock as every other
        measurement and track.
    sensor_id : str
        The sensor that made the measurement: the ``sensor_id`` of its
        :class:`SensorRoute`.
    measurement_model_id : str
        Name of the measurement model that explains it. The tracker uses this
        name, not the sensor, to choose the equations.

    Raises
    ------
    ValueError
        If ``sensor_id`` or ``measurement_model_id`` is empty, ``timestamp_s`` is
        not finite, or ``value`` is not a finite nonempty vector. Also if
        ``covariance`` is not finite, square, symmetric and positive
        semidefinite, with the same size as ``value``.

    Examples
    --------
    >>> import numpy as np
    >>> z = Measurement(np.array([1.0, 2.0]), np.eye(2), 0.0, "radar", "position")
    >>> z.value.flags.writeable
    False
    """

    value: NDArray[np.float64]
    covariance: NDArray[np.float64]
    timestamp_s: float
    sensor_id: str
    measurement_model_id: str

    def __post_init__(self) -> None:
        """Check the fields, and store frozen copies of the arrays."""
        if not self.sensor_id or not self.measurement_model_id:
            msg = (
                "sensor_id and measurement_model_id are required; got "
                f"{self.sensor_id!r} and {self.measurement_model_id!r}."
            )
            raise ValueError(msg)
        timestamp_s = float(self.timestamp_s)
        if not np.isfinite(timestamp_s):
            msg = f"timestamp_s must be finite; got {timestamp_s!r}."
            raise ValueError(msg)
        raw = np.asarray(self.value)
        if raw.ndim != 1 or not len(raw):
            msg = f"measurement value must be a nonempty vector; got shape {raw.shape}."
            raise ValueError(msg)
        value = as_vector(raw, len(raw), "measurement value")
        covariance = as_covariance(self.covariance, len(raw))

        value.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "timestamp_s", timestamp_s)


def _check_same_scan(
    measurements: tuple[Measurement, ...], sensor_id: str, timestamp_s: float
) -> None:
    """Raise unless every measurement carries the batch's sensor and time."""
    strangers = [
        (m.sensor_id, m.timestamp_s)
        for m in measurements
        if m.timestamp_s != timestamp_s or m.sensor_id != sensor_id
    ]
    if strangers:
        msg = (
            "all observations require the batch sensor and timestamp "
            f"({sensor_id!r}, {timestamp_s} s); got (sensor, time) {strangers}."
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class MeasurementBatch:
    """Every measurement from one scan of one sensor, possibly none.

    An empty batch still matters: it says the sensor looked and saw nothing, so
    the tracker counts a miss for each track the sensor covers.

    Parameters
    ----------
    timestamp_s : float
        Time of the scan, in seconds.
    sensor_id : str
        The sensor that made the scan.
    measurements : tuple of Measurement, optional
        The scan's measurements. Each must carry this ``sensor_id`` and exactly
        this ``timestamp_s``. Default empty.

    Raises
    ------
    ValueError
        If ``timestamp_s`` is not finite, ``sensor_id`` is empty, or a
        measurement's sensor or timestamp differs from the batch's.

    Notes
    -----
    **Limitation: one timestamp per batch.** The tracker predicts every track to
    the batch time once, then compares all the measurements against that one
    prediction. So every measurement must carry exactly (bit for bit) the batch
    timestamp. A scanning radar stamps each detection at the moment its beam
    passed the target, so its detections have different times. To use them,
    either give each time its own batch, or stamp the whole scan with one time
    and accept the error that adds.

    **One sensor per batch.** The :class:`SensorRoute` of the batch's
    ``sensor_id`` decides which models may explain its measurements and which
    tracks the sensor can see.
    Measurements from two sensors in one batch would need two such decisions,
    so association would be ambiguous.
    """

    timestamp_s: float
    sensor_id: str
    measurements: tuple[Measurement, ...] = ()

    def __post_init__(self) -> None:
        """Check that every measurement belongs to this scan."""
        timestamp_s = float(self.timestamp_s)
        measurements = tuple(self.measurements)
        if not np.isfinite(timestamp_s):
            msg = f"timestamp_s must be finite; got {timestamp_s!r}."
            raise ValueError(msg)
        if not self.sensor_id:
            msg = "batch sensor_id must be nonempty."
            raise ValueError(msg)
        _check_same_scan(measurements, self.sensor_id, timestamp_s)

        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "measurements", measurements)


@dataclass(frozen=True)
class SensorRoute:
    """A sensor's name, the measurement models it may use, and what it can see.

    This describes one sensor to the tracker. It does two jobs, and only these
    two. It routes each of the sensor's measurements to a measurement model, by
    name: hence the class name. And it says which tracks the sensor covers on a
    scan. It does not make detections.

    Parameters
    ----------
    sensor_id : str
        Name of the sensor. Its measurements and batches carry it as their
        ``sensor_id``.
    measurement_model_ids : tuple of str
        Names of the measurement models this sensor's measurements may use. They
        must be nonempty and unique. One sensor may report several kinds of
        measurement, so each measurement names its own model. The list is there
        to stop a measurement reaching the wrong model. For example, an angle
        measurement from this sensor cannot be sent to a range and Doppler
        model, unless this list names that model.
    observable : callable or None, optional
        Coverage test: given a :class:`~radar_forge.core.tracking.tracks.TrackSnapshot`,
        return True if the sensor can see that track on this scan. The tracker
        only compares this sensor's measurements against tracks it can see. Use
        it for a sensor with a limited field of view. None (the default) means
        the sensor sees every track.

    Raises
    ------
    ValueError
        If ``sensor_id`` is empty, or ``measurement_model_ids`` is empty, holds an
        empty name or repeats one.

    Notes
    -----
    Stone Soup has a ``Sensor`` class too, but its ``measure()`` method
    generates detections from the truth. This class only describes a sensor
    whose detections come from elsewhere.

    Examples
    --------
    >>> import numpy as np
    >>> radar = SensorRoute("radar", ("position",))
    >>> batch = radar.batch(1.0, [("position", np.array([5.0]), np.eye(1))])
    >>> batch.measurements[0].sensor_id
    'radar'
    """

    sensor_id: str
    measurement_model_ids: tuple[str, ...]
    observable: Callable[[TrackSnapshot], bool] | None = None

    def __post_init__(self) -> None:
        """Check the sensor name and the model names."""
        model_ids = tuple(self.measurement_model_ids)
        if (
            not self.sensor_id
            or not model_ids
            or not all(model_ids)
            or len(set(model_ids)) != len(model_ids)
        ):
            msg = (
                "a sensor ID and unique nonempty model IDs are required; got "
                f"{self.sensor_id!r} with models {model_ids}."
            )
            raise ValueError(msg)

        object.__setattr__(self, "measurement_model_ids", model_ids)

    def batch(
        self,
        timestamp_s: float,
        observations: Iterable[tuple[str, NDArray[np.float64], NDArray[np.float64]]],
    ) -> MeasurementBatch:
        """Package one scan's observations as a :class:`MeasurementBatch`.

        Parameters
        ----------
        timestamp_s : float
            Time of the scan, in seconds. Every measurement gets this time.
        observations : iterable of tuple
            One ``(model_id, value, covariance)`` per detection: the name of its
            measurement model, its value, shape ``(n_meas,)``, and its covariance,
            shape ``(n_meas, n_meas)``. May be empty.

        Returns
        -------
        MeasurementBatch
            The scan, with this sensor's ``sensor_id``. The arrays are copied.

        Raises
        ------
        ValueError
            If a ``model_id`` is not registered for this sensor, or as for
            :class:`Measurement` and :class:`MeasurementBatch`.
        """
        observations = list(observations)
        unknown = [o[0] for o in observations if o[0] not in self.measurement_model_ids]
        if unknown:
            msg = f"models {unknown} are not registered for sensor {self.sensor_id!r}."
            raise ValueError(msg)

        # Each detection becomes its own Measurement object, which checks itself,
        # so this is a loop over Python objects, not over numbers.
        measurements = tuple(
            Measurement(value, covariance, timestamp_s, self.sensor_id, model_id)
            for model_id, value, covariance in observations
        )
        return MeasurementBatch(timestamp_s, self.sensor_id, measurements)


class MeasurementModel(Protocol):
    """The interface of a measurement model: state in, expected measurement out.

    A measurement model maps a state x to the measurement h(x) a sensor would
    report if there were no noise. It knows nothing about the filter that calls it.

    Attributes
    ----------
    state_layout : StateLayout
        Layout of the states it accepts.
    measurement_layout : StateLayout
        Layout of the measurements it returns, including any periods.
    """

    state_layout: StateLayout
    measurement_layout: StateLayout

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the expected measurement for one state or for a stack of states.

        A model must accept both shapes, so that a filter can evaluate all its
        sigma points in one call.

        Parameters
        ----------
        state : numpy.ndarray
            One state, shape ``(n_state,)``, or a stack, shape
            ``(n_points, n_state)``, in ``state_layout`` order and units.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_meas,)`` for one state, or ``(n_points, n_meas)`` for a
            stack, in ``measurement_layout`` order and units.
        """
        ...


class CartesianPosition:
    """A sensor that reports some of the state's coordinates directly.

    This is the linear measurement z = Hx + w [1]_, where each row of H picks
    one coordinate. Despite the name, any coordinates can be chosen, velocities
    included.

    Parameters
    ----------
    state_layout : StateLayout
        Layout of the state.
    names : tuple of str
        The coordinates the sensor reports, in measurement order.
    periods : mapping of str to float, optional
        A period, in each coordinate's unit, for any reported coordinate whose
        measurement wraps. The predicted value is not wrapped; the wrap is used
        when the filter subtracts the prediction from the measurement, so the
        shorter way round is chosen. Coordinates not named here keep their period
        from ``state_layout``.

    Raises
    ------
    ValueError
        If ``names`` is empty, repeats a name or names a coordinate not in
        ``state_layout``, or ``periods`` names a coordinate not in ``names``.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §4.3.1; see the module References.

    Examples
    --------
    >>> import numpy as np
    >>> layout = StateLayout((Coordinate("x_m", "m"), Coordinate("xdot_mps", "m/s")))
    >>> model = CartesianPosition(layout, ("x_m",))
    >>> model.predict(np.array([10.0, 2.0]))
    array([10.])
    >>> model.predict(np.array([[10.0, 2.0], [20.0, 3.0]]))
    array([[10.],
           [20.]])
    """

    def __init__(
        self,
        state_layout: StateLayout,
        names: tuple[str, ...],
        periods: Mapping[str, float] | None = None,
    ) -> None:
        periods = dict(periods or {})
        if not names or len(set(names)) != len(names):
            msg = f"observed coordinate names must be nonempty and unique; got {names}."
            raise ValueError(msg)
        if set(periods) - set(names):
            msg = f"periods must refer to observed coordinates; got {sorted(periods)}."
            raise ValueError(msg)

        self.state_layout = state_layout
        self.indices = state_layout.indices(names)
        self.measurement_layout = StateLayout(
            tuple(
                Coordinate(name=c.name, unit=c.unit, period=periods.get(c.name, c.period))
                for c in (state_layout.coordinates[i] for i in self.indices)
            ),
            state_layout.frame,
            state_layout.origin_lla_deg_m,
        )

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the reported coordinates of one state or of a stack of states.

        Parameters
        ----------
        state : numpy.ndarray
            One state, shape ``(n_state,)``, or a stack, shape ``(n_points, n_state)``.

        Returns
        -------
        numpy.ndarray
            A new array, shape ``(n_meas,)`` or ``(n_points, n_meas)``, in
            measurement order.

        Raises
        ------
        ValueError
            If ``state`` has the wrong shape or is not finite.
        """
        states = np.asarray(state, dtype=np.float64)
        check_points(states, self.state_layout.dimension, "state")
        return states[..., list(self.indices)]


@dataclass(frozen=True)
class SensorPose:
    """Where a sensor stands: a fixed east-north-up position, its frame and origin.

    Parameters
    ----------
    position_m : numpy.ndarray
        East, north and up position, in metres, shape ``(3,)``. It is copied and
        made read-only.
    frame : str
        Name of the frame. It must match the ``frame`` of the tracking state.
    origin_lla_deg_m : tuple of float
        Origin of the east-north-up frame: latitude and longitude in degrees,
        altitude in metres. It must match the tracking state's origin.

    Raises
    ------
    ValueError
        If ``position_m`` is not three finite numbers, ``frame`` is empty, or
        ``origin_lla_deg_m`` is not a valid origin (see :class:`StateLayout`).

    Examples
    --------
    >>> import numpy as np
    >>> pose = SensorPose(np.array([100.0, 0.0, 10.0]), "ENU", (36.0, -78.9, 60.0))
    >>> pose.position_m
    array([100.,   0.,  10.])
    """

    position_m: NDArray[np.float64]
    frame: str
    origin_lla_deg_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Check the position and origin, and store a frozen copy of the position."""
        position = as_vector(self.position_m, 3, "position_m")
        # The same check StateLayout uses, so a sensor site and a state accept the
        # same frames and origins.
        origin = check_frame_and_origin(self.frame, self.origin_lla_deg_m)

        position.setflags(write=False)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "origin_lla_deg_m", origin)

    @classmethod
    def from_radar(
        cls,
        radar: Radar,
        origin_lla_deg_m: tuple[float, float, float],
        frame: str = "ENU",
    ) -> SensorPose:
        """Place a monostatic radar's site in the tracking frame.

        The site comes from the :class:`~radar_forge.core.radar.Radar` itself, so the
        radar's position is stated in one place only.

        Parameters
        ----------
        radar : radar_forge.core.radar.Radar
            The radar whose geodetic site is converted.
        origin_lla_deg_m : tuple of float
            ENU origin of the tracking state: latitude and longitude in degrees,
            altitude in metres.
        frame : str, optional
            State frame identifier; it must match the tracking layout's ``frame``.

        Returns
        -------
        SensorPose
            The site as east, north, up metres about ``origin_lla_deg_m``, shape ``(3,)``.

        See Also
        --------
        radar_forge.core.geodesy.geodetic_to_enu_m : The conversion used.
        """
        position_m = geodetic_to_enu_m(
            radar.latitude_deg, radar.longitude_deg, radar.altitude_m, *origin_lla_deg_m
        )
        return cls(position_m, frame, origin_lla_deg_m)

    @classmethod
    def from_bistatic_radar(
        cls,
        pair: BistaticRadar,
        origin_lla_deg_m: tuple[float, float, float],
        frame: str = "ENU",
    ) -> tuple[SensorPose, SensorPose]:
        """Place a bistatic pair's two sites in the tracking frame.

        Parameters
        ----------
        pair : radar_forge.core.radar.BistaticRadar
            The transmitter/receiver pair whose geodetic sites are converted.
        origin_lla_deg_m : tuple of float
            ENU origin of the tracking state: latitude and longitude in degrees,
            altitude in metres.
        frame : str, optional
            State frame identifier; it must match the tracking layout's ``frame``.

        Returns
        -------
        transmitter : SensorPose
            Transmitter site as east, north, up metres, shape ``(3,)``.
        receiver : SensorPose
            Receiver site as east, north, up metres, shape ``(3,)``.

        See Also
        --------
        radar_forge.core.geodesy.geodetic_to_enu_m : The conversion used.
        """
        transmitter_m = geodetic_to_enu_m(
            pair.transmitter_latitude_deg,
            pair.transmitter_longitude_deg,
            pair.transmitter_altitude_m,
            *origin_lla_deg_m,
        )
        receiver_m = geodetic_to_enu_m(
            pair.receiver_latitude_deg,
            pair.receiver_longitude_deg,
            pair.receiver_altitude_m,
            *origin_lla_deg_m,
        )
        return (
            cls(transmitter_m, frame, origin_lla_deg_m),
            cls(receiver_m, frame, origin_lla_deg_m),
        )


# The Cartesian coordinates a sensor geometry needs, keyed by its include_velocity flag: the
# positions always, and the velocities only when asked for.
_REQUIRED_NAMES = {
    False: ("x_m", "y_m", "z_m"),
    True: ("x_m", "y_m", "z_m", "xdot_mps", "ydot_mps", "zdot_mps"),
}


class _Geometry:
    """Read a target's position (and velocity) from a state, for one sensor site.

    A state need not hold every Cartesian coordinate. A 2-D tracker, for example,
    has no ``z_m``. Each missing coordinate must then be given a fixed value, so
    that the geometry never guesses one.

    Parameters
    ----------
    layout : StateLayout
        Layout of the state. It must share the pose's frame and origin.
    pose : SensorPose
        The sensor site.
    fixed : mapping of str to float or None
        A finite value for each required coordinate that the state lacks. None
        gives none.
    include_velocity : bool
        Keyword-only. If True, the velocities ``xdot_mps``, ``ydot_mps`` and
        ``zdot_mps`` are required as well as the positions ``x_m``, ``y_m`` and
        ``z_m``.

    Raises
    ------
    ValueError
        If the frame or origin differ, ``fixed`` names a coordinate that is not
        required or is already in the state, a required coordinate is missing
        from both, or a fixed value is not finite.
    """

    def __init__(
        self,
        layout: StateLayout,
        pose: SensorPose,
        fixed: Mapping[str, float] | None,
        *,
        include_velocity: bool,
    ) -> None:
        fixed_values = dict(fixed or {})
        if layout.frame != pose.frame or layout.origin_lla_deg_m != pose.origin_lla_deg_m:
            msg = (
                "sensor and state must share ENU frame and origin; got state "
                f"{layout.frame!r} {layout.origin_lla_deg_m} and sensor "
                f"{pose.frame!r} {pose.origin_lla_deg_m}."
            )
            raise ValueError(msg)
        required = _REQUIRED_NAMES[include_velocity]
        # A fixed value may stand in only for a required coordinate that the state lacks.
        if not set(fixed_values) <= set(required) - set(layout.names):
            msg = (
                "fixed coordinates must be required, absent state coordinates; got "
                f"{sorted(fixed_values)}."
            )
            raise ValueError(msg)
        missing = [n for n in required if n not in layout.names and n not in fixed_values]
        if missing:
            msg = f"geometry requires missing coordinates to be explicitly fixed; got {missing}."
            raise ValueError(msg)
        if not np.all(np.isfinite(list(fixed_values.values()))):
            msg = f"fixed coordinates must be finite; got {fixed_values}."
            raise ValueError(msg)

        self.layout = layout
        self.pose = pose
        self.fixed = fixed_values

    def values(self, state: NDArray[np.float64], suffix: str) -> NDArray[np.float64]:
        """Return the east, north and up parts of a position or velocity.

        Parameters
        ----------
        state : numpy.ndarray
            One state, shape ``(n_state,)``, or a stack, shape ``(n_points, n_state)``.
            The caller has already checked its shape.
        suffix : str
            ``"_m"`` for the position in metres, or ``"dot_mps"`` for the
            velocity in metres per second.

        Returns
        -------
        numpy.ndarray
            Shape ``(3,)`` or ``(n_points, 3)``. A coordinate the state lacks
            takes its fixed value.
        """
        names = [f"{a}{suffix}" for a in "xyz"]
        is_fixed = np.array([n not in self.layout.names for n in names])
        fixed_value = np.array([self.fixed.get(n, 0.0) for n in names])
        # A fixed coordinate has no column in the state; index column 0 in its place
        # and let np.where replace that value with the fixed one.
        column = [self.layout.names.index(n) if n in self.layout.names else 0 for n in names]
        return np.where(is_fixed, fixed_value, np.asarray(state, dtype=np.float64)[..., column])


class BistaticRangeDopplerModel:
    """Predict the path length and path rate seen by a separated transmitter and receiver.

    In a bistatic radar the transmitter and receiver stand at different places.
    The signal travels from the transmitter to the target, then on to the
    receiver. This model predicts two numbers from the target's state:

    - ``path_range_m``, the length of that path, R_tx + R_rx, in metres. Here
      R_tx is the distance from the transmitter to the target, and R_rx from the
      target to the receiver. With ``excess_range``, the baseline L (the direct
      distance from transmitter to receiver) is subtracted, so a target on the
      baseline has a path range of zero [1]_.
    - ``path_range_rate_mps``, how fast the path is getting shorter, in m/s.
      It is positive when the path is shrinking (closing).

    Parameters
    ----------
    state_layout : StateLayout
        Layout of the Cartesian state.
    transmitter, receiver : SensorPose
        The two sites. They must share the state's frame and origin.
    excess_range : bool, optional
        If True (the default), subtract the baseline from the path length.
    fixed_coordinates : mapping of str to float or None, optional
        A value, in SI units, for each of ``x_m``, ``y_m``, ``z_m``,
        ``xdot_mps``, ``ydot_mps`` and ``zdot_mps`` that the state does not hold.

    Raises
    ------
    ValueError
        If a site's frame or origin differs from the state's. Also if a fixed
        coordinate is not needed or is already in the state, a needed coordinate
        is missing from both, or a fixed value is not finite.

    Notes
    -----
    Let p be the target position and v its velocity. The unit vector from the
    transmitter to the target is u_tx = (p - p_tx) / R_tx, and u_rx likewise from
    the receiver. Moving the target by v changes R_tx at the rate u_tx · v, and
    R_rx at u_rx · v. So the path rate, positive closing, is

        path_range_rate_mps = -(u_tx + u_rx) · v.

    This is not the range rate of a monostatic radar, and it is not a Doppler
    shift in Hz. The Doppler shift is path_range_rate_mps divided by the
    wavelength; converting to Hz is the job of the code that reads the radar.
    Both unit vectors are undefined when the target sits exactly on a site, so
    :meth:`predict` rejects that state.

    Use :meth:`SensorPose.from_bistatic_radar` to place the two sites from a
    :class:`~radar_forge.core.radar.BistaticRadar`.

    No initiator here can start a track from this model (see
    :class:`~radar_forge.core.tracking.initiation.DirectStateInitiator`). A
    tracker whose only model is this one must be given its tracks with
    :meth:`~radar_forge.core.tracking.tracker.Tracker.seed`.

    See Also
    --------
    radar_forge.core.radar.BistaticRadar.target_ranges_m : The two ranges, from
        geodetic positions. Not used here: the tracking state is in local ENU
        metres, and converting every sigma point back to latitude, longitude and
        altitude would only add work and roundoff.
    radar_forge.core.signal.bistatic_doppler_hz : The same path rate, divided by
        the wavelength, in Hz. Not used here: the measurement is the path rate in
        m/s, which needs no wavelength.

    References
    ----------
    .. [1] Willis (2005), ch. 3; see the module References.

    Examples
    --------
    A target 12 m up, and 13 m from both sites. The sites share a place, so the
    baseline is zero and the path range is 13 + 13 m:

    >>> import numpy as np
    >>> names = ("x_m", "xdot_mps", "y_m", "ydot_mps", "z_m", "zdot_mps")
    >>> units = ("m", "m/s") * 3
    >>> origin = (36.0, -78.9, 60.0)
    >>> layout = StateLayout(
    ...     tuple(Coordinate(n, u) for n, u in zip(names, units)), "ENU", origin
    ... )
    >>> site = SensorPose(np.zeros(3), "ENU", origin)
    >>> model = BistaticRangeDopplerModel(layout, site, site)
    >>> model.predict(np.array([3.0, -3.0, 4.0, -4.0, 12.0, 0.0]))[0]
    np.float64(26.0)
    """

    def __init__(
        self,
        state_layout: StateLayout,
        transmitter: SensorPose,
        receiver: SensorPose,
        excess_range: bool = True,
        fixed_coordinates: Mapping[str, float] | None = None,
    ) -> None:
        self.state_layout = state_layout
        self._tx = _Geometry(state_layout, transmitter, fixed_coordinates, include_velocity=True)
        self._rx = _Geometry(state_layout, receiver, fixed_coordinates, include_velocity=True)
        self.excess_range = excess_range
        self.measurement_layout = StateLayout(
            (Coordinate("path_range_m", "m"), Coordinate("path_range_rate_mps", "m/s")),
            "bistatic",
        )

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the path length and path rate for one state or a stack of states.

        Parameters
        ----------
        state : numpy.ndarray
            One state, shape ``(n_state,)``, or a stack, shape ``(n_points, n_state)``.

        Returns
        -------
        numpy.ndarray
            ``(path_range_m, path_range_rate_mps)``, shape ``(2,)`` for one state
            or ``(n_points, 2)`` for a stack.

        Raises
        ------
        ValueError
            If ``state`` has the wrong shape or is not finite. Also if any target
            position is exactly on the transmitter or the receiver. There the
            direction to the target is undefined.
        """
        states = np.asarray(state, dtype=np.float64)
        check_points(states, self.state_layout.dimension, "state")
        position_m = self._tx.values(states, "_m")
        # A distance is exactly zero only when the two points are equal, so the
        # test compares positions instead of computing norms just for the guard.
        if np.any(
            np.all(position_m == self._tx.pose.position_m, axis=-1)
            | np.all(position_m == self._rx.pose.position_m, axis=-1)
        ):
            msg = (
                "bistatic geometry is singular at an asset: a target position is exactly "
                "on the transmitter or the receiver."
            )
            raise ValueError(msg)

        velocity_mps = self._tx.values(states, "dot_mps")
        to_target_tx_m = position_m - self._tx.pose.position_m
        to_target_rx_m = position_m - self._rx.pose.position_m
        range_tx_m = np.linalg.norm(to_target_tx_m, axis=-1)
        range_rx_m = np.linalg.norm(to_target_rx_m, axis=-1)
        baseline_m = float(np.linalg.norm(self._tx.pose.position_m - self._rx.pose.position_m))

        path_range_m = range_tx_m + range_rx_m - (baseline_m if self.excess_range else 0.0)
        unit_sum = to_target_tx_m / range_tx_m[..., None] + to_target_rx_m / range_rx_m[..., None]
        path_range_rate_mps = -np.sum(unit_sum * velocity_mps, axis=-1)
        return np.stack([path_range_m, path_range_rate_mps], axis=-1)
