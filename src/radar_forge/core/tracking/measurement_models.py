"""Sensor measurements, how they reach the tracker, and models that predict them.

A measurement is what a sensor reports: some numbers, such as a position or a
range, with their covariance (how uncertain each number is, and how the errors
vary together). A measurement model is the rule that says what a sensor *should*
report if the target's state were known. The tracker compares the two.

The path of a measurement into the tracker::

    raw sensor output
            │
            ▼
    SensorRegistration.batch()  checks the model names, copies the arrays
            │
            ▼
    MeasurementBatch            all measurements from one scan of one sensor
            │
            ▼
    Tracker.process()

The classes:

- :class:`Measurement` is one detection: its value, covariance, time, the sensor
  that made it, and the name of the measurement model that explains it.
- :class:`MeasurementBatch` is every measurement from one scan of one sensor,
  possibly none.
- :class:`SensorRegistration` names a sensor, and the models its measurements may use.
- :class:`MeasurementModel` is the interface every measurement model follows.
- :class:`CartesianPosition` reports some of the state's coordinates directly.
- :class:`MarginalModel`, built by :func:`marginal`, keeps some of another
  model's components, so that a track can use part of a measurement.
- :class:`SensorPose` is where a sensor stands.
- :class:`BistaticRangeDopplerModel` predicts the path length and path rate seen
  by a transmitter and receiver at two places.
- :class:`RangeAzimuthElevationRangeRate` predicts the range, azimuth, elevation
  and range rate a monostatic radar reports.

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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from radar_forge.core.geodesy import enu_to_range_azimuth_elevation_rad, geodetic_to_enu_m
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
    "DifferentiableMarginalModel",
    "DifferentiableMeasurementModel",
    "MarginalModel",
    "Measurement",
    "MeasurementBatch",
    "MeasurementModel",
    "RangeAzimuthElevationRangeRate",
    "SensorPose",
    "SensorRegistration",
    "marginal",
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
        :class:`SensorRegistration`.
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

    **One sensor per batch.** The :class:`SensorRegistration` of the batch's
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
class SensorRegistration:
    """A sensor's name, the measurement models it may use, and what it can see.

    This registers one sensor with the tracker. It does three jobs. It routes
    each of the sensor's measurements to a measurement model, by name. It says
    which tracks the sensor covers on a scan. And it builds the sensor's
    batches, through :meth:`batch`. It does not make detections.

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
    covers : callable or None, optional
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
    >>> radar = SensorRegistration("radar", ("position",))
    >>> batch = radar.batch(1.0, [("position", np.array([5.0]), np.eye(1))])
    >>> batch.measurements[0].sensor_id
    'radar'
    """

    sensor_id: str
    measurement_model_ids: tuple[str, ...]
    covers: Callable[[TrackSnapshot], bool] | None = None

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


@runtime_checkable
class DifferentiableMeasurementModel(MeasurementModel, Protocol):
    """A measurement model that can give its Jacobian, the matrix H of the linearised model.

    The Kalman filter and the extended Kalman filter need H: for a linear model z = H x + w it
    is the model itself, and for a nonlinear one it is the Jacobian, the first derivatives
    of h at the state (:class:`~radar_forge.core.tracking.linearisation.TaylorLinearisation`).
    A model without :meth:`jacobian` still works with the unscented filter, which needs no
    derivatives. There is no numerical fallback: a finite-difference Jacobian would hide a
    wrong analytic one.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §10.3 (the extended Kalman filter).
    """

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the Jacobian of :meth:`predict` at one state.

        Parameters
        ----------
        state : numpy.ndarray
            One state, shape ``(n_state,)``, in ``state_layout`` order and units.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_meas, n_state)``. Row i, column j is the derivative of measured
            element i with respect to state element j, in the unit of i per unit of j.
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

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return H, the matrix that picks the reported coordinates out of a state.

        The model is linear, so H is the same at every state: row i has a one in the column
        of the i-th reported coordinate, and zeros elsewhere.

        Parameters
        ----------
        state : numpy.ndarray
            One state, shape ``(n_state,)``. Only its shape is checked.

        Returns
        -------
        numpy.ndarray
            H, a new array of shape ``(n_meas, n_state)``.

        Raises
        ------
        ValueError
            If ``state`` has the wrong shape or is not finite.

        Examples
        --------
        >>> import numpy as np
        >>> layout = StateLayout((Coordinate("x_m", "m"), Coordinate("xdot_mps", "m/s")))
        >>> CartesianPosition(layout, ("x_m",)).jacobian(np.array([10.0, 2.0]))
        array([[1., 0.]])
        """
        as_vector(state, self.state_layout.dimension, "state")
        return np.eye(self.state_layout.dimension, dtype=np.float64)[list(self.indices)]


class MarginalModel:
    """Some of another measurement model's components, chosen by name.

    The marginal of a Gaussian vector over some of its components is those
    components of the mean and that block of the covariance [1]_. So a
    marginal model predicts the chosen components of its ``base`` model's
    prediction and nothing else, and :meth:`marginalise` cuts a measurement of
    the base model down to the same components.

    It is how a track uses part of a measurement. A track that cannot yet trust
    a measured range rate updates on the range alone, through the marginal of
    a range and range-rate model over ``("range_m",)``
    (``spec/tracker-001.md`` §8.3). Build one with :func:`marginal`, which
    keeps the Jacobian when the base model has one.

    Parameters
    ----------
    base : MeasurementModel
        The full model.
    names : tuple of str
        The components to keep, in the order wanted. Each must be a component
        of ``base.measurement_layout``.

    Attributes
    ----------
    base : MeasurementModel
        As given.
    indices : tuple of int
        Where each kept component sits in the base model's measurement.
    state_layout : StateLayout
        The base model's.
    measurement_layout : StateLayout
        The kept components, with their units and periods, in the base model's
        frame.

    Raises
    ------
    ValueError
        If ``names`` is empty, repeats a name, or names a component the base
        model does not measure.

    References
    ----------
    .. [1] S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University
           Press, 2013, Appendix A.1, Lemma A.2 (the marginals of a joint
           Gaussian).

    Examples
    --------
    >>> import numpy as np
    >>> layout = StateLayout((Coordinate("x_m", "m"), Coordinate("xdot_mps", "m/s")))
    >>> both = CartesianPosition(layout, ("x_m", "xdot_mps"))
    >>> position = marginal(both, ("x_m",))
    >>> position.predict(np.array([10.0, 2.0]))
    array([10.])
    >>> z = Measurement(np.array([9.0, 1.5]), np.diag([4.0, 0.25]), 0.0, "radar", "both")
    >>> part = position.marginalise(z, "position")
    >>> part.value, part.covariance, part.measurement_model_id
    (array([9.]), array([[4.]]), 'position')
    """

    def __init__(self, base: MeasurementModel, names: tuple[str, ...]) -> None:
        if not names or len(set(names)) != len(names):
            msg = f"marginal component names must be nonempty and unique; got {names}."
            raise ValueError(msg)
        self.base = base
        self.indices = base.measurement_layout.indices(names)
        self.state_layout = base.state_layout
        full = base.measurement_layout
        self.measurement_layout = StateLayout(
            tuple(full.coordinates[i] for i in self.indices), full.frame, full.origin_lla_deg_m
        )

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the kept components of the base model's prediction.

        Parameters
        ----------
        state : numpy.ndarray
            One state, shape ``(n_state,)``, or a stack, shape ``(n_points, n_state)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_kept,)`` or ``(n_points, n_kept)``, in the order of ``names``.

        Raises
        ------
        ValueError
            As the base model's ``predict`` does.
        """
        predicted: NDArray[np.float64] = self.base.predict(state)[..., list(self.indices)]
        return predicted

    def marginalise(self, measurement: Measurement, model_id: str) -> Measurement:
        """Return the kept components of a measurement of the base model.

        Parameters
        ----------
        measurement : Measurement
            A measurement the base model explains: value ``(n_meas,)`` and
            covariance ``(n_meas, n_meas)``.
        model_id : str
            The ID this marginal model is registered under, which the new
            measurement names as its ``measurement_model_id``.

        Returns
        -------
        Measurement
            Value ``(n_kept,)``, covariance ``(n_kept, n_kept)``, and the same time
            and sensor.

        Raises
        ------
        ValueError
            If the measurement's length is not the base model's dimension.
        """
        if len(measurement.value) != self.base.measurement_layout.dimension:
            msg = (
                f"a measurement of the base model has {self.base.measurement_layout.dimension} "
                f"components; got {len(measurement.value)}."
            )
            raise ValueError(msg)
        rows = list(self.indices)
        return Measurement(
            measurement.value[rows],
            measurement.covariance[np.ix_(rows, rows)],
            measurement.timestamp_s,
            measurement.sensor_id,
            model_id,
        )


class DifferentiableMarginalModel(MarginalModel):
    """A :class:`MarginalModel` of a model that has a Jacobian, so it has one too.

    It exists so that a marginal is a
    :class:`DifferentiableMeasurementModel` exactly when its base model is one.
    :func:`marginal` picks the class.
    """

    base: DifferentiableMeasurementModel

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the kept rows of the base model's Jacobian.

        Parameters
        ----------
        state : numpy.ndarray
            One state, shape ``(n_state,)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_kept, n_state)``.

        Raises
        ------
        ValueError
            As the base model's ``jacobian`` does.
        """
        rows: NDArray[np.float64] = self.base.jacobian(state)[list(self.indices)]
        return rows


def marginal(model: MeasurementModel, names: tuple[str, ...]) -> MarginalModel:
    """Return the marginal of a measurement model over the named components.

    Parameters
    ----------
    model : MeasurementModel
        The full model.
    names : tuple of str
        The components to keep, in the order wanted.

    Returns
    -------
    MarginalModel
        A :class:`DifferentiableMarginalModel` if ``model`` has a Jacobian, so
        that the KF and the EKF can use it; a plain :class:`MarginalModel`
        otherwise, which the UKF can use.

    Raises
    ------
    ValueError
        As for :class:`MarginalModel`.
    """
    if isinstance(model, DifferentiableMeasurementModel):
        return DifferentiableMarginalModel(model, names)
    return MarginalModel(model, names)


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
    radar_forge.core.propagation.bistatic_doppler_hz : The same path rate, divided by
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
        self._check_off_sites(position_m)

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

    def _check_off_sites(self, position_m: NDArray[np.float64]) -> None:
        """Raise if any target position is exactly on the transmitter or the receiver."""
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

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Return the derivatives of the path range and path rate at one state.

        Parameters
        ----------
        state : numpy.ndarray
            One state, shape ``(n_state,)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(2, n_state)``: row 0 for ``path_range_m``, row 1 for
            ``path_range_rate_mps``. A coordinate the state lacks, held at a fixed value, has
            no column.

        Raises
        ------
        ValueError
            If ``state`` has the wrong shape or is not finite, or the target is exactly on
            a site, where the direction to it is undefined.

        Notes
        -----
        With :math:`u_k = (p - p_k) / R_k` the unit vector from site k to the target, and v
        the target's velocity (class Notes),

        .. math::

            \frac{\partial (R_{tx} + R_{rx})}{\partial p} = u_{tx} + u_{rx}, \qquad
            \frac{\partial \dot{R}}{\partial v} = -(u_{tx} + u_{rx}), \qquad
            \frac{\partial \dot{R}}{\partial p} = -\sum_k \frac{(I - u_k u_k^T)\, v}{R_k},

        where :math:`\dot{R}` is the path rate, positive closing, and the path range does
        not depend on v. The last term holds because moving the target turns each unit
        vector: :math:`\partial u_k / \partial p = (I - u_k u_k^T) / R_k`, the projection
        onto the plane across the line of sight, scaled by one over the range. The baseline
        does not depend on the state, so ``excess_range`` changes nothing here.

        References
        ----------
        .. [1] Bar-Shalom, Li and Kirubarajan (2001), §10.3 (the Jacobian an extended Kalman
               filter needs).
        """
        x = as_vector(state, self.state_layout.dimension, "state")
        position_m = self._tx.values(x, "_m")
        velocity_mps = self._tx.values(x, "dot_mps")
        self._check_off_sites(position_m)

        # One row per site: the transmitter, then the receiver.
        sites_m = np.stack([self._tx.pose.position_m, self._rx.pose.position_m])
        to_target_m = position_m - sites_m
        range_m = np.linalg.norm(to_target_m, axis=-1)
        unit = to_target_m / range_m[:, None]
        unit_sum = unit.sum(axis=0)
        # (I - u uᵀ) v = v - u (u · v), for both sites at once, then divided by each range.
        across_mps = velocity_mps - unit * (unit @ velocity_mps)[:, None]
        rate_by_position = -np.sum(across_mps / range_m[:, None], axis=0)

        # Derivatives with respect to east, north and up, for position and for velocity.
        by_position = np.stack([unit_sum, rate_by_position])
        by_velocity = np.stack([np.zeros(3), -unit_sum])
        jacobian = np.zeros((2, self.state_layout.dimension), dtype=np.float64)
        for derivative, suffix in ((by_position, "_m"), (by_velocity, "dot_mps")):
            # A loop over the two kinds of coordinate: a fixed one has no state column, so
            # only the axes the state holds are copied across, by name.
            names = [f"{axis}{suffix}" for axis in "xyz"]
            axes = [i for i, name in enumerate(names) if name in self.state_layout.names]
            columns = list(self.state_layout.indices(tuple(names[i] for i in axes)))
            jacobian[:, columns] = derivative[:, axes]
        return jacobian


# The four components a monostatic radar can report, in their canonical order. Only azimuth
# wraps: elevation lies in [-pi/2, pi/2] and never crosses a period.
_SPHERICAL_COORDINATES = (
    Coordinate("range_m", "m"),
    Coordinate("azimuth_rad", "rad", period=2.0 * np.pi),
    Coordinate("elevation_rad", "rad"),
    Coordinate("range_rate_mps", "m/s"),
)
_SPHERICAL_NAMES = tuple(c.name for c in _SPHERICAL_COORDINATES)


class RangeAzimuthElevationRangeRate:
    """Predict the range, azimuth, elevation and range rate a monostatic radar reports.

    The state is Cartesian (east, north, up) and the measurement is in the radar's own
    spherical coordinates, relative to its site:

    - ``range_m``, the slant range r = |p - p_s|;
    - ``azimuth_rad``, atan2(Δe, Δn), zero at north and clockwise, with period 2π;
    - ``elevation_rad``, asin(Δu / r), up from the local horizontal;
    - ``range_rate_mps``, -(p - p_s) · v / r, positive for a closing target.

    The measurement stays spherical. The filter's linearisation bridges the two
    coordinate systems: the Jacobian for the EKF, the sigma points for the UKF [1]_.
    A measurement is converted to Cartesian only to start a track
    (:class:`~radar_forge.core.tracking.initiation.DebiasedSphericalInitiator`).

    Parameters
    ----------
    state_layout : StateLayout
        Layout of the Cartesian state.
    sensor : SensorPose
        The radar's site. It must share the state's frame and origin.
    components : tuple of str, optional
        The components the radar reports, by name and in measurement order, from
        ``range_m``, ``azimuth_rad``, ``elevation_rad`` and ``range_rate_mps``. The
        default is all four. Drop ``elevation_rad`` for a 2-D radar, and
        ``range_rate_mps`` for one that measures no Doppler.
    fixed_coordinates : mapping of str to float or None, optional
        A value, in SI units, for each of ``x_m``, ``y_m``, ``z_m`` and, when range
        rate is reported, ``xdot_mps``, ``ydot_mps`` and ``zdot_mps``, that the state
        does not hold. A 2-D state fixes ``z_m`` (and ``zdot_mps``).

    Raises
    ------
    ValueError
        If ``components`` is empty, repeats a name or names an unknown component. If
        the site's frame or origin differs from the state's. Also if a fixed
        coordinate is not needed or is already in the state, a needed coordinate is
        missing from both, or a fixed value is not finite.

    Notes
    -----
    The geometry is :func:`radar_forge.core.geodesy.enu_to_range_azimuth_elevation_rad`
    applied to Δp = p - p_s. The tracker and the rest of the library therefore share one
    implementation of it, and one angle convention (``spec/data-001-formats.md`` DF2).

    Two places are singular. The model raises there rather than return a non-finite
    value or Jacobian:

    - **The site itself**, r = 0. Every direction is undefined, and so is every row
      of the Jacobian.
    - **The zenith**, a target straight above the site, where the ground range
      g = sqrt(Δe² + Δn²) is zero. The azimuth is undefined there, and the azimuth
      and elevation rows of the Jacobian divide by g. This is checked only when an
      angle is reported.

    Both checks compare positions exactly, as :class:`BistaticRangeDopplerModel` does.
    Near either place the Jacobian is finite but large, and the linearisation the EKF
    relies on holds over a small region only.

    References
    ----------
    .. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications
           to Tracking and Navigation*, Wiley, 2001, §10.3 (the extended Kalman filter).
    .. [2] S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press,
           2013, ch. 5 (the extended and unscented Kalman filters).

    Examples
    --------
    A target 1 km due east of the radar, at the radar's height, flying straight at it
    at 10 m/s:

    >>> import numpy as np
    >>> names = ("x_m", "xdot_mps", "y_m", "ydot_mps", "z_m", "zdot_mps")
    >>> units = ("m", "m/s") * 3
    >>> origin = (36.0, -78.9, 60.0)
    >>> layout = StateLayout(
    ...     tuple(Coordinate(n, u) for n, u in zip(names, units)), "ENU", origin
    ... )
    >>> radar = SensorPose(np.zeros(3), "ENU", origin)
    >>> model = RangeAzimuthElevationRangeRate(layout, radar)
    >>> z = model.predict(np.array([1000.0, -10.0, 0.0, 0.0, 0.0, 0.0]))
    >>> [round(float(value), 6) for value in z]
    [1000.0, 1.570796, 0.0, 10.0]
    """

    def __init__(
        self,
        state_layout: StateLayout,
        sensor: SensorPose,
        components: tuple[str, ...] = _SPHERICAL_NAMES,
        fixed_coordinates: Mapping[str, float] | None = None,
    ) -> None:
        components = tuple(components)
        unknown = set(components) - set(_SPHERICAL_NAMES)
        if not components or len(set(components)) != len(components) or unknown:
            msg = (
                f"components must be nonempty and unique, from {_SPHERICAL_NAMES}; "
                f"got {components}."
            )
            raise ValueError(msg)

        self.state_layout = state_layout
        self.components = components
        self._measures_rate = "range_rate_mps" in components
        self._measures_angle = bool({"azimuth_rad", "elevation_rad"} & set(components))
        self._geometry = _Geometry(
            state_layout, sensor, fixed_coordinates, include_velocity=self._measures_rate
        )
        # Rows of the full (range, azimuth, elevation, range rate) vector, in the order
        # the radar reports them.
        self._rows = [_SPHERICAL_NAMES.index(name) for name in components]
        self.measurement_layout = StateLayout(
            tuple(_SPHERICAL_COORDINATES[row] for row in self._rows), "spherical"
        )

    @property
    def sensor(self) -> SensorPose:
        """The radar's site."""
        return self._geometry.pose

    def _check_not_singular(self, to_target_m: NDArray[np.float64]) -> None:
        """Raise if a target is on the site, or above it when an angle is reported."""
        # Exact comparisons, as in BistaticRangeDopplerModel: a distance is zero only when
        # the two points are equal.
        if np.any(np.all(to_target_m == 0.0, axis=-1)):
            msg = "spherical geometry is singular at the site: a target position is on the radar."
            raise ValueError(msg)
        if self._measures_angle and np.any(np.all(to_target_m[..., :2] == 0.0, axis=-1)):
            msg = (
                "spherical geometry is singular at the zenith: a target is straight above "
                "the radar, where azimuth is undefined."
            )
            raise ValueError(msg)

    def predict(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the reported components for one state or a stack of states.

        Parameters
        ----------
        state : numpy.ndarray
            One state, shape ``(n_state,)``, or a stack, shape ``(n_points, n_state)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_meas,)`` for one state or ``(n_points, n_meas)`` for a stack, in
            ``components`` order. Azimuth lies in [0, 2π).

        Raises
        ------
        ValueError
            If ``state`` has the wrong shape or is not finite, or a target is at a
            singular place (class Notes).
        """
        states = np.asarray(state, dtype=np.float64)
        check_points(states, self.state_layout.dimension, "state")
        to_target_m = self._geometry.values(states, "_m") - self._geometry.pose.position_m
        self._check_not_singular(to_target_m)

        range_m, azimuth_rad, elevation_rad = enu_to_range_azimuth_elevation_rad(to_target_m)
        if self._measures_rate:
            velocity_mps = self._geometry.values(states, "dot_mps")
            range_rate_mps = -np.sum(to_target_m * velocity_mps, axis=-1) / range_m
        else:
            # Not reported, and the state may hold no velocity; the row is dropped below.
            range_rate_mps = np.zeros_like(range_m)
        full = np.stack([range_m, azimuth_rad, elevation_rad, range_rate_mps], axis=-1)
        reported: NDArray[np.float64] = full[..., self._rows]
        return reported

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Return the derivatives of the reported components at one state.

        Parameters
        ----------
        state : numpy.ndarray
            One state, shape ``(n_state,)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_meas, n_state)``, one row per component in ``components``
            order. A coordinate the state lacks, held at a fixed value, has no column.

        Raises
        ------
        ValueError
            If ``state`` has the wrong shape or is not finite, or a target is at a
            singular place (class Notes).

        Notes
        -----
        Write Δp = (Δe, Δn, Δu) = p - p_s, r = |Δp|, u = Δp / r for the unit vector to
        the target, g = sqrt(Δe² + Δn²) for the ground range, and v for the velocity.
        The derivatives with respect to position are

        .. math::

            \frac{\partial r}{\partial p} &= u^T, \\
            \frac{\partial\, \mathrm{az}}{\partial p} &=
                \frac{1}{g^2} \begin{pmatrix} \Delta n & -\Delta e & 0 \end{pmatrix}, \\
            \frac{\partial\, \mathrm{el}}{\partial p} &=
                \frac{1}{r^2 g}
                \begin{pmatrix} -\Delta e\, \Delta u & -\Delta n\, \Delta u & g^2
                \end{pmatrix}, \\
            \frac{\partial \dot{r}}{\partial p} &= -\frac{(I - u u^T)\, v}{r},

        and the only derivative with respect to velocity is
        :math:`\partial \dot{r} / \partial v = -u^T`.

        - The azimuth row is the derivative of atan2(Δe, Δn).
        - The elevation row is that of atan2(Δu, g), which equals asin(Δu / r) and is
          easier to differentiate.
        - The range-rate row on position holds because moving the target turns u:
          :math:`\partial u / \partial p = (I - u u^T) / r`, as in
          :meth:`BistaticRangeDopplerModel.jacobian`.

        Angles are in radians, so the angle rows are in rad/m.

        References
        ----------
        .. [1] Bar-Shalom, Li and Kirubarajan (2001), §10.3 (the Jacobian an extended
               Kalman filter needs).
        """
        x = as_vector(state, self.state_layout.dimension, "state")
        to_target_m = self._geometry.values(x, "_m") - self._geometry.pose.position_m
        self._check_not_singular(to_target_m)

        east_m, north_m, up_m = to_target_m
        range_m = float(np.linalg.norm(to_target_m))
        ground_m = float(np.hypot(east_m, north_m))
        unit = to_target_m / range_m

        # One row per component in canonical order; the reported rows are picked out below.
        # A row that is not reported may be undefined here (the azimuth at the zenith when
        # only range is reported), so it is built only when needed.
        by_position = np.zeros((4, 3), dtype=np.float64)
        by_velocity = np.zeros((4, 3), dtype=np.float64)
        by_position[0] = unit
        if self._measures_angle:
            by_position[1] = np.array([north_m, -east_m, 0.0]) / ground_m**2
            by_position[2] = np.array([-east_m * up_m, -north_m * up_m, ground_m**2]) / (
                range_m**2 * ground_m
            )
        if self._measures_rate:
            velocity_mps = self._geometry.values(x, "dot_mps")
            # (I - u uᵀ) v = v - u (u · v), divided by the range.
            by_position[3] = -(velocity_mps - unit * (unit @ velocity_mps)) / range_m
            by_velocity[3] = -unit

        jacobian = np.zeros((len(self._rows), self.state_layout.dimension), dtype=np.float64)
        for derivative, suffix in ((by_position, "_m"), (by_velocity, "dot_mps")):
            # A loop over the two kinds of coordinate: a fixed one has no state column, so
            # only the axes the state holds are copied across, by name.
            names = [f"{axis}{suffix}" for axis in "xyz"]
            axes = [i for i, name in enumerate(names) if name in self.state_layout.names]
            columns = list(self.state_layout.indices(tuple(names[i] for i in axes)))
            jacobian[:, columns] = derivative[self._rows][:, axes]
        return jacobian
