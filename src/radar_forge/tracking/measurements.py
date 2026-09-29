"""Sensor observations and explicit mappings from configurable tracking states.

Classes:
- MeasurementModel(Protocol)
- CartesianPosition
- SensorPose
- _Geometry
- MonostaticRadar
- BistaticRangeDoppler
- CompositeMeasurementModel

References
----------
.. [1] StoneSoup measurement mappings, https://stonesoup.readthedocs.io/.
.. [2] Richards, Fundamentals of Radar Signal Processing, 2nd ed., 2014.
.. [3] Local observations and composite models; spec/tracker-001-provenance.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from radar_forge.tracking._numerics import FloatArray, vector

# Compatibility exports: observation records are owned by sensors, while this
# module defines the models that predict their values from a tracking state.
from radar_forge.tracking.sensors import Measurement, MeasurementBatch
from radar_forge.tracking.spaces import Coordinate, StateSpace

__all__ = [
    "BistaticRangeDoppler",
    "CartesianPosition",
    "CompositeMeasurementModel",
    "Measurement",
    "MeasurementBatch",
    "MeasurementModel",
    "MonostaticRadar",
    "SensorPose",
]


class MeasurementModel(Protocol):
    """Map a semantic state to an observation without estimator dependencies.

    References
    ----------
    .. [1] StoneSoup measurement model separation.
    """

    state_space: StateSpace
    measurement_space: StateSpace

    ###################################################################################################
    # This function defines how a custom sensor model must turn tracked values into expected
    # readings.
    #
    # Inputs:
    # - state (float64 NumPy array): Tracked values, shape (n,), one per named quantity, in its
    #   stated unit.
    #
    # Outputs:
    # - float64 NumPy array: Required sensor readings, shape (m,), in the model's measurement
    #   order and units.
    def predict(self, state: FloatArray) -> FloatArray:
        """Map state (n,) to observation (m,); see MeasurementModel References."""
        ...
    ###################################################################################################

class CartesianPosition:
    """Select any named coordinates, optionally with periodic observation residuals.

    Parameters
    ----------
    state_space : StateSpace
        State coordinate definition.
    names : tuple of str
        Observed coordinate names, in measurement order.
    periods : mapping of str to float or None
        Optional measurement wrapping periods, in each coordinate's unit. The
        predicted value stays on the continuous branch; residuals choose the alias.

    References
    ----------
    .. [1] StoneSoup LinearGaussian mapping; local CartesianPosition selection.
    """
    ###################################################################################################
    # This function selects which named tracked quantities a sensor reports directly.
    #
    # Inputs:
    # - state_space (StateSpace): Names, order, units, and shared origin of the tracked quantities.
    # - names (tuple of str): Quantities the sensor observes, in measurement order.
    # - periods (mapping of str to float, or None): Repeating intervals in coordinate units; None
    #   uses the existing definitions.
    #
    # Outputs:
    # - None; stores the selected array positions and measurement definitions.
    def __init__(
        self,
        state_space: StateSpace,
        names: tuple[str, ...],
        periods: Mapping[str, float] | None = None,
    ) -> None:
        if not names or len(set(names)) != len(names):
            raise ValueError("observed coordinate names must be nonempty and unique")

        self.state_space = state_space

        # Locate the observed coordinates within the tracker state vector
        self.indices = state_space.indices(names)

        periods = periods or {}

        # Wrapping periods may only be specified for observed coordinates
        if set(periods) - set(names):
            raise ValueError("periods must refer to observed coordinates")

        # Measurement space consists only of the selected coordinates
        self.measurement_space = StateSpace(
            tuple(
                Coordinate(
                    name=c.name, 
                    unit=c.unit, 
                    period=periods.get(c.name, c.period)
                )
                for c in (state_space.coordinates[i] for i in self.indices)
            ),
            state_space.frame,
            state_space.origin_lla_deg_m,
        )
    ###################################################################################################

    ###################################################################################################
    # This function extracts selected tracked values for comparison with a sensor reading.
    #
    # Inputs:
    # - state (float64 NumPy array): Tracked values, shape (n,), one per named quantity, in its
    #   stated unit.
    #
    # Outputs:
    # - float64 NumPy array: Selected values in measurement order, shape (m,); a new array.
    def predict(self, state: FloatArray) -> FloatArray:
        """Select observation (m,) from state (n,); see CartesianPosition References."""
        return vector(state, self.state_space.dimension)[list(self.indices)]
    ###################################################################################################

@dataclass(frozen=True)
class SensorPose:
    """Static ENU sensor position (3,), with matching frame and origin identity.

    Parameters
    ----------
    position_m : numpy.ndarray
        East/north/up position in metres, shape (3,).
    frame : str
        State frame identifier.
    origin_lla_deg_m : tuple of float
        ENU origin in degrees/degrees/metres.

    References
    ----------
    .. [1] Radar-Forge geodesy conventions; StoneSoup sensor translation offsets.
    """

    position_m: FloatArray
    frame: str
    origin_lla_deg_m: tuple[float, float, float]

    ###################################################################################################
    # This function checks the stored sensor location so radar calculations use the correct shared
    # origin.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - None; stores a read-only copy of the three position values and checks the origin.
    def __post_init__(self) -> None:
        """Validate the documented constructor contract."""

        # Validate and freeze the sensor position vector
        position = vector(self.position_m, 3)
        position.setflags(write=False)
        object.__setattr__(self, "position_m", position)

        # Normalise and validate the ENU origin definition
        check = StateSpace((Coordinate("x_m", "m"),), self.frame, self.origin_lla_deg_m)
        object.__setattr__(self, "origin_lla_deg_m", check.origin_lla_deg_m)
    ###################################################################################################

class _Geometry:
    ###################################################################################################
    # This function checks that radar calculations have every needed coordinate, either tracked or
    # explicitly fixed.
    #
    # Inputs:
    # - space (StateSpace): Tracked coordinate definitions and shared origin.
    # - pose (SensorPose): Sensor position in east/north/up metres and its shared origin.
    # - fixed (mapping of str to float, or None): Explicit values for absent coordinates; None
    #   supplies none.
    # - velocity (bool): Whether all three velocity components are required as well as position.
    #
    # Outputs:
    # - None; stores the geometry settings and rejects missing coordinates or conflicting origins.
    # The velocity flag documented above is named include_velocity in the signature.
    def __init__(
        self, space: StateSpace, pose: SensorPose, fixed: Mapping[str, float] | None,
        include_velocity: bool
    ) -> None:
        # Sensor (`pose`) and tracker (`space`) state must use the same ENU reference frame
        if space.frame != pose.frame or space.origin_lla_deg_m != pose.origin_lla_deg_m:
            raise ValueError("sensor and state must share ENU frame and origin")

        self.space = space
        self.pose = pose
        self.fixed = dict(fixed or {})

        # Position coordinates are always required (x_m, y_m, z_m)
        required = tuple(f"{a}_m" for a in "xyz")

        # Some models also require velocity coordinates
        if include_velocity:
            required += tuple(f"{a}dot_mps" for a in "xyz")

        # Fixed values are only allowed for missing required coordinates
        if set(self.fixed) - set(required) or set(self.fixed) & set(space.names):
            raise ValueError("fixed coordinates must be required, absent state coordinates")

        # Every required coordinate must exist in the state or be explicitly fixed
        if any(name not in space.names and name not in self.fixed for name in required):
            raise ValueError("geometry requires missing coordinates to be explicitly fixed")
        
        # Check that all fixed coordinates are finite
        if not all(np.isfinite(v) for v in self.fixed.values()):
            raise ValueError("fixed coordinates must be finite")
    ###################################################################################################

    ###################################################################################################
    # This function collects three coordinates for radar calculations, filling only explicitly
    # fixed missing values.
    #
    # Inputs:
    # - state (float64 NumPy array): Tracked values, shape (n,), one per named quantity, in its
    #   stated unit.
    # - suffix (str): Quantity to collect: _m for position or dot_mps for velocity.
    #
    # Outputs:
    # - float64 NumPy array: East, north, and up values, shape (3,), in metres or m/s.
    def values(self, state: FloatArray, suffix: str) -> FloatArray:
        # Build a dictionary (coordinate-name -> value) from the state vector
        values = dict(zip(self.space.names, vector(state, self.space.dimension), strict=True))
        
        # Fill in any coordinates supplied as fixed constants
        values.update(self.fixed)

        # E.g. values(state, "_m") -> [x_m, y_m, z_m] 
        return np.asarray([values[f"{a}{suffix}"] for a in "xyz"], dtype=np.float64)
    ###################################################################################################

class MonostaticRadar:
    """Static sensor range, clockwise-north azimuth, elevation and optional closing velocity.

    Parameters
    ----------
    state_space : StateSpace
        Named Cartesian coordinates.
    pose : SensorPose
        Sensor ENU position and frame.
    include_velocity : bool
        Append positive-closing radial velocity in m/s.
    fixed_coordinates : mapping or None
        Explicit values for omitted position/velocity coordinates, in SI units.

    Notes
    -----
    Observation order is range_m, azimuth_rad, elevation_rad, optionally
    radial_velocity_mps. All three position coordinates must be supplied or fixed.
    Elevation is bounded, not an independently wrapped angle. Geometry on the
    sensor's vertical axis is rejected because azimuth is undefined.

    References
    ----------
    .. [1] StoneSoup CartesianToElevationBearingRangeRate; Radar-Forge ENU/sign conventions.
    """
    ###################################################################################################
    # This function sets up expected readings for a radar whose transmitter and receiver share one
    # location.
    #
    # Inputs:
    # - state_space (StateSpace): Names, order, units, and shared origin of the tracked
    #   quantities.
    # - pose (SensorPose): Sensor position in east/north/up metres and its shared origin.
    # - include_velocity (bool): Whether to include velocity toward the radar in m/s.
    # - fixed_coordinates (mapping of str to float, or None): Explicit values for missing
    #   coordinates; None supplies none.
    #
    # Outputs:
    # - None; stores sensor geometry and the measurement order for later comparisons.
    def __init__(
        self,
        state_space: StateSpace,
        pose: SensorPose,
        include_velocity: bool = False,
        fixed_coordinates: Mapping[str, float] | None = None,
    ) -> None:
        # Initialise fields
        self.state_space = state_space
        self.pose = pose
        self.include_velocity = include_velocity

        # Helper for extracting position/velocity in radar geometry calculations
        self.geometry = _Geometry(state_space, pose, fixed_coordinates, include_velocity)

        # Standard monostatic radar measurement ordering
        coords: tuple[Coordinate, ...] = (
            Coordinate("range_m", "m"),
            Coordinate("azimuth_rad", "rad", 2 * np.pi),
            Coordinate("elevation_rad", "rad"),
        )

        # Optionally include radial velocity
        if include_velocity:
            coords += (Coordinate("radial_velocity_mps", "m/s"),)
        
        # Store the selected coordinates that have been initialised
        self.measurement_space = StateSpace(coords, f"sensor:{pose.frame}", pose.origin_lla_deg_m)
    ###################################################################################################

    ###################################################################################################
    # This function calculates what the radar would read for a proposed target state.
    #
    # Inputs:
    # - state (float64 NumPy array): Tracked values, shape (n,), one per named quantity, in its
    #   stated unit.
    #
    # Outputs:
    # - float64 NumPy array: Distance m, clockwise-from-north angle rad, angle above horizontal
    #   rad, and optional closing velocity m/s; shape (3,) or (4,).
    def predict(self, state: FloatArray) -> FloatArray:
        """Return radar observation (3,) or (4,); see MonostaticRadar References."""
        # Vector from radar to target
        delta = self.geometry.values(state, "_m") - self.pose.position_m

        distance_m = float(np.linalg.norm(delta))
        horizontal_m = float(np.hypot(delta[0], delta[1]))

        # Azimuth is undefined directly above or below the radar
        if horizontal_m == 0:
            raise ValueError("azimuth is undefined on the sensor vertical axis")
        
        result = [
            distance_m,                                 # range
            float(np.arctan2(delta[0], delta[1])),      # azimuth (clockwise-from-north)
            float(np.arctan2(delta[2], horizontal_m)),  # elevation (above horizontal plane)
        ]

        if self.include_velocity:
            # if we include Doppler, positive velocity means that target is closing in on radar
            result.append(-float(delta @ self.geometry.values(state, "dot_mps")) / distance_m)

        return np.asarray(result, dtype=np.float64)
    ###################################################################################################

class BistaticRangeDoppler:
    """Static bistatic path length and positive-closing path rate, in m and m/s.

    Parameters
    ----------
    state_space : StateSpace
        Named Cartesian state.
    transmitter, receiver : SensorPose
        Static assets in the same ENU frame.
    excess_range : bool
        Subtract the transmitter-receiver baseline from total path length.
    fixed_coordinates : mapping or None
        Explicit SI values for absent Cartesian coordinates.

    Notes
    -----
    Closing path rate is -(u_tx+u_rx).velocity; it is not monostatic radial
    velocity or Doppler Hz. Device-frequency conversion belongs in the adapter.

    References
    ----------
    .. [1] Richards, 2014; local bistatic model with closing-positive sign adaptation.
    """
    ###################################################################################################
    # This function sets up expected readings when the transmitter and receiver can be at
    # different locations.
    #
    # Inputs:
    # - state_space (StateSpace): Names, order, units, and shared origin of the tracked
    #   quantities.
    # - transmitter (SensorPose): Transmitter position and shared origin.
    # - receiver (SensorPose): Receiver position and the same shared origin.
    # - excess_range (bool): Whether to subtract the direct transmitter-to-receiver distance.
    # - fixed_coordinates (mapping of str to float, or None): Explicit values for missing
    #   coordinates; None supplies none.
    #
    # Outputs:
    # - None; stores both locations and the definition of the measured signal path.
    def __init__(
        self,
        state_space: StateSpace,
        transmitter: SensorPose,
        receiver: SensorPose,
        excess_range: bool = True,
        fixed_coordinates: Mapping[str, float] | None = None,
    ) -> None:
        self.state_space = state_space
        self._tx = _Geometry(state_space, transmitter, fixed_coordinates, True)     # Geometry relative to Tx
        self._rx = _Geometry(state_space, receiver, fixed_coordinates, True)        # Geometry relative to Rx
        self.excess_range = excess_range
        self.measurement_space = StateSpace(
            (
                Coordinate("path_range_m", "m"),            # Bistatic measurements contain both path length...
                Coordinate("closing_path_rate_mps", "m/s")  # ... and path rate
            ),
            "bistatic",
        )
    ###################################################################################################

    ###################################################################################################
    # This function calculates the signal path through a target and how quickly that path is shortening.
    #
    # Inputs:
    # - state (float64 NumPy array): Tracked values, shape (n,), one per named quantity, in its
    #   stated unit.
    #
    # Outputs:
    # - float64 NumPy array: Path length in metres and shortening rate in m/s, shape (2,);
    #   subtracts the direct path when configured.
    def predict(self, state: FloatArray) -> FloatArray:
        """Return path length/closing rate (2,); see BistaticRangeDoppler References."""
        # Target position in ENU coordinates
        position = self._tx.values(state, "_m")

        # Vectors from transmitter and receiver to target
        a = position - self._tx.pose.position_m
        b = position - self._rx.pose.position_m

        # Find magnitude of the vectors
        ra = float(np.linalg.norm(a))
        rb = float(np.linalg.norm(b))

        # Geometry becomes singular if the target is at a Tx or Rx
        if ra == 0 or rb == 0:
            raise ValueError("bistatic geometry is singular at an asset")
        
        # Direct Tx-Rx path length
        baseline = float(np.linalg.norm(self._tx.pose.position_m - self._rx.pose.position_m))
        
        return np.array(
            [
                ra + rb - (baseline if self.excess_range else 0),               # total (or excess) bistatic path length
                -float((a / ra + b / rb) @ self._tx.values(state, "dot_mps")),  # positive value means bistatic path is shortening
            ],
            dtype=np.float64,
        )
    ###################################################################################################

class CompositeMeasurementModel:
    """Concatenate simultaneous observations with a caller-supplied full covariance.

    Parameters
    ----------
    models : sequence of MeasurementModel
        Components with the exact same StateSpace. Cross-component covariance
        belongs in Measurement.covariance, not in separate independent updates.

    References
    ----------
    .. [1] Local composite measurement contract; StoneSoup combined measurement models.
    """
    ###################################################################################################
    # This function joins sensor models so related readings can be handled together in one update.
    #
    # Inputs:
    # - models (sequence of MeasurementModel): Ordered sensor models sharing the same tracked
    #   quantities and origin.
    #
    # Outputs:
    # - None; stores the models and creates their combined measurement definition.
    def __init__(self, models: Sequence[MeasurementModel]) -> None:
        # All component models must operate on the same tracker state
        if not models or any(m.state_space != models[0].state_space for m in models):
            raise ValueError("composite models must have a common StateSpace")
        
        # Initialise fields
        self.models = tuple(models)
        self.state_space = models[0].state_space

        # Build one measurement space by concatenating component measurements
        self.measurement_space = StateSpace(
            tuple(
                Coordinate(f"component{i}_{c.name}", c.unit, c.period)
                for i, m in enumerate(models)
                for c in m.measurement_space.coordinates
            ),
            "composite",
        )
    ###################################################################################################

    ###################################################################################################
    # This function joins each stored model's expected readings into one comparison vector.
    #
    # Inputs:
    # - state (float64 NumPy array): Tracked values, shape (n,), one per named quantity, in its
    #   stated unit.
    #
    # Outputs:
    # - float64 NumPy array: All component readings in model order, shape (total readings,), in
    #   their respective units.
    def predict(self, state: FloatArray) -> FloatArray:
        """Concatenate component predictions from every component model (sum(m),); 
        see composite References.
        """
        return np.concatenate([model.predict(state) for model in self.models]) 
    ###################################################################################################