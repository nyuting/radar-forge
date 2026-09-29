"""Named coordinates, residual topology, and detached Bayesian estimates.

Classes:
- Coordinate
- StateSpace
- MeasurementSpace (= StateSpace)
- StateEstimate

References
----------
.. [1] StoneSoup state and measurement mappings, https://stonesoup.readthedocs.io/.
.. [2] Local semantic spaces, spec/tracker-001-provenance.md. Periods are generalized
       from radians to arbitrary units so folded range is represented explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from radar_forge.tracking._numerics import FloatArray, covariance, timestamp, vector

__all__ = [
    "Coordinate",
    "MeasurementSpace",
    "StateEstimate",
    "StateSpace",
]


@dataclass(frozen=True)
class Coordinate:
    """One coordinate's name, SI unit, and optional period in that unit.

    Parameters
    ----------
    name, unit : str
        Unique unit-suffixed name and physical unit, e.g. x_m and m.
    period : float or None
        Positive wrapping period; None denotes a real-valued coordinate.

    References
    ----------
    .. [1] StoneSoup angle types and measurement mappings.
    """

    name: str
    unit: str
    period: float | None = None

    ###################################################################################################
    # This function checks the stored quantity name, unit, and optional repeating interval.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - None; rejects empty names or units and invalid repeating intervals.
    def __post_init__(self) -> None:
        """Validate the documented constructor contract."""

        if not self.name or not self.unit:
            raise ValueError("coordinate name and unit must be nonempty")
        
        # Periodic coordinates (e.g. angles) must have a valid positive period
        if self.period is not None and (not np.isfinite(self.period) or self.period <= 0):
            raise ValueError("coordinate period must be finite and positive")
    ###################################################################################################

@dataclass(frozen=True)
class StateSpace:
    """Ordered coordinates and frame identity, independent of dimension.

    Parameters
    ----------
    coordinates : tuple of Coordinate
        Vector order; covariance entries have products of the corresponding units.
    frame : str
        Frame identity, compared as part of model compatibility.
    origin_lla_deg_m : tuple of float or None
        ENU origin as latitude degrees, longitude degrees, altitude metres.
        Required by Cartesian built-ins; optional for nongeographic custom spaces.

    References
    ----------
    .. [1] StoneSoup explicit state mappings; local semantic spaces (provenance manifest).
    """

    coordinates: tuple[Coordinate, ...]
    frame: str = "local"
    origin_lla_deg_m: tuple[float, float, float] | None = None

    ###################################################################################################
    # This function checks the stored coordinate definitions so models agree on names, order, and origin.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - None; stores tuples and rejects duplicate names or an invalid origin.
    def __post_init__(self) -> None:
        """Validate the documented constructor contract."""

        # Store coordinates as an immutable tuple so that the vector cannot change after construction
        object.__setattr__(self, "coordinates", tuple(self.coordinates))

        # Coordinate names must be unique within a StateSpace
        if not self.coordinates or not self.frame or len(set(self.names)) != len(self.names):
            raise ValueError("state space requires unique coordinates and a nonempty frame")

        if self.origin_lla_deg_m is not None:

            # Validate ENU origin as (lat, long, altitude)
            origin = tuple(self.origin_lla_deg_m)

            # Check that the origin is in the correct units
            if (
                len(origin) != 3
                or not np.all(np.isfinite(origin))
                or abs(origin[0]) > 90      # latitude has limits
                or abs(origin[1]) > 180     # longitude has limits
            ):
                raise ValueError("ENU origin requires latitude/longitude degrees and altitude metres")
            
            object.__setattr__(self, "origin_lla_deg_m", origin)
    ###################################################################################################

    ###################################################################################################
    # This function lists the stored quantity names in the order used by the filter.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - tuple of str: Coordinate names, such as x_m followed by xdot_mps.
    @property
    def names(self) -> tuple[str, ...]:
        """Coordinate names in state vector order; see StateSpace References."""
        return tuple(c.name for c in self.coordinates) # preserve StateSpace-defined coordinate ordering
    ###################################################################################################

    ###################################################################################################
    # This function counts how many quantities this filter tracks.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - int: Number of coordinates in the stored definition.
    @property
    def dimension(self) -> int:
        """Number of vector elements; see StateSpace References."""
        return len(self.coordinates)
    ###################################################################################################

    ###################################################################################################
    # This function finds array positions for named quantities so models do not rely on fixed ordering.
    #
    # Inputs:
    # - names (tuple of str): Coordinate names to look up, in the desired order.
    #
    # Outputs:
    # - tuple of int: Matching array indices; an unknown name raises an error.
    def indices(self, names: tuple[str, ...]) -> tuple[int, ...]:
        """Map names to vector indices; unknown names raise ValueError.

        References
        ----------
        .. [1] StoneSoup measurement mapping convention.
        """
        return tuple(self.names.index(name) for name in names) # convert names into state-vector indices
    ###################################################################################################

    ###################################################################################################
    # This function copies tracked values and brings repeating values, such as angles, into their
    # shared interval.
    #
    # Inputs:
    # - value (ArrayLike): One number per stored coordinate, shape (n,), in its stated unit.
    #
    # Outputs:
    # - float64 NumPy array: Copy, shape (n,); repeating values lie from minus half a period up to
    #   half a period.
    def normalize(self, value: ArrayLike) -> FloatArray:
        """Copy a vector (n,), wrapping periodic entries to [-period/2, period/2).

        References
        ----------
        .. [1] StoneSoup angle types; generalized to arbitrary coordinate periods.
        """
        result = vector(value, self.dimension)

        # Wrap periodic coordinates onto their principal interval
        for i, coord in enumerate(self.coordinates):

            if coord.period is not None:

                # Ensure interval of (-180, 180]. E.g. 200 deg -> -160 deg
                result[i] = (result[i] + coord.period / 2) % coord.period - coord.period / 2
        
        return result
    ###################################################################################################

    ###################################################################################################
    # This function subtracts two sets of values, taking the shorter route across an angle or range wrap.
    #
    # Inputs:
    # - a (ArrayLike): First set of values, shape (n,), in coordinate order and units.
    # - b (ArrayLike): Values to subtract, with the same shape, order, and units.
    #
    # Outputs:
    # - float64 NumPy array: Differences, shape (n,), adjusted for repeating coordinates.
    def residual(self, a: ArrayLike, b: ArrayLike) -> FloatArray:
        """Return shortest coordinate differences, shape (n,), in coordinate units.

        References
        ----------
        .. [1] FilterPy UKF residual callbacks.
        """
        # Subtract using periodic-aware arithmetic (wrap subtraction in the normalize function)
        # E.g. instead of 359 deg - 1 deg = 358 deg, this returns 359 deg - 1 deg = -2 deg
        return self.normalize(vector(a, self.dimension) - vector(b, self.dimension))
    ###################################################################################################

    ###################################################################################################
    # This function combines possible states into one estimate while respecting repeating coordinates.
    #
    # Inputs:
    # - points (ArrayLike): Table of k possible states and n quantities, shape (k, n).
    # - weights (ArrayLike): Contribution of each state, shape (k,); values must sum to one.
    #
    # Outputs:
    # - float64 NumPy array: Combined values, shape (n,), in coordinate units; an undefined angle
    #   average raises an error.
    def weighted_mean(self, points: ArrayLike, weights: ArrayLike) -> FloatArray:
        """Compute a real/circular weighted mean of (k,n) points with (k,) weights.

        Notes
        -----
        An antipodal circular distribution has no unique mean and is rejected.
        This local Gaussian operation does not represent multimodal angle densities.

        References
        ----------
        .. [1] FilterPy UKF mean callbacks; Julier, scaled unscented transform, 2002.
        """

        array = np.asarray(points, dtype=np.float64)

        if array.ndim != 2 or array.shape[1] != self.dimension or not np.all(np.isfinite(array)):
            raise ValueError("points must be finite with shape (k, n)")
        
        weight = vector(weights, len(array), "weights")

        # Weights of all means represent one complete weighted distribution => weights must sum to 1
        if not np.isclose(weight.sum(), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("mean weights must sum to one")
        
        # Standard Euclidean weighted mean for non-periodic coordinates
        result = weight @ array
        
        # Periodic coordinates require circular averaging
        for i, coord in enumerate(self.coordinates):

            if coord.period is not None:

                # Convert coordinate values onto the unit circle (2pi phase) so that every periodic 
                # quantity can use the same circular-mean mathematics
                phase = array[:, i] * (2 * np.pi / coord.period)

                # Average points on the unit circle instead of averaging the raw numbers
                # This avoids the failure where +179 deg and -179 deg average to 0; it should be 180 deg
                sine = float(weight @ np.sin(phase))
                cosine = float(weight @ np.cos(phase))

                # Opposing directions have no unique circular mean; if the resultant vector has essentially
                # zero length, the samples are balanced around the circle and no unique circular mean exists
                # E.g. equal weight at 0 deg and 180 deg has no preferred direction
                if np.hypot(sine, cosine) < 1e-12:
                    raise ValueError("periodic coordinate mean is undefined")
                
                # Convert the circular mean back into coordinate units
                # atan2 recovers average direction; scale from radians into the original unit and period
                result[i] = np.arctan2(sine, cosine) * coord.period / (2 * np.pi)
        
        return np.asarray(result, dtype=np.float64)
    ###################################################################################################


#############################################################################
#  MeasurementSpace and StateSpace share the same properties and semantics  #
#  There is no need to duplicate the implementation, we can simply set      #
                                                                            #
MeasurementSpace = StateSpace                                               #
                                                                            #
#############################################################################

@dataclass(frozen=True)
class StateEstimate:
    """Detached mean (n,), covariance (n,n), timestamp_s and semantic state space.

    Notes
    -----
    Constructor inputs are copied. Covariances are symmetric positive semidefinite;
    every covariance element has the product of its two coordinate units.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    """

    mean: FloatArray
    covariance: FloatArray
    timestamp_s: float
    state_space: StateSpace

    ###################################################################################################
    # This function checks the stored estimate and makes read-only copies of its values and uncertainty.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - None; replaces stored arrays with checked copies and validates the time in seconds.
    def __post_init__(self) -> None:
        """Validate the documented constructor contract."""

        # Normalise periodic coordinates in the state mean to their unique canonical representation
        mean = self.state_space.normalize(self.mean)

        # Validate state covariance (P) dimensions and positive semi-definiteness
        cov = covariance(self.covariance, self.state_space.dimension)

        # Prevent modification after construction, ensuring this remains a true snapshot of the filter
        mean.setflags(write=False)
        cov.setflags(write=False)

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", cov)

        # Normalise timestamp representation
        object.__setattr__(self, "timestamp_s", timestamp(self.timestamp_s))
    ###################################################################################################
