"""Composable continuous-noise kinematics with named coordinate mappings.

Classes:
- MotionModel(Protocol)
- CartesianMotion
- RadialMotion
- CoordinatedTurn

References
----------
.. [1] StoneSoup CombinedLinearGaussianTransitionModel, https://stonesoup.readthedocs.io/.
.. [2] FilterPy Q_continuous_white_noise, https://filterpy.readthedocs.io/en/latest/common/common.html.
.. [3] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol

import numpy as np

from radar_forge.tracking._numerics import FloatArray, vector
from radar_forge.tracking.spaces import Coordinate, StateSpace

__all__ = [
    "CartesianMotion",
    "CoordinatedTurn",
    "MotionModel",
    "RadialMotion",
]


class MotionModel(Protocol):
    """Dynamics protocol; arrays use the StateSpace order and coordinate units.

    References
    ----------
    .. [1] StoneSoup transition model interface; local tracker motion protocol.
    """

    state_space: StateSpace
    ###################################################################################################
    # This function defines how a custom motion model must predict the next set of tracked values.
    #
    # Inputs:
    # - state (float64 NumPy array): Tracked values, shape (n,), one per named quantity, in its
    #   stated unit.
    # - dt_s (float): Time to advance, in seconds; must be zero or positive.
    #
    # Outputs:
    # - float64 NumPy array: Required predicted values, shape (n,), in the same order and units.
    def transition(self, state: FloatArray, dt_s: float) -> FloatArray:
        """Propagate (n,) state by dt_s seconds; see MotionModel References."""
        ...
    ###################################################################################################
    
    ###################################################################################################
    # This function defines how a custom motion model must describe uncertainty added during
    # prediction.
    #
    # Inputs:
    # - state (float64 NumPy array): Tracked values, shape (n,), one per named quantity, in its
    #   stated unit.
    # - dt_s (float): Time to advance, in seconds; must be zero or positive.
    #
    # Outputs:
    # - float64 NumPy array: Required added-uncertainty table, shape (n, n), in paired coordinate
    #   units.
    def process_noise(self, state: FloatArray, dt_s: float) -> FloatArray:
        """Return (n,n) additive covariance; see MotionModel References."""
        ...
    ###################################################################################################

###################################################################################################
# This function checks the time step before a motion model predicts forward.
#
# Inputs:
# - dt_s (float): Time to advance, in seconds; must be zero or positive.
#
# Outputs:
# - None; rejects negative, missing, or infinite time steps.
def _elapsed(dt_s: float) -> None:
    # Prediction backwards in time is not supported by these discrete motion models
    if not np.isfinite(dt_s) or dt_s < 0:
        raise ValueError("dt_s must be finite and nonnegative")
###################################################################################################

class CartesianMotion:
    """Combine CV/CA ENU axis models using coordinate names, with full covariance.

    Parameters
    ----------
    axes : mapping of str to str
        Nonempty subset of x/y/z mapped to CV or CA. Default order is x, y, z.
    origin_lla_deg_m : tuple of float
        Explicit ENU origin (latitude degrees, longitude degrees, altitude metres).
    noise_density : float or mapping of str to float
        Continuous driving densities per axis: m²/s³ for CV, m²/s⁵ for CA.
    order : tuple of str or None
        Exact permutation of required coordinate names; None uses axis interleaving.
    frame : str
        ENU frame identifier.

    References
    ----------
    .. [1] StoneSoup combined transition models and FilterPy Q_continuous_white_noise.
    .. [2] Bar-Shalom et al., 2001, integrated continuous white driving noise.
    """
    ###################################################################################################
    # This function sets up selected east/north/up motion models and locates their values by name.
    #
    # Inputs:
    # - axes (mapping of str to str): x/y/z selects east/north/up; CV keeps velocity, CA keeps
    #   acceleration constant.
    # - origin_lla_deg_m (tuple of three floats): Shared origin: latitude/longitude in degrees,
    #   height in metres.
    # - noise_density (float or mapping of str to float): Motion uncertainty added per axis; m²/s³
    #   for CV, m²/s⁵ for CA.
    # - order (tuple of str, or None): Tracked quantity order; None groups quantities by axis.
    # - frame (str): Name identifying the shared coordinate system.
    #
    # Outputs:
    # - None; stores the coordinate definition and motion settings for each selected axis.
    def __init__(
        self,
        axes: Mapping[str, str],
        *,
        origin_lla_deg_m: tuple[float, float, float],
        noise_density: float | Mapping[str, float] = 1.0,
        order: tuple[str, ...] | None = None,
        frame: str = "ENU",
    ) -> None:

        if not axes or set(axes) - set("xyz") or any(v not in ("CV", "CA") for v in axes.values()):
            raise ValueError("axes must map a nonempty subset of x/y/z to CV or CA")
        
        # Per-axis densities are useful when the motion is expected to be less predictable in one direction
        # than another; however, every configured axis must be specified
        if isinstance(noise_density, Mapping) and set(noise_density) != set(axes):
            raise ValueError("noise densities must specify exactly the selected axes")

        # Each block records the coordinates belonging to one independent kinematic axis (e.g. x, xdot, xddot)
        # Use names rather than hard-coded indices so users may choose any ordering without changing the model
        self._blocks: list[tuple[tuple[str, ...], float]] = [] 
        coordinates: list[Coordinate] = []

        # Each independent axis can have a different derivative order and density.
        for axis in "xyz":
            if axis not in axes:
                continue
            
            # CV state: [3D positions, 3D velocities]
            names: tuple[str, ...] = (f"{axis}_m", f"{axis}dot_mps")
            units: tuple[str, ...] = ("m", "m/s")

            # CA state adds acceleration: [3D positions, 3D velocities, 3D acceleration]
            if axes[axis] == "CA":
                names += (f"{axis}ddot_mps2",)
                units += ("m/s^2",)
            
            density = float(
                noise_density[axis] if isinstance(noise_density, Mapping) else noise_density
            )
            if not np.isfinite(density) or density < 0:
                raise ValueError("noise_density must be finite and nonnegative")
            
            # Store the kinematic structure and process-noise density
            self._blocks.append((names, density))

            # Add coordinates to the overall StateSpace
            coordinates.extend(
                Coordinate(name, unit) for name, unit in zip(names, units, strict=True)
            )
        
        # Name lookup used when the caller supplies a custom state ordering
        lookup = {c.name: c for c in coordinates}
        if order is not None:
            if len(order) != len(lookup) or set(order) != set(lookup):
                raise ValueError("order must be an exact permutation of required coordinates")
            coordinates = [lookup[name] for name in order]
        self.state_space = StateSpace(tuple(coordinates), frame, origin_lla_deg_m)
    ###################################################################################################

    ###################################################################################################
    # This function builds the tables the filter needs to predict movement and added uncertainty.
    #
    # Inputs:
    # - dt_s (float): Time to advance, in seconds; must be zero or positive.
    #
    # Outputs:
    # - tuple of two float64 NumPy arrays: Movement table and added-uncertainty table, each (n,
    #   n), for the stored coordinates.
    def matrices(self, dt_s: float) -> tuple[FloatArray, FloatArray]:
        """Return transition and noise matrices (n,n) for dt_s seconds.

        Notes
        -----
        Independent driving noise gives block-diagonal Q, but F P F.T preserves
        input cross-axis covariance. Density units are specified in the constructor.

        References
        ----------
        .. [1] FilterPy Q_continuous_white_noise; Bar-Shalom et al., 2001.
        """
        _elapsed(dt_s)

        # State transition matrix F
        transition = np.eye(self.state_space.dimension, dtype=np.float64)

        # Process noise covariance Q
        noise = np.zeros_like(transition)

        # Build one independent kinematic block per configured axis
        for names, density in self._blocks:

            indices = self.state_space.indices(names)

            # CV => 2 states (x, xdot), CA => 3 states (x, xdot, xddot)
            n_order = len(indices)

            # Iterate over the elements of the transition and covariance matrices
            for i, row in enumerate(indices):
                for j, col in enumerate(indices):

                    # Populate the motion model transition terms (upper triangular).
                    # This is the Taylor-series state transition:
                    #
                    # For CV [x, v]:
                    #     x' = x + v*dt
                    #     v' = v
                    #
                    # For CA [x, v, a]:
                    #     x' = x + v*dt + 1/2*a*dt²
                    #     v' = v + a*dt
                    #     a' = a
                    if j >= i:
                        transition[row, col] = dt_s ** (j - i) / math.factorial(j - i)

                    # Populate the continuous-white-noise covariance terms.
                    # Integrating continuous white driving noise over dt produces Q.
                    # The compact factorial expression works for both:
                    #
                    # CV, driven by white acceleration:
                    #   q * [[dt³/3, dt²/2],
                    #        [dt²/2, dt]]
                    #
                    # CA, driven by white jerk:
                    #   q * [[dt⁵/20, dt⁴/8, dt³/6],
                    #        [dt⁴/8,  dt³/3, dt²/2],
                    #        [dt³/6,  dt²/2, dt]]
                    a = n_order - 1 - i
                    b = n_order - 1 - j
                    noise[row, col] = (
                        density
                        * dt_s ** (a + b + 1)
                        / ((a + b + 1) * math.factorial(a) * math.factorial(b))
                    )
        return transition, noise
    ###################################################################################################

    ###################################################################################################
    # This function predicts positions and velocities, including acceleration where configured.
    #
    # Inputs:
    # - state (float64 NumPy array): Tracked values, shape (n,), one per named quantity, in its
    #   stated unit.
    # - dt_s (float): Time to advance, in seconds; must be zero or positive.
    #
    # Outputs:
    # - float64 NumPy array: Predicted tracked values, shape (n,), with the input unchanged.
    def transition(self, state: FloatArray, dt_s: float) -> FloatArray:
        """Propagate (n,) state; see CartesianMotion References."""
        # x(k+1) = F x(k)
        return self.matrices(dt_s)[0] @ vector(state, self.state_space.dimension)
    ###################################################################################################

    ###################################################################################################
    # This function calculates the uncertainty to add while predicting movement on the selected axes.
    #
    # Inputs:
    # - state (float64 NumPy array): Tracked values, shape (n,), one per named quantity, in its
    #   stated unit.
    # - dt_s (float): Time to advance, in seconds; must be zero or positive.
    #
    # Outputs:
    # - float64 NumPy array: Added-uncertainty table, shape (n, n), in paired coordinate units.
    def process_noise(self, state: FloatArray, dt_s: float) -> FloatArray:
        """Return continuous-noise covariance (n,n); see CartesianMotion References."""
        # Validate state size even though it is not otherwise used (we do this to keep the 
        # MotionModel interface consistent)
        vector(state, self.state_space.dimension)

        # Return Q only
        return self.matrices(dt_s)[1]
    ###################################################################################################

class RadialMotion:
    """Constant closing velocity with continuous white radial acceleration.

    Parameters
    ----------
    acceleration_noise_density_m2ps3 : float
        Nonnegative driving spectral density in m²/s³.

    Notes
    -----
    Range is a continuous internal coordinate even when measurements are folded.
    Positive closing velocity decreases range. This changes both F and Q signs.

    References
    ----------
    .. [1] Bar-Shalom et al., 2001, CV model with a velocity sign transformation.
    """
    ###################################################################################################
    # This function sets up tracking of distance and velocity toward the radar.
    #
    # Inputs:
    # - acceleration_noise_density_m2ps3 (float): Strength of unmodelled radial acceleration, in
    #   m²/s³.
    #
    # Outputs:
    # - None; stores the two-coordinate definition and motion-uncertainty setting.
    def __init__(self, acceleration_noise_density_m2ps3: float = 1.0) -> None:
        if (
            not np.isfinite(acceleration_noise_density_m2ps3)
            or acceleration_noise_density_m2ps3 < 0
        ):
            raise ValueError("acceleration_noise_density_m2ps3 must be finite and nonnegative")
        
        # Process-noise density for unmodelled radial acceleration
        self.density = acceleration_noise_density_m2ps3

        # State = [range, closing_velocity]
        self.state_space = StateSpace(
            (Coordinate("range_m", "m"), Coordinate("radial_velocity_mps", "m/s")), "radial"
        )
    ###################################################################################################

    ###################################################################################################
    # This function predicts distance by keeping radial velocity constant; positive velocity
    # brings the target closer.
    #
    # Inputs:
    # - state (float64 NumPy array): Distance in metres and closing velocity in m/s, shape (2,).
    # - dt_s (float): Time to advance, in seconds; must be zero or positive.
    #
    # Outputs:
    # - float64 NumPy array: Predicted distance and unchanged closing velocity, shape (2,).
    def transition(self, state: FloatArray, dt_s: float) -> FloatArray:
        """Propagate (2,) range/closing velocity; see RadialMotion References."""
        _elapsed(dt_s)

        result = vector(state, 2)

        # Positive closing velocity reduces range
        result[0] -= result[1] * dt_s

        return result
    ###################################################################################################

    ###################################################################################################
    # This function calculates uncertainty added to distance and closing velocity during
    # prediction.
    #
    # Inputs:
    # - state (float64 NumPy array): Distance and closing velocity, shape (2,); checked but not
    #   used in the calculation.
    # - dt_s (float): Time to advance, in seconds; must be zero or positive.
    #
    # Outputs:
    # - float64 NumPy array: Added-uncertainty table, shape (2, 2), with units matching distance
    #   and velocity.
    def process_noise(self, state: FloatArray, dt_s: float) -> FloatArray:
        """Return signed CV covariance (2,2); see RadialMotion References."""
        _elapsed(dt_s)
        vector(state, 2)

        # Continuous-white-acceleration noise model in radial coordinates; range / velocity cross terms are
        # negative because we have defined positive velocity as decreasing range
        return self.density * np.array(
            [
                [dt_s**3 / 3, -(dt_s**2) / 2], 
                [-(dt_s**2) / 2, dt_s]
            ], 
            dtype=np.float64
        )
    ###################################################################################################

class CoordinatedTurn:
    """Planar constant-turn model with explicit ENU origin and stable zero turn.

    Parameters
    ----------
    origin_lla_deg_m : tuple of float
        ENU origin in degrees/degrees/metres.
    acceleration_noise_density_m2ps3, turn_noise_density_rad2ps3 : float
        Driving densities for Cartesian acceleration and turn-rate diffusion.

    Notes
    -----
    State order is x_m, xdot_mps, y_m, ydot_mps, turn_rate_radps. Positive
    turn is mathematical counterclockwise in the east/north plane. Additive Q is
    an approximation. The angular rate itself is not a periodic coordinate.

    References
    ----------
    .. [1] Local coordinated-turn model; Bar-Shalom et al., 2001.
    """
    ###################################################################################################
    # This function sets up a horizontal turning model for east/north position and velocity.
    #
    # Inputs:
    # - origin_lla_deg_m (tuple of three floats): Shared origin: latitude/longitude in degrees,
    #   height in metres.
    # - acceleration_noise_density_m2ps3 (float): Strength of unmodelled acceleration, in m²/s³.
    # - turn_noise_density_rad2ps3 (float): Strength of unmodelled turn-rate changes, in rad²/s³.
    #
    # Outputs:
    # - None; stores the five-coordinate definition and movement-uncertainty settings.
    def __init__(
        self,
        *,
        origin_lla_deg_m: tuple[float, float, float],
        acceleration_noise_density_m2ps3: float = 1.0,
        turn_noise_density_rad2ps3: float = 0.01,
    ) -> None:

        # Reuse a 2D constant-velocity model for the translational component
        self._cv = CartesianMotion(
            {"x": "CV", "y": "CV"},
            origin_lla_deg_m=origin_lla_deg_m,
            noise_density=acceleration_noise_density_m2ps3,
        )

        if not np.isfinite(turn_noise_density_rad2ps3) or turn_noise_density_rad2ps3 < 0:
            raise ValueError("turn noise density must be finite and nonnegative")
        
        self._turn_noise = turn_noise_density_rad2ps3

        # Additional State = [east position, east velocity,
        #                    north position, north velocity,
        #                    angular turn rate].
        self.state_space = StateSpace(
            (*self._cv.state_space.coordinates, 
            Coordinate("turn_rate_radps", "rad/s")),
            "ENU",
            origin_lla_deg_m,
        )
    ###################################################################################################

    ###################################################################################################
    # This function predicts horizontal motion at a constant turn rate, including straight travel
    # when that rate is zero.
    #
    # Inputs:
    # - state (float64 NumPy array): East m, east m/s, north m, north m/s, and turn rate rad/s,
    #   shape (5,).
    # - dt_s (float): Time to advance, in seconds; must be zero or positive.
    #
    # Outputs:
    # - float64 NumPy array: Predicted position and velocity with unchanged turn rate, shape (5,).
    def transition(self, state: FloatArray, dt_s: float) -> FloatArray:
        """Propagate (5,) planar state; see CoordinatedTurn References."""
        
        _elapsed(dt_s)

        result = vector(state, 5)

        # Angle swept during this prediction interval
        angle = result[4] * dt_s

        sine, cosine = np.sin(angle), np.cos(angle)

        # Numerically stable CT coefficients (well-behaved near 0 turn rate)
        #
        # Explanation:
        # The exact CT position equations contain
        #
        #     sin(ωΔt) / ω
        #     (1 - cos(ωΔt)) / ω
        #
        # Direct evaluation becomes numerically awkward as ω -> 0 because both
        # appear to divide by zero. np.sinc gives the same quantities in forms
        # whose limits smoothly become Δt and 0 respectively.
        a = dt_s * np.sinc(angle / np.pi)
        b = dt_s * (angle / 2) * np.sinc(angle / (2 * np.pi)) ** 2

        east_mps = result[1]
        north_mps = result[3]

        # Advance position along the turning arc by the displacement over the interval
        result[0] += a * east_mps - b * north_mps
        result[2] += b * east_mps + a * north_mps

        # Rotate the velocity vector by the turn angle (+ve = counterclockwise in the East/North plane)
        result[1], result[3] = (
            cosine * east_mps - sine * north_mps,
            sine * east_mps + cosine * north_mps,
        )

        # turn_rate_radps itself remains unchanged (this is the determinstic CT assumption)
        # its uncertainty grows through Q below
        return result
    ###################################################################################################

    ###################################################################################################
    # This function estimates the uncertainty added while a target moves and turns.
    #
    # Inputs:
    # - state (float64 NumPy array): East m, east m/s, north m, north m/s, and turn rate rad/s,
    #   shape (5,).
    # - dt_s (float): Time to advance, in seconds; must be zero or positive.
    #
    # Outputs:
    # - float64 NumPy array: Approximate added-uncertainty table, shape (5, 5), in paired
    #   coordinate units.
    def process_noise(self, state: FloatArray, dt_s: float) -> FloatArray:
        """Return approximate additive covariance (5,5); see CoordinatedTurn References."""
        _elapsed(dt_s)
        vector(state, 5)

        result = np.zeros((5, 5), dtype=np.float64)

        # CV process noise for translational x/y (East/North) position and velocity
        result[:4, :4] = self._cv.process_noise(state[:4], dt_s)

        # Turn rate is modeled as a random-walk driven by continuous white noise
        result[4, 4] = self._turn_noise * dt_s

        return result
    ###################################################################################################