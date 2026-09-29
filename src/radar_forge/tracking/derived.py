"""Derived ENU kinematics without introducing redundant filter state variables.

Classes:
- DerivedValue: This object represents a derived scalar value or None, along with a physical unit. 
                If unavailable, it includes a reason for the unavailability.
                The method derive_kinematics is used to derive these outputs that are not part of the filter state 
                (e.g., range, azimuth, elevation, radial velocity, speed, acceleration).

References
----------
.. [1] Radar-Forge ENU geodesy and positive-closing Doppler conventions.
.. [2] Richards, Fundamentals of Radar Signal Processing, 2nd ed., 2014.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from radar_forge.tracking.measurements import SensorPose
from radar_forge.tracking.spaces import StateEstimate

__all__ = [
    "DerivedValue",
    "derive_kinematics",
]


@dataclass(frozen=True)
class DerivedValue:
    """Scalar value or None, physical unit, and an unavailable reason.

    References
    ----------
    .. [1] Tracker 001 derived-output contract; unavailable is distinct from zero.
    """

    value: float | None
    unit: str
    reason: str | None = None

###################################################################################################
# This function calculates readable distances, angles, speeds, and accelerations from a track
# without changing its estimate.
#
# Inputs:
# - estimate (StateEstimate): Named east/north/up values and any available velocities or
#   accelerations.
# - sensor (SensorPose): Reference sensor in the same coordinate system and origin.
# - fixed_coordinates (mapping of str to float, or None): Explicit values for missing coordinates;
#   None supplies none.
#
# Outputs:
# - dict of str to DerivedValue: Named values with units, or None with a reason when data is
#   missing or the calculation is undefined.
def derive_kinematics(
    estimate: StateEstimate,
    *,
    sensor: SensorPose,
    fixed_coordinates: Mapping[str, float] | None = None,
) -> dict[str, DerivedValue]:
    """Derive geometry, speed and acceleration from an ENU estimate.

    Parameters
    ----------
    estimate : StateEstimate
        Mean (n,) in its declared coordinate order.
    sensor : SensorPose
        Static sensor position in the same ENU frame and origin.
    fixed_coordinates : mapping or None
        Explicit SI values for absent Cartesian coordinates. Present state values
        cannot be overridden. CV acceleration is unavailable unless supplied here.

    Returns
    -------
    dict of str to DerivedValue
        Named scalar outputs with units and unavailable reasons. No uncertainty
        propagation is claimed by this scalar presentation interface.

    References
    ----------
    .. [1] Radar-Forge ENU conventions; Richards, 2014, monostatic radial Doppler.
    """
    space = estimate.state_space
    if space.frame != sensor.frame or space.origin_lla_deg_m != sensor.origin_lla_deg_m:
        raise ValueError("derived geometry requires matching frame and ENU origin")
    fixed = dict(fixed_coordinates or {})
    allowed = {f"{a}{suffix}" for a in "xyz" for suffix in ("_m", "dot_mps", "ddot_mps2")}
    if (
        set(fixed) - allowed
        or set(fixed) & set(space.names)
        or not all(np.isfinite(v) for v in fixed.values())
    ):
        raise ValueError("fixed coordinates must be finite absent Cartesian coordinates")
    values = dict(zip(space.names, estimate.mean, strict=True))
    values.update(fixed)
    result: dict[str, DerivedValue] = {}


    ###################################################################################################
    # This helper function combines stored components into a total speed or acceleration for the
    # surrounding output calculation.
    #
    # Inputs:
    # - output (str): Name under which to store the result.
    # - names (tuple of str): Component names to read from the surrounding function's values.
    # - unit (str): Unit to attach, such as m/s or m/s².
    #
    # Outputs:
    # - None; adds a DerivedValue to the surrounding result dictionary, or records which
    #   components are missing.
    def magnitude(output: str, names: tuple[str, ...], unit: str) -> None:
        missing = set(names) - values.keys()
        result[output] = (
            DerivedValue(None, unit, "missing: " + ", ".join(sorted(missing)))
            if missing
            else DerivedValue(float(np.linalg.norm([values[n] for n in names])), unit)
        )
    ###################################################################################################

    # Call helper function to find the magnitude of each vector quantity, or record if components are missing.
    magnitude("abs_speed_mps", tuple(f"{a}dot_mps" for a in "xyz"), "m/s")
    magnitude("acceleration_mps2", tuple(f"{a}ddot_mps2" for a in "xyz"), "m/s^2")
    magnitude("accelxy_mps2", ("xddot_mps2", "yddot_mps2"), "m/s^2")
    result["accelz_mps2"] = (
        DerivedValue(float(values["zddot_mps2"]), "m/s^2")
        if "zddot_mps2" in values
        else DerivedValue(None, "m/s^2", "missing: zddot_mps2")
    )
    horizontal_mps = None
    if "xdot_mps" in values and "ydot_mps" in values:
        horizontal_mps = float(np.hypot(values["xdot_mps"], values["ydot_mps"]))
    # Output availability differs by coordinate dependency.
    for axis in "xy":
        result[f"{axis}dot_norm"] = (
            DerivedValue(float(values[f"{axis}dot_mps"]) / horizontal_mps, "1")
            if horizontal_mps
            else DerivedValue(None, "1", "missing or zero horizontal velocity")
        )
    missing_position = any(f"{a}_m" not in values for a in "xyz")
    for key, unit in (
        ("range_m", "m"),
        ("azimuth_rad", "rad"),
        ("elevation_rad", "rad"),
        ("radial_velocity_mps", "m/s"),
    ):
        result[key] = DerivedValue(
            None,
            unit,
            "missing position" if missing_position else "singular geometry or missing velocity",
        )
    if not missing_position:
        delta = np.array([values[f"{a}_m"] for a in "xyz"], dtype=np.float64) - sensor.position_m
        distance_m = float(np.linalg.norm(delta))
        horizontal_m = float(np.hypot(delta[0], delta[1]))
        result["range_m"] = DerivedValue(distance_m, "m")
        if horizontal_m > 0:
            result["azimuth_rad"] = DerivedValue(float(np.arctan2(delta[0], delta[1])), "rad")
        if distance_m > 0:
            result["elevation_rad"] = DerivedValue(float(np.arctan2(delta[2], horizontal_m)), "rad")
            if all(f"{a}dot_mps" in values for a in "xyz"):
                speed = np.asarray([values[f"{a}dot_mps"] for a in "xyz"], dtype=np.float64)
                result["radial_velocity_mps"] = DerivedValue(
                    -float(delta @ speed) / distance_m, "m/s"
                )
    return result
###################################################################################################
