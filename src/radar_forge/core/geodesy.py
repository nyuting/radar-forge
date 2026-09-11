"""Geodetic, ECEF and local-tangent-plane coordinate transforms.

Radar scenarios are specified in the coordinates the world uses — latitude,
longitude, altitude — but every geometric calculation in a radar (range, angle,
Doppler) wants a local Cartesian frame with the radar at the origin. This module
is the bridge, and it is the only place in the library that knows the Earth is
not flat.

Three frames are used, in the order the conversions chain:

geodetic
    WGS-84 latitude, longitude and altitude above the reference ellipsoid.
ECEF
    Earth-Centred, Earth-Fixed Cartesian metres: ``x`` through the intersection
    of the equator and the prime meridian, ``z`` through the north pole,
    ``y`` completing a right-handed set.
ENU
    The local tangent plane at a reference point: ``x`` east, ``y`` north,
    ``z`` up. This is the frame radar geometry is done in.

Angle convention, matching the label schema in ``spec/structure.md`` D5:
**azimuth is zero at true north and increases clockwise** (north → east), which
is the compass convention, *not* the mathematical one. Elevation is measured up
from the local horizontal plane.

Notes
-----
Everything here is geometric. There is no atmospheric refraction, so elevation
angles are the true geometric ones rather than the apparent ones a radar
measures. For a target at 1.5 km altitude and 18 km range the difference is a
few millidegrees, but it grows at low elevation angles and long ranges; see
:data:`radar_forge.core.constants.FOUR_THIRDS_EARTH_RADIUS_M` for the standard
correction when that matters.

References
----------
.. [1] National Imagery and Mapping Agency, *Department of Defense World
       Geodetic System 1984*, NIMA TR8350.2, 3rd ed., 2000, §4.
.. [2] P. Misra and P. Enge, *Global Positioning System: Signals, Measurements,
       and Performance*, 2nd ed., Ganga-Jamuna Press, 2006, §4.A (geodetic to
       ECEF, and the Bowring iteration for the inverse).
.. [3] R. G. Brown and P. Y. C. Hwang, *Introduction to Random Signals and
       Applied Kalman Filtering*, 4th ed., Wiley, 2012, app. B (ENU rotation).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.core.constants import WGS84_FLATTENING, WGS84_SEMI_MAJOR_AXIS_M

__all__ = [
    "ecef_to_enu_m",
    "enu_to_range_azimuth_elevation",
    "geodetic_to_ecef_m",
    "geodetic_to_enu_m",
]

# First eccentricity squared, e^2 = f(2 - f). Derived from the two defining
# constants rather than stored as a third one that could drift out of step.
_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)


def geodetic_to_ecef_m(
    latitude_deg: ArrayLike,
    longitude_deg: ArrayLike,
    altitude_m: ArrayLike,
) -> NDArray[np.float64]:
    r"""Convert WGS-84 geodetic coordinates to ECEF Cartesian metres.

    Implements the closed-form forward transform [1]_, [2]_

    .. math::

        N(\phi) &= \frac{a}{\sqrt{1 - e^2 \sin^2\phi}} \\
        x &= (N + h)\cos\phi\cos\lambda \\
        y &= (N + h)\cos\phi\sin\lambda \\
        z &= \left(N(1 - e^2) + h\right)\sin\phi

    where :math:`N` is the radius of curvature in the prime vertical. The
    forward direction has no iteration; only the inverse does.

    Parameters
    ----------
    latitude_deg : array_like
        Geodetic latitude :math:`\phi`, degrees, positive north, in [-90, 90].
    longitude_deg : array_like
        Longitude :math:`\lambda`, degrees, positive east.
    altitude_m : array_like
        Height :math:`h` above the WGS-84 ellipsoid, metres. Note this is
        ellipsoidal height, not height above mean sea level; the two differ by
        the geoid undulation, which is about 7 m in Singapore.

    Returns
    -------
    numpy.ndarray
        ECEF coordinates in metres, shape ``(..., 3)``, the last axis being
        ``(x, y, z)``. The leading axes are the broadcast shape of the inputs,
        so scalar inputs give shape ``(3,)``.

    Raises
    ------
    ValueError
        If any latitude lies outside [-90, 90] degrees.

    See Also
    --------
    ecef_to_enu_m : Rotate the result into a local tangent plane.
    geodetic_to_enu_m : The two steps combined.

    Examples
    --------
    >>> import numpy as np
    >>> ecef = geodetic_to_ecef_m(0.0, 0.0, 0.0)
    >>> bool(np.isclose(ecef[0], 6_378_137.0))  # equator, prime meridian
    True
    >>> bool(np.allclose(ecef[1:], 0.0))
    True
    """
    latitude_arr = np.asarray(latitude_deg, dtype=np.float64)
    longitude_arr = np.asarray(longitude_deg, dtype=np.float64)
    altitude_arr = np.asarray(altitude_m, dtype=np.float64)

    if np.any(np.abs(latitude_arr) > 90.0):
        msg = "latitude_deg must lie in [-90, 90]; values outside it are not a latitude."
        raise ValueError(msg)

    latitude_rad = np.radians(latitude_arr)
    longitude_rad = np.radians(longitude_arr)
    sin_latitude = np.sin(latitude_rad)
    cos_latitude = np.cos(latitude_rad)

    prime_vertical_radius_m = WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(
        1.0 - _ECCENTRICITY_SQUARED * sin_latitude**2
    )

    x_m = (prime_vertical_radius_m + altitude_arr) * cos_latitude * np.cos(longitude_rad)
    y_m = (prime_vertical_radius_m + altitude_arr) * cos_latitude * np.sin(longitude_rad)
    z_m = (prime_vertical_radius_m * (1.0 - _ECCENTRICITY_SQUARED) + altitude_arr) * sin_latitude

    result: NDArray[np.float64] = np.stack(np.broadcast_arrays(x_m, y_m, z_m), axis=-1)
    return result


def ecef_to_enu_m(
    ecef_m: ArrayLike,
    reference_latitude_deg: ArrayLike,
    reference_longitude_deg: ArrayLike,
    reference_altitude_m: ArrayLike,
) -> NDArray[np.float64]:
    r"""Rotate ECEF coordinates into the local ENU frame at a reference point.

    The transform is a translation to the reference point followed by the
    rotation [3]_

    .. math::

        \begin{bmatrix} e \\ n \\ u \end{bmatrix} =
        \begin{bmatrix}
            -\sin\lambda_0 & \cos\lambda_0 & 0 \\
            -\sin\phi_0\cos\lambda_0 & -\sin\phi_0\sin\lambda_0 & \cos\phi_0 \\
             \cos\phi_0\cos\lambda_0 &  \cos\phi_0\sin\lambda_0 & \sin\phi_0
        \end{bmatrix}
        \begin{bmatrix} \Delta x \\ \Delta y \\ \Delta z \end{bmatrix}

    Being a rotation, it preserves distances exactly: the ENU vector norm is the
    straight-line (chord) distance between the two points, which is what a radar
    measures as slant range.

    Parameters
    ----------
    ecef_m : array_like
        ECEF coordinates in metres, shape ``(..., 3)``.
    reference_latitude_deg, reference_longitude_deg : array_like
        Geodetic latitude and longitude of the tangent-plane origin, degrees.
    reference_altitude_m : array_like
        Ellipsoidal height of the origin, metres.

    Returns
    -------
    numpy.ndarray
        Local east, north, up in metres, shape ``(..., 3)``.

    Raises
    ------
    ValueError
        If ``ecef_m`` does not have length 3 on its last axis.

    Notes
    -----
    The tangent plane is a *local* approximation: "up" is the ellipsoid normal
    at the reference point only. Over the ~20 km extents of a typical ground
    radar scenario the Earth curves away by about 30 m, which is well inside one
    range bin here but is not negligible for every scenario.
    """
    ecef_arr = np.asarray(ecef_m, dtype=np.float64)
    if ecef_arr.shape[-1] != 3:
        msg = f"ecef_m must have shape (..., 3) for (x, y, z); got {ecef_arr.shape}."
        raise ValueError(msg)

    origin_ecef_m = geodetic_to_ecef_m(
        reference_latitude_deg, reference_longitude_deg, reference_altitude_m
    )
    offset_m = ecef_arr - origin_ecef_m

    latitude_rad = np.radians(np.asarray(reference_latitude_deg, dtype=np.float64))
    longitude_rad = np.radians(np.asarray(reference_longitude_deg, dtype=np.float64))
    sin_latitude = np.sin(latitude_rad)
    cos_latitude = np.cos(latitude_rad)
    sin_longitude = np.sin(longitude_rad)
    cos_longitude = np.cos(longitude_rad)

    delta_x_m = offset_m[..., 0]
    delta_y_m = offset_m[..., 1]
    delta_z_m = offset_m[..., 2]

    east_m = -sin_longitude * delta_x_m + cos_longitude * delta_y_m
    north_m = (
        -sin_latitude * cos_longitude * delta_x_m
        - sin_latitude * sin_longitude * delta_y_m
        + cos_latitude * delta_z_m
    )
    up_m = (
        cos_latitude * cos_longitude * delta_x_m
        + cos_latitude * sin_longitude * delta_y_m
        + sin_latitude * delta_z_m
    )

    result: NDArray[np.float64] = np.stack(np.broadcast_arrays(east_m, north_m, up_m), axis=-1)
    return result


def geodetic_to_enu_m(
    latitude_deg: ArrayLike,
    longitude_deg: ArrayLike,
    altitude_m: ArrayLike,
    reference_latitude_deg: ArrayLike,
    reference_longitude_deg: ArrayLike,
    reference_altitude_m: ArrayLike,
) -> NDArray[np.float64]:
    """Convert geodetic coordinates straight to local ENU metres.

    A convenience composition of :func:`geodetic_to_ecef_m` and
    :func:`ecef_to_enu_m`; this is the call a scenario actually makes, turning a
    row of ``(timestamp, lat, lon)`` into a position relative to the radar.

    Parameters
    ----------
    latitude_deg, longitude_deg, altitude_m : array_like
        Target geodetic position: degrees, degrees, and metres above the
        ellipsoid.
    reference_latitude_deg, reference_longitude_deg, reference_altitude_m : array_like
        Geodetic position of the ENU origin — for a ground radar, the antenna.

    Returns
    -------
    numpy.ndarray
        Local east, north, up in metres, shape ``(..., 3)``.

    See Also
    --------
    enu_to_range_azimuth_elevation : Turn the result into radar observables.

    Examples
    --------
    >>> import numpy as np
    >>> enu = geodetic_to_enu_m(1.29, 103.78, 0.0, 1.29, 103.78, 0.0)
    >>> bool(np.allclose(enu, 0.0, atol=1e-9))  # the origin maps to itself
    True
    """
    ecef_m = geodetic_to_ecef_m(latitude_deg, longitude_deg, altitude_m)
    return ecef_to_enu_m(
        ecef_m, reference_latitude_deg, reference_longitude_deg, reference_altitude_m
    )


def enu_to_range_azimuth_elevation(
    enu_m: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Convert local ENU metres to the spherical coordinates a radar measures.

    .. math::

        R &= \sqrt{e^2 + n^2 + u^2} \\
        \mathrm{az} &= \operatorname{atan2}(e, n) \\
        \mathrm{el} &= \arcsin(u / R)

    Note the argument order of :func:`numpy.arctan2`: ``atan2(east, north)``,
    not the usual ``atan2(y, x)``. That is what puts zero azimuth at **true
    north** and makes it increase **clockwise** towards the east, which is the
    compass convention used throughout the library and by the label schema in
    ``spec/structure.md`` D5.

    Parameters
    ----------
    enu_m : array_like
        Local east, north, up in metres, shape ``(..., 3)``.

    Returns
    -------
    range_m : numpy.ndarray
        Slant range in metres, shape ``(...,)``. This is the straight-line
        distance, so it is the quantity that sets the two-way delay.
    azimuth_deg : numpy.ndarray
        Azimuth in degrees, wrapped to [0, 360), zero at north, clockwise.
    elevation_deg : numpy.ndarray
        Elevation in degrees above the local horizontal, in [-90, 90].

    Raises
    ------
    ValueError
        If ``enu_m`` does not have length 3 on its last axis.

    Notes
    -----
    Azimuth is ill-defined for a target directly overhead and both angles are
    ill-defined at zero range. Neither is guarded here: a radar target at zero
    range is a modelling error to be caught where the geometry is built, not a
    value to be silently substituted.
    """
    enu_arr = np.asarray(enu_m, dtype=np.float64)
    if enu_arr.shape[-1] != 3:
        msg = f"enu_m must have shape (..., 3) for (east, north, up); got {enu_arr.shape}."
        raise ValueError(msg)

    east_m = enu_arr[..., 0]
    north_m = enu_arr[..., 1]
    up_m = enu_arr[..., 2]

    range_m = np.sqrt(east_m**2 + north_m**2 + up_m**2)
    # atan2(east, north), not atan2(north, east): zero at north, clockwise.
    azimuth_deg = np.degrees(np.arctan2(east_m, north_m)) % 360.0
    elevation_deg = np.degrees(np.arcsin(np.divide(up_m, range_m)))

    return range_m, azimuth_deg, elevation_deg
