r"""Coordinate conversions: WGS-84 geodetic, ECEF, local east-north-up, range and angles.

A position is geodetic (latitude, longitude in degrees, height in metres above the WGS-84
ellipsoid), Earth-centred Earth-fixed (ECEF) metres, or metres east, north and up (ENU) of a
geodetic reference. From ENU, a direction is a range, an azimuth clockwise from true north
in :math:`[0, 2\pi)` and an elevation above the local horizontal (``spec/data-001-formats.md``
§5). Each function names its pymap3d equivalent, which takes degrees unless ``deg=False``.
:class:`radar_forge.radar.Site` calls these for its own position.

References
----------
.. [1] National Imagery and Mapping Agency, *Department of Defense World Geodetic
       System 1984*, NIMA TR8350.2, 3rd ed., Amendment 1, 2000.
.. [2] P. Misra and P. Enge, *Global Positioning System: Signals, Measurements,
       and Performance*, 2nd ed., Ganga-Jamuna Press, 2006.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge._periodic import fold
from radar_forge.constants import WGS84_FLATTENING, WGS84_SEMI_MAJOR_AXIS_M

__all__ = [
    "directions_unit",
    "east_north_to_range_azimuth_rad",
    "ecef_to_enu_m",
    "enu_to_range_azimuth_elevation_rad",
    "geodetic_to_ecef_m",
    "geodetic_to_enu_m",
    "range_azimuth_elevation_to_enu_m",
    "range_azimuth_to_east_north_m",
]

# First eccentricity squared, e^2 = f(2 - f), derived rather than stored.
_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)


def geodetic_to_ecef_m(
    latitude_deg: ArrayLike, longitude_deg: ArrayLike, altitude_m: ArrayLike
) -> NDArray[np.float64]:
    """Convert WGS-84 geodetic coordinates to ECEF metres ``(..., 3)`` (NIMA [1] eq. 4-14).

    pymap3d ``geodetic2ecef``.
    """
    latitude_arr = np.asarray(latitude_deg, dtype=np.float64)
    if np.any(np.abs(latitude_arr) > 90.0):
        msg = "latitude_deg must lie in [-90, 90]; values outside it are not a latitude."
        raise ValueError(msg)
    altitude_arr = np.asarray(altitude_m, dtype=np.float64)
    latitude_rad = np.radians(latitude_arr)
    longitude_rad = np.radians(np.asarray(longitude_deg, dtype=np.float64))
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
    """Translate ECEF metres to a geodetic reference, rotate into its ENU frame (Misra [2] 4.A).

    pymap3d ``ecef2enu``.
    """
    ecef_arr = np.asarray(ecef_m, dtype=np.float64)
    if ecef_arr.shape[-1] != 3:
        msg = f"ecef_m must have shape (..., 3) for (x, y, z); got {ecef_arr.shape}."
        raise ValueError(msg)
    offset_m = ecef_arr - geodetic_to_ecef_m(
        reference_latitude_deg, reference_longitude_deg, reference_altitude_m
    )
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
    """Convert geodetic points to east, north, up metres ``(..., 3)`` about a geodetic reference.

    pymap3d ``geodetic2enu``.
    """
    ecef_m = geodetic_to_ecef_m(latitude_deg, longitude_deg, altitude_m)
    return ecef_to_enu_m(
        ecef_m, reference_latitude_deg, reference_longitude_deg, reference_altitude_m
    )


def enu_to_range_azimuth_elevation_rad(
    enu_m: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert ENU metres ``(..., 3)`` to range, azimuth and elevation (Misra [2] app. 4.A).

    Returns range in m, azimuth ``atan2(east, north)`` in rad in [0, 2 pi), zero at true north
    and clockwise, and elevation ``asin(up / range)`` in rad. pymap3d ``enu2aer`` returns
    (az, el, range); this returns range first.
    """
    enu_arr = np.asarray(enu_m, dtype=np.float64)
    if enu_arr.shape[-1] != 3:
        msg = f"enu_m must have shape (..., 3) for (east, north, up); got {enu_arr.shape}."
        raise ValueError(msg)
    east_m = enu_arr[..., 0]
    north_m = enu_arr[..., 1]
    up_m = enu_arr[..., 2]
    range_m = np.sqrt(east_m**2 + north_m**2 + up_m**2)
    azimuth_rad = fold(np.arctan2(east_m, north_m), 2.0 * np.pi, lower=0.0)
    elevation_rad = np.arcsin(np.divide(up_m, range_m))
    return range_m, azimuth_rad, elevation_rad


def range_azimuth_elevation_to_enu_m(
    range_m: ArrayLike, azimuth_rad: ArrayLike, elevation_rad: ArrayLike
) -> NDArray[np.float64]:
    """Convert slant range, azimuth and elevation to east, north, up metres, ``(..., 3)``.

    pymap3d ``aer2enu``, which takes (az, el, range).
    """
    range_arr = np.asarray(range_m, dtype=np.float64)
    azimuth_arr = np.asarray(azimuth_rad, dtype=np.float64)
    elevation_arr = np.asarray(elevation_rad, dtype=np.float64)
    ground_range_m = range_arr * np.cos(elevation_arr)
    east_m = ground_range_m * np.sin(azimuth_arr)
    north_m = ground_range_m * np.cos(azimuth_arr)
    up_m = range_arr * np.sin(elevation_arr)
    result: NDArray[np.float64] = np.stack(np.broadcast_arrays(east_m, north_m, up_m), axis=-1)
    return result


def east_north_to_range_azimuth_rad(
    east_north_m: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the horizontal range, m, and azimuth of ground-plane positions ``(..., 2)``.

    :func:`enu_to_range_azimuth_elevation_rad` with no height: azimuth clockwise from north in
    [0, 2 pi). pymap3d ``enu2aer`` with ``u = 0``.
    """
    ground = np.asarray(east_north_m, dtype=np.float64)
    enu = np.concatenate([ground, np.zeros_like(ground[..., :1])], axis=-1)
    range_m, azimuth_rad, _ = enu_to_range_azimuth_elevation_rad(enu)
    return range_m, azimuth_rad


def range_azimuth_to_east_north_m(
    range_m: ArrayLike, azimuth_rad: ArrayLike
) -> NDArray[np.float64]:
    """Return the ground-plane east, north position ``(..., 2)`` of a horizontal range and azimuth.

    pymap3d ``aer2enu`` with ``el = 0``.
    """
    result: NDArray[np.float64] = range_azimuth_elevation_to_enu_m(range_m, azimuth_rad, 0.0)
    return result[..., :2]


def directions_unit(azimuth_rad: ArrayLike, elevation_rad: ArrayLike) -> NDArray[np.float64]:
    """Return unit vectors ``[sin az cos el, cos az cos el, sin el]``, ``(n_directions, 3)``.

    In the site's ENU frame for a compass azimuth; in the array frame for one from boresight.
    pymap3d ``aer2enu`` at unit range.
    """
    azimuth, elevation = np.broadcast_arrays(
        np.atleast_1d(np.asarray(azimuth_rad, dtype=np.float64)),
        np.asarray(elevation_rad, dtype=np.float64),
    )
    result: NDArray[np.float64] = np.stack(
        [
            np.sin(azimuth) * np.cos(elevation),
            np.cos(azimuth) * np.cos(elevation),
            np.sin(elevation),
        ],
        axis=-1,
    )
    return result


def _enu_axes_in_ecef(latitude_deg: float, longitude_deg: float) -> NDArray[np.float64]:
    """Return the east, north and up axes at a geodetic point in ECEF, as rows.

    The rotation :func:`ecef_to_enu_m` applies.
    """
    sin_lat, cos_lat = np.sin(np.radians(latitude_deg)), np.cos(np.radians(latitude_deg))
    sin_lon, cos_lon = (
        np.sin(np.radians(longitude_deg)),
        np.cos(np.radians(longitude_deg)),
    )
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )
