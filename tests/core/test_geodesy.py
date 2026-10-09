"""Tests for the geodetic, ECEF and ENU coordinate transforms.

Ground truth here is analytic wherever possible: the equatorial special cases
where the ellipsoid formulae collapse to exact expressions, the cardinal
azimuths, and the fact that an ENU transform is a rigid motion and so preserves
distance exactly. No recorded output is used.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core import geodesy
from radar_forge.core.constants import WGS84_FLATTENING, WGS84_SEMI_MAJOR_AXIS_M

# The scenario-001 radar site: the Duke Receiver, 101 Science Drive, Durham NC.
DUKE_LATITUDE_DEG = 36.00250
DUKE_LONGITUDE_DEG = -78.94100
DUKE_ALTITUDE_M = 60.0


class TestGeodeticToEcef:
    def test_equator_prime_meridian_is_the_semi_major_axis(self) -> None:
        """At phi = lambda = h = 0 the formulae collapse to x = a exactly."""
        ecef_m = geodesy.geodetic_to_ecef_m(0.0, 0.0, 0.0)
        np.testing.assert_allclose(ecef_m[0], WGS84_SEMI_MAJOR_AXIS_M, rtol=1e-15)
        np.testing.assert_allclose(ecef_m[1:], 0.0, atol=1e-9)

    def test_north_pole_is_the_semi_minor_axis(self) -> None:
        """At phi = 90 deg, z = a(1 - f) = b, the polar radius."""
        semi_minor_axis_m = WGS84_SEMI_MAJOR_AXIS_M * (1.0 - WGS84_FLATTENING)
        ecef_m = geodesy.geodetic_to_ecef_m(90.0, 0.0, 0.0)
        np.testing.assert_allclose(ecef_m[2], semi_minor_axis_m, rtol=1e-12)
        np.testing.assert_allclose(ecef_m[:2], 0.0, atol=1e-9)

    def test_rejects_a_latitude_south_of_the_pole(self) -> None:
        """A southern case catches a guard that forgot the absolute value."""
        with pytest.raises(ValueError, match=r"\[-90, 90\]"):
            geodesy.geodetic_to_ecef_m(-90.1, 0.0, 0.0)


class TestGeodeticToEnu:
    def test_altitude_above_the_origin_is_pure_up(self) -> None:
        enu_m = geodesy.geodetic_to_enu_m(
            DUKE_LATITUDE_DEG,
            DUKE_LONGITUDE_DEG,
            DUKE_ALTITUDE_M + 1500.0,
            DUKE_LATITUDE_DEG,
            DUKE_LONGITUDE_DEG,
            DUKE_ALTITUDE_M,
        )
        np.testing.assert_allclose(enu_m[2], 1500.0, rtol=1e-9)
        np.testing.assert_allclose(enu_m[:2], 0.0, atol=1e-6)

    def test_equatorial_east_step_and_the_earth_curving_away(self) -> None:
        """Catches east/north swapped, a flipped sign, or a wrong "up" (the Notes' ~30 m drop).

        At the equator the prime-vertical radius is exactly a and the origin's
        normal is the x axis, so a point 0.18 deg of longitude away -- 20 km, a
        scenario-sized extent -- sits at east = a sin(dlambda) and, because the
        tangent plane does not follow the Earth, at up = -a (1 - cos dlambda),
        about -31.5 m. Both are exact closed forms.
        """
        step_rad = np.radians(0.18)
        enu_m = geodesy.geodetic_to_enu_m(0.0, np.degrees(step_rad), 0.0, 0.0, 0.0, 0.0)
        # rtol 1e-12 on east: one sine and a subtraction of an exact zero.
        np.testing.assert_allclose(enu_m[0], WGS84_SEMI_MAJOR_AXIS_M * np.sin(step_rad), rtol=1e-12)
        np.testing.assert_allclose(enu_m[1], 0.0, atol=1e-9)
        # rtol 1e-9 on up: a cos(dlambda) - a cancels 6.4e6 m down to 31 m, so
        # about 1e-9 m of round-off on a 31 m answer.
        np.testing.assert_allclose(
            enu_m[2], -WGS84_SEMI_MAJOR_AXIS_M * (1.0 - np.cos(step_rad)), rtol=1e-9
        )

    def test_is_a_rigid_motion_so_distance_is_preserved(self) -> None:
        """ENU is a translation plus a rotation, so it preserves the chord exactly.

        This is the property the radar depends on: the ENU vector norm is the
        slant range that sets the two-way delay.
        """
        rng = np.random.default_rng(20260911)
        latitude_deg = DUKE_LATITUDE_DEG + rng.uniform(-0.1, 0.1, size=32)
        longitude_deg = DUKE_LONGITUDE_DEG + rng.uniform(-0.1, 0.1, size=32)
        altitude_m = rng.uniform(0.0, 3000.0, size=32)

        ecef_m = geodesy.geodetic_to_ecef_m(latitude_deg, longitude_deg, altitude_m)
        origin_ecef_m = geodesy.geodetic_to_ecef_m(
            DUKE_LATITUDE_DEG, DUKE_LONGITUDE_DEG, DUKE_ALTITUDE_M
        )
        ecef_distance_m = np.linalg.norm(ecef_m - origin_ecef_m, axis=-1)

        enu_m = geodesy.geodetic_to_enu_m(
            latitude_deg,
            longitude_deg,
            altitude_m,
            DUKE_LATITUDE_DEG,
            DUKE_LONGITUDE_DEG,
            DUKE_ALTITUDE_M,
        )
        enu_distance_m = np.linalg.norm(enu_m, axis=-1)
        # A rotation in float64: a handful of operations, so the tight tolerance.
        np.testing.assert_allclose(enu_distance_m, ecef_distance_m, rtol=1e-12)

    def test_rejects_a_last_axis_that_is_not_three(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
            geodesy.ecef_to_enu_m(np.zeros((4, 2)), 0.0, 0.0, 0.0)


class TestEnuToRangeAzimuthElevation:
    def test_azimuth_is_zero_at_north_and_increases_clockwise(self) -> None:
        """Catches the mathematical convention, a flipped sense, or no wrap to [0, 360).

        Those put north at 90, east at 270, and west at -90. The four cardinal
        points go in one (4, 3) call, so it also covers leading axes.
        """
        cardinal_enu_m = np.array(
            [[0.0, 1000.0, 0.0], [1000.0, 0.0, 0.0], [0.0, -1000.0, 0.0], [-1000.0, 0.0, 0.0]]
        )
        _, azimuth_deg, _ = geodesy.enu_to_range_azimuth_elevation(cardinal_enu_m)
        np.testing.assert_allclose(azimuth_deg, [0.0, 90.0, 180.0, 270.0], atol=1e-12)

    def test_a_pythagorean_point_gives_exact_range_and_angles(self) -> None:
        """Catches elevation taken from the zenith, a term missing from the norm, or radians.

        (300, 400, 1200) is a 3-4-5 triangle on a 5-12-13 one: range is 1300 m
        exactly, elevation asin(12/13) = 67.38 deg (a zenith angle would read
        22.62), azimuth atan2(3, 4) = 36.87 deg.
        """
        range_m, azimuth_deg, elevation_deg = geodesy.enu_to_range_azimuth_elevation(
            (300.0, 400.0, 1200.0)
        )
        # rtol 1e-12: a square root and one inverse trig function each.
        np.testing.assert_allclose(range_m, 1300.0, rtol=1e-12)
        np.testing.assert_allclose(elevation_deg, np.degrees(np.arcsin(12.0 / 13.0)), rtol=1e-12)
        np.testing.assert_allclose(azimuth_deg, np.degrees(np.arctan2(3.0, 4.0)), rtol=1e-12)

    def test_rejects_a_last_axis_that_is_not_three(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
            geodesy.enu_to_range_azimuth_elevation(np.zeros((4, 2)))
