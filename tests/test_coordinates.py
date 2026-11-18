"""Tests for the coordinate conversions: geodetic, ECEF, ENU, range and angles.

Ground truth: the exact equatorial, polar and cardinal cases of the WGS-84 transforms, a
Pythagorean point for the spherical conversion, and round trips.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.constants import WGS84_FLATTENING, WGS84_SEMI_MAJOR_AXIS_M
from radar_forge.coordinates import (
    east_north_to_range_azimuth_rad,
    ecef_to_enu_m,
    enu_axes_in_ecef,
    enu_to_range_azimuth_elevation_rad,
    geodetic_to_ecef_m,
    geodetic_to_enu_m,
    range_azimuth_elevation_to_enu_m,
    range_azimuth_to_east_north_m,
)

SEED = 20261106


class TestGeodeticToEcef:
    def test_equator_prime_meridian_is_the_semi_major_axis(self) -> None:
        ecef_m = geodetic_to_ecef_m(0.0, 0.0, 0.0)
        np.testing.assert_allclose(ecef_m[0], WGS84_SEMI_MAJOR_AXIS_M, rtol=1e-15)
        np.testing.assert_allclose(ecef_m[1:], 0.0, atol=1e-9)

    def test_north_pole_is_the_semi_minor_axis(self) -> None:
        """At phi = 90 deg, z = a (1 - f) = b, the polar radius."""
        ecef_m = geodetic_to_ecef_m(90.0, 0.0, 0.0)
        np.testing.assert_allclose(
            ecef_m[2], WGS84_SEMI_MAJOR_AXIS_M * (1.0 - WGS84_FLATTENING), rtol=1e-12
        )
        np.testing.assert_allclose(ecef_m[:2], 0.0, atol=1e-9)

    def test_rejects_a_latitude_south_of_the_pole(self) -> None:
        """A southern case catches a guard that forgot the absolute value."""
        with pytest.raises(ValueError, match=r"\[-90, 90\]"):
            geodetic_to_ecef_m(-90.1, 0.0, 0.0)


class TestGeodeticToEnu:
    def test_altitude_above_the_origin_is_pure_up(self) -> None:
        enu_m = geodetic_to_enu_m(36.0025, -78.941, 1560.0, 36.0025, -78.941, 60.0)
        np.testing.assert_allclose(enu_m[2], 1500.0, rtol=1e-9)
        np.testing.assert_allclose(enu_m[:2], 0.0, atol=1e-6)

    def test_equatorial_east_step_and_the_earth_curving_away(self) -> None:
        """At the equator, 0.18 deg east is a sin(d) east and -a (1 - cos d) up, about -31.5 m.

        Catches east and north swapped, a flipped sign, or a wrong "up".
        """
        step_rad = np.radians(0.18)
        enu_m = geodetic_to_enu_m(0.0, np.degrees(step_rad), 0.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(enu_m[0], WGS84_SEMI_MAJOR_AXIS_M * np.sin(step_rad), rtol=1e-12)
        np.testing.assert_allclose(enu_m[1], 0.0, atol=1e-9)
        # a cos(d) - a cancels 6.4e6 m to 31 m: about 1e-9 m of round-off.
        np.testing.assert_allclose(
            enu_m[2], -WGS84_SEMI_MAJOR_AXIS_M * (1.0 - np.cos(step_rad)), rtol=1e-9
        )

    def test_is_a_rigid_motion_so_distance_is_preserved(self) -> None:
        """The ENU norm is the chord, which is the slant range that sets the delay."""
        rng = np.random.default_rng(20260911)
        latitude_deg = 36.0025 + rng.uniform(-0.1, 0.1, size=32)
        longitude_deg = -78.941 + rng.uniform(-0.1, 0.1, size=32)
        altitude_m = rng.uniform(0.0, 3000.0, size=32)
        ecef_distance_m = np.linalg.norm(
            geodetic_to_ecef_m(latitude_deg, longitude_deg, altitude_m)
            - geodetic_to_ecef_m(36.0025, -78.941, 60.0),
            axis=-1,
        )
        enu_m = geodetic_to_enu_m(latitude_deg, longitude_deg, altitude_m, 36.0025, -78.941, 60.0)
        np.testing.assert_allclose(np.linalg.norm(enu_m, axis=-1), ecef_distance_m, rtol=1e-12)

    def test_rejects_a_last_axis_that_is_not_three(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
            ecef_to_enu_m(np.zeros((4, 2)), 0.0, 0.0, 0.0)


class TestSphericalCoordinates:
    def test_azimuth_is_zero_at_north_and_increases_clockwise(self) -> None:
        """Catches the mathematical convention, a flipped sense, or a centred wrap."""
        _, azimuth_rad, _ = enu_to_range_azimuth_elevation_rad(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        np.testing.assert_allclose(azimuth_rad, [0.0, np.pi / 2, np.pi, 3 * np.pi / 2], atol=1e-15)

    def test_a_pythagorean_point_gives_exact_range_and_angles(self) -> None:
        """(300, 400, 1200) is 3-4-5 on 5-12-13: range 1300 m, elevation asin(12/13)."""
        range_m, azimuth_rad, elevation_rad = enu_to_range_azimuth_elevation_rad(
            (300.0, 400.0, 1200.0)
        )
        np.testing.assert_allclose(range_m, 1300.0, rtol=1e-12)
        np.testing.assert_allclose(elevation_rad, np.arcsin(12.0 / 13.0), rtol=1e-12)
        np.testing.assert_allclose(azimuth_rad, np.arctan2(3.0, 4.0), rtol=1e-12)

    def test_rejects_a_last_axis_that_is_not_three(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
            enu_to_range_azimuth_elevation_rad(np.zeros((4, 2)))

    def test_the_inverse_puts_cardinal_azimuths_on_the_axes(self) -> None:
        """Catches sine and cosine swapped, which turns north into east."""
        enu_m = range_azimuth_elevation_to_enu_m(
            1000.0, np.array([np.pi / 2, 0.0, 1.0]), np.array([0.0, 0.0, np.pi / 2])
        )
        # atol 1e-9 m: cos(pi / 2) is 6e-17, times 1000 m.
        np.testing.assert_allclose(
            enu_m, [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]], atol=1e-9
        )

    def test_the_inverse_round_trips(self) -> None:
        enu_m = np.array(
            [[300.0, 400.0, 1200.0], [-250.0, 80.0, -40.0], [-10.0, -900.0, 5.0], [7.0, -3.0, 2.0]]
        )
        recovered_m = range_azimuth_elevation_to_enu_m(*enu_to_range_azimuth_elevation_rad(enu_m))
        np.testing.assert_allclose(recovered_m, enu_m, rtol=1e-12, atol=1e-12)


def test_the_ground_plane_pair_is_the_spherical_conversion_with_no_height() -> None:
    """Range and azimuth clockwise from north, as the 3-D conversion gives, and back again."""
    rng = np.random.default_rng(SEED)
    position_m = rng.normal(0.0, 2e4, (50, 2))
    range_m, azimuth_rad = east_north_to_range_azimuth_rad(position_m)
    expected = enu_to_range_azimuth_elevation_rad(np.column_stack([position_m, np.zeros(50)]))
    np.testing.assert_array_equal(range_m, expected[0])
    np.testing.assert_array_equal(azimuth_rad, expected[1])
    # rtol 1e-12: two trigonometric round trips.
    np.testing.assert_allclose(
        range_azimuth_to_east_north_m(range_m, azimuth_rad), position_m, rtol=1e-12, atol=1e-8
    )


def test_the_enu_axes_are_a_rotation_whose_up_is_the_ellipsoid_normal() -> None:
    """Orthonormal rows, up along (cos lat cos lon, cos lat sin lon, sin lat), as ecef_to_enu_m."""
    latitude_deg, longitude_deg = 36.0025, -78.941
    axes = enu_axes_in_ecef(latitude_deg, longitude_deg)
    # atol 1e-15: products of sines and cosines, a few float64 epsilons.
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-15)
    lat, lon = np.radians(latitude_deg), np.radians(longitude_deg)
    normal = [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    np.testing.assert_allclose(axes[2], normal, atol=1e-15)
    offset_m = np.array([[1200.0, -3400.0, 560.0], [-75.0, 20.0, 9000.0]])
    site_ecef_m = geodetic_to_ecef_m(latitude_deg, longitude_deg, 60.0)
    expected_m = ecef_to_enu_m(site_ecef_m + offset_m, latitude_deg, longitude_deg, 60.0)
    # atol 1e-8 m: ecef_to_enu_m subtracts 6e6 m coordinates, a few nm of round-off.
    np.testing.assert_allclose(offset_m @ axes.T, expected_m, rtol=0.0, atol=1e-8)
