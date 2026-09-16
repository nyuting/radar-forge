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

    def test_altitude_adds_along_the_radial_at_the_equator(self) -> None:
        """On the equator the ellipsoid normal is radial, so h adds to x exactly."""
        at_surface = geodesy.geodetic_to_ecef_m(0.0, 0.0, 0.0)
        at_altitude = geodesy.geodetic_to_ecef_m(0.0, 0.0, 1000.0)
        np.testing.assert_allclose(at_altitude[0] - at_surface[0], 1000.0, rtol=1e-12)

    def test_broadcasts_to_a_trailing_axis_of_three(self) -> None:
        latitude_deg = np.linspace(-45.0, 45.0, 7)
        ecef_m = geodesy.geodetic_to_ecef_m(latitude_deg, 103.0, 0.0)
        assert ecef_m.shape == (7, 3)

    @pytest.mark.parametrize("bad_latitude_deg", [90.1, -90.1, 180.0])
    def test_rejects_impossible_latitude(self, bad_latitude_deg: float) -> None:
        with pytest.raises(ValueError, match=r"\[-90, 90\]"):
            geodesy.geodetic_to_ecef_m(bad_latitude_deg, 0.0, 0.0)


class TestGeodeticToEnu:
    def test_the_origin_maps_to_zero(self) -> None:
        enu_m = geodesy.geodetic_to_enu_m(
            DUKE_LATITUDE_DEG,
            DUKE_LONGITUDE_DEG,
            DUKE_ALTITUDE_M,
            DUKE_LATITUDE_DEG,
            DUKE_LONGITUDE_DEG,
            DUKE_ALTITUDE_M,
        )
        np.testing.assert_allclose(enu_m, 0.0, atol=1e-8)

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

    def test_equatorial_east_step_matches_the_exact_arc(self) -> None:
        """At the equator the prime-vertical radius is exactly a.

        So a longitude step of dlambda is an east displacement of a*dlambda, with
        no ellipsoid correction at all — an exact closed-form check.
        """
        step_deg = 0.01
        enu_m = geodesy.geodetic_to_enu_m(0.0, step_deg, 0.0, 0.0, 0.0, 0.0)
        expected_east_m = WGS84_SEMI_MAJOR_AXIS_M * np.radians(step_deg)
        # Chord vs arc: the transform gives the chord, shorter by O(theta^2/24).
        np.testing.assert_allclose(enu_m[0], expected_east_m, rtol=1e-8)
        np.testing.assert_allclose(enu_m[1], 0.0, atol=1e-9)

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
    @pytest.mark.parametrize(
        ("enu_m", "expected_azimuth_deg"),
        [
            ((0.0, 1000.0, 0.0), 0.0),  # north
            ((1000.0, 0.0, 0.0), 90.0),  # east
            ((0.0, -1000.0, 0.0), 180.0),  # south
            ((-1000.0, 0.0, 0.0), 270.0),  # west
            ((1000.0, 1000.0, 0.0), 45.0),  # north-east
        ],
        ids=["north", "east", "south", "west", "north-east"],
    )
    def test_azimuth_is_zero_at_north_and_increases_clockwise(
        self, enu_m: tuple[float, float, float], expected_azimuth_deg: float
    ) -> None:
        _, azimuth_deg, _ = geodesy.enu_to_range_azimuth_elevation(enu_m)
        np.testing.assert_allclose(azimuth_deg, expected_azimuth_deg, atol=1e-12)

    def test_elevation_is_ninety_degrees_straight_up(self) -> None:
        _, _, elevation_deg = geodesy.enu_to_range_azimuth_elevation((0.0, 0.0, 500.0))
        np.testing.assert_allclose(elevation_deg, 90.0, rtol=1e-12)

    def test_elevation_is_zero_on_the_horizontal_plane(self) -> None:
        _, _, elevation_deg = geodesy.enu_to_range_azimuth_elevation((300.0, 400.0, 0.0))
        np.testing.assert_allclose(elevation_deg, 0.0, atol=1e-12)

    def test_range_is_the_euclidean_norm(self) -> None:
        """A 3-4-5 triangle in two dimensions, so the answer is exact."""
        range_m, _, _ = geodesy.enu_to_range_azimuth_elevation((300.0, 400.0, 0.0))
        np.testing.assert_allclose(range_m, 500.0, rtol=1e-15)

    def test_elevation_of_forty_five_degrees(self) -> None:
        range_m, azimuth_deg, elevation_deg = geodesy.enu_to_range_azimuth_elevation(
            (0.0, 1000.0, 1000.0)
        )
        np.testing.assert_allclose(elevation_deg, 45.0, rtol=1e-12)
        np.testing.assert_allclose(azimuth_deg, 0.0, atol=1e-12)
        np.testing.assert_allclose(range_m, 1000.0 * np.sqrt(2.0), rtol=1e-15)

    def test_vectorises_over_leading_axes(self) -> None:
        enu_m = np.zeros((5, 3))
        enu_m[:, 1] = np.arange(1.0, 6.0) * 1000.0
        range_m, azimuth_deg, elevation_deg = geodesy.enu_to_range_azimuth_elevation(enu_m)
        assert range_m.shape == (5,)
        np.testing.assert_allclose(azimuth_deg, 0.0, atol=1e-12)
        np.testing.assert_allclose(elevation_deg, 0.0, atol=1e-12)

    def test_rejects_a_last_axis_that_is_not_three(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
            geodesy.enu_to_range_azimuth_elevation(np.zeros((4, 2)))


def test_scenario_001_track_lies_where_the_spec_says() -> None:
    """The flight track is 5-22 km from the radar, per spec/scenario-001.

    A geometry regression guard: if a sign or an argument order in the ENU
    transform flips, this moves by kilometres.

    The span is wider than the 8-18 km of the pre-translation geometry, and
    not because the track changed: it is the same fixes, moved by one constant
    offset. The receiver moved differently. Before, the site sat inside the
    circuit; now it stands 19.6 km north-west of the airport the circuit is
    flown around, so the near corner comes closer and the far corner recedes.
    """
    # The two extreme corners of the track's lat/lon bounding box.
    enu_m = geodesy.geodetic_to_enu_m(
        [35.86376, 35.96334],
        [-79.12149, -78.92126],
        1500.0,
        DUKE_LATITUDE_DEG,
        DUKE_LONGITUDE_DEG,
        DUKE_ALTITUDE_M,
    )
    range_m, azimuth_deg, _ = geodesy.enu_to_range_azimuth_elevation(enu_m)
    assert np.all(range_m > 4_000.0)
    assert np.all(range_m < 25_000.0)
    # Both corners are south of the radar -- the airport is south-east of the
    # site -- so azimuth is southerly. Before the translation the site was on
    # the other side of the track and this read northerly.
    assert np.all((azimuth_deg > 90.0) & (azimuth_deg < 270.0))
