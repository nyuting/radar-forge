"""Tests for the shared physical constants.

These guard the values themselves. The *uniqueness* of the definitions — that no
other module redefines or re-hardcodes them — is enforced by
scripts/check_conventions.py, not from here.
"""

from __future__ import annotations

import math

from radar_forge.core import constants


def test_speed_of_light_is_the_exact_si_value() -> None:
    # Exact by the 2019 SI definition of the metre, so an exact comparison is
    # correct here: this is a defined integer, not a measurement.
    assert constants.SPEED_OF_LIGHT_MPS == 299_792_458.0


def test_boltzmann_is_the_exact_si_value() -> None:
    assert constants.BOLTZMANN_JPK == 1.380_649e-23


def test_thermal_noise_floor_matches_the_textbook_minus_174_dbm_per_hz() -> None:
    """kT0 at 290 K is the -174 dBm/Hz every radar engineer quotes."""
    noise_dbm_per_hz = 10.0 * math.log10(
        constants.BOLTZMANN_JPK * constants.STANDARD_NOISE_TEMPERATURE_K * 1e3
    )
    # rtol via abs: the quoted figure is itself rounded to the nearest 0.1 dB.
    assert abs(noise_dbm_per_hz - (-173.98)) < 0.01


def test_four_thirds_earth_radius_is_derived_not_retyped() -> None:
    assert constants.FOUR_THIRDS_EARTH_RADIUS_M == 4.0 / 3.0 * constants.EARTH_RADIUS_M


def test_all_is_complete() -> None:
    """Every public module-level constant is exported, so nothing hides."""
    public = {name for name in vars(constants) if name.isupper() and not name.startswith("_")}
    assert public == set(constants.__all__)


def test_wgs84_semi_major_axis_is_the_defining_value() -> None:
    # Defining constant of WGS-84 (NIMA TR8350.2), exact by definition.
    assert constants.WGS84_SEMI_MAJOR_AXIS_M == 6_378_137.0


def test_earth_radius_is_the_iugg_mean_radius() -> None:
    """Catches the WGS-84 equatorial radius (7.1 km larger) standing in for the mean one.

    The spherical-Earth constant is the IUGG mean radius R1 = (2a + b)/3 of the
    GRS 80 ellipsoid, 6 371 008.7714 m (Moritz, "Geodetic Reference System
    1980", J. Geodesy 74, 2000). Defined to that many digits, so the
    comparison is exact.
    """
    assert constants.EARTH_RADIUS_M == 6_371_008.771_4


def test_wgs84_flattening_matches_the_published_inverse() -> None:
    """TR8350.2 publishes 1/f; we store f, so check the round trip."""
    assert math.isclose(1.0 / constants.WGS84_FLATTENING, 298.257_223_563, rel_tol=1e-15)
