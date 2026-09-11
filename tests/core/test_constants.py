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


def test_speed_of_light_is_not_the_3e8_approximation() -> None:
    """The 3e8 shortcut is 0.07% high — 70 cm of range error at 1 km."""
    relative_error = abs(3e8 - constants.SPEED_OF_LIGHT_MPS) / constants.SPEED_OF_LIGHT_MPS
    assert relative_error > 6e-4
    assert constants.SPEED_OF_LIGHT_MPS != 3e8


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


def test_every_exported_constant_is_a_positive_float() -> None:
    for name in constants.__all__:
        value = getattr(constants, name)
        assert isinstance(value, float), f"{name} should be a float"
        assert value > 0.0, f"{name} should be positive"


def test_all_is_complete() -> None:
    """Every public module-level constant is exported, so nothing hides."""
    public = {name for name in vars(constants) if name.isupper() and not name.startswith("_")}
    assert public == set(constants.__all__)


def test_wgs84_semi_major_axis_is_the_defining_value() -> None:
    # Defining constant of WGS-84 (NIMA TR8350.2), exact by definition.
    assert constants.WGS84_SEMI_MAJOR_AXIS_M == 6_378_137.0


def test_wgs84_semi_major_axis_is_not_the_mean_radius() -> None:
    """The equatorial radius and the IUGG mean radius differ by ~7.1 km.

    Substituting one for the other in a geodetic conversion is a kilometre-scale
    position error, so the two constants must stay distinct.
    """
    difference_m = constants.WGS84_SEMI_MAJOR_AXIS_M - constants.EARTH_RADIUS_M
    assert 7_000.0 < difference_m < 7_200.0


def test_wgs84_flattening_matches_the_published_inverse() -> None:
    """TR8350.2 publishes 1/f; we store f, so check the round trip."""
    assert math.isclose(1.0 / constants.WGS84_FLATTENING, 298.257_223_563, rel_tol=1e-15)


def test_wgs84_polar_axis_is_about_21_km_shorter() -> None:
    """b = a(1 - f). The pole-to-equator difference is the classic ~21.4 km."""
    semi_minor_axis_m = constants.WGS84_SEMI_MAJOR_AXIS_M * (1.0 - constants.WGS84_FLATTENING)
    assert 21_000.0 < constants.WGS84_SEMI_MAJOR_AXIS_M - semi_minor_axis_m < 21_500.0
