"""Tests for the decibel conversions of a power and of an amplitude.

Ground truth is analytic: whole powers of ten have whole decibel values, a doubling of power is
10 log10 2 = 3.0103 dB and of amplitude twice that, a signal's power and amplitude are the same
number of decibels, and each pair inverts the other.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from radar_forge.units import amplitude_to_db, db_to_amplitude, db_to_power, power_to_db


class TestPower:
    def test_powers_of_ten_are_tens_of_decibels(self) -> None:
        np.testing.assert_allclose(
            power_to_db([1e-3, 1.0, 10.0, 1e3]), [-30, 0, 10, 30], atol=1e-12
        )
        np.testing.assert_allclose(db_to_power([-30.0, 0.0, 10.0, 30.0]), [1e-3, 1, 10, 1e3])

    def test_a_doubling_of_power_is_three_decibels(self) -> None:
        np.testing.assert_allclose(power_to_db(2.0), 10.0 * math.log10(2.0), rtol=1e-15)
        np.testing.assert_allclose(power_to_db(2.0), 3.0103, atol=5e-5)

    def test_the_pair_inverts_itself(self) -> None:
        value_db = np.linspace(-150.0, 150.0, 61)
        np.testing.assert_allclose(power_to_db(db_to_power(value_db)), value_db, atol=1e-12)


class TestAmplitude:
    def test_powers_of_ten_are_twenties_of_decibels(self) -> None:
        np.testing.assert_allclose(amplitude_to_db([0.1, 1.0, 10.0]), [-20, 0, 20], atol=1e-12)
        np.testing.assert_allclose(db_to_amplitude([-20.0, 0.0, 20.0]), [0.1, 1, 10])

    def test_an_amplitude_and_its_power_are_the_same_decibels(self) -> None:
        amplitude = np.array([1e-4, 0.5, 2.0, 3.7e5])
        np.testing.assert_allclose(amplitude_to_db(amplitude), power_to_db(amplitude**2))
        np.testing.assert_allclose(db_to_amplitude(6.0) ** 2, db_to_power(6.0))

    def test_the_pair_inverts_itself(self) -> None:
        value_db = np.linspace(-150.0, 150.0, 61)
        np.testing.assert_allclose(amplitude_to_db(db_to_amplitude(value_db)), value_db, atol=1e-12)


@pytest.mark.parametrize("to_db", [power_to_db, amplitude_to_db])
class TestEdges:
    def test_zero_is_minus_infinity_without_a_warning(self, to_db) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert to_db(0.0) == -np.inf
            assert to_db([0.0, 1.0]).tolist() == [-np.inf, 0.0]

    @pytest.mark.parametrize("value", [-1e-30, math.nan, [1.0, -2.0]])
    def test_a_negative_or_nan_value_raises(self, to_db, value) -> None:
        with pytest.raises(ValueError, match="value_linear must be non-negative"):
            to_db(value)

    def test_infinity_is_infinite_decibels(self, to_db) -> None:
        assert to_db(np.inf) == np.inf

    def test_shape_is_kept(self, to_db) -> None:
        assert to_db(np.ones((2, 3, 4))).shape == (2, 3, 4)


def test_minus_infinity_decibels_is_nothing() -> None:
    assert db_to_power(-np.inf) == 0.0
    assert db_to_amplitude(-np.inf) == 0.0
