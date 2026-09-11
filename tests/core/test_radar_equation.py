"""Tests for the monostatic radar range equation."""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core import received_power_w

# A representative 77 GHz automotive front-end.
NOMINAL = {
    "transmit_power_w": 0.01,
    "gain_tx_linear": 10 ** (15.0 / 10.0),
    "gain_rx_linear": 10 ** (15.0 / 10.0),
    "wavelength_m": 3.896e-3,
    "rcs_m2": 10 ** (5.0 / 10.0),
}


def test_matches_closed_form() -> None:
    # Independent evaluation of the textbook expression, not a golden number:
    # a golden number would only prove the code has not changed.
    range_m = 42.0
    expected = (
        NOMINAL["transmit_power_w"]
        * NOMINAL["gain_tx_linear"]
        * NOMINAL["gain_rx_linear"]
        * NOMINAL["wavelength_m"] ** 2
        * NOMINAL["rcs_m2"]
    ) / ((4 * np.pi) ** 3 * range_m**4)

    # rtol at 1e-12: this is a handful of float64 multiplies, so anything looser
    # would hide a genuine algebraic error.
    np.testing.assert_allclose(received_power_w(**NOMINAL, range_m=range_m), expected, rtol=1e-12)


def test_inverse_fourth_power_scaling() -> None:
    near = received_power_w(**NOMINAL, range_m=50.0)
    far = received_power_w(**NOMINAL, range_m=100.0)
    np.testing.assert_allclose(near / far, 16.0, rtol=1e-12)


def test_broadcasts_over_range() -> None:
    ranges = np.array([10.0, 20.0, 40.0])
    powers = received_power_w(**NOMINAL, range_m=ranges)
    assert powers.shape == ranges.shape
    np.testing.assert_allclose(powers[:-1] / powers[1:], 16.0, rtol=1e-12)


def test_loss_attenuates() -> None:
    lossless = received_power_w(**NOMINAL, range_m=25.0)
    lossy = received_power_w(**NOMINAL, range_m=25.0, loss_linear=10.0)
    np.testing.assert_allclose(lossy * 10.0, lossless, rtol=1e-12)


@pytest.mark.parametrize("bad_range", [0.0, -1.0])
def test_rejects_non_positive_range(bad_range: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        received_power_w(**NOMINAL, range_m=bad_range)


def test_rejects_loss_below_one() -> None:
    with pytest.raises(ValueError, match="linear factor"):
        received_power_w(**NOMINAL, range_m=25.0, loss_linear=0.5)
