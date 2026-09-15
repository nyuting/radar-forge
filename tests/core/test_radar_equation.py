"""Tests for the monostatic radar range equation."""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core import bistatic_received_power_w, received_power_w

# A representative 77 GHz automotive front-end.
NOMINAL = {
    "transmit_power_w": 0.01,
    "gain_tx_linear": 10 ** (15.0 / 10.0),
    "gain_rx_linear": 10 ** (15.0 / 10.0),
    "wavelength_m": 3.896e-3,
    "rcs_m2": 10 ** (5.0 / 10.0),
}

# The same front-end, keyed for the bistatic signature, which names the
# cross-section bistatic_rcs_m2 to keep the angle-dependence caveat visible.
BISTATIC_NOMINAL = {
    **{key: value for key, value in NOMINAL.items() if key != "rcs_m2"},
    "bistatic_rcs_m2": NOMINAL["rcs_m2"],
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


class TestBistaticReceivedPower:
    """The bistatic form, and its collapse onto the monostatic one."""

    def test_reduces_to_monostatic_at_equal_ranges(self) -> None:
        """The defining property: equal ranges are a monostatic radar."""
        range_m = 12.0e3
        np.testing.assert_allclose(
            bistatic_received_power_w(**BISTATIC_NOMINAL, range_tx_m=range_m, range_rx_m=range_m),
            received_power_w(**NOMINAL, range_m=range_m),
            # The same float64 products in a different grouping.
            rtol=1e-12,
        )

    def test_power_follows_the_range_product_not_the_range_sum(self) -> None:
        """R_t^2 R_r^2 in the denominator, so only the product of the ranges matters.

        Swapping the two ranges, and replacing them by their geometric mean,
        must all give the same power. A (R_t + R_r)^4 denominator would pass the
        swap and fail the geometric mean, which is why both are asserted.
        """
        near_m, far_m = 1.0e3, 1.0e4
        geometric_mean_m = np.sqrt(near_m * far_m)
        reference_w = bistatic_received_power_w(
            **BISTATIC_NOMINAL, range_tx_m=near_m, range_rx_m=far_m
        )
        np.testing.assert_allclose(
            bistatic_received_power_w(**BISTATIC_NOMINAL, range_tx_m=far_m, range_rx_m=near_m),
            reference_w,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            bistatic_received_power_w(
                **BISTATIC_NOMINAL, range_tx_m=geometric_mean_m, range_rx_m=geometric_mean_m
            ),
            reference_w,
            rtol=1e-12,
        )

    @pytest.mark.parametrize("doubled", ["range_tx_m", "range_rx_m"])
    def test_doubling_either_range_costs_twelve_decibels(self, doubled: str) -> None:
        """Each range enters squared, so doubling one costs a factor of four in power.

        Doubling *both* is the monostatic 16x; doubling one is 4x.
        """
        ranges_m = {"range_tx_m": 5.0e3, "range_rx_m": 5.0e3}
        reference_w = bistatic_received_power_w(**BISTATIC_NOMINAL, **ranges_m)
        ranges_m[doubled] *= 2.0
        np.testing.assert_allclose(
            bistatic_received_power_w(**BISTATIC_NOMINAL, **ranges_m), reference_w / 4.0, rtol=1e-12
        )

    def test_broadcasts_over_targets(self) -> None:
        ranges_tx_m = np.array([1.0e3, 2.0e3, 4.0e3])
        power_w = bistatic_received_power_w(
            **BISTATIC_NOMINAL, range_tx_m=ranges_tx_m, range_rx_m=3.0e3
        )
        assert power_w.shape == (3,)
        assert np.all(np.diff(power_w) < 0.0)

    @pytest.mark.parametrize(
        ("range_tx_m", "range_rx_m"), [(0.0, 1.0e3), (1.0e3, 0.0), (-1.0, 1.0e3)]
    )
    def test_rejects_non_positive_range(self, range_tx_m: float, range_rx_m: float) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            bistatic_received_power_w(
                **BISTATIC_NOMINAL, range_tx_m=range_tx_m, range_rx_m=range_rx_m
            )

    def test_rejects_a_gain_dressed_as_a_loss(self) -> None:
        with pytest.raises(ValueError, match="linear factor"):
            bistatic_received_power_w(
                **BISTATIC_NOMINAL, range_tx_m=1.0e3, range_rx_m=1.0e3, loss_linear=0.5
            )
