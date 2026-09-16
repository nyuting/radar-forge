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


def test_power_scales_with_the_square_of_the_wavelength() -> None:
    """The lambda^2 in the numerator is the receive aperture, not a free parameter.

    G_r lambda^2 / 4pi is the effective aperture of the receive antenna, so at
    fixed *gain* a longer wavelength collects more power. Writing the equation
    with lambda to the first power (the classic transcription slip) would make
    this ratio 2, not 4, and every link budget would be wrong by a factor that
    looks plausible.
    """
    base = received_power_w(**NOMINAL, range_m=1000.0)
    doubled = received_power_w(
        **{**NOMINAL, "wavelength_m": 2.0 * NOMINAL["wavelength_m"]}, range_m=1000.0
    )
    # rtol 1e-12: the two calls differ by one exact factor of four in float64.
    np.testing.assert_allclose(doubled / base, 4.0, rtol=1e-12)


def test_power_is_exactly_linear_in_the_transmitted_power() -> None:
    """A 3 dB power amplifier buys exactly 3 dB of received power, and no more.

    Superposition in the link budget: nothing in the equation is nonlinear in
    P_t, so the check also catches a stray square or a mis-parenthesised term.
    """
    base = received_power_w(**NOMINAL, range_m=1000.0)
    scaled = received_power_w(
        **{**NOMINAL, "transmit_power_w": 8.0 * NOMINAL["transmit_power_w"]}, range_m=1000.0
    )
    np.testing.assert_allclose(scaled, 8.0 * base, rtol=1e-12)


def test_the_two_gains_enter_symmetrically() -> None:
    """Reciprocity: swapping the transmit and receive gains changes nothing.

    G_t and G_r appear as a product, which is why a monostatic radar quotes one
    gain twice. A test that pins the symmetry catches an implementation that
    squares one gain and drops the other.
    """
    swapped = dict(NOMINAL)
    swapped["gain_tx_linear"], swapped["gain_rx_linear"] = 10**2.0, 10**1.0
    reversed_pair = dict(swapped)
    reversed_pair["gain_tx_linear"], reversed_pair["gain_rx_linear"] = 10**1.0, 10**2.0
    np.testing.assert_allclose(
        received_power_w(**swapped, range_m=1000.0),
        received_power_w(**reversed_pair, range_m=1000.0),
        rtol=0.0,
        atol=0.0,
    )
