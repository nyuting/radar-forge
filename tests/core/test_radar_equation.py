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
    """Catches any algebra slip: lambda^1, a squared P_t, one gain dropped or squared.

    The receive gain is set 3 dB below the transmit gain so that an
    implementation squaring G_t (or G_r) cannot pass, as it would with the
    equal gains of NOMINAL.
    """
    # Independent evaluation of the textbook expression, not a golden number:
    # a golden number would only prove the code has not changed.
    params = {**NOMINAL, "gain_rx_linear": 10 ** (12.0 / 10.0)}
    range_m = 42.0
    expected = (
        params["transmit_power_w"]
        * params["gain_tx_linear"]
        * params["gain_rx_linear"]
        * params["wavelength_m"] ** 2
        * params["rcs_m2"]
    ) / ((4 * np.pi) ** 3 * range_m**4)

    # rtol at 1e-12: this is a handful of float64 multiplies, so anything looser
    # would hide a genuine algebraic error.
    np.testing.assert_allclose(received_power_w(**params, range_m=range_m), expected, rtol=1e-12)


def test_inverse_fourth_power_scaling() -> None:
    """The Notes claim: doubling the range costs 12 dB, so a factor of 16.

    Passed as one array, so it also catches a scalar-only implementation.
    """
    powers_w = received_power_w(**NOMINAL, range_m=np.array([50.0, 100.0]))
    np.testing.assert_allclose(powers_w[0] / powers_w[1], 16.0, rtol=1e-12)


def test_loss_attenuates() -> None:
    lossless = received_power_w(**NOMINAL, range_m=25.0)
    lossy = received_power_w(**NOMINAL, range_m=25.0, loss_linear=10.0)
    np.testing.assert_allclose(lossy * 10.0, lossless, rtol=1e-12)


def test_rejects_a_zero_range() -> None:
    """Zero is the boundary: a ``< 0`` guard would divide by zero instead."""
    with pytest.raises(ValueError, match="strictly positive"):
        received_power_w(**NOMINAL, range_m=0.0)


def test_rejects_loss_below_one() -> None:
    with pytest.raises(ValueError, match="linear factor"):
        received_power_w(**NOMINAL, range_m=25.0, loss_linear=0.5)


class TestBistaticReceivedPower:
    """The bistatic form, and its collapse onto the monostatic one."""

    def test_reduces_to_monostatic_at_equal_ranges(self) -> None:
        """The defining property: equal ranges are a monostatic radar.

        Unequal gains, as in the monostatic closed-form test, so that a bistatic
        form squaring one gain cannot hide behind the symmetric NOMINAL pair.
        """
        range_m = 12.0e3
        gain_rx = {"gain_rx_linear": 10 ** (12.0 / 10.0)}
        np.testing.assert_allclose(
            bistatic_received_power_w(
                **{**BISTATIC_NOMINAL, **gain_rx}, range_tx_m=range_m, range_rx_m=range_m
            ),
            received_power_w(**{**NOMINAL, **gain_rx}, range_m=range_m),
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
        # One call over three geometries: (near, far), swapped, and the
        # geometric mean at both sites. All share the product R_t R_r.
        power_w = bistatic_received_power_w(
            **BISTATIC_NOMINAL,
            range_tx_m=np.array([near_m, far_m, geometric_mean_m]),
            range_rx_m=np.array([far_m, near_m, geometric_mean_m]),
        )
        np.testing.assert_allclose(power_w[1:], power_w[0], rtol=1e-12)

    @pytest.mark.parametrize(
        ("range_tx_m", "range_rx_m"), [(0.0, 1.0e3), (1.0e3, 0.0)], ids=["tx", "rx"]
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
