"""Tests for amplitude tapers and the cost of applying them."""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core import (
    TAPER_NAMES,
    apply_taper,
    coherent_gain_linear,
    processing_loss_db,
    taper,
)

# Long enough that the asymptotic sidelobe and loss figures in the docstring
# table are actually reached; short windows do not converge to them.
N_LONG = 4096


def peak_sidelobe_db(window: np.ndarray) -> float:
    """Return the highest sidelobe of a window, in dB below its mainlobe peak.

    The window is heavily zero-padded so the sidelobe peaks are resolved rather
    than straddled, and the mainlobe is excised by walking out from bin 0 to the
    first local minimum on each side.
    """
    spectrum = np.abs(np.fft.fft(window, n=32 * window.size))
    spectrum = spectrum / spectrum.max()
    # Walk out of the mainlobe: the first bin where the response stops falling.
    index = 1
    while index < spectrum.size // 2 and spectrum[index + 1] < spectrum[index]:
        index += 1
    return float(20.0 * np.log10(spectrum[index : spectrum.size // 2].max()))


class TestTaper:
    @pytest.mark.parametrize("name", [name for name in TAPER_NAMES if name != "rectangular"])
    def test_is_symmetric(self, name: str) -> None:
        """Catches a periodic (``sym=False``) window where a symmetric one is needed.

        The Harris sidelobe and loss figures barely move between the two forms
        at N = 4096, so this is the only test that sees the difference. One
        case per SciPy-built taper, since each is a separate call; the
        rectangular one is ``np.ones`` and cannot be periodic.
        """
        window = taper(name, 33)
        assert window.shape == (33,)
        # Not exact: SciPy builds these from cosines, so the two ends agree only
        # to rounding. atol 1e-15 covers that while still catching any window
        # genuinely built asymmetrically (the periodic sym=False variant, say),
        # which would differ in the first decimal, not the fifteenth.
        np.testing.assert_allclose(window, window[::-1], rtol=1e-12, atol=1e-15)

    def test_rectangular_is_all_ones(self) -> None:
        np.testing.assert_array_equal(taper("rectangular", 10, normalize=False), np.ones(10))

    @pytest.mark.parametrize(
        ("name", "expected_db"),
        [
            ("rectangular", -13.26),
            ("hann", -31.5),
            ("hamming", -42.7),
            ("blackman", -58.1),
            ("blackmanharris", -92.0),
        ],
        ids=["rectangular", "hann", "hamming", "blackman", "blackmanharris"],
    )
    def test_matches_published_sidelobe_levels(self, name: str, expected_db: float) -> None:
        """Peak sidelobe levels against Harris 1978, Table 1.

        Catches the wrong window behind a name. The levels are 10 dB or more
        apart, so this also pins the documented ordering.

        References
        ----------
        .. [1] F. J. Harris, "On the use of windows for harmonic analysis with
               the discrete Fourier transform," Proc. IEEE 66(1), 1978, Table 1
               (the 4-term Blackman-Harris is the "minimum 4-sample" row).
        """
        # atol 0.5 dB: Harris tabulates sidelobes to 1 dB, and a discrete window of
        # finite length straddles the true continuous sidelobe peak.
        np.testing.assert_allclose(
            peak_sidelobe_db(taper(name, N_LONG, normalize=False)), expected_db, atol=0.5
        )

    def test_taylor_achieves_its_design_sidelobe(self) -> None:
        """The auto-chosen nbar must actually deliver the requested level.

        This is the test that a fixed nbar=4 would fail: it returns a window
        whose sidelobes sit far above the level it was asked for.
        """
        for sidelobe_db in (30.0, 50.0, 70.0):
            achieved_db = peak_sidelobe_db(
                taper("taylor", N_LONG, normalize=False, sidelobe_db=sidelobe_db)
            )
            # atol 2 dB: Taylor equalises only the near-in sidelobes, and the
            # far-out ones decay, so the peak sits at the design level itself.
            np.testing.assert_allclose(achieved_db, -sidelobe_db, atol=2.0)

    def test_chebyshev_achieves_its_design_sidelobe(self) -> None:
        """Dolph-Chebyshev equalises every sidelobe at exactly the design level."""
        achieved_db = peak_sidelobe_db(
            taper("chebyshev", N_LONG, normalize=False, sidelobe_db=60.0)
        )
        # atol 0.5 dB: equiripple by construction, so this should be tight.
        np.testing.assert_allclose(achieved_db, -60.0, atol=0.5)

    def test_rejects_unknown_name_and_lists_the_accepted_ones(self) -> None:
        """An error that does not say what was expected is half an error."""
        with pytest.raises(ValueError, match=r"unknown taper.*blackmanharris"):
            taper("kaiser", 16)

    def test_rejects_a_zero_length(self) -> None:
        """Zero is the boundary of the ``n_samples >= 1`` guard."""
        with pytest.raises(ValueError, match="at least one"):
            taper("hann", 0)

    def test_rejects_a_zero_sidelobe_level(self) -> None:
        """Zero is the boundary: a ``< 0`` guard would let a 0 dB design through."""
        with pytest.raises(ValueError, match="must be positive"):
            taper("taylor", 64, sidelobe_db=0.0)

    def test_rejects_shallow_chebyshev(self) -> None:
        """Below ~45 dB the Chebyshev design grows spikes at its own edges."""
        with pytest.raises(ValueError, match="chebyshev"):
            taper("chebyshev", 64, sidelobe_db=20.0)


class TestApplyTaper:
    def test_weights_a_negative_axis(self) -> None:
        """Catches a negative axis left unnormalised: -1 is the last axis of a 2-D cube."""
        cube = np.ones((4, 8), dtype=np.complex128)
        window = taper("hann", 8, normalize=False)
        np.testing.assert_allclose(apply_taper(cube, window, axis=-1), cube * window, rtol=1e-12)

    def test_weights_slow_time_independently(self) -> None:
        """Tapering axis 0 of a cube must not touch the fast-time structure."""
        rng = np.random.default_rng(20260911)
        cube = rng.standard_normal((6, 5)) + 1j * rng.standard_normal((6, 5))
        window = taper("hamming", 6, normalize=False)
        tapered = apply_taper(cube, window, axis=0)
        np.testing.assert_allclose(tapered, cube * window[:, None], rtol=1e-12)

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must match samples"):
            apply_taper(np.ones((4, 8)), taper("hann", 7), axis=1)

    def test_rejects_two_dimensional_window(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            apply_taper(np.ones((4, 8)), np.ones((4, 8)), axis=1)

    def test_rejects_out_of_range_axis(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            apply_taper(np.ones((4, 8)), taper("hann", 8), axis=5)


class TestGainAndLoss:
    def test_normalized_tapers_have_unit_coherent_gain(self) -> None:
        """This is the definition of the normalisation, so it must hold exactly."""
        for name in TAPER_NAMES:
            np.testing.assert_allclose(coherent_gain_linear(taper(name, 128)), 1.0, rtol=1e-12)

    def test_loss_is_invariant_to_scaling(self) -> None:
        """Loss is a shape property, so normalising must not change it."""
        raw = processing_loss_db(taper("blackman", 256, normalize=False))
        normalized = processing_loss_db(taper("blackman", 256, normalize=True))
        np.testing.assert_allclose(raw, normalized, rtol=1e-12)

    @pytest.mark.parametrize(
        ("name", "expected_db"),
        [("hann", 1.76), ("hamming", 1.34), ("blackman", 2.37), ("blackmanharris", 3.02)],
        ids=["hann", "hamming", "blackman", "blackmanharris"],
    )
    def test_matches_published_processing_loss(self, name: str, expected_db: float) -> None:
        """Coherent processing loss against Harris 1978, Table 1.

        References
        ----------
        .. [1] F. J. Harris, Proc. IEEE 66(1), 1978, Table 1, "Equiv. noise BW"
               column, re-expressed as 10 log10(ENBW) dB.
        """
        # atol 0.01 dB: Harris tabulates to two decimals and this is an exact
        # closed-form ratio, so there is nothing to be loose about.
        np.testing.assert_allclose(processing_loss_db(taper(name, N_LONG)), expected_db, atol=0.01)

    def test_rejects_zero_sum_window(self) -> None:
        with pytest.raises(ValueError, match="sum to zero"):
            processing_loss_db(np.array([1.0, -1.0]))


class TestWindowMetricValidation:
    """The shared `_validated_window` guard, reached through both metrics."""

    @pytest.mark.parametrize("metric", [coherent_gain_linear, processing_loss_db])
    def test_rejects_a_two_dimensional_window(self, metric) -> None:
        """A window is one taper along one axis; a matrix is a caller's mistake.

        Without the guard, NumPy would happily average a whole cube and return a
        number with no meaning, which is exactly the failure mode numerical code
        has to be defended against.
        """
        with pytest.raises(ValueError, match="one-dimensional"):
            metric(np.ones((4, 4)))

    def test_rejects_an_empty_window(self) -> None:
        """The mean of nothing is nan. The 2-D cases already show both metrics share the guard."""
        with pytest.raises(ValueError, match="must not be empty"):
            coherent_gain_linear(np.zeros(0))
