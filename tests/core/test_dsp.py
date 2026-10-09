"""Tests for range and Doppler processing."""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core import (
    beat_frequency_hz,
    doppler_bin_centers_mps,
    doppler_fft,
    lfm_chirp,
    matched_filter,
    mti_filter,
    range_bin_centers_m,
    range_doppler_map,
    range_fft,
    taper,
)

# A representative 77 GHz automotive FMCW front-end, deramped.
NOMINAL = {
    "bandwidth_hz": 1e9,
    "chirp_duration_s": 40e-6,
    "sample_rate_hz": 1e6,
}
WAVELENGTH_M = 3.896e-3
PULSE_REPETITION_INTERVAL_S = 50e-6
N_SAMPLES = 64
N_PULSES = 32


def deramped_cube(range_bin: int, doppler_bin: int) -> np.ndarray:
    """Return a cube holding one target placed exactly on the named bins.

    The target's range and velocity are read back out of
    :func:`range_bin_centers_m` and :func:`doppler_bin_centers_mps`, so the
    expected answer is an exact FFT bin by construction rather than something
    that has to be located to within a tolerance.
    """
    range_m = range_bin_centers_m(N_SAMPLES, **NOMINAL)[range_bin]
    velocity_mps = doppler_bin_centers_mps(N_PULSES, PULSE_REPETITION_INTERVAL_S, WAVELENGTH_M)[
        doppler_bin
    ]

    beat_hz = beat_frequency_hz(range_m, NOMINAL["bandwidth_hz"], NOMINAL["chirp_duration_s"])
    doppler_hz = 2.0 * velocity_mps / WAVELENGTH_M

    fast_time_s = np.arange(N_SAMPLES) / NOMINAL["sample_rate_hz"]
    slow_time_s = np.arange(N_PULSES) * PULSE_REPETITION_INTERVAL_S
    fast_phase = np.exp(2j * np.pi * beat_hz * fast_time_s)
    slow_phase = np.exp(2j * np.pi * doppler_hz * slow_time_s)
    return slow_phase[:, None] * fast_phase[None, :]


class TestMatchedFilter:
    def test_matches_numpy_correlate(self) -> None:
        """Catches a circular correlation, a missing conjugate or reversal, or a reversed output.

        The echo is the chirp delayed by 17 samples and scaled, not the chirp
        itself: an autocorrelation is Hermitian-symmetric, so against
        ``samples == reference`` an output returned time-reversed and
        conjugated would still match. With a delay it cannot, and the peak
        must sit at zero lag (index n_reference - 1) plus the delay.
        """
        chirp = lfm_chirp(bandwidth_hz=10e6, chirp_duration_s=10e-6, sample_rate_hz=20e6)
        delay_samples = 17
        echo = 0.5j * np.concatenate([np.zeros(delay_samples, dtype=np.complex128), chirp])
        compressed = matched_filter(echo, chirp)
        expected = np.correlate(echo, chirp, mode="full")
        # atol, not rtol alone: a correlation has exact nulls, where the true
        # value is zero and a relative tolerance is meaningless. The FFT and
        # direct methods land on ~1e-13 there by different routes. atol is set
        # to 1e-12 of the peak (about chirp.size), so a genuine error anywhere
        # near the mainlobe still fails by orders of magnitude.
        np.testing.assert_allclose(compressed, expected, rtol=1e-10, atol=chirp.size * 1e-12)
        assert int(np.argmax(np.abs(compressed))) == chirp.size - 1 + delay_samples

    def test_compresses_each_chirp_of_a_cube_independently(self) -> None:
        """Compressing along an axis must equal compressing row by row."""
        chirp = lfm_chirp(bandwidth_hz=10e6, chirp_duration_s=10e-6, sample_rate_hz=20e6)
        cube = np.stack([chirp, 2.0 * chirp, -1j * chirp])
        compressed = matched_filter(cube, chirp, axis=1)
        for row in range(cube.shape[0]):
            np.testing.assert_allclose(
                compressed[row], matched_filter(cube[row], chirp), rtol=1e-10
            )

    def test_rejects_two_dimensional_reference(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            matched_filter(np.ones(8), np.ones((2, 4)))

    def test_rejects_empty_reference(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            matched_filter(np.ones(8), np.array([]))

    def test_rejects_out_of_range_axis(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            matched_filter(np.ones(8), np.ones(4), axis=3)


class TestRangeFft:
    def test_places_an_exact_tone_on_its_exact_bin(self) -> None:
        """Catches an ifft (bin N - k), a stray fftshift, or a hidden 1/N normalisation.

        A unit tone at k*fs/N belongs in bin k with magnitude exactly N, and
        every other bin is a numerical null.
        """
        n_samples = 64
        for bin_index in (1, 7, 31):
            tone = np.exp(2j * np.pi * bin_index * np.arange(n_samples) / n_samples)
            spectrum = np.abs(range_fft(tone))
            assert int(np.argmax(spectrum)) == bin_index
            # rtol 1e-10: an FFT round-off budget, per docs/conventions/testing.md.
            np.testing.assert_allclose(spectrum[bin_index], float(n_samples), rtol=1e-10)
            # Every other bin is a numerical null, not a small number: the
            # leakage is pure float error, so assert it is 200 dB down.
            others = np.delete(spectrum, bin_index)
            assert others.max() < spectrum[bin_index] * 1e-10

    def test_zero_padding_interpolates_without_moving_the_peak(self) -> None:
        """Zero-padding refines the grid; it does not add resolution."""
        n_samples = 64
        tone = np.exp(2j * np.pi * 7 * np.arange(n_samples) / n_samples)
        padded = np.abs(range_fft(tone, n_fft=256))
        # Bin 7 of 64 is bin 28 of 256 — the same frequency, a finer grid.
        assert int(np.argmax(padded)) == 28

    def test_applies_a_window(self) -> None:
        """A tapered transform must differ from an untapered one, and fall off."""
        rng = np.random.default_rng(20260911)
        samples = rng.standard_normal(32) + 1j * rng.standard_normal(32)
        window = taper("hann", 32)
        np.testing.assert_allclose(
            range_fft(samples, window=window), range_fft(samples * window), rtol=1e-10
        )

    def test_rejects_truncating_transform_length(self) -> None:
        """Silently dropping samples would be a quiet loss of energy."""
        with pytest.raises(ValueError, match="shorter than samples"):
            range_fft(np.ones(64), n_fft=32)

    def test_rejects_a_zero_transform_length(self) -> None:
        """Zero is the boundary of the ``n_fft >= 1`` guard."""
        with pytest.raises(ValueError, match="at least one"):
            range_fft(np.ones(64), n_fft=0)

    def test_rejects_mismatched_window(self) -> None:
        with pytest.raises(ValueError, match="must match samples"):
            range_fft(np.ones(64), window=taper("hann", 32))


class TestDopplerFft:
    def test_separates_closing_from_opening(self) -> None:
        """Catches a missing fftshift, a flipped Doppler sign, or a hidden normalisation.

        With zero Doppler at bin N/2 = 16, a closing tone advancing +3/32 of a
        turn per chirp lands on bin 19 and the opening one on bin 13, each with
        magnitude N. Without the shift they would land on 3 and 29; with the
        sign flipped, on 13 and 19.
        """
        n_pulses = 32
        closing = np.exp(2j * np.pi * 3 * np.arange(n_pulses) / n_pulses)
        opening = np.exp(-2j * np.pi * 3 * np.arange(n_pulses) / n_pulses)
        closing_spectrum = np.abs(doppler_fft(closing, axis=0))
        opening_spectrum = np.abs(doppler_fft(opening, axis=0))
        assert int(np.argmax(closing_spectrum)) == n_pulses // 2 + 3
        assert int(np.argmax(opening_spectrum)) == n_pulses // 2 - 3
        # rtol 1e-10: an FFT round-off budget.
        np.testing.assert_allclose(closing_spectrum.max(), float(n_pulses), rtol=1e-10)


class TestBinCenters:
    def test_range_bin_spacing_matches_the_deramp_relation(self) -> None:
        r"""Bin k sits at k c f_s / (2 alpha N), an exact closed form starting at zero.

        Catches a one-way (c f_s / alpha N) relation, an offset first bin, and
        the Notes' unambiguous limit: bin N/2 is R_max = c f_s T / (4 B).
        """
        bins_m = range_bin_centers_m(N_SAMPLES, **NOMINAL)
        sweep_rate = NOMINAL["bandwidth_hz"] / NOMINAL["chirp_duration_s"]
        # Independent re-derivation, not a golden number.
        from radar_forge.core import SPEED_OF_LIGHT_MPS

        spacing_m = SPEED_OF_LIGHT_MPS * NOMINAL["sample_rate_hz"] / (2.0 * sweep_rate * N_SAMPLES)
        # rtol 1e-12: a handful of float64 operations.
        np.testing.assert_allclose(bins_m, np.arange(N_SAMPLES) * spacing_m, rtol=1e-12)
        max_unambiguous_range_m = (
            SPEED_OF_LIGHT_MPS
            * NOMINAL["sample_rate_hz"]
            * NOMINAL["chirp_duration_s"]
            / (4.0 * NOMINAL["bandwidth_hz"])
        )
        np.testing.assert_allclose(bins_m[N_SAMPLES // 2], max_unambiguous_range_m, rtol=1e-12)

    def test_doppler_span_is_the_unambiguous_velocity(self) -> None:
        r"""Bin k is (k - N/2) lambda / (2 N T_PRI): zero at N/2, -lambda/(4 T_PRI) at 0.

        Catches a one-way (lambda f_d) relation, the wrong sign convention, an
        uncentred axis, or the top bin placed at +v_max rather than one step
        short of it, the usual FFT asymmetry.
        """
        n_bins = 32
        velocities = doppler_bin_centers_mps(n_bins, PULSE_REPETITION_INTERVAL_S, WAVELENGTH_M)
        step_mps = WAVELENGTH_M / (2.0 * n_bins * PULSE_REPETITION_INTERVAL_S)
        expected_mps = (np.arange(n_bins) - n_bins // 2) * step_mps
        # rtol 1e-12, with atol for the zero-Doppler bin, where rtol is meaningless.
        np.testing.assert_allclose(velocities, expected_mps, rtol=1e-12, atol=1e-12)
        unambiguous_mps = WAVELENGTH_M / (4.0 * PULSE_REPETITION_INTERVAL_S)
        np.testing.assert_allclose(velocities[0], -unambiguous_mps, rtol=1e-12)

    def test_rejects_a_zero_bin_count(self) -> None:
        """Zero is the boundary of both helpers' ``n_bins >= 1`` guard."""
        with pytest.raises(ValueError, match="at least one"):
            range_bin_centers_m(0, **NOMINAL)
        with pytest.raises(ValueError, match="at least one"):
            doppler_bin_centers_mps(0, PULSE_REPETITION_INTERVAL_S, WAVELENGTH_M)

    def test_rejects_non_positive_wavelength(self) -> None:
        with pytest.raises(ValueError, match="wavelength_m"):
            doppler_bin_centers_mps(8, PULSE_REPETITION_INTERVAL_S, 0.0)

    def test_rejects_non_positive_pri(self) -> None:
        with pytest.raises(ValueError, match="pulse_repetition_interval_s"):
            doppler_bin_centers_mps(8, 0.0, WAVELENGTH_M)


class TestMtiFilter:
    def test_cancels_a_stationary_return(self) -> None:
        """Clutter that did not move subtracts to exactly zero."""
        stationary = np.ones((8, 4), dtype=np.complex128)
        np.testing.assert_allclose(mti_filter(stationary), 0.0, atol=1e-15)

    def test_shortens_the_dwell(self) -> None:
        """Differencing consumes chirps; the docstring promises how many."""
        assert mti_filter(np.ones((128, 4)), n_pulses=2).shape == (127, 4)
        assert mti_filter(np.ones((128, 4)), n_pulses=3).shape == (126, 4)

    def test_matches_the_analytic_frequency_response(self) -> None:
        r"""|H(f_d)| = 2|sin(pi f_d T_PRI)| for the single canceller."""
        n_pulses = 64
        for doppler_fraction in (0.1, 0.25, 0.4):
            doppler_hz = doppler_fraction / PULSE_REPETITION_INTERVAL_S
            slow_time_s = np.arange(n_pulses) * PULSE_REPETITION_INTERVAL_S
            tone = np.exp(2j * np.pi * doppler_hz * slow_time_s)
            gain = np.abs(mti_filter(tone, n_pulses=2)).mean()
            expected = 2.0 * np.abs(np.sin(np.pi * doppler_hz * PULSE_REPETITION_INTERVAL_S))
            # rtol 1e-10: a subtraction and a magnitude in float64.
            np.testing.assert_allclose(gain, expected, rtol=1e-10)

    def test_nulls_the_blind_speed(self) -> None:
        """A target advancing a full turn per chirp is invisible to MTI.

        This is a property of the canceller, not a defect in it: staggering the
        PRI is the standard cure and is deliberately not implemented here.
        """
        n_pulses = 32
        blind_doppler_hz = 1.0 / PULSE_REPETITION_INTERVAL_S
        slow_time_s = np.arange(n_pulses) * PULSE_REPETITION_INTERVAL_S
        blind = np.exp(2j * np.pi * blind_doppler_hz * slow_time_s)
        np.testing.assert_allclose(mti_filter(blind), 0.0, atol=1e-12)

    def test_double_canceller_matches_its_analytic_response(self) -> None:
        r"""|H(f_d)| = 4 sin^2(pi f_d T_PRI) for the [1, -2, 1] canceller.

        Catches wrong taps (an [1, -1, 1] or [1, -2, -1] kernel) that the
        single-canceller response cannot see. At the creeping 0.01 PRF used
        here the gain is 3.9e-3, against 6.3e-2 for the single canceller: the
        deeper notch is what the second stage buys.
        """
        n_pulses = 32
        slow_doppler_hz = 0.01 / PULSE_REPETITION_INTERVAL_S
        slow_time_s = np.arange(n_pulses) * PULSE_REPETITION_INTERVAL_S
        creeping = np.exp(2j * np.pi * slow_doppler_hz * slow_time_s)
        expected = 4.0 * np.sin(np.pi * slow_doppler_hz * PULSE_REPETITION_INTERVAL_S) ** 2
        # rtol 1e-10: two subtractions and a magnitude in float64, on a gain of
        # 4e-3 -- no cancellation beyond what the closed form also carries.
        np.testing.assert_allclose(np.abs(mti_filter(creeping, n_pulses=3)), expected, rtol=1e-10)

    @pytest.mark.parametrize("bad_n_pulses", [1, 4], ids=["below", "above"])
    def test_rejects_unsupported_order(self, bad_n_pulses: int) -> None:
        """One order either side of {2, 3}: catches a one-sided range check."""
        with pytest.raises(ValueError, match="must be one of"):
            mti_filter(np.ones((8, 4)), n_pulses=bad_n_pulses)

    def test_rejects_too_short_a_dwell(self) -> None:
        with pytest.raises(ValueError, match="fewer than"):
            mti_filter(np.ones((2, 4)), n_pulses=3)


class TestRangeDopplerMap:
    def test_places_a_target_on_its_exact_bins(self) -> None:
        """End to end: a target built from the bin centres lands on those bins."""
        cube = deramped_cube(range_bin=10, doppler_bin=20)
        rd_map = np.abs(range_doppler_map(cube, fast_time_axis=1, slow_time_axis=0))
        doppler_index, range_index = np.unravel_index(int(np.argmax(rd_map)), rd_map.shape)
        assert (int(doppler_index), int(range_index)) == (20, 10)

    def test_equals_the_two_transforms_in_sequence(self) -> None:
        """Catches the wrapper swapping or dropping a window or a transform length.

        Two different tapers and two different padded lengths, so a fast-time
        window handed to the slow-time transform (or ignored) cannot pass.
        """
        cube = deramped_cube(range_bin=10, doppler_bin=20)
        fast_window = taper("hann", N_SAMPLES)
        slow_window = taper("hamming", N_PULSES)
        expected = doppler_fft(
            range_fft(cube, axis=1, window=fast_window, n_fft=2 * N_SAMPLES),
            axis=0,
            window=slow_window,
            n_fft=2 * N_PULSES,
        )
        np.testing.assert_allclose(
            range_doppler_map(
                cube,
                fast_time_window=fast_window,
                slow_time_window=slow_window,
                n_range_fft=2 * N_SAMPLES,
                n_doppler_fft=2 * N_PULSES,
                fast_time_axis=1,
                slow_time_axis=0,
            ),
            expected,
            rtol=1e-10,
            atol=float(N_PULSES * N_SAMPLES) * 1e-12,
        )

    def test_rejects_aliased_repeated_axis(self) -> None:
        """-1 and 1 are the same axis of a 2-D cube; say so rather than aliasing."""
        with pytest.raises(ValueError, match="different axes"):
            range_doppler_map(np.ones((8, 8)), fast_time_axis=-1, slow_time_axis=1)


class TestStraddleLoss:
    """What an off-bin target costs, against the closed-form Dirichlet kernel."""

    def test_a_half_bin_straddle_costs_exactly_3_92_decibels(self) -> None:
        """The worst-case scalloping loss of an unwindowed FFT is 20log10(2/pi).

        A tone exactly between two bins splits its energy, and the peak the map
        reports is the Dirichlet kernel evaluated at half a bin: sin(pi/2)/(N
        sin(pi/2N)) -> 2/pi = -3.9224 dB as N grows. This is the number a
        detection budget must carry for an arbitrarily placed target, and it is
        also the sharpest available check that `range_fft` neither normalises
        nor windows behind the caller's back.
        """
        n_samples = 4096
        index = np.arange(n_samples)
        on_bin = np.exp(2j * np.pi * 64.0 * index / n_samples)
        half_bin = np.exp(2j * np.pi * 64.5 * index / n_samples)

        peak_on = np.abs(range_fft(on_bin)).max()
        peak_off = np.abs(range_fft(half_bin)).max()
        loss_db = 20.0 * np.log10(peak_off / peak_on)

        # The exact finite-N value, not the 2/pi asymptote: at N = 4096 the two
        # differ in the fifth decimal, and rtol 1e-9 keeps the test a statement
        # about the transform rather than about the limit.
        exact_db = 20.0 * np.log10(
            np.sin(np.pi / 2.0) / (n_samples * np.sin(np.pi / (2.0 * n_samples)))
        )
        np.testing.assert_allclose(loss_db, exact_db, rtol=1e-9)


class TestBinCenterValidation:
    """The documented Raises of the bin-centre helpers."""

    def test_rejects_a_non_positive_sample_rate(self) -> None:
        """A zero sample rate collapses every bin onto zero range, silently.

        The guard is the difference between an error and a range axis of all
        zeros that looks like a target at the radar.
        """
        with pytest.raises(ValueError, match="sample_rate_hz must be strictly positive"):
            range_bin_centers_m(8, bandwidth_hz=1e9, chirp_duration_s=4e-5, sample_rate_hz=0.0)
