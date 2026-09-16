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
        """Cross-check against the direct correlation used in test_waveforms."""
        chirp = lfm_chirp(bandwidth_hz=10e6, chirp_duration_s=10e-6, sample_rate_hz=20e6)
        expected = np.correlate(chirp, chirp, mode="full")
        # atol, not rtol alone: an autocorrelation has exact nulls, where the
        # true value is zero and a relative tolerance is meaningless. The FFT
        # and direct methods land on ~1e-13 there by different routes. atol is
        # set to 1e-12 of the peak (which is chirp.size), so a genuine error
        # anywhere near the mainlobe still fails by orders of magnitude.
        np.testing.assert_allclose(
            matched_filter(chirp, chirp), expected, rtol=1e-10, atol=chirp.size * 1e-12
        )

    def test_peaks_at_zero_lag_with_the_pulse_energy(self) -> None:
        """A matched filter's peak is the energy of the pulse it matches."""
        chirp = lfm_chirp(bandwidth_hz=10e6, chirp_duration_s=10e-6, sample_rate_hz=20e6)
        compressed = matched_filter(chirp, chirp)
        assert int(np.argmax(np.abs(compressed))) == chirp.size - 1
        # A unit-modulus chirp of N samples carries energy N exactly.
        np.testing.assert_allclose(
            np.abs(compressed[chirp.size - 1]), float(chirp.size), rtol=1e-10
        )

    def test_compresses_by_the_time_bandwidth_product(self) -> None:
        """The -3 dB mainlobe narrows from T to about 1/B, a factor of BT."""
        # Oversampled 20x: at fs = 2B the compressed mainlobe is under two
        # samples wide, so counting samples above -3 dB cannot measure it at all.
        bandwidth_hz, chirp_duration_s, sample_rate_hz = 10e6, 10e-6, 200e6
        chirp = lfm_chirp(bandwidth_hz, chirp_duration_s, sample_rate_hz)
        compressed = np.abs(matched_filter(chirp, chirp))
        peak_index = int(np.argmax(compressed))
        above_half_power = compressed >= compressed[peak_index] / np.sqrt(2.0)
        width_s = int(np.sum(above_half_power)) / sample_rate_hz
        # The exact unweighted-LFM -3 dB width is 0.886/B (Richards FRSP 2e
        # eq. 8.29). A +/-20% band around that is tight enough to catch a
        # compression that is not happening, and loose enough for the residual
        # sample quantisation.
        np.testing.assert_allclose(width_s, 0.886 / bandwidth_hz, rtol=0.2)

    def test_delays_the_peak_by_the_target_delay(self) -> None:
        """A target delayed by k samples moves the peak by exactly k bins."""
        chirp = lfm_chirp(bandwidth_hz=10e6, chirp_duration_s=10e-6, sample_rate_hz=20e6)
        delay_samples = 17
        echo = np.concatenate([np.zeros(delay_samples, dtype=np.complex128), chirp])
        peak = int(np.argmax(np.abs(matched_filter(echo, chirp))))
        assert peak == chirp.size - 1 + delay_samples

    def test_is_linear_in_two_targets(self) -> None:
        """Superposition: compression of a sum is the sum of compressions."""
        chirp = lfm_chirp(bandwidth_hz=10e6, chirp_duration_s=10e-6, sample_rate_hz=20e6)
        pad = np.zeros(20, dtype=np.complex128)
        near = np.concatenate([chirp, pad])
        far = np.concatenate([pad, 0.5 * chirp])
        # atol at 1e-12 of the peak, for the null bins; see the cross-check test.
        np.testing.assert_allclose(
            matched_filter(near + far, chirp),
            matched_filter(near, chirp) + matched_filter(far, chirp),
            rtol=1e-10,
            atol=chirp.size * 1e-12,
        )

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
        """A tone at k*fs/N belongs in bin k and nowhere else."""
        n_samples = 64
        for bin_index in (1, 7, 31):
            tone = np.exp(2j * np.pi * bin_index * np.arange(n_samples) / n_samples)
            spectrum = np.abs(range_fft(tone))
            assert int(np.argmax(spectrum)) == bin_index
            # Every other bin is a numerical null, not a small number: the
            # leakage is pure float error, so assert it is 200 dB down.
            others = np.delete(spectrum, bin_index)
            assert others.max() < spectrum[bin_index] * 1e-10

    def test_is_not_shifted(self) -> None:
        """Zero beat frequency, hence zero range, must stay at bin 0."""
        dc = np.ones(16, dtype=np.complex128)
        assert int(np.argmax(np.abs(range_fft(dc)))) == 0

    def test_conserves_energy(self) -> None:
        """Parseval across the range transform."""
        rng = np.random.default_rng(20260911)
        samples = rng.standard_normal(64) + 1j * rng.standard_normal(64)
        spectrum = range_fft(samples)
        # rtol 1e-10: one forward FFT, per the tolerance table in testing.md.
        np.testing.assert_allclose(
            np.sum(np.abs(spectrum) ** 2) / samples.size, np.sum(np.abs(samples) ** 2), rtol=1e-10
        )

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

    def test_window_suppresses_sidelobes_of_an_off_bin_target(self) -> None:
        """The reason windowing exists: a half-bin target leaks without it."""
        n_samples = 64
        # Half a bin off centre is the worst case for spectral leakage.
        tone = np.exp(2j * np.pi * 7.5 * np.arange(n_samples) / n_samples)
        untapered = np.abs(range_fft(tone))
        tapered = np.abs(range_fft(tone, window=taper("blackmanharris", n_samples)))
        far_bins = slice(20, 44)
        assert tapered[far_bins].max() < untapered[far_bins].max()

    def test_rejects_truncating_transform_length(self) -> None:
        """Silently dropping samples would be a quiet loss of energy."""
        with pytest.raises(ValueError, match="shorter than samples"):
            range_fft(np.ones(64), n_fft=32)

    @pytest.mark.parametrize("bad_n_fft", [0, -8])
    def test_rejects_non_positive_transform_length(self, bad_n_fft: int) -> None:
        with pytest.raises(ValueError, match="at least one"):
            range_fft(np.ones(64), n_fft=bad_n_fft)

    def test_rejects_mismatched_window(self) -> None:
        with pytest.raises(ValueError, match="must match samples"):
            range_fft(np.ones(64), window=taper("hann", 32))


class TestDopplerFft:
    def test_puts_zero_doppler_at_the_centre(self) -> None:
        """A stationary target has no phase advance between chirps."""
        stationary = np.ones(16, dtype=np.complex128)
        assert int(np.argmax(np.abs(doppler_fft(stationary, axis=0)))) == 8

    def test_separates_closing_from_opening(self) -> None:
        """Closing targets sit above centre, opening targets below."""
        n_pulses = 32
        closing = np.exp(2j * np.pi * 3 * np.arange(n_pulses) / n_pulses)
        opening = np.exp(-2j * np.pi * 3 * np.arange(n_pulses) / n_pulses)
        assert int(np.argmax(np.abs(doppler_fft(closing, axis=0)))) > n_pulses // 2
        assert int(np.argmax(np.abs(doppler_fft(opening, axis=0)))) < n_pulses // 2

    def test_conserves_energy(self) -> None:
        """Parseval survives the fftshift, which only permutes bins."""
        rng = np.random.default_rng(20260911)
        samples = rng.standard_normal(32) + 1j * rng.standard_normal(32)
        spectrum = doppler_fft(samples, axis=0)
        np.testing.assert_allclose(
            np.sum(np.abs(spectrum) ** 2) / samples.size, np.sum(np.abs(samples) ** 2), rtol=1e-10
        )

    def test_transforms_slow_time_of_a_cube(self) -> None:
        cube = np.ones((16, 8), dtype=np.complex128)
        spectrum = np.abs(doppler_fft(cube, axis=0))
        assert spectrum.shape == (16, 8)
        # Stationary everywhere, so every range bin peaks at centre Doppler.
        assert np.all(np.argmax(spectrum, axis=0) == 8)


class TestBinCenters:
    def test_range_bins_start_at_zero_and_increase(self) -> None:
        bins_m = range_bin_centers_m(N_SAMPLES, **NOMINAL)
        np.testing.assert_allclose(bins_m[0], 0.0, atol=1e-12)
        assert np.all(np.diff(bins_m) > 0.0)

    def test_range_bin_spacing_matches_the_deramp_relation(self) -> None:
        r"""Bin spacing is c*fs/(2*alpha*N), an exact closed form."""
        bins_m = range_bin_centers_m(N_SAMPLES, **NOMINAL)
        sweep_rate = NOMINAL["bandwidth_hz"] / NOMINAL["chirp_duration_s"]
        # Independent re-derivation, not a golden number.
        from radar_forge.core import SPEED_OF_LIGHT_MPS

        expected = SPEED_OF_LIGHT_MPS * NOMINAL["sample_rate_hz"] / (2.0 * sweep_rate * N_SAMPLES)
        # rtol 1e-12: a handful of float64 operations.
        np.testing.assert_allclose(np.diff(bins_m), expected, rtol=1e-12)

    def test_doppler_bins_are_centred_on_zero(self) -> None:
        velocities = doppler_bin_centers_mps(8, PULSE_REPETITION_INTERVAL_S, WAVELENGTH_M)
        np.testing.assert_allclose(velocities[4], 0.0, atol=1e-15)
        assert np.all(np.diff(velocities) > 0.0)

    def test_doppler_span_is_the_unambiguous_velocity(self) -> None:
        r"""The axis spans exactly \pm \lambda/(4 T_PRI)."""
        n_bins = 32
        velocities = doppler_bin_centers_mps(n_bins, PULSE_REPETITION_INTERVAL_S, WAVELENGTH_M)
        unambiguous_mps = WAVELENGTH_M / (4.0 * PULSE_REPETITION_INTERVAL_S)
        np.testing.assert_allclose(velocities[0], -unambiguous_mps, rtol=1e-12)
        # The top bin is one step short of +v_max, the usual FFT asymmetry.
        np.testing.assert_allclose(
            velocities[-1], unambiguous_mps * (1.0 - 2.0 / n_bins), rtol=1e-12
        )

    @pytest.mark.parametrize("bad_n_bins", [0, -4])
    def test_rejects_non_positive_bin_count(self, bad_n_bins: int) -> None:
        with pytest.raises(ValueError, match="at least one"):
            range_bin_centers_m(bad_n_bins, **NOMINAL)
        with pytest.raises(ValueError, match="at least one"):
            doppler_bin_centers_mps(bad_n_bins, PULSE_REPETITION_INTERVAL_S, WAVELENGTH_M)

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

    def test_passes_the_optimum_doppler(self) -> None:
        """Half the PRF is the peak of the canceller's response, a gain of 2."""
        n_pulses = 32
        doppler_hz = 0.5 / PULSE_REPETITION_INTERVAL_S
        slow_time_s = np.arange(n_pulses) * PULSE_REPETITION_INTERVAL_S
        tone = np.exp(2j * np.pi * doppler_hz * slow_time_s)
        np.testing.assert_allclose(np.abs(mti_filter(tone)), 2.0, rtol=1e-10)

    def test_suppresses_clutter_under_a_moving_target(self) -> None:
        """The operational claim: strong clutter falls far below a weak target."""
        clutter = deramped_cube(range_bin=10, doppler_bin=N_PULSES // 2) * 1000.0
        target = deramped_cube(range_bin=10, doppler_bin=N_PULSES // 2 + 8)
        before = range_doppler_map(clutter + target, fast_time_axis=1, slow_time_axis=0)
        after = range_doppler_map(mti_filter(clutter + target), fast_time_axis=1, slow_time_axis=0)
        # Clutter dominates by 60 dB before the canceller and must not after.
        assert np.abs(before).max() / np.abs(before[N_PULSES // 2 + 8]).max() > 10.0
        assert int(np.argmax(np.abs(after)) // after.shape[1]) != after.shape[0] // 2

    def test_double_canceller_nulls_harder(self) -> None:
        """Two cancellers stack, so the near-zero-Doppler notch deepens."""
        n_pulses = 32
        slow_doppler_hz = 0.01 / PULSE_REPETITION_INTERVAL_S
        slow_time_s = np.arange(n_pulses) * PULSE_REPETITION_INTERVAL_S
        creeping = np.exp(2j * np.pi * slow_doppler_hz * slow_time_s)
        assert (
            np.abs(mti_filter(creeping, n_pulses=3)).max()
            < np.abs(mti_filter(creeping, n_pulses=2)).max()
        )

    @pytest.mark.parametrize("bad_n_pulses", [1, 4, 0])
    def test_rejects_unsupported_order(self, bad_n_pulses: int) -> None:
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

    def test_separates_two_targets_at_the_same_range(self) -> None:
        """Doppler resolves what range cannot: same bin, different velocity."""
        cube = deramped_cube(range_bin=15, doppler_bin=8) + deramped_cube(
            range_bin=15, doppler_bin=24
        )
        rd_map = np.abs(range_doppler_map(cube, fast_time_axis=1, slow_time_axis=0))
        column = rd_map[:, 15]
        assert int(np.argmax(column)) in (8, 24)
        # Both peaks stand far above the rest of the Doppler column.
        peaks = np.sort(column)[-2:]
        assert peaks.min() > np.median(column) * 100.0

    def test_equals_the_two_transforms_in_sequence(self) -> None:
        """The convenience wrapper must not diverge from its parts."""
        cube = deramped_cube(range_bin=10, doppler_bin=20)
        expected = doppler_fft(range_fft(cube, axis=1), axis=0)
        np.testing.assert_allclose(
            range_doppler_map(cube, fast_time_axis=1, slow_time_axis=0),
            expected,
            rtol=1e-10,
            atol=float(N_PULSES * N_SAMPLES) * 1e-12,
        )

    def test_transforms_commute(self) -> None:
        """They act on different axes, so the order is convention only."""
        cube = deramped_cube(range_bin=10, doppler_bin=20)
        # The map peaks at n_pulses * n_samples; every other bin is an exact
        # null, so compare against an atol pegged to that peak rather than a
        # relative tolerance on numbers whose true value is zero.
        peak = float(N_PULSES * N_SAMPLES)
        np.testing.assert_allclose(
            doppler_fft(range_fft(cube, axis=1), axis=0),
            range_fft(doppler_fft(cube, axis=0), axis=1),
            rtol=1e-10,
            atol=peak * 1e-12,
        )

    def test_rejects_a_repeated_axis(self) -> None:
        with pytest.raises(ValueError, match="different axes"):
            range_doppler_map(np.ones((8, 8)), fast_time_axis=0, slow_time_axis=0)

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

    def test_an_on_bin_tone_suffers_no_loss_at_all(self) -> None:
        """The other end of the same curve: a bin-centred tone keeps all its amplitude.

        An N-point FFT of a unit tone on bin k has peak magnitude exactly N, so
        this pins the transform's scaling as well as the absence of straddling.
        """
        n_samples = 1024
        tone = np.exp(2j * np.pi * 37.0 * np.arange(n_samples) / n_samples)
        # rtol 1e-10: an FFT round-off budget, per docs/conventions/testing.md.
        np.testing.assert_allclose(np.abs(range_fft(tone)).max(), float(n_samples), rtol=1e-10)


class TestBinCenterValidation:
    """The documented Raises of the bin-centre helpers."""

    def test_rejects_a_non_positive_sample_rate(self) -> None:
        """A zero sample rate collapses every bin onto zero range, silently.

        The guard is the difference between an error and a range axis of all
        zeros that looks like a target at the radar.
        """
        with pytest.raises(ValueError, match="sample_rate_hz must be strictly positive"):
            range_bin_centers_m(8, bandwidth_hz=1e9, chirp_duration_s=4e-5, sample_rate_hz=0.0)

    def test_rejects_a_negative_sample_rate(self) -> None:
        """A negative rate would run the range axis backwards into negative range."""
        with pytest.raises(ValueError, match="sample_rate_hz must be strictly positive"):
            range_bin_centers_m(8, bandwidth_hz=1e9, chirp_duration_s=4e-5, sample_rate_hz=-1e6)
