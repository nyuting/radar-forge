"""Tests for range-Doppler processing: windows, compression, the FFTs, MTI and unfolding.

Ground truth is closed form throughout: a tone on an exact bin, Harris's
tabulated window figures, Parseval's relation, the canceller's |H(f_d)|, and
folding as a known modular map. The axes are bistatic and the rate is positive
receding, so every sign test below states which way the target moves.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.dsp import (
    WINDOW_NAMES,
    apply_window,
    coherent_gain_linear,
    doppler_bin_centers_mps,
    doppler_fft,
    fold_into_period,
    fold_velocity_mps,
    matched_filter,
    mti_filter,
    processing_loss_db,
    range_bin_centers_m,
    range_doppler_map,
    range_fft,
    unfold_doppler_dual_prf,
    unfold_dual_period,
    unfold_range_dual_prf,
    window,
)

# A nominal 77 GHz automotive FMCW front end, deramped.
NOMINAL = {"bandwidth_hz": 1e9, "chirp_duration_s": 40e-6, "sample_rate_hz": 1e6}
WAVELENGTH_M = 3.896e-3
PULSE_REPETITION_INTERVAL_S = 50e-6
N_SAMPLES = 64
N_PULSES = 32
# Long enough that Harris's asymptotic sidelobe and loss figures are reached.
N_LONG = 4096
FIXED_WINDOWS = [name for name in WINDOW_NAMES if name not in ("taylor", "chebyshev")]
# Scenario 001 S3's two bursts: half-intervals in about 5:6, maps of 60 and 50 km.
V_UA_A, V_UA_B = 38.24, 45.89
SPAN_A_M, SPAN_B_M, REPEAT_M = 60_000.0, 50_000.0, 300_000.0


def any_window(name: str, n_samples: int, **kwargs: bool) -> np.ndarray:
    """Return ``window(name, n)``, giving the parametric shapes a -60 dB design."""
    level_db = -60.0 if name in ("taylor", "chebyshev") else None
    return window(name, n_samples, sidelobe_level_db=level_db, **kwargs)  # type: ignore[arg-type]


def peak_sidelobe_db(weights: np.ndarray) -> float:
    """Return the highest sidelobe of a window's spectrum, in dB below its peak.

    Zero-padded 32x so sidelobe peaks are resolved rather than straddled; the
    mainlobe is excised by walking out to the first local minimum.
    """
    spectrum = np.abs(np.fft.fft(weights, n=32 * weights.size))
    spectrum = spectrum / spectrum.max()
    index = 1
    while index < spectrum.size // 2 and spectrum[index + 1] < spectrum[index]:
        index += 1
    return float(20.0 * np.log10(spectrum[index : spectrum.size // 2].max()))


def tone(cycles_per_record: float, n_samples: int) -> np.ndarray:
    """Return a unit complex tone making ``cycles_per_record`` turns over the record."""
    return np.exp(2j * np.pi * cycles_per_record * np.arange(n_samples) / n_samples)


def lfm_pulse(bandwidth_hz: float, duration_s: float, sample_rate_hz: float) -> np.ndarray:
    """Return a baseband up-chirp, exp(j pi (B/T) t^2), centred on the pulse."""
    time_s = np.arange(round(duration_s * sample_rate_hz)) / sample_rate_hz - duration_s / 2.0
    return np.exp(1j * np.pi * (bandwidth_hz / duration_s) * time_s**2)


class TestWindow:
    @pytest.mark.parametrize("name", WINDOW_NAMES)
    def test_is_symmetric(self, name: str) -> None:
        """Catches the periodic (fftbins=True) form, which Harris's figures cannot see at N=4096."""
        weights = any_window(name, 33)
        # SciPy builds these from cosines, so the ends agree only to rounding.
        np.testing.assert_allclose(weights, weights[::-1], rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize(
        ("name", "expected_db"),
        [
            ("rectangular", -13.26),
            ("hann", -31.5),
            ("hamming", -42.7),
            ("blackman", -58.1),
            ("blackmanharris", -92.0),
        ],
    )
    def test_matches_harris_peak_sidelobe(self, name: str, expected_db: float) -> None:
        """Harris 1978, Table 1: catches the wrong SciPy shape behind a name."""
        # atol 0.5 dB: Harris tabulates to about 1 dB.
        np.testing.assert_allclose(peak_sidelobe_db(window(name, N_LONG)), expected_db, atol=0.5)

    @pytest.mark.parametrize(
        ("name", "expected_db"),
        [("hann", 1.76), ("hamming", 1.34), ("blackman", 2.37), ("blackmanharris", 3.02)],
    )
    def test_matches_harris_processing_loss(self, name: str, expected_db: float) -> None:
        """Harris 1978, Table 1, 10 log10(ENBW): catches a coherent-gain-only loss."""
        # atol 0.01 dB: Harris tabulates to two decimals; the ratio is closed form.
        np.testing.assert_allclose(processing_loss_db(window(name, N_LONG)), expected_db, atol=0.01)

    @pytest.mark.parametrize("level_db", [-30.0, -50.0, -70.0])
    def test_taylor_achieves_its_design_sidelobe(self, level_db: float) -> None:
        """Catches a fixed small nbar, which misses deep designs by many dB."""
        achieved_db = peak_sidelobe_db(window("taylor", N_LONG, sidelobe_level_db=level_db))
        # atol 0.5 dB: the near-in sidelobes are equalised at the design level.
        np.testing.assert_allclose(achieved_db, level_db, atol=0.5)

    def test_chebyshev_is_equiripple_at_its_design_level(self) -> None:
        """Catches the sign of the level being passed to chebwin the wrong way round."""
        achieved_db = peak_sidelobe_db(window("chebyshev", N_LONG, sidelobe_level_db=-60.0))
        np.testing.assert_allclose(achieved_db, -60.0, atol=0.5)

    @pytest.mark.parametrize("name", WINDOW_NAMES)
    def test_normalise_gives_unit_coherent_gain(self, name: str) -> None:
        """The definition of ``normalise``; and raw rectangular is exactly ones."""
        np.testing.assert_allclose(
            coherent_gain_linear(any_window(name, 128, normalise=True)), 1.0, rtol=1e-12
        )
        np.testing.assert_array_equal(window("rectangular", 10), np.ones(10))

    def test_loss_is_invariant_to_normalising(self) -> None:
        raw_db = processing_loss_db(window("blackman", 256))
        np.testing.assert_allclose(
            processing_loss_db(window("blackman", 256, normalise=True)), raw_db, rtol=1e-12
        )

    @pytest.mark.parametrize(
        ("name", "level_db", "match"),
        [
            ("kaiser", None, "need a name"),
            ("taylor", None, "required"),
            ("hann", -40.0, "refused"),
            ("taylor", 0.0, "must be negative"),
            ("chebyshev", -20.0, "chebyshev needs"),
        ],
    )
    def test_rejects_an_invalid_design(self, name: str, level_db: float | None, match: str) -> None:
        """A level a window cannot honour is an error, never silently ignored."""
        with pytest.raises(ValueError, match=match):
            window(name, 64, sidelobe_level_db=level_db)  # type: ignore[arg-type]

    def test_rejects_a_zero_length(self) -> None:
        with pytest.raises(ValueError, match="n_samples >= 1"):
            window("hann", 0)


class TestApplyWindow:
    @pytest.mark.parametrize("name", WINDOW_NAMES)
    def test_peak_scales_by_the_coherent_gain(self, name: str) -> None:
        """A tone on bin 5 peaks at N G_c after windowing: catches a hidden normalisation."""
        windowed = apply_window(tone(5, N_SAMPLES)[None, :], name, axis=1, **_level(name))
        expected = N_SAMPLES * coherent_gain_linear(any_window(name, N_SAMPLES))
        # rtol 1e-10: an FFT round-off budget.
        np.testing.assert_allclose(np.abs(range_fft(windowed))[0, 5], expected, rtol=1e-10)

    def test_weights_only_the_named_axis(self) -> None:
        """Catches the window landing on fast time when slow time was asked for."""
        cube = np.random.default_rng(20260911).standard_normal((6, 5))
        expected = cube * window("hamming", 6)[:, None]
        np.testing.assert_allclose(apply_window(cube, "hamming", axis=0), expected, rtol=1e-12)
        np.testing.assert_allclose(apply_window(cube, "hamming", axis=-2), expected, rtol=1e-12)

    def test_rejects_an_out_of_bounds_axis(self) -> None:
        with pytest.raises(ValueError, match="out of bounds"):
            apply_window(np.ones((4, 8)), "hann", axis=5)


def _level(name: str) -> dict[str, float]:
    """Return the ``sidelobe_level_db`` keyword the parametric windows need."""
    return {"sidelobe_level_db": -60.0} if name in ("taylor", "chebyshev") else {}


class TestWindowMetrics:
    @pytest.mark.parametrize("metric", [coherent_gain_linear, processing_loss_db])
    @pytest.mark.parametrize("weights", [np.ones((4, 4)), np.zeros(0)], ids=["2-d", "empty"])
    def test_rejects_a_malformed_window(self, metric, weights: np.ndarray) -> None:
        """Without the guard NumPy averages a whole cube, or returns nan for nothing."""
        with pytest.raises(ValueError, match="one-dimensional and non-empty"):
            metric(weights)

    def test_rejects_a_zero_sum_window(self) -> None:
        with pytest.raises(ValueError, match="sum to zero"):
            processing_loss_db(np.array([1.0, -1.0]))


class TestMatchedFilter:
    def test_matches_numpy_correlate_on_a_delayed_echo(self) -> None:
        """Catches circular correlation, a missing conjugate or reversal, or reversed output.

        A delayed, scaled echo rather than the pulse itself: an autocorrelation
        is Hermitian, so a reversed-and-conjugated output would still match it.
        """
        pulse = lfm_pulse(10e6, 10e-6, 20e6)
        echo = 0.5j * np.concatenate([np.zeros(17), pulse])
        # atol near the correlation's exact nulls, 1e-12 of the peak (about pulse.size).
        np.testing.assert_allclose(
            matched_filter(echo, pulse),
            np.correlate(echo, pulse, mode="full"),
            rtol=1e-10,
            atol=pulse.size * 1e-12,
        )

    def test_lfm_peaks_at_the_true_delay_with_a_13_3_db_sidelobe(self) -> None:
        """BT = 200, 16 samples per 1/B: the peak is at zero lag plus the delay, PSLR -13.26 dB.

        Catches an off-by-one in the lag convention and a windowed or mismatched
        reference (either drops the sidelobe far below -13 dB). Sampling 16x per
        resolution cell keeps the sidelobe peak from being straddled; the finite
        BT's Fresnel ripple leaves the measured -13.32 dB, so atol 0.1 dB.
        """
        pulse = lfm_pulse(5e6, 40e-6, 80e6)
        delay_samples = 37
        echo = np.concatenate([np.zeros(delay_samples), pulse, np.zeros(50)])
        response = np.abs(matched_filter(echo, pulse))
        peak = int(np.argmax(response))
        assert peak == pulse.size - 1 + delay_samples
        right, left = peak, peak
        while response[right + 1] < response[right]:
            right += 1
        while response[left - 1] < response[left]:
            left -= 1
        sidelobe = max(response[: left + 1].max(), response[right:].max())
        np.testing.assert_allclose(20.0 * np.log10(sidelobe / response[peak]), -13.26, atol=0.1)

    def test_compresses_each_row_of_a_cube_independently(self) -> None:
        pulse = lfm_pulse(10e6, 10e-6, 20e6)
        cube = np.stack([pulse, 2.0 * pulse, -1j * pulse])
        compressed = matched_filter(cube.T, pulse, axis=0).T
        for row in range(3):
            np.testing.assert_allclose(
                compressed[row], matched_filter(cube[row], pulse), rtol=1e-10
            )

    @pytest.mark.parametrize("reference", [np.ones((2, 4)), np.array([])], ids=["2-d", "empty"])
    def test_rejects_a_malformed_reference(self, reference: np.ndarray) -> None:
        with pytest.raises(ValueError, match="one-dimensional and non-empty"):
            matched_filter(np.ones(8), reference)


class TestRangeFft:
    @pytest.mark.parametrize("bin_index", [1, 7, 31])
    def test_places_an_exact_tone_on_its_exact_bin(self, bin_index: int) -> None:
        """Catches an ifft (bin N - k), a stray fftshift, or a hidden 1/N."""
        spectrum = np.abs(range_fft(tone(bin_index, N_SAMPLES)))
        # rtol 1e-10: an FFT round-off budget; every other bin is a 200 dB null.
        np.testing.assert_allclose(spectrum[bin_index], float(N_SAMPLES), rtol=1e-10)
        assert np.delete(spectrum, bin_index).max() < 1e-10 * N_SAMPLES

    def test_zero_padding_interpolates_without_moving_the_peak(self) -> None:
        """Bin 7 of 64 is bin 28 of 256: the same frequency on a finer grid."""
        assert int(np.argmax(np.abs(range_fft(tone(7, N_SAMPLES), n_fft=256)))) == 28

    def test_a_half_bin_straddle_costs_the_exact_dirichlet_loss(self) -> None:
        """A tone between bins splits its energy: the peak drops by sin(pi/2)/(N sin(pi/2N)).

        -3.92 dB at large N; catches a transform that windows or normalises
        behind the caller's back.
        """
        n_samples = 4096
        loss_db = 20.0 * np.log10(
            np.abs(range_fft(tone(64.5, n_samples))).max()
            / np.abs(range_fft(tone(64.0, n_samples))).max()
        )
        exact_db = 20.0 * np.log10(1.0 / (n_samples * np.sin(np.pi / (2.0 * n_samples))))
        np.testing.assert_allclose(loss_db, exact_db, rtol=1e-9)

    @pytest.mark.parametrize("transform", [range_fft, doppler_fft])
    @pytest.mark.parametrize("n_fft", [None, 2 * N_PULSES])
    def test_parseval_energy_out_is_n_fft_times_energy_in(self, transform, n_fft) -> None:
        """sum |X|^2 = N_fft sum |w x|^2: catches a hidden normalisation or a lost sample.

        Zero-padding adds no energy, so the factor is the transform length.
        """
        rng = np.random.default_rng(20260911)
        samples = rng.standard_normal((N_PULSES, 3)) + 1j * rng.standard_normal((N_PULSES, 3))
        weights = window("hann", N_PULSES)
        spectrum = transform(samples, n_fft=n_fft, weights=weights, axis=0)
        n_transform = n_fft or N_PULSES
        energy_in = np.sum(np.abs(samples * weights[:, None]) ** 2, axis=0)
        # rtol 1e-10: an FFT round-off budget.
        np.testing.assert_allclose(
            np.sum(np.abs(spectrum) ** 2, axis=0), n_transform * energy_in, rtol=1e-10
        )

    @pytest.mark.parametrize(
        ("n_fft", "weights", "match"),
        [(32, None, "not shorter"), (0, None, "at least one"), (None, np.ones(32), "must match")],
    )
    def test_rejects_an_invalid_transform(self, n_fft, weights, match: str) -> None:
        """Truncating would silently drop energy; a wrong-length window, misweight it."""
        with pytest.raises(ValueError, match=match):
            range_fft(np.ones(64), n_fft=n_fft, weights=weights)


class TestBinCenters:
    def test_range_bins_are_bistatic_c_tau_not_halved(self) -> None:
        r"""Bin k is c k f_s T / (B N), the range sum: catches a leftover monostatic 1/2.

        The span is one circle of c f_s T / B.
        """
        bins_m = range_bin_centers_m(N_SAMPLES, **NOMINAL)
        f_s, t_s, b_hz = (
            NOMINAL["sample_rate_hz"],
            NOMINAL["chirp_duration_s"],
            NOMINAL["bandwidth_hz"],
        )
        spacing_m = SPEED_OF_LIGHT_MPS * f_s * t_s / (b_hz * N_SAMPLES)
        # rtol 1e-12: a handful of float64 operations.
        np.testing.assert_allclose(bins_m, np.arange(N_SAMPLES) * spacing_m, rtol=1e-12)
        np.testing.assert_allclose(bins_m[-1] + spacing_m, SPEED_OF_LIGHT_MPS * f_s * t_s / b_hz)

    def test_rate_bins_are_positive_receding_and_closed_at_the_top(self) -> None:
        r"""Bin k is -(k - N/2) lambda / (N T): catches a closing-positive sign or a halving.

        Bin 0 holds f_d = -PRF/2, so its rate is +lambda/(2T): the interval is
        (-v_ua, +v_ua], and the array decreases.
        """
        rate_mps = doppler_bin_centers_mps(N_PULSES, PULSE_REPETITION_INTERVAL_S, WAVELENGTH_M)
        step_mps = WAVELENGTH_M / (N_PULSES * PULSE_REPETITION_INTERVAL_S)
        expected_mps = -(np.arange(N_PULSES) - N_PULSES // 2) * step_mps
        np.testing.assert_allclose(rate_mps, expected_mps, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(rate_mps[0], WAVELENGTH_M / (2.0 * PULSE_REPETITION_INTERVAL_S))

    def test_an_approaching_tone_lands_on_a_negative_rate(self) -> None:
        """A phase that advances +3/32 turn per pulse is f_d > 0, so approaching (rate < 0).

        Catches a missing fftshift (bin 3, not 19) and a flipped sign on either side.
        """
        peak = int(np.argmax(np.abs(doppler_fft(tone(3, N_PULSES)))))
        assert peak == N_PULSES // 2 + 3
        rate_mps = doppler_bin_centers_mps(N_PULSES, PULSE_REPETITION_INTERVAL_S, WAVELENGTH_M)
        assert rate_mps[peak] < 0.0

    @pytest.mark.parametrize("n_bins", [0, 8])
    def test_rejects_invalid_axis_parameters(self, n_bins: int) -> None:
        """n_bins = 0 is the boundary; at n_bins = 8 a zero sample rate or PRI is the fault."""
        bad = {**NOMINAL, "sample_rate_hz": 0.0} if n_bins else NOMINAL
        with pytest.raises(ValueError, match="positive"):
            range_bin_centers_m(n_bins, **bad)
        with pytest.raises(ValueError, match="positive"):
            doppler_bin_centers_mps(n_bins, 0.0 if n_bins else 1e-3, WAVELENGTH_M)

    def test_the_stationary_bin_is_positive_zero(self) -> None:
        """Negating 0 Hz gives -0.0, which prints as "-0.0" on an axis label."""
        rate_mps = doppler_bin_centers_mps(N_PULSES, PULSE_REPETITION_INTERVAL_S, WAVELENGTH_M)
        assert rate_mps[N_PULSES // 2] == 0.0
        assert not np.signbit(rate_mps[N_PULSES // 2])

    def test_an_odd_bin_count_is_symmetric_inside_the_interval(self) -> None:
        """N = 5: rates +-2 and +-1 steps of lambda/(N T), so no bin sits on +v_ua = 15 m/s."""
        np.testing.assert_allclose(
            doppler_bin_centers_mps(5, 1e-3, 0.03), [12.0, 6.0, 0.0, -6.0, -12.0], atol=1e-12
        )


class TestRangeDopplerMap:
    def deramped_cube(self, range_bin: int, rate_bin: int) -> np.ndarray:
        """Return a cube with one target on the named bins, built from the bin centres.

        Beat is alpha tau = (B/T) R / c for bistatic range R; Doppler is -rate/lambda.
        """
        range_m = range_bin_centers_m(N_SAMPLES, **NOMINAL)[range_bin]
        rate_mps = doppler_bin_centers_mps(N_PULSES, PULSE_REPETITION_INTERVAL_S, WAVELENGTH_M)
        beat_hz = (
            NOMINAL["bandwidth_hz"] / NOMINAL["chirp_duration_s"] * range_m / SPEED_OF_LIGHT_MPS
        )
        doppler_hz = -rate_mps[rate_bin] / WAVELENGTH_M
        fast_time_s = np.arange(N_SAMPLES) / NOMINAL["sample_rate_hz"]
        slow_time_s = np.arange(N_PULSES) * PULSE_REPETITION_INTERVAL_S
        return np.exp(2j * np.pi * (doppler_hz * slow_time_s[:, None] + beat_hz * fast_time_s))

    @pytest.mark.parametrize("rate_bin", [4, 20], ids=["receding", "approaching"])
    def test_places_a_target_on_its_exact_bins(self, rate_bin: int) -> None:
        """End to end, both signs: a target built from the axis labels lands on them."""
        rd_map = np.abs(range_doppler_map(self.deramped_cube(10, rate_bin)))
        assert np.unravel_index(int(np.argmax(rd_map)), rd_map.shape) == (rate_bin, 10)

    def test_equals_the_two_transforms_in_sequence(self) -> None:
        """Two windows and two lengths, so a swapped or dropped argument cannot pass."""
        cube = self.deramped_cube(10, 20)
        fast, slow = window("hann", N_SAMPLES), window("hamming", N_PULSES)
        expected = doppler_fft(
            range_fft(cube, weights=fast, n_fft=2 * N_SAMPLES), weights=slow, n_fft=2 * N_PULSES
        )
        actual = range_doppler_map(
            cube.T,
            fast_time_weights=fast,
            slow_time_weights=slow,
            n_range_fft=2 * N_SAMPLES,
            n_doppler_fft=2 * N_PULSES,
            fast_time_axis=0,
            slow_time_axis=1,
        )
        np.testing.assert_allclose(
            actual.T, expected, rtol=1e-10, atol=N_PULSES * N_SAMPLES * 1e-12
        )

    @pytest.mark.parametrize("axes", [(-1, 1), (0, 2)], ids=["aliased", "out-of-bounds"])
    def test_rejects_axes_that_are_not_two_different_valid_ones(self, axes) -> None:
        with pytest.raises(ValueError, match="two different valid axes"):
            range_doppler_map(np.ones((8, 8)), fast_time_axis=axes[0], slow_time_axis=axes[1])


class TestMtiFilter:
    def slow_time_tone(self, doppler_fraction_of_prf: float) -> np.ndarray:
        """Return a 32-pulse unit tone whose phase turns by the given fraction per pulse."""
        return np.exp(2j * np.pi * doppler_fraction_of_prf * np.arange(N_PULSES))

    @pytest.mark.parametrize("n_pulses", [2, 3])
    @pytest.mark.parametrize("doppler_fraction_of_prf", [0.0, 1.0], ids=["stationary", "blind"])
    def test_nulls_zero_doppler_and_the_blind_speed(
        self, n_pulses, doppler_fraction_of_prf
    ) -> None:
        """|H| = 0 at every multiple of the PRF, the blind speeds, not only at zero."""
        filtered = mti_filter(
            self.slow_time_tone(doppler_fraction_of_prf), n_pulses=n_pulses, axis=0
        )
        np.testing.assert_allclose(filtered, 0.0, atol=1e-12)

    @pytest.mark.parametrize(("n_pulses", "gain_linear"), [(2, 2.0), (3, 4.0)])
    def test_peak_gain_midway_between_blind_speeds(self, n_pulses: int, gain_linear: float) -> None:
        """At f_d = PRF/2 alternate pulses flip sign: |H| = 2^(n-1) exactly."""
        filtered = mti_filter(self.slow_time_tone(0.5), n_pulses=n_pulses, axis=0)
        np.testing.assert_allclose(np.abs(filtered), gain_linear, rtol=1e-12)

    @pytest.mark.parametrize("n_pulses", [2, 3])
    @pytest.mark.parametrize("doppler_fraction_of_prf", [0.01, 0.1, 0.25, 0.4])
    def test_matches_the_binomial_frequency_response(self, n_pulses, doppler_fraction_of_prf):
        r"""|H| = (2 |sin(pi f_d T)|)^(n-1): catches wrong taps such as [1, -2, -1]."""
        filtered = mti_filter(
            self.slow_time_tone(doppler_fraction_of_prf), n_pulses=n_pulses, axis=0
        )
        expected = (2.0 * abs(np.sin(np.pi * doppler_fraction_of_prf))) ** (n_pulses - 1)
        np.testing.assert_allclose(np.abs(filtered), expected, rtol=1e-10)

    def test_two_pulse_output_is_the_later_minus_the_earlier_pulse(self) -> None:
        """The documented y[m] = x[m+1] - x[m], and the dwell shortens by n - 1."""
        samples = np.arange(5.0)[:, None] ** 2 * np.ones((1, 3))
        np.testing.assert_allclose(mti_filter(samples, axis=0)[:, 0], [1.0, 3.0, 5.0, 7.0])
        assert mti_filter(np.ones((128, 4)), n_pulses=3, axis=0).shape == (126, 4)

    @pytest.mark.parametrize(("n_pulses", "n_available"), [(1, 8), (4, 8), (3, 2)])
    def test_rejects_an_unsupported_order_or_too_short_a_dwell(self, n_pulses, n_available):
        with pytest.raises(ValueError, match="must be 2 or 3"):
            mti_filter(np.ones((n_available, 4)), n_pulses=n_pulses, axis=0)


class TestFold:
    def test_fold_into_period_is_half_open_at_the_top(self) -> None:
        np.testing.assert_allclose(fold_into_period(130_000.0, SPAN_A_M, lower=0.0), 10_000.0)
        np.testing.assert_allclose(fold_into_period(SPAN_A_M, SPAN_A_M, lower=0.0), 0.0)

    def test_a_tiny_negative_value_folds_to_the_bottom_not_the_top(self) -> None:
        """np.mod(-1e-17, 15.3) rounds to 15.3 itself, the open edge; it must read 0.0."""
        assert float(fold_into_period(-1e-17, 15.3, lower=0.0)) == 0.0

    def test_fold_velocity_is_closed_at_the_top_as_the_map_is(self) -> None:
        """Both edges read +v_ua, since rate bin 0 is +v_ua: catches the closing-positive edge."""
        np.testing.assert_allclose(fold_velocity_mps([7.65, -7.65], 7.65), [7.65, 7.65])
        folded_mps = fold_velocity_mps(np.linspace(-500.0, 500.0, 2001), 7.65)
        assert np.all((folded_mps > -7.65) & (folded_mps <= 7.65))

    @pytest.mark.parametrize("n_folds", [-3, 2])
    def test_a_folded_target_lands_where_fold_velocity_says(self, n_folds: int) -> None:
        """Synthesise a target n_folds intervals off bin 7; the FFT and the fold must agree."""
        rate_mps = doppler_bin_centers_mps(N_PULSES, PULSE_REPETITION_INTERVAL_S, WAVELENGTH_M)
        v_ua_mps = WAVELENGTH_M / (2.0 * PULSE_REPETITION_INTERVAL_S)
        true_mps = rate_mps[7] + n_folds * 2.0 * v_ua_mps
        slow_time_s = np.arange(N_PULSES) * PULSE_REPETITION_INTERVAL_S
        echo = np.exp(-2j * np.pi * true_mps / WAVELENGTH_M * slow_time_s)
        peak = int(np.argmax(np.abs(doppler_fft(echo))))
        np.testing.assert_allclose(rate_mps[peak], fold_velocity_mps(true_mps, v_ua_mps), rtol=1e-9)

    @pytest.mark.parametrize("fold", [fold_into_period, fold_velocity_mps])
    def test_rejects_a_zero_period(self, fold) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            fold(10.0, 0.0, **({"lower": 0.0} if fold is fold_into_period else {}))


coprime_pair = st.tuples(st.integers(2, 9), st.integers(2, 9)).filter(
    lambda pq: pq[0] != pq[1] and math.gcd(*pq) == 1
)


class TestUnfold:
    @given(pq=coprime_pair, base_mps=st.floats(0.5, 50.0), fraction=st.floats(-0.999, 0.999))
    def test_doppler_round_trips_anywhere_in_the_pairs_span(self, pq, base_mps, fraction) -> None:
        """Periods p and q bases reach +-pq base / 2; every rate inside comes back exactly.

        Any wrong candidate misses burst B by a whole base, so a quarter-base
        tolerance separates them for every coprime pair.
        """
        p, q = pq
        max_velocity_mps = 0.999 * p * q * base_mps / 2.0
        true_mps = fraction * max_velocity_mps
        velocity_mps, residual_mps = unfold_doppler_dual_prf(
            fold_velocity_mps(true_mps, p * base_mps / 2.0),
            fold_velocity_mps(true_mps, q * base_mps / 2.0),
            p * base_mps / 2.0,
            q * base_mps / 2.0,
            max_velocity_mps=max_velocity_mps,
            tolerance_mps=base_mps / 4.0,
        )
        # A mod and a few adds: float error scales with the span.
        np.testing.assert_allclose(velocity_mps, true_mps, atol=1e-9 * max_velocity_mps)
        assert residual_mps < 1e-9 * max_velocity_mps

    @given(pq=coprime_pair, base_m=st.floats(100.0, 50_000.0), fraction=st.floats(0.0, 0.999))
    def test_range_round_trips_anywhere_in_the_pairs_span(self, pq, base_m, fraction) -> None:
        p, q = pq
        max_range_m = 0.999 * p * q * base_m
        true_m = fraction * max_range_m
        range_m, _ = unfold_range_dual_prf(
            fold_into_period(true_m, p * base_m, lower=0.0),
            fold_into_period(true_m, q * base_m, lower=0.0),
            p * base_m,
            q * base_m,
            max_range_m=max_range_m,
            tolerance_m=base_m / 4.0,
        )
        np.testing.assert_allclose(range_m, true_m, atol=1e-9 * max_range_m)

    def test_a_target_outside_the_bound_aliases_to_a_confident_wrong_answer(self) -> None:
        """300 m/s on S3's bursts reads as 300 - 6 x 2 x 38.24 m/s, with no warning."""
        velocity_mps, residual_mps = unfold_doppler_dual_prf(
            fold_velocity_mps(300.0, V_UA_A),
            fold_velocity_mps(300.0, V_UA_B),
            V_UA_A,
            V_UA_B,
            max_velocity_mps=191.0,
            tolerance_mps=1.0,
        )
        np.testing.assert_allclose(velocity_mps, 300.0 - 12.0 * V_UA_A, atol=1e-9)
        assert residual_mps < 1.0

    def test_pairs_every_a_reading_with_every_b_reading(self) -> None:
        """A column against a row: only the diagonal pairs a target with itself."""
        true_mps = np.array([10.0, 80.0, -120.0])
        velocity_mps, _ = unfold_doppler_dual_prf(
            fold_velocity_mps(true_mps, V_UA_A)[:, None],
            fold_velocity_mps(true_mps, V_UA_B)[None, :],
            V_UA_A,
            V_UA_B,
            max_velocity_mps=191.0,
            tolerance_mps=1.0,
        )
        np.testing.assert_allclose(np.diag(velocity_mps), true_mps, atol=1e-9)
        assert np.all(np.isnan(velocity_mps[~np.eye(3, dtype=bool)]))

    def test_the_miss_is_measured_round_the_second_maps_circle(self) -> None:
        """20 m below the 50 km wrap and 30 m past it are 50 m apart, not 49.95 km."""
        _, residual_m = unfold_range_dual_prf(
            49_980.0, 30.0, SPAN_A_M, SPAN_B_M, max_range_m=SPAN_A_M, tolerance_m=50.0
        )
        np.testing.assert_allclose(residual_m, 50.0, atol=1e-6)

    def test_an_inconsistent_pair_is_reported_unresolved(self) -> None:
        velocity_mps, residual_mps = unfold_doppler_dual_prf(
            3.52, -11.78 + 5.0, V_UA_A, V_UA_B, max_velocity_mps=191.0, tolerance_mps=1.0
        )
        assert np.isnan(velocity_mps)
        assert residual_mps > 1.0

    def test_a_value_on_the_lower_bound_is_recovered(self) -> None:
        """A reading of -1e-17 is the lower bound, 0, not a candidate grid starting at 6."""
        value, residual = unfold_dual_period(
            -1e-17, 0.0, 6.0, 5.0, lower_bound=0.0, upper_bound=29.0, tolerance=0.1
        )
        np.testing.assert_allclose(value, 0.0, atol=1e-12)
        assert residual < 1e-12

    @pytest.mark.parametrize(
        "call",
        [
            lambda: unfold_dual_period(
                0.0, 0.0, 6.0, 5.0, lower_bound=0.0, upper_bound=np.nan, tolerance=0.1
            ),
            lambda: unfold_doppler_dual_prf(
                0.0, 0.0, V_UA_A, V_UA_B, max_velocity_mps=np.nan, tolerance_mps=1.0
            ),
            lambda: range_bin_centers_m(8, np.nan, 40e-6, 1e6),
            lambda: doppler_bin_centers_mps(8, np.nan, WAVELENGTH_M),
        ],
        ids=["unfold_dual_period", "unfold_doppler_dual_prf", "range_bins", "rate_bins"],
    )
    def test_nan_is_rejected_at_the_boundary(self, call) -> None:
        """NaN fails every comparison, so `x <= 0` lets it through; `not x > 0` does not."""
        with pytest.raises(ValueError, match=r"positive|must exceed|upper_bound"):
            call()

    def test_equal_periods_resolve_within_one_period_only(self) -> None:
        value, _ = unfold_dual_period(
            12.0, 12.0, 20.0, 20.0, lower_bound=0.0, upper_bound=20.0, tolerance=0.1
        )
        np.testing.assert_allclose(value, 12.0)
        with pytest.raises(ValueError, match="equal periods"):
            unfold_dual_period(
                0.0, 0.0, 20.0, 20.0, lower_bound=0.0, upper_bound=21.0, tolerance=0.1
            )

    @pytest.mark.parametrize(
        ("periods", "tolerance", "bounds"),
        [
            ((0.0, 5.0), 0.1, (0.0, 10.0)),
            ((6.0, 0.0), 0.1, (0.0, 10.0)),
            ((6.0, 5.0), 0.0, (0.0, 10.0)),
            ((6.0, 5.0), 0.1, (10.0, 0.0)),
        ],
        ids=["period-a", "period-b", "tolerance", "inverted-bounds"],
    )
    def test_unfold_dual_period_rejects_invalid_arguments(self, periods, tolerance, bounds):
        with pytest.raises(ValueError, match="must be positive"):
            unfold_dual_period(
                0.0,
                0.0,
                *periods,
                lower_bound=bounds[0],
                upper_bound=bounds[1],
                tolerance=tolerance,
            )

    @pytest.mark.parametrize(
        ("v_ua_a", "v_ua_b", "max_mps", "match"),
        [
            (0.0, V_UA_B, 191.0, "positive and different"),
            (V_UA_A, V_UA_A, 191.0, "positive and different"),
            (V_UA_A, V_UA_B, 10.0, "nothing is folded"),
        ],
    )
    def test_unfold_doppler_rejects_invalid_arguments(self, v_ua_a, v_ua_b, max_mps, match):
        with pytest.raises(ValueError, match=match):
            unfold_doppler_dual_prf(
                0.0, 0.0, v_ua_a, v_ua_b, max_velocity_mps=max_mps, tolerance_mps=1.0
            )

    def test_unfold_range_rejects_a_negative_bound(self) -> None:
        with pytest.raises(ValueError, match="max_range_m"):
            unfold_range_dual_prf(0.0, 0.0, SPAN_A_M, SPAN_B_M, max_range_m=-1.0, tolerance_m=50.0)
