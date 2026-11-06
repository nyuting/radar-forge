"""Tests for the waveform, the ambiguity function, and baseband synthesis.

Ground truth is analytic: a target placed on an exact range and Doppler bin must
land in it, a linear-FM pulse must compress to the textbook mainlobe and
sidelobe, and noise must have the power it was asked for inside a chi-square
interval. The matched filter and the range-Doppler map are written here with
NumPy and SciPy, independent of the library's processing modules.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from scipy import stats
from scipy.signal import fftconvolve

from radar_forge.constants import SPEED_OF_LIGHT_MPS
from radar_forge.propagation import PropagationPaths, line_of_sight_paths
from radar_forge.radar import Radar, Receiver, Site, TargetGeometry, Transmitter
from radar_forge.signal import (
    ambiguity_function,
    beat_frequency_hz,
    bistatic_range_from_beat_frequency_m,
    cubic_amplifier,
    fmcw_deramp_baseband,
    lfm_chirp,
    noise_waveform,
    pulsed_baseband,
    range_gate_first_sample,
    range_resolution_m,
    stream_baseband,
    sweep_rate_hzps,
    thermal_noise,
)

DUKE = Site(36.00250, -78.94100, 60.0)
RDU = Site(35.87750, -78.78750, 25.0)
N_PULSES = 256

S1_RADAR = Radar(
    Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3), Receiver(1.0e6, 30.0, 3.0), DUKE
)
S2_RADAR = Radar(
    Transmitter(9.8e9, 2.0e6, 1.0e3, 30.0, 10.0e-6, 25.0e3, waveform="pulsed"),
    Receiver(2.5e6, 30.0, 3.0),
    DUKE,
)
S1_PAIR = Radar(S1_RADAR.transmitter, S1_RADAR.receiver, tx_site=RDU, rx_site=DUKE)
S2_PAIR = Radar(S2_RADAR.transmitter, S2_RADAR.receiver, tx_site=RDU, rx_site=DUKE)


def _route(range_tx_m: float, range_rx_m: float, rate_mps: float = 0.0) -> TargetGeometry:
    """A one-target geometry with the route split unevenly between the two ranges."""
    return TargetGeometry(
        range_tx_m=np.array([range_tx_m]),
        range_rx_m=np.array([range_rx_m]),
        bistatic_range_m=np.array([range_tx_m + range_rx_m]),
        bistatic_range_rate_mps=np.array([rate_mps]),
        bistatic_angle_rad=np.array([1.0]),
        aod_rad=np.zeros((1, 2)),
        aoa_rad=np.zeros((1, 2)),
    )


def _doppler_rate_mps(radar: Radar, doppler_bin: int, n_pulses: int = N_PULSES) -> float:
    """Bistatic range rate at the centre of a Doppler bin of the shifted slow-time FFT."""
    doppler_hz = (doppler_bin - n_pulses // 2) * radar.transmitter.prf_hz / n_pulses
    return -radar.wavelength_m * doppler_hz


def _fmcw_bistatic_range_m(range_bin: int) -> float:
    """S1's bistatic range at the centre of a fast-time FFT bin: R_b = c f_b / alpha."""
    beat_hz = range_bin * S1_RADAR.receiver.sample_rate_hz / S1_RADAR.n_samples_per_pri
    return SPEED_OF_LIGHT_MPS * beat_hz / S1_RADAR.transmitter.sweep_rate_hzps


def _range_doppler_peak(cube: np.ndarray) -> tuple[int, int]:
    """(Doppler bin, range bin) of the brightest cell, Doppler shifted to put zero at N/2."""
    rd_map = np.fft.fftshift(np.fft.fft(np.fft.fft(cube, axis=1), axis=0), axes=0)
    doppler_bin, range_bin = np.unravel_index(np.abs(rd_map).argmax(), rd_map.shape)
    return int(doppler_bin), int(range_bin)


def _compress(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Matched-filter along the last axis; output index i is lag i - (n_reference - 1)."""
    return fftconvolve(samples, np.conj(reference[::-1])[None, :], mode="full", axes=-1)


class TestFmcwRelations:
    def test_resolution_and_sweep_rate_match_closed_forms(self) -> None:
        np.testing.assert_allclose(
            range_resolution_m([1e9, 2e6]),
            SPEED_OF_LIGHT_MPS / (2.0 * np.array([1e9, 2e6])),
            rtol=1e-12,
        )
        np.testing.assert_allclose(sweep_rate_hzps(1e9, 40e-6), 2.5e13, rtol=1e-12)

    def test_beat_frequency_carries_no_factor_of_two(self) -> None:
        """f_b = alpha R_b / c: a monostatic target at R has R_b = 2R and beat 2 alpha R / c."""
        np.testing.assert_allclose(
            beat_frequency_hz(2.0 * 42.0, 1e9, 40e-6),
            2.5e13 * 84.0 / SPEED_OF_LIGHT_MPS,
            rtol=1e-12,
        )

    def test_beat_frequency_round_trips_through_bistatic_range(self) -> None:
        ranges_m = np.array([0.0, 1.5, 42.0, 2.5e4])
        beat_hz = beat_frequency_hz(ranges_m, 2.0e6, 1.0e-3)
        np.testing.assert_allclose(
            bistatic_range_from_beat_frequency_m(beat_hz, 2.0e6, 1.0e-3),
            ranges_m,
            rtol=1e-12,
            atol=1e-12,
        )


class TestLfmChirp:
    def test_sample_count_is_half_open(self) -> None:
        """[0, T) holds T fs samples: the sample at T starts the next chirp."""
        assert lfm_chirp(1e6, 1e-3, 2e6).shape == (2000,)

    def test_is_constant_modulus_at_the_requested_amplitude(self) -> None:
        np.testing.assert_allclose(
            np.abs(lfm_chirp(1e6, 1e-3, 2e6, amplitude_linear=3.0)), 3.0, rtol=1e-12
        )

    def test_default_sweep_is_centred_and_linear(self) -> None:
        """Catches a missing half in the quadratic phase, or a sweep that starts at zero.

        The phase difference samples f(t) = -B/2 + alpha t at bin midpoints.
        """
        bandwidth_hz, chirp_duration_s, sample_rate_hz = 1.0e6, 1.0e-3, 4.0e6
        chirp = lfm_chirp(bandwidth_hz, chirp_duration_s, sample_rate_hz)
        measured_hz = np.diff(np.unwrap(np.angle(chirp))) * sample_rate_hz / (2.0 * np.pi)
        midpoint_s = (np.arange(chirp.size - 1) + 0.5) / sample_rate_hz
        expected_hz = -bandwidth_hz / 2.0 + bandwidth_hz / chirp_duration_s * midpoint_s
        # Unwrapped float64 phase over 4000 samples: about 1e-6 Hz of noise per MHz.
        np.testing.assert_allclose(measured_hz, expected_hz, rtol=0.0, atol=1e-3)

    def test_the_down_sweep_is_the_conjugate_of_the_up_sweep(self) -> None:
        up = lfm_chirp(1e6, 1e-3, 2e6, sweep_direction="up")
        down = lfm_chirp(1e6, 1e-3, 2e6, sweep_direction="down")
        np.testing.assert_allclose(down, np.conjugate(up), rtol=1e-12)

    def test_a_start_frequency_shifts_the_sweep(self) -> None:
        """Starting at 0 rather than -B/2 is the centred sweep moved up by B/2."""
        time_s = np.arange(4000) / 4.0e6
        shifted = lfm_chirp(1e6, 1e-3, 4e6) * np.exp(2j * np.pi * 0.5e6 * time_s)
        np.testing.assert_allclose(
            lfm_chirp(1e6, 1e-3, 4e6, start_frequency_hz=0.0), shifted, rtol=1e-9, atol=1e-12
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"bandwidth_hz": 0.0}, "bandwidth_hz"),
            ({"chirp_duration_s": 0.0}, "chirp_duration_s"),
            ({"sample_rate_hz": 0.0}, "sample_rate_hz"),
            ({"sample_rate_hz": 0.5e6}, "would alias"),
            ({"chirp_duration_s": 1e-9}, "too short"),
            ({"sweep_direction": "sideways"}, "'up' or 'down'"),
        ],
    )
    def test_rejects_a_bad_parameter(self, kwargs: dict[str, object], match: str) -> None:
        base: dict[str, object] = {
            "bandwidth_hz": 1e6,
            "chirp_duration_s": 1e-3,
            "sample_rate_hz": 2e6,
        }
        with pytest.raises(ValueError, match=match):
            lfm_chirp(**{**base, **kwargs})  # type: ignore[arg-type]  # deliberately bad value


class TestMatchedFilter:
    """An LFM pulse through its own matched filter (scipy fftconvolve; BT = 100, 16x sampled)."""

    BANDWIDTH_HZ = 1.0e6
    SAMPLE_RATE_HZ = 16.0e6
    DELAY_SAMPLES = 37

    def _compressed(self) -> np.ndarray:
        pulse = lfm_chirp(self.BANDWIDTH_HZ, 100.0e-6, self.SAMPLE_RATE_HZ)
        echo = np.zeros(pulse.size + 200, dtype=np.complex128)
        echo[self.DELAY_SAMPLES : self.DELAY_SAMPLES + pulse.size] = pulse
        return np.abs(_compress(echo[None, :], pulse)[0, pulse.size - 1 :])

    def test_the_peak_is_at_the_true_delay(self) -> None:
        assert int(np.argmax(self._compressed())) == self.DELAY_SAMPLES

    def test_the_first_null_is_one_over_b_away(self) -> None:
        """Peak to first null is 1/B of delay, c/2B of range: the resolution, to a sample."""
        response = self._compressed()[self.DELAY_SAMPLES :]
        first_null = int(np.argmax(np.diff(response) > 0.0))
        assert abs(first_null - self.SAMPLE_RATE_HZ / self.BANDWIDTH_HZ) <= 1

    def test_the_peak_sidelobe_is_the_sinc_minus_thirteen_db(self) -> None:
        """-13.26 dB for large BT; at BT = 100 the Fresnel ripple moves it by about 0.1 dB."""
        response = self._compressed()[self.DELAY_SAMPLES :]
        first_null = int(np.argmax(np.diff(response) > 0.0))
        peak_sidelobe_db = 20.0 * np.log10(response[first_null:].max() / response[0])
        assert abs(peak_sidelobe_db - (-13.26)) < 0.3


class TestAmbiguityFunction:
    def test_is_one_at_the_origin_and_nowhere_larger(self) -> None:
        """Normalised by the energies: Cauchy-Schwarz caps an auto-ambiguity at the origin."""
        chirp = lfm_chirp(1.0e6, 20.0e-6, 4.0e6)
        magnitude, delays_s, dopplers_hz = ambiguity_function(chirp, 4.0e6)
        origin = magnitude[dopplers_hz == 0.0][:, delays_s == 0.0]
        np.testing.assert_allclose(origin, 1.0, rtol=1e-12)
        assert magnitude.shape == (dopplers_hz.size, delays_s.size) == (160, 159)
        assert magnitude.max() <= 1.0 + 1e-12

    @pytest.mark.parametrize(("sweep_direction", "sign"), [("up", -1.0), ("down", 1.0)])
    def test_the_lfm_ridge_has_slope_b_over_t(self, sweep_direction: str, sign: float) -> None:
        """Range-Doppler coupling: an up-chirp peaks at tau = -f_D / alpha, height 1 - |f_D|/B."""
        bandwidth_hz, chirp_duration_s, sample_rate_hz = 1.0e6, 100.0e-6, 16.0e6
        chirp = lfm_chirp(
            bandwidth_hz, chirp_duration_s, sample_rate_hz, sweep_direction=sweep_direction
        )  # type: ignore[arg-type]
        dopplers_hz = np.linspace(-0.4, 0.4, 9) * bandwidth_hz
        magnitude, delays_s, _ = ambiguity_function(chirp, sample_rate_hz, dopplers_hz=dopplers_hz)
        ridge_delays_s = delays_s[np.argmax(magnitude, axis=1)]
        slope_hzps = np.polyfit(ridge_delays_s, dopplers_hz, 1)[0]
        # Delays are whole samples, 1/16 of 1/B against shifts of up to 40/B.
        np.testing.assert_allclose(slope_hzps, sign * bandwidth_hz / chirp_duration_s, rtol=1e-3)
        np.testing.assert_allclose(
            magnitude.max(axis=1), 1.0 - np.abs(dopplers_hz) / bandwidth_hz, rtol=1e-3
        )

    def test_the_mainlobe_is_one_over_b_by_one_over_t(self) -> None:
        """-3 dB widths 0.886/B in delay and 0.886/T in Doppler (BT = 100).

        The delay cut is 64x oversampled, so counting samples errs by under 2 %;
        the Doppler cut, which oversampling does not change, is evaluated every
        0.01/T on a 4x-oversampled pulse.
        """
        bandwidth_hz, chirp_duration_s = 1.0e6, 100.0e-6
        sample_rate_hz = 64.0 * bandwidth_hz
        chirp = lfm_chirp(bandwidth_hz, chirp_duration_s, sample_rate_hz)
        zero_doppler, _, _ = ambiguity_function(chirp, sample_rate_hz, dopplers_hz=[0.0])
        above = np.flatnonzero(zero_doppler[0] >= 1.0 / np.sqrt(2.0))
        np.testing.assert_allclose(
            (above[-1] - above[0]) / sample_rate_hz, 0.886 / bandwidth_hz, rtol=0.03
        )
        dopplers_hz = np.linspace(-2.0, 2.0, 401) / chirp_duration_s
        zero_delay, _, _ = ambiguity_function(
            lfm_chirp(bandwidth_hz, chirp_duration_s, 4.0 * bandwidth_hz),
            4.0 * bandwidth_hz,
            delays_s=[0.0],
            dopplers_hz=dopplers_hz,
        )
        above = np.flatnonzero(zero_delay[:, 0] >= 1.0 / np.sqrt(2.0))
        np.testing.assert_allclose(
            dopplers_hz[above[-1]] - dopplers_hz[above[0]], 0.886 / chirp_duration_s, rtol=0.02
        )

    def test_a_pulse_train_peaks_at_every_multiple_of_the_pri(self) -> None:
        """N rectangular pulses overlap N - |k| of themselves at k PRIs: height (N - |k|)/N.

        Uses the PRF-narrowed default Doppler grid, which must contain zero Doppler.
        """
        n_pulses, pulse_samples, pri_samples = 4, 8, 32
        train = np.zeros(pri_samples * (n_pulses - 1) + pulse_samples, dtype=np.complex128)
        for k in range(n_pulses):  # Four slices: clearer than an index expression.
            train[k * pri_samples : k * pri_samples + pulse_samples] = 1.0
        magnitude, delays_s, dopplers_hz = ambiguity_function(train, 1.0, prf_hz=1.0 / pri_samples)
        assert np.all(np.abs(dopplers_hz) <= 0.5 / pri_samples)
        zero_doppler = magnitude[dopplers_hz == 0.0][0]
        middle = zero_doppler[1:-1]
        # The 0.01 floor ignores FFT round-off (1e-17) ripple between the pulses.
        is_peak = (middle > zero_doppler[:-2]) & (middle > zero_doppler[2:]) & (middle > 0.01)
        peaks = np.flatnonzero(is_peak) + 1
        lags = np.rint(delays_s[peaks]).astype(int)
        np.testing.assert_array_equal(lags, pri_samples * np.arange(1 - n_pulses, n_pulses))
        np.testing.assert_allclose(
            zero_doppler[peaks], (n_pulses - np.abs(lags // pri_samples)) / n_pulses, rtol=1e-12
        )

    def test_delays_are_rounded_to_whole_samples(self) -> None:
        """And returned as evaluated; a delay with no overlap is zero, not an index error."""
        _, delays_s, _ = ambiguity_function(
            np.ones(8), 10.0, delays_s=[0.04, 0.26, 5.0], dopplers_hz=[0.0]
        )
        np.testing.assert_allclose(delays_s, [0.0, 0.3, 5.0], rtol=1e-12)
        magnitude, _, _ = ambiguity_function(np.ones(8), 10.0, delays_s=[5.0], dopplers_hz=[0.0])
        assert magnitude[0, 0] == 0.0

    def test_a_mismatched_reference_costs_its_mismatch_at_the_origin(self) -> None:
        """|sum s r*| / sqrt(Es Er): a cross-ambiguity is below 1 at the origin."""
        up = lfm_chirp(1.0e6, 20.0e-6, 4.0e6)
        reference = up * np.hanning(up.size)
        magnitude, _, _ = ambiguity_function(
            up, 4.0e6, delays_s=[0.0], dopplers_hz=[0.0], reference=reference
        )
        expected = np.abs(np.vdot(reference, up)) / np.sqrt(
            np.vdot(up, up).real * np.vdot(reference, reference).real
        )
        np.testing.assert_allclose(magnitude[0, 0], expected, rtol=1e-12)
        assert expected < 0.9

    def test_rejects_a_waveform_that_is_not_a_vector(self) -> None:
        with pytest.raises(ValueError, match="non-empty vectors"):
            ambiguity_function(np.ones((2, 4)), 1.0)
        with pytest.raises(ValueError, match="strictly positive"):
            ambiguity_function(np.ones(4), 0.0)

    def test_rejects_an_inconsistent_doppler_request(self) -> None:
        """prf_hz only shapes the default grid, so with an explicit grid it would be ignored."""
        with pytest.raises(ValueError, match="not both"):
            ambiguity_function(np.ones(8), 1.0, prf_hz=0.1, dopplers_hz=[0.0])
        with pytest.raises(ValueError, match="prf_hz must be strictly positive"):
            ambiguity_function(np.ones(8), 1.0, prf_hz=0.0)

    @pytest.mark.parametrize("silent", ["waveform", "reference"])
    def test_rejects_a_signal_with_no_energy(self, silent: str) -> None:
        """The response is normalised by sqrt(Es Er): zero energy would be NaN everywhere."""
        signals = {"waveform": np.ones(8), "reference": np.ones(8), silent: np.zeros(8)}
        with pytest.raises(ValueError, match="carry energy"):
            ambiguity_function(
                signals["waveform"], 1.0, reference=signals["reference"], dopplers_hz=[0.0]
            )


class TestFmcwDerampBaseband:
    def test_a_closing_target_on_bin_centres_lands_in_exactly_those_bins(self) -> None:
        """Catches a flipped Doppler sign, a one-way delay, or an off-by-one bin, end to end.

        Range bin 133 (R = 9968 m) and Doppler bin N/2 + 50 (closing at 2.99 m/s).
        With an up sweep the target would land on N/2 - 50 and look plausible.
        """
        rate_mps = _doppler_rate_mps(S1_RADAR, N_PULSES // 2 + 50)
        assert rate_mps < 0.0  # closing
        geometry = TargetGeometry.monostatic(_fmcw_bistatic_range_m(133) / 2.0, rate_mps / 2.0)
        cube = fmcw_deramp_baseband(
            line_of_sight_paths(S1_RADAR, geometry, 10.0), S1_RADAR, N_PULSES, rng=None
        )
        assert _range_doppler_peak(cube) == (N_PULSES // 2 + 50, 133)

    def test_a_receding_target_lands_at_negative_doppler(self) -> None:
        rate_mps = _doppler_rate_mps(S1_RADAR, N_PULSES // 2 - 50)
        assert rate_mps > 0.0  # receding
        geometry = TargetGeometry.monostatic(_fmcw_bistatic_range_m(133) / 2.0, rate_mps / 2.0)
        cube = fmcw_deramp_baseband(
            line_of_sight_paths(S1_RADAR, geometry, 10.0), S1_RADAR, N_PULSES, rng=None
        )
        assert _range_doppler_peak(cube) == (N_PULSES // 2 - 50, 133)

    def test_a_fast_target_folds_in_doppler_but_not_in_range(self) -> None:
        """Scenario 001 S1: five rate spans (76.5 m/s radial) above a bin lands back on it.

        Stop-and-hop holds at this speed, and warnings are errors, so the
        generator must also stay silent.
        """
        rate_mps = (
            _doppler_rate_mps(S1_RADAR, N_PULSES // 2 + 50)
            - 5.0 * S1_RADAR.unambiguous_bistatic_range_rate_mps
        )
        geometry = TargetGeometry.monostatic(_fmcw_bistatic_range_m(133) / 2.0, rate_mps / 2.0)
        cube = fmcw_deramp_baseband(
            line_of_sight_paths(S1_RADAR, geometry, 10.0), S1_RADAR, N_PULSES, rng=None
        )
        assert _range_doppler_peak(cube) == (N_PULSES // 2 + 50, 133)

    def test_a_bistatic_target_sits_in_its_bistatic_range_bin(self) -> None:
        """The route is split 1:3, so a generator reading either range alone lands elsewhere."""
        bistatic_range_m = _fmcw_bistatic_range_m(40)
        geometry = _route(0.25 * bistatic_range_m, 0.75 * bistatic_range_m)
        cube = fmcw_deramp_baseband(
            line_of_sight_paths(S1_PAIR, geometry, 10.0), S1_PAIR, N_PULSES, rng=None
        )
        assert _range_doppler_peak(cube) == (N_PULSES // 2, 40)

    def test_warns_when_stop_and_hop_is_strained(self) -> None:
        """A half-second chirp lets a 100 m/s target cross a bin within one chirp."""
        slow_sweep = Radar(
            Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 0.5, 2.0), Receiver(1.0e6, 30.0, 3.0), DUKE
        )
        paths = line_of_sight_paths(slow_sweep, TargetGeometry.monostatic(10_000.0, -100.0), 10.0)
        with pytest.warns(UserWarning, match="range bin during one chirp"):
            fmcw_deramp_baseband(paths, slow_sweep, 4, rng=None)

    def test_rejects_a_pulsed_radar_and_an_empty_dwell(self) -> None:
        paths = line_of_sight_paths(S2_RADAR, TargetGeometry.monostatic(10_000.0, 0.0), 10.0)
        with pytest.raises(ValueError, match="needs a fmcw transmitter"):
            fmcw_deramp_baseband(paths, S2_RADAR, 4, rng=None)
        with pytest.raises(ValueError, match="at least one"):
            fmcw_deramp_baseband(paths, S1_RADAR, 0, rng=None)


class TestPulsedBaseband:
    REFERENCE = lfm_chirp(2.0e6, 10.0e-6, 2.5e6)
    LAG_SAMPLES = 30  # 12 us, an exact sample: R_b = c * 12 us

    def _peak_lag(self, radar: Radar, geometry: TargetGeometry) -> int:
        cube = pulsed_baseband(line_of_sight_paths(radar, geometry, 10.0), radar, 4, rng=None)
        compressed = np.abs(_compress(cube, self.REFERENCE)[0])
        return int(np.argmax(compressed)) - (self.REFERENCE.size - 1)

    def test_the_cube_spans_the_whole_repetition_interval(self) -> None:
        """100 samples per 40 us PRI, not the 25 the pulse occupies."""
        paths = line_of_sight_paths(S2_RADAR, TargetGeometry.monostatic(4_000.0, 0.0), 10.0)
        assert pulsed_baseband(paths, S2_RADAR, N_PULSES, rng=None).shape == (N_PULSES, 100)

    def test_an_unambiguous_target_compresses_at_its_true_delay(self) -> None:
        bistatic_range_m = SPEED_OF_LIGHT_MPS * self.LAG_SAMPLES / 2.5e6
        assert (
            self._peak_lag(S2_RADAR, TargetGeometry.monostatic(bistatic_range_m / 2.0, 0.0))
            == self.LAG_SAMPLES
        )

    def test_a_distant_bistatic_target_folds_on_the_bistatic_range(self) -> None:
        """One PRI further along the route, split 1:2, compresses to the same sample."""
        bistatic_range_m = SPEED_OF_LIGHT_MPS * (self.LAG_SAMPLES / 2.5e6 + 40.0e-6)
        geometry = _route(bistatic_range_m / 3.0, 2.0 * bistatic_range_m / 3.0)
        assert self._peak_lag(S2_PAIR, geometry) == self.LAG_SAMPLES

    def test_a_fast_closing_target_lands_at_positive_doppler_unfolded(self) -> None:
        """Doppler bin N/2 + 54 is 80.66 m/s closing, well inside S2's +/-191 m/s.

        Catches a flipped sign, or a carrier phase taken from the folded delay.
        """
        doppler_bin = N_PULSES // 2 + 54
        geometry = TargetGeometry.monostatic(
            4_000.0, _doppler_rate_mps(S2_RADAR, doppler_bin) / 2.0
        )
        cube = pulsed_baseband(
            line_of_sight_paths(S2_RADAR, geometry, 10.0), S2_RADAR, N_PULSES, rng=None
        )
        compressed = _compress(cube, self.REFERENCE)
        assert _range_doppler_peak(compressed)[0] == doppler_bin

    def test_rejects_an_fmcw_radar_and_an_empty_dwell(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, TargetGeometry.monostatic(10_000.0, 0.0), 10.0)
        with pytest.raises(ValueError, match="needs a pulsed transmitter"):
            pulsed_baseband(paths, S1_RADAR, 4, rng=None)
        with pytest.raises(ValueError, match="at least one"):
            pulsed_baseband(paths, S2_RADAR, 0, rng=None)

    def test_an_echo_overrunning_its_pri_finishes_at_the_start_of_the_next_row(self) -> None:
        """Catches a truncated echo: 5.25 us before a 1 ms PRI ends, a 10 us pulse overruns.

        The receiver keeps listening, so 10 of the 20 samples open the next row.
        The rows, laid end to end, compress to the same peak as the echo placed
        mid-PRI with the same quarter-sample offset (the offset keeps every edge
        off the sample grid). The last row's overrun is lost.
        """
        radar = Radar(
            Transmitter(9.8e9, 1.0e6, 1.0e3, 30.0, 10.0e-6, 1.0e3, waveform="pulsed"),
            Receiver(2.0e6, 30.0, 3.0),
            DUKE,
        )
        reference = lfm_chirp(1.0e6, 10.0e-6, 2.0e6)

        def peak(delay_s: float, n_rows: int) -> tuple[np.ndarray, float]:
            """The cube, and its compressed peak per unit path amplitude (ranges differ)."""
            geometry = TargetGeometry.monostatic(delay_s * SPEED_OF_LIGHT_MPS / 2.0, 0.0)
            paths = line_of_sight_paths(radar, geometry, 10.0)
            cube = pulsed_baseband(paths, radar, 3, rng=None)
            stream = cube[:n_rows].reshape(1, -1)  # the receiver's continuous record
            compressed_peak = np.abs(_compress(stream, reference)).max()
            return cube, float(compressed_peak / np.abs(paths.amplitude_linear[0]))

        cube, straddling = peak(1.0e-3 - 5.25e-6, n_rows=2)
        _, mid_pri = peak(0.5e-3 - 0.25e-6, n_rows=1)
        occupied = np.abs(cube) > 0.0
        np.testing.assert_array_equal(occupied.sum(axis=1), [10, 20, 20])
        assert occupied[0, -10:].all() and occupied[1, :10].all()
        # The same twenty samples, differing only in a constant carrier phase.
        np.testing.assert_allclose(straddling, mid_pri, rtol=1e-9)

    def test_warns_when_stop_and_hop_is_strained(self) -> None:
        """A 50 ms pulse lets a 500 m/s target cross a quarter of a 75 m bin within it."""
        long_pulse = Radar(
            Transmitter(9.8e9, 2.0e6, 1.0e3, 30.0, 50.0e-3, 10.0, waveform="pulsed"),
            Receiver(2.0e6, 30.0, 3.0),
            DUKE,
        )
        paths = line_of_sight_paths(long_pulse, TargetGeometry.monostatic(10_000.0, -500.0), 10.0)
        with pytest.warns(UserWarning, match="range bin during one chirp"):
            pulsed_baseband(paths, long_pulse, 1, rng=None)


# A 40 MHz carrier with 8 MHz of sweep: 20 % fractional bandwidth, so a Doppler
# referred to a band edge rather than the carrier is 10 % off and plainly visible.
_WIDEBAND_CARRIER_HZ = 40.0e6
_WIDEBAND_BANDWIDTH_HZ = 8.0e6


def _slow_time_doppler_hz(samples: np.ndarray, pulse_repetition_interval_s: float) -> float:
    """Doppler from the mean pulse-to-pulse phase step of one cell, Hz."""
    phase_step_rad = np.angle(samples[1:] * np.conj(samples[:-1]))
    return float(np.mean(phase_step_rad) / (2.0 * np.pi * pulse_repetition_interval_s))


class TestDopplerIsReferredToTheCarrier:
    """A closing route's cell advances at +2 v f_c / c: velocity converts with lambda = c / f_c."""

    def test_fmcw_range_bin_advances_at_the_carrier(self) -> None:
        radar = Radar(
            Transmitter(_WIDEBAND_CARRIER_HZ, _WIDEBAND_BANDWIDTH_HZ, 1.0, 0.0, 1.0e-3, 1.0e3),
            Receiver(1.0e6, 0.0, 3.0),
            DUKE,
        )
        paths = line_of_sight_paths(radar, TargetGeometry.monostatic(5_000.0, -1.0), 10.0)
        spectra = np.fft.fft(fmcw_deramp_baseband(paths, radar, 64, rng=None), axis=-1)
        peak_bin = int(np.abs(spectra[0]).argmax())
        measured_hz = _slow_time_doppler_hz(
            spectra[:, peak_bin], radar.transmitter.pulse_repetition_interval_s
        )
        np.testing.assert_allclose(measured_hz, 2.0 / radar.wavelength_m, rtol=1e-3)

    def test_pulsed_compressed_peak_advances_at_the_carrier(self) -> None:
        radar = Radar(
            Transmitter(
                _WIDEBAND_CARRIER_HZ,
                _WIDEBAND_BANDWIDTH_HZ,
                1.0,
                0.0,
                10.0e-6,
                10.0e3,
                waveform="pulsed",
            ),
            Receiver(10.0e6, 0.0, 3.0),
            DUKE,
        )
        paths = line_of_sight_paths(radar, TargetGeometry.monostatic(3_000.0, -1.0), 10.0)
        compressed = _compress(
            pulsed_baseband(paths, radar, 64, rng=None),
            lfm_chirp(_WIDEBAND_BANDWIDTH_HZ, 10.0e-6, 10.0e6),
        )
        peak_lag = int(np.abs(compressed[0]).argmax())
        measured_hz = _slow_time_doppler_hz(
            compressed[:, peak_lag], radar.transmitter.pulse_repetition_interval_s
        )
        np.testing.assert_allclose(measured_hz, 2.0 / radar.wavelength_m, rtol=1e-3)


def _eight_paths(radar: Radar) -> PropagationPaths:
    """Eight distinct paths: different ranges, rates and cross-sections."""
    geometry = TargetGeometry.monostatic(
        np.linspace(1_000.0, 15_000.0, 8), np.linspace(-70.0, 60.0, 8)
    )
    return line_of_sight_paths(radar, geometry, np.linspace(1.0, 20.0, 8))


class TestPathChunking:
    """Memory is bounded by summing chunks of paths, and the cube does not move a bit."""

    N_PULSES_SHORT = 16

    @pytest.mark.parametrize(
        ("radar", "generate"),
        [(S1_RADAR, fmcw_deramp_baseband), (S2_RADAR, pulsed_baseband)],
        ids=["fmcw", "pulsed"],
    )
    def test_four_chunks_equal_one_bit_for_bit(self, radar: Radar, generate: object) -> None:
        """Two paths per chunk, so eight paths take four chunks; the sum order is unchanged."""
        bytes_per_path = 16 * self.N_PULSES_SHORT * radar.n_samples_per_pri
        paths = _eight_paths(radar)
        whole = generate(paths, radar, self.N_PULSES_SHORT, rng=None)  # type: ignore[operator]
        chunked = generate(
            paths, radar, self.N_PULSES_SHORT, rng=None, max_bytes=2 * bytes_per_path
        )  # type: ignore[operator]
        assert np.any(whole != 0.0)
        np.testing.assert_array_equal(chunked, whole)

    def test_no_paths_give_a_silent_cube(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, TargetGeometry.monostatic(np.zeros(0), 0.0), 10.0)
        np.testing.assert_array_equal(fmcw_deramp_baseband(paths, S1_RADAR, 4, rng=None), 0.0)


def _power_interval_w(noise_power_w: float, n_samples: int) -> tuple[float, float]:
    """99.9 % interval of the mean |n|^2 of n_samples complex Gaussian samples: chi2 with 2n dof."""
    lower, upper = stats.chi2.ppf([0.0005, 0.9995], 2 * n_samples) / (2 * n_samples)
    return noise_power_w * lower, noise_power_w * upper


class TestThermalNoise:
    def test_power_lies_in_its_chi_square_interval(self) -> None:
        noise = thermal_noise((2000, 1000), 4.0e-15, rng=np.random.default_rng(20260911))
        lower_w, upper_w = _power_interval_w(4.0e-15, noise.size)
        assert lower_w < float(np.mean(np.abs(noise) ** 2)) < upper_w

    def test_is_circularly_symmetric(self) -> None:
        """Real and imaginary parts carry half the power each and do not correlate."""
        noise = thermal_noise((4000, 500), 2.0, rng=np.random.default_rng(20260911))
        np.testing.assert_allclose([np.var(noise.real), np.var(noise.imag)], 1.0, rtol=0.02)
        assert abs(float(np.mean(noise.real * noise.imag))) < 0.05

    def test_is_reproducible_and_silent_at_zero_power(self) -> None:
        first = thermal_noise((8, 8), 1.0, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(
            first, thermal_noise((8, 8), 1.0, rng=np.random.default_rng(7))
        )
        np.testing.assert_array_equal(thermal_noise((4, 4), 0.0, rng=np.random.default_rng(1)), 0.0)

    def test_rejects_negative_power(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            thermal_noise((2, 2), -1.0, rng=np.random.default_rng(1))

    @pytest.mark.parametrize(
        ("radar", "generate"),
        [(S1_RADAR, fmcw_deramp_baseband), (S2_RADAR, pulsed_baseband)],
        ids=["fmcw", "pulsed"],
    )
    def test_generators_add_the_noise_of_the_sampled_band(
        self, radar: Radar, generate: object
    ) -> None:
        """N0 fs per sample, not N0 B: S1 samples half its sweep, S2 1.25x its pulse.

        Both differ from N0 B by 20 % or more, outside the 99.9 % interval of
        64 x n_samples samples, so the test tells the two apart.
        """
        paths = line_of_sight_paths(radar, TargetGeometry.monostatic(3_000.0, 0.0), 10.0)
        clean = generate(paths, radar, 64, rng=None)  # type: ignore[operator]
        noisy = generate(paths, radar, 64, rng=np.random.default_rng(3))  # type: ignore[operator]
        lower_w, upper_w = _power_interval_w(radar.receiver.noise_power_per_sample_w, clean.size)
        assert lower_w < float(np.mean(np.abs(noisy - clean) ** 2)) < upper_w
        assert not lower_w < radar.noise_power_matched_w < upper_w


# --- Scenario 005: the noise code and its continuous stream ---


def _noise_radar(n_batch: int = 16) -> Radar:
    sample_rate_hz = 8.0e6
    transmitter = Transmitter(
        9.8e9,
        sample_rate_hz,
        1.0,
        30.0,
        n_batch / sample_rate_hz,
        sample_rate_hz / n_batch,
        waveform="noise",
    )
    return Radar(transmitter, Receiver(sample_rate_hz, 30.0, 3.0), Site(36.0, -78.9, 60.0))


def _path(radar: Radar, lag: float, doppler_hz: float = 0.0, power_w: float = 1.0):
    route_m = lag * SPEED_OF_LIGHT_MPS / radar.receiver.sample_rate_hz
    return PropagationPaths(
        delay_s=np.array([route_m / SPEED_OF_LIGHT_MPS]),
        doppler_hz=np.array([doppler_hz]),
        amplitude_linear=np.array([np.sqrt(power_w) + 0j]),
        aoa_rad=np.zeros((1, 2)),
        aod_rad=np.zeros((1, 2)),
        n_bounces=np.ones(1, dtype=np.int_),
    ), route_m


def test_noise_code_has_unit_power(rng):
    """Mean |x|^2 of 2^16 samples within five standard errors (1/sqrt(N)) of one."""
    code = noise_waveform(65_536, rng=rng)
    assert abs(np.mean(np.abs(code) ** 2) - 1.0) < 5.0 / np.sqrt(code.size)


def test_cubic_amplifier_keeps_unit_power_on_a_gaussian_code(rng):
    """E|f(x)|^2 = 1 by the normalisation; a3 = 0 is the identity."""
    code = noise_waveform(1_000_000, rng=rng)
    # |x|^6 moments: a million draws scatter about 0.3 %; 1 % is over three sigma.
    np.testing.assert_allclose(np.mean(np.abs(cubic_amplifier(code, -0.07)) ** 2), 1.0, rtol=0.01)
    np.testing.assert_array_equal(cubic_amplifier(code[:8], 0.0), code[:8])


def test_a_static_whole_lag_path_is_the_code_shifted(rng):
    """Noiseless: the echo is |a| e^{-j 2 pi R/lambda} x[(n - m) mod N], exactly."""
    radar = _noise_radar()
    paths, route_m = _path(radar, 5, power_w=4.0)
    code = noise_waveform(64 * 16, rng=rng)
    stream, reference = stream_baseband(paths, radar, 64, rng=None, reference=code)
    phase = np.exp(-2j * np.pi * route_m / radar.wavelength_m)
    np.testing.assert_array_equal(reference, code)
    # One complex multiply each: rtol 1e-12.
    np.testing.assert_allclose(stream, 2.0 * phase * np.roll(code, 5), rtol=1e-12)


def test_a_moving_path_is_the_code_delayed_batch_by_batch(rng):
    """Each batch holds the code delayed by its own route, a phase ramp on its spectrum; the
    carrier phase follows the route sample by sample.
    """
    radar, n_pulses = _noise_radar(), 32
    paths, _ = _path(radar, 7.3, doppler_hz=-90_000.0)
    code = noise_waveform(n_pulses * 16, rng=rng)
    stream, _ = stream_baseband(paths, radar, n_pulses, rng=None, reference=code)
    frequency = np.fft.fftfreq(code.size)
    time_s = np.arange(code.size) / radar.receiver.sample_rate_hz
    route_now_m = paths.bistatic_range_at_m(time_s, radar.wavelength_m)[:, 0]
    carrier = np.exp(-2j * np.pi * route_now_m / radar.wavelength_m)
    for batch in range(n_pulses):
        delay = route_now_m[batch * 16] * radar.receiver.sample_rate_hz / SPEED_OF_LIGHT_MPS
        delayed = np.fft.ifft(np.fft.fft(code) * np.exp(-2j * np.pi * frequency * delay))
        rows = slice(batch * 16, batch * 16 + 16)
        # The Taylor series stops at 1e-10 of the path; one FFT round trip on top.
        np.testing.assert_allclose(stream[rows], (delayed * carrier)[rows], rtol=0, atol=1e-9)


def test_the_amplifier_shapes_the_paths_not_the_reference(rng):
    """Every path carries f(x); the reference handed back is the digital code x."""
    radar = _noise_radar()
    paths, _ = _path(radar, 0)
    code = noise_waveform(16 * 16, rng=rng)
    amplifier = functools.partial(cubic_amplifier, a3=-0.07)
    stream, reference = stream_baseband(
        paths, radar, 16, rng=None, reference=code, amplifier=amplifier
    )
    np.testing.assert_array_equal(reference, code)
    np.testing.assert_allclose(stream, cubic_amplifier(code, -0.07), rtol=1e-12)


def test_stream_noise_is_the_noise_in_one_sample(rng):
    """With rng, the code is drawn and noise added at k T0 F f_s a sample (five standard errors)."""
    radar = _noise_radar()
    paths, _ = _path(radar, 0, power_w=0.0)
    stream, reference = stream_baseband(paths, radar, 4096, rng=rng)
    assert reference.shape == stream.shape == (4096 * 16,)
    measured = np.mean(np.abs(stream) ** 2) / radar.receiver.noise_power_per_sample_w
    assert abs(measured - 1.0) < 5.0 / np.sqrt(stream.size)


def test_stream_needs_a_noise_radar_and_a_code(rng, monostatic_radar):
    radar = _noise_radar()
    paths, _ = _path(radar, 0)
    with pytest.raises(ValueError, match="noise transmitter"):
        stream_baseband(paths, monostatic_radar, 4, rng=rng)
    with pytest.raises(ValueError, match="code"):
        stream_baseband(paths, radar, 4, rng=None)


class TestRangeGate:
    """The pulsed receiver records a window of each PRI: a gated cube is columns of the whole."""

    def test_a_gated_cube_is_those_columns_of_the_whole_one(self) -> None:
        """Bit for bit, with a closing echo that runs into the next row: one sample grid."""
        geometry = TargetGeometry.monostatic(5_400.0, -60.0)
        paths = line_of_sight_paths(S2_RADAR, geometry, 10.0)
        whole = pulsed_baseband(paths, S2_RADAR, 8, rng=None)
        gate_m = 36 * SPEED_OF_LIGHT_MPS / S2_RADAR.receiver.sample_rate_hz
        gated = pulsed_baseband(
            paths, S2_RADAR, 8, rng=None, range_gate_start_m=gate_m, n_gate_samples=50
        )
        assert range_gate_first_sample(gate_m, S2_RADAR.receiver.sample_rate_hz) == 36
        np.testing.assert_array_equal(gated, whole[:, 36:86])

    def test_the_gate_opens_on_the_nearest_sample(self) -> None:
        bin_m = SPEED_OF_LIGHT_MPS / 600.0e6
        assert range_gate_first_sample(4800.0, 600.0e6) == 9607
        assert range_gate_first_sample(9606.6 * bin_m, 600.0e6) == 9607

    def test_a_gate_past_the_repetition_interval_is_refused(self) -> None:
        paths = line_of_sight_paths(S2_RADAR, TargetGeometry.monostatic(4_000.0, 0.0), 10.0)
        with pytest.raises(ValueError, match="within"):
            pulsed_baseband(paths, S2_RADAR, 2, rng=None, n_gate_samples=101)
