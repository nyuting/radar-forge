"""Tests for :mod:`radar_forge.core.waveforms`.

Ground truth here is closed-form throughout: the range-resolution and beat-frequency
relations are exact algebra, and the chirp is checked against its own defining
instantaneous frequency rather than against recorded samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.core.waveforms import (
    beat_frequency_hz,
    lfm_chirp,
    range_from_beat_frequency_m,
    range_resolution_m,
    sweep_rate_hzps,
)

# A nominal 77 GHz automotive FMCW sweep, used throughout.
NOMINAL_BANDWIDTH_HZ = 1.0e9
NOMINAL_CHIRP_TIME_S = 40.0e-6


class TestRangeResolution:
    def test_matches_closed_form(self) -> None:
        # rtol at 1e-12: one divide in float64, so anything looser would hide
        # a genuine algebraic error such as a missing factor of two.
        np.testing.assert_allclose(
            range_resolution_m(NOMINAL_BANDWIDTH_HZ),
            SPEED_OF_LIGHT_MPS / (2.0 * NOMINAL_BANDWIDTH_HZ),
            rtol=1e-12,
        )

    def test_inversely_proportional_to_bandwidth(self) -> None:
        """Doubling bandwidth halves the resolution cell — the reason to chirp."""
        coarse = range_resolution_m(NOMINAL_BANDWIDTH_HZ)
        fine = range_resolution_m(2.0 * NOMINAL_BANDWIDTH_HZ)
        np.testing.assert_allclose(fine, coarse / 2.0, rtol=1e-12)

    def test_broadcasts_over_array_input(self) -> None:
        bandwidths = np.array([1e8, 1e9, 4e9])
        np.testing.assert_allclose(
            range_resolution_m(bandwidths),
            SPEED_OF_LIGHT_MPS / (2.0 * bandwidths),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("bad_bandwidth", [0.0, -1.0])
    def test_rejects_non_positive_bandwidth(self, bad_bandwidth: float) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            range_resolution_m(bad_bandwidth)


class TestSweepRate:
    def test_matches_closed_form(self) -> None:
        np.testing.assert_allclose(
            sweep_rate_hzps(NOMINAL_BANDWIDTH_HZ, NOMINAL_CHIRP_TIME_S),
            NOMINAL_BANDWIDTH_HZ / NOMINAL_CHIRP_TIME_S,
            rtol=1e-12,
        )

    @pytest.mark.parametrize(
        ("bandwidth_hz", "chirp_time_s"),
        [(0.0, 40e-6), (-1.0, 40e-6), (1e9, 0.0), (1e9, -1.0)],
    )
    def test_rejects_non_positive_parameters(
        self, bandwidth_hz: float, chirp_time_s: float
    ) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            sweep_rate_hzps(bandwidth_hz, chirp_time_s)


class TestBeatFrequency:
    def test_matches_closed_form(self) -> None:
        range_m = 42.0
        expected = (NOMINAL_BANDWIDTH_HZ / NOMINAL_CHIRP_TIME_S) * (
            2.0 * range_m / SPEED_OF_LIGHT_MPS
        )
        np.testing.assert_allclose(
            beat_frequency_hz(range_m, NOMINAL_BANDWIDTH_HZ, NOMINAL_CHIRP_TIME_S),
            expected,
            rtol=1e-12,
        )

    def test_is_linear_in_range(self) -> None:
        near = beat_frequency_hz(10.0, NOMINAL_BANDWIDTH_HZ, NOMINAL_CHIRP_TIME_S)
        far = beat_frequency_hz(30.0, NOMINAL_BANDWIDTH_HZ, NOMINAL_CHIRP_TIME_S)
        np.testing.assert_allclose(far, 3.0 * near, rtol=1e-12)

    def test_zero_range_gives_zero_beat(self) -> None:
        # A co-located target has no round-trip delay, so deramping leaves DC
        # exactly; atol rather than rtol because the true value is zero.
        np.testing.assert_allclose(
            beat_frequency_hz(0.0, NOMINAL_BANDWIDTH_HZ, NOMINAL_CHIRP_TIME_S),
            0.0,
            atol=0.0,
        )

    def test_round_trips_through_range(self) -> None:
        ranges_m = np.array([0.0, 1.5, 42.0, 250.0])
        beats = beat_frequency_hz(ranges_m, NOMINAL_BANDWIDTH_HZ, NOMINAL_CHIRP_TIME_S)
        recovered = range_from_beat_frequency_m(beats, NOMINAL_BANDWIDTH_HZ, NOMINAL_CHIRP_TIME_S)
        # Forward-then-inverse is four float64 operations; 1e-12 keeps it honest.
        np.testing.assert_allclose(recovered, ranges_m, rtol=1e-12, atol=1e-12)

    def test_rejects_negative_range(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            beat_frequency_hz(-1.0, NOMINAL_BANDWIDTH_HZ, NOMINAL_CHIRP_TIME_S)

    def test_rejects_negative_beat_frequency(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            range_from_beat_frequency_m(-1.0, NOMINAL_BANDWIDTH_HZ, NOMINAL_CHIRP_TIME_S)


class TestLfmChirp:
    def test_sample_count_is_half_open(self) -> None:
        """[0, T) holds exactly B*T samples — the t = T sample starts the next chirp."""
        chirp = lfm_chirp(bandwidth_hz=1e6, chirp_time_s=1e-3, sample_rate_hz=2e6)
        assert chirp.shape == (2000,)

    def test_is_constant_modulus(self) -> None:
        """Frequency modulation alone: a chirp carries no amplitude modulation."""
        chirp = lfm_chirp(bandwidth_hz=1e6, chirp_time_s=1e-3, sample_rate_hz=2e6)
        # rtol 1e-12: |exp(j phi)| is exactly 1 up to float64 round-off in sin/cos.
        np.testing.assert_allclose(np.abs(chirp), 1.0, rtol=1e-12)

    def test_amplitude_scales_linearly(self) -> None:
        kwargs = {"bandwidth_hz": 1e6, "chirp_time_s": 1e-3, "sample_rate_hz": 2e6}
        unit = lfm_chirp(**kwargs)  # type: ignore[arg-type]
        scaled = lfm_chirp(**kwargs, amplitude_linear=3.0)  # type: ignore[arg-type]
        np.testing.assert_allclose(scaled, 3.0 * unit, rtol=1e-12)

    def test_instantaneous_frequency_sweeps_linearly(self) -> None:
        """Unwrapped phase differences must reproduce f(t) = f0 + alpha*t."""
        bandwidth_hz = 1.0e6
        chirp_time_s = 1.0e-3
        sample_rate_hz = 4.0e6
        chirp = lfm_chirp(
            bandwidth_hz=bandwidth_hz,
            chirp_time_s=chirp_time_s,
            sample_rate_hz=sample_rate_hz,
        )

        phase_rad = np.unwrap(np.angle(chirp))
        measured_frequency_hz = np.diff(phase_rad) * sample_rate_hz / (2.0 * np.pi)

        # The finite difference samples f(t) at bin midpoints.
        midpoint_time_s = (np.arange(chirp.size - 1) + 0.5) / sample_rate_hz
        expected_frequency_hz = (bandwidth_hz / chirp_time_s) * midpoint_time_s

        # rtol 1e-6: a first-order difference of a quadratic phase is exact at the
        # midpoint in real arithmetic, so the residual here is float64 phase noise
        # amplified by the 1/(2*pi) * fs scaling.
        np.testing.assert_allclose(measured_frequency_hz, expected_frequency_hz, rtol=1e-6)

    def test_down_sweep_is_conjugate_of_up_sweep(self) -> None:
        kwargs = {"bandwidth_hz": 1e6, "chirp_time_s": 1e-3, "sample_rate_hz": 2e6}
        up = lfm_chirp(**kwargs, up_sweep=True)  # type: ignore[arg-type]
        down = lfm_chirp(**kwargs, up_sweep=False)  # type: ignore[arg-type]
        np.testing.assert_allclose(down, np.conjugate(up), rtol=1e-12)

    def test_start_frequency_offsets_the_sweep(self) -> None:
        """A centred sweep is the zero-started sweep shifted down by B/2."""
        bandwidth_hz = 1.0e6
        chirp_time_s = 1.0e-3
        sample_rate_hz = 4.0e6
        kwargs = {
            "bandwidth_hz": bandwidth_hz,
            "chirp_time_s": chirp_time_s,
            "sample_rate_hz": sample_rate_hz,
        }
        centred = lfm_chirp(**kwargs, start_frequency_hz=-bandwidth_hz / 2.0)  # type: ignore[arg-type]
        base = lfm_chirp(**kwargs)  # type: ignore[arg-type]

        time_s = np.arange(base.size, dtype=np.float64) / sample_rate_hz
        shift = np.exp(-1j * 2.0 * np.pi * (bandwidth_hz / 2.0) * time_s)
        np.testing.assert_allclose(centred, base * shift, rtol=1e-9, atol=1e-12)

    def test_matched_filter_compresses_by_time_bandwidth_product(self) -> None:
        """Autocorrelation mainlobe narrows from T to about 1/B: gain of B*T."""
        bandwidth_hz = 1.0e6
        chirp_time_s = 1.0e-3
        sample_rate_hz = 4.0e6
        chirp = lfm_chirp(
            bandwidth_hz=bandwidth_hz,
            chirp_time_s=chirp_time_s,
            sample_rate_hz=sample_rate_hz,
        )

        compressed = np.abs(np.correlate(chirp, chirp, mode="full"))
        peak_index = int(np.argmax(compressed))

        # The peak sits at zero lag, and equals the pulse energy there.
        assert peak_index == chirp.size - 1
        np.testing.assert_allclose(compressed[peak_index], float(chirp.size), rtol=1e-9)

        # Mainlobe width at -3 dB should be about one resolution cell, 1/B.
        half_power = compressed[peak_index] / np.sqrt(2.0)
        above = np.flatnonzero(compressed >= half_power)
        mainlobe_width_s = (above[-1] - above[0] + 1) / sample_rate_hz
        expected_width_s = 1.0 / bandwidth_hz
        # Factor-of-two band: the exact -3 dB width of an unweighted LFM is
        # ~0.886/B, and the sampled grid quantises it further.
        assert 0.5 * expected_width_s <= mainlobe_width_s <= 2.0 * expected_width_s

    def test_conserves_energy(self) -> None:
        """Parseval: a unit-modulus chirp of N samples carries energy N."""
        chirp = lfm_chirp(bandwidth_hz=1e6, chirp_time_s=1e-3, sample_rate_hz=2e6)
        time_energy = float(np.sum(np.abs(chirp) ** 2))
        freq_energy = float(np.sum(np.abs(np.fft.fft(chirp)) ** 2) / chirp.size)
        # rtol 1e-10: one forward FFT, per the tolerance table in testing.md.
        np.testing.assert_allclose(freq_energy, time_energy, rtol=1e-10)

    def test_occupies_the_swept_bandwidth(self) -> None:
        """Spectral energy is confined to [0, B] for a zero-started up-sweep."""
        bandwidth_hz = 1.0e6
        sample_rate_hz = 4.0e6
        chirp = lfm_chirp(
            bandwidth_hz=bandwidth_hz,
            chirp_time_s=1.0e-3,
            sample_rate_hz=sample_rate_hz,
        )
        spectrum = np.abs(np.fft.fft(chirp)) ** 2
        frequency_hz = np.fft.fftfreq(chirp.size, d=1.0 / sample_rate_hz)

        in_band = (frequency_hz >= 0.0) & (frequency_hz <= bandwidth_hz)
        in_band_fraction = float(np.sum(spectrum[in_band]) / np.sum(spectrum))
        # Fresnel ripple and the rectangular envelope leak a little past the band
        # edges; 99% in-band is the practical bar for an unweighted LFM.
        assert in_band_fraction > 0.99

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"bandwidth_hz": 0.0}, "bandwidth_hz"),
            ({"bandwidth_hz": -1e6}, "bandwidth_hz"),
            ({"chirp_time_s": 0.0}, "chirp_time_s"),
            ({"chirp_time_s": -1e-3}, "chirp_time_s"),
            ({"sample_rate_hz": 0.0}, "sample_rate_hz"),
            ({"sample_rate_hz": -2e6}, "sample_rate_hz"),
        ],
    )
    def test_rejects_non_positive_parameters(self, kwargs: dict[str, float], match: str) -> None:
        base = {"bandwidth_hz": 1e6, "chirp_time_s": 1e-3, "sample_rate_hz": 2e6}
        with pytest.raises(ValueError, match=match):
            lfm_chirp(**{**base, **kwargs})  # type: ignore[arg-type]

    def test_rejects_sample_rate_below_bandwidth(self) -> None:
        with pytest.raises(ValueError, match="would alias"):
            lfm_chirp(bandwidth_hz=2e6, chirp_time_s=1e-3, sample_rate_hz=1e6)

    def test_rejects_sweep_too_short_to_sample(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            lfm_chirp(bandwidth_hz=1e6, chirp_time_s=1e-9, sample_rate_hz=2e6)
