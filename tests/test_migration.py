"""Range migration: the range spectrum, the keystone, its fold factor, Doppler-rate compensation.

Unit tests run on small cubes built in the range-frequency domain, where an echo is
``|P(f)|^2 exp(-j 2 pi (f0 + f) R_b(t) / c)``, so the truth is the model itself. The O(N^2)
DFT keystone and the Radon-Fourier transform are oracles here, not library code. The slow
tests run scenario 004's full-size chain: CFAR on the uncorrected and corrected maps (A3, A4)
and the fold-factor search (A6), 100 and 200 trials.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.signal import czt

from radar_forge.constants import SPEED_OF_LIGHT_MPS
from radar_forge.detection import cfar_detect
from radar_forge.dsp import doppler_fft, matched_filter
from radar_forge.migration import (
    doppler_rate_compensation,
    fold_factor_search,
    integrated_peak,
    keystone,
    range_spectrum,
    second_order_keystone,
)
from radar_forge.pipeline import compensate_migration, pulse_compress, synthesise
from radar_forge.scenario import RangeGate, load_scenario, truth_geometry
from radar_forge.signal import lfm_chirp

REPO = Path(__file__).resolve().parents[1]

# A small C-band cube: 256 pulses, 64 range bins of c / f_s, the scenario's 5 % fractional
# bandwidth; 45 m/s, under half the 100 m/s rate span, walks 6.9 bins in the 128 ms dwell.
CARRIER_HZ = 6.0e9
SAMPLE_RATE_HZ = 360.0e6
BANDWIDTH_HZ = 300.0e6
PRF_HZ = 2.0e3
N_PULSES, N_RANGE = 256, 64
BIN_M = SPEED_OF_LIGHT_MPS / SAMPLE_RATE_HZ


def _model_cube(
    rates_mps: list[float],
    start_bins: list[float],
    acceleration_mps2: float = 0.0,
    n_pulses: int = N_PULSES,
    carrier_hz: float = CARRIER_HZ,
) -> np.ndarray:
    """Return ``data_rfst`` for point echoes with a flat band, built in the range spectrum."""
    n_fft = 2 * N_RANGE
    frequency_hz = np.fft.fftfreq(n_fft, 1.0 / SAMPLE_RATE_HZ)
    band = (np.abs(frequency_hz) <= BANDWIDTH_HZ / 2).astype(float)
    time_s = np.arange(n_pulses)[:, None] / PRF_HZ
    spectrum = np.zeros((n_pulses, n_fft), dtype=complex)
    for rate_mps, start in zip(rates_mps, start_bins, strict=True):
        range_m = start * BIN_M + rate_mps * time_s + 0.5 * acceleration_mps2 * time_s**2
        spectrum += band * np.exp(
            -2j * np.pi * (carrier_hz + frequency_hz) * range_m / SPEED_OF_LIGHT_MPS
        )
    return np.fft.ifft(spectrum, axis=1)[:, :N_RANGE]


def _peak_track(data: np.ndarray) -> np.ndarray:
    """Each pulse's range peak, fractional: the vertex of a parabola through |x|."""
    magnitude = np.abs(data)
    peak = np.clip(np.argmax(magnitude, axis=1), 1, magnitude.shape[1] - 2)
    rows = np.arange(magnitude.shape[0])
    below, centre, above = (magnitude[rows, peak + k] for k in (-1, 0, 1))
    return peak + 0.5 * (below - above) / (below - 2 * centre + above)


def _dft_keystone_oracle(data: np.ndarray, n_fft: int) -> np.ndarray:
    """The O(N^2) band-limited interpolation the keystone evaluates, written out directly."""
    spectrum = np.fft.fft(data, n=n_fft, axis=1)
    frequency_hz = np.fft.fftfreq(n_fft, 1.0 / SAMPLE_RATE_HZ)
    n_pulses = data.shape[0]
    signed = np.arange(n_pulses) - n_pulses // 2
    slow_dft = np.fft.fftshift(np.fft.fft(spectrum, axis=0), axes=0)
    out = np.empty_like(spectrum)
    # One frequency at a time: the oracle is the plain sum, as slow as it looks.
    for column, f_hz in enumerate(frequency_hz):
        scale = CARRIER_HZ / (CARRIER_HZ + f_hz)
        basis = np.exp(2j * np.pi * np.outer(np.arange(n_pulses) * scale, signed) / n_pulses)
        out[:, column] = basis @ slow_dft[:, column] / n_pulses
    return np.fft.ifft(out, axis=1)[:, : data.shape[1]]


def _radon_fourier_peak(data: np.ndarray, start_bin: float, rate_mps: float) -> float:
    """The RFT oracle: integrate along the walking range line with the Doppler phase undone."""
    time_s = np.arange(data.shape[0]) / PRF_HZ
    position = start_bin + rate_mps * time_s / BIN_M
    n_fft = 2 * data.shape[1]
    spectrum = np.fft.fft(data, n=n_fft, axis=1)
    # Band-limited interpolation of each pulse's profile at its fractional position.
    shift = np.exp(2j * np.pi * np.fft.fftfreq(n_fft)[None, :] * position[:, None])
    samples = np.sum(spectrum * shift, axis=1) / n_fft
    doppler = np.exp(2j * np.pi * CARRIER_HZ * rate_mps * time_s / SPEED_OF_LIGHT_MPS)
    return float(np.abs(np.sum(samples * doppler)) ** 2)


class TestRangeSpectrum:
    def test_matched_filter_output_has_the_spectrum_x_times_p_conjugate(self) -> None:
        """A17: the range spectrum of a compression is X(f) P*(f), in the convolution length.

        Catches a sign flip in the frequency axis or a conjugate on the wrong side. The group
        delay n_ref - 1 is rolled back first, so the product holds exactly; rtol 1e-12 is a few
        FFTs' rounding.
        """
        rng = np.random.default_rng(20261106)
        reference = lfm_chirp(BANDWIDTH_HZ, 0.2e-6, SAMPLE_RATE_HZ)
        samples = rng.normal(size=(3, 100)) + 1j * rng.normal(size=(3, 100))
        compressed = matched_filter(samples, reference, axis=1)
        n_fft = compressed.shape[1]
        rolled = np.roll(compressed, -(reference.size - 1), axis=1)
        result = range_spectrum(rolled, sample_rate_hz=SAMPLE_RATE_HZ, n_range_fft=n_fft)
        expected = np.fft.fft(samples, n=n_fft, axis=1) * np.conj(np.fft.fft(reference, n=n_fft))
        np.testing.assert_allclose(result.spectrum, expected, rtol=1e-12, atol=1e-9)

    def test_frequency_axis_is_signed_and_padding_defaults_to_twice_the_bins(self) -> None:
        """The axis is fftfreq (f0 + f is the radio frequency); 64 bins pad to 128."""
        result = range_spectrum(np.zeros((2, 64)), sample_rate_hz=SAMPLE_RATE_HZ)
        assert result.spectrum.shape == (2, 128)
        np.testing.assert_array_equal(
            result.range_frequency_hz, np.fft.fftfreq(128, 1.0 / SAMPLE_RATE_HZ)
        )
        assert result.range_frequency_hz.min() < 0.0


class TestKeystone:
    def test_chirp_z_equals_the_dft_interpolation_oracle(self) -> None:
        """A16: Bluestein's chirp-z keystone is the O(N^2) band-limited interpolation.

        rtol 1e-10: both are float64 sums of 64 terms of unit size.
        """
        data = _model_cube([25.0, -40.0], [20.0, 41.3], n_pulses=64)
        fast = keystone(data, carrier_frequency_hz=CARRIER_HZ, sample_rate_hz=SAMPLE_RATE_HZ)
        oracle = _dft_keystone_oracle(data, 2 * N_RANGE)
        np.testing.assert_allclose(fast.data_rfst, oracle, rtol=1e-10, atol=1e-12)

    def test_chirp_z_matches_scipy_per_frequency(self) -> None:
        """The batched transform is scipy.signal.czt along W = exp(j 2 pi s / N), row by row."""
        data = _model_cube([25.0], [20.0], n_pulses=64)
        result = keystone(data, carrier_frequency_hz=CARRIER_HZ, sample_rate_hz=SAMPLE_RATE_HZ)
        spectrum = np.fft.fft(data, n=2 * N_RANGE, axis=1)
        shifted = np.fft.fftshift(np.fft.fft(spectrum, axis=0), axes=0)
        pulse = np.arange(64)
        expected = np.empty_like(spectrum)
        # One scipy call per frequency, as the reference implementation would make.
        for column, scale in enumerate(result.slow_time_scale):
            ratio = np.exp(2j * np.pi * scale / 64)
            expected[:, column] = (
                czt(shifted[:, column], 64, ratio, 1.0) * np.exp(-1j * np.pi * scale * pulse) / 64
            )
        np.testing.assert_allclose(
            result.data_rfst, np.fft.ifft(expected, axis=1)[:, :N_RANGE], rtol=1e-10, atol=1e-12
        )

    @pytest.mark.parametrize("method", ["czt", "sinc"])
    def test_walk_is_removed_for_two_velocities_at_once(self, method: str) -> None:
        """A2 in small: each target's peak track stays within half a bin after the keystone.

        Two speeds in one cube, so a velocity-dependent correction cannot pass. The end pulses
        ring and are excluded; the sinc baseline's truncation shows in its loss, not here.
        """
        walks = [45.0, -30.0]
        for rate_mps, start in zip(walks, [15.0, 41.3], strict=True):
            data = _model_cube([rate_mps], [start])
            assert np.ptp(_peak_track(data)) > 3.0
            result = keystone(
                data, carrier_frequency_hz=CARRIER_HZ, sample_rate_hz=SAMPLE_RATE_HZ, method=method
            )
            assert np.ptp(_peak_track(result.data_rfst)[1:-1]) < 0.5

    def test_keystoned_peak_matches_the_radon_fourier_oracle(self) -> None:
        """The keystone integrates as the RFT does along the true line: within 0.2 dB.

        The RFT is the coherent sum along the walk [Xu et al. 2011]; the keystone reaches it
        without knowing the velocity. 0.2 dB covers the DFT interpolation's ringing end pulses.
        """
        data = _model_cube([45.0], [15.0])
        result = keystone(data, carrier_frequency_hz=CARRIER_HZ, sample_rate_hz=SAMPLE_RATE_HZ)
        ratio_db = 10 * np.log10(
            integrated_peak(result.data_rfst).power_linear / _radon_fourier_peak(data, 15.0, 45.0)
        )
        assert abs(ratio_db) < 0.2

    def test_trailing_channels_are_keystoned_each_alike(self) -> None:
        """A trailing n_rx axis is carried whole: each channel equals its own keystone."""
        data = np.stack([_model_cube([45.0], [15.0]), 2j * _model_cube([-10.0], [30.0])], axis=-1)
        both = keystone(data, carrier_frequency_hz=CARRIER_HZ, sample_rate_hz=SAMPLE_RATE_HZ)
        for channel in range(2):
            alone = keystone(
                data[..., channel], carrier_frequency_hz=CARRIER_HZ, sample_rate_hz=SAMPLE_RATE_HZ
            )
            np.testing.assert_allclose(both.data_rfst[..., channel], alone.data_rfst, rtol=1e-12)

    def test_a_cube_is_never_modified_in_place(self) -> None:
        data = _model_cube([25.0], [20.0])
        before = data.copy()
        keystone(data, carrier_frequency_hz=CARRIER_HZ, sample_rate_hz=SAMPLE_RATE_HZ)
        np.testing.assert_array_equal(data, before)


class TestFoldFactor:
    def test_search_finds_the_fold_of_an_ambiguous_target(self) -> None:
        """A6 in small: a target at 1.4 PRFs of Doppler keystones at F = 1, not at the alias.

        At 10 GHz the band is 3 % of the carrier, so 1.4 PRFs stays in fold 1 across it; rate
        -1.4 lambda PRF is closing (positive Doppler) and walks 6.5 bins.
        """
        carrier_hz = 10.0e9
        rate_mps = -1.4 * SPEED_OF_LIGHT_MPS / carrier_hz * PRF_HZ
        data = _model_cube([rate_mps], [10.0], n_pulses=128, carrier_hz=carrier_hz)
        assert np.ptp(_peak_track(data)) > 5.0
        found = fold_factor_search(
            data,
            carrier_frequency_hz=carrier_hz,
            sample_rate_hz=SAMPLE_RATE_HZ,
            bandwidth_hz=BANDWIDTH_HZ,
            fold_factor_max=3,
        )
        assert found.fold_factor == 1
        assert int(np.argmax(found.integrated_peak_linear)) == 4
        np.testing.assert_allclose(found.doppler_cycles_per_pulse, 1.4, atol=1e-3)
        assert np.ptp(_peak_track(found.data_rfst)[1:-1]) < 0.5
        explicit = keystone(
            data, carrier_frequency_hz=carrier_hz, sample_rate_hz=SAMPLE_RATE_HZ, fold_factor=1
        )
        np.testing.assert_allclose(found.data_rfst, explicit.data_rfst, rtol=1e-9, atol=1e-12)

    def test_a_target_spanning_two_folds_across_the_band_is_refused(self) -> None:
        """At 1.45 PRFs and 1 GHz, 30 % fractional bandwidth, the band spans 1.23 to 1.67
        PRFs, folds 1 and 2: no one fold factor keystones it, and it is refused."""
        carrier_hz = 1.0e9
        rate_mps = -1.45 * SPEED_OF_LIGHT_MPS / carrier_hz * PRF_HZ
        data = _model_cube([rate_mps], [10.0], n_pulses=128, carrier_hz=carrier_hz)
        with pytest.raises(ValueError, match="spans fold factors"):
            fold_factor_search(
                data,
                carrier_frequency_hz=carrier_hz,
                sample_rate_hz=SAMPLE_RATE_HZ,
                bandwidth_hz=BANDWIDTH_HZ,
                fold_factor_max=2,
            )


class TestDopplerRate:
    def test_search_finds_the_acceleration_and_recovers_the_peak(self) -> None:
        """A7 in small: the curve peaks within one step of the truth; the peak is restored.

        At 10 GHz, -100 m/s^2 spreads 13.6 Doppler bins and bends the range a quarter of a bin;
        the reference is the same target at constant rate. 0.3 dB is spec A4's bound.
        """
        carrier_hz, acceleration_mps2 = 10.0e9, -100.0
        data = _model_cube(
            [5.0], [30.0], acceleration_mps2=acceleration_mps2, n_pulses=128, carrier_hz=carrier_hz
        )
        steady = _model_cube([5.0], [30.0], n_pulses=128, carrier_hz=carrier_hz)
        kwargs = {"carrier_frequency_hz": carrier_hz, "sample_rate_hz": SAMPLE_RATE_HZ}
        keystoned = keystone(data, **kwargs).data_rfst
        found = doppler_rate_compensation(
            keystoned, **kwargs, prf_hz=PRF_HZ, accelerations_mps2=np.arange(-300.0, 1.0, 10.0)
        )
        assert abs(found.acceleration_mps2 - acceleration_mps2) <= 10.0
        ideal = integrated_peak(keystone(steady, **kwargs).data_rfst).power_linear
        before = integrated_peak(keystoned).power_linear
        after = integrated_peak(found.data_rfst).power_linear
        assert 10 * np.log10(before / ideal) < -3.0
        assert 10 * np.log10(after / ideal) > -0.3

    def test_second_order_keystone_straightens_a_quadratic_range_history(self) -> None:
        """A18: a target accelerating from rest at pulse 0, 6 bins of curvature, < 0.5 bin.

        At 1 GHz (30 % fractional bandwidth) the Doppler ends at 524 Hz, under PRF / 2, so
        slow time is not aliased. Pulses the lowest frequency reads past the dwell, where DFT
        interpolation wraps, are left out: the first (N - 1) / max sqrt(f0 / (f0 + f)).
        """
        carrier_hz, n_pulses = 1.0e9, 128
        acceleration_mps2 = 2.0 * 6.0 * BIN_M / ((n_pulses - 1) / PRF_HZ) ** 2
        data = _model_cube(
            [0.0],
            [20.0],
            acceleration_mps2=acceleration_mps2,
            n_pulses=n_pulses,
            carrier_hz=carrier_hz,
        )
        assert np.ptp(_peak_track(data)) > 5.5
        result = second_order_keystone(
            data, carrier_frequency_hz=carrier_hz, sample_rate_hz=SAMPLE_RATE_HZ
        )
        inside = int((n_pulses - 1) / result.slow_time_scale.max())
        assert np.ptp(_peak_track(result.data_rfst)[1:inside]) < 0.5


class TestIntegratedPeak:
    def test_an_off_grid_tone_reads_its_full_power(self) -> None:
        """A tone 0.37 bins off the Doppler grid and 0.41 off range reads N^2 times one pulse.

        The 1/16-bin range grid leaves at most a 1/32-bin miss: under 0.01 dB here.
        """
        data = _model_cube([0.0], [20.41], n_pulses=64)
        data = data * np.exp(2j * np.pi * 10.37 * np.arange(64) / 64)[:, None]
        single = integrated_peak(data[:1]).power_linear
        peak = integrated_peak(data)
        assert abs(10 * np.log10(peak.power_linear / (64**2 * single))) < 0.01
        np.testing.assert_allclose(peak.doppler_bin, 10.37, atol=1e-6)
        np.testing.assert_allclose(peak.range_bin, 20.4375, atol=1e-12)


@pytest.fixture(scope="module")
def full_size(request: pytest.FixtureRequest) -> dict[str, object]:
    """Scenario 004's three variants at frame 0: radar, gate, and the noiseless compressed cube."""
    found: dict[str, object] = {}
    for variant in ("k1", "k2", "k3"):
        scenario = load_scenario(REPO / "config" / f"scenario_004_keystone_{variant}.toml")
        truth = truth_geometry(scenario)[0]
        radar, gate = scenario.radar, scenario.gates[0]
        cube = synthesise(radar, truth, [0.1], 1024, rng=None, gate=gate)
        found[variant] = (scenario, truth, pulse_compress(cube, radar, range_gate_start_m=4800.0))
    return found


def _detected_fraction(
    scenario: object, clean_map: np.ndarray, cell: tuple[int, int], n_trials: int, seed: int
) -> float:
    """Fraction of trials whose CFAR detects within one cell of ``cell``, on a range strip.

    A trial adds a compressed-noise map to the noiseless one; the noise is white noise
    matched-filtered along range and Doppler-transformed, the statistics of the chain's map
    (the keystone resamples slow time near-unitarily). CFAR runs on 128 range bins about the
    target, enough for its 16 + 4 cells either side.
    """
    radar = scenario.radar  # type: ignore[attr-defined]
    detection = scenario.detection  # type: ignore[attr-defined]
    reference = lfm_chirp(
        radar.transmitter.bandwidth_hz,
        radar.transmitter.chirp_duration_s,
        radar.receiver.sample_rate_hz,
    )
    doppler_bin, range_bin = cell
    strip = slice(range_bin - 64, range_bin + 64)
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(radar.receiver.noise_power_per_sample_w / 2)
    hits = 0
    # One trial at a time: each is a fresh 1024 x 128 map.
    for _ in range(n_trials):
        white = rng.normal(0, sigma, (1024, 128 + reference.size - 1, 2)) @ [1, 1j]
        noise = matched_filter(white, reference, axis=1)[
            :, reference.size - 1 : -reference.size + 1
        ]
        power = np.abs(clean_map[:, strip] + doppler_fft(noise, axis=0)) ** 2
        detected, _ = cfar_detect(
            power, detection.pfa, n_train=detection.n_train, n_guard=detection.n_guard, axis=-1
        )
        rows, cols = np.nonzero(detected)
        near = (np.abs(rows - doppler_bin) <= 1) & (np.abs(cols - 64) <= 1)
        hits += bool(np.any(near))
    return hits / n_trials


# Slow: 100 trials of a 1024-pulse map and CFAR for each of three variants, twice.
@pytest.mark.slow
@pytest.mark.parametrize("variant", ["k1", "k2", "k3"])
def test_cfar_misses_the_uncorrected_target_and_finds_the_corrected_one(full_size, variant):
    """A3 and A4: at 28.89 dB, CA-CFAR at Pfa 1e-6 finds the target in <= 5 % of uncorrected
    maps and >= 95 % of corrected ones, 100 trials each."""
    scenario, _, clean = full_size[variant]
    corrected = compensate_migration(clean, scenario.radar, scenario.migration).data_rfst
    fractions = []
    for data in (clean, corrected):
        clean_map = doppler_fft(data, axis=0)
        cell = np.unravel_index(int(np.argmax(np.abs(clean_map))), clean_map.shape)
        fractions.append(_detected_fraction(scenario, clean_map, cell, 100, 20261106))
    assert fractions[0] <= 0.05
    assert fractions[1] >= 0.95


# Slow: 200 keystones and fold searches of 1024 pulses.
@pytest.mark.slow
def test_k3_fold_factor_is_found_in_every_noisy_trial(full_size):
    """A6: F = 1 in 200 of 200 trials at 28.89 dB.

    K3's radar, target and noise per sample, through a 256-sample gate opening 150 bins short
    of the target: the fold factor does not depend on the gate's length, and the search costs
    a quarter. Noise is added to the cube, as the chain adds it.
    """
    scenario, truth, _ = full_size["k3"]
    radar, rng = scenario.radar, np.random.default_rng(6)
    gate = RangeGate(5000.0 - 150 * SPEED_OF_LIGHT_MPS / 600.0e6, 256)
    start_m = gate.range_gate_start_m
    cube = synthesise(radar, truth, [0.1], 1024, rng=None, gate=gate)
    clean = pulse_compress(cube, radar, range_gate_start_m=start_m)
    sigma = np.sqrt(radar.receiver.noise_power_per_sample_w / 2)
    found = []
    # One trial at a time: each searches seven fold factors over a whole cube.
    for _ in range(200):
        noise = rng.normal(0, sigma, (1024, 256, 2)) @ [1, 1j]
        noisy = clean + pulse_compress(noise, radar, range_gate_start_m=start_m)
        found.append(compensate_migration(noisy, radar, scenario.migration).fold_factor)
    assert found == [1] * 200
