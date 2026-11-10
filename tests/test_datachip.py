"""The data chip, the tone interpolators, the polynomial-phase estimators and their bound.

Fast tests use noiseless tones and chips built from the range-frequency model, where the
truth is exact. The slow tests are spec 004's Monte Carlo: the chip of scenario 004's own
noiseless chain plus compressed noise, 500 trials (A8, A9), and the jerk (A14). A trial adds
white noise matched-filtered along range, the chip's noise statistics (the keystone and the
Doppler-rate filter resample slow time near-unitarily); the chain itself runs once.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from radar_forge.constants import SPEED_OF_LIGHT_MPS
from radar_forge.datachip import (
    FINE_METHODS,
    ChipProcessing,
    DataChip,
    cramer_rao_bounds,
    crop_range_doppler,
    crop_slow_time,
    estimate_fine,
    estimate_tone,
)
from radar_forge.dsp import delay_bin_centers_m, doppler_bin_centers_mps, matched_filter
from radar_forge.migration import compensate_migration, integrated_peak, pulse_compress
from radar_forge.pipeline import synthesise
from radar_forge.scenario import load_scenario, truth_geometry
from radar_forge.signal import lfm_chirp

REPO = Path(__file__).resolve().parents[1]

# Scenario 004's waveform: 9.8 GHz, 500 MHz LFM of 0.5 us at 600 MHz, PRF 50 kHz.
CARRIER_HZ, BANDWIDTH_HZ, CHIRP_S = 9.8e9, 500.0e6, 0.5e-6
SAMPLE_RATE_HZ, PRF_HZ, N_PULSES = 600.0e6, 50.0e3, 1024
WAVELENGTH_M = SPEED_OF_LIGHT_MPS / CARRIER_HZ
BIN_M = SPEED_OF_LIGHT_MPS / SAMPLE_RATE_HZ
# The nominal integrated SNR, 28.886 dB, and the spec's lowest asserted one.
NOMINAL_SNR = 10 ** (28.886 / 10)
TONE_METHODS = FINE_METHODS[:5]


def _processing(noise_power_w: float = 1.0, fold_factor: int = 0) -> ChipProcessing:
    return ChipProcessing(
        CARRIER_HZ,
        BANDWIDTH_HZ,
        SAMPLE_RATE_HZ,
        PRF_HZ,
        noise_power_w,
        keystone_applied=True,
        fold_factor=fold_factor,
    )


def _tone(offset_bins: float, n: int = 1024) -> np.ndarray:
    return np.exp(2j * np.pi * (100.0 + offset_bins) * np.arange(n) / n)


def _model_chip(
    delay_bins: float,
    rate_mps: float,
    acceleration_mps2: float = 0.0,
    jerk_mps3: float = 0.0,
    n_range: int = 33,
) -> np.ndarray:
    """A keystoned chip from the range-frequency model, the chirp's own spectrum, unit peak."""
    n_fft = 512
    frequency_hz = np.fft.fftfreq(n_fft, 1.0 / SAMPLE_RATE_HZ)
    spectrum = np.abs(np.fft.fft(lfm_chirp(BANDWIDTH_HZ, CHIRP_S, SAMPLE_RATE_HZ), n_fft)) ** 2
    tau_s = np.arange(N_PULSES)[:, None] / PRF_HZ
    scale = CARRIER_HZ / (CARRIER_HZ + frequency_hz)
    phase_m = (
        (CARRIER_HZ + frequency_hz) * delay_bins * BIN_M
        + CARRIER_HZ * rate_mps * tau_s
        + CARRIER_HZ * scale * acceleration_mps2 * tau_s**2 / 2
        + CARRIER_HZ * scale**2 * jerk_mps3 * tau_s**3 / 6
    )
    chip = np.fft.ifft(spectrum * np.exp(-2j * np.pi * phase_m / SPEED_OF_LIGHT_MPS), axis=1)
    return chip[:, :n_range] / np.abs(chip).max()


def _compressed_noise(rng: np.random.Generator, n_range: int, power_w: float) -> np.ndarray:
    """White noise matched-filtered along range, ``power_w`` per compressed cell.

    The fully overlapped part of the correlation of n_range + n_ref - 1 white samples: the
    compressed noise of any n_range cells of a longer record.
    """
    reference = lfm_chirp(BANDWIDTH_HZ, CHIRP_S, SAMPLE_RATE_HZ)
    sigma = np.sqrt(power_w / reference.size / 2)
    white = rng.normal(0, sigma, (N_PULSES, n_range + reference.size - 1, 2)) @ [1, 1j]
    full = matched_filter(white, reference, axis=1)
    return full[:, reference.size - 1 : reference.size - 1 + n_range]


def _settings_fields(scenario: object) -> dict[str, object]:
    """A scenario's [migration] table as compensate_migration's keyword arguments."""
    settings = scenario.migration  # type: ignore[attr-defined]
    return {
        "method": settings.method,
        "keystone": settings.keystone,
        "fold_factor_max": settings.fold_factor_max,
        "accelerations_mps2": settings.accelerations_mps2,
    }


class TestChip:
    def test_slow_time_chip_round_trips_exactly_and_clips_at_the_edge(self) -> None:
        """A10: integer cells map back with ==, metres are the parent's values, an edge clips."""
        parent = np.arange(8 * 40).reshape(8, 40) * (1 + 1j)
        range_m = delay_bin_centers_m(40, SAMPLE_RATE_HZ, range_gate_start_m=4800.0)
        rate_mps = doppler_bin_centers_mps(8, 1 / PRF_HZ, WAVELENGTH_M)
        chip = crop_slow_time(
            parent,
            2,
            range_bin_centers_m=range_m,
            rate_bin_centers_mps=rate_mps,
            processing=_processing(),
            half_width_bins=4,
        )
        assert chip.data.shape == (8, 7) and chip.range_offset == 0
        rows, cols = np.meshgrid(np.arange(8), np.arange(7), indexing="ij")
        parent_rows, parent_cols = chip.parent_index(rows, cols)
        np.testing.assert_array_equal(parent[parent_rows, parent_cols], chip.data)
        np.testing.assert_array_equal(chip.range_bin_centers_m, range_m[:7])
        np.testing.assert_allclose(chip.range_at_m(3.25), range_m[0] + 3.25 * BIN_M, rtol=1e-12)
        np.testing.assert_array_equal(chip.rate_bin_centers_mps, rate_mps)

    def test_doppler_chip_wraps_across_the_fold_and_keeps_the_parents_rates(self) -> None:
        """A10: a Doppler crop through bin 0 takes the circle's far end, rate jump included."""
        parent = np.arange(16 * 40).reshape(16, 40) + 0j
        range_m = delay_bin_centers_m(40, SAMPLE_RATE_HZ)
        rate_mps = doppler_bin_centers_mps(16, 1 / PRF_HZ, WAVELENGTH_M)
        chip = crop_range_doppler(
            parent,
            1,
            38,
            range_bin_centers_m=range_m,
            rate_bin_centers_mps=rate_mps,
            processing=_processing(),
            half_width_bins=3,
        )
        assert chip.domain == "doppler" and chip.data.shape == (7, 5)
        rows, cols = np.meshgrid(np.arange(7), np.arange(5), indexing="ij")
        parent_rows, parent_cols = chip.parent_index(rows, cols)
        np.testing.assert_array_equal(parent_rows[:, 0], [14, 15, 0, 1, 2, 3, 4])
        np.testing.assert_array_equal(parent[parent_rows, parent_cols], chip.data)
        np.testing.assert_array_equal(chip.rate_bin_centers_mps, rate_mps[parent_rows[:, 0]])

    def test_a_sixteen_bin_chip_keeps_the_target_energy(self) -> None:
        """A11: +-16 range bins keep all but 0.05 dB of a compressed echo, worst offset.

        The echo is the chirp's own autocorrelation at 41 sub-bin offsets (computed -0.040 dB;
        +-4 bins would lose 0.144 dB).
        """
        n_fft = 8192
        spectrum = np.abs(np.fft.fft(lfm_chirp(BANDWIDTH_HZ, CHIRP_S, SAMPLE_RATE_HZ), n_fft)) ** 2
        index = np.fft.fftfreq(n_fft) * n_fft
        worst_db = 0.0
        for offset in np.linspace(-0.5, 0.5, 41):
            energy = (
                np.abs(
                    np.fft.ifft(spectrum * np.exp(-2j * np.pi * index * (1000 + offset) / n_fft))
                )
                ** 2
            )
            peak = int(np.argmax(energy))
            worst_db = min(
                worst_db, 10 * np.log10(energy[peak - 16 : peak + 17].sum() / energy.sum())
            )
        assert worst_db >= -0.05


class TestTone:
    @pytest.mark.parametrize(
        ("method", "bound"),
        [("quinn", 2e-6), ("jacobsen", 2e-6), ("candan", 1e-6), ("aboutanios_mulgrew", 1e-9)],
    )
    def test_interpolator_bias_on_a_noiseless_tone(self, method: str, bound: float) -> None:
        """A12: the largest error over offsets in [-0.5, 0.5] on a 1024-point tone.

        Quinn and Jacobsen are biased O(1/N^2) (computed 1.18e-6), Candan corrects most of it
        (3.9e-7), and two Aboutanios-Mulgrew iterations reach rounding (5e-13).
        """
        errors = [
            abs(estimate_tone(_tone(o), method=method) - 100 - o)
            for o in np.linspace(-0.5, 0.5, 101)
        ]
        assert max(errors) <= bound

    def test_parabolic_interpolation_is_biased_by_a_fifth_of_a_bin(self) -> None:
        """A12: the vertex through |X| misses a rectangular-window tone by up to 0.234 bins."""
        errors = [
            abs(estimate_tone(_tone(o), method="parabolic") - 100 - o)
            for o in np.linspace(-0.5, 0.5, 101)
        ]
        assert max(errors) > 0.1

    def test_parabolic_is_bias_limited_while_aboutanios_mulgrew_reaches_the_bound(self) -> None:
        """A13: at 28.89 dB integrated SNR, parabolic RMSE exceeds 3 CRB; A&M stays <= 1.15.

        400 trials, offsets uniform over a bin: the RMSE ratio is known to about 3.5 %.
        """
        rng = np.random.default_rng(20261106)
        n = 1024
        noise_sigma = np.sqrt(n / NOMINAL_SNR / 2)
        bound_bins = np.sqrt(6 / (NOMINAL_SNR / n * n * (n**2 - 1))) * n / (2 * np.pi)
        errors: dict[str, list[float]] = {"parabolic": [], "aboutanios_mulgrew": []}
        # One noisy tone a trial, every estimator on the same draw.
        for _ in range(400):
            offset = rng.uniform(-0.5, 0.5)
            samples = _tone(offset) + rng.normal(0, noise_sigma, (n, 2)) @ [1, 1j]
            for method, found in errors.items():
                found.append(estimate_tone(samples, method=method) - 100 - offset)
        ratio = {m: np.sqrt(np.mean(np.square(e))) / bound_bins for m, e in errors.items()}
        assert ratio["parabolic"] > 3.0
        assert ratio["aboutanios_mulgrew"] <= 1.15


class TestBound:
    def test_constant_rate_bound_is_the_closed_form_of_range_rate_and_phase(self) -> None:
        """The three-parameter bound, phase eliminated by hand (Schur complement).

        With k = 2 pi (f0 + f) / c, sigma^2 its variance over the band ((2 pi / c)^2 B^2 / 12)
        and t from pulse 0: var(R0) = D / (2 rho sigma^2 (D - sigma^2 mean(t)^2)),
        var(rate) = 1 / (2 rho (k0^2 + sigma^2) var(t)), D = k0^2 var(t) + sigma^2 mean(t^2).
        These are the delay bound c / (2 pi beta sqrt(2 rho)) inflated 3.2e-4 by the walk's
        coupling, and Rife and Boorstyn's tone bound tightened 1e-4 by it. rtol 1e-9: a 3 x 3
        inverse.
        """
        bound = cramer_rao_bounds(
            NOMINAL_SNR,
            n_pulses=N_PULSES,
            carrier_frequency_hz=CARRIER_HZ,
            bandwidth_hz=BANDWIDTH_HZ,
            prf_hz=PRF_HZ,
        )
        time_s = np.arange(N_PULSES) / PRF_HZ
        k0 = 2 * np.pi / WAVELENGTH_M
        spread = (2 * np.pi / SPEED_OF_LIGHT_MPS) ** 2 * BANDWIDTH_HZ**2 / 12
        d = k0**2 * time_s.var() + spread * np.mean(time_s**2)
        range_m = np.sqrt(d / (2 * NOMINAL_SNR * spread * (d - spread * time_s.mean() ** 2)))
        rate_mps = 1 / np.sqrt(2 * NOMINAL_SNR * (k0**2 + spread) * time_s.var())
        np.testing.assert_allclose(bound.bistatic_range_m, range_m, rtol=1e-9)
        np.testing.assert_allclose(bound.bistatic_range_rate_mps, rate_mps, rtol=1e-9)
        tone_mps = (
            WAVELENGTH_M * PRF_HZ / (2 * np.pi) * np.sqrt(6 / (NOMINAL_SNR * (N_PULSES**2 - 1)))
        )
        np.testing.assert_allclose(bound.bistatic_range_rate_mps, tone_mps, rtol=2e-4)
        assert np.isnan(bound.bistatic_range_acceleration_mps2)

    def test_acceleration_bound_is_peleg_porats_quadratic_coefficient(self) -> None:
        """var(c2) >= 90 / (rho N^4) for large N, so std(a) = lambda/(pi T^2) sqrt(90/(rho N^4))."""
        bound = cramer_rao_bounds(
            NOMINAL_SNR,
            n_pulses=N_PULSES,
            carrier_frequency_hz=CARRIER_HZ,
            bandwidth_hz=BANDWIDTH_HZ,
            prf_hz=PRF_HZ,
            polynomial_order=2,
        )
        expected = WAVELENGTH_M * PRF_HZ**2 / np.pi * np.sqrt(90 / (NOMINAL_SNR * N_PULSES**4))
        # rtol 1e-2: the large-N form drops terms of order 1/N.
        np.testing.assert_allclose(bound.bistatic_range_acceleration_mps2, expected, rtol=1e-2)


class TestEstimate:
    @pytest.mark.parametrize("method", ["aboutanios_mulgrew", "grft"])
    def test_noiseless_chip_returns_its_truth_in_parent_coordinates(self, method: str) -> None:
        """A noiseless chip at 13.37 bins and -683 m/s (a fifth of a Doppler bin off the grid)
        returns them within a twentieth of their bound at 28.89 dB."""
        rate_mps, delay = -683.0, 13.37
        chip = DataChip(
            _model_chip(delay, rate_mps),
            "slow_time",
            0,
            100,
            (N_PULSES, 1024),
            delay_bin_centers_m(1024, SAMPLE_RATE_HZ, range_gate_start_m=4800.0)[100:133],
            doppler_bin_centers_mps(N_PULSES, 1 / PRF_HZ, WAVELENGTH_M),
            _processing(),
        )
        found = estimate_fine(chip, method=method)
        bound = cramer_rao_bounds(
            NOMINAL_SNR,
            n_pulses=N_PULSES,
            carrier_frequency_hz=CARRIER_HZ,
            bandwidth_hz=BANDWIDTH_HZ,
            prf_hz=PRF_HZ,
        )
        truth_m = chip.range_bin_centers_m[0] + delay * BIN_M
        assert abs(found.bistatic_range_m - truth_m) < 0.05 * bound.bistatic_range_m
        assert abs(found.bistatic_range_rate_mps - rate_mps) < 0.05 * bound.bistatic_range_rate_mps
        assert found.method == method

    def test_fold_factor_unfolds_the_reported_rate(self) -> None:
        """A19 in small: a chip keystoned with F = 1 reports the folded rate less lambda PRF."""
        folded_mps = -670.45
        chip = crop_slow_time(
            _model_chip(16.2, folded_mps),
            16,
            range_bin_centers_m=delay_bin_centers_m(33, SAMPLE_RATE_HZ),
            rate_bin_centers_mps=doppler_bin_centers_mps(N_PULSES, 1 / PRF_HZ, WAVELENGTH_M),
            processing=_processing(fold_factor=1),
        )
        found = estimate_fine(chip)
        np.testing.assert_allclose(
            found.bistatic_range_rate_mps, folded_mps - WAVELENGTH_M * PRF_HZ, atol=1e-3
        )

    def test_a_doppler_chip_is_refused(self) -> None:
        chip = crop_range_doppler(
            np.ones((16, 40), complex),
            3,
            20,
            range_bin_centers_m=np.arange(40.0),
            rate_bin_centers_mps=np.arange(16.0),
            processing=_processing(),
        )
        with pytest.raises(ValueError, match="slow-time chip"):
            estimate_fine(chip)


@pytest.fixture(scope="module")
def scenario_chips() -> dict[str, tuple[DataChip, object]]:
    """Each variant's noiseless chain at frame 0 to a chip at its peak, with the truth."""
    found = {}
    for variant in ("k1", "k2", "k3"):
        scenario = load_scenario(REPO / "config" / f"scenario_004_keystone_{variant}.toml")
        truth = truth_geometry(scenario)[0]
        radar, gate = scenario.radar, scenario.gates[0]
        clean = pulse_compress(
            synthesise(radar, truth, [0.1], N_PULSES, rng=None, gate=gate),
            radar,
            gate=gate,
        )
        compensated = compensate_migration(clean, radar, **_settings_fields(scenario))
        peak = integrated_peak(compensated.data_rfst, refine_range=False)
        processing = ChipProcessing(
            CARRIER_HZ,
            BANDWIDTH_HZ,
            SAMPLE_RATE_HZ,
            PRF_HZ,
            radar.receiver.noise_power_per_sample_w * radar.n_samples_per_chirp,
            keystone_applied=True,
            fold_factor=compensated.fold_factor,
            compensated_acceleration_mps2=compensated.acceleration_mps2,
        )
        chip = crop_slow_time(
            compensated.data_rfst,
            int(peak.range_bin),
            range_bin_centers_m=delay_bin_centers_m(
                1024, SAMPLE_RATE_HZ, range_gate_start_m=4800.0
            ),
            rate_bin_centers_mps=doppler_bin_centers_mps(N_PULSES, 1 / PRF_HZ, WAVELENGTH_M),
            processing=processing,
        )
        found[variant] = (chip, truth)
    return found


def _trials(
    chip: DataChip, snr_linear: float, n_trials: int, seed: int, **options: object
) -> np.ndarray:
    """Return ``(n_trials, 4)`` estimates (range, rate, acceleration, jerk) of noisy chips."""
    # The noise per compressed cell that makes the chip's integrated SNR snr_linear.
    nominal_w = chip.processing.noise_power_w
    noise_w = nominal_w * NOMINAL_SNR / snr_linear
    rng = np.random.default_rng(seed)
    errors = []
    # One noisy chip a trial.
    for _ in range(n_trials):
        noisy = DataChip(
            chip.data + _compressed_noise(rng, chip.data.shape[1], noise_w),
            chip.domain,
            chip.axis0_offset,
            chip.range_offset,
            chip.parent_shape,
            chip.range_bin_centers_m,
            chip.rate_bin_centers_mps,
            chip.processing,
        )
        found = estimate_fine(noisy, **options)  # type: ignore[arg-type]
        errors.append(
            [
                found.bistatic_range_m,
                found.bistatic_range_rate_mps,
                found.bistatic_range_acceleration_mps2,
                found.bistatic_range_jerk_mps3,
            ]
        )
    return np.array(errors)


def _bound(snr_linear: float, order: int) -> object:
    return cramer_rao_bounds(
        snr_linear,
        n_pulses=N_PULSES,
        carrier_frequency_hz=CARRIER_HZ,
        bandwidth_hz=BANDWIDTH_HZ,
        prf_hz=PRF_HZ,
        polynomial_order=order,
    )


# Slow: 500 estimates at each of two SNRs.
@pytest.mark.slow
@pytest.mark.parametrize("snr_db", [20.0, 28.886])
def test_aboutanios_mulgrew_range_and_rate_reach_the_bound_on_k1(scenario_chips, snr_db):
    """A8: A&M on K1's chip, RMSE / CRB <= 1.15 for range and rate, 500 trials.

    500 trials know an RMSE to about 3 % (1/sqrt(1000)); 1.15 is about 4 sigma above the 1.01
    a pure tone reaches.
    """
    chip, truth = scenario_chips["k1"]
    snr = 10 ** (snr_db / 10)
    found = _trials(chip, snr, 500, 8)
    bound = _bound(snr, 1)
    range_error = found[:, 0] - truth.bistatic_range_m[0]
    rate_error = found[:, 1] - truth.bistatic_range_rate_mps[0]
    assert np.sqrt(np.mean(range_error**2)) / bound.bistatic_range_m <= 1.15
    assert np.sqrt(np.mean(rate_error**2)) / bound.bistatic_range_rate_mps <= 1.15


# Slow: 500 trials of three phase estimators.
@pytest.mark.slow
def test_grft_and_cpf_acceleration_approach_the_bound_on_k2(scenario_chips):
    """A9: on K2's chip, acceleration RMSE / CRB: local GRFT <= 1.2, CPF <= 1.5, HAF finite.

    CPF's bound follows O'Shea: its phase-rate MSE is within 50 % of the bound at high SNR,
    1.22 in RMSE, with margin. HAF is the baseline, reported and not bounded.
    """
    chip, truth = scenario_chips["k2"]
    bound = _bound(NOMINAL_SNR, 2).bistatic_range_acceleration_mps2
    ratio = {}
    for method in ("grft", "cpf", "haf"):
        found = _trials(chip, NOMINAL_SNR, 500, 9, method=method)
        error = found[:, 2] - truth.bistatic_range_acceleration_mps2[0]
        ratio[method] = np.sqrt(np.mean(error**2)) / bound
    assert ratio["grft"] <= 1.2
    assert ratio["cpf"] <= 1.5
    assert np.isfinite(ratio["haf"])


# Slow: 100 cubic-model fits per variant.
@pytest.mark.slow
@pytest.mark.parametrize("variant", ["k1", "k2", "k3"])
def test_jerk_is_consistent_with_zero_where_the_dwell_cannot_see_it(scenario_chips, variant):
    """A14: the cubic fit's jerk lies within 3 CRB of zero in >= 99 % of 100 trials.

    At this dwell jerk is unobservable: its bound is 4574 m/s^3, while 600 m/s^3 puts 0.02 rad
    at the dwell's edges about its centre.
    """
    chip, _ = scenario_chips[variant]
    found = _trials(chip, NOMINAL_SNR, 100, 14, method="grft", polynomial_order=3)
    bound = _bound(NOMINAL_SNR, 3).bistatic_range_jerk_mps3
    assert np.mean(np.abs(found[:, 3]) <= 3 * bound) >= 0.99


# Slow: 200 cubic-model fits.
@pytest.mark.slow
def test_jerk_is_recovered_where_it_is_observable():
    """A14: on a model chip with 44 rad of cubic phase, CPF + ML recover jerk within 1.5 CRB.

    -1.5e5 m/s^3 is 33 of the cubic model's bounds at 28.89 dB; 200 trials. The chip's peak is
    unit amplitude, so N / rho of noise per cell gives that SNR.
    """
    jerk_mps3 = -1.5e5
    chip = DataChip(
        _model_chip(16.3, -10.0, jerk_mps3=jerk_mps3),
        "slow_time",
        0,
        0,
        (N_PULSES, 33),
        delay_bin_centers_m(33, SAMPLE_RATE_HZ),
        doppler_bin_centers_mps(N_PULSES, 1 / PRF_HZ, WAVELENGTH_M),
        _processing(noise_power_w=N_PULSES / NOMINAL_SNR),
    )
    found = _trials(chip, NOMINAL_SNR, 200, 15, method="grft", polynomial_order=3)
    bound = _bound(NOMINAL_SNR, 3).bistatic_range_jerk_mps3
    assert np.sqrt(np.mean((found[:, 3] - jerk_mps3) ** 2)) / bound <= 1.5


def test_an_fmcw_chip_is_ranged_where_its_one_sided_band_lies():
    """A deramped FMCW profile's range spectrum is the sweep's fast time, (-f_s, 0].

    Read as a band centred on zero, the delay's phase ramp wraps inside it and a range a quarter
    of a bin off a bin centre lands 0.39 to 0.75 of a bin out; read where it lies, it is right.
    """
    n_samples, n_pulses, sample_rate_hz = 1000, 64, 1.0e6
    centred = ChipProcessing(9.8e9, 2.0e6, sample_rate_hz, 1.0e3, 1.0)
    one_sided = dataclasses.replace(
        centred, band_centre_hz=-sample_rate_hz / 2.0, band_width_hz=sample_rate_hz
    )
    sample, pulse = np.arange(n_samples), np.arange(n_pulses)[:, None]
    errors: dict[str, list[float]] = {"centred": [], "one_sided": []}
    # Three beat tones, each a stationary deramped echo at a fractional range bin.
    for true_bin in (400.25, 400.5, 400.75):
        beat = np.exp(2j * np.pi * (true_bin * sample / n_samples + 0.13 * pulse))
        profiles = np.fft.fft(beat, axis=1)
        for name, processing in (("centred", centred), ("one_sided", one_sided)):
            chip = crop_slow_time(
                profiles,
                400,
                range_bin_centers_m=np.arange(n_samples) * 149.896229,
                rate_bin_centers_mps=np.arange(n_pulses) * 0.1,
                processing=processing,
            )
            found = estimate_fine(chip).range_index + chip.range_offset
            errors[name].append(abs(found - true_bin))
    assert min(errors["centred"]) > 0.3
    # The 33-bin crop cuts the profile's Dirichlet tails: 5e-3 of a bin at a quarter bin.
    assert max(errors["one_sided"]) < 1e-2
