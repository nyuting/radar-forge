"""Tests for baseband signal synthesis.

Ground truth is analytic throughout: a target placed at a chosen range and
velocity must appear in the range-Doppler bin that the waveform parameters
predict, with no recorded output anywhere. That is the property the whole
scenario rests on, so it is tested for both waveform families and for the
folding behaviour each one exhibits.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.ambiguity import fold_velocity_mps
from radar_forge.core.dsp import (
    doppler_bin_centers_mps,
    matched_filter,
    range_bin_centers_m,
    range_doppler_map,
)
from radar_forge.core.radar import Radar, Receiver, Transmitter
from radar_forge.core.signal import (
    PropagationPaths,
    fmcw_deramp_baseband,
    line_of_sight_paths,
    pulsed_baseband,
    thermal_noise,
)
from radar_forge.core.waveforms import lfm_chirp

DSO_SITE = (1.29150, 103.78710, 60.0)
N_PULSES = 256

S1_RADAR = Radar(
    Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3, waveform="fmcw"),
    Receiver(1.0e6, 30.0, 3.0),
    *DSO_SITE,
)
S2_RADAR = Radar(
    Transmitter(9.8e9, 2.0e6, 1.0e3, 30.0, 10.0e-6, 25.0e3, waveform="pulsed"),
    Receiver(2.5e6, 30.0, 3.0),
    *DSO_SITE,
)


def _fmcw_peak(range_m: float, velocity_mps: float) -> tuple[float, float]:
    """Return the (range_m, velocity_mps) of the brightest S1 range-Doppler cell."""
    paths = line_of_sight_paths(S1_RADAR, range_m, velocity_mps, 10.0)
    cube = fmcw_deramp_baseband(paths, S1_RADAR, N_PULSES)
    rd_map = range_doppler_map(cube)
    peak = np.unravel_index(np.abs(rd_map).argmax(), rd_map.shape)
    range_axis_m = range_bin_centers_m(
        rd_map.shape[1], 2.0e6, 1.0e-3, S1_RADAR.receiver.sample_rate_hz
    )
    velocity_axis_mps = doppler_bin_centers_mps(
        rd_map.shape[0],
        S1_RADAR.transmitter.pulse_repetition_interval_s,
        S1_RADAR.wavelength_m,
    )
    return float(range_axis_m[peak[1]]), float(velocity_axis_mps[peak[0]])


class TestLineOfSightPaths:
    def test_delay_is_the_two_way_transit_time(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, 15_000.0, 0.0, 10.0)
        np.testing.assert_allclose(paths.delay_s[0], 2.0 * 15_000.0 / 299_792_458.0, rtol=1e-15)

    def test_range_property_inverts_the_delay(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, 12_345.0, 0.0, 10.0)
        np.testing.assert_allclose(paths.range_m[0], 12_345.0, rtol=1e-12)

    def test_closing_velocity_gives_positive_doppler(self) -> None:
        """The library sign convention, per spec/structure.md D5."""
        closing = line_of_sight_paths(S1_RADAR, 10_000.0, 80.0, 10.0)
        opening = line_of_sight_paths(S1_RADAR, 10_000.0, -80.0, 10.0)
        assert closing.doppler_hz[0] > 0.0
        assert opening.doppler_hz[0] < 0.0
        np.testing.assert_allclose(
            closing.doppler_hz[0], 2.0 * 80.0 / S1_RADAR.wavelength_m, rtol=1e-12
        )

    def test_amplitude_squared_is_the_received_power(self) -> None:
        """Keeps the cube on the same scale as Radar.noise_power_w."""
        from radar_forge.core.radar_equation import received_power_w

        paths = line_of_sight_paths(S1_RADAR, 10_000.0, 0.0, 10.0)
        expected_w = received_power_w(
            transmit_power_w=100.0,
            gain_tx_linear=10.0**3.0,
            gain_rx_linear=10.0**3.0,
            wavelength_m=S1_RADAR.wavelength_m,
            rcs_m2=10.0,
            range_m=10_000.0,
        )
        np.testing.assert_allclose(np.abs(paths.amplitude_linear[0]) ** 2, expected_w, rtol=1e-12)

    def test_power_follows_the_inverse_fourth_power_law(self) -> None:
        near = line_of_sight_paths(S1_RADAR, 10_000.0, 0.0, 10.0)
        far = line_of_sight_paths(S1_RADAR, 20_000.0, 0.0, 10.0)
        ratio = np.abs(near.amplitude_linear[0]) ** 2 / np.abs(far.amplitude_linear[0]) ** 2
        np.testing.assert_allclose(ratio, 16.0, rtol=1e-12)

    def test_a_zero_cross_section_target_returns_nothing(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, 10_000.0, 80.0, 0.0)
        assert paths.amplitude_linear[0] == 0.0

    def test_handles_several_targets_at_once(self) -> None:
        paths = line_of_sight_paths(
            S1_RADAR, [9_000.0, 12_000.0, 15_000.0], [50.0, -20.0, 0.0], 10.0
        )
        assert paths.n_paths == 3
        assert paths.aoa_rad.shape == (3, 2)
        np.testing.assert_array_equal(paths.bounce_count, [1, 1, 1])

    @pytest.mark.parametrize("bad_range_m", [0.0, -1.0])
    def test_rejects_a_non_positive_range(self, bad_range_m: float) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            line_of_sight_paths(S1_RADAR, bad_range_m, 0.0, 10.0)

    def test_rejects_a_negative_cross_section(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            line_of_sight_paths(S1_RADAR, 10_000.0, 0.0, -1.0)


class TestPropagationPathsValidation:
    def test_rejects_mismatched_path_counts(self) -> None:
        with pytest.raises(ValueError, match="to match delay_s"):
            PropagationPaths(
                delay_s=np.zeros(3),
                doppler_hz=np.zeros(2),
                amplitude_linear=np.zeros(3, dtype=np.complex128),
                aoa_rad=np.zeros((3, 2)),
                aod_rad=np.zeros((3, 2)),
                bounce_count=np.ones(3, dtype=np.int_),
            )

    def test_rejects_angles_that_are_not_azimuth_elevation_pairs(self) -> None:
        with pytest.raises(ValueError, match=r"\(azimuth, elevation\)"):
            PropagationPaths(
                delay_s=np.zeros(3),
                doppler_hz=np.zeros(3),
                amplitude_linear=np.zeros(3, dtype=np.complex128),
                aoa_rad=np.zeros((3, 3)),
                aod_rad=np.zeros((3, 2)),
                bounce_count=np.ones(3, dtype=np.int_),
            )


class TestFmcwDerampBaseband:
    def test_cube_has_the_canonical_layout(self) -> None:
        """(n_pulses, n_samples): slow time axis 0, per style.md S4."""
        paths = line_of_sight_paths(S1_RADAR, 10_000.0, 0.0, 10.0)
        cube = fmcw_deramp_baseband(paths, S1_RADAR, N_PULSES)
        assert cube.shape == (N_PULSES, 1000)
        assert cube.dtype == np.complex128

    @pytest.mark.parametrize("true_range_m", [5_000.0, 10_000.0, 17_500.0])
    def test_a_stationary_target_lands_in_its_exact_range_bin(self, true_range_m: float) -> None:
        """Analytic ground truth: the beat frequency fixes the bin exactly."""
        peak_range_m, peak_velocity_mps = _fmcw_peak(true_range_m, 0.0)
        assert abs(peak_range_m - true_range_m) < S1_RADAR.range_resolution_m
        np.testing.assert_allclose(peak_velocity_mps, 0.0, atol=1e-12)

    def test_a_slow_closing_target_lands_at_positive_velocity(self) -> None:
        """The sign convention, end to end through synthesis and processing.

        This is the test the module docstring's sweep-direction argument exists
        to pass: with an up sweep the peak would appear at -3 m/s and look
        entirely plausible.
        """
        _, peak_velocity_mps = _fmcw_peak(10_000.0, 3.0)
        assert peak_velocity_mps > 0.0
        assert abs(peak_velocity_mps - 3.0) < 0.1

    def test_an_opening_target_lands_at_negative_velocity(self) -> None:
        _, peak_velocity_mps = _fmcw_peak(10_000.0, -3.0)
        assert peak_velocity_mps < 0.0
        assert abs(peak_velocity_mps + 3.0) < 0.1

    def test_a_fast_target_folds_in_doppler_but_not_in_range(self) -> None:
        """The defining behaviour of scenario 001 S1.

        An 80 m/s aircraft is ten times the unambiguous velocity, so it appears
        at a wrong, folded velocity — while its range stays correct.
        """
        true_velocity_mps = 80.0
        peak_range_m, peak_velocity_mps = _fmcw_peak(10_000.0, true_velocity_mps)

        expected_velocity_mps = fold_velocity_mps(
            true_velocity_mps, S1_RADAR.unambiguous_velocity_mps
        )
        assert abs(peak_range_m - 10_000.0) < S1_RADAR.range_resolution_m
        assert abs(peak_velocity_mps - expected_velocity_mps) < 0.1
        assert abs(peak_velocity_mps) < S1_RADAR.unambiguous_velocity_mps

    def test_warns_when_stop_and_hop_is_strained(self) -> None:
        """A half-second chirp is long enough for a target to cross a bin within it.

        Scenario 001 never comes close — 80 m/s over 1 ms is 0.08 m against a
        74.95 m bin — so the guard has to be provoked deliberately.
        """
        slow_sweep_radar = Radar(
            Transmitter(9.8e9, 2.0e6, 100.0, 30.0, chirp_time_s=0.5, prf_hz=2.0),
            Receiver(1.0e6, 30.0, 3.0),
            *DSO_SITE,
        )
        paths = line_of_sight_paths(slow_sweep_radar, 10_000.0, 100.0, 10.0)
        with pytest.warns(UserWarning, match="range bin during one chirp"):
            fmcw_deramp_baseband(paths, slow_sweep_radar, 4)

    def test_does_not_warn_at_the_scenario_parameters(self) -> None:
        """pyproject turns warnings into errors, so an unexpected warn fails here.

        This is the positive half of the approximation claim in the module
        docstring: at S1's parameters stop-and-hop is comfortably valid, even for
        a target ten times over the unambiguous velocity.
        """
        paths = line_of_sight_paths(S1_RADAR, 10_000.0, 80.0, 10.0)
        fmcw_deramp_baseband(paths, S1_RADAR, 8)

    def test_is_noiseless_without_a_generator(self) -> None:
        """Determinism by default, so a geometry bug cannot hide in the noise."""
        paths = line_of_sight_paths(S1_RADAR, 10_000.0, 0.0, 10.0)
        first = fmcw_deramp_baseband(paths, S1_RADAR, 4)
        second = fmcw_deramp_baseband(paths, S1_RADAR, 4)
        np.testing.assert_array_equal(first, second)

    def test_rejects_a_pulsed_radar(self) -> None:
        paths = line_of_sight_paths(S2_RADAR, 10_000.0, 0.0, 10.0)
        with pytest.raises(ValueError, match="needs an FMCW transmitter"):
            fmcw_deramp_baseband(paths, S2_RADAR, 4)

    def test_rejects_an_empty_dwell(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, 10_000.0, 0.0, 10.0)
        with pytest.raises(ValueError, match="at least one"):
            fmcw_deramp_baseband(paths, S1_RADAR, 0)


class TestPulsedBaseband:
    def test_cube_spans_the_full_repetition_interval(self) -> None:
        """100 samples per 40 us PRI, not the 25 the pulse itself occupies."""
        paths = line_of_sight_paths(S2_RADAR, 4_000.0, 0.0, 10.0)
        cube = pulsed_baseband(paths, S2_RADAR, N_PULSES)
        assert cube.shape == (N_PULSES, 100)

    def test_an_unambiguous_target_lands_at_its_true_delay(self) -> None:
        """A target inside 5.996 km does not fold, so the delay is exact."""
        true_range_m = 3_000.0
        paths = line_of_sight_paths(S2_RADAR, true_range_m, 0.0, 10.0)
        cube = pulsed_baseband(paths, S2_RADAR, 16)
        reference = lfm_chirp(2.0e6, 10.0e-6, 2.5e6)
        compressed = matched_filter(cube, reference, axis=-1)
        peak_sample = int(np.abs(compressed[0]).argmax())
        # Matched filtering adds the reference's group delay; compare the shift
        # between two ranges instead of the absolute index.
        near = line_of_sight_paths(S2_RADAR, true_range_m - 750.0, 0.0, 10.0)
        near_peak = int(
            np.abs(
                matched_filter(pulsed_baseband(near, S2_RADAR, 16), reference, axis=-1)[0]
            ).argmax()
        )
        samples_per_metre = 2.0 * 2.5e6 / 299_792_458.0
        np.testing.assert_allclose(peak_sample - near_peak, 750.0 * samples_per_metre, atol=1.0)

    def test_a_distant_target_folds_in_range(self) -> None:
        """The defining behaviour of scenario 001 S2.

        A target at 10 km, beyond the 5.996 km unambiguous range, appears at its
        delay modulo the repetition interval — about 4.0 km.
        """
        true_range_m = 10_000.0
        folded_range_m = true_range_m % S2_RADAR.unambiguous_range_m
        paths = line_of_sight_paths(S2_RADAR, true_range_m, 0.0, 10.0)
        cube = pulsed_baseband(paths, S2_RADAR, 16)
        reference = lfm_chirp(2.0e6, 10.0e-6, 2.5e6)
        compressed = matched_filter(cube, reference, axis=-1)

        unfolded = line_of_sight_paths(S2_RADAR, folded_range_m, 0.0, 10.0)
        unfolded_compressed = matched_filter(
            pulsed_baseband(unfolded, S2_RADAR, 16), reference, axis=-1
        )
        assert int(np.abs(compressed[0]).argmax()) == int(np.abs(unfolded_compressed[0]).argmax())
        assert folded_range_m < true_range_m

    def test_a_fast_closing_target_lands_at_positive_velocity_unfolded(self) -> None:
        """S2's compensating virtue: 80 m/s is well inside +/-191 m/s."""
        true_velocity_mps = 80.0
        paths = line_of_sight_paths(S2_RADAR, 4_000.0, true_velocity_mps, 10.0)
        cube = pulsed_baseband(paths, S2_RADAR, N_PULSES)
        rd_map = range_doppler_map(cube)
        peak = np.unravel_index(np.abs(rd_map).argmax(), rd_map.shape)
        velocity_axis_mps = doppler_bin_centers_mps(
            rd_map.shape[0],
            S2_RADAR.transmitter.pulse_repetition_interval_s,
            S2_RADAR.wavelength_m,
        )
        peak_velocity_mps = float(velocity_axis_mps[peak[0]])
        assert peak_velocity_mps > 0.0
        assert abs(peak_velocity_mps - true_velocity_mps) < 2.0

    def test_rejects_an_fmcw_radar(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, 10_000.0, 0.0, 10.0)
        with pytest.raises(ValueError, match="needs a pulsed transmitter"):
            pulsed_baseband(paths, S1_RADAR, 4)


class TestThermalNoise:
    def test_power_matches_the_requested_value(self) -> None:
        """A Monte-Carlo check, so this is a confidence interval, not a tolerance.

        Per docs/conventions/testing.md S2: with 2e6 complex samples the relative
        standard error of the power estimate is 1/sqrt(2e6) = 0.07%, so 1% is
        about fourteen sigma.
        """
        rng = np.random.default_rng(20260911)
        noise_power_w = 4.0e-15
        noise = thermal_noise((2000, 1000), noise_power_w, rng)
        measured_w = float(np.mean(np.abs(noise) ** 2))
        assert abs(measured_w - noise_power_w) / noise_power_w < 0.01

    def test_is_circularly_symmetric(self) -> None:
        """Real and imaginary parts carry half the power each and do not correlate."""
        rng = np.random.default_rng(20260911)
        noise = thermal_noise((4000, 500), 2.0, rng)
        np.testing.assert_allclose(np.var(noise.real), 1.0, rtol=0.02)
        np.testing.assert_allclose(np.var(noise.imag), 1.0, rtol=0.02)
        correlation = float(np.mean(noise.real * noise.imag))
        assert abs(correlation) < 0.05

    def test_is_reproducible_from_a_seed(self) -> None:
        first = thermal_noise((8, 8), 1.0, np.random.default_rng(7))
        second = thermal_noise((8, 8), 1.0, np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)

    def test_zero_power_gives_an_exactly_silent_cube(self) -> None:
        noise = thermal_noise((4, 4), 0.0, np.random.default_rng(1))
        np.testing.assert_array_equal(noise, 0.0)

    def test_rejects_negative_power(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            thermal_noise((2, 2), -1.0, np.random.default_rng(1))

    def test_noise_is_added_when_a_generator_is_supplied(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, 10_000.0, 0.0, 10.0)
        clean = fmcw_deramp_baseband(paths, S1_RADAR, 8)
        noisy = fmcw_deramp_baseband(paths, S1_RADAR, 8, rng=np.random.default_rng(3))
        assert not np.array_equal(clean, noisy)
        residual_power_w = float(np.mean(np.abs(noisy - clean) ** 2))
        np.testing.assert_allclose(residual_power_w, S1_RADAR.noise_power_w, rtol=0.05)
