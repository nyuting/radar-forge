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

from radar_forge.core.dsp import (
    doppler_bin_centers_mps,
    matched_filter,
    range_bin_centers_m,
    range_doppler_map,
)
from radar_forge.core.propagation import bistatic_line_of_sight_paths, line_of_sight_paths
from radar_forge.core.radar import BistaticRadar, Radar, Receiver, Transmitter
from radar_forge.core.signal import fmcw_deramp_baseband, pulsed_baseband, thermal_noise
from radar_forge.core.waveforms import lfm_chirp

DUKE_SITE = (36.00250, -78.94100, 60.0)
N_PULSES = 256

S1_RADAR = Radar(
    Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3, waveform="fmcw"),
    Receiver(1.0e6, 30.0, 3.0),
    *DUKE_SITE,
)
S2_RADAR = Radar(
    Transmitter(9.8e9, 2.0e6, 1.0e3, 30.0, 10.0e-6, 25.0e3, waveform="pulsed"),
    Receiver(2.5e6, 30.0, 3.0),
    *DUKE_SITE,
)


# The S1 range and Doppler axes for an N_PULSES dwell, from the axis helpers.
_S1_RANGE_AXIS_M = range_bin_centers_m(
    S1_RADAR.n_samples_per_pri, 2.0e6, 1.0e-3, S1_RADAR.receiver.sample_rate_hz
)
_S1_VELOCITY_AXIS_MPS = doppler_bin_centers_mps(
    N_PULSES, S1_RADAR.transmitter.pulse_repetition_interval_s, S1_RADAR.wavelength_m
)


def _fmcw_peak_bins(range_m: float, velocity_mps: float) -> tuple[int, int]:
    """Return the (doppler_bin, range_bin) of the brightest S1 range-Doppler cell."""
    paths = line_of_sight_paths(S1_RADAR, range_m, velocity_mps, 10.0)
    rd_map = range_doppler_map(fmcw_deramp_baseband(paths, S1_RADAR, N_PULSES))
    doppler_bin, range_bin = np.unravel_index(np.abs(rd_map).argmax(), rd_map.shape)
    return int(doppler_bin), int(range_bin)


class TestFmcwDerampBaseband:
    def test_a_closing_target_on_bin_centres_lands_in_exactly_those_bins(self) -> None:
        """Catches a flipped Doppler sign, a one-way delay, or an off-by-one bin, end to end.

        The target is placed on range bin 133 (9968 m) and Doppler bin
        N/2 + 50 (+2.99 m/s, closing), both read off the axis helpers, so the
        answer is an integer pair with no tolerance. This is the test the module
        docstring's sweep-direction argument exists to pass: with an up sweep
        the closing target would land on bin N/2 - 50 and look entirely
        plausible. Range migration over the dwell is 0.8 m, 1% of a bin.
        """
        range_bin, doppler_bin = 133, N_PULSES // 2 + 50
        range_m = float(_S1_RANGE_AXIS_M[range_bin])
        velocity_mps = float(_S1_VELOCITY_AXIS_MPS[doppler_bin])
        assert velocity_mps > 0.0  # closing
        assert _fmcw_peak_bins(range_m, velocity_mps) == (doppler_bin, range_bin)

    def test_a_fast_target_folds_in_doppler_but_not_in_range(self) -> None:
        """The defining behaviour of scenario 001 S1.

        A 79.5 m/s aircraft is five full Doppler spans (5 x 15.30 m/s) above the
        bin-centre velocity of +2.99 m/s, so it lands exactly where that slow
        target would -- a wrong, folded velocity -- while its range bin stays
        correct. Range migration over the dwell is 20 m, a quarter of a bin. At
        S1's parameters stop-and-hop holds even at this speed, and pyproject
        turns warnings into errors, so this also proves the generator stays
        silent where it should.
        """
        range_bin, doppler_bin = 133, N_PULSES // 2 + 50
        span_mps = 2.0 * S1_RADAR.unambiguous_velocity_mps
        true_velocity_mps = float(_S1_VELOCITY_AXIS_MPS[doppler_bin]) + 5.0 * span_mps
        peak = _fmcw_peak_bins(float(_S1_RANGE_AXIS_M[range_bin]), true_velocity_mps)
        assert peak == (doppler_bin, range_bin)

    def test_warns_when_stop_and_hop_is_strained(self) -> None:
        """A half-second chirp is long enough for a target to cross a bin within it.

        Scenario 001 never comes close — 80 m/s over 1 ms is 0.08 m against a
        74.95 m bin — so the guard has to be provoked deliberately.
        """
        slow_sweep_radar = Radar(
            Transmitter(9.8e9, 2.0e6, 100.0, 30.0, chirp_duration_s=0.5, prf_hz=2.0),
            Receiver(1.0e6, 30.0, 3.0),
            *DUKE_SITE,
        )
        paths = line_of_sight_paths(slow_sweep_radar, 10_000.0, 100.0, 10.0)
        with pytest.warns(UserWarning, match="range bin during one chirp"):
            fmcw_deramp_baseband(paths, slow_sweep_radar, 4)

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

    def test_a_fast_closing_target_lands_at_positive_velocity_unfolded(self) -> None:
        """S2's compensating virtue: 80 m/s is well inside +/-191 m/s.

        Catches a flipped Doppler sign, or a residual phase taken from the
        folded delay. The target is placed on Doppler bin N/2 + 54, +80.66 m/s,
        so the assertion is an exact bin; the old +/-2 m/s band was 1.3 bins
        wide at S2's 1.49 m/s spacing.
        """
        velocity_axis_mps = doppler_bin_centers_mps(
            N_PULSES,
            S2_RADAR.transmitter.pulse_repetition_interval_s,
            S2_RADAR.wavelength_m,
        )
        doppler_bin = N_PULSES // 2 + 54
        paths = line_of_sight_paths(S2_RADAR, 4_000.0, float(velocity_axis_mps[doppler_bin]), 10.0)
        rd_map = range_doppler_map(pulsed_baseband(paths, S2_RADAR, N_PULSES))
        peak = np.unravel_index(np.abs(rd_map).argmax(), rd_map.shape)
        assert int(peak[0]) == doppler_bin

    def test_rejects_an_fmcw_radar(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, 10_000.0, 0.0, 10.0)
        with pytest.raises(ValueError, match="needs a pulsed transmitter"):
            pulsed_baseband(paths, S1_RADAR, 4)

    def test_rejects_an_empty_dwell(self) -> None:
        """A cube with no slow-time axis has no Doppler, and no error either.

        The FMCW generator already guards this; the pulsed one documents the
        same Raises and was untested, so a zero-pulse dwell would have returned
        an empty cube that every downstream transform accepts and silently
        turns into an empty range-Doppler map.
        """
        paths = line_of_sight_paths(S2_RADAR, 10_000.0, 0.0, 10.0)
        with pytest.raises(ValueError, match="at least one"):
            pulsed_baseband(paths, S2_RADAR, 0)


class TestThermalNoise:
    def test_power_matches_the_requested_value(self) -> None:
        """A Monte-Carlo check, so this is a confidence interval, not a tolerance.

        Per docs/conventions/testing.md S2: with 2e6 complex samples the relative
        standard error of the power estimate is 1/sqrt(2e6) = 0.07%, so 1% is
        about fourteen sigma.
        """
        rng = np.random.default_rng(20260911)
        noise_power_w = 4.0e-15
        noise = thermal_noise((2000, 1000), noise_power_w, rng=rng)
        measured_w = float(np.mean(np.abs(noise) ** 2))
        assert abs(measured_w - noise_power_w) / noise_power_w < 0.01

    def test_is_circularly_symmetric(self) -> None:
        """Real and imaginary parts carry half the power each and do not correlate."""
        rng = np.random.default_rng(20260911)
        noise = thermal_noise((4000, 500), 2.0, rng=rng)
        np.testing.assert_allclose(np.var(noise.real), 1.0, rtol=0.02)
        np.testing.assert_allclose(np.var(noise.imag), 1.0, rtol=0.02)
        correlation = float(np.mean(noise.real * noise.imag))
        assert abs(correlation) < 0.05

    def test_is_reproducible_from_a_seed(self) -> None:
        first = thermal_noise((8, 8), 1.0, rng=np.random.default_rng(7))
        second = thermal_noise((8, 8), 1.0, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)

    def test_zero_power_gives_an_exactly_silent_cube(self) -> None:
        noise = thermal_noise((4, 4), 0.0, rng=np.random.default_rng(1))
        np.testing.assert_array_equal(noise, 0.0)

    def test_rejects_negative_power(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            thermal_noise((2, 2), -1.0, rng=np.random.default_rng(1))

    def test_noise_is_added_when_a_generator_is_supplied(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, 10_000.0, 0.0, 10.0)
        clean = fmcw_deramp_baseband(paths, S1_RADAR, 8)
        noisy = fmcw_deramp_baseband(paths, S1_RADAR, 8, rng=np.random.default_rng(3))
        assert not np.array_equal(clean, noisy)
        residual_power_w = float(np.mean(np.abs(noisy - clean) ** 2))
        np.testing.assert_allclose(residual_power_w, S1_RADAR.noise_power_w, rtol=0.05)


# A synthetic receiver site 0.2 deg due east of DUKE_SITE; baseline about
# 18 km. The same 0.2 deg spanned 22 km before the translation onto Raleigh-Durham
# (scripts/translate_flight_coordinates.py) -- an east-west degree shrinks as cos(latitude).
RDU_SITE = (36.00250, -78.74100, 60.0)

S1_PAIR = BistaticRadar(
    Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3, waveform="fmcw"),
    Receiver(1.0e6, 30.0, 3.0),
    *DUKE_SITE,
    *RDU_SITE,
)
S2_PAIR = BistaticRadar(
    Transmitter(9.8e9, 2.0e6, 1.0e3, 30.0, 10.0e-6, 25.0e3, waveform="pulsed"),
    Receiver(2.5e6, 30.0, 3.0),
    *DUKE_SITE,
    *RDU_SITE,
)


class TestBistaticGenerators:
    """The generators are reused unmodified; these say so in cube terms."""

    def test_fmcw_puts_a_bistatic_target_in_the_half_sum_range_bin(self) -> None:
        """The proof that no generator body needed changing.

        The range sum is chosen so its half lands on a bin centre exactly, and
        the assertion is on the integer bin index, so there is no tolerance to
        get wrong.
        """
        range_axis_m = range_bin_centers_m(
            S1_PAIR.n_samples_per_pri, 2.0e6, 1.0e-3, S1_PAIR.receiver.sample_rate_hz
        )
        target_bin = 40
        half_sum_m = float(range_axis_m[target_bin])
        # Split the sum unevenly, so a generator reading either range alone,
        # or the full sum rather than its half, lands in a different bin.
        range_tx_m = 0.25 * (2.0 * half_sum_m)
        range_rx_m = 0.75 * (2.0 * half_sum_m)

        paths = bistatic_line_of_sight_paths(S1_PAIR, range_tx_m, range_rx_m, 0.0, 10.0)
        rd_map = range_doppler_map(fmcw_deramp_baseband(paths, S1_PAIR, N_PULSES))
        peak = np.unravel_index(np.abs(rd_map).argmax(), rd_map.shape)
        assert int(peak[1]) == target_bin

    def test_pulsed_bistatic_target_folds_on_the_range_sum(self) -> None:
        """Folding happens on (R_t + R_r), so it is the sum that wraps, not a range."""
        unambiguous_sum_m = S2_PAIR.unambiguous_range_m
        range_sum_m = unambiguous_sum_m * 1.5  # beyond one repetition interval
        folded_sum_m = range_sum_m % unambiguous_sum_m

        reference = lfm_chirp(2.0e6, 10.0e-6, 2.5e6)
        folded_peak = np.abs(
            matched_filter(
                pulsed_baseband(
                    bistatic_line_of_sight_paths(
                        S2_PAIR, range_sum_m / 3.0, 2.0 * range_sum_m / 3.0, 0.0, 10.0
                    ),
                    S2_PAIR,
                    16,
                ),
                reference,
                axis=-1,
            )[0]
        ).argmax()
        # The same range sum, split differently and inside one interval: it must
        # compress to the same sample, because only the sum sets the delay.
        unfolded_peak = np.abs(
            matched_filter(
                pulsed_baseband(
                    bistatic_line_of_sight_paths(
                        S2_PAIR, folded_sum_m / 4.0, 3.0 * folded_sum_m / 4.0, 0.0, 10.0
                    ),
                    S2_PAIR,
                    16,
                ),
                reference,
                axis=-1,
            )[0]
        ).argmax()
        assert int(folded_peak) == int(unfolded_peak)

    def test_doppler_bin_follows_the_bisector_rate(self) -> None:
        """Closing-positive survives the bistatic path: the peak is at +v, not -v."""
        velocity_axis_mps = doppler_bin_centers_mps(
            N_PULSES,
            S1_PAIR.transmitter.pulse_repetition_interval_s,
            S1_PAIR.wavelength_m,
        )
        target_bin = N_PULSES // 2 + 3
        bisector_velocity_mps = float(velocity_axis_mps[target_bin])

        paths = bistatic_line_of_sight_paths(S1_PAIR, 12.0e3, 15.0e3, bisector_velocity_mps, 10.0)
        cube = fmcw_deramp_baseband(paths, S1_PAIR, N_PULSES)
        rd_map = range_doppler_map(cube)
        peak = np.unravel_index(np.abs(rd_map).argmax(), rd_map.shape)
        assert int(peak[0]) == target_bin
