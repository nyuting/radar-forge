"""Tests for the radar system value objects.

The properties are checked against the numbers derived independently in
spec/scenario-001-singapore-xband.md §4. That table and this module are the two
halves of the same claim: if they ever disagree, one of them is wrong and the
test is the one that decides.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.core.radar import Radar, Receiver, Transmitter

# Scenario 001 S1: FMCW, low PRF, Doppler folds.
S1_TRANSMITTER = Transmitter(
    f0_hz=9.8e9,
    bandwidth_hz=2.0e6,
    transmit_power_w=100.0,
    gain_tx_dbi=30.0,
    chirp_time_s=1.0e-3,
    prf_hz=1.0e3,
    waveform="fmcw",
)
S1_RECEIVER = Receiver(sample_rate_hz=1.0e6, gain_rx_dbi=30.0, noise_figure_db=3.0)

# Scenario 001 S2: pulsed, medium PRF, range folds.
S2_TRANSMITTER = Transmitter(
    f0_hz=9.8e9,
    bandwidth_hz=2.0e6,
    transmit_power_w=1.0e3,
    gain_tx_dbi=30.0,
    chirp_time_s=10.0e-6,
    prf_hz=25.0e3,
    waveform="pulsed",
)
S2_RECEIVER = Receiver(sample_rate_hz=2.5e6, gain_rx_dbi=30.0, noise_figure_db=3.0)

DSO_SITE = (1.29150, 103.78710, 60.0)


def _radar(transmitter: Transmitter, receiver: Receiver) -> Radar:
    return Radar(transmitter, receiver, *DSO_SITE)


class TestTransmitter:
    def test_wavelength_at_x_band(self) -> None:
        """9.8 GHz gives 30.591 mm, the figure quoted throughout scenario 001."""
        np.testing.assert_allclose(S1_TRANSMITTER.wavelength_m, 0.030_591_067, rtol=1e-7)

    def test_wavelength_is_c_over_f(self) -> None:
        expected_m = SPEED_OF_LIGHT_MPS / S1_TRANSMITTER.f0_hz
        np.testing.assert_allclose(S1_TRANSMITTER.wavelength_m, expected_m, rtol=1e-15)

    def test_fmcw_sweep_rate_is_two_ghz_per_second(self) -> None:
        """2 MHz over 1 ms is 2.000 GHz/s, per spec §4 S1."""
        np.testing.assert_allclose(S1_TRANSMITTER.sweep_rate_hzps, 2.0e9, rtol=1e-12)

    def test_fmcw_runs_at_full_duty(self) -> None:
        np.testing.assert_allclose(S1_TRANSMITTER.duty_cycle_linear, 1.0, rtol=1e-12)

    def test_pulsed_duty_cycle(self) -> None:
        """10 us at 25 kHz is a 25% duty cycle."""
        np.testing.assert_allclose(S2_TRANSMITTER.duty_cycle_linear, 0.25, rtol=1e-12)

    def test_pulse_repetition_interval_is_the_prf_reciprocal(self) -> None:
        np.testing.assert_allclose(S2_TRANSMITTER.pulse_repetition_interval_s, 40.0e-6, rtol=1e-12)

    @pytest.mark.parametrize(
        "field",
        ["f0_hz", "bandwidth_hz", "transmit_power_w", "chirp_time_s", "prf_hz"],
    )
    def test_rejects_non_positive_quantities(self, field: str) -> None:
        kwargs = {
            "f0_hz": 9.8e9,
            "bandwidth_hz": 2.0e6,
            "transmit_power_w": 100.0,
            "gain_tx_dbi": 30.0,
            "chirp_time_s": 1.0e-3,
            "prf_hz": 1.0e3,
        }
        kwargs[field] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            Transmitter(**kwargs)  # type: ignore[arg-type]  # deliberately bad value

    def test_rejects_a_duty_cycle_above_one(self) -> None:
        """A chirp that has not finished when the next starts is not a waveform."""
        with pytest.raises(ValueError, match=r"exceeds 1\.0"):
            Transmitter(9.8e9, 2.0e6, 100.0, 30.0, chirp_time_s=2.0e-3, prf_hz=1.0e3)

    def test_rejects_an_unknown_waveform(self) -> None:
        with pytest.raises(ValueError, match="'fmcw' or 'pulsed'"):
            Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3, waveform="cw")  # type: ignore[arg-type]  # deliberately bad value


class TestReceiver:
    def test_noise_figure_of_three_db_is_about_a_factor_of_two(self) -> None:
        np.testing.assert_allclose(S1_RECEIVER.noise_figure_linear, 1.995_262, rtol=1e-6)

    def test_zero_db_noise_figure_is_unity(self) -> None:
        np.testing.assert_allclose(Receiver(1.0e6, 30.0, 0.0).noise_figure_linear, 1.0, rtol=1e-15)

    def test_rejects_a_negative_noise_figure(self) -> None:
        with pytest.raises(ValueError, match="cannot improve the signal-to-noise ratio"):
            Receiver(1.0e6, 30.0, -1.0)

    def test_rejects_a_non_positive_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            Receiver(0.0, 30.0, 3.0)


class TestRadarAmbiguity:
    def test_range_resolution_is_seventy_five_metres(self) -> None:
        """c/2B at 2 MHz is 74.95 m, per spec §2."""
        np.testing.assert_allclose(
            _radar(S1_TRANSMITTER, S1_RECEIVER).range_resolution_m, 74.948, rtol=1e-4
        )

    def test_range_resolution_is_waveform_independent(self) -> None:
        """Both variants share a bandwidth, so both share a resolution."""
        np.testing.assert_allclose(
            _radar(S1_TRANSMITTER, S1_RECEIVER).range_resolution_m,
            _radar(S2_TRANSMITTER, S2_RECEIVER).range_resolution_m,
            rtol=1e-15,
        )

    def test_s1_unambiguous_range_covers_the_track(self) -> None:
        """FMCW deramp: c*fs/(4*alpha) = 37.47 km, against a track reaching 17.9 km."""
        unambiguous_range_m = _radar(S1_TRANSMITTER, S1_RECEIVER).unambiguous_range_m
        np.testing.assert_allclose(unambiguous_range_m, 37_474.06, rtol=1e-6)
        assert unambiguous_range_m > 18_000.0

    def test_s1_unambiguous_velocity_folds_an_aircraft(self) -> None:
        """+/-7.65 m/s, so an 80 m/s target folds about five times over."""
        unambiguous_velocity_mps = _radar(S1_TRANSMITTER, S1_RECEIVER).unambiguous_velocity_mps
        np.testing.assert_allclose(unambiguous_velocity_mps, 7.6478, rtol=1e-4)
        assert unambiguous_velocity_mps < 80.0

    def test_s2_unambiguous_range_folds_the_track(self) -> None:
        """Pulsed: c/2*PRF = 5.996 km, against a track starting at 8.4 km."""
        unambiguous_range_m = _radar(S2_TRANSMITTER, S2_RECEIVER).unambiguous_range_m
        np.testing.assert_allclose(unambiguous_range_m, 5_995.849, rtol=1e-6)
        assert unambiguous_range_m < 8_390.0

    def test_s2_unambiguous_velocity_does_not_fold(self) -> None:
        """+/-191.2 m/s covers any light aircraft."""
        unambiguous_velocity_mps = _radar(S2_TRANSMITTER, S2_RECEIVER).unambiguous_velocity_mps
        np.testing.assert_allclose(unambiguous_velocity_mps, 191.194, rtol=1e-5)
        assert unambiguous_velocity_mps > 100.0

    def test_the_two_variants_trade_one_ambiguity_for_the_other(self) -> None:
        """The central claim of spec §4, asserted rather than asserted in prose.

        S1 is unambiguous in range and folded in Doppler; S2 is the exact
        reverse. Neither is unambiguous in both, which is the design tension the
        three-variant scenario exists to show.
        """
        s1 = _radar(S1_TRANSMITTER, S1_RECEIVER)
        s2 = _radar(S2_TRANSMITTER, S2_RECEIVER)
        track_maximum_range_m = 17_870.0
        aircraft_speed_mps = 80.0

        assert s1.unambiguous_range_m > track_maximum_range_m
        assert s1.unambiguous_velocity_mps < aircraft_speed_mps
        assert s2.unambiguous_range_m < track_maximum_range_m
        assert s2.unambiguous_velocity_mps > aircraft_speed_mps

    def test_range_doppler_product_is_bounded_by_c_over_four(self) -> None:
        r"""R_ua * v_ua = c*lambda/8 for a pulsed radar, independent of PRF.

        This is why the trade cannot be escaped by choosing a better PRF: the
        product is fixed by the carrier alone.
        """
        radar = _radar(S2_TRANSMITTER, S2_RECEIVER)
        product = radar.unambiguous_range_m * radar.unambiguous_velocity_mps
        expected = SPEED_OF_LIGHT_MPS * radar.wavelength_m / 8.0
        np.testing.assert_allclose(product, expected, rtol=1e-12)


class TestRadarValidation:
    def test_rejects_an_impossible_site_latitude(self) -> None:
        with pytest.raises(ValueError, match=r"\[-90, 90\]"):
            Radar(S1_TRANSMITTER, S1_RECEIVER, 91.0, 103.0, 60.0)

    def test_rejects_a_pulsed_receiver_that_would_alias(self) -> None:
        """A pulsed receiver must cover the signal bandwidth; FMCW need not."""
        with pytest.raises(ValueError, match="would alias"):
            Radar(S2_TRANSMITTER, Receiver(1.0e6, 30.0, 3.0), *DSO_SITE)

    def test_permits_an_fmcw_receiver_below_the_swept_bandwidth(self) -> None:
        """1 MHz sampling of a 2 MHz sweep is correct: deramping comes first.

        This is the property that makes FMCW cheap, and the reason the check
        above is waveform-specific rather than universal.
        """
        radar = _radar(S1_TRANSMITTER, S1_RECEIVER)
        assert radar.receiver.sample_rate_hz < radar.transmitter.bandwidth_hz

    def test_noise_power_matches_ktbf(self) -> None:
        """-174 dBm/Hz + 10log10(B) + F, the standard link-budget line."""
        radar = _radar(S1_TRANSMITTER, S1_RECEIVER)
        noise_dbm = 10.0 * np.log10(radar.noise_power_w * 1e3)
        expected_dbm = -173.98 + 10.0 * np.log10(2.0e6) + 3.0
        assert abs(noise_dbm - expected_dbm) < 0.01

    def test_samples_per_chirp(self) -> None:
        """1 ms at 1 MHz is 1000 samples; a 10 us pulse at 2.5 MHz is 25."""
        assert _radar(S1_TRANSMITTER, S1_RECEIVER).n_samples_per_chirp == 1000
        assert _radar(S2_TRANSMITTER, S2_RECEIVER).n_samples_per_chirp == 25

    def test_samples_per_pri_is_the_receive_window(self) -> None:
        """The IQ row length: a full 40 us PRI at 2.5 MHz is 100 samples.

        For the pulsed variant this is four times n_samples_per_chirp, because
        the receiver listens for the whole interval and not just while the
        transmitter is on. Conflating the two would size the cube at a quarter
        of the unambiguous range.
        """
        assert _radar(S2_TRANSMITTER, S2_RECEIVER).n_samples_per_pri == 100
        assert _radar(S1_TRANSMITTER, S1_RECEIVER).n_samples_per_pri == 1000

    def test_pri_and_chirp_windows_agree_only_at_full_duty(self) -> None:
        s1 = _radar(S1_TRANSMITTER, S1_RECEIVER)
        s2 = _radar(S2_TRANSMITTER, S2_RECEIVER)
        assert s1.n_samples_per_pri == s1.n_samples_per_chirp
        assert s2.n_samples_per_pri > s2.n_samples_per_chirp

    def test_is_frozen(self) -> None:
        """Value objects, so a scenario cannot be mutated halfway through a run."""
        with pytest.raises(AttributeError):
            _radar(S1_TRANSMITTER, S1_RECEIVER).latitude_deg = 0.0  # type: ignore[misc]  # frozen
