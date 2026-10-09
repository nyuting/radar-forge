"""Tests for the radar system value objects.

The properties are checked against the numbers derived independently in
spec/scenario-001-xband.md §3.4. That table and this module are the two
halves of the same claim: if they ever disagree, one of them is wrong and the
test is the one that decides.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.core.geodesy import geodetic_to_enu_m
from radar_forge.core.radar import BistaticRadar, Radar, Receiver, Transmitter

# Scenario 001 S1: FMCW, low PRF, Doppler folds.
S1_TRANSMITTER = Transmitter(
    f0_hz=9.8e9,
    bandwidth_hz=2.0e6,
    transmit_power_w=100.0,
    gain_tx_dbi=30.0,
    chirp_duration_s=1.0e-3,
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
    chirp_duration_s=10.0e-6,
    prf_hz=25.0e3,
    waveform="pulsed",
)
S2_RECEIVER = Receiver(sample_rate_hz=2.5e6, gain_rx_dbi=30.0, noise_figure_db=3.0)

DUKE_SITE = (36.00250, -78.94100, 60.0)


def _radar(transmitter: Transmitter, receiver: Receiver) -> Radar:
    return Radar(transmitter, receiver, *DUKE_SITE)


class TestTransmitter:
    def test_wavelength_at_x_band(self) -> None:
        """9.8 GHz gives 30.591 mm, the figure quoted throughout scenario 001."""
        np.testing.assert_allclose(S1_TRANSMITTER.wavelength_m, 0.030_591_067, rtol=1e-7)

    def test_fmcw_sweep_rate_is_two_ghz_per_second(self) -> None:
        """2 MHz over 1 ms is 2.000 GHz/s, per spec §3.4 S1."""
        np.testing.assert_allclose(S1_TRANSMITTER.sweep_rate_hzps, 2.0e9, rtol=1e-12)

    def test_fmcw_runs_at_full_duty(self) -> None:
        """The Notes boundary: exactly 100% duty is permitted, and the formula is T * PRF."""
        np.testing.assert_allclose(S1_TRANSMITTER.duty_cycle_linear, 1.0, rtol=1e-12)

    @pytest.mark.parametrize(
        "field",
        ["f0_hz", "bandwidth_hz", "transmit_power_w", "chirp_duration_s", "prf_hz"],
    )
    def test_rejects_non_positive_quantities(self, field: str) -> None:
        kwargs = {
            "f0_hz": 9.8e9,
            "bandwidth_hz": 2.0e6,
            "transmit_power_w": 100.0,
            "gain_tx_dbi": 30.0,
            "chirp_duration_s": 1.0e-3,
            "prf_hz": 1.0e3,
        }
        kwargs[field] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            Transmitter(**kwargs)  # type: ignore[arg-type]  # deliberately bad value

    def test_rejects_a_duty_cycle_above_one(self) -> None:
        """A chirp that has not finished when the next starts is not a waveform."""
        with pytest.raises(ValueError, match=r"exceeds 1\.0"):
            Transmitter(9.8e9, 2.0e6, 100.0, 30.0, chirp_duration_s=2.0e-3, prf_hz=1.0e3)

    def test_rejects_an_unknown_waveform(self) -> None:
        with pytest.raises(ValueError, match="'fmcw' or 'pulsed'"):
            Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3, waveform="cw")  # type: ignore[arg-type]  # deliberately bad value


class TestReceiver:
    def test_noise_figure_of_three_db_is_about_a_factor_of_two(self) -> None:
        np.testing.assert_allclose(S1_RECEIVER.noise_figure_linear, 1.995_262, rtol=1e-6)

    def test_zero_db_noise_figure_is_unity(self) -> None:
        """Zero is the boundary of the guard: a ``<= 0`` check would reject an ideal receiver."""
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


class TestRadarValidation:
    def test_rejects_an_impossible_site_latitude(self) -> None:
        with pytest.raises(ValueError, match=r"\[-90, 90\]"):
            Radar(S1_TRANSMITTER, S1_RECEIVER, 91.0, 103.0, 60.0)

    def test_rejects_a_pulsed_receiver_that_would_alias(self) -> None:
        """A pulsed receiver must cover the signal bandwidth; FMCW need not."""
        with pytest.raises(ValueError, match="would alias"):
            Radar(S2_TRANSMITTER, Receiver(1.0e6, 30.0, 3.0), *DUKE_SITE)

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

    def test_is_frozen(self) -> None:
        """Value objects, so a scenario cannot be mutated halfway through a run."""
        with pytest.raises(AttributeError):
            _radar(S1_TRANSMITTER, S1_RECEIVER).latitude_deg = 0.0  # type: ignore[misc]  # frozen


# A second synthetic site 0.2 deg due east of DUKE_SITE. At 36.0025 deg N that
# is N(phi) cos(phi) * 0.2 deg = 6 385 527 * 0.80895 * 3.4907e-3 = 18.03 km, not
# the 22 km the same 0.2 deg spanned at the pre-translation latitude: an
# east-west degree is cos(phi) shorter, and cos(36.0) / cos(1.3) = 0.810.
RDU_SITE = (36.00250, -78.74100, 60.0)


def _bistatic(transmitter: Transmitter, receiver: Receiver) -> BistaticRadar:
    return BistaticRadar(transmitter, receiver, *DUKE_SITE, *RDU_SITE)


class TestBistaticGeometry:
    """The baseline and the bistatic angle, against closed-form triangles."""

    def test_baseline_matches_an_independent_geodetic_distance(self) -> None:
        pair = _bistatic(S1_TRANSMITTER, S1_RECEIVER)
        expected_m = float(
            np.linalg.norm(geodetic_to_enu_m(*RDU_SITE, *DUKE_SITE)),
        )
        # Re-derived through the same geodesy primitives but composed here, so
        # this pins the wiring rather than the WGS-84 maths: exact to float64.
        np.testing.assert_allclose(pair.baseline_m, expected_m, rtol=1e-12)

    def test_bistatic_angle_is_pi_on_the_baseline(self) -> None:
        """A target between the sites subtends a straight line.

        This is the case that requires clipping the cosine before arccos: the
        exact value is -1, rounding lands just past it, and an unclipped
        arccos returns NaN for precisely the geometry most worth asking about.
        """
        pair = _bistatic(S1_TRANSMITTER, S1_RECEIVER)
        half_baseline_m = pair.baseline_m / 2.0
        angle_rad = pair.bistatic_angle_rad(half_baseline_m, half_baseline_m)
        assert not np.isnan(angle_rad)
        # A null of a smooth function, so an absolute bound, not a relative one.
        np.testing.assert_allclose(angle_rad, np.pi, atol=1e-9)

    def test_bistatic_angle_is_a_right_angle_for_the_isoceles_case(self) -> None:
        """R_t = R_r = L / sqrt(2) is half a square: beta = pi / 2 exactly."""
        pair = _bistatic(S1_TRANSMITTER, S1_RECEIVER)
        range_m = pair.baseline_m / np.sqrt(2.0)
        # Three float64 operations on exact inputs; 1e-12 is ample.
        np.testing.assert_allclose(
            pair.bistatic_angle_rad(range_m, range_m), np.pi / 2.0, rtol=1e-12
        )

    def test_impossible_triangle_is_rejected(self) -> None:
        pair = _bistatic(S1_TRANSMITTER, S1_RECEIVER)
        too_near_m = pair.baseline_m / 4.0
        with pytest.raises(ValueError, match="not a triangle"):
            pair.bistatic_angle_rad(too_near_m, too_near_m)

    @pytest.mark.parametrize(("range_tx_m", "range_rx_m"), [(0.0, 1.0e3), (1.0e3, -1.0)])
    def test_bistatic_angle_rejects_non_positive_range(
        self, range_tx_m: float, range_rx_m: float
    ) -> None:
        pair = _bistatic(S1_TRANSMITTER, S1_RECEIVER)
        with pytest.raises(ValueError, match="strictly positive"):
            pair.bistatic_angle_rad(range_tx_m, range_rx_m)

    def test_target_ranges_are_zero_at_the_sites(self) -> None:
        """A target standing on a site is at zero range from it and at L from the other."""
        pair = _bistatic(S1_TRANSMITTER, S1_RECEIVER)
        range_tx_m, range_rx_m = pair.target_ranges_m(*[np.array([v]) for v in DUKE_SITE])
        np.testing.assert_allclose(range_tx_m[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(range_rx_m[0], pair.baseline_m, rtol=1e-9)


class TestBistaticResolution:
    """Range resolution stops being a function of bandwidth alone."""

    def test_resolution_degrades_as_secant_of_half_the_angle(self) -> None:
        """At beta = 120 deg, cos(beta / 2) = 1 / 2, so the bin is exactly twice as coarse."""
        pair = _bistatic(S1_TRANSMITTER, S1_RECEIVER)
        np.testing.assert_allclose(
            pair.range_resolution_at_bistatic_angle_m(np.radians(120.0)),
            2.0 * pair.range_resolution_m,
            rtol=1e-12,
        )

    def test_resolution_is_infinite_in_forward_scatter(self) -> None:
        """On the baseline every route has the same length, so range carries no information.

        The singularity is a real limit, returned as inf, not a numerical
        accident: pyproject turns warnings into errors, so a divide-by-zero
        RuntimeWarning on the way to inf fails this test.
        """
        pair = _bistatic(S1_TRANSMITTER, S1_RECEIVER)
        assert np.isinf(pair.range_resolution_at_bistatic_angle_m(np.pi))

    @pytest.mark.parametrize("bistatic_angle_rad", [-1.0e-9, np.pi + 1.0e-9])
    def test_resolution_rejects_angles_outside_the_half_turn(
        self, bistatic_angle_rad: float
    ) -> None:
        pair = _bistatic(S1_TRANSMITTER, S1_RECEIVER)
        with pytest.raises(ValueError, match=r"\[0, pi\]"):
            pair.range_resolution_at_bistatic_angle_m(bistatic_angle_rad)


class TestBistaticAmbiguity:
    """What folds is the range sum, not a range."""

    def test_pulsed_unambiguous_sum_range_is_c_over_prf(self) -> None:
        pair = _bistatic(S2_TRANSMITTER, S2_RECEIVER)
        # Re-derived: a bistatic echo is delayed by (R_t + R_r) / c, so one PRI
        # of delay buys c / PRF of range sum, not c / 2 PRF of range.
        expected_m = SPEED_OF_LIGHT_MPS / S2_TRANSMITTER.prf_hz
        np.testing.assert_allclose(pair.unambiguous_range_m, expected_m, rtol=1e-12)

    def test_waveform_independent_properties_match_the_monostatic_radar(self) -> None:
        """Wavelength, noise and cube shape belong to the chains, not to the siting."""
        pair = _bistatic(S1_TRANSMITTER, S1_RECEIVER)
        monostatic = _radar(S1_TRANSMITTER, S1_RECEIVER)
        assert pair.wavelength_m == monostatic.wavelength_m
        assert pair.noise_power_w == monostatic.noise_power_w
        assert pair.n_samples_per_pri == monostatic.n_samples_per_pri
        assert pair.n_samples_per_chirp == monostatic.n_samples_per_chirp
        assert pair.unambiguous_velocity_mps == monostatic.unambiguous_velocity_mps


class TestBistaticValidation:
    def test_coincident_sites_are_rejected(self) -> None:
        """A zero baseline is a monostatic radar described the hard way."""
        with pytest.raises(ValueError, match="use Radar instead"):
            BistaticRadar(S1_TRANSMITTER, S1_RECEIVER, *DUKE_SITE, *DUKE_SITE)

    @pytest.mark.parametrize("site_index", [0, 1])
    def test_latitude_outside_the_poles_is_rejected(self, site_index: int) -> None:
        sites = [list(DUKE_SITE), list(RDU_SITE)]
        sites[site_index][0] = 91.0
        with pytest.raises(ValueError, match=r"must lie in \[-90, 90\]"):
            BistaticRadar(S1_TRANSMITTER, S1_RECEIVER, *sites[0], *sites[1])

    def test_pulsed_receiver_below_the_bandwidth_is_rejected(self) -> None:
        slow_receiver = Receiver(sample_rate_hz=1.0e6, gain_rx_dbi=30.0, noise_figure_db=3.0)
        with pytest.raises(ValueError, match="would alias"):
            BistaticRadar(S2_TRANSMITTER, slow_receiver, *DUKE_SITE, *RDU_SITE)


class TestBistaticFmcwAmbiguity:
    """The deramp branch of the bistatic sum range, which the pulsed tests miss."""

    def test_fmcw_unambiguous_sum_range_is_c_fs_over_two_alpha(self) -> None:
        """A deramped bistatic echo folds where its beat tone reaches Nyquist.

        The beat frequency of a bistatic deramp is alpha * (R_t + R_r) / c, so
        the largest sum the receiver can represent is the one whose beat sits at
        f_s / 2. Re-derived here from the sweep rate rather than taken from the
        implementation, which is what makes it a check on the factor of two.
        """
        pair = _bistatic(S1_TRANSMITTER, S1_RECEIVER)
        expected_m = (
            SPEED_OF_LIGHT_MPS * (S1_RECEIVER.sample_rate_hz / 2.0) / S1_TRANSMITTER.sweep_rate_hzps
        )
        # rtol 1e-12: two float64 divides against the same constant.
        np.testing.assert_allclose(pair.unambiguous_range_m, expected_m, rtol=1e-12)
