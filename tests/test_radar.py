"""Tests for the radar, its sites, and the route geometry of a target.

Ground truth: the scenario-001 parameter table (config/scenario_001_*.toml)
for the waveform limits, closed-form triangles for the bistatic geometry, and
the exact equatorial and cardinal cases for the WGS-84 transforms.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from radar_forge.constants import (
    SPEED_OF_LIGHT_MPS,
    WGS84_FLATTENING,
    WGS84_SEMI_MAJOR_AXIS_M,
)
from radar_forge.radar import (
    Radar,
    Receiver,
    Site,
    TargetGeometry,
    Transmitter,
    ecef_to_enu_m,
    enu_to_range_azimuth_elevation_rad,
    geodetic_to_ecef_m,
    geodetic_to_enu_m,
    range_azimuth_elevation_to_enu_m,
)

# Scenario 001 S1 (FMCW, low PRF: Doppler folds) and S2 (pulsed, medium PRF: range folds).
S1_TRANSMITTER = Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3, waveform="fmcw")
S1_RECEIVER = Receiver(1.0e6, 30.0, 3.0)
S2_TRANSMITTER = Transmitter(9.8e9, 2.0e6, 1.0e3, 30.0, 10.0e-6, 25.0e3, waveform="pulsed")
S2_RECEIVER = Receiver(2.5e6, 30.0, 3.0)

# The Duke receiver and the Raleigh-Durham transmitter of scenario 002 (spec §3.1).
DUKE = Site(36.00250, -78.94100, 60.0)
RDU = Site(35.87750, -78.78750, 25.0)

S1_RADAR = Radar(S1_TRANSMITTER, S1_RECEIVER, DUKE)
S2_RADAR = Radar(S2_TRANSMITTER, S2_RECEIVER, DUKE)
S1_PAIR = Radar(S1_TRANSMITTER, S1_RECEIVER, tx_site=RDU, rx_site=DUKE)


class TestTransmitter:
    def test_wavelength_at_x_band(self) -> None:
        """9.8 GHz gives 30.591 mm, the figure quoted throughout scenario 001."""
        np.testing.assert_allclose(S1_TRANSMITTER.wavelength_m, 0.030_591_067, rtol=1e-7)

    def test_fmcw_sweep_rate_is_two_ghz_per_second(self) -> None:
        """2 MHz over 1 ms is 2.000 GHz/s, per spec §3.4 S1."""
        np.testing.assert_allclose(S1_TRANSMITTER.sweep_rate_hzps, 2.0e9, rtol=1e-12)

    def test_fmcw_runs_at_full_duty(self) -> None:
        """Exactly 100% duty is permitted, and the formula is T * PRF."""
        np.testing.assert_allclose(S1_TRANSMITTER.duty_cycle_linear, 1.0, rtol=1e-12)

    @pytest.mark.parametrize(
        "field",
        ["carrier_frequency_hz", "bandwidth_hz", "transmit_power_w", "chirp_duration_s", "prf_hz"],
    )
    def test_rejects_non_positive_quantities(self, field: str) -> None:
        kwargs = {
            "carrier_frequency_hz": 9.8e9,
            "bandwidth_hz": 2.0e6,
            "transmit_power_w": 100.0,
            "gain_tx_dbi": 30.0,
            "chirp_duration_s": 1.0e-3,
            "prf_hz": 1.0e3,
        }
        kwargs[field] = 0.0
        with pytest.raises(ValueError, match=f"{field} must be strictly positive"):
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
        """Zero is the guard's boundary: a ``<= 0`` check would reject an ideal receiver."""
        np.testing.assert_allclose(Receiver(1.0e6, 30.0, 0.0).noise_figure_linear, 1.0, rtol=1e-15)

    def test_rejects_a_negative_noise_figure(self) -> None:
        with pytest.raises(ValueError, match="cannot improve the SNR"):
            Receiver(1.0e6, 30.0, -1.0)

    def test_rejects_a_non_positive_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            Receiver(0.0, 30.0, 3.0)

    def test_noise_density_is_kt0_times_the_noise_figure(self) -> None:
        """-174 dBm/Hz + F, the first line of every link budget."""
        density_dbm_per_hz = 10.0 * np.log10(S1_RECEIVER.noise_density_w_per_hz * 1e3)
        assert abs(density_dbm_per_hz - (-173.98 + 3.0)) < 0.01

    def test_noise_per_sample_fills_the_sampled_band_not_the_swept_one(self) -> None:
        """S1 sweeps 2 MHz but samples 1 MHz: each sample holds N0 fs, 3.01 dB below kT0BF."""
        ratio_db = 10.0 * np.log10(
            S1_RECEIVER.noise_power_per_sample_w / S1_RADAR.noise_power_matched_w
        )
        np.testing.assert_allclose(ratio_db, 10.0 * np.log10(0.5), rtol=1e-12)


class TestSite:
    def test_rejects_an_impossible_latitude(self) -> None:
        with pytest.raises(ValueError, match=r"\[-90, 90\]"):
            Site(91.0, 103.0, 60.0)

    def test_a_site_is_the_origin_of_its_own_frame(self) -> None:
        """Exactly zero, which is what makes a collocated radar exactly monostatic."""
        np.testing.assert_array_equal(DUKE.enu_m(DUKE), 0.0)


class TestRadarLimits:
    """The scenario-001 waveform limits, now stated for the bistatic range and its rate."""

    def test_range_resolution_is_seventy_five_metres(self) -> None:
        """c/2B at 2 MHz is 74.95 m, per spec §2."""
        np.testing.assert_allclose(S1_RADAR.range_resolution_m, 74.948, rtol=1e-4)

    def test_s1_unambiguous_bistatic_range_is_c_fs_over_alpha(self) -> None:
        """FMCW folds where the beat alpha * R_b / c reaches fs: 149.9 km of R_b.

        That is the map's span; half is the 74.95 km slant that covers the
        track's 18.4 km.
        """
        expected_m = SPEED_OF_LIGHT_MPS * 1.0e6 / 2.0e9
        np.testing.assert_allclose(S1_RADAR.unambiguous_bistatic_range_m, expected_m, rtol=1e-12)
        np.testing.assert_allclose(
            S1_RADAR.unambiguous_bistatic_range_m / 2.0, 74_948.11, rtol=1e-6
        )

    def test_s2_unambiguous_bistatic_range_is_c_over_prf(self) -> None:
        """One PRI of delay is c / PRF of route: 5.996 km of monostatic range, folding 16.6 km."""
        np.testing.assert_allclose(
            S2_RADAR.unambiguous_bistatic_range_m, SPEED_OF_LIGHT_MPS / 25.0e3, rtol=1e-12
        )
        assert S2_RADAR.unambiguous_bistatic_range_m / 2.0 < 16_600.0

    def test_unambiguous_rate_span_is_a_wavelength_per_pulse(self) -> None:
        """lambda * PRF of rate; a quarter is the old +/-7.65 and +/-191.2 m/s radial limits."""
        np.testing.assert_allclose(S1_RADAR.unambiguous_bistatic_range_rate_mps, 30.591, rtol=1e-4)
        np.testing.assert_allclose(
            S1_RADAR.unambiguous_bistatic_range_rate_mps / 4.0, 7.6478, rtol=1e-4
        )
        np.testing.assert_allclose(
            S2_RADAR.unambiguous_bistatic_range_rate_mps / 4.0, 191.194, rtol=1e-5
        )

    def test_matched_noise_power_matches_ktbf(self) -> None:
        """-174 dBm/Hz + 10 log10(B) + F."""
        noise_dbm = 10.0 * np.log10(S1_RADAR.noise_power_matched_w * 1e3)
        assert abs(noise_dbm - (-173.98 + 10.0 * np.log10(2.0e6) + 3.0)) < 0.01

    def test_samples_per_chirp_and_per_pri(self) -> None:
        """A pulsed cube spans the whole 40 us PRI (100 samples), not the 10 us pulse (25)."""
        assert (S1_RADAR.n_samples_per_chirp, S1_RADAR.n_samples_per_pri) == (1000, 1000)
        assert (S2_RADAR.n_samples_per_chirp, S2_RADAR.n_samples_per_pri) == (25, 100)

    def test_rejects_a_pulsed_receiver_that_would_alias(self) -> None:
        """A pulsed receiver must cover the signal bandwidth; FMCW need not."""
        with pytest.raises(ValueError, match="would alias"):
            Radar(S2_TRANSMITTER, Receiver(1.0e6, 30.0, 3.0), DUKE)

    def test_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            S1_RADAR.rx_site = RDU  # type: ignore[misc]  # frozen


class TestSiting:
    """One class for both sitings: monostatic is the zero baseline, not an error."""

    @pytest.mark.parametrize("rx_site", [None, DUKE], ids=["rx-site-none", "rx-site-equal"])
    def test_collocated_sites_are_monostatic_with_zero_baseline(self, rx_site: Site | None) -> None:
        radar = Radar(S1_TRANSMITTER, S1_RECEIVER, DUKE, rx_site)
        assert radar.is_monostatic
        assert radar.baseline_m == 0.0

    def test_baseline_is_the_scenario_002_figure(self) -> None:
        """Spec §3.1 quotes 19.601 km between the two sites, computed rather than assumed."""
        assert not S1_PAIR.is_monostatic
        assert abs(S1_PAIR.baseline_m - 19_601.0) < 0.5

    def test_sites_default_to_the_receiver_frame(self) -> None:
        """The receiver is the origin; the transmitter sits one baseline from it."""
        tx_enu_m, rx_enu_m = S1_PAIR.sites_enu_m()
        np.testing.assert_array_equal(rx_enu_m, 0.0)
        # The norm of the same vector computed the other way round: a rotation apart.
        np.testing.assert_allclose(np.linalg.norm(tx_enu_m), S1_PAIR.baseline_m, rtol=1e-12)

    def test_waveform_properties_do_not_depend_on_the_siting(self) -> None:
        for name in (
            "wavelength_m",
            "noise_power_matched_w",
            "n_samples_per_pri",
            "unambiguous_delay_s",
        ):
            assert getattr(S1_PAIR, name) == getattr(S1_RADAR, name)


def _targets_enu_m() -> tuple[np.ndarray, np.ndarray]:
    """Three targets around the Duke site, with unequal ranges and velocities."""
    position_enu_m = np.array(
        [[3000.0, 9000.0, 1500.0], [-7000.0, 2000.0, 900.0], [500.0, -12000.0, 2500.0]]
    )
    velocity_enu_mps = np.array([[40.0, -70.0, 0.0], [-20.0, 5.0, 3.0], [0.0, 80.0, -1.0]])
    return position_enu_m, velocity_enu_mps


class TestGeometry:
    @pytest.mark.parametrize("rx_site", [None, DUKE], ids=["rx-site-none", "rx-site-equal"])
    def test_a_collocated_radar_reproduces_the_monostatic_geometry_exactly(
        self, rx_site: Site | None
    ) -> None:
        """Catches any arithmetic the zero-baseline route does differently from 2R and 2 dR/dt.

        Exact equality, not a tolerance: R + R is 2R, a sum of two equal dot
        products is twice one, and the law of cosines with L = 0 is exactly 1.
        """
        position_enu_m, velocity_enu_mps = _targets_enu_m()
        geometry = Radar(S1_TRANSMITTER, S1_RECEIVER, DUKE, rx_site).geometry(
            position_enu_m, velocity_enu_mps
        )
        range_m, azimuth_rad, elevation_rad = enu_to_range_azimuth_elevation_rad(position_enu_m)
        range_rate_mps = np.sum(velocity_enu_mps * position_enu_m / range_m[:, None], axis=-1)
        expected = TargetGeometry.monostatic(
            range_m, range_rate_mps, azimuth_rad=azimuth_rad, elevation_rad=elevation_rad
        )
        for name in expected.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(geometry, name), getattr(expected, name), err_msg=name
            )

    def test_a_receding_target_has_a_positive_rate(self) -> None:
        """The repo-wide convention: d(R_b)/dt, positive when the route lengthens."""
        geometry = S1_RADAR.geometry(
            [[0.0, 10_000.0, 0.0]] * 2, [[0.0, 50.0, 0.0], [0.0, -50.0, 0.0]]
        )
        np.testing.assert_allclose(geometry.bistatic_range_rate_mps, [100.0, -100.0], rtol=1e-12)

    @pytest.mark.parametrize("site_index", [0, 1], ids=["on-transmitter", "on-receiver"])
    def test_a_target_standing_on_a_site_is_rejected(self, site_index: int) -> None:
        """Catches a zero range dividing through to silent NaN angles and rate.

        Warnings are errors in this suite, so this also proves the check comes
        before any arc sine of 0/0.
        """
        on_site_m = S1_PAIR.sites_enu_m()[site_index]
        targets_m = np.stack([on_site_m, on_site_m + np.array([0.0, 5_000.0, 1_000.0])])
        with pytest.raises(ValueError, match=r"range > 0"):
            S1_PAIR.geometry(targets_m, np.zeros(3))

    def test_a_hand_built_geometry_must_agree_on_the_number_of_targets(self) -> None:
        good = TargetGeometry.monostatic([1.0e4, 2.0e4], 0.0)
        with pytest.raises(ValueError, match=r"bistatic_range_rate_mps must have shape \(2,\)"):
            TargetGeometry(**{**vars(good), "bistatic_range_rate_mps": np.zeros(3)})
        with pytest.raises(ValueError, match=r"aoa_rad must have shape \(2, 2\)"):
            TargetGeometry(**{**vars(good), "aoa_rad": np.zeros(2)})

    def test_a_target_on_the_baseline_is_forward_scatter(self) -> None:
        """beta = pi and R_b = L: the clipped cosine keeps the arc cosine off NaN.

        arccos near -1 turns a cosine error d into an angle error sqrt(2 d);
        the ranges carry about 1e-15 relative error, so 1e-6 rad is a bound,
        and still a million times tighter than a wrong answer.
        """
        tx_enu_m, rx_enu_m = S1_PAIR.sites_enu_m()
        on_baseline_m = tx_enu_m + 0.3 * (rx_enu_m - tx_enu_m)
        geometry = S1_PAIR.geometry(on_baseline_m, np.zeros(3))
        assert not np.isnan(geometry.bistatic_angle_rad[0])
        np.testing.assert_allclose(geometry.bistatic_angle_rad, np.pi, atol=1e-6)
        np.testing.assert_allclose(geometry.bistatic_range_m, S1_PAIR.baseline_m, rtol=1e-12)

    def test_the_isosceles_right_triangle_has_a_right_bistatic_angle(self) -> None:
        """R_T = R_R = L / sqrt(2) puts the target at the corner of half a square: beta = pi / 2."""
        tx_enu_m, rx_enu_m = S1_PAIR.sites_enu_m()
        baseline_vector_m = rx_enu_m - tx_enu_m
        across = np.cross(baseline_vector_m, [0.0, 0.0, 1.0])
        target_m = (tx_enu_m + rx_enu_m) / 2.0 + across / np.linalg.norm(
            across
        ) * S1_PAIR.baseline_m / 2.0
        geometry = S1_PAIR.geometry(target_m, np.zeros(3))
        # A handful of float64 operations on 2e4 m lengths.
        np.testing.assert_allclose(
            geometry.range_tx_m, S1_PAIR.baseline_m / np.sqrt(2.0), rtol=1e-12
        )
        np.testing.assert_allclose(geometry.bistatic_angle_rad, np.pi / 2.0, rtol=1e-12)

    def test_the_rate_is_twice_the_speed_times_cos_half_beta_along_the_bisector(self) -> None:
        """Willis §6.1: a target moving along the bisector has rate 2 v cos(beta / 2).

        Unequal from the monostatic 2v, so a rate that ignores the second
        site's direction cannot pass; moving away from the baseline, so positive.
        """
        tx_enu_m, rx_enu_m = S1_PAIR.sites_enu_m()
        midpoint_m = (tx_enu_m + rx_enu_m) / 2.0
        across = np.cross(rx_enu_m - tx_enu_m, [0.0, 0.0, 1.0])
        bisector = across / np.linalg.norm(across)
        target_m = midpoint_m + 15_000.0 * bisector
        speed_mps = 80.0
        geometry = S1_PAIR.geometry(target_m, speed_mps * bisector)
        expected_mps = 2.0 * speed_mps * np.cos(geometry.bistatic_angle_rad / 2.0)
        assert 0.3 < geometry.bistatic_angle_rad[0] < 2.0  # far from both degenerate cases
        np.testing.assert_allclose(geometry.bistatic_range_rate_mps, expected_mps, rtol=1e-9)

    def test_motion_along_the_baseline_does_not_change_the_route(self) -> None:
        """The bistatic-only null: one range shortens as fast as the other lengthens."""
        tx_enu_m, rx_enu_m = S1_PAIR.sites_enu_m()
        along = (rx_enu_m - tx_enu_m) / S1_PAIR.baseline_m
        geometry = S1_PAIR.geometry(tx_enu_m + 0.4 * (rx_enu_m - tx_enu_m), 80.0 * along)
        # A cancellation to a true null: round-off on an 80 m/s scale.
        np.testing.assert_allclose(geometry.bistatic_range_rate_mps, 0.0, atol=1e-9)

    def test_look_angles_are_measured_in_each_site_frame_whatever_the_origin(self) -> None:
        """Catches departure angles read in the receiver's frame instead of the transmitter's.

        The same targets, given about either site, must give the same angles.
        Over this 19.6 km baseline the two frames' north axes differ by about
        0.09 deg of azimuth (the meridians converge), which is also asserted, so
        the test cannot pass by the frames being aligned.
        """
        latitude_deg = np.array([35.95, 36.05, 35.90])
        longitude_deg = np.array([-78.85, -78.70, -79.00])
        altitude_m = np.array([1500.0, 3000.0, 800.0])
        about_rx_m = geodetic_to_enu_m(
            latitude_deg, longitude_deg, altitude_m, 36.0025, -78.941, 60.0
        )
        about_tx_m = geodetic_to_enu_m(
            latitude_deg, longitude_deg, altitude_m, 35.8775, -78.7875, 25.0
        )
        from_rx = S1_PAIR.geometry(about_rx_m, np.zeros(3))
        from_tx = S1_PAIR.geometry(about_tx_m, np.zeros(3), origin=RDU)
        # Rotations of 2e4 m vectors in float64.
        np.testing.assert_allclose(from_rx.aod_rad, from_tx.aod_rad, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(from_rx.aoa_rad, from_tx.aoa_rad, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(from_rx.range_tx_m, from_tx.range_tx_m, rtol=1e-12)
        tx_enu_m, _ = S1_PAIR.sites_enu_m()
        _, azimuth_in_rx_axes_rad, _ = enu_to_range_azimuth_elevation_rad(about_rx_m - tx_enu_m)
        assert np.all(np.abs(azimuth_in_rx_axes_rad - from_rx.aod_rad[:, 0]) > np.radians(0.05))


class TestGeodeticToEcef:
    def test_equator_prime_meridian_is_the_semi_major_axis(self) -> None:
        ecef_m = geodetic_to_ecef_m(0.0, 0.0, 0.0)
        np.testing.assert_allclose(ecef_m[0], WGS84_SEMI_MAJOR_AXIS_M, rtol=1e-15)
        np.testing.assert_allclose(ecef_m[1:], 0.0, atol=1e-9)

    def test_north_pole_is_the_semi_minor_axis(self) -> None:
        """At phi = 90 deg, z = a (1 - f) = b, the polar radius."""
        ecef_m = geodetic_to_ecef_m(90.0, 0.0, 0.0)
        np.testing.assert_allclose(
            ecef_m[2], WGS84_SEMI_MAJOR_AXIS_M * (1.0 - WGS84_FLATTENING), rtol=1e-12
        )
        np.testing.assert_allclose(ecef_m[:2], 0.0, atol=1e-9)

    def test_rejects_a_latitude_south_of_the_pole(self) -> None:
        """A southern case catches a guard that forgot the absolute value."""
        with pytest.raises(ValueError, match=r"\[-90, 90\]"):
            geodetic_to_ecef_m(-90.1, 0.0, 0.0)


class TestGeodeticToEnu:
    def test_altitude_above_the_origin_is_pure_up(self) -> None:
        enu_m = geodetic_to_enu_m(36.0025, -78.941, 1560.0, 36.0025, -78.941, 60.0)
        np.testing.assert_allclose(enu_m[2], 1500.0, rtol=1e-9)
        np.testing.assert_allclose(enu_m[:2], 0.0, atol=1e-6)

    def test_equatorial_east_step_and_the_earth_curving_away(self) -> None:
        """At the equator, 0.18 deg east is a sin(d) east and -a (1 - cos d) up, about -31.5 m.

        Catches east and north swapped, a flipped sign, or a wrong "up".
        """
        step_rad = np.radians(0.18)
        enu_m = geodetic_to_enu_m(0.0, np.degrees(step_rad), 0.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(enu_m[0], WGS84_SEMI_MAJOR_AXIS_M * np.sin(step_rad), rtol=1e-12)
        np.testing.assert_allclose(enu_m[1], 0.0, atol=1e-9)
        # a cos(d) - a cancels 6.4e6 m to 31 m: about 1e-9 m of round-off.
        np.testing.assert_allclose(
            enu_m[2], -WGS84_SEMI_MAJOR_AXIS_M * (1.0 - np.cos(step_rad)), rtol=1e-9
        )

    def test_is_a_rigid_motion_so_distance_is_preserved(self) -> None:
        """The ENU norm is the chord, which is the slant range that sets the delay."""
        rng = np.random.default_rng(20260911)
        latitude_deg = 36.0025 + rng.uniform(-0.1, 0.1, size=32)
        longitude_deg = -78.941 + rng.uniform(-0.1, 0.1, size=32)
        altitude_m = rng.uniform(0.0, 3000.0, size=32)
        ecef_distance_m = np.linalg.norm(
            geodetic_to_ecef_m(latitude_deg, longitude_deg, altitude_m)
            - geodetic_to_ecef_m(36.0025, -78.941, 60.0),
            axis=-1,
        )
        enu_m = geodetic_to_enu_m(latitude_deg, longitude_deg, altitude_m, 36.0025, -78.941, 60.0)
        np.testing.assert_allclose(np.linalg.norm(enu_m, axis=-1), ecef_distance_m, rtol=1e-12)

    def test_rejects_a_last_axis_that_is_not_three(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
            ecef_to_enu_m(np.zeros((4, 2)), 0.0, 0.0, 0.0)


class TestSphericalCoordinates:
    def test_azimuth_is_zero_at_north_and_increases_clockwise(self) -> None:
        """Catches the mathematical convention, a flipped sense, or a centred wrap."""
        _, azimuth_rad, _ = enu_to_range_azimuth_elevation_rad(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        np.testing.assert_allclose(azimuth_rad, [0.0, np.pi / 2, np.pi, 3 * np.pi / 2], atol=1e-15)

    def test_a_pythagorean_point_gives_exact_range_and_angles(self) -> None:
        """(300, 400, 1200) is 3-4-5 on 5-12-13: range 1300 m, elevation asin(12/13)."""
        range_m, azimuth_rad, elevation_rad = enu_to_range_azimuth_elevation_rad(
            (300.0, 400.0, 1200.0)
        )
        np.testing.assert_allclose(range_m, 1300.0, rtol=1e-12)
        np.testing.assert_allclose(elevation_rad, np.arcsin(12.0 / 13.0), rtol=1e-12)
        np.testing.assert_allclose(azimuth_rad, np.arctan2(3.0, 4.0), rtol=1e-12)

    def test_rejects_a_last_axis_that_is_not_three(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
            enu_to_range_azimuth_elevation_rad(np.zeros((4, 2)))

    def test_the_inverse_puts_cardinal_azimuths_on_the_axes(self) -> None:
        """Catches sine and cosine swapped, which turns north into east."""
        enu_m = range_azimuth_elevation_to_enu_m(
            1000.0, np.array([np.pi / 2, 0.0, 1.0]), np.array([0.0, 0.0, np.pi / 2])
        )
        # atol 1e-9 m: cos(pi / 2) is 6e-17, times 1000 m.
        np.testing.assert_allclose(
            enu_m, [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]], atol=1e-9
        )

    def test_the_inverse_round_trips(self) -> None:
        enu_m = np.array(
            [[300.0, 400.0, 1200.0], [-250.0, 80.0, -40.0], [-10.0, -900.0, 5.0], [7.0, -3.0, 2.0]]
        )
        recovered_m = range_azimuth_elevation_to_enu_m(*enu_to_range_azimuth_elevation_rad(enu_m))
        np.testing.assert_allclose(recovered_m, enu_m, rtol=1e-12, atol=1e-12)


# --- Scenario 005: a continuous noise code ---

NOISE_TRANSMITTER = Transmitter(9.8e9, 8.0e6, 1.0, 30.0, 50.0e-6, 20.0e3, waveform="noise")
DUKE = Site(36.0025, -78.941, 60.0)


def test_a_noise_codes_batch_is_its_pulse_at_duty_one():
    """The correlation batch stands in for the pulse: PRF is the batch rate, 400 samples a batch."""
    radar = Radar(NOISE_TRANSMITTER, Receiver(8.0e6, 30.0, 3.0), DUKE)
    assert NOISE_TRANSMITTER.duty_cycle_linear == 1.0
    assert radar.n_samples_per_chirp == radar.n_samples_per_pri == 400


def test_a_noise_code_fills_the_sampled_band():
    """A white code over f_s has bandwidth f_s; any other bandwidth labels a band not sent."""
    with pytest.raises(ValueError, match="fills the sampled band"):
        Radar(NOISE_TRANSMITTER, Receiver(16.0e6, 30.0, 3.0), DUKE)


def test_a_noise_code_folds_no_delay():
    """Batched correlation reads each lag whole: no delay period, so no range folds."""
    radar = Radar(NOISE_TRANSMITTER, Receiver(8.0e6, 30.0, 3.0), DUKE)
    assert math.isinf(radar.unambiguous_delay_s)
    assert math.isinf(radar.unambiguous_bistatic_range_m)
