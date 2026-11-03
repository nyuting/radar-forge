"""Tests for propagation paths.

Ground truth is analytic throughout: a path's delay, Doppler shift and amplitude
are checked against the closed-form transit time over the range sum, the
closing-positive Doppler of ``spec/structure.md`` D5, and the radar range
equation. The bistatic builder is checked against its monostatic limit and the
range-sum rate against the two geometries that null it.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.core.propagation import (
    PropagationPaths,
    bistatic_line_of_sight_paths,
    doppler_shift_hz,
    line_of_sight_paths,
    path_amplitude_linear,
    propagation_delay_s,
    propagation_phase_rad,
    range_sum_rate_from_doppler_mps,
    range_sum_rate_mps,
)
from radar_forge.core.radar import BistaticRadar, Radar, Receiver, Transmitter

DUKE_SITE = (36.00250, -78.94100, 60.0)
# A synthetic receiver site 0.2 deg due east of DUKE_SITE; baseline about 18 km.
RDU_SITE = (36.00250, -78.74100, 60.0)

S1_RADAR = Radar(
    Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3, waveform="fmcw"),
    Receiver(1.0e6, 30.0, 3.0),
    *DUKE_SITE,
)
S1_PAIR = BistaticRadar(S1_RADAR.transmitter, S1_RADAR.receiver, *DUKE_SITE, *RDU_SITE)


class TestLineOfSightPaths:
    def test_delay_is_the_two_way_transit_time(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, 15_000.0, 0.0, 10.0)
        np.testing.assert_allclose(paths.delay_s[0], 2.0 * 15_000.0 / 299_792_458.0, rtol=1e-15)

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
        """Keeps the cube on the same scale as Receiver.noise_power_per_sample_w."""
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

    def test_a_zero_cross_section_target_returns_nothing(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, 10_000.0, 80.0, 0.0)
        assert paths.amplitude_linear[0] == 0.0

    def test_handles_several_targets_at_once(self) -> None:
        """Catches broadcasting that reuses one target's range or rate for all of them.

        Each path's delay and Doppler is checked against its own target, and a
        scalar cross-section broadcasts to every path.
        """
        ranges_m = np.array([9_000.0, 12_000.0, 15_000.0])
        velocities_mps = np.array([50.0, -20.0, 0.0])
        paths = line_of_sight_paths(S1_RADAR, ranges_m, velocities_mps, 10.0)
        assert paths.n_paths == 3
        # rtol 1e-12: one multiply and one divide per path.
        np.testing.assert_allclose(paths.delay_s, 2.0 * ranges_m / SPEED_OF_LIGHT_MPS, rtol=1e-12)
        np.testing.assert_allclose(
            paths.doppler_hz, 2.0 * velocities_mps / S1_RADAR.wavelength_m, rtol=1e-12, atol=0.0
        )
        np.testing.assert_array_equal(paths.bounce_count, [1, 1, 1])

    def test_rejects_a_zero_range(self) -> None:
        """Zero is the boundary: a ``< 0`` guard would let a target sit on the radar."""
        with pytest.raises(ValueError, match="strictly positive"):
            line_of_sight_paths(S1_RADAR, 0.0, 0.0, 10.0)

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


class TestBistaticLineOfSightPaths:
    """Bistatic paths, and the half-sum convention the generators depend on."""

    def test_reduces_to_monostatic_when_the_sites_all_but_coincide(self) -> None:
        """The headline equivalence: collapse the baseline and the two agree.

        The sites cannot be made identical — BistaticRadar rejects a zero
        baseline — so they are put one metre apart, which is 1e-4 of a range bin
        and far below every tolerance asserted here.
        """
        near_site = (DUKE_SITE[0], DUKE_SITE[1], DUKE_SITE[2] + 1.0)
        pair = BistaticRadar(S1_RADAR.transmitter, S1_RADAR.receiver, *DUKE_SITE, *near_site)
        range_m, velocity_mps, rcs_m2 = 12.0e3, 80.0, 10.0

        monostatic = line_of_sight_paths(S1_RADAR, range_m, velocity_mps, rcs_m2)
        bistatic = bistatic_line_of_sight_paths(pair, range_m, range_m, 2.0 * velocity_mps, rcs_m2)

        # Identical arithmetic in a different grouping, so float64 exactness.
        np.testing.assert_allclose(bistatic.delay_s, monostatic.delay_s, rtol=1e-12)
        np.testing.assert_allclose(bistatic.doppler_hz, monostatic.doppler_hz, rtol=1e-12)
        np.testing.assert_allclose(
            bistatic.amplitude_linear, monostatic.amplitude_linear, rtol=1e-12
        )

    def test_range_sum_is_the_whole_route(self) -> None:
        """Pins what a path's length means: R_t + R_r, not either range nor half."""
        paths = bistatic_line_of_sight_paths(S1_PAIR, 12.0e3, 15.0e3, 0.0, 10.0)
        np.testing.assert_allclose(paths.range_sum_m, 27.0e3, rtol=1e-12)
        np.testing.assert_allclose(paths.delay_s, 27.0e3 / SPEED_OF_LIGHT_MPS, rtol=1e-15)

    def test_doppler_is_the_range_sum_rate_over_the_wavelength(self) -> None:
        """No factor of two: the rate passed in is already the whole route's."""
        paths = bistatic_line_of_sight_paths(S1_PAIR, 12.0e3, 15.0e3, 160.0, 10.0)
        np.testing.assert_allclose(paths.doppler_hz, 160.0 / S1_PAIR.wavelength_m, rtol=1e-15)

    def test_arrival_and_departure_angles_are_kept_separate(self) -> None:
        """Departure happens at one site and arrival at another, so they differ."""
        paths = bistatic_line_of_sight_paths(
            S1_PAIR,
            12.0e3,
            15.0e3,
            0.0,
            10.0,
            transmit_azimuth_deg=30.0,
            receive_azimuth_deg=210.0,
        )
        np.testing.assert_allclose(paths.aod_rad[0, 0], np.radians(30.0), rtol=1e-12)
        np.testing.assert_allclose(paths.aoa_rad[0, 0], np.radians(210.0), rtol=1e-12)

    def test_broadcasts_over_targets(self) -> None:
        """Every range, rate and cross-section is genuinely per-target."""
        ranges_tx_m = np.array([10.0e3, 12.0e3, 14.0e3])
        ranges_rx_m = np.array([15.0e3, 11.0e3, 20.0e3])
        paths = bistatic_line_of_sight_paths(
            S1_PAIR, ranges_tx_m, ranges_rx_m, np.array([10.0, -20.0, 30.0]), 10.0
        )
        assert paths.n_paths == 3
        np.testing.assert_allclose(paths.range_sum_m, ranges_tx_m + ranges_rx_m, rtol=1e-12)

    def test_zero_cross_section_gives_a_null_path(self) -> None:
        paths = bistatic_line_of_sight_paths(S1_PAIR, 12.0e3, 15.0e3, 0.0, 0.0)
        assert paths.amplitude_linear[0] == 0.0

    @pytest.mark.parametrize(("range_tx_m", "range_rx_m"), [(0.0, 1.0e3), (1.0e3, -1.0)])
    def test_rejects_non_positive_range(self, range_tx_m: float, range_rx_m: float) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            bistatic_line_of_sight_paths(S1_PAIR, range_tx_m, range_rx_m, 0.0, 10.0)

    def test_rejects_negative_cross_section(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            bistatic_line_of_sight_paths(S1_PAIR, 12.0e3, 15.0e3, 0.0, -1.0)


class TestRangeSumRate:
    """The bisector projection, its monostatic limit, and the two geometries that null it."""

    def test_matches_the_bisector_projection(self) -> None:
        """Isoceles geometry: the rate is 2 v cos(beta / 2) along the bisector.

        The target sits on the perpendicular bisector of the baseline and closes
        along it, so delta = 0 and the textbook form reduces to the cosine of
        half the bistatic angle (Willis section 6.1).
        """
        half_baseline_m, offset_m = 11.0e3, 20.0e3
        # Sites at (-L/2, 0, 0) and (+L/2, 0, 0); target on the north bisector.
        to_tx = np.array([-half_baseline_m, -offset_m, 0.0])
        to_rx = np.array([half_baseline_m, -offset_m, 0.0])
        unit_to_tx = to_tx / np.linalg.norm(to_tx)
        unit_to_rx = to_rx / np.linalg.norm(to_rx)

        speed_mps = 80.0
        velocity = np.array([0.0, -speed_mps, 0.0])  # straight down the bisector

        half_angle_rad = np.arctan2(half_baseline_m, offset_m)
        expected_mps = 2.0 * speed_mps * np.cos(half_angle_rad)
        np.testing.assert_allclose(
            range_sum_rate_mps(velocity, unit_to_tx, unit_to_rx), expected_mps, rtol=1e-12
        )

    def test_is_twice_the_radial_velocity_when_the_sites_coincide(self) -> None:
        """The monostatic limit: both unit vectors are the same line of sight."""
        to_site = np.array([3.0, 4.0, 12.0]) / 13.0
        velocity = np.array([30.0, 40.0, 0.0])
        radial_velocity_mps = float(velocity @ to_site)  # 250 / 13 m/s, closing
        np.testing.assert_allclose(
            range_sum_rate_mps(velocity, to_site, to_site), 2.0 * radial_velocity_mps, rtol=1e-15
        )

    def test_is_zero_crossing_the_bisector(self) -> None:
        """Motion perpendicular to the bisector changes no total path length."""
        to_tx = np.array([-1.0, -2.0, 0.0])
        to_rx = np.array([1.0, -2.0, 0.0])
        unit_to_tx = to_tx / np.linalg.norm(to_tx)
        unit_to_rx = to_rx / np.linalg.norm(to_rx)
        velocity = np.array([80.0, 0.0, 0.0])  # across the bisector
        # A cancellation to a true null, so an absolute bound: round-off on an
        # 80 m/s scale is around 1e-14 m/s.
        np.testing.assert_allclose(
            range_sum_rate_mps(velocity, unit_to_tx, unit_to_rx), 0.0, atol=1e-12
        )

    def test_is_zero_moving_along_the_baseline(self) -> None:
        """The bistatic-only null: one range shortens as fast as the other lengthens.

        A target on the baseline between the sites is closing on one at exactly
        the rate it opens on the other, so the pair is blind to motion that is
        purely radial to each site taken alone. A monostatic radar has no
        equivalent of this.
        """
        unit_to_tx = np.array([-1.0, 0.0, 0.0])
        unit_to_rx = np.array([1.0, 0.0, 0.0])
        velocity = np.array([80.0, 0.0, 0.0])
        np.testing.assert_allclose(
            range_sum_rate_mps(velocity, unit_to_tx, unit_to_rx), 0.0, atol=1e-12
        )

    def test_rejects_vectors_that_are_not_three_dimensional(self) -> None:
        with pytest.raises(ValueError, match="three components"):
            range_sum_rate_mps(np.zeros((1, 2)), np.zeros((1, 3)), np.zeros((1, 3)))


class TestPathRelations:
    """The relations every path builder shares, each against its closed form."""

    def test_delay_is_the_range_sum_at_the_speed_of_light(self) -> None:
        np.testing.assert_allclose(propagation_delay_s(27.0e3), 27.0e3 / 299_792_458.0, rtol=1e-15)

    def test_delay_and_range_sum_are_inverses(self) -> None:
        """PropagationPaths.range_sum_m undoes propagation_delay_s."""
        ranges_m = np.array([20.0e3, 27.0e3, 63.4e3])
        paths = PropagationPaths(
            delay_s=propagation_delay_s(ranges_m),
            doppler_hz=np.zeros(3),
            amplitude_linear=np.zeros(3, dtype=np.complex128),
            aoa_rad=np.zeros((3, 2)),
            aod_rad=np.zeros((3, 2)),
            bounce_count=np.ones(3, dtype=np.int_),
        )
        # A divide then a multiply by the same constant: one ulp each way.
        np.testing.assert_allclose(paths.range_sum_m, ranges_m, rtol=1e-15)

    def test_one_wavelength_of_route_is_one_cycle_of_lag(self) -> None:
        wavelength_m = S1_RADAR.wavelength_m
        np.testing.assert_allclose(
            propagation_phase_rad(3.0 * wavelength_m, wavelength_m), -6.0 * np.pi, rtol=1e-15
        )

    def test_doppler_of_a_monostatic_closing_target_is_two_v_over_lambda(self) -> None:
        wavelength_m = S1_RADAR.wavelength_m
        np.testing.assert_allclose(
            doppler_shift_hz(2.0 * 80.0, wavelength_m), 160.0 / wavelength_m, rtol=1e-15
        )

    def test_doppler_and_range_sum_rate_are_inverses(self) -> None:
        rates_mps = np.array([-160.0, 0.0, 37.5, 900.0])
        wavelength_m = S1_RADAR.wavelength_m
        # A divide then a multiply by the same wavelength: one ulp each way.
        np.testing.assert_allclose(
            range_sum_rate_from_doppler_mps(
                doppler_shift_hz(rates_mps, wavelength_m), wavelength_m
            ),
            rates_mps,
            rtol=1e-15,
        )

    def test_amplitude_squared_is_the_power_whatever_the_phase(self) -> None:
        power_w = np.array([1.0e-12, 4.0e-15, 0.0])
        phase_rad = np.array([0.3, -2.0e3, 1.0])
        amplitude = path_amplitude_linear(power_w, phase_rad)
        np.testing.assert_allclose(np.abs(amplitude) ** 2, power_w, rtol=1e-14)
        np.testing.assert_allclose(np.angle(amplitude[:1]), phase_rad[:1], rtol=1e-15)
