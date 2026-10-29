"""Tests for propagation paths.

Ground truth is analytic throughout: a path's delay, Doppler shift and amplitude
are checked against the closed-form two-way transit time, the closing-positive
Doppler of ``spec/structure.md`` D5, and the radar range equation. The bistatic
builder is checked against its monostatic limit and the two geometries that
null bistatic Doppler.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.core.propagation import (
    PropagationPaths,
    bistatic_doppler_hz,
    bistatic_line_of_sight_paths,
    line_of_sight_paths,
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
        bistatic = bistatic_line_of_sight_paths(pair, range_m, range_m, velocity_mps, rcs_m2)

        # Identical arithmetic in a different grouping, so float64 exactness.
        np.testing.assert_allclose(bistatic.delay_s, monostatic.delay_s, rtol=1e-12)
        np.testing.assert_allclose(bistatic.doppler_hz, monostatic.doppler_hz, rtol=1e-12)
        np.testing.assert_allclose(
            bistatic.amplitude_linear, monostatic.amplitude_linear, rtol=1e-12
        )

    def test_equivalent_range_is_half_the_path_sum(self) -> None:
        """Pins the convention every generator silently depends on.

        ``range_m`` is (R_t + R_r) / 2, not a distance to anything. A later
        "correction" of it to the full path sum, or to either range, would leave
        every cube in this module wrong by a factor of two with no other test
        necessarily noticing, so this one states it outright.
        """
        paths = bistatic_line_of_sight_paths(S1_PAIR, 12.0e3, 15.0e3, 0.0, 10.0)
        np.testing.assert_allclose(paths.range_m, 13.5e3, rtol=1e-12)

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
        np.testing.assert_allclose(paths.range_m, (ranges_tx_m + ranges_rx_m) / 2.0, rtol=1e-12)

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


class TestBistaticDoppler:
    """The bisector projection, and the two geometries that null it."""

    def test_matches_the_bisector_projection(self) -> None:
        """Isoceles geometry: f_d = (2 v / lambda) cos(beta / 2) along the bisector.

        The target sits on the perpendicular bisector of the baseline and closes
        along it, so delta = 0 and the textbook form reduces to the cosine of
        half the bistatic angle (Willis §6.1).
        """
        half_baseline_m, offset_m = 11.0e3, 20.0e3
        # Sites at (-L/2, 0, 0) and (+L/2, 0, 0); target on the north bisector.
        to_tx = np.array([-half_baseline_m, -offset_m, 0.0])
        to_rx = np.array([half_baseline_m, -offset_m, 0.0])
        unit_to_tx = to_tx / np.linalg.norm(to_tx)
        unit_to_rx = to_rx / np.linalg.norm(to_rx)

        speed_mps = 80.0
        velocity = np.array([0.0, -speed_mps, 0.0])  # straight down the bisector
        wavelength_m = S1_PAIR.wavelength_m

        half_angle_rad = np.arctan2(half_baseline_m, offset_m)
        expected_hz = 2.0 * speed_mps * np.cos(half_angle_rad) / wavelength_m
        np.testing.assert_allclose(
            bistatic_doppler_hz(wavelength_m, velocity, unit_to_tx, unit_to_rx),
            expected_hz,
            rtol=1e-12,
        )

    def test_is_zero_crossing_the_bisector(self) -> None:
        """Motion perpendicular to the bisector changes no total path length."""
        to_tx = np.array([-1.0, -2.0, 0.0])
        to_rx = np.array([1.0, -2.0, 0.0])
        unit_to_tx = to_tx / np.linalg.norm(to_tx)
        unit_to_rx = to_rx / np.linalg.norm(to_rx)
        velocity = np.array([80.0, 0.0, 0.0])  # across the bisector
        # A cancellation to a true null, so an absolute bound against a kHz scale.
        np.testing.assert_allclose(
            bistatic_doppler_hz(S1_PAIR.wavelength_m, velocity, unit_to_tx, unit_to_rx),
            0.0,
            atol=1e-9,
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
            bistatic_doppler_hz(S1_PAIR.wavelength_m, velocity, unit_to_tx, unit_to_rx),
            0.0,
            atol=1e-9,
        )

    def test_rejects_vectors_that_are_not_three_dimensional(self) -> None:
        with pytest.raises(ValueError, match="three components"):
            bistatic_doppler_hz(
                S1_PAIR.wavelength_m, np.zeros((1, 2)), np.zeros((1, 3)), np.zeros((1, 3))
            )
