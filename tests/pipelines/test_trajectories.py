"""Tests for trajectory loading, resampling and the radar-frame transform.

The geometry is checked against closed-form answers on synthetic tracks, per
docs/conventions/testing.md S3: a target flying straight at the radar must
report its speed exactly, and a target circling the radar must report zero.
The recorded CSV cannot check any of that -- it carries no speed column -- so
it is exercised only for loading and plumbing, which is the division the
scenario spec S3 asks for.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from radar_forge.core.constants import WGS84_FLATTENING, WGS84_SEMI_MAJOR_AXIS_M
from radar_forge.core.radar import BistaticRadar, Radar, Receiver, Transmitter
from radar_forge.pipelines.trajectories import (
    Trajectory,
    load_flight_csv,
    resample,
    to_bistatic_radar_frame,
    to_radar_frame,
)

GOLDEN_CSV = Path(__file__).parent.parent / "data" / "golden" / "flight_coordinates_head.csv"

DUKE_SITE = (36.00250, -78.94100, 60.0)
S1_RADAR = Radar(
    Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3, waveform="fmcw"),
    Receiver(1.0e6, 30.0, 3.0),
    *DUKE_SITE,
)

TARGET_ALTITUDE_M = 1500.0

# Scenario 002: the Raleigh-Durham illuminator, received at the Duke Receiver.
# A 19.6 km baseline, so the two ranges differ by kilometres, not by rounding.
RDU_SITE = (35.87750, -78.78750, 25.0)
_BISTATIC_PAIR = BistaticRadar(
    S1_RADAR.transmitter,
    S1_RADAR.receiver,
    *RDU_SITE,
    *DUKE_SITE,
)


def _curvature_radii_m(latitude_deg: float) -> tuple[float, float]:
    """Meridian and prime-vertical radii of curvature at a latitude, metres.

    Synthetic tracks are laid out in local metres and converted back to
    degrees, and doing that with a single flat scale factor is not good enough:
    the two radii differ by 0.44 % at the Duke Receiver's latitude, which turns a
    circle into an ellipse and leaks a false radial velocity into the null
    test below. See Torge, *Geodesy*, 4th ed., S4.2.
    """
    eccentricity_squared = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
    sin_latitude = np.sin(np.radians(latitude_deg))
    denominator = 1.0 - eccentricity_squared * sin_latitude**2
    meridian_m = WGS84_SEMI_MAJOR_AXIS_M * (1.0 - eccentricity_squared) / denominator**1.5
    prime_vertical_m = WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(denominator)
    return float(meridian_m), float(prime_vertical_m)


MERIDIAN_RADIUS_M, PRIME_VERTICAL_RADIUS_M = _curvature_radii_m(DUKE_SITE[0])


def _straight_north_track(speed_mps: float, duration_s: float, n_fixes: int) -> Trajectory:
    """A target due north of the radar, flying directly away from it."""
    time_s = np.linspace(0.0, duration_s, n_fixes)
    start_offset_m = 10_000.0
    north_m = start_offset_m + speed_mps * time_s
    latitude_deg = DUKE_SITE[0] + np.degrees(north_m / MERIDIAN_RADIUS_M)
    return Trajectory(
        time_s=time_s,
        latitude_deg=latitude_deg,
        longitude_deg=np.full(time_s.shape, DUKE_SITE[1]),
        altitude_m=np.full(time_s.shape, DUKE_SITE[2]),
    )


class TestLoadFlightCsv:
    def test_reads_every_row_of_the_golden_excerpt(self) -> None:
        trajectory = load_flight_csv(GOLDEN_CSV, altitude_m=TARGET_ALTITUDE_M)
        assert trajectory.n_fixes == 50  # 51 lines, one of them the header

    def test_time_is_measured_from_the_first_fix(self) -> None:
        trajectory = load_flight_csv(GOLDEN_CSV, altitude_m=TARGET_ALTITUDE_M)
        assert trajectory.time_s[0] == 0.0
        assert trajectory.epoch == datetime(2026, 9, 3, 0, 17, 56, tzinfo=UTC)

    def test_the_second_fix_is_eight_seconds_in(self) -> None:
        """00:17:56Z to 00:18:04Z -- the irregular sampling the spec S3 warns about."""
        trajectory = load_flight_csv(GOLDEN_CSV, altitude_m=TARGET_ALTITUDE_M)
        assert trajectory.time_s[1] == 8.0

    def test_timestamps_are_strictly_increasing(self) -> None:
        trajectory = load_flight_csv(GOLDEN_CSV, altitude_m=TARGET_ALTITUDE_M)
        assert np.all(np.diff(trajectory.time_s) > 0.0)

    def test_altitude_is_the_constant_it_was_given(self) -> None:
        """The CSV has no altitude column; it is a scenario parameter."""
        trajectory = load_flight_csv(GOLDEN_CSV, altitude_m=TARGET_ALTITUDE_M)
        np.testing.assert_array_equal(
            trajectory.altitude_m, np.full(trajectory.n_fixes, TARGET_ALTITUDE_M)
        )

    def test_rejects_a_file_missing_a_column(self, tmp_path: Path) -> None:
        bad = tmp_path / "no_lon.csv"
        bad.write_text("timestamp,lat\n2026-09-03T00:17:56Z,1.4\n")
        with pytest.raises(ValueError, match="missing required column"):
            load_flight_csv(bad, altitude_m=TARGET_ALTITUDE_M)

    def test_rejects_a_file_with_a_single_fix(self, tmp_path: Path) -> None:
        bad = tmp_path / "one_row.csv"
        bad.write_text("timestamp,lat,lon\n2026-09-03T00:17:56Z,35.9,-78.9\n")
        with pytest.raises(ValueError, match="at least two"):
            load_flight_csv(bad, altitude_m=TARGET_ALTITUDE_M)

    def test_rejects_out_of_order_timestamps(self, tmp_path: Path) -> None:
        bad = tmp_path / "backwards.csv"
        bad.write_text(
            "timestamp,lat,lon\n2026-09-03T00:17:56Z,35.9,-78.9\n2026-09-03T00:17:50Z,35.9,-78.9\n"
        )
        with pytest.raises(ValueError, match="strictly increasing"):
            load_flight_csv(bad, altitude_m=TARGET_ALTITUDE_M)

    def test_reads_a_crlf_file_unchanged(self, tmp_path: Path) -> None:
        """The shipped track has CRLF endings; csv with newline='' handles them."""
        crlf = tmp_path / "crlf.csv"
        crlf.write_bytes(
            b"timestamp,lat,lon\r\n"
            b"2026-09-03T00:17:56Z,1.38778,103.6824\r\n"
            b"2026-09-03T00:18:04Z,1.38649,103.68047\r\n"
        )
        trajectory = load_flight_csv(crlf, altitude_m=TARGET_ALTITUDE_M)
        assert trajectory.n_fixes == 2
        np.testing.assert_allclose(trajectory.longitude_deg[1], 103.68047, rtol=1e-15)


class TestResample:
    def test_lands_on_the_requested_grid(self) -> None:
        trajectory = load_flight_csv(GOLDEN_CSV, altitude_m=TARGET_ALTITUDE_M)
        times_s = np.arange(0.0, 100.0, 1.0)
        resampled = resample(trajectory, times_s)
        np.testing.assert_array_equal(resampled.time_s, times_s)

    def test_reproduces_the_original_fixes_exactly(self) -> None:
        """Interpolating onto the input's own times is the identity."""
        trajectory = load_flight_csv(GOLDEN_CSV, altitude_m=TARGET_ALTITUDE_M)
        resampled = resample(trajectory, trajectory.time_s)
        # Linear interpolation at a node returns the node, to the last bit.
        np.testing.assert_allclose(resampled.latitude_deg, trajectory.latitude_deg, rtol=1e-15)
        np.testing.assert_allclose(resampled.longitude_deg, trajectory.longitude_deg, rtol=1e-15)

    def test_the_midpoint_of_a_gap_is_the_mean_of_its_ends(self) -> None:
        """The defining property of linear interpolation."""
        trajectory = load_flight_csv(GOLDEN_CSV, altitude_m=TARGET_ALTITUDE_M)
        midpoints_s = 0.5 * (trajectory.time_s[:-1] + trajectory.time_s[1:])
        resampled = resample(trajectory, midpoints_s)
        np.testing.assert_allclose(
            resampled.latitude_deg,
            0.5 * (trajectory.latitude_deg[:-1] + trajectory.latitude_deg[1:]),
            rtol=1e-12,  # a multiply and an add on top of the interpolation
        )

    def test_keeps_the_epoch(self) -> None:
        trajectory = load_flight_csv(GOLDEN_CSV, altitude_m=TARGET_ALTITUDE_M)
        assert resample(trajectory, np.arange(0.0, 10.0)).epoch == trajectory.epoch

    @pytest.mark.parametrize("bad_time_s", [-1.0, 1.0e6])
    def test_refuses_to_extrapolate(self, bad_time_s: float) -> None:
        """np.interp would silently hold the endpoint, faking a stationary target."""
        trajectory = load_flight_csv(GOLDEN_CSV, altitude_m=TARGET_ALTITUDE_M)
        with pytest.raises(ValueError, match="outside the track"):
            resample(trajectory, np.array([bad_time_s]))


class TestToRadarFrame:
    def test_a_target_over_the_radar_site_is_at_its_own_height(self) -> None:
        """The known null of the geodesy: zero horizontal offset, range is height."""
        time_s = np.array([0.0, 1.0, 2.0])
        trajectory = Trajectory(
            time_s=time_s,
            latitude_deg=np.full(3, DUKE_SITE[0]),
            longitude_deg=np.full(3, DUKE_SITE[1]),
            altitude_m=np.full(3, DUKE_SITE[2] + 1000.0),
        )
        track = to_radar_frame(trajectory, S1_RADAR)
        np.testing.assert_allclose(track.range_m, 1000.0, rtol=1e-9)
        np.testing.assert_allclose(track.elevation_deg, 90.0, rtol=1e-9)

    def test_a_target_due_north_reads_zero_azimuth(self) -> None:
        """Azimuth zero at true north, increasing clockwise, per D5.

        Compared on the circle: the azimuth axis is wrapped to [0, 360), and a
        due-north target whose east offset rounds to -1e-12 metres legitimately
        reads 359.999... rather than 0. Both are the same bearing.
        """
        track = to_radar_frame(_straight_north_track(0.0, 10.0, 11), S1_RADAR)
        offset_deg = (track.azimuth_deg + 180.0) % 360.0 - 180.0
        np.testing.assert_allclose(offset_deg, 0.0, atol=1e-9)

    def test_a_target_due_east_reads_ninety_degrees(self) -> None:
        """Clockwise, so east is +90 -- the sign that distinguishes the convention."""
        time_s = np.array([0.0, 1.0, 2.0])
        east_m = 10_000.0
        trajectory = Trajectory(
            time_s=time_s,
            latitude_deg=np.full(3, DUKE_SITE[0]),
            longitude_deg=np.full(
                3,
                DUKE_SITE[1]
                + np.degrees(east_m / (PRIME_VERTICAL_RADIUS_M * np.cos(np.radians(DUKE_SITE[0])))),
            ),
            altitude_m=np.full(3, DUKE_SITE[2]),
        )
        track = to_radar_frame(trajectory, S1_RADAR)
        # Not exactly 90: meridian convergence. A point at the same geodetic
        # latitude d = 10 km east sits d^2 tan(phi) / 2N *north* in the local
        # tangent plane. At the Duke Receiver's 36.0025 deg N that is
        # 1e8 * 0.72668 / (2 * 6 385 527) = 5.690 m, or atan(5.690 / 10 000) =
        # 0.0326 deg of azimuth. The pre-translation site was at 1.2915 deg N,
        # where tan(phi) is 32 times smaller and the same offset was 0.177 m
        # and 0.001 deg; the whole of the change is tan(phi). That is a real
        # property of the ellipsoid, so the tolerance admits it rather than
        # pretending the flat-earth answer is right.
        np.testing.assert_allclose(track.azimuth_deg, 90.0, atol=4e-2)
        assert np.all(track.azimuth_deg < 90.0)

    def test_a_receding_target_reports_negative_velocity(self) -> None:
        """Closing is positive, so flying away must come back negative."""
        speed_mps = 80.0
        track = to_radar_frame(_straight_north_track(speed_mps, 100.0, 101), S1_RADAR)
        assert np.all(track.radial_velocity_mps < 0.0)
        # A straight radial flight: the whole speed is radial. Tolerance is loose
        # because the synthetic track uses a flat degrees-to-metres factor while
        # the transform uses full WGS-84 geometry.
        np.testing.assert_allclose(track.radial_velocity_mps, -speed_mps, rtol=1e-3)

    def test_a_closing_target_reports_positive_velocity(self) -> None:
        speed_mps = 80.0
        track = to_radar_frame(_straight_north_track(-speed_mps, 100.0, 101), S1_RADAR)
        assert np.all(track.radial_velocity_mps > 0.0)
        np.testing.assert_allclose(track.radial_velocity_mps, speed_mps, rtol=1e-3)

    def test_a_target_circling_the_radar_has_no_radial_velocity(self) -> None:
        """A known null: all motion is tangential, so Doppler must vanish.

        The circle is built in the local tangent plane and converted back, so
        the null tests the geometry rather than the construction.
        """
        time_s = np.linspace(0.0, 600.0, 601)
        radius_m = 10_000.0
        bearing_rad = 2.0 * np.pi * time_s / 600.0
        north_m = radius_m * np.cos(bearing_rad)
        east_m = radius_m * np.sin(bearing_rad)
        trajectory = Trajectory(
            time_s=time_s,
            latitude_deg=DUKE_SITE[0] + np.degrees(north_m / MERIDIAN_RADIUS_M),
            longitude_deg=DUKE_SITE[1]
            + np.degrees(east_m / (PRIME_VERTICAL_RADIUS_M * np.cos(np.radians(DUKE_SITE[0])))),
            altitude_m=np.full(time_s.shape, DUKE_SITE[2]),
        )
        track = to_radar_frame(trajectory, S1_RADAR)
        # Not exactly zero, because the circle is laid out to *first* order in
        # the two radii of curvature and the ellipsoid is not a sphere. The
        # leading neglected term is the same meridian convergence as above: a
        # point n north and e east of the site lands e * n * tan(phi) / M too
        # far east, so the range is off by e * (that) / R, whose maximum over
        # the circle is R^2 * 2 / (3 sqrt 3) * tan(phi) / M. At 36.0025 deg N
        # that is 1e8 * 0.3849 * 0.72668 / 6 357 485 = 4.40 m, and the circle's
        # range does wander by 4.35 m. Over the 600 s circuit that is 0.060 m/s
        # against a 105 m/s tangential speed.
        #
        # The threshold moved from 0.01 to 0.1 m/s purely because tan(phi) did:
        # at the pre-translation 1.2915 deg N the same expression gives 0.137 m
        # and 0.00185 m/s, a factor of 32 smaller, and 32 is tan(36.0025) /
        # tan(1.2915). Nothing about the transform changed. The failure mode
        # being guarded is unchanged too: laying the circle out with a single
        # flat scale factor instead of the two radii puts the residual at
        # 0.485 m/s here, eight times the threshold.
        assert np.max(np.abs(track.radial_velocity_mps)) < 0.1

    def test_shapes_follow_the_trajectory(self) -> None:
        trajectory = load_flight_csv(GOLDEN_CSV, altitude_m=TARGET_ALTITUDE_M)
        track = to_radar_frame(trajectory, S1_RADAR)
        assert track.n_frames == trajectory.n_fixes
        for name in ("range_m", "azimuth_deg", "elevation_deg", "radial_velocity_mps"):
            assert getattr(track, name).shape == (trajectory.n_fixes,)


class TestTrajectoryValidation:
    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="same shape as time_s"):
            Trajectory(
                time_s=np.zeros(3),
                latitude_deg=np.zeros(2),
                longitude_deg=np.zeros(3),
                altitude_m=np.zeros(3),
            )

    def test_rejects_a_single_fix(self) -> None:
        with pytest.raises(ValueError, match="at least two fixes"):
            Trajectory(
                time_s=np.zeros(1),
                latitude_deg=np.zeros(1),
                longitude_deg=np.zeros(1),
                altitude_m=np.zeros(1),
            )

    def test_rejects_repeated_times(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            Trajectory(
                time_s=np.array([0.0, 1.0, 1.0]),
                latitude_deg=np.zeros(3),
                longitude_deg=np.zeros(3),
                altitude_m=np.zeros(3),
            )

    def test_rejects_a_two_dimensional_time_axis(self) -> None:
        """Time is one axis; a (n, 1) column would broadcast against position.

        A column vector passes the equal-shape check if every field carries it,
        and then every geometry downstream silently gains an axis, so the
        dimensionality is checked in its own right.
        """
        with pytest.raises(ValueError, match="one-dimensional"):
            Trajectory(
                time_s=np.zeros((3, 1)),
                latitude_deg=np.zeros((3, 1)),
                longitude_deg=np.zeros((3, 1)),
                altitude_m=np.zeros((3, 1)),
            )

    def test_the_duration_is_the_span_from_the_first_fix_to_the_last(self) -> None:
        """duration_s is a span, not a count of fixes or an interval.

        The scenario window is validated against it, so an off-by-one reading
        here would let a window run off the end of the track.
        """
        trajectory = _straight_north_track(speed_mps=100.0, duration_s=37.5, n_fixes=4)
        # Exact: the endpoints are the linspace endpoints, subtracted once.
        assert trajectory.duration_s == 37.5
        assert trajectory.n_fixes == 4


class TestToBistaticRadarFrame:
    """The two-site transform, against the one-site transform it generalises."""

    @staticmethod
    def _collapsed_pair(offset_m: float = 1.0) -> BistaticRadar:
        """S1's radar re-sited as a pair a metre across.

        A zero baseline is rejected -- the bistatic angle is undefined there --
        so the smallest honest test of the degenerate case is a metre.
        """
        return BistaticRadar(
            transmitter=S1_RADAR.transmitter,
            receiver=S1_RADAR.receiver,
            transmitter_latitude_deg=S1_RADAR.latitude_deg,
            transmitter_longitude_deg=S1_RADAR.longitude_deg,
            transmitter_altitude_m=S1_RADAR.altitude_m,
            receiver_latitude_deg=S1_RADAR.latitude_deg,
            receiver_longitude_deg=S1_RADAR.longitude_deg,
            receiver_altitude_m=S1_RADAR.altitude_m + offset_m,
        )

    def test_a_collapsed_baseline_reproduces_the_monostatic_track(self) -> None:
        """The headline equivalence, at the trajectory layer."""
        trajectory = _straight_north_track(speed_mps=80.0, duration_s=60.0, n_fixes=61)
        monostatic = to_radar_frame(trajectory, S1_RADAR)
        bistatic = to_bistatic_radar_frame(trajectory, self._collapsed_pair())

        # Half a metre of baseline against ranges of kilometres.
        np.testing.assert_allclose(bistatic.range_m, monostatic.range_m, atol=1.0)
        np.testing.assert_allclose(
            bistatic.bisector_velocity_mps, monostatic.radial_velocity_mps, atol=1e-6
        )
        # Compared on the circle: a due-north track sits on the 0/360 seam, so
        # the two transforms straddle it and a direct difference reads 360.
        azimuth_error_deg = (bistatic.receive_azimuth_deg - monostatic.azimuth_deg + 180.0) % 360.0
        np.testing.assert_allclose(azimuth_error_deg, 180.0, atol=1e-6)

    def test_the_mean_range_is_the_half_sum_of_the_two_ranges(self) -> None:
        """Pins the convention the whole bistatic pipeline rests on."""
        trajectory = _straight_north_track(speed_mps=80.0, duration_s=60.0, n_fixes=61)
        track = to_bistatic_radar_frame(trajectory, _BISTATIC_PAIR)
        np.testing.assert_allclose(
            track.range_m, (track.range_tx_m + track.range_rx_m) / 2.0, rtol=1e-12
        )

    def test_the_two_ranges_genuinely_differ(self) -> None:
        """Otherwise every half-sum assertion would pass on a monostatic bug."""
        trajectory = _straight_north_track(speed_mps=80.0, duration_s=60.0, n_fixes=61)
        track = to_bistatic_radar_frame(trajectory, _BISTATIC_PAIR)
        assert np.all(np.abs(track.range_tx_m - track.range_rx_m) > 1.0e3)

    def test_the_departure_and_arrival_angles_differ(self) -> None:
        """Two sites, two look directions; that is what makes it bistatic."""
        trajectory = _straight_north_track(speed_mps=80.0, duration_s=60.0, n_fixes=61)
        track = to_bistatic_radar_frame(trajectory, _BISTATIC_PAIR)
        assert np.all(np.abs(track.transmit_azimuth_deg - track.receive_azimuth_deg) > 1.0)

    def test_a_stationary_target_has_no_bisector_rate(self) -> None:
        trajectory = _straight_north_track(speed_mps=0.0, duration_s=60.0, n_fixes=61)
        track = to_bistatic_radar_frame(trajectory, _BISTATIC_PAIR)
        np.testing.assert_allclose(track.bisector_velocity_mps, 0.0, atol=1e-9)

    def test_the_bisector_rate_is_positive_when_the_path_shortens(self) -> None:
        """Closing-positive, per spec/structure.md D5, on the *total* path."""
        trajectory = _straight_north_track(speed_mps=80.0, duration_s=60.0, n_fixes=61)
        track = to_bistatic_radar_frame(trajectory, _BISTATIC_PAIR)
        path_length_m = track.range_tx_m + track.range_rx_m
        shortening = np.diff(path_length_m) < 0.0
        assert np.all(track.bisector_velocity_mps[1:-1][shortening[1:]] > 0.0)

    def test_n_frames_counts_the_grid(self) -> None:
        trajectory = _straight_north_track(speed_mps=80.0, duration_s=60.0, n_fixes=61)
        assert to_bistatic_radar_frame(trajectory, _BISTATIC_PAIR).n_frames == 61
