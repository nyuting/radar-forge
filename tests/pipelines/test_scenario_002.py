"""Acceptance tests for scenario 002: the bistatic range-Doppler peak against truth.

The criteria of ``spec/scenario-002-bistatic.md`` §6. The first of
them is the one that matters most: collapse the baseline to a metre and the
bistatic pipeline must reproduce the monostatic scenario-001 map it replaces.
Every factor of two and every sign in the delay, the propagation phase and the
Doppler is pinned by that single comparison, against code that was already
trusted before this slice existed.

The rest assert what the geometry does once the baseline is real: the range axis
carries the bistatic mean range rather than a distance to either site, the
velocity axis carries the bisector rate, and a given separation in space maps to
a *smaller* separation in range than it would monostatically.

Marked slow: it synthesises and processes real IQ cubes. A five-frame window
keeps ``make check`` quick while still running every frame of it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from radar_forge.core.ambiguity import fold_velocity_mps
from radar_forge.core.constants import WGS84_SEMI_MAJOR_AXIS_M
from radar_forge.core.geodesy import geodetic_to_enu_m
from radar_forge.core.radar import BistaticRadar, Radar
from radar_forge.pipelines.scenarios import (
    Scenario,
    form_range_doppler_map,
    iterate_frames,
    load_scenario,
    peak_range_velocity,
)

pytestmark = pytest.mark.slow

SCENARIOS_DIR = Path(__file__).parent.parent.parent / "scenarios"
B1_TOML = SCENARIOS_DIR / "scenario_002_bistatic_xband.toml"
B2_TOML = SCENARIOS_DIR / "scenario_002_bistatic_sband.toml"
S1_TOML = SCENARIOS_DIR / "scenario_001_fmcw_low_prf.toml"

N_FRAMES = 5


def _short(toml_path: Path, start_time_s: float | None = None) -> Scenario:
    """The shipped scenario, cut to five frames at its own default window."""
    scenario = load_scenario(toml_path)
    return replace(
        scenario,
        start_time_s=scenario.start_time_s if start_time_s is None else start_time_s,
        duration_s=N_FRAMES / scenario.frame_rate_hz,
    )


def _velocity_bin_mps(burst: Radar | BistaticRadar, n_doppler_bins: int) -> float:
    """Width of one Doppler bin, m/s."""
    return 2.0 * burst.unambiguous_velocity_mps / n_doppler_bins


def _circular_error(measured: float, expected: float, half_interval: float) -> float:
    """Distance on the folded axis, so a value near each edge is not 'far'."""
    return abs(float(fold_velocity_mps(measured - expected, half_interval)))


class TestTheBaselineCollapsesToScenario001:
    """The load-bearing test: no baseline, no difference."""

    @staticmethod
    def _collapsed(monostatic: Scenario) -> Scenario:
        """Scenario 001 S1, re-sited as a bistatic pair one metre across.

        The sites cannot be made identical -- BistaticRadar rejects a zero
        baseline, and rightly, since the bistatic angle is undefined there -- so
        they are put one metre apart. That is 1/75 of a range bin.
        """
        bursts = tuple(
            BistaticRadar(
                transmitter=burst.transmitter,
                receiver=burst.receiver,
                transmitter_latitude_deg=burst.latitude_deg,
                transmitter_longitude_deg=burst.longitude_deg,
                transmitter_altitude_m=burst.altitude_m,
                receiver_latitude_deg=burst.latitude_deg,
                receiver_longitude_deg=burst.longitude_deg,
                receiver_altitude_m=burst.altitude_m + 1.0,
            )
            for burst in monostatic.bursts
            if isinstance(burst, Radar)
        )
        return replace(monostatic, bursts=bursts)

    def test_the_range_doppler_peak_agrees(self) -> None:
        """The whole bistatic chain against the monostatic one it generalises."""
        monostatic = _short(S1_TOML)
        bistatic = self._collapsed(monostatic)
        mono_burst = monostatic.bursts[0]
        bi_burst = bistatic.bursts[0]
        for mono_frame, bi_frame in zip(
            iterate_frames(monostatic), iterate_frames(bistatic), strict=True
        ):
            mono_range_m, mono_velocity_mps = peak_range_velocity(
                form_range_doppler_map(mono_frame.iq[0], mono_burst)
            )
            bi_range_m, bi_velocity_mps = peak_range_velocity(
                form_range_doppler_map(bi_frame.iq[0], bi_burst)
            )
            # Same bin, not merely a similar number.
            assert bi_range_m == pytest.approx(mono_range_m, abs=1e-6)
            assert bi_velocity_mps == pytest.approx(mono_velocity_mps, abs=1e-6)


class TestTheRangeAxisCarriesTheMeanRange:
    """Not R_t, not R_r, and not their sum: their half-sum. Spec D6."""

    @pytest.mark.parametrize("toml_path", [B1_TOML, B2_TOML], ids=["b1-xband", "b2-sband"])
    def test_the_peak_lands_at_the_bistatic_mean_range(self, toml_path: Path) -> None:
        """A2, on both variants as the spec's matrix asks.

        The range axis is the same for both, but the FMCW beat frequency also
        carries the Doppler shift, and that differs by the carrier ratio of 3.5.
        """
        scenario = _short(toml_path)
        burst = scenario.bursts[0]
        for frame in iterate_frames(scenario):
            product = form_range_doppler_map(frame.iq[0], burst)
            peak_range_m, _ = peak_range_velocity(product)
            # Range does not fold here: the sum never exceeds 53.9 km against a
            # 74.95 km limit, so this is the true value to within one bin.
            assert abs(peak_range_m - frame.range_m) < burst.range_resolution_m

    def test_the_mean_range_is_neither_of_the_two_ranges(self) -> None:
        """Guards the premise: over this window the geometry is truly lopsided.

        If R_t and R_r happened to be equal the test above would pass against a
        pipeline that had silently used either one of them.
        """
        scenario = _short(B1_TOML)
        for frame in iterate_frames(scenario):
            assert frame.range_tx_m is not None
            assert frame.range_rx_m is not None
            assert abs(frame.range_tx_m - frame.range_rx_m) > 400.0
            assert frame.range_m == pytest.approx(
                (frame.range_tx_m + frame.range_rx_m) / 2.0, rel=1e-12
            )

    def test_the_window_is_genuinely_bistatic(self) -> None:
        """A6: beta stays inside the 109.9-129.3 deg the spec quotes for the default window.

        Catches a bistatic angle measured at the wrong vertex or taken as its
        supplement (50.7-70.1 deg here), either of which would also defeat A4.
        """
        scenario = _short(B1_TOML)
        for frame in iterate_frames(scenario):
            assert frame.bistatic_angle_deg is not None
            assert 109.9 <= frame.bistatic_angle_deg <= 129.3


class TestTheVelocityAxisCarriesTheBisectorRate:
    """Both variants fold; how much is set by the carrier alone."""

    @pytest.mark.parametrize("toml_path", [B1_TOML, B2_TOML], ids=["b1-xband", "b2-sband"])
    def test_the_peak_lands_at_the_folded_bisector_rate(self, toml_path: Path) -> None:
        scenario = _short(toml_path)
        burst = scenario.bursts[0]
        for frame in iterate_frames(scenario):
            product = form_range_doppler_map(frame.iq[0], burst)
            _, peak_velocity_mps = peak_range_velocity(product)
            expected_mps = float(
                fold_velocity_mps(frame.radial_velocity_mps, burst.unambiguous_velocity_mps)
            )
            error_mps = _circular_error(
                peak_velocity_mps, expected_mps, burst.unambiguous_velocity_mps
            )
            assert error_mps <= _velocity_bin_mps(burst, product.rd_map.shape[0])

    @staticmethod
    def _n_folding_frames(toml_path: Path) -> int:
        scenario = _short(toml_path)
        burst = scenario.bursts[0]
        return sum(
            abs(frame.radial_velocity_mps) > burst.unambiguous_velocity_mps
            for frame in iterate_frames(scenario)
        )

    def test_the_s_band_variant_folds_in_fewer_frames(self) -> None:
        """The pair's whole lesson, as a number rather than as prose.

        The same trajectory and the same geometry, processed the same way: at
        X-band every frame folds, and at S-band some of them do not fold at all.
        The bisector rate reaches 33.6 m/s over these five frames, which is
        4.4 of X-band's unambiguous intervals but only 1.3 of S-band's.

        Also the premise for A3 on both variants: ``0 < n_s < n_x`` means each
        folds in at least one frame, so neither A3 case runs against nothing.
        """
        n_x_band = self._n_folding_frames(B1_TOML)
        n_s_band = self._n_folding_frames(B2_TOML)
        assert 0 < n_s_band < n_x_band


class TestRangeResolutionDegradesWithTheBistaticAngle:
    """Spec §6 criterion A4: a mapping from space to range, not a peak width."""

    def test_a_separation_in_space_shrinks_in_mean_range(self) -> None:
        r"""Two targets `d` apart along the bisector are `d\cos(\beta/2)` apart in range.

        This is the sense in which bistatic range resolution degrades: the
        iso-range surfaces are ellipsoids with the sites at the foci, so
        resolving two targets needs 1/cos(beta/2) times the spatial separation a
        monostatic radar would need. It is *not* a widening of the range peak --
        a point target has one delay, and its peak width is set by the window
        and the transform length whatever the geometry.
        """
        pair = _short(B1_TOML).bursts[0]
        assert isinstance(pair, BistaticRadar)
        # The track's own position at the start of B1's default window (13 329 s),
        # 14.8 km from the transmitter and 7.5 km from the receiver, where
        # beta = 119.2 deg and cos(beta/2) = 0.506. The spec's 1.07x-2.34x range
        # puts the answer far from 1, so a monostatic mapping cannot pass.
        latitude_deg, longitude_deg, altitude_m = 35.9362, -78.9338, 1500.0

        range_tx_m, range_rx_m = pair.target_ranges_m(latitude_deg, longitude_deg, altitude_m)
        bistatic_angle_rad = float(pair.bistatic_angle_rad(range_tx_m, range_rx_m))
        assert np.degrees(bistatic_angle_rad) > 109.9

        # The bisector at the target: the mean of the unit vectors towards the
        # two sites, in the target's own local tangent plane.
        to_transmitter = geodetic_to_enu_m(
            pair.transmitter_latitude_deg,
            pair.transmitter_longitude_deg,
            pair.transmitter_altitude_m,
            latitude_deg,
            longitude_deg,
            altitude_m,
        )
        to_receiver = geodetic_to_enu_m(
            pair.receiver_latitude_deg,
            pair.receiver_longitude_deg,
            pair.receiver_altitude_m,
            latitude_deg,
            longitude_deg,
            altitude_m,
        )
        bisector = to_transmitter / np.linalg.norm(to_transmitter) + to_receiver / np.linalg.norm(
            to_receiver
        )
        bisector /= np.linalg.norm(bisector)

        # Step the second target away from both sites along that bisector.
        # d cos(beta/2) is the first-order term; the iso-range ellipsoid's
        # curvature adds a second-order one that grows linearly in d/R, measured
        # at 7.4e-5 per metre of step here (1.5e-3 at 20 m, 1.5e-2 at 200 m).
        # The flat-earth conversion back to geodetic, on the semi-major axis
        # rather than the two radii of curvature, adds 3e-5. So 20 m and
        # rtol = 2e-3; a mapping with cos(beta) in place of cos(beta/2), or
        # none at all, misses by more than 50 %.
        separation_m = 20.0
        offset_enu_m = -separation_m * bisector
        second_latitude_deg = latitude_deg + np.degrees(offset_enu_m[1] / WGS84_SEMI_MAJOR_AXIS_M)
        second_longitude_deg = longitude_deg + np.degrees(
            offset_enu_m[0] / (WGS84_SEMI_MAJOR_AXIS_M * np.cos(np.radians(latitude_deg)))
        )
        second_range_tx_m, second_range_rx_m = pair.target_ranges_m(
            second_latitude_deg, second_longitude_deg, altitude_m + offset_enu_m[2]
        )

        mean_range_separation_m = float(
            (second_range_tx_m + second_range_rx_m - range_tx_m - range_rx_m) / 2.0
        )
        np.testing.assert_allclose(
            mean_range_separation_m / separation_m,
            np.cos(bistatic_angle_rad / 2.0),
            rtol=2e-3,
        )
