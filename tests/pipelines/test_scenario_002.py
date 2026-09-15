"""Acceptance tests for scenario 002: the bistatic range-Doppler peak against truth.

The criteria of ``spec/scenario-002-singapore-bistatic.md`` §7. The first of
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
    burst_range_doppler,
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

    def test_the_truth_labels_agree(self) -> None:
        """Before comparing maps, check the geometry they are drawn from."""
        monostatic = _short(S1_TOML)
        bistatic = self._collapsed(monostatic)
        for mono_frame, bi_frame in zip(
            iterate_frames(monostatic), iterate_frames(bistatic), strict=True
        ):
            # Half a metre of baseline, against ranges of 8-18 km.
            assert abs(bi_frame.range_m - mono_frame.range_m) < 1.0
            assert abs(bi_frame.radial_velocity_mps - mono_frame.radial_velocity_mps) < 1e-3
            assert bi_frame.bistatic_angle_deg is not None
            assert bi_frame.bistatic_angle_deg < 0.01

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
                burst_range_doppler(mono_frame.iq[0], mono_burst)
            )
            bi_range_m, bi_velocity_mps = peak_range_velocity(
                burst_range_doppler(bi_frame.iq[0], bi_burst)
            )
            # Same bin, not merely a similar number.
            assert bi_range_m == pytest.approx(mono_range_m, abs=1e-6)
            assert bi_velocity_mps == pytest.approx(mono_velocity_mps, abs=1e-6)


class TestTheRangeAxisCarriesTheMeanRange:
    """Not R_t, not R_r, and not their sum: their half-sum. Spec D6."""

    @pytest.mark.parametrize("toml_path", [B1_TOML, B2_TOML], ids=["b1-xband", "b2-sband"])
    def test_the_peak_lands_at_the_bistatic_mean_range(self, toml_path: Path) -> None:
        scenario = _short(toml_path)
        burst = scenario.bursts[0]
        for frame in iterate_frames(scenario):
            product = burst_range_doppler(frame.iq[0], burst)
            peak_range_m, _ = peak_range_velocity(product)
            # Range does not fold here: the sum never exceeds 53.9 km against a
            # 74.95 km limit, so this is the true value to within one bin.
            assert abs(peak_range_m - frame.range_m) < burst.range_resolution_m

    @pytest.mark.parametrize("toml_path", [B1_TOML, B2_TOML], ids=["b1-xband", "b2-sband"])
    def test_the_mean_range_is_neither_of_the_two_ranges(self, toml_path: Path) -> None:
        """Guards the premise: over this window the geometry is truly lopsided.

        If R_t and R_r happened to be equal the test above would pass against a
        pipeline that had silently used either one of them.
        """
        scenario = _short(toml_path)
        for frame in iterate_frames(scenario):
            assert frame.range_tx_m is not None
            assert frame.range_rx_m is not None
            assert abs(frame.range_tx_m - frame.range_rx_m) > 400.0
            assert frame.range_m == pytest.approx(
                (frame.range_tx_m + frame.range_rx_m) / 2.0, rel=1e-12
            )

    def test_the_window_is_genuinely_bistatic(self) -> None:
        """The scenario is worthless if the geometry is nearly monostatic."""
        scenario = _short(B1_TOML)
        for frame in iterate_frames(scenario):
            assert frame.bistatic_angle_deg is not None
            assert frame.bistatic_angle_deg > 90.0


class TestTheVelocityAxisCarriesTheBisectorRate:
    """Both variants fold; how much is set by the carrier alone."""

    @pytest.mark.parametrize("toml_path", [B1_TOML, B2_TOML], ids=["b1-xband", "b2-sband"])
    def test_the_peak_lands_at_the_folded_bisector_rate(self, toml_path: Path) -> None:
        scenario = _short(toml_path)
        burst = scenario.bursts[0]
        for frame in iterate_frames(scenario):
            product = burst_range_doppler(frame.iq[0], burst)
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

    def test_the_x_band_variant_folds_in_every_frame(self) -> None:
        """Otherwise the folding asserted above is asserted against nothing."""
        assert self._n_folding_frames(B1_TOML) == N_FRAMES

    def test_the_s_band_variant_folds_in_fewer_frames(self) -> None:
        """The pair's whole lesson, as a number rather than as prose.

        The same trajectory and the same geometry, processed the same way: at
        X-band every frame folds, and at S-band some of them do not fold at all.
        The bisector rate reaches 43.9 m/s here, which is 5.7 of X-band's
        unambiguous intervals but only 1.6 of S-band's.
        """
        n_x_band = self._n_folding_frames(B1_TOML)
        n_s_band = self._n_folding_frames(B2_TOML)
        assert 0 < n_s_band < n_x_band

    def test_the_longer_wavelength_buys_unambiguous_velocity(self) -> None:
        """The lesson of the pair: same geometry, same processing, one carrier."""
        x_band = _short(B1_TOML).bursts[0]
        s_band = _short(B2_TOML).bursts[0]
        assert s_band.unambiguous_velocity_mps > 3.0 * x_band.unambiguous_velocity_mps


class TestRangeResolutionDegradesWithTheBistaticAngle:
    """Spec §7 criterion 4: a mapping from space to range, not a peak width."""

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
        latitude_deg, longitude_deg, altitude_m = 1.40, 103.88, 1500.0

        range_tx_m, range_rx_m = pair.target_ranges_m(latitude_deg, longitude_deg, altitude_m)
        bistatic_angle_rad = float(pair.bistatic_angle_rad(range_tx_m, range_rx_m))

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

        # Step the second target away from both sites along that bisector. The
        # offset is small enough that a flat-earth conversion back to geodetic
        # is good to well under the 0.5 % tolerance asserted below.
        separation_m = 200.0
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
            rtol=5e-3,
        )
        # And the consequence: the pair is closer together in range than in
        # space, so a monostatic radar would resolve them and this one may not.
        assert mean_range_separation_m < separation_m
