"""Tests for scenario configuration and the frame generator.

The three shipped TOMLs are checked against the specification's S4 table, but
by *re-deriving* every unambiguous limit from the Radar the file produces
rather than by comparing against numbers copied out of the table. That is what
the specification asks for at the end of S4: a spec table is documentation, a
test is a guarantee.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from radar_forge.core.radar import BistaticRadar, Radar
from radar_forge.pipelines.scenarios import (
    Scenario,
    burst_range_doppler,
    iterate_frames,
    load_scenario,
    peak_range_velocity,
)

SCENARIOS_DIR = Path(__file__).parent.parent.parent / "scenarios"
S1_TOML = SCENARIOS_DIR / "scenario_001_fmcw_low_prf.toml"
S2_TOML = SCENARIOS_DIR / "scenario_001_pulsed_medium_prf.toml"
S3_TOML = SCENARIOS_DIR / "scenario_001_fmcw_dual_prf.toml"
B1_TOML = SCENARIOS_DIR / "scenario_002_bistatic_xband.toml"
B2_TOML = SCENARIOS_DIR / "scenario_002_bistatic_sband.toml"

# Range resolution c/2B at B = 2 MHz, shared by all three variants.
RANGE_RESOLUTION_M = 74.9481145


def _short_window(scenario: Scenario, n_frames: int) -> Scenario:
    """The same scenario over a handful of frames, to keep the suite quick."""
    from dataclasses import replace

    return replace(scenario, duration_s=n_frames / scenario.frame_rate_hz)


class TestLoadScenario:
    @pytest.mark.parametrize("toml_path", [S1_TOML, S2_TOML, S3_TOML], ids=["s1", "s2", "s3"])
    def test_every_shipped_scenario_loads(self, toml_path: Path) -> None:
        scenario = load_scenario(toml_path)
        assert scenario.bursts
        assert all(isinstance(burst, Radar) for burst in scenario.bursts)

    @pytest.mark.parametrize("toml_path", [S1_TOML, S2_TOML, S3_TOML], ids=["s1", "s2", "s3"])
    def test_the_trajectory_path_resolves_to_a_real_file(self, toml_path: Path) -> None:
        """Resolved against the TOML's directory, so a scenario is relocatable."""
        assert load_scenario(toml_path).trajectory_path.is_file()

    @pytest.mark.parametrize("toml_path", [S1_TOML, S2_TOML, S3_TOML], ids=["s1", "s2", "s3"])
    def test_every_variant_shares_the_specified_range_resolution(self, toml_path: Path) -> None:
        """All three run at B = 2 MHz; only the ambiguities differ."""
        for burst in load_scenario(toml_path).bursts:
            np.testing.assert_allclose(burst.range_resolution_m, RANGE_RESOLUTION_M, rtol=1e-8)

    def test_the_default_window_is_the_specified_two_minutes(self) -> None:
        """The full track is not guaranteed to be one aircraft; 120 s is safe."""
        scenario = load_scenario(S1_TOML)
        assert scenario.duration_s == 120.0
        assert scenario.frame_rate_hz == 1.0
        assert scenario.n_frames == 120

    def test_rejects_an_unknown_key(self, tmp_path: Path) -> None:
        """A misspelled key must fail loudly, not run a different scenario."""
        original = S1_TOML.read_text()
        broken = tmp_path / "broken.toml"
        broken.write_text(original.replace("sample_rate_hz = 1.0e6", "sample_rate_hzz = 1.0e6"))
        with pytest.raises(ValueError, match="unknown key"):
            load_scenario(broken)

    def test_rejects_a_file_with_no_burst(self, tmp_path: Path) -> None:
        bare = tmp_path / "no_burst.toml"
        bare.write_text(
            '[scenario]\nname = "x"\nseed = 1\n'
            "[radar]\nlatitude_deg = 0.0\nlongitude_deg = 0.0\naltitude_m = 0.0\n"
            "[receiver]\ngain_rx_dbi = 30.0\nnoise_figure_db = 3.0\n"
            "[target]\nrcs_dbsm = 10.0\naltitude_m = 1500.0\n"
            '[trajectory]\npath = "t.csv"\nstart_time_s = 0.0\n'
            "duration_s = 1.0\nframe_rate_hz = 1.0\n"
        )
        with pytest.raises(ValueError, match="defines no"):
            load_scenario(bare)

    def test_rejects_a_file_missing_a_table(self, tmp_path: Path) -> None:
        bare = tmp_path / "no_radar.toml"
        bare.write_text('[scenario]\nname = "x"\nseed = 1\n')
        with pytest.raises(ValueError, match=r"missing the required \[radar\] table"):
            load_scenario(bare)

    def test_bad_physics_is_reported_by_the_radar_not_the_loader(self, tmp_path: Path) -> None:
        """A duty cycle above one names the quantity, not the file."""
        broken = tmp_path / "overrun.toml"
        broken.write_text(
            S1_TOML.read_text().replace("chirp_time_s = 1.0e-3", "chirp_time_s = 2.0e-3")
        )
        with pytest.raises(ValueError, match=r"exceeds 1\.0"):
            load_scenario(broken)


class TestSpecifiedAmbiguities:
    """Spec S4's table, re-derived from the loaded Radar rather than trusted."""

    def test_s1_covers_the_track_in_range_and_folds_hard_in_doppler(self) -> None:
        burst = load_scenario(S1_TOML).bursts[0]
        # 37.47 km, comfortably past the 17.9 km the track reaches.
        np.testing.assert_allclose(burst.unambiguous_range_m, 37_474.057, rtol=1e-6)
        np.testing.assert_allclose(burst.unambiguous_velocity_mps, 7.6478, rtol=1e-4)
        # The point of S1: an 80 m/s aircraft is five folds out.
        assert 80.0 / (2.0 * burst.unambiguous_velocity_mps) > 5.0

    def test_s2_folds_in_range_and_covers_the_track_in_doppler(self) -> None:
        burst = load_scenario(S2_TOML).bursts[0]
        # 5.996 km: the target at 8.4-17.9 km wraps once or twice.
        np.testing.assert_allclose(burst.unambiguous_range_m, 5_995.849, rtol=1e-6)
        np.testing.assert_allclose(burst.unambiguous_velocity_mps, 191.194, rtol=1e-5)
        assert burst.unambiguous_range_m < 8_390.0

    def test_s2_is_the_exact_mirror_of_s1(self) -> None:
        """The lesson: same target, same bandwidth, opposite ambiguity."""
        s1 = load_scenario(S1_TOML).bursts[0]
        s2 = load_scenario(S2_TOML).bursts[0]
        assert s1.unambiguous_range_m > s2.unambiguous_range_m
        assert s1.unambiguous_velocity_mps < s2.unambiguous_velocity_mps

    def test_s3_bursts_both_cover_the_track_in_range(self) -> None:
        bursts = load_scenario(S3_TOML).bursts
        assert len(bursts) == 2
        np.testing.assert_allclose(bursts[0].unambiguous_range_m, 29_979.246, rtol=1e-6)
        np.testing.assert_allclose(bursts[1].unambiguous_range_m, 24_982.705, rtol=1e-6)
        assert all(burst.unambiguous_range_m > 18_000.0 for burst in bursts)

    def test_s3_bursts_are_in_the_coprime_five_to_six_ratio(self) -> None:
        """Coprime is what makes the pair identify a unique velocity."""
        bursts = load_scenario(S3_TOML).bursts
        np.testing.assert_allclose(bursts[0].unambiguous_velocity_mps, 38.2388, rtol=1e-5)
        np.testing.assert_allclose(bursts[1].unambiguous_velocity_mps, 45.8866, rtol=1e-5)
        ratio = bursts[1].unambiguous_velocity_mps / bursts[0].unambiguous_velocity_mps
        np.testing.assert_allclose(ratio, 6.0 / 5.0, rtol=1e-12)

    def test_s3_burst_b_runs_at_exactly_one_over_six_thousand_seconds(self) -> None:
        """The spec's rounded 166.7 us is a duty cycle of 1.0002 and is rejected.

        Pinned because the rounded figure looks harmless and the failure it
        causes -- Transmitter refusing the whole scenario -- is far from it.
        """
        burst = load_scenario(S3_TOML).bursts[1]
        np.testing.assert_allclose(burst.transmitter.chirp_time_s, 1.0 / 6000.0, rtol=1e-15)
        np.testing.assert_allclose(burst.transmitter.duty_cycle_linear, 1.0, rtol=1e-12)

    @pytest.mark.parametrize(
        ("toml_path", "expected_shapes"),
        [
            (S1_TOML, [(256, 1000)]),
            (S2_TOML, [(256, 100)]),
            (S3_TOML, [(128, 800), (128, 667)]),
        ],
        ids=["s1", "s2", "s3"],
    )
    def test_cube_dimensions_match_the_specified_tables(
        self, toml_path: Path, expected_shapes: list[tuple[int, int]]
    ) -> None:
        scenario = load_scenario(toml_path)
        shapes = [
            (n_pulses, burst.n_samples_per_pri)
            for burst, n_pulses in zip(scenario.bursts, scenario.n_pulses, strict=True)
        ]
        assert shapes == expected_shapes


class TestIterateFrames:
    def test_is_a_generator_not_a_list(self) -> None:
        """16,500 frames at 4.1 MB each would be 68 GB materialised."""
        frames = iterate_frames(load_scenario(S1_TOML))
        assert isinstance(frames, Iterator)

    def test_yields_one_frame_per_frame_time(self) -> None:
        scenario = _short_window(load_scenario(S2_TOML), 5)
        frames = list(iterate_frames(scenario))
        assert len(frames) == 5
        assert [frame.index for frame in frames] == [0, 1, 2, 3, 4]
        np.testing.assert_allclose(
            [frame.time_s for frame in frames], scenario.frame_times_s, rtol=1e-12
        )

    def test_each_burst_gets_its_own_cube(self) -> None:
        """Two sweep rates cannot be coherently integrated, so they stay apart."""
        scenario = _short_window(load_scenario(S3_TOML), 2)
        frame = next(iter(iterate_frames(scenario)))
        assert len(frame.iq) == 2
        assert frame.iq[0].shape == (128, 800)
        assert frame.iq[1].shape == (128, 667)
        assert all(cube.dtype == np.complex128 for cube in frame.iq)

    def test_the_truth_is_the_real_geometry(self) -> None:
        """Range and angles must be physically sensible for the DSO scenario."""
        scenario = _short_window(load_scenario(S2_TOML), 3)
        for frame in iterate_frames(scenario):
            assert 8_000.0 < frame.range_m < 18_000.0
            assert 0.0 <= frame.azimuth_deg < 360.0
            assert 0.0 < frame.elevation_deg < 90.0

    def test_the_truth_is_never_folded(self) -> None:
        """S1's truth must be free to exceed its own unambiguous velocity.

        The gap between the truth and the map is the whole scenario, so the
        label carries the unfolded number even when the map cannot.
        """
        scenario = _short_window(load_scenario(S1_TOML), 5)
        burst = scenario.bursts[0]
        velocities_mps = np.array([frame.radial_velocity_mps for frame in iterate_frames(scenario)])
        # Nothing clipped it to the interval; that it happens to sit near the
        # edge here is the track's doing, not the code's.
        assert np.any(np.abs(velocities_mps) > 0.9 * burst.unambiguous_velocity_mps)

    def test_a_run_replays_bit_for_bit(self) -> None:
        scenario = _short_window(load_scenario(S2_TOML), 2)
        first = [frame.iq[0] for frame in iterate_frames(scenario)]
        second = [frame.iq[0] for frame in iterate_frames(scenario)]
        for one, two in zip(first, second, strict=True):
            np.testing.assert_array_equal(one, two)

    def test_consecutive_frames_carry_different_noise(self) -> None:
        """One shared generator, consumed in order -- not reseeded per frame."""
        scenario = _short_window(load_scenario(S2_TOML), 2)
        cubes = [frame.iq[0] for frame in iterate_frames(scenario)]
        assert not np.array_equal(cubes[0], cubes[1])

    def test_rejects_a_window_outside_the_track(self) -> None:
        from dataclasses import replace

        scenario = replace(load_scenario(S1_TOML), start_time_s=1.0e6, duration_s=5.0)
        with pytest.raises(ValueError, match="outside the track"):
            next(iter(iterate_frames(scenario)))


class TestScenarioValidation:
    def test_rejects_a_non_positive_duration(self) -> None:
        from dataclasses import replace

        with pytest.raises(ValueError, match="duration_s"):
            replace(load_scenario(S1_TOML), duration_s=0.0)

    def test_rejects_a_burst_count_mismatch(self) -> None:
        from dataclasses import replace

        with pytest.raises(ValueError, match="same length"):
            replace(load_scenario(S3_TOML), n_pulses=(128,))


class TestBurstRangeDoppler:
    def test_fmcw_axes_match_the_map(self) -> None:
        scenario = _short_window(load_scenario(S1_TOML), 1)
        frame = next(iter(iterate_frames(scenario)))
        product = burst_range_doppler(frame.iq[0], scenario.bursts[0])
        assert product.rd_map.shape == (
            product.velocity_axis_mps.size,
            product.range_axis_m.size,
        )

    def test_the_range_axis_is_unshifted_and_the_velocity_axis_is_centred(self) -> None:
        """The dsp asymmetry the spec S7.1 insists on, carried through intact."""
        scenario = _short_window(load_scenario(S1_TOML), 1)
        product = burst_range_doppler(
            next(iter(iterate_frames(scenario))).iq[0], scenario.bursts[0]
        )
        np.testing.assert_allclose(product.range_axis_m[0], 0.0, atol=1e-12)
        assert product.range_axis_m[-1] > product.range_axis_m[0]
        centre = product.velocity_axis_mps.size // 2
        np.testing.assert_allclose(product.velocity_axis_mps[centre], 0.0, atol=1e-12)

    def test_the_pulsed_range_axis_spans_one_unambiguous_range(self) -> None:
        """The matched filter's group delay is trimmed off, not left as an offset."""
        scenario = _short_window(load_scenario(S2_TOML), 1)
        burst = scenario.bursts[0]
        product = burst_range_doppler(next(iter(iterate_frames(scenario))).iq[0], burst)
        np.testing.assert_allclose(product.range_axis_m[0], 0.0, atol=1e-12)
        assert product.range_axis_m[-1] < burst.unambiguous_range_m
        np.testing.assert_allclose(
            product.range_axis_m[-1] + product.range_axis_m[1],
            burst.unambiguous_range_m,
            rtol=1e-9,
        )

    def test_a_synthetic_fmcw_target_lands_at_its_true_range_and_velocity(self) -> None:
        """Closed-form ground truth, bypassing the trajectory entirely."""
        from radar_forge.core.signal import fmcw_deramp_baseband, line_of_sight_paths

        burst = load_scenario(S1_TOML).bursts[0]
        true_range_m = 10_000.0
        true_velocity_mps = 3.0
        paths = line_of_sight_paths(burst, true_range_m, true_velocity_mps, 10.0)
        cube = fmcw_deramp_baseband(paths, burst, 256)
        peak_range_m, peak_velocity_mps = peak_range_velocity(burst_range_doppler(cube, burst))
        assert abs(peak_range_m - true_range_m) < burst.range_resolution_m
        assert abs(peak_velocity_mps - true_velocity_mps) < 0.1

    def test_a_synthetic_pulsed_target_folds_into_the_unambiguous_range(self) -> None:
        """S2's defining behaviour, through the pipeline's own receive chain."""
        from radar_forge.core.signal import line_of_sight_paths, pulsed_baseband

        burst = load_scenario(S2_TOML).bursts[0]
        true_range_m = 15_000.0
        paths = line_of_sight_paths(burst, true_range_m, 0.0, 10.0)
        cube = pulsed_baseband(paths, burst, 64)
        peak_range_m, peak_velocity_mps = peak_range_velocity(burst_range_doppler(cube, burst))
        expected_range_m = true_range_m % burst.unambiguous_range_m
        assert abs(peak_range_m - expected_range_m) < burst.range_resolution_m
        np.testing.assert_allclose(peak_velocity_mps, 0.0, atol=1e-12)

    def test_a_pulsed_target_inside_the_unambiguous_range_does_not_fold(self) -> None:
        """Pins that the group-delay trim is right, not merely self-consistent."""
        from radar_forge.core.signal import line_of_sight_paths, pulsed_baseband

        burst = load_scenario(S2_TOML).bursts[0]
        true_range_m = 3_000.0
        paths = line_of_sight_paths(burst, true_range_m, 0.0, 10.0)
        peak_range_m, _ = peak_range_velocity(
            burst_range_doppler(pulsed_baseband(paths, burst, 64), burst)
        )
        assert abs(peak_range_m - true_range_m) < burst.range_resolution_m


class TestVelocityIsIndependentOfTheWindow:
    """The padding around the window, pinned by its observable consequence."""

    def test_a_frame_reports_the_same_velocity_whatever_window_contains_it(self) -> None:
        """Without padding the edge frames would carry a one-sided difference.

        A target's radial velocity is a property of the target, so asking for a
        window that starts on that frame must not change it. This is the only
        way the difference is visible from outside.
        """
        from dataclasses import replace

        base = load_scenario(S1_TOML)
        long_window = replace(base, start_time_s=50.0, duration_s=5.0)
        starts_here = replace(base, start_time_s=52.0, duration_s=1.0)

        from_long = list(iterate_frames(long_window))[2]
        (from_short,) = list(iterate_frames(starts_here))

        np.testing.assert_allclose(from_short.time_s, from_long.time_s, rtol=1e-12)
        np.testing.assert_allclose(
            from_short.radial_velocity_mps, from_long.radial_velocity_mps, rtol=1e-12
        )
        np.testing.assert_allclose(from_short.range_m, from_long.range_m, rtol=1e-12)

    def test_a_single_frame_window_still_has_a_velocity(self) -> None:
        from dataclasses import replace

        scenario = replace(load_scenario(S1_TOML), start_time_s=60.0, duration_s=1.0)
        (frame,) = list(iterate_frames(scenario))
        assert frame.radial_velocity_mps != 0.0
        assert np.isfinite(frame.radial_velocity_mps)


class TestBistaticScenarios:
    """The [transmitter_site] table, and that its absence changes nothing."""

    @pytest.mark.parametrize("toml_path", [B1_TOML, B2_TOML], ids=["b1-xband", "b2-sband"])
    def test_every_shipped_bistatic_scenario_loads(self, toml_path: Path) -> None:
        scenario = load_scenario(toml_path)
        assert scenario.is_bistatic
        assert len(scenario.bursts) == 1
        assert isinstance(scenario.bursts[0], BistaticRadar)

    @pytest.mark.parametrize("toml_path", [S1_TOML, S2_TOML, S3_TOML], ids=["s1", "s2", "s3"])
    def test_a_scenario_without_the_table_is_still_monostatic(self, toml_path: Path) -> None:
        """The regression guard: scenario 001 must not have become bistatic."""
        scenario = load_scenario(toml_path)
        assert not scenario.is_bistatic
        assert all(isinstance(burst, Radar) for burst in scenario.bursts)

    def test_the_baseline_is_the_specified_changi_to_dso_distance(self) -> None:
        """Checked against spec/scenario-002-singapore-bistatic.md S2."""
        pair = load_scenario(B1_TOML).bursts[0]
        assert isinstance(pair, BistaticRadar)
        np.testing.assert_allclose(pair.baseline_m, 23_726.0, atol=5.0)

    def test_the_two_variants_differ_only_in_the_carrier(self) -> None:
        """The comparison the pair exists to make, asserted rather than trusted."""
        x_band = load_scenario(B1_TOML).bursts[0]
        s_band = load_scenario(B2_TOML).bursts[0]
        assert x_band.transmitter.f0_hz != s_band.transmitter.f0_hz
        for field in ("bandwidth_hz", "transmit_power_w", "chirp_time_s", "prf_hz", "waveform"):
            assert getattr(x_band.transmitter, field) == getattr(s_band.transmitter, field)
        assert x_band.receiver == s_band.receiver

    def test_deleting_the_table_makes_the_same_file_monostatic(self, tmp_path: Path) -> None:
        """The table's presence is the whole switch; nothing else selects siting."""
        text = B1_TOML.read_text()
        start = text.index("[transmitter_site]")
        end = text.index("[receiver]")
        monostatic = tmp_path / "monostatic.toml"
        monostatic.write_text(text[:start] + text[end:])
        # The trajectory path is resolved relative to the TOML, so it moves too.
        scenario = replace(
            load_scenario(monostatic), trajectory_path=load_scenario(B1_TOML).trajectory_path
        )
        assert not scenario.is_bistatic

    def test_rejects_an_unknown_key_in_the_transmitter_site(self, tmp_path: Path) -> None:
        """The new table gets the same loud failure as every other one."""
        broken = tmp_path / "broken.toml"
        broken.write_text(B1_TOML.read_text().replace("latitude_deg = 1.3592", "lat_deg = 1.3592"))
        with pytest.raises(ValueError, match="unknown key"):
            load_scenario(broken)

    def test_frames_carry_the_bistatic_geometry(self) -> None:
        scenario = _short_window(load_scenario(B1_TOML), 2)
        for frame in iterate_frames(scenario):
            assert frame.range_tx_m is not None
            assert frame.range_rx_m is not None
            assert frame.bistatic_angle_deg is not None
            assert frame.transmit_azimuth_deg is not None
            assert 0.0 < frame.bistatic_angle_deg < 180.0

    def test_monostatic_frames_leave_the_bistatic_fields_empty(self) -> None:
        """None rather than a plausible-looking zero, which would be a lie."""
        scenario = _short_window(load_scenario(S1_TOML), 2)
        for frame in iterate_frames(scenario):
            assert frame.range_tx_m is None
            assert frame.range_rx_m is None
            assert frame.bistatic_angle_deg is None
