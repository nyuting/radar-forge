"""Tests for scenario configuration and the frame generator.

The three shipped TOMLs are checked against the specification's S4 table, but
by *re-deriving* every unambiguous limit from the Radar the file produces
rather than by comparing against numbers copied out of the table. That is what
the specification asks for at the end of S4: a spec table is documentation, a
test is a guarantee.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from radar_forge.core.radar import Radar
from radar_forge.pipelines.scenarios import Scenario, iterate_frames, load_scenario

SCENARIOS_DIR = Path(__file__).parent.parent.parent / "scenarios"
S1_TOML = SCENARIOS_DIR / "scenario_001_fmcw_low_prf.toml"
S2_TOML = SCENARIOS_DIR / "scenario_001_pulsed_medium_prf.toml"
S3_TOML = SCENARIOS_DIR / "scenario_001_fmcw_dual_prf.toml"

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
        assert scenario.legs
        assert all(isinstance(leg, Radar) for leg in scenario.legs)

    @pytest.mark.parametrize("toml_path", [S1_TOML, S2_TOML, S3_TOML], ids=["s1", "s2", "s3"])
    def test_the_trajectory_path_resolves_to_a_real_file(self, toml_path: Path) -> None:
        """Resolved against the TOML's directory, so a scenario is relocatable."""
        assert load_scenario(toml_path).trajectory_path.is_file()

    @pytest.mark.parametrize("toml_path", [S1_TOML, S2_TOML, S3_TOML], ids=["s1", "s2", "s3"])
    def test_every_variant_shares_the_specified_range_resolution(self, toml_path: Path) -> None:
        """All three run at B = 2 MHz; only the ambiguities differ."""
        for leg in load_scenario(toml_path).legs:
            np.testing.assert_allclose(leg.range_resolution_m, RANGE_RESOLUTION_M, rtol=1e-8)

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

    def test_rejects_a_file_with_no_leg(self, tmp_path: Path) -> None:
        bare = tmp_path / "no_leg.toml"
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
        leg = load_scenario(S1_TOML).legs[0]
        # 37.47 km, comfortably past the 17.9 km the track reaches.
        np.testing.assert_allclose(leg.unambiguous_range_m, 37_474.057, rtol=1e-6)
        np.testing.assert_allclose(leg.unambiguous_velocity_mps, 7.6478, rtol=1e-4)
        # The point of S1: an 80 m/s aircraft is five folds out.
        assert 80.0 / (2.0 * leg.unambiguous_velocity_mps) > 5.0

    def test_s2_folds_in_range_and_covers_the_track_in_doppler(self) -> None:
        leg = load_scenario(S2_TOML).legs[0]
        # 5.996 km: the target at 8.4-17.9 km wraps once or twice.
        np.testing.assert_allclose(leg.unambiguous_range_m, 5_995.849, rtol=1e-6)
        np.testing.assert_allclose(leg.unambiguous_velocity_mps, 191.194, rtol=1e-5)
        assert leg.unambiguous_range_m < 8_390.0

    def test_s2_is_the_exact_mirror_of_s1(self) -> None:
        """The lesson: same target, same bandwidth, opposite ambiguity."""
        s1 = load_scenario(S1_TOML).legs[0]
        s2 = load_scenario(S2_TOML).legs[0]
        assert s1.unambiguous_range_m > s2.unambiguous_range_m
        assert s1.unambiguous_velocity_mps < s2.unambiguous_velocity_mps

    def test_s3_legs_both_cover_the_track_in_range(self) -> None:
        legs = load_scenario(S3_TOML).legs
        assert len(legs) == 2
        np.testing.assert_allclose(legs[0].unambiguous_range_m, 29_979.246, rtol=1e-6)
        np.testing.assert_allclose(legs[1].unambiguous_range_m, 24_982.705, rtol=1e-6)
        assert all(leg.unambiguous_range_m > 18_000.0 for leg in legs)

    def test_s3_legs_are_in_the_coprime_five_to_six_ratio(self) -> None:
        """Coprime is what makes the pair identify a unique velocity."""
        legs = load_scenario(S3_TOML).legs
        np.testing.assert_allclose(legs[0].unambiguous_velocity_mps, 38.2388, rtol=1e-5)
        np.testing.assert_allclose(legs[1].unambiguous_velocity_mps, 45.8866, rtol=1e-5)
        ratio = legs[1].unambiguous_velocity_mps / legs[0].unambiguous_velocity_mps
        np.testing.assert_allclose(ratio, 6.0 / 5.0, rtol=1e-12)

    def test_s3_leg_b_runs_at_exactly_one_over_six_thousand_seconds(self) -> None:
        """The spec's rounded 166.7 us is a duty cycle of 1.0002 and is rejected.

        Pinned because the rounded figure looks harmless and the failure it
        causes -- Transmitter refusing the whole scenario -- is far from it.
        """
        leg = load_scenario(S3_TOML).legs[1]
        np.testing.assert_allclose(leg.transmitter.chirp_time_s, 1.0 / 6000.0, rtol=1e-15)
        np.testing.assert_allclose(leg.transmitter.duty_cycle_linear, 1.0, rtol=1e-12)

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
            (n_chirps, leg.n_samples_per_pri)
            for leg, n_chirps in zip(scenario.legs, scenario.n_chirps, strict=True)
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

    def test_each_leg_gets_its_own_cube(self) -> None:
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
        leg = scenario.legs[0]
        velocities_mps = np.array([frame.radial_velocity_mps for frame in iterate_frames(scenario)])
        # Nothing clipped it to the interval; that it happens to sit near the
        # edge here is the track's doing, not the code's.
        assert np.any(np.abs(velocities_mps) > 0.9 * leg.unambiguous_velocity_mps)

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

    def test_rejects_a_leg_count_mismatch(self) -> None:
        from dataclasses import replace

        with pytest.raises(ValueError, match="same length"):
            replace(load_scenario(S3_TOML), n_chirps=(128,))
