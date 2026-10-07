"""Tests for radar_forge.pipelines.tracking.

Split in two. The helpers -- fold arithmetic, the least-squares slope, the
bootstrap sizing -- have closed-form answers and are tested against them. The
detection path needs a real range-Doppler map, so those tests synthesise one
frame of scenario 001's S1 and are marked slow.

The Doppler-wrap test is the one worth reading. A target sitting on the
velocity wrap is returned by `cluster_detections` as two clusters at opposite
ends of the axis, and the mean of the two is nowhere near the target. Nothing
downstream notices: the tracker still produces a track, and the plot still
looks like a plot.
"""

from functools import cache
from pathlib import Path

import numpy as np
import pytest

from radar_forge.core.ambiguity import fold_velocity_mps
from radar_forge.core.radar import RadarLike
from radar_forge.pipelines.scenarios import load_scenario
from radar_forge.pipelines.tracking import (
    UNFOLDING_MODES,
    DetectionConfig,
    Measurement,
    ScenarioTracker,
    TrackingConfig,
    dual_prf_measurements,
    frame_detections,
    minimum_unfold_history_frames,
    range_slope_sigma_mps,
    replace_measurement,
    slope_velocity_mps,
    unfold_velocity_mps,
)

SIGMA_RANGE_M = 21.635652855125496
FOLD_SPAN_MPS = 15.295533571428571
SCENARIOS_DIR = Path(__file__).parent.parent.parent / "scenarios"


@cache
def bursts(variant: str) -> tuple[RadarLike, ...]:
    """The bursts of one of scenario 001's variants, read from its TOML."""
    return load_scenario(SCENARIOS_DIR / f"scenario_001_{variant}.toml").bursts


class TestUnfoldVelocity:
    """The fold arithmetic of §5.3, which has an exact answer."""

    @pytest.mark.parametrize("fold_index", range(-12, 13))
    def test_recovers_every_fold_in_the_unambiguous_span(self, fold_index):
        """+-191 m/s is 12 folds either side; each one must come back exactly."""
        truth_mps = fold_index * FOLD_SPAN_MPS + 3.0
        folded_mps = float(fold_velocity_mps(truth_mps, FOLD_SPAN_MPS / 2.0))
        velocity_mps, recovered = unfold_velocity_mps(folded_mps, truth_mps, FOLD_SPAN_MPS)
        assert recovered == fold_index
        # rtol 1e-12: one multiply and one add in float64.
        np.testing.assert_allclose(velocity_mps, truth_mps, rtol=1e-12)

    def test_a_prediction_within_half_a_span_selects_the_right_fold(self):
        """The guarantee the bootstrap is sized against."""
        truth_mps = -30.4
        folded_mps = float(fold_velocity_mps(truth_mps, FOLD_SPAN_MPS / 2.0))
        for offset_mps in (-7.0, -3.0, 0.0, 3.0, 7.0):
            velocity_mps, _ = unfold_velocity_mps(folded_mps, truth_mps + offset_mps, FOLD_SPAN_MPS)
            np.testing.assert_allclose(velocity_mps, truth_mps, atol=1e-9)

    def test_a_prediction_off_by_more_than_half_a_span_picks_the_wrong_fold(self):
        """Stated as a test because it is the scenario's central difficulty.

        Half a fold span is 7.65 m/s, and the simulated target's range rate
        moves by up to 11 m/s between frames. No selector can do better than
        this, which is why §13.4's dual-PRF variant is the real answer.
        """
        truth_mps = -30.4
        folded_mps = float(fold_velocity_mps(truth_mps, FOLD_SPAN_MPS / 2.0))
        velocity_mps, _ = unfold_velocity_mps(folded_mps, truth_mps + 9.0, FOLD_SPAN_MPS)
        assert abs(velocity_mps - truth_mps) == pytest.approx(FOLD_SPAN_MPS, abs=1e-9)

    def test_rejects_a_non_positive_span(self):
        with pytest.raises(ValueError, match="fold_span_mps"):
            unfold_velocity_mps(1.0, 1.0, 0.0)


class TestRangeSlope:
    """The independent, Doppler-free velocity estimate."""

    def test_matches_the_closed_form_sigma(self):
        """Reproduces the two values scenario 003 §14.4 quotes, 6.84 and 3.34."""
        assert range_slope_sigma_mps(5, SIGMA_RANGE_M, 1.0) == pytest.approx(6.842, abs=5e-4)
        assert range_slope_sigma_mps(8, SIGMA_RANGE_M, 1.0) == pytest.approx(3.338, abs=5e-4)

    def test_falls_as_the_window_grows(self):
        sigmas = [range_slope_sigma_mps(n, SIGMA_RANGE_M, 1.0) for n in range(2, 20)]
        assert sigmas == sorted(sigmas, reverse=True)

    def test_slope_velocity_is_positive_for_a_closing_target(self):
        """A falling range is a closing target, per D5."""
        ranges_m = [10_000.0, 9_900.0, 9_800.0, 9_700.0]
        assert slope_velocity_mps(ranges_m, 1.0) == pytest.approx(100.0, rel=1e-9)

    def test_slope_velocity_is_negative_for_an_opening_target(self):
        ranges_m = [10_000.0, 10_050.0, 10_100.0]
        assert slope_velocity_mps(ranges_m, 1.0) == pytest.approx(-50.0, rel=1e-9)

    def test_needs_two_points(self):
        assert slope_velocity_mps([1.0], 1.0) is None
        assert slope_velocity_mps([], 1.0) is None

    def test_the_bootstrap_takes_ten_frames_at_the_scenario_numbers(self):
        """§5.3's 'around ten frames', re-derived rather than trusted."""
        assert minimum_unfold_history_frames(SIGMA_RANGE_M, FOLD_SPAN_MPS, 1.0) == 10

    def test_a_tighter_gate_needs_a_longer_history(self):
        loose = minimum_unfold_history_frames(SIGMA_RANGE_M, FOLD_SPAN_MPS, 1.0, 3.0)
        tight = minimum_unfold_history_frames(SIGMA_RANGE_M, FOLD_SPAN_MPS, 1.0, 12.0)
        assert tight > loose

    @pytest.mark.parametrize("n_frames", [0, 1, -3])
    def test_rejects_too_few_frames(self, n_frames):
        with pytest.raises(ValueError, match="n_frames"):
            range_slope_sigma_mps(n_frames, SIGMA_RANGE_M, 1.0)

    @pytest.mark.parametrize("frame_time_s", [0.0, -1.0])
    def test_rejects_a_non_positive_frame_time(self, frame_time_s):
        """Sigma goes as 1/T, so a zero frame time is a division, not a limit."""
        with pytest.raises(ValueError, match="frame_time_s"):
            range_slope_sigma_mps(8, SIGMA_RANGE_M, frame_time_s)

    @pytest.mark.parametrize("unfold_sigma_gate", [0.0, -6.0])
    def test_rejects_a_non_positive_sigma_gate(self, unfold_sigma_gate):
        """The gate is a count of standard deviations that must fit in a fold."""
        with pytest.raises(ValueError, match="unfold_sigma_gate"):
            minimum_unfold_history_frames(SIGMA_RANGE_M, FOLD_SPAN_MPS, 1.0, unfold_sigma_gate)

    def test_reports_a_fold_that_range_history_can_never_resolve(self):
        """Some geometries simply cannot bootstrap, and must say so.

        Slope sigma falls only as N^-3/2, so a range measurement noisy enough
        against the fold span never reaches the gate. Returning a huge frame
        count instead of raising would be a track that stays in its bootstrap
        for the whole run while looking like it is about to leave it.
        """
        with pytest.raises(ValueError, match="never reaches"):
            minimum_unfold_history_frames(1.0e9, FOLD_SPAN_MPS, 1.0)

    def test_the_slope_sigma_matches_the_ordinary_least_squares_variance(self):
        """12 / (N (N^2 - 1)) is the exact OLS slope variance on a uniform grid.

        Derived rather than recorded: for x = 0..N-1 the least-squares slope
        variance is sigma^2 / Sxx with Sxx = N (N^2 - 1) / 12, so the closed
        form in the module is the textbook one and not a fitted approximation.
        """
        n_frames = 11
        times_s = np.arange(n_frames, dtype=float)
        sum_of_squares = float(((times_s - times_s.mean()) ** 2).sum())
        expected_mps = SIGMA_RANGE_M / np.sqrt(sum_of_squares)
        # rtol 1e-12: both sides are a handful of float64 operations.
        np.testing.assert_allclose(
            range_slope_sigma_mps(n_frames, SIGMA_RANGE_M, 1.0), expected_mps, rtol=1e-12
        )


class TestDualPrfMeasurements:
    """The dual-PRF path resolves the fold in the waveform, and needs a pair."""

    @pytest.mark.parametrize(
        ("n_products", "n_spans"),
        [(1, 1), (3, 3), (2, 1)],
        ids=["one-burst", "three-bursts", "mismatched"],
    )
    def test_rejects_anything_but_two_bursts(self, n_products, n_spans):
        """Two coprime folds identify a velocity; one or three do not.

        The guard runs before the maps are touched, so the check is on the
        arity of the call rather than on the contents -- which is exactly the
        error a caller wiring up a new scenario makes.
        """
        with pytest.raises(ValueError, match="exactly two bursts"):
            dual_prf_measurements([None] * n_products, [1.0] * n_spans)


class TestReplaceMeasurement:
    """Unfolding must change two fields and nothing else.

    A detection carries the evidence a plot and a CSV are drawn from -- the
    cluster's power, its cell count, the fractional indices its marker goes on.
    Unfolding is a statement about velocity alone, so if it also perturbed one
    of those the marker would drift off the peak it was measured at, and the
    picture would still look like a detection.
    """

    def test_carries_every_other_field_through_untouched(self):
        original = Measurement(
            range_m=18_160.5,
            velocity_folded_mps=7.502,
            peak_power_w=3.25e-12,
            total_power_w=9.5e-12,
            n_cells=58,
            range_index=242.375,
            velocity_index=125.5,
        )
        unfolded = replace_measurement(original, -7.79, -1)

        assert unfolded.velocity_unfolded_mps == -7.79
        assert unfolded.fold_index == -1
        for name in (
            "range_m",
            "velocity_folded_mps",
            "peak_power_w",
            "total_power_w",
            "n_cells",
            "range_index",
            "velocity_index",
        ):
            assert getattr(unfolded, name) == getattr(original, name), name

    def test_leaves_the_original_alone(self):
        original = Measurement(
            range_m=1.0,
            velocity_folded_mps=2.0,
            peak_power_w=3.0,
            total_power_w=4.0,
            n_cells=5,
            range_index=6.0,
            velocity_index=7.0,
        )
        replace_measurement(original, 99.0, 6)

        assert original.velocity_unfolded_mps is None
        assert original.fold_index is None


class TestConfiguration:
    """The documented failure modes of the tracker's own settings."""

    def test_rejects_an_unknown_unfolding_mode(self):
        with pytest.raises(ValueError, match="unfolding_mode"):
            ScenarioTracker(
                bursts("fmcw_low_prf"),
                tracking=TrackingConfig(unfolding_mode="magic"),  # type: ignore[arg-type]
            )

    def test_lists_its_unfolding_modes(self):
        assert set(UNFOLDING_MODES) == {"track_aided", "oracle", "none"}

    def test_the_default_bootstrap_is_ten_frames(self):
        assert ScenarioTracker(bursts("fmcw_low_prf")).min_unfold_frames == 10

    def test_the_fold_span_is_twice_the_first_bursts_unambiguous_velocity(self):
        np.testing.assert_allclose(
            ScenarioTracker(bursts("fmcw_low_prf")).fold_span_mps, FOLD_SPAN_MPS, rtol=1e-12
        )


# --------------------------------------------------------------------------- #
# The detection path, against a synthesised range-Doppler map
# --------------------------------------------------------------------------- #


def synthetic_product(range_m, velocity_mps, *, n_doppler_bins=256, n_range_bins=1000, seed=7):
    """A range-Doppler map with one point target, in noise.

    Built directly rather than through the scenario pipeline: these tests are
    about what `frame_detections` does with a map, and a hand-built one lets the
    target be placed exactly where the test needs it -- in particular, on the
    velocity wrap.
    """
    from radar_forge.pipelines.scenarios import RangeDopplerProduct

    rng = np.random.default_rng(seed)
    range_axis_m = np.arange(n_range_bins, dtype=np.float64) * 74.9481145
    half_span = FOLD_SPAN_MPS / 2.0
    velocity_axis_mps = np.linspace(
        -half_span, half_span, n_doppler_bins, endpoint=False, dtype=np.float64
    )

    rd_map = (
        rng.standard_normal((n_doppler_bins, n_range_bins))
        + 1j * rng.standard_normal((n_doppler_bins, n_range_bins))
    ) / np.sqrt(2.0)

    # A separable sinc-like response, wide enough in Doppler to straddle the
    # wrap when the target sits on it, which is the case under test.
    range_index = float(np.interp(range_m, range_axis_m, np.arange(n_range_bins)))
    folded_mps = float(fold_velocity_mps(velocity_mps, half_span))
    velocity_index = (folded_mps + half_span) / FOLD_SPAN_MPS * n_doppler_bins

    rows = np.arange(n_doppler_bins, dtype=np.float64)[:, None]
    cols = np.arange(n_range_bins, dtype=np.float64)[None, :]
    # Circular distance in Doppler: the axis wraps, and that is the point.
    row_offset = (rows - velocity_index + n_doppler_bins / 2) % n_doppler_bins - n_doppler_bins / 2
    amplitude = 3.0e3 * np.exp(-((row_offset / 2.5) ** 2) - ((cols - range_index) / 1.2) ** 2)
    rd_map = rd_map + amplitude

    return RangeDopplerProduct(
        rd_map=rd_map.astype(np.complex128),
        range_axis_m=range_axis_m,
        velocity_axis_mps=velocity_axis_mps,
    )


class TestFrameDetections:
    """Converting a map into measurements in metres and metres per second."""

    def test_finds_a_target_away_from_the_wrap(self):
        product = synthetic_product(15_000.0, 2.0)
        measurements = frame_detections(product)
        assert measurements
        best = max(measurements, key=lambda m: m.peak_power_w)
        # Within one range bin and one Doppler bin of where it was put.
        assert best.range_m == pytest.approx(15_000.0, abs=74.95)
        assert best.velocity_folded_mps == pytest.approx(2.0, abs=FOLD_SPAN_MPS / 256)

    def test_a_target_on_the_velocity_wrap_is_one_detection_not_two(self):
        """The Doppler axis is circular; cluster_detections is not.

        Regression for the failure this module's roll exists to prevent. In
        scenario 001's S1 at frame 0 the target straddles the wrap and comes
        back as 58 cells at +7.493 m/s and 37 cells at -7.511 m/s; the
        power-weighted mean of the pair is near zero, which is nowhere near the
        target's true folded velocity of +7.502 m/s.
        """
        half_span = FOLD_SPAN_MPS / 2.0
        # Just inside the negative edge, so the response spills across the wrap.
        product = synthetic_product(15_000.0, -half_span + 0.05)
        measurements = frame_detections(product)

        near_target = [m for m in measurements if abs(m.range_m - 15_000.0) < 150.0]
        assert len(near_target) == 1, "the wrap split the target into two detections"

        found = near_target[0].velocity_folded_mps
        # Compared on the folded axis: a value just inside each edge is a
        # neighbour of one just outside, not its opposite.
        error_mps = abs(float(fold_velocity_mps(found - (-half_span + 0.05), half_span)))
        assert error_mps < 2.0 * FOLD_SPAN_MPS / 256

    def test_range_sidelobes_of_one_target_do_not_become_several_detections(self):
        """One point target cannot produce two returns at one range."""
        product = synthetic_product(15_000.0, 2.0)
        measurements = frame_detections(product)
        indices = sorted(m.range_index for m in measurements)
        gaps = np.diff(indices)
        assert all(gap > 1.0 for gap in gaps)

    def test_merging_can_be_switched_off(self):
        product = synthetic_product(15_000.0, 2.0)
        merged = frame_detections(product, DetectionConfig())
        unmerged = frame_detections(product, DetectionConfig(merge_range_bins=0.0))
        assert len(unmerged) >= len(merged)

    def test_a_noise_only_map_yields_about_the_designed_false_alarm_count(self):
        """§3's arithmetic, measured rather than trusted.

        2.46 expected over 245,760 valid cells at pfa = 1e-5. Asserted loosely
        because a single frame is one Poisson draw, not a rate measurement.
        """
        rng = np.random.default_rng(11)
        from radar_forge.pipelines.scenarios import RangeDopplerProduct

        noise = (
            rng.standard_normal((256, 1000)) + 1j * rng.standard_normal((256, 1000))
        ) / np.sqrt(2.0)
        product = RangeDopplerProduct(
            rd_map=noise.astype(np.complex128),
            range_axis_m=np.arange(1000, dtype=np.float64) * 74.9481145,
            velocity_axis_mps=np.linspace(-7.65, 7.65, 256, endpoint=False),
        )
        assert len(frame_detections(product)) < 15

    def test_measurements_carry_no_unfolded_velocity_yet(self):
        """Unfolding needs a track, so detection cannot do it."""
        product = synthetic_product(15_000.0, 2.0)
        for measurement in frame_detections(product):
            assert measurement.velocity_unfolded_mps is None
            assert measurement.fold_index is None


class TestScenarioTrackerUnfolding:
    """The three unfolding modes of §5.3."""

    def _tracker(self, mode):
        return ScenarioTracker(
            bursts("fmcw_low_prf"),
            tracking=TrackingConfig(unfolding_mode=mode),
        )

    def test_oracle_mode_needs_a_truth_velocity(self):
        tracker = self._tracker("oracle")
        product = synthetic_product(15_000.0, 2.0)
        with pytest.raises(ValueError, match="oracle"):
            tracker.step([product], frame_index=0, time_s=0.0)

    def test_oracle_mode_unfolds_from_the_first_frame(self):
        """It has no bootstrap to wait for, which is what makes it an oracle."""
        tracker = self._tracker("oracle")
        product = synthetic_product(15_000.0, -30.4)
        frame = tracker.step([product], frame_index=0, time_s=0.0, truth_velocity_mps=-30.4)
        best = max(frame.measurements, key=lambda m: m.peak_power_w)
        assert best.velocity_unfolded_mps == pytest.approx(-30.4, abs=0.1)
        assert best.fold_index == -2

    def test_none_mode_passes_the_folded_value_through(self):
        """Present so §5.2's failure can be demonstrated, not only described."""
        tracker = self._tracker("none")
        product = synthetic_product(15_000.0, -30.4)
        frame = tracker.step([product], frame_index=0, time_s=0.0)
        best = max(frame.measurements, key=lambda m: m.peak_power_w)
        assert best.fold_index == 0
        assert best.velocity_unfolded_mps == pytest.approx(best.velocity_folded_mps)

    def test_track_aided_mode_waits_for_its_bootstrap(self):
        """A brand-new track cannot select a fold, and must not pretend to."""
        tracker = self._tracker("track_aided")
        frame = tracker.step([synthetic_product(15_000.0, 2.0)], frame_index=0, time_s=0.0)
        assert all(m.velocity_unfolded_mps is None for m in frame.measurements)
        assert frame.unfold_reference_id is None

    def test_a_track_leaves_the_bootstrap_after_about_ten_frames(self):
        """The §5.3 diagnostic: the frame at which measurement_dim steps 1 -> 2."""
        tracker = self._tracker("track_aided")
        range_m, closing_mps = 20_000.0, 40.0
        for index in range(16):
            product = synthetic_product(range_m - closing_mps * index, 2.0, seed=100 + index)
            tracker.step([product], frame_index=index, time_s=float(index))

        confirmed = tracker.tracker.confirmed_tracks
        assert confirmed
        unfold_frame = tracker.unfold_frame_of(confirmed[0].track_id)
        assert unfold_frame is not None
        assert 9 <= unfold_frame <= 13


# --------------------------------------------------------------------------- #
# Several targets on one hand-built map, for the dual-PRF and UKF paths
# --------------------------------------------------------------------------- #


def targets_product(
    targets,
    *,
    fold_span_mps,
    n_doppler_bins=128,
    n_range_bins=800,
    range_bin_m=74.9481145,
    amplitude=3.0e3,
    seed=7,
):
    """A range-Doppler map with point targets at ``(range_m, velocity_mps)``, in noise.

    Both axes are circular, as a real map's are: a response near one end of
    either axis spills onto the other end.
    """
    from radar_forge.pipelines.scenarios import RangeDopplerProduct

    rng = np.random.default_rng(seed)
    range_axis_m = np.arange(n_range_bins, dtype=np.float64) * range_bin_m
    half_span = fold_span_mps / 2.0
    velocity_axis_mps = np.linspace(
        -half_span, half_span, n_doppler_bins, endpoint=False, dtype=np.float64
    )
    rd_map = (
        rng.standard_normal((n_doppler_bins, n_range_bins))
        + 1j * rng.standard_normal((n_doppler_bins, n_range_bins))
    ) / np.sqrt(2.0)
    rows = np.arange(n_doppler_bins, dtype=np.float64)[:, None]
    cols = np.arange(n_range_bins, dtype=np.float64)[None, :]
    for range_m, velocity_mps in targets:
        range_index = (range_m / range_bin_m) % n_range_bins
        folded_mps = float(fold_velocity_mps(velocity_mps, half_span))
        velocity_index = (folded_mps + half_span) / fold_span_mps * n_doppler_bins
        row_offset = (rows - velocity_index + n_doppler_bins / 2) % n_doppler_bins
        col_offset = (cols - range_index + n_range_bins / 2) % n_range_bins
        rd_map = rd_map + amplitude * np.exp(
            -(((row_offset - n_doppler_bins / 2) / 1.5) ** 2)
            - ((col_offset - n_range_bins / 2) / 1.2) ** 2
        )
    return RangeDopplerProduct(
        rd_map=rd_map.astype(np.complex128),
        range_axis_m=range_axis_m,
        velocity_axis_mps=velocity_axis_mps,
    )


# Scenario 001's S3: two FMCW bursts at 5 and 6 kHz.
S3_SPANS_MPS = (76.47578, 91.77093)


def dual_products(targets_a, targets_b, *, seed=7):
    """One map per burst of an S3-like pair, each with its own targets."""
    return [
        targets_product(targets_a, fold_span_mps=S3_SPANS_MPS[0], seed=seed),
        targets_product(targets_b, fold_span_mps=S3_SPANS_MPS[1], seed=seed + 1),
    ]


class TestDualPrfDetections:
    """Pairing across a dual-PRF pair, and the status each detection gets."""

    def test_a_target_in_both_bursts_is_one_accepted_pair(self):
        from radar_forge.pipelines.tracking import dual_prf_detections

        target = (15_000.0, -30.4)
        records = dual_prf_detections(dual_products([target], [target]), S3_SPANS_MPS)
        paired = [r for r in records if abs(r.range_m - 15_000.0) < 150.0]

        assert sorted(r.burst_index for r in paired) == [0, 1]
        assert {r.status for r in paired} == {"accepted"}
        assert len({r.pair_id for r in paired}) == 1
        for record in paired:
            # Within one Doppler bin of the coarser burst.
            assert record.velocity_unfolded_mps == pytest.approx(-30.4, abs=S3_SPANS_MPS[1] / 128)

    def test_a_target_in_one_burst_only_is_a_missing_pair(self):
        from radar_forge.pipelines.tracking import dual_prf_detections

        records = dual_prf_detections(dual_products([(15_000.0, -30.4)], []), S3_SPANS_MPS)
        target = [r for r in records if abs(r.range_m - 15_000.0) < 150.0]
        assert [(r.burst_index, r.status, r.pair_id) for r in target] == [(0, "missing_pair", None)]

    def test_a_false_alarm_beside_the_target_does_not_stop_the_pair(self):
        """Mutual nearest neighbours: the target's own pair forms; the extra one is left out.

        Requiring each detection to have exactly one compatible partner would
        refuse the target's pair here, because two detections in burst B lie
        within the tolerance of the one in burst A.
        """
        from radar_forge.pipelines.tracking import dual_prf_detections

        target = (15_000.0, -30.4)
        beside = (15_000.0 + 2 * 74.9481145, 10.0)
        records = dual_prf_detections(dual_products([target], [target, beside]), S3_SPANS_MPS)
        near = [r for r in records if abs(r.range_m - 15_000.0) < 300.0]

        accepted = [r for r in near if r.status == "accepted"]
        assert sorted(r.burst_index for r in accepted) == [0, 1]
        assert [r.burst_index for r in near if r.status == "ambiguous_pair"] == [1]

    def test_velocities_that_agree_on_nothing_are_unresolved(self):
        from radar_forge.pipelines.tracking import dual_prf_detections

        # The same range, but folded velocities that no single true velocity
        # within +-191 m/s produces in both bursts.
        products = dual_products([(15_000.0, -30.4)], [(15_000.0, -30.4 + 20.0)])
        records = dual_prf_detections(products, S3_SPANS_MPS)
        target = [r for r in records if abs(r.range_m - 15_000.0) < 150.0]

        assert {r.status for r in target} == {"unresolved_velocity"}
        assert len({r.pair_id for r in target}) == 1
        assert all(r.velocity_unfolded_mps is None for r in target)

    def test_only_the_first_bursts_accepted_detections_are_measurements(self):
        target = (15_000.0, -30.4)
        measurements = dual_prf_measurements(dual_products([target], [target]), S3_SPANS_MPS)
        near = [m for m in measurements if abs(m.range_m - 15_000.0) < 150.0]
        assert [(m.burst_index, m.status) for m in near] == [(0, "accepted")]

    def test_a_tie_is_nearest_for_neither(self):
        from radar_forge.pipelines.tracking import _strictly_nearest

        distance_m = np.array([[1.0, 1.0, 2.0], [3.0, 0.5, 4.0]])
        np.testing.assert_array_equal(
            _strictly_nearest(distance_m, axis=1),
            [[False, False, False], [False, True, False]],
        )
        np.testing.assert_array_equal(
            _strictly_nearest(distance_m, axis=0),
            [[True, False, True], [False, True, False]],
        )

    def test_an_empty_burst_marks_nothing(self):
        from radar_forge.pipelines.tracking import _strictly_nearest

        assert _strictly_nearest(np.zeros((2, 0)), axis=1).shape == (2, 0)
        assert _strictly_nearest(np.zeros((0, 3)), axis=1).shape == (0, 3)


class TestCircularRange:
    """frame_detections with wrap_range: the range axis as the circle it is."""

    # S2's axis: 100 bins of c / 2fs, which is one PRI of delay, 5995.85 m.
    N_RANGE_BINS = 100
    RANGE_BIN_M = 59.958491600000004

    def _product(self, range_m):
        return targets_product(
            [(range_m, -17.0)],
            fold_span_mps=2.0 * 191.1935,
            n_doppler_bins=256,
            n_range_bins=self.N_RANGE_BINS,
            range_bin_m=self.RANGE_BIN_M,
        )

    def test_a_target_near_the_end_is_detected_only_on_a_circle(self):
        """Without wrapping, the 20 cells at each end have no reference window."""
        range_m = 5 * self.RANGE_BIN_M
        product = self._product(range_m)
        assert not [m for m in frame_detections(product) if abs(m.range_m - range_m) < 150.0]
        wrapped = frame_detections(product, wrap_range=True)
        assert [m for m in wrapped if abs(m.range_m - range_m) < 150.0]

    def test_a_target_straddling_the_ends_is_one_detection_at_its_range(self):
        period_m = self.N_RANGE_BINS * self.RANGE_BIN_M
        range_m = period_m - 0.3 * self.RANGE_BIN_M
        measurements = frame_detections(self._product(range_m), wrap_range=True)

        def gap_m(m):
            gap = abs(m.range_m - range_m) % period_m
            return min(gap, period_m - gap)

        near = [m for m in measurements if gap_m(m) < 3 * self.RANGE_BIN_M]
        assert len(near) == 1
        assert gap_m(near[0]) < self.RANGE_BIN_M
        # The centroid may lie past the last bin centre, still inside one period.
        assert 0.0 <= near[0].range_m < period_m


class TestBinQuantisationSigmas:
    def test_one_bin_over_root_twelve(self):
        from radar_forge.pipelines.tracking import bin_quantisation_sigmas

        product = synthetic_product(15_000.0, 2.0)
        sigma_range_m, sigma_velocity_mps = bin_quantisation_sigmas(product)
        np.testing.assert_allclose(sigma_range_m, 74.9481145 / np.sqrt(12.0), rtol=1e-12)
        np.testing.assert_allclose(
            sigma_velocity_mps, FOLD_SPAN_MPS / 256 / np.sqrt(12.0), rtol=1e-12
        )


class TestConfigsFromScenario:
    """TOML values are type-checked, never coerced."""

    def _scenario(self, detection=None, tracking=None):
        from dataclasses import replace

        scenario = load_scenario(SCENARIOS_DIR / "scenario_003_tracking.toml")
        return replace(scenario, detection_table=detection, tracking_table=tracking)

    @pytest.mark.parametrize(
        ("table", "key"),
        [
            ({"pfa": "1e-5"}, "pfa"),
            ({"n_train": 16.0}, "n_train"),
            ({"n_guard": True}, "n_guard"),
            ({"variant": 1}, "variant"),
        ],
        ids=["string-for-float", "float-for-int", "bool-for-int", "int-for-string"],
    )
    def test_rejects_a_value_of_the_wrong_type(self, table, key):
        from radar_forge.pipelines.tracking import configs_from_scenario

        with pytest.raises(TypeError, match=key):
            configs_from_scenario(self._scenario(detection=table))

    def test_a_float_field_takes_an_integer(self):
        from radar_forge.pipelines.tracking import configs_from_scenario

        _, tracking = configs_from_scenario(self._scenario(tracking={"v_max_mps": 100}))
        assert tracking.v_max_mps == 100

    def test_rejects_a_boolean_for_a_float(self):
        from radar_forge.pipelines.tracking import configs_from_scenario

        with pytest.raises(TypeError, match="gate_probability"):
            configs_from_scenario(self._scenario(tracking={"gate_probability": True}))


class TestScenarioTrackerSettings:
    """What ScenarioTracker refuses to build, and what it builds."""

    @pytest.mark.parametrize(
        ("tracking", "match"),
        [
            (TrackingConfig(estimator="ekf"), "estimator"),  # type: ignore[arg-type]
            (TrackingConfig(estimator="ukf", unfolding_mode="track_aided"), "unfolding_mode"),
            (
                TrackingConfig(estimator="ukf", unfolding_mode="none", state_model="enu_2d"),
                "range_1d",
            ),
        ],
        ids=["unknown-estimator", "ukf-track-aided", "ukf-enu"],
    )
    def test_rejects_settings_it_cannot_run(self, tracking, match):
        with pytest.raises(ValueError, match=match):
            ScenarioTracker(bursts("fmcw_low_prf"), tracking=tracking)

    @pytest.mark.parametrize("n_bursts", [0, 3])
    def test_rejects_anything_but_one_or_two_bursts(self, n_bursts):
        burst = bursts("fmcw_low_prf")[0]
        with pytest.raises(ValueError, match="one burst or a dual-PRF pair"):
            ScenarioTracker((burst,) * n_bursts)

    def test_rejects_a_pair_that_folds_velocity_alike(self):
        burst = bursts("fmcw_low_prf")[0]
        with pytest.raises(ValueError, match="fold velocity differently"):
            ScenarioTracker((burst, burst))

    def test_step_needs_one_product_per_burst(self):
        tracker = ScenarioTracker(bursts("fmcw_low_prf"))
        with pytest.raises(ValueError, match="one range-Doppler product per burst"):
            tracker.step([], frame_index=0, time_s=0.0)

    def test_step_rejects_a_frame_earlier_than_the_last(self):
        tracker = ScenarioTracker(bursts("fmcw_low_prf"))
        tracker.step([synthetic_product(15_000.0, 2.0)], frame_index=0, time_s=1.0)
        with pytest.raises(ValueError, match="time order"):
            tracker.step([synthetic_product(15_000.0, 2.0)], frame_index=1, time_s=0.5)

    def test_the_kalman_path_treats_range_as_unambiguous(self):
        tracker = ScenarioTracker(bursts("fmcw_low_prf"))
        assert tracker.folding_layout.coordinates[0].period is None

    def test_the_ukf_range_period_is_the_maps_range_span(self):
        """S2's pulsed axis is one PRI of delay, c / 2 PRF: exactly its unambiguous range."""
        burst = bursts("pulsed_medium_prf")[0]
        tracker = ScenarioTracker((burst,), tracking=UKF_SETTINGS)
        np.testing.assert_allclose(
            tracker.folding_layout.coordinates[0].period, burst.unambiguous_range_m, rtol=1e-12
        )

    def test_an_fmcw_range_span_is_twice_its_real_sampling_unambiguous_range(self):
        """Complex sampling represents beat frequencies over all of [0, f_s), not [0, f_s/2)."""
        burst = bursts("fmcw_low_prf")[0]
        tracker = ScenarioTracker((burst,), tracking=UKF_SETTINGS)
        np.testing.assert_allclose(
            tracker.folding_layout.coordinates[0].period, 2.0 * burst.unambiguous_range_m, rtol=1e-9
        )

    def test_from_scenario_reads_the_tables(self):
        scenario = load_scenario(SCENARIOS_DIR / "scenario_003_ukf_fmcw_dual_prf.toml")
        tracker = ScenarioTracker.from_scenario(scenario)
        assert tracker.tracking.estimator == "ukf"
        assert tracker.bursts == scenario.bursts


UKF_SETTINGS = TrackingConfig(
    estimator="ukf",
    unfolding_mode="none",
    sigma_accel_mps2=4.0,
    gate_probability=0.997,
    n_confirm_hits=3,
    n_confirm_frames=5,
    n_delete_misses=5,
    n_reacquire_frames=0,
    v_max_mps=100.0,
)


class TestUkfPath:
    """The UKF tracker on hand-built maps, where the truth is known exactly."""

    def _run(self, tracker, ranges_m, velocity_mps, **product_kwargs):
        frames = []
        for index, range_m in enumerate(ranges_m):
            product = targets_product([(range_m, velocity_mps)], seed=100 + index, **product_kwargs)
            products = [product] * len(tracker.bursts)
            frames.append(tracker.step(products, frame_index=index, time_s=float(index)))
        return frames

    def test_a_burst_that_folds_doppler_measures_range_alone(self):
        tracker = ScenarioTracker(bursts("fmcw_low_prf"), tracking=UKF_SETTINGS)
        frames = self._run(
            tracker,
            [20_000.0 - 40.0 * k for k in range(8)],
            -40.0,
            fold_span_mps=FOLD_SPAN_MPS,
            n_doppler_bins=256,
            n_range_bins=1000,
        )
        tracks = [t for t in frames[-1].tracks if t.is_confirmed]
        assert len(tracks) == 1
        assert tracks[0].measurement_dim == 1
        # Range alone was measured, so no detection was given a velocity.
        assert all(m.velocity_unfolded_mps is None for f in frames for m in f.measurements)

    def test_a_burst_that_cannot_fold_doppler_measures_range_rate_on_fold_zero(self):
        """S2's +-191 m/s exceeds v_max_mps, so its folded velocity is the velocity."""
        tracker = ScenarioTracker(bursts("pulsed_medium_prf"), tracking=UKF_SETTINGS)
        frames = self._run(
            tracker,
            [3_000.0 + 17.0 * k for k in range(8)],
            -17.0,
            fold_span_mps=2.0 * 191.1935,
            n_doppler_bins=256,
            n_range_bins=100,
            range_bin_m=59.958491600000004,
        )
        track = next(t for t in frames[-1].tracks if t.is_confirmed)
        assert track.measurement_dim == 2
        np.testing.assert_allclose(track.state[1], -17.0, atol=2.0 * 191.1935 / 256)
        associated = frames[-1].measurements[frames[-1].associations[track.track_id]]
        assert associated.fold_index == 0
        assert associated.velocity_unfolded_mps == associated.velocity_folded_mps

    def test_a_track_crosses_the_range_wrap_under_one_id(self):
        """The filter's range is continuous; the exported range is wrapped into one period."""
        burst = bursts("pulsed_medium_prf")[0]
        tracker = ScenarioTracker((burst,), tracking=UKF_SETTINGS)
        period_m = tracker.folding_layout.coordinates[0].period
        # Opening at 17 m/s from 5.85 km, so the target crosses the wrap near frame 9.
        frames = self._run(
            tracker,
            [5_850.0 + 17.0 * k for k in range(16)],
            -17.0,
            fold_span_mps=2.0 * 191.1935,
            n_doppler_bins=256,
            n_range_bins=100,
            range_bin_m=59.958491600000004,
        )
        confirmed_ids = {t.track_id for f in frames for t in f.tracks if t.is_confirmed}
        assert len(confirmed_ids) == 1
        last = next(t for t in frames[-1].tracks if t.is_confirmed)
        assert 0.0 <= last.state[0] < period_m
        np.testing.assert_allclose(last.state[0], (5_850.0 + 17.0 * 15) % period_m, atol=60.0)

    def test_records_say_which_detection_each_track_took(self):
        tracker = ScenarioTracker(bursts("pulsed_medium_prf"), tracking=UKF_SETTINGS)
        frames = self._run(
            tracker,
            [3_000.0 + 17.0 * k for k in range(6)],
            -17.0,
            fold_span_mps=2.0 * 191.1935,
            n_doppler_bins=256,
            n_range_bins=100,
            range_bin_m=59.958491600000004,
        )
        frame = frames[-1]
        track = next(t for t in frame.tracks if t.is_confirmed)
        detection = frame.measurements[frame.associations[track.track_id]]
        assert abs(detection.range_m - (3_000.0 + 17.0 * 5)) < 60.0
        assert track.nis is not None
        assert track.n_hits == 6
        assert track.n_misses == 0

    def test_a_dual_prf_pair_measures_range_rate_from_the_first_frame(self):
        tracker = ScenarioTracker(bursts("fmcw_dual_prf"), tracking=UKF_SETTINGS)
        target = (15_000.0, -30.4)
        frame = tracker.step(dual_products([target], [target]), frame_index=0, time_s=0.0)
        (track,) = [t for t in frame.tracks if abs(t.state[0] - 15_000.0) < 150.0]
        assert track.measurement_dim == 2
        np.testing.assert_allclose(track.state[1], -30.4, atol=S3_SPANS_MPS[1] / 128)


class TestScorePrimaryTrack:
    """Scoring against truth: closed-form answers on hand-built frames."""

    @staticmethod
    def _frames(states, *, status="confirmed", nis=1.0):
        from radar_forge.pipelines.tracking import FrameTracks, TrackRecord

        return [
            FrameTracks(
                frame_index=index,
                time_s=float(index),
                measurements=(),
                associations={},
                tracks=()
                if state is None
                else (
                    TrackRecord(
                        track_id=7,
                        status=status,
                        state=np.asarray(state, dtype=np.float64),
                        covariance=np.eye(2),
                        measurement_dim=2,
                        n_hits=index + 1,
                        n_misses=0,
                        nis=nis,
                    ),
                ),
            )
            for index, state in enumerate(states)
        ]

    @staticmethod
    def _layout(period_m=None):
        from radar_forge.core.tracking import Coordinate, StateLayout

        return StateLayout((Coordinate("range_m", "m", period=period_m),))

    def test_rmse_latency_and_counts(self):
        from radar_forge.pipelines.tracking import score_primary_track

        frames = self._frames([None, [1003.0, 10.0], [996.0, 13.0]])
        rows = score_primary_track(
            frames, [1000.0, 1000.0, 1000.0], [10.0, 10.0, 10.0], folding_layout=self._layout()
        )
        values = {row.metric: row.value for row in rows}

        assert values["n_tracked_frames"] == 2
        assert values["n_confirmed_tracks"] == 1
        assert values["n_confirmed_frames"] == 2
        assert values["confirmation_latency_s"] == 1.0
        np.testing.assert_allclose(values["range_rmse_m"], np.sqrt((9 + 16) / 2), rtol=1e-12)
        np.testing.assert_allclose(values["range_rate_rmse_mps"], np.sqrt(9 / 2), rtol=1e-12)
        assert values["mean_nis"] == 1.0
        assert {(row.frame_start, row.frame_end) for row in rows} == {(0, 2)}
        assert {row.track_id for row in rows if row.metric == "range_rmse_m"} == {7}

    def test_a_range_error_across_the_wrap_takes_the_short_way(self):
        from radar_forge.pipelines.tracking import score_primary_track

        rows = score_primary_track(
            self._frames([[5990.0, 0.0]]), [5.0], [0.0], folding_layout=self._layout(5996.0)
        )
        values = {row.metric: row.value for row in rows}
        np.testing.assert_allclose(values["range_rmse_m"], 11.0, rtol=1e-12)

    def test_with_no_confirmed_track_only_run_metrics_are_returned(self):
        from radar_forge.pipelines.tracking import score_primary_track

        rows = score_primary_track(
            self._frames([[1000.0, 10.0]], status="tentative"),
            [1000.0],
            [10.0],
            folding_layout=self._layout(),
        )
        assert all(row.track_id is None for row in rows)
        assert {row.metric for row in rows} >= {"n_tracked_frames", "n_detections_accepted"}

    @pytest.mark.parametrize("n_truth", [0, 2])
    def test_rejects_truth_that_does_not_match_the_frames(self, n_truth):
        from radar_forge.pipelines.tracking import score_primary_track

        with pytest.raises(ValueError, match="one truth value per frame"):
            score_primary_track(
                self._frames([[1.0, 1.0]]),
                [0.0] * n_truth,
                [0.0] * n_truth,
                folding_layout=self._layout(),
            )
