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

import numpy as np
import pytest

from radar_forge.core.ambiguity import fold_velocity_mps
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
        """Reproduces the two values scenario 003 §5.3 quotes, 6.84 and 3.34."""
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
                fold_span_mps=FOLD_SPAN_MPS,
                tracking=TrackingConfig(unfolding_mode="magic"),  # type: ignore[arg-type]
            )

    def test_rejects_a_non_positive_fold_span(self):
        with pytest.raises(ValueError, match="fold_span_mps"):
            ScenarioTracker(fold_span_mps=0.0)

    def test_lists_its_unfolding_modes(self):
        assert set(UNFOLDING_MODES) == {"track_aided", "oracle", "none"}

    def test_the_default_bootstrap_is_ten_frames(self):
        assert ScenarioTracker(fold_span_mps=FOLD_SPAN_MPS).min_unfold_frames == 10


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
            fold_span_mps=FOLD_SPAN_MPS,
            tracking=TrackingConfig(unfolding_mode=mode),
        )

    def test_oracle_mode_needs_a_truth_velocity(self):
        tracker = self._tracker("oracle")
        product = synthetic_product(15_000.0, 2.0)
        with pytest.raises(ValueError, match="oracle"):
            tracker.step(product, frame_index=0, time_s=0.0)

    def test_oracle_mode_unfolds_from_the_first_frame(self):
        """It has no bootstrap to wait for, which is what makes it an oracle."""
        tracker = self._tracker("oracle")
        product = synthetic_product(15_000.0, -30.4)
        frame = tracker.step(product, frame_index=0, time_s=0.0, truth_velocity_mps=-30.4)
        best = max(frame.measurements, key=lambda m: m.peak_power_w)
        assert best.velocity_unfolded_mps == pytest.approx(-30.4, abs=0.1)
        assert best.fold_index == -2

    def test_none_mode_passes_the_folded_value_through(self):
        """Present so §5.2's failure can be demonstrated, not only described."""
        tracker = self._tracker("none")
        product = synthetic_product(15_000.0, -30.4)
        frame = tracker.step(product, frame_index=0, time_s=0.0)
        best = max(frame.measurements, key=lambda m: m.peak_power_w)
        assert best.fold_index == 0
        assert best.velocity_unfolded_mps == pytest.approx(best.velocity_folded_mps)

    def test_track_aided_mode_waits_for_its_bootstrap(self):
        """A brand-new track cannot select a fold, and must not pretend to."""
        tracker = self._tracker("track_aided")
        frame = tracker.step(synthetic_product(15_000.0, 2.0), frame_index=0, time_s=0.0)
        assert all(m.velocity_unfolded_mps is None for m in frame.measurements)
        assert frame.unfold_reference_id is None

    def test_a_track_leaves_the_bootstrap_after_about_ten_frames(self):
        """The §5.3 diagnostic: the frame at which measurement_dim steps 1 -> 2."""
        tracker = self._tracker("track_aided")
        range_m, closing_mps = 20_000.0, 40.0
        for index in range(16):
            product = synthetic_product(range_m - closing_mps * index, 2.0, seed=100 + index)
            tracker.step(product, frame_index=index, time_s=float(index))

        confirmed = tracker.manager.confirmed_tracks
        assert confirmed
        unfold_frame = tracker.unfold_frame_of(confirmed[0].track_id)
        assert unfold_frame is not None
        assert 9 <= unfold_frame <= 13
