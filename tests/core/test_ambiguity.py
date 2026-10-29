"""Tests for dual-PRF range and Doppler ambiguity resolution.

Ground truth here is closed form and needs no simulation: folding is a known
modular map, so the strongest available check is to fold a swept set of true
values and require the unfolder to return exactly what went in. The scenario
001 S3 half-intervals and range spans are used throughout, because the property
that matters is not that the algorithm works for some pair of rates but that it
works for the 5:6 pair the scenario actually ships.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.ambiguity import (
    fold_into_period,
    fold_velocity_mps,
    unfold_doppler_dual_prf,
    unfold_dual_period,
    unfold_range_dual_prf,
)

# Scenario 001 S3: FMCW bursts at 5.0 kHz and 6.0 kHz, per spec §3.4.
V_UA_A = 38.24
V_UA_B = 45.89
MAX_VELOCITY_MPS = 191.0
TOLERANCE_MPS = 1.0

# Scenario 001 S1: the heavily folded low-PRF FMCW burst.
S1_V_UA = 7.65

# Scenario 001 S3's range spans, rounded: the two maps wrap at 60.0 and
# 50.0 km (spec/scenario-003-tracking.md §14.11), so the pair repeats at 300 km.
SPAN_A_M = 60_000.0
SPAN_B_M = 50_000.0
REPEAT_M = 300_000.0
TOLERANCE_M = 50.0


class TestFoldIntoPeriod:
    def test_a_range_interval_starts_at_zero(self) -> None:
        """130 km on a 60 km axis is two wraps and 10 km in."""
        np.testing.assert_allclose(
            fold_into_period(130_000.0, SPAN_A_M, lower=0.0), 10_000.0, rtol=1e-12
        )

    def test_the_interval_is_half_open_at_the_top(self) -> None:
        np.testing.assert_allclose(fold_into_period(SPAN_A_M, SPAN_A_M, lower=0.0), 0.0)

    def test_matches_fold_velocity_on_a_centred_interval(self) -> None:
        """fold_velocity_mps is the centred case, and must agree bit for bit."""
        velocity_mps = np.linspace(-500.0, 500.0, 2001)
        np.testing.assert_array_equal(
            fold_into_period(velocity_mps, 2.0 * S1_V_UA, lower=-S1_V_UA),
            fold_velocity_mps(velocity_mps, S1_V_UA),
        )

    def test_rejects_a_zero_period(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            fold_into_period(10.0, 0.0, lower=0.0)


class TestFoldVelocityMps:
    def test_folds_an_aircraft_by_the_predicted_number_of_intervals(self) -> None:
        """80 m/s at S1's 7.65 m/s half-interval is five folds down to 3.5 m/s."""
        span = 2.0 * S1_V_UA
        np.testing.assert_allclose(fold_velocity_mps(80.0, S1_V_UA), 80.0 - 5.0 * span, rtol=1e-12)

    def test_the_interval_is_half_open_at_the_top(self) -> None:
        """+v_ua wraps to -v_ua; the docstring promises [-v_ua, +v_ua)."""
        np.testing.assert_allclose(fold_velocity_mps(S1_V_UA, S1_V_UA), -S1_V_UA, rtol=1e-15)

    def test_every_result_lies_inside_the_interval(self) -> None:
        velocity_mps = np.linspace(-500.0, 500.0, 2001)
        folded_mps = fold_velocity_mps(velocity_mps, S1_V_UA)
        assert np.all(folded_mps >= -S1_V_UA)
        assert np.all(folded_mps < S1_V_UA)

    def test_rejects_a_zero_interval(self) -> None:
        """Zero is the boundary: a ``< 0`` guard would hand np.mod a zero span."""
        with pytest.raises(ValueError, match="strictly positive"):
            fold_velocity_mps(10.0, 0.0)


class TestUnfoldDopplerDualPrf:
    def test_reaches_the_full_span_the_lcm_of_the_two_spans_allows(self) -> None:
        """The limit is the least common multiple of the folding spans.

        For the 5:6 bursts that is 6 * 2 * 38.24 = 458.88 m/s, so a half-span of
        about 229 m/s is recoverable (``spec/scenario-001-xband.md``, S3).
        """
        true_mps = np.linspace(-225.0, 225.0, 901)
        velocity_mps, residual_mps = unfold_doppler_dual_prf(
            fold_velocity_mps(true_mps, V_UA_A),
            fold_velocity_mps(true_mps, V_UA_B),
            V_UA_A,
            V_UA_B,
            max_velocity_mps=229.0,
            tolerance_mps=TOLERANCE_MPS,
        )
        # Exact arithmetic recovery: the candidate is the true value to within
        # the float error of a handful of adds, so 1e-9 is generous.
        np.testing.assert_allclose(velocity_mps, true_mps, atol=1e-9)
        np.testing.assert_allclose(residual_mps, 0.0, atol=1e-9)

    def test_a_target_outside_the_bound_can_alias_to_a_confident_wrong_answer(self) -> None:
        """The failure the Notes section warns about, pinned so it cannot surprise.

        A 300 m/s target is beyond the 458.88 m/s repeat's half-span, so it
        folds identically to one at 300 - 458.88 m/s. That candidate is inside
        max_velocity_mps, so it is returned with a near-zero residual and no
        indication that anything went wrong. The bound is a claim about the
        target, and a wrong claim yields a wrong answer, not a missing one.
        """
        true_mps = 300.0
        velocity_mps, residual_mps = unfold_doppler_dual_prf(
            fold_velocity_mps(true_mps, V_UA_A),
            fold_velocity_mps(true_mps, V_UA_B),
            V_UA_A,
            V_UA_B,
            max_velocity_mps=MAX_VELOCITY_MPS,
            tolerance_mps=TOLERANCE_MPS,
        )
        alias_period_mps = 6.0 * 2.0 * V_UA_A
        # The answer is a burst-A candidate, folded A plus a whole number of A
        # spans, and 300 - 6 spans is exactly such a candidate: float error of a
        # few adds, so atol 1e-9 as in the span sweep.
        np.testing.assert_allclose(velocity_mps, true_mps - alias_period_mps, atol=1e-9)
        assert residual_mps < TOLERANCE_MPS

    def test_reports_an_inconsistent_pair_as_unresolved(self) -> None:
        """Noise beyond the tolerance yields nan, not a guess."""
        velocity_mps, residual_mps = unfold_doppler_dual_prf(
            fold_velocity_mps(80.0, V_UA_A),
            fold_velocity_mps(80.0, V_UA_B) + 5.0 * TOLERANCE_MPS,
            V_UA_A,
            V_UA_B,
            max_velocity_mps=MAX_VELOCITY_MPS,
            tolerance_mps=TOLERANCE_MPS,
        )
        assert np.isnan(velocity_mps)
        assert residual_mps > TOLERANCE_MPS

    def test_survives_noise_below_the_tolerance(self) -> None:
        rng = np.random.default_rng(20260911)
        true_mps = rng.uniform(-150.0, 150.0, 200)
        jitter_mps = rng.uniform(-0.2, 0.2, true_mps.shape)
        velocity_mps, _ = unfold_doppler_dual_prf(
            fold_velocity_mps(true_mps, V_UA_A),
            fold_velocity_mps(true_mps + jitter_mps, V_UA_B),
            V_UA_A,
            V_UA_B,
            max_velocity_mps=MAX_VELOCITY_MPS,
            tolerance_mps=TOLERANCE_MPS,
        )
        # The chosen candidate still comes from burst A, so it is exact; the
        # jitter only has to stay inside the tolerance to pick the right one.
        np.testing.assert_allclose(velocity_mps, true_mps, atol=1e-9)

    def test_pairs_every_burst_a_reading_with_every_burst_b_reading(self) -> None:
        """Catches broadcasting that pairs readings by position instead of by outer product.

        Three targets, burst A as a column and burst B as a row: the (3, 3)
        result is every A-B pairing. Only the diagonal pairs one target with
        itself, so only the diagonal resolves; every off-diagonal pairing misses
        burst B by more than the tolerance (by 1.1 m/s or more here) and is nan.
        """
        true_mps = np.array([10.0, 80.0, -120.0])
        velocity_mps, residual_mps = unfold_doppler_dual_prf(
            fold_velocity_mps(true_mps, V_UA_A)[:, None],
            fold_velocity_mps(true_mps, V_UA_B)[None, :],
            V_UA_A,
            V_UA_B,
            max_velocity_mps=MAX_VELOCITY_MPS,
            tolerance_mps=TOLERANCE_MPS,
        )
        assert residual_mps.shape == (3, 3)
        np.testing.assert_allclose(np.diag(velocity_mps), true_mps, atol=1e-9)
        off_diagonal = ~np.eye(3, dtype=bool)
        assert np.all(np.isnan(velocity_mps[off_diagonal]))

    @pytest.mark.parametrize(
        ("half_interval_a_mps", "half_interval_b_mps"),
        [(0.0, V_UA_B), (V_UA_A, 0.0)],
        ids=["burst-a", "burst-b"],
    )
    def test_rejects_a_zero_interval(
        self, half_interval_a_mps: float, half_interval_b_mps: float
    ) -> None:
        """Either burst's interval is checked, not only the first."""
        with pytest.raises(ValueError, match="strictly positive"):
            unfold_doppler_dual_prf(
                0.0,
                0.0,
                half_interval_a_mps,
                half_interval_b_mps,
                max_velocity_mps=MAX_VELOCITY_MPS,
                tolerance_mps=TOLERANCE_MPS,
            )

    def test_rejects_two_equal_rates(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            unfold_doppler_dual_prf(
                0.0,
                0.0,
                V_UA_A,
                V_UA_A,
                max_velocity_mps=MAX_VELOCITY_MPS,
                tolerance_mps=TOLERANCE_MPS,
            )

    def test_rejects_a_zero_tolerance(self) -> None:
        with pytest.raises(ValueError, match="tolerance_mps"):
            unfold_doppler_dual_prf(
                0.0,
                0.0,
                V_UA_A,
                V_UA_B,
                max_velocity_mps=MAX_VELOCITY_MPS,
                tolerance_mps=0.0,
            )

    def test_rejects_a_bound_below_the_smaller_interval(self) -> None:
        """Below it there is nothing folded to resolve, so the call is a mistake."""
        with pytest.raises(ValueError, match="no ambiguity to resolve"):
            unfold_doppler_dual_prf(
                0.0,
                0.0,
                V_UA_A,
                V_UA_B,
                max_velocity_mps=10.0,
                tolerance_mps=TOLERANCE_MPS,
            )


class TestUnfoldDualPeriod:
    """The period-agnostic solver both wrappers call."""

    def test_reaches_every_value_below_the_repeat_on_a_small_coprime_pair(self) -> None:
        """Periods 6 and 5 repeat at 30: every value in [0, 30) is recovered."""
        true = np.linspace(0.0, 29.9, 300)
        value, residual = unfold_dual_period(
            fold_into_period(true, 6.0, lower=0.0),
            fold_into_period(true, 5.0, lower=0.0),
            6.0,
            5.0,
            lower_bound=0.0,
            upper_bound=29.9,
            tolerance=0.1,
        )
        # Float error of a mod and a handful of adds on values below 30.
        np.testing.assert_allclose(value, true, atol=1e-9)
        np.testing.assert_allclose(residual, 0.0, atol=1e-9)

    def test_a_reading_reported_outside_the_bounds_is_still_resolved(self) -> None:
        """Reading A is brought into the bounds first, so how it was reported is moot."""
        value, _ = unfold_dual_period(
            4.0 + 6.0 * 7, 2.0, 6.0, 5.0, lower_bound=0.0, upper_bound=29.0, tolerance=0.1
        )
        np.testing.assert_allclose(value, 22.0, atol=1e-9)

    def test_equal_periods_resolve_within_one_period(self) -> None:
        """Equal periods add nothing, but agree on any value inside one period."""
        value, residual = unfold_dual_period(
            12.0, 12.0, 20.0, 20.0, lower_bound=0.0, upper_bound=20.0, tolerance=0.1
        )
        np.testing.assert_allclose(value, 12.0)
        np.testing.assert_allclose(residual, 0.0)

    def test_rejects_equal_periods_with_bounds_wider_than_one(self) -> None:
        with pytest.raises(ValueError, match="equal periods"):
            unfold_dual_period(
                0.0, 0.0, 20.0, 20.0, lower_bound=0.0, upper_bound=21.0, tolerance=0.1
            )

    @pytest.mark.parametrize(("period_a", "period_b"), [(0.0, 5.0), (6.0, 0.0)])
    def test_rejects_a_zero_period(self, period_a: float, period_b: float) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            unfold_dual_period(
                0.0, 0.0, period_a, period_b, lower_bound=0.0, upper_bound=10.0, tolerance=0.1
            )

    def test_rejects_a_zero_tolerance(self) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            unfold_dual_period(0.0, 0.0, 6.0, 5.0, lower_bound=0.0, upper_bound=10.0, tolerance=0.0)

    def test_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValueError, match="below lower_bound"):
            unfold_dual_period(0.0, 0.0, 6.0, 5.0, lower_bound=10.0, upper_bound=0.0, tolerance=0.1)


class TestUnfoldRangeDualPrf:
    """Range on S3's 60 km and 50 km maps, which repeat together at 300 km."""

    def test_reaches_the_full_span_the_lcm_of_the_two_spans_allows(self) -> None:
        true_m = np.linspace(0.0, REPEAT_M - 1_000.0, 1200)
        range_m, residual_m = unfold_range_dual_prf(
            fold_into_period(true_m, SPAN_A_M, lower=0.0),
            fold_into_period(true_m, SPAN_B_M, lower=0.0),
            SPAN_A_M,
            SPAN_B_M,
            max_range_m=REPEAT_M - 1_000.0,
            tolerance_m=TOLERANCE_M,
        )
        # A mod and a few adds on values below 3e5 m: 1e-6 m is generous.
        np.testing.assert_allclose(range_m, true_m, atol=1e-6)
        np.testing.assert_allclose(residual_m, 0.0, atol=1e-6)

    def test_a_target_past_the_shorter_span_reads_differently_on_each_map(self) -> None:
        """55 km reads 55 km on the 60 km map and 5 km on the 50 km map.

        Comparing the two readings modulo the shorter span pairs them, but it
        cannot say whether the target is at 5 or 55 km. The unfolding can.
        """
        range_m, residual_m = unfold_range_dual_prf(
            55_000.0, 5_000.0, SPAN_A_M, SPAN_B_M, max_range_m=SPAN_A_M, tolerance_m=TOLERANCE_M
        )
        np.testing.assert_allclose(range_m, 55_000.0)
        np.testing.assert_allclose(residual_m, 0.0, atol=1e-6)

    def test_a_target_outside_the_bound_can_alias_to_a_confident_wrong_answer(self) -> None:
        """310 km folds exactly as 10 km does, so a 300 km bound returns 10 km."""
        true_m = 310_000.0
        range_m, residual_m = unfold_range_dual_prf(
            fold_into_period(true_m, SPAN_A_M, lower=0.0),
            fold_into_period(true_m, SPAN_B_M, lower=0.0),
            SPAN_A_M,
            SPAN_B_M,
            max_range_m=REPEAT_M,
            tolerance_m=TOLERANCE_M,
        )
        np.testing.assert_allclose(range_m, true_m - REPEAT_M, atol=1e-6)
        assert residual_m < TOLERANCE_M

    def test_a_gap_is_measured_round_the_second_maps_circle(self) -> None:
        """20 m below the 50 km wrap and 30 m past it are 50 m apart, not 49.95 km."""
        _, residual_m = unfold_range_dual_prf(
            49_980.0, 30.0, SPAN_A_M, SPAN_B_M, max_range_m=SPAN_A_M, tolerance_m=TOLERANCE_M
        )
        np.testing.assert_allclose(residual_m, 50.0, atol=1e-6)

    def test_pairs_every_burst_a_reading_with_every_burst_b_reading(self) -> None:
        """A column against a row gives the (n_a, n_b) matrix dual_prf_detections uses."""
        true_m = np.array([15_000.0, 55_000.0, 120_000.0])
        range_m, residual_m = unfold_range_dual_prf(
            fold_into_period(true_m, SPAN_A_M, lower=0.0)[:, None],
            fold_into_period(true_m, SPAN_B_M, lower=0.0)[None, :],
            SPAN_A_M,
            SPAN_B_M,
            max_range_m=REPEAT_M - 1_000.0,
            tolerance_m=TOLERANCE_M,
        )
        assert residual_m.shape == (3, 3)
        np.testing.assert_allclose(np.diag(range_m), true_m, atol=1e-6)

    def test_rejects_a_negative_bound(self) -> None:
        with pytest.raises(ValueError, match="max_range_m"):
            unfold_range_dual_prf(
                0.0, 0.0, SPAN_A_M, SPAN_B_M, max_range_m=-1.0, tolerance_m=TOLERANCE_M
            )
