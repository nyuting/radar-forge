"""Tests for dual-PRF range and Doppler ambiguity resolution.

Ground truth here is closed form and needs no simulation: folding is a known
modular map, so the strongest available check is to fold a swept set of true
velocities and require the unfolder to return exactly what went in. The
scenario 001 S3 half-intervals are used throughout, because the property that
matters is not that the algorithm works for some pair of rates but that it
works for the 5:6 pair the scenario actually ships.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.ambiguity import fold_velocity_mps, unfold_doppler_dual_prf

# Scenario 001 S3: FMCW bursts at 5.0 kHz and 6.0 kHz, per spec S4.
V_UA_A = 38.24
V_UA_B = 45.89
MAX_VELOCITY_MPS = 191.0
TOLERANCE_MPS = 1.0

# Scenario 001 S1: the heavily folded low-PRF FMCW burst.
S1_V_UA = 7.65


class TestFoldVelocityMps:
    def test_leaves_an_unambiguous_velocity_alone(self) -> None:
        np.testing.assert_allclose(fold_velocity_mps(3.0, S1_V_UA), 3.0, rtol=1e-15)

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

    def test_folding_is_periodic_in_the_full_span(self) -> None:
        """Adding one span changes nothing — the defining property of the map."""
        velocity_mps = np.linspace(-100.0, 100.0, 401)
        np.testing.assert_allclose(
            fold_velocity_mps(velocity_mps + 2.0 * S1_V_UA, S1_V_UA),
            fold_velocity_mps(velocity_mps, S1_V_UA),
            atol=1e-12,  # near the wrap the two branches differ by float epsilon only
        )

    def test_preserves_shape(self) -> None:
        assert fold_velocity_mps(np.zeros((3, 4)), S1_V_UA).shape == (3, 4)

    @pytest.mark.parametrize("bad_half_interval_mps", [0.0, -1.0])
    def test_rejects_a_non_positive_interval(self, bad_half_interval_mps: float) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            fold_velocity_mps(10.0, bad_half_interval_mps)


class TestUnfoldDopplerDualPrf:
    def test_recovers_every_velocity_the_prf_pair_can_reach(self) -> None:
        """The closed-form round trip, swept across the whole reachable span.

        Five folds of burst A is 5 * 38.24 = 191.2 m/s, so a sweep stopping just
        inside that is exactly the set the 5:6 ratio is supposed to cover.
        """
        true_mps = np.linspace(-190.0, 190.0, 761)
        velocity_mps, residual_mps = unfold_doppler_dual_prf(
            fold_velocity_mps(true_mps, V_UA_A),
            fold_velocity_mps(true_mps, V_UA_B),
            V_UA_A,
            V_UA_B,
            max_velocity_mps=MAX_VELOCITY_MPS,
            tolerance_mps=TOLERANCE_MPS,
        )
        # Exact arithmetic recovery: the candidate is the true value to within
        # the float error of a handful of adds, so 1e-9 is generous.
        np.testing.assert_allclose(velocity_mps, true_mps, atol=1e-9)
        np.testing.assert_allclose(residual_mps, 0.0, atol=1e-9)

    def test_a_slow_target_needs_no_unfolding(self) -> None:
        velocity_mps, _ = unfold_doppler_dual_prf(
            fold_velocity_mps(12.0, V_UA_A),
            fold_velocity_mps(12.0, V_UA_B),
            V_UA_A,
            V_UA_B,
            max_velocity_mps=MAX_VELOCITY_MPS,
            tolerance_mps=TOLERANCE_MPS,
        )
        np.testing.assert_allclose(velocity_mps, 12.0, atol=1e-9)

    def test_reaches_the_full_span_the_lcm_of_the_two_spans_allows(self) -> None:
        """The limit is the least common multiple of the folding spans.

        For the 5:6 bursts that is 6 * 2 * 38.24 = 458.88 m/s, so a half-span of
        about 229 m/s is recoverable — more than the +/-191.2 m/s the scenario
        asks for, which is a design target chosen to match S2 rather than this
        function's ceiling.
        """
        true_mps = np.linspace(-225.0, 225.0, 901)
        velocity_mps, _ = unfold_doppler_dual_prf(
            fold_velocity_mps(true_mps, V_UA_A),
            fold_velocity_mps(true_mps, V_UA_B),
            V_UA_A,
            V_UA_B,
            max_velocity_mps=229.0,
            tolerance_mps=TOLERANCE_MPS,
        )
        np.testing.assert_allclose(velocity_mps, true_mps, atol=1e-9)

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
        np.testing.assert_allclose(velocity_mps, true_mps - alias_period_mps, atol=0.1)
        assert residual_mps < TOLERANCE_MPS

    def test_a_target_outside_the_bound_with_no_alias_is_unresolved(self) -> None:
        """250 m/s has no in-bound alias, so it is reported as nan, not guessed."""
        velocity_mps, residual_mps = unfold_doppler_dual_prf(
            fold_velocity_mps(250.0, V_UA_A),
            fold_velocity_mps(250.0, V_UA_B),
            V_UA_A,
            V_UA_B,
            max_velocity_mps=MAX_VELOCITY_MPS,
            tolerance_mps=TOLERANCE_MPS,
        )
        assert np.isnan(velocity_mps)
        assert residual_mps > TOLERANCE_MPS

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

    def test_broadcasts_the_two_bursts_against_each_other(self) -> None:
        true_mps = np.array([[10.0, 80.0, -120.0]])
        velocity_mps, residual_mps = unfold_doppler_dual_prf(
            fold_velocity_mps(true_mps, V_UA_A),
            fold_velocity_mps(true_mps, V_UA_B),
            V_UA_A,
            V_UA_B,
            max_velocity_mps=MAX_VELOCITY_MPS,
            tolerance_mps=TOLERANCE_MPS,
        )
        assert velocity_mps.shape == (1, 3)
        assert residual_mps.shape == (1, 3)
        np.testing.assert_allclose(velocity_mps, true_mps, atol=1e-9)

    @pytest.mark.parametrize("bad_half_interval_mps", [0.0, -1.0])
    def test_rejects_a_non_positive_interval(self, bad_half_interval_mps: float) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            unfold_doppler_dual_prf(
                0.0,
                0.0,
                bad_half_interval_mps,
                V_UA_B,
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

    @pytest.mark.parametrize("bad_tolerance_mps", [0.0, -1.0])
    def test_rejects_a_non_positive_tolerance(self, bad_tolerance_mps: float) -> None:
        with pytest.raises(ValueError, match="tolerance_mps"):
            unfold_doppler_dual_prf(
                0.0,
                0.0,
                V_UA_A,
                V_UA_B,
                max_velocity_mps=MAX_VELOCITY_MPS,
                tolerance_mps=bad_tolerance_mps,
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
