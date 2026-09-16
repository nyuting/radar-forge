"""Tests for radar_forge.core.tracking.

Ground truth here is analytic, per docs/conventions/testing.md §3: a noiseless
constant-velocity target must be tracked with zero steady-state innovation to
float precision; the gate thresholds must match tabulated chi-squared
quantiles; the process noise must reproduce the worked matrix in Bar-Shalom
§5.2; and the assignment must return the known optimum of a hand-built cost
matrix with a unique solution.

The consistency tests matter more than they look. A filter with a mis-scaled R
or Q still produces a track that follows the target and a plot that looks
entirely plausible; what it gets wrong is its own confidence, and the
normalised innovation is the only thing here that notices.
"""

import numpy as np
import pytest
from scipy.stats import beta

from radar_forge.core.tracking import (
    STATE_MODELS,
    KalmanState,
    Track,
    TrackManager,
    associate_gnn,
    gate_threshold,
    innovation_of,
    normalised_innovation_squared,
    predict,
    process_noise_dwna,
    state_model_matrices,
    update,
)

# Scenario 003's numbers, so the unit tests exercise the conditioning the
# scenario actually runs at -- in particular the five orders of magnitude
# between the two measurement variances, which is where a transposed R hides.
FRAME_TIME_S = 1.0
SIGMA_RANGE_M = 21.635652855125496
SIGMA_VELOCITY_MPS = 0.017247813329811023
SIGMA_ACCEL_MPS2 = 2.0
V_MAX_MPS = 200.0


@pytest.fixture
def rng():
    """One seeded generator, so a failure is reproducible."""
    return np.random.default_rng(20260915)


@pytest.fixture
def model():
    """The shipped range_1d model at the scenario's own parameters."""
    return state_model_matrices(
        "range_1d",
        FRAME_TIME_S,
        sigma_accel_mps2=SIGMA_ACCEL_MPS2,
        sigma_range_m=SIGMA_RANGE_M,
        sigma_velocity_mps=SIGMA_VELOCITY_MPS,
    )


def initial_covariance():
    """P0: measured components from R, unmeasured rate at v_max squared."""
    return np.diag([SIGMA_RANGE_M**2, V_MAX_MPS**2])


# --------------------------------------------------------------------------- #
# Process noise
# --------------------------------------------------------------------------- #


class TestProcessNoiseDwna:
    """The discrete white-noise acceleration model of Bar-Shalom §5.2."""

    def test_matches_the_worked_matrix(self):
        """Reproduces sigma_a^2 * [[T^4/4, T^3/2], [T^3/2, T^2]] exactly."""
        step_s, sigma = 2.0, 3.0
        q = process_noise_dwna(step_s, sigma)
        expected = sigma**2 * np.array(
            [[step_s**4 / 4.0, step_s**3 / 2.0], [step_s**3 / 2.0, step_s**2]]
        )
        # rtol 1e-15: a handful of float64 products, so anything looser would
        # hide a genuine algebra error in the powers of T.
        np.testing.assert_allclose(q, expected, rtol=1e-15)

    def test_each_axis_block_is_rank_one(self):
        """One scalar acceleration drives two states, so a block is singular."""
        assert np.linalg.matrix_rank(process_noise_dwna(1.0, 2.0)) == 1

    @pytest.mark.parametrize("n_axes", [1, 2, 3])
    def test_axes_are_independent_and_position_major(self, n_axes):
        """No cross-axis terms, and the state is [positions..., rates...]."""
        q = process_noise_dwna(1.5, 2.0, n_axes)
        assert q.shape == (2 * n_axes, 2 * n_axes)
        np.testing.assert_allclose(q, q.T, rtol=1e-15)
        # Within the position block, off-diagonal entries couple two axes and
        # must be zero; the diagonal is the shared T^4/4 term.
        position_block = q[:n_axes, :n_axes]
        np.testing.assert_allclose(position_block, np.diag(np.diag(position_block)), atol=1e-18)

    def test_is_positive_semi_definite(self):
        """A covariance that is not would break the filter silently."""
        eigenvalues = np.linalg.eigvalsh(process_noise_dwna(1.0, 2.0, 2))
        assert eigenvalues.min() > -1e-12

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"frame_time_s": 0.0}, "frame_time_s"),
            ({"sigma_accel_mps2": -1.0}, "sigma_accel_mps2"),
            ({"n_axes": 0}, "n_axes"),
        ],
    )
    def test_rejects_impossible_parameters(self, kwargs, match):
        defaults = {"frame_time_s": 1.0, "sigma_accel_mps2": 2.0, "n_axes": 1}
        with pytest.raises(ValueError, match=match):
            process_noise_dwna(**{**defaults, **kwargs})


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


class TestGateThreshold:
    """A statistical statement about the filter's uncertainty, not a distance."""

    @pytest.mark.parametrize(
        ("dim", "expected"),
        [(1, 6.635), (2, 9.210), (3, 11.345), (4, 13.277)],
    )
    def test_matches_the_tabulated_quantiles(self, dim, expected):
        """The table in scenario 003 §7, which the gate must not hard-code.

        atol 5e-4 because the specification tabulates three decimals.
        """
        assert gate_threshold(0.99, dim) == pytest.approx(expected, abs=5e-4)

    def test_grows_with_dimension(self):
        """More measurement components means a larger admissible d^2."""
        thresholds = [gate_threshold(0.99, dim) for dim in (1, 2, 3, 4)]
        assert thresholds == sorted(thresholds)

    def test_empirical_acceptance_rate_matches_the_design(self, rng):
        """The gate must actually admit the mass it claims to.

        A binomial confidence interval rather than a tolerance, per
        docs/conventions/testing.md §2: this is a Monte-Carlo statistic.
        """
        dim, probability, n_trials = 2, 0.99, 20_000
        threshold = gate_threshold(probability, dim)
        # d^2 of a correctly-modelled innovation is chi-squared(dim) by
        # construction, so draw from it directly rather than through a filter.
        accepted = int((rng.chisquare(dim, size=n_trials) <= threshold).sum())

        # An exact Clopper-Pearson interval, as test_detection.py uses.
        low = beta.ppf(0.0005, accepted, n_trials - accepted + 1)
        high = beta.ppf(0.9995, accepted + 1, n_trials - accepted)
        assert low <= probability <= high

    @pytest.mark.parametrize(("probability", "dim"), [(0.0, 2), (1.0, 2), (-0.1, 2), (0.99, 0)])
    def test_rejects_impossible_parameters(self, probability, dim):
        with pytest.raises(ValueError):
            gate_threshold(probability, dim)


# --------------------------------------------------------------------------- #
# Filter algebra, against analytic truth
# --------------------------------------------------------------------------- #


class TestNoiselessConstantVelocity:
    """The case with an exact answer: no process noise, no measurement error."""

    def test_prediction_is_exact_dead_reckoning(self):
        noiseless = state_model_matrices(
            "range_1d",
            FRAME_TIME_S,
            sigma_accel_mps2=0.0,
            sigma_range_m=SIGMA_RANGE_M,
            sigma_velocity_mps=SIGMA_VELOCITY_MPS,
        )
        # Positive range rate is closing, so the range falls.
        state = KalmanState(np.array([10_000.0, 80.0]), np.zeros((2, 2)))
        for step in range(1, 11):
            state = predict(state, noiseless)
            np.testing.assert_allclose(state.state, [10_000.0 - 80.0 * step, 80.0], rtol=1e-12)

    def test_a_closing_target_loses_range(self, model):
        """The sign convention, asserted rather than assumed.

        Regression for the one bug in this module that produced a plausible
        picture: scenario 003 §5.2 defines the measurement as positive closing,
        per spec/structure.md D5, but §6 pairs it with a transition of
        [[1, T], [0, 1]], which makes a closing target's range *grow*. The
        filter then dead-reckons the wrong way down the line of sight and every
        Doppler fold selected from its prediction is wrong, while the range
        track still looks roughly right for a few frames.
        """
        closing = predict(KalmanState(np.array([10_000.0, 80.0]), np.eye(2)), model)
        assert closing.state[0] < 10_000.0

        opening = predict(KalmanState(np.array([10_000.0, -80.0]), np.eye(2)), model)
        assert opening.state[0] > 10_000.0

    def test_the_process_noise_correlates_range_and_closing_rate_negatively(self, model):
        """A closing acceleration reduces range, so the cross term is negative."""
        assert model.process_noise[0, 1] < 0.0
        assert model.process_noise[0, 1] == pytest.approx(model.process_noise[1, 0])

    def test_innovation_is_zero_to_float_precision(self, model):
        """A perfect measurement of a perfectly predicted target.

        This is the specification's unit-level ground truth: any sign error in
        the transition or the measurement model shows up here immediately.
        """
        noiseless = state_model_matrices(
            "range_1d",
            FRAME_TIME_S,
            sigma_accel_mps2=0.0,
            sigma_range_m=SIGMA_RANGE_M,
            sigma_velocity_mps=SIGMA_VELOCITY_MPS,
        )
        range_m, closing_mps = 10_000.0, 80.0
        state = KalmanState(np.array([range_m, closing_mps]), initial_covariance())

        for step in range(1, 21):
            state = predict(state, noiseless)
            truth = np.array([range_m - closing_mps * step, closing_mps])
            result = update(state, truth, noiseless)
            # atol, not rtol: the quantity being checked is zero.
            np.testing.assert_allclose(result.innovation, 0.0, atol=1e-6)
            state = result.posterior

    def test_update_shrinks_every_variance(self, model):
        """Information can only be gained by measuring."""
        prior = KalmanState(np.array([10_000.0, 80.0]), initial_covariance())
        posterior = update(prior, np.array([10_000.0, 80.0]), model).posterior
        assert np.all(np.diag(posterior.covariance) <= np.diag(prior.covariance))

    def test_covariance_stays_symmetric_and_positive_definite(self, model, rng):
        """The Joseph form's reason for existing, at this R's conditioning."""
        state = KalmanState(np.array([10_000.0, 80.0]), initial_covariance())
        for _ in range(200):
            state = predict(state, model)
            measurement = np.array([10_000.0 + rng.normal(0.0, SIGMA_RANGE_M), 80.0])
            state = update(state, measurement, model).posterior
            np.testing.assert_allclose(state.covariance, state.covariance.T, rtol=1e-12)
            assert np.linalg.eigvalsh(state.covariance).min() > 0.0


class TestNormalisedInnovationSquared:
    """The statistic the gate tests and the consistency criterion averages."""

    def test_matches_the_scalar_case(self):
        """With one component it is simply (nu / sigma)^2."""
        assert normalised_innovation_squared([4.0], [[4.0]]) == pytest.approx(4.0)

    def test_is_invariant_to_the_units_of_each_component(self):
        """Scaling a component and its variance together cannot change d^2.

        This is the property that makes the five-orders-of-magnitude spread
        between the range and velocity variances harmless -- and the one a
        transposed R violates.
        """
        innovation = np.array([10.0, 0.02])
        covariance = np.diag([SIGMA_RANGE_M**2, SIGMA_VELOCITY_MPS**2])
        scale = np.diag([1e3, 1.0])
        scaled = normalised_innovation_squared(scale @ innovation, scale @ covariance @ scale.T)
        assert scaled == pytest.approx(
            normalised_innovation_squared(innovation, covariance), rel=1e-12
        )

    def test_is_chi_squared_distributed_under_a_correct_model(self, rng):
        """The consistency check, in its purest form.

        Draws innovations from the very covariance the statistic is told about,
        so the mean d^2 must be the dimension. Asserted as a confidence
        interval on the mean of n_trials chi-squared(dim) draws.
        """
        dim, n_trials = 2, 5_000
        covariance = np.diag([SIGMA_RANGE_M**2, SIGMA_VELOCITY_MPS**2])
        cholesky = np.linalg.cholesky(covariance)
        draws = [
            normalised_innovation_squared(cholesky @ rng.standard_normal(dim), covariance)
            for _ in range(n_trials)
        ]
        # The mean of n chi2(dim) variables has variance 2*dim/n.
        standard_error = np.sqrt(2.0 * dim / n_trials)
        assert abs(float(np.mean(draws)) - dim) < 4.0 * standard_error

    def test_rejects_a_singular_covariance(self):
        with pytest.raises(ValueError, match="singular"):
            normalised_innovation_squared([1.0, 1.0], np.zeros((2, 2)))

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="do not agree"):
            normalised_innovation_squared([1.0, 2.0], [[1.0]])


class TestMeasurementRestriction:
    """The §5.3 bootstrap: the same model, measuring fewer components."""

    def test_restricting_to_range_only_drops_the_velocity_row(self, model):
        restricted = model.restricted(1)
        assert restricted.measurement_dim == 1
        assert restricted.measurement_noise.shape == (1, 1)
        assert restricted.measurement_noise[0, 0] == pytest.approx(SIGMA_RANGE_M**2)
        assert restricted.measurement_jacobian(np.zeros(2)).shape == (1, 2)

    def test_the_full_dimension_returns_the_same_model(self, model):
        assert model.restricted(model.measurement_dim) is model

    def test_a_range_only_update_leaves_the_rate_uncorrelated_at_first(self, model):
        """With a diagonal P0, one range measurement cannot inform the rate."""
        prior = KalmanState(np.array([10_000.0, 0.0]), initial_covariance())
        posterior = update(prior, np.array([10_100.0]), model.restricted(1)).posterior
        assert posterior.state[1] == pytest.approx(0.0, abs=1e-9)
        assert posterior.covariance[1, 1] == pytest.approx(V_MAX_MPS**2, rel=1e-12)

    def test_range_only_tracking_still_converges_on_the_rate(self, model):
        """Over frames, the range slope is what resolves the velocity.

        The specification's §5.3 bootstrap depends on exactly this: a track
        with no velocity measurement must still reach a rate variance small
        enough to select a Doppler fold, within about ten frames.
        """
        closing_mps = 80.0
        state = KalmanState(np.array([20_000.0, 0.0]), initial_covariance())
        restricted = model.restricted(1)
        for step in range(1, 16):
            state = predict(state, model)
            state = update(state, np.array([20_000.0 - closing_mps * step]), restricted).posterior
        assert state.state[1] == pytest.approx(closing_mps, abs=5.0)

    def test_the_rate_variance_floors_above_the_bootstrap_threshold(self, model):
        """Why the unfold test cannot be the filter's own rate covariance.

        Scenario 003 §5.3 proposes promoting a track out of its range-only
        bootstrap once ``sqrt(P[rate, rate]) < v_span / 6`` = 2.55 m/s, and
        derives a ten-frame crossing from a least-squares range slope. But the
        filter it specifies carries ``sigma_accel_mps2 = 2.0``, and discrete
        white-noise acceleration adds ``sigma_a^2 T^2`` to the rate variance
        every frame. The rate standard deviation therefore converges to a floor
        of about 4.1 m/s and never reaches 2.55 -- the track would stay
        range-only for ever and never unfold.

        The batch estimator §5.3 actually computes is not floored this way, so
        that is the one the pipeline uses. This test pins the floor, because if
        it ever drops below the threshold the pipeline could use the simpler
        covariance test instead.
        """
        state = KalmanState(np.array([20_000.0, 0.0]), initial_covariance())
        restricted = model.restricted(1)
        for step in range(1, 61):
            state = predict(state, model)
            state = update(state, np.array([20_000.0 - 80.0 * step]), restricted).posterior
        assert np.sqrt(state.covariance[1, 1]) > 15.2955 / 6.0

    @pytest.mark.parametrize("n_rows", [0, 3, -1])
    def test_rejects_an_impossible_restriction(self, model, n_rows):
        with pytest.raises(ValueError, match="n_rows"):
            model.restricted(n_rows)


# --------------------------------------------------------------------------- #
# Association
# --------------------------------------------------------------------------- #


class TestAssociateGnn:
    """Global nearest neighbour over a hand-built cost matrix."""

    def test_returns_the_known_optimum(self):
        """Greedy would take (0, 1) first and score 1.0 + 8.0; GNN scores 3.0."""
        cost = [[2.0, 1.0], [8.0, 2.0]]
        assert associate_gnn(cost, 9.21) == [(0, 0), (1, 1)]

    def test_never_assigns_a_pair_outside_the_gate(self):
        cost = [[1.0, 50.0], [50.0, 50.0]]
        assert associate_gnn(cost, 9.21) == [(0, 0)]

    def test_returns_nothing_when_every_pair_is_gated_out(self):
        assert associate_gnn([[100.0, 200.0]], 9.21) == []

    def test_handles_an_empty_frame(self):
        assert associate_gnn(np.zeros((0, 0)), 9.21) == []
        assert associate_gnn(np.zeros((2, 0)), 9.21) == []

    def test_assigns_each_track_and_measurement_at_most_once(self):
        cost = [[1.0, 1.1, 1.2], [1.3, 1.0, 1.1]]
        pairs = associate_gnn(cost, 9.21)
        assert len({track for track, _ in pairs}) == len(pairs)
        assert len({measurement for _, measurement in pairs}) == len(pairs)

    def test_more_measurements_than_tracks_leaves_some_unassociated(self):
        """The scenario's normal case: one target, several false alarms."""
        pairs = associate_gnn([[0.5, 2.0, 3.0]], 9.21)
        assert pairs == [(0, 0)]

    def test_rejects_a_non_positive_gate(self):
        with pytest.raises(ValueError, match="gate"):
            associate_gnn([[1.0]], 0.0)


# --------------------------------------------------------------------------- #
# Track management
# --------------------------------------------------------------------------- #


def manager(model, **kwargs):
    """A TrackManager at the scenario's own M-of-N parameters."""
    return TrackManager(model=model, initial_covariance=initial_covariance(), **kwargs)


class TestTrackManagerInitiation:
    """M-of-N confirmation, sized in §3 against the false-alarm rate."""

    def test_a_steady_target_is_confirmed_on_the_fourth_hit(self, model):
        tracker = manager(model)
        for step in range(5):
            tracker.step([np.array([10_000.0 - 80.0 * step, 80.0])])
            statuses = [track.status for track in tracker.tracks]
            # Frame 0 initiates; frames 1-3 are hits two, three and four.
            expected = "confirmed" if step >= 3 else "tentative"
            assert expected in statuses

    def test_the_track_id_never_changes(self, model):
        tracker = manager(model)
        for step in range(10):
            tracker.step([np.array([10_000.0 - 80.0 * step, 80.0])])
        confirmed = tracker.confirmed_tracks
        assert len(confirmed) == 1
        assert confirmed[0].track_id == 1

    def test_an_isolated_false_alarm_is_deleted_and_never_confirmed(self, model):
        """One measurement, then nothing: the clutter case of §3."""
        tracker = manager(model)
        tracker.step([np.array([30_000.0, 0.0])])
        for _ in range(4):
            tracker.step([])
        assert tracker.tracks == []
        assert tracker.confirmed_tracks == []

    def test_a_tentative_track_dies_as_soon_as_m_is_unreachable(self, model):
        """Two misses in the first five frames make 4-of-5 arithmetically dead."""
        tracker = manager(model)
        tracker.step([np.array([30_000.0, 0.0])])
        tracker.step([])
        tracker.step([])
        assert tracker.tracks == []


class TestTrackManagerMaintenance:
    """Coasting and deletion, the logic the default link budget never exercises."""

    def _confirmed(self, model):
        tracker = manager(model)
        for step in range(4):
            tracker.step([np.array([10_000.0 - 80.0 * step, 80.0])])
        assert tracker.confirmed_tracks
        return tracker

    def test_a_confirmed_track_coasts_through_a_miss(self, model):
        tracker = self._confirmed(model)
        before = tracker.confirmed_tracks[0].estimate.state[0]
        tracker.step([])
        track = tracker.confirmed_tracks[0]
        assert track.status == "coasting"
        # Dead reckoning: one frame of the estimated rate.
        assert track.estimate.state[0] == pytest.approx(before - 80.0, abs=5.0)

    def test_coasting_grows_the_covariance(self, model):
        tracker = self._confirmed(model)
        before = tracker.confirmed_tracks[0].estimate.covariance[0, 0]
        tracker.step([])
        assert tracker.confirmed_tracks[0].estimate.covariance[0, 0] > before

    def test_a_confirmed_track_survives_three_misses_but_forgets_its_rate(self, model):
        """The grace period: three misses widen the gate rather than delete.

        A run of misses on this scenario is a run of gate rejections caused by
        a wrong Doppler fold, not a run of missed detections, so the track is
        given a chance to re-acquire the target it can still see.
        """
        tracker = self._confirmed(model)
        before = tracker.confirmed_tracks[0].estimate.covariance[1, 1]
        for _ in range(3):
            tracker.step([])
        track = tracker.confirmed_tracks[0]
        assert track.estimate.covariance[1, 1] > before
        assert track.estimate.covariance[1, 1] == pytest.approx(V_MAX_MPS**2, rel=1e-9)
        assert track.track_id in tracker.lost_track_ids

    def test_a_widened_track_re_acquires_its_target_and_keeps_its_id(self, model):
        """§12 criterion 1: the id must span the run, not restart after a loss."""
        tracker = self._confirmed(model)
        original = tracker.confirmed_tracks[0].track_id
        for _ in range(3):
            tracker.step([])
        # The target reappears where constant velocity says it should be.
        tracker.step([np.array([10_000.0 - 80.0 * 7, 80.0])])
        confirmed = tracker.confirmed_tracks
        assert len(confirmed) == 1
        assert confirmed[0].track_id == original
        assert confirmed[0].n_misses_in_a_row == 0

    def test_a_confirmed_track_is_deleted_once_the_grace_period_expires(self, model):
        tracker = self._confirmed(model)
        for _ in range(tracker.n_delete_misses + tracker.n_reacquire_frames):
            tracker.step([])
        assert tracker.confirmed_tracks == []
        assert tracker.tracks == []

    def test_re_acquisition_can_be_switched_off(self, model):
        tracker = manager(model, n_reacquire_frames=0)
        for step in range(4):
            tracker.step([np.array([10_000.0 - 80.0 * step, 80.0])])
        for _ in range(3):
            tracker.step([])
        assert tracker.tracks == []

    def test_a_hit_resets_the_miss_count(self, model):
        tracker = self._confirmed(model)
        tracker.step([])
        tracker.step([])
        # The target is still where constant velocity says it is: frames 0-3
        # were hits, 4 and 5 were misses, so this is frame 6.
        tracker.step([np.array([10_000.0 - 80.0 * 6, 80.0])])
        survivor = tracker.confirmed_tracks[0]
        assert survivor.n_misses_in_a_row == 0
        assert survivor.status == "confirmed"


class TestVelocityGateFallback:
    """A measurement that fails only on velocity is still a good range fix."""

    def _confirmed(self, model):
        tracker = manager(model)
        for step in range(4):
            tracker.step([np.array([10_000.0 - 80.0 * step, 80.0])])
        return tracker

    def test_a_wrong_velocity_still_updates_the_range(self, model):
        """A mis-unfolded measurement misses by a whole fold span in velocity.

        Coasting on it throws away a range measurement as good as any other
        frame's, and frees the detection to seed a rival track. The track
        updates on range alone instead, and records the lower dimension.
        """
        tracker = self._confirmed(model)
        before = tracker.confirmed_tracks[0].estimate.state[0]
        # Correct range, velocity wrong by one 15.2955 m/s fold span.
        tracker.step([np.array([10_000.0 - 80.0 * 4, 80.0 - 15.2955])])
        track = tracker.confirmed_tracks[0]
        assert track.status == "confirmed"
        assert track.measurement_dim == 1
        assert track.n_misses_in_a_row == 0
        assert abs(track.estimate.state[0] - (10_000.0 - 320.0)) < abs(before - (10_000.0 - 320.0))

    def test_a_measurement_wrong_in_range_is_still_rejected(self, model):
        """The fallback must not become a gate that admits anything."""
        tracker = self._confirmed(model)
        tracker.step([np.array([25_000.0, 80.0])])
        assert tracker.confirmed_tracks[0].n_misses_in_a_row == 1


class TestTrackManagerAgainstClutter:
    """One target plus false alarms, which is scenario 003's whole association load."""

    def test_the_target_keeps_its_track_through_clutter(self, model, rng):
        tracker = manager(model)
        range_m, closing_mps = 15_000.0, 40.0
        for step in range(20):
            truth = np.array([range_m - closing_mps * step, closing_mps])
            noise = np.array([rng.normal(0.0, SIGMA_RANGE_M), 0.0])
            measurements = [truth + noise]
            # Two or three false alarms per frame, as §3 sizes them, scattered
            # far enough away that they cannot enter a mature track's gate.
            measurements += [
                np.array([rng.uniform(1_000.0, 35_000.0), rng.uniform(-7.6, 7.6)]) for _ in range(3)
            ]
            rng.shuffle(measurements)
            tracker.step(measurements)
            if step == 5:
                target_id = min(
                    tracker.confirmed_tracks,
                    key=lambda t: abs(t.estimate.state[0] - (range_m - closing_mps * step)),
                ).track_id

        confirmed = tracker.confirmed_tracks
        assert len(confirmed) >= 1
        target = min(
            confirmed, key=lambda t: abs(t.estimate.state[0] - (range_m - closing_mps * 19))
        )
        assert target.estimate.state[0] == pytest.approx(range_m - closing_mps * 19, abs=75.0)
        # Criterion 1: the identity is stable. Which integer it is depends on
        # the shuffled order of the first frame's measurements and means nothing.
        assert target.track_id == target_id
        assert target.n_hits >= 19

    def test_a_frame_with_no_measurements_is_harmless(self, model):
        tracker = manager(model)
        assert tracker.step([]).tracks == ()


class TestTentativeTrackDeletion:
    """The two exits a tentative track has that are not confirmation."""

    def test_a_tentative_track_that_misses_is_deleted_with_no_grace_period(self, model):
        """Only a confirmed track earns the coast-and-re-acquire grace period.

        A tentative track is an unproven hypothesis, most often a false alarm,
        so a miss deletes it outright rather than widening its gate -- which is
        what keeps clutter from accumulating gates that swallow real
        measurements.
        """
        tracker = manager(model, n_delete_misses=1)
        tracker.step([np.array([10_000.0, 80.0])])
        assert tracker.tracks[0].status == "tentative"
        tracker.step([])
        # A deleted track is dropped from the manager's list, so its absence
        # is the observable: one miss took the hypothesis out of the world.
        assert tracker.tracks == []
        assert not tracker.confirmed_tracks

    def test_a_tentative_track_out_of_frames_is_deleted_on_the_last_one(self, model):
        """M-of-N is a deadline: three hits in five frames is a deleted track.

        Distinct from the early-deletion shortcut, which fires as soon as the
        remaining frames cannot reach M. Here the arithmetic stays alive to the
        final frame and the deadline itself does the deleting, which is the
        branch that decides whether N means anything at all.
        """
        tracker = manager(model)
        for step in range(3):
            tracker.step([np.array([10_000.0 - 80.0 * step, 80.0])])
        assert tracker.tracks[0].n_hits == 3
        # Two misses: after the first, 3 hits plus 1 remaining frame still
        # reaches M = 4, so the track survives; the second exhausts N = 5.
        tracker.step([])
        assert tracker.tracks[0].status == "tentative"
        tracker.step([])
        assert tracker.tracks == []


class TestValidation:
    """The documented failure modes."""

    def test_state_models_lists_what_can_be_built(self):
        assert "range_1d" in STATE_MODELS

    def test_rejects_an_unknown_state_model(self):
        with pytest.raises(ValueError, match="state_model"):
            state_model_matrices(
                "enu_9d",  # type: ignore[arg-type]
                1.0,
                sigma_accel_mps2=2.0,
                sigma_range_m=1.0,
                sigma_velocity_mps=1.0,
            )

    def test_rejects_a_negative_measurement_sigma(self):
        with pytest.raises(ValueError, match="sigma_range_m"):
            state_model_matrices(
                "range_1d",
                1.0,
                sigma_accel_mps2=2.0,
                sigma_range_m=-1.0,
                sigma_velocity_mps=1.0,
            )

    def test_rejects_a_measurement_of_the_wrong_length(self, model):
        state = KalmanState(np.zeros(2), np.eye(2))
        with pytest.raises(ValueError, match="measurement must have shape"):
            innovation_of(state, np.array([1.0, 2.0, 3.0]), model)

    def test_rejects_impossible_m_of_n(self, model):
        with pytest.raises(ValueError, match="n_confirm_hits"):
            manager(model, n_confirm_hits=6, n_confirm_frames=5)

    def test_rejects_an_impossible_gate_probability(self, model):
        with pytest.raises(ValueError, match="gate_probability"):
            manager(model, gate_probability=1.5)

    def test_track_reports_its_own_state(self, model):
        track = Track(track_id=7, estimate=KalmanState(np.zeros(2), np.eye(2)))
        assert track.is_alive
        assert not track.is_confirmed

    def test_rejects_a_deletion_count_below_one(self, model):
        """A track deleted after zero misses could never coast at all.

        n_delete_misses is the count of consecutive misses a confirmed track
        survives; at zero, a track would be deleted on the frame it was
        confirmed, and the manager would silently produce no tracks rather
        than reporting an impossible configuration.
        """
        with pytest.raises(ValueError, match="n_delete_misses"):
            manager(model, n_delete_misses=0)

    def test_rejects_a_per_track_dimension_list_of_the_wrong_length(self, model):
        """measurement_dims is positional against the live tracks, so it must align.

        A short or long list would otherwise silently pair each dimension with
        the wrong track, giving one track the bootstrap's range-only gate and
        another the full one -- an association error that looks like bad
        tuning rather than a bug.
        """
        tracker = manager(model)
        tracker.step([np.array([10_000.0, 100.0])])
        with pytest.raises(ValueError, match="measurement_dims"):
            tracker.step([np.array([9_900.0, 100.0])], measurement_dims=[1, 2])

    def test_rejects_a_cost_matrix_that_is_not_two_dimensional(self):
        """Assignment is (n_tracks, n_measurements); a cube has no such reading.

        `np.atleast_2d` promotes the 1-D case for convenience, so the guard is
        about *higher* rank, where there is no defensible interpretation and
        scipy would otherwise fail with a message about shapes rather than
        about tracks.
        """
        with pytest.raises(ValueError, match="two-dimensional"):
            associate_gnn(np.ones((2, 2, 2)), 9.21)


# --------------------------------------------------------------------------- #
# The ENU state models and the extended filter
# --------------------------------------------------------------------------- #


def enu_model(state_model, **kwargs):
    """An ENU model at the scenario's own measurement noise."""
    defaults = {
        "sigma_accel_mps2": SIGMA_ACCEL_MPS2,
        "sigma_range_m": SIGMA_RANGE_M,
        "sigma_velocity_mps": SIGMA_VELOCITY_MPS,
    }
    return state_model_matrices(state_model, FRAME_TIME_S, **{**defaults, **kwargs})


def central_difference_jacobian(model, state):
    """Numerical d h / d x, for comparison against the analytic Jacobian."""
    n_state = state.size
    rows = model.measurement_function(state).size
    jacobian = np.zeros((rows, n_state))
    for index in range(n_state):
        # A relative step: the position components are tens of kilometres and
        # the rate components tens of metres per second, so one absolute step
        # cannot suit both.
        step = 1e-5 * max(1.0, abs(float(state[index])))
        offset = np.zeros(n_state)
        offset[index] = step
        jacobian[:, index] = (
            model.measurement_function(state + offset) - model.measurement_function(state - offset)
        ) / (2.0 * step)
    return jacobian


class TestEnuMeasurementModel:
    """h and its Jacobian, against geodesy and against a numerical derivative."""

    def test_h_matches_the_geodesy_module(self):
        """The conventions must be the repository's, not a fresh invention.

        Azimuth zero at true north increasing clockwise, and elevation above the
        local horizontal, exactly as enu_to_range_azimuth_elevation defines
        them. A tracker that reinvents these produces a track that mirrors the
        truth and still looks like a track.
        """
        from radar_forge.core.geodesy import enu_to_range_azimuth_elevation

        position_m = np.array([3_000.0, 4_000.0, 1_500.0])
        state = np.concatenate([position_m, [10.0, -20.0, 0.0]])
        predicted = enu_model("enu_3d").measurement_function(state)

        range_m, azimuth_deg, elevation_deg = enu_to_range_azimuth_elevation(position_m)
        # rtol 1e-12: the same trigonometry evaluated twice in float64.
        np.testing.assert_allclose(predicted[0], float(range_m), rtol=1e-12)
        np.testing.assert_allclose(np.rad2deg(predicted[1]) % 360.0, float(azimuth_deg), rtol=1e-12)
        np.testing.assert_allclose(np.rad2deg(predicted[2]), float(elevation_deg), rtol=1e-12)

    def test_the_range_rate_row_is_positive_closing(self):
        """Per D5, and the opposite sign to p.v / |p|."""
        model = enu_model("enu_2d")
        # Due north, flying south: closing.
        closing = model.measurement_function(np.array([0.0, 10_000.0, 0.0, -100.0]))
        assert closing[-1] == pytest.approx(100.0, rel=1e-12)
        opening = model.measurement_function(np.array([0.0, 10_000.0, 0.0, 100.0]))
        assert opening[-1] == pytest.approx(-100.0, rel=1e-12)

    def test_a_crossing_target_has_no_range_rate(self):
        """Velocity perpendicular to the line of sight is invisible in Doppler."""
        model = enu_model("enu_2d")
        crossing = model.measurement_function(np.array([0.0, 10_000.0, 80.0, 0.0]))
        assert crossing[-1] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("state_model", ["enu_2d", "enu_3d"])
    def test_the_jacobian_matches_a_central_difference(self, state_model, rng):
        """§12's Jacobian check.

        A central difference cannot reach the specification's 1e-8: its
        truncation falls as the step squared while its round-off grows as one
        over the step, so the best attainable on this function is around 1e-7.
        A complex-step derivative would do better but is unavailable here --
        atan2 and the Euclidean norm are not analytic in the required sense.
        """
        model = enu_model(state_model)
        n_state = model.transition.shape[0]
        n_axes = n_state // 2
        for _ in range(50):
            position_m = rng.uniform(-20_000.0, 20_000.0, n_axes)
            if np.linalg.norm(position_m) < 1_000.0:
                continue
            state = np.concatenate([position_m, rng.uniform(-100.0, 100.0, n_axes)])
            np.testing.assert_allclose(
                model.measurement_jacobian(state),
                central_difference_jacobian(model, state),
                rtol=1e-6,
                atol=1e-10,
            )

    @pytest.mark.parametrize(
        ("state_model", "dim"), [("range_1d", 2), ("enu_2d", 3), ("enu_3d", 4)]
    )
    def test_each_model_reports_its_measurement_dimension(self, state_model, dim):
        """The table in §5.2, which the gate threshold is read from."""
        model = enu_model(state_model)
        assert model.measurement_dim == dim
        assert model.measurement_noise.shape == (dim, dim)

    def test_the_angle_rows_are_declared(self):
        """Without this an innovation near due north is nearly 2 pi."""
        assert enu_model("enu_2d").angle_rows == (1,)
        assert enu_model("enu_3d").angle_rows == (1, 2)
        assert enu_model("range_1d").angle_rows == ()

    def test_an_innovation_across_due_north_is_wrapped(self):
        """A target at 359.9 degrees and a prediction at 0.1 are neighbours."""
        model = enu_model("enu_2d")
        state = np.array([-10.0, 10_000.0, 0.0, 0.0])  # just west of north
        measurement = model.measurement_function(np.array([10.0, 10_000.0, 0.0, 0.0]))
        innovation, _, _ = innovation_of(KalmanState(state, np.eye(4)), measurement, model)
        assert abs(innovation[1]) < 0.01

    def test_rejects_a_negative_angle_sigma(self):
        with pytest.raises(ValueError, match="sigma_azimuth_deg"):
            enu_model("enu_2d", sigma_azimuth_deg=-1.0)


class TestEnuTracking:
    """The extended filter, against the same analytic truth as the linear one."""

    @pytest.mark.parametrize("state_model", ["enu_2d", "enu_3d"])
    def test_a_noiseless_constant_velocity_target_has_zero_innovation(self, state_model):
        """§12's unit-level ground truth, in every state model.

        Any sign error in h, and any transposed row of the Jacobian, shows up
        here immediately -- unlike in a scenario run, where the filter will
        happily track a target while being wrong about which way it is going.
        """
        model = enu_model(state_model, sigma_accel_mps2=0.0)
        n_axes = model.transition.shape[0] // 2
        position_m = np.array([8_000.0, 12_000.0, 1_500.0])[:n_axes]
        velocity_mps = np.array([-60.0, 40.0, 0.0])[:n_axes]
        truth = np.concatenate([position_m, velocity_mps])

        state = KalmanState(truth.copy(), np.eye(2 * n_axes) * 100.0)
        for step in range(1, 16):
            state = predict(state, model)
            moved = np.concatenate([position_m + velocity_mps * step, velocity_mps])
            result = update(state, model.measurement_function(moved), model)
            # atol: the quantity being checked is zero.
            np.testing.assert_allclose(result.innovation, 0.0, atol=1e-6)
            state = result.posterior

    def test_the_enu_model_agrees_with_range_1d_on_a_radial_target(self):
        """§12 criterion 7, at unit level, on the case where it must hold.

        The two state models are handed the same radial target and must report
        the same range and range rate. An ENU model whose Jacobian is wrong will
        not manage it, and none of the other criteria would catch that.

        The target is radial deliberately. A straight line in Cartesian does not
        have a constant range rate -- the geometry turns under it -- so on a
        crossing target the two models *should* disagree, and the next test says
        so. Radial motion is the case where the one-dimensional model's
        assumption is exactly true, which is what isolates the Jacobian from the
        motion model.

        The end-to-end version of this criterion, over a whole scenario with
        §6.3's simulated angles, is not yet run; this is the part that needs no
        simulated angle.
        """
        linear = state_model_matrices(
            "range_1d",
            FRAME_TIME_S,
            sigma_accel_mps2=0.0,
            sigma_range_m=SIGMA_RANGE_M,
            sigma_velocity_mps=SIGMA_VELOCITY_MPS,
        )
        cartesian = enu_model("enu_2d", sigma_accel_mps2=0.0)

        # Inbound along its own bearing: purely radial, so the range rate is
        # constant and both motion models are exactly right.
        bearing = np.array([0.6, 0.8])
        position_m = 20_000.0 * bearing
        closing_mps = 80.0
        velocity_mps = -closing_mps * bearing

        linear_state = KalmanState(np.array([20_000.0, closing_mps]), np.eye(2))
        enu_state = KalmanState(np.concatenate([position_m, velocity_mps]), np.eye(4) * 100.0)

        for step in range(1, 11):
            moved_m = position_m + velocity_mps * step
            truth_range_m = float(np.linalg.norm(moved_m))

            linear_state = update(
                predict(linear_state, linear),
                np.array([truth_range_m, closing_mps]),
                linear,
            ).posterior
            enu_state = update(
                predict(enu_state, cartesian),
                cartesian.measurement_function(np.concatenate([moved_m, velocity_mps])),
                cartesian,
            ).posterior

            enu_range_m = float(np.linalg.norm(enu_state.state[:2]))
            enu_closing_mps = -float(enu_state.state[:2] @ enu_state.state[2:]) / enu_range_m

            assert enu_range_m == pytest.approx(linear_state.state[0], abs=0.1 * SIGMA_RANGE_M)
            assert enu_closing_mps == pytest.approx(
                linear_state.state[1], abs=0.1 * SIGMA_VELOCITY_MPS
            )

    def test_a_crossing_target_is_where_the_two_models_part(self):
        """§6.1's argument, asserted: a range-only state cannot represent this.

        A target flying a straight line past the radar has a range rate that
        changes every frame, because the geometry turns under it. The ENU model
        carries that for free -- it is straight-line motion in the state it
        actually uses -- while the one-dimensional model, whose state says the
        range rate is constant, has to absorb the whole of it as process noise.
        That is what the state model is *for*, and it is invisible on a radial
        target.
        """
        cartesian = enu_model("enu_2d", sigma_accel_mps2=0.0)
        position_m = np.array([8_000.0, 12_000.0])
        velocity_mps = np.array([-60.0, 40.0])

        rates_mps = []
        for step in range(12):
            moved_m = position_m + velocity_mps * step
            rates_mps.append(
                cartesian.measurement_function(np.concatenate([moved_m, velocity_mps]))[-1]
            )
        # The range rate moves by more than a Doppler bin over the window, so a
        # constant-rate state is measurably wrong about it.
        assert abs(rates_mps[-1] - rates_mps[0]) > 20.0 * SIGMA_VELOCITY_MPS
