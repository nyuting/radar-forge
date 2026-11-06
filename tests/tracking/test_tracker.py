"""Tests for radar_forge.tracking.tracker: the scan loop over a bank, and its lifecycle.

Ground truth is hand-countable wherever it can be: M-of-N confirmation and deletion scan by
scan, dead reckoning through a miss as exactly F x, two targets that cross, a covariance
reset to exactly the prior's variance, and the debiased conversion against its closed form
and against quadrature.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from radar_forge.tracking.association import (
    GlobalNearestNeighbour,
    JointProbabilisticDataAssociation,
)
from radar_forge.tracking.filters import (
    Linearisation,
    TaylorLinearisation,
    TrackBank,
    TrackStatus,
    UnscentedTransform,
)
from radar_forge.tracking.imm import InteractingMultipleModel
from radar_forge.tracking.initiation import (
    DebiasedSphericalInitiator,
    DirectStateInitiator,
    debiased_spherical_to_enu,
)
from radar_forge.tracking.models import (
    BistaticRangeRateModel,
    CartesianMotion,
    LinearMeasurement,
    RadialConstantVelocity,
    RangeAzimuthElevationRangeRate,
    marginal,
)
from radar_forge.tracking.tracker import ExistencePolicy, LifecyclePolicy, Scan, Tracker

TENTATIVE, CONFIRMED = TrackStatus.TENTATIVE, TrackStatus.CONFIRMED
COASTING, DELETED = TrackStatus.COASTING, TrackStatus.DELETED


def _one_axis(
    policy: LifecyclePolicy | None = None,
    linearisation: Linearisation | None = None,
    sigma_acceleration_mps2: float = 1.0,
    **settings: Any,
) -> Tracker:
    """Position measured along x; q = 2 sigma² · 0.5 s, so q = 1 m²/s³ by default."""
    motion = CartesianMotion(
        "x", sigma_acceleration_mps2=sigma_acceleration_mps2, acceleration_correlation_time_s=0.5
    )
    model = LinearMeasurement(motion.state_layout, ("x_m",))
    initiator = DirectStateInitiator([0.0, 0.0], np.diag([1.0, 100.0**2]))
    return Tracker(
        {"x": model},
        motion,
        linearisation or TaylorLinearisation(),
        0.997,
        GlobalNearestNeighbour(),
        initiator,
        policy,
        **settings,
    )


def _scan(time_s: float, positions_m: Any, variance_m2: float = 1.0, model_id: str = "x") -> Scan:
    return Scan(time_s, np.reshape(positions_m, (-1, 1)), [[variance_m2]], model_id)


def _statuses(tracker: Tracker, pattern: str) -> list[list[int]]:
    """Run a target at 1000 + 20 t, hit ("h") or missed ("m") scan by scan; return statuses."""
    return [
        tracker.process(_scan(t, [1000.0 + 20.0 * t] if c == "h" else [])).status.tolist()
        for t, c in enumerate(pattern)
    ]


def test_a_track_is_confirmed_on_its_third_hit() -> None:
    assert _statuses(_one_axis(), "hhh") == [[TENTATIVE], [TENTATIVE], [CONFIRMED]]


def test_three_hits_in_the_first_five_scans_confirm_on_the_fifth() -> None:
    statuses = _statuses(_one_axis(), "hmhmh")
    assert statuses == [[TENTATIVE]] * 4 + [[CONFIRMED]]


def test_a_tentative_track_dies_as_soon_as_m_is_out_of_reach() -> None:
    """After hit, miss, miss, miss: one hit and one scan left cannot make three."""
    assert _statuses(_one_axis(), "hmmm") == [[TENTATIVE]] * 3 + [[DELETED]]


def test_a_tentative_track_out_of_scans_is_deleted_on_the_last() -> None:
    assert _statuses(_one_axis(), "hhmmm") == [[TENTATIVE]] * 4 + [[DELETED]]


def test_a_one_of_n_policy_confirms_a_track_at_birth() -> None:
    assert _statuses(_one_axis(LifecyclePolicy(1, 1)), "h") == [[CONFIRMED]]


def test_a_confirmed_track_coasts_then_is_deleted_on_its_last_miss_and_reported_once() -> None:
    tracker = _one_axis(LifecyclePolicy(n_delete_misses=3))
    statuses = _statuses(tracker, "hhhmmmm")
    assert statuses[3:] == [[COASTING], [COASTING], [DELETED], []]
    assert tracker.bank.n_tracks == 0


def test_a_hit_resets_the_misses_and_confirms_a_coasting_track_again() -> None:
    tracker = _one_axis()
    assert _statuses(tracker, "hhhmmh")[-1] == [CONFIRMED]
    assert tracker.bank.n_misses.tolist() == [0] and tracker.bank.n_hits.tolist() == [4]


def test_reacquisition_scans_extend_only_a_confirmed_track() -> None:
    policy = LifecyclePolicy(
        n_confirm_hits=3, n_confirm_scans=10, n_delete_misses=2, n_reacquire_misses=2
    )
    assert _statuses(_one_axis(policy), "hmm")[-1] == [DELETED]
    statuses = _statuses(_one_axis(policy), "hhhmmmm")
    assert statuses[-2:] == [[COASTING], [DELETED]]


def test_a_gap_of_exactly_the_coast_time_keeps_the_track() -> None:
    tracker = _one_axis(LifecyclePolicy(max_coast_time_s=5.0))
    _statuses(tracker, "hhh")
    bank = tracker.process(_scan(7.0, [1140.0]))
    assert bank.track_id.tolist() == [1] and bank.status.tolist() == [CONFIRMED]


def test_a_stale_track_is_deleted_before_it_can_take_a_measurement() -> None:
    tracker = _one_axis(LifecyclePolicy(max_coast_time_s=5.0))
    _statuses(tracker, "hhh")
    bank = tracker.process(_scan(7.5, [1150.0]))
    assert bank.track_id.tolist() == [1, 2]
    assert bank.status.tolist() == [DELETED, TENTATIVE]


def test_every_linearisation_confirms_and_dead_reckons_through_a_miss(
    linearisation: Linearisation,
) -> None:
    """A miss is exactly F x̂: the track is predicted and nothing updates it."""
    tracker = _one_axis(linearisation=linearisation)
    _statuses(tracker, "hhhh")
    before = tracker.bank.mean[0].copy()
    after = tracker.process(_scan(4.0, []))
    assert after.status.tolist() == [COASTING]
    # rtol 1e-12: one transition, through sigma points for the UKF.
    np.testing.assert_allclose(after.mean[0], [before[0] + before[1], before[1]], rtol=1e-12)
    assert np.isnan(after.nis[0]) and after.measurement_index[0] == -1


def test_crossing_targets_keep_their_identities_and_velocities() -> None:
    """Two targets cross between scans; every other scan lists them in reverse order."""
    tracker = _one_axis(sigma_acceleration_mps2=0.1)
    for scan in range(24):
        # Scans are processed in time order; each depends on the one before.
        time_s = scan / 2
        positions_m = [100.0 + 3.0 * time_s, 131.0 - 3.0 * time_s]
        bank = tracker.process(_scan(time_s, positions_m[:: 1 - 2 * (scan % 2)], 0.1))
        assert bank.track_id.tolist() == [1, 2]
    assert bank.status.tolist() == [CONFIRMED, CONFIRMED]
    # atol 0.1 m/s: each track must carry its own target's velocity, +3 or -3.
    np.testing.assert_allclose(bank.mean[:, 1], [3.0, -3.0], atol=0.1)


def _settled(policy: LifecyclePolicy | None = None, **settings: Any) -> Tracker:
    """A confirmed track on a target holding still at x = 1000 m, R = 1 m², q = 1 m²/s³."""
    tracker = _one_axis(policy, **settings)
    for time_s in range(6):
        # Scans are processed in time order; each depends on the one before.
        tracker.process(_scan(time_s, [1000.0]))
    return tracker


def test_a_confirmed_track_gets_first_pick_over_a_tentative_one() -> None:
    """A young track must not steal a confirmed one's detection (review #7).

    Scan 6 starts a tentative track 15 m away. In scan 7 one detection is 3 m from the
    confirmed track and 12 m from the tentative one, whose NIS is smaller because it is so
    uncertain; one joint solve gives it the detection, confirmed-first does not.
    """
    first, joint = _settled(), _settled(assignment="joint")
    for tracker in (first, joint):
        tracker.process(_scan(6.0, [1000.0, 1015.0]))
    assert first.process(_scan(7.0, [1003.0])).measurement_index.tolist() == [0, -1]
    assert joint.process(_scan(7.0, [1003.0])).measurement_index.tolist() == [-1, 0]


def test_no_track_starts_inside_a_confirmed_gate_unless_assignment_is_joint() -> None:
    assert _settled().process(_scan(6.0, [1000.0, 1001.0])).n_tracks == 1
    joint = _settled(assignment="joint").process(_scan(6.0, [1000.0, 1001.0]))
    assert joint.status.tolist() == [CONFIRMED, TENTATIVE]
    assert _settled().process(_scan(6.0, [1000.0, 1100.0])).n_tracks == 2


def test_a_snapshot_records_which_measurement_its_track_took_and_its_nis() -> None:
    tracker = _settled()
    updated, born = tracker.process(_scan(6.0, [1100.0, 1000.5])).snapshots()
    assert updated.measurement_index == 1 and updated.nis is not None and updated.nis < 1.0
    assert born.measurement_index == 0 and born.nis is None


def test_an_out_of_sequence_scan_is_rejected_and_changes_nothing() -> None:
    tracker = _settled()
    bank = tracker.bank
    with pytest.raises(ValueError, match="out-of-sequence"):
        tracker.process(_scan(4.0, [1000.0]))
    assert tracker.bank is bank and tracker.time_s == 5.0


def test_a_time_within_a_microsecond_is_not_out_of_sequence() -> None:
    tracker = _settled()
    assert tracker.process(_scan(5.0 - 1e-9, [1000.0])).time_s.tolist() == [5.0]


@pytest.mark.parametrize(
    "scan", [_scan(6.0, [1.0], model_id="y"), Scan(6.0, [[1.0, 2.0]], np.eye(2), "x")]
)
def test_a_scan_of_an_unknown_model_or_the_wrong_dimension_is_rejected(scan: Scan) -> None:
    with pytest.raises(ValueError, match="unknown"):
        _settled().process(scan)


@pytest.mark.parametrize(
    ("time_s", "z", "noise"),
    [
        (np.nan, [[1.0]], [[1.0]]),
        (0.0, [1.0], [[1.0]]),
        (0.0, [[np.inf]], [[1.0]]),
        (0.0, [[1.0]], [[-1.0]]),
    ],
)
def test_a_scan_checks_its_measurements_once_at_the_boundary(
    time_s: float, z: Any, noise: Any
) -> None:
    with pytest.raises(ValueError, match=r"finite|shape|definite"):
        Scan(time_s, z, noise, "x")


@pytest.mark.parametrize(
    "noise", [np.diag([25.0, 0.0]), [[[25.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 1.0]]]]
)
def test_a_scan_refuses_a_noise_covariance_with_a_zero_variance(noise: Any) -> None:
    """A zero variance in R makes S singular and the likelihood ratio meaningless."""
    with pytest.raises(ValueError, match="positive definite"):
        Scan(0.0, [[1.0, 2.0], [3.0, 4.0]], noise, "both")


def test_a_scan_stores_one_shared_r_as_a_frozen_stack() -> None:
    scan = Scan(0.0, [[1.0], [2.0], [3.0]], [[4.0]], "x")
    assert scan.noise_covariance.shape == (3, 1, 1) and not scan.measurements.flags.writeable


@pytest.mark.parametrize(
    "settings",
    [
        {"n_confirm_hits": 0},
        {"n_confirm_hits": 6},
        {"n_delete_misses": 0},
        {"n_reacquire_misses": -1},
        {"n_confirm_hits": True},
        {"max_coast_time_s": 0.0},
        {"reacquire_reset": ("x_m", "x_m")},
    ],
)
def test_a_policy_that_could_not_work_is_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="need"):
        LifecyclePolicy(**settings)


# The scenario mechanisms (tracker-001 §5.1, §5.4): the selector, its range-only retry, the
# likelihood ratio and the re-acquisition reset. q = 1 m²/s³, so a settled track's rate gate
# is about a metre per second wide and a detection 40 m/s out, a wrong fold, fails it.
RADIAL = RadialConstantVelocity(1.0, acceleration_correlation_time_s=0.5)
BOTH = LinearMeasurement(RADIAL.state_layout, ("bistatic_range_m", "bistatic_range_rate_mps"))
MODELS = {"both": BOTH, "range": marginal(BOTH, ("bistatic_range_m",))}
DENSITIES = {"both": 1e-6, "range": 1e-4}


def _full_then_range(bank: TrackBank, model_id: str) -> list[tuple[str, Any]]:
    """Scenario 003's selector: range and rate first, then range alone, for every track."""
    return [("both", np.ones(bank.n_tracks, bool)), ("range", np.ones(bank.n_tracks, bool))]


def _radial(policy: LifecyclePolicy | None = None, **settings: Any) -> Tracker:
    """A confirmed track on a target holding still at 5 km, after six scans."""
    initiator = DirectStateInitiator([0.0, 0.0], np.diag([1.0, 100.0**2]))
    tracker = Tracker(
        MODELS,
        RADIAL,
        TaylorLinearisation(),
        0.997,
        GlobalNearestNeighbour(),
        initiator,
        policy,
        **settings,
    )
    for time_s in range(6):
        # Scans are processed in time order; each depends on the one before.
        tracker.process(_radial_scan(time_s, [(5000.0, 0.0)]))
    return tracker


def _radial_scan(time_s: float, detections: Any) -> Scan:
    return Scan(time_s, np.reshape(detections, (-1, 2)), np.diag([25.0, 0.25]), "both")


def test_a_pair_that_fails_its_rate_gate_is_kept_through_range_alone() -> None:
    """Spec §5.4's retry: a wrong fold costs the rate, not the range."""
    plain = _radial()
    retrying = _radial(
        cost="negative_log_likelihood_ratio",
        clutter_density=DENSITIES,
        model_selector=_full_then_range,
    )
    missed = plain.process(_radial_scan(6.0, [(5000.0, 40.0)]))
    kept = retrying.process(_radial_scan(6.0, [(5000.0, 40.0)]))
    # Without the retry the track misses, and the unclaimed detection starts a second track.
    assert (missed.status.tolist(), missed.measurement_index.tolist()) == (
        [COASTING, TENTATIVE],
        [-1, 0],
    )
    assert (kept.status.tolist(), kept.measurement_index.tolist()) == ([CONFIRMED], [0])
    assert abs(kept.mean[0, 1]) < 1.0


def test_the_retry_reports_the_nis_of_the_model_it_passed() -> None:
    tracker = _radial(
        cost="negative_log_likelihood_ratio",
        clutter_density=DENSITIES,
        model_selector=_full_then_range,
    )
    variance = (
        tracker.bank.covariance[0, 0, 0]
        + tracker.bank.covariance[0, 1, 1]
        + 2 * tracker.bank.covariance[0, 0, 1]
    )
    bank = tracker.process(_radial_scan(6.0, [(5003.0, 40.0)]))
    # rtol 1e-12: the 1-D NIS is ν² / (P⁻ + R) on range alone.
    np.testing.assert_allclose(
        bank.nis[0],
        3.0**2 / (variance + RADIAL.process_noise(np.zeros(2), 1.0)[0, 0] + 25.0),
        rtol=1e-12,
    )


def test_nis_costs_of_two_dimensions_in_one_scan_are_refused_and_change_nothing() -> None:
    """Spec §5.1: a 1-D NIS undercuts a 2-D one. One detection gates at two components, the other
    through range alone."""
    tracker = _radial(model_selector=_full_then_range)
    bank = tracker.bank
    with pytest.raises(ValueError, match="likelihood ratio"):
        tracker.process(_radial_scan(6.0, [(5000.0, 0.0), (5001.0, 40.0)]))
    assert tracker.bank is bank


def test_the_selector_may_offer_only_the_scans_model_or_a_marginal_of_it() -> None:
    def rogue(bank: TrackBank, model_id: str) -> list[tuple[str, Any]]:
        return [("other", np.ones(bank.n_tracks, bool))]

    tracker = Tracker(
        {**MODELS, "other": LinearMeasurement(RADIAL.state_layout, ("bistatic_range_m",))},
        RADIAL,
        TaylorLinearisation(),
        0.997,
        GlobalNearestNeighbour(),
        DirectStateInitiator([0.0, 0.0], np.diag([1.0, 1e4])),
        model_selector=rogue,
    )
    with pytest.raises(ValueError, match="marginal"):
        tracker.process(_radial_scan(0.0, [(5000.0, 0.0)]))
    assert tracker.time_s is None and tracker.bank.n_tracks == 0


def test_the_reset_forgets_the_rate_on_the_miss_that_opens_reacquisition() -> None:
    """On the n_delete_misses-th miss the rate returns to exactly the prior's variance."""
    policy = LifecyclePolicy(
        n_delete_misses=2, n_reacquire_misses=3, reacquire_reset=("bistatic_range_rate_mps",)
    )
    reset, twin = _radial(policy), _radial(LifecyclePolicy(n_delete_misses=2, n_reacquire_misses=3))
    first = reset.process(_radial_scan(6.0, []))
    twin.process(_radial_scan(6.0, []))
    kept, plain = reset.process(_radial_scan(7.0, [])), twin.process(_radial_scan(7.0, []))
    assert first.covariance[0, 0, 1] != 0.0 and kept.status.tolist() == [COASTING]
    assert kept.covariance[0, 1, 1] == 100.0**2 and kept.covariance[0, 0, 1] == 0.0
    # rtol 1e-12: the reset touches neither the mean nor the range variance.
    np.testing.assert_allclose(kept.mean, plain.mean, rtol=1e-12)
    np.testing.assert_allclose(kept.covariance[0, 0, 0], plain.covariance[0, 0, 0], rtol=1e-12)
    # The next miss predicts from the reset and does not reset again: the gate widens by
    # about T² · 100² m², the rate is coupled to range again, and its variance grows by Q.
    after = reset.process(_radial_scan(8.0, []))
    assert after.covariance[0, 0, 0] > kept.covariance[0, 0, 0] + 100.0**2 / 2
    assert after.covariance[0, 0, 1] != 0.0 and after.covariance[0, 1, 1] > 100.0**2


@pytest.mark.parametrize(
    "settings",
    [
        {"cost": "nll"},
        {"assignment": "greedy"},
        {"clutter_density": {"both": 1e-6}},
        {"cost": "negative_log_likelihood_ratio", "clutter_density": {"both": 1.0}},
        {"cost": "negative_log_likelihood_ratio", "clutter_density": {"both": 1.0, "range": 0.0}},
        {"policy": LifecyclePolicy(reacquire_reset=("xdot_mps",))},
        {"gate_probability": 1.0},
    ],
)
def test_the_constructor_rejects_settings_that_do_not_fit(settings: dict[str, Any]) -> None:
    arguments: dict[str, Any] = {
        "models": MODELS,
        "motion": RADIAL,
        "linearisation": TaylorLinearisation(),
        "gate_probability": 0.997,
        "associator": GlobalNearestNeighbour(),
        "initiator": DirectStateInitiator([0.0, 0.0], np.eye(2)),
    }
    with pytest.raises(ValueError, match=r"cost|clutter|not in the layout|probability"):
        Tracker(**{**arguments, **settings})


def test_a_model_or_prior_in_another_layout_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"layout|coordinates"):
        Tracker(
            {"x": LinearMeasurement(CartesianMotion("x").state_layout, ("x_m",))},
            RADIAL,
            TaylorLinearisation(),
            0.997,
            GlobalNearestNeighbour(),
            DirectStateInitiator([0.0], np.eye(1)),
        )
    with pytest.raises(ValueError, match="coordinates"):
        Tracker(
            MODELS,
            RADIAL,
            TaylorLinearisation(),
            0.997,
            GlobalNearestNeighbour(),
            DirectStateInitiator([0.0], np.eye(1)),
        )


def test_a_bistatic_tracker_tracks_a_seeded_target_and_starts_none(
    linearisation: Linearisation, rng: np.random.Generator
) -> None:
    """One bistatic measurement cannot fix a position, so tracks come from a cue."""
    motion = CartesianMotion("xy", sigma_acceleration_mps2=0.5)
    model = BistaticRangeRateModel(
        motion.state_layout, np.zeros(3), [20_000.0, 0.0, 0.0], {"z_m": 3000.0, "zdot_mps": 0.0}
    )
    tracker = Tracker(
        {"bistatic": model},
        motion,
        linearisation,
        0.997,
        GlobalNearestNeighbour(),
        DirectStateInitiator(np.zeros(4), np.eye(4)),
    )
    truth = np.array([5_000.0, 50.0, 15_000.0, -100.0])
    prior = np.diag([200.0**2, 10.0**2] * 2)
    assert tracker.seed(
        truth[None] + rng.normal(0.0, 1.0, (1, 4)) * np.sqrt(np.diag(prior)), prior[None], 0.0
    ).tolist() == [1]
    noise = np.diag([20.0**2, 1.0])
    for step in range(1, 31):
        # Scans are processed in time order; each depends on the one before.
        truth = motion.transition(truth, 1.0)
        z = model.predict(truth)[None] + rng.multivariate_normal(np.zeros(2), noise, 1)
        bank = tracker.process(Scan(float(step), z, noise, "bistatic"))
    assert bank.track_id.tolist() == [1] and bank.status.tolist() == [CONFIRMED]


def test_every_linearisation_tracks_a_target_from_spherical_measurements(
    linearisation: Linearisation, rng: np.random.Generator
) -> None:
    """The debiased start, then spherical updates: one confirmed track, on the target."""
    motion = CartesianMotion()
    model = RangeAzimuthElevationRangeRate(motion.state_layout, np.zeros(3))
    prior = DebiasedSphericalInitiator(np.zeros(6), np.diag([1e8, 200.0**2] * 3))
    tracker = Tracker(
        {"radar": model}, motion, linearisation, 0.997, GlobalNearestNeighbour(), prior
    )
    noise = np.diag([20.0**2, 2e-3**2, 2e-3**2, 1.0])
    truth = np.array([12_000.0, -150.0, 16_000.0, 0.0, 3_000.0, 0.0])
    for step in range(20):
        # Scans are processed in time order; each depends on the one before.
        z = model.predict(truth)[None] + rng.multivariate_normal(np.zeros(4), noise, 1)
        bank = tracker.process(Scan(float(step), z, noise, "radar"))
        truth = motion.transition(truth, 1.0)
    assert bank.track_id.tolist() == [1] and bank.status.tolist() == [CONFIRMED]
    error_m = bank.mean[0, [0, 2, 4]] - motion.transition(truth, -1.0)[[0, 2, 4]]
    assert np.linalg.norm(error_m) < 100.0


# Slow: the full tracker over 40 scans of clutter.
@pytest.mark.slow
def test_a_target_keeps_one_track_through_clutter_and_missed_detections(
    rng: np.random.Generator,
) -> None:
    """P_d = 0.7 and three false alarms per scan over 5 km: one confirmed track, one ID."""
    tracker = _one_axis(sigma_acceleration_mps2=4.0)
    identifiers = set()
    for t in range(40):
        # Scans are processed in time order; each depends on the one before.
        truth_m = 1000.0 + 20.0 * t
        positions_m = list(rng.uniform(0.0, 5000.0, 3))
        if rng.random() < 0.7:
            positions_m.append(truth_m + rng.normal(0.0, np.sqrt(0.1)))
        bank = tracker.process(_scan(float(t), rng.permutation(positions_m), 0.1))
        near = (np.abs(bank.mean[:, 0] - truth_m) < 10.0) & np.isin(
            bank.status, [CONFIRMED, COASTING]
        )
        if t >= 10:
            assert near.sum() == 1
            identifiers.update(bank.track_id[near].tolist())
    assert len(identifiers) == 1


def test_unscented_and_taylor_trackers_agree_on_a_linear_scene() -> None:
    """On linear models the UKF is the KF (spec §3), so the whole loop gives the same banks."""
    banks = []
    for linearisation in (TaylorLinearisation(), UnscentedTransform()):
        tracker = _one_axis(linearisation=linearisation)
        for t in range(8):
            # Scans are processed in time order; each depends on the one before.
            bank = tracker.process(_scan(float(t), [1000.0 + 20.0 * t, 3000.0 - 5.0 * t]))
        banks.append(bank)
    np.testing.assert_array_equal(banks[0].track_id, banks[1].track_id)
    # rtol 1e-9: eight steps of sigma-point roundoff against the Joseph form.
    np.testing.assert_allclose(banks[0].mean, banks[1].mean, rtol=1e-9)


# Initiation.

LAYOUT = CartesianMotion().state_layout
PRIOR_MEAN = np.array([0.0, 1.0, 0.0, 2.0, 500.0, 3.0])
PRIOR_COVARIANCE = np.diag([1e8, 100.0, 1e8, 100.0, 1e6, 100.0])


def test_a_direct_start_copies_the_measured_coordinates_and_keeps_the_rest() -> None:
    model = LinearMeasurement(LAYOUT, ("y_m", "x_m"))
    initiator = DirectStateInitiator(PRIOR_MEAN, PRIOR_COVARIANCE)
    noise = np.array([[[4.0, 1.0], [1.0, 9.0]], [[1.0, 0.0], [0.0, 1.0]]])
    mean, covariance = initiator.initiate(np.array([[20.0, 10.0], [7.0, 8.0]]), noise, model)  # type: ignore[misc]
    np.testing.assert_array_equal(mean[0], [10.0, 1.0, 20.0, 2.0, 500.0, 3.0])
    np.testing.assert_array_equal(covariance[0][np.ix_([2, 0], [2, 0])], noise[0])
    np.testing.assert_array_equal(np.diag(covariance[1])[[1, 3, 4, 5]], [100.0, 100.0, 1e6, 100.0])


def test_seed_names_copy_only_the_named_coordinates() -> None:
    """A track whose folded rate cannot be trusted starts from its range alone."""
    model = LinearMeasurement(LAYOUT, ("x_m", "xdot_mps"))
    initiator = DirectStateInitiator(PRIOR_MEAN, PRIOR_COVARIANCE, seed_names=("x_m",))
    mean, covariance = initiator.initiate(np.array([[50.0, -9.0]]), np.eye(2)[None], model)  # type: ignore[misc]
    assert mean[0, 0] == 50.0 and mean[0, 1] == 1.0 and covariance[0, 1, 1] == 100.0


def test_a_prior_that_ties_measured_to_unmeasured_coordinates_is_refused() -> None:
    covariance = PRIOR_COVARIANCE.copy()
    covariance[0, 1] = covariance[1, 0] = 5.0
    initiator = DirectStateInitiator(PRIOR_MEAN, covariance)
    with pytest.raises(ValueError, match="ties"):
        initiator.initiate(np.zeros((1, 1)), np.eye(1)[None], LinearMeasurement(LAYOUT, ("x_m",)))


def test_a_nonlinear_model_starts_no_direct_track() -> None:
    model = BistaticRangeRateModel(LAYOUT, np.zeros(3), np.ones(3))
    initiator = DirectStateInitiator(PRIOR_MEAN, PRIOR_COVARIANCE)
    assert initiator.initiate(np.zeros((1, 2)), np.eye(2)[None], model) is None


def test_the_debiased_mean_is_the_naive_point_shrunk_by_the_damping_factors() -> None:
    """E[cos(θ + w)] = exp(-σ²/2) cos θ: north and east shrink by both factors, up by one."""
    sigma_az, sigma_el, azimuth, elevation = 0.1, 0.05, 0.7, 0.3
    position_m, _ = debiased_spherical_to_enu(1e4, azimuth, elevation, 10.0, sigma_az, sigma_el)
    naive = 1e4 * np.array(
        [
            np.cos(elevation) * np.sin(azimuth),
            np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation),
        ]
    )
    damping = np.exp(-np.array([sigma_az**2 + sigma_el**2] * 2 + [sigma_el**2]) / 2)
    # rtol 1e-14: a few products of exact factors.
    np.testing.assert_allclose(position_m, naive * damping, rtol=1e-14)


def test_with_no_angle_error_the_conversion_is_the_naive_one() -> None:
    position_m, covariance_m2 = debiased_spherical_to_enu(1e4, 0.7, 0.3, 0.0, 0.0, 0.0)
    assert np.linalg.norm(position_m) == pytest.approx(1e4, rel=1e-14)
    np.testing.assert_allclose(covariance_m2, 0.0, atol=1e-6)


def test_with_small_errors_the_covariance_is_the_linearised_one() -> None:
    """J R Jᵀ, with J the Jacobian of (r, θ, φ) to ENU, as the errors go to zero."""
    r, azimuth, elevation, sr, sa, se = 1e4, 0.7, 0.3, 1.0, 1e-5, 1e-5
    _, covariance_m2 = debiased_spherical_to_enu(r, azimuth, elevation, sr, sa, se)
    ce, se_, ca, sa_ = np.cos(elevation), np.sin(elevation), np.cos(azimuth), np.sin(azimuth)
    jacobian = np.array(
        [
            [ce * sa_, r * ce * ca, -r * se_ * sa_],
            [ce * ca, -r * ce * sa_, -r * se_ * ca],
            [se_, 0.0, r * ce],
        ]
    )
    expected = jacobian @ np.diag([sr**2, sa**2, se**2]) @ jacobian.T
    # rtol 1e-6: the exact moments differ from the linearised ones at order sigma².
    np.testing.assert_allclose(covariance_m2, expected, rtol=1e-6, atol=1e-9)


def test_the_debiased_moments_are_those_of_the_true_position_given_the_measurement() -> None:
    """Integrate p = r d over Gaussian (r, θ, φ) about the measured ones, by quadrature.

    40-point Gauss-Hermite rules per axis integrate these smooth functions to roundoff, so
    the comparison is exact rather than a Monte Carlo interval.
    """
    r, azimuth, elevation, sr, sa, se = 2e4, 0.2, 0.1, 30.0, 0.05, 0.03
    position_m, covariance_m2 = debiased_spherical_to_enu(r, azimuth, elevation, sr, sa, se)
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)
    weights = weights / weights.sum()
    rr, aa, ee = np.meshgrid(
        r + sr * nodes, azimuth + sa * nodes, elevation + se * nodes, indexing="ij"
    )
    points = np.stack(
        [rr * np.cos(ee) * np.sin(aa), rr * np.cos(ee) * np.cos(aa), rr * np.sin(ee)], -1
    )
    weight = np.einsum("i,j,k->ijk", weights, weights, weights)
    mean = np.einsum("ijk,ijkd->d", weight, points)
    deviation = points - mean
    covariance = np.einsum("ijk,ijkd,ijke->de", weight, deviation, deviation)
    # rtol 1e-10: quadrature and closed form agree to about 1e-13 of the largest entry.
    np.testing.assert_allclose(position_m, mean, rtol=1e-10)
    np.testing.assert_allclose(covariance_m2, covariance, rtol=1e-10, atol=1e-10 * covariance.max())


@pytest.mark.parametrize(
    "arguments",
    [(0.0, 0.0, 0.0, 1.0, 0.1), (1e3, np.nan, 0.0, 1.0, 0.1), (1e3, 0.0, 0.0, -1.0, 0.1)],
)
def test_an_impossible_spherical_measurement_is_refused(arguments: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="positive ranges"):
        debiased_spherical_to_enu(*arguments)


def test_a_spherical_start_places_the_track_from_the_site_and_keeps_its_velocity() -> None:
    site_m = np.array([100.0, -200.0, 30.0])
    model = RangeAzimuthElevationRangeRate(LAYOUT, site_m)
    initiator = DebiasedSphericalInitiator(PRIOR_MEAN, PRIOR_COVARIANCE)
    z = np.array([[1000.0, np.pi / 2, 0.0, -5.0]])
    mean, _ = initiator.initiate(z, np.diag([25.0, 1e-8, 1e-8, 1.0])[None], model)  # type: ignore[misc]
    # atol 1e-3 m: a 1e-4 rad sigma pulls the point in by 1e-5 m at 1 km.
    np.testing.assert_allclose(mean[0, [0, 2, 4]], site_m + np.array([1000.0, 0.0, 0.0]), atol=1e-3)
    np.testing.assert_array_equal(mean[0, [1, 3, 5]], [1.0, 2.0, 3.0])


def test_without_elevation_the_height_keeps_the_prior() -> None:
    model = RangeAzimuthElevationRangeRate(LAYOUT, np.zeros(3), ("range_m", "azimuth_rad"))
    initiator = DebiasedSphericalInitiator(PRIOR_MEAN, PRIOR_COVARIANCE)
    mean, covariance = initiator.initiate(
        np.array([[1000.0, 0.0]]), np.diag([25.0, 1e-6])[None], model
    )  # type: ignore[misc]
    assert mean[0, 4] == 500.0 and covariance[0, 4, 4] == 1e6


def test_a_spherical_model_without_azimuth_or_another_model_starts_no_track() -> None:
    initiator = DebiasedSphericalInitiator(PRIOR_MEAN, PRIOR_COVARIANCE)
    range_only = RangeAzimuthElevationRangeRate(LAYOUT, np.zeros(3), ("range_m",))
    assert initiator.initiate(np.ones((1, 1)), np.eye(1)[None], range_only) is None
    assert (
        initiator.initiate(np.ones((1, 1)), np.eye(1)[None], LinearMeasurement(LAYOUT, ("x_m",)))
        is None
    )


def test_coupled_range_and_angle_errors_are_refused() -> None:
    model = RangeAzimuthElevationRangeRate(LAYOUT, np.zeros(3), ("range_m", "azimuth_rad"))
    initiator = DebiasedSphericalInitiator(PRIOR_MEAN, PRIOR_COVARIANCE)
    with pytest.raises(ValueError, match="independent"):
        initiator.initiate(
            np.array([[1000.0, 0.0]]), np.array([[[25.0, 1e-4], [1e-4, 1e-6]]]), model
        )


def test_an_invalid_prior_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="prior"):
        DirectStateInitiator([0.0, np.nan], np.eye(2))
    with pytest.raises(ValueError, match="prior_covariance"):
        DirectStateInitiator([0.0, 0.0], -np.eye(2))


# --- scenario 011: coverage, existence, covariance deletion, birth inhibition ---


def _left_half(bank: TrackBank, scan: Scan) -> NDArray[np.bool_]:
    """A coverage that sees only the tracks at negative x."""
    return np.asarray(bank.mean[:, 0] < 0.0, dtype=bool)


def test_a_track_the_scan_cannot_see_keeps_its_row_time_and_counts() -> None:
    """A track outside the coverage is neither predicted nor counted; a covered one is."""
    tracker = _one_axis(LifecyclePolicy(1, 1), coverage=_left_half)
    tracker.process(_scan(0.0, [-500.0, 500.0]))
    before = tracker.bank
    after = tracker.process(_scan(1.0, [-480.0]))
    right, left = np.flatnonzero(before.mean[:, 0] > 0)[0], np.flatnonzero(before.mean[:, 0] < 0)[0]
    np.testing.assert_array_equal(after.mean[right], before.mean[right])
    assert after.time_s[right] == 0.0 and after.n_scans[right] == before.n_scans[right]
    assert after.time_s[left] == 1.0 and after.n_hits[left] == 2


def test_an_existence_policy_confirms_a_steady_target_and_deletes_a_one_off() -> None:
    """JIPDA: a target seen every scan climbs past 0.95; a plot seen once falls under 0.05."""
    motion = CartesianMotion("x", sigma_acceleration_mps2=1.0, acceleration_correlation_time_s=0.5)
    tracker = Tracker(
        {"x": LinearMeasurement(motion.state_layout, ("x_m",))},
        motion,
        TaylorLinearisation(),
        0.99,
        JointProbabilisticDataAssociation(0.9, 0.99),
        DirectStateInitiator([0.0, 0.0], np.diag([1.0, 30.0**2])),
        existence=ExistencePolicy(),
    )
    clutter = [1e-3]
    tracker.process(Scan(0.0, [[0.0], [5000.0]], [[1.0]], "x", clutter * 2))
    # Ten scans of the one target, and the one-off plot never again.
    for t in range(1, 11):
        bank = tracker.process(Scan(float(t), [[20.0 * t]], [[1.0]], "x", clutter))
    assert bank.n_tracks == 1 and bank.status[0] == TrackStatus.CONFIRMED
    assert bank.existence[0] > 0.95


def test_the_existence_policy_needs_an_associator_that_weights() -> None:
    tracker = _one_axis(LifecyclePolicy(1, 1), existence=ExistencePolicy())
    with pytest.raises(ValueError, match="existence"):
        tracker.process(Scan(0.0, [[0.0]], [[1.0]], "x", [1e-3]))


@pytest.mark.parametrize(
    "settings",
    [{"delete": 0.5, "initial": 0.2}, {"confirm": 0.01}, {"p_survival": 0.0}],
)
def test_an_existence_policy_that_could_not_work_is_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="delete"):
        ExistencePolicy(**settings)


def test_a_confirmed_track_whose_position_sigma_passes_the_limit_is_deleted() -> None:
    """Covariance deletion (tracker-001 §5.4): a coasting track's sigma grows past 150 m.

    Born with a 100 m/s velocity spread, its position sigma is 100 m a second later and about
    200 m the next, well before the ten misses its policy allows.
    """
    tracker = _one_axis(LifecyclePolicy(1, 1, 10, max_position_sigma_m=150.0))
    tracker.process(_scan(0.0, [0.0]))
    statuses = []
    # Coast until the row is reported deleted.
    for t in range(1, 10):
        statuses += tracker.process(_scan(float(t), [])).status.tolist()
    assert statuses == [TrackStatus.COASTING, TrackStatus.DELETED]


def test_no_track_starts_where_the_clutter_is_denser_than_the_birth_limit() -> None:
    """Births only from the measurement whose clutter intensity is at most the limit."""
    tracker = _one_axis(LifecyclePolicy(1, 1), max_birth_clutter_intensity=0.01)
    bank = tracker.process(Scan(0.0, [[0.0], [100.0], [200.0]], [[1.0]], "x", [0.001, 0.02, 0.01]))
    np.testing.assert_array_equal(bank.measurement_index, [0, 2])


def _one_axis_jpda(policy: LifecyclePolicy, existence: ExistencePolicy | None = None) -> Tracker:
    """The one-axis tracker of _one_axis with JPDA, gate 0.99, Pd 0.9."""
    motion = CartesianMotion("x", sigma_acceleration_mps2=1.0, acceleration_correlation_time_s=0.5)
    return Tracker(
        {"x": LinearMeasurement(motion.state_layout, ("x_m",))},
        motion,
        TaylorLinearisation(),
        0.99,
        JointProbabilisticDataAssociation(0.9, 0.99),
        DirectStateInitiator([0.0, 0.0], np.diag([1.0, 30.0**2])),
        policy,
        existence=existence,
    )


def test_a_reacquisition_reset_reaches_every_imm_mode() -> None:
    """An IMM predicts from its modes, so the reset must reset each mode, not only the mixture.

    Two CV modes; four hits, then misses: on the second miss, which opens re-acquisition, every
    mode's rate variance is the prior's and its cross-covariance zero.
    """
    layout_motion = CartesianMotion("x", sigma_acceleration_mps2=1.0)
    imm = InteractingMultipleModel(
        (layout_motion, CartesianMotion("x", sigma_acceleration_mps2=3.0)), (60.0, 20.0)
    )
    prior = np.diag([1.0, 100.0**2])
    tracker = Tracker(
        {"x": LinearMeasurement(imm.state_layout, ("x_m",))},
        imm,
        TaylorLinearisation(),
        0.997,
        GlobalNearestNeighbour(),
        DirectStateInitiator([0.0, 0.0], prior),
        LifecyclePolicy(1, 1, 2, 3, reacquire_reset=("xdot_mps",)),
    )
    # Four hits along x = 20 t, then two misses.
    for t in range(4):
        tracker.process(_scan(float(t), [20.0 * t]))
    tracker.process(_scan(4.0, []))
    bank = tracker.process(_scan(5.0, []))
    assert bank.n_misses[0] == 2 and bank.status[0] == TrackStatus.COASTING
    np.testing.assert_array_equal(bank.mode_covariance[0, :, 1, 1], prior[1, 1])
    np.testing.assert_array_equal(bank.mode_covariance[0, :, 0, 1], 0.0)
    np.testing.assert_array_equal(bank.covariance[0, 1, 1], prior[1, 1])


def test_a_jpda_track_whose_gate_holds_only_an_unlikely_plot_misses() -> None:
    """The hit is the measurement taken (weights over one half), not any plot in the gate.

    At 3.4 sigma-units in dense clutter (lambda 0.5) the plot weighs 0.2: a miss, and no
    measurement recorded. At 0.1 it weighs 0.8: a hit.
    """
    for z, hit in ((3.4, False), (0.1, True)):
        tracker = _one_axis_jpda(LifecyclePolicy(1, 1, 5))
        tracker.seed([[0.0, 0.0]], [np.diag([1.0, 0.01])], 0.0)
        bank = tracker.process(Scan(1.0, [[z]], [[1.0]], "x", [0.5]))
        assert (bank.n_misses[0] == 0) == hit
        assert (bank.measurement_index[0] == 0) == hit


def test_a_weighting_tracker_refuses_a_scan_without_clutter_intensity() -> None:
    """JPDA needs lambda at each measurement; the message names the Scan's field."""
    tracker = _one_axis_jpda(LifecyclePolicy(1, 1))
    with pytest.raises(ValueError, match="clutter_intensity"):
        tracker.process(Scan(0.0, [[0.0]], [[1.0]], "x"))


def test_existence_stays_at_most_one_when_survival_is_certain() -> None:
    """With p_survival = 1 a steady target's existence rounds towards 1 + eps; it is clipped."""
    tracker = _one_axis_jpda(LifecyclePolicy(1, 1), ExistencePolicy(p_survival=1.0))
    # Thirty scans of one target in light clutter: existence climbs to one and stays.
    for t in range(30):
        bank = tracker.process(Scan(float(t), [[20.0 * t]], [[1.0]], "x", [1e-4]))
    assert bank.n_tracks == 1
    assert np.all((bank.existence > 0) & (bank.existence <= 1.0))
    assert bank.existence[0] > 0.999
