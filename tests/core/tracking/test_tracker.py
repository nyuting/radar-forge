"""Tests for radar_forge.core.tracking.tracker: the scan loop and its builders.

Most tests use a one-axis tracker (``{"x": "CV"}``) with the default 3-of-5
policy, so each property can be checked against a short, hand-countable run
of scans. The clutter test is the one statistical test; it is seeded from the
module's ``rng`` fixture.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import pytest

from radar_forge.core.tracking.estimation import InnovationStats
from radar_forge.core.tracking.lifecycle import LifecyclePolicy
from radar_forge.core.tracking.measurement_models import Measurement, MeasurementBatch, SensorRoute
from radar_forge.core.tracking.tracker import Tracker, build_tracker_enu
from radar_forge.core.tracking.tracks import TrackSnapshot

ORIGIN = (36.00250, -78.94100, 60.0)
TENTATIVE, CONFIRMED, COASTING, DELETED = "tentative", "confirmed", "coasting", "deleted"


@pytest.fixture
def rng() -> np.random.Generator:
    """One seeded generator, so a failure is reproducible."""
    return np.random.default_rng(20261005)


def batch(
    tracker: Tracker,
    time_s: float,
    positions_m: Sequence[float],
    variance_m2: float = 0.1,
    sensor_id: str = "sensor",
) -> MeasurementBatch:
    """One scan of x-position measurements, all with the same variance."""
    return tracker.sensors[sensor_id].batch(
        time_s,
        [("measurement", np.array([p]), np.array([[variance_m2]])) for p in positions_m],
    )


def one_axis_tracker(**kwargs: object) -> Tracker:
    """A tracker on the x axis only, at constant velocity."""
    return build_tracker_enu({"x": "CV"}, origin_lla_deg_m=ORIGIN, **kwargs)  # type: ignore[arg-type]


def confirmed_tracker(n_scans: int = 5) -> Tracker:
    """A tracker holding one confirmed track on a target moving at 3 m/s.

    Scans 0 to ``n_scans - 1`` each hold one noiseless detection. With 3-of-5
    the track is confirmed on scan 2; the extra scans settle its velocity.
    """
    tracker = one_axis_tracker()
    # Scans are processed in time order; each depends on the one before.
    for time_s in range(n_scans):
        tracker.process(batch(tracker, time_s, [100.0 + 3.0 * time_s]))
    return tracker


# --------------------------------------------------------------------------- #
# Confirmation, coasting and deletion through the scan loop
# --------------------------------------------------------------------------- #


def test_a_track_is_confirmed_on_the_third_scan() -> None:
    """3-of-5: the birth scan is hit one, so scan 2 is hit three."""
    tracker = one_axis_tracker()
    statuses = [tracker.process(batch(tracker, t, [100.0 + 3.0 * t]))[0].status for t in range(3)]
    assert statuses == [TENTATIVE, TENTATIVE, CONFIRMED]


def test_no_track_stays_tentative_longer_than_its_first_n_scans() -> None:
    """Regression for review #10: a sliding M-of-N window kept a track tentative forever.

    One target, detected on every third scan: hit, miss, miss, hit, miss, ...
    Such a track never has 3 hits in any 5 scans. Counted over its first five
    scans, as the M-of-N rule is meant to be, it must be deleted by then, and a
    new track is started at the next detection. With a sliding window, track 1
    stayed tentative for all 60 scans.
    """
    tracker = one_axis_tracker()
    first_tentative: dict[str, int] = {}
    longest = 0
    # Scans are processed in time order; each depends on the one before.
    for t in range(60):
        if t % 3 == 0:
            snapshots = tracker.process(batch(tracker, t, [1000.0 + 20.0 * t]))
        else:
            snapshots = tracker.process(MeasurementBatch(t, "sensor"))
        for snapshot in snapshots:
            if snapshot.status == TENTATIVE:
                first_tentative.setdefault(snapshot.track_id, t)
                longest = max(longest, t - first_tentative[snapshot.track_id] + 1)
    assert longest <= 5


def test_a_confirmed_track_coasts_through_a_miss_by_dead_reckoning() -> None:
    """With no measurement, a CV track moves on by velocity times the time step."""
    tracker = confirmed_tracker()
    before = tracker.tracks[0].snapshot()
    after = tracker.process(MeasurementBatch(5, "sensor"))[0]
    assert after.status == COASTING
    # rtol 1e-9: the UKF reproduces a linear prediction to round-off, and these
    # are a handful of float64 operations on numbers near 100.
    np.testing.assert_allclose(after.state[0], before.state[0] + before.state[1], rtol=1e-9)


def test_coasting_grows_the_covariance() -> None:
    tracker = confirmed_tracker()
    before = tracker.tracks[0].snapshot().covariance[0, 0]
    after = tracker.process(MeasurementBatch(5, "sensor"))[0].covariance[0, 0]
    assert after > before


def test_a_confirmed_track_survives_one_miss_fewer_than_the_limit() -> None:
    """n_delete_misses = 5: four misses in a row leave the track alive, coasting."""
    tracker = confirmed_tracker()
    # Scans are processed in time order; each depends on the one before.
    for time_s in range(5, 9):
        snapshot = tracker.process(MeasurementBatch(time_s, "sensor"))[0]
    assert snapshot.status == COASTING


def test_a_coasting_track_is_confirmed_again_by_a_hit() -> None:
    tracker = confirmed_tracker()
    tracker.process(MeasurementBatch(5, "sensor"))
    assert tracker.process(batch(tracker, 6, [118.0]))[0].status == CONFIRMED


def test_a_confirmed_track_is_reported_deleted_on_the_last_allowed_miss() -> None:
    tracker = confirmed_tracker()
    # Scans are processed in time order; each depends on the one before.
    for time_s in range(5, 9):
        tracker.process(MeasurementBatch(time_s, "sensor"))
    assert tracker.process(MeasurementBatch(9, "sensor"))[0].status == DELETED


def test_a_deleted_track_is_reported_only_once() -> None:
    tracker = confirmed_tracker()
    # Scans are processed in time order; each depends on the one before.
    for time_s in range(5, 10):
        tracker.process(MeasurementBatch(time_s, "sensor"))
    assert tracker.process(MeasurementBatch(10, "sensor")) == ()


def test_a_gap_of_exactly_max_coast_time_keeps_the_track() -> None:
    """The coast limit is strict: a gap equal to it is still allowed."""
    tracker = one_axis_tracker()
    first = tracker.process(batch(tracker, 0, [100.0]))[0].track_id
    assert tracker.process(batch(tracker, 30, [100.0]))[0].track_id == first


def test_a_stale_track_is_deleted_before_it_can_take_a_measurement() -> None:
    """A track past its coast limit is removed first, so the detection starts a new one."""
    tracker = one_axis_tracker()
    first = tracker.process(batch(tracker, 0, [100.0]))[0].track_id
    result = tracker.process(batch(tracker, 31, [100.0]))
    assert [(s.track_id == first, s.status) for s in result] == [
        (True, DELETED),
        (False, TENTATIVE),
    ]


def test_history_keeps_only_the_last_n_history_snapshots() -> None:
    tracker = one_axis_tracker(policy=LifecyclePolicy(n_history=2))
    # Scans are processed in time order; each depends on the one before.
    for time_s in range(4):
        tracker.process(batch(tracker, time_s, [100.0 + 3.0 * time_s]))
    assert [s.timestamp_s for s in tracker.tracks[0].history] == [2.0, 3.0]


# --------------------------------------------------------------------------- #
# Association: who gets the measurement
# --------------------------------------------------------------------------- #


def stationary_confirmed_tracker() -> Tracker:
    """A confirmed track on a target sitting still at x = 1000 m, R = 1 m^2.

    The process noise is pinned at q = 2 sigma² tau = 1 m²/s³, so the track's gate stays
    narrow: at the default q = 32 m²/s³ a detection 15 m away is inside it, and no
    tentative track could start there for the tests below to use.
    """
    tracker = one_axis_tracker(sigma_acceleration_mps2=1.0, acceleration_correlation_time_s=0.5)
    # Scans are processed in time order; each depends on the one before.
    for time_s in range(6):
        tracker.process(batch(tracker, time_s, [1000.0], variance_m2=1.0))
    return tracker


def test_a_confirmed_track_gets_first_pick_over_a_tentative_one() -> None:
    """Review #7: a young track must not steal a confirmed track's detection.

    Scan 6 starts a tentative track 15 m from the confirmed one. In scan 7 the
    one detection is 3 m from the confirmed track and 12 m from the tentative
    one. The tentative track is still so uncertain that its NIS for that
    detection is far smaller (checked first, below), so one joint assignment
    by NIS would give it the detection. Confirmed tracks pick first, so the
    confirmed track keeps it.
    """
    tracker = stationary_confirmed_tracker()
    tracker.process(batch(tracker, 6, [1000.0, 1015.0], variance_m2=1.0))
    confirmed, tentative = tracker.tracks
    assert (confirmed.status, tentative.status) == (CONFIRMED, TENTATIVE)

    detection = Measurement(np.array([1003.0]), np.array([[1.0]]), 7.0, "sensor", "measurement")
    model = tracker.measurement_models["measurement"]
    nis = []
    # Two tracks, each with its own filter, copied so the check changes nothing.
    for track in (confirmed, tentative):
        estimator = copy.deepcopy(track.estimator)
        estimator.predict_to(7.0)
        stats = estimator.innovation_statistics(detection, model)
        assert tracker.gate.accepts(stats)
        nis.append(stats.nis)
    assert nis[1] < nis[0]

    tracker.process(batch(tracker, 7, [1003.0], variance_m2=1.0))
    assert (confirmed.last_measurement_time_s, tentative.last_measurement_time_s) == (7.0, 6.0)


def test_no_track_is_born_inside_a_confirmed_tracks_gate() -> None:
    """A second detection 1 m from a confirmed track is not a new target."""
    tracker = stationary_confirmed_tracker()
    snapshots = tracker.process(batch(tracker, 6, [1000.0, 1001.0], variance_m2=1.0))
    assert len(snapshots) == 1


def test_a_detection_outside_every_confirmed_gate_starts_a_track() -> None:
    tracker = stationary_confirmed_tracker()
    snapshots = tracker.process(batch(tracker, 6, [1000.0, 1100.0], variance_m2=1.0))
    assert [s.status for s in snapshots] == [CONFIRMED, TENTATIVE]


def test_the_update_reuses_the_innovation_computed_for_gating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contract C2: the tracker hands the gating innovation to ``update``."""
    tracker = one_axis_tracker()
    tracker.process(batch(tracker, 0, [100.0]))
    estimator = tracker.tracks[0].estimator
    received: list[InnovationStats | None] = []
    original = estimator.update

    def spy(
        measurement: Measurement, model: object, *, innovation: InnovationStats | None = None
    ) -> None:
        received.append(innovation)
        original(measurement, model, innovation=innovation)  # type: ignore[arg-type]

    monkeypatch.setattr(estimator, "update", spy)
    tracker.process(batch(tracker, 1, [101.0]))
    assert len(received) == 1
    assert received[0] is not None
    # rtol 1e-12: the residual is one subtraction, z minus the prediction.
    np.testing.assert_allclose(
        received[0].residual, 101.0 - received[0].predicted_measurement, rtol=1e-12
    )


def run_crossing_targets() -> tuple[list[list[int]], tuple[TrackSnapshot, ...]]:
    """Two targets crossing, seen by two sensors in turn; return IDs per scan."""
    # q = 2 sigma² tau = 0.01 m²/s³: the targets move almost exactly at constant velocity.
    tracker = one_axis_tracker(sigma_acceleration_mps2=0.1, acceleration_correlation_time_s=0.5)
    tracker.add_sensor(SensorRoute("second", ("measurement",)))
    identifiers: list[list[int]] = []
    snapshots: tuple[TrackSnapshot, ...] = ()
    # Alternate scans between the two sensors, and reverse the measurement
    # order on the second, so that association must follow the predicted
    # motion and not the list positions. The targets cross between scans,
    # never exactly at one, so which is which is always observable.
    # Scans are processed in time order; each depends on the one before.
    for scan in range(24):
        time_s = scan / 2
        sensor_id = "sensor" if scan % 2 == 0 else "second"
        positions_m = [100 + 3 * time_s, 131 - 3 * time_s]
        if scan % 2:
            positions_m.reverse()
        snapshots = tracker.process(batch(tracker, time_s, positions_m, sensor_id=sensor_id))
        identifiers.append([s.track_id for s in snapshots])
    return identifiers, snapshots


def test_crossing_targets_keep_their_identities_across_two_sensors() -> None:
    identifiers, _ = run_crossing_targets()
    assert all(ids == identifiers[0] for ids in identifiers)


def test_crossing_targets_both_end_confirmed() -> None:
    _, snapshots = run_crossing_targets()
    assert [s.status for s in snapshots] == [CONFIRMED] * 2


def test_each_crossing_track_is_updated_by_both_sensors() -> None:
    _, snapshots = run_crossing_targets()
    assert all(s.source_sensor_ids == frozenset({"sensor", "second"}) for s in snapshots)


def test_each_crossing_track_carries_its_own_targets_velocity() -> None:
    _, snapshots = run_crossing_targets()
    # atol 0.1 m/s on a 3 m/s velocity: the test only has to show that each
    # track carries its own target's velocity, +3 or -3, not how precisely.
    np.testing.assert_allclose([s.state[1] for s in snapshots], [3, -3], rtol=0, atol=0.1)


def test_a_target_keeps_one_track_through_clutter_and_missed_detections(
    rng: np.random.Generator,
) -> None:
    """P_d = 0.7 and three false alarms per scan: one confirmed track, one ID.

    P_d is the probability of detection: in 30 % of scans the target gives no
    detection at all. The false alarms are uniform over 5 km, so they rarely
    fall inside the target's gate. Adapted from review #18.
    """
    tracker = one_axis_tracker()
    identifiers = set()
    # Scans are processed in time order; each depends on the one before.
    for t in range(40):
        truth_m = 1000.0 + 20.0 * t
        positions_m = list(rng.uniform(0.0, 5000.0, 3))
        if rng.random() < 0.7:
            # The same sigma the batch declares (R = 0.1 m^2), or the gate is
            # too tight and the test fails for the test's own reason.
            positions_m.append(truth_m + rng.normal(0.0, np.sqrt(0.1)))
        rng.shuffle(positions_m)
        snapshots = tracker.process(batch(tracker, t, positions_m))
        near = [s for s in snapshots if abs(s.state[0] - truth_m) < 10.0]
        if t >= 10:
            # Confirmed or coasting: a missed detection makes the track coast.
            confirmed = [s for s in near if s.status in (CONFIRMED, COASTING)]
            assert len(confirmed) == 1
            identifiers.add(confirmed[0].track_id)
    assert len(identifiers) == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Several detections of one target in one scan each start a track, and GNN keeps "
        "all of them fed. Fixed on the detection side by PR B: one detection per target."
    ),
)
def test_several_detections_of_one_target_give_one_confirmed_track(
    rng: np.random.Generator,
) -> None:
    """Three detections per scan, scattered within one 74 m range bin.

    A CFAR detector can report one target several times, as scenario S1's does.
    A detection is only known to lie somewhere in its bin, so its variance is
    that of a uniform spread over the bin width: R = bin_width**2 / 12. All
    three detections arrive in the first scan, before any track is confirmed,
    so the tracker's rules against births inside confirmed gates cannot help.
    """
    bin_width_m = 74.0
    variance_m2 = bin_width_m**2 / 12.0
    tracker = one_axis_tracker()
    # Scans are processed in time order; each depends on the one before.
    for t in range(30):
        truth_m = 10_000.0 + 20.0 * t
        bin_start_m = np.floor(truth_m / bin_width_m) * bin_width_m
        positions_m = rng.uniform(bin_start_m, bin_start_m + bin_width_m, 3)
        snapshots = tracker.process(batch(tracker, t, positions_m, variance_m2=variance_m2))
    assert sum(s.status in (CONFIRMED, COASTING) for s in snapshots) == 1


# --------------------------------------------------------------------------- #
# Validation, sensors and time
# --------------------------------------------------------------------------- #


def test_an_out_of_sequence_batch_is_rejected() -> None:
    tracker = one_axis_tracker()
    tracker.process(batch(tracker, 1, [100.0]))
    with pytest.raises(ValueError, match="out-of-sequence"):
        tracker.process(batch(tracker, 0, [0.0]))


def test_a_rejected_batch_leaves_the_tracks_unchanged() -> None:
    tracker = one_axis_tracker()
    tracker.process(batch(tracker, 1, [100.0]))
    before = tracker.tracks[0].snapshot()
    with pytest.raises(ValueError, match="out-of-sequence"):
        tracker.process(batch(tracker, 0, [0.0]))
    after = tracker.tracks[0].snapshot()
    # rtol 1e-12: nothing should have touched the filter at all.
    np.testing.assert_allclose(after.state, before.state, rtol=1e-12)
    assert (after.timestamp_s, tracker.tracks[0].n_frames) == (before.timestamp_s, 1)


def test_a_time_within_the_tolerance_is_not_out_of_sequence() -> None:
    """The same time computed two ways, here 0.1 * 3 and 0.1 + 0.1 + 0.1."""
    tracker = one_axis_tracker()
    tracker.process(batch(tracker, 0.1 + 0.1 + 0.1, [100.0]))
    tracker.process(batch(tracker, 0.1 * 3 - 1e-9, [100.0]))
    assert tracker.tracks[0].n_hits == 2


def test_a_measurement_of_the_wrong_dimension_is_rejected() -> None:
    tracker = one_axis_tracker()
    wrong = tracker.sensors["sensor"].batch(0, [("measurement", np.zeros(2), np.eye(2))])
    with pytest.raises(ValueError, match="dimension"):
        tracker.process(wrong)


def test_a_measurement_whose_model_the_sensor_does_not_list_is_rejected() -> None:
    """A batch built by hand can skip SensorRoute.batch; process() still checks the sensor."""
    tracker = one_axis_tracker()
    stray = Measurement(np.zeros(1), np.eye(1), 0.0, "sensor", "other")
    with pytest.raises(ValueError, match="unregistered models"):
        tracker.process(MeasurementBatch(0.0, "sensor", (stray,)))


def test_a_batch_from_an_unknown_sensor_is_rejected() -> None:
    tracker = one_axis_tracker()
    with pytest.raises(ValueError, match="unknown sensor"):
        tracker.process(MeasurementBatch(0, "nobody"))


def test_a_track_the_sensor_cannot_see_records_no_miss() -> None:
    tracker = one_axis_tracker()
    tracker.add_sensor(SensorRoute("blind", ("measurement",), observable=lambda _: False))
    tracker.process(batch(tracker, 1, [100.0]))
    tracker.process(MeasurementBatch(1, "blind"))
    assert (tracker.tracks[0].n_frames, tracker.tracks[0].n_misses) == (1, 0)


def test_two_sensors_at_the_same_time_both_update_the_track() -> None:
    tracker = one_axis_tracker()
    tracker.add_sensor(SensorRoute("second", ("measurement",)))
    tracker.process(batch(tracker, 1, [100.0]))
    tracker.process(batch(tracker, 1, [100.0], sensor_id="second"))
    assert tracker.tracks[0].n_hits == 2


def test_the_sensor_registry_cannot_be_changed_directly() -> None:
    """Review #10: a sensor added this way would skip validation."""
    tracker = one_axis_tracker()
    with pytest.raises(TypeError):
        tracker.sensors["other"] = SensorRoute("other", ("measurement",))  # type: ignore[index]


def test_add_sensor_rejects_an_unregistered_model() -> None:
    tracker = one_axis_tracker()
    with pytest.raises(ValueError, match="registered measurement model"):
        tracker.add_sensor(SensorRoute("other", ("nonexistent",)))


def test_add_sensor_rejects_a_sensor_already_registered() -> None:
    tracker = one_axis_tracker()
    with pytest.raises(ValueError, match="already registered"):
        tracker.add_sensor(SensorRoute("sensor", ("measurement",)))


def test_the_constructor_rejects_a_key_that_is_not_the_sensor_id() -> None:
    tracker = one_axis_tracker()
    with pytest.raises(ValueError, match="sensor_id"):
        Tracker(
            tracker.measurement_models,
            {"wrong": SensorRoute("sensor", ("measurement",))},
            tracker.gate,
            tracker.associator,
            tracker.initiator,
        )


def test_the_constructor_rejects_an_unknown_cost() -> None:
    tracker = one_axis_tracker()
    with pytest.raises(ValueError, match="cost"):
        Tracker(
            tracker.measurement_models,
            tracker.sensors,
            tracker.gate,
            tracker.associator,
            tracker.initiator,
            cost="distance",  # type: ignore[arg-type]
        )


def test_seed_rejects_an_estimator_another_track_owns() -> None:
    tracker = one_axis_tracker()
    tracker.process(batch(tracker, 0, [100.0]))
    with pytest.raises(ValueError, match="independent estimator"):
        tracker.seed(tracker.tracks[0].estimator)


def test_seed_rejects_an_estimator_at_another_time() -> None:
    tracker = one_axis_tracker()
    tracker.process(batch(tracker, 0, [100.0]))
    estimator = copy.deepcopy(tracker.tracks[0].estimator)
    estimator.predict_to(1.0)
    with pytest.raises(ValueError, match="current time"):
        tracker.seed(estimator)


def test_build_tracker_rejects_an_unknown_association() -> None:
    with pytest.raises(ValueError, match="association"):
        build_tracker_enu(
            {"x": "CV"},
            origin_lla_deg_m=ORIGIN,
            association="JPDA",  # type: ignore[arg-type]
        )


def test_build_tracker_rejects_impossible_sigma_point_settings_before_any_track() -> None:
    with pytest.raises(ValueError, match="alpha"):
        one_axis_tracker(alpha=0.0)
