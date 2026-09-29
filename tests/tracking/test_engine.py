import numpy as np
import pytest

from radar_forge.pipelines.tracking_config import TrackingConfig, build_enu_tracker
from radar_forge.tracking import GlobalNearestNeighbour, MeasurementBatch, Sensor, TrackStatus


def batch(engine, time_s, positions):
    return engine.sensors["sensor"].batch(
        time_s, [("measurement", np.array([p]), np.array([[0.1]])) for p in positions]
    )


def test_gnn_prioritizes_cardinality_and_handles_negative_likelihood_costs():
    result = GlobalNearestNeighbour().associate(np.array([[-100.0, 0.0], [-99.0, np.inf]]))
    assert set(result.matches) == {(0, 1), (1, 0)}
    empty = GlobalNearestNeighbour().associate(np.full((2, 0), np.inf))
    assert empty.unassigned_tracks == (0, 1)


def test_confirmation_coasting_deletion_history_and_detached_snapshots():
    engine = build_enu_tracker(
        {"x": "CV"},
        origin_lla_deg_m=(36.00250, -78.94100, 60.0),
        config=TrackingConfig(history_size=2),
    )
    snapshots = []
    for time_s in range(3):
        snapshots = engine.process(batch(engine, time_s, [100 + 3 * time_s]))
    assert snapshots[0].status == TrackStatus.CONFIRMED
    assert len(engine.tracks[0].history) == 2
    snapshots[0].state.setflags(write=True)
    snapshots[0].state[:] = -10000
    assert engine.tracks[0].estimator.state.mean[0] > 100
    for time_s in range(3, 7):
        assert engine.process(MeasurementBatch(time_s, "sensor"))[0].status == TrackStatus.CONFIRMED
    deleted = engine.process(MeasurementBatch(7, "sensor"))
    assert deleted[0].status == TrackStatus.DELETED
    assert not engine.process(MeasurementBatch(8, "sensor"))


def test_equal_time_asynchronous_scans_coverage_and_oosm_validation():
    engine = build_enu_tracker({"x": "CV"}, origin_lla_deg_m=(36.00250, -78.94100, 60.0))
    engine.process(batch(engine, 1, [100]))
    prior = engine.tracks[0].snapshot()
    with pytest.raises(ValueError, match="out-of-sequence"):
        engine.process(batch(engine, 0, [0]))
    np.testing.assert_allclose(engine.tracks[0].snapshot().state, prior.state, rtol=1e-12)
    other = Sensor("other", ("measurement",), observable=lambda _: False)
    engine.sensors[other.id] = other
    engine.process(MeasurementBatch(1, "other"))
    assert engine.tracks[0].miss_count == 0
    assert engine.tracks[0].hit_count == 1
    engine.process(batch(engine, 1, [100]))
    assert engine.tracks[0].hit_count == 2
    before = engine.tracks[0].snapshot()
    wrong = engine.sensors["sensor"].batch(2, [("measurement", np.zeros(2), np.eye(2))])
    with pytest.raises(ValueError, match="dimension"):
        engine.process(wrong)
    np.testing.assert_allclose(engine.tracks[0].snapshot().state, before.state, rtol=1e-12)


def test_coast_limit_is_strict_and_expires_before_reassociation():
    engine = build_enu_tracker({"x": "CV"}, origin_lla_deg_m=(36.00250, -78.94100, 60.0))
    first = engine.process(batch(engine, 0, [100]))[0].track_id
    assert engine.process(batch(engine, 30, [100]))[0].track_id == first
    result = engine.process(batch(engine, 61, [100]))
    assert result[0].status == TrackStatus.DELETED
    assert result[1].track_id != first


def test_crossing_targets_keep_identity_using_velocity_predictions():
    engine = build_enu_tracker(
        {"x": "CV"}, origin_lla_deg_m=(36.00250, -78.94100, 60.0), noise_density=0.01
    )
    for time_s in range(12):
        # Cross between scans, avoiding an unobservable exact-coincidence identity claim.
        engine.process(batch(engine, time_s, [100 + 3 * time_s, 131 - 3 * time_s]))
    assert len(engine.tracks) == 2
    assert engine.tracks[0].estimator.state.mean[1] > 2.5
    assert engine.tracks[1].estimator.state.mean[1] < -2.5
