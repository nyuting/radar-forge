"""Tests for radar_forge.core.tracking.tracks: live tracks and their snapshots."""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.motion import CartesianMotion
from radar_forge.core.tracking.tracks import TRACK_STATUSES, Track
from radar_forge.core.tracking.ukf import UKF

ORIGIN = (36.00250, -78.94100, 60.0)
MOTION = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)


def track() -> Track[UKF]:
    """A track at x = 100 m moving at 3 m/s, at time 0."""
    state = StateEstimate(np.array([100.0, 3.0]), np.diag([1.0, 4.0]), 0.0, MOTION.state_layout)
    return Track(1, UKF(state, MOTION), source_sensor_ids={"sensor"})


def test_snapshot_state_is_the_mean_only() -> None:
    snapshot = track().snapshot()
    assert snapshot.state.shape == (2,)
    # rtol 1e-12: a copy of two numbers, no arithmetic.
    np.testing.assert_allclose(snapshot.state, [100.0, 3.0], rtol=1e-12)


def test_snapshot_covariance_is_the_full_matrix() -> None:
    snapshot = track().snapshot()
    # rtol 1e-12: a copy of the prior's matrix, no arithmetic.
    np.testing.assert_allclose(snapshot.covariance, np.diag([1.0, 4.0]), rtol=1e-12)


def test_a_snapshot_shares_no_memory_with_the_filter() -> None:
    live = track()
    snapshot = live.snapshot()
    assert not np.shares_memory(snapshot.state, live.estimator.state.mean)
    assert not np.shares_memory(snapshot.covariance, live.estimator.state.covariance)


def test_a_snapshot_is_read_only() -> None:
    snapshot = track().snapshot()
    with pytest.raises(ValueError, match="read-only"):
        snapshot.state[0] = -1.0


def test_a_snapshot_does_not_change_when_the_track_moves_on() -> None:
    live = track()
    snapshot = live.snapshot()
    live.estimator.predict_to(10.0)
    live.status = "confirmed"
    assert (snapshot.timestamp_s, snapshot.status) == (0.0, "tentative")
    # rtol 1e-12: the snapshot must hold exactly what it held before.
    np.testing.assert_allclose(snapshot.state, [100.0, 3.0], rtol=1e-12)


def test_a_snapshot_freezes_its_source_sensors() -> None:
    live = track()
    snapshot = live.snapshot()
    live.source_sensor_ids.add("second")
    assert snapshot.source_sensor_ids == frozenset({"sensor"})


def test_snapshot_time_is_the_estimate_time_not_the_last_measurement() -> None:
    live = track()
    live.estimator.predict_to(2.0)
    snapshot = live.snapshot()
    assert (snapshot.timestamp_s, snapshot.last_measurement_time_s) == (2.0, 0.0)


@pytest.mark.parametrize(
    ("status", "alive", "confirmed"),
    [
        ("tentative", True, False),
        ("confirmed", True, True),
        ("coasting", True, True),
        ("deleted", False, False),
    ],
)
def test_alive_and_confirmed_follow_the_status(status: str, alive: bool, confirmed: bool) -> None:
    live = track()
    live.status = status  # type: ignore[assignment]
    assert (live.is_alive, live.is_confirmed) == (alive, confirmed)


def test_track_statuses_list_every_status_in_life_order() -> None:
    assert TRACK_STATUSES == ("tentative", "confirmed", "coasting", "deleted")
