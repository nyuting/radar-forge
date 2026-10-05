"""Tests for radar_forge.core.tracking.lifecycle: M-of-N confirmation and deletion.

The manager is driven directly with ``record``, one call per scan, so each
test can count hits and misses by hand against the M-of-N rule. Most tests
use 4-of-5, as main's tracker does, because it makes "M unreachable" happen
after only two misses.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.initiation import DirectStateInitiator
from radar_forge.core.tracking.lifecycle import LifecyclePolicy, TrackManager
from radar_forge.core.tracking.measurement_models import Measurement
from radar_forge.core.tracking.motion import CartesianMotion
from radar_forge.core.tracking.tracks import Track, TrackStatus
from radar_forge.core.tracking.ukf import UKF

ORIGIN = (36.00250, -78.94100, 60.0)
MOTION = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
PRIOR = StateEstimate(np.zeros(2), np.diag([1.0, 100.0**2]), 0.0, MOTION.state_layout)


def manager(**policy: int | float) -> TrackManager:
    """A manager with the given policy fields and defaults for the rest."""
    initiator = DirectStateInitiator(lambda state: UKF(state, MOTION), PRIOR)
    return TrackManager(initiator, LifecyclePolicy(**policy))  # type: ignore[arg-type]


def hit(time_s: float) -> Measurement:
    """A measurement at ``time_s``, to record as a hit."""
    return Measurement(np.array([0.0]), np.array([[1.0]]), time_s, "sensor", "measurement")


def run(track_manager: TrackManager, scans: str) -> tuple[Track, list[TrackStatus]]:
    """Seed a track, then record ``scans`` ("h" hit, "m" miss); return statuses.

    The first status is the track's at birth, the rest one per scan.
    """
    track = track_manager.seed(UKF(PRIOR, MOTION), "sensor")
    statuses = [track.status]
    # Each record depends on the counts the previous one left behind.
    for scan, kind in enumerate(scans, start=1):
        track_manager.record(track, hit(float(scan)) if kind == "h" else None)
        statuses.append(track.status)
    return track, statuses


TENTATIVE, CONFIRMED, DELETED = TrackStatus.TENTATIVE, TrackStatus.CONFIRMED, TrackStatus.DELETED


def test_a_track_is_confirmed_on_the_mth_hit() -> None:
    """3-of-5 with a miss in between: birth, hit, miss, hit is the third hit."""
    _, statuses = run(manager(n_confirm_hits=3), "hmh")
    assert statuses == [TENTATIVE, TENTATIVE, TENTATIVE, CONFIRMED]


def test_an_isolated_false_alarm_is_deleted_and_never_confirmed() -> None:
    """One detection, then nothing: deleted within its first N scans."""
    _, statuses = run(manager(n_confirm_hits=4), "mmmm")
    assert CONFIRMED not in statuses
    assert DELETED in statuses[: 5 + 1]


def test_a_tentative_track_dies_as_soon_as_m_is_unreachable() -> None:
    """4-of-5: after two misses, 1 hit + 2 scans left cannot make 4."""
    _, statuses = run(manager(n_confirm_hits=4), "mm")
    assert statuses == [TENTATIVE, TENTATIVE, DELETED]


def test_a_tentative_track_out_of_frames_is_deleted_on_the_last_one() -> None:
    """4-of-5: three hits, then two misses. M stays reachable until scan 5 ends.

    After the first miss, 3 hits and 1 scan left can still make 4, so the
    track lives. The second miss uses up the fifth scan, so the deadline itself
    deletes it.
    """
    _, statuses = run(manager(n_confirm_hits=4), "hhmm")
    assert statuses == [TENTATIVE, TENTATIVE, TENTATIVE, TENTATIVE, DELETED]


def test_a_hit_after_the_first_n_scans_cannot_save_a_tentative_track() -> None:
    """The window does not slide: a track that is deleted stays deleted."""
    track, _ = run(manager(n_confirm_hits=3), "mmmhh")
    assert track.status == DELETED


def test_a_tentative_track_that_misses_is_deleted_with_no_grace_period() -> None:
    """n_delete_misses = 1: one miss deletes a tentative track outright.

    By M-of-N alone it would live: 1 hit and 3 scans left can still make 3.
    """
    _, statuses = run(manager(n_confirm_hits=3, n_delete_misses=1), "m")
    assert statuses == [TENTATIVE, DELETED]


def test_a_confirmed_track_coasts_until_n_delete_misses() -> None:
    """3-of-5 and 5 misses allowed: confirmed on scan 2, deleted on the fifth miss."""
    _, statuses = run(manager(n_confirm_hits=3, n_delete_misses=5), "hhmmmmm")
    assert statuses == [TENTATIVE, TENTATIVE, CONFIRMED] + [CONFIRMED] * 4 + [DELETED]


def test_a_hit_resets_the_miss_count() -> None:
    track, statuses = run(manager(n_confirm_hits=3, n_delete_misses=3), "hhmmhmm")
    assert statuses[-1] == CONFIRMED
    assert track.n_misses == 2


def test_a_one_of_n_policy_confirms_a_track_at_birth() -> None:
    _, statuses = run(manager(n_confirm_hits=1), "")
    assert statuses == [CONFIRMED]


def test_track_ids_count_up_from_one() -> None:
    track_manager = manager()
    ids = [track_manager.seed(UKF(PRIOR, MOTION)).track_id for _ in range(3)]
    assert ids == ["1", "2", "3"]


def test_a_gap_of_exactly_max_coast_time_keeps_the_track() -> None:
    track_manager = manager(max_coast_time_s=30.0)
    track, _ = run(track_manager, "")
    track_manager.expire(track, 30.0)
    assert track.status == TENTATIVE


def test_a_gap_longer_than_max_coast_time_deletes_the_track() -> None:
    track_manager = manager(max_coast_time_s=30.0)
    track, _ = run(track_manager, "")
    track_manager.expire(track, 30.5)
    assert track.status == DELETED


@pytest.mark.parametrize(
    ("policy", "match"),
    [
        ({"n_confirm_hits": 6, "n_confirm_frames": 5}, "n_confirm_hits"),
        ({"n_confirm_hits": 0}, "n_confirm_hits"),
        ({"n_delete_misses": 0}, "n_delete_misses"),
        ({"n_history": -1}, "n_history"),
        ({"max_coast_time_s": 0.0}, "max_coast_time_s"),
        ({"max_coast_time_s": float("inf")}, "max_coast_time_s"),
        ({"n_confirm_hits": True}, "int"),
        ({"n_confirm_frames": 5.0}, "int"),
    ],
    ids=[
        "M above N",
        "M of zero",
        "deletion count below one",
        "negative history",
        "zero coast time",
        "infinite coast time",
        "bool count",
        "float count",
    ],
)
def test_the_policy_rejects_an_impossible_setting(
    policy: dict[str, int | float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        LifecyclePolicy(**policy)  # type: ignore[arg-type]
