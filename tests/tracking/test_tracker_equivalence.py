"""Temporary: the old ``core.tracking`` tracker and the new batched one agree on one stream.

The integration PR that rewires the pipelines to ``radar_forge.tracking`` deletes this file,
with ``core/tracking/``. Until then it is the evidence that the move changed no tracking
result that matters.

Both trackers see the same seeded stream, 60 scans of a straight-line target with Gaussian
noise plus Poisson clutter, with equal tuning: an unscented filter, a 0.997 gate, NIS costs,
GNN with confirmed tracks first, and 3-of-5 confirmation. The old tracker states one-way range
and a closing-positive rate; the new one bistatic range (twice the one-way range, monostatic)
and a receding-positive rate. So at the boundary the new tracker's measurements are
``(2 r, -2 r')``, its noise, prior and acceleration standard deviation are doubled to match,
and its estimates are halved, and the rate negated, before comparison. That map is linear, so
the two filters are the same filter in different units, and only assignment tie-breaks may
differ: the test asserts tolerances, not identical tracks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from radar_forge.core.tracking import (
    CartesianPosition,
    RadialMotion,
    StateEstimate,
    build_tracker,
)
from radar_forge.core.tracking import LifecyclePolicy as OldPolicy
from radar_forge.tracking.association import GlobalNearestNeighbour
from radar_forge.tracking.filters import TrackStatus, UnscentedTransform
from radar_forge.tracking.initiation import DirectStateInitiator
from radar_forge.tracking.measurement_models import LinearMeasurement
from radar_forge.tracking.models import RadialConstantVelocity
from radar_forge.tracking.tracker import LifecyclePolicy, Scan, Tracker

SEED = 20261104
N_SCANS = 60
SIGMA_RANGE_M, SIGMA_RATE_MPS = 20.0, 1.0
SIGMA_ACCELERATION_MPS2, CORRELATION_TIME_S = 2.0, 0.5
V_MAX_MPS, CLUTTER_PER_SCAN = 200.0, 5.0
# The boundary map from one-way range and closing rate to bistatic range and receding rate.
TO_BISTATIC = np.array([2.0, -2.0])


def _stream() -> list[tuple[float, float, np.ndarray]]:
    """Truth (one-way range, closing rate) and the scan's detections, in old units."""
    rng = np.random.default_rng(SEED)
    stream = []
    for scan in range(N_SCANS):
        # Scans are generated in time order, each from one generator.
        range_m, closing_mps = 10_000.0 - 80.0 * scan, 80.0
        target = [
            range_m + rng.normal(0.0, SIGMA_RANGE_M),
            closing_mps + rng.normal(0.0, SIGMA_RATE_MPS),
        ]
        n_clutter = rng.poisson(CLUTTER_PER_SCAN)
        clutter = np.column_stack(
            [rng.uniform(0.0, 20_000.0, n_clutter), rng.uniform(-V_MAX_MPS, V_MAX_MPS, n_clutter)]
        )
        detections = rng.permutation(np.vstack([target, clutter]))
        stream.append((range_m, closing_mps, detections))
    return stream


def _old_run(stream: Any) -> list[list[tuple[int, int, float, float]]]:
    motion = RadialMotion(SIGMA_ACCELERATION_MPS2, CORRELATION_TIME_S)
    observation = CartesianPosition(motion.state_layout, ("range_m", "range_rate_mps"))
    prior = StateEstimate(np.zeros(2), np.diag([1.0, V_MAX_MPS**2]), 0.0, motion.state_layout)
    tracker = build_tracker(
        motion, observation, prior, policy=OldPolicy(3, 5, 5), gate_probability=0.997
    )
    sensor = tracker.sensors["sensor"]
    noise = np.diag([SIGMA_RANGE_M**2, SIGMA_RATE_MPS**2])
    statuses = {
        "tentative": TrackStatus.TENTATIVE,
        "confirmed": TrackStatus.CONFIRMED,
        "coasting": TrackStatus.COASTING,
        "deleted": TrackStatus.DELETED,
    }
    history = []
    for scan, (_, _, detections) in enumerate(stream):
        # Scans are processed in time order; each depends on the one before.
        batch = sensor.batch(float(scan), [("measurement", d, noise) for d in detections])
        snapshots = tracker.process(batch)
        history.append(
            [(s.track_id, int(statuses[s.status]), s.state[0], s.state[1]) for s in snapshots]
        )
    return history


def _new_run(stream: Any) -> list[list[tuple[int, int, float, float]]]:
    motion = RadialConstantVelocity(
        2 * SIGMA_ACCELERATION_MPS2, acceleration_correlation_time_s=CORRELATION_TIME_S
    )
    model = LinearMeasurement(motion.state_layout, ("bistatic_range_m", "bistatic_range_rate_mps"))
    initiator = DirectStateInitiator(np.zeros(2), np.diag(TO_BISTATIC**2 * [1.0, V_MAX_MPS**2]))
    tracker = Tracker(
        {"radar": model},
        motion,
        UnscentedTransform(),
        0.997,
        GlobalNearestNeighbour(),
        initiator,
        LifecyclePolicy(3, 5, 5),
    )
    noise = np.diag(TO_BISTATIC**2 * [SIGMA_RANGE_M**2, SIGMA_RATE_MPS**2])
    history = []
    for scan, (_, _, detections) in enumerate(stream):
        # Scans are processed in time order; each depends on the one before.
        bank = tracker.process(Scan(float(scan), detections * TO_BISTATIC, noise, "radar"))
        one_way = bank.mean / TO_BISTATIC
        history.append(
            list(
                zip(
                    bank.track_id.tolist(),
                    bank.status.tolist(),
                    one_way[:, 0],
                    one_way[:, 1],
                    strict=True,
                )
            )
        )
    return history


def _summary(history: Any, stream: Any) -> tuple[int, int, float]:
    """Confirmed track count, the scan the target's track is confirmed, and its range RMSE."""
    confirmed = {"ids": set(), "scan": None, "errors": []}
    live = (TrackStatus.CONFIRMED, TrackStatus.COASTING)
    for scan, (rows, (range_m, _, _)) in enumerate(zip(history, stream, strict=True)):
        # A loop over scans of recorded output: each scan's rows are a handful of tracks.
        confirmed["ids"].update(i for i, status, _, _ in rows if status in live)
        on_target = [
            r - range_m
            for _, status, r, _ in rows
            if status in live and abs(r - range_m) < 5 * SIGMA_RANGE_M
        ]
        if on_target:
            confirmed["scan"] = scan if confirmed["scan"] is None else confirmed["scan"]
            confirmed["errors"].append(min(on_target, key=abs))
    rmse_m = float(np.sqrt(np.mean(np.square(confirmed["errors"]))))
    return len(confirmed["ids"]), int(confirmed["scan"]), rmse_m


@pytest.fixture(scope="module")
def summaries() -> tuple[tuple[int, int, float], tuple[int, int, float]]:
    """(old, new) summaries of the one seeded stream."""
    stream = _stream()
    return _summary(_old_run(stream), stream), _summary(_new_run(stream), stream)


def test_both_trackers_confirm_the_same_number_of_tracks(summaries: Any) -> None:
    old, new = summaries
    assert new[0] == old[0]


def test_both_trackers_confirm_the_target_within_one_scan_of_each_other(summaries: Any) -> None:
    old, new = summaries
    assert abs(new[1] - old[1]) <= 1


def test_both_trackers_place_the_target_within_five_percent_rmse(summaries: Any) -> None:
    """Tie-breaking may differ, so the bound is the brief's 5 %, not roundoff."""
    old, new = summaries
    assert new[2] == pytest.approx(old[2], rel=0.05)
