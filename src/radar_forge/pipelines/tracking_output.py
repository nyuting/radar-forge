"""Streaming tracking exports and evaluation separated from estimator input.

References
----------
.. [1] spec/tracker-001-integration.md; Norfair consumer separation.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

import numpy as np

from radar_forge.pipelines.general_tracking import ScenarioTracker, TrackingFrame
from radar_forge.tracking import TrackSnapshot, TrackStatus

__all__ = [
    "TrackingWriter",
    "snapshot_record",
]


class TrackingWriter:
    """Stream CSV records and accumulate bounded-memory single-target metrics.

    Parameters
    ----------
    out_dir : Path
        Output directory; generated artifacts must not be committed.
    tracker : ScenarioTracker
        Supplies interpretation and resolved configuration only.

    Notes
    -----
    Evaluation chooses the first confirmed track and follows that ID permanently.
    It never selects tracks by truth proximity. Additional IDs and deleted primary
    tracks remain visible in metrics. Scenario truth is interpolation-derived.

    References
    ----------
    .. [1] Tracker 001 measurement/truth separation and reproducible export contract.
    """

    def __init__(self, out_dir: Path, tracker: ScenarioTracker) -> None:
        self.out_dir, self.tracker = out_dir, tracker
        self._stack = ExitStack()
        self._detections: Any = None
        self._tracks: Any = None
        self._frames = 0
        self._missing = 0
        self._confirmed = 0
        self._sum_range_squared = 0.0
        self._sum_velocity_squared = 0.0
        self._first_time_s: float | None = None
        self._latency_s: float | None = None
        self._primary: str | None = None
        self._highest_id = 0
        self._statuses: Counter[str] = Counter()

    def __enter__(self) -> TrackingWriter:
        """Open output streams; see TrackingWriter References."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            detection_file: TextIO = self._stack.enter_context(
                (self.out_dir / "detections.csv").open("w", newline="", encoding="utf-8")
            )
            track_file: TextIO = self._stack.enter_context(
                (self.out_dir / "tracks.csv").open("w", newline="", encoding="utf-8")
            )
            self._detections = csv.writer(detection_file)
            self._tracks = csv.writer(track_file)
            self._detections.writerow(
                [
                    "time_s",
                    "leg_index",
                    "doppler_bin",
                    "range_bin",
                    "range_m",
                    "radial_velocity_mps",
                    "covariance_json",
                    "noise_power_linear",
                    "status",
                    "pair_id",
                    "unfolded_velocity_mps",
                    "unfolding_residual_mps",
                    "range_interpretation",
                    "velocity_interpretation",
                ]
            )
            self._tracks.writerow(
                [
                    "time_s",
                    "track_id",
                    "status",
                    "range_m",
                    "radial_velocity_mps",
                    "range_interpretation",
                    "coordinates_json",
                    "state_json",
                    "covariance_json",
                ]
            )
            return self
        except BaseException:
            self._stack.close()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close streams and write metrics on successful completion."""
        self._stack.close()
        if exc_type is None:
            (self.out_dir / "tracking_metrics.json").write_text(
                json.dumps(self.metrics(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )

    def write(
        self, result: TrackingFrame, *, truth_range_m: float, truth_velocity_mps: float
    ) -> None:
        """Export measured/estimated values, then evaluate against separate truth.

        Parameters
        ----------
        result : TrackingFrame
            Already-completed measurement-only processing output.
        truth_range_m, truth_velocity_mps : float
            Evaluation labels; never supplied to the estimator.

        References
        ----------
        .. [1] TrackingWriter class references.
        """
        self._frames += 1
        self._missing += int(result.n_observations == 0)
        if self._first_time_s is None:
            self._first_time_s = result.timestamp_s
        interpretation = (
            "modulo_absolute_unresolved" if result.range_period_m is not None else "absolute"
        )
        # Serialization is sequential to bound memory across long scenarios.
        for d in result.detections:
            self._statuses[d.status] += 1
            self._detections.writerow(
                [
                    d.timestamp_s,
                    d.leg_index,
                    d.doppler_bin,
                    d.range_bin,
                    d.range_m,
                    d.radial_velocity_mps,
                    json.dumps(d.covariance.tolist()),
                    d.noise_power_linear,
                    d.status,
                    d.pair_id,
                    d.unfolded_velocity_mps,
                    d.unfolding_residual_mps,
                    interpretation,
                    "unambiguous" if self.tracker.variant == "S2" else "folded",
                ]
            )
        primary_snapshot: TrackSnapshot | None = None
        for snapshot in result.snapshots:
            self._highest_id = max(self._highest_id, int(snapshot.track_id))
            range_m = float(snapshot.state[0])
            if result.range_period_m is not None:
                range_m %= result.range_period_m
            self._tracks.writerow(
                [
                    snapshot.timestamp_s,
                    snapshot.track_id,
                    snapshot.status.value,
                    range_m,
                    float(snapshot.state[1]),
                    interpretation,
                    json.dumps([asdict(c) for c in snapshot.estimate.state_space.coordinates]),
                    json.dumps(snapshot.state.tolist()),
                    json.dumps(snapshot.covariance.tolist()),
                ]
            )
            if self._primary is None and snapshot.status == TrackStatus.CONFIRMED:
                self._primary = snapshot.track_id
                self._latency_s = result.timestamp_s - self._first_time_s
            if snapshot.track_id == self._primary and snapshot.status == TrackStatus.CONFIRMED:
                primary_snapshot = snapshot
        if primary_snapshot is not None:
            self._confirmed += 1
            error_m = float(primary_snapshot.state[0]) - truth_range_m
            if result.range_period_m is not None:
                period = result.range_period_m
                error_m = (error_m + period / 2) % period - period / 2
            self._sum_range_squared += error_m**2
            self._sum_velocity_squared += (
                float(primary_snapshot.state[1]) - truth_velocity_mps
            ) ** 2

    def metrics(self) -> dict[str, Any]:
        """Return bounded-memory metrics, explicit denominator and interpretation.

        References
        ----------
        .. [1] TrackingWriter class references.
        """
        return {
            "variant": self.tracker.variant,
            "n_frames": self._frames,
            "primary_track_id": self._primary,
            "n_tracks_created": self._highest_id,
            "confirmation_latency_s": self._latency_s,
            "confirmed_primary_frames": self._confirmed,
            "continuity_fraction": self._confirmed / self._frames if self._frames else None,
            "frames_without_observations": self._missing,
            "detection_status_counts": dict(self._statuses),
            "range_interpretation": "modulo_absolute_unresolved"
            if self.tracker.range_period_m
            else "absolute",
            "range_rmse_m": float(np.sqrt(self._sum_range_squared / self._confirmed))
            if self._confirmed
            else None,
            "radial_velocity_rmse_mps": float(np.sqrt(self._sum_velocity_squared / self._confirmed))
            if self._confirmed
            else None,
            "evaluation": (
                "First confirmed ID, confirmed frames only; interpolation-derived truth. "
                "No truth-based reassignment."
            ),
        }

    def metadata(self) -> dict[str, Any]:
        """Describe resolved settings and coordinate conventions for reproduction.

        References
        ----------
        .. [1] Tracker 001 export contract.
        """
        return {
            "variant": self.tracker.variant,
            "detection": asdict(self.tracker.detection),
            "tracking": asdict(self.tracker.config),
            "range_period_m": self.tracker.range_period_m,
            "state_space": asdict(self.tracker.motion.state_space),
            "velocity_sign": "positive_closing",
            "source_manifest": "spec/tracker-001-provenance.md",
        }


def snapshot_record(snapshot: TrackSnapshot) -> dict[str, Any]:
    """Serialize a generic ENU/custom snapshot with coordinate ordering and covariance.

    Parameters
    ----------
    snapshot : TrackSnapshot
        Estimate with mean (n,) and covariance (n,n).

    Returns
    -------
    dict
        JSON-compatible units, frame, origin, state, covariance and lifecycle.

    References
    ----------
    .. [1] Tracker 001 generic coordinate export contract.
    """
    return {
        "time_s": snapshot.timestamp_s,
        "track_id": snapshot.track_id,
        "status": snapshot.status.value,
        "state_space": asdict(snapshot.estimate.state_space),
        "mean": snapshot.state.tolist(),
        "covariance": snapshot.covariance.tolist(),
    }
