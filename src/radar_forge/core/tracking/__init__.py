"""Tracking: filters, gating, association and track management.

``spec/structure.md`` D2 grows ``core/tracking.py`` into this package. Its public names are
re-exported here, so ``from radar_forge.core.tracking import TrackManager`` keeps working.

Modules
-------
kalman
    The linear Kalman tracker that scenario 003 uses: a constant-velocity filter, a
    chi-squared gate, global nearest-neighbour assignment and an M-of-N track manager.
"""

from __future__ import annotations

from radar_forge.core.tracking import kalman
from radar_forge.core.tracking.kalman import (
    STATE_MODELS,
    TRACK_STATUSES,
    FrameResult,
    KalmanState,
    StateModel,
    Track,
    TrackManager,
    TrackModel,
    TrackStatus,
    UpdateResult,
    associate_gnn,
    gate_threshold,
    innovation_of,
    normalised_innovation_squared,
    predict,
    process_noise_dwna,
    state_model_matrices,
    update,
)

__all__ = [
    "STATE_MODELS",
    "TRACK_STATUSES",
    "FrameResult",
    "KalmanState",
    "StateModel",
    "Track",
    "TrackManager",
    "TrackModel",
    "TrackStatus",
    "UpdateResult",
    "associate_gnn",
    "gate_threshold",
    "innovation_of",
    "kalman",
    "normalised_innovation_squared",
    "predict",
    "process_noise_dwna",
    "state_model_matrices",
    "update",
]
