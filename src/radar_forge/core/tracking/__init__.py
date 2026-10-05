"""Tracking: filters, gating, association and track management.

``spec/structure.md`` D2 grows ``core/tracking.py`` into this package. Its public names are
re-exported here, so ``from radar_forge.core.tracking import TrackManager`` keeps working.

Modules
-------
kalman
    The linear Kalman tracker that scenario 003 uses today.
coordinates
    Named coordinates, the state layout they form, and a state estimate in that layout.
motion
    Motion models that predict a state forward in time: Cartesian CV/CA and range-only.
measurement_models
    Measurements, the sensor routes they arrive on, and the models that predict them.
estimation
    The estimator interface and the innovation statistics used for gating.
ukf
    The unscented Kalman filter (UKF).
association
    The chi-square gate, and nearest-neighbour and global nearest-neighbour assignment.
initiation
    Rules that start a new track from a measurement no track took.
tracks
    A live track, its status, and the read-only snapshot handed to callers.
lifecycle
    The M-of-N rule that confirms and deletes tracks, and the manager that applies it.
tracker
    The tracker that runs one scan at a time, and two builders for common set-ups.

``Track``, ``TrackManager`` and ``TrackStatus`` from ``tracks`` and ``lifecycle`` share
their names with ``kalman``'s, so only ``kalman``'s are re-exported here. Import the
others from their modules, e.g. ``from radar_forge.core.tracking.tracks import Track``.
"""

from __future__ import annotations

from radar_forge.core.tracking import (
    association,
    coordinates,
    estimation,
    initiation,
    kalman,
    lifecycle,
    measurement_models,
    motion,
    tracker,
    tracks,
    ukf,
)
from radar_forge.core.tracking.association import (
    AssociationResult,
    Associator,
    ChiSquareGate,
    GlobalNearestNeighbour,
    NearestNeighbour,
)
from radar_forge.core.tracking.coordinates import Coordinate, StateEstimate, StateLayout
from radar_forge.core.tracking.estimation import Estimator, InnovationStats, innovation_stats
from radar_forge.core.tracking.initiation import DirectStateInitiator, TrackInitiator
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
from radar_forge.core.tracking.lifecycle import LifecyclePolicy
from radar_forge.core.tracking.measurement_models import (
    BistaticRangeDopplerModel,
    CartesianPosition,
    Measurement,
    MeasurementBatch,
    MeasurementModel,
    SensorPose,
    SensorRoute,
)
from radar_forge.core.tracking.motion import (
    CartesianMotion,
    MotionKind,
    MotionModel,
    RadialMotion,
)
from radar_forge.core.tracking.tracker import Tracker, build_tracker, build_tracker_enu
from radar_forge.core.tracking.tracks import TrackSnapshot
from radar_forge.core.tracking.ukf import UKF

__all__ = [
    "STATE_MODELS",
    "TRACK_STATUSES",
    "UKF",
    "AssociationResult",
    "Associator",
    "BistaticRangeDopplerModel",
    "CartesianMotion",
    "CartesianPosition",
    "ChiSquareGate",
    "Coordinate",
    "DirectStateInitiator",
    "Estimator",
    "FrameResult",
    "GlobalNearestNeighbour",
    "InnovationStats",
    "KalmanState",
    "LifecyclePolicy",
    "Measurement",
    "MeasurementBatch",
    "MeasurementModel",
    "MotionKind",
    "MotionModel",
    "NearestNeighbour",
    "RadialMotion",
    "SensorPose",
    "SensorRoute",
    "StateEstimate",
    "StateLayout",
    "StateModel",
    "Track",
    "TrackInitiator",
    "TrackManager",
    "TrackModel",
    "TrackSnapshot",
    "TrackStatus",
    "Tracker",
    "UpdateResult",
    "associate_gnn",
    "association",
    "build_tracker",
    "build_tracker_enu",
    "coordinates",
    "estimation",
    "gate_threshold",
    "initiation",
    "innovation_of",
    "innovation_stats",
    "kalman",
    "lifecycle",
    "measurement_models",
    "motion",
    "normalised_innovation_squared",
    "predict",
    "process_noise_dwna",
    "state_model_matrices",
    "tracker",
    "tracks",
    "ukf",
    "update",
]
