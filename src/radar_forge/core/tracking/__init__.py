"""Tracking: filters, gating, association and track management.

``core/tracking.py`` grew into this package once it held more than one tracker's
worth of code. Every public name is re-exported here, so
``from radar_forge.core.tracking import TrackManager`` works whichever module
defines it.

Modules
-------
kalman
    The linear Kalman tracker that scenario 003 uses today.
coordinates
    Named coordinates, the state layout they form, and a state estimate in that layout.
motion
    Motion models that predict a state forward in time: Cartesian CV/CA and range-only.
measurement_models
    Measurements, the sensors they arrive from, and the models that predict them.
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

Both trackers, :class:`KalmanTracker` and :class:`Tracker`, keep their tracks
as :class:`Track` objects and confirm and delete them with :class:`TrackManager`.
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
    FrameResult,
    KalmanFilter,
    KalmanState,
    KalmanTracker,
    StateModel,
    TrackModel,
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
from radar_forge.core.tracking.lifecycle import LifecyclePolicy, TrackManager
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
from radar_forge.core.tracking.tracks import TRACK_STATUSES, Track, TrackSnapshot, TrackStatus
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
    "KalmanFilter",
    "KalmanState",
    "KalmanTracker",
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
