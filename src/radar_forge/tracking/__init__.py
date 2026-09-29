"""Configurable Bayesian tracking with SI coordinates and explicit sensor mappings.

References
----------
.. [1] spec/tracker-001-integration.md and spec/tracker-001-provenance.md.
"""

from __future__ import annotations

from radar_forge.tracking.association import (
    AssociationResult,
    Associator,
    ChiSquareGate,
    GlobalNearestNeighbour,
    NearestNeighbour,
)
from radar_forge.tracking.derived import DerivedValue, derive_kinematics
from radar_forge.tracking.engine import TrackerEngine
from radar_forge.tracking.estimation import Estimator, InnovationStats
from radar_forge.tracking.imm import IMM
from radar_forge.tracking.initiation import (
    DirectStateInitiator,
    NoInitiation,
    RangeBearingInitiator,
    RoutedInitiator,
    TrackInitiator,
)
from radar_forge.tracking.management import LifecyclePolicy, TrackManager
from radar_forge.tracking.measurements import (
    BistaticRangeDoppler,
    CartesianPosition,
    CompositeMeasurementModel,
    MeasurementModel,
    MonostaticRadar,
    SensorPose,
)
from radar_forge.tracking.motion import (
    CartesianMotion,
    CoordinatedTurn,
    MotionModel,
    RadialMotion,
)
from radar_forge.tracking.sensors import Measurement, MeasurementBatch, Sensor
from radar_forge.tracking.spaces import Coordinate, MeasurementSpace, StateEstimate, StateSpace
from radar_forge.tracking.tracks import Track, TrackSnapshot, TrackStatus
from radar_forge.tracking.ukf import UKF

__all__ = [
    "IMM",
    "UKF",
    "AssociationResult",
    "Associator",
    "BistaticRangeDoppler",
    "CartesianMotion",
    "CartesianPosition",
    "ChiSquareGate",
    "CompositeMeasurementModel",
    "Coordinate",
    "CoordinatedTurn",
    "DerivedValue",
    "DirectStateInitiator",
    "Estimator",
    "GlobalNearestNeighbour",
    "InnovationStats",
    "LifecyclePolicy",
    "Measurement",
    "MeasurementBatch",
    "MeasurementModel",
    "MeasurementSpace",
    "MonostaticRadar",
    "MotionModel",
    "NearestNeighbour",
    "NoInitiation",
    "RadialMotion",
    "RangeBearingInitiator",
    "RoutedInitiator",
    "Sensor",
    "SensorPose",
    "StateEstimate",
    "StateSpace",
    "Track",
    "TrackInitiator",
    "TrackManager",
    "TrackSnapshot",
    "TrackStatus",
    "TrackerEngine",
    "derive_kinematics",
]
