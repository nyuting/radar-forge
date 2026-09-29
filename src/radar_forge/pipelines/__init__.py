"""End-to-end simulation pipelines: trajectories, scenarios, and frames."""

from __future__ import annotations

from radar_forge.pipelines import (
    general_tracking,
    scenarios,
    tracking,
    tracking_config,
    tracking_output,
    trajectories,
)
from radar_forge.pipelines.general_tracking import (
    RadarDetection,
    TrackingFrame,
    detect_product,
)
from radar_forge.pipelines.general_tracking import ScenarioTracker as GeneralScenarioTracker
from radar_forge.pipelines.scenarios import (
    Frame,
    RangeDopplerProduct,
    Scenario,
    form_range_doppler_map,
    iterate_frames,
    load_scenario,
    peak_range_velocity,
)
from radar_forge.pipelines.tracking import (
    DetectionConfig,
    FrameTracks,
    Measurement,
    ScenarioTracker,
    TrackingConfig,
    configs_from_scenario,
    dual_prf_measurements,
    frame_detections,
    unfold_velocity_mps,
)
from radar_forge.pipelines.tracking_config import (
    ModelRegistry,
    build_enu_tracker,
    build_tracker,
    parse_tracking_table,
)
from radar_forge.pipelines.tracking_config import TrackingConfig as GeneralTrackingConfig
from radar_forge.pipelines.tracking_output import TrackingWriter, snapshot_record
from radar_forge.pipelines.trajectories import (
    BistaticTargetTrack,
    TargetTrack,
    Trajectory,
    load_flight_csv,
    resample,
    to_bistatic_radar_frame,
    to_radar_frame,
)

__all__ = [
    "BistaticTargetTrack",
    "DetectionConfig",
    "Frame",
    "FrameTracks",
    "GeneralScenarioTracker",
    "GeneralTrackingConfig",
    "Measurement",
    "ModelRegistry",
    "RadarDetection",
    "RangeDopplerProduct",
    "Scenario",
    "ScenarioTracker",
    "TargetTrack",
    "TrackingConfig",
    "TrackingFrame",
    "TrackingWriter",
    "Trajectory",
    "build_enu_tracker",
    "build_tracker",
    "configs_from_scenario",
    "detect_product",
    "dual_prf_measurements",
    "form_range_doppler_map",
    "frame_detections",
    "general_tracking",
    "iterate_frames",
    "load_flight_csv",
    "load_scenario",
    "parse_tracking_table",
    "peak_range_velocity",
    "resample",
    "scenarios",
    "snapshot_record",
    "to_bistatic_radar_frame",
    "to_radar_frame",
    "tracking",
    "tracking_config",
    "tracking_output",
    "trajectories",
    "unfold_velocity_mps",
]
