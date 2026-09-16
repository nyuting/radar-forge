"""End-to-end simulation pipelines: trajectories, scenarios, and frames."""

from __future__ import annotations

from radar_forge.pipelines import scenarios, tracking, trajectories
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
    "Measurement",
    "RangeDopplerProduct",
    "Scenario",
    "ScenarioTracker",
    "TargetTrack",
    "TrackingConfig",
    "Trajectory",
    "configs_from_scenario",
    "dual_prf_measurements",
    "form_range_doppler_map",
    "frame_detections",
    "iterate_frames",
    "load_flight_csv",
    "load_scenario",
    "peak_range_velocity",
    "resample",
    "scenarios",
    "to_bistatic_radar_frame",
    "to_radar_frame",
    "tracking",
    "trajectories",
    "unfold_velocity_mps",
]
