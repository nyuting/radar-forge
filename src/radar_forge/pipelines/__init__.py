"""End-to-end simulation pipelines: trajectories, scenarios, and frames."""

from __future__ import annotations

from radar_forge.pipelines import scenarios, trajectories
from radar_forge.pipelines.scenarios import (
    Frame,
    RangeDopplerProduct,
    Scenario,
    burst_range_doppler,
    iterate_frames,
    load_scenario,
    peak_range_velocity,
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
    "Frame",
    "RangeDopplerProduct",
    "Scenario",
    "TargetTrack",
    "Trajectory",
    "burst_range_doppler",
    "iterate_frames",
    "load_flight_csv",
    "load_scenario",
    "peak_range_velocity",
    "resample",
    "scenarios",
    "to_bistatic_radar_frame",
    "to_radar_frame",
    "trajectories",
]
