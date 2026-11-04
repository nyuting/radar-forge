"""Multi-target tracking with every track a row of one batched bank.

``models`` holds the state layouts and motion models; ``measurement_models`` the measurement
models; ``filters`` the bank and the one Gaussian filter (KF, EKF, UKF) over it;
``association`` gating and assignment; ``initiation`` track birth; ``tracker`` the scan
loop and its lifecycle; ``metrics`` NEES and RMSE. ``spec/tracker-001.md`` is the design.
"""

from radar_forge.tracking import (
    association,
    filters,
    initiation,
    measurement_models,
    metrics,
    models,
    tracker,
)
from radar_forge.tracking.tracker import Scan, Tracker

__all__ = [
    "Scan",
    "Tracker",
    "association",
    "filters",
    "initiation",
    "measurement_models",
    "metrics",
    "models",
    "tracker",
]
