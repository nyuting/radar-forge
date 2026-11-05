"""Multi-target tracking in which every track is a row of one batched bank.

``models`` holds the state layouts and motion and measurement models; ``filters`` the bank and
the one Gaussian filter (KF, EKF, UKF) over it; ``association`` gating and assignment;
``initiation`` the initiators; ``tracker`` the scan loop and lifecycle; ``metrics`` NEES and RMSE.
"""

from radar_forge.tracking import association, filters, initiation, metrics, models, tracker
from radar_forge.tracking.tracker import Scan, Tracker

__all__ = [
    "Scan",
    "Tracker",
    "association",
    "filters",
    "initiation",
    "metrics",
    "models",
    "tracker",
]
