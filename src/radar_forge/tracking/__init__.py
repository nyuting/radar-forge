"""Multi-target tracking in which every track is a row of one batched bank.

``motion`` holds the state layouts and motion models; ``measurement`` the measurement models;
``filters`` the bank and the one Gaussian filter (KF, EKF, UKF) over it; ``imm`` the
interacting multiple model; ``association`` gating, assignment, JPDA and Murty's k best;
``initiation`` the initiators; ``lifecycle`` M-of-N and existence; ``tracker`` the scan loop;
``hypotheses`` track-oriented MHT; ``pmbm`` the PMBM tracker; ``metrics`` NEES, RMSE and GOSPA;
``scene`` a plot-level scene and the plots a surveillance radar makes of it; ``scans`` that
scene played scan by scan into a tracker.
"""

from radar_forge.tracking import (
    association,
    filters,
    hypotheses,
    imm,
    initiation,
    lifecycle,
    measurement,
    metrics,
    motion,
    pmbm,
    scans,
    scene,
    tracker,
)
from radar_forge.tracking.tracker import Scan, Tracker

__all__ = [
    "Scan",
    "Tracker",
    "association",
    "filters",
    "hypotheses",
    "imm",
    "initiation",
    "lifecycle",
    "measurement",
    "metrics",
    "motion",
    "pmbm",
    "scans",
    "scene",
    "tracker",
]
