"""Multi-target tracking in which every track is a row of one batched bank.

``motion`` holds the state layouts and motion models; ``measurement`` the measurement models;
``filters`` the bank and the one Gaussian filter (KF, EKF, UKF) over it; ``imm`` the
interacting multiple model; ``association`` gating, assignment, JPDA and Murty's k best;
``initiation`` the initiators; ``lifecycle`` M-of-N and existence; ``tracker`` the scan loop;
``hypotheses`` track-oriented MHT; ``pmbm`` the PMBM tracker; ``metrics`` NEES, RMSE and GOSPA;
``clutter_map`` the clutter density learnt from plots. A scenario's tracker is built from
its ``[tracking]`` table outside the package, by :mod:`radar_forge.scan_tracker`; the scene
and its plots are simulated by :mod:`radar_forge.trajectory` and
:mod:`radar_forge.plot_simulator`.
"""

from radar_forge.tracking import (
    association,
    clutter_map,
    filters,
    hypotheses,
    imm,
    initiation,
    lifecycle,
    measurement,
    metrics,
    motion,
    pmbm,
    tracker,
)
from radar_forge.tracking.tracker import Scan, Tracker

__all__ = [
    "Scan",
    "Tracker",
    "association",
    "clutter_map",
    "filters",
    "hypotheses",
    "imm",
    "initiation",
    "lifecycle",
    "measurement",
    "metrics",
    "motion",
    "pmbm",
    "tracker",
]
