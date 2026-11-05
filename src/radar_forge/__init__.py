"""radar-forge: a radar workbench for research and education, from link budget to tracks.

The public API is what this module re-exports; a name reached through its module path is
there too, but may move. Modules, in signal-path order:

``constants``
    Every physical constant, defined once.
``radar``, ``propagation``, ``signal``
    The radar and its sites; propagation paths and the radar equation; waveforms and IQ.
``dsp``, ``detection``, ``tracking``
    Range-Doppler processing; CFAR and clustering; batched multi-target tracking.
``scenario``, ``pipeline``, ``evaluate``, ``plots``
    A scenario file, the chain from it to tracks, the stepper that checks each stage, and
    figures (the ``visualisation`` extra, imported only when drawing).
"""

from __future__ import annotations

from radar_forge.detection import Detection, cfar_detect, cfar_detect_2d, cluster_detections
from radar_forge.dsp import (
    doppler_bin_centers_mps,
    doppler_fft,
    matched_filter,
    range_bin_centers_m,
    range_doppler_map,
    range_fft,
    window,
)
from radar_forge.pipeline import frames, range_doppler_product
from radar_forge.propagation import (
    doppler_shift_hz,
    line_of_sight_paths,
    received_power_w,
    swerling_rcs_m2,
)
from radar_forge.radar import Radar, Receiver, Site, TargetGeometry, Transmitter
from radar_forge.scenario import Scenario, load_scenario
from radar_forge.signal import (
    ambiguity_function,
    fmcw_deramp_baseband,
    lfm_chirp,
    pulsed_baseband,
    thermal_noise,
)
from radar_forge.tracking import Scan, Tracker

__version__ = "0.0.0"

__all__ = [
    "Detection",
    "Radar",
    "Receiver",
    "Scan",
    "Scenario",
    "Site",
    "TargetGeometry",
    "Tracker",
    "Transmitter",
    "__version__",
    "ambiguity_function",
    "cfar_detect",
    "cfar_detect_2d",
    "cluster_detections",
    "doppler_bin_centers_mps",
    "doppler_fft",
    "doppler_shift_hz",
    "fmcw_deramp_baseband",
    "frames",
    "lfm_chirp",
    "line_of_sight_paths",
    "load_scenario",
    "matched_filter",
    "pulsed_baseband",
    "range_bin_centers_m",
    "range_doppler_map",
    "range_doppler_product",
    "range_fft",
    "received_power_w",
    "swerling_rcs_m2",
    "thermal_noise",
    "window",
]
