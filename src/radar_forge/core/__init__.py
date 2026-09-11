"""Core radar physics and signal processing primitives."""

from __future__ import annotations

from radar_forge.core import constants, dsp, windows
from radar_forge.core.constants import (
    BOLTZMANN_JPK,
    EARTH_RADIUS_M,
    FOUR_THIRDS_EARTH_RADIUS_M,
    SPEED_OF_LIGHT_MPS,
    STANDARD_NOISE_TEMPERATURE_K,
)
from radar_forge.core.dsp import (
    doppler_bin_centers_mps,
    doppler_fft,
    matched_filter,
    mti_filter,
    range_bin_centers_m,
    range_doppler_map,
    range_fft,
)
from radar_forge.core.radar_equation import received_power_w
from radar_forge.core.waveforms import (
    beat_frequency_hz,
    lfm_chirp,
    range_from_beat_frequency_m,
    range_resolution_m,
    sweep_rate_hzps,
)
from radar_forge.core.windows import (
    TAPER_NAMES,
    apply_taper,
    coherent_gain_linear,
    processing_loss_db,
    taper,
)

__all__ = [
    "BOLTZMANN_JPK",
    "EARTH_RADIUS_M",
    "FOUR_THIRDS_EARTH_RADIUS_M",
    "SPEED_OF_LIGHT_MPS",
    "STANDARD_NOISE_TEMPERATURE_K",
    "TAPER_NAMES",
    "apply_taper",
    "beat_frequency_hz",
    "coherent_gain_linear",
    "constants",
    "doppler_bin_centers_mps",
    "doppler_fft",
    "dsp",
    "lfm_chirp",
    "matched_filter",
    "mti_filter",
    "processing_loss_db",
    "range_bin_centers_m",
    "range_doppler_map",
    "range_fft",
    "range_from_beat_frequency_m",
    "range_resolution_m",
    "received_power_w",
    "sweep_rate_hzps",
    "taper",
    "windows",
]
