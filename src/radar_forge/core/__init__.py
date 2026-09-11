"""Core radar physics and signal processing primitives."""

from __future__ import annotations

from radar_forge.core import constants
from radar_forge.core.constants import (
    BOLTZMANN_JPK,
    EARTH_RADIUS_M,
    FOUR_THIRDS_EARTH_RADIUS_M,
    SPEED_OF_LIGHT_MPS,
    STANDARD_NOISE_TEMPERATURE_K,
)
from radar_forge.core.radar_equation import received_power_w
from radar_forge.core.waveforms import (
    beat_frequency_hz,
    lfm_chirp,
    range_from_beat_frequency_m,
    range_resolution_m,
    sweep_rate_hzps,
)

__all__ = [
    "BOLTZMANN_JPK",
    "EARTH_RADIUS_M",
    "FOUR_THIRDS_EARTH_RADIUS_M",
    "SPEED_OF_LIGHT_MPS",
    "STANDARD_NOISE_TEMPERATURE_K",
    "beat_frequency_hz",
    "constants",
    "lfm_chirp",
    "range_from_beat_frequency_m",
    "range_resolution_m",
    "received_power_w",
    "sweep_rate_hzps",
]
