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

__all__ = [
    "BOLTZMANN_JPK",
    "EARTH_RADIUS_M",
    "FOUR_THIRDS_EARTH_RADIUS_M",
    "SPEED_OF_LIGHT_MPS",
    "STANDARD_NOISE_TEMPERATURE_K",
    "constants",
    "received_power_w",
]
