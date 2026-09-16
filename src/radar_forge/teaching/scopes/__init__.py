"""Radar scopes: displays that show what the receiver saw."""

from __future__ import annotations

from radar_forge.teaching.scopes import rd_map, track_plot
from radar_forge.teaching.scopes.rd_map import render_range_doppler
from radar_forge.teaching.scopes.track_plot import render_range_time_history

__all__ = ["rd_map", "render_range_doppler", "render_range_time_history", "track_plot"]
