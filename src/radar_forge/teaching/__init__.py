"""Teaching aids: scopes and plots for reading what the simulator produced.

Everything here needs ``matplotlib``, which lives in the ``teaching`` extra.
This package is therefore **not** imported by :mod:`radar_forge` at top level:
``import radar_forge`` must work with no extras installed. Import it
explicitly when you want it.
"""

from __future__ import annotations

from radar_forge.teaching import plotting, scopes
from radar_forge.teaching.plotting import magnitude_db, require_pyplot, save_figure

__all__ = ["magnitude_db", "plotting", "require_pyplot", "save_figure", "scopes"]
