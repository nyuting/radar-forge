"""Optional range/velocity history plots from streaming tracker CSV outputs.

References
----------
.. [1] spec/tracker-001-integration.md, ambiguity-aware presentation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

__all__ = [
    "render_track_history",
]


def render_track_history(path: Path) -> Any:
    """Plot the first track's range and closing velocity from tracks.csv.

    Parameters
    ----------
    path : Path
        Tracking CSV export; no estimator is accessed.

    Returns
    -------
    matplotlib.figure.Figure
        Two labelled panels. Plotting loads one track's history into memory.

    References
    ----------
    .. [1] Tracker 001 export/presentation separation.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Tracking plots require the teaching extra: uv sync --extra teaching"
        ) from exc
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first = next(reader, None)
        rows = (
            []
            if first is None
            else [first, *(r for r in reader if r["track_id"] == first["track_id"])]
        )
    figure, axes = plt.subplots(2, 1, sharex=True)
    times = [float(r["time_s"]) for r in rows]
    axes[0].plot(times, [float(r["range_m"]) for r in rows])
    axes[1].plot(times, [float(r["radial_velocity_mps"]) for r in rows])
    ambiguous = rows and rows[0]["range_interpretation"] == "modulo_absolute_unresolved"
    axes[0].set_ylabel("Modulo range (m)" if ambiguous else "Range (m)")
    axes[0].set_title("Absolute range unresolved" if ambiguous else "Estimated track history")
    axes[1].set_ylabel("Closing velocity (m/s)")
    axes[1].set_xlabel("Time (s)")
    figure.tight_layout()
    return figure
