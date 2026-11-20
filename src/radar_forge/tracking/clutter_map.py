r"""The clutter map: the false-plot density per (m rad) per scan, estimated from the plots.

A rotating radar's tracker reads it in place of the scene's own density
(``[tracking] clutter_density_source = "map"``).

References
----------
.. [1] M. I. Skolnik (ed.), *Radar Handbook*, 3rd ed., McGraw-Hill, 2008.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["ClutterMap"]


class ClutterMap:
    """The false-plot density per (m rad) per scan, estimated per cell by forgetting average.

    Each scan adds its plots' counts: with f the forgetting factor, the cell's count sum and
    its scan sum both decay by f per scan, and their ratio over the cell's volume is the
    estimate, unbiased from the first scan. A cell never below ``(1 - f)`` of one plot over
    its volume, the least a forgetting average resolves, keeps every likelihood ratio finite.
    ``prior_per_m_rad`` counts as one scan already seen. The caller leaves out plots its
    confirmed tracks took. Azimuth cells wrap.

    References
    ----------
    Skolnik [1], ch. 2 (clutter maps in surveillance radars).
    """

    def __init__(
        self,
        cell_range_m: float = 1e3,
        cell_azimuth_rad: float = math.radians(5.0),
        forgetting: float = 0.9,
        *,
        range_max_m: float = 40e3,
        prior_per_m_rad: float = 0.0,
    ) -> None:
        if not (cell_range_m > 0 and 0 < cell_azimuth_rad <= 2 * np.pi and 0 <= forgetting < 1):
            msg = "need cells > 0 and 0 <= forgetting < 1."
            raise ValueError(msg)
        self.cell_range_m, self.cell_azimuth_rad, self.forgetting = (
            cell_range_m,
            cell_azimuth_rad,
            forgetting,
        )
        self.shape = (
            math.ceil(range_max_m / cell_range_m),
            math.ceil(2 * np.pi / cell_azimuth_rad),
        )
        self.counts = np.full(self.shape, prior_per_m_rad * cell_range_m * cell_azimuth_rad)
        self.n_scans = 1.0 if prior_per_m_rad > 0 else 0.0
        self.floor_per_m_rad = (1 - forgetting) / (cell_range_m * cell_azimuth_rad)

    def _cells(self, range_m: ArrayLike, azimuth_rad: ArrayLike) -> tuple[NDArray[np.intp], ...]:
        """Return each point's (range, azimuth) cell, clipped to the map."""
        rows = np.clip(
            (np.asarray(range_m) // self.cell_range_m).astype(np.intp), 0, self.shape[0] - 1
        )
        # Unguarded: a tiny negative angle rounds up to exactly 2 pi, the last column, where
        # _periodic.fold gives 0, the first.
        columns = (np.mod(azimuth_rad, 2 * np.pi) // self.cell_azimuth_rad).astype(np.intp)
        return rows, np.minimum(columns, self.shape[1] - 1)

    def update(self, range_m: ArrayLike, azimuth_rad: ArrayLike) -> None:
        """Fold one scan's false plots into the estimate."""
        counts = np.zeros(self.shape)
        np.add.at(counts, self._cells(range_m, azimuth_rad), 1.0)
        self.counts = self.forgetting * self.counts + counts
        self.n_scans = self.forgetting * self.n_scans + 1.0

    def density(self, range_m: ArrayLike, azimuth_rad: ArrayLike) -> NDArray[np.float64]:
        """Return the estimated density at each point, plots per (m rad) per scan."""
        volume = self.cell_range_m * self.cell_azimuth_rad
        estimate = self.counts[self._cells(range_m, azimuth_rad)] / max(self.n_scans, 1.0) / volume
        result: NDArray[np.float64] = np.maximum(estimate, self.floor_per_m_rad)
        return result
