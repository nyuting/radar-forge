"""Linear-power 2D CA-CFAR and deterministic peak suppression.

References
----------
.. [1] M. Richards, Fundamentals of Radar Signal Processing, 2nd ed., 2014, ch. 6.
.. [2] Norfair detector/tracker separation, https://github.com/tryolabs/norfair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import maximum_filter, uniform_filter

__all__ = [
    "CfarResult",
    "DetectionConfig",
    "cfar_2d",
]


@dataclass(frozen=True)
class DetectionConfig:
    """CA-CFAR windows and physical uncertainty floors.

    Parameters
    ----------
    guard_cells, training_cells : int
        Per-side counts on each axis; the training ring excludes the guard box.
    pfa : float
        Nominal per-cell false alarm probability under independent exponential noise.
    range_std_floor_m, velocity_std_floor_mps : float
        Nonnegative floors for the bin-quantization standard deviations.

    References
    ----------
    .. [1] Richards, 2014, CA-CFAR threshold multiplier.
    """

    guard_cells: int = 2
    training_cells: int = 8
    pfa: float = 1e-6
    range_std_floor_m: float = 0.0
    velocity_std_floor_mps: float = 0.0

    def __post_init__(self) -> None:
        """Validate counts, false alarm probability and physical uncertainty floors."""
        if (
            type(self.guard_cells) is not int
            or type(self.training_cells) is not int
            or self.guard_cells < 0
            or self.training_cells < 1
        ):
            raise ValueError("guard_cells >= 0 and training_cells >= 1 must be integers")
        if not 0 < self.pfa < 1:
            raise ValueError("pfa must lie strictly between zero and one")
        if (
            not np.all(np.isfinite([self.range_std_floor_m, self.velocity_std_floor_mps]))
            or min(self.range_std_floor_m, self.velocity_std_floor_mps) < 0
        ):
            raise ValueError("uncertainty floors must be finite and nonnegative")


@dataclass(frozen=True)
class CfarResult:
    """Detection mask, noise and threshold (n_doppler,n_range), plus peak indices (k,2).

    Notes
    -----
    Noise/threshold outside complete nonperiodic windows are NaN. Power units
    are those of the input map, not necessarily calibrated receiver watts.

    References
    ----------
    .. [1] Richards, 2014, CA-CFAR; Tracker 001 deterministic peak contract.
    """

    mask: NDArray[np.bool_]
    noise_power_linear: NDArray[np.float64]
    threshold_power_linear: NDArray[np.float64]
    peak_indices: NDArray[np.int64]


def cfar_2d(
    power_linear: ArrayLike, config: DetectionConfig | None = None, *, wrap_range: bool = False
) -> CfarResult:
    """Detect isolated peaks in linear power with Doppler on axis 0 and range on axis 1.

    Parameters
    ----------
    power_linear : array_like
        Finite nonnegative power, shape (n_doppler,n_range).
    config : DetectionConfig or None
        Window geometry and nominal false alarm rate.
    wrap_range : bool
        True for folded pulsed range; Doppler always wraps.

    Returns
    -------
    CfarResult
        Threshold exceedances and deterministic locally suppressed peak indices.

    Notes
    -----
    Alpha = N * (Pfa**(-1/N) - 1) assumes independent exponential training samples
    and a single exponential cell under test. Pulse compression correlates bins;
    its empirical Pfa need not equal this nominal value. No truth or strongest-cell
    fallback is used. Ties use row-major order. Peak suppression spans the guard
    radius (at least one cell), using the same periodic boundaries as detection.

    References
    ----------
    .. [1] Richards, 2014, CA-CFAR for square-law detection.
    """
    cfg = config or DetectionConfig()
    power = np.asarray(power_linear, dtype=np.float64)
    radius = cfg.guard_cells + cfg.training_cells
    width, guard_width = 2 * radius + 1, 2 * cfg.guard_cells + 1
    if (
        power.ndim != 2
        or min(power.shape) < width
        or not np.all(np.isfinite(power))
        or np.any(power < 0)
    ):
        raise ValueError(
            "power_linear must be finite, nonnegative, 2D and larger than the CFAR window"
        )
    mode = ("wrap", "wrap" if wrap_range else "constant")
    n_training = width**2 - guard_width**2
    total = uniform_filter(power, size=width, mode=mode) * width**2
    guard = uniform_filter(power, size=guard_width, mode=mode) * guard_width**2
    # Window subtraction can leave tiny negative roundoff on exactly zero noise.
    noise = np.maximum((total - guard) / n_training, 0)
    alpha = n_training * np.expm1(-np.log(cfg.pfa) / n_training)
    threshold = alpha * noise
    if not wrap_range:
        threshold[:, :radius] = np.nan
        threshold[:, -radius:] = np.nan
        noise[:, :radius] = np.nan
        noise[:, -radius:] = np.nan
    mask = power > threshold
    suppression = max(1, cfg.guard_cells)
    local_max = maximum_filter(power, size=2 * suppression + 1, mode=mode)
    candidates = np.argwhere(mask & (power == local_max))
    strength = power[candidates[:, 0], candidates[:, 1]]
    candidates = candidates[np.argsort(-strength, kind="stable")]
    peaks: list[tuple[int, int]] = []
    blocked = np.zeros(power.shape, dtype=np.bool_)
    # Greedy suppression is sequential: a selected peak suppresses later tied neighbours.
    for row, col in candidates:
        if blocked[row, col]:
            continue
        peaks.append((int(row), int(col)))
        rows = (np.arange(row - suppression, row + suppression + 1) % power.shape[0]).astype(int)
        cols = np.arange(col - suppression, col + suppression + 1)
        cols = cols % power.shape[1] if wrap_range else cols[(cols >= 0) & (cols < power.shape[1])]
        blocked[np.ix_(rows, cols)] = True
    indices = np.asarray(peaks, dtype=np.int64).reshape(-1, 2)
    return CfarResult(
        mask, np.asarray(noise, dtype=np.float64), np.asarray(threshold, dtype=np.float64), indices
    )
