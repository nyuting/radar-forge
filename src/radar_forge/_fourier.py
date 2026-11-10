r"""Interpolation on Fourier coefficients: private, shared by the migration and chip modules.

References
----------
.. [1] E. Aboutanios and B. Mulgrew, "Iterative frequency estimation by interpolation on
       Fourier coefficients," *IEEE Trans. Signal Processing*, vol. 53, no. 4, pp. 1237-1242,
       2005.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["aboutanios_mulgrew", "signed_cycles"]


def aboutanios_mulgrew(
    samples: NDArray[np.complex128], peak_bin: float, n_iterations: int
) -> float:
    r"""Return a tone's frequency, DFT bins, refined from its peak bin by Aboutanios-Mulgrew.

    Each iteration adds :math:`\tfrac12\mathrm{Re}[(X_+ + X_-)/(X_+ - X_-)]`, :math:`X_\pm` the
    DTFT half a bin either side of the estimate [1]_; exact on a noiseless tone. Fewer than two
    samples, or equal coefficients, leave the estimate where it is.
    """
    n_samples = samples.size
    if n_samples < 2:
        return peak_bin
    index = np.arange(n_samples)
    estimate = peak_bin
    # Each iteration refines the last; two reach a noiseless tone to rounding.
    for _ in range(n_iterations):
        plus, minus = (
            samples @ np.exp(-2j * np.pi * index * (estimate + half) / n_samples)
            for half in (0.5, -0.5)
        )
        if plus == minus:
            break
        estimate += 0.5 * float(np.real((plus + minus) / (plus - minus)))
    return estimate


def signed_cycles(cycles: float) -> float:
    """Return a frequency in cycles per sample wrapped into ``[-0.5, 0.5)``."""
    return float((cycles + 0.5) % 1.0 - 0.5)
