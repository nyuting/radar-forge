"""Matrix helpers: private, shared by the models, filters, IMM, initiators and PMBM."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["symmetrise"]


def symmetrise(matrix: NDArray[Any]) -> NDArray[np.float64]:
    """Return the symmetric part of a stack of matrices, removing asymmetric roundoff."""
    result: NDArray[np.float64] = (matrix + np.swapaxes(matrix, -1, -2)) / 2
    return result
