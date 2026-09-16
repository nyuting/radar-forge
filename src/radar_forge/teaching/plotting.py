"""Plotting helpers shared by the teaching scopes.

Everything in :mod:`radar_forge.teaching` depends on ``matplotlib``, which is
an optional extra rather than a core dependency: ``import radar_forge`` must
succeed in an environment that has only ``numpy`` and ``scipy``. So the import
happens inside :func:`require_pyplot` at call time, and the error names the
extra to install rather than leaving the reader to guess.

The decibel conversion lives here rather than in :mod:`radar_forge.core.dsp`
because it is a *display* concern. A floor is not a physical quantity; it is a
choice about how much of the noise to show, and putting it next to the signal
processing would invite someone to apply it before a detector, where the
logarithm would distort the statistics the detector assumes.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, S1.4 (decibel conventions).
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["magnitude_db", "require_pyplot", "save_figure"]

_TEACHING_EXTRA_HINT = (
    "matplotlib is required for radar_forge.teaching but is not installed. "
    "It ships in the 'teaching' extra: install with `uv sync --extra teaching`, "
    "or `pip install 'radar-forge[teaching]'`."
)


def require_pyplot() -> ModuleType:
    """Import :mod:`matplotlib.pyplot`, or explain which extra supplies it.

    Returns
    -------
    module
        The ``matplotlib.pyplot`` module.

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed. The message names the ``teaching``
        extra and gives the command to install it.

    Notes
    -----
    Called at the top of every function that draws, rather than at module
    import, so that importing :mod:`radar_forge` costs nothing in an
    environment without the extra.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(_TEACHING_EXTRA_HINT) from error
    pyplot: ModuleType = plt
    return pyplot


def magnitude_db(
    values: ArrayLike,
    *,
    reference_linear: float | None = None,
    floor_db: float = -120.0,
) -> NDArray[np.float64]:
    r"""Convert complex amplitudes to a decibel magnitude for display.

    .. math:: X_{\mathrm{dB}} = 20 \log_{10}\frac{|x|}{x_{\mathrm{ref}}}

    Parameters
    ----------
    values : array_like
        Complex or real amplitudes, any shape.
    reference_linear : float, optional
        Amplitude to call 0 dB. Defaults to the largest magnitude present, so
        the result is dB relative to the peak -- the usual range-Doppler
        display, and the one that makes two frames with different received
        power comparable. Pass a fixed value to hold the scale across frames
        instead. Must be strictly positive.
    floor_db : float, optional
        Values below this are clamped to it, default -120 dB. Without a floor
        an empty cell takes :math:`\log_{10} 0` and the colour scale collapses
        onto a single point at negative infinity.

    Returns
    -------
    numpy.ndarray
        Decibel magnitudes, same shape as ``values``.

    Raises
    ------
    ValueError
        If ``reference_linear`` is not strictly positive.

    Notes
    -----
    This is an **amplitude** ratio, so the factor is 20, not 10. Passing a
    power array here would halve every number on the scale.

    Examples
    --------
    >>> import numpy as np
    >>> float(magnitude_db(np.array([1.0, 0.5]))[1])
    -6.020599913279624
    """
    magnitudes = np.abs(np.asarray(values))
    if reference_linear is None:
        peak_linear = float(magnitudes.max()) if magnitudes.size else 0.0
        reference_linear = peak_linear if peak_linear > 0.0 else 1.0
    if reference_linear <= 0.0:
        msg = f"reference_linear must be strictly positive; got {reference_linear!r}."
        raise ValueError(msg)

    with np.errstate(divide="ignore"):
        decibels = 20.0 * np.log10(magnitudes / reference_linear)
    result: NDArray[np.float64] = np.maximum(decibels, floor_db)
    return result


def save_figure(figure: Any, path: Path | str, *, dpi: int = 120) -> None:
    """Write a figure to disk and close it.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to write.
    path : pathlib.Path or str
        Destination; the extension picks the format.
    dpi : int, optional
        Dots per inch, default 120.

    Notes
    -----
    Closing is the point. A scenario run writes one figure per frame, and
    matplotlib keeps every unclosed figure alive, so a 16,500-frame run that
    forgets this exhausts memory long before it finishes.
    """
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt = require_pyplot()
    plt.close(figure)
