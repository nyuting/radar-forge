"""Amplitude tapers (windows) and the cost of applying them.

A finite record of samples is an infinite signal multiplied by a rectangle, and
the rectangle's transform is a sinc whose first sidelobe sits only 13 dB below
its peak. A strong target therefore smears energy across the whole range or
Doppler axis and buries weak ones. Tapering trades mainlobe width and signal-to-
noise ratio for sidelobe suppression, and this module is where that trade is made
explicit: :func:`taper` generates the weights, :func:`apply_taper` puts them on a
chosen axis of a data cube, and :func:`coherent_gain_linear` and
:func:`processing_loss_db` quantify what the taper cost.

The window shapes themselves come from :mod:`scipy.signal.windows` rather than
being reimplemented here; what this module adds is the radar-facing naming, the
normalisation that keeps a tapered peak readable in the original units, and the
documentation of the trade.

References
----------
.. [1] F. J. Harris, "On the use of windows for harmonic analysis with the
       discrete Fourier transform," *Proc. IEEE*, vol. 66, no. 1, pp. 51-83,
       Jan. 1978.
.. [2] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §14.4 (windowing in Doppler processing).
.. [3] T. T. Taylor, "Design of line-source antennas for narrow beamwidth and
       low side lobes," *IRE Trans. Antennas Propag.*, vol. 3, no. 1, 1955.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import windows as _scipy_windows

__all__ = [
    "TAPER_NAMES",
    "apply_taper",
    "coherent_gain_linear",
    "processing_loss_db",
    "taper",
]

TAPER_NAMES: tuple[str, ...] = (
    "rectangular",
    "hann",
    "hamming",
    "blackman",
    "blackmanharris",
    "taylor",
    "chebyshev",
)
"""The taper names :func:`taper` accepts, in order of increasing sidelobe suppression.

``taylor`` and ``chebyshev`` are parametric: their sidelobe level is set by the
``sidelobe_db`` argument rather than fixed by the window shape.
"""

# Chebyshev designs below about 45 dB degenerate — the "window" grows end spikes
# taller than its centre — and SciPy warns rather than refusing. Reject earlier.
_MIN_CHEBYSHEV_SIDELOBE_DB = 45.0


def taper(
    name: str,
    n_samples: int,
    *,
    normalize: bool = True,
    sidelobe_db: float = 60.0,
) -> NDArray[np.float64]:
    r"""Return a real amplitude taper of ``n_samples`` weights.

    Tapers are symmetric — ``w[i] == w[-1 - i]`` exactly — which is the right
    choice for weighting a finite aperture or a finite pulse train. (The
    *periodic* variant, one sample shorter and wrapped, is for spectral density
    estimation and is not what radar processing wants.)

    Parameters
    ----------
    name : str
        One of :data:`TAPER_NAMES`.
    n_samples : int
        Number of weights :math:`N`. Must be at least one.
    normalize : bool, optional
        If True (default), scale the weights so their mean is 1.0. A tone that
        peaked at amplitude :math:`A` in an untapered FFT then still peaks at
        :math:`A` after tapering, so a tapered range profile reads in the same
        units as an untapered one. Set False for the raw window shape.
    sidelobe_db : float, optional
        Design sidelobe level for the parametric windows ``"taylor"`` and
        ``"chebyshev"``, in decibels **below** the mainlobe peak, so 60.0 means
        -60 dB. Default 60.0. Ignored by the fixed-shape windows. Must be
        positive, and at least 45.0 for ``"chebyshev"``.

    Returns
    -------
    numpy.ndarray
        Real weights of shape ``(n_samples,)``.

    Raises
    ------
    ValueError
        If ``name`` is not a recognised taper, if ``n_samples`` is less than
        one, or if ``sidelobe_db`` is invalid for the requested window.

    Notes
    -----
    Approximate costs, for the fixed-shape windows [1]_. "Broadening" is the
    -3 dB mainlobe width relative to the rectangular case, and "SNR loss" is
    :func:`processing_loss_db`:

    ========================  ===============  ===========  =========
    Taper                     First sidelobe   Broadening   SNR loss
    ========================  ===============  ===========  =========
    ``rectangular``                  -13 dB         1.00x      0.00 dB
    ``hann``                         -32 dB         1.50x      1.76 dB
    ``hamming``                      -43 dB         1.47x      1.34 dB
    ``blackman``                     -58 dB         1.73x      2.37 dB
    ``blackmanharris``               -92 dB         2.00x      3.02 dB
    ========================  ===============  ===========  =========

    There is no free lunch in the table: every decibel of sidelobe suppression
    is paid for in resolution and in detection range.

    For ``"taylor"`` the number of near-in equal-level sidelobes :math:`\bar{n}`
    is chosen as the smallest value that makes the requested ``sidelobe_db``
    physically realisable, :math:`\bar{n} \ge 2A^2 + 1/2` with
    :math:`A = \cosh^{-1}(10^{\mathrm{SLL}/20}) / \pi` [3]_. Fixing
    :math:`\bar{n}` at a small constant instead — a common shortcut — silently
    returns a window that does not achieve the sidelobe level it was asked for.

    Examples
    --------
    >>> import numpy as np
    >>> w = taper("hann", 8)
    >>> w.shape
    (8,)
    >>> bool(np.allclose(w, w[::-1]))
    True
    >>> bool(np.isclose(w.mean(), 1.0))
    True
    """
    if name not in TAPER_NAMES:
        msg = f"unknown taper {name!r}; expected one of {', '.join(TAPER_NAMES)}."
        raise ValueError(msg)
    if n_samples < 1:
        msg = f"n_samples ({n_samples}) must be at least one."
        raise ValueError(msg)
    if sidelobe_db <= 0.0:
        msg = (
            f"sidelobe_db ({sidelobe_db:g}) must be positive; it is a depth below the "
            f"mainlobe peak, so 60.0 means -60 dB."
        )
        raise ValueError(msg)

    weights: NDArray[np.float64]
    if name == "rectangular":
        weights = np.ones(n_samples, dtype=np.float64)
    elif name == "taylor":
        weights = np.asarray(
            _scipy_windows.taylor(
                n_samples,
                nbar=_taylor_nbar(sidelobe_db),
                sll=sidelobe_db,
                norm=False,
                sym=True,
            ),
            dtype=np.float64,
        )
    elif name == "chebyshev":
        if sidelobe_db < _MIN_CHEBYSHEV_SIDELOBE_DB:
            msg = (
                f"sidelobe_db ({sidelobe_db:g}) must be at least "
                f"{_MIN_CHEBYSHEV_SIDELOBE_DB:g} for the chebyshev taper; shallower "
                f"designs put the window's largest weights at its two ends."
            )
            raise ValueError(msg)
        weights = np.asarray(
            _scipy_windows.chebwin(n_samples, at=sidelobe_db, sym=True), dtype=np.float64
        )
    else:
        weights = np.asarray(getattr(_scipy_windows, name)(n_samples, sym=True), dtype=np.float64)

    if normalize:
        mean_weight = float(np.mean(weights))
        if mean_weight <= 0.0:
            msg = f"taper {name!r} of {n_samples} samples has non-positive mean; cannot normalize."
            raise ValueError(msg)
        weights = weights / mean_weight

    result: NDArray[np.float64] = weights
    return result


def apply_taper(
    samples: ArrayLike,
    window: ArrayLike,
    *,
    axis: int = -1,
) -> NDArray[np.complex128]:
    """Multiply ``samples`` by a one-dimensional ``window`` along ``axis``.

    Parameters
    ----------
    samples : array_like
        Data of any shape — typically a complex baseband cube of shape
        ``(n_pulses, n_samples, n_rx)``, slow time on axis 0 and fast time on
        axis 1.
    window : array_like
        Real weights of shape ``(n,)``, where ``n`` is the length of
        ``samples`` along ``axis``. Usually from :func:`taper`.
    axis : int, optional
        Axis of ``samples`` the window weights. Default -1 (the last axis,
        fast time in the canonical layout).

    Returns
    -------
    numpy.ndarray
        Tapered samples, complex, of the same shape as ``samples``.

    Raises
    ------
    ValueError
        If ``window`` is not one-dimensional, or its length does not match
        ``samples`` along ``axis``, or ``axis`` is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> cube = np.ones((4, 8), dtype=np.complex128)
    >>> tapered = apply_taper(cube, taper("hann", 8), axis=1)
    >>> tapered.shape
    (4, 8)
    """
    samples_arr = np.asarray(samples, dtype=np.complex128)
    window_arr = np.asarray(window, dtype=np.float64)

    if window_arr.ndim != 1:
        msg = f"window must be one-dimensional, got shape {window_arr.shape}."
        raise ValueError(msg)
    axis = _normalize_axis(axis, samples_arr.ndim)
    if window_arr.size != samples_arr.shape[axis]:
        msg = (
            f"window length ({window_arr.size}) must match samples along axis {axis} "
            f"({samples_arr.shape[axis]})."
        )
        raise ValueError(msg)

    # Reshape to (1, ..., n, ..., 1) and let NumPy broadcast. Materialising the
    # window to the full cube shape with np.broadcast_to would allocate for
    # nothing; broadcasting is both the cheaper and the house convention.
    shape = [1] * samples_arr.ndim
    shape[axis] = window_arr.size
    result: NDArray[np.complex128] = samples_arr * window_arr.reshape(shape)
    return result


def coherent_gain_linear(window: ArrayLike) -> float:
    r"""Return the coherent gain of a taper, a dimensionless linear factor.

    .. math:: G_c = \frac{1}{N} \sum_{n=0}^{N-1} w[n]

    The factor by which the taper scales the peak of a coherently integrated
    tone. A rectangular window has :math:`G_c = 1`; every other window has
    :math:`G_c < 1` unless it has been normalised, which is exactly what
    ``taper(..., normalize=True)`` divides out.

    Parameters
    ----------
    window : array_like
        Real weights of shape ``(n,)``.

    Returns
    -------
    float
        The coherent gain, dimensionless.

    Raises
    ------
    ValueError
        If ``window`` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isclose(coherent_gain_linear(taper("rectangular", 16, normalize=False)), 1.0))
    True
    """
    window_arr = _validated_window(window)
    return float(np.mean(window_arr))


def processing_loss_db(window: ArrayLike) -> float:
    r"""Return the SNR loss a taper costs, in decibels.

    .. math::

        L = -10 \log_{10}
            \frac{\left(\sum_n w[n]\right)^2}{N \sum_n w[n]^2}

    Tapering suppresses sidelobes by throwing away the samples at the edges of
    the record, and those samples carried signal. The ratio above is the
    mismatch between the tapered filter and the matched filter, so :math:`L` is
    how much detection SNR the sidelobe suppression cost [1]_.

    The expression is invariant to scaling ``window``, so a normalised and an
    unnormalised taper of the same shape report the same loss.

    Parameters
    ----------
    window : array_like
        Real weights of shape ``(n,)``.

    Returns
    -------
    float
        The loss in dB. Always non-negative; exactly zero only for a uniform
        window.

    Raises
    ------
    ValueError
        If ``window`` is empty, or its weights sum to zero.

    Examples
    --------
    A rectangular taper is the matched filter, so it loses nothing:

    >>> import numpy as np
    >>> bool(np.isclose(processing_loss_db(taper("rectangular", 32)), 0.0, atol=1e-12))
    True

    A Hann taper costs about 1.76 dB:

    >>> bool(np.isclose(processing_loss_db(taper("hann", 4096)), 1.76, atol=0.01))
    True
    """
    window_arr = _validated_window(window)
    coherent_sum = float(np.sum(window_arr))
    if coherent_sum == 0.0:
        msg = "window weights sum to zero; processing loss is undefined."
        raise ValueError(msg)
    incoherent_sum = float(np.sum(window_arr**2))
    efficiency_linear = coherent_sum**2 / (window_arr.size * incoherent_sum)
    return float(-10.0 * np.log10(efficiency_linear))


def _taylor_nbar(sidelobe_db: float) -> int:
    r"""Return the smallest :math:`\bar{n}` that realises ``sidelobe_db`` [3]_."""
    a_parameter = np.arccosh(10.0 ** (sidelobe_db / 20.0)) / np.pi
    return max(2, int(np.ceil(2.0 * a_parameter**2 + 0.5)))


def _validated_window(window: ArrayLike) -> NDArray[np.float64]:
    """Return ``window`` as a non-empty 1-D float array, or raise."""
    window_arr = np.asarray(window, dtype=np.float64)
    if window_arr.ndim != 1:
        msg = f"window must be one-dimensional, got shape {window_arr.shape}."
        raise ValueError(msg)
    if window_arr.size == 0:
        msg = "window must not be empty."
        raise ValueError(msg)
    return window_arr


def _normalize_axis(axis: int, ndim: int) -> int:
    """Return ``axis`` as a non-negative index into an ``ndim``-dimensional array."""
    if not -ndim <= axis < ndim:
        msg = f"axis ({axis}) is out of range for an array of {ndim} dimension(s)."
        raise ValueError(msg)
    return axis % ndim
