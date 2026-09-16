r"""Constant-false-alarm-rate detection and detection clustering.

A radar detector compares each cell of a detected (square-law) map against a
threshold. A *fixed* threshold is useless in practice: the noise-plus-clutter
floor varies with range, with the antenna pattern, and with the weather, so a
threshold that gives one false alarm an hour in one range interval gives
thousands in another. A CFAR detector instead estimates the local floor from
the cells *around* the cell under test (CUT) and scales that estimate by a
constant :math:`\alpha` chosen so that the probability of false alarm holds at a
design value regardless of the floor's absolute level.

The reference window is one-dimensional, laid out along ``axis``:

.. code-block:: text

    [ train ][ guard ][ CUT ][ guard ][ train ]
     n_train  n_guard    1    n_guard  n_train
    <--- lagging --->         <--- leading --->

The guard cells are excluded from the estimate so that energy spilling out of a
strong target does not inflate the very threshold meant to detect it.

The four variants differ only in how they reduce the :math:`2N` reference cells
(``N = n_train``) to one noise estimate:

=========  ==========================================================
Variant    Noise estimate
=========  ==========================================================
``"ca"``   mean of all :math:`2N` cells — best in homogeneous noise
``"go"``   greater of the two half-window means — fewer false alarms
           at a clutter edge, at some detection loss
``"so"``   smaller of the two half-window means — holds detection when
           a second target sits in one half of the window
``"os"``   the ``rank``-th smallest cell — robust to several
           interfering targets, at ~0.5 dB CFAR loss versus ``"ca"``
=========  ==========================================================

Each variant has a closed-form :math:`P_{fa}(\alpha)` for square-law-detected
complex Gaussian noise, in which the cell powers are i.i.d. exponential.
:func:`cfar_probability_of_false_alarm` evaluates it and
:func:`cfar_threshold_factor` inverts it. Those two are the whole reason this
module is worth testing carefully: a threshold constant that is wrong by a
factor still produces a perfectly plausible-looking detection map, and only a
false-alarm-rate measurement over many noise realisations exposes it.

Notes
-----
Cells closer than ``n_guard + n_train`` to either end of ``axis`` have no
complete reference window. Rather than silently estimating the floor from a
truncated window — which changes :math:`P_{fa}` in exactly the way CFAR exists
to prevent — those cells are given a ``nan`` threshold and never declared
detections. :func:`cfar_valid_mask` reports which cells were actually tested,
and any measured false-alarm rate must be taken over those cells alone.

References
----------
.. [1] H. M. Finn and R. S. Johnson, "Adaptive detection mode with threshold
       control as a function of spatially sampled clutter-level estimates,"
       *RCA Review*, vol. 29, pp. 414-465, 1968. (CA-CFAR.)
.. [2] V. G. Hansen and J. H. Sawyers, "Detectability loss due to greatest-of
       selection in a cell-averaging CFAR," *IEEE Trans. Aerosp. Electron.
       Syst.*, vol. AES-16, no. 1, pp. 115-118, 1980. (GO-CFAR.)
.. [3] M. Weiss, "Analysis of some modified cell-averaging CFAR processors in
       multiple-target situations," *IEEE Trans. Aerosp. Electron. Syst.*,
       vol. AES-18, no. 1, pp. 102-114, 1982. (SO-CFAR.)
.. [4] H. Rohling, "Radar CFAR thresholding in clutter and multiple target
       situations," *IEEE Trans. Aerosp. Electron. Syst.*, vol. AES-19, no. 4,
       pp. 608-621, 1983. (OS-CFAR, eq. 8.)
.. [5] P. P. Gandhi and S. A. Kassam, "Analysis of CFAR processors in
       nonhomogeneous background," *IEEE Trans. Aerosp. Electron. Syst.*,
       vol. 24, no. 4, pp. 427-445, 1988. (GO/SO closed forms, eqs. 12-13.)
.. [6] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §6.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

__all__ = [
    "CFAR_VARIANTS",
    "CfarVariant",
    "Detection",
    "cfar_detect",
    "cfar_noise_estimate_w",
    "cfar_probability_of_false_alarm",
    "cfar_threshold_factor",
    "cfar_threshold_w",
    "cluster_detections",
    "default_os_rank",
]

CfarVariant = Literal["ca", "go", "so", "os"]

CFAR_VARIANTS: tuple[CfarVariant, ...] = get_args(CfarVariant)

# Upper bound for the threshold-factor bracket search. Pfa(alpha) falls at least
# geometrically in alpha, so a design Pfa small enough to need alpha > 1e12 is a
# sign of a mis-specified window rather than a bracket that needs widening.
_MAX_ALPHA_LINEAR = 1.0e12

# OS-CFAR needs every reference cell, not a running sum, so it materialises a
# sliding window. This caps that temporary at roughly a few hundred megabytes.
_OS_MAX_WINDOW_ELEMENTS = 1 << 24


@dataclass(frozen=True)
class Detection:
    """One clustered detection: a connected group of threshold crossings.

    Attributes
    ----------
    peak_index : tuple of int
        Index of the strongest cell in the cluster, as an index into the
        detected map. A tuple with one entry per map dimension.
    centroid_index : tuple of float
        Power-weighted centroid of the cluster, in fractional cell indices.
        Interpolates between cells, so it resolves a target's position more
        finely than ``peak_index`` when the target straddles two bins.
    peak_power_w : float
        Detected power of the peak cell, in watts.
    total_power_w : float
        Summed detected power over every cell in the cluster, in watts.
    n_cells : int
        Number of threshold crossings in the cluster. A cluster of one cell is
        the signature of a false alarm; a real target usually spans several.
    """

    peak_index: tuple[int, ...]
    centroid_index: tuple[float, ...]
    peak_power_w: float
    total_power_w: float
    n_cells: int


def default_os_rank(n_train: int) -> int:
    r"""Return the conventional OS-CFAR rank for ``2 * n_train`` reference cells.

    Parameters
    ----------
    n_train : int
        Number of training cells on each side of the guard band.

    Returns
    -------
    int
        The rank :math:`k`, one-based, of the order statistic to use as the
        noise estimate.

    Notes
    -----
    Rohling's recommendation is :math:`k \approx 3M/4` for :math:`M` reference
    cells [4]_: high enough that the estimate is not dominated by the low tail
    of the noise distribution, low enough that up to :math:`M/4` interfering
    targets in the window are discarded rather than averaged in.

    Examples
    --------
    >>> default_os_rank(16)
    24
    """
    _validate_window(n_train=n_train, n_guard=0)
    return max(1, round(0.75 * 2 * n_train))


def cfar_probability_of_false_alarm(
    alpha_linear: float,
    *,
    n_train: int,
    variant: CfarVariant = "ca",
    rank: int | None = None,
) -> float:
    r"""Return the false-alarm probability of a CFAR detector, analytically.

    Assumes square-law-detected complex Gaussian noise, so that each reference
    cell holds an independent exponentially distributed power with the same
    mean as the cell under test. Under that assumption :math:`P_{fa}` depends on
    ``alpha_linear`` and the window geometry but *not* on the noise level, which
    is the whole point of CFAR.

    With :math:`N` = ``n_train`` cells per side and :math:`\beta = \alpha / N`:

    .. math::

        P_{fa}^{CA} &= \left(1 + \frac{\alpha}{2N}\right)^{-2N} \\
        P_{fa}^{SO} &= 2 \sum_{k=0}^{N-1} \binom{N-1+k}{k}
                       (2 + \beta)^{-(N+k)} \\
        P_{fa}^{GO} &= 2 (1 + \beta)^{-N} - P_{fa}^{SO} \\
        P_{fa}^{OS} &= \prod_{i=0}^{k-1} \frac{M - i}{M - i + \alpha},
                       \quad M = 2N

    Parameters
    ----------
    alpha_linear : float
        Threshold factor multiplying the noise estimate, linear (not dB). Must
        be strictly positive.
    n_train : int
        Number of training cells on each side of the guard band, so
        :math:`2N` reference cells in total. Must be at least one.
    variant : {'ca', 'go', 'so', 'os'}, optional
        Which reduction of the reference cells the threshold uses. Default
        ``'ca'``.
    rank : int, optional
        For ``variant='os'`` only: the one-based rank of the order statistic,
        in ``1 .. 2 * n_train``. Defaults to :func:`default_os_rank`. Ignored
        by the other variants.

    Returns
    -------
    float
        Probability of false alarm, in ``(0, 1)``.

    Raises
    ------
    ValueError
        If ``alpha_linear`` is not strictly positive, if ``n_train`` is less
        than one, if ``variant`` is not one of the four names, or if ``rank``
        is outside ``1 .. 2 * n_train``.

    Notes
    -----
    The GO expression is a difference of two nearly equal terms, because the
    leading :math:`\beta^{-N}` behaviour cancels and GO decays one power faster
    than SO. For the window sizes and design rates used in practice
    (:math:`\beta` of order one) the cancellation costs well under one
    significant digit in float64; it only becomes a concern for a very small
    window driven to a very small :math:`P_{fa}`, where :math:`\beta` grows.

    The window always holds an even number of reference cells, :math:`M = 2N`,
    so the smallest case is two. That case is hand-checkable and pins down two
    cross-variant identities, since with :math:`N = 1` each half-window *mean*
    is just the one cell in it:

    * ``'so'`` and ``'os'`` at ``rank=1`` are both the minimum of two cells.
      The minimum of two unit-mean exponentials is exponential with mean
      :math:`1/2`, so :math:`P_{fa} = 2/(2 + \alpha)`.
    * ``'go'`` and ``'os'`` at ``rank=2`` are both the maximum of the two, giving
      :math:`P_{fa} = 2/\big((1 + \alpha)(2 + \alpha)\big)`.

    Note that a *single* reference cell would give :math:`1/(1 + \alpha)`, but no
    window geometry here produces one; reading the two-cell minimum as though it
    were one cell is an easy factor-level mistake to make.

    References
    ----------
    See module references [4]_ (OS) and [5]_ (CA, GO, SO).

    Examples
    --------
    The smallest window, checked against the hand derivation in the notes:

    >>> minimum_of_two = cfar_probability_of_false_alarm(9.0, n_train=1, variant="os", rank=1)
    >>> bool(np.isclose(minimum_of_two, 2 / (2 + 9.0)))
    True
    >>> smallest_of = cfar_probability_of_false_alarm(9.0, n_train=1, variant="so")
    >>> bool(np.isclose(minimum_of_two, smallest_of))
    True

    A wider window needs a lower threshold factor for the same rate, because
    its noise estimate is less noisy -- this difference is the CFAR loss:

    >>> narrow = cfar_probability_of_false_alarm(10.0, n_train=8, variant="ca")
    >>> wide = cfar_probability_of_false_alarm(10.0, n_train=32, variant="ca")
    >>> bool(wide < narrow)
    True

    GO and SO partition the same total, which is a strong check on both:

    >>> beta = 4.0 / 8
    >>> go = cfar_probability_of_false_alarm(4.0, n_train=8, variant="go")
    >>> so = cfar_probability_of_false_alarm(4.0, n_train=8, variant="so")
    >>> bool(np.isclose(go + so, 2.0 * (1.0 + beta) ** -8))
    True
    """
    _validate_variant(variant)
    _validate_window(n_train=n_train, n_guard=0)
    if not np.isfinite(alpha_linear) or alpha_linear <= 0.0:
        msg = f"alpha_linear must be a finite positive threshold factor, got {alpha_linear!r}."
        raise ValueError(msg)

    n_ref = 2 * n_train

    if variant == "ca":
        # Sum of n_ref exponentials is Gamma(n_ref); Pfa is its MGF at alpha/n_ref.
        return float((1.0 + alpha_linear / n_ref) ** -n_ref)

    if variant == "os":
        k = default_os_rank(n_train) if rank is None else rank
        _validate_rank(k, n_ref)
        # Product form of Rohling eq. 8; equals k * C(M, k) * B(M - k + 1 + alpha, k).
        pfa = 1.0
        for i in range(k):
            pfa *= (n_ref - i) / (n_ref - i + alpha_linear)
        return float(pfa)

    # GO and SO share the same series; GO is its complement within 2 (1 + beta)^-N.
    beta = alpha_linear / n_train
    series = 2.0 * math.fsum(
        math.comb(n_train - 1 + k, k) * (2.0 + beta) ** -(n_train + k) for k in range(n_train)
    )
    if variant == "so":
        return float(series)
    return float(2.0 * (1.0 + beta) ** -n_train - series)


def cfar_threshold_factor(
    *,
    pfa: float,
    n_train: int,
    variant: CfarVariant = "ca",
    rank: int | None = None,
) -> float:
    r"""Return the threshold factor :math:`\alpha` that achieves a design ``pfa``.

    Inverts :func:`cfar_probability_of_false_alarm`. CA-CFAR has a closed form,

    .. math:: \alpha = 2N \left( P_{fa}^{-1/2N} - 1 \right),

    and is evaluated directly; the other three variants have no closed-form
    inverse and are solved by bisection on :math:`P_{fa}(\alpha)`, which is
    continuous and strictly decreasing, so the root is unique.

    Parameters
    ----------
    pfa : float
        Design probability of false alarm, in ``(0, 1)``. A surveillance radar
        typically designs to ``1e-6`` per cell; ``1e-4`` is a convenient value
        for testing, since it is small enough to be interesting and large enough
        to measure in a few million cells.
    n_train : int
        Number of training cells on each side of the guard band.
    variant : {'ca', 'go', 'so', 'os'}, optional
        Which reduction the threshold uses. Default ``'ca'``.
    rank : int, optional
        For ``variant='os'`` only: one-based rank of the order statistic.
        Defaults to :func:`default_os_rank`.

    Returns
    -------
    float
        The threshold factor :math:`\alpha`, linear. Multiply the noise estimate
        by it to get the threshold.

    Raises
    ------
    ValueError
        If ``pfa`` is outside ``(0, 1)``, if the window or rank is invalid, or
        if no root is found below ``1e12`` — which means the requested ``pfa``
        is unreachable with a window this small.

    Notes
    -----
    :math:`\alpha` grows as the window shrinks: a small window gives a noisy
    noise estimate, and the threshold must be raised to keep the false-alarm
    rate at its design value. That rise is the *CFAR loss* — the extra SNR a
    target needs relative to a detector that knew the true noise power. It is
    why ``n_train`` is usually at least 16.

    Examples
    --------
    >>> alpha_linear = cfar_threshold_factor(pfa=1e-4, n_train=16, variant="ca")
    >>> bool(np.isclose(cfar_probability_of_false_alarm(alpha_linear, n_train=16), 1e-4))
    True

    The closed-form CA inverse agrees with the bisection used for the others:

    >>> bool(np.isclose(alpha_linear, 32 * (1e-4 ** (-1 / 32) - 1)))
    True
    """
    _validate_variant(variant)
    _validate_window(n_train=n_train, n_guard=0)
    if not 0.0 < pfa < 1.0:
        msg = f"pfa must be a probability in the open interval (0, 1), got {pfa!r}."
        raise ValueError(msg)

    n_ref = 2 * n_train
    if variant == "ca":
        return float(n_ref * (pfa ** (-1.0 / n_ref) - 1.0))

    if variant == "os" and rank is not None:
        _validate_rank(rank, n_ref)

    def excess(alpha_linear: float) -> float:
        return (
            cfar_probability_of_false_alarm(
                alpha_linear, n_train=n_train, variant=variant, rank=rank
            )
            - pfa
        )

    # Pfa(alpha) decreases from 1 towards 0, so doubling finds a bracket in a
    # few dozen steps; a Python loop is unavoidable because each step depends on
    # the previous one.
    low, high = 1e-9, 1.0
    while excess(high) > 0.0:
        high *= 2.0
        if high > _MAX_ALPHA_LINEAR:
            msg = (
                f"no threshold factor below {_MAX_ALPHA_LINEAR:g} achieves pfa={pfa!r} "
                f"for variant={variant!r} with n_train={n_train}. The reference window is "
                f"too small for a false-alarm rate this low; increase n_train."
            )
            raise ValueError(msg)

    # Bisection rather than a faster root finder: Pfa spans many decades, so a
    # derivative-based method needs careful scaling, while bisection cannot fail
    # on a bracketed monotone function. Sixty halvings reach float64 resolution.
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if excess(middle) > 0.0:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))


def cfar_valid_mask(
    shape: tuple[int, ...],
    *,
    n_train: int,
    n_guard: int,
    axis: int = -1,
) -> NDArray[np.bool_]:
    """Return which cells have a complete reference window, and so were tested.

    Parameters
    ----------
    shape : tuple of int
        Shape of the detected map.
    n_train, n_guard : int
        Training and guard cells per side.
    axis : int, optional
        Axis the reference window lies along. Default ``-1``.

    Returns
    -------
    numpy.ndarray
        Boolean array of ``shape``, True where the cell was tested. False in the
        ``n_guard + n_train`` cells at each end of ``axis``.

    Notes
    -----
    Any measured false-alarm rate must be taken over these cells alone. Counting
    the untested edge cells as non-detections dilutes the rate towards zero and
    would make a detector with far too high a threshold look correct.

    Examples
    --------
    >>> cfar_valid_mask((7,), n_train=2, n_guard=0)
    array([False, False,  True,  True,  True, False, False])
    """
    _validate_window(n_train=n_train, n_guard=n_guard)
    margin = n_guard + n_train
    valid = np.zeros(shape, dtype=np.bool_)
    axis = _normalize_axis(axis, len(shape))
    if shape[axis] < 2 * margin + 1:
        return valid
    index: list[slice] = [slice(None)] * len(shape)
    index[axis] = slice(margin, shape[axis] - margin)
    valid[tuple(index)] = True
    return valid


def cfar_noise_estimate_w(
    power_w: ArrayLike,
    *,
    n_train: int,
    n_guard: int = 1,
    variant: CfarVariant = "ca",
    rank: int | None = None,
    axis: int = -1,
) -> NDArray[np.float64]:
    r"""Estimate the local noise floor around each cell, in watts.

    Reduces the :math:`2N` reference cells around each cell under test to one
    number, by the rule that ``variant`` selects. Multiply the result by
    :func:`cfar_threshold_factor` to get a threshold; :func:`cfar_threshold_w`
    does both.

    Parameters
    ----------
    power_w : array_like
        Detected (square-law) power, in watts, of any shape. For a
        range-Doppler map from :func:`radar_forge.core.dsp.range_doppler_map`
        this is ``np.abs(cube) ** 2``.
    n_train : int
        Training cells per side, so :math:`2N` reference cells in total.
    n_guard : int, optional
        Guard cells per side, excluded from the estimate. Default 1. Set wide
        enough to cover a target's spread in ``axis``: at least the mainlobe
        width of the range window, or a strong target raises its own threshold
        and masks itself.
    variant : {'ca', 'go', 'so', 'os'}, optional
        Reduction rule. Default ``'ca'``.
    rank : int, optional
        For ``variant='os'``: one-based rank of the order statistic, in
        ``1 .. 2 * n_train``. Defaults to :func:`default_os_rank`.
    axis : int, optional
        Axis the window lies along. Default ``-1``. For a
        ``(n_doppler, n_range)`` map, ``-1`` averages along range independently
        for every Doppler bin, which is the usual choice.

    Returns
    -------
    numpy.ndarray
        Noise-floor estimate in watts, the same shape as ``power_w``, and
        ``nan`` wherever the reference window is incomplete
        (see :func:`cfar_valid_mask`).

    Raises
    ------
    ValueError
        If ``n_train`` is less than one, ``n_guard`` is negative, ``variant`` is
        unknown, ``rank`` is out of range, or ``power_w`` holds negative values
        (it is a power, so a negative entry means an amplitude was passed by
        mistake).

    Notes
    -----
    ``'ca'``, ``'go'`` and ``'so'`` are computed from a cumulative sum, so the
    cost is linear in the map size and independent of ``n_train``. ``'os'``
    needs the individual cells, so it materialises a sliding window and is
    substantially slower and hungrier.

    Examples
    --------
    In a flat floor every variant recovers the floor itself:

    >>> flat_w = np.full(11, 4.0)
    >>> estimate_w = cfar_noise_estimate_w(flat_w, n_train=2, n_guard=1)
    >>> float(estimate_w[5])
    4.0
    >>> bool(np.all(np.isnan(estimate_w[:3])))
    True
    """
    _validate_variant(variant)
    _validate_window(n_train=n_train, n_guard=n_guard)

    values_w = np.asarray(power_w, dtype=np.float64)
    if values_w.ndim == 0:
        msg = "power_w must have at least one dimension; a scalar has no reference window."
        raise ValueError(msg)
    if np.any(values_w < 0.0):
        msg = (
            "power_w must be non-negative; it is a detected power in watts. "
            "A negative entry usually means an amplitude or a dB value was passed "
            "instead of |x| ** 2."
        )
        raise ValueError(msg)

    axis = _normalize_axis(axis, values_w.ndim)
    margin = n_guard + n_train
    n_cells = values_w.shape[axis]

    estimate_w = np.full(values_w.shape, np.nan, dtype=np.float64)
    if n_cells < 2 * margin + 1:
        # No cell has a complete window; every entry stays nan.
        return estimate_w

    if variant == "os":
        k = default_os_rank(n_train) if rank is None else rank
        _validate_rank(k, 2 * n_train)
        interior_w = _order_statistic(values_w, n_train, n_guard, k, axis)
    else:
        # Each half-window is a fixed-offset band, so one cumulative sum gives
        # every window total as a difference of two slices.
        totals = np.cumsum(values_w, axis=axis)
        zero_shape = list(values_w.shape)
        zero_shape[axis] = 1
        totals = np.concatenate([np.zeros(zero_shape, dtype=np.float64), totals], axis=axis)
        lagging_w = _band_mean(totals, n_cells, margin, -margin, -n_guard - 1, n_train, axis)
        leading_w = _band_mean(totals, n_cells, margin, n_guard + 1, margin, n_train, axis)
        if variant == "ca":
            interior_w = 0.5 * (lagging_w + leading_w)
        elif variant == "go":
            interior_w = np.maximum(lagging_w, leading_w)
        else:
            interior_w = np.minimum(lagging_w, leading_w)

    index: list[slice] = [slice(None)] * values_w.ndim
    index[axis] = slice(margin, n_cells - margin)
    estimate_w[tuple(index)] = interior_w
    return estimate_w


def cfar_threshold_w(
    power_w: ArrayLike,
    *,
    pfa: float,
    n_train: int,
    n_guard: int = 1,
    variant: CfarVariant = "ca",
    rank: int | None = None,
    axis: int = -1,
) -> NDArray[np.float64]:
    """Return the adaptive detection threshold for each cell, in watts.

    The local noise estimate from :func:`cfar_noise_estimate_w` scaled by the
    threshold factor from :func:`cfar_threshold_factor`.

    Parameters
    ----------
    power_w : array_like
        Detected power in watts.
    pfa : float
        Design probability of false alarm per cell, in ``(0, 1)``.
    n_train, n_guard : int
        Training and guard cells per side; ``n_guard`` defaults to 1.
    variant : {'ca', 'go', 'so', 'os'}, optional
        Reduction rule. Default ``'ca'``.
    rank : int, optional
        Order-statistic rank for ``variant='os'``.
    axis : int, optional
        Axis the window lies along. Default ``-1``.

    Returns
    -------
    numpy.ndarray
        Threshold in watts, the same shape as ``power_w``, ``nan`` where the
        reference window is incomplete.

    Examples
    --------
    The threshold tracks the floor: scale the noise, scale the threshold.

    >>> floor_w = np.full(64, 2.0)
    >>> low_w = cfar_threshold_w(floor_w, pfa=1e-3, n_train=8, n_guard=2)
    >>> high_w = cfar_threshold_w(100.0 * floor_w, pfa=1e-3, n_train=8, n_guard=2)
    >>> bool(np.isclose(np.nanmax(high_w), 100.0 * np.nanmax(low_w)))
    True
    """
    alpha_linear = cfar_threshold_factor(pfa=pfa, n_train=n_train, variant=variant, rank=rank)
    estimate_w = cfar_noise_estimate_w(
        power_w, n_train=n_train, n_guard=n_guard, variant=variant, rank=rank, axis=axis
    )
    return alpha_linear * estimate_w


def cfar_detect(
    power_w: ArrayLike,
    *,
    pfa: float,
    n_train: int,
    n_guard: int = 1,
    variant: CfarVariant = "ca",
    rank: int | None = None,
    axis: int = -1,
) -> NDArray[np.bool_]:
    """Return a boolean mask of the cells whose power exceeds their threshold.

    Parameters
    ----------
    power_w : array_like
        Detected power in watts.
    pfa : float
        Design probability of false alarm per cell, in ``(0, 1)``.
    n_train, n_guard : int
        Training and guard cells per side; ``n_guard`` defaults to 1.
    variant : {'ca', 'go', 'so', 'os'}, optional
        Reduction rule. Default ``'ca'``.
    rank : int, optional
        Order-statistic rank for ``variant='os'``.
    axis : int, optional
        Axis the window lies along. Default ``-1``.

    Returns
    -------
    numpy.ndarray
        Boolean mask, the same shape as ``power_w``. Always False in the
        untested edge cells reported by :func:`cfar_valid_mask`.

    Notes
    -----
    The mask marks individual cells, and one target usually lights several
    adjacent ones. Pass it to :func:`cluster_detections` to collapse each group
    into a single detection.

    Examples
    --------
    A target 25 dB above a flat floor, detected while the floor is not:

    >>> rng = np.random.default_rng(20260911)
    >>> power_w = rng.exponential(1.0, size=512)
    >>> power_w[256] += 10 ** 2.5
    >>> mask = cfar_detect(power_w, pfa=1e-4, n_train=16, n_guard=2)
    >>> bool(mask[256])
    True
    >>> int(mask.sum())
    1
    """
    values_w = np.asarray(power_w, dtype=np.float64)
    threshold_w = cfar_threshold_w(
        values_w,
        pfa=pfa,
        n_train=n_train,
        n_guard=n_guard,
        variant=variant,
        rank=rank,
        axis=axis,
    )
    # Compare only where the threshold is defined: NaN comparisons are False
    # anyway, but doing it explicitly keeps numpy from warning under the
    # error-on-warning pytest policy.
    tested = ~np.isnan(threshold_w)
    detected = np.zeros(values_w.shape, dtype=np.bool_)
    np.greater(values_w, threshold_w, out=detected, where=tested)
    return detected


def cluster_detections(
    mask: ArrayLike,
    power_w: ArrayLike,
    *,
    connectivity: int = 1,
) -> list[Detection]:
    """Collapse adjacent threshold crossings into one :class:`Detection` each.

    A single target spreads over several cells — the range window's mainlobe,
    the Doppler response, and straddling between bins all contribute — so a raw
    CFAR mask over-counts targets. Grouping connected crossings and reporting
    one detection per group is what a tracker expects to be fed.

    Parameters
    ----------
    mask : array_like
        Boolean detection mask, as returned by :func:`cfar_detect`.
    power_w : array_like
        Detected power in watts, the same shape as ``mask``. Used to locate each
        cluster's peak and centroid.
    connectivity : int, optional
        How many axes two cells may differ along and still count as adjacent.
        ``1`` (default) means face-connected only — in 2-D, the 4-neighbourhood.
        ``mask.ndim`` means fully connected, the 8-neighbourhood in 2-D, which
        also merges targets touching only at a corner.

    Returns
    -------
    list of Detection
        One entry per cluster, sorted by ``peak_power_w`` descending, so the
        strongest detection comes first. Empty if nothing crossed.

    Raises
    ------
    ValueError
        If ``mask`` and ``power_w`` have different shapes, or ``connectivity``
        is outside ``1 .. mask.ndim``.

    Notes
    -----
    ``n_cells`` is worth looking at rather than discarding: a cluster of a
    single cell is the signature of a noise spike, while a real target at
    reasonable SNR spans several. Filtering on it trades detection probability
    for false-alarm rate outside the CFAR threshold, so it changes the operating
    point that ``pfa`` was calibrated for.

    Examples
    --------
    Two separated targets, each spread over three cells, come back as two
    detections with the peaks in the right bins:

    >>> power_w = np.zeros(64)
    >>> power_w[20:23] = [1.0, 5.0, 1.0]
    >>> power_w[50:53] = [1.0, 9.0, 1.0]
    >>> mask = power_w > 0.5
    >>> found = cluster_detections(mask, power_w)
    >>> [d.peak_index[0] for d in found]
    [51, 21]
    >>> [d.n_cells for d in found]
    [3, 3]
    """
    flags = np.asarray(mask, dtype=bool)
    values_w = np.asarray(power_w, dtype=np.float64)
    if flags.shape != values_w.shape:
        msg = f"mask shape {flags.shape} does not match power_w shape {values_w.shape}."
        raise ValueError(msg)
    if not 1 <= connectivity <= max(1, flags.ndim):
        msg = f"connectivity must be between 1 and mask.ndim ({flags.ndim}), got {connectivity!r}."
        raise ValueError(msg)

    structure = ndimage.generate_binary_structure(flags.ndim, connectivity)
    labels, n_found = ndimage.label(flags, structure=structure)
    if n_found == 0:
        return []

    found: list[Detection] = []
    # One pass per cluster: scipy.ndimage returns per-label scalars, but the
    # power-weighted centroid needs the cluster's own cell indices, and there
    # are only as many clusters as there are targets.
    for label in range(1, n_found + 1):
        cells = np.nonzero(labels == label)
        cluster_w = values_w[cells]
        total_power_w = float(cluster_w.sum())
        peak = int(np.argmax(cluster_w))
        # Fall back to the unweighted centroid if the cluster carries no power,
        # which happens when a mask is passed with an all-zero map.
        weights = cluster_w if total_power_w > 0.0 else np.ones_like(cluster_w)
        centroid = tuple(float(np.average(axis_cells, weights=weights)) for axis_cells in cells)
        found.append(
            Detection(
                peak_index=tuple(int(axis_cells[peak]) for axis_cells in cells),
                centroid_index=centroid,
                peak_power_w=float(cluster_w[peak]),
                total_power_w=total_power_w,
                n_cells=int(cluster_w.size),
            )
        )

    found.sort(key=lambda detection: detection.peak_power_w, reverse=True)
    return found


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _validate_variant(variant: str) -> None:
    """Reject a variant name that is not one of the four implemented rules."""
    if variant not in CFAR_VARIANTS:
        msg = f"variant must be one of {CFAR_VARIANTS!r}, got {variant!r}."
        raise ValueError(msg)


def _validate_window(*, n_train: int, n_guard: int) -> None:
    """Reject a window geometry that cannot produce a noise estimate."""
    if n_train < 1:
        msg = f"n_train must be at least 1, got {n_train!r}."
        raise ValueError(msg)
    if n_guard < 0:
        msg = f"n_guard must be non-negative, got {n_guard!r}."
        raise ValueError(msg)


def _validate_rank(rank: int, n_ref: int) -> None:
    """Reject an order-statistic rank outside the reference window."""
    if not 1 <= rank <= n_ref:
        msg = (
            f"rank must be a one-based index into the {n_ref} reference cells, "
            f"i.e. in 1..{n_ref}, got {rank!r}."
        )
        raise ValueError(msg)


def _normalize_axis(axis: int, ndim: int) -> int:
    """Return ``axis`` as a non-negative index into an ``ndim``-dimensional array."""
    if not -ndim <= axis < ndim:
        msg = f"axis {axis} is out of bounds for an array with {ndim} dimensions."
        raise ValueError(msg)
    return axis % ndim


def _band_mean(
    totals: NDArray[np.float64],
    n_cells: int,
    margin: int,
    offset_low: int,
    offset_high: int,
    n_band: int,
    axis: int,
) -> NDArray[np.float64]:
    """Mean over a fixed-offset band of cells, for every cell with a full window.

    ``totals`` is a cumulative sum prefixed with a zero, so ``totals[j]`` is the
    sum of the first ``j`` cells and the sum over the inclusive index range
    ``[a, b]`` is ``totals[b + 1] - totals[a]``. For a cell under test at ``i``
    the band spans ``[i + offset_low, i + offset_high]``, and ``i`` runs over
    ``margin .. n_cells - 1 - margin``.
    """
    first, last = margin, n_cells - 1 - margin
    upper = _slice_along(totals, first + offset_high + 1, last + offset_high + 2, axis)
    lower = _slice_along(totals, first + offset_low, last + offset_low + 1, axis)
    return (upper - lower) / n_band


def _slice_along(
    values: NDArray[np.float64], start: int, stop: int, axis: int
) -> NDArray[np.float64]:
    """Return ``values[..., start:stop, ...]`` sliced along ``axis``."""
    index: list[slice] = [slice(None)] * values.ndim
    index[axis] = slice(start, stop)
    return values[tuple(index)]


def _order_statistic(
    values_w: NDArray[np.float64],
    n_train: int,
    n_guard: int,
    rank: int,
    axis: int,
) -> NDArray[np.float64]:
    """Return the ``rank``-th smallest reference cell for every full-window cell.

    Unlike the cell-averaging variants there is no running-sum shortcut: an
    order statistic depends on all :math:`2N` cells at once, so the reference
    window is materialised and partitioned.
    """
    margin = n_guard + n_train
    window = 2 * margin + 1
    moved = np.moveaxis(values_w, axis, -1)
    windows = np.lib.stride_tricks.sliding_window_view(moved, window, axis=-1)
    # Reference cells are the two training bands; the guard cells and the cell
    # under test sit in the middle and are dropped.
    reference = np.concatenate([windows[..., :n_train], windows[..., window - n_train :]], axis=-1)

    # sliding_window_view is free but np.partition copies, so partition in
    # chunks to keep the temporary bounded. A Python loop is unavoidable here:
    # the point is to *not* have the whole array live at once.
    flat = reference.reshape(-1, 2 * n_train)
    chunk = max(1, _OS_MAX_WINDOW_ELEMENTS // (2 * n_train))
    picked = np.empty(flat.shape[0], dtype=np.float64)
    for start in range(0, flat.shape[0], chunk):
        block = flat[start : start + chunk]
        picked[start : start + chunk] = np.partition(block, rank - 1, axis=-1)[..., rank - 1]

    return np.moveaxis(picked.reshape(reference.shape[:-1]), -1, axis)
