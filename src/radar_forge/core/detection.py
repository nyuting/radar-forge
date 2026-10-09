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
           interfering targets, at 0.5-0.9 dB CFAR loss versus ``"ca"``
=========  ==========================================================

The ``_2d`` functions replace the window with a rectangular ring over two axes,
usually the ``(n_doppler, n_range)`` axes of a range-Doppler map, with one
``n_train`` and one ``n_guard`` per axis:

.. code-block:: text

    T T T T T T T T T
    T T G G G G G T T      T  training cell
    T T G G X G G T T      G  guard cell
    T T G G G G G T T      X  cell under test
    T T T T T T T T T

    n_train = (1, 2), n_guard = (1, 2)

Only ``"ca"`` and ``"os"`` are offered for the ring. Their :math:`P_{fa}`
depends on the number of reference cells :math:`M`, not on where they lie, so a
ring is calibrated as a window of :math:`M/2` cells per side; ``"go"`` and
``"so"`` need two half-windows, which a ring does not have. An axis named in
``wrap_axes`` is circular, as the Doppler axis of a range-Doppler map is.

Each variant has a closed-form :math:`P_{fa}(\alpha)` for square-law-detected
complex Gaussian noise, in which the cell powers are i.i.d. exponential.
:func:`cfar_probability_of_false_alarm` evaluates it and
:func:`cfar_threshold_factor` inverts it. Those two are the whole reason this
module is worth testing carefully: a threshold constant that is wrong by a
factor still produces a perfectly plausible-looking detection map, and only a
false-alarm-rate measurement over many noise realisations exposes it.

Notes
-----
Cells closer than ``n_guard + n_train`` to either end of a non-circular axis
have no complete reference window. Rather than silently estimating the floor
from a truncated window — which changes :math:`P_{fa}` in exactly the way CFAR
exists to prevent — those cells are given a ``nan`` threshold and never declared
detections. :func:`cfar_valid_mask` and :func:`cfar_valid_mask_2d` report which
cells were actually tested, and any measured false-alarm rate must be taken over
those cells alone.

The closed forms also assume the reference cells are *independent*, which holds
for an untapered, unpadded FFT of white noise and fails for a tapered one. A
taper correlates neighbouring bins — after a Hann window the complex amplitudes
of adjacent bins have a correlation of magnitude about 2/3 — and zero-padding
the FFT does the same. Correlated cells carry less information than as many
independent ones, so the noise estimate is noisier than the calibration assumed
and the false-alarm rate comes out above ``pfa``. This applies to the 1-D window
and the ring alike: check a calibration by measurement on an untapered map, or
expect the measured rate to be high.

References
----------
.. [1] H. M. Finn and R. S. Johnson, "Adaptive detection mode with threshold
       control as a function of spatially sampled clutter-level estimates,"
       *RCA Review*, vol. 29, pp. 414-464, Sept. 1968. (CA-CFAR.)
.. [2] V. G. Hansen and J. H. Sawyers, "Detectability loss due to 'greatest
       of' selection in a cell-averaging CFAR," *IEEE Trans. Aerosp. Electron.
       Syst.*, vol. AES-16, no. 1, pp. 115-118, 1980. (GO-CFAR.)
.. [3] M. Weiss, "Analysis of some modified cell-averaging CFAR processors in
       multiple-target situations," *IEEE Trans. Aerosp. Electron. Syst.*,
       vol. AES-18, no. 1, pp. 102-114, 1982. (SO-CFAR.)
.. [4] H. Rohling, "Radar CFAR thresholding in clutter and multiple target
       situations," *IEEE Trans. Aerosp. Electron. Syst.*, vol. AES-19, no. 4,
       pp. 608-621, 1983. (OS-CFAR, eq. 14; CFAR loss in Table IV.)
.. [5] P. P. Gandhi and S. A. Kassam, "Analysis of CFAR processors in
       nonhomogeneous background," *IEEE Trans. Aerosp. Electron. Syst.*,
       vol. 24, no. 4, pp. 427-445, 1988. (GO/SO closed forms.)
.. [6] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §6.5.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage, sparse
from scipy.sparse import csgraph

__all__ = [
    "CFAR_VARIANTS",
    "CFAR_VARIANTS_2D",
    "CfarVariant",
    "CfarVariant2d",
    "Detection",
    "cfar_detect",
    "cfar_detect_2d",
    "cfar_noise_estimate_2d_w",
    "cfar_noise_estimate_w",
    "cfar_probability_of_false_alarm",
    "cfar_threshold_2d_w",
    "cfar_threshold_factor",
    "cfar_threshold_w",
    "cfar_valid_mask",
    "cfar_valid_mask_2d",
    "cluster_detections",
    "default_os_rank",
]

CfarVariant = Literal["ca", "go", "so", "os"]

CFAR_VARIANTS: tuple[CfarVariant, ...] = get_args(CfarVariant)

# GO and SO compare two half-windows, which a 2-D ring does not have (see the
# module docstring), so the 2-D functions offer only these two.
CfarVariant2d = Literal["ca", "os"]

CFAR_VARIANTS_2D: tuple[CfarVariant2d, ...] = get_args(CfarVariant2d)

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
        finely than ``peak_index`` when the target straddles two bins. On a
        circular axis it lies in ``[0, n)``.
    peak_power_w : float
        Detected power of the peak cell, in watts.
    total_power_w : float
        Summed detected power over every cell in the cluster, in watts.
    n_cells : int
        Number of threshold crossings in the cluster. A cluster of one cell is
        the signature of a false alarm; a real target usually spans several.
    cfar_noise_estimate_w : float
        CFAR noise estimate at the peak cell, in watts, if
        :func:`cluster_detections` was given one; ``nan`` otherwise. This is
        the local estimate, not the receiver's thermal noise power.
    """

    peak_index: tuple[int, ...]
    centroid_index: tuple[float, ...]
    peak_power_w: float
    total_power_w: float
    n_cells: int
    cfar_noise_estimate_w: float = math.nan


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
        # Product form of Rohling eq. 14; equals k * C(M, k) * B(M - k + 1 + alpha, k).
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
        # P^(-1/M) - 1 as expm1(-ln P / M): P^(-1/M) is close to 1 when M is
        # large, as it is for a 2-D ring, and subtracting 1 from it loses digits.
        return float(n_ref * math.expm1(-math.log(pfa) / n_ref))

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
        mistake) or values that are not finite.

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

    values_w = _as_power_w(power_w)
    if values_w.ndim == 0:
        msg = "power_w must have at least one dimension; a scalar has no reference window."
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


def cfar_valid_mask_2d(
    shape: tuple[int, ...],
    *,
    n_train: tuple[int, int],
    n_guard: tuple[int, int],
    axes: tuple[int, int] = (-2, -1),
    wrap_axes: tuple[int, ...] = (),
) -> NDArray[np.bool_]:
    """Return which cells have a complete 2-D reference ring, and so were tested.

    Parameters
    ----------
    shape : tuple of int
        Shape of the detected map, e.g. ``(n_doppler, n_range)``.
    n_train, n_guard : tuple of int
        Training and guard cells per side, one count for each of ``axes``.
    axes : tuple of int, optional
        The two axes the ring spans. Default ``(-2, -1)``.
    wrap_axes : tuple of int, optional
        Which of ``axes`` are circular. Default none.

    Returns
    -------
    numpy.ndarray
        Boolean array of ``shape``, True where the cell was tested. False in the
        ``n_guard + n_train`` cells at each end of an axis that does not wrap,
        and everywhere if either axis is shorter than the ring, which would
        then overlap itself.

    Raises
    ------
    ValueError
        If the map has fewer than two dimensions, ``n_train``, ``n_guard`` or
        ``axes`` is not exactly two integers, a count is invalid, the axes are
        out of bounds or repeated, or ``wrap_axes`` names an axis not in
        ``axes``.

    Examples
    --------
    Doppler (axis 0) wraps, so every Doppler row is tested; range does not:

    >>> valid = cfar_valid_mask_2d((8, 9), n_train=(1, 2), n_guard=(1, 1), wrap_axes=(0,))
    >>> valid.sum(axis=0)
    array([0, 0, 0, 8, 8, 8, 0, 0, 0])
    """
    ring = _ring(len(shape), n_train=n_train, n_guard=n_guard, axes=axes, wrap_axes=wrap_axes)
    valid = np.zeros(shape, dtype=np.bool_)
    if ring.fits(shape):
        valid[ring.interior(shape)] = True
    return valid


def cfar_noise_estimate_2d_w(
    power_w: ArrayLike,
    *,
    n_train: tuple[int, int],
    n_guard: tuple[int, int],
    variant: CfarVariant2d = "ca",
    rank: int | None = None,
    axes: tuple[int, int] = (-2, -1),
    wrap_axes: tuple[int, ...] = (),
) -> NDArray[np.float64]:
    r"""Estimate the local noise floor around each cell from a 2-D ring, in watts.

    The ring is the box of ``n_guard + n_train`` cells per side around the cell
    under test, less the box of ``n_guard`` cells per side. With
    :math:`t_i` = ``n_train[i]`` and :math:`g_i` = ``n_guard[i]`` it holds

    .. math::

        M = (2 g_0 + 2 t_0 + 1)(2 g_1 + 2 t_1 + 1) - (2 g_0 + 1)(2 g_1 + 1)

    reference cells.

    Parameters
    ----------
    power_w : array_like
        Detected (square-law) power, in watts, with at least two dimensions. For
        a range-Doppler map this is ``np.abs(rd_map) ** 2``, of shape
        ``(n_doppler, n_range)``.
    n_train : tuple of int
        Training cells per side, one count for each of ``axes``. Each must be
        at least one.
    n_guard : tuple of int
        Guard cells per side, one count for each of ``axes``. Set each wide
        enough to cover a target's mainlobe along that axis, or a strong target
        raises its own threshold and masks itself.
    variant : {'ca', 'os'}, optional
        Reduction rule. Default ``'ca'``.
    rank : int, optional
        For ``variant='os'``: one-based rank of the order statistic, in
        ``1 .. M``. Defaults to :func:`default_os_rank` of ``M // 2``.
    axes : tuple of int, optional
        The two axes the ring spans. Default ``(-2, -1)``.
    wrap_axes : tuple of int, optional
        Which of ``axes`` are circular. Default none. For a
        ``(n_doppler, n_range)`` map, pass ``(0,)``.

    Returns
    -------
    numpy.ndarray
        Noise-floor estimate in watts, the same shape as ``power_w``, and
        ``nan`` wherever the ring is incomplete
        (see :func:`cfar_valid_mask_2d`).

    Raises
    ------
    ValueError
        If ``variant`` is not ``'ca'`` or ``'os'``, ``n_train``, ``n_guard`` or
        ``axes`` is not exactly two integers, a count or ``rank`` is out of
        range, ``axes`` or ``wrap_axes`` are invalid, or ``power_w`` has fewer
        than two dimensions or holds negative or non-finite values.

    Notes
    -----
    A circular axis is first extended by ``n_guard + n_train`` wrapped cells at
    each end. ``'ca'`` is then the outer box's total less the guard box's, each
    from :func:`scipy.ndimage.uniform_filter`, so its cost is linear in the map
    size and independent of the ring's size. ``'os'`` runs
    :func:`scipy.ndimage.rank_filter` with the ring as its footprint, which
    visits every ring cell and is substantially slower.

    References
    ----------
    See module references [1]_ (CA) and [4]_ (OS).

    Examples
    --------
    In a flat floor the ring recovers the floor itself, and with Doppler
    wrapping every Doppler row has an estimate:

    >>> flat_w = np.full((8, 16), 4.0)
    >>> estimate_w = cfar_noise_estimate_2d_w(
    ...     flat_w, n_train=(1, 2), n_guard=(1, 1), wrap_axes=(0,)
    ... )
    >>> float(estimate_w[0, 8])
    4.0
    >>> bool(np.all(np.isnan(estimate_w[:, :3])))
    True
    """
    _validate_variant_2d(variant)
    values_w = _as_power_w(power_w)
    ring = _ring(values_w.ndim, n_train=n_train, n_guard=n_guard, axes=axes, wrap_axes=wrap_axes)

    estimate_w = np.full(values_w.shape, np.nan, dtype=np.float64)
    if not ring.fits(values_w.shape):
        # No cell has a complete ring; every entry stays nan.
        return estimate_w

    padded_w = np.pad(values_w, ring.wrap_padding(values_w.ndim), mode="wrap")
    if variant == "ca":
        interior_w = _ring_mean_w(padded_w, ring)
    else:
        k = default_os_rank(ring.n_reference // 2) if rank is None else rank
        _validate_rank(k, ring.n_reference)
        interior_w = _ring_order_statistic_w(padded_w, ring, k)

    estimate_w[ring.interior(values_w.shape)] = interior_w
    return estimate_w


def cfar_threshold_2d_w(
    power_w: ArrayLike,
    *,
    pfa: float,
    n_train: tuple[int, int],
    n_guard: tuple[int, int],
    variant: CfarVariant2d = "ca",
    rank: int | None = None,
    axes: tuple[int, int] = (-2, -1),
    wrap_axes: tuple[int, ...] = (),
) -> NDArray[np.float64]:
    """Return the adaptive detection threshold for each cell from a 2-D ring, in watts.

    The ring's noise estimate from :func:`cfar_noise_estimate_2d_w`, scaled by
    the threshold factor from :func:`cfar_threshold_factor`.

    Parameters
    ----------
    power_w : array_like
        Detected power in watts, at least two-dimensional.
    pfa : float
        Design probability of false alarm per cell, in ``(0, 1)``.
    n_train, n_guard : tuple of int
        Training and guard cells per side, one count for each of ``axes``.
    variant : {'ca', 'os'}, optional
        Reduction rule. Default ``'ca'``.
    rank : int, optional
        Order-statistic rank for ``variant='os'``.
    axes : tuple of int, optional
        The two axes the ring spans. Default ``(-2, -1)``.
    wrap_axes : tuple of int, optional
        Which of ``axes`` are circular. Default none.

    Returns
    -------
    numpy.ndarray
        Threshold in watts, the same shape as ``power_w``, ``nan`` where the
        ring is incomplete.

    Notes
    -----
    Under the noise model the reference cells are i.i.d., and neither CA nor OS
    depends on where they lie, so a ring of :math:`M` cells takes the threshold
    factor of a window of :math:`M/2` cells per side. A taper along either axis
    breaks the independence; see the module notes.

    Examples
    --------
    A larger ring needs a lower threshold factor for the same rate:

    >>> floor_w = np.ones((32, 64))
    >>> small_w = cfar_threshold_2d_w(floor_w, pfa=1e-4, n_train=(1, 2), n_guard=(1, 1))
    >>> large_w = cfar_threshold_2d_w(floor_w, pfa=1e-4, n_train=(4, 8), n_guard=(1, 1))
    >>> bool(np.nanmax(large_w) < np.nanmax(small_w))
    True
    """
    _validate_variant_2d(variant)
    ring = _ring(np.ndim(power_w), n_train=n_train, n_guard=n_guard, axes=axes, wrap_axes=wrap_axes)
    alpha_linear = cfar_threshold_factor(
        pfa=pfa, n_train=ring.n_reference // 2, variant=variant, rank=rank
    )
    estimate_w = cfar_noise_estimate_2d_w(
        power_w,
        n_train=n_train,
        n_guard=n_guard,
        variant=variant,
        rank=rank,
        axes=axes,
        wrap_axes=wrap_axes,
    )
    return alpha_linear * estimate_w


def cfar_detect_2d(
    power_w: ArrayLike,
    *,
    pfa: float,
    n_train: tuple[int, int],
    n_guard: tuple[int, int],
    variant: CfarVariant2d = "ca",
    rank: int | None = None,
    axes: tuple[int, int] = (-2, -1),
    wrap_axes: tuple[int, ...] = (),
) -> NDArray[np.bool_]:
    """Return a boolean mask of the cells whose power exceeds their 2-D threshold.

    Parameters
    ----------
    power_w : array_like
        Detected power in watts, at least two-dimensional, e.g.
        ``(n_doppler, n_range)``.
    pfa : float
        Design probability of false alarm per cell, in ``(0, 1)``.
    n_train, n_guard : tuple of int
        Training and guard cells per side, one count for each of ``axes``.
    variant : {'ca', 'os'}, optional
        Reduction rule. Default ``'ca'``.
    rank : int, optional
        Order-statistic rank for ``variant='os'``.
    axes : tuple of int, optional
        The two axes the ring spans. Default ``(-2, -1)``.
    wrap_axes : tuple of int, optional
        Which of ``axes`` are circular. Default none.

    Returns
    -------
    numpy.ndarray
        Boolean mask, the same shape as ``power_w``. Always False in the
        untested cells reported by :func:`cfar_valid_mask_2d`.

    Notes
    -----
    Pass the mask to :func:`cluster_detections`, with the same ``wrap_axes``,
    to collapse each target's cells into one detection.

    Examples
    --------
    A target 25 dB above the floor on the first Doppler row, which only a
    wrapping ring can test:

    >>> rng = np.random.default_rng(20261006)
    >>> power_w = rng.exponential(1.0, size=(32, 128))
    >>> power_w[0, 64] += 10 ** 2.5
    >>> mask = cfar_detect_2d(
    ...     power_w, pfa=1e-4, n_train=(4, 8), n_guard=(1, 2), wrap_axes=(0,)
    ... )
    >>> bool(mask[0, 64])
    True
    """
    values_w = np.asarray(power_w, dtype=np.float64)
    threshold_w = cfar_threshold_2d_w(
        values_w,
        pfa=pfa,
        n_train=n_train,
        n_guard=n_guard,
        variant=variant,
        rank=rank,
        axes=axes,
        wrap_axes=wrap_axes,
    )
    # As in cfar_detect: compare only where the threshold is defined, so numpy
    # does not warn about nan under the error-on-warning pytest policy.
    tested = ~np.isnan(threshold_w)
    detected = np.zeros(values_w.shape, dtype=np.bool_)
    np.greater(values_w, threshold_w, out=detected, where=tested)
    return detected


def cluster_detections(
    mask: ArrayLike,
    power_w: ArrayLike,
    *,
    noise_estimate_w: ArrayLike | None = None,
    connectivity: int = 1,
    wrap_axes: tuple[int, ...] = (),
) -> list[Detection]:
    """Collapse adjacent threshold crossings into one :class:`Detection` each.

    A single target spreads over several cells — the range window's mainlobe,
    the Doppler response, and straddling between bins all contribute — so a raw
    CFAR mask over-counts targets. Grouping connected crossings and reporting
    one detection per group is what a tracker expects to be fed.

    Parameters
    ----------
    mask : array_like
        Boolean detection mask, as returned by :func:`cfar_detect` or
        :func:`cfar_detect_2d`.
    power_w : array_like
        Detected power in watts, the same shape as ``mask``. Used to locate each
        cluster's peak and centroid.
    noise_estimate_w : array_like, optional
        CFAR noise estimate in watts, the same shape as ``mask``, from
        :func:`cfar_noise_estimate_w` or :func:`cfar_noise_estimate_2d_w`. If
        given, each detection carries its value at the peak cell as
        ``cfar_noise_estimate_w``.
    connectivity : int, optional
        How many axes two cells may differ along and still count as adjacent.
        ``1`` (default) means face-connected only — in 2-D, the 4-neighbourhood.
        ``mask.ndim`` means fully connected, the 8-neighbourhood in 2-D, which
        also merges targets touching only at a corner.
    wrap_axes : tuple of int, optional
        Axes that are circular, so that the cells at the two ends are adjacent.
        Default none. Pass the Doppler axis of a range-Doppler map, so that a
        target on the ±v wrap is one detection, not two.

    Returns
    -------
    list of Detection
        One entry per cluster, sorted by ``peak_power_w`` descending, so the
        strongest detection comes first. Empty if nothing crossed.

    Raises
    ------
    ValueError
        If ``mask``, ``power_w`` and ``noise_estimate_w`` differ in shape,
        if ``connectivity`` is outside ``1 .. mask.ndim``, or if an axis in
        ``wrap_axes`` is out of bounds.

    Notes
    -----
    ``n_cells`` is worth looking at rather than discarding: a cluster of a
    single cell is the signature of a noise spike, while a real target at
    reasonable SNR spans several. Filtering on it trades detection probability
    for false-alarm rate outside the CFAR threshold, so it changes the operating
    point that ``pfa`` was calibrated for.

    On a circular axis the centroid is a weighted mean of each cell's offset from
    the peak, taken the short way round, and is reported modulo the axis length.
    That assumes a cluster covers less than half the axis.

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

    On a circular axis, crossings at the two ends are one target, centred on
    the wrap between them:

    >>> power_w = np.zeros(16)
    >>> power_w[[15, 0]] = 4.0
    >>> [d.centroid_index[0] for d in cluster_detections(power_w > 0, power_w, wrap_axes=(0,))]
    [15.5]
    """
    flags = np.asarray(mask, dtype=bool)
    values_w = np.asarray(power_w, dtype=np.float64)
    if flags.shape != values_w.shape:
        msg = f"mask shape {flags.shape} does not match power_w shape {values_w.shape}."
        raise ValueError(msg)
    estimates_w = (
        None if noise_estimate_w is None else np.asarray(noise_estimate_w, dtype=np.float64)
    )
    if estimates_w is not None and estimates_w.shape != flags.shape:
        msg = f"noise_estimate_w shape {estimates_w.shape} does not match mask shape {flags.shape}."
        raise ValueError(msg)
    if not 1 <= connectivity <= max(1, flags.ndim):
        msg = f"connectivity must be between 1 and mask.ndim ({flags.ndim}), got {connectivity!r}."
        raise ValueError(msg)
    circular = {_normalize_axis(axis, flags.ndim) for axis in wrap_axes}

    structure = ndimage.generate_binary_structure(flags.ndim, connectivity)
    labels, n_found = _label_on_circles(flags, structure, circular)
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
        peak_index = tuple(int(axis_cells[peak]) for axis_cells in cells)
        # Fall back to the unweighted centroid if the cluster carries no power,
        # which happens when a mask is passed with an all-zero map.
        weights = cluster_w if total_power_w > 0.0 else np.ones_like(cluster_w)
        centroid = tuple(
            _centroid_on_circle(axis_cells, weights, peak_index[axis], flags.shape[axis])
            if axis in circular
            else float(np.average(axis_cells, weights=weights))
            for axis, axis_cells in enumerate(cells)
        )
        found.append(
            Detection(
                peak_index=peak_index,
                centroid_index=centroid,
                peak_power_w=float(cluster_w[peak]),
                total_power_w=total_power_w,
                n_cells=int(cluster_w.size),
                cfar_noise_estimate_w=(
                    math.nan if estimates_w is None else float(estimates_w[peak_index])
                ),
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


def _validate_variant_2d(variant: str) -> None:
    """Reject a variant that the 2-D ring does not offer."""
    if variant not in CFAR_VARIANTS_2D:
        msg = (
            f"variant must be one of {CFAR_VARIANTS_2D!r} for a 2-D ring, got {variant!r}. "
            "'go' and 'so' compare two half-windows, which a ring does not have."
        )
        raise ValueError(msg)


def _as_power_w(power_w: ArrayLike) -> NDArray[np.float64]:
    """Return ``power_w`` as a float64 array, rejecting values no power can take."""
    values_w = np.asarray(power_w, dtype=np.float64)
    if not np.all(np.isfinite(values_w)):
        msg = "power_w must be finite; a nan or inf spreads into every window that contains it."
        raise ValueError(msg)
    if np.any(values_w < 0.0):
        msg = (
            "power_w must be non-negative; it is a detected power in watts. "
            "A negative entry usually means an amplitude or a dB value was passed "
            "instead of |x| ** 2."
        )
        raise ValueError(msg)
    return values_w


@dataclass(frozen=True)
class _Ring:
    """A checked 2-D reference ring: its two axes, its sizes and which axes wrap.

    Every pair is in the order of ``axes``.
    """

    axes: tuple[int, int]
    n_train: tuple[int, int]
    n_guard: tuple[int, int]
    wraps: tuple[bool, bool]

    @property
    def margins(self) -> tuple[int, int]:
        """Cells from the cell under test to the ring's outer edge, per axis."""
        return (self.n_guard[0] + self.n_train[0], self.n_guard[1] + self.n_train[1])

    @property
    def n_reference(self) -> int:
        """Number of cells in the ring: the outer box less the guard box."""
        outer = (2 * self.margins[0] + 1) * (2 * self.margins[1] + 1)
        guard = (2 * self.n_guard[0] + 1) * (2 * self.n_guard[1] + 1)
        return outer - guard

    def fits(self, shape: tuple[int, ...]) -> bool:
        """Whether a map of ``shape`` is at least one window long on both axes."""
        return all(
            shape[axis] >= 2 * margin + 1
            for axis, margin in zip(self.axes, self.margins, strict=True)
        )

    def interior(self, shape: tuple[int, ...]) -> tuple[slice, ...]:
        """Index of the cells with a complete ring: all of a circular axis, the middle of others."""
        widths = (0 if self.wraps[0] else self.margins[0], 0 if self.wraps[1] else self.margins[1])
        return _inner(shape, self.axes, widths)

    def box_size(self, ndim: int, half_widths: tuple[int, int]) -> tuple[int, ...]:
        """Filter size of a box ``half_widths`` cells per side on the ring axes, 1 elsewhere."""
        size = [1] * ndim
        for axis, half_width in zip(self.axes, half_widths, strict=True):
            size[axis] = 2 * half_width + 1
        return tuple(size)

    def wrap_padding(self, ndim: int) -> list[tuple[int, int]]:
        """``np.pad`` widths that extend each circular axis by one margin at each end."""
        padding = [(0, 0)] * ndim
        for axis, margin, wraps in zip(self.axes, self.margins, self.wraps, strict=True):
            if wraps:
                padding[axis] = (margin, margin)
        return padding


def _ring(
    ndim: int,
    *,
    n_train: tuple[int, int],
    n_guard: tuple[int, int],
    axes: tuple[int, int],
    wrap_axes: tuple[int, ...],
) -> _Ring:
    """Check a 2-D ring's arguments against an ``ndim``-dimensional map."""
    if ndim < 2:
        msg = f"a 2-D ring needs a map with at least two dimensions, got {ndim}."
        raise ValueError(msg)
    n_train = _integer_pair("n_train", n_train)
    n_guard = _integer_pair("n_guard", n_guard)
    for train, guard in zip(n_train, n_guard, strict=True):
        _validate_window(n_train=train, n_guard=guard)
    first, second = (_normalize_axis(axis, ndim) for axis in _integer_pair("axes", axes))
    if first == second:
        msg = f"axes must name two different axes, got {axes!r}."
        raise ValueError(msg)
    circular = {_normalize_axis(axis, ndim) for axis in wrap_axes}
    if not circular <= {first, second}:
        msg = f"wrap_axes {wrap_axes!r} must be among the ring's axes {axes!r}."
        raise ValueError(msg)
    return _Ring(
        axes=(first, second),
        n_train=n_train,
        n_guard=n_guard,
        wraps=(first in circular, second in circular),
    )


def _integer_pair(name: str, value: tuple[int, int]) -> tuple[int, int]:
    """Return ``value`` as two Python ints, or raise naming the argument.

    The ring takes one count per axis, so anything but exactly two integers is a
    mistake: a third entry would otherwise be dropped, a fractional count
    truncated, and the 1-D habit of a bare ``n_train=16`` fail on iteration.
    ``operator.index`` accepts NumPy integers and refuses floats.
    """
    try:
        entries = tuple(operator.index(entry) for entry in value)
    except TypeError:
        entries = ()
    if len(entries) != 2:
        msg = f"{name} must be two integers, one for each of the ring's axes, got {value!r}."
        raise ValueError(msg)
    return entries[0], entries[1]


def _ring_mean_w(padded_w: NDArray[np.float64], ring: _Ring) -> NDArray[np.float64]:
    """Mean over the ring of every cell with a complete ring.

    The ring's total is the outer box's total less the guard box's, and
    ``uniform_filter`` gives each box's mean with a running sum along each axis.
    """
    outer_size = ring.box_size(padded_w.ndim, ring.margins)
    guard_size = ring.box_size(padded_w.ndim, ring.n_guard)
    outer_mean_w: NDArray[np.float64] = ndimage.uniform_filter(padded_w, size=outer_size)
    guard_mean_w: NDArray[np.float64] = ndimage.uniform_filter(padded_w, size=guard_size)
    ring_total_w = outer_mean_w * math.prod(outer_size) - guard_mean_w * math.prod(guard_size)
    # The difference of two box totals can come out a roundoff-sized negative
    # where the ring holds only zeros, and a noise power cannot.
    centres = _inner(padded_w.shape, ring.axes, ring.margins)
    return np.maximum(ring_total_w[centres], 0.0) / ring.n_reference


def _ring_order_statistic_w(
    padded_w: NDArray[np.float64], ring: _Ring, rank: int
) -> NDArray[np.float64]:
    """Return the ``rank``-th smallest ring cell for every cell with a complete ring.

    The footprint is the outer box with the guard box, ``n_train`` cells in from
    each edge, switched off.
    """
    footprint = np.ones(ring.box_size(padded_w.ndim, ring.margins), dtype=np.bool_)
    footprint[_inner(footprint.shape, ring.axes, ring.n_train)] = False
    picked_w: NDArray[np.float64] = ndimage.rank_filter(padded_w, rank - 1, footprint=footprint)
    return picked_w[_inner(padded_w.shape, ring.axes, ring.margins)]


def _inner(
    shape: tuple[int, ...], axes: tuple[int, int], widths: tuple[int, int]
) -> tuple[slice, ...]:
    """Index that drops ``widths[i]`` cells from each end of axis ``axes[i]``."""
    index = [slice(None)] * len(shape)
    for axis, width in zip(axes, widths, strict=True):
        index[axis] = slice(width, shape[axis] - width)
    return tuple(index)


def _label_on_circles(
    flags: NDArray[np.bool_], structure: NDArray[np.bool_], circular: set[int]
) -> tuple[NDArray[np.int_], int]:
    """Label connected cells, treating each axis in ``circular`` as a circle.

    ``scipy.ndimage.label`` has no periodic boundary. So the mask is padded with
    one wrapped cell at each end of each circular axis, and then labelled. Each
    padded cell is a copy of a cell at the far end of the axis, so a cluster
    that touches a copy touches its original across the wrap. Merging the two
    labels is a connected-components problem on a small graph of labels.
    """
    if not circular:
        labels, n_found = ndimage.label(flags, structure=structure)
        return labels, int(n_found)

    padding = [(1, 1) if axis in circular else (0, 0) for axis in range(flags.ndim)]
    padded_labels, n_padded = ndimage.label(
        np.pad(flags, padding, mode="wrap"), structure=structure
    )
    if n_padded == 0:
        return np.zeros(flags.shape, dtype=np.int_), 0
    inner = tuple(slice(1, -1) if axis in circular else slice(None) for axis in range(flags.ndim))
    labels = padded_labels[inner]

    # Join the label of every labelled padded cell to the label of the cell it
    # copies. For a cell inside the original map, that is its own label.
    cells = np.nonzero(padded_labels)
    originals = tuple(
        (index - 1) % flags.shape[axis] if axis in circular else index
        for axis, index in enumerate(cells)
    )
    joins = sparse.coo_matrix(
        (np.ones(cells[0].size), (padded_labels[cells] - 1, labels[originals] - 1)),
        shape=(n_padded, n_padded),
    )
    n_found, component = csgraph.connected_components(joins, directed=False)
    return np.where(labels > 0, component[labels - 1] + 1, 0), int(n_found)


def _centroid_on_circle(
    indices: NDArray[np.intp], weights: NDArray[np.float64], peak: int, n_cells: int
) -> float:
    """Weighted mean of cell indices on a circle of ``n_cells``, in ``[0, n_cells)``.

    Each index is replaced by its offset from the peak, taken the short way
    round, so the mean is an ordinary weighted mean of small offsets.
    """
    offsets = (indices - peak + n_cells // 2) % n_cells - n_cells // 2
    centre = (peak + float(np.average(offsets, weights=weights))) % n_cells
    # Python's % of a tiny negative number can round up to exactly n_cells.
    return centre if centre < n_cells else 0.0
