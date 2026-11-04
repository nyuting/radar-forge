r"""Constant-false-alarm-rate (CFAR) detection, and clustering of what it detects.

A CFAR detector estimates the noise floor around each cell under test (CUT) from its
*reference footprint*, a training box with a guard box around the CUT cut out of it, and
multiplies that estimate by a factor :math:`\alpha` chosen so that noise crosses with
probability ``pfa`` whatever its level [1]_ [6]_. One private N-D engine on
:mod:`scipy.ndimage` serves :func:`cfar_detect` (a window along one axis) and
:func:`cfar_detect_2d` (a ring over the last two axes). The variants differ only in how they
reduce the :math:`M` reference cells (:func:`cfar_reference_cell_count`):

* ``"ca"``, their mean [1]_: best in homogeneous noise.
* ``"go"`` and ``"so"``, the greater or smaller of the two half-footprint means [2]_ [3]_ [5]_:
  fewer false alarms at a clutter edge, or detection held beside an interferer. The leading
  half is the cells beyond the CUT along the last axis, plus those beyond it along the first
  axis in the CUT's own column; the lagging half is its mirror image. Each holds :math:`M/2`.
* ``"os"``, the ``rank``-th smallest [4]_: robust to several interfering targets.

**Edge policy**, the same for every variant and dimensionality: a circular axis (``wrap``,
``wrap_axes``) reads its footprint round the wrap. On any other axis the footprint is clipped
to the map, and each cell is calibrated for the :math:`M_c \le M` reference cells it keeps.
CA takes their mean. GO and SO need two equal halves, so at a clipped cell they reduce to CA
over the cells left, as a one-sided window does [6]_. OS reads the missing cells as zero
power, which sorts below every real cell, so it takes rank :math:`k - (M - M_c)` of the
:math:`M_c` real cells; a cell where that falls below one is not tested (``nan`` threshold).
Every tested cell, edge or interior, holds the design ``pfa``.

The calibration assumes independent exponential cells (square-law complex Gaussian noise).
A taper, zero padding or oversampling correlates neighbours, and heavy-tailed clutter
(Weibull, K) is not exponential: either raises the rate above ``pfa``. The
variability-index CFAR [7]_ and distribution-matched detectors are out of scope.

References
----------
.. [1] H. M. Finn and R. S. Johnson, "Adaptive detection mode with threshold control as a
       function of spatially sampled clutter-level estimates," *RCA Review*, vol. 29,
       pp. 414-464, 1968. (CA-CFAR.)
.. [2] V. G. Hansen and J. H. Sawyers, "Detectability loss due to 'greatest of' selection
       in a cell-averaging CFAR," *IEEE Trans. AES*, vol. AES-16, no. 1, pp. 115-118, 1980.
.. [3] M. Weiss, "Analysis of some modified cell-averaging CFAR processors in
       multiple-target situations," *IEEE Trans. AES*, vol. AES-18, no. 1, pp. 102-114, 1982.
.. [4] H. Rohling, "Radar CFAR thresholding in clutter and multiple target situations,"
       *IEEE Trans. AES*, vol. AES-19, no. 4, pp. 608-621, 1983. (OS-CFAR, eq. 14.)
.. [5] P. P. Gandhi and S. A. Kassam, "Analysis of CFAR processors in nonhomogeneous
       background," *IEEE Trans. AES*, vol. 24, no. 4, pp. 427-445, 1988. (GO/SO.)
.. [6] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill,
       2014, §6.5.
.. [7] M. E. Smith and P. K. Varshney, "Intelligent CFAR processor based in data
       variability," *IEEE Trans. AES*, vol. 36, no. 3, pp. 837-847, 2000.
.. [8] A. Rosenfeld and J. L. Pfaltz, "Sequential operations in digital picture
       processing," *J. ACM*, vol. 13, no. 4, pp. 471-494, 1966. (Component labelling.)
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage, optimize, sparse, special
from scipy.sparse import csgraph

__all__ = [
    "CFAR_VARIANTS",
    "CfarVariant",
    "Detection",
    "cfar_detect",
    "cfar_detect_2d",
    "cfar_probability_of_false_alarm",
    "cfar_reference_cell_count",
    "cfar_threshold_factor",
    "cluster_detections",
    "default_os_rank",
    "label_connected_cells",
]

CfarVariant = Literal["ca", "go", "so", "os"]
CFAR_VARIANTS: tuple[CfarVariant, ...] = get_args(CfarVariant)

# A pfa that needs alpha above this means the window is too small, not the bracket.
_MAX_ALPHA_LINEAR = 1.0e12


@dataclass(frozen=True)
class Detection:
    """One cluster of connected threshold crossings, reduced to a point.

    Attributes
    ----------
    peak_index : tuple of int
        Index of the strongest cell, one entry per map dimension.
    centroid_index : tuple of float
        Power-weighted centroid in fractional cell indices; in ``[0, n)`` on a circular axis.
    peak_power_w, total_power_w : float
        Power of the peak cell, and summed over the cluster, in watts.
    n_cells : int
        Number of crossings in the cluster. One cell is the signature of a false alarm.
    cfar_threshold_w : float
        CFAR threshold at the peak cell in watts, if :func:`cluster_detections` was given
        one, else ``nan``. ``peak_power_w / cfar_threshold_w`` is the margin it crossed by.
    """

    peak_index: tuple[int, ...]
    centroid_index: tuple[float, ...]
    peak_power_w: float
    total_power_w: float
    n_cells: int
    cfar_threshold_w: float = math.nan


def default_os_rank(n_train: int) -> int:
    r"""Return Rohling's OS-CFAR rank :math:`k = \mathrm{round}(3M/4)`, ``M = 2 * n_train`` [4]_.

    High enough to ignore the low tail of the noise, low enough to discard up to :math:`M/4`
    interfering targets.

    Examples
    --------
    >>> default_os_rank(16)
    24
    """
    return max(1, round(1.5 * n_train))


def cfar_reference_cell_count(
    n_train: int | tuple[int, ...], n_guard: int | tuple[int, ...]
) -> int:
    r"""Return the number of reference cells :math:`M` in a CFAR footprint.

    The box of ``n_guard + n_train`` cells per side less the box of ``n_guard`` per side:
    :math:`M = \prod_i (2g_i + 2t_i + 1) - \prod_i (2g_i + 1)`, always even. A bare int is
    one axis, where :math:`M = 2t`.

    Parameters
    ----------
    n_train, n_guard : int or tuple of int
        Training and guard cells per side, one count per window axis.

    Examples
    --------
    >>> cfar_reference_cell_count((1, 2), (1, 2))  # a 5 x 9 box less a 3 x 5 guard box
    30
    """
    train, guard = _window(n_train, n_guard)
    outer = math.prod(2 * (g + t) + 1 for t, g in zip(train, guard, strict=True))
    return outer - math.prod(2 * g + 1 for g in guard)


def cfar_probability_of_false_alarm(
    alpha_linear: ArrayLike, *, n_train: int, variant: CfarVariant = "ca", rank: int | None = None
) -> NDArray[np.float64]:
    r"""Return the false-alarm probability of a CFAR detector with factor ``alpha_linear``.

    For :math:`M = 2N` independent exponential reference cells, :math:`\beta = \alpha / N`
    and :math:`I_x(a, b)` the regularised incomplete beta function:

    .. math::

        P_{fa}^{CA} &= (1 + \alpha / M)^{-M} \\
        P_{fa}^{SO} &= 2 (1 + \beta)^{-N} I_{(1+\beta)/(2+\beta)}(N, N) \\
        P_{fa}^{GO} &= 2 (1 + \beta)^{-N} I_{1/(2+\beta)}(N, N) \\
        P_{fa}^{OS} &= \frac{\Gamma(M + 1)\, \Gamma(M - k + 1 + \alpha)}
                            {\Gamma(M - k + 1)\, \Gamma(M + 1 + \alpha)}

    CA is the Gamma moment-generating function [6]_ and OS is Rohling's eq. 14 [4]_. GO and SO
    sum the negative-binomial series of [5]_ in closed form, so neither is a difference of
    near-equal terms.

    Parameters
    ----------
    alpha_linear : array_like
        Threshold factor, linear, strictly positive. The result has its shape.
    n_train : int
        Training cells per side, :math:`N`; for an N-D footprint, ``M // 2``.
    variant : {'ca', 'go', 'so', 'os'}, optional
        Reduction rule. Default ``'ca'``.
    rank : int, optional
        OS rank :math:`k` in ``1 .. 2N``; default :func:`default_os_rank`. Ignored otherwise.

    Examples
    --------
    The minimum of two unit exponentials has mean 1/2, so :math:`P_{fa} = 2 / (2 + \alpha)`:

    >>> float(cfar_probability_of_false_alarm(9.0, n_train=1, variant="so"))
    0.18181818181818182
    """
    alpha = np.asarray(alpha_linear, dtype=np.float64)
    if not np.all(np.isfinite(alpha) & (alpha > 0.0)):
        msg = f"alpha_linear must be a finite positive threshold factor, got {alpha_linear!r}."
        raise ValueError(msg)
    n_reference = cfar_reference_cell_count(n_train, 0)
    k = _rank(rank, n_reference) if variant == "os" else 0
    return _pfa(alpha, n_reference, _variant(variant), k)


def cfar_threshold_factor(
    *, pfa: float, n_train: int, variant: CfarVariant = "ca", rank: int | None = None
) -> float:
    r"""Return the threshold factor :math:`\alpha` that gives a design ``pfa``.

    CA inverts in closed form, :math:`\alpha = M (P_{fa}^{-1/M} - 1)`. GO, SO and OS are
    solved by :func:`scipy.optimize.brentq` on :math:`\ln \alpha`; :math:`P_{fa}` is
    continuous and strictly decreasing in :math:`\alpha`, so the root is unique. A smaller
    window needs a larger :math:`\alpha`: that rise is the CFAR loss.

    Parameters
    ----------
    pfa : float
        Design probability of false alarm per cell, in ``(0, 1)``.
    n_train, variant, rank
        As :func:`cfar_probability_of_false_alarm`.

    Raises
    ------
    ValueError
        If an argument is out of range, or no :math:`\alpha` below ``1e12`` reaches ``pfa``.

    Examples
    --------
    >>> round(10 * math.log10(cfar_threshold_factor(pfa=1e-5, n_train=16)), 3)
    11.417
    """
    n_reference = cfar_reference_cell_count(n_train, 0)
    k = _rank(rank, n_reference) if variant == "os" else 0
    return _alpha(_pfa_checked(pfa), n_reference, _variant(variant), k)


def cfar_detect(
    power_w: ArrayLike,
    pfa: float,
    *,
    n_train: int,
    n_guard: int = 1,
    variant: CfarVariant = "ca",
    rank: int | None = None,
    axis: int = -1,
    wrap: bool = False,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Detect the cells whose power exceeds a CFAR threshold from a window along one axis.

    Parameters
    ----------
    power_w : array_like
        Detected (square-law) power in watts, non-negative, e.g. ``np.abs(rd_map) ** 2`` of
        shape ``(n_doppler, n_range)``.
    pfa : float
        Design probability of false alarm per cell, in ``(0, 1)``.
    n_train, n_guard : int
        Training and guard cells per side. Set ``n_guard`` to cover a target's spread.
    variant : {'ca', 'go', 'so', 'os'}, optional
        Reduction rule. Default ``'ca'``.
    rank : int, optional
        OS rank in ``1 .. 2 * n_train``; default :func:`default_os_rank`.
    axis : int, optional
        Axis the window lies along. Default ``-1``.
    wrap : bool, optional
        Whether ``axis`` is circular. Default False: the window is clipped at the ends.

    Returns
    -------
    detected : numpy.ndarray of bool
        True where ``power_w`` exceeds the threshold.
    threshold_w : numpy.ndarray
        Threshold in watts, ``nan`` in an untested cell. Both have the shape of ``power_w``.

    Examples
    --------
    A target 25 dB above a unit floor, in the first cell of a line:

    >>> power_w = np.random.default_rng(1).exponential(1.0, size=512)
    >>> power_w[0] += 10**2.5
    >>> detected, threshold_w = cfar_detect(power_w, 1e-4, n_train=16, n_guard=2)
    >>> np.flatnonzero(detected).tolist()
    [0]
    """
    values_w = _power(power_w, min_ndim=1)
    axes = (_axis(axis, values_w.ndim),)
    threshold_w, _, _ = _cfar(values_w, pfa, (n_train,), (n_guard,), variant, rank, axes, (wrap,))
    return values_w > threshold_w, threshold_w


def cfar_detect_2d(
    power_w: ArrayLike,
    pfa: float,
    *,
    n_train: tuple[int, int],
    n_guard: tuple[int, int],
    variant: CfarVariant = "ca",
    rank: int | None = None,
    wrap_axes: tuple[int, ...] = (),
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Detect the cells whose power exceeds a CFAR threshold from a ring over the last two axes.

    Parameters
    ----------
    power_w : array_like
        Detected power in watts, at least 2-D, e.g. ``(n_doppler, n_range)``.
    pfa, variant
        As :func:`cfar_detect`.
    n_train, n_guard : tuple of int
        Training and guard cells per side, one count for each of the last two axes. Zero on
        one axis reduces the ring to :func:`cfar_detect` along the other.
    rank : int, optional
        OS rank in ``1 .. M``; default :func:`default_os_rank` of :math:`M/2`.
    wrap_axes : tuple of int, optional
        Which of the last two axes are circular, as array axes; ``(0,)`` for Doppler.

    Returns
    -------
    detected, threshold_w : numpy.ndarray
        As :func:`cfar_detect`.

    Examples
    --------
    >>> power_w = np.random.default_rng(2).exponential(1.0, size=(32, 128))
    >>> power_w[0, 64] += 10**2.5
    >>> detected, _ = cfar_detect_2d(power_w, 1e-4, n_train=(4, 8), n_guard=(1, 2), wrap_axes=(0,))
    >>> np.argwhere(detected).tolist()
    [[0, 64]]
    """
    values_w = _power(power_w, min_ndim=2)
    circular = {_axis(axis, values_w.ndim) for axis in wrap_axes}
    axes = (values_w.ndim - 2, values_w.ndim - 1)
    if not circular <= set(axes):
        msg = f"wrap_axes {wrap_axes!r} must be among the last two axes."
        raise ValueError(msg)
    wraps = tuple(axis in circular for axis in axes)
    threshold_w, _, _ = _cfar(values_w, pfa, n_train, n_guard, variant, rank, axes, wraps)
    return values_w > threshold_w, threshold_w


def label_connected_cells(
    mask: ArrayLike, *, connectivity: int = 1, wrap_axes: tuple[int, ...] = ()
) -> tuple[NDArray[np.int_], int]:
    """Give each connected group of True cells its own label, ``1 .. n_groups`` [8]_.

    :func:`scipy.ndimage.label` has no periodic boundary, so a circular axis is padded with
    one wrapped cell at each end, the padded map is labelled, and each padded copy's label is
    joined to its original's as connected components of a graph of labels.

    Parameters
    ----------
    mask : array_like of bool
        Detection mask.
    connectivity : int, optional
        ``1`` (default) joins face neighbours only; ``mask.ndim`` joins corners too.
    wrap_axes : tuple of int, optional
        Circular axes, whose two ends are adjacent.

    Returns
    -------
    labels : numpy.ndarray of int
        0 off the mask, else the cell's group.
    n_groups : int
        Number of groups.

    Examples
    --------
    >>> label_connected_cells(np.array([1, 1, 0, 1], dtype=bool), wrap_axes=(0,))[0].tolist()
    [1, 1, 0, 1]
    """
    flags = np.asarray(mask, dtype=bool)
    if not 1 <= connectivity <= max(1, flags.ndim):
        msg = f"connectivity must be between 1 and mask.ndim ({flags.ndim}), got {connectivity!r}."
        raise ValueError(msg)
    circular = {_axis(axis, flags.ndim) for axis in wrap_axes}
    structure = ndimage.generate_binary_structure(flags.ndim, connectivity)
    padding = [(1, 1) if axis in circular else (0, 0) for axis in range(flags.ndim)]
    padded, n_padded = ndimage.label(np.pad(flags, padding, mode="wrap"), structure=structure)
    labels = padded[_crop(padding, flags.shape)]
    if not circular or n_padded == 0:
        return labels, int(n_padded)
    # Join the label of every labelled padded cell to that of the cell it copies.
    cells = np.nonzero(padded)
    originals = tuple(
        (index - 1) % flags.shape[axis] if axis in circular else index
        for axis, index in enumerate(cells)
    )
    joins = sparse.coo_matrix(
        (np.ones(cells[0].size), (padded[cells] - 1, labels[originals] - 1)),
        shape=(n_padded, n_padded),
    )
    n_groups, component = csgraph.connected_components(joins, directed=False)
    return np.where(labels > 0, component[labels - 1] + 1, 0), int(n_groups)


def cluster_detections(
    mask: ArrayLike,
    power_w: ArrayLike,
    *,
    threshold_w: ArrayLike | None = None,
    connectivity: int = 1,
    wrap_axes: tuple[int, ...] = (),
) -> list[Detection]:
    """Collapse each connected group of crossings into one :class:`Detection`.

    Groups come from :func:`label_connected_cells`; each is reduced to its peak, power-weighted
    centroid and total power with :mod:`scipy.ndimage` label statistics. Two targets whose
    crossings touch are one cluster, and the centroid moves with any sidelobe or noise cell
    that crosses; there is no non-maximum suppression.

    Parameters
    ----------
    mask : array_like of bool
        Detection mask, e.g. from :func:`cfar_detect`.
    power_w : array_like
        Detected power in watts, the shape of ``mask``.
    threshold_w : array_like, optional
        CFAR threshold in watts, the shape of ``mask``, recorded at each peak.
    connectivity : int, optional
        As :func:`label_connected_cells`. Default ``1``.
    wrap_axes : tuple of int, optional
        Circular axes. Pass the Doppler axis, or a target on the velocity wrap is two.

    Returns
    -------
    list of Detection
        One per group, strongest peak first.

    Examples
    --------
    >>> power_w = np.zeros(16)
    >>> power_w[[15, 0]] = 4.0
    >>> [d.centroid_index for d in cluster_detections(power_w > 0, power_w, wrap_axes=(0,))]
    [(15.5,)]
    """
    flags = np.asarray(mask, dtype=bool)
    values_w = np.asarray(power_w, dtype=np.float64)
    nan_w = np.full_like(values_w, np.nan)
    at_peak_w = nan_w if threshold_w is None else np.asarray(threshold_w, dtype=np.float64)
    if not flags.shape == values_w.shape == at_peak_w.shape:
        msg = f"mask {flags.shape}, power_w {values_w.shape} and threshold_w do not match."
        raise ValueError(msg)
    labels, n_groups = label_connected_cells(flags, connectivity=connectivity, wrap_axes=wrap_axes)
    if n_groups == 0:
        return []
    index = np.arange(1, n_groups + 1)
    total_w = ndimage.sum_labels(values_w, labels, index)
    n_cells = np.bincount(labels.ravel(), minlength=n_groups + 1)[1:]
    peaks = np.array(ndimage.maximum_position(values_w, labels, index), dtype=np.intp)
    centroids = np.array(ndimage.center_of_mass(values_w, labels, index), dtype=np.float64)
    for axis in {_axis(axis, flags.ndim) for axis in wrap_axes}:
        centroids[:, axis] = _circular_centroid(values_w, labels, peaks[:, axis], axis, total_w)
    found = [
        Detection(
            peak_index=tuple(int(i) for i in peak),
            centroid_index=tuple(float(c) for c in centroid),
            peak_power_w=float(values_w[tuple(peak)]),
            total_power_w=float(total),
            n_cells=int(count),
            cfar_threshold_w=float(at_peak_w[tuple(peak)]),
        )
        for peak, centroid, total, count in zip(peaks, centroids, total_w, n_cells, strict=True)
    ]
    return sorted(found, key=lambda detection: detection.peak_power_w, reverse=True)


def _footprint(
    train: tuple[int, ...], guard: tuple[int, ...]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return the leading and lagging halves of the reference footprint, mirror images.

    The footprint is the box of ``guard + train`` cells per side less the guard box, which
    holds the CUT. In Fortran order the last axis is slowest, so the cells that come after
    the CUT are those beyond it along the last axis, then along the first in its column.
    """
    box = np.ones([2 * (g + t) + 1 for t, g in zip(train, guard, strict=True)], dtype=bool)
    box[tuple(slice(t, t + 2 * g + 1) for t, g in zip(train, guard, strict=True))] = False
    order = np.arange(box.size).reshape(box.shape, order="F")
    return box & (order > box.size // 2), box & (order < box.size // 2)


def _cfar(
    power_w: NDArray[np.float64],
    pfa: float,
    n_train: tuple[int, ...],
    n_guard: tuple[int, ...],
    variant: str,
    rank: int | None,
    axes: tuple[int, ...],
    wraps: tuple[bool, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(threshold_w, noise_estimate_w, alpha_linear)`` of every cell.

    The footprint spans ``axes``, with one ``n_train``, ``n_guard`` and ``wraps`` entry per
    axis. ``alpha_linear`` has the input's ndim and broadcasts against it.
    """
    train, guard = _window(n_train, n_guard)
    if len(train) != len(axes):
        msg = f"n_train and n_guard need one count per window axis ({len(axes)}), got {n_train!r}."
        raise ValueError(msg)
    pfa, variant = _pfa_checked(pfa), _variant(variant)
    margins = [g + t for t, g in zip(train, guard, strict=True)]
    for axis, margin, wrap in zip(axes, margins, wraps, strict=True):
        if wrap and power_w.shape[axis] < 2 * margin + 1:
            msg = f"circular axis {axis} is shorter than its {2 * margin + 1}-cell window."
            raise ValueError(msg)
    leading, lagging = _footprint(train, guard)
    n_reference = int(leading.sum() + lagging.sum())
    rank = _rank(rank, n_reference) if variant == "os" else 0

    # Window axes go last, so a footprint of shape (1, ..., 1, *window) filters every line and
    # the per-cell calibration, which depends only on position along them, broadcasts.
    n_window = len(axes)
    values_w = np.moveaxis(power_w, axes, range(-n_window, 0))
    lead = (np.newaxis,) * (values_w.ndim - n_window)
    # Circular axes gain one margin of wrapped cells, cropped afterwards. Beyond the map every
    # filter reads zeros (mode="constant"), which is what clips the footprint.
    pad = [(0, 0)] * (values_w.ndim - n_window)
    pad += [(m, m) if w else (0, 0) for m, w in zip(margins, wraps, strict=True)]
    padded_w = np.pad(values_w, pad, mode="wrap")
    crop = _crop(pad, values_w.shape)
    ones = np.ones(padded_w.shape[-n_window:])

    def in_map(half: NDArray[np.bool_]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Sum of a half's in-map cells around every cell, and how many there are."""
        sum_w = ndimage.correlate(padded_w, half[lead].astype(float), mode="constant")[crop]
        count = ndimage.correlate(ones, half.astype(float), mode="constant")[crop[-n_window:]]
        return sum_w, np.rint(count).astype(np.int64)

    (lead_w, n_lead), (lag_w, n_lag) = in_map(leading), in_map(lagging)
    n_cells = n_lead + n_lag
    estimate_w = (lead_w + lag_w) / np.maximum(n_cells, 1)
    if variant in ("go", "so"):
        pick = np.maximum if variant == "go" else np.minimum
        halves_w = pick(lead_w / np.maximum(n_lead, 1), lag_w / np.maximum(n_lag, 1))
        estimate_w = np.where(n_cells == n_reference, halves_w, estimate_w)
    elif variant == "os":
        footprint = (leading | lagging)[lead]
        estimate_w = ndimage.rank_filter(padded_w, rank - 1, footprint=footprint, mode="constant")
        estimate_w = estimate_w[crop]

    # A Python loop over the distinct in-map counts, one in the interior and a few at the
    # edges, so that each count is calibrated once rather than once per cell.
    counts, inverse = np.unique(n_cells, return_inverse=True)
    alphas = np.array([_cell_alpha(pfa, int(m), n_reference, variant, rank) for m in counts])
    alpha_linear = alphas[inverse].reshape(n_cells.shape)
    back = (range(-n_window, 0), axes)
    threshold_w = np.moveaxis(alpha_linear * estimate_w, *back)
    return threshold_w, np.moveaxis(estimate_w, *back), np.moveaxis(alpha_linear[lead], *back)


def _cell_alpha(pfa: float, n_cells: int, n_reference: int, variant: str, rank: int) -> float:
    """Threshold factor of a cell that keeps ``n_cells`` of the ``n_reference`` cells."""
    if variant == "os":
        # Missing cells read as zero and sort first, so rank k lands on k - missing.
        rank_in_map = rank - (n_reference - n_cells)
        return _alpha(pfa, n_cells, "os", rank_in_map) if rank_in_map >= 1 else math.nan
    if n_cells == 0:
        return math.nan
    return _alpha(pfa, n_cells, "ca" if n_cells < n_reference else variant, 0)


def _pfa(
    alpha: NDArray[np.float64], n_reference: int, variant: str, rank: int
) -> NDArray[np.float64]:
    """Closed-form Pfa for ``n_reference`` cells (``rank`` for OS only); see the public one."""
    if variant == "ca":
        return np.asarray((1.0 + alpha / n_reference) ** -n_reference, dtype=np.float64)
    if variant == "os":
        upper, lower = n_reference + 1.0, n_reference - rank + 1.0
        log_pfa = special.gammaln(upper) - special.gammaln(lower)
        log_pfa = log_pfa + special.gammaln(lower + alpha) - special.gammaln(upper + alpha)
        return np.asarray(np.exp(log_pfa), dtype=np.float64)
    n_half = n_reference // 2
    beta = alpha / n_half
    x = (1.0 + beta) / (2.0 + beta) if variant == "so" else 1.0 / (2.0 + beta)
    return np.asarray(2.0 * (1.0 + beta) ** -n_half * special.betainc(n_half, n_half, x))


def _alpha(pfa: float, n_reference: int, variant: str, rank: int) -> float:
    """Invert :func:`_pfa` at ``pfa``: CA in closed form, the others by Brent on log alpha."""
    if variant == "ca":
        # expm1 keeps the digits that P^(-1/M) - 1 loses when M is large.
        return float(n_reference * math.expm1(-math.log(pfa) / n_reference))

    def excess(log_alpha: float) -> float:
        pfa_of_alpha = float(_pfa(np.asarray(math.exp(log_alpha)), n_reference, variant, rank))
        return math.log(max(pfa_of_alpha, np.finfo(float).tiny)) - math.log(pfa)

    upper = math.log(_MAX_ALPHA_LINEAR)
    if excess(upper) > 0.0:
        msg = (
            f"no threshold factor below {_MAX_ALPHA_LINEAR:g} reaches pfa={pfa!r} with "
            f"{n_reference} reference cells; widen the window."
        )
        raise ValueError(msg)
    return math.exp(optimize.brentq(excess, -upper, upper, xtol=1e-14))


def _window(
    n_train: int | tuple[int, ...], n_guard: int | tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Check per-axis window counts and return them as tuples of Python ints."""
    try:
        train = tuple(operator.index(t) for t in np.atleast_1d(n_train).tolist())
        guard = tuple(operator.index(g) for g in np.atleast_1d(n_guard).tolist())
    except TypeError:
        train = guard = ()
    if not train or len(train) != len(guard):
        msg = f"n_train {n_train!r} and n_guard {n_guard!r} must be integers, one per axis."
        raise ValueError(msg)
    if min(guard) < 0:
        msg = f"n_guard must be non-negative, got {n_guard!r}."
        raise ValueError(msg)
    if min(train) < 0 or max(train) < 1:
        msg = f"n_train must be at least 1 (zero on all but one axis), got {n_train!r}."
        raise ValueError(msg)
    return train, guard


def _pfa_checked(pfa: float) -> float:
    if not 0.0 < pfa < 1.0:
        msg = f"pfa must be a probability in the open interval (0, 1), got {pfa!r}."
        raise ValueError(msg)
    return float(pfa)


def _variant(variant: str) -> str:
    if variant not in CFAR_VARIANTS:
        msg = f"variant must be one of {CFAR_VARIANTS!r}, got {variant!r}."
        raise ValueError(msg)
    return variant


def _rank(rank: int | None, n_reference: int) -> int:
    """Return the OS rank, defaulting to round(3M/4), checked against ``1 .. M``."""
    k = default_os_rank(n_reference // 2) if rank is None else rank
    if not 1 <= k <= n_reference:
        msg = f"rank must be a one-based index into the {n_reference} reference cells, got {k!r}."
        raise ValueError(msg)
    return k


def _axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        msg = f"axis {axis} is out of bounds for an array of dimension {ndim}."
        raise ValueError(msg)
    return axis % ndim


def _crop(pad: list[tuple[int, int]], shape: tuple[int, ...]) -> tuple[slice, ...]:
    """Index that removes ``np.pad`` widths ``pad`` from a padded array of unpadded ``shape``."""
    return tuple(slice(low, low + n) for (low, _), n in zip(pad, shape, strict=True))


def _power(power_w: ArrayLike, *, min_ndim: int) -> NDArray[np.float64]:
    """Return ``power_w`` as float64, refusing too few dimensions or a negative power."""
    values_w = np.asarray(power_w, dtype=np.float64)
    if values_w.ndim < min_ndim:
        msg = f"power_w must have at least {min_ndim} dimension(s), got {values_w.ndim}."
        raise ValueError(msg)
    if np.any(values_w < 0.0):
        msg = "power_w must be non-negative: pass |x| ** 2, not an amplitude or a dB value."
        raise ValueError(msg)
    return values_w


def _circular_centroid(
    values_w: NDArray[np.float64],
    labels: NDArray[np.int_],
    peaks: NDArray[np.intp],
    axis: int,
    total_w: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Power-weighted centroid of each label along a circular ``axis``, in ``[0, n)``.

    Each cell's index is replaced by its offset from its cluster's peak taken the short way
    round, so the mean is ordinary. Assumes a cluster spans less than half the axis.
    """
    n = values_w.shape[axis]
    index = np.arange(n).reshape([-1 if a == axis else 1 for a in range(values_w.ndim)])
    peak_of_cell = np.concatenate([[0], peaks])[labels]
    offset = (index - peak_of_cell + n // 2) % n - n // 2
    mean_offset = ndimage.sum_labels(values_w * offset, labels, np.arange(1, peaks.size + 1))
    centroid = (peaks + mean_offset / total_w) % n
    # A tiny negative offset can round up to exactly n.
    return np.where(centroid < n, centroid, 0.0)
