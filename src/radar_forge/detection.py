r"""Constant-false-alarm-rate (CFAR) detection, and clustering of what it detects.

Each cell under test is compared with :math:`\alpha` times a noise estimate taken from its
reference footprint, a training box less a guard box, with :math:`\alpha` set so that
independent exponential noise cells cross with probability ``pfa`` [1]_ [6]_. The estimate is
the mean (``"ca"``) [1]_, the greater or smaller half mean (``"go"``, ``"so"``) [2]_ [3]_ [5]_,
or the ``rank``-th smallest cell (``"os"``) [4]_. One private N-D :mod:`scipy.ndimage` engine
serves :func:`cfar_detect` (one axis) and :func:`cfar_detect_2d` (the last two axes).

Edge policy, for every variant and dimensionality: a circular axis wraps, and on any other the
footprint is clipped to the map and each cell is calibrated for the reference cells it keeps.
A tapered, zero-padded or oversampled map correlates neighbouring cells, which breaks that
calibration and raises the false-alarm rate above ``pfa``.

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
.. [7] A. Rosenfeld and J. L. Pfaltz, "Sequential operations in digital picture
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
    centroid_index : tuple of float
        Power-weighted centroid in fractional cell indices; in ``[0, n)`` on a circular axis.
    cfar_threshold_w : float
        Threshold at the peak cell, or ``nan`` if :func:`cluster_detections` was given none.
    """

    peak_index: tuple[int, ...]
    centroid_index: tuple[float, ...]
    peak_power_w: float
    total_power_w: float
    n_cells: int
    cfar_threshold_w: float = math.nan


def default_os_rank(n_train: int) -> int:
    r"""Return Rohling's OS-CFAR rank :math:`k = \mathrm{round}(3M/4)`, ``M = 2 * n_train``.

    References
    ----------
    Rohling [4]_, the recommended :math:`k \approx 3M/4`.
    """
    return max(1, round(1.5 * n_train))


def cfar_reference_cell_count(
    n_train: int | tuple[int, ...], n_guard: int | tuple[int, ...]
) -> int:
    r"""Return :math:`M = \prod_i (2g_i + 2t_i + 1) - \prod_i (2g_i + 1)`, the reference cells.

    Parameters
    ----------
    n_train, n_guard : int or tuple of int
        Cells per side, one count per window axis; a bare int is one axis.

    References
    ----------
    Richards [6]_, §6.5.
    """
    train, guard = _window(n_train, n_guard)
    outer = math.prod(2 * (g + t) + 1 for t, g in zip(train, guard, strict=True))
    return outer - math.prod(2 * g + 1 for g in guard)


def cfar_probability_of_false_alarm(
    alpha_linear: ArrayLike, *, n_train: int, variant: CfarVariant = "ca", rank: int | None = None
) -> NDArray[np.float64]:
    r"""Return the closed-form :math:`P_{fa}` of a CFAR detector for :math:`M = 2N` cells.

    With :math:`\beta = \alpha / N`, CA is :math:`(1 + \alpha/M)^{-M}`, OS is
    :math:`\Gamma(M+1)\Gamma(M-k+1+\alpha) / \Gamma(M-k+1)\Gamma(M+1+\alpha)`, and SO and GO are
    :math:`2(1+\beta)^{-N} I_x(N, N)` with :math:`x = (1+\beta)/(2+\beta)` and
    :math:`1/(2+\beta)`: the negative-binomial series of [5]_ summed as a regularised
    incomplete beta function, so GO is no longer a difference of near-equal terms.

    Parameters
    ----------
    n_train : int
        :math:`N`; for an N-D footprint pass ``cfar_reference_cell_count(...) // 2``.
    rank : int, optional
        OS rank :math:`k` in ``1 .. 2N``, default :func:`default_os_rank`; ignored otherwise.

    References
    ----------
    Richards [6]_, §6.5 (CA); Rohling [4]_, eq. 14 (OS); Gandhi and Kassam [5]_ (GO, SO).
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
    r"""Return the factor :math:`\alpha` that inverts :func:`cfar_probability_of_false_alarm`.

    CA is closed form, :math:`\alpha = M (P_{fa}^{-1/M} - 1)`; GO, SO and OS are found by
    :func:`scipy.optimize.brentq` on :math:`\ln \alpha`.

    References
    ----------
    Richards [6]_, §6.5.

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
    """Detect the cells above a CFAR threshold from a window of training cells along one axis.

    Parameters
    ----------
    power_w : array_like
        Square-law power, non-negative, e.g. ``np.abs(rd_map) ** 2``.
    n_train, n_guard : int
        Cells per side; set ``n_guard`` to cover a target's spread.

    Returns
    -------
    tuple of numpy.ndarray
        ``(detected, threshold_w)``, each the shape of ``power_w``; ``nan`` where untested.

    References
    ----------
    Richards [6]_, §6.5.
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
    """Detect the cells above a CFAR threshold from a ring over the last two axes.

    GO and SO split the ring into the cells beyond the cell under test along the last axis,
    plus those beyond it along the first in its own column, and the mirror image of that.

    Parameters
    ----------
    power_w : array_like
        Square-law power, at least 2-D, e.g. ``(n_doppler, n_range)``.
    n_train, n_guard : tuple of int
        Cells per side for the last two axes. Both zero on one axis gives :func:`cfar_detect`
        along the other; ``n_train`` zero with ``n_guard`` non-zero is refused.
    wrap_axes : tuple of int, optional
        Circular axes among the last two, counted from the end: ``(-2,)`` for Doppler.

    Returns
    -------
    tuple of numpy.ndarray
        ``(detected, threshold_w)``, each the shape of ``power_w``; ``nan`` where untested.

    References
    ----------
    Richards [6]_, §6.5.
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
    """Return ``(labels, n_groups)``: each connected group of True cells labelled 1, 2, ...

    :func:`scipy.ndimage.label` has no periodic boundary, so a circular axis is padded with one
    wrapped cell at each end and each copy's label is joined to its original's.

    Parameters
    ----------
    connectivity : int, optional
        ``1`` (default) joins face neighbours only; ``mask.ndim`` joins corners too.

    References
    ----------
    Rosenfeld and Pfaltz [7]_.
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
    """Return one :class:`Detection` per connected group of crossings, strongest peak first.

    Touching targets are one cluster: there is no non-maximum suppression.

    Parameters
    ----------
    wrap_axes : tuple of int, optional
        Circular axes, counted from the end: ``(-2,)`` for Doppler, or a target on the
        velocity wrap is two.

    References
    ----------
    Rosenfeld and Pfaltz [7]_ (labelling).

    Examples
    --------
    >>> power_w = np.zeros(16)
    >>> power_w[[15, 0]] = 4.0
    >>> [d.centroid_index for d in cluster_detections(power_w > 0, power_w, wrap_axes=(-1,))]
    [(15.5,)]
    """
    flags = np.asarray(mask, dtype=bool)
    values_w = np.asarray(power_w, dtype=np.float64)
    at_peak_w = None if threshold_w is None else np.asarray(threshold_w, dtype=np.float64)
    shapes = {flags.shape, values_w.shape} | ({at_peak_w.shape} if at_peak_w is not None else set())
    if len(shapes) > 1:
        msg = f"mask {flags.shape}, power_w {values_w.shape} and threshold_w do not match."
        raise ValueError(msg)
    labels, n_groups = label_connected_cells(flags, connectivity=connectivity, wrap_axes=wrap_axes)
    if n_groups == 0:
        return []
    index = np.arange(1, n_groups + 1)
    total_w = ndimage.sum_labels(values_w, labels, index)
    n_cells = np.bincount(labels.ravel(), minlength=n_groups + 1)[1:]
    peaks = np.array(ndimage.maximum_position(values_w, labels, index), dtype=np.intp)
    # errstate-exempt: a cluster with no power has no weighted centroid, and nan says so.
    with np.errstate(invalid="ignore"):
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
            cfar_threshold_w=math.nan if at_peak_w is None else float(at_peak_w[tuple(peak)]),
        )
        for peak, centroid, total, count in zip(peaks, centroids, total_w, n_cells, strict=True)
    ]
    return sorted(found, key=lambda detection: detection.peak_power_w, reverse=True)


def _footprint(
    train: tuple[int, ...], guard: tuple[int, ...]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return the leading and lagging halves of the footprint (training box less guard box).

    The leading half is the cells after the cell under test in Fortran order, where the last
    axis is slowest; the lagging half is its mirror image. Each holds M/2 cells.
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
    """Return ``(threshold_w, noise_estimate_w, alpha_linear)``; ``alpha_linear`` broadcasts.

    The footprint spans ``axes``, with one ``n_train``, ``n_guard`` and ``wraps`` entry each.
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
    footprint = leading | lagging
    n_reference = int(footprint.sum())
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
    padded_w = np.pad(values_w, pad, mode="wrap") if any(wraps) else values_w
    crop = _crop(pad, values_w.shape)
    ones = np.ones(padded_w.shape[-n_window:])

    def in_map_sum_w(cells: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Sum of the in-map ``cells`` around every cell under test, in watts."""
        sum_w: NDArray[np.float64] = ndimage.correlate(
            padded_w, cells[lead].astype(float), mode="constant"
        )
        return sum_w[crop]

    def in_map_count(cells: NDArray[np.bool_]) -> NDArray[np.int64]:
        """How many of ``cells`` lie in the map, per position along the window axes."""
        count = ndimage.correlate(ones, cells.astype(float), mode="constant")[crop[-n_window:]]
        return np.asarray(np.rint(count), dtype=np.int64)

    n_cells = in_map_count(footprint)
    if variant == "os":
        rank_w = ndimage.rank_filter(padded_w, rank - 1, footprint=footprint[lead], mode="constant")
        estimate_w = rank_w[crop]
    elif variant == "ca":
        estimate_w = in_map_sum_w(footprint) / np.maximum(n_cells, 1)
    else:
        lead_w, lag_w = in_map_sum_w(leading), in_map_sum_w(lagging)
        n_lead, n_lag = in_map_count(leading), in_map_count(lagging)
        pick = np.maximum if variant == "go" else np.minimum
        halves_w = pick(lead_w / np.maximum(n_lead, 1), lag_w / np.maximum(n_lag, 1))
        mean_w = (lead_w + lag_w) / np.maximum(n_cells, 1)
        estimate_w = np.where(n_cells == n_reference, halves_w, mean_w)

    # A Python loop over the distinct in-map counts, one in the interior and a few at the
    # edges, so that each count is calibrated once rather than once per cell.
    counts, inverse = np.unique(n_cells, return_inverse=True)
    alphas = np.array([_cell_alpha(pfa, int(m), n_reference, variant, rank) for m in counts])
    alpha_linear = alphas[inverse].reshape(n_cells.shape)
    back = (range(-n_window, 0), axes)
    threshold_w = np.moveaxis(alpha_linear * estimate_w, *back)
    return threshold_w, np.moveaxis(estimate_w, *back), np.moveaxis(alpha_linear[lead], *back)


def _cell_alpha(pfa: float, n_cells: int, n_reference: int, variant: str, rank: int) -> float:
    """Return the threshold factor of a cell that keeps ``n_cells`` of ``n_reference`` cells.

    CA uses its ``n_cells``. GO and SO need two equal halves, so where clipped they fall back
    to CA. OS reads the missing cells as zero, which sorts first, so rank k becomes
    k - missing; below 1 the cell is not tested (``nan``).
    """
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
    if any(t == 0 and g > 0 for t, g in zip(train, guard, strict=True)):
        # A guard with no training beside it would widen the ring, not remove the axis.
        msg = f"n_guard must be zero on an axis where n_train is, got {n_train!r}, {n_guard!r}."
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
    if isinstance(k, bool) or not isinstance(k, int | np.integer):
        msg = f"rank must be an integer, got {k!r}."
        raise ValueError(msg)
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
    """Return ``power_w`` as float64: the one data scan, at the public boundary.

    A nan or inf would spread into every window that holds it; a negative value means an
    amplitude or a dB value was passed.
    """
    values_w = np.asarray(power_w, dtype=np.float64)
    if values_w.ndim < min_ndim:
        msg = f"power_w must have at least {min_ndim} dimension(s), got {values_w.ndim}."
        raise ValueError(msg)
    if not np.all(np.isfinite(values_w) & (values_w >= 0.0)):
        msg = "power_w must be finite and non-negative: pass |x| ** 2, not an amplitude or dB."
        raise ValueError(msg)
    return values_w


def _circular_centroid(
    values_w: NDArray[np.float64],
    labels: NDArray[np.int_],
    peaks: NDArray[np.intp],
    axis: int,
    total_w: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return each label's power-weighted centroid along a circular ``axis``, in ``[0, n)``.

    Offsets from the peak are taken the short way round, so a cluster must span under half.
    """
    n = values_w.shape[axis]
    index = np.arange(n).reshape([-1 if a == axis else 1 for a in range(values_w.ndim)])
    peak_of_cell = np.concatenate([[0], peaks])[labels]
    offset = (index - peak_of_cell + n // 2) % n - n // 2
    moment_w = ndimage.sum_labels(values_w * offset, labels, np.arange(1, peaks.size + 1))
    # A cluster with no power has no weighted centroid: nan, as center_of_mass gives.
    no_power = np.full(peaks.size, np.nan)
    mean_offset = np.divide(moment_w, total_w, out=no_power, where=total_w > 0.0)
    centroid = (peaks + mean_offset) % n
    # A tiny negative offset can round up to exactly n.
    return np.where(centroid >= n, 0.0, centroid)
