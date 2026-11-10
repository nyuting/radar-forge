r"""Leakage and near-clutter cancellation: least squares on delayed copies of the radar's code.

A continuous-wave radar hears its own transmitter and knows what it sent, so its leakage and
near clutter, delayed and scaled copies of the code, can be fitted and subtracted before
correlation: Widrow's two-input canceller [1]_ with the transmit code as reference input.
Every function takes ``reference`` (the digital code :math:`x`, noise-free) before
``surveillance`` (the received :math:`s`), each one CPI ``(n_samples,)`` of a circular stream
(:func:`~radar_forge.signal.stream_baseband`): tap :math:`k` reads :math:`x[(n - k) \bmod N]`.
The basis is ``n_taps`` delays of each odd order :math:`|x|^{p-1}x`, order-major, the
parallel-Hammerstein model [4]_; ``odd_orders=(1,)`` is the linear canceller. The fit is a
projection, :math:`y = (I - X(X^HX)^{-1}X^H)s` [2]_, one per block of ``n_samples_per_batch``
(0: the CPI) estimated on the block widened by ``n_samples_extension`` a side, clipped to the
CPI: one block is the full-CPI fit, blocks the batched one, a widening the sliding one [5]_.
Weights are returned, never applied silently.

References
----------
.. [1] B. Widrow, J. R. Glover, J. M. McCool, J. Kaunitz, C. S. Williams, R. H. Hearn et al.,
       "Adaptive noise cancelling: principles and applications," *Proc. IEEE*, vol. 63, no. 12,
       pp. 1692-1716, 1975, §III.
.. [2] S. Haykin, *Adaptive Filter Theory*, 5th ed., Pearson, 2014.
.. [3] K. Kulpa, *Signal Processing in Noise Waveform Radar*, Artech House, 2013.
.. [4] L. Anttila, D. Korpi, V. Syrjälä and M. Valkama, "Cancellation of power amplifier
       induced nonlinear self-interference in full-duplex transceivers," *Proc. Asilomar Conf.
       Signals, Systems and Computers*, pp. 1193-1198, 2013 (arXiv:1401.3129).
.. [5] F. Colone, C. Palmarini, T. Martelli and E. Tilli, "Sliding extensive cancellation
       algorithm for disturbance removal in passive radar," *IEEE Trans. Aerosp. Electron.
       Syst.*, vol. 52, no. 3, pp. 1309-1326, 2016.
.. [6] J. J. Bussgang, "Crosscorrelation functions of amplitude-distorted Gaussian signals,"
       MIT Research Laboratory of Electronics, Tech. Rep. 216, 1952.
.. [7] N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, lapack
from scipy.signal import fftconvolve

if TYPE_CHECKING:
    from radar_forge.settings import CancellationSettings

# Lag sums run over units of at most this many taps' length: an FFT a few K long, not the CPI.
_UNIT_TAPS = 16
# Windows whose Grams are assembled at once: a chunk of at most this many bytes of Gram.
_MAX_BYTES = 16 * 2**20

__all__ = [
    "CancellationResult",
    "cancel",
    "cancellation_depth_db",
    "delay_basis_gram",
    "distortion_fraction_linear",
    "doppler_notch_response_linear",
    "expected_cancellation_depth_db",
    "fill_notch",
    "least_squares_cancel",
    "ls_cancel",
    "ls_cancel_batched",
    "ls_cancel_sliding",
    "ls_weights",
    "parallel_hammerstein_basis",
    "wiener_smi",
]


@dataclass(frozen=True)
class CancellationResult:
    """A canceller's output: the residual stream, and each block's weights and Gram condition.

    ``weights`` is ``(n_blocks, n_basis)``, order-major as the basis; ``gram_condition_number``
    ``(n_blocks,)`` is the 1-norm condition of each block's (loaded) Gram matrix.
    """

    residual: NDArray[np.complex128]
    weights: NDArray[np.complex128]
    gram_condition_number: NDArray[np.float64]


def parallel_hammerstein_basis(
    reference: ArrayLike,
    n_taps: int,
    odd_orders: Sequence[int] = (1,),
    start: int = 0,
    stop: int | None = None,
) -> NDArray[np.complex128]:
    r"""Return the basis rows ``start`` to ``stop``, ``(stop - start, n_taps * len(odd_orders))``.

    Column :math:`pK + k` is order ``odd_orders[p]`` delayed :math:`k` samples, circularly:
    :math:`|x[n-k]|^{o-1} x[n-k]` (Anttila [4]_ eq. (1)). Explicit, so for a short window: the
    cancellers never build it.
    """
    bases = _odd_order_bases(_stream(reference), odd_orders)
    n_samples = bases.shape[1]
    stop = n_samples if stop is None else stop
    _check_taps(n_taps, n_samples)
    rows = (start + np.arange(stop - start)[:, None] - np.arange(n_taps)[None, :]) % n_samples
    return np.concatenate([basis[rows] for basis in bases], axis=1)


def delay_basis_gram(
    reference: ArrayLike,
    surveillance: ArrayLike,
    n_taps: int,
    *,
    start: int,
    stop: int,
    odd_orders: Sequence[int] = (1,),
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    r"""Return the normal equations of rows ``[start, stop)``: ``(gram (P, P), cross (P,))``.

    :math:`X^H X` and :math:`X^H s` (Haykin [2]_ §9.5), without forming :math:`X`. Each block
    of the Gram, linear and Hammerstein alike, is a correlation of two basis sequences by FFT
    for its first row and column, plus the displacement
    :math:`G[i, j] = G[i-1, j-1] + \bar z_p[a-i] z_q[a-j] - \bar z_p[b-i] z_q[b-j]`, the
    rows entering and leaving the window: O(N log N + P^2), not O(N P^2).
    """
    bases = _odd_order_bases(_stream(reference), odd_orders)
    signal = _stream(surveillance)
    n_samples = bases.shape[1]
    _check_taps(n_taps, n_samples)
    if not 0 <= start < stop <= n_samples:
        msg = f"need 0 <= start < stop <= {n_samples}; got start={start}, stop={stop}."
        raise ValueError(msg)
    late = np.stack([_lagged_window(basis, n_taps, start, stop) for basis in bases])
    columns = _correlate(late[:, None], bases[None, :, start:stop])
    cross = _correlate(late, np.broadcast_to(signal[start:stop], bases[:, start:stop].shape))
    gram = _assemble(columns[None], bases, np.array([start]), np.array([stop]))[0]
    return gram, cross.reshape(-1)


def ls_weights(
    gram: ArrayLike, cross: ArrayLike, *, diagonal_loading: float = 0.0
) -> tuple[NDArray[np.complex128], float]:
    r"""Solve :math:`(G + \delta \bar g I) w = c` by Cholesky; return ``(w, condition number)``.

    ``diagonal_loading`` :math:`\delta` is a fraction of the mean diagonal :math:`\bar g`, so it
    does not depend on the record's length or power; 0 solves the normal equations as they are
    (Haykin [2]_ §9.5). The condition number is LAPACK's 1-norm estimate from the Cholesky
    factor, ``1 / ?pocon`` (Higham [7]_ ch. 15): O(P^2), where an exact one costs O(P^3).
    """
    matrix = np.array(gram, dtype=np.complex128)
    rhs = np.asarray(cross, dtype=np.complex128)
    size = matrix.shape[0]
    if matrix.shape != (size, size) or rhs.shape != (size,) or diagonal_loading < 0.0:
        msg = f"need a square gram, a matching cross, loading >= 0; got {matrix.shape}."
        raise ValueError(msg)
    matrix[np.diag_indices(size)] += diagonal_loading * np.mean(np.diag(matrix).real)
    factor = cho_factor(matrix, lower=False, check_finite=False)
    reciprocal, _ = lapack.zpocon(factor[0], np.max(np.sum(np.abs(matrix), axis=0)))
    weights: NDArray[np.complex128] = cho_solve(factor, rhs, check_finite=False)
    return weights, float(1.0 / reciprocal)


def least_squares_cancel(
    reference: ArrayLike,
    surveillance: ArrayLike,
    n_taps: int,
    *,
    n_samples_per_batch: int = 0,
    n_samples_extension: int = 0,
    odd_orders: Sequence[int] = (1,),
    diagonal_loading: float = 0.0,
) -> CancellationResult:
    r"""Project the stream off the delay basis, block by block: full CPI, batched or sliding.

    Block :math:`b` is filtered with the weights fitted on its window (Colone [5]_ eqs. (3)-(4)
    batched, (18)-(19) sliding; Haykin [2]_ §9.7): :math:`X_b^H y_b = 0` exactly when the
    window is the block. ``n_samples_extension`` must be whole blocks. A full-CPI fit notches
    one Doppler bin, a block of :math:`T_B` widens it to :math:`\pm 1/T_B` at the price of
    lines at multiples of :math:`1/T_B`, and the widening moves those to :math:`1/T_f` while
    keeping the wide notch (Colone [5]_ §III, §IV).
    """
    bases = _odd_order_bases(_stream(reference), odd_orders)
    signal = _stream(surveillance)
    n_samples = bases.shape[1]
    _check_taps(n_taps, n_samples)
    n_block = n_samples_per_batch or n_samples
    if n_samples % n_block or n_samples_extension % n_block or n_samples_extension < 0:
        msg = "n_samples_per_batch must divide the stream, and n_samples_extension be whole blocks."
        raise ValueError(msg)
    n_blocks, reach = n_samples // n_block, n_samples_extension // n_block
    lower = np.maximum(np.arange(n_blocks) - reach, 0)
    upper = np.minimum(np.arange(n_blocks) + 1 + reach, n_blocks)
    if np.min(upper - lower) * n_block < n_taps * bases.shape[0]:
        msg = "every fitting window needs at least as many samples as basis columns."
        raise ValueError(msg)
    n_unit = _unit_length(n_block, n_taps)
    units_per_block = n_block // n_unit
    columns, cross = _lag_sums(bases, signal, n_taps, n_unit)
    lower, upper = lower * units_per_block, upper * units_per_block
    start, stop = lower * n_unit, upper * n_unit
    # Window sums as differences of running sums over units: one cumsum serves every window.
    running_columns = np.cumsum(np.concatenate([0.0 * columns[:, :, :1], columns], axis=2), axis=2)
    running_cross = np.cumsum(np.concatenate([0.0 * cross[:, :1], cross], axis=1), axis=1)
    window_columns = np.moveaxis(running_columns[:, :, upper] - running_columns[:, :, lower], 2, 0)
    window_cross = np.moveaxis(running_cross[:, upper] - running_cross[:, lower], 1, 0)
    weights = np.empty((n_blocks, bases.shape[0] * n_taps), dtype=np.complex128)
    condition = np.empty(n_blocks)
    per_chunk = max(1, _MAX_BYTES // (2 * n_taps**2 * bases.shape[0] ** 2 * 16))
    # Chunks of windows bound the Gram stack; within one, each block's fit is an independent
    # solve on its own window: that loop is the algorithm.
    for first in range(0, n_blocks, per_chunk):
        chunk = slice(first, first + per_chunk)
        grams = _assemble(window_columns[chunk], bases, start[chunk], stop[chunk])
        for offset, gram in enumerate(grams):
            block = first + offset
            weights[block], condition[block] = ls_weights(
                gram, window_cross[block].reshape(-1), diagonal_loading=diagonal_loading
            )
    residual = signal - _fitted(bases, weights, n_block, n_unit)
    return CancellationResult(residual, weights, condition)


def ls_cancel(
    reference: ArrayLike,
    surveillance: ArrayLike,
    n_taps: int,
    *,
    odd_orders: Sequence[int] = (1,),
    diagonal_loading: float = 0.0,
) -> CancellationResult:
    """Fit the whole CPI at once: :func:`least_squares_cancel` with one block."""
    return least_squares_cancel(
        reference, surveillance, n_taps, odd_orders=odd_orders, diagonal_loading=diagonal_loading
    )


def ls_cancel_batched(
    reference: ArrayLike,
    surveillance: ArrayLike,
    n_taps: int,
    n_samples_per_batch: int,
    *,
    odd_orders: Sequence[int] = (1,),
    diagonal_loading: float = 0.0,
) -> CancellationResult:
    """Fit each batch on itself (Colone [5]_ eqs. (3)-(4)): :func:`least_squares_cancel`."""
    return least_squares_cancel(
        reference,
        surveillance,
        n_taps,
        n_samples_per_batch=n_samples_per_batch,
        odd_orders=odd_orders,
        diagonal_loading=diagonal_loading,
    )


def ls_cancel_sliding(
    reference: ArrayLike,
    surveillance: ArrayLike,
    n_taps: int,
    n_samples_per_block: int,
    n_samples_extension: int,
    *,
    odd_orders: Sequence[int] = (1,),
    diagonal_loading: float = 0.0,
) -> CancellationResult:
    """Filter each block with weights from its widened window (Colone [5]_ eqs. (18)-(19))."""
    return least_squares_cancel(
        reference,
        surveillance,
        n_taps,
        n_samples_per_batch=n_samples_per_block,
        n_samples_extension=n_samples_extension,
        odd_orders=odd_orders,
        diagonal_loading=diagonal_loading,
    )


def wiener_smi(reference: ArrayLike, surveillance: ArrayLike, n_taps: int) -> CancellationResult:
    r"""Solve Wiener-Hopf on the biased sample statistics of the CPI, by FFT (Haykin [2]_ §2.4).

    :math:`\hat R[k] = \frac{1}{N}\sum_n x[n]\bar x[n-k]` and
    :math:`\hat r[k] = \frac{1}{N}\sum_n s[n]\bar x[n-k]`, with the record taken as zero
    outside the CPI: the autocorrelation method of Haykin [2]_ §9.2, a Toeplitz :math:`\hat R`.
    The full-CPI least-squares fit uses the stream itself there, so the two differ by edge
    terms of order :math:`K/N`. The weights filter the stream as :func:`ls_cancel`'s do.
    """
    code, signal = _stream(reference), _stream(surveillance)
    n_samples = code.size
    _check_taps(n_taps, n_samples)
    template = np.conjugate(code[::-1])
    lags = slice(n_samples - 1, n_samples - 1 + n_taps)
    autocorrelation = fftconvolve(code, template)[lags] / n_samples
    cross = fftconvolve(signal, template)[lags] / n_samples
    offset = np.arange(n_taps)[:, None] - np.arange(n_taps)[None, :]
    gram = np.where(
        offset >= 0, autocorrelation[np.abs(offset)], np.conj(autocorrelation[np.abs(offset)])
    )
    weights, condition = ls_weights(gram, cross)
    bases = code[None, :]
    residual = signal - _fitted(bases, weights[None, :], n_samples, _unit_length(n_samples, n_taps))
    return CancellationResult(residual, weights[None, :], np.array([condition]))


def cancel(
    reference: ArrayLike, surveillance: ArrayLike, settings: CancellationSettings | None
) -> CancellationResult:
    """Run a ``[cancellation]`` table's canceller on one stream; none (or no table) passes it.

    The one entry point :mod:`radar_forge.pipeline` calls: ``least_squares`` is
    :func:`least_squares_cancel`, ``wiener`` is :func:`wiener_smi`.
    """
    stream = np.asarray(surveillance, dtype=np.complex128).reshape(-1)
    if settings is None or settings.method == "none":
        return CancellationResult(stream, np.zeros((0, 0), dtype=np.complex128), np.zeros(0))
    code = np.asarray(reference, dtype=np.complex128).reshape(-1)
    if settings.method == "wiener":
        return wiener_smi(code, stream, settings.n_taps)
    return least_squares_cancel(
        code,
        stream,
        settings.n_taps,
        n_samples_per_batch=settings.n_samples_per_batch,
        n_samples_extension=settings.n_samples_extension,
        odd_orders=settings.odd_orders,
        diagonal_loading=settings.diagonal_loading,
    )


def fill_notch(rd_map: ArrayLike, n_taps: int) -> NDArray[np.complex128]:
    r"""Return a cancelled map with its notch, zero Doppler below lag K, at the map's noise level.

    A projection empties those cells exactly (:math:`X^H y = 0` summed over the CPI), so CFAR
    there would weigh round-off against round-off and its crossings would depend on the
    platform. They are set to the map's median power over ln 2, an exponential cell's mean,
    which neither crosses a threshold nor pulls down a neighbour's. Rows are Doppler, shifted.
    """
    filled = np.array(rd_map, dtype=np.complex128)
    noise_w = float(np.median(np.abs(filled) ** 2)) / math.log(2.0)
    filled[filled.shape[0] // 2, :n_taps] = math.sqrt(noise_w)
    return filled


def cancellation_depth_db(surveillance: ArrayLike, residual: ArrayLike) -> float:
    r"""Return the cancellation :math:`\mathrm{CA} = \sum|s|^2 / \sum|y|^2` in dB.

    A residual of zero power is cancelled completely: ``inf``. A stream of zero power has nothing
    to cancel, and is refused.
    """
    power_in = float(np.sum(np.abs(np.asarray(surveillance)) ** 2))
    power_out = float(np.sum(np.abs(np.asarray(residual)) ** 2))
    if power_in == 0.0:
        msg = "the surveillance stream has zero power: there is no cancellation to measure."
        raise ValueError(msg)
    if power_out == 0.0:
        return math.inf
    return 10.0 * math.log10(power_in / power_out)


def expected_cancellation_depth_db(
    inr_linear: float, n_basis: int, n_samples: int, inr_uncancelled_linear: float = 0.0
) -> float:
    r"""Return :math:`E[\mathrm{CA}] = (1 + \mathrm{INR})/[(1 - P/N)(1 + \mathrm{INR_{unc}})]`, dB.

    ``inr_linear`` is the disturbance over noise per sample; ``n_samples`` the fit's length.
    The projection takes :math:`P/N` of the white noise with the disturbance it spans, and
    leaves ``inr_uncancelled_linear``, the part outside the span: echoes past the last tap,
    a Doppler component's escape from the notch, an amplifier's distortion. With none the
    residual is the noise itself, the ceiling (Haykin [2]_ §9.8).

    Examples
    --------
    >>> round(expected_cancellation_depth_db(12430.13, 192, 819200), 3)  # scenario 005, P1
    40.946
    """
    if not 0 < n_basis < n_samples or inr_linear < 0.0 or inr_uncancelled_linear < 0.0:
        msg = "need 0 < n_basis < n_samples and non-negative interference ratios."
        raise ValueError(msg)
    kept = (1.0 - n_basis / n_samples) * (1.0 + inr_uncancelled_linear)
    return 10.0 * math.log10((1.0 + inr_linear) / kept)


def doppler_notch_response_linear(
    doppler_hz: ArrayLike, n_samples_cancellation: int, sample_rate_hz: float
) -> NDArray[np.float64]:
    r"""Return the map-peak power :math:`\rho(f) = (1 - |D(f)|^2)^2` a lag-0 echo keeps.

    :math:`D(f) = \sin(\pi f N_c/f_s) / (N_c \sin(\pi f/f_s))`, the Dirichlet kernel of the fit
    length :math:`N_c`: the fit removes :math:`D` of a Doppler-:math:`f` echo's amplitude, and
    correlation projects the rest onto the echo again (Colone [5]_ §III, eq. (16)). Nulls of
    :math:`D` at :math:`k f_s/N_c` make the notch one Doppler bin for the full CPI and
    :math:`\pm 1/T_B` batched; the residual power is :math:`1 - |D|^2`, :math:`\sqrt\rho`.

    Examples
    --------
    >>> round(10 * math.log10(doppler_notch_response_linear(78.125, 51200, 8e6)), 2)
    -4.51
    """
    ratio = np.asarray(doppler_hz, dtype=np.float64) / sample_rate_hz
    dirichlet = np.sinc(ratio * n_samples_cancellation) / np.sinc(ratio)
    result: NDArray[np.float64] = (1.0 - dirichlet**2) ** 2
    return result


def distortion_fraction_linear(a3: float) -> float:
    r"""Return the part of :func:`~radar_forge.signal.cubic_amplifier` no linear basis reaches.

    By Bussgang [6]_ §III eq. (20), a memoryless distortion of a Gaussian :math:`x` is
    :math:`\beta x` plus a part uncorrelated with :math:`x`;
    :math:`\beta = E[f(x)\bar x] = (1 + 2a_3)/\sqrt{1 + 4a_3 + 6a_3^2}`, so the rest is
    :math:`1 - \beta^2 = 2a_3^2/(1 + 4a_3 + 6a_3^2)` of the power.

    Examples
    --------
    >>> round(10 * math.log10(distortion_fraction_linear(-0.07)), 3)
    -18.835
    """
    return 2.0 * a3**2 / (1.0 + 4.0 * a3 + 6.0 * a3**2)


def _stream(samples: ArrayLike) -> NDArray[np.complex128]:
    """Return one stream as a complex vector, or raise."""
    stream = np.asarray(samples, dtype=np.complex128)
    if stream.ndim != 1 or stream.size == 0:
        msg = f"a stream is one non-empty vector, (n_samples,); got shape {stream.shape}."
        raise ValueError(msg)
    return stream


def _check_taps(n_taps: int, n_samples: int) -> None:
    """Reject a tap count the stream cannot hold."""
    if not 1 <= n_taps <= n_samples:
        msg = f"n_taps must lie in [1, {n_samples}]; got {n_taps}."
        raise ValueError(msg)


def _odd_order_bases(
    reference: NDArray[np.complex128], odd_orders: Sequence[int]
) -> NDArray[np.complex128]:
    """Return each odd order's sequence :math:`|x|^{o-1}x`, ``(len(odd_orders), n_samples)``."""
    orders = tuple(odd_orders)
    if not orders or len(set(orders)) != len(orders) or any(o < 1 or o % 2 == 0 for o in orders):
        msg = f"odd_orders must be distinct odd positive orders, such as (1, 3); got {orders}."
        raise ValueError(msg)
    magnitude = np.abs(reference)
    return np.stack([magnitude ** (order - 1) * reference for order in orders])


def _lagged_window(
    basis: NDArray[np.complex128], n_taps: int, start: int, stop: int
) -> NDArray[np.complex128]:
    """Return ``basis[start - n_taps + 1 : stop]``, circularly: every tap's reach into a window."""
    index = np.arange(start - n_taps + 1, stop) % basis.size
    return basis[index]


def _lagged_blocks(
    basis: NDArray[np.complex128], n_taps: int, n_block: int
) -> NDArray[np.complex128]:
    """Return each block with the ``n_taps - 1`` samples before it, ``(n_blocks, n_block+K-1)``."""
    extended = np.concatenate([basis[basis.size - n_taps + 1 :], basis])
    return np.lib.stride_tricks.sliding_window_view(extended, n_block + n_taps - 1)[::n_block]


def _correlate(
    late: NDArray[np.complex128], early: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    r"""Return :math:`\sum_n \bar l[n - k]\, e[n]`, :math:`k = 0 \dots K-1`, on the last axis."""
    n_taps = late.shape[-1] - early.shape[-1] + 1
    forward = fftconvolve(late, np.conjugate(early[..., ::-1]), mode="valid", axes=-1)
    result: NDArray[np.complex128] = np.conjugate(forward[..., ::-1])
    return result[..., :n_taps]


def _unit_length(n_block: int, n_taps: int) -> int:
    """Return the longest divisor of the block up to ``_UNIT_TAPS`` taps: the FFT's stretch."""
    return next(d for d in range(min(n_block, _UNIT_TAPS * n_taps), 0, -1) if n_block % d == 0)


def _lag_sums(
    bases: NDArray[np.complex128], signal: NDArray[np.complex128], n_taps: int, n_unit: int
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Return each unit's first Gram columns ``(Q, Q, n_units, K)``, cross ``(Q, n_units, K)``."""
    n_orders, n_samples = bases.shape
    n_units = n_samples // n_unit
    late = [_lagged_blocks(basis, n_taps, n_unit) for basis in bases]
    early = bases.reshape(n_orders, n_units, n_unit)
    target = signal.reshape(n_units, n_unit)
    columns = np.empty((n_orders, n_orders, n_units, n_taps), dtype=np.complex128)
    cross = np.empty((n_orders, n_units, n_taps), dtype=np.complex128)
    per_chunk = max(1, _MAX_BYTES // (64 * n_orders * (n_orders + 1) * (2 * n_unit + n_taps)))
    # Chunks of units bound the FFT buffers; the count depends only on the shapes.
    for first in range(0, n_units, per_chunk):
        chunk = slice(first, first + per_chunk)
        lagged = np.stack([view[chunk] for view in late])
        columns[:, :, chunk] = _correlate(lagged[:, None], early[None, :, chunk])
        cross[:, chunk] = _correlate(lagged, np.broadcast_to(target[chunk], early[:, chunk].shape))
    return columns, cross


def _assemble(
    columns: NDArray[np.complex128],
    bases: NDArray[np.complex128],
    start: NDArray[np.int_],
    stop: NDArray[np.int_],
) -> NDArray[np.complex128]:
    r"""Return each window's Gram ``(n_windows, P, P)`` from first-column sums ``(.., Q, Q, K)``.

    Rows ``[start, stop)`` per window. Unrolled, the displacement recursion gives entry
    :math:`(d + j, j)` as the sum :math:`A[d]` plus the running sum over :math:`r \le j` of
    :math:`\bar h_p[d + r] h_q[r] - \bar t_p[d + r] t_q[r]`, :math:`h` the rows before
    ``start`` and :math:`t` before ``stop``. Laid out by ``(d, j)``, that is a running sum and
    one shear per order pair; the upper triangle is the Hermitian image of the swapped pair.
    """
    n_orders, n_taps, n_samples = bases.shape[0], columns.shape[-1], bases.shape[1]
    reach = np.arange(n_taps)
    # Index 0 of the edge rows is zero: the recursion starts from the first row and column.
    head = np.where(reach > 0, bases[:, (start[:, None] - reach) % n_samples], 0.0)
    tail = np.where(reach > 0, bases[:, (stop[:, None] - reach) % n_samples], 0.0)
    # Order pairs: one for the linear basis, four for orders 1 and 3.
    below = [
        [_shear(columns[:, p, q, :, None] + _edge_sums(head, tail, p, q)) for q in range(n_orders)]
        for p in range(n_orders)
    ]
    lower = np.tri(n_taps, dtype=bool)
    blocks = [
        [
            np.where(lower, below[p][q], np.conj(np.swapaxes(below[q][p], 1, 2)))
            for q in range(n_orders)
        ]
        for p in range(n_orders)
    ]
    return np.block(blocks)


def _edge_sums(
    head: NDArray[np.complex128], tail: NDArray[np.complex128], p: int, q: int
) -> NDArray[np.complex128]:
    r"""Return :math:`\sum_{r \le j} \bar h_p[d+r] h_q[r] - \bar t_p[d+r] t_q[r]`, ``[w, d, j]``."""
    n_taps = head.shape[-1]

    def hankel(rows: NDArray[np.complex128]) -> NDArray[np.complex128]:
        padded = np.concatenate([rows, np.zeros_like(rows)], axis=-1)
        view = np.lib.stride_tricks.sliding_window_view(padded, n_taps, axis=-1)
        result: NDArray[np.complex128] = view[:, :n_taps]
        return result

    terms = np.conj(hankel(head[p])) * head[q][:, None, :]
    terms -= np.conj(hankel(tail[p])) * tail[q][:, None, :]
    result: NDArray[np.complex128] = np.cumsum(terms, axis=-1)
    return result


def _shear(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return ``out[w, i, j] = matrix[w, i - j, j]`` on and below the diagonal (junk above)."""
    n_windows, n_taps = matrix.shape[0], matrix.shape[1]
    # Column j of the input, padded by one, starts j(K + 1) along the buffer: read in rows of K,
    # entry (j, i) of the buffer is column j's entry i - j.
    columns = np.concatenate(
        [np.swapaxes(matrix, 1, 2), np.zeros((n_windows, n_taps, 1), dtype=matrix.dtype)],
        axis=2,
    )
    flat = columns.reshape(n_windows, -1)[:, : n_taps * n_taps]
    return np.swapaxes(flat.reshape(n_windows, n_taps, n_taps), 1, 2)


def _fitted(
    bases: NDArray[np.complex128], weights: NDArray[np.complex128], n_block: int, n_unit: int
) -> NDArray[np.complex128]:
    """Return the fit ``(n_samples,)``: each unit's delayed bases on its block's weights."""
    n_orders, n_samples = bases.shape
    taps = weights.reshape(weights.shape[0], n_orders, -1)
    n_taps, n_units = taps.shape[-1], n_samples // n_unit
    late = [_lagged_blocks(basis, n_taps, n_unit) for basis in bases]
    owner = np.arange(n_units) // (n_block // n_unit)
    fitted = np.empty((n_units, n_unit), dtype=np.complex128)
    per_chunk = max(1, _MAX_BYTES // (64 * n_orders * (n_unit + 2 * n_taps)))
    # Chunks of units bound the FFT buffers; the count depends only on the shapes.
    for first in range(0, n_units, per_chunk):
        chunk = slice(first, first + per_chunk)
        lagged = np.stack([view[chunk] for view in late], axis=1)
        convolved = fftconvolve(lagged, taps[owner[chunk]], mode="valid", axes=-1)
        fitted[chunk] = np.sum(convolved, axis=1)
    return fitted.reshape(-1)
