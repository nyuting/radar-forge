r"""Range-Doppler processing: windows, pulse compression, the two FFTs, MTI and unfolding.

Turns a baseband cube ``(n_pulses, n_samples)`` (slow time on axis 0, fast time on
axis 1) into a range-Doppler map, and undoes the folding of its two circular axes.

Both axes are **bistatic**. Range is the range sum :math:`R_T + R_R = c\tau`, twice the
monostatic range for collocated sites; rate is its time derivative, **positive
receding**, with :math:`f_d = -\dot{R}/\lambda` [5]_. A monostatic display halves both
where it draws them. :func:`range_fft` leaves bin 0 at zero delay, :func:`doppler_fft`
shifts zero Doppler to the middle, and neither normalises: a unit tone peaks at :math:`N`.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §4.6 (FM pulse compression), §5.2.1 (pulse
       cancellers), §5.3 (pulse Doppler), §5.5.4 (ambiguity resolution).
.. [2] F. J. Harris, "On the use of windows for harmonic analysis with the
       discrete Fourier transform," *Proc. IEEE*, vol. 66, no. 1, pp. 51-83, 1978.
.. [3] T. T. Taylor, "Design of line-source antennas for narrow beamwidth and
       low side lobes," *IRE Trans. Antennas Propag.*, vol. AP-3, no. 1, pp. 16-28, 1955.
.. [4] G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014,
       ch. 3 (the FMCW beat-frequency relation).
.. [5] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech, 2005, ch. 3 (the
       range-sum ellipse), §6.1 (target Doppler).
.. [6] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*,
       3rd ed., Pearson, 2010, ch. 8 (the DFT and Parseval's relation).
"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve, get_window

from radar_forge.constants import SPEED_OF_LIGHT_MPS
from radar_forge.propagation import bistatic_range_rate_from_doppler_mps

__all__ = [
    "WINDOW_NAMES",
    "WindowName",
    "apply_window",
    "coherent_gain_linear",
    "doppler_bin_centers_mps",
    "doppler_fft",
    "fold_into_period",
    "fold_velocity_mps",
    "matched_filter",
    "mti_filter",
    "processing_loss_db",
    "range_bin_centers_m",
    "range_doppler_map",
    "range_fft",
    "unfold_doppler_dual_prf",
    "unfold_dual_period",
    "unfold_range_dual_prf",
    "window",
]

WindowName = Literal[
    "rectangular", "hann", "hamming", "blackman", "blackmanharris", "taylor", "chebyshev"
]
"""Window shapes :func:`window` accepts, in order of increasing sidelobe suppression."""

WINDOW_NAMES: tuple[WindowName, ...] = get_args(WindowName)
"""The window names as a tuple; ``taylor`` and ``chebyshev`` take ``sidelobe_level_db``."""

_SCIPY_NAME = {"rectangular": "boxcar"}
_PARAMETRIC = ("taylor", "chebyshev")
# Shallower Chebyshev designs grow end spikes taller than the centre; SciPy only warns.
_MIN_CHEBYSHEV_DEPTH_DB = 45.0


def window(
    name: WindowName,
    n_samples: int,
    *,
    normalise: bool = False,
    sidelobe_level_db: float | None = None,
) -> NDArray[np.float64]:
    r"""Return symmetric window weights, shape ``(n_samples,)``, raw peak 1.

    Thin over :func:`scipy.signal.get_window` with ``fftbins=False``. Taylor's
    :math:`\bar{n}` is the smallest that realises the level, :math:`\bar{n} \ge 2A^2 + 1/2`.

    Parameters
    ----------
    normalise : bool, optional
        Divide by the mean, so :func:`coherent_gain_linear` is 1.
    sidelobe_level_db : float, optional
        Peak sidelobe, negative (-60.0); required by ``"taylor"`` and ``"chebyshev"``
        (at most -45.0), refused by the fixed shapes.

    References
    ----------
    Harris [2]_, Table 1; Taylor [3]_.
    """
    if name not in WINDOW_NAMES or n_samples < 1:
        msg = f"need a name in {WINDOW_NAMES} and n_samples >= 1; got {name!r}, {n_samples}."
        raise ValueError(msg)
    if (name in _PARAMETRIC) != (sidelobe_level_db is not None):
        msg = f"sidelobe_level_db is required for {_PARAMETRIC} and refused for {name!r}."
        raise ValueError(msg)
    depth_db = 0.0 if sidelobe_level_db is None else -sidelobe_level_db
    if sidelobe_level_db is not None and depth_db <= 0.0:
        msg = f"sidelobe_level_db ({sidelobe_level_db:g}) must be negative, e.g. -60.0."
        raise ValueError(msg)
    if name == "chebyshev" and depth_db < _MIN_CHEBYSHEV_DEPTH_DB:
        msg = f"chebyshev needs sidelobe_level_db <= -{_MIN_CHEBYSHEV_DEPTH_DB:g}."
        raise ValueError(msg)

    spec: str | tuple[str, float] | tuple[str, int, float] = _SCIPY_NAME.get(name, name)
    if name == "taylor":
        spec = ("taylor", _taylor_nbar(depth_db), depth_db)
    elif name == "chebyshev":
        spec = ("chebwin", depth_db)
    weights = np.asarray(get_window(spec, n_samples, fftbins=False), dtype=np.float64)
    result: NDArray[np.float64] = weights / weights.mean() if normalise else weights
    return result


def apply_window(
    samples: ArrayLike,
    name: WindowName,
    *,
    axis: int,
    normalise: bool = False,
    sidelobe_level_db: float | None = None,
) -> NDArray[np.complex128]:
    """Multiply ``samples`` along ``axis`` by :func:`window` of that axis's length.

    References
    ----------
    Harris [2]_, Table 1.
    """
    moved = np.moveaxis(np.asarray(samples, dtype=np.complex128), axis, -1)
    weights = window(
        name, moved.shape[-1], normalise=normalise, sidelobe_level_db=sidelobe_level_db
    )
    result: NDArray[np.complex128] = np.moveaxis(moved * weights, -1, axis)
    return result


def coherent_gain_linear(weights: ArrayLike) -> float:
    r"""Return :math:`G_c = \frac{1}{N}\sum_n w[n]`, the factor a window scales a peak by.

    References
    ----------
    Harris [2]_, Table 1.
    """
    return float(np.mean(_validated_window(weights)))


def processing_loss_db(weights: ArrayLike) -> float:
    r"""Return the SNR a window costs, :math:`-10\log_{10}\frac{(\sum w)^2}{N\sum w^2}` dB.

    References
    ----------
    Harris [2]_, Table 1 (equivalent noise bandwidth).
    """
    weights_arr = _validated_window(weights)
    coherent_sum = float(np.sum(weights_arr))
    if coherent_sum == 0.0:
        msg = "window weights sum to zero; processing loss is undefined."
        raise ValueError(msg)
    efficiency_linear = coherent_sum**2 / (weights_arr.size * float(np.sum(weights_arr**2)))
    return float(-10.0 * np.log10(efficiency_linear))


def matched_filter(
    samples: ArrayLike, reference: ArrayLike, *, axis: int = -1
) -> NDArray[np.complex128]:
    r"""Correlate ``samples`` with ``reference``: :math:`y[n] = \sum_k x[k]\,h^*[k-n]`.

    Full linear correlation by :func:`scipy.signal.fftconvolve`, so zero lag is index
    ``n_reference - 1``, not 0. A bare ``ifft(fft(x) * conj(fft(h)))`` is circular and
    wraps the far end of the record onto zero delay.

    Parameters
    ----------
    reference : array_like
        Transmitted waveform, ``(n_reference,)``, not pre-conjugated.

    Returns
    -------
    numpy.ndarray
        ``axis`` of length ``n_samples + n_reference - 1``.

    References
    ----------
    Richards [1]_, §4.6.
    """
    reference_arr = np.asarray(reference, dtype=np.complex128)
    if reference_arr.ndim != 1 or reference_arr.size == 0:
        msg = f"reference must be one-dimensional and non-empty, got shape {reference_arr.shape}."
        raise ValueError(msg)

    moved = np.moveaxis(np.asarray(samples, dtype=np.complex128), axis, -1)
    # Correlation is convolution with the conjugate time-reverse of the template.
    template = np.conjugate(reference_arr[::-1]).reshape((1,) * (moved.ndim - 1) + (-1,))
    compressed = np.asarray(fftconvolve(moved, template, axes=-1), dtype=np.complex128)
    result: NDArray[np.complex128] = np.moveaxis(compressed, -1, axis)
    return result


def range_fft(
    samples: ArrayLike,
    *,
    n_fft: int | None = None,
    weights: ArrayLike | None = None,
    axis: int = -1,
) -> NDArray[np.complex128]:
    r"""Transform fast time into range bins; label them with :func:`range_bin_centers_m`.

    Unshifted (bin 0 is zero delay) and unnormalised, so
    :math:`\sum_k |X_k|^2 = N_{\mathrm{fft}} \sum_n |w_n x_n|^2`. Padding to ``n_fft``
    interpolates; it adds no resolution.

    Parameters
    ----------
    samples : array_like
        Complex baseband, e.g. ``(n_pulses, n_samples)``.
    weights : array_like, optional
        Window along ``axis``, e.g. from :func:`window`.

    References
    ----------
    Charvat [4]_, ch. 3; Oppenheim and Schafer [6]_, ch. 8.
    """
    samples_arr, n_fft = _prepare_transform(samples, n_fft, weights, axis)
    result: NDArray[np.complex128] = np.fft.fft(samples_arr, n=n_fft, axis=axis)
    return result


def doppler_fft(
    profiles: ArrayLike,
    *,
    n_fft: int | None = None,
    weights: ArrayLike | None = None,
    axis: int = 0,
) -> NDArray[np.complex128]:
    """Transform slow time into Doppler bins; label them with :func:`doppler_bin_centers_mps`.

    Shifted, zero Doppler at index ``n_fft // 2``: index rises with :math:`f_d`, so it
    falls with range rate. Unnormalised; Parseval as :func:`range_fft`.

    Parameters
    ----------
    profiles : array_like
        Complex range profiles, e.g. ``(n_pulses, n_range_bins)``.

    References
    ----------
    Richards [1]_, §5.3; Oppenheim and Schafer [6]_, ch. 8.
    """
    profiles_arr, n_fft = _prepare_transform(profiles, n_fft, weights, axis)
    spectrum = np.fft.fft(profiles_arr, n=n_fft, axis=axis)
    result: NDArray[np.complex128] = np.fft.fftshift(spectrum, axes=axis)
    return result


def range_doppler_map(
    samples: ArrayLike,
    *,
    fast_time_weights: ArrayLike | None = None,
    slow_time_weights: ArrayLike | None = None,
    n_range_fft: int | None = None,
    n_doppler_fft: int | None = None,
    fast_time_axis: int = -1,
    slow_time_axis: int = 0,
) -> NDArray[np.complex128]:
    """Return :func:`doppler_fft` of :func:`range_fft`, each along its own axis.

    Returns
    -------
    numpy.ndarray
        Complex map, e.g. ``(n_doppler_bins, n_range_bins)``.

    References
    ----------
    Richards [1]_, §5.3.
    """
    samples_arr = np.asarray(samples, dtype=np.complex128)
    n_dims, axes = samples_arr.ndim, (fast_time_axis, slow_time_axis)
    if any(not -n_dims <= axis < n_dims for axis in axes) or len({a % n_dims for a in axes}) < 2:
        msg = f"fast_time_axis and slow_time_axis must be two different valid axes; got {axes}."
        raise ValueError(msg)

    profiles = range_fft(samples_arr, n_fft=n_range_fft, weights=fast_time_weights, axis=axes[0])
    return doppler_fft(profiles, n_fft=n_doppler_fft, weights=slow_time_weights, axis=axes[1])


def range_bin_centers_m(
    n_bins: int, bandwidth_hz: float, chirp_duration_s: float, sample_rate_hz: float
) -> NDArray[np.float64]:
    r"""Return the **bistatic range** :math:`c\tau = R_T + R_R` of each range bin, in metres.

    **Not halved**: twice the monostatic range for collocated sites. Bin :math:`k` holds
    :math:`R_k = c k f_s T / (B N)`, on a circle of circumference :math:`c f_s T / B`.
    ``sample_rate_hz`` is the ADC rate after deramp.

    References
    ----------
    Charvat [4]_, ch. 3; Willis [5]_, ch. 3.
    """
    # `not x > 0` rather than `x <= 0`, so NaN is rejected too.
    if n_bins < 1 or not all(x > 0.0 for x in (bandwidth_hz, chirp_duration_s, sample_rate_hz)):
        msg = "n_bins must be >= 1, and bandwidth, sweep time and sample rate positive."
        raise ValueError(msg)

    beat_hz = np.arange(n_bins, dtype=np.float64) * sample_rate_hz / n_bins
    delay_s = beat_hz * chirp_duration_s / bandwidth_hz
    result: NDArray[np.float64] = SPEED_OF_LIGHT_MPS * delay_s
    return result


def doppler_bin_centers_mps(
    n_bins: int, pulse_repetition_interval_s: float, wavelength_m: float
) -> NDArray[np.float64]:
    r"""Return the **bistatic range rate, positive receding**, of each shifted Doppler bin.

    :math:`\dot{R} = -\lambda f_d`, **not halved**, and so **decreasing** with index over
    :math:`(-v_{ua}, +v_{ua}]`, :math:`v_{ua} = \lambda / (2 T_{\mathrm{PRI}})`; zero at
    ``n_bins // 2``. The interval and bin 0 at exactly :math:`+v_{ua}` hold for even
    ``n_bins``; an odd count is symmetric, :math:`\pm (N-1)/2` steps, strictly inside it.

    References
    ----------
    Richards [1]_, §5.3; Willis [5]_, §6.1.

    Examples
    --------
    >>> [float(r) for r in doppler_bin_centers_mps(4, 1e-3, 0.03)]
    [15.0, 7.5, 0.0, -7.5]
    """
    if n_bins < 1 or not all(x > 0.0 for x in (pulse_repetition_interval_s, wavelength_m)):
        msg = "n_bins must be >= 1, and pulse_repetition_interval_s, wavelength_m positive."
        raise ValueError(msg)

    doppler_hz = np.fft.fftshift(np.fft.fftfreq(n_bins, d=pulse_repetition_interval_s))
    # The sign is the one propagation.doppler_shift_hz defines; adding +0.0 turns the
    # stationary bin's -0.0 (from negating 0 Hz) into +0.0.
    result: NDArray[np.float64] = (
        bistatic_range_rate_from_doppler_mps(doppler_hz, wavelength_m) + 0.0
    )
    return result


def mti_filter(samples: ArrayLike, *, n_pulses: int = 2, axis: int) -> NDArray[np.complex128]:
    """Cancel stationary returns with :math:`H(z) = (1 - z^{-1})^{n-1}`, ``n = n_pulses``.

    Exact nulls at zero Doppler and every multiple of the PRF, gain :math:`2^{n-1}` midway.
    ``np.diff`` is this causal canceller with its transient dropped, so ``axis`` shortens
    by ``n_pulses - 1`` and the two-pulse output is :math:`x[m+1] - x[m]`.

    Parameters
    ----------
    samples : array_like
        Complex baseband, e.g. ``(n_pulses, n_samples)``.

    References
    ----------
    Richards [1]_, §5.2.1.
    """
    samples_arr = np.asarray(samples, dtype=np.complex128)
    if n_pulses not in (2, 3) or np.moveaxis(samples_arr, axis, 0).shape[0] < n_pulses:
        msg = f"n_pulses ({n_pulses}) must be 2 or 3, and no more than the pulses along axis."
        raise ValueError(msg)
    result: NDArray[np.complex128] = np.diff(samples_arr, n=n_pulses - 1, axis=axis)
    return result


def fold_into_period(
    value: ArrayLike, period: ArrayLike, *, lower: ArrayLike
) -> NDArray[np.float64]:
    """Wrap ``value`` into the half-open ``[lower, lower + period)``, in any shared unit.

    References
    ----------
    Richards [1]_, §5.5.4.
    """
    period_arr = np.asarray(period, dtype=np.float64)
    if np.any(period_arr <= 0.0):
        msg = "period must be strictly positive."
        raise ValueError(msg)
    return _wrap(np.asarray(value, dtype=np.float64), period_arr, lower)


def unfold_dual_period(
    folded_a: ArrayLike,
    folded_b: ArrayLike,
    period_a: float,
    period_b: float,
    *,
    lower_bound: float,
    upper_bound: float,
    tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Recover a value from two readings folded at different periods.

    Tries every :math:`a + k P_a` within the inclusive bounds and keeps the one whose fold
    at :math:`P_b` lands nearest :math:`b`. The readings broadcast, so a column against a
    row pairs every one. A true value outside the bounds can alias onto one inside and come
    back confident.

    Returns
    -------
    value, residual : numpy.ndarray
        Recovered value (``nan`` past ``tolerance``) and its miss on ``folded_b``.

    References
    ----------
    Richards [1]_, §5.5.4.

    Examples
    --------
    >>> bounds = {"lower_bound": 0.0, "upper_bound": 29.0}
    >>> float(unfold_dual_period(4.0, 2.0, 6.0, 5.0, **bounds, tolerance=0.1)[0])
    22.0
    """
    # Written as `not ...` so NaN fails each test instead of slipping past it.
    positive = all(x > 0.0 for x in (period_a, period_b, tolerance))
    if not positive or not upper_bound >= lower_bound:
        msg = "periods and tolerance must be positive, and upper_bound >= lower_bound."
        raise ValueError(msg)
    if period_a == period_b and upper_bound - lower_bound > period_a:
        msg = "equal periods cannot tell apart values more than one period apart."
        raise ValueError(msg)

    reading_a, reading_b = np.broadcast_arrays(
        np.asarray(folded_a, dtype=np.float64), np.asarray(folded_b, dtype=np.float64)
    )
    # Start A at the bottom fold of the bounds, so every reading has candidates
    # k = 0, 1, ...; the candidate axis broadcasts last, (..., n_candidates).
    fold_index = np.arange(int(np.ceil((upper_bound - lower_bound) / period_a)) + 1)
    candidate = _wrap(reading_a, period_a, lower_bound)[..., None] + fold_index * period_a

    # Compare on the circle: just below a fold edge and just above it are neighbours.
    miss = np.abs(_wrap(candidate - reading_b[..., None], period_b, -period_b / 2.0))
    miss = np.where(candidate <= upper_bound, miss, np.inf)
    best = np.argmin(miss, axis=-1)[..., None]
    residual = np.take_along_axis(miss, best, axis=-1)[..., 0]
    value = np.take_along_axis(candidate, best, axis=-1)[..., 0]
    return np.where(residual <= tolerance, value, np.nan), residual


def fold_velocity_mps(velocity_mps: ArrayLike, span_mps: ArrayLike) -> NDArray[np.float64]:
    r"""Wrap a range rate, positive receding, into ``(-span/2, +span/2]``, as the map shows it.

    Closed at the top, unlike :func:`fold_into_period`: rate is :math:`-\lambda f_d` and the
    shifted Doppler axis is :math:`[-\mathrm{PRF}/2, +\mathrm{PRF}/2)`. For
    :func:`doppler_bin_centers_mps` the span is :math:`\lambda / T_{\mathrm{PRI}}`,
    and its map has a bin on the closed edge only for an even number of bins.

    References
    ----------
    Richards [1]_, §5.5.4.
    """
    span = np.asarray(span_mps, dtype=np.float64)
    flipped = -np.asarray(velocity_mps, dtype=np.float64)
    return -fold_into_period(flipped, span, lower=-span / 2.0)


def unfold_doppler_dual_prf(
    folded_velocity_a_mps: ArrayLike,
    folded_velocity_b_mps: ArrayLike,
    span_a_mps: float,
    span_b_mps: float,
    *,
    max_velocity_mps: float,
    tolerance_mps: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Recover a range rate from two bursts by :func:`unfold_dual_period`.

    Periods are the two rate spans, bounds :math:`\pm` ``max_velocity_mps``; returns
    ``(velocity_mps, residual_mps)``.

    References
    ----------
    Richards [1]_, §5.5.4.
    """
    if not (span_a_mps > 0.0 and span_b_mps > 0.0) or span_a_mps == span_b_mps:
        msg = "the two rate spans must be positive and different."
        raise ValueError(msg)
    # `not >` so a NaN bound is rejected here rather than inside np.arange.
    if not max_velocity_mps > min(span_a_mps, span_b_mps) / 2.0:
        msg = "max_velocity_mps must exceed half the smaller span, or nothing is folded."
        raise ValueError(msg)
    return unfold_dual_period(
        folded_velocity_a_mps,
        folded_velocity_b_mps,
        span_a_mps,
        span_b_mps,
        lower_bound=-max_velocity_mps,
        upper_bound=max_velocity_mps,
        tolerance=tolerance_mps,
    )


def unfold_range_dual_prf(
    folded_range_a_m: ArrayLike,
    folded_range_b_m: ArrayLike,
    span_a_m: float,
    span_b_m: float,
    *,
    max_range_m: float,
    tolerance_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Recover a range from two maps by :func:`unfold_dual_period` over ``[0, max_range_m]``.

    ``span_a_m`` and ``span_b_m`` are each map's range circumference; returns
    ``(range_m, residual_m)``.

    References
    ----------
    Richards [1]_, §5.5.4.
    """
    if max_range_m < 0.0:
        msg = f"max_range_m must be non-negative; got {max_range_m!r}."
        raise ValueError(msg)
    return unfold_dual_period(
        folded_range_a_m,
        folded_range_b_m,
        span_a_m,
        span_b_m,
        lower_bound=0.0,
        upper_bound=max_range_m,
        tolerance=tolerance_m,
    )


def _wrap(value: NDArray[np.float64], period: ArrayLike, lower: ArrayLike) -> NDArray[np.float64]:
    """Return ``value`` folded into ``[lower, lower + period)``, unchecked."""
    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = lower_arr + np.asarray(period, dtype=np.float64)
    wrapped = np.mod(value - lower_arr, period) + lower_arr
    # np.mod of a tiny negative offset rounds up to the divisor itself, which would
    # land on the open top edge; that point is the bottom of the next fold.
    result: NDArray[np.float64] = np.where(wrapped >= upper_arr, lower_arr, wrapped)
    return result


def _taylor_nbar(depth_db: float) -> int:
    r"""Return the smallest :math:`\bar{n}` realising sidelobes ``depth_db`` below the peak."""
    a_parameter = np.arccosh(10.0 ** (depth_db / 20.0)) / np.pi
    return max(2, int(np.ceil(2.0 * a_parameter**2 + 0.5)))


def _validated_window(weights: ArrayLike) -> NDArray[np.float64]:
    """Return ``weights`` as a non-empty 1-D float array, or raise."""
    weights_arr = np.asarray(weights, dtype=np.float64)
    if weights_arr.ndim != 1 or weights_arr.size == 0:
        msg = f"window must be one-dimensional and non-empty, got shape {weights_arr.shape}."
        raise ValueError(msg)
    return weights_arr


def _prepare_transform(
    samples: ArrayLike, n_fft: int | None, weights: ArrayLike | None, axis: int
) -> tuple[NDArray[np.complex128], int]:
    """Validate and apply ``weights`` along ``axis``, and resolve ``n_fft``."""
    moved = np.moveaxis(np.asarray(samples, dtype=np.complex128), axis, -1)
    n_samples = moved.shape[-1]
    n_fft = n_samples if n_fft is None else n_fft
    if n_fft < max(n_samples, 1):
        msg = f"n_fft ({n_fft}) must be at least one and not shorter than the {n_samples} samples."
        raise ValueError(msg)
    if weights is not None:
        weights_arr = _validated_window(weights)
        if weights_arr.size != n_samples:
            msg = f"weights length ({weights_arr.size}) must match the {n_samples} samples."
            raise ValueError(msg)
        moved = moved * weights_arr
    return np.moveaxis(moved, -1, axis), n_fft
