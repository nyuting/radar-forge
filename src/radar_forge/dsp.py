r"""Range-Doppler processing: windows, pulse compression, the two FFTs, MTI and unfolding.

Turns a cube of complex baseband, canonically ``(n_pulses, n_samples)`` (slow
time on axis 0, fast time on axis 1), into a range-Doppler map, and undoes the
folding of its two circular axes. Every function takes an explicit ``axis``.

Both axes are **bistatic**. Range is the range sum :math:`R_T + R_R = c\tau`
(twice the monostatic range when the sites coincide); rate is its time
derivative, **positive receding**, and Doppler is :math:`f_d = -\dot{R}/\lambda`
[5]_. A monostatic display halves both where it draws them, not here.

:func:`range_fft` does not shift (bin 0 is zero delay); :func:`doppler_fft`
does (zero Doppler in the middle). Neither normalises: a unit tone peaks at
:math:`N`, and Parseval reads :math:`\sum_k |X_k|^2 = N_{\mathrm{fft}} \sum_n |x_n|^2` [6]_.

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

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS

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
    r"""Return symmetric window weights of shape ``(n_samples,)``, raw peak 1.

    :func:`scipy.signal.get_window` with ``fftbins=False``. First sidelobe and
    :func:`processing_loss_db` per Harris [2]_, Table 1: rectangular -13 dB,
    0 dB; hann -32, 1.76; hamming -43, 1.34; blackman -58, 2.37; blackmanharris
    -92, 3.02. Taylor's :math:`\bar{n}` is the smallest that realises the level
    [3]_, :math:`\bar{n} \ge 2A^2 + 1/2`, :math:`A = \cosh^{-1}(10^{-\mathrm{SLL}/20})/\pi`;
    a fixed small :math:`\bar{n}` silently misses deep designs.

    Parameters
    ----------
    name : WindowName
        One of :data:`WINDOW_NAMES`.
    n_samples : int
        Number of weights, at least one.
    normalise : bool, optional
        Divide by the mean, so :func:`coherent_gain_linear` is 1. Default False.
    sidelobe_level_db : float, optional
        Peak sidelobe, negative (-60.0), for ``"taylor"`` and ``"chebyshev"``
        (at most -45.0) only: required by them, refused by the rest.

    Returns
    -------
    numpy.ndarray
        Real weights, shape ``(n_samples,)``.

    Examples
    --------
    >>> float(window("hann", 9).max())
    1.0
    >>> bool(np.isclose(window("hann", 8, normalise=True).mean(), 1.0))
    True
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

    A tone's FFT peak then scales by the weights' :func:`coherent_gain_linear`
    (Harris [2]_, Table 1).

    Parameters
    ----------
    samples : array_like
        Data of any shape, e.g. ``(n_pulses, n_samples)``.
    name, normalise, sidelobe_level_db
        As :func:`window`.
    axis : int
        Axis to weight. Required: range and Doppler windows go on different axes.

    Returns
    -------
    numpy.ndarray
        Windowed samples, complex, of the shape of ``samples``.
    """
    moved = np.moveaxis(np.asarray(samples, dtype=np.complex128), axis, -1)
    weights = window(
        name, moved.shape[-1], normalise=normalise, sidelobe_level_db=sidelobe_level_db
    )
    result: NDArray[np.complex128] = np.moveaxis(moved * weights, -1, axis)
    return result


def coherent_gain_linear(weights: ArrayLike) -> float:
    r"""Return :math:`G_c = \frac{1}{N}\sum_n w[n]`, the factor a window scales a peak by.

    Harris [2]_, Table 1, "coherent gain".

    Parameters
    ----------
    weights : array_like
        Window, shape ``(n,)``, non-empty.

    Returns
    -------
    float
        Linear gain; 1 for rectangular and for any ``normalise=True`` window.
    """
    return float(np.mean(_validated_window(weights)))


def processing_loss_db(weights: ArrayLike) -> float:
    r"""Return the SNR a window costs, :math:`L = -10\log_{10}\frac{(\sum w)^2}{N\sum w^2}` dB.

    The windowed filter's mismatch to the matched filter: :math:`10\log_{10}`
    of the equivalent noise bandwidth in bins (Harris [2]_, Table 1).
    Invariant to scaling.

    Parameters
    ----------
    weights : array_like
        Window, shape ``(n,)``, non-empty, not summing to zero.

    Returns
    -------
    float
        Loss in dB, zero only for a uniform window.

    Examples
    --------
    >>> round(processing_loss_db(window("hann", 4096)), 2)
    1.76
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

    An LFM pulse of bandwidth :math:`B` and length :math:`T` compresses to a
    :math:`1/B` mainlobe with SNR gain :math:`BT` and a -13.3 dB peak sidelobe
    (Richards [1]_, §4.6). :func:`scipy.signal.fftconvolve` pads to a *linear*
    correlation; a bare ``ifft(fft(x) * conj(fft(h)))`` is circular and wraps
    the far end of the record onto zero delay as a ghost.

    Parameters
    ----------
    samples : array_like
        Complex baseband, e.g. ``(n_pulses, n_samples)``.
    reference : array_like
        Transmitted waveform, shape ``(n_reference,)``, not pre-conjugated.
    axis : int, optional
        Fast-time axis. Default -1.

    Returns
    -------
    numpy.ndarray
        Complex; ``axis`` has length ``n_samples + n_reference - 1`` and **zero
        lag is index** ``n_reference - 1``.

    Examples
    --------
    >>> pulse = np.exp(10j * np.pi * np.linspace(-1.0, 1.0, 50) ** 2)
    >>> int(np.argmax(np.abs(matched_filter(pulse, pulse))))
    49
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
    r"""Transform fast time into range bins, unnormalised and unshifted.

    A deramped echo is a beat tone :math:`f_b = \alpha\tau` (Charvat [4]_,
    ch. 3), so the FFT along fast time is the range profile, bin 0 at zero
    delay. Parseval: :math:`\sum_k |X_k|^2 = N_{\mathrm{fft}} \sum_n |w_n x_n|^2`
    (Oppenheim and Schafer [6]_, ch. 8).

    Parameters
    ----------
    samples : array_like
        Complex baseband, e.g. ``(n_pulses, n_samples)``.
    n_fft : int, optional
        Transform length, not shorter than ``axis`` (the default). Padding
        interpolates; it adds no resolution.
    weights : array_like, optional
        Window along ``axis``, e.g. from :func:`window`. Default none.
    axis : int, optional
        Fast-time axis. Default -1.

    Returns
    -------
    numpy.ndarray
        Complex profiles, ``axis`` of length ``n_fft``; see :func:`range_bin_centers_m`.

    Examples
    --------
    >>> int(np.argmax(np.abs(range_fft(np.exp(2j * np.pi * 3 * np.arange(64) / 64)))))
    3
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
    r"""Transform slow time into Doppler bins, unnormalised, zero Doppler at ``n_fft // 2``.

    Phase advances :math:`2\pi f_d T_{\mathrm{PRI}}` per pulse (Richards [1]_,
    §5.3), so index rises with :math:`f_d` and *falls* with range rate.
    Parseval as :func:`range_fft`. Unwindowed, strong clutter leaks into every bin.

    Parameters
    ----------
    profiles : array_like
        Complex range profiles, e.g. ``(n_pulses, n_range_bins)``.
    n_fft, weights
        As :func:`range_fft`, along ``axis``.
    axis : int, optional
        Slow-time axis. Default 0.

    Returns
    -------
    numpy.ndarray
        Complex spectra, ``axis`` of length ``n_fft``; see :func:`doppler_bin_centers_mps`.

    Examples
    --------
    >>> int(np.argmax(np.abs(doppler_fft(np.ones(16)))))
    8
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
    """Return :func:`doppler_fft` of :func:`range_fft`; the two commute (Richards [1]_, §5.3).

    Parameters
    ----------
    samples : array_like
        Complex cube, canonically ``(n_pulses, n_samples)``.
    fast_time_weights, slow_time_weights, n_range_fft, n_doppler_fft
        Each transform's ``weights`` and ``n_fft``. Default none.
    fast_time_axis, slow_time_axis : int, optional
        Two different axes. Defaults -1 and 0.

    Returns
    -------
    numpy.ndarray
        Complex map, e.g. ``(n_doppler_bins, n_range_bins)``.
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
    r"""Return the **bistatic range** :math:`R_T + R_R = c\tau` of each range bin, in metres.

    **Not halved**: twice the monostatic range when the sites coincide. Bin
    :math:`k` holds beat :math:`k f_s / N`, delay :math:`\tau = f_b T / B`
    (Charvat [4]_, ch. 3), so :math:`R_k = c k f_s T / (B N)` (Willis [5]_,
    ch. 3). The axis is a circle of circumference :math:`c f_s T / B`.

    Parameters
    ----------
    n_bins : int
        :func:`range_fft` length :math:`N`, at least one.
    bandwidth_hz, chirp_duration_s, sample_rate_hz : float
        Swept bandwidth :math:`B`, sweep time :math:`T`, complex sample rate
        :math:`f_s`; strictly positive.

    Returns
    -------
    numpy.ndarray
        Bistatic range per bin, shape ``(n_bins,)``, from 0 upward.

    Examples
    --------
    >>> [round(float(r), 6) for r in range_bin_centers_m(2, 1e9, 40e-6, 1e6)]
    [0.0, 5.995849]
    """
    if n_bins < 1 or min(bandwidth_hz, chirp_duration_s, sample_rate_hz) <= 0.0:
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

    :math:`\dot{R} = -\lambda f_d` (Willis [5]_, §6.1) on :func:`doppler_fft`'s
    shifted bins (Richards [1]_, §5.3). **Not halved** (a monostatic radial
    velocity is half this) and, because of the sign, **decreasing** with
    index; zero at ``n_bins // 2``.

    Parameters
    ----------
    n_bins : int
        :func:`doppler_fft` length :math:`N`, at least one.
    pulse_repetition_interval_s, wavelength_m : float
        :math:`T_{\mathrm{PRI}}` and :math:`\lambda`, strictly positive.

    Returns
    -------
    numpy.ndarray
        Rate per bin, m/s, shape ``(n_bins,)``, spaced :math:`\lambda/(N T_{\mathrm{PRI}})`
        over :math:`(-v_{ua}, +v_{ua}]`, :math:`v_{ua} = \lambda / (2 T_{\mathrm{PRI}})`.

    Examples
    --------
    >>> [float(r) for r in doppler_bin_centers_mps(4, 1e-3, 0.03) + 0.0]
    [15.0, 7.5, 0.0, -7.5]
    """
    if n_bins < 1 or min(pulse_repetition_interval_s, wavelength_m) <= 0.0:
        msg = "n_bins must be >= 1, and pulse_repetition_interval_s, wavelength_m positive."
        raise ValueError(msg)

    doppler_hz = np.fft.fftshift(np.fft.fftfreq(n_bins, d=pulse_repetition_interval_s))
    result: NDArray[np.float64] = -wavelength_m * doppler_hz
    return result


def mti_filter(samples: ArrayLike, *, n_pulses: int = 2, axis: int) -> NDArray[np.complex128]:
    r"""Cancel stationary returns with :math:`H(z) = (1 - z^{-1})^{n-1}`.

    Binomial taps ``[1, -1]`` or ``[1, -2, 1]`` (Richards [1]_, §5.2.1), so
    :math:`|H(f_d)| = (2|\sin\pi f_d T_{\mathrm{PRI}}|)^{n-1}`: exact nulls at
    zero Doppler and each multiple of the PRF (the blind speeds), gain
    :math:`2^{n-1}` midway. ``np.diff`` of order :math:`n-1` *is* this causal
    canceller with its start-up transient dropped, in exact integer taps;
    ``scipy.signal.lfilter`` computes the same plus transients to discard.

    Parameters
    ----------
    samples : array_like
        Complex baseband, e.g. ``(n_pulses, n_samples)``.
    n_pulses : int, optional
        Canceller order, 2 or 3. Default 2.
    axis : int
        Slow-time axis, with at least ``n_pulses`` entries. Required.

    Returns
    -------
    numpy.ndarray
        Complex, ``axis`` **shortened by** ``n_pulses - 1``; two-pulse output
        is :math:`y[m] = x[m+1] - x[m]`.

    Examples
    --------
    >>> mti_filter(np.ones((4, 3)), axis=0).shape
    (3, 3)
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
    """Wrap ``value`` into ``[lower, lower + period)``: the forward model of every fold.

    Range folds with ``lower = 0``; Doppler round zero (Richards [1]_, §5.5.4).

    Parameters
    ----------
    value, period, lower : array_like
        True value, its strictly positive repeat, and the inclusive bottom of
        the reporting interval, in any one unit; broadcast together.

    Returns
    -------
    numpy.ndarray
        Reported value, in ``[lower, lower + period)``.

    Examples
    --------
    >>> float(fold_into_period(70.0, 60.0, lower=0.0))
    10.0
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
    r"""Recover a value from readings folded at two periods (Chinese remainder theorem).

    Tries every :math:`x_k = a + k P_a` within the bounds and keeps the one
    whose fold at :math:`P_b` lands nearest :math:`b` (Richards [1]_, §5.5.4):
    exact search, so noise degrades it gracefully. Periods in lowest ratio
    :math:`p:q` repeat after :math:`q P_a`. A value outside the bounds can
    alias onto one inside and come back confident; hence required bounds.

    Parameters
    ----------
    folded_a, folded_b : array_like
        The readings; broadcast (a column against a row pairs every one).
    period_a, period_b, tolerance : float
        Each reading's period, and the largest accepted miss (about half a
        bin); strictly positive.
    lower_bound, upper_bound : float
        Inclusive interval the true value lies in; at most one period wide if
        the periods are equal.

    Returns
    -------
    value, residual : numpy.ndarray
        The recovered value (``nan`` past ``tolerance``) and its miss on
        reading B, measured the short way round :math:`P_b`.

    Examples
    --------
    >>> bounds = {"lower_bound": 0.0, "upper_bound": 29.0}
    >>> float(unfold_dual_period(4.0, 2.0, 6.0, 5.0, **bounds, tolerance=0.1)[0])
    22.0
    """
    if min(period_a, period_b, tolerance) <= 0.0 or upper_bound < lower_bound:
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


def fold_velocity_mps(
    velocity_mps: ArrayLike, unambiguous_velocity_mps: ArrayLike
) -> NDArray[np.float64]:
    r"""Wrap a rate, positive receding, into ``(-v_ua, +v_ua]``, where the map shows it.

    Doppler folds into the shifted FFT's :math:`[-\mathrm{PRF}/2, +\mathrm{PRF}/2)`
    (Richards [1]_, §5.5.4) and rate is :math:`-\lambda f_d`, so the rate
    interval is closed at the **top**. That edge is the one change from the
    closing-positive original, and why the fold is taken on :math:`-v`.

    Parameters
    ----------
    velocity_mps : array_like
        True rate, m/s, positive receding.
    unambiguous_velocity_mps : array_like
        Half-interval :math:`v_{ua}` in the same quantity, strictly positive:
        :math:`\lambda / (2 T_{\mathrm{PRI}})` for :func:`doppler_bin_centers_mps`.

    Returns
    -------
    numpy.ndarray
        Apparent rate, in ``(-v_ua, +v_ua]``.

    Examples
    --------
    >>> round(float(fold_velocity_mps(80.0, 7.65)), 9), float(fold_velocity_mps(-7.65, 7.65))
    (3.5, 7.65)
    """
    half_interval = np.asarray(unambiguous_velocity_mps, dtype=np.float64)
    flipped = -np.asarray(velocity_mps, dtype=np.float64)
    return -fold_into_period(flipped, 2.0 * half_interval, lower=-half_interval)


def unfold_doppler_dual_prf(
    folded_velocity_a_mps: ArrayLike,
    folded_velocity_b_mps: ArrayLike,
    unambiguous_velocity_a_mps: float,
    unambiguous_velocity_b_mps: float,
    *,
    max_velocity_mps: float,
    tolerance_mps: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Recover a rate from two bursts: :func:`unfold_dual_period`, periods :math:`2 v_{ua}`.

    The bounds :math:`\pm v_{\max}` are symmetric, so the sign convention
    (positive receding) passes straight through. Half-intervals in lowest
    ratio :math:`p:q` reach :math:`q\, v_{ua,a}` (Richards [1]_, §5.5.4).

    Parameters
    ----------
    folded_velocity_a_mps, folded_velocity_b_mps : array_like
        Apparent rates, m/s; broadcast together.
    unambiguous_velocity_a_mps, unambiguous_velocity_b_mps : float
        The two half-intervals, m/s; positive and different.
    max_velocity_mps, tolerance_mps : float
        Largest true speed (above the smaller half-interval) and largest
        accepted miss (about half a bin), m/s.

    Returns
    -------
    velocity_mps, residual_mps : numpy.ndarray
        Recovered rate, positive receding (``nan`` if unresolved), and its miss.

    Examples
    --------
    >>> limits = {"max_velocity_mps": 191.0, "tolerance_mps": 1.0}
    >>> round(float(unfold_doppler_dual_prf(3.52, -11.78, 38.24, 45.89, **limits)[0]), 6)
    80.0
    """
    smaller_mps = min(unambiguous_velocity_a_mps, unambiguous_velocity_b_mps)
    if smaller_mps <= 0.0 or unambiguous_velocity_a_mps == unambiguous_velocity_b_mps:
        msg = "the two unambiguous velocities must be positive and different."
        raise ValueError(msg)
    if max_velocity_mps <= smaller_mps:
        msg = "max_velocity_mps must exceed the smaller half-interval, or nothing is folded."
        raise ValueError(msg)
    return unfold_dual_period(
        folded_velocity_a_mps,
        folded_velocity_b_mps,
        2.0 * unambiguous_velocity_a_mps,
        2.0 * unambiguous_velocity_b_mps,
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
    r"""Recover a range from two maps: :func:`unfold_dual_period` over :math:`[0, R_{\max}]`.

    Each map's range axis is a circle of circumference :math:`S`, so a target
    at :math:`R` reads :math:`R \bmod S` (Richards [1]_, §5.5.4). Any range
    will do, bistatic from :func:`range_bin_centers_m` included, if shared.

    Parameters
    ----------
    folded_range_a_m, folded_range_b_m : array_like
        Ranges read off the two maps, m; broadcast together.
    span_a_m, span_b_m : float
        Each map's range circumference, m; strictly positive.
    max_range_m, tolerance_m : float
        Largest true range (non-negative) and largest accepted miss, m.

    Returns
    -------
    range_m, residual_m : numpy.ndarray
        Recovered range (``nan`` if unresolved) and its miss on map B.

    Examples
    --------
    >>> limits = {"max_range_m": 300e3, "tolerance_m": 50.0}
    >>> float(unfold_range_dual_prf(0.0, 20e3, 60e3, 50e3, **limits)[0])
    120000.0
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
    result: NDArray[np.float64] = np.mod(value - lower_arr, period) + lower_arr
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
