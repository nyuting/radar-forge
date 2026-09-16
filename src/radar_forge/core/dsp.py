"""Range and Doppler processing: pulse compression, the two FFTs, and MTI.

This module is the chain that turns a cube of complex baseband samples into a
range-Doppler map. It is deliberately *axis-agnostic*: every function takes an
explicit ``axis`` rather than assuming a layout, so a cube from any simulator
drops in. The canonical layout the docstrings refer to is

``(n_pulses, n_samples, n_rx)`` — slow time on axis 0, fast time on axis 1.

"Fast time" is the sample index within one chirp, and maps to range. "Slow
time" is the chirp index within a coherent processing interval, and maps to
Doppler. The two are processed by the same operation, an FFT, applied to
different axes; that they are the same operation is the point of the module.

One asymmetry is deliberate and worth stating twice, because it surprises
everyone the first time: :func:`range_fft` does **not** shift its output, since
bin 0 is zero range and negative range is meaningless, while
:func:`doppler_fft` **does**, since zero Doppler belongs in the middle with
closing and opening targets either side of it.

Sign convention, per ``spec/structure.md`` §D5: **closing velocity is positive**.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §8.3 (pulse compression), §14 (Doppler processing),
       §17.3 (MTI pulse cancellers).
.. [2] L. Harrison and G. Andrews, *Introduction to Radar Using Python and
       MATLAB*, Artech House, 2020, ch. 4-5.
.. [3] G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014,
       §2.5 (FMCW deramp and the beat-frequency relation).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve

from radar_forge.core.waveforms import range_from_beat_frequency_m
from radar_forge.core.windows import apply_taper

__all__ = [
    "doppler_bin_centers_mps",
    "doppler_fft",
    "matched_filter",
    "mti_filter",
    "range_bin_centers_m",
    "range_doppler_map",
    "range_fft",
]

# Coefficients of the n-pulse MTI cancellers: binomial, with alternating signs.
# Two entries only — beyond three pulses the notch is wide enough to reject real
# targets, and a proper Doppler filter bank is the right tool instead.
_MTI_COEFFICIENTS: dict[int, list[float]] = {
    2: [1.0, -1.0],
    3: [1.0, -2.0, 1.0],
}


def matched_filter(
    samples: ArrayLike,
    reference: ArrayLike,
    *,
    axis: int = -1,
) -> NDArray[np.complex128]:
    r"""Compress ``samples`` against a ``reference`` waveform.

    The matched filter is correlation with the transmitted waveform, which for
    complex baseband is convolution with its conjugate time-reverse:

    .. math:: y[n] = \sum_{k} x[k]\, h^{*}[k - n]

    Applied to a linear-FM chirp this is *pulse compression*: a pulse of length
    :math:`T` and bandwidth :math:`B` collapses to a mainlobe of width
    :math:`\approx 1/B`, a narrowing by the time-bandwidth product :math:`BT`,
    and the peak SNR rises by :math:`10\log_{10}(BT)` dB [1]_.

    Parameters
    ----------
    samples : array_like
        Received complex baseband, of any shape. In the canonical cube layout
        ``(n_pulses, n_samples, n_rx)`` this is compressed along fast time.
    reference : array_like
        Transmitted waveform of shape ``(n_reference,)``, typically from
        :func:`radar_forge.core.lfm_chirp`. Not conjugated or reversed by the
        caller — this function does both.
    axis : int, optional
        Axis of ``samples`` to compress along. Default -1.

    Returns
    -------
    numpy.ndarray
        Compressed output, complex. Same shape as ``samples`` except along
        ``axis``, which has length ``n_samples + n_reference - 1``: the full
        linear correlation, so **zero lag sits at index ``n_reference - 1``**,
        not at index 0. Slice from there to recover a range profile aligned
        with the start of the record.

    Raises
    ------
    ValueError
        If ``reference`` is not one-dimensional or is empty, or if ``axis`` is
        out of range.

    See Also
    --------
    radar_forge.core.lfm_chirp : Generates the reference waveform.
    range_fft : The deramped-FMCW alternative to explicit compression.

    Notes
    -----
    Implemented with :func:`scipy.signal.fftconvolve`, which zero-pads to a full
    *linear* convolution. A hand-written ``ifft(fft(x) * conj(fft(h)))`` computes
    a *circular* correlation instead, wrapping energy from the far end of the
    record around to zero range — a silent error that looks like a near-range
    ghost target. The padding is the whole difference, and it is easy to omit.

    The transform of ``reference`` is recomputed on every call. For a fixed
    waveform compressed against many cubes it would be worth caching, at the
    cost of a stateful API; this library prefers the readable version until a
    profile says otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from radar_forge.core import lfm_chirp
    >>> chirp = lfm_chirp(bandwidth_hz=10e6, chirp_duration_s=10e-6, sample_rate_hz=20e6)
    >>> compressed = matched_filter(chirp, chirp)
    >>> int(np.argmax(np.abs(compressed))) == chirp.size - 1
    True
    """
    samples_arr = np.asarray(samples, dtype=np.complex128)
    reference_arr = np.asarray(reference, dtype=np.complex128)

    if reference_arr.ndim != 1:
        msg = f"reference must be one-dimensional, got shape {reference_arr.shape}."
        raise ValueError(msg)
    if reference_arr.size == 0:
        msg = "reference must not be empty."
        raise ValueError(msg)
    axis = _normalize_axis(axis, samples_arr.ndim)

    # Correlation is convolution with the conjugate time-reverse of the template.
    template = np.conjugate(reference_arr[::-1])
    filter_shape = [1] * samples_arr.ndim
    filter_shape[axis] = template.size

    result: NDArray[np.complex128] = np.asarray(
        fftconvolve(samples_arr, template.reshape(filter_shape), mode="full", axes=axis),
        dtype=np.complex128,
    )
    return result


def range_fft(
    samples: ArrayLike,
    *,
    n_fft: int | None = None,
    window: ArrayLike | None = None,
    axis: int = -1,
) -> NDArray[np.complex128]:
    """Transform fast time into range bins.

    For a deramped FMCW receiver each target contributes a beat tone whose
    frequency is proportional to range [3]_, so the FFT along fast time *is*
    the range profile.

    The output is **not** ``fftshift``-ed: bin 0 is zero beat frequency, hence
    zero range, and bins count upward in range from there. Only the first half
    of the output is physically meaningful for a real deramped signal; see
    :func:`range_bin_centers_m`.

    Parameters
    ----------
    samples : array_like
        Complex baseband, typically of shape ``(n_pulses, n_samples, n_rx)``.
    n_fft : int, optional
        Transform length. Default is the length of ``samples`` along ``axis``.
        A larger value zero-pads, which interpolates the profile onto a finer
        grid — it does **not** improve resolution, which bandwidth alone sets.
    window : array_like, optional
        Amplitude taper of shape ``(n,)`` matching ``samples`` along ``axis``,
        applied before the transform. Default None, i.e. rectangular. See
        :func:`radar_forge.core.windows.taper`.
    axis : int, optional
        Fast-time axis. Default -1.

    Returns
    -------
    numpy.ndarray
        Complex range profiles, of the shape of ``samples`` except that
        ``axis`` has length ``n_fft``.

    Raises
    ------
    ValueError
        If ``n_fft`` is less than one or shorter than ``samples`` along
        ``axis``, if ``window`` does not match, or if ``axis`` is out of range.

    See Also
    --------
    doppler_fft : The same operation on slow time, but shifted.
    range_bin_centers_m : Maps the returned bins to metres.

    Examples
    --------
    A tone placed exactly on bin 3 lands exactly on bin 3:

    >>> import numpy as np
    >>> n = 64
    >>> tone = np.exp(2j * np.pi * 3 * np.arange(n) / n)
    >>> int(np.argmax(np.abs(range_fft(tone))))
    3
    """
    samples_arr, n_fft, axis = _prepare_transform(samples, n_fft, window, axis)
    result: NDArray[np.complex128] = np.fft.fft(samples_arr, n=n_fft, axis=axis)
    return result


def doppler_fft(
    profiles: ArrayLike,
    *,
    n_fft: int | None = None,
    window: ArrayLike | None = None,
    axis: int = 0,
) -> NDArray[np.complex128]:
    """Transform slow time into Doppler bins.

    A target moving in range advances the phase of its beat tone by a fixed
    increment from chirp to chirp, so the FFT across chirps resolves targets by
    radial velocity even when they share a range bin [1]_.

    The output **is** ``fftshift``-ed, so zero Doppler sits at the centre bin,
    opening targets below it and closing targets above it. This is the opposite
    of :func:`range_fft`, which leaves zero range at bin 0.

    Parameters
    ----------
    profiles : array_like
        Complex range profiles, typically the output of :func:`range_fft` with
        shape ``(n_pulses, n_range_bins, n_rx)``.
    n_fft : int, optional
        Transform length. Default is the length along ``axis``. Zero-padding
        interpolates the Doppler axis; it does not add velocity resolution,
        which only a longer dwell buys.
    window : array_like, optional
        Amplitude taper of shape ``(n,)`` matching ``profiles`` along
        ``axis``. Default None. Tapering matters more here than in range: a
        strong stationary clutter return leaks across the whole Doppler axis
        through an untapered FFT's sidelobes and masks slow-moving targets.
    axis : int, optional
        Slow-time axis. Default 0.

    Returns
    -------
    numpy.ndarray
        Complex Doppler spectra, of the shape of ``profiles`` except that
        ``axis`` has length ``n_fft``, with zero Doppler at index ``n_fft // 2``.

    Raises
    ------
    ValueError
        If ``n_fft`` is less than one or shorter than ``profiles`` along
        ``axis``, if ``window`` does not match, or if ``axis`` is out of range.

    See Also
    --------
    doppler_bin_centers_mps : Maps the returned bins to velocities.

    Examples
    --------
    A phase ramp of zero slope is stationary, so it peaks at the centre bin:

    >>> import numpy as np
    >>> stationary = np.ones(16, dtype=np.complex128)
    >>> int(np.argmax(np.abs(doppler_fft(stationary, axis=0))))
    8
    """
    profiles_arr, n_fft, axis = _prepare_transform(profiles, n_fft, window, axis)
    spectrum = np.fft.fft(profiles_arr, n=n_fft, axis=axis)
    result: NDArray[np.complex128] = np.fft.fftshift(spectrum, axes=axis)
    return result


def range_doppler_map(
    samples: ArrayLike,
    *,
    fast_time_window: ArrayLike | None = None,
    slow_time_window: ArrayLike | None = None,
    n_range_fft: int | None = None,
    n_doppler_fft: int | None = None,
    fast_time_axis: int = -1,
    slow_time_axis: int = 0,
) -> NDArray[np.complex128]:
    """Run the full range-Doppler chain over a baseband cube.

    Equivalent to :func:`range_fft` along ``fast_time_axis`` followed by
    :func:`doppler_fft` along ``slow_time_axis``. The two transforms commute —
    they act on different axes — so the order is a matter of convention, not
    correctness.

    Parameters
    ----------
    samples : array_like
        Complex baseband cube, canonically ``(n_pulses, n_samples, n_rx)``.
    fast_time_window, slow_time_window : array_like, optional
        Amplitude tapers for the respective axes. Default None.
    n_range_fft, n_doppler_fft : int, optional
        Transform lengths for the respective axes. Default is no zero-padding.
    fast_time_axis : int, optional
        Axis carrying samples within a chirp. Default -1.
    slow_time_axis : int, optional
        Axis carrying chirps within the dwell. Default 0.

    Returns
    -------
    numpy.ndarray
        Complex range-Doppler map, range unshifted along ``fast_time_axis`` and
        Doppler shifted to centre along ``slow_time_axis``. Take
        ``20 * np.log10(np.abs(...))`` for the usual dB image.

    Raises
    ------
    ValueError
        If the two axes are the same, if either transform length is invalid, or
        if either window does not match its axis.

    Examples
    --------
    >>> import numpy as np
    >>> cube = np.ones((16, 32), dtype=np.complex128)
    >>> range_doppler_map(cube, fast_time_axis=1, slow_time_axis=0).shape
    (16, 32)
    """
    samples_arr = np.asarray(samples, dtype=np.complex128)
    fast_axis = _normalize_axis(fast_time_axis, samples_arr.ndim)
    slow_axis = _normalize_axis(slow_time_axis, samples_arr.ndim)
    if fast_axis == slow_axis:
        msg = (
            f"fast_time_axis and slow_time_axis both resolve to axis {fast_axis}; "
            f"range and Doppler must be taken over different axes."
        )
        raise ValueError(msg)

    profiles = range_fft(samples_arr, n_fft=n_range_fft, window=fast_time_window, axis=fast_axis)
    result: NDArray[np.complex128] = doppler_fft(
        profiles, n_fft=n_doppler_fft, window=slow_time_window, axis=slow_axis
    )
    return result


def range_bin_centers_m(
    n_bins: int,
    bandwidth_hz: float,
    chirp_duration_s: float,
    sample_rate_hz: float,
) -> NDArray[np.float64]:
    r"""Return the range of each unshifted FFT bin, in metres.

    Bin :math:`k` of an ``n_bins``-point FFT of a deramped signal sampled at
    :math:`f_s` holds beat frequency :math:`f_b = k f_s / N`, and
    :func:`radar_forge.core.range_from_beat_frequency_m` turns that into a range.

    Parameters
    ----------
    n_bins : int
        Transform length :math:`N` used for :func:`range_fft`. Must be at least
        one.
    bandwidth_hz : float
        Swept bandwidth :math:`B`, in hertz. Must be strictly positive.
    chirp_duration_s : float
        Sweep duration :math:`T`, in seconds. Must be strictly positive.
    sample_rate_hz : float
        Complex sample rate :math:`f_s`, in hertz. Must be strictly positive.

    Returns
    -------
    numpy.ndarray
        Range of each bin in metres, shape ``(n_bins,)``, increasing from 0.

    Raises
    ------
    ValueError
        If ``n_bins`` is less than one, or any parameter is non-positive.

    Notes
    -----
    Only the first ``n_bins // 2`` entries are unambiguous: beyond that a
    complex-baseband FFT is reporting negative frequencies, which correspond to
    no physical range. The maximum unambiguous range is
    :math:`R_{max} = c f_s T / (4 B)`.

    Examples
    --------
    >>> import numpy as np
    >>> bins = range_bin_centers_m(8, bandwidth_hz=1e9, chirp_duration_s=40e-6, sample_rate_hz=1e6)
    >>> bool(np.isclose(bins[0], 0.0))
    True
    >>> bool(np.all(np.diff(bins) > 0.0))
    True
    """
    if n_bins < 1:
        msg = f"n_bins ({n_bins}) must be at least one."
        raise ValueError(msg)
    if sample_rate_hz <= 0.0:
        msg = "sample_rate_hz must be strictly positive."
        raise ValueError(msg)

    beat_hz = np.arange(n_bins, dtype=np.float64) * sample_rate_hz / n_bins
    result: NDArray[np.float64] = range_from_beat_frequency_m(
        beat_hz, bandwidth_hz, chirp_duration_s
    )
    return result


def doppler_bin_centers_mps(
    n_bins: int,
    pulse_repetition_interval_s: float,
    wavelength_m: float,
) -> NDArray[np.float64]:
    r"""Return the radial velocity of each shifted Doppler bin, in metres/second.

    .. math:: v = \frac{\lambda f_d}{2}

    The bins match the ``fftshift``-ed output of :func:`doppler_fft`, so zero
    velocity is at index ``n_bins // 2`` and the array increases monotonically.

    Parameters
    ----------
    n_bins : int
        Transform length used for :func:`doppler_fft`. Must be at least one.
    pulse_repetition_interval_s : float
        Time between chirps :math:`T_{PRI}`, in seconds. Must be strictly
        positive.
    wavelength_m : float
        Carrier wavelength :math:`\lambda`, in metres. Must be strictly
        positive.

    Returns
    -------
    numpy.ndarray
        Radial velocity of each bin in m/s, shape ``(n_bins,)``, increasing.
        **Positive is closing**, per the library's sign convention.

    Raises
    ------
    ValueError
        If ``n_bins`` is less than one, or either parameter is non-positive.

    Notes
    -----
    The span is the unambiguous velocity interval
    :math:`\pm \lambda / (4 T_{PRI})`: a target faster than that folds back into
    the map at a false velocity, indistinguishable from a slow target, which is
    why a fast closing target can appear to be opening.

    Examples
    --------
    >>> import numpy as np
    >>> v = doppler_bin_centers_mps(8, pulse_repetition_interval_s=50e-6, wavelength_m=3.9e-3)
    >>> bool(np.isclose(v[4], 0.0))
    True
    >>> bool(np.all(np.diff(v) > 0.0))
    True
    """
    if n_bins < 1:
        msg = f"n_bins ({n_bins}) must be at least one."
        raise ValueError(msg)
    if pulse_repetition_interval_s <= 0.0:
        msg = "pulse_repetition_interval_s must be strictly positive."
        raise ValueError(msg)
    if wavelength_m <= 0.0:
        msg = "wavelength_m must be strictly positive."
        raise ValueError(msg)

    doppler_hz = np.fft.fftshift(np.fft.fftfreq(n_bins, d=pulse_repetition_interval_s))
    result: NDArray[np.float64] = wavelength_m * doppler_hz / 2.0
    return result


def mti_filter(
    samples: ArrayLike,
    *,
    n_pulses: int = 2,
    axis: int = 0,
) -> NDArray[np.complex128]:
    r"""Cancel stationary returns by differencing successive chirps.

    The :math:`n`-pulse canceller subtracts consecutive chirps with binomial
    coefficients [1]_ — ``[1, -1]`` for two pulses, ``[1, -2, 1]`` for three.
    Anything that did not move between chirps subtracts to zero, so ground
    clutter vanishes while moving targets survive.

    Parameters
    ----------
    samples : array_like
        Complex baseband, canonically ``(n_pulses, n_samples, n_rx)`` with slow
        time on axis 0.
    n_pulses : int, optional
        Canceller order, 2 (single) or 3 (double). Default 2. The double
        canceller nulls clutter harder and over a wider band, at the cost of a
        wider notch that also rejects genuinely slow targets.
    axis : int, optional
        Slow-time axis to difference along. Default 0.

    Returns
    -------
    numpy.ndarray
        Filtered samples, complex, of the shape of ``samples`` except that
        ``axis`` is **shortened by ``n_pulses - 1``**: differencing consumes
        chirps. A 128-chirp dwell yields 127 after a single canceller.

    Raises
    ------
    ValueError
        If ``n_pulses`` is not 2 or 3, if ``axis`` is out of range, or if
        ``samples`` has fewer than ``n_pulses`` entries along ``axis``.

    Notes
    -----
    The frequency response of the single canceller is
    :math:`|H(f_d)| = 2|\sin(\pi f_d T_{PRI})|`, which is zero not only at zero
    Doppler but at every multiple of the PRF. Those are the **blind speeds**:
    a target whose phase advances by exactly one full turn between chirps is
    indistinguishable from one that did not move at all, and is cancelled along
    with the clutter. Staggering the PRI is the standard cure, and is not
    implemented here.

    Examples
    --------
    A stationary return cancels completely:

    >>> import numpy as np
    >>> stationary = np.ones((4, 3), dtype=np.complex128)
    >>> bool(np.allclose(mti_filter(stationary), 0.0))
    True
    >>> mti_filter(stationary).shape
    (3, 3)
    """
    samples_arr = np.asarray(samples, dtype=np.complex128)
    if n_pulses not in _MTI_COEFFICIENTS:
        accepted = ", ".join(str(key) for key in sorted(_MTI_COEFFICIENTS))
        msg = f"n_pulses ({n_pulses}) must be one of {accepted}."
        raise ValueError(msg)
    axis = _normalize_axis(axis, samples_arr.ndim)

    n_available = samples_arr.shape[axis]
    if n_available < n_pulses:
        msg = (
            f"samples has {n_available} entries along axis {axis}, fewer than the "
            f"{n_pulses} an {n_pulses}-pulse canceller needs."
        )
        raise ValueError(msg)

    # Accumulate the weighted taps by slicing rather than convolving: it makes
    # the binomial structure visible and avoids the edge effects a 'full'
    # convolution would introduce at both ends of the dwell. A Python loop over
    # the two or three coefficients is unavoidable and costs nothing; each body
    # is a whole-array operation.
    #
    # The slices are basic indexing, so each tap is a view and only the products
    # allocate. np.take with a range() would be fancy indexing, which copies the
    # band first -- bit-identical output, and measured at about half the speed.
    n_output = n_available - (n_pulses - 1)
    coefficients = _MTI_COEFFICIENTS[n_pulses]
    accumulator = coefficients[0] * _slice_along(samples_arr, 0, n_output, axis)
    for offset, coefficient in enumerate(coefficients[1:], start=1):
        accumulator = accumulator + coefficient * _slice_along(
            samples_arr, offset, offset + n_output, axis
        )

    result: NDArray[np.complex128] = accumulator
    return result


def _slice_along(
    values: NDArray[np.complex128], start: int, stop: int, axis: int
) -> NDArray[np.complex128]:
    """Return ``values[..., start:stop, ...]`` sliced along ``axis``, as a view."""
    index: list[slice] = [slice(None)] * values.ndim
    index[axis] = slice(start, stop)
    return values[tuple(index)]


def _prepare_transform(
    samples: ArrayLike,
    n_fft: int | None,
    window: ArrayLike | None,
    axis: int,
) -> tuple[NDArray[np.complex128], int, int]:
    """Validate, taper and resolve the transform length shared by the two FFTs."""
    samples_arr = np.asarray(samples, dtype=np.complex128)
    axis = _normalize_axis(axis, samples_arr.ndim)
    n_along_axis = samples_arr.shape[axis]

    if window is not None:
        samples_arr = apply_taper(samples_arr, window, axis=axis)

    if n_fft is None:
        return samples_arr, n_along_axis, axis
    if n_fft < 1:
        msg = f"n_fft ({n_fft}) must be at least one."
        raise ValueError(msg)
    if n_fft < n_along_axis:
        msg = (
            f"n_fft ({n_fft}) is shorter than samples along axis {axis} "
            f"({n_along_axis}); that would discard samples rather than zero-pad. "
            f"Slice the input explicitly if truncation is intended."
        )
        raise ValueError(msg)
    return samples_arr, n_fft, axis


def _normalize_axis(axis: int, ndim: int) -> int:
    """Return ``axis`` as a non-negative index into an ``ndim``-dimensional array."""
    if not -ndim <= axis < ndim:
        msg = f"axis ({axis}) is out of range for an array of {ndim} dimension(s)."
        raise ValueError(msg)
    return axis % ndim
