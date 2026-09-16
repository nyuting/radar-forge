"""Linear-FM (chirp) waveform generation and the FMCW deramp relations.

The linear frequency-modulated chirp is the waveform almost every modern radar
in this library will use: it decouples range resolution from pulse length, so a
long pulse can carry the energy of its duration while resolving like its
bandwidth. This module generates the waveform itself and provides the four
scalar relations that connect its parameters to what a receiver measures.

The FMCW convention here is *deramp-on-receive*: the echo is mixed with the
transmitted chirp, leaving a beat tone whose frequency is proportional to
target range. :func:`beat_frequency_hz` and :func:`range_from_beat_frequency_m`
are that mapping and its inverse.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §4.4 (linear FM), §8.3 (pulse compression).
.. [2] L. Harrison and G. Andrews, *Introduction to Radar Using Python and
       MATLAB*, Artech House, 2020, ch. 4.
.. [3] G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014,
       §2.5 (FMCW deramp and the beat-frequency relation).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS

__all__ = [
    "beat_frequency_hz",
    "lfm_chirp",
    "range_from_beat_frequency_m",
    "range_resolution_m",
    "sweep_rate_hzps",
]


def range_resolution_m(bandwidth_hz: ArrayLike) -> NDArray[np.float64]:
    r"""Return the range resolution of a chirp of the given bandwidth, in metres.

    .. math:: \Delta R = \frac{c}{2B}

    Resolution is set by bandwidth alone — not by pulse length, carrier
    frequency, or transmitted power. This is the entire reason to chirp [1]_.

    Parameters
    ----------
    bandwidth_hz : array_like
        Swept bandwidth :math:`B`, in hertz. Must be strictly positive.

    Returns
    -------
    numpy.ndarray
        Range resolution in metres, of the shape of ``bandwidth_hz``.

    Raises
    ------
    ValueError
        If any bandwidth is non-positive.

    Examples
    --------
    A 1 GHz automotive sweep resolves to about 15 cm:

    >>> import numpy as np
    >>> bool(np.isclose(range_resolution_m(1e9), 0.1498962290, rtol=1e-9))
    True
    """
    bandwidth_arr = np.asarray(bandwidth_hz, dtype=np.float64)
    if np.any(bandwidth_arr <= 0.0):
        msg = "bandwidth_hz must be strictly positive; resolution diverges at B = 0."
        raise ValueError(msg)
    result: NDArray[np.float64] = SPEED_OF_LIGHT_MPS / (2.0 * bandwidth_arr)
    return result


def sweep_rate_hzps(bandwidth_hz: ArrayLike, chirp_duration_s: ArrayLike) -> NDArray[np.float64]:
    r"""Return the chirp slope :math:`\alpha = B / T`, in hertz per second.

    Parameters
    ----------
    bandwidth_hz : array_like
        Swept bandwidth :math:`B`, in hertz. Must be strictly positive.
    chirp_duration_s : array_like
        Sweep duration :math:`T`, in seconds. Must be strictly positive.

    Returns
    -------
    numpy.ndarray
        Sweep rate in Hz/s, of the broadcast shape of the inputs.

    Raises
    ------
    ValueError
        If any bandwidth or duration is non-positive.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isclose(sweep_rate_hzps(1e9, 40e-6), 2.5e13, rtol=1e-12))
    True
    """
    bandwidth_arr = np.asarray(bandwidth_hz, dtype=np.float64)
    chirp_duration_arr = np.asarray(chirp_duration_s, dtype=np.float64)
    if np.any(bandwidth_arr <= 0.0):
        msg = "bandwidth_hz must be strictly positive."
        raise ValueError(msg)
    if np.any(chirp_duration_arr <= 0.0):
        msg = "chirp_duration_s must be strictly positive."
        raise ValueError(msg)
    result: NDArray[np.float64] = bandwidth_arr / chirp_duration_arr
    return result


def beat_frequency_hz(
    range_m: ArrayLike,
    bandwidth_hz: ArrayLike,
    chirp_duration_s: ArrayLike,
) -> NDArray[np.float64]:
    r"""Return the deramped beat frequency of a target at ``range_m``, in hertz.

    .. math:: f_b = \alpha \tau = \frac{B}{T} \cdot \frac{2R}{c}

    The round-trip delay :math:`\tau = 2R/c` displaces the echo along the sweep,
    so mixing it with the transmitted chirp leaves a tone linear in range [3]_.
    This is the stationary-target form; a moving target adds a Doppler term that
    range-Doppler processing separates across chirps.

    Parameters
    ----------
    range_m : array_like
        Target range :math:`R`, in metres. Must be non-negative.
    bandwidth_hz : array_like
        Swept bandwidth :math:`B`, in hertz. Must be strictly positive.
    chirp_duration_s : array_like
        Sweep duration :math:`T`, in seconds. Must be strictly positive.

    Returns
    -------
    numpy.ndarray
        Beat frequency in hertz, of the broadcast shape of the inputs.

    Raises
    ------
    ValueError
        If any range is negative, or any bandwidth or duration is non-positive.

    See Also
    --------
    range_from_beat_frequency_m : The inverse mapping.

    Examples
    --------
    >>> import numpy as np
    >>> f_b = beat_frequency_hz(range_m=42.0, bandwidth_hz=1e9, chirp_duration_s=40e-6)
    >>> bool(np.isclose(range_from_beat_frequency_m(f_b, 1e9, 40e-6), 42.0, rtol=1e-12))
    True
    """
    range_arr = np.asarray(range_m, dtype=np.float64)
    if np.any(range_arr < 0.0):
        msg = "range_m must be non-negative."
        raise ValueError(msg)
    round_trip_delay_s = 2.0 * range_arr / SPEED_OF_LIGHT_MPS
    result: NDArray[np.float64] = (
        sweep_rate_hzps(bandwidth_hz, chirp_duration_s) * round_trip_delay_s
    )
    return result


def range_from_beat_frequency_m(
    beat_frequency_hz: ArrayLike,
    bandwidth_hz: ArrayLike,
    chirp_duration_s: ArrayLike,
) -> NDArray[np.float64]:
    r"""Return the range implied by a deramped beat frequency, in metres.

    .. math:: R = \frac{c\, f_b}{2\alpha} = \frac{c\, f_b\, T}{2B}

    The inverse of :func:`beat_frequency_hz`, and the step that turns an FFT bin
    index into a range.

    Parameters
    ----------
    beat_frequency_hz : array_like
        Measured beat frequency :math:`f_b`, in hertz. Must be non-negative.
    bandwidth_hz : array_like
        Swept bandwidth :math:`B`, in hertz. Must be strictly positive.
    chirp_duration_s : array_like
        Sweep duration :math:`T`, in seconds. Must be strictly positive.

    Returns
    -------
    numpy.ndarray
        Range in metres, of the broadcast shape of the inputs.

    Raises
    ------
    ValueError
        If any beat frequency is negative, or any bandwidth or duration is
        non-positive.

    See Also
    --------
    beat_frequency_hz : The forward mapping.
    """
    beat_arr = np.asarray(beat_frequency_hz, dtype=np.float64)
    if np.any(beat_arr < 0.0):
        msg = "beat_frequency_hz must be non-negative."
        raise ValueError(msg)
    result: NDArray[np.float64] = (
        SPEED_OF_LIGHT_MPS * beat_arr / (2.0 * sweep_rate_hzps(bandwidth_hz, chirp_duration_s))
    )
    return result


def lfm_chirp(
    bandwidth_hz: float,
    chirp_duration_s: float,
    sample_rate_hz: float,
    *,
    start_frequency_hz: float = 0.0,
    amplitude_linear: float = 1.0,
    up_sweep: bool = True,
) -> NDArray[np.complex128]:
    r"""Return one complex-baseband linear-FM chirp.

    The instantaneous frequency sweeps linearly across ``bandwidth_hz`` over
    ``chirp_duration_s``, giving the analytic signal

    .. math::

        s(t) = A \exp\!\left[\, j 2\pi \left(f_0 t
               \pm \tfrac{1}{2}\alpha t^2 \right)\right],
        \qquad 0 \le t < T,\; \alpha = B/T

    Sampling is half-open on :math:`[0, T)` — the sample at :math:`t = T` is the
    first sample of the *next* chirp, so including it would double-count one
    sample per sweep in a coherent processing interval.

    Parameters
    ----------
    bandwidth_hz : float
        Swept bandwidth :math:`B`, in hertz. Must be strictly positive.
    chirp_duration_s : float
        Sweep duration :math:`T`, in seconds. Must be strictly positive and long
        enough to hold at least one sample at ``sample_rate_hz``.
    sample_rate_hz : float
        Complex sampling rate :math:`f_s`, in hertz. Must be at least
        ``bandwidth_hz``: a complex-baseband signal of bandwidth :math:`B` needs
        :math:`f_s \ge B`, not :math:`2B`.
    start_frequency_hz : float, optional
        Baseband frequency :math:`f_0` at :math:`t = 0`, in hertz. Default 0.0.
        Pass ``-bandwidth_hz / 2`` for a sweep centred on zero baseband.
    amplitude_linear : float, optional
        Peak amplitude :math:`A`, a dimensionless scale on the unit-modulus
        waveform. Default 1.0.
    up_sweep : bool, optional
        If True (default) frequency increases with time; if False it decreases.

    Returns
    -------
    numpy.ndarray
        Complex chirp samples of shape ``(n_samples,)``, where
        ``n_samples = round(chirp_duration_s * sample_rate_hz)``.

    Raises
    ------
    ValueError
        If any parameter is non-positive, if ``sample_rate_hz`` is below
        ``bandwidth_hz``, or if the sweep is too short to yield one sample.

    Notes
    -----
    The time-bandwidth product :math:`BT` is the pulse-compression gain: matched
    filtering this waveform narrows it by a factor of :math:`BT` and raises the
    peak SNR by :math:`10\log_{10}(BT)` dB [1]_.

    Examples
    --------
    >>> import numpy as np
    >>> chirp = lfm_chirp(bandwidth_hz=1e6, chirp_duration_s=1e-3, sample_rate_hz=2e6)
    >>> chirp.shape
    (2000,)
    >>> bool(np.allclose(np.abs(chirp), 1.0))
    True
    """
    if bandwidth_hz <= 0.0:
        msg = "bandwidth_hz must be strictly positive."
        raise ValueError(msg)
    if chirp_duration_s <= 0.0:
        msg = "chirp_duration_s must be strictly positive."
        raise ValueError(msg)
    if sample_rate_hz <= 0.0:
        msg = "sample_rate_hz must be strictly positive."
        raise ValueError(msg)
    if sample_rate_hz < bandwidth_hz:
        msg = (
            f"sample_rate_hz ({sample_rate_hz:g}) must be at least bandwidth_hz "
            f"({bandwidth_hz:g}); the sweep would alias."
        )
        raise ValueError(msg)

    n_samples: int = round(chirp_duration_s * sample_rate_hz)
    if n_samples < 1:
        msg = (
            f"chirp_duration_s ({chirp_duration_s:g}) is too short to yield a sample at "
            f"sample_rate_hz ({sample_rate_hz:g})."
        )
        raise ValueError(msg)

    # Half-open [0, T): t = T belongs to the next chirp.
    time_s = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    sweep_rate = bandwidth_hz / chirp_duration_s
    if not up_sweep:
        sweep_rate = -sweep_rate

    phase_rad = 2.0 * np.pi * (start_frequency_hz * time_s + 0.5 * sweep_rate * time_s**2)
    result: NDArray[np.complex128] = amplitude_linear * np.exp(1j * phase_rad)
    return result
