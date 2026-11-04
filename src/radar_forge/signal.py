r"""Baseband signals: the transmitted waveform, and propagation paths to complex IQ.

:func:`lfm_chirp` is the waveform, :func:`ambiguity_function` what it can
resolve, and the FMCW relations (:func:`beat_frequency_hz` and its inverse) map
a bistatic range to the tone a deramping receiver hears. The generators turn a
:class:`~radar_forge.propagation.PropagationPaths` into a ``(n_pulses,
n_samples)`` cube, monostatic and bistatic alike:

:func:`fmcw_deramp_baseband`
    A continuous chirp, deramped against the transmitted sweep: range appears
    as a beat frequency.
:func:`pulsed_baseband`
    A pulse train sampled over each repetition interval, left for a matched
    filter: range appears as a delay, folded modulo the interval.

**Doppler sign.** A path's Doppler shift is positive closing
(:func:`~radar_forge.propagation.doppler_shift_hz`), and both cubes carry it so:
a closing target advances in slow-time phase. In an FMCW deramp the range and
Doppler signs are locked together, since both come from one delay; a **down**
sweep is the one that puts range at a positive beat frequency and Doppler on
its true sign, so the FMCW generator models a down sweep [1]_, [2]_.

**Approximations.** Stop-and-hop: a route's length is held across each pulse
and advanced between pulses [1]_, which the FMCW generator checks and warns
about. Constant Doppler rate within a coherent interval.

**Memory.** The per-path signal, ``(n_pulses, n_paths, n_samples)`` complex,
is summed over chunks of paths no larger than ``max_bytes``, so memory does not
grow with the path count. Paths are added in the same order whatever the
chunking, so the cube is bit-identical to an unchunked sum.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §2.6.2 (stop-and-hop), ch. 4 (the matched filter and
       the ambiguity function), §4.6.1 (linear FM), §4.6.5 (stretch processing).
.. [2] G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014,
       ch. 3 (FMCW deramp and the beat frequency).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.propagation import PropagationPaths, propagation_delay_s, propagation_phase_rad
from radar_forge.radar import Radar

__all__ = [
    "ambiguity_function",
    "beat_frequency_hz",
    "bistatic_range_from_beat_frequency_m",
    "fmcw_deramp_baseband",
    "lfm_chirp",
    "pulsed_baseband",
    "range_resolution_m",
    "sweep_rate_hzps",
    "thermal_noise",
]

# Above this fraction of a range bin of motion within one chirp, stop-and-hop smears the peak.
_STOP_AND_HOP_BIN_FRACTION = 0.25

_BYTES_PER_COMPLEX = 16


def range_resolution_m(bandwidth_hz: ArrayLike) -> NDArray[np.float64]:
    r"""Return the monostatic range resolution :math:`c/2B`, m, set by bandwidth alone [1]_."""
    result: NDArray[np.float64] = SPEED_OF_LIGHT_MPS / (
        2.0 * np.asarray(bandwidth_hz, dtype=np.float64)
    )
    return result


def sweep_rate_hzps(bandwidth_hz: ArrayLike, chirp_duration_s: ArrayLike) -> NDArray[np.float64]:
    r"""Return the chirp slope :math:`\alpha = B/T`, Hz/s, of the broadcast inputs [1]_."""
    result: NDArray[np.float64] = np.asarray(bandwidth_hz, dtype=np.float64) / np.asarray(
        chirp_duration_s, dtype=np.float64
    )
    return result


def beat_frequency_hz(
    bistatic_range_m: ArrayLike, bandwidth_hz: ArrayLike, chirp_duration_s: ArrayLike
) -> NDArray[np.float64]:
    r"""Return the deramped beat frequency of a route, :math:`f_b = \alpha R_b / c`, Hz [2]_.

    The delay :math:`R_b/c` displaces the echo along the sweep, leaving a tone
    linear in the bistatic range (m). No factor of two: monostatic, :math:`R_b = 2R`.
    Inputs broadcast together.

    Examples
    --------
    >>> round(float(beat_frequency_hz(30_000.0, 2.0e6, 1.0e-3)), 3)  # 2 GHz/s for 100 us
    200138.457
    """
    result: NDArray[np.float64] = (
        sweep_rate_hzps(bandwidth_hz, chirp_duration_s)
        * np.asarray(bistatic_range_m, dtype=np.float64)
        / SPEED_OF_LIGHT_MPS
    )
    return result


def bistatic_range_from_beat_frequency_m(
    beat_hz: ArrayLike, bandwidth_hz: ArrayLike, chirp_duration_s: ArrayLike
) -> NDArray[np.float64]:
    r"""Return the bistatic range behind a beat frequency, :math:`R_b = c f_b/\alpha`, m.

    The inverse of :func:`beat_frequency_hz` [2]_; halve it for a monostatic range.
    """
    result: NDArray[np.float64] = (
        SPEED_OF_LIGHT_MPS
        * np.asarray(beat_hz, dtype=np.float64)
        / sweep_rate_hzps(bandwidth_hz, chirp_duration_s)
    )
    return result


def lfm_chirp(
    bandwidth_hz: float,
    chirp_duration_s: float,
    sample_rate_hz: float,
    *,
    start_frequency_hz: float | None = None,
    amplitude_linear: float = 1.0,
    sweep_direction: Literal["up", "down"] = "up",
) -> NDArray[np.complex128]:
    r"""Return one complex-baseband linear-FM chirp.

    .. math::

        s(t) = A \exp\left[j 2\pi \left(f_0 t \pm \tfrac{1}{2}\alpha t^2\right)\right],
        \qquad 0 \le t < T,\quad \alpha = B/T

    Sampled on the half-open :math:`[0, T)`: the sample at :math:`T` starts the
    next chirp. Matched filtering compresses it by the time-bandwidth product
    :math:`BT` [1]_.

    Parameters
    ----------
    bandwidth_hz : float
        Swept bandwidth :math:`B`, Hz.
    chirp_duration_s : float
        Sweep duration :math:`T`, s.
    sample_rate_hz : float
        Complex sample rate, Hz, at least ``bandwidth_hz``.
    start_frequency_hz : float, optional
        Frequency :math:`f_0` at :math:`t = 0`, Hz. Default: the sweep is
        centred on zero, starting at :math:`-B/2` up or :math:`+B/2` down.
    amplitude_linear : float, optional
        Amplitude :math:`A` of the unit-modulus waveform. Default 1.0.
    sweep_direction : {"up", "down"}, optional
        Whether frequency rises or falls. Default ``"up"``.

    Returns
    -------
    numpy.ndarray
        Chirp samples, ``(n_samples,)``, ``n_samples = round(chirp_duration_s * sample_rate_hz)``.

    Raises
    ------
    ValueError
        If a parameter is non-positive, the sample rate is below the bandwidth,
        the sweep is too short for one sample, or the direction is unknown.

    Examples
    --------
    >>> chirp = lfm_chirp(1.0e6, 1.0e-3, 2.0e6)
    >>> chirp.shape, bool(np.allclose(np.abs(chirp), 1.0))
    ((2000,), True)
    """
    for name, value in (
        ("bandwidth_hz", bandwidth_hz),
        ("chirp_duration_s", chirp_duration_s),
        ("sample_rate_hz", sample_rate_hz),
    ):
        if value <= 0.0:
            msg = f"{name} must be strictly positive; got {value!r}."
            raise ValueError(msg)
    if sample_rate_hz < bandwidth_hz:
        msg = f"sample_rate_hz {sample_rate_hz:g} < bandwidth_hz {bandwidth_hz:g}: would alias."
        raise ValueError(msg)
    if sweep_direction not in ("up", "down"):
        msg = f"sweep_direction must be 'up' or 'down'; got {sweep_direction!r}."
        raise ValueError(msg)
    n_samples = round(chirp_duration_s * sample_rate_hz)
    if n_samples < 1:
        msg = f"chirp_duration_s {chirp_duration_s:g} is too short for one sample."
        raise ValueError(msg)

    sweep_rate = bandwidth_hz / chirp_duration_s
    if sweep_direction == "down":
        sweep_rate = -sweep_rate
    if start_frequency_hz is None:
        start_frequency_hz = -sweep_rate * chirp_duration_s / 2.0
    time_s = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    phase_rad = 2.0 * np.pi * (start_frequency_hz * time_s + 0.5 * sweep_rate * time_s**2)
    carrier: NDArray[np.complex128] = np.exp(1j * phase_rad)
    result: NDArray[np.complex128] = amplitude_linear * carrier
    return result


def thermal_noise(
    shape: tuple[int, ...], noise_power_w: float, *, rng: np.random.Generator
) -> NDArray[np.complex128]:
    r"""Draw circularly-symmetric complex Gaussian receiver noise.

    Real and imaginary parts each have variance :math:`N/2`, so the expected
    ``abs(noise) ** 2`` of the returned ``shape`` array is ``noise_power_w`` (W
    per sample, normally :attr:`~radar_forge.radar.Receiver.noise_power_per_sample_w`;
    zero is exactly silent). ``rng`` is a seeded generator.

    Raises
    ------
    ValueError
        If ``noise_power_w`` is negative.
    """
    if noise_power_w < 0.0:
        msg = f"noise_power_w must be non-negative; got {noise_power_w!r}."
        raise ValueError(msg)
    standard_deviation = np.sqrt(noise_power_w / 2.0)
    real_part = rng.normal(0.0, standard_deviation, size=shape)
    imaginary_part = rng.normal(0.0, standard_deviation, size=shape)
    return np.asarray(real_part + 1j * imaginary_part, dtype=np.complex128)


def fmcw_deramp_baseband(
    paths: PropagationPaths,
    radar: Radar,
    n_pulses: int,
    *,
    rng: np.random.Generator | None,
    max_bytes: int = 256 * 2**20,
) -> NDArray[np.complex128]:
    r"""Synthesise a deramped FMCW baseband cube.

    A **down** sweep from :math:`f_c + B/2` to :math:`f_c - B/2`, mixed as
    :math:`s_{rx} s_{tx}^*` [2]_:

    .. math::

        s[m, n] = \sum_p |a_p|\,
            e^{-j 2\pi (f_c + B/2)\, \tau_p[m]}\,
            e^{+j 2\pi \alpha\, \tau_p[m]\, t_n}

    with :math:`\tau_p[m] = R_{b,p}(mT_{PRI})/c` held across chirp :math:`m`
    and :math:`t_n = n/f_s`. The first factor becomes Doppler across slow time,
    the second the beat frequency :math:`\alpha\tau` across fast time. Read at
    a range bin, the slow-time phase advances at the carrier :math:`f_c`, so
    Doppler converts to rate with :math:`\lambda = c/f_c`.

    Parameters
    ----------
    paths : radar_forge.propagation.PropagationPaths
        The paths to synthesise.
    radar : radar_forge.radar.Radar
        The observing radar; must be FMCW.
    n_pulses : int
        Chirps in the coherent interval, at least one.
    rng : numpy.random.Generator or None
        Generator for receiver noise of
        :attr:`~radar_forge.radar.Receiver.noise_power_per_sample_w`, or
        ``None`` for a noiseless cube.
    max_bytes : int, optional
        Largest per-path intermediate to hold at once, bytes. Default 256 MiB.

    Returns
    -------
    numpy.ndarray
        Complex baseband, ``(n_pulses, n_samples_per_pri)``.

    Raises
    ------
    ValueError
        If ``n_pulses`` is below one, or the radar is not FMCW.

    Warns
    -----
    UserWarning
        If a path moves more than a quarter of a range bin within one chirp.
    """
    _check_dwell(radar, n_pulses, "fmcw")
    _warn_if_stop_and_hop_is_strained(paths, radar)
    transmitter = radar.transmitter

    bistatic_range_m = paths.bistatic_range_at_m(_slow_time_s(n_pulses, radar), radar.wavelength_m)
    delay_s = propagation_delay_s(bistatic_range_m)
    # The residual phase is referred to the sweep start, the frequency sent at t_n = 0.
    sweep_start_wavelength_m = SPEED_OF_LIGHT_MPS / (
        transmitter.carrier_frequency_hz + transmitter.bandwidth_hz / 2.0
    )
    residual_phase_rad = propagation_phase_rad(bistatic_range_m, sweep_start_wavelength_m)
    magnitude = np.abs(paths.amplitude_linear)
    fast_time_s = _fast_time_s(radar)

    def chunk_signal(chunk: slice) -> NDArray[np.complex128]:
        beat_phase_rad = (
            2.0 * np.pi * transmitter.sweep_rate_hzps * delay_s[:, chunk, None] * fast_time_s
        )
        signal: NDArray[np.complex128] = magnitude[None, chunk, None] * np.exp(
            1j * (residual_phase_rad[:, chunk, None] + beat_phase_rad)
        )
        return signal

    cube = _sum_over_paths(
        chunk_signal, paths.n_paths, (n_pulses, radar.n_samples_per_pri), max_bytes
    )
    return _add_noise(cube, radar, rng)


def pulsed_baseband(
    paths: PropagationPaths,
    radar: Radar,
    n_pulses: int,
    *,
    rng: np.random.Generator | None,
    max_bytes: int = 256 * 2**20,
) -> NDArray[np.complex128]:
    r"""Synthesise a pulsed baseband cube, before pulse compression.

    Each path contributes a delayed copy of the transmitted pulse :math:`p`,
    ``lfm_chirp(B, T, f_s)`` swept from :math:`-B/2` to :math:`+B/2` about the
    carrier:

    .. math::

        s[m, n] = \sum_p |a_p|\, p\left(t_n - \tau_p[m] \bmod T_{PRI}\right)\,
                  e^{-j 2\pi f_c \tau_p[m]}

    The delay is folded modulo the repetition interval, which is how a pulsed
    radar folds range; the carrier phase keeps the true delay, which is what
    keeps the Doppler right [1]_.

    Parameters, return and errors are those of :func:`fmcw_deramp_baseband`,
    except that the radar must be pulsed. The cube is not pulse-compressed.

    Notes
    -----
    Eclipsing is not modelled: an echo arriving while the transmitter is on is
    received rather than lost.
    """
    _check_dwell(radar, n_pulses, "pulsed")
    transmitter = radar.transmitter

    bistatic_range_m = paths.bistatic_range_at_m(_slow_time_s(n_pulses, radar), radar.wavelength_m)
    folded_delay_s = np.mod(
        propagation_delay_s(bistatic_range_m), transmitter.pulse_repetition_interval_s
    )
    residual_phase_rad = propagation_phase_rad(bistatic_range_m, radar.wavelength_m)
    magnitude = np.abs(paths.amplitude_linear)
    fast_time_s = _fast_time_s(radar)

    def chunk_signal(chunk: slice) -> NDArray[np.complex128]:
        elapsed_s = fast_time_s - folded_delay_s[:, chunk, None]
        within_pulse = (elapsed_s >= 0.0) & (elapsed_s < transmitter.chirp_duration_s)
        # The phase of lfm_chirp(B, T, f_s), the centred up sweep, at the elapsed time.
        envelope_phase_rad = (
            np.pi * elapsed_s * (transmitter.sweep_rate_hzps * elapsed_s - transmitter.bandwidth_hz)
        )
        signal: NDArray[np.complex128] = (
            magnitude[None, chunk, None]
            * within_pulse
            * np.exp(1j * (envelope_phase_rad + residual_phase_rad[:, chunk, None]))
        )
        return signal

    cube = _sum_over_paths(
        chunk_signal, paths.n_paths, (n_pulses, radar.n_samples_per_pri), max_bytes
    )
    return _add_noise(cube, radar, rng)


def ambiguity_function(
    waveform: ArrayLike,
    sample_rate_hz: float,
    *,
    prf_hz: float | None = None,
    delays_s: ArrayLike | None = None,
    dopplers_hz: ArrayLike | None = None,
    reference: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Return the normalised magnitude of a waveform's ambiguity function.

    The matched-filter response to an echo of ``waveform`` shifted by a Doppler
    :math:`f_D`, at delay :math:`\tau` [1]_:

    .. math::

        |\chi(\tau, f_D)| = \frac{\left|\sum_n s[n]\,e^{j 2\pi f_D n / f_s}\,
                            r^*[n - \tau f_s]\right|}{\sqrt{E_s E_r}}

    with :math:`r` the reference (``waveform`` itself unless given). A positive
    delay is a later echo. Normalised by the energies, so it is 1 at the origin
    of an auto-ambiguity and nowhere larger. An up-chirp's ridge runs along
    :math:`f_D = -\alpha\tau`.

    Parameters
    ----------
    waveform : array_like
        Complex samples, ``(n_samples,)``: one pulse, or a pulse train.
    sample_rate_hz : float
        Sample rate, Hz.
    prf_hz : float, optional
        Pulse repetition frequency of a pulse train. Only narrows the default
        Doppler grid to one repetition, :math:`[-\mathrm{PRF}/2, \mathrm{PRF}/2)`.
    delays_s : array_like, optional
        Delays to evaluate, s, rounded to whole samples. Default: every lag at
        which the two overlap.
    dopplers_hz : array_like, optional
        Doppler shifts to evaluate, Hz. Default: :math:`[-f_s/2, f_s/2)`, or one
        PRF, at a spacing of half the inverse waveform duration.
    reference : array_like, optional
        Filter reference, for a cross-ambiguity (a mismatched filter). Default: ``waveform``.

    Returns
    -------
    magnitude : numpy.ndarray
        :math:`|\chi|`, ``(n_dopplers, n_delays)``: Doppler down the rows, as
        MATLAB's ``ambgfun`` returns it.
    delays_s : numpy.ndarray
        The delays evaluated, s, ``(n_delays,)``.
    dopplers_hz : numpy.ndarray
        The Doppler shifts evaluated, Hz, ``(n_dopplers,)``.

    Raises
    ------
    ValueError
        If a waveform is not a non-empty vector or the sample rate is not positive.

    Notes
    -----
    One Doppler-shifted correlation per Doppler row, all rows in a single
    :func:`scipy.signal.fftconvolve` call, vectorised across delay and Doppler.
    Memory is ``n_dopplers * (n_samples + n_reference - 1)`` complex values.

    Examples
    --------
    >>> magnitude, delays_s, dopplers_hz = ambiguity_function(np.ones(8), 1.0, dopplers_hz=[0.0])
    >>> magnitude.shape, float(magnitude[0, delays_s == 0.0][0])
    ((1, 15), 1.0)
    """
    signal = np.asarray(waveform, dtype=np.complex128)
    matched = signal if reference is None else np.asarray(reference, dtype=np.complex128)
    if signal.ndim != 1 or matched.ndim != 1 or signal.size == 0 or matched.size == 0:
        msg = "waveform and reference must be non-empty vectors, shape (n_samples,)."
        raise ValueError(msg)
    if sample_rate_hz <= 0.0:
        msg = f"sample_rate_hz must be strictly positive; got {sample_rate_hz!r}."
        raise ValueError(msg)

    if dopplers_hz is None:
        span_hz = sample_rate_hz if prf_hz is None else prf_hz
        n_dopplers = max(1, int(np.ceil(2.0 * span_hz * signal.size / sample_rate_hz)))
        dopplers_hz = (np.arange(n_dopplers) - n_dopplers // 2) * (span_hz / n_dopplers)
    doppler_arr = np.atleast_1d(np.asarray(dopplers_hz, dtype=np.float64))
    # Column k of the full correlation is the lag k - (n_reference - 1).
    first_lag = 1 - matched.size
    if delays_s is None:
        lags = np.arange(first_lag, signal.size)
    else:
        lags = np.rint(np.atleast_1d(np.asarray(delays_s, dtype=np.float64)) * sample_rate_hz)
        lags = lags.astype(np.int64)

    sample_index = np.arange(signal.size)
    shifted = signal * np.exp(2j * np.pi * doppler_arr[:, None] * sample_index / sample_rate_hz)
    correlation = fftconvolve(shifted, np.conj(matched[::-1])[None, :], mode="full", axes=-1)
    energy = np.sqrt(np.sum(np.abs(signal) ** 2) * np.sum(np.abs(matched) ** 2))
    column = lags - first_lag
    inside = (column >= 0) & (column < correlation.shape[-1])
    magnitude = np.where(
        inside, np.abs(correlation[:, np.clip(column, 0, correlation.shape[-1] - 1)]) / energy, 0.0
    )
    return magnitude, lags / sample_rate_hz, doppler_arr


def _check_dwell(radar: Radar, n_pulses: int, waveform: str) -> None:
    """Reject an empty dwell, or a radar of the other waveform family."""
    if n_pulses < 1:
        msg = f"n_pulses must be at least one; got {n_pulses}."
        raise ValueError(msg)
    if radar.transmitter.waveform != waveform:
        msg = f"this generator needs a {waveform} transmitter; got {radar.transmitter.waveform!r}."
        raise ValueError(msg)


def _sum_over_paths(
    chunk_signal: Callable[[slice], NDArray[np.complex128]],
    n_paths: int,
    shape: tuple[int, int],
    max_bytes: int,
) -> NDArray[np.complex128]:
    """Sum the per-path signal over paths, a chunk of paths at a time, ``shape`` out."""
    n_pulses, n_samples = shape
    paths_per_chunk = max(1, max_bytes // (_BYTES_PER_COMPLEX * n_pulses * n_samples))
    cube = np.zeros(shape, dtype=np.complex128)
    # A loop over chunks bounds memory; the chunk count depends only on the shapes.
    for start in range(0, n_paths, paths_per_chunk):
        chunk = chunk_signal(slice(start, start + paths_per_chunk))
        # Prepending the running sum adds paths in one order however they are chunked.
        cube = np.concatenate([cube[:, None, :], chunk], axis=1).sum(axis=1)
    return cube


def _add_noise(
    cube: NDArray[np.complex128], radar: Radar, rng: np.random.Generator | None
) -> NDArray[np.complex128]:
    """Add receiver noise to a cube, unless ``rng`` is ``None``."""
    if rng is None:
        return cube
    return cube + thermal_noise(cube.shape, radar.receiver.noise_power_per_sample_w, rng=rng)


def _slow_time_s(n_pulses: int, radar: Radar) -> NDArray[np.float64]:
    """Return the start time of each pulse, ``(n_pulses,)``, s."""
    result: NDArray[np.float64] = (
        np.arange(n_pulses, dtype=np.float64) * radar.transmitter.pulse_repetition_interval_s
    )
    return result


def _fast_time_s(radar: Radar) -> NDArray[np.float64]:
    """Return the sample instants within one pulse, ``(n_samples_per_pri,)``, s."""
    result: NDArray[np.float64] = (
        np.arange(radar.n_samples_per_pri, dtype=np.float64) / radar.receiver.sample_rate_hz
    )
    return result


def _warn_if_stop_and_hop_is_strained(paths: PropagationPaths, radar: Radar) -> None:
    r"""Warn when a path moves a noticeable fraction of a range bin within one chirp.

    The route moves at :math:`\lambda |f_D|` against a bin of :math:`c/B` (the
    best case; a wide bistatic angle coarsens it). One reduction, before any chunk.
    """
    fastest_mps = float(np.max(np.abs(paths.doppler_hz), initial=0.0)) * radar.wavelength_m
    bin_fraction = (
        fastest_mps * radar.transmitter.chirp_duration_s / (2.0 * radar.range_resolution_m)
    )
    if bin_fraction > _STOP_AND_HOP_BIN_FRACTION:
        msg = (
            f"a target moves {bin_fraction:.2f} of a range bin during one chirp, above the "
            f"{_STOP_AND_HOP_BIN_FRACTION} threshold: stop-and-hop will smear its range peak. "
            "Shorten chirp_duration_s or accept the smearing."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
