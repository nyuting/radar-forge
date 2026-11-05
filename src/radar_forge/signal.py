r"""Baseband signals: the transmitted waveform, and propagation paths to complex IQ.

:func:`lfm_chirp` is the waveform, :func:`ambiguity_function` what it can
resolve, and the FMCW relations map a bistatic range to its beat tone. The
generators turn a :class:`~radar_forge.propagation.PropagationPaths` into a
``(n_pulses, n_samples)`` cube, monostatic and bistatic alike, under
stop-and-hop: range as a beat frequency (:func:`fmcw_deramp_baseband`) or as a
delay folded modulo the PRI (:func:`pulsed_baseband`). A closing target
advances in slow-time phase.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014.
.. [2] G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve

from radar_forge.constants import SPEED_OF_LIGHT_MPS
from radar_forge.propagation import PropagationPaths, propagation_delay_s, propagation_phase_rad
from radar_forge.radar import Radar, Transmitter

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
    r"""Return the monostatic range resolution :math:`c/2B`, m (Richards [1], §4.6.1)."""
    result: NDArray[np.float64] = SPEED_OF_LIGHT_MPS / (
        2.0 * np.asarray(bandwidth_hz, dtype=np.float64)
    )
    return result


def sweep_rate_hzps(bandwidth_hz: ArrayLike, chirp_duration_s: ArrayLike) -> NDArray[np.float64]:
    r"""Return the chirp slope :math:`\alpha = B/T`, Hz/s."""
    result: NDArray[np.float64] = np.asarray(bandwidth_hz, dtype=np.float64) / np.asarray(
        chirp_duration_s, dtype=np.float64
    )
    return result


def beat_frequency_hz(
    bistatic_range_m: ArrayLike, bandwidth_hz: ArrayLike, chirp_duration_s: ArrayLike
) -> NDArray[np.float64]:
    r"""Return the deramped beat :math:`\alpha R_b / c`, Hz: no factor of 2 (Charvat [2] ch. 3)."""
    result: NDArray[np.float64] = (
        sweep_rate_hzps(bandwidth_hz, chirp_duration_s)
        * np.asarray(bistatic_range_m, dtype=np.float64)
        / SPEED_OF_LIGHT_MPS
    )
    return result


def bistatic_range_from_beat_frequency_m(
    beat_hz: ArrayLike, bandwidth_hz: ArrayLike, chirp_duration_s: ArrayLike
) -> NDArray[np.float64]:
    r"""Return the bistatic range :math:`c f_b/\alpha` behind a beat, m; halve for monostatic."""
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
    r"""Return one linear-FM chirp on the half-open :math:`[0, T)` (Richards [1] §4.6.1).

    By default it is centred on zero, starting at :math:`-B/2` up or :math:`+B/2` down.
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
    """Draw circular complex Gaussian noise, ``noise_power_w`` per sample, half in I, half in Q."""
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
    """Synthesise a deramped FMCW cube, ``(n_pulses, n_samples_per_pri)``; noiseless if rng=None.

    The sweep is modelled **down**, from f_c + B/2: range and Doppler signs are
    locked in a deramp, and only a down sweep puts range at a positive beat and
    Doppler on its true sign. Paths are summed in chunks of
    ``max(1, max_bytes // (16 n_pulses n_samples))``, each added onto the
    running sum, so memory does not grow with the path count and the cube is
    bit-identical however it is chunked. Warns if a path moves a quarter of a
    range bin within one chirp (Charvat [2] ch. 3; Richards [1] §4.6.5).
    """
    _check_dwell(radar, n_pulses, "fmcw")
    return _synthesise(paths, radar, n_pulses, _deramp_kernel, rng, max_bytes)


def pulsed_baseband(
    paths: PropagationPaths,
    radar: Radar,
    n_pulses: int,
    *,
    rng: np.random.Generator | None,
    max_bytes: int = 256 * 2**20,
) -> NDArray[np.complex128]:
    """Synthesise a pulsed cube before compression, ``(n_pulses, n_samples_per_pri)``.

    Each path is ``lfm_chirp(B, T, f_s)`` at its delay folded modulo the PRI (how
    range folds) with the carrier phase of the true delay (which keeps Doppler
    right). An echo that runs past the end of its PRI goes on at the start of the
    next row, because the receiver keeps listening; the last row's overrun is
    lost. No eclipsing. Chunking, ``rng`` and the warning as :func:`fmcw_deramp_baseband`.
    """
    _check_dwell(radar, n_pulses, "pulsed")
    return _synthesise(paths, radar, n_pulses, _pulse_kernel, rng, max_bytes)


def _synthesise(
    paths: PropagationPaths,
    radar: Radar,
    n_pulses: int,
    kernel: Callable[[NDArray[np.float64], NDArray[np.float64], Radar], NDArray[np.complex128]],
    rng: np.random.Generator | None,
    max_bytes: int,
) -> NDArray[np.complex128]:
    """Sum ``|a_p|`` times each path's unit-amplitude ``kernel`` echo over paths, then add noise."""
    _warn_if_stop_and_hop_is_strained(paths, radar)
    bistatic_range_m = paths.bistatic_range_at_m(_slow_time_s(n_pulses, radar), radar.wavelength_m)
    magnitude = np.abs(paths.amplitude_linear)
    fast_time_s = _fast_time_s(radar)

    def chunk_signal(chunk: slice) -> NDArray[np.complex128]:
        signal = kernel(bistatic_range_m[:, chunk], fast_time_s, radar)
        signal *= magnitude[None, chunk, None]
        return signal

    cube = _sum_over_paths(
        chunk_signal, paths.n_paths, (n_pulses, radar.n_samples_per_pri), max_bytes
    )
    return _add_noise(cube, radar, rng)


def _deramp_kernel(
    bistatic_range_m: NDArray[np.float64], fast_time_s: NDArray[np.float64], radar: Radar
) -> NDArray[np.complex128]:
    """Return each route's deramped echo, ``(n_pulses, n_paths, n_samples)``: residual and beat."""
    transmitter = radar.transmitter
    delay_s = propagation_delay_s(bistatic_range_m)
    # The residual phase is referred to the sweep start, the frequency sent at t_n = 0.
    sweep_start_wavelength_m = SPEED_OF_LIGHT_MPS / (
        transmitter.carrier_frequency_hz + transmitter.bandwidth_hz / 2.0
    )
    residual_phase_rad = propagation_phase_rad(bistatic_range_m, sweep_start_wavelength_m)
    beat_phase_rad = 2.0 * np.pi * transmitter.sweep_rate_hzps * delay_s[..., None] * fast_time_s
    echo: NDArray[np.complex128] = np.exp(1j * (residual_phase_rad[..., None] + beat_phase_rad))
    return echo


def _pulse_kernel(
    bistatic_range_m: NDArray[np.float64], fast_time_s: NDArray[np.float64], radar: Radar
) -> NDArray[np.complex128]:
    """Return each route's pulse echo, ``(n_pulses, n_paths, n_samples)``, plus the last overrun."""
    transmitter = radar.transmitter
    pri_s = transmitter.pulse_repetition_interval_s
    folded_delay_s = np.mod(propagation_delay_s(bistatic_range_m), pri_s)
    residual_phase_rad = propagation_phase_rad(bistatic_range_m, radar.wavelength_m)
    echo = _pulse_echo(fast_time_s - folded_delay_s[..., None], residual_phase_rad, transmitter)
    # Row m also hears the end of row m - 1's echo. Row 0 has none: a zero delay puts
    # its predecessor a whole PRI back, at least one pulse length out of reach.
    previous_delay_s = np.concatenate([np.zeros_like(folded_delay_s[:1]), folded_delay_s[:-1]])
    previous_phase_rad = np.concatenate([residual_phase_rad[:1], residual_phase_rad[:-1]])
    echo += _pulse_echo(
        fast_time_s + pri_s - previous_delay_s[..., None], previous_phase_rad, transmitter
    )
    return echo


def _pulse_echo(
    elapsed_s: NDArray[np.float64],
    residual_phase_rad: NDArray[np.float64],
    transmitter: Transmitter,
) -> NDArray[np.complex128]:
    """Return the pulse ``elapsed_s`` into itself, zero outside, with each route's carrier phase."""
    within_pulse = (elapsed_s >= 0.0) & (elapsed_s < transmitter.chirp_duration_s)
    # The phase of lfm_chirp(B, T, f_s), the centred up sweep, at the elapsed time.
    envelope_phase_rad = (
        np.pi * elapsed_s * (transmitter.sweep_rate_hzps * elapsed_s - transmitter.bandwidth_hz)
    )
    carrier = np.exp(1j * (envelope_phase_rad + residual_phase_rad[..., None]))
    echo: NDArray[np.complex128] = np.where(within_pulse, carrier, 0j)
    return echo


def ambiguity_function(
    waveform: ArrayLike,
    sample_rate_hz: float,
    *,
    prf_hz: float | None = None,
    delays_s: ArrayLike | None = None,
    dopplers_hz: ArrayLike | None = None,
    reference: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Return :math:`|\chi(\tau, f_D)|`, normalised by :math:`\sqrt{E_s E_r}`, with its axes.

    The matched-filter response at delay :math:`\tau` to an echo shifted by
    :math:`f_D`: 1 at the origin of an auto-ambiguity and nowhere larger. A later
    echo has positive delay, so an up-chirp's ridge runs along :math:`f_D = -\alpha\tau`.
    One Doppler-shifted correlation per row, all in one :func:`scipy.signal.fftconvolve`.
    Delays are rounded to whole samples (default: every overlap); Doppler defaults to
    :math:`[-f_s/2, f_s/2)`, or one ``prf_hz``, every half inverse duration. A
    ``reference`` gives a cross-ambiguity. Returns the ``(n_dopplers, n_delays)``
    magnitude, as MATLAB's ``ambgfun``, and the delays and Dopplers evaluated
    (Richards [1] ch. 4).

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
    if prf_hz is not None and dopplers_hz is not None:
        msg = "give prf_hz or dopplers_hz, not both: prf_hz only sets the default Doppler grid."
        raise ValueError(msg)
    if prf_hz is not None and not prf_hz > 0.0:
        msg = f"prf_hz must be strictly positive; got {prf_hz!r}."
        raise ValueError(msg)
    energy = np.sqrt(np.sum(np.abs(signal) ** 2) * np.sum(np.abs(matched) ** 2))
    if energy == 0.0:
        msg = "waveform and reference must carry energy: the response is normalised by it."
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
        # Floating-point addition is not associative, so the order is the result:
        # adding one path at a time, in index order, in place, is the order of an
        # unchunked sum over the path axis, so the cube is bit-identical however it
        # is chunked, and nothing is allocated beyond the chunk.
        for path in range(chunk.shape[1]):
            cube += chunk[:, path]
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
    r"""Warn when a path moves a quarter of a range bin (:math:`c/B` of route) within one chirp."""
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
