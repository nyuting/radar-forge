r"""Baseband signal synthesis: propagation paths to complex IQ.

This module owns the step from *paths* to *samples*, and it owns it for every
propagation model in the library. :mod:`radar_forge.core.propagation` describes
how energy reaches a receiver as a :class:`~radar_forge.core.propagation.PropagationPaths`,
and the generators here turn that into a baseband cube. That split is decision
**D1** in ``spec/structure.md``: backends compute paths, this module computes
signals, and a new backend therefore costs no new signal code.

The generators implement two receive architectures:

:func:`fmcw_deramp_baseband`
    Continuous chirp, deramped against the transmitted sweep. Range appears as a
    beat frequency.
:func:`pulsed_baseband`
    Pulse train, sampled over the full repetition interval and left for a matched
    filter downstream. Range appears as a delay, and delays beyond one interval
    wrap, which is where the pulsed variant's range folding comes from.

Both consume monostatic and bistatic path sets alike; the
:mod:`radar_forge.core.propagation` docstring explains why no bistatic twin is
needed.

Sign conventions
----------------
The library's convention is **closing velocity positive**
(``spec/structure.md`` D5), and :func:`radar_forge.core.dsp.doppler_bin_centers_mps`
labels its shifted axis that way.

Getting a simulator to honour that is not free, because in an FMCW deramp the
range and Doppler signs are **locked together**: both the beat-frequency term and
the residual phase term scale with the same delay :math:`\tau`, so choosing the
mixer sideband that puts range at a positive beat frequency also fixes the
Doppler sign. Conjugating to fix one breaks the other.

The sweep *direction* is the free parameter that unlocks them. With an **up**
sweep, a closing target lands at positive range and **negative** Doppler. With a
**down** sweep it lands at positive range and positive Doppler. This module
therefore models FMCW as a down sweep, which is the only combination that is both
physically consistent and compliant with D5 — and the reason
:func:`radar_forge.core.waveforms.lfm_chirp` takes an ``up_sweep`` flag.

The pulsed generator needs no such care: standard baseband downconversion carries
the two-way phase :math:`e^{-j 4\pi R/\lambda}` directly, and a closing target is
positive without adjustment.

Approximations
--------------
**Stop-and-hop.** Range is held constant across each chirp and advanced between
chirps. This is the standard pulse-Doppler idealisation and it is what removes
intra-chirp range-Doppler coupling, so a target's range bin is exact rather than
biased by its velocity. It is valid while the target moves much less than a range
bin during one chirp: at 80 m/s over the 1 ms chirp of scenario 001 S1 that is
0.08 m against a 74.95 m bin, so the error is four orders of magnitude below the
resolution. It would **not** be valid for a long chirp against a fast target, and
:func:`fmcw_deramp_baseband` checks the bound and warns.

**Constant velocity within a coherent processing interval.** Acceleration is
neglected over the dwell, which spreads a real target's Doppler slightly.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §2.6.2 (the stop-and-hop approximation), §4.6.5
       (stretch processing: the FMCW deramp).
.. [2] G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014,
       ch. 3, "Mathematics of FMCW Radar" (FMCW deramp).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.core.propagation import PropagationPaths
from radar_forge.core.radar import RadarLike

__all__ = [
    "fmcw_deramp_baseband",
    "pulsed_baseband",
    "thermal_noise",
]

# Above this fraction of a range bin of target motion within one chirp, the
# stop-and-hop approximation starts to smear the range peak.
_STOP_AND_HOP_BIN_FRACTION = 0.25


def thermal_noise(
    shape: tuple[int, ...],
    noise_power_w: float,
    *,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    r"""Draw circularly-symmetric complex Gaussian receiver noise.

    The total power is split equally between the real and imaginary parts, each
    getting variance :math:`N/2`, so that the expected value of
    ``abs(noise) ** 2`` is ``noise_power_w``.

    Parameters
    ----------
    shape : tuple of int
        Output shape.
    noise_power_w : float
        Total noise power per sample, watts — normally
        :attr:`Radar.noise_power_w`. Must be non-negative; zero gives an exactly
        noiseless cube, which is useful for isolating a geometry bug.
    rng : numpy.random.Generator
        Seeded generator. Required, not optional: every random draw in the
        library is reproducible, per ``docs/conventions/style.md`` §7.

    Returns
    -------
    numpy.ndarray
        Complex noise of the requested shape.

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
    radar: RadarLike,
    n_pulses: int,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    r"""Synthesise a deramped FMCW baseband cube.

    Models a **down** sweep mixed as :math:`s_{rx} \cdot s_{tx}^{*}`, giving

    .. math::

        s[m, n] = \sum_p A_p \,
            \exp\!\left(-j 2\pi f_0 \tau_p[m]\right)\,
            \exp\!\left(+j 2\pi \alpha \tau_p[m]\, t_n\right)

    with :math:`\tau_p[m] = 2 R_p[m]/c` held constant across chirp :math:`m`
    (stop-and-hop) and :math:`t_n = n/f_s`. The first exponential is the residual
    phase that becomes Doppler across slow time; the second is the beat frequency
    that becomes range across fast time. See the module docstring for why the
    sweep must descend for both to carry the library's sign convention.

    Parameters
    ----------
    paths : PropagationPaths
        Propagation paths, typically from
        :func:`radar_forge.core.propagation.line_of_sight_paths`.
    radar : Radar or BistaticRadar
        The observing radar. Must carry an FMCW transmitter.
    n_pulses : int
        Number of chirps in the coherent processing interval — the slow-time
        length of the cube. Must be at least one.
    rng : numpy.random.Generator, optional
        Seeded generator for receiver noise. Default None, which produces a
        **noiseless** cube; pass a generator to add
        :attr:`Radar.noise_power_w` per sample.

    Returns
    -------
    numpy.ndarray
        Complex baseband of shape ``(n_pulses, n_samples)``, slow time along
        axis 0 and fast time along axis 1 — the canonical layout of
        ``docs/conventions/style.md`` §3.1, so that
        :func:`radar_forge.core.dsp.range_doppler_map` needs no axis arguments.

    Raises
    ------
    ValueError
        If ``n_pulses`` is less than one, or the radar is not FMCW.

    Warns
    -----
    UserWarning
        If a target moves more than a quarter of a range bin during one chirp,
        where stop-and-hop begins to smear the range peak.

    See Also
    --------
    pulsed_baseband : The pulse-train equivalent.
    radar_forge.core.dsp.range_doppler_map : Consumes this cube directly.
    """
    if n_pulses < 1:
        msg = f"n_pulses must be at least one; got {n_pulses}."
        raise ValueError(msg)
    if radar.transmitter.waveform != "fmcw":
        msg = (
            f"fmcw_deramp_baseband needs an FMCW transmitter; got {radar.transmitter.waveform!r}. "
            "Use pulsed_baseband instead."
        )
        raise ValueError(msg)

    pulse_repetition_interval_s = radar.transmitter.pulse_repetition_interval_s
    closing_velocity_mps = paths.doppler_hz * radar.wavelength_m / 2.0
    _warn_if_stop_and_hop_is_strained(closing_velocity_mps, radar)

    # (n_pulses, n_paths): range held constant within a chirp, advancing between
    # them. Broadcasting an outer difference, not tiling.
    chirp_index = np.arange(n_pulses, dtype=np.float64)[:, None]
    range_m = (
        paths.range_m[None, :]
        - closing_velocity_mps[None, :] * chirp_index * pulse_repetition_interval_s
    )
    delay_s = 2.0 * range_m / SPEED_OF_LIGHT_MPS

    n_samples = radar.n_samples_per_pri
    fast_time_s = np.arange(n_samples, dtype=np.float64) / radar.receiver.sample_rate_hz

    # (n_pulses, n_paths, 1) against (n_samples,) broadcasts to the full cube
    # without materialising an intermediate of that size per path.
    residual_phase_rad = -2.0 * np.pi * radar.transmitter.f0_hz * delay_s
    beat_phase_rad = (
        2.0 * np.pi * radar.transmitter.sweep_rate_hzps * delay_s[..., None] * fast_time_s
    )
    per_path = np.abs(paths.amplitude_linear)[None, :, None] * np.exp(
        1j * (residual_phase_rad[..., None] + beat_phase_rad)
    )
    cube: NDArray[np.complex128] = np.asarray(per_path.sum(axis=1), dtype=np.complex128)

    if rng is not None:
        cube = cube + thermal_noise(cube.shape, radar.noise_power_w, rng=rng)
    return cube


def pulsed_baseband(
    paths: PropagationPaths,
    radar: RadarLike,
    n_pulses: int,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    r"""Synthesise a pulsed baseband cube, before pulse compression.

    Each path contributes a delayed copy of the transmitted LFM pulse,

    .. math::

        s[m, n] = \sum_p A_p \, p\!\left(t_n - \tau_p[m]\right)\,
                  \exp\!\left(-j 2\pi f_0 \tau_p[m]\right)

    where :math:`p` is the baseband chirp and the delay is taken **modulo the
    repetition interval**. That modulo is not a convenience: it is exactly how a
    real pulsed radar folds targets beyond its unambiguous range, and it is what
    makes the scenario-001 S2 variant show its target at the wrong range.

    The output is *not* pulse-compressed. Pass it through
    :func:`radar_forge.core.dsp.matched_filter` with the reference returned by
    :func:`radar_forge.core.waveforms.lfm_chirp`.

    Parameters
    ----------
    paths : PropagationPaths
        Propagation paths, typically from
        :func:`radar_forge.core.propagation.line_of_sight_paths`.
    radar : Radar or BistaticRadar
        The observing radar. Must carry a pulsed transmitter.
    n_pulses : int
        Number of pulses in the coherent processing interval. At least one.
    rng : numpy.random.Generator, optional
        Seeded generator for receiver noise. Default None, i.e. noiseless.

    Returns
    -------
    numpy.ndarray
        Complex baseband of shape ``(n_pulses, n_samples_per_pri)``, slow time
        along axis 0.

    Raises
    ------
    ValueError
        If ``n_pulses`` is less than one, or the radar is not pulsed.

    Notes
    -----
    Eclipsing is not modelled: a return arriving while the transmitter is on
    would in reality be lost, and here it is received. At the 25% duty cycle of
    scenario 001 S2 that affects a quarter of all ranges, so a target whose
    folded delay lands under the transmit pulse is being treated more kindly
    than a real radar would treat it.
    """
    if n_pulses < 1:
        msg = f"n_pulses must be at least one; got {n_pulses}."
        raise ValueError(msg)
    if radar.transmitter.waveform != "pulsed":
        msg = (
            f"pulsed_baseband needs a pulsed transmitter; got {radar.transmitter.waveform!r}. "
            "Use fmcw_deramp_baseband instead."
        )
        raise ValueError(msg)

    pulse_repetition_interval_s = radar.transmitter.pulse_repetition_interval_s
    closing_velocity_mps = paths.doppler_hz * radar.wavelength_m / 2.0

    chirp_index = np.arange(n_pulses, dtype=np.float64)[:, None]
    range_m = (
        paths.range_m[None, :]
        - closing_velocity_mps[None, :] * chirp_index * pulse_repetition_interval_s
    )
    true_delay_s = 2.0 * range_m / SPEED_OF_LIGHT_MPS
    folded_delay_s = np.mod(true_delay_s, pulse_repetition_interval_s)

    n_samples = radar.n_samples_per_pri
    fast_time_s = np.arange(n_samples, dtype=np.float64) / radar.receiver.sample_rate_hz

    elapsed_s = fast_time_s - folded_delay_s[..., None]
    within_pulse = (elapsed_s >= 0.0) & (elapsed_s < radar.transmitter.chirp_duration_s)
    envelope_phase_rad = np.pi * radar.transmitter.sweep_rate_hzps * elapsed_s**2
    # The residual phase uses the TRUE delay, not the folded one: folding is an
    # artefact of when the receiver listens, and does not change the carrier
    # phase the target actually imposed. Using the folded delay here would
    # corrupt the Doppler.
    residual_phase_rad = -2.0 * np.pi * radar.transmitter.f0_hz * true_delay_s

    per_path = (
        np.abs(paths.amplitude_linear)[None, :, None]
        * within_pulse
        * np.exp(1j * (envelope_phase_rad + residual_phase_rad[..., None]))
    )
    cube: NDArray[np.complex128] = np.asarray(per_path.sum(axis=1), dtype=np.complex128)

    if rng is not None:
        cube = cube + thermal_noise(cube.shape, radar.noise_power_w, rng=rng)
    return cube


def _warn_if_stop_and_hop_is_strained(
    closing_velocity_mps: NDArray[np.float64], radar: RadarLike
) -> None:
    """Warn when a target moves a noticeable fraction of a bin within one chirp.

    For a :class:`~radar_forge.core.radar.BistaticRadar` the bin used is the
    best-case one at zero bistatic angle, so the warning is slightly
    conservative: a target at a wide bistatic angle has a coarser bin than this
    and is smeared a little less than the message claims.
    """
    motion_m = np.abs(closing_velocity_mps) * radar.transmitter.chirp_duration_s
    bin_fraction = motion_m / radar.range_resolution_m
    worst = float(np.max(bin_fraction, initial=0.0))
    if worst > _STOP_AND_HOP_BIN_FRACTION:
        msg = (
            f"a target moves {worst:.2f} of a range bin during one chirp, above the "
            f"{_STOP_AND_HOP_BIN_FRACTION} threshold: the stop-and-hop approximation in this "
            "module will smear its range peak. Shorten chirp_duration_s or accept the smearing."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
