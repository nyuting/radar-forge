r"""Baseband signal synthesis: propagation paths to complex IQ.

This module owns the step from *geometry* to *samples*, and it owns it for every
propagation model in the library. Anything that can describe how energy reaches a
receiver — the line-of-sight calculation here, and later any ray-tracing backend
— produces a :class:`PropagationPaths`, and the generators here turn that into a
baseband cube. That split is decision **D1** in ``spec/structure.md``: backends
compute paths, this module computes signals, and a new backend therefore costs no
new signal code.

The generators implement two receive architectures:

:func:`fmcw_deramp_baseband`
    Continuous chirp, deramped against the transmitted sweep. Range appears as a
    beat frequency.
:func:`pulsed_baseband`
    Pulse train, sampled over the full repetition interval and left for a matched
    filter downstream. Range appears as a delay, and delays beyond one interval
    wrap, which is where the pulsed variant's range folding comes from.

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

**Free space.** No atmospheric absorption, no multipath, no clutter. Amplitude
comes from the monostatic range equation alone.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §8.2 (the stop-and-hop model), §8.3, §14.2.
.. [2] G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014,
       §2.5 (FMCW deramp).
.. [3] A. G. Stove, "Linear FMCW radar techniques", *IEE Proceedings F*, vol. 139,
       no. 5, pp. 343-350, 1992 (sweep direction and the range-Doppler sign lock).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.core.radar import Radar
from radar_forge.core.radar_equation import received_power_w

__all__ = [
    "PropagationPaths",
    "fmcw_deramp_baseband",
    "line_of_sight_paths",
    "pulsed_baseband",
    "thermal_noise",
]

# Above this fraction of a range bin of target motion within one chirp, the
# stop-and-hop approximation starts to smear the range peak.
_STOP_AND_HOP_BIN_FRACTION = 0.25


@dataclass(frozen=True)
class PropagationPaths:
    """A set of propagation paths between a radar and its scene.

    The interface decision **D1** of ``spec/structure.md``: every propagation
    model, from the line-of-sight calculation in this module to a future
    ray-tracing backend, produces one of these, and the signal generators consume
    nothing else. Field names carry unit suffixes as
    ``docs/conventions/style.md`` §2 requires, which is the one departure from
    the names written in D1.

    Parameters
    ----------
    delay_s : numpy.ndarray
        Two-way propagation delay of each path, shape ``(n_paths,)``, seconds.
    doppler_hz : numpy.ndarray
        Doppler shift of each path, shape ``(n_paths,)``, hertz. **Positive is
        closing**, so ``doppler_hz = 2 * closing_velocity_mps / wavelength_m``.
    amplitude_linear : numpy.ndarray
        Complex voltage amplitude of each path, shape ``(n_paths,)``. Its squared
        magnitude is the received power in watts, so that a cube built from it
        sits on the same scale as :attr:`Radar.noise_power_w`.
    aoa_rad : numpy.ndarray
        Angle of arrival, shape ``(n_paths, 2)``, as (azimuth, elevation) in
        radians. Unused until the array processing modules land, but carried so
        that a backend need not be revised to add it.
    aod_rad : numpy.ndarray
        Angle of departure, same shape and convention as ``aoa_rad``.
    bounce_count : numpy.ndarray
        Number of interactions along each path, shape ``(n_paths,)``. One for a
        direct line-of-sight return.

    Raises
    ------
    ValueError
        If the per-path arrays do not share a leading dimension, or if the angle
        arrays are not ``(n_paths, 2)``.
    """

    delay_s: NDArray[np.float64]
    doppler_hz: NDArray[np.float64]
    amplitude_linear: NDArray[np.complex128]
    aoa_rad: NDArray[np.float64]
    aod_rad: NDArray[np.float64]
    bounce_count: NDArray[np.int_]

    def __post_init__(self) -> None:
        """Validate that every per-path array agrees on the number of paths."""
        n_paths = self.delay_s.shape[0]
        for name in ("doppler_hz", "amplitude_linear", "bounce_count"):
            shape = getattr(self, name).shape
            if shape != (n_paths,):
                msg = f"{name} must have shape ({n_paths},) to match delay_s; got {shape}."
                raise ValueError(msg)
        for name in ("aoa_rad", "aod_rad"):
            shape = getattr(self, name).shape
            if shape != (n_paths, 2):
                msg = (
                    f"{name} must have shape ({n_paths}, 2) for (azimuth, elevation); got {shape}."
                )
                raise ValueError(msg)

    @property
    def n_paths(self) -> int:
        """Number of propagation paths."""
        return int(self.delay_s.shape[0])

    @property
    def range_m(self) -> NDArray[np.float64]:
        """One-way equivalent range of each path, metres, from the delay."""
        result: NDArray[np.float64] = self.delay_s * SPEED_OF_LIGHT_MPS / 2.0
        return result


def line_of_sight_paths(
    radar: Radar,
    range_m: ArrayLike,
    radial_velocity_mps: ArrayLike,
    rcs_m2: ArrayLike,
    *,
    azimuth_deg: ArrayLike = 0.0,
    elevation_deg: ArrayLike = 0.0,
) -> PropagationPaths:
    r"""Build the direct-return paths for point targets in free space.

    One path per target, single bounce, no multipath and no obstruction. This is
    the zeroth-order case of a ray-tracing backend, and
    ``raytracing/backends/analytic.py`` should build on it rather than repeat it.

    Amplitude is the square root of the monostatic radar range equation, so that
    ``abs(amplitude_linear) ** 2`` is the received power in watts and a cube built
    from it is directly comparable with :attr:`Radar.noise_power_w`. The phase is
    the two-way propagation phase :math:`-4\pi R/\lambda`.

    Parameters
    ----------
    radar : Radar
        The observing radar; supplies wavelength, power and antenna gains.
    range_m : array_like
        Slant range to each target, metres. Must be strictly positive.
    radial_velocity_mps : array_like
        Radial velocity of each target, metres/second, **positive closing**.
    rcs_m2 : array_like
        Radar cross-section of each target, square metres. Zero is permitted and
        produces a null path, which is how a noise-only scenario is built.
    azimuth_deg, elevation_deg : array_like, optional
        Arrival angles, degrees, in the library's compass convention. Default 0,
        i.e. boresight. Carried through to the path set but not yet used by the
        generators, which assume a single isotropic channel.

    Returns
    -------
    PropagationPaths
        One path per target, in the order given.

    Raises
    ------
    ValueError
        If any range is non-positive or any cross-section is negative.

    See Also
    --------
    fmcw_deramp_baseband : Turn these paths into a deramped FMCW cube.
    pulsed_baseband : Turn these paths into a pulsed cube.

    Examples
    --------
    >>> from radar_forge.core.radar import Radar, Receiver, Transmitter
    >>> tx = Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3)
    >>> radar = Radar(tx, Receiver(1.0e6, 30.0, 3.0), 1.2915, 103.7871, 60.0)
    >>> paths = line_of_sight_paths(radar, 10_000.0, 80.0, 10.0)
    >>> paths.n_paths
    1
    >>> bool(paths.doppler_hz[0] > 0.0)  # closing is positive
    True
    """
    range_arr = np.atleast_1d(np.asarray(range_m, dtype=np.float64))
    velocity_arr = np.atleast_1d(np.asarray(radial_velocity_mps, dtype=np.float64))
    rcs_arr = np.atleast_1d(np.asarray(rcs_m2, dtype=np.float64))

    if np.any(range_arr <= 0.0):
        msg = "range_m must be strictly positive; a target at zero range is a modelling error."
        raise ValueError(msg)
    if np.any(rcs_arr < 0.0):
        msg = "rcs_m2 must be non-negative; use 0.0 for a target that returns nothing."
        raise ValueError(msg)

    range_arr, velocity_arr, rcs_arr = np.broadcast_arrays(range_arr, velocity_arr, rcs_arr)
    n_paths = range_arr.shape[0]

    # The range equation diverges at zero cross-section only through the power,
    # so substitute a placeholder area and zero the amplitude afterwards rather
    # than letting a legitimate 0 m^2 target raise.
    illuminated = rcs_arr > 0.0
    power_w = np.zeros_like(range_arr)
    power_w[illuminated] = received_power_w(
        transmit_power_w=radar.transmitter.transmit_power_w,
        gain_tx_linear=10.0 ** (radar.transmitter.gain_tx_dbi / 10.0),
        gain_rx_linear=10.0 ** (radar.receiver.gain_rx_dbi / 10.0),
        wavelength_m=radar.wavelength_m,
        rcs_m2=rcs_arr[illuminated],
        range_m=range_arr[illuminated],
    )

    two_way_phase_rad = -4.0 * np.pi * range_arr / radar.wavelength_m
    amplitude_linear = np.sqrt(power_w) * np.exp(1j * two_way_phase_rad)

    angles_rad = np.stack(
        np.broadcast_arrays(
            np.radians(np.asarray(azimuth_deg, dtype=np.float64)),
            np.radians(np.asarray(elevation_deg, dtype=np.float64)),
            np.zeros(n_paths),
        )[:2],
        axis=-1,
    )

    return PropagationPaths(
        delay_s=2.0 * range_arr / SPEED_OF_LIGHT_MPS,
        doppler_hz=2.0 * velocity_arr / radar.wavelength_m,
        amplitude_linear=np.asarray(amplitude_linear, dtype=np.complex128),
        aoa_rad=angles_rad,
        aod_rad=angles_rad.copy(),
        bounce_count=np.ones(n_paths, dtype=np.int_),
    )


def thermal_noise(
    shape: tuple[int, ...],
    noise_power_w: float,
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
    radar: Radar,
    n_chirps: int,
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
        Propagation paths, typically from :func:`line_of_sight_paths`.
    radar : Radar
        The observing radar. Must carry an FMCW transmitter.
    n_chirps : int
        Number of chirps in the coherent processing interval — the slow-time
        length of the cube. Must be at least one.
    rng : numpy.random.Generator, optional
        Seeded generator for receiver noise. Default None, which produces a
        **noiseless** cube; pass a generator to add
        :attr:`Radar.noise_power_w` per sample.

    Returns
    -------
    numpy.ndarray
        Complex baseband of shape ``(n_chirps, n_samples)``, slow time along
        axis 0 and fast time along axis 1 — the canonical layout of
        ``docs/conventions/style.md`` §4, so that
        :func:`radar_forge.core.dsp.range_doppler_map` needs no axis arguments.

    Raises
    ------
    ValueError
        If ``n_chirps`` is less than one, or the radar is not FMCW.

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
    if n_chirps < 1:
        msg = f"n_chirps must be at least one; got {n_chirps}."
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

    # (n_chirps, n_paths): range held constant within a chirp, advancing between
    # them. Broadcasting an outer difference, not tiling.
    chirp_index = np.arange(n_chirps, dtype=np.float64)[:, None]
    range_m = (
        paths.range_m[None, :]
        - closing_velocity_mps[None, :] * chirp_index * pulse_repetition_interval_s
    )
    delay_s = 2.0 * range_m / SPEED_OF_LIGHT_MPS

    n_samples = radar.n_samples_per_pri
    fast_time_s = np.arange(n_samples, dtype=np.float64) / radar.receiver.sample_rate_hz

    # (n_chirps, n_paths, 1) against (n_samples,) broadcasts to the full cube
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
        cube = cube + thermal_noise(cube.shape, radar.noise_power_w, rng)
    return cube


def pulsed_baseband(
    paths: PropagationPaths,
    radar: Radar,
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
        Propagation paths, typically from :func:`line_of_sight_paths`.
    radar : Radar
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
    within_pulse = (elapsed_s >= 0.0) & (elapsed_s < radar.transmitter.chirp_time_s)
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
        cube = cube + thermal_noise(cube.shape, radar.noise_power_w, rng)
    return cube


def _warn_if_stop_and_hop_is_strained(
    closing_velocity_mps: NDArray[np.float64], radar: Radar
) -> None:
    """Warn when a target moves a noticeable fraction of a bin within one chirp."""
    motion_m = np.abs(closing_velocity_mps) * radar.transmitter.chirp_time_s
    bin_fraction = motion_m / radar.range_resolution_m
    worst = float(np.max(bin_fraction, initial=0.0))
    if worst > _STOP_AND_HOP_BIN_FRACTION:
        msg = (
            f"a target moves {worst:.2f} of a range bin during one chirp, above the "
            f"{_STOP_AND_HOP_BIN_FRACTION} threshold: the stop-and-hop approximation in this "
            "module will smear its range peak. Shorten chirp_time_s or accept the smearing."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
