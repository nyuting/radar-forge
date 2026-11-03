r"""Propagation paths: how energy gets from the transmitter to the receiver.

This module owns the step from *geometry* to *paths*. Anything that can describe
how energy reaches a receiver — the line-of-sight calculations here, and later
any ray-tracing backend — produces a :class:`PropagationPaths`: one delay,
Doppler shift, complex amplitude and pair of angles per path.
:mod:`radar_forge.core.signal` turns a path set into a baseband cube. That split
is decision **D1** in ``spec/structure.md``: backends compute paths, the signal
module computes samples, and a new backend therefore costs no new signal code.

Approximations
--------------
**Free space.** No atmospheric absorption, no multipath, no clutter. Amplitude
comes from the range equation alone, through
:func:`radar_forge.core.radar_equation.received_power_w` and its bistatic
counterpart; this module adds no link-budget mathematics of its own.

One geometry for both sitings
-----------------------------
A path is the whole transmitter-target-receiver route, and everything a path
carries follows from two numbers: its length, the **range sum**
:math:`L = R_t + R_r`, and how fast that length is shrinking, the **range-sum
rate** :math:`\dot{L}_c = -dL/dt`, positive closing [1]_, [2]_:

.. math::

    \tau = \frac{L}{c},
    \qquad
    \phi = -\frac{2\pi L}{\lambda},
    \qquad
    f_d = \frac{\dot{L}_c}{\lambda}

A monostatic radar is the case :math:`R_t = R_r = R`, where these become the
familiar :math:`2R/c`, :math:`-4\pi R/\lambda` and :math:`2 v_r/\lambda`. There is
no separate monostatic physics, so each relation is one public function here
(:func:`propagation_delay_s`, :func:`propagation_phase_rad`,
:func:`doppler_shift_hz`) and both path builders call the same ones. They differ
only in where the two ranges come from and which range equation sets the power.

Range-Doppler maps are still *labelled* in the monostatic convention, half the
range sum and half the range-sum rate, so that a monostatic map reads in range
and radial velocity. That halving is display, and it lives with the axes in
:mod:`radar_forge.core.dsp`, not here.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §2.2 (the radar range equation), §2.6.1 (the two-way
       propagation phase and Doppler shift).
.. [2] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005, §6.1
       (target Doppler) and ch. 3 (the range-sum ellipse).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.core.radar import BistaticRadar, Radar
from radar_forge.core.radar_equation import bistatic_received_power_w, received_power_w

__all__ = [
    "PropagationPaths",
    "bistatic_line_of_sight_paths",
    "doppler_shift_hz",
    "line_of_sight_paths",
    "path_amplitude_linear",
    "propagation_delay_s",
    "propagation_phase_rad",
    "range_sum_rate_from_doppler_mps",
    "range_sum_rate_mps",
]


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
        Total propagation delay of each path, transmitter to target to receiver,
        shape ``(n_paths,)``, seconds.
    doppler_hz : numpy.ndarray
        Doppler shift of each path, shape ``(n_paths,)``, hertz. **Positive is
        closing**: it is :func:`doppler_shift_hz` of the range-sum rate.
    amplitude_linear : numpy.ndarray
        Complex voltage amplitude of each path, shape ``(n_paths,)``, from
        :func:`path_amplitude_linear`. Its squared magnitude is the received
        power in watts, so that a cube built from it sits on the same scale as
        :attr:`Receiver.noise_power_per_sample_w`.

        **Only the magnitude reaches the generators.**
        :func:`~radar_forge.core.signal.fmcw_deramp_baseband` and
        :func:`~radar_forge.core.signal.pulsed_baseband` take
        ``abs(amplitude_linear)`` and rebuild the propagation phase from
        ``delay_s``, because the phase is what becomes Doppler across slow time
        and it must advance with the delay rather than be frozen at the value
        one path set recorded. The path builders here still populate the phase,
        so a path set reads correctly on its own; a backend that wants to impose
        an extra phase -- a complex reflection coefficient, say -- cannot do it
        through this field today, and that is a gap in D1 rather than an
        oversight in the generators.
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
    def range_sum_m(self) -> NDArray[np.float64]:
        """Total length of each path, :math:`R_t + R_r`, metres, from the delay.

        The inverse of :func:`propagation_delay_s`. For a monostatic path it is
        twice the range to the target; it is never a distance to anything on its
        own, and the surfaces of constant range sum are ellipsoids with the two
        sites at the foci.
        """
        result: NDArray[np.float64] = self.delay_s * SPEED_OF_LIGHT_MPS
        return result

    def range_sum_at_m(self, slow_time_s: ArrayLike, wavelength_m: float) -> NDArray[np.float64]:
        r"""Return the range sum of each path at the given slow-time instants, metres.

        .. math:: L_p(t) = L_p - \dot{L}_{c,p}\, t, \qquad \dot{L}_{c,p} = f_{d,p}\,\lambda

        Each path's length shrinks at its range-sum rate, recovered from its
        Doppler shift through :func:`range_sum_rate_from_doppler_mps`, so a
        closing path gets shorter as ``slow_time_s`` grows. Velocity is held
        constant, as everywhere in the generators. The signal generators pass
        the start of each pulse, :math:`m \cdot T_{PRI}`, and hold the result
        constant across that pulse, which is the stop-and-hop approximation [1]_.

        Monostatic and bistatic paths are treated alike: both carry a range sum
        and a Doppler shift, and nothing here depends on which siting built them.

        Parameters
        ----------
        slow_time_s : array_like
            Instants at which to evaluate, seconds, relative to the time this
            path set describes. Normally ``(n_pulses,)`` pulse start times.
        wavelength_m : float
            Carrier wavelength, metres: the one ``doppler_hz`` was computed
            with, normally ``radar.wavelength_m``.

        Returns
        -------
        numpy.ndarray
            Range sum in metres, shape ``slow_time_s.shape + (n_paths,)``, so
            ``(n_pulses, n_paths)`` for a vector of pulse start times.

        See Also
        --------
        propagation_delay_s : Turn the result into a delay.
        propagation_phase_rad : Turn the result into a carrier phase.

        References
        ----------
        .. [1] Richards (2014), §2.6.2; see the module References.

        Examples
        --------
        >>> import numpy as np
        >>> paths = PropagationPaths(
        ...     delay_s=np.array([1.0e-4]),
        ...     doppler_hz=np.array([100.0]),
        ...     amplitude_linear=np.array([1.0 + 0.0j]),
        ...     aoa_rad=np.zeros((1, 2)),
        ...     aod_rad=np.zeros((1, 2)),
        ...     bounce_count=np.array([1]),
        ... )
        >>> range_sum_m = paths.range_sum_at_m([0.0, 1.0], wavelength_m=0.03)
        >>> range_sum_m.shape
        (2, 1)
        >>> bool(range_sum_m[0, 0] == paths.range_sum_m[0])  # unchanged at t = 0
        True
        >>> round(float(range_sum_m[0, 0] - range_sum_m[1, 0]), 9)  # 100 Hz x 3 cm over 1 s
        3.0
        """
        time_arr = np.asarray(slow_time_s, dtype=np.float64)
        rate_mps = range_sum_rate_from_doppler_mps(self.doppler_hz, wavelength_m)
        result: NDArray[np.float64] = self.range_sum_m - rate_mps * time_arr[..., None]
        return result


def propagation_delay_s(range_sum_m: ArrayLike) -> NDArray[np.float64]:
    r"""Return the time for energy to travel a path, in seconds.

    .. math:: \tau = \frac{R_t + R_r}{c}

    The whole route at the speed of light [1]_. For a monostatic radar the range
    sum is :math:`2R` and this is the familiar two-way delay :math:`2R/c`.

    Parameters
    ----------
    range_sum_m : array_like
        Total path length :math:`R_t + R_r`, metres.

    Returns
    -------
    numpy.ndarray
        Delay in seconds, of the shape of the input.

    See Also
    --------
    PropagationPaths.range_sum_m : The inverse.

    References
    ----------
    .. [1] Richards (2014), §2.6.1; see the module References.
    """
    result: NDArray[np.float64] = np.asarray(range_sum_m, dtype=np.float64) / SPEED_OF_LIGHT_MPS
    return result


def propagation_phase_rad(range_sum_m: ArrayLike, wavelength_m: ArrayLike) -> NDArray[np.float64]:
    r"""Return the carrier phase a path imposes, in radians.

    .. math:: \phi = -\frac{2\pi (R_t + R_r)}{\lambda}

    One cycle of lag for every wavelength of route [1]_. For a monostatic radar
    this is the two-way phase :math:`-4\pi R/\lambda`. Its change from one pulse
    to the next is what a Doppler filter measures, which is why the signal
    generators recompute it as the range sum advances rather than read it off a
    path set.

    Parameters
    ----------
    range_sum_m : array_like
        Total path length :math:`R_t + R_r`, metres.
    wavelength_m : array_like
        Carrier wavelength, metres. Broadcast against ``range_sum_m``.

    Returns
    -------
    numpy.ndarray
        Phase in radians, unwrapped, of the broadcast shape of the inputs.

    References
    ----------
    .. [1] Richards (2014), §2.6.1; see the module References.
    """
    range_sum_arr = np.asarray(range_sum_m, dtype=np.float64)
    result: NDArray[np.float64] = -2.0 * np.pi * range_sum_arr / np.asarray(wavelength_m)
    return result


def path_amplitude_linear(power_w: ArrayLike, phase_rad: ArrayLike) -> NDArray[np.complex128]:
    r"""Return the complex voltage amplitude of a path.

    .. math:: a = \sqrt{P_r}\, e^{j\phi}

    The library's amplitude convention, stated once: ``abs(a) ** 2`` is the
    received power in watts, so that a cube built from these amplitudes sits on
    the same scale as
    :attr:`~radar_forge.core.radar.Receiver.noise_power_per_sample_w` and
    signal-to-noise ratios can be read straight off it [1]_.

    Parameters
    ----------
    power_w : array_like
        Received power, watts, normally from the range equation. Non-negative.
    phase_rad : array_like
        Carrier phase, radians, normally from :func:`propagation_phase_rad`.
        Broadcast against ``power_w``.

    Returns
    -------
    numpy.ndarray
        Complex amplitude, of the broadcast shape of the inputs.

    References
    ----------
    .. [1] Richards (2014), §2.2; see the module References.
    """
    amplitude = np.sqrt(np.asarray(power_w, dtype=np.float64)) * np.exp(
        1j * np.asarray(phase_rad, dtype=np.float64)
    )
    return np.asarray(amplitude, dtype=np.complex128)


def doppler_shift_hz(range_sum_rate_mps: ArrayLike, wavelength_m: ArrayLike) -> NDArray[np.float64]:
    r"""Return the Doppler shift of a path, in hertz.

    .. math:: f_d = \frac{\dot{L}_c}{\lambda} = -\frac{1}{\lambda}\frac{d(R_t + R_r)}{dt}

    The rate at which the path sheds wavelengths [1]_, [2]_. For a monostatic
    radar the range-sum rate is twice the radial velocity and this is the
    familiar :math:`2 v_r/\lambda`.

    Parameters
    ----------
    range_sum_rate_mps : array_like
        Range-sum rate, metres/second, **positive closing**: the rate at which the
        total path is getting shorter. See :func:`range_sum_rate_mps`.
    wavelength_m : array_like
        Carrier wavelength, metres. Broadcast against the rate.

    Returns
    -------
    numpy.ndarray
        Doppler shift in hertz, positive closing, of the broadcast shape of the
        inputs.

    See Also
    --------
    range_sum_rate_from_doppler_mps : The inverse.

    References
    ----------
    .. [1] Richards (2014), §2.6.1; see the module References.
    .. [2] Willis (2005), §6.1; see the module References.
    """
    range_sum_rate_arr = np.asarray(range_sum_rate_mps, dtype=np.float64)
    result: NDArray[np.float64] = range_sum_rate_arr / np.asarray(wavelength_m)
    return result


def range_sum_rate_from_doppler_mps(
    doppler_hz: ArrayLike, wavelength_m: ArrayLike
) -> NDArray[np.float64]:
    r"""Return the range-sum rate that produces a Doppler shift, in metres/second.

    .. math:: \dot{L}_c = f_d\,\lambda

    The inverse of :func:`doppler_shift_hz`. Halve it for the monostatic radial
    velocity, which is what a range-Doppler map's velocity axis shows.

    Parameters
    ----------
    doppler_hz : array_like
        Doppler shift, hertz, positive closing.
    wavelength_m : array_like
        Carrier wavelength, metres. Broadcast against the shift.

    Returns
    -------
    numpy.ndarray
        Range-sum rate in metres/second, positive closing, of the broadcast
        shape of the inputs.

    References
    ----------
    .. [1] Richards (2014), §2.6.1; see the module References.
    """
    result: NDArray[np.float64] = np.asarray(doppler_hz, dtype=np.float64) * np.asarray(
        wavelength_m
    )
    return result


def range_sum_rate_mps(
    velocity_enu_mps: ArrayLike,
    unit_vector_to_tx: ArrayLike,
    unit_vector_to_rx: ArrayLike,
) -> NDArray[np.float64]:
    r"""Return how fast moving targets shorten their total path, in metres/second.

    The rate at which the range sum shrinks [1]_,

    .. math::

        \dot{L}_c = -\frac{d(R_t + R_r)}{dt}
            = \vec{v}\cdot\hat{u}_t + \vec{v}\cdot\hat{u}_r

    which is equivalent to the textbook form
    :math:`\dot{L}_c = 2v\cos\delta\,\cos(\beta/2)`, with :math:`\beta` the
    bistatic angle and :math:`\delta` the angle between the velocity and the
    bisector of the two directions. The vector form is used here because a
    scenario has velocities and positions to hand, and extracting
    :math:`\delta` and :math:`\beta` first would only add a step that can go
    wrong. Collocate the sites and it is twice the radial velocity.

    The sign follows the library's **closing-positive** convention: approaching
    either site shortens that range and pushes the rate positive. Pass the
    result to :func:`doppler_shift_hz` for the shift in hertz.

    Parameters
    ----------
    velocity_enu_mps : array_like
        Target velocity in local east, north, up metres/second, shape
        ``(n_targets, 3)``.
    unit_vector_to_tx : array_like
        Unit vector from each target **towards the transmitter**, shape
        ``(n_targets, 3)``, in the same frame as ``velocity_enu_mps``.
    unit_vector_to_rx : array_like
        Unit vector from each target **towards the receiver**, same shape and
        frame.

    Returns
    -------
    numpy.ndarray
        Range-sum rate in metres/second, shape ``(n_targets,)``, positive closing.

    Raises
    ------
    ValueError
        If any of the three arrays does not have three components per target.

    Notes
    -----
    Two geometries give exactly zero rate, and both are worth recognising in a
    range-Doppler map. A target crossing the bisector at right angles is
    momentarily neither closing nor opening on the pair. A target moving **along
    the baseline** also nulls, because the transmit range shortens at exactly
    the rate the receive range lengthens; a monostatic radar has no equivalent
    of this second null, and it is why a bistatic pair is blind to some motion
    that is plainly radial to one of its two sites.

    See Also
    --------
    radar_forge.core.radar.BistaticRadar.bistatic_angle_rad : The angle in the
        textbook form of this equation.

    References
    ----------
    .. [1] Willis (2005), §6.1; see the module References.
    """
    velocity_arr = np.atleast_2d(np.asarray(velocity_enu_mps, dtype=np.float64))
    to_tx_arr = np.atleast_2d(np.asarray(unit_vector_to_tx, dtype=np.float64))
    to_rx_arr = np.atleast_2d(np.asarray(unit_vector_to_rx, dtype=np.float64))

    for name, array in (
        ("velocity_enu_mps", velocity_arr),
        ("unit_vector_to_tx", to_tx_arr),
        ("unit_vector_to_rx", to_rx_arr),
    ):
        if array.shape[-1] != 3:
            msg = f"{name} must have three components per target; got shape {array.shape}."
            raise ValueError(msg)

    # Each unit vector points from the target towards a site, so a velocity with
    # a positive component along one is closing on it and shortens that range.
    result: NDArray[np.float64] = np.sum(velocity_arr * to_tx_arr, axis=-1) + np.sum(
        velocity_arr * to_rx_arr, axis=-1
    )
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

    The monostatic case of :func:`bistatic_line_of_sight_paths`: the range sum
    is :math:`2R` and the range-sum rate :math:`2 v_r`, and from there the
    delay, phase, Doppler and amplitude come from the same functions. Only the
    power differs, coming from the monostatic range equation.

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
    radar_forge.core.signal.fmcw_deramp_baseband : Turn these paths into a deramped FMCW cube.
    radar_forge.core.signal.pulsed_baseband : Turn these paths into a pulsed cube.

    Examples
    --------
    >>> from radar_forge.core.radar import Radar, Receiver, Transmitter
    >>> tx = Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3)
    >>> radar = Radar(tx, Receiver(1.0e6, 30.0, 3.0), 36.0025, -78.941, 60.0)
    >>> paths = line_of_sight_paths(radar, 10_000.0, 80.0, 10.0)
    >>> paths.n_paths
    1
    >>> float(paths.range_sum_m[0])  # out and back
    20000.0
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

    # One site, so the departure and arrival angles are the same pair.
    angles_rad = _angle_pairs_rad(azimuth_deg, elevation_deg, n_paths)
    return _direct_paths(
        range_sum_m=2.0 * range_arr,
        range_sum_rate_mps=2.0 * velocity_arr,
        power_w=power_w,
        wavelength_m=radar.wavelength_m,
        aod_rad=angles_rad.copy(),
        aoa_rad=angles_rad,
    )


def bistatic_line_of_sight_paths(
    radar: BistaticRadar,
    range_tx_m: ArrayLike,
    range_rx_m: ArrayLike,
    range_sum_rate_mps: ArrayLike,
    bistatic_rcs_m2: ArrayLike,
    *,
    transmit_azimuth_deg: ArrayLike = 0.0,
    transmit_elevation_deg: ArrayLike = 0.0,
    receive_azimuth_deg: ArrayLike = 0.0,
    receive_elevation_deg: ArrayLike = 0.0,
) -> PropagationPaths:
    r"""Build the direct-return paths for point targets seen by a bistatic pair.

    One path per target, single bounce, no multipath and no obstruction. The
    delay, phase, Doppler and amplitude are those of the module's one geometry,
    applied to the range sum :math:`R_t + R_r`; the power comes from the
    bistatic radar range equation, so ``abs(amplitude_linear) ** 2`` is the
    received power in watts. The generators in :mod:`radar_forge.core.signal`
    consume these paths exactly as they consume monostatic ones.

    Parameters
    ----------
    radar : BistaticRadar
        The observing pair; supplies wavelength, power and antenna gains.
    range_tx_m : array_like
        Transmit range to each target — transmitter to target, metres. Strictly
        positive.
    range_rx_m : array_like
        Receive range to each target — target to receiver, metres. Strictly
        positive.
    range_sum_rate_mps : array_like
        The rate at which the total path :math:`R_t + R_r` is getting shorter,
        metres/second, **positive closing**. See :func:`range_sum_rate_mps`.
    bistatic_rcs_m2 : array_like
        Bistatic radar cross-section of each target, square metres, at this
        geometry. Zero is permitted and produces a null path. Not in general the
        monostatic cross-section — see
        :func:`radar_forge.core.radar_equation.bistatic_received_power_w`.
    transmit_azimuth_deg, transmit_elevation_deg : array_like, optional
        Angles of departure at the **transmitter** site, degrees, in the
        library's compass convention. Default 0, i.e. boresight.
    receive_azimuth_deg, receive_elevation_deg : array_like, optional
        Angles of arrival at the **receiver** site, degrees. Default 0.

    Returns
    -------
    PropagationPaths
        One path per target, in the order given. Unlike the monostatic case the
        ``aod_rad`` and ``aoa_rad`` fields genuinely differ, because departure
        and arrival happen at different places.

    Raises
    ------
    ValueError
        If any transmit or receive range is non-positive, or any cross-section
        is negative.

    Notes
    -----
    The delay carried here is the **absolute** one, :math:`(R_t + R_r)/c`. A real
    bistatic receiver times echoes against the direct pulse from the
    transmitter, so what it actually measures is the range sum minus the
    baseline, :math:`R_t + R_r - L`, and its zero of range sits at the baseline
    rather than at the receiver. Keeping the absolute delay here is deliberate —
    it is what carries the correct carrier phase — and subtracting the baseline
    belongs to a synchronisation model this library does not yet have.

    See Also
    --------
    line_of_sight_paths : The monostatic case.
    range_sum_rate_mps : Computes the rate this takes, from velocity and
        geometry vectors.

    Examples
    --------
    >>> from radar_forge.core.radar import BistaticRadar, Receiver, Transmitter
    >>> tx = Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3)
    >>> pair = BistaticRadar(
    ...     tx, Receiver(1.0e6, 30.0, 3.0), 36.0025, -78.941, 60.0, 36.0025, -78.741, 60.0
    ... )
    >>> paths = bistatic_line_of_sight_paths(pair, 12_000.0, 15_000.0, 160.0, 10.0)
    >>> float(paths.range_sum_m[0])
    27000.0
    >>> bool(paths.doppler_hz[0] > 0.0)  # closing is positive
    True
    """
    range_tx_arr = np.atleast_1d(np.asarray(range_tx_m, dtype=np.float64))
    range_rx_arr = np.atleast_1d(np.asarray(range_rx_m, dtype=np.float64))
    rate_arr = np.atleast_1d(np.asarray(range_sum_rate_mps, dtype=np.float64))
    rcs_arr = np.atleast_1d(np.asarray(bistatic_rcs_m2, dtype=np.float64))

    if np.any(range_tx_arr <= 0.0) or np.any(range_rx_arr <= 0.0):
        msg = (
            "range_tx_m and range_rx_m must be strictly positive; a target standing at either "
            "site is a modelling error."
        )
        raise ValueError(msg)
    if np.any(rcs_arr < 0.0):
        msg = "bistatic_rcs_m2 must be non-negative; use 0.0 for a target that returns nothing."
        raise ValueError(msg)

    range_tx_arr, range_rx_arr, rate_arr, rcs_arr = np.broadcast_arrays(
        range_tx_arr, range_rx_arr, rate_arr, rcs_arr
    )
    n_paths = range_tx_arr.shape[0]

    # As in the monostatic case, a legitimate 0 m^2 target would divide by a
    # zero-power path, so substitute and zero the amplitude afterwards.
    illuminated = rcs_arr > 0.0
    power_w = np.zeros_like(range_tx_arr)
    power_w[illuminated] = bistatic_received_power_w(
        transmit_power_w=radar.transmitter.transmit_power_w,
        gain_tx_linear=10.0 ** (radar.transmitter.gain_tx_dbi / 10.0),
        gain_rx_linear=10.0 ** (radar.receiver.gain_rx_dbi / 10.0),
        wavelength_m=radar.wavelength_m,
        bistatic_rcs_m2=rcs_arr[illuminated],
        range_tx_m=range_tx_arr[illuminated],
        range_rx_m=range_rx_arr[illuminated],
    )

    return _direct_paths(
        range_sum_m=range_tx_arr + range_rx_arr,
        range_sum_rate_mps=rate_arr,
        power_w=power_w,
        wavelength_m=radar.wavelength_m,
        aod_rad=_angle_pairs_rad(transmit_azimuth_deg, transmit_elevation_deg, n_paths),
        aoa_rad=_angle_pairs_rad(receive_azimuth_deg, receive_elevation_deg, n_paths),
    )


def _direct_paths(
    *,
    range_sum_m: NDArray[np.float64],
    range_sum_rate_mps: NDArray[np.float64],
    power_w: NDArray[np.float64],
    wavelength_m: float,
    aod_rad: NDArray[np.float64],
    aoa_rad: NDArray[np.float64],
) -> PropagationPaths:
    """Assemble single-bounce paths from their range sum, rate and power.

    The part both builders share: the module's one geometry, applied once.
    """
    phase_rad = propagation_phase_rad(range_sum_m, wavelength_m)
    return PropagationPaths(
        delay_s=propagation_delay_s(range_sum_m),
        doppler_hz=doppler_shift_hz(range_sum_rate_mps, wavelength_m),
        amplitude_linear=path_amplitude_linear(power_w, phase_rad),
        aoa_rad=aoa_rad,
        aod_rad=aod_rad,
        bounce_count=np.ones(range_sum_m.shape[0], dtype=np.int_),
    )


def _angle_pairs_rad(
    azimuth_deg: ArrayLike, elevation_deg: ArrayLike, n_paths: int
) -> NDArray[np.float64]:
    """Return ``(n_paths, 2)`` radian (azimuth, elevation) pairs from degrees.

    The zeros array is a broadcasting device, not data: it forces a scalar angle
    up to one value per path without materialising a tiled copy.
    """
    pairs: NDArray[np.float64] = np.stack(
        np.broadcast_arrays(
            np.radians(np.asarray(azimuth_deg, dtype=np.float64)),
            np.radians(np.asarray(elevation_deg, dtype=np.float64)),
            np.zeros(n_paths),
        )[:2],
        axis=-1,
    )
    return pairs
