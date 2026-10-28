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

Bistatic geometry
-----------------
:func:`bistatic_line_of_sight_paths` builds paths for a transmitter and receiver
at different sites, and the generators in :mod:`radar_forge.core.signal` consume
them **unchanged**. That works because of how a path reports its range and
Doppler:

.. math::

    \mathrm{range\_m} = \frac{R_t + R_r}{2},
    \qquad
    \mathrm{bisector\_velocity\_mps} = \frac{1}{2}\frac{d(R_t + R_r)}{dt}

Every use of range in a generator is through :math:`\tau = 2\,R/c`, and every use
of Doppler is through :math:`v = f_d \lambda / 2`. Feeding them half the total
path length and half its rate of change therefore produces exactly the bistatic
delay :math:`(R_t + R_r)/c`, the bistatic propagation phase
:math:`-2\pi (R_t + R_r)/\lambda`, and the correct slow-time evolution — the
factor of two that the monostatic convention puts in cancels the one the
definition takes out. A bistatic scenario costs no new generator code, only a
different path builder.

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
    "bistatic_doppler_hz",
    "bistatic_line_of_sight_paths",
    "line_of_sight_paths",
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
        Two-way propagation delay of each path, shape ``(n_paths,)``, seconds.
    doppler_hz : numpy.ndarray
        Doppler shift of each path, shape ``(n_paths,)``, hertz. **Positive is
        closing**, so ``doppler_hz = 2 * closing_velocity_mps / wavelength_m``.
    amplitude_linear : numpy.ndarray
        Complex voltage amplitude of each path, shape ``(n_paths,)``. Its squared
        magnitude is the received power in watts, so that a cube built from it
        sits on the same scale as :attr:`Radar.noise_power_w`.

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
    def range_m(self) -> NDArray[np.float64]:
        """One-way equivalent range of each path, metres, from the delay.

        For a monostatic path this is the range to the target. In general it is
        **half the total propagation path length**, :math:`(R_t + R_r)/2` — the
        range a monostatic radar would have to be at to produce the same delay.
        The two coincide only when the transmitter and receiver are collocated.

        This is the convention the signal generators are built on, so changing
        it would silently break every one of them; see the module docstring.
        For a bistatic path it also means the surfaces of constant
        ``range_m`` are ellipsoids with the two sites at the foci, not spheres,
        and that :func:`radar_forge.core.dsp.range_bin_centers_m` labels a
        bistatic map in this half-sum rather than in distance to anything.
        """
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

    angles_rad = _angle_pairs_rad(azimuth_deg, elevation_deg, n_paths)

    return PropagationPaths(
        delay_s=2.0 * range_arr / SPEED_OF_LIGHT_MPS,
        doppler_hz=2.0 * velocity_arr / radar.wavelength_m,
        amplitude_linear=np.asarray(amplitude_linear, dtype=np.complex128),
        aoa_rad=angles_rad,
        aod_rad=angles_rad.copy(),
        bounce_count=np.ones(n_paths, dtype=np.int_),
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


def bistatic_doppler_hz(
    wavelength_m: ArrayLike,
    velocity_enu_mps: ArrayLike,
    unit_vector_to_tx: ArrayLike,
    unit_vector_to_rx: ArrayLike,
) -> NDArray[np.float64]:
    r"""Return the bistatic Doppler shift of moving point targets, in hertz.

    The shift is the rate at which the total path length changes [2]_,

    .. math::

        f_d = -\frac{1}{\lambda}\frac{d(R_t + R_r)}{dt}
            = \frac{1}{\lambda}\left(
                \vec{v}\cdot\hat{u}_t + \vec{v}\cdot\hat{u}_r\right)

    which is equivalent to the textbook form
    :math:`f_d = (2v/\lambda)\cos\delta\,\cos(\beta/2)`, with :math:`\beta` the
    bistatic angle and :math:`\delta` the angle between the velocity and the
    bisector of the two directions. The vector form is used here because a
    scenario has velocities and positions to hand, and extracting
    :math:`\delta` and :math:`\beta` first would only add a step that can go
    wrong.

    The sign follows the library's **closing-positive** convention: approaching
    either site shortens that range and pushes the shift positive.

    Parameters
    ----------
    wavelength_m : array_like
        Carrier wavelength, metres.
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
        Doppler shift in hertz, shape ``(n_targets,)``, positive closing.

    Notes
    -----
    Two geometries give exactly zero shift, and both are worth recognising in a
    range-Doppler map. A target crossing the bisector at right angles is
    momentarily neither closing nor opening on the pair. A target moving **along
    the baseline** also nulls, because the transmit range shortens at exactly
    the rate the receive range lengthens; a monostatic radar has no equivalent
    of this second null, and it is why a bistatic pair is blind to some motion
    that is plainly radial to one of its two sites.

    See Also
    --------
    bistatic_line_of_sight_paths : Consumes the bisector range rate this implies.
    radar_forge.core.radar.BistaticRadar.bistatic_angle_rad : The angle in the
        textbook form of this equation.
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
    # a positive component along one is closing on it and shortens that range:
    # d(R_t + R_r)/dt is the negative of this sum, and the Doppler shift is the
    # negative of that again. Collocate the sites and it becomes the monostatic
    # 2 v / lambda.
    closing_rate_mps = np.sum(velocity_arr * to_tx_arr, axis=-1) + np.sum(
        velocity_arr * to_rx_arr, axis=-1
    )
    result: NDArray[np.float64] = closing_rate_mps / np.asarray(wavelength_m, dtype=np.float64)
    return result


def bistatic_line_of_sight_paths(
    radar: BistaticRadar,
    range_tx_m: ArrayLike,
    range_rx_m: ArrayLike,
    bisector_velocity_mps: ArrayLike,
    bistatic_rcs_m2: ArrayLike,
    *,
    transmit_azimuth_deg: ArrayLike = 0.0,
    transmit_elevation_deg: ArrayLike = 0.0,
    receive_azimuth_deg: ArrayLike = 0.0,
    receive_elevation_deg: ArrayLike = 0.0,
) -> PropagationPaths:
    r"""Build the direct-return paths for point targets seen by a bistatic pair.

    The bistatic counterpart of :func:`line_of_sight_paths`: one path per target,
    single bounce, no multipath and no obstruction. Amplitude is the square root
    of the bistatic radar range equation, so ``abs(amplitude_linear) ** 2`` is
    the received power in watts, and the phase is the propagation phase over the
    whole transmitter-target-receiver route, :math:`-2\pi (R_t + R_r)/\lambda`.

    The returned paths carry :math:`(R_t + R_r)/2` as their ``range_m`` and are
    consumed by :func:`~radar_forge.core.signal.fmcw_deramp_baseband` and
    :func:`~radar_forge.core.signal.pulsed_baseband` unchanged; the module
    docstring explains why that works.

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
    bisector_velocity_mps : array_like
        Half the rate of change of the total path length,
        :math:`\tfrac{1}{2}\,d(R_t + R_r)/dt`, metres/second, **positive
        closing**. This is the bistatic stand-in for a radial velocity; it is
        ``bistatic_doppler_hz(...) * wavelength_m / 2`` and is defined with the
        same factor of two so that a bistatic target lands in the Doppler bin a
        monostatic one with this closing rate would.
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
    line_of_sight_paths : The monostatic equivalent, which this reduces to when
        the two sites are collocated.
    bistatic_doppler_hz : Computes the bisector rate this takes, from velocity
        and geometry vectors.

    Examples
    --------
    >>> from radar_forge.core.radar import BistaticRadar, Receiver, Transmitter
    >>> tx = Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3)
    >>> pair = BistaticRadar(
    ...     tx, Receiver(1.0e6, 30.0, 3.0), 36.0025, -78.941, 60.0, 36.0025, -78.741, 60.0
    ... )
    >>> paths = bistatic_line_of_sight_paths(pair, 12_000.0, 15_000.0, 80.0, 10.0)
    >>> float(paths.range_m[0])  # the half-sum, not either range
    13500.0
    >>> bool(paths.doppler_hz[0] > 0.0)  # closing is positive
    True
    """
    range_tx_arr = np.atleast_1d(np.asarray(range_tx_m, dtype=np.float64))
    range_rx_arr = np.atleast_1d(np.asarray(range_rx_m, dtype=np.float64))
    velocity_arr = np.atleast_1d(np.asarray(bisector_velocity_mps, dtype=np.float64))
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

    range_tx_arr, range_rx_arr, velocity_arr, rcs_arr = np.broadcast_arrays(
        range_tx_arr, range_rx_arr, velocity_arr, rcs_arr
    )
    n_paths = range_tx_arr.shape[0]
    sum_range_m = range_tx_arr + range_rx_arr

    # As in the monostatic case, a legitimate 0 m^2 target would divide by a
    # zero-power path, so substitute and zero the amplitude afterwards.
    illuminated = rcs_arr > 0.0
    power_w = np.zeros_like(sum_range_m)
    power_w[illuminated] = bistatic_received_power_w(
        transmit_power_w=radar.transmitter.transmit_power_w,
        gain_tx_linear=10.0 ** (radar.transmitter.gain_tx_dbi / 10.0),
        gain_rx_linear=10.0 ** (radar.receiver.gain_rx_dbi / 10.0),
        wavelength_m=radar.wavelength_m,
        bistatic_rcs_m2=rcs_arr[illuminated],
        range_tx_m=range_tx_arr[illuminated],
        range_rx_m=range_rx_arr[illuminated],
    )

    # The whole route, not twice one range: this is the monostatic -4 pi R /
    # lambda generalised, and it collapses to it when the two ranges are equal.
    propagation_phase_rad = -2.0 * np.pi * sum_range_m / radar.wavelength_m
    amplitude_linear = np.sqrt(power_w) * np.exp(1j * propagation_phase_rad)

    return PropagationPaths(
        delay_s=sum_range_m / SPEED_OF_LIGHT_MPS,
        doppler_hz=2.0 * velocity_arr / radar.wavelength_m,
        amplitude_linear=np.asarray(amplitude_linear, dtype=np.complex128),
        aod_rad=_angle_pairs_rad(transmit_azimuth_deg, transmit_elevation_deg, n_paths),
        aoa_rad=_angle_pairs_rad(receive_azimuth_deg, receive_elevation_deg, n_paths),
        bounce_count=np.ones(n_paths, dtype=np.int_),
    )
