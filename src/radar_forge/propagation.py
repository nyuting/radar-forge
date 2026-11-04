r"""Propagation paths: how energy gets from the transmitter, via the scene, to the receiver.

Every propagation model, from the line-of-sight model here to a future ray
tracer, produces a :class:`PropagationPaths`, and :mod:`radar_forge.signal`
turns it into samples (``spec/structure.md`` D1). A path is the whole route
from transmitter to receiver. Everything it carries follows from the length of
that route, the bistatic range :math:`R_b = R_T + R_R`, and its rate of change
:math:`\dot R_b`, positive receding [1]_, [2]_:

.. math::

    \tau = \frac{R_b}{c},
    \qquad
    \phi = -\frac{2\pi R_b}{\lambda},
    \qquad
    f_D = -\frac{\dot R_b}{\lambda}

A monostatic radar is :math:`R_T = R_R = R`, where these become the familiar
:math:`2R/c`, :math:`-4\pi R/\lambda` and :math:`-2\dot R/\lambda`. The Doppler
sign is written once, in :func:`doppler_shift_hz`: a closing path shortens, so
its Doppler shift is positive.

The model is **free space**: no absorption, multipath or clutter. Power comes
from the bistatic range equation, :func:`received_power_w`. Targets are points
whose cross-section is constant (Swerling 0) or drawn by
:func:`swerling_rcs_m2`.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §2.2.1 (the radar range equation), §2.6.1 (the
       two-way phase and the Doppler shift), §2.6.2 (stop-and-hop).
.. [2] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005, §4.1,
       eq. (4.1) (the bistatic range equation), §6.1 (target Doppler).
.. [3] P. Swerling, "Probability of detection for fluctuating targets," *IRE
       Transactions on Information Theory*, vol. IT-6, no. 2, pp. 269-308, 1960.
.. [4] M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill,
       2001, §2.7 (radar cross section), §2.8 (cross-section fluctuations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.radar import Radar, TargetGeometry

__all__ = [
    "PointTarget",
    "PropagationPaths",
    "bistatic_range_rate_from_doppler_mps",
    "dbsm_to_m2",
    "doppler_shift_hz",
    "line_of_sight_paths",
    "m2_to_dbsm",
    "path_amplitude_linear",
    "propagation_delay_s",
    "propagation_phase_rad",
    "received_power_w",
    "swerling_rcs_m2",
]

_FOUR_PI_CUBED = (4.0 * np.pi) ** 3

# Swerling 1-4 as chi-square degrees of freedom; Swerling 0 is the infinite limit.
_SWERLING_DEGREES_OF_FREEDOM = {1: 2, 2: 2, 3: 4, 4: 4}


@dataclass(frozen=True)
class PropagationPaths:
    """A set of propagation paths between a radar and its scene.

    Parameters
    ----------
    delay_s : numpy.ndarray
        Delay along each path, transmitter to receiver, s, ``(n_paths,)``.
    doppler_hz : numpy.ndarray
        Doppler shift of each path, Hz, ``(n_paths,)``, from
        :func:`doppler_shift_hz`: positive for a path that is shortening.
    amplitude_linear : numpy.ndarray
        Complex voltage amplitude of each path, √W, ``(n_paths,)``.
        ``abs(amplitude_linear) ** 2`` is the received power, on the scale of
        :attr:`~radar_forge.radar.Receiver.noise_power_per_sample_w`. The
        generators use only its magnitude and rebuild the phase from the delay,
        pulse by pulse.
    aoa_rad, aod_rad : numpy.ndarray
        (azimuth, elevation) of arrival at the receiver and of departure from
        the transmitter, rad, ``(n_paths, 2)``.
    n_bounces : numpy.ndarray
        Interactions along each path, ``(n_paths,)``; one for a direct return.
    doppler_rate_hzps : numpy.ndarray, optional
        Rate of change of each Doppler shift, Hz/s, ``(n_paths,)``, or one value
        for every path. Default zero: constant velocity. Stored per path.

    Raises
    ------
    ValueError
        If a per-path array does not have ``n_paths`` entries, or an angle
        array is not ``(n_paths, 2)``.
    """

    delay_s: NDArray[np.float64]
    doppler_hz: NDArray[np.float64]
    amplitude_linear: NDArray[np.complex128]
    aoa_rad: NDArray[np.float64]
    aod_rad: NDArray[np.float64]
    n_bounces: NDArray[np.int_]
    doppler_rate_hzps: NDArray[np.float64] = field(default_factory=lambda: np.zeros(()))

    def __post_init__(self) -> None:
        """Validate the per-path shapes; see the class ``Raises`` section."""
        n_paths = self.delay_s.shape[0]
        for name in ("doppler_hz", "amplitude_linear", "n_bounces"):
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
        if np.shape(self.doppler_rate_hzps) not in ((), (n_paths,)):
            msg = f"doppler_rate_hzps must be a scalar or have shape ({n_paths},)."
            raise ValueError(msg)
        # Store one value per path, so every per-path field indexes alike.
        object.__setattr__(self, "doppler_rate_hzps", np.zeros(n_paths) + self.doppler_rate_hzps)

    @property
    def n_paths(self) -> int:
        """Number of propagation paths."""
        return int(self.delay_s.shape[0])

    @property
    def bistatic_range_m(self) -> NDArray[np.float64]:
        r"""Length of each route, :math:`R_b = c\tau`, m, ``(n_paths,)``. Monostatic, :math:`2R`."""
        result: NDArray[np.float64] = self.delay_s * SPEED_OF_LIGHT_MPS
        return result

    def bistatic_range_at_m(
        self, slow_time_s: ArrayLike, wavelength_m: float
    ) -> NDArray[np.float64]:
        r"""Return the length of each route at the given slow-time instants, m.

        Integrate the rate, which :func:`doppler_shift_hz` defines through
        :math:`f_D = -\dot R_b/\lambda`, so :math:`\dot R_b = -\lambda f_D` and
        :math:`\ddot R_b = -\lambda \dot f_D`:

        .. math::

            R_b(t) = R_b(0) + \dot R_b t + \tfrac{1}{2}\ddot R_b t^2
                   = R_b(0) - \lambda\left(f_D t + \tfrac{1}{2}\dot f_D t^2\right)

        A closing path (:math:`f_D > 0`) shortens. The generators evaluate this
        at each pulse start and hold it across the pulse: stop-and-hop [1]_.

        Parameters
        ----------
        slow_time_s : array_like
            Instants after the time this path set describes, s; normally the
            ``(n_pulses,)`` pulse start times.
        wavelength_m : float
            The carrier wavelength ``doppler_hz`` was computed with, m.

        Returns
        -------
        numpy.ndarray
            Bistatic range, m, ``slow_time_s.shape + (n_paths,)``.

        Examples
        --------
        >>> paths = PropagationPaths(
        ...     np.array([1.0e-4]), np.array([100.0]), np.array([1.0 + 0.0j]),
        ...     np.zeros((1, 2)), np.zeros((1, 2)), np.array([1]),
        ... )
        >>> start_m, later_m = paths.bistatic_range_at_m([0.0, 1.0], wavelength_m=0.03)[:, 0]
        >>> round(float(start_m - later_m), 9)  # 100 Hz x 3 cm over 1 s, closing
        3.0
        """
        time_s = np.asarray(slow_time_s, dtype=np.float64)[..., None]
        rate_mps = bistatic_range_rate_from_doppler_mps(self.doppler_hz, wavelength_m)
        acceleration_mps2 = bistatic_range_rate_from_doppler_mps(
            self.doppler_rate_hzps, wavelength_m
        )
        result: NDArray[np.float64] = (
            self.bistatic_range_m + rate_mps * time_s + 0.5 * acceleration_mps2 * time_s**2
        )
        return result


def propagation_delay_s(bistatic_range_m: ArrayLike) -> NDArray[np.float64]:
    r"""Return the time to travel a route of :math:`R_b` metres, :math:`\tau = R_b/c`, s [1]_."""
    result: NDArray[np.float64] = (
        np.asarray(bistatic_range_m, dtype=np.float64) / SPEED_OF_LIGHT_MPS
    )
    return result


def propagation_phase_rad(
    bistatic_range_m: ArrayLike, wavelength_m: ArrayLike
) -> NDArray[np.float64]:
    r"""Return the carrier phase a route imposes, :math:`\phi = -2\pi R_b/\lambda`, rad [1]_.

    One cycle of lag per wavelength of route, unwrapped; its change from pulse
    to pulse is what a Doppler filter measures. Inputs in m, broadcast together.
    """
    range_arr = np.asarray(bistatic_range_m, dtype=np.float64)
    result: NDArray[np.float64] = -2.0 * np.pi * range_arr / np.asarray(wavelength_m)
    return result


def doppler_shift_hz(
    bistatic_range_rate_mps: ArrayLike, wavelength_m: ArrayLike
) -> NDArray[np.float64]:
    r"""Return the Doppler shift of a route, :math:`f_D = -\dot R_b/\lambda`, Hz.

    The one place the sign convention lives [1]_, [2]_: the rate
    :math:`dR_b/dt` (m/s) is positive receding, the shift positive closing.
    Monostatic, :math:`-2\dot R/\lambda`. Inputs broadcast together.

    Examples
    --------
    >>> float(doppler_shift_hz(30.0, 0.03))  # receding at 30 m/s: negative
    -1000.0
    """
    rate_arr = np.asarray(bistatic_range_rate_mps, dtype=np.float64)
    result: NDArray[np.float64] = -rate_arr / np.asarray(wavelength_m)
    return result


def bistatic_range_rate_from_doppler_mps(
    doppler_hz: ArrayLike, wavelength_m: ArrayLike
) -> NDArray[np.float64]:
    r"""Return the rate :math:`dR_b/dt = -\lambda f_D` behind a Doppler shift, m/s [1]_.

    The inverse of :func:`doppler_shift_hz`: Doppler positive closing in, rate
    positive receding out. Inputs broadcast together.
    """
    result: NDArray[np.float64] = -np.asarray(doppler_hz, dtype=np.float64) * np.asarray(
        wavelength_m
    )
    return result


def path_amplitude_linear(power_w: ArrayLike, phase_rad: ArrayLike) -> NDArray[np.complex128]:
    r"""Return a path's complex voltage amplitude, :math:`a = \sqrt{P_r}\,e^{j\phi}`, √W.

    ``abs(a) ** 2`` is the received power ``power_w`` (W, non-negative), so a
    cube built from these amplitudes is on the scale of the receiver noise [1]_.
    ``phase_rad`` broadcasts against it.
    """
    amplitude = np.sqrt(np.asarray(power_w, dtype=np.float64)) * np.exp(
        1j * np.asarray(phase_rad, dtype=np.float64)
    )
    return np.asarray(amplitude, dtype=np.complex128)


def received_power_w(
    *,
    transmit_power_w: ArrayLike,
    gain_tx_linear: ArrayLike,
    gain_rx_linear: ArrayLike,
    wavelength_m: ArrayLike,
    rcs_m2: ArrayLike,
    range_tx_m: ArrayLike,
    range_rx_m: ArrayLike,
    loss_linear: ArrayLike = 1.0,
) -> NDArray[np.float64]:
    r"""Return the power received from a point target, W.

    The bistatic radar range equation [2]_

    .. math::

        P_r = \frac{P_t\, G_t\, G_r\, \lambda^2\, \sigma}{(4\pi)^3\, R_T^2\, R_R^2\, L}

    With :math:`R_T = R_R = R` it is the monostatic
    :math:`P_t G_t G_r \lambda^2 \sigma / (4\pi)^3 R^4 L` [1]_. The product
    :math:`R_T^2 R_R^2`, not a power of the sum, makes the contours of constant
    power ovals of Cassini. All arguments are linear SI and broadcast together.

    Parameters
    ----------
    transmit_power_w : array_like
        Transmitted power :math:`P_t`, W.
    gain_tx_linear, gain_rx_linear : array_like
        Antenna gains, linear (30 dBi is ``1000.0``).
    wavelength_m : array_like
        Carrier wavelength, m.
    rcs_m2 : array_like
        Cross-section :math:`\sigma`, m². Zero gives zero power. Bistatically
        it is the bistatic cross-section, taken as given (``spec/structure.md`` D8).
    range_tx_m, range_rx_m : array_like
        Transmit range :math:`R_T` and receive range :math:`R_R`, m.
    loss_linear : array_like, optional
        Loss factor :math:`L \ge 1`, linear. Default 1.0, lossless.

    Returns
    -------
    numpy.ndarray
        Received power, W, of the broadcast shape.

    Notes
    -----
    Inputs are not validated: this runs once per frame. A zero range gives
    ``inf`` and NumPy's divide-by-zero warning. Forward scatter is not modelled.

    Examples
    --------
    >>> near_w, far_w = received_power_w(
    ...     transmit_power_w=1.0, gain_tx_linear=1e3, gain_rx_linear=1e3,
    ...     wavelength_m=0.03, rcs_m2=10.0, range_tx_m=[100.0, 200.0], range_rx_m=[100.0, 200.0],
    ... )
    >>> round(float(near_w / far_w), 9)  # doubling a monostatic range costs 12 dB
    16.0
    """
    numerator = (
        np.asarray(transmit_power_w, dtype=np.float64)
        * np.asarray(gain_tx_linear, dtype=np.float64)
        * np.asarray(gain_rx_linear, dtype=np.float64)
        * np.asarray(wavelength_m, dtype=np.float64) ** 2
        * np.asarray(rcs_m2, dtype=np.float64)
    )
    denominator = (
        _FOUR_PI_CUBED
        * np.asarray(range_tx_m, dtype=np.float64) ** 2
        * np.asarray(range_rx_m, dtype=np.float64) ** 2
        * np.asarray(loss_linear, dtype=np.float64)
    )
    result: NDArray[np.float64] = numerator / denominator
    return result


def line_of_sight_paths(
    radar: Radar,
    geometry: TargetGeometry,
    rcs_m2: ArrayLike,
    *,
    gain_tx_linear: ArrayLike | None = None,
    gain_rx_linear: ArrayLike | None = None,
) -> PropagationPaths:
    r"""Build the direct-return paths for point targets in free space.

    One single-bounce path per target, with delay, phase and Doppler from the
    route geometry and power from :func:`received_power_w`; monostatic and
    bistatic radars alike.

    Parameters
    ----------
    radar : radar_forge.radar.Radar
        The observing radar: wavelength, power and antenna gains.
    geometry : radar_forge.radar.TargetGeometry
        Where each target is, ``(n_targets,)``.
    rcs_m2 : array_like
        Cross-section of each target, m², broadcast to ``(n_targets,)``. Zero
        gives a null path, which is how a noise-only scene is built.
    gain_tx_linear, gain_rx_linear : array_like, optional
        Antenna gain towards each target, linear. Default: the boresight gains
        ``gain_tx_dbi`` and ``gain_rx_dbi``, converted.

    Returns
    -------
    PropagationPaths
        One path per target, in order.

    Examples
    --------
    >>> from radar_forge.radar import Receiver, Site, Transmitter
    >>> tx = Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3)
    >>> radar = Radar(tx, Receiver(1.0e6, 30.0, 3.0), Site(36.0025, -78.941, 60.0))
    >>> paths = line_of_sight_paths(radar, TargetGeometry.monostatic(10_000.0, -80.0), 10.0)
    >>> float(paths.bistatic_range_m[0]), bool(paths.doppler_hz[0] > 0.0)  # closing
    (20000.0, True)
    """
    if gain_tx_linear is None:
        gain_tx_linear = 10.0 ** (radar.transmitter.gain_tx_dbi / 10.0)
    if gain_rx_linear is None:
        gain_rx_linear = 10.0 ** (radar.receiver.gain_rx_dbi / 10.0)
    power_w = received_power_w(
        transmit_power_w=radar.transmitter.transmit_power_w,
        gain_tx_linear=gain_tx_linear,
        gain_rx_linear=gain_rx_linear,
        wavelength_m=radar.wavelength_m,
        rcs_m2=rcs_m2,
        range_tx_m=geometry.range_tx_m,
        range_rx_m=geometry.range_rx_m,
    )
    phase_rad = propagation_phase_rad(geometry.bistatic_range_m, radar.wavelength_m)
    return PropagationPaths(
        delay_s=propagation_delay_s(geometry.bistatic_range_m),
        doppler_hz=doppler_shift_hz(geometry.bistatic_range_rate_mps, radar.wavelength_m),
        amplitude_linear=path_amplitude_linear(power_w, phase_rad),
        aoa_rad=geometry.aoa_rad,
        aod_rad=geometry.aod_rad,
        n_bounces=np.ones(geometry.bistatic_range_m.shape[0], dtype=np.int_),
    )


def dbsm_to_m2(rcs_dbsm: ArrayLike) -> NDArray[np.float64]:
    """Convert a cross-section from dBsm (dB relative to one square metre) to m² [4]_.

    Examples
    --------
    >>> float(dbsm_to_m2(20.0))
    100.0
    """
    return np.asarray(10.0 ** (np.asarray(rcs_dbsm, dtype=np.float64) / 10.0), dtype=np.float64)


def m2_to_dbsm(rcs_m2: ArrayLike) -> NDArray[np.float64]:
    """Convert a strictly positive cross-section from m² to dBsm [4]_.

    Raises
    ------
    ValueError
        If any cross-section is non-positive, which has no decibel value.
    """
    rcs_arr = np.asarray(rcs_m2, dtype=np.float64)
    if np.any(rcs_arr <= 0.0):
        msg = "rcs_m2 must be strictly positive; a non-positive area has no decibel value."
        raise ValueError(msg)
    return np.asarray(10.0 * np.log10(rcs_arr), dtype=np.float64)


@dataclass(frozen=True)
class PointTarget:
    """A point target with a constant cross-section (Swerling 0).

    Smaller than a range bin, with one cross-section for every aspect and
    pulse. Draw a fluctuating one with :func:`swerling_rcs_m2`.

    Parameters
    ----------
    rcs_m2 : float
        Cross-section, m². 10.0 (10 dBsm) suits a light aircraft.
    name : str
        A label carried to the truth output. Default ``"target"``.

    Raises
    ------
    ValueError
        If ``rcs_m2`` is negative. Zero is allowed, for a noise-only scene.

    Examples
    --------
    >>> round(PointTarget.from_dbsm(10.0).rcs_m2, 6)
    10.0
    """

    rcs_m2: float
    name: str = "target"

    def __post_init__(self) -> None:
        """Validate the cross-section; see the class ``Raises`` section."""
        if self.rcs_m2 < 0.0:
            msg = f"rcs_m2 must be non-negative; got {self.rcs_m2!r}. Use 0.0 for noise only."
            raise ValueError(msg)

    @classmethod
    def from_dbsm(cls, rcs_dbsm: float, name: str = "target") -> PointTarget:
        """Return a target whose cross-section is given in dBsm, the usual way it is quoted."""
        return cls(rcs_m2=float(dbsm_to_m2(rcs_dbsm)), name=name)

    @property
    def rcs_dbsm(self) -> float:
        """Cross-section, dBsm; ``-inf`` for a zero cross-section."""
        if self.rcs_m2 == 0.0:
            return float("-inf")
        return float(m2_to_dbsm(self.rcs_m2))


def swerling_rcs_m2(
    mean_rcs_m2: ArrayLike, case: int, shape: tuple[int, ...], *, rng: np.random.Generator
) -> NDArray[np.float64]:
    r"""Draw radar cross-sections from a Swerling fluctuation model.

    Swerling's models [3]_, [4]_ are two densities of :math:`\sigma` about its
    mean :math:`\bar\sigma`, each redrawn at one of two rates:

    ======  =====================================================  ==============
    case    density                                                redrawn
    ======  =====================================================  ==============
    0       none: :math:`\sigma = \bar\sigma`                      never
    1, 2    exponential, :math:`e^{-\sigma/\bar\sigma}/\bar\sigma`  scan, pulse
    3, 4    chi-square, 4 degrees of freedom,                      scan, pulse
            :math:`(4\sigma/\bar\sigma^2)\,e^{-2\sigma/\bar\sigma}`
    ======  =====================================================  ==============

    Both are :math:`\bar\sigma\chi^2_k/k`, with :math:`k = 2` or :math:`4`.

    Parameters
    ----------
    mean_rcs_m2 : array_like
        Mean cross-section :math:`\bar\sigma`, m², broadcast against ``shape``.
    case : {0, 1, 2, 3, 4}
        Swerling case.
    shape : tuple of int
        Shape of the draw.
    rng : numpy.random.Generator
        Seeded generator.

    Returns
    -------
    numpy.ndarray
        Independent draws, m², of ``shape``. The rate is the caller's to set:
        ask for one per scan (``(n_scans,)``) and hold it across the scan's
        pulses for cases 1 and 3, one per pulse (``(n_scans, n_pulses)``) for 2 and 4.

    Raises
    ------
    ValueError
        If ``case`` is not 0, 1, 2, 3 or 4.

    Notes
    -----
    The variance is :math:`\bar\sigma^2` for cases 1 and 2 and :math:`\bar\sigma^2/2`
    for 3 and 4: one dominant scatterer among many small ones fluctuates less.
    """
    if case != 0 and case not in _SWERLING_DEGREES_OF_FREEDOM:
        msg = f"case must be a Swerling case 0, 1, 2, 3 or 4; got {case!r}."
        raise ValueError(msg)
    mean_arr = np.asarray(mean_rcs_m2, dtype=np.float64)
    if case == 0:
        return np.zeros(shape) + mean_arr
    degrees_of_freedom = _SWERLING_DEGREES_OF_FREEDOM[case]
    draws: NDArray[np.float64] = (
        mean_arr * rng.chisquare(degrees_of_freedom, size=shape) / degrees_of_freedom
    )
    return draws
