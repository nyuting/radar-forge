r"""Radar system description — transmitter, receiver, and the sited radar.

These are plain, frozen, validated value objects. They hold *what the radar is*;
they do not simulate anything. Signal generation lives in
:mod:`radar_forge.core.signal` and processing in :mod:`radar_forge.core.dsp`,
so that a scenario can be described, printed and checked before a single sample
is generated.

The properties are where the value lies: :attr:`Radar.unambiguous_range_m` and
:attr:`Radar.unambiguous_velocity_mps` turn a parameter set into the two numbers
that decide whether a scenario is observable, and they are the first thing to
look at when a target appears in the wrong place.

Two waveform families are supported, and they differ in how range is recovered,
which is why :class:`Transmitter` carries an explicit ``waveform`` tag:

``"fmcw"``
    A continuous sawtooth chirp, deramped on receive. Range appears as a beat
    frequency, so unambiguous range is set by the receiver sample rate against
    the sweep rate, not by the PRF.
``"pulsed"``
    A pulse train, matched-filtered on receive. Range appears as a delay, so
    unambiguous range is the classic :math:`c/2\,\mathrm{PRF}`.

Two sitings are supported. :class:`Radar` is **monostatic**: one site, and the
transmitter and receiver share it. :class:`BistaticRadar` puts them at two
different sites, which is what a passive or multistatic scenario needs.

Naming of the two ranges, used by this module, by
:mod:`radar_forge.core.radar_equation` and by :mod:`radar_forge.core.signal`:

``range_tx_m``
    The **transmit range** :math:`R_t`, from the transmitter site to the target.
``range_rx_m``
    The **receive range** :math:`R_r`, from the target to the receiver site.

Note what the ``tx``/``rx`` token qualifies: it names the *site the range is
measured to*, exactly as it names the antenna in ``gain_tx_linear`` and
``gain_rx_linear``. It is not a property of the transmitter itself. For a
monostatic radar the two are the same number and neither name is needed. These
two ranges are never called "legs" or "paths": a path in
:mod:`radar_forge.core.signal` is the whole transmitter-target-receiver route,
and a burst is a waveform configuration, so both terms are already taken.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §1.3 (PRF and ambiguity), §5.3 (pulse-Doppler).
.. [2] G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014,
       §2.5 (FMCW deramp and the beat-frequency range relation).
.. [3] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005, §1.3
       (geometry and the bistatic angle), §4.2 (range resolution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.core.constants import (
    BOLTZMANN_JPK,
    SPEED_OF_LIGHT_MPS,
    STANDARD_NOISE_TEMPERATURE_K,
)
from radar_forge.core.geodesy import geodetic_to_enu_m
from radar_forge.core.waveforms import sweep_rate_hzps

__all__ = ["BistaticRadar", "Radar", "RadarLike", "Receiver", "Transmitter", "WaveformKind"]

# Below this cosine of half the bistatic angle, the geometry is forward scatter.
# arccos loses half its significant digits near beta = pi, so an angle within
# roughly sqrt(eps) of pi carries no information distinguishing it from exactly
# pi; see BistaticRadar.range_resolution_at_bistatic_angle_m.
_FORWARD_SCATTER_COSINE_FLOOR = 1.0e-8

WaveformKind = Literal["fmcw", "pulsed"]
"""Which waveform family a transmitter emits; see the module docstring."""


@dataclass(frozen=True)
class Transmitter:
    """A radar transmitter.

    Parameters
    ----------
    f0_hz : float
        Carrier frequency, Hz. 9.8e9 for scenario 001.
    bandwidth_hz : float
        Swept or modulated bandwidth, Hz. Sets range resolution as
        :math:`c/2B`, independently of the waveform family.
    transmit_power_w : float
        Transmitted power, W. Peak power for a pulsed radar, average power for
        FMCW, which transmits continuously.
    gain_tx_dbi : float
        Transmit antenna gain, dBi. Decibels are correct here: this is an API
        boundary, and antenna gains are quoted in dBi everywhere.
    chirp_time_s : float
        Duration of one chirp (FMCW) or one pulse (pulsed), s.
    prf_hz : float
        Pulse or chirp repetition frequency, Hz. Its reciprocal is the interval
        between chirp starts.
    waveform : {"fmcw", "pulsed"}
        Waveform family. Default ``"fmcw"``.

    Raises
    ------
    ValueError
        If any quantity is non-positive, or if ``chirp_time_s * prf_hz`` exceeds
        one, which would mean a chirp had not finished before the next began.

    Notes
    -----
    A 100% duty cycle (``chirp_time_s * prf_hz == 1``) is permitted and is the
    normal case for FMCW; anything above it is not a waveform.
    """

    f0_hz: float
    bandwidth_hz: float
    transmit_power_w: float
    gain_tx_dbi: float
    chirp_time_s: float
    prf_hz: float
    waveform: WaveformKind = "fmcw"

    def __post_init__(self) -> None:
        """Validate the transmitter parameters; see the class ``Raises`` section."""
        positive = {
            "f0_hz": self.f0_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "transmit_power_w": self.transmit_power_w,
            "chirp_time_s": self.chirp_time_s,
            "prf_hz": self.prf_hz,
        }
        for name, value in positive.items():
            if value <= 0.0:
                msg = f"{name} must be strictly positive; got {value!r}."
                raise ValueError(msg)
        if self.waveform not in ("fmcw", "pulsed"):
            msg = f"waveform must be 'fmcw' or 'pulsed'; got {self.waveform!r}."
            raise ValueError(msg)
        if self.duty_cycle_linear > 1.0:
            msg = (
                f"chirp_time_s * prf_hz = {self.duty_cycle_linear:.4f} exceeds 1.0: the next "
                "chirp would start before this one finished."
            )
            raise ValueError(msg)

    @property
    def wavelength_m(self) -> float:
        r"""Carrier wavelength :math:`\lambda = c/f_0`, m."""
        return SPEED_OF_LIGHT_MPS / self.f0_hz

    @property
    def pulse_repetition_interval_s(self) -> float:
        """Time between chirp starts, s. The slow-time sample interval."""
        return 1.0 / self.prf_hz

    @property
    def duty_cycle_linear(self) -> float:
        """Fraction of time the transmitter is on, in [0, 1]."""
        return self.chirp_time_s * self.prf_hz

    @property
    def sweep_rate_hzps(self) -> float:
        r"""Chirp sweep rate :math:`\alpha = B/T`, Hz/s."""
        return float(sweep_rate_hzps(self.bandwidth_hz, self.chirp_time_s))


@dataclass(frozen=True)
class Receiver:
    """A radar receiver.

    Parameters
    ----------
    sample_rate_hz : float
        Complex-baseband sampling rate, Hz. For an FMCW receiver this is the
        rate at which the *deramped* beat signal is sampled and is usually far
        below the swept bandwidth; for a pulsed receiver it must cover the
        signal bandwidth.
    gain_rx_dbi : float
        Receive antenna gain, dBi.
    noise_figure_db : float
        Receiver noise figure, dB, referred to the 290 K IEEE reference
        temperature. Must be non-negative: a noise figure below 0 dB would mean
        the receiver improved the signal-to-noise ratio.

    Raises
    ------
    ValueError
        If ``sample_rate_hz`` is non-positive or ``noise_figure_db`` is negative.
    """

    sample_rate_hz: float
    gain_rx_dbi: float
    noise_figure_db: float

    def __post_init__(self) -> None:
        """Validate the receiver parameters; see the class ``Raises`` section."""
        if self.sample_rate_hz <= 0.0:
            msg = f"sample_rate_hz must be strictly positive; got {self.sample_rate_hz!r}."
            raise ValueError(msg)
        if self.noise_figure_db < 0.0:
            msg = (
                f"noise_figure_db must be non-negative; got {self.noise_figure_db!r}. "
                "A receiver cannot improve the signal-to-noise ratio."
            )
            raise ValueError(msg)

    @property
    def noise_figure_linear(self) -> float:
        r"""Noise figure as a linear factor :math:`F \ge 1`."""
        return float(10.0 ** (self.noise_figure_db / 10.0))


@dataclass(frozen=True)
class Radar:
    """A transmitter and receiver at a fixed geodetic site.

    Parameters
    ----------
    transmitter : Transmitter
        The transmit chain.
    receiver : Receiver
        The receive chain.
    latitude_deg, longitude_deg : float
        Antenna geodetic position, degrees.
    altitude_m : float
        Antenna height above the WGS-84 ellipsoid, metres.

    Raises
    ------
    ValueError
        If the site latitude is outside [-90, 90], or if a pulsed receiver
        samples below the transmitted bandwidth.

    Notes
    -----
    The radar is stationary. A moving platform would add a velocity to the
    Doppler of every path, including the clutter, and is out of scope for
    scenario 001.

    Examples
    --------
    >>> tx = Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3)
    >>> radar = Radar(tx, Receiver(1.0e6, 30.0, 3.0), 1.2915, 103.7871, 60.0)
    >>> round(radar.wavelength_m * 1e3, 3)  # millimetres
    30.591
    >>> round(radar.unambiguous_velocity_mps, 2)
    7.65
    """

    transmitter: Transmitter
    receiver: Receiver
    latitude_deg: float
    longitude_deg: float
    altitude_m: float

    def __post_init__(self) -> None:
        """Validate the site and the transmitter/receiver pairing."""
        if abs(self.latitude_deg) > 90.0:
            msg = f"latitude_deg must lie in [-90, 90]; got {self.latitude_deg!r}."
            raise ValueError(msg)
        if (
            self.transmitter.waveform == "pulsed"
            and self.receiver.sample_rate_hz < self.transmitter.bandwidth_hz
        ):
            msg = (
                f"sample_rate_hz {self.receiver.sample_rate_hz:.4g} Hz is below the transmitted "
                f"bandwidth {self.transmitter.bandwidth_hz:.4g} Hz; the pulse would alias. "
                "An FMCW receiver may sample below the bandwidth because it deramps first."
            )
            raise ValueError(msg)

    @property
    def wavelength_m(self) -> float:
        """Carrier wavelength, m."""
        return self.transmitter.wavelength_m

    @property
    def range_resolution_m(self) -> float:
        """Range resolution :math:`c/2B`, m. Set by bandwidth alone."""
        return SPEED_OF_LIGHT_MPS / (2.0 * self.transmitter.bandwidth_hz)

    @property
    def unambiguous_range_m(self) -> float:
        r"""Maximum range that is not folded, m.

        The two waveform families arrive at this differently [1]_, [2]_:

        .. math::

            R_{ua} = \frac{c}{2\,\mathrm{PRF}} \quad\text{(pulsed)},
            \qquad
            R_{ua} = \frac{c\, f_s}{4\alpha} \quad\text{(FMCW deramp)}

        For FMCW the limit is the highest beat frequency the receiver can
        represent without aliasing, :math:`f_s/2`, converted to range through the
        sweep rate. Note this makes FMCW unambiguous range a *receiver* property.
        """
        if self.transmitter.waveform == "pulsed":
            return SPEED_OF_LIGHT_MPS / (2.0 * self.transmitter.prf_hz)
        highest_beat_frequency_hz = self.receiver.sample_rate_hz / 2.0
        return (
            SPEED_OF_LIGHT_MPS
            * highest_beat_frequency_hz
            / (2.0 * self.transmitter.sweep_rate_hzps)
        )

    @property
    def unambiguous_velocity_mps(self) -> float:
        r"""Maximum unaliased radial speed, m/s, as a half-interval.

        Slow time is sampled once per chirp, so Doppler is unambiguous over
        :math:`\pm \mathrm{PRF}/2`, which is :math:`\pm \lambda\,\mathrm{PRF}/4`
        in velocity. Targets outside this fold back into it.
        """
        return self.wavelength_m * self.transmitter.prf_hz / 4.0

    @property
    def noise_power_w(self) -> float:
        r"""Receiver thermal noise power :math:`k T_0 B F`, W.

        The bandwidth used is the transmitted bandwidth, which is the noise
        bandwidth of a matched receiver.
        """
        return (
            BOLTZMANN_JPK
            * STANDARD_NOISE_TEMPERATURE_K
            * self.transmitter.bandwidth_hz
            * self.receiver.noise_figure_linear
        )

    @property
    def n_samples_per_chirp(self) -> int:
        """Fast-time samples spanning one chirp or pulse, from the receive rate.

        For FMCW at 100% duty this is also the receive window; for a pulsed
        radar it is only the transmit-on portion, and the receive window is
        :attr:`n_samples_per_pri`. Use that one to size an IQ cube.
        """
        return round(self.transmitter.chirp_time_s * self.receiver.sample_rate_hz)

    @property
    def n_samples_per_pri(self) -> int:
        """Fast-time samples in one full repetition interval — the IQ row length.

        This is the fast-time dimension of the ``(n_pulses, n_samples)`` cube
        that :mod:`radar_forge.core.signal` produces. It equals
        :attr:`n_samples_per_chirp` only at 100% duty.
        """
        return round(self.transmitter.pulse_repetition_interval_s * self.receiver.sample_rate_hz)


@dataclass(frozen=True)
class BistaticRadar:
    """A transmitter and receiver at two different geodetic sites.

    The bistatic counterpart of :class:`Radar`. Everything set by the waveform
    alone — wavelength, noise power, cube dimensions — carries over unchanged;
    what changes is that there are now two propagation ranges instead of one,
    and that several quantities a monostatic radar treats as constants become
    functions of where the target is. See the module docstring for the
    ``range_tx_m``/``range_rx_m`` naming.

    Parameters
    ----------
    transmitter : Transmitter
        The transmit chain, at the transmitter site.
    receiver : Receiver
        The receive chain, at the receiver site.
    transmitter_latitude_deg, transmitter_longitude_deg : float
        Transmit antenna geodetic position, degrees.
    transmitter_altitude_m : float
        Transmit antenna height above the WGS-84 ellipsoid, metres.
    receiver_latitude_deg, receiver_longitude_deg : float
        Receive antenna geodetic position, degrees.
    receiver_altitude_m : float
        Receive antenna height above the WGS-84 ellipsoid, metres.

    Raises
    ------
    ValueError
        If either site latitude is outside [-90, 90], if the two sites coincide,
        or if a pulsed receiver samples below the transmitted bandwidth.

    Notes
    -----
    Both sites are stationary, as for :class:`Radar`.

    A zero baseline is rejected rather than silently allowed: it is a monostatic
    radar described the hard way, :class:`Radar` models it directly, and the
    bistatic angle is undefined there.

    Two idealisations worth knowing before trusting a number out of this class:
    the transmitter and receiver are assumed **perfectly synchronised** in time
    and phase, where a real pair has an oscillator offset that appears as a range
    bias and a Doppler ramp; and the **direct signal from transmitter to
    receiver is not modelled**, though in practice that baseline breakthrough
    dominates a bistatic receiver's dynamic range.

    Examples
    --------
    >>> tx = Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3)
    >>> pair = BistaticRadar(
    ...     tx, Receiver(1.0e6, 30.0, 3.0), 1.2915, 103.7871, 60.0, 1.2915, 103.9871, 60.0
    ... )
    >>> round(pair.baseline_m / 1e3, 2)  # kilometres between the two sites
    22.26
    >>> round(pair.wavelength_m * 1e3, 3)  # millimetres, as for a monostatic radar
    30.591
    """

    transmitter: Transmitter
    receiver: Receiver
    transmitter_latitude_deg: float
    transmitter_longitude_deg: float
    transmitter_altitude_m: float
    receiver_latitude_deg: float
    receiver_longitude_deg: float
    receiver_altitude_m: float

    def __post_init__(self) -> None:
        """Validate both sites and the transmitter/receiver pairing."""
        latitudes = {
            "transmitter_latitude_deg": self.transmitter_latitude_deg,
            "receiver_latitude_deg": self.receiver_latitude_deg,
        }
        for name, value in latitudes.items():
            if abs(value) > 90.0:
                msg = f"{name} must lie in [-90, 90]; got {value!r}."
                raise ValueError(msg)
        if (
            self.transmitter.waveform == "pulsed"
            and self.receiver.sample_rate_hz < self.transmitter.bandwidth_hz
        ):
            msg = (
                f"sample_rate_hz {self.receiver.sample_rate_hz:.4g} Hz is below the transmitted "
                f"bandwidth {self.transmitter.bandwidth_hz:.4g} Hz; the pulse would alias. "
                "An FMCW receiver may sample below the bandwidth because it deramps first."
            )
            raise ValueError(msg)
        if self.baseline_m <= 0.0:
            msg = (
                "the transmitter and receiver sites coincide, so the baseline is zero and the "
                "bistatic angle is undefined. That is a monostatic radar; use Radar instead."
            )
            raise ValueError(msg)

    @property
    def receiver_enu_m(self) -> NDArray[np.float64]:
        """Receiver site as local east, north, up metres about the transmitter site."""
        return geodetic_to_enu_m(
            self.receiver_latitude_deg,
            self.receiver_longitude_deg,
            self.receiver_altitude_m,
            self.transmitter_latitude_deg,
            self.transmitter_longitude_deg,
            self.transmitter_altitude_m,
        )

    @property
    def baseline_m(self) -> float:
        r"""Distance :math:`L` between the two sites, m.

        The bistatic geometry's one fixed length. Every other distance in the
        triangle moves with the target; this one does not.
        """
        return float(np.linalg.norm(self.receiver_enu_m))

    @property
    def wavelength_m(self) -> float:
        """Carrier wavelength, m."""
        return self.transmitter.wavelength_m

    @property
    def range_resolution_m(self) -> float:
        r"""Best-case range resolution :math:`c/2B`, m.

        This is the value at zero bistatic angle only, where the geometry is
        effectively monostatic. It is the *floor*: resolution degrades
        everywhere else, and
        :meth:`range_resolution_at_bistatic_angle_m` gives the value that
        actually applies to a target.
        """
        return SPEED_OF_LIGHT_MPS / (2.0 * self.transmitter.bandwidth_hz)

    @property
    def unambiguous_range_m(self) -> float:
        r"""Maximum **range sum** :math:`R_t + R_r` that is not folded, m.

        The same two waveform rules as :attr:`Radar.unambiguous_range_m` [1]_,
        [2]_, but the quantity they bound is the sum of the two ranges rather
        than a single range, because a bistatic echo is delayed by
        :math:`(R_t + R_r)/c` and not by :math:`2R/c`:

        .. math::

            (R_t + R_r)_{ua} = \frac{c}{\mathrm{PRF}} \quad\text{(pulsed)},
            \qquad
            (R_t + R_r)_{ua} = \frac{c\, f_s}{2\alpha} \quad\text{(FMCW deramp)}

        These are twice the monostatic numbers, which is not a bistatic bonus:
        the sum they bound is itself roughly twice a one-way range, so a target
        folds at about the same distance from the pair.
        """
        if self.transmitter.waveform == "pulsed":
            return SPEED_OF_LIGHT_MPS / self.transmitter.prf_hz
        highest_beat_frequency_hz = self.receiver.sample_rate_hz / 2.0
        return SPEED_OF_LIGHT_MPS * highest_beat_frequency_hz / self.transmitter.sweep_rate_hzps

    @property
    def unambiguous_velocity_mps(self) -> float:
        r"""Maximum unaliased **bisector** range rate, m/s, as a half-interval.

        Identical in form to :attr:`Radar.unambiguous_velocity_mps`, because slow
        time is sampled once per pulse either way. What folds is the bisector
        range rate of :func:`radar_forge.core.signal.bistatic_doppler_hz`, not a
        radial velocity towards either site.
        """
        return self.wavelength_m * self.transmitter.prf_hz / 4.0

    @property
    def noise_power_w(self) -> float:
        r"""Receiver thermal noise power :math:`k T_0 B F`, W.

        A property of the receive chain alone, so it is unchanged from
        :attr:`Radar.noise_power_w`.
        """
        return (
            BOLTZMANN_JPK
            * STANDARD_NOISE_TEMPERATURE_K
            * self.transmitter.bandwidth_hz
            * self.receiver.noise_figure_linear
        )

    @property
    def n_samples_per_chirp(self) -> int:
        """Fast-time samples spanning one chirp or pulse, from the receive rate."""
        return round(self.transmitter.chirp_time_s * self.receiver.sample_rate_hz)

    @property
    def n_samples_per_pri(self) -> int:
        """Fast-time samples in one full repetition interval — the IQ row length."""
        return round(self.transmitter.pulse_repetition_interval_s * self.receiver.sample_rate_hz)

    def bistatic_angle_rad(
        self, range_tx_m: ArrayLike, range_rx_m: ArrayLike
    ) -> NDArray[np.float64]:
        r"""Return the bistatic angle :math:`\beta` at the target, radians.

        The angle subtended at the target by the two sites [3]_, from the law of
        cosines on the triangle whose third side is the baseline:

        .. math::

            \cos\beta = \frac{R_t^2 + R_r^2 - L^2}{2 R_t R_r}

        It is zero when the sites are effectively collocated as seen from the
        target, and :math:`\pi` when the target lies on the baseline between
        them.

        Parameters
        ----------
        range_tx_m, range_rx_m : array_like
            Transmit and receive ranges, metres, broadcast against one another.
            Both must be strictly positive.

        Returns
        -------
        numpy.ndarray
            Bistatic angle in radians, in :math:`[0, \pi]`, of the broadcast
            shape of the inputs.

        Raises
        ------
        ValueError
            If any range is non-positive, or if the three lengths cannot form a
            triangle — a target cannot be closer to both sites than they are to
            each other.

        Notes
        -----
        The cosine is clipped to :math:`[-1, 1]` before the arc cosine. That is
        not defensive decoration: for a target exactly on the baseline the exact
        value is :math:`-1`, floating-point rounding lands a little past it, and
        an unclipped :func:`numpy.arccos` would return NaN for the one geometry
        most worth asking about.
        """
        range_tx_arr = np.asarray(range_tx_m, dtype=np.float64)
        range_rx_arr = np.asarray(range_rx_m, dtype=np.float64)
        if np.any(range_tx_arr <= 0.0) or np.any(range_rx_arr <= 0.0):
            msg = (
                "range_tx_m and range_rx_m must be strictly positive; the bistatic angle is "
                "undefined for a target standing at either site."
            )
            raise ValueError(msg)

        baseline_m = self.baseline_m
        if np.any(range_tx_arr + range_rx_arr < baseline_m):
            msg = (
                f"range_tx_m + range_rx_m is less than the {baseline_m:.4g} m baseline, which is "
                "not a triangle: no target can be nearer to both sites than they are to each "
                "other. Check that the ranges belong to this pair of sites."
            )
            raise ValueError(msg)

        cosine = (range_tx_arr**2 + range_rx_arr**2 - baseline_m**2) / (
            2.0 * range_tx_arr * range_rx_arr
        )
        result: NDArray[np.float64] = np.arccos(np.clip(cosine, -1.0, 1.0))
        return result

    def range_resolution_at_bistatic_angle_m(
        self, bistatic_angle_rad: ArrayLike
    ) -> NDArray[np.float64]:
        r"""Return the range resolution at a given bistatic angle, m.

        .. math::

            \Delta R = \frac{c}{2 B \cos(\beta/2)}

        Resolution is set by bandwidth alone only in the monostatic case [3]_.
        Here it also depends on where the target stands: the iso-range surfaces
        are ellipsoids with the two sites at the foci, and they crowd together
        near the sites and spread out near the baseline.

        Parameters
        ----------
        bistatic_angle_rad : array_like
            Bistatic angle, radians, from :meth:`bistatic_angle_rad`. Must lie in
            :math:`[0, \pi]`.

        Returns
        -------
        numpy.ndarray
            Range resolution in metres, of the shape of the input. Infinite at
            :math:`\beta = \pi`.

        Raises
        ------
        ValueError
            If any angle is outside :math:`[0, \pi]`.

        Notes
        -----
        At :math:`\beta = \pi` the target sits on the baseline, every path from
        transmitter to target to receiver has the same total length, and the
        resolution is infinite — there is no range information at all in forward
        scatter. That is returned as ``inf`` rather than raised, because it is a
        real limit of a real geometry and a caller plotting resolution across a
        scene should see it rather than crash at one point.

        Angles within about :math:`\\sqrt{\\epsilon}` of :math:`\\pi` return
        ``inf`` too. :meth:`bistatic_angle_rad` cannot resolve them from
        :math:`\\pi` — the arc cosine loses half its digits there — so reporting
        a large finite resolution would be claiming precision that the angle
        behind it does not have.
        """
        angle_arr = np.asarray(bistatic_angle_rad, dtype=np.float64)
        if np.any(angle_arr < 0.0) or np.any(angle_arr > np.pi):
            msg = (
                "bistatic_angle_rad must lie in [0, pi] radians; got values outside it. "
                "The angle is subtended at the target, so it cannot exceed a straight line."
            )
            raise ValueError(msg)

        # In exact arithmetic cos(beta/2) vanishes at forward scatter, but
        # np.cos(np.pi / 2) is 6.1e-17 rather than 0, which would turn a genuine
        # singularity into a finite 1e18 m that looks like a measurement. Snap
        # the indistinguishable-from-pi case to the true limit instead.
        cosine_half = np.cos(angle_arr / 2.0)
        cosine_half = np.where(cosine_half < _FORWARD_SCATTER_COSINE_FLOOR, 0.0, cosine_half)
        with np.errstate(divide="ignore"):
            result: NDArray[np.float64] = self.range_resolution_m / cosine_half
        return result

    def target_ranges_m(
        self,
        latitude_deg: ArrayLike,
        longitude_deg: ArrayLike,
        altitude_m: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the transmit and receive ranges to geodetic target positions.

        The bridge from a scenario's world coordinates to the two ranges every
        other bistatic function wants.

        Parameters
        ----------
        latitude_deg, longitude_deg, altitude_m : array_like
            Target geodetic positions, degrees and metres, broadcast against one
            another. Shape ``(n_targets,)`` for a set of targets.

        Returns
        -------
        range_tx_m : numpy.ndarray
            Transmit range to each target, metres.
        range_rx_m : numpy.ndarray
            Receive range to each target, metres.

        Notes
        -----
        A target standing exactly on one of the sites gives zero for that range.
        That is the honest answer geometrically, but it is not a usable input to
        the range equation or to :meth:`bistatic_angle_rad`, both of which
        require strictly positive ranges and will say so.

        See Also
        --------
        radar_forge.core.geodesy.geodetic_to_enu_m : The conversion used for each
            site in turn.
        """
        # Only the magnitudes are wanted, so take the norms rather than going
        # through enu_to_range_azimuth_elevation: its arcsin(up / range) is 0/0
        # for a target standing on a site, which is a legitimate thing to ask
        # about here and would come back NaN with a warning.
        range_tx_m = np.linalg.norm(
            geodetic_to_enu_m(
                latitude_deg,
                longitude_deg,
                altitude_m,
                self.transmitter_latitude_deg,
                self.transmitter_longitude_deg,
                self.transmitter_altitude_m,
            ),
            axis=-1,
        )
        range_rx_m = np.linalg.norm(
            geodetic_to_enu_m(
                latitude_deg,
                longitude_deg,
                altitude_m,
                self.receiver_latitude_deg,
                self.receiver_longitude_deg,
                self.receiver_altitude_m,
            ),
            axis=-1,
        )
        return range_tx_m, range_rx_m


RadarLike = Radar | BistaticRadar
"""Either siting of a radar.

The signal generators in :mod:`radar_forge.core.signal` depend only on the
waveform and receiver attributes that :class:`Radar` and :class:`BistaticRadar`
both carry, so they accept either.
"""
