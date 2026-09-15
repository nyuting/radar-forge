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

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §1.3 (PRF and ambiguity), §5.3 (pulse-Doppler).
.. [2] G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014,
       §2.5 (FMCW deramp and the beat-frequency range relation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from radar_forge.core.constants import (
    BOLTZMANN_JPK,
    SPEED_OF_LIGHT_MPS,
    STANDARD_NOISE_TEMPERATURE_K,
)
from radar_forge.core.waveforms import sweep_rate_hzps

__all__ = ["Radar", "Receiver", "Transmitter", "WaveformKind"]

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
