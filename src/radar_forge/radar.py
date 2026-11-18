r"""The radar: transmitter, receiver, sites, and the route geometry of a target.

One :class:`Radar` covers monostatic and bistatic: ``rx_site=None`` is the zero-baseline,
monostatic case. Sites are stationary and synchronised, with no direct signal. A
target's route has transmit and receive ranges :math:`R_T`, :math:`R_R`, the
bistatic range :math:`R_b = R_T + R_R` (monostatic :math:`2R`) and its rate
:math:`dR_b/dt`, positive receding. Positions are WGS-84 local east, north, up
metres; azimuth is zero north, clockwise (``spec/data-001-formats.md`` §5). The conversions
between geodetic, ENU and range-azimuth-elevation live in :mod:`radar_forge.coordinates`.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014.
.. [2] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005.
.. [3] National Imagery and Mapping Agency, *Department of Defense World Geodetic
       System 1984*, NIMA TR8350.2, 3rd ed., Amendment 1, 2000.
.. [4] P. Misra and P. Enge, *Global Positioning System: Signals, Measurements,
       and Performance*, 2nd ed., Ganga-Jamuna Press, 2006.
.. [5] M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill, 2001.
.. [6] G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014.
.. [7] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., 2010.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.constants import (
    BOLTZMANN_JPK,
    SPEED_OF_LIGHT_MPS,
    STANDARD_NOISE_TEMPERATURE_K,
)
from radar_forge.coordinates import (
    _enu_axes_in_ecef,
    enu_to_range_azimuth_elevation_rad,
    geodetic_to_enu_m,
)

__all__ = [
    "Radar",
    "Receiver",
    "Site",
    "TargetGeometry",
    "Transmitter",
    "WaveformKind",
]

WaveformKind = Literal["fmcw", "pulsed", "noise"]
"""Which waveform family a transmitter emits: deramped FMCW, a pulse train or a noise code."""


@dataclass(frozen=True)
class Transmitter:
    """A radar transmitter; when pulsed, its power is peak power and its chirp is the pulse.

    A continuous noise code has no chirp: ``chirp_duration_s`` is its correlation batch, the
    slow-time interval, and ``prf_hz`` the batch rate, so the duty cycle is one.
    """

    carrier_frequency_hz: float
    bandwidth_hz: float
    transmit_power_w: float
    gain_tx_dbi: float
    chirp_duration_s: float
    prf_hz: float
    waveform: WaveformKind = "fmcw"

    def __post_init__(self) -> None:
        """Reject a non-positive quantity, an unknown waveform, or a duty cycle above one."""
        positive = ("carrier_frequency_hz", "bandwidth_hz", "transmit_power_w", "chirp_duration_s")
        for name in (*positive, "prf_hz"):
            if getattr(self, name) <= 0.0:
                msg = f"{name} must be strictly positive; got {getattr(self, name)!r}."
                raise ValueError(msg)
        if self.waveform not in ("fmcw", "pulsed", "noise"):
            msg = f"waveform must be 'noise', 'fmcw' or 'pulsed'; got {self.waveform!r}."
            raise ValueError(msg)
        if self.duty_cycle_linear > 1.0:
            msg = f"chirp_duration_s * prf_hz = {self.duty_cycle_linear:.4f} exceeds 1.0."
            raise ValueError(msg)

    @property
    def wavelength_m(self) -> float:
        r"""Carrier wavelength :math:`\lambda = c/f_c`, m."""
        return SPEED_OF_LIGHT_MPS / self.carrier_frequency_hz

    @property
    def pulse_repetition_interval_s(self) -> float:
        """Time between chirp starts, s: the slow-time sample interval."""
        return 1.0 / self.prf_hz

    @property
    def duty_cycle_linear(self) -> float:
        """Fraction of time the transmitter is on, in [0, 1]."""
        return self.chirp_duration_s * self.prf_hz

    @property
    def sweep_rate_hzps(self) -> float:
        r"""Chirp slope :math:`\alpha = B/T`, Hz/s."""
        return self.bandwidth_hz / self.chirp_duration_s


@dataclass(frozen=True)
class Receiver:
    """A complex-baseband radar receiver; FMCW samples the deramped beat, NF is at 290 K."""

    sample_rate_hz: float
    gain_rx_dbi: float
    noise_figure_db: float

    def __post_init__(self) -> None:
        """Reject a non-positive sample rate or a negative noise figure."""
        if self.sample_rate_hz <= 0.0:
            msg = f"sample_rate_hz must be strictly positive; got {self.sample_rate_hz!r}."
            raise ValueError(msg)
        if self.noise_figure_db < 0.0:
            msg = "noise_figure_db must be non-negative; a receiver cannot improve the SNR."
            raise ValueError(msg)

    @property
    def noise_figure_linear(self) -> float:
        r"""Noise figure as a linear factor :math:`F \ge 1`."""
        return float(10.0 ** (self.noise_figure_db / 10.0))

    @property
    def noise_density_w_per_hz(self) -> float:
        r"""Thermal noise density :math:`N_0 = k T_0 F`, W/Hz (Skolnik [5], §2.3)."""
        return BOLTZMANN_JPK * STANDARD_NOISE_TEMPERATURE_K * self.noise_figure_linear

    @property
    def noise_power_per_sample_w(self) -> float:
        r"""Noise in one complex sample, :math:`N_0 f_s`, W: a sampler spans :math:`f_s` of band."""
        return self.noise_density_w_per_hz * self.sample_rate_hz


@dataclass(frozen=True)
class Site:
    """A fixed antenna position; ``altitude_m`` is above the WGS-84 ellipsoid, not sea level.

    The ellipsoid is NIMA [3]'s; :mod:`radar_forge.coordinates` does the conversions.
    """

    latitude_deg: float
    longitude_deg: float
    altitude_m: float

    def __post_init__(self) -> None:
        """Reject a latitude outside [-90, 90] degrees."""
        if abs(self.latitude_deg) > 90.0:
            msg = f"latitude_deg must lie in [-90, 90]; got {self.latitude_deg!r}."
            raise ValueError(msg)

    def enu_m(self, origin: Site) -> NDArray[np.float64]:
        """Return this site in ``origin``'s east, north, up frame, ``(3,)``, m."""
        site = (self.latitude_deg, self.longitude_deg, self.altitude_m)
        return geodetic_to_enu_m(
            *site, origin.latitude_deg, origin.longitude_deg, origin.altitude_m
        )


@dataclass(frozen=True)
class TargetGeometry:
    r"""Each target's route, transmitter to target to receiver, as ``(n_targets,)`` arrays.

    ``range_tx_m`` runs from the transmitter site and ``range_rx_m`` to the
    receiver site; the rate is positive receding; :math:`\beta` is the angle at
    the target, 0 monostatic and :math:`\pi` on the baseline; ``aod_rad`` and
    ``aoa_rad`` are ``(n_targets, 2)`` (azimuth, elevation), each in its own
    site's frame (Willis [2] §1.2). ``bistatic_range_acceleration_mps2`` is
    :math:`\ddot R_b`, positive when the rate grows; it defaults to zero.
    """

    range_tx_m: NDArray[np.float64]
    range_rx_m: NDArray[np.float64]
    bistatic_range_m: NDArray[np.float64]
    bistatic_range_rate_mps: NDArray[np.float64]
    bistatic_angle_rad: NDArray[np.float64]
    aod_rad: NDArray[np.float64]
    aoa_rad: NDArray[np.float64]
    bistatic_range_acceleration_mps2: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(())
    )

    def __post_init__(self) -> None:
        """Reject fields that disagree on the number of targets; store one acceleration each."""
        n_targets = np.size(self.range_tx_m)
        if np.shape(self.bistatic_range_acceleration_mps2) == ():
            acceleration_mps2 = np.zeros(n_targets) + self.bistatic_range_acceleration_mps2
            object.__setattr__(self, "bistatic_range_acceleration_mps2", acceleration_mps2)
        for name, value in vars(self).items():
            expected = (n_targets, 2) if name in ("aod_rad", "aoa_rad") else (n_targets,)
            if np.shape(value) != expected:
                msg = f"{name} must have shape {expected}, as range_tx_m; got {np.shape(value)}."
                raise ValueError(msg)

    @classmethod
    def monostatic(
        cls,
        range_m: ArrayLike,
        radial_velocity_mps: ArrayLike,
        *,
        azimuth_rad: ArrayLike = 0.0,
        elevation_rad: ArrayLike = 0.0,
        radial_acceleration_mps2: ArrayLike = 0.0,
    ) -> TargetGeometry:
        """Return one site's geometry: twice the range and its derivatives (receding), beta 0."""
        range_arr, rate_arr, azimuth_arr, elevation_arr, acceleration_arr = np.broadcast_arrays(
            np.atleast_1d(np.asarray(range_m, dtype=np.float64)),
            np.asarray(radial_velocity_mps, dtype=np.float64),
            np.asarray(azimuth_rad, dtype=np.float64),
            np.asarray(elevation_rad, dtype=np.float64),
            np.asarray(radial_acceleration_mps2, dtype=np.float64),
        )
        angles_rad = np.stack([azimuth_arr, elevation_arr], axis=-1)
        return cls(
            range_tx_m=range_arr.copy(),
            range_rx_m=range_arr.copy(),
            bistatic_range_m=2.0 * range_arr,
            bistatic_range_rate_mps=2.0 * rate_arr,
            bistatic_angle_rad=np.zeros(range_arr.shape),
            aod_rad=angles_rad,
            aoa_rad=angles_rad.copy(),
            bistatic_range_acceleration_mps2=2.0 * acceleration_arr,
        )


@dataclass(frozen=True)
class Radar:
    """A transmitter and a receiver; ``rx_site=None`` puts both at ``tx_site``, monostatic.

    Examples
    --------
    >>> tx = Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3)
    >>> radar = Radar(tx, Receiver(1.0e6, 30.0, 3.0), Site(36.0025, -78.941, 60.0))
    >>> radar.is_monostatic, radar.baseline_m, round(radar.unambiguous_bistatic_range_m, 1)
    (True, 0.0, 149896.2)
    """

    transmitter: Transmitter
    receiver: Receiver
    tx_site: Site
    rx_site: Site | None = None

    def __post_init__(self) -> None:
        """Reject a pulsed receiver sampling below the bandwidth (FMCW may: it deramps).

        A noise code is white across the sampled band, so its bandwidth is the sample rate, and
        continuous, so its batch is its PRI.
        """
        waveform, sample_rate_hz = self.transmitter.waveform, self.receiver.sample_rate_hz
        if waveform == "pulsed" and sample_rate_hz < self.transmitter.bandwidth_hz:
            msg = "a pulsed receiver's sample_rate_hz is below bandwidth_hz; the pulse would alias."
            raise ValueError(msg)
        if waveform == "noise" and sample_rate_hz != self.transmitter.bandwidth_hz:
            msg = "a noise code fills the sampled band: bandwidth_hz must equal sample_rate_hz."
            raise ValueError(msg)
        if waveform == "noise" and self.n_samples_per_pri != self.n_samples_per_chirp:
            msg = (
                "a noise code is continuous: its correlation batch fills its PRI, "
                "chirp_duration_s * prf_hz = 1."
            )
            raise ValueError(msg)

    @property
    def _receive_site(self) -> Site:
        return self.tx_site if self.rx_site is None else self.rx_site

    @property
    def is_monostatic(self) -> bool:
        """Whether transmitter and receiver share one site."""
        return self.rx_site is None or self.rx_site == self.tx_site

    @property
    def baseline_m(self) -> float:
        """Distance between the sites, m; 0.0 when monostatic."""
        return float(np.linalg.norm(self._receive_site.enu_m(self.tx_site)))

    @property
    def wavelength_m(self) -> float:
        """Carrier wavelength, m."""
        return self.transmitter.wavelength_m

    @property
    def range_resolution_m(self) -> float:
        r"""Range resolution :math:`c/2B`, m: monostatic, and the bistatic best case."""
        return SPEED_OF_LIGHT_MPS / (2.0 * self.transmitter.bandwidth_hz)

    @property
    def unambiguous_delay_s(self) -> float:
        r"""Delay period the map wraps at: one PRI; FMCW :math:`f_s/\alpha` (Oppenheim [7]_ §8.5).

        Pulsed, the map holds ``round(PRI f_s)`` samples, equal whenever the PRI is a whole number
        of samples (every config is). FMCW, it mirrors :func:`~radar_forge.dsp.range_bin_centers_m`
        and can exceed the chirp (S3: :math:`2T_c` at :math:`f_s = 2B`), though a sawtooth deramp
        caps the physical reach at :math:`T_c` (Charvat [6]_, ch. 3). A noise code folds nothing,
        so it is infinite: batched correlation reads each lag whole, past its own batch, and the
        map is a window of lags, :math:`c\,m/f_s` for ``m < n_range_lags``
        (:func:`~radar_forge.pipeline.map_axes`), which an echo beyond it misses.
        """
        if self.transmitter.waveform == "noise":
            return math.inf
        if self.transmitter.waveform == "pulsed":
            return self.transmitter.pulse_repetition_interval_s
        return self.receiver.sample_rate_hz / self.transmitter.sweep_rate_hzps

    @property
    def unambiguous_bistatic_range_m(self) -> float:
        r"""Span of the map's range axis, :math:`c\,\tau_{ua}`, m; half of it monostatic."""
        return SPEED_OF_LIGHT_MPS * self.unambiguous_delay_s

    @property
    def unambiguous_bistatic_range_rate_mps(self) -> float:
        r"""Rate span before folding, :math:`\lambda\,\mathrm{PRF}`, m/s (Richards [1] §3.1.3)."""
        return self.wavelength_m * self.transmitter.prf_hz

    @property
    def noise_power_matched_w(self) -> float:
        r"""Noise in the transmitted band, :math:`k T_0 B F`, W: the radar-equation term."""
        return self.receiver.noise_density_w_per_hz * self.transmitter.bandwidth_hz

    @property
    def n_samples_per_chirp(self) -> int:
        """Fast-time samples spanning one chirp or pulse; a noise code's correlation batch."""
        return round(self.transmitter.chirp_duration_s * self.receiver.sample_rate_hz)

    @property
    def n_samples_per_pri(self) -> int:
        """Fast-time samples in one repetition interval: a cube's row length."""
        return round(self.transmitter.pulse_repetition_interval_s * self.receiver.sample_rate_hz)

    def sites_enu_m(self, origin: Site | None = None) -> NDArray[np.float64]:
        """Return the transmitter and receiver sites in ``origin``'s frame (default: receiver's)."""
        origin_site = self._receive_site if origin is None else origin
        return np.stack([self.tx_site.enu_m(origin_site), self._receive_site.enu_m(origin_site)])

    def geometry(
        self,
        position_enu_m: ArrayLike,
        velocity_enu_mps: ArrayLike,
        *,
        acceleration_enu_mps2: ArrayLike | None = None,
        origin: Site | None = None,
    ) -> TargetGeometry:
        r"""Return the route geometry of targets at given positions, velocities, accelerations.

        Each is ``(n_targets, 3)`` in ``origin``'s ENU frame, by default the
        receiver's. The rate is :math:`\vec v\cdot(\hat u_T + \hat u_R)`; the law of
        cosines gives :math:`\beta`, clipped to be exactly 0 at zero baseline and
        :math:`\pi`, not NaN, on it. The departure vector is rotated into the
        transmitter's own axes, so each angle is in its site's frame (Willis [2] ch. 3).
        A leg :math:`R = |\vec p - \vec s|` has :math:`\dot R = \hat u\cdot\vec v`, and
        :math:`\dot{\hat u} = (\vec v - \hat u\dot R)/R` gives its acceleration

        .. math::

            \ddot R = \hat u\cdot\vec a + \left(|\vec v|^2 - \dot R^2\right)/R,

        the second term the velocity across the line of sight turning it;
        :math:`\ddot R_b` sums both legs. Without ``acceleration_enu_mps2`` it is
        zero, not this with :math:`\vec a = 0`: the paths then keep a constant rate.

        Examples
        --------
        >>> tx = Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3)
        >>> radar = Radar(tx, Receiver(1.0e6, 30.0, 3.0), Site(36.0025, -78.941, 60.0))
        >>> geometry = radar.geometry([0.0, 3000.0, 4000.0], [0.0, 30.0, 40.0])  # receding
        >>> float(geometry.bistatic_range_m[0]), float(geometry.bistatic_range_rate_mps[0])
        (10000.0, 100.0)
        """
        origin_site = self._receive_site if origin is None else origin
        position_m = np.atleast_2d(np.asarray(position_enu_m, dtype=np.float64))
        velocity_mps = np.asarray(velocity_enu_mps, dtype=np.float64)
        tx_enu_m, rx_enu_m = self.sites_enu_m(origin_site)
        offset_tx_m, offset_rx_m = position_m - tx_enu_m, position_m - rx_enu_m
        range_tx_m, range_rx_m = _slant_range_m(offset_tx_m), _slant_range_m(offset_rx_m)
        if not (np.all(range_tx_m > 0.0) and np.all(range_rx_m > 0.0)):
            msg = "every target must be off both sites: a direction from a site needs range > 0."
            raise ValueError(msg)
        unit_tx, aod_rad = _look(offset_tx_m, range_tx_m, self.tx_site, origin_site)
        unit_rx, aoa_rad = _look(offset_rx_m, range_rx_m, self._receive_site, origin_site)
        baseline_m = np.sqrt(np.sum((rx_enu_m - tx_enu_m) ** 2))
        cosine = (range_tx_m**2 + range_rx_m**2 - baseline_m**2) / (2.0 * range_tx_m * range_rx_m)
        rate_tx_mps = np.sum(velocity_mps * unit_tx, axis=-1)
        rate_rx_mps = np.sum(velocity_mps * unit_rx, axis=-1)
        acceleration_mps2: ArrayLike = 0.0
        if acceleration_enu_mps2 is not None:
            accel_mps2 = np.asarray(acceleration_enu_mps2, dtype=np.float64)
            speed_squared = np.sum(velocity_mps**2, axis=-1)
            acceleration_mps2 = (
                np.sum(accel_mps2 * unit_tx, axis=-1)
                + (speed_squared - rate_tx_mps**2) / range_tx_m
                + np.sum(accel_mps2 * unit_rx, axis=-1)
                + (speed_squared - rate_rx_mps**2) / range_rx_m
            )
        return TargetGeometry(
            range_tx_m=range_tx_m,
            range_rx_m=range_rx_m,
            bistatic_range_m=range_tx_m + range_rx_m,
            bistatic_range_rate_mps=rate_tx_mps + rate_rx_mps,
            bistatic_angle_rad=np.arccos(np.clip(cosine, -1.0, 1.0)),
            aod_rad=aod_rad,
            aoa_rad=aoa_rad,
            bistatic_range_acceleration_mps2=np.asarray(acceleration_mps2, dtype=np.float64),
        )


def _slant_range_m(offset_enu_m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the length of each ENU vector, as ``enu_to_range_azimuth_elevation_rad`` does."""
    result: NDArray[np.float64] = np.sqrt(
        offset_enu_m[..., 0] ** 2 + offset_enu_m[..., 1] ** 2 + offset_enu_m[..., 2] ** 2
    )
    return result


def _look(
    offset_enu_m: NDArray[np.float64], range_m: NDArray[np.float64], site: Site, origin: Site
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the unit vector, and (azimuth, elevation) in ``site``'s own frame (Misra [4] 4.A)."""
    site_axes_m = offset_enu_m
    if site != origin:
        origin_axes = _enu_axes_in_ecef(origin.latitude_deg, origin.longitude_deg)
        site_axes = _enu_axes_in_ecef(site.latitude_deg, site.longitude_deg)
        site_axes_m = offset_enu_m @ (origin_axes @ site_axes.T)
    _, azimuth_rad, elevation_rad = enu_to_range_azimuth_elevation_rad(site_axes_m)
    unit_vector = offset_enu_m / range_m[..., None]
    return unit_vector, np.stack([azimuth_rad, elevation_rad], axis=-1)
