r"""The radar: transmitter, receiver, sites, and the route geometry of a target.

One :class:`Radar` covers both sitings: ``rx_site=None`` is monostatic, the
zero-baseline case of the bistatic radar, not a separate class. A target is
described by its route, transmitter to target to receiver [2]_: the transmit
and receive ranges :math:`R_T` and :math:`R_R`, the bistatic range
:math:`R_b = R_T + R_R` (monostatic, :math:`2R`), and its rate
:math:`\dot R_b = dR_b/dt`, **positive receding**.

Positions are local east, north, up (ENU) metres about a WGS-84 site [3]_, [4]_.
Azimuth is zero at true north and increases clockwise; elevation is up from
the local horizontal (``spec/data-001-formats.md`` DF2). There is no refraction.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §3.1.3 (range and Doppler ambiguities).
.. [2] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005, §1.2
       (definitions and the bistatic angle), ch. 3 (geometry).
.. [3] National Imagery and Mapping Agency, *Department of Defense World
       Geodetic System 1984*, NIMA TR8350.2, 3rd ed., Amendment 1, 2000, §4.3,
       eqs. (4-14)-(4-15).
.. [4] P. Misra and P. Enge, *Global Positioning System: Signals, Measurements,
       and Performance*, 2nd ed., Ganga-Jamuna Press, 2006, app. 4.A.
.. [5] M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill,
       2001, §2.3 (receiver noise).
.. [6] G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014,
       ch. 3 (FMCW deramp).
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
    WGS84_FLATTENING,
    WGS84_SEMI_MAJOR_AXIS_M,
)

__all__ = [
    "Radar",
    "Receiver",
    "Site",
    "TargetGeometry",
    "Transmitter",
    "WaveformKind",
    "ecef_to_enu_m",
    "enu_to_range_azimuth_elevation_rad",
    "geodetic_to_ecef_m",
    "geodetic_to_enu_m",
    "range_azimuth_elevation_to_enu_m",
]

# First eccentricity squared, e^2 = f(2 - f), derived rather than stored.
_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)

WaveformKind = Literal["fmcw", "pulsed"]
"""Which waveform family a transmitter emits: deramped FMCW or a pulse train."""


@dataclass(frozen=True)
class Transmitter:
    """A radar transmitter.

    Parameters
    ----------
    carrier_frequency_hz, bandwidth_hz : float
        Carrier (the centre of the band) and swept bandwidth, Hz.
    transmit_power_w : float
        Peak power (pulsed) or average power (FMCW), W.
    gain_tx_dbi : float
        Transmit antenna gain, dBi.
    chirp_duration_s, prf_hz : float
        Duration of one chirp or pulse, s, and the repetition frequency, Hz.
    waveform : {"fmcw", "pulsed"}
        Waveform family. Default ``"fmcw"``.

    Raises
    ------
    ValueError
        If a quantity is non-positive, the waveform unknown, or the duty cycle
        ``chirp_duration_s * prf_hz`` above one (exactly one is FMCW's norm).
    """

    carrier_frequency_hz: float
    bandwidth_hz: float
    transmit_power_w: float
    gain_tx_dbi: float
    chirp_duration_s: float
    prf_hz: float
    waveform: WaveformKind = "fmcw"

    def __post_init__(self) -> None:
        """Validate the parameters; see the class ``Raises`` section."""
        positive = ("carrier_frequency_hz", "bandwidth_hz", "transmit_power_w", "chirp_duration_s")
        for name in (*positive, "prf_hz"):
            if getattr(self, name) <= 0.0:
                msg = f"{name} must be strictly positive; got {getattr(self, name)!r}."
                raise ValueError(msg)
        if self.waveform not in ("fmcw", "pulsed"):
            msg = f"waveform must be 'fmcw' or 'pulsed'; got {self.waveform!r}."
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
    """A radar receiver.

    Parameters
    ----------
    sample_rate_hz : float
        Complex-baseband sample rate, Hz; FMCW samples the deramped beat signal.
    gain_rx_dbi : float
        Receive antenna gain, dBi.
    noise_figure_db : float
        Noise figure, dB, referred to 290 K.

    Raises
    ------
    ValueError
        If the sample rate is non-positive or the noise figure negative.
    """

    sample_rate_hz: float
    gain_rx_dbi: float
    noise_figure_db: float

    def __post_init__(self) -> None:
        """Validate the parameters; see the class ``Raises`` section."""
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
        r"""Thermal noise density :math:`N_0 = k T_0 F`, W/Hz [5]_."""
        return BOLTZMANN_JPK * STANDARD_NOISE_TEMPERATURE_K * self.noise_figure_linear

    @property
    def noise_power_per_sample_w(self) -> float:
        r"""Noise in one complex sample, :math:`N_0 f_s`, W: a sampler spans :math:`f_s` of band."""
        return self.noise_density_w_per_hz * self.sample_rate_hz


@dataclass(frozen=True)
class Site:
    """A fixed antenna position on the WGS-84 ellipsoid.

    Parameters
    ----------
    latitude_deg, longitude_deg : float
        Geodetic latitude and longitude, degrees, positive north and east.
    altitude_m : float
        Height above the ellipsoid, m.

    Raises
    ------
    ValueError
        If the latitude lies outside [-90, 90] degrees.
    """

    latitude_deg: float
    longitude_deg: float
    altitude_m: float

    def __post_init__(self) -> None:
        """Validate the latitude; see the class ``Raises`` section."""
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
    r"""Each target's route from transmitter to receiver: a struct of ``(n_targets,)`` arrays.

    Parameters
    ----------
    range_tx_m, range_rx_m : numpy.ndarray
        Transmit range :math:`R_T` and receive range :math:`R_R`, m.
    bistatic_range_m : numpy.ndarray
        :math:`R_b = R_T + R_R`, m.
    bistatic_range_rate_mps : numpy.ndarray
        :math:`dR_b/dt`, m/s, positive receding.
    bistatic_angle_rad : numpy.ndarray
        Angle :math:`\beta` at the target between the sites, rad: 0 monostatic,
        :math:`\pi` on the baseline [2]_.
    aod_rad, aoa_rad : numpy.ndarray
        (azimuth, elevation) from the transmitter (departure) and the receiver
        (arrival), rad, each in its own site's frame, ``(n_targets, 2)``.
    """

    range_tx_m: NDArray[np.float64]
    range_rx_m: NDArray[np.float64]
    bistatic_range_m: NDArray[np.float64]
    bistatic_range_rate_mps: NDArray[np.float64]
    bistatic_angle_rad: NDArray[np.float64]
    aod_rad: NDArray[np.float64]
    aoa_rad: NDArray[np.float64]

    @classmethod
    def monostatic(
        cls,
        range_m: ArrayLike,
        radial_velocity_mps: ArrayLike,
        *,
        azimuth_rad: ArrayLike = 0.0,
        elevation_rad: ArrayLike = 0.0,
    ) -> TargetGeometry:
        r"""Return the geometry seen from one site.

        :math:`R_T = R_R = R`, so :math:`R_b = 2R`, :math:`\dot R_b = 2\dot R`, :math:`\beta = 0`.

        Parameters
        ----------
        range_m : array_like
            Range :math:`R`, m.
        radial_velocity_mps : array_like
            :math:`dR/dt`, m/s, **positive receding**.
        azimuth_rad, elevation_rad : array_like, optional
            Look angles, rad, for both departure and arrival. Default 0.

        Returns
        -------
        TargetGeometry
            ``(n_targets,)`` after broadcasting.

        Examples
        --------
        >>> geometry = TargetGeometry.monostatic(10_000.0, -80.0)  # closing at 80 m/s
        >>> float(geometry.bistatic_range_m[0]), float(geometry.bistatic_range_rate_mps[0])
        (20000.0, -160.0)
        """
        range_arr, rate_arr, azimuth_arr, elevation_arr = np.broadcast_arrays(
            np.atleast_1d(np.asarray(range_m, dtype=np.float64)),
            np.asarray(radial_velocity_mps, dtype=np.float64),
            np.asarray(azimuth_rad, dtype=np.float64),
            np.asarray(elevation_rad, dtype=np.float64),
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
        )


@dataclass(frozen=True)
class Radar:
    r"""A transmitter and a receiver, at one site or two.

    Parameters
    ----------
    transmitter : Transmitter
        The transmit chain.
    receiver : Receiver
        The receive chain.
    tx_site : Site
        Where the transmitter stands.
    rx_site : Site or None, optional
        Where the receiver stands. Default ``None``: at ``tx_site``, monostatic.

    Raises
    ------
    ValueError
        If a pulsed receiver samples below the bandwidth (FMCW may: it deramps).

    Notes
    -----
    Sites are stationary and perfectly synchronised; the direct transmitter-to-
    receiver signal is not modelled.

    Examples
    --------
    >>> tx = Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3)
    >>> radar = Radar(tx, Receiver(1.0e6, 30.0, 3.0), Site(36.0025, -78.941, 60.0))
    >>> radar.is_monostatic, radar.baseline_m, round(radar.unambiguous_bistatic_range_m, 1)
    (True, 0.0, 74948.1)
    """

    transmitter: Transmitter
    receiver: Receiver
    tx_site: Site
    rx_site: Site | None = None

    def __post_init__(self) -> None:
        """Validate the pairing; see the class ``Raises`` section."""
        if (
            self.transmitter.waveform == "pulsed"
            and self.receiver.sample_rate_hz < self.transmitter.bandwidth_hz
        ):
            msg = "a pulsed receiver's sample_rate_hz is below bandwidth_hz; the pulse would alias."
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
        r"""Longest unfolded delay, s: :math:`1/\mathrm{PRF}` pulsed, :math:`f_s/2\alpha` FMCW.

        A pulse folds once it arrives after the next pulse; a deramped echo once
        its beat frequency :math:`\alpha\tau` passes the receiver's :math:`f_s/2`
        [1]_, [6]_.
        """
        if self.transmitter.waveform == "pulsed":
            return self.transmitter.pulse_repetition_interval_s
        return (self.receiver.sample_rate_hz / 2.0) / self.transmitter.sweep_rate_hzps

    @property
    def unambiguous_bistatic_range_m(self) -> float:
        r"""Longest unfolded bistatic range, :math:`c\,\tau_{ua}`, m; half of it monostatic."""
        return SPEED_OF_LIGHT_MPS * self.unambiguous_delay_s

    @property
    def unambiguous_bistatic_range_rate_mps(self) -> float:
        r"""Span of bistatic range rate before it folds, :math:`\lambda\,\mathrm{PRF}`, m/s [1]_.

        Doppler is unambiguous over one PRF; through :math:`f_D = -\dot R_b/\lambda`
        that is :math:`\lambda\,\mathrm{PRF}` of rate, :math:`\pm\lambda\,\mathrm{PRF}/4`
        of monostatic radial velocity.
        """
        return self.wavelength_m * self.transmitter.prf_hz

    @property
    def noise_power_matched_w(self) -> float:
        r"""Noise in the transmitted band, :math:`k T_0 B F`, W: the radar-equation term [5]_."""
        return self.receiver.noise_density_w_per_hz * self.transmitter.bandwidth_hz

    @property
    def n_samples_per_chirp(self) -> int:
        """Fast-time samples spanning one chirp or pulse."""
        return round(self.transmitter.chirp_duration_s * self.receiver.sample_rate_hz)

    @property
    def n_samples_per_pri(self) -> int:
        """Fast-time samples in one repetition interval: a cube's row length."""
        return round(self.transmitter.pulse_repetition_interval_s * self.receiver.sample_rate_hz)

    def sites_enu_m(self, origin: Site | None = None) -> NDArray[np.float64]:
        """Return the transmitter and receiver sites in ``origin``'s frame, ``(2, 3)``, m.

        ``origin`` defaults to the receiver's site.
        """
        origin_site = self._receive_site if origin is None else origin
        return np.stack([self.tx_site.enu_m(origin_site), self._receive_site.enu_m(origin_site)])

    def geometry(
        self,
        position_enu_m: ArrayLike,
        velocity_enu_mps: ArrayLike,
        *,
        origin: Site | None = None,
    ) -> TargetGeometry:
        r"""Return the route geometry of targets at given positions and velocities.

        With :math:`\hat u_T`, :math:`\hat u_R` the unit vectors from each site
        to the target and :math:`L` the baseline [2]_,

        .. math::

            \dot R_b = \vec v\cdot\hat u_T + \vec v\cdot\hat u_R,
            \qquad
            \cos\beta = \frac{R_T^2 + R_R^2 - L^2}{2 R_T R_R}

        The cosine is clipped to :math:`[-1, 1]`: :math:`\beta` is exactly 0 at
        zero baseline, and :math:`\pi`, not NaN, on the baseline.

        Parameters
        ----------
        position_enu_m : array_like
            Positions in ``origin``'s ENU frame, m, ``(n_targets, 3)`` or ``(3,)``.
        velocity_enu_mps : array_like
            Velocities in the same frame, m/s, broadcast against the positions.
        origin : Site, optional
            The frame's origin. Default: the receiver's site.

        Returns
        -------
        TargetGeometry
            ``(n_targets,)``. A target on a site gives NaN angles and rate there.

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
        range_tx_m, unit_tx, aod_rad = _look(position_m - tx_enu_m, self.tx_site, origin_site)
        range_rx_m, unit_rx, aoa_rad = _look(position_m - rx_enu_m, self._receive_site, origin_site)
        baseline_m = np.sqrt(np.sum((rx_enu_m - tx_enu_m) ** 2))
        cosine = (range_tx_m**2 + range_rx_m**2 - baseline_m**2) / (2.0 * range_tx_m * range_rx_m)
        return TargetGeometry(
            range_tx_m=range_tx_m,
            range_rx_m=range_rx_m,
            bistatic_range_m=range_tx_m + range_rx_m,
            bistatic_range_rate_mps=np.sum(velocity_mps * unit_tx, axis=-1)
            + np.sum(velocity_mps * unit_rx, axis=-1),
            bistatic_angle_rad=np.arccos(np.clip(cosine, -1.0, 1.0)),
            aod_rad=aod_rad,
            aoa_rad=aoa_rad,
        )


def _look(
    offset_enu_m: NDArray[np.float64], site: Site, origin: Site
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return range, unit vector and (azimuth, elevation) from ``site`` to each target.

    ``offset_enu_m`` is in ``origin``'s axes; the angles are measured in the
    site's own frame, so the vector is rotated into it unless the site is the origin.
    """
    range_m, azimuth_rad, elevation_rad = enu_to_range_azimuth_elevation_rad(offset_enu_m)
    if site != origin:
        site_axes_m = offset_enu_m @ (_enu_axes_in_ecef(origin) @ _enu_axes_in_ecef(site).T)
        _, azimuth_rad, elevation_rad = enu_to_range_azimuth_elevation_rad(site_axes_m)
    unit_vector = offset_enu_m / range_m[..., None]
    return range_m, unit_vector, np.stack([azimuth_rad, elevation_rad], axis=-1)


def _enu_axes_in_ecef(site: Site) -> NDArray[np.float64]:
    """Return the site's east, north and up axes in ECEF, as rows: ``ecef_to_enu_m``'s rotation."""
    sin_lat, cos_lat = np.sin(np.radians(site.latitude_deg)), np.cos(np.radians(site.latitude_deg))
    sin_lon, cos_lon = (
        np.sin(np.radians(site.longitude_deg)),
        np.cos(np.radians(site.longitude_deg)),
    )
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )


def geodetic_to_ecef_m(
    latitude_deg: ArrayLike, longitude_deg: ArrayLike, altitude_m: ArrayLike
) -> NDArray[np.float64]:
    r"""Convert WGS-84 latitude, longitude and altitude to ECEF metres, ``(..., 3)``.

    .. math::

        (x, y, z) = \left((N + h)\cos\phi\cos\lambda,\; (N + h)\cos\phi\sin\lambda,\;
                    (N(1 - e^2) + h)\sin\phi\right), \qquad N = a/\sqrt{1 - e^2\sin^2\phi}

    Closed form [3]_. Latitude in [-90, 90] and longitude in degrees, altitude
    above the ellipsoid in m, all broadcast together.

    Raises
    ------
    ValueError
        If any latitude lies outside [-90, 90] degrees.

    Examples
    --------
    >>> geodetic_to_ecef_m(0.0, 0.0, 0.0).tolist()  # the semi-major axis
    [6378137.0, 0.0, 0.0]
    """
    latitude_arr = np.asarray(latitude_deg, dtype=np.float64)
    if np.any(np.abs(latitude_arr) > 90.0):
        msg = "latitude_deg must lie in [-90, 90]; values outside it are not a latitude."
        raise ValueError(msg)
    altitude_arr = np.asarray(altitude_m, dtype=np.float64)
    latitude_rad = np.radians(latitude_arr)
    longitude_rad = np.radians(np.asarray(longitude_deg, dtype=np.float64))
    sin_latitude = np.sin(latitude_rad)
    cos_latitude = np.cos(latitude_rad)
    prime_vertical_radius_m = WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(
        1.0 - _ECCENTRICITY_SQUARED * sin_latitude**2
    )
    x_m = (prime_vertical_radius_m + altitude_arr) * cos_latitude * np.cos(longitude_rad)
    y_m = (prime_vertical_radius_m + altitude_arr) * cos_latitude * np.sin(longitude_rad)
    z_m = (prime_vertical_radius_m * (1.0 - _ECCENTRICITY_SQUARED) + altitude_arr) * sin_latitude
    result: NDArray[np.float64] = np.stack(np.broadcast_arrays(x_m, y_m, z_m), axis=-1)
    return result


def ecef_to_enu_m(
    ecef_m: ArrayLike,
    reference_latitude_deg: ArrayLike,
    reference_longitude_deg: ArrayLike,
    reference_altitude_m: ArrayLike,
) -> NDArray[np.float64]:
    r"""Translate ECEF metres ``(..., 3)`` to a reference point and rotate them into its ENU frame.

    .. math::

        \begin{bmatrix} e \\ n \\ u \end{bmatrix} =
        \begin{bmatrix}
            -\sin\lambda_0 & \cos\lambda_0 & 0 \\
            -\sin\phi_0\cos\lambda_0 & -\sin\phi_0\sin\lambda_0 & \cos\phi_0 \\
             \cos\phi_0\cos\lambda_0 &  \cos\phi_0\sin\lambda_0 & \sin\phi_0
        \end{bmatrix}
        \begin{bmatrix} \Delta x \\ \Delta y \\ \Delta z \end{bmatrix}

    A rotation, so the ENU norm is the slant range [4]_. The reference is
    geodetic, degrees and m. Returns east, north, up, m, ``(..., 3)``.

    Raises
    ------
    ValueError
        If ``ecef_m`` does not have length 3 on its last axis.
    """
    ecef_arr = np.asarray(ecef_m, dtype=np.float64)
    if ecef_arr.shape[-1] != 3:
        msg = f"ecef_m must have shape (..., 3) for (x, y, z); got {ecef_arr.shape}."
        raise ValueError(msg)
    offset_m = ecef_arr - geodetic_to_ecef_m(
        reference_latitude_deg, reference_longitude_deg, reference_altitude_m
    )
    latitude_rad = np.radians(np.asarray(reference_latitude_deg, dtype=np.float64))
    longitude_rad = np.radians(np.asarray(reference_longitude_deg, dtype=np.float64))
    sin_latitude = np.sin(latitude_rad)
    cos_latitude = np.cos(latitude_rad)
    sin_longitude = np.sin(longitude_rad)
    cos_longitude = np.cos(longitude_rad)
    delta_x_m = offset_m[..., 0]
    delta_y_m = offset_m[..., 1]
    delta_z_m = offset_m[..., 2]

    east_m = -sin_longitude * delta_x_m + cos_longitude * delta_y_m
    north_m = (
        -sin_latitude * cos_longitude * delta_x_m
        - sin_latitude * sin_longitude * delta_y_m
        + cos_latitude * delta_z_m
    )
    up_m = (
        cos_latitude * cos_longitude * delta_x_m
        + cos_latitude * sin_longitude * delta_y_m
        + sin_latitude * delta_z_m
    )
    result: NDArray[np.float64] = np.stack(np.broadcast_arrays(east_m, north_m, up_m), axis=-1)
    return result


def geodetic_to_enu_m(
    latitude_deg: ArrayLike,
    longitude_deg: ArrayLike,
    altitude_m: ArrayLike,
    reference_latitude_deg: ArrayLike,
    reference_longitude_deg: ArrayLike,
    reference_altitude_m: ArrayLike,
) -> NDArray[np.float64]:
    """Convert geodetic points to east, north, up metres ``(..., 3)`` about a geodetic reference.

    :func:`geodetic_to_ecef_m` then :func:`ecef_to_enu_m` [3]_, [4]_; degrees and m in.
    """
    ecef_m = geodetic_to_ecef_m(latitude_deg, longitude_deg, altitude_m)
    return ecef_to_enu_m(
        ecef_m, reference_latitude_deg, reference_longitude_deg, reference_altitude_m
    )


def enu_to_range_azimuth_elevation_rad(
    enu_m: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Convert east, north, up metres ``(..., 3)`` to slant range, azimuth and elevation.

    :math:`R = \sqrt{e^2 + n^2 + u^2}`, :math:`\mathrm{az} = \operatorname{atan2}(e, n)`
    in :math:`[0, 2\pi)`, zero north and clockwise, :math:`\mathrm{el} = \arcsin(u/R)` [4]_.
    Returns ``(range_m, azimuth_rad, elevation_rad)``, each ``(...,)``.

    Raises
    ------
    ValueError
        If ``enu_m`` does not have length 3 on its last axis.

    Examples
    --------
    >>> range_m, azimuth_rad, elevation_rad = enu_to_range_azimuth_elevation_rad((1000.0, 0, 0))
    >>> float(range_m), round(float(azimuth_rad), 12), float(elevation_rad)  # due east
    (1000.0, 1.570796326795, 0.0)
    """
    enu_arr = np.asarray(enu_m, dtype=np.float64)
    if enu_arr.shape[-1] != 3:
        msg = f"enu_m must have shape (..., 3) for (east, north, up); got {enu_arr.shape}."
        raise ValueError(msg)
    east_m = enu_arr[..., 0]
    north_m = enu_arr[..., 1]
    up_m = enu_arr[..., 2]
    range_m = np.sqrt(east_m**2 + north_m**2 + up_m**2)
    azimuth_rad = np.mod(np.arctan2(east_m, north_m), 2.0 * np.pi)
    # A tiny negative angle wraps to 2 pi - eps, which can round to exactly 2 pi.
    azimuth_rad = np.where(azimuth_rad >= 2.0 * np.pi, 0.0, azimuth_rad)
    elevation_rad = np.arcsin(np.divide(up_m, range_m))
    return range_m, azimuth_rad, elevation_rad


def range_azimuth_elevation_to_enu_m(
    range_m: ArrayLike, azimuth_rad: ArrayLike, elevation_rad: ArrayLike
) -> NDArray[np.float64]:
    r"""Convert slant range (m), azimuth and elevation (rad) to east, north, up metres ``(..., 3)``.

    The inverse of :func:`enu_to_range_azimuth_elevation_rad` [4]_: (east, north, up)
    :math:`= R(\cos\mathrm{el}\sin\mathrm{az}, \cos\mathrm{el}\cos\mathrm{az}, \sin\mathrm{el})`.

    Examples
    --------
    >>> np.round(range_azimuth_elevation_to_enu_m(1000.0, np.pi / 2, 0.0), 9).tolist()
    [1000.0, 0.0, 0.0]
    """
    range_arr = np.asarray(range_m, dtype=np.float64)
    azimuth_arr = np.asarray(azimuth_rad, dtype=np.float64)
    elevation_arr = np.asarray(elevation_rad, dtype=np.float64)
    ground_range_m = range_arr * np.cos(elevation_arr)
    east_m = ground_range_m * np.sin(azimuth_arr)
    north_m = ground_range_m * np.cos(azimuth_arr)
    up_m = range_arr * np.sin(elevation_arr)
    result: NDArray[np.float64] = np.stack(np.broadcast_arrays(east_m, north_m, up_m), axis=-1)
    return result
