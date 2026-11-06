r"""Propagation paths: how energy gets from the transmitter, via the scene, to the receiver.

Every propagation model produces a :class:`PropagationPaths` for
:mod:`radar_forge.signal` to sample (``spec/data-001-formats.md`` D1). A path is the
whole route, transmitter to receiver; its delay :math:`R_b/c`, phase
:math:`-2\pi R_b/\lambda` and Doppler :math:`-\dot R_b/\lambda` follow from its
length :math:`R_b = R_T + R_R` and rate, positive receding. Free space only.
Per-frame inputs are not validated: zero cross-section gives zero power. Leakage and near
clutter are specified as levels at the ADC rather than derived from a cross-section.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014.
.. [2] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005.
.. [3] P. Swerling, "Probability of detection for fluctuating targets," *IRE
       Transactions on Information Theory*, vol. IT-6, no. 2, pp. 269-308, 1960.
.. [4] M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill, 2001.
.. [5] A. Sabharwal, P. Schniter, D. Guo, D. W. Bliss, S. Rangarajan and R. Wichman, "In-band
       full-duplex wireless: challenges and opportunities," *IEEE J. Sel. Areas Commun.*,
       vol. 32, no. 9, pp. 1637-1652, 2014 (arXiv:1311.0456).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.constants import SPEED_OF_LIGHT_MPS
from radar_forge.radar import Radar, TargetGeometry

if TYPE_CHECKING:
    from radar_forge.scenario import ClutterSettings, LeakageSettings

__all__ = [
    "PointTarget",
    "PropagationPaths",
    "bistatic_range_rate_from_doppler_mps",
    "dbsm_to_m2",
    "disturbance_paths",
    "doppler_rate_hzps",
    "doppler_shift_hz",
    "leakage_paths",
    "line_of_sight_paths",
    "m2_to_dbsm",
    "path_amplitude_linear",
    "propagation_delay_s",
    "propagation_phase_rad",
    "received_power_w",
    "static_clutter_paths",
    "swerling_rcs_m2",
]

_FOUR_PI_CUBED = (4.0 * np.pi) ** 3

# Swerling 1-4 as chi-square degrees of freedom; Swerling 0 is the infinite limit.
_SWERLING_DEGREES_OF_FREEDOM = {1: 2, 2: 2, 3: 4, 4: 4}


@dataclass(frozen=True)
class PropagationPaths:
    """Propagation paths between a radar and its scene, one entry per path.

    ``doppler_hz`` is positive for a shortening path. ``amplitude_linear`` is a
    complex voltage, √W; the generators use its magnitude and rebuild the phase
    from the delay. ``aoa_rad`` and ``aod_rad`` are ``(n_paths, 2)`` (azimuth,
    elevation). ``doppler_rate_hzps`` and ``doppler_jerk_hzps2``, the first and
    second time derivatives of ``doppler_hz``, default to zero and are stored per path.
    """

    delay_s: NDArray[np.float64]
    doppler_hz: NDArray[np.float64]
    amplitude_linear: NDArray[np.complex128]
    aoa_rad: NDArray[np.float64]
    aod_rad: NDArray[np.float64]
    n_bounces: NDArray[np.int_]
    doppler_rate_hzps: NDArray[np.float64] = field(default_factory=lambda: np.zeros(()))
    doppler_jerk_hzps2: NDArray[np.float64] = field(default_factory=lambda: np.zeros(()))

    def __post_init__(self) -> None:
        """Reject per-path arrays that disagree on the number of paths."""
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
        for name in ("doppler_rate_hzps", "doppler_jerk_hzps2"):
            if np.shape(getattr(self, name)) not in ((), (n_paths,)):
                msg = f"{name} must be a scalar or have shape ({n_paths},)."
                raise ValueError(msg)
            # Store one value per path, so every per-path field indexes alike.
            object.__setattr__(self, name, np.zeros(n_paths) + getattr(self, name))

    @classmethod
    def concatenate(cls, *path_sets: PropagationPaths) -> PropagationPaths:
        """Join path sets into one, their paths in the order given."""
        if not path_sets:
            msg = "concatenate needs at least one PropagationPaths."
            raise ValueError(msg)
        columns = {f.name: [getattr(p, f.name) for p in path_sets] for f in fields(cls)}
        return cls(**{name: np.concatenate(values) for name, values in columns.items()})

    @property
    def n_paths(self) -> int:
        """Number of propagation paths."""
        return int(self.delay_s.shape[0])

    @property
    def bistatic_range_m(self) -> NDArray[np.float64]:
        r"""Length of each route, :math:`R_b = c\tau`, m. Monostatic, :math:`2R`."""
        result: NDArray[np.float64] = self.delay_s * SPEED_OF_LIGHT_MPS
        return result

    def bistatic_range_at_m(
        self, slow_time_s: ArrayLike, wavelength_m: float
    ) -> NDArray[np.float64]:
        r"""Return each route's length at slow-time instants, ``slow_time_s.shape + (n_paths,)``.

        Integrate the rate, which :func:`doppler_shift_hz` defines through
        :math:`f_D = -\dot R_b/\lambda`, so each derivative of :math:`R_b` is
        :math:`-\lambda` times one of :math:`f_D`:

        .. math::

            R_b(t) = R_b(0) - \lambda\left(f_D t + \tfrac{1}{2}\dot f_D t^2
                     + \tfrac{1}{6}\ddot f_D t^3\right)

        A closing path (:math:`f_D > 0`) shortens. The generators evaluate this
        at each pulse start and hold it across the pulse: stop-and-hop (Richards
        [1] §2.6.2). ``wavelength_m`` must be the one ``doppler_hz`` used.
        """
        time_s = np.asarray(slow_time_s, dtype=np.float64)[..., None]
        rate_mps = bistatic_range_rate_from_doppler_mps(self.doppler_hz, wavelength_m)
        acceleration_mps2 = bistatic_range_rate_from_doppler_mps(
            self.doppler_rate_hzps, wavelength_m
        )
        jerk_mps3 = bistatic_range_rate_from_doppler_mps(self.doppler_jerk_hzps2, wavelength_m)
        result: NDArray[np.float64] = (
            self.bistatic_range_m + rate_mps * time_s + 0.5 * acceleration_mps2 * time_s**2
        )
        # Added last, so zero jerk leaves the quadratic history bit for bit.
        result += jerk_mps3 * time_s**3 / 6.0
        return result


def propagation_delay_s(bistatic_range_m: ArrayLike) -> NDArray[np.float64]:
    r"""Return the time to travel a route, :math:`\tau = R_b/c`, s (Richards [1], §2.6.1)."""
    result: NDArray[np.float64] = (
        np.asarray(bistatic_range_m, dtype=np.float64) / SPEED_OF_LIGHT_MPS
    )
    return result


def propagation_phase_rad(
    bistatic_range_m: ArrayLike, wavelength_m: ArrayLike
) -> NDArray[np.float64]:
    r"""Return a route's unwrapped carrier phase :math:`-2\pi R_b/\lambda` (Richards [1] §2.6.1)."""
    range_arr = np.asarray(bistatic_range_m, dtype=np.float64)
    result: NDArray[np.float64] = -2.0 * np.pi * range_arr / np.asarray(wavelength_m)
    return result


def doppler_shift_hz(
    bistatic_range_rate_mps: ArrayLike, wavelength_m: ArrayLike
) -> NDArray[np.float64]:
    r"""Return the Doppler shift of a route, :math:`f_D = -\dot R_b/\lambda`, Hz.

    The one place the sign convention lives: the rate is positive receding, so
    the shift is positive closing (Willis [2] §6.1).

    Examples
    --------
    >>> float(doppler_shift_hz(30.0, 0.03))  # receding at 30 m/s: negative
    -1000.0
    """
    rate_arr = np.asarray(bistatic_range_rate_mps, dtype=np.float64)
    result: NDArray[np.float64] = -rate_arr / np.asarray(wavelength_m)
    return result


def doppler_rate_hzps(
    bistatic_range_acceleration_mps2: ArrayLike, wavelength_m: ArrayLike
) -> NDArray[np.float64]:
    r"""Return the Doppler rate :math:`\dot f_D = -\ddot R_b/\lambda`, Hz/s.

    The derivative of :func:`doppler_shift_hz`, which it calls, so the sign lives there:
    a closing target whose closing speed grows has a positive rate (Willis [2] §6.1).
    """
    return doppler_shift_hz(bistatic_range_acceleration_mps2, wavelength_m)


def bistatic_range_rate_from_doppler_mps(
    doppler_hz: ArrayLike, wavelength_m: ArrayLike
) -> NDArray[np.float64]:
    r"""Return the rate :math:`dR_b/dt = -\lambda f_D`, positive receding: the inverse Doppler."""
    result: NDArray[np.float64] = -np.asarray(doppler_hz, dtype=np.float64) * np.asarray(
        wavelength_m
    )
    return result


def path_amplitude_linear(power_w: ArrayLike, phase_rad: ArrayLike) -> NDArray[np.complex128]:
    r"""Return a path's complex voltage amplitude :math:`\sqrt{P_r}\,e^{j\phi}`, √W."""
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
    r"""Return the received power :math:`P_t G_t G_r \lambda^2\sigma/(4\pi)^3 R_T^2 R_R^2 L`, W.

    Willis [2] eq. (4.1); equal ranges give Richards' [1] §2.2.1 :math:`R^4` form.
    ``rcs_m2`` is the bistatic cross-section, taken as given: angle-independent here.
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
    """Build one direct path per target; the gains default to the boresight gains."""
    rcs_arr = np.asarray(rcs_m2, dtype=np.float64)
    if not np.all(rcs_arr >= 0.0):
        msg = "rcs_m2 must be non-negative; 0.0 is a target that returns nothing."
        raise ValueError(msg)
    if gain_tx_linear is None:
        gain_tx_linear = 10.0 ** (radar.transmitter.gain_tx_dbi / 10.0)
    if gain_rx_linear is None:
        gain_rx_linear = 10.0 ** (radar.receiver.gain_rx_dbi / 10.0)
    power_w = received_power_w(
        transmit_power_w=radar.transmitter.transmit_power_w,
        gain_tx_linear=gain_tx_linear,
        gain_rx_linear=gain_rx_linear,
        wavelength_m=radar.wavelength_m,
        rcs_m2=rcs_arr,
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
        doppler_rate_hzps=doppler_rate_hzps(
            geometry.bistatic_range_acceleration_mps2, radar.wavelength_m
        ),
    )


def leakage_paths(radar: Radar, lags: ArrayLike, inr_db: ArrayLike) -> PropagationPaths:
    r"""Build the transmitter's leakage into the receiver, one path per whole-sample lag.

    Leakage is specified at the ADC, not derived (Sabharwal [5] §IV-B to IV-D): what antenna
    isolation and an analog canceller leave. Lag :math:`m` is the route :math:`c\,m/f_s`, at
    zero Doppler; ``inr_db`` is each path's power over the noise in one sample,
    :attr:`~radar_forge.radar.Receiver.noise_power_per_sample_w`, not over the matched band.
    """
    lags_arr = np.atleast_1d(np.asarray(lags, dtype=np.float64))
    range_m = SPEED_OF_LIGHT_MPS * lags_arr / radar.receiver.sample_rate_hz
    return _specified_paths(radar, range_m, inr_db, 0.0)


def static_clutter_paths(
    radar: Radar, bistatic_range_m: ArrayLike, cnr_db: ArrayLike, *, doppler_hz: ArrayLike = 0.0
) -> PropagationPaths:
    """Build near clutter as fixed scatterers, one path per route, at a set Doppler (default 0).

    ``cnr_db`` is a per-sample ratio, as :func:`leakage_paths`: a level at the ADC, not a
    cross-section. A Doppler stands for a slowly moving scatterer, a tree or a mast in wind.
    """
    return _specified_paths(radar, bistatic_range_m, cnr_db, doppler_hz)


def disturbance_paths(
    radar: Radar, leakage: LeakageSettings | None, clutter: Sequence[ClutterSettings]
) -> PropagationPaths:
    """Join a scenario's ``[leakage]`` and ``[[clutter]]`` into one path set, leakage first."""
    lags, inr_db = (leakage.lags, leakage.inr_db) if leakage is not None else ((), ())
    return PropagationPaths.concatenate(
        leakage_paths(radar, lags, inr_db),
        static_clutter_paths(
            radar,
            [c.range_m for c in clutter],
            [c.relative_power_db for c in clutter],
            doppler_hz=[c.doppler_hz for c in clutter],
        ),
    )


def _specified_paths(
    radar: Radar, bistatic_range_m: ArrayLike, ratio_db: ArrayLike, doppler_hz: ArrayLike
) -> PropagationPaths:
    """Return paths at given routes and per-sample power ratios over noise, one bounce each."""
    range_m, ratio, doppler = np.broadcast_arrays(
        np.atleast_1d(np.asarray(bistatic_range_m, dtype=np.float64)),
        np.asarray(ratio_db, dtype=np.float64),
        np.asarray(doppler_hz, dtype=np.float64),
    )
    if not np.all(range_m >= 0.0):
        msg = "a leakage or clutter route must be non-negative."
        raise ValueError(msg)
    power_w = radar.receiver.noise_power_per_sample_w * 10.0 ** (ratio / 10.0)
    phase_rad = propagation_phase_rad(range_m, radar.wavelength_m)
    return PropagationPaths(
        delay_s=propagation_delay_s(range_m),
        doppler_hz=doppler.astype(np.float64),
        amplitude_linear=path_amplitude_linear(power_w, phase_rad),
        aoa_rad=np.zeros((range_m.size, 2)),
        aod_rad=np.zeros((range_m.size, 2)),
        n_bounces=np.ones(range_m.size, dtype=np.int_),
    )


def dbsm_to_m2(rcs_dbsm: ArrayLike) -> NDArray[np.float64]:
    """Convert a cross-section from dBsm to m²."""
    return np.asarray(10.0 ** (np.asarray(rcs_dbsm, dtype=np.float64) / 10.0), dtype=np.float64)


def m2_to_dbsm(rcs_m2: ArrayLike) -> NDArray[np.float64]:
    """Convert a strictly positive cross-section from m² to dBsm."""
    rcs_arr = np.asarray(rcs_m2, dtype=np.float64)
    if np.any(rcs_arr <= 0.0):
        msg = "rcs_m2 must be strictly positive; a non-positive area has no decibel value."
        raise ValueError(msg)
    return np.asarray(10.0 * np.log10(rcs_arr), dtype=np.float64)


@dataclass(frozen=True)
class PointTarget:
    """A point target with a constant cross-section (Swerling 0); zero is a noise-only scene."""

    rcs_m2: float
    name: str = "target"

    def __post_init__(self) -> None:
        """Reject a negative cross-section."""
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
    r"""Draw independent cross-sections, m², from Swerling case 0 to 4 about ``mean_rcs_m2``.

    Case 0 is the mean; the others are :math:`\bar\sigma\,\chi^2_k/k`, with
    :math:`k = 2` (exponential) for 1 and 2 and :math:`k = 4` for 3 and 4. Odd
    cases are redrawn per scan, even per pulse: the caller asks for ``(n_scans,)``
    and holds it, or for ``(n_scans, n_pulses)`` (Swerling [3]; Skolnik [4] §2.8).
    ``mean_rcs_m2`` broadcasts against ``shape`` by NumPy's rules, trailing axes
    first, so a per-target mean is ``(n_targets, 1)`` against ``(n_targets,
    n_pulses)``; a ``(n_targets,)`` mean would pair with the pulse axis instead.
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
