r"""Initiators: starting a track from a measurement no track took.

An initiator writes what one measurement says into a prior, once per measurement, all at once:
:class:`DirectStateInitiator` copies a linear measurement's coordinates, and
:class:`DebiasedSphericalInitiator` converts a radar's range and angles to a position without
the naive conversion's bias (:func:`debiased_spherical_to_enu`). Track birth is kept apart
from the lifecycle (``tracker.py``), which decides when a track is confirmed or deleted.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001, §5.5 (initialisation of state estimators).
.. [2] S. Bordonaro, P. Willett and Y. Bar-Shalom, "Decorrelated unbiased converted
       measurement Kalman filter," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 50, no. 2,
       pp. 1431-1444, 2014, as restated by W. Wang et al., arXiv:1412.5524, §III-B,
       eqs. (8)-(14); Z. Duan, C. Han and X. R. Li, "Comments on 'Unbiased converted
       measurements for tracking'," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 40, no. 4,
       pp. 1374-1377, 2004.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.tracking.filters import _check_covariances
from radar_forge.tracking.measurement_models import (
    LinearMeasurement,
    MeasurementModel,
    RangeAzimuthElevationRangeRate,
)

__all__ = [
    "DebiasedSphericalInitiator",
    "DirectStateInitiator",
    "Initiator",
    "debiased_spherical_to_enu",
]

Rows = tuple[NDArray[np.float64], NDArray[np.float64]]


class Initiator(Protocol):
    """What starts tracks from measurements no track took.

    Attributes
    ----------
    prior_mean, prior_covariance : numpy.ndarray
        A new track's belief before its first measurement, ``(n_state,)`` and
        ``(n_state, n_state)``; the re-acquisition reset reads the covariance too.
    """

    prior_mean: NDArray[np.float64]
    prior_covariance: NDArray[np.float64]

    def initiate(
        self,
        measurements: NDArray[np.float64],
        noise_covariance: NDArray[np.float64],
        model: MeasurementModel,
    ) -> Rows | None:
        """Return a mean and covariance per measurement, or None if ``model`` cannot seed."""
        ...


class _PriorInitiator:
    """A prior, checked once, and the rule for writing measured coordinates into it."""

    def __init__(self, prior_mean: ArrayLike, prior_covariance: ArrayLike) -> None:
        mean = np.array(prior_mean, dtype=np.float64)
        if mean.ndim != 1 or not np.all(np.isfinite(mean)):
            msg = f"prior_mean must be a finite vector; got shape {mean.shape}."
            raise ValueError(msg)
        self.prior_mean = mean
        shape = (len(mean), len(mean))
        self.prior_covariance = _check_covariances(prior_covariance, shape, "prior_covariance")

    def _seed(
        self, rows: Sequence[int], values: NDArray[np.float64], blocks: NDArray[np.float64]
    ) -> Rows:
        """Copy the prior per measurement, and replace the measured rows and their block."""
        index = np.asarray(rows, dtype=np.intp)
        # Replacing a block the prior ties to other coordinates would leave cross terms that
        # no longer fit, and the covariance could stop being positive semidefinite.
        if np.any(np.delete(self.prior_covariance[index], index, axis=1) != 0):
            msg = f"the prior ties the measured coordinates {index.tolist()} to the others."
            raise ValueError(msg)
        mean = np.zeros((len(values), len(self.prior_mean))) + self.prior_mean
        covariance = np.zeros((len(values), *self.prior_covariance.shape)) + self.prior_covariance
        mean[:, index] = values
        covariance[:, index[:, None], index[None, :]] = blocks
        return mean, covariance


class DirectStateInitiator(_PriorInitiator):
    """Start a track by writing a linear measurement's coordinates into a prior [1]_.

    The measured coordinates take the values and their block takes R; the rest, such as an
    unmeasured velocity, keeps the prior, whose measured and unmeasured blocks must be
    uncorrelated. A model that is not a
    :class:`~radar_forge.tracking.measurement_models.LinearMeasurement` starts no track.

    Parameters
    ----------
    prior_mean : array_like
        ``(n_state,)``.
    prior_covariance : array_like
        ``(n_state, n_state)``, symmetric positive semidefinite.
    seed_names : sequence of str or None, optional
        The measured coordinates to copy, such as ``("bistatic_range_m",)`` while a folded
        rate cannot be trusted; None copies all.

    Raises
    ------
    ValueError
        If the prior is invalid.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.5; see the module References.
    """

    def __init__(
        self,
        prior_mean: ArrayLike,
        prior_covariance: ArrayLike,
        *,
        seed_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(prior_mean, prior_covariance)
        self.seed_names = None if seed_names is None else tuple(seed_names)

    def initiate(
        self,
        measurements: NDArray[np.float64],
        noise_covariance: NDArray[np.float64],
        model: MeasurementModel,
    ) -> Rows | None:
        """Return the seeded rows; raise if the prior ties a seeded coordinate to the rest."""
        if not isinstance(model, LinearMeasurement):
            return None
        names = self.seed_names or model.measurement_layout.names
        rows = np.asarray(model.measurement_layout.indices(names))
        blocks = noise_covariance[:, rows[:, None], rows[None, :]]
        return self._seed(model.state_layout.indices(names), measurements[:, rows], blocks)


def _angle_moments(
    angle_rad: NDArray[np.float64], sigma_rad: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    """Return E[cos a], E[sin a], E[cos² a], E[sin² a], E[sin a cos a] for a Gaussian angle.

    With λ = exp(-sigma²/2), E[cos a] = λ cos ā; the squares follow from the double angle,
    whose damping is λ⁴ = exp(-2 sigma²).
    """
    damping, double_damping = np.exp(-(sigma_rad**2) / 2), np.exp(-2 * sigma_rad**2)
    cos_2a = np.cos(2 * angle_rad)
    return (
        damping * np.cos(angle_rad),
        damping * np.sin(angle_rad),
        (1 + double_damping * cos_2a) / 2,
        (1 - double_damping * cos_2a) / 2,
        double_damping * np.sin(2 * angle_rad) / 2,
    )


def debiased_spherical_to_enu(
    range_m: ArrayLike,
    azimuth_rad: ArrayLike,
    elevation_rad: ArrayLike,
    sigma_range_m: ArrayLike,
    sigma_azimuth_rad: ArrayLike,
    sigma_elevation_rad: ArrayLike = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Convert spherical measurements to ENU positions and covariances, without bias.

    The naive conversion is biased towards the radar: the mean of cos(θ + w) is λ cos θ, with
    λ = exp(-σ²/2) < 1. Given the measurement, the true range and angles are independent
    Gaussians about the measured ones, so with p = r d and d = (cos φ sin θ, cos φ cos θ,
    sin φ) [2]_,

    .. math::

        E[p \mid z] = r_m E[d], \qquad
        \operatorname{cov}[p \mid z] = (r_m^2 + \sigma_r^2) E[d d^T] - r_m^2 E[d] E[d]^T:

    exact for Gaussian errors, and the posterior under a flat prior, which is what a new
    track's prior should be.

    Parameters
    ----------
    range_m, azimuth_rad, elevation_rad : array_like
        Slant range (positive), azimuth clockwise from north, elevation (0 for a 2-D radar),
        broadcastable to one shape ``(...)``.
    sigma_range_m, sigma_azimuth_rad, sigma_elevation_rad : array_like
        Their standard deviations, nonnegative.

    Returns
    -------
    position_m : numpy.ndarray
        ENU offset from the radar, ``(..., 3)``.
    covariance_m2 : numpy.ndarray
        Its covariance, ``(..., 3, 3)``.

    Raises
    ------
    ValueError
        If a value is not finite, a range is not positive or a sigma is negative.

    References
    ----------
    .. [2] Bordonaro, Willett and Bar-Shalom (2014); Duan, Han and Li (2004). See the module
           References.

    Examples
    --------
    20 km due north with 0.1 rad of azimuth error: 99.5 % of the naive point, on average.

    >>> position_m, _ = debiased_spherical_to_enu(20_000.0, 0.0, 0.0, 10.0, 0.1)
    >>> round(float(position_m[1]), 1)
    19900.2
    """
    given = (range_m, azimuth_rad, elevation_rad, sigma_range_m, sigma_azimuth_rad)
    values = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in given),
        np.asarray(sigma_elevation_rad, dtype=np.float64),
    )
    radius_m, azimuth, elevation, sigma_r, sigma_az, sigma_el = values
    if not np.all(np.isfinite(values)) or np.any(radius_m <= 0) or np.any(np.stack(values[3:]) < 0):
        msg = "need finite values, positive ranges and nonnegative standard deviations."
        raise ValueError(msg)
    cos_az, sin_az, cos2_az, sin2_az, sincos_az = _angle_moments(azimuth, sigma_az)
    cos_el, sin_el, cos2_el, sin2_el, sincos_el = _angle_moments(elevation, sigma_el)
    direction = np.stack([cos_el * sin_az, cos_el * cos_az, sin_el], axis=-1)
    # E[d dᵀ]: azimuth and elevation are independent, so each entry factorises.
    rows = [
        [cos2_el * sin2_az, cos2_el * sincos_az, sincos_el * sin_az],
        [cos2_el * sincos_az, cos2_el * cos2_az, sincos_el * cos_az],
        [sincos_el * sin_az, sincos_el * cos_az, sin2_el],
    ]
    second_moment = np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)
    position_m = radius_m[..., None] * direction
    covariance_m2 = (radius_m**2 + sigma_r**2)[..., None, None] * second_moment - (
        position_m[..., :, None] * position_m[..., None, :]
    )
    return position_m, (covariance_m2 + np.swapaxes(covariance_m2, -1, -2)) / 2


class DebiasedSphericalInitiator(_PriorInitiator):
    """Start a track from a spherical measurement converted to a position without bias.

    For a :class:`~radar_forge.tracking.measurement_models.RangeAzimuthElevationRangeRate`
    model that reports range and azimuth, :func:`debiased_spherical_to_enu` (offset by the
    site) replaces the prior's position and its block. The velocity keeps the prior, as does
    the height when no elevation is reported; every later update uses the spherical model
    itself [2]_.

    Parameters
    ----------
    prior_mean : array_like
        ``(n_state,)``.
    prior_covariance : array_like
        ``(n_state, n_state)``, its position block uncorrelated with the rest.

    Raises
    ------
    ValueError
        If the prior is invalid.

    References
    ----------
    .. [2] Bordonaro, Willett and Bar-Shalom (2014); see the module References.
    """

    def initiate(
        self,
        measurements: NDArray[np.float64],
        noise_covariance: NDArray[np.float64],
        model: MeasurementModel,
    ) -> Rows | None:
        """Return the converted rows; raise if R couples the range and angle errors."""
        if not isinstance(model, RangeAzimuthElevationRangeRate) or not (
            {"range_m", "azimuth_rad"} <= set(model.components)
        ):
            return None
        names = [n for n in ("range_m", "azimuth_rad", "elevation_rad") if n in model.components]
        rows = np.asarray(model.measurement_layout.indices(names))
        block = noise_covariance[:, rows[:, None], rows[None, :]]
        if np.any(block != block * np.eye(len(rows))):
            msg = f"the debiased conversion needs independent errors in {names}."
            raise ValueError(msg)
        value, sigma = measurements[:, rows], np.sqrt(np.diagonal(block, axis1=1, axis2=2))
        has_elevation = len(rows) == 3
        position_m, covariance_m2 = debiased_spherical_to_enu(
            value[:, 0],
            value[:, 1],
            value[:, 2] if has_elevation else 0.0,
            sigma[:, 0],
            sigma[:, 1],
            sigma[:, 2] if has_elevation else 0.0,
        )
        layout = model.state_layout
        axes = [a for a in "xyz" if f"{a}_m" in layout.names and (a != "z" or has_elevation)]
        enu = np.array(["xyz".index(a) for a in axes])
        return self._seed(
            layout.indices([f"{a}_m" for a in axes]),
            (position_m + model.sensor_enu_m)[:, enu],
            covariance_m2[:, enu[:, None], enu[None, :]],
        )
