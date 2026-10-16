"""Starting a new track from one measurement.

When a measurement matches no existing track, the tracker asks an *initiator*
for a new filter. Track initiation is kept apart from track management
(:mod:`radar_forge.core.tracking.lifecycle`), which decides when a track is
confirmed or deleted.

- :class:`TrackInitiator` is the interface every initiator follows.
- :class:`DirectStateInitiator` copies the measured coordinates straight into the
  new state, and takes everything else from a fixed prior: the starting guess
  used before any measurement.
- :class:`DebiasedSphericalInitiator` converts a radar's range and angles to a
  position, without the bias of the naive conversion
  (:func:`debiased_spherical_to_enu`), and takes everything else from a prior.

Not implemented yet: an initiator that picks a different rule for each
measurement model, and one that never starts a track from a detection, for a
sensor whose tracks must all come from
:meth:`~radar_forge.core.tracking.tracker.Tracker.seed`.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications
       to Tracking and Navigation*, Wiley, 2001, §5.5 (initialisation of
       state estimators). Its §5.5.3 starts a track from two measurements; the
       one-measurement-plus-prior rule here is the simpler variant.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.estimation import Estimator
from radar_forge.core.tracking.measurement_models import (
    CartesianPosition,
    Measurement,
    MeasurementModel,
    RangeAzimuthElevationRangeRate,
)

__all__ = [
    "DebiasedSphericalInitiator",
    "DirectStateInitiator",
    "TrackInitiator",
    "debiased_spherical_to_enu",
]


def _check_independent_blocks(covariance: NDArray[np.float64], indices: tuple[int, ...]) -> None:
    """Raise unless the measured and unmeasured parts of a covariance are uncorrelated.

    Parameters
    ----------
    covariance : numpy.ndarray
        Prior covariance, shape ``(n_state, n_state)``.
    indices : tuple of int
        Positions of the measured coordinates.

    Raises
    ------
    ValueError
        If any entry linking a measured coordinate to an unmeasured one is
        nonzero.
    """
    # Take the rows of the measured coordinates, then delete their columns. What is left is
    # the block that links each measured coordinate to every unmeasured one.
    if np.any(np.delete(covariance[list(indices)], list(indices), axis=1) != 0):
        msg = (
            "direct initiation requires independent observed/unobserved prior blocks: "
            f"the prior covariance links coordinates {indices} to the others."
        )
        raise ValueError(msg)


class TrackInitiator(Protocol):
    """The interface of an initiator: one measurement in, a new filter (or None) out."""

    def initiate(self, measurement: Measurement, model: MeasurementModel) -> Estimator | None:
        """Return a new filter started from ``measurement``, or None.

        Parameters
        ----------
        measurement : Measurement
            The measurement that matched no track.
        model : MeasurementModel
            The model that explains it.

        Returns
        -------
        Estimator or None
            A new filter at the measurement's time, independent of every other
            filter. None means this measurement cannot start a track, for example
            because it does not pin down enough of the state.
        """
        ...


class DirectStateInitiator:
    """Start a track by copying the measured coordinates into a prior.

    The new state starts as a copy of ``prior``. The coordinates the measurement
    reports are then replaced: their values by the measured value, and their
    block of the covariance by the measurement covariance R. Everything else,
    such as the velocity of a position-only sensor, keeps the prior's value.

    This only works for a :class:`CartesianPosition` model, which reports
    coordinates directly. It also needs the prior's measured and unmeasured
    coordinates to be independent (zero covariance between them). Otherwise,
    replacing one block would leave cross terms that no longer fit, and the
    covariance could stop being positive semidefinite.

    Parameters
    ----------
    estimator_factory : callable
        Builds a fresh filter from a :class:`StateEstimate`.
    prior : StateEstimate
        The starting guess: mean, shape ``(n_state,)``, and covariance, shape
        ``(n_state, n_state)``. Its timestamp is not used.
    measured_names : tuple of str or None, optional
        Keyword-only. The coordinates the measurements will report, such as
        ``("x_m", "y_m")``. If given, the independence of the prior is checked
        once, here, when the initiator is built. A model reporting other
        coordinates then gets None from :meth:`initiate`. If None (the
        default), the check runs on every call to :meth:`initiate`, during a
        scan. A bad prior is then found only when the first measurement tries
        to start a track.

    Raises
    ------
    ValueError
        If ``measured_names`` names a coordinate not in the prior's layout, or
        the prior links those coordinates to the others.

    Notes
    -----
    There is no initiator for
    :class:`~radar_forge.core.tracking.measurement_models.BistaticRangeDopplerModel`.
    One bistatic measurement gives a path length and a path rate. That pins
    the target to a surface, not to a point, so it cannot be copied into a
    state. This initiator returns None for it. So a tracker whose only model
    is bistatic never starts a track by itself. Its tracks must be added with
    :meth:`~radar_forge.core.tracking.tracker.Tracker.seed`.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.5; see the module References.

    Examples
    --------
    >>> import numpy as np
    >>> from radar_forge.core.tracking.motion import CartesianMotion
    >>> from radar_forge.core.tracking.estimation import GaussianFilter
    >>> from radar_forge.core.tracking.linearisation import UnscentedLinearisation
    >>> motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=(36.0, -78.9, 60.0))
    >>> layout = motion.state_layout
    >>> prior = StateEstimate(np.zeros(2), np.diag([1.0, 100.0]), 0.0, layout)
    >>> initiator = DirectStateInitiator(
    ...     lambda state: GaussianFilter(state, motion, UnscentedLinearisation()),
    ...     prior,
    ...     measured_names=("x_m",),
    ... )
    >>> z = Measurement(np.array([50.0]), np.array([[4.0]]), 2.0, "radar", "position")
    >>> ukf = initiator.initiate(z, CartesianPosition(layout, ("x_m",)))
    >>> ukf.state.mean, ukf.state.timestamp_s
    (array([50.,  0.]), 2.0)
    """

    def __init__(
        self,
        estimator_factory: Callable[[StateEstimate], Estimator],
        prior: StateEstimate,
        *,
        measured_names: tuple[str, ...] | None = None,
    ) -> None:
        measured = None if measured_names is None else prior.state_layout.indices(measured_names)
        if measured is not None:
            _check_independent_blocks(prior.covariance, measured)

        self.estimator_factory = estimator_factory
        self.prior = prior
        self._measured = measured

    def initiate(self, measurement: Measurement, model: MeasurementModel) -> Estimator | None:
        """Return a new filter whose measured coordinates come from ``measurement``.

        Parameters
        ----------
        measurement : Measurement
            The measurement that matched no track. Its value has shape
            ``(n_meas,)`` and its covariance ``(n_meas, n_meas)``.
        model : MeasurementModel
            The model that explains it.

        Returns
        -------
        Estimator or None
            A new filter at the measurement's time. None if ``model`` is not a
            :class:`CartesianPosition` on the prior's layout, or reports other
            coordinates than ``measured_names``.

        Raises
        ------
        ValueError
            Only when ``measured_names`` was not given: if the prior links the
            model's coordinates to the others.
        """
        if not isinstance(model, CartesianPosition) or model.state_layout != (
            self.prior.state_layout
        ):
            return None
        if self._measured is not None and model.indices != self._measured:
            return None
        if self._measured is None:
            _check_independent_blocks(self.prior.covariance, model.indices)

        indices = list(model.indices)
        mean = self.prior.mean.copy()
        covariance = self.prior.covariance.copy()
        mean[indices] = measurement.value
        covariance[np.ix_(indices, indices)] = measurement.covariance
        return self.estimator_factory(
            StateEstimate(mean, covariance, measurement.timestamp_s, model.state_layout)
        )


def _angle_moments(angle_rad: float, sigma_rad: float) -> tuple[float, float, float, float, float]:
    """Return the moments of cos and sin of a Gaussian angle, about its measured value.

    For an angle a ~ N(angle_rad, sigma_rad²), with λ = exp(-sigma_rad²/2), the characteristic
    function of a Gaussian gives E[cos a] = λ cos ā and E[sin a] = λ sin ā. The squares
    and the product follow from the double-angle formulas, with λ⁴ = exp(-2 sigma_rad²)
    for the angle 2a.

    Parameters
    ----------
    angle_rad : float
        The mean angle ā, in radians.
    sigma_rad : float
        The standard deviation, in radians.

    Returns
    -------
    tuple of float
        E[cos a], E[sin a], E[cos² a], E[sin² a] and E[sin a cos a].
    """
    damping = np.exp(-(sigma_rad**2) / 2)  # λ
    double_damping = damping**4  # exp(-2 sigma_rad²), the λ of the angle 2a
    cos_2a = np.cos(2 * angle_rad)
    return (
        float(damping * np.cos(angle_rad)),
        float(damping * np.sin(angle_rad)),
        float((1 + double_damping * cos_2a) / 2),
        float((1 - double_damping * cos_2a) / 2),
        float(double_damping * np.sin(2 * angle_rad) / 2),
    )


def debiased_spherical_to_enu(
    range_m: float,
    azimuth_rad: float,
    elevation_rad: float,
    sigma_range_m: float,
    sigma_azimuth_rad: float,
    sigma_elevation_rad: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Convert one spherical measurement to a Cartesian position and covariance, without bias.

    The naive conversion, putting the measured range and angles into
    :func:`~radar_forge.core.geodesy.range_azimuth_elevation_to_enu_m`, is biased: a
    noisy angle pulls the converted point towards the radar, because the average of
    cos(θ + w) is λ cos θ with λ = exp(-σ²/2) < 1. At long range and coarse angle
    accuracy the bias is many range cells, and the covariance from linearising the
    conversion is too small across the beam. This function returns instead the mean
    and covariance of the true position given the measurement [1]_ [2]_:

    .. math::

        \hat{p} = E[p \mid z], \qquad P = \operatorname{cov}[p \mid z].

    Parameters
    ----------
    range_m : float
        Measured slant range, in metres.
    azimuth_rad : float
        Measured azimuth, radians, zero at north and clockwise.
    elevation_rad : float
        Measured elevation, radians. Zero for a 2-D radar, which assumes the target is
        at the radar's height.
    sigma_range_m : float
        Standard deviation of the range error, in metres.
    sigma_azimuth_rad : float
        Standard deviation of the azimuth error, in radians.
    sigma_elevation_rad : float, optional
        Standard deviation of the elevation error, in radians. Zero (the default) for a
        2-D radar.

    Returns
    -------
    position_m : numpy.ndarray
        East, north and up of the target relative to the radar, in metres, shape ``(3,)``.
    covariance_m2 : numpy.ndarray
        Its covariance, in m², shape ``(3, 3)``.

    Raises
    ------
    ValueError
        If any value is not finite, the range is not positive, or a standard deviation
        is negative.

    Notes
    -----
    Write the true position as p = r d, with r the true range and d the unit vector
    (cos φ sin θ, cos φ cos θ, sin φ) in east, north and up, θ the azimuth and φ the
    elevation. Given the measurement, the true range, azimuth and elevation are
    independent Gaussians about the measured ones [1]_. So

    .. math::

        E[p \mid z] = r_m\, E[d], \qquad
        \operatorname{cov}[p \mid z] = (r_m^2 + \sigma_r^2)\, E[d d^T] - r_m^2\, E[d]\, E[d]^T,

    and every entry of E[d] and E[d dᵀ] is a product of one azimuth moment and one
    elevation moment, E[cos θ] = λ_θ cos θ_m and so on (``_angle_moments``). Written out,
    these are the explicit entries of [1]_, eqs. (8)-(14), in this library's angle
    convention: azimuth from north, so east takes the sine.

    The mean is λ_θ λ_φ times the naive conversion's horizontal part, and λ_φ times its
    vertical part: the debiasing *shrinks* the point towards the radar. The naive point
    is the conversion of the measured angles, but the true angles are spread about
    them, and the average of a spread-out direction is shorter than one unit vector.

    This is the measurement-conditioned conversion of Duan, Han and Li [2]_, whose
    moments are exact for Gaussian errors. Lerro and Bar-Shalom [3]_ (2-D) and
    Suchomski [4]_ (3-D) reach approximate moments by nested conditioning, which leaves a
    small bias [1]_. Under a flat prior on the true position, these moments are those of
    its posterior, which is what a new track's prior should be.

    References
    ----------
    .. [1] S. Bordonaro, P. Willett and Y. Bar-Shalom, "Decorrelated unbiased converted
           measurement Kalman filter," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 50,
           no. 2, pp. 1431-1444, 2014. As restated, with the range rate added, by
           W. Wang et al., arXiv:1412.5524, §III-B, eqs. (8)-(14).
    .. [2] Z. Duan, C. Han and X. R. Li, "Comments on 'Unbiased converted measurements
           for tracking'," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 40, no. 4,
           pp. 1374-1377, 2004.
    .. [3] D. Lerro and Y. Bar-Shalom, "Tracking with debiased consistent converted
           measurements versus EKF," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 29,
           no. 3, pp. 1015-1022, 1993. Its 2-D equations are restated in A. J. Pei,
           "Tracking methods for converted radar measurements," *Johns Hopkins APL
           Technical Digest*, vol. 38, no. 1, 2025, eqs. (11)-(26).
    .. [4] P. Suchomski, "Explicit expressions for debiased statistics of 3D converted
           measurements," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 35, no. 1,
           pp. 368-370, 1999.

    Examples
    --------
    A target 20 km due north, level with the radar, with 0.1 rad of azimuth error. The
    naive conversion puts it at 20 000 m; the true position is on average 99.5 % of that:

    >>> position_m, covariance_m2 = debiased_spherical_to_enu(20_000.0, 0.0, 0.0, 10.0, 0.1)
    >>> round(float(position_m[1]), 1)
    19900.2
    """
    values = np.array(
        [range_m, azimuth_rad, elevation_rad, sigma_range_m, sigma_azimuth_rad, sigma_elevation_rad]
    )
    if not np.all(np.isfinite(values)) or range_m <= 0 or np.any(values[3:] < 0):
        msg = (
            "a spherical measurement needs a finite positive range, finite angles and "
            f"non-negative standard deviations; got {values.tolist()}."
        )
        raise ValueError(msg)

    cos_az, sin_az, cos2_az, sin2_az, sincos_az = _angle_moments(azimuth_rad, sigma_azimuth_rad)
    cos_el, sin_el, cos2_el, sin2_el, sincos_el = _angle_moments(elevation_rad, sigma_elevation_rad)
    # E[d] for d = (cos φ sin θ, cos φ cos θ, sin φ): east, north, up.
    mean_direction = np.array([cos_el * sin_az, cos_el * cos_az, sin_el])
    # E[d dᵀ]: azimuth and elevation are independent, so each entry factorises.
    second_moment = np.array(
        [
            [cos2_el * sin2_az, cos2_el * sincos_az, sincos_el * sin_az],
            [cos2_el * sincos_az, cos2_el * cos2_az, sincos_el * cos_az],
            [sincos_el * sin_az, sincos_el * cos_az, sin2_el],
        ]
    )
    position_m = range_m * mean_direction
    covariance_m2 = (range_m**2 + sigma_range_m**2) * second_moment - np.outer(
        position_m, position_m
    )
    return position_m, (covariance_m2 + covariance_m2.T) / 2


class DebiasedSphericalInitiator:
    """Start a track from one spherical measurement, converted to a position without bias.

    For a :class:`~radar_forge.core.tracking.measurement_models.RangeAzimuthElevationRangeRate`
    measurement, the range and angles are converted with :func:`debiased_spherical_to_enu`,
    offset by the radar's site, and written into a copy of ``prior``: the converted
    position as the mean of the position coordinates, its covariance as their block.
    Every later update uses the spherical model itself (``spec/tracker-001.md`` §6.1).

    Everything else keeps the prior's value:

    - the velocity. A range rate is one component of the velocity, so it cannot seed
      all three; the first update with it corrects the velocity along the line of sight;
    - the height, for a radar that reports no elevation.

    As with :class:`DirectStateInitiator`, the prior's position block must be independent
    of everything else, so that replacing it keeps the covariance positive semidefinite.

    Parameters
    ----------
    estimator_factory : callable
        Builds a fresh filter from a :class:`StateEstimate`.
    prior : StateEstimate
        The starting guess: mean, shape ``(n_state,)``, and covariance, shape
        ``(n_state, n_state)``. Its timestamp is not used.

    Raises
    ------
    ValueError
        If the prior holds no ``x_m`` or no ``y_m``, or links the positions to the other
        coordinates.

    Notes
    -----
    A 2-D radar (no elevation) is converted in the horizontal plane through the site,
    which assumes the target is at the radar's height. The error of that assumption
    grows with the target's elevation angle.

    References
    ----------
    .. [1] Bordonaro, Willett and Bar-Shalom (2014); see :func:`debiased_spherical_to_enu`.

    Examples
    --------
    >>> import numpy as np
    >>> from radar_forge.core.tracking.motion import CartesianMotion
    >>> from radar_forge.core.tracking.estimation import GaussianFilter
    >>> from radar_forge.core.tracking.linearisation import TaylorLinearisation
    >>> from radar_forge.core.tracking.measurement_models import (
    ...     RangeAzimuthElevationRangeRate,
    ...     SensorPose,
    ... )
    >>> origin = (36.0, -78.9, 60.0)
    >>> motion = CartesianMotion(dict.fromkeys("xyz", "CV"), origin_lla_deg_m=origin)
    >>> layout = motion.state_layout
    >>> prior = StateEstimate(np.zeros(6), np.diag([1e8, 1e4] * 3), 0.0, layout)
    >>> initiator = DebiasedSphericalInitiator(
    ...     lambda state: GaussianFilter(state, motion, TaylorLinearisation()), prior
    ... )
    >>> model = RangeAzimuthElevationRangeRate(
    ...     layout, SensorPose(np.zeros(3), "ENU", origin), ("range_m", "azimuth_rad")
    ... )
    >>> z = Measurement(np.array([1000.0, np.pi / 2]), np.diag([25.0, 1e-6]), 2.0, "r", "sph")
    >>> ekf = initiator.initiate(z, model)
    >>> round(float(ekf.state.mean[layout.indices(("x_m",))[0]]), 1)
    1000.0
    """

    def __init__(
        self,
        estimator_factory: Callable[[StateEstimate], Estimator],
        prior: StateEstimate,
    ) -> None:
        names = prior.state_layout.names
        if "x_m" not in names or "y_m" not in names:
            msg = f"spherical initiation needs x_m and y_m in the state; got {names}."
            raise ValueError(msg)
        self._axes = tuple(axis for axis in "xyz" if f"{axis}_m" in names)
        _check_independent_blocks(
            prior.covariance, prior.state_layout.indices(tuple(f"{a}_m" for a in self._axes))
        )
        self.estimator_factory = estimator_factory
        self.prior = prior

    def initiate(self, measurement: Measurement, model: MeasurementModel) -> Estimator | None:
        """Return a new filter whose position comes from the converted ``measurement``.

        Parameters
        ----------
        measurement : Measurement
            The measurement that matched no track. Its value has shape ``(n_meas,)`` and
            its covariance ``(n_meas, n_meas)``, in the model's component order.
        model : MeasurementModel
            The model that explains it.

        Returns
        -------
        Estimator or None
            A new filter at the measurement's time. None if ``model`` is not a
            :class:`~radar_forge.core.tracking.measurement_models.RangeAzimuthElevationRangeRate`
            on the prior's layout, or does not report both range and azimuth.

        Raises
        ------
        ValueError
            If the measurement covariance correlates the range, azimuth and elevation
            errors, which the conversion assumes independent.
        """
        if not isinstance(model, RangeAzimuthElevationRangeRate) or (
            model.state_layout != self.prior.state_layout
        ):
            return None
        components = model.measurement_layout.names
        if "range_m" not in components or "azimuth_rad" not in components:
            return None

        positional = [n for n in ("range_m", "azimuth_rad", "elevation_rad") if n in components]
        rows = list(model.measurement_layout.indices(tuple(positional)))
        block = measurement.covariance[np.ix_(rows, rows)]
        if np.any(block != np.diag(np.diag(block))):
            msg = (
                "debiased spherical conversion needs independent range, azimuth and "
                f"elevation errors; the measurement covariance couples {positional}."
            )
            raise ValueError(msg)
        value = dict(zip(positional, measurement.value[rows], strict=True))
        sigma = dict(zip(positional, np.sqrt(np.diag(block)), strict=True))
        position_m, covariance_m2 = debiased_spherical_to_enu(
            value["range_m"],
            value["azimuth_rad"],
            value.get("elevation_rad", 0.0),
            sigma["range_m"],
            sigma["azimuth_rad"],
            sigma.get("elevation_rad", 0.0),
        )
        position_m = position_m + model.sensor.position_m

        # Without elevation the height is unknown, so it keeps the prior's value.
        axes = [a for a in self._axes if a != "z" or "elevation_rad" in components]
        enu = ["xyz".index(a) for a in axes]
        state_rows = list(self.prior.state_layout.indices(tuple(f"{a}_m" for a in axes)))
        mean = self.prior.mean.copy()
        covariance = self.prior.covariance.copy()
        mean[state_rows] = position_m[enu]
        covariance[np.ix_(state_rows, state_rows)] = covariance_m2[np.ix_(enu, enu)]
        return self.estimator_factory(
            StateEstimate(mean, covariance, measurement.timestamp_s, model.state_layout)
        )
