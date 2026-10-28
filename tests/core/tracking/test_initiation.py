"""Tests for radar_forge.core.tracking.initiation: starting a track from one measurement."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss
from numpy.typing import NDArray

from radar_forge.core.geodesy import range_azimuth_elevation_to_enu_m
from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.estimation import GaussianFilter
from radar_forge.core.tracking.initiation import (
    DebiasedSphericalInitiator,
    DirectStateInitiator,
    debiased_spherical_to_enu,
)
from radar_forge.core.tracking.linearisation import TaylorLinearisation, UnscentedLinearisation
from radar_forge.core.tracking.measurement_models import (
    BistaticRangeDopplerModel,
    CartesianPosition,
    Measurement,
    RangeAzimuthElevationRangeRate,
    SensorPose,
)
from radar_forge.core.tracking.motion import CartesianMotion

ORIGIN = (36.00250, -78.94100, 60.0)
MOTION = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
LAYOUT = MOTION.state_layout
POSITION = CartesianPosition(LAYOUT, ("x_m",))
MEASUREMENT = Measurement(np.array([50.0]), np.array([[4.0]]), 2.0, "radar", "position")

# Position and velocity correlated: replacing the position block would leave a
# cross term that no longer fits the new position variance.
CORRELATED = StateEstimate(np.zeros(2), np.array([[1.0, 0.5], [0.5, 1.0]]), 0.0, LAYOUT)
INDEPENDENT = StateEstimate(np.array([0.0, 7.0]), np.diag([1.0, 100.0]), 0.0, LAYOUT)


def _ukf(state: StateEstimate) -> GaussianFilter:
    return GaussianFilter(state, MOTION, UnscentedLinearisation())


def test_a_correlated_prior_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="independent observed/unobserved"):
        DirectStateInitiator(_ukf, CORRELATED, measured_names=("x_m",))


def test_an_unknown_measured_name_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="not in the layout"):
        DirectStateInitiator(_ukf, INDEPENDENT, measured_names=("y_m",))


def test_without_measured_names_a_correlated_prior_is_rejected_at_first_use() -> None:
    initiator = DirectStateInitiator(_ukf, CORRELATED)
    with pytest.raises(ValueError, match="independent observed/unobserved"):
        initiator.initiate(MEASUREMENT, POSITION)


def test_the_measured_coordinate_takes_the_measured_value() -> None:
    initiator = DirectStateInitiator(_ukf, INDEPENDENT, measured_names=("x_m",))
    estimator = initiator.initiate(MEASUREMENT, POSITION)
    assert estimator is not None
    # Exact: the value is copied.
    np.testing.assert_allclose(estimator.state.mean[0], 50.0, rtol=0, atol=0)


def test_an_unmeasured_coordinate_keeps_the_prior_value() -> None:
    initiator = DirectStateInitiator(_ukf, INDEPENDENT, measured_names=("x_m",))
    estimator = initiator.initiate(MEASUREMENT, POSITION)
    assert estimator is not None
    # Exact: the value is copied.
    np.testing.assert_allclose(estimator.state.mean[1], 7.0, rtol=0, atol=0)


def test_the_measured_block_takes_the_measurement_covariance() -> None:
    initiator = DirectStateInitiator(_ukf, INDEPENDENT, measured_names=("x_m",))
    estimator = initiator.initiate(MEASUREMENT, POSITION)
    assert estimator is not None
    # Exact: the blocks are copied, and the diagonal result is already symmetric.
    np.testing.assert_allclose(estimator.state.covariance, np.diag([4.0, 100.0]), rtol=0, atol=0)


def test_the_new_track_starts_at_the_measurement_time() -> None:
    estimator = DirectStateInitiator(_ukf, INDEPENDENT).initiate(MEASUREMENT, POSITION)
    assert estimator is not None
    assert estimator.state.timestamp_s == 2.0


def test_a_model_reporting_other_coordinates_than_declared_starts_no_track() -> None:
    initiator = DirectStateInitiator(_ukf, INDEPENDENT, measured_names=("x_m",))
    velocity = CartesianPosition(LAYOUT, ("xdot_mps",))
    assert initiator.initiate(MEASUREMENT, velocity) is None


def test_a_nonlinear_model_starts_no_track() -> None:
    layout = CartesianMotion(dict.fromkeys("xyz", "CV"), origin_lla_deg_m=ORIGIN).state_layout
    site = SensorPose(np.zeros(3), "ENU", ORIGIN)
    prior = StateEstimate(np.zeros(6), np.eye(6), 0.0, layout)
    initiator = DirectStateInitiator(_ukf, prior)
    assert initiator.initiate(MEASUREMENT, BistaticRangeDopplerModel(layout, site, site)) is None


# --------------------------------------------------------------------------- #
# debiased_spherical_to_enu and DebiasedSphericalInitiator
# --------------------------------------------------------------------------- #

# A target 20 km out with coarse angles: 0.1 rad is 2 km across the beam, where the naive
# conversion's bias, r (1 - λ), is about 120 m.
SPHERICAL = (20_000.0, 2.2, 0.3)  # range_m, azimuth_rad, elevation_rad
SIGMAS = (15.0, 0.1, 0.05)  # sigma_range_m, sigma_azimuth_rad, sigma_elevation_rad


def _posterior_moments(
    measured: tuple[float, float, float], sigmas: tuple[float, float, float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """E[p | z] and cov[p | z] by Gauss-Hermite quadrature, independently of the closed form.

    Given the measurement, the true range, azimuth and elevation are independent Gaussians
    about the measured ones. 40 probabilists' nodes per axis integrate cos and sin of a
    Gaussian of standard deviation 0.1 rad to roundoff.
    """
    nodes, weights = hermegauss(40)
    weights = weights / weights.sum()
    grids = [m + s * nodes for m, s in zip(measured, sigmas, strict=True)]
    range_m, azimuth_rad, elevation_rad = np.meshgrid(*grids, indexing="ij")
    weight = weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
    points = range_azimuth_elevation_to_enu_m(range_m, azimuth_rad, elevation_rad)
    mean = np.einsum("abc,abci->i", weight, points)
    deviation = points - mean
    covariance = np.einsum("abc,abci,abcj->ij", weight, deviation, deviation)
    return mean, covariance


def test_the_converted_position_is_the_mean_of_the_true_position() -> None:
    """Catches a wrong damping factor, a missing one, or east and north swapped.

    The quadrature integrates the geometry itself, so it shares no formula with the code.
    """
    position_m, _ = debiased_spherical_to_enu(*SPHERICAL, *SIGMAS)
    expected_m, _ = _posterior_moments(SPHERICAL, SIGMAS)
    # rtol 1e-12: both are exact; they differ by roundoff in 64 000 quadrature terms.
    np.testing.assert_allclose(position_m, expected_m, rtol=1e-12)


def test_the_converted_covariance_is_the_covariance_of_the_true_position() -> None:
    """Catches a wrong second moment: λ⁴ for the double angle, or the range variance left out."""
    _, covariance_m2 = debiased_spherical_to_enu(*SPHERICAL, *SIGMAS)
    _, expected_m2 = _posterior_moments(SPHERICAL, SIGMAS)
    # rtol 1e-9 on the largest entry: a difference of two terms of size r² ~ 4e8 m²
    # cancels to ~4e6 m², losing two digits to roundoff.
    np.testing.assert_allclose(
        covariance_m2, expected_m2, rtol=0, atol=1e-9 * np.abs(expected_m2).max()
    )


def test_the_naive_conversion_overshoots_by_the_damping_factors() -> None:
    """Shows why the conversion is debiased: the naive point is too far out.

    The true position's mean is λ_θ λ_φ times the naive point across the ground and λ_φ
    times it in height. Here the horizontal overshoot is about 119 m, nearly eight range
    standard deviations.
    """
    naive_m = range_azimuth_elevation_to_enu_m(*SPHERICAL)
    expected_m, _ = _posterior_moments(SPHERICAL, SIGMAS)
    damping_azimuth = np.exp(-(SIGMAS[1] ** 2) / 2)
    damping_elevation = np.exp(-(SIGMAS[2] ** 2) / 2)
    # rtol 1e-12: the quadrature is exact to roundoff.
    np.testing.assert_allclose(
        expected_m[:2], damping_azimuth * damping_elevation * naive_m[:2], rtol=1e-12
    )
    np.testing.assert_allclose(expected_m[2], damping_elevation * naive_m[2], rtol=1e-12)
    assert np.linalg.norm(naive_m[:2] - expected_m[:2]) > 7 * SIGMAS[0]


def test_with_small_errors_the_covariance_is_the_linearised_one() -> None:
    """At small angle errors the conversion reduces to J diag(sigma²) Jᵀ, the textbook form.

    J is the Jacobian of the polar-to-Cartesian map; the gap is second order in the angle
    variance, so at 1e-4 rad they agree well inside 1e-6 of the largest entry.
    """
    sigmas = (15.0, 1e-4, 1e-4)
    range_m, azimuth_rad, elevation_rad = SPHERICAL
    _, covariance_m2 = debiased_spherical_to_enu(*SPHERICAL, *sigmas)
    cos_el, sin_el = np.cos(elevation_rad), np.sin(elevation_rad)
    cos_az, sin_az = np.cos(azimuth_rad), np.sin(azimuth_rad)
    # Columns: d/dr, d/daz, d/del of (east, north, up) = r (cos el sin az, cos el cos az, sin el).
    jacobian = np.array(
        [
            [cos_el * sin_az, range_m * cos_el * cos_az, -range_m * sin_el * sin_az],
            [cos_el * cos_az, -range_m * cos_el * sin_az, -range_m * sin_el * cos_az],
            [sin_el, 0.0, range_m * cos_el],
        ]
    )
    linearised_m2 = jacobian @ np.diag(np.square(sigmas)) @ jacobian.T
    np.testing.assert_allclose(
        covariance_m2, linearised_m2, rtol=0, atol=1e-6 * np.abs(linearised_m2).max()
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (0.0, 0.0, 0.0, 1.0, 0.1, 0.0),
        (1000.0, np.nan, 0.0, 1.0, 0.1, 0.0),
        (1000.0, 0.0, 0.0, -1.0, 0.1, 0.0),
    ],
    ids=["zero-range", "nan-angle", "negative-sigma"],
)
def test_an_impossible_spherical_measurement_is_rejected(arguments: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="finite positive range"):
        debiased_spherical_to_enu(*arguments)


SPHERICAL_MOTION = CartesianMotion(dict.fromkeys("xyz", "CV"), origin_lla_deg_m=ORIGIN)
SPHERICAL_LAYOUT = SPHERICAL_MOTION.state_layout
SITE_M = np.array([500.0, -200.0, 40.0])
SPHERICAL_PRIOR = StateEstimate(
    np.array([0.0, 11.0, 0.0, 12.0, 777.0, 13.0]),
    np.diag([1e8, 400.0, 1e8, 400.0, 1e6, 400.0]),
    0.0,
    SPHERICAL_LAYOUT,
)


def _spherical_model(components: tuple[str, ...]) -> RangeAzimuthElevationRangeRate:
    return RangeAzimuthElevationRangeRate(
        SPHERICAL_LAYOUT, SensorPose(SITE_M, "ENU", ORIGIN), components
    )


def _spherical_measurement(components: tuple[str, ...]) -> Measurement:
    values = dict(zip(("range_m", "azimuth_rad", "elevation_rad"), SPHERICAL, strict=True))
    values["range_rate_mps"] = 30.0
    variances = dict(
        zip(("range_m", "azimuth_rad", "elevation_rad"), np.square(SIGMAS), strict=True)
    )
    variances["range_rate_mps"] = 1.0
    return Measurement(
        np.array([values[c] for c in components]),
        np.diag([variances[c] for c in components]),
        4.0,
        "radar",
        "spherical",
    )


def _ekf(state: StateEstimate) -> GaussianFilter:
    return GaussianFilter(state, SPHERICAL_MOTION, TaylorLinearisation())


FULL = ("range_m", "azimuth_rad", "elevation_rad", "range_rate_mps")
POSITIONS = ("x_m", "y_m", "z_m")
VELOCITIES = ("xdot_mps", "ydot_mps", "zdot_mps")


def test_a_spherical_track_starts_at_the_converted_position_from_the_site() -> None:
    """Catches the site offset left out, or a block written to the wrong coordinates."""
    estimator = DebiasedSphericalInitiator(_ekf, SPHERICAL_PRIOR).initiate(
        _spherical_measurement(FULL), _spherical_model(FULL)
    )
    assert estimator is not None
    position_m, covariance_m2 = debiased_spherical_to_enu(*SPHERICAL, *SIGMAS)
    rows = list(SPHERICAL_LAYOUT.indices(POSITIONS))
    # Exact: the converted values are copied in, after one addition of the site.
    np.testing.assert_array_equal(estimator.state.mean[rows], position_m + SITE_M)
    np.testing.assert_array_equal(estimator.state.covariance[np.ix_(rows, rows)], covariance_m2)
    assert estimator.state.timestamp_s == 4.0


def test_a_spherical_track_keeps_the_prior_velocity() -> None:
    """The range rate seeds nothing: it is one component of a three-component velocity."""
    estimator = DebiasedSphericalInitiator(_ekf, SPHERICAL_PRIOR).initiate(
        _spherical_measurement(FULL), _spherical_model(FULL)
    )
    assert estimator is not None
    rows = list(SPHERICAL_LAYOUT.indices(VELOCITIES))
    np.testing.assert_array_equal(estimator.state.mean[rows], [11.0, 12.0, 13.0])
    np.testing.assert_array_equal(
        estimator.state.covariance[np.ix_(rows, rows)], np.diag([400.0, 400.0, 400.0])
    )


def test_without_elevation_the_height_keeps_the_prior() -> None:
    """A 2-D radar fixes east and north only; the height stays the prior's 777 m."""
    components = ("range_m", "azimuth_rad", "range_rate_mps")
    estimator = DebiasedSphericalInitiator(_ekf, SPHERICAL_PRIOR).initiate(
        _spherical_measurement(components), _spherical_model(components)
    )
    assert estimator is not None
    (z_row,) = SPHERICAL_LAYOUT.indices(("z_m",))
    assert estimator.state.mean[z_row] == 777.0
    assert estimator.state.covariance[z_row, z_row] == 1e6
    position_m, _ = debiased_spherical_to_enu(SPHERICAL[0], SPHERICAL[1], 0.0, *SIGMAS[:2])
    rows = list(SPHERICAL_LAYOUT.indices(("x_m", "y_m")))
    np.testing.assert_array_equal(estimator.state.mean[rows], position_m[:2] + SITE_M[:2])


def test_a_cartesian_model_starts_no_spherical_track() -> None:
    initiator = DebiasedSphericalInitiator(_ekf, SPHERICAL_PRIOR)
    position = CartesianPosition(SPHERICAL_LAYOUT, ("x_m",))
    z = Measurement(np.array([50.0]), np.array([[4.0]]), 2.0, "radar", "position")
    assert initiator.initiate(z, position) is None


def test_a_spherical_model_without_azimuth_starts_no_track() -> None:
    """Range and range rate pin the target to a sphere, not to a point."""
    components = ("range_m", "range_rate_mps")
    initiator = DebiasedSphericalInitiator(_ekf, SPHERICAL_PRIOR)
    assert (
        initiator.initiate(_spherical_measurement(components), _spherical_model(components)) is None
    )


def test_a_coupled_spherical_covariance_is_rejected() -> None:
    measurement = _spherical_measurement(FULL)
    covariance = measurement.covariance.copy()
    covariance[0, 1] = covariance[1, 0] = 1e-3
    coupled = Measurement(measurement.value, covariance, 4.0, "radar", "spherical")
    with pytest.raises(ValueError, match="independent range, azimuth and elevation"):
        DebiasedSphericalInitiator(_ekf, SPHERICAL_PRIOR).initiate(coupled, _spherical_model(FULL))


def test_a_prior_linking_position_to_velocity_is_rejected_at_construction() -> None:
    covariance = SPHERICAL_PRIOR.covariance.copy()
    covariance[0, 1] = covariance[1, 0] = 10.0
    linked = StateEstimate(SPHERICAL_PRIOR.mean, covariance, 0.0, SPHERICAL_LAYOUT)
    with pytest.raises(ValueError, match="independent observed/unobserved"):
        DebiasedSphericalInitiator(_ekf, linked)
