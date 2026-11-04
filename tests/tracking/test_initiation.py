"""Tests for radar_forge.tracking.initiation: starting tracks from measurements.

Ground truth is closed form: the Gaussian characteristic function gives the debiasing factor
exactly, a zero-error measurement converts with no bias, and the small-error covariance is the
linearised one. A Monte Carlo of the true position given the measurement checks the moments.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.tracking.initiation import (
    DebiasedSphericalInitiator,
    DirectStateInitiator,
    debiased_spherical_to_enu,
)
from radar_forge.tracking.measurement_models import (
    BistaticRangeRateModel,
    LinearMeasurement,
    RangeAzimuthElevationRangeRate,
)
from radar_forge.tracking.models import CartesianMotion

LAYOUT = CartesianMotion().state_layout
PRIOR_MEAN = np.array([0.0, 1.0, 0.0, 2.0, 500.0, 3.0])
PRIOR_COVARIANCE = np.diag([1e8, 100.0, 1e8, 100.0, 1e6, 100.0])


def test_a_direct_start_copies_the_measured_coordinates_and_keeps_the_rest() -> None:
    model = LinearMeasurement(LAYOUT, ("y_m", "x_m"))
    initiator = DirectStateInitiator(PRIOR_MEAN, PRIOR_COVARIANCE)
    noise = np.array([[[4.0, 1.0], [1.0, 9.0]], [[1.0, 0.0], [0.0, 1.0]]])
    mean, covariance = initiator.initiate(np.array([[20.0, 10.0], [7.0, 8.0]]), noise, model)  # type: ignore[misc]
    np.testing.assert_array_equal(mean[0], [10.0, 1.0, 20.0, 2.0, 500.0, 3.0])
    np.testing.assert_array_equal(covariance[0][np.ix_([2, 0], [2, 0])], noise[0])
    np.testing.assert_array_equal(np.diag(covariance[1])[[1, 3, 4, 5]], [100.0, 100.0, 1e6, 100.0])


def test_seed_names_copy_only_the_named_coordinates() -> None:
    """A track whose folded rate cannot be trusted starts from its range alone."""
    model = LinearMeasurement(LAYOUT, ("x_m", "xdot_mps"))
    initiator = DirectStateInitiator(PRIOR_MEAN, PRIOR_COVARIANCE, seed_names=("x_m",))
    mean, covariance = initiator.initiate(np.array([[50.0, -9.0]]), np.eye(2)[None], model)  # type: ignore[misc]
    assert mean[0, 0] == 50.0 and mean[0, 1] == 1.0 and covariance[0, 1, 1] == 100.0


def test_a_prior_that_ties_measured_to_unmeasured_coordinates_is_refused() -> None:
    covariance = PRIOR_COVARIANCE.copy()
    covariance[0, 1] = covariance[1, 0] = 5.0
    initiator = DirectStateInitiator(PRIOR_MEAN, covariance)
    with pytest.raises(ValueError, match="ties"):
        initiator.initiate(np.zeros((1, 1)), np.eye(1)[None], LinearMeasurement(LAYOUT, ("x_m",)))


def test_a_nonlinear_model_starts_no_direct_track() -> None:
    model = BistaticRangeRateModel(LAYOUT, np.zeros(3), np.ones(3))
    initiator = DirectStateInitiator(PRIOR_MEAN, PRIOR_COVARIANCE)
    assert initiator.initiate(np.zeros((1, 2)), np.eye(2)[None], model) is None


def test_the_debiased_mean_is_the_naive_point_shrunk_by_the_damping_factors() -> None:
    """E[cos(θ + w)] = exp(-σ²/2) cos θ: north and east shrink by both factors, up by one."""
    sigma_az, sigma_el, azimuth, elevation = 0.1, 0.05, 0.7, 0.3
    position_m, _ = debiased_spherical_to_enu(1e4, azimuth, elevation, 10.0, sigma_az, sigma_el)
    naive = 1e4 * np.array(
        [
            np.cos(elevation) * np.sin(azimuth),
            np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation),
        ]
    )
    damping = np.exp(-np.array([sigma_az**2 + sigma_el**2] * 2 + [sigma_el**2]) / 2)
    # rtol 1e-14: a few products of exact factors.
    np.testing.assert_allclose(position_m, naive * damping, rtol=1e-14)


def test_with_no_angle_error_the_conversion_is_the_naive_one() -> None:
    position_m, covariance_m2 = debiased_spherical_to_enu(1e4, 0.7, 0.3, 0.0, 0.0, 0.0)
    assert np.linalg.norm(position_m) == pytest.approx(1e4, rel=1e-14)
    np.testing.assert_allclose(covariance_m2, 0.0, atol=1e-6)


def test_with_small_errors_the_covariance_is_the_linearised_one() -> None:
    """J R Jᵀ, with J the Jacobian of (r, θ, φ) to ENU, as the errors go to zero."""
    r, azimuth, elevation, sr, sa, se = 1e4, 0.7, 0.3, 1.0, 1e-5, 1e-5
    _, covariance_m2 = debiased_spherical_to_enu(r, azimuth, elevation, sr, sa, se)
    ce, se_, ca, sa_ = np.cos(elevation), np.sin(elevation), np.cos(azimuth), np.sin(azimuth)
    jacobian = np.array(
        [
            [ce * sa_, r * ce * ca, -r * se_ * sa_],
            [ce * ca, -r * ce * sa_, -r * se_ * ca],
            [se_, 0.0, r * ce],
        ]
    )
    expected = jacobian @ np.diag([sr**2, sa**2, se**2]) @ jacobian.T
    # rtol 1e-6: the exact moments differ from the linearised ones at order sigma².
    np.testing.assert_allclose(covariance_m2, expected, rtol=1e-6, atol=1e-9)


def test_the_debiased_moments_are_those_of_the_true_position_given_the_measurement() -> None:
    """Integrate p = r d over Gaussian (r, θ, φ) about the measured ones, by quadrature.

    40-point Gauss-Hermite rules per axis integrate these smooth functions to roundoff, so
    the comparison is exact rather than a Monte Carlo interval.
    """
    r, azimuth, elevation, sr, sa, se = 2e4, 0.2, 0.1, 30.0, 0.05, 0.03
    position_m, covariance_m2 = debiased_spherical_to_enu(r, azimuth, elevation, sr, sa, se)
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)
    weights = weights / weights.sum()
    rr, aa, ee = np.meshgrid(
        r + sr * nodes, azimuth + sa * nodes, elevation + se * nodes, indexing="ij"
    )
    points = np.stack(
        [rr * np.cos(ee) * np.sin(aa), rr * np.cos(ee) * np.cos(aa), rr * np.sin(ee)], -1
    )
    weight = np.einsum("i,j,k->ijk", weights, weights, weights)
    mean = np.einsum("ijk,ijkd->d", weight, points)
    deviation = points - mean
    covariance = np.einsum("ijk,ijkd,ijke->de", weight, deviation, deviation)
    # rtol 1e-10: quadrature and closed form agree to about 1e-13 of the largest entry.
    np.testing.assert_allclose(position_m, mean, rtol=1e-10)
    np.testing.assert_allclose(covariance_m2, covariance, rtol=1e-10, atol=1e-10 * covariance.max())


@pytest.mark.parametrize(
    "arguments",
    [(0.0, 0.0, 0.0, 1.0, 0.1), (1e3, np.nan, 0.0, 1.0, 0.1), (1e3, 0.0, 0.0, -1.0, 0.1)],
)
def test_an_impossible_spherical_measurement_is_refused(arguments: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="positive ranges"):
        debiased_spherical_to_enu(*arguments)


def test_a_spherical_start_places_the_track_from_the_site_and_keeps_its_velocity() -> None:
    site_m = np.array([100.0, -200.0, 30.0])
    model = RangeAzimuthElevationRangeRate(LAYOUT, site_m)
    initiator = DebiasedSphericalInitiator(PRIOR_MEAN, PRIOR_COVARIANCE)
    z = np.array([[1000.0, np.pi / 2, 0.0, -5.0]])
    mean, _ = initiator.initiate(z, np.diag([25.0, 1e-8, 1e-8, 1.0])[None], model)  # type: ignore[misc]
    # atol 1e-3 m: a 1e-4 rad sigma pulls the point in by 1e-5 m at 1 km.
    np.testing.assert_allclose(mean[0, [0, 2, 4]], site_m + np.array([1000.0, 0.0, 0.0]), atol=1e-3)
    np.testing.assert_array_equal(mean[0, [1, 3, 5]], [1.0, 2.0, 3.0])


def test_without_elevation_the_height_keeps_the_prior() -> None:
    model = RangeAzimuthElevationRangeRate(LAYOUT, np.zeros(3), ("range_m", "azimuth_rad"))
    initiator = DebiasedSphericalInitiator(PRIOR_MEAN, PRIOR_COVARIANCE)
    mean, covariance = initiator.initiate(
        np.array([[1000.0, 0.0]]), np.diag([25.0, 1e-6])[None], model
    )  # type: ignore[misc]
    assert mean[0, 4] == 500.0 and covariance[0, 4, 4] == 1e6


def test_a_spherical_model_without_azimuth_or_another_model_starts_no_track() -> None:
    initiator = DebiasedSphericalInitiator(PRIOR_MEAN, PRIOR_COVARIANCE)
    range_only = RangeAzimuthElevationRangeRate(LAYOUT, np.zeros(3), ("range_m",))
    assert initiator.initiate(np.ones((1, 1)), np.eye(1)[None], range_only) is None
    assert (
        initiator.initiate(np.ones((1, 1)), np.eye(1)[None], LinearMeasurement(LAYOUT, ("x_m",)))
        is None
    )


def test_coupled_range_and_angle_errors_are_refused() -> None:
    model = RangeAzimuthElevationRangeRate(LAYOUT, np.zeros(3), ("range_m", "azimuth_rad"))
    initiator = DebiasedSphericalInitiator(PRIOR_MEAN, PRIOR_COVARIANCE)
    with pytest.raises(ValueError, match="independent"):
        initiator.initiate(
            np.array([[1000.0, 0.0]]), np.array([[[25.0, 1e-4], [1e-4, 1e-6]]]), model
        )


def test_an_invalid_prior_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="prior"):
        DirectStateInitiator([0.0, np.nan], np.eye(2))
    with pytest.raises(ValueError, match="prior_covariance"):
        DirectStateInitiator([0.0, 0.0], -np.eye(2))
