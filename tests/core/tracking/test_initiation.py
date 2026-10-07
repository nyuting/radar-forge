"""Tests for radar_forge.core.tracking.initiation: starting a track from one measurement."""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.initiation import DirectStateInitiator
from radar_forge.core.tracking.measurement_models import (
    BistaticRangeDopplerModel,
    CartesianPosition,
    Measurement,
    SensorPose,
)
from radar_forge.core.tracking.motion import CartesianMotion
from radar_forge.core.tracking.ukf import UKF

ORIGIN = (36.00250, -78.94100, 60.0)
MOTION = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
LAYOUT = MOTION.state_layout
POSITION = CartesianPosition(LAYOUT, ("x_m",))
MEASUREMENT = Measurement(np.array([50.0]), np.array([[4.0]]), 2.0, "radar", "position")

# Position and velocity correlated: replacing the position block would leave a
# cross term that no longer fits the new position variance.
CORRELATED = StateEstimate(np.zeros(2), np.array([[1.0, 0.5], [0.5, 1.0]]), 0.0, LAYOUT)
INDEPENDENT = StateEstimate(np.array([0.0, 7.0]), np.diag([1.0, 100.0]), 0.0, LAYOUT)


def _ukf(state: StateEstimate) -> UKF:
    return UKF(state, MOTION)


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
