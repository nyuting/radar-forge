"""Observation contracts and compatibility imports after the file-boundary refactor."""

import numpy as np
import pytest

import radar_forge.tracking as tracking
from radar_forge.tracking import measurements, sensors
from radar_forge.tracking.sensors import Measurement, MeasurementBatch, Sensor


def test_observation_exports_are_identical_classes():
    for name in ("Measurement", "MeasurementBatch"):
        owner = getattr(sensors, name)
        assert getattr(measurements, name) is owner
        assert getattr(tracking, name) is owner
        assert owner.__module__ == sensors.__name__


def test_sensor_batch_detaches_observations_and_allows_empty_scans():
    sensor = Sensor("sensor", ("position",))
    value = np.array([1.0, 2.0])
    covariance = np.eye(2)
    batch = sensor.batch(1.0, [("position", value, covariance)])
    observation = batch.measurements[0]
    value[:] = 0
    covariance[:] = 0
    np.testing.assert_array_equal(observation.value, [1.0, 2.0])
    np.testing.assert_array_equal(observation.covariance, np.eye(2))
    assert not observation.value.flags.writeable
    assert not observation.covariance.flags.writeable
    assert observation.sensor_id == batch.sensor_id == sensor.id
    assert observation.timestamp_s == batch.timestamp_s == 1.0
    assert sensor.batch(2.0, []).measurements == ()


@pytest.mark.parametrize(("time_s", "sensor_id"), [(2.0, "sensor"), (1.0, "other")])
def test_batch_rejects_mismatched_timestamp_or_sensor(time_s, sensor_id):
    observation = Measurement(np.ones(1), np.eye(1), time_s, sensor_id, "position")
    with pytest.raises(ValueError, match="batch sensor and timestamp"):
        MeasurementBatch(1.0, "sensor", (observation,))


@pytest.mark.parametrize(
    ("value", "covariance"),
    [
        (np.array([]), np.empty((0, 0))),
        (np.ones((1, 1)), np.eye(1)),
        (np.ones(2), np.eye(1)),
        (np.array([np.nan]), np.eye(1)),
        (np.ones(1), np.array([[-1.0]])),
        (np.ones(1), np.array([[np.inf]])),
    ],
)
def test_observation_rejects_invalid_shapes_or_covariance(value, covariance):
    with pytest.raises(ValueError):
        Measurement(value, covariance, 1.0, "sensor", "position")


def test_scan_rejects_nonfinite_time_and_unregistered_routes():
    sensor = Sensor("sensor", ("position",))
    with pytest.raises(ValueError, match="timestamp_s must be finite"):
        sensor.batch(np.nan, [])
    with pytest.raises(ValueError, match="not registered"):
        sensor.batch(1.0, [("unknown", np.ones(1), np.eye(1))])
