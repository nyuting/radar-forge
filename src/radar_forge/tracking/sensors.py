"""Integration adapters normalize observations; they never perform tracking.

Classes:
- Measurement
- MeasurementBatch
- Sensor

Raw sensor outputs
        │
        ▼
Sensor.batch()
        │
        ▼
Measurement objects
        │
        ▼
MeasurementBatch
        │
        ▼
Tracker engine

References
----------
.. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
.. [2] Adapted local tracker; spec/tracker-001-provenance.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from radar_forge.tracking._numerics import FloatArray, covariance, timestamp, vector

if TYPE_CHECKING:
    from radar_forge.tracking.tracks import TrackSnapshot

__all__ = [
    "Measurement",
    "MeasurementBatch",
    "Sensor",
]


# Sensor registration and the observations it packages share one input contract.
# Prediction models remain in measurements.py and do not own these records.
@dataclass(frozen=True)
class Measurement:
    """Detached observation value (m,) and covariance (m,m) in model-defined units.

    Parameters
    ----------
    value, covariance : numpy.ndarray
        Finite float64 observation and positive semidefinite covariance.
    timestamp_s : float
        Seconds in the shared event epoch.
    sensor_id, measurement_model_id : str
        Explicit registered routes; metadata never selects mathematical behavior.

    References
    ----------
    .. [1] StoneSoup Detection; local normalized observation contract.
    """

    value: FloatArray
    covariance: FloatArray
    timestamp_s: float
    sensor_id: str
    measurement_model_id: str

    ###################################################################################################
    # This function checks a sensor reading and stores read-only copies before it enters the
    # tracker.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - None; checks the stored values, uncertainty, time, and source IDs, and replaces the arrays
    #   with copies.
    def __post_init__(self) -> None:
        """Validate the documented constructor contract."""

        # Validate that the measurement is a non-empty vector
        raw = np.asarray(self.value)
        if raw.ndim != 1 or not len(raw):
            raise ValueError("measurement value must be a nonempty vector")

        # Normalise all input arrays to validated float64 arrays; this lets the rest of our tracker
        # assume consistent dtype, shape, finiteness, and covariance properties
        value = vector(raw, len(raw))
        cov = covariance(self.covariance, len(raw))

        # Prevent accidental modification of measurements; this makes the measurement history a truly
        # immutable snapshot
        value.setflags(write=False)
        cov.setflags(write=False)

        # Note: __post_init__ must use object.__setattr__ because this dataclass is frozen
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "covariance", cov)

        # Normalise timestamp representation
        object.__setattr__(self, "timestamp_s", timestamp(self.timestamp_s))

        # Every measurement must identify its sensor and measurement model
        # - sensor_id:             where the observation came from
        # - measurement_model_id:  how its numbers should be interpreted mathematically
        if not self.sensor_id or not self.measurement_model_id:
            raise ValueError("sensor_id and measurement_model_id are required")
    ###################################################################################################

@dataclass(frozen=True)
class MeasurementBatch:
    """One sensor association opportunity, including an empty scan.

    Parameters
    ----------
    timestamp_s : float
        Shared scan time in seconds.
    sensor_id : str
        Registered sensor.
    measurements : tuple of Measurement
        All observations have the same sensor and timestamp.

    References
    ----------
    .. [1] Local asynchronous batch contract; spec/tracker-001-provenance.md.
    """

    timestamp_s: float
    sensor_id: str
    measurements: tuple[Measurement, ...] = ()

    ###################################################################################################
    # This function checks that all stored readings belong to one sensor scan, allowing an empty scan.
    # Empty batches are still valid; i.e. "sensor scanned but detected nothing" is still meaningful info
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - None; stores readings as a tuple and checks their sensor ID and time against the batch.
    def __post_init__(self) -> None:
        """Validate the documented constructor contract."""

        # Normalise the batch timestamp
        object.__setattr__(self, "timestamp_s", timestamp(self.timestamp_s))

        # Store measurements as an immutable tuple
        object.__setattr__(self, "measurements", tuple(self.measurements))

        # All measurements in a batch must come from the same sensor and scan at one event time
        # Mixing timestamps or sensor IDs would make prediction and association ambiguous
        if not self.sensor_id or any(
            m.timestamp_s != self.timestamp_s or m.sensor_id != self.sensor_id
            for m in self.measurements
        ):
            raise ValueError("all observations require the batch sensor and timestamp")
    ###################################################################################################

@dataclass(frozen=True)
class Sensor:
    """Allowed model routes and optional coverage predicate (track snapshot -> bool).

    Default coverage means every compatible track is observable on each scan.
    Applications with limited fields of view should supply observable explicitly.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    id: str
    measurement_model_ids: tuple[str, ...]
    observable: Callable[[TrackSnapshot], bool] | None = None

    ###################################################################################################
    # This function checks the stored sensor ID and its allowed reading types before scans can be created.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - None; stores model IDs as a tuple and rejects empty or duplicate IDs.
    def __post_init__(self) -> None:
        """Validate the documented constructor contract."""

        # Store model routes as an immutable tuple
        object.__setattr__(self, "measurement_model_ids", tuple(self.measurement_model_ids))
        
        # Require a valid sensor ID and at least one unique measurement route. A sensor explicitly declares
        # which measurement models it may produce. This prevents accidental routing of some measurement (e.g.
        # angle) to a model that doesn't support it (e.g. range / Doppler observation model)
        if (
            not self.id
            or not self.measurement_model_ids
            or any(not mid for mid in self.measurement_model_ids)
            or len(set(self.measurement_model_ids)) != len(self.measurement_model_ids)
        ):
            raise ValueError("sensor ID and unique nonempty model routes are required")
    ###################################################################################################

    ###################################################################################################
    # This function packages one sensor scan for track matching, including scans with no detections.
    #
    # Inputs:
    # - timestamp_s (float): Scan time in seconds, shared by all readings.
    # - observations (iterable of tuples): Each holds a model ID (str), readings (float64 array, (m,)), 
    #   and uncertainty (float64 array, (m, m)) in model units.
    #
    # Outputs:
    # - MeasurementBatch: Validated readings with this sensor's ID and scan time; values and
    #   uncertainty are copied.
    def batch(
        self, timestamp_s: float, observations: Iterable[tuple[str, FloatArray, FloatArray]]
    ) -> MeasurementBatch:
        """Normalize iterable of (model_id, value (m,), covariance (m,m)).

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        
        measurements = []   # this list will store all our Measurement objects

        # One physical sensor may have several observation types. Each detection therefore has an explicit
        # model route instead of assuming that the sensor ID alone determines the measurement equation.

        # Iterate over observations...
        for model_id, value, cov in observations:

            # ...reject measurement types that this sensor is not configured to produce...
            if model_id not in self.measurement_model_ids:
                raise ValueError(f"model {model_id!r} is not registered for sensor {self.id!r}")
            
            # ...and convert raw observations into validated Measurement objects (which package detections
            # and measurement models together)
            measurements.append(Measurement(value, cov, timestamp_s, self.id, model_id))
        
        # Group all detections from this scan into one batch. This batch will then undergo association
        return MeasurementBatch(timestamp_s, self.id, tuple(measurements))
    ###################################################################################################