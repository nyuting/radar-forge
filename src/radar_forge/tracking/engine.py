"""Event orchestration; state coordinates and sensor mathematics remain opaque.

Classes:
- TrackerEngine: The main class that orchestrates the entire tracking process. 
                 Predicts tracks, matches readings, updates estimates, and manages track lifetimes.

References
----------
.. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
.. [2] Adapted local tracker; spec/tracker-001-provenance.md.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from radar_forge.tracking.association import Associator, ChiSquareGate
from radar_forge.tracking.estimation import Estimator
from radar_forge.tracking.management import TrackManager
from radar_forge.tracking.measurements import MeasurementModel
from radar_forge.tracking.sensors import MeasurementBatch, Sensor
from radar_forge.tracking.tracks import Track, TrackSnapshot, TrackStatus

__all__ = [
    "TrackerEngine",
]


# One TrackerEngine instance is expected to manage all sensors, models, and tracks for a single
# scenario. Each track owns an independent estimator, and each estimator is expected to be 
# initialised from a justified prior. 
class TrackerEngine:
    """Process nondecreasing timestamp_s batches, jointly assigning each scan.

    Tracks outside sensor coverage receive no miss. Empty scans are valid.
    Model routing and dimensions are validated before prediction. The engine is
    single-threaded; custom component exceptions are not transactionally rolled back.

    References
    ----------
    .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
    .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
    """

    ###################################################################################################
    # This function connects sensors, prediction rules, matching, and track management into one
    # tracker.
    #
    # Inputs:
    # - measurement_models (mapping of str to MeasurementModel): Reading-type IDs and their
    #   prediction rules.
    # - sensors (mapping of str to Sensor): Sensor IDs, allowed reading types, and optional
    #   visibility checks.
    # - gate (ChiSquareGate): Rule for rejecting reading/track pairs that disagree too much.
    # - associator (Associator): Rule for choosing among the remaining reading/track pairs.
    # - manager (TrackManager): Rules for starting, confirming, and removing tracks.
    # - cost (str): Pair score: nis uses uncertainty-scaled differences; negative_log_likelihood
    #   uses how well a reading fits.
    #
    # Outputs:
    # - None; checks registered IDs, stores components, and starts with no active tracks.
    def __init__(
        self,
        measurement_models: Mapping[str, MeasurementModel],
        sensors: Mapping[str, Sensor],
        gate: ChiSquareGate,
        associator: Associator,
        manager: TrackManager,
        cost: str = "nis",
    ) -> None:

        # Initialise the tracker engine
        self.measurement_models = dict(measurement_models)  # E.g. {"model1": MeasurementModel(...), "model2": MeasurementModel(...)}
        self.sensors = dict(sensors)                        # E.g. {"radar1": Sensor(...), "radar2": Sensor(...)}
        self.gate = gate                                    # E.g. ChiSquareGate(probability=0.997)
        self.associator = associator                        # E.g. NearestNeighbor() or GlobalNearestNeighbor()
        self.manager = manager                              # E.g. TrackManager(confirm_threshold=3, delete_threshold=5)
        if cost not in ("nis", "negative_log_likelihood"):
            raise ValueError("cost must be nis or negative_log_likelihood")
        self.cost = cost                                    # E.g. "nis" or "negative_log_likelihood"

        # Initialise tracks as an empty list (since no tracks detected yet), and last_timestamp_s for measurement synchronisation
        self.tracks: list[Track] = []
        self.last_timestamp_s: float | None = None

        # Check that each sensor has a valid ID and that its measurement models are registered in the engine.
        for key, sensor in self.sensors.items():
            if (
                key != sensor.id
                or not sensor.measurement_model_ids
                or any(mid not in self.measurement_models for mid in sensor.measurement_model_ids)
            ):
                raise ValueError("sensor registry contains invalid model routes or IDs")
    ###################################################################################################

    ###################################################################################################
    # This function initialises a filter (supplied in the function input) for a Track, then appends the 
    # new Track to all tracks
    #
    # Inputs:
    # - estimator (Estimator): Independent filter at the engine's current time, or the first
    #   track's time.
    # - sensor_id (str or None): Starting sensor ID to record, if supplied.
    #
    # Outputs:
    # - TrackSnapshot: Independent snapshot of the new track; the live track is also added to the
    #   engine.
    def seed(self, estimator: Estimator, sensor_id: str | None = None) -> TrackSnapshot:
        """Register an independent estimator initialized from a justified target prior.

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        # If the provided filter is already tracking some existing track, it can't be initialised to the new track
        if any(t.estimator is estimator for t in self.tracks):
            raise ValueError("each track must own an independent estimator")
        
        # If the provided filter's timestamp is not updated to the current timestamp, raise an error
        if (self.last_timestamp_s is not None and estimator.state.timestamp_s != self.last_timestamp_s):
            raise ValueError("seed timestamp_s must equal current engine time_s")

        # Check that the provided filter's timestamp is the same as the timestamp of all existing filters
        if (self.tracks and estimator.state.timestamp_s != self.tracks[0].estimator.state.timestamp_s):
            raise ValueError("seed tracks must share the current timestamp_s")

        # If the above sanity checks pass, the supplied filter, 'estimator', gets seeded to the new Track
        track = self.manager.seed(estimator, sensor_id)
        self.tracks.append(track)
        return track.snapshot()
    ###################################################################################################

    ###################################################################################################
    # This function processes one scan: predicts tracks, matches readings, updates estimates, and
    # starts or removes tracks.
    #
    # Inputs:
    # - batch (MeasurementBatch): Readings from one sensor at one time; may be empty. Older scans
    #   are rejected. 
    #
    # Outputs:
    # - tuple of TrackSnapshot: Current tracks and tracks just removed; the engine also updates
    #   its live tracks and time.
    def process(self, batch: MeasurementBatch) -> tuple[TrackSnapshot, ...]:
        """Consume one scan and return current snapshots plus newly deleted tracks.

        TODO: Below is the current MeasurementBatch. To consider whether batching of measurements 
        is needed, and need to look into how the timestamp_s works (not every measurement comes in 
        at EXACTLY the same time...)

        E.g. batch = MeasurementBatch(
            timestamp_s = 42.0
            sensor_id = "Radar1"
            measurements = (
                                Measurement(value, covariance, timestamp_s, sensor_id, measurement_model_id),
                                Measurement(...),
                                ...
                            )
        )

        References
        ----------
        .. [1] Bar-Shalom et al., Estimation with Applications to Tracking, 2001.
        .. [2] Local tracker adaptation; spec/tracker-001-provenance.md.
        """
        ############################ SANITY CHECKS ON MEASUREMENTS ############################
        # Timestamps must be nondecreasing, tracks must not precede the batch, 
        # and the sensor must be registered.
        
        #  TODO: consider whether an out-of-sequence batch (e.g. if it was less than 1 second ago) 
        # could somehow be used to update the track
        if self.last_timestamp_s is not None and batch.timestamp_s < self.last_timestamp_s:
            raise ValueError(
                "out-of-sequence batch rejected; buffering/smoothing is not implemented"
            )
        if any(t.estimator.state.timestamp_s > batch.timestamp_s for t in self.tracks):
            raise ValueError("batch precedes a seeded track")
        if batch.sensor_id not in self.sensors:
            raise ValueError(f"unknown sensor {batch.sensor_id!r}")

        ############################### MEASUREMENT MODEL VALIDATION ###############################
        # Check that each measurement's model is registered for the sensor and that the measurement's
        # dimension matches the model's measurement space. Store the models for later use.
        sensor = self.sensors[batch.sensor_id]
        models = []
        for measurement in batch.measurements:
            mid = measurement.measurement_model_id
            if mid not in sensor.measurement_model_ids:
                raise ValueError(f"unregistered model {mid!r} for sensor {sensor.id!r}")
            model = self.measurement_models[mid]
            if len(measurement.value) != model.measurement_space.dimension:
                raise ValueError("measurement dimension does not match registered model")
            models.append(model)

        ######################### PREDICT TO CURRENT TIME AND MANAGE TRACKS #########################
        # Make predictions and expire old tracks 
        expired = []
        for track in self.tracks:
            track.estimator.predict_to(batch.timestamp_s)
            self.manager.expire(track, batch.timestamp_s)
            if track.status == TrackStatus.DELETED:
                expired.append(track.snapshot())
        
        # Filter eligible tracks (measurement model maps to same state space for the track and sensor
        # must be observable for the current sensor)
        self.tracks = [track for track in self.tracks if track.status != TrackStatus.DELETED]
        eligible = [t for t in self.tracks
            if any(
                t.estimator.state.state_space == self.measurement_models[mid].state_space
                for mid in sensor.measurement_model_ids
            )
            and (sensor.observable is None or sensor.observable(t.snapshot())) # sensor is not required to be 
            # observable (i.e. passive sensor), but if it is (i.e. active donor), the track must be observable
        ]

        ###################### COMPUTE COSTS AND ASSOCIATE TRACKS WITH MEASUREMENTS ######################
        # Compute the cost matrix for eligible tracks and measurements. The cost is either the normalized
        # innovation squared (nis) or the negative log likelihood. If a track and measurement are
        # incompatible (different state spaces or outside the gate), the cost is set to infinity.
        costs = np.full((len(eligible), len(models)), np.inf)
        for i, track in enumerate(eligible): # for each eligible track
            for j, (measurement, model) in enumerate(zip(batch.measurements, models, strict=True)): # for each measurement and its model
                if track.estimator.state.state_space != model.state_space:
                    continue
                stats = track.estimator.innovation_statistics(measurement, model)
                if self.gate.accepts(stats):
                    costs[i, j] = stats.nis if self.cost == "nis" else -stats.log_likelihood
        association = self.associator.associate(costs)

        ######################################## TRACK MANAGEMENT ###########################################
        # For each matched track and measurement, update the track's estimator with the measurement and record the update in the manager.
        for i, j in association.matches:
            measurement = batch.measurements[j]
            eligible[i].estimator.update(measurement, models[j])
            self.manager.record(eligible[i], measurement)

        # For unassigned tracks, record them as missed. 
        for i in association.unassigned_tracks:
            self.manager.record(eligible[i])

        # For unassigned measurements, create new tracks if possible. 
        # TODO: consider adding additional support to wait for 2 unassigned measurements (within a certain time window and measurement gate) 
        # before initialising a new track based on those measurements, to improve gating andreduce state uncertainty and false positives. 
        for j in association.unassigned_measurements:
            new_track = self.manager.create(batch.measurements[j], models[j])
            if new_track is not None:
                self.tracks.append(new_track)

        # Finally, expire old tracks and return snapshots of current and expired tracks.
        snapshots = list(expired)
        for track in self.tracks:
            self.manager.expire(track, batch.timestamp_s)
            snapshot = track.snapshot()
            track.history.append(snapshot)
            snapshots.append(snapshot)
        self.tracks = [t for t in self.tracks if t.status != TrackStatus.DELETED]
        self.last_timestamp_s = batch.timestamp_s
        return tuple(snapshots)
    ###################################################################################################
