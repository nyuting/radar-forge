"""The tracker: one scan of measurements in, the current tracks out.

:class:`Tracker` runs the standard scan loop. For each batch of measurements
from one sensor at one time it does six steps:

1. **Predict** every track forward to the batch time, and delete any track
   that has gone too long without a measurement.
2. **Gate** every track-measurement pair with a chi-square test on the NIS
   (normalised innovation squared), and score the pairs that pass.
3. **Assign** measurements to tracks in two stages. Confirmed tracks pick
   first, from all the measurements. Tentative tracks then pick from the
   measurements that are left.
4. **Update** each track that got a measurement. Each innovation, the
   difference between a measurement and what the track expected to see, is
   computed once, in step 2, and reused here.
5. **Start** a tentative track from each measurement that no track took,
   unless it lies inside a confirmed track's gate.
6. **Confirm or delete** tracks by the M-of-N rule in
   :mod:`~radar_forge.core.tracking.lifecycle`.

Steps 3 and 5 are there so that a new, unproven track cannot take a
confirmed track's measurement. A tentative track is young, so its covariance
is large, and by NIS it matches almost anything. In a single joint assignment
it can win the detection that belongs to the confirmed track. The confirmed
track then misses and is eventually deleted, and the target's identity passes
to a newer track. Giving confirmed tracks first pick, and not starting tracks
inside their gates, are the standard remedies (Blackman & Popoli).

Two builders cover the common cases: :func:`build_tracker` for one sensor and
one measurement model, and :func:`build_tracker_enu` for position
measurements in a local east-north-up frame.

References
----------
.. [1] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking
       Systems*, Artech House, 1999, ch. 6 (gating, assignment and track
       management).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from radar_forge.core.tracking._validation import TIMESTAMP_TOLERANCE_S
from radar_forge.core.tracking.association import (
    Associator,
    ChiSquareGate,
    GlobalNearestNeighbour,
    NearestNeighbour,
)
from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.estimation import Estimator, InnovationStats
from radar_forge.core.tracking.initiation import DirectStateInitiator, TrackInitiator
from radar_forge.core.tracking.lifecycle import LifecyclePolicy, TrackManager
from radar_forge.core.tracking.measurement_models import (
    CartesianPosition,
    MeasurementBatch,
    MeasurementModel,
    SensorRoute,
)
from radar_forge.core.tracking.motion import CartesianMotion, MotionKind, MotionModel
from radar_forge.core.tracking.tracks import Track, TrackSnapshot, TrackStatus
from radar_forge.core.tracking.ukf import UKF

__all__ = [
    "Tracker",
    "build_tracker",
    "build_tracker_enu",
]

# Prior standard deviation of an unobserved acceleration coordinate, m/s^2, used
# by build_tracker_enu. It is carried over unchanged from the original builder
# and is not derived from any scenario.
_SIGMA_ACCELERATION_PRIOR_MPS2 = 10.0


class Tracker:
    r"""Track many targets from batches of measurements, one scan at a time.

    This is the class Stone Soup calls ``MultiTargetTracker``. Stone Soup
    builds it from a detector, an initiator, deleters, a data associator and an
    updater. Here the initiator and deleters are the one ``manager``, the data
    associator is ``gate`` plus ``associator``, and each track's estimator does
    the predicting and updating.

    Parameters
    ----------
    measurement_models : mapping of str to MeasurementModel
        Every measurement model, by the ID that a ``Measurement`` names in its
        ``measurement_model_id``.
    sensors : mapping of str to SensorRoute
        Every sensor route, keyed by its ``route_id``. A batch's ``sensor_id``
        picks one. The route lists which measurement models that sensor may
        use, and may say which tracks the sensor can see.
    gate : ChiSquareGate
        Rejects track-measurement pairs that are too far apart.
    associator : Associator
        Chooses among the pairs that pass the gate.
    manager : TrackManager
        Starts, confirms and deletes tracks.
    cost : {"nis", "negative_log_likelihood"}, default "nis"
        How a pair that passes the gate is scored; smaller is better.

        - ``"nis"`` scores :math:`d^2 = \nu^\top S^{-1} \nu`, the squared
          distance between measurement and prediction measured in units of
          their expected spread :math:`S`, the innovation covariance.
        - ``"negative_log_likelihood"`` scores
          :math:`\tfrac12 (d^2 + \ln|S| + n_z \ln 2\pi)`. Up to a constant
          and a factor of one half, this is Blackman's generalised distance
          :math:`d^2 + \ln|S|`.

        Why the choice matters: dividing by :math:`S` makes a very uncertain
        track look like a good match for anything nearby. Take a confirmed
        track with :math:`S = 2\,\mathrm{m}^2` and a detection 3 m from it:
        :math:`d^2 = 9 / 2 = 4.5`. A tentative track one scan old still has
        :math:`S \approx 10^4\,\mathrm{m}^2`, from its 100 m/s velocity
        prior. The same detection 12 m from it gives
        :math:`d^2 = 144 / 10^4 \approx 0.01`. So NIS gives the detection to
        the tentative track. The :math:`\ln|S|` term charges a track for
        being uncertain: :math:`4.5 + \ln 2 \approx 5.2` against
        :math:`0.01 + \ln 10^4 \approx 9.2`. So the generalised distance
        gives it to the confirmed track. The two-stage assignment in
        :meth:`process` already stops a tentative track winning here. But the
        same effect can still favour an uncertain confirmed track over a
        well-placed one. An example is a track that has coasted for several
        scans.

    Attributes
    ----------
    gate, associator, manager, cost
        As given.
    tracks : list of Track
        The live tracks, in order of creation. Deleted tracks are removed.
    last_timestamp_s : float or None
        Time of the last processed batch, seconds. None before the first.

    Raises
    ------
    ValueError
        If ``cost`` is not one of the two names, a key of ``sensors`` is not
        its route's ``route_id``, or a sensor route is invalid (see
        :meth:`add_sensor`).

    Notes
    -----
    Each track owns its own estimator, which should start from a prior that can
    be justified for the target. The tracker is single-threaded. If a
    user-supplied component raises in the middle of a scan, the tracks already
    changed in that scan are not rolled back.

    References
    ----------
    .. [1] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking
           Systems*, Artech House, 1999, ch. 6.

    Examples
    --------
    A one-axis tracker built by hand. :func:`build_tracker` does the same in
    one call.

    >>> import numpy as np
    >>> from radar_forge.core.tracking import (
    ...     UKF, CartesianMotion, CartesianPosition, ChiSquareGate,
    ...     DirectStateInitiator, GlobalNearestNeighbour, LifecyclePolicy,
    ...     SensorRoute, StateEstimate)
    >>> from radar_forge.core.tracking.lifecycle import TrackManager
    >>> motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=(36.0, -78.9, 60.0))
    >>> model = CartesianPosition(motion.state_layout, ("x_m",))
    >>> prior = StateEstimate(np.zeros(2), np.diag([1.0, 100.0**2]), 0.0, motion.state_layout)
    >>> initiator = DirectStateInitiator(lambda state: UKF(state, motion), prior)
    >>> tracker = Tracker(
    ...     {"position": model},
    ...     {"radar": SensorRoute("radar", ("position",))},
    ...     ChiSquareGate(0.997),
    ...     GlobalNearestNeighbour(),
    ...     TrackManager(initiator, LifecyclePolicy(n_confirm_hits=3, n_confirm_frames=5)),
    ... )
    >>> scan = tracker.sensors["radar"].batch(
    ...     0.0, [("position", np.array([1000.0]), np.array([[4.0]]))])
    >>> [snapshot.status.value for snapshot in tracker.process(scan)]
    ['tentative']
    """

    def __init__(
        self,
        measurement_models: Mapping[str, MeasurementModel],
        sensors: Mapping[str, SensorRoute],
        gate: ChiSquareGate,
        associator: Associator,
        manager: TrackManager,
        cost: Literal["nis", "negative_log_likelihood"] = "nis",
    ) -> None:
        if cost not in ("nis", "negative_log_likelihood"):
            msg = f"cost must be 'nis' or 'negative_log_likelihood'; got {cost!r}."
            raise ValueError(msg)
        if any(key != route.route_id for key, route in sensors.items()):
            mismatched = {
                key: route.route_id for key, route in sensors.items() if key != route.route_id
            }
            msg = f"each sensor key must be its route's route_id; got key: route_id {mismatched}."
            raise ValueError(msg)
        self._measurement_models = dict(measurement_models)
        self._sensors: dict[str, SensorRoute] = {}
        # Routes are few and each is checked against the model registry on its
        # own, so a plain loop is the clearest form.
        for route in sensors.values():
            self.add_sensor(route)
        self.gate = gate
        self.associator = associator
        self.manager = manager
        self.cost = cost
        self.tracks: list[Track] = []
        self.last_timestamp_s: float | None = None

    @property
    def measurement_models(self) -> Mapping[str, MeasurementModel]:
        """The registered measurement models, by ID. Read-only."""
        return MappingProxyType(self._measurement_models)

    @property
    def sensors(self) -> Mapping[str, SensorRoute]:
        """The registered sensor routes, by ``route_id``. Read-only.

        Use :meth:`add_sensor` to register another, so that it is checked.
        """
        return MappingProxyType(self._sensors)

    def add_sensor(self, route: SensorRoute) -> None:
        """Register one more sensor route, after checking it.

        Parameters
        ----------
        route : SensorRoute
            The route to add, registered under ``route.route_id``.

        Raises
        ------
        ValueError
            If ``route.route_id`` is already registered, the route lists no
            measurement models, or it lists a model that is not registered.
        """
        if route.route_id in self._sensors:
            msg = f"sensor {route.route_id!r} is already registered."
            raise ValueError(msg)
        if not route.measurement_model_ids or not set(route.measurement_model_ids) <= set(
            self._measurement_models
        ):
            unknown = [
                mid for mid in route.measurement_model_ids if mid not in self._measurement_models
            ]
            msg = (
                f"sensor {route.route_id!r} must list at least one registered measurement "
                f"model; got {route.measurement_model_ids}, unregistered {unknown}."
            )
            raise ValueError(msg)
        self._sensors[route.route_id] = route

    def seed(self, estimator: Estimator, sensor_id: str | None = None) -> TrackSnapshot:
        """Add a track whose filter the caller has already set up.

        Use this to start a track from outside knowledge, such as a cue from
        another system, rather than from a measurement.

        Parameters
        ----------
        estimator : Estimator
            The new track's own filter, holding its starting estimate. Its time
            must equal the tracker's current time; before the first batch, it
            must equal the time of any track already seeded.
        sensor_id : str or None, optional
            The sensor to record as the track's first source. None records none.

        Returns
        -------
        TrackSnapshot
            A frozen copy of the new track. The live track is added to
            :attr:`tracks`.

        Raises
        ------
        ValueError
            If another track already uses ``estimator``, or its time does not
            match, within one microsecond, the tracker's or the other tracks'.
        """
        if any(track.estimator is estimator for track in self.tracks):
            msg = "each track must own an independent estimator; this one is already in use."
            raise ValueError(msg)
        if self.last_timestamp_s is not None and not self._same_time(
            estimator.state.timestamp_s, self.last_timestamp_s
        ):
            msg = (
                f"seed timestamp_s must equal the tracker's current time "
                f"{self.last_timestamp_s}; got {estimator.state.timestamp_s}."
            )
            raise ValueError(msg)
        if self.tracks and not self._same_time(
            estimator.state.timestamp_s, self.tracks[0].estimator.state.timestamp_s
        ):
            msg = (
                f"seed tracks must share one timestamp_s "
                f"{self.tracks[0].estimator.state.timestamp_s}; got {estimator.state.timestamp_s}."
            )
            raise ValueError(msg)
        track = self.manager.seed(estimator, sensor_id)
        self.tracks.append(track)
        return track.snapshot()

    def process(self, batch: MeasurementBatch) -> tuple[TrackSnapshot, ...]:
        """Run one scan: predict, gate, assign, update, start and delete tracks.

        The six steps are listed in the module docstring. Association runs in
        two stages: confirmed tracks are assigned first, from all the
        measurements, and tentative tracks then share what is left. A
        measurement that no track took starts a new tentative track only if it
        lies outside every confirmed track's gate. Both rules stop a young
        track with a large covariance from taking a confirmed track's
        measurement, following Blackman & Popoli's track management.

        Parameters
        ----------
        batch : MeasurementBatch
            The measurements from one sensor at one time. It may be empty: an
            empty scan is a scan in which nothing was detected, so every track
            the sensor can see records a miss.

        Returns
        -------
        tuple of TrackSnapshot
            First, the tracks deleted for going too long without a measurement;
            then every other track, in order of creation, including any just
            started. A track deleted in this scan appears once, with status
            DELETED, and never again.

        Raises
        ------
        ValueError
            If the batch is older than the last one by more than one
            microsecond ("out-of-sequence"), older than a seeded track, or from
            an unknown sensor. Also if a measurement's model is not registered
            for that sensor, or its length does not match its model's
            dimension. These checks run before any track changes.

        Notes
        -----
        A track that the sensor cannot see is left out of the scan: it is
        predicted but records neither a hit nor a miss. A track is out of
        sight when no model on the route works in the track's state layout, or
        when the route's ``observable`` test says so.

        Limitations:

        - Every measurement in a batch carries the batch time. A scanning
          radar that stamps each detection separately must group them first.
        - A batch older than the last one is rejected, not used to correct the
          tracks (no out-of-sequence measurement handling).
        - A track starts from a single measurement. Starting it from two
          measurements close in time and space would give a better first
          velocity and fewer false tracks.
        """
        self._check_batch(batch)
        route = self._sensors[batch.sensor_id]
        models = [self._measurement_models[m.measurement_model_id] for m in batch.measurements]
        time_s = batch.timestamp_s
        # Snap a time within the tolerance of the last scan onto it, so that no
        # filter is asked to predict backwards by a round-off error.
        if self.last_timestamp_s is not None and self._same_time(time_s, self.last_timestamp_s):
            time_s = self.last_timestamp_s

        reported: list[TrackSnapshot] = []
        # Each track is a Python object with its own filter state, so there is
        # no array of tracks to vectorise over.
        for track in self.tracks:
            track.estimator.predict_to(max(time_s, track.estimator.state.timestamp_s))
            self.manager.expire(track, time_s)
            if track.status == TrackStatus.DELETED:
                reported.append(track.snapshot())
        self.tracks = [track for track in self.tracks if track.status != TrackStatus.DELETED]

        visible = [track for track in self.tracks if self._can_see(route, track)]
        costs, innovations = self._score_pairs(visible, batch, models)
        confirmed = [i for i, track in enumerate(visible) if track.status == TrackStatus.CONFIRMED]
        tentative = [i for i, track in enumerate(visible) if track.status == TrackStatus.TENTATIVE]
        matches = self._assign(costs, confirmed)
        taken = {j for _, j in matches}
        free = [j for j in range(costs.shape[1]) if j not in taken]
        matches += self._assign(costs, tentative, free)

        hit = dict(matches)
        # Each update changes one track's own filter, so tracks are updated one
        # at a time.
        for i, track in enumerate(visible):
            j = hit.get(i)
            if j is None:
                self.manager.record(track)
                continue
            measurement = batch.measurements[j]
            track.estimator.update(measurement, models[j], innovation=innovations[i, j])
            self.manager.record(track, measurement)

        claimed = set(hit.values())
        inside_confirmed = np.asarray(np.isfinite(costs[confirmed]).any(axis=0), dtype=np.bool_)
        # Each birth builds a new filter object from one measurement, so births
        # are made one at a time.
        for j, (measurement, model) in enumerate(zip(batch.measurements, models, strict=True)):
            if j in claimed or inside_confirmed[j]:
                continue
            new_track = self.manager.create(measurement, model)
            if new_track is not None:
                self.tracks.append(new_track)

        # A snapshot copies one track's filter state, so it is taken per track.
        for track in self.tracks:
            snapshot = track.snapshot()
            track.history.append(snapshot)
            reported.append(snapshot)
        self.tracks = [track for track in self.tracks if track.status != TrackStatus.DELETED]
        self.last_timestamp_s = time_s
        return tuple(reported)

    def _check_batch(self, batch: MeasurementBatch) -> None:
        """Raise unless the batch can be processed; called before any track changes."""
        if self.last_timestamp_s is not None and (
            batch.timestamp_s < self.last_timestamp_s - TIMESTAMP_TOLERANCE_S
        ):
            msg = (
                f"out-of-sequence batch rejected at {batch.timestamp_s} s, after "
                f"{self.last_timestamp_s} s; buffering is not implemented."
            )
            raise ValueError(msg)
        # Each track keeps its own time, so each must be checked; there are few.
        if any(
            track.estimator.state.timestamp_s > batch.timestamp_s + TIMESTAMP_TOLERANCE_S
            for track in self.tracks
        ):
            msg = f"batch at {batch.timestamp_s} s precedes a seeded track."
            raise ValueError(msg)
        if batch.sensor_id not in self._sensors:
            msg = f"unknown sensor {batch.sensor_id!r}; register it with add_sensor."
            raise ValueError(msg)
        if any(
            m.measurement_model_id not in self._sensors[batch.sensor_id].measurement_model_ids
            for m in batch.measurements
        ):
            unregistered = [
                m.measurement_model_id
                for m in batch.measurements
                if m.measurement_model_id
                not in self._sensors[batch.sensor_id].measurement_model_ids
            ]
            msg = f"unregistered models {unregistered} for sensor {batch.sensor_id!r}."
            raise ValueError(msg)
        # The check above makes every model ID a registered one, so these lookups succeed.
        if any(
            len(m.value)
            != self._measurement_models[m.measurement_model_id].measurement_layout.dimension
            for m in batch.measurements
        ):
            mismatched = [
                (m.measurement_model_id, len(m.value))
                for m in batch.measurements
                if len(m.value)
                != self._measurement_models[m.measurement_model_id].measurement_layout.dimension
            ]
            msg = (
                "measurement dimensions do not match their models; got (model, dimension) "
                f"{mismatched}."
            )
            raise ValueError(msg)

    def _can_see(self, route: SensorRoute, track: Track) -> bool:
        """Return whether this scan's sensor could have detected the track."""
        layout = track.estimator.state.state_layout
        fits = any(
            self._measurement_models[mid].state_layout == layout
            for mid in route.measurement_model_ids
        )
        return fits and (route.observable is None or route.observable(track.snapshot()))

    def _score_pairs(
        self,
        tracks: Sequence[Track],
        batch: MeasurementBatch,
        models: Sequence[MeasurementModel],
    ) -> tuple[NDArray[np.float64], dict[tuple[int, int], InnovationStats]]:
        """Gate and score every track-measurement pair.

        Returns two things. The first is the cost matrix, shape
        ``(n_tracks, n_measurements)``, with ``+inf`` for a pair that is gated
        out or whose layouts differ. The second is the innovation of each pair
        that passed, keyed by ``(track, measurement)``.
        """
        costs = np.full((len(tracks), len(models)), np.inf)
        innovations: dict[tuple[int, int], InnovationStats] = {}
        # Each innovation runs one track's own filter through one measurement's
        # model, and both can differ from pair to pair, so pairs are scored one
        # at a time.
        for i, track in enumerate(tracks):
            for j, (measurement, model) in enumerate(zip(batch.measurements, models, strict=True)):
                if track.estimator.state.state_layout != model.state_layout:
                    continue
                stats = track.estimator.innovation_statistics(measurement, model)
                if self.gate.accepts(stats):
                    costs[i, j] = stats.nis if self.cost == "nis" else -stats.log_likelihood
                    innovations[i, j] = stats
        return costs, innovations

    def _assign(
        self,
        costs: NDArray[np.float64],
        rows: Sequence[int],
        cols: Sequence[int] | None = None,
    ) -> list[tuple[int, int]]:
        """Run the associator on some rows and columns of the cost matrix.

        Returns the matches as ``(row, column)`` indices of the full matrix.
        ``cols`` of None means every column.
        """
        row_index = np.asarray(rows, dtype=np.intp)
        col_index = np.arange(costs.shape[1]) if cols is None else np.asarray(cols, dtype=np.intp)
        result = self.associator.associate(costs[np.ix_(row_index, col_index)])
        return [(int(row_index[r]), int(col_index[c])) for r, c in result.matches]

    @staticmethod
    def _same_time(a_s: float, b_s: float) -> bool:
        """Return whether two timestamps are equal within the tolerance."""
        return abs(a_s - b_s) <= TIMESTAMP_TOLERANCE_S


def _check_filter_settings(
    motion: MotionModel, prior: StateEstimate, alpha: float, beta: float, kappa: float
) -> None:
    """Build one UKF and discard it, so bad sigma-point settings fail now.

    The settings that are valid depend on the state dimension. Without this
    check they would fail at the first track birth, in the middle of a scan.
    """
    UKF(prior, motion, alpha, beta, kappa)


def build_tracker(
    motion: MotionModel,
    observation: MeasurementModel,
    prior: StateEstimate,
    *,
    policy: LifecyclePolicy | None = None,
    gate_probability: float = 0.997,
    association: Literal["GNN", "NN"] = "GNN",
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    initiator: TrackInitiator | None = None,
    sensor_id: str = "sensor",
    model_id: str = "measurement",
) -> Tracker:
    r"""Build a UKF tracker for one sensor and one measurement model.

    Parameters
    ----------
    motion : MotionModel
        Motion model; its ``state_layout`` must match ``observation`` and ``prior``.
    observation : MeasurementModel
        Measurement model, registered under ``model_id``.
    prior : StateEstimate
        Prior for the coordinates a measurement does not observe: mean
        ``(n_state,)`` and covariance ``(n_state, n_state)`` in the layout's
        units.
    policy : LifecyclePolicy or None, optional
        Confirmation, deletion, coast and history settings; None uses the defaults.
    gate_probability : float, optional
        Chi-square gate probability :math:`P_G`, strictly between 0 and 1.
    association : {"GNN", "NN"}, optional
        Global nearest neighbour or greedy nearest neighbour. Both score pairs
        by NIS.
    alpha, beta, kappa : float, optional
        The scaled sigma-point parameters of the UKF: the spread :math:`\alpha`,
        the prior-distribution correction :math:`\beta` (2 suits a Gaussian) and
        the secondary scaling :math:`\kappa`. The defaults, 1, 2 and 0, match
        :class:`UKF`'s, whose Notes explain the choice.
    initiator : TrackInitiator or None, optional
        Track-birth rule; None uses :class:`DirectStateInitiator` with ``prior``.
    sensor_id, model_id : str, optional
        Route and measurement-model IDs the measurements must carry.

    Returns
    -------
    Tracker
        A tracker with no tracks, one :class:`SensorRoute` and one measurement model.

    Raises
    ------
    ValueError
        If the three layouts differ, ``association`` is not GNN or NN, or the
        UKF or gate parameters are invalid. Also, with the default initiator and
        a :class:`CartesianPosition` observation, if the prior correlates the
        measured coordinates with the unmeasured ones.

    References
    ----------
    .. [1] E. A. Wan and R. van der Merwe, "The unscented Kalman filter for
           nonlinear estimation," *Proc. IEEE AS-SPCC Symposium*, 2000
           (the parameters alpha, beta and kappa).
    """
    if motion.state_layout != observation.state_layout or prior.state_layout != motion.state_layout:
        msg = "motion, observation and prior require the same StateLayout."
        raise ValueError(msg)
    if association not in ("GNN", "NN"):
        msg = f"association must be GNN or NN; got {association!r}."
        raise ValueError(msg)
    _check_filter_settings(motion, prior, alpha, beta, kappa)

    def factory(state: StateEstimate) -> UKF:
        return UKF(state, motion, alpha, beta, kappa)

    # A position model's measured names are state names, so the initiator can
    # check now that the prior does not tie them to the unmeasured coordinates.
    # Any other model cannot start a track through DirectStateInitiator anyway.
    measured_names = (
        observation.measurement_layout.names if isinstance(observation, CartesianPosition) else None
    )
    birth = initiator or DirectStateInitiator(factory, prior, measured_names=measured_names)
    return Tracker(
        {model_id: observation},
        {sensor_id: SensorRoute(sensor_id, (model_id,))},
        ChiSquareGate(gate_probability),
        GlobalNearestNeighbour() if association == "GNN" else NearestNeighbour(),
        TrackManager(birth, policy),
    )


def build_tracker_enu(
    axes: Mapping[str, MotionKind],
    *,
    origin_lla_deg_m: tuple[float, float, float],
    order: tuple[str, ...] | None = None,
    acceleration_noise_density_m2ps3: float | Mapping[str, float] = 1.0,
    jerk_noise_density_m2ps5: float | Mapping[str, float] = 1.0,
    sigma_velocity_mps: float = 100.0,
    policy: LifecyclePolicy | None = None,
    gate_probability: float = 0.997,
    association: Literal["GNN", "NN"] = "GNN",
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> Tracker:
    """Build a tracker that measures x/y/z position and estimates the motion.

    x is east, y is north and z is up, in the ENU frame about ``origin_lla_deg_m``.

    Parameters
    ----------
    axes : mapping of str to {"CV", "CA"}
        Axes to track, each mapped to its motion model: ``"CV"`` (constant
        velocity) or ``"CA"`` (constant acceleration). See :class:`CartesianMotion`.
    origin_lla_deg_m : tuple of float
        ENU origin: latitude and longitude in degrees, altitude in metres.
    order : tuple of str or None, optional
        State coordinate order; None groups the coordinates by axis.
    acceleration_noise_density_m2ps3 : float or mapping of str to float, optional
        White-acceleration density for the CV axes, m²/s³.
    jerk_noise_density_m2ps5 : float or mapping of str to float, optional
        White-jerk density for the CA axes, m²/s⁵.
    sigma_velocity_mps : float, optional
        Prior standard deviation of each unobserved velocity, m/s.
    policy, gate_probability, association, alpha, beta, kappa
        As for :func:`build_tracker`.

    Returns
    -------
    Tracker
        A tracker whose one route measures the tracked axes' positions, in metres.

    Raises
    ------
    ValueError
        If ``sigma_velocity_mps`` is not finite and positive, or as for
        :class:`CartesianMotion` and :func:`build_tracker`.

    Notes
    -----
    A new track takes its position from its first measurement. Its velocity
    starts at 0 with standard deviation ``sigma_velocity_mps``. On CA axes its
    acceleration starts at 0 with a fixed standard deviation of 10 m/s². A
    position measurement says nothing about an axis left out of ``axes``, so
    that axis is not tracked at all.

    References
    ----------
    .. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with
           Applications to Tracking and Navigation*, Wiley, 2001, §6.2
           (continuous white-noise acceleration and Wiener-process
           acceleration models).
    """
    if not np.isfinite(sigma_velocity_mps) or sigma_velocity_mps <= 0:
        msg = f"sigma_velocity_mps must be finite and positive; got {sigma_velocity_mps!r}."
        raise ValueError(msg)

    motion = CartesianMotion(
        axes,
        origin_lla_deg_m=origin_lla_deg_m,
        order=order,
        acceleration_noise_density_m2ps3=acceleration_noise_density_m2ps3,
        jerk_noise_density_m2ps5=jerk_noise_density_m2ps5,
    )
    observation = CartesianPosition(
        motion.state_layout, tuple(f"{a}_m" for a in "xyz" if a in axes)
    )
    # The position variance of 1.0 is a placeholder: DirectStateInitiator
    # replaces it with the measurement covariance at birth.
    variance = [
        sigma_velocity_mps**2
        if c.unit == "m/s"
        else _SIGMA_ACCELERATION_PRIOR_MPS2**2
        if c.unit == "m/s^2"
        else 1.0
        for c in motion.state_layout.coordinates
    ]
    prior = StateEstimate(
        np.zeros(motion.state_layout.dimension), np.diag(variance), 0, motion.state_layout
    )
    return build_tracker(
        motion,
        observation,
        prior,
        policy=policy,
        gate_probability=gate_probability,
        association=association,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
    )
