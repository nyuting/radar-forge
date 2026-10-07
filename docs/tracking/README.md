# The tracking package

`radar_forge.core.tracking` turns a stream of radar measurements into tracks. A **track** is
the tracker's running estimate of one target: its state (for example its position and velocity),
how uncertain that estimate is, and whether the tracker trusts it yet. This page says what is in
the package, the order a scan runs in, how to try it, and what it can't do yet.

## What's in the package

The package runs two tracker loops for now. Both keep their tracks as the same `Track`
(`tracks.py`), with the same `TrackStatus`, and both confirm and delete them with the same
`TrackManager` (`lifecycle.py`). Only the filter inside each track differs.

- **`KalmanTracker`** (`kalman.py`) is the linear Kalman tracker. Each of its tracks carries a
  `KalmanFilter`. Scenario 003's own runs use it: `estimator = "kalman"` in the scenario's
  `[tracking]` table, read by `pipelines/tracking.py`.
- **`Tracker`** (`tracker.py`) is the UKF tracker. Each of its tracks carries an `Estimator`,
  the UKF. It is built from small, swappable pieces, and it is multi-sensor: one `Tracker`
  should handle every sensor in a scenario, each registered with `Tracker.add_sensor`, so
  that each target has one track whichever sensor sees it. The scenario-003 runs over
  scenario 001's S1, S2 and S3 use it: `estimator = "ukf"`.

A later change moves scenario 003's own runs onto `Tracker` too, and deletes `KalmanTracker`,
once `Tracker` has what the Limitations below list.

| Module | What it holds |
| :-- | :-- |
| `coordinates.py` | Named coordinates (`x_m`, `xdot_mps`, …), the state layout they form, and `StateEstimate`: a mean, a covariance and a time. x is east, y is north and z is up, so an azimuth is measured clockwise from north. |
| `motion.py` | Motion models, which predict a state forward in time: `CartesianMotion` (constant velocity, CV, or constant acceleration, CA, per axis) and `RadialMotion` (range and range rate). |
| `measurement_models.py` | `Measurement` (a value, its covariance, a time, a sensor and a model), `MeasurementBatch` (one scan from one sensor), `SensorRoute` (a sensor, and the models its measurements may use), and the models that predict a measurement from a state. |
| `estimation.py` | The `Estimator` interface, and the innovation statistics used for gating. |
| `ukf.py` | The unscented Kalman filter (UKF), the estimator every track uses. |
| `association.py` | The chi-square gate, and nearest-neighbour (NN) and global nearest-neighbour (GNN) assignment. |
| `initiation.py` | `DirectStateInitiator`, which starts a track from a measurement no track took. |
| `kalman.py` | `KalmanTracker`, scenario 003's linear Kalman tracker, and the filter functions it uses. |
| `tracks.py` | A live `Track`, its status, and the read-only `TrackSnapshot` handed to callers. |
| `lifecycle.py` | The M-of-N rule that confirms and deletes tracks, and the `TrackManager` that applies it. |
| `tracker.py` | `Tracker`, which runs one scan at a time, and two builders for common set-ups. |
| `_validation.py` | Private input checks shared by the modules above. |

The pieces follow [Stone Soup](https://stonesoup.readthedocs.io)'s split: motion model,
measurement model, estimator, gate, associator, initiator and track manager. So a student
who moves on to Stone Soup will recognise them. Where the two differ, the module docstring
says so.

## Reading the tracker: one scan, step by step

`Tracker.process(batch)` takes one scan, meaning every measurement from one sensor at one time.
It does these five steps in order:

1. **Predict.** Move every track forward to the scan time (`UKF.predict_to`). A track that has
   gone longer than `max_coast_time_s` without a measurement is deleted.
2. **Gate and score.** For each track and measurement, work out the **innovation**: the
   difference between the measurement and what the track expected to see
   (`UKF.innovation_statistics`). From it comes the **NIS**, the normalised innovation squared,
   which measures how surprising the measurement is; its square root is the *Mahalanobis
   distance*. A pair fails the *chi-square gate* (`ChiSquareGate.accepts`) when its NIS is
   larger than a true measurement's would be, at the gate probability, and is dropped.
3. **Assign.** Pair measurements with tracks, in two stages. Confirmed tracks pick first, from
   all the measurements. Tentative tracks then pick from what is left (`GlobalNearestNeighbour`).
4. **Update, and confirm or delete.** Correct each track that got a measurement
   (`UKF.update`), reusing the innovation from step 2. Then count a hit or a miss for every
   track the sensor can see (`TrackManager.record_hit`, `TrackManager.record_miss`), and apply
   the M-of-N rule as it is counted:
   - A tentative track is confirmed once it has M hits within its first N scans.
   - It is deleted as soon as M hits can no longer be reached.
   - A confirmed track that misses *coasts*: it is predicted forward with nothing to update
     it, and its status is `"coasting"` until its next hit. It is deleted after
     `n_delete_misses` misses in a row.
5. **Start.** Begin a tentative track from each measurement that no track took, unless it falls
   inside a confirmed track's gate (`DirectStateInitiator`).

`lifecycle.py`'s module docstring works an M-of-N example through scan by scan. Steps 3 and 5
stop a new, unproven track from taking a confirmed track's measurement; `tracker.py`'s module
docstring explains why.

## Try it

This tracks one target moving east at 20 m/s, from position measurements with a 2 m standard
deviation:

```python
import numpy as np

from radar_forge.core.tracking import build_tracker_enu

# Track east (x) at constant velocity, from position-only measurements.
tracker = build_tracker_enu({"x": "CV"}, origin_lla_deg_m=(36.0, -78.9, 60.0))
sensor = next(iter(tracker.sensors.values()))
model_id = sensor.measurement_model_ids[0]
measurement_noise_m2 = np.array([[4.0]])  # a 2 m standard deviation

for time_s in range(6):
    position_m = np.array([1000.0 + 20.0 * time_s])  # moving east at 20 m/s
    scan = sensor.batch(time_s, [(model_id, position_m, measurement_noise_m2)])
    for snapshot in tracker.process(scan):
        print(time_s, snapshot.track_id, snapshot.status, snapshot.state.round(1))
```

It prints one line per scan: the time, the track ID, its status, and its state
`[position_m, velocity_mps]`. The default rule is 3-of-5, so the track is confirmed on its third
hit:

```text
0 1 tentative [1000.    0.]
1 1 tentative [1020.   20.]
2 1 confirmed [1040.   20.]
...
```

## Choosing the process noise

Real targets don't move at exactly constant velocity. The **process noise** says how far they
may stray from it. You give it as two physical numbers:

- `sigma_acceleration_mps2`, σ_a: the standard deviation of the target's acceleration. If you
  know a largest acceleration `a_max` and treat it as a 3σ bound, σ_a = a_max / 3.
- `acceleration_correlation_time_s`, τ: roughly how long one acceleration lasts.

The motion model turns them into a noise density, and builds Q from the actual time step at
every prediction:

```text
q = 2 · σ_a² · τ        (m²/s³)
Q(T) = q · [[T³/3, T²/2], [T²/2, T]]
```

The defaults, σ_a = 4 m/s² and τ = 1 s (q = 32 m²/s³), are measured from the truth of
scenario 001's 120 s window. The acceleration there has a standard deviation of 3.3 m/s² on the
east and north axes, and 4.0 m/s² along the line of sight. Its correlation falls to between
0.44 and 0.51 after 1 s, and changes sign by 2 s. Run through `RadialMotion` and the UKF on that
truth, with scenario 003's measurement noise (21.6 m in range, 0.017 m/s in range rate), 0.35 %
of the true measurements fall outside a 99.7 % gate, close to the 0.3 % the gate is designed
for. The old default, q = 1 m²/s³, left 31 % outside.

If σ_a is too small, the track lags a manoeuvring target until its measurements fall outside
the gate. If it is too large, the track follows the noise.

A common rule of thumb is q ≈ a_max² · T for a scan of length T. This repo doesn't use it,
because the scan length stands in for τ there: the same q then means a different target at a
different update rate, and with two sensors there is no single T. `motion.py`'s module
docstring explains this, gives the CA case (`sigma_jerk_mps3`), and compares the model with
`kalman.py`'s discrete noise model.

## Limitations

**Not yet supported here.** Scenario 003's own runs still use `kalman.py` because the UKF
tracker doesn't yet have these, and the change that switches them over adds them:

- choosing per track whether a measurement uses range alone or range and range rate (the
  range-only start that scenario 003's S1 needs);
- a range-only retry when a measurement fails the gate on velocity;
- re-finding a lost target with a widened gate while keeping its track ID;
- a per-track `measurement_dim`, as `KalmanFilter` records. Which measurement each track
  took in the last scan, and its NIS, are `Tracker.last_associations` and `Tracker.last_nis`;
- range, azimuth and range-rate measurement models for a monostatic radar;
- the interacting multiple model (IMM) filter and the coordinated-turn motion model;
- Singer's motion model, which carries the acceleration in the state and is exact at every
  time step. Its white-noise limit sets the process noise now;
- an initiator for the bistatic model. One bistatic measurement (a path length and a path
  rate) cannot fix a target's position, and `DirectStateInitiator` returns None for it. So a
  tracker whose only model is `BistaticRangeDopplerModel` never starts a track by itself.
  Add its tracks with `Tracker.seed`;
- a forward-scatter (FSR) measurement model, for a target near the bistatic baseline, and an
  initiator for it. PR #1 planned both; how to start an FSR track is still to be decided;
- an initiator that picks a different rule for each measurement model (PR #1's
  `RoutedInitiator`), and one that never starts a track from a detection (PR #1's
  `NoInitiation`), for a sensor whose tracks must all come from `Tracker.seed`.

**Built-in assumptions.**

- Every measurement in a batch carries exactly the batch's time. A scanning radar that stamps
  each detection separately must send one batch per time.
- Batches must arrive in time order. Out-of-sequence batches are rejected, not buffered.
- Association is GNN by default (NN is also available). Either way, each scan, each track
  takes at most one measurement and each measurement goes to at most one track. In heavy
  clutter, probabilistic association (PDA, JPDA) or multiple hypotheses (MHT) do better.
  Neither is implemented.
- One target that gives several detections in one scan starts several tracks
  (`test_several_detections_of_one_target_give_one_confirmed_track` is marked `xfail`). The
  fix belongs in detection: one detection per target.
- Expect track fragmentation with the default settings: one target's track breaking into
  several over a run. PR #1's runner saw it on its source trajectory with these defaults. The
  runner comes back in a later PR, which measures it again.

**Tests that come later**, each with the scenario that needs it:

- Doppler blind zones: a target crossing the line of sight has no range rate.
- GOSPA/OSPA: one score for a multi-target scenario.
- Track purity and fragmentation rates.
- JPDA or MHT against GNN, scored by the number of track swaps.
