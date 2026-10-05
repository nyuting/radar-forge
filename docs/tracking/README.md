# The tracking package

`radar_forge.core.tracking` turns a stream of radar measurements into tracks. A **track** is
the tracker's running estimate of one target: its state (for example its position and velocity),
how uncertain that estimate is, and whether the tracker trusts it yet. This page says what is in
the package, the order a scan runs in, how to try it, and what it can't do yet.

## What's in the package

The package holds two trackers for now:

- **`kalman.py`** is the linear Kalman tracker that scenario 003's pipeline uses
  (`pipelines/tracking.py`). The package re-exports its names, so
  `from radar_forge.core.tracking import TrackManager` gives you `kalman`'s.
- **The other modules** make up the UKF tracker. It is multi-sensor, and it is built from
  small, swappable pieces.

A later change moves the pipeline onto the UKF tracker and deletes `kalman.py`. Until then,
three names exist twice: `Track`, `TrackManager` and `TrackStatus`. Import the UKF tracker's
versions from their own modules:

```python
from radar_forge.core.tracking.lifecycle import TrackManager
from radar_forge.core.tracking.tracks import Track, TrackStatus
```

| Module | What it holds |
| :-- | :-- |
| `coordinates.py` | Named coordinates (`x_m`, `xdot_mps`, …), the state layout they form, and `StateEstimate`: a mean, a covariance and a time. x is east, y is north and z is up. |
| `motion.py` | Motion models, which predict a state forward in time: `CartesianMotion` (constant velocity, CV, or constant acceleration, CA, per axis) and `RadialMotion` (range and range rate). |
| `measurement_models.py` | `Measurement` (a value, its covariance, a time, a sensor and a model), `MeasurementBatch` (one scan from one sensor), `SensorRoute`, and the models that predict a measurement from a state. |
| `estimation.py` | The `Estimator` interface, and the innovation statistics used for gating. |
| `ukf.py` | The unscented Kalman filter (UKF), the estimator every track uses. |
| `association.py` | The chi-square gate, and nearest-neighbour (NN) and global nearest-neighbour (GNN) assignment. |
| `initiation.py` | `DirectStateInitiator`, which starts a track from a measurement no track took. |
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
It does these steps in order:

1. **Predict.** Move every track forward to the scan time (`UKF.predict_to`). A track that has
   gone longer than `max_coast_time_s` without a measurement is deleted.
2. **Gate and score.** For each track and measurement, work out the **innovation**: the
   difference between the measurement and what the track expected to see
   (`UKF.innovation_statistics`). From it comes the **NIS**, the normalised innovation squared,
   which measures how surprising the measurement is. Pairs that fail the chi-square gate are
   dropped (`ChiSquareGate.accepts`).
3. **Assign.** Pair measurements with tracks, in two stages. Confirmed tracks pick first, from
   all the measurements. Tentative tracks then pick from what is left (`GlobalNearestNeighbour`).
4. **Update.** Correct each track that got a measurement (`UKF.update`). It reuses the
   innovation from step 2.
5. **Start.** Begin a tentative track from each measurement that no track took, unless it falls
   inside a confirmed track's gate (`DirectStateInitiator`).
6. **Confirm or delete.** Apply the M-of-N rule (`TrackManager.record`):
   - A tentative track is confirmed once it has M hits within its first N scans.
   - It is deleted as soon as M hits can no longer be reached.
   - A confirmed track survives a few misses, but is deleted after `n_delete_misses` misses in
     a row.

Steps 3 and 5 stop a new, unproven track from taking a confirmed track's measurement. A
tentative track is young, so its uncertainty is large, and by NIS it matches almost anything.
`tracker.py`'s module docstring explains this in more detail.

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
        print(time_s, snapshot.track_id, snapshot.status.value, snapshot.state.round(1))
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
may stray from it. For a CV axis it is a density `q`, in m²/s³. A useful starting point is to
choose `q` so that the velocity can change by about the largest expected acceleration
`a_max` in one scan of length `T`:

```text
q ≈ a_max² · T
```

For example, a target that may accelerate at 4 m/s², scanned once a second, gives
q ≈ 16 m²/s³. If `q` is too small, the track lags a manoeuvring target until its measurements
fall outside the gate. If `q` is too large, the track follows the noise. `motion.py`'s module
docstring gives the CA case (m²/s⁵), and shows how `q` relates to `kalman.py`'s discrete
noise model.

## Limitations

**Not yet supported here.** Main's pipeline still runs on `kalman.py` because the UKF
tracker doesn't yet have these, and the change that switches it over adds them:

- choosing per track whether a measurement uses range alone or range and range rate (the
  range-only start that scenario 003's S1 needs);
- a range-only retry when a measurement fails the gate on velocity;
- re-finding a lost target with a widened gate while keeping its track ID;
- a report of which measurement went to which track, and track fields such as `last_nis`
  and `measurement_dim`;
- range, azimuth and range-rate measurement models for a monostatic radar;
- the interacting multiple model (IMM) filter and the coordinated-turn motion model.

**Built-in assumptions.**

- Every measurement in a batch carries exactly the batch's time. A scanning radar that stamps
  each detection separately must send one batch per time.
- Batches must arrive in time order. Out-of-sequence batches are rejected, not buffered.
- Association is GNN: each scan, each track takes at most one measurement and each measurement
  goes to at most one track. In heavy clutter, probabilistic association (PDA, JPDA) or multiple
  hypotheses (MHT) do better. Neither is implemented.
- One target that gives several detections in one scan starts several tracks
  (`test_several_detections_of_one_target_give_one_confirmed_track` is marked `xfail`). The fix belongs in detection: one
  detection per target.

**Tests that come later**, each with the scenario that needs it:

- Doppler blind zones: a target crossing the line of sight has no range rate.
- GOSPA/OSPA: one score for a multi-target scenario.
- Track purity and fragmentation rates.
- JPDA or MHT against GNN, scored by the number of track swaps.
