# 6. Tracking

`radar_forge.core.tracking` turns a stream of radar measurements into tracks. A **track** is
the tracker's running estimate of one target: its state (for example its position and velocity),
how uncertain that estimate is, and whether the tracker trusts it yet. It is the last stage of
[the chain](README.md#the-chain): it takes the detections from [page 5](05-detection.md), scan
after scan. This page says what is in the package, the equations it runs, the order a scan runs
in, how to try it, and what it can't do yet.

## What this stage does

One detection says where a target was at one moment, with some error. A track combines many of
them: it predicts where the target should be next, compares that with what the radar reports,
and corrects itself. Doing that well means answering three questions every scan. Which
detection belongs to which track? How much should a new detection move the estimate? And when
is a track real enough to report, or gone for good?

### What's in the package

The package runs two tracker loops for now. Both keep their tracks as the same `Track`
(`tracks.py`), with the same `TrackStatus`, and both confirm and delete them with the same
`TrackManager` (`lifecycle.py`). Only the filter inside each track differs.

- **`KalmanTracker`** (`kalman.py`) is the older linear Kalman tracker. Each of its tracks
  carries a `KalmanFilter`. Nothing in the pipeline uses it any more, and step 7 of
  `spec/tracker-001.md` §13 deletes it.
- **`Tracker`** (`tracker.py`) is the scan-loop tracker. Each of its tracks carries an
  `Estimator`: by default a `GaussianFilter` with the unscented linearisation, which is the UKF.
  It is built from small, swappable pieces, and it is multi-sensor: one `Tracker`
  should handle every sensor in a scenario, each registered with `Tracker.add_sensor`, so
  that each target has one track whichever sensor sees it. Every scenario-003 run uses it:
  `estimator = "ukf"` runs a UKF, and `estimator = "kalman"` runs a Kalman filter
  (`GaussianFilter` with `TaylorLinearisation`) with scenario 003's bootstrap as its model
  selector.

| Module | What it holds |
| :-- | :-- |
| `coordinates.py` | Named coordinates (`x_m`, `xdot_mps`, …), the state layout they form, and `StateEstimate`: a mean, a covariance and a time. x is east, y is north and z is up, so an azimuth is measured clockwise from north. |
| `motion.py` | Motion models, which predict a state forward in time: `CartesianMotion` (constant velocity, CV, or constant acceleration, CA, per axis) and `RadialMotion` (range and range rate). |
| `measurement_models.py` | `Measurement` (a value, its covariance, a time, a sensor and a model), `MeasurementBatch` (one scan from one sensor), `SensorRegistration` (a sensor, and the models its measurements may use), and the models that predict a measurement from a state: `CartesianPosition` (coordinates reported directly), `RangeAzimuthElevationRangeRate` (a monostatic radar's range, azimuth, elevation and range rate) and `BistaticRangeDopplerModel`. `marginal` keeps some of a model's components, so that a track can use part of a measurement, such as its range alone. |
| `estimation.py` | The `Estimator` interface, the innovation statistics used for gating, and `GaussianFilter`: the Kalman filter (KF), the extended Kalman filter (EKF) and the unscented Kalman filter (UKF) in one predict and one update. |
| `linearisation.py` | How `GaussianFilter` carries a mean and a covariance through a model, the one thing that differs between its filters: `TaylorLinearisation` (the KF on a linear model, the EKF otherwise) and `UnscentedLinearisation` (the UKF). |
| `association.py` | The chi-square gate, and nearest-neighbour (NN) and global nearest-neighbour (GNN) assignment. |
| `initiation.py` | Initiators, which start a track from a measurement no track took: `DirectStateInitiator` copies reported coordinates into a prior, and `DebiasedSphericalInitiator` converts a radar's range and angles to a position without the naive conversion's bias. |
| `kalman.py` | `KalmanTracker`, scenario 003's linear Kalman tracker, and the filter functions it uses. |
| `tracks.py` | A live `Track`, its status, and the read-only `TrackSnapshot` handed to callers. |
| `lifecycle.py` | The M-of-N rule that confirms and deletes tracks, and the `TrackManager` that applies it. |
| `tracker.py` | `Tracker`, which runs one scan at a time, and two builders for common set-ups. |
| `_validation.py` | Private input checks shared by the modules above. |

The pieces follow [Stone Soup](https://stonesoup.readthedocs.io)'s split: motion model,
measurement model, estimator, gate, associator, initiator and track manager. So a student
who moves on to Stone Soup will recognise them. Where the two differ, the module docstring
says so.

## Quantities and equations

The filter is written once, as a Gaussian filter, for the KF, EKF and UKF alike. The worked
values are for scenario 003's UKF run on the S1 waveform
(`scenarios/scenario_003_ukf_fmcw_low_prf.toml`), whose state is range and range rate and whose
scans come once a second.

| Quantity | Symbol | Equation | In the code | Scenario 003, UKF on S1 |
| :-- | :-- | :-- | :-- | --: |
| State and its covariance | $\mathbf{x}$, $P$ | mean and covariance of the estimate | `StateEstimate.mean`, `.covariance` | $[R, \dot R]$ |
| Transition over a step $T$ | $F$ | $R \to R - T\dot R$, $\dot R \to \dot R$ (matrix below) | `RadialMotion.transition` | $T$ = 1 s |
| Process-noise density | $q$ | $q = 2\sigma_a^2\tau$ | `RadialMotion.noise_density_m2ps3` | 32.0 m²/s³ |
| Process noise over a step | $Q(T)$ | $q$ times the matrix below | `RadialMotion.process_noise` | [[10.667, −16], [−16, 32]] m², m²/s, m²/s² |
| Predict | $\mathbf{x}^-$, $P^-$ | $\mathbf{x}^- = F\mathbf{x}$, $P^- = FPF^\mathsf{T} + Q$ | `GaussianFilter.predict_to` | — |
| Measurement noise | $R_z$ | one bin over $\sqrt{12}$, on the diagonal | from the map's bins | 21.6357 m, 0.01725 m/s |
| Innovation | $\boldsymbol{\nu}$ | $\boldsymbol{\nu} = \mathbf{z} - h(\mathbf{x}^-)$ | `GaussianFilter.innovation_statistics` | — |
| Innovation covariance | $S$ | $S = HP^-H^\mathsf{T} + R_z$ | same | — |
| Normalised innovation squared (NIS) | $d^2$ | $d^2 = \boldsymbol{\nu}^\mathsf{T} S^{-1} \boldsymbol{\nu}$ | same | — |
| Gate | $\gamma$ | accept when $d^2 \le \gamma = \chi^2_m(P_G)$ | `gate_threshold`, `ChiSquareGate` | 8.81 (range only), 11.62 (range and rate), $P_G$ = 0.997 |
| Kalman gain | $K$ | $K = P^-H^\mathsf{T}S^{-1}$ | `GaussianFilter.update` | — |
| Update | $\mathbf{x}^+$, $P^+$ | $\mathbf{x}^+ = \mathbf{x}^- + K\boldsymbol{\nu}$, $P^+ = (I-KH)P^-(I-KH)^\mathsf{T} + KR_zK^\mathsf{T}$ | same | — |
| Confirmation | M of N | M hits in a track's first N scans | `LifecyclePolicy`, `TrackManager` | 3 of 5 |

Symbols used only on this page: $\sigma_a$, the standard deviation of the target's
acceleration; $\tau$, how long one acceleration lasts; $\mathbf{z}$, a measurement; $h$, the
measurement model, with $H$ its Jacobian; $m$, the measurement's dimension; $P_G$, the gate
probability; $\dot R$, the range rate, positive closing.

Four equations carry the filter. The motion model gives $F$ and $Q$:

$$F = \begin{bmatrix} 1 & -T \\ 0 & 1 \end{bmatrix}, \qquad
Q(T) = q\begin{bmatrix} T^3/3 & -T^2/2 \\ -T^2/2 & T \end{bmatrix}$$

The off-diagonal terms are negative because range rate is positive when closing, so a closing
target's range falls.

$$\mathbf{x}^- = F\mathbf{x}, \qquad P^- = FPF^\mathsf{T} + Q(T)$$

Predict: move the estimate forward, and grow its uncertainty by what the target could have done
in the meantime.

$$d^2 = \boldsymbol{\nu}^\mathsf{T} S^{-1} \boldsymbol{\nu} \le \chi^2_m(P_G)$$

Gate: a measurement is a candidate for a track only if it is no more surprising than a true one
would be, $P_G$ of the time.

$$K = P^-H^\mathsf{T}S^{-1}, \qquad \mathbf{x}^+ = \mathbf{x}^- + K\boldsymbol{\nu}$$

Update: move the estimate towards the measurement, further when the prediction is uncertain
($P^-$ large) and less when the measurement is noisy ($R_z$ large).

The covariance update in the table is the Joseph form, which stays symmetric and positive in
floating point. The KF and EKF use it, because their Taylor linearisation gives an $H$. The UKF
has no $H$: it pushes sigma points through $h$ instead, and uses $P^- - KSK^\mathsf{T}$.
`GaussianFilter.update`'s docstring shows that the two forms agree.

## How it works, step by step

`Tracker.process(batch)` takes one scan, meaning every measurement from one sensor at one time.
It does these five steps in order:

1. **Predict.** Move every track forward to the scan time (`GaussianFilter.predict_to`). A track that has
   gone longer than `max_coast_time_s` without a measurement is deleted.
2. **Gate and score.** For each track and measurement, work out the **innovation**: the
   difference between the measurement and what the track expected to see
   (`GaussianFilter.innovation_statistics`). From it comes the **NIS**, the normalised innovation squared,
   which measures how surprising the measurement is; its square root is the *Mahalanobis
   distance*. A pair fails the *chi-square gate* (`ChiSquareGate.accepts`) when its NIS is
   larger than a true measurement's would be, at the gate probability, and is dropped.
   A `model_selector` may offer a pair more than one model, best first, such as range and
   range rate and then range alone (a `marginal` of the first). The pair keeps the first
   model it passes. When pairs of different dimensions pass, the cost must be the negative
   log-likelihood: NIS grows with the dimension, so it cannot compare them.
3. **Assign.** Pair measurements with tracks (`GlobalNearestNeighbour`). By default this is
   done in two stages. Confirmed tracks pick first, from all the measurements. Tentative tracks
   then pick from what is left. With `assignment="joint"`, every track picks in one solve.
4. **Update, and confirm or delete.** Correct each track that got a measurement
   (`GaussianFilter.update`), reusing the innovation from step 2. Then count a hit or a miss for every
   track the sensor can see (`TrackManager.record_hit`, `TrackManager.record_miss`), and apply
   the M-of-N rule as it is counted:
   - A tentative track is confirmed once it has M hits within its first N scans.
   - It is deleted as soon as M hits can no longer be reached.
   - A confirmed track that misses *coasts*: it is predicted forward with nothing to update
     it, and its status is `"coasting"` until its next hit. It is deleted after
     `n_delete_misses` misses in a row, or `n_reacquire_frames` more if it is confirmed.
   - On the miss that starts those extra frames, the coordinates named in
     `LifecyclePolicy.reacquire_reset` are forgotten: their variances go back to the
     initiator's prior. The gate then widens as the track coasts, so it can find its target
     again under the same track ID.
5. **Start.** Begin a tentative track from each measurement that no track took
   (`DirectStateInitiator`). In two-stage assignment, a measurement inside a confirmed track's
   gate starts none.

`lifecycle.py`'s module docstring works an M-of-N example through scan by scan. Steps 3 and 5
stop a new, unproven track from taking a confirmed track's measurement; `tracker.py`'s module
docstring explains why.

### Choosing the process noise

Real targets don't move at exactly constant velocity. The **process noise** says how far they
may stray from it. You give it as two physical numbers:

- `sigma_acceleration_mps2`, σ_a: the standard deviation of the target's acceleration. If you
  know a largest acceleration `a_max` and treat it as a 3σ bound, σ_a = a_max / 3.
- `acceleration_correlation_time_s`, τ: roughly how long one acceleration lasts.

The motion model turns them into the noise density $q = 2\sigma_a^2\tau$, and builds $Q(T)$
from the actual time step at every prediction ([the equations above](#quantities-and-equations)).

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

The table's worked values come from this:

```python
import numpy as np

from radar_forge.core.radar import Transmitter
from radar_forge.core.tracking import RadialMotion, gate_threshold
from radar_forge.core.waveforms import range_resolution_m

# Scenario 003's UKF run: sigma_a = 4 m/s^2, tau = 1 s, one scan per second.
motion = RadialMotion(sigma_acceleration_mps2=4.0, acceleration_correlation_time_s=1.0)
print(f"q = {motion.noise_density_m2ps3:.1f} m^2/s^3")
print("Q(1 s) =", motion.process_noise(np.zeros(2), 1.0).round(3).tolist())

# Measurement noise: one range bin, and one velocity bin, over sqrt(12).
transmitter = Transmitter(
    carrier_frequency_hz=9.8e9,
    bandwidth_hz=2.0e6,
    transmit_power_w=100.0,
    gain_tx_dbi=30.0,
    chirp_duration_s=1.0e-3,
    prf_hz=1.0e3,
)
n_pulses = 256
range_bin_m = range_resolution_m(transmitter.bandwidth_hz)
cpi_s = n_pulses * transmitter.pulse_repetition_interval_s
velocity_bin_mps = transmitter.wavelength_m / (2 * cpi_s)
print(f"sigma_range = {range_bin_m / np.sqrt(12):.4f} m")
print(f"sigma_rate = {velocity_bin_mps / np.sqrt(12):.5f} m/s")

for dim in (1, 2):
    print(f"gate on d^2 for {dim} dimension(s), P_G = 0.997: {gate_threshold(0.997, dim):.2f}")
```

```text
q = 32.0 m^2/s^3
Q(1 s) = [[10.667, -16.0], [-16.0, 32.0]]
sigma_range = 21.6357 m
sigma_rate = 0.01725 m/s
gate on d^2 for 1 dimension(s), P_G = 0.997: 8.81
gate on d^2 for 2 dimension(s), P_G = 0.997: 11.62
```

## Try it in a scenario

Scenario 003 runs detection and tracking over scenario 001's S1 waveform. Its scenario file,
[scenario 003 — tracking](../../scenarios/scenario_003_tracking.toml), sets the run up.
Run the UKF variant:

```bash
uv run python scripts/run_scenario.py scenarios/scenario_003_ukf_fmcw_low_prf.toml --out out/ukf
```

It prints a summary at the end: the track is confirmed in 118 of the 120 frames, under one
track ID, with a range error (RMSE) of 7.6 m.

**Now change one thing.** Copy the TOML, set `sigma_accel_mps2 = 0.3` (the filter now believes
the aircraft barely accelerates), and run it again. The filter now trusts its prediction too
much, so whenever the aircraft's range rate changes the track falls behind, until the
detections land outside its gate. The summary now shows
**three** confirmed track IDs instead of one: the target's track broke and restarted. The main
track is confirmed in only 63 frames, and its range error grows to 38.6 m. `tracks.csv` and
`metrics.csv` in the output folder hold the details, scan by scan.

Want an open problem? [Where does the tracker break?](../../interns/tracker-breakpoint.md) and
[Does the tracker know its own uncertainty?](../../interns/conformal-tracking.md) both start
from here.

## Things that look like bugs and aren't

- **A track's range rate is positive while its range is falling.** Range rate is positive
  closing, everywhere in the library, so $F$ has $-T$ where a textbook's has $+T$.
- **The gate is a number, not a distance in metres.** It bounds $d^2$, which has no unit, and
  its value depends on the measurement's dimension: 8.81 for range alone, 11.62 for range and
  rate, at the same $P_G$. A track that starts range-only and later uses range rate changes
  dimension mid-run, so the gate is never hard-coded.
- **A target that turns makes the track fall behind, then catch up.** The motion model is
  constant velocity, so a turn is "noise" to it. The process noise decides how quickly it
  catches up; see [Choosing the process noise](#choosing-the-process-noise).

## Limits

**Not yet supported here:**

- keeping a wrong Doppler fold out of the gate while a target manoeuvres. The likelihood
  ratio (`cost="likelihood_ratio"`) compares range-only and range-and-rate pairs fairly, but
  in scenario 003's S1 a wrong-fold detection still passes the gate during the manoeuvre and
  pulls the track 15 m/s out (`spec/scenario-003-tracking.md` §14.12);
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

## References

- Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and
  Navigation*, Wiley, 2001: §5.2.4 (the Kalman filter, its innovation and covariance), §5.4.2
  (the NIS consistency test), §6.2 (continuous white-noise acceleration).
- Y. Bar-Shalom and X. R. Li, *Multitarget-Multisensor Tracking: Principles and Techniques*,
  YBS Publishing, 1995: §2.3.2 (the validation gate), §2.6.1 and §7.3 (M-of-N track
  formation), §7.1.1 (association as an assignment problem).
- S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013: ch. 5 (the
  EKF and UKF) and ch. 6 (one Gaussian filter, with the Taylor and unscented approximations).
- E. A. Wan and R. van der Merwe, "The unscented Kalman filter for nonlinear estimation,"
  *Proc. IEEE AS-SPCC*, 2000, pp. 153–158.
- S. J. Julier, "The scaled unscented transformation," *Proc. American Control Conference*,
  2002, pp. 4555–4559.
- R. A. Singer, "Estimating optimal tracking filter performance for manned maneuvering
  targets," *IEEE Trans. Aerospace and Electronic Systems*, vol. AES-6, no. 4, pp. 473–483,
  1970.
- [Stone Soup](https://stonesoup.readthedocs.io), whose split into components this package
  follows.

## Three things that look like bugs and are not

Scenario 003's plots show three behaviours that read as faults and are not.

- **The range-1D transition matrix carries $-T$, not $+T$.** While the library is closing-positive
  a closing target's range must shrink, so the matrix is the textbook constant-velocity pair with
  the sign of the rate flipped. A $+T$ version gives a plausible picture in which the filter
  dead-reckons the target the wrong way down the line of sight.
- **A track spends its first ten or so frames measuring range only.** Track-aided unfolding needs
  a velocity prediction to select a fold, and a new track has none, so it bootstraps on a batch
  range-slope estimator instead.
- **On S1 the fold index is right about 86 % of the time, and no tracker can do better.** The
  target's range rate moves by more than half a fold span in 3.4 % of frames, so the information
  is not there. The dual-PRF variant is right every frame, because the waveform resolves what the
  tracker cannot.
