# 6. Tracking

`radar_forge.tracking` turns plots, scan after scan, into tracks. A **track** is the tracker's
running estimate of one target: its state (for example its range and range rate), how
uncertain that estimate is, and whether the tracker trusts it yet. Every track is one row of a
`TrackBank`, and each step of the filter runs over every row at once. It is the last stage of
[the chain](README.md#the-chain): it takes the plots from [page 5](05-detection.md), and
`pipeline.ScanTracker` builds it from a scenario's `[tracking]` table. This page says what is
in the package, the equations it runs, the order a scan runs in, how to try it, and what it
can't do yet.

## What this stage does

One plot says where a target was at one moment, with some error. A track combines many of
them: it predicts where the target should be next, compares that with what the radar reports,
and corrects itself. Doing that well means answering three questions every scan. Which plot
belongs to which track? How much should a new plot move the estimate? And when is a track real
enough to report, or gone for good?

The scenarios' state is the map's own: bistatic range $R_b = R_T + R_R$ and its rate
$\dot R_b$, the time derivative, **positive receding**. For a monostatic radar $R_b$ is twice
the slant range. Nothing in the tracker halves it; only a figure or a CSV does
(`plots.display_axes`). So every rate and acceleration in `[tracking]` is of bistatic range,
twice the radial value when monostatic.

### What's in the package

| Module | What it holds |
| :-- | :-- |
| `models.py` | `StateLayout` and `Coordinate`, the named coordinates of a state or measurement, which hold the package's one wrap of a periodic value; `check_covariances`, the one boundary check (symmetric, positive definite). Motion models, which predict a state forward: `RadialConstantVelocity` (state `[bistatic_range_m, bistatic_range_rate_mps]`) and `CartesianMotion` (constant velocity or acceleration on each ENU axis). Measurement models, which predict a measurement from a state: `LinearMeasurement` (some state coordinates, reported directly, optionally folded at a period), `RangeAzimuthElevationRangeRate` (a monostatic radar's spherical measurement of an ENU state), `BistaticRangeRateModel` ($R_T + R_R$ and its rate; $(2r, 2\dot r)$ when the sites coincide) and `marginal` (some of a model's components, such as range alone). |
| `filters.py` | `TrackBank`, every track a row of one immutable struct of arrays; `TrackStatus` and the read-only `TrackSnapshot`; the one Gaussian filter, `predict`, `innovate` and `update`, through `TaylorLinearisation` (the KF on a linear model, the EKF otherwise) or `UnscentedTransform` (the UKF). |
| `association.py` | `chi_square_gate_threshold` and `gate`; `association_cost`, `"nis"` or `"negative_log_likelihood_ratio"`; `GlobalNearestNeighbour` (GNN) and `GreedyNearestNeighbour`. |
| `tracker.py` | `Tracker`, which runs one `Scan` at a time; `Scan`, every measurement one sensor made at one time with its R, checked and frozen; `LifecyclePolicy`, the M-of-N rule, deletion and re-acquisition; the initiators `DirectStateInitiator` and `DebiasedSphericalInitiator`. |
| `metrics.py` | `nees`, `consistency_interval` and `rmse`, which score a tracker against the truth. |

`radar_forge.tracking` exports `Scan` and `Tracker`; everything else is imported from its
module.

## Quantities and equations

The filter is written once, as a Gaussian filter, for the KF, EKF and UKF alike. The worked
values are for scenario 003's UKF on the S1 waveform
([`config/scenario_003_ukf_fmcw_low_prf.toml`](../../config/scenario_003_ukf_fmcw_low_prf.toml)),
whose scans come once a second. S1 folds the rate at ±15.30 m/s, so this tracker measures range
alone and infers the rate.

| Quantity | Symbol | Equation | In the code | Scenario 003, UKF on S1 |
| :-- | :-- | :-- | :-- | --: |
| State and its covariance | $\mathbf{x}$, $P$ | mean and covariance of the estimate | a row of `TrackBank.mean`, `.covariance` | $[R_b, \dot R_b]$ |
| Transition over a step $T$ | $F$ | $R_b \to R_b + T\dot R_b$, $\dot R_b \to \dot R_b$ | `RadialConstantVelocity.jacobian` | $T$ = 1 s |
| Process-noise intensity | $q$ | $q = 2\sigma_a^2\tau$ (`"cwna"`) | `RadialConstantVelocity(sigma_acceleration_mps2, ...)` | 128 m²/s³ |
| Process noise over a step | $Q(T)$ | $q$ times the matrix below | `.process_noise` | [[42.667, 64], [64, 128]] m², m²/s, m²/s² |
| Predict | $\mathbf{x}^-$, $P^-$ | $\mathbf{x}^- = F\mathbf{x}$, $P^- = FPF^\mathsf{T} + Q$ | `predict` | — |
| Measurement noise | $R_z$ | one bin over $\sqrt{12}$, on the diagonal | `ScanTracker.noise_covariance` | 43.27 m (0.0345 m/s when the rate is measured) |
| Innovation | $\boldsymbol{\nu}$ | $\boldsymbol{\nu} = \mathbf{z} - h(\mathbf{x}^-)$ | `innovate`, `Innovation.innovation` | — |
| Innovation covariance | $S$ | $S = HP^-H^\mathsf{T} + R_z$ | `Innovation.innovation_covariance` | — |
| Normalised innovation squared (NIS) | $d^2$ | $d^2 = \boldsymbol{\nu}^\mathsf{T} S^{-1} \boldsymbol{\nu}$ | `Innovation.nis` | — |
| Gate | $\gamma$ | accept when $d^2 \le \gamma = \chi^2_m(P_G)$ | `chi_square_gate_threshold`, `gate` | 8.81 (range), 11.62 (range and rate), $P_G$ = 0.997 |
| Kalman gain | $K$ | $K = P_{xz}S^{-1}$, with $P_{xz} = P^-H^\mathsf{T}$ | `update` | — |
| Update | $\mathbf{x}^+$, $P^+$ | $\mathbf{x}^+ = \mathbf{x}^- + K\boldsymbol{\nu}$; $P^+$ below | `update` | — |
| Normalised estimation error squared (NEES) | $\epsilon$ | $\epsilon = \mathbf{e}^\mathsf{T} P^{-1} \mathbf{e}$, $\mathbf{e} = \mathbf{x}_{\mathrm{true}} - \hat{\mathbf{x}}$ | `metrics.nees` | inside $\chi^2_2$'s 95 % interval in 0.89 of the confirmed frames |
| Confirmation | M of N | M hits in a track's first N scans | `LifecyclePolicy` | 3 of 5 |

Symbols used only on this page: $R_b$ and $\dot R_b$, bistatic range and its rate;
$\sigma_a$, the standard deviation of the acceleration of $R_b$; $\tau$, how long one
acceleration lasts; $\mathbf{z}$, a measurement; $h$, the measurement model, with $H$ its
Jacobian; $P_{xz}$, the cross-covariance of state and measurement; $m$, the measurement's
dimension; $P_G$, the gate probability.

The motion model gives $F$ and $Q$:

$$F = \begin{bmatrix} 1 & T \\ 0 & 1 \end{bmatrix}, \qquad
Q(T) = q\begin{bmatrix} T^3/3 & T^2/2 \\ T^2/2 & T \end{bmatrix}$$

This is continuous white-noise acceleration (CWNA). The off-diagonal terms are positive
because the rate is the range's derivative: a receding target's range grows. Discrete
white-noise acceleration (`"dwna"`), which scenario 003's Kalman path uses, holds one
acceleration of standard deviation $\sigma_a$ through each step instead:
$Q(T) = \sigma_a^2 \begin{bmatrix} T^4/4 & T^3/2 \\ T^3/2 & T^2 \end{bmatrix}$, so
[[25, 50], [50, 100]] at its $\sigma_a$ = 10 m/s².

$$\mathbf{x}^- = F\mathbf{x}, \qquad P^- = FPF^\mathsf{T} + Q(T)$$

Predict: move the estimate forward, and grow its uncertainty by what the target could have done
in the meantime.

$$d^2 = \boldsymbol{\nu}^\mathsf{T} S^{-1} \boldsymbol{\nu} \le \chi^2_m(P_G)$$

Gate: a measurement is a candidate for a track only if it is no more surprising than a true one
would be, $P_G$ of the time.

$$K = P_{xz}S^{-1}, \qquad \mathbf{x}^+ = \mathbf{x}^- + K\boldsymbol{\nu}$$

Update: move the estimate towards the measurement, further when the prediction is uncertain
($P^-$ large) and less when the measurement is noisy ($R_z$ large).

With an $H$ (the Taylor linearisation) the covariance update is the Joseph form,
$P^+ = (I-KH)P^-(I-KH)^\mathsf{T} + KR_zK^\mathsf{T}$, which stays symmetric and positive in
floating point. The UKF has no $H$: it pushes sigma points through $h$ instead, and uses
$P^+ = P^- - KSK^\mathsf{T}$. At the optimal gain the two are equal.

A consistent filter's NEES is $\chi^2$ with as many degrees of freedom as the state has, and
its NIS with as many as the measurement has. The average of $N$ such draws is $\chi^2_{nN}/N$,
whose interval `consistency_interval(n, N)` gives. A filter whose NEES sits below its interval
is pessimistic (its $P$ is too large); above, optimistic.

## How it works, step by step

`Tracker.process(scan)` takes one `Scan`: every measurement from one sensor at one time, shape
`(n_meas, m)`, with its R. It runs these steps over every track and measurement at once:

1. **Predict.** Move every row to the scan time (`predict`). A row that has gone longer than
   `max_coast_time_s` without a hit is no longer live, and is deleted.
2. **Gate and score.** `innovate` scores every row against every measurement, an
   `(n_tracks, n_meas)` array of pairs from one batched Cholesky factorisation of S, with no
   loop over pairs. `gate` keeps a pair whose NIS is at most $\chi^2_m(P_G)$. A
   `model_selector` may offer a row more than one model, best first, such as range and rate
   and then range alone (a `marginal`); a pair keeps the first model it passes. Each kept pair
   is costed by its NIS, or by the negative log-likelihood ratio
   $-\ln g + \ln \lambda$, where $g$ is the pair's Gaussian likelihood and $\lambda$ a clutter
   density per model. NIS grows with the dimension, so it cannot compare a range-only pair with
   a range-and-rate one, and a scan that mixes them under `cost="nis"` raises.
3. **Assign.** `GlobalNearestNeighbour` solves the scan's assignment with
   `scipy.optimize.linear_sum_assignment`, making as many pairs as the gate allows before
   minimising cost. By default (`assignment="confirmed_first"`) confirmed tracks pick first,
   from all the measurements, and tentative tracks pick from what is left; `"joint"` assigns
   every track in one solve.
4. **Update.** Correct each assigned row (`update`), once per model offered, reusing the
   innovation from step 2.
5. **Confirm or delete.** Count a hit or a miss for every live row, as integer arithmetic on
   the bank's count columns, and apply `LifecyclePolicy`:
   - A tentative track is confirmed on its M-th hit within its first N scans, and deleted as
     soon as M can no longer be reached.
   - A confirmed track that misses *coasts*: it is predicted forward with nothing to update it,
     status `COASTING` until its next hit. It is deleted after `n_delete_misses` misses in a
     row, or `n_reacquire_misses` more if it was confirmed.
   - On the miss that opens those extra misses, the coordinates named in `reacquire_reset`
     return to the initiator's prior variance, so the gate widens and the track can find its
     target again under the same ID.
6. **Start.** Begin a tentative row from each measurement no track took (`DirectStateInitiator`,
   which copies the measured coordinates into a prior). With `seed_names` it copies only those,
   such as range while a folded rate is untrusted. In two-stage assignment, a measurement inside
   a confirmed track's gate starts none.

`process` returns the bank with this scan's deleted rows in it, once; `tracker.bank` keeps the
living. The bank is replaced only when a scan completes, so a scan that raises leaves the
tracker as it was. `tracker.py`'s module docstring works an M-of-N example: at M = 3, N = 5,
hit, miss, hit, miss, hit confirms on scan 5, and hit, miss, miss, miss deletes on scan 4.

### The scenario's tracker

`pipeline.ScanTracker(scenario)` builds a `Tracker` from `[tracking]` (`linearisation`,
`process_noise`, `initiator`, `cost`, `assignment`, the lifecycle counts) on
`RadialConstantVelocity`, and `ScanTracker.scan(plots, time_s)` feeds it a frame's accepted
plots. Range is measured modulo the map's span, 149,896 m for S1, while the state stays
continuous. The rate is measured only where it cannot fold: after dual-PRF pairing, from
one burst whose rate interval covers `max_bistatic_range_rate_mps`, or once a track can unfold
it (`rate_unfolding = "track_aided"`); otherwise range alone. Track-aided unfolding lets a track
unfold once it holds N associated ranges, the least N whose least-squares slope is certain to a
`unfold_sigma_gate`-th of a fold span:

$$\frac{\sigma_R}{T}\sqrt{\frac{12}{N(N^2-1)}} < \frac{v_{\mathrm{span}}}{g}$$

with $v_{\mathrm{span}}$ the rate span and $g$ the gate. On S1, $\sigma_R$ = 43.27 m,
$v_{\mathrm{span}}$ = 30.591 m/s and $g$ = 6 give N = 10. Of the tracks that may unfold, the
one with the longest range history then gives each plot the fold nearest its rate, unless that
lies over half a span from the slope of its last `n_slope_scans` ranges (the **fold monitor**):
a wrong fold agrees with the prediction that chose it, never with the range.

### Choosing the process noise

Real targets don't move at exactly constant velocity. The **process noise** says how far they
may stray from it, given as two physical numbers:

- `sigma_acceleration_mps2`, $\sigma_a$: the standard deviation of the acceleration of
  bistatic range, twice the radial one when monostatic. A largest acceleration `a_max` read as
  a 3σ bound gives $\sigma_a$ = `a_max` / 3.
- `acceleration_correlation_time_s`, $\tau$: roughly how long one acceleration lasts.

`"cwna"` turns them into $q = 2\sigma_a^2\tau$ and builds $Q(T)$ from the actual step at every
prediction ([the equations above](#quantities-and-equations)); it is right for any step.
`"dwna"` reads $\sigma_a$ alone. The two add equal velocity variance per step only when
$\tau = T/2$. The UKF scenario's 8 m/s² and 1 s are measured from scenario 001's truth: 4.0 m/s²
radial, with a correlation that falls to about a half after 1 s.

If $\sigma_a$ is too small, the track lags a manoeuvring target until its plots fall outside the
gate, and it breaks. If it is too large, the track follows the noise.

## Try it

This tracks one target receding at 60 m/s of bistatic range, from range-and-rate measurements
with scenario 003's noise:

```python
import numpy as np

from radar_forge.tracking import Scan, Tracker
from radar_forge.tracking.association import GlobalNearestNeighbour
from radar_forge.tracking.filters import UnscentedTransform
from radar_forge.tracking.models import LinearMeasurement, RadialConstantVelocity
from radar_forge.tracking.tracker import DirectStateInitiator, LifecyclePolicy

# State and measurement: bistatic range and its rate, positive receding.
motion = RadialConstantVelocity(8.0)  # sigma_a of bistatic range, m/s^2
names = ("bistatic_range_m", "bistatic_range_rate_mps")
models = {"range_rate": LinearMeasurement(motion.state_layout, names)}
sigma = np.array([43.27, 0.0345])  # one bin over sqrt(12): m, m/s
tracker = Tracker(
    models,
    motion,
    UnscentedTransform(),
    0.997,
    GlobalNearestNeighbour(),
    DirectStateInitiator(np.zeros(2), np.diag([43.27, 200.0]) ** 2),
    LifecyclePolicy(n_confirm_hits=3, n_confirm_scans=5),
)

# A target receding at 60 m/s of bistatic range, one scan a second.
rng = np.random.default_rng(20260911)
for time_s in range(5):
    measured = [36_000.0 + 60.0 * time_s, 60.0] + sigma * rng.standard_normal(2)
    bank = tracker.process(Scan(time_s, [measured], np.diag(sigma**2), "range_rate"))
    (track,) = bank.snapshots()  # one row: the bank holds one track
    range_m, rate_mps = track.mean
    print(
        f"{time_s} s: track {track.track_id}, {track.status.name.lower()}, {range_m:.1f} m,"
        f" {rate_mps:.3f} m/s, sigma {np.sqrt(track.covariance[0, 0]):.1f} m"
    )
```

The rule is 3-of-5, so the track is confirmed on its third hit. Its range grows while its rate
is positive, and its range uncertainty falls below one measurement's 43.27 m:

```text
0 s: track 1, tentative, 35972.9 m, 60.028 m/s, sigma 43.3 m
1 s: track 1, tentative, 36023.1 m, 59.976 m/s, sigma 30.6 m
2 s: track 1, confirmed, 36085.7 m, 60.013 m/s, sigma 25.1 m
3 s: track 1, confirmed, 36153.3 m, 59.969 m/s, sigma 21.8 m
4 s: track 1, confirmed, 36224.6 m, 59.970 m/s, sigma 19.7 m
```

The table's worked values come from the scenario file and the map's axes:

```python
import numpy as np

from radar_forge.pipeline import map_axes
from radar_forge.scenario import load_scenario
from radar_forge.tracking.association import chi_square_gate_threshold
from radar_forge.tracking.models import RadialConstantVelocity

# The UKF on S1: [tracking] gives sigma_a = 8 m/s^2 of bistatic range, tau = 1 s, CWNA.
scenario = load_scenario("config/scenario_003_ukf_fmcw_low_prf.toml")
settings = scenario.tracking
motion = RadialConstantVelocity(
    settings.sigma_acceleration_mps2,
    acceleration_correlation_time_s=settings.acceleration_correlation_time_s,
    process_noise=settings.process_noise,
)
step_s = 1.0 / scenario.frame_rate_hz
print("F(1 s) =", motion.jacobian(np.zeros(2), step_s).tolist())
print("Q(1 s) =", motion.process_noise(np.zeros(2), step_s).round(3).tolist())

# Measurement noise: one bin of the map's own axes over sqrt(12).
range_axis_m, rate_axis_mps = map_axes(scenario.radar, scenario.n_pulses[0])
range_bin_m = range_axis_m[1] - range_axis_m[0]
rate_bin_mps = rate_axis_mps[0] - rate_axis_mps[1]  # the rate axis falls with index
print(f"sigma_range = {range_bin_m:.3f} / sqrt(12) = {range_bin_m / np.sqrt(12):.2f} m")
print(f"sigma_rate = {rate_bin_mps:.6f} / sqrt(12) = {rate_bin_mps / np.sqrt(12):.4f} m/s")
print(f"rate folds at +-{rate_axis_mps.size * rate_bin_mps / 2:.2f} m/s")

for probability in (0.997, 0.99):
    gates = [chi_square_gate_threshold(probability, m) for m in (1, 2)]
    print(f"gate at P_G = {probability}: {gates[0]:.2f} (range), {gates[1]:.2f} (range and rate)")
```

```text
F(1 s) = [[1.0, 1.0], [0.0, 1.0]]
Q(1 s) = [[42.667, 64.0], [64.0, 128.0]]
sigma_range = 149.896 / sqrt(12) = 43.27 m
sigma_rate = 0.119496 / sqrt(12) = 0.0345 m/s
rate folds at +-15.30 m/s
gate at P_G = 0.997: 8.81 (range), 11.62 (range and rate)
gate at P_G = 0.99: 6.63 (range), 9.21 (range and rate)
```

$P_G$ = 0.99 is scenario 003's Kalman path. Last, the bank at work: 200 Monte Carlo runs of
one UKF are the 200 rows of one bank, so one `predict`, `innovate` and `update` step them all,
and NEES and NIS check that the filter knows its own uncertainty:

```python
import numpy as np

from radar_forge.tracking.filters import TrackBank, UnscentedTransform, innovate, predict, update
from radar_forge.tracking.metrics import consistency_interval, nees
from radar_forge.tracking.models import LinearMeasurement, RadialConstantVelocity

# 200 runs of one UKF as the 200 rows of one bank, measuring range alone.
motion = RadialConstantVelocity(8.0)
model = LinearMeasurement(motion.state_layout, ["bistatic_range_m"])
ukf, noise_m2 = UnscentedTransform(), np.array([[43.27**2]])
n_runs, prior = 200, np.diag([43.27, 20.0]) ** 2
rng = np.random.default_rng(20260911)
start = np.array([36_000.0, 60.0])
truth = start + rng.multivariate_normal(np.zeros(2), prior, n_runs)
bank = TrackBank.empty(motion.state_layout).append(
    np.zeros((n_runs, 2)) + start,
    np.zeros((n_runs, 2, 2)) + prior,
    0.0,
    track_id=np.arange(1, n_runs + 1),
    status=np.zeros(n_runs),
)
own = np.arange(n_runs)  # row i takes measurement i
step_noise = motion.process_noise(np.zeros(2), 1.0)
for time_s in np.arange(1.0, 21.0):
    kick = rng.multivariate_normal(np.zeros(2), step_noise, n_runs)
    truth = motion.transition(truth, 1.0) + kick
    measured = truth[:, :1] + 43.27 * rng.standard_normal((n_runs, 1))
    bank = predict(bank, motion, ukf, time_s)
    scored = innovate(bank, model, ukf, measured, noise_m2)  # every row, every measurement
    bank = update(bank, scored, own)

print("pairs scored in one scan:", scored.nis.shape)
low, high = consistency_interval(2, n_runs)
mean_nees = nees(bank.mean, bank.covariance, truth).mean()
print(f"mean NEES {mean_nees:.3f}, 99.9 % interval ({low:.3f}, {high:.3f})")
low, high = consistency_interval(1, n_runs)
print(f"mean NIS {scored.nis[own, own].mean():.3f}, 99.9 % interval ({low:.3f}, {high:.3f})")
```

```text
pairs scored in one scan: (200, 200)
mean NEES 1.979, 99.9 % interval (1.567, 2.498)
mean NIS 1.079, 99.9 % interval (0.703, 1.362)
```

After 20 scans both averages sit inside their intervals: the truth moves as the motion model
says, so the filter is consistent. A real target does not, which is what the stepper's
`nees_fraction` measures.

## Try it in a scenario

Run the UKF on S1 through the stepper, over its whole 120-frame window:

```bash
uv run python scripts/run_scenario.py config/scenario_003_ukf_fmcw_low_prf.toml
```

`[track]` prints its scores of the target's primary track: `range_rmse_m = 15.1152`,
`range_rate_rmse_mps = 12.0151`, `nees_fraction = 0.889831`, `confirmation_latency_scans = 2`
and `n_confirmed_tracks = 1`, and all 11 checks pass. The one track is confirmed in 118 of the
120 frames. `runs/scenario_003_ukf_fmcw_low_prf/` holds `tracks.csv` (one row per track per
frame, with its NIS and the detection it took), `metrics.csv` and `track.png`. A run shorter
than 2 × `n_confirm_scans` frames, 10 here, skips the tracker's checks.

**Knob:** in a copy of the TOML in `config/`, set `sigma_acceleration_mps2 = 0.6` in
`[tracking]` (0.3 m/s² radial: the filter now believes the aircraft barely accelerates), and run
it again. The filter trusts its prediction too much, so whenever the aircraft's rate changes the
track falls behind until its plots land outside the gate. `[track]` now counts **three**
confirmed tracks where there was one: the target's track broke and restarted. The primary track
is confirmed in only 63 frames, its range RMSE grows to 77.2 m and its rate RMSE to 16.6 m/s,
and the copy ends `2 check(s) failed`, on `range_rate_rmse_mps` and on `nees_fraction` (0.16).
`nees_fraction` is the share of the primary track's confirmed frames whose NEES lies inside
$\chi^2_2$'s two-sided 95 % interval, 0.051 to 7.38.

For the hard case, run [scenario 003](../../config/scenario_003_tracking.toml) itself: a Kalman
path that measures the rate once track-aided unfolding has a fold, re-acquisition, and the
likelihood-ratio cost.

Want an open problem? [Where does the tracker break?](../../interns/tracker-breakpoint.md) and
[Does the tracker know its own uncertainty?](../../interns/conformal-tracking.md) both start
from here.

## Things that look like bugs and aren't

- **A track's range is twice the target's distance.** The state is bistatic range,
  $R_T + R_R$: $2R$ for a monostatic radar. `tracks.csv` shows it halved, as `range_m`, through
  `plots.display_axes`; the bank never is.
- **A receding target has a positive rate but a negative Doppler.** The rate is the range's
  time derivative, so $F$ has $+T$, and $f_d = -\dot R_b / \lambda$.
- **The gate is a number, not a distance in metres.** It bounds $d^2$, which has no unit, and
  its value depends on the measurement's dimension: 8.81 for range alone, 11.62 for range and
  rate, at the same $P_G$. A track that starts range-only and later uses the rate changes
  dimension mid-run, so the gate is never hard-coded.
- **The UKF on S1 never measures a rate, yet reports one.** S1 folds the rate at ±15.30 m/s,
  and the target may open or close at up to 200 m/s (`max_bistatic_range_rate_mps`), so
  `ScanTracker` measures range alone and the filter infers the rate from successive ranges: a
  rate RMSE of 12.0 m/s.
- **A new scenario-003 track has a rate of zero.** With `initiator = "range_only"` a track is
  born from range alone: its rate is the prior's 0 m/s, with a standard deviation of
  `max_bistatic_range_rate_mps`, 400 m/s, because the plot's rate is folded and would start the
  track confidently wrong.
- **A scenario-003 track measures range only until it holds ten ranges.** Track-aided unfolding
  needs a range slope certain to a sixth of a fold span first, which takes N = 10 ranges
  ([the scenario's tracker](#the-scenarios-tracker)).

## Limits

**Not yet supported here:**

- the interacting multiple model (IMM) filter and a coordinated-turn motion model: every motion
  model is linear (scenario 011);
- probabilistic association (JPDA) and Murty's k-best assignments, for heavy clutter
  (scenario 011); association is GNN or greedy nearest neighbour, each track taking at most one
  measurement a scan;
- OSPA and GOSPA, one score for a multi-target scene (scenario 011);
- an initiator for `BistaticRangeRateModel`: one bistatic measurement cannot fix a position, so
  `DirectStateInitiator` returns None for it. Add its tracks with `Tracker.seed`;
- sensor coverage: a scan counts a miss against every live track it does not update, whether
  or not that sensor could see it.

**Built-in assumptions.**

- Every measurement in a scan carries the scan's time, and scans arrive in time order; an
  out-of-sequence scan is rejected, not buffered.
- Covariances are positive definite, not just semidefinite: a zero variance in a prior, a seed
  or a scan's R is refused at the boundary (`check_covariances`).
- Track-aided unfolding takes one reference track per scan: a single-target policy.
- One target that gives several plots in one scan can start several tracks; only two-stage
  assignment holds back a plot inside a confirmed track's gate. The fix belongs in detection,
  which keeps one plot per range cell (`merge_range_bins`).

## References

- Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and
  Navigation*, Wiley, 2001: §5.2.4 (the Kalman filter, its innovation and covariance), §5.4.2
  (the NEES and NIS consistency tests), §5.5 (initialisation), §6.2 and §6.3.2 (continuous and
  discrete white-noise acceleration), §10.3 (the EKF).
- Y. Bar-Shalom and X. R. Li, *Multitarget-Multisensor Tracking: Principles and Techniques*,
  YBS Publishing, 1995: §2.3.2 (the validation gate), §2.6.1 (M-of-N track formation), §7.1.1
  (association as an assignment problem).
- S. S. Blackman, "Multiple hypothesis tracking for multiple target tracking," *IEEE Aerosp.
  Electron. Syst. Mag.*, vol. 19, no. 1, pp. 5–18, 2004 (the log-likelihood-ratio score).
- D. F. Crouse, "On implementing 2D rectangular assignment algorithms," *IEEE Trans. Aerosp.
  Electron. Syst.*, vol. 52, no. 4, pp. 1679–1696, 2016.
- S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013: ch. 5 (the
  EKF and UKF) and ch. 6 (one Gaussian filter, with the Taylor and unscented approximations).
- E. A. Wan and R. van der Merwe, "The unscented Kalman filter for nonlinear estimation,"
  *Proc. IEEE AS-SPCC*, 2000, pp. 153–158.
- R. A. Singer, "Estimating optimal tracking filter performance for manned maneuvering
  targets," *IEEE Trans. Aerospace and Electronic Systems*, vol. AES-6, no. 4, pp. 473–483,
  1970.
- N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005, ch. 3 (bistatic range and
  its rate).
