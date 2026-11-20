# 6. Tracking

`radar_forge.tracking` turns plots, scan after scan, into tracks. A **track** is the tracker's
running estimate of one target: its state (for example its range and range rate), how
uncertain that estimate is, and whether the tracker trusts it yet. Every track is one row of a
`TrackBank`, and each step of the filter runs over every row at once. It is the last stage of
[the chain](README.md#the-chain): it takes the plots from [page 5](05-detection.md), and
`scan_tracker.ScanTracker` builds it from a scenario's `[tracking]` table. This page says what is
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
(`figures.display_axes`). So every rate and acceleration in `[tracking]` is of bistatic range,
twice the radial value when monostatic.

### What's in the package

One line per module. **Tracker** code estimates; **scoring** compares tracks with the truth;
**simulation** makes the truth and the plots of a plot-level scenario.

| Module | Kind | What it is for |
| :-- | :-- | :-- |
| `tracking/motion.py` | tracker | The state's named coordinates, `CoordinateLayout`, which wraps a periodic one, and the motion models, F and Q: `ConstantVelocity` (on ENU axes, or on `[bistatic_range_m, bistatic_range_rate_mps]` built by `ConstantVelocity.bistatic_range`), `ConstantAcceleration`, `ConstantTurn` (the coordinated turn); `check_covariances`, the one boundary check. |
| `tracking/measurement.py` | tracker | The measurement models, h and H: `LinearMeasurement` (state coordinates reported directly, optionally folded at a period), `RangeAzimuthElevationRangeRate` (a monostatic radar's spherical measurement of an ENU state), `BistaticRangeAndRangeRate` ($R_T + R_R$ and its rate) and `marginal` (some of a model's components, such as range alone). |
| `tracking/filters.py` | tracker | `TrackBank`, every track a row of one immutable struct of arrays, with `TrackStatus` and the read-only `TrackSnapshot`; the one Gaussian filter, `predict`, `innovate`, `update` and `update_weighted`, through `TaylorLinearisation` (the KF on a linear model, the EKF otherwise) or `UnscentedTransform` (the UKF). |
| `tracking/initiation.py` | tracker | A new track's prior from one measurement: `DirectStateInitiator`, `DebiasedSphericalInitiator`. |
| `tracking/association.py` | tracker | The gate and the cost (`chi_square_gate_threshold`, `gate`, `association_cost`); GNN and greedy assignment; PDA, JPDA and its coalescence fixes; Murty's k best. |
| `tracking/lifecycle.py` | tracker | `LifecyclePolicy`, the M-of-N rule this page uses, with deletion and re-acquisition; `ExistencePolicy`, confirmation by existence probability (JIPDA). |
| `tracking/imm.py` | tracker | The interacting multiple model: each track's estimate under every mode, mixed, predicted and corrected. |
| `tracking/hypotheses.py` | tracker | Track-oriented MHT: each target a tree of association histories, scored, ranked by Murty's method and pruned N scans back. |
| `tracking/pmbm.py` | tracker | `PmbmTracker`, the Poisson multi-Bernoulli mixture tracker. |
| `tracking/clutter_map.py` | tracker | `ClutterMap`, the clutter density per (m rad) learnt from the plots no confirmed track took. |
| `tracking/tracker.py` | tracker | `Scan`, every measurement one sensor made at one time with its R, checked and frozen; `Tracker`, which runs one scan at a time; `Coverage`, which tracks a scan could see. |
| `tracking/metrics.py` | scoring | `nees`, `consistency_interval` and `rmse` for one target; `gospa` and `trajectory_gospa` for a many-target scene. |
| `scan_tracker.py` | tracker | `ScanTracker`: a scenario's `[tracking]` table built into one tracker, and the policy that feeds it each scan ([below](#the-scenarios-tracker)). |
| `trajectory.py` | simulation | `Arc` and `Leg`: the closed-form true trajectories of scenario 011's targets. |
| `plot_simulator.py` | simulation | `plot_scans`: the plots a rotating surveillance radar makes of those trajectories, scan by scan (merged echoes, missed detections, false plots), each carrying the clutter density at it. |
| `verify/chain.py` | scoring | The stepper's `track` stage (one target, from IQ): scores the tracks against the truth and checks the scores against `[verify]`. |
| `verify/multitarget.py` | scoring | The `multitarget` stage (many targets, from plots): the same scoring and check (011's IQ check: detection statistics only). |

`radar_forge.tracking` exports `Scan` and `Tracker`; everything else is imported from its
module. The package imports nothing else from `radar_forge` but the shared fold
(`_periodic`): the last four rows, which know about scenarios, TOML and the truth, live
outside it. Four of scenario 011's six variants track the plot simulator's plots in place of
IQ.

## Quantities and equations

The filter is written once, as a Gaussian filter, for the KF, EKF and UKF alike. The worked
values are for scenario 003's UKF on the S1 waveform
([`config/scenario_003_ukf_fmcw_low_prf.toml`](../../config/scenario_003_ukf_fmcw_low_prf.toml)),
whose scans come once a second. S1 folds the rate at ±15.30 m/s, so this tracker measures range
alone and infers the rate.

| Quantity | Symbol | Equation | In the code | Scenario 003, UKF on S1 |
| :-- | :-- | :-- | :-- | --: |
| State and its covariance | $\mathbf{x}$, $P$ | mean and covariance of the estimate | a row of `TrackBank.mean`, `.covariance` | $[R_b, \dot R_b]$ |
| Transition over a step $T$ | $F$ | $R_b \to R_b + T\dot R_b$, $\dot R_b \to \dot R_b$ | `ConstantVelocity.bistatic_range(...).jacobian` | $T$ = 1 s |
| Process-noise intensity | $q$ | $q = 2\sigma_a^2\tau$ (`white_noise = "continuous"`) | `ConstantVelocity.bistatic_range(sigma_acceleration_mps2, ...)` | 128 m²/s³ |
| Process noise over a step | $Q(T)$ | $q$ times the matrix below | `.process_noise` | [[42.667, 64], [64, 128]] m², m²/s, m²/s² |
| Predict | $\mathbf{x}^-$, $P^-$ | $\mathbf{x}^- = F\mathbf{x}$, $P^- = FPF^\mathsf{T} + Q$ | `predict` | — |
| Measurement noise | $R_z$ | one bin over $\sqrt{12}$, on the diagonal | `ScanTracker.noise_covariance` | 43.27 m (0.0345 m/s when the rate is measured) |
| Innovation | $\boldsymbol{\nu}$ | $\boldsymbol{\nu} = \mathbf{z} - h(\mathbf{x}^-)$ | `innovate`, `Innovation.innovation` | — |
| Innovation covariance | $S$ | $S = HP^-H^\mathsf{T} + R_z$ | `Innovation.innovation_covariance` | — |
| Normalised innovation squared (NIS) | $d^2$ | $d^2 = \boldsymbol{\nu}^\mathsf{T} S^{-1} \boldsymbol{\nu}$ | `Innovation.nis` | — |
| Gate | $\gamma$ | accept when $d^2 \le \gamma = \chi^2_m(P_G)$ | `chi_square_gate_threshold`, `gate` | 8.81 (range), 11.62 (range and rate), $P_G$ = 0.997 |
| Kalman gain | $K$ | $K = P_{xz}S^{-1}$, with $P_{xz} = P^-H^\mathsf{T}$ | `update` | — |
| Update | $\mathbf{x}^+$, $P^+$ | $\mathbf{x}^+ = \mathbf{x}^- + K\boldsymbol{\nu}$; $P^+$ below | `update` | — |
| Normalised estimation error squared (NEES) | $\epsilon$ | $\epsilon = \mathbf{e}^\mathsf{T} P^{-1} \mathbf{e}$, $\mathbf{e} = \mathbf{x}_{\mathrm{true}} - \hat{\mathbf{x}}$ | `metrics.nees` | inside $\chi^2_2$'s 95 % interval in 0.89 of the confirmed scans |
| Confirmation | M of N | M hits in a track's first N scans | `LifecyclePolicy` | 3 of 5 |

Symbols used only on this page: $R_b$ and $\dot R_b$, bistatic range and its rate;
$\sigma_a$, the standard deviation of the acceleration of $R_b$; $\tau$, how long one
acceleration lasts; $\mathbf{z}$, a measurement; $h$, the measurement model, with $H$ its
Jacobian; $P_{xz}$, the cross-covariance of state and measurement; $m$, the measurement's
dimension; $P_G$, the gate probability.

The motion model gives $F$ and $Q$:

$$F = \begin{bmatrix} 1 & T \\ 0 & 1 \end{bmatrix}, \qquad
Q(T) = q\begin{bmatrix} T^3/3 & T^2/2 \\ T^2/2 & T \end{bmatrix}$$

This is continuous white-noise acceleration (`"continuous"`). The off-diagonal terms are positive
because the rate is the range's derivative: a receding target's range grows. Discrete
white-noise acceleration (`"discrete"`), which scenario 003's Kalman path uses, holds one
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

`scan_tracker.ScanTracker(scenario)` builds a `Tracker` from `[tracking]` (`linearisation`,
`white_noise`, `initiator`, `cost`, `assignment`, `association`, the lifecycle counts) and
feeds it each scan ([below](#how-a-scan-reaches-the-tracker)). For an IQ scenario such as 003
the motion model is `ConstantVelocity.bistatic_range`. Range is measured modulo the map's
period, 149,896 m for S1, while the state stays continuous. The rate is measured only where it
cannot fold: after dual-PRF pairing, from one burst whose rate interval covers
`max_bistatic_range_rate_mps`, or once a track can unfold it
(`rate_unfolding = "track_aided"`); otherwise range alone. Track-aided unfolding lets a track
unfold once it holds N associated ranges, the least N whose least-squares slope is certain to a
`unfold_sigma_gate`-th of the rate period:

$$\frac{\sigma_R}{T}\sqrt{\frac{12}{N(N^2-1)}} < \frac{v_{\mathrm{period}}}{g}$$

with $v_{\mathrm{period}}$ the rate period and $g$ the gate. On S1, $\sigma_R$ = 43.27 m,
$v_{\mathrm{period}}$ = 30.591 m/s and $g$ = 6 give N = 10. Of the tracks that may unfold, the
one with the longest range history then gives each plot the fold nearest its rate, unless that
lies over half a period from the slope of its last `n_slope_scans` ranges (the **fold monitor**):
a wrong fold agrees with the prediction that chose it, never with the range.

### How a scan reaches the tracker

`ScanTracker.process` takes one scan and returns one `(bank, association)` pair for each batch
it ran, in order; `association` holds a weighting associator's clusters, and is None for GNN,
MHT and PMBM. What a batch is depends on how the radar looks at the scene:

- **Staring**, an IQ scenario with no `[sensor]`: the radar sees the whole scene in one scan,
  so `process(plots, time_s)` makes the scan's accepted `Plots` one `Scan` at the scan's time.
  The state is bistatic range and its rate, and every plot carries the map's uniform clutter
  density: one false plot a scan over the range period and the rate span a plot can take.
- **Rotating**, a plot-level scenario with a `[sensor]` (scenario 011): the beam crosses each
  azimuth at its own time, so `process(plot_scan)` cuts the `PlotScan` into `n_sectors`
  batches by azimuth, each a `Scan` at its sector's centre time, each sector's tracks the next
  one's prior. The `Tracker`'s `coverage`, here `ScanTracker`'s sector test, hands a batch only
  the tracks whose azimuth, extrapolated to its time, lies in its sector: a track outside is
  neither predicted, gated nor charged a miss, and waits at its own time for the sector that
  sees it (`tracking.tracker.Coverage`). The state is ENU position and velocity on the ground
  plane, z is (horizontal range, azimuth), and each plot carries the clutter density at it:
  the scene's (`clutter_density_source = "oracle"`) or a `ClutterMap`'s (`"map"`).
  `n_sectors = 1` makes a rotating scan one batch, as `scenario_011_cst_sweep.toml` does.

MATLAB delivers a rotating radar's scan the same way, a field of view at a time, and sets
`IsScanDone` at the end of the rotation.

### Choosing the associator and the motion model

`[tracking]` chooses both, for either kind of scenario. `ScanTracker` refuses when built what it
cannot run (JPDA or MHT beside a range-only fallback; PMBM or `seed_truth` on an IQ scenario),
rather than on the first scan; an IQ scenario's `track` stage does not read `motion` or
`trackers`:

| Choice | `[tracking]` keys | Staring (IQ) | Rotating (plots) |
| :-- | :-- | :-- | :-- |
| GNN | `association = "gnn"`, the default; with `detection_probability`, a pair costlier than ln(P_D / (1 − P_D P_G)) is left unassigned | yes | yes |
| JPDA, JPDA*, set JPDA | `association = "jpda"`, `coalescence = "none"`, `"jpda_star"` or `"set"`, and `detection_probability` | with one measurement model a scan | yes |
| JIPDA | as JPDA, with `existence = true` and `survival_probability` | as JPDA | yes |
| MHT | `association = "mht"`, `detection_probability`, `k_best`, `n_scan_depth`, `confirm_score`, `delete_drop` | as JPDA | yes |
| PMBM | `trackers = [..., "pmbm"]` beside the configured tracker, or `ScanTracker(scenario, "pmbm")`; `detection_probability`, `birth_intensity_per_scan`, `survival_probability`, `prune_existence`, `prune_weight` | `ScanTracker(scenario, "pmbm")` refused; `trackers` not read | yes, on one CV model |
| IMM | `motion = "imm"`: modes CV quiet, constant turn and CV agile, `mode_sigma_acceleration_mps2`, `sojourn_time_s`, `turn_rate_noise_density_rad2ps3`, `turn_rate_sigma_degps` | not read: the state is range and rate | yes, under GNN, JPDA or MHT; PMBM takes the agile mode alone |

JPDA and MHT weigh or branch on one measurement model a scan. An IQ scenario offers two where
a track may fall back to range alone, with `range_only_retry = true` or
`rate_unfolding = "track_aided"` as in both `scenario_003_tracking*.toml` files; there
`ScanTracker` refuses them when built. The three `scenario_003_ukf_*` files offer one, and take
them. For example, two lines added to the `[tracking]` table of
`config/scenario_003_ukf_fmcw_dual_prf.toml`:

```toml
association = "jpda"
detection_probability = 0.9
```

A copy of the file with these lines passes all 11 of its checks over the 120-scan window, as
GNN does: a range RMSE of 5.05 m under both, and a `nees_fraction` of 0.856 against 0.864.
`tests/test_scan_tracker.py` confirms the target's track under them.

A rotating scenario sets the same keys beside its motion model. This is the `[tracking]` table
of `config/scenario_011_terminal_dense.toml`, cut to the keys that choose: an IMM under JIPDA
with a clutter map, and GNN, MHT and PMBM scored beside it on the same plots, each score
prefixed by its tracker's name:

```toml
motion = "imm"
mode_sigma_acceleration_mps2 = [0.5, 0.5, 3.1623]
sojourn_time_s = [120.0, 30.0, 20.0]
association = "jpda"
coalescence = "none"
existence = true
detection_probability = 0.90
clutter_density_source = "map"
trackers = ["gnn", "mht", "pmbm"]
```

`uv run python scripts/run_scenario.py config/scenario_011_terminal_dense.toml --scans 10`
runs all four in about 40 s and passes the checks a ten-scan run can make.

### Choosing the process noise

Real targets don't move at exactly constant velocity. The **process noise** says how far they
may stray from it, given as two physical numbers:

- `sigma_acceleration_mps2`, $\sigma_a$: the standard deviation of the acceleration of
  bistatic range, twice the radial one when monostatic. A largest acceleration `a_max` read as
  a 3σ bound gives $\sigma_a$ = `a_max` / 3.
- `acceleration_correlation_time_s`, $\tau$: roughly how long one acceleration lasts.

`"continuous"` turns them into $q = 2\sigma_a^2\tau$ and builds $Q(T)$ from the actual step
at every prediction ([the equations above](#quantities-and-equations)). That $Q$ is exact for
any step under continuous white-noise acceleration; $q = 2\sigma_a^2\tau$ itself is Singer's
intensity, an approximation to a correlated acceleration that holds for steps long against
$\tau$. `"discrete"` reads $\sigma_a$ alone. The two add equal velocity variance per step only when
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
from radar_forge.tracking.initiation import DirectStateInitiator
from radar_forge.tracking.lifecycle import LifecyclePolicy
from radar_forge.tracking.measurement import LinearMeasurement
from radar_forge.tracking.motion import ConstantVelocity

# State and measurement: bistatic range and its rate, positive receding.
motion = ConstantVelocity.bistatic_range(8.0)  # sigma_a of bistatic range, m/s^2
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
from radar_forge.tracking.motion import ConstantVelocity

# The UKF on S1: [tracking] gives sigma_a = 8 m/s^2 of bistatic range, tau = 1 s and
# continuous white noise.
scenario = load_scenario("config/scenario_003_ukf_fmcw_low_prf.toml")
settings = scenario.tracking
motion = ConstantVelocity.bistatic_range(
    settings.sigma_acceleration_mps2,
    acceleration_correlation_time_s=settings.acceleration_correlation_time_s,
    white_noise=settings.white_noise,
)
step_s = scenario.scan_period_s
print("F(1 s) =", motion.jacobian(np.zeros(2), step_s).tolist())
print("Q(1 s) =", motion.process_noise(np.zeros(2), step_s).round(3).tolist())

# Measurement noise: one bin of the map's own axes over sqrt(12).
range_axis_m, rate_axis_mps = map_axes(scenario.radar, scenario.n_pulses_per_burst[0])
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
from radar_forge.tracking.measurement import LinearMeasurement
from radar_forge.tracking.motion import ConstantVelocity

# 200 runs of one UKF as the 200 rows of one bank, measuring range alone.
motion = ConstantVelocity.bistatic_range(8.0)
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

Run the UKF on S1 through the stepper, over its whole 120-scan window:

```bash
uv run python scripts/run_scenario.py config/scenario_003_ukf_fmcw_low_prf.toml
```

`[track]` prints its scores of the target's primary track: `range_rmse_m = 15.1152`,
`range_rate_rmse_mps = 12.0151`, `nees_fraction = 0.889831`, `confirmation_latency_scans = 2`
and `n_confirmed_tracks = 1`, and all 11 checks pass. The one track is confirmed in 118 of the
120 scans. `runs/scenario_003_ukf_fmcw_low_prf/` holds `tracks.csv` (one row per track per
scan, with its NIS and the detection it took), `metrics.csv` and `track.png`. A run shorter
than 2 × `n_confirm_scans` scans, 10 here, skips the tracker's checks.

**Knob:** in a copy of the TOML in `config/`, set `sigma_acceleration_mps2 = 0.6` in
`[tracking]` (0.3 m/s² radial: the filter now believes the aircraft barely accelerates), and run
it again. The filter trusts its prediction too much, so whenever the aircraft's rate changes the
track falls behind until its plots land outside the gate. `[track]` now counts **three**
confirmed tracks where there was one: the target's track broke and restarted. The primary track
is confirmed in only 63 scans, its range RMSE grows to 77.2 m and its rate RMSE to 16.6 m/s,
and the copy ends `2 check(s) failed`, on `range_rate_rmse_mps` and on `nees_fraction` (0.16).
`nees_fraction` is the share of the primary track's confirmed scans whose NEES lies inside
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
  `figures.display_axes`; the bank never is.
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
  needs a range slope certain to a sixth of the rate period first, which takes N = 10 ranges
  ([the scenario's tracker](#the-scenarios-tracker)).

## Limits

**Built, but not walked through on this page.** The page walks through the single-target
tracker of scenario 003. The many-target methods of scenario 011 (the IMM and `ConstantTurn`,
PDA and the JPDA family, Murty's k best, track-oriented MHT, PMBM, confirmation by existence,
GOSPA and trajectory GOSPA, and coverage) are configured as
[above](#choosing-the-associator-and-the-motion-model); their docstrings and
`spec/tracker-001.md` describe them. An IQ scenario takes JPDA or MHT only with one
measurement model a scan, and PMBM and the IMM not at all.

**Not yet supported:**

- an initiator for `BistaticRangeAndRangeRate`: one bistatic measurement cannot fix a position, so
  `DirectStateInitiator` returns None for it. Add its tracks with `Tracker.seed`.

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
