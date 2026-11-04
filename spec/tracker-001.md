# Tracker 001 — Tracking architecture

> The tracker after the tracking refactor: one Gaussian filter, a batched track bank, one
> correction core, one scan loop. Running it is in [the user guide](../docs/user-guide/06-tracking.md).

## 1. Scope and principles

In scope: `core/tracking/`: one Gaussian filter (Kalman, extended and unscented are its
transforms), motion and measurement models, gating, assignment, M-of-N confirmation with
covariance-based deletion, and the interface the pipelines use. Out of scope: random-finite-set
trackers inside the loop, track-to-track fusion, real-time operation.

1. **Models are data; the filter is generic.** A new sensor or target needs a new model, never a
   new filter.
2. **One implementation of each equation, and one flow.** Two copies drift.
3. **Names, not positions.** Code reaches a state component through `StateLayout` by name
   (`"x_m"`, `"xdot_mps"`).
4. **Range rate is the time derivative of range, positive receding** (`spec/data-001-formats.md`
   §5); Doppler is `-rate/wavelength`. A sign slip passes every test that uses one sign, so tests
   use both.
5. **Covariances stay symmetric and positive definite**; every update symmetrises.
6. **Consistency is tested statistically** (NEES and NIS against chi-square bounds, fixed seeds),
   and ground truth is analytic (`docs/conventions/testing.md`).
7. **Readable and fast, neither traded**: the hot path is batched NumPy that still reads as the
   textbook equation, with shapes written out.

## 2. Architecture

Pieces behind small protocols, close to [Stone Soup](https://stonesoup.readthedocs.io)'s cut.
Predictor and updater are one `Estimator` here, and the hypothesiser is the gate.

| Piece | Type | Knows about |
| :--- | :--- | :--- |
| State layout, estimate | `StateLayout`, `Coordinate`, `StateEstimate` | names, units, periods; mean, covariance, time |
| Motion model | `MotionModel` | layout, `f(x, dt)`, `Q(dt)` |
| Measurement, scan | `Measurement`, `Scan` | z, R, time, sensor, model ID; one revisit at one time |
| Measurement model | `MeasurementModel` | layouts, `h(x)` |
| Filter, transform | `GaussianFilter`; protocol `Linearisation`, `TaylorLinearisation`, `UnscentedTransform` | a motion model; a function, a mean, a covariance |
| Gate, associator | `ChiSquareGate`; `GreedyNearestNeighbour`, `GlobalNearestNeighbour` | innovation statistics; a cost matrix |
| Initiator, lifecycle | `TrackInitiator`, `LifecyclePolicy` | a prior; hits, misses, times, covariance |
| Track bank | `TrackBank`, `TrackSnapshot` | batched means and covariances, status, counts |
| Tracker | `Tracker` | all of the above, through their protocols |

**Decoupling.** Swapping the transform (KF, EKF to UKF), the associator or the initiator changes
one constructor argument and no pipeline code. A branch on the type of a transform, filter or
tracker in the filter, the tracker or `pipelines/` breaks this. There is one `Tracker`.

**Dependencies.** `core/tracking/` imports from `core/` only, never `pipelines/`; pipelines build
the `Measurement`s and the `Scan`.

## 3. The filter

The KF, EKF and UKF are one algorithm, the general Gaussian filter (Särkkä 2013, ch. 6): match the
mean and covariance of the density, then apply the same Kalman correction. They differ only in how
a mean and covariance are carried through a function.

| Transform | Gives | Moments of g(x), x with mean x̂ and covariance P | Supplies H |
| :--- | :--- | :--- | :--- |
| `TaylorLinearisation` | the KF on a linear model, the EKF otherwise | mean g(x̂), covariance J P Jᵀ, J = ∂g/∂x at x̂ | yes |
| `UnscentedTransform(alpha, beta, kappa)` | the UKF | weighted sums over 2n + 1 sigma points | no |

The protocol `Linearisation` has `predict(motion, state, dt_s)` (mean `(n_state,)` and covariance
`(n_state, n_state)` with Q added), `measure(model, state)` (ẑ `(n_meas,)`, P_xz `(n_state, n_meas)`,
S without R `(n_meas, n_meas)`, and H `(n_meas, n_state)` or `None`) and `check_motion_model(motion)`,
which refuses a bad pairing at construction. `GaussianFilter` holds an estimate, a motion model and
a transform; `predict_to` calls `predict`; `innovation_statistics` calls `measure`, adds R and
returns ν, S, NIS and the log-likelihood from the one `innovation_stats` (Cholesky-whitened);
`update` hands the same moments to the correction core. It never branches on the transform type.
A cubature or Gauss-Hermite transform is a new `Linearisation`, never a new filter.

**The correction core** is one private function shared by every filter:

1. gain K = P_xz S⁻¹ by solving S Kᵀ = P_xzᵀ, never inverting S;
2. mean x̂ = x̂⁻ + K ν, wrapping only the state layout's periodic components;
3. covariance by the Joseph form (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ when H is supplied, else the short
   form P⁻ − K S Kᵀ;
4. the symmetric part of P, returned as a new `StateEstimate`.

The Joseph form is positive semidefinite for any K. The short form (UKF) is the Schur complement
of S in a positive semidefinite joint covariance, so it is too, up to roundoff, provided every
sigma-point covariance weight is non-negative. The two agree at the optimal gain. The core's only
branch is whether H is `None`.

**Two layouts, two wraps.** ν = z − ẑ goes through the measurement layout's `residual`, so a
periodic measurement (an azimuth, a range folded at the map's span) takes the short way round; the
mean update wraps only the state layout's periodic components, so a folded range stays continuous
in the state.

**Taylor.** Needs F and Q from a motion model with `matrices(dt_s)`, and H or J from a measurement
model with `jacobian(state)`. A model without them raises `TypeError` naming it; there is no
finite-difference fallback, because a numeric Jacobian hides a wrong analytic one. On a linear
model it is the textbook KF line for line.

**Unscented** (Wan and van der Merwe 2000). With c = α²(n + κ) the weights are w₀ᵐ = 1 − n/c,
w₀ᶜ = w₀ᵐ + 1 − α² + β and wᵢ = 1/(2c) for the other 2n points. Settings with w₀ᶜ < 0 are rejected,
because the short form needs non-negative covariance weights; a small α such as 10⁻³ is refused on
purpose. Sigma points straddling a wrap are averaged and differenced the short way round through
the layout's `weighted_mean_about` and `residual`.

**On a linear model the UKF is the KF.** The ± sigma pairs cancel through a linear map, so the mean
is F x̂; the centre point's deviation is zero, which removes β and α from the covariance, and the
other 2n points give Σᵢ F Lᵢ Lᵢᵀ Fᵀ = F P Fᵀ (Lᵢ the columns of the Cholesky factor). The weights
matter only once g bends.

**Drift guards.** One parametrised fixture holds every estimator, so a new transform joins every
guard once. UKF and KF moments agree to `rtol=1e-10` on linear models for several accepted
(α, β, κ), per step and over 200 steps; EKF and UKF on a nonlinear model differ at second order in
the prior scale s (fitted slope of log gap on log s at least 1.8); NEES and NIS pass a chi-square
interval at 99.9 %; covariances stay symmetric to `rtol=1e-12` with Cholesky succeeding.

## 4. State and measurement models

- **Cartesian state**: per-axis interleaved `[x_m, xdot_mps, y_m, ydot_mps, z_m, zdot_mps]` in
  local ENU (x east, y north, z up); other orders via `CartesianMotion(order=...)`.
- **Radial state**: `RadialConstantVelocity`, `[range_m, range_rate_mps]`, transition
  `[[1, T], [0, 1]]`. This is scenario 003's `range_1d`.
- **Process noise**: `"cwna"`, continuous white-noise acceleration (q in m²/s³; Bar-Shalom, Li and
  Kirubarajan 2001, §6.2.2), is the default because its Q is right for any step length. `"dwna"`,
  the discrete model (σₐ² in m²/s⁴, §6.3.2), is kept because scenario 003 pins it.
- **State models are configurations**: `range_1d`, `enu_2d`, `enu_3d` are named pairs of a motion
  and a measurement model, built by one function.
- **Spherical measurements stay spherical.** `RangeAzimuthElevationRangeRate` maps an ENU state to
  `(range_m, azimuth_rad, elevation_rad, range_rate_mps)` about a sensor pose: r = |p − pₛ|,
  azimuth atan2(Δe, Δn) clockwise from north (periodic), elevation asin(Δu/r), range rate
  +(p − pₛ)·v/r. The geometry is one function shared with `core.geodesy`; the model raises rather
  than return a non-finite Jacobian at r → 0 or the zenith. Converting to Cartesian is allowed only
  to seed a track, with the measurement-conditioned conversion of Duan, Han and Li (2004) and
  Bordonaro, Willett and Bar-Shalom (2014).
- **Range rate is measured, not stated.** It is a row of z and h(x), so it enters S and the gate:
  a false alarm matching in position but not Doppler fails. A Cartesian state carries no range-rate
  component (redundant with velocity, and it makes the covariance singular). The pipeline unfolds a
  folded Doppler first; a track that cannot yet trust its unfolding uses range alone (§5.4).
- **Bistatic**: `BistaticRangeRateModel` maps an ENU state to `(bistatic_range_m,
  bistatic_range_rate_mps)` (or excess bistatic range) about transmitter and receiver poses. One
  bistatic measurement cannot fix a position, so seed such tracks with `Tracker.seed`.

## 5. The scan loop

`Tracker.process(scan)` runs gate, assign, update, initiate, lifecycle.

### 5.1 Gate and assign

`ChiSquareGate(probability)` accepts a pair when its NIS is at most the chi-square quantile for the
measurement dimension. `GlobalNearestNeighbour` builds an (n_tracks, n_measurements) cost matrix,
forbids pairs outside the gate and solves it with `scipy.optimize.linear_sum_assignment` (Crouse
2016), maximising the number of matches before minimising cost. `GreedyNearestNeighbour` is the
greedy version, for teaching.

**Costs across dimensions.** With the range-only retry of §5.4 one matrix can mix pairs scored in
different dimensions, and neither NIS nor the negative log-likelihood is comparable across them (the
density's value depends on the units of what it measures). So then the cost is
`cost="negative_log_likelihood_ratio"`: −ln(g/λ), with g = 𝒩(ν; 0, S) and λ the clutter density over
the same components (the per-update increment of Blackman's 2004 track score).
`Tracker(clutter_density=...)` takes λ per model ID, required for every model under this cost,
refused under any other, positive and finite; the values are the caller's. `cost="nis"` is allowed
only when every pair has one dimension, and `Tracker` raises rather than mix them.

**Assignment order** is `assignment: Literal["confirmed_first", "joint"]`. `"confirmed_first"`
(default) lets confirmed tracks pick first and starts no track inside a confirmed gate; `"joint"`
solves once. On scenario 003 confirmed-first was worse on every count (range RMSE 25.3 m against
17.5 m), because a widened re-acquisition gate outbids tentative tracks for false alarms, so that
scenario sets `"joint"`. Changing a default needs a measurement.

### 5.2 Update and initiate

Each assigned pair updates through the correction core. An unassigned measurement may seed a track
through `TrackInitiator`, which takes a prior and an estimator factory (a callable from a prior
`StateEstimate` to an `Estimator`, with the motion model and transform bound in).

### 5.3 The track bank

`TrackBank` holds the live tracks of one layout as arrays: means `(n_tracks, n_state)`, covariances
`(n_tracks, n_state, n_state)`, and per-track status, hit and miss counts and times. Predict, gate
and update act on the whole bank; tracks of different layouts live in separate banks.
`TrackSnapshot` is the read-only view of one track for `tracks.csv`: `mean`, `covariance`, status,
counts, the measurement index taken this scan and that update's NIS.

### 5.4 Lifecycle

`tentative → confirmed ⇄ coasting → deleted`. One `LifecyclePolicy` holds every setting; pipeline
configs hold one and never copy its fields. A *hit* is a scan in which a detection was assigned to
the track, a *miss* one in which none was.

| Rule | Setting |
| :--- | :--- |
| M-of-N confirmation, counted from birth | `n_confirm_hits`, `n_confirm_scans` |
| Tentative deletion once M is unreachable | (derived) |
| Deletion after misses in a row; after a coast time | `n_delete_misses`; `max_coast_time_s` |
| Re-acquisition window for confirmed tracks | `n_reacquire_scans` |
| Covariance-based deletion | `max_position_sigma_m` |
| Existence-based confirm and delete (JIPDA) | `ExistencePolicy(confirm, delete)` |

**Covariance-based deletion.** Delete when the square root of the largest eigenvalue of the
position covariance exceeds `max_position_sigma_m`, whatever the miss count. Miss counts are proxies
for "we no longer know where it is"; the covariance is the thing itself. `None` turns it off.
(Stone Soup's `CovarianceBasedDeleter` thresholds the trace, which mixes units.)

**Range-only bootstrap, retry and reset.** Mechanism in core, policy in the scenario. A
measurement model can return its marginal over named components. The pipeline passes a selector (a
callable from a `TrackSnapshot` and a `Measurement` to the ordered model IDs to try); the tracker
gates against them in order and keeps the first that passes, which gives the range-only bootstrap
and the range-only retry when the velocity gate fails. On the miss that starts re-acquisition, the
coordinates named in `LifecyclePolicy.reacquire_reset` (scenario 003: `("range_rate_mps",)`) have
their variances reset to the prior and their cross-covariances zeroed. The default selector returns
the measurement's own model.

## 6. Extension points

Specified by [`scenario-011-terminal-area-tracking.md`](scenario-011-terminal-area-tracking.md),
not yet built; none changes the loop.

- **IMM** (Blom and Bar-Shalom 1988) is an `Estimator` of `Estimator`s. Its state is the
  mode-weighted mixture in the union layout of its modes, a coordinate a mode lacks being added by
  name at zero mean on mixing. The transition matrix is `expm(Λ T)` for the actual interval.
- **JPDA** (Fortmann, Bar-Shalom and Scheffe 1983; JPDA\*, Set-JPDA, JIPDA) fills optional weights
  `(n_tracks, n_measurements + 1)` on `AssociationResult`, applied through the correction core;
  clusters are enumerated exactly, with a bound that raises.
- **Track-oriented MHT** (Reid 1979; Blackman 2004): `TrackTree` satisfies `Estimator`;
  `TrackOrientedMHT` is an associator using Murty's k-best assignment with N-scan pruning. With
  `n_scan = 1` and `k_best = 1` it reduces to GNN, its first test.
- **PMBM** (García-Fernández et al. 2018) cannot sit inside `Tracker`: hypotheses, existence and
  birth are part of its density, leaving no associator, initiator or lifecycle to swap. It is a
  comparison tracker in `core/tracking/rfs/pmbm.py`, sharing the models, `innovation_stats`,
  Murty's algorithm, `clutter_density` and the metrics.

## 7. Integration, performance and validation

**Data flow.** `core.detection.Detection` (cell indices, W) is converted by `pipelines/tracking.py`
to the pipeline's detection report (m, m/s, positive receding), built into a `Scan` of
`Measurement`s, passed to `Tracker.process(scan)`, and returned as one `TrackSnapshot` per live
track plus the scan's association. `scripts/run_scenario.py` writes `tracks.csv` and
`detections.csv` (`spec/data-001-formats.md` §6.5, §6.6). Settings come from the scenario TOML's
`[tracking]` table, read once by the pipeline; scenario-only settings stay in `pipelines/`. Every
measurement in a scan carries the scan's time and scans arrive in order. Every sensor registers in
one `Tracker`, so one target has one track whichever sensor sees it.

**Performance.** The cost of a scan is the gate. The (n_tracks, n_measurements) gate and cost
matrices come from one batched computation, not a loop over pairs: Taylor Jacobians as `(n_tracks,
n_meas, n_state)`, sigma points as `(n_tracks, 2 n_state + 1, n_state)`. `np.linalg.cholesky` and
`solve` take stacks over leading axes, so batched functions keep the textbook names. Gating is
O(n_tracks · n_measurements · n_meas³); assignment is O(n³). A loop off the hot path carries a
comment saying why. No compiled dependency in core; a backend comes only when a benchmark shows
NumPy is the limit, and then as an extra.

**Validation.** Analytic (dead reckoning, Q against the published matrices, a known GNN optimum,
M-of-N by hand, Jacobians against central differences, `rtol=1e-12`); consistency (NEES and NIS over
200 runs × 30 steps, every transform, Bar-Shalom, Li and Kirubarajan 2001 §5.4.2); the drift guards
of §3; hand-countable lifecycles; the batched gate against `innovation_stats` pair by pair
(`rtol=1e-12`); `benchmark` tests holding a scan of scenario 003's load and of 100 tracks × 1000
measurements to a committed budget; scenario 003 end to end; OSPA (Schuhmacher, Vo and Vo 2008) and
the metrics of `spec/data-001-formats.md` §6.9. Never loosen a tolerance to pass.

## 8. Decisions

- **One tracker, one Gaussian filter, one correction core.** Rejected: two filter classes sharing
  a core (two flows, which is where they drift); a filter-type switch (breaks decoupling); a KF
  built as a UKF (hides the KF a student looks for); Joseph form everywhere (the UKF has no H).
- **Spherical measurements stay spherical.** Rejected: Cartesian conversion then a linear KF
  (state-dependent covariance, bias growing with range × angle error).
- **Mechanism in core, policy in pipelines** (the range-only selector), not in `Tracker`.
- **D26: `CoordinatedTurn`, with the turn rate in the state.** One `MotionModel` in `motion.py`,
  exact discretisation with a series form near zero turn rate.
- **D27: `IMM` is an `Estimator` of `Estimator`s.** Mixing by name in the union layout; transition
  matrix `expm(Λ T)` for the actual interval.
- **D28: JPDA, JPDA\*, Set-JPDA and JIPDA fill `AssociationResult.weights`.** Exact events per
  cluster with a bound that raises; clutter intensity from an oracle or a clutter map; JIPDA's
  existence drives an `ExistencePolicy`.
- **D29: Track-oriented MHT, the tree is the estimator.** `TrackTree` satisfies `Estimator`;
  `TrackOrientedMHT` is an associator using Murty's k-best; the one loop stands.
- **D30: PMBM is a comparison tracker beside `Tracker`.** It shares the models and metrics and owns
  its density; it is the reference score, not a fourth associator.

## 9. Limitations

Out-of-sequence measurements are rejected, not buffered. GNN gives a track at most one measurement
per scan, so one target giving several detections starts several tracks. A target crossing the
line of sight is lost in the Doppler blind zone. There is no bistatic initiator or Singer model.

## References

- Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and
  Navigation*, Wiley, 2001: §5.4.2 (consistency), §6.2.2, §6.3.2 (white-noise acceleration),
  §10.3 (EKF), ch. 11 (IMM).
- S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013, ch. 5-6.
- E. A. Wan and R. van der Merwe, "The unscented Kalman filter for nonlinear estimation," *Proc.
  IEEE AS-SPCC*, 2000, pp. 153-158.
- S. S. Blackman, "Multiple hypothesis tracking for multiple target tracking," *IEEE Aerosp.
  Electron. Syst. Mag.*, 19(1), 2004, pp. 5-18.
- D. F. Crouse, "On implementing 2D rectangular assignment algorithms," *IEEE Trans. Aerosp.
  Electron. Syst.*, 52(4), 2016, pp. 1679-1696.
- Z. Duan, C. Han and X. R. Li, *IEEE Trans. Aerosp. Electron. Syst.*, 40(4), 2004, pp. 1374-1377;
  S. Bordonaro, P. Willett and Y. Bar-Shalom, *IEEE Trans. Aerosp. Electron. Syst.*, 50(2), 2014,
  pp. 1431-1444 (converted measurements).
- H. A. P. Blom and Y. Bar-Shalom, *IEEE Trans. Autom. Control*, 33(8), 1988, pp. 780-783 (IMM);
  T. E. Fortmann, Y. Bar-Shalom and M. Scheffe, *IEEE J. Oceanic Eng.*, 8(3), 1983, pp. 173-184
  (JPDA); D. B. Reid, *IEEE Trans. Autom. Control*, 24(6), 1979, pp. 843-854 (MHT).
- Á. F. García-Fernández, J. L. Williams, K. Granström and L. Svensson, *IEEE Trans. Aerosp.
  Electron. Syst.*, 54(4), 2018, pp. 1883-1901 (PMBM).
- D. Schuhmacher, B.-T. Vo and B.-N. Vo, *IEEE Trans. Signal Process.*, 56(8), 2008, pp. 3447-3457
  (OSPA).
