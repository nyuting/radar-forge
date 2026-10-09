# Tracker 001 — Tracking architecture

> **Reading order.** §1–§3 say what the tracking package is for and how it is cut into pieces.
> §4 answers the question this spec was written for: how the Kalman filter and the UKF live in
> one package without drifting apart. §5–§10 specify each piece and how it meets the rest of
> radar-forge. §11–§13 cover performance, validation and the path from today's code. §14 holds
> the decisions, §15 the acceptance criteria.
>
> | Part | Sections |
> | :--- | :--- |
> | Purpose and principles | [§1](#1-status-purpose-and-scope), [§2](#2-what-a-good-tracking-module-is) |
> | Architecture | [§3](#3-architecture-strict-decoupling), [§4](#4-filters-kf-ekf-and-ukf-behind-one-interface) |
> | The pieces | [§5](#5-state-and-motion-models) to [§9](#9-multiple-models-imm) |
> | Integration | [§10](#10-integration-with-the-rest-of-radar-forge) |
> | Performance, validation, migration | [§11](#11-performance) to [§13](#13-migration-from-todays-code) |
> | Decisions and acceptance | [§14](#14-decisions), [§15](#15-acceptance-criteria) |

## 1. Status, purpose and scope

**Status: specified, partially implemented.** Most of the pieces in §3 exist in
`src/radar_forge/core/tracking/`. What does not yet hold is the "one tracker" end state of §4,
and §13 lists the steps to it. Where this spec and the code disagree, the code is behind and
this spec is the target, except where §14 says otherwise. Step 9 of §13 also holds the refactor-002 audit of this package, deferred until
the migration is done.

This spec covers `core/tracking/`: filters, motion and measurement models, gating, association,
track initiation and track lifecycle, and the interface they offer the pipelines. How to run the
tracker, and how to choose its process noise, are in the user guide,
[`docs/tracking/README.md`](../docs/tracking/README.md). This spec does not repeat them.

### 1.1 Scope

**In scope.**

- One scan-loop tracker, `Tracker`, used by every pipeline.
- Three interchangeable filters behind one interface: the linear Kalman filter (KF), the
  extended Kalman filter (EKF) and the unscented Kalman filter (UKF).
- Cartesian motion models, and monostatic and bistatic radar measurement models, including the
  spherical-to-Cartesian geometry.
- Chi-square gating, global nearest-neighbour (GNN) and nearest-neighbour (NN) assignment.
- M-of-N track confirmation, miss-count, coast-time and covariance-based deletion.
- Extension points for the interacting multiple model (IMM) filter and for joint probabilistic
  data association (JPDA).

**Out of scope.**

- Implementing IMM and JPDA. Each waits for the first scenario that needs it (§9, §7.4).
- Random-finite-set trackers such as PMBM. §7.5 says why.
- Track-to-track fusion across sensors. One `Tracker` already takes every sensor's
  measurements, which is measurement-level fusion; fusing tracks from separate trackers is
  a later spec.
- Real-time operation. §11 bounds the performance requirement.

### 1.2 Provenance

The UKF stack in `core/tracking/` comes from `unified-extensible-tracker`, an earlier tracker
written by Ng Zeming Michael Eugene, who contributes it to radar-forge under the repository's
MIT licence ([`LICENSE`](../LICENSE)). The linear Kalman stack was written for scenario 003
(`spec/scenario-003-tracking.md`). Where the code implements a published method, the module or
class docstring cites the source in its `References` section. This section replaces the former
`spec/tracker-001-provenance.md`.

### 1.3 Where the requested features' premises do not match the code

This spec was asked to rule on seven candidate requirements, and §14.2 rules on each. Two of
them were phrased in a way this spec corrects, and the corrections win over the original wording:

- **"Doppler integration in the state vector."** Range rate is not a state component of a
  Cartesian tracker. The state carries the velocity vector, and range rate is one row of the
  *measurement*, a nonlinear function of position and velocity (§6.2). Using range rate in the
  gate does not need it in the state: it needs it in z and in h(x). For `range_1d`, the state
  is range and range rate, so the two coincide.
- **"GNN via Hungarian/LAPJV."** `scipy.optimize.linear_sum_assignment` already solves the
  assignment, and is a modified Jonker–Volgenant algorithm (Crouse 2016). No new dependency
  and no hand-written Hungarian algorithm are needed.

## 2. What a good tracking module is

A good tracking module is one an intern can read, trust and extend. Each principle below is a
requirement, and has a reason.

1. **Models are data; the filter is generic.** A filter knows nothing about radars, axes or
   units. It is handed a motion model and a measurement model. *Why:* a new sensor or target
   then needs a new model, never a new filter, and the filter is tested once.
2. **One implementation of each equation.** The Kalman gain, the covariance update, the NIS
   and the angle wrap each exist in exactly one function. *Why:* two copies drift. Today's two
   copies of the covariance update already argue opposite things in their docstrings (§4.2).
3. **Every state component has a name and a unit.** Code reaches a component through
   `StateLayout` by name (`"x_m"`, `"xdot_mps"`), never by position. *Why:* positional indexing
   silently breaks when the order changes. `KalmanTracker._forget_rate` assumes positions come
   first, and is wrong for the UKF stack's interleaved order.
4. **SI units, closing-positive range rate.** Every quantity is SI internally, with its unit in
   its name. Range rate is positive for a closing target, everywhere
   (`spec/data-001-formats.md` §5, DF7). *Why:* it is the repo's convention, and a sign flip in
   range rate passes every test that uses only one sign.
5. **Covariances stay symmetric and positive definite, and tests prove it.** Every update
   ends with an explicit symmetrisation; long-run tests check definiteness. *Why:* a
   non-positive-definite covariance fails the Cholesky in the next gate, far from its cause.
6. **Consistency is tested statistically.** NEES and NIS are checked against chi-square bounds
   over Monte Carlo runs with fixed seeds (§12). *Why:* a filter can track well and still be
   overconfident. Only a consistency test sees it.
7. **Ground truth is analytic.** Tests use known nulls, exact dead reckoning and hand-countable
   lifecycles, not recorded output (`docs/conventions/testing.md`).
8. **Fail before mutating.** A bad input raises before any track changes. *Why:* a `ValueError`
   mid-scan leaves half the tracks updated and the rest not.
9. **Every equation cites its source.** Docstrings carry `References` with chapter or equation
   numbers. *Why:* the reader is a student, and the citation is how they check us.
10. **Readable before fast.** Vectorise the obvious (§11), but never at the cost of the code
    reading like the textbook.

## 3. Architecture: strict decoupling

The tracker is cut into pieces, each behind a small protocol. Each piece sees only the
protocols of its neighbours. The cut is close to [Stone Soup](https://stonesoup.readthedocs.io)'s,
so a student who moves on to Stone Soup recognises it. Stone Soup cuts finer in two places: its
predictor and updater are one `Estimator` here, and its hypothesiser is the gate here.

| Piece | Protocol or type | Module (target, §13) | Knows about |
| :--- | :--- | :--- | :--- |
| State layout | `StateLayout`, `Coordinate` | `coordinates.py` | names, units, periods |
| Estimate | `StateEstimate` | `estimation.py` | mean, covariance, time, layout |
| Motion model | `MotionModel` | `motion.py` | layout, `f(x, dt)`, `Q(dt)` |
| Measurement | `Measurement`, `MeasurementBatch` | `measurements.py` | z, R, time, sensor, model ID |
| Measurement model | `MeasurementModel` | `measurement_models.py` | layouts, `h(x)` |
| Sensor | `SensorRegistration`, `SensorPose` | `sensors.py` | model IDs, coverage, pose |
| Filter | `Estimator` | `estimation.py`, `kalman.py`, `ukf.py` | motion and measurement models |
| Gate | `ChiSquareGate` | `association.py` | innovation statistics |
| Associator | `Associator` | `association.py` | a cost matrix |
| Initiator | `TrackInitiator` | `initiation.py` | a measurement, a model, a prior |
| Lifecycle | `LifecyclePolicy`, `TrackManager` | `lifecycle.py` | hits, misses, times, covariance |
| Track | `Track`, `TrackSnapshot` | `tracks.py` | an estimator, a status, counts |
| Tracker | `Tracker` | `tracker.py` | all of the above, through their protocols |

**The decoupling rule.** Swapping the filter (KF → EKF → UKF → IMM), the associator (GNN → NN
→ JPDA) or the initiator changes one constructor argument of `Tracker`, and no pipeline code.
A branch on the type of a filter or of a tracker, in `tracker.py` or in `pipelines/`, breaks
this rule: an `isinstance` check, or one code path per filter or per tracker class. A pipeline
that keeps a code path for each of two trackers is the shape this rule forbids, and TD1 is why
there will be only one.

**The dependency rule.** `core/tracking/` imports from `core/` only (`core.constants`,
`core.geodesy`), never from `pipelines/`. Pipelines adapt their products into `Measurement`s
(§10).

## 4. Filters: KF, EKF and UKF behind one interface

### 4.1 One contract

Every filter satisfies `Estimator` (`core/tracking/estimation.py`):

| Member | Does |
| :--- | :--- |
| `state` | the current `StateEstimate`, read-only |
| `set_state(state)` | replace the estimate, at birth or merge |
| `predict_to(timestamp_s)` | move the estimate forward, growing its covariance |
| `innovation_statistics(measurement, model)` | score a measurement without changing anything |
| `update(measurement, model, *, innovation=None)` | correct the estimate, optionally reusing the score |

`Tracker` and `Track` see only this protocol. A filter is chosen by passing an *estimator
factory* (a callable from a prior `StateEstimate` to an `Estimator`, with the motion model bound
in) to the initiator, which is what `DirectStateInitiator(estimator_factory, ...)` already takes.

### 4.2 One correction core

All three filters compute the same three things for a measurement, and differ only in how:

| Filter | Predicted measurement ẑ | Cross-covariance P_xz | Innovation covariance S |
| :--- | :--- | :--- | :--- |
| KF | H x̂⁻ | P⁻ Hᵀ | H P⁻ Hᵀ + R |
| EKF | h(x̂⁻) | P⁻ Jᵀ, with J = ∂h/∂x at x̂⁻ | J P⁻ Jᵀ + R |
| UKF | Σ wₘ h(χᵢ) | Σ w_c (χᵢ − x̂⁻)(h(χᵢ) − ẑ)ᵀ | Σ w_c (h(χᵢ) − ẑ)(h(χᵢ) − ẑ)ᵀ + R |

Everything after that is the same, and lives in **one** private function in `estimation.py`,
which every filter calls. It takes the prior, the innovation statistics, P_xz and, when the
filter has one, the measurement matrix (H or J). It:

1. computes the gain K = P_xz S⁻¹ by solving S Kᵀ = P_xzᵀ, never by inverting S;
2. updates the mean, x̂ = x̂⁻ + K ν, wrapping only the *state* layout's periodic components
   through `StateLayout.wrap`;
3. updates the covariance (TD3, below);
4. takes the symmetric part of P;
5. returns a new `StateEstimate`.

The innovation ν, S, the NIS and the log-likelihood come from the existing `innovation_stats`
(`estimation.py`), which whitens with a Cholesky factor. It is the one place NIS is computed.
`kalman.normalised_innovation_squared` and `kalman.innovation_of` go (§13).

**Two layouts, two wraps.** The innovation ν = z − ẑ is taken through the *measurement*
layout's `residual`, so a periodic measurement (an azimuth, a range folded at the map's span)
takes the short way round. The mean update wraps only the periodic components of the *state*
layout. The two are kept apart on purpose: a folded range has its period on the measurement
coordinate, while the range state stays continuous and unwrapped. Wrapping the state at the
measurement's period would make the track jump by one span whenever the target crosses the wrap.

**The covariance update (TD3).** The correction core uses:

- the **Joseph form**, (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ, when the filter supplies H or J. It is
  positive semidefinite for any K, because both terms have the form A M Aᵀ with M positive
  semidefinite, so roundoff cannot make a variance negative. This is what
  `kalman.update` does today;
- the **short form**, P⁻ − K S Kᵀ, when it does not (the UKF). The UKF has no H. With every
  sigma-point *covariance* weight non-negative (§4.4), P⁻ − K S Kᵀ is the Schur complement of S in a
  positive semidefinite joint covariance, so it is positive semidefinite up to roundoff, which
  the symmetrisation contains. This is the argument in `UKF.update`'s docstring.

Both branches are in the same function, side by side, with this reasoning in its docstring
once. Today the argument is split across `kalman.py` (against the short form) and `ukf.py` (for
it), and each reads as if the other were wrong. They are not: they are about filters with and
without H.

### 4.3 The KF and the EKF

`KalmanFilter` (target `kalman.py`) is a stateful `Estimator`, shaped like `UKF`: it holds a
`StateEstimate` and a motion model, and implements the protocol. One class covers the KF and
the EKF, because the EKF is the KF with H replaced by the Jacobian at the prediction
(Bar-Shalom, Li and Kirubarajan 2001, §10.3).

- **Motion.** It needs F and Q, so it accepts a *linear* motion model: one with
  `matrices(dt_s) -> (F, Q)`. `CartesianMotion.matrices` exists; `RadialMotion` gains it. Every
  motion model in the repo is linear, so there is no EKF on the motion side yet; a nonlinear
  motion model (coordinated turn) adds `jacobian(state, dt_s)` when it arrives.
- **Measurement.** It needs H or J, so it accepts a measurement model that also has
  `jacobian(state) -> (n_meas, n_state)`. A linear model returns its constant H. Passing a
  model without `jacobian` raises `TypeError` at construction, naming the model. There is **no
  silent finite-difference fallback**: a numeric Jacobian hides a wrong analytic one, and a
  student cannot tell which filter they are running. Tests check each analytic Jacobian
  against a central difference (§12).
- **Teaching layer.** The free functions `predict` and `update` in today's `kalman.py` are
  worth keeping for notebooks, as the textbook equations on bare arrays. If kept, they call
  the correction core; they are never a second implementation (TD2).

### 4.4 The UKF

`UKF` (`ukf.py`) keeps its current shape. Two requirements:

- **Non-negative covariance weights.** The weights are w₀ᵐ = 1 − n/(α²(n + κ)) for the
  mean, w₀ᶜ = w₀ᵐ + 1 − α² + β for the covariance, and wᵢ = 1/(2α²(n + κ)) for every other
  point, in both. §4.2's short-form argument needs every *covariance* weight non-negative,
  because a weighted sum of outer products is positive semidefinite only then. The mean
  weights don't enter that argument: a negative w₀ᵐ changes how the mean is formed, not
  whether P stays positive semidefinite. So `check_sigma_point_settings` must reject settings
  with w₀ᶜ < 0, rather than only requiring n + κ > 0 as today, and its message must say why.
  - The defaults α = 1, β = 2, κ = 0 give w₀ᵐ = 0, w₀ᶜ = 2 and wᵢ = 1/(2n), for any n.
  - With α = 1 and β = 2, w₀ᶜ = κ/(n + κ) + 2, which is non-negative for κ ≥ −2n/3. The
    heuristic n + κ = 3 of Julier and Uhlmann (1997), κ = 3 − n, makes w₀ᵐ negative for n > 3, but keeps w₀ᶜ ≥ 0 up to n = 9, so it is accepted.
  - A small α, such as the 10⁻³ that Wan and van der Merwe (2000) call usual, makes w₀ᶜ large
    and negative, so it is rejected. That departs from the paper this UKF follows, on purpose:
    the short-form update of §4.2 is only safe without it.
- **Protocol names.** `predict_to` takes `timestamp_s`, as `Estimator` says. Today it takes
  `time_s`, so a keyword call through the protocol raises `TypeError`
  (TD13).

### 4.5 Drift guards

Principle 2 is enforced by tests that run every filter through the same checks. The estimator
list is one parametrised fixture, so a new filter (IMM included) joins every guard by being
added once.

| Guard | Check | Tolerance |
| :--- | :--- | :--- |
| KF ≡ UKF on linear models | the real `KalmanFilter` and `UKF`, same prior, same `LinearMeasurement`, 200 steps | `rtol=1e-10` on mean and covariance, with each `atol` 10⁻¹⁰ times the scale of that entry's prior variance, since the entries mix m² and (m/s)² |
| EKF ≈ UKF on a nonlinear model | the spherical model of §6.1; scale the prior covariance by s², and the gap in the updated mean must shrink at second order in s, because it is ½ tr(∇²h P) to leading order | fitted slope of log gap against log s ≥ 1.8 (against log s², it would be 1) |
| Consistency, every filter | NEES and NIS over Monte Carlo runs (§12) | chi-square interval at 99.9 % |
| Definiteness, every filter | 200 steps, symmetric to `rtol=1e-12`, Cholesky succeeds | exact |

The first guard replaces today's `tests/core/tracking/test_ukf.py` check, which compares the
UKF against a hand-coded Kalman update rather than against `kalman.py`: two filters agreeing
with a third copy proves nothing about each other.

## 5. State and motion models

- **Canonical Cartesian order.** Per-axis interleaved, `[x_m, xdot_mps, y_m, ydot_mps, z_m,
  zdot_mps]`, which is `CartesianMotion`'s default. Axes are local ENU:
  x east, y north, z up. Any other order is legal through `CartesianMotion(order=...)`, and is
  safe because nothing indexes by position (principle 3). The positions-first order of
  `kalman.py`'s ENU models goes with them.
- **The 1-D radial state.** `RadialMotion` (to be renamed `RadialConstantVelocity`, TD13),
  `[range_m, range_rate_mps]`, closing-positive, so its F carries −T. This is scenario 003's
  `range_1d`.
- **Process noise: two conventions, both named.** The continuous white-noise acceleration
  model (q in m²/s³, Bar-Shalom, Li and Kirubarajan 2001, §6.2.2) is the library default, because
  its Q is correct for any step length and so for asynchronous sensors. The discrete
  white-noise acceleration model (DWNA, σₐ² in m²/s⁴, the same book's §6.3.2) is kept as a named option,
  because scenario 003 chose it on purpose (`spec/scenario-003-tracking.md` §6.2) and its
  results must not move during migration. Both live in `motion.py`; each docstring points at
  the other and states when they agree (`motion.py` module docstring). TD5.
- **State models are configurations, not classes.** Scenario 003's `range_1d`, `enu_2d` and
  `enu_3d` (`spec/data-001-formats.md` §6.6 `state_model`) become named pairs of a motion model
  and a measurement model, built by one function. This keeps scenario 003's §6.2 rule ("data,
  not a class hierarchy") with the new pieces.

## 6. Measurement models and coordinate frames

A measurement model maps a state in its `state_layout` to a measurement in its
`measurement_layout`. The state is Cartesian; the measurement is in the sensor's native
coordinates. The filter, not the measurement, bridges the two: the EKF through the Jacobian,
the UKF through sigma points. Converting measurements to Cartesian before filtering is not
allowed, except to seed a track (§6.1).

### 6.1 Spherical measurements of a Cartesian state

A new monostatic model, `RangeAzimuthElevationRangeRate`, maps an ENU state to
`(range_m, azimuth_rad, elevation_rad, range_rate_mps)`, relative to a `SensorPose`:

- range r = |p − pₛ|; azimuth atan2(Δe, Δn), clockwise from north, a `Coordinate` with period
  2π; elevation asin(Δu / r);
- range rate −(p − pₛ)·v / r, closing-positive;
- a 2-D variant drops elevation, and any variant may drop range rate; the measured components
  are chosen by name, as `CartesianPosition` does today.

It **reuses** the geometry of `core.geodesy.enu_to_range_azimuth_elevation`, factored so that
both call one function in radians. `kalman._enu_model` re-derives it today, with elevation as
atan2(u, ground) where geodesy uses asin(u/R): the two agree, but there are two of them.

It provides the analytic Jacobian (§4.3). The azimuth row is singular at r → 0 and the
elevation row at the zenith; the docstring says so, and the model raises rather than return a
non-finite Jacobian.

**Converted measurements, for initiation only.** One spherical measurement can seed a Cartesian
track: position from (r, θ, φ), with a covariance from the polar-to-Cartesian Jacobian. At
long range and coarse angle accuracy, the naive conversion is biased and its covariance is
wrong; the initiator uses a debiased conversion instead: Lerro and Bar-Shalom (1993) for 2-D
(r, θ), and Suchomski (1999), which extends it, for 3-D (r, θ, φ). The converted
position seeds the prior; every later update uses the spherical model.

### 6.2 Range rate in the gate

The range-rate row is a component of z and a row of h(x), so it enters S, and so the gate,
like any other component. A false alarm that matches in position but not in Doppler then fails
the gate. This is the mechanism by which Doppler reduces false associations in clutter; this
spec does not claim a fixed improvement, and §12 measures it on scenario 003.

Two radar facts the model does not hide:

- **Ambiguity.** A measured Doppler is folded. The model expects an unfolded range rate; the
  pipeline unfolds it first (scenario 003 §5.3, `pipelines/tracking.py`). A track that cannot
  yet trust its unfolding uses range alone (§8.3).
- **The blind zone.** A target crossing the line of sight has range rate near zero and is
  lost in the clutter notch. That is a detection limit, listed in §16.

### 6.3 Bistatic models

`BistaticRangeDopplerModel` exists and is unchanged: the state is ENU, the measurement is
`(path_range_m, path_range_rate_mps)` relative to a transmitter and receiver `SensorPose`,
closing-positive. One bistatic measurement cannot fix a position, so its initiator is still
open (§16).

## 7. Gating and data association

### 7.1 The gate

`ChiSquareGate(probability)` accepts a pair when its NIS is at most the chi-square quantile
for the measurement dimension. The gate threshold and GNN move into `association.py`, which
today imports them from `kalman.py` (TD13).

### 7.2 GNN and NN

`GlobalNearestNeighbour` builds an (n_tracks, n_measurements) cost matrix, NIS or negative
log-likelihood (`Tracker(cost=...)`), forbids pairs outside the gate, and solves it with
`scipy.optimize.linear_sum_assignment`. It maximises the number of matches before minimising
cost; the docstring says so, and names a finite non-assignment cost as the alternative.
`NearestNeighbour` is the greedy version, for teaching.

**Costs across measurement dimensions.** With the range-only retry of §8.3, one cost matrix
can hold pairs scored at different measurement dimensions: range and rate (2) beside range
alone (1). Raw NIS can't be compared across dimensions, because its expectation is the
dimension. A 1-D pair costs about 1 and a 2-D pair about 2, so a pair that fell back to range
alone undercuts its full-dimension rivals in the solve. `KalmanTracker` does this today: each
pair tries the full model first and falls back only if that gate fails, but the fallback
pairs then enter the same matrix as raw NIS (`kalman.py:1186-1196`). So
when the dimensions in one matrix differ, the cost is the negative log-likelihood,
½ νᵀ S⁻¹ ν + ½ ln |2π S|, the negative of the `log_likelihood` that `innovation_stats` already
returns. NIS stays available as a
cost only when every pair has the same dimension. If a scan would put NIS costs of different
dimensions in one matrix, `Tracker` raises rather than solve it. This may move scenario 003's numbers when it migrates
(§13 step 5). The commit records the move, as §12 requires.

### 7.3 Assignment order

The two trackers disagree today, and the disagreement is measured, so this spec makes it a
setting rather than choosing:

- `Tracker` assigns in two stages: confirmed tracks pick first, then tentative ones, and no
  track starts inside a confirmed track's gate (`tracker.py` module docstring);
- `KalmanTracker` solves once over every track. Its step-3 comment records that confirmed-first
  was tried on scenario 003 and was worse on every count (range RMSE 25.3 m against 17.5 m;
  fold selection 82 % against 86 %), because a confirmed track with a gate widened for
  re-acquisition outbids tentative tracks for false alarms.

`Tracker` takes `assignment: Literal["confirmed_first", "joint"]`. The library default stays
`"confirmed_first"`; scenario 003 sets `"joint"` in its TOML. Changing either default needs a
measurement, recorded in the docstring. TD8.

### 7.4 Extension point: JPDA

JPDA (Fortmann, Bar-Shalom and Scheffe 1983) updates each track with a weighted mixture of the
measurements in its gate, instead of one. To admit it without changing `Tracker`'s loop:

- `AssociationResult` gains optional association weights, shape (n_tracks, n_measurements + 1),
  the last column for "no measurement". GNN and NN leave it empty.
- `Estimator` gains no method. JPDA's combined innovation and spread-of-innovations term are
  computed from each pair's `InnovationStats` by a helper in `association.py`, and applied
  through the correction core.

JPDA itself is deferred to the first multi-target scenario: scenario 003 §7 records GNN as a
deliberate limit, and its §13.3 names a second target as what makes JPDA or MHT necessary.

### 7.5 Out of scope: PMBM

The Poisson multi-Bernoulli mixture filter (García-Fernández et al. 2018) models clutter,
missed detections and the birth of new targets inside the filter itself. It is out of scope, for a structural reason: it replaces the split
this spec is built on. In PMBM, association hypotheses, track existence and birth are all part
of the filter's density, so there is no separate associator, initiator or lifecycle to swap.
It would be a second tracker, not a fourth filter, and belongs in a library built on random
finite sets.

## 8. Track lifecycle

### 8.1 States

`tentative → confirmed ⇄ coasting → deleted`, the vocabulary of `spec/data-001-formats.md` §5.
A confirmed track that misses becomes coasting; a hit makes it confirmed again.

### 8.2 Rules

All lifecycle settings live in `LifecyclePolicy`, once. Pipeline configs hold a
`LifecyclePolicy` (or build one from their TOML table); they never copy its fields one by one
(the PR #13 review found them declared twice and copied across).

| Rule | Setting | Status |
| :--- | :--- | :--- |
| M-of-N confirmation, counted from birth | `n_confirm_hits`, `n_confirm_frames` | exists |
| Tentative deletion once M is unreachable | (derived) | exists |
| Deletion after misses in a row | `n_delete_misses` | exists |
| Re-acquisition window for confirmed tracks | `n_reacquire_misses` (renamed, TD13) | the count exists in both stacks (`LifecyclePolicy`); the rate reset that widens the gate is KF stack only (§8.3) |
| Deletion after a coast time | `max_coast_time_s` | exists |
| **Covariance-based deletion** | `max_position_sigma_m` | **new** |

**Covariance-based deletion.** A track is deleted when the square root of the largest
eigenvalue of its position covariance exceeds `max_position_sigma_m`, whatever its miss count.
*Why:* miss counts and coast time are proxies for "we no longer know where it is"; the
covariance is the thing itself, and it grows at the rate the motion model says, so a fast
target and a slow scan are handled without retuning counts. `None`, the default, turns the rule
off. Position components are found by name through `StateLayout`. Stone Soup's
`CovarianceBasedDeleter` is the nearest precedent, with a different measure: it thresholds the
trace of the covariance (optionally over chosen components), which mixes units unless those are
all positions. A standard deviation in metres is a number an intern can choose.

`Track.score`, `age` and `existence_probability` (unused in any decision,
`docs/reviews/pr1-review.md` #24) stay out of `Track` until a decision uses them.

### 8.3 Scenario 003's special cases, made general

`KalmanTracker` has three behaviours `Tracker` lacks (`docs/tracking/README.md`,
"Limitations"). They are what blocks the pipeline's move to `Tracker`, and where they should
live was left open until now. This spec settles it: the
mechanism goes in `core/tracking/`, generic and named by coordinates; the scenario's policy goes
in `pipelines/`.

| Behaviour today | Core mechanism | Scenario policy |
| :--- | :--- | :--- |
| Per-track range-only bootstrap (`measurement_dims`) | a measurement model can return its **marginal** over a subset of its components, by name | the pipeline passes a selector: a callable from a `TrackSnapshot` and a `Measurement` to the ordered model IDs to try |
| Range-only retry when the velocity gate fails | `Tracker` gates each pair against the selector's models in order, and keeps the first that passes | scenario 003's selector returns (range and rate, range only) |
| `_forget_rate` on entering re-acquisition | on the miss that starts the re-acquisition window, reset the named coordinates' variances to the initiator's prior and zero their cross-covariances | `LifecyclePolicy.reacquire_reset = ("range_rate_mps",)` |

The default selector returns the measurement's own model only, so every other user of `Tracker`
sees no change. `_forget_rate`'s positional slicing (`n_axes = n_state // 2`) goes, per
principle 3.

## 9. Multiple models (IMM)

IMM runs a bank of filters with different motion models and mixes them by mode probability
(Blom and Bar-Shalom 1988; Bar-Shalom, Li and Kirubarajan 2001, ch. 11). It is specified here only as an extension point:

- `IMM` is an `Estimator` that holds `Estimator`s. `Tracker` sees one estimator per track, so
  nothing outside `IMM` changes.
- Its `state` is the mode-probability-weighted mixture, in the **union** layout of its modes.
  A mode whose layout lacks a coordinate (a turn rate, an acceleration) is augmented with it at
  zero mean and a stated variance on mixing, and the coordinate dropped on return. The mapping
  is by `StateLayout` name, which is why principle 3 matters (`docs/reviews/pr1-review.md`
  #11, on modes with different state spaces).
- Its `innovation_statistics` returns the mode-mixed likelihood, so gating and GNN work
  unchanged.

It is built with the first manoeuvring scenario, together with the coordinated-turn motion
model, as the PR #1 review planned. Scenario 003 §13.4 sets the same bar from the other side:
IMM only once `sigma_accel_mps2` is shown to be the binding error on the circuit's turns.

## 10. Integration with the rest of radar-forge

### 10.1 Data flow

| Step | Type | Units and frame | Owner |
| :--- | :--- | :--- | :--- |
| Detection | `core.detection.Detection` | cell indices, powers in W | `core/detection.py` |
| Plot | `Plot` (today `pipelines.tracking.Measurement`, TD12) | m, m/s, closing-positive, sensor frame | `pipelines/tracking.py` |
| Scan | `MeasurementBatch` of `Measurement`s | as the measurement model's layout | `pipelines/` builds, `core/tracking` defines |
| Tracking | `Tracker.process(batch)` | state in ENU or radial | `core/tracking/tracker.py` |
| Result | `TrackSnapshot` per live track, and the association of the scan | as the state layout | `core/tracking/tracks.py` |
| Output | `tracks.csv`, `detections.csv` | `spec/data-001-formats.md` §6.5, §6.6 | `scripts/run_scenario.py` |

`TrackSnapshot` is to carry what `tracks.csv` needs from one object: the estimate, status, hit
and miss counts, the measurement index it took this scan, and that update's NIS, as the PR #13
review asked. Today it carries the first two only, with the last measurement time and the
source sensors. The layout's coordinate names become
`metadata.json` `tracking.state_fields`, so `cov_<i>_<j>` is indexed by name, not by guess.

### 10.2 Rules

- `core/tracking/` imports nothing from `pipelines/` (§3).
- Constants come from `core.constants` (CLAUDE.md), and geodetic conversions from
  `core.geodesy`; neither is redefined in `core/tracking/` (§6.1).
- Tracker settings come from the scenario TOML's `[tracking]` table, read once by the pipeline
  into `LifecyclePolicy`, the gate, the motion model and the filter factory. Scenario-only
  settings (`max_velocity_mps`, unfolding tolerances) stay in `pipelines/`.
- **Timestamps.** Every measurement in a batch carries the batch's time; batches arrive in
  time order. These are stated limitations (§16), not silent assumptions.
- **One sensor registry.** Every sensor is a `SensorRegistration` in one `Tracker`, so one
  target has one track whichever sensor sees it.

## 11. Performance

The library is for teaching, so correctness and readability bound performance, not the other
way round. The requirement is:

- **Vectorise** where NumPy makes it natural: the sigma points, and the (n_tracks,
  n_measurements) gate and cost matrices once the innovation statistics are batched.
- **Loops over tracks are allowed**, each with a comment saying why (style.md §7): a track's
  update is a small dense solve, and a vectorised version across tracks of different layouts
  would be unreadable.
- **No JIT or compiled dependency** in core (numba, Cython). If one is ever justified, it goes in
  an extra, per CLAUDE.md.
- **Measured, not assumed.** A `benchmark`-marked test records time per scan for scenario 003's
  load and for a synthetic 100-track, 1000-measurement scan. It records; it does not assert a
  budget, until a scenario sets one.

## 12. Validation strategy

| Level | What | How |
| :--- | :--- | :--- |
| Unit, analytic | dead reckoning; Q against the published matrices; a known GNN optimum; M-of-N by hand; Jacobians against central differences | exact or `rtol=1e-12`; Jacobians at the tolerance `tests/core/test_tracking.py` already justifies |
| Unit, statistical | gate acceptance rate | Clopper–Pearson interval, fixed seed |
| Filter consistency | NEES and NIS over 200 runs × 30 steps, every estimator (§4.5), per Bar-Shalom, Li and Kirubarajan (2001) §5.4.2 | chi-square interval at 99.9 %, `slow` |
| Drift guards | §4.5 | as tabled there |
| Lifecycle | confirmation, deletion, re-acquisition, covariance deletion on hand-countable sequences | exact |
| End to end | scenario 003 over S1, and its dual-PRF variant over S3 | `spec/scenario-003-tracking.md` §12 and §14.7, unchanged by migration |
| Track metrics | range and rate RMSE, mean NIS, OSPA (Schuhmacher, Vo and Vo 2008), track breaks | `spec/data-001-formats.md` §6.9 |

Two rules carry over from `docs/conventions/testing.md`: never loosen a tolerance to pass, and
prefer analytic truth to recorded output. Migration adds one: **each step of §13 leaves
scenario 003's end-to-end numbers where they were**, or explains the move in the commit.

## 13. Migration from today's code

Each step is one PR, and each leaves `make check` green. The order makes each step smaller than
it would be later. The names and module moves are TD13's.

0. **Renames that do not wait.** `SensorRoute` → `SensorRegistration`, `observable` → `covers`,
   `UKF.predict_to(time_s)` → `predict_to(timestamp_s)`. A small PR on main.
   *Check:* `git grep -nE 'SensorRoute|^\s*observable ?:|observable=|\.observable\b|``observable``|def predict_to\(self, time_s' -- src tests docs/tracking`
   returns nothing. The pattern skips the observability prose in `kalman.py`, which stays.
1. **Gate and GNN into `association.py`;** `kalman.py` imports them from there.
   *Check:* `git grep -n 'from radar_forge.core.tracking.kalman import' -- src/radar_forge/core/tracking/association.py`
   returns nothing.
2. **One innovation API and the correction core** (§4.2). `InnovationStats.residual` becomes
   `innovation`. `UKF` calls the core. *Check:* AC2.
3. **`KalmanFilter` as an `Estimator`** (§4.3), on the core, with `RadialMotion.matrices`. The
   KF ≡ UKF guard (§4.5) lands here. The same PR first renames today's `KalmanFilter`
   (`kalman.py`), which holds a track's estimate and is not a filter, to `KalmanTrackState`.
   Otherwise two public classes share one name until step 7, the trap TD12 avoids for
   `Measurement`. *Check:* AC3 and AC4, and `git grep -nE '^class KalmanFilter\b' -- src` finds
   exactly one class.
4. **The spherical measurement model** (§6.1) with its Jacobian and the debiased initiator.
5. **Scenario mechanisms in `Tracker`** (§8.3), and `assignment="joint"` (§7.3).
6. **Scenario 003 on `Tracker`.** `pipelines/tracking.py` becomes an adapter: plots in,
   batches out, snapshots back. `TrackingConfig.state_model` is honoured (it is ignored today).
7. **Delete the old stack:** `KalmanTracker`, `KalmanTrackState`, `KalmanState`, `TrackModel`,
   `FrameResult`, and the rest of TD13's renames. Two of those renames are scenario TOML keys
   (TD13), so this step also updates the scenario 003 TOMLs and spec, and the data-001
   `metadata.json` schema. Module moves (§3 target column). Refresh the stale text:
   `kalman.py`'s "promotes this module to a subpackage", `estimation.py`'s "the one
   implementation", and `spec/structure.md` D2. *Check:* AC1, AC7 and AC8;
   `git grep -nE '^class (KalmanTracker|KalmanTrackState|KalmanState|TrackModel|FrameResult)\b' -- src`
   returns nothing; and so does
   `git grep -nE 'n_reacquire_frames|\bCartesianPosition\b|\bRadialMotion\b|def normalise|\.normalise\(|\.noise_density\b[^_]|self\.noise_density\b|sigma_accel_mps2' -- src tests scripts scenarios spec/scenario-003-tracking.md docs/tracking`.
   Also fix `kalman.py`'s module docstring if it survives: it says "a linear Kalman filter",
   but the ENU models make it an extended one (`:138`, `:479`).
8. **Covariance-based deletion** (§8.2). Independent of the rest; any time after step 2.
9. **The refactor-002 audit of the finished package.** Deferred here from
   [`refactor-002`](refactor-002-spec-first-audit.md#45-tracking-is-deferred-to-tracker-001),
   because steps 0–7 replace most of what it would audit. Once step 7 has merged, run both of its
   audits over the tracking code as it then stands:
   - **§1.2, code against spec**, Q1–Q8, over `src/radar_forge/core/tracking/**`. The reference
     is this spec and [`docs/tracking/README.md`](../docs/tracking/README.md). Record the result
     as a `core/tracking/` section of `docs/audits/core-audit.md`, numbering findings on from the
     last F and R there.
   - **§2, test audit and pruning**, Sound / Redundant / Weak / Wrong, over
     `tests/core/tracking/**`, `tests/core/test_tracking.py` (or what survives it),
     `tests/pipelines/test_tracking.py` and `tests/pipelines/test_scenario_003.py`. Delete the
     Redundant ones; rewrite or fix the Weak and Wrong ones. The rules are refactor-002 §2.4:
     the `docs/conventions/testing.md` §8 floor holds, every deletion names the test that still
     catches its bug, and `src/` line and branch coverage do not drop. Record it in
     `docs/audits/tests-audit.md`.

   *Check:* AC11.

## 14. Decisions

### 14.1 Design decisions

### TD1 — One tracker loop

`Tracker` is the only scan loop. `KalmanTracker` is deleted once scenario 003 runs on `Tracker`.
*Rejected:* keeping `KalmanTracker` as a simpler teaching tracker. Two loops are what the PR #1,
#2 and #13 reviews kept finding drifting; the simple version for teaching is the free functions
of §4.3, not a second tracker.

### TD2 — Filters are interchangeable estimators

KF, EKF and UKF implement `Estimator`, and are chosen by the estimator factory. *Rejected:* a
filter-type switch inside `Tracker` (breaks §3's decoupling rule); a KF implemented as a UKF on
linear models (correct, but hides the textbook KF a student is looking for).

### TD3 — One correction core, two covariance forms

§4.2. Joseph form when H or J exists, short form for the UKF, in one function. *Rejected:*
Joseph form everywhere (the UKF has no H; a statistically linearised H can be built, but adds a
solve for no gain under §4.4's weights); the short form everywhere (gives up the KF's guarantee
at no saving that matters here).

### TD4 — Names, not positions

Every state component is reached through `StateLayout` by name. *Rejected:* a canonical fixed
order that code may assume (works until the first model with a different order, which IMM is).

### TD5 — Continuous white noise by default, DWNA kept

§5. *Rejected:* one model only. Continuous noise is right for asynchronous sensors; DWNA is what
scenario 003 specified and what its tests pin.

### TD6 — Spherical measurements stay spherical

§6.1. *Rejected:* converting measurements to Cartesian and running a linear KF. Its covariance
is state-dependent and its bias grows with range × angle error; it is used only to seed.

### TD7 — Range rate is measured, not stated

§1.3, §6.2. Range rate is a measurement row, h(x) = −(p − pₛ)·v / r. *Rejected:* a range-rate
state component in a Cartesian state (redundant with velocity, and makes the covariance
singular).

### TD8 — Assignment order is a setting

§7.3. *Rejected:* choosing one order for all scenarios. Each is measured better on a different
problem.

### TD9 — Mechanism in core, scenario policy in pipelines

§8.3. *Rejected:* moving scenario 003's bootstrap and re-acquisition into `Tracker` as named
scenario features (couples core to one scenario); leaving them in `pipelines/` around a
`Tracker` that cannot express them (the pipeline would re-implement gating).

### TD10 — Covariance-based deletion

§8.2. *Rejected:* relying on miss counts and coast time alone; they need retuning whenever the
scan rate or target speed changes.

### TD11 — IMM and JPDA as extension points; PMBM out of scope

§7.4, §7.5, §9. *Rejected:* implementing them now (no scenario needs them; each would be
untested against real data); implementing PMBM at all (§7.5).

### TD12 — `Plot` for the pipeline's detection report

`pipelines.tracking.Measurement` is renamed `Plot`. ASTERIX CAT048 ("Monoradar Target Reports")
sends a target report either as a plot or as a track, and a plot is the untracked form, which is
what this class holds; `spec/data-001-formats.md` §4 already says "Detections (plots)". *Why:*
two public classes named `Measurement` in one package tree, holding different things, is a
trap. `core.tracking.Measurement` keeps the name, as the textbook word for z. (Stone Soup's
equivalent class is `Detection`; that name is taken here by `core.detection.Detection`.)

### TD13 — Target names

The target names are below; the module moves are §3's target column, and `InnovationStats.residual`
becomes `innovation`, the textbook term (Bar-Shalom, Li and Kirubarajan 2001, §5.2). `SensorRoute` becomes
`SensorRegistration`: it does more than route (model IDs, the coverage test and building
batches), the rest of the API already says "sensor" (`sensor_id`, `sensors=`, `add_sensor`),
and "route" is not a tracking term a student can look up.
*Rejected:* `Sensor` (Stone Soup's `Sensor` generates detections, a different job);
`SensorSpec` (hardware parameters live in `core.radar.Radar`).

| Now | Target | Why |
| :--- | :--- | :--- |
| `SensorRoute`, `.observable` | `SensorRegistration`, `.covers` | as above; "observable" already means observability |
| `LifecyclePolicy.n_reacquire_frames` | `n_reacquire_misses` | it counts misses, not frames |
| `CartesianPosition`, `indices` | `LinearMeasurement`, `mapping` | its docstring says "despite the name, any coordinates"; it is z = Hx + w |
| `RadialMotion` | `RadialConstantVelocity` | says which motion |
| `build_tracker`, `build_tracker_enu` | same names, each with an `estimator_factory` argument defaulting to the UKF | both always build a UKF today. Per TD2, the filter is one argument, not one builder per filter |
| `UKF.predict_to(time_s)` | `predict_to(timestamp_s)` | the `Estimator` protocol's name |
| `StateLayout.normalise` | removed | it is `wrap` under a second name |
| `CartesianMotion.noise_density` | `acceleration_noise_density_m2ps3`, `jerk_noise_density_m2ps5` | one dict held two units |
| `sigma_accel_mps2` | `sigma_acceleration_mps2` | as `motion.py` spells it |
| `pipelines.tracking.Measurement` | `Plot` | TD12 |

Textbook names stay: `UKF`, `KalmanFilter`, `ChiSquareGate`, `NearestNeighbour`,
`GlobalNearestNeighbour`, `TrackInitiator`, `StateEstimate`, `TrackSnapshot`, `Measurement`,
`TrackManager`, `BistaticRangeDopplerModel`.

`sigma_accel_mps2` and `n_reacquire_frames` are also keys in the scenario 003 TOMLs'
`[tracking]` table (`spec/scenario-003-tracking.md`, "Parameters as shipped"), and
`metadata.json` records them. So renaming them changes a file format, not only code:
the TOMLs, scenario 003's spec and `spec/data-001-formats.md` change in the same PR, the
`metadata.json` schema version is bumped, and a TOML that still uses an old key fails with a
message naming the new one, rather than being silently ignored.

### TD14 — Readable before fast

§11. *Rejected:* a real-time performance budget, and compiled dependencies in core.

### 14.2 Rulings on the requested features

| Requested | Ruling | Where |
| :--- | :--- | :--- |
| Strict decoupling of models, filters, association, lifecycle | **Adopted.** Already the shape; §3 makes it a rule | §3, TD2 |
| Polar/spherical to Cartesian | **Adopted.** Spherical model with Jacobian; debiased conversion for initiation only | §6.1, TD6 |
| Doppler in the state vector | **Adopted as corrected:** in the measurement and the gate, not the state | §1.3, §6.2, TD7 |
| GNN, JPDA, PMBM | **GNN adopted** (exists); **JPDA** as an extension point; **PMBM** out of scope | §7, TD11 |
| M-of-N lifecycle, covariance deletion | **Adopted.** M-of-N exists; covariance deletion is new | §8, TD10 |
| High-performance implementation | **Adopted, bounded:** vectorise and measure; readability first | §11, TD14 |
| Validation strategies | **Adopted:** analytic, statistical, drift guards, end to end | §4.5, §12 |

## 15. Acceptance criteria

| # | Criterion | Checked by |
| :--- | :--- | :--- |
| AC1 | One scan loop: `git grep -nE '^class (KalmanTracker\|KalmanState\|TrackModel\|FrameResult)\b' -- src` returns nothing | [review] |
| AC2 | One gain and covariance update: `git grep -n 'np.linalg.solve' -- src/radar_forge/core/tracking` finds no gain computed outside the correction core | [review] |
| AC3 | `KalmanFilter` and `UKF` both satisfy `Estimator`, and a test runs `Tracker` with each | [test] |
| AC4 | KF ≡ UKF on linear models at `rtol=atol=1e-10`, using the real `KalmanFilter` | [test] |
| AC5 | The NEES/NIS and definiteness tests are parametrised over every estimator | [test] |
| AC6 | Every analytic Jacobian matches a central difference | [test] |
| AC7 | No tracking code indexes a state by position: a reviewer finds no slice of a state or covariance by integer outside `StateLayout` | [review] |
| AC8 | Scenario 003's end-to-end criteria (`spec/scenario-003-tracking.md` §12) pass on `Tracker`, and every number that differs from `KalmanTracker`'s is explained in the commit that moved it (§12; §7.2's cost change may move some) | [test] |
| AC9 | `core/tracking/` imports nothing from `pipelines/` | [review] |
| AC10 | This spec passes `scripts/check_conventions.py` R7 | [hook] |
| AC11 | Step 9 is done: `docs/audits/core-audit.md` has a verdict for every public name in `core/tracking/`, and `docs/audits/tests-audit.md` classifies every tracking test, with the coverage comparison (refactor-002 §1.2, §2.4) | [review] |

## 16. Limitations

These hold after migration; each is stated in a docstring, not discovered by a user.

- Every measurement in a batch carries the batch's time; batches must arrive in time order.
  Out-of-sequence measurements are rejected, not buffered.
- GNN gives each track at most one measurement per scan. One target giving several detections
  in a scan starts several tracks; the fix belongs in detection
  (`test_several_detections_of_one_target_give_one_confirmed_track` is `xfail`).
- A target crossing the line of sight is lost in the Doppler blind zone.
- No initiator for the bistatic model: one bistatic measurement cannot fix a position. Seed
  such tracks with `Tracker.seed`.
- Expect track fragmentation with the default settings; it is measured once the runner returns.
- No IMM, coordinated turn, Singer model, JPDA or MHT (§7.4, §9).

## References

- Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and
  Navigation*, Wiley, 2001. §5.2 (linear estimation in dynamic systems: the Kalman filter, and
  the properties of the innovations), §5.4.2 (the statistical tests for filter consistency),
  §6.2.2 (the continuous white noise acceleration model), §6.3.2 (the discrete white noise
  acceleration model), §10.3 (the extended Kalman filter), ch. 11 (adaptive estimation and
  maneuvering targets, including the interacting multiple model estimator).
- S. J. Julier and J. K. Uhlmann, "A new extension of the Kalman filter to nonlinear systems,"
  *Proc. SPIE*, vol. 3068, *Signal Processing, Sensor Fusion, and Target Recognition VI*,
  1997, p. 182 ff. (the heuristic n + κ = 3).
- E. A. Wan and R. van der Merwe, "The unscented Kalman filter for nonlinear estimation,"
  *Proc. IEEE Adaptive Systems for Signal Processing, Communications, and Control Symposium*,
  2000, pp. 153-158 (the UKF, its sigma-point weights, and the usual α, β and κ).
- S. J. Julier, "The scaled unscented transformation," *Proc. American Control Conference*,
  vol. 6, 2002, pp. 4555-4559.
- D. F. Crouse, "On implementing 2D rectangular assignment algorithms," *IEEE Trans. Aerosp.
  Electron. Syst.*, vol. 52, no. 4, pp. 1679-1696, 2016.
- D. Lerro and Y. Bar-Shalom, "Tracking with debiased consistent converted measurements versus
  EKF," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 29, no. 3, pp. 1015-1022, 1993 (2-D polar
  measurements).
- P. Suchomski, "Explicit expressions for debiased statistics of 3D converted measurements,"
  *IEEE Trans. Aerosp. Electron. Syst.*, vol. 35, no. 1, pp. 368-370, 1999.
- H. A. P. Blom and Y. Bar-Shalom, "The interacting multiple model algorithm for systems with
  Markovian switching coefficients," *IEEE Trans. Autom. Control*, vol. 33, no. 8,
  pp. 780-783, 1988.
- T. E. Fortmann, Y. Bar-Shalom and M. Scheffe, "Sonar tracking of multiple targets using joint
  probabilistic data association," *IEEE J. Oceanic Eng.*, vol. 8, no. 3, pp. 173-184, 1983.
- Á. F. García-Fernández, J. L. Williams, K. Granström and L. Svensson, "Poisson
  multi-Bernoulli mixture filter: direct derivation and implementation," *IEEE Trans. Aerosp.
  Electron. Syst.*, vol. 54, no. 4, pp. 1883-1901, 2018.
- D. Schuhmacher, B.-T. Vo and B.-N. Vo, "A consistent metric for performance evaluation of
  multi-object filters," *IEEE Trans. Signal Process.*, vol. 56, no. 8, pp. 3447-3457, 2008.
- Stone Soup, <https://stonesoup.readthedocs.io>: the component split of §3; `Sensor.measure`;
  `stonesoup.deleter.error.CovarianceBasedDeleter`; `stonesoup.types.detection.Detection`.
- EUROCONTROL, *ASTERIX Part 4, Category 048: Monoradar Target Reports*,
  EUROCONTROL-SPEC-0149-4, ed. 1.32, 2024, §4.6.2 (plots and tracks).
