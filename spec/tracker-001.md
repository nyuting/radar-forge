# Tracker 001 — Tracking architecture

> **Reading order.** §1–§3 say what the tracking package is for and how it is cut into pieces.
> §4 answers the question this spec was written for: how the KF, the EKF and the UKF live in
> one package without drifting apart. They are one filter with one flow, and differ only in how
> they linearise. §5–§10 specify each piece and how it meets the rest of radar-forge. §11–§13
> cover performance, validation and the path from today's code. §14 holds the decisions, §15
> the acceptance criteria.
>
> | Part | Sections |
> | :--- | :--- |
> | Purpose and principles | [§1](#1-status-purpose-and-scope), [§2](#2-what-a-good-tracking-module-is) |
> | Architecture | [§3](#3-architecture-strict-decoupling), [§4](#4-filters-one-gaussian-filter-three-linearisations) |
> | The pieces | [§5](#5-state-and-motion-models) to [§9](#9-multiple-models-imm) |
> | Integration | [§10](#10-integration-with-the-rest-of-radar-forge) |
> | Performance, validation, migration | [§11](#11-performance) to [§13](#13-migration-from-todays-code) |
> | Decisions and acceptance | [§14](#14-decisions), [§15](#15-acceptance-criteria) |

## 1. Status, purpose and scope

**Status: specified, partially implemented.** Most of the pieces in §3 exist in
`src/radar_forge/core/tracking/`. What does not yet hold is the end state of one tracker and
one filter (§4, TD1), and §13 lists the steps to it. Where this spec and the code disagree, the code is behind and
this spec is the target, except where §14 says otherwise. Step 9 of §13 also holds the audit of this package, deferred until the migration
is done.

This spec covers `core/tracking/`: filters, motion and measurement models, gating, association,
track initiation and track lifecycle, and the interface they offer the pipelines. How to run the
tracker, and how to choose its process noise, are in the user guide,
[`docs/tracking/README.md`](../docs/tracking/README.md). This spec does not repeat them.

### 1.1 Scope

**In scope.**

- One scan-loop tracker, `Tracker`, used by every pipeline.
- One Gaussian filter, `GaussianFilter`, with the linear Kalman filter (KF), the extended
  Kalman filter (EKF) and the unscented Kalman filter (UKF) as its interchangeable
  linearisations.
- Cartesian motion models, and monostatic and bistatic radar measurement models, including the
  spherical-to-Cartesian geometry.
- Chi-square gating, global nearest-neighbour (GNN) and nearest-neighbour (NN) assignment.
- M-of-N track confirmation, miss-count, coast-time and covariance-based deletion.
- The interacting multiple model (IMM) filter, joint probabilistic data association (JPDA and
  its JPDA\*, Set-JPDA and JIPDA forms) and track-oriented multiple hypothesis tracking (MHT),
  specified by the first scenario that needs them,
  [`scenario-011-terminal-area-tracking.md`](scenario-011-terminal-area-tracking.md) (§7.4, §7.6,
  §9).
- One random-finite-set tracker, PMBM, as a **comparison tracker** beside `Tracker`, not inside
  it (§7.5).

**Out of scope.**

- Random-finite-set trackers inside `Tracker`'s loop, and GLMB. §7.5 says why.
- Track-to-track fusion across sensors. One `Tracker` already takes every sensor's
  measurements, which is measurement-level fusion; fusing tracks from separate trackers is
  a later spec.
- Real-time operation. §11 bounds the performance requirement.

### 1.2 Provenance

The UKF stack in `core/tracking/` comes from `unified-extensible-tracker`, an earlier tracker
written by Ng Zeming Michael Eugene, who contributes it to radar-forge under the repository's
MIT licence ([`LICENSE`](../LICENSE)). The linear Kalman stack was written for scenario 003
(`spec/scenario-003-tracking.md`). Where the code implements a published method, the module or
class docstring cites the source in its `References` section. 

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
2. **One implementation of each equation, and one flow.** The Kalman gain, the covariance
   update, the NIS and the angle wrap each exist in exactly one function, and predict and
   update each exist in one filter (§4). *Why:* two copies drift. Today's two copies of the
   covariance update already argue opposite things in their docstrings (§4.3).
3. **Every state component has a name and a unit.** Code reaches a component through
   `StateLayout` by name (`"x_m"`, `"xdot_mps"`), never by position. *Why:* positional indexing
   silently breaks when the order changes. Today's re-acquisition reset (§8.3) assumes
   positions come first, and is wrong for the interleaved order of §5.
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
10. **Readable and fast.** Neither is traded for the other (`spec/starter.md` §1.1). The hot
    path, gating every track against every measurement, is batched NumPy that still reads as
    the textbook equation, with its array shapes written out (§11). *Why:* this is a
    state-of-the-art workbench. A tracker that is clear but too slow for a realistic scan is
    not one a practitioner can use, and one that is fast but opaque is not one a student can
    check.

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
| Filter | `Estimator`, `GaussianFilter` | `estimation.py` | a motion model, a linearisation |
| Linearisation | `Linearisation`, `TaylorLinearisation`, `UnscentedLinearisation` | `linearisation.py` | a function, a mean, a covariance |
| Gate | `ChiSquareGate` | `association.py` | innovation statistics |
| Associator | `Associator` | `association.py` | a cost matrix |
| Initiator | `TrackInitiator` | `initiation.py` | a measurement, a model, a prior |
| Lifecycle | `LifecyclePolicy`, `TrackManager` | `lifecycle.py` | hits, misses, times, covariance |
| Track | `Track`, `TrackSnapshot` | `tracks.py` | an estimator, a status, counts |
| Tracker | `Tracker` | `tracker.py` | all of the above, through their protocols |

**The decoupling rule.** Swapping the linearisation (KF → EKF → UKF), the estimator (Gaussian
filter → IMM), the associator (GNN → NN → JPDA) or the initiator changes one constructor
argument, and no pipeline code. A branch on the type of a linearisation, a filter or a
tracker, in `estimation.py`, `tracker.py` or `pipelines/`, breaks this rule: an `isinstance`
check, or one code path per filter. There is one tracker class, `Tracker` (TD1).

**The dependency rule.** `core/tracking/` imports from `core/` only (`core.constants`,
`core.geodesy`), never from `pipelines/`. Pipelines adapt their products into `Measurement`s
(§10).

## 4. Filters: one Gaussian filter, three linearisations

### 4.1 One contract

Every filter satisfies `Estimator` (`core/tracking/estimation.py`):

| Member | Does |
| :--- | :--- |
| `state` | the current `StateEstimate`, read-only |
| `set_state(state)` | replace the estimate, at birth or merge |
| `predict_to(timestamp_s)` | move the estimate forward, growing its covariance |
| `innovation_statistics(measurement, model)` | score a measurement without changing anything |
| `update(measurement, model, *, innovation=None)` | correct the estimate, optionally reusing the score |

`Tracker` and `Track` see only this protocol. Two classes implement it: `GaussianFilter`
(§4.2), which is the KF, the EKF and the UKF, and `IMM` (§9), which holds `GaussianFilter`s. A
filter is chosen by passing an *estimator factory* to the initiator. The factory is a callable
from a prior `StateEstimate` to an `Estimator`, with the motion model and the linearisation
bound in, and it is what `DirectStateInitiator(estimator_factory, ...)` already takes.

### 4.2 One flow

The KF, the EKF and the UKF are one algorithm, the general Gaussian filter (Särkkä 2013,
ch. 6). At each step it approximates the density as Gaussian by matching its mean and
covariance, then applies the same Kalman correction. The three differ in one thing only: how
they carry a mean and a covariance through a function. That is the filter's *linearisation*.

| Linearisation | Gives | Moments of g(x), for x with mean x̂ and covariance P | Supplies H |
| :--- | :--- | :--- | :--- |
| `TaylorLinearisation` | the KF on a linear model, the EKF otherwise | mean g(x̂), covariance J P Jᵀ, with J = ∂g/∂x at x̂ | yes: H, or J |
| `UnscentedLinearisation(alpha, beta, kappa)` | the UKF | weighted sums over 2n + 1 sigma points (§4.4) | no |

A `Linearisation` has two methods, one per step of the filter, and a third that checks the
motion model when the filter is built:

| Method | Returns |
| :--- | :--- |
| `predict(motion, state, dt_s)` | the predicted mean x̂⁻ `(n_state,)` and covariance P⁻ `(n_state, n_state)`, with Q(dt_s) added |
| `measure(model, state)` | `MeasurementMoments`: ẑ `(n_meas,)`, P_xz `(n_state, n_meas)`, S without R `(n_meas, n_meas)`, and H `(n_meas, n_state)` or `None` |
| `check_motion_model(motion)` | nothing; raises `TypeError` if the linearisation cannot use the model (no `matrices` for the Taylor one), or `ValueError` if its settings do not suit the state size (the unscented weights). It is how `GaussianFilter` refuses a bad pairing at construction without asking which linearisation it holds |

`GaussianFilter` holds a `StateEstimate`, a motion model and a linearisation. Each of its
methods is the same for all three filters:

- `predict_to` calls `linearisation.predict` and stores the result.
- `innovation_statistics` calls `linearisation.measure`, adds R to S, and passes ẑ, z and S
  to `innovation_stats`, which returns ν, S, the NIS and the log-likelihood.
- `update` takes the same moments, reused from `innovation_statistics` when its result is
  passed back, and hands them to the correction core (§4.3).

`GaussianFilter` never branches on the type of its linearisation. The only thing that varies
downstream is whether H is `None`, and it is the correction core, not the filter, that chooses
the covariance form from that. A further linearisation, such as cubature or Gauss–Hermite, is a
new `Linearisation`, never a new filter.

**Periodic coordinates in the moments.** The unscented linearisation forms its means with the
layout's `weighted_mean` and its deviations with the layout's `residual`, so sigma points that
straddle a wrap (an azimuth near ±π) are averaged and differenced the short way round.
`UKF._innovation` does this today, and it moves into `UnscentedLinearisation` unchanged.

### 4.3 One correction core

All three filters compute the same three things for a measurement, and differ only in how:

| Filter | Predicted measurement ẑ | Cross-covariance P_xz | Innovation covariance S |
| :--- | :--- | :--- | :--- |
| KF | H x̂⁻ | P⁻ Hᵀ | H P⁻ Hᵀ + R |
| EKF | h(x̂⁻) | P⁻ Jᵀ, with J = ∂h/∂x at x̂⁻ | J P⁻ Jᵀ + R |
| UKF | Σ wₘ h(χᵢ) | Σ w_c (χᵢ − x̂⁻)(h(χᵢ) − ẑ)ᵀ | Σ w_c (h(χᵢ) − ẑ)(h(χᵢ) − ẑ)ᵀ + R |

Everything after that is the same, and lives in **one** private function in `estimation.py`,
which `GaussianFilter.update` calls. It takes the prior, the innovation statistics, P_xz and,
when the linearisation supplies one, the measurement matrix (H or J). It:

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

- the **Joseph form**, (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ, when the linearisation supplies H or J.
  It is positive semidefinite for any K, because both terms have the form A M Aᵀ with M
  positive semidefinite, so roundoff cannot make a variance negative. This is what
  `kalman.update` does today;
- the **short form**, P⁻ − K S Kᵀ, when it does not (the UKF). The UKF has no H. With every
  sigma-point *covariance* weight non-negative (§4.4), P⁻ − K S Kᵀ is the Schur complement of S in a
  positive semidefinite joint covariance, so it is positive semidefinite up to roundoff, which
  the symmetrisation contains. This is the argument in `UKF.update`'s docstring.

Both branches are in the same function, side by side, with this reasoning in its docstring
once. Today the argument is split across `kalman.py` (against the short form) and `ukf.py` (for
it), and each reads as if the other were wrong. They are not: they are about filters with and
without H.

The two forms agree in exact arithmetic at the optimal gain. Expanding the Joseph form with
S = H P⁻ Hᵀ + R and P_xz = P⁻ Hᵀ gives P⁻ − K P_xzᵀ − P_xz Kᵀ + K S Kᵀ. With K = P_xz S⁻¹, each
of the last three terms equals K S Kᵀ, so the sum is P⁻ − K S Kᵀ. They differ only in how
roundoff enters, which is why §4.4's reduction of the UKF to the KF holds to roundoff.

### 4.4 The linearisations

**`TaylorLinearisation`: the KF and the EKF.** The EKF is the KF with H replaced by the
Jacobian at the prediction (Bar-Shalom, Li and Kirubarajan 2001, §10.3), so one linearisation
gives both.

- **Motion.** It needs F and Q, so it accepts a *linear* motion model: one with
  `matrices(dt_s) -> (F, Q)`. `CartesianMotion.matrices` exists; `RadialMotion` gains it. Every
  motion model in the repo is linear, so there is no EKF on the motion side yet; a nonlinear
  motion model (coordinated turn) adds `jacobian(state, dt_s)` when it arrives. A
  `GaussianFilter` built with this linearisation and a motion model without `matrices`
  raises `TypeError` at construction, naming the model.
- **Measurement.** It needs H or J, so it accepts a measurement model that also has
  `jacobian(state) -> (n_meas, n_state)`. A linear model returns its constant H. Passing a
  model without `jacobian` raises `TypeError`, naming the model, at the first call, before
  anything changes. There is **no silent finite-difference fallback**: a numeric Jacobian
  hides a wrong analytic one, and a student cannot tell which filter they are running. Tests
  check each analytic Jacobian against a central difference (§12).
- **On a linear model this is the textbook KF**, line for line: x̂⁻ = F x̂, P⁻ = F P Fᵀ + Q,
  and the update of §4.3 with H. A student looking for the KF finds it here (TD2).

**`UnscentedLinearisation`: the UKF.** It keeps today's `UKF` algorithm (Wan and van der Merwe
2000, §3), and its sigma points move here from `ukf.py`.

- **Non-negative covariance weights.** The weights are w₀ᵐ = 1 − n/(α²(n + κ)) for the
  mean, w₀ᶜ = w₀ᵐ + 1 − α² + β for the covariance, and wᵢ = 1/(2α²(n + κ)) for every other
  point, in both. §4.3's short-form argument needs every *covariance* weight non-negative,
  because a weighted sum of outer products is positive semidefinite only then. The mean
  weights don't enter that argument: a negative w₀ᵐ changes how the mean is formed, not
  whether P stays positive semidefinite. So `check_sigma_point_settings` must reject settings
  with w₀ᶜ < 0, rather than only requiring n + κ > 0 as today, and its message must say why.
  - The defaults α = 1, β = 2, κ = 0 give w₀ᵐ = 0, w₀ᶜ = 2 and wᵢ = 1/(2n), for any n.
  - With α = 1 and β = 2, w₀ᶜ = κ/(n + κ) + 2, which is non-negative for κ ≥ −2n/3. The
    heuristic n + κ = 3 of Julier and Uhlmann (1997), κ = 3 − n, makes w₀ᵐ negative for n > 3, but keeps w₀ᶜ ≥ 0 up to n = 9, so it is accepted.
  - A small α, such as the 10⁻³ that Wan and van der Merwe (2000) call usual, makes w₀ᶜ large
    and negative, so it is rejected. That departs from the paper this UKF follows, on purpose:
    the short-form update of §4.3 is only safe without it.

**When the UKF is the KF.** On a linear motion model and a linear measurement model, with no
periodic state coordinate, the unscented linearisation returns exactly the Taylor moments, for
any accepted α, β and κ. The argument is short enough to put in `UnscentedLinearisation`'s
docstring, and it is the clearest way to see what the sigma points do. Write c = α²(n + κ), and
let Lᵢ be column i of the Cholesky factor of P, so that Σᵢ Lᵢ Lᵢᵀ = P.

1. **The points.** χ₀ = x̂, and χ±ᵢ = x̂ ± √c Lᵢ for i = 1 … n.
2. **The mean.** The mean weights sum to one: w₀ᵐ + 2n · 1/(2c) = 1. Through a linear map F,
   the ± pairs cancel, so Σ wₘ F χ = F x̂. The Taylor mean is the same.
3. **The covariance.** The centre point's deviation F χ₀ − F x̂ is zero, so w₀ᶜ multiplies zero.
   That removes the whole of β, and α's effect on w₀ᶜ. The other 2n points give
   Σᵢ 2 · (1/(2c)) · c F Lᵢ Lᵢᵀ Fᵀ = F P Fᵀ. Adding Q gives the KF's P⁻.
4. **The update.** The same steps with H in place of F give ẑ = H x̂⁻, P_xz = P⁻ Hᵀ and
   S = H P⁻ Hᵀ + R, the KF's row of §4.3's table.
5. **The covariance form.** The only difference left is that the UKF uses the short form and
   the KF the Joseph form. The two are equal at the optimal gain (§4.3).

So on linear models the UKF *is* the KF, up to roundoff. The weights matter only once g bends.
§4.5 tests both halves of this argument.

**The teaching layer.** The free functions `predict` and `update` in today's `kalman.py` are
worth keeping for notebooks, as the textbook equations on bare arrays. If kept, they call the
correction core; they are never a second implementation (TD2).

### 4.5 Drift guards

Principle 2 is enforced by tests that run every filter through the same checks. The list of
estimators, one `GaussianFilter` per linearisation plus `IMM` once it exists, is one
parametrised fixture, so a new linearisation joins every guard by being added once.

| Guard | Check | Tolerance |
| :--- | :--- | :--- |
| UKF moments ≡ KF moments | `CartesianMotion` with a `LinearMeasurement`: `UnscentedLinearisation` and `TaylorLinearisation` return the same x̂⁻, P⁻, ẑ, P_xz and S, for several accepted (α, β, κ): the defaults, κ = 3 − n, and β = 0. This is §4.4's argument, step by step | `rtol=1e-10`, with each `atol` 10⁻¹⁰ times the scale of that entry's prior variance, since the entries mix m² and (m/s)² |
| UKF ≡ KF over a run | `GaussianFilter` with each linearisation, same prior, same `LinearMeasurement`, 200 steps: the two estimates agree in mean and covariance | as above |
| EKF ≈ UKF on a nonlinear model | the spherical model of §6.1; scale the prior covariance by s², and the gap in the updated mean must shrink at second order in s, because it is ½ tr(∇²h P) to leading order | fitted slope of log gap against log s ≥ 1.8 (against log s², it would be 1) |
| Consistency, every estimator | NEES and NIS over Monte Carlo runs (§12) | chi-square interval at 99.9 % |
| Definiteness, every estimator | 200 steps, symmetric to `rtol=1e-12`, Cholesky succeeds | exact |

The first two guards replace today's `tests/core/tracking/test_ukf.py` check, which compares
the UKF against a hand-coded Kalman update rather than against `kalman.py`: two filters agreeing
with a third copy proves nothing about each other. The first guard also pins the claim a student
is most likely to doubt, that β and the weights do not matter on a linear model.

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
coordinates. The filter's linearisation, not the measurement, bridges the two: the Jacobian
for the EKF, sigma points for the UKF (§4.4). Converting measurements to Cartesian before
filtering is not allowed, except to seed a track (§6.1).

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

It provides the analytic Jacobian (§4.4). The azimuth row is singular at r → 0 and the
elevation row at the zenith; the docstring says so, and the model raises rather than return a
non-finite Jacobian.

**Converted measurements, for initiation only.** One spherical measurement can seed a Cartesian
track: position from (r, θ, φ), with a covariance from the polar-to-Cartesian Jacobian. At
long range and coarse angle accuracy, the naive conversion is biased and its covariance is
wrong; the initiator uses a debiased conversion instead. It returns the mean and covariance of
the true position given the measurement, the measurement-conditioned conversion of Duan, Han and
Li (2004) and Bordonaro, Willett and Bar-Shalom (2014), in 2-D (r, θ) and 3-D (r, θ, φ). For
Gaussian errors these moments are exact, and they are the posterior under a flat prior, which is
what a new track's prior should be. Lerro and Bar-Shalom (1993) for 2-D, and Suchomski (1999)
for 3-D, reach approximate moments by nested conditioning, which leaves a small bias. The
converted position seeds the prior; every later update uses the spherical model.

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
alone undercuts its full-dimension rivals in the solve. So when the dimensions in one matrix
differ, the cost is the negative log-likelihood, ½ νᵀ S⁻¹ ν + ½ ln |2π S|, the negative of the
`log_likelihood` that `innovation_stats` already returns. NIS stays available as a cost only
when every pair has the same dimension. If a scan would put NIS costs of different dimensions
in one matrix, `Tracker` raises rather than solve it. Today's scenario 003 code mixes raw NIS
across dimensions, so its numbers may move when it migrates (§13 step 6); the commit records
the move, as §12 requires.

### 7.3 Assignment order

Which tracks pick measurements first is measured to matter, and the better order depends on
the problem, so it is a setting of `Tracker`, `assignment: Literal["confirmed_first",
"joint"]`:

- `"confirmed_first"` assigns in two stages: confirmed tracks pick first, then tentative ones,
  and no track starts inside a confirmed track's gate (`tracker.py` module docstring). It
  protects established tracks from being stolen by new ones. It is the library default.
- `"joint"` solves once over every track.

On scenario 003, confirmed-first was worse on every count (range RMSE 25.3 m against 17.5 m;
fold selection 82 % against 86 %). A confirmed track whose gate is widened for re-acquisition
(§8.3) outbids tentative tracks for false alarms. So scenario 003 sets `"joint"` in its TOML.
Changing either default needs a measurement, recorded in the docstring. TD8.

### 7.4 JPDA

JPDA (Fortmann, Bar-Shalom and Scheffe 1983) updates each track with a weighted mixture of the
measurements in its gate, instead of one. It is admitted without changing `Tracker`'s loop:

- `AssociationResult` gains optional association weights, shape (n_tracks, n_measurements + 1),
  the last column for "no measurement". GNN and NN leave it empty.
- `Estimator` gains no method. JPDA's combined innovation and spread-of-innovations term are
  computed from each pair's `InnovationStats` by a helper in `association.py`, and applied
  through the correction core.

Scenario 003 §7 records GNN as a deliberate limit, and its §13.3 names a second target as what
makes JPDA or MHT necessary. That scenario is
[`scenario-011-terminal-area-tracking.md`](scenario-011-terminal-area-tracking.md), and its D28
specifies JPDA in `core/tracking/jpda.py`:

- Tracks are clustered by shared gated measurements, and each cluster's feasible events are
  enumerated exactly, with a bound that raises rather than approximates.
- `coalescence` selects textbook JPDA, JPDA\* (Blom and Bloem 2000) or Set-JPDA (Svensson et al.
  2011).
- The clutter intensity is passed per measurement, from an oracle or a clutter map.
- JIPDA (Mušicki and Evans 2004) propagates `Track.existence_probability`, which drives the
  `ExistencePolicy` of §8.2.

### 7.5 PMBM, beside the tracker

The Poisson multi-Bernoulli mixture filter (García-Fernández et al. 2018) models clutter, missed
detections and the birth of new targets inside the filter itself. It cannot sit inside
`Tracker`, for a structural reason: it replaces the split this spec is built on. In PMBM,
association hypotheses, track existence and birth are all part of the filter's density, so
there is no separate associator, initiator or lifecycle to swap. It is a second tracker, not a
fourth filter.

Scenario 011 (D30) builds it as exactly that: `PmbmTracker` in `core/tracking/rfs/pmbm.py`, a
**comparison tracker** beside `Tracker`. It shares the motion and measurement models,
`innovation_stats`, Murty's k-best assignment, the clutter intensity and the metrics, and owns
its density. It earns its place as the reference score. Williams (2015) shows MHT and JIPDA
approximate the same conjugate prior, so PMBM measures what they lose. GLMB stays out: one RFS
reference is enough to score against.

### 7.6 MHT

Track-oriented MHT (Reid 1979; Kurien 1990; Blackman 2004) defers association by keeping
several association histories per target. Scenario 011 (D29) fits it inside `Tracker`'s one
loop:

- `TrackTree` satisfies `Estimator`: its leaves are estimators, and its `state` is the leaf in
  the current best global hypothesis.
- `TrackOrientedMHT` is an associator that finds the k best global hypotheses per cluster by
  Murty's algorithm over `linear_sum_assignment`, branches the trees, and prunes by N-scan.
- Score-based confirmation replaces M-of-N for MHT, through the selector of §8.3.

The estimate reported at a scan is the best hypothesis at that scan; MHT may later prefer
another, and reported tracks are not rewritten. With `n_scan = 1` and `k_best = 1` it reduces to
GNN with a finite non-assignment cost (§7.2), which is its first test.

## 8. Track lifecycle

### 8.1 States

`tentative → confirmed ⇄ coasting → deleted`, the vocabulary of `spec/data-001-formats.md` §5.
A confirmed track that misses becomes coasting; a hit makes it confirmed again.

### 8.2 Rules

All lifecycle settings live in `LifecyclePolicy`, once. Pipeline configs hold a
`LifecyclePolicy` (or build one from their TOML table); they never copy its fields one by one
(the PR #13 review found them declared twice and copied across).

The counts below use *hit* and *miss* per scan: a track scores a hit in a scan where a plot
was assigned to it, and a miss in one where none was. This M-of-N sense, counted per track per
scan, is the only one the specs use (`spec/data-001-formats.md` §5 Vocabulary).

| Rule | Setting | Status |
| :--- | :--- | :--- |
| M-of-N confirmation, counted from birth | `n_confirm_hits`, `n_confirm_frames` | exists |
| Tentative deletion once M is unreachable | (derived) | exists |
| Deletion after misses in a row | `n_delete_misses` | exists |
| Re-acquisition window for confirmed tracks | `n_reacquire_misses` (renamed, TD13) | exists: the count in `LifecyclePolicy`, and the reset that widens the gate in `Tracker` (§8.3, step 5) |
| Deletion after a coast time | `max_coast_time_s` | exists |
| **Covariance-based deletion** | `max_position_sigma_m` | **new** |
| **Existence-based confirm and delete** (JIPDA only) | `ExistencePolicy(confirm, delete)` | **new**, scenario 011 D28 |

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
`docs/reviews/pr1-review.md` #24) stay out of `Track` until a decision uses them. Scenario 011
supplies the first two such decisions: JIPDA's existence drives `ExistencePolicy` (confirm at
0.95, delete below 0.05 in that scenario), and MHT's score drives its Wald-threshold
confirmation (§7.6). `existence_probability` and `score` enter `Track` with them, optional and
`None` for every other associator. `age` stays out.

### 8.3 Range-only bootstrap, retry and re-acquisition reset

A track whose Doppler cannot yet be trusted (§6.2) needs three behaviours from `Tracker`: to
start on range alone, to fall back to range alone when its velocity gate fails, and to forget
its range rate when it starts re-acquiring. The mechanism goes in `core/tracking/`, generic
and named by coordinates; the scenario's policy goes in `pipelines/` (TD9).

| Behaviour | Core mechanism | Scenario policy |
| :--- | :--- | :--- |
| Per-track range-only bootstrap | a measurement model can return its **marginal** over a subset of its components, by name | the pipeline passes a selector: a callable from a `TrackSnapshot` and a `Measurement` to the ordered model IDs to try |
| Range-only retry when the velocity gate fails | `Tracker` gates each pair against the selector's models in order, and keeps the first that passes | scenario 003's selector returns (range and rate, range only) |
| Velocity reset on entering re-acquisition | on the miss that starts the re-acquisition window, reset the named coordinates' variances to the initiator's prior and zero their cross-covariances | `LifecyclePolicy.reacquire_reset = ("range_rate_mps",)` |

The default selector returns the measurement's own model only, so every other user of `Tracker`
sees no change. The reset finds its coordinates by name, never by slicing (principle 3).

## 9. Multiple models (IMM)

IMM runs a bank of filters with different motion models and mixes them by mode probability
(Blom and Bar-Shalom 1988; Bar-Shalom, Li and Kirubarajan 2001, ch. 11). Its shape:

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
IMM only once `sigma_accel_mps2` is shown to be the binding error on the circuit's turns. That
scenario is [`scenario-011-terminal-area-tracking.md`](scenario-011-terminal-area-tracking.md):
D26 specifies `CoordinatedTurn` (turn rate in the state) and D27 the `IMM`. Two details it adds:

- The mode transition matrix is `expm(Λ T)` from a generator in s⁻¹, formed for each actual
  interval, because a rotating radar's sector batches make intervals unequal.
- Mode probabilities are exposed as an intermediate quantity and written to `tracks.csv`.

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

The tracker must be readable and fast (principle 10, TD14). The cost of a scan is dominated by
the gate: every track scored against every measurement, n_tracks × n_measurements innovation
statistics, each a small dense solve. That is the hot path, and it is batched.

- **Batch across tracks and measurements on the hot path.** The (n_tracks, n_measurements)
  gate and cost matrices come from one batched computation of the innovation statistics, not
  a Python loop over pairs. Both linearisations broadcast over leading axes: the Taylor
  linearisation's Jacobians as (n_tracks, n_meas, n_state), the unscented linearisation's
  sigma points as (n_tracks, 2 n_state + 1, n_state). The one flow of §4 is what makes this a
  single batched path, rather than one per filter.
- **Broadcast the linear algebra.** `np.linalg.cholesky`, `np.linalg.solve` and the triangular
  solves of `innovation_stats` take stacks of matrices over leading axes. The batched
  functions keep the textbook names and equations, and their docstrings give every shape.
- **State the cost.** Gating is O(n_tracks · n_measurements · n_meas³) floating-point work with
  no per-pair Python overhead. Assignment is `scipy.optimize.linear_sum_assignment` (§7.2),
  O(n³) in the larger side of the cost matrix.
- **Loops are allowed off the hot path**, each with a comment saying why (style.md §7): the
  lifecycle bookkeeping, the one update per assigned track, and tracks of different layouts,
  which are batched per layout.
- **Measured, and held.** `benchmark`-marked tests time a scan of scenario 003's load and a
  synthetic scan of 100 tracks and 1000 measurements. Each asserts a budget: it fails when a
  scan takes longer than a stated factor of a baseline committed beside it. The PR that
  batches the gate (§13 step 8b) measures the baseline, and this section then records the
  number, the factor and the machine.
- **No compiled dependency in core.** NumPy with batched linear algebra is the
  implementation. A compiled backend (numba, JAX) may come only when a benchmark shows NumPy is
  the binding limit, and then as an optional extra behind the same API, per CLAUDE.md.

## 12. Validation strategy

| Level | What | How |
| :--- | :--- | :--- |
| Unit, analytic | dead reckoning; Q against the published matrices; a known GNN optimum; M-of-N by hand; Jacobians against central differences | exact or `rtol=1e-12`; Jacobians at the tolerance `tests/core/test_tracking.py` already justifies |
| Unit, statistical | gate acceptance rate | Clopper–Pearson interval, fixed seed |
| Filter consistency | NEES and NIS over 200 runs × 30 steps, every linearisation, and IMM once it exists (§4.5), per Bar-Shalom, Li and Kirubarajan (2001) §5.4.2 | chi-square interval at 99.9 %, `slow` |
| Drift guards | §4.5 | as tabled there |
| Lifecycle | confirmation, deletion, re-acquisition, covariance deletion on hand-countable sequences | exact |
| End to end | scenario 003 over S1, and its dual-PRF variant over S3 | `spec/scenario-003-tracking.md` §12 and §14.7, unchanged by migration |
| Track metrics | range and rate RMSE, mean NIS, OSPA (Schuhmacher, Vo and Vo 2008), track breaks | `spec/data-001-formats.md` §6.9 |
| Batched ≡ per pair | the batched gate's NIS and log-likelihood against `innovation_stats` called pair by pair | `rtol=1e-12` |
| Performance | time per scan: scenario 003, and the synthetic 100 × 1000 scan (§11) | budget against the committed baseline, `benchmark` |

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
2. **One innovation API, the correction core and `GaussianFilter`** (§4.2, §4.3).
   `InnovationStats.residual` becomes `innovation`. The `Linearisation` protocol and
   `UnscentedLinearisation` land, with today's `UKF` algorithm moved into them unchanged, and
   `GaussianFilter(motion, UnscentedLinearisation())` replaces `UKF` wherever one is built
   (`build_tracker`, `build_tracker_enu`, the tests). The UKF's numbers do not move. `UKF` and
   `ukf.py` are deleted here, not in step 7: once nothing builds a `UKF`, it is a dead second
   copy of the update, the drift principle 2 forbids. `Linearisation` has `check_motion_model`
   (§4.2), so a bad pairing fails at construction without the filter asking which
   linearisation it holds.
   *Check:* AC2, and the existing UKF tests pass against `GaussianFilter` without a tolerance
   change.
3. **`TaylorLinearisation`** (§4.4), with `RadialMotion.matrices` and analytic Jacobians for
   `CartesianPosition` and `BistaticRangeDopplerModel`, so every existing model runs under
   every linearisation and the definiteness tests cover both (AC5). The KF and the EKF are now
   `GaussianFilter(motion, TaylorLinearisation())`, and the two UKF ≡ KF guards (§4.5) land
   here. Today's `KalmanFilter` (`kalman.py`) holds a track's estimate and is not a filter. It
   keeps its name until step 7 deletes it, since no new class takes the name.
   *Check:* AC3, AC4 and AC12.
4. **The spherical measurement model** (§6.1) with its Jacobian and the debiased initiator,
   and the EKF ≈ UKF guard of §4.5, which runs on it. The geometry is
   `geodesy.enu_to_range_azimuth_elevation_rad`, which the degree form now calls.
   `kalman._enu_model` keeps its own copy until step 7 deletes it. The initiator,
   `DebiasedSphericalInitiator`, uses the measurement-conditioned conversion rather than Lerro's
   and Suchomski's nested conditioning (§6.1). The guard scales the prior by s² over
   s ∈ [1/8, 1], where J P Jᵀ dominates R; at much smaller s the gap falls as s⁴, and the
   slope bound would test nothing.
   *Check:* AC3, AC5 and AC6 for the new model; the §4.5 guard passes.
5. **Scenario mechanisms in `Tracker`** (§8.3), and `assignment="joint"` (§7.3).
   The marginal is a wrapper, `marginal(model, names)` in `measurement_models.py`, not a method
   every model must implement: it is a `DifferentiableMarginalModel` exactly when its base has
   a Jacobian, so the KF and the EKF can use it. The selector may offer only a measurement's own
   model or a registered marginal of that model (`MarginalModel.base is model`), so a model that
   merely shares component names cannot stand in. The reset reads the prior from the
   initiator, so `TrackInitiator` gains a `prior` attribute, which both initiators already had.
   `"joint"` also starts a track from every measurement left over, inside a confirmed gate or
   not, as `KalmanTracker` does; the no-birth rule belongs to `"confirmed_first"`. An error
   while predicting or gating, such as the §7.2 refusal to compare NIS across dimensions, puts
   every track back as it was (principle 8); before this step an error there left the
   predictions already made.
6. **Scenario 003 on `Tracker`.** `pipelines/tracking.py` becomes an adapter: plots in,
   batches out, snapshots back. `TrackingConfig.state_model` is honoured (it is ignored today).
7. **Delete the old stack:** `KalmanTracker`, `KalmanFilter`, `KalmanState`, `TrackModel` and
   `FrameResult` (`UKF` and `ukf.py` went in step 2), and the rest of TD13's renames. Two of
   those renames are scenario TOML keys (TD13), so this step also updates the scenario 003
   TOMLs and spec, and the data-001 `metadata.json` schema. Module moves (§3 target
   column). Refresh the stale text in `spec/structure.md` D2 (steps 1 and 2 refreshed
   `kalman.py`'s "promotes this module to a subpackage" and `estimation.py`'s "the one
   implementation"). *Check:* AC1, AC7 and AC8;
   `git grep -nE '^class (KalmanTracker|KalmanFilter|KalmanState|TrackModel|FrameResult|UKF)\b' -- src`
   returns nothing; and so does
   `git grep -nE 'n_reacquire_frames|\bCartesianPosition\b|\bRadialMotion\b|def normalise|\.normalise\(|\.noise_density\b[^_]|self\.noise_density\b|sigma_accel_mps2' -- src tests scripts scenarios spec/scenario-003-tracking.md docs/tracking`.
   If `kalman.py` survives as the teaching layer of §4.4, its module docstring says so; today
   it says "a linear Kalman filter", but the ENU models make it an extended one (`:138`,
   `:479`).
8. Two steps independent of the rest, each its own PR, both before step 9:
   - **8a. Covariance-based deletion** (§8.2). Any time after step 2.
   - **8b. Batched innovation statistics and gating** (§11). Any time after step 3. The
     batched gate replaces the per-pair loop in `Tracker`, the benchmarks gain their budget,
     and §11 records the baseline. *Check:* AC13.
9. **Audit the finished package.** The audits of the rest of the library
   ([`docs/conventions/audit.md`](../docs/conventions/audit.md)) left tracking out, because steps 0–7 replace most
   of it. Once step 7 has merged, run both audits over the tracking code as it then stands:
   - **Code against spec**, Q1–Q8 ([method](../docs/conventions/audit.md#1-auditing-code-against-the-spec)), over `src/radar_forge/core/tracking/**`. The reference
     is this spec and [`docs/tracking/README.md`](../docs/tracking/README.md). Record the result
     in a new `docs/audits/tracking-audit.md`, numbering findings from F1 and R1.
   - **Test audit and pruning**, Sound / Redundant / Weak / Wrong, over
     `tests/core/tracking/**`, `tests/core/test_tracking.py` (or what survives it),
     `tests/pipelines/test_tracking.py` and `tests/pipelines/test_scenario_003.py`. Delete the
     Redundant ones; rewrite or fix the Weak and Wrong ones. The rules are
     [`testing.md` §10.10](../docs/conventions/testing.md#1010-pruning-a-suite-review), P1–P6:
     the §8 floor holds, every deletion names the test that still
     catches its bug, and `src/` line and branch coverage do not drop. Record it in
     `docs/audits/tests-audit.md`.

   *Check:* AC11.

## 14. Decisions

### 14.1 Design decisions

### TD1 — One tracker

`Tracker` is the one tracker class and the one scan loop. Every filter, associator, initiator
and lifecycle policy plugs into it through §3's protocols, and every pipeline uses it. The
simple version for teaching is the free functions of §4.4, on bare arrays.

### TD2 — One Gaussian filter, interchangeable linearisations

§4.2. KF, EKF and UKF are one class, `GaussianFilter`, with one predict and one update flow,
and differ only in the `Linearisation` the estimator factory binds in. On a linear model the
Taylor linearisation is the textbook KF line for line, so a student looking for the KF finds
it; §4.4 shows, and §4.5 tests, that the unscented linearisation gives the same answer there.
*Rejected:* two filter classes sharing only a correction core (two predict and two update
flows, which is where they drift); a filter-type switch inside `GaussianFilter` or `Tracker`
(breaks §3's decoupling rule); a KF implemented as a UKF on linear models (correct, but hides
the textbook KF a student is looking for).

### TD3 — One correction core, two covariance forms

§4.3. Joseph form when the linearisation supplies H or J, short form for the UKF, in one
function. *Rejected:* Joseph form everywhere (the UKF has no H; a statistically linearised H
can be built, but adds a solve for no gain under §4.4's weights); the short form everywhere
(gives up the KF's guarantee at no saving that matters here).

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

§7.3. `Tracker` takes `assignment="confirmed_first"` or `"joint"`. *Rejected:* fixing one order
for all scenarios. Each is measured better on a different problem.

### TD9 — Mechanism in core, scenario policy in pipelines

§8.3. *Rejected:* moving scenario 003's bootstrap and re-acquisition into `Tracker` as named
scenario features (couples core to one scenario); leaving them in `pipelines/` around a
`Tracker` that cannot express them (the pipeline would re-implement gating).

### TD10 — Covariance-based deletion

§8.2. *Rejected:* relying on miss counts and coast time alone; they need retuning whenever the
scan rate or target speed changes.

### TD11 — IMM, JPDA and MHT inside the loop; PMBM beside it

§7.4, §7.5, §7.6, §9. *Amended by scenario 011.* As first written, this decision kept IMM and
JPDA as extension points and ruled PMBM out, because no scenario needed them. Scenario 011 is
the scenario that needs them, so they are specified there (D26–D30):

- IMM, JPDA and MHT fit `Tracker`'s one loop, as an estimator, an associator and a
  tree-as-estimator respectively.
- PMBM does not fit (§7.5) and is built as a comparison tracker beside it.

*Rejected:* PMBM inside `Tracker` (it has no associator, initiator or lifecycle to swap); GLMB as
a second RFS reference (one is enough to score against).

### TD12 — `Plot` for the pipeline's detection report

`pipelines.tracking.Measurement` is renamed `Plot`. The repo's vocabulary
(`spec/data-001-formats.md` §5) defines a *detection* as a connected group of threshold
crossings, in cell indices (`core.detection.Detection`), and a *plot* as a detection converted
to physical units. This class holds detections in metres and metres per second, with a
`status`, so it is a plot. ASTERIX CAT048
("Monoradar Target Reports") agrees: it sends a target report either as a plot or as a track,
and a plot is the untracked form. *Why:*
two public classes named `Measurement` in one package tree, holding different things, is a
trap. `core.tracking.Measurement` keeps the name, as the textbook word for z. (Stone Soup's
equivalent class is `Detection`, which here is `core.detection.Detection`, one step earlier.)

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
| `UKF` | `GaussianFilter(motion, UnscentedLinearisation())` | TD2: one filter, the linearisation is an argument |
| `kalman.KalmanFilter` (a track's estimate) | deleted (§13 step 7) | it is not a filter; `GaussianFilter(motion, TaylorLinearisation())` is the KF |
| `build_tracker`, `build_tracker_enu` | same names. `build_tracker` takes an `estimator_factory`; `build_tracker_enu` takes a `linearisation`. Both default to the unscented linearisation | both always build a UKF today. Per TD2, the filter is one argument, not one builder per filter. `build_tracker_enu` builds its motion model itself, so a caller has no model to bind into a factory, and passes the linearisation instead |
| `UKF.predict_to(time_s)` | `predict_to(timestamp_s)` | the `Estimator` protocol's name |
| `StateLayout.normalise` | removed | it is `wrap` under a second name |
| `CartesianMotion.noise_density` | `acceleration_noise_density_m2ps3`, `jerk_noise_density_m2ps5` | one dict held two units |
| `sigma_accel_mps2` | `sigma_acceleration_mps2` | as `motion.py` spells it |
| `pipelines.tracking.Measurement` | `Plot` | TD12 |

Textbook names stay: `ChiSquareGate`, `NearestNeighbour`, `GlobalNearestNeighbour`,
`TrackInitiator`, `StateEstimate`, `TrackSnapshot`, `Measurement`, `TrackManager`,
`BistaticRangeDopplerModel`. `GaussianFilter` is Särkkä's (2013) name for the family; its
docstring and the two linearisations' docstrings name the KF, the EKF and the UKF, so a student
searching for any of them lands on the right class.

`sigma_accel_mps2` and `n_reacquire_frames` are also keys in the scenario 003 TOMLs'
`[tracking]` table (`spec/scenario-003-tracking.md`, "Parameters as shipped"), and
`metadata.json` records them. So renaming them changes a file format, not only code:
the TOMLs, scenario 003's spec and `spec/data-001-formats.md` change in the same PR, the
`metadata.json` schema version is bumped, and a TOML that still uses an old key fails with a
message naming the new one, rather than being silently ignored.

### TD14 — Readable and fast

Principle 10, §11. The hot path is batched across tracks and measurements, and benchmarks
hold a budget against a committed baseline. *Rejected:* readable before fast (leaves a Python
loop over every track-measurement pair in the gate, which a realistic scan cannot afford);
fast at the cost of reading like the textbook (a student must still match each batched line
to its equation); compiled dependencies in core (an optional extra, only once a benchmark shows
NumPy is the limit).

### 14.2 Rulings on the requested features

| Requested | Ruling | Where |
| :--- | :--- | :--- |
| Strict decoupling of models, filters, association, lifecycle | **Adopted.** Already the shape; §3 makes it a rule | §3, TD2 |
| Polar/spherical to Cartesian | **Adopted.** Spherical model with Jacobian; debiased conversion for initiation only | §6.1, TD6 |
| Doppler in the state vector | **Adopted as corrected:** in the measurement and the gate, not the state | §1.3, §6.2, TD7 |
| GNN, JPDA, PMBM | **GNN adopted** (exists); **JPDA** and MHT specified by scenario 011; **PMBM** as a comparison tracker beside `Tracker` | §7, TD11 |
| M-of-N lifecycle, covariance deletion | **Adopted.** M-of-N exists; covariance deletion is new | §8, TD10 |
| High-performance implementation | **Adopted:** batched across tracks and measurements, with a benchmark budget; readable and fast | §11, TD14 |
| Validation strategies | **Adopted:** analytic, statistical, drift guards, end to end | §4.5, §12 |

## 15. Acceptance criteria

| # | Criterion | Checked by |
| :--- | :--- | :--- |
| AC1 | One tracker and one filter: `git grep -nE '^class (KalmanTracker\|KalmanFilter\|KalmanState\|TrackModel\|FrameResult\|UKF\|ExtendedKalmanFilter)\b' -- src` returns nothing | [review] |
| AC2 | One gain and covariance update: `git grep -n 'np.linalg.solve' -- src/radar_forge/core/tracking` finds no gain computed outside the correction core | [review] |
| AC3 | `GaussianFilter` satisfies `Estimator` with each linearisation, and a test runs `Tracker` with each | [test] |
| AC4 | Both UKF ≡ KF guards of §4.5 pass through the real `GaussianFilter`: the per-step moments for several (α, β, κ), and the 200-step run | [test] |
| AC5 | The NEES/NIS and definiteness tests are parametrised over every linearisation | [test] |
| AC6 | Every analytic Jacobian matches a central difference | [test] |
| AC7 | No tracking code indexes a state by position: a reviewer finds no slice of a state or covariance by integer outside `StateLayout` | [review] |
| AC8 | Scenario 003's end-to-end criteria (`spec/scenario-003-tracking.md` §12) pass on `Tracker`, and every number that differs from the scenario's output before migration is explained in the commit that moved it (§12; §7.2's cost change may move some) | [test] |
| AC9 | `core/tracking/` imports nothing from `pipelines/` | [review] |
| AC10 | This spec passes `scripts/check_conventions.py` R7 | [hook] |
| AC11 | Step 9 is done: `docs/audits/tracking-audit.md` has a verdict for every public name in `core/tracking/`, and `docs/audits/tests-audit.md` classifies every tracking test, with the coverage comparison (`docs/conventions/audit.md`, `testing.md` §10.10) | [review] |
| AC12 | One flow: `GaussianFilter` and the correction core hold no `isinstance` check, or other branch, on the type of a linearisation; the core's only branch is on whether H is `None` | [review] |
| AC13 | The gate is batched: no Python loop over track-measurement pairs in `Tracker` or `association.py`, the batched ≡ per-pair test of §12 passes, and the `benchmark` tests assert §11's budget | [test] |

## 16. Limitations

These hold after migration; each is stated in a docstring, not discovered by a user.

- Every measurement in a batch carries the batch's time; batches must arrive in time order.
  Out-of-sequence measurements are rejected, not buffered.
- GNN gives each track at most one measurement per scan. One target giving several plots in
  a scan starts several tracks; the fix belongs in clustering, in detection
  (`test_several_detections_of_one_target_give_one_confirmed_track` is `xfail`).
- A target crossing the line of sight is lost in the Doppler blind zone.
- No initiator for the bistatic model: one bistatic measurement cannot fix a position. Seed
  such tracks with `Tracker.seed`.
- Expect track fragmentation with the default settings; it is measured once the runner returns.
- No Singer model. IMM, the coordinated turn, JPDA, MHT and PMBM are specified by scenario 011
  and not yet implemented (§7.4–§7.6, §9).

## References

- Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and
  Navigation*, Wiley, 2001. §5.2 (linear estimation in dynamic systems: the Kalman filter, and
  the properties of the innovations), §5.4.2 (the statistical tests for filter consistency),
  §6.2.2 (the continuous white noise acceleration model), §6.3.2 (the discrete white noise
  acceleration model), §10.3 (the extended Kalman filter), ch. 11 (adaptive estimation and
  maneuvering targets, including the interacting multiple model estimator).
- S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013. Ch. 5
  (the extended and unscented Kalman filters) and ch. 6 (general Gaussian filtering: one filter,
  with the Taylor and unscented approximations as instances). Chapter numbers are from the first
  edition and are to be checked against the copy cited in the code.
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
- Z. Duan, C. Han and X. R. Li, "Comments on 'Unbiased converted measurements for tracking',"
  *IEEE Trans. Aerosp. Electron. Syst.*, vol. 40, no. 4, pp. 1374-1377, 2004 (the
  measurement-conditioned moments).
- S. Bordonaro, P. Willett and Y. Bar-Shalom, "Decorrelated unbiased converted measurement
  Kalman filter," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 50, no. 2, pp. 1431-1444, 2014.
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
