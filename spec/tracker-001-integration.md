# Tracker 001 — configurable state spaces and Duke integration

## Intent and boundaries

A native, NumPy/SciPy-only tracker lives in `radar_forge.tracking`. Its numerical
interfaces do not assume a state dimension, coordinate ordering, or sensor modality.
StoneSoup informs model composition, FilterPy informs numerical validation, and Norfair
informs detector/tracker separation. The local `unified-extensible-tracker` supplies useful
semantic spaces, asynchronous orchestration, lifecycle rules, UKF and compatible-state IMM.
This is an adapted implementation, not a dependency on the sibling checkout.

The original Scenario 003 tracker remains available under `radar_forge.core.tracking` and
`radar_forge.pipelines.tracking`. The general tracker is additive; its IQ adapter lives in
`radar_forge.pipelines.general_tracking`, its detector in `radar_forge.core.detection_2d`.
The radial IQ runner accepts monostatic bursts only; general bistatic measurements are
available through Python composition of `BistaticRangeDoppler` and `TrackerEngine`.

The first built-ins cover any nonempty subset of east/north/up with CV or CA independently
on each axis, and radial tracking for all three Scenario 001 variants. Custom models may
use other coordinates when they supply dynamics, noise, observation and initiation rules.
Changing a live track's dimension, mixed-state IMM, PF, JPDA, MHT, smoothing, moving radar
assets and multi-target dual-PRF pairing are deferred. No placeholder claims these features.

## State and numerical contracts

`StateSpace` records ordered coordinate definitions (name, unit, optional period) and frame
identity. ENU states also identify their geodetic origin. Public numerical vectors are
float64 `(n,)`; covariance is `(n,n)` with coordinate-product units. Estimates and snapshots
own detached read-only arrays. Public functions have SI names, complete types and cited
NumPy-style documentation. Constants and geodesy are reused from Radar-Forge.

| Built-in | Default coordinate ordering |
| :--- | :--- |
| 1D CV | `x_m, xdot_mps` |
| 2D CV | `x_m, xdot_mps, y_m, ydot_mps` |
| 3D CV | `x_m, xdot_mps, y_m, ydot_mps, z_m, zdot_mps` |
| 3D CA | `x_m, xdot_mps, xddot_mps2, y_m, ydot_mps, yddot_mps2, z_m, zdot_mps, zddot_mps2` |
| Radial CV | `range_m, radial_velocity_mps` |

Any axis subset and explicit permutation is supported. Per-axis transition blocks are
assembled using names, not vector offsets; full state covariance is preserved. CV uses
continuous white acceleration (density m²/s³), CA continuous white jerk (m²/s⁵). Missing
coordinates are never assumed zero. Measurement geometry requiring omitted dimensions
must receive explicit fixed values or reject the model. Cartesian velocity is the position
derivative; radial velocity is positive closing, so radial transition has `F[0,1] = -dt_s`
and the corresponding negative process-noise cross terms.

Derived outputs: sensor-relative range, closing radial velocity, azimuth clockwise from
north, elevation above horizontal, 3D speed magnitude, 3D acceleration magnitude, horizontal
acceleration magnitude, signed up acceleration, and east/north velocity divided by horizontal
speed. Acceleration names end in `_mps2`; normalized components in `_norm`. Values with
missing source coordinates or singular geometry carry `None` and a reason. CV does not
pretend to estimate acceleration. Normalized horizontal velocity is unavailable at zero
horizontal speed. Derived values are not independently estimated duplicate coordinates.

Custom coordinates require a complete motion and measurement model and an initiation route.
Periodic quantities supply residual and weighted-mean semantics. Models must be defined over
the sigma-point region. Broad angular distributions and near-singular geometry remain limits
of Gaussian filtering. IMM modes share the exact space, frame, order and timestamp.

## Subsystem decisions and attribution

| Subsystem | Reference and retained behavior | Deliberate adaptation |
| :--- | :--- | :--- |
| State/model composition | StoneSoup combined transition models and explicit measurement mappings; local semantic spaces | Named coordinates, units, arbitrary permutations and explicit missing-coordinate policy |
| UKF and IMM | Julier's scaled unscented transform; FilterPy; local additive-Q regeneration and mixture likelihood | Strict float64 types and shared roundoff-only covariance policy |
| Association | StoneSoup association separation; local NN/GNN | GNN maximizes gated cardinality then minimizes cost; no clutter likelihood claim |
| Lifecycle | Local sensor coverage, routed initiation, M-of-N confirmation | Preserved event semantics and bounded detached histories |
| Detection adapter | Norfair detector/tracker separation | Radar-specific SI measurements, CFAR covariance and ambiguity provenance |
| Geometry/configuration | Radar-Forge geodesy, TOML conventions | No duplicated Earth constants, Pydantic, JSON configuration or sibling imports |

References are architectural/equation sources, not vendored runtime libraries. Adapted source
retains applicable notices. `tracker-001-provenance.md` records SHA-256 hashes of local input
files because the supplied directory has no `.git`. Upstream source adaptation, if any, must
record its revision and licence notice; independent equation implementations cite their source.

References:

1. [StoneSoup transition models](https://stonesoup.readthedocs.io/en/latest/stonesoup.models.transition.html)
   and [measurement models](https://stonesoup.readthedocs.io/en/latest/stonesoup.models.measurement.html).
2. [FilterPy process noise](https://filterpy.readthedocs.io/en/latest/common/common.html) and
   [UKF](https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html).
3. [Norfair](https://github.com/tryolabs/norfair).
4. S. Julier, *The scaled unscented transformation*, ACC 2002.
5. Y. Bar-Shalom, X. Li, T. Kirubarajan, *Estimation with Applications to Tracking and
   Navigation*, Wiley, 2001 (continuous motion noise, IMM and association).
6. M. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., 2014 (CA-CFAR).

## Pipeline and defaults

`IQ → range–Doppler products → CA-CFAR → peaks → measurements → tracker → exports`.
Tracker interfaces receive no truth. Simulation, evaluation and plotting may consume truth.

2D CA-CFAR uses linear power, two guard cells and eight training cells per side per axis,
and nominal per-cell Pfa `1e-6`. Doppler wraps; only S2 range wraps. Other range edges with
incomplete training windows are excluded. Deterministic peak suppression collapses plateaus.
The exponential independent-noise threshold is calibrated on independent noise; matched-filter
correlation is explicitly outside that assumption. Measurement covariance starts at squared
bin width / 12 plus configurable physical uncertainty floors.

| Scenario | Update and interpretation |
| :--- | :--- |
| S1 | Unambiguous range only; velocity learned over time. Folded measured Doppler exported for comparison. |
| S2 | Folded range and closing velocity; shortest modulo-range innovation. Internal continuous branch is not absolute range; exports mark it unresolved. |
| S3 | Unique range-compatible pair, dual-PRF unfolding, one observation per frame/pair. |

S3 requires range agreement within one resolution cell, max speed 191 m/s and residual tolerance
twice the coarser Doppler bin. Reject ambiguous pairings and nonfinite unfolding; a missing pair
is a missed observation. Use leg A range/velocity quantization covariance and retain both legs'
provenance. Never update twice from the same pair. Reuse `core.ambiguity`.

UKF defaults `(alpha,beta,kappa)=(0.5,2,0)`, GNN gate probability 0.997, radial acceleration
noise density 1 m²/s³. S1 initiation has velocity mean 0 and standard deviation 100 m/s.
Confirmation requires 3 of the last 5 eligible events, birth included. Five consecutive misses
or a gap **greater than** 30 seconds deletes a track. History holds 100 snapshots. Tracks outside
coverage do not accrue misses. Older events are rejected before mutation; equal times use caller
order. Each positive-time IMM prediction performs one transition (per event, not per second).
Custom component exceptions are not transactional.

Optional `[detection]` and `[tracking]` TOML settings in a separate `--tracker-config` file
configure `scripts/run_general_tracking.py`, which enables tracking by default without
changing ordinary scenario execution. Numerical configuration lives in typed pipeline dataclasses.
Core imports require only NumPy/SciPy. Plotting is optional and imported lazily.

Stream `detections.csv`, `tracks.csv`, and `tracking_metrics.json`; include measured values,
covariance, coordinate definitions, lifecycle status, pairing/ambiguity status, and resolved
configuration in metadata. S2 error is modulo range, never absolute range. Generic ENU exports
carry configured ordering. Optional plots show range/velocity histories.

## Validation and delivery

1. Install uv, run `scripts/setup-dev.sh`, establish `make check` baseline.
2. Implement mapped spaces/models, estimators, association and lifecycle with analytic tests.
3. Add detection, scenario adapters, streaming exports and synthetic ENU examples.
4. Run all three 120-second Duke windows and `make check`.

Required tests cover every axis subset, CV/CA/mixed dynamics, permutations and cross covariance,
analytic process noise, linear-Gaussian UKF equivalence, custom-coordinate extension, missing
coordinate validation, angle/velocity conventions, derived singularities, IMM compatibility,
covariance validation, asynchronous events, lifecycle, snapshots, CFAR calibration and wrapping,
S1 convergence, S2 wrap continuity, S3 rejection, and truth-metadata independence.

Synthetic radial fixtures: range RMSE below one resolution cell after initialization (modulo for
S2), S2/S3 velocity RMSE below one coarser Doppler bin, and S1 velocity RMSE below 5 m/s after
30 seconds in a 120-second CV fixture. Separate synthetic measurement examples prove 1D/2D/3D
ENU tracking. The real Duke trajectory validates plumbing/continuity; report errors without
claiming its interpolated radial velocity is exact physical truth. Do not loosen existing tests.

## Implemented interfaces and runnable configuration

Public numerical contracts are re-exported by `radar_forge.tracking`.
`CartesianMotion(axes, origin_lla_deg_m=..., order=..., noise_density=...)` builds the
mapped CV/CA dynamics. `build_enu_tracker` supplies position-only observation and explicit
unobserved priors. `ModelRegistry` registers application-specific motion/measurement builders;
`build_tracker(..., initiator=...)` supplies custom birth rules. Direct Python composition of
`TrackerEngine` supports multiple sensors and routes, custom estimators and IMM.

`derive_kinematics` returns `DerivedValue(value, unit, reason)` entries, not a redundant
augmented state. `snapshot_record` exports arbitrary named states and their complete covariance.

Optional settings in a separate tracker TOML passed with `--tracker-config`
(all omitted values retain defaults):

```toml
[detection]
guard_cells = 2
training_cells = 8
pfa = 1e-6
range_std_floor_m = 0.0
velocity_std_floor_mps = 0.0

[tracking]
association = "GNN"
alpha = 0.5
beta = 2.0
kappa = 0.0
gate_probability = 0.997
acceleration_noise_density_m2ps3 = 1.0
initial_velocity_std_mps = 100.0
max_velocity_mps = 191.0
unfolding_tolerance_bins = 2.0
confirmation_hits = 3
confirmation_window = 5
deletion_misses = 5
max_coast_time_s = 30.0
history_size = 100
```

Project/tool configuration remains in `pyproject.toml`; scenario data remains in the established
`scenarios/*.toml` convention. Unknown keys and incorrect scalar kinds in these tables are errors.
The dedicated general-tracker runner enables tracking by default; `--track` remains accepted
for compatibility. Existing scenario `[detection]`/`[tracking]` tables belong to the original
tracker and are not interpreted as general-tracker settings.

```bash
uv run python scripts/run_general_tracking.py scenarios/scenario_001_fmcw_low_prf.toml --track --out out/tracker-s1 --no-iq --no-plots --no-movie
uv run python scripts/run_general_tracking.py scenarios/scenario_001_pulsed_medium_prf.toml --track --out out/tracker-s2 --no-iq --no-plots --no-movie
uv run python scripts/run_general_tracking.py scenarios/scenario_001_fmcw_dual_prf.toml --track --out out/tracker-s3 --no-iq --no-plots --no-movie
uv run python scripts/run_tracking_example.py --axes x --out out/enu-x.jsonl
uv run python scripts/run_tracking_example.py --axes xy --out out/enu-xy.jsonl
uv run python scripts/run_tracking_example.py --axes xyz --acceleration --out out/enu-xyz-ca.jsonl
make check
```

### Historical source-window limitations

The following measurements were recorded in the deprecated checkout before this port.
They are retained as historical evidence, not measurements of the current Duke scenarios.
All three source 120-second windows used the configurations above and seed 20260911.
Current Duke validation is recorded in [the port audit](tracker-001-port.md).
The evaluation chooses the first confirmed ID without consulting truth, follows that ID permanently,
and computes RMSE only on frames where it remains confirmed. Continuity includes the initial
confirmation delay. Errors below must be interpreted with their limited frame coverage: they do
not describe accuracy over the complete flight window or imply identity continuity.

| Variant | Births | First confirmation (s) | Confirmed primary frames / 120 | Range RMSE (m) | Velocity RMSE (m/s) | Frames without observations |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 70 | 2.0 | 21 | 26.82 | 10.95 | 0 |
| S2 | 139 | 2.0 | 10 | 20.28 | 6.95 | 0 |
| S3 | 21 | 2.0 | 10 | 14.50 | 6.39 | 17 |

S2 range errors are modulo errors; absolute range remains unresolved. The default detector can
produce multiple high-SNR peaks and extra births. The interpolated real track has abrupt velocity
changes, and this CV model with density 1 m²/s³ and tight quantization covariance fragments tracks.
S3 additionally rejects non-unique cross-leg pairings. These are exposed limitations, not resolved
by truth-based selection, hidden resets, or loosened acceptance tolerances. The specified synthetic
CV acceptance tests pass; real-flight robustness requires a separately validated detector/dynamics
configuration (for example, measurement clustering and maneuver-model selection).

Artifacts are generated under ignored `out/`; the commands above now run the current Duke
scenarios and do not reproduce the historical table. Optional
plots are presentation-only. Core tests do not require the reference frameworks or a display.
