# Scenario 003 — Detection, association and tracking over the Singapore X-band scenario

Status: **specification**, pre-implementation. The second vertical slice, stacked on
`spec/scenario-001-singapore-xband.md`.

## 1. Purpose

Scenario 001 turns a real aircraft trajectory into baseband IQ and a range-Doppler map, and stops
at the RD peak. Reading a peak off a map is not detection, and one peak per frame is not a track.
This document specifies the next slice: **CFAR detection → data association → Kalman tracking**,
run over scenario 001's S1 variant, one frame per second.

Three reasons to build it as the next slice:

1. `core/detection.py` already exists — CA/GO/SO/OS-CFAR, `cfar_valid_mask`, `cluster_detections`
   — and is claimed by no scenario. It has unit tests and no end-to-end exercise. This slice is
   that exercise.
2. `core/tracking.py` is reserved in `spec/structure.md` with nothing specifying it. D2 there
   fixes its shape and its promotion trigger; this document fixes its contents.
3. It produces two more pictures: truth and detections overlaid on the RD map, and a plan-view
   track plot against truth. Both are things an intern can look at and immediately say is wrong.

It deliberately does **not** introduce a new radar, a new waveform, or a new trajectory. Every
number in §2 is inherited. What is new is everything downstream of the RD map.

Two choices shape the rest of the document. The measurement is the natural one — range *and*
radial velocity, with the velocity **unfolded** before the tracker sees it (§5) — because a
filter whose motion model is unfolded and whose measurement is not is simply wrong, and S1's
five-fold ambiguity makes that failure vivid rather than academic. And the state vector is a
**parameter**, not a decision (§6): slant range and range rate by default, or a 2-D or 3-D ENU
Cartesian state, all driven by the same tracker. Carrying both lets the scenario show what a
range-only state cannot represent and what a Cartesian state cannot observe from this radar's
measurements, and it turns the agreement between them into the sharpest test in §12.

## 2. Scenario

The radar, the site, the waveform, the target and the trajectory are **exactly** scenario 001's
S1 (`fmcw-low-prf`) variant. See `spec/scenario-001-singapore-xband.md` §2 and §4; those tables
are not repeated here, and if they disagree with anything below, they win.

The inherited numbers this document depends on:

| | |
| :--- | :--- |
| Carrier / wavelength | 9.8 GHz / 30.591 mm |
| Bandwidth, range resolution | 2.0 MHz, **74.95 m** |
| PRF, CPI | 1.0 kHz, 256 chirps = 256 ms inside each 1 s frame |
| RD map shape | `(256, 1000)` — 256 Doppler bins × 1000 range bins |
| Unambiguous range | 37.47 km — the target at 8.39–17.87 km never folds |
| Unambiguous velocity | ±7.65 m/s, a fold span of **15.2955 m/s** — an 80 m/s target folds ~5× |
| Velocity bin | 0.059748 m/s |
| Frame interval `T` | 1.0 s |
| Default window | 120 s = 120 frames |

Added by this scenario, in new TOML blocks:

| | |
| :--- | :--- |
| CFAR variant | CA, along the **range** axis (`axis = -1`) |
| `n_train`, `n_guard` | 16, 4 per side — 245,760 of 256,000 cells carry a complete window |
| `pfa` | **1e-5** — see §3 |
| Threshold factor α | 11.417 dB, from `cfar_threshold_factor` |
| Measurement | slant range **and unfolded** range rate — see §5 |
| `sigma_range_m` | 21.64 m — see §5.4 |
| `sigma_velocity_mps` | 0.01725 m/s — see §5.4 |
| `state_model` | `range_1d` (default), `enu_2d` or `enu_3d` — see §6 |
| `velocity_unfolding` | `track_aided` (default), `oracle` (tests) or `none` — see §5.3 |
| `sigma_accel_mps2` | 2.0 |
| `gate_probability` | 0.99 → χ² threshold set by the measurement dimension, §7 |
| Initiation / deletion | confirm on **4 of 5**; delete after 3 consecutive misses |

## 3. Why the false-alarm rate is the design parameter

A tracker with no false alarms is a Kalman filter with extra steps. Association only becomes real
work when more measurements arrive than there are targets, so `pfa` is not a detail of the
detector here — it sets the difficulty of the whole scenario, and it is chosen, not inherited.

The arithmetic below is for the **initiation** gate, which is the binding case: a brand-new track
has no velocity estimate, so its gate spans the whole Doppler axis and ±`v_max` in range. A mature
track's gate is far smaller (§7), and that asymmetry is the point.

| | |
| :--- | :--- |
| Cells with a complete reference window | 256 × (1000 − 2·20) = **245,760** |
| Expected false alarms per frame, at `pfa = 1e-5` | 245,760 × 1e-5 = **2.46** |
| Initiation gate, at ±200 m/s over 1 s and all Doppler | 5.34 range bins × 256 = 1,366 cells |
| Expected false alarms inside it | 2.46 × 1366/245,760 = 0.0137 per frame |
| P(a false alarm is confirmed as a track), at 4-of-5 | 1 − F_Binom(2; 4, 0.0137) ≈ **1.2e-5** |
| Expected false confirmed tracks over 120 frames | 120 × 2.46 × 1.2e-5 ≈ **0.003** |

So `pfa = 1e-5` puts roughly two or three spurious measurements in front of the associator every
single frame — enough that gating and assignment are exercised continuously — while making a
spurious *confirmed track* a once-in-four-hundred-runs event. Raising `pfa` to 1e-4 multiplies
the last row by roughly 7,000 and the scenario stops having one answer; that is a good exercise
to set an intern, and a bad default.

> Every number in §3 is derived from the parameters in §2 and was confirmed numerically. The
> implementation must nonetheless re-derive the valid-cell count from `cfar_valid_mask`, the
> expected false-alarm count from `pfa`, and the confirmation probability from the M-of-N
> parameters, and assert each in a test rather than trusting this table. A spec table is
> documentation; a test is a guarantee.

## 4. Known limitations

Stated up front, in the same spirit as scenario 001 §3. These are properties of the scenario, not
defects in the tracker, and §13 turns the first three into the work that follows this slice.

- **The target is far too easy to detect.** The link budget gives 51.7 dB of post-integration SNR
  at 17.87 km and 64.8 dB at 8.39 km, against an 11.4 dB threshold factor: Pd is 1 to many decimal
  places. **Track maintenance under missed detections is therefore not exercised at the default
  settings**, and neither is the coast-and-delete logic of §7, which exists here largely untested
  by the scenario. §13.1 is the fix.
- **CFAR is one-dimensional.** `core/detection.py` slides its window along one axis. Applied along
  range it is correct but wasteful here, and it is the wrong detector for a scenario with more than
  one target or with Doppler-spread clutter. §13.2.
- **One target.** Association is exercised against clutter, not against a second aircraft. Track
  crossings, track swaps and the cases that motivate JPDA and MHT are out of scope. §13.3.
- **Velocity unfolding is assumed, not free.** §5 feeds the tracker an *unfolded* range rate.
  Obtaining one at S1's PRF takes either the track's own prediction or several frames of range
  history; §5.3 specifies the mechanism and its bootstrap, and is the part of this document most
  likely to surprise a reader who expects the measurement to arrive ready-made.
- **Angles are not measured.** There is no array, so this radar measures range and range rate and
  nothing else. The ENU state models of §6 need azimuth (and elevation) and therefore run on a
  **simulated** angle measurement until `array/` exists; §6.3 says exactly what that means and why
  the default state model is the one that needs no such crutch.
- **Truth is itself smoothed.** Scenario 001 §3 notes that radial velocity is a central difference
  over an irregularly sampled track. Track velocity error is therefore measured against a
  reference that is already low-pass filtered over several seconds. Range truth does not have this
  problem, which is why §12's tight acceptance criterion is stated on range and the velocity one
  is deliberately loose.

## 5. The measurement

### 5.1 From detection clusters to metres

`cfar_detect` returns a boolean map; `cluster_detections` reduces connected runs of it to
`Detection` records carrying `peak_index`, a power-weighted `centroid_index` in **fractional cell
indices**, `peak_power_w`, `total_power_w` and `n_cells`. Converting those indices to physical
units is the whole of the pipeline's detection step, and it must use the existing helpers:

- range, from `dsp.range_bin_centers_m(n_bins, bandwidth_hz, chirp_time_s, sample_rate_hz)`,
- velocity, from `dsp.doppler_bin_centers_mps(n_bins, pulse_repetition_interval_s, wavelength_m)`.

Interpolating a fractional index into those axes — rather than deriving bin spacing locally —
keeps the unshifted-range / fftshifted-Doppler asymmetry of scenario 001 §7.1 in one place. A
tracker that silently reinvents the Doppler axis will produce closing targets that open, and the
picture will still look plausible.

### 5.2 The measurement vector

The measurement is the natural one:

```text
z = [range_m, range_rate_mps]ᵀ          closing velocity positive, per structure.md D5
```

with the standing assumption that **`range_rate_mps` has been unfolded before it reaches the
tracker** — it is a true radial velocity in ±191 m/s, not the ±7.65 m/s the RD map shows. The
filter's motion model is unfolded, so its measurement model must be too; feeding it the folded
quantity instead is not a tuning problem but a wrong measurement model, and it produces a track
that is confidently wrong.

For the ENU state models of §6 the vector gains the angles those states need:

| `state_model` | `z` | dim |
| :--- | :--- | :--- |
| `range_1d` | `[range_m, range_rate_mps]` | 2 |
| `enu_2d` | `[range_m, azimuth_rad, range_rate_mps]` | 3 |
| `enu_3d` | `[range_m, azimuth_rad, elevation_rad, range_rate_mps]` | 4 |

### 5.3 Where the unfolded velocity comes from

`velocity_unfolding` selects the mechanism. This is the one place where the scenario cannot simply
assume its way past S1's ambiguity, so all three modes are specified and the default is the honest
one.

| Mode | Mechanism |
| :--- | :--- |
| `track_aided` *(default)* | The track's **predicted** range rate is unfolded, so it selects the fold: `k = round((ṙ_pred − ṙ_meas) / v_span)`, `ṙ = ṙ_meas + k·v_span`, `v_span = 15.2955 m/s`. |
| `oracle` | The fold index is taken from `truth.csv`. **Tests and teaching only** — it isolates tracker error from unfolding error, and it must never be a scenario default. |
| `none` | The folded value is passed through unchanged. Present so that the failure in §5.2 can be *demonstrated* rather than only described. |

`track_aided` is circular at initiation: a track with no velocity estimate cannot select a fold.
The bootstrap is to **fall back to the range-only measurement** — `z = [range_m]`, dim 1 — until
the filter's own covariance says the fold is resolvable, and the covariance is exactly the right
thing to ask. Unfolding is safe once the predicted range rate is certain to well inside half a
fold span:

```text
sqrt(P[range_rate, range_rate]) < v_span / unfold_sigma_gate        unfold_sigma_gate = 6
```

At `sigma_range_m = 21.64 m` and `T = 1 s`, a least-squares range slope over `N` frames has
standard deviation `sigma_range_m · sqrt(12 / (N(N²−1))) / T`: 6.8 m/s at `N = 5`, 3.3 m/s at
`N = 8`. The threshold `v_span/6 = 2.55 m/s` is therefore crossed at around ten frames. **A track
spends its first ~10 s range-only and then switches to the two-dimensional measurement**, and the
implementation must report the frame at which each track switched, because that number is the most
informative single diagnostic this scenario produces.

An unfolding error is not modelled by `R` and must not be: one wrong fold displaces the
measurement by 15.2955 m/s against a `sigma_velocity_mps` of 0.017, which is `d² ≈ 8e5` against a
gate of 9.21. A mis-unfolded measurement is therefore rejected by the gate and counts as a
**miss**, never as a corrupted update. That is the desired behaviour and §12 asserts it.

Scenario 001's S3 variant resolves the ambiguity in the waveform, via the existing
`core/ambiguity.py`. Tracking over S3 needs none of this subsection, which is the argument for
doing it next (§13.4).

### 5.4 Measurement noise

Each component takes the larger of its CRLB and its bin-quantisation floor:

| Component | CRLB at 51.7 dB | Quantisation, `bin/sqrt(12)` | Used |
| :--- | :--- | :--- | :--- |
| Range | 0.14 m | 74.9481 / √12 = **21.64 m** | `sigma_range_m = 21.64` |
| Range rate | negligible | 0.059748 / √12 = **0.01725 m/s** | `sigma_velocity_mps = 0.01725` |

`R = diag(sigma_range_m², sigma_velocity_mps²)`, both quantisation-limited. The ENU models add
`sigma_azimuth_deg` and `sigma_elevation_deg`, which are properties of the simulated angle
measurement rather than of this radar; see §6.3.

The five-order-of-magnitude spread between the two variances is not a mistake. It is what a
range-Doppler radar looks like: a coarse ranging measurement and an exquisite velocity one. It is
also why the ENU Jacobian must be right — a 1 % error in the range-rate row of `H` is worth more
than a 100 % error in the range row.

## 6. State, model and filter

`state_model` selects the state vector. All three share the same tracker, the same associator and
the same track management; they differ in the state, the transition, the measurement model and
whether the update is linear.

| `state_model` | State `x` | Filter | Measurement |
| :--- | :--- | :--- | :--- |
| `range_1d` *(default)* | `[r, ṙ]` | linear KF | linear, `H = I₂` on `[r, ṙ]` |
| `enu_2d` | `[e, n, ė, ṅ]` | EKF | nonlinear, `h(x) = [‖p‖, atan2(e, n), p·v/‖p‖]` |
| `enu_3d` | `[e, n, u, ė, ṅ, u̇]` | EKF | nonlinear, as above plus elevation |

Positions are metres in the local ENU frame of `core/geodesy.py`, azimuth 0° at true north
increasing clockwise, per `spec/structure.md` D5 and scenario 001 §7 step 1.

### 6.1 Why the state is a parameter and not a choice

`range_1d` is observable from this radar's own measurements and needs no filter beyond the
textbook two-state case, so it is the default and the one §12's tight criteria are stated on. The
ENU models are what a real tracker uses — they carry the target's actual motion, they extend to
multiple sensors, and they are the state `pipelines/exporters/` will eventually want — but they
are **not observable from range and range rate alone**: with no angle, the target lies somewhere
on a circle of unknown bearing, and an EKF handed that will produce a covariance that shrinks
while the estimate is wrong.

Making the state a parameter rather than picking one lets the scenario show that directly: run
`range_1d` and `enu_2d` on the same frames and compare. §12's cross-check criterion is exactly
this comparison, and it is the strongest test in the document, because an ENU model that agrees
with the 1-D model on range and range rate almost certainly has its Jacobian right.

### 6.2 Transition and process noise

Constant velocity throughout, with discrete white-noise acceleration:

| | |
| :--- | :--- |
| Transition, per axis | `[[1, T], [0, 1]]`, `T = 1.0 s` |
| Process noise, per axis | `σ_a² · [[T⁴/4, T³/2], [T³/2, T²]]` |
| Multi-axis assembly | the same 2×2 blocks, one per spatial axis, with no cross-axis terms |
| `sigma_accel_mps2` | 2.0 |
| Initial covariance `P₀` | measured components from `R`; unmeasured velocity components `v_max² = (200 m/s)²` |

`sigma_accel_mps2 = 2.0` is chosen for a light aircraft flying a circuit: the radial acceleration
is the projection of a roughly 1–2 m/s² lateral acceleration onto the line of sight, and 2.0 gives
the filter enough process noise not to lag through the turns. It is the one number in this
specification arrived at by judgement rather than derivation, and the docstring must say so.

DWNA is chosen over continuous white noise because it is the form every reference below states,
and because a single `sigma_accel_mps2` in m·s⁻² is a knob an intern can reason about.
`core/tracking.py` builds `Q` from that one parameter and does not expose a raw matrix.

The state models must be **data, not a class hierarchy**: a factory returning `(F, Q, h, H_jac, R,
dim)` for a given `state_model` and `T`. `spec/structure.md` D2 keeps `core/tracking.py` a single
module, and three state models implemented as three subclasses with three update paths would
exhaust that budget on structure instead of on tracking.

### 6.3 The simulated angle measurement

`enu_2d` and `enu_3d` need angles this radar does not measure. Until `array/` lands, the pipeline
**synthesises** them: truth azimuth (and elevation) from `pipelines/trajectories.py`, plus
zero-mean Gaussian noise of `sigma_azimuth_deg` / `sigma_elevation_deg`, defaulting to **0.5°**
— a plausible monopulse accuracy for the 5.1° beam implied by 30 dBi, and far worse than the
0.008° the SNR alone would suggest, because the real error budget is boresight calibration, not
thermal noise.

This is a crutch, and it must be labelled as one everywhere it appears:

- the parameter lives in `[tracking]` as `simulated_angles = true`, never defaulted silently on;
- `metadata.json` records it, so no stored run is ambiguous about whether its angles were real;
- `run_scenario.py` prints a one-line warning when it is active;
- `core/tracking.py` never knows about it — it is a `pipelines/` concern, because the tracker's
  contract is "you give me measurements and a model", and where the measurements came from is not
  its business.

`enu_3d` carries an additional degeneracy: scenario 001 holds target altitude constant at 1500 m,
so the up channel has no true dynamics and its estimate is driven entirely by process noise. It is
specified for completeness and for the day a trajectory with altitude arrives; it is not part of
§12's acceptance criteria.

## 7. Association and track management

**Gating.** For each existing track, predict, form the innovation `ν = z − h(x̂)` and its covariance
`S = H P Hᵀ + R`, and compute the normalised innovation squared `d² = νᵀ S⁻¹ ν`. A measurement is
in the gate when `d² ≤ χ²(gate_probability, dim)`, where `dim` is the **measurement** dimension of
the moment — which varies within a single run, because §5.3's bootstrap starts a track at dim 1
and promotes it to dim 2:

| dim | Measurement | χ² at 0.99 |
| :--- | :--- | :--- |
| 1 | range only, during the bootstrap of §5.3 | 6.635 |
| 2 | `range_1d` after unfolding | 9.210 |
| 3 | `enu_2d` | 11.345 |
| 4 | `enu_3d` | 13.277 |

The gate is a statistical statement about the filter's own uncertainty, not a fixed number of
metres, and the implementation must not hard-code its width — nor its dimension, which is the
easier mistake to make here.

Adding the velocity component shrinks a mature track's gate from about 5.3 RD cells to about 3,
and almost all of that comes from the velocity dimension: `sigma_velocity_mps` is 0.017 m/s
against a 0.06 m/s bin. Once a track is unfolded and settled, a false alarm essentially never
enters its gate, and §3's initiation arithmetic is the binding case precisely because of this.

**Assignment.** Global nearest neighbour: build the cost matrix of `d²` over gated (track,
measurement) pairs, with non-gated pairs forbidden, and solve it with
`scipy.optimize.linear_sum_assignment`. `scipy` is already a dependency (`core/detection.py` uses
`scipy.ndimage`), so this adds none.

GNN rather than JPDA is a deliberate limit, and the reason is written into `spec/structure.md` D2:
*a second association strategy is exactly the trigger that promotes `core/tracking.py` to
`core/tracking/{filters,association,fusion}.py`*. This scenario ships one strategy and does not
fire the trigger. Whoever adds JPDA does the promotion in the same change.

**Track management.** Tentative tracks are initiated from every unassociated measurement. Because
the measurement now carries velocity, initiation is **single-point** once a track is unfolded —
range from the range component, range rate from the velocity component — but a *new* track is by
§5.3 still in its range-only bootstrap, so it is seeded with `P₀`'s `v_max²` velocity variance and
converges over the first frames. A tentative track is **confirmed** on 4 hits in its first 5
frames and **deleted** otherwise; a confirmed track coasts on prediction alone through a miss and
is deleted after 3 consecutive misses. §3 sizes 4-of-5 against the false-alarm rate; the deletion
count is sized against the coast time an intern can see on the plot, not derived. Note that at the
default link budget a miss is nearly always a *mis-unfold*, not a missed detection (§4), which is
another reason §13.1 matters.

## 8. Frame and CPI structure

Unchanged from scenario 001 §5: one frame per second, a 256 ms CPI inside it, range migration
negligible at a quarter of a range bin. The tracker's `T` is the **frame** interval, 1.0 s, not
the CPI length — measurements are timestamped at frame time, and the 744 ms of idle time between
CPIs is absorbed into the process noise. This is the standard approximation and must be named in
`core/tracking.py`'s `Notes`.

## 9. Outputs

Extending scenario 001 §6. Written under `--out <dir>` by `scripts/run_scenario.py`:

| File | Content |
| :--- | :--- |
| `rd_{frame:05d}.png` | As before, **plus** CFAR detection markers and the confirmed track's gate |
| `track_{frame:05d}.png` | Plan view: truth trajectory, detections so far, track history, current gate |
| `track.mp4` | The `track_*.png` frames at 1 fps (GIF fallback), as `rd.mp4` already is |
| `detections.csv` | `frame, time_s, range_m, velocity_folded_mps, velocity_unfolded_mps, fold_index, peak_power_w, n_cells, associated_track_id` |
| `tracks.csv` | `frame, time_s, track_id, status, measurement_dim, range_m, range_rate_mps, var_range_m2, var_range_rate_m2ps2, nis, associated` |

`status` is one of `tentative`, `confirmed`, `coasting`, `deleted`. `associated_track_id` is empty
for a measurement that went unassociated, which is how the plot distinguishes a false alarm from a
hit without rerunning the associator. `measurement_dim` records §5.3's bootstrap: the frame at
which it steps from 1 to 2 is the frame the track became unfoldable, and `velocity_unfolded_mps`
is empty until then.

For an ENU `state_model`, `tracks.csv` gains the state columns (`east_m`, `north_m`, … and their
variances) **in addition to** `range_m` and `range_rate_mps`, which are computed from the ENU
state. Keeping range and range rate in every run is what makes §12's cross-check possible without
a second parser.

`clear_previous_frames` in `scripts/run_scenario.py` globs per-frame artefact names to remove
stale output from a longer previous run. **It must learn `track_[0-9]{5}.png`**, or a short re-run
silently leaves a movie assembled from two different runs.

IQ and PNG output remain gitignored; per `CLAUDE.md` the generating script is committed and the
arrays are not.

## 10. Module build order

Each module lands with its tests in the same commit, and `make check` passes before the next
begins.

| Step | Module | Notes |
| :--- | :--- | :--- |
| 0 | `core/tracking.py` *(new)* | `KalmanState`, `predict`, `update`, `process_noise_dwna`, `normalised_innovation_squared`, `gate_threshold`, `associate_gnn`, `Track`, `TrackManager`, and `state_model_matrices(state_model, frame_time_s, ...)` returning `(F, Q, h, H_jac, R, dim)`. One module, per D2; state models are data, per §6.2. Constants from `core.constants` only. |
| 0a | — | Build `range_1d` end to end first, with the linear update. The EKF path lands only once §12's criteria 1–6 pass on `range_1d`, so that a Jacobian bug cannot hide behind a tracker bug. |
| 1 | `pipelines/tracking.py` *(new)* | `frame_detections(product, ...) -> list[Measurement]` via §5.1's bin-centre helpers; §5.3's unfolding and its bootstrap; the simulated angle measurement of §6.3; then drives `TrackManager` frame by frame. Kept out of `pipelines/scenarios.py`, which is near the 400-line guidance of `docs/conventions/style.md` §9. |
| 2 | `scenarios/scenario_003_tracking.toml` *(new)* | S1's blocks verbatim, `[scenario].name = "scenario-003-tracking"`, plus new `[detection]` and `[tracking]` blocks. `_require_keys` in `pipelines/scenarios.py` is extended **additively**, both blocks optional, so the three scenario-001 TOMLs keep loading byte-identically. |
| 3 | `teaching/scopes/rd_map.py` *(edit)* | Additive keyword-only `detections=`, `track_estimate=`, `gate_extent=`, all defaulting to `None`. Follow the existing private `_mark_truth` helper; do not restructure it. |
| 4 | `teaching/scopes/track_plot.py` *(new)* | `render_track_plan_view(...)`. `matplotlib` lazily imported through the existing `teaching.plotting.require_pyplot`. |
| 5 | `scripts/run_scenario.py` *(edit)* | Track when the TOML carries `[tracking]`; write the §9 files; extend `clear_previous_frames`; add `--no-tracking`; warn when §6.3's simulated angles are active. |
| 6 | Docs | `README.md`, `spec/starter.md`, `spec/structure.md`, `docs/README.md` — §11. |

### 10.1 What this scenario consumes but never edits

`core/detection.py`, `core/dsp.py`, `core/windows.py`, `core/signal.py`, `core/geodesy.py`,
`core/ambiguity.py`, `pipelines/trajectories.py`, and `data/flight_coordinates.csv`. If the
tracker appears to need a change in any of them, that is a finding to raise, not a change to make
inside this workstream. §13.1 and §13.2 in particular are changes to `scenarios/*.toml` and to
`core/detection.py` respectively, and belong to their own slices.

### 10.2 Workstream boundary

This scenario claims, and another workstream should not plan:

```text
spec/scenario-003-singapore-tracking.md        scenarios/scenario_003_tracking.toml
src/radar_forge/core/tracking.py               src/radar_forge/pipelines/tracking.py
src/radar_forge/teaching/scopes/track_plot.py
tests/core/test_tracking.py                    tests/pipelines/test_tracking.py
tests/pipelines/test_scenario_003.py           tests/teaching/test_track_plot.py
```

`scripts/run_scenario.py`, `src/radar_forge/pipelines/scenarios.py` and
`src/radar_forge/teaching/scopes/rd_map.py` are **shared with both scenario 001 and scenario 002**
— `spec/scenario-002-singapore-bistatic.md` §10 claims the same three files. This slice makes only
the additive edits in steps 2, 3 and 5, and every existing call site and TOML must keep working
unchanged. Coordinate before touching them: scenario 002 is in flight at the time of writing, and
its `Radar | BistaticRadar` union already reaches `scripts/run_scenario.py`.

That union is the one place where scenario 002 could reach this one. It does not: this scenario
consumes `RangeDopplerProduct` and the axis helpers, not the radar object, so a tracker written
against §5.1 works unchanged for a bistatic geometry — with the caveat that for a bistatic radar
"range" is bistatic range and "range rate" its derivative, so the §6 state models measure the
**sum** of the two path lengths and an ENU state would need the transmitter position as well.
Tracking a bistatic scenario is therefore a real extension, not a free one, and it is not claimed
here.

## 11. Amendments to `spec/structure.md`

1. **`core/tracking.py`'s `Draws from` row in B.1 is extended.** It currently reads
   `RadarSim tracking and fusion; RadarBook tracking-filter chapters`. Add Stone Soup (data model
   and vocabulary), FilterPy (filter and process-noise formulation) and motpy (association loop
   shape).
2. **Five projects are added to Part A** as `### A.13 Tracking and data fusion`, sub-blocked
   A.13a–A.13e after the A.12 precedent, each in the existing fixed shape (`**Link:**`,
   `**Language / licence:**`, a `| Module | Responsibility |` table, a closing
   `**What radar-forge borrows:**` paragraph): Stone Soup (MIT), FilterPy (MIT), motpy (MIT),
   Tracktable (BSD-3-Clause) and labeledRFS/VisualRFS (MIT ports of Vo's MATLAB). The licence
   summary shifts to `### A.14` and gains a row each. The same five are added to `README.md`'s
   `### Reference frameworks` table and `spec/starter.md` §2.1.
3. **D2's promotion trigger is not fired.** Three state models are not a second association
   strategy and not a fusion layer; per §6.2 they are data, not structure. One module. Restating
   it here so that the next person to open `core/tracking.py` knows the rule before they add JPDA
   to it.
4. **No new runtime dependency.** `scipy.optimize.linear_sum_assignment` comes from a dependency
   already in core. None of the five reference projects is vendored or imported; per `CLAUDE.md`,
   a dependency needs a justification in the PR, and "we already have `scipy`" is the
   justification for not adding one.

## 12. Acceptance criteria

Stated on `state_model = "range_1d"`, `velocity_unfolding = "track_aided"`, over the default 120 s
window of `scenarios/scenario_003_tracking.toml`, except where noted:

1. **One track.** Exactly one confirmed track exists in at least 95 % of frames, and its
   `track_id` never changes across the run.
2. **Range accuracy.** The confirmed track's range RMSE against `truth.csv` is below one range
   bin (74.95 m). The expected value is well under `sigma_range_m`, because the filter smooths.
3. **Velocity accuracy.** Range-rate RMSE against `truth.csv` is below 2 m/s once the track has
   left its §5.3 bootstrap. The bound is deliberately loose: per §4, truth velocity is a central
   difference over an irregularly sampled track, so this criterion checks plumbing and sign, not
   filter performance. It is **not** tightened without first fixing the truth.
4. **Unfolding is correct.** In every confirmed, unfolded frame, the fold index chosen by §5.3
   equals the true fold index computed from `truth.csv`. A mis-unfold must show up as a gate
   rejection and a miss — never as an accepted update — and the test asserts both halves.
5. **Consistency.** The normalised innovation squared, averaged over confirmed frames, lies inside
   the 95 % confidence interval of a χ²(`dim`) mean, with `dim` taken per frame from
   `measurement_dim` — an interval, not a tolerance, per `docs/conventions/testing.md` §2. This is
   the check that catches a mis-scaled `R` or `Q`, and with `sigma_velocity_mps` five orders of
   magnitude below `sigma_range_m` it is also the check that catches a transposed `R`.
6. **False alarms behave.** The measured false-alarm rate over `cfar_valid_mask` cells agrees with
   `pfa` inside its Poisson confidence interval, and **no false alarm is promoted to a confirmed
   track**. The test asserts §3's analytic bound of 0.003 expected false tracks, not merely the
   observed zero — observing zero is consistent with an associator that never associates anything.
7. **State models agree.** Running the same frames under `state_model = "enu_2d"` with
   `simulated_angles` on must reproduce the `range_1d` run's range and range rate to within
   `0.1 · sigma_range_m` and `0.1 · sigma_velocity_mps` respectively over confirmed frames. This
   is the strongest criterion in the document: an EKF whose Jacobian is wrong will not pass it,
   and criteria 1–6 individually will not catch that.
8. **The pictures exist.** `run_scenario.py` writes `track_00000.png`, an `rd_00000.png` carrying
   detection markers, `tracks.csv` and `detections.csv`.

Encoded as `tests/pipelines/test_scenario_003.py`, marked `slow`, over a 15-frame window so that
`make check` stays fast — 15 rather than scenario 001's 5, because 4-of-5 initiation needs five
frames before there is anything to assert and §5.3's bootstrap needs about ten more before
criteria 3, 4 and 7 have any unfolded frames to measure.

Unit-level ground truth, per `docs/conventions/testing.md` §3, comes first and is analytic: a
noiseless constant-velocity target must be tracked with zero steady-state innovation to float
precision, in every `state_model`; `associate_gnn` must return the known optimal assignment of a
hand-built cost matrix with a unique solution; `gate_threshold` must match the tabulated χ²
quantile at each `dim` in §7's table; `process_noise_dwna` must reproduce the worked `Q` in
reference [2]; the EKF Jacobians must match a complex-step or central-difference derivative of
`h` to `1e-8`; the §5.3 fold selector must recover the correct index for every fold in
±191 m/s; and the empirical gate acceptance rate over a seeded draw must match
`gate_probability` inside its binomial confidence interval.

## 13. What comes next, and why not now

The first three items are the limitations of §4, in the order that makes each one testable. None
is in scope here; each is a slice of its own, and this section exists so that whoever picks one up
inherits the reasoning rather than rediscovering it.

### 13.1 Lower the link budget until Pd < 1

At the default 100 W the target is detected in every frame, so §7's coast-and-delete logic and
criterion 1's 95 % threshold are never tested against a real miss. Dropping `transmit_power_w`
until post-integration SNR at 17.87 km sits a few dB above the 11.4 dB threshold factor puts Pd
somewhere interesting and makes the whole of track *maintenance* — as opposed to track accuracy —
part of the scenario. This needs only a TOML change and a re-derivation of the expected Pd from
`radar_equation.received_power_w` and the CFAR threshold, so it should be the next thing done.

The reason it is not done here: mixing it in would mean this slice's acceptance criteria could
fail for two unrelated reasons at once, and §12 criterion 1 exists to pin down the easy case
first.

### 13.2 Two-dimensional CFAR

`core/detection.py` slides a 1-D window. Applied along range, in one-target thermal noise, it is
correct — but it is the wrong detector the moment there is Doppler-spread clutter or a second
target, and a 2-D reference window would also roughly halve the number of training cells needed
for the same `pfa`. This is a change to `core/detection.py` and belongs to that module's
workstream, not to this scenario (§10.1); it lands as a new `axis`-pair argument or a sibling
function, not as a rewrite, so that this scenario's `pfa` calibration stays valid.

### 13.3 Multiple targets — a note only

Everything in §7 that makes GNN sufficient depends on there being one target: gates never overlap,
assignment is trivially optimal, and a track swap is impossible. A second aircraft changes all
three, and the honest consequence is that JPDA or MHT becomes necessary — which per
`spec/structure.md` D2 and §11.3 above fires the promotion of `core/tracking.py` to
`core/tracking/{filters,association,fusion}.py`. It also needs a synthetic second trajectory,
since `data/flight_coordinates.csv` has one aircraft, and it needs OSPA or GOSPA rather than RMSE
as its accuracy measure, since with multiple targets "the error" is no longer a single number.
**This is recorded as a direction, not as a plan.** It should not be started until §13.1 and
§13.2 are done, because a multi-target scenario built on a detector that cannot resolve two
targets in Doppler and a link budget that never misses would be measuring the wrong things.

### 13.4 The smaller ones

| Extension | Trigger |
| :--- | :--- |
| Tracking over S3, where the waveform resolves the ambiguity and §5.3 disappears entirely | Cheap and clarifying; do it alongside §13.1 |
| A real angle measurement, replacing §6.3's simulated one | Needs `array/`; until then `enu_2d` is a Jacobian test, not a radar |
| A trajectory with real altitude, making `enu_3d` non-degenerate | Needs input data scenario 001 §3 does not have |
| IMM, for the turns the circuit contains | Only once `sigma_accel_mps2` is demonstrably the binding error |
| GLMB/LMB random-finite-set tracking | Only with a multi-target, high-clutter scenario to justify it |
| Track fusion across sensors | Fires D2's promotion |

## References

.. [1] Y. Bar-Shalom, P. K. Willett and X. Tian, *Tracking and Data Fusion: A Handbook of
       Algorithms*, YBS Publishing, 2011, ch. 2 (gating), ch. 3 (assignment).
.. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and
       Navigation*, Wiley, 2001, §5.2 (discrete white-noise acceleration), §6.3 (validation
       gating and nearest neighbour), §10.3 (the extended Kalman filter and range/range-rate
       measurement Jacobians), §11.7 (track initiation).
.. [3] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking Systems*, Artech
       House, 1999, ch. 6 (M-of-N initiation, global nearest neighbour), §4.3 (Doppler-aided
       tracking and ambiguity resolution).
.. [4] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014,
       §6.5 (CFAR), §7.3 (measurement accuracy and the range and Doppler CRLBs).
.. [5] D. F. Crouse, "On implementing 2D rectangular assignment algorithms," *IEEE Trans. Aerosp.
       Electron. Syst.*, vol. 52, no. 4, pp. 1679-1696, 2016. The algorithm behind
       `scipy.optimize.linear_sum_assignment`.
.. [6] P. A. Thomas, J. Barr, B. Balaji and K. White, "An open source framework for tracking and
       state estimation ('Stone Soup')," *Proc. SPIE 10200, Signal Processing, Sensor/Information
       Fusion, and Target Recognition XXVI*, 2017.
.. [7] R. R. Labbe, *Kalman and Bayesian Filters in Python*, 2020. The FilterPy companion text.
.. [8] B.-T. Vo and B.-N. Vo, "Labeled random finite sets and multi-object conjugate priors,"
       *IEEE Trans. Signal Process.*, vol. 61, no. 13, pp. 3460-3475, 2013. Cited for the
       future work in §13.3, not implemented here.
