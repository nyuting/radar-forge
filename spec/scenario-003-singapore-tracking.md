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
| Unambiguous velocity | ±7.65 m/s — an 80 m/s target folds ~5× |
| Frame interval `T` | 1.0 s |
| Default window | 120 s = 120 frames |

Added by this scenario, in new TOML blocks:

| | |
| :--- | :--- |
| CFAR variant | CA, along the **range** axis (`axis = -1`) |
| `n_train`, `n_guard` | 16, 4 per side — 245,760 of 256,000 cells carry a complete window |
| `pfa` | **1e-5** — see §3 |
| Threshold factor α | 11.417 dB, from `cfar_threshold_factor` |
| Measurement | slant range only — see §5 |
| `sigma_range_m` | 21.64 m — see §5.2 |
| Filter | 2-state constant-velocity Kalman filter, DWNA process noise |
| `sigma_accel_mps2` | 2.0 |
| `gate_probability` | 0.99 → χ² threshold 6.635 at 1 degree of freedom |
| Initiation / deletion | confirm on **4 of 5**; delete after 3 consecutive misses |

## 3. Why the false-alarm rate is the design parameter

A tracker with no false alarms is a Kalman filter with extra steps. Association only becomes real
work when more measurements arrive than there are targets, so `pfa` is not a detail of the
detector here — it sets the difficulty of the whole scenario, and it is chosen, not inherited.

The arithmetic, for the RD map and window above:

| | |
| :--- | :--- |
| Cells with a complete reference window | 256 × (1000 − 2·20) = **245,760** |
| Expected false alarms per frame, at `pfa = 1e-5` | 245,760 × 1e-5 = **2.46** |
| Gate width for a track with no velocity estimate, at ±200 m/s over 1 s | 400 m = 5.34 range bins |
| Expected false alarms inside that gate | 2.46 × 5.34/960 = 0.0137 per frame |
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
defects in the tracker, and each one is a knob an intern can turn.

- **The target is far too easy to detect.** The link budget gives 51.7 dB of post-integration SNR
  at 17.87 km and 64.8 dB at 8.39 km, against an 11.4 dB threshold factor: Pd is 1 to many decimal
  places. **Track maintenance under missed detections is therefore not exercised at the default
  settings.** Lowering `transmit_power_w` until Pd falls below 1 is the intended way to exercise
  it, and the spec keeps the coast-and-delete logic of §7 precisely so that this works.
- **One target.** Association is exercised against clutter, not against a second aircraft. Track
  crossings, track swaps and the cases that motivate JPDA and MHT are out of scope; §11 records
  what would have to change.
- **Azimuth is not measured.** There is no array yet, so the scenario carries a single beam and
  `truth.csv`'s azimuth is a label, never an input. The state is one-dimensional slant range. A
  genuine 3-D ENU state needs `array/` and an EKF; §11.
- **Truth is itself smoothed.** Scenario 001 §3 notes that radial velocity is a central difference
  over an irregularly sampled track. Track velocity error is therefore measured against a
  reference that is already low-pass filtered over several seconds, and is not a clean estimate of
  filter performance. Range truth does not have this problem, which is why §10's acceptance
  criteria are stated on range.
- **CFAR is one-dimensional.** `core/detection.py` slides its window along one axis. Applied along
  range, a strong target leaks into the training cells of nearby *Doppler* bins not at all, and of
  nearby *range* bins only within the guard band — acceptable here because there is one target.
  Two-dimensional CFAR is a `core/detection.py` change and belongs to that module's workstream,
  not to this scenario.

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

### 5.2 The measurement is range only, and why

The natural measurement vector is `(range_m, radial_velocity_mps)`. **It is not used here.** At
S1's ±7.65 m/s unambiguous velocity an 80 m/s target folds about five times, so the measured
velocity is the true velocity modulo 15.29 m/s. Feeding a folded quantity to a filter whose model
is unfolded is not a tuning problem; it is a wrong measurement model, and it will produce a track
that is confidently wrong — the worst kind of output for a teaching repository.

The measurement is therefore **scalar slant range**, and radial velocity is a *hidden state*
estimated by the constant-velocity model from the range sequence. This is honest, it is the
textbook two-state case, and it makes the Doppler folding visible as a limitation rather than
hiding it inside a filter.

Measurement noise is the larger of two floors:

| | |
| :--- | :--- |
| CRLB, `c/(2B) / sqrt(2·SNR)` at 51.7 dB | 0.14 m — negligible |
| Centroid quantisation, one bin uniform: `74.95 / sqrt(12)` | **21.64 m** |

so `sigma_range_m = 21.64`, quantisation-limited, and `R = sigma_range_m²`. The spec states the
derivation rather than a tuned number so that a reader changing the bandwidth knows what to
recompute.

### 5.3 The extension

Unfolding the measured velocity against the track's *predicted* velocity — the prediction is
unfolded, so it selects the correct fold — turns this into a two-state measurement and roughly
halves the settling time. It is a natural exercise, it needs `core/ambiguity.py` which already
exists, and it is **not** part of this scenario's acceptance criteria. Scenario 001's S3 variant
resolves the ambiguity in the waveform instead; tracking over S3 is the other extension (§11).

## 6. State, model and filter

State `x = [range_m, range_rate_mps]ᵀ`, with closing velocity positive, matching
`spec/structure.md` D5 and scenario 001 §7.1.

| | |
| :--- | :--- |
| Transition `F` | `[[1, T], [0, 1]]`, `T = 1.0 s` |
| Measurement `H` | `[1, 0]` |
| Measurement noise `R` | `sigma_range_m²` = 468.3 m² |
| Process noise `Q` | DWNA: `G Gᵀ σ_a²`, `G = [T²/2, T]ᵀ`, `σ_a = sigma_accel_mps2` |
| Initial covariance `P₀` | `diag(sigma_range_m², v_max²)` with `v_max = 200 m/s` |

`sigma_accel_mps2 = 2.0` is chosen for a light aircraft flying a circuit: the radial acceleration
is the projection of a roughly 1–2 m/s² lateral acceleration onto the line of sight, and 2.0 gives
the filter enough process noise not to lag through the turns. It is the one number in this
specification arrived at by judgement rather than derivation, and the docstring must say so.

The discrete-white-noise-acceleration form is chosen over continuous white noise because it is the
form every reference below states, and because a single `sigma_accel_mps2` in m·s⁻² is a knob an
intern can reason about. `core/tracking.py` implements `Q` from that one parameter; it does not
expose a raw matrix.

## 7. Association and track management

**Gating.** For each existing track, predict, form the innovation `ν = z − H x̂` and its covariance
`S = H P Hᵀ + R`, and compute the normalised innovation squared `d² = νᵀ S⁻¹ ν`. A measurement is
in the gate when `d² ≤ χ²(gate_probability, dof)`, `dof = 1` here, giving 6.635 at 0.99. The gate
is a statistical statement about the filter's own uncertainty, not a fixed number of metres, and
the implementation must not hard-code its width.

**Assignment.** Global nearest neighbour: build the cost matrix of `d²` over gated (track,
measurement) pairs, with non-gated pairs forbidden, and solve it with
`scipy.optimize.linear_sum_assignment`. `scipy` is already a dependency (`core/detection.py` uses
`scipy.ndimage`), so this adds none.

GNN rather than JPDA is a deliberate limit, and the reason is written into `spec/structure.md` D2:
*a second association strategy is exactly the trigger that promotes `core/tracking.py` to
`core/tracking/{filters,association,fusion}.py`*. This scenario ships one strategy and does not
fire the trigger. Whoever adds JPDA does the promotion in the same change.

**Track management.** Tentative tracks are initiated from every unassociated measurement, with
velocity unknown and `P₀` as above. A tentative track is **confirmed** on 4 hits in its first 5
frames and **deleted** otherwise; a confirmed track coasts on prediction alone through a miss and
is deleted after 3 consecutive misses. §3 sizes 4-of-5 against the false-alarm rate; the deletion
count is sized against the coast time an intern can see on the plot, not derived.

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
| `detections.csv` | `frame, time_s, range_m, velocity_mps, peak_power_w, n_cells, associated_track_id` |
| `tracks.csv` | `frame, time_s, track_id, status, range_m, range_rate_mps, var_range_m2, var_range_rate_m2ps2, nis, associated` |

`status` is one of `tentative`, `confirmed`, `coasting`, `deleted`. `associated_track_id` is empty
for a measurement that went unassociated, which is how the plot distinguishes a false alarm from a
hit without rerunning the associator.

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
| 0 | `core/tracking.py` *(new)* | `KalmanState`, `predict`, `update`, `process_noise_dwna`, `normalised_innovation_squared`, `gate_threshold`, `associate_gnn`, `Track`, `TrackManager`. One module, per D2. Constants from `core.constants` only. |
| 1 | `pipelines/tracking.py` *(new)* | `frame_detections(product, ...) -> list[Measurement]` via §5.1's bin-centre helpers, then drives `TrackManager` frame by frame. Kept out of `pipelines/scenarios.py`, which is near the 400-line guidance of `docs/conventions/style.md` §9. |
| 2 | `scenarios/scenario_003_tracking.toml` *(new)* | S1's blocks verbatim, `[scenario].name = "scenario-003-tracking"`, plus new `[detection]` and `[tracking]` blocks. `_require_keys` in `pipelines/scenarios.py` is extended **additively**, both blocks optional, so the three scenario-001 TOMLs keep loading byte-identically. |
| 3 | `teaching/scopes/rd_map.py` *(edit)* | Additive keyword-only `detections=`, `track_estimate=`, `gate_extent=`, all defaulting to `None`. Follow the existing private `_mark_truth` helper; do not restructure it. |
| 4 | `teaching/scopes/track_plot.py` *(new)* | `render_track_plan_view(...)`. `matplotlib` lazily imported through the existing `teaching.plotting.require_pyplot`. |
| 5 | `scripts/run_scenario.py` *(edit)* | Track when the TOML carries `[tracking]`; write the §9 files; extend `clear_previous_frames`; add `--no-tracking`. |
| 6 | Docs | `README.md`, `spec/starter.md`, `spec/structure.md`, `docs/README.md` — §11. |

### 10.1 What this scenario consumes but never edits

`core/detection.py`, `core/dsp.py`, `core/windows.py`, `core/signal.py`, `core/geodesy.py`,
`core/ambiguity.py`, `pipelines/trajectories.py`, and `data/flight_coordinates.csv`. If the
tracker appears to need a change in any of them, that is a finding to raise, not a change to make
inside this workstream.

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
`src/radar_forge/teaching/scopes/rd_map.py` are **shared** with scenario 001: this slice makes
only the additive edits in steps 2, 3 and 5, and every existing call site and TOML must keep
working unchanged.

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
3. **D2's promotion trigger is not fired.** One association strategy, no fusion layer, one module.
   Restating it here so that the next person to open `core/tracking.py` knows the rule before they
   add JPDA to it.
4. **No new runtime dependency.** `scipy.optimize.linear_sum_assignment` comes from a dependency
   already in core. None of the five reference projects is vendored or imported; per `CLAUDE.md`,
   a dependency needs a justification in the PR, and "we already have `scipy`" is the
   justification for not adding one.

Named as future work, with the trigger that would justify each:

| Extension | Trigger |
| :--- | :--- |
| Two-state measurement via velocity unfolding (§5.3) | Wanted as soon as settling time matters |
| Tracking over S3, where velocity is genuinely unambiguous | After `unfold_doppler_dual_prf` is exercised end to end |
| A second target, crossing tracks, JPDA/MHT | Fires D2's promotion to `core/tracking/` |
| GLMB/LMB random-finite-set tracking | Only with a multi-target, high-clutter scenario to justify it |
| 3-D ENU state and an EKF | Needs `array/` and a measured azimuth |
| Track fusion across sensors | Fires D2's promotion |

## 12. Acceptance criteria

Over the default 120 s window of `scenarios/scenario_003_tracking.toml`:

1. **One track.** Exactly one confirmed track exists in at least 95 % of frames, and its
   `track_id` never changes across the run.
2. **Range accuracy.** The confirmed track's range RMSE against `truth.csv` is below one range
   bin (74.95 m). The expected value is well under `sigma_range_m`, because the filter smooths.
3. **Consistency.** The normalised innovation squared, averaged over confirmed frames, lies inside
   the 95 % confidence interval of a χ²(1) mean — an interval, not a tolerance, per
   `docs/conventions/testing.md` §2. This is the check that catches a mis-scaled `R` or `Q`, which
   criterion 2 alone will not.
4. **False alarms behave.** The measured false-alarm rate over `cfar_valid_mask` cells agrees with
   `pfa` inside its Poisson confidence interval, and **no false alarm is promoted to a confirmed
   track**. The test asserts §3's analytic bound of 0.003 expected false tracks, not merely the
   observed zero — observing zero is consistent with an associator that never associates anything.
5. **Velocity sign.** The track's estimated `range_rate_mps` has the same sign as `truth.csv`'s
   `radial_velocity_mps` in every confirmed frame. A sign error here produces a perfectly
   plausible-looking plot, exactly as scenario 001 §7.1 warns.
6. **The pictures exist.** `run_scenario.py` writes `track_00000.png`, an `rd_00000.png` carrying
   detection markers, `tracks.csv` and `detections.csv`.

Encoded as `tests/pipelines/test_scenario_003.py`, marked `slow`, over a 10-frame window so that
`make check` stays fast — 10 rather than scenario 001's 5, because 4-of-5 initiation needs five
frames before there is anything to assert.

Unit-level ground truth, per `docs/conventions/testing.md` §3, comes first and is analytic: a
noiseless constant-velocity target must be tracked with zero steady-state innovation to float
precision; `associate_gnn` must return the known optimal assignment of a hand-built cost matrix
with a unique solution; `gate_threshold` must match the tabulated χ² quantile; `process_noise_dwna`
must reproduce the worked `Q` in reference [2]; and the empirical gate acceptance rate over a
seeded draw must match `gate_probability` inside its binomial confidence interval.

## References

.. [1] Y. Bar-Shalom, P. K. Willett and X. Tian, *Tracking and Data Fusion: A Handbook of
       Algorithms*, YBS Publishing, 2011, ch. 2 (gating), ch. 3 (assignment).
.. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and
       Navigation*, Wiley, 2001, §5.2 (discrete white-noise acceleration), §6.3 (validation
       gating and nearest neighbour), §11.7 (track initiation).
.. [3] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking Systems*, Artech
       House, 1999, ch. 6 (M-of-N initiation, global nearest neighbour).
.. [4] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014,
       §6.5 (CFAR), §7.3 (measurement accuracy and the range CRLB).
.. [5] D. F. Crouse, "On implementing 2D rectangular assignment algorithms," *IEEE Trans. Aerosp.
       Electron. Syst.*, vol. 52, no. 4, pp. 1679-1696, 2016. The algorithm behind
       `scipy.optimize.linear_sum_assignment`.
.. [6] P. A. Thomas, J. Barr, B. Balaji and K. White, "An open source framework for tracking and
       state estimation ('Stone Soup')," *Proc. SPIE 10200, Signal Processing, Sensor/Information
       Fusion, and Target Recognition XXVI*, 2017.
.. [7] R. R. Labbe, *Kalman and Bayesian Filters in Python*, 2020. The FilterPy companion text.
.. [8] B.-T. Vo and B.-N. Vo, "Labeled random finite sets and multi-object conjugate priors,"
       *IEEE Trans. Signal Process.*, vol. 61, no. 13, pp. 3460-3475, 2013. Cited for the
       future work in §11, not implemented here.
