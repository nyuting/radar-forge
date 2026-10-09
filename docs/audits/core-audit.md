# `core/` audit — code against spec

Scope: every public name in `src/radar_forge/core/`. `pipelines/`, `viz/` (then `teaching/`) and `scripts/`
are a separate stream and are untouched here, as are `docs/conventions/style.md` and
`scripts/check_conventions.py`.

The audit has two halves, per the [audit method](../conventions/audit.md#1-auditing-code-against-the-spec). **Conformance**: does the code compute what the specs say, to the accuracy they claim,
with the contract they document, under `structure.md`'s decisions D1–D8? **Quality**: is this
the best available form, across all eight Q dimensions, whether or not conformance holds?

There are no prototype implementations in `spec/` to diff against — one 8-line dataclass
interface sketch and nothing else — so "reference" throughout means the *specified behaviour*:
the governing equation as written or cited, the numerical bounds the specs fix, the documented
array shapes, frames, signs and units, and D1–D8.

## Status at 2026-10-08

Re-checked against `main` at `8a9e579`. Every fix (F1–F5) is still in place, and every
recommendation (R1–R4) is still unapplied. Two later merges changed `core/` under this audit,
so parts of the text below describe code that has since moved:

- **`tracking.py` is now the package `core/tracking/`** (PR #2, `91a3454`). The audited module
  survives as `core/tracking/kalman.py`, with the same filter mathematics. What `TrackManager`
  did is now `KalmanTracker` (`kalman.py`). The name `TrackManager` belongs to a different,
  smaller class in `lifecycle.py` that has no `step`. `Track`, `TrackStatus` and
  `TRACK_STATUSES` moved to `tracks.py`, and `Track` is now generic over its filter. The rows
  in [`tracking.py`](#trackingpy) are kept as audited, with the current location noted there.
- **Not covered by this audit:** the modules PR #2 added beside `kalman.py` — `coordinates`,
  `motion`, `measurement_models`, `estimation`, `ukf`, `association`, `initiation`, `tracks`,
  `lifecycle` and `tracker`. They need their own pass. The 2-D CFAR functions PR #4 added to
  `detection.py` were listed here too; they are now audited, see
  [Status at 2026-10-09](#status-at-2026-10-09).
- **`Detection` changed shape** (PR #4, `3652134`). `noise_power_w` is now
  `cfar_noise_estimate_w`, and `snr_db` is gone.

## Status at 2026-10-09

Audited against `main` at `20d4610`: the four 2-D CFAR functions, the `Detection` reshape and
the rest of what PR #4 changed in `detection.py`. They are in
[`detection.py` — 2-D CFAR](#detectionpy--2-d-cfar). That pass adds F6–F10 and R5–R7. Two of the
findings are fixes to `detection.py` (F6, F7), one is documentation (F8), and two correct a spec
(F9, F10). F8 also measured something that belongs to the pipelines stream. Scenario 003's
pulsed TOML gets more false alarms than its `pfa`, 1.29 times design at `1e-4`, because S2's
matched-filter output is oversampled. The tracking modules listed above are still not covered
here: their audit is left to the `spec/tracker-001.md` workstream.

Since then, R1 and R3 have been applied, both as API breaks. R2 and R4 are declined, and stay
recorded for the reasons given. R5–R7 are still open.

## Verdict key

| Mark | Meaning |
| :--- | :--- |
| ✓ | Sound. Nothing better available at reasonable cost; no action. |
| **F*n*** | A finding. Links to the section that records it. Fixed unless the section says otherwise. |
| **R*n*** | A recommendation, deliberately **not** applied unless its entry says **Applied** — see [Recommended, not applied](#recommended-not-applied). |

Q1 mathematical and radar-physics accuracy · Q2 clarity · Q3 speed and vectorisation ·
Q4 conciseness · Q5 documentation · Q6 naming · Q7 inputs · Q8 outputs.

## Findings

| # | Where | Dimension | Verdict |
| :--- | :--- | :--- | :--- |
| [F1](#f1--frameresult-was-returned-but-never-exported) | `tracking.TrackManager.step` (now `tracking.kalman.KalmanTracker.step`) | Q8 | Code wrong — fixed |
| [F2](#f2--three-docstrings-that-misdescribe-their-code) | `tracking.state_model_matrices`, `radar.BistaticRadar.range_resolution_at_bistatic_angle_m`, `signal.PropagationPaths` | Q5 | Docs wrong — fixed |
| [F3](#f3--the-mti-canceller-copied-every-tap) | `dsp.mti_filter` | Q3, Q4 | Suboptimal — fixed, measured |
| [F4](#f4--taper-names-were-a-bare-str-where-cfar-variants-are-a-literal) | `windows.taper` | Q6, Q7 | Inconsistent — fixed |
| [F5](#f5--the-scenario-001-noise-bandwidth-was-stated-as-1-mhz) | `spec/scenario-001-xband.md` §3.2 | Conformance | **Spec** wrong — fixed |
| [F6](#f6--a-2-d-cfar-pair-was-not-checked-to-be-two-integers) | `detection._ring`, so all four `_2d` functions | Q7 | Code wrong — fixed, red verified |
| [F7](#f7--the-2-d-variants-were-an-inline-literal-and-a-private-tuple) | `detection.cfar_noise_estimate_2d_w`, `cfar_threshold_2d_w`, `cfar_detect_2d` | Q4, Q6, Q7 | Inconsistent — fixed |
| [F8](#f8--the-calibrations-independence-assumption-did-not-name-tapering) | `detection` module notes, `cfar_threshold_2d_w` | Q1, Q5 | Docs incomplete — fixed, measured |
| [F9](#f9--scenario-003-said-a-2-d-window-halves-the-cells-needed) | `spec/scenario-003-tracking.md` §13.2 | Conformance | **Spec** wrong — fixed |
| [F10](#f10--data-001-called-a-thermal-noise-power-the-cfar-estimate) | `spec/data-001-formats.md` §6.8 | Conformance, Q8 | **Spec** wrong — fixed |

Everything else audited below carries no finding. That is the expected result: the suite is
strong, the earlier standardisation pass hardened it, and most of `core/` is already the best
available form.

## `constants.py`

One row per constant; all are `Final`, unit-suffixed, cited, and enforced as unique by rule R4.

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `SPEED_OF_LIGHT_MPS` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BOLTZMANN_JPK` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `STANDARD_NOISE_TEMPERATURE_K` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `EARTH_RADIUS_M` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `FOUR_THIRDS_EARTH_RADIUS_M` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `WGS84_SEMI_MAJOR_AXIS_M` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `WGS84_FLATTENING` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Both SI-defined values are exact to their 2019 definitions. Deriving `e² = f(2 − f)` in
`geodesy.py` rather than storing an eighth constant is right: it cannot drift out of step with
the two defining ones.

## `radar_equation.py`

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `received_power_w` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [R1](#recommended-not-applied) | ✓ |
| `bistatic_received_power_w` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [R1](#recommended-not-applied) | ✓ |

Both match Richards §2.2.1 and Willis §4.1 (eq. 4.1a) term for term, including the `(4π)³` hoisted to a
module constant. The bistatic denominator carries the product `R_t²R_r²`, not a power of a sum,
so the ovals of Cassini are right and the equal-range reduction to the monostatic form is exact —
the doctest asserts it. The ranges are validated strictly positive and the loss factor is
validated `≥ 1`, which catches the commonest call-site error of passing a dB loss. D8 is
honoured: the bistatic cross-section parameter is named `bistatic_rcs_m2` precisely so a caller
cannot quietly substitute a monostatic one, and the Notes say what that costs at wide angles.

## `targets.py`

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `dbsm_to_m2` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `m2_to_dbsm` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PointTarget` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PointTarget.from_dbsm` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PointTarget.rcs_dbsm` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Swerling 0 only, and the module says so and says why rather than stubbing the fluctuating
models. `rcs_m2 = 0` is admitted (a noise-only scenario) while `rcs_dbsm` returns `-inf` for it
rather than raising, which is the right asymmetry: the zero is meaningful, its logarithm is not.

## `waveforms.py`

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `range_resolution_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `sweep_rate_hzps` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `beat_frequency_hz` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `range_from_beat_frequency_m` | ✓ | ✓ | ✓ | ✓ | ✓ | [R2](#recommended-not-applied) | [R2](#recommended-not-applied) | ✓ |
| `lfm_chirp` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Conformance to scenario 001 §3.2 is exact: `c/2B` at `B = 2 MHz` is 74.9481 m, and the three
sweep rates 2.000, 10.000 and 12.000 GHz/s follow from `B/T` at the three chirp durations.
`lfm_chirp` samples the half-open interval `[0, T)` and says why — including `t = T` would
double-count one sample per sweep across a CPI, which is the kind of off-by-one that shows up as
a slow phase creep rather than as a wrong answer. The complex-baseband Nyquist bound is
`f_s ≥ B`, not `2B`, and the check and the docstring both get that right; scenario 001's `fs/B`
of 0.50 for S1 is legal because an FMCW receiver samples the *deramped* beat, which the spec's
own §3.2 sidebar derives and which `Radar.__post_init__` enforces only for the pulsed case.

## `windows.py`

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `TaperName`, `TAPER_NAMES` | ✓ | ✓ | ✓ | [F4](#f4--taper-names-were-a-bare-str-where-cfar-variants-are-a-literal) | ✓ | [F4](#f4--taper-names-were-a-bare-str-where-cfar-variants-are-a-literal) | — | — |
| `taper` | ✓ | ✓ | ✓ | ✓ | ✓ | [F4](#f4--taper-names-were-a-bare-str-where-cfar-variants-are-a-literal) | [F4](#f4--taper-names-were-a-bare-str-where-cfar-variants-are-a-literal) | ✓ |
| `apply_taper` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `coherent_gain_linear` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `processing_loss_db` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

The window shapes come from `scipy.signal.windows` rather than being reimplemented, which is
both the right dependency call and clean under audit rule C3. Two details are better than the usual textbook
treatment and are worth naming. The Taylor `nbar` is *derived* from the requested sidelobe
level, `nbar ≥ 2A² + 1/2`; fixing it at a small constant, which is the common shortcut, silently
returns a window that does not achieve the level it was asked for. And the Chebyshev design is
refused below 45 dB, where SciPy merely warns and returns a "window" whose end weights exceed
its centre. `apply_taper` reshapes the window to `(1, …, n, …, 1)` and broadcasts rather than
materialising it, per the house rule, and says so in a comment.

`processing_loss_db` is scale-invariant in its argument, so a normalised and an unnormalised
taper of the same shape report the same loss — asserted in the docstring and worth keeping,
because it is the property that makes the number comparable across call sites.

## `ambiguity.py`

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `fold_velocity_mps` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `unfold_doppler_dual_prf` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

The strongest piece of Q7/Q8 design in the module. `max_velocity_mps` and `tolerance_mps` are
**required keyword-only** arguments, not defaulted, because each is a claim about the target
rather than a tuning knob — and the Notes demonstrate the failure a generous default would hide:
a 300 m/s target on scenario 001 S3's bursts aliases onto a candidate inside the bound and comes
back as −158.88 m/s with a near-zero residual. The second return value is the residual, so an
unresolved measurement is reported as `nan` and a marginal one is thresholdable, rather than the
function guessing. Comparing candidates on the circle (folding the difference before taking its
magnitude) is correct and easy to get wrong; a candidate near `+v_ua` and a measurement near
`−v_ua` are neighbours.

One apparent spec divergence, already reconciled in the code's own Notes and re-checked here:
the docstring gives the reachable span as `q · v_ua,a ≈ 229 m/s` for the 5:6 bursts, while
scenario 001 §3 gives ±191.194 m/s. Both are right. 229.4 m/s is where the *pair* of readings
repeats — half the least common multiple of the two fold spans, 458.87 m/s — and 191.194 m/s is
the scenario's design target, chosen to match what S2 reaches by a different mechanism. The
docstring says exactly that. No action.

## `geodesy.py`

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `geodetic_to_ecef_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ecef_to_enu_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `geodetic_to_enu_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `enu_to_range_azimuth_elevation` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

The closed-form WGS-84 forward transform, with `N(φ)` the prime-vertical radius; no iteration,
because only the inverse needs one. The ENU rotation matrix matches Brown & Hwang appendix B
row for row.

The load-bearing convention is `atan2(east, north)`, not `atan2(y, x)`, which is what puts zero
azimuth at true north and makes it increase clockwise — D5's sign convention. It is called out
in the module docstring, in the function docstring, and in an inline comment at the call, which
is the right amount of repetition for a convention whose violation produces a mirrored track
that still looks like a track.

Q8 note, deliberate and documented: zero range and the overhead singularity are **not** guarded.
That is the correct choice here — a target at zero range is a modelling error to be caught where
the geometry is built, not a value to substitute — and `BistaticRadar.target_ranges_m` routes
around it by taking norms directly rather than going through this function, with a comment
saying why.

## `radar.py`

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `WaveformKind` | ✓ | ✓ | — | ✓ | ✓ | ✓ | — | — |
| `RadarLike` | ✓ | ✓ | — | ✓ | ✓ | ✓ | — | — |
| `Transmitter` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Transmitter.wavelength_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Transmitter.pulse_repetition_interval_s` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Transmitter.duty_cycle_linear` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Transmitter.sweep_rate_hzps` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Receiver` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Receiver.noise_figure_linear` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Radar` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Radar.wavelength_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Radar.range_resolution_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Radar.unambiguous_range_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Radar.unambiguous_velocity_mps` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Radar.noise_power_w` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [F5](#f5--the-scenario-001-noise-bandwidth-was-stated-as-1-mhz) |
| `Radar.n_samples_per_chirp` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Radar.n_samples_per_pri` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BistaticRadar` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BistaticRadar.receiver_enu_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BistaticRadar.baseline_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BistaticRadar.wavelength_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BistaticRadar.range_resolution_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BistaticRadar.unambiguous_range_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BistaticRadar.unambiguous_velocity_mps` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BistaticRadar.noise_power_w` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BistaticRadar.n_samples_per_chirp` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BistaticRadar.n_samples_per_pri` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BistaticRadar.bistatic_angle_rad` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BistaticRadar.range_resolution_at_bistatic_angle_m` | ✓ | ✓ | ✓ | ✓ | [F2](#f2--three-docstrings-that-misdescribe-their-code) | ✓ | ✓ | ✓ |
| `BistaticRadar.target_ranges_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Conformance, re-derived rather than read across.** Every ambiguity figure in scenario 001 §3.1
and scenario 002 §3 comes out of these properties exactly:

| Quantity | Formula in code | Spec | Computed |
| :--- | :--- | ---: | ---: |
| λ at 9.8 GHz | `c/f0` | 30.5911 mm | 30.5911 mm |
| Range resolution | `c/2B` at 2 MHz | 74.9481 m | 74.9481 m |
| S1 `R_ua` (FMCW) | `c·(f_s/2)/(2α)` | 37.474 km | 37.474 km |
| S2 `R_ua` (pulsed) | `c/(2·PRF)` | 5.996 km | 5.9958 km |
| S3 `R_ua` A / B | `c·(f_s/2)/(2α)` | 29.979 / 24.983 km | 29.979 / 24.983 km |
| S1 `v_ua` | `λ·PRF/4` | ±7.6478 m/s | ±7.6478 m/s |
| S2 `v_ua` | `λ·PRF/4` | ±191.194 m/s | ±191.194 m/s |
| S3 `v_ua` A / B | `λ·PRF/4` | ±38.2388 / ±45.8866 m/s | ±38.2388 / ±45.8866 m/s |
| Bistatic `ΔR(β)` | `c/(2B·cos(β/2))` | 79.85 m at 40.3°, 175.21 m at 129.3° | 79.84 m, 175.16 m |

The bistatic resolution figures agree to the precision of the spec's own rounding of β to a
tenth of a degree. `BistaticRadar.unambiguous_range_m` correctly bounds the **range sum**
`R_t + R_r` rather than a single range, and its docstring says that the factor of two over the
monostatic case is not a bonus — the sum it bounds is itself about twice a one-way range.

Two pieces of numerical care are right and are the sort of thing an audit exists to confirm
rather than to change. `bistatic_angle_rad` clips the law-of-cosines cosine to `[−1, 1]` before
`arccos`, because for a target exactly on the baseline the exact value is −1, rounding lands
just past it, and the unclipped form returns NaN for the single geometry most worth asking
about. And `range_resolution_at_bistatic_angle_m` snaps `cos(β/2)` to zero below `1e-8` rather
than dividing by `np.cos(np.pi/2)`'s 6.1e-17: that would turn a genuine singularity into a
finite 1e18 m that reads like a measurement. The floor is about `√ε`, which is the resolution
`arccos` itself has near β = π, so reporting a large finite number there would be claiming
precision the angle behind it does not have. `inf` is returned rather than raised, which is
right for a caller plotting resolution across a scene.

The `range_tx_m` / `range_rx_m` naming is documented at module level as qualifying *the site the
range is measured to*, consistent with `gain_tx_linear` / `gain_rx_linear`, with a note that
"leg" and "path" are already taken. Per D6 these are correctly **not** fields of
`PropagationPaths`.

## `signal.py`

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `PropagationPaths` | ✓ | ✓ | ✓ | ✓ | [F2](#f2--three-docstrings-that-misdescribe-their-code) | ✓ | ✓ | ✓ |
| `PropagationPaths.n_paths` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PropagationPaths.range_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `line_of_sight_paths` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `bistatic_line_of_sight_paths` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `bistatic_doppler_hz` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `thermal_noise` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [R3](#recommended-not-applied) | ✓ |
| `fmcw_deramp_baseband` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pulsed_baseband` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

This module carries the highest density of load-bearing decisions in `core/`, and they hold up.

**D1 and D6.** `PropagationPaths` is D1's dataclass with unit suffixes added, and D6's claim
that it covers bistatic geometry unchanged is borne out: `delay_s` carries `(R_t + R_r)/c`,
`range_m` is the half-sum, and the generators consume both sitings with no branch, because every
use of range is through `τ = 2R/c` and every use of Doppler through `v = f_d λ/2`. The factor of
two the monostatic convention inserts cancels the one the bistatic definition removes. The
`__post_init__` shape validation is the right Q8 guard for a dataclass a backend author will
construct by hand.

**The sign lock, and why FMCW is a down sweep.** In a deramp the range and Doppler signs are not
independent: both terms scale with the same delay, so the mixer sideband that puts range at a
positive beat frequency also fixes the Doppler sign, and conjugating to fix one breaks the
other. The sweep *direction* is the free parameter, and a down sweep is the only combination
that is both physically consistent and D5-compliant (closing positive). This is stated in the
module docstring with a citation to Stove; it is exactly the kind of thing that, undocumented,
gets "fixed" by a later contributor with a conjugate.

**Two modelling limits stated rather than hidden.** `pulsed_baseband` folds the delay modulo the
PRI — which is how a real radar folds targets beyond its unambiguous range, and what makes
scenario 001 S2 show its target at the wrong range — while computing the residual phase from the
**true** delay, since folding is an artefact of when the receiver listens and does not change the
carrier phase the target imposed. Using the folded delay there would corrupt the Doppler, and
the comment says so. Eclipsing is named as unmodelled, with the 25% duty cycle of S2 quantified.
`fmcw_deramp_baseband` checks the stop-and-hop bound and *warns* rather than silently smearing,
at a quarter of a range bin of intra-chirp motion; scenario 001 §4.2 says range migration must be
documented in this module, and it is, with the S1 arithmetic (0.08 m against a 74.95 m bin).

`bistatic_doppler_hz` uses the vector form `(v·û_t + v·û_r)/λ` rather than
`(2v/λ)cos δ cos(β/2)`. Those are equivalent, and the vector form is the better choice for the
stated reason: a scenario has positions and velocities to hand, so extracting δ and β first only
adds a step that can go wrong. The Notes name both nulls, including the baseline-parallel one
that has no monostatic analogue.

The zero-cross-section path is handled by masking rather than by letting a legitimate 0 m² target
divide by zero — correct, and the alternative (rejecting `rcs_m2 = 0`) would remove the
documented way to build a noise-only scenario.

## `dsp.py`

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `matched_filter` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `range_fft` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `doppler_fft` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `range_doppler_map` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `range_bin_centers_m` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `doppler_bin_centers_mps` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `mti_filter` | ✓ | ✓ | [F3](#f3--the-mti-canceller-copied-every-tap) | [F3](#f3--the-mti-canceller-copied-every-tap) | ✓ | ✓ | ✓ | ✓ |

The axis-agnostic design — every function takes an explicit `axis` rather than assuming a
layout — is the right Q7 call for a module that must accept cubes from any simulator, and the
canonical `(n_pulses, n_samples, n_rx)` layout is documented in one place and referred to from
each function.

The asymmetry that `range_fft` does **not** `fftshift` and `doppler_fft` **does** is the
module's most valuable Q8 statement, and it is correct physics rather than a convention: bin 0
of a range profile is zero range and negative range is meaningless, while zero Doppler belongs
in the middle with closing and opening targets either side. It is stated in the module docstring
and in both function docstrings.

`matched_filter` is `scipy.signal.fftconvolve` at `mode="full"` rather than a hand-written
`ifft(fft(x)·conj(fft(h)))`. The Notes explain the difference — the hand-written form is a
*circular* correlation that wraps far-range energy round to zero range, a silent error that
looks like a near-range ghost — and the return contract says that zero lag sits at index
`n_reference − 1`, which is the number a caller needs and the one most often assumed to be zero.
The docstring's peak-index doctest pins it.

`range_fft` refuses an `n_fft` shorter than the data rather than truncating silently, with a
message telling the caller to slice explicitly if that is what they meant. That is the right
Q7 behaviour: NumPy's own `fft` truncates, and the truncation is invisible in the output.

`doppler_bin_centers_mps` returns velocities matching the shifted axis, positive closing per D5,
and derives them from `fftfreq` rather than by hand.

## `detection.py`

The four `_2d` CFAR functions and the `Detection` field renames came after this audit; see
[Status at 2026-10-08](#status-at-2026-10-08). They are audited in
[`detection.py` — 2-D CFAR](#detectionpy--2-d-cfar).

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `CfarVariant`, `CFAR_VARIANTS` | ✓ | ✓ | — | ✓ | ✓ | ✓ | — | — |
| `Detection` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `default_os_rank` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `cfar_probability_of_false_alarm` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `cfar_threshold_factor` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `cfar_valid_mask` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `cfar_noise_estimate_w` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `cfar_threshold_w` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `cfar_detect` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `cluster_detections` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

The four closed-form `P_fa(α)` expressions match their cited sources — Gandhi & Kassam for
CA/GO/SO, Rohling eq. 14 for OS — and the module's own Notes pin down two cross-variant
identities at `N = 1` that are hand-checkable: SO and OS at rank 1 are both the minimum of two
unit-mean exponentials, giving `2/(2+α)`, and GO and OS at rank 2 are both the maximum, giving
`2/((1+α)(2+α))`. The warning that a *single* reference cell would give `1/(1+α)`, and that no
window geometry here produces one, is exactly the factor-level trap this module exists to avoid.
The GO cancellation is analysed rather than asserted safe.

Q3 is genuinely thought through and documented. CA, GO and SO run off one cumulative sum, so
they cost O(map) independent of `n_train`; OS cannot, because an order statistic needs all `2N`
cells, so it materialises a sliding window (free, a view) and partitions it in bounded chunks —
with the Python loop justified in a comment, per the house rule, on the grounds that the point
of the loop is to *not* have the whole temporary live at once.

The edge handling is the best Q8 decision in the module: cells without a complete reference
window get a `nan` threshold and are never declared detections, rather than being estimated from
a truncated window, which would change `P_fa` in exactly the way CFAR exists to prevent.
`cfar_valid_mask` exists so a measured false-alarm rate can be taken over the tested cells alone
— counting untested edges as non-detections dilutes the rate towards zero and makes a badly
thresholded detector look correct. `cfar_detect` compares under an explicit `where=` mask rather
than relying on NaN comparisons being False, which keeps the error-on-warning test policy clean.

`cfar_threshold_factor` brackets by doubling and then bisects, with both choices justified:
bisection cannot fail on a bracketed monotone function, where a derivative method over a
many-decade range needs careful scaling. The bracket ceiling raises a message that names the
real cause — the window is too small for the requested rate — rather than a numerical one.

## `detection.py` — 2-D CFAR

Audited at `20d4610`. It covers what PR #4 added (`8eb9b2e`) and reshaped (`3652134`): the four
`_2d` functions, the variant names they take, and the `Detection` and `cluster_detections`
renames.

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `CfarVariant2d`, `CFAR_VARIANTS_2D` (new) | ✓ | ✓ | — | [F7](#f7--the-2-d-variants-were-an-inline-literal-and-a-private-tuple) | ✓ | [F7](#f7--the-2-d-variants-were-an-inline-literal-and-a-private-tuple) | — | — |
| `Detection` (`cfar_noise_estimate_w`, no `snr_db`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `cluster_detections` (`noise_estimate_w`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [R5](#recommended-not-applied) | ✓ |
| `cfar_valid_mask_2d` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [F6](#f6--a-2-d-cfar-pair-was-not-checked-to-be-two-integers), [R5](#recommended-not-applied) | ✓ |
| `cfar_noise_estimate_2d_w` | ✓ | ✓ | ✓ | ✓ | ✓ | [F7](#f7--the-2-d-variants-were-an-inline-literal-and-a-private-tuple) | [F6](#f6--a-2-d-cfar-pair-was-not-checked-to-be-two-integers), [R5](#recommended-not-applied) | ✓ |
| `cfar_threshold_2d_w` | ✓ | ✓ | ✓ | ✓ | [F8](#f8--the-calibrations-independence-assumption-did-not-name-tapering) | [F7](#f7--the-2-d-variants-were-an-inline-literal-and-a-private-tuple) | [F6](#f6--a-2-d-cfar-pair-was-not-checked-to-be-two-integers), [R5](#recommended-not-applied) | ✓ |
| `cfar_detect_2d` | ✓ | ✓ | ✓ | ✓ | ✓ | [F7](#f7--the-2-d-variants-were-an-inline-literal-and-a-private-tuple) | [F6](#f6--a-2-d-cfar-pair-was-not-checked-to-be-two-integers), [R5](#recommended-not-applied) | [R6](#recommended-not-applied) |

**Conformance.** Scenario 003 §13.2's "Landed" paragraph is the specification, and the code
matches each clause of it. `n_train` and `n_guard` are per axis, `axes` defaults to `(-2, -1)`,
and `wrap_axes` exists. Only CA and OS are offered. A ring of M cells is calibrated as a 1-D
window of M/2 per side, which is exact because M is always even: it is an odd product less an
odd product. The default `axes` match data-001's `rd_layout = "frame,doppler,range"`, so a
stack of frames works unchanged, and the rate test runs on just such a 3-D stack. The default
OS rank is `default_os_rank(M // 2)`, which is Rohling's `3M/4` for the ring's own M. The
threshold and the estimate compute it the same way, so they cannot disagree.
`scipy.ndimage` is BSD-licensed, so D4 and audit rule C3 hold. The paragraph above that one was not
right, and F9 corrects it.

**Edge handling matches the 1-D sibling.** An incomplete ring gives `nan`, and the cell is
never declared. `cfar_valid_mask_2d` agrees with `~isnan(estimate)` in all three wrap
combinations, and a test asserts that. A circular axis is extended by one margin of wrapped
cells at each end, then cut back. An axis, circular or not, that is shorter than one window
leaves every cell untested. Wrapping a ring round a circle shorter than itself would count
cells twice. The test for that is `test_valid_mask_2d_is_empty_when_an_axis_is_shorter_than_the_ring`.

**Q1, roundoff from the box filters, measured.** CA is the outer box's total less the guard
box's, each from `uniform_filter`. That filter is a running sum, so a strong cell leaves a
roundoff residue along every line it passes through, including the line through the cell
under test, whose own power sits in both boxes and cancels. The table below puts one target in a
`(256, 1000)` unit-mean exponential map and compares the result with a ring summed term by term.
The ring is `n_train = (4, 8)` and `n_guard = (1, 2)`, wrapping Doppler. Each entry is the worst
relative error anywhere:

| Target above floor | 60 dB | 80 dB | 100 dB | 120 dB | 140 dB |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 2-D ring | 2.8e-12 | 5.4e-10 | 1.9e-8 | 3.1e-6 | 3.5e-4 |
| 1-D window, `n_train = 16`, for comparison | — | — | 3.6e-7 | — | 2.3e-3 |

Scenario 003's strongest return is about 69 dB, where the error is of order 1e-11. Even at
140 dB it is 0.0015 dB of threshold. The 1-D cumulative sum, already audited ✓, is 7 to 19 times
worse, because its prefix runs the whole axis. No change.

**Q3, measured.** The CA docstring claims a cost that is independent of the ring's size. The
OS docstring claims "substantially slower". On a `(256, 1000)` map (S1's cube) with Doppler
wrapping, the best of 5 on an Apple M2:

| `n_train`, `n_guard` | M | CA estimate | OS estimate |
| :--- | ---: | ---: | ---: |
| (1, 2), (1, 1) | 26 | 2.98 ms | 72 ms |
| (2, 4), (1, 2) | 76 | 3.00 ms | 193 ms |
| (4, 8), (1, 2) | 216 | 3.01 ms | 509 ms |
| (8, 16), (2, 4) | 816 | 3.31 ms | 1871 ms |

CA is flat in M, as claimed. OS is close to linear in M, at 2.3 to 2.8 ms per reference cell. Two
alternatives were measured against the same map, and neither earns a change:

- **PR #4's predecessor** (`8eb9b2e`: a hand-written summed-area table and a chunked ring
  gather). It agrees with today's code to `rtol = 1e-12` and is slightly slower: CA 3.57
  against 3.12 ms, OS 215 against 194 ms at M = 76. PR #4's simplification cost nothing.
- **A sliding-window gather and `np.partition`**, the 1-D OS method, applied to the ring. It is
  bit-identical, at 194 ms against `rank_filter`'s 194 ms. `rank_filter` is the shorter form.

The OS threshold factor is a bisection over Rohling's product. At M = 816 it costs 1.6 ms, so it
is never the bottleneck.

**The 1-D/2-D pair, compared clause by clause.**

| Aspect | 1-D | 2-D | Verdict |
| :--- | :--- | :--- | :--- |
| Names | `cfar_noise_estimate_w`, `cfar_threshold_w`, `cfar_detect`, `cfar_valid_mask` | Same, with `_2d` before the unit suffix | ✓ |
| Guard and training | Per side; guard excluded; the cell under test sits in the guard band | Per side, per axis; the guard box holds the cell under test | ✓ |
| Variant type | `CfarVariant`, `CFAR_VARIANTS = get_args(...)` | Inline `Literal`, private tuple | [F7](#f7--the-2-d-variants-were-an-inline-literal-and-a-private-tuple) |
| Argument order | `power_w, *, pfa, n_train, n_guard, variant, rank, axis` | `power_w, *, pfa, n_train, n_guard, variant, rank, axes, wrap_axes` | ✓ |
| Keyword-only | Everything after the map | Everything after the map | ✓ |
| `n_guard` default | 1 | Required | ✓, deliberately — see [left alone](#what-was-deliberately-left-alone) |
| Argument checks | Shared `_validate_window`, `_validate_rank`, `_as_power_w` | Same helpers; the pairs were not checked | [F6](#f6--a-2-d-cfar-pair-was-not-checked-to-be-two-integers) |
| Return contract | Same shape; `nan` (estimate, threshold) or `False` (mask) where untested | Same | ✓ |
| Error messages | `variant must be one of (...), got ...` | The same text plus "for a 2-D ring" and the reason | ✓ |

**References.** The 2-D docstrings cite Finn & Johnson [1] for CA and Rohling [4] for OS. Both
papers treat a 1-D window. What the ring borrows from them is the P_fa of M i.i.d. reference
cells. That P_fa holds for any arrangement of the cells, because a sum and an order statistic do
not depend on the order of their inputs. The docstrings claim no more than that, so the
citations fit. The locators were checked against their sources in PR #11 (`c48aaf2`). This pass
had no source to hand and did not re-check them, so it judges only whether each claim fits its
citation. The module's own reference list is [R7](#recommended-not-applied).

**The `Detection` reshape.** The rename is right on Q6 and Q8. `noise_power_w` means thermal
noise in `Radar.noise_power_w` and in data-001's `bursts[].noise_power_w`. A per-cell CFAR
estimate under the same name would let a caller compute `snr_db` against the wrong denominator
without noticing. The keyword is `noise_estimate_w` rather than `cfar_noise_estimate_w`, so it
does not shadow the function of that name, the hazard R2 records for
`range_from_beat_frequency_m`. Removing `snr_db` is right as well: data-001 §6.5 defines it
against the burst's thermal noise, and `scripts/run_scenario.py` now computes it that way, with a
comment saying so. One place in the specs had not caught up. Data-001 §6.8 still described a
`noise_power_w` dataset as "the noise estimate used by CFAR": F10.

PR #4's two changes to the 1-D path were re-checked. CA's factor at scenario 003's `n_train = 16`
and `pfa = 1e-5` is 13.856402247582809 by `expm1` and …807 by the textbook form, so 11.417 dB is
unchanged. Rejecting non-finite power is the right Q7 call, and
`test_rejects_power_that_is_not_finite` covers both paths.

## `tracking.py`

Now `core/tracking/kalman.py`; see [Status at 2026-10-08](#status-at-2026-10-08). The
`TrackManager` rows below describe what is now `KalmanTracker`, and `Track` with its status
names now lives in `tracks.py`.

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `StateModel`, `STATE_MODELS` | ✓ | ✓ | — | ✓ | ✓ | ✓ | — | — |
| `TrackStatus`, `TRACK_STATUSES` | ✓ | ✓ | — | ✓ | ✓ | ✓ | — | — |
| `KalmanState` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KalmanState.n_state` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TrackModel` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TrackModel.restricted` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `UpdateResult` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `FrameResult` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [F1](#f1--frameresult-was-returned-but-never-exported) |
| `Track` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Track.is_alive`, `Track.is_confirmed` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `process_noise_dwna` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `state_model_matrices` | ✓ | ✓ | ✓ | ✓ | [F2](#f2--three-docstrings-that-misdescribe-their-code) | ✓ | ✓ | ✓ |
| `predict` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `innovation_of` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `normalised_innovation_squared` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `gate_threshold` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `update` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `associate_gnn` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TrackManager` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TrackManager.lost_track_ids` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TrackManager.confirmed_tracks` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TrackManager.step` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [F1](#f1--frameresult-was-returned-but-never-exported) |

**D2 conformance.** Three state models are data returned by a factory, not a class hierarchy,
and there is one association strategy. The promotion trigger has not fired, and the module says
so and says what would fire it.

**Scenario 003 conformance.** Defaults match §7 and §5.4 exactly: `gate_probability = 0.99`,
confirm 4 of 5, delete after 3 consecutive misses, `sigma_azimuth_deg = 0.5`. `R` is
`diag(σ_R², σ_v²)`, and the five-order-of-magnitude gap between the two is deliberate — a
range-Doppler radar has a coarse range measurement and an exquisite velocity one — with the
docstring pointing out that this is what makes a transposed `R` obvious in the consistency
statistic rather than subtle.

**Three pieces of correctness worth confirming explicitly.** The `range_1d` transition carries
`−T`, not `+T`, because range rate is positive *closing* and closing reduces range; this is
obtained by conjugating the textbook constant-velocity pair with `diag(1, −1)`, which flips the
cross terms of `Q` in step and leaves `process_noise_dwna` in the form the references state. The
comment explains that getting it wrong dead-reckons the target the wrong way down the line of
sight while the range track still looks roughly right for a few frames — which is precisely why
it earns a comment. The ENU measurement Jacobian was re-derived by hand here and is right in
every row, including the elevation row `∂el/∂e = −u·e/(r²g)`, `∂el/∂u = g/r²`, and the
closing-rate row's `−(v/r − (p·v)p/r³)`. And the update uses the Joseph form, justified by the
same `R` conditioning, with an explicit symmetrisation to stop drift over a long track.

`_FORBIDDEN_COST` is a large finite number rather than `inf` because `scipy` rejects `inf` in a
cost matrix, and the comment says so — the kind of one-line note that saves the next reader a
confused half hour. The newer `core/tracking/association.py` uses `+inf` for a gated-out pair,
which looks like a second convention but is the layer above this one: its
`GlobalNearestNeighbour` passes the `+inf` costs to `associate_gnn`, which substitutes
`_FORBIDDEN_COST` only for the solver call. `+inf` at the interface, a finite stand-in at the
solver, is the right split and needs no reconciling.

The velocity-gate fallback in `step` (a pair that fails the full gate may still pass a
range-only one) and `_forget_rate` both carry their reasoning *and their measured alternative*:
the comment records that assigning confirmed tracks ahead of tentative ones was tried and was
worse on every count, 25.3 m range RMSE against 17.5 m and 82% fold selection against 86%. That
is the standard this audit asks for on Q3 claims, applied to a design claim.

---

## F1 — `FrameResult` was returned but never exported

`TrackManager.step` (now `KalmanTracker.step` in `core/tracking/kalman.py`) is documented as
returning `FrameResult`, and it does. But `tracking.__all__` did not name it and `radar_forge/core/__init__.py` did not import it, so the
type was absent from both `from radar_forge.core.tracking import *` and `radar_forge.core`. A
caller who wanted to annotate the result, or to write an `isinstance` check, had to reach past
the declared surface for a type the contract already promised.

`mypy` cannot see this — the annotation resolves fine inside the module — and no existing test
saw it either, because every one of them constructs the value and reads its attributes rather
than naming its type. It is the same class of defect as the accidental `radar_equation` export
fixed in the base commit: a public contract that is true by accident of how callers happen to
use it.

**Fixed**, with a regression test that generalises rather than pinning the one name.
`test_every_public_return_type_is_exported` walks every module in the package, takes its public
functions and the public methods of its exported classes, and requires any class named in a
return annotation and defined in that same module to appear in its `__all__`. Verified red
against the old code: it failed on exactly one name across the whole package, `FrameResult`.

## F2 — Three docstrings that misdescribe their code

None of these changes behaviour; all three are contract defects in the sense §1.2 means, because
the documented contract is what a caller relies on.

1. **`state_model_matrices`** listed `state_model : {'range_1d'}` in its parameter table while
   accepting all three of `STATE_MODELS`. The prose immediately above describes `enu_2d` and
   `enu_3d` at length, so the enumeration was left behind when they landed. A reader who trusts
   the parameter table concludes the ENU models are unreachable.
2. **`BistaticRadar.range_resolution_at_bistatic_angle_m`** wrote `\\sqrt`, `\\epsilon` and
   `\\pi` inside an `r"""` docstring, so the paragraph explaining the forward-scatter floor —
   the most subtle thing in the method — rendered with literal double backslashes. The rest of
   the same docstring uses single backslashes correctly.
3. **`PropagationPaths.amplitude_linear`** did not record that the generators take only
   `abs(amplitude_linear)`. They rebuild the propagation phase from `delay_s`, and they must:
   the phase has to advance with the delay across slow time to become Doppler, rather than be
   frozen at whatever one path set recorded. The path builders here do populate the phase, so a
   path set is internally consistent, but a future backend that wants to impose an extra phase —
   a complex reflection coefficient, say — cannot do it through this field. That is a gap in
   D1's contract rather than a bug in the generators, and it is now written where a backend
   author will read it.

**Fixed**, documentation only.

## F3 — The MTI canceller copied every tap

`mti_filter` gathered each tap band with `np.take(samples, range(offset, offset + n_output))`.
A `range` is a sequence, so that is fancy indexing: NumPy builds an index array and copies the
band before the multiply. Basic slicing gives a view instead, so only the products allocate. The
accumulator seed — a `zeros_like` of a further `np.take` — was a third copy, and the first
coefficient's term serves instead.

Output is **bit-identical**, checked with `np.array_equal` for both canceller orders, so there is
no behaviour to test under audit rule C1; the existing tests cover the result. Measured on this machine,
best of 5 repeats of 20 calls:

| Case | Before | After | Speedup |
| :--- | ---: | ---: | ---: |
| `(256, 1000)`, 2-pulse | 1.457 ms | 0.689 ms | 2.11× |
| `(256, 1000)`, 3-pulse | 1.993 ms | 1.175 ms | 1.70× |
| `(1024, 2048)`, 2-pulse | 15.228 ms | 7.762 ms | 1.96× |

`(256, 1000)` is scenario 001 S1's cube shape, so the first row is the case the library actually
runs.

## F4 — Taper names were a bare `str` where CFAR variants are a `Literal`

`detection.py` states its closed set of variant names as `CfarVariant = Literal[...]`, exports
it, and derives `CFAR_VARIANTS` from it with `get_args`. `windows.py` solved the identical
problem the other way: `taper(name: str, ...)` accepted any string, rejected the bad ones at run
time, and carried a hand-maintained `TAPER_NAMES` tuple that could drift from the set the
function actually implements.

Two modules solving one problem two ways is a Q6 defect, and the `str` signature is a Q7 one: a
misspelled taper survives `mypy --strict` and fails on the first run.

**Fixed.** `TaperName` is now the `Literal`, `TAPER_NAMES` is `get_args(TaperName)`, and the two
cannot diverge. No run-time change — the same seven names in the same order, and the membership
check that raises a descriptive `ValueError` is untouched, so a caller coming from a config file
still gets the good error rather than a type error it cannot act on.

## F5 — The scenario 001 noise bandwidth was stated as 1 MHz

This is the one divergence where the **spec** was the wrong party, handled under audit rule C4.

`spec/scenario-001-xband.md` §3.2 gave the thermal noise power as
"1.598e-14 W (kT₀·B·F at B = 1 MHz, F = 3 dB)". The *value* is right and `Radar.noise_power_w`
agrees with it to every digit; the parenthetical does not. At 1 MHz and 3 dB, kT₀BF is
7.989e-15 W. 1.598e-14 W is the 2 MHz figure — the transmitted bandwidth, which is what
`Radar.noise_power_w` uses as the noise bandwidth of a matched receiver. 1 MHz is S1's *sample
rate*.

It has to be the transmitted bandwidth for the row to belong in §3.2 at all: that section is the
waveform-*independent* derived quantities, and the three variants sample at 1.0, 2.5 and
4.0 MHz. A sample-rate-based noise power would differ between them.

**The spec is corrected and the code is unchanged.** The note left in place records what the row
used to say, so a reader who has the old number in their head can see what moved and what did
not.

## F6 — A 2-D CFAR pair was not checked to be two integers

`n_train`, `n_guard` and `axes` are each typed `tuple[int, int]`, but `_ring` checked only
the values, never the shape of the argument. The mistakes a caller could make went three ways.
Two of them were silent and wrong:

| Call | Before | After |
| :--- | :--- | :--- |
| `n_train=(2, 4, 6), n_guard=(1, 1, 1)` | Accepted. The third count was dropped | `ValueError` naming `n_train` |
| `n_train=(2.5, 4)` | Accepted. `int()` truncated 2.5 to 2 | `ValueError` naming `n_train` |
| `n_train=16` (the 1-D habit) | `TypeError: 'int' object is not iterable` | `ValueError` naming `n_train` |
| `n_guard=(1, 1, 1)` against a 2-entry `n_train` | `zip()`'s message, naming neither argument | `ValueError` naming `n_guard` |
| `axes=(-1,)` | "not enough values to unpack" | `ValueError` naming `axes` |

`mypy --strict` catches all five in typed code. A pair read from a TOML file or built in a
notebook is not typed code, and the first two return a detection map that looks plausible. The
1-D sibling has no such hazard: a single `int` has nothing to truncate, and a float `n_train`
fails there on slicing.

**Fixed.** `_integer_pair` requires exactly two entries that pass `operator.index`, so NumPy
integers are accepted and floats are refused, and its error names the argument. The `Raises`
sections of `cfar_valid_mask_2d` and `cfar_noise_estimate_2d_w` say so.
`test_ring_rejects_a_pair_that_is_not_two_integers` covers all five rows through both
`cfar_noise_estimate_2d_w` and `cfar_valid_mask_2d`. Verified red: with the source change
stashed, all five cases fail. Two fail with "DID NOT RAISE", one with the `TypeError`, and two
with a non-matching message. `test_ring_accepts_numpy_integer_counts` guards the other side, and
passes before and after.

## F7 — The 2-D variants were an inline `Literal` and a private tuple

This is F4's defect in the module F4 used as its model. The 1-D functions take `CfarVariant`,
from which `CFAR_VARIANTS` is derived with `get_args`. The three 2-D functions each wrote
`Literal["ca", "os"]` inline. The run-time check used `_CFAR_2D_VARIANTS = ("ca", "os")`, a
hand-maintained private tuple, so the static set and the checked set could drift. A caller had
no exported name to check a configured variant against.

**Fixed.** `CfarVariant2d` is the `Literal`, `CFAR_VARIANTS_2D` is `get_args` of it, and both
are exported from `detection` and `radar_forge.core` beside their 1-D counterparts. There is no
run-time change: the same two names, the same order, and the same error message.
`test_the_ring_variants_are_the_one_dimensional_ones_without_half_windows` pins the relation that
justifies the subset: the ring borrows the 1-D calibration, so its variants must be among the 1-D
ones.

## F8 — The calibration's independence assumption did not name tapering

Every closed form in the module assumes the reference cells are independent, and the module
says "i.i.d.". Nothing said that the commonest step in a radar chain breaks it. A taper
correlates neighbouring bins. For Hann, the complex amplitudes of adjacent bins have a
correlation of magnitude `|Σw²e^{-j2πn/N}| / Σw²`, which computes to 0.675 at N = 64 and tends
to 2/3. Zero-padding the FFT does the same, and so does sampling a compressed pulse faster than
its bandwidth. Correlated cells give a noisier estimate than M independent ones, so the threshold
factor is too low.

Measured with the library's own chain: complex white noise through `range_doppler_map`, Hann on
both axes, `(64, 256)` maps, 30 maps per seed, `pfa = 1e-3`:

| Seed | 1-D window, `n_train = 16`, `n_guard = 4` | Ring, `(2, 4)`, `(1, 2)`, Doppler wrapping |
| :--- | ---: | ---: |
| 0 | 1.67 × pfa | 1.72 × pfa |
| 1 | 1.62 × pfa | 1.58 × pfa |
| 2 | 1.70 × pfa | 1.74 × pfa |
| Untapered, for contrast | 1.01 × pfa | 0.97 × pfa |

So a student who tapers, then measures, sees 60–70% too many false alarms and concludes that
the detector is broken. The ring is no worse than the window, even though it draws from both
correlated axes.

None of the scenarios applies a taper, so the FMCW ones are unaffected: an untapered range FFT of
deramped noise gives independent bins. **The pulsed one is affected.** S2 samples its 2 MHz
chirp at 2.5 MHz, so after `matched_filter` neighbouring range cells correlate with magnitude
0.27, and 0.20 at two cells apart. That is S2's chain as `form_range_doppler_map` builds it,
measured with the 1-D CFAR that `scenario_003_ukf_pulsed_medium_prf.toml` configures:

| `pfa` | Cells tested | Measured / design |
| :--- | ---: | ---: |
| 1e-3 | 921 600 | 1.17 |
| 1e-4 | 6 144 000 | 1.29 |

The excess grows as `pfa` falls. At that TOML's design `1e-5` it is larger still, though not
measured here. No acceptance test measures that run's false-alarm rate, so nothing fails. But
its `pfa` is not the rate it gets. Repairing that is a pipelines decision, not a `core/` one. It
could calibrate against an effective number of independent cells, or decimate to the bandwidth
before CFAR. It is recorded here and left to that stream.

**Fixed, documentation only.** The module Notes now name the mechanism, its direction and the
remedy. `cfar_threshold_2d_w`, whose Notes make the i.i.d. argument for the ring, points to them.
`test_a_tapered_map_breaks_the_calibration_as_the_module_notes_say` pins the direction for both
the window and the ring: the lower end of the 99.9% interval must clear `pfa`. There is no
behaviour to verify red. The test pins a property of the noise, not of the code.

## F9 — Scenario 003 said a 2-D window halves the cells needed

Corrected under audit rule C4, because the **spec** was the wrong party. §13.2 said a 2-D reference
window "would also roughly halve the number of training cells needed for the same `pfa`".
Its own "Landed" paragraph says, correctly, that the `pfa` depends only on the number of
reference cells M. So the same `pfa` at the same CFAR loss needs the same M in any shape. What
a ring changes is the reach. It draws M cells from both axes, so it extends fewer cells along
range for the same M. For the same reach it has more cells and a lower threshold factor, which
is what `test_a_ring_has_less_cfar_loss_than_a_line_of_the_same_reach` asserts.

The same section said the 1-D functions were "unchanged". Their calibration is, but PR #4 changed
two things on the 1-D path, both re-checked above.

**The spec is corrected and the code is unchanged.** A note in §13.2 records the old wording.

## F10 — data-001 called a thermal noise power the CFAR estimate

Corrected under audit rule C4. Data-001 §6.8 lists `rd/burst{k}/noise_power_w`, of shape
`(n_frames,)`, as "the noise estimate used by CFAR". That contradicts the vocabulary PR #4
renamed `Detection` to protect. `noise_power_w` is the thermal noise power in `Radar`, in the
same spec's `bursts[].noise_power_w`, and as the denominator of its own `snr_db` in §6.5. It also
contradicts the shape. This library's CFAR estimates the floor per cell, so no per-frame scalar
can be "the estimate used by CFAR". A reader who took it at its word would compute an SNR
against the wrong quantity, and would find no per-cell estimate anywhere in the file.

**The spec is corrected.** The dataset is now the burst's thermal noise power, and a new bullet
says how to recover the per-cell estimate: `threshold_w / α`. It also records the old wording.
No `products.h5` writer exists yet (§1 lists what is written), so no file and no code moves, and
the schema version stays where it is.

## Recommended, not applied

Each of these is a real observation. When the audit was written, none was unambiguous enough to
justify an API break or the churn, so they were recorded rather than acted on. R1 and R3 were
later taken deliberately; their entries say so.

**R1 — `received_power_w` takes six adjacent positional `ArrayLike` arguments.** Swapping
`gain_tx_linear` and `gain_rx_linear` is harmless (they multiply), but swapping `wavelength_m`
and `rcs_m2` is silent and wrong. Making the trailing arguments keyword-only would remove the
hazard. Not applied: this module is the repository's declared style exemplar, its own doctests
call positionally, and the change would break every positional call site in and outside the
repository. If it is taken, it should be taken deliberately and across the pair.
**Applied** later, across the pair: every argument of both functions is now keyword-only. The
only positional callers in the repository were the two doctests.

**R2 — `range_from_beat_frequency_m`'s first parameter is named `beat_frequency_hz`, shadowing
the sibling function of that name.** Inside the body, `beat_frequency_hz` is the array, so the
forward function cannot be called there. The name is otherwise exactly right, which is the
difficulty. `measured_beat_frequency_hz` would resolve it. Not applied: renaming a
keyword-accessible parameter is an API break, and the shadowing costs nothing today.

**R3 — `thermal_noise(shape, noise_power_w, rng)` takes `rng` positionally**, where both
generators in the same module take it keyword-only. Making it keyword-only would be consistent
and is the direction `docs/conventions/style.md` §7 points. Not applied: it is an API break for
a function with existing positional callers, and the current signature is not wrong, only
inconsistent. **Applied** later: `thermal_noise(shape, noise_power_w, *, rng)`.

**R4 — `_normalize_axis` is written three times**, in `windows.py`, `dsp.py` and
`detection.py`, with two different error-message wordings. Consolidating into one private helper
would remove ten lines. Not applied: the two wordings differ, so consolidating changes an
observable error message in at least one module for no functional gain, and the duplication is
five lines of the most obvious code in the package. Recorded so that a fourth copy is a
deliberate choice.

**R5 — `wrap_axes=0` fails with "'int' object is not iterable".** F6's check covers the
three pairs but not `wrap_axes`, which is a variable-length tuple in the four 2-D functions and
in `cluster_detections`. Unlike F6's first two rows, this mistake is loud rather than silent:
it raises at once, but with Python's message rather than one that names the argument. A shared
check would be four lines. Not applied: it changes an exception type, from `TypeError` to
`ValueError`, in a function the first pass already audited (`cluster_detections`), and nothing
wrong is ever computed.

**R6 — A caller who wants both the detections and their noise estimates computes the ring
twice.** `cluster_detections(..., noise_estimate_w=...)` needs the estimate, which
`cfar_detect_2d` computes internally and discards, so the caller must call
`cfar_noise_estimate_2d_w` again. On S1's `(256, 1000)` map with a `(2, 4)`, `(1, 2)` ring,
measured best of 5:

| Variant | `cfar_detect_2d` | Plus `cfar_noise_estimate_2d_w` |
| :--- | ---: | ---: |
| CA | 3.3 ms | 7.1 ms |
| OS | 204 ms | 396 ms |

A form that returns the estimate with the mask, or that accepts a precomputed one, would halve
the OS case. Not applied: it is new API, and it would need the same shape on the 1-D pair,
which has the same double cost. Nothing in the repository calls the 2-D path yet. Scenario 003
§13.2 leaves the switch to its pipeline, and that switch is the time to choose a shape from a
real caller.

**R7 — Three entries in the module's reference list have no citation marker.** Finn &
Johnson [1] is cited by `cfar_noise_estimate_2d_w`, and Rohling [4] and Gandhi & Kassam [5] by
the 1-D functions. Hansen & Sawyers [2] and Weiss [3] are attributed only by the parentheticals
in the list itself. Richards [6] has been cited by nothing since PR #4 removed both of its
markers, in the ring paragraph and in `cfar_noise_estimate_2d_w`'s references.
`docs/conventions/style.md` §4.1 says an entry nothing cites is removed. Not applied: whether
to cite Richards §6.5 again or drop it is a citation decision. Under §4.1 that needs the source
in hand, and this pass did not have it. PR #11's verification of the record itself still
stands.

## What was deliberately left alone

- **`cfar_threshold_factor`'s bisection.** 200 iterations of a function that itself sums a
  series looks expensive, but the bracket is monotone, the alternative needs scaling care over
  many decades, and the function is called once per map rather than once per cell. No
  measurement suggested a problem, so there is no speed claim and no change.
- **`matched_filter` recomputing the reference transform on every call.** Caching it would help
  a fixed waveform against many cubes, at the cost of a stateful API. The docstring already
  names the trade and says the library prefers the readable version until a profile says
  otherwise. That is the right call for a teaching library, and nothing here profiled it.
- **`enu_to_range_azimuth_elevation`'s unguarded singularities.** Zero range and the overhead
  azimuth are left to produce NaN. Guarding them would substitute a value for a modelling error
  and hide it; the docstring says exactly that, and `BistaticRadar.target_ranges_m` already
  routes around the one legitimate case.
- **`PointTarget`'s aspect-independent cross-section.** An approximation, stated as one, and D8
  depends on it.
- **The `range_fft` / `doppler_fft` shift asymmetry.** It surprises every new reader and it is
  correct; the fix is the documentation it already has.
- **The 2-D OS cost.** `rank_filter` is linear in the ring's size, at 72 ms for M = 26 and
  1.9 s for M = 816 on S1's map. Both alternatives that were measured, the predecessor's
  chunked gather and a sliding-window `np.partition`, are no faster (see
  [`detection.py` — 2-D CFAR](#detectionpy--2-d-cfar)). Sub-linear running-rank algorithms
  exist, but none is in NumPy or SciPy. The docstring already says "substantially slower".
- **`n_guard` has a default in 1-D and none in 2-D.** The asymmetry is the right way round.
  The 1-D default of one guard cell is documented against the range mainlobe. A 2-D default
  would have to guess two mainlobes, one per axis, and a wrong guess along Doppler masks the
  target silently. Removing the 1-D default would break every call that relies on it.
- **The box-filter roundoff.** It is measured above at 1e-11 for scenario 003's strongest
  return, and it is smaller than the 1-D cumulative sum's. An exact summed-area table would
  buy digits nobody can use.
- **The absolute bistatic delay.** `bistatic_line_of_sight_paths` carries `(R_t + R_r)/c` rather
  than the range sum minus the baseline that a synchronised receiver measures. The docstring
  explains that the absolute delay is what carries the correct carrier phase and that baseline
  subtraction belongs to a synchronisation model the library does not yet have. Agreed.
