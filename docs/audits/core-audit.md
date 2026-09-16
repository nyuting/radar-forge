# `core/` audit — refactor-002 §1.2 and §1.3

Scope: every public name in `src/radar_forge/core/`. `pipelines/`, `teaching/` and `scripts/`
are a separate stream and are untouched here, as are `docs/conventions/style.md` and
`scripts/check_conventions.py`.

The audit has two halves, per `spec/refactor-002-spec-first-audit.md` §1.2 as reframed by its
§4.1. **Conformance**: does the code compute what the specs say, to the accuracy they claim,
with the contract they document, under `structure.md`'s decisions D1–D8? **Quality**: is this
the best available form, across all eight Q dimensions, whether or not conformance holds?

There are no prototype implementations in `spec/` to diff against — one 8-line dataclass
interface sketch and nothing else — so "reference" throughout means the *specified behaviour*:
the governing equation as written or cited, the numerical bounds the specs fix, the documented
array shapes, frames, signs and units, and D1–D8.

## Verdict key

| Mark | Meaning |
| :--- | :--- |
| ✓ | Sound. Nothing better available at reasonable cost; no action. |
| **F*n*** | A finding. Links to the section that records it. Fixed unless the section says otherwise. |
| **R*n*** | A recommendation, deliberately **not** applied — see [Recommended, not applied](#recommended-not-applied). |

Q1 mathematical and radar-physics accuracy · Q2 clarity · Q3 speed and vectorisation ·
Q4 conciseness · Q5 documentation · Q6 naming · Q7 inputs · Q8 outputs.

## Findings

| # | Where | Dimension | Verdict |
| :--- | :--- | :--- | :--- |
| [F1](#f1-frameresult-was-returned-but-never-exported) | `tracking.TrackManager.step` | Q8 | Code wrong — fixed |
| [F2](#f2-three-docstrings-that-misdescribe-their-code) | `tracking.state_model_matrices`, `radar.BistaticRadar.range_resolution_at_bistatic_angle_m`, `signal.PropagationPaths` | Q5 | Docs wrong — fixed |
| [F3](#f3-the-mti-canceller-copied-every-tap) | `dsp.mti_filter` | Q3, Q4 | Suboptimal — fixed, measured |
| [F4](#f4-taper-names-were-a-bare-str-where-cfar-variants-are-a-literal) | `windows.taper` | Q6, Q7 | Inconsistent — fixed |
| [F5](#f5-the-scenario-001-noise-bandwidth-was-stated-as-1-mhz) | `spec/scenario-001-xband.md` §3.2 | Conformance | **Spec** wrong — fixed |

Everything else audited below carries no finding. That is the expected result: the suite is
strong, refactor-001 hardened it, and most of `core/` is already the best available form.

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

Both match Richards eq. 2.11 and Willis §2.2 term for term, including the `(4π)³` hoisted to a
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
| `TaperName`, `TAPER_NAMES` | ✓ | ✓ | ✓ | [F4](#f4-taper-names-were-a-bare-str-where-cfar-variants-are-a-literal) | ✓ | [F4](#f4-taper-names-were-a-bare-str-where-cfar-variants-are-a-literal) | — | — |
| `taper` | ✓ | ✓ | ✓ | ✓ | ✓ | [F4](#f4-taper-names-were-a-bare-str-where-cfar-variants-are-a-literal) | [F4](#f4-taper-names-were-a-bare-str-where-cfar-variants-are-a-literal) | ✓ |
| `apply_taper` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `coherent_gain_linear` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `processing_loss_db` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

The window shapes come from `scipy.signal.windows` rather than being reimplemented, which is
both the right dependency call and R1.3.3-clean. Two details are better than the usual textbook
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
| `Radar.noise_power_w` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [F5](#f5-the-scenario-001-noise-bandwidth-was-stated-as-1-mhz) |
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
| `BistaticRadar.range_resolution_at_bistatic_angle_m` | ✓ | ✓ | ✓ | ✓ | [F2](#f2-three-docstrings-that-misdescribe-their-code) | ✓ | ✓ | ✓ |
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
| `PropagationPaths` | ✓ | ✓ | ✓ | ✓ | [F2](#f2-three-docstrings-that-misdescribe-their-code) | ✓ | ✓ | ✓ |
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
at a quarter of a range bin of intra-chirp motion; scenario 001 §5 says range migration must be
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
| `mti_filter` | ✓ | ✓ | [F3](#f3-the-mti-canceller-copied-every-tap) | [F3](#f3-the-mti-canceller-copied-every-tap) | ✓ | ✓ | ✓ | ✓ |

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

The four closed-form `P_fa(α)` expressions match their cited sources — Gandhi & Kassam eqs.
12–13 for CA/GO/SO, Rohling eq. 8 for OS — and the module's own Notes pin down two cross-variant
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

## `tracking.py`

| Name | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `StateModel`, `STATE_MODELS` | ✓ | ✓ | — | ✓ | ✓ | ✓ | — | — |
| `TrackStatus`, `TRACK_STATUSES` | ✓ | ✓ | — | ✓ | ✓ | ✓ | — | — |
| `KalmanState` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KalmanState.n_state` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TrackModel` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TrackModel.restricted` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `UpdateResult` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `FrameResult` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [F1](#f1-frameresult-was-returned-but-never-exported) |
| `Track` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Track.is_alive`, `Track.is_confirmed` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `process_noise_dwna` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `state_model_matrices` | ✓ | ✓ | ✓ | ✓ | [F2](#f2-three-docstrings-that-misdescribe-their-code) | ✓ | ✓ | ✓ |
| `predict` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `innovation_of` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `normalised_innovation_squared` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `gate_threshold` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `update` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `associate_gnn` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TrackManager` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TrackManager.lost_track_ids` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TrackManager.confirmed_tracks` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TrackManager.step` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [F1](#f1-frameresult-was-returned-but-never-exported) |

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
confused half hour.

The velocity-gate fallback in `step` (a pair that fails the full gate may still pass a
range-only one) and `_forget_rate` both carry their reasoning *and their measured alternative*:
the comment records that assigning confirmed tracks ahead of tentative ones was tried and was
worse on every count, 25.3 m range RMSE against 17.5 m and 82% fold selection against 86%. That
is the standard this audit asks for on Q3 claims, applied to a design claim.

---

## F1 — `FrameResult` was returned but never exported

`TrackManager.step` is documented as returning `FrameResult`, and it does. But
`tracking.__all__` did not name it and `radar_forge/core/__init__.py` did not import it, so the
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
no behaviour to test under R1.3.1; the existing tests cover the result. Measured on this machine,
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

This is the one divergence where the **spec** was the wrong party, handled under R1.3.4.

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

## Recommended, not applied

Each of these is a real observation. None is unambiguous enough to justify an API break or the
churn, so they are recorded rather than acted on.

**R1 — `received_power_w` takes six adjacent positional `ArrayLike` arguments.** Swapping
`gain_tx_linear` and `gain_rx_linear` is harmless (they multiply), but swapping `wavelength_m`
and `rcs_m2` is silent and wrong. Making the trailing arguments keyword-only would remove the
hazard. Not applied: this module is the repository's declared style exemplar, its own doctests
call positionally, and the change would break every positional call site in and outside the
repository. If it is taken, it should be taken deliberately and across the pair.

**R2 — `range_from_beat_frequency_m`'s first parameter is named `beat_frequency_hz`, shadowing
the sibling function of that name.** Inside the body, `beat_frequency_hz` is the array, so the
forward function cannot be called there. The name is otherwise exactly right, which is the
difficulty. `measured_beat_frequency_hz` would resolve it. Not applied: renaming a
keyword-accessible parameter is an API break, and the shadowing costs nothing today.

**R3 — `thermal_noise(shape, noise_power_w, rng)` takes `rng` positionally**, where both
generators in the same module take it keyword-only. Making it keyword-only would be consistent
and is the direction `docs/conventions/style.md` §7 points. Not applied: it is an API break for
a function with existing positional callers, and the current signature is not wrong, only
inconsistent.

**R4 — `_normalize_axis` is written three times**, in `windows.py`, `dsp.py` and
`detection.py`, with two different error-message wordings. Consolidating into one private helper
would remove ten lines. Not applied: the two wordings differ, so consolidating changes an
observable error message in at least one module for no functional gain, and the duplication is
five lines of the most obvious code in the package. Recorded so that a fourth copy is a
deliberate choice.

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
- **The absolute bistatic delay.** `bistatic_line_of_sight_paths` carries `(R_t + R_r)/c` rather
  than the range sum minus the baseline that a synchronised receiver measures. The docstring
  explains that the absolute delay is what carries the correct carrier phase and that baseline
  subtraction belongs to a synchronisation model the library does not yet have. Agreed.
