# Scenario 011 — Dense terminal-area traffic: crossings, clutter, manoeuvres and closely spaced aircraft (IMM, JPDA, MHT, PMBM)

## 1. Status & Purpose

**Status: specified, not implemented.** The first multi-target tracking scenario.

Scenario 003 tracks one aircraft against thermal false alarms, with GNN. Its §13.3 records why that
is enough there and nowhere else: with one target, gates never overlap, assignment is trivially
optimal and a track swap is impossible. `spec/tracker-001.md` leaves IMM (§9) and JPDA (§7.4) as
extension points "until the first manoeuvring / multi-target scenario", and scenario 003 §13.4
names "a multi-target, high-clutter scenario" as the trigger for random-finite-set tracking. This
document is that scenario.

The setting is civil air traffic in the terminal area around Raleigh-Durham, seen from the Duke
Receiver (the site of 001, 002 and 006). An airport-surveillance-class radar at the site reports
plots once per rotation. The traffic is arrivals on two parallel finals, arrivals turning onto
final, a go-around, a holding stack, two overflights, two light aircraft in a circuit, an
air-ambulance helicopter and a flock of birds. Each stress the trackers must survive has a named
cause in that traffic (§2).

Three teaching points, each a test in §6:

1. **Manoeuvres break a single motion model, and IMM repairs it.** A standard-rate turn at
   72 m/s is 3.77 m/s² of centripetal acceleration. Over three 4.8 s scans the aircraft leaves a
   constant-velocity prediction by 391 m, six to eight innovation standard deviations (§3.2). A CV filter
   tuned quiet loses the track; one tuned loose is noisy on every straight leg. IMM runs both and
   switches by mode probability (D27).
2. **Clutter and closely spaced aircraft break hard assignment, and the fixes fail in
   instructive ways.** GNN swaps tracks at a shallow crossing. JPDA shares measurements and does
   not swap, but on two aircraft in a holding stack it pulls the two estimates together
   (coalescence). Set-JPDA and MHT each fix that, at different costs (D28, D29).
3. **The separation at which each tracker stops resolving two aircraft is measured, not
   asserted.** V3 sweeps separation in units of the radar's resolution cell and clutter density
   in expected plots per gate. Each tracker gets a resolution curve, and PMBM, the
   random-finite-set tracker, is the reference line (D30).

It also fires two promotions that earlier specs left waiting. `Track.existence_probability`
gets its first decision, JIPDA's confirm and delete (`tracker-001.md` §8.2). And the plot level
of simulation, which 003 §13.3 said a multi-target scenario would need before it could measure
the right things, gets a spec (D25).

| Section | What it fixes |
| :--- | :--- |
| §2 Overview & Geometry | The sensor, the scene, the resolution cell in metres, the crossings, the separation sweep |
| §3 Parameter Matrix | Sensor and plots, clutter, traffic, trackers and their settings, variants, contracts, outputs |
| §4 Core Decisions | D24–D30; what moved in `structure.md` and `tracker-001.md`; the workstream boundary |
| §5 Build Sequence | What is new, what is widened, what is untouched |
| §6 Acceptance Criteria | Analytic reductions first, then statistical consistency, then the trackers compared |
| §7 Limitations | What this scenario does not model, stated rather than discovered |

### 1.1 Upstream inspiration

| Upstream | What it is | What radar-forge takes | Licence | Baseline or current |
| :--- | :--- | :--- | :--- | :--- |
| GNN; Crouse [17] | One-to-one assignment by `linear_sum_assignment` | Scenario 003's associator, unchanged; the baseline every row below is compared with | existing code | Baseline |
| PDA [1]; JPDA [2] | Each track updated with a weighted mixture of the measurements in its gate | `JointProbabilisticDataAssociation`; shown coalescing on the holding stack | paper, written from equations | Baseline |
| JPDA\* [3]; Set-JPDA [4] | JPDA with the coalescing hypotheses pruned (JPDA\*), or with the posterior approximated over unlabelled sets (Set-JPDA) | `coalescence="none" \| "jpda_star" \| "set"`; the fix, and its cost in track identity | paper, written from equations | Current |
| JIPDA [5] | JPDA with a track-existence probability propagated and used for confirm and delete | `existence_probability` on `Track`; the lifecycle rule it drives | paper, written from equations | **Current** (single-scan) |
| IMM [6]; IMM-JPDA [3]; manoeuvre models [7] | A bank of motion models mixed by Markov mode probability | `IMM` as an `Estimator` of `Estimator`s; `CoordinatedTurn` with the turn rate in the state | paper, written from equations | **Current** (manoeuvring) |
| Track-oriented MHT [8], [9], [10]; ranked assignment [11], [12] | Deferred association: hypothesis trees, N-scan pruning, k-best global hypotheses | `TrackOrientedMHT`, `TrackTree`, `murty_k_best` | paper, written from equations | **Current** (multi-scan) |
| PMBM [13]; marginal MB view of MHT and JIPDA [14]; GLMB [15] | Random-finite-set conjugate prior over sets of targets | `PmbmTracker`, beside `Tracker`, as the reference score | paper, written from equations | Comparison |
| Finite resolution [18], [19], [20] | Two targets inside one resolution cell give one merged plot | The merged-plot model of the plot simulator (D25) | paper, written from equations | Ground truth |
| GOSPA [21]; trajectory GOSPA [22]; OSPA [23] | Multi-target distance with localisation, missed and false terms; a trajectory version that charges track switches | `gospa`, `trajectory_gospa` in `core/tracking/metrics.py` | paper, written from equations | Metric |

---

## 2. Overview & Geometry

The sensor is a rotating airport-surveillance-class radar at the Duke Receiver: S-band, 4.8 s per
rotation (12.5 rpm), a 1.4° azimuth beam and a 115.75 m range cell (1/16 nautical mile). Those are
the parameters of the class (Skolnik [24], the air-traffic-control radar chapter), not of a
particular installation. It reports **plots**, once per rotation per resolved echo: horizontal
range and azimuth, with no range rate (§3.1). Raleigh-Durham's airport reference point is
19.601 km from the site at bearing 135.0° (computed from the 002 coordinates through
`core/geodesy.py`).

```
   N (km)
   16 |               B  bird flock (6, 15), 12 m/s towards 200 deg
   12 |  L  GA circuit (-9, 12), L1 and L2, rate-2 turns
   10 |  T1 ===========================================> 090 deg    H  holding fix (25, 10)
      |                                                    H1-H4 racetrack, inbound 230 deg
    4 |                                       R  rain cell (20, 4), r = 4 km, drifts to 045 deg
    0 +--D  Duke Receiver ----------------------------------------- A1, A2 start 22 / 20 km out
      |   HE helicopter (1.6, 0.8): hover, then 120 deg               /  finals 23R / 23L, 1 km
      |                                                              /   apart, 230 deg, 72 m/s
   -8 |                                     T2 crosses A1 at (20.5, -7.6), t = 200 s, 30 deg
      |                                            A4, A5 downwind -> base -> final, 3 deg/s
  -14 |                         *  RDU (13.9, -13.9); A3 goes around, turns left to 140 deg
      +-----------------------------------------------------------------------------> E (km)
```

The traffic is chosen so that every stress has a cause:

| Stress | Cause in the scene | What it exercises |
| :--- | :--- | :--- |
| Manoeuvre onset | A4 and A5 turning base and final at 3 °/s; A3's go-around; the stack's 180° turns every 60 s; L1, L2's rate-2 circuit | IMM mode switching (D26, D27) |
| Longitudinal acceleration | A3 accelerating 1 m/s² in the go-around; HE from hover to 60 m/s at 2 m/s² | The agile CV mode |
| Closely spaced, long-lived | H1–H2 (4 s apart in the stack) and H3–H4 (8 s apart) | Coalescence; merged plots; MHT's deferred decision |
| Crossing, shallow | T2 over A1, 30° | GNN swap; JPDA's sharing |
| Crossing, wide | T1 over H2, 140° | The easy crossing, as a control |
| Doppler-blind interval | HE hovering; L1 and L2 tangential on their circuit; A1 and A2 tangential near RDU | Missed plots: Pd drops in the MTI notch (§3.1) |
| Many unresolved real objects | Twelve birds in about 0.4 × 0.1 resolution cells | Structured clutter that is not false: track discipline |
| Dense, nonhomogeneous clutter | Ground-clutter residue inside 6 km; a rain cell drifting across the stack; 25 fixed discretes | Clutter-map estimation (D28); JIPDA existence |

**Altitude is not in the measurement.** The plots report horizontal range, the scene's aircraft are
tracked in the ground plane, and the holding stack's 300 m vertical spacing is invisible to the
radar. That is what makes a holding stack closely spaced in the first place, and it is what a
two-dimensional surveillance radar sees (§7).

### 2.1 The resolution cell, in metres

Two echoes are unresolved when they share a range cell and a beam. The range cell is the same
115.75 m everywhere. The beam's cross-range width grows with range:

| Range | 5 km | 10 km | 20 km | 26.9 km (the fix) | 30 km | 40 km |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.4° beam, cross-range | 122.2 m | 244.3 m | 488.7 m | 657.9 m | 733.0 m | 977.4 m |
| σ_az = 0.15°, cross-range | 13.09 m | 26.18 m | 52.36 m | 70.5 m | 78.54 m | 104.72 m |

Separations in this spec are given in **resolution units**: the normalised distance

```
   d = sqrt( (Δr / α_r)² + (Δaz / α_az)² ),     α_r = 115.75 m,  α_az = 1.4° = 0.024435 rad
```

and the probability that a pair at distance `d` gives one merged plot is
`P_u(d) = exp(−d²/2)` (Koch and van Keuk's Gaussian resolution model [18]; §3.6). At `d = 1`,
`P_u = 0.607`; at `d = 2`, 0.135; at `d = 3`, 0.011.

The scene's close encounters, from the closed-form trajectories of D24 at 0.1 s steps over the
600 s run (scratch computation; the numbers are the scenario's, the script is not code):

| Pair | Minimum `d` | At | Ground separation then | Track angle | Scans with `P_u > 0.5` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| H1–H2 (stack, 4 s apart) | 0.52 | 151.7 s | 439 m | 12° | **10** |
| H3–H4 (stack, 8 s apart) | 1.03 | 29.7 s | 874 m | 24° | 3 |
| A1–T2 (shallow crossing) | 0.00 | 200.0 s | 0 m | 30° | 1 |
| H2–T1 (wide crossing) | 0.00 | 236.0 s | 0 m | 140° | 0 |
| A2–A3 (same final, in trail) | 2.36 | 244.4 s | 815 m | 0° | 0 |

The parallel finals are 1 km apart and are **not** closely spaced from this site. The line of
sight from Duke meets the final course at about 95° near the airport, so the 1 km lateral offset
lies mostly along range: 5–9 range cells. They are ordinary busy traffic. The holding stack is
where aircraft overlap, because in the ground plane its levels share one racetrack.

### 2.2 Crossings, in scans

At a crossing angle θ between equal speeds `v`, the relative speed is `2 v sin(θ/2)`. The time
inside one resolution unit depends on which way the pair close:

| θ at 72 m/s | Relative speed | Inside `\|Δr\| < α_r` (closing along range) | Inside `\|Δaz\| < α_az` at 20 km (closing across range) |
| :--- | ---: | ---: | ---: |
| 90° | 101.8 m/s | 2.27 s = 0.47 scans | 9.6 s = 2.0 scans |
| 30° | 37.27 m/s | 6.21 s = 1.3 scans | 26.2 s = 5.5 scans |
| 10° | 12.55 m/s | 18.45 s = 3.8 scans | 77.9 s = **16.2 scans** |

A 10° crossing closing across the beam spends sixteen scans unresolved. That is the case GNN
swaps on, JPDA coalesces on, and MHT is built for. V2 runs all three angles in both orientations.

### 2.3 The separation sweep

V3 places pairs **in radar coordinates**, as 006 §4.3 did: on parallel courses, the same velocity,
offset by a stated `d` along a stated direction in the normalised (range, azimuth) plane. This
removes the geometry from the question, which is then only "at what `d` and what clutter density
does each tracker keep two tracks on two aircraft". The sweep is `d ∈ {0.25, 0.5, 1, 2, 4}` by
`n_gate ∈ {0.1, 1, 3}` expected clutter plots per gate (§3.2), each in both orientations.

---

## 3. Parameter Matrix

### 3.1 Sensor and plots

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Site | Duke Receiver, 36.00250 °N, 78.94100 °W, 60 m | Scenario-001 §3.1, unchanged |
| `f0_hz` | 2.8e9 (λ = 0.10707 m) | S-band airport-surveillance class [24] |
| `scan_period_s` | **4.8** (12.5 rpm); V1 also 1.2 and 2.4 | [24]; the revisit sweep of V1 |
| `beamwidth_azimuth_deg` | 1.4 | [24] |
| `range_cell_m` | 115.75 (1/16 nautical mile) | [24] |
| Coverage | 1–40 km, 360° | Terminal area |
| Measurement | `z = [range_m, azimuth_rad]`, horizontal range | §2, §7 |
| `sigma_range_m` | **33.414** = 115.75/√12 | Quantisation-limited, as 003 §5.4 |
| `sigma_azimuth_deg` | **0.15** | Plot-extraction and calibration budget, as 003 §6.3 sets 0.5° for its 5.1° beam; thermal alone gives θ₃/(1.6·√(2·SNR)) = 0.062° at 20 dB |
| `pd` | 0.90 per aircraft echo; 0.30 per bird | Specified |
| `mti_notch_mps`, `pd_notch` | 4.0, 0.30: an echo with `\|ṙ\| < 4 m/s` is detected with `pd_notch` | Moving-target-indicator notch [24] |
| Plot time | The time the beam centre crossed the echo's azimuth | D25 item 4 |

Fraction of the run each aircraft spends in the notch (`|ṙ| < 4 m/s`): HE 0.156, L1 0.102, L2 0.095,
A4 0.053, A2 0.052, A1 0.049, all others below 0.012.

### 3.2 Clutter

Clutter plots are Poisson in the measurement space, with intensity in plots per (m·rad) per scan.
The CFAR behind a real extractor holds the false-alarm rate per resolution cell, which is uniform
in range and azimuth, so the intensity is specified in (m·rad), not per square metre.

Intensities are set by **expected clutter plots per gate**, `n_gate`, at a reference gate: the
99 % chi-square gate (γ = 9.2103 for two dimensions) of a steady-state quiet-CV filter
(`acceleration_noise_density_m2ps3 = 0.5`, T = 4.8 s) on an aircraft at 20 km. Its innovation
standard deviations are 46.6 m in range and 68.3 m across, so its area is
`π γ sqrt(|S|)` = 9.225e4 m² = **4.612 m·rad**. Gates at other ranges, revisit times and process
noise differ, and §3.6 states the formula for each.

| `n_gate` | Intensity (plots / m·rad / scan) | Plots per scan over the whole 1–40 km coverage (245 044 m·rad) |
| :--- | ---: | ---: |
| 0.01 | 0.0021683 | 531 |
| 0.1 | 0.021683 | 5 313 |
| 0.3 | 0.065048 | 15 939 |
| 1 | 0.21683 | 53 128 |
| 3 | 0.65048 | 159 384 |

Those coverage-wide totals are why V2 and V3 run in a **local window**, range ±2.5 km and azimuth
±5° about the pair (872.7 m·rad at 20 km: 19, 189 and 568 clutter plots per scan at
`n_gate` = 0.1, 1, 3). V4 uses the whole coverage with a light background and heavy patches:

| V4 component | Region | `n_gate` | Plots per scan |
| :--- | :--- | ---: | ---: |
| Background | Whole coverage | 0.01 | 531 |
| Ground-clutter residue | Range 1–6 km, all azimuths | 0.3 | 2 044 |
| Rain cell | Disc of radius 4 km, centre (20.0, 4.0) km at t = 0, drifting 8 m/s towards 045° | 3 | 1 603 at the start (centre at 20.4 km); 1 333 at the end (24.5 km) |
| Fixed discretes | 25 points, uniform in 1–15 km, each reported with probability 0.5 per scan | — | 12.5 |
| **Total** | | | **≈ 4 190** at the start, ≈ 3 920 at the end; about 0.5 million over 125 scans |

The rain cell starts 7.81 km from the fix and drifts over the holding stack: its centre ends the run
3.06 km from the fix, inside the cell's radius.

### 3.3 Traffic

All trajectories are closed-form arcs (D24) in the site's ENU frame, metres, from t = 0 to 600 s.
Headings are true; turns are positive to the right. RDU is (13 861.5, −13 858.9) m. The runway
axis is taken as 050°/230°, representative and not surveyed, with the parallel thresholds 1 km
apart (006 §2.2 uses the same spacing) and 1.5 km from the airport reference point.

| ID | Kind | Start | Segments | Range span |
| :--- | :--- | :--- | :--- | :--- |
| A1 | Arrival, 23R | 22 km out on final, t = 0 | 230°, 72 m/s, straight | 19.0–31.6 km |
| A2 | Arrival, 23L | 20 km out on final, t = 0 | 230°, 72 m/s, straight | 20.0–30.7 km |
| A3 | Arrival, 23L, go-around | 12 km out, t = 100 s | Final to 2 km out; +1 m/s² for 13 s to 85 m/s; 27 s straight; left turn 3 °/s for 30 s to 140°; straight | 20.1–46.4 km |
| A4 | Arrival, 23R, right downwind | Downwind 4.75 km NW of the final, t = 0 | 050° for 12 km; right 90° at 3 °/s; base 2 km; right 90°; final | 15.2–26.0 km |
| A5 | Arrival, 23L, left downwind | Downwind 4.75 km SE of the final, t = 150 s | Mirror of A4, left turns | 20.3–30.4 km |
| H1–H4 | Holding stack | Fix (25.0, 10.0) km, inbound 230°, 110 m/s | Right 180° at 3 °/s (radius 2 101 m), 60 s outbound, right 180°, 60 s inbound; period 240 s. Phases 0, 4, 120, 128 s | 24.2–34.9 km |
| T1 | Overflight | (−29.28, 10.0) km, t = 0 | 090°, 230 m/s, straight; crosses H2 at t = 236 s | 10.0 km at closest |
| T2 | Overflight | 40 km before A1's t = 200 s position | 200°, 200 m/s, straight; crosses A1 at t = 200 s | 21.9 km at closest |
| L1, L2 | Light aircraft, circuit | General-aviation field (−9.0, 12.0) km, runway 05 | 50 m/s; left-hand rectangle, 3 km legs and 1.5 km ends, rate-2 turns (6 °/s, radius 477 m); L2 60 s behind L1 | 15.0–16.9 km |
| HE | Air-ambulance helicopter | (1.6, 0.8) km | Hover 90 s; +2 m/s² for 30 s; 120° at 60 m/s | 1.79–30.7 km |
| B00–B11 | Bird flock | Centre (6.0, 15.0) km | 12 m/s towards 200°; fixed offsets `N(0, 60 m)` per axis, seeded | 8.9–16.2 km |

The flock spans about 49 m in range and 43 m across at its start, 0.42 × 0.11 resolution units, so
most scans see one to three merged bird plots.

T1, T2 and A3 leave the 40 km coverage for part of the run. Plots exist only inside it; truth
exists everywhere, and a target outside coverage is not a missed target (§3.6, GOSPA).

### 3.4 Trackers and their settings

Every tracker uses the same measurement model (spherical 2-D, `tracker-001.md` §6.1, the
marginal over `(range_m, azimuth_rad)`), the same gate probability 0.99, and the same plots.

**Motion models and IMM.** State names follow `core/tracking/motion.py`: `x_m, xdot_mps, y_m,
ydot_mps`, plus `turn_rate_radps` for the coordinated turn.

| Mode | Model | Noise |
| :--- | :--- | :--- |
| M1 quiet | CV, continuous white acceleration (`tracker-001.md` TD5) | `acceleration_noise_density_m2ps3 = 0.5` |
| M2 turn | Coordinated turn, turn rate in the state [7] | 0.5 on each position axis; `turn_rate_noise_density_rad2ps3 = 1.6e-5` (σ_Δω = 0.5 °/s per 4.8 s scan) |
| M3 agile | CV | `acceleration_noise_density_m2ps3 = 20.0` |

Mode switching is a **continuous-time** Markov chain, so the matrix stays consistent when the
revisit time changes. Mean sojourn times are τ = (120, 30, 20) s, with leaving probability split
equally between the other two modes. The generator `Λ` has `Λ_ii = −1/τ_i` and
`Λ_ij = 1/(2τ_i)`, and `Π(T) = expm(Λ T)`:

```
   T = 4.8 s:   [0.9627 0.0192 0.0181]      T = 1.2 s:   [0.9902 0.0049 0.0049]
                [0.0767 0.8569 0.0664]                   [0.0198 0.9611 0.0191]
                [0.1089 0.0995 0.7916]                   [0.0293 0.0286 0.9421]
```

On mixing into M1 or M3, which lack `turn_rate_radps`, the coordinate is added at zero mean and
standard deviation 3 °/s, and dropped on return (`tracker-001.md` §9).

**Associators.**

| Name | Setting |
| :--- | :--- |
| GNN | Scenario 003's, with the finite non-assignment cost of `tracker-001.md` §7.2 |
| JPDA | Exact enumeration of feasible events per cluster; `max_events_per_cluster = 100_000`, above which it raises, naming the cluster's size |
| JPDA\*, Set-JPDA | `coalescence = "jpda_star"` [3], `"set"` [4] |
| JIPDA | JPDA with existence; `p_survival = 0.995` per scan; confirm at `existence ≥ 0.95`, delete at `existence < 0.05` |
| MHT | Track-oriented; `n_scan = 3` (14.4 s of deferral at 4.8 s); `k_best = 10` global hypotheses per cluster per scan; score-based confirmation at the Wald threshold `ln((1 − β)/α)` = 9.105 with α = 1e-4, β = 0.1; deletion when the score falls 2.303 = `−ln(β/(1−α))` below its maximum |
| PMBM | Single CV model at M3's noise; Poisson birth intensity 0.05 per scan, uniform over coverage; `k_best = 100` per cluster; Bernoulli pruning at existence 1e-5, global-hypothesis pruning at weight 1e-4 |

**Clutter intensity seen by the trackers.** JPDA, JIPDA, MHT and PMBM all need λ(z), the clutter
intensity at each measurement. Two settings: `clutter_intensity = "oracle"` reads the truth of
§3.2 (for the analytic tests), and `"map"` estimates it from a **clutter map**: plot counts per
(range, azimuth) cell of 1 km × 5°, averaged with a forgetting factor 0.9 per scan, excluding
plots assigned to confirmed tracks. Real surveillance radars keep a clutter map for the same
reason. V4 reports both.

**Lifecycle.** GNN and JPDA keep 003's M-of-N rules, 4 of 5. JIPDA uses existence; MHT uses its
score; PMBM's existence is internal. Every tracker reports a track once it is confirmed by its
own rule.

### 3.5 Variants

| | **V1** `manoeuvre` | **V2** `crossing` | **V3** `cst-sweep` | **V4** `terminal-dense` | **V5** `iq-check` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Traffic | A3, A4, H1, L1, HE alone, one at a time | Pairs at 72 m/s crossing at 90°, 30°, 10°, closing along and across range, at 20 km | Pairs on parallel courses at `d` ∈ {0.25, 0.5, 1, 2, 4}, both orientations, at 20 km | All of §3.3 | Four aircraft placed in radar coordinates: two pairs at Δr = 0.25–2 range bins, same Doppler |
| Clutter | `n_gate` = 0.01, uniform | `n_gate` = 1, local window | `n_gate` ∈ {0.1, 1, 3}, local window | §3.2 table | Thermal, from the chain |
| Scan period | 1.2, 2.4, 4.8 s | 4.8 s | 4.8 s | 4.8 s | 1.0 s frames |
| Trackers | CV-quiet, CV-agile, CT, IMM; each with GNN | GNN, JPDA, JPDA\*, Set-JPDA, JIPDA, MHT, PMBM; CV-quiet | Same as V2 | IMM with GNN, JIPDA, MHT; PMBM | None: plot statistics only |
| Trials | 200 per target and revisit | 200 per angle and orientation | 100 per point, 40 scans | 20 seeds, 125 scans | 120 frames |
| Purpose | IMM's benefit, and that it grows with T | Swaps, sharing, coalescence | Resolution curves | Everything at once; GOSPA decomposition | The plot model is honest |
| Level | Plot | Plot | Plot | Plot | **IQ** → RD → CFAR → `Plot` |
| Marker | `slow`; one point in the suite | `slow` | `slow`; curves in a notebook, two points in the suite | `slow`; one seed in the suite | `slow` |

**V5 is the check on the plot model.** It runs scenario 001's S3 dual-PRF chain (range bin
74.948 m; velocity bins 0.5975 and 0.7170 m/s for bursts A and B; unambiguous after
`unfold_doppler_dual_prf`), the one waveform where Doppler does not fold, through 003's detection
path. It places pairs whose range separation is swept across the merge region. It measures three
things: Pd, the false-alarm rate over the valid CFAR cells, and the fraction of frames in which a
pair gives one detection. It then fits the range resolution parameter α_r of [18] to that
fraction. The plot model passes if the measured Pd and false-alarm rate lie inside the plot
model's confidence intervals and the merged fraction follows `exp(−d²/2)` at the fitted α_r
(A13, A14). Azimuth is synthesised from truth there, as in 003 §6.3, so V5 checks the
range-Doppler half of the plot model and says so.

### 3.6 Mathematical contracts

`tracker-001.md` §4–§7 hold unchanged. Added:

| Contract | Statement |
| :--- | :--- |
| Truth arc, straight | `p(t) = p₀ + (v₀ τ + ½ a τ²) h`, `h = (sin ψ, cos ψ)`, τ = t − t₀ (D24) |
| Truth arc, turn | `p(t) = p₀ + (v/ω) [cos ψ₀ − cos ψ(t), sin ψ(t) − sin ψ₀]`, `ψ(t) = ψ₀ + ω τ`, bearing clockwise from north (D24) |
| Coordinated turn, F | For state `[x, ẋ, y, ẏ, ω]`, the exact discretisation of [7] §4.2: `x' = x + (sin ωT/ω) ẋ − ((1 − cos ωT)/ω) ẏ`, `ẋ' = cos ωT ẋ − sin ωT ẏ`, symmetric in y; ω constant. For `\|ωT\| < 1e-6` the Taylor series to second order is used, so `F → F_CV` continuously (A1). Equation numbers to confirm at implementation |
| IMM mixing [6] | `μ_{j\|i} = Π_ij μ_i / c̄_j`, `c̄_j = Σ_i Π_ij μ_i`; mixed means and covariances by name in the union layout; mode update `μ_j ∝ Λ_j c̄_j` with `Λ_j` the mode likelihood |
| IMM output | Mean and covariance of the mixture in the union layout, mode probabilities alongside |
| Gate volume | `V = π γ sqrt(\|S\|)` in (m·rad) for the 2-D measurement; `n_gate = λ V` |
| JPDA weights [2] | For cluster events θ: `P(θ) ∝ Π_{j assoc} (P_D N(z_j; ẑ_t, S_t) / λ(z_j)) Π_{t unassoc} (1 − P_D P_G)`; `β_tj = Σ_{θ ∋ (t,j)} P(θ)`, `β_t0 = 1 − Σ_j β_tj` |
| JPDA update | `x̂ = x̂⁻ + K ν̄`, `ν̄ = Σ_j β_tj ν_j`; `P = β_t0 P⁻ + (1 − β_t0) P^c + K (Σ_j β_tj ν_j ν_jᵀ − ν̄ ν̄ᵀ) Kᵀ` (the spread-of-innovations term), applied through the correction core (`tracker-001.md` §4.2, §7.4) |
| JIPDA existence [5] | `P(E⁻) = p_s P(E)`; posterior existence from the event probabilities with the track's `1 − P_D P_G P(E⁻)` miss term |
| Clutter intensity | `λ(z)` per measurement in plots per (m·rad), shape `(n_measurements,)`: the oracle of §3.2 or the clutter map of §3.4 |
| MHT track score [10] | `L_k = L_{k−1} + ln(P_D / λ(z)) − ½ ν ᵀ S⁻¹ ν − ½ ln \|2π S\|` on a hit, `+ ln(1 − P_D P_G)` on a miss |
| Merged plots [18] | Pairwise unresolved with probability `P_u(d) = exp(−d²/2)`; groups are formed by union over pairs, in increasing `d`; one plot per group at the group's centroid plus noise of the §3.1 `R`. A group is detected with `1 − Π_i (1 − P_D,i)` |
| GOSPA [21] | `p = 2`, `c = 300 m`, `α = 2`, over aircraft in coverage; reported as localisation, missed and false components, in metres |
| Trajectory GOSPA [22] | Same `p`, `c`; switching cost `γ = c`; reported per run with its switch count |
| Resolved pair (V3) | Two confirmed tracks, each within `c/2` of its own aircraft in at least 90 % of the last 20 scans, with no switch |
| Coalescence ratio (V3) | Mean estimated separation over true separation, last 20 scans |
| Swap (V2) | The truth-to-track GOSPA assignment changes for either aircraft between the scan before the crossing and 5 scans after |

### 3.7 Output schema

As `spec/data-001-formats.md`, with:

| File | Change |
| :--- | :--- |
| `truth.csv` | One row per target per scan: `target_id` (A1…B11), `kind` (`aircraft`, `bird`), ENU position and velocity, `range_m`, `azimuth_deg`, `range_rate_mps`, `in_coverage` |
| `plots.csv` | **New.** One row per plot: `scan`, `time_s`, `plot_id`, `range_m`, `azimuth_deg`, `source` (`target`, `merged`, `clutter`, `discrete`), `source_ids` (semicolon-separated truth IDs, empty for clutter). The tracker never reads `source` or `source_ids` |
| `tracks.csv` | As data-001 §5, per tracker (`tracker` column); adds `existence_probability` (JIPDA), `score` (MHT), and `mode_probability_{quiet,turn,agile}` (IMM) |
| `metrics.csv` | **New.** Per scan per tracker: GOSPA and its three components; per run: trajectory GOSPA, switch count, track count, confirmed false tracks |
| `metadata.json` | New `plot_model` block: `pd`, `mti_notch_mps`, `pd_notch`, `alpha_range_m`, `alpha_azimuth_rad`, clutter components with their intensities |

`tracks.csv` is causal: the row at scan k is what the tracker reported at scan k, including MHT,
whose later preference for another hypothesis is not written back (D29).

---

## 4. Core Decisions

The newest decision on this branch is scenario 010's D23. This spec owns D24–D30.

### 4.1 D24 — Truth is continuous-time and exact

Every trajectory is a chain of closed-form arcs: straight with constant acceleration along
track, or a turn at constant rate and speed (§3.6). Positions and velocities are evaluated
analytically at each plot's own time.

*Why:* generating truth with the filter's own discretisation (the CV transition, the CT
transition) is an inverse crime: the motion model is then exact by construction, and IMM's
benefit and the NEES tests measure nothing. Closed-form arcs are exact, cheap and readable, and a
turn's radius and acceleration are numbers an intern can check (`v/ω`, `v ω`).
*Rejected:* random-walk truth (no manoeuvre has a cause); recorded tracks (one aircraft in
`data/flight_coordinates.csv`, no ground truth for turn onsets).

### 4.2 D25 — The plot-level simulator: mechanism in core, scene in pipelines

1. **Why plots.** Scenario 003 §13.3 warns that a multi-target scenario built on its detector,
   which cannot resolve two targets in Doppler and whose link budget never misses, "would be
   measuring the wrong things". At plot level Pd, resolution and clutter are each one stated
   parameter, swept independently, with exact truth for every plot. V5 shows the parameters are
   honest against the IQ chain.
2. **Mechanism in core.**
   - `core/plots.py` (new): `detect` (Pd with the MTI notch), `resolve` (the merged-plot model of
     §3.6, union-find over pairs), `measure` (noise of `R`).
   - `core/clutter.py` (widened; 005 and 008 own its other contents): `ClutterComponent` (region,
     intensity), `poisson_clutter_plots`, `FixedDiscretes`. Clutter is generated in (range,
     azimuth) at intensity per (m·rad).
3. **Scene in pipelines.** `pipelines/terminal_area.py` holds the traffic of §3.3, the clutter
   layout of §3.2, the scan loop and the batching. `tracker-001.md` TD9 is the rule.
4. **Plot time and batching.** A rotating radar sees each azimuth at a different time. Each plot
   carries the time the beam centre crossed it, `t = t_scan + az / ω_rot`. `Tracker` takes
   batches that share one time (`tracker-001.md` §16), so the pipeline cuts each rotation into
   16 sectors of 22.5° (0.3 s) and stamps each sector's batch at its centre time.
   - **Error.** The stamp is at most 0.15 s off; at 230 m/s, the fastest aircraft here, that is
     34.5 m, about 1.0 σ_r. For every other aircraft it is under 16.5 m, or 0.5 σ_r.
   - **Sector size.** A smaller sector cuts the error and adds batches. 16 is the setting, and A12
     shows that halving the sector moves V4's GOSPA by less than its seed-to-seed spread.
   - **Pairs across a sector edge.** Two plots on either side of an edge land in different
     batches, and so in different clusters. The spec accepts that and the test is honest about
     it.

### 4.3 D26 — `CoordinatedTurn`, with the turn rate in the state

`core/tracking/motion.py` gains `CoordinatedTurn(MotionModel)`: state
`(x_m, xdot_mps, y_m, ydot_mps, turn_rate_radps)`. It has the exact discretisation of §3.6, its
analytic Jacobian for the EKF, and the series form below `|ωT| = 1e-6`. Its process noise is
continuous white acceleration on position (TD5) plus a white turn-rate derivative
(`turn_rate_noise_density_rad2ps3`). A known-turn-rate variant is not added: with the rate in the
state, the UKF and EKF estimate it, and a fixed-rate bank is the IMM of a lesser model.

### 4.4 D27 — `IMM` is an `Estimator` of `Estimator`s

Exactly the extension point of `tracker-001.md` §9, now specified:

- `IMM(modes: Mapping[str, Estimator], transition_generator_per_s: NDArray, …)` satisfies
  `Estimator`. `Tracker` sees one estimator per track. Nothing outside `IMM` changes.
- `predict_to(timestamp_s)` forms `Π(T) = expm(Λ T)` for the actual interval. A fixed per-scan
  matrix is rejected, because sector batching (D25) makes intervals unequal.
- Mixing and output are in the **union layout**, by name. A mode lacking a coordinate is
  augmented at zero mean and a stated standard deviation (3 °/s for `turn_rate_radps`).
- `innovation_statistics` returns the mode-mixed likelihood, so GNN, JPDA, MHT and gating work
  unchanged, and the cost of §7.2 is its negative log.
- Mode probabilities are an exposed intermediate quantity (`starter.md` §1.1) and go to
  `tracks.csv`.

The modes may be KF, EKF or UKF. The test of A2 runs a one-mode IMM over each.

### 4.5 D28 — JPDA, JPDA\*, Set-JPDA and JIPDA

1. **Interface.** `tracker-001.md` §7.4's extension point, without changing `Tracker`'s loop.
   - `JointProbabilisticDataAssociation.associate(costs, *, detection_probability,
     clutter_intensity)` takes the (n_tracks, n_measurements) negative-log-likelihood cost
     matrix (§7.2) and `λ(z)`, shape `(n_measurements,)`.
   - It fills `AssociationResult.weights`, shape `(n_tracks, n_measurements + 1)`, with the last
     column for "no measurement".
   - The correction core applies the combined innovation and the spread term (§3.6). `Estimator`
     gains no method.
2. **Clusters, then exact events.** Tracks are clustered by shared gated measurements (connected
   components of the validation matrix), and each cluster's feasible events are enumerated
   exactly. If a cluster's events exceed `max_events_per_cluster`, the associator raises,
   naming the cluster's size; it never approximates silently. V4's stack and bird flock are the
   largest clusters, and the event count is a reported intermediate.
3. **Coalescence.** `coalescence="none"` is textbook JPDA; `"jpda_star"` keeps only the event
   per assignment pattern that does not swap identities [3]; `"set"` is Set-JPDA [4], which
   relabels the posterior mixture to minimise its spread at the cost of track identity inside
   the cluster. V3 shows all three.
4. **Existence.** JIPDA [5] propagates `existence_probability` on each `Track`
   (`tracker-001.md` §8.2 keeps it out until a decision uses it; this is that decision). A new
   `ExistencePolicy(confirm, delete)` in `LifecyclePolicy` replaces M-of-N for JIPDA only. It
   is selected by name, and M-of-N stays the default.
5. **Clutter intensity.** `core/tracking/clutter_map.py` (new): `ClutterMap`, the forgetting
   average of §3.4, with the oracle as a second implementation of the same small protocol.

### 4.6 D29 — Track-oriented MHT inside the one tracker loop

MHT defers association, and `Tracker` (TD1) decides once per scan. Two designs were weighed:

- *A second tracker loop, as PMBM has.* Rejected: MHT keeps the predict–gate–associate–update
  shape scan by scan, and its track trees are a richer estimator, not a different loop.
- **Adopted: the tree is the estimator.**
  - `TrackTree` satisfies `Estimator`. Its leaves are `Estimator`s, one per surviving
    association history. Its `state` is the leaf in the current best global hypothesis.
    `predict_to` predicts every leaf.
  - `TrackOrientedMHT` is an associator. It scores every (leaf, measurement) pair (§3.6). For each
    cluster it finds the `k_best` global hypotheses by Murty's algorithm [11], [12] over
    `linear_sum_assignment`. It then tells each tree to branch on the surviving hypotheses and
    prunes by N-scan [9].
  - New trees start from unused measurements, scored from zero, so for MHT the M-of-N initiator
    is replaced by score confirmation. The selector of `tracker-001.md` §8.3 picks it.

What this costs, recorded rather than hidden:

- The estimate reported at scan k is the best hypothesis *at k*. MHT can later prefer another,
  and `tracks.csv` is not rewritten (§3.7).
- A tree's memory grows as leaves × scans until N-scan pruning bounds it at `k_best` per
  cluster per scan, times `n_scan`. The leaf count is a reported intermediate.
- `murty_k_best` lives in `core/tracking/association.py` beside GNN, so the k = 1 case is
  provably GNN (A4).

### 4.7 D30 — PMBM is a comparison tracker beside `Tracker`, not inside it

`tracker-001.md` §7.5's argument stands: PMBM's association hypotheses, existence and birth are
part of the filter's density, so there is no separate associator, initiator or lifecycle to
swap. It is therefore **a second tracker**, `PmbmTracker`, in `core/tracking/rfs/pmbm.py`:

- It **shares** the motion models, measurement models, `innovation_stats`, `murty_k_best`, the
  clutter intensity and the metrics.
- It **owns** its Poisson undetected component, Bernoulli tracks and global hypotheses [13].
- It **reports** tracks in the same `tracks.csv` schema. Its "track" is a Bernoulli with
  existence above 0.5, and its label is the measurement history that created it.

*Why include it at all:* the brief requires the current method, and for dense clutter with
closely spaced targets PMBM is that method. Williams [14] shows MHT and JIPDA are approximations
of the same conjugate prior, which makes PMBM the reference against which their losses are
measured rather than a rival design. It runs one CV model, not IMM. V3 has no manoeuvres, so it
is a fair test of association; on V4 the comparison carries that caveat in its row of
`metrics.csv`. GLMB [15] is named, not built: it targets the same posterior with labels, and
one RFS reference is enough to score against.

`tracker-001.md` TD11 is amended from "PMBM out of scope" to this.

### 4.8 Amendments to `spec/structure.md` and `spec/tracker-001.md`

> **Applied.** These amendments are folded into both documents in the same change; they are
> authoritative. This section records what changed and why.

1. **`structure.md` Part B tree.**
   - New: `core/plots.py`, `core/tracking/imm.py`, `core/tracking/jpda.py`,
     `core/tracking/mht.py`, `core/tracking/clutter_map.py`, `core/tracking/metrics.py`,
     `core/tracking/rfs/pmbm.py`, `pipelines/terminal_area.py`.
   - Widened: `core/clutter.py` (Poisson clutter plots, 011); `core/tracking/motion.py`
     (`CoordinatedTurn`).
2. **`structure.md` scenario table and B.1 rows**, citing [1]–[23].
3. **`tracker-001.md`.**
   - §1.1: IMM and JPDA move from "extension points" to "specified by scenario 011".
   - §7.4 points here.
   - New §7.6 is MHT, per D29.
   - §7.5 and TD11 are amended per D30.
   - §8.2: `existence_probability` enters `Track` for JIPDA.
   - §9 points to D26 and D27.
   - §16's "No IMM, coordinated turn, Singer model, JPDA or MHT" becomes "No Singer model;
     IMM, JPDA, MHT and PMBM per scenario 011".
4. **`scenario-003-tracking.md` §13.3 and §13.4** gain a pointer to this spec as the multi-target
   and RFS trigger they named.

### 4.9 Workstream boundary

This scenario claims:

```
scenarios/scenario_011_{manoeuvre,crossing,cst_sweep,terminal_dense,iq_check}.toml
spec/scenario-011-terminal-area-tracking.md
src/radar_forge/core/plots.py
src/radar_forge/core/clutter.py               (Poisson clutter plots, fixed discretes only)
src/radar_forge/core/tracking/motion.py       (CoordinatedTurn only)
src/radar_forge/core/tracking/{imm,jpda,mht,clutter_map,metrics}.py
src/radar_forge/core/tracking/association.py  (murty_k_best; AssociationResult.weights)
src/radar_forge/core/tracking/lifecycle.py    (ExistencePolicy)
src/radar_forge/core/tracking/rfs/{__init__,pmbm}.py
src/radar_forge/pipelines/terminal_area.py
notebooks/011_terminal_area_tracking.ipynb
```

Consumed, never edited: `core/{geodesy,constants,detection,dsp}.py`, `core/tracking/{kalman,ukf,
estimation,measurement_models,tracker}.py` beyond what `tracker-001.md` §13 already plans, and
`pipelines/tracking.py`. `test_scenario_003.py` staying green is a precondition for every commit.
`tracker-001.md` §13's migration to one `Tracker` must be finished first: this scenario runs on
`Tracker`, never on `KalmanTracker`.

---

## 5. Build Sequence

```
  1. core/tracking/motion.py     CoordinatedTurn: exact F, Jacobian, series below |wT| = 1e-6
             |
             v
  2. core/tracking/imm.py        IMM over Estimators; expm(Lambda T); union layout by name
             |
             v
  3. core/plots.py, core/clutter.py   detect, resolve, measure; Poisson clutter, discretes
             |
             v
  4. pipelines/terminal_area.py  closed-form traffic (D24), sector batching (D25), V1-V4 scenes
             |
             v
  5. core/tracking/jpda.py, clutter_map.py   clusters, exact events, JPDA*/Set; JIPDA existence
             |
             v
  6. core/tracking/association.py, mht.py    murty_k_best; TrackTree; N-scan pruning
             |
             v
  7. core/tracking/metrics.py    GOSPA, trajectory GOSPA, swaps, coalescence ratio
             |
             v
  8. core/tracking/rfs/pmbm.py   PMBM beside Tracker, sharing models and metrics
             |
             v
  9. V5: 001 S3 chain with placed pairs; fit alpha_r; scenarios/*.toml; notebook
```

| Step | Module | Notes |
| :--- | :--- | :--- |
| 1 | `motion.py` *(widen)* | Lands with A1 and the Jacobian check of `tracker-001.md` AC6. |
| 2 | `imm.py` *(new)* | Lands with A2 and A5 on V1 before any multi-target code. |
| 3 | `plots.py` *(new)*, `clutter.py` *(widen)* | Lands with A6 and A7; no tracker needed. |
| 4 | `terminal_area.py` *(new)* | The trajectories of §3.3 regenerate the §2.1 table (A8). |
| 5 | `jpda.py`, `clutter_map.py` *(new)* | A3 (reductions to KF and PDA) before the scenario. |
| 6 | `association.py` *(widen)*, `mht.py` *(new)* | A4 (MHT k = 1, N = 1 is GNN) first. |
| 7 | `metrics.py` *(new)* | A9 (GOSPA hand-computed cases) first. |
| 8 | `rfs/pmbm.py` *(new)* | A10 (one target, no clutter, Pd = 1: the KF). |
| 9 | V5, TOMLs, notebook | V3's curves are the notebook's; the suite checks two points (A16). |

No new dependency: `scipy.linalg.expm` and `scipy.optimize.linear_sum_assignment` cover the matrix exponential and every assignment, including each step of Murty's algorithm.

---

## 6. Acceptance Criteria

Unit criteria live in `tests/core/tracking/test_{motion,imm,jpda,mht,metrics,pmbm}.py` and
`tests/core/test_plots.py`; scenario criteria in `tests/pipelines/test_scenario_011.py`. Trials are
seeded plot-level draws.

| # | Criterion | Bound | Justification | Applies to |
| :--- | :--- | :--- | :--- | :--- |
| A1 | **CT reduces to CV.** `CoordinatedTurn` with ω = 1e-9 rad/s equals CV's F and Q on the shared coordinates; F is continuous across `\|ωT\| = 1e-6` | `rtol = 1e-10` | Catches a 0/0 or a sign in the series branch | unit |
| A2 | **One-mode IMM is its mode.** For KF, EKF and UKF modes, a one-mode IMM's state, covariance and likelihood equal the bare filter's over 50 scans | `rtol = atol = 1e-10` | Mixing with one mode is the identity | unit |
| A3 | **JPDA reductions.** (a) Pd = 1, no clutter, gates disjoint: JPDA equals the KF per track. (b) One track with clutter: JPDA equals PDA [1] | `rtol = 1e-10` | Exact algebra | unit |
| A4 | **MHT reduces to GNN.** `n_scan = 1`, `k_best = 1`: MHT's assignments and states equal GNN's with the finite non-assignment cost, on the V2 30° trials | identical assignments; states `rtol = 1e-10` | Murty's first solution is the assignment | unit |
| A5 | **IMM is consistent and switches.** V1, A4's turn, 200 trials: time-averaged NEES of IMM inside its 95 % chi-square interval; M2's probability exceeds 0.5 within 3 scans of turn onset at T = 4.8 s in ≥ 90 % of trials | as stated | Three scans is 391 m of departure, 8 innovation σ (§1). The 90 % to confirm at implementation | V1, `slow` |
| A6 | **Clutter rate.** Plot counts in each component of §3.2 inside the 99 % Poisson interval of the stated intensity × volume, over 125 scans | 99 % | Catches a per-m² versus per-m·rad error | unit |
| A7 | **Merge model.** Over 10 000 seeded pairs at each `d` ∈ {0.5, 1, 2}, the merged fraction is within the 99.9 % Clopper–Pearson interval of `exp(−d²/2)` | 99.9 % | The model is what V3 sweeps | unit |
| A8 | **Scene is the scene.** The §2.1 encounter table and the §3.1 notch fractions regenerate from `pipelines/terminal_area.py` | min `d` to 0.01; times to 0.1 s; fractions to 0.001 | The spec's numbers must be the code's | V4 |
| A9 | **GOSPA by hand.** Three hand-computed cases (one missed, one false, one localisation error) | exact | Catches an α or c^p slip | unit |
| A10 | **PMBM reduces to the KF.** One target, Pd = 1, no clutter, no birth after scan 0 | `rtol = 1e-9` | The single-Bernoulli case | unit |
| A11 | **IMM's benefit grows with revisit.** V1, position RMSE through A4's turns: IMM < min(CV-quiet, CV-agile) at T = 4.8 s, and the ratio IMM / best-CV decreases from T = 1.2 to 4.8 | strict | The teaching point (§1 item 1) | V1, `slow` |
| A12 | **Sector batching.** Halving the sector (D25 item 4) changes V4 mean GOSPA by less than one standard deviation across the 20 seeds | as stated | The 0.15 s stamp error is negligible | V4, `slow` |
| A13 | **Plot model honest: rates.** V5 measured Pd and false-alarm rate inside the 99 % intervals of the values the plot model is given | 99 % | D25 item 1 | V5, `slow` |
| A14 | **Plot model honest: merging.** V5 merged-detection fraction against `exp(−(Δr/α_r)²/2)` at the fitted `α_r`, at four separations | each inside its 99 % Clopper–Pearson interval | The resolution model is the CST model | V5, `slow` |
| A15 | **GNN swaps; MHT does not.** V2, 10° across-range crossing, 200 trials: swap rate GNN ≥ 0.2; MHT ≤ 0.05; JPDA's reported, not bounded | as stated | §2.2: sixteen unresolved scans. Bounds to confirm against a prototype at implementation | V2, `slow` |
| A16 | **Coalescence shown, and fixed.** V3, `d` = 1, across range, `n_gate` = 1: JPDA coalescence ratio ≤ 0.8; JPDA\* and Set-JPDA ≥ 0.9; P(resolved) for MHT ≥ P(resolved) for JPDA | as stated | The teaching point (§1 item 2) | V3, `slow` |
| A17 | **The reference line.** V3, every point: PMBM's P(resolved) ≥ every other tracker's, less 0.05 | as stated | [14]: the others approximate PMBM's posterior. A tracker beating it materially is a bug in one of them | V3, `slow` |
| A18 | **No silent truncation.** A cluster above `max_events_per_cluster` raises `ValueError` naming its size | raises | D28 item 2 | unit |
| A19 | **Causal output.** For MHT, the `tracks.csv` row at scan k equals the tracker's state at scan k, recorded live | identical | D29 | V4 |
| A20 | **Markdown.** This spec passes `scripts/check_conventions.py` R7 | hook | | spec |

---

## 7. Limitations

- **Two dimensions.** Plots report horizontal range; altitude is neither measured nor tracked.
  At 25 km, two stack levels 300 m apart differ in slant range by 16 m (at 1.2 and 1.5 km) to
  26 m (at 2.0 and 2.3 km): under one σ_r and under a quarter of a range cell. So the
  simplification creates none of the closely spaced cases. It removes the 3-D tracking
  question, which is `enu_3d`'s in 003.
- **No range rate in the plot.** Surveillance plots may carry coarse Doppler. Without it,
  association rests on position alone, which is the harder case and the one this scenario
  studies. A variant with range rate is a later extension.
- **The radar is a class, not an installation.** §3.1's parameters are representative of
  airport surveillance radars [24]. Nothing here is a model of a fielded system.
- **The runway geometry is representative, not surveyed.** §3.3's 050°/230° axis and 1 km
  spacing are stated assumptions.
- **No secondary radar.** Real terminal traffic carries transponders. This scenario is
  primary radar only, because transponder replies would remove most of the association problem
  it exists to study.
- **Exact JPDA is exponential.** `max_events_per_cluster` bounds it and raises above the bound;
  approximate JPDA (k-best events) is a later extension.
- **PMBM with one motion model.** §4.7.
- **Sector batching splits pairs at sector edges.** §4.2 item 4.

---

## References

.. [1] Y. Bar-Shalom and E. Tse, "Tracking in a cluttered environment with probabilistic data
       association," *Automatica*, 11(5), pp. 451–460, 1975.
.. [2] T. E. Fortmann, Y. Bar-Shalom and M. Scheffe, "Sonar tracking of multiple targets using
       joint probabilistic data association," *IEEE J. Oceanic Eng.*, 8(3), pp. 173–184, 1983.
.. [3] H. A. P. Blom and E. A. Bloem, "Probabilistic data association avoiding track
       coalescence," *IEEE Trans. Autom. Control*, 45(2), pp. 247–259, 2000.
.. [4] L. Svensson, D. Svensson, M. Guerriero and P. Willett, "Set JPDA filter for multitarget
       tracking," *IEEE Trans. Signal Process.*, 59(10), pp. 4677–4691, 2011.
.. [5] D. Mušicki and R. Evans, "Joint integrated probabilistic data association: JIPDA," *IEEE
       Trans. Aerosp. Electron. Syst.*, 40(3), pp. 1093–1099, 2004.
.. [6] H. A. P. Blom and Y. Bar-Shalom, "The interacting multiple model algorithm for systems
       with Markovian switching coefficients," *IEEE Trans. Autom. Control*, 33(8),
       pp. 780–783, 1988.
.. [7] X. R. Li and V. P. Jilkov, "Survey of maneuvering target tracking. Part I: Dynamic
       models," *IEEE Trans. Aerosp. Electron. Syst.*, 39(4), pp. 1333–1364, 2003.
.. [8] D. B. Reid, "An algorithm for tracking multiple targets," *IEEE Trans. Autom. Control*,
       24(6), pp. 843–854, 1979.
.. [9] T. Kurien, "Issues in the design of practical multitarget tracking algorithms," in
       Y. Bar-Shalom (ed.), *Multitarget-Multisensor Tracking: Advanced Applications*, Artech
       House, 1990, pp. 43–83.
.. [10] S. S. Blackman, "Multiple hypothesis tracking for multiple target tracking," *IEEE
        Aerosp. Electron. Syst. Mag.*, 19(1), pp. 5–18, 2004.
.. [11] K. G. Murty, "An algorithm for ranking all the assignments in order of increasing
        cost," *Operations Research*, 16(3), pp. 682–687, 1968.
.. [12] I. J. Cox and M. L. Miller, "On finding ranked assignments with application to
        multitarget tracking and motion correspondence," *IEEE Trans. Aerosp. Electron. Syst.*,
        31(1), pp. 486–489, 1995.
.. [13] Á. F. García-Fernández, J. L. Williams, K. Granström and L. Svensson, "Poisson
        multi-Bernoulli mixture filter: direct derivation and implementation," *IEEE Trans.
        Aerosp. Electron. Syst.*, 54(4), pp. 1883–1901, 2018.
.. [14] J. L. Williams, "Marginal multi-Bernoulli filters: RFS derivation of MHT, JIPDA, and
        association-based MeMBer," *IEEE Trans. Aerosp. Electron. Syst.*, 51(3),
        pp. 1664–1687, 2015.
.. [15] B.-T. Vo and B.-N. Vo, "Labeled random finite sets and multi-object conjugate priors,"
        *IEEE Trans. Signal Process.*, 61(13), pp. 3460–3475, 2013.
.. [16] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
        and Navigation*, Wiley, 2001, ch. 11 (IMM).
.. [17] D. F. Crouse, "On implementing 2D rectangular assignment algorithms," *IEEE Trans.
        Aerosp. Electron. Syst.*, 52(4), pp. 1679–1696, 2016.
.. [18] W. Koch and G. van Keuk, "Multiple hypothesis track maintenance with possibly unresolved
        measurements," *IEEE Trans. Aerosp. Electron. Syst.*, 33(3), pp. 883–892, 1997.
.. [19] F. E. Daum and R. J. Fitzgerald, "Importance of resolution in multiple-target tracking,"
        *Proc. SPIE* 2235, Signal and Data Processing of Small Targets, 1994 (pages to confirm at
        implementation).
.. [20] D. Mušicki and M. R. Morelande, "Finite resolution multitarget tracking," *Proc. SPIE*
        5913, Signal and Data Processing of Small Targets, 2005 (pages to confirm at
        implementation).
.. [21] A. S. Rahmathullah, Á. F. García-Fernández and L. Svensson, "Generalized optimal
        sub-pattern assignment metric," *Proc. 20th Int. Conf. Information Fusion*, 2017.
.. [22] Á. F. García-Fernández, A. S. Rahmathullah and L. Svensson, "A metric on the space of
        finite sets of trajectories for evaluation of multi-target tracking algorithms," *IEEE
        Trans. Signal Process.*, 68, pp. 3917–3928, 2020.
.. [23] D. Schuhmacher, B.-T. Vo and B.-N. Vo, "A consistent metric for performance evaluation
        of multi-object filters," *IEEE Trans. Signal Process.*, 56(8), pp. 3447–3457, 2008.
.. [24] M. I. Skolnik (ed.), *Radar Handbook*, 3rd ed., McGraw-Hill, 2008, the chapter on
        air-traffic-control radar (chapter number to confirm at implementation).
