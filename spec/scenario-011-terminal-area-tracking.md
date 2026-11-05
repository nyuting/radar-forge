# Scenario 011 — Dense terminal-area traffic: crossings, clutter, manoeuvres and closely spaced aircraft (IMM, JPDA, JIPDA)

## 1. Status & Purpose

**Status: specified; being implemented as 011a.** The first multi-target tracking scenario. It
lands in two PRs: this document is 011a (coordinated turn, IMM, JPDA with its coalescence
fixes, JIPDA, the clutter map, the plot model, GOSPA; variants V1, V2, V4, V5).
[`scenario-011b-mht-pmbm.md`](scenario-011b-mht-pmbm.md) holds the rest: track-oriented MHT,
PMBM, the separation sweep V3 and their criteria.

Scenario 003 tracks one aircraft against thermal false alarms, with GNN. With one target, gates
never overlap, assignment is trivially optimal and a track swap is impossible.
`spec/tracker-001.md` §6 left IMM and JPDA as extension points until the first manoeuvring,
multi-target scenario. This document is that scenario.

The setting is civil air traffic in the terminal area around Raleigh-Durham, seen from the Duke
Receiver (the site of 001, 002 and 006). An airport-surveillance-class radar at the site reports
plots once per rotation. The traffic is arrivals on two parallel finals, arrivals turning onto
final, a go-around, a holding stack, two overflights, two light aircraft in a circuit, an
air-ambulance helicopter and a flock of birds. Each stress the trackers must survive has a named
cause in that traffic (§2).

Three teaching points, each a test in §6:

1. **Manoeuvres break a single motion model, and IMM repairs it.** A standard-rate turn at
   72 m/s is 3.77 m/s² of centripetal acceleration. Over three 4.8 s scans the aircraft leaves a
   constant-velocity prediction by 391 m (½ a t², t = 14.4 s). A CV filter tuned quiet loses the
   track; one tuned loose is noisy on every straight leg. IMM runs both and switches by mode
   probability (D27).
2. **Clutter and closely spaced aircraft break hard assignment, and the fixes fail in
   instructive ways.** GNN swaps tracks at a shallow crossing. JPDA shares measurements and does
   not swap, but on two close aircraft it pulls the two estimates together (coalescence).
   JPDA\* and the set relabelling counter that (D28); MHT does too, in 011b.
3. **The plot model is honest.** A plot-level simulator states Pd, resolution and clutter as
   parameters; V5 checks them against scenario 001's IQ chain (D25).

It also brings in two things earlier specs left waiting: a per-track existence probability
(a `TrackBank` column) with JIPDA's confirm and delete, and the plot level of simulation.

| Section | What it fixes |
| :--- | :--- |
| §2 Overview & Geometry | The sensor, the scene, the resolution cell in metres, the crossings |
| §3 Parameter Matrix | Sensor and plots, clutter, traffic, trackers and their settings, variants, contracts, outputs |
| §4 Core Decisions | D24–D28; the amendments to `tracker-001.md`; the workstream boundary |
| §5 Build Sequence | What is new, what is widened |
| §6 Acceptance Criteria | Analytic reductions first, then statistical consistency, then the trackers compared |
| §7 Limitations | What this scenario does not model, stated rather than discovered |

### 1.1 Upstream inspiration

| Upstream | What it is | What radar-forge takes | Licence | Baseline or current |
| :--- | :--- | :--- | :--- | :--- |
| GNN; Crouse [17] | One-to-one assignment by `linear_sum_assignment` | Scenario 003's associator, with a finite non-assignment cost; the baseline | existing code | Baseline |
| PDA [1]; JPDA [2] | Each track updated with a weighted mixture of the measurements in its gate | `probabilistic_data_association`, `joint_probabilistic_data_association`, `JointProbabilisticDataAssociation`, `filters.update_weighted` | paper, written from equations | Baseline |
| JPDA\* [3]; Set-JPDA [4] | JPDA with the swapped labellings pruned (JPDA\*), or with the posterior relabelled over unlabelled sets (Set-JPDA) | `coalescence="none" \| "jpda_star" \| "set"` | paper, written from equations | Current |
| IPDA [26]; JIPDA [5] | JPDA with a track-existence probability propagated and used to confirm and delete | the `existence` column of `TrackBank`; `ExistencePolicy` | paper, written from equations | **Current** (single-scan) |
| IMM [6], [16]; unequal-dimension mixing [25]; manoeuvre models [7] | A bank of motion models mixed by Markov mode probability | `tracking/imm.py`; `CoordinatedTurn` with the turn rate in the state | paper, written from equations | **Current** (manoeuvring) |
| Finite resolution [18], [19], [20] | Two targets inside one resolution cell give one merged plot | the merged-plot model of `tracking/scene.py` (D25) | paper, written from equations | Ground truth |
| GOSPA [21]; trajectory GOSPA [22]; OSPA [23] | Multi-target distance with localisation, missed and false terms; a trajectory version that charges track switches | `gospa`, `trajectory_gospa` in `tracking/metrics.py` | paper, written from equations | Metric |

Track-oriented MHT [8]–[12] and PMBM [13]–[15] are 011b's.

---

## 2. Overview & Geometry

The sensor is a rotating airport-surveillance-class radar at the Duke Receiver: S-band, 4.8 s per
rotation (12.5 rpm), a 1.4° azimuth beam and a 115.75 m range cell (1/16 nautical mile). Those are
the parameters of the class (Skolnik [24], ch. 2), not of a particular installation. It reports
**plots**, once per rotation per resolved echo: horizontal range and azimuth, with no range rate
(§3.1). Raleigh-Durham's airport reference point (ARP) is 19.601 km from the site at bearing
135.0°, (13 861.5, −13 858.9) m in the site's ENU frame (computed from the 002 coordinates
through the geodesy in `radar.py`).

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
   -8 |                                     T2 crosses A1 at 200 s, 30 deg
      |                                            A4, A5 downwind -> base -> final, 3 deg/s
  -14 |                         *  RDU (13.9, -13.9); A3 goes around, turns left to 140 deg
      +-----------------------------------------------------------------------------> E (km)
```

The traffic is chosen so that every stress has a cause:

| Stress | Cause in the scene | What it exercises |
| :--- | :--- | :--- |
| Manoeuvre onset | A4 and A5 turning base and final at 3 °/s; A3's go-around; the stack's 180° turns every 120 s; L1, L2's rate-2 circuit | IMM mode switching (D26, D27) |
| Longitudinal acceleration | A3 accelerating 1 m/s² in the go-around; HE from hover to 60 m/s at 2 m/s² | The agile CV mode |
| Closely spaced, long-lived | H1–H2 (4 s apart in the stack) and H3–H4 (8 s apart) | Coalescence; merged plots |
| Crossing, shallow | T2 over A1, 30° | GNN swap; JPDA's sharing |
| Crossing, wide | T1 over H2, 140° | The easy crossing, as a control |
| Doppler-blind interval | HE hovering; L1 and L2 tangential on their circuit; A1, A2 and A4 tangential near RDU | Missed plots: Pd drops in the MTI notch (§3.1) |
| Many unresolved real objects | Twelve birds in about 0.4 × 0.1 resolution cells | Structured clutter that is not false: track discipline |
| Dense, nonhomogeneous clutter | Ground-clutter residue inside 6 km; a rain cell drifting across the stack; 25 fixed discretes | The clutter map (D28); JIPDA existence |

**Altitude is not in the measurement.** The plots report horizontal range, the scene's aircraft are
tracked in the ground plane, and the holding stack's 300 m vertical spacing is invisible to the
radar (§7).

### 2.1 The resolution cell, in metres

Two echoes are unresolved when they share a range cell and a beam. The range cell is 115.75 m
everywhere. The beam's cross-range width grows with range:

| Range | 5 km | 10 km | 20 km | 26.9 km (the fix) | 30 km | 40 km |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.4° beam, cross-range | 122.2 m | 244.3 m | 488.7 m | 657.9 m | 733.0 m | 977.4 m |
| σ_az = 0.15°, cross-range | 13.09 m | 26.18 m | 52.36 m | 70.5 m | 78.54 m | 104.72 m |

Separations are given in **resolution units**, the normalised distance

```
   d = sqrt( (Δr / α_r)² + (Δaz / α_az)² ),     α_r = 115.75 m,  α_az = 1.4° = 0.024435 rad
```

and a pair at distance `d` gives one merged plot with probability `P_u(d) = exp(−d²/2)` (Koch and
van Keuk's Gaussian resolution model [18], §3.6): 0.607 at `d = 1`, 0.135 at 2, 0.011 at 3.

The scene's close encounters, from the closed-form trajectories of D24 at 0.1 s steps over the
600 s run, regenerated from `config/scenario_011_terminal_dense.toml` by
`tests/tracking/test_scene.py` (A8). A scan's beam crosses the pair at their mean azimuth,
`t = t_rotation + az/(2π) T`, found by fixed point.

| Pair | Minimum `d` | First at | Ground separation then | Track angle | Scans with `P_u > 0.5` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| H1–H2 (stack, 4 s apart) | 0.52 | 151.7 s | 439 m | 12° | **10** |
| H3–H4 (stack, 8 s apart) | 1.03 | 29.7 s | 874 m | 24° | 3 |
| A1–T2 (shallow crossing) | 0.00 | 200.0 s | 0 m | 30° | 1 |
| H2–T1 (wide crossing) | 0.00 | 236.0 s | 0 m | 140° | 1 |
| A2–A3 (same final, in trail) | 2.36 | 244.4 s | 815 m | 0° | 0 |

The first version of this table gave H2–T1 no unresolved scan; with the beam timing above, the
rotation at 235.2 s crosses the pair 0.11 s after the crossing, at `d` well under 1.

The parallel finals are 1 km apart and are **not** closely spaced from this site. The line of
sight from Duke meets the final course at about 95° near the airport, so the 1 km lateral offset
lies mostly along range: 5–9 range cells. The holding stack is where aircraft overlap, because
in the ground plane its levels share one racetrack.

### 2.2 Crossings, in scans

At a crossing angle θ between equal speeds `v`, the relative speed is `2 v sin(θ/2)`. The time
inside one resolution unit depends on which way the pair close:

| θ at 72 m/s | Relative speed | Inside `\|Δr\| < α_r` (closing along range) | Inside `\|Δaz\| < α_az` at 20 km (closing across range) |
| :--- | ---: | ---: | ---: |
| 90° | 101.8 m/s | 2.27 s = 0.47 scans | 9.6 s = 2.0 scans |
| 30° | 37.27 m/s | 6.21 s = 1.3 scans | 26.2 s = 5.5 scans |
| 10° | 12.55 m/s | 18.45 s = 3.8 scans | 77.9 s = **16.2 scans** |

A 10° crossing closing across the beam spends sixteen scans unresolved: the case GNN swaps on
and JPDA coalesces on. V2 runs all three angles in both orientations.

---

## 3. Parameter Matrix

### 3.1 Sensor and plots

A plot-level scenario's TOML has a `[sensor]` table instead of `[site]`, `[[burst]]` and
`[receiver]` (`scenario.SensorSettings`); the radar sits at the ENU origin.

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Site | Duke Receiver, 36.00250 °N, 78.94100 °W, 60 m; the ENU origin | Scenario-001's site |
| `carrier_frequency_hz` | 2.8e9 (λ = 0.10707 m) | S-band airport-surveillance class [24] |
| `scan_period_s` | **4.8** (12.5 rpm); V1's Monte Carlo also 1.2 and 2.4 | [24]; the revisit sweep of A11 |
| `beamwidth_azimuth_deg` | 1.4 | [24] |
| `range_cell_m` | 115.75 (1/16 nautical mile) | [24] |
| `range_min_m`, `range_max_m` | 1000, 40 000; all azimuths | Terminal area |
| Measurement | `z = [range_m, azimuth_rad]`, horizontal range, azimuth clockwise from north | §2, §7 |
| `sigma_range_m` | **33.414** = 115.75/√12 | Quantisation-limited, as scenario 003 |
| `sigma_azimuth_deg` | **0.15** | Plot-extraction and calibration budget; thermal alone gives θ₃/(1.6·√(2·SNR)) = 0.062° at 20 dB |
| `pd_aircraft`, `pd_bird` | 0.90, 0.30 per echo | Specified |
| `mti_notch_mps`, `pd_notch` | 4.0, 0.30: an echo with `\|ṙ\| < 4 m/s` is detected with `pd_notch` | Moving-target-indicator notch [24] |
| Plot time | The time the beam centre crossed the plot's azimuth, `t_rotation + az/(2π) T` | D25 item 4 |
| `n_sectors` | 16 batches of 22.5° (0.3 s) a rotation, each stamped at its centre | D25 item 4 |

Fraction of each aircraft's life spent in the notch (`|ṙ| < 4 m/s`, horizontal): HE 0.156,
L2 0.130, L1 0.120, A4 0.053, A2 0.052, A1 0.049, all others below 0.0125 (A5 0.005). The first
version gave L1 and L2 0.102 and 0.095 and a range span of 15.0–16.9 km: those are a
sharp-cornered 3 km × 1.5 km rectangle, without the rate-2 turns §3.3 states. With the turns
inside the rectangle's corners the span is 15.0–16.7 km and the fractions above.

### 3.2 Clutter

Clutter plots are Poisson in the measurement space, with intensity in plots per (m·rad) per scan
(`[[clutter_plots.component]]`, `tracking.scene.ClutterComponent`). The CFAR behind a real
extractor holds the false-alarm rate per resolution cell, uniform in range and azimuth, so the
intensity is per (m·rad), not per square metre.

Intensities are set by **expected clutter plots per gate**, `n_gate`, at a reference gate: the
99 % chi-square gate (γ = 9.2103 for two dimensions) of a steady-state quiet-CV filter
(CWNA q = 0.5 m²/s³, T = 4.8 s) on an aircraft at 20 km. Its innovation standard deviations are
46.64 m in range and 68.35 m across, so its area is `π γ sqrt(|S|)` = 9.225e4 m² =
**4.612 m·rad**.

| `n_gate` | Intensity (plots / m·rad / scan) | Plots per scan over the whole 1–40 km coverage (245 044 m·rad) |
| :--- | ---: | ---: |
| 0.01 | 0.0021683 | 531 |
| 0.1 | 0.021683 | 5 313 |
| 0.3 | 0.065048 | 15 939 |
| 1 | 0.21683 | 53 128 |
| 3 | 0.65048 | 159 384 |

Those coverage-wide totals are why V2 runs in **local windows**: a disc of clutter of radius
2.5 km about each crossing, 985.6 m·rad at 20 km, 214 plots a scan at `n_gate` = 1. (The first
version used a box, range ±2.5 km by azimuth ±5°, 872.7 m·rad; a disc is the region the code
draws, and its volume is stated.) V4 uses the whole coverage with a light background and heavy
patches:

| V4 component | Region | `n_gate` | Plots per scan |
| :--- | :--- | ---: | ---: |
| Background | Whole coverage | 0.01 | 531 |
| Ground-clutter residue | Range 1–6 km, all azimuths | 0.3 | 2 044 |
| Rain cell | Disc of radius 4 km, centre (20.0, 4.0) km at t = 0, drifting 8 m/s towards 045° | 3 | 1 619 at the start (centre at 20.40 km); 1 342 at 600 s (24.54 km) |
| Fixed discretes | 25 points, uniform in range 1–15 km and in azimuth, drawn once from a seed, each reported with probability 0.5 per scan | — | 12.5 |
| **Total** | | | **≈ 4 207** at the start, ≈ 3 930 at the end; about 0.5 million over 125 scans |

A disc's volume in (m·rad) is `∫ dA / r = 2π (c − sqrt(c² − R²))`, c its centre's range: 2 488.6
m·rad at the start. The first version used πR²/c, 2 464 m·rad and 1 603 plots. The rain cell
starts 7.81 km from the fix and drifts over the holding stack: its centre ends the run 3.06 km
from the fix, inside the cell's radius.

### 3.3 Traffic

All trajectories are closed-form arcs (D24) in the site's ENU frame, metres, from t = 0 to 600 s,
written in each config as `[[target]] kind = "arc"` with a class, a start time, a start position,
heading and speed, and legs (straight, accelerating, or turning; `repeat` repeats them). Headings
are clockwise from north; turns are positive to the right. The runway axis is 050°/230°,
representative and not surveyed; the 23R and 23L thresholds lie 1.5 km from the ARP along 050°
and 0.5 km either side of it (23R towards 320°, 23L towards 140°).

| ID | Kind | Start | Segments | Range span |
| :--- | :--- | :--- | :--- | :--- |
| A1 | Arrival, 23R | 22 km out on final, t = 0 | 230°, 72 m/s, straight | 19.0–31.6 km |
| A2 | Arrival, 23L | 20 km out on final, t = 0 | 230°, 72 m/s, straight | 20.0–30.7 km |
| A3 | Arrival, 23L, go-around | 12 km out, t = 100 s | Final to 2 km out (138.9 s); +1 m/s² for 13 s to 85 m/s; 27 s straight; left turn 3 °/s for 30 s to 140°; straight | 20.1–46.4 km |
| A4 | Arrival, 23R, right downwind | Downwind 4.75 km NW of the final, abeam 2 km past the threshold, t = 0 | 050° for 12 km; right 90° at 3 °/s; base 2 km; right 90°; final | 15.2–26.0 km |
| A5 | Arrival, 23L, left downwind | Downwind 4.75 km SE of the final, abeam 2 km past the threshold, t = 150 s | Mirror of A4, left turns | 20.3–30.4 km |
| H1–H4 | Holding stack | Fix (25.0, 10.0) km, inbound 230°, 110 m/s | Right 180° at 3 °/s (radius 2 101 m), 60 s outbound, right 180°, 60 s inbound; period 240 s. At t, target Hk is `φ_k` into its lap, φ = 0, 4, 120, 128 s (so H2 is 4 s ahead of H1) | 24.2–34.9 km |
| T1 | Overflight | (−29.28, 10.0) km, t = 0 | 090°, 230 m/s, straight; crosses H2 at the fix at t = 236 s | 10.0 km at closest |
| T2 | Overflight | A1's position at 200 s, less 40 km along 200° | 200°, 200 m/s, straight; crosses A1 at t = 200 s | 21.9 km at closest |
| L1, L2 | Light aircraft, circuit | General-aviation field (−9.0, 12.0) km, runway 05; L1 from the field at 0 s, L2 at 60 s | 50 m/s; left-hand 3 km × 1.5 km rectangle, the field at the start of the upwind straight, rate-2 turns (6 °/s, radius 477 m) inside its corners: straights of 2 045 m and 545 m, 163.6 s a lap | 15.0–16.7 km |
| HE | Air-ambulance helicopter | (1.6, 0.8) km | Hover 90 s; +2 m/s² for 30 s; 120° at 60 m/s | 1.79–30.7 km |
| B00–B11 | Bird flock | Centre (6.0, 15.0) km | 12 m/s towards 200°; fixed offsets `N(0, 60 m)` per axis, `default_rng(20261106)` | 8.9–16.2 km |

A4's downwind start is not in the first version; "abeam 2 km past the threshold" is the one that
gives its stated range span (15.2–26.0 km) and notch fraction (0.053), and A5's (20.3–30.4 km).

T1, T2 and A3 leave the 40 km coverage for part of the run. Plots exist only inside it; truth
exists everywhere, and a target outside coverage is not a missed target (§3.6, GOSPA).

### 3.4 Trackers and their settings

Every tracker uses the same measurement model: `RangeAzimuthElevationRangeRate` with
`components=("range_m", "azimuth_rad")`, the radar at the ENU origin and `fixed={"z_m": 0.0}`,
so its slant range is the horizontal range. The same gate probability 0.99 and the same plots.
The state is the ground plane: `x_m, xdot_mps, y_m, ydot_mps`, plus `turn_rate_radps` for the
coordinated turn. Settings come from the `[tracking]` table (`pipeline.PlotTracker`).

**Motion models and IMM.** Process noise is CWNA with intensity `q = 2 σ² τ`
(`sigma_acceleration_mps2`, `acceleration_correlation_time_s` = 1 s): the first version's
`acceleration_noise_density_m2ps3` of 0.5 and 20.0 m²/s³ are σ = 0.5 and √10 = 3.1623 m/s².

| Mode | Model | Noise |
| :--- | :--- | :--- |
| M1 quiet | `CartesianMotion("xy")` | σ = 0.5 m/s² (q = 0.5 m²/s³) |
| M2 turn | `CoordinatedTurn`, turn rate in the state [7] | σ = 0.5 m/s² on each position axis; `turn_rate_noise_density_rad2ps3 = 1.6e-5` (σ_Δω = 0.5 °/s per 4.8 s scan) |
| M3 agile | `CartesianMotion("xy")` | σ = 3.1623 m/s² (q = 20 m²/s³) |

Mode switching is a **continuous-time** Markov chain, so the matrix stays consistent when the
revisit changes. Mean sojourn times are τ = (120, 30, 20) s, leaving split equally. The
generator `Λ` has `Λ_ii = −1/τ_i` and `Λ_ij = 1/(2τ_i)`, and `Π(T) = expm(Λ T)`:

```
   T = 4.8 s:   [0.9627 0.0192 0.0181]      T = 1.2 s:   [0.9902 0.0049 0.0049]
                [0.0767 0.8569 0.0664]                   [0.0198 0.9611 0.0191]
                [0.1089 0.0995 0.7916]                   [0.0293 0.0286 0.9421]
```

Mixing is in the union layout, the coordinated turn's (Granström, Willett and Bar-Shalom [25]
§III, eq. (10), with a fixed distribution). A CV mode carries `turn_rate_radps` at zero mean and
3 °/s standard deviation, uncorrelated, reset on every mixing and held through its prediction:
that is what it contributes when mixed into M2. New tracks start at the chain's stationary
mode probabilities, ∝ τ: (0.706, 0.176, 0.118).

**Associators.**

| Name | Setting |
| :--- | :--- |
| GNN | Scenario 003's, costed by the negative log-likelihood ratio against the clutter regions' mean intensity, with `non_assignment_cost = ln(P_D / (1 − P_D P_G))` = 2.11, the track score's break-even |
| JPDA | Exact enumeration of feasible events per cluster; `max_events_per_cluster = 100_000`, above which it raises, naming the cluster's size |
| JPDA\*, Set-JPDA | `coalescence = "jpda_star"` [3], `"set"` [4] |
| JIPDA | JPDA with existence; `ExistencePolicy(p_survival=0.995, confirm=0.95, delete=0.05, initial=0.2)`: confirm at `existence ≥ 0.95`, delete below 0.05, a new track at 0.2 |

In `"confirmed_first"` assignment (the default) confirmed tracks are weighted jointly by JPDA
(JIPDA) and tentative tracks each alone by PDA (IPDA) over the measurements outside every
confirmed gate, so a new track's wide first gate never joins a cluster. A hit, for M-of-N, is a
scan with a measurement in the gate.

**Clutter intensity seen by the trackers.** JPDA and JIPDA need λ(z), the clutter intensity at
each measurement, which `Scan.clutter_intensity` carries. `[tracking] clutter_intensity =
"oracle"` reads the truth of §3.2; `"map"` estimates it with a `ClutterMap`: plot counts per
(range, azimuth) cell of 1 km × 5°, averaged with a forgetting factor 0.9 per scan, excluding
plots the confirmed tracks took. The map starts at twice the birth limit below, as one scan
already seen, so no track starts in a cell before the map has looked at it.

**Lifecycle.** GNN and JPDA keep scenario 003's M-of-N rule, 4 of 5, deleting after 3 misses in
a row. JIPDA uses existence. Every tracker also:

- starts no track from a plot where λ(z) exceeds `max_birth_clutter_intensity_per_m_rad` = 0.01
  (n_gate ≈ 0.05): a new track's first gate, its velocity unknown, is a hundred reference gates;
- deletes a confirmed track whose position standard deviation (the root of the largest eigenvalue
  of its position covariance) exceeds `max_position_sigma_m` = 300 m after its update, the
  covariance-based deletion of `tracker-001.md` §5.4: under PDA in dense clutter a track's
  covariance can inflate until its gate holds hundreds of plots.

A new track starts at the plot's debiased position (`DebiasedSphericalInitiator`), a zero
velocity of standard deviation 150 m/s per axis and the turn-rate stand-in.

### 3.5 Variants

| | **V1** `manoeuvre` | **V2** `crossing` | **V4** `terminal_dense` | **V5** `iq_check` |
| :--- | :--- | :--- | :--- | :--- |
| Traffic | A3, A4, H1, L1, HE in one scene, kilometres apart | Six pairs at 72 m/s crossing at 90°, 30°, 10°, each closing along and across range, at 20 km, at 200 s | All of §3.3 | Four pairs of aircraft placed in radar coordinates at 0.5, 1.5, 3 and 6 bistatic range bins, same velocity |
| Clutter | `n_gate` = 0.01, uniform | `n_gate` = 1 in a disc of 2.5 km about each crossing | §3.2 table | Thermal, from the chain |
| Scan period | 4.8 s; 1.2, 2.4, 4.8 s in A11's Monte Carlo | 4.8 s | 4.8 s | 1.0 s frames |
| Tracker (config) | IMM, GNN | CV quiet, JPDA | IMM, JIPDA, clutter map | None: plot statistics only |
| Monte Carlo | A5: 200 trials; A11: CV quiet, CV agile, IMM at three revisits | — | A12: 20 seeds | — |
| Level | Plot | Plot | Plot | **IQ** → RD → CFAR → `Plots` |

V3 (the separation sweep) is 011b's; A16's JPDA part runs V3's point `d = 1` across range here.

**V5 is the check on the plot model.** It runs scenario 001's S3 dual-PRF chain (slant range bin
74.948 m, so 149.896 m on the map's bistatic axis; Doppler unambiguous after
`unfold_doppler_dual_prf`), the one waveform where Doppler does not fold, through scenario 003's
detection (`pipeline.frame_plots`, the `rd` stage). It places four pairs, each pair's members on
one bearing at one velocity so their separation holds, at 0.5, 1.5, 3 and 6 bistatic range bins.
(The first version had two pairs; A14's four separations need four.) It measures Pd (a target is
detected when a burst-0 detection lies in its pair's window, alone or merged), the false-alarm
rate over the cells away from every pair, and the fraction of frames in which a pair gives one
detection; it fits `α_r` to that fraction. Azimuth is not modelled there, so V5 checks the
range-Doppler half of the plot model.

### 3.6 Mathematical contracts

`tracker-001.md` §3–§5 hold unchanged. Added:

| Contract | Statement |
| :--- | :--- |
| Truth arc, straight | `p(t) = p₀ + (v₀ τ + ½ a τ²) h(ψ₀)`, `h(ψ) = (sin ψ, cos ψ)` east and north, τ = t − t₀ (D24) |
| Truth arc, turn | `p(t) = p₀ + (v/ω) [cos ψ₀ − cos ψ(t), sin ψ(t) − sin ψ₀]`, `ψ(t) = ψ₀ + ω τ`, heading clockwise from north, ω positive to the right (D24) |
| Coordinated turn, F | State `[x, ẋ, y, ẏ, ω]`, ω the heading's rate, positive right (Li and Jilkov's [7] §V counter-clockwise ω with its sign reversed). With s = sin ωT, c = cos ωT: `x' = x + (s/ω) ẋ + ((1 − c)/ω) ẏ`, `ẋ' = c ẋ + s ẏ`, `y' = y − ((1 − c)/ω) ẋ + (s/ω) ẏ`, `ẏ' = −s ẋ + c ẏ`, ω held; `1 − c = 2 sin²(ωT/2)`. For `\|ωT\| < 1e-6` the coefficients are their series to second order, so F → F_CV continuously (A1); the rate derivative of `s/ω` is its series below `\|ωT\| = 1e-2`, where its exact form cancels |
| Coordinated turn, Q | CV's CWNA block on each position axis and `q_ω T` on the rate |
| IMM mixing [16] §11.6.6 | `c̄_j = Σ_i Π_ij μ_i`, `μ_{i\|j} = Π_ij μ_i / c̄_j`; mode j starts from the moments of `Σ_i μ_{i\|j} N(x̂_i, P_i)` in the union layout |
| IMM likelihood and update | The gate and every associator see the combined estimate with the mode-mixed likelihood `g(z) = Σ_j c̄_j N(z; ẑ_j, S_j)`. With association weights β: `μ_j ∝ c̄_j (β_0 + Σ_i β_i N_j(z_i)/g(z_i))`, and mode j is corrected with the weights conditioned on j; one-hot weights give `μ_j ∝ Λ_j c̄_j` |
| IMM output | Mean and covariance of the mixture in the union layout, mode probabilities alongside |
| Gate volume | `V = π γ sqrt(\|S\|)` in (m·rad) for the 2-D measurement; `n_gate = λ V` |
| JPDA weights [2], [27] | For cluster events θ: `P(θ) ∝ Π_{t seen} (P_D r_t g_tθ_t / λ(z_θ_t)) Π_{t missed} (1 − P_D P_G r_t)`, r = 1 without existence; `β_tj = Σ_{θ ∋ (t,j)} P(θ)` |
| JPDA\*, set | Events detecting the same tracks with the same measurements differ only by labelling: JPDA\* keeps the likeliest of each group and renormalises [3]; `"set"` gives it the group's whole probability, [4]'s relabelling restricted to weights a track's own estimate can carry |
| Weighted update [27] | The moments of the mixture of per-measurement Kalman updates: `x̂ = x̂⁻ + Σ_j β_j K_j ν_j`, `P = β_0 P⁻ + Σ_j β_j (P_j + s_j s_jᵀ) − s̄ s̄ᵀ`, `s_j = K_j ν_j`; with one R for every measurement this is the PDA update `β_0 P⁻ + (1 − β_0) P^c + K (Σ_j β_j ν_j ν_jᵀ − ν̄ ν̄ᵀ) Kᵀ`, through the correction core (`tracker-001.md` §3) |
| JIPDA existence [5], [26] | `P(E⁻) = p_s P(E)` once per scan that sees the track; `r⁺ = Σ_j β_j + β_0 (1 − P_D P_G) r⁻ / (1 − P_D P_G r⁻)`; the state update uses the weights given existence |
| Clutter intensity | `λ(z)` per measurement in plots per (m·rad), `Scan.clutter_intensity` `(n_measurements,)`: the oracle of §3.2 or the clutter map of §3.4 |
| Merged plots [18] | Pairwise unresolved with probability `P_u(d) = exp(−d²/2)`, each pair drawn once, nearest first; groups are the connected components of the unresolved pairs; one plot per group at the group's centroid plus noise of the §3.1 `R`. A group is detected with `1 − Π_i (1 − P_D,i)` |
| GOSPA [21] | Definition 1, eq. (1), and Proposition 1 (α = 2): `p = 2`, `c = 300 m`, over the aircraft in coverage, each taken at its track's own time; reported as localisation, missed and false parts, metres |
| Trajectory GOSPA [22] | The LP metric of Proposition 2, eq. (23): same `p`, `c`, switching cost `γ = c`; reported per run with its switch count |
| Swap (V2) | The truth-to-track GOSPA assignment changes for either aircraft between the scan before the crossing and 5 scans after (a crossing: a pair of aircraft within c/2) |
| Turn-mode latency (V1) | Scans from a turn's onset until the turn mode's probability passes 0.5, for turns of at least 4 scans begun with a confirmed track |

### 3.7 Output schema

As `spec/data-001-formats.md`, with the `multitarget` stage writing:

| File | Contents |
| :--- | :--- |
| `plots.csv` | **New.** One row per plot: `scan`, `time_s`, `plot_id`, `range_m`, `azimuth_deg`, `source` (`target`, `merged`, `clutter`, `discrete`), `source_ids` (semicolon-separated truth names, empty for clutter). The tracker never reads `source` or `source_ids` |
| `tracks.csv` | One row per live track per rotation: `scan`, `time_s`, `track_id`, `status`, the state by name, `existence_probability`, and `mode_probability_{quiet,turn,agile}` (IMM) |
| `gospa.csv` | **New.** Per rotation: GOSPA and its localisation, missed and false parts, and the counts of truths and confirmed tracks |
| `scene.png` | Truth paths, target and false plots, and confirmed track paths, in the ground plane |

The truth is the config's closed-form arcs, so no `truth.csv` is written at plot level;
`metrics.csv` stays the `track` stage's. `tracks.csv` is causal: the row at scan k is what the
tracker held at scan k.

---

## 4. Core Decisions

### 4.1 D24 — Truth is continuous-time and exact

Every trajectory is a chain of closed-form arcs: straight with constant acceleration along
track, or a turn at constant rate and speed (§3.6), evaluated analytically at each plot's own
time (`tracking.scene.Arc`, `Leg`).

*Why:* generating truth with the filter's own discretisation is an inverse crime: the motion
model is then exact by construction, and IMM's benefit and the NEES tests measure nothing.
Closed-form arcs are exact, cheap and readable, and a turn's radius and acceleration are numbers
an intern can check (`v/ω`, `v ω`).
*Rejected:* random-walk truth (no manoeuvre has a cause); recorded tracks (one aircraft in
`data/flight_coordinates.csv`, no ground truth for turn onsets).

### 4.2 D25 — The plot-level simulator: mechanism in the library, scene in the config

1. **Why plots.** A multi-target scenario built on scenario 003's detector, which cannot
   resolve two targets in Doppler and whose link budget never misses, would measure the wrong
   things. At plot level Pd, resolution and clutter are each one stated parameter, swept
   independently, with exact truth for every plot. V5 shows the parameters are honest against
   the IQ chain.
2. **Mechanism in the library**, `tracking/scene.py` (`clutter.py` is scenario 005's, for
   signal-level cancellers): `detection_probability` (Pd with the MTI notch), `resolve` (the
   merged-plot model, connected components over pairs), `detect`, `measure` (centroid plus the
   noise of `R`); `ClutterComponent`, `poisson_clutter_plots`, `FixedDiscretes`,
   `clutter_intensity`; `ClutterMap`.
3. **Scene in the config, loop in `pipeline.py`.** The traffic of §3.3 and the clutter of §3.2
   are each config's `[[target]]` and `[clutter_plots]` tables; `pipeline.scans` plays a scenario
   rotation by rotation (exact truth at each beam crossing, the plot model, clutter and
   discretes, the oracle λ at each plot), and `pipeline.PlotTracker` builds the `Tracker` from
   `[tracking]` and feeds it.
4. **Plot time and batching.** Each plot carries the time the beam centre crossed it,
   `t = t_rotation + az / ω_rot`. `Tracker` takes `Scan`s each at one time, so `PlotTracker` cuts
   each rotation into 16 sectors of 22.5° (0.3 s) and stamps each sector's batch at its centre.
   A `Coverage` predicate tells `Tracker` which tracks a batch could see, those whose azimuth,
   extrapolated to the batch, lies in its sector: only they are predicted, gated and counted, so
   each track is predicted, mixed and counted once a rotation, not sixteen times.
   - **Error.** The stamp is at most 0.15 s off; at 230 m/s, the fastest aircraft here, that is
     34.5 m, about 1.0 σ_r. For every other aircraft it is under 16.5 m, or 0.5 σ_r.
   - **Sector size.** A12 checks that halving the sector moves V4's GOSPA by less than its
     seed-to-seed spread.
   - **Pairs across a sector edge** land in different batches, and a track predicted into the
     neighbouring sector misses its plot that rotation. The spec accepts that.

### 4.3 D26 — `CoordinatedTurn`, with the turn rate in the state

`tracking/models.py` gains `CoordinatedTurn`: state `(x_m, xdot_mps, y_m, ydot_mps,
turn_rate_radps)`, the exact discretisation of §3.6, its analytic Jacobian for the EKF, and the
series forms near zero rate. The rate is the heading's, positive to the right, the convention of
the scene's arcs. Because the model is nonlinear, `TaylorLinearisation.predict` moves the mean
through `motion.transition` and the covariance through its Jacobian; on every linear model the
mean is still `F x` to the bit. A known-turn-rate variant is not added: with the rate in the
state the EKF and UKF estimate it, and a fixed-rate bank is the IMM of a lesser model.

### 4.4 D27 — IMM as functions over a bank with per-mode columns

`tracker-001.md` names an `Estimator` protocol and `predict_to`; the code's filters are
functions over a `TrackBank`, and the IMM is too, in `tracking/imm.py` (`filters.py` would
otherwise pass R10's 800 lines):

- `TrackBank` gains `mode_mean` `(n_tracks, n_modes, n_state)`, `mode_covariance` and
  `mode_probability` `(n_tracks, n_modes)`, `n_modes` = 0 for a one-model bank, and its `mean`
  and `covariance` are the mixture's. Per-mode columns, not a wrapper bank per mode: every column
  keeps the track as its first axis, so `compact`, `append` and the lifecycle need no change and
  a row stays one track.
- `InteractingMultipleModel(motions, sojourn_time_s, absent_sigma)` holds the modes, the union
  layout and the generator; `switching_matrix(dt_s)` is `expm(Λ T)` for each row's own step,
  never a fixed per-scan matrix.
- `imm_mix`, `imm_predict`, `imm_innovate` (the combined estimate gates, its log-likelihood the
  mode-mixed one, so GNN, JPDA and gating work unchanged) and `imm_update` (hard or weighted
  association alike, through `filters.update_weighted`).
- `Tracker(motion=InteractingMultipleModel(...))` takes the IMM path; mode probabilities go to
  `tracks.csv`. The modes may be KF, EKF or UKF; A2 runs a one-mode IMM over each.

### 4.5 D28 — JPDA, JPDA\*, Set-JPDA and JIPDA

1. **Interface.** The `Associator` protocol gains `weights(log_likelihood, in_gate, *,
   clutter_intensity, existence=None, joint=True)`, returning `AssociationWeights` (weights
   `(n_tracks, n_meas + 1)`, the last column "none"; posterior existence; each cluster's tracks,
   measurements and event count) or None for a rule that only assigns. GNN and greedy return
   None and keep their hard-assignment path bit for bit. `JointProbabilisticDataAssociation`
   returns weights; the tracker then corrects through `filters.update_weighted`, beside the
   unchanged `update`.
2. **Clusters, then exact events.** Tracks are clustered by shared gated measurements (connected
   components of the validation matrix), and each cluster's feasible events are enumerated
   exactly. If a cluster's events exceed `max_events_per_cluster`, the associator raises,
   naming the cluster's size; it never approximates silently. The cluster sizes are a reported
   intermediate.
3. **Coalescence.** `coalescence="none"` is textbook JPDA; `"jpda_star"` keeps, of the events
   that differ only by which detected track took which measurement, the likeliest [3]; `"set"`
   moves the group's probability onto that likeliest labelling [4].
4. **Existence.** JIPDA [5] propagates `existence` on each `TrackBank` row. `ExistencePolicy`
   replaces M-of-N when given; existence is the JPDA's per-track detection weight `P_D r`.
5. **Clutter intensity.** `Scan.clutter_intensity` per measurement, from the oracle or the
   `ClutterMap` (D25), which `PlotTracker` keeps as policy.

### 4.6 Amendments to `spec/tracker-001.md`

Applied in the same change:

1. §2's table and §3–§5 name what the code has: `TrackBank` and the functions over it instead
   of `StateEstimate`, `GaussianFilter`, `predict_to` and `innovation_statistics`; `Scan`'s rows
   instead of `Measurement`; `gate` and `chi_square_gate_threshold` instead of `ChiSquareGate`;
   `Initiator` (`tracking/initiation.py`) instead of `TrackInitiator`; `jacobian(state, dt_s)`
   and `process_noise` instead of `matrices(dt_s)`; no `check_motion_model`.
2. §5.4: `ExistencePolicy` and `max_position_sigma_m` are in the code.
3. §6: IMM and JPDA move from extension points to built (`tracking/imm.py`,
   `tracking/association.py`); MHT and PMBM point to 011b.
4. §7: GOSPA and trajectory GOSPA replace OSPA as the multi-target score.

### 4.7 Workstream boundary

```
config/scenario_011_{manoeuvre,crossing,terminal_dense,iq_check}.toml
spec/scenario-011-terminal-area-tracking.md, spec/scenario-011b-mht-pmbm.md
src/radar_forge/tracking/models.py         (CoordinatedTurn)
src/radar_forge/tracking/filters.py        (bank columns, update_weighted, Taylor mean)
src/radar_forge/tracking/imm.py            (new)
src/radar_forge/tracking/initiation.py     (moved from tracker.py, unchanged)
src/radar_forge/tracking/association.py    (PDA, JPDA, JPDA*, set, JIPDA weights)
src/radar_forge/tracking/tracker.py        (ExistencePolicy, Coverage, weighted path, IMM path)
src/radar_forge/tracking/metrics.py        (GOSPA, trajectory GOSPA)
src/radar_forge/tracking/scene.py          (new)
src/radar_forge/stages/tracking.py         (new: the multitarget stage)
src/radar_forge/{scenario,pipeline,plots}.py, stages/__init__.py   (additive)
```

Scenario 003's `[verify]` tables and fingerprints staying unchanged is a precondition for every
commit.

---

## 5. Build Sequence

| Step | Module | Notes |
| :--- | :--- | :--- |
| 1 | `tracking/models.py`, `filters.py` | `CoordinatedTurn`, the Taylor mean; A1 and the Jacobian against central differences |
| 2 | `tracking/imm.py` | A2 before any multi-target code |
| 3 | `tracking/association.py`, `filters.update_weighted` | A3 (reductions to KF and PDA), A18 |
| 4 | `tracking/tracker.py` | Existence, coverage, the weighted and IMM paths; scenario 003 unchanged |
| 5 | `tracking/metrics.py` | A9 (GOSPA hand-computed cases) first |
| 6 | `tracking/scene.py` | A6, A7; the arcs regenerate §2.1 (A8) |
| 7 | `scenario.py`, `pipeline.py`, `stages/tracking.py` | The plot-level path; V1, V2, V4 configs |
| 8 | V5 | 001 S3 chain with placed pairs; fit `α_r` (A13, A14) |

No new dependency: `scipy.linalg.expm`, `scipy.optimize.linear_sum_assignment`,
`scipy.optimize.linprog` (HiGHS) and `scipy.sparse.csgraph.connected_components` cover the
matrix exponential, every assignment, the trajectory metric and the clusters.

---

## 6. Acceptance Criteria

Unit criteria live in `tests/tracking/test_{models,imm,filters,association,tracker,metrics,scene}.py`;
scenario criteria in the `[verify]` tables of `config/scenario_011_*.toml`, run by
`tests/test_scenarios.py`; Monte Carlo criteria in `slow` tests beside them.

| # | Criterion | Bound | Justification | Applies to |
| :--- | :--- | :--- | :--- | :--- |
| A1 | **CT reduces to CV.** At ω = 0 `CoordinatedTurn`'s F and Q are CV's on the shared coordinates, to the bit; at ω = 1e-9 F differs from CV's by its first-order term (±ωT²/2 across axes, ±ωT on velocity); F is continuous across `\|ωT\| = 1e-6` | exact; `rtol = 1e-10`; `rtol = 1e-10` | Catches a 0/0 or a sign in the series branch. (The first version asked F at 1e-9 to equal CV's at rtol 1e-10, which no entry zero in CV can meet: the turn term there is a real 1.2e-8) | unit |
| A2 | **One-mode IMM is its mode.** For KF, EKF and UKF modes, a one-mode IMM's state, covariance and likelihood equal the bare filter's over 50 scans | `rtol = atol = 1e-10` | Mixing with one mode is the identity | unit |
| A3 | **JPDA reductions.** (a) Pd = P_G = 1, no clutter in the gates, gates disjoint: JPDA equals the KF per track. (b) One track with clutter: JPDA equals PDA [1] | `rtol = 1e-10` | Exact algebra | unit |
| A5 | **IMM is consistent and switches.** V1's A4 alone, 200 trials at T = 4.8 s, from 100 to 300 s: time-averaged position NEES of the IMM inside its 95 % chi-square interval; the turn mode's probability passes 0.5 within 5 scans of the first turn's onset in ≥ 90 % of trials | as stated | Three scans is 391 m of departure, 8 innovation σ; the first version asked 3 scans "to confirm": measured, 37 % of trials within 3, 87.5 % within 4, 97 % within 5 | V1, `slow` |
| A6 | **Clutter rate.** Plot counts in each V4 component inside the 99 % Poisson interval of the stated intensity × volume, over 125 scans | 99 % | Catches a per-m² versus per-m·rad error | unit |
| A7 | **Merge model.** Over 10 000 seeded pairs at each `d` ∈ {0.5, 1, 2}, the merged fraction is within the 99.9 % Clopper–Pearson interval of `exp(−d²/2)` | 99.9 % | The model is what V3 sweeps | unit |
| A8 | **Scene is the scene.** The §2.1 encounter table and the §3.1 notch fractions regenerate from the V4 config | min `d` to 0.01; times to 0.1 s; fractions to 0.001 | The spec's numbers must be the code's | unit |
| A9 | **GOSPA by hand.** Three hand-computed cases (one missed, one false, one localisation error) | exact | Catches an α or c^p slip | unit |
| A11 | **IMM's benefit.** V1's A4, 200 trials: position RMSE through A4's two turns, IMM < min(CV quiet, CV agile) at T = 1.2, 2.4 and 4.8 s | strict | The teaching point (§1 item 1). The first version also asked the ratio IMM / best CV to fall from 1.2 to 4.8 s: measured, it rises, 0.83, 0.86, 0.92, the agile CV keeping up better at a long revisit | V1, `slow` |
| A12 | **Sector batching.** Halving the sector changes V4's mean GOSPA by less than one standard deviation across 20 seeds | as stated | The 0.15 s stamp error is negligible | V4, `slow` |
| A13 | **Plot model honest: rates.** V5's measured Pd and false-alarm rate inside the 99 % intervals of the values the plot model is given (Pd 0.99 for these targets; the design Pfa over the cells away from the pairs) | 99 % | D25 item 1 | V5, `slow` |
| A14 | **Plot model honest: merging.** V5's fraction of frames in which a pair gives one detection, at four separations, against the plot model's `P_u (1 − (1 − p)²) + (1 − P_u) 2 p (1 − p)`, `P_u = exp(−(Δr/α_r)²/2)`, p the measured Pd, at `α_r` fitted by maximum likelihood | each inside its 99 % Clopper–Pearson interval | The resolution model is the CST model. The first version compared the fraction with `P_u` alone; at 6 bins, where `P_u` is 1e-4, one detection is a missed partner, 2 p (1 − p) | V5, `slow` |
| A16 | **Coalescence shown (JPDA part).** V3's point `d` = 1 across range, `n_gate` = 1, 200 trials of 40 scans, tracks seeded on the pair: the median over trials of the coalescence ratio (mean estimated over true separation, last 20 scans) is lower under JPDA than under JPDA\* and under Set-JPDA | strict | The teaching point (§1 item 2). The first version bounded the mean ratio, JPDA ≤ 0.8 and the fixes ≥ 0.9: measured, the means are 1.35, 1.53 and 1.42 and the medians 1.14, 1.40 and 1.24, because in `n_gate` = 1 clutter a quiet CV track loses the pair in some trials and wanders off; the ordering survives, the bounds do not | V3 point, `slow` |
| A18 | **No silent truncation.** A cluster above `max_events_per_cluster` raises `ValueError` naming its size | raises | D28 item 2 | unit |
| A20 | **Markdown.** This spec passes `scripts/check_conventions.py` R7 | hook | | spec |

A4, A10, A15, A17, A19 and A16's MHT clause are 011b's.

---

## 7. Limitations

- **Two dimensions.** Plots report horizontal range; altitude is neither measured nor tracked.
  At 25 km, two stack levels 300 m apart differ in slant range by 16 m (at 1.2 and 1.5 km) to
  26 m (at 2.0 and 2.3 km): under one σ_r. So the simplification creates none of the closely
  spaced cases.
- **No range rate in the plot.** Association rests on position alone, the harder case and the one
  this scenario studies.
- **The radar is a class, not an installation**, and **the runway geometry is representative**,
  not surveyed (§3.3).
- **No secondary radar.** Transponder replies would remove most of the association problem.
- **Exact JPDA is exponential.** `max_events_per_cluster` bounds it and raises above the bound;
  covariance-based deletion keeps confirmed gates from swallowing a rain cell.
- **Births are inhibited in dense clutter.** No track starts where λ(z) > 0.01 per m·rad: a
  target entering the rain or the ground residue keeps a track it already has, but one that
  appears there is not found until it leaves.
- **Set-JPDA is restricted to weights.** The tracker carries one Gaussian per track over its own
  measurements, so `"set"` relabels within that representation; the full relabelled mixture of
  [4] is not built.
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
       models," *IEEE Trans. Aerosp. Electron. Syst.*, 39(4), pp. 1333–1364, 2003, §V.
.. [8]–[15] Track-oriented MHT and PMBM: see `scenario-011b-mht-pmbm.md`.
.. [16] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
        and Navigation*, Wiley, 2001, §11.6.6 (the IMM estimator).
.. [17] D. F. Crouse, "On implementing 2D rectangular assignment algorithms," *IEEE Trans.
        Aerosp. Electron. Syst.*, 52(4), pp. 1679–1696, 2016.
.. [18] W. Koch and G. van Keuk, "Multiple hypothesis track maintenance with possibly unresolved
        measurements," *IEEE Trans. Aerosp. Electron. Syst.*, 33(3), pp. 883–892, 1997.
.. [19] F. E. Daum and R. J. Fitzgerald, "Importance of resolution in multiple-target tracking,"
        *Proc. SPIE* 2235, Signal and Data Processing of Small Targets, 1994.
.. [20] D. Mušicki and M. R. Morelande, "Finite resolution multitarget tracking," *Proc. SPIE*
        5913, Signal and Data Processing of Small Targets, 2005.
.. [21] A. S. Rahmathullah, Á. F. García-Fernández and L. Svensson, "Generalized optimal
        sub-pattern assignment metric," *Proc. 20th Int. Conf. Information Fusion*, 2017;
        arXiv:1601.05585, Definition 1, eq. (1), and Proposition 1.
.. [22] Á. F. García-Fernández, A. S. Rahmathullah and L. Svensson, "A metric on the space of
        finite sets of trajectories for evaluation of multi-target tracking algorithms," *IEEE
        Trans. Signal Process.*, 68, pp. 3917–3928, 2020; arXiv:1605.01177, Proposition 2,
        eq. (23).
.. [23] D. Schuhmacher, B.-T. Vo and B.-N. Vo, "A consistent metric for performance evaluation
        of multi-object filters," *IEEE Trans. Signal Process.*, 56(8), pp. 3447–3457, 2008.
.. [24] M. I. Skolnik (ed.), *Radar Handbook*, 3rd ed., McGraw-Hill, 2008, ch. 2 (MTI radar,
        whose examples are airport surveillance radars).
.. [25] K. Granström, P. Willett and Y. Bar-Shalom, "Systematic approach to IMM mixing for
        unequal dimension states," *IEEE Trans. Aerosp. Electron. Syst.*, 51(4), pp. 2975–2986,
        2015, §III, eq. (10).
.. [26] D. Mušicki, R. Evans and S. Stanković, "Integrated probabilistic data association,"
        *IEEE Trans. Autom. Control*, 39(6), pp. 1237–1241, 1994.
.. [27] Y. Bar-Shalom, F. Daum and J. Huang, "The probabilistic data association filter," *IEEE
        Control Syst. Mag.*, 29(6), pp. 82–100, 2009.
