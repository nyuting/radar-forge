# Scenario 010 — Array self-calibration: the perturbed manifold, calibration sources and robust beamforming

## 1. Status & Purpose

**Status: specified, not implemented.** The last of the array slices. It depends on scenario 006
(the array frame, the steering vector "scenario-006 D13", `array/doa.py`), scenario 007
(`array/adaptive.py` and its `AdaptiveWeights` record, D16) and scenario 009 (per-channel
impairments and pilot-tone calibration, D21). It assumes all three exist.

Every array estimator in 006–009 trusts a steering vector `a(u)`. That vector is a model, and the
model is wrong in three ways a pilot tone cannot see: the elements are not where the drawing puts
them, they couple to their neighbours, and whatever lies between an element and 009's injection
point (radome, feed, cable) has its own gain and phase, which drift with temperature. This slice
puts all three into one object, `ArrayManifold` (D22). It estimates them from signals from sources
already on the air or deployed for the purpose, keeps those estimators apart from the beamformers
that use their output (D23), and feeds the refined manifold to the robust beamformers.

Three teaching points, each a test in §6:

1. **What the data cannot tell you comes first.** Self-calibration has exact ambiguities:
   translation, rotation, a dilation peculiar to linear arrays, and a phase ramp across a ULA that
   looks exactly like a steering error. Every estimator computes its Fisher information before it
   solves, and if the problem is not identifiable it says so instead of returning an answer (§2.2).
2. **A known direction is only as good as the source's reported position.** The scenario-002
   transmitter is surveyed to about a metre at 19.6 km, which is 51 µrad. An aircraft's reported
   position is good to tens of metres at 5–17 km. With 30 m of reported-position error the
   position estimate is 27–47 times worse than its Cramér–Rao bound (§4.5).
3. **Calibrate and also be robust.** With the target in the training data, MPDR on the nominal
   manifold loses a mean **29.97 dB** of SINR. Calibration alone still leaves **2.51 dB**. The
   robust Capon beamformer, with its uncertainty set sized to the calibration residual, loses
   **0.69 dB**. The same RCB on the *nominal* manifold loses **19.76 dB**, because its uncertainty
   set is wide enough to take in a strong interferer half a beamwidth from the target. Neither
   step is enough without the other.

**"Arrival phase."** In this scenario it means the **arrival phase** of a calibration source
across the channels: the phase, channel by channel, of a signal that reaches the array straight
from a source at a known position. Relative to channel 0, that phase is `2π(p_n − p_0)·u/λ`
plus whatever the manifold adds. With `u` known, what remains is the calibration. The rest of
this spec uses the term in this sense.

| Section | What it fixes |
| :--- | :--- |
| §1.1–1.2 | Upstream methods, and which calibration method does what |
| §2 Overview & Geometry | The site, the sources, the perturbation layers, identifiability |
| §3 Parameter Matrix | Every number, the variants M1–M3, the contracts, the output schema |
| §4 Core Decisions | D22, D23, the identifiability rule, the re-pointed boresight, what is not built |
| §5 Build Sequence | What is new, what is widened, what is untouched |
| §6 Acceptance Criteria | Bounds against the CRB and the closed-form optimum SINR, with the baselines failing |

### 1.1 Upstream inspiration

| Upstream | What it is | What radar-forge takes | Licence | Baseline or current |
| :--- | :--- | :--- | :--- | :--- |
| Friedlander & Weiss (1991) [1] | DOA with unknown mutual coupling and gain/phase: model `x = G·C·A(θ)s + n`, banded symmetric Toeplitz `C` for a ULA, alternating minimisation of a noise-subspace cost | The perturbation model, the alternating DOA / coupling / gain steps, each step a closed-form constrained quadratic | paper, written from equations | **Current** for coupling (M2) |
| Weiss & Friedlander (1990) [2] | Eigenstructure DOA with unknown sensor gain and phase | The gain/phase step inside [1]'s alternation | paper, written from equations | Current (the gain/phase half of M2) |
| Rockah & Schultheiss (1987) [3]; Weiss & Friedlander (1989) [4] | Array-shape calibration from sources at unknown locations: identifiability (one sensor position plus one direction fixed, three sources) and the ML estimator | The identifiability statement, and the reference-element and reference-direction gauge | paper, written from equations | Oracle for identifiability; the shape ML is not built (§4.7) |
| Wan, Tang, Zhu & Zhang (2014) [5] | Hybrid-CRB identifiability of array-shape self-calibration | The result that a nominally linear array is not self-calibratable under small errors, which §2.2 derives as a dilation | paper, written from equations | Oracle |
| Ng & See (1996) [6] | ML calibration of gain, phase, coupling and positions from sources in **known** directions, with CRBs | Known-direction position calibration (M1) as weighted least squares on the arrival phases, and its CRB | paper, written from equations | **Current** for known sources (M1) |
| Ye, Dai, Xu & Wu (2009) [7] | ULA coupling: the middle subarray sees coupling as a direction-dependent scalar | The banded-Toeplitz model's consequence used as a check (A5) | paper, written from equations | Reference |
| Yang (1995) [8] | Projection-approximation subspace tracking (PAST), O(N·r) per snapshot | Rank-one PAST on the scenario-002 transmitter's line-of-sight signal, for online drift (M3) | paper, written from equations | **Current** for drift (M3) |
| Viberg & Ottersten (1991) [9] | Subspace fitting; weighted subspace fitting (WSF) | The framing: M2's cost is a noise-subspace fit; WSF weighting is the recorded upgrade path (§4.7) | paper, written from equations | Reference |
| Dai, Hu, Xu & Chang (2015) [10] | Sparse Bayesian learning with mutual coupling | Listed as the sparse alternative; not built (§4.7) | paper, written from equations | Reference |
| Li, Stoica & Wang (2003) [11] | Robust Capon beamforming: spherical steering uncertainty, exact diagonal loading from `ε` | `robust_capon_weights` | paper, written from equations | **Current** |
| Vorobyov, Gershman & Luo (2003) [12] | Worst-case performance optimisation, an SOCP | `worst_case_weights`, solved through its Lagrangian as a scalar search (§3.5) | paper, written from equations | **Current** |
| Frost (1972) [13]; Cox, Zeskind & Owen (1987) [14] | LCMV; white-noise-gain constraint as robustness | `robust_lcmv_weights` ("robust Frost"): refined-manifold constraints plus a WNG floor | paper, written from equations | Current (Frost alone is the baseline) |
| Van Trees (2002) [15] | MPDR versus MVDR, mismatch, CRB | The MPDR/MVDR distinction (§4.6) and the SINR-loss definitions | book, written from equations | Oracle |
| doatools.py (M. Wang) | Python DOA toolbox: array designs carry perturbation objects (location, gain, phase, mutual coupling), CRBs | The pattern: perturbations attached to the array object, not the estimator. Last commit 2018 | MIT | Pattern only |

### 1.2 Calibration methods, classified

| Method | Sources | Estimates | Fixes ambiguity by | Role here |
| :--- | :--- | :--- | :--- | :--- |
| Pilot tone (009) | injected reference | channel gain, phase, delay up to the injection point | channel 0 | Upstream of this slice; sees nothing beyond the injection point |
| Known-direction LS / ML [6] | ≥ 3 known, linearly independent directions | element positions (3-D) | reference element only | **M1, current** |
| Gain/phase eigenstructure [2] | unknown directions | gain, phase, DOAs | channel 0 **and** a phase-slope gauge or one known direction | M2, inside [1] |
| Coupling + gain/phase, alternating [1] | ≥ 2 unknown (§2.2), + 1 known for absolute DOAs | `c_1…c_P`, gain, phase, DOAs | channel 0, `c_0 = 1`, the scenario-002 transmitter | **M2, current** |
| Shape self-calibration [3], [4] | ≥ 3 unknown | positions, DOAs | reference element + reference direction; **not enough for a ULA** (dilation) | Identifiability oracle only |
| Subspace tracking [8] | one known, continuously present | channel response vs time | channel 0 | **M3, current** |
| Nominal manifold, no calibration | — | nothing | — | **Baseline** throughout |

---

## 2. Overview & Geometry

The site is scenario 001's Duke Receiver, with the shared 16-element ULA of scenario 006. Two
kinds of calibration source are already in the repository:

- **The transmitter** at Raleigh-Durham (scenario 002), fixed at a surveyed position. Its
  line-of-sight signal is a calibration source of known direction.
- **The aircraft** of `data/flight_coordinates.csv`, whose reported positions (ADS-B-style) make
  its echo a source of known, but less well known, direction.

```
                                 N (0 deg)
                                 |
      W ---------------- [ Duke Receiver ] ---------------- E
                    16-element ULA, d = lambda/2 = 15.2955 mm
                    array +x (right, looking out) points to 275 deg
                        /        |         \
                       /         |          \
                      /          |           \
     aircraft track              boresight    Raleigh-Durham transmitter
     166.4-238.6 deg true        185.0 deg    134.99 deg true, el -0.19 deg
     array az -18.6 .. +53.6     (§4.4)       array az -50.01 deg, 19.601 km
     el 3.7-15.7 deg                          surveyed: 1 m -> 51 urad
```

What the slice builds, end to end:

```
  ArrayManifold (TRUTH)                     scenarios/scenario_010_*.toml
  geometry + delta, C, g(t)    <---------   [impairments] [[calibration_source]]
        |  rx_response_linear(u)  (scenario-006 D13 item 7)
        v
  core/signal.py  -> cube (n_pulses, n_samples, n_rx)  -> calibration snapshots
        |                                         (transmitter; D14 cells; beacons)
        v
  array/calibration.py   identifiability (FIM)  ->  NonIdentifiableError, or
        |                known_direction_positions (M1)
        |                coupling_gain_alternating (M2)
        |                track_channel_response    (M3, PAST)
        v
  ManifoldEstimate  .manifold : ArrayManifold (ESTIMATE)  + CRB, gauge, diagnostics
        |                                                     (D23: no data touched)
        v  steering_vectors(u_target)
  array/adaptive.py   robust_capon_weights | worst_case_weights | robust_lcmv_weights
        |             -> AdaptiveWeights (scenario-007 D16)
        v
  apply_weights  ->  SINR loss vs the closed-form optimum, per frame
```

Truth and estimate are the **same type**. A test compares them field by field, and the synthesis
uses exactly the object the estimator would have produced, so there is no second code path that
could drift.

### 2.1 The perturbation model, in layers

```
  a_true(u) = diag(g) · C · a(u ; p + delta)

  a(u ; p)   geometric steering vector, scenario-006 D13:  a_n = exp(+j 2 pi p_n.u / lambda)
  delta      element position error, (n_rx, 3) m: in-plane (x, y) and out-of-plane (z)
  C          mutual coupling, (n_rx, n_rx): banded symmetric Toeplitz for a ULA,
             C[m, n] = c_|m-n| for |m-n| <= P, else 0;  c_0 = 1
  g          per-channel complex response, (n_rx,): what 009's pilot cannot see, plus drift
```

The order is physical: the wave meets the elements where they really are (`δ`), the elements
exchange energy (`C`), and then each channel scales what it receives (`g`). `C` is symmetric and
not Hermitian, because coupling between reciprocal elements is reciprocal. An early prototype
built `C` with `scipy.linalg.toeplitz(c)`, which returns a *Hermitian* matrix when given one
argument, and the alternating estimator converged to the wrong answer with a non-zero cost. The
contract (§3.5) calls `toeplitz(col, col)`, and A3 checks `C == Cᵀ`.

### 2.2 Identifiability, stated once

Each line is a null direction of the Fisher information, counted numerically (normalised singular
values below 1e-10; the smallest non-null value in every case below is at least 2.3e-5) for the
16-element ULA with sources at array azimuths −30°, 20° and 26° (M2) or the M1 sources:

| Ambiguity | Why | Null dimensions | Fixed by |
| :--- | :--- | ---: | :--- |
| Translation | A common shift of every element adds `k·u·t` to each source's phase, which the unknown source phase absorbs | 3 (2 in plane) | Reference element: `δ_0 = 0` |
| Rotation (unknown directions) | Rotating the array and every source together changes nothing | 1 in plane | A reference direction (element 0 → 1) **or** a known source |
| **Dilation (linear arrays only)** | A ULA measures only `x_n·sin θ`, so `x_n → (1+a)·x_n` with `sin θ_k → sin θ_k/(1+a)` is exact. The null vector computed has `δx_n ∝ x_n` | 1 | A second known direction, or one known direction plus the reference direction. One known source alone leaves the rotation–dilation combination that keeps its direction fixed (1 null) |
| Common complex gain | `g → α·g` is absorbed into the source amplitudes | 1 (complex) | Channel 0: `g_0 = 1`; and `c_0 = 1` |
| **Phase ramp (ULA)** | `g_n → g_n·e^{jnβ}` is exactly a shift of every source by `β/π` in `u`. With symmetric coupling the ramp is only weakly observable: smallest normalised singular value 3.2e-5, condition number 7.5e4 | 1 exact (gain/phase only); weak (with `C`) | One known direction (the transmitter), or a declared gauge: zero mean phase slope, DOAs then reported relative |
| Too few sources | One unknown source cannot separate `g`, `C` and its own direction | 4 (M2, K = 1) | K ≥ 2 unknown sources |
| Out-of-plane `δz` with sources near el = 0 | `δz` enters only as `δz·sin el` | weak | Elevated sources (the aircraft at 10.8° and 15.7°) |

With every M1 direction known and the reference element fixed, the 3-D FIM has no null; its
condition number is 7.2e3, set by `δz` (smallest normalised singular value 3.5e-4). M2 with the
transmitter as anchor has condition number 93.7. Wan et
al. [5] show that large position errors make a perturbed linear array self-calibratable in
principle. The FIM here is evaluated at the current estimate, so that case is detected as
ill-conditioned rather than assumed away.

---

## 3. Parameter Matrix

### 3.1 Site, array and sources

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Site | Duke Receiver, 36.00250 °N, 78.94100 °W, 60 m | Scenario-001 §3.1 |
| Array | ULA, `n_rx` = 16, `element_spacing_wavelengths` 0.5, phase centre at the origin | Scenario-006 D13 |
| `f0_hz`, λ, d | 9.8 GHz, **30.5911 mm**, **15.2955 mm** | Scenario-001 |
| Aperture `n_rx·d` | 0.24473 m; Fraunhofer distance `2D²/λ` **3.92 m** | Every source is far field |
| HPBW at broadside (untapered) | **6.346°**, 0.11075 in `u`; 0.1 HPBW = **0.6346°** | Adaptive beamformers run untapered; 009's 8.52° is the Taylor-tapered beam |
| Per-channel `gain_rx_dbi` | 20.0 | Scenario-006 §3.1 |
| `boresight_azimuth_deg` | **185.0** (not 006's 235.804; §4.4) | Specified |
| Scenario-002 transmitter | 35.87750 °N, 78.78750 °W, 25 m; **19.601 km**, bearing **134.99°**, el **−0.19°**; array az **−50.01°** | Computed with `core/geodesy.py` |
| Transmitter in band at 9.8 GHz | Scenario-002 B1's fiction, kept so the array is one hardware item across 006–010 | Scenario-002 §4.5 |
| Transmitter line-of-sight SNR | 30 dB per channel per sample, `n_snapshots` 256 per frame | Specified: its antenna pattern towards Duke is unknown |
| Line of sight | 4/3-Earth radio horizons sum to 52.5 km, against 19.6 km | Computed; terrain ignored |
| Aircraft | `data/flight_coordinates.csv`, 1500 m AMSL, 10 dBsm, Swerling 0 | Scenario-001 |
| Aircraft calibration snapshot | One D14 peak cell per epoch; per-channel SNR `35.08 + 40·log₁₀(22.010 km/R)` dB | Scenario-006 §3.3, scaled as R⁻⁴; element-pattern roll-off not modelled |
| Reported-position error | 0 (default) or 30 m per axis, 1σ | Specified, for §4.5 |

Over the full track the aircraft spans array azimuth −18.63° to +53.57° and elevation 3.65° to
15.66°. Over the default window (`start_time_s = 0`, 120 s) it spans +47.34° to +53.57°, range
16.64–18.36 km.

### 3.2 Perturbations

| Parameter | M1 | M2 | M3 | Model |
| :--- | :--- | :--- | :--- | :--- |
| `position_error_rms_m` (x, y, z, each) | 1.5296e-3 (0.05 λ) | 0 | 0 | Gaussian, seeded; relative to element 0 |
| `coupling_coefficients_linear` `[c_0, c_1, c_2]` | `[1, 0, 0]` | `[1, 0.2·e^{jπ/3}, 0.06·e^{−jπ/4}]` (−13.98 dB, −24.44 dB) | as M2, known | Banded symmetric Toeplitz, P = 2 |
| `gain_error_rms_db`, `phase_error_rms_deg` | 0, 0 | 0.5, 5.0 | 0, 0 at t = 0 | Beyond 009's injection point; channel 0 the reference |
| `phase_drift_rms_deg_per_k`, `gain_drift_rms_db_per_k` | — | — | 2.0, 0.05 | Per channel, `φ_n(t) = κ_n·ΔT(t)` |
| `temperature_offset_k` | — | — | 0 → 10 linearly over the 120 frames | Compressed: a diurnal swing in two minutes, so the test is short |

Over M3 the drawn drift reaches **14.35°** rms phase and **0.466 dB** rms gain at the last frame.
Thermal expansion over the same 10 K, at aluminium's ≈23 × 10⁻⁶ K⁻¹, moves the end element by
52.8 µm, **0.0017 λ**. That is why M3's drift is channel-only, and why one known direction is
enough to track it.

### 3.3 The three variants

| | **M1** `position_known_source` | **M2** `coupling_blind` | **M3** `online_drift` |
| :--- | :--- | :--- | :--- |
| Perturbation | Positions only | Coupling + gain/phase | Channel drift; coupling known |
| Calibration sources | Transmitter + aircraft at 3 epochs, all known | 3 calibration beacons at unsurveyed positions, array az −30°, 20°, 26° **unknown** to the estimator (SNR 20 dB, 200 snapshots, uncorrelated); transmitter as optional anchor | Transmitter, every frame |
| Estimator | `known_direction_positions` (weighted LS on arrival phases) | `coupling_gain_alternating` [1] | `track_channel_response` (PAST, β = 0.99) |
| Then | MPDR, RCB, worst case, robust LCMV toward the aircraft at t = 94 s | MUSIC on the refined manifold (006's `Music`) | MPDR and RCB, every frame |
| Teaching contrast | Nominal MPDR self-nulls; calibration alone is not enough | Nominal MUSIC bias is 21–28 σ but under 0.1 HPBW | A stale calibration decays into self-nulling |

M1's aircraft epochs are chosen for direction diversity: t = 13349 s (7.659 km, az −18.63°, el
10.80°, 53.42 dB), t = 13074 s (5.326 km, −9.34°, 15.66°, 59.73 dB) and t = 94 s (17.267 km,
+53.57°, 4.71°, 39.30 dB). The third is also M1's beamforming target.

**Interference for the SINR tests (M1, M3).** The transmitter's line-of-sight signal at INR 30 dB
per channel, and a second strong interferer 0.5 HPBW (0.0554 in `u`) inside the target's direction
at INR 20 dB. The target is in the training covariance at 10 dB per channel, which is what forming
the covariance over a DataChip's cells does (scenario-006 D14 item 2). SINR is evaluated in
closed form on the model covariance. Sample-support effects belong to 007.

### 3.4 Derived quantities

All computed by the scenario's prototype scripts, 300 Monte-Carlo draws of the perturbation where
a distribution is quoted.

| Quantity | Value |
| :--- | ---: |
| M1 position CRB, rms per axis (x, y, z) | **0.0410, 0.0472, 0.1477 mm** (0.00134, 0.00154, 0.00483 λ) |
| M1 LS estimator RMSE ÷ CRB, per axis | 0.986, 0.992, 0.991 |
| M1 with 30 m reported-position error: RMSE ÷ CRB | 26.7, 32.2, 47.1 |
| 30 m as an angle at 5.326 / 7.659 / 17.267 km | 0.323°, 0.224°, 0.100° |
| M1 uncertainty radius `ε` (95th percentile of ‖a_true − ā‖²), nominal / refined | 7.565 / 0.00206 (‖a‖² = 16) |
| M2 joint CRB on DOA (−30°, 20°, 26°), anchored | 0.00908°, 0.01137°, 0.01149° |
| M2 CRB with the manifold known | 0.00684°, 0.00862°, 0.00973° |
| M2 CRB on `c_1`, `c_2` (complex magnitude) | 0.00304, 0.00308 |
| M2 nominal MUSIC bias (asymptotic and 100-trial mean) | 0.059°, −0.320°, −0.244° = 6.5, 28.1, 21.2 × CRB; 0.050 HPBW at most |
| M2 pair separation | 6.0°, 0.096 in `u`, 0.87 HPBW |
| M3 PAST memory `1/(1 − β)` | 100 snapshots; drift per frame 0.121° rms |

SINR loss against the optimum, M1 (σ = 0.05 λ, interferer at 0.5 HPBW):

| Beamformer | Manifold | Mean | 5 % | 95 % | Max |
| :--- | :--- | ---: | ---: | ---: | ---: |
| MPDR (target in training) | nominal | **29.97 dB** | 27.08 | 33.00 | 34.55 |
| RCB, `ε` = 7.565 | nominal | **19.76 dB** | 14.70 | 23.84 | 25.34 |
| Robust LCMV, WNG ≥ 6.02 dB | nominal | 8.63 dB | 6.28 | 11.59 | 12.90 |
| MPDR | refined (CRB residual) | **2.51 dB** | 1.41 | 3.81 | 5.10 |
| Robust LCMV, WNG ≥ 6.02 dB, null on transmitter | refined | 2.45 dB | 1.50 | 3.16 | 3.34 |
| **RCB**, `ε` = 0.00206 | refined | **0.69 dB** | 0.33 | 1.18 | 1.90 |
| MVDR (target-free training) | nominal | 0.70 dB | 0.34 | 1.17 | 1.69 |
| MVDR (target-free training) | refined | 0.00 dB | 0.00 | 0.00 | 0.00 |

The last two rows explain the slice. Without the target in the training data, a nominal manifold
costs under 2 dB. The damage is done by mismatch **and** a target in the covariance together. The
worst-case beamformer with `‖Δa‖ ≤ √ε` gives weights collinear with RCB's (|cos| = 1.000000;
loading 51.99 against RCB's 51.87 in the grid-searched prototype).

M2 alternating estimator, anchored, 100 trials of 200 snapshots: DOA mean bias −0.0047°,
−0.0043°, −0.0013°; RMSE ÷ joint CRB 1.11, 1.08, 1.00. Coupling RMSE ÷ CRB 1.11 (`c_1`), 1.28
(`c_2`), largest error in any trial 0.0102. Phase RMSE 0.270°, 1.18 × CRB. Blind (gauge: zero
phase slope), asymptotic: every DOA shifted by a common −0.00069 to −0.00079 in `u`, against the
true phase slope's −0.00081; pair separation 5.992° against 6.000°.

M3, per frame (frames 10–119 for the maxima):

| | t = 0 | t = 30 | t = 60 | t = 119 | max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| MPDR, calibration frozen at t = 0 | 0.00 | 15.55 | 21.69 | **27.80 dB** | 27.80 |
| RCB, frozen, `ε` = actual drift × 1.1 (oracle) | 0.00 | 3.76 | 5.12 | 6.37 | 6.37 |
| MPDR, PAST-tracked | 0.20 | 0.21 | 0.19 | 0.18 | **0.42 dB** |
| RCB, PAST-tracked | 0.13 | 0.13 | 0.13 | 0.13 | **0.24 dB** |
| PAST channel-phase error, rms | 0.112° | 0.130° | 0.094° | 0.078° | 0.226° |

### 3.5 Mathematical contracts

Scenario-006 §3.6 and scenario-007 §3.5 hold unchanged.

| Contract | Statement |
| :--- | :--- |
| Manifold | `a_true(u) = diag(g)·C·a(u; p + δ)`, shape `(n_rx, n_directions)`; `u` is the D13 unit vector, `(n_directions, 3)`. The geometric factor is 006's `steering_vectors`, so the D13 sign lives in one place |
| Synthesis | `rx_response_linear = steering_vectors(u_paths).T`, shape `(n_paths, n_rx)`, as 006 D13 item 7 consumes it |
| Coupling | `C = scipy.linalg.toeplitz(col, col)`, `col = [c_0, …, c_P, 0, …]`, `c_0 = 1`. Symmetric, not Hermitian |
| Narrowband | As 006 D13 item 8: aperture transit time ÷ `1/B` = 1.53e-3. 009's group delay enters `g` as its band-centre phase |
| Arrival phase | For a known-direction source `k`, `ψ_{k,n} = arg(v_n·v_0*) − arg(a_n(u_k; p)·a_0(u_k; p)*)`, `v` the principal eigenvector of the source's sample covariance (or the single D14 snapshot). Then `ψ_{k,n}/k₀ = u_k·(δ_n − δ_0)` to first order, `k₀ = 2π/λ`, with no wrap while `|δ| < λ/2` (9.8 σ here) |
| Known-direction LS [6] | Per element: minimise `Σ_k w_k (u_k·δ_n − ψ_{k,n}/k₀)²`, `w_k = L_k·SNR_k`. Three unknowns per element; needs ≥ 3 linearly independent `u_k`. Equation numbers of [6] to confirm at implementation |
| M1 CRB | `F = 2·Σ_k L_k·SNR_k·Re(D_kᴴ P⊥_{a_k} D_k)`, `D_k = ∂a(u_k)/∂δ`, `P⊥_a = I − aaᴴ/N`: deterministic model, complex amplitude per snapshot a nuisance |
| M2 cost [1] | `J(g, c, θ) = Σ_k ‖Ê_nᴴ·diag(g)·C(c)·a(θ_k)‖²`, `Ê_n` the sample noise subspace, `N − K_total` columns |
| M2 steps | (i) DOA: MUSIC peaks with the current `diag(g)·C`. (ii) `c`: `C(c)a = Q(a)c`, `Q[m, p] = Σ_{|m−n| = p} a_n`, so `J = cᴴMc`; minimise with `c_0 = 1`: `c = M⁻¹e_0/(e_0ᵀM⁻¹e_0)`. (iii) `g`: `diag(g)Ca = diag(Ca)g`, the same form with `g_0 = 1` [2]. `M` is singular on noise-free data; solve with a relative load of 1e-9·tr(M)/n. Stop when `ΔJ/J < 1e-12` or at 200 iterations |
| Gauge (blind) | After each `g` step, remove the least-squares phase slope of `unwrap(arg g)`. Results are then **relative**: DOAs carry one unknown common `u` shift, and the result says so |
| Anchor | A known-direction source enters steps (ii)–(iii) with its `a` fixed and is excluded from step (i) |
| Stochastic FIM | `F_ij = L·Re tr(R⁻¹ ∂_iR R⁻¹ ∂_jR)`, central differences, source powers and noise known (an optimistic bound, stated in the result) |
| Identifiability | `F̃ = D^{-1/2}FD^{-1/2}`, `D = diag F`. Null dimension = #{σ_i/σ_max < 1e-10}: raise. `σ_min/σ_max < 1e-4`: weakly identifiable, a declared gauge or anchor required. M1 (3.5e-4) passes; blind M2 (3.2e-5) does not |
| PAST [8] | Rank one: `y = wᴴx`, `h = P·y`, `γ = h/(β + y*h)`, `P = (P − γh*)/β`, then `P ← (P + P*)/2`, `w ← w + (x − w·y)·γ*`. The Hermitian step is Yang's `Tri{·}`; the prototype without it diverged at frame 13. Channel response: `ĝ = w ⊘ (C·a(u_tx))`, `ĝ ← ĝ/ĝ_0` |
| SINR | `SINR(w) = σ_s²|wᴴa_s|²/(wᴴR_in w)`, `SINR_opt = σ_s²·a_sᴴR_in⁻¹a_s`, loss = `SINR_opt/SINR(w)`, with `a_s` the **true** steering vector |
| MPDR / MVDR | `w ∝ R⁻¹ā`, `R = R_in + σ_s²a_sa_sᴴ` (MPDR) or `R_in` (MVDR) [15] |
| RCB [11] | `â = ā − (I + λR)⁻¹ā` with `‖(I + λR)⁻¹ā‖² = ε`, solved on `R = VΓVᴴ` as a monotone scalar root in λ; `â ← √N·â/‖â‖`; `w = R⁻¹â/(âᴴR⁻¹â)`. Equivalent loading `1/λ`. Requires `ε < ‖ā‖²` |
| Worst case [12] | `min wᴴRw` s.t. `Re(wᴴā) − √ε‖w‖ ≥ 1`. The KKT condition is `(R + γI)w = μā`, `γ = μ√ε/‖w‖ ≥ 0`, so `w ∝ (R + γI)⁻¹ā`. γ by a bounded scalar minimisation of the scaled cost (`scipy.optimize`); no SOCP solver |
| Robust LCMV [13], [14] | `w = (R + γI)⁻¹C_c(C_cᴴ(R + γI)⁻¹C_c)⁻¹f`, `C_c = [â_s, â_tx]`, `f = [1, 0]`; γ the smallest load with `WNG = |wᴴâ_s|²/‖w‖² ≥ min_white_noise_gain` (monotone in γ) |
| Units | `u` dimensionless inside `array/`; angles in radians in code, `_deg` at file boundaries (data-001 DF2) |

### 3.6 Output schema

As scenario-006 §3.7, plus:

| Where | What |
| :--- | :--- |
| `products.h5` `/manifold_truth/` | `element_positions_m` (n_rx, 3), `coupling_matrix_linear` (n_rx, n_rx) complex128, `channel_gain_linear` (n_rx,) complex128; for M3, `channel_gain_linear` is `(n_frames, n_rx)` |
| `products.h5` `/manifold_estimate/` | The same fields, from `ManifoldEstimate.manifold`; attributes `method`, `gauge`, `null_space_dim`, `condition_number` |
| `products.h5` `/calibration/` | `fisher_information` (n_params, n_params), `parameter_names` (n_params,) str, `crb_std` (n_params,), `cost_history` (n_iterations,) |
| `beamformer.csv` | `frame`, `time_s`, `method` (`mpdr`, `mvdr`, `rcb`, `worst_case`, `robust_lcmv`), `manifold` (`nominal`, `refined`, `tracked`, `frozen`), `sinr_loss_db`, `white_noise_gain_db`, `loading_w` |
| `metadata.json` `receive_array` | As 006, with `boresight_azimuth_deg` = 185.0 |

---

## 4. Core Decisions

### 4.1 D22 — the perturbed manifold `a(θ, φ, d̂)` is one `ArrayManifold`

`array/manifold.py` holds a frozen dataclass:

| Field | Shape | Meaning |
| :--- | :--- | :--- |
| `geometry` | 006's `ArrayGeometry` | `element_positions_m` `(n_rx, 3)`: the true positions `p + δ` in a truth manifold, the believed `p + δ̂` in an estimate |
| `wavelength_m` | float | |
| `coupling_matrix_linear` | `(n_rx, n_rx)` complex128 | Identity when ideal |
| `channel_gain_linear` | `(n_rx,)` complex128 | Ones when ideal; 009's estimated gain drops in unchanged |

Methods: `steering_vectors(u)` → `(n_rx, n_directions)`, `rx_response_linear(u)` →
`(n_paths, n_rx)`, and `ideal(geometry, wavelength_m)`. A refined manifold is
`dataclasses.replace(nominal, …)`: there is no setter and no in-place correction.

Why one object. Position error, coupling and channel response all act on the steering vector, and
every consumer (synthesis, MUSIC, the beamformers, the CRB) needs the same composition in the same
order. Spreading them over three call sites is how a coupling matrix ends up applied twice, or
not at all, in one consumer. Why a new module rather than `geometry.py`: `ArrayGeometry` stays
006's pure geometry, and `manifold.py` composes it with the electrical terms. `impairments.py`
(truth side) and `calibration.py` (estimate side) both depend on `manifold.py`, and neither on the
other.

`array/impairments.py` gains `MutualCoupling` (`coefficients_linear` `(P+1,)`, `matrix(n_rx)`),
`PositionError` (`offset_m` `(n_rx, 3)`, `draw_position_error(n_rx, rms_m, rng)`), `ChannelDrift`
(`phase_rad_per_k`, `gain_db_per_k`, `channel_gain_at(temperature_offset_k)`) and
`perturb(manifold, *, position_error, coupling, channel)` → `ArrayManifold`. These are 009 D21's
explicit per-channel models, extended to the aperture.

### 4.2 D23 — calibration estimators are separate from the beamformers that use them

`array/calibration.py` estimators take snapshots and a prior `ArrayManifold` and return a
`ManifoldEstimate`. They never return weights or corrected data:

| Field | Meaning |
| :--- | :--- |
| `manifold` | The refined `ArrayManifold` |
| `parameter_names`, `fisher_information`, `crb_std` | The FIM at the estimate, and the bound it implies |
| `null_space_dim`, `condition_number`, `weakly_identifiable` | §3.5's identifiability report |
| `gauge` | What was fixed: `"reference_element"`, `"reference_channel"`, `"zero_phase_slope"`, `"anchor:<source>"` |
| `source_u` | Estimated directions of unknown sources, `(n_sources, 3)`, or `None` |
| `cost_history`, `n_iterations` | M2's convergence; one entry per frame for M3 |

Estimators: `known_direction_positions(snapshots_per_source, source_u, prior, *, snr_linear)`,
`coupling_gain_alternating(snapshots, prior, *, n_sources, n_coupling, anchors=(), gauge=…)`,
`track_channel_response(snapshot_stream, prior, transmitter_u, *, forgetting_factor)` (a
generator, one `ManifoldEstimate` per frame), and `fisher_information_*` for each model.

The beamformers in `array/adaptive.py` take a covariance and a **steering vector**. They do not
import `calibration.py` or `manifold.py`. The pipeline (`pipelines/calibration.py`) is the only
place the two meet: `estimate.manifold.steering_vectors(u_target)` is handed to
`robust_capon_weights`. So one estimate can be scored against several beamformers, one beamformer
against several estimates, and each tested alone. This is the same separation as 009 D21 and
007 D16.

The robust beamformers follow 007 D16 and return `AdaptiveWeights`, with new `method` values
`"rcb"`, `"worst_case"` and `"robust_lcmv"`:

- `robust_capon_weights(covariance, steering_vector, uncertainty_norm_sq_linear)`; diagnostics:
  λ, `â`, the equivalent loading in `loading_w`.
- `worst_case_weights(covariance, steering_vector, uncertainty_norm_linear)`; diagnostics: γ, the
  constraint margin.
- `robust_lcmv_weights(covariance, constraint_matrix, constraint_values,
  min_white_noise_gain_db)`; diagnostics: γ, the achieved WNG.

007's INCM reconstruction is **not** duplicated; a pipeline can hand an INCM covariance to any of
the three.

### 4.3 Identifiability is checked before solving

Every estimator computes its FIM at the initial point, and again at the solution. A null
dimension above zero after the declared gauge raises `NonIdentifiableError(ValueError)`, carrying
the null dimension and the null vectors in parameter names (A6–A9). A weakly identifiable problem
without a declared gauge or anchor raises the same error. Returning a least-norm or regularised
answer to a non-identifiable problem is how a self-calibration "converges" to a dilated array;
the error is the honest output.

### 4.4 Why this slice re-points the boresight to 185°

006 points the array at the lead aircraft (235.804°). From there the scenario-002 transmitter is
at −100.8°, behind the array face. At 185.0° the transmitter is at −50.01° and the default-window
aircraft at +47.34° to +53.57°, both in front, and the 104° spread between them is what M1 needs for direction diversity.
The hardware and D13 are unchanged; only `boresight_azimuth_deg` in this slice's TOMLs differs,
and `metadata.json` records it.

### 4.5 What a known direction buys, and what a reported position costs

With exact directions, the LS estimator meets the CRB (ratio 0.99). With 30 m reported-position
error per axis, the error is 27–47 times the CRB, and at the target the refined steering vector's
phase error rises to 6.07° rms. That is worse than not calibrating some elements at all. The
transmitter's 1 m survey is 51 µrad. The lesson is that the source's direction is part of the
error budget, and the high-SNR, close-range epochs, which carry the most information, are also
where a metre of position is the biggest angle. Averaging over many epochs brings the ratio down
(2.2× at 30 epochs, in the prototype), but it does not remove it. Treating the reported positions
as uncertain parameters with a prior is the principled fix; it is deferred (§4.7).

### 4.6 MPDR, not MVDR, is what self-nulls

The request names "Capon (MVDR)". Van Trees [15] separates MVDR (the covariance holds
interference and noise only) from MPDR (it also holds the target). Mismatch costs MVDR 0.70 dB
here and MPDR 29.97 dB. This slice therefore runs the target-in-training case, which is what a
covariance over a DataChip's cells is (006 D14 item 2), and reports both. 007 A12–A13 show
self-nulling from a *pointing* error. This slice shows it from a *manifold* error with perfect
pointing.

### 4.7 Deliberately not built

| Not built | Why |
| :--- | :--- |
| Shape ML from unknown sources [3], [4] | For this ULA it is not identifiable (§2.2, dilation); it is an identifiability oracle only |
| WSF weighting of M2's cost [9] | FW is already within 1.00–1.28× the joint CRB, so at most about 1.1 dB is left to gain. WSF is the recorded next step |
| SBL with coupling [10] | Needs no more sources but adds a hyperparameter loop; the FIM-checked alternating method meets the bound here |
| Gu & Leshem's steering-vector QCQP | 007 §4.6 deferred it here. RCB covers the same need in closed form, without a convex solver |
| Reported-position uncertainty as a prior | §4.5. A hybrid-CRB estimator [5] is the extension |
| Direction-dependent element patterns | Every element is assumed identical; a common pattern does not affect calibration |

### 4.8 Amendments to `spec/structure.md`

> **Applied.** These amendments are folded into `spec/structure.md`; it is the authoritative
> tree and decision list. This section records what changed and why.

1. **Part B tree:** add `array/manifold.py`; widen the comments of `array/impairments.py`,
   `array/calibration.py` and `array/adaptive.py`; add `pipelines/calibration.py`.
2. **B.1 rows** for the four modules.
3. **Decisions D22 and D23**, as §4.1–§4.2.

### 4.9 Workstream boundary

This scenario claims, and another workstream should not plan:

```
scenarios/scenario_010_{position_known_source,coupling_blind,online_drift}.toml
spec/scenario-010-self-calibration.md
src/radar_forge/array/manifold.py                 (new)
src/radar_forge/array/impairments.py              (widened: coupling, position error, drift)
src/radar_forge/array/calibration.py              (widened: self-calibration estimators, FIM)
src/radar_forge/array/adaptive.py                 (widened: RCB, worst case, robust LCMV)
src/radar_forge/pipelines/calibration.py          (new)
tests/array/test_manifold.py, tests/array/test_self_calibration.py,
tests/array/test_robust_beamforming.py, tests/pipelines/test_scenario_010.py
```

`array/{geometry,beamforming,doa}.py` (006), the D16 record and `apply_weights` (007), and 009's
pilot and IQ estimators are consumed, never edited. `core/{signal,dsp,geodesy}.py`,
`pipelines/trajectories.py` and the scenario-002 site constants are consumed. 001–009's tests
staying green is a precondition for every commit.

---

## 5. Build Sequence

```
  1. array/manifold.py           ArrayManifold; ideal(); steering_vectors via 006
             |
  2. array/impairments.py        MutualCoupling, PositionError, ChannelDrift, perturb()
             |                    009's ChannelResponse equivalence test (A2) first
  3. array/calibration.py        fisher_information_*; identifiability; NonIdentifiableError
             |                    the identifiability tests (A6-A9) before any estimator
  4. array/calibration.py        known_direction_positions (M1)
             |
  5. array/calibration.py        coupling_gain_alternating (M2), gauge and anchor
             |
  6. array/calibration.py        track_channel_response, PAST (M3)
             |
  7. array/adaptive.py           robust_capon_weights, worst_case_weights, robust_lcmv_weights
             |
  8. pipelines/calibration.py    snapshots -> estimate -> steering -> weights -> SINR
             |
  9. scenarios/*.toml, tests/pipelines/test_scenario_010.py
```

| Step | Module | Notes |
| :--- | :--- | :--- |
| 1 | `array/manifold.py` *(new)* | An ideal manifold's `steering_vectors` must equal 006's bit for bit (A1) |
| 2 | `array/impairments.py` *(widen)* | 009's per-channel half is untouched |
| 3 | `array/calibration.py` *(widen)* | The FIM functions come first: every estimator calls them |
| 4–6 | `array/calibration.py` *(widen)* | Each estimator is a page of NumPy; the M2 steps are three closed forms |
| 7 | `array/adaptive.py` *(widen)* | `AdaptiveWeights` and `apply_weights` unchanged |
| 8 | `pipelines/calibration.py` *(new)* | Synthesis uses `ArrayManifold.rx_response_linear`; `core/signal.py` is not edited beyond 006's widening |
| 9 | tests | `core/dsp.py` is called with `fast_time_axis=1` (006 D13) |

---

## 6. Acceptance Criteria

Encoded in `tests/array/test_manifold.py`, `test_self_calibration.py`,
`test_robust_beamforming.py` and `tests/pipelines/test_scenario_010.py`. Seeds are fixed. SINR
bounds are on the closed-form model covariance. Monte-Carlo tolerances are set from the
prototype's measured values, with the margin stated.

| # | Criterion | Bound | Why this tolerance | Applies to |
| :--- | :--- | :--- | :--- | :--- |
| A1 | Ideal `ArrayManifold.steering_vectors` equals 006's `steering_vectors` | bit-identical | The same function, composed with identities | unit |
| A2 | `channel_gain_linear = g` in the manifold equals 009's `apply_channel_response` with zero delay | `rtol = 1e-12` | Narrowband gain commutes with synthesis | unit |
| A3 | Coupling matrix is symmetric and banded | `C == Cᵀ` exactly; zero beyond `P` | Reciprocity (§2.1) | unit |
| A4 | Middle subarray (rows `P … N−1−P`) of `C·a(u)` equals `β(u)·a(u)` with `β = c_0 + 2Σ c_p cos(pπu_x)` | `rtol = 1e-12` | Exact for a banded symmetric Toeplitz ULA [7] | unit |
| A5 | **Identifiability, dilation.** Self-localisation, 3 unknown directions, reference element + reference direction | raises; `null_space_dim == 1`; null vector `δx ∝ x_n` to 1e-6 | §2.2; nulls ≤ 1e-14 against a smallest true value of 2.9e-4 | unit |
| A6 | One known direction only, in-plane positions | raises; `null_space_dim == 15` | §2.2 | unit |
| A7 | M2 with one unknown source | raises; `null_space_dim == 4` | §2.2 | unit |
| A8 | Gain/phase only, 3 unknown sources, no gauge | raises (`null_space_dim == 1`); with `gauge = "zero_phase_slope"` succeeds and reports it | §2.2 | unit |
| A9 | M2 blind with coupling and no gauge | raises as weakly identifiable (`σ_min/σ_max` = 3.2e-5 < 1e-4); M1's 3.5e-4 does not raise | §3.5 rule | unit |
| A10 | **M1 at the CRB.** LS position RMSE per axis, 300 trials, exact directions | ≤ 1.25 × √CRB (measured 0.99) | ML is efficient at this SNR; 300 × 15 elements gives under 3 % sampling error | M1 |
| A11 | M1 with 30 m reported-position error | RMSE ≥ 10 × √CRB (measured 27–47) | Documents §4.5: the source, not the SNR, limits it | M1 |
| A12 | **Baseline failing.** Nominal MPDR mean loss; nominal RCB (`ε` = 7.565) mean loss | ≥ 25 dB (29.97); ≥ 15 dB (19.76) | 5th percentiles 27.08 and 14.70 dB | M1 |
| A13 | Refined MPDR mean loss | ≥ 1.5 dB (2.51) | Calibration alone is not enough | M1 |
| A14 | **Current.** RCB on the refined manifold | mean ≤ 1.0 dB (0.69); max ≤ 2.5 dB (1.90) | 300 draws | M1 |
| A15 | MVDR, target-free, refined | mean ≤ 0.01 dB (0.00) | CRB-level residual | M1 |
| A16 | Worst-case and RCB weights for the same `ε`, solved exactly | `|cos| ≥ 1 − 1e-6` | Both are `(R + γI)⁻¹ā` with the same γ (§3.5); the prototype matched to 1e-6 with a grid | unit |
| A17 | Robust LCMV (WNG ≥ 6.02 dB) refined; nominal | max ≤ 3.5 dB (3.34); mean ≥ 6 dB (8.63) | 100 draws | M1 |
| A18 | **Baseline bias.** Nominal MUSIC on the pair, mean over 100 trials | ≥ 20 × √CRB on the 20°/26° pair (28.1, 21.2), although < 0.1 HPBW | The beamwidth criterion is too weak to see the bias; the CRB is not | M2 |
| A19 | **Current.** Anchored alternating estimator, 100 trials | mean bias < 0.1 HPBW (0.6346°; measured ≤ 0.0047°); RMSE ≤ 1.25 × √CRB (1.00–1.11) | Not ML, so efficiency is measured, not assumed | M2, `slow` |
| A20 | Coupling recovered | RMSE ≤ 1.5 × √CRB (1.11, 1.28); every trial `|ĉ_p − c_p|` ≤ 0.02 (max 0.0102) | ≈2× the largest observed error | M2, `slow` |
| A21 | Blind M2, asymptotic covariance | common `u` shift within 2e-4 of the true phase slope ÷ π (−0.00081); pair separation within 0.02° of 6° (0.008°) | The gauge moves every source together and nothing else | M2 |
| A22 | PAST channel-phase error, frames 10–119 | ≤ 0.3° rms (max 0.226°) | 100-snapshot memory against 0.121° drift per frame | M3 |
| A23 | **Baseline failing.** Frozen-calibration MPDR at the last frame | ≥ 20 dB (27.80) | The drift is deterministic per seed | M3 |
| A24 | **Current.** Tracked MPDR and tracked RCB, every frame 10–119 | ≤ 0.5 dB (0.42); ≤ 0.3 dB (0.24) | Margins about 20 % | M3 |

---

## References

.. [1] B. Friedlander and A. J. Weiss, "Direction finding in the presence of mutual coupling,"
       *IEEE Trans. Antennas Propag.*, 39(3), pp. 273–284, 1991. doi:10.1109/8.76322.
       (Equation numbers to confirm at implementation.)
.. [2] A. J. Weiss and B. Friedlander, "Eigenstructure methods for direction finding with sensor
       gain and phase uncertainties," *Circuits, Systems, and Signal Processing*, 9,
       pp. 271–300, 1990. doi:10.1007/BF01201215.
.. [3] Y. Rockah and P. M. Schultheiss, "Array shape calibration using sources in unknown
       locations — Part I: Far-field sources," *IEEE Trans. Acoust., Speech, Signal Process.*,
       35, pp. 286–299, 1987.
.. [4] A. J. Weiss and B. Friedlander, "Array shape calibration using sources in unknown
       locations — a maximum likelihood approach," *IEEE Trans. Acoust., Speech, Signal
       Process.*, 37(12), pp. 1958–1966, 1989.
.. [5] S. Wan, J. Tang, W. Zhu and N. Zhang, "Identifiability analysis for array shape
       self-calibration based on hybrid Cramér–Rao bound," *IEEE Signal Process. Lett.*, 21(4),
       pp. 473–477, 2014. doi:10.1109/LSP.2014.2306326.
.. [6] B. C. Ng and C. M. S. See, "Sensor-array calibration using a maximum-likelihood approach,"
       *IEEE Trans. Antennas Propag.*, 44(6), pp. 827–835, 1996.
.. [7] Z. Ye, J. Dai, X. Xu and X. Wu, "DOA estimation for uniform linear array with mutual
       coupling," *IEEE Trans. Aerosp. Electron. Syst.*, 45(1), pp. 280–288, 2009.
.. [8] B. Yang, "Projection approximation subspace tracking," *IEEE Trans. Signal Process.*,
       43(1), pp. 95–107, 1995.
.. [9] M. Viberg and B. Ottersten, "Sensor array processing based on subspace fitting,"
        *IEEE Trans. Signal Process.*, 39(5), pp. 1110–1121, 1991.
.. [10] J. Dai, N. Hu, W. Xu and C. Chang, "Sparse Bayesian learning for DOA estimation with
        mutual coupling," *Sensors*, 15(10), pp. 26267–26280, 2015. doi:10.3390/s151026267.
.. [11] J. Li, P. Stoica and Z. Wang, "On robust Capon beamforming and diagonal loading,"
        *IEEE Trans. Signal Process.*, 51(7), pp. 1702–1715, 2003.
.. [12] S. A. Vorobyov, A. B. Gershman and Z.-Q. Luo, "Robust adaptive beamforming using
        worst-case performance optimization: a solution to the signal mismatch problem,"
        *IEEE Trans. Signal Process.*, 51(2), pp. 313–324, 2003.
.. [13] O. L. Frost III, "An algorithm for linearly constrained adaptive array processing,"
        *Proc. IEEE*, 60(8), pp. 926–935, 1972.
.. [14] H. Cox, R. M. Zeskind and M. M. Owen, "Robust adaptive beamforming," *IEEE Trans.
        Acoust., Speech, Signal Process.*, 35(10), pp. 1365–1376, 1987.
        doi:10.1109/TASSP.1987.1165054.
.. [15] H. L. Van Trees, *Optimum Array Processing* (Detection, Estimation, and Modulation
        Theory, Part IV), Wiley, 2002.
