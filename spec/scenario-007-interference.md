# Scenario 007 — Strong RF interference against a weak target: adaptive nulling (SLC, MVDR, LCMV, mainlobe suppression)

## 1. Status & Purpose

**Status: specified, not implemented.** The first adaptive-array slice. It depends on scenario 006,
which builds the array geometry, the steering-vector convention (scenario-006 D13) and the
multi-channel cube; this slice assumes all three exist.

A small aircraft, with the radar cross-section of a small drone, sits on the beam axis of the Duke
Receiver's 16-element receive array. Strong wideband RF interferers (for example a communications
transmitter whose emission spills into the radar band) share its range and Doppler cells: each
fills *every* cell. Range and Doppler processing cannot separate them from the aircraft. Only the
array's spatial degrees of freedom can, and this slice is about how to spend them.

Three reasons to build it now:

1. **It turns the array into an instrument.** Scenario 006 estimates angles. This slice *uses*
   angles to remove power, and every adaptive method in `array/adaptive.py` is first exercised
   here against an analytic answer: the closed-form optimum SINR, the Reed–Mallett–Brennan loss
   distribution, and the exact null of the clairvoyant beamformer.
2. **It teaches the classical-to-current path in one place.** The sidelobe canceller and SMI-MVDR
   are the baselines. The constrained form (LCMV and its GSC realisation), diagonal loading,
   interference-plus-noise covariance reconstruction, and eigen-projection mainlobe suppression
   are each derived from them in a few lines, with the intermediate quantities exposed.
3. **Each variant has a baseline that fails visibly.** The quiescent beam loses the aircraft under
   a sidelobe interferer (I1). MVDR's beam peak walks off the aircraft when the interferer reaches
   the mainlobe edge (I2). The current method fixes each, and §6 tests both halves.

| Section | What it fixes |
| :--- | :--- |
| §1.1 Upstream inspiration | What each method is taken from, and which is baseline or current |
| §2 Overview & Geometry | The array, the aircraft, the two interference layouts, the processing chain |
| §3 Parameter Matrix | Every number, the budgets, the mathematical contracts and the output schema |
| §4 Core Decisions | D15, D16, the training-set rule, and what mainlobe suppression cannot do |
| §5 Build Sequence | What is new, what is widened, what is untouched |
| §6 Acceptance Criteria | Analytic bounds, the baseline failing, and the end-to-end detection test |

### 1.1 Upstream inspiration

| Upstream | What it is | What radar-forge takes | Licence | Baseline or current |
| :--- | :--- | :--- | :--- | :--- |
| Howells (1965) [1]; Applebaum (1976) [2] | Sidelobe canceller: auxiliaries weighted to cancel the main channel's interference; the analog loop settles to the Wiener solution | The steady-state Wiener SLC, as weights on the main beam and auxiliary taps | paper, written from equations | **Baseline** |
| Capon (1969) [13]; Reed, Mallett & Brennan (1974) [6] | MVDR, and its sample-matrix-inversion form with the Beta-distributed SINR loss | SMI-MVDR; the RMB loss as a test oracle | paper, written from equations | **Baseline** and oracle |
| Frost (1972) [3]; Griffiths & Jim (1982) [4]; Buckley & Griffiths (1986) [5] | Linearly constrained minimum variance, its unconstrained GSC realisation, and derivative constraints | Closed-form LCMV, the GSC split into quiescent + blocking + adaptive parts, a derivative constraint for peak maintenance | paper, written from equations | **Baseline** (LCMV/GSC); the derivative constraint is used inside the current I2 method |
| Carlson (1988) [7] | Diagonal loading of the sample covariance | `loading_w` on every SMI-type method; sample support ≈ 2× interference rank | paper, written from equations | **Current** practice for short training |
| Gu & Leshem (2012) [8] | Interference-plus-noise covariance reconstruction (INCM) from the Capon spectrum outside the signal sector | The reconstruction step; it removes self-nulling when the aircraft is in the training data | paper, written from equations | **Current** |
| Yang, Zhang, Zeng, Long & Sarkar (2014) [9] | Eigen-projection preprocessing (EMP) with covariance reconstruction (CMR) for mainlobe interference | Project the mainlobe interferer out, reconstruct the covariance, then adapt with a mainlobe-peak constraint | paper, written from equations | **Current** |
| Phased-Array-Antenna-Model | NumPy array-pattern library (`structure.md` §A.3), `beamforming.py` with null steering and adaptive weights | Cross-check of quiescent patterns and deterministic null steering only | MIT | Cross-check |

The mainlobe method in I2 is a composition: EMP-CMR as published [9], followed by an LCMV
derivative constraint [5] at the look direction. §4.5 says why the composition is needed and where
it stops working. The steering-vector estimation half of Gu & Leshem [8] (a quadratically
constrained programme) is **not** built; §4.6 records why.

---

## 2. Overview & Geometry

The radar is scenario-001's S2 pulsed X-band radar at the Duke Receiver, with its receive antenna
replaced by the shared 16-element ULA (scenario-006 D13). Geometry is fixed and synthetic so that
every quantity has an analytic value. Angles are `azimuth_boresight_deg` (data-001 DF2): from
array boresight, positive towards the array's +x.

```
                                   boresight (0 deg)
                                         |
                                         |  small aircraft
                                         |  0 deg, 5516.18 m
                                        [*]  -30 dBsm, +19.418 m/s
                                         |
                                         |       mainlobe-edge interferer (I2)
                                         |    .  +4.5 deg, 40 km
                                         |  .      (3 dB half-width 3.180 deg,
                                         |.         first null 7.181 deg)
                                         |                   sidelobe wideband
                                         |               .   interferer (I1, I2)
                                         |          .        +26 deg, 40 km
                                         |     .
                                         v  .
        ===o==o==o==o==o==o==o==o==o==o==o==o==o==o==o==o===   x ->
           0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
             16-element ULA, d = lambda/2 = 15.2955 mm, aperture n_rx*d = 0.24473 m
                  [ Duke Receiver, 36.00250 N, 78.94100 W, 60 m AMSL ]
```

### 2.1 Why only space separates them

| Source | Range cells it occupies | Doppler cells it occupies | What separates it from the aircraft |
| :--- | :--- | :--- | :--- |
| Thermal noise | all | all | nothing; the floor |
| Wideband interferer (I1, I2) | **all**: broadband, continuous | **all**: white across pulses | angle only |
| Small aircraft | range sample 92 | Doppler bin 13 | — |

A wideband noise-like interferer is the pure case: it fills the cell under test and every training
cell, so the only lever is the array.

### 2.2 The chain

```
  scenarios/scenario_007_interference_*.toml
          |  [target] fixed geometry   [[noise_interferer]]
          v
  core/signal.py        line_of_sight_paths ------> PropagationPaths (two-way, aircraft)
  core/interference.py  noise_interferer_sources -> NoiseInterfererSources (one-way power, aoa)  D15
          |
          v
  pulsed_baseband(..., rx_response_linear) (006 D13) + thermal_noise + interferer_noise
          |   cube (n_pulses, n_samples, n_rx) = (256, 100, 16)
          v
  core/dsp.py  matched_filter(axis=1) -> doppler_fft(axis=0)    <-- UNCHANGED (006 D13)
          |   data_rv (n_doppler_bins, n_range_bins, n_rx)
          v
  array/adaptive.py   training_snapshots (guard cells excluded)                         D16
          |           -> AdaptiveWeights: slc | smi_mvdr | lcmv | gsc | incm | emp_peak
          |              (weights + covariance + diagnostics; nothing applied)
          v
  array/adaptive.py   apply_weights  y = w^H x   (the only place weights touch data)
          |
          v
  core/detection.py   cfar_detect_2d (Pfa = 1e-6)   <-- UNCHANGED
          |
          v
  detections.csv, interference.csv, products.h5 /beam
```

### 2.3 Baseline and current, per variant

| Variant | Interference | Baseline (expected to fail) | Current method |
| :--- | :--- | :--- | :--- |
| **I1** `sidelobe` | wideband interferer at +26° | quiescent beam (uniform or Taylor −30 dB); SLC and SMI-MVDR shown as baselines that *succeed* at K = 2N | SMI with loading; LCMV/GSC with mainlobe constraints; INCM when the aircraft is in the training set |
| **I2** `mainlobe-edge` | wideband interferers at +4.5° and +26° | SMI-MVDR (beam peak walks 1.4–1.5° off the aircraft); SLC (destroys the beam) | EMP-CMR + peak-maintenance constraint |

---

## 3. Parameter Matrix

### 3.1 Radar, array and channels

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Site | Duke Receiver, 36.00250 °N, 78.94100 °W, 60 m | scenario-001 §3.1 |
| Waveform | scenario-001 **S2** unchanged: LFM pulse, `chirp_duration_s` = 10 µs, `bandwidth_hz` = 2.0 MHz, `prf_hz` = 25 kHz, `sample_rate_hz` = 2.5 MHz, `n_pulses` = 256, `f0_hz` = 9.8 GHz | scenario-001 S2; its matched filter and Doppler FFT give the processing gains of §3.4 |
| `transmit_power_w`, `gain_tx_dbi` | 1 kW, 30.0 dBi (separate transmit aperture, on boresight) | scenario-001 S2 |
| `noise_figure_db` | 3.0 dB, every channel | scenario-001 |
| Receive array | 16-element ULA on x, `element_spacing_m` = λ/2 = **15.2955 mm**, aperture `n_rx·d` **0.24473 m**, phase centre at the origin | shared ground array, scenario-006 D13 and §3.1 |
| Per-channel `gain_rx_dbi` | **20.0** dBi, flat over ±60° azimuth; beamformed **32.04 dBi** | scenario-006 §3.1 (a λ/2-wide column); the flat azimuth pattern is this slice's assumption |
| SLC auxiliaries | channels 7 and 8 (the two central ULA elements), tapped in parallel with the beamformer | §4.4 |
| `n_rx` | **16**. Adaptive weights act on all 16 channels | |
| Frame model | 1 Hz frames, **fixed geometry**; each frame is an independent noise realisation | analytic truth; §6 averages over frames |

### 3.2 The aircraft

| Parameter | Value | Why |
| :--- | :--- | :--- |
| `rcs_dbsm` | −30.0 (0.001 m²), Swerling 0 | a small aircraft, the size of a small drone |
| `range_m` | **5516.18 m** = 92 × c/(2·f_s), range sample 92 exactly | inside R_ua = 5995.8 m and outside the 1499.0 m eclipse; an exact bin (testing.md §3) |
| `radial_velocity_mps` | **+19.4182 m/s** closing = 13 Doppler bins of 1.4937 m/s exactly | no straddle loss |
| `azimuth_boresight_deg`, `elevation_deg` | 0.0, 0.0 | on the steered look direction |

### 3.3 Interferers per variant

| | **I1** `sidelobe` | **I2** `mainlobe-edge` |
| :--- | :--- | :--- |
| Wideband interferer A | 100 W, 20 dBi (ERP 10 kW = 40.00 dBW), `bandwidth_hz` 20 MHz (in-band fraction 0.100), 40 km, **+26°** | as I1 |
| Wideband interferer B | — | identical to A, at **+4.5°** (past the 3 dB half-width 3.180°, inside the first null 7.181°) |
| Main beam | uniform (SLC, MVDR quiescent); Taylor −30 dB as the low-sidelobe quiescent baseline | uniform |

### 3.4 Derived budgets

All computed by script from `core/radar_equation.py`'s `received_power_w` and the one-way link of
§3.5; quoted verbatim. "Per element, processed" means after the matched filter (gain BT = 20,
13.01 dB) and the 256-pulse Doppler FFT (24.08 dB), which raise a coherent signal by 37.09 dB and
leave white noise and the wideband interferer where they were.

| Quantity | Value |
| :--- | ---: |
| λ | 30.5911 mm |
| Noise per element, kT₀FB | 1.5978e-14 W |
| Range sample / velocity bin | 59.958 m / 1.4937 m/s |
| Unambiguous range / velocity | 5995.8 m / ±191.194 m/s |
| **Uniform beam**: 3 dB beamwidth, first null, peak sidelobe | **6.359°**, **7.181°**, **−13.15 dB** (at 10.313°; −1.11 dB relative to one element) |
| **Taylor −30 dB beam**: 3 dB beamwidth, first null, peak sidelobe | 8.069°, 10.843°, −30.05 dB (−18.70 dB relative to one element); SNR efficiency −0.689 dB |
| Aircraft: received power per element | 5.0934e-17 W |
| Aircraft: SNR per element per pulse | −24.97 dB |
| Aircraft: SNR per element, processed | 12.13 dB |
| Aircraft: SNR after the array, no interference (N·SNR) | **24.17 dB** |
| Interferer A: received power per element | 3.7038e-10 W |
| Interferer A: **INR per element** | **43.65 dB** |
| I1: **I/S per element per pulse** (interferer power over aircraft power) | **68.62 dB** |
| I1: I/S per element, processed (signal gains 37.09 dB, interference does not) | 31.52 dB |
| I1: uniform pattern at +26° | −20.15 dB re peak (Taylor: −30.46 dB) |
| I1: **quiescent SINR**, uniform / Taylor | **−11.38 dB** / −1.08 dB |
| I1: **optimum SINR**, closed form | **24.13 dB**; SINR loss **−0.0422 dB** |
| I2: optimum SINR (interferers at +4.5° and +26°) | 23.02 dB; SINR loss −1.15 dB |
| I2: \|ρ\|² between look and +4.5° | −6.59 dB |
| Pd, Swerling 0, Pfa = 1e-6, at quiescent I1 SINR, uniform / Taylor | 2.239e-06 / 6.552e-05 |
| Pd at the optimum I1 SINR / at the RMB 0.1 % quantile (18.71 dB) | 1.0000 / 1.0000 |

Two things the table teaches. The interferer is 68.62 dB above the aircraft before processing and
still 31.52 dB above it after: integration does not help against noise that fills every cell.
And a −30 dB Taylor beam, a very good deterministic sidelobe level, lifts the quiescent SINR only
from −11.38 dB to −1.08 dB. Low sidelobes alone do not detect the aircraft; adaptation does.

### 3.5 Mathematical contracts

Scenario 001 §3.5 holds. Steering vectors, frame and weight application are scenario-006 D13:
`a_n(θ) = exp(+j·2π·x_n·sin θ/λ)` at zero elevation with `x_n = (n − 7.5)·d`, unnormalised,
`y = wᴴx`. What this slice adds:

| Contract | Statement |
| :--- | :--- |
| One-way link | `P = ERP · min(1, B_rx/B_j) · G_rx · λ² / ((4π)² R²)`, Friis [11]. `one_way_received_power_w` in `core/radar_equation.py`. Doubling the range costs 6 dB, not 12 |
| Snapshot | For Doppler bin d, range bin r: `x[d, r] = data_rv[d, r, 0:16]`, shape `(n_rx_adaptive,)` = `(16,)`. A training set is `(n_snapshots, 16)`, snapshots first |
| Sample covariance | `R̂ = Xᴴ X / K`, X of shape `(K, 16)`; loaded form `R̂ + δI`, `δ` = `loading_w` [7] |
| SINR | `SINR(w) = σ_s² \|wᴴ a_s\|² / (wᴴ R_in w)`, with `R_in = σ²I + Σ_j σ_j² a_j a_jᴴ` per element, processed |
| Optimum SINR | `σ_s² a_sᴴ R_in⁻¹ a_s`. One interferer: `N·SNR·[1 − N·INR·\|ρ\|²/(1 + N·INR)]`, `ρ = a_jᴴ a_s / N` [10] (equation number to confirm at implementation) |
| SINR loss | `L(w) = SINR(w) / SINR_opt`, in `[0, 1]` |
| RMB | With K iid target-free snapshots, `L(w_SMI) ~ Beta(K − N + 2, N − 1)`, mean `(K − N + 2)/(K + 1)` [6]. N = 16, K = 32: mean **0.5455 (−2.632 dB)**, std 0.0854, 0.1 % quantile 0.2868 (−5.42 dB) |
| Clairvoyant null | Exact MVDR response at the interferer, relative to look: `\|ρ\|² / (1 + N·INR·(1 − \|ρ\|²))²`. I1: **−131.45 dB** re look (quiescent −20.15 dB) |
| SMI-MVDR | `w = R̂⁻¹ a₀ / (a₀ᴴ R̂⁻¹ a₀)` [13]; the closed form lives in `array/beamforming.py` (006) |
| SLC | `y = w_qᴴ x − w_auxᴴ x_aux`, `w_aux = R_aa⁻¹ r_am`, `r_am = E[x_aux y_main*]` [1], [2]. Reported as the full-length weight `w = w_q − E w_aux`, E selecting channels 7 and 8 |
| LCMV | `w = R⁻¹ C (Cᴴ R⁻¹ C)⁻¹ f` [3]. I1: `C = [a(0), a(−δ), a(+δ)]`, `f = [1, g, g]`, δ = 1.590° (a quarter of the 3 dB width), g = 0.92121 (the uniform beam's own response at ±δ) |
| GSC | `w = w_q − B w_a`, `w_q = C(CᴴC)⁻¹f`, `BᴴC = 0`, `w_a = (BᴴRB)⁻¹ BᴴR w_q` [4]. Must equal LCMV |
| INCM | `R_in = Σ_{θ_i ∈ Θ̄} a(θ_i) a(θ_i)ᴴ / (a(θ_i)ᴴ R̂⁻¹ a(θ_i)) · Δθ`, Θ̄ = {\|θ\| > 5°}, Δθ = 0.1° in radians; `w ∝ R_in⁻¹ a₀` [8] (equation number to confirm at implementation) |
| EMP-CMR | (1) eigen-decompose R̂; (2) dominant subspace U_s: eigenvalues above 10× the median; (3) mainlobe interferer angle θ_m = argmax over \|θ\| ≤ 7.181° of `‖U_sᴴ a(θ)‖²/N`, accepted if > 0.9; (4) `B = I − a_m a_mᴴ`, `a_m = a(θ_m)/√N`; (5) `R_rec = B R̂ B + σ̂² a_m a_mᴴ`, σ̂² the mean of the non-dominant eigenvalues; (6) adapt on `R_rec` and apply to `Bx` [9] |
| Peak maintenance | LCMV on `R_rec` with `C = [B a₀, B ȧ₀]`, `f = [1, 0]`, `ȧ_n = j(2π x_n cos θ/λ) a_n` at θ = 0 [5]. Effective weight `w_eff = B w`. The derivative constraint makes θ = 0 a stationary point of the pattern |
| Detection | Swerling 0: `Pd = Q₁(√(2·SNR), √(−2 ln Pfa))`, Marcum Q [12] (section to confirm at implementation) |
| Main-beam normalisation | Where a beam pattern is quoted relative to one element, the beam is scaled to unit noise gain, `wᴴw = 1` |

### 3.6 Output schema

As scenario-001 §3.6, plus:

| File / group | Content |
| :--- | :--- |
| `interference.csv` | One row per interferer per frame: `frame`, `interferer_id`, `kind` (`noise`), `azimuth_boresight_deg`, `elevation_deg`, `range_m` (true, one-way), `erp_w`, `received_power_w` (per element), `inr_db` (per element, unprocessed) |
| `products.h5` `/beam/{method}/burst{k}/data` | Beam output `(n_frames, n_doppler_bins, n_range_bins)` complex128, one group per method (`quiescent`, `slc`, `smi_mvdr`, `lcmv`, `incm`, `emp_peak`); attrs `n_snapshots`, `n_guard_cells`, `loading_w` |
| `metrics.csv` | `sinr_loss_db`, `null_depth_db`, `peak_shift_deg` per method per frame window |

`truth.csv` lists the aircraft only. Interferers are not targets, so they live in
`interference.csv`.

---

## 4. Core Decisions

### 4.1 D15 — interferers are one-way sources, in `PropagationPaths`-compatible form

An interferer emits; it does not scatter. Its power falls as 1/R² over one leg, and the radar's
transmit gain and the target's RCS play no part. `core/interference.py` represents it in the
names, units and angle convention the existing generators already use:

| Kind | Representation | Field readings |
| :--- | :--- | :--- |
| Wideband noise interferer | **`NoiseInterfererSources`**: `power_w` `(n_sources,)`, `aoa_rad` `(n_sources, 2)`, `range_m` `(n_sources,)` | the same names, units and angle convention as `PropagationPaths`; `power_w` is what `abs(amplitude_linear)**2` would be. No delay or Doppler: the source is stationary noise, so it has none |

`interferer_noise(sources, element_positions_m, wavelength_m, shape, rng)` is the sibling of
`thermal_noise`: it returns `(n_pulses, n_samples, n_rx)` spatially coloured noise
`Σ_j a(θ_j) s_j[m, k]`, each `s_j` complex white Gaussian of variance `power_w[j]`, so the cube
covariance is exactly `σ²I + Σ_j σ_j² a_j a_jᴴ`. The power comes from one function,
`one_way_received_power_w` in `core/radar_equation.py`, which is the link-budget half that file
already promises.

`PropagationPaths` itself does not change; D1 and D6 stand.

### 4.2 D16 — adaptive weights are returned, never applied silently

Every function in `array/adaptive.py` that adapts returns an `AdaptiveWeights` and never touches
the cube:

| Field | Meaning |
| :--- | :--- |
| `weights` | `(n_rx_adaptive,)` complex128, to be used as `y = wᴴx` |
| `method` | `"slc"`, `"smi_mvdr"`, `"lcmv"`, `"gsc"`, `"incm"`, `"emp_cmr"`, `"emp_peak"` |
| `covariance` | the matrix actually inverted: after loading, reconstruction or projection |
| `n_snapshots`, `loading_w` | the training support and the diagonal load |
| `constraint_matrix`, `constraint_values` | C and f, or `None` |
| `diagnostics` | method-specific intermediates: eigenvalues, the dominant rank, `mainlobe_angle_rad`, the blocking matrix, `w_q` and `w_a` for GSC, `w_aux` for SLC |

`apply_weights(weights, data)` is the only place `y = wᴴx` happens. The reasons: a reader can
print and plot the weights and their pattern before anything is beamformed; tests can assert on
weights (constraints, nulls) without synthesising data; and the same weights can be applied to a
second cube, which is how the self-nulling demonstration is built.

An SMI with fewer snapshots than channels and no loading raises `ValueError` rather than returning
a pseudo-inverse: a rank-deficient covariance is a configuration error, and loading is the named
fix.

### 4.3 Training-set selection, and why the cell under test is excluded

`training_snapshots(data_rv, doppler_bin, range_bin, n_guard_cells, n_train_cells)` takes
snapshots from the **same Doppler bin**, `n_train_cells` = 16 on each side in range (K = 32 = 2N),
skipping `n_guard_cells` = 2 on each side of the cell under test, and the cell itself. Near the edge
of the 75 usable range cells (1.5–6.0 km) the window slides inward and K stays 32. It returns the
`(K, 16)` snapshots and the index mask, so the selection is visible.

Including the cell under test is the classic error, and §6 demonstrates it: the aircraft's own
snapshot enters R̂, and MVDR, minimising output power under a unit look-direction constraint, puts
a partial null on it. At K = 32, the aircraft alone (12.13 dB per element, processed) costs a mean
**6.54 dB** of SINR with perfect pointing and **9.57 dB** with a 1° pointing error. A 10 dB
stronger aircraft costs 25.47 dB. INCM (§3.5) removes the effect, because it rebuilds the
covariance only from directions outside the signal sector.

The matched filter at f_s/B = 1.25 correlates adjacent range snapshots slightly, so the cube's
training set is not quite iid. RMB is therefore asserted on iid snapshots drawn from the exact
covariance (§6 A6), and the end-to-end test asserts detection, not the Beta law.

### 4.4 SLC auxiliaries

**SLC auxiliaries are element taps, not new antennas.** Channels 7 and 8 are read before the
beamformer, so the SLC is a two-degree-of-freedom partially adaptive array on the same cube. Each
auxiliary's gain is the element's, above every sidelobe of the uniform beam (by 1.11 dB at the
first sidelobe). Central taps leave the aircraft's response almost unchanged (+0.218
dB); end taps (0, 15) cost −1.290 dB, which is why the centre is chosen.

### 4.5 What mainlobe suppression can and cannot do

A null inside the mainlobe necessarily reshapes the mainlobe. Two regimes, computed on the exact
covariance:

| Interferer at | Method | Beam peak | Look gain re own peak | White-noise gain at look | SINR at 0° / −1° / +1° |
| :--- | :--- | ---: | ---: | ---: | :--- |
| +4.5° (past −3 dB) | MVDR | **−1.428°** | −0.72 dB | 10.89 dB | 23.02 / 23.67 / 21.56 dB |
| +4.5° | EMP-CMR [9] | −1.450° | −0.73 dB | 10.89 dB | 22.94 / 23.61 / 21.48 dB |
| +4.5° | **EMP-CMR + peak constraint** | **+0.000°** | 0.00 dB | 8.72 dB | 20.76 / 20.24 / 20.23 dB |
| +2.5° (inside −3 dB) | MVDR | −2.828° | −3.61 dB | 7.32 dB | 19.45 / 21.68 / 15.37 dB |
| +2.5° | EMP-CMR + peak constraint | +7.181° (the mainlobe splits) | −11.45 dB | −2.21 dB | 9.88 / 8.31 / 8.28 dB |

Three lessons, each asserted in §6:

1. **At high INR, EMP-CMR alone is MVDR.** Projection is the infinite-INR limit of the MVDR
   null, so the projected beam inherits MVDR's peak shift. Its gain is a deeper, estimation-robust
   null on the mainlobe interferer (SMI median −72.5 dB against MVDR's −67.9 dB) and a clean
   covariance for the sidelobe interferers. That is why I2 follows EMP-CMR with a derivative
   constraint at the look direction [5].
2. **Peak maintenance costs gain, and buys a flat mainlobe.** Holding the peak at 0° costs 2.17 dB
   of white-noise gain against MVDR, and returns a response that varies by 0.53 dB over ±1°
   instead of 2.11 dB. A beam whose peak is where the steering says is what monopulse angle
   estimation and the tracker downstream need.
3. **Inside the 3 dB beamwidth nothing works.** With the interferer at +2.5° the constrained
   stationary point at 0° is a *minimum*; the peaks move to the first nulls. No linear weight
   both nulls 2.5° and keeps a single beam at 0°. The spec records this as a negative result
   rather than tuning it away.

The mainlobe sector for EMP's angle search is the first-null half-width, ±7.181°. An earlier draft
classified eigenvectors by their own beam peak; with two interferers of equal power the dominant
eigenvectors are arbitrary mixtures of both steering vectors, and both were classed as mainlobe.
The subspace search of §3.5 step 3 does not depend on how the eigenvectors happen to mix.

### 4.6 Deliberately not built

| Not built | Why |
| :--- | :--- |
| Gu & Leshem's steering-vector estimation (QCQP) | Needs a convex solver beyond `numpy`/`scipy`; the reconstruction half carries the teaching point. Belongs with the robust beamformers of scenario 010 |
| The Howells–Applebaum and Frost *iterative* loops | They converge to the closed forms built here; LMS convergence is a separate lesson |
| Cross-polarised or terrain-scattered interference | No polarisation or multipath model exists yet |

### 4.7 Amendments to `spec/structure.md`

> **Applied.** These amendments are folded into `spec/structure.md`; it is the authoritative
> tree and decision list. This section records what changed and why.

1. **Part B tree.** Add `core/interference.py  # wideband noise interferer sources, one-way (D15)`
   and `array/adaptive.py  # SLC, SMI-MVDR, LCMV/GSC, loading, INCM, EMP-CMR, peak maintenance; returns weights (D16)`.
   Widen `core/radar_equation.py`'s comment to "… link budget, one-way (Friis) link for interferers".
2. **B.1 mapping.** `core/interference.py` ← Friis (1946).
   `array/adaptive.py` ← Howells (1965), Applebaum (1976), Frost (1972),
   Griffiths & Jim (1982), Buckley & Griffiths (1986), Reed–Mallett–Brennan (1974),
   Carlson (1988), Gu & Leshem (2012), Yang et al. (2014);
   Phased-Array-Antenna-Model `beamforming.py` as a cross-check.
3. **Decisions D15 and D16**, as §4.1 and §4.2.

### 4.8 Workstream boundary

This scenario claims:

```
scenarios/scenario_007_interference_{sidelobe,mainlobe_edge}.toml
spec/scenario-007-interference.md
docs/scenarios/scenario-007-interference.md
src/radar_forge/core/interference.py
src/radar_forge/array/adaptive.py
src/radar_forge/core/radar_equation.py   (one_way_received_power_w only)
src/radar_forge/pipelines/scenarios.py   ([[noise_interferer]], fixed-geometry [target])
tests/pipelines/test_scenario_007.py, tests/array/test_adaptive.py, tests/core/test_interference.py
```

`array/geometry.py`, `array/beamforming.py` and the multi-channel generators belong to scenario
006 and are **consumed**; MVDR's closed form stays there. `core/{dsp,windows,detection}.py` are
consumed unchanged. Robust beamformers (RCB, worst-case) join `array/adaptive.py` in scenario 010
and must reuse `AdaptiveWeights`.

---

## 5. Build Sequence

```
  1. core/radar_equation.py   + one_way_received_power_w            (Friis; 6 dB per range doubling)
             |
  2. core/interference.py     + NoiseInterfererSources, noise_interferer_sources,
             |                  interferer_noise                    (D15)
  3. array/adaptive.py        + AdaptiveWeights, apply_weights, sample_covariance,
             |                  training_snapshots                 (D16; §4.3)
  4.                          + smi_mvdr (wraps 006's mvdr_weights), slc_weights, lcmv_weights,
             |                  gsc_weights; loading_w on all       (baselines first)
  5.                          + incm_covariance                     (Gu & Leshem)
  6.                          + eigen_projection, emp_cmr_weights, peak_maintained_weights
             |
  7. pipelines/scenarios.py   [[noise_interferer]], fixed-geometry [target]
             |
  8. scenarios/scenario_007_interference_*.toml  (two)
             |
  9. tests: unit tests per step, then tests/pipelines/test_scenario_007.py
```

| Step | Module | Notes |
| :--- | :--- | :--- |
| 1 | `core/radar_equation.py` *(edit)* | Additive. `received_power_w` and `bistatic_received_power_w` untouched |
| 2 | `core/interference.py` *(new)* | Consumes `PropagationPaths` and `thermal_noise`'s conventions; adds no field to either |
| 3–6 | `array/adaptive.py` *(new)* | One function per step of §3.5, each returning its intermediate quantities. Baselines are built and tested before the current methods that are derived from them |
| 7 | `pipelines/scenarios.py` *(edit)* | TOML order: `[scenario] [target] [[noise_interferer]] [[burst]] [receiver_site] [receive_array] [receiver] [adaptive]`: the truth, the emitters, the transmit end, the receive end, then processing. `[receive_array]` is 006's table unchanged; `[target]` without `[trajectory]` selects the fixed geometry |
| 9 | tests | `core/dsp.py`, `core/detection.py` consumed unchanged; `test_scenario_001.py` and `test_scenario_006.py` stay green throughout |

---

## 6. Acceptance Criteria

Unit criteria live in `tests/array/test_adaptive.py` and `tests/core/test_interference.py`;
end-to-end ones in `tests/pipelines/test_scenario_007.py`, marked `slow`. Monte Carlo criteria use
fixed seeds and assert against a stated confidence band, never a loosened tolerance (testing.md
§10.9). Criterion numbers are stable because other specs cite them; A2 and A20–A23 are unused.

| # | Criterion | Bound | Justification | Variant |
| :--- | :--- | :--- | :--- | :--- |
| A1 | One-way budget: INR per element, aircraft SNR, I/S | 43.65 dB, 24.17 dB, 68.62 dB; `rtol = 1e-12` on the linear values against the closed forms | a handful of float64 operations | I1 |
| A3 | Quiescent patterns: uniform 3 dB width 6.359°, first null 7.181°, PSL −13.15 dB; Taylor 8.069°, 10.843°, −30.05 dB | ±0.001° (pattern grid 0.0005°), ±0.01 dB | analytic patterns | all |
| A4 | Optimum SINR: matrix form equals the one-interferer closed form; loss −0.0422 dB | `rtol = 1e-10` | linear algebra | I1 |
| A5 | **Clairvoyant null**: MVDR response at +26° equals `\|ρ\|²/(1 + N·INR·(1 − \|ρ\|²))²` = −131.45 dB re look | `rtol = 1e-6` on the amplitude | a cancellation: relative error ≈ ε/amplitude ≈ 1e-9 | I1 |
| A6 | **RMB**: mean SMI loss over 500 iid trials at K = 2N = 32 | within ±0.0153 (4σ of the mean) of 0.5455 | Beta(18, 15) variance; seeded | I1 |
| A7 | **SMI null depth**: in every one of 500 trials at K = 32, response at +26° ≤ −50 dB re look and residual interference at the output below thermal noise | worst observed −57.59 dB and −6.17 dB | the residual must sit below the floor or the aircraft is not detected | I1 |
| A8 | **Diagonal loading**: at K = 8 < N, unloaded SMI raises `ValueError`; loaded at 10 dB above noise, mean loss over 500 trials ≤ 1 dB | measured 0.526 dB | Carlson [7]: support ≈ 2× interference rank (rank 1) | I1 |
| A9 | **SLC**: exact-covariance SLC (taps 7, 8) within 1 dB of optimum and response at +26° ≤ −100 dB re look | 23.67 vs 24.13 dB; −107.7 dB | two DOF suffice for one interferer | I1 |
| A10 | **LCMV constraints** `‖Cᴴw − f‖∞` | ≤ 1e-12 (measured 1.27e-16) | machine precision on a 3×3 solve | I1 |
| A11 | **GSC equals LCMV** | `rtol = 1e-9` on the weights (measured 1.97e-11) | linear algebra | I1, I2 |
| A12 | **Self-nulling, the baseline failing**: SMI with the cell under test in training, mean SINR over 500 trials at K = 32, against the cell under test excluded | ≥ 3 dB worse (measured 6.54 dB at 0°, 9.57 dB at 1° mismatch) | the aircraft's own snapshot in R̂ | I1 |
| A13 | Self-nulling oracle: exact MPDR (aircraft in the covariance) with 1° mismatch | −12.43 dB vs MVDR 23.85 dB, `rtol = 1e-9` | closed-form evaluation, no sampling | I1 |
| A14 | **INCM fixes it**: with the cell under test in training, mean SINR within 0.5 dB of optimum at 0° and within 1 dB at 1° mismatch | measured 23.95 and 23.70 dB vs 24.13 dB | Gu & Leshem [8] | I1 |
| A15 | **I2 baseline failing**: MVDR peak shift, exact covariance; SMI median over 200 trials; every trial | 1.428° exact; median ≥ 1.0° (1.490°); every trial ≥ 0.4° (min 0.522°) | peak walks off the aircraft | I2 |
| A16 | **I2 SLC failing**: exact-covariance SLC SINR at 0° | ≥ 10 dB below MVDR (7.44 vs 23.02 dB) | two central taps cannot null 4.5° without cancelling the beam | I2 |
| A17 | **I2 current method**: EMP angle estimate within 0.02° of 4.5° in every trial (std 0.001°); peak-maintained beam's peak within 0.01° of 0° in every one of 200 trials; null at +4.5° ≤ −50 dB re look in every trial (worst −56.8 dB); constraint residual ≤ 1e-12 | as stated | derivative constraint [5] on the EMP-CMR covariance [9] | I2 |
| A18 | I2 cost of maintenance: exact white-noise gain at look 8.72 dB (MVDR 10.89 dB); SINR over ±1° varies by ≤ 0.6 dB (measured 0.53 dB) vs MVDR's 2.11 dB | `rtol = 1e-9` on the exact values | the price is stated, not hidden | I2 |
| A19 | **I2 negative result**: interferer at +2.5°, peak-maintained beam's look gain re own peak | ≤ −10 dB (measured −11.45 dB) | §4.5 lesson 3 | I2 |
| A24 | **I1 end to end**: over 20 frames, the aircraft is **not** detected on the quiescent uniform or Taylor beam in any frame, and **is** detected after SMI-MVDR (K = 32, cell under test excluded) in every frame, within one range and one Doppler bin | Pd 2.239e-06 / 6.552e-05 before; ≥ 0.999999 after at the RMB 0.1 % quantile | detected after cancellation, not before | I1 |
| A25 | **Weights never applied silently**: no `array/adaptive.py` function except `apply_weights` returns an array with a range or Doppler axis, and none mutates its input | static and runtime check | D16 | all |

A12, A15, A16 and A19 are the baselines failing. A14, A17 and A24 are the current methods
succeeding on the same data.

---

## References

.. [1] P. W. Howells, "Intermediate frequency side-lobe canceller," U.S. Patent 3,202,990,
       Aug. 24, 1965.
.. [2] S. P. Applebaum, "Adaptive arrays," *IEEE Trans. Antennas Propag.*, vol. AP-24, no. 5,
       pp. 585-598, Sep. 1976.
.. [3] O. L. Frost, "An algorithm for linearly constrained adaptive array processing,"
       *Proc. IEEE*, vol. 60, no. 8, pp. 926-935, Aug. 1972.
.. [4] L. J. Griffiths and C. W. Jim, "An alternative approach to linearly constrained adaptive
       beamforming," *IEEE Trans. Antennas Propag.*, vol. AP-30, no. 1, pp. 27-34, Jan. 1982.
.. [5] K. Buckley and L. J. Griffiths, "An adaptive generalized sidelobe canceller with
       derivative constraints," *IEEE Trans. Antennas Propag.*, 1986 (volume and pages to
       confirm at implementation).
.. [6] I. S. Reed, J. D. Mallett and L. E. Brennan, "Rapid convergence rate in adaptive arrays,"
       *IEEE Trans. Aerosp. Electron. Syst.*, vol. AES-10, no. 6, pp. 853-863, Nov. 1974.
.. [7] B. D. Carlson, "Covariance matrix estimation errors and diagonal loading in adaptive
       arrays," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 24, no. 4, pp. 397-401, Jul. 1988.
.. [8] Y. Gu and A. Leshem, "Robust adaptive beamforming based on interference covariance matrix
       reconstruction and steering vector estimation," *IEEE Trans. Signal Process.*, vol. 60,
       no. 7, pp. 3881-3885, Jul. 2012.
.. [9] X. Yang, Z. Zhang, T. Zeng, T. Long and T. K. Sarkar, "Mainlobe interference suppression
       based on eigen-projection processing and covariance matrix reconstruction," *IEEE
       Antennas Wireless Propag. Lett.*, vol. 13, pp. 1369-1372, 2014,
       doi:10.1109/LAWP.2014.2339224.
.. [10] H. L. Van Trees, *Optimum Array Processing* (Part IV of *Detection, Estimation, and
        Modulation Theory*), Wiley, 2002 (MVDR/MPDR, mismatch and the optimum SINR; sections to
        confirm at implementation).
.. [11] H. T. Friis, "A note on a simple transmission formula," *Proc. IRE*, vol. 34, no. 5,
        pp. 254-256, May 1946.
.. [12] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014
        (detection of a non-fluctuating target; section to confirm at implementation).
.. [13] J. Capon, "High-resolution frequency-wavenumber spectrum analysis," *Proc. IEEE*,
        vol. 57, no. 8, pp. 1408-1418, Aug. 1969.
