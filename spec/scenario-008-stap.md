# Scenario 008 — Airborne ground-clutter ridge: space-time adaptive processing

## 1. Status & Purpose

**Status: specified, not implemented.** The first scenario that adapts in space *and* time.

An airborne radar looking at the ground sees clutter whose Doppler shift depends on the direction
it comes from. Every ground patch at angle θ off the array axis returns at Doppler
`2·v_p·cos(cone)/λ`, so in the angle–Doppler plane the clutter is not a blob but a **ridge**. A
slow ground mover in the main beam sits on that ridge: Doppler filtering alone cannot separate it
from the mainlobe clutter, and beamforming alone cannot separate it from the clutter in its own
direction. Only a filter that weights elements and pulses jointly can place a null *along the
ridge* while passing the target just off it. That joint filter is space-time adaptive processing
(STAP), and this scenario builds it.

Three reasons to make this the next multi-channel slice:

1. **It is the cross-cutting requirement in its sharpest form.** Target and clutter share range
   and Doppler, and only the spatial degrees of freedom separate them. §6 asserts that the
   Doppler-only baseline fails on the slow target and STAP succeeds, with numbers.
2. **It ties a public benchmark to analytic truth.** RASPNet [1] is an open dataset of airborne
   clutter generated with RFView. This scenario takes RASPNet's radar exactly as the paper
   tabulates it, runs a synthetic Ward clutter model [2] on it offline, and, if the user has
   downloaded a RASPNet scenario, runs the same chain on the real files. A fifth variant takes
   the MCARM radar that RASPNet's authors use to validate RFView [1, §2.4 and App. B], a moving
   side-looking L-band platform, so one ridge comes from a published radar rather than from
   motion this spec adds.
3. **Its classical form has well-known closed-form answers.** Brennan's rule for the clutter rank
   [2, 15], the Reed–Mallett–Brennan sample-support loss [4] and the optimum SINR loss are exact,
   so the tests check against theory rather than against recorded output.

| Section | What it fixes |
| :--- | :--- |
| §2 Overview & Geometry | The platform, the ridge, what β does, and the chain |
| §3 Parameter Matrix | RASPNet's radar, the MCARM radar, the five variants, derived quantities, contracts |
| §4 Core Decisions | D17 snapshot order, D18 where STAP lives, D19 the optional RASPNet path |
| §5 Build Sequence | What is new, what is widened, what is untouched |
| §6 Acceptance Criteria | Brennan rank, notch, RMB, reduced-rank vs SMI, the slow target, non-stationarity |

### 1.1 Upstream inspiration

| Upstream | What it is | What radar-forge takes | Licence | Baseline or current |
| :--- | :--- | :--- | :--- | :--- |
| Ward, MIT LL TR-1015 [2] | The standard STAP tutorial report: clutter-patch model, ridge, β, Brennan's rule, SINR loss, the reduced-dimension taxonomy | The clutter-patch model, the notation, the stacking order (D17) and the optimum/SINR-loss definitions | report, written from equations | baseline (the oracle every method is measured against) |
| Brennan & Reed 1973 [3] | Theory of the adaptive radar: the optimum space-time filter `w ∝ R⁻¹v` | `optimum_weights` | paper, written from equations | baseline (oracle) |
| Reed, Mallett & Brennan 1974 [4] | Sample-matrix-inversion convergence: the loss is Beta-distributed, mean `(K+2−NM)/(K+1)` | `smi_weights`; `rmb_expected_loss_linear`; the RMB test | paper, written from equations | baseline |
| Brennan & Staudaher 1992 [15]; Goodman & Stiles 2007 [16] | Brennan's rule for the clutter rank; its generalisation to arbitrary arrays as an aperture-bandwidth product | `brennan_rank`; the rank test, and the non-integer β caveat | paper, written from equations | baseline (rank oracle) |
| Klemm [5]; Guerci [6]; Melvin [7] | The textbooks and the tutorial review of STAP | derivation notes, the taxonomy in §4.4; Ward and Melvin suffice, Klemm and Guerci are further reading | books, written from equations | reference |
| Wang & Cai 1994, JDL [8] | Joint-domain localised processing: adapt over a few angle × Doppler beams around the look cell | `jdl_weights` | paper, written from equations | current (reduced-dimension) |
| DiPietro 1992 [9] | Factored and extended-factored post-Doppler STAP | `extended_factored_weights` (EFA) | paper, written from equations | current (reduced-dimension) |
| Haimovich 1996 [10] | The eigencanceler: project out the dominant clutter eigenvectors | `principal_components_weights` | paper, written from equations | current (reduced-rank) |
| Goldstein, Reed & Scharf 1998 [11]; Honig & Xiao 2001 [12] | The multistage Wiener filter, and its Krylov-subspace form | `multistage_wiener_weights` | paper, written from equations | current (reduced-rank) |
| Guerci & Baranoski 2006 (DARPA KASSPER) [13]; Stoica, Li, Zhu & Guerci 2008 [14] | Knowledge-aided STAP: a prior covariance from terrain and platform data, blended with the sample covariance | `knowledge_aided_covariance` | paper, written from equations | current |
| Sun, Meng, Wang & Wang 2011 [17]; Yardibi et al. 2010 (IAA) [18] | Sparse-recovery STAP: estimate the angle–Doppler clutter spectrum from a few snapshots, rebuild the covariance | `sparse_recovery_covariance`, using 006's IAA solver | paper, written from equations | current |
| Borsari 1998 [19]; Himed, Zhang & Hajjari 2002 [20] | Doppler warping and angle–Doppler compensation for range-dependent clutter | `doppler_warp`, `angle_doppler_compensation` (R3) | paper, written from equations | current (compensation) |
| Billingsley 2002 [21] | MIT LL land-clutter measurements; the exponential intrinsic-clutter-motion spectrum | the optional ICM taper in `core/clutter.py` | book, written from equations | current (clutter realism) |
| Babu, Torres & Melvin 1996 [24]; Himed 1999 [25] | MCARM: Rome Laboratory's L-band multichannel airborne measurement programme, and the AFRL report that analyses STAP on its data | R5's radar parameters, as tabulated in RASPNet App. B Table 2 [1], which cites both | papers/report, written from the published parameters; no MCARM data used | reference radar (parameters only) |
| **RASPNet** [1] (data on AFRL SDMS) | A benchmark **dataset**: 100 CONUS scenarios × 10 000 RFView clutter realisations, over 16 TB, CC BY 4.0 | The radar parameters (§3.1); the optional loader (D19); R4 | dataset CC BY 4.0; example code (`shyamven/RASPNet`) has no LICENSE file, so it is never vendored | benchmark data |
| Venkatasubramanian et al. 2022, 2024 [22, 23] | The RASPNet authors' data-driven localisation after adaptive detection (regression CNN on NAMF heatmaps) | A learned comparison, behind the `ml` extra only | paper, written from equations | current (learned), optional |

---

## 2. Overview & Geometry

The radar is RASPNet's: an X-band array 1000 m up, in spotlight mode, aimed 20 km to the
south-west [1, §2.1 and Table 4]. The array face carries 48 × 5 elements at 15 mm, beamformed into
L = 16 receive channels of 3 × 5 elements each.

```
                          v_p  -->   (platform velocity; R1: zero)
                 [ 48 x 5 element array, 16 subarray channels ]
                  \            h = 1000 m
                   \ depression 2.862 deg at 20 km ground range
                    \
                     \_____________ clutter ring at the cell under test ________
        ground  ......:::::::::::[ mainlobe footprint ]:::::::::::......
                                    ^ slow mover, inside the mainlobe clutter Doppler
```

**The ridge.** For a side-looking array (velocity along the array axis), a ground patch at
azimuth `az` and elevation `el` has normalised spatial frequency `θ_s = (d/λ)·cos el·sin az` and
normalised Doppler `ω = f_d·T_r = β·θ_s`, with

```
    beta = 2 * v_p * T_r / d          (Ward [2]; platform motion per PRI, in half-spacings)
```

so the clutter lies on a straight line of slope β through the origin of the (θ_s, ω) plane.

```
      omega (normalised Doppler)                  omega
       +0.5 |            /                        +0.5 |       /      /
            |          /                               |     /      /
            |        /      beta = 1                   |   /      /      beta = 1.3: the line
          0 |------x-------  ridge from corner         0 |-x-----/----   leaves the square at
            |    /           to corner: fills          | /     /        |theta| = 0.3846 and
            |  /             Doppler exactly once      |/    /          re-enters from the other
       -0.5 |/____________ theta_s                -0.5 |___/________     side: ALIASED clutter
          -0.5          +0.5                          -0.5         +0.5
```

**β > 1 aliases the ridge in Doppler**, and that is the teaching point of R2's second case: some
Doppler bins then hold clutter from two directions at once, and the integer-β sharp eigenvalue
cutoff of Brennan's rule becomes a gradual roll-off (§6 A3).

**A non-side-looking array bends the ridge into a curve whose position depends on elevation, and
so on range.** Training data from neighbouring range bins then have a different covariance from
the cell under test. That is R3.

The chain, end to end:

```
  scenarios/scenario_008_stap_*.toml              (R4: $RASPNET_DIR/CLUTTER/num{i}_.../real{j}.mat)
              |                                                     |
              v                                                     v
  core/clutter.py   ward_clutter_patches -> ward_clutter_covariance   pipelines/raspnet.py
              |      (exposes per-patch theta_s, omega, CNR)          load_raspnet_clutter
              v                                                     |
  core/clutter.py   synthesize_clutter (per range bin, CN(0, R_c))    |  (L, Lambda, G) -> cube
              |  + thermal_noise + injected targets (array/ steering) |
              v                                                     v
        data_rfst  (n_pulses, n_range_bins, n_rx)  complex128   <-- the SAME array either way
              |
              v
  array/stap.py   space_time_snapshot (D17)  ->  training_snapshots (guard + train)
              |        sample_covariance / knowledge_aided / sparse_recovery
              v
     weights: optimum | smi | jdl | efa | pc | mwf    (D16 API: weights returned, applied explicitly)
              |
              v
  array/adaptive.py apply_weights (007, D16) per Doppler bin -> data_stap_rv (n_doppler_bins, n_range_bins)
              |
              v
  core/detection.py  cfar_detect_2d   (UNCHANGED)  ->  detections, SINR-loss curve, eigenspectrum
```

---

## 3. Parameter Matrix

### 3.1 The radar — RASPNet Table 4, as published

Every row is from RASPNet's "Complete list of radar platform parameters" [1, App. G.1, Table 4],
checked identical in arXiv v1 (as Table 5), v2 and v3. CNR is from the paper's signal model
[1, App. C.2].

| Parameter | Value | Name in the TOML |
| :--- | :--- | :--- |
| Carrier | 10 GHz → λ = **0.029979 m** | `f0_hz` |
| Bandwidth | 5 MHz → c/2B = **29.9792 m** (the paper rounds to 30 m) | `bandwidth_hz` |
| PRF | 1100 Hz → CPI **58.182 ms** at 64 pulses | `prf_hz` |
| Platform altitude | 1000 m | `platform_altitude_m` |
| Platform speed | **not given: "the radar was assumed as static"** [1, §2.1] | `platform_speed_mps` |
| Antenna | 48 × 5 elements, spacing 0.015 m = **0.50035 λ** | `n_elements_horizontal`, `element_spacing_m` |
| Receive channels L | 16 × 1: subarrays of **3 × 5** elements; phase-centre spacing 0.045 m = **1.5010 λ** | `n_rx`, `subarray_width_elements` |
| Pulses Λ | 64 | `n_pulses` |
| Range bins G | 1134 over a 20 372 m swath (17.9647 m per bin) | `n_range_bins` |
| Pointing | spotlight, aimed 20 km SW (depression 2.862° at 20 km) | `look_azimuth_deg = 225` |
| CNR | 20 dB, fixed | `cnr_db` |
| Unambiguous range c/2PRF | 136.269 km — no range ambiguity anywhere in this scenario | |
| Unambiguous velocity ±λ·PRF/4 | ±8.2443 m/s | |
| Doppler resolution PRF/Λ | 17.1875 Hz = 0.2576 m/s | |

> **RASPNet's platform is static.** RASPNet's radar is the X-band, static-platform radar above,
> and this spec uses it. Its data therefore contains **no platform-motion Doppler at all**: an
> airborne *position* without airborne *motion*. The ridge in R2 and R3 comes from motion this
> spec adds, and §3.2 says so. The moving-platform parameters in the same paper (its Table 2)
> belong to MCARM, the measured dataset RFView was validated against, not to RASPNet.

Two consequences shape the variants:

- **The subarray channels are spaced 1.5 λ.** Their spatial frequency spans ±1.5010 cycles per
  channel, so channel-level steering vectors have grating lobes, damped only by the 3-element
  subarray pattern. This is spatial aliasing, the twin of β > 1 in Doppler. Brennan's rule
  assumes λ/2 spacing, so the rank test runs on element-level channels (R2), not on RASPNet's
  subarrays.
- **NM = 16 × 64 = 1024 adaptive DOF, so RMB wants K = 2NM = 2048 training snapshots, but the
  swath has G = 1134 range bins** (2NM/G = 1.806). Full-DOF SMI cannot be trained from one CPI's
  range bins. The paper trains instead on i.i.d. *realisations* of the same bin [1, App. C.2],
  which is possible only in simulation. This is why reduced-dimension STAP is the operational
  default here.

### 3.2 Platform motion added, and β

The platform speed is chosen to set β with the PRF held at RASPNet's 1100 Hz, so that nothing in
the scenario is range-ambiguous:

```
    v_p = beta * d * PRF / 2        (d = 0.015 m)
```

| β | `platform_speed_mps` | Ridge | Brennan `N + β(M−1)`, N = M = 16 |
| :--- | ---: | :--- | ---: |
| 0 | 0 | R1: all clutter at zero Doppler | N = 16 (as rank of the spatial covariance) |
| 1 | **8.2500** | corner to corner, fills Doppler once | **31.00** → 31 |
| 1.3 | **10.7250** | aliased beyond \|θ_s\| = 0.3846 | 35.50 → ⌈35.50⌉ = **36** |

These are slow platforms (an aerostat, or a rotorcraft in light drift); that is the price of
keeping RASPNet's carrier and PRF unchanged and range-unambiguous. Every result is a function of
β and of normalised frequencies only. One unit of ω is `λ·PRF/2` = **16.4886 m/s** of radial
velocity, so the m/s figures below scale with that number, and a faster platform with a
proportionally higher PRF reproduces every table, range ambiguity aside.

### 3.3 The five variants

| | **R1** `raspnet-static` | **R2** `sidelooking-ridge` | **R3** `forward-looking` | **R4** `raspnet-data` | **R5** `mcarm-sidelooking` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Purpose | RASPNet exactly: the synthetic twin of R4 | Ward's textbook ridge; rank, notch, RMB, the slow target | Range-dependent ridge; non-stationary training | The same chain on a downloaded RASPNet scenario | A published moving radar: ridge, MDV and training support at full MCARM size |
| Channels × pulses | 16 subarrays × 64 | 16 elements (λ/2) × 16 | 16 × 16 | 16 × 64, from the file | 16 columns (0.1092 m) × 128; see §3.4.1 |
| Array orientation | side-looking (RASPNet does not state it; the static platform makes it irrelevant) | side-looking, look broadside | velocity along boresight, beam steered 60° off it | as recorded | side-looking, look broadside (assumed; §3.4.1) |
| β | 0 | 1 and 1.3 | 1 | 0 (static, per [1]) | **0.92314**, from MCARM's own speed and PRF |
| CNR per element per pulse | 20 dB | 40 dB | 40 dB | as recorded | 40 dB (chosen; Table 2 gives none) |
| Cell under test | 20 km | 10 km slant (el −5.7392°) | 10 km | user's choice | the Table 2 aim point: 105.886 km slant, depression 1.663° |
| Clutter source | `core/clutter.py` Ward model | Ward model | Ward model, per range ring | RFView, via `pipelines/raspnet.py` | Ward model |
| Tests | fast | fast; RMB/reduced-rank Monte Carlo `slow` | `slow` | **skipped unless `RASPNET_DIR` is set** | `slow` (NM = 2048) |

**Why N = M = 16 for R2.** It matches RASPNet's channel count and the shared 16-element ground
array of 006/007/009/010; with λ/2 elements it satisfies Brennan's assumptions exactly; and
NM = 256 makes K = 2NM = 512 training bins fit inside RASPNet's 1134. Ward's own worked example
uses different numbers, which this spec does not reproduce (to confirm at implementation if a
like-for-like figure is wanted).

### 3.4 Derived quantities (R2, β = 1, look broadside, ω on a 4001-point grid)

From the scratch derivation script, quoted verbatim:

| Quantity | Value |
| :--- | ---: |
| Clutter eigenvalues > σ²·{1.01, 2, 10, 100} | 31, 31, 31, 31 — a sharp cutoff at Brennan's 31 |
| Same, β = 1.3 | 40, 38, 36, 35 — gradual; Brennan's ⌈35.5⌉ = 36 at the 10σ² threshold |
| Same, R1 (β = 0, 16 × 64) | 16; the 16th eigenvalue 19.00 dB, the 17th 0.00 dB |
| Optimum SINR-loss minimum | −59.59 dB at ω = 0 |
| Notch half-width at −3 / −5 / −10 dB | ω = 0.0415 / 0.0302 / 0.0158 → 0.6843 / **0.4988** / 0.2597 m/s |
| Mainlobe clutter Doppler half-extent (null to null) β/N | ω = 0.0625 → 1.0305 m/s |
| Usable Doppler-space fraction (UDSF) at −5 dB | optimum 0.9393; Doppler-only 0.0000 |
| Doppler-only loss (50 dB Chebyshev space × 60 dB Chebyshev time) | −5.25 dB at ω = 0.25, −5.00 dB at ω = 0.5; −62.62 dB at worst |
| **Minimum detectable velocity** (−5 dB SINR-loss threshold) | **0.4988 m/s** |
| RMB K = 2NM = 512: SMI loss ρ ~ Beta(258, 255) | mean 0.50292 = **−2.9850 dB**, sd 0.02205 |
| RMB K = 288 (1.125 NM) | mean 0.11765 = −9.2942 dB |
| Required coherent output SNR, Pd 0.9 at Pfa 10⁻⁶, non-fluctuating | 13.183 dB |

For R3 (forward-looking, β = 1, beam 60° off the velocity, CUT 10 km): training over
2NM = 512 rings minus 4 guard cells spans **2.3403–17.6597 km**, elevation −25.296° to −3.246°
against −5.739° at the CUT. The CUT's mainlobe clutter sits at (θ_s, ω) = (0.4311, 0.2489), and
across the span it moves over θ_s 0.3918–0.4326 and ω 0.2262–0.2498: both coordinates move,
which is why Doppler warping alone (a shift in ω) is not enough and ADC (a shift in both) is.

The MDV threshold of −5 dB is this spec's choice; Ward [2] defines the UDSF against a stated
threshold, and the threshold travels with the number.

**What 64 pulses (R1) and 16 pulses (R2) buy.** Temporal DOF set the Doppler resolution and so
the notch width (≈ 1/M in ω): R1's 64 pulses resolve 0.2576 m/s; R2's 16 resolve 1.0305 m/s, and
the optimum −5 dB notch half-width, 0.4988 m/s, is less than half a Doppler bin because the
adaptive null sits *on* the ridge rather than across a whole bin.

#### 3.4.1 R5 — the MCARM radar

Parameters come from RASPNet's "MCARM Dataset Radar Platform Parameters" [1, App. B, Table 2],
which cites Babu et al. [24] and Himed [25]: 1.25 GHz, 1.25 MHz, PRF 1984 Hz, altitude 3072 m,
speed 100 m/s, range resolution 120 m, 16 × 8 elements (horizontal × vertical) at
0.1092 m × 0.1407 m, 24 × 1 receive channels, 128 pulses, a 63-sample transmitted waveform, 630
range bins over a 75 500 m swath, platform at (39.3805°, −75.9746°) aimed at
(39.3630°, −74.7435°).

**The channel mapping, and the simplification taken.** Table 2's 24 channels do not equal the 16
columns. What could be verified: the MCARM array has 16 columns, each of two four-element
subarrays; each subarray has its own output or is combined into one output per column, with up
to 24 outputs for the array [24, abstract]. Which 24 outputs an acquisition records, and where
their phase centres sit, is in Himed's report [25], which could not be retrieved (to confirm at
implementation). This spec therefore does **not** model the 24 channels. R5 uses **one channel
per column: a 16-channel horizontal ULA at d = 0.1092 m**, the column-combined mode that [24]
describes. It keeps the full horizontal aperture, which is the only dimension the side-looking
ridge depends on, and it invents no layout. The vertical spacing (0.1407 m = 0.58666 λ) only
shapes the elevation beam, which at 1.663° depression this model folds into the patch gain. The
side-looking orientation is also an assumption: Table 2 does not state it.

**Which d goes into β.** The horizontal spacing, 0.1092 m. β counts the platform's motion along
the array axis in half-spacings *along that axis*, so the column pitch is what the ridge sees.

| Quantity (R5) | Value |
| :--- | ---: |
| λ | **0.239834 m**; d/λ = 0.45531, below λ/2, so spatial frequency occupies only ±0.45531 |
| β = 2·v·T_r/d | **0.92314**: the ridge reaches ω = ±0.42032 and does **not** alias |
| Range resolution c/2B; swath/G | 119.9170 m; 119.8413 m per bin |
| Unambiguous range c/2PRF | 75.5525 km |
| Aim point (Table 2 positions, spherical Earth) | ground 105.841 km, bearing 90.66°, slant 105.886 km, depression 1.663° |
| Unambiguous velocity ±λ·PRF/4 | ±118.9576 m/s; one unit of ω = 237.9153 m/s |
| CPI; Doppler resolution | 64.516 ms; 15.5000 Hz = 1.8587 m/s |
| Brennan N + β(M−1) | 133.238 → 134 |
| Eigenvalues > σ²·{1.01, 2, 10, 100} | **127, 125, 124, 120** |
| Brennan scaled by the occupied spatial band 2d/λ = 0.91063 | 121.33 (a heuristic in the spirit of the aperture–bandwidth product of [16], not its formula) |
| NM; RMB 2NM; range bins G | 2048; **4096**; **630** (2NM/G = 6.502): full-DOF SMI is impossible from one CPI |
| JDL 3 × 3: DOF, RMB 2·DOF | 9, 18; expected RMB loss at K = 18 / 30 / 600: −2.3736 / −1.2963 / −0.0582 dB |
| Optimum notch half-width at −3 / −5 / −10 dB | 6.3642 / **4.7583** / 2.4981 m/s; UDSF (−5 dB) 0.9598 |
| **MDV** (−5 dB threshold, as in §3.4) | **4.7583 m/s** |
| Mainlobe clutter Doppler half-extent β/N | ω = 0.05770 → 13.7268 m/s |
| Doppler-only loss (the tapers of R2) | −10.14 dB at best; UDSF 0.0000 |

The aim is beyond the unambiguous range (105.886 km against 75.5525 km), so its clutter is
second-time-around. For a side-looking array, ambiguous range rings sit on the same ridge: A14
shows a range-averaged side-looking covariance loses 0.00 dB. The single-ring model is therefore
kept, and range ambiguity is not modelled.

R5 does not make R2 redundant. R2 has integer β (the sharp Brennan cutoff), λ/2 spacing, the
aliasing case β = 1.3, and NM = 256, which is small enough for the RMB Monte Carlo. R5 has
non-integer β, spacing under λ/2 and NM = 2048, and that size is what sharpens its
training-support lesson.

The MCARM *measured* data are not used. This spec does not say where they can be obtained,
because no public download location was verified.

### 3.5 Mathematical contracts

| Contract | Statement |
| :--- | :--- |
| Cube | `data_rfst`, shape `(n_pulses, n_range_bins, n_rx)`, complex128 (style.md §3.1) |
| Spatial steering | scenario-006 D13: `a_n = exp(+j2π p_n·u/λ)`; for 006's centred ULA `a_n = exp(+j2π θ_s (n − (n_rx−1)/2))`, `θ_s = (d/λ)·cos el·sin az`; unnormalised. The centring is a common phase and changes no SINR |
| Temporal steering | `b_m = exp(+j2π ω m)`, `ω = f_d·T_r`, closing-positive `f_d` — the same sign as `core/signal.py`, where a closing target's carrier phase advances pulse to pulse |
| Space-time snapshot (D17) | `x = data_rfst[:, g, :].reshape(-1)`, C order, index `m·n_rx + n`; steering `v = b(ω) ⊗ a(θ_s)`, length `n_rx·n_pulses` |
| Clutter covariance | `R = σ²( Σ_k ξ_k v_k v_kᴴ + I )`, patches k on the iso-range ring; `ξ_k` the patch CNR from transmit gain, element pattern and constant-γ reflectivity [2] |
| Ridge slope | `β = 2 v_p T_r / d`; side-looking `ω_k = β θ_s,k` |
| Brennan's rule | `rank(R_c) ≈ N + β(M−1)`, exact for integer β with λ/2 spacing [2, 15]; for non-integer β, `⌈N + β(M−1)⌉` is an approximation and the eigenspectrum rolls off [16] |
| Optimum weight | `w_opt = R⁻¹v` (any scale); `SINR_opt = σ_s²·vᴴR⁻¹v` [3] |
| SINR loss | `L(w) = SINR(w) / (NM·σ_s²/σ²)`, linear in `(0, 1]`; reported as `sinr_loss_db` |
| SMI | `R̂ = (1/K) Σ x_k x_kᴴ` over training bins; `ρ = SINR_SMI/SINR_opt ~ Beta(K+2−NM, NM−1)`, `E[ρ] = (K+2−NM)/(K+1)` [4] |
| JDL | transform to the 3 × 3 angle–Doppler beams around the look cell, `T` of orthonormal `v(θ_i, ω_j)/√NM`, spacings 1/N and 1/M; `w = T(TᴴR̂T)⁻¹Tᴴv` [8] |
| PC (eigencanceler) | `w = (I − U_r U_rᴴ) v`, `U_r` the r dominant eigenvectors of `R̂`, r from Brennan [10] |
| MWF | D stages; implemented as MVDR restricted to the Krylov subspace `span{v, R̂v, …, R̂^{D−1}v}` [11, 12] |
| KA | `R̂_KA = α R_0 + (1−α) R̂`, `R_0` from `ward_clutter_covariance` with the platform's reported velocity [14] |
| Doppler warping / ADC | per training ring k, `R_k ← D_k R_k D_kᴴ`, `D_k = diag(b(ω_c − ω_k) ⊗ 1)` (DW [19]) or `diag(b(ω_c − ω_k) ⊗ a(θ_c − θ_k))` (ADC [20]), (ω_k, θ_k) the ring's mainlobe clutter point |
| RASPNet file | `real{j}.mat` holds `X_clut`, complex, `(L, Λ, G)` [1, App. E.1]; the loader returns `X_clut.transpose(1, 2, 0)` = `(n_pulses, n_range_bins, n_rx)` |

### 3.6 Output schema

As scenario-001 §3.6, plus a `/stap` group in `products.h5`:

| Dataset | Shape | Meaning |
| :--- | :--- | :--- |
| `/stap/burst{k}/data` | `(n_frames, n_doppler_bins, n_range_bins)` complex128 | the STAP output `data_stap_rv`; attr `method`, `n_training`, `n_guard` |
| `/stap/burst{k}/sinr_loss_db` | `(n_frames, n_doppler_bins)` float64 | the SINR-loss curve at the look angle; attr `covariance` = `"clairvoyant"` or `"estimated"` |
| `/stap/burst{k}/clutter_eigenvalues_linear` | `(n_frames, n_rx·n_pulses)` float64 | descending, normalised by σ² |
| `/stap/burst{k}/weights` | `(n_frames, n_doppler_bins, n_rx·n_pulses)` complex128 | only if `save_weights = true`; stacked per D17 |

`truth.csv` gains `doppler_normalized` (ω, dimensionless) and `spatial_frequency_normalized`
(θ_s) per target, so every test can read the target's place on the ridge without re-deriving it.

---

## 4. Core Decisions

### 4.1 D17 — the space-time snapshot is the C-order flatten of `(n_pulses, n_rx)`

For range bin g, the snapshot is `x = data_rfst[:, g, :].reshape(-1)`. Its index is
`pulse_index * n_rx + rx_index`: **temporal outer, spatial inner**. The matching steering vector
is `v = np.kron(b, a)`, b temporal (length `n_pulses`) and a spatial (length `n_rx`). This is
Ward's ordering [2] and also RASPNet's: its effective space-time steering vector is
`d(v) ⊗ a(θ, φ)` [1, App. C.2, eq. (6)].

Every STAP bug starts here: a snapshot flattened in the other order still has the right length
and a perfectly invertible covariance, and the filter quietly nulls the wrong ridge. So:

- `array/stap.py` has exactly **one** function that flattens (`space_time_snapshot`) and **one**
  that builds `v` (`space_time_steering_vector`). Nothing else calls `reshape` on a slab.
- §6 A1 asserts the identity `space_time_snapshot` of a synthetic single-target cube equals
  `kron(b, a)` scaled by the amplitude, to `rtol = 1e-12`.
- RASPNet's temporal vector is written with `e^{−i2π f_d/f_PR}` [1, App. C.2]. radar-forge's is
  `e^{+j2π f_d T_r}` (§3.5). RASPNet's clutter is static, so the sign cannot show in R4's
  clutter; it shows on injected targets, which use radar-forge's convention throughout.

### 4.2 D18 — STAP lives in `array/stap.py`; the Ward clutter model in `core/clutter.py`

`array/stap.py` holds the steering and snapshot helpers, training-set selection, covariance
estimators (sample, KA, sparse-recovery), weight solvers (optimum, SMI, JDL, EFA, PC, MWF), the
compensations (Doppler warping, ADC), and the diagnostics
(`sinr_loss_linear`, `rmb_expected_loss_linear`, `brennan_rank`). It is array processing and
depends on `array/beamforming.py`'s steering vectors (006). Weight solvers follow D16 (007)
exactly: **each returns 007's `AdaptiveWeights`**, with `weights` of length `n_rx·n_pulses`
stacked per D17, `method` one of `"stap_optimum"`, `"stap_smi"`, `"jdl"`, `"efa"`, `"pc"`,
`"mwf"`, `"ka"`, `"sr"`, `covariance` the matrix actually inverted, `n_snapshots`, `loading_w`,
and `diagnostics` holding the STAP intermediates (eigenvalues, Brennan rank, JDL transform `T`,
MWF basis, KA prior and α). `sample_covariance`, `training_snapshots` and `apply_weights` are
007's and are reused, never duplicated; STAP adds only `space_time_snapshot` in front of them.
No solver applies its weights.

`core/clutter.py` gains the Ward discrete-patch model, the physics of the scene rather than a
processing step:

- `ward_clutter_patches(...)` → per-patch `azimuth_rad`, `spatial_frequency`, `doppler_normalized`,
  `cnr_linear`. The ridge is a plot of two of these columns; nothing is hidden.
- `ward_clutter_covariance(patches, n_rx, n_pulses)` → `R/σ²`, with an optional
  `icm_taper` from Billingsley's exponential spectrum [21] (off by default; R1–R3 assert with
  it off, so the rank tests stay exact).
- `synthesize_clutter(covariance, n_range_bins, rng)` → CN(0, R) per range bin, shaped into the
  cube. Per-bin covariances (R3) are passed as a sequence.

This sits beside 005's least-squares leakage cancellers in the same module, as the module map
says: cancelling a known waveform and modelling an unknown ridge are both "clutter", and neither is
beamforming.

### 4.3 D19 — RASPNet is optional; its data is never committed; its test skips without it

- `pipelines/raspnet.py` exposes `load_raspnet_clutter(path, *, realization=None)` →
  `data_rfst (n_pulses, n_range_bins, n_rx)` plus a `RaspnetScenario` record (index, latitude and
  longitude parsed from the `num{i}_lat{..}_lon{..}` folder name [1, App. E.1]).
- **File format.** `real{j}.mat` holds `X_clut` of size `L × Λ × G` [1, App. E.1]. The authors'
  own CVNN notebook reads the processed files with `scipy.io.loadmat` [`shyamven/RASPNet`,
  `Localization_Examples/RASPNet_CVNN.ipynb`], so those are MAT v5/v7. Whether the CLUTTER
  files are v7.3 is not stated (to confirm on the first download). The loader tries
  `scipy.io.loadmat` (core dependency) first and, on `NotImplementedError` (v7.3 is HDF5), falls
  back to `h5py`, which arrives through data-001's **`io` extra** (DF8). No new extra is needed.
  An h5py read of a MATLAB array comes back with its axes reversed, `(G, Λ, L)`; the loader
  normalises both paths to `(Λ, G, L)` and tests that it did.
- **Access.** The SDMS collection page offers 50 sample realisations per scenario for direct
  download; full 10 000-realisation scenarios are by e-mail request, shipped on user storage to
  US addresses only. The scenario documents both and assumes neither.
- **Never committed.** `RASPNET_DIR` points at the user's copy. `.gitignore` is unchanged
  because nothing is ever written inside the repo; the test reads only.
- **Skipped when absent.** `tests/pipelines/test_scenario_008_raspnet.py` begins with
  `pytest.skip` unless `RASPNET_DIR` is set and contains `CLUTTER/`. It carries a new
  `dataset` marker (registered in `pyproject.toml`), so `make check` stays green offline.
- **Two training modes, labelled.** `training="range"` (operational: neighbouring bins of one
  realisation) and `training="realizations"` (the paper's: the same bin across i.i.d.
  realisations [1, App. C.2]). The second is an idealisation available only in simulation, and
  the API says so in its name and docstring.

### 4.4 The method, derived from the classical one

Kept to one idea per function, in the order a reader meets them:

1. **Optimum** `w = R⁻¹v` with the true R: the oracle. It cannot be built in practice.
2. **SMI**: replace R by the sample covariance of K training bins. Needs K ≈ 2NM [4]. With
   NM = 1024 and 1134 bins (R1, R4), that is not available.
3. **Reduced dimension** (JDL, EFA): adapt in a small transformed space chosen *before* looking at
   data. DOF fall to 9 (JDL 3 × 3), so K ≈ 18. Loses a little against the optimum; never fails
   for lack of data.
4. **Reduced rank** (PC, MWF): adapt in a small space chosen *from* the data. Brennan says the
   clutter occupies about `N + β(M−1)` dimensions, so a rank-31 eigencanceler or a 16-stage MWF
   nearly reaches the optimum.
5. **Knowledge-aided**: put the platform's own knowledge (velocity, geometry, terrain) into a
   prior covariance and blend. Excellent when the prior is right; harmful when it is not (§6 A9
   shows both).
6. **Sparse recovery**: the clutter ridge is a thin set in the angle–Doppler plane, so a sparse
   spectrum estimate from very few snapshots rebuilds R, provided the dictionary contains it.
7. **Learned** (optional, `ml` extra): the RASPNet authors' regression CNN localises a target
   from NAMF heatmaps after adaptive detection [22, 23]. It is a comparison on R4 data, never a
   dependency of the chain, and never asserted in `make check`.

Every step exposes its intermediates: the eigenspectrum, the training set used, the condition
number, the SINR-loss curve, and for KA the prior and α.

### 4.5 Amendments to `spec/structure.md`

> **Applied.** These amendments are folded into `spec/structure.md`; it is the authoritative
> tree and decision list. This section records what changed and why.

1. **Part B tree.** Under `core/`: `clutter.py  # land/sea clutter; least-squares leakage
   cancellation (005); Ward clutter-patch model, ICM taper (008)`. Under `array/`: `stap.py  # space-time snapshot (D17),
   optimum/SMI/JDL/EFA/PC/MWF, KA and sparse-recovery covariance, DW/ADC, SINR loss, Brennan
   rank`. Under `pipelines/`: `raspnet.py  # optional RASPNet loader (D19); reads, never writes`.
2. **B.1 rows.** `array/stap.py` | Ward TR-1015; Brennan & Reed 1973; Reed–Mallett–Brennan 1974;
   Wang & Cai 1994 (JDL); DiPietro 1992 (EFA); Haimovich 1996 (PC); Goldstein–Reed–Scharf 1998
   (MWF); Guerci & Baranoski 2006, Stoica et al. 2008 (KA); Sun et al. 2011 (SR); Borsari 1998,
   Himed et al. 2002 (DW, ADC). `core/clutter.py` (append) | Ward TR-1015 clutter-patch model;
   Billingsley 2002 ICM. `pipelines/raspnet.py` | RASPNet dataset conventions (file layout and
   `X_clut` shape).
3. **Decisions D17, D18, D19**, as §4.1–§4.3.
4. **§A.7 RASPNet, replace the opening and the table** with: "RASPNet (Venkatasubramanian,
   Kang, Pezeshki, Rangaswamy, Tarokh; arXiv:2406.09638) is a **benchmark dataset**, not a
   network: 100 CONUS scenarios × 10 000 RFView-simulated clutter realisations, over 16 TB,
   licensed CC BY 4.0 and hosted on AFRL SDMS
   (`https://www.sdms.afrl.af.mil/index.php?collection=raspnet`). The radar is X-band (10 GHz,
   5 MHz, PRF 1100 Hz), 1000 m up and **static**, a 48 × 5 array at 15 mm beamformed to
   L = 16 channels, 64 pulses, 1134 range bins, spotlight 20 km SW (Table 4). Each
   `CLUTTER/num{i}_lat_lon/real{j}.mat` holds `X_clut`, `L × Λ × G`; `EXAMPLES` and `CVNN` hold the
   processed localisation sets. The GitHub repository holds only example notebooks and has no
   LICENSE file." Keep the "what radar-forge borrows" paragraph, adding: "and, through
   `pipelines/raspnet.py`, the clutter files themselves as an optional STAP test input (scenario
   008 D19)." In §A.14, the RASPNet row becomes: `dataset CC BY 4.0; code none declared` |
   `no (data via SDMS)` | `no — dataset read at runtime, code never vendored`.
5. **Slice table**: `Scenario 8 | scenario-008-stap.md | airborne clutter ridge → STAP (optimum,
   SMI, JDL, PC, MWF, KA, SR) on RASPNet and MCARM radar parameters, with optional RASPNet data; D17–D19`.
6. **B.3 extras**: no new extra. `h5py` (v7.3 fallback) is data-001's `io` extra.

### 4.6 Workstream boundary

This scenario claims:

```
scenarios/scenario_008_stap_{raspnet_static,sidelooking_ridge,forward_looking,raspnet_data,
                             mcarm_sidelooking}.toml
spec/scenario-008-stap.md, docs/scenarios/scenario-008-stap.md
src/radar_forge/array/stap.py                      (new)
src/radar_forge/core/clutter.py                    (Ward model half only; 005 owns the leakage cancellers)
src/radar_forge/pipelines/raspnet.py               (new)
tests/array/test_stap.py, tests/core/test_clutter_ward.py
tests/pipelines/test_scenario_008.py, tests/pipelines/test_scenario_008_raspnet.py
```

Consumed, never edited: `core/dsp.py` (`doppler_fft`, `doppler_bin_centers_mps`, `range_fft`),
`core/windows.py` (`taper`, `coherent_gain_linear`), `core/detection.py` (`cfar_detect_2d`,
`cluster_detections`), `core/signal.py` (`thermal_noise`), `core/constants.py`. Consumed from
006: `array/geometry.py`, `array/beamforming.py` steering vectors; from 007: the D16 weight
convention in `array/adaptive.py` (diagonal loading is reused from there, not reimplemented).
`pipelines/scenarios.py` is widened (an `[platform]` table: `platform_speed_mps`,
`platform_altitude_m`, `array_orientation`), and its existing variants must not shift.

---

## 5. Build Sequence

```
  1. core/clutter.py     ward_clutter_patches -> ward_clutter_covariance -> synthesize_clutter
           |               (test: rank vs Brennan first; nothing else is meaningful before it)
           v
  2. array/stap.py       space_time_steering_vector, space_time_snapshot  (D17 identity test)
           |
           v
  3. array/stap.py       optimum_weights, sinr_loss_linear     (notch, MDV, UDSF)
           |
           v
  4. array/stap.py       smi_weights over 007's training_snapshots, rmb_expected_loss_linear
           |
           v
  5. array/stap.py       jdl_weights, extended_factored_weights, principal_components_weights,
           |             multistage_wiener_weights
           v
  6. array/stap.py       knowledge_aided_covariance, sparse_recovery_covariance (006's IAA)
           |
           v
  7. array/stap.py       doppler_warp, angle_doppler_compensation        (R3)
           |
           v
  8. pipelines/scenarios.py  [platform] table; R1-R3 and R5 TOMLs; apply_weights -> CFAR
           |
           v
  9. pipelines/raspnet.py    loader, both MAT paths; R4 TOML; skip-if-absent test
```

| Step | Module | Notes |
| :--- | :--- | :--- |
| 1 | `core/clutter.py` *(edit)* | Ward model; constant-γ reflectivity; element pattern `cos²` front, −30 dB backlobe; transmit array factor of the full aperture. Per-patch intermediates exposed. |
| 2 | `array/stap.py` *(new)* | D17. Spatial steering delegated to `array/beamforming.py` (006 D13). |
| 3 | `array/stap.py` | `sinr_loss_linear` takes weights, steering, covariance; returns linear. |
| 4 | `array/stap.py` | Training via 007's `training_snapshots` (guard cells excluded), fed D17 snapshots. |
| 5 | `array/stap.py` | JDL uses `core/dsp.py`'s Doppler conventions for its beam centres. |
| 6 | `array/stap.py` | SR reuses 006's IAA solver; KA takes the prior as an argument, never builds it silently. |
| 7 | `array/stap.py` | The compensations return the transform `D_k`, so it can be inspected and inverted. |
| 8 | `pipelines/scenarios.py` *(widen)* | `[platform]` absent → today's behaviour. |
| 9 | `pipelines/raspnet.py` *(new)* | No write path. |

---

## 6. Acceptance Criteria

`tests/pipelines/test_scenario_008.py` (R1–R3, R5) and `tests/pipelines/test_scenario_008_raspnet.py`
(R4). Unit tests for each function live in `tests/array/test_stap.py` and
`tests/core/test_clutter_ward.py`. Clairvoyant quantities are deterministic linear algebra, so
their bounds are tight; Monte Carlo bounds are set from the exact sampling distribution, never
loosened.

| # | Criterion | Bound | Applies to |
| :--- | :--- | :--- | :--- |
| A1 | **D17 identity.** The snapshot of a noise-free single-target cube equals `amp·kron(b(ω), a(θ_s))` | `rtol = 1e-12` | all |
| A2 | **Brennan, integer β.** Eigenvalues of `R/σ²` above 1.01, 2, 10 and 100 | exactly **31** at every threshold (sharp cutoff) | R2 β = 1 |
| A3 | **Brennan, non-integer β.** Count above 10σ² equals ⌈N + β(M−1)⌉; the count rises as the threshold falls | **36** at 10σ²; 40 / 38 / 36 / 35 at 1.01 / 2 / 10 / 100 | R2 β = 1.3 |
| A4 | **β = 0.** Rank equals the spatial rank | **16**; 17th eigenvalue within `1e-9` of σ² | R1 |
| A5 | **Optimum notch at the mainlobe clutter Doppler.** Minimum of the optimum SINR-loss curve, and its −3/−5/−10 dB half-widths | minimum at ω = 0 (grid step 2.5e-4); half-widths 0.0415 / 0.0302 / 0.0158, ± one grid step | R2 β = 1 |
| A6 | **Aliasing.** Look at θ_s = 0.4 (az 53.463°): the notch sits at the wrapped mainlobe Doppler | β = 1: ω = 0.4000; β = 1.3: ω = **−0.4800** (not +0.52), ± one grid step | R2 |
| A7 | **RMB.** Mean SMI loss relative to optimum over 200 trials at K = 2NM = 512 | in **[−3.039, −2.931] dB** (4σ of Beta(258, 255)); the script's own 200-trial run gave −2.9925 dB | R2, `slow` |
| A8 | **Reduced dimension/rank beat SMI at K just above NM.** K = 288, slow target, 100 trials, mean loss vs optimum | SMI −9.17 dB; JDL −2.26, PC (r = 31) −0.50, MWF (D = 16) −0.49. Assert each of JDL, PC, MWF beats SMI by **≥ 6 dB** | R2, `slow` |
| A9 | **Few samples.** K = 32 (NM/8) | JDL −3.28 dB beats 10 dB-loaded SMI (−7.76 dB) by **≥ 3 dB**; KA with exact prior, α = 0.5: **≥ −1 dB** (got −0.43); KA with a 1 % platform-speed error in the prior: worse than JDL (got −11.81) — the prior-sensitivity lesson, asserted | R2, `slow` |
| A10 | **MWF stages.** Clairvoyant loss vs optimum | D = 16: within **0.2 dB** at ω = 0.25 (got −0.11) and at the slow target (got −0.00); D = 4: worse than −15 dB at both (got −19.52, −31.41) | R2 |
| A11 | **SR-STAP, and its failure mode.** IAA spectrum on a 2N × 2M angle–Doppler grid, K = 4 snapshots, 20 trials, slow target | β = 1 (ridge on the grid): mean −0.31 dB, 10th pct −0.49; assert **≥ −1 dB**. β = 1.3 (ridge off the grid), same grid: mean −40.19 dB; assert **≤ −30 dB** | R2, `slow` |
| A12 | **The slow target — the teaching point.** v_r = 0.4122 m/s (ω = 0.025, inside the mainlobe clutter Doppler ±1.0305 m/s), SNR 0 dB per element per pulse (24.08 dB integrated). Optimum STAP output SINR 17.67 dB > 13.183 dB required: **detected** (Pd 1.000000 at Pfa 10⁻⁶). Doppler-only (beam then tapered Doppler filter): −37.82 dB, Pd 1.002e-06 = Pfa: **not detected** | margins 4.487 dB and 51.003 dB; asserted from the clairvoyant SINR, then a 200-CPI `cfar_detect_2d` check: STAP Pd ≥ 0.9, Doppler-only Pd ≤ 0.01 | R2 |
| A13 | **Control: the baseline is not always wrong.** With the static RASPNet platform, Doppler-only loss at 1.0 m/s | −3.60 dB (passes); at 0.3 m/s −41.20 dB while optimum is −0.08 dB | R1 |
| A14 | **Stationarity, side-looking.** Range-averaged covariance over 2NM rings (2.3403–17.6597 km) vs the CUT's own | loss **0.00 dB** across the whole Doppler axis | R2 geometry |
| A15 | **Non-stationary ridge (R3).** Same training span; beam 60° off the velocity; band 0.04 < \|ω − ω_c\| < 0.15 | uncompensated mean −9.43 dB, worst −30.01 dB; Doppler warping mean −8.54 dB (insufficient off-boresight); **ADC mean −1.97 dB, worst −3.30 dB**. Assert ADC ≥ uncompensated + 6 dB in band mean, and ADC worst ≥ −4 dB | R3, `slow` |
| A16 | **R3 slow target** at ω_c − 0.08 = 0.1689: optimum loss −2.85 dB; relative to it, uncompensated −28.55, DW −21.14, **ADC −3.17** | ADC ≥ −4 dB; DW ≤ −20 dB; uncompensated ≤ −25 dB (relative to the CUT optimum) | R3, `slow` |
| A17 | **RASPNet sample support.** With R1/R4 dimensions | `rmb_required_snapshots(1024) = 2048 > 1134`: the loader warns when `training="range"` is used with full DOF; JDL with 50 realisations has expected RMB loss −0.741 dB | R1, R4 |
| A18 | **R4 skipped without data**; with data: shape `(64, 1134, 16)` complex; an injected target at a chosen (θ_s, ω, bin) is the JDL output's peak bin; the eigenspectrum is written, **not asserted** (RFView's clutter is not this spec's model) | skip reason names `RASPNET_DIR` | R4, `dataset` |

| A19 | **R5 rank.** Eigenvalues of `R/σ²` above 10σ² | **124**, deterministic. It lies between the band-scaled 121.33 and Brennan's 134, because with d < λ/2 part of the spatial-frequency axis is empty | R5, `slow` |
| A20 | **R5 notch and MDV.** Optimum −5 dB notch half-width | ω = 0.0200 → **4.7583 m/s**, ± one grid step (2.5e-4 in ω = 0.0595 m/s) | R5, `slow` |
| A21 | **R5 slow target.** v_r = 4.0 m/s, below the MDV and inside the ±13.7268 m/s mainlobe clutter Doppler; SNR −10 dB per element per pulse (23.11 dB integrated). Optimum STAP 16.78 dB: **detected** (Pd 1.0000). Doppler-only −41.40 dB (Pd 1.001e-06): **not detected** | asserted from the clairvoyant SINR, then 200 CPIs through `cfar_detect_2d`: STAP Pd ≥ 0.9, Doppler-only Pd ≤ 0.01 | R5, `slow` |
| A22 | **R5 training support.** `rmb_required_snapshots(2048) = 4096 > 630`: full-DOF SMI from range bins is refused with a clear error. JDL 3 × 3 trained on K = 600 bins, 10 trials: mean −3.18 dB against the optimum, within **0.2 dB** of its own clairvoyant −3.123 dB (RMB predicts −0.0582 dB for 9 DOF at K = 600). 10 dB-loaded SMI at the same K gives −0.99 dB; it is recorded, not asserted | R5, `slow` |

Notes on the bounds:

- **A5, A6, A10, A14–A16** are clairvoyant: deterministic functions of the model. The ± one grid
  step on A5/A6 is the resolution of the ω grid, not a tolerance on the physics.
- **A7** uses the exact RMB distribution: ρ ~ Beta(K+2−NM, NM−1) [4], sd 0.02205 per trial, so
  the 200-trial mean has sd 0.00156 and the 4σ band is the stated interval.
- **A8, A9** margins are differences of means far larger than the trial-to-trial spread (the
  10th percentiles were SMI −10.09, JDL −2.33, PC −0.60, MWF −0.60 at K = 288).
- **A12** is the criterion the scenario exists for: the baseline fails where the current method
  succeeds.

### 6.1 Why A11 asserts a failure

Sparse recovery rebuilds R from a dictionary of angle–Doppler atoms. When the ridge passes
through the atoms (β = 1 on a 2N × 2M grid, where the ridge θ_s = ω meets the grid diagonal),
four snapshots suffice: −0.31 dB against the optimum, where SMI is not even defined below
K = NM = 256. When the ridge falls between atoms (β = 1.3 on the same grid), the reconstructed
covariance misses the clutter and the loss is −40.19 dB at the same K. Both are asserted, because
grid mismatch is the documented weakness of the method and a learner should see it fail before
trusting it. The remedy, a dictionary laid along the ridge predicted from the platform's own
velocity (a knowledge-aided dictionary), is the extension (to confirm at implementation); it is
not asserted here.

Bounds A15 and A16 come from the derivation's discretisation (360 patches per ring, every
eighth training ring of the 508). An implementation on a finer discretisation must reproduce
them within the stated margins; if it does not, the margin is investigated, not widened.

---

## References

.. [1] S. Venkatasubramanian, B. Kang, A. Pezeshki, M. Rangaswamy, V. Tarokh, "RASPNet: A
       Benchmark Dataset for Radar Adaptive Signal Processing Applications," arXiv:2406.09638,
       v3, 4 Nov. 2025. §2.1 (platform, static radar, 48/L × 5 subarrays), App. C.2 (signal
       model, CNR 20 dB, eq. (6) `d(v) ⊗ a(θ, φ)`), App. E.1 (folder layout, `X_clut`
       `L × Λ × G`), App. G.1 Table 4 (radar parameters). Data: AFRL SDMS,
       https://www.sdms.afrl.af.mil/index.php?collection=raspnet. Code:
       https://github.com/shyamven/RASPNet.
.. [2] J. Ward, *Space-Time Adaptive Processing for Airborne Radar*, Technical Report 1015, MIT
       Lincoln Laboratory, Lexington, MA, Dec. 1994. (Equation numbers to confirm at
       implementation.)
.. [3] L. E. Brennan, L. S. Reed, "Theory of adaptive radar," *IEEE Trans. Aerosp. Electron.
       Syst.*, AES-9(2), pp. 237-252, Mar. 1973.
.. [4] I. S. Reed, J. D. Mallett, L. E. Brennan, "Rapid convergence rate in adaptive arrays,"
       *IEEE Trans. Aerosp. Electron. Syst.*, AES-10(6), pp. 853-863, 1974.
.. [5] R. Klemm, *Principles of Space-Time Adaptive Processing*, 3rd ed., IET, 2006.
.. [6] J. R. Guerci, *Space-Time Adaptive Processing for Radar*, Artech House, 2003.
.. [7] W. L. Melvin, "A STAP overview," *IEEE Aerosp. Electron. Syst. Mag.*, 19(1), pp. 19-35,
       Jan. 2004.
.. [8] H. Wang, L. Cai, "On adaptive spatial-temporal processing for airborne surveillance radar
       systems," *IEEE Trans. Aerosp. Electron. Syst.*, 30(3), pp. 660-669, Jul. 1994.
.. [9] R. C. DiPietro, "Extended factored space-time processing for airborne radar systems,"
       *Proc. 26th Asilomar Conf. Signals, Systems and Computers*, vol. 1, pp. 425-430, 1992.
.. [10] A. M. Haimovich, "The eigencanceler: adaptive radar by eigenanalysis methods," *IEEE
        Trans. Aerosp. Electron. Syst.*, 32(2), pp. 532-542, Apr. 1996.
.. [11] J. S. Goldstein, I. S. Reed, L. L. Scharf, "A multistage representation of the Wiener
        filter based on orthogonal projections," *IEEE Trans. Inf. Theory*, 44(7), Nov. 1998,
        p. 2943 ff.
.. [12] M. L. Honig, W. Xiao, "Performance of reduced-rank linear interference suppression,"
        *IEEE Trans. Inf. Theory*, 47(5), pp. 1928-1946, Jul. 2001.
.. [13] J. R. Guerci, E. J. Baranoski, "Knowledge-aided adaptive radar at DARPA: an overview,"
        *IEEE Signal Process. Mag.*, 23(1), pp. 41-50, 2006.
.. [14] P. Stoica, J. Li, X. Zhu, J. R. Guerci, "On using a priori knowledge in space-time
        adaptive processing," *IEEE Trans. Signal Process.*, 56(6), pp. 2598-2602, Jun. 2008.
.. [15] L. E. Brennan, F. M. Staudaher, *Subclutter Visibility Demonstration*, Adaptive Sensors
        Inc., Technical Report RL-TR-92-21, Mar. 1992.
.. [16] N. A. Goodman, J. M. Stiles, "On clutter rank observed by arbitrary arrays," *IEEE Trans.
        Signal Process.*, 55(1), pp. 178-186, Jan. 2007. §III-A (Brennan's rule, eq. (20)).
.. [17] K. Sun, H. Meng, Y. Wang, X. Wang, "Direct data domain STAP using sparse representation
        of clutter spectrum," *Signal Processing*, 91, pp. 2222-2236, 2011.
.. [18] T. Yardibi, J. Li, P. Stoica, M. Xue, A. B. Baggeroer, "Source localization and sensing:
        a nonparametric iterative adaptive approach based on weighted least squares," *IEEE
        Trans. Aerosp. Electron. Syst.*, 46(1), pp. 425-443, 2010.
.. [19] G. K. Borsari, "Mitigating effects on STAP processing caused by an inclined array,"
        *Proc. IEEE National Radar Conf.*, Dallas, TX, pp. 135-140, 1998.
.. [20] B. Himed, Y. Zhang, A. Hajjari, "STAP with angle-Doppler compensation for bistatic
        airborne radars," *Proc. IEEE Radar Conf.*, Long Beach, CA, 2002.
.. [21] J. B. Billingsley, *Low-Angle Radar Land Clutter: Measurements and Empirical Models*,
        William Andrew / SciTech, 2002.
.. [22] S. Venkatasubramanian, C. Wongkamthong, M. Soltani, B. Kang, S. Gogineni, A. Pezeshki,
        M. Rangaswamy, V. Tarokh, "Toward data-driven STAP radar," *Proc. IEEE Radar Conf.*,
        2022. arXiv:2201.10712.
.. [23] S. Venkatasubramanian, S. Gogineni, B. Kang, A. Pezeshki, M. Rangaswamy, V. Tarokh,
        "Data-driven target localization using adaptive radar processing and convolutional
        neural networks," *IET Radar, Sonar & Navigation*, 18(10), pp. 1638-1651, 2024.
.. [24] B. N. S. Babu, J. A. Torres, W. L. Melvin, "Processing and evaluation of multichannel
        airborne radar measurements (MCARM) measured data," *Proc. IEEE Int. Symp. Phased Array
        Systems and Technology*, pp. 395-399, 1996. doi:10.1109/PAST.1996.566123.
.. [25] B. Himed, *MCARM/STAP Data Analysis, Volume II*, Final Technical Report
        AFRL-SN-RS-TR-1999-48, Air Force Research Laboratory, May 1999 (DTIC ADA364942).
