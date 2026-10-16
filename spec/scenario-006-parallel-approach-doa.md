# Scenario 006 — Two aircraft on closely spaced parallel approaches in one range-Doppler cell: spatial super-resolution on the datachip

## 1. Status & Purpose

**Status: specified, not implemented.** The first array scenario.

Scenarios 001–005 have one receive channel. Two aircraft at the same range and the same radial
velocity are therefore one detection, and nothing downstream can separate them. This document
specifies the **first multi-channel slice**: the scenario-001 radar at the Duke Receiver grows a
16-element receive array, and two aircraft on closely spaced parallel approaches, aircraft A and
aircraft B, share a single range-Doppler cell. Only the spatial degrees of freedom separate them.

Three reasons to make this the first array slice:

1. **It fixes the array contracts once.** The multi-channel cube, the array frame and the
   steering-vector sign (D13) are used by every later array scenario (007–010). They are fixed
   here, against a case where a sign error shows up as a wrong angle, not as a plausible picture.
2. **It teaches the coherent-source problem, which is badly taught.** Two echoes in one cell
   share range and Doppler, so their phase difference is constant over the dwell. A covariance
   built from pulses or from neighbouring cells then has a rank-one signal part, and textbook
   MUSIC and ESPRIT fail however high the SNR. D14 says which snapshots form a covariance, and §6
   shows the baseline failing.
3. **It consumes the datachip.** Scenario 004 (D9) defines `DataChip`. This slice is its first
   multi-channel consumer: fine estimation runs on the chip and reports in parent coordinates.

| Section | What it fixes |
| :--- | :--- |
| §2 Overview & Geometry | The array, the pair, the separation sweep, and why the echoes are coherent |
| §3 Parameter Matrix | Array, waveform, link budget, separations, variants, contracts, outputs |
| §4 Core Decisions | D13, D14; aircraft B's construction; scattering phase; what moved in `structure.md` |
| §5 Build Sequence | What is new, what is widened, what is untouched |
| §6 Acceptance Criteria | Conventions first, then the baseline failing and the current methods succeeding |

### 1.1 Upstream inspiration

| Upstream | What it is | What radar-forge takes | Licence | Baseline or current |
| :--- | :--- | :--- | :--- | :--- |
| Conventional (Bartlett) beam and monopulse; Sherman & Barton [9] | Matched spatial filter; sum/difference angle estimation | The teaching contrast: one blob, a power-weighted centroid, a per-look indication that wanders with relative phase | book, written from equations | Baseline |
| Capon MVDR [10] | Distortionless minimum-variance beamformer | `mvdr_weights`; shown failing by signal cancellation on the coherent pair | paper, written from equations | Baseline |
| MUSIC [1], Root-MUSIC [2]; ESPRIT [3] | Subspace estimators on a sample covariance | The estimators; fail without decorrelation | paper, written from equations | Baseline (subspace core) |
| Spatial smoothing [4]; forward-backward averaging [5] | Restore signal-subspace rank from subarrays and the conjugate-reversed array | `spatial_smoothing`, `forward_backward`; FBSS-MUSIC/ESPRIT | paper, written from equations | Current for subspace methods |
| IAA [6] | Single-snapshot, grid-based, iterative weighted least squares | `iaa_spectrum`: no covariance, no decorrelation step | paper, written from equations | **Current** (single snapshot) |
| SPICE [7] | Sparse covariance fitting, hyperparameter-free | `spice_spectrum`, reported beside IAA | paper, written from equations | Current, reported |
| RELAX / deterministic ML [8] | Concentrated-likelihood fit of K sources | `deterministic_ml`: the oracle every method is scored against | paper, written from equations | Oracle |
| Stoica & Nehorai CRB [11]; statistical resolution limit [12] | Lower bound on angle variance; separation resolvable at a given SNR | `crb_deterministic`, `resolution_snr_db` | paper, written from equations | Ground truth |
| pyroomacoustics `doa` [13] | Acoustic DOA estimators behind a common base class | The `DoAEstimator` base-class pattern; MUSIC as an external cross-check (`structure.md` §A.11) | MIT | Cross-check |

---

## 2. Overview & Geometry

The scenario-001 transmitter (30 dBi, 100 W, 9.8 GHz FMCW, waveform S1) illuminates the scene. The
receive end is now a horizontal uniform linear array (ULA) at the Duke Receiver: 16 channels,
half-wavelength spacing, boresight at aircraft A's mean bearing over the default window. Aircraft
A flies the real track in `data/flight_coordinates.csv`. Aircraft B flies a path offset laterally
from it, as on a closely spaced parallel approach, at the same range and radial velocity.

```
                           aircraft B             aircraft A
                     (same range, same v_r) (data/flight_coordinates.csv)
                                  *                    *
                                   \   delta_az       /
                                    \<------------->/      R = 16.626 - 18.358 km
                                     \             /
                                      \           /
                                       \         /
                                        v       v
          x <---------------- [ 16-element ULA, d = lambda/2 ] ----> boresight (y)
                               Duke Receiver, boresight 235.804 deg true
```

The pair occupy **one** range bin (74.9481 m) and **one** velocity bin (0.0597482 m/s). Range and
Doppler processing cannot split them. What is left is the 16-element snapshot of that one cell:

```
   x  =  a(u_1) s_1  +  a(u_2) s_2  +  n            x in C^16, u = sin(az) cos(el)
```

### 2.1 Why the two echoes are coherent

Each cell of the range-Doppler map is the sum of both echoes weighted by the same range and
Doppler response, because they share range and Doppler. Any snapshot the processor can gather,
pulse `m` or neighbouring cell `k`, is therefore

```
   x_k  =  g_k (a(u_1) s_1 + a(u_2) s_2)  +  n_k        (the same g_k for both echoes)
```

and the sample covariance has signal part `(sum |g_k|^2) y y^H` with `y = A s`: **rank one**,
whatever the number of snapshots. MUSIC needs a rank-two signal subspace, so it fails. A Doppler
offset of a fraction δ of a bin breaks the tie only partly: the slow-time correlation of the two
echoes is `|sin(π δ) / (M sin(π δ / M))|`, with M = 256:

| Doppler offset δ (bins) | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Slow-time correlation \|ρ\| | 1.00000 | 0.99589 | 0.98363 | 0.90032 | 0.63662 | 0.00000 |

### 2.2 The separation sweep, in Rayleigh widths

The array's Rayleigh width in direction cosine is `Δu_R = λ/(n_rx·d) = 0.125` (7.1808° at
broadside). Separations are swept **in Rayleigh widths**. What that means in metres depends on
range, and it is large:

| Lateral offset for | 2× | 1× | 0.5× | 0.25× |
| :--- | ---: | ---: | ---: | ---: |
| R = 5.326 km (track minimum) | 1.346 km | 0.667 km | 0.333 km | 0.166 km |
| R = 16.640 km (window, near) | 4.205 km | 2.085 km | 1.041 km | 0.520 km |
| R = 18.360 km (window, far) | 4.639 km | 2.301 km | 1.148 km | 0.574 km |
| R = 22.010 km (track maximum) | 5.561 km | 2.758 km | 1.377 km | 0.688 km |

The other way round, in Rayleigh widths for a given lateral offset:

| Lateral offset | 5.33 km | 16.64 km | 18.36 km | 22.01 km |
| :--- | ---: | ---: | ---: | ---: |
| 15 m (less than one airliner's wingspan) | 0.0225 | 0.0072 | 0.0065 | 0.0055 |
| 150 m | 0.2253 | 0.0721 | 0.0654 | 0.0545 |
| 500 m | 0.7499 | 0.2403 | 0.2178 | 0.1817 |
| 1000 m | 1.4933 | 0.4805 | 0.4355 | 0.3633 |
| 2000 m | 2.9340 | 0.9592 | 0.8697 | 0.7259 |

For context, closely spaced parallel runways are roughly 760–1300 m apart (Raleigh-Durham's are
about 1 km apart), which at 16.64 km is roughly 0.37–0.62 Rayleigh widths.

The aperture is 0.245 m. Two aircraft 15 m apart at 17 km are 0.007 Rayleigh widths apart, and
§3.4 shows that no estimator can resolve them at this SNR: the bound forbids it, not the
algorithm. The variants therefore sweep from 2× down to 0.1× Rayleigh, where the methods differ.

---

## 3. Parameter Matrix

### 3.1 Site, array and targets

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Site | Duke Receiver, 36.00250 °N, 78.94100 °W, 60 m | Scenario-001 §3.1, unchanged |
| Transmitter | Scenario-001 S1: 100 W, `gain_tx_dbi` 30.0, 9.8 GHz, 2.0 MHz, 1.0 ms chirp, 1.0 kHz PRF | One separate transmit antenna, unchanged |
| Receive array | ULA along the array x axis, `n_rx` = 16, `element_spacing_wavelengths` = 0.5 | Shared ground array (007, 009, 010) |
| λ, d, aperture `n_rx·d` | 30.5911 mm, 15.2955 mm, **0.24473 m** | Computed |
| Phase centre | Array centre; element x positions `(n − 7.5)·d`, n = 0…15 | D13 |
| `boresight_azimuth_deg` | **235.804** (circular mean of aircraft A's bearing over the default window) | Computed |
| `boresight_elevation_deg` | 0.0 (no tilt) | Specified |
| Per-channel `gain_rx_dbi` | **20.0**: a λ/2 × 0.4869 m (15.92 λ) column, from `G = 4πA/λ²` | Physically sized; see below |
| Beamformed receive gain | 20.0 + 10·log₁₀ 16 = **32.04 dBi** | Within 2.04 dB of scenario-001's 30 dBi dish |
| `noise_figure_db` | 3.0 per channel; independent noise per channel | Scenario-001 |
| Aircraft A | `data/flight_coordinates.csv`, 1500 m AMSL, 10 dBsm, Swerling 0 | Scenario-001 |
| Aircraft B | Same range, same elevation, same radial velocity; offset in `u` by `separation_rayleigh × Δu_R`; 10 dBsm | §4.3 |
| Default window | `start_time_s = 0.0`, `duration_s = 120.0` | Scenario-001's |

> The per-channel gain is not scenario-001's 30 dBi. A channel λ/2 wide cannot be a 30 dBi
> antenna: that needs a 4.9 m column. 20 dBi is a half-metre column, and 16 of them combine to
> 32.04 dBi, so the beamformed link budget stays within 2 dB of scenario 001's.

### 3.2 Geometry over the default window

| Quantity | Value |
| :--- | ---: |
| Aircraft A slant range | 16.626 – 18.358 km |
| Aircraft A bearing | 232.34 – 238.57 ° true |
| Aircraft A off-boresight azimuth | −3.468 – +2.763 ° (`u` = −0.06031 – +0.04805) |
| Aircraft A elevation | 4.42 – 4.89 ° |
| Aircraft A radial velocity | −39.32 – +55.30 m/s |
| Aircraft A cross-range speed | 0.81 – 41.72 m/s |

**Array approximations, checked.** Far field begins at `2D²/λ` = 3.92 m. The largest transit
time across the aperture is 0.765 ns against `1/B` = 500 ns, a ratio of 1.53 × 10⁻³, so a
phase-only (narrowband) steering vector is exact to that order. A horizontal ULA measures the
direction cosine `u = sin(az)·cos(el)`, not azimuth. Converting `u` to azimuth with `el = 0`
errs by at most `1/cos(4.89°) − 1` = 0.37 % of the off-boresight angle, which is 0.039° at the
widest F1 offset (10.7° off boresight). `u` is the estimated quantity, and azimuth is derived at the output boundary.

### 3.3 Waveform and link budget, per channel

Waveform S1 of scenario-001 §3.4, unchanged: 256 chirps of 1000 samples, CPI 256.0 ms, range
bin 74.9481 m, velocity bin 0.0597482 m/s. Rectangular windows in range and Doppler, so that the
cell response is a known Dirichlet kernel and no taper widens the cell.

| Range | Received power | Per-sample SNR | Per-channel cell SNR (×256 000) | Beamformed (×16) |
| :--- | ---: | ---: | ---: | ---: |
| 5.326 km | 5.8608e-14 W | 5.64 dB | 59.73 dB | 71.77 dB |
| 16.640 km | 6.1510e-16 W | −14.15 dB | **39.94 dB** | 51.98 dB |
| 18.360 km | 4.1502e-16 W | −15.85 dB | **38.23 dB** | 50.27 dB |
| 22.010 km | 2.0095e-16 W | −19.00 dB | 35.08 dB | 47.12 dB |

Noise power per sample is 1.5978e-14 W (scenario-001 §3.2). The cube is
`(256, 1000, 16)` complex128, **65.54 MB per frame**, so the pipeline tests run 5 frames.

### 3.4 What the bound allows: the statistical resolution limit

Smith's criterion [12]: two sources are resolvable when the separation exceeds its own standard
deviation, `sqrt(CRB(δ)) = δ`. With the deterministic CRB of [11] for one snapshot, equal powers,
the per-channel cell SNR this requires is:

| Separation δ (Rayleigh) | relative phase 0 | π/2 | π |
| :--- | ---: | ---: | ---: |
| 1.0 | −18.3 dB | −15.6 dB | −6.5 dB |
| 0.5 | −6.4 dB | −3.5 dB | 11.5 dB |
| 0.25 | 5.7 dB | 8.7 dB | 29.5 dB |
| 0.1 | 21.7 dB | 24.7 dB | 53.4 dB |
| 0.0225 (15 m at 5.33 km) | 47.6 dB | 50.6 dB | 92.2 dB |
| 0.0072 (15 m at 16.64 km) | 67.4 dB | 70.4 dB | 121.9 dB |

Two lessons. The **relative phase sets the bound**: echoes in anti-phase at the phase centre
nearly cancel in the signal subspace, and need 12 dB more SNR at one Rayleigh width and 32 dB
more at 0.1. And with 38–40 dB available,
the floor is about 0.1 Rayleigh in the favourable phases. Two aircraft 15 m apart at 17 km are
out of reach for any estimator with this aperture.

### 3.5 Variants

All three share §3.1–§3.3 and differ in `[second_aircraft]` and the run length.

| | **F1** `parallel-resolved` | **F2** `parallel-subrayleigh` | **F3** `parallel-sweep` |
| :--- | :--- | :--- | :--- |
| `separation_rayleigh` | 2.0 (Δu = 0.25, 14.48°) | 0.5 (Δu = 0.0625, 3.58°) | 0.1, 0.25, 0.5, 1.0, 2.0 |
| SNR | Link budget, 38.23–39.94 dB | Link budget, 38.23–39.94 dB | 0 to 40 dB per-channel cell SNR, 5 dB steps |
| Aircraft B's scattering phase | Uniform per frame, seeded | Uniform per frame, seeded | Uniform per trial, seeded |
| Level | Full chain, IQ to `doa.csv` | Full chain, IQ to `doa.csv` | Snapshot model of §3.6, 200 trials per point |
| Purpose | Sanity: everything resolves | Teaching point: raw MUSIC fails; FBSS, IAA, ML succeed | Probability of resolution and RMSE/CRB curves |
| Marker | `slow` (5 frames) | `slow` (5 frames) | `slow`; curves in a notebook, one point in the suite |

F3 runs on the snapshot model, not on synthesised cubes: 9 SNRs × 5 separations × 200 trials of
65 MB cubes would be 590 GB of IQ to learn what one 16-vector per trial already says. Criterion
A13 proves the snapshot model equal to the full chain, which is what licenses the shortcut.

Prototype results at F2 (Δu = 0.5 Rayleigh, 38.23 dB, 150–200 trials, scratch script, not code):

| Estimator | Snapshots | P(resolved) | RMSE / √CRB |
| :--- | :--- | ---: | ---: |
| Conventional beam (Bartlett) | peak cell | 0.067 | — |
| MUSIC, raw chip covariance | 25 chip cells | 0.073–0.085 | — |
| ESPRIT, raw chip covariance | 25 chip cells | 0.005 | — |
| FB only (no subarrays), MUSIC | 25 chip cells | 0.970; **fails at phase 0 and π** | 3.20 |
| FBSS-MUSIC, L = 12 | 25 chip cells | 1.000 | 1.08–1.12 |
| FBSS-ESPRIT, L = 12 | 25 chip cells | 1.000 | 1.00–1.04 |
| IAA, 15 iterations, 4096-point grid | peak cell | 1.000 | 2.87 |
| SPICE, 60 iterations | peak cell | 0.833 | 18.66 |
| Deterministic ML | peak cell | 1.000 | 1.04 |

At F1 every method resolves, but MUSIC on the raw covariance and the conventional beam carry a
coherent bias of 90–95× √CRB: resolving is not the same as estimating well.

### 3.6 Mathematical contracts

Scenario-001 §3.5 and scenario-002 §3.5 hold unchanged. Added:

| Contract | Statement |
| :--- | :--- |
| Multi-channel cube | `(n_pulses, n_samples, n_rx)` complex128, receive channel last (D13). Without an array the generators return `(n_pulses, n_samples)` exactly as today |
| Array frame | x along the array face (right, looking out), y boresight, z up. `element_positions_m` shape `(n_rx, 3)` (D13) |
| Direction | `u = [sin az cos el, cos az cos el, sin el]`, azimuth from boresight towards +x, elevation from the x–y plane towards +z (D13) |
| Steering vector | `a_n(u) = exp(+j 2π p_n·u / λ)`, shape `(n_rx,)`, unnormalised, `‖a‖² = n_rx`. An element nearer the source leads in phase (D13) |
| Weights | `y = wᴴ x`. A beamformer returns `w`; it never applies it silently |
| Snapshot | `x = data_rv[k_d, k_r, :]`, shape `(n_rx,)`: one range-Doppler cell across channels (D14) |
| Snapshot model | `x = A(u) s + n`, `A = [a(u_1) a(u_2)]` `(n_rx, 2)`, `n ~ CN(0, σ² I)`; per-channel cell SNR `|s_i|²/σ²` |
| Spatial smoothing [4] | `R_ss = (1/K) Σ_{k=0}^{K−1} R[k:k+L, k:k+L]`, `K = n_rx − L + 1` subarrays of `L` channels |
| Forward-backward [5] | `R_fb = ½ (R + J R* J)`, `J` the exchange matrix. Requires a centro-symmetric array |
| IAA [6] | `p_k ← |a_kᴴ R⁻¹ x|² / (a_kᴴ R⁻¹ a_k)²`, `R = Σ_k p_k a_k a_kᴴ`, from `p_k = |a_kᴴ x|²/n_rx²` (equation numbers to confirm at implementation) |
| Deterministic ML [8], [11] | `û = argmax_u xᴴ P_A(u) x`, `P_A = A (AᴴA)⁻¹ Aᴴ`; RELAX alternation or a 2-D grid then a local refine |
| CRB [11] | `CRB(u) = (σ² / 2) {Re[(Dᴴ P_A^⊥ D) ⊙ (s sᴴ)ᵀ]}⁻¹`, one snapshot, `D = ∂A/∂u`. One source: `6σ² / (π² |s|² n_rx (n_rx² − 1))` at d = λ/2 (equation number to confirm at implementation) |
| Resolved | Two estimates, sorted, each within `Δu/2` of the matching true `u`. The same criterion for every method |
| Angle units | `u` dimensionless inside `array/`; radians inside `core/`; `azimuth_boresight_deg` and `azimuth_deg` at file boundaries (data-001 DF2) |

### 3.7 Output schema

As scenario-001 §3.6, with:

| File | Change |
| :--- | :--- |
| `truth.csv` | One row per aircraft per frame (`target_id` `aircraft_a`, `aircraft_b`), plus `azimuth_boresight_deg` and `direction_cosine_x` |
| `iq.h5` | `data` carries the trailing `rx` axis that data-001 §6.7 already provides |
| `products.h5` | `/rd/burst0/data` gains a trailing `n_rx` axis; new optional `/doa/burst0/spectrum` |
| `doa.csv` | **New.** One row per estimated source per detection per method; §4.6 gives the columns |
| `metadata.json` | New `receive_array` block: `n_rx`, `element_positions_m`, `boresight_azimuth_deg`, `boresight_elevation_deg` |

---

## 4. Core Decisions

### 4.1 D13 — the multi-channel cube and the steering-vector convention

Fixed here and referenced by 007–010 as "scenario-006 D13".

1. **Cube.** `(n_pulses, n_samples, n_rx)`, slow time axis 0, fast time axis 1, receive channel
   last (`docs/conventions/style.md` §3.1). `n_rx` counts receive channels. `n_elements` means
   physical elements where the two differ, as with subarrays.
2. **Array frame.** x along the array face (right), y boresight, z up. Element positions are
   `element_positions_m`, shape `(n_rx, 3)`, in metres in this frame.
3. **Direction.** Azimuth from boresight towards +x; elevation from the x–y plane towards +z.
   Unit vector `u = [sin az cos el, cos az cos el, sin el]`, passed to functions as
   `(n_directions, 3)`. A ULA along x observes only the first component, the direction cosine
   `u_x = sin az cos el`. The estimators return it as `direction_cosine_x`, and wherever this
   spec writes a scalar `u` it means `u_x`.
4. **Steering vector.** For a plane wave arriving from `u`, `a_n = exp(+j·2π·(p_n·u)/λ)`. An
   element closer to the source leads in phase. Unnormalised, `‖a‖² = n_rx`. Weights apply as
   `y = wᴴx`.
5. **Phase centre.** `array/geometry.py`'s ULA and URA factories put the phase centre at the
   origin, so a centred array's `a` is conjugate centro-symmetric, `J a* = a`. Forward-backward
   averaging relies on it and raises `ValueError` on a geometry that is not centro-symmetric.
6. **Site to array.** The array's boresight is given as a compass bearing and an elevation tilt,
   `boresight_azimuth_deg` and `boresight_elevation_deg`. `array/geometry.py` owns the one
   rotation from the site's ENU frame (data-001 DF2) into the array frame. With zero tilt,
   off-boresight azimuth is the bearing minus the boresight bearing, wrapped to ±180°.
7. **Synthesis.** `core/signal.py` stays geometry-agnostic. The generators gain one optional
   argument, `rx_response_linear`, shape `(n_paths, n_rx)` complex: each channel's complex
   response to each path. `array/beamforming.py::steering_vectors` builds it from the paths'
   arrival angles. With it the cube is 3-D. Without it the 2-D cube is bit-identical to today's,
   which is how 001–003 stay untouched. Noise is drawn independently per channel at
   `Radar.noise_power_w` each.
8. **Narrowband.** The steering vector is phase-only. Valid while the aperture transit time is
   small against `1/B`: 1.53 × 10⁻³ here, stated in `signal.py`'s Notes with the bound. 009 owns
   the wideband case.

One wrinkle in existing code: `core/dsp.py::range_fft` and `range_doppler_map` default to
`fast_time_axis = -1`, which on a 3-D cube is the channel axis. `pipelines/scenarios.py` must
pass `fast_time_axis=1` (and `matched_filter(..., axis=1)`) explicitly. On a 2-D cube axis 1 is
axis −1, so the change alters no 001–003 result. `core/dsp.py` itself is not edited.

### 4.2 D14 — which snapshots form a covariance

1. **The estimation snapshot is one cell.** Every array estimator here consumes the chip's peak
   cell across channels, `x = chip.data[k_d, k_r, :]`, at the full coherent gain of 256 000.
   Scalloping from an off-bin target multiplies all channels by the same complex factor, so the
   peak cell's direction information is exact even where its amplitude is not.
2. **A covariance from pulses or neighbouring cells is the baseline, not a remedy.** For a
   co-range, co-Doppler pair, §2.1 shows its signal part is rank one. `sample_covariance` over
   the chip's cells is implemented and used, by MUSIC and ESPRIT, precisely to show the failure
   (A7, A8). It also imports anything else in those cells.
3. **The rank is restored spatially.** The remedy covariances are formed from that same sample
   covariance by spatial smoothing over `K = n_rx − L + 1` subarrays, by forward-backward
   averaging, or by both (FBSS). `L = 12` (`K = 5`), so the subarray's Rayleigh width is
   `2/12 = 0.1667` in `u`, 1.33× coarser than the full array's. That is the aperture cost [4].
   FB alone fails at relative phases 0 and π (A9); FBSS with `K ≥ 2` does not.
4. **Single-snapshot methods take `x` directly.** IAA, SPICE and deterministic ML need no
   covariance and no decorrelation step, and keep the full aperture. That is why they are the
   current methods for this problem.
5. **The model order is given** (`n_sources = 2`). Order selection on a coherent covariance
   reports one source, which is the same symptom; it is out of scope here.

### 4.3 Aircraft B is constructed in radar coordinates

Aircraft B is **not** a second trajectory. It is placed at aircraft A's range and elevation,
offset in `u` by `separation_rayleigh × Δu_R`, and given aircraft A's radial velocity.
Co-range and co-Doppler hold by construction, and they would not otherwise. A pair flying in
parallel on this track, same velocity vector, offset across the line of sight:

| Separation | Chord range offset | Radial-velocity offset (cross-range speed 41.72 m/s) |
| :--- | ---: | ---: |
| 2× Rayleigh at 16.64 km | 523.0 m = 6.98 range bins | 10.542 m/s = 176.4 velocity bins |
| 1× Rayleigh at 16.64 km | 130.2 m = 1.74 range bins | 5.229 m/s = 87.5 velocity bins |
| 0.5× Rayleigh at 16.64 km | 32.5 m = 0.43 range bins | 2.609 m/s = 43.7 velocity bins |

A real pair shares a cell only when it flies nearly along the line of sight, abreast and close.
The scenario imposes the shared cell because that cell is what it exists to study, and says so.

### 4.4 Aircraft B's scattering phase

Co-range means the two echoes have the same propagation phase. Without more, the relative phase
at the phase centre would always be zero, which is FB's degenerate case and the most favourable
case for the bound (§3.4). A real airframe's echo phase changes with aspect from look to look. So
the second path carries a **scattering phase**, drawn uniformly per frame from the scenario's
seeded generator in F1 and F2, and set explicitly in tests.

Today's generators discard the phase of `amplitude_linear` and rebuild it from `delay_s`; the
`PropagationPaths` docstring records this as a gap in D1. 006 closes it: `PropagationPaths`
gains an optional field `scattering_phase_rad`, shape `(n_paths,)`, default zeros, added to the
propagation phase by both generators. Zeros reproduce today's cubes bit for bit.

### 4.5 Module split and the estimator interface

- `array/geometry.py`: `ArrayGeometry` (frozen: `element_positions_m`, boresight), `ula`,
  `is_centro_symmetric`, `site_to_array_frame`.
- `array/beamforming.py`: `steering_vectors(geometry, u, wavelength_m)`, `u` of shape
  `(n_directions, 3)` (D13 item 3), returning `(n_rx, n_directions)`,
  `conventional_weights`, `mvdr_weights` (returned, never applied), `beam_power`.
- `array/doa.py`: a `DoAEstimator` base class in pyroomacoustics' pattern [13]:
  `estimate(snapshots, n_sources) -> DoAResult`, where `DoAResult` carries `u`, the spectrum
  and its grid where one exists, the covariance used, and its eigenvalues. Subclasses:
  `Bartlett`, `Capon`, `Music`, `RootMusic`, `Esprit`, `Iaa`, `Spice`, `DeterministicMl`.
  Free functions: `sample_covariance`, `spatial_smoothing`, `forward_backward`,
  `crb_deterministic`, `resolution_snr_db`, `monopulse_ratio`, `is_resolved`.

Every estimator exposes its intermediate quantities (`starter.md` §1.1 item 6): the covariance
eigenvalues, which show the rank deficiency directly, the pseudo-spectrum, and the √CRB at the
estimate. The IAA grid covers the whole visible region `u ∈ [−1, 1)`. A sector grid makes
`Σ p_k a_k a_kᴴ` ill-conditioned (condition number ~10¹⁷ in the prototype), and IAA then
diverges.

Each subclass docstring derives its method from the one before: Bartlett → Capon (adaptive
weights) → MUSIC (subspace) → FBSS-MUSIC (rank restored) → IAA (Capon's weights refitted to one
snapshot, iterated) → ML (the model fitted exactly).

### 4.6 `doa.csv`

| Column | dtype | Unit | Meaning |
| :--- | :--- | :--- | :--- |
| `frame`, `time_s` | int, float64 | —, s | As `truth.csv` |
| `detection_id` | int | — | The CFAR detection whose chip was used |
| `method` | string | — | `bartlett`, `capon`, `music`, `fbss_music`, `fbss_esprit`, `iaa`, `spice`, `dml` |
| `source_index` | int | — | 0, 1, sorted by `direction_cosine_x` |
| `direction_cosine_x` | float64 | — | Estimated `u` |
| `azimuth_boresight_deg`, `azimuth_deg` | float64 | deg | `u` converted with `el = 0` (§3.2), then to a bearing |
| `range_m`, `velocity_mps` | float64 | m, m/s | Refined on the chip, **parent coordinates** (D9) |
| `range_bin`, `doppler_bin` | int | — | Peak cell index in the **parent** map |
| `snr_db` | float64 | dB | Per-channel cell SNR estimate |
| `crb_std_direction_cosine` | float64 | — | √CRB evaluated at the estimate |

### 4.7 Amendments to `spec/structure.md`

> **Applied.** These amendments are folded into `spec/structure.md`; it is the authoritative
> tree and decision list. This section records what changed and why.

1. **Part B tree.** `array/geometry.py` reads "ULA, URA, …; phase centre at origin; site-to-array
   rotation (D13)"; `array/beamforming.py` reads "steering vectors (D13), conventional, MVDR
   weights"; `array/doa.py` reads "DoAEstimator base + Bartlett, Capon, MUSIC, Root-MUSIC,
   ESPRIT, spatial smoothing, FB, IAA, SPICE, deterministic ML; CRB, resolution limit".
   `core/signal.py` reads "baseband synthesis (single and multi-channel), superposition, noise,
   phase noise".
2. **B.1 rows.** `array/beamforming.py`: Capon (1969); Phased-Array-Antenna-Model steering
   vectors. `array/doa.py`: unchanged row, plus "Shan–Wax–Kailath, Pillai–Kwon, Yardibi et al.
   (IAA), Stoica–Babu–Li (SPICE), Li–Stoica (RELAX), Stoica–Nehorai (CRB), all written from the
   papers".
3. **Decisions D13 and D14**, as §4.1 and §4.2.
4. **D1 note.** `PropagationPaths` gains `scattering_phase_rad` (default zeros), closing the gap
   its docstring records (§4.4).

### 4.8 Workstream boundary

This scenario claims:

```
scenarios/scenario_006_parallel_{resolved,subrayleigh,sweep}.toml
spec/scenario-006-parallel-approach-doa.md
src/radar_forge/array/{__init__,geometry,beamforming,doa}.py
src/radar_forge/core/signal.py            (rx_response_linear, scattering_phase_rad)
src/radar_forge/pipelines/scenarios.py    ([second_aircraft], [receive_array], [doa]; explicit axes)
src/radar_forge/pipelines/doa.py          (detection -> chip -> estimators -> doa.csv)
notebooks/006_parallel_approach_doa.ipynb
```

Consumed, never edited: `core/{dsp,windows,detection,radar,radar_equation,constants}.py`,
`core/datachip.py` (004's), `pipelines/trajectories.py`, `data/flight_coordinates.csv`.
`test_scenario_001.py`, `test_scenario_002.py` and `test_scenario_003.py` staying green is a
precondition for every commit. `array/{adaptive,stap,calibration,impairments}.py` belong to
007–010.

---

## 5. Build Sequence

```
  1. array/geometry.py          ArrayGeometry, ula (centred), site_to_array_frame
             |
             v
  2. array/beamforming.py       steering_vectors (D13), conventional, mvdr_weights
             |
             v
  3. core/signal.py (widen)     rx_response_linear -> (n_pulses, n_samples, n_rx)
             |                    scattering_phase_rad on PropagationPaths
             v
  4. array/doa.py               covariance, smoothing, FB; Bartlett, Capon, MUSIC,
             |                    Root-MUSIC, ESPRIT; IAA, SPICE, ML; CRB, monopulse
             v
  5. pipelines/scenarios.py     [second_aircraft] [receive_array] [doa]; fast_time_axis=1
             |
             v
  6. pipelines/doa.py           CFAR -> DataChip (004) -> D14 snapshot -> estimators
             |
             v
  7. scenarios/*.toml, scripts/run_scenario.py writes doa.csv
             |
             v
  8. tests, then notebooks/006_parallel_approach_doa.ipynb (F3 curves)
```

| Step | Module | Notes |
| :--- | :--- | :--- |
| 1 | `array/geometry.py` *(new)* | Pure geometry; no signal code. |
| 2 | `array/beamforming.py` *(new)* | The D13 sign lives here and nowhere else. |
| 3 | `core/signal.py` *(widen)* | Two optional arguments; the 2-D path is bit-identical (A12). |
| 4 | `array/doa.py` *(new)* | Each estimator lands with its analytic test (A3–A6) before the scenario tests. |
| 5 | `pipelines/scenarios.py` *(edit)* | TOML order: `[scenario] [target] [trajectory] [second_aircraft] [[burst]] [receiver_site] [receive_array] [receiver] [doa]`. |
| 6 | `pipelines/doa.py` *(new)* | Consumes `core/detection.py::cfar_detect_2d` on the beam-summed map, and 004's `crop`. |
| 7 | `scenarios/*.toml` *(new)* | Three files differing only in `[second_aircraft]`. |
| 8 | tests, notebook | F3's curves are the notebook's; the suite checks one point (A14). |

`core/dsp.py`, `core/windows.py`, `core/detection.py` and `core/datachip.py` are consumed
unchanged. pyroomacoustics is not a dependency: the A15 cross-check imports it with
`pytest.importorskip` and is skipped when it is absent.

---

## 6. Acceptance Criteria

Unit criteria live in `tests/array/test_{geometry,beamforming,doa}.py`; scenario criteria in
`tests/pipelines/test_scenario_006.py`. "Trials" are seeded snapshot-model draws (§3.6).

| # | Criterion | Bound | Justification | Applies to |
| :--- | :--- | :--- | :--- | :--- |
| A1 | **Broadside.** `a(u = 0)` is all ones; `‖a‖² = n_rx` | `rtol = 1e-12` | A few float64 operations | unit |
| A2 | **Endfire and sign.** At `u = +1`, the phase step between adjacent elements is `+π` at d = λ/2; for `u > 0` the element at larger x leads | `rtol = 1e-12` | Catches the `exp(−j…)` sign that mirrors every angle | unit |
| A3 | **Single source, noiseless.** Root-MUSIC and ESPRIT return `u = 0.2` | `atol = 1e-10` | Exact algebra; catches a root-selection or rotation sign error | unit |
| A4 | **Single source, 20 dB, 400 trials.** \|bias\| of MUSIC, Root-MUSIC, ESPRIT | ≤ 0.25 √CRB | Monte-Carlo SE of the mean is 0.05–0.09 √CRB at n = 400 | unit, `slow` |
| A5 | Same trials: RMSE / √CRB | MUSIC, Root-MUSIC ≤ 1.2; LS-ESPRIT ≤ 2.0 | Prototype 1.02 and 1.75; LS-ESPRIT is known to be inefficient | unit, `slow` |
| A6 | **CRB closed form.** `crb_deterministic` for one source equals `6σ²/(π²|s|² n_rx(n_rx²−1))` | `rtol = 1e-10` | Catches the `σ²` vs `σ²/2` factor | unit |
| A7 | **Coherence is rank one.** Noiseless chip covariance of a co-cell pair: `λ₂/λ₁` | < 1e-12 | §2.1, exact | F2 |
| A8 | **The baseline fails.** MUSIC and ESPRIT on the raw chip covariance, F2 geometry at 38.23 dB, 200 trials: P(resolved) | ≤ 0.25 | Prototype 0.005–0.085; the 99.9 % Clopper–Pearson upper limit at 17/200 is 0.17 | F2, `slow` |
| A9 | **FB alone is phase-blind.** Noiseless, equal powers: rank of `R_fb` | 1 at relative phase 0 and π; 2 at π/2 | `J x* = a_1 + e^{−jφ} a_2` is parallel to `x` iff φ ∈ {0, π} | unit |
| A10 | **The current methods succeed.** Same trials as A8: P(resolved) for FBSS-MUSIC, FBSS-ESPRIT, IAA, ML | ≥ 0.95 | Prototype 1.000 for all four | F2, `slow` |
| A11 | Same trials: RMSE / √CRB | ML ≤ 1.3; FBSS-ESPRIT, FBSS-MUSIC ≤ 1.5; IAA ≤ 4.0 | Prototype 1.04, 1.00–1.12, 2.87. ML ≤ 1.3 fails a CRB off by √2 | F2, `slow` |
| A12 | **Single channel unchanged.** Without `rx_response_linear`, the 2-D cube equals today's for every 001 variant | bit-identical | The widening must not move 001–003 | 001 |
| A13 | **Snapshot model equals the chain.** Noiseless F2 frame: the chip's peak-cell snapshot is `g·A(u)s`, with the truth `u` and scattering phase | `rtol = 1e-9` per channel | Catches a sign flip between `signal.py` and `beamforming.py`; licenses F3 | F2 |
| A14 | **One point of F3.** Δu = 0.5 Rayleigh, 20 dB, 200 trials: P(resolved) | ML ≥ 0.90; FBSS-ESPRIT ≥ 0.85; IAA ≥ 0.65 | Prototype 0.99, 0.94, 0.80; 99.9 % Clopper–Pearson lower limits 0.94, 0.86, 0.69 | F3, `slow` |
| A15 | **Cross-check.** pyroomacoustics MUSIC on an incoherent two-source covariance agrees with `Music` | within one grid step | Skipped when absent; mapping its azimuth convention to D13 to confirm at implementation | unit |
| A16 | **Chip in parent coordinates.** `doa.csv` range, velocity and bin indices equal those from the uncropped cube; `u` identical | `rtol = 1e-12` | D9's contract | F1, F2 |
| A17 | **Conventional beam, in phase.** Equal powers, relative phase 0, Δu ≤ 0.5 Rayleigh: one mainlobe peak at the midpoint | within one 2.5e-4 grid step | Exact by symmetry of `|aᴴ(a_1 + a_2)|²` about the midpoint | unit |
| A18 | **Conventional beam, power centroid.** −6 dB second source, phase-averaged covariance: the one local maximum within ±Δu_R of the power centroid lies near it | ≤ 0.1 Δu from the centroid, at Δu ≤ 0.5 Rayleigh | Prototype 0.070 Δu at 0.5, 0.016 Δu at 0.25 | unit |
| A19 | **Per-look wander.** Equal powers, half-array phase monopulse over relative phase: the per-look indication leaves `[u_1, u_2]`; the phase-averaged indication is within 0.1 Δu of the power centroid. At phase π the conventional beam peaks **outside** the pair | as stated | Prototype: per-look −0.0987 against a pair at [−0.0112, 0.0513]; averaged error 0.042 Δu | unit |
| A20 | **MVDR is distortionless.** `w(u₀)ᴴ a(u₀) = 1`; with `R = σ²I` MVDR equals the conventional beam | `rtol = 1e-12` | The defining constraint | unit |

> A8 and A10 are two-target resolution tests, which `docs/conventions/testing.md` §10.1
> normally excludes because window and exact-bin tests already imply them. Here they do not.
> Whether a covariance resolves a coherent pair is a property of the snapshot set, not of the
> window, and it is the bug this scenario exists to catch: an implementation that builds its
> covariance from pulses and calls the result super-resolution.

---

## References

.. [1] R. O. Schmidt, "Multiple emitter location and signal parameter estimation," *IEEE Trans.
       Antennas Propag.*, 34(3), pp. 276–280, 1986.
.. [2] A. J. Barabell, "Improving the resolution performance of eigenstructure-based
       direction-finding algorithms," *Proc. IEEE ICASSP*, vol. 8, pp. 336–339, 1983.
.. [3] R. Roy and T. Kailath, "ESPRIT — estimation of signal parameters via rotational
       invariance techniques," *IEEE Trans. Acoust., Speech, Signal Process.*, 37(7),
       pp. 984–995, 1989.
.. [4] T.-J. Shan, M. Wax and T. Kailath, "On spatial smoothing for direction-of-arrival
       estimation of coherent signals," *IEEE Trans. Acoust., Speech, Signal Process.*, 33(4),
       pp. 806–811, 1985.
.. [5] S. U. Pillai and B. H. Kwon, "Forward/backward spatial smoothing techniques for coherent
       signal identification," *IEEE Trans. Acoust., Speech, Signal Process.*, 37(1), pp. 8–15,
       1989.
.. [6] T. Yardibi, J. Li, P. Stoica, M. Xue and A. B. Baggeroer, "Source localization and
       sensing: a nonparametric iterative adaptive approach based on weighted least squares,"
       *IEEE Trans. Aerosp. Electron. Syst.*, 46(1), pp. 425–443, 2010.
.. [7] P. Stoica, P. Babu and J. Li, "SPICE: a sparse covariance-based estimation method for
       array processing," *IEEE Trans. Signal Process.*, 59(2), pp. 629–638, 2011.
.. [8] J. Li and P. Stoica, "Efficient mixed-spectrum estimation with applications to target
       feature extraction," *IEEE Trans. Signal Process.*, 44(2), pp. 281–295, 1996.
.. [9] S. M. Sherman and D. K. Barton, *Monopulse Principles and Techniques*, 2nd ed., Artech
        House, 2011. The chapter on response to unresolved targets (title only; chapter number
        to confirm at implementation).
.. [10] J. Capon, "High-resolution frequency-wavenumber spectrum analysis," *Proc. IEEE*, 57(8),
        pp. 1408–1418, 1969.
.. [11] P. Stoica and A. Nehorai, "MUSIC, maximum likelihood, and Cramér-Rao bound," *IEEE
        Trans. Acoust., Speech, Signal Process.*, 37(5), pp. 720–741, 1989.
.. [12] S. T. Smith, "Statistical resolution limits and the complexified Cramér-Rao bound,"
        *IEEE Trans. Signal Process.*, 53(5), pp. 1597–1609, 2005.
.. [13] LCAV, pyroomacoustics, `pyroomacoustics.doa`, https://github.com/LCAV/pyroomacoustics
        (MIT licence).
