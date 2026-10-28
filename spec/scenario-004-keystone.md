# Scenario 004 — High-speed target: keystone, polynomial-phase compensation and the datachip

## 1. Status & Purpose

**Status: specified, not implemented.** The fourth vertical slice, and the first in which the
target moves far enough within one coherent processing interval (CPI) to matter.

Scenario 001 §4.2 records the approximation every slice so far has relied on: at 2 MHz bandwidth
an 80 m/s target moves a quarter of a 74.95 m range bin per CPI, so range migration "is
negligible". That stays true only while the range bins are coarse and the target is slow. This
slice changes both. It uses a wideband pulse (500 MHz, 0.3 m resolution) and a fast,
accelerating target, so that over one 20.48 ms CPI the echo **walks across 28 range bins** and an
acceleration **smears the Doppler peak over 4 bins**. Plain range-Doppler processing then loses
**26.5 dB** of its 30.1 dB coherent gain, and the target falls below any detection threshold.

The slice builds four things, in the order the signal meets them:

1. **Migration correction** (`core/migration.py`, decision D10). The keystone transform removes
   linear range walk for every target at once, without knowing its velocity. A Doppler-rate
   matched filter in the range-frequency domain then removes the quadratic and cubic phase that
   acceleration and jerk leave behind. A fold-factor search handles Doppler-ambiguous targets.
2. **Polynomial-phase estimation.** "Polyphase" is read here as
   *polynomial-phase* motion compensation: velocity, acceleration and jerk are the first three
   coefficients of the slow-time phase. (The other reading, the polyphase channeliser, belongs to
   scenario 009.)
3. **The datachip** (`core/datachip.py`, decision D9). A small crop of the cube around a
   detection, carrying its offsets into the parent and the parent's bin-centre axes. Scenario 006
   consumes it for angle estimation on multi-channel chips.
4. **Fine parameter estimation** on the chip. Sub-bin range and velocity come from interpolation
   on Fourier coefficients, and acceleration comes from the cubic phase function. A local
   maximum-likelihood refinement then brings every estimate to within a stated factor of the
   Cramér–Rao bound (CRB). This search is too expensive over the whole cube, but on the chip it
   is cheap, and that is the reason the chip exists.

Single channel, monostatic. It extends scenario 001's pulsed chain (S2: LFM pulse, matched filter)
and does not touch the FMCW path.

| Section | What it fixes |
| :--- | :--- |
| §2 Overview & Geometry | The target, why walk appears, the processing chain |
| §3 Parameter Matrix | Waveform, link budget, the three variants, derived walk/spread/loss, contracts, schema |
| §4 Core Decisions | D9 `DataChip`, D10 migration on `data_rfst`; kinematic paths; what is not modelled |
| §5 Build Sequence | What is new, what is widened, what is untouched |
| §6 Acceptance Criteria | Analytic bounds, led by the baseline that fails |

### 1.1 Upstream inspiration

| Upstream | What it is | What radar-forge takes | Licence | Baseline or current |
| :--- | :--- | :--- | :--- | :--- |
| Moving-target detection (MTD): range-compress, then Doppler FFT per range bin [11] | The textbook pulse-Doppler chain; scenario 001 S2 | The uncompensated reference whose loss §6 predicts and asserts | textbook, written from equations | **Baseline** |
| Perry, DiPietro & Fante 1999 [1] | Origin of the keystone transform: rescale slow time per range frequency so linear walk vanishes for all velocities at once | First-order keystone in `core/migration.py` | paper, written from equations | Current (for linear walk) |
| Zhu, Li & Zhu 2007 [2] | Keystone without interpolation, by a scaled transform | The keystone's time scaling as a chirp-z transform per range frequency; truncated-sinc interpolation kept as the baseline | paper, written from equations | Current (implementation) |
| `scipy.signal.czt` | Chirp-z transform | The scaled DFT inside the keystone; already a core dependency | BSD-3-Clause | Dependency |
| Xu, Yu, Peng & Xia, RFT I–III, 2011–2012 [3]–[5] | Radon–Fourier transform: a generalised Doppler filter bank that integrates along range-walk trajectories; blind-speed sidelobes; optimality | RFT as the test oracle for linear walk; its blind-speed-sidelobe analysis informs the fold-factor search | paper, written from equations | Oracle |
| Xu et al. 2012, GRFT [6] | Generalised RFT: the maximum-likelihood estimator of high-order motion parameters, with their CRB | Local GRFT on the chip as the ML refinement; its CRB form for the acceptance bounds | paper, written from equations | **Current** (on the chip), oracle (on the cube) |
| Huang, Liao, Yang, Xia & Ma 2016 [7] | Keystone, then matched filtering in range frequency and slow time, for velocity, acceleration and jerk, with the Doppler-ambiguous cases | The K2 and K3 chain: keystone, Doppler-rate matched filter, fold factor | paper, written from equations | **Current** |
| Kirkland 2011 [8] | Second-order keystone transform, for range curvature | `second_order_keystone`, unit-tested; not needed by K2 (§3.4) | paper, written from equations | Current (extension) |
| Peleg & Porat 1991, HAF [9] | High-order ambiguity function for polynomial-phase signals | Acceleration and jerk baseline on the chip | paper, written from equations | **Baseline** |
| Peleg & Porat 1991, CRB [10] | CRB for constant-amplitude polynomial-phase signals | Bound for the acceleration and jerk criteria | paper, written from equations | Oracle |
| O'Shea 2004, CPF [12] | Cubic phase function: quadratic-FM parameters by 1-D maximisations | Acceleration and jerk on the chip, as the start for ML | paper, written from equations | Current |
| Rife & Boorstyn 1974 [13] | ML single-tone estimation and its CRB | The tone CRB; the frame for every sub-bin estimator | paper, written from equations | Oracle |
| Parabolic interpolation on \|X\|; Quinn 1994 [14]; Jacobsen & Kootsookos 2007 [15]; Candan 2011 [16] | Three-bin DFT interpolators | Sub-bin range and velocity baselines; parabolic is the one shown to fail | paper, written from equations | **Baseline** |
| Aboutanios & Mulgrew 2005 [17] | Iterative interpolation on Fourier coefficients; near-CRB everywhere on the bin | Sub-bin range and velocity on the chip | paper, written from equations | **Current** |

No code is vendored. File Exchange implementations of GRFT and of the second-order keystone exist,
but none is a reference here, because this spec is written from the papers.

---

## 2. Overview & Geometry

A wideband X-band tracking radar sited downrange of a sounding-rocket launch pad observes one fast
target in the radial direction: a sounding rocket flying towards it along the radar's line of
sight, coasting (K1), under thrust (K2) or after burnout (K3). The rocket is closing throughout,
so under thrust its closing speed increases. The
geometry is one-dimensional. Only range and its derivatives matter, so there is no trajectory
file. The target follows a polynomial range history about the centre of the CPI:

```
   R(t) = R0 - v t - a t^2/2 - j t^3/6          closing positive (structure.md D5)
          t in [-T/2, T/2),  t = 0 at pulse n_pulses // 2

   [ radar ]  <-----------  R0 = 2500 m  ------------  [ target ]  ===> closing at v
                                                         a: closing speed increasing
```

Walk is what the target does during a CPI. What it costs depends on the bin size:

```
  range bin index (0.2498 m each)    0    5   10   15   20   25   30
  pulse 0       (t = -T/2)                                       ##
  pulse 256                                               ##
  pulse 512     (t = 0)                            ##
  pulse 768                                 ##
  pulse 1023    (t = +T/2)           ##
                                     |<----- 27.87 bins: K1 ----->|

  Uncompensated: each range bin holds ~1/28 of the pulses     -> -26.52 dB
  After keystone: every pulse in one bin                      ->  -0.05 dB (grid)
```

The chain this slice builds. New stages are in capitals; everything else exists:

```
  scenarios/scenario_004_keystone_k{1,2,3}.toml
            |       [trajectory] kind = "polynomial"   [[burst]] range gate
            v
  pipelines/trajectories.py    polynomial radial history -> truth at CPI centre
            |
            v
  core/propagation.py  line_of_sight_paths(+accel, +jerk) -> PropagationPaths (+2 defaulted fields)
            |
            v
     pulsed_baseband(..., range gate)  + thermal_noise    -> data_ftst (n_pulses, n_gate)
            |
            v
  core/dsp.py  matched_filter, trim group delay           -> data_rfst (n_pulses, n_range_bins)
            |
            +-------------------------------> doppler_fft -> MTD map (BASELINE, loses 26.5 dB)
            v
  core/migration.py   RANGE SPECTRUM  (FFT along range, zero-padded, signed f)
            |          KEYSTONE  t -> f0/(f0+f) tau       [fold factor F]
            |          DOPPLER-RATE MATCHED FILTER  (a, j)   search curve exposed
            |          back to data_rfst
            v
  core/dsp.py  doppler_fft -> data_rv  -> core/detection.py cfar_detect_2d, cluster_detections
            |
            v
  core/datachip.py   CROP -> DataChip (offsets, parent axes, processing metadata)
            |          FINE ESTIMATION: Aboutanios-Mulgrew (R, v), CPF (a, j), local GRFT (ML)
            v
     FineEstimate in PARENT coordinates, with its CRB      -> estimates.csv, products.h5
```

---

## 3. Parameter Matrix

### 3.1 Waveform: the same in all three variants

All numbers below come from `derive.py` and `crb.py` in the scenario's scratch directory. They
will be re-derived in tests from `core/radar.py` rather than read from this table.

| Parameter | Value | Why |
| :--- | :--- | :--- |
| Waveform | LFM pulse, matched filter | Scenario 001 S2's chain; the range spectrum after compression is what the keystone acts on |
| `f0_hz` | 9.8 GHz → λ = **30.5911 mm** | Scenario 001's carrier, so the Doppler numbers compare directly |
| `bandwidth_hz` | 500 MHz → c/2B = **0.29979 m** | Fractional bandwidth B/f0 = **0.05102**. The keystone's correction scales with it; at 2 MHz it would be 2·10⁻⁴ and walk would not exist |
| `chirp_duration_s` | 0.5 µs, time-bandwidth product **250** | Short enough that intra-pulse motion is 0.55 mm at 1100 m/s (§4.6) |
| `sample_rate_hz` | 600 MHz, fs/B = 1.2 → range bin c/2fs = **0.24983 m** | Complex sampling with 20% margin; 300 reference samples |
| `prf_hz` | 50 kHz, PRI 20 µs, duty 0.025 | v_ua = λ·PRF/4 = **±382.3883 m/s**, so K1/K2 at 340 m/s are unambiguous and K3 is not |
| `n_pulses` | 1024 → CPI T = **20.480 ms** | Long enough for walk and Doppler spread to be several bins; ideal gain 10·log10(1024) = **30.10 dB** |
| Velocity resolution | λ/2T = **0.74685 m/s** (48.8281 Hz) | |
| Unambiguous range | c/2PRF = **2997.925 m** | The target sits inside it: range folding is scenario 001's lesson, not this one's |
| Range gate | `range_gate_start_m` = 2400 m, `n_gate_samples` = 1024 (1.7067 µs, 255.823 m) | Fully-compressed span **181.125 m**, covering 2400–2581 m. Without a gate the cube would be 1024 × 12 000 samples |
| `transmit_power_w` | 100 W | Sets the link budget below |
| `gain_tx_dbi`, `gain_rx_dbi` | 25.0 dBi each | |
| `noise_figure_db` | 3.0 dB | As scenario 001 |
| IQ cube | `(1024, 1024)` complex128, **16.78 MB** | Range spectrum `(1024, 2048)`, 33.55 MB |

**Why this waveform and no other.** For an unambiguous target (|v|·PRI < λ/4), the walk over a CPI
is bounded by `N·B/(2·f0)` resolution cells, which is **26.12 cells** (31.35 sample bins) here.
Walk therefore needs fractional bandwidth *and* pulses. Scenario 001 S2 (2 MHz, 256 pulses) walks
**0.0109** of a bin at 80 m/s. No target speed fixes that without making Doppler ambiguous, and
that is why the bandwidth had to change.

### 3.2 Link budget, at R0 = 2500 m, RCS 0.1 m² (−10 dBsm)

| Quantity | Value |
| :--- | ---: |
| Received power | 1.2073e-14 W |
| Thermal noise kT₀BF at 500 MHz | 3.9944e-12 W |
| SNR per fast-time sample | −25.20 dB |
| After pulse compression, E/N₀ per pulse | **−1.22 dB** |
| After 1024 pulses, ideally integrated, ρ | **28.89 dB** |

The target is invisible in any one pulse and comfortably detectable after ideal integration, but
**not** after uncompensated integration (§3.4). The RCS is small because the scenario is about
integration gain; a 10 m² target would detect even with the 26.5 dB loss and teach nothing.

### 3.3 The three variants

| | **K1** `keystone-walk` | **K2** `keystone-accel` | **K3** `keystone-ambiguous` |
| :--- | :--- | :--- | :--- |
| Target | sounding rocket, coasting | sounding rocket under thrust | sounding rocket after burnout |
| `radial_velocity_mps` at t = 0 | 340.0 | 340.0 | 1100.0 |
| `radial_acceleration_mps2` | 0 | **147.0998** (15 g) | 0 |
| `radial_jerk_mps3` | 0 | 0 (see §4.5) | 0 |
| Doppler f_d / PRF | 0.4446 | 0.4446 (±1.506 m/s over the CPI) | **1.4383** |
| Fold factor F | 0 | 0 | **1**, folded to 335.2233 m/s |
| What it adds | linear walk | + quadratic phase (Doppler spread) | + Doppler ambiguity |
| Method | first-order keystone | keystone + Doppler-rate matched filter [7] | keystone + fold-factor search [7] |

All three share `range_m` = 2500 m at t = 0 and the waveform of §3.1.

### 3.4 Derived walk, spread and loss, per variant

Losses are the peak of the processed range-Doppler map relative to the ideal matched value
`N·K` (K = 1707 in-band range-frequency samples), with the peak refined on a 1/16-bin grid in
both axes so that scalloping does not masquerade as migration. The ideal cube itself reads
−0.049 dB on that grid; that is the grid's own residual and is included in every figure below.

| Quantity | K1 | K2 | K3 |
| :--- | ---: | ---: | ---: |
| Range walk v·T | 6.9632 m | 6.9632 m | 22.5280 m |
| … in sample bins (0.2498 m) | **27.872** | **27.872** | **90.174** |
| … in resolution cells (0.2998 m) | 23.227 | 23.227 | 75.145 |
| Range curvature a·T²/8 | 0 | 7.7123 mm = **0.031 bins** | 0 |
| Doppler spread 2aT/λ | 0 | 196.960 Hz = **4.034 bins** | 0 |
| Quadratic phase at the CPI edge | 0 | 3.168 rad | 0 |
| Slow-time frequency across the band, ν(f0 ± B/2)/f0 | 0.4332 – 0.4559 | 0.4332 – 0.4559 | 1.4016 – 1.4750 |
| **MTD (no compensation) loss** | **−26.52 dB** | **−25.53 dB** | **−31.80 dB** |
| Keystone, F = 0 | **−0.05 dB** | −4.06 dB | −32.94 dB |
| Keystone + Doppler-rate matched filter (true a) | — | **−0.055 dB** | — |
| Dechirp only, no keystone | — | −26.51 dB | — |
| Keystone, F = 1 | — | — | **−0.06 dB** |
| Integrated SNR, MTD | 2.37 dB | 3.36 dB | −2.91 dB |
| Integrated SNR, compensated | ≈ 28.8 dB | ≈ 28.8 dB | ≈ 28.8 dB |

Three things to read from the table:

- **K2's acceleration shows up as Doppler spread, not as range curvature.** 15 g bends the range
  history by 7.7 mm, 0.031 of a bin, but it puts 3.17 rad of quadratic phase across the CPI. So
  the second-order keystone (which corrects curvature) is the wrong tool here, and the
  Doppler-rate matched filter (which corrects phase) is the right one. Each step fixes one thing.
  The keystone alone leaves 4.06 dB on the table; dechirping alone, without the keystone, recovers
  nothing (−26.51 dB).
- **K3 keystoned with the wrong fold factor is worse than not keystoning at all** (−32.94 vs
  −31.80 dB). The keystone rescales the alias, not the true frequency. §3.5 gives the correction.
- **K3 is near the edge of one fold factor.** The fold factor is constant across the band only
  while ν(f0 + B/2)/f0 stays below F + 1/2. For F = 1 that holds up to **1118.63 m/s**; K3 at
  1100 m/s has 1.4750 < 1.5. A faster target spans two fold factors across the band, which [7]
  treats as a separate case. It is out of scope here and the code must detect it and say so.

### 3.5 Mathematical contracts

| Contract | Statement |
| :--- | :--- |
| Range history | `R(t) = R0 − v t − a t²/2 − j t³/6`, closing positive, t = 0 at pulse `n_pulses // 2`. `R0`, `v`, `a`, `j` are reported **at the CPI centre** |
| `data_ftst` | `(n_pulses, n_gate_samples[, n_rx])` complex128, from `pulsed_baseband` with a range gate. Sample 0 is delay `2·range_gate_start_m/c` |
| `data_rfst` | `(n_pulses, n_range_bins[, n_rx])` complex128: `matched_filter` output with the reference's group delay `n_reference − 1` trimmed, as `pipelines/scenarios.py` does today. Range axis `range_gate_start_m + k·c/(2·fs)` |
| Range spectrum (internal to D10) | `FFT(data_rfst, n = n_range_fft, axis = 1)`, unshifted. Its frequency axis is the **signed** baseband `range_frequency_hz = fftfreq(n_range_fft, 1/fs)`, so the true RF frequency is `f0 + f`. `n_range_fft ≥ n_range_bins + max walk`, default the next power of two ≥ `2·n_range_bins` (2048 here) so walk cannot wrap |
| Signal model in the range spectrum | `S(f, t) = A·\|P(f)\|²·exp(−j·4π·(f0 + f)·R(t)/c)`; the `f·v·t` term is the walk, the `f0·a·t²` term the spread |
| Keystone | `S_K(f, τ) = S(f, t = f0/(f0 + f)·τ)`, evaluated as band-limited (DFT) interpolation in slow time, by a chirp-z transform per range frequency. Reference index `n_pulses // 2`. After it, the linear term is `exp(+j·4π·f0·v·τ/c)`, independent of f [1] |
| Fold factor | With `v = v_folded + F·λ·PRF/2`, band-limited interpolation sees the alias, so multiply by `exp(+j·2π·F·(f0/(f0+f))·m_c)` after the keystone, `m_c = m − n_pulses//2`. F is found by maximising the integrated peak over `F ∈ [−F_max, F_max]` [7] (equation number to confirm at implementation) |
| Doppler-rate matched filter | After the keystone, multiply by `H(f, τ) = exp(−j·(4π/c)·[f0²/(f0+f)·(a/2)·τ² + f0³/(f0+f)²·(j/6)·τ³])`, derived by substituting the keystone into the model above; the form of [7] (equation number to confirm at implementation). `a` (and `j`) by 1-D (2-D) search; the search curve is returned |
| Second-order keystone | `t = sqrt(f0/(f0 + f))·τ` [8]; removes the `f·a·t²` curvature term. Exposed, unit-tested, not in the K2 chain |
| Acceleration from phase | Slow-time phase `φ(t) = (4π f0/c)(v t + a t²/2 + j t³/6)`, so `a = (λ/2)·ḟ_d` and `j = (λ/2)·f̈_d` |
| Velocity in parent coordinates | `v = velocity_bin_centers_mps` at the fractional Doppler index, plus `F·λ·PRF/2` |
| CRB | Fisher information of the model above with complex white noise, parameters `(|A|, φ, R0, v, a[, j])`, `F = (2/σ²)·Re(JᴴJ)` [6], [10], [13]. Single-tone special case `var(ω̂) ≥ 6/(SNR·N(N²−1))` [13] |
| Shapes | Every `core/migration.py` and `core/datachip.py` function accepts a trailing `n_rx` axis and acts on axes 0 and 1 only |

CRB at the nominal ρ = 28.89 dB, and at 20 dB (the lowest SNR at which §6 asserts efficiency):

| Parameter | CRB std, ρ = 28.89 dB | in bins | CRB std, ρ = 20 dB | in bins |
| :--- | ---: | ---: | ---: | ---: |
| Range R0 | 4.1994 mm | 0.01681 | 11.6867 mm | 0.04678 |
| Velocity v | 10.4613 mm/s | 0.01401 | 29.1130 mm/s | 0.03898 |
| Acceleration a | 3.9567 m/s² | — | 11.0111 m/s² | — |
| Jerk j (cubic model) | **2285.9 m/s³** | — | — | — |

Adding the acceleration parameter barely changes the range and velocity bounds (4.1989 → 4.1994 mm,
10.4612 → 10.4613 mm/s): with t centred, odd and even phase terms decouple.

### 3.6 Output schema

As scenario 001 §3.6, written through `spec/data-001-formats.md`, with these additions:

| File | Content |
| :--- | :--- |
| `truth.csv` | gains `radial_acceleration_mps2`, `radial_jerk_mps3`, `fold_factor`; `time_s` is the **CPI centre** |
| `estimates.csv` *(new)* | per frame and detection: `range_m`, `radial_velocity_mps`, `radial_acceleration_mps2`, `radial_jerk_mps3`, `fold_factor`, each with a `_crb_std` sibling (e.g. `range_crb_std_m`), `snr_db`, `estimator` |
| `products.h5` `/chips/burst{k}/chip{i}/` *(new)* | the `DataChip` of §4.1 on disk: `data`, `range_m`, `velocity_mps`, attrs for offsets and processing |
| `products.h5` `/migration/burst{k}/` *(new)* | `acceleration_search_mps2` and its `integrated_peak_linear` curve; `fold_factor_search` and its curve |
| `figures/` | MTD map, keystoned map, compensated map side by side; walk overlay; search curves |

---

## 4. Core Decisions

### 4.1 D9 — the `DataChip` contract

A `DataChip` is a crop of a cube around one detection, carrying everything needed to refine
that detection without the parent. It is a frozen dataclass in `core/datachip.py`:

```python
@dataclass(frozen=True)
class ChipProcessing:  # the parent's processing, as the chip needs it to refine
    f0_hz: float
    bandwidth_hz: float
    sample_rate_hz: float
    prf_hz: float
    n_pulses: int
    n_range_fft: int  # the range spectrum length used by D10
    n_doppler_fft: int
    range_gate_start_m: float
    fast_time_window: NDArray[np.float64] | None  # (n_range_bins,) as applied, or None
    slow_time_window: NDArray[np.float64] | None  # (n_pulses,) as applied, or None
    keystone_applied: bool
    fold_factor: int  # F used, 0 if none
    compensated_acceleration_mps2: float
    compensated_jerk_mps3: float


@dataclass(frozen=True)
class DataChip:
    data: NDArray[np.complex128]  # (n_axis0_chip, n_range_chip[, n_rx]), PARENT axis order
    domain: Literal["slow_time", "doppler"]  # axis 0 is pulses, or fftshifted Doppler bins
    axis0_offset: int  # first pulse (always 0 for slow_time) or Doppler bin
    range_offset: int  # first range bin in the parent
    range_bin_centers_m: NDArray[np.float64]  # (n_range_chip,)  parent values
    velocity_bin_centers_mps: NDArray[np.float64]  # (n_axis0_chip,) for doppler;
    # the parent's full axis for slow_time
    processing: ChipProcessing
```

The rules, each of which is a test (§6, A10–A11):

1. **Parent axis order, untouched.** Axis 0 is slow time or Doppler, axis 1 range, and a trailing
   `n_rx` axis is carried whole and never cropped. Scenario 006 crops `(n_doppler, n_range, n_rx)`
   chips from a multi-channel map with the same function.
2. **Range does not wrap; Doppler does.** A range crop that would leave the parent is clipped,
   the chip gets shorter, and `range_offset` records where it starts. A Doppler crop is taken
   modulo `n_doppler_fft`, because the Doppler axis is circular. A chip straddling ±v_ua
   therefore carries the parent's velocity values verbatim, **including the jump at the fold**.
3. **Exact round trip.** `parent = (chip + offset)` on the range axis and
   `parent = (chip + offset) mod n_doppler_fft` on the Doppler axis, in integers. Fractional
   positions map through the parent's uniform spacing, `range_m = range_bin_centers_m[0] + δ·Δr`,
   so a sub-bin estimate never passes through an interpolation table.
4. **The chip is unwindowed data.** The windows the parent applied for display or detection are
   recorded, not baked in. Fine estimators assume a rectangular window [13], [17]; baking a taper
   in would bias them silently.
5. **Results leave in parent coordinates.** Every estimator returns `range_m`,
   `radial_velocity_mps` (with the fold factor applied), `radial_acceleration_mps2` and their
   CRB, never chip-local indices. The fractional chip indices are exposed as intermediates.

Two crop functions, one per domain: `crop_slow_time(data_rfst, …)` for refinement (the chip keeps
every pulse, so acceleration is still observable), and `crop_range_doppler(data_rv, …)` for angle
estimation (006). A ±16-bin range chip keeps all but **0.034 dB** of a point target's energy at the
worst sub-bin offset, against 0.139 dB for ±4 bins. That is why the default half-width is 16.

### 4.2 D10 — migration correction lives in `core/migration.py` and operates on `data_rfst`

Every public function in `core/migration.py` takes `data_rfst` and returns `data_rfst` of the same
shape. The range spectrum is an internal domain, entered and left inside the module. Three
reasons:

- **The caller's contract does not change.** A keystoned `data_rfst` goes into
  `dsp.doppler_fft` exactly as an uncompensated one does, so the MTD baseline and the compensated
  chain differ by one function call, which is what §6 compares.
- **The range spectrum is fragile.** It needs a signed frequency axis (`f0 + f`, not `f0 + |f|`),
  enough zero-padding that walk cannot wrap, and the same `n_range_fft` on the way back. Keeping
  it inside one module keeps those three facts in one place.
- **It composes.** `keystone`, `doppler_rate_compensation` and `second_order_keystone` each map
  `data_rfst → data_rfst`, so K2 is `doppler_rate_compensation(keystone(x))` and reads that way.

Functions return a result dataclass with the corrected `data_rfst` and their intermediates
(range-frequency axis, scale factors, search curves), never a bare array. That satisfies
`spec/starter.md` §1.1 item 6. The cube is never modified in place.

The keystone's slow-time interpolation has one known artefact, and the spec records it rather
than hiding it: band-limited (DFT) interpolation treats the CPI as periodic, so the **first and
last pulse** ring (they are the only pulses that do; §6 A2 excludes them, by name).

### 4.3 `PropagationPaths` gains two defaulted fields

D1 has survived a bistatic geometry unchanged (scenario 002 §4.1). It does not survive
acceleration unchanged, because a path's Doppler is a single number, so a path set can only
describe constant velocity. Two fields are added, each **defaulting to zero**:

| Field | Shape | Meaning |
| :--- | :--- | :--- |
| `doppler_rate_hzps` | `(n_paths,)` | `2·a/λ`, closing positive |
| `doppler_jerk_hzps2` | `(n_paths,)` | `2·j/λ` |

Every existing builder and test therefore produces the same cube bit for bit (§6 A15).
`line_of_sight_paths` gains `radial_acceleration_mps2` and `radial_jerk_mps3` keywords, default 0.
`pulsed_baseband` advances range per pulse with the full polynomial instead of `v·t`. This widens
D1 rather than changing it, and `structure.md` D1 should record it (§4.8).

The alternative considered was a `range_history_m` override argument on the generator. It was
rejected because it puts geometry into the signal generator and would let a ray-tracing backend
describe its paths in two incompatible ways.

### 4.4 The range gate

`pulsed_baseband` today samples the whole PRI: 12 000 samples at 600 MHz. It gains
`range_gate_start_m: float = 0.0` and `n_gate_samples: int | None = None` (default: the whole
PRI, today's behaviour). Range axes come from a new `dsp.delay_bin_centers_m(n_bins,
sample_rate_hz, *, range_gate_start_m=0.0)`. This is the helper that
`pipelines/scenarios.py::form_range_doppler_map` already says should exist and belongs in
`core/dsp.py`.

### 4.5 Jerk is in the model and is not observable at this CPI

The code estimates velocity, acceleration and jerk. At this
CPI, though, jerk is a thing to measure the absence of. A jerk of 300 m/s³ (about 30 g/s, an abrupt
thrust change such as burnout or staging) puts **0.0221 rad** of cubic phase at the CPI edge. Jerk would need **21 367.6 m/s³** to
reach π/2, and its CRB at the nominal SNR is **2285.9 m/s³**. So the scenario sets `j = 0`. The
chip estimator still fits the cubic model, reports `j` with its CRB, and §6 asserts consistency
with zero. Jerk *recovery* is tested on a synthetic polynomial-phase signal where it is
observable. A teaching scenario should say outright which parameters a CPI can support, and this
is how this one says it.

### 4.6 What is not modelled, with the size of the omission

| Effect | Size here | Why it is left out |
| :--- | :--- | :--- |
| LFM range–Doppler coupling (peak shift f_d/α) | 3.332 mm (K1), **10.780 mm** (K3), against a 4.2 mm range CRB | `pulsed_baseband` is stop-and-hop: no intra-pulse Doppler, so the simulation cannot produce the shift and the estimator must not "correct" it. On recorded data it would be a 2.6σ bias in K3; a future intra-pulse generator must add both |
| Intra-pulse motion | 0.170 mm (K1), 0.550 mm (K3) | Four orders below a bin |
| Two fold factors across the band | none: K3 max 1.4750 < 1.5 | Out of scope (§3.4); detected and refused |
| Range ambiguity, eclipsing | none: 2.5 km < 2.998 km, gate clear of the pulse | Scenario 001's lesson; not repeated |

### 4.7 Why sub-bin estimation is not parabolic interpolation

Parabolic interpolation on |X| is what most code does, and for a rectangular window it is biased
by up to **0.234 bins** even without noise. At ρ = 28.89 dB that bias is **12.2×** the CRB, so
the error budget is entirely bias. The three-bin complex interpolators (Quinn, Jacobsen, Candan)
remove the bias (≤ 1.2·10⁻⁶ bins) and sit within 1.29–1.55× the CRB. Aboutanios–Mulgrew, with two
iterations, reaches **1.01×** the CRB at every offset. All five are implemented, because the
comparison is the lesson. The A&M estimator is the default.

### 4.8 Amendments to `spec/structure.md`

> **Applied.** These amendments are folded into `spec/structure.md`; it is the authoritative
> tree and decision list. This section records what changed and why.

1. **Part B tree:** under `core/`, add
   `migration.py  # keystone (1st/2nd order, fold factor), Doppler-rate matched filter, RFT/GRFT; data_rfst in and out (D10)`
   and `datachip.py  # DataChip, crop, fine range/velocity/acceleration estimators with CRB (D9)`.
   Amend `signal.py`'s comment to add "polynomial radial kinematics; range gate".
2. **B.1 rows:** `core/migration.py` ← Perry–DiPietro–Fante keystone; Zhu–Li–Zhu interpolation-free
   keystone; Huang et al. 2016 keystone + matched filtering; Kirkland second-order keystone;
   Xu et al. RFT/GRFT (oracle). `core/datachip.py` ← Aboutanios–Mulgrew; Quinn, Jacobsen–Kootsookos,
   Candan (baselines); O'Shea CPF; Peleg–Porat HAF (baseline) and polynomial-phase CRB; Rife–Boorstyn.
3. **Decisions D9 and D10**, as §4.1 and §4.2.
4. **D1 widened**: `PropagationPaths` gains `doppler_rate_hzps` and `doppler_jerk_hzps2`,
   defaulting to zero (§4.3).
5. **`scripts/check_conventions.py` `UNIT_SUFFIXES`** gains `_mps2`, `_mps3`, `_hzps`, `_hzps2`
   (shared file; additive edit in step 0).

### 4.9 Workstream boundary

This scenario claims:

```
scenarios/scenario_004_keystone_{k1,k2,k3}.toml
spec/scenario-004-keystone.md, docs/scenarios/scenario-004-keystone.md
src/radar_forge/core/{migration,datachip}.py                       (new)
src/radar_forge/core/propagation.py   (widened: two path fields, kinematics)
src/radar_forge/core/signal.py        (widened: polynomial range, range gate)
src/radar_forge/core/dsp.py           (additive: delay_bin_centers_m only)
src/radar_forge/pipelines/{trajectories,scenarios}.py   (polynomial trajectory, gate keys)
tests/core/test_{migration,datachip}.py, tests/pipelines/test_scenario_004.py
```

`core/{windows,detection,radar,radar_equation,waveforms,ambiguity}.py` are consumed, never
edited. Scenario 001–003 TOMLs and tests are read, never edited, and must stay green. Scenario
006 owns multi-channel synthesis and consumes `DataChip` as specified here; a change to D9 is a
change to both specs.

---

## 5. Build Sequence

```
  0. scripts/check_conventions.py   + _mps2 _mps3 _hzps _hzps2   (shared, additive)
             |
  1. core/propagation.py (widen)     PropagationPaths +2 defaulted fields; line_of_sight_paths
             |                        +accel/jerk; core/signal.py pulsed_baseband polynomial range + range gate
  2. core/dsp.py (additive)          delay_bin_centers_m
             |
  3. core/migration.py (new)         range_spectrum -> keystone (czt | sinc | dft oracle)
             |                        -> fold factor -> doppler_rate_compensation -> searches
             |                        -> second_order_keystone -> rft / grft (oracles)
  4. core/datachip.py (new)          DataChip, crop_slow_time, crop_range_doppler, round trip
             |                        -> tone interpolators -> HAF, CPF -> local GRFT ML -> CRB
  5. pipelines/trajectories.py       kind = "polynomial" radial history; truth at CPI centre
  6. pipelines/scenarios.py          gate keys in [[burst]]; [target] accel/jerk
  7. scenarios/scenario_004_keystone_k{1,2,3}.toml
  8. tests/pipelines/test_scenario_004.py     A3 (baseline fails) first
```

| Step | Module | Consumes / widens / untouched |
| :--- | :--- | :--- |
| 0 | `scripts/check_conventions.py` | Widened (suffix list) |
| 1 | `core/propagation.py`, `core/signal.py` | **Widened.** `PropagationPaths`, `line_of_sight_paths` (propagation); `pulsed_baseband` (signal); `fmcw_deramp_baseband` untouched (its stop-and-hop warning stays) |
| 2 | `core/dsp.py` | **Widened** by one helper; `matched_filter`, `doppler_fft`, `doppler_bin_centers_mps` consumed |
| 3 | `core/migration.py` | New. Uses `scipy.signal.czt`, already a dependency |
| 4 | `core/datachip.py` | New. Consumes `core/windows.py` (window record only) and `core/detection.py` (`cfar_detect_2d`, `cluster_detections` give the crop centre) |
| 5–6 | `pipelines/` | Widened. `load_scenario` / `iterate_frames` gain a polynomial trajectory and gate keys; CSV trajectories untouched |
| 7–8 | TOMLs, tests | New. Three TOMLs differing only in `[target]` kinematics |

`core/radar.py`, `core/waveforms.py` (`lfm_chirp`), `core/windows.py`, `core/detection.py`,
`core/constants.py` and `core/ambiguity.py` are consumed unchanged.

---

## 6. Acceptance Criteria

Encoded as `tests/core/test_migration.py`, `tests/core/test_datachip.py` (fast, small analytic
cases) and `tests/pipelines/test_scenario_004.py` (full-size, marked `slow`). Losses are measured
on noiseless cubes against the matched value `N·K`, with the peak refined on a 1/16-bin grid
(§3.4). The baseline tolerance is wider than the compensated one because a smeared peak is
sensitive to the exact pulse spectrum (the LFM's Fresnel ripple versus §3.4's flat band), while a
focused peak is not.

| # | Criterion | Bound | Applies to |
| :--- | :--- | :--- | :--- |
| A1 | Uncompensated walk, measured as the per-pulse range-peak track, matches v·T/Δr | 27.872 (K1, K2), 90.174 (K3) bins, ±1 bin | K1–K3 |
| A2 | **Residual walk after the keystone**, per-pulse peak track over pulses 1 … N−2 (the two end pulses ring, §4.2), noiseless | < 0.5 bin (computed: only pulses 0 and 1023 exceed it; pulses 8 … 1015 read 0.000 on a 1/8-bin grid) | K1; K3 with F = 1 |
| A3 | **The baseline fails.** MTD loss matches §3.4, and at ρ = 28.89 dB a CA-CFAR at Pfa = 10⁻⁶ on the MTD map detects the target in ≤ 5 of 100 trials | −26.52 / −25.53 / −31.80 dB, ±1.0 dB | K1, K2, K3 |
| A4 | **Compensation recovers the gain.** Compensated peak within 0.3 dB of `10·log10(N)` ideal, and the same CFAR detects in ≥ 95 of 100 trials | ≥ −0.3 dB; ≥ 95% | K1 (keystone), K2 (keystone + matched filter), K3 (keystone, F = 1) |
| A5 | **Each step fixes one thing (K2).** Keystone alone leaves the spread; dechirp alone leaves the walk | −4.06 ± 0.5 dB; −26.51 ± 1.0 dB | K2 |
| A6 | **K3 fold factor recovered** by the search over F ∈ [−3, 3]; the wrong F = 0 loses as predicted | F̂ = 1 in 200 of 200 trials at ρ = 28.89 dB; F = 0: −32.94 ± 1.0 dB | K3 |
| A7 | Acceleration search peak at the true a | within one search step; the curve is returned | K2 |
| A8 | **Fine range and velocity** (A&M on the chip), RMSE / CRB, 500 trials, `slow` | ≤ 1.15 at ρ = 20 and 28.89 dB (computed for a pure tone: 0.984, 1.011) | K1 |
| A9 | **Fine acceleration**, RMSE / CRB: local GRFT ≤ 1.2; CPF alone ≤ 1.5; HAF reported, not bounded | as stated | K2 |
| A10 | **Chip round trip is exact**: integer indices `==`; metres to `rtol = 1e-12`; Doppler crops across ±v_ua wrap; range crops at the edge clip | exact | unit |
| A11 | A ±16-bin chip keeps ≥ −0.05 dB of a point target's energy at the worst offset | computed −0.034 dB | unit |
| A12 | **Interpolator bias** on a noiseless tone, max over offsets in [−0.5, 0.5]: parabolic > 0.1 bin (it **fails**, computed 0.234); Quinn and Jacobsen ≤ 2·10⁻⁶; Candan ≤ 1·10⁻⁶; A&M ≤ 1·10⁻⁹ | as stated | unit |
| A13 | Parabolic RMSE / CRB at ρ = 28.89 dB **exceeds 3** (bias-limited; computed 12.2) while A&M stays ≤ 1.15 | as stated | unit |
| A14 | Jerk: on K1–K3 the cubic-model estimate of j is within 3 CRB std of 0 in ≥ 99% of trials. On a synthetic cubic-phase tone where j is observable, CPF + ML recovers it within 1.5 CRB std | as stated | K1–K3; unit |
| A15 | `PropagationPaths` defaults reproduce every scenario 001–003 cube bit for bit; their tests stay green | exact | regression |
| A16 | Keystone implementations agree: `czt` equals the O(N²) `dft` oracle; truncated-sinc error reported | `rtol = 1e-10` (computed 3.3·10⁻¹⁴ on a 64-point test) | unit |
| A17 | Range spectrum of `dsp.matched_filter`'s output equals X(f)·P*(f) in the linear-convolution length | `rtol = 1e-12` | unit |
| A18 | Second-order keystone straightens a pure quadratic range history with ≥ 3 bins of curvature | residual < 0.5 bin | unit |
| A19 | Velocity estimates reported in parent coordinates with F applied: K3 recovers 1100 m/s, not 335.22 | within 3 CRB std | K3 |

The tolerances: 1/16-bin peak refinement leaves 0.05 dB, so 0.3 dB is six times the grid
residual. 500 Monte Carlo trials estimate an RMSE to about ±3% (1/√1000), so a bound of 1.15
against a computed 1.01 is about 4σ of sampling error. CPF's 1.5 follows O'Shea's result that two
of its four phase parameters have mean-square errors within 50% of the bound at high SNR [12],
which is about 1.22 in RMSE, plus margin. No bound is below ρ = 20 dB, where every estimator
starts producing outliers (the threshold effect [13]). That region is plotted, not asserted.

---

## References

.. [1] R. P. Perry, R. C. DiPietro and R. L. Fante, "SAR imaging of moving targets," *IEEE Trans.
       Aerospace and Electronic Systems*, 35(1), pp. 188-200, 1999. doi:10.1109/7.745691
       (equation numbers to confirm at implementation).
.. [2] D. Zhu, Y. Li and Z. Zhu, "A keystone transform without interpolation for SAR ground
       moving-target imaging," *IEEE Geoscience and Remote Sensing Letters*, 4(1), pp. 18-22,
       2007. doi:10.1109/LGRS.2006.882147.
.. [3] J. Xu, J. Yu, Y.-N. Peng and X.-G. Xia, "Radon-Fourier transform for radar target
       detection, I: Generalized Doppler filter bank," *IEEE Trans. Aerospace and Electronic
       Systems*, 47(2), pp. 1186-1202, 2011. doi:10.1109/TAES.2011.5751251.
.. [4] J. Xu, J. Yu, Y.-N. Peng and X.-G. Xia, "Radon-Fourier transform for radar target
       detection (II): Blind speed sidelobe suppression," *IEEE Trans. Aerospace and Electronic
       Systems*, 47(4), pp. 2473-2489, 2011. doi:10.1109/TAES.2011.6034645.
.. [5] J. Yu, J. Xu, Y.-N. Peng and X.-G. Xia, "Radon-Fourier transform for radar target
       detection (III): Optimality and fast implementations," *IEEE Trans. Aerospace and
       Electronic Systems*, 48(2), pp. 991-1004, 2012. doi:10.1109/TAES.2012.6178044.
.. [6] J. Xu, X.-G. Xia, S.-B. Peng, J. Yu, Y.-N. Peng and L.-C. Qian, "Radar maneuvering target
       motion estimation based on generalized Radon-Fourier transform," *IEEE Trans. Signal
       Processing*, 60(12), pp. 6190-6201, 2012. doi:10.1109/TSP.2012.2217137.
.. [7] P. Huang, G. Liao, Z. Yang, X.-G. Xia, J.-T. Ma and J. Ma, "Long-time coherent integration
       for weak maneuvering target detection and high-order motion parameter estimation based on
       keystone transform," *IEEE Trans. Signal Processing*, 64(15), pp. 4013-4026, 2016.
       doi:10.1109/TSP.2016.2558161 (equation numbers to confirm at implementation).
.. [8] D. Kirkland, "Imaging moving targets using the second-order keystone transform," *IET
       Radar, Sonar & Navigation*, 5(8), pp. 902-910, 2011. doi:10.1049/iet-rsn.2010.0304.
.. [9] S. Peleg and B. Porat, "Estimation and classification of polynomial-phase signals," *IEEE
       Trans. Information Theory*, 37(2), pp. 422-430, 1991. doi:10.1109/18.75269.
.. [10] S. Peleg and B. Porat, "The Cramer-Rao lower bound for signals with constant amplitude and
        polynomial phase," *IEEE Trans. Signal Processing*, 39(3), pp. 749-752, 1991.
        doi:10.1109/78.80864.
.. [11] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014,
        §4.6.1 (LFM), §4.2 (the matched filter); the Doppler-processing chapter (section number
        to confirm at implementation).
.. [12] P. O'Shea, "A fast algorithm for estimating the parameters of a quadratic FM signal,"
        *IEEE Trans. Signal Processing*, 52(2), pp. 385-393, 2004. doi:10.1109/TSP.2003.821097.
.. [13] D. C. Rife and R. R. Boorstyn, "Single-tone parameter estimation from discrete-time
        observations," *IEEE Trans. Information Theory*, 20(5), pp. 591-598, 1974.
        doi:10.1109/TIT.1974.1055282.
.. [14] B. G. Quinn, "Estimating frequency by interpolation using Fourier coefficients," *IEEE
        Trans. Signal Processing*, 42(5), pp. 1264-1268, 1994. doi:10.1109/78.295186.
.. [15] E. Jacobsen and P. Kootsookos, "Fast, accurate frequency estimators [DSP Tips & Tricks],"
        *IEEE Signal Processing Magazine*, 24(3), pp. 123-125, 2007. doi:10.1109/MSP.2007.361611.
.. [16] Ç. Candan, "A method for fine resolution frequency estimation from three DFT samples,"
        *IEEE Signal Processing Letters*, 18(6), pp. 351-354, 2011. doi:10.1109/LSP.2011.2136378.
.. [17] E. Aboutanios and B. Mulgrew, "Iterative frequency estimation by interpolation on Fourier
        coefficients," *IEEE Trans. Signal Processing*, 53(4), pp. 1237-1242, 2005.
        doi:10.1109/TSP.2005.843719.
