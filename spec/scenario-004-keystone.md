# Scenario 004 — High-speed target: keystone, polynomial-phase compensation and the datachip

## 1. Status & Purpose

**Status: specified, not implemented.** The fourth vertical slice, and the first in which the
target moves far enough within one coherent processing interval (CPI) to matter.

Every slice so far has relied on one approximation: at 2 MHz bandwidth an 80 m/s target moves
**0.0109** of a 74.95 m range bin in scenario 001 S2's 10.24 ms CPI (80 × 0.01024 / 74.95), so
range migration is negligible. That stays true only while the range bins are coarse and the target
is slow. This slice changes both. It uses a wideband pulse (500 MHz, 0.3 m slant resolution) and a
fast, accelerating target, so that over one 20.48 ms CPI the echo **walks across 27.9 range bins**
and an acceleration **spreads the Doppler peak over 4.03 bins**. Plain range-Doppler processing
then loses **25.8 dB** of its 30.1 dB coherent gain, and the target falls below any detection
threshold.

All ranges and rates are **bistatic**, as the repo's maps carry them: range is
:math:`R_b = R_T + R_R = c\tau` (twice the slant range here, monostatic), its rate
:math:`\dot R_b` is **positive receding**, and Doppler is :math:`f_D = -\dot R_b/\lambda`, written
once in `propagation.doppler_shift_hz`. Only `plots.display_axes` halves them, for figures and CSVs.

The slice builds four things, in the order the signal meets them:

1. **Migration correction** (`migration.py`). The keystone transform removes linear range walk for
   every target at once, without knowing its velocity. A Doppler-rate filter in the
   range-frequency domain then removes the quadratic and cubic phase that acceleration and jerk
   leave behind. A fold-factor search handles Doppler-ambiguous targets.
2. **Polynomial-phase estimation.** "Polyphase" is read as *polynomial-phase* motion
   compensation: rate, acceleration and jerk are the first three coefficients of the slow-time
   phase. (The polyphase channeliser belongs to scenario 009.)
3. **The datachip** (`datachip.py`). A small crop of the cube around a target, carrying its
   offsets into the parent and the parent's bin values. Scenario 006 consumes it for angle
   estimation on multi-channel chips.
4. **Fine estimation** on the chip: sub-bin range and rate by interpolation on Fourier
   coefficients, acceleration from the cubic phase function, a local maximum-likelihood
   refinement, each against its Cramér–Rao bound (CRB). Too expensive over the cube, cheap on the
   chip: that is why the chip exists.

Single channel, monostatic. It extends scenario 001's pulsed chain (LFM pulse, matched filter) and
does not touch the FMCW path.

### 1.1 Upstream inspiration

| Upstream | What radar-forge takes | Baseline or current |
| :--- | :--- | :--- |
| Moving-target detection: compress, then Doppler FFT per range bin (Richards [11] §5.3) | The uncompensated map whose loss §6 asserts | **Baseline** |
| Perry, DiPietro & Fante 1999 [1] | The keystone: rescale slow time per range frequency, :math:`t = \tau f_0/(f_0+f)` | Current |
| Zhu, Li & Zhu 2007 [2] | The keystone's rescaling as a chirp-z transform per range frequency; truncated sinc kept as the baseline | Current (implementation) |
| Bluestein 1970 [18] | The chirp-z transform as one batched convolution; `scipy.signal.czt` is its test reference | Current (implementation) |
| Xu, Yu, Peng & Xia, RFT I–III [3]–[5] | The Radon–Fourier transform as a **test oracle** for linear walk | Oracle |
| Xu et al. 2012, GRFT [6] | Local GRFT on the chip as the ML refinement | **Current** (on the chip) |
| Huang, Liao, Yang, Xia & Ma 2016 [7] | Keystone, Doppler-rate filter, fold-factor search: the K2 and K3 chain | **Current** |
| Kirkland 2011 [8] | Second-order keystone :math:`t = \tau\sqrt{f_0/(f_0+f)}`, unit-tested, not in a chain | Current (extension) |
| Peleg & Porat 1991, HAF [9] | Acceleration and jerk baseline on the chip | **Baseline** |
| Peleg & Porat 1991, CRB [10] | Bound for acceleration and jerk | Oracle |
| O'Shea 2004, CPF [12] | Acceleration and jerk on the chip; the ML start | Current |
| Rife & Boorstyn 1974 [13] | The tone CRB; the frame for every sub-bin estimator | Oracle |
| Parabolic on \|X\|; Quinn 1994 [14]; Jacobsen & Kootsookos 2007 [15]; Candan 2011 [16] | Three-bin interpolators; parabolic is the one shown to fail | **Baseline** |
| Aboutanios & Mulgrew 2005 [17] | Iterative interpolation on Fourier coefficients | **Current** |

No code is vendored; everything is written from the papers.

---

## 2. Overview & Geometry

A wideband X-band tracking radar sited downrange of a sounding-rocket launch pad observes one
target along its line of sight: a rocket closing on the radar, coasting (K1), under thrust (K2) or
after burnout (K3). Only range and its derivatives matter. Slow time starts at **pulse 0**, the
frame's truth time; the target follows

```
   R_b(t) = R_b0 + Rdot_b t + Rddot_b t^2/2 + J_b t^3/6        bistatic, positive receding
            t = m / PRF, m = 0 ... N-1; truth at t = 0

   [ radar ]  <-----------  2500 m slant (5000 m bistatic)  ---------  [ target ]  ===> closing
```

The TOML gives the target as `position_enu_m`, `velocity_enu_mps` and an optional
`acceleration_enu_mps2` (it then follows p + v t + a t²/2); `Radar.geometry` returns
:math:`\ddot R_b` exactly, the velocity across the line of sight included. Jerk is a field of
`PropagationPaths` (default 0); no scenario sets it.

Walk is what the target does during a CPI; what it costs depends on the bin:

```
  range bin (0.49965 m bistatic)     370  375  380  385  390  395  400
  pulse 0       (t = 0)                                              ##
  pulse 512                                       ##
  pulse 1023    (t = T)              ##
                                     |<---- 27.87 bins: K1 ---->|
  Uncompensated: each range bin holds ~1/28 of the pulses       -> -25.8 dB
  After keystone (reference pulse 0): every pulse in bin 400    -> -0.08 dB
```

The chain; new steps are in capitals:

```
  config/scenario_004_keystone_k{1,2,3}.toml   [[burst]] range gate, [migration]
     |
  scenario.py     target p + v t + a t^2/2 -> truth at the first pulse
  propagation.py  line_of_sight_paths: doppler_rate_hzps from the route's acceleration
  signal.py       pulsed_baseband(..., RANGE GATE)                       -> (n_pulses, n_gate)
  pipeline.py     pulse_compress: matched filter, group delay cut         -> data_rfst
     |---------------------------------------------> Doppler FFT -> MTD map (BASELINE)
  migration.py    RANGE SPECTRUM (FFT along range, zero-padded, signed f)
                  KEYSTONE t = tau f0/(f0+f)  [FOLD-FACTOR SEARCH]
                  DOPPLER-RATE FILTER (a[, j]), search curve exposed      -> data_rfst
  pipeline.py     Doppler FFT -> map -> CFAR, clustering (unchanged)
  datachip.py     CROP -> DataChip; FINE ESTIMATION: A&M (range, rate), CPF, local GRFT, CRB
  stages/migration.py   stage "keystone": migration.csv, keystone.png
```

---

## 3. Parameter Matrix

### 3.1 Waveform: the same in all three variants

| Parameter | Value | Why |
| :--- | :--- | :--- |
| Waveform | LFM pulse, matched filter | Scenario 001's pulsed chain |
| `carrier_frequency_hz` | 9.8 GHz → λ = **30.5911 mm** | Scenario 001's carrier |
| `bandwidth_hz` | 500 MHz → c/2B = **0.29979 m** slant | Fractional bandwidth **0.05102**: the keystone's correction scales with it |
| `chirp_duration_s` | 0.5 µs, time-bandwidth **250** | Intra-pulse motion 0.55 mm at 1100 m/s (§4.6) |
| `sample_rate_hz` | 600 MHz, f_s/B = 1.2 → one bin c/f_s = **0.49965 m** bistatic | 300 samples per pulse |
| `prf_hz` | 50 kHz, PRI 20 µs | Rate span `Radar.unambiguous_bistatic_range_rate_mps` = λ·PRF = **1529.55 m/s** (±764.78): K1, K2 unambiguous, K3 not |
| `n_pulses` | 1024 → CPI **20.48 ms** | Ideal gain 10·log10(1024) = **30.10 dB** |
| Rate bin | λ·PRF/N = **1.49370 m/s** (48.83 Hz) | `dsp.doppler_bin_centers_mps`; rates fall with index |
| Unambiguous range | c·PRI = **5995.85 m** bistatic | The target, at 5000 m, sits inside it |
| Range gate | `range_gate_start_m` = **4800 m bistatic**, `n_gate_samples` = 1024 | Opens on sample round(4800 f_s/c) = 9607, i.e. **4800.18 m**; 511.65 m long; 725 fully compressed bins (362.2 m). Without a gate the cube is 1024 × 12 000 (197 MB) |
| `transmit_power_w` | 100 W | |
| `gain_tx_dbi`, `gain_rx_dbi` | 25.0 dBi each | |
| `noise_figure_db` | 3.0 dB | As scenario 001 |
| IQ cube | `(1024, 1024)` complex128, 16.8 MB | Range spectrum `(1024, 2048)` |

**Why this waveform.** For an unambiguous target (|f_D| < PRF/2) the walk over a CPI is bounded
by N·B/(2 f0) = **26.12 resolution cells** (31.35 bins). Walk needs fractional bandwidth and
pulses; scenario 001 S2 (2 MHz, 256 pulses) walks 0.0109 of a bin at 80 m/s.

### 3.2 Link budget, at 2500 m slant, RCS 0.1 m² (−10 dBsm)

| Quantity | Value |
| :--- | ---: |
| Received power (`received_power_w`) | 1.2073e-14 W, −139.182 dBW |
| Noise in the band, kT₀BF (`Radar.noise_power_matched_w`) | 3.9944e-12 W; SNR in band −25.20 dB |
| Noise per sample, kT₀F·f_s | −113.194 dBW; SNR per sample −25.99 dB |
| After pulse compression, E/N₀ per pulse | **−1.217 dB** |
| After 1024 pulses, ideally integrated, ρ | **28.886 dB** |

### 3.3 The three variants

| | **K1** coasting | **K2** under thrust | **K3** after burnout |
| :--- | :--- | :--- | :--- |
| Bistatic rate at pulse 0 | −680 m/s (340 radial, closing) | −680 m/s | **−2200 m/s** (1100 radial) |
| Bistatic acceleration | 0 | **−294.1995 m/s²** (15 g closing) | 0 |
| Doppler f_D / PRF | 0.4446 | 0.4446 → 0.4486 over the CPI | **1.4383** |
| Fold factor F | 0 | 0 | **1**; folded rate −670.45 m/s |
| `[migration]` method | `keystone` | `keystone_doppler_rate`, `acceleration_search_mps2 = [-400, -200, 10]` | `keystone`, `fold_factor_max = 3` |

The fold factor counts whole PRFs of Doppler: :math:`f_D = f_{D,\mathrm{folded}} + F\,\mathrm{PRF}`,
so the unfolded rate is the folded one less :math:`F\lambda\,\mathrm{PRF}`.

### 3.4 Derived walk, spread and loss, per variant

A loss is the map's peak, refined off the bin grid (Doppler by Aboutanios–Mulgrew iterations,
range on a 1/16-bin grid), over N² times pulse 0's compressed peak: 0 dB is ideal integration.
Expected values are re-derived from the range-frequency model
:math:`|P(f)|^2 e^{-j2\pi(f_0+f)R_b(t)/c}` with the chirp's own spectrum ("LFM"), and with a flat
band ("flat") for comparison; the time-domain chain (synthesis, matched filter, keystone) must match
the LFM column within the tolerances of §6.

| Quantity | K1 | K2 | K3 |
| :--- | ---: | ---: | ---: |
| Walk, rate × T / bin | **27.872 bins** | 27.872; least-squares slope over the CPI (the mid-CPI rate, 683.01 m/s) **27.995** | **90.174** |
| Range curvature a T²/8 | 0 | 0.031 bins | 0 |
| Doppler spread |a| T/λ | 0 | 196.96 Hz = **4.034 bins** | 0 |
| Quadratic phase, sagitta π|a|T²/(4λ) | 0 | 3.168 rad | 0 |
| Doppler across the band, ν(1 ∓ B/2f0) | 0.4332 – 0.4559 | 0.4332 – 0.4559 | 1.4016 – 1.4750 |
| **MTD (no compensation)**, LFM / flat | **−25.78** / −26.56 dB | **−25.60** / −25.61 dB | **−34.67** / −36.78 dB |
| Keystone, F = 0, LFM / flat | 0.00 dB | **−4.26** / −4.26 dB | **−31.21** / −32.86 dB |
| Dechirp (true a) only, no keystone | — | **−25.90** / −26.64 dB | — |
| Compensated (keystone [+ filter], right F) | ≈ 0 | ≈ 0 | ≈ 0 |

Three things to read from the table:

- **K2's acceleration shows as Doppler spread, not range curvature**: 0.031 bins of curvature,
  3.17 rad of quadratic phase. The second-order keystone corrects curvature and is the wrong tool;
  the Doppler-rate filter corrects phase. The keystone alone leaves 4.26 dB; dechirping alone,
  without the keystone, recovers nothing.
- **K3 keystoned at the wrong fold factor keystones the alias.** With F = 0 the rescaling treats
  the Doppler as 0.438 PRF and removes that share of the walk, 27.5 of 90.2 bins; 62.7 remain.
  That is 3.5 dB better than no keystone and 31 dB short of F = 1.
- **The smeared peak depends on the pulse's spectrum.** An uncompensated map integrates a slice
  of the band at a time, so the LFM's Fresnel ripple moves its loss by up to 2.1 dB (K3) from the
  flat-band value; a focused peak integrates the whole band and does not care.
- **K3 is near the edge of one fold factor.** F is constant across the band only while
  ν(1 + B/2f0) < F + ½: up to **2237.26 m/s** bistatic for F = 1; K3 has 1.4750 < 1.5. A faster
  target spans two fold factors across the band, a case [7] treats separately: the code refuses
  it.

### 3.5 Mathematical contracts

| Contract | Statement |
| :--- | :--- |
| Range history | §2; truth and the keystone's reference at pulse 0 |
| `data_rfst` | `(n_pulses, n_range[, n_rx])` complex128: the gated cube compressed row by row, the reference's group delay `n_ref − 1` cut. Range axis `dsp.delay_bin_centers_m` = c(n₀ + n)/f_s, n₀ the gate's first sample |
| Range spectrum | `FFT(data_rfst, n = n_range_fft, axis = 1)`, unshifted; signed `range_frequency_hz = fftfreq(n_range_fft, 1/f_s)`, the radio frequency f0 + f. Default `n_range_fft` the next power of two ≥ 2·n_range, so walk cannot wrap |
| Signal model | `S(f, t) = A·|P(f)|²·exp(−j·2π·(f0 + f)·R_b(t)/c)`; the f·Ṙ_b·t term is the walk, the f0·R̈_b·t² term the spread |
| Keystone | `S_K(f, τ) = S(f, t = f0/(f0 + f)·τ)` by band-limited (DFT) interpolation of the N pulses, a chirp-z transform per range frequency [2], reference pulse 0. Then the rate term is f0·Ṙ_b·τ, independent of f [1]. Interpolation past the last pulse (low frequencies are stretched there) reads the periodic extension: the closing pulses ring |
| Fold factor | Slow time sees the alias ν − F; multiply by `exp(+j·2π·F·m·f0/(f0+f))` after the keystone. F maximises the integrated peak over [−F_max, F_max] [7] |
| Doppler-rate filter | After the keystone multiply by `H(f, τ) = exp(+j·(2π/c)·[f0²/(f0+f)·a·τ²/2 + f0³/(f0+f)²·j·τ³/6])`, a, j bistatic, by substituting the keystone into the model [7]. Searched on a grid; the curve is returned |
| Second-order keystone | `t = sqrt(f0/(f0 + f))·τ` [8]; removes the f·R̈_b·t² curvature term |
| Phase polynomial | The range-focused slow-time phase is c1 m + c2 m² + c3 m³ with c1 = −2πṘ_b T/λ (folded), c2 = −πR̈_b T²/λ, c3 = −πJ_b T³/(3λ), T the PRI |
| CRB | Fisher information of the model with complex white noise, parameters (|A|, φ, R0, Ṙ[, R̈[, J]]), flat band, t from pulse 0: `F = 2ρ·mean(h hᵀ)`. Single-tone case `var(ω̂) ≥ 6/(ρ (N² − 1))` [13] |
| Shapes | Every `migration.py` function accepts a trailing `n_rx` axis and acts on axes 0 and 1 only |

CRB at ρ = 28.886 dB and 20 dB, bistatic, t from pulse 0 (`datachip.cramer_rao_bounds`):

| Model | Range | Rate | Acceleration | Jerk |
| :--- | ---: | ---: | ---: | ---: |
| Range, rate; 28.886 dB | 8.406 mm (0.0168 bins) | 0.02093 m/s (0.0140 bins) | — | — |
| Range, rate; 20 dB | 23.38 mm | 0.05823 m/s | — | — |
| + acceleration; 28.886 dB | 8.410 mm | 0.0837 m/s at pulse 0 | 7.917 m/s² | — |
| + jerk; 28.886 dB | 8.417 mm | 0.209 m/s | 47.46 m/s² | **4574 m/s³** |

With t from pulse 0 the rate at pulse 0 couples to the acceleration (centred, its bound would stay
0.0209 m/s); range barely moves. These replace the spec's slant, CPI-centred figures (4.1994 mm,
10.46 mm/s, 3.957 m/s², 2286 m/s³), which are the same bounds halved and centred.

### 3.6 Output

The stepper writes PNG and CSV only (no HDF5 here). The `keystone` stage writes
`migration.csv`, one row for frame 0: walk, residual walk, the losses, the searched fold factor
and acceleration, and the fine estimates with their CRBs and errors in CRB units; and
`keystone.png`, range against pulse for the compressed, keystoned and compensated frame 0. The
search curves are printed. `truth.csv` is scenario 001's.

---

## 4. Core Decisions

### 4.1 The `DataChip`

A frozen dataclass in `datachip.py`:

```python
@dataclass(frozen=True)
class ChipProcessing:          # the parent's processing, as the chip needs it to refine
    carrier_frequency_hz: float
    bandwidth_hz: float
    sample_rate_hz: float
    prf_hz: float
    noise_power_w: float       # per cell of the parent's data_rfst
    keystone_applied: bool = False
    fold_factor: int = 0
    compensated_acceleration_mps2: float = 0.0
    compensated_jerk_mps3: float = 0.0

@dataclass(frozen=True)
class DataChip:
    data: NDArray[np.complex128]           # (n_axis0, n_range[, n_rx]), parent axis order
    domain: Literal["slow_time", "doppler"]
    axis0_offset: int                      # 0 for slow_time; first (wrapped) Doppler bin
    range_offset: int
    parent_shape: tuple[int, int]
    range_bin_centers_m: NDArray[np.float64]   # the parent's values
    rate_bin_centers_mps: NDArray[np.float64]  # doppler: the chip's; slow_time: the parent's
    processing: ChipProcessing
```

Rules, each a test (§6 A10–A11):

1. **Parent axis order.** Axis 0 slow time or shifted Doppler, axis 1 range; a trailing channel
   axis is carried whole.
2. **Range does not wrap; Doppler does.** A range crop leaving the parent is clipped; a Doppler
   crop is taken modulo the parent's Doppler bins and carries the parent's rates verbatim, the
   jump at the fold included.
3. **Exact round trip** in integers; a fractional range maps through the parent's uniform spacing.
4. **Unwindowed.** The pipeline applies no window, so there is nothing to record.
5. **Results leave in parent coordinates**: range, rate unfolded by F, acceleration with what the
   compensation removed added back, each with its CRB; chip-local indices are intermediates.

`crop_slow_time(data_rfst, …)` keeps every pulse; `crop_range_doppler(data_rv, …)` is for 006.
A ±16-bin range chip keeps all but **0.040 dB** of a compressed echo at the worst sub-bin offset
(0.144 dB for ±4): the default half-width is 16.

### 4.2 Migration correction lives in `migration.py` and maps `data_rfst` to `data_rfst`

Every public function takes `data_rfst` and returns a frozen result holding the new `data_rfst`
and its intermediates (range-frequency axis, scale factors, search curves); nothing is modified in
place. The range spectrum is entered and left inside the module. K2 reads
`doppler_rate_compensation(keystone(x).data_rfst)`. `pipeline.range_doppler_product(...,
migration=...)` applies `[migration]` between compression and the Doppler FFT, so detection sees
the compensated map.

Searches, as implemented: the fold factor is scored on every ⌊M/n_range⌋-th range frequency (the
map's range circle is then the gate, so nothing aliases); the acceleration on the five range bins
about the keystoned map's brightest cell, with H at f = 0 (the rest of H straightens a curvature
under a tenth of a bin and moves no Doppler); the best hypothesis's whole filter is applied.

### 4.3 `PropagationPaths` carries two defaulted fields

`doppler_rate_hzps` (exists) and `doppler_jerk_hzps2` (new), each defaulting to zero, so every
existing path set synthesises bit for bit (§6 A15). `line_of_sight_paths` fills the rate from
`TargetGeometry.bistatic_range_acceleration_mps2` through `propagation.doppler_rate_hzps`
(= −R̈_b/λ, which calls `doppler_shift_hz`, so the sign lives once). `bistatic_range_at_m` gains the
cubic term, added last so zero jerk changes nothing. `Radar.geometry` gains
`acceleration_enu_mps2`; without it R̈_b is zero (not the centripetal term), so trajectory scenarios
are unchanged.

### 4.4 The range gate

`pulsed_baseband(..., range_gate_start_m=0.0, n_gate_samples=None)` synthesises only the gate:
samples from `round(start·f_s/c)` (start bistatic), `None` the rest of the PRI, so a gated cube is
those columns of the whole one. `dsp.delay_bin_centers_m(n_bins, sample_rate_hz, *,
range_gate_start_m=0.0)` labels them; `pipeline.map_axes` uses it for pulsed bursts (identical
numbers without a gate). A gated map's range axis is a window on the unambiguous range circle.

### 4.5 Jerk is in the model and is not observable at this CPI

A radial jerk of 300 m/s³ (about 30 g/s, burnout or staging; 600 m/s³ bistatic) puts 0.022 rad
of cubic phase at the CPI's edges about its centre; a bistatic jerk would need 42 700 m/s³ to
reach π/2, and its bound is 4574 m/s³. So no scenario sets jerk. The
chip estimator fits the cubic model on request (`polynomial_order = 3`), reports jerk with its
bound, and §6 asserts consistency with zero; recovery is tested on a model chip where jerk is
observable.

### 4.6 What is not modelled, with the size of the omission

| Effect | Size here | Why it is left out |
| :--- | :--- | :--- |
| LFM range–Doppler coupling | 6.66 mm (K1), 21.6 mm (K3) bistatic, against an 8.4 mm range CRB | `pulsed_baseband` is stop-and-hop: the simulation cannot produce the shift and the estimator must not correct it. On recorded data it would be a 2.6σ bias in K3 |
| Intra-pulse motion | 0.34 mm (K1), 1.1 mm (K3) bistatic | Far below a bin |
| Two fold factors across the band | none (K3 1.4750 < 1.5) | Detected and refused |
| Range ambiguity, eclipsing | none: 5000 m < 5995.85 m, gate clear of the pulse | Scenario 001's lesson |
| Range-gate tracking | the gate is fixed, so a frame window is three frames (75 ms) | Out of scope |

### 4.7 Why sub-bin estimation is not parabolic interpolation

Parabolic interpolation on |X| is biased by up to **0.234 bins** for a rectangular window, without
noise; at ρ = 28.886 dB that bias is about ten CRBs. Quinn and Jacobsen remove it to 1.18·10⁻⁶ bins,
Candan to 3.9·10⁻⁷, and Aboutanios–Mulgrew, two iterations, to rounding (5·10⁻¹³). All five are
implemented because the comparison is the lesson; A&M is the default.

---

## 5. Build Sequence

| Step | Module | Change |
| :--- | :--- | :--- |
| 1 | `radar.py`, `propagation.py` | `geometry(acceleration_enu_mps2=)`, `TargetGeometry.bistatic_range_acceleration_mps2`; `doppler_rate_hzps`, `doppler_jerk_hzps2`, the cubic term |
| 2 | `signal.py`, `dsp.py` | The range gate; `delay_bin_centers_m` |
| 3 | `migration.py` | range spectrum, keystone (chirp-z, sinc), fold-factor search, Doppler-rate filter, second-order keystone, integrated peak |
| 4 | `datachip.py` | chip, crops, tone interpolators, HAF, CPF, local GRFT, CRB |
| 5 | `scenario.py`, `pipeline.py` | `[[target]] acceleration_enu_mps2`, `[[burst]]` gate keys, `[migration]`; `pulse_compress`, `compensate_migration`, `frame_products` |
| 6 | `stages/migration.py`, `plots.py` | Stage `keystone`; `render_range_slow_time` |
| 7 | `config/scenario_004_keystone_k{1,2,3}.toml` | With their `[verify]` tables |

---

## 6. Acceptance Criteria

Encoded in `tests/test_migration.py`, `tests/test_datachip.py` and the `[verify]` tables of the
three TOMLs (run by `tests/test_scenarios.py`; their three frames are the whole window). Losses
are measured on frame 0's noiseless cube (the generator's `rng=None` path). Monte Carlo trials add
compressed noise (white noise matched-filtered along range) to the chain's own noiseless chip or
map: the keystone and the Doppler-rate filter resample slow time near-unitarily, so this is the
chain's noise, and the chain runs once.

| # | Criterion | Bound | Where |
| :--- | :--- | :--- | :--- |
| A1 | Uncompensated walk, the per-pulse peak track's least-squares slope times N | 27.872 (K1), 27.995 (K2), 90.174 (K3) bins, ±1 | `[verify]` `walk_bins`, `walk_tolerance_bins` |
| A2 | Residual walk after the configured correction, peak-to-peak over pulses 1 … N−2, noiseless | < 0.5 bin | `residual_walk_max_bins`; unit tests |
| A3 | **The baseline fails.** MTD loss as §3.4 (LFM), and CA-CFAR at Pfa 10⁻⁶ on the MTD map detects in ≤ 5 of 100 trials at 28.886 dB | −25.78 / −25.60 / −34.67 dB, ±1.0 dB | `uncompensated_loss_db`, `loss_tolerance_db`; slow test |
| A4 | **Compensation recovers the gain**, and the same CFAR detects in ≥ 95 of 100 trials | ≥ −0.3 dB; ≥ 95 % | `compensated_loss_min_db`; slow test |
| A5 | **Each step fixes one thing (K2)**: keystone alone; dechirp alone | −4.26 ± 0.5 dB; −25.90 ± 1.0 dB | `keystone_only_loss_db`, `keystone_only_loss_tolerance_db`, `dechirp_only_loss_db` |
| A6 | **K3 fold factor found** over F ∈ [−3, 3]; the wrong F = 0 loses as predicted | F̂ = 1 (and in 200 of 200 trials at 28.886 dB); F = 0: −31.21 ± 1.0 dB | `fold_factor`, `keystone_only_loss_db`; slow test |
| A7 | Acceleration search peak at the true acceleration | within one 10 m/s² step | `acceleration_tolerance_mps2` |
| A8 | **Fine range and rate** (A&M on the K1 chip), RMSE / CRB, 500 trials | ≤ 1.15 at 20 and 28.886 dB | slow test |
| A9 | **Fine acceleration** (K2), RMSE / CRB: local GRFT ≤ 1.2; CPF ≤ 1.5; HAF reported | as stated | slow test |
| A10 | Chip round trip exact: integers `==`, metres the parent's, Doppler crops wrap, range crops clip | exact | unit |
| A11 | A ±16-bin chip keeps ≥ −0.05 dB of a compressed echo at the worst offset | computed −0.040 dB | unit |
| A12 | Interpolator bias on a noiseless 1024-point tone, offsets in [−0.5, 0.5]: parabolic > 0.1 bin; Quinn, Jacobsen ≤ 2·10⁻⁶; Candan ≤ 1·10⁻⁶; A&M ≤ 1·10⁻⁹ | computed 0.234, 1.18e-6, 3.9e-7, 5e-13 | unit |
| A13 | Parabolic RMSE / CRB at 28.886 dB **exceeds 3**, A&M ≤ 1.15 | as stated | unit |
| A14 | Jerk: on K1–K3 the cubic fit's j within 3 CRB of 0 in ≥ 99 % of 100 trials; on a model chip with observable jerk, recovered within 1.5 CRB (RMSE, 200 trials) | as stated | slow tests |
| A15 | Defaults reproduce every scenario 001–003 cube bit for bit: their fingerprints stay | exact | `tests/data/golden/fingerprints.csv` |
| A16 | The chirp-z keystone equals the O(N²) DFT-interpolation oracle and `scipy.signal.czt` per frequency; the sinc baseline also straightens walk | rtol 1e-10; < 0.5 bin | unit |
| A17 | Range spectrum of `dsp.matched_filter`'s output is X(f)·P*(f) in the linear-convolution length | rtol 1e-12 | unit |
| A18 | Second-order keystone straightens a quadratic history with ≥ 3 bins of curvature | residual < 0.5 bin | unit |
| A19 | Rate reported in parent coordinates with F applied: K3 recovers −2200 m/s bistatic, not −670.45 | within 3 CRB | `rate_error_max_crb` (all variants); unit |

Tolerances: the losses compare a time-domain chain with a frequency-domain model; ±1.0 dB covers
the smeared peak's sensitivity, ±0.5 dB the keystone-only peak's. 500 Monte Carlo trials know an
RMSE to about 3 %, so 1.15 against a computed 1.01 is about 4σ. CPF's 1.5 follows O'Shea's result
that its estimates' mean-square errors come within 50 % of the bound at high SNR [12] (1.22 in
RMSE) with margin. No bound is asserted below 20 dB, where every estimator starts producing
outliers (the threshold effect [13]).

---

## References

.. [1] R. P. Perry, R. C. DiPietro and R. L. Fante, "SAR imaging of moving targets," *IEEE Trans.
       Aerospace and Electronic Systems*, 35(1), pp. 188-200, 1999. doi:10.1109/7.745691 — the
       keystone formatting and its rescaling of slow time per range frequency.
.. [2] D. Zhu, Y. Li and Z. Zhu, "A keystone transform without interpolation for SAR ground
       moving-target imaging," *IEEE Geoscience and Remote Sensing Letters*, 4(1), pp. 18-22,
       2007. doi:10.1109/LGRS.2006.882147 — the keystone as a scaled DFT (chirp-z).
.. [3] J. Xu, J. Yu, Y.-N. Peng and X.-G. Xia, "Radon-Fourier transform for radar target
       detection, I: Generalized Doppler filter bank," *IEEE Trans. Aerospace and Electronic
       Systems*, 47(2), pp. 1186-1202, 2011. doi:10.1109/TAES.2011.5751251.
.. [4] J. Xu, J. Yu, Y.-N. Peng and X.-G. Xia, "Radon-Fourier transform for radar target
       detection (II): Blind speed sidelobe suppression," *IEEE Trans. AES*, 47(4),
       pp. 2473-2489, 2011. doi:10.1109/TAES.2011.6034645.
.. [5] J. Yu, J. Xu, Y.-N. Peng and X.-G. Xia, "Radon-Fourier transform for radar target
       detection (III): Optimality and fast implementations," *IEEE Trans. AES*, 48(2),
       pp. 991-1004, 2012. doi:10.1109/TAES.2012.6178044.
.. [6] J. Xu, X.-G. Xia, S.-B. Peng, J. Yu, Y.-N. Peng and L.-C. Qian, "Radar maneuvering target
       motion estimation based on generalized Radon-Fourier transform," *IEEE Trans. Signal
       Processing*, 60(12), pp. 6190-6201, 2012. doi:10.1109/TSP.2012.2217137.
.. [7] P. Huang, G. Liao, Z. Yang, X.-G. Xia, J.-T. Ma and J. Ma, "Long-time coherent integration
       for weak maneuvering target detection and high-order motion parameter estimation based on
       keystone transform," *IEEE Trans. Signal Processing*, 64(15), pp. 4013-4026, 2016.
       doi:10.1109/TSP.2016.2558161 — keystone, the matched filter in range frequency and slow
       time, and the fold factor.
.. [8] D. Kirkland, "Imaging moving targets using the second-order keystone transform," *IET
       Radar, Sonar & Navigation*, 5(8), pp. 902-910, 2011. doi:10.1049/iet-rsn.2010.0304.
.. [9] S. Peleg and B. Porat, "Estimation and classification of polynomial-phase signals," *IEEE
       Trans. Information Theory*, 37(2), pp. 422-430, 1991. doi:10.1109/18.75269.
.. [10] S. Peleg and B. Porat, "The Cramer-Rao lower bound for signals with constant amplitude and
        polynomial phase," *IEEE Trans. Signal Processing*, 39(3), pp. 749-752, 1991.
        doi:10.1109/78.80864.
.. [11] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014,
        §4.6 (FM pulse compression), §5.3 (pulse Doppler processing).
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
.. [18] L. I. Bluestein, "A linear filtering approach to the computation of discrete Fourier
        transform," *IEEE Trans. Audio and Electroacoustics*, 18(4), pp. 451-455, 1970.

Equation numbers are not given for [1]–[10], [12]–[17]: the papers were not available to check
them against (IEEE paywall), and a guessed number is worse than none. Each citation names the
result it supports.

## Decisions inherited

- **D1, widened.** A propagation backend returns `PropagationPaths`, never baseband; `signal.py`
  synthesises samples from any path set. The path set gains Doppler rate (filled from the route's
  acceleration) and jerk (defaulted), through `line_of_sight_paths` and `bistatic_range_at_m`.
- **D5, sign.** Range rate is the time derivative of bistatic range, positive receding; Doppler is
  `−rate/wavelength`, written once in `propagation.doppler_shift_hz` (`spec/data-001-formats.md`
  §5).
