# Scenario 005 — Transmitter leakage and near-range clutter in a continuous-wave radar: least-squares cancellation

## 1. Status & Purpose

**Status: specified, not implemented.**

Scenario 001's radar at the Duke Receiver transmits chirps. Its FMCW variant transmits all the
time, but deramping hides one problem that every continuous-wave radar has: **the receiver listens
while the transmitter is on.** Some of the transmitted power couples straight across from the
transmit antenna to the receive antenna. This is the **leakage**. Strong echoes from buildings and
terrain a few hundred metres away arrive almost at the same moment. Both are many orders of
magnitude stronger than an aircraft at 17 km.

This slice gives 001's radar a **continuous noise code** in place of the chirp. The radar knows
the code it sent, so it can model the leakage and near-range clutter as delayed, scaled copies of
its own waveform and subtract them before range-Doppler processing. That is the two-input canceller
of Widrow et al. [1]_. The received stream is the primary input, and the radar's own transmit
waveform is the reference input. Full-duplex radios face the same problem: antenna isolation and an
analog canceller remove most of the leakage, and a digital canceller removes the rest [3]_. This
slice is that last, digital stage.

Three reasons to make this the fifth slice:

1. **It is the first slice where the strongest signal is not the target.** The leakage and
   clutter in the received stream are 71.1 – 72.8 dB above the aircraft echo. A noise code's
   correlation sidelobes sit only 59.13 dB below its own peak, so without cancellation the aircraft
   lies about 14 dB *under* the leakage's sidelobe floor. Range-Doppler processing alone cannot
   reach it.
2. **It builds the cancellation half of `core/clutter.py`.** `spec/structure.md` plans the
   leakage and clutter cancellers there (Part B, D12). This slice writes them from Haykin [2]_,
   Colone et al. [8]_ and Anttila et al. [4]_.
3. **The method is a projection, and the projection can be checked exactly.** Every claim in §6
   follows from closed-form linear algebra: the cancellation depth from the INR and the tap count,
   the Doppler notch from the fit length, and the distortion floor from the amplifier's polynomial.
   The tests check those numbers, not recorded output.

| Section | What it fixes |
| :--- | :--- |
| §1.1 Upstream inspiration | What each method is, where it comes from, and which are baseline and which current |
| §2 Overview & Geometry | The single site, the disturbance stack, the processing chain |
| §3 Parameter Matrix | Waveform, link budget, the three variants, derived numbers, contracts, schema |
| §4 Core Decisions | D11, D12; the stream; the noise code; what is out of scope; `structure.md` amendments |
| §5 Build Sequence | What is new, what is widened, what is untouched |
| §6 Acceptance Criteria | Analytic bounds, led by the degeneracy chain and the cancellation bound |

### 1.1 Upstream inspiration

| Upstream | What it is | What radar-forge takes | Licence | Baseline or current |
| :--- | :--- | :--- | :--- | :--- |
| Widrow et al. 1975 [1]_ | Adaptive noise cancelling: a primary input holding signal plus interference, a reference input correlated with the interference, and a filter that subtracts the reference's best fit | The problem statement: the received stream is the primary input, the transmit waveform the reference input | paper, written from equations | background |
| Sabharwal et al. 2014 [3]_ | In-band full-duplex: self-interference, and its removal in stages (antenna isolation, analog, digital) | The leakage budget of §3.3: what reaches the ADC is what the analog stages left | paper, written from equations | background |
| Melzer, Onic, Starzer & Huemer 2015 [7]_ | Transmitter leakage and short-range clutter in an FMCW radar transceiver | The problem in a radar: a fixed object close to the antennas adds to the leakage and degrades sensitivity | paper, written from equations | background |
| Kulpa 2013 [6]_ | Noise waveform radar: the waveform, its ambiguity function, and range-Doppler processing in batches | The noise code (§4.4), its flat sidelobe floor, and `batched_correlation` | book, written from equations | **current** (batched correlation) |
| Wiener filter with sample-matrix inversion (Wiener-SMI) [2]_ | Wiener-Hopf solution with estimated auto- and cross-correlation over the whole CPI | `wiener_smi`, the fast correlation form; the baseline the least-squares family is checked against | book, written from equations | **baseline** |
| Least squares and its projection form [2]_ | The least-squares fit on the data matrix, `y = (I − X(XᴴX)⁻¹Xᴴ)s` | `ls_cancel` (P1); `ls_cancel_batched`, one fit per batch (P2) | book, written from equations | full CPI: **baseline**; batched: **baseline** for a time-varying channel |
| Colone, Palmarini, Martelli & Tilli 2016 [8]_ | The sliding scheme: weights estimated on a window wider than the block they filter, moving block by block, with the analysis of the batched fit's notch and its parasitic Doppler lines | `ls_cancel_sliding` (P2): the notch of the wide window without the batched fit's parasitic lines | book, written from equations | **current** |
| Anttila, Korpi, Syrjälä & Valkama 2013 [4]_ | Digital cancellation of power-amplifier-induced nonlinear self-interference with a parallel-Hammerstein basis | `parallel_hammerstein_basis` (P3): odd-order basis functions `|x|^{2(p−1)}x` beside the linear taps | paper, written from equations | **current** |
| Korpi, Anttila, Syrjälä & Valkama 2014 [5]_ | Widely linear digital cancellation, which adds the conjugate basis for IQ imbalance | Named as the successor for IQ-imbalanced front ends. Out of scope (§4.5) | paper, written from equations | **current**, out of scope |
| padasip (`structure.md` §A.6) | A Python library of adaptive filters behind one interface | The interface shape of the cancellers, as structure only. No code, no dependency | MIT | code reference |

The current method in this slice is **sliding-window least squares** for a changing channel, with a
**parallel-Hammerstein basis** for a nonlinear transmitter, over **batched correlation**.
Wiener-SMI and full-CPI least squares are the baselines it is derived from (§3.5 walks the
derivation). P2 shows the full-CPI fit failing where batched and sliding fits succeed, and P3 shows
a linear basis failing where the parallel-Hammerstein basis succeeds. No cancellation is the
teaching contrast, and P1 shows it failing.

---

## 2. Overview & Geometry

The site, target and window are scenario 001's. One radar at the Duke Receiver has a transmit and
a receive antenna side by side, and it transmits a noise code continuously. The receiver samples the
same band the whole time.

```
                                   [ Light aircraft, 1500 m, 16.6 - 18.4 km ]
                                                ^
                                                |  echo, -31 dB per sample
                                                |
   [ Duke Receiver ]  tx antenna ---------------+
        |    ^                                  |
        |    |  leakage: tx -> rx across        |
        |    |  the mast (lags 0, 1, 2)         |
        v    |                                  |
     rx antenna <------ near-range clutter: buildings and terrain
                        at 299.79, 749.48, 1498.96, 2997.92 m (lags 16, 40, 80, 160)
```

What the received stream holds, per sample and then on the range-Doppler map (frame 0, the on-bin
acceptance frame, levels relative to thermal noise):

```
   per sample (dB re noise)                  on the map (dB re the map's noise floor)
   +40.00 .. +20.00  leakage, lags 0-2       +99.1  leakage peak, lag 0, zero Doppler
   +30.00 .. +12.00  four clutter echoes     +40.9  leakage + clutter sidelobe floor  <-- everywhere
     0.00  thermal noise                     +27.2  aircraft peak (clean)             <-- 13.7 dB under it
   -31.10  aircraft echo                       0.0  thermal floor
```

The correlation of a noise code has a flat sidelobe floor 10·log10(N) = 59.13 dB below its peak
**at every range and Doppler**. A disturbance that is never cancelled therefore does not stay in
its own cell. Its floor covers the whole map. This is why cancellation runs before correlation, on
the stream, and not as a notch on the map afterwards.

The processing chain. New blocks are marked `+`. Everything else is scenario 001 code, unchanged:

```
  data/flight_coordinates.csv      scenarios/scenario_005_*.toml
            |                         [[burst]] waveform="noise"
            v                         [leakage] [[clutter]] [transmitter_distortion] [cancellation]
  pipelines/trajectories.py  to_radar_frame  (001, unchanged)
            |
            v
  core/signal.py  line_of_sight_paths (001)   + leakage_paths, static_clutter_paths
            |
            v
  + core/signal.py  stream_baseband  ->  data_t (n_stream_samples,), tx_t (n_stream_samples,)
            |        time origin = the transmit instant; lag m is range m*c/(2*fs)
            v
  + core/clutter.py  ls_cancel | ls_cancel_batched | ls_cancel_sliding | wiener_smi
            |        stream projected off span{tx delayed by 0..K-1} (+ |tx|^2 tx for P3)
            v
  + core/dsp.py  batched_correlation  ->  data_rfst (n_pulses, n_range_lags)
            |
            v
  core/dsp.py  doppler_fft (unchanged)  ->  data_rv
            |
            v
  core/detection.py  cfar_detect_2d (unchanged)
```

---

## 3. Parameter Matrix

### 3.1 Site, target and waveform

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Site (Duke Receiver) | 36.00250 °N, 78.94100 °W, 60 m | **Scenario 001's site, unchanged**. Transmit and receive antennas share it |
| Target | `data/flight_coordinates.csv`, 1500 m, 10 dBsm, Swerling 0 | As 001 |
| Window | `start_time_s = 0.0`, `duration_s = 120.0`, 1 Hz frames | As 001: 16.63 – 18.36 km, −39.3 … +55.3 m/s |
| Waveform | Continuous noise code, 9.8 GHz, 8 MHz wide | See below |

**Why a noise code.** The aim is a wideband waveform whose ambiguity function is a thumbtack [6]_,
so that the leakage's correlation sidelobes form a flat floor. That floor is what buries the
target. A long pseudo-random code, as in phase-modulated CW (PMCW) radar, approaches it. Here it is
complex white Gaussian noise over the sampled band (§4.4), so every bound in §3.5 is exact. 8 MHz
gives an 18.7370 m range bin at 001's carrier.

> **Fictions, stated plainly.** The transmitter radiates 1 W, set so the clean SINR lands near
> 27 dB, where cancellation losses are visible. The leakage at the ADC is *specified*, not derived
> (§3.3). It stands for what antenna isolation and an analog canceller leave. Raising the power
> raises the leakage and the echo together and leaves their 71 dB ratio, which is the problem,
> unchanged.

### 3.2 Waveform, receiver and processing

| Parameter | Value | Note |
| :--- | :--- | :--- |
| `waveform` | `"noise"` | New `WaveformKind`; §4.4 |
| `f0_hz` | 9.8e9 → λ = 30.5911 mm | 001's carrier |
| `bandwidth_hz` = `sample_rate_hz` | 8.0e6 | White over the sampled band |
| `transmit_power_w`, `gain_tx_dbi` | 1.0, 30.0 | |
| `gain_rx_dbi`, `noise_figure_db` | 30.0, 3.0 | 001's receiver. Noise power kT₀·f_s·F = 6.3910e-14 W (−131.94 dBW): the code fills the sampled band, so the noise bandwidth is `f_s` |
| `batch_duration_s` | 50.0e-6 → 400 samples per batch | The correlation batch. It plays the part of the pulse: `prf_hz` = 20 000.0 (duty 1) |
| `n_pulses` | 2048 batches → CPI 0.1024 s, **N = 819 200** samples | Processing gain 10·log10 N = **59.13 dB** |
| `n_range_lags` | 1024 → range 0 – 19 168.0 m | Covers the window's 16 626 – 18 358 m (lags 887.3 – 979.8) |
| Range bin `c/(2f_s)` | **18.7370 m** | |
| Doppler bin `1/CPI` | 9.765625 Hz → radial velocity bin **0.14937 m/s** | |
| Unambiguous radial velocity | ±152.956 m/s (±10 kHz) | The track's −39.3 … +55.3 m/s never folds |
| `n_taps` K | 192 | ≥ 161, the largest clutter lag + 1 |
| Batched cancellation | 6.4 ms = 51 200 samples, 16 per CPI | P2 |
| Sliding cancellation | filter block 50 µs (= the correlation batch), estimation window 6.45 ms (block ± 25 600 samples) | P2 |
| Parallel-Hammerstein basis | orders 1 and 3 on all K taps → 384 columns | P3 |
| Doppler window | rectangular | §3.4's straddle losses assume it |
| CFAR | `cfar_detect_2d`, CA, `pfa = 1e-6` | Ideal threshold 11.40 dB. Pd = 0.5 needs 11.24 dB and Pd = 0.9 needs 13.18 dB (non-fluctuating) |

### 3.3 Disturbance budget and the three variants

The target's level uses the repo's monostatic `received_power_w` with 001's antennas. The leakage
and clutter are specified as levels at the ADC. Values are per sample, relative to noise:

| Component | Range | Lag | Doppler | Level in the received stream |
| :--- | ---: | ---: | ---: | ---: |
| Leakage | 0 / 18.74 / 37.47 m | 0 / 1 / 2 | 0 | INR **40.00 / 30.00 / 20.00 dB** |
| Clutter 1–4 | 299.79 / 749.48 / 1498.96 / 2997.92 m | 16 / 40 / 80 / 160 | 0 | CNR 30.00 / 24.00 / 18.00 / 12.00 dB |
| Total disturbance | | | | INR **40.94 dB** |
| Aircraft, default window | 16 626 – 18 358 m | 887.3 – 979.8 | −2570.4 … +3615.6 Hz | SNR −31.87 … −30.15 dB; target/disturbance −72.8 … −71.1 dB |

*Where 40 dB comes from.* A 1 W transmitter behind 50 dB of antenna isolation would leak about
82 dB above the receiver's noise. An analog canceller in the RF front end takes most of that [3]_
[7]_. The 40 dB that reaches the ADC is this slice's problem. The leakage lags 1 and 2 stand for the
front end's short impulse response.

The clutter lags are whole samples by construction, so the claim that K taps span the clutter is
exact. Clutter off the sample grid has band-limited sinc tails into the neighbouring lags, which is
why K carries margin above 161.

| | **P1** `static-leakage` | **P2** `varying-clutter` | **P3** `distorted-transmitter` |
| :--- | :--- | :--- | :--- |
| Channel | Static: leakage and four clutter echoes at zero Doppler | P1 **plus** a 2.0 Hz copy of clutter 1 (lag 16), CNR 30.00 dB | P1, but the power amplifier is nonlinear: every path carries `f(x)`, not `x` |
| Methods | none · Wiener-SMI · **least squares, K = 192** · least squares with K = 96 | none · least squares (full CPI) · **batched, 6.4 ms** · **sliding, 50 µs / 6.45 ms** | least squares, linear basis · **least squares, parallel-Hammerstein basis** |
| What it teaches | The leakage sidelobe floor buries the target. A projection onto the known code with enough taps removes it to the noise. Too few taps leave the far clutter | A channel that changes inside the CPI escapes a notch one Doppler bin wide. Shorter fits widen the notch, and a sliding window removes the batched fit's parasitic lines | The canceller's basis must model what the hardware did. A linear basis stops at the amplifier's distortion floor. Odd-order basis functions remove it |

The 2 Hz component stands for a slowly varying near scatterer, such as a tree or a mast moving in
the wind. It is 0.205 of a Doppler bin from zero. That is close enough to look static on the map
and far enough to escape a one-bin notch.

**The amplifier (P3).** `f(x) = (x + a₃|x|²x) / √(1 + 4a₃ + 6a₃²)` with `a₃ = −0.07`: third-order
gain compression of 7% at mean power, normalised to unit power. The normalisation uses the
complex-Gaussian moments `E|x|⁴ = 2`, `E|x|⁶ = 6`. Leakage, clutter and target all carry it,
because they all left the same amplifier. That is the Hammerstein structure (nonlinearity, then
linear channel) that the parallel-Hammerstein basis models [4]_. A linear canceller can remove only
the part of `f(x)` that correlates with `x`, the Bussgang gain `β = (1 + 2a₃) / √(1 + 4a₃ + 6a₃²)`.
The rest is the fraction `2a₃² / (1 + 4a₃ + 6a₃²)` = **−18.835 dB** of every path. The target loses
`|β|²` = −0.057 dB to the mismatch between `f(x)` and the replica `x`.

### 3.4 Default window and the acceptance frames

The window is 001's: `start_time_s = 0.0`, `duration_s = 120.0`. The five acceptance frames are
0 – 4, where the track opens at a steady 28.08 m/s (f_d = −1835.9 Hz) and lies on a Doppler bin.
In the table, *straddle* is range / Doppler / batch loss, the batch loss being
`sinc²(f_d·T_batch)` [6]_. *Clean* is N·SNR after straddle, with no disturbance. All values are dB.
P3's columns include the −0.057 dB replica mismatch.

| t (s) | Range (m) | Lag | f_d (Hz) | Straddle | Clean | None | P1 LS | P1 K = 96 | P2 full | P2 batched | P3 linear | P3 PH |
| ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 17 560.7 | 937.22 | −1835.94 | −0.70 / −0.00 / −0.12 | 27.22 | −13.73 | 27.22 | 14.95 | 6.02 | 25.33 | 5.02 | 27.16 |
| 1 | 17 588.8 | 938.72 | −1835.95 | −1.17 / −0.00 / −0.12 | 26.71 | −14.23 | 26.71 | 14.45 | 5.52 | 24.82 | 4.52 | 26.65 |
| 2 | 17 616.9 | 940.22 | −1835.97 | −0.68 / −0.00 / −0.12 | 27.18 | −13.77 | 27.18 | 14.91 | 5.98 | 25.29 | 4.98 | 27.12 |
| 3 | 17 644.9 | 941.72 | −1836.00 | −1.19 / −0.00 / −0.12 | 26.63 | −14.31 | 26.63 | 14.37 | 5.44 | 24.75 | 4.44 | 26.58 |
| 4 | 17 673.0 | 943.21 | −1836.02 | −0.66 / −0.00 / −0.12 | 27.14 | −13.81 | 27.14 | 14.87 | 5.94 | 25.25 | 4.94 | 27.08 |

Predicted Pd at `pfa = 1e-6`, per frame:

- **None:** ≤ 1.7e-6.
- **P2, full-CPI least squares:** ≤ 1.1e-2.
- **P3, linear basis:** ≤ 4.7e-3.
- **P1 with K = 96:** ≥ 0.987. It is detected, 12.27 dB down.
- **P1 least squares, P2 batched and sliding, P3 parallel-Hammerstein:** 1.000 to three figures.

The SINR columns are predictions from §3.5's closed forms. The prototype in §6 measured them on one
realisation.

### 3.5 Mathematical contracts

Scenario 001 §3.5 holds. Notation: `x[n]` is the digital transmit waveform, `f(x)` what the
amplifier radiates, `s[n]` the received stream, `N` the CPI in samples, `K` taps, `σ²` noise power.

| Contract | Statement |
| :--- | :--- |
| Time origin | The stream is timed from the **transmit instant**. Lag `m` is the round-trip delay `m/f_s`, so range is **`m·c/(2f_s)`**. The noise code never repeats, so range is unambiguous to the end of the lag window, `n_range_lags·c/(2f_s)` |
| Signal model | `s[n] = Σ_p a_p·f(x)(n − τ_p)·e^{j2π f_p n/f_s} + w[n]` over every path: leakage, clutter, target. `f(x) = x` except in P3. Fractional delays are applied as a frequency-domain phase ramp, which is exact for a periodic band-limited signal |
| Delay basis | `X = [x[n], x[n−1], …, x[n−K+1]]`, shape `(n_samples, K)`, from the **digital** waveform `tx_t`, read from the stream's lead guard (§4.3) and never zero-padded. It holds no noise |
| Least squares [2]_ | `w = (XᴴX)⁻¹Xᴴs`, `y = s − Xw = (I − X(XᴴX)⁻¹Xᴴ)s`. A projection: `Xᴴy = 0` exactly |
| Wiener-SMI [2]_ | The same `w` from the biased correlation estimates `R̂[i−j] = (1/N) Σ x[n−i]x*[n−j]` (Toeplitz) and `r̂[i] = (1/N) Σ s[n] x*[n−i]`, computed by FFT. It equals least squares up to edge terms of order K/N |
| Batched [2]_ | Split the CPI into batches of `N_B` samples and fit each with its own `w_b`. One batch is the full-CPI fit |
| Sliding [8]_ | Filter blocks of `N_f` samples, each with `w` estimated on the block extended by `N_e` samples each side, clipped to the CPI. `N_e = 0` is the batched fit with `N_B = N_f` |
| Parallel-Hammerstein basis [4]_ | `X_PH = [X, X₃]`, with `X₃` the columns `|x[n−k]|²x[n−k]`, k = 0 … K−1. The projection is unchanged; only the basis grows. The Gram matrix of one tap's pair `[x, |x|²x]` is `[[1, 2], [2, 6]]`, condition number 22.46 |
| Spill | Correlation reads `n_range_lags − 1` samples past the CPI. They are filtered with the last batch's (or block's) weights. Leaving them raw would leak the leakage's sidelobes across the whole map |
| Doppler notch [8]_ | A lag-0 probe at Doppler `f` keeps the fraction `ρ(f) = (1 − |D(f)|²)²` of its correlation peak power, with `D(f) = sin(π f N_c/f_s) / (N_c sin(π f/f_s))`, the Dirichlet kernel, and `N_c` the fit length (`N` for the full CPI, `N_B` batched). Derivation: the fit removes the probe's projection onto `x`, a fraction `D(f)` of its amplitude, and the correlation then projects the residual onto the probe again. Nulls of `D` fall at `f = k·f_s/N_c`, so the notch spans exactly **one Doppler bin** for the full CPI, and `±1/T_B` batched. The residual *power* of a Doppler-`f` disturbance is `1 − |D(f)|²` |
| CA | `CA = Σ|s|² / Σ|y|²` over the CPI, in dB at the boundary |
| CA bound | With every disturbance inside the basis, `E[CA] = (1 + INR) / [(1 − K/N)(1 + INR_unc)]`. `INR_unc` is the disturbance the basis cannot represent: echoes beyond lag K−1; `P_c(1 − |D(f_c)|²)` for a Doppler-`f_c` component; and `INR·2a₃²/(1 + 4a₃ + 6a₃²)` for amplifier distortion under a linear basis. The basis is noise-free, so the ceiling is `CA ≤ (1 + INR)/(1 − K/N)`: cancel to the noise |
| Floor rise | Away from the zero-Doppler ridge, the map's floor rises by the denominator of the CA bound, and SINR falls by the same amount. Measured as the mean power of the 216 cells 2 – 4 Doppler bins and 3 – 20 lags from the target, relative to the clean run's same cells. The target's own Doppler rows are excluded: a flat-spectrum code has a sinc range response, and its sidelobes along that row would bias the floor |
| Batched correlation [6]_ | `y_b[m] = Σ_{n=0}^{L_b−1} y[bL_b + n + m]·x*[bL_b + n]`, `m = 0 … n_range_lags − 1`, then `doppler_fft` over `b`. Valid while `|f_d|·T_batch < ½`. Batch loss `sinc²(f_d T_batch)` |
| Array names | `data_t (n_stream_samples,)` is the received stream and `tx_t (n_stream_samples,)` the digital transmit waveform over the same samples. `data_ftst (n_pulses, n_samples_per_batch)` is the stream's CPI view. The correlation output is `data_rfst (n_pulses, n_range_lags)` and the map is `data_rv` (style.md §3.1) |
| TOML layout | 001's order, plus `[leakage]` (`lags`, `inr_db`), `[[clutter]]` (each with `range_m`, `relative_power_db` re noise, `doppler_hz`), `[transmitter_distortion]` (`a3`) and `[cancellation]` after `[receiver]`. `[[burst]]` with `waveform = "noise"` takes `batch_duration_s` in place of `chirp_duration_s` |

**From the baseline to the current method, in five steps.** Each step is one function and one
test.

1. The Wiener filter [2]_ estimating `s` from delayed copies of `x` is a least-squares fit. With
   estimated statistics it is Wiener-SMI.
2. Written on the data matrix instead of on the statistics, the same fit is a projection. It is
   exact, and it has the `Xᴴy = 0` property a test can check.
3. A projection holds one set of weights for the whole CPI, so it notches one Doppler bin. Fit per
   batch instead and the notch widens to `±1/T_B`.
4. Batch-wise weights jump at every batch edge and modulate the residual at harmonics of `1/T_B`
   [8]_. Estimate on a wide window and filter on a short block, and the notch keeps the wide
   window's width while the harmonics move to `1/T_f`. With `T_f` equal to the correlation batch,
   they land at ±20 kHz, outside the map.
5. A nonlinear amplifier puts power where no delayed copy of `x` reaches. Add the odd-order
   columns `|x|²x` to the basis [4]_. The projection, the test and the function are the same; only
   the basis grows.

**Intermediate quantities exposed** (starter.md §1.1 item 6): the weights per batch or block,
shape `(n_cancellation_batches, n_basis)`; the Gram matrix condition number; CA; the predicted CA
bound; the floor rise; the notch response `ρ(f)`; and the residual stream itself.

### 3.6 Output schema

As 001 §3.6. Additions:

| File | Addition |
| :--- | :--- |
| `metadata.json` | `bursts[].waveform = "noise"`, `batch_duration_s`, `n_range_lags`, and the `[leakage]`, `[[clutter]]`, `[transmitter_distortion]` and `[cancellation]` blocks verbatim |
| `iq.h5` | `/iq/burst0/stream`, `(n_frames, n_stream_samples)`, and `/iq/burst0/tx`, the same shape, with `n_lead_guard_samples` and `n_tail_guard_samples` attributes. The cube view is not stored (§4.3) |
| `products.h5` | `/clutter/burst0/weights`, `(n_frames, n_cancellation_batches, n_basis)`; `/rt/burst0/data` holds the correlation batches |
| `metrics.csv` | `cancellation_depth_db`, `cancellation_depth_bound_db`, `floor_rise_db` per frame |

---

## 4. Core Decisions

### 4.1 D11 — the canceller's basis is the radar's own transmit waveform, beside a single-channel stream

A continuous-wave radar knows what it sent. The basis for cancelling its leakage and near-range
clutter is therefore the **digital transmit waveform** `tx_t`, carried beside the received stream
`data_t` over the same samples, with the same guards. It is not a receive channel. 005 has one
receive channel, so it needs no channel axis before scenario 006's D13 creates one. Every function
in `core/clutter.py` takes the transmit waveform and the received stream as separate arrays.

*Why the digital waveform and not a coupled-off sample of the transmitter:* the digital waveform
holds no receiver noise, so the cancellation ceiling is the noise itself (§3.5, CA bound). What it
does not hold is what the hardware did to it. P3 shows the cost, and the parallel-Hammerstein basis
is the remedy.

### 4.2 D12 — the least-squares cancellers live in `core/clutter.py`

As `structure.md` plans (Part B, D12). The public functions:

| Function | Returns | Note |
| :--- | :--- | :--- |
| `delay_basis_gram(tx_t, data_t, n_taps, start, stop, *, odd_orders=(1,))` | `(gram (P, P), cross (P,))` | The normal equations, built in chunks so memory is O(chunk·P). `P = n_taps·len(odd_orders)` |
| `ls_weights(gram, cross, *, diagonal_loading=0.0)` | `(P,)` | Cholesky solve. The loading defaults to zero, and the condition number is returned beside it |
| `ls_cancel(tx_t, data_t, n_taps, *, odd_orders=(1,), doppler_offsets_hz=(0.0,))` | `CancellationResult` | Full CPI |
| `ls_cancel_batched(…, n_samples_per_cancellation_batch)` | `CancellationResult` | Batched |
| `ls_cancel_sliding(…, n_samples_per_filter_block, n_samples_estimation_extension)` | `CancellationResult` | Sliding window |
| `wiener_smi(tx_t, data_t, n_taps)` | `CancellationResult` | FFT correlation estimates |
| `parallel_hammerstein_basis(tx_t, n_taps, odd_orders, start, stop)` | `(n_samples, P)` | The basis itself, for notebooks and tests |
| `cancellation_depth_db(data_t, residual_t)` | `float` | CA |
| `expected_cancellation_depth_db(inr_linear, n_basis, n_samples, inr_uncancelled_linear=0.0)` | `float` | The §3.5 bound, so tests and notebooks share one formula |
| `doppler_notch_response_linear(doppler_hz, n_samples_cancellation, sample_rate_hz)` | `NDArray` | `ρ(f)` |
| `distortion_fraction_linear(a3)` | `float` | `2a₃²/(1 + 4a₃ + 6a₃²)`, the part of `f(x)` no linear basis reaches |

`CancellationResult` is a frozen dataclass: `residual_t`, `weights`
`(n_cancellation_batches, P)`, `gram_condition_number` `(n_cancellation_batches,)`. **Weights are
returned, never applied silently**, which is the same rule scenario 007 states as D16 for spatial
weights. A Python loop runs over cancellation batches and filter blocks. Each is an independent
solve, so the loop is the algorithm and not an unvectorised one. The code comment says so.

Per `structure.md` D2, this adds the cancellation half of `core/clutter.py` alongside no clutter
model yet. Scenario 008 adds the Ward patch model, which is one clutter model. The promotion
trigger, more than two clutter models plus the cancellation algorithms, is not reached by
005 + 008.

### 4.3 The stream, not the cube, is what cancellation sees

A pulse radar's cube is a set of independent repetitions. A continuous transmission is one signal:
a tap reaches 191 samples into the previous batch, and a lag reads 1023 samples into the next ones.
So synthesis produces a **stream** `data_t`, shape `(n_stream_samples,)`, with
`n_lead_guard_samples = n_range_lags` before the CPI and `n_tail_guard_samples = n_range_lags − 1`
after it, and `tx_t` over the same samples. `n_taps ≤ n_range_lags` is enforced, so every tap and
every lag reads recorded samples. The style.md cube is the view
`data_t[lead : lead + N].reshape(n_pulses, n_samples_per_batch)`, and nothing that needs the guards
takes it. A zero-padded history would leave each clutter echo uncancelled for its first `d` samples.
That breaks the exact projection that A2 and A6 test, and it grows with clutter delay and strength.
The guard costs nothing.

### 4.4 Why white Gaussian noise

A practical code has structure, such as a repetition period or a finite alphabet, and structure
puts deterministic ridges into the ambiguity function [6]_. Modelling them would make §6's sidelobe
floor a function of the code instead of `10·log10 N`, and the slice would be testing the waveform
instead of the cancellation. White Gaussian noise has no structure, so its floor is flat. It also
gives the amplifier model of P3 closed-form moments. `core/waveforms.py` gains
`noise_waveform(n_samples, rng)`, unit power.

### 4.5 What is out of scope, and why it is not a gap in the bar

- **Recursive adaptive filters: LMS, NLMS and RLS** [1]_ [2]_. They converge towards the same
  Wiener solution, with convergence behaviour as an extra variable. They are baselines in this
  family and add nothing a test here would catch.
- **Widely linear cancellation** [5]_. An IQ-imbalanced front end adds a conjugate term the
  parallel-Hammerstein basis lacks. The basis grows by the conjugate columns, and nothing else
  changes. It belongs with a receiver-chain scenario that models IQ imbalance.
- **Analog cancellation and phase noise** [3]_ [7]_. The leakage is specified at the ADC (§3.3).
  Transmitter phase noise decorrelates the leakage from the digital waveform. It is the next
  limit after amplifier distortion, and it needs the LO model of scenario 004.
- **Target self-cancellation.** The aircraft is at lags 887 – 980, outside the K = 192 tap span,
  so the projection cannot reach it. A slow target inside the first 3.6 km is lost to the notch,
  and that is the notch working (A5).

### 4.6 Amendments to `spec/structure.md`

> **Applied.** These amendments are folded into `spec/structure.md`; it is the authoritative
> tree and decision list. This section records what changed and why.

1. **Part B tree.** Annotate `core/clutter.py` as
   `# land/sea clutter; least-squares leakage and clutter cancellation (005); Ward clutter patch, ICM taper (008)`.
   `core/signal.py` adds `stream_baseband, leakage_paths, static_clutter_paths`.
   `core/waveforms.py` adds `noise (a long pseudo-random code)`. `core/dsp.py` adds
   `batched correlation for continuous waveforms`.
2. **B.1 mapping.** The `core/clutter.py` row reads: *"Least-squares, batched and Wiener-SMI
   cancellers (Haykin 2014); sliding-window estimation (Colone et al. 2016); parallel-Hammerstein self-interference cancellation
   (Anttila et al. 2013), written from the papers; padasip (MIT, adaptive-filter structure); …"*.
   The `core/dsp.py` batched-correlation row reads *Kulpa 2013*.
3. **Part A §A.6** is padasip (MIT), the interface reference for the cancellers.
4. **Decisions D11 and D12**, as §4.1 and §4.2:
   - *D11 — The canceller's basis is the radar's own transmitted waveform.* It is carried beside a
     single-channel stream, not as a channel column.
   - *D12 — Leakage and clutter cancellation lives in `core/clutter.py`.* The block least-squares,
     batched, sliding-window, Wiener-SMI and parallel-Hammerstein cancellers are written from the
     texts. Each returns its residual and its weights, and none applies weights silently. The D2
     trigger is unchanged.
5. **`RadarLike` widening.**
   - `WaveformKind` gains `"noise"`. For it, `chirp_duration_s` holds the correlation batch and
     `prf_hz` its reciprocal, so `unambiguous_velocity_mps`, `n_samples_per_pri` and
     `doppler_bin_centers_mps` apply unchanged. That is D7 working again.
   - `Radar.unambiguous_range_m` gains a `"noise"` branch, `n_range_lags·c/(2f_s)`.
   - The correlation receiver gets its own range axis, `m·c/(2f_s)`, beside the deramp
     `range_bin_centers_m`.

### 4.7 Workstream boundary

This scenario claims:

```
scenarios/scenario_005_{static_leakage,varying_clutter,distorted_transmitter}.toml
spec/scenario-005-leakage-cancellation.md
src/radar_forge/core/clutter.py            (cancellation half)
src/radar_forge/core/{signal,waveforms,dsp,radar}.py   (additive widenings only)
src/radar_forge/pipelines/scenarios.py     ([leakage], [[clutter]], [transmitter_distortion], [cancellation]; the "noise" dispatch)
tests/core/test_clutter.py, tests/pipelines/test_scenario_005.py
```

`core/detection.py`, `core/windows.py`, `core/geodesy.py` and `pipelines/trajectories.py` are
consumed and never edited. `test_scenario_001.py`, `test_scenario_002.py` and
`test_scenario_003.py` must stay green at every commit. The Ward patch model in `core/clutter.py`
belongs to scenario 008, and `array/` to 006 onward.

---

## 5. Build Sequence

```
  1. core/waveforms.py   + noise_waveform                  (new function)
  2. core/radar.py       WaveformKind += "noise"; unambiguous_range_m branch   (widen)
  3. core/signal.py      + leakage_paths, static_clutter_paths, stream_baseband (amplifier model optional)
  4. core/dsp.py         + batched_correlation, correlation range axis   (doppler_fft unchanged)
  5. core/clutter.py     + delay_basis_gram, ls_weights, wiener_smi, ls_cancel,
                           ls_cancel_batched, ls_cancel_sliding, parallel_hammerstein_basis,
                           depth, notch and distortion helpers
  6. tests/core/test_clutter.py        A1-A5 first: projection, degeneracy, notch
  7. pipelines/scenarios.py            new tables; "noise" dispatch; stream frames
  8. scenarios/scenario_005_*.toml     three files
  9. tests/pipelines/test_scenario_005.py   A6-A14
 10. notebooks/: the dynamic-range ladder, the notch plot, the distortion floor (viz extra)
```

| Step | Module | Reuse |
| :--- | :--- | :--- |
| 1 | `core/waveforms.py` | New. Unit-power complex Gaussian from a passed `rng` |
| 2 | `core/radar.py` | Widened. `Transmitter` validation is unchanged: duty cycle 1, as FMCW |
| 3 | `core/signal.py` | **Consumes** `PropagationPaths` unchanged (D1, D6). Leakage and clutter are paths with `bounce_count = 1` and zero Doppler (P2 adds one with 2 Hz). The target comes from 001's `line_of_sight_paths`. `thermal_noise` is consumed. A loop over paths (fewer than ten) applies each FFT delay. The comment says why |
| 4 | `core/dsp.py` | New function. `doppler_fft` and `doppler_bin_centers_mps` are consumed |
| 5 | `core/clutter.py` | New cancellation half; `scipy.linalg.cho_factor`. No new dependency |
| 7 | `pipelines/scenarios.py` | No `[transmitter_site]`, so `Radar` (monostatic). `iterate_frames` dispatches `"noise"` to `stream_baseband` (today it falls through to the FMCW generator) and yields the stream |
| 9 | tests | `to_radar_frame` gives truth exactly as in 001's A2 |

---

## 6. Acceptance Criteria

The unit tests (`tests/core/test_clutter.py`) are small and fast. The scenario tests
(`tests/pipelines/test_scenario_005.py`) are **`slow`**. The bounds were checked against a numpy
prototype of the §3.5 model, run on one realisation per case. Its measured values are given in
brackets.

| # | Criterion | Bound | Tolerance and why | Applies to |
| :--- | :--- | :--- | :--- | :--- |
| A1 | **Degeneracy chain.** Sliding with `N_e = 0` equals batched with `N_B = N_f`. Batched with one batch equals the full-CPI fit. The full-CPI fit equals the exact-statistics Wiener solution. The parallel-Hammerstein fit with `odd_orders=(1,)` equals the linear fit | residual streams equal | `rtol = 1e-9`: the same normal equations solved twice | unit |
| A2 | **Projection.** `Xᴴy = 0` for every batch's basis, linear and parallel-Hammerstein | `‖Xᴴy‖ / (‖X‖‖s‖) ≤ 1e-10` | float64 round-off on a Cholesky solve, at the logged condition number [linear 1.09; parallel-Hammerstein 24.3, against 22.46 for one tap's pair] | unit |
| A3 | Wiener-SMI against least squares | residual powers agree | `rtol = 1e-3`: edge terms O(K/N) = 2.3e-4, with 4× margin [9.7e-6] | P1 |
| A4 | **Notch shape.** A noiseless lag-0 probe, K = 1. Its response equals `ρ(f)` computed from the realised waveform exactly. Its expectation `ρ(f)` from §3.5 is 0.00 dB at 9.765625 Hz for the full-CPI fit (a null of `D`); for the 6.4 ms batched fit, −37.87 dB at 9.765625 Hz, −4.51 dB at 78.125 Hz and 0.00 dB at 156.25 Hz | exact: `rtol = 1e-9`. Expectation: ±1.0 dB at 9.765625 Hz, ±0.3 dB elsewhere | Batch-coefficient scatter is about 1/√N_B. At 9.765625 Hz, where `1 − |D|²` = 0.0128, that moves the response by up to about ±0.7 dB, an order-of-magnitude estimate the exact form makes moot [−37.88, −4.52, 0.00, 0.00] | unit |
| A5 | **The notch is one bin wide.** The full-CPI fit leaves the neighbouring bins, ±9.765625 Hz, at 0 dB. A probe at f_d = 0 inside the tap span is removed | ±0.1 dB; removal ≤ −100 dB noiseless | `|D|²` at a null is O(1/N) = 1.2e-6 | unit |
| A6 | **Cancellation bound.** P1, least squares, K = 192: CA = 40.946 dB | ±0.05 dB | Power estimated from 8.192e5 samples has a std of 0.005 dB, so this is ≥ 10σ, plus O(K/N) model error [40.943] | P1 |
| A7 | **Tap count.** K = 96 cannot reach the lag-160 echo: CA = 28.680 dB, floor rise 12.27 dB | CA ±0.05 dB; floor ±1.0 dB | As A6; floor as A9 [28.676; 11.84] | P1 |
| A8 | **Target at the true bins.** After cancellation (P1 least squares; P2 batched and sliding; P3 parallel-Hammerstein), the CFAR peak nearest the truth sits at range `m·c/(2f_s)` and radial velocity from `to_radar_frame` | one range bin (18.7370 m) and one velocity bin (0.14937 m/s), in all five frames | 001's A2/A3 contract, through the correlation range axis | P1, P2, P3 |
| A9 | **Floor rise matches the budget.** Floor over the clean run: P1 least squares 0.00 dB; P2 batched 1.89 dB; P3 linear 22.14 dB | ±1.0 dB | The floor is a mean of 216 cells. Across 400 positions on one map its std was 0.33 dB, and the bound is 3σ [P1 0.01; P2 2.06; P3 21.85] | P1, P2, P3 |
| A10 | **SINR after cancellation.** Within 1.0 dB of clean for P1 least squares and P3 parallel-Hammerstein, and within 1.0 dB of §3.4's prediction for P2 batched (1.89 dB loss) | as stated | The peak cell carries interference with amplitude ratio 10^(−SINR/20). At 25 dB that is ±0.5 dB, plus the floor's 0.33 dB [P1 −0.01 dB, P3 −0.01 dB from clean; P2 batched −0.72 dB from prediction] | P1, P2, P3 |
| A11 | **The baselines fail; the current methods succeed.** At `pfa = 1e-6`: **no cancellation** detects the target in **none** of the five frames (Pd ≤ 1.7e-6 each). **Full-CPI least squares on P2** and **the linear basis on P3** detect it in **at most one** (Pd ≤ 1.1e-2 and ≤ 4.7e-3 each; P(≥ 2 of 5) ≤ 1.2e-3). Least squares (P1), batched and sliding (P2) and parallel-Hammerstein (P3) detect it in **all five** (predicted SINR ≥ 24.7 dB, Pd ≈ 1) | counts | Fixed seed. The counts carry margin against the predicted Pd, and the margin is what makes the test not flaky | P1, P2, P3 |
| A12 | **Sliding removes the batched fit's parasitic lines.** At (lag 16, ±156.25 Hz), sliding 50 µs / 6.45 ms is at least 20 dB below batched 6.4 ms. Its floor is no higher than batched | ≥ 20 dB; ≤ | Measured 27.7 and 35.0 dB lower (50.5 → 22.8 and 50.3 → 15.3 dB re the far floor), and floor rise 2.06 → 0.11 dB. 20 dB leaves 7.7 dB margin | P2 |
| A13 | **The amplifier limits a linear basis.** P3, linear basis: CA = 18.810 dB. The parallel-Hammerstein basis: CA = 40.947 dB | ±0.1 dB | The distortion power involves `|x|⁶`, a heavy-tailed statistic, so its estimate from N samples scatters more than A6's: about 0.02 dB. 0.1 dB is 5σ [18.783; 40.942] | P3 |
| A14 | Absolute received power | **not asserted**, per D8 | — | — |

> A6, A7 and A13 assert CA. A11 asserts detection. CA alone can reward a canceller that also
> removes the target, which is why A9 and A10 measure the floor and the peak separately, and why no
> criterion here passes on CA alone.

---

## References

.. [1] B. Widrow, J. R. Glover, J. M. McCool, J. Kaunitz, C. S. Williams, R. H. Hearn et al.,
       "Adaptive noise cancelling: principles and applications," *Proc. IEEE*, 63(12),
       pp. 1692-1716, 1975. doi:10.1109/PROC.1975.10036. The two-input canceller: primary and
       reference inputs; LMS.
.. [2] S. Haykin, *Adaptive Filter Theory*, 5th ed., Pearson, 2014. The Wiener filter; ch. 9, the
       method of least squares and its projection form; sliding-window estimation (section numbers
       to confirm at implementation).
.. [3] A. Sabharwal, P. Schniter, D. Guo, D. W. Bliss, S. Rangarajan and R. Wichman, "In-band
       full-duplex wireless: challenges and opportunities," *IEEE J. Selected Areas in
       Communications*, 32(9), pp. 1637-1652, 2014. doi:10.1109/JSAC.2014.2330193.
       Self-interference and its staged cancellation.
.. [4] L. Anttila, D. Korpi, V. Syrjälä and M. Valkama, "Cancellation of power amplifier induced
       nonlinear self-interference in full-duplex transceivers," *Proc. Asilomar Conf. Signals,
       Systems and Computers*, pp. 1193-1198, 2013. arXiv:1401.3129. The parallel-Hammerstein
       basis (equation numbers to confirm at implementation).
.. [5] D. Korpi, L. Anttila, V. Syrjälä and M. Valkama, "Widely linear digital self-interference
       cancellation in direct-conversion full-duplex transceiver," *IEEE J. Selected Areas in
       Communications*, 32(9), pp. 1674-1687, 2014. arXiv:1402.6083.
.. [6] K. Kulpa, *Signal Processing in Noise Waveform Radar*, Artech House, 2013.
       ISBN 978-1-60807-661-1. The noise waveform, its ambiguity function, and range-Doppler
       processing in batches (chapters to confirm at implementation).
.. [7] A. Melzer, A. Onic, F. Starzer and M. Huemer, "Short-range leakage cancelation in FMCW radar
       transceivers using an artificial on-chip target," *IEEE J. Selected Topics in Signal
       Processing*, 9(8), pp. 1650-1660, 2015. Leakage and short-range clutter in a radar
       transceiver.
.. [8] F. Colone, C. Palmarini, T. Martelli and E. Tilli, "Sliding extensive cancellation
       algorithm for disturbance removal in passive radar," *IEEE Trans. Aerospace and Electronic
       Systems*, 52(3), pp. 1309-1326, 2016. doi:10.1109/TAES.2016.150477. The sliding
       estimation window, the batched fit's notch and its parasitic Doppler structure (equation
       numbers to confirm at implementation).
