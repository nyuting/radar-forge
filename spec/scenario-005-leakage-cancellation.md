# Scenario 005 — Transmitter leakage and near clutter in a continuous-wave radar: least-squares cancellation

## 1. Status & Purpose

**Status: specified, not implemented.** Corrected to the repository's conventions: every range
and rate is bistatic (range sum `R_T + R_R = cτ`, its rate positive receding, Doppler
`f_D = −rate/λ` written once in `propagation.doppler_shift_hz`); a monostatic figure or CSV halves
them in `plots.display_axes` only.

Scenario 001's radar at the Duke Receiver transmits chirps. Its FMCW variant transmits all the
time, but deramping hides one problem that every continuous-wave radar has: **the receiver listens
while the transmitter is on.** Some of the transmitted power couples straight across from the
transmit antenna to the receive antenna. This is the **leakage**. Strong echoes from buildings and
terrain a few hundred metres away arrive almost at the same moment. Both are many orders of
magnitude stronger than an aircraft at 17 km.

This slice gives 001's radar a **continuous noise code** in place of the chirp. The radar knows
the code it sent, so it can model the leakage and near clutter as delayed, scaled copies of its
own waveform and subtract them before range-Doppler processing. That is the two-input canceller of
Widrow et al. [1]_ (§III, eq. (1)). The received stream is the primary input, and the radar's own
transmit code is the reference input. Full-duplex radios face the same problem: antenna isolation
and an analog canceller remove most of the leakage, and a digital canceller removes the rest [3]_
(§IV-B to §IV-D). This slice is that last, digital stage.

Three reasons to make this the fifth slice:

1. **It is the first slice where the strongest signal is not the target.** The leakage and
   clutter in the received stream are 71.1 – 72.8 dB above the aircraft echo. A noise code's
   correlation sidelobes sit only 59.13 dB below its own peak, so without cancellation the aircraft
   lies about 14 dB *under* the leakage's sidelobe floor. Range-Doppler processing alone cannot
   reach it.
2. **It builds the cancellation half of `clutter.py`**, from Haykin [2]_, Colone et al. [8]_ and
   Anttila et al. [4]_. Scenario 008 adds the Ward clutter model to the same module; scenario 011
   puts Poisson clutter *plots* in `tracking/scene.py`, not here.
3. **The method is a projection, and the projection can be checked exactly.** Every claim in §6
   follows from closed-form linear algebra: the cancellation depth from the INR and the tap count,
   the Doppler notch from the fit length, and the distortion floor from the amplifier's polynomial.
   The tests check those numbers, not recorded output.

### 1.1 Upstream inspiration

| Upstream | What it is | What radar-forge takes | Baseline or current |
| :--- | :--- | :--- | :--- |
| Widrow et al. 1975 [1]_ | Adaptive noise cancelling: primary input, reference input, a filter subtracting the reference's best fit (§III) | The problem statement | background |
| Sabharwal et al. 2014 [3]_ | In-band full duplex: self-interference removed in stages (§IV-B propagation, §IV-C analog, §IV-D digital) | The leakage budget of §3.3 | background |
| Melzer et al. 2015 [7]_ | Transmitter leakage and short-range clutter in an FMCW transceiver | The problem in a radar | background |
| Kulpa 2013 [6]_ | Noise radar (ch. 3): the code, its correlation receiver, processing in batches | The noise code (§4.4), its flat sidelobe floor, `dsp.batched_correlation` | **current** (batched correlation) |
| Wiener-SMI [2]_ | Wiener-Hopf (§2.4) on the biased sample statistics, the autocorrelation method of §9.2 | `clutter.wiener_smi` | **baseline** |
| Least squares and its projection [2]_ | Normal equations (§9.5) on the data matrix (§9.7) | `clutter.ls_cancel` (P1), `ls_cancel_batched` (P2) | full CPI and batched: **baseline** |
| Colone et al. 2016 [8]_ | ECA-B (§II, eqs. (3)-(4)), its notch and parasitic Doppler lines (§III, eqs. (16)-(17)), ECA-S (§IV, eqs. (18)-(19)) | `clutter.ls_cancel_sliding` (P2) | **current** |
| Anttila et al. 2013 [4]_ | Parallel-Hammerstein model of a nonlinear amplifier (eq. (1)), its LS canceller (§III-B) | `clutter.parallel_hammerstein_basis` and the `odd_orders` basis (P3) | **current** |
| Korpi et al. 2014 [5]_ | Widely linear cancellation for IQ imbalance | Named successor; out of scope (§4.5) | current, out of scope |

The current method in this slice is **sliding-window least squares** for a changing channel, with a
**parallel-Hammerstein basis** for a nonlinear transmitter, over **batched correlation**.
Wiener-SMI and full-CPI least squares are the baselines it is derived from (§3.5). P2 shows the
full-CPI fit failing where batched and sliding fits succeed, and P3 shows a linear basis failing
where the parallel-Hammerstein basis succeeds. No cancellation is the teaching contrast.

---

## 2. Overview & Geometry

The site, target and window are scenario 001's. One radar at the Duke Receiver has a transmit and
a receive antenna side by side (no `[site.rx]`: monostatic), and it transmits a noise code
continuously. Routes below are bistatic, `R_T + R_R`, twice the slant range.

```
                                   [ Light aircraft, 1500 m, 33.3 - 36.7 km route ]
                                                ^
                                                |  echo, -31 dB a sample
   [ Duke Receiver ]  tx antenna ---------------+
        |    ^  leakage: tx -> rx across        |
        |    |  the mast (lags 0, 1, 2)         |
        v    |                                  |
     rx antenna <------ near clutter: buildings and terrain, routes of
                        599.58, 1498.96, 2997.92, 5995.85 m (lags 16, 40, 80, 160)
```

Levels on frame 0, relative to thermal noise:

```
   per sample (dB re noise)                  on the map (dB re the map's floor)
   +40.00 .. +20.00  leakage, lags 0-2       +99.1  leakage peak, lag 0, zero Doppler
   +30.00 .. +12.00  four clutter echoes     +40.9  leakage + clutter sidelobe floor  <-- everywhere
     0.00  thermal noise                     +26.6  aircraft peak (clean)             <-- 14.3 dB under it
   -31.10  aircraft echo                       0.0  thermal floor
```

The correlation of a noise code has a flat sidelobe floor `10·log10 N` = 59.13 dB below its peak
**at every lag and Doppler**. A disturbance never cancelled covers the whole map with its floor, so
cancellation runs before correlation, on the stream, not as a notch on the map afterwards.

```
  data/flight_coordinates.csv      config/scenario_005_*.toml
            |                         [[burst]] waveform = "noise", n_range_lags
            v                         [leakage] [[clutter]] [transmitter_distortion] [cancellation]
  scenario.py  truth_geometry (unchanged)
            v
  propagation.py  line_of_sight_paths  + leakage_paths, static_clutter_paths, PropagationPaths.concatenate
            v
  signal.py  + noise_waveform, cubic_amplifier, stream_baseband -> (surveillance, reference)
            v
  clutter.py  + least_squares_cancel (full | batched | sliding), wiener_smi     (pipeline.cancel)
            v
  dsp.py  + batched_correlation -> (n_pulses, n_range_lags); doppler_fft (unchanged)
            v
  pipeline.py  frame_products -> RangeDopplerProduct; frame_plots: cfar_detect along range (unchanged)
            v
  stages/clutter.py  + the "leakage" stage: leakage.csv, cancellation.png
```

---

## 3. Parameter Matrix

### 3.1 Site, target and waveform

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Site (Duke Receiver) | 36.00250 °N, 78.94100 °W, 60 m | Scenario 001's site; both antennas |
| Target | `data/flight_coordinates.csv`, 1500 m, 10 dBsm, Swerling 0 | As 001 |
| Window | `start_time_s = 0.0`, `duration_s = 120.0`, 1 Hz frames | Routes 33 278 – 36 717 m (slant 16 639 – 18 359 m); bistatic rate −110.6 … +78.6 m/s |
| Waveform | Continuous white Gaussian noise code, 9.8 GHz, 8 MHz wide | §4.4 |

**Why a noise code.** The aim is a wideband waveform whose ambiguity function is a thumbtack [6]_,
so that the leakage's correlation sidelobes form a flat floor. That floor is what buries the
target. Here the code is complex white Gaussian noise over the sampled band (§4.4), so every bound
in §3.5 is exact. 8 MHz gives a lag of `c/f_s` = 37.474 m of route at 001's carrier.

> **Fictions, stated plainly.** The transmitter radiates 1 W, set so the clean SINR lands near
> 27 dB, where cancellation losses are visible. The leakage at the ADC is *specified*, not derived
> (§3.3): it stands for what antenna isolation and an analog canceller leave. Raising the power
> raises the leakage and the echo together and leaves their 71 dB ratio, which is the problem.

### 3.2 Waveform, receiver and processing

| Parameter | Value | Note |
| :--- | :--- | :--- |
| `waveform` | `"noise"` | New `WaveformKind`; `Radar` requires `bandwidth_hz == sample_rate_hz` |
| `carrier_frequency_hz` | 9.8e9 → λ = 30.5911 mm | 001's carrier |
| `bandwidth_hz` = `sample_rate_hz` | 8.0e6 | White over the sampled band |
| `transmit_power_w`, `gain_tx_dbi` | 1.0, 30.0 | |
| `gain_rx_dbi`, `noise_figure_db` | 30.0, 3.0 | Noise a sample, `Receiver.noise_power_per_sample_w` = kT₀F·f_s = 6.3910e-14 W (−131.94 dBW) |
| `chirp_duration_s` | 50.0e-6 → 400 samples | For `"noise"` the correlation batch, the continuous radar's pulse; `prf_hz` = 20 000.0, its rate, duty one. No new key |
| `n_pulses` | 2048 batches → CPI 0.1024 s, **N = 819 200** samples | Processing gain 10·log10 N = **59.13 dB** |
| `n_range_lags` | 1024, a `[[burst]]` processing key like `n_pulses` | Lags 0 – 1023: routes 0 – 38 336 m, a window, not a circle |
| Lag `c/f_s` | **37.474 m** of route (18.737 m slant) | `pipeline.map_axes`, lag m at `c·m/f_s` |
| Doppler bin `1/CPI` | 9.765625 Hz → **0.29874 m/s** of bistatic rate (0.14937 m/s radial) | |
| Rate span `λ·PRF` | 611.82 m/s, ±305.91 m/s (±10 kHz) | The track's −110.6 … +78.6 m/s never folds |
| `n_taps` K | 192 | ≥ 161, the largest clutter lag + 1 |
| Batched cancellation | 6.4 ms = 51 200 samples, 16 a CPI | P2 |
| Sliding cancellation | block 50 µs = 400 samples (the correlation batch), window block ± 25 600 samples (6.45 ms), clipped to the CPI | P2 |
| Parallel-Hammerstein basis | orders 1 and 3 on all K taps → 384 columns | P3 |
| Doppler window | rectangular | §3.4's straddle losses assume it |
| CFAR | `cfar_detect` along range (the map's range axis, wrapped), CA, `pfa = 1e-6`, 16 training and 4 guard cells a side; then `cluster_detections` | Ideal threshold 11.40 dB. Pd = 0.5 needs 11.23 dB and Pd = 0.9 needs 13.11 dB (Albersheim, non-fluctuating) |

### 3.3 Disturbance budget and the three variants

The target's level uses the repo's `received_power_w`, both ranges equal, with 001's antennas. The
leakage and clutter are specified as levels at the ADC, per sample over noise:

| Component | Route `R_T + R_R` | Lag | Doppler | Level in the received stream |
| :--- | ---: | ---: | ---: | ---: |
| Leakage | 0 / 37.47 / 74.95 m | 0 / 1 / 2 | 0 | INR **40.00 / 30.00 / 20.00 dB** |
| Clutter 1 – 4 | 599.58 / 1498.96 / 2997.92 / 5995.85 m | 16 / 40 / 80 / 160 | 0 | CNR 30.00 / 24.00 / 18.00 / 12.00 dB |
| Total disturbance, P1 and P3 | | | | INR **40.945 dB** (12 430.1) |
| Aircraft, default window | 33 278 – 36 717 m | 888.0 – 979.8 | −2570.4 … +3615.6 Hz | SNR −31.87 … −30.17 dB; target/disturbance −72.8 … −71.1 dB |

*Where 40 dB comes from.* A 1 W transmitter behind 50 dB of antenna isolation would leak about
82 dB above the receiver's noise. An analog canceller takes most of that [3]_ [7]_. The 40 dB that
reaches the ADC is this slice's problem. Lags 1 and 2 stand for the front end's impulse response.

The clutter lags are whole samples by construction (`range_m = m·c/f_s`), so the claim that K taps
span the clutter is exact. Clutter off the sample grid has band-limited sinc tails into the
neighbouring lags, which is why K carries margin above 161.

| | **P1** `static_leakage` | **P2** `varying_clutter` | **P3** `distorted_transmitter` |
| :--- | :--- | :--- | :--- |
| Channel | Leakage and four clutter echoes at zero Doppler | P1 **plus** a 2.0 Hz copy of clutter 1 (route 599.58 m, lag 16), CNR 30.00 dB | P1 through a nonlinear amplifier: every path carries `f(x)`, not `x` |
| Config's method | **least squares, full CPI, K = 192** | **batched, 6.4 ms** | **least squares, parallel-Hammerstein basis** |
| Other methods (slow tests) | none · Wiener-SMI · K = 96 | full-CPI least squares · **sliding, 50 µs / 6.45 ms** | linear basis |
| What it teaches | The leakage floor buries the target; a projection onto the known code with enough taps removes it to the noise; too few taps leave the far clutter | A channel that changes inside the CPI escapes a one-bin notch; shorter fits widen the notch; a sliding window removes the batched fit's parasitic lines | The basis must model what the hardware did: a linear basis stops at the amplifier's distortion floor; odd-order columns remove it |

The 2 Hz component stands for a slowly varying near scatterer, a tree or a mast in wind, 0.205 of
a Doppler bin from zero: close enough to look static, far enough to escape a one-bin notch. It
shares clutter 1's route, so it starts in phase with it and turns 2π·0.2048 over the CPI; the
pair's mean power is `2 + 2·sin θ/θ` = 3.492 times 30 dB (θ = 2π·2 Hz·0.1024 s), not twice. P2's
total INR is therefore **41.738 dB** (14 922).

**The amplifier (P3).** `f(x) = (x + a₃|x|²x)/√(1 + 4a₃ + 6a₃²)` with `a₃ = −0.07` (Anttila [4]_
eq. (1), its first two odd terms): third-order compression of 7 % at mean power, normalised to
unit power with the complex-Gaussian moments `E|x|⁴ = 2`, `E|x|⁶ = 6`. Leakage, clutter and target
all carry it: the Hammerstein structure, a nonlinearity then a linear channel. By Bussgang [9]_
(§III, eq. (20)) a linear canceller removes only the part of `f(x)` correlated with `x`, the gain
`β = (1 + 2a₃)/√(1 + 4a₃ + 6a₃²)`; the rest is `2a₃²/(1 + 4a₃ + 6a₃²)` = **−18.835 dB** of every
path. The target loses `|β|²` = −0.057 dB to the mismatch between `f(x)` and the replica `x`.

### 3.4 The acceptance frames

The five acceptance frames are 0 – 4, where the track opens receding at a steady 56.16 m/s of
bistatic rate (28.08 m/s radial, `f_D` = −1835.9 Hz), on a Doppler bin. *Straddle* is range /
Doppler / batch loss. Range straddle is the CPI's mean of `sinc(m − τ_b)`: the route walks 0.153
of a lag over the CPI (5.75 m), from the frame's lag at the CPI start. The batch loss is
`sinc²(f_D T_B)` [6]_. *Clean* is N·SNR after straddle with no disturbance. All values are dB,
SINR over the floor's mean. P3's columns, and its clean run, carry the −0.057 dB mismatch.

| frame | Route (m) | Lag | f_D (Hz) | Straddle | Clean | None | P1 LS | P1 K = 96 | P2 full | P2 batched | P2 sliding | P3 linear | P3 PH |
| ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 35 121.4 | 937.22 | −1835.93 | −1.31 / −0.00 / −0.12 | 26.60 | −14.34 | 26.60 | 14.34 | 5.41 | 24.73 | 26.58 | 4.47 | 26.60 |
| 1 | 35 177.6 | 938.72 | −1835.95 | −0.64 / −0.00 / −0.12 | 27.24 | −13.70 | 27.24 | 14.98 | 6.05 | 25.37 | 27.22 | 5.11 | 27.25 |
| 2 | 35 233.7 | 940.22 | −1835.97 | −1.29 / −0.00 / −0.12 | 26.57 | −14.38 | 26.57 | 14.30 | 5.38 | 24.70 | 26.55 | 4.43 | 26.57 |
| 3 | 35 289.9 | 941.72 | −1836.00 | −0.66 / −0.00 / −0.12 | 27.17 | −13.77 | 27.17 | 14.91 | 5.98 | 25.30 | 27.15 | 5.04 | 27.17 |
| 4 | 35 346.0 | 943.21 | −1836.02 | −1.27 / −0.00 / −0.12 | 26.53 | −14.41 | 26.53 | 14.27 | 5.34 | 24.66 | 26.51 | 4.40 | 26.54 |

(P2's *None* is 0.79 dB lower again: its INR is 41.74 dB.) Predicted Pd at `pfa = 1e-6`, per frame:

- **None:** ≤ 1e-6.
- **P2, full-CPI least squares:** ≤ 1.1e-2.
- **P3, linear basis:** ≤ 4.7e-3.
- **P1 with K = 96:** ≥ 0.98. It is detected, 12.27 dB down.
- **P1 least squares, P2 batched and sliding, P3 parallel-Hammerstein:** 1.000 to three figures.

### 3.5 Mathematical contracts

Notation: `x[n]` the digital code, `f(x)` what the amplifier radiates, `s[n]` the received stream,
`N` the CPI in samples, `K` taps, `σ²` the noise a sample.

| Contract | Statement |
| :--- | :--- |
| The stream | One CPI, `(n_pulses·n_samples_per_chirp,)`, of a code repeated every CPI: delays are **circular**, so every tap and every lag reads a recorded sample, `x[(n − k) mod N]`, and no guard is needed. A zero-padded history would leave each clutter echo uncancelled for its first `d` samples and break the exact projection A2 and A6 test. An echo wraps only past `c·N/f_s` = 30 700 km of route |
| Time and range | Lag `m` is the delay `m/f_s`, route **`m·c/f_s`**. Nothing folds: `Radar.unambiguous_delay_s` is infinite for `"noise"`, since batched correlation reads each lag whole, past its own batch; the map is the window of `n_range_lags` lags |
| Signal model | `s[n] = Σ_p |a_p|·f(x)(n − τ_p,b)·e^{−j2π R_p(t_n)/λ} + w[n]` over every path. The envelope delay is held over each batch (stop-and-hop; the fastest path moves 1.5e-4 of a sample in a batch); the carrier phase follows `bistatic_range_at_m` at every sample, so each batch sees its Doppler as a ramp. A fractional delay is a phase ramp on the code's spectrum, exact for a periodic band-limited code; its change over the CPI is a Taylor series in the code's derivatives, stopped below 1e-10 of the path |
| Delay basis | `X = [z_p[n − k]]`, `z_p = |x|^{p−1}x` for each odd order p, k = 0 … K−1, order-major: from the **digital** code, which holds no noise |
| Normal equations | `G = XᴴX`, `c = Xᴴs` (Haykin [2]_ §9.5), never forming X: each order pair's first row and column are FFT correlations, and `G[i,j] = G[i−1,j−1] + z̄_p[a−i]z_q[a−j] − z̄_p[b−i]z_q[b−j]` adds the rows entering and leaving the window `[a, b)` |
| Least squares [2]_ | `w = G⁻¹c` by Cholesky, `y = s − Xw = (I − X(XᴴX)⁻¹Xᴴ)s` (§9.7). A projection: `Xᴴy = 0` exactly |
| Wiener-SMI [2]_ | The same solve on `R̂[k] = (1/N)Σ x[n]x̄[n−k]` (Toeplitz) and `r̂[k] = (1/N)Σ s[n]x̄[n−k]` with the record zero outside the CPI, the autocorrelation method of §9.2, by FFT. It equals least squares up to edge terms of order K/N |
| Batched [8]_ | Blocks of `N_B` samples, each fitted on itself (eqs. (3)-(4)). One block is the full-CPI fit |
| Sliding [8]_ | Blocks of `N_f` samples, each filtered with `w` fitted on the block widened by `N_e` a side, clipped to the CPI (eqs. (18)-(19)). `N_e = 0` is the batched fit with `N_B = N_f` |
| Parallel-Hammerstein basis [4]_ | `X_PH = [X, X₃]`, `X₃` the columns `|x[n−k]|²x[n−k]` (§III-B). The projection is unchanged; only the basis grows. The Gram of one tap's pair `[x, |x|²x]` is `N·[[1, 2], [2, 6]]`, 2-norm condition 22.46 |
| Doppler notch [8]_ | A lag-0 probe at Doppler `f` keeps `ρ(f) = (1 − |D(f)|²)²` of its correlation peak power, `D(f) = sin(πfN_c/f_s)/(N_c sin(πf/f_s))` the Dirichlet kernel of the fit length `N_c` (§III, eq. (16)): the fit removes `D` of the probe's amplitude, and correlation projects the rest onto the probe again. Nulls at `k·f_s/N_c`: the notch is **one Doppler bin** for the full CPI and `±1/T_B` batched. The residual *power* of a Doppler-`f` disturbance is `1 − |D|²`. For the sliding fit, block-centred, it is `1 − 2D_wD_f + D_w²` (`D_w`, `D_f` of the window and the block): fourth order in `f` |
| CA | `CA = Σ|s|² / Σ|y|²` over the CPI, dB at the boundary |
| CA bound | `E[CA] = (1 + INR)/[(1 − P/N_fit)(1 + INR_unc)]`, P basis columns. `INR_unc` is what the basis cannot represent: echoes past lag K−1; `P_c(1 − |D(f_c)|²)` for a Doppler-`f_c` component; `INR·2a₃²/(1 + 4a₃ + 6a₃²)` for distortion under a linear basis. The basis is noise-free, so the ceiling is `(1 + INR)/(1 − P/N_fit)`: cancel to the noise (§9.8) |
| Floor rise | Away from the zero-Doppler ridge the floor rises by the bound's denominator, and SINR falls by `1 + INR_unc` (the target lies past the last tap, so its peak is not projected away). Measured as the mean power of the 216 cells 2 – 4 Doppler bins and 3 – 20 lags from the target's cell, over a clean run of the same frame: the same code, noise and target, without the leakage and clutter paths. The target's own row and column are excluded: a white code's range response is a sinc |
| Batched correlation [6]_ | `y_b[m] = (1/L)Σ_{n<L} s[(bL + n + m) mod N]·x̄[bL + n]`, `m < n_range_lags`, then `doppler_fft` across b. Scaled so a unit-power code under `a·x[n − m]` gives `a`. Valid while `|f_D|T_B < ½`; batch loss `sinc²(f_D T_B)` |
| Notch and CFAR | The full-CPI and batched fits make the zero-Doppler cells below lag K exactly empty (`Xᴴy = 0` summed over the CPI); CFAR there would weigh round-off against round-off, so a crossing fainter than the map's median is not a detection |
| TOML layout | 001's tables, plus `n_range_lags` in `[[burst]]` and, after `[receiver]`: `[leakage]` (`lags`, `inr_db`), `[[clutter]]` (`range_m` route, `relative_power_db` over the noise a sample, `doppler_hz`), `[transmitter_distortion]` (`a3`), `[cancellation]` (`method = "none" | "least_squares" | "wiener"`, `n_taps`, `n_samples_per_batch` 0 = the CPI, `n_samples_extension`, `odd_orders = [1]`, `diagonal_loading = 0.0`, a fraction of the Gram's mean diagonal) |

**From the baseline to the current method, in five steps.** Each step is one function and one test.

1. The Wiener filter [2]_ (§2.4) estimating `s` from delayed copies of `x` is a least-squares fit.
   With sample statistics it is Wiener-SMI.
2. Written on the data matrix, the same fit is a projection (§9.7): exact, with `Xᴴy = 0`.
3. A projection holds one set of weights for the CPI, so it notches one Doppler bin. Fit per batch
   and the notch widens to `±1/T_B` [8]_.
4. Batch-wise weights jump at every batch edge and modulate the residual at multiples of `1/T_B`
   [8]_ (§III, eq. (17)). Estimate on a wide window and filter a short block, and the notch keeps
   the wide window's width while the lines move to `1/T_f`: with `T_f` the correlation batch, to
   ±20 kHz, outside the map (§IV).
5. A nonlinear amplifier puts power where no delayed copy of `x` reaches. Add the odd-order columns
   `|x|²x` [4]_. The projection, the test and the function are the same; only the basis grows.

**Intermediate quantities exposed**: the weights per block, `(n_blocks, n_basis)`; each block's
Gram condition number (LAPACK's 1-norm estimate from the Cholesky factor, Higham [10]_ ch. 15:
O(P²) a block, where an exact one is O(P³)); CA and its ceiling; the floor rise; the SINR loss; the
notch response `ρ(f)`; and the residual stream.

### 3.6 Outputs

The stepper's files (`spec/data-001-formats.md`). Additions:

| File | Addition |
| :--- | :--- |
| `iq_NNNNN.npz` | `burst0` is the received stream cut into its `(n_pulses, 400)` batches; the code is regenerated from the seed |
| `leakage.csv` | One row a frame: `cancellation_depth_db`, `cancellation_depth_bound_db`, `floor_rise_db`, `uncancelled_floor_rise_db`, `sinr_db`, `clean_sinr_db`, `sinr_loss_db`, `uncancelled_sinr_db`, `target_detections`, `uncancelled_target_detections`, `gram_condition_number_max` |
| `cancellation.png` | Frame 0's map before and after cancellation, one scale over the clean floor, truth crossed |

---

## 4. Core Decisions

### 4.1 The canceller's basis is the radar's own transmit code, beside a single-channel stream

A continuous-wave radar knows what it sent. The basis is the **digital transmit code**, carried
beside the received stream over the same samples: `Frame.references`, empty for other waveforms.
Every function in `clutter.py` takes the reference before the surveillance stream. The digital code
holds no receiver noise, so the cancellation ceiling is the noise itself (§3.5). What it does not
hold is what the hardware did to it: P3 shows the cost, and the parallel-Hammerstein basis is the
remedy.

### 4.2 The least-squares cancellers live in `clutter.py`

| Function | Returns | Note |
| :--- | :--- | :--- |
| `delay_basis_gram(reference, surveillance, n_taps, *, start, stop, odd_orders=(1,))` | `(gram (P, P), cross (P,))` | The normal equations of rows `[start, stop)`, by FFT and displacement; never forms X |
| `parallel_hammerstein_basis(reference, n_taps, odd_orders, start, stop)` | `(stop − start, P)` | The explicit basis, for tests |
| `ls_weights(gram, cross, *, diagonal_loading=0.0)` | `(w (P,), condition)` | `scipy.linalg.cho_factor`; loading a fraction of the mean diagonal |
| `least_squares_cancel(reference, surveillance, n_taps, *, n_samples_per_batch=0, n_samples_extension=0, odd_orders=(1,), diagonal_loading=0.0)` | `CancellationResult` | Full CPI, batched or sliding: one implementation |
| `ls_cancel`, `ls_cancel_batched`, `ls_cancel_sliding` | `CancellationResult` | One-line named wrappers |
| `wiener_smi(reference, surveillance, n_taps)` | `CancellationResult` | FFT sample statistics |
| `cancellation_depth_db(surveillance, residual)` | `float` | CA |
| `expected_cancellation_depth_db(inr_linear, n_basis, n_samples, inr_uncancelled_linear=0.0)` | `float` | The §3.5 bound |
| `doppler_notch_response_linear(doppler_hz, n_samples_cancellation, sample_rate_hz)` | `NDArray` | `ρ(f)` |
| `distortion_fraction_linear(a3)` | `float` | `2a₃²/(1 + 4a₃ + 6a₃²)` |

`CancellationResult` is a frozen dataclass: `residual`, `weights (n_blocks, P)`,
`gram_condition_number (n_blocks,)`. **Weights are returned, never applied silently.** A Python
loop runs over blocks: each is an independent solve, so the loop is the algorithm. The lag sums
and the filter run over short units of the CPI, at most 16 K long, chunked under a byte budget; a
block's window sums are differences of running sums over units, so one pass serves every window.

### 4.3 The stream, not the cube, is what cancellation sees

A pulse radar's cube is a set of independent repetitions. A continuous transmission is one signal:
a tap reaches 191 samples into the previous batch, and a lag reads 1023 samples into the next. So
synthesis produces one circular CPI of stream and code (§3.5), and the `(n_pulses, 400)` cube in
`Frame.cubes` is only its batches. Nothing that needs a neighbouring batch takes the cube view.

### 4.4 Why white Gaussian noise

A practical code has structure, such as a repetition period or a finite alphabet, and structure
puts deterministic ridges into the ambiguity function [6]_. Modelling them would make the sidelobe
floor a function of the code instead of `10·log10 N`, and the slice would be testing the waveform
instead of the cancellation. White Gaussian noise has no structure, so its floor is flat, and it
gives the amplifier of P3 closed-form moments: `signal.noise_waveform(n_samples, *, rng)`.

### 4.5 Out of scope, and why it is not a gap in the bar

- **Recursive adaptive filters, LMS, NLMS and RLS** [1]_ [2]_: they converge towards the same
  Wiener solution, with convergence as an extra variable; nothing a test here would catch.
- **Widely linear cancellation** [5]_: an IQ-imbalanced front end adds a conjugate term. The basis
  grows by conjugate columns, nothing else changes; it belongs with a receiver-chain scenario.
- **Analog cancellation and phase noise** [3]_ [7]_: leakage is specified at the ADC. Phase noise
  decorrelates the leakage from the digital code, the next limit after distortion; it needs an LO
  model.
- **Target self-cancellation**: the aircraft is at lags 888 – 980, outside K = 192, so the
  projection cannot reach it. A slow target inside the first 7.2 km of route is lost to the notch:
  the notch working (A5).
- **Notebooks**: dropped from this slice; the stepper's `leakage.csv` and `cancellation.png` show
  the dynamic-range ladder, the notch and the distortion floor.

### 4.6 Library widenings

- `radar.py`: `WaveformKind` gains `"noise"`; `Radar` requires `bandwidth_hz == sample_rate_hz` for
  it; `unambiguous_delay_s` is infinite for it; `n_samples_per_chirp` is the batch.
- `propagation.py`: `leakage_paths(radar, lags, inr_db)`, `static_clutter_paths(radar,
  bistatic_range_m, cnr_db, *, doppler_hz=0.0)`, each one bounce, per-sample levels;
  `PropagationPaths.concatenate`.
- `signal.py`: `noise_waveform`, `cubic_amplifier(samples, a3)`, `stream_baseband(paths, radar,
  n_pulses, *, rng, reference=None, amplifier=None) -> (surveillance, reference)`, one path at a
  time (memory stays a few streams, so no byte budget).
- `dsp.py`: `batched_correlation(surveillance, reference, n_range_lags, n_samples_per_batch)`,
  `fftconvolve` per chunk of batches.
- `scenario.py`: `n_range_lags`; `LeakageSettings`, `ClutterSettings`, `DistortionSettings`,
  `CancellationSettings`, typed and validated like `DetectionSettings`; a noise burst stands alone.
- `pipeline.py`: `synthesise` dispatches `"noise"` (its cube; a fixed code when noiseless);
  `frames` carries the code in `Frame.references`; `cancel`, `disturbance_paths`,
  `frame_products`; `range_doppler_product(..., reference=, n_range_lags=)` and `map_axes(...,
  n_range_lags=)` for the noise map. The other waveforms are bit-identical (golden fingerprints).

### 4.7 Workstream boundary

```
config/scenario_005_{static_leakage,varying_clutter,distorted_transmitter}.toml   (with [verify])
src/radar_forge/clutter.py, src/radar_forge/stages/clutter.py ("leakage"), tests/test_clutter.py
src/radar_forge/{radar,propagation,signal,dsp,scenario,pipeline,plots}.py         (additive widenings)
```

`detection.py`, the windows in `dsp.py`, the geodesy in `radar.py` and the trajectory reading in
`scenario.py` are consumed, never edited. The `[verify]` tables of scenarios 001–003, run by
`tests/test_scenarios.py`, stay green at every commit.

---

## 5. Build Sequence

```
  1. radar.py, propagation.py, signal.py, dsp.py     waveform, paths, stream, batched correlation
  2. clutter.py, tests/test_clutter.py               A1, A2, A4, A5 first: projection, degeneracy, notch
  3. scenario.py, pipeline.py                        tables; the "noise" dispatch; frame_products
  4. stages/clutter.py, plots.render_cancellation    the "leakage" stage
  5. config/scenario_005_*.toml                      three files, their [verify] tables (A6, A8-A11, A13)
  6. tests/test_clutter.py, slow                     A3, A7, A11's baselines, P2 sliding, A12, A13 linear
```

---

## 6. Acceptance Criteria

Unit tests (`tests/test_clutter.py`) are small and fast. The configs' `[verify]` tables run three
frames in the fast suite and the whole window under `slow`; the other methods' checks are slow tests
on the same frames.

| # | Criterion | Bound | Tolerance and why | Where |
| :--- | :--- | :--- | :--- | :--- |
| A1 | **Degeneracy chain.** Sliding with `N_e = 0` equals batched with `N_B = N_f`; one batch equals the full-CPI fit; the full-CPI fit equals the exact-statistics Wiener solution `(XᴴX/N)⁻¹XᴴS/N`; the parallel-Hammerstein fit with `odd_orders=(1,)` equals lstsq on the linear basis | residuals equal | `rtol = 1e-9`: the same normal equations solved twice | unit |
| A2 | **Projection.** `Xᴴy = 0` for every batch, linear and parallel-Hammerstein | `‖Xᴴy‖ / (‖X‖‖s‖) ≤ 1e-10` | float64 round-off of a Cholesky solve at a condition number under 100 | unit |
| A3 | Wiener-SMI against least squares | residual powers agree | `rtol = 1e-3`: edge terms O(K/N) = 2.3e-4, four times over | P1, slow |
| A4 | **Notch shape.** A noiseless lag-0 probe, K = 1. Its response equals `ρ(f)` from the realised code exactly; in expectation 0.00 dB at 9.765625 Hz for the full-CPI fit (a null of `D`); for 6.4 ms batches −37.87 dB at 9.765625 Hz, −4.51 dB at 78.125 Hz, 0.00 dB at 156.25 Hz | exact: `rtol = 1e-9`; expectation ±1.0 dB at 9.765625 Hz batched, ±0.3 dB elsewhere | Batch-coefficient scatter about `1/√N_B` | unit |
| A5 | **One bin wide.** The full-CPI fit leaves ±9.765625 Hz at 0 dB; a static echo inside the tap span is removed | ±0.1 dB; removal ≤ −100 dB noiseless | `|D|²` at a null is O(1/N) | unit |
| A6 | **Cancellation bound.** P1, K = 192: CA = 40.946 dB | ±0.05 dB | A power over 819 200 samples scatters 0.005 dB | P1 `[verify]` |
| A7 | **Tap count.** K = 96 cannot reach lag 160: CA = 28.680 dB, floor rise 12.27 dB, the target still detected | CA ±0.05; floor ±1.0 | As A6 and A9 | P1, slow |
| A8 | **Target at the true bins.** After cancellation the CFAR detection nearest the truth sits within one lag (37.474 m of route) and one Doppler bin (0.29874 m/s of bistatic rate) of `truth_geometry` | every frame | 001's contract, through the lag axis | P1, P2, P3 `[verify]` (`truth_within_bins`, `rate_tolerance_mps`) |
| A9 | **Floor rise matches the budget.** P1 least squares 0.00 dB; P2 batched 1.86 dB; P3 linear 22.14 dB | ±1.0 dB | 216 cells of exponential power scatter 0.3 dB; the clean run shares the noise, so less | P1, P2 `[verify]`; P3 linear slow |
| A10 | **SINR after cancellation.** Within 1.0 dB of the clean run for P1 least squares and P3 parallel-Hammerstein, and of the predicted 1.87 dB loss for P2 batched | ±1.0 dB | At 25 dB the noise in the peak cell moves it ±0.5 dB | P1, P2, P3 `[verify]` (`sinr_loss_db`) |
| A11 | **The baselines fail; the current methods succeed.** No cancellation detects the target in none of the frames; full-CPI least squares on P2 and the linear basis on P3 in at most one of five (P(≥ 2 of 5) ≤ 1.2e-3); P1 least squares, P2 batched and sliding, P3 parallel-Hammerstein in every frame | counts | Fixed seed; the counts carry margin against the predicted Pd | `[verify]` (`detections_per_frame`, `uncancelled_detections_per_frame`); baselines slow |
| A12 | **Sliding removes the batched fit's parasitic lines.** At lag 16 and ±156.25 Hz the sliding map (50 µs / 6.45 ms) is ≥ 20 dB below the batched one; its far floor is no higher | ≥ 20 dB; ≤ | The lines are the batched fit's 1/T_B modulation of the 2 Hz residual | P2, slow |
| A13 | **The amplifier limits a linear basis.** P3 linear: CA = 18.810 dB. Parallel-Hammerstein: CA = 40.947 dB | ±0.1 dB | `|x|⁶` statistics scatter about 0.02 dB | P3 `[verify]`; linear slow |
| A14 | Absolute received power | **not asserted** | — | — |

The P2 sliding fit (2048 Cholesky solves a frame) is checked by a slow test on the P2 config: CA
41.717 dB ±0.05 (the bound with the clipped windows' 2 Hz residual, 8.7e-6 of its power, and their
noise loss), floor rise 0.02 dB ±1.0, SINR within 1.0 dB of clean, the target every frame.

> A6, A7 and A13 assert CA. A11 asserts detection. CA alone can reward a canceller that also
> removes the target, which is why A9 and A10 measure the floor and the peak separately.

---

## References

.. [1] B. Widrow, J. R. Glover, J. M. McCool, J. Kaunitz, C. S. Williams, R. H. Hearn et al.,
       "Adaptive noise cancelling: principles and applications," *Proc. IEEE*, 63(12),
       pp. 1692-1716, 1975. §III, eq. (1): the two-input canceller.
.. [2] S. Haykin, *Adaptive Filter Theory*, 5th ed., Pearson, 2014. §2.4 Wiener-Hopf equations;
       §9.2 data windowing; §9.5 normal equations; §9.7 the normal equations on data matrices;
       §9.8 properties of least-squares estimates.
.. [3] A. Sabharwal, P. Schniter, D. Guo, D. W. Bliss, S. Rangarajan and R. Wichman, "In-band
       full-duplex wireless: challenges and opportunities," *IEEE J. Selected Areas in
       Communications*, 32(9), pp. 1637-1652, 2014, arXiv:1311.0456. §IV-B to §IV-D.
.. [4] L. Anttila, D. Korpi, V. Syrjälä and M. Valkama, "Cancellation of power amplifier induced
       nonlinear self-interference in full-duplex transceivers," *Proc. Asilomar Conf. Signals,
       Systems and Computers*, pp. 1193-1198, 2013, arXiv:1401.3129. Eq. (1), the parallel
       Hammerstein model; §III-B, its least-squares estimate.
.. [5] D. Korpi, L. Anttila, V. Syrjälä and M. Valkama, "Widely linear digital self-interference
       cancellation in direct-conversion full-duplex transceiver," *IEEE J. Selected Areas in
       Communications*, 32(9), pp. 1674-1687, 2014, arXiv:1402.6083.
.. [6] K. Kulpa, *Signal Processing in Noise Waveform Radar*, Artech House, 2013,
       ISBN 978-1-60807-661-1. Ch. 3, noise radar: its correlation receiver and batch processing.
.. [7] A. Melzer, A. Onic, F. Starzer and M. Huemer, "Short-range leakage cancelation in FMCW radar
       transceivers using an artificial on-chip target," *IEEE J. Selected Topics in Signal
       Processing*, 9(8), pp. 1650-1660, 2015.
.. [8] F. Colone, C. Palmarini, T. Martelli and E. Tilli, "Sliding extensive cancellation
       algorithm for disturbance removal in passive radar," *IEEE Trans. Aerospace and Electronic
       Systems*, 52(3), pp. 1309-1326, 2016. §II eqs. (3)-(4), ECA-B; §III eqs. (16)-(17), its
       notch and parasitic lines; §IV eqs. (18)-(19), ECA-S.
.. [9] J. J. Bussgang, "Crosscorrelation functions of amplitude-distorted Gaussian signals," MIT
       Research Laboratory of Electronics, Tech. Rep. 216, 1952. §III, eq. (20).
.. [10] N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
        Ch. 15, condition number estimation.
