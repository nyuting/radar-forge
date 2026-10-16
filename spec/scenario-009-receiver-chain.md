# Scenario 009 — Receiver chain: front end, digital down-conversion, channel impairments and pilot-tone calibration

## 1. Status & Purpose

**Status: specified, not implemented.**

Scenarios 001–003 synthesise complex baseband directly: `core/signal.py` writes the samples a
perfect quadrature receiver would deliver, at the output rate, with the noise power `k·T0·B·F`
assumed rather than produced. That is the right idealisation for teaching range and Doppler. It
hides everything that happens between the antenna port and the cube, though, and a multi-channel
array (scenario-006 D13) is only as good as the agreement between its channels.

This document specifies the **receive chain as a vertical slice**, on the 16-channel ground array:

1. **Front end (C1).** A real-IF signal with wideband thermal noise and an off-channel interferer
   is sampled, digitally down-converted, filtered and decimated to the cube's rate. The slice shows
   where the SNR gain of decimation comes from, how far the interferer is pushed down, and where it
   would have landed without the filter.
2. **Channel mismatch and pilot-tone calibration (C2).** Per-channel complex gain and group delay
   raise the array's sidelobes. A single injected reference recovers them. A single *tone* fails at
   the band edge and a *multi-tone* pilot does not: that is the teaching contrast.
3. **IQ imbalance and phase noise (C3).** Analog quadrature demodulation creates an image at −f,
   and digital down-conversion does not. Blind circularity-based compensation is the current method
   and a pilot-based estimate is the baseline. A common LO's phase noise is range-correlated: the
   slice shows its Doppler skirt against the closed form, and what calibration can and cannot remove.

The direct-baseband path is **not changed**. The front end is an optional stage selected by the
presence of a `[front_end]` table, exactly as `[transmitter_site]` selects bistatic siting in
scenario 002. Absent, every cube is bit-identical to today's (§6, A1).

| Section | What it fixes |
| :--- | :--- |
| §2 Overview & Geometry | The chain, stage by stage, with the sample rate at each |
| §3 Parameter Matrix | Array, waveform, front end, impairments, pilot, the three variants, the contracts |
| §4 Core Decisions | D20 (chain order and rates), D21 (explicit impairments; calibration returns parameters) |
| §5 Build Sequence | What is new, what is widened, what is untouched |
| §6 Acceptance Criteria | Analytic bounds, led by bypass bit-identity |

### 1.1 Upstream inspiration

| Upstream | What it is | What radar-forge takes | Licence | Baseline or current |
| :--- | :--- | :--- | :--- | :--- |
| Crochiere & Rabiner, *Multirate Digital Signal Processing*, 1983 [1] | The multirate textbook: decimation, polyphase decomposition | The noble identity: filter-then-discard equals the polyphase form at 1/D the arithmetic | book, written from equations | Baseline (FIR-then-decimate is the oracle) |
| Harris, *Multirate Signal Processing for Communication Systems*, 2004 [2]; Harris, Dick & Rice, IEEE TMTT 2003 [3] | Polyphase channelisers for digital receivers | The M-path polyphase decimator and the FFT channeliser that down-converts every channel at once | book and paper, written from equations | **Current** |
| GNU Radio `gr-filter` `pfb_channelizer_ccf` | Production polyphase channeliser | Structure only, as a cross-check of the channeliser's indexing | GPL-3.0; never vendored | Reference |
| liquid-dsp `firpfbch` | Polyphase filter-bank channeliser in C | Structure only | MIT | Reference |
| Parks & McClellan, IEEE TCT 1972 [4], via `scipy.signal.remez` | Equiripple FIR design | The decimation filter's design; the window method is the baseline it beats | paper; SciPy BSD-3 (existing dependency) | Current (equiripple) / baseline (Kaiser window) |
| Tarighat, Bagheri & Sayed, IEEE TSP 2005 [5] | IQ-imbalance model and pilot-based compensation | The `K1·s + K2·s*` model; the pilot-tone estimator | paper, written from equations | Baseline |
| Anttila, Valkama & Renfors, IEEE TVT 2008 [6] | Blind, circularity-based IQ-imbalance compensation | The one-tap special case: drive `E[y²]` to zero on operating data | paper, written from equations | **Current** |
| Raven, Proc. IEEE 1966 [7]; Budge & Burt, IEEE NRC 1993 [8] | Master-oscillator requirements; range correlation of LO phase noise | The `4·sin²(π f τ)` filter on a common LO's phase noise | paper, written from equations | Current (the closed form is still the truth) |
| IEEE Std 1139-2008 [9] | Definitions of phase-noise quantities | `L(f) = S_φ(f)/2`, one-sided | standard, written from definitions | — |
| Mailloux, *Phased Array Antenna Handbook*, 2005 [10] | Random-error effects on array patterns | The mean sidelobe floor `σ²·Σw²/(Σw)²` | book, written from equations | Oracle |

---

## 2. Overview & Geometry

The scene is scenario 001's: the Duke Receiver site and the lead aircraft's track from
`data/flight_coordinates.csv`. The receive side becomes the shared 16-element ULA of
scenario-006 D13. The slice adds what sits between each element and the cube:

```
     element n  (n = 0..15, half-wavelength ULA, scenario-006 D13)
         |
         v
  [ RF/IF analog ] -- group delay tau_n, complex gain g_n, per-pulse LO phase   (array/impairments.py)
         |                                   ^ common LO phase noise phi(t)     (core/signal.py)
         |                                   | pilot injected here, one source split to all channels
         +------------------- digital_if ---------------------+------- analog_iq -------+
         v                                                    |                         v
  real IF at f_IF = 5 MHz                                     |       quadrature mixer at RF/IF
  + wideband thermal noise, N0 = k T0 F over 0..10 MHz        |       + noise, then IQ imbalance
  + off-channel CW interferer at f_IF + 3 MHz                 |         z = K1 s + K2 s*
         |                                                    |                         |
         v  ADC, real, 20 MHz                                 |                         |
  NCO: x * sqrt(2) exp(-j 2 pi f_IF n / fs)  complex, 20 MHz  |                         |
         |                                                    |                         |
         v                                                    |                         |
  lowpass + decimate by 8, polyphase  (core/frontend.py)      |                         |
         |   complex, 2.5 MHz                                 |                         |
         v                                                    |                         |
  drop 9-sample group delay, re-centre +B/2                   |                         |
         +----------------------------------------------------+-------------------------+
         v
  cube (n_pulses, n_samples, n_rx) = (256, 100, 16), complex, 2.5 MHz  -- identical layout to today
         |
         v
  core/dsp.py  matched_filter -> range_doppler_map        <-- UNCHANGED
  array/calibration.py  pilot -> ChannelCalibration       (parameters only, D21)
  array/beamforming.py  (scenario 006) consumes the calibration
```

Sample rate at each stage, digital-IF path:

| Stage | Signal | Rate | Noise bandwidth |
| :--- | :--- | ---: | ---: |
| Synthesis | complex baseband, noiseless, `pulsed_baseband` at the ADC rate | 20 MHz complex | — |
| Up-conversion | real IF, signal at 4–6 MHz | 20 MHz real | — |
| ADC input | + thermal noise + interferer | 20 MHz real | **10 MHz** (first Nyquist zone) |
| NCO mix | complex, signal at 0 Hz, its negative-frequency copy at −10 MHz | 20 MHz complex | 20 MHz |
| Polyphase decimator | lowpass ±1.0 MHz, stop ≥1.5 MHz, keep 1 in 8 | **2.5 MHz complex** | ENBW **2.3057 MHz** |

The analog-IQ path skips the IF: it is today's direct-baseband synthesis followed by an explicit
IQ-imbalance model. Which path a variant uses is a TOML choice (§3.4).

---

## 3. Parameter Matrix

### 3.1 Array, site and waveform

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Site, target track | Duke Receiver; `data/flight_coordinates.csv` | Scenario-001 §3.1, unchanged |
| Array | 16-element ULA, `n_rx = 16`, spacing λ/2 = **15.296 mm** | Shared ground array, scenario-006 D13 |
| Steering vector | `a_n = exp(+j·2π·(p_n·u)/λ)`, unnormalised, `y = wᴴx` | Scenario-006 D13 |
| Beam taper | Taylor [11], 35 dB, n̄ = 6 (`core/windows.py` `taper("taylor", 16, sidelobe_db=35.0)`) | Ideal PSL **−34.73 dB**, HPBW **8.52°**, first null **11.99°** |
| `f0_hz` | 9.8 GHz, λ = **30.591 mm** | Scenario-001 |
| Waveform | Scenario-001 **S2**: LFM pulse, `bandwidth_hz` 2.0 MHz, `chirp_duration_s` 10 µs, `prf_hz` 25 kHz | Scenario-001 §2.1 |
| Output `sample_rate_hz` | 2.5 MHz, 100 samples/PRI, `n_pulses` 256 | Scenario-001 S2 |
| `noise_figure_db` | 3.0 | Scenario-001 |

S2, not S1, because a front end must carry the signal's RF bandwidth. S1's deramp collapses the
2 MHz sweep before sampling (scenario-001 §2.1), so a real-IF chain in front of it would teach
nothing about the 2 MHz channel. S2 samples the signal itself.

### 3.2 Front end (`[front_end]`, digital-IF path)

| Parameter | Value | Derived / why |
| :--- | :--- | :--- |
| `adc_sample_rate_hz` | 20.0 MHz, real | `decimation_factor · sample_rate_hz` |
| `if_frequency_hz` | 5.0 MHz | `fs/4`: the NCO is the sequence `1, −j, −1, +j` |
| `decimation_factor` | 8 | 20 MHz → 2.5 MHz |
| `passband_edge_hz`, `stopband_edge_hz` | 1.0 MHz, 1.5 MHz | Stop edge `fs_out − f_pass`: nothing that folds into ±1 MHz survives |
| `passband_ripple_db`, `stopband_attenuation_db` | 0.1 (peak to peak), 80 | Design targets |
| Filter design | `scipy.signal.remez`, **145 taps** (odd, 16k+1), normalised to unit DC gain | Achieves **0.0960 dB** ripple, **84.83 dB** stopband |
| Filter group delay | 72 input samples = **9 output samples** = 3.6 µs | Odd length `16k+1` makes it an integer number of output samples |
| Polyphase branches | 19 (145 taps zero-padded to 152) | 145 MACs per output, against 1160 for filter-then-discard |
| Window-method baseline | Kaiser at 145 taps: **36.09 dB** stopband; `kaiserord` needs **202 taps** for 80 dB | Why equiripple is the design |
| ENBW | **2.3057 MHz** = 0.9223 · `fs_out` | `fs_adc · Σh²/(Σh)²` |
| Interferer | CW at `if_frequency_hz + 3.0 MHz` = 8 MHz, `interference_to_noise_db` = 50 dB | INR referred to the ADC's real noise power |
| Interferer, no filter | Folds to **±0.500 MHz**, inside the passband, at +3.01 dB total | +3 MHz → `3.0 − 2.5`; its +7 MHz line → `7.0 − 7.5` |
| Interferer, channeliser | Channel 3 (centre 7.5 MHz), +0.5 MHz offset; the signal is channel 2 | `round(f / (fs_adc/D))` |
| ADC | Ideal, no quantisation | §4.6 |

Noise and SNR, with `N0 = k·T0·F` = **7.9888e-21 W/Hz**:

| Quantity | Value |
| :--- | ---: |
| Real noise at the ADC, `N0·fs_adc/2` | 7.9888e-14 W |
| Complex noise per output sample, `N0·fs_adc·Σh²` | **1.8420e-14 W** |
| Ideal receiver's assumed noise, `k·T0·B·F` | 1.5978e-14 W (0.618 dB lower) |
| SNR gain, real ADC to output, `10·log10(fs_adc / (2·ENBW))` | **6.372 dB** |
| The same, decomposed | `10·log10 8` (9.031) − 3.010 (real ADC: noise occupies `fs/2`) + 0.351 (ENBW is 0.9223 `fs_out`) |
| Simulated (40 000 output samples) | 6.356 dB |
| Interferer suppression, predicted `−10·log10(|H(3 MHz)|² + |H(7 MHz)|²)` | **84.88 dB** (simulated 84.88 dB) |

The real CW tone has two spectral lines after the NCO: +3 MHz, and −13 MHz, which folds to
+7 MHz at the 20 MHz rate. Both fall in the stopband, so the suppression bound is
`A_s − 10·log10 2` = **81.82 dB**, not `A_s`.

The output noise is 0.618 dB above the ideal receiver's `k·T0·B·F`. The ideal receiver assumes a
noise bandwidth equal to the signal bandwidth; a real filter's ENBW is 2.3057 MHz. Neither is wrong.
The front-end path reports its own `noise_power_w`, and the gap is a teaching quantity (§3.6).

### 3.3 Impairments (`[impairments]`) and pilot (`[calibration]`)

| Parameter | C2 | C3 | Model |
| :--- | :--- | :--- | :--- |
| `gain_error_rms_db` | 1.0 | — | `|g_n|` log-normal |
| `phase_error_rms_deg` | 20.0 | — | `arg g_n` Gaussian |
| `group_delay_rms_s` | 20 ns | — | `H_n(f) = g_n·exp(−j2π f τ_n)` at baseband; carrier phase is in `g_n` |
| `iq_gain_imbalance_db`, `iq_phase_imbalance_deg` | — | 0.5, 3.0 (drifted to 0.8, 5.0 for A13) | `K1 = (1 + g·e^{−jφ})/2`, `K2 = (1 − g·e^{+jφ})/2` |
| Common LO phase noise `L(f)` | — | −60, −85, −105, −125, −145 dBc/Hz at 100 Hz, 1 k, 10 k, 100 k, 1 MHz; log-log interpolated, flat outside, band-limited to ±1.25 MHz | Total **1.046°** rms |
| `channel_lo_phase_rms_deg` | — | 5.0 per channel per pulse | Independent synthesisers re-locking each pulse |
| Draws | seeded, `[scenario].seed` | | `style.md` §7 |

Group delay: 20 ns is about 4 m of cable at 0.67 c, plausible before calibration. It moves the
range peak by 3 m against a 74.95 m bin, so its only visible effect is the phase slope across the
band: **7.200°** rms at the band edge (±1 MHz), 4.157° rms averaged over the band.

| Pilot parameter | Value | Derived |
| :--- | :--- | :--- |
| Injection | One source, split to every channel's input (equivalently a boresight far-field source, `a(0) = 1`) | Single reference; channel 0 is the anchor |
| `pilot_n_samples` | 1000 at 2.5 MHz (0.4 ms) | — |
| `pilot_snr_db` | 20.0 per sample per channel, total over all tones | — |
| Baseline pilot | one tone at 0 Hz | Phase CRB **0.128°** per channel, **0.181°** relative to the reference |
| Current pilot | 9 tones, −0.8 … +0.8 MHz in 0.2 MHz steps, equal power, on DFT bins | Phase-at-centre CRB 0.128°; delay CRB **0.689 ns**, **0.975 ns** relative to the reference |
| CPE pilot gate (C3) | last 20 samples of each PRI (8 µs; ranges 4.80–6.00 km blanked), SNR 30 dB | Per-pulse differential phase CRB **0.405°** |

### 3.4 The three variants

| | **C1** `frontend-ddc` | **C2** `channel-mismatch` | **C3** `iq-phase-noise` |
| :--- | :--- | :--- | :--- |
| Front-end path | digital IF | digital IF | **analog IQ** (and digital IF for the A12 contrast) |
| Interferer | CW, +3 MHz, 50 dB INR | none | none |
| Impairments | none | gain, phase, group delay | IQ imbalance, common LO phase noise, per-channel LO phase |
| Calibration | — | single-tone (baseline) vs 9-tone pilot (current) | pilot IQ (baseline) vs blind circularity (current); pilot-gate CPE |
| What it teaches | Noise bandwidth, stopband, aliasing, polyphase = filter-then-discard | Sidelobe rise and recovery; why a tone cannot calibrate a delay | Digital DDC has no image; range correlation; what calibration cannot fix |

C2 runs on the digital-IF path so that IQ imbalance cannot exist, and the sidelobe rise is due to
gain, phase and delay alone. C3 runs on the analog-IQ path because that is where the image
arises; A12 runs the same tone through C1's chain to show the image is absent there.

Sidelobe levels for C2, 400 seeded draws, Taylor 35 dB weights, pattern from the effective weights
`w_n·c_n(f)/ĉ_n(f)` against the true manifold:

| Case | PSL median | PSL 95th pct | Mean sidelobe power |
| :--- | ---: | ---: | ---: |
| Ideal | −34.73 dB | — | — |
| Uncalibrated (predicted floor `σ²·Σw²/(Σw)²` = **−19.81 dB**) | −15.35 dB | −11.68 dB | −19.85 dB |
| Single-tone pilot, at band centre | −34.36 dB | −34.06 dB | −38.10 dB |
| Single-tone pilot, **at band edge** (+1 MHz) | **−23.59 dB** | −20.72 dB | −28.59 dB |
| 9-tone pilot, at band edge | −34.09 dB | **−33.56 dB** | −38.06 dB |

`Σw²/(Σw)²` = 0.07726 (−11.12 dB) for this taper. The single tone calibrates the centre
frequency perfectly and leaves `2π f τ_n` untouched; at the band edge that is a 7.2° rms phase
error and an 11 dB sidelobe rise.

C3's phase-noise quantities. The two delays are chosen on the 0.4 µs sample grid, and 160 µs puts
`1/τ` = 6.25 kHz exactly on Doppler bin 64 of 256:

| `τ` (range) | `4 sin²(πfτ)` at 195.3 Hz | at 976.6 Hz | at 4980.5 Hz | at 6250 Hz | Residual rms |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 10 µs (1498.96 m) | −38.22 dB | −24.24 dB | −10.13 dB | −8.17 dB | 0.0641° |
| 160 µs (23 983.40 m) | −14.15 dB | −0.51 dB | +1.52 dB | **−121 dB (null)** | 0.2745° |

Doppler skirt per bin relative to the carrier, folded formula of §3.5, with a simulated check
(300 runs):

| Bin (offset) | τ = 10 µs formula / sim | τ = 160 µs formula / sim | Independent oscillators |
| :--- | ---: | ---: | ---: |
| 2 (195.3 Hz) | −81.95 / −81.98 dB | −61.52 / −61.99 dB | −44.36 dB |
| 10 (976.6 Hz) | −83.14 / −82.97 dB | −65.34 / −65.25 dB | −61.81 dB |
| 51 (4980.5 Hz) | −83.15 / −83.45 dB | −76.96 / −77.31 dB | −75.47 dB |
| 64 (6250 Hz) | −83.15 / −83.34 dB | first-order null / −92.02 dB | −77.11 dB |

IQ imbalance at (0.5 dB, 3°): IRR **28.200 dB** closed form and by a simulated tone (the
small-error form `4/(ε² + φ²)` gives 28.06 dB). After blind compensation over one CPI
(K = 25 600 samples per channel), median IRR 48.46 dB and mean residual image −47.25 dB, against
the sample-limited prediction `−10·log10(2K)` = **−47.09 dB**. Drift to (0.8 dB, 5°): raw IRR
23.95 dB. The pilot estimate taken before the drift then gives **32.18 dB**, and blind
re-estimation on the drifted data gives a median of 48.71 dB.

### 3.5 Mathematical contracts

Scenario-001 §3.5 holds unchanged for the cube, the RD map and the sign conventions.

| Contract | Statement |
| :--- | :--- |
| Up-conversion | `x[n] = √2·Re{s[n]·e^{−jπ B n/fs_adc}·e^{+j2π f_IF n/fs_adc}}`. The `√2` keeps real power equal to `|s|²`. The `e^{−jπB…}` re-centres the library's 0…B sweep (`pulsed_baseband`'s convention) on 0 Hz, and the output is shifted back by `+B/2` after decimation |
| Thermal noise at the ADC | Real Gaussian, variance `N0·fs_adc/2`, `N0 = k·T0·F` (W/Hz, one-sided), drawn from a **child** generator of the scenario seed, so the parent stream is not consumed |
| Interferer | `√(2·P_i)·cos(2π(f_IF + Δf)n/fs_adc + ψ)`, `P_i = INR·N0·fs_adc/2`; `0 < f_IF + Δf < fs_adc/2`, or `ValueError` (an analog anti-alias filter is assumed; the first Nyquist zone is all the model represents) |
| DDC | `z[n] = √2·x[n]·e^{−j2π f_IF n/fs_adc}`, then `y[m] = Σ_k h[k]·z[mD − k]`, `Σh = 1` |
| Polyphase identity | `y` from `scipy.signal.upfirdn(h, z, 1, D)` equals filter-then-discard to float round-off [1] |
| Channeliser | Branch `p` takes `z[mD − p]` with taps `h[p + kD]`; a D-point IFFT across branches gives channel `c` centred on `c·fs_adc/D`. Channel `c` equals the DDC at `f = c·fs_adc/D` up to a known phase factor per output sample [2], [3] (equation numbers to confirm at implementation) |
| Group-delay alignment | Output sample `m` is taken from index `m + (N_taps − 1)/(2D)` = `m + 9`; the generator synthesises `n_pulses + 1` PRIs and the stream is cropped, so the cube is `(n_pulses, n_samples, n_rx)` with no edge loss |
| Output noise | `N0·fs_adc·Σh²` W per complex sample; this is what `noise_power_w` reports in the front-end path |
| SNR gain | `10·log10(fs_adc/(2·ENBW))` for real IF sampling; `10·log10(fs_adc/ENBW)` for a complex input. Either equals `10·log10 D` (± the real-sampling 3.01 dB) only when ENBW = `fs_out` exactly |
| Channel response | `c_n(f) = g_n·exp(−j2π f τ_n)`, applied in the frequency domain along the zero-padded fast-time stream |
| IQ imbalance | `z = K1·s + K2·s*`, `K1 = (1 + g e^{−jφ})/2`, `K2 = (1 − g e^{+jφ})/2`, `g` the Q/I **gain ratio** (1 is ideal), `φ` the phase error, in the form of [6] (equation number to confirm at implementation); [5] writes the same widely-linear model in other symbols. Physically: `I = Re s`, `Q = g·(Im s·cos φ − Re s·sin φ)`, which equals the `K1, K2` form to round-off |
| IRR | `|K1|²/|K2|² = (1 + 2g cos φ + g²)/(1 − 2g cos φ + g²)`. Here `g` is the ratio, not the deviation; with `g = 1 + ε`, IRR ≈ `4/(ε² + φ²)` |
| Compensator | `y = z + w·z*`. Pilot (baseline): tone at `+f_p`, `w = −Y(−f_p)/conj(Y(+f_p))`. Blind (current): `w = −E[z²]/(2·E[|z|²])`, the one-tap first-order solution of `E[y²] = 0` [6] (equation number to confirm at implementation); infinite-sample bias leaves IRR at 84.60 dB here |
| Pilot estimate | `ĉ_n(f_m) = (1/K)·Σ_k y_n[k]·e^{−j2π f_m k/fs}·√M`, relative `ĉ_n/ĉ_0`. Fit `arg = ψ_n − 2π f τ_n` by least squares; amplitude is the geometric mean over tones |
| Pilot CRB | Phase per tone `M/(2K·SNR)` rad²; delay `var(τ̂) = σ²_φ/(4π²·Σ(f_m − f̄)²)`; ×2 for channel 0's own error |
| Phase-noise PSD | `S_φ(f) = L(|f|)` two-sided (rad²/Hz), `L` per [9] |
| Phase-noise synthesis | Gaussian spectral synthesis on the fast-time grid (FFT of white Gaussian noise shaped by `√(L(f)·Δf)`), over the CPI plus a lead-in of the largest delay. Common to all channels |
| Applied phase | Received sample at `t`: `exp(j(φ(t − τ_p) − φ(t)))`, with `τ_p` rounded to the grid (≤ 0.2 µs error) |
| Range correlation | `S_Δφ(f) = 4·sin²(π f τ)·S_φ(f)` [7], [8] |
| Doppler skirt | Bin `k`, relative to the carrier: `(PRF/N)·Σ_l 4 sin²(π f_l τ)·L(|f_l|)`, `f_l = k·PRF/N − l·PRF`, `|f_l| ≤ 1.25 MHz` |
| Mean sidelobe floor | `E|ΔAF|²/|AF(0)|² = σ²·Σw²/(Σw)²`, `σ² = σ²_amp + σ²_phase` (rad²) [10] |

### 3.6 Output schema

As scenario-001 §3.6, with a 16-channel cube, plus:

| Where | What |
| :--- | :--- |
| `metadata.json` `front_end` | `null` when bypassed; otherwise the resolved `[front_end]` table plus `n_taps`, `enbw_hz`, `passband_ripple_db` and `stopband_attenuation_db` as achieved, and `group_delay_samples` |
| `metadata.json` `bursts[].noise_power_w` | In the front-end path, `N0·fs_adc·Σh²`, not `k·T0·B·F` |
| `metadata.json` `impairments` | The resolved `[impairments]` table and its seed. The **drawn** values go to `products.h5`, not here |
| `products.h5` `/impairments/` | `channel_gain_linear` (n_rx,) complex128, `channel_delay_s` (n_rx,), `iq_k1_linear`, `iq_k2_linear` (n_rx,) complex128: the truth |
| `products.h5` `/calibration/` | `channel_gain_linear`, `channel_delay_s`, `iq_compensation_linear` (n_rx,), `pilot_snr_db`, and `method` as an attribute: the estimates |

---

## 4. Core Decisions

### 4.1 D20 — front-end chain order and sample-rate plumbing live in `core/frontend.py`

The chain order is fixed and is the order of §2's table: synthesise at the ADC rate → impairments
(analog side) → up-convert → noise → interferers → ADC → NCO → polyphase lowpass-decimate →
group-delay crop → re-centre. A `FrontEnd` frozen dataclass carries the rates and the filter
specification. `design_decimation_filter` returns the taps and their achieved ripple, attenuation
and ENBW, so every number in §3.2 is an exposed intermediate. `receive(cube_bb, front_end, *,
rng)` returns the output cube and a `FrontEndReport` (per-stage rates, noise bandwidths, the noise
power per sample).

Rates are derived, never repeated. `adc_sample_rate_hz` is computed from the burst's
`sample_rate_hz` and `decimation_factor`. A TOML that states both and disagrees is rejected.

The ideal receiver path is untouched:

1. With no `[front_end]` table, `pipelines/scenarios.py` calls today's generators with today's
   arguments, so 001–003 are bit-identical by construction (A1).
2. The front end calls the **same** `pulsed_baseband` with
   `dataclasses.replace(radar.receiver, sample_rate_hz=adc_sample_rate_hz)` and `rng=None`. It adds
   its own noise from a child generator (`rng.spawn(1)`), so the parent stream is never consumed
   differently. No second generator is written.
3. The output has the cube's exact shape and rate, so `core/dsp.py` consumes it unchanged.

Why digital IF is the default path, and what that teaches: an NCO is exact, so the
negative-frequency copy of a real IF signal is suppressed by the stopband (≥ 84.8 dB) and there is
no IQ image. Analog quadrature demodulation balances two mixers and two filters to perhaps
25–35 dB. A radar that samples at IF has moved the image problem into arithmetic.

### 4.2 D21 — impairments are explicit per-channel models; calibration returns parameters

`array/impairments.py` holds `ChannelResponse` (`gain_linear`, `delay_s`, both `(n_rx,)`),
`IqImbalance` (`gain_ratio_linear`, `phase_rad`, `(n_rx,)`) and `ChannelLoPhase`. Each has a
`draw_*(…, rng)` constructor from rms values and an `apply_*(data, model)` that works on the last
axis. Nothing is hidden in a generator: the cube a test sees is
`apply_iq_imbalance(apply_channel_response(clean, response), iq)`, and each step can be tested
alone.

`array/calibration.py` estimators return parameters:
`estimate_channel_response_from_pilot(pilot_data, tones_hz, …) -> ChannelCalibration`,
`estimate_iq_compensation_circular(data) -> NDArray` (one `w` per channel) and
`estimate_iq_compensation_pilot(…)`. Applying them is a separate, explicit call
(`correct_channel_response`, `correct_iq_imbalance`). A beamformer is handed the
`ChannelCalibration` and never calibrates silently. This is the same separation as 010's D23,
and it is what lets A10 score a calibration by its pattern without touching the data.

### 4.3 Phase noise lives in `core/signal.py`, as an argument

`lo_phase_noise_rad(n_samples_total, sample_rate_hz, offset_hz, ssb_dbc_per_hz, *, rng)` returns
a sampled phase process. `pulsed_baseband` and `fmcw_deramp_baseband` gain an optional keyword
`lo_phase_rad: NDArray[np.float64] | None = None`. When it is `None`, nothing changes. When given,
the generator applies `exp(j(φ(t − τ_p) − φ(t)))` per sample, so range correlation is not
modelled separately: it **emerges** from indexing one process at two times. A test can compare
the emergent skirt with the closed form (A14).

The common LO is one process shared by all channels. It therefore multiplies every channel by the
same factor and leaves the spatial pattern unchanged (A16).

### 4.4 What calibration can and cannot remove

| Error | Correctable? | By |
| :--- | :--- | :--- |
| Channel gain and phase at one frequency | Yes | Any pilot |
| Channel group delay | Yes, only with a pilot that spans the band | Multi-tone pilot (A10) |
| IQ imbalance, static | Yes | Pilot or blind |
| IQ imbalance, drifting | Only by re-estimating on current data | Blind circularity (A13) |
| Per-channel, per-pulse LO phase | Yes, the differential part | Pilot gate each PRI, relative to channel 0 (CPE, A17) |
| Common LO phase noise at the target's delay | **No** | It is a property of the oscillator and the range; specify a better oscillator |

### 4.5 Ownership of `array/calibration.py` and `array/impairments.py`

This slice creates both modules. It writes the per-channel half: gain, phase, group delay, IQ
imbalance and LO phase in `impairments.py`, and pilot and IQ estimators in `calibration.py`.
Scenario 010 adds mutual coupling, element-position error and the self-calibration estimators to
the same modules. The phase-quantisation and element-failure entries already in Part B's
`impairments.py` line are unaffected.

### 4.6 Out of scope

ADC quantisation and clipping, aperture jitter, analog anti-alias filter shape, frequency-selective
IQ imbalance (the multi-tap case of [6]), DC offset, and amplitude noise. Each is a later addition
to the same two modules, and none is needed for the three teaching points.

### 4.7 Amendments to `spec/structure.md`

> **Applied.** These amendments are folded into `spec/structure.md`; it is the authoritative
> tree and decision list. This section records what changed and why.

1. **Part B tree:** add `core/frontend.py` (real-IF front end, NCO/DDC, polyphase decimation and
   channeliser) and `array/calibration.py`. Widen the `array/impairments.py` comment to name the
   per-channel models, and the `core/signal.py` comment to "range-correlated LO phase noise".
2. **B.1 rows** for `core/frontend.py`, `array/impairments.py` (per-channel) and
   `array/calibration.py` (pilot, IQ).
3. **Decisions D20 and D21**, as §4.1–§4.2.

### 4.8 Workstream boundary

This scenario claims, and another workstream should not plan:

```
scenarios/scenario_009_{frontend_ddc,channel_mismatch,iq_phase_noise}.toml
spec/scenario-009-receiver-chain.md
src/radar_forge/core/frontend.py                         (new)
src/radar_forge/array/impairments.py                     (new; per-channel half)
src/radar_forge/array/calibration.py                     (new; pilot and IQ half)
src/radar_forge/core/signal.py        (lo_phase_noise_rad; optional lo_phase_rad keyword)
src/radar_forge/pipelines/scenarios.py ([front_end], [impairments], [calibration] tables)
tests/core/test_frontend.py, tests/array/test_impairments.py, tests/array/test_calibration.py
tests/pipelines/test_scenario_009.py
```

**Mutual coupling and array-position errors belong to scenario 010**, together with the
self-calibration estimators that need no injected reference, and the robust beamformers. This
slice assumes an ideal manifold apart from the per-channel receive chain.

`core/{dsp,windows,detection,ambiguity}.py` and `core/radar.py` are consumed, never edited. The
steering vector and `array/beamforming.py` are scenario 006's, and are consumed. Scenario-001's
TOMLs and tests are read, never edited, and `test_scenario_001.py` staying green is a precondition
for every commit.

---

## 5. Build Sequence

```
  1. core/frontend.py      design_decimation_filter -> taps, ripple, A_s, ENBW
             |             polyphase_decimate (upfirdn), channelise (M-path + IFFT)
             v
  2. core/frontend.py      FrontEnd, receive(): up-convert, noise, interferer, NCO, decimate, crop
             |
             v
  3. core/signal.py        lo_phase_noise_rad; optional lo_phase_rad in both generators
             |
             v
  4. array/impairments.py  ChannelResponse, IqImbalance, ChannelLoPhase: draw_*, apply_*
             |
             v
  5. array/calibration.py  pilot -> ChannelCalibration; IQ pilot and blind; CPE gate
             |
             v
  6. pipelines/scenarios.py  [front_end] / [impairments] / [calibration] present -> wired in
             |
             v
  7. scenarios/scenario_009_*.toml, then tests: bypass identity (A1) first
```

| Step | Module | Notes |
| :--- | :--- | :--- |
| 1 | `core/frontend.py` *(new)* | Pure functions on 1-D streams with a trailing channel axis. Test the polyphase identity and the filter numbers before anything else |
| 2 | `core/frontend.py` *(new)* | Consumes `pulsed_baseband` and `thermal_noise` unchanged |
| 3 | `core/signal.py` *(widen)* | Additive keyword, default `None`. No generator twin |
| 4 | `array/impairments.py` *(new)* | Per-channel only, per §4.5 |
| 5 | `array/calibration.py` *(new)* | Returns parameters (D21) |
| 6 | `pipelines/scenarios.py` *(edit)* | TOML order: `… [receiver_site] [receiver] [front_end] [impairments] [calibration]`, signal-path order as scenario-002 §3.5 |
| 7 | tests | A1 first: nothing else means anything until bypass is bit-identical |

Consumed unchanged: `core/dsp.py` (`matched_filter`, `range_doppler_map`, `range_bin_centers_m`,
`doppler_bin_centers_mps`), `core/windows.py` (`taper`), `core/waveforms.py` (`lfm_chirp`),
`core/constants.py`. Consumed from scenario 006: the steering vector and `array/beamforming.py`.

---

## 6. Acceptance Criteria

Unit criteria in `tests/core/test_frontend.py`, `tests/array/test_impairments.py` and
`tests/array/test_calibration.py`. Pipeline criteria in `tests/pipelines/test_scenario_009.py`,
marked `slow`, on a 2-frame window.

| # | Criterion | Bound | Justification | Variant |
| :--- | :--- | :--- | :--- | :--- |
| A1 | **Bypass bit-identity.** No `[front_end]`/`[impairments]`: cube equals today's generator output for the same seed; and both generators with `lo_phase_rad=None` | `np.array_equal` | Same code path, same RNG stream | 001 S1–S3, 002 |
| A2 | Filter as designed | ripple ≤ 0.1 dB p-p (0.0960); stopband ≥ 80 dB over 1.5–10 MHz (84.83); group delay 9 output samples | Closed form from the taps | C1 |
| A3 | Polyphase = filter-then-discard; channeliser channel 2 = DDC at 5 MHz after the stated phase factor | `rtol = 1e-10` | Same sums, reordered (round-off 8e-15 observed) | C1 |
| A4 | SNR gain, noise only | `10·log10(fs_adc/(2·ENBW))` = 6.372 dB, ±0.1 dB | 40 000 output samples: 1σ of the power estimate is 0.02 dB | C1 |
| A5 | Output noise power per sample | `N0·fs_adc·Σh²` = 1.8420e-14 W, 99 % χ² interval | Monte Carlo statistic (`testing.md` §10) | C1 |
| A6 | Interferer suppression | ≥ 81.82 dB (`A_s − 3.01`); predicted 84.88 dB, ±0.1 dB | Two stopband lines per real tone | C1 |
| A6b | **Baseline fails:** decimating without the filter | lines at ±0.500 MHz (one bin each), total power 3.01 dB *above* the input (±0.1 dB) | Both NCO lines alias into the passband, unattenuated | C1 |
| A7 | Front-end path vs ideal path, noiseless target | same range and Doppler bin; matched-filter peak within 0.5 dB (0.26 dB computed) | Chirp spectral tails beyond ±1 MHz are filtered. Omitting the `B/2` re-centring loses 4.33 dB: the bug this catches | C1 |
| A8 | Relative phase residual, 9-tone pilot | rms within the 99 % χ² interval of 0.181° (15 channels × 400 draws; 0.185° observed) | CRB, `testing.md` §10 | C2 |
| A9 | Relative delay residual | rms within the 99 % χ² interval of 0.975 ns (1.017 ns observed) | CRB; 400 effective draws, because channel 0 is common within a draw | C2 |
| A10 | Mean sidelobe power, uncalibrated | −19.81 dB ± 1 dB (−19.85 observed) | Mailloux floor [10] | C2 |
| A10b | **Baseline fails:** single-tone pilot at the band edge | median PSL ≥ ideal + 6 dB (−23.59 vs −34.73) | `2π f τ_n` uncorrected | C2 |
| A10c | **Current recovers:** 9-tone pilot at the band edge | 95th-percentile PSL ≤ ideal + 1.5 dB (−33.56) | CRB residual floor ≈ −55 dB adds ≤ 1.2 dB to a −34.7 dB sidelobe | C2 |
| A11 | IRR of the injected imbalance, by test tone | 28.200 dB, `rtol = 1e-9` | Closed form (§3.5) | C3 |
| A12 | **Digital DDC has no image:** the same tone through C1's chain | image ≤ −84.8 dB | Stopband, not matching, sets it | C1/C3 |
| A13 | Blind compensation, one CPI per channel | mean residual image over 16 channels ≤ −10·log10(2K) + 3 dB = −44.09 dB | Gamma(16) tail at 2× the mean: p = 6.6e-4 | C3 |
| A13b | **Baseline fails under drift:** pilot `w` from (0.5 dB, 3°) applied to (0.8 dB, 5°) | IRR 32.18 dB < 40 dB; blind on the same data ≥ 40 dB median (48.71) | Pilot is stale; blind uses current data | C3 |
| A14 | Doppler skirt vs folded range-correlation formula, bins 2, 10, 51, τ = 10 and 160 µs. Unit test on the slow-time sequence `exp(j(φ(t_m + τ) − φ(t_m)))` from `lo_phase_noise_rad`, rectangular window, target on a bin | ±0.75 dB | 300 runs: 1σ per bin 0.25 dB, bound is 3σ | C3 |
| A14b | Range-correlation null at `1/τ` | τ = 160 µs: bin 64 ≥ 10 dB below bin 51 (14.7 dB observed; second-order floor) | `sin²(π·6250·160e-6) = 0` at every alias | C3 |
| A15 | Range correlation suppresses close-in noise | τ = 10 µs, bin 2: ≥ 30 dB below independent oscillators (37.6 dB) | `4 sin²(πfτ)` = −38.2 dB at 195 Hz | C3 |
| A16 | Common LO phase noise leaves the beam pattern unchanged | pattern equals ideal, `rtol = 1e-12` | Common factor across channels | C3 |
| A17 | Per-channel LO phase, CPE-corrected | residual rms within the 99 % χ² interval of 0.405°; floor −54.13 dB (from −32.30 dB) | Pilot-gate CRB | C3 |
| A18 | Not asserted: absolute SNR against scenario 001 in the front-end path | — | Noise bandwidths differ by design (0.618 dB, §3.2) | — |

---

## References

.. [1] R. E. Crochiere and L. R. Rabiner, *Multirate Digital Signal Processing*, Prentice-Hall,
       1983. Polyphase decimation structures (chapter to confirm at implementation).
.. [2] F. J. Harris, *Multirate Signal Processing for Communication Systems*, Prentice Hall,
       2004, ISBN 0-13-146511-2. Polyphase channelisers (chapter to confirm at implementation).
.. [3] F. J. Harris, C. Dick and M. Rice, "Digital receivers and transmitters using polyphase
       filter banks for wireless communications," *IEEE Trans. Microwave Theory Tech.*, vol. 51,
       no. 4, pp. 1395–1412, Apr. 2003, doi:10.1109/TMTT.2003.809176.
.. [4] T. W. Parks and J. H. McClellan, "Chebyshev approximation for nonrecursive digital filters
       with linear phase," *IEEE Trans. Circuit Theory*, vol. 19, no. 2, pp. 189–194, 1972,
       doi:10.1109/TCT.1972.1083419.
.. [5] A. Tarighat, R. Bagheri and A. H. Sayed, "Compensation schemes and performance analysis of
       IQ imbalances in OFDM receivers," *IEEE Trans. Signal Process.*, vol. 53, no. 8,
       pp. 3257–3268, Aug. 2005, doi:10.1109/TSP.2005.851156.
.. [6] L. Anttila, M. Valkama and M. Renfors, "Circularity-based I/Q imbalance compensation in
       wideband direct-conversion receivers," *IEEE Trans. Veh. Technol.*, vol. 57, no. 4,
       pp. 2099–2113, Jul. 2008, doi:10.1109/TVT.2007.909269.
.. [7] R. S. Raven, "Requirements on master oscillators for coherent radar," *Proc. IEEE*,
       vol. 54, no. 2, pp. 237–243, 1966, doi:10.1109/PROC.1966.4636. Correction: vol. 55, no. 8,
       p. 1425, Aug. 1967.
.. [8] M. C. Budge and M. P. Burt, "Range correlation effects in radars," in *Record of the 1993
       IEEE National Radar Conference*, pp. 212–216, 1993, doi:10.1109/NRC.1993.270463.
.. [9] *IEEE Standard Definitions of Physical Quantities for Fundamental Frequency and Time
       Metrology — Random Instabilities*, IEEE Std 1139-2008, doi:10.1109/IEEESTD.2008.4797525.
.. [10] R. J. Mailloux, *Phased Array Antenna Handbook*, 2nd ed., Artech House, 2005. Random
        amplitude and phase errors (section to confirm at implementation).
.. [11] T. T. Taylor, "Design of line-source antennas for narrow beamwidth and low side lobes,"
        *Trans. IRE PGAP*, vol. 3, no. 1, pp. 16–28, 1955, doi:10.1109/TPGAP.1955.5720407.
