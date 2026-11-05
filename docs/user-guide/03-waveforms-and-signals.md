# 3. Waveforms and signals

This stage builds what a receiver records: a block of complex samples, one row per chirp. It
sits between propagation, which says how long each echo's route is and how fast it changes, and
range-Doppler processing, which turns the samples into a map (see [the chain](README.md#the-chain)).
It covers one module, `radar_forge.signal`: the chirp, the equations that link its settings to
what a receiver measures, and the generators that turn propagation paths into a baseband cube
with noise.

## What this stage does

Every waveform in the library is a **linear-FM chirp**: a tone whose frequency sweeps in a
straight line across a bandwidth $B$ in a time $T_c$. The sweep is centred on the carrier, so it
covers $f_c \pm B/2$. Its **range resolution** is $c/2B$: the bandwidth alone decides it, not the
chirp's length or power. That is why radars chirp: a long chirp carries a lot of energy and still
resolves like a short pulse.

The library has two receivers. An **FMCW** (frequency-modulated continuous-wave) radar transmits
chirp after chirp and **deramps** each echo: it mixes the echo with the chirp being transmitted.
An echo delayed by $\tau$ is a copy of the chirp shifted in time, so the mixer's output is a
steady tone, the **beat frequency**, proportional to the delay. A **pulsed** radar transmits short
chirps with gaps between them and records the echo itself, to be compressed by a matched filter
on [page 4](04-range-doppler-processing.md).

Every range here is a **bistatic range** $R_b = R_T + R_R$, the length of the whole route from
transmitter to target to receiver, and every rate is its time derivative $\dot R_b$, positive
receding. For a monostatic radar $R_b = 2R$ and $\dot R_b$ is twice the radial velocity, so the
numbers below are twice the slant values you may expect; only a display halves them
([page 4](04-range-doppler-processing.md)).

Both generators take a `PropagationPaths` from `propagation` (one entry per route, with its
delay, Doppler and amplitude) and return a cube of shape `(n_pulses, n_samples)`: slow time
(one row per chirp) on axis 0, fast time (samples within a chirp) on axis 1. Both add thermal
noise if you give them a random generator. `pipeline.synthesise` builds the paths and picks the
generator for you; `pipeline.frames` does that for every frame of a scenario.

```
 frequency
 f_c + B/2 |\        \           transmitted down sweep (solid)
           | \  .     \  .       echo, delayed by tau (dotted)
           |  \   .    \   .
 f_c - B/2 |   \    .   \    .   mixing the two leaves a steady
           +----+-------+------> time     beat tone f_b = alpha * tau
           0   T_c    2 T_c
```

## Quantities and equations

| Quantity | Symbol | Equation | In the code | Scenario 001 S1 |
| :-- | :-- | :-- | :-- | :-- |
| Band swept | — | $f_c \pm B/2$ | `Transmitter.carrier_frequency_hz`, `bandwidth_hz` | 9.799 to 9.801 GHz |
| Sweep rate | $\alpha$ | $\alpha = B / T_c$ | `signal.sweep_rate_hzps`, `Transmitter.sweep_rate_hzps` | 2.0000e9 Hz/s |
| Range resolution | $\Delta R$ | $\Delta R = c / 2B$ | `signal.range_resolution_m`, `Radar.range_resolution_m` | 74.9481 m |
| The same, in bistatic range | $\Delta R_b$ | $\Delta R_b = c / B$ | the bin spacing of `dsp.range_bin_centers_m` | 149.8962 m |
| Time-bandwidth product | — | $B T_c$ | `signal.lfm_chirp` | 2000 |
| Bistatic range | $R_b$ | $R_b = R_T + R_R$, $2R$ monostatic | `TargetGeometry.bistatic_range_m` | 35121.4 m at $R$ = 17560.7 m |
| Delay | $\tau$ | $\tau = R_b / c$ | `propagation.propagation_delay_s` | 117.15 µs |
| Beat frequency | $f_b$ | $f_b = \alpha \tau = \alpha R_b / c$ | `signal.beat_frequency_hz` | 234304.7 Hz |
| Bistatic range from beat | $R_b$ | $R_b = c f_b / \alpha$ | `signal.bistatic_range_from_beat_frequency_m` | 35121.4 m |
| Samples per chirp | $N_s$ | $N_s = T_c f_s$ | `Radar.n_samples_per_chirp`, `Radar.n_samples_per_pri` | 1000 |
| Span of the FMCW range axis | $R_{\mathrm{span}}$ | $R_{\mathrm{span}} = c f_s / \alpha$ | `pipeline.map_axes`, `RangeDopplerProduct.range_period_m` | 149896.2 m (74948.1 m slant) |
| Motion during one chirp | $\delta R_b$ | $\delta R_b = \lvert \dot R_b \rvert T_c$ | checked in both generators | 0.0562 m (warns above $\Delta R_b/4$ = 37.4741 m) |
| Noise density | $N_0$ | $N_0 = k T_0 F$ | `Receiver.noise_density_w_per_hz` | 7.9888e-21 W/Hz |
| Noise power per sample | $P_n$ | $P_n = N_0 f_s$ | `Receiver.noise_power_per_sample_w` | 7.9888e-15 W |

The S1 values are for frame 0, where the target is at 17560.7 m and opening at 28.08 m/s: a
bistatic range of 35121.4 m and a rate of +56.16 m/s.

Symbols used only on this page:

- $R_b$, bistatic range $R_T + R_R$ (m); $\dot R_b$, its rate, positive receding (m/s);
  $\Delta R$, range resolution (m), and $\Delta R_b$ the same in bistatic range (m); $N_s$,
  fast-time samples per chirp; $R_{\mathrm{span}}$, the bistatic range an FMCW map covers before
  it wraps (m); $\delta R_b$, route change during one chirp (m); $N_0$, noise power spectral
  density (W/Hz); $P_n$, noise power in one complex sample (W).

The four equations that matter most:

$$ \Delta R = \frac{c}{2B} $$

Two targets closer together than this in slant range fall in the same range bin. On the bistatic
axis the same bin is $c/B$ wide, because a metre of slant range adds two metres of route. Only
the bandwidth sets it.

$$ f_b = \alpha \tau = \frac{\alpha R_b}{c} $$

Deramping turns route length into a frequency. There is no factor of 2: for a monostatic radar
$R_b = 2R$ gives the textbook $2 \alpha R / c$. The range FFT on page 4 measures $f_b$, and
`bistatic_range_from_beat_frequency_m` turns it back into metres of route.

$$ s[m, n] = \sum_p A_p \exp\left(-j 2\pi \left(f_c + \tfrac{B}{2}\right) \tau_p[m]\right) \exp\left(j 2\pi \alpha \tau_p[m] t_n\right) $$

The deramped FMCW model. For chirp $m$ and sample $n$ at $t_n = n / f_s$, each path $p$ gives a
tone at its beat frequency (second factor) with a phase that changes from chirp to chirp as the
delay $\tau_p[m] = R_{b,p}[m] / c$ changes (first factor). Across chirps, that phase change is the
Doppler shift, $f_d = -\dot R_b / \lambda$: positive for a closing target.

$$ P_n = N_0 f_s = k T_0 F f_s $$

The noise in one sample fills the band the sampler sees, $f_s$ wide, not the swept band $B$.

## How it works, step by step

1. **Pick the waveform.** `signal.sweep_rate_hzps(bandwidth_hz, chirp_duration_s)` and
   `signal.range_resolution_m(bandwidth_hz)` give $\alpha$ and $\Delta R$. `Transmitter` and
   `Radar` expose both as properties.
2. **Map bistatic range to beat frequency, and back.** `signal.beat_frequency_hz(bistatic_range_m,
   bandwidth_hz, chirp_duration_s)` gives $f_b$ for a stationary target.
   `bistatic_range_from_beat_frequency_m` is its inverse; it is how an FFT bin becomes a range.
   For a monostatic slant range, pass $2R$ in, or halve what comes out.
3. **Build a chirp, when you need one.** `signal.lfm_chirp(bandwidth_hz, chirp_duration_s,
   sample_rate_hz, start_frequency_hz=..., sweep_direction="up")` returns one chirp of shape
   `(n_samples,)`, with $n_\mathrm{samples} = \mathrm{round}(T_c f_s)$, sampled on $[0, T_c)$. By
   default it is centred on zero, starting at $-B/2$ (or $+B/2$ with `sweep_direction="down"`),
   like the echo; that default is the pulsed receiver's matched-filter reference.
   `signal.ambiguity_function(waveform, sample_rate_hz)` returns $\lvert\chi(\tau, f_d)\rvert$
   and its two axes: what the chirp can resolve in delay and Doppler.
4. **Synthesise the FMCW cube.** `signal.fmcw_deramp_baseband(paths, radar, n_pulses, rng=...)`
   holds each path's bistatic range fixed during a chirp and moves it by
   $\dot R_b\, T_{\mathrm{PRI}}$ between chirps (`PropagationPaths.bistatic_range_at_m`). It
   computes the delay, shape `(n_pulses, n_paths)`, then broadcasts it against the fast-time axis
   `(n_samples,)` to get the residual phase and beat phase, and sums the paths a chunk at a time
   (`max_bytes`), so memory does not grow with the number of paths. The result is
   `(n_pulses, n_samples)` with `n_samples = radar.n_samples_per_pri`. Before that it checks the
   stop-and-hop bound (below) and warns if it fails.
5. **Or synthesise the pulsed cube.** `signal.pulsed_baseband(paths, radar, n_pulses, rng=...)`
   places a delayed copy of `lfm_chirp(B, T_c, f_s)` in each row of length `n_samples_per_pri`.
   The delay that places the pulse is the true delay **modulo the pulse repetition interval**,
   which is how a pulsed radar folds distant targets. The carrier phase uses the **true** delay,
   because folding is about when the receiver listens and does not change the phase the target
   imposed. An echo that runs past the end of its row goes on at the start of the next, because
   the receiver keeps listening; the last row's overrun is lost. The output is not compressed yet.
6. **Add noise.** Both generators call `signal.thermal_noise(shape, noise_power_w, rng=rng)` with
   `radar.receiver.noise_power_per_sample_w`. It draws complex Gaussian noise with variance
   $P_n/2$ in each of the real and imaginary parts. `rng` has no default: pass `rng=None` for a
   noiseless cube, which helps when you are chasing a geometry bug.

## Try it

This loads scenario 001 S1, builds frame 0's cube from its truth, and checks the beat frequency
against the cube itself:

```python
import numpy as np

from radar_forge import signal
from radar_forge.pipeline import map_axes
from radar_forge.propagation import line_of_sight_paths, propagation_delay_s
from radar_forge.scenario import load_scenario, truth_geometry

scenario = load_scenario("config/scenario_001_fmcw_low_prf.toml")
radar, n_pulses = scenario.radar, scenario.n_pulses[0]
tx, rx = radar.transmitter, radar.receiver
bandwidth_hz, chirp_duration_s = tx.bandwidth_hz, tx.chirp_duration_s
truth = truth_geometry(scenario)[0]  # frame 0
paths = line_of_sight_paths(radar, truth, [target.rcs_m2 for target in scenario.targets])
rng = np.random.default_rng(scenario.seed)
cube = signal.fmcw_deramp_baseband(paths, radar, n_pulses, rng=rng)  # (n_pulses, n_samples)

bistatic_range_m = float(truth.bistatic_range_m[0])  # R_T + R_R: twice the slant range
rate_mps = float(truth.bistatic_range_rate_mps[0])  # its time derivative, positive receding
delay_s = float(propagation_delay_s(bistatic_range_m))
beat_hz = float(signal.beat_frequency_hz(bistatic_range_m, bandwidth_hz, chirp_duration_s))
back_m = float(signal.bistatic_range_from_beat_frequency_m(beat_hz, bandwidth_hz, chirp_duration_s))
spectrum = np.abs(np.fft.fft(cube, axis=1)).sum(axis=0)  # range FFT, summed over chirps
peak_hz = spectrum.argmax() * rx.sample_rate_hz / cube.shape[1]
range_axis_m, _ = map_axes(radar, n_pulses)
motion_m = abs(rate_mps) * chirp_duration_s
bin_m = range_axis_m[1]

print(f"cube {cube.shape}, sweep rate {tx.sweep_rate_hzps:.4e} Hz/s")
print(f"range resolution {radar.range_resolution_m:.4f} m, bistatic range bin {bin_m:.4f} m")
print(f"bistatic range {bistatic_range_m:.1f} m (slant {truth.range_tx_m[0]:.1f} m)")
print(f"rate {rate_mps:+.2f} m/s, Doppler {paths.doppler_hz[0]:+.1f} Hz")
print(f"delay {delay_s * 1e6:.2f} us, beat {beat_hz:.1f} Hz, back to {back_m:.1f} m")
print(f"FFT peak of the cube {peak_hz:.1f} Hz")
print(f"map span {range_axis_m.size * bin_m:.1f} m")
print(f"unambiguous_bistatic_range_m {radar.unambiguous_bistatic_range_m:.1f} m")
print(f"motion in one chirp {motion_m:.4f} m, warning above {bin_m / 4:.4f} m")
print(f"noise per sample {rx.noise_power_per_sample_w:.4e} W")
```

The FFT bins are $f_s / N_s$ = 1 kHz apart, so the peak lands on the bin nearest 234.3 kHz. The
target recedes, so its rate is positive and its Doppler negative:

```text
cube (256, 1000), sweep rate 2.0000e+09 Hz/s
range resolution 74.9481 m, bistatic range bin 149.8962 m
bistatic range 35121.4 m (slant 17560.7 m)
rate +56.16 m/s, Doppler -1835.9 Hz
delay 117.15 us, beat 234304.7 Hz, back to 35121.4 m
FFT peak of the cube 234000.0 Hz
map span 149896.2 m
unambiguous_bistatic_range_m 74948.1 m
motion in one chirp 0.0562 m, warning above 37.4741 m
noise per sample 7.9888e-15 W
```

## Try it in a scenario

Read the scenario file first: [scenario 001, X-band](../../config/scenario_001_fmcw_low_prf.toml).
Run S1 through the stepper:

```bash
uv run python scripts/run_scenario.py config/scenario_001_fmcw_low_prf.toml --frames 3
```

It prints each stage's quantities and wall time, checks them against the TOML's `[verify]`
table, and writes `runs/scenario_001_fmcw_low_prf/`: `truth.csv`, `rd_00000.png` to
`rd_00002.png` and `detections.csv`. The figures and CSVs show slant range and radial velocity,
half the bistatic values; everything printed is bistatic.

**The knob: `sample_rate_hz` in the `[[burst]]` table.** In an FMCW radar the sample rate, not the
PRF, sets how far the map reaches, because the highest beat frequency the receiver can tell apart
is set by $f_s$. Copy the TOML into `config/` (its `trajectory_path` is relative to the file),
change `sample_rate_hz = 1.0e6` to `2.0e5`, and run the copy with `--out runs/s1_slow`.

What changes:

- The range axis of `rd_00000.png` shrinks from 0–75 km to 0–15 km: $R_{\mathrm{span}}$ goes from
  149896.2 m to 29979.2 m of bistatic range. `[geometry]` prints
  `unambiguous_bistatic_range_m = 14989.6`, half the span (see below).
- The target's beat, 234.3 kHz, is now above $f_s$ = 200 kHz, so it wraps. `[rd]` prints
  `rd_peak_range_index = 34` instead of 234, and in `rd_00000.png` and `detections.csv` the
  target sits near 2.57 km instead of 17.56 km: 35121.4 − 29979.2 = 5142.2 m of route, 2571.1 m
  of slant range. `truth.csv` is unchanged.
- `noise_power_per_sample_dbw` falls by 6.99 dB, from −140.975 to −147.965 dBW (7.9888e-15 W to
  1.5978e-15 W), yet `predicted_snr_db` stays 52.0115 and every check passes: each chirp holds a
  fifth of the samples, each with a fifth of the noise. The range resolution stays 74.9481 m: it
  depends on $B$ only.

Want to go further? The "differentiable simulation" direction in
[AI for radar](../../interns/ai-for-radar.md#directions) builds a PyTorch copy of this FMCW IQ
model, checked against `fmcw_deramp_baseband`.

## Things that look like bugs and aren't

**The FMCW sweep runs downwards.** In a deramp, the range and Doppler signs are locked
together: both come from the same delay $\tau$. With an up sweep, a target at a positive beat
frequency would show its Doppler with the wrong sign, so a closing target would look as if it
receded. A down sweep, from $f_c + B/2$ to $f_c - B/2$, gives both the right sign.
`fmcw_deramp_baseband`'s docstring says so, and it is why `lfm_chirp` takes `sweep_direction`.

**`beat_frequency_hz` has no factor of 2.** Textbooks write $f_b = 2\alpha R / c$ for a slant
range $R$. The library passes the route length $R_b$, which is $2R$ for a monostatic radar, so the
2 is already inside it.

**The residual phase uses $f_c + B/2$, yet Doppler converts with $\lambda = c/f_c$.**
$f_c + B/2$ is the frequency transmitted at the start of each chirp. Read at a fixed range bin
after the range FFT, though, the beat tone adds back $\pi B$ times the change in delay, so the
phase from chirp to chirp advances at the mid-sweep frequency $f_c$.

**S1 samples at 1 MHz, half its 2 MHz bandwidth, and nothing aliases.** After the deramp, the
receiver samples the beat tone, not the chirp. The beat is at most a few hundred kilohertz for
S1's ranges. A pulsed receiver records the chirp itself, so `Radar` rejects a pulsed radar with
$f_s < B$.

**The map reaches 149.9 km of bistatic range, but `unambiguous_bistatic_range_m` says 74.9 km.**
The property uses the real-sampling limit $f_s/2$; the complex samples tell apart a band $f_s$
wide, so the map spans twice that. Correcting the property is a follow-up.


## Limits

- **Stop-and-hop.** Range is held constant during each chirp and moves only between chirps. This
  is good while a target moves much less than a range bin per chirp; both generators warn when a
  route changes by more than a quarter of a bin, $\Delta R_b / 4$.
- **Constant velocity over a CPI.** Acceleration during the dwell is ignored unless a path
  carries a `doppler_rate_hzps`; line-of-sight paths do not.
- **No eclipsing.** A pulsed echo that arrives while the transmitter is on is received anyway.
  At S2's 25 % duty cycle that is a quarter of all ranges.
- **No sawtooth delay limit.** A real FMCW echo must arrive within the chirp, $\tau < T_c$; the
  simulator does not model this.
- **Magnitude only.** The generators use $\lvert A_p \rvert$ and rebuild the phase from the
  delay, so a backend cannot add a phase of its own, such as a complex reflection coefficient
  (`PropagationPaths`'s docstring).
- **Two waveforms only**: linear-FM FMCW and linear-FM pulses. No phase codes, no non-linear FM,
  no stepped frequency.

## References

- M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014, §2.6.2
  (the stop-and-hop approximation), §4.6.1 (linear frequency modulation), §4.6.5 (stretch
  processing: the FMCW deramp).
- A. Harrison, *Introduction to Radar Using Python and MATLAB*, Artech House, 2020, ch. 8 (pulse
  compression and linear FM).
- G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014, ch. 3, "Mathematics of
  FMCW Radar" (FMCW deramp and the beat-frequency relation).
- N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech, 2005, ch. 3 (the range-sum ellipse), §6.1
  (target Doppler).
