# 3. Waveforms and signals

This stage builds what a receiver records: a block of complex samples, one row per chirp. It
sits between propagation, which says how far away each target is and how fast it moves, and
range-Doppler processing, which turns the samples into a map (see [the chain](README.md#the-chain)).
It covers two modules: `radar_forge.core.waveforms`, which holds the chirp and the equations that
link its settings to what a receiver measures, and `radar_forge.core.signal`, which turns
propagation paths into a baseband cube with noise.

## What this stage does

Every waveform in the library is a **linear-FM chirp**: a tone whose frequency sweeps in a
straight line across a bandwidth $B$ in a time $T_c$. The sweep is centred on the carrier, so it
covers $f_c \pm B/2$. Its **range resolution** is $c/2B$: the bandwidth alone decides it, not the
chirp's length or power. That is why radars chirp: a long chirp carries a lot of energy and still
resolves like a short pulse.

The library has two receivers. An **FMCW** (frequency-modulated continuous-wave) radar transmits
chirp after chirp and **deramps** each echo: it mixes the echo with the chirp being transmitted.
An echo delayed by $\tau$ is a copy of the chirp shifted in time, so the mixer's output is a
steady tone, the **beat frequency**, proportional to range. A **pulsed** radar transmits short
chirps with gaps between them and records the echo itself, to be compressed by a matched filter
on [page 4](04-range-doppler-processing.md).

Both generators take a `PropagationPaths` from `propagation` (one entry per echo path, with its
range, Doppler and amplitude) and return a cube of shape `(n_pulses, n_samples)`: slow time
(one row per chirp) on axis 0, fast time (samples within a chirp) on axis 1. Both add thermal
noise if you give them a random generator.

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
| Sweep rate | $\alpha$ | $\alpha = B / T_c$ | `waveforms.sweep_rate_hzps`, `Transmitter.sweep_rate_hzps` | 2.0000e9 Hz/s |
| Range resolution | $\Delta R$ | $\Delta R = c / 2B$ | `waveforms.range_resolution_m`, `Radar.range_resolution_m` | 74.9481 m |
| Time-bandwidth product | — | $B T_c$ | `waveforms.lfm_chirp` (Notes) | 2000 |
| Round-trip delay | $\tau$ | $\tau = 2R / c$ | inside `signal.fmcw_deramp_baseband` | 117.15 µs at $R$ = 17560.7 m |
| Beat frequency | $f_b$ | $f_b = \alpha \tau = 2 \alpha R / c$ | `waveforms.beat_frequency_hz` | 234304.7 Hz |
| Range from beat | $R$ | $R = c f_b / 2\alpha$ | `waveforms.range_from_beat_frequency_m` | 17560.7 m |
| Samples per chirp | $N_s$ | $N_s = T_c f_s$ | `Radar.n_samples_per_chirp`, `Radar.n_samples_per_pri` | 1000 |
| Span of the FMCW range axis | $R_{\mathrm{span}}$ | $R_{\mathrm{span}} = c f_s / 2\alpha$ | `pipelines.scenarios.range_axis_m` | 74948.1 m |
| Motion during one chirp | $\delta R$ | $\delta R = \lvert v_r \rvert T_c$ | checked in `signal.fmcw_deramp_baseband` | 0.0281 m (warns above $\Delta R/4$ = 18.7370 m) |
| Noise density | $N_0$ | $N_0 = k T_0 F$ | `Receiver.noise_density_w_per_hz` | 7.9888e-21 W/Hz |
| Noise power per sample | $P_n$ | $P_n = N_0 f_s$ | `Receiver.noise_power_per_sample_w` | 7.9888e-15 W |

The S1 values are for frame 0, where the target is at 17560.7 m and opening at 28.08 m/s.

Symbols used only on this page:

- $\Delta R$, range resolution (m); $N_s$, fast-time samples per chirp; $R_{\mathrm{span}}$, the
  range an FMCW map covers before it wraps (m); $\delta R$, target motion during one chirp (m);
  $N_0$, noise power spectral density (W/Hz); $P_n$, noise power in one complex sample (W).

The four equations that matter most:

$$ \Delta R = \frac{c}{2B} $$

Two targets closer together than this fall in the same range bin. Only the bandwidth sets it.

$$ f_b = \alpha \tau = \frac{2 \alpha R}{c} $$

Deramping turns range into a frequency. The range FFT on page 4 measures $f_b$, and
`range_from_beat_frequency_m` turns it back into metres.

$$ s[m, n] = \sum_p A_p \exp\left(-j 2\pi \left(f_c + \tfrac{B}{2}\right) \tau_p[m]\right) \exp\left(j 2\pi \alpha \tau_p[m] t_n\right) $$

The deramped FMCW model. For chirp $m$ and sample $n$ at $t_n = n / f_s$, each path $p$ gives a
tone at its beat frequency (second factor) with a phase that changes from chirp to chirp as the
delay changes (first factor). Across chirps, that phase change is the Doppler shift.

$$ P_n = N_0 f_s = k T_0 F f_s $$

The noise in one sample fills the band the sampler sees, $f_s$ wide, not the swept band $B$.

## How it works, step by step

1. **Pick the waveform.** `waveforms.sweep_rate_hzps(bandwidth_hz, chirp_duration_s)` and
   `waveforms.range_resolution_m(bandwidth_hz)` give $\alpha$ and $\Delta R$. `Transmitter`
   exposes both as properties.
2. **Map range to beat frequency, and back.** `waveforms.beat_frequency_hz(range_m, bandwidth_hz,
   chirp_duration_s)` gives $f_b$ for a stationary target. `range_from_beat_frequency_m` is its
   inverse; it is how an FFT bin becomes a range.
3. **Build a chirp, when you need one.** `waveforms.lfm_chirp(bandwidth_hz, chirp_duration_s,
   sample_rate_hz, start_frequency_hz=..., up_sweep=...)` returns one chirp of shape
   `(n_samples,)`, with $n_\mathrm{samples} = \mathrm{round}(T_c f_s)$, sampled on $[0, T_c)$. The
   pulsed receiver's matched filter uses it with `start_frequency_hz=-bandwidth_hz / 2`, so the
   reference is centred on the carrier like the echo.
4. **Synthesise the FMCW cube.** `signal.fmcw_deramp_baseband(paths, radar, n_pulses, rng=...)`
   holds each path's range fixed during a chirp and moves it by $v_r / \mathrm{PRF}$ between
   chirps. It computes the delay, shape `(n_pulses, n_paths)`, then broadcasts it against the fast-time
   axis `(n_samples,)` to get the residual phase and beat phase, and sums the paths. The result
   is `(n_pulses, n_samples)` with `n_samples = radar.n_samples_per_pri`. Before that it checks
   the stop-and-hop bound (below) and warns if it fails.
5. **Or synthesise the pulsed cube.** `signal.pulsed_baseband(paths, radar, n_pulses, rng=...)`
   places a delayed copy of the chirp, swept $-B/2$ to $+B/2$, in each row of length
   `n_samples_per_pri`. The delay that places the pulse is the true delay **modulo the pulse
   repetition interval**, which is how a pulsed radar folds distant targets. The carrier phase
   uses the **true** delay, because folding is about when the receiver listens and does not change
   the phase the target imposed. The output is not compressed yet.
6. **Add noise.** Both generators call `signal.thermal_noise(shape, noise_power_w, rng=rng)` with
   `radar.receiver.noise_power_per_sample_w` when you pass `rng`. It draws complex Gaussian noise
   with variance $P_n/2$ in each of the real and imaginary parts. Without `rng` the cube is
   noiseless, which helps when you are chasing a geometry bug.

## Try it

This loads scenario 001 S1, takes its first frame, and checks the beat frequency against the
cube itself:

```python
import numpy as np

from radar_forge.core import waveforms
from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.pipelines.scenarios import iterate_frames, load_scenario, range_axis_m

scenario = load_scenario("scenarios/scenario_001_fmcw_low_prf.toml")
radar = scenario.bursts[0]
tx, rx = radar.transmitter, radar.receiver
bandwidth_hz, chirp_duration_s = tx.bandwidth_hz, tx.chirp_duration_s
frame = next(iterate_frames(scenario))  # frame 0: the truth, and one noisy cube
cube = frame.iq[0]  # (n_pulses, n_samples)

delay_s = 2.0 * frame.range_m / SPEED_OF_LIGHT_MPS
beat_hz = float(waveforms.beat_frequency_hz(frame.range_m, bandwidth_hz, chirp_duration_s))
back_m = float(waveforms.range_from_beat_frequency_m(beat_hz, bandwidth_hz, chirp_duration_s))
spectrum = np.abs(np.fft.fft(cube, axis=1)).sum(axis=0)  # range FFT, summed over chirps
peak_hz = spectrum.argmax() * rx.sample_rate_hz / cube.shape[1]
axis_m = range_axis_m(radar)
motion_m = abs(frame.radial_velocity_mps) * chirp_duration_s

print(f"cube {cube.shape}, sweep rate {tx.sweep_rate_hzps:.4e} Hz/s")
print(f"range resolution {radar.range_resolution_m:.4f} m")
print(f"range {frame.range_m:.1f} m, delay {delay_s * 1e6:.2f} us")
print(f"beat {beat_hz:.1f} Hz, back to range {back_m:.1f} m")
print(f"FFT peak of the cube {peak_hz:.1f} Hz")
print(f"map span {axis_m.size * (axis_m[1] - axis_m[0]):.1f} m")
print(f"unambiguous_range_m {radar.unambiguous_range_m:.1f} m")
print(f"motion in one chirp {motion_m:.4f} m, warning above {radar.range_resolution_m / 4:.4f} m")
print(f"noise per sample {rx.noise_power_per_sample_w:.4e} W")
```

The FFT bins are $f_s / N_s$ = 1 kHz apart, so the peak lands on the bin nearest 234.3 kHz:

```text
cube (256, 1000), sweep rate 2.0000e+09 Hz/s
range resolution 74.9481 m
range 17560.7 m, delay 117.15 us
beat 234304.7 Hz, back to range 17560.7 m
FFT peak of the cube 234000.0 Hz
map span 74948.1 m
unambiguous_range_m 37474.1 m
motion in one chirp 0.0281 m, warning above 18.7370 m
noise per sample 7.9888e-15 W
```

## Try it in a scenario

Read the companion first: [scenario 001, X-band](../scenarios/scenario-001-xband.md). Run S1:

```bash
uv run python scripts/run_scenario.py scenarios/scenario_001_fmcw_low_prf.toml --out out/s1 --frames 10
```

**The knob: `sample_rate_hz` in the `[[burst]]` table.** In an FMCW radar the sample rate, not the
PRF, sets how far the map reaches, because the highest beat frequency the receiver can tell apart
is set by $f_s$. Copy the TOML, change `sample_rate_hz = 1.0e6` to `2.0e5`, and run the copy
with `--out out/s1_slow`.

What changes:

- The range axis of `rd_00000.png` shrinks from 0–75 km to 0–15 km ($R_{\mathrm{span}}$ goes from
  74948.1 m to 14989.6 m), and the row length in `metadata.json` (`n_samples`) drops from 1000
  to 200.
- The target's beat, 234.3 kHz, is now above $f_s$ = 200 kHz, so it wraps. In `rd_00000.png`
  the target sits near 2.57 km instead of 17.56 km: 17560.7 − 14989.6 = 2571.1 m. `truth.csv`
  is unchanged. The run prints the last frame's peak; with `--frames 3` that is
  `peak [  2.62 km ...]` against `[ 17.61 km ...]` before (truth 17616.9 m, which wraps to
  2627.2 m).
- The noise per sample, `noise_power_w` in `metadata.json`, falls fivefold, from 7.9888e-15 W to
  1.5978e-15 W. The range resolution stays 74.9481 m: it depends on $B$ only.

Want to go further? The "differentiable simulation" direction in
[AI for radar](../../interns/ai-for-radar.md#directions) builds a PyTorch copy of this FMCW IQ
model, checked against `fmcw_deramp_baseband`.

## Things that look like bugs and aren't

**The FMCW sweep runs downwards.** In a deramp, the range and Doppler signs are locked
together: both come from the same delay $\tau$. With an up sweep, a closing target would sit at a
positive beat frequency but a negative Doppler, breaking the library's "closing is positive" rule.
A down sweep, from $f_c + B/2$ to $f_c - B/2$, gives both the right sign. `signal`'s module
docstring explains this, and it is why `lfm_chirp` has an `up_sweep` flag.

**The residual phase uses $f_c + B/2$, yet Doppler converts with $\lambda = c/f_c$.**
$f_c + B/2$ is the frequency transmitted at the start of each chirp. Read at a fixed range bin
after the range FFT, though, the phase from chirp to chirp advances at the mid-sweep frequency
$f_c$. `fmcw_deramp_baseband`'s docstring derives this.

**S1 samples at 1 MHz, half its 2 MHz bandwidth, and nothing aliases.** After the deramp, the
receiver samples the beat tone, not the chirp. The beat is at most a few hundred kilohertz for
S1's ranges. A pulsed receiver records the chirp itself, so `Radar` rejects a pulsed radar with
$f_s < B$.

**The map reaches 74.9 km, but `unambiguous_range_m` says 37.5 km.** The property uses the
real-sampling limit $f_s/2$; the complex samples tell apart a band $f_s$ wide, so the map spans
twice that. `range_axis_m`'s Notes explain the difference; correcting the property is a
follow-up (`spec/scenario-003-tracking.md` §14.11).

## Limits

- **Stop-and-hop.** Range is held constant during each chirp and moves only between chirps. This
  is good while a target moves much less than a range bin per chirp; `fmcw_deramp_baseband` warns
  when it moves more than a quarter of a bin. `pulsed_baseband` does not check.
- **Constant velocity over a CPI.** Acceleration during the dwell is ignored.
- **No eclipsing.** A pulsed echo that arrives while the transmitter is on is received anyway.
  At S2's 25 % duty cycle that is a quarter of all ranges.
- **No sawtooth delay limit.** A real FMCW echo must arrive within the chirp, $\tau < T_c$; the
  simulator does not model this (`range_axis_m`'s Notes).
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
