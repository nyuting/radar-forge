# 4. Range-Doppler processing

This stage turns the cube of complex samples from [page 3](03-waveforms-and-signals.md) into a
**range-Doppler map**: a picture of echo strength against range and radial velocity. It sits in
the middle of [the chain](README.md#the-chain), between the samples and detection. It covers three
modules: `windows` (tapers and what they cost), `dsp` (pulse compression, the two FFTs and MTI)
and `ambiguity_resolution` (folding, and undoing it with two PRFs).

## What this stage does

The cube is `(n_pulses, n_samples)`. Along a row (fast time) a target is a beat tone whose
frequency grows with range. Down a column (slow time) the same target's phase turns a little more
from chirp to chirp, at a rate set by its velocity. An FFT along each axis turns both into
frequencies, so one target becomes one bright cell.

```
  IQ cube (n_pulses, n_samples)
     |  taper along fast time (optional)   windows.taper, apply_taper
     v
  range_fft, axis 1        -> range profiles (n_pulses, n_range_bins), NOT shifted
     |  taper along slow time (optional)
     v
  doppler_fft, axis 0      -> map (n_doppler_bins, n_range_bins), zero velocity in the middle
```

A pulsed radar has one extra step first: a **matched filter** compresses each long pulse into a
short peak. Two more tools sit beside the chain. An **MTI** (moving-target indicator) filter
subtracts chirps to remove echoes from things that don't move. And when the PRF is too low for
the target's speed, the velocity **folds** (wraps round the map); two bursts at different PRFs can
undo that.

## Quantities and equations

| Quantity | Symbol | Equation | In the code | Scenario 001 S1 |
| :-- | :-- | :-- | :-- | :-- |
| Rectangular sidelobe | — | first sidelobe of a sinc | `windows` module docstring | −13 dB |
| Coherent gain of a taper | $G_c$ | $G_c = \frac{1}{N}\sum_n w[n]$ | `windows.coherent_gain_linear` | 1.000 (no taper); raw Hann 0.498 |
| Processing loss of a taper | $L$ | $L = -10\log_{10}\frac{(\sum_n w[n])^2}{N\sum_n w[n]^2}$ | `windows.processing_loss_db` | 0 dB (the pipeline applies no taper); Hann 1.78 dB, 60 dB Taylor 1.89 dB over 256 chirps |
| Pulse compression gain | — | $B T_c$ | `dsp.matched_filter` | — (S1 is FMCW; S2's pulse has $B T_c = 20$, 13.0 dB) |
| Range bins | $N_s$ | $N_s = f_s / \mathrm{PRF}$ | `Radar.n_samples_per_pri` | 1000 |
| Range bin spacing | $\Delta R$ | $\Delta R = \frac{c f_s T_c}{2 B N_s}$ | `dsp.range_bin_centers_m`, `range_axis_m` | 74.9481 m |
| Range at bin $N_s/2$ | $R_{ua}$ | $R_{ua} = \frac{c f_s T_c}{4B}$ | `Radar.unambiguous_range_m` | 37 474.1 m (the map spans twice this) |
| Doppler bins | $N$ | one per chirp | `Scenario.n_pulses` | 256 |
| Velocity resolution | $\Delta v$ | $\Delta v = \frac{\lambda}{2 N T_{\mathrm{PRI}}}$ | `dsp.doppler_bin_centers_mps` | 0.059748 m/s |
| Unambiguous velocity | $v_{ua}$ | $v_{ua} = \lambda\,\mathrm{PRF}/4$ | `Radar.unambiguous_velocity_mps` | ±7.6478 m/s |
| Folded velocity | — | $v$ wrapped into $[-v_{ua}, v_{ua})$ | `fold_velocity_mps` | 80 m/s reads as 3.5223 m/s |
| Integration gain of the two FFTs | — | $N_s N$ | `dsp.range_doppler_map` | 256 000, 54.08 dB |
| MTI blind speed | — | $\lambda\,\mathrm{PRF}/2 = 2 v_{ua}$ | `dsp.mti_filter` | 15.2955 m/s |
| Dual-PRF $v_{ua}$ (S3) | $v_{ua,a}$, $v_{ua,b}$ | $\lambda\,\mathrm{PRF}/4$ per burst | `Radar.unambiguous_velocity_mps` | S3: ±38.2388 / ±45.8866 m/s |
| Dual-PRF velocity reach (S3) | — | $q\, v_{ua,a}$ for PRFs in ratio $p:q$ | `unfold_doppler_dual_prf` | S3: ±229.433 m/s; 80 m/s unfolds to 80.0000 |
| Dual-PRF range reach (S3) | $S_a$, $S_b$ | map spans repeat after $q S_a = p S_b$ | `unfold_range_dual_prf` | S3: spans 59 958.5 / 49 965.4 m, repeat after ≈300 km |

The dual-PRF rows use S3 (`scenarios/scenario_001_fmcw_dual_prf.toml`), because S1 has one burst.

Symbols used only on this page:

| Symbol | Meaning | Unit | In the code |
| :-- | :-- | :-- | :-- |
| $w[n]$ | taper weights | — | `taper(...)` |
| $G_c$ | coherent gain | linear | `coherent_gain_linear` |
| $L$ | processing loss | dB | `processing_loss_db` |
| $N_s$ | fast-time samples, and so range bins, per chirp | — | `n_samples_per_pri` |
| $T_{\mathrm{PRI}}$ | time between chirps, $1/\mathrm{PRF}$ | s | `pulse_repetition_interval_s` |
| $\Delta R$, $\Delta v$ | range bin spacing, velocity bin spacing | m, m/s | — |
| $R_{ua}$, $v_{ua}$ | unambiguous range and velocity | m, m/s | `unambiguous_range_m`, `unambiguous_velocity_mps` |
| $S$ | span of one FMCW map's range axis, $N_s \Delta R$ | m | `RangeDopplerProduct.range_period_m` |

The bin centres come from the deramp relation $R = c f_b / 2\alpha$. Bin $k$ of the range FFT
holds beat frequency $f_b = k f_s / N_s$, so

$$R_k = \frac{c\, k\, f_s\, T_c}{2 B N_s}.$$

In S1, $f_s T_c = N_s$, so the bin spacing is exactly the range resolution $c/2B$.

The Doppler bins are $T_{\mathrm{PRI}}$ apart in slow time, so after the FFT they are spaced

$$\Delta v = \frac{\lambda}{2 N T_{\mathrm{PRI}}},$$

and only a longer dwell (more chirps) makes $\Delta v$ smaller. Zero-padding does not.

A taper's cost is its processing loss, the SNR it throws away by down-weighting the edges:

$$L = -10\log_{10}\frac{\left(\sum_n w[n]\right)^2}{N\sum_n w[n]^2}.$$

## How it works, step by step

1. **Choose a taper (optional).** `windows.taper(name, n)` returns `(n,)` weights, scaled to mean
   1 so a tapered peak reads in the same units. `apply_taper(cube, w, axis=...)` multiplies them
   along one axis by broadcasting. Without a taper the FFT's sinc sidelobes sit 13 dB below a
   strong target and can hide a weak one beside it. `coherent_gain_linear` and
   `processing_loss_db` say what the taper cost; the `taper` Notes table lists each shape.
2. **Pulsed only: compress.** `dsp.matched_filter(cube, reference, axis=-1)` correlates each row
   with the transmitted chirp. The output is longer than the input, and zero lag sits at index
   `n_reference - 1`, so `form_range_doppler_map` trims from there. It uses
   `scipy.signal.fftconvolve`, which pads; a hand-written FFT product would wrap energy round to
   zero range.
3. **Range FFT.** `dsp.range_fft(cube, window=..., axis=1)` turns fast time into range. Input
   `(n_pulses, n_samples)`, output `(n_pulses, n_range_bins)`, **not** `fftshift`-ed: bin 0 is
   zero range.
4. **Doppler FFT.** `dsp.doppler_fft(profiles, window=..., axis=0)` turns slow time into velocity.
   Output `(n_doppler_bins, n_range_bins)`, `fftshift`-ed: zero velocity at bin `n // 2`, closing
   targets above it. `dsp.range_doppler_map` runs steps 3 and 4 in one call.
5. **Label the axes.** `dsp.range_bin_centers_m(n_bins, bandwidth_hz, chirp_duration_s,
   sample_rate_hz)` gives `(n_range_bins,)` ranges; `dsp.doppler_bin_centers_mps(n_bins,
   pulse_repetition_interval_s, wavelength_m)` gives `(n_doppler_bins,)` velocities. In the
   pipeline, `form_range_doppler_map(cube, burst)` does steps 2 to 5 and returns a
   `RangeDopplerProduct` holding `rd_map`, `range_axis_m` and `velocity_axis_mps`.
6. **MTI (optional).** `dsp.mti_filter(cube, n_pulses=2, axis=0)` subtracts neighbouring chirps
   (`[1, -1]`, or `[1, -2, 1]` for three). Stationary echoes cancel. The slow-time axis gets
   `n_pulses - 1` shorter, and targets at multiples of the blind speed cancel too.
7. **Fold and unfold.** `fold_velocity_mps(v, v_ua)` predicts where a target of true velocity `v`
   appears; it is `fold_into_period` with period $2v_{ua}$ centred on zero (range uses
   `lower=0`). `unfold_doppler_dual_prf` and `unfold_range_dual_prf` take one reading from each
   of two bursts and try every candidate $a + kP_a$ inside the bounds you give, keeping the one
   that best matches the second reading. Both call `unfold_dual_period`, which returns `nan`
   when no candidate is within `tolerance`, and the residual so you can judge the match.

## Try it

This loads S1 and S3, labels S1's axes, prices three tapers over S1's 256 chirps, and unfolds an
80 m/s target with S3's two bursts:

```python
import numpy as np

from radar_forge.core import dsp, windows
from radar_forge.core.ambiguity_resolution import fold_velocity_mps, unfold_doppler_dual_prf
from radar_forge.pipelines.scenarios import load_scenario, range_axis_m

s1 = load_scenario("scenarios/scenario_001_fmcw_low_prf.toml")
radar, n_pulses = s1.bursts[0], s1.n_pulses[0]
range_m = range_axis_m(radar)
pri_s = radar.transmitter.pulse_repetition_interval_s
velocity_mps = dsp.doppler_bin_centers_mps(n_pulses, pri_s, radar.wavelength_m)
v_ua = radar.unambiguous_velocity_mps
print(f"{range_m.size} range bins, {range_m[1]:.4f} m apart; bin N/2 at {range_m[500]:.1f} m")
print(f"{velocity_mps.size} Doppler bins, {velocity_mps[1] - velocity_mps[0]:.6f} m/s apart")
print(f"v_ua {v_ua:.4f} m/s; 80 m/s reads as {fold_velocity_mps(80.0, v_ua):.4f} m/s")
print(f"gain of the two FFTs {10 * np.log10(range_m.size * n_pulses):.2f} dB")
for name in ("rectangular", "hann", "taylor"):
    w = windows.taper(name, n_pulses, normalize=False)
    gain, loss = windows.coherent_gain_linear(w), windows.processing_loss_db(w)
    print(f"{name:<11} gain {gain:.3f}  loss {loss:.2f} dB")

a, b = load_scenario("scenarios/scenario_001_fmcw_dual_prf.toml").bursts
v_a, v_b = a.unambiguous_velocity_mps, b.unambiguous_velocity_mps
read_a, read_b = fold_velocity_mps(80.0, v_a), fold_velocity_mps(80.0, v_b)
true_mps, _ = unfold_doppler_dual_prf(
    read_a, read_b, v_a, v_b, max_velocity_mps=229.0, tolerance_mps=0.35
)
print(f"S3 v_ua {v_a:.4f} / {v_b:.4f} m/s; 80 m/s reads {read_a:.4f} / {read_b:.4f}")
print(f"unfolds to {true_mps:.4f} m/s")
```

It prints:

```text
1000 range bins, 74.9481 m apart; bin N/2 at 37474.1 m
256 Doppler bins, 0.059748 m/s apart
v_ua 7.6478 m/s; 80 m/s reads as 3.5223 m/s
gain of the two FFTs 54.08 dB
rectangular gain 1.000  loss -0.00 dB
hann        gain 0.498  loss 1.78 dB
taylor      gain 1.000  loss 1.89 dB
S3 v_ua 38.2388 / 45.8866 m/s; 80 m/s reads 3.5223 / -11.7732
unfolds to 80.0000 m/s
```

80 m/s reads 3.5223 m/s on both S1 and S3's burst A. That is a coincidence of the numbers: burst
A's PRF is five times S1's, so its fold period is too. The tolerance, 0.35 m/s, is about half a
Doppler bin of the coarser burst (B's bins are 0.717 m/s apart).

## Try it in a scenario

Use [scenario 001](../scenarios/scenario-001-xband.md), S1:

```bash
uv run python scripts/run_scenario.py scenarios/scenario_001_fmcw_low_prf.toml --out out/s1 --frames 10 --no-iq --no-movie
```

**Knob:** in a copy of the TOML, set `n_pulses = 64` (was 256) in `[[burst]]`, and run it into
another folder. Then compare `rd_00009.png` in the two folders, and the last progress line.

- The velocity bins get four times wider, 0.0597 → 0.2390 m/s. The map's rows look coarser.
- $v_{ua}$ does not change: it is set by the PRF, not the dwell. Both maps still run ±7.65 m/s,
  and the target still sits at the folded truth marker (truth −39.3 m/s at frame 9, folded to
  +6.57 m/s).
- The peak reads +6.57 m/s with 256 chirps and +6.45 m/s with 64. The folded truth falls almost
  midway between two of the wider bins.
- The noise floor rises towards the peak: the peak stands about 53 dB above the median cell with
  256 chirps and about 43 dB with 64. That is 6 dB from a quarter of the Doppler integration gain,
  and about 4 dB of straddle loss from falling between bins.

For dual-PRF unfolding, run `scenarios/scenario_001_fmcw_dual_prf.toml` the same way: it writes
one map per burst, `rd_burst0_*.png` and `rd_burst1_*.png`, each folded at its own $v_{ua}$.

Intern task: [Can an unrolled solver beat a taper?](../../interns/unrolled-sparse-rd.md)

## Things that look like bugs and aren't

**The target's velocity on the S1 map is nowhere near the truth.** S1's $v_{ua}$ is only
±7.65 m/s, so an aircraft at tens of m/s folds several times. The red cross marks where folding
puts it; the triangle on the edge is the true velocity, off the scale.

**The S1 map runs to 74.9 km, but `Radar.unambiguous_range_m` says 37.5 km.** The samples are
complex, so the range FFT tells apart beat frequencies over all of $[0, f_s)$, not $[0, f_s/2)$.
The property takes the real-sampling limit, which is bin $N_s/2$. `range_axis_m`'s Notes explain
this, and why neither is a real radar's limit ($cT_c/2$ for a sawtooth sweep).

**The rectangular taper costs "−0.00 dB", and a raw Taylor taper has a coherent gain of 1.000.**
The first is negative zero, $-10\log_{10} 1$. The second is how SciPy's Taylor shape is built:
its weights already average 1. Its processing loss is still 1.89 dB, so read the loss, not the
gain, for what a taper costs.

**S3's burst B has range bins 74.9107 m apart, not 74.9481 m.** Its repetition interval is
666.67 samples, rounded to 667, and the FFT bins are $f_s/667$ apart. The labels are right for the
FFT that was taken.

## Limits

- The pipeline (`form_range_doppler_map`) applies **no taper**. Tapering is there to call, not
  switched on.
- The reference chirp's transform is recomputed on every `matched_filter` call; nothing caches it.
- MTI has only the two- and three-pulse cancellers. No staggered PRI, so blind speeds stay, and no
  Doppler filter bank.
- No range migration correction. S1's short CPI keeps it to a quarter of a bin (see the
  `scenarios` module docstring).
- Dual-PRF unfolding assumes one target per range. Two targets give ghost pairings, and the
  answer degrades sharply once noise nears half a bin. A target outside the bounds you pass can
  come back as a confident wrong answer: on S3, 300 m/s comes back as −158.87 m/s.
- FMCW range unfolding reaches past $cT_c/2$, which a real sawtooth radar cannot; the simulator
  does not model that limit (`spec/scenario-003-tracking.md` §14.11).
- The axes are DFT outputs and circular. `RangeDopplerProduct.range_at_index_m` and
  `velocity_at_index_mps` handle a fractional index across the wrap; `np.interp` would not.

## References

- F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier
  transform," *Proc. IEEE*, vol. 66, no. 1, pp. 51–83, Jan. 1978.
- T. T. Taylor, "Design of line-source antennas for narrow beamwidth and low side lobes," *IRE
  Trans. Antennas Propag.*, vol. AP-3, no. 1, pp. 16–28, Jan. 1955.
- M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014, §3.1.3
  (Doppler and range ambiguities), §4.6 (FM pulse compression), §5.2.1 (pulse cancellers), §5.3
  (pulse Doppler processing), §5.5.4 (ambiguity resolution).
- A. Harrison, *Introduction to Radar Using Python and MATLAB*, Artech House, 2020, ch. 8 (pulse
  compression).
- G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014, ch. 3, "Mathematics of
  FMCW Radar".
- G. W. Stimson, *Introduction to Airborne Radar*, 2nd ed., SciTech, 1998, ch. 12, 21, 25 and 27
  (resolving range and Doppler ambiguities, the choice of PRF, blind zones).
