# 4. Range-Doppler processing

This stage turns the cube of complex samples from [page 3](03-waveforms-and-signals.md) into a
**range-Doppler map**: a picture of echo strength against range and range rate. It sits in the
middle of [the chain](README.md#the-chain), between the samples and detection. It covers one
module, `dsp` (windows and what they cost, pulse compression, the two FFTs, MTI, folding and
undoing it with two PRFs), and `pipeline.range_doppler_product`, which runs it on a scenario's
cube.

## What this stage does

The cube is `(n_pulses, n_samples)`. Along a row (fast time) a target is a beat tone whose
frequency grows with range. Down a column (slow time) the same target's phase turns a little more
from chirp to chirp, at a rate set by how fast its range changes. An FFT along each axis turns
both into frequencies, so one target becomes one bright cell.

```
  IQ cube (n_pulses, n_samples)
     |  window along fast time (optional)   dsp.window, apply_window
     v
  range_fft, axis 1        -> range profiles (n_pulses, n_range_bins), NOT shifted
     |  window along slow time (optional)
     v
  doppler_fft, axis 0      -> map (n_doppler_bins, n_range_bins), zero Doppler in the middle
```

Both axes are **bistatic**. Range is $R_b = R_T + R_R$, the length of the route, twice the slant
range for a monostatic radar; rate is its time derivative $\dot R_b$, **positive receding**. The
Doppler FFT measures $f_d = -\dot R_b / \lambda$ (written once, in `propagation.doppler_shift_hz`),
so the rate axis falls as the bin index rises. A monostatic figure or CSV halves both and calls
them range and radial velocity (`plots.display_axes`); nothing before the display does.

A pulsed radar has one extra step first: a **matched filter** compresses each long pulse into a
short peak. Two more tools sit beside the chain. An **MTI** (moving-target indicator) filter
subtracts chirps to remove echoes from things that don't move. And when the PRF is too low for
the target's rate, the rate **folds** (wraps round the map); two bursts at different PRFs can
undo that.

## Quantities and equations

| Quantity | Symbol | Equation | In the code | Scenario 001 S1 |
| :-- | :-- | :-- | :-- | :-- |
| Rectangular sidelobe | — | first sidelobe of a sinc | `dsp.window` | −13 dB |
| Coherent gain of a window | $G_c$ | $G_c = \frac{1}{N}\sum_n w[n]$ | `dsp.coherent_gain_linear` | 1.000 (no window); raw Hann 0.498, raw 60 dB Taylor 0.470 |
| Processing loss of a window | $L$ | $L = -10\log_{10}\frac{(\sum_n w[n])^2}{N\sum_n w[n]^2}$ | `dsp.processing_loss_db` | 0 dB (the pipeline applies no window); Hann 1.78 dB, 60 dB Taylor 1.89 dB over 256 chirps |
| Pulse compression gain | — | $B T_c$ | `dsp.matched_filter` | — (S1 is FMCW; S2's pulse has $B T_c = 20$, 13.0 dB) |
| Range bins | $N_s$ | $N_s = f_s / \mathrm{PRF}$ | `Radar.n_samples_per_pri` | 1000 |
| Range bin spacing | $\Delta R_b$ | $\Delta R_b = \frac{c f_s T_c}{B N_s}$ | `dsp.range_bin_centers_m`, `RangeDopplerProduct.range_bin_m` | 149.8962 m (74.9481 m slant) |
| Range span | $R_{\mathrm{span}}$ | $\frac{c f_s T_c}{B}$ | `RangeDopplerProduct.range_period_m`, `Radar.unambiguous_bistatic_range_m` | 149 896.2 m (74 948.1 m slant) |
| Doppler bins | $N$ | one per chirp | `Scenario.n_pulses` | 256 |
| Rate resolution | $\Delta v$ | $\Delta v = \frac{\lambda}{N T_{\mathrm{PRI}}}$ | `dsp.doppler_bin_centers_mps`, `RangeDopplerProduct.rate_bin_mps` | 0.119496 m/s (0.059748 m/s radial) |
| Unambiguous rate | $v_{ua}$ | $v_{ua} = \lambda\,\mathrm{PRF}/2$; the map spans $(-v_{ua}, +v_{ua}]$ | `RangeDopplerProduct.rate_period_mps` $/2$ | 15.2955 m/s (7.6478 m/s radial) |
| Folded rate | — | $\dot R_b$ wrapped into $(-v_{ua}, +v_{ua}]$ | `dsp.fold_velocity_mps` | +160 m/s (receding at 80 m/s) reads as +7.0447 m/s |
| Integration gain of the two FFTs | — | $N_s N$ | `dsp.range_doppler_map` | 256 000, 54.08 dB |
| MTI blind rate | — | $\lambda\,\mathrm{PRF} = 2 v_{ua}$ | `dsp.mti_filter`, `Radar.unambiguous_bistatic_range_rate_mps` | 30.5911 m/s |
| Dual-PRF $v_{ua}$ (S3) | $v_{ua,a}$, $v_{ua,b}$ | $\lambda\,\mathrm{PRF}/2$ per burst | as above | S3: 76.4777 / 91.7732 m/s |
| Dual-PRF rate reach (S3) | — | $q\, v_{ua,a}$ for PRFs in ratio $p:q$ | `dsp.unfold_doppler_dual_prf` | S3: ±458.866 m/s; +160 m/s unfolds to +160.0000 |
| Dual-PRF range reach (S3) | $S_a$, $S_b$ | map spans repeat after $p S_a = q S_b$ | `dsp.unfold_range_dual_prf` | S3: spans 119 917.0 / 99 930.8 m, repeat after ≈600 km |

The dual-PRF rows use S3 (`config/scenario_001_fmcw_dual_prf.toml`), because S1 has one burst.
Every range and rate is bistatic; a monostatic display shows half of each.

Symbols used only on this page:

| Symbol | Meaning | Unit | In the code |
| :-- | :-- | :-- | :-- |
| $w[n]$ | window weights | — | `window(...)` |
| $G_c$ | coherent gain | linear | `coherent_gain_linear` |
| $L$ | processing loss | dB | `processing_loss_db` |
| $N_s$ | fast-time samples, and so range bins, per chirp | — | `n_samples_per_pri` |
| $T_{\mathrm{PRI}}$ | time between chirps, $1/\mathrm{PRF}$ | s | `pulse_repetition_interval_s` |
| $R_b$, $\dot R_b$ | bistatic range $R_T + R_R$, and its rate, positive receding | m, m/s | `bistatic_range_m`, `bistatic_range_rate_mps` |
| $\Delta R_b$, $\Delta v$ | range bin spacing, rate bin spacing | m, m/s | `range_bin_m`, `rate_bin_mps` (negative: the axis falls) |
| $v_{ua}$ | unambiguous rate, half the rate span | m/s | `rate_period_mps / 2` |
| $S$ | span of one FMCW map's range axis, $N_s \Delta R_b$ | m | `RangeDopplerProduct.range_period_m` |

The bin centres come from the deramp relation $R_b = c f_b / \alpha$. Bin $k$ of the range FFT
holds beat frequency $f_b = k f_s / N_s$, so

$$R_{b,k} = \frac{c\, k\, f_s\, T_c}{B N_s}.$$

In S1, $f_s T_c = N_s$, so the bins are $c/B$ apart: 149.8962 m of route, which is the range
resolution $c/2B$, 74.9481 m, on a monostatic display.

The chirps are $T_{\mathrm{PRI}}$ apart, so the Doppler bins are $1/(N T_{\mathrm{PRI}})$ apart,
and with $\dot R_b = -\lambda f_d$ the rate bins are

$$\Delta v = \frac{\lambda}{N T_{\mathrm{PRI}}}$$

apart, over $(-v_{ua}, +v_{ua}]$ with $v_{ua} = \lambda / (2 T_{\mathrm{PRI}})$. Only a longer
dwell (more chirps) makes $\Delta v$ smaller. Zero-padding does not.

A window's cost is its processing loss, the SNR it throws away by down-weighting the edges:

$$L = -10\log_{10}\frac{\left(\sum_n w[n]\right)^2}{N\sum_n w[n]^2}.$$

## How it works, step by step

1. **Choose a window (optional).** `dsp.window(name, n)` returns `(n,)` weights from SciPy,
   peaking at 1; `normalise=True` divides them by their mean, so a windowed peak reads in the same
   units. `"taylor"` and `"chebyshev"` need `sidelobe_level_db`, a negative level such as
   `-60.0`; the fixed shapes refuse it. `dsp.apply_window(cube, name, axis=...)` multiplies along
   one axis by broadcasting. Without a window the FFT's sinc sidelobes sit 13 dB below a strong
   target and can hide a weak one beside it. `coherent_gain_linear` and `processing_loss_db` say
   what a window costs; `dsp.WINDOW_NAMES` lists the shapes.
2. **Pulsed only: compress.** `dsp.matched_filter(samples, reference, axis=-1)` correlates with
   the transmitted chirp, `lfm_chirp(B, T_c, f_s)`. The output is longer than the input, and zero
   lag sits at index `n_reference - 1`. It uses `scipy.signal.fftconvolve`, which pads; a
   hand-written FFT product would wrap energy round to zero range. `range_doppler_product`
   compresses the rows as one stream, because an echo that runs off the end of a row carries on
   in the next, then cuts it back into rows from `n_reference - 1`, so zero delay lands at bin 0.
3. **Range FFT.** `dsp.range_fft(cube, weights=..., axis=1)` turns fast time into range. Input
   `(n_pulses, n_samples)`, output `(n_pulses, n_range_bins)`, **not** `fftshift`-ed: bin 0 is
   zero delay.
4. **Doppler FFT.** `dsp.doppler_fft(profiles, weights=..., axis=0)` turns slow time into Doppler.
   Output `(n_doppler_bins, n_range_bins)`, `fftshift`-ed: zero Doppler at bin `n // 2`. The
   index rises with $f_d$, so it falls with rate: receding targets sit at indices below `n // 2`,
   closing targets at indices above it, and a figure draws row 0, $+v_{ua}$, at the top.
   `dsp.range_doppler_map(cube, fast_time_weights=..., slow_time_weights=...)` runs steps 3 and
   4 in one call.
5. **Label the axes.** `dsp.range_bin_centers_m(n_bins, bandwidth_hz, chirp_duration_s,
   sample_rate_hz)` gives `(n_range_bins,)` bistatic ranges, not halved;
   `dsp.doppler_bin_centers_mps(n_bins, pulse_repetition_interval_s, wavelength_m)` gives
   `(n_doppler_bins,)` rates, from $+v_{ua}$ down. `pipeline.map_axes(radar, n_pulses)` returns
   both for a burst (a pulsed range bin is one sample of delay, $c/f_s$).
   `pipeline.range_doppler_product(cube, radar)` does steps 2 to 5 and returns a
   `RangeDopplerProduct` holding `rd_map`, `range_axis_m` and `rate_axis_mps`, with each axis's
   span as `range_period_m` and `rate_period_mps`.
6. **MTI (optional).** `dsp.mti_filter(cube, n_pulses=2, axis=0)` differences neighbouring
   chirps: $x[m+1] - x[m]$, or $x[m+2] - 2x[m+1] + x[m]$ for three. `axis` has no default.
   Stationary echoes cancel. The slow-time axis gets `n_pulses - 1` shorter, and targets at
   multiples of the blind rate $\lambda\,\mathrm{PRF}$ cancel too.
7. **Fold and unfold.** `dsp.fold_velocity_mps(rate_mps, span_mps)` predicts where a target of
   true rate `rate_mps` appears: in $(-\mathrm{span}/2, +\mathrm{span}/2]$, closed at the top as
   the map is. Range folds
   with `dsp.fold_into_period(range_m, span_m, lower=0.0)`. `unfold_doppler_dual_prf` and
   `unfold_range_dual_prf` take one reading from each of two bursts and try every candidate
   $a + kP_a$ inside the bounds you give, keeping the one that best matches the second reading.
   Both call `unfold_dual_period`, which returns `nan` when no candidate is within `tolerance`,
   and the residual so you can judge the match.

## Try it

This forms S1's first map, checks its peak against the folded truth, prices three windows over
S1's 256 chirps, and unfolds a target receding at 80 m/s (+160 m/s of bistatic rate) with S3's
two bursts:

```python
import numpy as np

from radar_forge import dsp, plots
from radar_forge.pipeline import frames, range_doppler_product
from radar_forge.scenario import load_scenario

s1 = load_scenario("config/scenario_001_fmcw_low_prf.toml")
frame = next(frames(s1, n_frames=1))
product = range_doppler_product(frame.cubes[0], s1.radar)  # FMCW: range FFT, then Doppler FFT
range_m, rate_mps = product.range_axis_m, product.rate_axis_mps
gain_db = 10 * np.log10(product.rd_map.size)
show = plots.display_axes(s1.radar)
print(f"map {product.rd_map.shape}, gain of the two FFTs {gain_db:.2f} dB")
print(f"range bins {product.range_bin_m:.4f} m apart; bin N/2 at {range_m[500]:.1f} m")
print(f"rate step {product.rate_bin_mps:+.6f} m/s, {rate_mps[0]:+.4f} down to {rate_mps[-1]:+.4f}")
print(f"shown as {show.range_name} and {show.rate_name}, times {show.scale}")

truth_m = frame.truth.bistatic_range_m[0]
truth_mps = frame.truth.bistatic_range_rate_mps[0]
folded_mps = dsp.fold_velocity_mps(truth_mps, product.rate_period_mps)
print(f"truth {truth_m:.1f} m, {truth_mps:+.4f} m/s, folded {folded_mps:+.4f} m/s")
rate_bin, range_bin = np.unravel_index(np.abs(product.rd_map).argmax(), product.rd_map.shape)
print(f"peak ({rate_bin}, {range_bin}): {range_m[range_bin]:.1f} m, {rate_mps[rate_bin]:+.4f} m/s")
for name, level in (("rectangular", None), ("hann", None), ("taylor", -60.0)):
    w = dsp.window(name, s1.n_pulses[0], sidelobe_level_db=level)
    gain, loss = dsp.coherent_gain_linear(w), dsp.processing_loss_db(w)
    print(f"{name:<11} gain {gain:.3f}  loss {loss:.2f} dB")

a, b = load_scenario("config/scenario_001_fmcw_dual_prf.toml").radars
span_a = a.unambiguous_bistatic_range_rate_mps
span_b = b.unambiguous_bistatic_range_rate_mps
read_a, read_b = dsp.fold_velocity_mps(160.0, span_a), dsp.fold_velocity_mps(160.0, span_b)
true_mps, _ = dsp.unfold_doppler_dual_prf(
    read_a, read_b, span_a, span_b, max_velocity_mps=458.0, tolerance_mps=0.7
)
print(f"S3 spans {span_a:.4f} / {span_b:.4f} m/s; +160 m/s reads {read_a:+.4f} / {read_b:+.4f}")
print(f"unfolds to {true_mps:+.4f} m/s")
```

It prints:

```text
map (256, 1000), gain of the two FFTs 54.08 dB
range bins 149.8962 m apart; bin N/2 at 74948.1 m
rate step -0.119496 m/s, +15.2955 down to -15.1760
shown as range_m and range_rate_mps, times 0.5
truth 35121.4 m, +56.1629 m/s, folded -5.0192 m/s
peak (170, 234): 35075.7 m, -5.0188 m/s
rectangular gain 1.000  loss -0.00 dB
hann        gain 0.498  loss 1.78 dB
taylor      gain 0.470  loss 1.89 dB
S3 spans 152.9553 / 183.5464 m/s; +160 m/s reads +7.0447 / -23.5464
unfolds to +160.0000 m/s
```

The target recedes at +56.16 m/s, which folds to −5.02 m/s: bin 170, 42 past the zero-rate bin
128. The peak is where folding says, within half a bin on each axis. +160 m/s reads +7.0447 m/s
on both S1 and S3's burst A. That is a coincidence of the numbers: burst A's PRF is five times
S1's, so its fold period is too. The tolerance, 0.7 m/s, is about half a Doppler bin of the
coarser burst (B's bins are 1.4340 m/s apart).

## Try it in a scenario

Use [scenario 001](../../config/scenario_001_fmcw_low_prf.toml), S1:

```bash
uv run python scripts/run_scenario.py config/scenario_001_fmcw_low_prf.toml --frames 10
```

**Knob:** in a copy of the TOML in `config/`, set `n_pulses = 64` (was 256) in `[[burst]]`, and
run it with `--out runs/s1_64`. Then compare `rd_00009.png` in the two run folders, and what
`[rd]` prints.

- The rate bins get four times wider, 0.1195 → 0.4780 m/s, which the figures show halved,
  0.0597 → 0.2390 m/s. The map's rows look coarser, and `sigma_rate_mps` in `detections.csv`, a
  bin over √12, grows from 0.0172 to 0.0690 m/s.
- $v_{ua}$ does not change: it is set by the PRF, not the dwell. Both figures still run
  ±7.65 m/s, and the peak still sits on the dashed crosshair of the folded truth (at frame 9 the
  target recedes at 39.32 m/s, which folds to −6.57 m/s).
- `[rd]` prints frame 0's peak and median cell. The peak stands 51.71 dB above the median with
  256 chirps (−36.75 against −88.47 dBW) and 42.10 dB with 64 (−52.31 against −94.40 dBW). That
  is 6.02 dB from a quarter of the Doppler integration gain, and most of the other 3.6 dB is
  straddle loss: frame 0's folded rate, −5.019 m/s, lies 10.50 of the wider bins from zero,
  halfway between two of them.
- The copy ends `1 check(s) failed`: `[geometry]` predicts 45.99 dB of SNR, and `[verify]`
  holds 52.01 dB, the figure for 256 chirps. The check is right to fail; a TOML's numbers belong
  to its own settings.

For dual-PRF unfolding, run `config/scenario_001_fmcw_dual_prf.toml` the same way: it writes
one map per burst, `rd_burst0_*.png` and `rd_burst1_*.png`, each folded at its own $v_{ua}$, and
`detections.csv` gives each pair a `pair_id` and its unfolded rate.

Intern task: [Can an unrolled solver beat a taper?](../../interns/unrolled-sparse-rd.md)

## Things that look like bugs and aren't

**The rate axis runs backwards.** `doppler_bin_centers_mps` falls with index, from exactly
$+v_{ua}$ at bin 0 to $-v_{ua} + \Delta v$ at the last bin. The FFT index rises with Doppler, and
Doppler is minus the rate over $\lambda$. So the interval is $(-v_{ua}, +v_{ua}]$, closed at the
top, and `fold_velocity_mps` folds into the same one.

**The target's rate on the S1 map is nowhere near the truth.** S1's $v_{ua}$ is only 15.30 m/s
(7.65 m/s radial on the figure), so an aircraft at tens of m/s folds several times. The dashed
crosshair marks where folding puts it; the red cross marks the true rate, off the plot when it
folds.

**`Radar.unambiguous_bistatic_range_rate_mps` is twice $v_{ua}$.** It is the span the rate axis
covers before it repeats, $\lambda\,\mathrm{PRF}$. `fold_velocity_mps` and the unfolding
functions take the same span, as the example does.

**The rectangular window costs "−0.00 dB", and raw windows have a coherent gain below 1.** The
first is negative zero, $-10\log_{10} 1$. The second is SciPy's scaling: its shapes peak at 1,
so their mean is less, 0.498 for Hann and 0.470 for a 60 dB Taylor. `normalise=True` brings the
gain to 1.000 and leaves the loss where it was, so read the loss, not the gain, for what a
window costs.

**S3's burst B has range bins 149.8213 m apart, not 149.8962 m.** Its repetition interval is
666.67 samples, rounded to 667, and the FFT bins are $f_s/667$ apart. The labels are right for the
FFT that was taken.

## Limits

- The pipeline (`range_doppler_product`) applies **no window**. Windowing is there to call, not
  switched on.
- The reference chirp's transform is recomputed on every `matched_filter` call; nothing caches it.
- MTI has only the two- and three-pulse cancellers. No staggered PRI, so blind rates stay, and no
  Doppler filter bank.
- No range migration correction. Over S1's 256 ms CPI the target's route changes by at most
  0.19 of a bin in the 120-frame window.
- Dual-PRF unfolding assumes one target per range. Two targets give ghost pairings, and the
  answer degrades sharply once noise nears half a bin. A target outside the bounds you pass can
  come back as a confident wrong answer: on S3, +600 m/s comes back as −317.73 m/s.
- FMCW range unfolding reaches past $c T_c$ of route, which a real sawtooth radar cannot; the
  simulator does not model that limit.
- The axes are DFT outputs and circular. `RangeDopplerProduct.range_at_index_m`,
  `rate_at_index_mps` and `index_of` handle a fractional index across the wrap; `np.interp`
  would not.

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
- N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech, 2005, ch. 3 (the range-sum ellipse), §6.1
  (target Doppler).
- G. W. Stimson, *Introduction to Airborne Radar*, 2nd ed., SciTech, 1998, ch. 12, 21, 25 and 27
  (resolving range and Doppler ambiguities, the choice of PRF, blind zones).
