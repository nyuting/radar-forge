# 5. Detection

A range-Doppler map holds a power in every cell. Detection decides which cells hold a target.
This page covers `radar_forge.detection`: constant-false-alarm-rate (CFAR) thresholds, the
false-alarm arithmetic behind them, and clustering the cells that cross into one detection per
target. It also covers `pipeline.frame_plots`, which runs all of it on a scenario's maps and
hands the tracker one **plot** per detection. It sits between
[range-Doppler processing](04-range-doppler-processing.md) and [tracking](06-tracking.md) in
[the chain](README.md#the-chain).

## What this stage does

Every cell is compared with a threshold. A cell above it is a **detection**; a noise cell above
it is a **false alarm**. The probability that a noise-only cell crosses is the probability of
false alarm, $P_{fa}$, and it is the number you design to.

A **fixed** threshold cannot hold $P_{fa}$. The noise floor changes with range, the antenna
pattern and the weather. A level that gives one false alarm an hour in one place gives thousands
in another. CFAR instead estimates the floor around each **cell under test** (CUT) from its
neighbours, the **training cells**, and multiplies that estimate by a constant factor
$\alpha_{\mathrm{cfar}}$. Scale the noise and the threshold scales with it, so $P_{fa}$ stays
where you put it.

Between the CUT and the training cells sit **guard cells**, left out of the estimate. A target
spreads over a few cells; without guards its own energy would raise its own threshold.

```
[ train ][ guard ][ CUT ][ guard ][ train ]
 N_train  N_guard    1    N_guard  N_train
```

One target lights several neighbouring cells. The last step groups touching cells and reports
one detection per group, which is what a tracker expects. Detection works in cells and leaves
the map's axes as they are: bistatic range $R_b = R_T + R_R$ (twice the slant range for a
monostatic radar) and its rate $\dot R_b$, positive receding.

## Quantities and equations

| Quantity | Symbol | Equation | In the code | Scenario 003 |
| :-- | :-- | :-- | :-- | :-- |
| Detected (square-law) power of a cell | $P$ | $P = \lvert x \rvert^2$ | `np.abs(rd_map) ** 2` | — |
| Training and guard cells per side | $N_{\mathrm{train}}$, $N_{\mathrm{guard}}$ | — | `n_train`, `n_guard` | 16, 4 |
| Reference cells | $M$ | $M = 2 N_{\mathrm{train}}$ | `cfar_reference_cell_count` | 32 |
| CA noise estimate | $\hat{P}_n$ | $\hat{P}_n = \frac{1}{M} \sum_{i=1}^{M} P_i$ | inside `cfar_detect` | — |
| Threshold | $T$ | $T = \alpha_{\mathrm{cfar}} \hat{P}_n$ | `cfar_detect` returns `(detected, threshold_w)` | — |
| CA threshold factor | $\alpha_{\mathrm{cfar}}$ | $\alpha_{\mathrm{cfar}} = M \left( P_{fa}^{-1/M} - 1 \right)$ | `cfar_threshold_factor` | 13.856 (11.417 dB) |
| CA false-alarm probability | $P_{fa}$ | $P_{fa} = \left( 1 + \alpha_{\mathrm{cfar}} / M \right)^{-M}$ | `cfar_probability_of_false_alarm` | 1e-5 |
| Reference cells a clipped cell keeps, $i$ cells from an end | $M_c$ | $N_{\mathrm{train}} + \min(N_{\mathrm{train}}, \max(0, i - N_{\mathrm{guard}}))$ | inside `cfar_detect` | 16 for $i \le 4$, 22 at $i = 10$; not used: range wraps |
| CA factor at a clipped cell | $\alpha_{\mathrm{cfar}}$ | as above with $M = M_c$ | inside `cfar_detect` | 16.856 (12.268 dB) at $M_c$ = 16, 15.127 (11.798 dB) at 22 |
| Factor if the noise power were known | — | $-\ln P_{fa}$ | — | 11.513 (10.612 dB) |
| GO and SO factors, on a half-window mean | $\alpha_{\mathrm{cfar}}$ | Gandhi and Kassam's closed forms | `cfar_threshold_factor(variant=)` | 12.561 (10.990 dB), 18.274 (12.618 dB) |
| OS-CFAR default rank | $k$ | $k = \mathrm{round}(3M/4)$ | `default_os_rank` | 24, with $\alpha_{\mathrm{cfar}}$ = 11.342 (10.547 dB) |
| Reference cells of a 2-D ring | $M$ | $(2g_0 + 2t_0 + 1)(2g_1 + 2t_1 + 1) - (2g_0 + 1)(2g_1 + 1)$ | `cfar_reference_cell_count((t0, t1), (g0, g1))` | — (1-D only) |
| Cells tested per map | $N_{\mathrm{cells}}$ | every cell | — | 256 × 1000 = 256,000 |
| Expected false alarms per map | — | $N_{\mathrm{cells}} P_{fa}$ | — | 2.56 |
| A plot's noise | $\sigma_R$, $\sigma_{\dot R}$ | one bin over $\sqrt{12}$ | `Plots.sigma_range_m`, `.sigma_rate_mps` | 149.896 / √12 = 43.27 m, 0.119496 / √12 = 0.0345 m/s |

Symbols used only on this page:

- $P$, $P_i$: the detected power of the CUT and of reference cell $i$, in W.
- $N_{\mathrm{train}}$, $N_{\mathrm{guard}}$: training and guard cells on each side. The
  docstrings call $N_{\mathrm{train}}$ just $N$; on this page $N$ stays the chirps per CPI.
- $M$: reference cells in the window; $M_c$: those a cell near the end of an axis keeps;
  $\hat{P}_n$: the noise estimate from them, in W.
- $i$: a cell's index counted from the nearer end of a non-circular axis, from 0.
- $T$: the threshold, in W; $k$: the OS-CFAR rank, counted from the smallest.
- $t_i$, $g_i$: a 2-D ring's training and guard cells per side along axis $i$.
- $N_{\mathrm{cells}}$: the number of cells tested.
- $R_b$, $\dot R_b$: bistatic range and its rate; $\sigma_R$, $\sigma_{\dot R}$: a plot's
  standard deviation in each.

Cell-averaging (CA) CFAR sets the threshold at a multiple of the mean of the $M$ reference cells:

$$T = \alpha_{\mathrm{cfar}} \hat{P}_n, \qquad \hat{P}_n = \frac{1}{M} \sum_{i=1}^{M} P_i$$

In complex Gaussian noise each cell's power is exponential, and $P_{fa}$ comes out independent
of the noise level. That is what "constant false-alarm rate" means:

$$P_{fa} = \left( 1 + \frac{\alpha_{\mathrm{cfar}}}{M} \right)^{-M}$$

Inverting it gives the factor for the $P_{fa}$ you want:

$$\alpha_{\mathrm{cfar}} = M \left( P_{fa}^{-1/M} - 1 \right)$$

As $M$ grows, $\alpha_{\mathrm{cfar}}$ falls towards $-\ln P_{fa}$, the factor you would use if
you knew the noise power exactly. The gap is the price of estimating it. For scenario 003 it is
11.417 dB against 10.612 dB. (The **CFAR loss** proper is the extra SNR a target then needs to
be detected as often; it grows with this gap, but the two are not the same number.) A cell with
fewer reference cells pays more: 12.268 dB with 16.

## How it works, step by step

1. **Square-law detect.** Take $\lvert x \rvert^2$ of the complex map, shape
   `(n_doppler, n_range)`. Every function here takes power in watts and rejects negative or
   non-finite values, which usually mean an amplitude or a dB value was passed by mistake.
2. **Threshold every cell.** `cfar_detect(power_w, pfa, n_train=, n_guard=, variant=, axis=-1,
   wrap=)` slides the 1-D window along one axis, usually range, for every Doppler row, and
   returns `(detected, threshold_w)`, both the map's shape. Four variants reduce the reference
   cells differently:
   - `"ca"`: the mean of all $M$. Best in uniform noise.
   - `"go"`: the greater of the two half-window means. Fewer false alarms at a clutter edge.
   - `"so"`: the smaller of the two. Keeps detecting when a second target sits in one half.
   - `"os"`: the $k$-th smallest cell (ordered statistic). Robust to several interfering
     targets. `default_os_rank` gives Rohling's $k \approx 3M/4$.
3. **Choose the factor.** `cfar_threshold_factor(pfa=, n_train=, variant=)` inverts
   `cfar_probability_of_false_alarm`. CA has the closed form above; GO, SO and OS are solved
   with `scipy.optimize.brentq` on $\ln \alpha_{\mathrm{cfar}}$. `cfar_detect` does this
   itself; call it to see the number.
4. **The ends of a non-circular axis: clip and recalibrate.** Near an end the window runs off
   the map. The cells it loses are left out, and the cell is calibrated for the $M_c$ it keeps,
   so every cell is tested at the design $P_{fa}$, at a higher threshold near the ends. CA uses
   the factor for $M_c$; GO and SO, which need two equal halves, fall back to CA over the cells
   left; OS reads rank $k - (M - M_c)$ of the cells left, and a cell where that falls below 1
   is not tested (its threshold is `nan`), which only a very low rank reaches.
5. **Circular axes.** Pass `wrap=True` when the axis is a circle. The Doppler axis always is,
   and so is a range axis that wraps at the chirp period. The window then runs off one end and
   on at the other, and every cell keeps all $M$. The pipeline wraps range.
6. **Or a 2-D ring.** `cfar_detect_2d(power_w, pfa, n_train=(t0, t1), n_guard=(g0, g1),
   variant=, wrap_axes=(-2,))` takes its reference cells from a rectangular ring over the last
   two axes, for all four variants. GO and SO split the ring in two equal halves: the cells
   beyond the CUT in range, plus those beyond it in Doppler in its own range column; and the
   mirror image. Edges are clipped and recalibrated as in 1-D, so no border goes untested. A
   zero pair on one axis reduces the ring to the 1-D window along the other.
7. **Group the crossings.** `label_connected_cells(mask, wrap_axes=)` gives every connected
   group of True cells its own integer label. `cluster_detections(mask, power_w, threshold_w=,
   wrap_axes=)` turns each group into one `Detection`, strongest first: its peak cell, its
   power-weighted centroid in fractional cell indices, its total power, its `n_cells`, and
   `cfar_threshold_w`, the threshold at its peak. Pass the Doppler axis in `wrap_axes` (`-2`),
   or a target on the rate wrap becomes two detections.
8. **Plots.** `pipeline.frame_plots(products, scenario.detection)` runs steps 1 to 7 on each
   burst's map with the scenario's `[detection]` table: CA-CFAR along range with range wrapped,
   so every cell keeps its full window; clusters on the circle of both axes; then keeps only the
   strongest cluster within `merge_range_bins` range cells (scenario 003: 1), since an untapered
   map's Doppler sidelobes cross at the target's own range. Each kept centroid becomes a
   bistatic range and a folded rate through the map's own axes, with one bin over $\sqrt{12}$
   as its sigma, the quantisation floor. `Plots.bistatic_range_rate_mps`, the unfolded rate, is
   `nan` until a fold is known: from a dual-PRF pair, or from one burst whose rate interval
   covers `max_bistatic_range_rate_mps`. On S1 neither holds; [page 6](06-tracking.md) shows
   what the tracker does instead. `Plots.accepted` lists the plots that reach the tracker.

## Try it

This builds scenario 003's CFAR, shows how a clipped edge is calibrated, checks the false-alarm
rate on maps of pure noise, then adds one target and clusters it:

```python
import numpy as np

from radar_forge import detection

# Scenario 003's [detection] table: CA-CFAR along range, 16 training and 4 guard cells a side.
pfa, window = 1e-5, {"n_train": 16, "n_guard": 4}
alpha = detection.cfar_threshold_factor(pfa=pfa, n_train=16)
n_reference = detection.cfar_reference_cell_count(16, 4)
pfa_back = detection.cfar_probability_of_false_alarm(alpha, n_train=16)
print(f"M = {n_reference}, alpha_cfar = {alpha:.3f} ({10 * np.log10(alpha):.3f} dB)")
print(f"P_fa from alpha_cfar: {pfa_back:.3e}")

# On a flat 1 W profile each cell's threshold is its own factor.
for wrap in (False, True):
    _, threshold_w = detection.cfar_detect(np.ones(1000), pfa, wrap=wrap, **window)
    cells_db = 10 * np.log10(threshold_w[[0, 4, 5, 10, 20, 500]])
    print(f"wrap={wrap}: cells 0, 4, 5, 10, 20, 500 at {cells_db.round(3).tolist()} dB")

# Ten maps of noise alone, (n_doppler, n_range) each: exponential power, mean 1 W.
rng = np.random.default_rng(20260911)
power_w = rng.exponential(1.0, size=(10, 256, 1000))
detected, _ = detection.cfar_detect(power_w, pfa, wrap=True, **window)
expected = detected.size * pfa
print(f"measured P_fa: {detected.mean():.2e}, {detected.sum()} alarms, {expected:.1f} expected")

# One map, with a target spread over 2 Doppler bins and 3 range cells.
one_w = power_w[0].copy()
one_w[100:102, 400:403] += [300.0, 1000.0, 300.0]
detected, threshold_w = detection.cfar_detect(one_w, pfa, wrap=True, **window)
found = detection.cluster_detections(
    detected,
    one_w,
    threshold_w=threshold_w,
    wrap_axes=(-2, -1),  # both axes are circles
)
target = found[0]
centroid = np.round(target.centroid_index, 2).tolist()
print(f"cells over threshold: {detected.sum()}, detections: {len(found)}")
print(f"strongest: peak {target.peak_index}, centroid {centroid}")
print(f"  {target.n_cells} cells, threshold at its peak {target.cfar_threshold_w:.2f} W")
```

It prints:

```text
M = 32, alpha_cfar = 13.856 (11.417 dB)
P_fa from alpha_cfar: 1.000e-05
wrap=False: cells 0, 4, 5, 10, 20, 500 at [12.268, 12.268, 12.165, 11.798, 11.417, 11.417] dB
wrap=True: cells 0, 4, 5, 10, 20, 500 at [11.417, 11.417, 11.417, 11.417, 11.417, 11.417] dB
measured P_fa: 1.09e-05, 28 alarms, 25.6 expected
cells over threshold: 9, detections: 4
strongest: peak (101, 401), centroid [100.5, 401.0]
  6 cells, threshold at its peak 12.36 W
```

Without wrapping, cells 0 to 4 keep only the 16 cells of their far band; from cell 5 the near
band comes back one cell at a time, and from cell 20 the window is whole. The measured count, 28
where 25.6 were expected, is within one standard deviation (about 5) of a Poisson count. The
target's six cells become one detection; its threshold, 12.36 W, is 13.856 times the mean of
its 32 reference cells. The other three detections are that noise map's own false alarms, one
cell each.

The pipeline does the same on a real map. This runs frame 0 of scenario 003 through it:

```python
from radar_forge.dsp import fold_velocity_mps
from radar_forge.pipeline import frame_plots, frames, range_doppler_product
from radar_forge.scenario import load_scenario

# Frame 0 of scenario 003: one burst, one map, CFAR along range, clusters, plots.
scenario = load_scenario("config/scenario_003_tracking.toml")
frame = next(frames(scenario, n_frames=1))
product = range_doppler_product(frame.cubes[0], scenario.radar)
plots = frame_plots([product], scenario.detection)

range_m, rate_mps = frame.truth.bistatic_range_m[0], frame.truth.bistatic_range_rate_mps[0]
folded_mps = fold_velocity_mps(rate_mps, product.rate_period_mps)
bins = f"{product.range_bin_m:.3f} m, {product.rate_bin_mps:.6f} m/s"
print(f"map {product.rd_map.shape}: bins {bins}")
print(f"truth: {range_m:.1f} m, {rate_mps:.3f} m/s, folded to {folded_mps:.3f} m/s")
for k, cluster in enumerate(plots.detections):
    print(
        f"plot {k}: {plots.bistatic_range_m[k]:.1f} m, {plots.folded_rate_mps[k]:.3f} m/s,"
        f" {cluster.n_cells} cells, {plots.status[k]}"
    )
print(f"sigmas: {plots.sigma_range_m[0]:.2f} m, {plots.sigma_rate_mps[0]:.4f} m/s")
print(f"unfolded rates: {plots.bistatic_range_rate_mps}")
```

```text
map (256, 1000): bins 149.896 m, -0.119496 m/s
truth: 36316.4 m, 59.532 m/s, folded to -1.650 m/s
plot 0: 36302.9 m, -1.668 m/s, 53 cells, accepted
plot 1: 114220.9 m, -0.836 m/s, 1 cells, accepted
plot 2: 97882.2 m, -1.075 m/s, 1 cells, accepted
sigmas: 43.27 m, 0.0345 m/s
unfolded rates: [nan nan nan]
```

The target, 18,158 m away and receding at 29.8 m/s, sits at twice both on the map. Its 53 cells
are the peak and its Doppler sidelobes; two more clusters at its range cell, sidelobes too, were
dropped by `merge_range_bins`. Plots 1 and 2 are false alarms. No rate is unfolded yet: S1's
map folds the rate at ±15.30 m/s.

## Try it in a scenario

[Scenario 003](../../config/scenario_003_tracking.toml) runs this on every frame of scenario
001's S1 radar and tracks what it finds. Run ten frames through the stepper:

```bash
uv run python scripts/run_scenario.py config/scenario_003_tracking.toml --frames 10
```

`[rd]` prints `n_detections`, the plots over all frames, and `false_alarm_ratio`: the measured
over the design $P_{fa}$ on noise-only maps, which `[verify]` holds within a factor of two.
`[detect]` checks the rate of the plot nearest each target against the truth, folded as the
map folds it, and writes `runs/scenario_003_tracking/detections.csv`, one row per detection. A
monostatic radar's CSV shows range and rate halved (`plots.display_axes`), so `range_m` is the
slant range and `sigma_range_m` is 21.64 m.

**Knob:** in a copy of the TOML in `config/`, change `pfa = 1e-5` to `pfa = 1e-4` in
`[detection]`, and run it the same way with `--out runs/t1_pfa4`. Then compare what `[rd]` and
`[track]` print, `tracks.csv`, and `track.png`, range against time with every accepted plot
drawn.

| | `pfa = 1e-5` | `pfa = 1e-4` |
| :-- | --: | --: |
| Expected false alarms per frame, $N_{\mathrm{cells}} P_{fa}$ | 2.56 | 25.6 |
| `n_detections`, 10 frames | 40 | 235 |
| Track IDs started (largest `track_id` in `tracks.csv`), 10 frames | 29 | 154 |
| `n_confirmed_tracks`, whole 120-frame window (drop `--frames`) | 1 | 10 |

Ten times the $P_{fa}$ gives about six times the detections and five times the tentative tracks
in ten frames: every plot no track takes starts one. Both ten-frame runs pass every check, and
neither confirms a false track. The threshold drops only 1.13 dB (11.417 to 10.283 dB), yet
over the full window nine false tracks are confirmed where there were none. The target's own
track gets worse too: its range RMSE rises from 23.8 to 25.6 m, and its `nees_fraction` falls
from 0.37 to 0.29, so that check fails. That is why scenario 003 treats `pfa` as its main
difficulty knob. The target itself is detected in every frame either way: it sits far above
both thresholds.

Count the false alarms yourself (plots more than two range bins from the truth) and they come
out a little under $N_{\mathrm{cells}} P_{fa}$: over the full window, 2.1 per frame at 1e-5 and
23.0 at 1e-4. At 1e-4 about one a frame is lost to `merge_range_bins`, which keeps only the
stronger of two alarms within a range cell of each other.

Intern tasks that build on this page: [Thresholds from the tail](../../interns/evt-thresholds.md)
and [Where does the tracker break?](../../interns/tracker-breakpoint.md).

## Things that look like bugs and aren't

**OS-CFAR's factor is smaller than CA's, yet OS is said to cost more.** At scenario 003's
settings OS needs 10.547 dB and CA 11.417 dB. The factors multiply different estimates: rank 24
of 32 sits above the mean, so it needs a smaller multiplier. Never reuse one variant's factor
with another; OS at CA's factor gives $P_{fa}$ = 1.5e-6, not 1e-5. Compare variants by the SNR a
target needs, not by $\alpha_{\mathrm{cfar}}$.

**The end cells of a range profile have a higher threshold.** With `wrap=False`, the 5
outermost cells at each end of scenario 003's range axis use 12.268 dB, cell 10 uses 11.798 dB,
and only from cell 20 is it 11.417 dB. Their windows hold 16 to 31 cells, not 32, so the
estimate is noisier and the factor rises to hold the same $P_{fa}$.

**A receding target's plot has a negative rate.** The rate is folded: S1's map spans
30.591 m/s, so the target's +59.532 m/s reads 59.532 − 2 × 30.591 = −1.650 m/s. The rate axis
also falls with bin index, because $f_d = -\dot R_b / \lambda$ and the Doppler axis rises.

**A plot's range is twice the target's distance.** The map's range is bistatic,
$R_T + R_R$; for a monostatic radar that is twice the slant range. Only `plots.display_axes`
halves it, for a figure or a CSV.

**A measured false-alarm rate comes out above `pfa` on a real map.** The closed forms assume
independent reference cells. A taper, zero-padding or oversampling correlates neighbouring
cells, and the measured rate rises (the `detection` module docstring). Clutter heavier-tailed
than Gaussian noise does the same. Measure on independent, Gaussian cells first.

**A single target crosses in dozens of cells.** Frame 0's target crosses in 53, and frame 1's in
eight separate clusters at its range: Doppler sidelobes, since scenario 001 applies no Doppler
taper. `merge_range_bins` keeps the strongest per range cell. Two clusters at opposite
ends of the Doppler axis instead mean `wrap_axes` was left out.

## Limits

- The calibration assumes exponential (Gaussian-noise) cells. Weibull or K-distributed clutter
  needs a distribution-matched CFAR (log-t, Weibull), which is not implemented.
- No variability-index CFAR, which picks between CA, GO and SO from the data.
- OS with a rank below $M - M_c + 1$ leaves the cells that keep only $M_c$ untested (`nan`
  threshold).
- Two targets whose crossings touch are one cluster: there is no non-maximum suppression. And
  `merge_range_bins` keeps one plot per range cell, so two targets at one range are one plot: a
  single-target simplification for an untapered map.
- The centroid is power-weighted over every crossing cell, sidelobes included; there is no
  parabolic peak interpolation. A plot's sigma is the quantisation floor, a bin over
  $\sqrt{12}$, whatever its SNR.

## References

- H. M. Finn and R. S. Johnson, "Adaptive detection mode with threshold control as a function of
  spatially sampled clutter-level estimates," *RCA Review*, vol. 29, pp. 414–464, 1968 (CA-CFAR)
- V. G. Hansen and J. H. Sawyers, "Detectability loss due to 'greatest of' selection in a
  cell-averaging CFAR," *IEEE Trans. Aerosp. Electron. Syst.*, vol. AES-16, no. 1, pp. 115–118,
  1980 (GO-CFAR)
- M. Weiss, "Analysis of some modified cell-averaging CFAR processors in multiple-target
  situations," *IEEE Trans. Aerosp. Electron. Syst.*, vol. AES-18, no. 1, pp. 102–114, 1982
  (SO-CFAR)
- H. Rohling, "Radar CFAR thresholding in clutter and multiple target situations," *IEEE Trans.
  Aerosp. Electron. Syst.*, vol. AES-19, no. 4, pp. 608–621, 1983, eq. 14 (OS-CFAR)
- P. P. Gandhi and S. A. Kassam, "Analysis of CFAR processors in nonhomogeneous background,"
  *IEEE Trans. Aerosp. Electron. Syst.*, vol. 24, no. 4, pp. 427–445, 1988 (GO and SO closed
  forms)
- M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014, §6.5
  (CFAR)
- M. E. Smith and P. K. Varshney, "Intelligent CFAR processor based in data variability," *IEEE
  Trans. Aerosp. Electron. Syst.*, vol. 36, no. 3, pp. 837–847, 2000 (VI-CFAR; not implemented)
- A. Rosenfeld and J. L. Pfaltz, "Sequential operations in digital picture processing," *J. ACM*,
  vol. 13, no. 4, pp. 471–494, 1966 (connected-component labelling)
