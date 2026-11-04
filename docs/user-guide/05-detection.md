# 5. Detection

A range-Doppler map holds a power in every cell. Detection decides which cells hold a target.
This page covers `radar_forge.core.detection`: constant-false-alarm-rate (CFAR) thresholds, the
false-alarm arithmetic behind them, and clustering the cells that cross into one detection per
target. It sits between [range-Doppler processing](04-range-doppler-processing.md) and
[tracking](06-tracking.md) in [the chain](README.md#the-chain).

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
one detection per group, which is what a tracker expects.

## Quantities and equations

| Quantity | Symbol | Equation | In the code | Scenario 003 |
| :-- | :-- | :-- | :-- | :-- |
| Detected (square-law) power of a cell | $P$ | $P = \lvert x \rvert^2$ | `np.abs(rd_map) ** 2` | — |
| Training and guard cells per side | $N_{\mathrm{train}}$, $N_{\mathrm{guard}}$ | — | `n_train`, `n_guard` | 16, 4 |
| Reference cells | $M$ | $M = 2 N_{\mathrm{train}}$ | — | 32 |
| CA noise estimate | $\hat{P}_n$ | $\hat{P}_n = \frac{1}{M} \sum_{i=1}^{M} P_i$ | `cfar_noise_estimate_w` | — |
| Threshold | $T$ | $T = \alpha_{\mathrm{cfar}} \hat{P}_n$ | `cfar_threshold_w` | — |
| CA threshold factor | $\alpha_{\mathrm{cfar}}$ | $\alpha_{\mathrm{cfar}} = M \left( P_{fa}^{-1/M} - 1 \right)$ | `cfar_threshold_factor` | 13.856 (11.417 dB) |
| CA false-alarm probability | $P_{fa}$ | $P_{fa} = \left( 1 + \alpha_{\mathrm{cfar}} / M \right)^{-M}$ | `cfar_probability_of_false_alarm` | 1e-5 |
| Factor at a one-sided end cell | $\alpha_{\mathrm{cfar}}$ | as above with $M = N_{\mathrm{train}}$ | `cfar_threshold_factors` | 16.856 (12.268 dB), not used: range wraps |
| Factor if the noise power were known | — | $-\ln P_{fa}$ | — | 11.513 (10.612 dB) |
| OS-CFAR default rank | $k$ | $k = \mathrm{round}(3M/4)$ | `default_os_rank` | 24, with $\alpha_{\mathrm{cfar}}$ = 11.342 (10.547 dB) |
| Reference cells of a 2-D ring | $M$ | $(2g_0 + 2t_0 + 1)(2g_1 + 2t_1 + 1) - (2g_0 + 1)(2g_1 + 1)$ | `cfar_ring_reference_count` | — (1-D only) |
| Cells tested per map | $N_{\mathrm{cells}}$ | — | `cfar_valid_mask` | 256 × 1000 = 256,000 |
| Expected false alarms per map | — | $N_{\mathrm{cells}} P_{fa}$ | — | 2.56 |

Symbols used only on this page:

- $P$, $P_i$: the detected power of the CUT and of reference cell $i$, in W.
- $N_{\mathrm{train}}$, $N_{\mathrm{guard}}$: training and guard cells on each side. The
  docstrings call $N_{\mathrm{train}}$ just $N$; on this page $N$ stays the chirps per CPI.
- $M$: reference cells in the window; $\hat{P}_n$: the noise estimate from them, in W.
- $T$: the threshold, in W; $k$: the OS-CFAR rank, counted from the smallest.
- $t_i$, $g_i$: a 2-D ring's training and guard cells per side along axis $i$.
- $N_{\mathrm{cells}}$: the number of cells tested.

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
be detected as often; it grows with this gap, but the two are not the same number.) `cfar_threshold_factor`'s Notes explain why
$N_{\mathrm{train}}$ is usually at least 16.

## How it works, step by step

1. **Square-law detect.** Take $\lvert x \rvert^2$ of the complex map, shape
   `(n_doppler, n_range)`. Every function here takes power in watts and rejects negative or
   non-finite values, which usually mean an amplitude or a dB value was passed by mistake.
2. **Estimate the floor.** `cfar_noise_estimate_w(power_w, n_train=, n_guard=, variant=,
   axis=-1)` slides the 1-D window along one axis, usually range, for every Doppler row. Four
   variants reduce the reference cells differently:
   - `"ca"`: the mean of all $M$. Best in uniform noise.
   - `"go"`: the greater of the two half-window means. Fewer false alarms at a clutter edge.
   - `"so"`: the smaller of the two. Keeps detecting when a second target sits in one half.
   - `"os"`: the $k$-th smallest cell (ordered statistic). Robust to several interfering
     targets. `default_os_rank` gives Rohling's $k \approx 3M/4$.
3. **Choose the factor.** `cfar_threshold_factor(pfa=, n_train=, variant=)` inverts
   `cfar_probability_of_false_alarm`. CA has the closed form above; GO, SO and OS are solved
   by bisection. `cfar_threshold_factors(shape, ...)` returns one factor per cell along the
   axis, shape `(1, n_range)` for a `(n_doppler, n_range)` map, so it broadcasts.
4. **Threshold and compare.** `cfar_threshold_w` multiplies the two; `cfar_detect` returns the
   boolean mask, shape `(n_doppler, n_range)`.
5. **The ends of the axis.** The $N_{\mathrm{guard}} + N_{\mathrm{train}}$ cells at each end
   of a non-circular axis have no full two-sided window. Each uses a **one-sided** window: the
   complete $N_{\mathrm{train}}$ band on the far side, with a factor calibrated for those
   $N_{\mathrm{train}}$ cells, so they keep the design $P_{fa}$ at a higher threshold. So every
   cell is tested. `cfar_valid_mask` is all True unless the axis is shorter than one window,
   $2(N_{\mathrm{guard}} + N_{\mathrm{train}}) + 1$ cells.
6. **Circular axes.** Pass `wrap=True` when the axis is a circle. The Doppler axis always is,
   and so is a range axis that wraps at the chirp period. The window then runs off one end and
   on at the other, and every cell uses the interior factor. Scenario 003 wraps range
   (`frame_detections(..., wrap_range=True)`), so its end cells use real cells from the far
   end, not a one-sided window.
7. **Or a 2-D ring.** `cfar_detect_2d(power_w, n_train=(t0, t1), n_guard=(g0, g1),
   wrap_axes=(0,))` takes its reference cells from a rectangular ring over both axes. Only CA
   and OS are offered: their $P_{fa}$ depends on $M$ alone, so `cfar_threshold_factor_2d`
   calibrates the ring as $M$ cells. A cell without a complete ring is not tested; check
   `cfar_valid_mask_2d`.
8. **Group the crossings.** `label_connected_cells(mask, wrap_axes=)` gives every connected
   group of True cells its own integer label. `cluster_detections(mask, power_w, wrap_axes=)`
   turns each group into one `Detection`: its peak cell, its power-weighted centroid in
   fractional cell indices, its total power and its `n_cells`. Pass the Doppler axis in
   `wrap_axes`, or a target on the ±velocity wrap becomes two detections.

## Try it

This builds scenario 003's CFAR on maps of pure noise, checks the false-alarm rate, then adds
one target and clusters it:

```python
import numpy as np

from radar_forge.core import detection

# Scenario 003's [detection] table: CA-CFAR along range, on a (256, 1000) map.
pfa, n_train, n_guard = 1e-5, 16, 4
map_shape = (256, 1000)  # (n_doppler, n_range)
window = {"n_train": n_train, "n_guard": n_guard}

alpha = detection.cfar_threshold_factors(map_shape, pfa=pfa, **window)[0]
print(f"alpha_cfar, interior: {alpha[500]:.3f} ({10 * np.log10(alpha[500]):.3f} dB)")
print(f"alpha_cfar, end cells: {alpha[0]:.3f} ({10 * np.log10(alpha[0]):.3f} dB)")
pfa_back = detection.cfar_probability_of_false_alarm(alpha[500], n_train=n_train)
print(f"P_fa from alpha_cfar: {pfa_back:.3e}")
n_tested = int(detection.cfar_valid_mask(map_shape, **window).sum())
print(f"cells tested: {n_tested}, false alarms per map: {n_tested * pfa:.2f}")

# Forty maps of noise alone: exponential power with a mean of 1 W.
rng = np.random.default_rng(20260911)
power_w = rng.exponential(1.0, size=(40 * 256, 1000))
mask = detection.cfar_detect(power_w, pfa=pfa, **window)
print(f"measured P_fa: {mask.mean():.2e} ({int(mask.sum())} alarms in {mask.size} cells)")

# One map, with a target spread over 2 Doppler bins and 3 range cells.
one_w = power_w[:256].copy()
one_w[100:102, 400:403] += [300.0, 1000.0, 300.0]
mask = detection.cfar_detect(one_w, pfa=pfa, **window)
found = detection.cluster_detections(mask, one_w, wrap_axes=(0,))
print(f"cells over threshold: {int(mask.sum())}, detections: {len(found)}")
print(f"strongest: peak {found[0].peak_index}, {found[0].n_cells} cells")
```

It prints:

```text
alpha_cfar, interior: 13.856 (11.417 dB)
alpha_cfar, end cells: 16.856 (12.268 dB)
P_fa from alpha_cfar: 1.000e-05
cells tested: 256000, false alarms per map: 2.56
measured P_fa: 1.09e-05 (112 alarms in 10240000 cells)
cells over threshold: 9, detections: 4
strongest: peak (101, 401), 6 cells
```

The measured count, 112 where 102.4 were expected, is within one standard deviation (about 10)
of a Poisson count. The target's six cells become one detection. The other three detections are
that noise map's own false alarms, one cell each.

## Try it in a scenario

[Scenario 003](../../scenarios/scenario_003_tracking.toml) runs this CFAR on every frame of
scenario 001's S1 radar and tracks what it finds. Run it:

```bash
uv run python scripts/run_scenario.py scenarios/scenario_003_tracking.toml --out out/t1 --frames 10 --no-iq --no-movie
```

Now copy the TOML, change one line in `[detection]`, from `pfa = 1e-5` to `pfa = 1e-4`, and
run the copy into `out/t1-pfa4`. Look at `detections.csv`, one row per detection, and at
`track_00009.png`, range against time with every detection drawn.

| | `pfa = 1e-5` | `pfa = 1e-4` |
| :-- | --: | --: |
| Expected false alarms per frame, $N_{\mathrm{cells}} P_{fa}$ | 2.56 | 25.6 |
| Rows in `detections.csv`, 10 frames | 40 | 234 |
| Track IDs started, 10 frames | 29 | 154 |
| Confirmed tracks, whole 120-frame window (`metrics.csv`, `n_confirmed_tracks`) | 1 | 10 |

Ten times the $P_{fa}$ gives about six times the detections and five times the tentative tracks
in ten frames: every detection no track takes starts one. The threshold drops only 1.1 dB
(11.417 to 10.283 dB), yet over the full window (drop `--frames`) nine false tracks get
confirmed where there were none (in the first ten frames, neither run confirms a false track).
That is why scenario 003 treats `pfa` as its main difficulty knob (spec §3). The target itself
is detected in every frame either way: it sits far above both thresholds.

Count the false alarms yourself (detections more than 300 m from the range in `truth.csv`) and
they come out a little under $N_{\mathrm{cells}} P_{fa}$: over the full window, 2.1 per frame at
1e-5 and 22.9 at 1e-4. The simple product leaves out `merge_range_bins`, which drops any
detection within one range cell of a stronger one.

Intern tasks that build on this page: [Thresholds from the tail](../../interns/evt-thresholds.md)
and [Where does the tracker break?](../../interns/tracker-breakpoint.md).

## Things that look like bugs and aren't

**OS-CFAR's factor is smaller than CA's, yet OS is said to cost more.** At scenario 003's
settings OS needs 10.547 dB and CA 11.417 dB. The factors multiply different estimates: rank 24
of 32 sits above the mean, so it needs a smaller multiplier. Never reuse one variant's factor
with another; OS at CA's factor gives $P_{fa}$ = 1.5e-6, not 1e-5. Compare variants by the SNR a
target needs, not by $\alpha_{\mathrm{cfar}}$.

**The end cells of a range profile have a higher threshold.** With `wrap=False`, the 20 cells
at each end of scenario 003's range axis use 12.268 dB, not 11.417 dB. Their window has
16 cells, not 32, so the estimate is noisier and the factor rises to hold the same $P_{fa}$.

**A measured false-alarm rate comes out above `pfa` on a real map.** The closed forms assume
independent reference cells. A taper, zero-padding or oversampling correlates neighbouring
cells, and the measured rate rises (the module Notes). Clutter heavier-tailed than Gaussian
noise does the same. Measure on independent, Gaussian cells first.

**A single target gives more than one detection.** Two clusters at opposite ends of the
Doppler axis mean `wrap_axes` was left out. Satellites at the target's range, 10–28 Doppler bins
away, are its own Doppler sidelobes, since scenario 001 applies no Doppler taper; scenario 003
keeps only the strongest detection per range cell (`merge_range_bins`, spec §14.3).

## Limits

- The calibration assumes exponential (Gaussian-noise) cells. Weibull or K-distributed clutter
  needs a distribution-matched CFAR (log-t, Weibull), which is not implemented.
- No variability-index CFAR, which picks between CA, GO and SO from the data.
- The 2-D ring offers only CA and OS, and leaves cells without a complete ring untested
  (`nan` threshold); the 1-D window tests them one-sided.
- Running sums carry rounding error from strong cells: keep a map's dynamic range under about
  130 dB, or split it, before CFAR.
- Two targets whose crossings touch are one cluster: there is no non-maximum suppression.
- The centroid is power-weighted over every crossing cell, sidelobes included; there is no
  parabolic peak interpolation.

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
  Aerosp. Electron. Syst.*, vol. AES-19, no. 4, pp. 608–621, 1983 (OS-CFAR)
- P. P. Gandhi and S. A. Kassam, "Analysis of CFAR processors in nonhomogeneous background,"
  *IEEE Trans. Aerosp. Electron. Syst.*, vol. 24, no. 4, pp. 427–445, 1988 (GO and SO closed
  forms)
- M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014, §6.5
  (one-sided windows at the ends)
- M. E. Smith and P. K. Varshney, "Intelligent CFAR processor based in data variability," *IEEE
  Trans. Aerosp. Electron. Syst.*, vol. 36, no. 3, pp. 837–847, 2000 (VI-CFAR; not implemented)
- A. Rosenfeld and J. L. Pfaltz, "Sequential operations in digital picture processing," *J. ACM*,
  vol. 13, no. 4, pp. 471–494, 1966 (connected-component labelling)
