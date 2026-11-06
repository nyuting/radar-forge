# Scenario 011b — Terminal-area tracking: track-oriented MHT, PMBM and the separation sweep

## 1. Status & Purpose

**Status: specified, not implemented.** The second half of scenario 011. 011a built the scene,
the plot model, the coordinated turn, IMM, JPDA (with JPDA\* and the set relabelling), JIPDA,
the clutter map and GOSPA; its acceptance numbers live in `config/scenario_011_*.toml` and
its tests. This document adds what defers association: track-oriented MHT, inside the one
`Tracker` loop, and PMBM, the random-finite-set tracker beside it as the reference score, and
the separation sweep V3 on which they are compared.

Everything 011b shares is 011a's code, used by reference, not copied:

| Shared | Where |
| :--- | :--- |
| Sensor, plots, merge model, plot timing and sector batching | `[sensor]`; `tracking/scene.py` (`resolve`, `detect`, `measure`); `tracking/scans.py` (`scans`, `PlotTracker`) |
| Clutter: intensity per (m rad), `n_gate` against the 4.612 m rad reference gate, local windows, the clutter map | `[clutter_plots]`; `ClutterComponent`, `poisson_clutter_plots`, `ClutterMap` |
| Traffic and its closed-form arcs | `config/scenario_011_terminal_dense.toml`; `tracking.scene.Arc` |
| Motion models, IMM, measurement model, gate probability 0.99 | `[tracking]`; `tracking/models.py`, `tracking/imm.py` |
| JPDA family, existence, the event bound | `tracking/association.py`; `tracking/tracker.ExistencePolicy` |
| GOSPA (p = 2, c = 300 m, α = 2), trajectory GOSPA, the swap and coalescence definitions | `tracking/metrics.py`; `stages/tracking.py` |

## 2. What 011b adds

### 2.1 V3 `cst_sweep`: the separation sweep

Pairs placed **in radar coordinates**, as 006 §4.3 did: the same velocity, offset by a stated
`d` along a stated direction (range or azimuth) in the normalised (range, azimuth) plane of
`tracking.scene.resolve`, at 20 km, so `d` holds for the run. The sweep is `d ∈ {0.25, 0.5, 1, 2, 4}` by
`n_gate ∈ {0.1, 1, 3}` expected clutter plots per gate, each in both orientations, in a local
window of clutter (011a's V2 windows: 19, 189 and 568 plots a scan in a box of 872.7 m rad, or
the 2.5 km disc's 983.7 m rad). Trackers: GNN, JPDA, JPDA\*, Set-JPDA, JIPDA, MHT, PMBM, each
CV quiet. 100 trials per point, 40 scans. 011a's A16 test runs V3's point `d = 1` across range
at `n_gate = 1` for the JPDA family; its pair generator is the template.

| Contract | Statement |
| :--- | :--- |
| Resolved pair (V3) | Two confirmed tracks, each within c/2 of its own aircraft in at least 90 % of the last 20 scans, with no switch |
| Coalescence ratio (V3) | Each trial's mean estimated over true separation over the last 20 scans; reported by its median over trials (011a: the mean is dominated by lost tracks) |

### 2.2 Track-oriented MHT (D29)

MHT defers association; `Tracker` decides once per scan. The tree is the estimator:

- A `TrackTree` per track holds, as rows of a `TrackBank`, one leaf per surviving association
  history; its reported estimate is the leaf in the current best global hypothesis.
- `TrackOrientedMHT` is an associator. It scores every (leaf, measurement) pair with the track
  score [10]: `L_k = L_{k−1} + ln(P_D / λ(z)) − ½ νᵀS⁻¹ν − ½ ln|2πS|` on a hit,
  `+ ln(1 − P_D P_G)` on a miss. For each cluster it finds the `k_best` global hypotheses by
  Murty's algorithm [11], [12] over `linear_sum_assignment`, and prunes by N-scan [9].
- `murty_k_best` lives in `tracking/association.py` beside GNN, so the k = 1 case is GNN (A4).
- New trees start from unused measurements, scored from zero; MHT confirms by score at the Wald
  threshold `ln((1 − β)/α)` = 9.105 (α = 1e-4, β = 0.1) and deletes when the score falls
  2.303 = `−ln(β/(1 − α))` below its maximum. `n_scan = 3` (14.4 s of deferral at 4.8 s),
  `k_best = 10` global hypotheses per cluster per scan.
- What it costs, recorded: the estimate at scan k is the best hypothesis at k and is not
  rewritten if MHT later prefers another (`tracks.csv` is causal, A19); a tree's memory is
  bounded by `k_best` per cluster per scan times `n_scan`, and the leaf count is a reported
  intermediate.

### 2.3 PMBM (D30)

PMBM's association hypotheses, existence and birth are part of its density, so there is no
associator, initiator or lifecycle to swap: it is **a second tracker**, `PmbmTracker`, in
`tracking/pmbm.py`. It shares the motion and measurement models, `innovate`, `murty_k_best`,
`Scan.clutter_intensity` and the metrics; it owns its Poisson undetected component, Bernoulli
tracks and global hypotheses [13]; it reports a Bernoulli with existence above 0.5 as a track in
`tracks.csv`, labelled by the measurement history that created it. Single CV model at M3's
noise (σ = 3.1623 m/s²); Poisson birth intensity 0.05 a scan, uniform over coverage;
`k_best = 100` per cluster; Bernoulli pruning at existence 1e-5, global-hypothesis pruning at
weight 1e-4. Williams [14] shows MHT and JIPDA approximate the same conjugate prior, which
makes PMBM the reference for their losses. GLMB [15] is named, not built.

### 2.4 Variants and outputs

`config/scenario_011_cst_sweep.toml` holds V3's scene; its sweep runs as `slow` Monte Carlo
tests, two points in the unmarked suite. V2 and V4 gain the MHT and PMBM rows: V2's trackers are
GNN, JPDA, JPDA\*, Set-JPDA, JIPDA, MHT, PMBM; V4's are IMM with GNN, JIPDA and MHT, and PMBM.
`tracks.csv` gains `score` (MHT); `gospa.csv` a `tracker` column where one run scores several.

## 3. Acceptance criteria

| # | Criterion | Bound | Justification | Applies to |
| :--- | :--- | :--- | :--- | :--- |
| A4 | **MHT reduces to GNN.** `n_scan = 1`, `k_best = 1`: MHT's assignments and states equal GNN's with the finite non-assignment cost, on the V2 30° crossings | identical assignments; states `rtol = 1e-10` | Murty's first solution is the assignment | unit |
| A10 | **PMBM reduces to the KF.** One target, Pd = 1, no clutter, no birth after scan 0 | `rtol = 1e-9` | The single-Bernoulli case | unit |
| A15 | **GNN swaps; MHT does not.** V2, 10° across-range crossing, 200 trials: swap rate GNN ≥ 0.2; MHT ≤ 0.05; JPDA's reported, not bounded | as stated, the bounds checked against a prototype first | A 10° pair closing across the beam spends sixteen scans inside one resolution cell | V2, `slow` |
| A16 (MHT) | **MHT resolves where JPDA coalesces.** V3, `d` = 1 across range, `n_gate` = 1: P(resolved) for MHT ≥ P(resolved) for JPDA | as stated | JPDA coalesces where hypotheses are deferred | V3, `slow` |
| A17 | **The reference line.** V3, every point: PMBM's P(resolved) ≥ every other tracker's, less 0.05 | as stated | [14]: the others approximate PMBM's posterior; a tracker beating it materially is a bug in one of them | V3, `slow` |
| A19 | **Causal output.** For MHT, the `tracks.csv` row at scan k equals the tracker's state at scan k, recorded live | identical | D29 | V4 |

## 4. Limitations

PMBM runs one motion model, not the IMM: V3 has no manoeuvres, so it is a fair test of
association, and on V4 the comparison carries that caveat. MHT's reported estimate is causal,
never smoothed.

## References

.. [8] D. B. Reid, "An algorithm for tracking multiple targets," *IEEE Trans. Autom. Control*,
       24(6), pp. 843–854, 1979.
.. [9] T. Kurien, "Issues in the design of practical multitarget tracking algorithms," in
       Y. Bar-Shalom (ed.), *Multitarget-Multisensor Tracking: Advanced Applications*, Artech
       House, 1990, pp. 43–83.
.. [10] S. S. Blackman, "Multiple hypothesis tracking for multiple target tracking," *IEEE
        Aerosp. Electron. Syst. Mag.*, 19(1), pp. 5–18, 2004.
.. [11] K. G. Murty, "An algorithm for ranking all the assignments in order of increasing
        cost," *Operations Research*, 16(3), pp. 682–687, 1968.
.. [12] I. J. Cox and M. L. Miller, "On finding ranked assignments with application to
        multitarget tracking and motion correspondence," *IEEE Trans. Aerosp. Electron. Syst.*,
        31(1), pp. 486–489, 1995.
.. [13] Á. F. García-Fernández, J. L. Williams, K. Granström and L. Svensson, "Poisson
        multi-Bernoulli mixture filter: direct derivation and implementation," *IEEE Trans.
        Aerosp. Electron. Syst.*, 54(4), pp. 1883–1901, 2018.
.. [14] J. L. Williams, "Marginal multi-Bernoulli filters: RFS derivation of MHT, JIPDA, and
        association-based MeMBer," *IEEE Trans. Aerosp. Electron. Syst.*, 51(3),
        pp. 1664–1687, 2015.
.. [15] B.-T. Vo and B.-N. Vo, "Labeled random finite sets and multi-object conjugate priors,"
        *IEEE Trans. Signal Process.*, 61(13), pp. 3460–3475, 2013.
