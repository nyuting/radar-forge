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
| Clutter: intensity per (m rad), `n_gate` against the 4.6123 m rad reference gate, local windows, the clutter map | `[clutter_plots]`; `ClutterComponent`, `poisson_clutter_plots`, `ClutterMap` |
| Traffic and its closed-form arcs | `config/scenario_011_terminal_dense.toml`; `tracking.scene.Arc` |
| Motion models, IMM, measurement model, gate probability 0.99 | `[tracking]`; `tracking/models.py`, `tracking/imm.py` |
| JPDA family, existence, the event bound | `tracking/association.py`; `tracking/tracker.ExistencePolicy` |
| GOSPA (p = 2, c = 300 m, α = 2), trajectory GOSPA, the swap and coalescence definitions | `tracking/metrics.py`; `stages/tracking.py` |

## 2. What 011b adds

### 2.1 V3 `cst_sweep`: the separation sweep

Pairs placed **in radar coordinates**, as 006 §4.3 did: offset by a stated `d` along a stated
direction (range or azimuth) in the normalised (range, azimuth) plane of
`tracking.scene.resolve`, at 20 km, both flying out at 72 m/s along their own bearings (across
range) or along one bearing (along range), so `d` holds for the run. The sweep is
`d ∈ {0.25, 0.5, 1, 2, 4}` by `n_gate ∈ {0.1, 1, 3}` expected clutter plots per gate
(0.021683, 0.21683 and 0.65048 plots per m rad a scan), each in both orientations, in a 2.5 km
disc of clutter that drifts with the pair (983.7 m rad: 21, 213 and 640 plots a scan; 011a's
A16 test uses the 5 km by 10° box, 872.7 m rad: 19, 189 and 568). Trackers: GNN, JPDA,
JPDA\*, Set-JPDA, JIPDA, MHT, PMBM, each CV quiet (σ = 0.5 m/s²). 100 trials per point, 40
scans. V3 compares association, not initiation, as 011a's A16 test does: every tracker starts
with each aircraft's track at its truth, confirmed (MHT at the confirmation score, PMBM a
Bernoulli of existence 1), no track is born inside the clutter, and the lifecycle deletes
nothing in 40 scans (M-of-N 1 of 1 with 40 misses; MHT's `delete_drop` infinite). The pair
sits at azimuth 180°, where one sector batch a rotation (`n_sectors = 1`) is stamped at the
beam's crossing.

| Contract | Statement |
| :--- | :--- |
| Resolved pair (V3) | Two confirmed tracks, each within c/2 of its own aircraft in at least 90 % of the last 20 scans, with no switch |
| Coalescence ratio (V3) | Each trial's mean estimated over true separation over the last 20 scans; reported by its median over trials (011a: the mean is dominated by lost tracks) |

### 2.2 Track-oriented MHT (D29)

MHT defers association; `Tracker` decides once per scan. The tree is the estimator:

- A `TrackTree` holds every tree's leaves as rows of a `TrackBank`, one per surviving
  association history, with its track score, the highest score along its branch, its
  ancestors over the last `n_scan` scans and the global hypotheses holding it; a tree's
  reported estimate is its leaf in its cluster's best global hypothesis.
- `TrackOrientedMHT` is an associator. It scores every (leaf, measurement) pair with the track
  score, the log-likelihood ratio of target against false alarm [10]:
  `L_k = L_{k−1} + ln(P_D / λ(z)) − ½ νᵀS⁻¹ν − ½ ln|2πS|` on a hit,
  `+ ln(1 − P_D P_G)` on a miss, λ(z) the clutter intensity at the plot
  (`Scan.clutter_intensity`). Trees sharing a gated measurement, or a global hypothesis, are
  one cluster. For each cluster it extends each of its `k_best` global hypotheses by that
  hypothesis's `k_best` cheapest assignments (Murty [11], [12], over `linear_sum_assignment`),
  keeps the `k_best` with the highest summed score, and prunes by N-scan [9]: a hypothesis
  that disagrees with the best one on any tree's branch `n_scan` scans back is dropped, and a
  leaf in no hypothesis goes. A tree whose one leaf is in every hypothesis of its cluster
  leaves it as a cluster of its own. Under `assignment = "confirmed_first"` (V2, V4) only
  confirmed branches' gates join clusters and block births, as GNN's confirmed tracks do; a
  tentative tree takes the plots in no confirmed gate alone, and a plot no best hypothesis
  took starts a tree. Without it, V4's tentative trees, born with 150 m/s of velocity spread,
  chain through shared clutter plots into one cluster of 1787 leaves in three rotations.
- The cost matrix is GNN's: −ln(g/λ) per gated pair and the finite non-assignment cost
  ln(P_D / (1 − P_D P_G)), the score's break-even, so a hypothesis's cost is its score's
  negative up to a constant per row. `murty_k_best` lives in `tracking/association.py` beside
  GNN, so the k = 1 case is GNN (A4).
- New trees start from unused measurements where `Scan.clutter_intensity` is at most the
  birth limit, as every tracker's births do, scored from zero; a cue (`Tracker.seed`) starts at
  the confirmation score. MHT confirms by score at the Wald threshold `ln((1 − β)/α)` = 9.1050
  (α = 1e-4, β = 0.1) and deletes a branch when its score falls `−ln(β/(1 − α))` = 2.3025
  below the highest along it; a tree goes with its last branch. Score replaces M-of-N,
  existence, the coast limit and covariance deletion on this path. `n_scan = 3` (14.4 s of
  deferral at 4.8 s), `k_best = 10` global hypotheses per cluster.
- What it costs, recorded: the estimate at scan k is the best hypothesis at k and is not
  rewritten if MHT later prefers another (`tracks.csv` is causal, A19); a tree holds at most
  `k_best` leaves, one per hypothesis of its cluster, and its deferred branches at most
  `k_best × n_scan` nodes; the leaf count is a reported intermediate.

### 2.3 PMBM (D30)

PMBM's association hypotheses, existence and birth are part of its density, so there is no
associator, initiator or lifecycle to swap: it is **a second tracker**, `PmbmTracker`, in
`tracking/rfs.py`. It shares the motion and measurement models, `innovate`, `murty_k_best`,
`Scan.clutter_intensity`, the initiator that places a first detection, and the metrics; it owns
its Poisson undetected component, Bernoulli tracks and global hypotheses [13]:

- Prediction (§III-D, (41)): each Bernoulli's existence times `p_S` = 0.995 a scan (the
  JIPDA survival of `ExistencePolicy`), its density by the motion model; the undetected
  intensity is uniform over coverage at the fixed point of `λ_u ← p_S (1 − P_D) λ_u + λ_b`,
  λ_b the birth intensity 0.05 a scan over the coverage volume (2.04e-7 per m rad at 1-40 km),
  so every sector batch sees it alike, and zero where the clutter passes the birth limit
  every tracker shares (V4's clutter map starts at 0.02 per m rad, where every plot would
  open a track of existence 1.02e-5, just over the pruning threshold).
- Update: a track's detection hypothesis weighs `r P_D g` and has existence 1 ((30)); its
  misdetection weighs `1 − r P_D P_G` and keeps `r (1 − P_D P_G) / (1 − r P_D P_G)` ((31),
  gated); each measurement opens a new track weighing `λ(z) + e(z)`, `e(z) = P_D λ_u`, with
  existence `e/(λ(z) + e)` ((21)-(24), (43), (45)); a global hypothesis weighs the product of
  its tracks' ((35)). The k-best global hypotheses come from `murty_k_best` over the cost
  `−[ln W_ot, ln W_nt]` of §V-C3, ⌈`k_best` w_j⌉ from each prior hypothesis of weight w_j.
- Pruning (Algorithm 1): global hypotheses below weight 1e-4, then the `k_best` = 100 best per
  cluster; Bernoullis below existence 1e-5 or in no hypothesis.
- Report: each track's marginal Bernoulli, its hypotheses' mixture weighted by their global
  hypotheses' weights and moment-matched (the track-oriented projection of Williams [14]
  §IV-A, (65)), is a track in `tracks.csv` where its existence passes 0.5, labelled by the
  measurement that opened it. The paper's estimators (§VI) report one global hypothesis's
  Bernoullis; A17 compares posteriors, and the marginal is the posterior's own mean, as
  JPDA's moment matching is. Measured at `d = 2` across range, `n_gate = 1`, 60 trials:
  P(resolved) 0.30 with estimator 1, 0.38 with the marginal, JPDA 0.42.
- Under `confirmed_first`, as for MHT, only Bernoullis of existence 0.5 or more join
  clusters.

Motion: one CV model, the configured `sigma_acceleration_mps2` (CV quiet, 0.5 m/s², on V2 and
V3); under `motion = "imm"` (V4) the agile mode's noise, M3's σ = 3.1623 m/s², since one model
must carry the turns. Williams [14] shows MHT and JIPDA approximate the same conjugate prior,
which makes PMBM the reference for their losses. GLMB [15] is named, not built.

### 2.4 Variants and outputs

`config/scenario_011_cst_sweep.toml` holds V3's scene at `d = 1` across range, `n_gate = 1`;
its sweep runs as `slow` Monte Carlo tests through the stage's code, two points in the
unmarked suite. `[tracking]` gains `association = "mht"` (`k_best`, `n_scan`,
`confirm_score`, `delete_drop`), PMBM's `birth_intensity_per_scan`, `prune_existence` and
`prune_weight`, `seed_truth` (V3's cues), and `trackers`, the further trackers a run scores
beside the configured one (`"gnn"`, `"jpda"`, `"jpda_star"`, `"set"`, `"jipda"`, `"mht"`,
`"pmbm"`); a run of PMBM alone lists `trackers = ["pmbm"]`. V2 and V4 gain the MHT and PMBM
rows: V2's trackers are GNN, JPDA, JPDA\*, Set-JPDA, JIPDA, MHT, PMBM; V4's are IMM with JIPDA,
GNN and MHT, and PMBM. The configured tracker's scores keep their names and `[verify]` checks;
another's carry its name as a prefix. `tracks.csv` gains `score` (MHT) and `tracker`;
`gospa.csv` gains `tracker`.

## 3. Acceptance criteria

| # | Criterion | Bound | Justification | Applies to |
| :--- | :--- | :--- | :--- | :--- |
| A4 | **MHT reduces to GNN.** `n_scan = 1`, `k_best = 1`: MHT's assignments and states equal GNN's with the finite non-assignment cost, on V2's two 30° crossings, both seeded at the truth with deletion off and one clutter intensity per scan | identical assignments; states `rtol = 1e-10` | Murty's first solution is the assignment | unit |
| A10 | **PMBM reduces to the KF.** One target, Pd = 1, no clutter, no birth after scan 0 | `rtol = 1e-9` | The single-Bernoulli case | unit |
| A15 | **GNN swaps; MHT does not.** V2, 10° across-range crossing, 200 trials, both seeded at the truth before the crossing with deletion off: swap rate GNN ≥ 0.2; MHT ≤ 0.05; JPDA's reported, not bounded | as stated, the bounds checked against a prototype first | A 10° pair closing across the beam spends sixteen scans inside one resolution cell (2 × 488.7 m at 12.55 m/s is 16.2 scans) | V2, `slow` |
| A16 (MHT) | **MHT resolves where JPDA coalesces.** V3, `d` = 1 across range, `n_gate` = 1: P(resolved) for MHT ≥ P(resolved) for JPDA | as stated | JPDA coalesces where hypotheses are deferred. Measured, 60 trials: every tracker 0.00, since 61 % of scans merge the pair into one plot 244 m from each aircraft, outside both 99 % gates, and n_gate = 1 clutter draws both tracks off; the clause holds and says little | V3, `slow` |
| A17 | **The reference line.** V3, every point: PMBM's P(resolved) ≥ every other tracker's, less 0.05 | as stated | [14]: the others approximate PMBM's posterior; a tracker beating it materially is a bug in one of them | V3, `slow` |
| A19 | **Causal output.** For MHT, the `tracks.csv` row at scan k equals the tracker's state at scan k, recorded live | identical | D29 | V4 |

## 4. Limitations

PMBM runs one motion model, not the IMM: V3 has no manoeuvres, so it is a fair test of
association, and on V4 the comparison carries that caveat. MHT's reported estimate is causal,
never smoothed. Blackman [10], Murty [11] and Cox and Miller [12] are cited for their results
(the track score, the ranked-assignment partition, its use in tracking), not for an equation
number: the derivations above stand on their own.

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
        Aerosp. Electron. Syst.*, 54(4), pp. 1883–1901, 2018; arXiv:1703.04264.
.. [14] J. L. Williams, "Marginal multi-Bernoulli filters: RFS derivation of MHT, JIPDA, and
        association-based MeMBer," *IEEE Trans. Aerosp. Electron. Syst.*, 51(3),
        pp. 1664–1687, 2015.
.. [15] B.-T. Vo and B.-N. Vo, "Labeled random finite sets and multi-object conjugate priors,"
        *IEEE Trans. Signal Process.*, 61(13), pp. 3460–3475, 2013.
