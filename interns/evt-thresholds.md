# Task 8: Thresholds from the tail

**Time:** about two working days. Stop when you reach it, and tell us what you would have done
next.

**AI tools:** allowed, and we expect you to use them. Tell us how you used them (see
[Deliverables](#deliverables)).

**Afterwards:** see [the follow-up call](README.md#the-follow-up-call).

Read [the intern README](README.md) and the [AI-for-radar background](ai-for-radar.md) first.
The background lists what the library does not have, and the rules every comparison must keep.

## Background

Every detector needs a threshold, and every fair comparison in these tasks sets thresholds at a
fixed false-alarm probability. That is easy when theory gives the answer. For cell-averaging CFAR
(CA-CFAR) in complex Gaussian noise it does: `cfar_threshold_factor` in
`radar_forge.core.detection` computes the factor exactly, for any Pfa. It is hard when it does
not:

- in clutter that is not Gaussian;
- for a learned detector;
- for a track-before-detect score, which is a maximum over very many trajectories.

Then the threshold has to be estimated from noise-only or clutter-only data. Counting
exceedances needs many times 1/Pfa samples: at Pfa = 1e-6, many millions of independent cells.

**Extreme value theory** (EVT) offers a shortcut. Above a high enough level, the exceedances of
a wide class of distributions approach a generalised Pareto distribution (peaks over threshold).
Maxima of large blocks approach a generalised extreme value distribution. Fit the tail and
extrapolate to the Pfa you need, from far fewer samples.

The shortcut has costs, and this task is about them:

- you must choose where the tail starts;
- the fit assumes independent samples, and neighbouring CFAR outputs share training cells;
- an extrapolated threshold is an *estimate*, with its own uncertainty;
- a threshold fitted to one clutter level is not constant-false-alarm-rate unless the statistic
  you fit is normalised by the level.

Nothing EVT-based is implemented in `radar-forge`, and there is no clutter. You will write both in
your own script. Leave `src/` alone. SciPy, a core dependency, has the distributions you need.

## Setup

- Use this repository at the latest `main`, and record the commit hash you used in your report.
- Run `uv sync --extra dev --extra viz`. Add `--extra ml` only for the optional flow.
- Useful entry points:
  - `radar_forge.core.detection`: `cfar_threshold_factor`, `cfar_probability_of_false_alarm`,
    `cfar_noise_estimate_w`, `cfar_detect`, `cfar_detect_2d`. Read the module docstring; it
    explains what each variant assumes.
  - `radar_forge.core.signal.thermal_noise`: complex Gaussian noise of any shape.
  - `radar_forge.pipelines.scenarios.form_range_doppler_map`, and the
    [noise-only maps recipe](track-before-detect.md#noise-only-maps), if you want maps with
    scenario 003's exact shape.

## The questions

1. **Predict first.** You want a CA-CFAR threshold for Pfa = 1e-6, with 16 training cells each
   side, set from data and not from the formula. Before you run anything, estimate two numbers:
   - how many independent noise-only samples the empirical quantile needs, so that the Pfa it
     actually achieves is within a factor of two of 1e-6 nine times out of ten;
   - how many a peaks-over-threshold fit needs for the same.

   Say what would refute each.
2. **Gaussian noise, where you know the answer.** Here `cfar_probability_of_false_alarm` tells
   you the *achieved* Pfa of any threshold exactly, so you can score an estimate without
   simulating 1e8 cells. For a range of sample sizes, plot the distribution of achieved Pfa for:
   - the empirical quantile;
   - a generalised Pareto fit (state how you choose the tail level);
   - the analytic factor.

   Report each estimate's own confidence interval, and check it covers the truth as often as it
   claims.
3. **Clutter, where you do not.** Generate compound-Gaussian clutter in your own script, for
   example with a gamma texture (K-distributed amplitude). Justify the shape parameter from a
   cited source on real radar data. Get the true threshold from one very long run, and say how
   long is long enough. Then:
   - show how far the Gaussian formula's Pfa is off;
   - show how few samples an EVT fit needs to do better.
4. **Maxima.** A track-before-detect detector, or any detector that declares "a target
   somewhere in this map", thresholds a maximum, not a cell. Set a per-map false-alarm threshold
   for the maximum of a CA-CFAR statistic over a whole map, from block maxima and from peaks over
   threshold. Compare the two and say which you would trust.
5. **Shift and dependence.** Change one thing the fit assumed:
   - give the texture spatial correlation, so that neighbouring cells are no longer
     independent; or
   - change the clutter level or shape between calibration and test.

   What happens to the achieved Pfa? What would you do about it, for example declustering, or
   fitting a statistic normalised by the local level?

## Deliverables

- **A short report** (at most 4 pages plus figures; PDF or Markdown) answering the five questions.
  Lead with your findings, not your process. Start with one line on why you chose this task.
- **One script** (for example `scripts/studies/evt_thresholds.py`) that regenerates every figure
  and number in the report from a clean checkout. Fix your random seeds. Do not commit generated
  data.
- **An AI-use note** (half a page):
  - what you used AI tools for;
  - what you checked by hand, and how;
  - at least one place where the tool was wrong or misleading, and how you caught it;
  - which of your sources you found through an AI tool, and how you checked them (see
    [Sources](ai-for-radar.md#sources)).
- **Bugs and suggestions**, as described in
  [the intern README](README.md#bugs-and-suggestions).

Send the report, a link to your branch, and the note.

## What we are looking for

Honest uncertainty about a small probability: every threshold comes with an interval, and
every interval is checked against the truth where the truth is known. "EVT saved a factor of 50
in samples in Gaussian noise but was biased in correlated clutter" is the kind of answer we
want. A careful answer to questions 1–3 beats a rushed answer to all five.

### Optional, if you have time

Fit a normalizing flow to the clutter amplitude (this needs `--extra ml`), and read its
threshold at Pfa = 1e-6 from the learned density. Find out what the literature says about how
well flows represent heavy tails, then check it: at the same number of samples, is the flow's
threshold closer to the truth than the generalised Pareto fit's, or further?
