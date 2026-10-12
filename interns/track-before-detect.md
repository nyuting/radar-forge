# Track-before-detect: background for tasks 2–4

This page is shared background for three tasks. It is not a task itself. Read it, pick **one** of
the [three tasks](#pick-one-task), and read [the intern README](README.md) for setup, house
rules and how to submit.

**Track-before-detect is not currently implemented in `radar-forge`.** There is no TBD code in
`src/` and the specification does not mention it. You will build it yourself, in your own script,
on top of the range-Doppler maps the library already produces. That is the point of these tasks.
As always, leave `src/` alone.

## What TBD is

The scenario 003 chain is **detect-before-track** (DBT). Every frame it thresholds the
range-Doppler map with CFAR, and hands the tracker only the cells that crossed. Anything below
the threshold is gone for good. A target too dim to cross the threshold in any single frame is
never tracked, even though its energy is there, frame after frame, in cells that line up along a
plausible trajectory.

**Track-before-detect** keeps the unthresholded data. It accumulates evidence along candidate
trajectories over several frames, and declares a target only when a whole trajectory's evidence
is strong enough. The gain comes from integrating K frames of the target's energy. The price is
that you do not know the trajectory: you search over very many of them, and the best of many
noise-only trajectories is much stronger than a typical one.

There are three classical families:

- **Dynamic programming** (Viterbi-style): a recursion over the map that keeps, for every cell,
  the best score of any trajectory ending there.
- **Hough transform**: maps range–time data into a space of straight lines and integrates along
  each line.
- **Particle filter with a target-existence variable**: a Bayesian filter whose state includes
  whether a target is there at all, updated directly from the unthresholded map.

Read about them yourself; [Sources](#sources) says how we expect you to cite what you read.

## Pick one task

| # | Task | The question | Time |
| :--- | :--- | :--- | :--- |
| 2 | [How many dB does TBD buy?](tbd-dynamic-programming.md) | Build dynamic-programming TBD on scenario 003's maps. How far below the single-frame threshold can it go, and how close is that to the best any multi-frame detector could do? | about 2 days |
| 3 | [Particle-filter TBD](tbd-particle-filter.md) | Build a particle filter that decides target existence from unthresholded maps. How quickly and reliably does it declare a dim target, and what does it cost? | about 2 days |
| 4 | [Where can learning beat the textbook?](tbd-learned.md) | Put a learned component into a TBD detector. Where, if anywhere, does it beat the classical one at the same false-track rate? | about 2–3 days |

Pick the one you find most technically interesting, and say why in one line at the top of your
report. If you have been told which one to do, do that one.

## Common ground

All three tasks use the same data and the same definitions, so results can be compared.

### Getting low-SNR maps

At its default 100 W, scenario 003's target is so strong that TBD has nothing to do. Lower the
power first:

- Copy `scenarios/scenario_003_tracking.toml` and change `transmit_power_w` in the copy. The
  `[trajectory] path` is resolved relative to the TOML's own folder, so either keep the copy next
  to the original or make the path absolute. Working out what power puts the target where you
  need it is part of your task.
- Load it with `radar_forge.pipelines.scenarios.load_scenario` and step through
  `iterate_frames`. For each frame, `form_range_doppler_map(frame.iq[0], scenario.bursts[0])`
  gives the complex map `rd_map`, of shape `(n_doppler, n_range)`, with its `range_axis_m`
  and `velocity_axis_mps`. Power is `abs(rd_map) ** 2`. This map has not been thresholded.
- Frames share one random generator, so consume them in order. Skipping or reordering frames
  changes the noise.
- The map's power is an unnormalised FFT output, not a calibrated power in watts. Normalise by an
  estimate of the noise level, and say which estimate you used.
- Each frame carries the truth: `frame.range_m` and `frame.radial_velocity_mps` (unfolded,
  positive closing). The map's velocity axis folds at its edges; the waveform's unambiguous
  interval is something you should derive and check, not assume.

### Noise-only maps

Setting a TBD threshold takes many noise-only runs. You do not need the full scenario for them:
`radar_forge.core.signal.thermal_noise((n_pulses, n_samples), burst.noise_power_w, rng=...)`
passed through the same `form_range_doppler_map` gives a noise-only map with the same statistics
as the real one. Check that claim yourself before you rely on it.

### Definitions

Use these, or say exactly where and why you depart from them.

- **SNR** is the per-cell signal-to-noise ratio in the range-Doppler map, after the coherent
  integration of one frame. Say how you compute it, and whether it includes straddle loss: the
  target rarely sits at the centre of a range and Doppler bin.
- **K** is the number of frames a TBD decision uses.
- **False-track probability** is the probability that a TBD detector, run over K noise-only
  frames, declares a target anywhere in the map. This replaces the per-cell false-alarm
  probability of CFAR, which does not carry over: a per-cell probability of 1e-3 on a map of this
  size is hundreds of false alarms per frame, and a TBD detector does not decide cell by cell.
- **Detection** means the detector declares a target, and the declared trajectory's final cell is
  within one range bin and one velocity bin of the truth. A declaration anywhere else is a false
  track, even if a target is present.

### The bounds

Quote every TBD result against two bounds at the same false-track probability:

- **The floor: single-frame detection.** What a per-frame threshold achieves on one map.
- **The ceiling: known-trajectory integration.** If you knew exactly which cell the target
  occupied in each of the K frames, you could sum those K cells' power and threshold the sum.
  For a non-fluctuating target in complex Gaussian noise, the statistics of that sum are known in
  closed form. No detector that does not know the trajectory can beat it.

The gap between floor and ceiling is the most any TBD detector can buy. Where yours lands inside
it, and why, is the interesting part.

### Things that will trip you up

- **SNR versus SCR.** There is no clutter in the library: the background is thermal noise only.
  Anything you write about clutter is about clutter you have added yourself.
- **The threshold.** A TBD score is a maximum over very many trajectories. Its noise-only
  distribution is not the per-cell one, and you will need to estimate its tail at small
  probabilities from a finite number of runs. Say how, and how confident you are.
- **Velocity folding.** The target's true velocity is well outside the map's unambiguous
  interval. Its Doppler cell wraps around.
- **How far the target moves.** Measure, from the truth, how far the target moves in range and in
  Doppler between frames, before you design anything that depends on it. Do not assume both are
  small.

## Sources

There is no reading list. Find your own sources, and cite one for every algorithm, threshold
method and model you use: the paper or book, and the section, equation or page. Put the citation
in your report, and in the `References` section of the docstring of the function that implements
it, as the [project style](../docs/conventions/style.md) requires.

Read what you cite. In the follow-up call we will pick some of your citations and ask what the
source says, where your implementation follows it, and where it departs from it. A citation you
cannot discuss, or one an AI tool invented, counts against you more than no citation at all. Say
in your AI-use note which sources you found through an AI tool and how you checked them.
