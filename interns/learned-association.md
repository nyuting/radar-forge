# Task 9: Learned data association

**Time:** about three working days. Stop when you reach it, and tell us what you would have done
next.

**AI tools:** allowed, and we expect you to use them. Tell us how you used them (see
[Deliverables](#deliverables)).

**Afterwards:** see [the follow-up call](README.md#the-follow-up-call).

Read [the intern README](README.md) and the [AI-for-radar background](ai-for-radar.md) first.
The background lists what the library does not have, and the rules every comparison must keep.

## Background

**A warning about names.** In tracking, "GNN" usually means *global nearest neighbour*, the
assignment rule this library uses. In machine learning it means *graph neural network*. This
task is about whether the second can beat the first. Write both out in full in your report.

Data association decides which measurement belongs to which track. The library's tracker does
it in two steps (read the module docstring of `radar_forge.core.tracking.association`):

1. a chi-square gate rules out pairs that are too far apart;
2. `GlobalNearestNeighbour` picks the set of pairs with the smallest total NIS, with each track
   and each measurement used at most once.

This is fast and works well when targets are far apart. It fails in a known way when two targets
cross, or when false alarms fall inside a gate: one wrong hard decision can swap two tracks'
identities or pull a track onto clutter. Classical methods that share or defer the decision
(PDA, JPDA, MHT) are not implemented in the library.

A **graph neural network** (here, a small message-passing network) can score track–measurement
pairs using context that a per-pair NIS cannot see: the other tracks nearby, the other
measurements in the gate, and each track's recent history. The question is whether that context
is worth anything, at the same false-track rate.

The library has one target per scenario and no clutter. You will build a multi-target,
false-alarm scenario in your own script. Leave `src/` alone.

## Setup

- Use this repository at the latest `main`, and record the commit hash you used in your report.
- Run `uv sync --extra dev --extra viz --extra ml`.
- **Work at the measurement level.** You do not need IQ or range-Doppler maps for this task.
  Generate:
  - target trajectories, some of which cross or run close together;
  - noisy position measurements with a detection probability below one;
  - false alarms, uniformly distributed in the surveillance region with a Poisson number per
    scan.

  Say how you chose each parameter, and keep them in one place in your script.
- Useful entry points:
  - `radar_forge.core.tracking.tracker`:
    - `build_tracker_enu` builds a tracker for x/y/z position measurements;
    - `Tracker` takes a `gate` and an `associator` in its constructor;
    - `Tracker.process` takes one `MeasurementBatch` per scan.
  - `radar_forge.core.tracking.association`: `ChiSquareGate`, `GlobalNearestNeighbour`,
    `NearestNeighbour`, and the `Associator` protocol. Any object with an
    `associate(costs)` method can be passed to `Tracker`.
  - `radar_forge.core.tracking.measurement_models`: `Measurement`, `MeasurementBatch`,
    `CartesianPosition`.
- **The protocol is narrow on purpose.** An `Associator` sees only the cost matrix, never the
  tracks or the measurements. A network that needs more context than a cost matrix does not fit
  through it. Decide how to handle that without changing `src/`. One way is to run your own scan
  loop around the library's filters. If you conclude the protocol should change, say how in your
  report. That is a finding, not a failure.

## The questions

1. **Predict first.** For two targets crossing at a fixed angle, with a fixed false-alarm
   density, estimate the probability that global nearest neighbour swaps their identities.
   Estimate it as a function of measurement noise, or of crossing angle. Then state one
   falsifiable hypothesis, with a number in it, about how much a learned associator can reduce
   that probability. Write both down before you build anything.
2. **Baseline.** Run the library's tracker with global nearest neighbour on your scenarios, and
   tune it honestly: process noise, gate probability, and confirmation and deletion rules. Score
   it with:
   - a multi-target metric such as OSPA or GOSPA (cite the definition you use);
   - identity swaps;
   - track continuity;
   - false-track rate.
3. **Learn.** Build a small message-passing network on a graph of tracks and measurements in
   each scan. Train it to score track–measurement pairs, and turn the scores into an assignment.
   Say what it is trained on and how you get the labels. Keep training, tuning and test seeds
   separate.
4. **Compare fairly.** Tune both trackers to the **same** false-track rate, on held-out seeds.
   Then compare identity swaps and the multi-target metric, against crossing angle and
   false-alarm density. Where does the network help, if anywhere?
5. **Shift and cost.** Test with more targets, a higher false-alarm density or a different
   detection probability than you trained on. How fast does any advantage disappear? Report
   operations and memory per scan against global nearest neighbour, and how each grows with the
   number of tracks and measurements.

## Deliverables

- **A short report** (at most 4 pages plus figures; PDF or Markdown) answering the five questions.
  Lead with your findings, not your process. Start with one line on why you chose this task.
- **One script** (for example `scripts/studies/learned_association.py`) that regenerates every
  figure and number in the report from a clean checkout, including training. Fix your random
  seeds. Do not commit generated data or trained weights.
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

A fair comparison above all: one false-track rate, held-out seeds, and global nearest neighbour
tuned as well as you could tune it. "The network only helped below 20° crossing angles, and a
better-tuned gate recovered most of that" is a strong answer if it is true. A careful answer to
questions 1–4 beats a rushed answer to all five.

### Optional, if you have time

Ask how much of the network's advantage a classical method recovers. Implement probabilistic
data association for the crossing case, or a simple track-oriented MHT with a few hypotheses,
and compare at the same false-track rate.
