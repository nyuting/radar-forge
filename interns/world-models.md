# Task 5: Do world models help a radar?

**Time:** about three working days. Stop when you reach it, and tell us what you would have done
next.

**AI tools:** allowed, and we expect you to use them. Tell us how you used them (see
[Deliverables](#deliverables)).

**Afterwards:** see [the follow-up call](README.md#the-follow-up-call).

Read [the intern README](README.md) first. It covers setup, house rules and how to submit.

This task is exploratory on purpose. Everyone answers the same two framing questions, then picks
**one** of several directions: the one you find most technically interesting. We would rather
see one direction done properly than a tour of all of them.

## Background

**AI, machine learning and deep learning are not currently implemented in `radar-forge`.** There
is no learned model, no training code and no PyTorch import anywhere in `src/`. The project
`README.md` and [`spec/structure.md`](../spec/structure.md) describe a `pipelines/datasets.py`
with PyTorch wrappers; it does not exist yet. There is no clutter either: the background is
thermal noise only. You will build what you need yourself, in your own script, on top of what the
library already produces. That is the point of this task. As always, leave `src/` alone.

### What a world model is

A **world model** is a model of how an environment evolves, which an agent uses to predict,
filter and plan without acting in the real environment at every step. Most share three parts:

1. an **encoder** that maps an observation to a compact latent state;
2. a **transition** that moves the latent state forward in time, possibly conditioned on an
   action;
3. a **decoder** that predicts the next observation, or a quantity you care about, from the
   latent state.

The learned versions in the literature differ in what the latent state is (continuous or
discrete, deterministic or stochastic), how the transition is parameterised, and what the
model is trained to predict. Find and read the main ones yourself.

### You already have one

The tracker in scenario 003 is a world model. It has a latent state (range and range rate), a
transition (a constant-velocity motion model with process noise), an observation model (what the
range-Doppler map measures), and calibrated uncertainty on all of them. It is linear and
Gaussian, and it was written by hand rather than learned. That makes the honest question not
"can a world model do radar?" but:

> **Where does a *learned* world model beat the one we already have, and at what cost?**

Papers that claim learned models help often compare against a weak or mistuned baseline, or at
a different false-alarm rate. Your job is to find out on a problem we control.

## Setup

- Use this repository at the latest `main`, and record the commit hash you used in your report.
- This task needs PyTorch, which is in the `ml` extra. `uv sync` removes every extra you do not
  name, so name them all:

  ```bash
  uv sync --extra dev --extra viz --extra ml
  ```

- Keep models small. They should train in minutes on a laptop CPU. A large model that wins is a
  weaker result than a small one that wins.
- `uv run python scripts/run_scenario.py scenarios/scenario_003_tracking.toml --out out/s1 --no-iq`
  runs the classical chain end to end. Read that script to see how the pieces fit together. Don't
  commit `out/`.
- Useful entry points:
  - `radar_forge.pipelines.scenarios`: `load_scenario`, `iterate_frames`,
    `form_range_doppler_map`. Each frame carries its IQ and the truth (`frame.range_m`,
    `frame.radial_velocity_mps`, unfolded). The map has shape `(n_doppler, n_range)`.
  - `radar_forge.pipelines.tracking`: `ScenarioTracker.from_scenario`, `frame_detections`,
    `score_primary_track`. This is the classical baseline.
  - `radar_forge.pipelines.trajectories`: `load_flight_csv`, `resample`, `to_radar_frame`.
  - `radar_forge.core.tracking.motion.MotionModel`: the interface a motion model follows.
    `radar_forge.core.tracking.tracker.build_tracker` and the `UKF` accept **any** object that
    follows it, so you can put your own transition into the library's tracker from your script.
  - `radar_forge.core.signal.thermal_noise`: noise-only data, as in the
    [track-before-detect background](track-before-detect.md#noise-only-maps).
- **Data.** `data/flight_coordinates.csv` is about 16,500 s of real ADS-B fixes for one light
  aircraft. Scenario 003 uses only 120 s of it (from 663 s). The rest is yours for training,
  validation and testing; decide how to split it and justify the split. The fixes are 2–8 s
  apart and noisy, which is why the simulated target's range rate jumps by several m/s between
  frames. Read the comment above `sigma_accel_mps2` in `scenarios/scenario_003_tracking.toml`.

## The questions

Everyone answers questions 1 and 2. Then choose one direction for question 3.

1. **Define it.** Write a formal definition of a world model for a radar: the state, the action,
   the transition, the observation and what is learned. Say what an *action* is for a radar. Say
   how a world model differs from plain autoregressive next-frame prediction. Place the scenario
   003 Kalman filter inside your definition, term by term. Half a page at most.
2. **Predict before you build.** For the direction you choose, state one falsifiable hypothesis
   with a number in it, and say what result would refute it. Write it down before you run
   anything. Being wrong is fine; we are interested in what you learn from it.
3. **Build and test one direction.**

   - **A. A learned motion model.** Learn the transition, and process noise that is honestly
     calibrated, from trajectory data the scenario 003 window does not use. Put it into the
     library's UKF through `MotionModel`, and compare it with the constant-velocity and
     constant-acceleration models on the same window. Report range error and track continuity,
     and check whether the filter's covariance is consistent with its errors (for example with
     the normalised estimation error squared). Is a learned model with a wrong covariance better
     or worse than a simple one with a right covariance?
   - **B. Predicting the map.** Predict the next range-Doppler map from the past few. The
     physics baseline is to shift the last map by the tracker's predicted change in range and
     Doppler. Where does a learned predictor beat it, and where does it break physics? Check, for
     example, energy (Parseval), the Doppler fold at the edge of the unambiguous interval, and
     whether the predicted range change agrees with the predicted range rate. Say whether you
     model the complex map or its power, and what you lose by the choice.
   - **C. Clutter that evolves.** The library has none, so generate it in your own script: clutter
     that is correlated in space and time, for example compound-Gaussian with a slowly varying
     texture. Justify the parameters from a cited source on real radar data. Learn how it
     evolves, and use the prediction to set a detection threshold or to whiten the map. Compare
     detection with CA-CFAR **at the same false-alarm probability**, measured on held-out seeds.
   - **D. Planning with a world model.** This is the use world models were invented for. The
     scenarios offer three waveforms with different weaknesses: the low-PRF FMCW of
     `scenario_003_tracking.toml` folds velocity; the pulsed medium-PRF of
     `scenario_003_ukf_pulsed_medium_prf.toml` folds range; the dual-PRF FMCW of
     `scenario_003_tracking_dual_prf.toml` resolves velocity but splits its pulses between two
     bursts. Treat the waveform as the action. Use a world model (learned, or the tracker's own)
     to choose the waveform frame by frame, and compare with always using one. Count what each
     action costs in time and energy on target, so the comparison is fair.
   - **E. Your own.** You may propose a different direction, if it uses the real library and has
     a classical baseline. Agree it with us before you start.

   Whichever you choose:
   - **Baseline.** Do your best to make the classical baseline good. Tune it with the same care
     and on the same data as the learned model.
   - **Separation.** Keep training, tuning and test data separate: trajectory windows, and random
     seeds.
   - **Uncertainty.** Say which uncertainty your model represents: aleatoric (noise in the world)
     or epistemic (what the model does not know). Check whether it is calibrated.
   - **Shift.** Test under at least one change from the training conditions, such as a different
     part of the flight, a different SNR or different clutter parameters. How fast does any
     advantage disappear?
   - **Cost.** Report training time, and operations and memory per frame at inference, against
     the baseline.

4. **What next.** Half a page: if you had four more weeks, what would you do, what would you
   expect to find, and what result would change your mind?

## Deliverables

- **A short report** (at most 4 pages plus figures; PDF or Markdown) answering the four
  questions. Lead with your findings, not your process. Start with one line on which direction
  you chose and why.
- **One script** (for example `scripts/studies/world_model.py`) that regenerates every figure and
  number in the report from a clean checkout, including training. Fix your random seeds. Do not
  commit generated data or trained weights.
- **An AI-use note** (half a page):
  - what you used AI tools for;
  - what you checked by hand, and how;
  - at least one place where the tool was wrong or misleading, and how you caught it;
  - which of your sources you found through an AI tool, and how you checked them.
- **Bugs and suggestions**, as described in
  [the intern README](README.md#bugs-and-suggestions).

Send the report, a link to your branch, and the note.

## Sources

The rule is the one in the [track-before-detect background](track-before-detect.md#sources).
There is no reading list. Cite a source for every architecture, loss, clutter model and metric
you use, in your report and in the `References` section of the docstring that implements it.
Read what you cite: in the follow-up call we will ask what your sources say. Be careful with
this topic in particular. "World model" is used loosely, and an AI tool will happily give you
confident but wrong claims about it, including invented citations.

## What we are looking for

Clear thinking about what a world model is and is not, a fair comparison against a classical
baseline you did your best to make good, and physics you can defend. "The learned model did not
help, and here is why" is a strong answer if it is true. A careful answer to one direction beats
a rushed answer to several. Every number should be reproducible and every claim should come with
a reason.

### Optional, if you have time

Take the part of your learned model that helped most and ask whether a classical model could
capture it instead: a better motion model, a fitted clutter distribution, a smarter fixed
waveform schedule. How much of the learned model's advantage does that recover?
