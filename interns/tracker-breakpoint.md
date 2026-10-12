# Task 1: Where does the tracker break?

**Time:** about one working day. Stop when you reach it, and tell us what you would have done
next.

**AI tools:** allowed, and we expect you to use them. Tell us how you used them (see
[Deliverables](#deliverables)).

**Afterwards:** see [the follow-up call](README.md#the-follow-up-call).

Read [the intern README](README.md) first. It covers setup, house rules and how to submit.

## Background

`radar-forge` is a Python radar simulation and signal-processing library we use for teaching
and research. Scenario 003 simulates an X-band FMCW radar watching a light aircraft at about
20 km. Each frame, the chain:

1. synthesises the IQ data;
2. forms a range-Doppler map;
3. detects with cell-averaging CFAR (CA-CFAR; Pfa = 1e-5, 16 training cells each side);
4. tracks the detections with a Kalman filter that confirms, coasts and deletes tracks.

At the default 100 W transmit power the target is detected in every frame. So the parts of the
tracker that handle a *missed* detection have never been tested against a real one. The
specification says so itself, in
[`spec/scenario-003-tracking.md`](../spec/scenario-003-tracking.md) §13.1 and §14.9. Your job is
to find out what happens when the radar starts missing.

## Setup

- Use this repository at the latest `main`, and record the commit hash you used in your report.
- Run `uv sync --extra dev --extra viz`.
- `uv run python scripts/run_scenario.py scenarios/scenario_003_tracking.toml --out out/s1 --no-iq`
  runs the scenario end to end and writes detections, tracks and `metrics.csv` to `out/s1`. Don't
  commit `out/`. Read that script to see how the pieces fit together.
- Useful entry points:
  - `radar_forge.core.radar_equation.received_power_w`
  - `radar_forge.core.radar.Radar.noise_power_w`
  - `radar_forge.core.detection`
  - `radar_forge.pipelines.scenarios`: `load_scenario`, `iterate_frames`, `form_range_doppler_map`
  - `radar_forge.pipelines.tracking`: `frame_detections`, `ScenarioTracker.from_scenario`,
    `score_primary_track`

You should not need to change anything under `src/`. If you think you do, tell us why instead.

## The questions

1. **Predict.** Derive the single-frame probability of detection, Pd, as a function of
   `transmit_power_w` for `scenarios/scenario_003_tracking.toml`, from first principles (radar
   equation, processing gain, CFAR threshold). The target is non-fluctuating (Swerling 0). State
   every assumption and every loss you include or leave out.
2. **Measure.** Estimate Pd empirically by running the real detection chain over a range of
   transmit powers. Decide yourself what counts as "detected", and justify it. Use enough runs and
   seeds that your error bars mean something, and show them.
3. **Explain the gap.** Overlay prediction and measurement. Wherever they disagree, explain why,
   quantitatively if you can, and say how you would confirm your explanation.
4. **Consequences for tracking.** Over the same power range, measure how the tracker copes. At
   minimum report:
   - the fraction of frames with a confirmed track on the target;
   - the number of distinct track ids;
   - range error.

   Where does track maintenance fall apart, and how does that relate to Pd?
5. **A second waveform.** `scenarios/scenario_003_tracking_dual_prf.toml` resolves velocity with
   two bursts at different PRFs instead. **Before you run it**, write down what you expect its
   Pd and tracking curves to look like compared with question 1, and why. Then test that
   prediction. Being wrong is fine; we are interested in what you learn from it.
6. **Recommend.** If we wanted to change the scenario so that it exercises missed detections, what
   `transmit_power_w` (or other change) would you choose, and why? Do your results call anything in
   the tracker or the specification into question?

## Deliverables

- **A short report** (at most 4 pages plus figures; PDF or Markdown) answering the six questions.
  Lead with your findings, not your process.
- **One script** (for example `scripts/studies/pd_vs_power.py`) that regenerates every figure and
  number in the report from a clean checkout. Do not commit generated data.
- **An AI-use note** (half a page):
  - what you used AI tools for;
  - what you checked by hand, and how;
  - at least one place where the tool was wrong or misleading, and how you caught it.
- **Bugs and suggestions**, as described in
  [the intern README](README.md#bugs-and-suggestions). This task takes you through the radar
  equation, the CFAR detector, the scenario pipeline and the tracker, so you will read a lot of
  code. Tell us what you found.

Send the report, a link to your branch, and the note.

## What we are looking for

Correct physics, honest experiments, and clear reasoning about the gap between theory and
simulation. A careful answer to questions 1–4 beats a rushed answer to all six. We don't expect
polish; we do expect every number to be reproducible and every claim to come with a reason.

### Optional, if you have time

Check the library's CA-CFAR Pfa and Pd against a textbook result (for example Richards,
*Fundamentals of Radar Signal Processing*, §6.5) in a controlled noise-only experiment, separate
from the scenario.
