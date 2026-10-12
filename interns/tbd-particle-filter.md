# Task 3: Particle-filter track-before-detect

**Time:** about two working days. Stop when you reach it, and tell us what you would have done
next.

**AI tools:** allowed, and we expect you to use them. Tell us how you used them (see
[Deliverables](#deliverables)).

**Afterwards:** see [the follow-up call](README.md#the-follow-up-call).

Read [the intern README](README.md) and the
[track-before-detect background](track-before-detect.md) first. The background defines SNR,
false-track probability, detection and the two bounds used below, and explains how to get
low-SNR maps out of scenario 003.

## Background

Track-before-detect is not currently implemented in `radar-forge`. You will write it, in your own
script, on top of the library's range-Doppler maps.

The scenario 003 tracker sees only CFAR detections, and decides that a target exists by counting
them (confirm after 4 hits in 5 frames). A particle-filter TBD instead carries the probability
that a target exists as part of its state, and updates it from the unthresholded map every frame.
It can declare a target before any single frame would have, and it says how sure it is. Your job
is to build one, measure how fast and how reliably it declares a dim target, and find out what
it costs.

## The questions

1. **Model.** Write down, before you code:
   - the state, including the existence variable;
   - how existence evolves between frames (target birth and death), and the probabilities you
     choose;
   - the measurement likelihood ratio, and how a target spreads its energy over neighbouring
     cells. Measure the spread from the map itself: the library applies no window, so look at the
     sidelobes. Say whether straddle loss matters.
   - the motion model, sized from what the target actually does in range and Doppler between
     frames. Measure that from the truth.
2. **Implement** it, citing the formulation you follow and saying where you depart from it.
3. **Declare.** Plot the existence probability against time on the real scenario, at a few
   powers. Set the declaration threshold for a false-track probability you choose and justify,
   from noise-only runs. Then measure Pd and the declaration delay (frames from the start until
   the target is declared) against SNR.
4. **Particles.** How do Pd, range and velocity error, and runtime change with the number of
   particles? Show particle depletion happening (for example through the effective sample size),
   and try one fix. Does it help, and what does it cost?
5. **Compare.** Put your Pd against SNR on the same plot as the
   [floor and the ceiling](track-before-detect.md#the-bounds). Then, at a power where your filter
   declares and the DBT chain struggles, compare its range error and its time to a confirmed
   track with the scenario 003 tracker's (`ScenarioTracker.from_scenario`,
   `score_primary_track` in `radar_forge.pipelines.tracking`).

## Deliverables

- **A short report** (at most 4 pages plus figures; PDF or Markdown) answering the five questions.
  Lead with your findings, not your process. Start with one line on why you chose this task.
- **One script** (for example `scripts/studies/tbd_pf.py`) that regenerates every figure and
  number in the report from a clean checkout. Fix your random seeds. Do not commit generated data.
- **An AI-use note** (half a page):
  - what you used AI tools for;
  - what you checked by hand, and how;
  - at least one place where the tool was wrong or misleading, and how you caught it;
  - which of your sources you found through an AI tool, and how you checked them (see
    [Sources](track-before-detect.md#sources)).
- **Bugs and suggestions**, as described in
  [the intern README](README.md#bugs-and-suggestions).

Send the report, a link to your branch, and the note.

## What we are looking for

A model you wrote down and can justify term by term, a declaration threshold set from evidence
rather than picked, and honest numbers on cost. A careful answer to questions 1–3 beats a rushed
answer to all five. Every method you use has a cited source, and you can explain it.

### Optional, if you have time

Recent papers propose diffusion models to generate or resample particles. Pick one, try its idea
in your filter, and say plainly whether it helps on this problem, and at what cost.
