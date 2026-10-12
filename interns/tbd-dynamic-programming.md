# Task 2: How many dB does track-before-detect buy?

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

Published dynamic-programming TBD results often quote a gain of "a few dB" over single-frame
detection. A number like that means nothing without the number of frames, the false-track rate,
the target's motion and the reference it is measured against. Your job is to produce that
number for one real radar, scenario 003, with all four stated, and to say where it comes from.

## The questions

1. **Bounds first.** Before you write a TBD detector, derive the
   [floor and the ceiling](track-before-detect.md#the-bounds) analytically, as Pd against SNR at a
   false-track probability you choose and justify, for K = 1, 5, 10 and 20 frames. Plot them.
   How much could TBD buy at each K, at most?
2. **Build** a dynamic-programming TBD over the low-SNR maps.
   - Choose the transition neighbourhood (how far a trajectory may move between frames, in range
     and in Doppler) from what the target actually does. Measure that from the truth. Say what
     happens to a target that moves outside it.
   - Compare two per-cell scores: raw power, and the log-likelihood ratio for a non-fluctuating
     target in complex Gaussian noise.
3. **Set the threshold.** For your chosen false-track probability, set the TBD threshold from
   noise-only runs. Show that the per-cell CFAR threshold would give the wrong answer, and by how
   much. Say how you estimate a small tail probability from a finite number of runs, and how
   confident you are in the threshold.
4. **Measure.** Estimate Pd against SNR, and against K, on the real scenario. Report TBD's gain
   in dB over the floor at Pd = 0.5 and Pd = 0.9, and its loss in dB below the ceiling.
   Explain that loss: how much of it is the cost of not knowing the trajectory, and how does it
   change with the size of the neighbourhood? Show error bars that mean something.
5. **Recover the track.** At a power where TBD detects and the DBT chain struggles, compare the
   range error of TBD's best trajectory with the scenario 003 tracker's
   (`ScenarioTracker.from_scenario`, `score_primary_track` in `radar_forge.pipelines.tracking`).
   Which would you trust, and for what?
6. **Cost.** Count the operations and memory per frame as functions of the map size, the
   neighbourhood size and K. What would have to change to run this once a second on embedded
   radar hardware?

## Deliverables

- **A short report** (at most 4 pages plus figures; PDF or Markdown) answering the six questions.
  Lead with your findings, not your process. Start with one line on why you chose this task.
- **One script** (for example `scripts/studies/tbd_dp.py`) that regenerates every figure and
  number in the report from a clean checkout. Do not commit generated data.
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

A threshold you can defend, a gain quoted against the right reference, and an honest account of
the gap to the ceiling. A careful answer to questions 1–4 beats a rushed answer to all six. Every
method you use has a cited source, and you can explain it.

### Optional, if you have time

Implement a Hough-transform TBD on range against time, and compare it with your dynamic-
programming detector at the same false-track probability. Where does each win?
