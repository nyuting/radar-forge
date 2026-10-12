# Task 4: Where can learning beat the textbook?

**Time:** about two to three working days. Stop when you reach it, and tell us what you would
have done next.

**AI tools:** allowed, and we expect you to use them. Tell us how you used them (see
[Deliverables](#deliverables)).

**Afterwards:** see [the follow-up call](README.md#the-follow-up-call).

Read [the intern README](README.md) and the
[track-before-detect background](track-before-detect.md) first. The background defines SNR,
false-track probability, detection and the two bounds used below, and explains how to get
low-SNR maps out of scenario 003.

## Background

Track-before-detect is not currently implemented in `radar-forge`, and neither is anything
learned. You will write both, in your own scripts, on top of the library's range-Doppler maps.

Many papers now propose neural networks inside TBD: learned cell scores, learned motion models,
learned association. Some claims are real and some are not, and the difference usually comes down
to one question: **was the learned detector compared with the classical one at the same
false-alarm rate, on a problem where the classical one's model is wrong?** Your job is to find
out, on a problem we control, where learning helps and where it cannot.

## Setup

This task needs PyTorch, which is in the `ml` extra. `uv sync` removes every extra you do not
name, so name them all:

```bash
uv sync --extra dev --extra viz --extra ml
```

Keep the network small. It should train in minutes on a laptop CPU. A large model that wins is a
weaker result than a small one that wins.

## The questions

1. **Predict first.** The library's target is non-fluctuating, and its background is complex
   Gaussian noise. For that case, can a learned per-cell score beat the log-likelihood ratio, at
   the same false-track probability? Argue it from detection theory before you run anything,
   then build a minimal dynamic-programming TBD (cite the formulation you follow), train a small
   network to replace its per-cell score, and test your argument.
2. **Break the model.** Add one thing the classical score gets wrong, generated in your own
   script (leave `src/` alone). Choose one:
   - heavy-tailed, spiky clutter, such as compound-Gaussian clutter with a gamma texture;
   - a target that fluctuates from frame to frame;
   - a target that manoeuvres harder than your transition neighbourhood allows.

   Justify your parameters from a cited source on real radar data, and show what the mismatch
   does to the classical detector: its false-track probability, its Pd, or both.
3. **Learn.** Train a small learned component (a cell score, a transition prior, or something
   else you can justify) to fix that mismatch. Say what it is trained on, and keep training,
   threshold-setting and test seeds separate.
4. **Compare fairly.** Set every detector's threshold for the **same** false-track probability,
   from noise-only (or clutter-only) runs on held-out seeds. Report the learned detector's gain in
   dB over the classical one, against the [floor and the ceiling](track-before-detect.md#the-bounds)
   where they apply. Then change the clutter or motion parameters it was trained on. How fast does
   its advantage disappear?
5. **Cost and deployment.** Training time, inference operations per frame, and memory, against
   the classical detector. Then half a page: what would stop you deploying the learned detector on
   embedded radar hardware (latency, fixed-point arithmetic, and how you would convince someone
   its false-alarm rate holds on data it has never seen)?

## Deliverables

- **A short report** (at most 4 pages plus figures; PDF or Markdown) answering the five questions.
  Lead with your findings, not your process. Start with one line on why you chose this task.
- **One script** (for example `scripts/studies/tbd_learned.py`) that regenerates every figure and
  number in the report from a clean checkout, including training. Fix your random seeds. Do not
  commit generated data or trained weights.
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

A fair comparison above all: the same false-track probability, held-out seeds, and a classical
baseline you did your best to make good. "The network did not help, and here is why" is a strong
answer if it is true. A careful answer to questions 1–4 beats a rushed answer to all five. Every
method and every clutter model you use has a cited source, and you can explain it.

### Optional, if you have time

Make the classical detector adapt instead, for example by estimating the clutter's distribution
from the data and using the matching likelihood ratio. How much of the learned detector's
advantage does that recover?
