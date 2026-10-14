# Intern research tasks

Welcome. This folder holds research tasks for intern candidates and interns. Each one is a real,
open question about `radar-forge`, answered with the real library. They are not puzzles with a
known answer key you are meant to guess. Good answers become part of the project: the scripts,
settings and findings from a strong submission are what we would otherwise have to work out
ourselves.

## The tasks

| # | Task | The question | Time |
| :--- | :--- | :--- | :--- |
| 1 | [Where does the tracker break?](tracker-breakpoint.md) | As transmit power falls and the radar starts missing the target, how does detection degrade compared with theory, and when does tracking fall apart? | about 1 day |
| 2 | [How many dB does TBD buy?](tbd-dynamic-programming.md) | Build dynamic-programming track-before-detect. How far below the single-frame threshold can it go, and how close is that to the best possible? | about 2 days |
| 3 | [Particle-filter TBD](tbd-particle-filter.md) | Build a particle filter that decides target existence from unthresholded maps. How fast and how reliably does it declare a dim target? | about 2 days |
| 4 | [Where can learning beat the textbook?](tbd-learned.md) | Put a learned component into a TBD detector. Where, if anywhere, does it beat the classical one at the same false-track rate? | about 2–3 days |
| 5 | [Do world models help a radar?](world-models.md) | The tracker is already a hand-written world model. Pick one direction (motion, map prediction, clutter or planning) and find out where a learned one beats it. | about 3 days |
| 6 | [Can an unrolled solver beat a taper?](unrolled-sparse-rd.md) | Does a learned, unrolled sparse solver separate two close targets better than a Taylor taper, at the same false-peak rate? | about 2–3 days |
| 7 | [Does the tracker know its own uncertainty?](conformal-tracking.md) | Is the tracker's covariance honest, and when it is not, can a conformal gate or interval restore the coverage it promises? | about 2 days |
| 8 | [Thresholds from the tail](evt-thresholds.md) | How few samples does an extreme-value fit need to set a reliable threshold at Pfa = 1e-6? | about 2 days |
| 9 | [Learned data association](learned-association.md) | With crossing targets and false alarms, does a small graph network beat global nearest neighbour at the same false-track rate? | about 3 days |

Tasks 2–4 are about **track-before-detect** (TBD), which `radar-forge` does not implement yet.
They share [one background page](track-before-detect.md); read it before you choose.

Tasks 6–9 are about **AI for radar**: a learned or statistical block against the classical one
it would replace. They share [one background page](ai-for-radar.md). It also lists further
directions you may propose instead, if you agree one with us first.

Tasks 4, 5, 6 and 9 need PyTorch, and so do the optional parts of 7 and 8. For those, run
`uv sync --extra dev --extra viz --extra ml` instead of the setup command below.

**Choose the task you find most technically interesting.** We learn more from you chasing a
question you care about than from you finishing one you don't. Tell us briefly why you chose it.
If we have assigned you a specific task, do that one.

## How to work

- **Get the code.** Fork this repository, or clone it and work on your own branch. Start from the
  latest `main` and write down the commit hash you started from; your numbers depend on it.
- **Make your work visible to us.** Push your branch to GitHub and make sure we can see it: a
  public fork, or a private fork or repository with us added as collaborators. Check the link
  works from a logged-out browser, or ask us to confirm we can open it, *before* you submit. If we
  can't open your branch, we can't grade it.
- **Set up.** Install [uv](https://docs.astral.sh/uv/), then run `uv sync --extra dev --extra viz`
  from the repository root. Run everything through `uv run …`. Never use `pip install`.
- **Read before you write.** [`CLAUDE.md`](../CLAUDE.md) is a one-page summary of the project's
  rules:
  - SI units in every name (`transmit_power_w`, not `power`);
  - physical constants from `radar_forge.core.constants`, never typed in;
  - NumPy-style docstrings.

  [`CONTRIBUTING.md`](../CONTRIBUTING.md) says the same at more length. Follow these rules in
  your own scripts too.
- **Leave `src/` alone.** The tasks are designed so that you never need to change the library. If
  you think you do, explain why in your report instead. That explanation is a finding.
- **Use AI tools.** You may use any AI assistant or coding agent, and we expect you to. Each task
  asks for a short AI-use note saying what the tool did, what you checked yourself and how, and
  where the tool was wrong. We grade your judgment, not your typing.

## Bugs and suggestions

Every submission ends with a **Bugs and suggestions** section. List anything in the code or
documentation that looks wrong, confusing, slow or improvable, even if it has nothing to do with
your task. For each item give:

- **where**: `path/to/file.py:line`, or a document and section;
- **what** you saw;
- **why** you think it is a problem, and what you would do instead.

A real bug report with a reproducing snippet is one of the most valuable things you can send us.
"I found nothing" is an acceptable answer only if you genuinely looked.

## Submitting

Send us:

1. your report;
2. a link to your branch or fork, visible to us (see [How to work](#how-to-work));
3. your AI-use note.

The task file says what goes in each. Do not commit generated data such as `.npy`, `.h5` or large
CSVs. Commit the script that makes it.

## The follow-up call

After you submit, we normally hold a **30-minute call** where you walk us through your results and
we ask questions about them. Be ready to rerun your script and explain any number in your report.

If you received a task **after** you have already had your interview call with us, there is no
second call by default. Either side can still ask for one: you can request a call if you want to
talk your work through, and we may request one if we have questions.
