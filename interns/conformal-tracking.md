# Task 7: Does the tracker know its own uncertainty?

**Time:** about two working days. Stop when you reach it, and tell us what you would have done
next.

**AI tools:** allowed, and we expect you to use them. Tell us how you used them (see
[Deliverables](#deliverables)).

**Afterwards:** see [the follow-up call](README.md#the-follow-up-call).

Read [the intern README](README.md) and the [AI-for-radar background](ai-for-radar.md) first.
The background lists what the library does not have, and the rules every comparison must keep.

## Background

A Kalman filter reports a covariance with every estimate. The tracker relies on that
covariance being honest in two places:

- **The gate.** A measurement is accepted for a track when its normalised innovation squared
  (NIS) is below a chi-square quantile. If the filter is consistent, the NIS of the true
  measurement is chi-square distributed, so `gate_probability = 0.99` keeps it 99 % of the
  time. Read `ChiSquareGate` in `radar_forge.core.tracking.association`.
- **The reported uncertainty.** Anyone using a track's position reads its covariance as an
  error bar.

Both are promises that hold only if the motion and measurement models are right. Scenario 003's
target is not a constant-velocity target. It is built from noisy ADS-B fixes 2–8 s apart, and
its range rate jumps by several m/s between frames; read the comment above `sigma_accel_mps2`
in `scenarios/scenario_003_tracking.toml`. So the promise may already be broken, and the
question is by how much.

**Conformal prediction** offers a different promise. Calibrate a threshold on the empirical
distribution of a score, such as the NIS, from held-out data. Then a new score falls below it
with probability at least 1 − α, whatever the score's distribution. That guarantee is real, but
narrower than it is often sold:

- it is *marginal*: it holds on average over the calibration data, not for each track or each
  manoeuvre;
- it needs the scores to be *exchangeable*, so that the order of the data carries no information.

A track's innovations come in time order and are correlated. Online variants of conformal
prediction adjust the level as errors arrive, to keep long-run coverage without exchangeability.
Find them, and read what they actually guarantee.

Nothing conformal is implemented in `radar-forge`. You will write it in your own script. Leave
`src/` alone.

## Setup

- Use this repository at the latest `main`, and record the commit hash you used in your report.
- Run `uv sync --extra dev --extra viz`. Add `--extra ml` only for the optional learned model.
- `uv run python scripts/run_scenario.py scenarios/scenario_003_tracking.toml --out out/s1 --no-iq`
  runs the chain end to end. Read that script to see how the pieces fit together. Don't commit
  `out/`.
- Two tracker paths gate by chi-square:
  - `scenario_003_tracking.toml` runs the linear Kalman path;
  - the `scenario_003_ukf_*.toml` files run the UKF (`Tracker`).

  Pick one and say why.
- Useful entry points:
  - `radar_forge.pipelines.tracking`: `ScenarioTracker.from_scenario` and its `step`. Each frame
    gives a `FrameTracks`, and each `TrackRecord` in it carries the track's `state`,
    `covariance` and `nis`. `score_primary_track` reports the mean NIS per measurement
    dimension.
  - `radar_forge.pipelines.scenarios`: `load_scenario`, `iterate_frames`. Each frame carries the
    truth (`frame.range_m`, `frame.radial_velocity_mps`), so you can compute the normalised
    estimation error squared (NEES) as well as the NIS.
  - `radar_forge.core.tracking.estimation.innovation_stats`: NIS and log-likelihood of one
    innovation.
- **Data.** `data/flight_coordinates.csv` holds about 16,500 s of flight. Scenario 003 uses
  120 s of it, from `start_time_s = 663.0`. To get calibration and test windows, copy the TOML
  and change `start_time_s`. The `[trajectory] path` is resolved relative to the TOML's own
  folder, so keep the copy next to the original or make the path absolute. Decide how to split
  the flight, and justify the split.
- A gate is a threshold on the NIS. Work out how to apply a threshold of your own without
  changing `src/`: on recorded NIS values, or through the tracker's existing settings.

## The questions

1. **Predict first.** Before you run anything:
   - Over scenario 003's window, will the true measurement's NIS fall inside the 0.99 gate more
     or less often than 99 %? By roughly how much? Say why.
   - Will a split-conformal gate, calibrated on a different window of the same flight, keep its
     coverage on scenario 003's window?

   Give a number for each, and say what would refute it.
2. **Measure consistency.** Run the tracker over several windows and noise seeds. Report:
   - NIS and NEES against their chi-square bands;
   - the empirical coverage of the chi-square gate;
   - how often the gate rejects the true measurement, and what the tracker does next (a coast, a
     new track, a lost target).

   Is the filter optimistic or pessimistic, and is that the same over the whole window?
3. **Make it wrong on purpose.** Repeat with a motion model you know is wrong: a much smaller
   `sigma_accel_mps2`, or a stretch of the flight with turns. Show how the chi-square gate's
   coverage falls.
4. **Conformal gates.** Calibrate a split-conformal threshold on the NIS from separate windows
   and seeds, and test it. Then try one online conformal method that adapts its level as it
   goes. For each, against the chi-square gate, report:
   - coverage over the whole run, and in short sliding windows;
   - the gate volume, because a larger gate admits more false alarms;
   - the effect on track continuity.

   Where does exchangeability fail, and what does that cost you?
5. **Intervals on the state.** Do the same for the reported range: a conformal interval
   against the filter's ±kσ. Does conformal coverage on range errors tell you something the
   NEES did not?

## Deliverables

- **A short report** (at most 4 pages plus figures; PDF or Markdown) answering the five questions.
  Lead with your findings, not your process. Start with one line on why you chose this task.
- **One script** (for example `scripts/studies/conformal_tracking.py`) that regenerates every
  figure and number in the report from a clean checkout. Fix your random seeds. Do not commit
  generated data.
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

Statistics you can defend: the right chi-square bands, coverage with error bars, and a clear
account of what the conformal guarantee does and does not cover here. "Retuning
`sigma_accel_mps2` fixed the gate, and conformal added nothing" is a strong answer if it is
true. A careful answer to questions 1–4 beats a rushed answer to all five.

### Optional, if you have time

Learn the motion instead. Train a small network to predict the next range and range rate from
the last few track states (this needs `--extra ml`), and wrap its predictions in conformal
intervals. Is the learned model with conformal intervals better or worse calibrated than the
constant-velocity Kalman filter with its own covariance? Is it more accurate?
