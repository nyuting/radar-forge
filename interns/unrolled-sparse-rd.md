# Task 6: Can an unrolled solver beat a taper?

**Time:** about two to three working days. Stop when you reach it, and tell us what you would
have done next.

**AI tools:** allowed, and we expect you to use them. Tell us how you used them (see
[Deliverables](#deliverables)).

**Afterwards:** see [the follow-up call](README.md#the-follow-up-call).

Read [the intern README](README.md) and the [AI-for-radar background](ai-for-radar.md) first.
The background lists what the library does not have, and the rules every comparison must keep.

## Background

Scenario 003 forms its range-Doppler map with a plain two-dimensional FFT and no taper
(`form_range_doppler_map`). A finite FFT has a resolution limit of about one bin, and sidelobes
that can hide a weak target next to a strong one. The textbook remedy is a **taper**: a Taylor or
Chebyshev window lowers the sidelobes and widens the main lobe, trading resolution for dynamic
range. `radar_forge.core.windows.taper` provides both.

A different remedy treats the map as a **sparse** sum of a few point targets and solves for them,
for example with the iterative shrinkage-thresholding algorithm (ISTA) or ADMM. Sparse solvers
can separate targets closer than one bin when the SNR allows it. But they are slow, need many
iterations, and invent peaks that are not there when their threshold is wrong. **Deep unrolling**
turns a fixed number of solver iterations into network layers with learned weights (LISTA is the
best-known example). It promises the solver's resolution in a few layers.

This task asks whether that promise survives a fair comparison. "Better peak-to-sidelobe ratio"
is not a fair comparison on its own. A sparse solver can push sidelobes to zero by declaring
fewer peaks, and that also deletes real targets. Resolution and false peaks have to be measured
together, at the same false-peak rate.

Nothing sparse or learned is implemented in `radar-forge`. You will write it in your own
script. Leave `src/` alone.

## Setup

- Use this repository at the latest `main`, and record the commit hash you used in your report.
- Run `uv sync --extra dev --extra viz --extra ml`.
- Useful entry points:
  - `radar_forge.pipelines.scenarios`: `load_scenario` (for scenario 003's waveform in
    `scenario.bursts[0]`), `form_range_doppler_map`.
  - `radar_forge.core.propagation.line_of_sight_paths` takes arrays, so one call places two
    targets. `radar_forge.core.signal.fmcw_deramp_baseband` turns the paths into IQ, and
    `thermal_noise` gives noise-only data.
  - `radar_forge.core.dsp.range_doppler_map`: the same FFT chain, with `fast_time_window` and
    `slow_time_window` arguments for a taper.
  - `radar_forge.core.windows`: `taper`, `coherent_gain_linear`, `processing_loss_db`.
- **You may work along one axis.** Doppler, across the 256 chirps at one range bin, is a
  one-dimensional problem with the same structure as the full map, and is much cheaper to train
  on. If you do, say why one axis is enough for your conclusion, or what would change in two.

## The questions

1. **Predict first.** Work out the Rayleigh resolution of scenario 003's waveform along the axis
   you chose, in metres or metres per second and in bins. Then give two numbers, before you run
   anything:
   - the separation, in bins, below which a Taylor taper no longer separates two equal targets;
   - the per-cell SNR you expect a sparse solver to need to separate them at half a bin.

   Say what result would refute each.
2. **Baselines.** Build two classical detectors, and tune each one honestly:
   - the windowed FFT with peak picking, with rectangular and Taylor tapers at a few sidelobe
     levels;
   - ISTA (or FISTA, or ADMM) on the same signal model, run to convergence.

   Define "resolved": for example, two declared peaks, each within half a bin of a true target.
   Define a "false peak": a declared peak that is not within that tolerance of any target. State
   your definitions and keep them fixed.
3. **Unroll.** Build an unrolled solver with a small, fixed number of layers. Train it on
   synthetic one- and two-target signals with known truth. Say exactly what you train on: the
   separations, amplitude ratios, SNRs and on-grid or off-grid positions. Keep training,
   threshold-setting and test seeds separate.
4. **Compare fairly.** Set each method's detection threshold for the **same** false-peak
   probability, from held-out noise-only and one-target runs. Then plot the probability of
   resolving two targets against their separation, in bins, at a few SNRs, for all three methods.
   Report the peak-to-sidelobe ratio too, and explain why it alone would have told a misleading
   story.
5. **Shift and cost.** Test on at least one condition you did not train on. Examples:
   - targets off the bin grid, which the on-grid model the solver assumes does not describe;
   - a 20 dB amplitude ratio;
   - three targets;
   - a different SNR.

   How fast does the unrolled solver's advantage, if any, disappear? Report operations and memory
   per map against the FFT and against converged ISTA.

## Deliverables

- **A short report** (at most 4 pages plus figures; PDF or Markdown) answering the five questions.
  Lead with your findings, not your process. Start with one line on why you chose this task.
- **One script** (for example `scripts/studies/unrolled_sparse_rd.py`) that regenerates every
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

A fair comparison above all: one false-peak rate for every method, held-out seeds, and a taper
and a converged solver you did your best to make good. "The unrolled solver matched ISTA in
eight layers but never beat it" is a strong answer if it is true, and so is "a taper was enough
above half a bin". Every number should be reproducible, and every method should have a cited
source you can explain.

### Optional, if you have time

Compare against a subspace method, such as MUSIC with spatial smoothing over subarrays of
chirps, at the same false-peak rate. Where does it sit between the taper and the sparse
solvers, and at what cost?
