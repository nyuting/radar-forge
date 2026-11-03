# Pipelines audit — open items

**Scope:** `src/radar_forge/pipelines/`, `src/radar_forge/visualisation/` and
`scripts/run_scenario.py`, checked against the specs by the [audit method](../conventions/audit.md).
**Last checked:** 2026-11-03, at `61e63a9`.

This file lists only what is still to do. When you fix an item, delete it in the same commit. The
full October 2026 report, with every verdict table and the findings that were fixed, is in git:

```sh
git show 7bfb654:docs/audits/pipelines-audit.md
```

## Summary

| # | Problem | Size |
| :--- | :--- | :--- |
| [P1](#p1-the-target-snr-is-quoted-three-different-ways) | The target SNR is quoted three different ways | small, docs |
| [P2](#p2-two-citations-use-the-old-s42-form) | Two citations use the old "S4.2" form | small, docs |
| [P3](#p3-sibling-functions-disagree-on-a-keyword-only-config) | Sibling functions disagree on a keyword-only `config` | small, API |
| [P4](#p4-scenarion_frames-does-not-say-it-rounds) | `Scenario.n_frames` does not say it rounds | small, docs |
| [P5](#p5-render_frames-dynamic_range_db-is-positional) | `render_frame`'s `dynamic_range_db` is positional | small, script |
| [P6](#p6-no-core-helper-for-the-pulsed-range-axis) | No `core/` helper for the pulsed range axis | medium, `core/dsp.py` |
| [P7](#p7-the-sigma_accel_mps2-default-has-no-measured-margin) | The `sigma_accel_mps2` default has no measured margin | waits on tracker-001 step 7 |

## P1. The target SNR is quoted three different ways

Three places give the scenario 003 target's post-integration SNR, and no two agree:

| Where | Says |
| :--- | :--- |
| `spec/scenario-003-tracking.md` §4 | 48–52 dB over the window, about 72 dB at closest approach |
| `frame_detections` docstring, `pipelines/tracking.py:470` | 45–69 dB |
| `scenarios/scenario_003_tracking.toml:67, :79` and `scenario_003_tracking_dual_prf.toml:80, :92` | 51–65 dB |

The spec is the current figure. It was re-measured after `core/signal.py` changed its noise per
sample to `N0·fs`. The other two are older.

**Fix:** make the docstring and both TOML comments match §4, or point them at §4 and drop the
number. The TOMLs' "CRLB is 0.14 m" was worked out from the old SNR, so check it too.

## P2. Two citations use the old "S4.2" form

`pipelines/scenarios.py:53` cites "S4.2" and `:853` cites "S4.3". Both are the right sections of
`spec/scenario-001-xband.md`, but everywhere else the form is "§4.2 of `spec/scenario-001-xband.md`".

**Fix:** rewrite both in that form.

## P3. Sibling functions disagree on a keyword-only `config`

`frame_detections(product, config=None, *, ...)` accepts `config` by position, while
`dual_prf_measurements(products, fold_spans_mps, *, config=None)` requires a keyword
(`pipelines/tracking.py:418`, `:799`).

**Fix:** make `config` keyword-only in `frame_detections`. This breaks the signature, so update the
call sites and tests in the same commit.

## P4. `Scenario.n_frames` does not say it rounds

`n_frames` is `round(duration_s * frame_rate_hz)`, so a 120.4 s window at 1 Hz gives 120 frames and
silently drops the last 0.4 s (`pipelines/scenarios.py:250`). That behaviour is right, because a
partial frame is not a frame. But the docstring says only "Number of frames the window produces".

**Fix:** one more sentence in the docstring. Do not add a raise, because a reader may want to round.

## P5. `render_frame`'s `dynamic_range_db` is positional

`render_frame` in `scripts/run_scenario.py:485` takes `dynamic_range_db` as its fifth positional
argument. A bare float in the middle of a call can be swapped silently.

**Fix:** put `*` before `dynamic_range_db`. There is one call site.

## P6. No `core/` helper for the pulsed range axis

The FMCW range axis comes from `core.dsp.range_bin_centers_m`. The pulsed one is worked out locally
as `n·c / 2f_s` (`range_axis_m` and `form_range_doppler_map`, `pipelines/scenarios.py:752`, `:861`).
The formula is right, and it holds for a bistatic burst too. But `range_bin_centers_m` maps a beat
frequency, not a delay, so `core/` has no helper for the pulsed case.

**Fix:** add `delay_bin_centers_m` to `core/dsp.py` and call it from both places. Then delete the
docstring sentence that says the seam is open.

## P7. The `sigma_accel_mps2` default has no measured margin

`TrackingConfig.sigma_accel_mps2` is 5.0, where scenario 003 §6.2 says 2.0. The margin that once
justified 5.0 no longer exists. Over the current window both values hold a confirmed track in
117 of 120 frames. Spec §14.6 and the `TrackingConfig` docstring both say so now, so nothing is
wrong. The setting is simply kept for want of a reason to move it.

**Waits on:** [tracker-001 §13](../../spec/tracker-001.md#13-migration-from-todays-code) step 7,
which retires `sigma_accel_mps2`. Delete this item then. A `slow` test pinning the 117/120 figure
was considered and rejected: it would pin one window's measurement, not a property of the tracker.

## Decided, don't reopen

These were looked at and left alone on purpose. Reopen one only if what it rests on changes.

- **`RangeDopplerProduct` keeps its name.** `RangeDopplerMap.rd_map` would be a tautology, and no
  better field name exists. `values` is vaguer, and `amplitude` is a bare physical name.
- **`burst` keeps meaning a configured `Radar`.** It is the `[[burst]]` table in every scenario
  TOML, so renaming it would change the file format.
- **The scopes return `Any`.** Naming `matplotlib.figure.Figure` needs the import the package
  defers. The docstrings give the real type.
- **`peak_range_velocity` returns a plain `(range_m, velocity_mps)` tuple.** It has one call site
  (`scripts/run_scenario.py:832`), the order is documented, and nothing has misused it.
- **The pipelines are not worth speeding up.** In October 2026 they took 2.6% of a scenario 003 run.
  Signal synthesis and CFAR in `core/` took most of the rest. Profile again before you make any
  speed change here.
