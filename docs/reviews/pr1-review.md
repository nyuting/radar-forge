Hi Michael, thanks for this, and welcome! It's a big piece of work, and the filter maths is careful. This is also Yuting's first review on this repo. Yuting drafted it with Claude, so each point explains *why* as well as what. 

**Verdict: Request changes.** Two things block merging: the licence question (#1), and the lint and format exemptions in `pyproject.toml` (#2). Everything else is a should-fix, a point for discussion or a nit.

**How to use this review.** The table below says *what* and links to a short section on *why*. The [re-review checklist](#re-review-checklist) at the end says *in what order* and *how each item will be checked*. When you split the PR (#4), copy each PR's part of the checklist into that PR's description, and the "Every PR" part into all of them. Tick it as you go. Yuting will re-review against the same list, so each PR can close its items in one pass.

## Summary

Labels: **blocking** = must be settled before merge · **should-fix** = fix in the split PR named · **discuss** = let's talk · **nit** = small.

| # | Ask | Label | PR |
| :-- | :-- | :-- | :-- |
| | **Blocking** | | |
| 1 | [Say where the tracker came from and whether it can be MIT](#1-licence-and-provenance) | blocking | before any |
| 2 | [Remove the lint and format exemptions from `pyproject.toml`; fix the comments instead](#2-lint-and-format-exemptions) | blocking | every PR |
| | **The shape of the change** | | |
| 3 | [One tracker: move into `core/tracking/`, not a top-level `tracking/` (or argue against D2 in a spec PR)](#3-one-tracker-in-coretracking) | should-fix | A |
| 4 | [Split into four PRs](#4-split-the-pr) | should-fix | — |
| 5 | [Apply the renames](#5-renames) | should-fix | A, C, D |
| 6 | [Reuse `core`, remove duplicates, defer unused code](#6-reuse-remove-defer) | should-fix | A, C, D |
| | **PR A: tracker behaviour and logic** | | |
| 7 | [Stop tentative tracks stealing detections](#7-tentative-tracks-compete-with-confirmed-ones) | should-fix | A |
| 8 | [Raise the default process noise and say why](#8-process-noise-too-low) | should-fix | A/C |
| 9 | [Document NIS vs negative log-likelihood](#9-nis-vs-negative-log-likelihood) | discuss | A |
| 10 | [Logic: delete dead tentative tracks, Joseph-form update, and smaller fixes](#10-logic-and-implementation) | should-fix | A |
| 11 | [IMM: make CV + CT buildable and test the mixing](#11-imm) | should-fix | when IMM lands |
| 12 | [Vectorise the sigma-point path](#12-vectorise-the-sigma-point-path) | should-fix | A |
| | **PR B: readability** | | |
| 13 | [Remove the `####` banners and Inputs/Outputs blocks, and fold them into docstrings (includes `tracking_config.py` L324+)](#13-comments-that-repeat-the-docstring) | should-fix | A, C |
| 14 | [One specific reference per module or class](#14-references) | should-fix | A |
| 15 | [A "why" comment on each Python loop](#15-loops-need-a-why) | should-fix | A |
| 16 | [Reading the tracker: fewer hops, docstrings that match the code, standard names](#16-reading-the-tracker) | should-fix | A |
| 17 | [Reading view: guard-only values left behind](#17-the-reading-view) | should-fix | A, C |
| | **PR AB: tests** | | |
| 18 | [Tests: give the new tracker main's tests (consistency, clutter, P_d < 1, M-of-N, covariance health, Jacobian)](#18-tests-what-a-tracker-needs) | should-fix | A, C; deferred code when it lands |
| 19 | [Remove tests that lock in the fork or can't fail; merge duplicates](#19-tests-that-cost-more-than-they-catch) | should-fix | A, C, D |
| | **PR C: the pipeline** | | |
| 20 | [Write the files data-001 specifies, with writers that only write](#20-one-file-format) | should-fix | C, D |
| 21 | [`ScenarioTracker` and the pipeline docs: match main's conventions, fix docs that contradict the code](#21-scenariotracker-and-the-pipeline-docs) | should-fix | C |
| 22 | [Pipeline size: about 970 lines that redo `pipelines/tracking.py`; move each piece to its home](#22-pipeline-size-and-placement) | should-fix | C |
| | **PR D: runner and docs** | | |
| 23 | [Docs for the next reader: specs, notes file, "Duke"](#23-docs-for-the-next-reader) | should-fix | D |
| | **Anywhere** | | |
| 24 | [Nits](#24-nits) | nit | any |

## 1. Licence and provenance

`spec/tracker-001-provenance.md` says the source, `unified-extensible-tracker`, was "supplied by the user", and that it had no Git metadata or licence file. This repo is MIT, and `spec/structure.md` B.2 rule 4 and D4 treat "no licence" as "we can't copy it; rewrite it from the published equations". **Did you write the original?** If so, please say so plainly in the PR and in the provenance doc. If someone else wrote it, we need their permission first. This comes first because merged code is hard to un-publish. It also decides #2: if the annotations are yours, there's no "preserved author's text" to protect.

## 2. Lint and format exemptions

This is the second blocking item. The PR adds three things to `pyproject.toml`:

- **Per-file ignores for all fourteen `tracking/` modules.** They cover E501 (line length), W291/W293 (trailing whitespace), W292 (no newline at end of file), D200, D202, D204, D205, D210 and D400 (docstring layout), and RUF002 (an ambiguous Unicode character in a docstring, in `tracks.py`).
- **A `[tool.ruff.format] exclude`** that lists the same fourteen files, so `ruff format` never touches them.
- **`force-exclude = true`.** Ruff normally checks any file you name explicitly, even an excluded one. The pre-commit hook names each staged file (`.githooks/pre-commit:20`), so without this setting it would still catch unformatted code. With it, the hook's format check skips these fourteen files as well, and nothing local catches format drift.

**Main has none of this for library code.** Its only `per-file-ignores` are `tests/**` and `scripts/**`: whole directories, each with a one-line reason. There is no `[tool.ruff.format]` table and no `force-exclude`, and `src/` has no `# noqa` comments. The repo has one way to exempt something, and it covers one line and gives a reason: `# broadcast-exempt: <reason>` (CLAUDE.md). Nothing exempts a whole file.

**Why it blocks.** Every exempted rule is about layout, not meaning. Claude ran ruff on `tracking/` with main's `pyproject.toml`. It reports 279 errors: 123 lines too long, 112 lines of trailing whitespace or whitespace-only blank lines, 37 docstring-layout errors, 6 missing final newlines, and one ambiguous Unicode character. All fourteen files would be reformatted. Ruff can fix 131 of these by itself (`ruff check --fix`); 76 of the 123 long lines are comments, which folding the banners into docstrings (#13) removes. The other 47 need wrapping by hand. The stated reason, "preserve the imported author's annotations verbatim", depends on #1. If you wrote the original, there's nothing to preserve verbatim. If someone else did, we can't copy it. Either way, the exemption has no job left to do. And once it merges, every future edit to those files skips the gate that everyone else passes. Exemptions like these are almost never removed later.

**The fix is to do the comments properly, not to exempt them.** Keep what your annotations *say*: #13 shows how to fold each banner into its NumPy docstring. Then delete all three additions and let the tools run:

```sh
uv run ruff format src
uv run ruff check src
```

Please don't swap the file-wide ignores for `# noqa` comments. If a line really can't be wrapped, say why in a comment and we'll talk about it.

## 3. One tracker, in `core/tracking/`

`spec/structure.md` D2 plans for `core/tracking.py` to grow *into* `core/tracking/`, re-exported "so the public import path never changes". The PR instead adds a second, parallel `radar_forge.tracking` package. It also adds a second CFAR, a second pipeline module and a second runner script. The paragraph added to `structure.md` contradicts D2 and leaves D2's text unchanged, so the spec now says both things at once.

Please put the package at **`src/radar_forge/core/tracking/`** instead:

- **Imports keep working.** Turning `core/tracking.py` into `core/tracking/__init__.py` keeps `from radar_forge.core.tracking import KalmanState, TrackManager, …` working unchanged. It has to be a *move*: if `tracking.py` and `tracking/` both exist, Python imports the package and silently ignores the file.
- **One tracker, not two.** In one package there's room for only one `Track` and one track manager, so the two have to merge rather than drift apart. Only numpy and scipy are needed, so B.2 rule 1 ("core is dependency-light") still holds.
- **Students know where to look.** `core/` already means "the algorithms".

Suggested layout: `core/tracking/{kalman, motion, measurement_models, coordinates, ukf, association, initiation, lifecycle, tracker, _validation}.py`, with `kalman.py` holding today's `core/tracking.py` unchanged. If you think D2 is wrong, that's fine: make the case in a small PR against the spec, so the decision is explicit.

## 4. Split the PR

7.6k lines in one commit is more than anyone can review carefully. Suggested order, each building on the last:

| PR | Contents |
| :-- | :-- |
| **A** | Move `core/tracking.py` into a `core/tracking/` package (no behaviour change). Then add the tracking core and its unit tests, but only the pieces a scenario uses today (#6). |
| **B** | 2-D CFAR and `wrap_axes` in `core/detection.py` (spec 003 §13.2, §13.5). **Yuting writes this one**, so you don't need to. |
| **C** | Pipeline, config and output, merged into `pipelines/tracking.py` and built on B. |
| **D** | Runner (extend `scripts/run_scenario.py`), plotting, README, spec. |

A doesn't touch detection, so B doesn't block you. Yuting is happy to pair on the split.

## 5. Renames

The rule of thumb:

- **Algorithm names:** keep the term the textbooks and Stone Soup/FilterPy use, so a student can look it up.
- **Names around the edges** (modules, fields, configs, file formats): follow this repo's conventions. `docs/conventions/style.md` §2–3 asks for units in names and `n_` for counts. Main's code adds two habits: the family word first (`velocity_unfolded_mps`, `cfar_threshold_w`) and `sigma_` for standard deviations. And no name is reused for a different thing.

**Rename:**

| Now | Proposed | Why |
| :-- | :-- | :-- |
| `radar_forge/tracking/` | `radar_forge/core/tracking/` | D2 (#3) |
| `pipelines/general_tracking.py`, `tracking_config.py`, `tracking_output.py` | merge into `pipelines/tracking.py` | one of each (see below) |
| `scripts/run_general_tracking.py` | extend `scripts/run_scenario.py` | ~280 of its 408 lines are copied from it |
| `GeneralScenarioTracker`, `GeneralTrackingConfig` (aliases) | one `ScenarioTracker`, one `TrackingConfig` | clash |
| `leg`, `legs`, `leg_index`, `leg_metadata()`, `tracking_legs()`, `leg{i}`, `rd_leg{i}` | `burst`, `bursts`, `burst_index`, `burst_metadata()`, … | main's name: `burst_metadata()`, and `bursts` in `metadata.json` (data-001 §6.1, §6.7). `core/radar.py:40–42`: "a burst is a waveform configuration", and "leg" is kept out of it |
| `chirp_time_s`, `n_chirps` (in `metadata.json`) | `chirp_duration_s`, `n_pulses` | main's names; style §3.1 forbids `n_chirps` |
| `TrackerEngine` (`engine.py`) | `Tracker` (`tracker.py`) | clarity: "Engine" adds nothing |
| `tracks.Track`, `management.TrackManager`, `TrackStatus` | merge with `core.tracking.Track` / `TrackManager` / its status `Literal` | clash |
| `management.py`, `LifecyclePolicy` | `lifecycle.py` | clarity |
| `spaces.py`, `StateSpace`, `MeasurementSpace = StateSpace` | `coordinates.py`, `StateLayout`; drop the alias | clarity: "state space" means (A, B, C, D) to engineers |
| `x_m`, `y_m`, `z_m`, `xdot_mps`, … | `east_m`, `north_m`, `up_m`, `east_rate_mps`, … (or say x = east in the docstring) | convention: `core/geodesy.py` says east/north/up |
| `Coordinate.period` | `period` with its unit documented, or split into `period_m`/`period_rad` | convention: units |
| `normalize` | `normalise` | convention: the repo spells it the British way (`normalised_`) |
| `measurements.py` | `measurement_models.py` | clarity (Stone Soup's term) |
| `MonostaticRadar`, `BistaticRangeDoppler` | `MonostaticRangeBearingModel`, `BistaticRangeDopplerModel` | clash in spirit with `core.radar.Radar`/`BistaticRadar` |
| `sensors.py`, `Sensor` (`id`, `measurement_model_ids`) | `SensorRoute` (`route_id`) inside `measurement_models.py` | clarity: it only routes, and `id` shadows a builtin |
| `SensorPose` (own `position_m`, `origin_lla_deg_m`) | build it from `core.radar.Radar`/`BistaticRadar` | one place says where the radar is |
| `RadarDetection` | build on `core.detection.Detection` | clash: this would be a third detection type |
| `.radial_velocity_mps` | `.velocity_folded_mps` | convention: says it is folded |
| `.unfolded_velocity_mps`, `.unfolding_residual_mps` | `.velocity_unfolded_mps`, `.velocity_unfolding_residual_mps` | convention: family word first |
| `.range_bin`, `.doppler_bin` | `.range_index`, `.velocity_index` | main's names |
| `.noise_power_linear` (also `CfarResult`) | `.noise_power_w` | convention: it is a power. In files it goes per burst in `metadata.json`, not in `detections.csv` (#20) |
| `.status: str` | `status: Literal["accepted", "missing_pair", "ambiguous_pair", "unresolved_velocity"]` | typing |
| `TrackingFrame` | merge with `pipelines.tracking.FrameTracks` | clash: a third name for one idea |
| `detect_product` | the existing `frame_detections` | duplicate (#6) |
| `confirmation_hits`, `confirmation_window`, `deletion_misses`, `history_size`, `hit_count`, `miss_count`, `age` | `n_confirm_hits`, `n_confirm_frames`, `n_delete_misses`, `n_history`, `n_hits`, `n_misses`, … | convention: `n_` for counts, as on main |
| `initial_velocity_std_mps`, `range_std_floor_m` | `sigma_velocity_mps`, `sigma_range_floor_m` | convention: main writes `sigma_` |
| `noise_density` (in `build_enu_tracker`) | `acceleration_noise_density_m2ps3` (CV) / `jerk_noise_density_m2ps5` (CA) | convention: the unit differs between CV and CA |
| `build_enu_tracker` | move into `scripts/run_tracking_example.py`, or `build_tracker_enu` if it stays public | clarity: "ENU" names the least important choice |
| `build_tracker(motion, …)` | keep. The script's `build_tracker(scenario)` becomes `ScenarioTracker.from_scenario()` | clash with `scripts/run_scenario.py:310` |
| "radial velocity" vs "range rate" | pick one for the closing-positive quantity (main's track output says `range_rate_mps`) | consistency |
| `_numerics.py`: `vector`, `covariance`, `cholesky`, `timestamp`, `FloatArray` | `_validation.py`: `as_vector`, `as_covariance`, `cholesky_factor`; inline `timestamp`; use `NDArray[np.float64]` | clarity: verb-first, and CLAUDE.md's array type |
| `core/detection_2d.py`, `cfar_2d`, its `DetectionConfig` | folded into `core/detection.py` (PR B) | clash with `pipelines.tracking.DetectionConfig` |
| `teaching/scopes/track_history.py`, `render_track_history` | extend the existing `teaching/scopes/track_plot.py` (`render_range_time_history`) | duplicate |

**Keep. These are the standard terms, and the PR uses them well:** `UKF`, `IMM`, `ChiSquareGate`, `NearestNeighbour`, `GlobalNearestNeighbour`, `Associator`, `MotionModel`, `MeasurementModel`, `CartesianMotion`, `CoordinatedTurn`, `TrackInitiator`, `StateEstimate`, `TrackSnapshot`, `Measurement` (z, R, time, sensor), `nis`, and `alpha`/`beta`/`kappa` (standard sigma-point symbols; say so in the docstring).

**Why the clashes matter.** `pipelines/__init__.py` in this PR already has to import `ScenarioTracker as GeneralScenarioTracker` because the plain name is taken. "General" describes how the code relates to a sibling, not what it does, and it stops meaning anything once the sibling is gone, like `new_` or `v2`. The same applies to prose. The README's `## General tracking` heading, its "general runner" and `docs/README.md`'s "general tracker" all mean "not the other tracker". Main never uses "general" as a name: the word appears only in ordinary sentences like "in general". The convention gives the plain name to the general thing and a qualifier to each specialisation (`numpy.fft.fft` / `rfft`).

**Where `RadarDetection` belongs.** `pipelines/` is the scenario-facing glue: it loads the TOML scenario, forms the RD map, and turns it into detections in SI units that go to `core.tracking`. That's why `pipelines/tracking.py` holds `Measurement`, `DetectionConfig`, `TrackingConfig` and `ScenarioTracker`. So `RadarDetection` is in the right *layer*. It just shouldn't live in a new module, or be a new type next to `pipelines.tracking.Measurement` and `core.detection.Detection`.

**Scope names.** `render_track_history` follows main's `render_<what it draws>` pattern, which is good. Four things differ from main's scopes:

- **Module and function share a stem.** Main names the module after the scope and the function after what it draws: `rd_map.render_range_doppler`, `track_plot.render_range_time_history`. `track_plot.py:10–16` explains why it plots range against time and says "the function is named for what it draws". `track_history.render_track_history` breaks that pattern, and it sits one word from `render_range_time_history`, which draws the same range-against-time plot.
- **It reads a file.** It takes a CSV `Path` and parses it (`track_history.py:42–49`), so the plot breaks whenever the file's columns change (#20). Main's scopes take arrays, so a notebook can plot a tracker's output without writing a file first.
- **It re-implements `require_pyplot`** (`:36–41`), with a different hint text from `teaching.plotting`.
- **It plots the first track in the file**, confirmed or not. Main's runner plots the longest-lived confirmed track (`scripts/run_scenario.py:418–435`).

So: add the velocity panel and the modulo-range label to `render_range_time_history`, as keyword arguments that take arrays, and drop the new module.

## 6. Reuse, remove, defer

About a third of the PR's lines either repeat something `main` already has or have no caller outside the tests. Removing them makes each split PR small enough to review.

**Use what already exists:**

| PR code | Use instead |
| :-- | :-- |
| `MonostaticRadar.predict`, `derived.derive_kinematics` (range/az/el by hand, two copies) | `core.geodesy.enu_to_range_azimuth_elevation` (`core/geodesy.py:279`) |
| Closing-positive radial velocity | the formula already in `core/tracking.py` (`_enu_model` at :495, the formula at ~:520) |
| `BistaticRangeDoppler` path ranges and rate | `core.radar.BistaticRadar.target_ranges_m`, `core.signal.bistatic_doppler_hz` |
| `ChiSquareGate` threshold, GNN assignment | `core.tracking.gate_threshold`, `associate_gnn` (keep the class wrappers if the protocol needs them, but call these) |
| `cfar_2d` alpha, peak picking, per-cell noise, mask | `cfar_threshold_factor`, `cluster_detections`, `cfar_noise_estimate_w`, `cfar_valid_mask` (`core/detection.py`; PR B adds the 2-D window) |
| `detect_product`, `_pair` | `pipelines.tracking.frame_detections`, `dual_prf_measurements`: add your `status`/`pair_id` to them. data-001 §6.5 doesn't have these columns yet; Yuting is adding them |
| `track_history.py` pyplot import guard and range–time plot | `teaching.plotting.require_pyplot`, `track_plot.render_range_time_history` |
| `run_general_tracking.py`: `git_commit`, `clear_previous_frames`, `assemble_movie`, `leg_metadata`, `parse_args` | the same functions in `scripts/run_scenario.py` |

**Duplicates inside the PR:**
- There are three range/azimuth/elevation computations: `derived.py`, `MonostaticRadar.predict`, and core geodesy.
- The range-wrap logic is in both `tracking_output.py` (~L199) and `spaces.py` (~L201).
- The lifecycle fields are declared in both `tracking_config.py` and `management.py`.
- `test_port_integration.py` (two sensors, crossing tracks) largely repeats `test_engine.py`.
- `test_sensors.py` tests a compatibility re-export in `measurements.py` that can itself go.

**Defer to the PR whose scenario needs it.** This means keeping the code on a branch, not deleting it. Each piece has no caller outside tests today:

| Code | ≈ lines | Lands with |
| :-- | --: | :-- |
| `imm.py` (`IMM`) | 435 | the first manoeuvring scenario (and #11) |
| `CoordinatedTurn` | 140 | same |
| `RangeBearingInitiator`, `RoutedInitiator`, `NoInitiation` (the last two have no tests) | 180 | the first angle-measuring sensor |
| `MonostaticRadar`, `CompositeMeasurementModel` | 160 | the first nonlinear radar measurement |
| `derived.py` | 170 | drop it: geodesy covers it |
| `ModelRegistry`, `MeasurementSpace` alias, `snapshot_record` | 80 | when something calls them |

`spec/structure.md` A.13a is the reason: "radar-forge needs one tracker an intern can read end to end". Stone Soup's decomposition is the right one, and you've got it. The ask is to bring each piece in with a scenario and a test that show it earns its place.

**Is `_numerics.py` needed?** Yes, a shared validation helper is right: style guide §5 expects one once shape errors recur, and `core/` has nothing like it. But not all of it, and only about 43 of its 185 lines are code. Keep the rest as `core/tracking/_validation.py` (#5 names). Claude counted the callers at bedfde0:

| Name | Callers | Verdict |
| :-- | :-- | :-- |
| `FloatArray` | 9 `tracking/` modules, plus `pipelines/general_tracking.py:28` | Drop it. It is a second spelling of `NDArray[np.float64]`, which CLAUDE.md and main use |
| `vector` | 17 call sites in 6 modules | Keep, as `as_vector`. Four calls throw the result away and use it only as a check (`motion.py:293`, `:378`, `:522`, `ukf.py:256`), and it also runs on every sigma point (`ukf.py:205`). Check once, at the boundary |
| `covariance` | 4 | Keep, as `as_covariance`. It runs `eigvalsh` every time a `StateEstimate` is built, so every predict and update pays for an eigendecomposition |
| `cholesky` | 2 (`estimation.py`, `ukf.py`) | Keep, as `cholesky_factor`. Its tolerance is scaled by the largest diagonal element, while `covariance` scales by the largest absolute element. Pick one, and name the `1e-10` in a constant with a comment |
| `timestamp` | 5 | Inline it. It overlaps `motion._elapsed` (`motion.py:84–87`), and most calls re-check a time that `StateEstimate` or `Measurement` has already checked |

Two further points:
- Code outside the package shouldn't import a `_private` module. `pipelines/general_tracking.py` imports `FloatArray` from it; write `NDArray[np.float64]` instead.
- A module with a leading underscore shouldn't declare an `__all__` of public-looking names.

## 7. Tentative tracks compete with confirmed ones

The default demo fragments tracks, and you documented that honestly. Claude traced it. In S1 at t = 15 s the detector reports one target four times: the same ~74 m range bin (bin 240, 17,987.5 m), at four Doppler bins. S1 measures range only, so the tracker sees four identical ranges. Track 1 takes one detection. The other three start tentative tracks sitting exactly on the bin centre, which then match the next detections better than track 1 does. Track 1 coasts and is deleted at t = 20 s. Tentative and confirmed tracks share one assignment, and both get `deletion_misses = 5`. Standard remedies:
- **Two-stage association:** confirmed tracks first, then tentative tracks on the leftover detections. Stone Soup's `MultiMeasurementInitiator` gets this by holding tentative tracks in their own tracker.
- **No births inside a confirmed track's gate.**
- **Early deletion:** delete a tentative track as soon as M-of-N becomes unreachable.

(The four-detections-per-target issue is on the detection side; Yuting will look at it in PR B.)

## 8. Process noise too low

In S3 the true radial velocity goes from −39 to −17 m/s between t = 9 and 14 s, roughly 4 m/s². With `acceleration_noise_density_m2ps3 = 1.0`, track 1's velocity stays at −32.8 m/s. The correct detections then fall outside the gate. A common starting point is to choose q so that √(q·T) is about the largest expected acceleration. That gives ~16 m²/s³ at T = 1 s. Please derive the default from the scenario's target dynamics and put the reasoning in the docstring.

**What changes the numbers** (primary-track confirmed frames out of 120 · range RMSE):

| Run | S1 | S2 | S3 |
| :--- | ---: | ---: | ---: |
| Defaults (matches your table) | 18 · 32.8 m | 10 · 11.4 m | 14 · 12.5 m |
| Confirmed tracks pick first | 45 · 30.7 m | 10 · 11.4 m | 14 · 12.5 m |
| … and no births inside a confirmed gate | 63 · 31.5 m | 10 · 11.4 m | 14 · 12.5 m |
| `acceleration_noise_density_m2ps3 = 16` only | 18 · 29.8 m | 13 · 12.8 m | 94 · 10.8 m |
| All three together | **118** · 13.6 m | 13 · 12.8 m | 94 · 10.8 m |
| `cost = "negative_log_likelihood"` only | 21 · 42.3 m | 10 · 11.4 m | 14 · 12.5 m |

So single-measurement initiation (your TODO near `engine.py` L245) isn't the main problem, and nor is the cost function. S2 is still open. Its folded-range ghosts look like a third cause, and it's worth understanding before more defaults change.

## 9. NIS vs negative log-likelihood

NIS (d²/S) makes a very uncertain track look like a good match. Take a confirmed track 3 m from a detection and a one-scan-old tentative track 12 m away. NIS gives the detection to the tentative track; d² + ln|S| (Blackman's generalised distance) gives it to the confirmed one. On S1–S3 the switch alone changed little (table above), so this isn't what's hurting today. It deserves a sentence in the `cost` docstring, because students will reach for NIS.

## 10. Logic and implementation

The filter maths checks out (What's good). The logic around it has one real bug and a few fragile spots.

**A tentative track that can never be confirmed is never deleted.** `TrackManager.expire` (`management.py:244–257`) deletes a track only after `deletion_misses` misses in a row or `max_coast_time_s` without an update. Nothing deletes a tentative track that can no longer reach M hits in its window. Claude ran it at bedfde0: one target, detected on every third scan (hit, miss, miss, …), with the default 3-of-5. After 60 scans and 20 hits, the track is still `tentative`, with `recent_hits = [False, False, True, False, False]` and `miss_count = 2`. It never confirms and is never deleted. Real clutter does the same thing: a false alarm that keeps getting refreshed stays in the assignment and competes for detections, which makes #7 worse. The PR's M-of-N window slides (`recent_hits`), so a track can stay tentative forever. Main counts M hits over a track's *first* N scans, and deletes it as soon as M becomes unreachable (`core/tracking.py:1321–1333`), and `test_a_tentative_track_dies_as_soon_as_m_is_unreachable` tests it (#18). Please carry both over.

The script Claude ran, as a starting point for the test:

```python
engine = build_enu_tracker({"x": "CV"}, origin_lla_deg_m=(36.00250, -78.94100, 60.0))
sensor = engine.sensors["sensor"]
first_tentative: dict[int, int] = {}
longest = 0
for t in range(60):
    if t % 3 == 0:
        z = np.array([1000.0 + 20.0 * t])
        snapshots = engine.process(sensor.batch(t, [("measurement", z, np.array([[0.1]]))]))
    else:
        snapshots = engine.process(MeasurementBatch(t, "sensor"))
    for s in snapshots:
        if s.status == TrackStatus.TENTATIVE:
            first_tentative.setdefault(s.track_id, t)
            longest = max(longest, t - first_tentative[s.track_id] + 1)
# 3-of-5, counted over a track's first five scans as main does: a track that
# hasn't confirmed by then is deleted. Fails today: track 1 is tentative for
# all 60 scans.
assert longest <= 5
```

**No Joseph-form update.** `UKF.update` computes `P − K S Kᵀ` and then symmetrises it (`ukf.py:335–341`). That is exact in exact arithmetic, but roundoff can push P slightly indefinite. `StateEstimate` then raises "semidefinite" and aborts the scan. Main uses the Joseph form (`core/tracking.py:829–831`), which stays positive semidefinite by construction. With `kappa = 0` the centre weight is already negative (#24), so this is the filter that needs it most. The UKF has no measurement Jacobian, so the Joseph form doesn't carry over directly. Either derive the UKF equivalent, or say in the docstring why the simple form is safe here, and add main's 200-step covariance-health test (#18) to back it up.

**Smaller things.** Each is a line or two:

- **`IMM.state` re-mixes on every access.** It returns `mixture(...)` (`imm.py:187`), and the engine reads `.state` several times per track per scan (`engine.py:179`, `:212`, `:226`). Compute it once per predict or update. This is part of #12.
- **Exact `==` on float times.** `engine.py:123` and `:127` compare timestamps with `!=`. That works while every time comes from the same float, but not once times are computed two ways (`k * dt` against a running sum). Compare with a tolerance, or say in the docstring that times must be bit-identical.
- **Validation that can be skipped.** `engine.sensors` is a public dict that is checked only in `__init__` (`engine.py:76`). The PR's own test adds a sensor afterwards (`test_engine.py:49–50`). Make it read-only, or add an `add_sensor` method that validates.
- **Strings where the type could say it.** `cost: str` (`engine.py:71`) is checked at run time (:80–81); `Literal["nis", "negative_log_likelihood"]` lets mypy check it instead. `CartesianMotion`'s `"CV"`/`"CA"` is the same.
- **A positional `True`.** `_Geometry(state_space, transmitter, fixed_coordinates, True)` (`measurements.py:414–415`) gives the reader no hint what `True` means. Make the argument keyword-only.

**Overall.** The equations are right, and the tests of the equations are good. The engineering around them is heavier than the job needs: more files, registries and validation passes than one radar and one scenario call for. That, not the maths, is what makes the tracker hard to read and slow. The lifecycle has one real bug. None of it needs a redesign: #6 removes most of the weight, and this item and #7 fix the lifecycle.

## 11. IMM

This applies when IMM lands (#6). `IMM` needs every mode to share one `StateSpace`, but 2-D CV is `(x, ẋ, y, ẏ)` and `CoordinatedTurn` adds a turn rate. So the textbook CV + CT pair raises `"IMM modes require a common StateSpace"`. The only IMM test uses two identical CV modes and calls `predict_to(0)`, which returns early, so the mixing code never runs. Please add:
- a CV mode in the 5-state CT space (turn rate held at 0);
- a test where the target starts turning and the CT mode's probability rises;
- a limitation note that the transition matrix is per step, not per second. That's FilterPy's choice too, but with two asynchronous sensors it doubles the switching rate. The principled fix is Π(Δt) = expm(ΛΔt).

## 12. Vectorise the sigma-point path

`UKF._sigma_points`, `predict_to` and `_innovation` handle sigma points one at a time in list comprehensions. `CartesianMotion.matrices` rebuilds F and Q with nested Python loops on every call, which is 13 times per predict for a 6-state track. If `transition` and `predict` accepted `(n_points, n_state)` arrays, each of these would be one array operation. The engine also computes every innovation twice: once for the cost matrix and again inside `update`.

## 13. Comments that repeat the docstring

**The convention.** `docs/conventions/style.md` §10 (:377): "Comment the **why**, never the what. The code already says what it does." Main's comments follow a pattern worth copying:

- **Full sentences, on their own lines, above the code they explain.** Inline comments are rare.
- **They give the reason.** It might be a numerical trap ("scipy rejects a cost matrix containing inf, so the forbidden cost has to be a large finite number", `core/tracking.py:94–98`), the reason a loop can't be vectorised (`core/detection.py:397–399`), or an alternative that was tried and rejected, with its numbers (`core/tracking.py:1194–1200`).
- **They cite.** An equation number, a spec section or an issue.
- **Section dividers are rare.** Where there is one, it's a thin `# ---- #` header at module level (`core/detection.py:820–822`, and three in `core/constants.py`).
- **Parameters, returns and shapes go in the NumPy docstring.** They never go in a comment.

**What the PR does instead.** Two things:

- **Banners that repeat the docstring.** Most functions have a `####` banner plus an `Inputs:`/`Outputs:` block that restates the NumPy docstring right below it. (From here on, "banner" means either kind of block. `tracking_config.py` has only the `Inputs:`/`Outputs:` kind.) Claude counted at bedfde0: 198 banner lines, all in `tracking/`, and 204 `Inputs:`/`Outputs:` lines (190 in `tracking/`, 14 in `pipelines/tracking_config.py`), against 0 of each on main. With the text between them, that's about 1,000 of the 4,400 lines in `tracking/`, roughly a quarter. Move anything not already in the docstring into `Parameters`/`Returns`, then delete the banners. The explanations are genuinely good for learners; they just belong in one place.
- **Inline comments that say what the line does.** For example, in `management.py:223–225`:

  <!-- fmt:off -->
  ```python
  track.hit_count += 1                                      # successful association
  track.miss_count = 0                                      # consecutive miss streak is broken
  track.score += 1                                          # simple quality score used by local track management
  ```
  <!-- fmt:on -->

  Each comment restates its line. The one fact a reader needs isn't said anywhere: `score` is never read (#24). Delete comments like these. Where there *is* a why, write it as a sentence above the block.

Until now only a reviewer checked this rule. Yuting is adding rule R8 to `scripts/check_conventions.py`, which rejects `####` banners and `Inputs:`/`Outputs:` comment blocks, so after that `make check` will catch them. Whether a comment says *why* stays a reviewer's call.

**A worked example: `pipelines/tracking_config.py` L324–340.** This banner is the one to fix first, because it shows every problem at once:

- **Three copies of one text.** It's adapted from the banner on `CartesianMotion.__init__` (`tracking/motion.py:111–125`). Then the docstring below it (:353) says "see CartesianMotion". So the text exists three times, and a reader has to open another file to learn this function's parameters.
- **The layout breaks.** The hanging bullets wrap mid-phrase ("height / in metres", "m²/s³ for / CV"), which is what the lint exemptions (#2) were hiding.
- **Placeholder sections.** Other banners in the file have "Inputs: - None.", which is an empty section written out.
- **Facts in the wrong place.** The unit that changes with the model (m²/s³ for CV, m²/s⁵ for CA) is in a comment, not in `Parameters`, which is where a reader or the rendered docs look. There's no `Returns` section.

Suggested replacement. The function and class names are #5's; the parameters and config fields keep today's names (`noise_density`, `x_m`, `initial_velocity_std_mps`), so rename those per #5 when you paste it. Delete the banner:

```python
def build_tracker_enu(
    axes: Mapping[str, str],
    *,
    origin_lla_deg_m: tuple[float, float, float],
    order: tuple[str, ...] | None = None,
    noise_density: float | Mapping[str, float] = 1.0,
    config: TrackingConfig | None = None,
) -> Tracker:
    """Build a tracker for position measurements in a local east-north-up frame.

    Parameters
    ----------
    axes : mapping of str to str
        The axes to track, each mapped to its motion model, "CV" (constant
        velocity) or "CA" (constant acceleration). x, y and z are east, north
        and up: ``{"x": "CV", "y": "CA"}`` tracks east at constant velocity and
        north at constant acceleration.
    origin_lla_deg_m : tuple of float
        The ENU origin as (latitude in degrees, longitude in degrees, height in
        metres).
    order : tuple of str, optional
        The order of the state elements, by name, e.g. ``("x_m", "xdot_mps")``.
        The default keeps each axis's elements together: x, then y, then z.
    noise_density : float or mapping of str to float, default 1.0
        Continuous white-noise density, one value for all axes or one per axis.
        Its unit depends on the axis's model: m²/s³ (white acceleration) for CV,
        m²/s⁵ (white jerk) for CA. A mapping must name exactly the axes in
        ``axes``.
    config : TrackingConfig, optional
        Filter, gate and lifecycle settings.

    Returns
    -------
    Tracker
        A tracker with no tracks yet, which takes position measurements of the
        chosen axes.

    Notes
    -----
    Each new track takes its position from its first measurement. Its velocity
    starts at 0 with standard deviation ``config.initial_velocity_std_mps``
    (100 m/s by default). On CA axes its acceleration starts at 0 with a fixed
    standard deviation of 10 m/s². A position measurement says nothing about an
    axis left out of ``axes``, so that axis isn't tracked at all.

    References
    ----------
    .. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with
           Applications to Tracking and Navigation*, Wiley, 2001, §6.2
           (continuous white-noise acceleration and Wiener-process
           acceleration models).
    """
```

Two things this surfaces, worth a line each in the code: the 10 m/s² is hardcoded (the `100.0` variance at :374), and the position variance `1.0` there is a placeholder that the first measurement overwrites.

**The rest of the file.** `tracking_config.py` has seven banners, 82 lines in all: 92–99, 129–136, 148–157, 196–213, 247–254, 289–301 and 324–340. Each one gets the same treatment. Two of them hold facts the docstring lacks: the type-mismatch error (:157) and the claim about the prior (:212–213, which is wrong; see #21). Move those facts into the docstring before you delete the banner.

## 14. References

"Bar-Shalom et al." appears 63 times and "Local tracker adaptation" 46 times, usually as the same two lines under every method. Please give one specific reference per module or class, with the chapter or equation: e.g. Wan & van der Merwe (2000) for the UKF, Bar-Shalom et al. for IMM, and Blackman & Popoli for GNN. A student should be able to open the book at the right page. A spec file (`spec/tracker-001-*.md`) isn't a reference.

## 15. Loops need a "why"

`engine.py` has about ten Python loops over track objects. They're fine, but CLAUDE.md asks for a short comment on each saying why it can't be vectorised.

## 16. Reading the tracker

This repo's readers are students, so the question for `tracking/` is whether they can read it: is it clear, accurate and concise?

**Clear: not yet, because of breadth rather than length.** No function is unusually long: `TrackerEngine.process` is 115 lines, and main's `step` is 147. The cost is the number of places a reader has to visit. Claude traced one call of `process` (predict, gate, associate, update, initiate, lifecycle):

- It passes through about 12 files and 5 `Protocol` layers.
- It looks things up in 5 string-keyed registries: `measurement_models`, `sensors`, `Sensor.measurement_model_ids`, `RoutedInitiator.strategies` and `Measurement.measurement_model_id`.
- The estimator factory the initiator calls is a closure defined outside the package, in `pipelines/tracking_config.py:255`.
- State changes in place in several objects: `Track`, `UKF._state`, `IMM.probabilities` and `TrackManager._next_id`.

Main's tracker does the same steps in one file, as pure functions on an immutable `KalmanState` (`core/tracking.py`, `step` at :1101). The Protocols are the right seams (What's good). Most of the cost is in the extras around them: the registries, the aliases and the code with no caller. Removing those (#6) is what makes the rest readable. After that, a "reading the tracker" page, listing the call order with one line per step, would help more than any comment (#23).

**Accurate: several docstrings say something the code doesn't do.** As in #21, these matter more than style, because a student believes the docstring:

- **`UKF.state` (`ukf.py:114`)** says "Return a detached semantic estimate", but it returns `self._state` (:121). The comment above it says "no fresh copy is made here".
- **`TrackSnapshot.state` (`tracks.py:106`)** says "mean (n,), covariance (n,n)", but it returns only the mean (:113).
- **The constructor example (`engine.py:78–79`)** writes `NearestNeighbor()`, `GlobalNearestNeighbor()` and `TrackManager(confirm_threshold=3, delete_threshold=5)`. Neither spelling exists, and neither do those keyword arguments: the real signature is `TrackManager(initiator, policy)` (`management.py:97`). A student who copies the example gets a `NameError`.
- **The `IMM` docstring (`imm.py:105–109`)** offers "Straight (CV) / Accelerating (CA) / Turning (CT)" as modes. The constructor rejects exactly that combination (:123–130), because the three have different state spaces (#11).

**Concise: about a third of `tracking/` is code.** The rest is the banners (#13), the repeated references (#14) and code with no caller (#6). Each of those items shrinks it.

**Are these the objects and names people use?** Mostly, yes. Claude checked the Stone Soup and FilterPy names against their GitHub source:

| This PR | Stone Soup | FilterPy | Note |
| :-- | :-- | :-- | :-- |
| `TrackerEngine` | `MultiTargetTracker` (initiator, deleter, detector, data associator, updater) | — | Hence the `Tracker` rename (#5) |
| `Measurement` (z, R, time, sensor, model) | `Detection` (state vector, timestamp, `measurement_model`) | — | Same idea. Keep `Measurement`: the detection/measurement split is one of the PR's strengths (What's good) |
| `Sensor` (routes measurements, checks coverage) | `Sensor`, which *generates* detections through `measure()` | — | Same word, different job. This is the strongest reason for the `SensorRoute` rename (#5) |
| `UKF` (stateful: `predict_to`, `update`) | split into `UnscentedKalmanPredictor` and `UnscentedKalmanUpdater`, both stateless | `UnscentedKalmanFilter`, stateful, holding `x` and `P` | The PR follows FilterPy. That's fine for teaching; say so in the module docstring, so a student who moves to Stone Soup knows why the shapes differ |
| `TrackManager` + `LifecyclePolicy` | an initiator plus deleters (`UpdateTimeStepsDeleter`, `UpdateTimeDeleter`) | — | The PR folds confirmation and deletion into one manager, which is simpler to read. `lifecycle.py` (#5) names it well |
| `NearestNeighbour`, `GlobalNearestNeighbour`, `IMM`, `UKF` | the same terms | the same terms | Keep them (#5) |

**Is it state of the art?** It doesn't need to be. This is a teaching repo, and the goal is the textbook method, done correctly and readably. The PR has most of the textbook pieces: a scaled UKF with 2n+1 sigma points, chi-square gating, GNN through `linear_sum_assignment`, M-of-N confirmation, and IMM (not wired in, and its mixing untested, #11). It lacks four standard pieces:

- vectorised sigma points (#12);
- a Joseph-form covariance update (#10);
- deleting tentative tracks early (#10);
- association that models clutter (PDA, JPDA or MHT).

The last one isn't needed until a scenario has real clutter; the first three are small.

## 17. The reading view

**Will the reading view pick up the new modules?** Yes, with no changes. `tools/generate_reading_view.py` walks all of `src/radar_forge` (`generate()` at :348, the walk at :372), so `tracking/` and the new `pipelines/` modules get pages automatically. It deploys on push to `main`, so they appear once this merges. Claude ran it on this PR's tree, and it wrote 87 files. Your banners don't appear there, because the view drops comments.

**What looks wrong in it.** The view deletes guard blocks: an `if` whose body only builds a message and raises (style §8). Values computed only *for* a guard are left behind with nothing reading them, so a student sees a computation that leads nowhere:

- `numeric = [...]` in `TrackingConfig.__post_init__`
- `shape = ...` in `detect_product`
- `defaults`, `allowed` and `valid` in `parse_tracking_table`, where a whole `for` loop is left doing nothing
- `scale` in `_numerics.covariance`
- `if self.variant == 'S3': a, b = ...` at the end of `ScenarioTracker.__init__`
- the bare `factory(prior)` in `build_tracker`, whose comment was the only explanation

**The fix is the guard shape from style §8 and §11.** Put guards first. Fold guard-only values into the condition, or into a small `_check_*` helper that the view can drop whole. Build the message as `msg = f"…"`. For example:

```python
        if not np.all(np.isfinite([self.alpha, self.beta, self.kappa])):
            msg = f"alpha, beta and kappa must be finite; got {self.alpha}, {self.beta}, {self.kappa}."
            raise ValueError(msg)
```

**Check it yourself:** run `uv run python tools/generate_reading_view.py` and read your modules' pages in `radar_forge_reading/`. If a line there does nothing, it's residue.

## 18. Tests: what a tracker needs

A tracker can fail in four places, and each needs its own kind of test:

- **The maths.** Coordinate conversions, Jacobians and covariance health. An error here gives a plausible wrong number, not a crash.
- **The kinematics.** Does the filter converge on a known trajectory, and recover after a manoeuvre?
- **Association.** Clutter, missed detections and crossing targets. This is where trackers break in practice, and it's what #7 and #8 are about.
- **Consistency.** Does the covariance match the actual errors (NEES and NIS)? This is the only test that catches a mis-scaled Q or R. Every other test can pass with a filter that is confidently wrong.

**Main already sets the bar.** `tests/core/test_tracking.py` tests main's tracker in all four groups. So the ask is to give the new tracker the same tests, not to invent new ones. The table maps the PR's 42 test functions (82 cases once parametrised) onto the four groups. Claude read all ten test files at bedfde0.

| What | This PR | Main's test to copy | Status |
| :-- | :-- | :-- | :-- |
| Coordinate transforms | `test_measurements.py:46` (3-4-12 geometry, closing-positive), `:107` (a birth inverts the geometry) | — | ✓ analytic, good |
| Jacobian vs a finite difference | `RangeBearingInitiator`'s Jacobian (`initiation.py:238`) is only checked to be positive definite (`test_measurements.py:119`) | `test_the_jacobian_matches_a_central_difference` (:800), helper `central_difference_jacobian` (:743) | gap, lands with the initiator (#6) |
| Singularities | `MonostaticRadar.predict` raises on the vertical axis (`measurements.py:351`), but no test triggers it. `derive_kinematics` at the origin ✓ (`test_measurements.py:92`) | — | gap: testing.md §8 wants a test for every `Raises` |
| P symmetric and positive definite over many steps | none; every filter test is one or two steps | `test_covariance_stays_symmetric_and_positive_definite` (:242), 200 steps | gap, A. It matters here because `kappa = 0` gives a negative centre weight (#24) |
| Innovation and S | indirectly, through the UKF ≡ KF test (`test_ukf.py:16`); periodic wrap ✓ (`:37`) | `test_an_innovation_across_due_north_is_wrapped` (:839) for azimuth | mostly ✓ |
| Near-zero R | none | — | small gap: with R → 0 the measured elements should equal the measurement |
| Tentative → confirmed, confirmed → coasting → deleted | `test_engine.py:21`, `:63`, with strict boundaries | — | ✓ good |
| Tentative track fails M-of-N; coasting grows P | none | `TestTentativeTrackDeletion` (:608), `test_a_tentative_track_dies_as_soon_as_m_is_unreachable` (:446), `test_coasting_grows_the_covariance` (:474) | gap, A. This is #7's early deletion |
| CV and CA convergence | `test_general_tracking.py:76` (RMSE), `:151` (final point) | — | ✓, but see below |
| Manoeuvres: CT, IMM | `CoordinatedTurn` only at ω = 0 (`test_motion.py:68`). IMM mixing never runs (#11) | — | gap, lands with IMM (#6, #11) |
| Clutter and gating | none; no test names `ChiSquareGate` | `TestTrackManagerAgainstClutter` (:569), `test_empirical_acceptance_rate_matches_the_design` (:145) | gap, A. The largest after consistency |
| P_d < 1 | only runs where every detection is missing | — | gap, A |
| Crossing targets | `test_engine.py:72`, `test_port_integration.py:39`; both noise-free and 1-D | — | ✓ but weak, and the two repeat each other (#19) |
| NEES and NIS | none | `TestNormalisedInnovationSquared` (:253); scenario-level NIS in `tests/pipelines/test_scenario_003.py` (criterion 5) | gap, A (UKF) and C (`ScenarioTracker`) |
| RMSE, track continuity | RMSE ✓ (`test_general_tracking.py:110`). `continuity_fraction` is computed but never asserted | — | partial |

**For later: tests that don't apply yet.** These are on every tracking checklist, but nothing in the repo exercises them today. Each one should arrive in the same PR as the scenario or code that needs it (#6), so please note them in the tracker spec's limitations:

- **Doppler blind zones.** A target flying across the line of sight has zero radial velocity, so its velocity can't be observed and it can fall into a zero-Doppler clutter notch. The test checks that the covariance grows while the velocity is unobservable, and that the track survives the notch. It needs an angle-measuring sensor and a crossing trajectory, so it lands with `MonostaticRadar` (#6).
- **GOSPA/OSPA.** One number that combines localisation error, missed targets and false tracks. It only means something with several targets, so it lands with the first multi-target scenario. Stone Soup's metric generators are the reference implementation.
- **JPDA and MHT.** Association that weighs several detections, or several hypotheses, instead of GNN's one best assignment. Neither is implemented. If one is added, it needs the clutter and crossing-target tests in #18 to show it beats GNN, and a track-swap count as the metric.
- **Track purity and fragmentation rates.** These need several targets and a truth-to-track assignment. They land with GOSPA.

**How the existing tests could be stronger:**

- **Derive the bounds.** `rmse[0] < 75` (`test_general_tracking.py:111`) is one range bin, and `< 5` m/s for S1 has no stated reason. For a CV filter in steady state, the expected RMSE follows from q and R. Assert against that, or a stated multiple of it, so the test fails if the filter gets worse. Today it fails only if the filter breaks.
- **One property per test.** `test_confirmation_coasting_deletion_history_and_detached_snapshots` checks five things. When it fails, the name doesn't say which one broke. testing.md §1 asks for a name that states the property: `test_a_track_is_confirmed_on_the_third_hit`, `test_a_snapshot_is_detached_from_the_track`, and so on.
- **One test module per source module** (testing.md §1). Nine of the fourteen `tracking/` modules have none: `association`, `estimation`, `imm`, `initiation`, `management`, `spaces`, `tracks`, `derived` and `_numerics`. Their tests are spread across `test_engine.py` and `test_measurements.py`, so it's hard to see what is untested.
- **Small things.** Every `assert_allclose` needs a comment saying why its tolerance is what it is (§2); about a quarter of the 29 have one. Seeds come from one fixture (§4), as main's `rng` does (:46). The two `@pytest.mark.slow` tests take 0.15 s and 0.36 s here, and `slow` means over a second (§7), so those markers can go.

**Suggested tests.** Claude wrote these and ran both against bedfde0, and they pass. They use today's names; rename per #5.

A consistency test for the UKF. NEES is averaged over independent runs at one time step, so each sum is exactly χ² under a correct filter, and the check is a confidence interval, not a tolerance. It takes about 36 s today, because each UKF step costs about 6 ms; #12 would cut that. Until then it needs `slow`:

```python
@pytest.mark.slow
def test_nees_and_nis_are_chi_squared_over_independent_runs(rng):
    """A consistent filter's NEES is chi2(n_x) and its NIS chi2(n_z).

    Bar-Shalom, Li and Kirubarajan (2001), the filter-consistency section.
    Truth is simulated with the same F, Q and R the filter assumes, so any
    failure is the filter's.
    """
    n_runs, n_steps = 200, 30
    motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN, noise_density=1.0)
    model = CartesianPosition(motion.state_space, ("x_m",))
    r = np.array([[4.0]])
    nees_last, nis_all = [], []
    for _ in range(n_runs):
        p0 = np.diag([100.0, 25.0])
        truth = rng.multivariate_normal([1000.0, 10.0], p0)
        ukf = UKF(StateEstimate(np.array([1000.0, 10.0]), p0, 0, motion.state_space), motion)
        for k in range(1, n_steps + 1):
            f, q = motion.matrices(1.0)
            truth = f @ truth + rng.multivariate_normal(np.zeros(2), q)
            z = truth[:1] + rng.normal(0.0, 2.0, 1)
            ukf.predict_to(k)
            measurement = Measurement(z, r, k, "s", "p")
            nis_all.append(ukf.innovation_statistics(measurement, model).nis)
            ukf.update(measurement, model)
        error = truth - ukf.state.mean
        nees_last.append(error @ np.linalg.solve(ukf.state.covariance, error))
    # The sum over n_runs independent chi2(2) draws is chi2(2 * n_runs).
    low, high = chi2.ppf([0.0005, 0.9995], 2 * n_runs)
    assert low <= np.sum(nees_last) <= high
    # NIS draws within a run are independent for a consistent filter.
    low, high = chi2.ppf([0.0005, 0.9995], len(nis_all))
    assert low <= np.sum(nis_all) <= high
```

A clutter and P_d < 1 test for the engine, modelled on main's `TestTrackManagerAgainstClutter` (:569; the test itself is at :572):

```python
def test_a_target_keeps_its_track_through_clutter_and_missed_detections(rng):
    """P_d = 0.7, three uniform false alarms per scan: one confirmed track, one ID."""
    engine = build_enu_tracker({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    ids = set()
    for t in range(40):
        truth_m = 1000.0 + 20.0 * t
        positions = list(rng.uniform(0.0, 5000.0, 3))
        if rng.random() < 0.7:
            # The same sigma the batch helper declares (R = 0.1 m^2), or the
            # gate is too tight and the test fails for the test's own reason.
            positions.append(truth_m + rng.normal(0.0, np.sqrt(0.1)))
        rng.shuffle(positions)
        snapshots = engine.process(batch(engine, t, positions))
        near = [s for s in snapshots if abs(s.state[0] - truth_m) < 10.0]
        if t >= 10:
            confirmed = [s for s in near if s.status == TrackStatus.CONFIRMED]
            assert len(confirmed) == 1
            ids.add(confirmed[0].track_id)
    assert len(ids) == 1
```

It passes today (five seeds, about 1 s each), so it guards the engine against regressions rather than exposing a bug. The failure in #7 needs one target that gives several detections per scan, as S1's CFAR does. Claude ran a synthetic stand-in for it: three detections per scan scattered within one 74 m bin, with R = Δ²/12. On every one of five seeds it ends with three confirmed tracks on the one target, from frame 10 to the end, because GNN gives each track one detection and all three stay fed. #7's "no births inside a confirmed gate" doesn't stop this, since all three are born in the first scan, before any track is confirmed. The fix is one detection per target (PR B). So add this case as a test that is expected to fail (`xfail`) until PR B lands. It then becomes the regression test for that fix.

**What Claude ran, so you can reproduce it.** Everything below ran against this PR's commit (bedfde0) in a separate worktree:

| What | Result |
| :-- | :-- |
| The PR's 82 test cases, with `--durations` | all pass in 9.4 s. Nothing takes over a second except the plot test (3.4 s, most likely the first matplotlib import). The two tests marked `slow` take 0.15 s and 0.36 s |
| The NEES/NIS test above | passes, in about 36 s |
| The clutter and P_d = 0.7 test above | passes on five seeds, about 1 s each |
| One target giving three detections per scan | fails on all five seeds: three confirmed tracks on one target |
| The truth-independence comparison (#19) | the altered frame's IQ arrays are the same objects as the original's |

One more observation, for discussion rather than a fix. Claude's first draft of the clutter test drew noise with σ = 1 m but declared R = 0.1 m², so the gate was too tight. The true detection then fell outside the confirmed track's gate and started a second track on the same target, which was confirmed alongside the first. The trigger was the test's mistake. But it shows that this tracker, when a confirmed track's gate misses its target, starts a duplicate instead of re-acquiring. Main's tracker widens the gate and re-acquires (`test_a_widened_track_re_acquires_its_target_and_keeps_its_id`, `tests/core/test_tracking.py:496`). Is that a behaviour you want to carry over?

## 19. Tests that cost more than they catch

A test earns its place when it can fail for a reason someone cares about. Some tests here can't fail at all, and others will fail exactly when you make the changes this review asks for. Both kinds cost CI time and teach the next reader the wrong thing about what matters.

- **`test_port_integration.py:12` locks in the fork.** It asserts that the two trackers are *different* (`general.Track is not original.Track`, `GeneralScenarioTracker is not ScenarioTracker`, :24–26). It also checks that no module path contains "DEPRECATED", which is left over from the port, and that the package directory is called `tracking`. All of this breaks the moment #3 and #5 land, so the test works against the review. Its one useful check, that importing doesn't load matplotlib, is already on main (`tests/test_import.py:34`). **Delete it.**
- **`test_sensors.py:11` tests a re-export.** It checks that `measurements.Measurement is sensors.Measurement` and which module owns the class. That's a test of file layout, for a compatibility shim that #6 removes. **Delete it with the shim.**
- **The "truth independence" half of `test_general_tracking.py:44` can't fail.** `replace(frame, range_m=1e6, …)` changes only the truth fields. `frame.iq` is a stored field (`scenarios.py:284`), so the altered frame carries the *same* IQ arrays. Claude checked: they are the same objects. Both trackers get identical products, and `process(timestamp_s, products)` has no way to see truth anyway. So the second tracker only shows that the pipeline is deterministic. The guarantee you want, that no truth reaches the tracker, is already given by `process`'s signature. **Drop the second tracker**, and keep the rest as a smoke test from IQ to tracks. `confirmation_latency_s == 2` is fine: it follows from 3-of-5 at 1 s frames. Say so in a comment, so it doesn't read as a recorded number.
- **`test_run_general_tracking.py:114` asserts the runner ignores the scenario's own `[tracking]` table.** That's the behaviour #20 and #22 ask you to change. Its first assert also repeats `:27`. **Delete it when #22 lands.**
- **Duplicates.**
  - `test_port_integration.py:39` is `test_engine.py:72` plus a second sensor and a reversed measurement order. Merge them into one test that keeps the new parts (#6).
  - `parse_tracking_table`'s validation is tested twice: directly (`test_general_tracking.py:170`) and again through the runner (`test_run_general_tracking.py:47`). Keep the direct one. The `ModelRegistry` half of `:170` tests code #6 defers.
  - `test_run_general_tracking.py:58` runs the CLI in a subprocess three times (0.7 s each) and asserts that the files aren't empty. The per-variant behaviour is already tested in `test_general_tracking.py`, so one variant is enough here.

**Not nuisance, and worth keeping:**
- `test_track_history.py`: it checks that the plot says the range is modulo, which is something a student must not miss. It moves with #5.
- The binomial P_fa test in `test_detection_2d.py`.
- The small analytic GNN test (`test_engine.py:14`).

## 20. One file format

This is the change that costs most if it merges, because file formats outlive code.

**What changed since you opened the PR.** `spec/data-001-formats.md` landed on main on 29 September, four days after this PR. It specifies every file a run writes: `metadata.json` (§6.1), `detections.csv` (§6.5), `tracks.csv` (§6.6) and `metrics.csv` (§6.9). So this is a moving target, and none of the mismatches below are your fault. Please write what data-001 specifies. It is a superset of main's current columns (§12), so main's readers keep working.

**The problem today.** The PR writes `detections.csv` and `tracks.csv`, the same file names as `scripts/run_scenario.py`, but with different columns. Any notebook that reads one will silently misread the other. Against data-001 (the bullets list only what the PR gets wrong; §6.5 and §6.6 have the full column lists, in order):

- **`detections.csv` (§6.5):**
  - `frame`, `detection_id` and `sensor_id` are missing. `frame` (with `time_s`) is the join key across every file of a run, and `(frame, detection_id)` is the detection key (§5).
  - `radial_velocity_mps` should be `velocity_folded_mps`, and `unfolded_velocity_mps` should be `velocity_unfolded_mps` (the family word goes first, #5).
  - `range_bin` and `doppler_bin` should be `range_index` and `velocity_index`.
  - `noise_power_linear` isn't a unit, and noise power isn't a detection column. Write `peak_power_w` (as main does) and `total_power_w`. The noise power goes per burst in `metadata.json` (`bursts[].noise_power_w`, §6.1).
  - Keep `status` and `pair_id`. §6.5 doesn't have them yet; Yuting is adding them to data-001.
- **`tracks.csv` (§6.6):**
  - `state_json` and `covariance_json` put a matrix inside a cell. DF3 keeps the covariance tabular instead. Write one `var_<field>` column per state element, with the unit squared (`var_east_m2`), and `cov_<i>_<j>` columns for the off-diagonals, where `i < j` are positions in `metadata.json`'s `tracking.state_fields`. That keeps the full covariance on disk as plain columns.
  - `nis`, `n_hits`, `n_misses_in_a_row`, `range_rate_mps` (you write `radial_velocity_mps`), `associated` and `associated_detection_id` are missing.
  - `status` has no `coasting`.
- **Metrics (§6.9):** `tracking_metrics.json` should be `metrics.csv`, in tidy long form: one row per metric, with columns `metric`, `track_id`, `target_id`, `frame_start`, `frame_end` and `value`.
  - `radial_velocity_rmse_mps` should be `range_rate_rmse_mps`.
  - `variant`, `evaluation` (a sentence) and `detection_status_counts` (a dict) aren't metrics. The first two belong in metadata. The status counts can be one row per status, e.g. `n_detections_ambiguous_pair`.
- **`metadata.json` (§6.1):**
  - It writes `legs`, `chirp_time_s` and `n_chirps` where main writes `bursts`, `chirp_duration_s` and `n_pulses`.
  - `velocity_sign: "positive_closing"` should be `conventions.velocity_sign: "closing_positive"`.
  - `state_fields` is missing, and the full `asdict(state_space)` goes in its place.
- **Per-run constants on every row.** `range_interpretation` and `velocity_interpretation` have the same value on every row of a run. Write them once, in `metadata.json`.
- **TOML.** The `[detection]` and `[tracking]` tables reuse main's table names but with incompatible keys. Extend main's tables instead (#22).

**Writers that only write.** You asked what to call the writer. "Serialise" is the general word; Python's own spelling is `json.dump`/`dumps`, and pandas uses `to_csv`/`to_dict`. Main's convention is `write_<file>` for a function that writes a file (`write_metadata`, `write_tracking_csvs`) and `<thing>_metadata` for one that builds a dict (`burst_metadata`). So `write` is fine.

The real problem is that `TrackingWriter.write` (`tracking_output.py:128–205`) does two jobs:

- **Writing.** It writes one row per detection and one per track.
- **Scoring.** In the same loop, it picks the primary track, measures confirmation latency, and accumulates range and velocity squared error against truth, with a range wrap.

That's why it needs `truth_range_m` on every call, and why its metrics can't be recomputed from the files it writes. Please split it:

- **One function per file,** each taking rows and writing them, with the columns in one tuple in §6's order. An iterable of rows keeps your streaming (the comment at :151).
- **Scoring as its own function:** frames and truth in, metric rows out. Then `write_metrics_csv` writes those rows like any other table.

```python
# data-001 §6.5, in its column order. When pipelines/io lands (data-001 §12,
# the pipelines/io follow-up), this tuple moves into its schema registry unchanged.
DETECTION_COLUMNS = ("frame", "time_s", "detection_id", "sensor_id", "range_m", ...)


def write_detections_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None: ...
def write_tracks_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None: ...
def write_metrics_csv(path: Path, rows: Iterable[MetricRow]) -> None: ...
def score_primary_track(frames: Iterable[FrameTracks], truth: TargetTrack) -> list[MetricRow]: ...
```

This sketch is Claude's proposal, not spec wording. The names are placeholders.

**Where the writers go.** data-001 plans one general writer: `pipelines/io/`, a schema registry plus stdlib CSV/JSON writers that every file goes through (§2.6, §9, and §12's `pipelines/io` follow-up). It doesn't exist yet, and it isn't yours to build in this PR. Until it does, put the writers next to main's in `scripts/run_scenario.py`, extending `write_metadata` and `write_tracking_csvs`. Don't put them in a new `pipelines/tracking_output.py`, and don't make them a writer class. With the columns kept in tuples, moving the writers into `pipelines/io` later is a cut and paste.

## 21. `ScenarioTracker` and the pipeline docs

`ScenarioTracker` (`pipelines/general_tracking.py:158`) is the first class a student meets when they run a scenario, so it's the one most worth getting right. It doesn't yet follow the conventions main's `pipelines.tracking.ScenarioTracker` (`tracking.py:716`) set:

| | Main | This PR |
| :-- | :-- | :-- |
| Form | `@dataclass` | plain class |
| Settings attribute | `.tracking`, same as the parameter | `.config`, although the parameter is `tracking` (:194) |
| Per-frame call | `step(product, *, frame_index, time_s, …)`, keyword-only | `process(timestamp_s, products)`, positional |
| Time name | `time_s` | `timestamp_s` |
| Errors | `msg = f"… got {value}"`, then `raise` | inline literals that don't say what the bad value was |
| Docstring | Attributes; `step` has Returns and Raises | no Attributes; `process` has no Returns or Raises, though it raises at :288 and :290 |
| Shapes | `(n_doppler_bins, n_range_bins)` style | `(2,2)` (:51), `(n_doppler,n_range)` (:109) |
| References | a real citation with a section | "ScenarioTracker class references" (:285), a pointer to nothing |

Two design points:

- **Scenario labels in library logic.** "S1"/"S2"/"S3" are set at :195–197 and compared at :198, :200, :210, :220, :292 and :295, and in `tracking_output.py` at :169. The code should branch on what actually differs: does range fold (`range_period_m is not None`)? Are there two bursts? Keep the label for output only. Then a fourth waveform doesn't need a fourth string threaded through three modules.
- **Hidden IDs.** `"sensor"` and `"measurement"` (:230, :304) only work because they match `build_tracker`'s defaults (`tracking_config.py:221–222`). Pass them explicitly, or make them constants both sides import.

**Docs that say something the code doesn't do.** Students trust a docstring over the code beneath it, so these matter more than style:

- **`general_tracking.py:1`, "the three Duke range/Doppler variants".** Duke is the receiver site for scenarios 001–003, not a name for these variants (#23). Say "scenario 001's three variants (S1, S2, S3)".
- **`tracking_config.py:224`, "an estimator-independent engine".** It always builds a `UKF` (:255–256). There's a second catch. With a custom `initiator`, `alpha`, `beta`, `kappa` and `prior` are only *validated* (:259) and never used, which makes the banner's "the prior is reserved for later track starts" (:212–213) false in that case. Say so, or reject the combination.
- **`tracking_config.py:49`, "for the radial scenario adapter".** `build_enu_tracker` uses the same class for ENU tracks (:368).
- **`parse_tracking_table` (:158).** It parses `DetectionConfig` too, so the name is too narrow: `parse_config_table`, or fold it into main's loader (#22). It also raises on a type mismatch (:191–192), which only the banner (:157) says. Add it to a Raises section.
- **`general_tracking.py:124–126`, a misleading message (minor).** "map shape must match finite, uniformly increasing … axes" is raised by a check of shape, size and timestamp only. The uniformity check is the next one (:128–133). Suggest "map shape must match its axes, and timestamp_s must be finite", with the values.
- **`tracking_output.py:43`, "deleted primary tracks remain visible in metrics".** No metric field mentions deletion. A deleted primary track shows up only as a lower `continuity_fraction`. `n_tracks_created` is the highest track ID seen (:174, :218), which equals the count only because IDs start at 1 and step by 1 (`management.py:100`). Say how each thing is visible.
- **`tracking_output.py:44` and :234, "interpolation-derived truth".** This is true, but it describes the *input*: the truth comes from `pipelines/trajectories.py`, which interpolates the fixes onto the frame grid. The writer itself does no interpolation. Move the sentence to `write`'s `truth_range_m` parameter and point at `trajectories.py`.
- **`tracking_output.py:268`, "…state, covariance and lifecycle".** Only `status` is written, and the state goes under the key `"mean"`.
- **`tracking_output.py:176–177` and :199–201, two undocumented range wraps.** Exported range wraps to [0, P) and the range error to [−P/2, P/2). Neither is documented. The second duplicates `StateSpace.normalize` (`spaces.py:186`). The runner wraps range a third time (#22). Put one documented wrap helper on `StateLayout` (today's `StateSpace`, #5) and call it from all three places.
- **`tracking/spaces.py:200`.** The comment says "(-180, 180]"; the docstring says [−period/2, period/2).

**Suggested text.** It uses #5's names where they affect the docs (`bursts`, `.tracking`, `Tracker`, `FrameTracks`); the rest keep today's names (`timestamp_s`, `engine`, `RadarDetection`), so rename those per #5 as you paste. The Notes are Claude's reading of the code, so please correct anything that's not what you meant.

The class docstring:

```python
class ScenarioTracker:
    """Detect, pair and track one scenario-001 variant, frame by frame.

    Parameters
    ----------
    bursts : tuple of Radar
        One FMCW burst (S1), one pulsed burst (S2), or two FMCW bursts at
        different PRFs (S3).
    detection : DetectionConfig, optional
        CFAR settings.
    tracking : TrackingConfig, optional
        Filter, gate, lifecycle and dual-PRF search settings.

    Attributes
    ----------
    bursts : tuple of Radar
        As passed in.
    detection : DetectionConfig
        The CFAR settings in use.
    tracking : TrackingConfig
        The tracking settings in use.
    variant : {"S1", "S2", "S3"}
        The scenario-001 variant, from the number of bursts and their waveform.
        Used to label output only.
    range_period_m : float or None
        The unambiguous range when range folds (S2), otherwise None.
    motion : RadialMotion
        The range and range-rate motion model.
    observation : CartesianPosition
        The measurement model: range alone for S1, range and velocity otherwise.
    engine : Tracker
        The tracker that holds the tracks, built by ``build_tracker``.

    Raises
    ------
    ValueError
        If there are not one or two bursts; if two bursts are not both FMCW;
        or if two bursts share a velocity interval, or
        ``tracking.max_velocity_mps`` does not exceed the smaller of the two.

    Notes
    -----
    In S2, range folds. The radar only ever sees range modulo
    ``range_period_m``, so it can't say which fold the target is in. The
    filter's range state is left unwrapped: it sits on whichever fold the first
    detection happened to give it, and only its value modulo
    ``range_period_m`` means anything. That is why the exported range is
    wrapped.

    In S3, a peak in one burst is paired with a peak in the other only when each
    has exactly one partner within a range-resolution cell. When two targets
    are close in range, their peaks are marked ``"ambiguous_pair"`` and dropped
    rather than guessed, and nothing here looks at the true trajectory.

    References
    ----------
    .. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
           McGraw-Hill, 2014, §1.3 (PRF and ambiguity).
    .. [2] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking
           Systems*, Artech House, 1999, §4.3 (Doppler-aided tracking and
           ambiguity resolution).
    """
```

The `process` (or `step`) docstring:

```python
        """Detect, pair and track one frame.

        Parameters
        ----------
        timestamp_s : float
            Frame time, in seconds. Must not be earlier than the previous frame.
        products : tuple of RangeDopplerProduct
            One range-Doppler map per burst, in the same order as ``bursts``.

        Returns
        -------
        FrameTracks
            Every detection with its status, the track snapshots after this
            frame's update, and how many measurements reached the tracker.

        Raises
        ------
        ValueError
            If the number of products differs from the number of bursts, or
            ``timestamp_s`` is earlier than the previous frame's.
        """
```

The `_pair` docstring, and a comment on its indexing. #6 moves `_pair`'s behaviour into main's `dual_prf_measurements`, so put this text there, not on a new `_pair`:

```python
        """Pair S3's peaks across the two bursts and unfold their velocity.

        Two peaks are compatible when their ranges differ by at most one
        range-resolution cell. A pair is used only when it's one-to-one: each
        peak has exactly one compatible partner in the other burst. A peak with
        two or more partners is marked "ambiguous_pair"; a peak with none keeps
        "missing_pair". A one-to-one pair whose velocities don't agree on any
        unfolded value within ``tolerance_mps`` is "unresolved_velocity".

        Returns
        -------
        records : list of RadarDetection
            Every peak from both bursts, the first burst's first, with its status.
        measurements : list of Measurement
            One per accepted pair, at the first burst's range and the unfolded
            velocity.
        """

        ...
        # records holds the first burst's peaks, then the second's, in one flat
        # list, so the second burst's peak j is records[len(a) + j].
```

The "why" comment in `_measurement`:

```python
        # Measure only what this variant doesn't fold. S1's Doppler folds, so S1
        # measures range alone. S2's range folds but its Doppler doesn't, so it
        # measures both, with range periodic in range_period_m. S3 measures range
        # and the velocity unfolded from its two PRFs.
```

The comment on the prior in `__init__`:

```python
        # DirectStateInitiator overwrites every measured element of this prior,
        # and its timestamp, with the first detection. So the range mean 0, its
        # 1 m² variance and the time 0 are placeholders that never reach a
        # filter. Only the velocity variance is used, and only in S1, which
        # doesn't measure velocity.
```

Also move the S3 guard (:210–215) to the top of `__init__`, with the other checks, so the object is never half-built (style §8).

`detect_product`: its shapes, Returns and Raises, and a citation in place of the bare formula. As with `_pair`, this text goes on main's `frame_detections` once `detect_product`'s behaviour moves there (#6):

```python
    """Find CFAR peaks in one range-Doppler map, without reading simulation truth.

    Parameters
    ----------
    product : RangeDopplerProduct
        The map, shape (n_doppler_bins, n_range_bins), and its bin axes.
    ...

    Returns
    -------
    tuple of RadarDetection
        One per CFAR peak, at the bin centre, with covariance
        diag(Δr²/12, Δv²/12) floored at the config's floors.

    Raises
    ------
    ValueError
        If the map's shape doesn't match its axes, an axis has fewer than two
        bins or isn't finite, uniform and increasing, or ``timestamp_s`` isn't
        finite.

    Notes
    -----
    A peak says only that the target is somewhere in a bin of width Δ. Treated
    as uniform across the bin, that position has variance Δ²/12.

    References
    ----------
    .. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
           McGraw-Hill, 2014, §6.5 (CFAR), §7.3 (measurement accuracy).
    """
```

## 22. Pipeline size and placement

**Is there too much pipeline code?** Yes. `general_tracking.py`, `tracking_config.py` and `tracking_output.py` add 969 lines: 520 of code, 270 of docstring, 82 in `Inputs:`/`Outputs:` comment blocks (all in `tracking_config.py`) and 93 blank. Most of it redoes what main's `pipelines/tracking.py` (1,069 lines) already does: detect, unfold, track and write CSVs. Almost none of it has a caller outside this PR's own new scripts, and `ModelRegistry` is used only by tests.

**Repetition inside the three modules.** Even before comparing with main, the new modules say several things more than once. A reader has to check that each copy agrees with the others, and a fix has to find every copy:

- **The variant label is tested seven times.** `ScenarioTracker.__init__` works out "S1"/"S2"/"S3" once (`general_tracking.py:195–197`), and the code then compares against it at `:198`, `:200`, `:210`, `:220`, `:292`, `:295` and `tracking_output.py:169`. The range interpretation is derived again from `range_period_m` at `tracking_output.py:148–150` and `:224–226`. #21 has the fix: branch on what differs (does range fold? are there two bursts?) and keep the label for output.
- **Range is wrapped three ways.** `range_m %= period` (`tracking_output.py:176–177`), a centred wrap of the error (`:199–201`), and `frame.range_m % leg.unambiguous_range_m` in the runner (`run_general_tracking.py:213`). One documented helper on `StateLayout` (today's `StateSpace`) replaces all three (#21).
- **Each CSV's columns are written twice, matched by position.** The detection header (`tracking_output.py:79–96`) and its row (`:155–170`) are two 14-item lists that must stay in the same order, and so are the track header (`:97–109`) and its row (`:178–189`). `snapshot_record` (`:257–281`) writes the same snapshot a third way, with different keys (`mean` vs `state_json`). Main keeps one column list per file (`DETECTION_COLUMNS`, `scripts/run_scenario.py:64`) and builds each row by name. #20 has the shape.
- **The prior is built twice.** It's zero mean with the same variances in `general_tracking.py:203–208` and `tracking_config.py:373–379`.
- **Main's work, done again.** `_pair` re-implements `dual_prf_measurements`, down to the tolerance rule and the 191 m/s default (`tracking_config.py:84`). The lifecycle fields are declared twice (#6). About 280 lines of the runner repeat `scripts/run_scenario.py` (#5).

Most of this goes away when the pieces move to their homes below. The rest is one helper or one table each.

**Does some of it belong elsewhere?** Yes. `pipelines/` is the scenario glue (#5). Anything that doesn't read a scenario belongs in `core/`, and anything that repeats main belongs in main's version:

| Piece | Home |
| :-- | :-- |
| `build_tracker`, `build_enu_tracker` (neither knows about scenarios) | `core/tracking/`, or the example script for `build_enu_tracker` (#5). Defer `ModelRegistry` (#6) |
| `TrackingConfig` | split it: filter and lifecycle fields go to `core/tracking/` next to `LifecyclePolicy`; the scenario fields (`max_velocity_mps`, `unfolding_tolerance_bins`) stay in `pipelines/` |
| `parse_tracking_table` | add its strict type check to main's `load_scenario` → `_require_keys` → `configs_from_scenario`, and read the scenario TOML's own `[tracking]` table. `--tracker-config` then goes |
| All three range wraps | one helper on `StateLayout` (#21) |
| `snapshot_record` | next to `TrackSnapshot`, in `tracks.py`, or deferred (#6) |
| `TrackingWriter` | extend `write_tracking_csvs` in `scripts/run_scenario.py`. Keep writing files apart from scoring (RMSE, latency): they're two jobs, and scoring needs truth, which writing doesn't |
| `detect_product`, `_pair` | main's `frame_detections`, `dual_prf_measurements` (#6) |
| Variant labels | output only (#21) |

After that, what's left for `pipelines/tracking.py` is a small adapter: take one product per burst, make measurements, call the tracker. #6 lists the reuse targets and #20 the file formats, so this item doesn't repeat them.

## 23. Docs for the next reader

- `tracker-001-port.md`, `tracker-001-integration.md` and the README section talk about "approved substitutions", "supplied by the user", "the deprecated checkout" and SHA-256 inventories. That's a working log. A future reader needs the design, the limitations and how to run it. Move the port history to the PR description, and merge the two specs, since their scope overlaps. If a coding agent helped draft them, that's fine, but you should be able to stand behind every sentence.
- **The README section.** `## General tracking` (`README.md:214`) doesn't follow main's README. Main has no per-feature sections: it covers tracking in the Status list (a spec link) and in the Scope table (`core` … tracking). This section also sits after `## Licence`, which should stay last. And "general" is the name problem from #5. Please drop the section, and move its run commands and limitations into the merged tracker spec. Then add one line to the Status list, in the style of the scenario 003 entry. Something like this (Claude's wording; the spec file name depends on the merge):

  ```markdown
  - [`spec/tracker-001.md`](spec/tracker-001.md) — multi-sensor tracking in `core/tracking/`:
    UKF, GNN association and explicit measurement models, run on scenario 001's S1/S2/S3.
  ```

  In `docs/README.md:38`, change "general tracker" in the same way.
- The "General tracking extension" section appended to `spec/scenario-001-xband.md` puts tracking into scenario 001. Tracking is scenario 003's job, so fold it into `spec/scenario-003-tracking.md`.
- "Duke integration", "Duke windows" and "the Duke runs" use a real name for the wrong thing. On main, Duke is the receiver *site*: the Duke Receiver, shared by scenarios 001, 002 and 003. So "Duke" doesn't pick out scenario 001's three variants. Say "scenario 001 S1/S2/S3", which is what the scenario TOMLs call them. The same fix applies to the `general_tracking.py` module docstring (#21).
- `docs/tracking/_michael_notes.md` has useful notes, but personal notes don't belong in `docs/`. Put them in the PR description, or turn them into a "reading the tracker" page.
- The `structure.md` edit: see #3.

## 24. Nits

- **`--track` does nothing.** In `run_general_tracking.py` it is `store_true` with `default=True`, so the flag has no effect and the `tracker is not None` branches never run. Main's convention is `--no-tracking` (every switch is a `--no-*` opt-out).
- **Two process-noise conventions.** `core/tracking.py` uses discrete white-noise acceleration (q in m²/s⁴), and the PR uses continuous white noise (q in m²/s³). Both are fine; add a sentence in each docstring pointing at the other.
- **UKF `kappa = 0`.** With n = 2 the centre weight is −3. That's legal, but it makes the covariance easier to push non-positive-definite. Stone Soup uses `kappa = 3 − n` with the same `alpha`/`beta` as yours. Pick one and say why.
- **Matches before cost.** `GlobalNearestNeighbour` always maximises the number of matches before cost, as `core.tracking.associate_gnn` and Stone Soup's default do. Say so in the docstring, and mention a finite non-assignment cost as the alternative.
- **Unused `Track` fields.** `Track.score`, `age` and `existence_probability` are never used in a decision. Use them or drop them.
- **Late `ValueError`.** `DirectStateInitiator` can raise `ValueError` mid-scan, after other tracks were already updated. Check that the prior blocks are independent in `__init__`.
- **One timestamp per batch.** Every detection in a batch must carry exactly the batch timestamp. A scanning radar stamps each detection separately, so add this to the limitations.
- **The commit subject.** "feat(tracking): add general multi-sensor tracking" has the same "general" as #5. When you split the PR, say what each commit adds, e.g. `feat(tracking): add a UKF to core/tracking`.
- **New public names.** `radar_forge/__init__.py` exports a new top-level `tracking`, and `core/__init__.py` exports a second `DetectionConfig`. Both go away with #3 and PR B. Anything in `__all__` is public API, and removing it later needs a deprecation.

## Why these matter later

These are the learning points. Each is cheap to fix now and expensive to fix after merge.

| Habit in this PR | What it costs later |
| :-- | :-- |
| Same file names, different columns (#20) | Readers silently misparse; old output files can't be compared with new ones |
| Two classes with one name (#5) | Every reader checks imports to know which `Track` they have; aliases like `General*` spread into user code |
| A parallel package instead of extending `core` (#3) | Two trackers to teach, test and fix; bug fixes land in one and not the other |
| Code with no caller yet (#6) | It isn't exercised, so it rots; reviewers can't judge it without a use |
| Lint/format exemptions (#2) | Every future edit to those files bypasses the gate, and `force-exclude` means the pre-commit hook doesn't catch it either; the exemptions are never removed |
| A name that says "general" (#5, #23) | It means "not the other one", so it stops meaning anything once the other one is gone, and it spreads into headings, flags and file names |
| Public names added to `__all__` (#24) | Removing them needs a deprecation cycle |
| A second word for one of main's ideas ("leg" for burst, #5) | Readers have to learn that two words mean one thing, and the second word gets into file formats, where it is hardest to change |
| A no-op CLI flag (#24) | Users think they turned something off |
| Importing another package's `_private` module (#6) | The private module can't change without breaking a caller it didn't know about |
| Docs written as a port log (#23) | The next reader can't find the design among the history |
| Docs that contradict the code (#21) | Students trust a docstring over the code beneath it, so a wrong one teaches the wrong thing, and nobody rereads it once it has passed review |
| Values computed only for a guard (#17) | The reading view shows computations that lead nowhere, and a student tracing the maths can't tell they're dead |
| No consistency test (#18) | A mis-scaled Q or R passes every other test; the filter is confidently wrong and nothing says so |
| A test that asserts today's structure (#19) | The next refactor has to delete tests before it can start, and a reader takes the structure for a requirement |
| A test that can't fail (#19) | It reads as a guarantee that nobody has checked, and it costs CI time on every push |
| A writer that also scores (#20) | The metrics can't be recomputed from the files; changing a metric means changing the writer, and scoring a different track means rerunning the whole scenario |
| A docstring example that doesn't run (#16) | It's the first thing a student copies, and the first error they meet is in the documentation, not their code |
| One idea written out several times (#22) | A fix lands in one copy and not the others, and a reader has to diff the copies to learn they agree |

## What's good

- **Analytic tests.** That's exactly what this repo wants. The best ones:
  - the UKF against the closed-form KF at 1e-10;
  - F and Q unchanged by a permutation of the state;
  - Q against the published CV and CA blocks;
  - the coast and deletion limits tested exactly at their boundaries;
  - snapshots shown to be detached from the live track;
  - one test for each dual-PRF status;
  - a binomial interval, not a tolerance, for the 2-D CFAR's P_fa.
- **Gate and conventions.** `mypy --strict` is clean, units are in most names, and there are no new dependencies. There's no `np.tile`/`repeat`/`broadcast_to` and no hardcoded speed of light.
- **Limitations.** They're stated honestly rather than tuned away.
- **The split inside the package is right.** Motion model, measurement model, estimator, gate, associator, initiator, lifecycle and engine each have a small `Protocol`, and those are Stone Soup's seams. Stone Soup splits prediction from update and keeps confirmation apart from deletion, while yours uses a stateful filter as FilterPy does, and one lifecycle manager. For teaching, that's the simpler choice (#16). #5 and #6 ask you to rename, move and trim, not to redesign.
- **The maths checks out.** Claude checked it line by line against the published equations:
  - UKF weights, the gain via `solve`, and the symmetrised update
  - the exact CV/CA transition and the continuous-noise Q
  - `CoordinatedTurn`'s sinc forms, which are stable as ω → 0
  - the spherical-to-ENU Jacobian
  - the monostatic and bistatic closing-positive rates
  - IMM mixing and the log-space mode update
  - `ChiSquareGate` sharing one Cholesky factor for gate, score and gain
  - circular means on periodic coordinates
  - `cfar_2d`'s `N·expm1(−ln Pfa / N)` and its binomial-banded Pfa test

**Where this PR does better than `main`.** Keep these through the merge, and in places main should change to match:
- **Detection vs measurement.** Main's `pipelines.tracking.Measurement` is really a detection. Yours separates what the detector found from what the filter consumes (z, R, time, sensor, model), which is the textbook distinction.
- **R from the map.** You compute R as Δ²/12 from the product's own axes, with a floor. Main freezes `sigma_range_m = 21.6356…` (one bin / √12) for one waveform.
- **Per-batch timestamps.** You take a timestamp per batch and reject out-of-order frames. Main uses a fixed `frame_time_s`. Yours is what multi-sensor tracking needs.
- **Every detection says why.** `status` and `pair_id` make dual-PRF failures visible.

## Re-review checklist

Each step says what to do and how Yuting will check it, so you can run the same check first. The order matters: each step makes the next one smaller. Steps are named by PR (L for the licence, A, C and D for the split PRs, E for every PR), so a step's name never clashes with a section number like #6. PR B (2-D CFAR) is Yuting's, so it isn't here.

Commands run from the repo root on your branch. If your shell exports `VIRTUAL_ENV` for another checkout, `uv run` silently uses that checkout's code: run `env -u VIRTUAL_ENV uv sync --extra dev` once, and prefix `env -u VIRTUAL_ENV` to the `uv run` and `make` commands below.

**Before any PR**

- [ ] **L1. Licence (#1).** *Do:* say who wrote the original, in the PR and in `spec/tracker-001-provenance.md`. *Check:* Yuting reads the answer. Nothing else merges until this is settled.

**PR A: `core/tracking/`**

- [ ] **A1. The move (#3).** *Do:* turn `core/tracking.py` into the `core/tracking/` package in its own commit, with no behaviour change. *Check:* `test ! -e src/radar_forge/core/tracking.py`. On that commit, `uv run pytest tests/core` passes with no test edited, and `git grep -nE 'radar_forge\.tracking\b|from radar_forge import tracking'` returns nothing.
- [ ] **A2. Renames (#5).** *Do:* apply the rename table, including every "leg". *Check:* `git grep -nE '\b(legs?|leg_[a-z_]+|[a-z_]+_legs?)\b|leg\{|rd_leg|"leg' -- src scripts tests ':!src/radar_forge/core/radar.py'` returns nothing (it finds 70 lines today; `git grep -w leg` alone misses `legs` and `leg_index`). `git grep -nE '\bgeneral([ _-]?(track|runner|tracker|scenario)|[A-Z])|General[A-Z ]' -- README.md docs spec src scripts tests` returns nothing (it finds 31 lines today, including the README heading).
- [ ] **A3. Reuse, remove, defer (#6).** *Do:* call the `core` functions in the reuse table, delete the in-PR duplicates, and move the deferred code to a branch. *Check:* `git diff --stat origin/main...` lists no `imm.py` or `derived.py`, and `git grep -nE 'class (IMM|CoordinatedTurn|ModelRegistry)\b'` returns nothing.
- [ ] **A4. Behaviour (#7, #8).** *Do:* confirmed tracks pick first, no births inside a confirmed gate, and a process-noise default derived in its docstring. *Check:* this needs the pipeline, so it's checked in C4.
- [ ] **A5. Tracker logic and docs (#16, #10).** *Do:*
  - Delete a tentative track once it can't reach M hits in its first N scans, and add #10's test for it.
  - Use a Joseph-form update, or say in the docstring why the simple form is safe, backed by the 200-step covariance test.
  - Fix the four docstrings in #16.
  - Make the smaller fixes in #10.

  *Check:*
  - #10's test passes.
  - `git grep -nE 'NearestNeighbor\(|confirm_threshold=|detached semantic estimate|Straight \(CV\)' -- src` returns nothing (it finds 7 lines today).
  - `git grep -n 'cost: str' -- src` returns nothing.
- [ ] **A6. Speed (#12).** *Do:* batch the sigma points, and compute each innovation once. *Check:* no list comprehension over sigma points is left in `ukf.py`.
- [ ] **A7. Comments, references and loops (#13, #14, #15, #9).** *Do:* fold each banner into its docstring and delete it; give one specific reference per module or class; add a "why" comment on each loop; add the NIS sentence to the `cost` docstring. *Check:* `git grep -nE '^\s*# (This function|Inputs:|Outputs:)|^\s*#{10,}' -- src` returns nothing. Once R8 is on main (#13), `make check` enforces this for you. `git grep -nE 'class references|Local tracker adaptation' -- src` also returns nothing. Yuting reads each `for` in `tracker.py` for its comment.
- [ ] **A8. Tests (#18, and #11 if IMM comes in).** *Do:*
  - For the code in PR A, add the tests in #18's table that main already has: NEES/NIS, clutter with P_d < 1, the gate's empirical acceptance rate, covariance health over many steps, tentative-track deletion and coasting growth.
  - Add the several-detections-per-target case as `xfail` until PR B lands.
  - Split the multi-property tests, and give each `tracking/` module its own test module.
  - Add the Jacobian and singularity tests with the code they test.
  - List #18's "for later" tests in the tracker spec's limitations.

  *Check:* `git diff origin/main... -- tests | grep -E '^\+\s*def test_'` lists the new tests (only added ones, not main's), and `uv run pytest tests/core -v` passes them. Yuting goes down #18's table.
- [ ] **A9. Nuisance tests (#19).** *Do:* delete the fork and re-export tests, drop the second tracker from the truth-independence test, and merge the duplicates. *Check:* `git grep -nE 'is not (original\.Track|pipelines\.(ScenarioTracker|TrackingConfig))|DEPRECATED|getattr\(measurements, name\) is owner' -- tests` returns nothing (it finds 5 lines today).

**PR C: the pipeline**

- [ ] **C1. One pipeline, one file format (#20, #22).** *Do:*
  - Merge into `pipelines/tracking.py`, and move each piece to the home in #22.
  - Write `detections.csv`, `tracks.csv`, `metrics.csv` and `metadata.json` as data-001 §6.1, §6.5, §6.6 and §6.9 specify. Use one function per file, and keep scoring separate from writing.
  - Extend main's TOML loader.
  - Plot through `render_range_time_history` (#5).

  *Check:*
  - The three PR modules and `teaching/scopes/track_history.py` are gone.
  - `git grep -nE '_json"|class TrackingWriter|tracking_metrics\.json|_interpretation"' -- src scripts` returns nothing (it finds 11 lines today).
  - Yuting runs the runner on each scenario-001 variant and on `scenarios/scenario_003_tracking.toml`. The header of each CSV lists data-001's columns in §6's order, and `metadata.json` has `"bursts"`, `conventions` and `tracking.state_fields`.
- [ ] **C2. Docs that match the code (#21).** *Do:* resolve each bullet under "Docs that say something the code doesn't do", and add the suggested docstrings and comments. *Check:* Yuting goes down the #21 list line by line against the new code.
- [ ] **C3. Reading view (#17).** *Do:* put guards first, with no guard-only values left outside them, in PR A's code too. *Check:* `uv run python tools/generate_reading_view.py --out /tmp/rv`, then read the pages for `core/tracking/` and `pipelines/tracking.py`. None of the residue listed in #17 is there.
- [ ] **C4. Behaviour, end to end (A4).** *Check:* Yuting reruns S1–S3 with the defaults. The confirmed primary-track frames should be at least the "All three together" row of #8: 118, 13 and 94 of 120.

**PR D: runner and docs**

- [ ] **D1. Runner, specs and notes (#23, #24).** *Do:* extend `scripts/run_scenario.py` rather than adding a runner; `--track` becomes `--no-tracking`; merge the tracker specs; move the port log and personal notes out of `docs/`; replace the README's `## General tracking` section with one Status-list line (#23). *Check:* `scripts/run_general_tracking.py` and `docs/tracking/_michael_notes.md` are gone, `git grep -n 'Duke integration\|Duke runs\|Duke windows'` returns nothing, and `git grep -niw general -- README.md docs/README.md` returns nothing.

**Every PR**

- [ ] **E1. No exemptions (#2).** *Do:* leave `pyproject.toml`'s ruff sections as on main: no `per-file-ignores` for `src/`, no `[tool.ruff.format]` table, no `force-exclude`. Fix the comments and docstrings as in #13 until ruff passes. *Check:* `git diff origin/main... -- pyproject.toml` shows no `[tool.ruff*]` change, and `uv run ruff format --check src && uv run ruff check src` passes.
- [ ] **E2. The gate.** *Do:* fix the nits (#24) in whichever PR touches that code; use Conventional Commits; link the CI run in the PR description; tick this list there. *Check:* `make check` passes on the branch, and the CI run is green. `git log --format=%s origin/main..` shows only Conventional Commit subjects.

Thanks again! Let's start with the licence question, and then Yuting is happy to talk through the split.

---

# Notes for Yuting (not posted)

## Before posting

1. **Check `main` is current.** `git fetch && git rev-list --count main..origin/main` should print 0 (it did on 29 Sep; `origin/main` was 045846d). The PR's parent is 6eb0736.
2. **Approve CI.** GitHub holds workflows on a first-time contributor's PR. Approve the run in the Actions tab, then ask Michael to point to the CI run from now on, not "passes locally". Claude's local run at bedfde0 passed: ruff, mypy `--strict`, conventions, and 1,151 tests with matplotlib installed. Ruff passed only because of the exemptions (#2). Claude checked by running ruff on bedfde0's `src/radar_forge/tracking` with main's `pyproject.toml`: 279 lint errors, and all 14 files would be reformatted.
3. **Check citations.** Claude didn't check section numbers. #9 and #18 name sources without sections on purpose; add them if you have the books. The suggested docstrings in #13 and #21 cite sections. Richards §1.3, §6.5 and §7.3 and Blackman & Popoli §4.3 are copied from main's own modules. Bar-Shalom §6.2 (in #13) is Claude's and unchecked.
4. **Open an issue or draft PR for PR B** (2-D CFAR §13.2 + `wrap_axes` §13.5) so Michael can link to it.
5. **Add `status` and `pair_id` to data-001 §6.5** as a minor-version addition (§8). #20 and #6 now tell Michael you're adding them. Also note that this branch's b7d1a27 renumbers §12's follow-ups (the `pipelines/io` one moves from 2 to 4). The review cites it by name, so it's right either way.
6. **Merge or open R8 first.** #13 and checklist step A7 say R8 will make `make check` enforce the comment rule. It's committed on `feat/conventions-r8-comments` (e3bfa76, not pushed), in the worktree `../radar-forge-r8`. Push it and open a PR before posting, or soften #13's last paragraph to "will add".
7. **Post** only the part above the `---` line, as Request changes:
   `sed '/^---$/q' docs/reviews/pr1-review.md | sed '$d' | gh pr review 1 --request-changes --body-file -`
   The checklist's anchor links are GitHub-style and work in the posted review.

## Re-reviewing the edited PR

Run from a clean worktree so your own checkout stays untouched. Replace `N` with the PR number:

```sh
git fetch origin main pull/N/head:pr-N
git diff --stat origin/main...pr-N            # never against local main
git worktree add ../radar-forge-pr-N pr-N
cd ../radar-forge-pr-N
env -u VIRTUAL_ENV uv sync --extra dev        # or uv runs your main checkout's code
env -u VIRTUAL_ENV make check                 # same gate as CI
env -u VIRTUAL_ENV uv run python tools/generate_reading_view.py --out /tmp/rv-pr-N
git grep -nE '^\s*# (This function|Inputs:|Outputs:)|^\s*#{10,}' -- src   # expect nothing
git grep -nE '\b(legs?|leg_[a-z_]+|[a-z_]+_legs?)\b|leg\{|rd_leg|"leg' -- src scripts tests ':!src/radar_forge/core/radar.py'   # expect nothing
```

Then rerun S1–S3 and compare them with #8's table. Go down the re-review checklist in order. Each item's *Check* is the command or file to look at. When you're done, `git worktree remove ../radar-forge-pr-N`.

## What was verified, and how

- **This round:**
  - **The tentative-track bug (#10)** was run, not reasoned. The scratch test in #10 fails at bedfde0 in the `../radar-forge-pr-1` worktree.
  - **The Stone Soup and FilterPy mapping (#16)** was checked at GitHub `main` / `master`: `stonesoup/sensor/sensor.py` (`Sensor.measure()` generates detections), `tracker/simple.py` (`MultiTargetTracker`'s initiator, deleter, detector, data associator and updater), `predictor/kalman.py` (`UnscentedKalmanPredictor`), `deleter/time.py` (`UpdateTimeStepsDeleter`, `UpdateTimeDeleter`), `types/detection.py` (`Detection.measurement_model`), and FilterPy's `kalman/UKF.py` (stateful `UnscentedKalmanFilter`, with `x` and `P`).
  - **Counts** were re-grepped at bedfde0 and on `origin/main`: 198 `####` and 204 `Inputs:`/`Outputs:` lines (0 on main); 7 variant comparisons; 17 `vector` call sites; the column lists in #20 against data-001 §6 on `origin/main`.
  - **The new checklist commands** (7b, 9) find 7, 1 and 11 lines at bedfde0 and 0 on main.
  - **Not verified:** "12 files, 5 Protocol hops" and "about a third is code" in #16 come from a subagent's trace and an AST count. They're approximate, and #16 words them that way.

- **Scenario numbers.** The #8 table comes from running bedfde0 in a separate worktree. The patches were scratch scripts, not in the repo:
  - tentative-track costs +1000 ("confirmed pick first");
  - births suppressed inside a confirmed gate;
  - `--tracker-config` with q = 16.
- **Cases actually run.** The IMM(CV, CT) failure (#11) and the NIS-vs-NLL case (#9) were run, not only reasoned.
- **Claims about other projects.** These were checked against each project's GitHub `main`:
  - Stone Soup: `MultiMeasurementInitiator`, `missed_distance = inf`, and the UKF defaults `alpha = 0.5`, `beta = 2`, `kappa = 3 − n`;
  - FilterPy: the fixed IMM transition matrix.
- **Counts and references.** Every count, file-format claim and reuse target in #5, #6 and #20 was checked with `git show`/`git grep` at bedfde0: leg lines, banner lines, the `_numerics` importers, the `run_scenario` diff (128 of 408 lines differ), the CSV headers, and the production callers of the deferred classes.
- **Tests (#18, #19).**
  - Claude read all ten PR test files (42 tests) at bedfde0 and mapped each one onto your checklist and onto main's `tests/core/test_tracking.py`.
  - In a detached worktree at bedfde0 with its own venv, all 82 test cases pass in 9.4 s. `--durations` gives the timings #19 quotes.
  - The truth-independence claim was checked directly: the altered frame's `iq` arrays are the same objects as the original's.
  - Both suggested tests in #18 were run against bedfde0 and pass: NEES/NIS in about 36 s; clutter on five seeds, about 1 s each. The several-detections-per-bin case was run on five seeds and fails on all of them.
  - The #19 *Check* grep finds 5 lines at bedfde0 and 0 on `origin/main`.
  - The tolerance-comment count ("about a quarter of 29") counts a comment up to three lines above the call, so treat it as approximate.
  - Unchecked: the Bar-Shalom section for filter consistency, which the NEES docstring cites without a number.
- **Environment gotcha.** Your shell exports `VIRTUAL_ENV` for the main checkout, so `uv run pytest` in a worktree silently runs main's pytest against main's code. Use `env -u VIRTUAL_ENV uv sync --extra dev` and then `env -u VIRTUAL_ENV uv run python -m pytest`. Don't use `--all-extras`: torch has no wheel for macOS 13.
- **Claude's own proposals, not spec wording.** Two things are Claude's suggestions:
  - The `core/tracking/` layout in #3. D2 itself names only `{filters,association,fusion}.py`, so change the layout if you prefer D2's wording.
  - "One piece at a time, each with a scenario" (#6). It comes from A.13a and the repo's habit of pairing each `core/` module with a scenario.
- **What D2 says** (`spec/structure.md` :611–623): promote `core/tracking.py` to a subpackage re-exported from `__init__`, "do not pre-split". The trigger is "a second association strategy … or a track-fusion layer". This PR adds both (`NearestNeighbour`, multi-sensor), so the trigger has arguably fired, and D2's answer is to promote, not to fork.
- **Aside, on main (not for the review).** `core/tracking.py:21` and `:462` cite "spec/structure.md D5" for the closing-positive sign, but D5 is COCO label-schema parity (:637). The convention is in the table row at :659. It's a one-line fix on main.

## Your questions, answered

**This round**

- **"I see duplicate code in `pipelines/`. Am I wrong?"** You're right. #22 now opens with "Repetition inside the three modules":
  - the variant label is tested seven times;
  - range is wrapped three ways;
  - each CSV has a header and a row list matched by position, and `snapshot_record` writes the same snapshot a third way;
  - the prior is built twice;
  - `_pair` redoes `dual_prf_measurements`.

  All of it was re-grepped at bedfde0.
- **"Is `write` like serialise? What do people call it? Does it belong in the pipeline?"** "Serialise" is the general word. The stdlib says `json.dump`/`dumps`, and pandas says `to_csv`/`to_dict`. Main's `write_<file>` convention is fine, so keep `write`. The real problem is that `TrackingWriter.write` also *scores* against truth. #20 asks for one writer per file plus a separate scoring function. On placement: not in `pipelines/tracking_output.py`. For now the writers go next to main's in `scripts/run_scenario.py`, and later into data-001's `pipelines/io/`.
- **"Should there be a general function for CSV, and the same for metrics and metadata?"** Yes, and data-001 already decides it: one registry plus stdlib CSV/JSON writers in `pipelines/io/` (§2.6, §9, §12's `pipelines/io` follow-up). It doesn't exist yet. As you chose, #20 asks Michael to write data-001's columns now, with the columns kept in tuples so the move is mechanical, but not to build the registry. #20 lists every mismatch in §6.1, §6.5, §6.6 and §6.9. **data-001 landed after his PR** (29 Sep vs 25 Sep), and #20 says so.
- **"Are the names in scopes good?"** Half. `render_track_history` follows `render_<what it draws>`. But the module shares the function's stem, where main names modules by scope. It reads a CSV path rather than arrays, re-implements `require_pyplot`, and plots the first track rather than the longest confirmed one. See #5, "Scope names".
- **"Do we need everything in `_numerics`?"** No. #6 now has a table: keep `vector`, `covariance` and `cholesky` (renamed); drop `FloatArray`; inline `timestamp`.
- **"Is there a comment convention? Can it follow the existing code?"** Yes: `style.md` §10, "why, never what". But until now only a reviewer checked it. #13 now describes main's pattern with examples, and adds the PR's inline "what" comments as the second problem. **R8** (on its own branch, see Before posting) makes `make check` reject banners and `Inputs:`/`Outputs:` blocks. On main it finds 0; on bedfde0 it finds 402 (198 + 204). It also switches on ruff's `TD` rules for the `TODO(author): … <link>` format. The R8 agent left out ruff `FIX002` on purpose: it rejects *every* TODO, including the documented format. Add it only if you'd rather ban TODOs outright. Whether a comment says *why* stays `[review]`, as does the loop-why rule: main has 23 loops without one.
- **"Tracking is hard to read. Is it state of the art? Clear, accurate, concise? Same objects as other people?"** See #16:
  - **Clear:** not yet. A scan goes through 12 files, 5 Protocol hops and 5 string registries.
  - **Accurate:** four docstrings contradict the code, one of them an example that raises `NameError`.
  - **Concise:** about a third of it is code.
  - **The objects:** they mostly match Stone Soup and FilterPy. The mapping was checked at their GitHub source. The clash is `Sensor`, which in Stone Soup generates detections.
  - **State of the art:** not needed for a teaching repo. It lacks vectorised sigma points, the Joseph form and early tentative-track deletion.
- **"How good is the coding standard?"** See #10. The maths is right. **One real bug, run and reproduced:** a tentative track that can't reach 3-of-5 is never deleted (hit, miss, miss, … is still tentative after 60 scans). There's no Joseph form, and there are a few fragile spots: float `==` on times, a public mutable `sensors` dict, `cost: str`, a positional `True`. The engineering is heavier than one radar needs, and that, more than the maths, is what makes it hard to read.

**Earlier rounds**

- **Do the `pyproject.toml` exemptions match the repo's existing code?** No. On main, `per-file-ignores` has only `tests/**` and `scripts/**`: whole directories, each with a reason. There's no `[tool.ruff.format]` table, no `force-exclude`, and no `# noqa` anywhere in `src/`. The one exemption mechanism is per line and states its reason (`# broadcast-exempt:`). The PR adds 14 file-level ignores in `src/`, a format exclude for the same 14 files, and `force-exclude`, which makes the pre-commit hook skip them. #2 is now **blocking**, with "fix the comments properly" as the fix. You can drop it back to should-fix if you'd rather have the licence as the only blocker.
- **Does "general" for tracking match the repo?** No. On main the word appears only in ordinary sentences ("the general form", "in general"), never as a name, and main's README has no per-feature sections. The PR uses it in `README.md:214` (`## General tracking`), `:248` ("general runner") and `docs/README.md:38` ("general tracker"), as well as in the code names #5 already listed. #5 and #23 now cover the prose, and #23 has a replacement Status line.
- **Do we need `tracking/_numerics.py`?** A shared validator, yes. Ten modules use it, `core/` has no covariance or Cholesky checks, and style §5 asks for a shared helper. But not in this form: about 43 of its 185 lines are code, the names are vague, and `pipelines/` imports it across a package boundary. See the end of #6.
- **Do we need `pipelines/general_tracking.py` / `RadarDetection`?** No new module, and no new type.
  - **What `pipelines/` does on main:** it's the layer that turns the scenario TOML and RD map into SI detections, then drives `core.tracking` (`structure.md` :78–82). The `pipelines/tracking.py` docstring says it "owns everything between a RangeDopplerProduct and a call to TrackManager".
  - **Does main put dataclasses there?** Yes. `Measurement`, `DetectionConfig`, `TrackingConfig`, `FrameTracks` and `ScenarioTracker` all live there, while algorithm-level types (`Detection`, `Track`, `KalmanState`) live in `core/`. So the PR's *layer* is right.
  - **The module and name are wrong:** "General" means only "not the other one", and `detect_product` and `_pair` duplicate `frame_detections` and `dual_prf_measurements`. Keep its better detection/measurement split through the merge (What's good).
- **"leg":** it was covered in the old draft. It's now in the #5 table, with a better count: 70 lines at bedfde0, all added by the PR. #20 covers the file-format side: the `"legs"` key, `leg{i}` npz keys and `rd_leg{i}` PNGs.
- **Repo convention vs standard tracking names:** keep standard names for algorithms (the "Keep" list in #5). Use repo convention for modules, fields, configs and file formats. The one judgement call is `build_tracker_enu` vs `build_enu_tracker`. Family-word-first is a repo-consistency argument, not a universal rule; the stronger point is that "enu" names the wrong thing.
- **Does `ScenarioTracker` follow repo conventions?** No. It differs from main's class in form, attribute names, call signature, error style, docstring sections and references (the #21 table). It also branches on scenario labels. It's fixable without a redesign.
- **Does it need more in-code docs?** Yes, and the text is written out in #21, ready to paste. The more important half is the docs that are *wrong* (#21's list). Two first-draft findings were softened after checking: "interpolation-derived truth" is true but in the wrong place, and "deleted tracks remain visible" is vague rather than false.
- **Too much pipeline code for tracking?** Yes: 969 lines that redo main's `pipelines/tracking.py`, almost all without an outside caller (#22).
- **Does some of it belong elsewhere?** Most of it. `build_tracker` and the filter half of `TrackingConfig` go to `core/tracking/`; the loader, writer and detection go into main's existing functions (#22 table).
- **Will the reading view pick up the tracking modules?** Yes, automatically, since it walks all of `src/radar_forge`. It will show the guard-residue problem (#17), which is the thing to fix before merge.
- **`tracking_config.py` L324+:** #13 has the worked example and a full replacement docstring. The same applies to the file's other six banners.
- **What tests does tracking code need, and does the PR have them?** Your pasted list is the standard one: maths, kinematics, association, and consistency. #18 maps it onto this PR, and the short answer is "the maths half, not the other half":
  - **Covered well:** coordinate geometry; the UKF against the KF; CV/CA transitions and Q; lifecycle boundaries; crossing targets (weakly, noise-free).
  - **Missing:** NEES/NIS; clutter; P_d < 1; the gate's acceptance rate; covariance health over many steps; tentative-track deletion; Jacobian vs finite difference; manoeuvres/IMM.
  - **Main already has almost all the missing ones** in `tests/core/test_tracking.py`. So the review asks Michael to port them, which is a much smaller ask than "read Bar-Shalom".
  - **Not applicable yet** (in your list, but no scenario needs them): Doppler blind zones, GOSPA/OSPA, JPDA/MHT, and track purity/fragmentation rates.
- **Nuisance tests?** Yes, #19. Two lock in the fork, so they fail when Michael does what #3 and #5 ask. One comparison can't fail. One test locks in the TOML schema split. And there are three duplicate pairs. Before calling anything slow, Claude timed it: nothing in the PR is over a second except the plot test (3.4 s, most likely the first matplotlib import), and the two tests marked `slow` aren't.
- **Bloat:** about 2,000–2,500 of 7.6k lines. Roughly 1,000 are banners (#13), about 1,150 have no production caller (#6 defer table), and about 280 are copied from `run_scenario.py`. These overlap: deferring `imm.py` also removes its banners.

## Corrections to the earlier draft

- **#22's "82 of banner" wasn't banner.** `tracking_config.py` has no `####` lines. Its 82 lines are `Inputs:`/`Outputs:` comment blocks. #22 is reworded. The 198 `####` lines are all in `tracking/`.
- **#20 was written against main's writers.** It now targets data-001, which landed after the PR. Say so if Michael asks why the target moved.
- **The "General" check missed the README.** Checklist step A2 (then "item 3") used `\bGeneral[A-Z]`, which finds 0 of the three README/docs lines. The new regex finds 31 lines at bedfde0 and 0 on `origin/main`.
- **#2 missed `force-exclude = true`.** That setting is what switches off the pre-commit hook's format check for the excluded files.
- **The `EARTH_RADIUS_M` change is not in the PR.** It's your commit 6eb0736, the PR's parent. A diff against your stale local `main` shows it, so don't raise it with Michael. Its subject line "Update constants.py" isn't a Conventional Commit, because a GitHub web edit skips the hooks. The value is correct, and it's worth knowing when you ask Michael for commit discipline.
- **"The PR only adds" was too strong.** It also adds a top-level `radar_forge.tracking` export and a second `core.DetectionConfig` (#24).
- **`teaching/scopes/track_plot.py` already exists on main**, with `render_range_time_history`. So the ask is "extend it", not "use that file name".
- **NIS and cardinality-first GNN aren't the cause.** The first pass guessed they caused the fragmentation, and running it showed they don't. The causes are #7 and #8.
- **The old #18 said a clutter test "would have caught #7 and #8".** On its own it wouldn't. A seeded clutter + P_d = 0.7 test passes on bedfde0. #7 shows up only when one target gives several detections per scan, and that case fails on every seed (#18).
- **A first sketch of the clutter test failed for its own reason.** It drew noise with σ = 1 m while declaring R = 0.1 m². The true detection fell outside the too-tight gate and started a second track, which was confirmed next to the first. The corrected sketch passes. Worth knowing: when a confirmed track's gate misses its target, this tracker starts a duplicate rather than re-acquiring. Main has a re-acquisition path (`test_a_widened_track_re_acquires_its_target_and_keeps_its_id`, `tests/core/test_tracking.py:496`). It's now raised with Michael as a discussion point at the end of #18, at your request.
- **"Duke" *is* a repo term.** The earlier #23 said the repo doesn't use it. It does: the Duke Receiver is the site in scenarios 001–003. The point is that it names a site, not scenario 001's variants. #23 is reworded.
- **This round's findings, rechecked.** Four first-draft claims were corrected before they went into #13 and #21: the "uniformly increasing" message is misleading, not wrong (the uniformity check is on the next lines); `parse_tracking_table`'s docstring *does* mention `DetectionConfig`; only one of the two range wraps duplicates `StateSpace`; and "Other 6 banners, 82 lines" was really seven banners, 82 lines in all.
- **The accuracy pass (29 Sep).** Three read-only agents re-checked every claim against `origin/main`, bedfde0 and data-001. The corrections:
  - **#2's parts didn't sum to 279.** The docstring-layout count is 37, not 43 (D202 13, D204 13, D205 7, D210 3, D400 1). The total of 279 was right.
  - **Noise vs peak power.** #5 renamed `.noise_power_linear` to `.noise_power_w`, but #20 said to write `peak_power_w`. `detections.csv` has no noise column: noise goes per burst in `metadata.json`. Both now say so.
  - **The join key.** `frame` (with `time_s`) joins across files; `(frame, detection_id)` is only the detection key (§5).
  - **DF3.** It doesn't reject JSON cells "by name". It keeps the covariance tabular.
  - **`status`/`pair_id`** aren't in §6.5. You chose to add them (Before posting, step 5).
  - **"leg was retired on purpose"** had no source on main. It now cites main's `bursts` and `core/radar.py:40–42`.
  - **"Family word first" and `sigma_`** are main's habits, not style.md rules. #5 now says so.
  - **"The only divider"** was wrong: `core/constants.py` has three more.
  - **Line numbers.** `generate()` is at :348, not :372; `_enu_model` is at :495; `cost: str` is at `engine.py:71`; #21's variant-label lines now match #22's.
  - **Checklist step A8's check** was `pytest -k "nees or nis …"`, which selects many of main's own tests. It's now a diff of added `def test_` lines.
  - **The behaviour check (S1–S3 rerun)** can't run on PR A, so it moved to C4.
  - **#10's reproducer** asserted on the last frame only, and would pass after a fix only because 60 happens to end after a deletion. It now tracks the longest tentative run, and still fails at bedfde0 (60 scans). The PR's window slides (`deque(maxlen=5)`), while main counts the first N scans.
  - **#7's four detections** were confirmed by running S1: range bin 240 at four Doppler bins, at t = 15 s.
  - **Order.** The sections were regrouped by split PR and renumbered. The old → new map is 1→1, 12→2, 2→3, 3→4, 4→5, 5→6, 7→7, 8→8, 9→9, 24→10, 10→11, 16→12, 13→13, 14→14, 15→15, 23→16, 21→17, 11→18, 22→19, 6→20, 19→21, 20→22, 17→23, 18→24. Every `#N` in this file uses the new numbers.

## Habits worth building

**As reviewer:**
- Rerun the checks yourself.
- Read the spec the PR touches before reading the code.
- Settle licence before code quality.
- Label every comment.
- Ask rather than decree.
- Diff against `origin/main`, not your local `main`: that's how the Earth-radius false alarm happened.
- Always say what's good.

**As author (for Michael):**
- Keep PRs small.
- Say what you checked and what you're unsure of.
- Be able to explain every line, including text a coding agent wrote.
- Never loosen a check to make code pass.
- Extend before you fork.

## After this review

Update [`docs/conventions/review.md`](../conventions/review.md) with anything new this round taught: checks, mistakes, workflow. It's what Claude Code loads as `/pr-review`. It runs Claude Code's built-in `/code-review` and `/security-review` alongside, and adds what they can't know: the spec, the conventions and the teaching voice (§10).
