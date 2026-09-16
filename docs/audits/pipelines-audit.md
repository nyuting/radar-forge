# Pipelines audit — refactor-002 §1.2 and §1.3

Scope: `src/radar_forge/pipelines/`, `src/radar_forge/teaching/` and
`scripts/run_scenario.py`. `src/radar_forge/core/` belongs to a parallel stream and is not
touched here; neither is `docs/conventions/style.md` or `scripts/check_conventions.py`.

Read `spec/refactor-002-spec-first-audit.md` §4.1 first. `spec/` carries no prototype
implementation — one 8-line dataclass sketch and no other `def`, `import`, `return` or `np.`
in any fenced block — so there is no rival code to diff against. "Reference" here means the
**specified behaviour**: the build sequences and acceptance matrices of
`spec/scenario-001-xband.md` §5–6, `spec/scenario-002-bistatic.md` §5–6 and
`spec/scenario-003-tracking.md` §5, §12 and §14; the documented contracts — shapes, frames,
sign conventions, units; and `spec/structure.md`'s decisions D1–D8.

Each public function is judged on both halves: does it compute what the spec says, and is it
the better implementation available. An honest "no change, here is why" is a result.

## Findings index

| # | Finding | Q | Side that was wrong | Status |
| :--- | :--- | :--- | :--- | :--- |
| [F1](#f1-two-form-feed-characters-ate-the-bisector-velocity-equation) | Two form-feed characters ate the bisector-velocity equation | Q5 | code | fixed |
| [F2](#f2-four-citations-pointed-at-a-section-that-does-not-exist) | Four citations pointed at a section that does not exist | Q5 | code | fixed |
| [F3](#f3-the-sigma_accel_mps2-margin-no-longer-exists) | The `sigma_accel_mps2` margin no longer exists | Q1 | both | re-measured, recorded |
| [F4](#f4-the-pulsed-range-axis-is-re-derived-and-the-docstring-denied-it) | The pulsed range axis is re-derived, and the docstring denied it | Q5 | code | fixed |
| [F5](#f5-the-track-scope-was-never-a-plan-view) | The track scope was never a plan view | Q6 | spec | renamed both sides |
| [F6](#f6-three-more-names-that-did-not-say-what-they-meant) | Three more names that did not say what they meant | Q6 | code | renamed |
| [F7](#f7-replace_measurement-restated-nine-fields-to-change-two) | `replace_measurement` restated nine fields to change two | Q4 | code | fixed |
| [F8](#f8-a-fold-that-was-the-identity-function) | A fold that was the identity function | Q4 | code | removed |
| [F9](#f9-two-contracts-a-caller-could-not-see) | Two contracts a caller could not see | Q7, Q8 | code | fixed |
| [F10](#f10-what-was-measured-and-not-changed) | What was measured and not changed | Q3 | — | no change |
| [F11](#f11-left-alone-deliberately) | Left alone deliberately | — | — | recommended only |

## Verdicts

Legend: **ok** — no defect found. **fixed** — a defect found and corrected in this audit.
**note** — a judgement recorded below, no change. Every public symbol in scope has a row.

### `pipelines/scenarios.py`

| Symbol | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Scenario` | ok | ok | ok | ok | fixed | fixed | ok | ok |
| `Scenario.is_bistatic` | ok | ok | ok | ok | ok | ok | ok | ok |
| `Scenario.n_frames` | ok | ok | ok | ok | ok | ok | ok | note |
| `Scenario.frame_times_s` | ok | ok | ok | ok | ok | ok | ok | ok |
| `Frame` | ok | ok | ok | ok | ok | ok | ok | ok |
| `load_scenario` | ok | ok | ok | ok | fixed | ok | ok | ok |
| `iterate_frames` | ok | ok | note | ok | fixed | fixed | ok | ok |
| `RangeDopplerProduct` | ok | ok | ok | ok | ok | note | ok | ok |
| `form_range_doppler_map` | ok | ok | note | ok | fixed | fixed | fixed | ok |
| `peak_range_velocity` | ok | ok | ok | ok | ok | ok | ok | note |

`Scenario.n_frames` rounds `duration_s * frame_rate_hz`, so a window of 120.4 s at 1 Hz
yields 120 frames and the last 0.4 s is dropped silently. That is the right behaviour — a
partial frame is not a frame — but the docstring says only "Number of frames the window
produces". Recorded, not changed: every shipped scenario has an integral product, and adding
a raise would reject a window a reader may legitimately want to round.

`peak_range_velocity` returns a bare `tuple[float, float]`. The order is documented and both
call sites unpack it immediately, so a `NamedTuple` would buy attribute access at the cost of
a type change in a return contract nothing has misused. Recorded under
[F11](#f11-left-alone-deliberately).

### `pipelines/trajectories.py`

| Symbol | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Trajectory` | ok | ok | ok | ok | ok | ok | ok | ok |
| `Trajectory.n_fixes` | ok | ok | ok | ok | ok | ok | ok | ok |
| `Trajectory.duration_s` | ok | ok | ok | ok | ok | ok | ok | ok |
| `TargetTrack` | ok | ok | ok | ok | ok | ok | ok | ok |
| `TargetTrack.n_frames` | ok | ok | ok | ok | ok | ok | ok | ok |
| `BistaticTargetTrack` | ok | ok | ok | ok | ok | ok | ok | ok |
| `BistaticTargetTrack.n_frames` | ok | ok | ok | ok | ok | ok | ok | ok |
| `BistaticTargetTrack.range_m` | ok | ok | ok | ok | ok | ok | ok | ok |
| `load_flight_csv` | ok | ok | ok | ok | ok | ok | ok | ok |
| `resample` | ok | ok | ok | ok | ok | ok | ok | ok |
| `to_radar_frame` | ok | ok | ok | ok | ok | ok | ok | ok |
| `to_bistatic_radar_frame` | ok | ok | ok | ok | fixed | ok | ok | ok |

This module came out of the audit best. The sign convention is D5's throughout and stated at
every boundary; `BistaticTargetTrack`'s deliberate *absence* of a `radial_velocity_mps` field,
with a Notes paragraph saying why, is the single best piece of Q8 design in scope — it makes
the wrong quantity unavailable rather than merely discouraged. The one defect was
typographic, and it destroyed the module's only equation: [F1](#f1-two-form-feed-characters-ate-the-bisector-velocity-equation).

The velocity in both transforms is a central difference of an *interpolated* range, so it is a
mean rate over the several seconds between real fixes, not an instantaneous Doppler. The
module says so, in the module docstring and again in each function. That is the right
disclosure and it is why nothing here needed a numerical change.

### `pipelines/tracking.py`

| Symbol | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Measurement` | ok | ok | ok | ok | ok | ok | ok | ok |
| `DetectionConfig` | ok | ok | ok | ok | ok | ok | ok | ok |
| `TrackingConfig` | fixed | ok | ok | ok | fixed | ok | ok | ok |
| `UnfoldingMode`, `UNFOLDING_MODES` | ok | ok | ok | ok | ok | ok | ok | ok |
| `frame_detections` | ok | ok | note | ok | ok | ok | note | ok |
| `configs_from_scenario` | ok | ok | ok | ok | ok | ok | ok | ok |
| `dual_prf_measurements` | ok | ok | note | ok | ok | ok | ok | ok |
| `unfold_velocity_mps` | ok | ok | ok | ok | ok | ok | ok | ok |
| `slope_velocity_mps` | ok | ok | note | ok | ok | ok | ok | ok |
| `range_slope_sigma_mps` | ok | ok | ok | ok | ok | ok | ok | ok |
| `minimum_unfold_history_frames` | ok | ok | ok | ok | ok | ok | ok | ok |
| `FrameTracks` | ok | ok | ok | ok | ok | ok | ok | ok |
| `ScenarioTracker` | ok | ok | ok | ok | fixed | ok | ok | fixed |
| `ScenarioTracker.is_unfoldable` | ok | ok | ok | ok | ok | ok | ok | ok |
| `ScenarioTracker.unfold_frame_of` | ok | ok | ok | ok | ok | ok | ok | ok |
| `ScenarioTracker.step` | ok | ok | ok | ok | ok | ok | ok | ok |
| `ScenarioTracker.step_unfolded` | ok | ok | ok | ok | ok | ok | ok | ok |
| `replace_measurement` | ok | ok | ok | fixed | fixed | ok | ok | ok |

This is the file the brief expected the most bloat in, at ~1030 lines. It is not bloated. The
length is prose: the module docstring, the fold-consistency comment inside `_unfold_all` and
the `TrackingConfig` docstring together run to about 90 lines and every one of them records a
measurement or a failure mode that a reader would otherwise rediscover the hard way. That is
the brief — "plain, documented, and cites its source" — not padding. The one genuinely
redundant block was `replace_measurement`'s body, [F7](#f7-replace_measurement-restated-nine-fields-to-change-two).

Conformance against `spec/scenario-003-tracking.md` is close. §14 already documents the seven
known departures, and each is implemented as §14 describes it: the Doppler roll before
clustering (§14.2), the per-range-cell sidelobe merge (§14.3), the batch range-slope bootstrap
rather than the filter covariance (§14.4), the range-only seeding of a new track (§14.5).
`range_slope_sigma_mps` reproduces §5.3's quoted 6.84 m/s at N=5 and 3.34 m/s at N=8, and
`minimum_unfold_history_frames` returns §5.3's ten frames, both pinned by doctests. The one
figure that did not survive re-measurement is [F3](#f3-the-sigma_accel_mps2-margin-no-longer-exists).

D2 — tracking stays in `core/` until a second association strategy or a fusion layer appears —
is not at risk from this module. `pipelines/tracking.py` is the *pipeline* half and holds no
filter mathematics; it owns detection-to-measurement conversion, unfolding and the frame loop,
and calls into `core.tracking` for everything else. Nothing here moves the promotion trigger.

### `teaching/`

| Symbol | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `plotting.require_pyplot` | ok | ok | ok | ok | ok | ok | ok | ok |
| `plotting.magnitude_db` | ok | ok | ok | ok | ok | ok | ok | ok |
| `plotting.save_figure` | ok | ok | ok | ok | ok | ok | fixed | ok |
| `scopes.rd_map.render_range_doppler` | ok | ok | ok | ok | fixed | ok | ok | ok |
| `scopes.track_plot.render_range_time_history` | ok | ok | ok | ok | fixed | fixed | ok | ok |

Both scopes return the figure and document that the caller owns it and must close it, with
`save_figure` named as the thing that does both — a correct and unusually explicit Q8
contract for a plotting function, and the reason a 16,500-frame run does not exhaust memory.

`magnitude_db` is a 20·log₁₀ amplitude ratio and says so twice, including the consequence of
handing it a power array. The floor is argued for rather than asserted. No change.

`render_range_doppler` and `render_range_time_history` both return `Any`. That is forced:
naming `matplotlib.figure.Figure` would require the import the package exists to defer, and
the docstring carries the real type. Recorded, not changed.

### `scripts/run_scenario.py`

Outside `src/`, so rule R3 does not reach it, and it is not under test. Judged anyway.

| Symbol | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `parse_args` | ok | ok | ok | ok | fixed | ok | ok | ok |
| `git_commit` | ok | ok | ok | ok | ok | ok | ok | ok |
| `burst_metadata` | ok | ok | ok | ok | ok | ok | ok | ok |
| `site_metadata` | ok | ok | ok | ok | ok | ok | ok | ok |
| `write_metadata` | ok | ok | ok | ok | fixed | ok | ok | ok |
| `render_frame` | ok | ok | ok | fixed | ok | ok | note | ok |
| `build_tracker` | ok | ok | ok | ok | ok | ok | ok | note |
| `track_frame` | ok | ok | ok | ok | ok | ok | ok | ok |
| `write_tracking_csvs` | ok | ok | ok | ok | ok | ok | ok | ok |
| `render_track_frame` | ok | ok | note | ok | fixed | ok | ok | ok |
| `clear_previous_frames` | ok | ok | ok | ok | ok | ok | ok | ok |
| `assemble_movie` | ok | ok | ok | ok | ok | ok | ok | ok |
| `main` | ok | ok | ok | ok | ok | ok | ok | ok |

The script is thin, as `spec/scenario-001-xband.md` §5 item 9 requires: no physics, no
processing decision and no geometry lives here. `clear_previous_frames` and `assemble_movie`'s
five-digit glob both exist to stop a short re-run splicing a previous run's tail onto the new
one, which is the kind of failure that produces a movie nobody questions.

`build_tracker` returns `tuple[ScenarioTracker, bool]`, and the bare `bool` is the weakest
return contract in scope — the caller has to read the docstring to learn it means
"simulated_angles". One call site, and the alternative is a second dataclass for one flag.
Recorded under [F11](#f11-left-alone-deliberately).

`render_frame` takes six positional parameters, of which `dynamic_range_db` and `record` want
to be keyword-only. One call site; recorded, not changed.

## Findings in detail

### F1 — Two form-feed characters ate the bisector-velocity equation

`to_bistatic_radar_frame`'s Notes section rendered as

```
.. math:: v_b = -rac{1}{2}rac{\mathrm{d}(R_t + R_r)}{\mathrm{d}t},
```

The file held two literal `0x0C` form feeds. The docstring was once a plain string, Python
consumed each `\f` as an escape, and promoting it to `r"""` later froze the damage in. `cat`,
`grep` and `sed` all render a form feed as nothing, which is why three of my own attempts to
patch the line matched nothing at all.

The module's only equation was therefore unreadable, in the one function whose factor of
one-half is the entire reason the bistatic and monostatic Doppler expressions agree. Fixed to
`\frac`. A scan of every `.py` file in `src/`, `scripts/` and `tests/` finds no other control
character, so this was a single event rather than a pattern.

### F2 — Four citations pointed at a section that does not exist

`spec/scenario-001-xband.md` has six sections. `pipelines/scenarios.py` and
`teaching/scopes/rd_map.py` both cited an "S7.1" of it for the `dsp` axis seam; that material
is §4.3. `scenarios.py` also cited §5 — the build sequence — for the frame/CPI structure, which
is §4.2, and §4–§5 for the three variants, which is §4.1.

CLAUDE.md makes citation a non-negotiable, and a citation that resolves to nothing fails it as
completely as no citation. All four now name real sections. The `rd_map.py` reference gained
§6 as well, since the acceptance criteria are what its two markers exist to make visible.

### F3 — The `sigma_accel_mps2` margin no longer exists

Three places recorded the evidence for `sigma_accel_mps2 = 5.0` against §6.2's judgement value
of 2.0, and no two agreed. `spec/scenario-003-tracking.md` §14.6 and
`scenarios/scenario_003_tracking.toml` said 98% of frames tracked against 93% at 2.0;
`TrackingConfig`'s docstring said 95% against 93%.

Re-measured over the current default 120-frame window, seeded, and reproduced identically on
repeat runs:

| `sigma_accel_mps2` | frames holding a confirmed track | confirmed track ids | fold selected correctly |
| :--- | :--- | :--- | :--- |
| 5.0 | 117 / 120 | 2 | 82 / 91 (90%) |
| 2.0 | 117 / 120 | 2 | 90 / 96 (94%) |

Both sides were wrong, in different ways. The code had drifted to a third figure, 95%, which
matches nothing. The specification's margin was real when measured but has since been overtaken
by two of its own later decisions: §14.10 moved the default window to 663 s, and §14.7 added
re-acquisition, which holds a confirmed track through a mis-unfold instead of deleting it.
Re-acquisition is precisely the mechanism that makes track retention insensitive to the process
noise here, and §14.7 already credits it with taking the run from six track ids to two.

The default stays at 5.0. Nothing measured argues for moving it, and the one figure that now
separates the two — fold selection, where 2.0 is four points better — turns on a handful of
frames either side of a single fold crossing and needs a wider window before it can decide
anything. What changed is that all three places now say the same thing, and say that the
number that once justified the setting no longer does.

The drift happened because nothing checked it. The obvious guard is a `slow`-marked test
pinning the retention figure, and it is recommended rather than written: it would pin a
measurement of a window, not an invariant of the tracker, and `docs/conventions/testing.md`
prefers analytic ground truth to recorded output.

### F4 — The pulsed range axis is re-derived, and the docstring denied it

`form_range_doppler_map` claimed both axes come "from the `dsp` helpers rather than
re-derived". The FMCW branch does. The pulsed branch computes `n·c / 2f_s` locally, and has
to: `range_bin_centers_m` maps a *beat frequency* to range and says nothing about a compressed
pulse's delay, so there is no helper to call.

The code is right — one fast-time sample is one round-trip delay step, and `c/2` also yields
the bistatic mean range under D6, so the pulsed path is siting-agnostic exactly as D7 requires.
The docstring was wrong, and wrong in the direction that discourages anyone from noticing the
gap. It now says what the pulsed branch computes, why the factor is right in both sitings, and
that a `delay_bin_centers_m` helper in `core/dsp.py` is what would close the seam — that
module's workstream, not this one.

### F5 — The track scope was never a plan view

`render_track_plan_view` did not draw a plan view. A plan view is the ground plane; this draws
range against time. The module docstring spent a paragraph explaining that the name was not
what the function did, and the reason it gave is correct and decisive: §4's radar measures
range and range rate and nothing else, so an east-north plot would have to invent a bearing for
every point it drew.

Here the specification was the party that was wrong. `spec/scenario-003-tracking.md` §9 and
`spec/structure.md` both call it a plan view and the implementation inherited the term.
Renamed to `render_range_time_history`, with every call site, test, docstring and specification
reference moved in the same commit, and recorded in §14.8 per R1.3.4.

### F6 — Three more names that did not say what they meant

**`burst_range_doppler` → `form_range_doppler_map`.** PEP 8 wants a verb for a function, and
"form a range-Doppler map" is what the radar literature calls the step. The old name read as a
noun — a burst's range-Doppler *something* — which is the return value's job to say.

**Its `cube` parameter → `baseband`.** The array is 2-D `(n_pulses, n_samples)`; a cube is
conventionally the 3-D range × Doppler × channel datacube. It was the only parameter in the
library named that way: `core/dsp.py` calls the same array `samples` and `profiles`, and
`docs/conventions/style.md` §4's own shape-documentation exemplar names it `baseband`. Never
passed by keyword, so nothing outside the definition moved. "Cube" stays in the prose
everywhere — it is the project's and the specification's word for the array, and the objection
was only to it as an identifier.

**`Scenario.detection` / `.tracking` → `.detection_table` / `.tracking_table`.** They hold raw
`tomllib` output, while `DetectionConfig` and `TrackingConfig` hold the parsed settings — and
`ScenarioTracker.detection` and `.tracking` *are* those. Two attributes one dot apart meant two
different things. `load_scenario`'s own locals were already `detection_table` and
`tracking_table`, so the module now agrees with itself.

`iterate_frames`' locals `range_m_all` and `velocity_mps_all` became `track_range_m` and
`track_velocity_mps`, with the azimuth and elevation pair to match. `_all` trailing the unit
reads as a unit of "metres-all"; the distinction being drawn is that these are the whole
track's arrays, indexed one frame at a time.

### F7 — `replace_measurement` restated nine fields to change two

The function existed to attach an unfolded velocity and a fold index to a frozen
`Measurement`, and did it by naming all nine fields in a fresh constructor call.
`dataclasses.replace` is that operation, `replace` was already imported in the module, and the
hand-written form silently drops any field added to `Measurement` later.

Now one call, with the Parameters and Returns sections a public function owes. The reduction is
eleven lines to four. A test was added that unfolding leaves the seven fields it must not touch
alone, and verified red against a deliberately broken `replace` before being kept — a detection
carries the evidence a plot and a CSV are drawn from, so a perturbed `range_index` would put
the marker off the peak it was measured at and the picture would still look like a detection.

### F8 — A fold that was the identity function

`scripts/run_scenario.py` carried `folded_measurement_velocity_mps(measurement, burst)`, which
folded `measurement.velocity_folded_mps` into the burst's unambiguous interval. That value was
read off the map's own velocity axis, so it was already inside `[-v_unamb, +v_unamb)`, and
`fold_velocity_mps` is the identity there. The function returned its input, under a name and a
docstring describing the fold of an *unfolded* value, typed `(Any, Any) -> float`.

Inlined to `measurement.velocity_folded_mps`, with a comment saying why the folded component is
the one the marker goes on and what plotting the unfolded one would do.

### F9 — Two contracts a caller could not see

`ScenarioTracker.min_unfold_frames` is public, is printed by `scripts/run_scenario.py` and is
asserted by a test, but was assigned in `__post_init__` without being declared as a field or
appearing in the class docstring — so neither `help()` nor the dataclass signature admitted it
exists. Declared `field(init=False)` and given an Attributes entry, alongside `manager` and
`frames`, saying why it is derived rather than passed: it is a consequence of `sigma_range_m`,
`fold_span_mps`, `frame_time_s` and `unfold_sigma_gate`, and setting it independently would let
a track unfold before its range slope can tell it which fold to take.

`save_figure(path: Any)` — the docstring already said `pathlib.Path or str`, so the annotation
was the half that was wrong. `figure` stays `Any`: naming matplotlib's type would need the
import this module exists to defer.

`render_range_doppler`'s Parameters listed `bistatic` and `dynamic_range_db` after the three
tracking arguments, in neither the signature's order nor any other. Reordered.

### F10 — What was measured, and not changed

Q3 claims carry a benchmark or they are not findings. The harness is the end-to-end scenario
run, which is what the brief asks for.

`scenarios/scenario_003_tracking.toml`, 120 frames, every frame detected and tracked, under
`cProfile`: 3.29 s wall, of which

| Where | `tottime` | Share |
| :--- | :--- | :--- |
| `core/signal.py` — `fmcw_deramp_baseband` + `thermal_noise` | 0.88 s | 27% |
| `core/detection.py` — CFAR, clustering, `nonzero` | ~0.50 s | 15% |
| `numpy.fft` | 0.21 s | 6% |
| **everything in `pipelines/`** | **0.09 s** | **2.6%** |

`frame_detections` is the whole of that 0.09 s, and it is the per-cluster interpolation loop,
not the two `np.roll` calls (0.036 s over 480 calls). `slope_velocity_mps`'s `np.polyfit` — the
one call in scope that looks like a candidate, since it runs an SVD to fit five points — does
not appear in the profile at all.

So there is no speed change to make here. The pipeline layer is 2.6% of a run that is dominated
by signal synthesis and CFAR, both in `core/`, and a change justified on speed grounds in this
scope would be unmeasurable by construction. Recorded so that the next reader does not
re-derive it.

For reference, the tracking run itself takes 2.12 s of that 3.29 s, reproducible to 0.01 s
across repeats, and is identical at `sigma_accel_mps2` of 5.0 and 2.0 — see
[F3](#f3-the-sigma_accel_mps2-margin-no-longer-exists).

### F11 — Left alone deliberately

**`RangeDopplerProduct` → `RangeDopplerMap`: recommended against, as framed.** refactor-001
§2.3 blocked this on the field, since `RangeDopplerMap.rd_map` is a tautology, and it is still
blocked — because no candidate replacement is better than what it replaces. `values` is vaguer;
`amplitude` is a bare physical name of exactly the kind §3 exists to reject; `map_complex`
describes the dtype rather than the quantity. Meanwhile `rd_map` is the term the specification,
the scenario documents and every docstring already use for this array. And with
`form_range_doppler_map` now naming the map at the point of production, the word is where the
reader needed it. "Product" is a weak noun, but 32 call sites to trade one wart for another is
not a good trade. Revisit if a better field name presents itself.

**`burst` for a configured `Radar`: reported, not changed.** A burst is conventionally a group
of pulses, not a radar. Here it denotes a fully specified `Radar` transmitting one CPI within
the frame, which is a stretch of the word. But it is the `[[burst]]` table name in every
scenario TOML, so it is a config-format decision and not a rename; changing it would break
every shipped scenario file. The module docstring already explains what a burst is here, at
length, which is the right mitigation.

**`frame_detections(product, config)` vs `dual_prf_measurements(products, spans, *, config)`.**
Sibling functions disagree on whether `config` is keyword-only. Making them agree is a Q7
improvement worth about two call sites and a test edit, and it is a breaking signature change
for a benefit that is purely consistency. Recommended, not applied.

**`peak_range_velocity`'s bare tuple, and `build_tracker`'s bare `bool`.** Both are weak return
contracts. Both have one call site, both are documented, and neither has been misused.

**No test tolerance was touched**, and no number in any test moved. R1.3.2 did not come up: the
one figure that moved is a docstring claim about a measurement, re-derived by re-running the
measurement, and it is recorded in [F3](#f3-the-sigma_accel_mps2-margin-no-longer-exists).

## Result

`make check` passes. 917 tests, against the 915 the audit started from — the two additions are
`replace_measurement`'s field-preservation guards, verified red against a broken implementation
before being kept.

No behaviour changed. Every edit in this audit is a rename, a deletion of provably dead
arithmetic, a type annotation, or a correction to prose — which is the honest answer to §1.2's
question for this scope: the pipelines compute what the specification says, and what was wrong
with them was mostly what they said about themselves.
