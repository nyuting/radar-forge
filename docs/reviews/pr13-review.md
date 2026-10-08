Hi Michael, thanks for this. It closes most of PR C, and the PR description made it easy to check. Yuting drafted this re-review with Claude. Line numbers are at `5a9495a` unless they say otherwise. Section numbers like (#21) are `pr1-review.md`'s; pull requests are always written "PR #11". Two specs are cited, so each section number names its spec: "scenario-003 §14.11" is `spec/scenario-003-tracking.md`, "tracker-001 §13" is `spec/tracker-001.md`. "PR D" means the follow-up PRs in tracker-001 §13, steps 1–7, one PR per step (step 4, the spherical model, is a new feature and can go in any order).

**Verdict: nearly there.** C1, C2, C3, E1 and E2 are done. Claude re-ran the C1 grep at `5a9495a` and it returns nothing. Before merge, please:

- rebase onto main, which now includes PR #14, and fix the `time_utc` test that breaks after it;
- fix the dual-PRF plot;
- help us fix one wrong figure (191 m/s), which main got wrong first;
- make the pairing match the circular range axis it now runs on;
- pin C4 with a test;
- add a drift check for the copied TOMLs.

The rest is for discussion, or small.

Claude checked the external citations in this review and in `spec/tracker-001.md` against the sources themselves. The citations are Richards §5.5.4, Bar-Shalom, Li and Kirubarajan §5.4.2, Singer (1970) and the Stone Soup source paths. The spec's fixes landed on main before this was posted.

### Your five decisions

1. **Mutual nearest neighbours: agreed.** The "exactly one partner" rule was ours, from #21's suggested docstring, and your frames 1 and 4 show it was too strict. One ask remains: make the distance circular (see "pairing on a circular range axis" below).
2. **The range period follows the map: agreed.** It's `core/radar.py` that's wrong, and not only by the factor of 2 (see "two definitions of unambiguous range"). In this PR, the wording only; the formula is a follow-up.
3. **The blind zone is reported, not tuned away: agreed.** But as the simulator stands it isn't eclipsing (Inaccurate statements, item 5). In this PR, the wording only; the tracker side is a follow-up.
4. **PR A's settings, untuned: agreed.** σ_a and τ were derived from scenario 001's truth, so they aren't arbitrary. Please report each run's mean NIS beside its RMSEs (see the C4 section), so "untuned" comes with a consistency number.
5. **`last_associations` and `last_nis`: please move them onto `TrackSnapshot`, in this PR** (see "where the last scan's associations live").

The two follow-ups are under "After this PR", and we'd love your help with both.

### Before merge: rebase onto main (PR #11, PR #14)

PR #11 (the citation check) merged after you opened this. It changes only the References blocks of `pipelines/scenarios.py` and `teaching/scopes/track_plot.py`, and your hunks in both files start lower down. So git should merge them cleanly, and GitHub reports the PR as mergeable. But if a conflict is resolved by taking this branch's copy of either file, Blackman & Popoli and the old `S5.3`/`S8.2` pointers come back.
*Check:* after the rebase, `git grep -nE 'Blackman|S5\.3|S8\.2' -- src` returns nothing.

[PR #14](https://github.com/nyuting/radar-forge/pull/14) (`feat/trajectory-csv-data-001-names`, the rebase your "Not in this PR" was waiting for) has merged too. `git merge-tree` against main finds no textual conflicts, but one test will fail after the rebase. PR #14 rewrote `data/flight_coordinates.csv` to data-001 §6.3's `time_s,latitude_deg,longitude_deg`, and its `time_s` counts from the first fix, so the shipped track has no epoch any more (`load_flight_csv(...).epoch` is `None`). All three `scenario_003_ukf_*` TOMLs read that file. So `trajectory_epoch` returns `None`, `_columns` drops `time_utc` (`run_scenario.py:371-373`), and `metadata.json`'s `epoch_utc` is null. `test_run_scenario.py:141-152` asserts the opposite: "The trajectory has an absolute epoch, so time_utc is written", full column tuples, and `metadata["epoch_utc"].endswith("Z")`.

Suggest: keep the code, since writing `time_utc` only when there's an epoch is what §6.5 asks for, and change the test. Either assert the no-epoch case on the shipped track, or run that assertion on a small `time_utc` trajectory fixture so the with-epoch branch stays covered. Also check that the PR description's "`time_utc` and `epoch_utc` come from the trajectory's first fix" still holds. `load_flight_csv`'s `altitude_m` is now optional, and an `altitude_m` column in the CSV takes precedence. The shipped CSV has no such column, so the TOMLs' `altitude_m = 1500.0` still applies.
*Check:* after the rebase, `uv run pytest tests/scripts/test_run_scenario.py` passes, and `scripts/run_scenario.py` still imports `load_flight_csv` from `pipelines/trajectories.py`.

### Before merge: the dual-PRF plot draws burst B's detections on burst A's map (#21)

`FrameTracks.measurements` now holds every detection from both bursts (`tracking.py:1078-1082`). But `render_frame` still draws all of `record.measurements` on burst 0's map (`run_scenario.py:507-515`), each at its own folded velocity. So in an S3 run, burst B's detections land on burst A's map at velocities folded into burst B's interval. The comment just above (`:503-505`) says "repeating them on burst B would imply they were measured there", which is now false, and this is the docstring-against-code problem #21 was about. The end-to-end test runs with `--no-plots`, so nothing catches it.

Suggest: draw each burst's own detections on its own map (`if measurement.burst_index == index`), or keep burst 0's only and say so in the comment.

### Before merge: the 5:6 kHz pair resolves ±229 m/s, not ±191 m/s

At f0 = 9.8 GHz, λ = 30.6 mm. Burst A (5 kHz) folds at ±38.24 m/s and burst B (6 kHz) at ±45.89 m/s. The pair is unambiguous out to 6 × 38.24 = 5 × 45.89 ≈ **±229 m/s**. 191 m/s is S2's pulsed unambiguous velocity.

This mistake is ours: main says ±191 m/s for the S3 pair, and the PR inherited it. Since you're in these files anyway, could you help us fix it everywhere in this PR? On the PR head:

- `dual_prf_detections`: "191 m/s for scenario 001's 5:6 kHz pair" (`tracking.py:585`), and the default `max_velocity_mps: float = 191.0` (`:576`);
- the `description` of `scenario_003_ukf_fmcw_dual_prf.toml`: "unfolds velocity to +-191 m/s".

And on main, where it started:

- `dual_prf_measurements` (`pipelines/tracking.py:389`, `:397`, `:415` on main): the same default and the same sentence, which `dual_prf_detections` copied;
- the `description` of `scenario_001_fmcw_dual_prf.toml` (`:7`): "reaching +-191 m/s";
- `scenario_003_tracking_dual_prf.toml`: the header comment (`:11`) and the `description` (`:19`);
- scenario-003 §14.7 (`spec/scenario-003-tracking.md:925`): "resolves velocity in the *waveform* to ±191 m/s".

`unfold_doppler_dual_prf`'s example passes `max_velocity_mps=191.0` (`ambiguity.py:179`). That one is fine: it's a search bound for an 80 m/s target, not a claim about the pair.

Your own test (`test_the_dual_prf_search_is_bounded_by_v_max_mps`: "resolves up to about 229 m/s") and commit `5a9495a` have it right. Claude worked this out from the TOMLs; please check it against `unfold_doppler_dual_prf`.

The default itself is the bigger problem. `5a9495a` had to fix `step`, which silently took the 191 default instead of `v_max_mps`. A scenario-tuned default on a library function invites the next caller to do the same. Please make `max_velocity_mps` required (on `dual_prf_measurements` too). `range_tolerance_m=150.0` ("about two of S3's range bins", `:624`) could come from the two maps' bin widths in the same way. This is pr2-review's "scenario-specific numbers in docstrings" point, in a default.

### Before merge: pairing on a circular range axis

On the UKF path `step` calls `dual_prf_detections(..., wrap_range=True)` (`tracking.py:1456-1465`), so range is circular. Two things don't match that:

- **The distance isn't circular.** `distance_m = np.abs(ranges_a_m[:, None] - ranges_b_m[None, :])` (`:664`). A target just past the wrap shows at about 0 km in one burst and about 50 km in the other, and never pairs.
- **S3's two bursts wrap at different spans.** Burst A sweeps 2 MHz in 200 µs at f_s = 4 MHz, so its axis spans c·f_s·T/(2B) = 60 km. Burst B's 166.7 µs gives 50 km. Your scenario-003 §14.11 has the same numbers. A target beyond 50 km folds differently in each burst. Meanwhile `folding_layout` takes burst A's span only (`:1224-1227`).

The docstring's "Range is unambiguous in both bursts" (`:591-592`) is the assumption that makes both safe. It holds for scenario 001's 0–120 s window, where the S2 TOML puts the target at 8.4–17.9 km, but nothing checks it. Suggest one of these:

- a guard in `_check_settings` that the two bursts' `range_axis_m` spans agree, with the FMCW sweep times chosen to match;
- or keep the assumption, state it in `ScenarioTracker`'s Notes, and use the circular distance `folding_layout.residual` gives you, so the wrap edge at least pairs.

### Before merge: one wrap rule (#21, #22)

`folding_layout` is used for the exported range (`tracking.py:1537`), the truth line (`run_scenario.py:592`) and the scored error (`tracking.py:1880`). That is what #21 asked for. One place still wraps by hand: the range-Doppler truth marker, `folded_range_m = frame.range_m % burst.unambiguous_range_m` (`run_scenario.py:498`). Your Decision 2 shows that's half an FMCW map's span. There's no visible effect at S1's or S3's target ranges, but it is now known to be wrong. Please wrap it at that burst's `range_axis_m` span, for example through a per-burst `StateLayout`.
*Check:* `git grep -n '% burst.unambiguous_range_m' -- scripts` returns nothing (it finds `:498` today; the `metadata.json` key at `:203` is fine).

### Before merge: C4 isn't pinned by a test (#18)

The C4 table (118 / 59 / 118 against targets of 118 / 13 / 94) is the PR's headline result, and it's recorded in scenario-003 §14.11. But the only UKF end-to-end test is 3 frames (`test_run_scenario.py:120-158`). Please add a `@pytest.mark.slow` test per UKF TOML asserting at least the C4 target (≥ 118, ≥ 13, ≥ 94 confirmed frames of the primary track), as `test_scenario_003.py` does for the Kalman runs. Otherwise the next change to `core/tracking` can halve S3 and stay green.

Please also add each run's mean NIS to the C4 table and scenario-003 §14.11, beside the RMSEs. Decision 4 is right not to tune here, and the NIS shows whether PR A's settings are consistent on these waveforms anyway. Report it per measurement dimension, as under "Smaller fixes": S1 measures range alone, S2 and S3 measure two quantities.

### Before merge: `metadata.json` records settings the run didn't use (#20)

`tracking_metadata` writes `asdict(tracker.tracking)` (`run_scenario.py:267`), and its docstring says these are "the settings … the run used". A UKF run therefore reports:

- `sigma_range_m = 21.635652855125496`, S1's frozen bin σ (74.948 m / √12);
- `sigma_velocity_mps`;
- `unfold_sigma_gate` and `n_slope_frames`.

The UKF TOMLs say "No sigma_range_m or sigma_velocity_mps: the UKF path takes its measurement noise from …". A Kalman run likewise reports `acceleration_correlation_time_s`. A student reading `metadata.json` will believe the frozen σ was used. Please write only the fields the chosen estimator reads, or put the others under an `unused` key.

### Before merge: the copied TOMLs need a drift check (#18)

Each `scenario_003_ukf_*.toml` is its scenario 001 twin above `[detection]`, apart from the name and description. Claude diffed all three at the PR head and they match. But nothing keeps them matching: change a burst in `scenario_001_pulsed_medium_prf.toml` and the UKF S2 run silently becomes a different scenario. Main's two Kalman files have the same gap: `scenario_003_tracking.toml` differs from S1, and `scenario_003_tracking_dual_prf.toml` from S3, only in name, description and the 663 s window.

The copies are fine for now. `load_scenario` has no include, and `[detection]`/`[tracking]` don't belong in scenario 001's files. Please add one parametrised test to `test_scenario_003.py`:

- load each pair with `tomllib`;
- drop `[scenario]`, `[detection]` and `[tracking]`;
- assert the remaining tables are equal.

For the two Kalman files, also drop `[trajectory].start_time_s`. Comparing tables rather than text leaves the comments free to differ.
*Check:* change one `[[burst]]` value in a scenario 001 TOML locally, and the test fails.

**Later, not this PR:** all three UKF files end in the same `[detection]`/`[tracking]` block, and the two Kalman files share theirs too. So scenario 003 is really a few scene files paired with two processing setups. A follow-up after PR D could make that explicit:

- a top-level scenario file names a `scene` file, a `processing` file and an optional `[override]` table (for the 663 s window);
- the loader stops with an error on a key set in both parts, on an override of a key that doesn't exist, and on a pairing that doesn't work (dual-PRF pairing with one burst, `track_aided` unfolding on a waveform that never folds);
- `metadata.json` records the full merged config.

That PR would delete the copies, and this test with them. We'll open an issue for it so it doesn't depend on this thread. Would you help us scope it? You've just lived with the copies, so you'll know where the seams are.

### Before merge: References (#14)

- **`range_axis_m`** (`scenarios.py:668`) is public, states the `c/2 PRF` span and the complex-sampling `[0, f_s)` span, and has no References section. CLAUDE.md requires one. Richards (2014) §5.5.4 (ambiguity resolution, verified in PR #11) would do for the pulsed span. For the complex-baseband point, cite a section that covers complex (I/Q) sampling, or say it's derived here.
- **`dual_prf_detections`** has no References either. Richards §5.5.4 covers coprime-PRF unfolding.
- **`score_primary_track`** (`:1838-1840`) cites Bar-Shalom, Li and Kirubarajan §5.4. `estimation.py` and PR #11's citation check use **§5.4.2** for the NIS test, so please match.
- **`TrackingConfig`** names "the white-noise limit of Singer's model" (`:298`) without a citation. `core/tracking/motion.py` already cites Singer (1970), so copy that one.

### Discuss: where the last scan's associations live (#5, #10)

`Tracker.last_associations` and `Tracker.last_nis` (`tracker.py:398-410`) are public dicts that each `process()` overwrites. That works, but:

- **Stone Soup keeps no per-scan state on the tracker.** `MultiTargetTracker` (`stonesoup/tracker/simple.py`) holds only its tracks. The association rides on the track: each `GaussianStateUpdate.hypothesis` carries `.measurement` and `.measurement_prediction` (`stonesoup/types/update.py`, `types/hypothesis.py`).
- **The repo already does it per track.** `KalmanFilter` keeps `last_nis`, and `KalmanTracker.step` returns associations in its `FrameResult`.
- **The snapshot is too thin.** `_track_ukf` already has to read the live tracks for `n_hits` and `n_misses` (`tracking.py:1543-1544`).
- **Births are left out.** A track born in the scan isn't in `last_associations` (the birth loop, `tracker.py:413-421`, records nothing). So a new track's first `tracks.csv` row has no `associated_detection_id`, although the track was started from a detection. A student reading the file sees a track appear from nowhere.

Suggest: give `TrackSnapshot` `measurement_index: int | None`, `nis: float | None`, `n_hits` and `n_misses`, and set `measurement_index` at birth too. `spec/tracker-001.md` §10.1 records this as the target shape, so it's where the package is going anyway, not only one reviewer's preference. Then `process()`'s return value carries everything `tracks.csv` needs, and `TrackRecord` is built from one object, not three. Please do it in this PR. The change is small, and public attributes that ship need a deprecation before they can be removed (#24). If you'd still rather leave it for PR D, please at least say in the attribute docs that the dicts hold only until the next `process()`. Either way, one sentence in `Tracker`'s Notes comparing it with Stone Soup's `hypothesis` would keep `docs/tracking/README.md`'s promise that the docstrings say where the two differ.

### Discuss: `SensorRoute` → `SensorRegistration` (#5)

This one is about `core/tracking`, not your diff, and it reverses pr1-review's own suggestion. #5 renamed `Sensor` to `SensorRoute` because the class "only routes". That was the wrong call, for three reasons:

- **It doesn't only route.** It holds the model IDs a sensor's measurements may use, which is the routing. It also holds `observable`, the coverage test the tracker calls in `_can_see` (`tracker.py:492-499`), and it holds `batch()` (`measurement_models.py:301`). The docstring's "It does two jobs, and only these two" (`:235`) misses the third.
- **The rest of the API never took up "route".** The field is `sensor_id`, not #5's `route_id`. The tracker takes `sensors=`, exposes `.sensors` and registers through `add_sensor()`. Your own code reads `tracker.sensors[SENSOR_ID].batch(...)` (`pipelines/tracking.py:1525`). pr2-review flagged the same mismatch.
- **"Route" isn't a tracking term,** so a student can't look it up.

The class is the tracker's record of a sensor whose detections are made elsewhere. Suggest `SensorRegistration`. Then `tracker.add_sensor(SensorRegistration("radar", ("position",)))` says what it does, `.sensors` is the registry, and `sensor_id` is already right. The new name still avoids the clash #5 was about: Stone Soup's `Sensor` *generates* detections through `measure()`. So please keep the Notes paragraph that says so (`:267-269`).

While it's open, two small fixes:

- **`observable` → `covers`.** In estimation, "observable" means observability: whether the state can be recovered from the measurements. That's a textbook term, and `kalman.py:354` and `:375` already use it that way. This field is a field-of-view test, which the tracker itself calls "can see".
- **The docstring** should list all three jobs. Or move `batch()` out to a `MeasurementBatch` constructor, and the "two jobs" sentence becomes true.

Your PR never names the class: it goes through `build_tracker(sensor_id=...)` and `.sensors[...]`. So this needn't be done here. tracker-001 §13 step 0 does it, with `covers` and `predict_to(timestamp_s)`, as a small PR on main, and it doesn't touch your diff. Would you help with step 0, either by taking it or by reviewing it? You've used these names more than anyone. The wider naming and module-shape decisions, with the reason for each, are tracker-001's TD13 and §3. Please read them before starting PR D.
*Check:* `git grep -nE 'SensorRoute|^\s*observable ?:|observable=|\.observable\b|``observable``' -- src tests docs/tracking` returns nothing (on main at `82961d4` it finds 43 lines). The pattern is narrow on purpose, so it skips the observability prose that should stay.

### Discuss: size and shape of `pipelines/tracking.py` (#22, #24)

The file goes from 1080 to 1923 lines, and `__all__` grows to 28 names. #22 hoped for a small adapter. Most of the growth is two trackers in one class:

- `__post_init__` returns early on the UKF path (`:1223-1237`);
- `step` dispatches on `isinstance(self.tracker, Tracker)` (`:1456`, `:1470`);
- `_n_measured` is UKF-only, while `min_unfold_frames`, `_range_history`, `_unfold_frame` and `unfold_reference_id` are Kalman-only.

This is planned (the PR says scenario 003 moves to `Tracker` later), so it's not a blocker. `spec/tracker-001.md` agrees: §13 step 6 moves scenario 003 onto `Tracker`, and §3's decoupling rule says that no pipeline keeps a code path per tracker after that. So the split is fine as a stage on the way. The asks are about making its removal a deletion rather than surgery:

- **Decide the path once.** `__post_init__` already chooses. `step` then asks again, twice, through `isinstance(self.tracker, Tracker)` (`:1456`, `:1470`). Set a flag (or keep `tracking.estimator`) in `__post_init__`, and have `step` read that. Also, `wrap_range` is a property of the path, not of the tracker class.
- **Keep the Kalman-only members together.** `min_unfold_frames`, `_range_history`, `_unfold_frame`, `is_unfoldable`, `unfold_frame_of`, `_reference_track`, `_advance` and `_unfold_all` exist only for the Kalman path, but they're spread through the class. Put them in one block, under a comment that says they go with `KalmanTracker` (tracker-001 §13 step 6). Then step 6 deletes one block and one branch.
- **Which new names must be public?** 13 are new: 12 in `pipelines.tracking.__all__`, plus `scenarios.range_axis_m`. Removing a name later needs a deprecation (#24). `dual_prf_measurements` no longer has a caller in the library, and unlike `dual_prf_detections` it has no `wrap_range`.
- **The lifecycle settings are declared twice,** in `TrackingConfig` and in `LifecyclePolicy`, and copied across field by field at `:1339-1344` (#22). Could `TrackingConfig` hold a `LifecyclePolicy`? That's the rule `spec/tracker-001.md` §8.2 sets for pipeline configs.

The module split itself is tracker-001 §3's target column, done in §13 step 7. Please say if it doesn't fit what you found writing this file.

### Discuss: your view on spec/tracker-001.md §8.3 and §7.2

You know scenario 003's bootstrap better than anyone, and the spec now decides where it goes after this PR. §8.3 (TD9) splits it in two. The mechanism goes in `core/tracking`: a measurement model's marginal over named components, a per-pair model selector with the range-only retry, and a variance reset on entering re-acquisition. The policy stays in `pipelines/`. Please say if that can't express something `KalmanTracker` does today.

One part of it affects your numbers. The range-only retry puts 1-D and 2-D pairs in one cost matrix, and `KalmanTracker` scores both by raw NIS (`kalman.py:1186-1196`). A 1-D NIS averages 1 and a 2-D NIS averages 2, so the solver prefers the range-only pair. §7.2 asks for negative log-likelihood whenever the dimensions mix, and that may move scenario 003's results when it migrates. Nothing to change in this PR. Your sense of whether it matters on S1 would help.

### Discuss: two definitions of "unambiguous range"

Decision 2 is right for the tracker: the period has to be where the map wraps. And `Radar.unambiguous_range_m` (`radar.py:288-310`) is wrong, though not only by the factor of 2:

- **The simulator samples complex.** `fmcw_deramp_baseband` gives a beat of α·τ for any delay (`signal.py:617-717`), so the map wraps at c·f_s/(2α), as you say. S1's property says 37.5 km, but its map spans 74.9 km.
- **A sawtooth has a second limit, and the simulator doesn't model it.** The delay must be shorter than the chirp, τ < T. Past that, the echo is mixed against the next ramp. That puts a ceiling of c·T/2 on range. For S1 it's 150 km, so f_s is the tighter limit. S3 samples at f_s = 2B, which makes c·T/2 equal to c·f_s/(4α): 30.0 km for burst A and 25.0 km for burst B. So the property's current value happens to be right for S3, and the far half of S3's 60 km and 50 km maps doesn't exist in hardware.

So the FMCW number that matches a real radar is min(c·f_s/(2α), c·T/2), and doubling the formula would make S3 wrong. The target stays inside every limit (8.4–17.9 km in scenario 001's window, against 25 km at the tightest), so none of the C4 numbers change.

For this PR, the wording only. Please say in `range_axis_m`'s Notes and scenario-003 §14.11 that `Radar.unambiguous_range_m` is half of every FMCW map's span. For S1 that understates what a real radar could reach. For S3 it happens to match hardware, because S3's maps span past c·T/2, where the simulator is more generous than hardware. Then link the issue. The formula is a follow-up (see "After this PR").

### Inaccurate statements

1. **`cluster_detections` is circular now.** The module docstring (`tracking.py:35-45`) says "the clean fix is a `wrap_axes` option on `cluster_detections` itself, which belongs to that module's own workstream". The `frame_detections` Notes (`:412-417`) say "`cluster_detections` is not [circular]". The option landed in PR #4 (`detection.py:1062`), scenario-003 §13.5 says so, and this PR uses it for range (`:459`). Please update both to say the roll stays until the Doppler axis adopts `wrap_axes` too, as scenario-003 §13.5 does.
2. **"One-to-one"** survives from the old rule in `DetectionStatus` (`:149`, "the pair is one-to-one") and in `dual_prf_measurements` (`:734`, "no one-to-one partner"). Both should say "mutually nearest".
3. **Scenario numbers in docstrings** (pr2-review): `ScenarioTracker`'s Notes give "a target at 16-22 km" and "spans … 50 km and more" (`:1180-1182`), and the S2 TOML on main says the target is at 8.4–17.9 km. They may cover different windows, but a reader can't tell which. Numbers like these belong in scenario-003 §14.11, where the window is stated.
4. **DF9 is overclaimed.** The PR says floats are written at shortest round-trip precision. That holds for the three new CSVs, but `truth.csv` is still `f"{frame.time_s:.3f}"` and `:.6f` (`run_scenario.py:745-751`). Move it onto `_cell`, or say in the PR that it's left for PR D.
5. **S2's blind zone isn't eclipsing, as the simulator stands.** Scenario-003 §14.11 says "This is eclipsing by the next transmitted pulse" (`spec/scenario-003-tracking.md:1019`). But `pulsed_baseband`'s Notes say "Eclipsing is not modelled" (`signal.py:769`). What loses the target is the receive window. `within_pulse` (`signal.py:800`) keeps only the part of an echo that falls inside its own row, so an echo that starts in the interval's last 10 µs loses its tail. That matches the trailing half of real eclipsing. The leading half isn't modelled: a real receiver is off while it transmits, so it would also lose the leading part of an echo at a folded delay under 10 µs (folded range 0–1.5 km), the mirror image of the tail the simulator cuts. A real S2's blind zone would be about twice as wide, centred on the wrap. Suggest, for scenario-003 §14.11 and anywhere a docstring repeats it: "The simulated receive window truncates an echo that starts in the interval's last 10 µs, which reproduces the trailing half of eclipsing. The leading half, the start of an echo that arrives while the next pulse is being transmitted, is not modelled (see `pulsed_baseband`'s Notes), so a real radar's blind zone would be about twice as wide."

### Smaller fixes

- **`score_primary_track`:**
  - `confirmation_latency_s` runs from the run's first frame to the primary track's first confirmed frame (`:1883`). S2's run gives six track IDs, so the primary track may start late, so this isn't a confirmation latency. Measure it from the track's birth, or rename it.
  - `mean_nis` on the Kalman path averages 1-D bootstrap NIS with 2-D NIS. The docstring compares the mean with "the measurement dimension" (`:1804-1806`), which changes partway through. Report it per dimension, or divide each NIS by its dimension.
  - S2's 1.4 m range RMSE is measured modulo 6 km (`folding_layout.residual`, `:1880`). Please say so in the docstring and scenario-003 §14.11, or a student will read it as absolute range accuracy.
  - "Primary track" stands in for a track-to-truth associator (Stone Soup: `TrackToTruth` with SIAP or OSPA). One sentence saying so gives the README's GOSPA follow-up a place to plug in.
- **`bin_quantisation_sigmas`** (`:766-806`): Δ/√12 is a quantisation-only floor. It ignores the SNR term, and since `frame_detections` returns power-weighted centroids it may overstate the error. Please say so in a Notes section. A high mean NIS at low SNR is then expected, not a bug. Stone Soup's `RangeRangeRateBinning` is the nearest analogue.
- **Guards** (style.md §8; pr2-review's `_check_*` ask):
  - `_check_settings` computes `min(burst.unambiguous_velocity_mps …)` twice, once in the condition and once in the message (`:1283-1290`).
  - The `math.nan` placeholder inside a comprehension (`:1511-1524`) relies on the tracker to reject it "loudly" downstream. An explicit check before the batch is built would be easier to follow.
  - `DetectionConfig.variant` is only checked to be a string. `estimator` and `unfolding_mode` are checked against their `Literal` in `_check_settings`, but `variant` isn't.
- **Which track is primary is decided in three places, with two rules.** `score_primary_track` (`:1853-1870`) and `render_track_frame` (`run_scenario.py:573-584`) pick the track confirmed in the most frames. `render_frame` picks `max(confirmed, key=n_hits)` (`:516-518`). So the range-Doppler marker can show a different track from the one that is plotted and scored. One helper would fix it.
- **Loops (#15).** `run_scenario.py:575-590` and `score_primary_track`'s counting loop (`:1854-1857`) have no "why" comment.
- **dtype.** Please state it on `np.diag([1.0, …])` (`:1335`), `np.diag([...])[:n, :n]` (`:1505`) and `np.asarray(...)` (`:1878-1879`). `run_scenario.py:581` has `list[np.ndarray[Any, Any]]`, where style.md §5 wants `NDArray[np.float64]`.

### Tests (#18, testing.md)

- **Private helpers tested directly:** `_strictly_nearest` (`test_tracking.py:562-578`) and `_break_at_wraps` (`test_track_plot.py`). Test them through `dual_prf_detections` and `render_range_time_history`. The tie case is already a good public test.
- **Tolerances without a reason:** `atol=60.0` (`test_a_track_crosses_the_range_wrap_under_one_id`) and `atol=2.0 * 191.1935 / 256`. Please add a comment saying why each number, as testing.md §2 asks.
- **Literals derived from c:** `74.9481145` (`test_tracking.py:275`, `:369`, `:448`, `:534`) and `191.1935` are what `range_axis_m` and `Radar.unambiguous_velocity_mps` compute. Please derive them from the bursts, so a change to `SPEED_OF_LIGHT_MPS` or a TOML can't leave them stale.
- **`bin_quantisation_sigmas`'s Raises** (fewer than two bins) has no test.

### After this PR: two follow-ups we'd like your help with

You found both of these, and you have the numbers to hand, so we'd love your help with all of each. Both stay out of PR #13, so it doesn't grow.

- **`core/radar.py`'s FMCW unambiguous range** (Decision 2). Please open the issue and the PR. The PR would:
  - set `Radar.unambiguous_range_m` to min(c·f_s/(2α), c·T/2) for FMCW, and `BistaticRadar`'s range-sum version to match, with a Notes line and a reference for each limit;
  - update S1's literal from 37 474 m to 74 948 m (`test_radar.py:135`, `test_scenarios.py:117`). S2 and S3 don't change;
  - warn when a map spans past c·T/2, as `_warn_if_stop_and_hop_is_strained` does for motion within a chirp;
  - update the "twice" in `range_axis_m`'s and `ScenarioTracker`'s Notes.
- **S2's blind zone, on the tracker side** (Decision 3).
  - A miss rule that knows the detection probability: a miss doesn't count while the predicted folded range is in a known blind zone, where P_D = 0. That's the principled alternative to a longer coast, and it keeps the blind zone visible.
  - A test with analytic ground truth: S2's undetected frames are exactly those whose folded truth range lies within the blind zone below the 5996 m wrap. That needs the zone's width derived from the pulse length and the detection threshold, not read off a run (the PR measured about 200 m).

### What's good

- **One writer and one column tuple per data-001 file,** in §6's order, with a test that parses the spec's own tables to check it. That test is the best thing in the PR. Schema 1.1.0 changes the spec, the writer and the test together.
- **Scoring kept apart from writing,** so metrics can be recomputed from the files. That's Stone Soup's `MetricGenerator` split too.
- **TOML type-checked, nothing coerced,** with integer-for-float accepted (`:497-499`) and booleans refused. Stone Soup's `Property` says it is "not used for any type checking", so this is a deliberate, well-chosen divergence. One line in `configs_from_scenario` saying so would match the other modules' Stone Soup notes.
- **Folded range done the Stone Soup way:**
  - the period is on the measurement coordinate;
  - the filter state is continuous;
  - residuals take the short way round;
  - the export is wrapped.

  It's the same pattern as Stone Soup's `Bearing` measurements, applied to range. Stone Soup has no folded range of its own.
- **`TrackRecord` copies** fix the "every frame shows the final state" trap, and the docstring says why.
- **Mutual-nearest pairing** is argued with measured frames (Decision 1). Ties are refused rather than broken, every detection says what became of it, and `test_a_false_alarm_beside_the_target_does_not_stop_the_pair` pins the case that broke the old rule. That rule was our suggestion, and you were right to measure it rather than take it on trust.
- **S2's blind zone is reported, not tuned away** (Decision 3, scenario-003 §14.11). That's the honest call.
- **The conventions hold.** There are no scenario labels in the logic; `SENSOR_ID` and `MEASUREMENT_MODEL_ID` are shared constants; every error message ends with a full stop; there's no `3e8`, no `tile`/`repeat`. The comment density, about 8%, is the same as `detection.py`'s.
- **The `velocity_unfolding` key round-trips** back into `metadata.json` (`a52f696`), with a test. Good catch.

Thanks again, Michael!

---

# Notes for Yuting (not posted)

## How this was checked

- Claude read every line cited above at `5a9495a` through `gh api` (no checkout). Three helper agents also did read-only passes: one against `pr1-review.md`/`pr2-review.md`, one against conventions and #11's citations, and one against Stone Soup's source and docs. Each agent finding kept here was re-read at the cited line. Two were changed after re-reading:
  - `cluster_detections`: scenario-003 §13.5 already records that `wrap_axes` landed, so only the module docstring and Notes are stale. Adopting it for Doppler is a follow-up, not a blocker.
  - The Bar-Shalom section number: `estimation.py:27,71` cites §5.4.2.
- The figures were recomputed from the TOMLs:
  - λ = c / 9.8 GHz; v_ua = 38.24 and 45.89 m/s; the pair is unambiguous to ±229 m/s;
  - FMCW spans c·f_s·T/(2B) = 60 km and 50 km.
- **Not done:**
  - `make check` wasn't run on the PR branch. CI is green on py3.11/3.12.
  - S1–S3 weren't re-run, so the C4 numbers are the PR's own.
  - The reading-view generator wasn't run, so C3 is "by reading".
- External citations (this review's and tracker-001's) were checked against their sources by a separate agent session before posting. Its fixes went into `spec/tracker-001.md` only; this file needed none.
- The UKF TOMLs were diffed against scenario 001 at the PR head. Above `[detection]` only the header comment, `name` and `description` differ. The three files' `[detection]`/`[tracking]` blocks are identical.
- The five decisions were checked against the code at `5a9495a` (`git show pr13:…`):
  - Decision 2: `fmcw_deramp_baseband` (`signal.py:617-717`) has no τ < T limit and no real-sampling step. Recomputed: c·f_s/(2α) = 74.95, 59.96 and 49.97 km; c·T/2 = 149.9, 29.98 and 24.98 km (S1, S3 A, S3 B). S3's c·T/2 equals its current c·f_s/(4α) because f_s = 2B.
  - Decision 3: `pulsed_baseband`'s Notes (`signal.py:769`) and `within_pulse` (`:800`). τ = 10 µs is 1.499 km. `spec/scenario-003-tracking.md:1019` is the only "eclipsing" claim on the PR head.
  - Decision 5: the birth loop (`tracker.py:413-421`) doesn't touch `last_associations`.
  - Not done: S2 wasn't re-run, so the blind zone's ~200 m width and ~7 frames are the PR's own figures.
- **`SensorRegistration`.**
  - `SensorRoute` use on the PR head was read through `gh api` at `5a9495a`. The PR only reaches it through `build_tracker(sensor_id=...)` and `tracker.sensors[SENSOR_ID]`, so the rename doesn't touch Michael's diff.
  - It was chosen over `SensorSpec`, because "spec" suggests hardware parameters, which `core.radar.Radar` holds.
  - The *Check* grep was run on main and finds 43 lines. A bare `\bobservable\b` grep would never come back empty: it also matches the observability prose at `kalman.py:354,375` and comments in two tests.
- PR #11's removals are absent from PR #13's added lines. The only Blackman & Popoli and `S5.3` hits on the PR head are PR #11's pre-merge text, which the rebase replaces.

## Before posting

- **Blocking or not?** Everything under "Before merge" is labelled that way because it's either a wrong statement students will copy (the plot comment, 191 m/s, `metadata.json`) or a gate the PR claims (C4). If you'd rather merge and fix in PR D, the dual-PRF pairing on a circular axis and the per-burst wrap are the two that could wait. Neither affects scenario 001's numbers.
- **`TrackSnapshot`: this PR.** The review now asks for it here, with PR D as a fallback. It touches `core/tracking`, which was PR A's, and grows this PR a little, but the birth-row gap is a data-001 output problem, not only a design one.
- **Land `spec/tracker-001.md` on main before posting.** The review now cites its §3, §7.2, §8.2, §8.3, §10.1 and §13, and the links need to resolve for Michael.
- **`SensorRegistration` reverses pr1-review #5,** and the posted text says so plainly. Where it goes is settled: tracker-001 §13 step 0, a small PR on main. The posted text asks Michael to help, by taking it or reviewing it, so decide which you'd prefer before posting.
- **The module split** is settled too: tracker-001 §3's target column, done in step 7. The posted text now says so and asks Michael only whether it fits.
- **"PR D" = spec steps 1–7, one PR each** is a definition I proposed, not one agreed with Michael. Change the opening line if you'd rather scope it differently.
- **The two-tracker split.** The posted text accepts it as transitional and asks for two cheap changes (decide once, group the Kalman-only members). I'd suggested a pipeline-level KF/UKF parity test earlier, but dropped it: the two paths differ by design (DWNA against continuous Q, `sigma_range_m` against bin-quantisation R, joint against confirmed-first assignment), so it would fail for reasons that aren't bugs. The core-level guard is spec §4.5 / AC4.
- **±191 m/s started on main.** It came in with 7a4b931 (scenario_001_fmcw_dual_prf.toml) and was copied into `scenario_003_tracking_dual_prf.toml` and scenario-003 §14.7. The posted text owns the mistake and asks Michael to help fix main's copies in this PR. `spec/scenario-003-tracking.md:301` and `:768` also say ±191 m/s, about the unfolded velocity the tracker sees. Check whether those mean S2's limit or the S3 pair before asking for them too.
- **Comment density "about 8%"** (What's good) has no record of how it was counted. Re-count it or drop the figure.
- **Composition issue:** open it before posting, so the "Later" paragraph in the drift-check section can link it. Scope: the loader, `run_scenario.py`, the scenario-003 spec and data-001 (merged config in `metadata.json`).
- **`core/radar.py` is yours, and the review asks Michael to take the whole fix.** The recommended formula is min(c·f_s/(2α), c·T/2), not simply doubling it. Please confirm you're happy with both before posting.
