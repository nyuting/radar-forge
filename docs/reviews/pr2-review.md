**Please tick A4.** It's done: confirmed tracks pick first (`tracker.py:386-391`), there are no births inside a confirmed gate (`:406-410`), q is derived in `motion.py`'s docstring, and each has a test.

### Before merge: one tracker, not two (#3, #5)

There are still two of each:

| Name | `kalman.py` | new |
| :-- | :-- | :-- |
| `Track` | `kalman.py:925` | `tracks.py:120` |
| `TrackManager` | `kalman.py:1000` | `lifecycle.py:125` |
| `TrackStatus` | `kalman.py:89` (`Literal`) | `tracks.py:44` (`StrEnum`) |

Both are live: `pipelines/tracking.py` uses `kalman.py`'s, and `tracker.py` uses the new ones. Leaving the new ones out of `__init__.py` hides the name clash, but doesn't merge them. Please either:

- merge each pair into one type here, or
- if the merge needs PR C's pipeline change, say so in `__init__.py` and the PR description, and open an issue that PR C closes.

Either way, `TrackStatus` should be one `Literal`, as #5 asked.
*Check:* `git grep -nE '^class (Track|TrackManager)\b|^TrackStatus\b|class TrackStatus' -- src` gives one hit per name.

### Reuse (#6): reuse, or say why not

- **GNN.** `GlobalNearestNeighbour.associate` (`association.py:244-262`) solves the assignment itself. `:180` mentions `kalman.associate_gnn` only to say it behaves the same. `associate_gnn(costs, np.inf)` accepts negative NLL costs, so I think the body could be `_result(costs, associate_gnn(costs, np.inf))`. If you keep your version for its scaling, which guarantees the most pairs first whatever the cost magnitudes, say so in the Notes.
- **Bistatic.** `BistaticRangeDopplerModel.predict` computes ranges and rates by hand (`measurement_models.py:816-824`). That's reasonable: `BistaticRadar.target_ranges_m` takes geodetic input, and `signal.bistatic_doppler_hz` returns Hz. Please add a See Also naming both and saying why they aren't used here.

### Names (#5)

- `_Geometry`'s `space` parameter and attribute (`measurement_models.py:598`, `:620`, `:653`) → `layout`, to match `StateLayout`.
- `Measurement.sensor_id` and `MeasurementBatch.sensor_id` hold a `SensorRoute.route_id`. Use one name for both.

### Tests (#18, testing.md §8)

`test_ukf.py` tests only the out-of-sequence, `alpha` and predict-first errors. Please add a test for each remaining `Raises`:

- a state whose layout isn't the motion model's (`UKF(...)` and `set_state`);
- a mismatched layout in `innovation_statistics` and `update`;
- a non-finite `time_s` in `predict_to`.

### References (#14): narrower, please

These name a chapter, which #14 allowed. A section number would let a student open the book at the right page:

- **Blackman & Popoli "ch. 6":** `tracks.py:20`, `lifecycle.py:28` and `:150`, `tracker.py:35` and `:158`, and `association.py:25`, `:210` and `:293`.
- **Bar-Shalom "ch. 5":** `coordinates.py:28-30` and `:491`, `association.py:290`, and `measurement_models.py:39-41`. `estimation.py` already cites §5.4/§5.2 for the same material, so make them match.
- **Mardia & Jupp "ch. 2":** `coordinates.py:25` and `:424`.
- **`tracker.py:604-606`:** add §3 to Wan & van der Merwe, to match `ukf.py:131`.
- **`measurement_models.py:526` and `:561`:** `geodetic_to_enu_m` is a function, not a source. Move it to See Also.

### Comments (#13, #15)

- **`tracker.py:209-210`:** the loop is right, but "routes are few" and "clearest form" aren't why it's a loop. Suggest: `# Each route goes through add_sensor so it gets the same checks as one added later.`
- **The "dataclass is frozen, so … stored through object" note** appears six times: `coordinates.py:222` and `:519`, and `measurement_models.py:141`, `:212`, `:278` and `:493`. Keep the first and delete the rest.
- **`motion.py:373`** only says what `i`, `j` are, so delete it. Keep `:375`, which ties `a`, `b` to the Notes formula.

### Guards (#17): done; small follow-ups

- **`SensorPose.__post_init__`** (`measurement_models.py:490`) builds a `StateLayout` only to check the frame and origin. Move those checks into a helper in `_validation.py`, and call it from both classes.
- **`_check_points`** is private to `coordinates.py` but imported at `measurement_models.py:57`. Move it to `_validation.py`.
- **`coordinates.py:442`:** give `rtol=1e-10, atol=1e-12` a named constant, with a comment saying why those values.
- **Repeated conditions.** Several guards repeat their condition in a comprehension to build the message: `tracker.py:201-205` and `:445-473`, and `measurement_models.py:200-205`. Move each into a `_check_*` helper that builds the offending list once and tests `if offenders:`. The reading view drops the helper whole, so this doesn't bring back #17's residue.
- **Full stops.** End every error message with one, as in style.md's example ("…diverges at R = 0."). `tracker.py`, `association.py` and `lifecycle.py` already do. `coordinates.py`, `measurement_models.py`, `ukf.py`, `motion.py`, `initiation.py`, `estimation.py` and `_validation.py` don't.
- **Optional:** `_check_filter_settings` (`tracker.py:542`) builds a `UKF` and throws it away. Lifting the `alpha`/`beta`/`kappa` guards from `UKF.__init__` into `_validation.py` lets `build_tracker` call them directly.

### Nits

- **`test_crossing_targets_keep_their_identities_across_two_sensors`** (`test_tracker.py:282`) also asserts statuses and source sensors. Split it, or rename it to say all three.
- **`CartesianMotion.transition` and `.process_noise`** each call `matrices()`, so it runs twice per predict.

### Where it isn't concise

- The "Why not the Joseph form" proof in `ukf.py:401-432` runs about 27 lines, which is a lot for an intern.
- The `cost` parameter in `tracker.py:117-131` carries about 15 lines of numbers.
- The reason for two-stage assignment is written out three times (twice in `tracker.py`, once in the README).
- Scenario-specific numbers sit inside docstrings (`motion.py:56-63`, `ukf.py:429-432`) and will go out of date.

### Content that was lost

- **Medium:** a worked M-of-N hit/miss example (PR #1 `management.py:166`). This matters more now, because the rule changed from a sliding window to "the first N scans".
- **Medium:** the initiators that start a track differently depending on the measurement type (`RoutedInitiator`) or never start one from a detection (`NoInitiation`) were dropped. The README's deferred list doesn't mention them.
- **Low:**
  - that one tracker should handle every sensor in a scenario;
  - "expect track fragmentation with the default settings";
  - the azimuth convention (clockwise from north);
  - the term "Mahalanobis distance";
  - the plan for a forward-scatter (FSR) model.

Leaving out IMM, the coordinated-turn model and the other deferred models was planned, so I didn't count it as a loss.

### Inaccurate statements

1. `measurement_models.py:746` says "A target 13 m **above** both sites", but the state is (3, 4, 12). The target is 12 m up and 13 m away. I checked this one myself.
2. `motion.py` refers to "(module Notes)" five times, but the module docstring has no Notes section; the heading is "Process noise". I checked this one too.
3. `tracker.py:18-19` and the README list confirm/delete after starting new tracks. In the code, it happens during the update, before new tracks start.
4. `lifecycle.py:18-19` ("A tentative track gets no such grace") reads as if the coasting time limit doesn't apply to tentative tracks. It does.
5. The `_check_covariance` docstring in `coordinates.py` describes work that actually happens in `_validation._is_semidefinite`.

### Smaller readability fixes

- Several terms are used without a definition: coast, chi-square gate, whitened innovation, principal interval, and the w in z = Hx + w.
- "Route" and "sensor" are both used for the same thing.
- Several docstrings open with Stone Soup class names before saying what the class does.
- `__init__.py:3` points to "spec/structure.md D2", which a newcomer won't understand.
