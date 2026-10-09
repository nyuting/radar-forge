# Test audit — refactor-002 §2 (pipelines, viz, tools, scripts)

Scope: `tests/pipelines/test_scenario_001.py`, `test_scenario_002.py`, `test_scenarios.py`,
`test_trajectories.py`; `tests/viz/`, `tests/tools/`, `tests/scripts/`; and the top-level
`tests/test_*.py`. Excluded, by the user's decision to defer them to tracker-001:
`tests/pipelines/test_tracking.py`, `tests/pipelines/test_scenario_003.py` and `tests/core/`.
Nothing in `src/`, `scripts/` or `tools/` was changed.

The user asked for a smaller suite: "del tests for being redundant … only keep the insightful and
necessary ones". That overrides the line in `spec/refactor-002-spec-first-audit.md` §2.4 that no test
is deleted for redundancy. Each collected item (a parametrize case counts as one) gets one class:

| Class | Meaning | Action |
| :--- | :--- | :--- |
| **Sound** | Names a plausible bug, and is the cheapest test that catches it | kept |
| **Redundant** | Catches no bug a kept test does not already catch | deleted, survivor named |
| **Weak** | Real bug, but the setup or assertion does not convincingly catch it | rewritten, or deleted if a kept test already catches it |
| **Wrong** | Asserts something untrue or unphysical | fixed, and the write-up says what it hid |

The bar for keeping a test is `docs/conventions/testing.md` §10.1 items 1–4. The §8 floor holds for
every public function these files test: a test against known values, every documented `Raises`, and
every `Notes` edge case. A deletion that left a `src/` line or branch unexecuted was put back (see
the result section).

All counts and coverage are measured against `origin/main` at `e92a9be`, after PR #20 renamed the
scenario TOML tables. Two files had a constraint. `test_scenarios.py` and `test_public_api.py` may
still be changed by the rest of the unmerged `docs/retire-refactor-001` branch. In those two, only
whole tests or whole parametrize cases were deleted, and no surviving test body was edited. The tests
PR #20 rewrote (`test_rejects_a_file_with_no_burst`, `test_rejects_a_file_missing_a_table`, all of
`TestReceiveEndTables`, `test_deleting_the_table_makes_the_same_file_monostatic`) were left alone.

For the viz, tools, scripts and top-level files, §2.1 (radar physics) and §2.2 (signal processing)
of the refactor-002 spec do not apply: these tests check structure, not radar behaviour. They are
judged on §2.3 and §10.1 alone.

## tests/pipelines/test_scenario_001.py

These are the acceptance tests for spec `scenario-001-xband.md` §6 (A1–A5, A7). The spec cites the
file, not individual tests.

**Physics (§2.1).** A light aircraft at 1500 m and 8–18 km, flying at 55–80 m/s. RCS is 10 dBsm,
Swerling 0. That is at the high end for a light aircraft at X-band (1–5 m² is more usual), but no
assertion here depends on absolute power: every test locates a peak. The waveforms are consistent.
S1 is FMCW with a 1 ms chirp at 1 kHz (duty 1). S2 is pulsed, 10 µs at 25 kHz (duty 0.25, which a
1 kW TWT or solid-state transmitter can do). S3 is two FMCW bursts at 5 and 6 kHz, both at duty 1.

**Signal processing (§2.2).** The 5-frame window at 2835 s folds everything it claims to. The
target closes at 54.7–55.0 m/s, which is 7.2 of S1's 15.30 m/s fold spans and past both of S3's
38.24 and 45.89 m/s. The range is 15.0–15.3 km, 2.5 unambiguous ranges of S2. Each premise guard
asserts this, so no folding test runs against a target that does not fold. Bin tolerances are one
bin, the spec's bound. Only peaks are located (no CFAR), so §2.2's threshold question does not
arise.

The module docstring still described the 2646 s window from before the sites moved to Durham
(50 m/s, 6.6 folds, 14 km). It now describes the 2835 s window the constant uses.

| Test or group | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `TestS1DopplerFolds::test_the_window_really_does_fold_the_doppler` | Sound | Premise guard for A2; also the only check that the truth label is not itself folded (mutation M1) | — |
| `TestS1DopplerFolds::test_the_peak_lands_at_the_true_range_and_the_folded_velocity` | Sound | A1 + A2 | — |
| `TestS1DopplerFolds::test_the_peak_velocity_is_nothing_like_the_truth` | Redundant | If the truth folds at least 4 times and the peak is within one bin of the folded value, the peak is far from the truth | the two S1 tests above |
| `TestS2RangeFolds::test_the_window_really_does_fold_the_range` | Sound | Premise guard for A3 | — |
| `TestS2RangeFolds::test_the_peak_lands_at_the_folded_range_and_the_true_velocity` | Sound | A3 + A4, and A7 for S2: velocity to one bin fixes its sign | — |
| `TestS2RangeFolds::test_the_peak_range_is_nothing_like_the_truth` | Redundant | Same argument as S1, for range | the two S2 tests above |
| `TestS3TheAmbiguityIsResolved::test_both_bursts_really_do_fold` | Sound | Premise guard for A5 | — |
| `TestS3TheAmbiguityIsResolved::test_each_burst_alone_reports_the_wrong_velocity` | Redundant | Follows from the premise guard: a burst whose truth folds cannot report it | `test_both_bursts_really_do_fold`, `test_the_pair_recovers_the_true_unfolded_velocity` |
| `TestS3TheAmbiguityIsResolved::test_the_pair_recovers_the_true_unfolded_velocity` | Sound | A5, and A7 for S3 | — |
| `TestS3TheAmbiguityIsResolved::test_both_bursts_agree_on_the_range` | Redundant | Each burst's range peak against the truth is what the S3 default-window case checks | `test_both_s3_bursts_find_the_target_in_the_default_window` |
| `TestTheShippedDefaultWindowRuns::…[s1]`, `[s2]` | Redundant | The same range check as A1 and A3, at another window that exercises no other code | A1 and A3 tests above |
| `TestTheShippedDefaultWindowRuns::…[s3]` | Sound | The only per-burst range check for S3, whose bursts have different sample counts (800 and 667). Now a plain test, `test_both_s3_bursts_find_the_target_in_the_default_window`, with a docstring naming that bug (mutation M2) | — |

A7 ("closing targets close") cannot hold for S1, because a folded peak's sign is not the truth's.
It is enforced where it can be: S2's unfolded velocity and S3's unfolded pair, each to one bin.

## tests/pipelines/test_scenario_002.py

These are the acceptance tests for spec `scenario-002-bistatic.md` §6 (A1–A6).

**Physics (§2.1).** The scenario uses the Raleigh-Durham illuminator and the Duke receiver, 19.6 km
apart, with the same waveform as scenario-001 S1. B2's 2.8 GHz carrier is a realistic S-band ATC
illuminator.

**Signal processing (§2.2).** Over B1's default window β runs 119.2–120.6°. The bisector rate
reaches 33.6 m/s, which is 4.4 of X-band's intervals and 1.3 of S-band's, so both variants fold.

| Test or group | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `TestTheBaselineCollapsesToScenario001::test_the_truth_labels_agree` | Redundant | The peak comparison below fails on any label error that reaches the IQ. The trajectory-layer collapse checks range to 1 m and velocity to 1e-6 m/s | `test_the_range_doppler_peak_agrees`; `test_trajectories.py::…::test_a_collapsed_baseline_reproduces_the_monostatic_track` |
| `TestTheBaselineCollapsesToScenario001::test_the_range_doppler_peak_agrees` | Sound | A1, the load-bearing test (mutation M3) | — |
| `TestTheRangeAxisCarriesTheMeanRange::test_the_peak_lands_at_the_bistatic_mean_range[b1]`, `[b2]` | Sound | A2. Both cases are kept: the spec lists B1 and B2, and the FMCW beat frequency carries a Doppler term that scales with the carrier | — |
| `TestTheRangeAxisCarriesTheMeanRange::test_the_mean_range_is_neither_of_the_two_ranges[b1]` | Sound | Premise guard for A2 | — |
| `…::test_the_mean_range_is_neither_of_the_two_ranges[b2]` | Redundant | Same sites and same trajectory as B1, so the same geometry | the `[b1]` case |
| `TestTheRangeAxisCarriesTheMeanRange::test_the_window_is_genuinely_bistatic` | Weak | Encoded A6 as β > 90° against the spec's 109.9–129.3°. Now asserts the spec's bounds (finding F3) | — |
| `TestTheVelocityAxisCarriesTheBisectorRate::test_the_peak_lands_at_the_folded_bisector_rate[b1]`, `[b2]` | Sound | A3. The carrier sets the Doppler axis, so the two cases are different regimes | — |
| `TestTheVelocityAxisCarriesTheBisectorRate::test_the_x_band_variant_folds_in_every_frame` | Redundant | `0 < n_s < n_x` already shows that both variants fold | `test_the_s_band_variant_folds_in_fewer_frames` |
| `TestTheVelocityAxisCarriesTheBisectorRate::test_the_s_band_variant_folds_in_fewer_frames` | Sound | The variant pair's lesson, and now the stated premise guard for both A3 cases | — |
| `TestTheVelocityAxisCarriesTheBisectorRate::test_the_longer_wavelength_buys_unambiguous_velocity` | Redundant | If S-band's v_ua were not the larger, `n_s < n_x` would fail. The formula is pinned in core | `test_the_s_band_variant_folds_in_fewer_frames`; `test_scenarios.py::…::test_the_two_variants_differ_only_in_the_carrier`; `tests/core/test_radar.py` |
| `TestRangeResolutionDegradesWithTheBistaticAngle::test_a_separation_in_space_shrinks_in_mean_range` | Wrong | A4 placed its target at 1.40 N 103.88 E, 12 086 km from the sites, where β = 0.09°. Fixed (finding F1, mutation M11) | — |

## tests/pipelines/test_scenarios.py

These test the scenario loader, the frame generator and the range-Doppler product. A6 of
scenario-001 cites this file. Deletions here are whole tests or whole cases only.

| Test or group | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `TestLoadScenario::test_every_shipped_scenario_loads[s1,s2,s3]` | Redundant | Loads and type-checks the same three files as the monostatic regression guard | `TestBistaticScenarios::test_a_scenario_without_the_table_is_still_monostatic[s1,s2,s3]` |
| `TestLoadScenario::test_the_trajectory_path_resolves_to_a_real_file[s1]` | Sound | Relative path resolved against the TOML | — |
| `…::test_the_trajectory_path_resolves_to_a_real_file[s2]`, `[s3]` | Redundant | All three TOMLs hold the same `../data/flight_coordinates.csv` | the `[s1]` case |
| `TestLoadScenario::test_every_variant_shares_the_specified_range_resolution[s1,s2,s3]` | Sound | The only per-file check of `bandwidth_hz` | — |
| `TestLoadScenario::test_the_default_window_is_the_specified_two_minutes` | Sound | Spec default | — |
| `TestLoadScenario::test_rejects_an_unknown_key`, `…_with_no_burst`, `…_missing_a_table`, `test_bad_physics_is_reported_by_the_radar_not_the_loader` | Sound | Documented `Raises` | — |
| `TestSpecifiedAmbiguities::test_s1_…`, `test_s2_…`, `test_s3_bursts_both_cover_the_track_in_range`, `test_s3_bursts_are_in_the_coprime_five_to_six_ratio` | Sound | A6: each limit re-derived from the loaded radar | — |
| `TestSpecifiedAmbiguities::test_s2_is_the_exact_mirror_of_s1` | Redundant | Implied by the two pinned values (37.47 km > 5.996 km, 7.65 m/s < 191 m/s) | `test_s1_…`, `test_s2_…` |
| `TestSpecifiedAmbiguities::test_s3_burst_b_runs_at_exactly_one_over_six_thousand_seconds` | Sound | Live regression (duty cycle 1.0002 from the rounded spec value) | — |
| `TestSpecifiedAmbiguities::test_cube_dimensions_match_the_specified_tables[s1,s2,s3]` | Sound | Per-file config check | — |
| `TestIterateFrames::test_is_a_generator_not_a_list`, `test_yields_one_frame_per_frame_time`, `test_each_burst_gets_its_own_cube`, `test_the_truth_is_the_real_geometry`, `test_a_run_replays_bit_for_bit`, `test_consecutive_frames_carry_different_noise`, `test_rejects_a_window_outside_the_track` | Sound | Each names its own bug | — |
| `TestIterateFrames::test_the_truth_is_never_folded` | Weak | Bound of 0.9·v_ua and a stale comment (finding F4). Deleted, not fixed, because the survivor checks the same thing with a bound no folded truth can pass | `test_scenario_001.py::TestS1DopplerFolds::test_the_window_really_does_fold_the_doppler` |
| `TestScenarioValidation::*` except `[-8]` | Sound | Documented `Raises` | — |
| `TestScenarioValidation::test_rejects_a_burst_with_no_pulses[-8]` | Redundant | 0 catches a `< 0` guard; −8 catches nothing 0 does not | the `[0]` case |
| `TestDegenerateWindows::test_a_single_frame_with_no_room_to_pad_is_rejected` | Sound | Edge case named in the docstring | — |
| `TestReceiveEndTables::*` (4) | Sound | Rewritten by PR #20 and left alone | — |
| `TestBurstRangeDoppler::test_fmcw_axes_match_the_map` | Redundant | Shape only. The synthetic-target test indexes both axes at the peak | `test_a_synthetic_fmcw_target_lands_at_its_true_range_and_velocity` |
| `TestBurstRangeDoppler::test_the_range_axis_is_unshifted_and_the_velocity_axis_is_centred`, `test_the_pulsed_range_axis_spans_one_unambiguous_range`, `test_a_synthetic_fmcw_target_lands_at_its_true_range_and_velocity`, `test_a_synthetic_pulsed_target_folds_into_the_unambiguous_range` | Sound | fftshift asymmetry, axis span, closed-form placement, range folding | — |
| `TestBurstRangeDoppler::test_a_pulsed_target_inside_the_unambiguous_range_does_not_fold` | Redundant | A wrong group-delay trim moves both peaks by the same number of bins (mutation M9) | `test_a_synthetic_pulsed_target_folds_into_the_unambiguous_range` |
| `TestVelocityIsIndependentOfTheWindow::test_a_frame_reports_the_same_velocity_whatever_window_contains_it` | Sound | Padding, through its observable effect | — |
| `TestVelocityIsIndependentOfTheWindow::test_a_single_frame_window_still_has_a_velocity` | Redundant | The test above already compares a one-frame window's velocity exactly | the test above |
| `TestBistaticScenarios::test_every_shipped_bistatic_scenario_loads[b1]` | Redundant | The baseline test type-checks B1 as `BistaticRadar` | `test_the_baseline_is_the_specified_airport_to_receiver_distance` |
| `TestBistaticScenarios::test_every_shipped_bistatic_scenario_loads[b2]` | Sound | The only type check of B2 | — |
| `TestBistaticScenarios::` the other seven | Sound | A5, the carrier-only difference, the table switch, the new table's unknown key, bistatic and monostatic frame fields | — |
| `TestScenario003StateModelParameters::*` (2) | Sound | Tracking configuration; left for tracker-001, which retires `sigma_accel_mps2` | — |

## tests/pipelines/test_trajectories.py

These test the CSV loader, resampling, and the monostatic and bistatic radar-frame transforms.

**Physics (§2.1).** Synthetic tracks at the Duke site: straight radial at 80 m/s, a 10 km circle at
105 m/s, and fixed points. All are realisable, and each tolerance has a comment deriving it from
meridian convergence at 36° N.

| Test or group | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `TestLoadFlightCsv::test_reads_every_row_of_the_golden_excerpt` | Redundant | A dropped row fails the two-row files' exact `time_s` checks | `test_a_time_utc_file_keeps_its_epoch`, `test_a_time_s_file_has_no_epoch` |
| `TestLoadFlightCsv::test_time_is_measured_from_the_first_fix` | Redundant | The golden file starts at 0, so it cannot see a missing subtraction. The `time_s` file starting at 10 s can | `test_a_time_s_file_has_no_epoch` |
| `TestLoadFlightCsv::test_the_second_fix_is_eight_seconds_in` | Redundant | Pins the golden data, not the parser | `test_a_time_s_file_has_no_epoch` |
| `TestLoadFlightCsv::test_timestamps_are_strictly_increasing` | Redundant | `Trajectory.__post_init__` raises otherwise, so a loaded trajectory cannot fail it | `test_rejects_out_of_order_timestamps`, `TestTrajectoryValidation::test_rejects_repeated_times` |
| `TestLoadFlightCsv::test_rejects_a_file_with_no_time_column` | Redundant | Same branch and message as both columns present | `test_rejects_both_time_columns` |
| `TestLoadFlightCsv::` the other 13 | Sound | Each documented `Raises` and `Notes` case (CRLF, legacy header, single `target_id`, constant altitude, altitude column) and both time formats | — |
| `TestResample::test_lands_on_the_requested_grid`, `test_reproduces_the_original_fixes_exactly` | Redundant | The midpoint test implies the grid. Identity at the nodes is `np.interp`'s property | `test_the_midpoint_of_a_gap_is_the_mean_of_its_ends` |
| `TestResample::` the other 4 | Sound | Interpolation, epoch carried, and both ends of the extrapolation guard | — |
| `TestToRadarFrame::test_a_target_due_north_reads_zero_azimuth` | Redundant | Due east catches the same site/target swap (270°) and the from-east convention (0°), and catches counter-clockwise, which due north cannot | `test_a_target_due_east_reads_ninety_degrees` |
| `TestToRadarFrame::test_a_closing_target_reports_positive_velocity` | Redundant | The receding test pins sign and magnitude | `test_a_receding_target_reports_negative_velocity` |
| `TestToRadarFrame::test_shapes_follow_the_trajectory` | Redundant | Shape only. `n_frames` is now asserted in `test_a_target_over_the_radar_site_is_at_its_own_height` | value tests |
| `TestToRadarFrame::` the other 4 | Sound | Height null, east, receding, circling null | — |
| `TestTrajectoryValidation::*` (5) | Sound | One branch each. Repeated times catches a `>= 0` bug that out-of-order input cannot | — |
| `TestToBistaticRadarFrame::test_a_collapsed_baseline_reproduces_the_monostatic_track`, `test_the_mean_range_is_the_half_sum_of_the_two_ranges`, `test_the_two_ranges_genuinely_differ` | Sound | Degeneracy, D6, site mix-up (mutation M5) | — |
| `TestToBistaticRadarFrame::test_the_departure_and_arrival_angles_differ` | Redundant | Ranges and angles come from one call per site, so a site mix-up breaks both together | `test_the_two_ranges_genuinely_differ` |
| `TestToBistaticRadarFrame::test_a_stationary_target_has_no_bisector_rate` | Redundant | Any rate built from range gradients is zero here, including a sign-flipped one (mutation M4a) | the rewritten closed-form test |
| `TestToBistaticRadarFrame::test_the_bisector_rate_is_positive_when_the_path_shortens` | Wrong | Asserted over an empty set. Now `test_a_target_flying_at_the_receiver_closes_at_v_cos_squared_half_beta` (finding F2) | — |
| `TestToBistaticRadarFrame::test_n_frames_counts_the_grid` | Redundant | Now asserted in the closed-form test | the closed-form test |

## tests/viz/test_plotting.py

Structural tests; §2.1–2.2 do not apply.

| Test or group | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `TestMagnitudeDb::test_the_peak_is_zero_decibels_by_default` | Redundant | The −6 dB test uses the default reference: a mean reference (0.75) would give −3.5 dB there | `test_halving_the_amplitude_costs_six_decibels` |
| `TestMagnitudeDb::test_halving_the_amplitude_costs_six_decibels`, `test_a_fixed_reference_holds_the_scale`, `test_uses_complex_magnitude` | Sound | The factor of 20 (`Notes`), the fixed reference, \|z\| not Re z | — |
| `TestMagnitudeDb::test_a_zero_cell_is_clamped_to_the_floor` | Redundant | The all-zero test applies the same floor, and also takes the no-peak branch (mutation M6) | `test_an_all_zero_array_is_all_floor` |
| `TestMagnitudeDb::test_an_all_zero_array_is_all_floor` | Sound | Floor and the reference fallback | — |
| `TestMagnitudeDb::test_preserves_shape` | Redundant | `np.maximum`'s shape is NumPy's behaviour | — |
| `TestMagnitudeDb::test_rejects_a_non_positive_reference[0.0]` | Sound | Now `test_rejects_a_zero_reference`: zero catches a `< 0` guard | — |
| `…[-1.0]` | Redundant | Catches nothing zero does not | the zero case |
| `TestRequirePyplot::test_names_the_extra_when_matplotlib_is_missing` | Redundant | Matching "uv sync --extra viz" also matches "viz" | `test_the_message_gives_a_command_to_run` |
| `TestRequirePyplot::test_the_message_gives_a_command_to_run` | Sound | Documented `Raises` | — |
| `TestRequirePyplot::test_returns_pyplot_when_the_extra_is_present` | Redundant | Both scopes call `require_pyplot` first, and every scope test needs it to return pyplot | every `test_rd_map.py` and `test_track_plot.py` rendering test |
| `TestSaveFigure::test_writes_the_file_and_closes_the_figure` | Sound | The figure-leak contract | — |

## tests/viz/test_rd_map.py

Structural tests; §2.1–2.2 do not apply. Every `_mark_truth` and `_mark_tracking` branch has a
surviving test.

| Test or group | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `TestRenderRangeDoppler::test_returns_a_figure` | Redundant | Every other test reads `figure.axes` | any rendering test |
| `TestRenderRangeDoppler::test_range_is_on_the_horizontal_axis_in_kilometres`, `test_velocity_is_on_the_vertical_axis_and_says_which_sign_closes` | Sound | Axis assignment, km scaling, limits from the axes | — |
| `TestRenderRangeDoppler::` on-scale truth, folded marker, off-scale up, off-scale range, no marker without truth | Sound | One drawing branch each | — |
| `TestRenderRangeDoppler::test_a_target_below_the_velocity_axis_points_down` | Weak | Asserted only that a "v" marker exists. Now pins it to the lower edge, which catches swapped edges | — |
| `TestRenderRangeDoppler::test_rejects_axes_that_do_not_match_the_map` | Redundant | Same branch and message as the transposed map (mutation M10) | `test_rejects_a_transposed_map` |
| `TestRenderRangeDoppler::test_rejects_a_transposed_map`, `test_rejects_a_one_dimensional_map` | Sound | Documented `Raises` | — |
| `…::test_rejects_a_non_positive_dynamic_range[0.0]` | Sound | Now `test_rejects_a_zero_dynamic_range` | — |
| `…[-10.0]` | Redundant | Catches nothing zero does not | the zero case |
| `TestBistaticLabelling::test_the_default_labels_are_monostatic` | Redundant | The axis tests run the monostatic branch and check its labels | the two axis tests |
| `TestBistaticLabelling::test_the_bistatic_flag_renames_both_axes` | Sound | D6 labels | — |
| `TestBistaticLabelling::test_the_flag_changes_nothing_but_the_labels` | Redundant | Limits come from the axes on both branches. The axis tests pin them | the two axis tests |
| `TestTrackingOverlays::test_the_overlays_are_absent_by_default` | Redundant | `get_lines() == []` with no overlays is stronger | `test_no_marker_without_a_truth` |
| `TestTrackingOverlays::` the other 3 | Sound | Detections, estimate with gate, gate without estimate | — |
| `TestRenderRangeDopplerEdgeCases::test_a_title_is_placed_on_the_axes`, `test_a_truth_left_of_the_range_axis_is_pinned_to_that_edge` | Sound | Title branch, the fourth edge | — |
| `TestRenderRangeDopplerEdgeCases::test_a_track_estimate_without_a_gate_is_still_drawn` | Weak | Asserted only that a legend exists. Now asserts the estimate is drawn and the gate is not | — |

## tests/viz/test_track_plot.py

Structural tests; §2.1–2.2 do not apply.

| Test or group | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `TestRendering::test_truth_alone_renders` | Redundant | The axis test renders truth alone and checks its values | `test_the_truth_is_drawn_in_seconds_and_kilometres` |
| `TestRendering::test_the_y_axis_is_kilometres` | Sound | Now `test_the_truth_is_drawn_in_seconds_and_kilometres` and also checks the x data | — |
| `TestRendering::test_the_x_axis_is_seconds` | Weak | The fixture's time equalled the frame index, so plotting the index would have passed. The fixture now runs from 100 s in 2 s steps, and the check moved into the test above (finding F7) | `test_the_truth_is_drawn_in_seconds_and_kilometres` |
| `TestRendering::test_detections_are_drawn_as_a_scatter`, `test_the_track_is_a_second_line`, `test_the_current_frame_is_marked` | Weak | Asserted only that something was drawn. Each now pins its data (scatter offsets in km, track y data, the vertical's x) | — |
| `TestRendering::test_the_uncertainty_band_follows_the_track_not_the_truth`, `test_the_title_is_used` | Sound | Band placement, title branch | — |
| `TestValidation::*` (4) | Sound | Both clauses of the shape guard, and each `_require_pair` call site | — |
| `test_off_scale_detections_are_counted_in_a_corner_note` | Sound | Off-scale branch | — |
| `TestRangeRatePanel::test_a_truth_range_rate_adds_a_panel_below_sharing_time`, `test_rejects_a_track_range_rate_without_the_truth` | Sound | Second panel, documented `Raises` | — |
| `TestRangeRatePanel::test_without_a_truth_range_rate_there_is_one_panel` | Redundant | `(axes,) = figure.axes` asserts one panel in every rendering test | `test_the_truth_is_drawn_in_seconds_and_kilometres` |
| `TestFoldedRange::test_the_axis_spans_one_period_and_says_it_is_modulo`, `test_a_line_is_broken_where_it_wraps[period]` | Sound | Modulo axis, wrap gaps | — |
| `TestFoldedRange::test_a_line_is_broken_where_it_wraps[None]` | Redundant | The axis test checks the whole truth line's y data with no period | `test_the_truth_is_drawn_in_seconds_and_kilometres` |

## tests/tools/test_generate_reading_view.py

Structural tests; §2.1–2.2 do not apply.

| Test or group | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `test_sample_output_is_valid_python` | Redundant | The `sample_tree` fixture runs `ast.parse`; every sample-tree test errors if it fails | every `sample_tree` test |
| The eight sample transform tests, `test_non_guard_raise_survives`, `test_emptied_body_gets_a_pass` | Sound | One transform or edge case each | — |
| `test_generate_runs_clean_over_the_package`, `test_generated_package_is_valid_stripped_python`, `test_generated_package_keeps_imports_and_returns`, `test_index_links_every_generated_file`, `test_every_module_has_a_page_showing_its_code`, `test_module_page_escapes_html_and_links_home` | Sound | Package-wide properties over constructs the sample lacks; one shared generation | — |
| `test_main_writes_the_tree` | Weak | Took 0.89 s, over the 0.5 s unmarked budget, regenerating the package to check argument wiring. Now runs on a one-module tree and checks the module came through | — |
| `test_main_rejects_a_missing_source` | Sound | Exit status | — |

## tests/scripts/test_run_scenario.py

Structural tests; §2.1–2.2 do not apply.

| Test or group | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `test_columns_are_the_specs_in_the_specs_order[3]` | Sound | Spec drift guard against data-001 | — |
| `TestCell::test_none_is_an_empty_cell`, `test_booleans_are_zero_or_one[True,False]` | Sound | data-001 §5 | — |
| `TestCell::test_a_float_reads_back_exactly[0.1]`, `[1/3]` | Redundant | 0.1 reads back under `%g` and 1/3 under `%.16g`, so neither catches a format the 17-digit case misses | `[17-digit]` |
| `TestCell::test_a_float_reads_back_exactly[21.635652855125496]`, `[1e-300]`, `[-0.0]` | Sound | 17 significant digits, exponent form, sign and falsy zero | — |
| `TestCell::test_a_float_is_not_padded_to_a_fixed_precision` | Weak | Used 0.5, which every format except `%f` writes as "0.5" (finding F5, mutation M8) | — |
| `test_metadata_tracking_keeps_every_key_of_the_toml_table[scenario_003_tracking]`, `[…ukf_fmcw_dual_prf]` | Sound | One per estimator path | — |
| `…[scenario_003_tracking_dual_prf]` | Redundant | Its `[tracking]` table has exactly the keys of S1's | the `[scenario_003_tracking]` case |
| `test_metadata_says_which_settings_the_estimator_did_not_read[2]` | Sound | KF and UKF paths | — |
| `test_a_ukf_run_writes_the_data_001_files`, `test_a_trajectory_with_an_epoch_writes_time_utc` | Sound | The one end-to-end run, and the `time_utc` branch | — |

## tests/test_docstrings.py, tests/test_import.py, tests/test_public_api.py

Package-wide guards; §2.1–2.2 do not apply.

| Test or group | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `test_docstrings.py::test_every_docstring_example_evaluates_to_what_it_claims[29]` | Sound | pytest does not run doctests (`addopts` has no `--doctest-modules`) | — |
| `test_docstrings.py::test_the_prediction_example_respects_the_closing_sign_convention` | Sound | Live regression | — |
| `test_docstrings.py::test_every_public_name_is_documented[29]` | Sound | Export docstrings; also fails on a stale `__all__` entry (mutation M7) | — |
| `test_import.py::test_package_imports_and_has_version`, `test_the_public_api_is_what_it_re_exports` | Sound | The top-level package is not in `walk_packages`' list, so `test_public_api.py` does not cover it | — |
| `test_import.py::test_importing_the_package_does_not_pull_in_the_viz_extra` | Sound | A8. Now also reads `radar_forge.pipelines` in the same subprocess, before the no-viz asserts | — |
| `test_import.py::test_the_pipelines_subpackage_is_available_without_extras` | Redundant | A second interpreter (0.7 s) for one attribute; that check moved into the test above | `test_importing_the_package_does_not_pull_in_the_viz_extra` |
| `test_public_api.py::test_every_module_declares_its_public_surface[28]`, `test_no_name_is_exported_twice[29]`, `test_no_private_name_is_exported[29]`, `test_every_public_return_type_is_exported[29]` | Sound | One API rule each; nothing else checks them | — |
| `test_public_api.py::test_every_exported_name_exists[29]` | Redundant | Same modules and names. `getattr` in the docstring test fails on the same stale entry | `test_docstrings.py::test_every_public_name_is_documented` |
| `test_public_api.py::test_a_python_list_is_accepted_and_a_float64_array_returned[range_resolution_m, sweep_rate_hzps, range_from_beat]` | Sound | The three contract shapes the case list's own comment names | — |
| `…[dbsm_to_m2]`, `[m2_to_dbsm]` | Redundant | A single `array_like` argument, the same shape as `range_resolution_m` | the `[range_resolution_m]` case |
| `test_public_api.py::test_a_list_and_an_array_give_bit_identical_results[5]` | Redundant | Checks that `np.asarray` of a list equals the array: NumPy's behaviour (§10.1 item 3) | `test_a_python_list_is_accepted_and_a_float64_array_returned` |
| `test_public_api.py::test_a_scalar_argument_returns_a_zero_dimensional_float64`, `test_a_taper_is_returned_as_a_fresh_writable_array`, `test_every_submodule_is_exported[2]`, `test_every_exported_submodule_is_imported_explicitly[2]` | Sound | 0-d corner, aliasing, submodule surface | — |

## Findings — pipelines, viz, tools, scripts

**F1 — Wrong: A4 tested a geometry where the bistatic angle is nearly zero.**
`test_a_separation_in_space_shrinks_in_mean_range` put its targets at 1.40 N, 103.88 E. That is the
pre-Durham site, now 12 086 km from both Durham sites and below their horizon. There β = 0.09°, so
the asserted cos(β/2) is 0.9999997, and a monostatic range mapping passes (mutation M11: β forced
to 0, the old test stays green). The test now uses the track's own position at the start of B1's
default window (35.9362 N, −78.9338 E, 1500 m). That point is 14.8 km from the transmitter and
7.5 km from the receiver, with β = 119.2° and cos(β/2) = 0.506. The step along the bisector is now
20 m instead of 200 m. The first-order term is d·cos(β/2). Iso-range curvature adds a second-order
error that grows linearly with d, measured at 7.4 × 10⁻⁵ per metre: 1.5 × 10⁻³ at 20 m, 1.5 × 10⁻²
at 200 m. Converting back to geodetic on the semi-major axis, rather than the two radii of
curvature, adds 3 × 10⁻⁵. So rtol tightens from 5 × 10⁻³ to 2 × 10⁻³. The final `< separation_m`
assertion was dropped, because the new bound implies it.

**F2 — Wrong: the bisector-sign test asserted over an empty set.**
`test_the_bisector_rate_is_positive_when_the_path_shortens` kept only the frames where the total
path shortened. Its target flies due north, away from both sites, so no frame qualifies and
`np.all([])` is `True`. A flipped sign and a rate from one range both passed it (mutations M4a,
M4b). It is now `test_a_target_flying_at_the_receiver_closes_at_v_cos_squared_half_beta`. The target
flies straight at the receiver at v = 80 m/s. The receive range then closes at v and the transmit
range at v·cos β, so v_b = v(1 + cos β)/2 = v·cos²(β/2) (Willis, *Bistatic Radar*, §6.1). The test
first asserts that every interval shortens the path, then checks v·cos²(β/2) at rtol 10⁻³. That
rtol is the same flat-track allowance as the monostatic sign test, and the measured worst case is
3.4 × 10⁻⁴, at the one-sided end differences. A missing half doubles the answer. A rate from one
range alone is 7 % or 15 % off at the β of 30–36° here. The collapsed-baseline test cannot see that
last bug (M4b: it stays green).

**F3 — Weak: A6 was encoded as β > 90°.** The spec bound is 109.9–129.3°, and the test now asserts
that. Taking the supplement (50.7–70.1°) or measuring at the wrong vertex fails it.

**F4 — Weak: `test_the_truth_is_never_folded` (test_scenarios.py) had a bound inside the interval.**
It asserted |v| > 0.9·v_ua to show the truth label is not folded into ±v_ua. A folded truth whose
value lands in the outer tenth of the interval passes that. Its comment, that the track "happens to
sit near the edge here", is from before the sites moved: the default window's truth is now
−28.1 m/s, 3.7·v_ua. At this window it did catch M1, because the folded value is 2.5 m/s. It was
deleted, not fixed, because the coordination rule allows only whole-test deletions in that file and
the survivor (|v| > 4·v_ua in every frame) checks the same thing with a bound no folded truth can
pass.

**F5 — Weak: the shortest-float check could not tell `%.17g` from `repr`.**
`test_a_float_is_not_padded_to_a_fixed_precision` used 0.5, which `%.17g` also writes as "0.5".
Every round-trip case reads back under `%.17g` by construction, so a `%.17g` writer passed the whole
class (mutation M8: old `TestCell` green, new red). It now uses 0.1, which `%.17g` writes as
0.10000000000000001.

**F6 — Weak: two plot assertions checked only that something was drawn.** In `test_rd_map.py`, the
down-pointing marker and the estimate-without-gate test. In `test_track_plot.py`, the detection
scatter, the track line and the current-frame line. Each now asserts the drawn data.

**F7 — Weak: the track-plot fixture's time equalled the frame index.** With `time_s = arange(20)`, a
scope that plotted the index on the x axis would have passed the x-axis test. The fixture now runs
from 100 s in 2 s steps.

**F8 — Weak (budget): `test_main_writes_the_tree` ran 0.89 s unmarked.** It regenerated the whole
package. It now runs on a one-module tree in a few milliseconds.

**F9 — Not fixed, reported: `resample` has an undocumented `Raises`.** `resample(trajectory, [t])`
with a single time raises "a trajectory needs at least two fixes", from `Trajectory.__post_init__`.
The docstring says `times_s` has shape `(n_frames,)` and lists only the outside-the-span `Raises`.
`iterate_frames` pads the grid, so no shipped path hits it. The fix belongs in `src/` (document it,
or let `resample` return a one-fix track), which is outside this stream.

**F10 — Gap, reported: no §10.8 link-budget test for scenario 001.** `docs/conventions/testing.md`
§10.8 asks each pipeline for a link-budget test and one Pd point. Scenario 001's tests locate peaks
only. Scenario 002 does not assert power, by its own D8. Nothing here was deleted that did either.

**Note on `spec/scenario-002-bistatic.md` §5.** That section says `test_scenario_001.py` is "read,
never edited". It is a build-order rule for the scenario-002 slice: the monostatic path must not move
while the bistatic one is built. This audit edits the file under refactor-002's remit. Every A1–A5
check is still in the file and green.

## Result — pipelines, viz, tools, scripts

### Counts

Collected pytest items at `e92a9be` and at this branch's head; a parametrize case counts as one.

| File | Before | After | Deleted |
| :--- | ---: | ---: | ---: |
| `tests/pipelines/test_scenario_001.py` | 13 | 7 | 6 |
| `tests/pipelines/test_scenario_002.py` | 13 | 9 | 4 |
| `tests/pipelines/test_scenarios.py` | 63 | 51 | 12 |
| `tests/pipelines/test_trajectories.py` | 43 | 30 | 13 |
| `tests/viz/test_plotting.py` | 13 | 7 | 6 |
| `tests/viz/test_rd_map.py` | 24 | 18 | 6 |
| `tests/viz/test_track_plot.py` | 19 | 15 | 4 |
| `tests/tools/test_generate_reading_view.py` | 18 | 17 | 1 |
| `tests/scripts/test_run_scenario.py` | 19 | 16 | 3 |
| `tests/test_docstrings.py` | 59 | 59 | 0 |
| `tests/test_import.py` | 4 | 3 | 1 |
| `tests/test_public_api.py` | 160 | 124 | 36 |
| **In scope** | **448** | **356** | **92** |
| Whole suite | 1554 | 1462 | 92 |

By class, over the 448 items before: 346 Sound, 90 Redundant (all deleted), 10 Weak (8 rewritten,
2 deleted) and 2 Wrong (both fixed).

### Coverage

`src/radar_forge` line and branch coverage, before and after, compared file by file on missing lines
and missing branches:

| | Before | After |
| :--- | ---: | ---: |
| Statements covered | 3386 / 3419 | 3386 / 3419 |
| Branches covered | 818 / 838 | 818 / 838 |
| Newly uncovered lines | — | 0 |
| Newly uncovered branches | — | 0 |

The first comparison after pruning found two newly uncovered lines: `TargetTrack.n_frames` and
`BistaticTargetTrack.n_frames` (`pipelines/trajectories.py:157, 454`). They are now asserted against
known frame counts in two kept transform tests, instead of bringing back the shape-only tests.

### Mutation spot-check

Each row breaks one behaviour in `src/` or `scripts/`, runs the named survivor, and for most rows
also runs the deleted or replaced test from `e92a9be`. The file was restored with `git checkout`
after each row. "Red" means the test failed.

| # | Mutation | Survivor | Survivor result | Deleted or old test | Old result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| M1 | Truth label folded into ±v_ua (`iterate_frames`) | `test_scenario_001::…::test_the_window_really_does_fold_the_doppler` | red | `test_scenarios::…::test_the_truth_is_never_folded` | red |
| M2 | Range axis 1 % long for the 667-sample burst only | `test_scenario_001::…::test_both_s3_bursts_find_the_target_in_the_default_window` | red | `test_both_bursts_agree_on_the_range` | — |
| M3 | Bisector rate without its ½ | `test_scenario_002::…::test_the_range_doppler_peak_agrees`; trajectories closed-form test | red, red | `test_scenario_002::…::test_the_truth_labels_agree` | red |
| M4a | Bisector rate sign flipped | trajectories closed-form test | red | old shortening test; stationary null | **green, green** |
| M4b | Bisector rate from the receive range alone | trajectories closed-form test; collapsed-baseline test | red, green | old shortening test | **green** |
| M5 | Receive leg measured from the transmitter site | `test_the_two_ranges_genuinely_differ` | red | `test_the_departure_and_arrival_angles_differ` | red |
| M6 | dB floor removed | `test_an_all_zero_array_is_all_floor` | red | `test_a_zero_cell_is_clamped_to_the_floor` | — |
| M7 | Stale name added to `trajectories.__all__` | `test_docstrings::test_every_public_name_is_documented` | red | `test_public_api::test_every_exported_name_exists` | — |
| M8 | CSV floats written with `%.17g` | `TestCell` | red | old `TestCell` | **green** |
| M9 | Pulsed group-delay trim three samples late | `test_a_synthetic_pulsed_target_folds_into_the_unambiguous_range` | red | `test_a_pulsed_target_inside_the_unambiguous_range_does_not_fold` | red |
| M10 | `rd_map` shape guard removed | `test_rejects_a_transposed_map` | red | `test_rejects_axes_that_do_not_match_the_map` | — |
| M11 | Bistatic angle forced to 0 (a monostatic mapping) | A4, as fixed | red | A4, as it was | **green** |

Every survivor went red when its behaviour broke. The bold "green" results are the old tests that
did not notice the bug they were written for: F2 (M4a, M4b), F5 (M8) and F1 (M11). Where a deleted
test went red too (M1, M3, M5, M9), it was redundant with its survivor, not a stronger test.
