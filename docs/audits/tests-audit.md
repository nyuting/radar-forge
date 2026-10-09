# Test audit — refactor-002 §2

Two parts, audited in parallel against the same rules. **Part 1** covers the eleven `tests/core/`
modules that test the signal chain. **[Part 2](#part-2--pipelines-viz-tools-scripts-and-top-level)**
covers `tests/pipelines/` (less tracking), `tests/viz/`, `tests/tools/`, `tests/scripts/` and the
top-level tests. Part 1 numbers its findings W1–W20, Part 2 F1–F10.

| Part | Before | After | Deleted | Weak | Wrong |
| :--- | --: | --: | --: | --: | --: |
| [1, `tests/core/`](#result--testscore) | 478 | 304 | 163 | 36 | 0 |
| [2, pipelines and the rest](#result--pipelines-viz-tools-scripts) | 448 | 356 | 90 | 10 | 2 |
| **In scope** | **926** | **660** | | | |
| **Whole suite** | **1563** | **1297** | | | |

Counts are collected pytest items against `origin/main` at `e92a9be` plus the 2-D CFAR audit's nine
new tests. Deleted counts Redundant items only; Weak items deleted against a rewritten survivor are
in each part's table. Tracking is not in either part: it is deferred to
[`tracker-001` §13 step 9](../../spec/tracker-001.md#13-migration-from-todays-code).

## Part 1 — `tests/core/`

Scope: the eleven `tests/core/` modules that test the signal chain: `test_ambiguity`,
`test_constants`, `test_detection`, `test_dsp`, `test_geodesy`, `test_radar`,
`test_radar_equation`, `test_signal`, `test_targets`, `test_waveforms` and `test_windows`. None of
them uses a shared conftest fixture or helper. Tracking (`tests/core/tracking/**` and
`tests/core/test_tracking.py`) is deferred to `spec/tracker-001.md` by the user's decision and is
untouched. This stream changed no
`src/`. It was done on `origin/main` at `e92a9be` and then merged with the 2-D CFAR audit
(`refactor/audit-cfar-2d`, "Stream A"), whose four new `test_detection.py` tests are classified
below with the rest.

Every test was judged against `spec/refactor-002-spec-first-audit.md` §2.1–§2.3 and against
`docs/conventions/testing.md` §10.1 items 1–4: it names a plausible bug, it is the cheapest test
that catches that bug, it tests our code rather than NumPy's or SciPy's, and it fits the time
budget. The floor of testing.md §8 holds for every public function in the modules these files
test: at least one test against known-correct values, every documented `Raises` triggered, and
every `Notes` edge case tested.

## Change of aim

Spec §2.4 said no test is deleted for redundancy. On 2026-10-09 the user reversed that, and §2.4
now records the reversal and the rules pruning is held to (R2.4.1–R2.4.6): "please
del tests for being redundant. try to reduce the total number of tests. only keep the insightful
and necessary ones." This audit therefore deletes as well as fixes. Every deletion names its
survivor, the kept test that catches the same bug. No tolerance was loosened.

## Verdict key

| Class | Meaning | Action |
| :--- | :--- | :--- |
| **Sound** | Earns its place | Kept |
| **Redundant** | Catches no bug that a kept test does not already catch | Deleted, with its survivor named |
| **Weak** | Aims at a real bug, but the setup or assertion does not convince | Rewritten, or deleted if a kept test already catches the bug |
| **Wrong** | Asserts something untrue or unphysical | Fixed, with what it was hiding. None found |

Counts are collected pytest items, so each parametrize case counts once. A row covering several
cases gives the number in brackets.

## Findings — `tests/core/`

No test asserted something untrue, so there are no Wrong findings. Three Weak findings were
confirmed by mutation: the original test stayed green under the bug it was meant to catch (see
[Mutation spot-check](#mutation-spot-check)). W20 is a spec-to-test gap rather than a test verdict.

| # | Where | Verdict |
| :--- | :--- | :--- |
| [W1](#w1-equal-gains-hide-a-squared-gain) | `test_radar_equation.py` closed form, bistatic reduction | Weak — rewritten, confirmed by mutation |
| [W2](#w2-the-mean-earth-radius-was-pinned-to-a-200-m-band) | `test_constants.py` | Weak — rewritten |
| [W3](#w3-the-alias-answer-had-an-unjustified-01-ms-tolerance) | `test_ambiguity.py` alias | Weak — tightened |
| [W4](#w4-the-broadcast-test-broadcast-nothing) | `test_ambiguity.py` broadcast | Weak — rewritten |
| [W5](#w5-only-burst-as-interval-was-ever-validated) | `test_ambiguity.py` interval `Raises` | Weak — rewritten |
| [W6](#w6-the-equatorial-step-quoted-the-wrong-error-term) | `test_geodesy.py` equatorial step | Weak — rewritten |
| [W7](#w7-a-45-elevation-cannot-see-a-zenith-angle) | `test_geodesy.py` elevation | Weak — rewritten |
| [W8](#w8-an-autocorrelation-cannot-see-a-reversed-output) | `test_dsp.py` matched filter | Weak — rewritten, confirmed by mutation |
| [W9](#w9-doppler-sign-was-a-half-plane-check) | `test_dsp.py` Doppler sign | Weak — rewritten |
| [W10](#w10-the-double-canceller-had-no-value-check) | `test_dsp.py` MTI | Weak — rewritten |
| [W11](#w11-the-wrapper-test-forwarded-no-arguments) | `test_dsp.py` range-Doppler map | Weak — rewritten |
| [W12](#w12-the-compression-width-had-a-factor-of-two-band) | `test_waveforms.py` | Weak — tightened |
| [W13](#w13-the-negative-axis-was-compared-with-itself) | `test_windows.py` | Weak — rewritten |
| [W14](#w14-exact-bin-tests-allowed-a-whole-bin) | `test_signal.py` FMCW placement (5 items) | Weak — 2 rewritten, 3 deleted |
| [W15](#w15-the-several-targets-test-checked-shapes-only) | `test_signal.py` line of sight | Weak — rewritten |
| [W16](#w16-the-s2-doppler-band-was-13-bins-wide) | `test_signal.py` pulsed Doppler | Weak — tightened |
| [W17](#w17-the-guard-test-did-not-produce-the-masking-it-describes) | `test_detection.py` guard cells | Weak — rewritten |
| [W18](#w18-the-power-weighted-centroid-was-bounded-not-pinned) | `test_detection.py` centroid | Weak — tightened |
| [W19](#w19-the-go-so-partition-held-by-construction) | `test_detection.py` GO+SO partition (12 items) | Weak — 2 rewritten, 10 deleted, confirmed by mutation |
| [W20](#w20-a-spec-acceptance-figure-cited-a-test-that-did-not-exist) | `spec/scenario-003-tracking.md` §12 | Gap — test added, spec row pointed at it |

### W1. Equal gains hide a squared gain

`test_matches_closed_form` and `test_reduces_to_monostatic_at_equal_ranges` used NOMINAL's equal
15 dBi gains. An equation computing `G_t**2` (one gain squared, the other dropped) gives the same
number at equal gains, so only `test_the_two_gains_enter_symmetrically` could see it. Both tests now
set G_r 3 dB below G_t, which also lets the closed form subsume the λ², linear-P_t and gain-symmetry
tests. Mutation M7 (G_t squared): the old closed-form test passed and the new one fails.

### W2. The mean Earth radius was pinned to a 200 m band

`EARTH_RADIUS_M` was checked only through `7000 < a − R < 7200` m. A value wrong by 150 m would have
passed. It is now pinned exactly to the IUGG mean radius R1 = 6 371 008.7714 m (Moritz, GRS 80),
which also makes the "not the equatorial radius" test redundant.

### W3. The alias answer had an unjustified 0.1 m/s tolerance

The 300 m/s alias is returned as a burst-A candidate, folded A plus a whole number of A spans, and
300 − 6·(2 × 38.24) = −158.88 m/s is exactly such a candidate. Measured error: 2.8e-14 m/s. The
tolerance is now 1e-9, as in the span sweep.

### W4. The broadcast test broadcast nothing

`test_broadcasts_the_two_bursts_against_each_other` passed two arrays of the same shape (1, 3).
Burst A is now a (3, 1) column and burst B a (1, 3) row. The diagonal must recover the three
truths, and every off-diagonal pairing must be nan: they miss burst B by 1.14 to 14.1 m/s, against
a 1 m/s tolerance.

### W5. Only burst A's interval was ever validated

Both rejection cases put the bad interval on burst A, so the `or unambiguous_velocity_b_mps <= 0`
half of the guard was never exercised. The two cases are now a zero on A and a zero on B, so the
item count is unchanged. The −1 case added nothing beyond the zero.

### W6. The equatorial step quoted the wrong error term

The comment justified rtol 1e-8 by "chord vs arc, O(θ²/24)". The ENU east component at the equator
is a·sin Δλ, which is shorter than the arc a·Δλ by θ²/6, not θ²/24. It was inside 1e-8 only because
θ was small. The test now asserts east = a·sin Δλ to 1e-12 (measured: exact). It also asserts
up = −a(1 − cos Δλ) to 1e-9 at Δλ = 0.18°, i.e. 20 km. That is the "about 30 m" curvature drop the
`ecef_to_enu_m` Notes describe, which no test had covered (31.47 m; the cancellation of a·cos Δλ −
a leaves about 1e-9 m of round-off, measured 1.3e-11 relative).

### W7. A 45° elevation cannot see a zenith angle

At (0, 1000, 1000), arcsin(up/R) and arccos(up/R) both give 45°, so
`test_elevation_of_forty_five_degrees` could not catch elevation measured from the zenith. A
Pythagorean point (300, 400, 1200) replaces it and the three single-angle tests. It gives range
1300 m exactly, elevation asin(12/13) = 67.38° (a zenith angle would read 22.62°) and azimuth
atan2(3, 4) = 36.87°. Mutation M9 confirms the new test catches the zenith convention.

### W8. An autocorrelation cannot see a reversed output

The cross-check against `np.correlate` used `samples == reference`. An autocorrelation is
Hermitian-symmetric, r[−k] = r[k]*, so an output returned time-reversed and conjugated matches it
exactly. The echo is now the chirp delayed 17 samples and scaled by 0.5j. The test also asserts
the peak at n_ref − 1 + 17, which absorbs the delay, linearity, peak-energy and compression tests.
Mutation M1 (reversed, conjugated output): the old test passed and the new one fails.

### W9. Doppler sign was a half-plane check

`test_separates_closing_from_opening` asserted only `> N/2` and `< N/2`. It now asserts bins 19
and 13 exactly for ±3/32 turn per chirp, plus magnitude N. That catches a missing fftshift (bins 3
and 29), a sign flip, and a hidden normalisation, and it absorbs the zero-Doppler-centre, Parseval
and cube tests.

### W10. The double canceller had no value check

`test_double_canceller_nulls_harder` asserted only that the 3-pulse output was smaller than the
2-pulse one. Wrong taps such as [1, −1, 1] would pass. It now asserts the closed form
|H| = 4 sin²(π f_d T) = 3.95e-3 at 0.01 PRF to 1e-10.

### W11. The wrapper test forwarded no arguments

`test_equals_the_two_transforms_in_sequence` called `range_doppler_map` with no windows and no
lengths, so a wrapper that swapped or dropped them passed. It now passes a Hann fast-time window, a
Hamming slow-time window and two different padded lengths.

### W12. The compression width had a factor-of-two band

At 4× oversampling the −3 dB width was asserted inside [0.5/B, 2/B]. It now runs at BT = 100 and
64× oversampling, and asserts 0.886/B to 3%. The bound is re-derived as follows: counting samples
quantises the width to 1/(64B), which is 1.8% of 0.886/B, and at BT = 100 the mainlobe departs from
the large-BT sinc by well under 1%. Measured: +0.5%. This is now the Notes compression test, and
the 20%-band copy in `test_dsp.py` was deleted against it.

### W13. The negative axis was compared with itself

`test_weights_a_negative_axis` compared `axis=-1` with `axis=1`, two calls through the same
normalisation. It now compares `axis=-1` with `cube * window`, which absorbs the axis-1 test.

### W14. Exact-bin tests allowed a whole bin

The FMCW stationary test (3 cases), the slow-closing test and the folding test said "exact bin".
Their targets sat off bin centres (5000 m is bin 66.7) and their assertions allowed one range bin,
or ±0.1 m/s, which is 1.7 Doppler bins at S1's 0.0597 m/s spacing. An off-by-one bin passed. Two
tests replace the five. One places a closing target on range bin 133 and Doppler bin N/2 + 50, and
asserts the integer pair; it absorbs the opening-sign test. The other places a 79.5 m/s target
five spans above that Doppler bin and asserts the same pair. The S1 bin spacing equals c/2B
(74.948 m), and the Doppler span is exactly λ·PRF/2 = 15.2955 m/s, so both placements are exact.
Range migration over the dwell is 0.8 m and 20 m, both under a third of a bin.

### W15. The several-targets test checked shapes only

`test_handles_several_targets_at_once` asserted `n_paths`, an angle shape and the bounce count, so
a broadcast that reused one target's range or rate for all of them passed. It now checks each
path's delay and Doppler against its own target.

### W16. The S2 Doppler band was 1.3 bins wide

±2 m/s at S2's 1.494 m/s spacing. The target is now placed on bin N/2 + 54 (+80.66 m/s) and the
bin is asserted exactly.

### W17. The guard test did not produce the masking it describes

The docstring said a spread target masks itself without guard cells. At n_train = 16 it did not:
without guards it was still detected at cell 62, and the assertion was only "no fewer cells with
guards". At n_train = 4, alpha(1e-4) = 17.30. Without guards the cut's skirts (30 + 200 + 200 + 30)
set the threshold to 17.30 × 464/8 = 1003 against a 600 peak, so nothing is detected. A 2-cell
guard leaves only floor cells, a threshold of 17.3, and a detection. The test now asserts both
halves.

### W18. The power-weighted centroid was bounded, not pinned

Cells 10, 11, 12 at powers 1, 4, 3 were asserted to centroid in (11, 12). The value is
(10 + 44 + 36)/8 = 11.25 exactly, now asserted to 1e-12. The symmetric half of the test is implied.

### W19. The GO-SO partition held by construction

`test_greatest_and_smallest_of_partition_the_same_total` checked GO + SO = 2(1 + β)^-N over 12
cases and said it "would fail if either series were mis-indexed". The code computes GO as exactly
that total minus the SO series, so the identity holds for any series, right or wrong. Mutation M17
mis-indexes the binomial coefficient. Under it, the partition test passes, and so does the N = 1
hand derivation (its one-term series is unaffected); only the Monte Carlo GO and SO rates would
have caught it.

It is replaced by `test_greatest_and_smallest_of_match_a_numerical_integration`, which shares no
code with the series. Each half-window mean of unit-mean exponential cells is Gamma(N, 1/N), and
Pfa = E[exp(−α g)] for g the larger (density 2fF) or smaller (2f(1 − F)) of two of them, integrated
with `scipy.integrate.quad`. The two cases are β = 2 (N = 5, α = 10), where the Notes' GO
cancellation bites, and β = 0.125 (N = 16, α = 2), the typical regime. The other ten cases added no
regime. quad agrees with the series to 2e-15; the assertion is rtol 1e-10.

### W20. A spec acceptance figure cited a test that did not exist

`spec/scenario-003-tracking.md` §12 attributed "245 760 cells; alpha = 11.417 dB" to
`tests/core/test_detection.py`, and its §3 says the implementation must re-derive the valid-cell
count and assert it in a test. No test asserted either number, here or in `tests/pipelines/`.
Under refactor-002 R1.3.4 the figure is worth pinning: it is the operating point the scenario's
whole false-alarm budget is derived from. So it gets a test rather than a spec correction.

`test_scenario_003_calibration_matches_its_specification` reads the operating point from
`scenarios/scenario_003_tracking.toml`: CA along range, n_train = 16 and n_guard = 4 per side,
pfa = 1e-5, on a (256, 1000) map (256 pulses; 1 ms × 1 MHz = 1000 samples). It asserts:

- the valid cells: 256 × (1000 − 2 × 20) = 245 760, exactly;
- α against the CA closed form M(pfa^(−1/M) − 1), to rtol 1e-12. With M = 32 that is
  32(e^(ln 10⁵/32) − 1) = 32 × 0.433013 = 13.8564;
- 10 log10 α = 11.4165 dB to 1e-4 dB, which the spec rounds to 11.417.

Mutation M22 (the TOML's `pfa` changed to 1e-4) turns it red, so config drift away from the spec
fails too. The spec row now names the test and gives α in both forms.

### Coordination with Stream A

While Stream A was open, deletions in `test_detection.py` were whole tests only, and sixteen
parametrize cases were held. After the merge, each was re-checked against the merged source:

- Ten were the W19 partition cases.
- Five are deleted with their survivors named in the table: pfa −0.1 and 1.5, alpha −1, rank −1,
  and the ring's SO case.
- One is not Redundant after all, and is kept: `test_rejects_power_that_is_not_finite[inf]`. The
  guard is `np.isfinite`, and a regression to `np.isnan` passes the nan case and fails only the inf
  one (mutation M21). Calling it held was a mistake.

`test_a_ring_has_less_cfar_loss_than_a_line_of_the_same_reach` was deleted on the pre-merge base.
After the merge, Stream A's F9 correction (scenario-003 §13.2, core-audit F9) cites it by name as
the evidence for the reach claim, so it is restored and reclassified Sound. No other deleted test
name appears anywhere in `spec/`, `docs/` (outside `testing.md`'s illustrative examples), `src/`,
`scripts/` or the rest of `tests/`.

## `tests/core/test_constants.py`

| Test | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `test_speed_of_light_is_the_exact_si_value` | Sound | Exact SI definition | |
| `test_speed_of_light_is_not_the_3e8_approximation` | Redundant | Implied by the exact pin | `test_speed_of_light_is_the_exact_si_value` |
| `test_boltzmann_is_the_exact_si_value` | Sound | Exact SI definition | |
| `test_thermal_noise_floor_matches_the_textbook_minus_174_dbm_per_hz` | Sound | Only check of T0 = 290 K, to ±0.7 K | |
| `test_four_thirds_earth_radius_is_derived_not_retyped` | Sound | Derivation, not a second literal | |
| `test_every_exported_constant_is_a_positive_float` | Redundant | Every constant now has an exact value pin | the exact-value tests |
| `test_all_is_complete` | Sound | `test_public_api` checks `__all__` names exist, not that every public constant is in it | |
| `test_wgs84_semi_major_axis_is_the_defining_value` | Sound | Defining constant | |
| `test_wgs84_semi_major_axis_is_not_the_mean_radius` → `test_earth_radius_is_the_iugg_mean_radius` | Weak | [W2](#w2-the-mean-earth-radius-was-pinned-to-a-200-m-band) | |
| `test_wgs84_flattening_matches_the_published_inverse` | Sound | Published 1/f | |
| `test_wgs84_polar_axis_is_about_21_km_shorter` | Redundant | b = a(1 − f), both pinned | the a and f pins |

## `tests/core/test_targets.py`

| Test | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `test_decade_values_are_exact_powers_of_ten` [5] | Sound [1], Redundant [4] | One decade point catches 20 log, sign and ln slips; the other four add no regime. Kept as `test_twenty_dbsm_is_one_hundred_square_metres` | `test_twenty_dbsm_is_one_hundred_square_metres` |
| `test_round_trips` | Sound | Only value check of `m2_to_dbsm` over a range | |
| `test_three_db_is_a_factor_of_two` | Redundant | Same power law as the decade point | `test_twenty_dbsm_is_one_hundred_square_metres` |
| `test_vectorises` | Redundant | Array input is covered by the round trip and `test_public_api` | `test_round_trips` |
| `test_rejects_non_positive_area_in_decibels` [2] | Sound [1], Redundant [1] | Zero is the boundary of the guard; −1 adds nothing. Kept as `test_rejects_a_zero_area_in_decibels` | `test_rejects_a_zero_area_in_decibels` |
| `test_from_dbsm_matches_the_linear_constructor` | Sound | Known value of `from_dbsm` | |
| `test_dbsm_property_round_trips` | Sound | `rcs_dbsm` and `name` pass-through | |
| `test_a_zero_cross_section_target_is_permitted_for_noise_only_runs` | Sound | `Raises` boundary: zero allowed, `-inf` dB | |
| `test_rejects_a_negative_cross_section` | Sound | `Raises` | |
| `test_is_frozen` | Sound | Removing `frozen=True` breaks nothing else | |
| `test_swerling_fluctuation_is_not_implemented` | Redundant | A check for a feature that does not exist, which testing.md §10.1 rules out; it guards no behaviour | none needed |

## `tests/core/test_radar_equation.py`

| Test | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `test_matches_closed_form` | Weak | [W1](#w1-equal-gains-hide-a-squared-gain) | |
| `test_inverse_fourth_power_scaling` | Sound | Notes: doubling range costs 12 dB. Now takes an array, absorbing the broadcast test | |
| `test_broadcasts_over_range` | Redundant | Array input folded into the 1/R⁴ test | `test_inverse_fourth_power_scaling` |
| `test_loss_attenuates` | Sound | Loss divides, not multiplies | |
| `test_rejects_non_positive_range` [2] | Sound [1], Redundant [1] | Zero is the boundary; kept as `test_rejects_a_zero_range` | `test_rejects_a_zero_range` |
| `test_rejects_loss_below_one` | Sound | `Raises` | |
| `test_reduces_to_monostatic_at_equal_ranges` | Weak | [W1](#w1-equal-gains-hide-a-squared-gain) | |
| `test_power_follows_the_range_product_not_the_range_sum` | Sound | Notes: product, not sum. Now one array call over three geometries | |
| `test_doubling_either_range_costs_twelve_decibels` [2] | Redundant | A function of R_t·R_r only (product test) that is R⁻⁴ at R_t = R_r (reduction test) is (R_t R_r)⁻² | the product and reduction tests |
| `TestBistaticReceivedPower::test_broadcasts_over_targets` | Redundant | Array input folded into the product test | `test_power_follows_the_range_product_not_the_range_sum` |
| `TestBistaticReceivedPower::test_rejects_non_positive_range` [3] | Sound [2], Redundant [1] | A zero on each range; −1 adds nothing | the tx and rx zero cases |
| `test_rejects_a_gain_dressed_as_a_loss` | Sound | `Raises` | |
| `test_power_scales_with_the_square_of_the_wavelength` | Redundant | λ¹ fails the closed form by a factor of 257 | `test_matches_closed_form` |
| `test_power_is_exactly_linear_in_the_transmitted_power` | Redundant | A squared P_t fails the closed form by 100× | `test_matches_closed_form` |
| `test_the_two_gains_enter_symmetrically` | Redundant | Unequal gains in the closed form catch a squared or dropped gain (M7) | `test_matches_closed_form` |

## `tests/core/test_ambiguity.py`

| Test | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `test_leaves_an_unambiguous_velocity_alone` | Redundant | Any centring error also moves the 80 m/s fold or breaks the interval | `test_folds_an_aircraft_by_the_predicted_number_of_intervals`, `test_every_result_lies_inside_the_interval` |
| `test_folds_an_aircraft_by_the_predicted_number_of_intervals` | Sound | Known value | |
| `test_the_interval_is_half_open_at_the_top` | Sound | Documented `[-v, +v)` | |
| `test_every_result_lies_inside_the_interval` | Sound | Catches `np.fmod` on negatives | |
| `test_folding_is_periodic_in_the_full_span` | Redundant | A correct mod is periodic; a wrong span fails the 80 m/s value | `test_folds_an_aircraft_by_the_predicted_number_of_intervals` |
| `test_preserves_shape` | Redundant | Elementwise NumPy; the sweep passes arrays | `test_every_result_lies_inside_the_interval` |
| `TestFoldVelocityMps::test_rejects_a_non_positive_interval` [2] | Sound [1], Redundant [1] | Zero boundary kept | `test_rejects_a_zero_interval` |
| `test_recovers_every_velocity_the_prf_pair_can_reach` | Redundant | The ±225 m/s sweep at the 229 m/s bound covers the ±190 m/s one; its residual check moved there | `test_reaches_the_full_span_the_lcm_of_the_two_spans_allows` |
| `test_a_slow_target_needs_no_unfolding` | Redundant | 12 m/s is inside the sweep | `test_reaches_the_full_span_the_lcm_of_the_two_spans_allows` |
| `test_reaches_the_full_span_the_lcm_of_the_two_spans_allows` | Sound | Notes: LCM span. Now also asserts zero residual | |
| `test_a_target_outside_the_bound_can_alias_to_a_confident_wrong_answer` | Weak | [W3](#w3-the-alias-answer-had-an-unjustified-01-ms-tolerance) | |
| `test_a_target_outside_the_bound_with_no_alias_is_unresolved` | Redundant | Same nan branch, residual over tolerance | `test_reports_an_inconsistent_pair_as_unresolved` |
| `test_reports_an_inconsistent_pair_as_unresolved` | Sound | Returns: nan, not a guess | |
| `test_survives_noise_below_the_tolerance` | Sound | Tolerance window, circular residual | |
| `test_broadcasts_the_two_bursts_against_each_other` → `test_pairs_every_burst_a_reading_with_every_burst_b_reading` | Weak | [W4](#w4-the-broadcast-test-broadcast-nothing) | |
| `TestUnfoldDopplerDualPrf::test_rejects_a_non_positive_interval` [2] | Weak [2] | [W5](#w5-only-burst-as-interval-was-ever-validated) | |
| `test_rejects_two_equal_rates` | Sound | `Raises` | |
| `test_rejects_a_non_positive_tolerance` [2] | Sound [1], Redundant [1] | Zero boundary kept | `test_rejects_a_zero_tolerance` |
| `test_rejects_a_bound_below_the_smaller_interval` | Sound | `Raises` | |

## `tests/core/test_geodesy.py`

| Test | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `test_equator_prime_meridian_is_the_semi_major_axis` | Sound | Closed form | |
| `test_north_pole_is_the_semi_minor_axis` | Sound | Catches e² misuse | |
| `test_altitude_adds_along_the_radial_at_the_equator` | Redundant | The pure-up test at 36° N catches h omitted or added off the normal (a 6 m error) | `test_altitude_above_the_origin_is_pure_up` |
| `test_broadcasts_to_a_trailing_axis_of_three` | Redundant | The rigid-motion test feeds (32,) arrays and takes `norm(axis=-1)` | `test_is_a_rigid_motion_so_distance_is_preserved` |
| `test_rejects_impossible_latitude` [3] | Sound [1], Redundant [2] | −90.1 catches a guard missing `abs`; +90.1 and 180 add nothing. Kept as `test_rejects_a_latitude_south_of_the_pole` | `test_rejects_a_latitude_south_of_the_pole` |
| `test_the_origin_maps_to_zero` | Redundant | A missing origin subtraction fails pure-up by kilometres | `test_altitude_above_the_origin_is_pure_up` |
| `test_altitude_above_the_origin_is_pure_up` | Sound | Normal direction at a real latitude | |
| `test_equatorial_east_step_matches_the_exact_arc` → `test_equatorial_east_step_and_the_earth_curving_away` | Weak | [W6](#w6-the-equatorial-step-quoted-the-wrong-error-term) | |
| `test_is_a_rigid_motion_so_distance_is_preserved` | Sound | Orthogonality of the rotation | |
| `TestGeodeticToEnu::test_rejects_a_last_axis_that_is_not_three` | Sound | `Raises` | |
| `test_azimuth_is_zero_at_north_and_increases_clockwise` [5] | Sound [1], Redundant [4] | The four cardinal points now go in one (4, 3) call, which also covers leading axes. North-east adds nothing to them | the rewritten single-call test |
| `test_elevation_is_ninety_degrees_straight_up` | Redundant | Zenith convention caught by the Pythagorean point | `test_a_pythagorean_point_gives_exact_range_and_angles` |
| `test_elevation_is_zero_on_the_horizontal_plane` | Redundant | Same | `test_a_pythagorean_point_gives_exact_range_and_angles` |
| `test_range_is_the_euclidean_norm` | Redundant | Range 1300 m asserted there | `test_a_pythagorean_point_gives_exact_range_and_angles` |
| `test_elevation_of_forty_five_degrees` → `test_a_pythagorean_point_gives_exact_range_and_angles` | Weak | [W7](#w7-a-45-elevation-cannot-see-a-zenith-angle) | |
| `test_vectorises_over_leading_axes` | Redundant | The cardinal test is a (4, 3) call | `test_azimuth_is_zero_at_north_and_increases_clockwise` |
| `TestEnuToRangeAzimuthElevation::test_rejects_a_last_axis_that_is_not_three` | Sound | `Raises` | |
| `test_scenario_001_track_lies_where_the_spec_says` | Redundant | Checks test-local coordinates against a 4–25 km band. A flipped sign or swapped argument is caught by the closed forms | `test_equatorial_east_step_and_the_earth_curving_away`, `test_altitude_above_the_origin_is_pure_up` |

## `tests/core/test_dsp.py`

| Test | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `test_matches_numpy_correlate` | Weak | [W8](#w8-an-autocorrelation-cannot-see-a-reversed-output) | |
| `test_peaks_at_zero_lag_with_the_pulse_energy` | Redundant | The whole output is pinned, and the peak index is asserted | `test_matches_numpy_correlate` |
| `test_compresses_by_the_time_bandwidth_product` | Redundant | Output pinned; the Notes compression claim is tested at 3% in waveforms | `test_matches_numpy_correlate`, `test_waveforms.py::test_matched_filter_compresses_by_time_bandwidth_product` |
| `test_delays_the_peak_by_the_target_delay` | Redundant | The cross-check now uses a 17-sample delayed echo | `test_matches_numpy_correlate` |
| `test_is_linear_in_two_targets` | Redundant | Linearity of `fftconvolve`; a general input is pinned | `test_matches_numpy_correlate` |
| `test_compresses_each_chirp_of_a_cube_independently` | Sound | Our axis handling | |
| `TestMatchedFilter` rejections [3] | Sound | `Raises` | |
| `test_places_an_exact_tone_on_its_exact_bin` | Sound | Now also pins magnitude N | |
| `test_is_not_shifted` | Redundant | A shift moves bin 1 to 33 | `test_places_an_exact_tone_on_its_exact_bin` |
| `TestRangeFft::test_conserves_energy` | Redundant | Parseval for `np.fft` (§10.1 item 3); scaling is pinned by the magnitude-N assertion | `test_places_an_exact_tone_on_its_exact_bin` |
| `test_zero_padding_interpolates_without_moving_the_peak` | Sound | `n_fft` path | |
| `test_applies_a_window` | Sound | Window path | |
| `test_window_suppresses_sidelobes_of_an_off_bin_target` | Redundant | Re-tests Blackman-Harris sidelobes once the window is shown to be applied | `test_applies_a_window`, `test_windows.py::test_matches_published_sidelobe_levels` |
| `test_rejects_truncating_transform_length` | Sound | `Raises` | |
| `test_rejects_non_positive_transform_length` [2] | Sound [1], Redundant [1] | Zero boundary kept | `test_rejects_a_zero_transform_length` |
| `test_rejects_mismatched_window` | Sound | `Raises` | |
| `test_puts_zero_doppler_at_the_centre` | Redundant | Exact bins N/2 ± 3 imply the shift | `test_separates_closing_from_opening` |
| `test_separates_closing_from_opening` | Weak | [W9](#w9-doppler-sign-was-a-half-plane-check) | |
| `TestDopplerFft::test_conserves_energy` | Redundant | Parseval for `np.fft`; scaling pinned by magnitude N | `test_separates_closing_from_opening` |
| `test_transforms_slow_time_of_a_cube` | Redundant | Axis 0 of a 2-D cube is exercised by the exact-bins map test | `test_places_a_target_on_its_exact_bins` |
| `test_range_bins_start_at_zero_and_increase` | Redundant | The spacing test now compares the whole axis with kΔ | `test_range_bin_spacing_matches_the_deramp_relation` |
| `test_range_bin_spacing_matches_the_deramp_relation` | Sound | Now the whole axis, plus the Notes R_max = c f_s T/4B at bin N/2 | |
| `test_doppler_bins_are_centred_on_zero` | Redundant | The span test now compares the whole axis | `test_doppler_span_is_the_unambiguous_velocity` |
| `test_doppler_span_is_the_unambiguous_velocity` | Sound | Notes span; now the whole axis | |
| `test_rejects_non_positive_bin_count` [2] | Sound [1], Redundant [1] | Zero boundary kept | `test_rejects_a_zero_bin_count` |
| `test_rejects_non_positive_wavelength`, `test_rejects_non_positive_pri` | Sound | `Raises` | |
| `test_cancels_a_stationary_return` | Sound | Zero-Doppler null | |
| `test_shortens_the_dwell` | Sound | Returns: 127 from 128 | |
| `test_matches_the_analytic_frequency_response` | Sound | Notes closed form | |
| `test_nulls_the_blind_speed` | Sound | Notes blind speed | |
| `test_passes_the_optimum_doppler` | Redundant | 2 sin(π/2) = 2 is the closed form at 0.5 PRF | `test_matches_the_analytic_frequency_response` |
| `test_suppresses_clutter_under_a_moving_target` | Redundant | An integration re-check of the null and the response. Its "60 dB" was asserted as `> 10` | `test_cancels_a_stationary_return`, `test_matches_the_analytic_frequency_response` |
| `test_double_canceller_nulls_harder` → `test_double_canceller_matches_its_analytic_response` | Weak | [W10](#w10-the-double-canceller-had-no-value-check) | |
| `test_rejects_unsupported_order` [3] | Sound [2], Redundant [1] | 1 and 4 bracket {2, 3}; 0 adds nothing | the 1 and 4 cases |
| `test_rejects_too_short_a_dwell` | Sound | `Raises` | |
| `test_places_a_target_on_its_exact_bins` | Sound | §10.3 target placement | |
| `test_separates_two_targets_at_the_same_range` | Redundant | Two-target resolution demonstration, which §10.1 rules out | `test_places_a_target_on_its_exact_bins` |
| `test_equals_the_two_transforms_in_sequence` | Weak | [W11](#w11-the-wrapper-test-forwarded-no-arguments) | |
| `test_transforms_commute` | Redundant | A property of `np.fft` along different axes (§10.1 item 3) | none needed |
| `test_rejects_a_repeated_axis` | Redundant | If −1 and 1 are caught after normalisation, so are 0 and 0 | `test_rejects_aliased_repeated_axis` |
| `test_rejects_aliased_repeated_axis` | Sound | `Raises`, after normalisation | |
| `test_a_half_bin_straddle_costs_exactly_3_92_decibels` | Sound | Scalloping loss, catches hidden windowing | |
| `test_an_on_bin_tone_suffers_no_loss_at_all` | Redundant | Magnitude N now asserted in the exact-tone test | `test_places_an_exact_tone_on_its_exact_bin` |
| `test_rejects_a_non_positive_sample_rate` | Sound | `Raises`, zero boundary | |
| `test_rejects_a_negative_sample_rate` | Redundant | Same guard | `test_rejects_a_non_positive_sample_rate` |

## `tests/core/test_waveforms.py`

| Test | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `TestRangeResolution::test_matches_closed_form` | Sound | Known value | |
| `test_inversely_proportional_to_bandwidth` | Redundant | Implied by the closed form | `TestRangeResolution::test_matches_closed_form` |
| `test_broadcasts_over_array_input` | Redundant | `test_public_api` covers list and array input for this function | `test_public_api.py::test_a_python_list_is_accepted_and_a_float64_array_returned` |
| `test_rejects_non_positive_bandwidth` [2] | Sound [1], Redundant [1] | Zero boundary kept | `test_rejects_a_zero_bandwidth` |
| `TestSweepRate::test_matches_closed_form` | Sound | Known value | |
| `TestSweepRate::test_rejects_non_positive_parameters` [4] | Sound [2], Redundant [2] | One zero per parameter | the bandwidth and duration zero cases |
| `TestBeatFrequency::test_matches_closed_form` | Sound | Known value | |
| `test_is_linear_in_range` | Redundant | Implied by the closed form | `TestBeatFrequency::test_matches_closed_form` |
| `test_zero_range_gives_zero_beat` | Redundant | The round trip includes 0 m and recovers it | `test_round_trips_through_range` |
| `test_round_trips_through_range` | Sound | Only value check of the inverse | |
| `test_rejects_negative_range`, `test_rejects_negative_beat_frequency` | Sound | `Raises` | |
| `test_sample_count_is_half_open` | Sound | Returns: half-open sampling | |
| `test_is_constant_modulus` → `test_is_constant_modulus_at_the_requested_amplitude` | Sound | Now at A = 3, absorbing amplitude scaling | |
| `test_amplitude_scales_linearly` | Redundant | Modulus is asserted to equal A (M5) | `test_is_constant_modulus_at_the_requested_amplitude` |
| `test_instantaneous_frequency_sweeps_linearly` | Sound | §10.2 missing ½ (M6) | |
| `test_down_sweep_is_conjugate_of_up_sweep`, `test_start_frequency_offsets_the_sweep` | Sound | `up_sweep` and f0 paths | |
| `test_matched_filter_compresses_by_time_bandwidth_product` | Weak | [W12](#w12-the-compression-width-had-a-factor-of-two-band) | |
| `test_conserves_energy` | Redundant | Parseval for `np.fft`; energy N follows from constant modulus | `test_is_constant_modulus_at_the_requested_amplitude` |
| `test_occupies_the_swept_bandwidth` | Redundant | A slope or offset error fails the instantaneous-frequency or start-frequency test first | `test_instantaneous_frequency_sweeps_linearly`, `test_start_frequency_offsets_the_sweep` |
| `TestLfmChirp::test_rejects_non_positive_parameters` [6] | Sound [3], Redundant [3] | One zero per parameter; the negatives add nothing | the three zero cases |
| `test_rejects_sample_rate_below_bandwidth`, `test_rejects_sweep_too_short_to_sample` | Sound | `Raises` | |

## `tests/core/test_windows.py`

| Test | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `test_is_symmetric` [7] | Sound [6], Redundant [1] | Catches a periodic window, which the Harris figures at N = 4096 cannot. Rectangular is `np.ones` and cannot be periodic. Now also asserts the length | the six SciPy-built cases |
| `test_has_requested_length` [7] | Redundant | Asserted in the symmetry test; rectangular by the all-ones test | `test_is_symmetric`, `test_rectangular_is_all_ones` |
| `test_is_non_negative` [7] | Redundant | Tests SciPy's window definitions; the Harris values pin each shape | `test_matches_published_sidelobe_levels`, `test_matches_published_processing_loss` |
| `test_interior_weights_are_strictly_positive` [7] | Redundant | Same | same |
| `test_normalize_sets_unit_mean` [7] | Redundant | Mean is the coherent gain, asserted to be 1 for every taper | `test_normalized_tapers_have_unit_coherent_gain` |
| `test_rectangular_is_all_ones` | Sound | Exact | |
| `test_normalize_is_a_pure_scaling` | Redundant | A shape-changing normalisation changes the loss | `test_loss_is_invariant_to_scaling` |
| `test_sidelobes_fall_in_the_documented_order` | Redundant | Blackman-Harris −92 dB added to the Harris table; levels are ≥ 10 dB apart | `test_matches_published_sidelobe_levels` |
| `test_matches_published_sidelobe_levels` [4] | Sound [4], +1 case | Harris 1978 Table 1, referenced by testing.md §10.3. Adds `blackmanharris` (measured −92.03 dB) | |
| `test_taylor_achieves_its_design_sidelobe` | Sound | Notes: auto n̄ | |
| `test_chebyshev_achieves_its_design_sidelobe` | Sound | Equiripple | |
| `test_rejects_unknown_name` → `test_rejects_unknown_name_and_lists_the_accepted_ones` | Sound | One call, one regex | |
| `test_lists_the_accepted_names_in_the_error` | Redundant | Same call | `test_rejects_unknown_name_and_lists_the_accepted_ones` |
| `test_rejects_non_positive_length` [2], `test_rejects_non_positive_sidelobe_level` [2] | Sound [2], Redundant [2] | Zero boundaries kept | `test_rejects_a_zero_length`, `test_rejects_a_zero_sidelobe_level` |
| `test_rejects_shallow_chebyshev` | Sound | `Raises` | |
| `test_weights_the_requested_axis` | Redundant | The negative-axis test now compares with `cube * window` along the last axis | `test_weights_a_negative_axis` |
| `test_weights_a_negative_axis` | Weak | [W13](#w13-the-negative-axis-was-compared-with-itself) | |
| `test_weights_slow_time_independently` | Sound | Axis 0, random cube, exact | |
| `TestApplyTaper::test_preserves_shape` | Redundant | Value tests compare whole arrays | `test_weights_slow_time_independently` |
| `TestApplyTaper` rejections [3] | Sound | `Raises` | |
| `test_rectangular_has_unit_coherent_gain` | Redundant | All ones and unit gain for every normalised taper | `test_rectangular_is_all_ones`, `test_normalized_tapers_have_unit_coherent_gain` |
| `test_normalized_tapers_have_unit_coherent_gain` | Sound | Known value of `coherent_gain_linear` (M15) | |
| `test_rectangular_loses_nothing` | Redundant | The loss formula is pinned by four Harris values | `test_matches_published_processing_loss` |
| `test_loss_is_invariant_to_scaling` | Sound | Only test that sees a scale-dependent loss formula: the Harris tests use unit-mean windows | |
| `test_loss_is_never_negative` | Redundant | Cauchy–Schwarz; a sign error fails the Harris values | `test_matches_published_processing_loss` |
| `test_matches_published_processing_loss` [4] | Sound | Harris 1978 | |
| `test_loss_tracks_sidelobe_suppression` | Redundant | 1.76 < 2.37 dB, both pinned | `test_matches_published_processing_loss` |
| `TestGainAndLoss::test_rejects_empty_window` | Redundant | Same call as the validation class | `TestWindowMetricValidation::test_rejects_an_empty_window` |
| `test_rejects_zero_sum_window` | Sound | `Raises` | |
| `TestWindowMetricValidation::test_rejects_a_two_dimensional_window` [2] | Sound | Shows both metrics use the shared guard | |
| `TestWindowMetricValidation::test_rejects_an_empty_window` [2] | Sound [1], Redundant [1] | Once both metrics are shown to share the guard, one empty case suffices | the `coherent_gain_linear` case |

## `tests/core/test_signal.py`

| Test | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `test_delay_is_the_two_way_transit_time` | Sound | Two-way delay | |
| `test_range_property_inverts_the_delay` | Redundant | The same `range_m` property, from a known delay | `test_equivalent_range_is_half_the_path_sum` |
| `test_closing_velocity_gives_positive_doppler` | Sound | D5 sign, value 2v/λ (M11) | |
| `test_amplitude_squared_is_the_received_power` | Sound | Ties paths to the radar equation and the dBi conversion | |
| `test_power_follows_the_inverse_fourth_power_law` | Redundant | Equal to `received_power_w`, whose R⁴ is pinned | `test_amplitude_squared_is_the_received_power` |
| `test_a_zero_cross_section_target_returns_nothing` | Sound | `Raises` boundary, masked branch | |
| `test_handles_several_targets_at_once` | Weak | [W15](#w15-the-several-targets-test-checked-shapes-only) | |
| `TestLineOfSightPaths::test_rejects_a_non_positive_range` [2] | Sound [1], Redundant [1] | Zero boundary kept | `test_rejects_a_zero_range` |
| `test_rejects_a_negative_cross_section` | Sound | `Raises` | |
| `TestPropagationPathsValidation` [2] | Sound | Both `Raises` | |
| `test_cube_has_the_canonical_layout` | Redundant | A transposed cube fails exact placement through `range_doppler_map`'s default axes | `test_a_closing_target_on_bin_centres_lands_in_exactly_those_bins` |
| `test_a_stationary_target_lands_in_its_exact_range_bin` [3] | Weak [3] | [W14](#w14-exact-bin-tests-allowed-a-whole-bin); one rewritten, two deleted | `test_a_closing_target_on_bin_centres_lands_in_exactly_those_bins` |
| `test_a_slow_closing_target_lands_at_positive_velocity` | Weak | [W14](#w14-exact-bin-tests-allowed-a-whole-bin); deleted | `test_a_closing_target_on_bin_centres_lands_in_exactly_those_bins` |
| `test_an_opening_target_lands_at_negative_velocity` | Redundant | A sign flip puts the closing target on bin N/2 − 50 | `test_a_closing_target_on_bin_centres_lands_in_exactly_those_bins` |
| `test_a_fast_target_folds_in_doppler_but_not_in_range` | Weak | [W14](#w14-exact-bin-tests-allowed-a-whole-bin); rewritten | |
| `test_warns_when_stop_and_hop_is_strained` | Sound | `Warns` | |
| `test_does_not_warn_at_the_scenario_parameters` | Redundant | pyproject sets warnings to errors, and the folding test runs 79.5 m/s at S1 | `test_a_fast_target_folds_in_doppler_but_not_in_range` |
| `test_is_noiseless_without_a_generator` | Redundant | If default noise were added, "clean" would carry noise and the residual would be 2N, failing rtol 0.05 | `test_noise_is_added_when_a_generator_is_supplied` |
| `test_rejects_a_pulsed_radar`, `TestFmcwDerampBaseband::test_rejects_an_empty_dwell` | Sound | `Raises` | |
| `test_cube_spans_the_full_repetition_interval` | Sound | 100 samples, not 25 | |
| `test_an_unambiguous_target_lands_at_its_true_delay` | Sound | 12.5-sample shift to ±1 sample; one-way delay gives 6.25 | |
| `TestPulsedBaseband::test_a_distant_target_folds_in_range` | Sound | Defining S2 behaviour. Its `folded < true` assertion tested the test's own arithmetic and was removed | |
| `test_a_fast_closing_target_lands_at_positive_velocity_unfolded` | Weak | [W16](#w16-the-s2-doppler-band-was-13-bins-wide) | |
| `test_rejects_an_fmcw_radar` | Sound | `Raises` | |
| `TestPulsedBaseband::test_rejects_an_empty_dwell` [2] | Sound [1], Redundant [1] | Zero boundary kept | the 0 case |
| `TestThermalNoise` [6] | Sound | Total power, half per channel (§10.2), seed contract, zero power, `Raises`, generator wiring | |
| `test_reduces_to_monostatic_when_the_sites_all_but_coincide` | Sound | Headline equivalence | |
| `test_equivalent_range_is_half_the_path_sum` | Sound | The half-sum convention | |
| `test_delay_is_the_whole_route_over_the_speed_of_light` | Redundant | range_m = c·τ/2 = 13.5 km is the same assertion | `test_equivalent_range_is_half_the_path_sum` |
| `test_arrival_and_departure_angles_are_kept_separate` | Sound | AoA vs AoD | |
| `TestBistaticLineOfSightPaths::test_broadcasts_over_targets` | Sound | Only per-target bistatic check | |
| `test_zero_cross_section_gives_a_null_path`, bistatic rejections [3] | Sound | Boundary and `Raises` | |
| `test_matches_the_bisector_projection` | Sound | Willis §6.1 | |
| `test_is_zero_crossing_the_bisector`, `test_is_zero_moving_along_the_baseline` | Sound | Both Notes nulls | |
| `test_closing_on_both_sites_is_positive` | Redundant | The projection test asserts a positive closing value | `test_matches_the_bisector_projection` |
| `test_rejects_vectors_that_are_not_three_dimensional` | Sound | `Raises` | |
| `test_fmcw_puts_a_bistatic_target_in_the_half_sum_range_bin` | Sound | Exact bin, uneven split | |
| `test_pulsed_bistatic_target_folds_on_the_range_sum` | Sound | The sum wraps. Tautological `folded < sum` removed | |
| `test_doppler_bin_follows_the_bisector_rate` | Sound | Exact bin, bistatic sign | |
| `test_generators_accept_a_bistatic_radar` | Redundant | Both generators already run on `S1_PAIR` | `test_fmcw_puts_a_bistatic_target_in_the_half_sum_range_bin` |

## `tests/core/test_radar.py`

No Weak findings. The four scenario-001 limits stay, because spec `scenario-001-xband.md` A6 names
this file as their check.

| Test | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `test_wavelength_at_x_band` | Sound | 30.591 mm | |
| `test_wavelength_is_c_over_f` | Redundant | Same property; the literal also catches a wrong c | `test_wavelength_at_x_band` |
| `test_fmcw_sweep_rate_is_two_ghz_per_second` | Sound | Spec §3.4 | |
| `test_fmcw_runs_at_full_duty` | Sound | Notes boundary: 100% permitted, T·PRF (M12) | |
| `test_pulsed_duty_cycle` | Redundant | Same formula | `test_fmcw_runs_at_full_duty` |
| `test_pulse_repetition_interval_is_the_prf_reciprocal` | Redundant | 100 samples = PRI·f_s | `test_samples_per_pri_is_the_receive_window` |
| `test_rejects_non_positive_quantities` [5] | Sound | One case per validated field | |
| `test_rejects_a_duty_cycle_above_one`, `test_rejects_an_unknown_waveform` | Sound | `Raises` | |
| `test_noise_figure_of_three_db_is_about_a_factor_of_two` | Sound | Known value | |
| `test_zero_db_noise_figure_is_unity` | Sound | Boundary of the `< 0` guard | |
| `TestReceiver` rejections [2] | Sound | `Raises` | |
| `test_range_resolution_is_seventy_five_metres` | Sound | c/2B | |
| `test_range_resolution_is_waveform_independent` | Redundant | No waveform branch in the formula | `test_range_resolution_is_seventy_five_metres` |
| S1/S2 unambiguous range and velocity [4] | Sound | Spec A6 | |
| `test_the_two_variants_trade_one_ambiguity_for_the_other` | Redundant | Inequalities implied by the four pins | the four A6 pins |
| `test_range_doppler_product_is_bounded_by_c_over_four` | Redundant | Algebra of the two pinned S2 values | the S2 pins |
| `test_rejects_an_impossible_site_latitude`, `test_rejects_a_pulsed_receiver_that_would_alias` | Sound | `Raises` | |
| `test_permits_an_fmcw_receiver_below_the_swept_bandwidth` | Redundant | Every S1 test constructs exactly that radar | `test_s1_unambiguous_range_covers_the_track` |
| `test_noise_power_matches_ktbf` | Sound | kT0BF | |
| `test_samples_per_chirp`, `test_samples_per_pri_is_the_receive_window` | Sound | The 25 vs 100 distinction | |
| `test_pri_and_chirp_windows_agree_only_at_full_duty` | Redundant | 1000 = 1000 and 100 > 25 are asserted above | the two sample-count tests |
| `TestRadarValidation::test_is_frozen` | Sound | As for `PointTarget` | |
| `test_baseline_matches_an_independent_geodetic_distance` | Sound | Wiring | |
| `test_baseline_is_about_eighteen_kilometres` | Redundant | A frame or unit slip in src differs from the test's composition; geodesy is pinned separately | `test_baseline_matches_an_independent_geodetic_distance` |
| `test_bistatic_angle_is_pi_on_the_baseline` | Sound | Clipping before arccos | |
| `test_bistatic_angle_is_a_right_angle_for_the_isoceles_case` | Sound | Closed form | |
| `test_bistatic_angle_vanishes_for_a_distant_target` | Redundant | Formula pinned; asserted only 0 < β < 1e-2 | `test_bistatic_angle_is_a_right_angle_for_the_isoceles_case` |
| `test_bistatic_angle_broadcasts_over_targets` | Redundant | Elementwise NumPy plus monotonicity of a pinned formula | `test_bistatic_angle_is_a_right_angle_for_the_isoceles_case` |
| `test_impossible_triangle_is_rejected`, angle rejections [2] | Sound | `Raises` | |
| `test_target_ranges_are_zero_at_the_sites` | Sound | Only test of `target_ranges_m` | |
| `test_resolution_at_zero_angle_is_the_monostatic_value` | Redundant | sec(0) = 1; cos β in place of cos β/2 passes at 0 and fails at 120° | `test_resolution_degrades_as_secant_of_half_the_angle` |
| `test_resolution_degrades_as_secant_of_half_the_angle` | Sound | Closed form | |
| `test_resolution_is_infinite_in_forward_scatter` | Sound | Notes. Docstring now says it also fails on any warning (M13) | |
| `test_forward_scatter_does_not_warn` | Redundant | pyproject sets warnings to errors, so the inf test already fails on a divide warning | `test_resolution_is_infinite_in_forward_scatter` |
| `test_resolution_rejects_angles_outside_the_half_turn` [2] | Sound | Both sides | |
| `test_pulsed_unambiguous_sum_range_is_c_over_prf` | Sound | Re-derived | |
| `test_unambiguous_sum_range_is_twice_the_monostatic_range` | Redundant | c/PRF and the S2 pin | `test_pulsed_unambiguous_sum_range_is_c_over_prf` |
| `test_waveform_independent_properties_match_the_monostatic_radar` | Sound | Only test of the delegated properties | |
| `TestBistaticValidation` [4] | Sound | `Raises`, each site | |
| `test_fmcw_unambiguous_sum_range_is_c_fs_over_two_alpha` | Sound | Re-derived | |
| `test_the_fmcw_sum_range_is_twice_the_monostatic_range` | Redundant | Closed form and the S1 pin | `test_fmcw_unambiguous_sum_range_is_c_fs_over_two_alpha` |

## `tests/core/test_detection.py`

The `hypothesis` round trip, the Clopper–Pearson rate tests and their helper
`assert_rate_consistent_with_design` (named in testing.md §10.4), and the clutter-edge and
masking tests are kept. The table covers the merged file, including Stream A's four tests (marked
A); see [Coordination with Stream A](#coordination-with-stream-a).

| Test | Class | Reason | Survivor if deleted |
| :--- | :--- | :--- | :--- |
| `test_greatest_and_smallest_of_partition_the_same_total` [12] → `test_greatest_and_smallest_of_match_a_numerical_integration` [2] | Weak [12] | [W19](#w19-the-go-so-partition-held-by-construction); 2 rewritten, 10 deleted | the two integration cases |
| `test_smallest_window_matches_the_hand_derivation` | Sound | Notes; distinguishes GO from SO, which the partition cannot | |
| `test_cell_averaging_matches_the_gamma_moment_generating_function` | Sound | CA known values | |
| `test_closed_form_cell_averaging_inverse_agrees_with_bisection` | Redundant | Forward(α_CA(pfa)) = pfa is the round trip's `ca` case | `test_threshold_factor_round_trips_through_pfa[ca]` |
| `test_threshold_factor_round_trips_through_pfa` [4] | Sound | Hypothesis property; kept by rule | |
| `test_probability_of_false_alarm_decreases_with_threshold` [4] | Redundant | Monotonicity of forward maps whose values are pinned | the partition, MGF and hand-derivation tests |
| `test_threshold_factor_decreases_with_window_size` | Sound | Notes: CFAR loss | |
| `test_greatest_of_needs_a_lower_factor_than_smallest_of` | Redundant | A GO/SO swap fails the N = 1 hand derivation | `test_smallest_window_matches_the_hand_derivation` |
| `test_default_order_statistic_rank_is_three_quarters_of_the_window` | Sound | Notes | |
| `test_scenario_003_calibration_matches_its_specification` | added | [W20](#w20-a-spec-acceptance-figure-cited-a-test-that-did-not-exist) | |
| `test_false_alarm_rate_matches_design_pfa` [3] | Sound | §10.4 | |
| `test_order_statistic_false_alarm_rate_matches_design_pfa` | Sound | §10.4, OS | |
| `test_false_alarm_rate_holds_at_one_in_ten_thousand` | Redundant | testing.md §10.1: 1e-3 catches the same exponent and factor errors; it was the slowest test (0.11 s) | `test_false_alarm_rate_matches_design_pfa[ca]` |
| `test_false_alarm_rate_is_independent_of_noise_power` | Sound | The CFAR property itself. Kept rather than replaced by the floor-tracking test, given Stream A's independence-assumption work | |
| `test_threshold_tracks_a_noise_floor_that_varies_with_range` | Sound | Deterministic scale tracking over 40 dB | |
| `test_valid_mask_excludes_exactly_the_incomplete_windows` | Sound | Exact | |
| `test_valid_mask_is_empty_when_the_map_is_shorter_than_the_window` | Sound | Edge branch of the mask | |
| `test_noise_estimate_is_nan_outside_the_valid_mask` | Sound | Notes | |
| `test_edge_cells_are_never_detected` | Redundant | A nan-threshold comparison that detected edges would fail `sum == 1` (M14) | `test_detects_a_point_target_well_above_the_floor` |
| `test_noise_estimate_recovers_a_flat_floor` [4] | Redundant | CA and OS indexing are pinned by the explicit-window tests; GO and SO scale by their rate tests | `test_cell_averaging_estimate_equals_the_explicit_window_mean`, `test_order_statistic_estimate_equals_the_explicit_sorted_window`, `test_false_alarm_rate_matches_design_pfa[go/so]` |
| `test_cell_averaging_estimate_equals_the_explicit_window_mean`, `test_order_statistic_estimate_equals_the_explicit_sorted_window` | Sound | Off-by-one in the bands | |
| `test_guard_cells_protect_the_threshold_from_target_spill` | Weak | [W17](#w17-the-guard-test-did-not-produce-the-masking-it-describes) | |
| `test_smallest_of_holds_detection_against_an_interfering_target`, `test_greatest_of_suppresses_false_alarms_at_a_clutter_edge` | Sound | §10.4 | |
| `test_detects_a_point_target_well_above_the_floor` | Sound | Detection, and nothing else | |
| `test_cfar_runs_along_the_requested_axis` | Sound | Only `axis=0` test | |
| `test_clusters_two_separated_targets` | Sound | Peaks, order, sizes, totals | |
| `test_cluster_centroid_is_power_weighted` | Weak | [W18](#w18-the-power-weighted-centroid-was-bounded-not-pinned) | |
| `test_clusters_are_connected_in_two_dimensions`, `test_clustering_an_empty_mask_returns_nothing`, `test_detection_is_immutable` | Sound | Connectivity, empty branch, frozen | |
| `test_detects_a_point_target_in_a_range_doppler_map` | Redundant | A pipeline re-check: bins are pinned in `test_dsp.py`, detection and clustering here | `test_dsp.py::test_places_a_target_on_its_exact_bins`, `test_detects_a_point_target_well_above_the_floor` |
| `test_rejects_an_unknown_variant` | Sound | `Raises` | |
| `test_rejects_a_probability_outside_the_unit_interval` [4] | Sound [2], Redundant [2] | 0 and 1 are the boundaries of the open interval (M20); −0.1 and 1.5 add nothing | the 0 and 1 cases |
| `test_rejects_a_non_positive_threshold_factor` [4] | Sound [3], Redundant [1] | 0, nan and inf are distinct; −1 adds nothing | the 0 case |
| `test_rejects_a_rank_outside_the_reference_window` [3] | Sound [2], Redundant [1] | 0 and 17 bracket 1..16 (M18); −1 adds nothing | the 0 case |
| remaining 1-D `Raises` tests [10] | Sound | One per documented `Raises`, including the too-short estimate branch | |
| `test_ring_mean_equals_the_explicit_ring_mean`, `test_ring_order_statistic_equals_the_explicit_sorted_ring` | Sound | Explicit ring enumeration | |
| `test_ring_follows_its_axes_when_the_map_is_transposed` [2] | Sound | `uniform_filter` and `rank_filter` paths | |
| `cfar_valid_mask_2d` tests [3] | Sound | Wrapped, unwrapped, defined-where, too short | |
| `test_ring_threshold_factor_satisfies_the_cell_averaging_closed_form` | Sound | Pins M | |
| `test_a_ring_has_less_cfar_loss_than_a_line_of_the_same_reach` | Sound | Cited by scenario-003 §13.2 and core-audit F9 as the evidence for the reach claim. The survivors pin its two halves separately, but not the comparison. Deleted before the merge, restored after | |
| `test_ring_false_alarm_rate_matches_design_pfa` [2] | Sound | OS ring needs it; CA kept at Stream A's request | |
| A `test_a_tapered_map_breaks_the_calibration_as_the_module_notes_say` [2] | Sound | Module Notes: a taper correlates cells and the rate exceeds pfa. The 1-D window and the ring are separate code paths | |
| `test_order_statistic_ring_holds_detection_where_cell_averaging_loses_it` | Sound | 2-D masking | |
| circular clustering [6] | Sound | Notes: wrap centroid, connectivity across the wrap, torus, noise estimate carried or nan | |
| `test_a_target_on_the_doppler_wrap_gives_one_detection` | Sound | §9 regression for spec 003 §14.2 | |
| A `test_the_ring_variants_are_the_one_dimensional_ones_without_half_windows` | Sound | A 2-D variant must be one the 1-D calibration covers (M19) | |
| `test_ring_rejects_the_half_window_variants` [2] | Sound [1], Redundant [1] | The guard tests membership of `CFAR_VARIANTS_2D`, whose contents the variant-set test pins | the `go` case, `test_the_ring_variants_are_the_one_dimensional_ones_without_half_windows` |
| 2-D `_ring` argument and rank checks [4] | Sound | `Raises`; kept at Stream A's request | |
| A `test_ring_rejects_a_pair_that_is_not_two_integers` [5] | Sound | Five distinct failure modes of `_integer_pair`: a third count, a float, a bare int, `n_guard`, `axes` | |
| A `test_ring_accepts_numpy_integer_counts` | Sound | `operator.index` accepts NumPy integers; an `isinstance(int)` check would not | |
| `test_rejects_power_that_is_not_finite` [2] | Sound [2] | nan and inf each catch a different narrowing of `isfinite` (M21) | |
| `test_rejects_a_noise_estimate_of_the_wrong_shape`, `test_rejects_an_out_of_bounds_wrap_axis` | Sound | `Raises` | |

## Result — `tests/core/`

### Counts

Collected pytest items. "Before" is `origin/main` at `e92a9be` plus Stream A's nine new
`test_detection.py` items, which is the tree this branch is merged onto. A's items are classified
here like any other.

| File | Before | After | Sound | Weak | Redundant deleted |
| :--- | --: | --: | --: | --: | --: |
| `test_ambiguity.py` | 22 | 14 | 10 | 4 | 8 |
| `test_constants.py` | 11 | 8 | 7 | 1 | 3 |
| `test_detection.py` | 114 | 87 | 82 | 14 | 18 |
| `test_dsp.py` | 52 | 30 | 26 | 4 | 22 |
| `test_geodesy.py` | 24 | 10 | 8 | 2 | 14 |
| `test_radar.py` | 58 | 43 | 43 | 0 | 15 |
| `test_radar_equation.py` | 19 | 10 | 8 | 2 | 9 |
| `test_signal.py` | 55 | 41 | 37 | 7 | 11 |
| `test_targets.py` | 16 | 8 | 8 | 0 | 8 |
| `test_waveforms.py` | 33 | 20 | 19 | 1 | 13 |
| `test_windows.py` | 74 | 33 | 31 | 1 | 42 |
| **In scope** | **478** | **304** | **279** | **36** | **163** |
| **Whole suite** | **1563** | **1389** | | | |

After = Sound + the Weak tests kept (rewritten) + two added tests. Thirteen Weak items were
deleted against a rewritten survivor: three in signal ([W14](#w14-exact-bin-tests-allowed-a-whole-bin))
and ten in detection ([W19](#w19-the-go-so-partition-held-by-construction)). The two added tests
are the Blackman-Harris Harris case and the scenario-003 calibration test.

Against `e92a9be` alone, without Stream A's nine items: in scope 469 → 295, whole suite
1554 → 1380, `test_detection.py` 105 → 78. The in-scope files run in 2.3 s, down from 2.9 s.

### Coverage

Measured with `pytest --cov=radar_forge --cov-branch`, before and after, compared per source file
for newly missing lines and branches. Each round of pruning was compared against its own base:
`e92a9be` for the first, and the merged tree (after Stream A, before the second round) for the
second. Both rounds give the same result.

| Run | Missing lines | Missing branches | Files with a new gap |
| :--- | --: | --: | --: |
| Whole suite | 33 → 33 | 20 → 20 | 0 |
| The eleven in-scope files alone | 1613 → 1613 | 567 → 567 | 0 |

The second row is the stricter check, because other suites also exercise `core/`. Pruning left no
line or branch uncovered that the in-scope tests had covered.

### Mutation spot-check

Each mutation was applied to `src/` alone, the named survivor was run, and the file was restored
with `git checkout`. "Old test" is the pre-audit version of a rewritten test, run against the same
mutation.

| # | Mutation | Deleted test(s) it targeted | Survivor | Survivor | Old test |
| :--- | :--- | :--- | :--- | :-: | :-: |
| M1 | `matched_filter` output reversed and conjugated | `test_delays_the_peak_by_the_target_delay` | `test_matches_numpy_correlate` | red | green |
| M2 | `range_fft` fftshifted | `test_is_not_shifted` | `test_places_an_exact_tone_on_its_exact_bin` | red | |
| M3 | `range_fft` scaled by 1/N | `TestRangeFft::test_conserves_energy`, `test_an_on_bin_tone_suffers_no_loss_at_all` | `test_places_an_exact_tone_on_its_exact_bin` | red | |
| M4 | single-canceller taps [1, −0.9] | `test_passes_the_optimum_doppler` | `test_matches_the_analytic_frequency_response` | red | |
| M5 | `lfm_chirp` ignores `amplitude_linear` | `test_amplitude_scales_linearly` | `test_is_constant_modulus_at_the_requested_amplitude` | red | |
| M6 | LFM phase missing the ½ | `test_occupies_the_swept_bandwidth` | `test_instantaneous_frequency_sweeps_linearly` | red | |
| M7 | received power squares G_t, drops G_r | `test_the_two_gains_enter_symmetrically` | `test_matches_closed_form` | red | green |
| M8 | azimuth not wrapped to [0, 360) | west case of the azimuth parametrize | `test_azimuth_is_zero_at_north_and_increases_clockwise` | red | |
| M9 | elevation from the zenith | `test_elevation_is_ninety_degrees_straight_up` | `test_a_pythagorean_point_gives_exact_range_and_angles` | red | |
| M10 | fold not centred: `mod(v, span)` | `test_leaves_an_unambiguous_velocity_alone`, `test_folding_is_periodic_in_the_full_span` | `TestFoldVelocityMps` | red | |
| M11 | monostatic Doppler sign flipped | `test_an_opening_target_lands_at_negative_velocity` | `test_closing_velocity_gives_positive_doppler` | red | |
| M12 | duty cycle T/PRF | `test_pulsed_duty_cycle` | `test_fmcw_runs_at_full_duty` | red | |
| M13 | `np.errstate` removed from forward scatter | `test_forward_scatter_does_not_warn` | `test_resolution_is_infinite_in_forward_scatter` | red | |
| M14 | 1-D detect declares nan-threshold edge cells | `test_edge_cells_are_never_detected` | `test_detects_a_point_target_well_above_the_floor` | red | |
| M15 | coherent gain as a sum | `test_rectangular_has_unit_coherent_gain` | `test_normalized_tapers_have_unit_coherent_gain` | red | |
| M16 | CA training bands ignore the guard | — (rewrite check) | `test_guard_cells_protect_the_threshold_from_target_spill` | red | red |
| M17 | GO/SO series coefficient mis-indexed, C(N + k, k) | the 10 partition cases | `test_greatest_and_smallest_of_match_a_numerical_integration` | red | green |
| M18 | rank guard `0 <= rank` | rank −1 case | `test_rejects_a_rank_outside_the_reference_window[0]` | red | |
| M19 | `CfarVariant2d` admits `so` | ring `so` case | `test_the_ring_variants_are_the_one_dimensional_ones_without_half_windows` | red | |
| M20 | `0.0 <= pfa` accepted | pfa −0.1, 1.5 cases | `test_rejects_a_probability_outside_the_unit_interval[0.0]` | red | |
| M21 | finite check narrowed to `np.isnan` | (held `inf` case: kept) | `test_rejects_power_that_is_not_finite[inf]` | red | |
| M22 | scenario TOML `pfa` changed to 1e-4 | — (new test check) | `test_scenario_003_calibration_matches_its_specification` | red | |

All twenty-two survivors fail under their mutation. Under M21 the nan case stays green, which is
why the inf case is kept. M1, M7 and M17 show the three pre-audit tests that stayed green:
[W8](#w8-an-autocorrelation-cannot-see-a-reversed-output),
[W1](#w1-equal-gains-hide-a-squared-gain) and
[W19](#w19-the-go-so-partition-held-by-construction). Under M17 the N = 1 hand derivation stays
green as well.

## Part 2 — pipelines, viz, tools, scripts and top level

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
scenario TOML tables. Two files had a constraint. `test_scenarios.py` and `test_public_api.py` were
also being changed by `docs/retire-refactor-001`, then unmerged (it merged later as PR #23, with no
conflict). In those two, only
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
