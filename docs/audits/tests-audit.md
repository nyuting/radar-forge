# Test audit — refactor-002 §2

Scope: the eleven `tests/core/` modules that test the signal chain: `test_ambiguity`,
`test_constants`, `test_detection`, `test_dsp`, `test_geodesy`, `test_radar`,
`test_radar_equation`, `test_signal`, `test_targets`, `test_waveforms` and `test_windows`. None of
them uses a shared conftest fixture or helper. Tracking (`tests/core/tracking/**` and
`tests/core/test_tracking.py`) is deferred to `spec/tracker-001.md` by the user's decision and is
untouched. The `tests/pipelines/`, `tests/viz/`, `tests/tools/`, `tests/scripts/` and top-level
sections are written by a parallel stream and will be appended to this file. This stream changed no
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

Spec §2.4 says no test is deleted for redundancy. On 2026-10-09 the user reversed that: "please
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

## Findings

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

## Result

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
