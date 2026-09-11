"""Tests for radar_forge.core.detection.

The analytic tests here catch algebra errors in the threshold expressions. They
cannot catch a shared misunderstanding: a forward Pfa expression that is wrong
by a factor will round-trip perfectly through its own inverse and produce a
detection map that looks entirely plausible. The Monte-Carlo false-alarm-rate
tests are the ones that would catch that, which is why they exist despite being
the slowest thing in the module.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import beta

from radar_forge.core.detection import (
    CFAR_VARIANTS,
    Detection,
    cfar_detect,
    cfar_noise_estimate_w,
    cfar_probability_of_false_alarm,
    cfar_threshold_factor,
    cfar_threshold_w,
    cfar_valid_mask,
    cluster_detections,
    default_os_rank,
)
from radar_forge.core.dsp import doppler_bin_centers_mps, range_doppler_map

# --------------------------------------------------------------------------- #
# Monte-Carlo sizing
#
# These are cheap today, which is not obvious and is worth recording. The
# cell-averaging family is computed from a cumulative sum, so a run is linear in
# the number of cells and independent of n_train: measuring a 1e-4 rate over
# three million cells costs about 0.1 s. None of these tests is therefore marked
# `slow`, which docs/conventions/testing.md §7 reserves for over a second.
#
# If that changes -- a larger window family, a lower design rate, or an OS-heavy
# sweep, since OS has no running-sum shortcut -- the lever is: add
# `@pytest.mark.slow` to the rate tests below and move them to a nightly job.
# A rate measurement validates the threshold constants, and those only move if
# the closed forms in detection.py do, so re-running them on every push earns
# little once they have passed. Mark them slow rather than weakening pre-push
# itself, per §7. EXPECTED_FALSE_ALARMS is the single knob for the cost.
#
# The counts below are chosen for ~400 expected false alarms. At that count the
# 99.9% Clopper-Pearson interval is about +/-17% relative, which a 5% error in
# the threshold factor already falls outside of -- verified by perturbing alpha
# and watching every rate test below fail.
#
# The confidence level is 99.9% rather than a conventional 95% because these are
# hypothesis tests run as a gate. A 95% interval rejects a *correct*
# implementation 5% of the time, and across the six rate tests here that is a
# 26% chance that some unlucky seed fails the suite. Seeding makes any single
# run reproducible, but it does not make the choice robust: any change to the
# order in which random numbers are drawn re-rolls it. Widening to 99.9% drops
# that to under 1% while still catching the errors worth catching.
# --------------------------------------------------------------------------- #

EXPECTED_FALSE_ALARMS = 400
CONFIDENCE = 0.999
NOMINAL_N_TRAIN = 16
NOMINAL_N_GUARD = 2

# A 77 GHz-style range-Doppler geometry, used by the end-to-end test.
NOMINAL_N_CHIRPS = 64
NOMINAL_N_SAMPLES = 256
NOMINAL_PRI_S = 50e-6
NOMINAL_WAVELENGTH_M = 3.9e-3


@pytest.fixture
def rng() -> np.random.Generator:
    """One seeded generator, so a failure is reproducible."""
    return np.random.default_rng(20260911)


def clopper_pearson_interval(n_hit: int, n_trial: int, confidence: float = CONFIDENCE):
    """Return the exact binomial confidence interval for a measured rate.

    Clopper-Pearson rather than a normal approximation: the normal interval
    under-covers badly for a rare event, and at pfa=1e-4 the count is exactly
    that. Clopper-Pearson inverts the binomial CDF, so its coverage is
    guaranteed to be at least the nominal level.
    """
    tail = 0.5 * (1.0 - confidence)
    low = 0.0 if n_hit == 0 else float(beta.ppf(tail, n_hit, n_trial - n_hit + 1))
    high = 1.0 if n_hit == n_trial else float(beta.ppf(1.0 - tail, n_hit + 1, n_trial - n_hit))
    return low, high


def measure_false_alarm_rate(
    rng: np.random.Generator,
    *,
    pfa: float,
    variant: str,
    n_train: int = NOMINAL_N_TRAIN,
    n_guard: int = NOMINAL_N_GUARD,
    noise_power_w: float = 1.0,
    n_cells: int = 4096,
):
    """Run CFAR over seeded complex Gaussian noise and count threshold crossings.

    Returns
    -------
    tuple of (int, int)
        False alarms, and the number of cells actually tested. Only cells with a
        complete reference window are counted: including the untested edge cells
        would dilute the rate towards zero and make too high a threshold look
        correct.
    """
    margin = n_guard + n_train
    tested_per_row = n_cells - 2 * margin
    n_rows = max(1, math.ceil(EXPECTED_FALSE_ALARMS / pfa / tested_per_row))

    n_hit = 0
    n_tested = 0
    # Chunked over rows to bound peak memory; the loop is over independent
    # noise realisations, so it carries no state.
    rows_per_chunk = max(1, int(2e6 // n_cells))
    remaining = n_rows
    while remaining > 0:
        rows = min(rows_per_chunk, remaining)
        # Square-law detection of complex Gaussian noise: |z|^2 with
        # z ~ CN(0, noise_power_w) is exponential with mean noise_power_w.
        real = rng.normal(0.0, math.sqrt(0.5 * noise_power_w), size=(rows, n_cells))
        imag = rng.normal(0.0, math.sqrt(0.5 * noise_power_w), size=(rows, n_cells))
        power_w = real**2 + imag**2

        mask = cfar_detect(
            power_w, pfa=pfa, n_train=n_train, n_guard=n_guard, variant=variant, axis=-1
        )
        n_hit += int(np.count_nonzero(mask))
        n_tested += rows * tested_per_row
        remaining -= rows

    return n_hit, n_tested


def assert_rate_consistent_with_design(n_hit: int, n_tested: int, pfa: float) -> None:
    """Assert the design pfa lies inside the measured confidence interval.

    A confidence interval, not an rtol: the measured rate is a Monte-Carlo
    statistic, and docs/conventions/testing.md §2 calls that out as the case
    where the tolerance table does not apply.
    """
    low, high = clopper_pearson_interval(n_hit, n_tested)
    assert low <= pfa <= high, (
        f"measured false-alarm rate {n_hit / n_tested:.3e} over {n_tested} tested cells "
        f"gives a {CONFIDENCE:.0%} interval [{low:.3e}, {high:.3e}] that excludes the "
        f"design pfa {pfa:.3e}. The threshold factor is wrong, not the tolerance."
    )


# --------------------------------------------------------------------------- #
# Analytic properties of the Pfa expressions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_train", [1, 2, 5, 16])
@pytest.mark.parametrize("alpha_linear", [0.5, 2.0, 10.0])
def test_greatest_and_smallest_of_partition_the_same_total(n_train, alpha_linear):
    """GO and SO split 2 (1 + beta)^-N between them, per Gandhi & Kassam eqs. 12-13.

    An independent identity: it constrains both expressions at once and would
    fail if either series were mis-indexed.
    """
    beta_linear = alpha_linear / n_train
    go = cfar_probability_of_false_alarm(alpha_linear, n_train=n_train, variant="go")
    so = cfar_probability_of_false_alarm(alpha_linear, n_train=n_train, variant="so")
    # rtol at 1e-12: a handful of float64 powers and a binomial sum, so anything
    # looser would hide a genuine indexing error in the series.
    np.testing.assert_allclose(go + so, 2.0 * (1.0 + beta_linear) ** -n_train, rtol=1e-12)


def test_smallest_window_matches_the_hand_derivation():
    """With n_train=1 the window is two cells, and every variant is hand-checkable.

    The reference window always holds an even 2N cells, so two is the smallest
    case. Reading it as a *single* cell gives 1/(1+alpha) and is a factor-level
    mistake that no single detection map would reveal.
    """
    alpha_linear = 9.0
    # min of two unit-mean exponentials is exponential with mean 1/2.
    expected_min = 2.0 / (2.0 + alpha_linear)
    # max of two: E[e^-a*max] = 2/(1+a) - 2/(2+a).
    expected_max = 2.0 / ((1.0 + alpha_linear) * (2.0 + alpha_linear))

    np.testing.assert_allclose(
        cfar_probability_of_false_alarm(alpha_linear, n_train=1, variant="so"),
        expected_min,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        cfar_probability_of_false_alarm(alpha_linear, n_train=1, variant="go"),
        expected_max,
        rtol=1e-12,
    )
    # With N=1 each half-window mean is one cell, so OS reproduces SO and GO.
    np.testing.assert_allclose(
        cfar_probability_of_false_alarm(alpha_linear, n_train=1, variant="os", rank=1),
        expected_min,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        cfar_probability_of_false_alarm(alpha_linear, n_train=1, variant="os", rank=2),
        expected_max,
        rtol=1e-12,
    )


def test_cell_averaging_matches_the_gamma_moment_generating_function():
    """CA-CFAR Pfa is the MGF of a Gamma(2N) sum, re-derived here from scratch.

    Independent re-derivation (testing.md §3.2) of (1 + alpha/2N)^-2N, citing
    Richards, FRSP 2e, §6.5.
    """
    for n_train in (2, 8, 16):
        n_ref = 2 * n_train
        for alpha_linear in (1.0, 7.5, 30.0):
            # P(CUT > alpha * mean) = E[exp(-alpha * Z / (2N))], Z ~ Gamma(2N, 1).
            expected = math.prod([1.0 / (1.0 + alpha_linear / n_ref)] * n_ref)
            np.testing.assert_allclose(
                cfar_probability_of_false_alarm(alpha_linear, n_train=n_train, variant="ca"),
                expected,
                rtol=1e-12,
            )


def test_closed_form_cell_averaging_inverse_agrees_with_bisection():
    """The CA closed-form inverse agrees with a bisection on its own forward map.

    CA is the one variant inverted analytically; solving it numerically instead
    is an independent route to the same alpha.
    """
    for pfa in (1e-2, 1e-4, 1e-6):
        for n_train in (4, 16):
            alpha_linear = cfar_threshold_factor(pfa=pfa, n_train=n_train, variant="ca")

            low, high = 1e-9, 1e9
            for _ in range(200):
                middle = 0.5 * (low + high)
                if cfar_probability_of_false_alarm(middle, n_train=n_train, variant="ca") > pfa:
                    low = middle
                else:
                    high = middle
            # rtol at 1e-9: bisection converges to float64 resolution on a
            # monotone function, so the two routes should agree to near-exactly.
            np.testing.assert_allclose(alpha_linear, 0.5 * (low + high), rtol=1e-9)


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
@given(
    pfa=st.floats(min_value=1e-6, max_value=1e-1, allow_nan=False, allow_infinity=False),
    n_train=st.integers(min_value=2, max_value=32),
)
@settings(deadline=None, max_examples=40)
def test_threshold_factor_round_trips_through_pfa(variant, pfa, n_train):
    """alpha(pfa) fed back through Pfa(alpha) returns the design pfa.

    This validates the inverse against the forward map and nothing more: a
    forward expression wrong by a factor round-trips perfectly. The Monte-Carlo
    rate tests are what actually pin the forward map to reality.
    """
    alpha_linear = cfar_threshold_factor(pfa=pfa, n_train=n_train, variant=variant)
    recovered = cfar_probability_of_false_alarm(alpha_linear, n_train=n_train, variant=variant)
    # rtol at 1e-8: bisection is stopped at float64 resolution in alpha, which
    # maps to a slightly larger relative error in a Pfa spanning six decades.
    np.testing.assert_allclose(recovered, pfa, rtol=1e-8)


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
def test_probability_of_false_alarm_decreases_with_threshold(variant):
    """Pfa is strictly decreasing in alpha, which is what makes the inverse unique."""
    alphas = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    values = [cfar_probability_of_false_alarm(a, n_train=8, variant=variant) for a in alphas]
    assert np.all(np.diff(values) < 0.0)


def test_threshold_factor_decreases_with_window_size():
    """A wider window needs a lower alpha for the same rate: this is the CFAR loss.

    A short window gives a noisy floor estimate, so the threshold must be raised
    to hold the design rate, and a target needs correspondingly more SNR.
    """
    alphas = [
        cfar_threshold_factor(pfa=1e-4, n_train=n, variant="ca") for n in (2, 4, 8, 16, 32, 64)
    ]
    assert np.all(np.diff(alphas) < 0.0)
    # The loss should be vanishing by 64 cells a side and large at 2.
    assert alphas[0] > 2.0 * alphas[-1]


def test_greatest_of_needs_a_lower_factor_than_smallest_of():
    """For a fixed rate, GO's threshold factor is below SO's.

    GO selects the larger half-window mean, so it already sits higher above the
    floor and needs less scaling; SO must compensate for selecting the smaller.
    """
    for n_train in (4, 16):
        go = cfar_threshold_factor(pfa=1e-4, n_train=n_train, variant="go")
        so = cfar_threshold_factor(pfa=1e-4, n_train=n_train, variant="so")
        assert go < so


def test_default_order_statistic_rank_is_three_quarters_of_the_window():
    """Rohling's k ~ 3M/4 recommendation, for M = 2N reference cells."""
    assert default_os_rank(16) == 24
    assert default_os_rank(8) == 12
    assert default_os_rank(1) == 2


# --------------------------------------------------------------------------- #
# Monte-Carlo false-alarm rate -- the central tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", ["ca", "go", "so"])
def test_false_alarm_rate_matches_design_pfa(rng, variant):
    """The measured false-alarm rate is consistent with the design pfa at 1e-3.

    The test the whole module exists to support: it is the only check here that
    would catch a threshold factor wrong by a constant factor.
    """
    pfa = 1e-3
    n_hit, n_tested = measure_false_alarm_rate(rng, pfa=pfa, variant=variant)
    assert_rate_consistent_with_design(n_hit, n_tested, pfa)


def test_order_statistic_false_alarm_rate_matches_design_pfa(rng):
    """OS-CFAR holds its design rate too, measured at a cheaper 1e-2.

    OS has no running-sum shortcut, so it is measured at a higher rate where
    fewer cells give the same number of false alarms.
    """
    pfa = 1e-2
    n_hit, n_tested = measure_false_alarm_rate(rng, pfa=pfa, variant="os")
    assert_rate_consistent_with_design(n_hit, n_tested, pfa)


def test_false_alarm_rate_holds_at_one_in_ten_thousand(rng):
    """The rate still holds at pfa=1e-4, over a few million cells.

    Small-pfa behaviour is where an error in the *exponent* -- as opposed to a
    scale factor -- shows up, since the expression is evaluated far out in its
    tail. Not marked slow: the cumulative-sum path makes this about 0.1 s.
    """
    pfa = 1e-4
    n_hit, n_tested = measure_false_alarm_rate(rng, pfa=pfa, variant="ca")
    assert_rate_consistent_with_design(n_hit, n_tested, pfa)


def test_false_alarm_rate_is_independent_of_noise_power(rng):
    """Scaling the noise floor by 1e6 leaves the false-alarm rate unchanged.

    This is the CFAR property itself, and what separates the detector from a
    fixed threshold: a fixed threshold's rate would go to one.
    """
    pfa = 1e-2
    low_hit, low_tested = measure_false_alarm_rate(rng, pfa=pfa, variant="ca", noise_power_w=1e-9)
    high_hit, high_tested = measure_false_alarm_rate(rng, pfa=pfa, variant="ca", noise_power_w=1e-3)
    assert_rate_consistent_with_design(low_hit, low_tested, pfa)
    assert_rate_consistent_with_design(high_hit, high_tested, pfa)


def test_threshold_tracks_a_noise_floor_that_varies_with_range(rng):
    """A floor sloping by 40 dB is tracked, where a fixed threshold could not.

    The practical reason CFAR exists: one threshold cannot serve both ends of a
    range profile whose floor varies with the sensitivity-time-control curve.
    """
    n_cells = 2048
    floor_w = np.logspace(-4.0, 0.0, n_cells)
    power_w = rng.exponential(floor_w)
    threshold_w = cfar_threshold_w(power_w, pfa=1e-2, n_train=32, n_guard=2)

    tested = ~np.isnan(threshold_w)
    # The threshold must ride the floor, so their ratio stays bounded across
    # four decades of floor variation.
    ratio = threshold_w[tested] / floor_w[tested]
    assert ratio.min() > 1.0
    assert ratio.max() / ratio.min() < 3.0


# --------------------------------------------------------------------------- #
# Window geometry and edge policy
# --------------------------------------------------------------------------- #


def test_valid_mask_excludes_exactly_the_incomplete_windows():
    """Cells within n_guard + n_train of an end have no full window."""
    mask = cfar_valid_mask((20,), n_train=4, n_guard=2)
    assert not mask[:6].any()
    assert mask[6:14].all()
    assert not mask[14:].any()


def test_valid_mask_is_empty_when_the_map_is_shorter_than_the_window():
    """A map too short for one complete window tests nothing at all."""
    assert not cfar_valid_mask((8,), n_train=8, n_guard=2).any()


def test_noise_estimate_is_nan_outside_the_valid_mask(rng):
    """Edge cells get nan rather than an estimate from a truncated window.

    Truncating would change Pfa in exactly the way CFAR exists to prevent.
    """
    power_w = rng.exponential(1.0, size=(4, 64))
    estimate_w = cfar_noise_estimate_w(power_w, n_train=8, n_guard=2)
    valid = cfar_valid_mask(power_w.shape, n_train=8, n_guard=2)
    assert np.all(np.isnan(estimate_w[~valid]))
    assert np.all(np.isfinite(estimate_w[valid]))


def test_edge_cells_are_never_detected():
    """Even an enormous return in an untested cell is not declared a detection."""
    power_w = np.full(64, 1.0)
    power_w[0] = 1e12
    power_w[-1] = 1e12
    mask = cfar_detect(power_w, pfa=1e-3, n_train=8, n_guard=2)
    assert not mask[0]
    assert not mask[-1]
    assert not mask.any()


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
def test_noise_estimate_recovers_a_flat_floor(variant):
    """In a constant floor every variant returns the floor exactly.

    All four reduce to the same number when the reference cells are equal, which
    makes this a check on the window indexing rather than on the statistics.
    """
    floor_w = 7.0
    power_w = np.full(64, floor_w)
    estimate_w = cfar_noise_estimate_w(power_w, n_train=8, n_guard=2, variant=variant)
    valid = cfar_valid_mask(power_w.shape, n_train=8, n_guard=2)
    np.testing.assert_allclose(estimate_w[valid], floor_w, rtol=1e-12)


def test_cell_averaging_estimate_equals_the_explicit_window_mean():
    """The cumulative-sum path reproduces a directly computed window mean.

    The O(n) cumsum implementation is the part most likely to be off by one, and
    an off-by-one in the guard band is invisible in a flat floor.
    """
    rng = np.random.default_rng(20260911)
    power_w = rng.exponential(1.0, size=200)
    n_train, n_guard = 6, 3
    estimate_w = cfar_noise_estimate_w(power_w, n_train=n_train, n_guard=n_guard, variant="ca")

    margin = n_guard + n_train
    for cut in (margin, 57, 120, len(power_w) - margin - 1):
        lagging = power_w[cut - margin : cut - n_guard]
        leading = power_w[cut + n_guard + 1 : cut + margin + 1]
        assert lagging.size == leading.size == n_train
        expected = np.concatenate([lagging, leading]).mean()
        np.testing.assert_allclose(estimate_w[cut], expected, rtol=1e-12)


def test_order_statistic_estimate_equals_the_explicit_sorted_window():
    """The sliding-window OS path reproduces a directly sorted reference window."""
    rng = np.random.default_rng(20260911)
    power_w = rng.exponential(1.0, size=200)
    n_train, n_guard, rank = 6, 3, 9
    estimate_w = cfar_noise_estimate_w(
        power_w, n_train=n_train, n_guard=n_guard, variant="os", rank=rank
    )

    margin = n_guard + n_train
    for cut in (margin, 57, 120, len(power_w) - margin - 1):
        reference = np.concatenate(
            [power_w[cut - margin : cut - n_guard], power_w[cut + n_guard + 1 : cut + margin + 1]]
        )
        np.testing.assert_allclose(estimate_w[cut], np.sort(reference)[rank - 1], rtol=1e-12)


def test_guard_cells_protect_the_threshold_from_target_spill():
    """A target spread over several cells masks itself without guard cells.

    The reason the guard band exists: energy leaking into the training cells
    inflates the very threshold meant to detect the target.
    """
    power_w = np.full(128, 1.0)
    # A target spanning five range cells, as a windowed point target would.
    power_w[60:65] = [30.0, 200.0, 600.0, 200.0, 30.0]

    with_guard = cfar_detect(power_w, pfa=1e-4, n_train=16, n_guard=4, variant="ca")
    without_guard = cfar_detect(power_w, pfa=1e-4, n_train=16, n_guard=0, variant="ca")
    assert with_guard[62]
    assert int(with_guard.sum()) >= int(without_guard.sum())


def test_smallest_of_holds_detection_against_an_interfering_target():
    """SO-CFAR detects a target that CA-CFAR misses when a second one is nearby.

    The textbook motivation for SO: an interferer in one half-window drags the
    cell-averaged floor up, while taking the smaller half ignores it.
    """
    power_w = np.full(128, 1.0)
    power_w[64] = 60.0  # the target
    power_w[72:80] = 4000.0  # a strong interferer inside the leading window

    cell_averaging = cfar_detect(power_w, pfa=1e-3, n_train=16, n_guard=2, variant="ca")
    smallest_of = cfar_detect(power_w, pfa=1e-3, n_train=16, n_guard=2, variant="so")
    assert not cell_averaging[64]
    assert smallest_of[64]


def test_greatest_of_suppresses_false_alarms_at_a_clutter_edge(rng):
    """GO-CFAR raises fewer false alarms than CA on the low side of a clutter edge.

    The textbook motivation for GO: for a cell just inside the clutter, part of
    the reference window still sits in the quiet region, so CA averages across
    the step and under-estimates the local floor. The threshold drops and the
    clutter itself produces a burst of false alarms. Taking the greater of the
    two half-windows picks the half that is actually in the clutter.
    """
    n_cells, n_rows = 512, 200
    floor_w = np.where(np.arange(n_cells) < n_cells // 2, 1.0, 1000.0)
    power_w = rng.exponential(np.broadcast_to(floor_w, (n_rows, n_cells)))

    kwargs = {"pfa": 1e-4, "n_train": 16, "n_guard": 2, "axis": -1}
    ca_mask = cfar_detect(power_w, variant="ca", **kwargs)
    go_mask = cfar_detect(power_w, variant="go", **kwargs)

    # Count the first cells inside the clutter, where the window still straddles
    # the step. This is where the burst lives; on the quiet side the same
    # straddling over-estimates the floor and causes masking instead.
    edge = slice(n_cells // 2, n_cells // 2 + 16)
    assert int(go_mask[:, edge].sum()) < int(ca_mask[:, edge].sum())


def test_detects_a_point_target_well_above_the_floor(rng):
    """A single strong cell is detected, and the surrounding noise is not."""
    power_w = rng.exponential(1.0, size=4096)
    power_w[2048] = 500.0
    mask = cfar_detect(power_w, pfa=1e-6, n_train=16, n_guard=2)
    assert mask[2048]
    assert int(mask.sum()) == 1


def test_cfar_runs_along_the_requested_axis(rng):
    """Choosing axis=0 windows down columns, not across rows."""
    power_w = rng.exponential(1.0, size=(64, 8))
    along_rows = cfar_noise_estimate_w(power_w, n_train=4, n_guard=1, axis=0)
    valid_rows = cfar_valid_mask(power_w.shape, n_train=4, n_guard=1, axis=0)
    assert np.all(np.isfinite(along_rows[valid_rows]))
    # A window along axis 0 cannot be satisfied along axis -1 here, since the
    # map is only 8 wide and the window needs 11.
    assert not cfar_valid_mask(power_w.shape, n_train=4, n_guard=1, axis=-1).any()


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


def test_clusters_two_separated_targets():
    """Two groups of crossings collapse to two detections with the right peaks."""
    power_w = np.zeros(64)
    power_w[20:23] = [1.0, 5.0, 1.0]
    power_w[50:53] = [1.0, 9.0, 1.0]
    found = cluster_detections(power_w > 0.5, power_w)

    assert len(found) == 2
    # Sorted by peak power descending, so the stronger target comes first.
    assert [d.peak_index[0] for d in found] == [51, 21]
    assert [d.n_cells for d in found] == [3, 3]
    np.testing.assert_allclose(found[0].peak_power_w, 9.0, rtol=1e-12)
    np.testing.assert_allclose(found[0].total_power_w, 11.0, rtol=1e-12)


def test_cluster_centroid_is_power_weighted():
    """A symmetric cluster centroids on its peak; a lopsided one leans towards it."""
    symmetric_w = np.zeros(32)
    symmetric_w[10:13] = [1.0, 4.0, 1.0]
    centre = cluster_detections(symmetric_w > 0.5, symmetric_w)[0].centroid_index[0]
    np.testing.assert_allclose(centre, 11.0, rtol=1e-12)

    lopsided_w = np.zeros(32)
    lopsided_w[10:13] = [1.0, 4.0, 3.0]
    leaning = cluster_detections(lopsided_w > 0.5, lopsided_w)[0].centroid_index[0]
    assert 11.0 < leaning < 12.0


def test_clusters_are_connected_in_two_dimensions():
    """A 2-D blob is one detection, and connectivity controls diagonal joining."""
    power_w = np.zeros((16, 16))
    power_w[4:6, 4:6] = 5.0
    # Touches the first blob only at a corner.
    power_w[6, 6] = 3.0
    mask = power_w > 0.5

    face_connected = cluster_detections(mask, power_w, connectivity=1)
    fully_connected = cluster_detections(mask, power_w, connectivity=2)
    assert len(face_connected) == 2
    assert len(fully_connected) == 1
    assert fully_connected[0].n_cells == 5


def test_clustering_an_empty_mask_returns_nothing():
    """No crossings means no detections, not an error."""
    assert cluster_detections(np.zeros(16, dtype=bool), np.zeros(16)) == []


def test_detection_is_immutable():
    """Detection is frozen, so a downstream tracker cannot rewrite a measurement."""
    found = cluster_detections(np.array([False, True, False]), np.array([0.0, 1.0, 0.0]))[0]
    assert isinstance(found, Detection)
    with pytest.raises(AttributeError):
        found.peak_power_w = 2.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# End to end, against analytic ground truth
# --------------------------------------------------------------------------- #


def test_detects_a_point_target_in_a_range_doppler_map(rng):
    """A synthetic target is detected in its analytically known range-Doppler bin.

    Closed-form truth (testing.md §3.1): the target is injected at a chosen
    range bin with a Doppler phase advance of an exact integer number of cycles
    across the aperture, so it lands in one Doppler bin with no straddling.
    """
    n_range_bin = 40
    n_doppler_cycles = 9

    chirp_index = np.arange(NOMINAL_N_CHIRPS)[:, None]
    sample_index = np.arange(NOMINAL_N_SAMPLES)[None, :]
    # A beat tone at bin n_range_bin in fast time, advancing by an exact
    # n_doppler_cycles over the chirp aperture in slow time.
    target = 40.0 * np.exp(
        2j
        * np.pi
        * (
            n_range_bin * sample_index / NOMINAL_N_SAMPLES
            + n_doppler_cycles * chirp_index / NOMINAL_N_CHIRPS
        )
    )
    noise = rng.normal(size=target.shape) + 1j * rng.normal(size=target.shape)
    cube = target + noise / math.sqrt(2.0)

    power_w = np.abs(range_doppler_map(cube)) ** 2
    mask = cfar_detect(power_w, pfa=1e-6, n_train=16, n_guard=4, axis=-1)
    found = cluster_detections(mask, power_w, connectivity=2)

    assert len(found) == 1
    doppler_bin, range_bin = found[0].peak_index
    assert range_bin == n_range_bin
    # range_doppler_map fftshifts the Doppler axis, so bin 0 is the most
    # negative velocity and the zero-Doppler bin sits at n_chirps // 2.
    assert doppler_bin == NOMINAL_N_CHIRPS // 2 + n_doppler_cycles

    # The detection lands at the velocity the Doppler bin centres predict.
    velocities_mps = doppler_bin_centers_mps(
        NOMINAL_N_CHIRPS,
        pulse_repetition_interval_s=NOMINAL_PRI_S,
        wavelength_m=NOMINAL_WAVELENGTH_M,
    )
    expected_mps = (
        n_doppler_cycles / NOMINAL_N_CHIRPS * NOMINAL_WAVELENGTH_M / (2.0 * NOMINAL_PRI_S)
    )
    np.testing.assert_allclose(velocities_mps[doppler_bin], expected_mps, rtol=1e-12)


# --------------------------------------------------------------------------- #
# Documented failure modes
# --------------------------------------------------------------------------- #


def test_rejects_an_unknown_variant():
    with pytest.raises(ValueError, match="variant must be one of"):
        cfar_probability_of_false_alarm(10.0, n_train=8, variant="median")  # type: ignore[arg-type]


@pytest.mark.parametrize("pfa", [0.0, 1.0, -0.1, 1.5])
def test_rejects_a_probability_outside_the_unit_interval(pfa):
    with pytest.raises(ValueError, match="pfa must be a probability"):
        cfar_threshold_factor(pfa=pfa, n_train=8)


@pytest.mark.parametrize("alpha_linear", [0.0, -1.0, np.nan, np.inf])
def test_rejects_a_non_positive_threshold_factor(alpha_linear):
    with pytest.raises(ValueError, match="alpha_linear must be"):
        cfar_probability_of_false_alarm(alpha_linear, n_train=8)


@pytest.mark.parametrize("rank", [0, -1, 17])
def test_rejects_a_rank_outside_the_reference_window(rank):
    with pytest.raises(ValueError, match="rank must be a one-based index"):
        cfar_probability_of_false_alarm(10.0, n_train=8, variant="os", rank=rank)


def test_rejects_a_rank_outside_the_reference_window_when_calibrating():
    """A bad rank is caught by the calibration too, not only by the forward map.

    cfar_threshold_factor validates rank up front rather than letting it surface
    from inside the bisection, where the message would be far less clear.
    """
    with pytest.raises(ValueError, match="rank must be a one-based index"):
        cfar_threshold_factor(pfa=1e-3, n_train=8, variant="os", rank=99)


def test_noise_estimate_is_all_nan_when_the_map_is_too_short():
    """A map shorter than one full window yields no estimates and no detections."""
    power_w = np.ones(8)
    estimate_w = cfar_noise_estimate_w(power_w, n_train=8, n_guard=2)
    assert np.all(np.isnan(estimate_w))
    assert not cfar_detect(power_w, pfa=1e-3, n_train=8, n_guard=2).any()


def test_rejects_an_empty_training_window():
    with pytest.raises(ValueError, match="n_train must be at least 1"):
        cfar_threshold_factor(pfa=1e-3, n_train=0)


def test_rejects_a_negative_guard_band():
    with pytest.raises(ValueError, match="n_guard must be non-negative"):
        cfar_noise_estimate_w(np.ones(32), n_train=4, n_guard=-1)


def test_rejects_negative_power():
    """A negative entry means an amplitude or a dB value was passed as a power."""
    with pytest.raises(ValueError, match="power_w must be non-negative"):
        cfar_detect(np.array([-1.0] * 32), pfa=1e-3, n_train=4)


def test_rejects_a_scalar_map():
    with pytest.raises(ValueError, match="at least one dimension"):
        cfar_noise_estimate_w(1.0, n_train=4)


def test_rejects_mismatched_cluster_shapes():
    with pytest.raises(ValueError, match="does not match"):
        cluster_detections(np.zeros(8, dtype=bool), np.zeros(9))


def test_rejects_an_out_of_range_connectivity():
    with pytest.raises(ValueError, match="connectivity must be"):
        cluster_detections(np.zeros((4, 4), dtype=bool), np.zeros((4, 4)), connectivity=3)


def test_rejects_an_unreachable_false_alarm_rate():
    """A tiny window cannot reach an arbitrarily low rate, and says so."""
    with pytest.raises(ValueError, match="no threshold factor below"):
        cfar_threshold_factor(pfa=1e-300, n_train=1, variant="so")


def test_rejects_an_out_of_bounds_axis():
    with pytest.raises(ValueError, match="out of bounds"):
        cfar_noise_estimate_w(np.ones((4, 32)), n_train=4, axis=5)
