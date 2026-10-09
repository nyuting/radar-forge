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
    cfar_detect_2d,
    cfar_noise_estimate_2d_w,
    cfar_noise_estimate_w,
    cfar_probability_of_false_alarm,
    cfar_threshold_2d_w,
    cfar_threshold_factor,
    cfar_threshold_w,
    cfar_valid_mask,
    cfar_valid_mask_2d,
    cluster_detections,
    default_os_rank,
)
from radar_forge.core.dsp import doppler_bin_centers_mps, range_doppler_map
from radar_forge.core.windows import taper

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
NOMINAL_N_PULSES = 64
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
    """GO and SO split 2 (1 + beta)^-N between them.

    {max, min} = {X, Y} pointwise, so E[exp(-beta max)] + E[exp(-beta min)] =
    E[exp(-beta X)] + E[exp(-beta Y)] = 2 (1 + beta)^-N.

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

    pulse_index = np.arange(NOMINAL_N_PULSES)[:, None]
    sample_index = np.arange(NOMINAL_N_SAMPLES)[None, :]
    # A beat tone at bin n_range_bin in fast time, advancing by an exact
    # n_doppler_cycles over the chirp aperture in slow time.
    target = 40.0 * np.exp(
        2j
        * np.pi
        * (
            n_range_bin * sample_index / NOMINAL_N_SAMPLES
            + n_doppler_cycles * pulse_index / NOMINAL_N_PULSES
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
    # negative velocity and the zero-Doppler bin sits at n_pulses // 2.
    assert doppler_bin == NOMINAL_N_PULSES // 2 + n_doppler_cycles

    # The detection lands at the velocity the Doppler bin centres predict.
    velocities_mps = doppler_bin_centers_mps(
        NOMINAL_N_PULSES,
        pulse_repetition_interval_s=NOMINAL_PRI_S,
        wavelength_m=NOMINAL_WAVELENGTH_M,
    )
    expected_mps = (
        n_doppler_cycles / NOMINAL_N_PULSES * NOMINAL_WAVELENGTH_M / (2.0 * NOMINAL_PRI_S)
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


# --------------------------------------------------------------------------- #
# Two-dimensional CFAR (spec 003 §13.2)
#
# The ring is (doppler, range) throughout: RING_N_TRAIN = (2, 4) and
# RING_N_GUARD = (1, 2) give margins (3, 6) and a ring of
# 7 * 13 - 3 * 5 = 76 cells.
# --------------------------------------------------------------------------- #

RING_N_TRAIN = (2, 4)
RING_N_GUARD = (1, 2)
RING_N_REFERENCE = 76


def explicit_ring(power_w, cut, *, wrap_doppler):
    """Return the ring cells around ``cut`` of a (n_doppler, n_range) map, one by one.

    Deliberately the plainest possible enumeration, so it shares no code, and no
    off-by-one, with the box filters under test. The loops are the point.
    """
    margin_doppler = RING_N_GUARD[0] + RING_N_TRAIN[0]
    margin_range = RING_N_GUARD[1] + RING_N_TRAIN[1]
    n_doppler = power_w.shape[0]
    cells = []
    for offset_doppler in range(-margin_doppler, margin_doppler + 1):
        for offset_range in range(-margin_range, margin_range + 1):
            in_guard = (
                abs(offset_doppler) <= RING_N_GUARD[0] and abs(offset_range) <= RING_N_GUARD[1]
            )
            if in_guard:
                continue
            row = cut[0] + offset_doppler
            if wrap_doppler:
                row %= n_doppler
            cells.append(power_w[row, cut[1] + offset_range])
    return np.array(cells)


def measure_false_alarm_rate_2d(rng, *, pfa, variant, n_maps_per_chunk=8):
    """Run a Doppler-wrapping 2-D CFAR over seeded noise maps and count crossings.

    The maps are stacked on a leading axis, which also checks that the ring
    works on the last two axes of a 3-D array. Neighbouring cells share most of
    their rings, so their decisions are slightly dependent; the 1-D rate tests
    above make the same approximation, and the 99.9% interval absorbs it.
    """
    shape = (64, 256)
    n_tested_per_map = int(
        cfar_valid_mask_2d(shape, n_train=RING_N_TRAIN, n_guard=RING_N_GUARD, wrap_axes=(0,)).sum()
    )
    n_maps = max(1, math.ceil(EXPECTED_FALSE_ALARMS / pfa / n_tested_per_map))

    n_hit = 0
    # Chunked over maps to bound peak memory; the maps are independent.
    for start in range(0, n_maps, n_maps_per_chunk):
        n_chunk = min(n_maps_per_chunk, n_maps - start)
        power_w = rng.exponential(1.0, size=(n_chunk, *shape))
        mask = cfar_detect_2d(
            power_w,
            pfa=pfa,
            n_train=RING_N_TRAIN,
            n_guard=RING_N_GUARD,
            variant=variant,
            wrap_axes=(-2,),
        )
        n_hit += int(np.count_nonzero(mask))
    return n_hit, n_maps * n_tested_per_map


def test_ring_mean_equals_the_explicit_ring_mean():
    """The ring mean matches the explicit ring, at the centre and across the wrap.

    The cells at Doppler row 0 and at the last row take half their ring from the
    far end of the map, which is where an off-by-one in the padding would show.
    """
    rng = np.random.default_rng(20261006)
    power_w = rng.exponential(1.0, size=(24, 40))
    estimate_w = cfar_noise_estimate_2d_w(
        power_w, n_train=RING_N_TRAIN, n_guard=RING_N_GUARD, wrap_axes=(0,)
    )
    for cut in [(12, 20), (0, 6), (23, 33), (1, 7)]:
        ring = explicit_ring(power_w, cut, wrap_doppler=True)
        assert ring.size == RING_N_REFERENCE
        # Roundoff only: running sums of a few hundred unit-mean terms.
        np.testing.assert_allclose(estimate_w[cut], ring.mean(), rtol=1e-12)


def test_ring_order_statistic_equals_the_explicit_sorted_ring():
    """The OS ring picks the rank-th smallest of exactly the ring's cells."""
    rng = np.random.default_rng(20261006)
    power_w = rng.exponential(1.0, size=(24, 40))
    rank = 50
    estimate_w = cfar_noise_estimate_2d_w(
        power_w,
        n_train=RING_N_TRAIN,
        n_guard=RING_N_GUARD,
        variant="os",
        rank=rank,
        wrap_axes=(0,),
    )
    for cut in [(12, 20), (0, 6), (23, 33)]:
        ring = explicit_ring(power_w, cut, wrap_doppler=True)
        # Exact: an order statistic selects one of the inputs.
        assert estimate_w[cut] == np.sort(ring)[rank - 1]


@pytest.mark.parametrize("variant", ["ca", "os"])
def test_ring_follows_its_axes_when_the_map_is_transposed(variant):
    """A (range, doppler) map with axes=(-1, -2) gives the transposed estimate."""
    rng = np.random.default_rng(20261006)
    power_w = rng.exponential(1.0, size=(24, 40))
    settings = {"n_train": RING_N_TRAIN, "n_guard": RING_N_GUARD, "variant": variant}
    estimate_w = cfar_noise_estimate_2d_w(power_w, wrap_axes=(0,), **settings)
    transposed_w = cfar_noise_estimate_2d_w(power_w.T, axes=(-1, -2), wrap_axes=(-1,), **settings)
    np.testing.assert_array_equal(np.isnan(transposed_w), np.isnan(estimate_w.T))
    # Roundoff only: the same sums, accumulated along the other axis first.
    np.testing.assert_allclose(transposed_w, estimate_w.T, rtol=1e-12)


def test_valid_mask_2d_tests_every_doppler_row_but_not_the_range_edges():
    """With Doppler wrapping, only the range margins go untested."""
    wrapped = cfar_valid_mask_2d(
        (8, 20), n_train=RING_N_TRAIN, n_guard=RING_N_GUARD, wrap_axes=(0,)
    )
    expected = np.zeros((8, 20), dtype=bool)
    expected[:, 6:14] = True
    np.testing.assert_array_equal(wrapped, expected)

    unwrapped = cfar_valid_mask_2d((8, 20), n_train=RING_N_TRAIN, n_guard=RING_N_GUARD)
    expected[:3] = False
    expected[-3:] = False
    np.testing.assert_array_equal(unwrapped, expected)


def test_valid_mask_2d_matches_where_the_estimate_is_defined():
    rng = np.random.default_rng(20261006)
    power_w = rng.exponential(1.0, size=(16, 40))
    for wrap_axes in [(), (0,), (0, 1)]:
        estimate_w = cfar_noise_estimate_2d_w(
            power_w, n_train=RING_N_TRAIN, n_guard=RING_N_GUARD, wrap_axes=wrap_axes
        )
        valid = cfar_valid_mask_2d(
            power_w.shape, n_train=RING_N_TRAIN, n_guard=RING_N_GUARD, wrap_axes=wrap_axes
        )
        np.testing.assert_array_equal(valid, ~np.isnan(estimate_w))


def test_valid_mask_2d_is_empty_when_an_axis_is_shorter_than_the_ring():
    """Wrapping a ring round an axis shorter than itself would count cells twice."""
    valid = cfar_valid_mask_2d((6, 40), n_train=RING_N_TRAIN, n_guard=RING_N_GUARD, wrap_axes=(0,))
    assert not valid.any()


def test_ring_threshold_factor_satisfies_the_cell_averaging_closed_form():
    """In a unit floor the threshold is alpha, and (1 + alpha / M) ** -M is pfa.

    The closed form is the Gamma moment-generating function for M cells, which
    shares no code with the calibration, so this checks that the ring is
    calibrated for its own cell count M.

    This is the test that pins M exactly. The Monte-Carlo rate test below cannot:
    calibrating this ring for 2M cells instead of M gives a measured rate of
    about 1.15e-3 against a design 1e-3, inside its 99.9% interval, because the
    CA threshold factor changes slowly once M is large.
    """
    pfa = 1e-4
    threshold_w = cfar_threshold_2d_w(
        np.ones((32, 64)), pfa=pfa, n_train=RING_N_TRAIN, n_guard=RING_N_GUARD
    )
    alpha_linear = float(np.nanmax(threshold_w))
    # Roundoff only: a closed form evaluated at its own inverse.
    np.testing.assert_allclose(
        (1.0 + alpha_linear / RING_N_REFERENCE) ** -RING_N_REFERENCE, pfa, rtol=1e-10
    )


def test_a_ring_has_less_cfar_loss_than_a_line_of_the_same_reach():
    """The reason for a 2-D window: more reference cells for the same reach in range.

    Both windows reach 10 cells along range; the ring also draws on two Doppler
    rows each side, so its estimate is less noisy and its threshold factor lower.
    """
    floor_w = np.ones((32, 64))
    line_w = cfar_threshold_w(floor_w, pfa=1e-4, n_train=8, n_guard=2)
    ring_w = cfar_threshold_2d_w(floor_w, pfa=1e-4, n_train=(2, 8), n_guard=(1, 2))
    assert np.nanmax(ring_w) < np.nanmax(line_w)


@pytest.mark.parametrize(("variant", "pfa"), [("ca", 1e-3), ("os", 1e-2)])
def test_ring_false_alarm_rate_matches_design_pfa(rng, variant, pfa):
    """The measured false-alarm rate of the ring is consistent with the design pfa.

    The test that would catch a ring calibrated for the wrong number of cells.
    OS is measured at a cheaper 1e-2 for the same reason as in 1-D.
    """
    n_hit, n_tested = measure_false_alarm_rate_2d(rng, pfa=pfa, variant=variant)
    assert_rate_consistent_with_design(n_hit, n_tested, pfa)


def test_order_statistic_ring_holds_detection_where_cell_averaging_loses_it():
    """Four strong interferers in the ring mask a target from CA but not from OS.

    A flat floor makes the outcome exact: CA's ring mean is pulled up to about
    590, so its threshold is in the thousands; OS's 51st-smallest cell is still
    the floor.
    """
    power_w = np.ones((32, 64))
    power_w[16, 32] = 1000.0
    for interferer in [(13, 32), (19, 32), (16, 27), (16, 37)]:
        power_w[interferer] = 1e4
    settings = {"pfa": 1e-4, "n_train": (2, 4), "n_guard": (1, 1)}
    assert not cfar_detect_2d(power_w, variant="ca", **settings)[16, 32]
    assert cfar_detect_2d(power_w, variant="os", **settings)[16, 32]


# --------------------------------------------------------------------------- #
# Circular axes in clustering (spec 003 §13.5, §14.2)
# --------------------------------------------------------------------------- #


def test_a_cluster_across_a_circular_axis_is_one_detection():
    """Cells at both ends of a circular axis are one target, centred on the wrap."""
    power_w = np.zeros(64)
    power_w[[63, 0]] = 4.0
    mask = power_w > 0.5

    assert len(cluster_detections(mask, power_w)) == 2
    found = cluster_detections(mask, power_w, wrap_axes=(0,))
    assert len(found) == 1
    assert found[0].n_cells == 2
    np.testing.assert_allclose(found[0].centroid_index[0], 63.5, rtol=1e-12)


def test_wrapping_leaves_a_centroid_away_from_the_wrap_unchanged():
    """A cluster that does not straddle the wrap gets its ordinary centroid."""
    power_w = np.zeros(32)
    power_w[10:13] = [1.0, 4.0, 3.0]
    mask = power_w > 0.5
    plain = cluster_detections(mask, power_w)[0].centroid_index[0]
    wrapped = cluster_detections(mask, power_w, wrap_axes=(0,))[0].centroid_index[0]
    # Roundoff only: the same mean, taken relative to the peak.
    np.testing.assert_allclose(wrapped, plain, rtol=1e-12)


def test_diagonal_neighbours_across_the_wrap_join_only_when_fully_connected():
    """Connectivity applies across the wrap exactly as it does inside the map."""
    power_w = np.zeros((8, 8))
    power_w[7, 3] = 2.0
    power_w[0, 4] = 1.0
    mask = power_w > 0.5
    assert len(cluster_detections(mask, power_w, connectivity=1, wrap_axes=(0,))) == 2
    assert len(cluster_detections(mask, power_w, connectivity=2)) == 2
    assert len(cluster_detections(mask, power_w, connectivity=2, wrap_axes=(0,))) == 1


def test_opposite_corners_join_when_both_axes_wrap():
    """On a torus the corners (0, 0) and (n-1, n-1) are diagonal neighbours."""
    power_w = np.zeros((8, 8))
    power_w[0, 0] = 2.0
    power_w[7, 7] = 1.0
    mask = power_w > 0.5
    assert len(cluster_detections(mask, power_w, connectivity=2, wrap_axes=(0,))) == 2
    found = cluster_detections(mask, power_w, connectivity=2, wrap_axes=(0, 1))
    assert len(found) == 1
    np.testing.assert_allclose(found[0].centroid_index, (7 + 2 / 3, 7 + 2 / 3), rtol=1e-12)


def test_a_detection_carries_the_noise_estimate_at_its_peak():
    """In a flat floor the ring around the peak holds only floor cells."""
    power_w = np.full((32, 64), 2.0)
    power_w[16, 32] = 200.0
    estimate_w = cfar_noise_estimate_2d_w(power_w, n_train=(2, 4), n_guard=(1, 1))
    found = cluster_detections(power_w > 100.0, power_w, noise_estimate_w=estimate_w)
    assert len(found) == 1
    # Roundoff only: box means over a constant floor.
    np.testing.assert_allclose(found[0].cfar_noise_estimate_w, 2.0, rtol=1e-12)


def test_a_detection_without_a_noise_estimate_carries_nan():
    found = cluster_detections(np.array([False, True, False]), np.array([0.0, 1.0, 0.0]))[0]
    assert math.isnan(found.cfar_noise_estimate_w)


def test_a_target_on_the_doppler_wrap_gives_one_detection(rng):
    """End to end: one target straddling the ±v wrap comes back as one detection.

    The regression test for spec 003 §14.2, and the detection-level form of
    "one detection per target" that a tracker relies on. The target advances by
    n_pulses / 2 - 0.5 cycles across the aperture, which after the fftshift
    lies exactly halfway between the last Doppler bin and the first. A Hann
    slow-time taper keeps its sidelobes below the threshold, so the only
    question is whether the two halves of its mainlobe are joined.
    """
    n_range_bin = 40
    n_doppler_cycles = NOMINAL_N_PULSES / 2 - 0.5

    pulse_index = np.arange(NOMINAL_N_PULSES)[:, None]
    sample_index = np.arange(NOMINAL_N_SAMPLES)[None, :]
    target = 0.5 * np.exp(
        2j
        * np.pi
        * (
            n_range_bin * sample_index / NOMINAL_N_SAMPLES
            + n_doppler_cycles * pulse_index / NOMINAL_N_PULSES
        )
    )
    noise = rng.normal(size=target.shape) + 1j * rng.normal(size=target.shape)
    cube = target + noise / math.sqrt(2.0)
    power_w = np.abs(range_doppler_map(cube, slow_time_window=taper("hann", NOMINAL_N_PULSES))) ** 2

    # Doppler guard of 3 covers the Hann mainlobe, which is 2 bins either side.
    settings = {"n_train": (4, 8), "n_guard": (3, 2), "wrap_axes": (0,)}
    pfa = 1e-6
    mask = cfar_detect_2d(power_w, pfa=pfa, **settings)
    estimate_w = cfar_noise_estimate_2d_w(power_w, **settings)

    assert len(cluster_detections(mask, power_w)) == 2
    found = cluster_detections(mask, power_w, noise_estimate_w=estimate_w, wrap_axes=(0,))
    assert len(found) == 1
    assert found[0].peak_index[1] == n_range_bin
    assert found[0].peak_index[0] in (NOMINAL_N_PULSES - 1, 0)
    # The mainlobe is symmetric about the wrap, so only noise moves the centroid.
    # At this SNR (over 30 dB) that is a few hundredths of a bin; 0.1 bin is a
    # bound that a split or a one-sided centroid (off by 1 or more) cannot meet.
    assert abs(found[0].centroid_index[0] - (NOMINAL_N_PULSES - 0.5)) < 0.1

    # The peak crossed its threshold, so it exceeds its noise estimate by alpha.
    n_reference = 15 * 21 - 7 * 5
    alpha_linear = cfar_threshold_factor(pfa=pfa, n_train=n_reference // 2)
    assert found[0].peak_power_w / found[0].cfar_noise_estimate_w > alpha_linear


# --------------------------------------------------------------------------- #
# Documented failure modes of the 2-D functions and the new arguments
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", ["go", "so"])
def test_ring_rejects_the_half_window_variants(variant):
    with pytest.raises(ValueError, match="for a 2-D ring"):
        cfar_threshold_2d_w(
            np.ones((32, 64)), pfa=1e-3, n_train=(2, 4), n_guard=(1, 1), variant=variant
        )


def test_ring_rejects_a_one_dimensional_map():
    with pytest.raises(ValueError, match="at least two dimensions"):
        cfar_noise_estimate_2d_w(np.ones(64), n_train=(2, 4), n_guard=(1, 1))


def test_ring_rejects_a_repeated_axis():
    with pytest.raises(ValueError, match="two different axes"):
        cfar_noise_estimate_2d_w(np.ones((32, 64)), n_train=(2, 4), n_guard=(1, 1), axes=(1, -1))


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        # Was accepted, and the third count silently dropped.
        ("n_train", {"n_train": (2, 4, 6), "n_guard": (1, 1, 1)}),
        # Was accepted, and int() silently truncated 2.5 to 2.
        ("n_train", {"n_train": (2.5, 4), "n_guard": (1, 1)}),
        # The 1-D habit of a bare count; was a TypeError about iterating an int.
        ("n_train", {"n_train": 16, "n_guard": (1, 1)}),
        # Was zip()'s own message, which names neither argument.
        ("n_guard", {"n_train": (2, 4), "n_guard": (1, 1, 1)}),
        # Was an unpacking error.
        ("axes", {"n_train": (2, 4), "n_guard": (1, 1), "axes": (-1,)}),
    ],
)
def test_ring_rejects_a_pair_that_is_not_two_integers(name, arguments):
    """Each per-axis argument is exactly two integers, and the error names it."""
    with pytest.raises(ValueError, match=f"{name} must be two integers"):
        cfar_noise_estimate_2d_w(np.ones((32, 64)), **arguments)
    with pytest.raises(ValueError, match=f"{name} must be two integers"):
        cfar_valid_mask_2d((32, 64), **arguments)


def test_ring_accepts_numpy_integer_counts():
    """A pair read out of an array holds NumPy integers, and those are integers."""
    estimate_w = cfar_noise_estimate_2d_w(
        np.ones((32, 64)), n_train=tuple(np.array(RING_N_TRAIN)), n_guard=RING_N_GUARD
    )
    np.testing.assert_array_equal(
        ~np.isnan(estimate_w),
        cfar_valid_mask_2d((32, 64), n_train=RING_N_TRAIN, n_guard=RING_N_GUARD),
    )


def test_ring_rejects_a_wrap_axis_it_does_not_span():
    with pytest.raises(ValueError, match="must be among the ring's axes"):
        cfar_noise_estimate_2d_w(
            np.ones((3, 32, 64)), n_train=(2, 4), n_guard=(1, 1), wrap_axes=(0,)
        )


def test_ring_rejects_a_rank_outside_the_ring():
    with pytest.raises(ValueError, match="rank must be a one-based index"):
        cfar_noise_estimate_2d_w(
            np.ones((32, 64)),
            n_train=RING_N_TRAIN,
            n_guard=RING_N_GUARD,
            variant="os",
            rank=RING_N_REFERENCE + 1,
        )


@pytest.mark.parametrize("bad_power_w", [np.nan, np.inf])
def test_rejects_power_that_is_not_finite(bad_power_w):
    """A nan or inf would otherwise spread silently into every window it touches."""
    power_w = np.ones((32, 64))
    power_w[16, 32] = bad_power_w
    with pytest.raises(ValueError, match="power_w must be finite"):
        cfar_detect(power_w, pfa=1e-3, n_train=4)
    with pytest.raises(ValueError, match="power_w must be finite"):
        cfar_detect_2d(power_w, pfa=1e-3, n_train=(2, 4), n_guard=(1, 1))


def test_rejects_a_noise_estimate_of_the_wrong_shape():
    with pytest.raises(ValueError, match="noise_estimate_w shape"):
        cluster_detections(np.zeros(8, dtype=bool), np.zeros(8), noise_estimate_w=np.zeros(9))


def test_rejects_an_out_of_bounds_wrap_axis():
    with pytest.raises(ValueError, match="out of bounds"):
        cluster_detections(np.zeros((4, 4), dtype=bool), np.zeros((4, 4)), wrap_axes=(2,))
