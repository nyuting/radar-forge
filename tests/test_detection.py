"""Tests for radar_forge.detection.

The analytic tests catch algebra errors in the threshold expressions. They cannot catch a
shared misunderstanding: a forward Pfa wrong by a factor round-trips perfectly through its own
inverse. The Clopper-Pearson false-alarm-rate tests are what pin the forward map to reality,
for every variant, in 1-D and 2-D, in the interior and in the clipped edge cells.

Sizing: each rate test draws enough cells for about 400 expected false alarms, where the
99.9% Clopper-Pearson interval is about +/-17% relative, so a 5% error in alpha falls outside
it. 99.9% rather than 95% because these are gates run on every push: across the sixteen
intervals here a 95% level would fail a correct implementation more often than not.
"""

import math
import tomllib
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate, ndimage
from scipy.stats import beta, gamma

from radar_forge.detection import (
    CFAR_VARIANTS,
    Detection,
    _cfar,
    cfar_detect,
    cfar_detect_2d,
    cfar_probability_of_false_alarm,
    cfar_reference_cell_count,
    cfar_threshold_factor,
    cluster_detections,
    default_os_rank,
    label_connected_cells,
)

EXPECTED_FALSE_ALARMS = 400
CONFIDENCE = 0.999
N_TRAIN, N_GUARD = 16, 2
# The (doppler, range) ring: margins (3, 6), a 7 x 13 box less a 3 x 5 guard box.
RING = {"n_train": (2, 4), "n_guard": (1, 2)}
RING_N_REFERENCE = 76
# The cheaper design rate at which OS, which has no running-sum shortcut, is measured.
RATE_OF = {"ca": 1e-3, "go": 1e-3, "so": 1e-3, "os": 1e-2}


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260911)


def assert_rate_consistent_with_design(detected, counted, pfa):
    """The design pfa lies inside the exact binomial interval of the measured rate.

    Clopper-Pearson rather than a normal approximation, which under-covers for a rare event.
    """
    n_hit, n_trial = int(np.count_nonzero(detected[counted])), int(np.count_nonzero(counted))
    tail = 0.5 * (1.0 - CONFIDENCE)
    low = float(beta.ppf(tail, n_hit, n_trial - n_hit + 1)) if n_hit else 0.0
    high = float(beta.ppf(1.0 - tail, n_hit + 1, n_trial - n_hit))
    assert low <= pfa <= high, (
        f"measured {n_hit / n_trial:.3e} over {n_trial} cells, interval [{low:.3e}, "
        f"{high:.3e}] excludes the design {pfa:.1e}: the threshold factor is wrong."
    )


def n_maps(pfa, cells_per_map):
    return math.ceil(EXPECTED_FALSE_ALARMS / pfa / cells_per_map)


def explicit_reference_cells(power_w, cut, n_train, n_guard, wraps=(False, False)):
    """The in-map reference cells around ``cut`` of a 2-D map, enumerated one by one.

    The plainest possible loop, sharing no code and no off-by-one with the filters.
    """
    (t0, t1), (g0, g1) = n_train, n_guard
    cells = []
    for o0 in range(-(g0 + t0), g0 + t0 + 1):
        for o1 in range(-(g1 + t1), g1 + t1 + 1):
            if abs(o0) <= g0 and abs(o1) <= g1:
                continue
            row, column = cut[0] + o0, cut[1] + o1
            row = row % power_w.shape[0] if wraps[0] else row
            column = column % power_w.shape[1] if wraps[1] else column
            if 0 <= row < power_w.shape[0] and 0 <= column < power_w.shape[1]:
                cells.append(power_w[row, column])
    return np.array(cells)


# Analytic properties of the Pfa and the threshold factor


@pytest.mark.parametrize(("n_train", "alpha_linear"), [(5, 10.0), (16, 2.0), (16, 300.0)])
def test_greatest_and_smallest_of_match_a_numerical_integration(n_train, alpha_linear):
    """GO and SO against a route sharing no code with the incomplete-beta closed form.

    Each half mean is Gamma(N, 1/N), so Pfa = E[exp(-alpha g)] with g the larger (GO) or
    smaller (SO) of two independent half means, densities 2 f F and 2 f (1 - F). The third
    case, beta = 18.75, is where the old series form of GO lost digits to cancellation.
    """
    half_mean = gamma(n_train, scale=1.0 / n_train)

    def expected(weight):
        def integrand(m):
            return math.exp(-alpha_linear * m) * 2.0 * half_mean.pdf(m) * weight(m)

        return integrate.quad(integrand, 0.0, math.inf, epsabs=0.0, epsrel=1e-13, limit=200)[0]

    # rtol 1e-10: quad's own error estimate, not slack in the closed form.
    for variant, weight in (("so", half_mean.sf), ("go", half_mean.cdf)):
        pfa = cfar_probability_of_false_alarm(alpha_linear, n_train=n_train, variant=variant)
        np.testing.assert_allclose(pfa, expected(weight), rtol=1e-10)


def test_smallest_window_matches_the_hand_derivation():
    """Two cells: min is exponential with mean 1/2, so 2/(2 + a); max gives 2/((1 + a)(2 + a)).

    Reading two cells as one, 1/(1 + a), is the factor-level mistake this pins.
    """
    alpha_linear = 9.0
    minimum, maximum = (
        2.0 / (2.0 + alpha_linear),
        2.0 / ((1.0 + alpha_linear) * (2.0 + alpha_linear)),
    )
    for variant, rank, expected in (
        ("so", None, minimum),
        ("os", 1, minimum),
        ("go", None, maximum),
        ("os", 2, maximum),
    ):
        pfa = cfar_probability_of_false_alarm(alpha_linear, n_train=1, variant=variant, rank=rank)
        np.testing.assert_allclose(pfa, expected, rtol=1e-12)  # a few float64 operations


@pytest.mark.parametrize("n_reference", [2, 16, 32, 76, 1000])
def test_cell_averaging_factor_is_the_textbook_closed_form(n_reference):
    """alpha = M (Pfa^(-1/M) - 1) (Richards, FRSP 2e, §6.5), and Pfa = (1 + alpha/M)^-M back."""
    pfa = 1e-5
    alpha_linear = cfar_threshold_factor(pfa=pfa, n_train=n_reference // 2)
    # rtol 1e-12: a power and a subtraction; M = 1000 loses a few digits to 1 + tiny.
    np.testing.assert_allclose(
        alpha_linear, n_reference * (pfa ** (-1 / n_reference) - 1), rtol=1e-12
    )
    np.testing.assert_allclose(
        math.prod([1 / (1 + alpha_linear / n_reference)] * n_reference), pfa, rtol=1e-12
    )


def test_order_statistic_factor_satisfies_rohlings_product_form():
    """Rohling's eq. 14 for M = 24, k = 18 = 3M/4, at Pfa = 1e-6, by an explicit product.

    Pfa = prod_{i<k} (M - i) / (M - i + alpha), evaluated term by term, shares no code with
    the gammaln form. Its root is alpha = 16.293, which is pinned too so that a reader can
    check a hand calculation against it.
    """
    alpha_linear = cfar_threshold_factor(pfa=1e-6, n_train=12, variant="os", rank=18)
    product = math.prod((24 - i) / (24 - i + alpha_linear) for i in range(18))
    np.testing.assert_allclose(product, 1e-6, rtol=1e-9)  # Brent stops at 1e-14 in log alpha
    np.testing.assert_allclose(alpha_linear, 16.2933, atol=1e-4)


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
@given(pfa=st.floats(min_value=1e-8, max_value=0.1), n_train=st.integers(min_value=1, max_value=64))
@settings(deadline=None, max_examples=40)
def test_threshold_factor_round_trips_through_pfa(variant, pfa, n_train):
    """The inverse agrees with the forward map; the rate tests pin the forward map itself."""
    alpha_linear = cfar_threshold_factor(pfa=pfa, n_train=n_train, variant=variant)
    recovered = cfar_probability_of_false_alarm(alpha_linear, n_train=n_train, variant=variant)
    np.testing.assert_allclose(recovered, pfa, rtol=1e-9)  # Brent's xtol of 1e-14 in log alpha


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
@given(pfa=st.lists(st.floats(min_value=1e-8, max_value=0.2), min_size=2, max_size=2, unique=True))
@settings(deadline=None, max_examples=25)
def test_threshold_is_monotone_in_pfa(variant, pfa):
    """A lower design rate never lowers the threshold, in any cell, edge or interior."""
    power_w = np.random.default_rng(7).exponential(1.0, size=(6, 40))
    low, high = sorted(pfa)
    _, strict_w = cfar_detect_2d(power_w, low, variant=variant, **RING)
    _, loose_w = cfar_detect_2d(power_w, high, variant=variant, **RING)
    assert np.all(strict_w >= loose_w)


def test_threshold_factor_decreases_with_window_size():
    """A wider window needs a lower alpha for the same rate: the CFAR loss shrinks."""
    alphas = [cfar_threshold_factor(pfa=1e-4, n_train=n) for n in (2, 4, 8, 16, 32, 64)]
    assert np.all(np.diff(alphas) < 0.0)
    assert alphas[0] > 2.0 * alphas[-1]


def test_default_order_statistic_rank_is_three_quarters_of_the_window():
    assert [default_os_rank(n) for n in (1, 8, 16, RING_N_REFERENCE // 2)] == [2, 12, 24, 57]


def test_reference_cell_count_matches_the_explicit_footprint():
    ring = explicit_reference_cells(np.ones((24, 40)), (12, 20), **RING)
    assert (
        cfar_reference_cell_count(RING["n_train"], RING["n_guard"]) == ring.size == RING_N_REFERENCE
    )
    assert cfar_reference_cell_count(16, 4) == 32


def test_scenario_003_calibration_matches_its_specification():
    """spec/scenario-003-tracking.md §2 and §12: cells tested, and alpha inside and at the ends.

    CA along range, N = 16, guard 4, pfa = 1e-5 on (256, 1000). Interior M = 32:
    32 (1e5^(1/32) - 1) = 13.8564 = 11.4165 dB. The outermost n_guard + 1 = 5 cells of each
    end keep only the far band, M = 16: 16 (1e5^(1/16) - 1) = 16.8564 = 12.2676 dB. Cell 10
    keeps the far band and 6 near cells (indices 0..5), M = 22: 15.1275 = 11.7977 dB. On a
    unit floor the threshold is alpha itself.
    """
    scenario = tomllib.loads(
        (Path(__file__).parents[1] / "scenarios" / "scenario_003_tracking.toml").read_text()
    )
    window = scenario["detection"]
    burst = scenario["burst"][0]
    shape = (burst["n_pulses"], round(burst["chirp_duration_s"] * burst["sample_rate_hz"]))
    _, threshold_w = cfar_detect(
        np.ones(shape), window["pfa"], n_train=window["n_train"], n_guard=window["n_guard"]
    )
    assert np.count_nonzero(np.isfinite(threshold_w)) == 256_000
    # atol 1e-4 dB: the spec quotes three decimals.
    np.testing.assert_allclose(
        10 * np.log10(threshold_w[0, [500, 0, 4, -1, 10, -11]]),
        [11.4165, 12.2676, 12.2676, 12.2676, 11.7977, 11.7977],
        atol=1e-4,
    )


# Monte-Carlo false-alarm rate: the central tests


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
def test_interior_false_alarm_rate_matches_design_pfa(rng, variant):
    pfa, n_cells, margin = RATE_OF[variant], 4096, N_TRAIN + N_GUARD
    power_w = rng.exponential(1.0, size=(n_maps(pfa, n_cells - 2 * margin), n_cells))
    detected, _ = cfar_detect(power_w, pfa, n_train=N_TRAIN, n_guard=N_GUARD, variant=variant)
    counted = np.zeros(power_w.shape, dtype=bool)
    counted[:, margin:-margin] = True
    assert_rate_consistent_with_design(detected, counted, pfa)


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
def test_clipped_edge_false_alarm_rate_matches_design_pfa(rng, variant):
    """Rows one window long: every cell but the centre has a clipped window, counted alone."""
    pfa, margin = RATE_OF[variant], N_TRAIN + N_GUARD
    power_w = rng.exponential(1.0, size=(n_maps(pfa, 2 * margin), 2 * margin + 1))
    detected, _ = cfar_detect(power_w, pfa, n_train=N_TRAIN, n_guard=N_GUARD, variant=variant)
    counted = np.ones(power_w.shape, dtype=bool)
    counted[:, margin] = False
    assert_rate_consistent_with_design(detected, counted, pfa)


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
def test_ring_false_alarm_rate_matches_design_pfa(rng, variant):
    """A Doppler-wrapping ring over a stack of maps, every cell counted, range edges included."""
    pfa, shape = RATE_OF[variant], (64, 256)
    power_w = rng.exponential(1.0, size=(n_maps(pfa, shape[0] * shape[1]), *shape))
    detected, _ = cfar_detect_2d(power_w, pfa, variant=variant, wrap_axes=(-2,), **RING)
    assert_rate_consistent_with_design(detected, np.ones(power_w.shape, dtype=bool), pfa)


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
def test_clipped_ring_false_alarm_rate_matches_design_pfa(rng, variant):
    """7 x 13 maps, one ring each: every cell but the centre is clipped, corners hardest."""
    pfa, shape = RATE_OF[variant], (7, 13)
    power_w = rng.exponential(1.0, size=(n_maps(pfa, 90), *shape))
    detected, threshold_w = cfar_detect_2d(power_w, pfa, variant=variant, **RING)
    assert np.all(np.isfinite(threshold_w))
    counted = np.ones(power_w.shape, dtype=bool)
    counted[:, 3, 6] = False
    assert_rate_consistent_with_design(detected, counted, pfa)


def test_false_alarm_rate_is_independent_of_noise_power(rng):
    """The CFAR property itself: a floor of 1e-9 W or 1e-3 W, the same rate."""
    for noise_power_w in (1e-9, 1e-3):
        power_w = rng.exponential(noise_power_w, size=(n_maps(1e-2, 1024), 1024))
        detected, _ = cfar_detect(power_w, 1e-2, n_train=N_TRAIN, n_guard=N_GUARD)
        assert_rate_consistent_with_design(detected, np.ones(power_w.shape, dtype=bool), 1e-2)


@pytest.mark.parametrize("dimensions", [1, 2])
def test_a_tapered_map_breaks_the_calibration_as_the_module_notes_say(rng, dimensions):
    """Hann on both axes correlates neighbouring cells, so the rate exceeds pfa.

    Measured at 1.65 to 1.9 times pfa over three seeds, line and ring alike, so the lower end
    of the 99.9% interval clears pfa by a wide margin.
    """
    noise = rng.normal(size=(30, 64, 256)) + 1j * rng.normal(size=(30, 64, 256))
    taper = np.hanning(64)[:, None] * np.hanning(256)
    power_w = np.abs(np.fft.fft2(noise * taper)) ** 2
    if dimensions == 1:
        detected, _ = cfar_detect(power_w, 1e-3, n_train=16, n_guard=4)
    else:
        detected, _ = cfar_detect_2d(power_w, 1e-3, wrap_axes=(-2,), **RING)
    n_hit = int(np.count_nonzero(detected))
    low = float(beta.ppf(0.5 * (1 - CONFIDENCE), n_hit, detected.size - n_hit + 1))
    assert low > 1e-3, f"measured {n_hit / detected.size:.3e}, interval from {low:.3e}"


# The engine: footprint, halves, edge policy, wrap


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
@pytest.mark.parametrize("axis", [-1, -2])
def test_one_dimensional_window_equals_the_ring_with_a_degenerate_axis(rng, variant, axis):
    """cfar_detect along an axis is cfar_detect_2d with zero training and guard on the other."""
    power_w = rng.exponential(1.0, size=(30, 40))
    _, line_w = cfar_detect(power_w, 1e-3, n_train=5, n_guard=2, variant=variant, axis=axis)
    pair = (
        {"n_train": (0, 5), "n_guard": (0, 2)}
        if axis == -1
        else {"n_train": (5, 0), "n_guard": (2, 0)}
    )
    _, ring_w = cfar_detect_2d(power_w, 1e-3, variant=variant, **pair)
    np.testing.assert_allclose(ring_w, line_w, rtol=1e-14)  # the same cells summed in order


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
@pytest.mark.parametrize("dimensions", [1, 2])
def test_a_wrapped_axis_equals_the_engine_on_a_periodically_extended_map(rng, variant, dimensions):
    """Wrapping is exactly padding with wrapped cells, running unwrapped, and cropping."""
    power_w = rng.exponential(1.0, size=(24, 48))
    if dimensions == 1:
        window, margin = {"n_train": 6, "n_guard": 2, "axis": 0}, 8
        _, wrapped_w = cfar_detect(power_w, 1e-2, variant=variant, wrap=True, **window)
        _, extended_w = cfar_detect(
            np.pad(power_w, ((margin, margin), (0, 0)), mode="wrap"),
            1e-2,
            variant=variant,
            **window,
        )
    else:
        margin = 3
        _, wrapped_w = cfar_detect_2d(power_w, 1e-2, variant=variant, wrap_axes=(0,), **RING)
        _, extended_w = cfar_detect_2d(
            np.pad(power_w, ((margin, margin), (0, 0)), mode="wrap"), 1e-2, variant=variant, **RING
        )
    np.testing.assert_array_equal(wrapped_w, extended_w[margin:-margin])


def test_an_impulse_is_detected_and_its_guard_cells_are_not_training():
    """Guard cells loaded with 500 W leave the threshold at exactly alpha on a unit floor.

    One cell further out the same spill is training and lifts the threshold above the target.
    """
    power_w = np.ones(101)
    power_w[50] = 400.0
    power_w[[48, 49, 51, 52]] = 500.0  # the whole guard band, n_guard = 2
    detected, threshold_w = cfar_detect(power_w, 1e-4, n_train=4, n_guard=2)
    assert detected[50]
    np.testing.assert_allclose(
        threshold_w[50], cfar_threshold_factor(pfa=1e-4, n_train=4), rtol=1e-14
    )
    power_w[53] = 500.0  # now inside the leading training band
    assert not cfar_detect(power_w, 1e-4, n_train=4, n_guard=2)[0][50]


@pytest.mark.parametrize("cut", [(0, 0), (0, 20), (1, 3), (6, 39), (12, 20)])
def test_clipped_cell_averaging_is_the_mean_of_the_in_map_cells(rng, cut):
    """Corner, edge and interior: the mean of exactly the in-map cells, alpha for their count."""
    power_w = (
        rng.exponential(1.0, size=(7, 40)) if cut[0] < 7 else rng.exponential(1.0, size=(24, 40))
    )
    threshold_w, estimate_w, alpha_linear = _cfar(
        power_w, 1e-4, (2, 4), (1, 2), "ca", None, (0, 1), (False, False)
    )
    cells = explicit_reference_cells(power_w, cut, **RING)
    np.testing.assert_allclose(estimate_w[cut], cells.mean(), rtol=1e-13)  # roundoff in a sum
    m = cells.size
    np.testing.assert_allclose(alpha_linear[cut], m * (1e-4 ** (-1 / m) - 1), rtol=1e-12)
    np.testing.assert_allclose(threshold_w[cut], alpha_linear[cut] * estimate_w[cut], rtol=1e-15)


@pytest.mark.parametrize("cut", [(0, 0), (0, 20), (6, 39), (3, 20)])
def test_clipped_order_statistic_takes_the_shifted_rank_of_the_in_map_cells(rng, cut):
    """Missing cells read as zero, so rank k of M lands on rank k - (M - M_c) of the M_c cells.

    The factor is checked against Rohling's product for that rank and count.
    """
    power_w = rng.exponential(1.0, size=(7, 40))
    _, estimate_w, alpha_linear = _cfar(
        power_w, 1e-3, (2, 4), (1, 2), "os", None, (0, 1), (False, False)
    )
    cells = np.sort(explicit_reference_cells(power_w, cut, **RING))
    rank = default_os_rank(RING_N_REFERENCE // 2) - (RING_N_REFERENCE - cells.size)
    assert estimate_w[cut] == cells[rank - 1]  # a selection, so exact
    pfa = math.prod((cells.size - i) / (cells.size - i + alpha_linear[cut]) for i in range(rank))
    np.testing.assert_allclose(pfa, 1e-3, rtol=1e-9)


def first_merged_engine_estimate_w(power_w, variant):
    """The ring's noise estimate as the engine first computed it, rebuilt from scipy alone.

    Every variant took two half-footprint correlate passes, halves written out by offset
    here, and OS a rank filter over the ring. Doppler (axis -2) wraps; range is clipped.
    """
    o0, o1 = np.meshgrid(np.arange(-3, 4), np.arange(-6, 7), indexing="ij")
    ring = ~((abs(o0) <= 1) & (abs(o1) <= 2))
    halves = [
        ring & ((o1 > 0) | ((o1 == 0) & (o0 > 0))),
        ring & ((o1 < 0) | ((o1 == 0) & (o0 < 0))),
    ]
    padded_w = np.pad(power_w, ((0, 0), (3, 3), (0, 0)), mode="wrap")
    ones = np.ones(padded_w.shape[1:])
    sums_w = [ndimage.correlate(padded_w, h[None] * 1.0, mode="constant")[:, 3:-3] for h in halves]
    counts = [np.rint(ndimage.correlate(ones, h * 1.0, mode="constant")[3:-3]) for h in halves]
    n_cells = counts[0] + counts[1]
    if variant == "os":
        rank = default_os_rank(RING_N_REFERENCE // 2)
        return ndimage.rank_filter(padded_w, rank - 1, footprint=ring[None], mode="constant")[
            :, 3:-3
        ]
    mean_w = (sums_w[0] + sums_w[1]) / n_cells
    if variant == "ca":
        return mean_w
    pick = np.maximum if variant == "go" else np.minimum
    halves_w = pick(sums_w[0] / counts[0], sums_w[1] / counts[1])
    return np.where(n_cells == RING_N_REFERENCE, halves_w, mean_w)


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
def test_one_pass_engine_matches_the_first_merged_engine(rng, variant):
    """CA now sums the whole ring in one pass, and OS and CA no longer compute half sums.

    GO, SO and OS select or divide the same numbers, so they are bit for bit the same. CA adds
    the 76 ring cells in one grouping instead of two half sums, which moves the last bits:
    measured at 1.3e-15 relative at most, so rtol 4e-15 (about 18 ulp).
    """
    power_w = rng.exponential(1.0, size=(3, 24, 60))
    window = ((2, 4), (1, 2), variant, None, (1, 2), (True, False))
    threshold_w, estimate_w, alpha_linear = _cfar(power_w, 1e-3, *window)
    expected_w = first_merged_engine_estimate_w(power_w, variant)
    if variant == "ca":
        np.testing.assert_allclose(estimate_w, expected_w, rtol=4e-15)
    else:
        np.testing.assert_array_equal(estimate_w, expected_w)
    np.testing.assert_array_equal(threshold_w, alpha_linear * estimate_w)


def test_an_order_statistic_rank_with_no_cells_left_is_not_tested():
    """Rank 1 of 32 has no rank left once one cell is missing: every clipped cell goes untested.

    With 18 cells from the CUT to the window's edge, that is the first and last 18; nan
    thresholds compare False, so even 1e9 W there is not detected.
    """
    power_w = np.ones(64)
    power_w[0] = 1e9
    detected, threshold_w = cfar_detect(power_w, 1e-3, n_train=16, n_guard=2, variant="os", rank=1)
    clipped = (np.arange(64) < 18) | (np.arange(64) >= 46)
    np.testing.assert_array_equal(np.isnan(threshold_w), clipped)
    assert not detected.any()


@pytest.mark.parametrize("variant", ["go", "so"])
def test_greatest_and_smallest_of_reduce_to_cell_averaging_where_clipped(rng, variant):
    power_w = rng.exponential(1.0, size=(4, 60))
    _, half_w = cfar_detect(power_w, 1e-3, n_train=8, n_guard=2, variant=variant)
    _, mean_w = cfar_detect(power_w, 1e-3, n_train=8, n_guard=2, variant="ca")
    # rtol 4e-15: CA sums the window in one pass, GO/SO's fallback as two half sums.
    np.testing.assert_allclose(half_w[:, :10], mean_w[:, :10], rtol=4e-15)
    np.testing.assert_allclose(half_w[:, -10:], mean_w[:, -10:], rtol=4e-15)
    assert not np.allclose(half_w[:, 10:-10], mean_w[:, 10:-10], rtol=1e-6)


def test_the_halves_split_on_the_last_axis_then_on_the_first():
    """Leading half: range offset > 0, plus the CUT's own column below it. Mirror for lagging.

    Two loud cells in the centre column below the CUT (Doppler offsets +2, +3, outside the
    guard) belong to the leading half alone: GO sees them, SO does not, and each half has 38.
    """
    power_w = np.ones((16, 32))
    power_w[10:12, 16] = 39.0
    _, estimate_go, _ = _cfar(power_w, 1e-3, (2, 4), (1, 2), "go", None, (0, 1), (False, False))
    _, estimate_so, _ = _cfar(power_w, 1e-3, (2, 4), (1, 2), "so", None, (0, 1), (False, False))
    np.testing.assert_allclose(estimate_go[8, 16], (36 + 2 * 39) / 38, rtol=1e-14)
    assert estimate_so[8, 16] == 1.0


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
def test_threshold_scales_exactly_with_the_noise_floor(rng, variant):
    """Scale by 2**20, an exact float scaling: the threshold scales and the mask cannot move."""
    power_w = rng.exponential(1.0, size=(32, 64))
    detected, threshold_w = cfar_detect_2d(power_w, 1e-2, variant=variant, wrap_axes=(0,), **RING)
    scaled, scaled_w = cfar_detect_2d(
        2.0**20 * power_w, 1e-2, variant=variant, wrap_axes=(0,), **RING
    )
    np.testing.assert_array_equal(scaled_w, 2.0**20 * threshold_w)
    np.testing.assert_array_equal(scaled, detected)


def test_threshold_tracks_a_floor_sloping_by_40_db(rng):
    floor_w = np.logspace(-4.0, 0.0, 2048)
    _, threshold_w = cfar_detect(rng.exponential(floor_w), 1e-2, n_train=32, n_guard=2)
    ratio = threshold_w / floor_w
    assert ratio.min() > 1.0 and ratio.max() / ratio.min() < 3.0


def test_a_ring_has_less_cfar_loss_than_a_line_of_the_same_reach():
    _, line_w = cfar_detect(np.ones((32, 64)), 1e-4, n_train=8, n_guard=2)
    _, ring_w = cfar_detect_2d(np.ones((32, 64)), 1e-4, n_train=(2, 8), n_guard=(1, 2))
    assert ring_w[16, 32] < line_w[16, 32]


# Detection behaviour the variants exist for


@pytest.mark.parametrize("variant", CFAR_VARIANTS)
def test_a_target_in_the_first_and_last_cells_is_detected(rng, variant):
    power_w = rng.exponential(1.0, size=256)
    power_w[[0, -1]] += 1e3
    detected, _ = cfar_detect(power_w, 1e-6, n_train=8, n_guard=2, variant=variant)
    assert np.flatnonzero(detected).tolist() == [0, 255]


def test_guard_cells_protect_the_threshold_from_target_spill():
    """Without a guard the target's skirts are training: 17.30 * 464 / 8 = 1003 > 600."""
    power_w = np.ones(128)
    power_w[60:65] = [30.0, 200.0, 600.0, 200.0, 30.0]
    assert cfar_detect(power_w, 1e-4, n_train=4, n_guard=2)[0][62]
    assert not cfar_detect(power_w, 1e-4, n_train=4, n_guard=0)[0].any()


def test_smallest_of_holds_detection_against_an_interfering_target():
    power_w = np.ones(128)
    power_w[64] = 60.0
    power_w[72:80] = 4000.0  # inside the leading band
    assert not cfar_detect(power_w, 1e-3, n_train=16, n_guard=2, variant="ca")[0][64]
    assert cfar_detect(power_w, 1e-3, n_train=16, n_guard=2, variant="so")[0][64]


def test_greatest_of_suppresses_false_alarms_at_a_clutter_edge(rng):
    """Just inside a 30 dB clutter step CA averages across it and fires; GO takes the clutter."""
    floor_w = np.where(np.arange(512) < 256, 1.0, 1000.0)
    power_w = rng.exponential(floor_w, size=(200, 512))
    ca, _ = cfar_detect(power_w, 1e-4, n_train=16, n_guard=2, variant="ca")
    go, _ = cfar_detect(power_w, 1e-4, n_train=16, n_guard=2, variant="go")
    assert int(go[:, 256:272].sum()) < int(ca[:, 256:272].sum())


def test_order_statistic_ring_holds_detection_where_cell_averaging_loses_it():
    power_w = np.ones((32, 64))
    power_w[16, 32] = 1000.0
    power_w[[13, 19, 16, 16], [32, 32, 27, 37]] = 1e4
    window = {"n_train": (2, 4), "n_guard": (1, 1)}
    assert not cfar_detect_2d(power_w, 1e-4, variant="ca", **window)[0][16, 32]
    assert cfar_detect_2d(power_w, 1e-4, variant="os", **window)[0][16, 32]


def test_a_doppler_spread_target_masks_itself_unless_the_doppler_guard_covers_it():
    """Five Doppler bins at 1000: (10 + 4000)/14 * 13.0 = 3727 > 1000 with no Doppler guard."""
    power_w = np.ones((32, 64))
    power_w[14:19, 32] = 1000.0
    assert not cfar_detect_2d(power_w, 1e-4, n_train=(2, 1), n_guard=(0, 0))[0][16, 32]
    assert cfar_detect_2d(power_w, 1e-4, n_train=(2, 1), n_guard=(2, 0))[0][16, 32]


# Clustering


def test_two_separated_blobs_give_two_detections_at_their_centroids():
    """By hand: rows 4, 5 weigh 4 and 12, so 4.75; columns 10, 11, 12 weigh 4, 8, 4, so 11."""
    power_w = np.zeros((16, 32))
    power_w[4:6, 10:13] = [[1.0, 2.0, 1.0], [3.0, 6.0, 3.0]]
    power_w[12, 25] = 9.0
    found = cluster_detections(power_w > 0, power_w)
    assert [d.peak_index for d in found] == [(12, 25), (5, 11)]
    assert [d.n_cells for d in found] == [1, 6]
    np.testing.assert_allclose(found[1].centroid_index, (4.75, 11.0), rtol=1e-14)
    np.testing.assert_allclose(found[1].total_power_w, 16.0, rtol=1e-14)


def test_a_blob_straddling_a_wrapped_axis_is_one_detection():
    """Rows 15 and 0 of 16 at powers 1 and 3: centroid 15 + 3/4 = 15.75, one cluster."""
    power_w = np.zeros((16, 8))
    power_w[[15, 0], 4] = [1.0, 3.0]
    assert len(cluster_detections(power_w > 0, power_w)) == 2
    found = cluster_detections(power_w > 0, power_w, wrap_axes=(0,))
    assert len(found) == 1
    np.testing.assert_allclose(found[0].centroid_index, (15.75, 4.0), rtol=1e-14)


def test_wrapping_leaves_a_centroid_away_from_the_wrap_unchanged():
    """Cells 10, 11, 12 at 1, 4, 3: (10 + 44 + 36) / 8 = 11.25, wrapped or not."""
    power_w = np.zeros(32)
    power_w[10:13] = [1.0, 4.0, 3.0]
    for wrap_axes in [(), (0,)]:
        centroid = cluster_detections(power_w > 0, power_w, wrap_axes=wrap_axes)[0].centroid_index
        np.testing.assert_allclose(centroid, (11.25,), rtol=1e-14)


def test_labels_are_the_connected_components_of_the_mask():
    mask = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 0]], dtype=bool)
    labels, n_groups = label_connected_cells(mask)
    assert n_groups == 3
    np.testing.assert_array_equal(labels, [[1, 0, 0, 2], [1, 1, 0, 0], [0, 0, 3, 0]])
    labels, n_groups = label_connected_cells(mask, connectivity=2)
    assert n_groups == 2
    np.testing.assert_array_equal(labels, [[1, 0, 0, 2], [1, 1, 0, 0], [0, 0, 1, 0]])


def test_diagonal_neighbours_across_the_wrap_join_only_when_fully_connected():
    mask = np.zeros((8, 8), dtype=bool)
    mask[7, 3] = mask[0, 4] = True
    assert label_connected_cells(mask, connectivity=1, wrap_axes=(0,))[1] == 2
    assert label_connected_cells(mask, connectivity=2)[1] == 2
    assert label_connected_cells(mask, connectivity=2, wrap_axes=(0,))[1] == 1


def test_opposite_corners_join_when_both_axes_wrap():
    power_w = np.zeros((8, 8))
    power_w[0, 0], power_w[7, 7] = 2.0, 1.0
    assert len(cluster_detections(power_w > 0, power_w, connectivity=2, wrap_axes=(0,))) == 2
    found = cluster_detections(power_w > 0, power_w, connectivity=2, wrap_axes=(0, 1))
    np.testing.assert_allclose(found[0].centroid_index, (7 + 2 / 3, 7 + 2 / 3), rtol=1e-14)


def test_each_detection_summarises_exactly_one_label(rng):
    power_w = rng.exponential(1.0, size=(32, 64))
    labels, n_groups = label_connected_cells(power_w > 3.0, wrap_axes=(0,))
    found = cluster_detections(power_w > 3.0, power_w, wrap_axes=(0,))
    assert len(found) == n_groups
    assert sorted(d.n_cells for d in found) == sorted(np.bincount(labels.ravel())[1:].tolist())
    for detection in found:
        assert detection.peak_power_w == power_w[labels == labels[detection.peak_index]].max()
    assert [d.peak_power_w for d in found] == sorted((d.peak_power_w for d in found), reverse=True)


def test_a_detection_carries_the_threshold_at_its_peak_or_nan():
    power_w = np.full((32, 64), 2.0)
    power_w[16, 32] = 200.0
    detected, threshold_w = cfar_detect_2d(power_w, 1e-4, n_train=(2, 4), n_guard=(1, 1))
    (found,) = cluster_detections(detected, power_w, threshold_w=threshold_w)
    alpha_linear = cfar_threshold_factor(
        pfa=1e-4, n_train=cfar_reference_cell_count((2, 4), (1, 1)) // 2
    )
    np.testing.assert_allclose(found.cfar_threshold_w, 2.0 * alpha_linear, rtol=1e-14)
    assert math.isnan(cluster_detections(detected, power_w)[0].cfar_threshold_w)


@pytest.mark.parametrize("wrap_axes", [(), (-2,), (-2, -1)])
def test_a_cluster_with_no_power_has_a_nan_centroid_on_every_axis(wrap_axes):
    """Zero weights define no weighted mean: nan on a linear axis and on a circular one alike."""
    mask = np.zeros((8, 8), dtype=bool)
    mask[7, 3] = True
    (found,) = cluster_detections(mask, np.zeros((8, 8)), wrap_axes=wrap_axes)
    assert np.isnan(found.centroid_index).all()


def test_clustering_an_empty_mask_returns_nothing():
    assert cluster_detections(np.zeros(16, dtype=bool), np.zeros(16)) == []


def test_detection_is_immutable():
    found = cluster_detections(np.array([False, True, False]), np.array([0.0, 1.0, 0.0]))[0]
    assert isinstance(found, Detection)
    with pytest.raises(AttributeError):
        found.peak_power_w = 2.0  # type: ignore[misc]


def test_a_target_on_the_doppler_wrap_gives_one_detection(rng):
    """End to end: a target half way between the last Doppler bin and the first is one detection.

    It advances 31.5 cycles over 64 pulses, which after the fftshift lies at bin 63.5; a Hann
    slow-time taper keeps its sidelobes under a 1e-6 threshold. Doppler guard 3 covers the
    mainlobe. At over 30 dB the centroid sits within 0.1 bin of 63.5; a split or a one-sided
    centroid misses by a bin or more.
    """
    pulse, sample = np.arange(64)[:, None], np.arange(256)
    cube = 0.5 * np.exp(2j * np.pi * (40 * sample / 256 + 31.5 * pulse / 64))
    cube = cube + (rng.normal(size=cube.shape) + 1j * rng.normal(size=cube.shape)) / math.sqrt(2)
    spectrum = np.fft.fft(np.hanning(64)[:, None] * np.fft.fft(cube, axis=1), axis=0)
    power_w = np.abs(np.fft.fftshift(spectrum, axes=0)) ** 2
    detected, threshold_w = cfar_detect_2d(
        power_w, 1e-6, n_train=(4, 8), n_guard=(3, 2), wrap_axes=(0,)
    )
    assert len(cluster_detections(detected, power_w)) == 2
    (found,) = cluster_detections(detected, power_w, threshold_w=threshold_w, wrap_axes=(0,))
    assert found.peak_index[1] == 40 and found.peak_index[0] in (63, 0)
    assert abs(found.centroid_index[0] - 63.5) < 0.1
    assert found.peak_power_w > found.cfar_threshold_w


# Documented failure modes


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: cfar_threshold_factor(pfa=0.0, n_train=8), "pfa must be a probability"),
        (lambda: cfar_detect(np.ones(32), 1.0, n_train=4), "pfa must be a probability"),
        (
            lambda: cfar_probability_of_false_alarm(10.0, n_train=8, variant="median"),
            "variant must be one of",
        ),
        (lambda: cfar_probability_of_false_alarm(np.nan, n_train=8), "alpha_linear must be"),
        (lambda: cfar_probability_of_false_alarm(0.0, n_train=8), "alpha_linear must be"),
        (lambda: cfar_threshold_factor(pfa=1e-3, n_train=8, variant="os", rank=17), "rank must be"),
        (
            lambda: cfar_detect_2d(np.ones((32, 64)), 1e-3, variant="os", rank=77, **RING),
            "rank must be",
        ),
        (lambda: cfar_threshold_factor(pfa=1e-3, n_train=0), "n_train must be at least 1"),
        (
            lambda: cfar_detect(np.ones(32), 1e-3, n_train=4, n_guard=-1),
            "n_guard must be non-negative",
        ),
        (
            lambda: cfar_detect_2d(np.ones((8, 8)), 1e-3, n_train=(2.5, 4), n_guard=(1, 1)),
            "must be integers",
        ),
        (
            lambda: cfar_detect_2d(np.ones((8, 8)), 1e-3, n_train=(2, 4, 6), n_guard=(1, 1, 1)),
            "one count per",
        ),
        (lambda: cfar_detect(-np.ones(32), 1e-3, n_train=4), "finite and non-negative"),
        (lambda: cfar_detect(np.r_[np.nan, np.ones(31)], 1e-3, n_train=4), "finite and non"),
        (
            lambda: cfar_detect_2d(np.r_[np.inf, np.ones(99)].reshape(10, 10), 1e-3, **RING),
            "finite",
        ),
        (
            lambda: cfar_detect_2d(np.ones((8, 40)), 1e-3, n_train=(0, 5), n_guard=(2, 2)),
            "n_guard must be zero on an axis where n_train is",
        ),
        (
            lambda: cfar_detect(np.ones(32), 1e-3, n_train=4, variant="os", rank=2.5),
            "rank must be an integer",
        ),
        (lambda: cfar_detect(1.0, 1e-3, n_train=4), "at least 1 dimension"),
        (lambda: cfar_detect_2d(np.ones(64), 1e-3, **RING), "at least 2 dimension"),
        (lambda: cfar_detect(np.ones((4, 32)), 1e-3, n_train=4, axis=5), "out of bounds"),
        (
            lambda: cfar_detect_2d(np.ones((3, 32, 64)), 1e-3, wrap_axes=(0,), **RING),
            "among the last two",
        ),
        (
            lambda: cfar_detect_2d(np.ones((6, 64)), 1e-3, wrap_axes=(0,), **RING),
            "shorter than its",
        ),
        (lambda: cfar_threshold_factor(pfa=1e-300, n_train=1, variant="so"), "no threshold factor"),
        (lambda: cluster_detections(np.zeros(8, dtype=bool), np.zeros(9)), "do not match"),
        (
            lambda: cluster_detections(
                np.zeros(8, dtype=bool), np.zeros(8), threshold_w=np.zeros(9)
            ),
            "do not match",
        ),
        (lambda: label_connected_cells(np.zeros((4, 4)), connectivity=3), "connectivity must be"),
        (lambda: label_connected_cells(np.zeros((4, 4)), wrap_axes=(2,)), "out of bounds"),
    ],
)
def test_rejects_an_invalid_argument_and_says_which(call, match):
    with pytest.raises(ValueError, match=match):
        call()
