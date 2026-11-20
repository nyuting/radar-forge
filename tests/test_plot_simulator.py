"""Tests for radar_forge.plot_simulator: the plot model, clutter, the encounter table.

Ground truth is closed form where there is one (a disc's volume in m rad), a binomial or
Poisson interval for every random draw, and the scene's own encounter table and notch
fractions, which the configured arcs must regenerate.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta, binom, poisson

from radar_forge.coordinates import east_north_to_range_azimuth_rad
from radar_forge.plot_simulator import (
    ClutterComponent,
    FixedDiscretes,
    clutter_density,
    detect,
    detection_probability,
    measure,
    plot_scans,
    poisson_clutter_plots,
    resolve,
)
from radar_forge.scenario import load_scenario

SEED = 20261106
TERMINAL = Path(__file__).resolve().parents[1] / "config" / "scenario_011_terminal_dense.toml"
ALPHA_RANGE_M, ALPHA_AZIMUTH_RAD = 115.75, math.radians(1.4)


def test_an_echo_in_the_notch_takes_the_notch_pd() -> None:
    pd = detection_probability(
        [0.0, 3.9, -4.1, 50.0],
        [0.9, 0.9, 0.9, 0.3],
        mti_notch_mps=4.0,
        notch_detection_probability=0.3,
    )
    np.testing.assert_array_equal(pd, [0.3, 0.3, 0.9, 0.3])


@pytest.mark.parametrize("separation", [0.5, 1.0, 2.0])
def test_a_pair_merges_with_the_gaussian_resolution_probability(separation: float) -> None:
    """10 000 seeded pairs at d resolution units; merged fraction in its 99.9 % interval.

    Pairs lie 10^6 m apart in range, so only the intended partners can merge; half are offset
    in range and half in azimuth.
    """
    rng = np.random.default_rng(SEED)
    n_pairs, merged = 10_000, 0
    # 100 batches of 100 pairs: resolve is O(n^2) in one scan's echoes.
    for batch in range(100):
        base = 1e6 * np.arange(100) + 1e8 * batch
        along_range = batch % 2 == 0
        offset_r = separation * ALPHA_RANGE_M if along_range else 0.0
        offset_az = 0.0 if along_range else separation * ALPHA_AZIMUTH_RAD
        ranges = np.concatenate([base, base + offset_r])
        azimuths = np.concatenate([np.full(100, 1.0), np.full(100, 1.0 + offset_az)])
        group = resolve(
            ranges,
            azimuths,
            alpha_range_m=ALPHA_RANGE_M,
            alpha_azimuth_rad=ALPHA_AZIMUTH_RAD,
            rng=rng,
        )
        merged += int(np.sum(group[:100] == group[100:]))
    low, high = (
        beta.ppf(0.0005, merged, n_pairs - merged + 1),
        beta.ppf(0.9995, merged + 1, n_pairs - merged),
    )
    assert low <= math.exp(-(separation**2) / 2) <= high


def test_a_group_is_detected_unless_every_echo_is_missed() -> None:
    """Two echoes at Pd 0.5 make one group seen with 0.75: 20 000 groups, 99.9 % interval."""
    rng = np.random.default_rng(SEED)
    found = detect(
        np.repeat(np.arange(20_000), 2), np.full(40_000, 0.5), rng
    )  # broadcast-exempt: two echoes a group
    low, high = binom.ppf([0.0005, 0.9995], 20_000, 0.75) / 20_000
    assert low <= found.mean() <= high


def test_a_merged_plot_lies_at_its_echoes_centroid_even_across_north() -> None:
    """Echoes at 359.9 and 0.1 deg, 20.0 and 20.1 km: one plot at 0 deg, 20.05 km, no noise."""
    rng = np.random.default_rng(SEED)
    azimuths = np.radians([359.9, 0.1])
    range_m, azimuth = measure(
        [20e3, 20.1e3], azimuths, [0, 0], sigma_range_m=0.0, sigma_azimuth_rad=0.0, rng=rng
    )
    np.testing.assert_allclose(range_m, [20.05e3], rtol=1e-12)
    np.testing.assert_allclose(np.mod(azimuth + np.pi, 2 * np.pi) - np.pi, [0.0], atol=1e-12)


def _components() -> list[ClutterComponent]:
    """V4's layers: background, ground residue, the drifting rain cell."""
    return list(load_scenario(TERMINAL).clutter_plots.components)


@pytest.mark.parametrize(
    ("centre_m", "radius_m", "volume_m_rad"),
    [(10.0, 5.0, 8.126), (math.hypot(20e3, 4e3), 4e3, 2476.5)],
    ids=["c10-r5", "rain-cell"],
)
def test_a_disc_of_clutter_has_the_volume_of_its_integral_of_one_over_range(
    centre_m: float, radius_m: float, volume_m_rad: float
) -> None:
    """The volume in (m rad) is the integral over azimuth of each chord, done numerically here.

    The closed form in K and E must equal quad's integral of 2 sqrt(R^2 - c^2 sin^2 phi):
    8.126 for c = 10, R = 5 (the earlier 2 pi (c - sqrt(c^2 - R^2)) gave 8.418), and
    2476.5 m rad for the rain cell at the start.
    """
    from scipy.integrate import quad

    disc = ClutterComponent(1.0, 0.0, 40e3, (centre_m, 0.0), radius_m)
    edge = math.asin(radius_m / centre_m)

    def chord(angle: float) -> float:
        return 2 * math.sqrt(max(radius_m**2 - (centre_m * math.sin(angle)) ** 2, 0.0))

    # rtol 1e-9: quad's own accuracy on a square root that vanishes at the ends.
    assert disc.volume_m_rad(0.0) == pytest.approx(quad(chord, -edge, edge)[0], rel=1e-9)
    # abs 5e-4 of the value: the tabulated figures carry four or five digits.
    assert disc.volume_m_rad(0.0) == pytest.approx(volume_m_rad, rel=5e-4)


def test_every_clutter_component_draws_its_density_times_volume() -> None:
    """Each layer's plots over 125 scans in the 99 % Poisson interval of lambda V.

    Catches a density per square metre where one per (m rad) was meant: the counts would be
    off by the range, thousands of times.
    """
    rng = np.random.default_rng(SEED)
    components = _components()
    counts, expected = np.zeros(len(components)), np.zeros(len(components))
    # 125 scans, each one batched draw.
    for scan in range(125):
        time_s = 4.8 * scan
        _, _, index = poisson_clutter_plots(components, time_s, rng)
        counts += np.bincount(index, minlength=len(components))
        expected += [c.density_per_m_rad * c.volume_m_rad(time_s) for c in components]
    low, high = poisson.ppf(0.005, expected), poisson.ppf(0.995, expected)
    assert np.all((low <= counts) & (counts <= high)), (counts, expected)


def test_the_oracle_density_sums_the_layers_a_point_lies_in() -> None:
    components = _components()
    # The rain centre at 0 s is inside the background and the rain; 3 km is ground residue too.
    rain_r, rain_az = east_north_to_range_azimuth_rad(np.array([20e3, 4e3]))
    got = clutter_density(components, [float(rain_r), 3e3], [float(rain_az), 0.0], 0.0)
    np.testing.assert_allclose(got, [0.0021683 + 0.65048, 0.0021683 + 0.065048], rtol=1e-12)


def test_fixed_discretes_are_the_same_points_every_run_and_report_half_the_time() -> None:
    first = FixedDiscretes.uniform(25, 1e3, 15e3, 0.5, seed=7)
    second = FixedDiscretes.uniform(25, 1e3, 15e3, 0.5, seed=7)
    np.testing.assert_array_equal(first.range_m, second.range_m)
    rng = np.random.default_rng(SEED)
    reports = sum(first.plots(0.0, 0.0, rng)[0].size for _ in range(400))
    low, high = binom.ppf([0.005, 0.995], 25 * 400, 0.5)
    assert low <= reports <= high


def test_a_plot_level_scan_holds_its_echoes_clutter_and_discretes() -> None:
    """One V4 scan: every plot timed within its scan and batched by its azimuth."""
    scenario = load_scenario(TERMINAL)
    plot_scan = next(plot_scans(scenario, n_scans=1))
    assert scenario.sensor is not None
    period_s = scenario.scan_period_s
    assert np.all((plot_scan.time_s >= 0.0) & (plot_scan.time_s < period_s))
    sector = np.floor(plot_scan.azimuth_rad / (2 * np.pi / scenario.sensor.n_sectors))
    np.testing.assert_array_equal(
        plot_scan.sector, np.minimum(sector, scenario.sensor.n_sectors - 1)
    )
    kinds = set(plot_scan.source)
    assert {"target", "clutter", "discrete"} <= kinds
    assert plot_scan.echo_in_coverage.sum() >= 10


# --- the scene regenerates the encounter table and notch fractions ---

ENCOUNTERS = [
    # (pair, minimum d, at s, ground separation m, track angle deg, scans with P_u > 0.5)
    (("H1", "H2"), 0.52, 151.7, 439.0, 12.0, 10),
    (("H3", "H4"), 1.03, 29.7, 874.0, 24.0, 3),
    (("A1", "T2"), 0.00, 200.0, 0.0, 30.0, 1),
    (("H2", "T1"), 0.00, 236.0, 0.0, 140.0, 1),
    (("A2", "A3"), 2.36, 244.4, 815.0, 0.0, 0),
]
NOTCH = {"HE": 0.156, "L1": 0.120, "L2": 0.130, "A4": 0.053, "A2": 0.052, "A1": 0.049}


def _distance(a_m: np.ndarray, b_m: np.ndarray) -> np.ndarray:
    """Return the normalised (range, azimuth) distance d between two position tracks."""
    range_a, az_a = east_north_to_range_azimuth_rad(a_m)
    range_b, az_b = east_north_to_range_azimuth_rad(b_m)
    gap_az = np.mod(az_a - az_b + np.pi, 2 * np.pi) - np.pi
    return np.hypot((range_a - range_b) / ALPHA_RANGE_M, gap_az / ALPHA_AZIMUTH_RAD)


@pytest.mark.parametrize("encounter", ENCOUNTERS, ids=["-".join(e[0]) for e in ENCOUNTERS])
def test_the_scene_regenerates_its_closest_approaches(encounter: tuple) -> None:
    """Min d to 0.01, its first time to 0.1 s, separation to 1 m, angle to 1 deg, scans exact.

    A scan's beam crosses the pair at its mean azimuth, found by fixed point from the scan's
    start.
    """
    (first, second), min_d, at_s, gap_m, angle_deg, n_scans = encounter
    arcs = {target.name: target.arc for target in load_scenario(TERMINAL).arcs}
    grid_s = np.arange(0.0, 600.0001, 0.1)
    (a_m, a_v), (b_m, b_v) = arcs[first].state(grid_s), arcs[second].state(grid_s)
    d = _distance(a_m, b_m)
    k = int(np.flatnonzero(d <= np.nanmin(d) + 1e-9)[0])
    assert round(float(d[k]), 2) == min_d
    assert abs(grid_s[k] - at_s) < 0.05 + 1e-9
    assert abs(np.linalg.norm(a_m[k] - b_m[k]) - gap_m) < 1.0
    cosine = a_v[k] @ b_v[k] / np.linalg.norm(a_v[k]) / np.linalg.norm(b_v[k])
    assert abs(math.degrees(math.acos(np.clip(cosine, -1, 1))) - angle_deg) < 1.0
    seen = 0
    # One scan at a time: the beam's crossing is a short fixed point.
    for scan in range(125):
        time_s = 4.8 * scan
        for _ in range(3):
            middle = (arcs[first].state(time_s)[0] + arcs[second].state(time_s)[0]) / 2
            time_s = (
                4.8 * scan + float(east_north_to_range_azimuth_rad(middle)[1]) / (2 * np.pi) * 4.8
            )
        separation = _distance(arcs[first].state(time_s)[0], arcs[second].state(time_s)[0])
        seen += bool(np.exp(-(separation**2) / 2) > 0.5)
    assert seen == n_scans


def test_the_scene_regenerates_its_notch_fractions() -> None:
    """The fraction of each aircraft's life with |range rate| < 4 m/s, to 0.001."""
    grid_s = np.arange(0.0, 600.0001, 0.1)
    for target in load_scenario(TERMINAL).arcs:
        position_m, velocity_mps = target.arc.state(grid_s)
        alive = np.isfinite(position_m[:, 0])
        rate = np.sum(position_m * velocity_mps, axis=1)[alive] / np.linalg.norm(
            position_m[alive], axis=1
        )
        fraction = float(np.mean(np.abs(rate) < 4.0))
        if target.name in NOTCH:
            assert round(fraction, 3) == NOTCH[target.name], target.name
        elif target.target_class == "aircraft":
            assert fraction < 0.0125, target.name


# Slow: 120 scans of scenario 001's S3 chain, two FFTs and CFAR a burst, about three minutes.
@pytest.mark.slow
def test_the_iq_chain_merges_close_pairs_as_the_gaussian_resolution_model(tmp_path: Path) -> None:
    """V5's one-detection fraction at four separations against the plot model's.

    A pair gives one detection when it merges and is seen, or is resolved and one member is
    missed: P_u (1 - (1 - p)^2) + (1 - P_u) 2 p (1 - p), P_u = exp(-(dr / alpha_r)^2 / 2) and p
    the Pd the plot model is given for V5, 0.99. At 6 bins only the second term is left:
    there the chain's one detection is a member masked by its partner in the CFAR's training
    cells, a miss the plot model draws at random. alpha_r is fitted by maximum likelihood to the
    four binomial counts; each measured fraction must lie in its 99 % Clopper-Pearson interval
    about the fitted model's value.
    """
    from scipy.optimize import minimize_scalar

    from radar_forge import verify

    config = TERMINAL.parent / "scenario_011_iq_check.toml"
    report = verify.run(
        load_scenario(config), ["rd", "multitarget"], out_dir=tmp_path, render=False
    )
    measured = {
        float(key.removeprefix("merged_fraction_").removesuffix("m")): value
        for key, value in report.values.items()
        if key.startswith("merged_fraction_")
    }
    separation_m, fraction = np.array(list(measured)), np.array(list(measured.values()))
    n_scans, merged = 120, np.round(fraction * 120).astype(int)
    assert separation_m.size == 4

    pd = 0.99

    def one_detection(alpha_m: float) -> np.ndarray:
        unresolved = np.exp(-((separation_m / alpha_m) ** 2) / 2)
        both = unresolved * (1 - (1 - pd) ** 2) + (1 - unresolved) * 2 * pd * (1 - pd)
        return np.clip(both, 1e-12, 1 - 1e-12)

    def negative_log_likelihood(alpha_m: float) -> float:
        p = one_detection(alpha_m)
        return -float(np.sum(merged * np.log(p) + (n_scans - merged) * np.log(1 - p)))

    alpha_m = minimize_scalar(negative_log_likelihood, bounds=(10.0, 2000.0), method="bounded").x
    model = one_detection(alpha_m)
    low = beta.ppf(0.005, merged, n_scans - merged + 1)
    high = beta.ppf(0.995, merged + 1, n_scans - merged)
    low, high = np.nan_to_num(low, nan=0.0), np.nan_to_num(high, nan=1.0)
    print("alpha_r", alpha_m, "separations", separation_m, "measured", fraction, "model", model)
    assert np.all((low <= model) & (model <= high))
