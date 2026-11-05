"""Tests for radar_forge.tracking.scene: closed-form arcs, the plot model, clutter, the map.

Ground truth is closed form where there is one (a turn's radius v/omega, a straight leg's
kinematics, a disc's volume in m rad), a binomial or Poisson interval for every random
draw (spec 011 A6, A7), and the scene's own encounter table and notch fractions, which the
spec states and the configured arcs must regenerate (A8).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta, binom, poisson

from radar_forge.scenario import load_scenario
from radar_forge.tracking.scene import (
    Arc,
    ClutterComponent,
    ClutterMap,
    FixedDiscretes,
    Leg,
    clutter_intensity,
    detect,
    detection_probability,
    measure,
    poisson_clutter_plots,
    polar,
    resolve,
)

SEED = 20261106
TERMINAL = Path(__file__).resolve().parents[2] / "config" / "scenario_011_terminal_dense.toml"
ALPHA_RANGE_M, ALPHA_AZIMUTH_RAD = 115.75, math.radians(1.4)


def test_a_straight_leg_follows_constant_acceleration_along_its_heading() -> None:
    """Heading 120 deg, 10 m/s, +2 m/s^2 for 30 s: 1200 m along track, 70 m/s."""
    arc = Arc(5.0, (100.0, 200.0), math.radians(120.0), 10.0, (Leg(30.0, 0.0, 2.0),))
    position_m, velocity_mps = arc.state(35.0)
    along = np.array([math.sin(math.radians(120.0)), math.cos(math.radians(120.0))])
    # rtol 1e-12: a few products.
    np.testing.assert_allclose(position_m, [100.0, 200.0] + 1200.0 * along, rtol=1e-12)
    np.testing.assert_allclose(velocity_mps, 70.0 * along, rtol=1e-12)


def test_a_turn_holds_its_radius_and_speed_and_after_the_legs_it_flies_straight() -> None:
    """3 deg/s at 72 m/s: every point 1375.1 m from the centre; then straight at 72 m/s."""
    rate = math.radians(3.0)
    arc = Arc(0.0, (0.0, 0.0), 0.0, 72.0, (Leg(60.0, rate),))
    position_m, velocity_mps = arc.state(np.linspace(0.0, 60.0, 61))
    centre = np.array([72.0 / rate, 0.0])
    np.testing.assert_allclose(np.linalg.norm(position_m - centre, axis=1), 72.0 / rate, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(velocity_mps, axis=1), 72.0, rtol=1e-12)
    later, later_velocity = arc.state(np.array([60.0, 70.0]))
    # atol 1e-9 m: the eastward step is zero, so only an absolute bound means anything there.
    np.testing.assert_allclose(later[1] - later[0], 10.0 * later_velocity[0], rtol=1e-12, atol=1e-9)


def test_a_target_does_not_exist_before_its_start() -> None:
    arc = Arc(100.0, (0.0, 0.0), 0.0, 50.0)
    position_m, _ = arc.state(np.array([99.0, 100.0]))
    assert np.isnan(position_m[0]).all() and np.isfinite(position_m[1]).all()


def test_a_leg_may_not_turn_and_accelerate_at_once() -> None:
    with pytest.raises(ValueError, match="leg"):
        Leg(10.0, 0.1, 1.0)


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
    """A7: 10 000 seeded pairs at d resolution units; merged fraction in its 99.9 % interval.

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
    """Spec 011 §3.2's V4 layers: background, ground residue, the drifting rain cell."""
    return list(load_scenario(TERMINAL).clutter_plots.components)


def test_a_disc_of_clutter_has_the_volume_of_its_integral_over_range() -> None:
    """2 pi (c - sqrt(c^2 - R^2)) m rad: 2488.6 for the rain cell at 20.4 km, not pi R^2 / c."""
    rain = _components()[2]
    centre_m = math.hypot(20e3, 4e3)
    expected = 2 * math.pi * (centre_m - math.sqrt(centre_m**2 - 4e3**2))
    assert rain.volume_m_rad(0.0) == pytest.approx(expected, rel=1e-12)
    assert rain.volume_m_rad(0.0) == pytest.approx(2488.6, abs=0.1)


def test_every_clutter_component_draws_its_intensity_times_volume() -> None:
    """A6: each layer's plots over 125 scans in the 99 % Poisson interval of lambda V.

    Catches an intensity per square metre where one per (m rad) was meant: the counts would be
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
        expected += [c.intensity_per_m_rad * c.volume_m_rad(time_s) for c in components]
    low, high = poisson.ppf(0.005, expected), poisson.ppf(0.995, expected)
    assert np.all((low <= counts) & (counts <= high)), (counts, expected)


def test_the_oracle_intensity_sums_the_layers_a_point_lies_in() -> None:
    components = _components()
    # The rain centre at 0 s is inside the background and the rain; 3 km is ground residue too.
    rain_r, rain_az = polar(np.array([20e3, 4e3]))
    got = clutter_intensity(components, [float(rain_r), 3e3], [float(rain_az), 0.0], 0.0)
    np.testing.assert_allclose(got, [0.0021683 + 0.65048, 0.0021683 + 0.065048], rtol=1e-12)


def test_fixed_discretes_are_the_same_points_every_run_and_report_half_the_time() -> None:
    first = FixedDiscretes.uniform(25, 1e3, 15e3, 0.5, seed=7)
    second = FixedDiscretes.uniform(25, 1e3, 15e3, 0.5, seed=7)
    np.testing.assert_array_equal(first.range_m, second.range_m)
    rng = np.random.default_rng(SEED)
    reports = sum(first.plots(0.0, 0.0, rng)[0].size for _ in range(400))
    low, high = binom.ppf([0.005, 0.995], 25 * 400, 0.5)
    assert low <= reports <= high


def test_the_clutter_map_learns_a_uniform_intensity() -> None:
    """Uniform clutter at 0.02 per m rad for 60 scans: the map's mean estimate within 3 %.

    The estimate over 1440 cells averages ~ 1.7 plots a cell a scan; 3 % is over five standard
    errors of that mean.
    """
    rng = np.random.default_rng(SEED)
    layer = [ClutterComponent(0.02, 0.0, 20e3)]
    clutter_map = ClutterMap(range_max_m=20e3)
    # Sixty scans, each folded in turn.
    for _ in range(60):
        ranges, azimuths, _ = poisson_clutter_plots(layer, 0.0, rng)
        clutter_map.update(ranges, azimuths)
    centres_r, centres_az = np.meshgrid(
        np.arange(500.0, 20e3, 1e3), np.arange(72) * math.radians(5.0) + 0.01
    )
    estimate = clutter_map.intensity(centres_r.ravel(), centres_az.ravel())
    assert estimate.mean() == pytest.approx(0.02, rel=0.03)


def test_an_empty_clutter_map_holds_its_prior_then_its_floor() -> None:
    """Before any scan: the prior; a cell that never saw a plot later: the forgetting floor."""
    clutter_map = ClutterMap(prior_per_m_rad=0.02)
    assert clutter_map.intensity([5e3], [1.0])[0] == pytest.approx(0.02, rel=1e-12)
    for _ in range(200):
        clutter_map.update(np.zeros(0), np.zeros(0))
    assert clutter_map.intensity([5e3], [1.0])[0] == pytest.approx(
        clutter_map.floor_per_m_rad, rel=1e-12
    )


# --- A8: the scene regenerates spec 011's encounter table and notch fractions ---

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
    range_a, az_a = polar(a_m)
    range_b, az_b = polar(b_m)
    gap_az = np.mod(az_a - az_b + np.pi, 2 * np.pi) - np.pi
    return np.hypot((range_a - range_b) / ALPHA_RANGE_M, gap_az / ALPHA_AZIMUTH_RAD)


@pytest.mark.parametrize("encounter", ENCOUNTERS, ids=["-".join(e[0]) for e in ENCOUNTERS])
def test_the_scene_regenerates_its_closest_approaches(encounter: tuple) -> None:
    """A8: min d to 0.01, its first time to 0.1 s, separation to 1 m, angle to 1 deg, scans exact.

    A scan's beam crosses the pair at its mean azimuth, found by fixed point from the rotation's
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
    # One rotation at a time: the beam's crossing is a short fixed point.
    for scan in range(125):
        time_s = 4.8 * scan
        for _ in range(3):
            middle = (arcs[first].state(time_s)[0] + arcs[second].state(time_s)[0]) / 2
            time_s = 4.8 * scan + float(polar(middle)[1]) / (2 * np.pi) * 4.8
        separation = _distance(arcs[first].state(time_s)[0], arcs[second].state(time_s)[0])
        seen += bool(np.exp(-(separation**2) / 2) > 0.5)
    assert seen == n_scans


def test_the_scene_regenerates_its_notch_fractions() -> None:
    """A8: the fraction of each aircraft's life with |range rate| < 4 m/s, to 0.001."""
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


# Slow: 120 frames of scenario 001's S3 chain, two FFTs and CFAR a burst, about three minutes.
@pytest.mark.slow
def test_the_iq_chain_merges_close_pairs_as_the_gaussian_resolution_model(tmp_path: Path) -> None:
    """A14: V5's one-detection fraction at four separations against the plot model's.

    A pair gives one detection when it merges and is seen, or is resolved and one member is
    missed: P_u (1 - (1 - p)^2) + (1 - P_u) 2 p (1 - p), P_u = exp(-(dr / alpha_r)^2 / 2) and p
    the Pd the plot model is given for V5, 0.99 (A13). At 6 bins only the second term is left:
    there the chain's one detection is a member masked by its partner in the CFAR's training
    cells, a miss the plot model draws at random. alpha_r is fitted by maximum likelihood to the
    four binomial counts; each measured fraction must lie in its 99 % Clopper-Pearson interval
    about the fitted model's value.
    """
    from scipy.optimize import minimize_scalar

    from radar_forge import evaluate

    config = TERMINAL.parent / "scenario_011_iq_check.toml"
    report = evaluate.run(
        load_scenario(config), ["rd", "multitarget"], out_dir=tmp_path, render=False
    )
    measured = {
        float(key.removeprefix("merged_fraction_").removesuffix("m")): value
        for key, value in report.values.items()
        if key.startswith("merged_fraction_")
    }
    separation_m, fraction = np.array(list(measured)), np.array(list(measured.values()))
    n_frames, merged = 120, np.round(fraction * 120).astype(int)
    assert separation_m.size == 4

    pd = 0.99

    def one_detection(alpha_m: float) -> np.ndarray:
        unresolved = np.exp(-((separation_m / alpha_m) ** 2) / 2)
        both = unresolved * (1 - (1 - pd) ** 2) + (1 - unresolved) * 2 * pd * (1 - pd)
        return np.clip(both, 1e-12, 1 - 1e-12)

    def negative_log_likelihood(alpha_m: float) -> float:
        p = one_detection(alpha_m)
        return -float(np.sum(merged * np.log(p) + (n_frames - merged) * np.log(1 - p)))

    alpha_m = minimize_scalar(negative_log_likelihood, bounds=(10.0, 2000.0), method="bounded").x
    model = one_detection(alpha_m)
    low = beta.ppf(0.005, merged, n_frames - merged + 1)
    high = beta.ppf(0.995, merged + 1, n_frames - merged)
    low, high = np.nan_to_num(low, nan=0.0), np.nan_to_num(high, nan=1.0)
    print("alpha_r", alpha_m, "separations", separation_m, "measured", fraction, "model", model)
    assert np.all((low <= model) & (model <= high))
