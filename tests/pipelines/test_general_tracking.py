from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from radar_forge.core.ambiguity import fold_velocity_mps
from radar_forge.core.detection_2d import DetectionConfig
from radar_forge.pipelines.general_tracking import ScenarioTracker
from radar_forge.pipelines.scenarios import (
    RangeDopplerProduct,
    form_range_doppler_map,
    iterate_frames,
    load_scenario,
)
from radar_forge.pipelines.tracking_config import (
    ModelRegistry,
    TrackingConfig,
    build_enu_tracker,
    build_tracker,
    parse_tracking_table,
)
from radar_forge.pipelines.tracking_output import TrackingWriter
from radar_forge.tracking import (
    CartesianPosition,
    Coordinate,
    Measurement,
    MeasurementBatch,
    RadialMotion,
    StateEstimate,
    StateSpace,
    TrackStatus,
)

ROOT = Path(__file__).resolve().parents[2]
PATHS = [
    ROOT / "scenarios" / f"scenario_001_{variant}.toml"
    for variant in ("fmcw_low_prf", "pulsed_medium_prf", "fmcw_dual_prf")
]


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.slow
def test_seeded_iq_to_track_and_truth_independence(variant, tmp_path):
    scenario = replace(load_scenario(PATHS[variant]), duration_s=5)
    first = ScenarioTracker(scenario.bursts)
    second = ScenarioTracker(scenario.bursts)
    with TrackingWriter(tmp_path, first) as writer:
        for frame in iterate_frames(scenario):
            products = tuple(
                form_range_doppler_map(c, l) for c, l in zip(frame.iq, scenario.bursts, strict=True)
            )
            altered = replace(
                frame, range_m=1e6, radial_velocity_mps=1000, azimuth_deg=0, elevation_deg=0
            )
            other_products = tuple(
                form_range_doppler_map(c, l)
                for c, l in zip(altered.iq, scenario.bursts, strict=True)
            )
            result = first.process(frame.time_s, products)
            other = second.process(altered.time_s, other_products)
            assert len(result.detections) == len(other.detections)
            assert len(result.snapshots) == len(other.snapshots)
            for a, b in zip(result.snapshots, other.snapshots, strict=True):
                np.testing.assert_allclose(a.state, b.state, rtol=0, atol=0)
            writer.write(
                result, truth_range_m=frame.range_m, truth_velocity_mps=frame.radial_velocity_mps
            )
    assert writer.metrics()["confirmation_latency_s"] == 2
    assert (tmp_path / "tracking_metrics.json").exists()
    if variant == 1:
        assert "modulo_absolute_unresolved" in (tmp_path / "tracks.csv").read_text()


@pytest.mark.parametrize("variant", ["S1", "S2", "S3"])
def test_synthetic_radial_acceptance_and_s2_wrap_continuity(variant):
    motion = RadialMotion(0.01)
    names = ("range_m",) if variant == "S1" else motion.state_space.names
    period_m = 6000.0 if variant == "S2" else None
    model = CartesianPosition(
        motion.state_space, names, {"range_m": period_m} if period_m else None
    )
    prior = StateEstimate(np.zeros(2), np.diag([1.0, 10000.0]), 0, motion.state_space)
    engine = build_tracker(motion, model, prior)
    rng = np.random.default_rng(142)
    errors = []
    ids = set()
    for t in range(120):
        truth_range_m, truth_velocity_mps = 6050 - 20 * t, 20.0
        observed_range_m = round(truth_range_m / 75) * 75 + rng.normal(0, 0.1)
        if period_m:
            observed_range_m %= period_m
        values = (
            np.array([observed_range_m])
            if variant == "S1"
            else np.array([observed_range_m, 20 + rng.uniform(-0.3, 0.3)])
        )
        cov = np.diag([75**2 / 12] if variant == "S1" else [75**2 / 12, 0.6**2 / 12])
        result = engine.process(
            MeasurementBatch(t, "sensor", (Measurement(values, cov, t, "sensor", "measurement"),))
        )
        assert len(result) == 1
        ids.add(result[0].track_id)
        if t >= 30:
            error = result[0].state - [truth_range_m, truth_velocity_mps]
            if period_m:
                error[0] = (error[0] + period_m / 2) % period_m - period_m / 2
            errors.append(error)
    assert len(ids) == 1
    rmse = np.sqrt(np.mean(np.array(errors) ** 2, axis=0))
    assert rmse[0] < 75
    assert rmse[1] < (5 if variant == "S1" else 0.6)


def product_with_peaks(leg, peaks):
    n = 128
    range_axis = np.arange(n, dtype=np.float64) * 75
    velocity_axis = np.linspace(
        -leg.unambiguous_velocity_mps, leg.unambiguous_velocity_mps, n, endpoint=False
    )
    power = np.ones((n, n), dtype=np.complex128)
    for range_bin, velocity_mps in peaks:
        folded = float(fold_velocity_mps(velocity_mps, leg.unambiguous_velocity_mps))
        row = int(np.argmin(abs(velocity_axis - folded)))
        power[row, range_bin] = 1000
    return RangeDopplerProduct(power, range_axis, velocity_axis)


def test_dual_prf_pairing_rejects_missing_and_non_unique_peaks():
    scenario = load_scenario(PATHS[2])
    a, b = scenario.bursts
    tracker = ScenarioTracker(scenario.bursts)
    result = tracker.process(0, (product_with_peaks(a, [(50, 80)]), product_with_peaks(b, [])))
    assert result.n_observations == 0
    assert result.detections[0].status == "missing_pair"
    result = tracker.process(
        1, (product_with_peaks(a, [(50, 80)]), product_with_peaks(b, [(50, 80), (50, 20)]))
    )
    assert result.n_observations == 0
    assert all(d.status == "ambiguous_pair" for d in result.detections)
    result = tracker.process(
        2, (product_with_peaks(a, [(50, 80)]), product_with_peaks(b, [(50, 80)]))
    )
    assert result.n_observations == 1
    assert len(result.snapshots) == 1
    assert abs(result.snapshots[0].state[1] - 80) < 1


@pytest.mark.parametrize("axes", ["x", "xy", "xyz"])
@pytest.mark.parametrize("kind", ["CV", "CA"])
def test_enu_synthetic_examples(axes, kind):
    engine = build_enu_tracker(
        dict.fromkeys(axes, kind), origin_lla_deg_m=(36.00250, -78.94100, 60.0), noise_density=0.001
    )
    n = len(axes)
    rng = np.random.default_rng(99)
    for t in range(60):
        truth = np.arange(1, n + 1) * 100 + 2 * t + (0.05 * t * t if kind == "CA" else 0)
        batch = engine.sensors["sensor"].batch(
            t, [("measurement", truth + rng.normal(0, 0.1, n), np.eye(n) * 0.01)]
        )
        snapshots = engine.process(batch)
    assert len(snapshots) == 1
    assert snapshots[0].status == TrackStatus.CONFIRMED
    space = snapshots[0].estimate.state_space
    positions = list(space.indices(tuple(f"{a}_m" for a in axes)))
    np.testing.assert_allclose(snapshots[0].state[positions], truth, rtol=0, atol=0.5)


def test_strict_toml_settings_and_custom_registry():
    assert parse_tracking_table({"training_cells": 4}, DetectionConfig).training_cells == 4
    for table in (
        {"history_size": 1.5},
        {"alpha": "0.5"},
        {"unknown": 1},
        {"confirmation_hits": True},
    ):
        with pytest.raises(ValueError):
            parse_tracking_table(table, TrackingConfig)

    class Custom:
        state_space = StateSpace((Coordinate("custom_m", "m"),))

        def transition(self, state, dt_s):
            return state

        def process_noise(self, state, dt_s):
            return np.eye(1) * dt_s

    registry = ModelRegistry(
        {"custom": lambda options: Custom()},
        {"direct": lambda options, space: CartesianPosition(space, ("custom_m",))},
    )
    motion, model = registry.build("custom", {}, "direct", {})
    engine = build_tracker(
        motion, model, StateEstimate(np.zeros(1), np.eye(1), 0, motion.state_space)
    )
    result = engine.process(
        engine.sensors["sensor"].batch(0, [("measurement", np.array([2.0]), np.eye(1))])
    )
    np.testing.assert_allclose(result[0].state, [2], rtol=1e-12)


def test_dual_prf_unresolved_velocity_is_a_missed_observation():
    scenario = load_scenario(PATHS[2])
    tracker = ScenarioTracker(
        scenario.bursts, tracking=TrackingConfig(unfolding_tolerance_bins=0.001)
    )
    a, b = scenario.bursts
    result = tracker.process(
        0, (product_with_peaks(a, [(50, 80)]), product_with_peaks(b, [(50, 11)]))
    )
    assert result.n_observations == 0
    assert all(d.status == "unresolved_velocity" for d in result.detections)
    assert not result.snapshots
