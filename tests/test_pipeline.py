"""The chain from a scenario file to tracks, end to end and stage by stage.

Ground truth is closed form wherever the test can compute it apart from the code: the
bistatic route length |p - tx| + |p| and its rate v . (u_T + u_R), the map bin of an exact
range and rate, the noise of every range cell, and a rate that folds in both bursts of a pair.
The tier-3 scene is four targets at mixed range, rate and cross-section, monostatic and
bistatic, run three frames through ``evaluate.run``.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from radar_forge import evaluate, plots
from radar_forge import pipeline as pipeline_module
from radar_forge.clutter import cancel
from radar_forge.constants import SPEED_OF_LIGHT_MPS
from radar_forge.pipeline import (
    Plots,
    ScanTracker,
    frame_plots,
    frame_products,
    frames,
    map_axes,
    range_doppler_product,
    synthesise,
)
from radar_forge.radar import Radar, Receiver, Site, TargetGeometry, Transmitter
from radar_forge.scenario import CancellationSettings, Target, load_scenario, truth_geometry
from radar_forge.signal import noise_waveform, thermal_noise
from radar_forge.stages import core

REPO = Path(__file__).resolve().parents[1]

# (rcs_dbsm, position_enu_m, velocity_enu_mps) about the receiver: slant ranges 2.0 to 4.4 km,
# radial rates -60 to +30 m/s, cross-sections 0 to 15 dBsm, each about 33 dB after integration.
SCENE = [
    (0.0, (0.0, 2000.0, 0.0), (0.0, 30.0, 0.0)),
    (5.0, (2800.0, 0.0, 0.0), (-45.0, 0.0, 0.0)),
    (10.0, (0.0, -3600.0, 0.0), (0.0, -10.0, 0.0)),
    (15.0, (-4400.0, 0.0, 0.0), (60.0, 0.0, 0.0)),
]
SCENE_SETTINGS = """
[detection]
pfa = 1e-6
max_bistatic_range_rate_mps = 145.0

[tracking]
sigma_acceleration_mps2 = 2.0
n_confirm_hits = 2
n_confirm_scans = 3
n_delete_misses = 3

[verify]
stages = ["geometry", "iq", "rd", "detect", "track"]
truth_within_bins = 1.0
detection_fraction = 1.0
"""


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _closed_form(tx_enu_m: np.ndarray, time_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return each scene target's |p - tx| + |p| and its rate v . (u_T + u_R) at a time."""
    position_m = np.array([p for _, p, _ in SCENE]) + time_s * np.array([v for _, _, v in SCENE])
    velocity_mps = np.array([v for _, _, v in SCENE])
    to_tx_m, to_rx_m = position_m - tx_enu_m, position_m
    range_tx_m, range_rx_m = np.linalg.norm(to_tx_m, axis=1), np.linalg.norm(to_rx_m, axis=1)
    rate_mps = np.sum(
        velocity_mps * (to_tx_m / range_tx_m[:, None] + to_rx_m / range_rx_m[:, None]), axis=1
    )
    return range_tx_m + range_rx_m, rate_mps


@pytest.fixture(params=[False, True], ids=["monostatic", "bistatic"])
def scene_run(request, write_scene, tmp_path):
    """The four-target scene, three frames through every stage, and its run directory."""
    scenario = write_scene(SCENE, bistatic=request.param, n_frames=3, extra=SCENE_SETTINGS)
    report = evaluate.run(scenario, out_dir=tmp_path / "run", render=False)
    return scenario, report, tmp_path / "run"


def test_truth_is_the_closed_form_route_and_its_rate(scene_run):
    """Bistatic range and rate equal |p - tx| + |p| and v . (u_T + u_R), halved monostatic."""
    scenario, _, run_dir = scene_run
    radar = scenario.radar
    tx_enu_m = radar.tx_site.enu_m(radar.rx_site or radar.tx_site)
    scale, range_name, rate_name = (
        (0.5, "range_m", "range_rate_mps")
        if radar.is_monostatic
        else (1.0, "bistatic_range_m", "bistatic_range_rate_mps")
    )
    rows = _rows(run_dir / "truth.csv")
    assert len(rows) == 3 * len(SCENE)
    for frame in range(3):
        bistatic_m, rate_mps = _closed_form(tx_enu_m, float(frame))
        written = [r for r in rows if int(r["frame"]) == frame]
        # rtol 1e-12: the same few float64 operations in a different order.
        np.testing.assert_allclose(
            [float(r[range_name]) for r in written], scale * bistatic_m, rtol=1e-12
        )
        np.testing.assert_allclose(
            [float(r[rate_name]) for r in written], scale * rate_mps, rtol=1e-12, atol=1e-9
        )


def _predicted_bins(scenario, frame: int) -> np.ndarray:
    """Return each target's (rate, range) bin, computed here from c/B and lambda PRF / N."""
    radar = scenario.radar
    tx_enu_m = radar.tx_site.enu_m(radar.rx_site or radar.tx_site)
    bistatic_m, rate_mps = _closed_form(tx_enu_m, float(frame))
    range_bin_m = SPEED_OF_LIGHT_MPS / radar.transmitter.bandwidth_hz
    n_pulses = scenario.n_pulses[0]
    rate_bin_mps = radar.wavelength_m * radar.transmitter.prf_hz / n_pulses
    # Rate falls with Doppler index: zero at n / 2, receding below it.
    return np.column_stack([n_pulses / 2 - rate_mps / rate_bin_mps, bistatic_m / range_bin_m])


def test_every_target_is_one_cluster_and_nothing_else_is_detected(scene_run):
    """One detection within a bin of each target's own bin, every frame, and none elsewhere."""
    scenario, _, run_dir = scene_run
    detections = _rows(run_dir / "detections.csv")
    for frame in range(3):
        cells = np.array(
            [
                [float(r["rate_index"]), float(r["range_index"])]
                for r in detections
                if int(r["frame"]) == frame
            ]
        )
        gaps = np.abs(cells[None, :, :] - _predicted_bins(scenario, frame)[:, None, :])
        near = np.all(gaps <= 1.0, axis=2)
        assert cells.shape[0] == len(SCENE)
        assert np.all(near.sum(axis=1) == 1), f"frame {frame}: a target is not exactly one cluster"
        assert np.all(near.any(axis=0)), f"frame {frame}: a detection no target predicts"


def test_one_confirmed_track_per_target_after_confirmation(scene_run):
    """Two hits in three scans confirm every target by the last frame, and nothing else."""
    _, report, run_dir = scene_run
    last = [r for r in _rows(run_dir / "tracks.csv") if int(r["frame"]) == 2]
    assert sum(r["status"] == "confirmed" for r in last) == len(SCENE)
    assert report.values["n_confirmed_tracks"] == len(SCENE)


def test_a_target_on_an_exact_bin_lands_in_that_bin(monostatic_radar):
    """A receding target 30 bins out and 6 rate bins from zero peaks in exactly that cell."""
    range_axis_m, rate_axis_mps = map_axes(monostatic_radar, 64)
    range_m, rate_mps = range_axis_m[30], rate_axis_mps[32 - 6]
    geometry = TargetGeometry.monostatic(range_m / 2.0, rate_mps / 2.0)
    cube = synthesise(monostatic_radar, geometry, [1.0], 64, rng=None)
    power_w = np.abs(range_doppler_product(cube, monostatic_radar).rd_map) ** 2
    assert rate_mps > 0.0
    assert np.unravel_index(np.argmax(power_w), power_w.shape) == (26, 30)
    # On an exact bin every other cell is near a sidelobe null; not exactly, because the
    # carrier phase is referred to the sweep start, which offsets Doppler by B / 2 f_c.
    assert np.sort(power_w.ravel())[-2] < 1e-3 * power_w.max()


def test_a_pulsed_map_holds_the_same_noise_in_every_range_cell(rng):
    """Regression: compressed row by row, a map's last cells held 4 % of the noise, and CFAR
    there false-alarmed six times its design rate; compressed as one stream they hold all of it."""
    transmitter = Transmitter(9.8e9, 2.0e6, 1.0e3, 30.0, 10.0e-6, 25.0e3, waveform="pulsed")
    radar = Radar(transmitter, Receiver(2.5e6, 30.0, 3.0), Site(36.0, -78.9, 60.0))
    noise = thermal_noise((64, 64, radar.n_samples_per_pri), 1.0, rng=rng)
    profile_w = np.mean(
        [np.abs(range_doppler_product(c, radar).rd_map) ** 2 for c in noise], axis=(0, 1)
    )
    # 4096 exponential cells a range cell: a standard error of 1.6 %, so 10 % is six of them.
    np.testing.assert_allclose(profile_w, np.median(profile_w), rtol=0.1)


def test_a_dual_prf_pair_unfolds_a_rate_neither_burst_resolves(rng):
    """S3's bursts fold 300 m/s of bistatic rate; their pair unfolds it to within a bin."""
    scenario = load_scenario(REPO / "config" / "scenario_001_fmcw_dual_prf.toml")
    geometry = TargetGeometry.monostatic(15_000.0, 150.0)
    products = [
        range_doppler_product(synthesise(r, geometry, [10.0], n, rng=rng), r)
        for r, n in zip(scenario.radars, scenario.n_pulses, strict=True)
    ]
    assert all(p.rate_period_mps / 2.0 < 300.0 for p in products)
    plots = frame_plots(products, scenario.detection)
    accepted = plots.accepted
    assert accepted.size == 1
    assert abs(plots.bistatic_range_rate_mps[accepted[0]] - 300.0) < abs(products[0].rate_bin_mps)
    assert abs(plots.unfolded_range_m[accepted[0]] - 30_000.0) < abs(products[0].range_bin_m)


@pytest.mark.parametrize("path", sorted((REPO / "config").glob("*.toml")), ids=lambda p: p.stem)
def test_the_radars_unambiguous_limits_are_the_maps_periods(path: Path) -> None:
    """Each burst's unambiguous range and rate are the spans its map wraps at, every waveform.

    A noise burst's lags are a window that folds nothing: its delay period is infinite.
    """
    scenario = load_scenario(path)
    bursts = zip(scenario.radars, scenario.n_pulses, scenario.n_range_lags, strict=True)
    for radar, n_pulses, n_range_lags in bursts:
        range_m, rate_mps = map_axes(radar, n_pulses, n_range_lags=n_range_lags)
        assert min(range_m.size, rate_mps.size) > 1, f"{path.stem}: an axis of one bin has no step"
        if radar.transmitter.waveform != "noise":
            np.testing.assert_allclose(
                range_m.size * (range_m[1] - range_m[0]),
                radar.unambiguous_bistatic_range_m,
                rtol=1e-12,
            )
        np.testing.assert_allclose(
            rate_mps.size * abs(rate_mps[1] - rate_mps[0]),
            radar.unambiguous_bistatic_range_rate_mps,
            rtol=1e-12,
        )


def test_an_unknown_key_or_a_wrong_type_is_refused(tmp_path):
    """A misspelled key or a quoted number stops the load rather than being ignored."""
    text = (REPO / "config" / "scenario_003_tracking.toml").read_text(encoding="utf-8")
    for wrong, right, error in (
        ("\nn_train = 16\n", "\nn_trian = 16\n", ValueError),
        ("\npfa = 1e-5\n", '\npfa = "1e-5"\n', TypeError),
    ):
        assert wrong in text
        changed = text.replace(wrong, right, 1)
        path = tmp_path / "wrong.toml"
        path.write_text(changed, encoding="utf-8")
        with pytest.raises(error):
            load_scenario(path)


def test_a_failed_check_names_every_failure(one_frame_scenario, tmp_path):
    """VerificationError lists each failed check, not only the first."""
    verify = {"stages": ["geometry"], "expected_snr_db": 0.0, "truth_in_map_fraction": [2.0, 3.0]}
    scenario = dataclasses.replace(one_frame_scenario, verify=verify)
    with pytest.raises(evaluate.VerificationError) as caught:
        evaluate.run(scenario, out_dir=tmp_path / "run")
    assert len(caught.value.report.failures) == 2
    assert all(c.name in str(caught.value) for c in caught.value.report.failures)
    assert math.isfinite(caught.value.report.values["predicted_snr_db"])


# Review fixes: each a behaviour the tracker policy or the loader got wrong.


def _one_plot(range_m: float, folded_rate_mps: float) -> Plots:
    """One accepted plot of burst 0, read as the map shows it."""
    return Plots(
        (),
        np.array([0]),
        np.array([range_m]),
        np.array([folded_rate_mps]),
        np.array([np.nan]),
        np.array([43.3]),
        np.array([0.0345]),
        ("accepted",),
        np.array([-1]),
        np.array([np.nan]),
    )


def test_the_fold_monitor_fits_the_slope_on_the_scans_own_times():
    """Regression: a coast squeezed the slope's time axis, so a four-scan gap read 80 m/s for 50.

    The prediction sits one fold high; the slope over the real times (50 m/s) disagrees by a
    whole span and refolds the plot to the truth. On scan indices the slope agreed with the
    wrong prediction and the wrong fold stood.
    """
    tracker = ScanTracker(load_scenario(REPO / "config" / "scenario_003_tracking.toml"))
    span_mps, rate_mps = tracker.fold_span_mps, 50.0
    ids = tracker.tracker.seed([[20_600.0, rate_mps + span_mps]], np.diag([100.0, 1.0])[None], 12.0)
    times_s = [*range(9), 12.0]
    tracker.history[int(ids[0])] = [(float(t), 20_000.0 + rate_mps * t) for t in times_s]
    folded_mps = rate_mps - span_mps * round(rate_mps / span_mps)
    # abs 1e-9: whole spans added to a folded reading, a few float64 operations.
    np.testing.assert_allclose(tracker.unfold(np.array([folded_mps])), [rate_mps], atol=1e-9)


def test_a_track_birth_starts_its_range_history():
    """Regression: a birth has no NIS, so its range was dropped and unfolding came a scan late."""
    tracker = ScanTracker(load_scenario(REPO / "config" / "scenario_003_tracking.toml"))
    bank, _ = tracker.scan(_one_plot(20_000.0, 1.0), 0.0)
    assert tracker.history == {int(bank.track_id[0]): [(0.0, 20_000.0)]}


@pytest.mark.parametrize(
    ("file", "old", "new", "match"),
    [
        ("scenario_003_tracking", 'initiator = "range_only"', 'initiator = "direct"', "initiator"),
        (
            "scenario_003_tracking",
            'rate_unfolding = "track_aided"',
            'rate_unfolding = "oracle"',
            "one of",
        ),
        ("scenario_003_tracking", 'linearisation = "taylor"', 'linearisation = "ukf"', "one of"),
        (
            "scenario_003_tracking_dual_prf",
            'rate_unfolding = "none"',
            'rate_unfolding = "track_aided"',
            "two bursts",
        ),
    ],
    ids=[
        "track-aided-direct-seed",
        "unknown-unfolding",
        "unknown-linearisation",
        "track-aided-dual-prf",
    ],
)
def test_a_contradictory_or_unknown_tracking_choice_is_refused_at_load(
    tmp_path, file, old, new, match
):
    """A setting that would be dropped, or die minutes later as a KeyError, stops the load."""
    text = (REPO / "config" / f"{file}.toml").read_text(encoding="utf-8")
    if "two bursts" in match:
        # A range-only seed, so only the two bursts can be what is refused.
        text = text.replace('initiator = "direct"', 'initiator = "range_only"')
    assert old in text
    path = tmp_path / "changed.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_scenario(path)


def test_a_second_stage_of_the_same_name_is_refused():
    """Catches a later stage module silently replacing a core stage and its [verify] keys."""
    keys = evaluate.VERIFY_KEYS["geometry"]
    with pytest.raises(ValueError, match="'geometry' is already registered"):
        evaluate.stage("geometry", "detection_fraction")(lambda ctx: ctx)
    assert evaluate.STAGES["geometry"] is core.geometry
    assert evaluate.VERIFY_KEYS["geometry"] == keys


def test_an_unknown_verify_key_is_refused_before_any_stage(one_frame_scenario, tmp_path):
    """A misspelt threshold raises, naming it, instead of being skipped in silence."""
    verify = {"stages": ["geometry"], "nees_fraction_mn": 0.5}
    scenario = dataclasses.replace(one_frame_scenario, verify=verify)
    with pytest.raises(ValueError, match="nees_fraction_mn"):
        evaluate.run(scenario, out_dir=tmp_path / "run")
    assert not (tmp_path / "run" / "truth.csv").exists()


def test_a_window_with_no_frame_is_refused(one_frame_scenario):
    """0.4 s at 1 Hz rounds to no frame: refused at construction, not deep in the geometry."""
    with pytest.raises(ValueError, match="no frame"):
        dataclasses.replace(one_frame_scenario, duration_s=0.4)


def _trajectory(path: Path, times_s: list[float]) -> Path:
    """Write a track north-bound at 1e-4 degrees a second, absolute in time."""
    rows = [f"{t},{36.01 + 1e-4 * t},-78.9" for t in times_s]
    path.write_text("\n".join(["time_s,latitude_deg,longitude_deg", *rows, ""]))
    return path


def test_a_one_frame_window_with_no_fix_either_side_is_refused(one_frame_scenario, tmp_path):
    """One frame, its track ending half a frame later: no velocity, refused by name."""
    target = Target("short", 0.0, _trajectory(tmp_path / "short.csv", [0.0, 0.5]), 1000.0)
    scenario = dataclasses.replace(one_frame_scenario, targets=(target,))
    with pytest.raises(ValueError, match="one-frame window"):
        truth_geometry(scenario)


def test_trajectory_files_share_one_time_base(one_frame_scenario, tmp_path):
    """Regression: each file was rebased to its own first fix, so these two played 10 s apart.

    Both files follow one law in absolute time, from first fixes 10 s apart; at 12 s they are
    the same point, so their routes are equal.
    """
    early = Target("early", 0.0, _trajectory(tmp_path / "a.csv", [0.0, 10.0, 20.0]), 1000.0)
    late = Target("late", 0.0, _trajectory(tmp_path / "b.csv", [10.0, 20.0, 30.0]), 1000.0)
    scenario = dataclasses.replace(one_frame_scenario, targets=(early, late), start_time_s=12.0)
    truth = truth_geometry(scenario)[0]
    # rtol 1e-12: one interpolation each, between different fixes of the same straight line.
    np.testing.assert_allclose(truth.bistatic_range_m[1], truth.bistatic_range_m[0], rtol=1e-12)
    np.testing.assert_allclose(
        truth.bistatic_range_rate_mps[1], truth.bistatic_range_rate_mps[0], rtol=1e-9
    )


# --- Scenario 005: a noise burst's map, and the tables that describe it ---

NOISE_CONFIG = REPO / "config" / "scenario_005_static_leakage.toml"


def test_a_noise_maps_range_axis_is_one_lag_a_bin():
    """Lag m is the route c m / f_s, a window that does not wrap; it needs a lag count."""
    radar = load_scenario(NOISE_CONFIG).radar
    range_axis_m, rate_axis_mps = map_axes(radar, 2048, n_range_lags=1024)
    np.testing.assert_allclose(range_axis_m, SPEED_OF_LIGHT_MPS * np.arange(1024) / 8.0e6)
    assert rate_axis_mps.size == 2048
    with pytest.raises(ValueError, match="n_range_lags"):
        map_axes(radar, 2048)


def test_a_noise_target_lands_on_its_lag_and_doppler_bin():
    """A noiseless echo at lag 500, closing at 30 Doppler bins, peaks in that cell."""
    radar = load_scenario(NOISE_CONFIG).radar
    doppler_hz = 30 * 9.765625
    route_m = 500 * SPEED_OF_LIGHT_MPS / 8.0e6
    geometry = TargetGeometry.monostatic(route_m / 2.0, -doppler_hz * radar.wavelength_m / 2.0)
    cube = synthesise(radar, geometry, [10.0], 2048, rng=None)
    code = noise_waveform(cube.size, rng=np.random.default_rng(0))  # synthesise's noiseless code
    product = range_doppler_product(cube, radar, reference=code, n_range_lags=1024)
    power_w = np.abs(product.rd_map) ** 2
    assert np.unravel_index(np.argmax(power_w), power_w.shape) == (1024 + 30, 500)
    with pytest.raises(ValueError, match="reference"):
        range_doppler_product(cube, radar, n_range_lags=1024)


def test_noise_tables_need_a_noise_burst(tmp_path):
    """[leakage] and the rest describe a noise burst; a scenario without one refuses them."""
    text = (REPO / "config" / "scenario_001_fmcw_low_prf.toml").read_text(encoding="utf-8")
    path = tmp_path / "fmcw_with_leakage.toml"
    path.write_text(text + "\n[leakage]\nlags = [0]\ninr_db = [40.0]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="noise burst"):
        load_scenario(path)


def test_cancellation_settings_refuse_what_wiener_smi_cannot_do():
    """Wiener-SMI fits the whole CPI on the linear basis; an extension needs a batch."""
    with pytest.raises(ValueError, match="wiener"):
        CancellationSettings("wiener", 192, n_samples_per_batch=400)
    with pytest.raises(ValueError, match="extension"):
        CancellationSettings("least_squares", 192, n_samples_extension=400)


def test_each_noise_frame_carries_its_code_beside_the_stream():
    """frames() hands a noise burst's stream with the code it sent, the same shape."""
    scenario = load_scenario(NOISE_CONFIG)
    frame = next(frames(scenario, n_frames=1))
    assert len(frame.cubes) == len(frame.references) == 1
    assert frame.cubes[0].shape == frame.references[0].shape == (2048, 400)
    product = frame_products(frame, scenario)[0]
    assert product.rd_map.shape == (2048, 1024)


def _short_noise_scenario():
    """P1 cut to 64 batches and 64 lags: the same chain, 25 600 samples a frame."""
    scenario = load_scenario(NOISE_CONFIG)
    return dataclasses.replace(scenario, n_pulses=(64,), n_range_lags=(64,))


def test_a_noise_maps_lags_do_not_wrap():
    """A route past the lag window has no index; inside it, the unwrapped lag."""
    scenario = _short_noise_scenario()
    frame = next(frames(scenario, n_frames=1))
    product = frame_products(frame, scenario)[0]
    lag_m = SPEED_OF_LIGHT_MPS / 8.0e6
    range_index, _ = product.index_of([10.5 * lag_m, 64.2 * lag_m, 130.0 * lag_m], 0.0)
    np.testing.assert_allclose(range_index[0], 10.5, rtol=1e-12)
    assert np.isnan(range_index[1:]).all()
    assert not product.range_wraps


def test_the_cancellers_notch_holds_the_maps_noise_level_not_round_off():
    """Zero Doppler below lag K is exactly emptied by the fit; frame_products floors it to the
    map's noise power, so CFAR finds no round-off there.
    """
    scenario = _short_noise_scenario()
    frame = next(frames(scenario, n_frames=1))
    code, stream = frame.references[0], frame.cubes[0]
    residual = cancel(code, stream, scenario.cancellation).residual.reshape(stream.shape)
    raw = range_doppler_product(residual, scenario.radar, reference=code, n_range_lags=64).rd_map
    noise_w = np.median(np.abs(raw) ** 2) / np.log(2.0)
    # The projection's null: round-off, far under the noise.
    assert np.max(np.abs(raw[32, :64]) ** 2) < 1e-20 * noise_w
    product = frame_products(frame, scenario)[0]
    np.testing.assert_allclose(np.abs(product.rd_map[32, :64]) ** 2, noise_w, rtol=1e-12)
    found = frame_plots([product], scenario.detection)
    rows = np.array([d.centroid_index[0] for d in found.detections])
    assert not np.any(np.abs(rows - 32) < 0.5)


def test_cancellation_settings_refuse_a_loaded_wiener_fit():
    """Wiener-SMI solves the sample statistics as they are: a loading would be dropped."""
    with pytest.raises(ValueError, match="unloaded"):
        CancellationSettings("wiener", 192, diagonal_loading=0.01)


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ('method = "least_squares"\nn_taps = 192\nn_samples_per_batch = 3000\n', "divide"),
        (
            'method = "least_squares"\nn_taps = 192\nn_samples_per_batch = 400\n'
            "n_samples_extension = 200\n",
            "whole batches",
        ),
        ('method = "least_squares"\nn_taps = 500\nn_samples_per_batch = 400\n', "basis"),
    ],
    ids=["batch", "extension", "window"],
)
def test_a_cancellation_that_cannot_tile_the_cpi_fails_at_load(tmp_path, table, message):
    """A batch that does not divide the CPI, a ragged extension or a window short of the basis."""
    text = NOISE_CONFIG.read_text(encoding="utf-8")
    head, tail = text.split("[cancellation]\n")
    tail = tail.split("\n\n", 1)[1]
    path = tmp_path / "bad_cancellation.toml"
    path.write_text(head + "[cancellation]\n" + table + "\n" + tail, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_scenario(path)


def test_the_cancellation_figure_labels_truth_once_and_draws_without_it():
    """One legend entry for two targets; no truth, no legend (an empty legend warns)."""
    radar = load_scenario(NOISE_CONFIG).radar
    rd_map = np.ones((8, 16), dtype=complex)
    range_m = SPEED_OF_LIGHT_MPS * np.arange(16) / 8.0e6
    rate_mps = np.linspace(10.0, -10.0, 8, endpoint=False)
    figure = plots.render_cancellation(
        rd_map,
        rd_map,
        range_m,
        rate_mps,
        radar,
        floor_w=1.0,
        truth_range_m=[400.0, 500.0],
        truth_rate_mps=[0.0, 2.0],
    )
    labels = [t.get_text() for t in figure.axes[1].get_legend().get_texts()]
    assert labels == ["truth"]
    bare = plots.render_cancellation(rd_map, rd_map, range_m, rate_mps, radar, floor_w=1.0)
    assert bare.axes[1].get_legend() is None
    pyplot = pytest.importorskip("matplotlib.pyplot")
    pyplot.close(figure)
    pyplot.close(bare)


# Slow: one frame of P1 through rd and leakage, about five seconds.
@pytest.mark.slow
def test_the_leakage_stage_reuses_rds_frames(tmp_path, monkeypatch):
    """After rd, leakage measures rd's frames and cancellations; alone, it makes its own."""
    calls = []
    original = pipeline_module.frames

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "frames", counted)
    scenario = load_scenario(NOISE_CONFIG)
    together = evaluate.run(scenario, ["rd", "leakage"], n_frames=1, out_dir=tmp_path, render=False)
    assert calls == []
    alone = evaluate.run(scenario, ["leakage"], n_frames=1, out_dir=tmp_path, render=False)
    assert calls == [1]
    # The same seed, the same frame: the same numbers either way.
    for key in ("cancellation_depth_db[0]", "floor_rise_db[0]", "sinr_loss_db[0]"):
        assert together.values[key] == alone.values[key]
