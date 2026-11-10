"""The receive array: geometry, the steering sign, weights, covariances and the channel axis.

Analytic ground truth throughout: broadside and endfire steering vectors, each element's own
path length to a far target, rank-one covariances of one source, Capon's constraint, and the
generators' channel axis against their one-channel cube. Scenario 006's construction of aircraft
B and its scattering phase are checked on the F2 config itself.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from radar_forge import evaluate
from radar_forge.array import (
    ArrayGeometry,
    apply_weights,
    beam_power,
    conventional_weights,
    direction_cosine,
    directions_unit,
    enu_to_array_frame,
    forward_backward_average,
    is_centro_symmetric,
    mvdr_weights,
    offset_direction_cosine,
    rayleigh_width,
    sample_covariance,
    spatial_smoothing,
    steering_vectors,
    subarray,
    uniform_linear_array,
)
from radar_forge.pipeline import range_doppler_product, synthesise
from radar_forge.propagation import line_of_sight_paths
from radar_forge.radar import Radar, Receiver, Site, TargetGeometry, Transmitter
from radar_forge.scenario import load_scenario, scattering_phase_rad, truth_geometry
from radar_forge.signal import fmcw_deramp_baseband, pulsed_baseband
from radar_forge.stages import array as array_stage

REPO = Path(__file__).resolve().parents[1]
F2 = REPO / "config" / "scenario_006_parallel_subrayleigh.toml"
WAVELENGTH_M = 0.030591067142857142  # 9.8 GHz
ULA = uniform_linear_array(16, 0.5, WAVELENGTH_M, boresight_azimuth_deg=235.795)


def _toward(u: float) -> np.ndarray:
    """Return the array-frame unit vector at direction cosine ``u`` in the horizontal plane."""
    return np.array([[u, np.sqrt(1.0 - u**2), 0.0]])


def test_broadside_steering_vector_is_all_ones_of_norm_n_rx():
    """A1: a(u = 0) is all ones, ||a||^2 = n_rx."""
    a = steering_vectors(ULA, _toward(0.0), WAVELENGTH_M)[:, 0]
    # A few float64 operations from exp(0): rtol 1e-12.
    np.testing.assert_allclose(a, np.ones(16), rtol=1e-12)
    np.testing.assert_allclose(np.vdot(a, a).real, 16.0, rtol=1e-12)


def test_endfire_phase_step_is_pi_and_the_element_at_larger_x_leads():
    """A2: at u = +1 adjacent elements step by +pi at d = lambda/2; for u > 0, larger x leads."""
    a = steering_vectors(ULA, _toward(1.0), WAVELENGTH_M)[:, 0]
    # Each step is a(n+1)/a(n) = exp(+j pi): rtol 1e-12 on the complex ratio.
    np.testing.assert_allclose(a[1:] / a[:-1], -np.ones(15), rtol=1e-12)
    step = np.angle(steering_vectors(ULA, _toward(0.2), WAVELENGTH_M)[1:, 0])
    step -= np.angle(steering_vectors(ULA, _toward(0.2), WAVELENGTH_M)[:-1, 0])
    # 2 pi d u / lambda = 0.2 pi, positive: the element further along +x is ahead.
    np.testing.assert_allclose(np.mod(step, 2 * np.pi), 0.2 * np.pi, rtol=1e-12)


def test_steering_phase_is_each_elements_own_path_phase_to_a_far_target():
    """Ground truth from geometry: angle(a_n) = -2 pi (R_n - R_centre) / lambda at 1000 km."""
    bearing, elevation = np.radians(241.0), np.radians(4.5)
    target_enu_m = 1.0e6 * directions_unit(bearing, elevation)[0]
    # Each element's ENU position: its array-frame position on the array's axes.
    axes = enu_to_array_frame(ULA, np.eye(3)).T
    elements_enu_m = ULA.element_positions_m @ axes
    path_m = np.linalg.norm(target_enu_m - elements_enu_m, axis=1)
    path_phase = -2.0 * np.pi * (path_m - np.linalg.norm(target_enu_m)) / WAVELENGTH_M
    in_array = enu_to_array_frame(ULA, directions_unit(bearing, elevation))
    a = steering_vectors(ULA, in_array, WAVELENGTH_M)[:, 0]
    # The far-field residual is pi x^2 / (lambda R) <= 2e-6 rad over the 0.24 m aperture.
    np.testing.assert_allclose(np.angle(a * np.exp(-1j * path_phase)), 0.0, atol=1e-5)


def test_azimuth_from_boresight_is_the_bearing_less_the_boresight_without_tilt():
    """D13 item 6: with no tilt, the array-frame azimuth is the bearing minus 235.795 deg."""
    bearing_deg = np.array([200.0, 235.795, 250.0, 300.0])
    in_array = enu_to_array_frame(ULA, directions_unit(np.radians(bearing_deg), 0.1))
    azimuth_deg = np.degrees(np.arctan2(in_array[:, 0], in_array[:, 1]))
    # A rotation of unit vectors: float64 round-off, 1e-9 deg.
    np.testing.assert_allclose(azimuth_deg, bearing_deg - 235.795, atol=1e-9)
    np.testing.assert_allclose(in_array[:, 2], np.sin(0.1), atol=1e-12)


def test_uniform_linear_array_is_centred_and_conjugate_centro_symmetric():
    """D13 item 5: phase centre at the origin, so J a* = a."""
    assert is_centro_symmetric(ULA)
    a = steering_vectors(ULA, _toward(0.37), WAVELENGTH_M)[:, 0]
    np.testing.assert_allclose(a[::-1].conj(), a, rtol=1e-12)
    assert not is_centro_symmetric(ULA.__class__(ULA.element_positions_m + 0.01))


def test_rayleigh_width_is_one_eighth_for_sixteen_elements_at_half_a_wavelength():
    """lambda / (n_rx d) = 2 / 16."""
    assert rayleigh_width(ULA, WAVELENGTH_M) == pytest.approx(0.125, rel=1e-12)


def test_sample_covariance_conjugates_the_second_index():
    """R = mean of x x^H: entry (i, j) is x_i x_j*, tested on a non-symmetric steering vector."""
    a = steering_vectors(ULA, _toward(0.3), WAVELENGTH_M)[:, 0]
    snapshots = np.outer([2.0, 1j, -1.5], a)
    r = sample_covariance(snapshots)
    power = np.mean(np.abs([2.0, 1j, -1.5]) ** 2)
    np.testing.assert_allclose(r, power * np.outer(a, a.conj()), rtol=1e-12)
    assert r[0, 1] == pytest.approx(power * a[0] * np.conj(a[1]), rel=1e-12)


def test_spatial_smoothing_of_one_source_is_its_subarrays_covariance():
    """Every subarray of a ULA sees one source alike: the average is a_L a_L^H exactly."""
    a = steering_vectors(ULA, _toward(-0.2), WAVELENGTH_M)[:, 0]
    smoothed = spatial_smoothing(np.outer(a, a.conj()), 12)
    a_sub = steering_vectors(subarray(ULA, 12), _toward(-0.2), WAVELENGTH_M)[:, 0]
    np.testing.assert_allclose(smoothed, np.outer(a_sub, a_sub.conj()), rtol=1e-12, atol=1e-12)


def test_forward_backward_keeps_one_source_and_refuses_an_uncentred_array():
    """J R* J = R for one source on a centred array; an uncentred one cannot be averaged."""
    a = steering_vectors(ULA, _toward(0.41), WAVELENGTH_M)[:, 0]
    r = np.outer(a, a.conj())
    np.testing.assert_allclose(forward_backward_average(r, ULA), r, rtol=1e-12, atol=1e-12)
    shifted = ArrayGeometry(ULA.element_positions_m + np.array([0.05, 0.0, 0.0]))
    with pytest.raises(ValueError, match="centro-symmetric"):
        forward_backward_average(r, shifted)


def test_mvdr_is_distortionless_and_white_noise_mvdr_is_the_conventional_beam():
    """A20: w(u0)^H a(u0) = 1; with R = sigma^2 I, MVDR equals the conventional weights."""
    a = steering_vectors(ULA, np.concatenate([_toward(0.1), _toward(-0.3)]), WAVELENGTH_M)
    interference = steering_vectors(ULA, _toward(0.4), WAVELENGTH_M)
    r = 1e3 * interference @ interference.conj().T + 2.0 * np.eye(16)
    w = mvdr_weights(r, a)
    # The constraint is exact to float64 solves: rtol 1e-12.
    np.testing.assert_allclose(np.sum(w.conj() * a, axis=0), [1.0, 1.0], rtol=1e-12)
    np.testing.assert_allclose(
        mvdr_weights(2.0 * np.eye(16), a), conventional_weights(a), rtol=1e-12
    )
    assert np.all(beam_power(w, r) < beam_power(conventional_weights(a), r))


def test_apply_weights_forms_w_hermitian_x_along_the_channel_axis():
    """y = w^H x along the last axis, or a named one."""
    rng = np.random.default_rng(6)
    cube = rng.normal(size=(3, 4, 16)) + 1j * rng.normal(size=(3, 4, 16))
    w = rng.normal(size=16) + 1j * rng.normal(size=16)
    np.testing.assert_allclose(apply_weights(w, cube), np.einsum("n,ijn->ij", w.conj(), cube))
    moved = np.moveaxis(cube, -1, 0)
    np.testing.assert_allclose(apply_weights(w, moved, axis=0), apply_weights(w, cube))


def test_offset_direction_cosine_moves_u_and_keeps_the_array_frame_elevation():
    """Aircraft B: u moves by the offset, the array-frame elevation stays."""
    aoa_rad = np.radians([[232.3, 4.4], [238.6, 4.9]])
    moved = offset_direction_cosine(ULA, aoa_rad, 0.0625)
    np.testing.assert_allclose(
        direction_cosine(ULA, moved) - direction_cosine(ULA, aoa_rad), 0.0625, rtol=1e-12
    )
    np.testing.assert_allclose(moved[:, 1], aoa_rad[:, 1], rtol=1e-12)
    with pytest.raises(ValueError, match="visible region"):
        offset_direction_cosine(ULA, aoa_rad, 1.5)


# The generators' channel axis: a small FMCW radar and a pulsed one, two paths.
@pytest.fixture
def radars() -> tuple[Radar, Radar]:
    site = Site(36.0025, -78.941, 60.0)
    fmcw = Radar(
        Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-4, 1.0e4), Receiver(1.0e6, 20.0, 3.0), site
    )
    pulsed = Radar(
        Transmitter(9.8e9, 1.0e6, 1.0e3, 30.0, 4.0e-6, 2.0e3, "pulsed"),
        Receiver(2.0e6, 20.0, 3.0),
        site,
    )
    return fmcw, pulsed


def _paths(radar: Radar, scattering_rad: float = 0.0):
    geometry = TargetGeometry.monostatic(
        [6000.0, 6000.0], [-40.0, -40.0], azimuth_rad=[0.1, 0.25], elevation_rad=0.05
    )
    return line_of_sight_paths(
        radar, geometry, [10.0, 4.0], scattering_phase_rad=[0.0, scattering_rad]
    )


@pytest.mark.parametrize("waveform", ["fmcw", "pulsed"])
def test_each_channel_of_a_noiseless_cube_is_the_paths_times_their_responses(radars, waveform):
    """With rx_response_linear, channel n is the sum over paths of each path's cube times r_pn."""
    radar = radars[waveform == "pulsed"]
    generate = fmcw_deramp_baseband if waveform == "fmcw" else pulsed_baseband
    paths = _paths(radar)
    response = np.array([[1.0, 1j, -0.5], [0.5 - 0.5j, 2.0, 1.0]])
    cube = generate(paths, radar, 8, rng=None, rx_response_linear=response)
    singles = [
        generate(dataclasses.replace(paths, **_one(paths, p)), radar, 8, rng=None) for p in (0, 1)
    ]
    expected = singles[0][..., None] * response[0] + singles[1][..., None] * response[1]
    assert cube.shape == (*singles[0].shape, 3)
    # The same products summed in another order: rtol 1e-12.
    np.testing.assert_allclose(cube, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


def _one(paths, p: int) -> dict:
    """Return the fields of one path of ``paths``, for dataclasses.replace."""
    return {f.name: getattr(paths, f.name)[p : p + 1] for f in dataclasses.fields(paths)}


def test_a_zero_scattering_phase_is_the_default_bit_for_bit(radars):
    """A12 at the generator: zero phase and no response leave the one-channel cube as it was."""
    fmcw, _ = radars
    geometry = TargetGeometry.monostatic([6000.0, 6100.0], [-40.0, 25.0], azimuth_rad=[0.1, 0.25])
    default = line_of_sight_paths(fmcw, geometry, [10.0, 4.0])
    zero = line_of_sight_paths(fmcw, geometry, [10.0, 4.0], scattering_phase_rad=0.0)
    plain = fmcw_deramp_baseband(default, fmcw, 8, rng=np.random.default_rng(1))
    again = fmcw_deramp_baseband(
        zero, fmcw, 8, rng=np.random.default_rng(1), rx_response_linear=None
    )
    assert np.array_equal(plain, again)


def test_a_scattering_phase_turns_its_paths_echo_by_that_phase(radars):
    """The second path alone, with phase 1.2 rad, is the zero-phase echo times exp(1.2 j)."""
    fmcw, _ = radars
    turned = _one(_paths(fmcw, 1.2), 1)
    plain = _one(_paths(fmcw, 0.0), 1)
    paths = _paths(fmcw)
    cube = fmcw_deramp_baseband(dataclasses.replace(paths, **turned), fmcw, 8, rng=None)
    reference = fmcw_deramp_baseband(dataclasses.replace(paths, **plain), fmcw, 8, rng=None)
    # The phase is added inside an argument of 2.5e6 rad (12 km over 0.03 m), whose float64
    # spacing is 4.7e-10 rad: rtol 2e-9.
    np.testing.assert_allclose(cube, reference * np.exp(1.2j), rtol=2e-9)


def test_channel_noise_is_independent_at_the_per_sample_power(radars):
    """No signal: each channel's noise has power N0 f_s, and the channels decorrelate."""
    fmcw, _ = radars
    paths = line_of_sight_paths(fmcw, TargetGeometry.monostatic([6000.0], [0.0]), [0.0])
    cube = fmcw_deramp_baseband(
        paths, fmcw, 64, rng=np.random.default_rng(2), rx_response_linear=np.ones((1, 4))
    )
    samples = cube.reshape(-1, 4)
    power = np.mean(np.abs(samples) ** 2, axis=0) / fmcw.receiver.noise_power_per_sample_w
    # 6400 samples a channel: the power's standard error is 1/80; five of them.
    np.testing.assert_allclose(power, 1.0, atol=5 / 80)
    correlation = np.abs(sample_covariance(samples)) / np.mean(np.abs(samples) ** 2)
    # Off the diagonal, each magnitude has standard deviation about 1/80 too.
    assert np.max(correlation[~np.eye(4, dtype=bool)]) < 5 / 80


def test_a_noise_burst_refuses_a_receive_array():
    """A noise radar's stream has one channel: synthesise says so rather than drop the array."""
    noise = Radar(
        Transmitter(9.8e9, 1.0e6, 100.0, 30.0, 1.0e-3, 1.0e3, "noise"),
        Receiver(1.0e6, 20.0, 3.0),
        Site(36.0025, -78.941, 60.0),
    )
    geometry = TargetGeometry.monostatic([6000.0], [-40.0])
    with pytest.raises(ValueError, match="one channel"):
        synthesise(noise, geometry, [10.0], 4, rng=None, array=ULA)


def test_a_channel_map_keeps_the_channel_axis_and_detects_on_its_boresight_beam():
    """The 3-D map keeps n_rx; rd_map is the channels' mean; a 2-D cube's map is unchanged."""
    scenario = load_scenario(F2)
    truth = truth_geometry(dataclasses.replace(scenario, duration_s=1.0))[0]
    cube = synthesise(scenario.radar, truth, scenario.rcs_m2, 256, rng=None, array=scenario.array)
    product = range_doppler_product(cube, scenario.radar)
    assert product.channels is not None and product.channels.shape == (256, 1000, 16)
    np.testing.assert_allclose(product.rd_map, product.channels.mean(axis=-1), rtol=1e-12)
    single = range_doppler_product(cube[..., 0], scenario.radar)
    assert single.channels is None
    # The same transforms, batched differently: rtol 1e-12 of the peak.
    peak = np.abs(single.rd_map).max()
    np.testing.assert_allclose(single.rd_map, product.channels[..., 0], atol=1e-12 * peak)


def test_aircraft_b_shares_aircraft_as_range_and_rate_at_half_a_rayleigh_width():
    """Spec 4.3: B has A's ranges, rate and elevation, and u + 0.5 x 0.125."""
    scenario = load_scenario(F2)
    for frame in truth_geometry(scenario)[:5]:
        np.testing.assert_array_equal(frame.bistatic_range_m[1], frame.bistatic_range_m[0])
        np.testing.assert_array_equal(
            frame.bistatic_range_rate_mps[1], frame.bistatic_range_rate_mps[0]
        )
        u = direction_cosine(scenario.array, frame.aoa_rad)
        assert u[1] - u[0] == pytest.approx(0.0625, rel=1e-12)
    phases = scattering_phase_rad(scenario)
    assert phases.shape == (120, 2) and not phases[:, 0].any()
    assert np.all((phases[:, 1] >= 0.0) & (phases[:, 1] < 2.0 * np.pi))


def test_truth_csv_carries_both_aircraft_and_their_direction_cosines(tmp_path):
    """The geometry stage writes a row per aircraft per frame, u and the off-boresight azimuth."""
    evaluate.run(load_scenario(F2), ["geometry"], n_frames=2, out_dir=tmp_path, render=False)
    with (tmp_path / "truth.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["target"] for row in rows] == ["aircraft-a", "aircraft-b"] * 2
    u = np.array([float(row["direction_cosine_x"]) for row in rows]).reshape(2, 2)
    # The CSV writes float64 by repr, which round-trips exactly: rtol 1e-12 on the difference.
    np.testing.assert_allclose(u[:, 1] - u[:, 0], 0.0625, rtol=1e-12)
    azimuth_deg = np.array([float(row["azimuth_boresight_deg"]) for row in rows])
    # The cone angle asin(u), the estimates' convention: a linear array sees no elevation.
    np.testing.assert_allclose(np.sin(np.radians(azimuth_deg)), u.ravel(), rtol=1e-9)


def test_the_loader_refuses_an_array_off_fmcw_and_aircraft_b_without_one(tmp_path):
    """[receive_array] needs FMCW bursts; [second_aircraft] and [doa] need [receive_array]."""
    text = F2.read_text(encoding="utf-8").replace(
        "../data/", str(REPO / "data").replace("\\", "/") + "/"
    )
    path = tmp_path / "pulsed.toml"
    pulsed = text.replace('waveform = "fmcw"', 'waveform = "pulsed"')
    path.write_text(pulsed.replace("sample_rate_hz = 1.0e6", "sample_rate_hz = 2.0e6"), "utf-8")
    with pytest.raises(ValueError, match="FMCW"):
        load_scenario(path)
    no_array = text.split("[receive_array]")[0] + "[receiver]" + text.split("[receiver]")[1]
    path.write_text(no_array, encoding="utf-8")
    with pytest.raises(ValueError, match="need a \\[receive_array\\]"):
        load_scenario(path)


def test_doa_csv_truth_rows_carry_the_link_budgets_cell_snr(tmp_path):
    """Truth rows' snr_db is the per-channel cell SNR, F2's 42.01 dB at frame 0 for both."""
    scenario = load_scenario(F2)
    ctx = evaluate.Context(scenario, tmp_path, 1, False, evaluate.Report(scenario.name))
    truth = truth_geometry(dataclasses.replace(scenario, duration_s=1.0))[0]
    u = direction_cosine(scenario.array, truth.aoa_rad)
    rows = array_stage._truth_rows(ctx, 0, truth, u, scenario.array)
    # The config's expected_snr_db and its tolerance: the radar equation by hand.
    assert [row["snr_db"] for row in rows] == pytest.approx([42.01, 42.01], abs=0.01)
    assert abs(rows[0]["range_rate_mps_folded"]) <= 30.5911 / 4  # folded: half the radial span


SECOND_AIRCRAFT = (
    '[second_aircraft]\nname = "aircraft-b"\nseparation_rayleigh = 0.5\nrcs_dbsm = 10.0\n'
)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("n_sources = 2", "n_sources = 3", "n_sources = 2"),
        (SECOND_AIRCRAFT + 'scattering_phase = "uniform"', "", r"needs \[second_aircraft\]"),
        ('methods = ["bartlett"', "methods = [] #", "at least one"),
        ("n_subarray_elements = 12", "n_subarray_elements = 16", "n_subarray_elements < n_rx"),
        ("half_width_bins = 2", "half_width_bins = 1", "more than n_rx"),
        (
            'baseline_methods = ["music", "esprit"]',
            'baseline_methods = ["fb_esprit"]',
            "does not run",
        ),
    ],
    ids=["three sources", "no aircraft B", "no method", "one subarray", "nine cells", "unrun"],
)
def test_the_loader_refuses_a_doa_table_the_stage_cannot_score(tmp_path, old, new, match):
    """Two sources, B, a method, two subarrays, more cells than channels, scored methods run."""
    text = F2.read_text(encoding="utf-8").replace(
        "../data/", str(REPO / "data").replace("\\", "/") + "/"
    )
    assert old in text
    path = tmp_path / "doa.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_scenario(path)
