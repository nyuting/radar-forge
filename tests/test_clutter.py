"""Least-squares leakage and clutter cancellation (``radar_forge.clutter``), scenario 005.

The unit tests hold the projection to closed forms on short streams: the Gram matrix against
the explicit basis, the orthogonality of the residual (A2), the degeneracy chain from sliding
to batched to full-CPI to Wiener (A1), and the Doppler notch (A4, A5). The slow tests run the
scenario's own frames for the comparisons its configs cannot hold, one method each: Wiener-SMI
(A3), too few taps (A7), a full-CPI fit of moving clutter and a linear fit of a distorted
transmitter (A9, A11, A13), and the sliding fit against the batched one (A12).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from radar_forge import clutter, evaluate
from radar_forge.clutter import (
    cancellation_depth_db,
    delay_basis_gram,
    distortion_fraction_linear,
    doppler_notch_response_linear,
    expected_cancellation_depth_db,
    least_squares_cancel,
    ls_cancel,
    ls_cancel_batched,
    ls_cancel_sliding,
    ls_weights,
    parallel_hammerstein_basis,
    wiener_smi,
)
from radar_forge.pipeline import cancel, frames, range_doppler_product
from radar_forge.scenario import load_scenario
from radar_forge.signal import cubic_amplifier, noise_waveform

REPO = Path(__file__).resolve().parents[1]
SAMPLE_RATE_HZ = 8.0e6
N_TAPS = 6
N_SHORT = 4000


@pytest.fixture
def code(rng):
    """A short unit-power noise code, 4000 samples."""
    return noise_waveform(N_SHORT, rng=rng)


@pytest.fixture
def received(code, rng):
    """Leakage, an echo and a slowly turning echo of ``code``, through an amplifier, and noise."""
    radiated = cubic_amplifier(code, -0.07)
    turning = np.exp(2j * np.pi * 3e-5 * np.arange(N_SHORT))
    noise = (rng.standard_normal(N_SHORT) + 1j * rng.standard_normal(N_SHORT)) / math.sqrt(2)
    return (
        100.0 * radiated
        + 30.0 * np.roll(radiated, 2)
        + 10.0 * np.roll(radiated, 4) * turning
        + noise
    )


def _explicit_fit(code, signal, n_taps, odd_orders, start, stop):
    """Return the least-squares residual of rows [start, stop) by lstsq on the explicit basis."""
    basis = parallel_hammerstein_basis(code, n_taps, odd_orders, start, stop)
    weights, *_ = np.linalg.lstsq(basis, signal[start:stop], rcond=None)
    return signal[start:stop] - basis @ weights, basis


@pytest.mark.parametrize("odd_orders", [(1,), (1, 3)], ids=["linear", "hammerstein"])
@pytest.mark.parametrize(("start", "stop"), [(0, N_SHORT), (0, 900), (1700, 2600), (3100, N_SHORT)])
def test_gram_and_cross_equal_the_explicit_basis_products(code, received, odd_orders, start, stop):
    """The FFT-and-displacement normal equations are X^H X and X^H s, the circular wrap included."""
    basis = parallel_hammerstein_basis(code, N_TAPS, odd_orders, start, stop)
    gram, cross = delay_basis_gram(
        code, received, N_TAPS, start=start, stop=stop, odd_orders=odd_orders
    )
    # FFT round-off on sums of thousands of products: 1e-12 of the largest entry.
    scale = np.max(np.abs(gram))
    np.testing.assert_allclose(gram, basis.conj().T @ basis, rtol=0, atol=1e-12 * scale)
    scale = np.max(np.abs(cross))
    np.testing.assert_allclose(
        cross, basis.conj().T @ received[start:stop], rtol=0, atol=1e-12 * scale
    )


@pytest.mark.parametrize("odd_orders", [(1,), (1, 3)], ids=["linear", "hammerstein"])
def test_residual_is_orthogonal_to_every_batch_basis(code, received, odd_orders):
    """A2: X_b^H y_b = 0 for each batch, to float64 round-off on a well-conditioned solve."""
    result = least_squares_cancel(
        code, received, N_TAPS, n_samples_per_batch=1000, odd_orders=odd_orders
    )
    for start in range(0, N_SHORT, 1000):
        basis = parallel_hammerstein_basis(code, N_TAPS, odd_orders, start, start + 1000)
        residual = result.residual[start : start + 1000]
        ratio = np.linalg.norm(basis.conj().T @ residual) / (
            np.linalg.norm(basis) * np.linalg.norm(received[start : start + 1000])
        )
        # The spec's bound: round-off of a Cholesky solve at a condition number below 100.
        assert ratio <= 1e-10


@pytest.mark.parametrize("odd_orders", [(1,), (1, 3)], ids=["linear", "hammerstein"])
def test_full_cpi_fit_equals_least_squares_on_the_explicit_basis(code, received, odd_orders):
    """The structured solve is lstsq on X; with (1,) it is the linear canceller (A1, last link)."""
    expected, _ = _explicit_fit(code, received, N_TAPS, odd_orders, 0, N_SHORT)
    # Two solves of the same normal equations: rtol 1e-9, as the spec's A1.
    np.testing.assert_allclose(
        ls_cancel(code, received, N_TAPS, odd_orders=odd_orders).residual,
        expected,
        rtol=1e-9,
        atol=1e-9 * np.max(np.abs(expected)),
    )


def test_sliding_without_extension_is_the_batched_fit(code, received):
    """A1: a block estimated on itself is a batch."""
    sliding = ls_cancel_sliding(code, received, N_TAPS, 500, 0).residual
    batched = ls_cancel_batched(code, received, N_TAPS, 500).residual
    np.testing.assert_allclose(sliding, batched, rtol=1e-9, atol=0)


def test_one_batch_is_the_full_cpi_fit(code, received):
    """A1: a batch as long as the CPI is the full-CPI fit."""
    one_batch = ls_cancel_batched(code, received, N_TAPS, N_SHORT).residual
    np.testing.assert_allclose(one_batch, ls_cancel(code, received, N_TAPS).residual, rtol=1e-9)


def test_full_cpi_fit_is_the_wiener_solution_of_the_exact_statistics(code, received):
    """A1: w = R^-1 r with R = X^H X / N and r = X^H s / N, the CPI's own statistics."""
    basis = parallel_hammerstein_basis(code, N_TAPS)
    weights = np.linalg.solve(basis.conj().T @ basis / N_SHORT, basis.conj().T @ received / N_SHORT)
    expected = received - basis @ weights
    # Two solves of the same normal equations, scaled by N: rtol 1e-9.
    np.testing.assert_allclose(
        ls_cancel(code, received, N_TAPS).residual,
        expected,
        rtol=1e-9,
        atol=1e-9 * np.max(np.abs(expected)),
    )


def test_sliding_fit_filters_each_block_with_its_widened_windows_weights(code, received):
    """Block b's residual is s_b - X_b w, w fitted by lstsq on the block widened and clipped."""
    result = ls_cancel_sliding(code, received, N_TAPS, 500, 1000)
    for block in range(N_SHORT // 500):
        low, high = max(0, 500 * block - 1000), min(N_SHORT, 500 * block + 1500)
        window = parallel_hammerstein_basis(code, N_TAPS, (1,), low, high)
        weights, *_ = np.linalg.lstsq(window, received[low:high], rcond=None)
        rows = parallel_hammerstein_basis(code, N_TAPS, (1,), 500 * block, 500 * block + 500)
        expected = received[500 * block : 500 * block + 500] - rows @ weights
        # Two solves of the same normal equations.
        np.testing.assert_allclose(
            result.residual[500 * block : 500 * block + 500],
            expected,
            rtol=1e-9,
            atol=1e-9 * np.max(np.abs(expected)),
        )
    assert result.weights.shape == (8, N_TAPS)
    assert result.gram_condition_number.shape == (8,)


def test_wiener_smi_matches_least_squares_up_to_edge_terms(code, received):
    """Wiener-SMI zero-pads the record (Haykin §9.2): residual power within O(K/N) of LS."""
    wiener = np.sum(np.abs(wiener_smi(code, received, N_TAPS).residual) ** 2)
    least = np.sum(np.abs(ls_cancel(code, received, N_TAPS).residual) ** 2)
    assert least <= wiener  # least squares is the minimum
    # Edge terms: the zero-padded statistics miss lag k's first k products, a few K/N.
    assert wiener / least - 1.0 < 10 * N_TAPS / N_SHORT


def test_a_probe_inside_the_tap_span_is_removed_noiselessly(code):
    """A5: a static echo at a lag below K leaves nothing but round-off, under -100 dB."""
    echo = 3.0 * np.roll(code, 4)
    residual = ls_cancel(code, echo, N_TAPS).residual
    assert cancellation_depth_db(echo, residual) > 100.0


def _probe_response(code, doppler_hz, n_samples_per_batch):
    """Return a lag-0 probe's matched-filter power after a K = 1 fit, and the realised rho."""
    probe = code * np.exp(2j * np.pi * doppler_hz * np.arange(code.size) / SAMPLE_RATE_HZ)
    residual = ls_cancel_batched(code, probe, 1, n_samples_per_batch).residual
    measured = abs(np.vdot(probe, residual)) ** 2 / np.vdot(probe, probe).real ** 2
    energy = np.sum((np.abs(code) ** 2).reshape(-1, n_samples_per_batch), axis=1)
    overlap = np.sum((np.conj(code) * probe).reshape(-1, n_samples_per_batch), axis=1)
    realised = np.sum(energy - np.abs(overlap) ** 2 / energy) ** 2 / np.sum(energy) ** 2
    return measured, realised


@pytest.fixture(scope="module")
def cpi_code():
    """A unit-power code as long as scenario 005's CPI, 819 200 samples."""
    return noise_waveform(819_200, rng=np.random.default_rng(20261106))


@pytest.mark.parametrize(
    ("doppler_hz", "n_samples_per_batch", "expected_db", "tolerance_db"),
    [
        # Full CPI: 9.765625 Hz is the first null of D, so nothing is removed (A5's neighbour).
        (9.765625, 819_200, 0.0, 0.1),
        (-9.765625, 819_200, 0.0, 0.1),
        # 6.4 ms batches: inside the notch, half way out, and on its first null (A4).
        (9.765625, 51_200, -37.866, 1.0),
        (78.125, 51_200, -4.514, 0.3),
        (156.25, 51_200, 0.0, 0.3),
    ],
)
def test_doppler_notch_follows_the_dirichlet_kernel(
    cpi_code, doppler_hz, n_samples_per_batch, expected_db, tolerance_db
):
    """A4, A5: rho(f) = (1 - |D|^2)^2 exactly from the realised code, and in expectation.

    The tolerances are the spec's: batch-coefficient scatter of order 1/sqrt(N_B) moves the
    deep notch by under 1 dB and the rest by under 0.3; at a null |D|^2 is O(1/N).
    """
    measured, realised = _probe_response(cpi_code, doppler_hz, n_samples_per_batch)
    # The same projection written two ways: rtol 1e-9.
    np.testing.assert_allclose(measured, realised, rtol=1e-9)
    expected = doppler_notch_response_linear(doppler_hz, n_samples_per_batch, SAMPLE_RATE_HZ)
    np.testing.assert_allclose(10 * np.log10(expected), expected_db, atol=0.001)
    assert abs(10 * np.log10(measured) - expected_db) <= tolerance_db


def test_notch_is_total_at_zero_doppler_and_absent_at_a_null():
    """D(0) = 1 removes all of a static echo; D(k f_s / N_c) = 0 removes none."""
    assert doppler_notch_response_linear(0.0, 51_200, SAMPLE_RATE_HZ) == 0.0
    np.testing.assert_allclose(
        doppler_notch_response_linear(np.array([156.25, 312.5]), 51_200, SAMPLE_RATE_HZ), 1.0
    )


def test_cancellation_bound_with_taps_short_of_the_far_clutter():
    """A7's closed form: the lag-160 echo, 12 dB, escapes 96 taps: 28.680 dB."""
    inr_linear = sum(10 ** (db / 10) for db in (40, 30, 20, 30, 24, 18, 12))
    value = expected_cancellation_depth_db(inr_linear, 96, 819_200, 10**1.2)
    np.testing.assert_allclose(value, 28.680, atol=0.0005)


def test_distortion_fraction_is_what_the_bussgang_gain_leaves(rng):
    """1 - beta^2 with beta = E[f(x) x*], estimated from a million Gaussian samples."""
    samples = noise_waveform(1_000_000, rng=rng)
    beta = np.mean(cubic_amplifier(samples, -0.07) * np.conj(samples)).real
    # beta's estimate scatters about 1e-3 over a million samples: 5 sigma on 1 - beta^2.
    np.testing.assert_allclose(1.0 - beta**2, distortion_fraction_linear(-0.07), atol=5e-3)


def test_weights_solve_the_loaded_normal_equations():
    """(G + delta mean(diag G) I) w = c, and an identity Gram has condition number one."""
    gram = np.array([[2.0, 0.5j], [-0.5j, 1.0]])
    cross = np.array([1.0, 2.0j])
    weights, condition = ls_weights(gram, cross, diagonal_loading=0.1)
    np.testing.assert_allclose((gram + 0.15 * np.eye(2)) @ weights, cross, rtol=1e-12)
    assert condition >= 1.0
    _, unit = ls_weights(4.0 * np.eye(3), np.ones(3))
    np.testing.assert_allclose(unit, 1.0, rtol=1e-12)


def test_hammerstein_pair_of_one_tap_has_the_gaussian_moment_gram(cpi_code):
    """[x, |x|^2 x] has Gram N [[1, 2], [2, 6]] (E|x|^4 = 2, E|x|^6 = 6), 2-norm condition 22.46."""
    gram, _ = delay_basis_gram(
        cpi_code, cpi_code, 1, start=0, stop=cpi_code.size, odd_orders=(1, 3)
    )
    # Sample moments of 819 200 draws: |x|^6 scatters about 2 % of its mean.
    np.testing.assert_allclose(gram.real / cpi_code.size, [[1, 2], [2, 6]], rtol=0.03, atol=0.01)
    np.testing.assert_allclose(np.linalg.cond(np.array([[1, 2], [2, 6]])), 22.46, atol=0.005)


def test_bad_orders_and_block_lengths_are_refused(code, received):
    with pytest.raises(ValueError, match="odd_orders"):
        ls_cancel(code, received, N_TAPS, odd_orders=(1, 2))
    with pytest.raises(ValueError, match="divide"):
        ls_cancel_batched(code, received, N_TAPS, 3000)
    with pytest.raises(ValueError, match="whole blocks"):
        ls_cancel_sliding(code, received, N_TAPS, 500, 250)
    with pytest.raises(ValueError, match="n_taps"):
        ls_cancel(code, received, N_SHORT + 1)


# --- Slow: scenario 005's own frames, 819 200 samples each, through one other method. ---


def _variant(config, verify, **cancellation):
    """Return a 005 config with its [cancellation] changed and only the leakage stage checked."""
    scenario = load_scenario(REPO / "config" / config)
    settings = dataclasses.replace(scenario.cancellation, **cancellation)
    return dataclasses.replace(
        scenario, cancellation=settings, verify={"stages": ["leakage"], **verify}
    )


@pytest.fixture(scope="module")
def static_frame():
    """P1's frame 0: the stream and the code."""
    scenario = load_scenario(REPO / "config" / "scenario_005_static_leakage.toml")
    frame = next(frames(scenario, n_frames=1))
    return frame.references[0].reshape(-1), frame.cubes[0].reshape(-1)


# Slow: one frame through Wiener-SMI and least squares, a second each.
@pytest.mark.slow
def test_wiener_smi_matches_least_squares_on_the_static_scene(static_frame):
    """A3: residual powers agree to rtol 1e-3 (edge terms O(K/N) = 2.3e-4, four times over)."""
    code, stream = static_frame
    wiener = np.sum(np.abs(wiener_smi(code, stream, 192).residual) ** 2)
    least = np.sum(np.abs(ls_cancel(code, stream, 192).residual) ** 2)
    np.testing.assert_allclose(wiener, least, rtol=1e-3)


# Slow: five frames of the static scene with 96 taps.
@pytest.mark.slow
def test_96_taps_cannot_reach_the_lag_160_echo(tmp_path):
    """A7: CA 28.680 dB and the floor 12.27 dB up; the target still detected every frame."""
    verify = {
        # As A6: ten standard errors of a power over 819 200 samples.
        "cancellation_depth_db": 28.680,
        "cancellation_tolerance_db": 0.05,
        # 10 log10((1 - 96/N)(1 + 15.85)) = 12.27 dB; 216 cells of floor, as A9.
        "floor_rise_db": 12.27,
        "floor_rise_tolerance_db": 1.0,
        # Predicted SINR 14.4-15.0 dB: Pd >= 0.987 a frame.
        "detections_per_frame": 1,
    }
    scenario = _variant("scenario_005_static_leakage.toml", verify, n_taps=96)
    evaluate.run(scenario, n_frames=5, out_dir=tmp_path, render=False)


# Slow: five frames of the distorted transmitter through a linear basis.
@pytest.mark.slow
def test_a_linear_basis_stops_at_the_amplifier_distortion(tmp_path):
    """A13, A9, A11: CA 18.810 dB, the floor 22.14 dB up, the target found in at most one frame."""
    verify = {
        # 40.946 - 10 log10(1 + 12 430 x 0.013077) = 18.810 dB; |x|^6 statistics, 0.1 dB.
        "cancellation_depth_db": 18.810,
        "cancellation_tolerance_db": 0.1,
        "floor_rise_db": 22.14,
        "floor_rise_tolerance_db": 1.0,
    }
    scenario = _variant("scenario_005_distorted_transmitter.toml", verify, odd_orders=(1,))
    report = evaluate.run(scenario, n_frames=5, out_dir=tmp_path, render=False)
    # Predicted SINR about 5 dB, Pd <= 4.7e-3 a frame: two hits in five has P <= 2e-4.
    assert sum(report.values[f"target_detections[{k}]"] >= 1 for k in range(5)) <= 1


# Slow: five frames of the moving clutter through one fit over the whole CPI.
@pytest.mark.slow
def test_a_full_cpi_fit_lets_the_moving_clutter_through(tmp_path):
    """A11: one-bin notch, 13 % of a 30 dB echo escapes: floor 21.19 dB up, at most one hit."""
    verify = {"floor_rise_db": 21.19, "floor_rise_tolerance_db": 1.0}
    scenario = _variant("scenario_005_varying_clutter.toml", verify, n_samples_per_batch=0)
    report = evaluate.run(scenario, n_frames=5, out_dir=tmp_path, render=False)
    # Predicted SINR about 6 dB, Pd <= 1.1e-2 a frame: two hits in five has P <= 1.2e-3.
    assert sum(report.values[f"target_detections[{k}]"] >= 1 for k in range(5)) <= 1


# Slow: the sliding fit is 2048 solves a frame, about 5-15 s here.
@pytest.mark.slow
def test_the_sliding_fit_cancels_the_moving_clutter_to_the_noise(tmp_path):
    """P2's current method: 50 us blocks, 6.45 ms windows; floor and SINR as clean, every frame.

    The window follows the 2 Hz phase block by block, so the moving copy keeps 8.7e-6 of its
    power (from the clipped windows at the CPI's ends); the noise loses K over each window.
    """
    verify = {
        # 41.738 + 0.017 - 0.038 = 41.717 dB, as A6's bound with the sliding residual.
        "cancellation_depth_db": 41.717,
        "cancellation_tolerance_db": 0.05,
        "floor_rise_db": 0.02,
        "floor_rise_tolerance_db": 1.0,
        "sinr_loss_db": 0.0,
        "sinr_loss_tolerance_db": 1.0,
        "detections_per_frame": 1,
        "uncancelled_detections_per_frame": 0,
    }
    scenario = _variant(
        "scenario_005_varying_clutter.toml",
        verify,
        n_samples_per_batch=400,
        n_samples_extension=25_600,
    )
    evaluate.run(scenario, n_frames=5, out_dir=tmp_path, render=False)


# Slow: one frame through the batched and the sliding fits.
@pytest.mark.slow
def test_sliding_removes_the_batched_fits_parasitic_lines():
    """A12: at lag 16 and +-156.25 Hz the sliding map is >= 20 dB under the batched one."""
    scenario = load_scenario(REPO / "config" / "scenario_005_varying_clutter.toml")
    frame = next(frames(scenario, n_frames=1))
    code, stream = frame.references[0], frame.cubes[0]
    maps = {}
    for name, settings in (
        ("batched", scenario.cancellation),
        (
            "sliding",
            dataclasses.replace(
                scenario.cancellation, n_samples_per_batch=400, n_samples_extension=25_600
            ),
        ),
    ):
        residual = cancel(code, stream, settings).residual.reshape(stream.shape)
        product = range_doppler_product(residual, scenario.radar, reference=code, n_range_lags=1024)
        maps[name] = np.abs(product.rd_map) ** 2
    zero = maps["batched"].shape[0] // 2
    lines = [zero - 16, zero + 16]  # +-156.25 Hz: 16 bins of 9.765625 Hz
    for row in lines:
        assert 10 * np.log10(maps["batched"][row, 16] / maps["sliding"][row, 16]) >= 20.0
    # The floor away from lag 16: no higher for the sliding fit (same noise, same code).
    far = (slice(None), slice(300, 800))
    assert np.mean(maps["sliding"][far]) <= np.mean(maps["batched"][far])
    assert clutter.cancellation_depth_db(stream, stream) == 0.0
