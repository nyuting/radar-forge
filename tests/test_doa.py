"""Direction of arrival: the estimators against exact algebra, the bound, and the snapshot model.

Fast tests are analytic: noiseless one- and two-source snapshots, the Cramér-Rao bound's closed
form, the rank a coherent pair leaves and forward-backward averaging restores, the conventional
beam's and monopulse's response to an unresolved pair (spec 006 A3, A6, A7, A9, A17-A19). The
slow tests are the spec's Monte Carlo on the snapshot model of section 3.5,
``x_k = g_k A(u) s + n_k`` over a 5 x 5 chip, which A13 proves equal to the chain: A4/A5 at 400
trials, A8/A10/A11 at 200 at F2's geometry, A14 at 200 at F3's point. Each draws its own seeded
trials once; the bounds are the spec's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from radar_forge.array import (
    ArrayGeometry,
    forward_backward_average,
    sample_covariance,
    uniform_linear_array,
)
from radar_forge.doa import (
    DOA_LABELS,
    bartlett_spectrum,
    crb_deterministic,
    esprit,
    estimate_doa,
    is_resolved,
    label_parts,
    monopulse_direction_cosine,
    monopulse_ratio,
    resolution_snr_db,
    root_music,
    steering_derivative_u,
    steering_u,
)

WAVELENGTH_M = 0.030591067142857142
ULA = uniform_linear_array(16, 0.5, WAVELENGTH_M)
RAYLEIGH = 0.125
# The prototype's F2 SNR, where A8-A11's bounds were set, and F2's far end re-derived (3.3).
PROTOTYPE_SNR_DB, FAR_END_SNR_DB = 38.23, 41.24


def _a(u) -> np.ndarray:
    return steering_u(ULA, u, WAVELENGTH_M)


def _derivative(u) -> np.ndarray:
    return steering_derivative_u(ULA, u, WAVELENGTH_M)


def _lambda_ratio(covariance: np.ndarray) -> float:
    values = np.linalg.eigvalsh(covariance)[::-1]
    return float(values[1] / values[0])


def _local_maxima(spectrum: np.ndarray) -> np.ndarray:
    return np.flatnonzero((spectrum > np.roll(spectrum, 1)) & (spectrum >= np.roll(spectrum, -1)))


def test_root_music_and_esprit_find_one_noiseless_source_exactly():
    """A3: one noiseless source at u = 0.2; a root-selection or rotation sign error moves it."""
    r = np.outer(_a(0.2)[:, 0], _a(0.2)[:, 0].conj())
    # Exact algebra on a rank-one covariance: atol 1e-10 in u.
    np.testing.assert_allclose(root_music(r, 1, spacing_wavelengths=0.5), [0.2], atol=1e-10)
    np.testing.assert_allclose(esprit(r, 1, spacing_wavelengths=0.5), [0.2], atol=1e-10)


def test_one_source_bound_is_the_closed_form():
    """A6: 6 sigma^2 / (pi^2 |s|^2 n (n^2 - 1)) at d = lambda/2; catches sigma^2 vs sigma^2/2."""
    bound = crb_deterministic(_a(0.13), _derivative(0.13), [np.sqrt(3.0) * 1j], 2.0)
    expected = 6.0 * 2.0 / (np.pi**2 * 3.0 * 16 * (16**2 - 1))
    np.testing.assert_allclose(bound[0, 0], expected, rtol=1e-10)


def test_a_coherent_pairs_chip_covariance_is_rank_one():
    """A7 on the model: every cell holds g_k (a1 s1 + a2 s2), so lambda_2 / lambda_1 < 1e-12."""
    y = _a([0.0, 0.0625]) @ np.array([1.0, np.exp(0.7j)])
    gains = np.exp(1j * np.arange(25)) * np.linspace(0.1, 1.0, 25)
    assert _lambda_ratio(sample_covariance(np.outer(gains, y))) < 1e-12


@pytest.mark.parametrize(
    ("phase_rad", "rank"), [(0.0, 1), (np.pi, 1), (np.pi / 2, 2)], ids=["0", "pi", "pi/2"]
)
def test_forward_backward_restores_the_rank_except_in_and_anti_phase(phase_rad, rank):
    """A9: J x* = a1 + exp(-j phi) a2 is parallel to x only at phi = 0 or pi."""
    x = _a([-0.03125, 0.03125]) @ np.array([1.0, np.exp(1j * phase_rad)])
    ratio = _lambda_ratio(forward_backward_average(np.outer(x, x.conj()), ULA))
    # Rank one is exact up to float64 eigenvalues; rank two leaves lambda_2 near lambda_1/4 here.
    assert ratio < 1e-12 if rank == 1 else ratio > 1e-2


@pytest.mark.parametrize("separation_rayleigh", [0.5, 0.25])
def test_an_in_phase_pairs_beam_has_one_peak_at_the_midpoint(separation_rayleigh):
    """A17: |a^H (a1 + a2)|^2 is symmetric about the midpoint, its one mainlobe peak there."""
    midpoint, half = 0.07, separation_rayleigh * RAYLEIGH / 2.0
    x = _a([midpoint - half, midpoint + half]).sum(axis=1)
    grid = np.arange(-4000, 4000) * 2.5e-4
    spectrum = bartlett_spectrum(np.outer(x, x.conj()), _a(grid))
    near = [i for i in _local_maxima(spectrum) if abs(grid[i] - midpoint) < RAYLEIGH]
    assert len(near) == 1
    # Within one grid step of 2.5e-4.
    assert abs(grid[near[0]] - midpoint) <= 2.5e-4


@pytest.mark.parametrize("separation_rayleigh", [0.5, 0.25])
def test_a_weaker_incoherent_seconds_beam_peaks_near_the_power_centroid(separation_rayleigh):
    """A18: a -6 dB second source, phase-averaged: one peak, within 0.1 du of the centroid."""
    separation = separation_rayleigh * RAYLEIGH
    a = _a([0.0, separation])
    r = np.outer(a[:, 0], a[:, 0].conj()) + 0.25 * np.outer(a[:, 1], a[:, 1].conj())
    centroid = 0.25 * separation / 1.25
    grid = np.arange(-4000, 4000) * 2.5e-4
    spectrum = bartlett_spectrum(r, _a(grid))
    near = [i for i in _local_maxima(spectrum) if abs(grid[i] - centroid) < RAYLEIGH]
    assert len(near) == 1
    assert abs(grid[near[0]] - centroid) <= 0.1 * separation


def test_monopulse_wanders_per_look_and_points_at_the_centroid_on_average():
    """A19: per look the indication leaves the pair; averaged over phase, within 0.1 du of it."""
    pair = np.array([-0.0112, 0.0513])
    phases = np.arange(360) * np.pi / 180.0
    looks = (_a(pair) @ np.stack([np.ones(360), np.exp(1j * phases)])).T
    per_look = [
        monopulse_direction_cosine(
            monopulse_ratio(look, ULA, wavelength_m=WAVELENGTH_M), ULA, wavelength_m=WAVELENGTH_M
        )
        for look in looks
    ]
    assert min(per_look) < pair[0] or max(per_look) > pair[1]
    averaged = monopulse_direction_cosine(
        monopulse_ratio(looks, ULA, wavelength_m=WAVELENGTH_M), ULA, wavelength_m=WAVELENGTH_M
    )
    assert abs(averaged - pair.mean()) <= 0.1 * (pair[1] - pair[0])
    # In anti-phase the beam's peak lies outside the pair.
    grid = np.arange(-4000, 4000) * 2.5e-4
    spectrum = bartlett_spectrum(np.outer(looks[180], looks[180].conj()), _a(grid))
    peak = grid[np.argmax(spectrum)]
    assert peak < pair[0] or peak > pair[1]


def test_monopulse_reads_one_source_exactly():
    """One source: Delta / Sigma = j tan(pi D (u - u0) / lambda), inverted exactly."""
    ratio = monopulse_ratio(_a(0.031)[:, 0], ULA, wavelength_m=WAVELENGTH_M)
    found = monopulse_direction_cosine(ratio, ULA, wavelength_m=WAVELENGTH_M)
    assert found == pytest.approx(0.031, abs=1e-12)


def _noiseless_chip(phase_rad: float, separation: float = 0.0625):
    """A noiseless 5 x 5 chip of a coherent pair, off-bin, and its peak cell."""
    truths = np.array([-0.06, -0.06 + separation])
    y = _a(truths) @ np.array([1.0, np.exp(1j * phase_rad)])
    gains = _cell_gains(0.3, -0.2)
    return np.outer(gains, y), np.outer(gains, y)[12], truths


@pytest.mark.parametrize(
    "label", ["fbss_music", "fbss_esprit", "fbss_root_music", "deterministic_ml"]
)
def test_rank_restoring_methods_and_ml_recover_a_noiseless_coherent_pair(label):
    """Half a Rayleigh width, phase 1 rad, no noise: FBSS methods and ML are exact."""
    cells, snapshot, truths = _noiseless_chip(1.0)
    method, kind = label_parts(label)
    estimate = estimate_doa(
        snapshot,
        ULA,
        2,
        wavelength_m=WAVELENGTH_M,
        method=method,
        covariance=kind,
        training=None if method == "deterministic_ml" else cells,
    )
    # The searches stop at 1e-9 of a 4.9e-4 grid step; FBSS's subspace is exact: atol 1e-8.
    np.testing.assert_allclose(estimate.direction_cosine_x, truths, atol=1e-8)


def test_raw_music_cannot_resolve_a_noiseless_coherent_pair():
    """Rank one: the second "source" of MUSIC's signal subspace is noise, so it fails."""
    cells, snapshot, truths = _noiseless_chip(1.0)
    estimate = estimate_doa(
        snapshot, ULA, 2, wavelength_m=WAVELENGTH_M, method="music", training=cells
    )
    assert not is_resolved(estimate.direction_cosine_x, truths, 0.0625)
    assert estimate.eigenvalues[1] / estimate.eigenvalues[0] < 1e-12


def test_iaa_spans_the_visible_region_and_resolves_a_wide_noiseless_pair():
    """IAA's grid is u in [-1, 1); two Rayleigh widths apart, both sources are its peaks."""
    _, snapshot, truths = _noiseless_chip(2.0, separation=0.25)
    estimate = estimate_doa(snapshot, ULA, 2, wavelength_m=WAVELENGTH_M, method="iaa", n_grid=4096)
    assert estimate.grid[0] == -1.0 and estimate.grid[-1] < 1.0 and estimate.spectrum.size == 4096
    # IAA is a grid fit, biased by its neighbours: within a twentieth of the separation.
    np.testing.assert_allclose(estimate.direction_cosine_x, truths, atol=0.25 / 20)


@pytest.mark.parametrize(
    ("separation_rayleigh", "phase_rad", "snr_db"),
    [(1.0, 0.0, -18.3), (1.0, np.pi, -6.5), (0.5, np.pi / 2, -3.5), (0.1, np.pi, 53.4)],
)
def test_smiths_resolution_limit_reproduces_the_spec_table(separation_rayleigh, phase_rad, snr_db):
    """Section 3.4 re-derived: the SNR where sqrt(CRB(delta)) = delta, to its 0.1 dB rounding."""
    found = resolution_snr_db(ULA, separation_rayleigh, phase_rad, wavelength_m=WAVELENGTH_M)
    assert found == pytest.approx(snr_db, abs=0.05)


def test_a_flat_spectrum_has_no_peak_and_says_so():
    """No signal at all: the beam is flat, and there is nothing to locate."""
    with pytest.raises(ValueError, match="no peak"):
        estimate_doa(np.zeros(16), ULA, 1, wavelength_m=WAVELENGTH_M, method="bartlett")


def test_loaded_capon_finds_one_source_in_one_snapshot():
    """One snapshot's covariance is rank one; loaded, Capon's peak is the source's direction."""
    x = 3.0 * _a(0.3)[:, 0]
    estimate = estimate_doa(
        x, ULA, 1, wavelength_m=WAVELENGTH_M, method="capon", diagonal_loading_w=0.01
    )
    # x x^H + d I: the spectrum is maximal where |a^H x| is, exactly; the search stops at 1e-9
    # of a grid step.
    np.testing.assert_allclose(estimate.direction_cosine_x, [0.3], atol=1e-9)


def test_estimate_doa_refuses_what_a_method_cannot_take():
    """A snapshot method takes no training cells; root-MUSIC needs a uniform linear array."""
    snapshot = _a(0.1)[:, 0]
    with pytest.raises(ValueError, match="snapshot alone"):
        estimate_doa(snapshot, ULA, 1, wavelength_m=WAVELENGTH_M, method="iaa", covariance="fbss")
    uneven = ArrayGeometry(ULA.element_positions_m * (1.0 + 0.01 * np.arange(16))[:, None])
    with pytest.raises(ValueError, match="uniform linear"):
        estimate_doa(snapshot, uneven, 1, wavelength_m=WAVELENGTH_M, method="root_music")
    with pytest.raises(ValueError, match="unknown DOA method"):
        label_parts("fbss_iaa")
    assert label_parts("fbss_music") == ("music", "fbss") and "fb_esprit" in DOA_LABELS


def test_music_spectrum_matches_pyroomacoustics_on_an_incoherent_pair():
    """A15: pyroomacoustics' MUSIC, its azimuth phi counter-clockwise from its x axis.

    The array lies on pyroomacoustics' x axis, so a source at D13's direction cosine u is at
    phi = arccos(u) in its frame. The two spectra peak within one grid step of each other.
    """
    pra = pytest.importorskip("pyroomacoustics")
    truths = np.array([-0.2, 0.15])
    a = _a(truths)
    frequency_hz, sound_mps = 1000.0, 343.0
    # The same electrical spacing: half a wavelength of sound at 1 kHz.
    positions_m = np.zeros((2, 16))
    positions_m[0] = ULA.element_positions_m[:, 0] / WAVELENGTH_M * sound_mps / frequency_hz
    grid_u = np.linspace(-0.99, 0.99, 3961)
    azimuth = np.arccos(grid_u)
    doa = pra.doa.algorithms["MUSIC"](
        positions_m, 16000, 256, c=sound_mps, num_src=2, azimuth=azimuth
    )
    rng = np.random.default_rng(15)
    n = 2000
    sources = (rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))) / np.sqrt(2)
    noise = 0.1 * (rng.normal(size=(16, n)) + 1j * rng.normal(size=(16, n))) / np.sqrt(2)
    snapshots = a @ sources + noise
    spectra = np.zeros((16, 129, n), dtype=complex)
    spectra[:, 16, :] = snapshots
    doa.locate_sources(spectra, freq_bins=[16])
    theirs = np.sort(np.cos(np.asarray(doa.azimuth_recon)))
    ours = estimate_doa(
        snapshots[:, 0],
        ULA,
        2,
        wavelength_m=WAVELENGTH_M,
        method="music",
        training=snapshots.T,
    ).direction_cosine_x
    np.testing.assert_allclose(theirs, ours, atol=2 * (grid_u[1] - grid_u[0]))
    np.testing.assert_allclose(ours, truths, atol=1e-2)


# The snapshot model of spec 006 section 3.5: a pair at fractional range and Doppler offsets.
def _dirichlet(x: np.ndarray, n: int) -> np.ndarray:
    """Return a rectangular window's cell response at offset x bins, unit at zero."""
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.sin(np.pi * x) / (n * np.sin(np.pi * x / n))
    value = np.where(np.abs(x) < 1e-12, 1.0, value)
    return value * np.exp(1j * np.pi * x * (n - 1) / n)


def _cell_gains(range_offset: float, doppler_offset: float) -> np.ndarray:
    """Return g_k over a 5 x 5 chip, rows Doppler, for S1's 1000 range and 256 Doppler bins."""
    k = np.arange(-2, 3)
    return (
        _dirichlet(k[:, None] - doppler_offset, 256) * _dirichlet(k[None, :] - range_offset, 1000)
    ).ravel()


def _trials(snr_db: float, separation_rayleigh: float, n_trials: int, seed: int):
    """Yield (cells, peak snapshot, truths, peak amplitudes) of seeded snapshot-model trials.

    SNR is per channel on an exact bin, the link budget's convention; noise is unit per cell.
    """
    rng = np.random.default_rng(seed)
    separation = separation_rayleigh * RAYLEIGH
    # A loop over trials: each draws its own phases, offsets and noise.
    for _ in range(n_trials):
        u_a = rng.uniform(-0.06, 0.05)
        truths = np.array([u_a, u_a + separation])
        amplitude = math.sqrt(10.0 ** (snr_db / 10.0)) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        s = amplitude * np.array([1.0, np.exp(1j * rng.uniform(0, 2 * np.pi))])
        gains = _cell_gains(*rng.uniform(-0.5, 0.5, 2))
        noise = (rng.normal(size=(25, 16)) + 1j * rng.normal(size=(25, 16))) / math.sqrt(2.0)
        cells = np.outer(gains, _a(truths) @ s) + noise
        peak = int(np.argmax(np.abs(gains)))
        yield cells, cells[peak], truths, gains[peak] * s


def _score(labels, snr_db, separation_rayleigh, n_trials, seed):
    """Return each label's P(resolved) and RMSE over the bound at the truth, on the same trials."""
    resolved = {label: [] for label in labels}
    scaled = {label: [] for label in labels}
    for cells, snapshot, truths, s in _trials(snr_db, separation_rayleigh, n_trials, seed):
        a = _a(truths)
        bound = np.sqrt(np.diag(crb_deterministic(a, _derivative(truths), s, 1.0)))
        for label in labels:
            method, kind = label_parts(label)
            trained = method in ("capon", "music", "root_music", "esprit")
            found = estimate_doa(
                snapshot,
                ULA,
                2,
                wavelength_m=WAVELENGTH_M,
                method=method,
                covariance=kind,
                training=cells if trained else None,
            ).direction_cosine_x
            resolved[label].append(is_resolved(found, truths, separation_rayleigh * RAYLEIGH))
            scaled[label] += list(((found - truths) / bound) ** 2)
    return (
        {label: float(np.mean(v)) for label, v in resolved.items()},
        {label: math.sqrt(float(np.mean(v))) for label, v in scaled.items()},
    )


@pytest.mark.slow
def test_one_source_subspace_estimates_are_unbiased_and_near_the_bound():
    """A4, A5: 400 trials at 20 dB, one snapshot; |bias| <= 0.25 sqrt(CRB), RMSE as the spec."""
    rng = np.random.default_rng(4)
    snr = 100.0
    sigma = math.sqrt(6.0 / (np.pi**2 * snr * 16 * (16**2 - 1)))
    errors = {"music": [], "root_music": [], "esprit": []}
    # A loop over trials, three estimators each.
    for _ in range(400):
        phase = np.exp(1j * rng.uniform(0, 2 * np.pi))
        noise = (rng.normal(size=16) + 1j * rng.normal(size=16)) / math.sqrt(2.0)
        x = math.sqrt(snr) * phase * _a(0.2)[:, 0] + noise
        for method in errors:
            found = estimate_doa(x, ULA, 1, wavelength_m=WAVELENGTH_M, method=method)
            errors[method].append(found.direction_cosine_x[0] - 0.2)
    # The mean's standard error is 0.05 sqrt(CRB) at n = 400: 0.25 is five of them.
    for method, error in errors.items():
        assert abs(np.mean(error)) <= 0.25 * sigma, method
    rmse = {m: math.sqrt(np.mean(np.square(e))) / sigma for m, e in errors.items()}
    assert rmse["music"] <= 1.2 and rmse["root_music"] <= 1.2, rmse
    assert rmse["esprit"] <= 2.0, rmse


@pytest.mark.slow
@pytest.mark.parametrize("snr_db", [PROTOTYPE_SNR_DB, FAR_END_SNR_DB], ids=["38.23dB", "41.24dB"])
def test_at_f2_the_baseline_fails_and_the_current_methods_resolve_near_the_bound(snr_db):
    """A8, A10, A11: 200 trials, half a Rayleigh width, at the prototype's and far end's SNR."""
    labels = ["music", "esprit", "fbss_music", "fbss_esprit", "iaa", "deterministic_ml"]
    fraction, rmse = _score(labels, snr_db, 0.5, 200, seed=8)
    # A8: Clopper-Pearson 99.9 % upper limit at the prototype's 17 / 200 is 0.163.
    assert fraction["music"] <= 0.25 and fraction["esprit"] <= 0.25, fraction
    # A10: the prototype's 1.000 for all four.
    assert all(fraction[m] >= 0.95 for m in labels[2:]), fraction
    # A11: ML <= 1.3 fails a bound off by sqrt(2); the others as the spec gives.
    assert rmse["deterministic_ml"] <= 1.3, rmse
    assert rmse["fbss_music"] <= 1.5 and rmse["fbss_esprit"] <= 1.5, rmse
    assert rmse["iaa"] <= 4.0, rmse


@pytest.mark.slow
def test_at_f3s_point_ml_fbss_esprit_and_iaa_resolve_as_the_spec_bounds():
    """A14: half a Rayleigh width at 20 dB, 200 trials; 99.9 % Clopper-Pearson-backed bounds."""
    fraction, _ = _score(["fbss_esprit", "iaa", "deterministic_ml"], 20.0, 0.5, 200, seed=14)
    assert fraction["deterministic_ml"] >= 0.90, fraction
    assert fraction["fbss_esprit"] >= 0.85, fraction
    assert fraction["iaa"] >= 0.65, fraction
