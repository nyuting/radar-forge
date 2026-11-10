r"""Direction of arrival on a linear array: spectra, subspace and sparse estimators, the bound.

Every estimator returns direction cosines :math:`u` along the array's x axis
(:mod:`radar_forge.array`). :func:`estimate_doa` runs one, by name, on the snapshot of one
range-Doppler cell, or on a covariance it forms from training snapshots, raw, forward-backward or
smoothed. Grid methods search :math:`u \in [-1, 1)`, every direction once at half-wavelength
spacing, and refine each peak to the continuous spectrum's maximum between its grid neighbours.
The chain of methods, each from the one before: the conventional beam scans the matched filter;
Capon minimises the power each look lets through; MUSIC keeps only the noise subspace's
orthogonality; spatial smoothing restores the rank a coherent pair takes from the covariance; IAA
refits Capon's weights to one snapshot; deterministic ML fits the two-source model exactly. Their
error is scored against :func:`crb_deterministic`, and :func:`resolution_snr_db` says where the
bound itself forbids resolving a pair.

References
----------
.. [1] R. O. Schmidt, "Multiple emitter location and signal parameter estimation," *IEEE Trans.
       Antennas Propag.*, vol. 34, no. 3, pp. 276-280, 1986.
.. [2] A. J. Barabell, "Improving the resolution performance of eigenstructure-based
       direction-finding algorithms," *Proc. IEEE ICASSP*, vol. 8, pp. 336-339, 1983.
.. [3] R. Roy and T. Kailath, "ESPRIT: estimation of signal parameters via rotational
       invariance techniques," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 37, no. 7,
       pp. 984-995, 1989.
.. [4] T. Yardibi, J. Li, P. Stoica, M. Xue and A. B. Baggeroer, "Source localization and
       sensing: a nonparametric iterative adaptive approach based on weighted least squares,"
       *IEEE Trans. Aerosp. Electron. Syst.*, vol. 46, no. 1, pp. 425-443, 2010.
.. [5] P. Stoica, P. Babu and J. Li, "SPICE: a sparse covariance-based estimation method for
       array processing," *IEEE Trans. Signal Process.*, vol. 59, no. 2, pp. 629-638, 2011.
.. [6] J. Li and P. Stoica, "Efficient mixed-spectrum estimation with applications to target
       feature extraction," *IEEE Trans. Signal Process.*, vol. 44, no. 2, pp. 281-295, 1996.
.. [7] P. Stoica and A. Nehorai, "MUSIC, maximum likelihood, and Cramer-Rao bound," *IEEE
       Trans. Acoust., Speech, Signal Process.*, vol. 37, no. 5, pp. 720-741, 1989.
.. [8] S. T. Smith, "Statistical resolution limits and the complexified Cramer-Rao bound,"
       *IEEE Trans. Signal Process.*, vol. 53, no. 5, pp. 1597-1609, 2005.
.. [9] J. Capon, "High-resolution frequency-wavenumber spectrum analysis," *Proc. IEEE*,
       vol. 57, no. 8, pp. 1408-1418, 1969.
.. [10] S. M. Sherman and D. K. Barton, *Monopulse Principles and Techniques*, 2nd ed., Artech
        House, 2011, ch. 9 (response to unresolved targets).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize, minimize_scalar

from radar_forge.array import (
    ArrayGeometry,
    beam_power,
    conventional_weights,
    forward_backward_average,
    rayleigh_width,
    sample_covariance,
    spatial_smoothing,
    subarray,
)

__all__ = [
    "COVARIANCE_KINDS",
    "DOA_LABELS",
    "DOA_METHODS",
    "CovarianceKind",
    "DoaEstimate",
    "DoaMethod",
    "bartlett_spectrum",
    "capon_spectrum",
    "crb_deterministic",
    "deterministic_ml",
    "esprit",
    "estimate_doa",
    "iaa_spectrum",
    "is_resolved",
    "label_parts",
    "monopulse_direction_cosine",
    "monopulse_ratio",
    "music_spectrum",
    "resolution_snr_db",
    "root_music",
    "spice_spectrum",
    "steering_derivative_u",
    "steering_u",
]

DoaMethod = Literal[
    "bartlett", "capon", "music", "root_music", "esprit", "iaa", "spice", "deterministic_ml"
]
"""The estimators: two beamformers, three subspace methods, two sparse ones, and ML."""
DOA_METHODS: tuple[DoaMethod, ...] = get_args(DoaMethod)

CovarianceKind = Literal["raw", "forward_backward", "fbss"]
"""The covariance a covariance method runs on: as sampled, forward-backward, or both smoothed."""
COVARIANCE_KINDS: tuple[CovarianceKind, ...] = get_args(CovarianceKind)

# The methods that take the one snapshot and no covariance.
_SNAPSHOT_METHODS = ("iaa", "spice", "deterministic_ml")
_PREFIXES: dict[str, CovarianceKind] = {"fb_": "forward_backward", "fbss_": "fbss"}
DOA_LABELS: tuple[str, ...] = tuple(
    [*DOA_METHODS] + [p + m for p in _PREFIXES for m in DOA_METHODS if m not in _SNAPSHOT_METHODS]
)
"""Names for a method on a covariance: the method, prefixed ``fb_`` or ``fbss_`` unless raw."""

_ITERATIONS = {"iaa": 15, "spice": 60}
# Alternating-projection sweeps of deterministic ML before its joint refinement.
_ML_SWEEPS = 20
# A refined peak is found to this fraction of a grid step.
_REFINE_TOLERANCE = 1e-9
# Mirrored roots closer than this are one (z, 1/z*) pair, read twice: a degree 2(n - 1)
# polynomial's roots are found to about 1e-7.
_PAIR_TOLERANCE = 1e-6
# Estimates whose steering vectors' Gram matrix is worse conditioned than this coincide: the
# bound at them is undefined, nan.
_MAX_CONDITION = 1e10


@dataclass(frozen=True, eq=False)
class DoaEstimate:
    r"""One estimator's directions and what it computed on the way.

    ``direction_cosine_x`` is sorted. ``amplitude`` is the least-squares fit of the snapshot at
    those directions, :math:`A^\dagger x`; ``crb_std_direction_cosine`` the bound's standard
    deviation there, nan without a noise power. ``covariance`` and its ``eigenvalues``
    (descending) are what the method ran on, the snapshot's outer product for a snapshot method;
    ``grid`` and ``spectrum`` are empty for root-MUSIC, ESPRIT and ML.
    """

    method: DoaMethod
    covariance_kind: CovarianceKind
    direction_cosine_x: NDArray[np.float64]
    amplitude: NDArray[np.complex128]
    crb_std_direction_cosine: NDArray[np.float64]
    covariance: NDArray[np.complex128]
    eigenvalues: NDArray[np.float64]
    grid: NDArray[np.float64]
    spectrum: NDArray[np.float64]


def label_parts(label: str) -> tuple[DoaMethod, CovarianceKind]:
    """Split a :data:`DOA_LABELS` name into its method and covariance: ``fbss_music``."""
    if label not in DOA_LABELS:
        msg = f"unknown DOA method {label!r}; known: {list(DOA_LABELS)}."
        raise ValueError(msg)
    for prefix, kind in _PREFIXES.items():
        if label.startswith(prefix):
            return _method(label.removeprefix(prefix)), kind
    return _method(label), "raw"


def _method(name: str) -> DoaMethod:
    """Return ``name`` as a :data:`DoaMethod`."""
    return next(m for m in DOA_METHODS if m == name)


def steering_u(
    geometry: ArrayGeometry, direction_cosine_x: ArrayLike, wavelength_m: float
) -> NDArray[np.complex128]:
    r"""Return a linear array's steering vectors at direction cosines, ``(n_rx, n)``.

    :math:`e^{+j2\pi x_n u/\lambda}`: :func:`~radar_forge.array.steering_vectors`' sign, for an
    array along x, which sees only :math:`u`.
    """
    u = np.atleast_1d(np.asarray(direction_cosine_x, dtype=np.float64))
    phase_rad = 2.0 * np.pi * np.outer(geometry.element_positions_m[:, 0], u) / wavelength_m
    result: NDArray[np.complex128] = np.exp(1j * phase_rad)
    return result


def bartlett_spectrum(covariance: ArrayLike, steering: ArrayLike) -> NDArray[np.float64]:
    r"""Return the conventional beam's power :math:`a^H R a / (a^H a)^2` at each look, W.

    The matched filter scanned over direction: a look's output power through
    :func:`~radar_forge.array.conventional_weights`. Its width is the Rayleigh width, and a pair
    inside it is one blob, centred near the power centroid (Sherman and Barton [10] ch. 9).
    """
    return beam_power(conventional_weights(steering), covariance)


def capon_spectrum(
    covariance: ArrayLike, steering: ArrayLike, *, diagonal_loading_w: float = 0.0
) -> NDArray[np.float64]:
    r"""Return Capon's power :math:`1/(a^H R^{-1} a)` at each look, W.

    Bartlett with adaptive weights: each look passes its own direction undistorted and
    minimises the power from every other (Capon [9] eq. (18)). A coherent second source is not
    rejected but cancelled, since it is correlated with the look's own signal.
    """
    r = np.asarray(covariance, dtype=np.complex128)
    a = np.asarray(steering, dtype=np.complex128)
    whitened = np.linalg.solve(r + diagonal_loading_w * np.eye(r.shape[0]), a)
    result: NDArray[np.float64] = 1.0 / np.real(np.sum(a.conj() * whitened, axis=0))
    return result


def _noise_subspace(covariance: NDArray[np.complex128], n_sources: int) -> NDArray[np.complex128]:
    """Return the eigenvectors of all but the ``n_sources`` largest eigenvalues, as columns."""
    _, vectors = np.linalg.eigh(covariance)
    result: NDArray[np.complex128] = vectors[:, : covariance.shape[0] - n_sources]
    return result


def _signal_subspace(covariance: NDArray[np.complex128], n_sources: int) -> NDArray[np.complex128]:
    """Return the eigenvectors of the ``n_sources`` largest eigenvalues, as columns."""
    _, vectors = np.linalg.eigh(covariance)
    result: NDArray[np.complex128] = vectors[:, covariance.shape[0] - n_sources :]
    return result


def music_spectrum(
    covariance: ArrayLike, steering: ArrayLike, n_sources: int
) -> NDArray[np.float64]:
    r"""Return MUSIC's pseudo-spectrum :math:`a^H a / (a^H E_n E_n^H a)`, dimensionless.

    Capon's :math:`R^{-1}` with the signal eigenvalues dropped: a source's steering vector is
    orthogonal to the noise subspace :math:`E_n`, the eigenvectors of all but the
    ``n_sources`` largest eigenvalues, so the spectrum peaks there without bound (Schmidt [1];
    the MUSIC function, Stoica and Nehorai [7] eq. (3.7a)). A coherent pair's signal subspace
    is one-dimensional, so the second "source" is noise, and MUSIC fails.
    """
    a = np.asarray(steering, dtype=np.complex128)
    noise = _noise_subspace(np.asarray(covariance, dtype=np.complex128), n_sources)
    leakage = np.sum(np.abs(noise.conj().T @ a) ** 2, axis=0)
    result: NDArray[np.float64] = np.sum(np.abs(a) ** 2, axis=0) / leakage
    return result


def root_music(
    covariance: ArrayLike, n_sources: int, *, spacing_wavelengths: float
) -> NDArray[np.float64]:
    r"""Return MUSIC's directions as roots: :math:`u = \arg z / (2\pi d/\lambda)`, sorted.

    On a ULA :math:`a_n = z^n`, :math:`z = e^{j2\pi d u/\lambda}`, so :math:`a^H E_n E_n^H a` is
    the polynomial :math:`\sum_k c_k z^k`, :math:`c_k` the sum of :math:`E_n E_n^H`'s
    :math:`k`-th diagonal. Its roots pair as :math:`z, 1/z^*`; the ``n_sources`` inside the unit
    circle nearest it are the sources, off the grid (Barabell [2]).
    """
    r = np.asarray(covariance, dtype=np.complex128)
    noise = _noise_subspace(r, n_sources)
    projector = noise @ noise.conj().T
    n = r.shape[0]
    # Highest power first: the coefficient of z^(k + n - 1) is the sum of diagonal k.
    coefficients = [np.trace(projector, offset=k) for k in range(n - 1, -n, -1)]
    roots = np.roots(coefficients)
    # Round-off can put both of a pair a hair outside the circle, so mirror every root inside
    # and take one of each pair, nearest the circle first.
    mirrored = np.where(np.abs(roots) <= 1.0, roots, 1.0 / np.conj(roots))
    chosen: list[complex] = []
    # A loop: each pick skips the twins of the picks before it.
    for z in mirrored[np.argsort(-np.abs(mirrored))]:
        if all(abs(z - c) > _PAIR_TOLERANCE for c in chosen):
            chosen.append(complex(z))
        if len(chosen) == n_sources:
            break
    return np.sort(np.angle(np.array(chosen)) / (2.0 * np.pi * spacing_wavelengths))


def esprit(
    covariance: ArrayLike, n_sources: int, *, spacing_wavelengths: float
) -> NDArray[np.float64]:
    r"""Return least-squares ESPRIT's directions, sorted.

    The array less its last element and less its first are shifts of one another, so their
    rows of the signal subspace satisfy :math:`E_2 = E_1\Psi`, and :math:`\Psi`'s eigenvalues
    are :math:`e^{j2\pi d u_k/\lambda}`: no search at all (Roy and Kailath [3]). :math:`\Psi`
    is fitted by least squares, which is inefficient beside total least squares.
    """
    signal = _signal_subspace(np.asarray(covariance, dtype=np.complex128), n_sources)
    rotation, *_ = np.linalg.lstsq(signal[:-1], signal[1:], rcond=None)
    phases = np.angle(np.linalg.eigvals(rotation))
    return np.sort(phases / (2.0 * np.pi * spacing_wavelengths))


def _iaa(
    snapshot: NDArray[np.complex128], steering: NDArray[np.complex128], n_iterations: int
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Return IAA's powers on the grid and the covariance they model."""
    n_rx = steering.shape[0]
    power = np.abs(steering.conj().T @ snapshot) ** 2 / n_rx**2
    # Each iteration refits every power to the covariance the last ones model: sequential.
    for _ in range(n_iterations):
        power = _weighted_power((steering * power) @ steering.conj().T, snapshot, steering)
    return power, (steering * power) @ steering.conj().T


def _weighted_power(
    model: NDArray[np.complex128],
    snapshot: NDArray[np.complex128],
    steering: NDArray[np.complex128],
) -> NDArray[np.float64]:
    r"""Return :math:`|a^H R^{-1}x|^2/(a^H R^{-1}a)^2`, the power Capon's weights pass."""
    whitened = np.linalg.solve(model, steering)
    result: NDArray[np.float64] = (
        np.abs(whitened.conj().T @ snapshot) ** 2
        / np.real(np.sum(steering.conj() * whitened, axis=0)) ** 2
    )
    return result


def iaa_spectrum(
    snapshot: ArrayLike, steering: ArrayLike, *, n_iterations: int = 15
) -> NDArray[np.float64]:
    r"""Return IAA's power at each grid direction from one snapshot, W.

    Capon's weights need a covariance; one snapshot has none of rank above one. IAA models it,
    :math:`R = \sum_k p_k a_k a_k^H` over the grid, starting from the beam's
    :math:`|a_k^H x|^2/n_{rx}^2`, and refits each power as the squared amplitude Capon's weights
    pass, :math:`p_k = |a_k^H R^{-1}x|^2/(a_k^H R^{-1}a_k)^2`, ``n_iterations`` times (Yardibi
    et al. [4]). The grid must span the visible region, ``u`` in [-1, 1): on a sector, :math:`R`
    misses the rest of the space and is ill-conditioned (about :math:`10^{17}`), and IAA diverges.
    """
    power, _ = _iaa(
        np.asarray(snapshot, dtype=np.complex128),
        np.asarray(steering, dtype=np.complex128),
        n_iterations,
    )
    return power


def spice_spectrum(
    snapshot: ArrayLike, steering: ArrayLike, *, n_iterations: int = 60
) -> NDArray[np.float64]:
    r"""Return SPICE's power at each grid direction from one snapshot, W.

    A covariance fit with no parameter to tune: :math:`R = B\,\mathrm{diag}(p)\,B^H`,
    :math:`B = [A\ I]`, a noise power per channel beside each grid power, from
    :math:`p_k = |b_k^H x|^2/\|b_k\|^4`; each iteration sets
    :math:`p_k \leftarrow p_k |b_k^H R^{-1}x| / (w_k^{1/2}\rho)`, :math:`w_k = \|b_k\|^2/\|x\|^2`,
    :math:`\rho = \sum_j w_j^{1/2} p_j |b_j^H R^{-1}x|` (Stoica, Babu and Li [5] eqs. (33)-(34)
    with Remark 2, one snapshot). Sparser than IAA and biased at high SNR.
    """
    x = np.asarray(snapshot, dtype=np.complex128)
    a = np.asarray(steering, dtype=np.complex128)
    columns = np.concatenate([a, np.eye(a.shape[0])], axis=1)
    norm_squared = np.sum(np.abs(columns) ** 2, axis=0)
    weight_root = np.sqrt(norm_squared / np.sum(np.abs(x) ** 2))
    power = np.abs(columns.conj().T @ x) ** 2 / norm_squared**2
    # Each iteration rescales every power by the fit the last ones give: sequential.
    for _ in range(n_iterations):
        model = (columns * power) @ columns.conj().T
        fit = np.abs(columns.conj().T @ np.linalg.solve(model, x))
        power = power * fit / (weight_root * np.sum(weight_root * power * fit))
    result: NDArray[np.float64] = power[: a.shape[1]]
    return result


def _likelihood(snapshot: NDArray[np.complex128], steering: NDArray[np.complex128]) -> float:
    r"""Return :math:`x^H P_A x`, the snapshot's power in the span of ``steering``'s columns."""
    fit, *_ = np.linalg.lstsq(steering, snapshot, rcond=None)
    return float(np.real(np.vdot(steering @ fit, snapshot)))


def deterministic_ml(
    snapshot: ArrayLike,
    geometry: ArrayGeometry,
    n_sources: int,
    *,
    wavelength_m: float,
    grid: ArrayLike,
) -> NDArray[np.float64]:
    r"""Return the deterministic maximum-likelihood directions of one snapshot, sorted.

    The model fitted exactly: :math:`\hat u = \arg\max x^H P_A(u) x`, the amplitudes
    concentrated out (Stoica and Nehorai [7] eq. (5.4), one snapshot). Each source in turn is
    moved to the ``grid`` point that, with the others fixed and every amplitude refitted, adds
    the most power to the fit (RELAX's cyclic step, Li and Stoica [6]); the first sweep starts
    from none, the rest repeat until no source moves, and Nelder-Mead then refines all jointly.
    """
    x = np.asarray(snapshot, dtype=np.complex128)
    u_grid = np.asarray(grid, dtype=np.float64)
    a_grid = steering_u(geometry, u_grid, wavelength_m)
    chosen: list[int] = []
    # Coordinate ascent: each source is placed given the others, sweep after sweep.
    for sweep in range(_ML_SWEEPS):
        previous = list(chosen)
        for k in range(n_sources):
            others = [i for j, i in enumerate(chosen) if j != k]
            best = int(np.argmax(_added_power(x, a_grid, a_grid[:, others])))
            chosen = [*chosen[:k], best, *chosen[k + 1 :]] if sweep else [*chosen, best]
        if chosen == previous:
            break

    def negative_fit(u: NDArray[np.float64]) -> float:
        return -_likelihood(x, steering_u(geometry, u, wavelength_m))

    step = float(u_grid[1] - u_grid[0])
    start = u_grid[chosen]
    simplex = start + np.vstack([np.zeros(n_sources), step * np.eye(n_sources)])
    found = minimize(
        negative_fit,
        start,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": step * _REFINE_TOLERANCE, "fatol": 0.0},
    )
    return np.sort(np.asarray(found.x, dtype=np.float64))


def _added_power(
    snapshot: NDArray[np.complex128],
    candidates: NDArray[np.complex128],
    others: NDArray[np.complex128],
) -> NDArray[np.float64]:
    r"""Return the power each candidate column adds to the fit of ``others``.

    :math:`|b^H x|^2/\|b\|^2`, :math:`b = P^\perp a` the candidate less its projection on the
    others; zero where :math:`b` vanishes, at the others' own directions.
    """
    if others.shape[1]:
        fit, *_ = np.linalg.lstsq(others, candidates, rcond=None)
        candidates = candidates - others @ fit
    norm_squared = np.sum(np.abs(candidates) ** 2, axis=0)
    gain = np.abs(candidates.conj().T @ snapshot) ** 2
    floor = 1e-9 * snapshot.size
    result: NDArray[np.float64] = np.where(
        norm_squared > floor, gain / np.maximum(norm_squared, floor), 0.0
    )
    return result


def crb_deterministic(
    steering: ArrayLike, derivative: ArrayLike, signal: ArrayLike, noise_power_w: float
) -> NDArray[np.float64]:
    r"""Return the deterministic Cramér-Rao bound on :math:`u` from one snapshot, ``(K, K)``.

    :math:`\frac{\sigma^2}{2}\{\mathrm{Re}[(D^H P_A^\perp D)\odot(s s^H)^T]\}^{-1}`,
    :math:`D = \partial A/\partial u` (Stoica and Nehorai [7] eq. (4.1), exact for one
    snapshot): the amplitudes are unknown, so only the part of each derivative outside the
    span of the steering vectors informs. One source at half-wavelength spacing gives
    :math:`6\sigma^2/(\pi^2|s|^2 n(n^2-1))`.
    """
    a = np.asarray(steering, dtype=np.complex128)
    d = np.asarray(derivative, dtype=np.complex128)
    s = np.asarray(signal, dtype=np.complex128)
    projection = a @ np.linalg.solve(a.conj().T @ a, a.conj().T)
    outside = d.conj().T @ (np.eye(a.shape[0]) - projection) @ d
    fisher = np.real(outside * np.outer(s, s.conj()).T)
    result = np.asarray(noise_power_w / 2.0 * np.linalg.inv(fisher), dtype=np.float64)
    return result


def steering_derivative_u(
    geometry: ArrayGeometry, direction_cosine_x: ArrayLike, wavelength_m: float
) -> NDArray[np.complex128]:
    r"""Return the steering vectors' derivative in u, ``(n_rx, n)``: the bound's :math:`D`.

    :math:`\partial a_n/\partial u = j2\pi x_n/\lambda\; a_n(u)`.
    """
    x_m = geometry.element_positions_m[:, 0]
    steering = steering_u(geometry, direction_cosine_x, wavelength_m)
    result: NDArray[np.complex128] = 2j * np.pi * x_m[:, None] / wavelength_m * steering
    return result


def resolution_snr_db(
    geometry: ArrayGeometry,
    separation_rayleigh: float,
    relative_phase_rad: float,
    *,
    wavelength_m: float,
) -> float:
    r"""Return the per-channel SNR at which a pair's separation equals its own bound's deviation.

    Smith's statistical resolution limit [8]: two equal sources :math:`\delta` apart, symmetric
    about broadside, with relative phase :math:`\phi`, are resolvable once
    :math:`\sqrt{\mathrm{CRB}(\delta)} \le \delta`, :math:`\mathrm{CRB}(\delta) = [1\ {-1}]\,
    \mathrm{CRB}(u)\,[1\ {-1}]^T` from :func:`crb_deterministic`, which scales as 1/SNR.
    """
    separation = separation_rayleigh * rayleigh_width(geometry, wavelength_m)
    u = np.array([-separation / 2.0, separation / 2.0])
    a = steering_u(geometry, u, wavelength_m)
    signal = np.array([1.0, np.exp(1j * relative_phase_rad)])
    bound = crb_deterministic(a, steering_derivative_u(geometry, u, wavelength_m), signal, 1.0)
    difference = np.array([1.0, -1.0])
    return 10.0 * math.log10(float(difference @ bound @ difference) / separation**2)


def _halves(geometry: ArrayGeometry) -> tuple[NDArray[np.bool_], float]:
    """Return which elements form the left half, and the halves' phase-centre spacing, m."""
    x_m = geometry.element_positions_m[:, 0]
    left = np.zeros(geometry.n_rx, dtype=bool)
    left[np.argsort(x_m)[: geometry.n_rx // 2]] = True
    return left, float(x_m[~left].mean() - x_m[left].mean())


def monopulse_ratio(
    snapshots: ArrayLike,
    geometry: ArrayGeometry,
    *,
    wavelength_m: float,
    steering_direction_cosine: float = 0.0,
) -> complex:
    r"""Return half-array phase monopulse's :math:`\Delta/\Sigma` over looks ``(n_looks, n_rx)``.

    Each half is beamed to :math:`u_0`; :math:`\Sigma = b_L + b_R`, :math:`\Delta = b_R - b_L`.
    For one source, :math:`\Delta/\Sigma = j\tan(\pi D(u - u_0)/\lambda)`, :math:`D` the halves'
    spacing. Over several looks it returns :math:`\sum\Delta\Sigma^*/\sum|\Sigma|^2`, the
    phase-averaged ratio, which points at an unresolved pair's power centroid while each look
    wanders with their relative phase (Sherman and Barton [10] ch. 9).
    """
    x = np.atleast_2d(np.asarray(snapshots, dtype=np.complex128))
    left, _ = _halves(geometry)
    weights = np.conj(steering_u(geometry, steering_direction_cosine, wavelength_m)[:, 0])
    beamed_left = x[:, left] @ weights[left]
    beamed_right = x[:, ~left] @ weights[~left]
    total = beamed_left + beamed_right
    return complex(np.sum((beamed_right - beamed_left) * total.conj()) / np.sum(np.abs(total) ** 2))


def monopulse_direction_cosine(
    ratio_linear: complex,
    geometry: ArrayGeometry,
    *,
    wavelength_m: float,
    steering_direction_cosine: float = 0.0,
) -> float:
    r"""Return the direction cosine a monopulse ratio :math:`r` indicates.

    Inverting the one-source ratio: :math:`u_0 + \lambda\arctan(\mathrm{Im}\,r)/(\pi D)`.
    """
    _, spacing_m = _halves(geometry)
    return steering_direction_cosine + wavelength_m * math.atan(ratio_linear.imag) / (
        math.pi * spacing_m
    )


def is_resolved(estimates: ArrayLike, truths: ArrayLike, separation: float) -> bool:
    """Return whether each sorted estimate lies within half the separation of its sorted truth.

    One criterion for every method: two estimates on one blob, or one wandered off, fail.

    Examples
    --------
    >>> truths = [0.1, 0.1625]
    >>> is_resolved([0.10, 0.16], truths, 0.0625), is_resolved([0.10, 0.25], truths, 0.0625)
    (True, False)
    """
    found = np.sort(np.asarray(estimates, dtype=np.float64))
    true = np.sort(np.asarray(truths, dtype=np.float64))
    if found.shape != true.shape:
        msg = f"need one estimate per truth; got {found.shape} and {true.shape}."
        raise ValueError(msg)
    return bool(np.all(np.abs(found - true) < separation / 2.0))


def estimate_doa(
    snapshot: ArrayLike,
    geometry: ArrayGeometry,
    n_sources: int,
    *,
    wavelength_m: float,
    method: DoaMethod,
    covariance: CovarianceKind = "raw",
    training: ArrayLike | None = None,
    n_subarray_elements: int = 12,
    n_grid: int = 4096,
    n_iterations: int | None = None,
    noise_power_w: float = math.nan,
    diagonal_loading_w: float = 0.0,
) -> DoaEstimate:
    """Estimate ``n_sources`` directions in a snapshot ``(n_rx,)`` of a linear array along x.

    A covariance method (Bartlett, Capon, MUSIC, root-MUSIC, ESPRIT) runs on the sample
    covariance of ``training`` ``(n_snapshots, n_rx)``, by default the snapshot alone;
    ``"forward_backward"`` averages it with the backward array's, and ``"fbss"`` smooths it over
    subarrays of ``n_subarray_elements`` first, the method then seeing that subarray. A snapshot
    method (IAA, SPICE, ML) takes the snapshot itself. Grid methods search ``n_grid`` points of
    ``u`` in [-1, 1) and refine each of the ``n_sources`` highest peaks. The amplitudes are then
    fitted to the snapshot at the estimate, and the bound evaluated there with
    ``noise_power_w``, per channel. ``diagonal_loading_w`` loads Capon's covariance as
    :func:`~radar_forge.array.mvdr_weights` does: one snapshot's alone is rank one, singular.
    """
    x = np.asarray(snapshot, dtype=np.complex128)
    if x.shape != (geometry.n_rx,) or np.any(geometry.element_positions_m[:, 1:] != 0.0):
        msg = f"need a snapshot ({geometry.n_rx},) of an array along x; got {x.shape}."
        raise ValueError(msg)
    if method not in DOA_METHODS or covariance not in COVARIANCE_KINDS:
        msg = f"method must be in {DOA_METHODS}, covariance in {COVARIANCE_KINDS}."
        raise ValueError(msg)
    if method in _SNAPSHOT_METHODS and (covariance != "raw" or training is not None):
        msg = f"{method} takes the snapshot alone: no training cells, covariance 'raw'."
        raise ValueError(msg)
    r, array = _prepared(
        sample_covariance(x[None] if training is None else training),
        geometry,
        covariance,
        n_subarray_elements,
    )
    if not 1 <= n_sources < array.n_rx:
        msg = f"n_sources must lie in [1, {array.n_rx}); got {n_sources}."
        raise ValueError(msg)
    grid = np.asarray(np.arange(n_grid) * (2.0 / n_grid) - 1.0, dtype=np.float64)
    spectrum = np.zeros(0)
    if method in ("root_music", "esprit"):
        solve = root_music if method == "root_music" else esprit
        u = solve(r, n_sources, spacing_wavelengths=_spacing_wavelengths(array, wavelength_m))
    elif method == "deterministic_ml":
        u = deterministic_ml(x, geometry, n_sources, wavelength_m=wavelength_m, grid=grid)
    elif method == "spice":
        a_grid = steering_u(array, grid, wavelength_m)
        iterations = n_iterations or _ITERATIONS["spice"]
        spectrum = spice_spectrum(x, a_grid, n_iterations=iterations)
        u = _peaks(spectrum, grid, n_sources, None, array, wavelength_m)
    else:
        a_grid = steering_u(array, grid, wavelength_m)
        evaluate = _spectrum(method, r, x, a_grid, n_sources, n_iterations, diagonal_loading_w)
        spectrum = evaluate(a_grid)
        u = _peaks(spectrum, grid, n_sources, evaluate, array, wavelength_m)
    a_hat = steering_u(geometry, u, wavelength_m)
    fit, *_ = np.linalg.lstsq(a_hat, x, rcond=None)
    amplitude = np.asarray(fit, dtype=np.complex128)
    crb_std = np.full(n_sources, np.nan)
    if math.isfinite(noise_power_w) and np.linalg.cond(a_hat.conj().T @ a_hat) < _MAX_CONDITION:
        derivative = steering_derivative_u(geometry, u, wavelength_m)
        variance = np.diag(crb_deterministic(a_hat, derivative, amplitude, noise_power_w))
        # Nearly coincident estimates leave the Fisher matrix singular to round-off, and its
        # inverse can lose its sign: the bound there is undefined, nan.
        crb_std = np.sqrt(np.where(variance > 0.0, variance, np.nan))
    return DoaEstimate(
        method,
        covariance,
        u,
        amplitude,
        crb_std,
        r,
        np.asarray(np.linalg.eigvalsh(r)[::-1], dtype=np.float64),
        grid if spectrum.size else np.zeros(0),
        spectrum,
    )


def _prepared(
    raw: NDArray[np.complex128], geometry: ArrayGeometry, kind: CovarianceKind, n_subarray: int
) -> tuple[NDArray[np.complex128], ArrayGeometry]:
    """Return the covariance a method runs on, and the array it belongs to."""
    if kind == "raw":
        return raw, geometry
    if kind == "forward_backward":
        return forward_backward_average(raw, geometry), geometry
    part = subarray(geometry, n_subarray)
    return forward_backward_average(spatial_smoothing(raw, n_subarray), part), part


def _spacing_wavelengths(geometry: ArrayGeometry, wavelength_m: float) -> float:
    """Return a ULA's element spacing in wavelengths, or raise for any other array."""
    steps_m = np.diff(geometry.element_positions_m[:, 0])
    if steps_m.size < 1 or not np.allclose(steps_m, steps_m[0], rtol=1e-9, atol=0.0):
        msg = "root-MUSIC and ESPRIT need a uniform linear array."
        raise ValueError(msg)
    return float(steps_m[0]) / wavelength_m


def _spectrum(
    method: DoaMethod,
    covariance: NDArray[np.complex128],
    snapshot: NDArray[np.complex128],
    a_grid: NDArray[np.complex128],
    n_sources: int,
    n_iterations: int | None,
    diagonal_loading_w: float,
) -> Callable[[NDArray[np.complex128]], NDArray[np.float64]]:
    """Return a grid method's spectrum as a function of steering vectors, for any direction.

    IAA's is evaluated with the covariance its iterations converged to.
    """
    if method == "bartlett":
        return lambda a: bartlett_spectrum(covariance, a)
    if method == "capon":
        return lambda a: capon_spectrum(covariance, a, diagonal_loading_w=diagonal_loading_w)
    if method == "music":
        return lambda a: music_spectrum(covariance, a, n_sources)
    _, model = _iaa(snapshot, a_grid, n_iterations or _ITERATIONS["iaa"])
    return lambda a: _weighted_power(model, snapshot, a)


def _peaks(
    spectrum: NDArray[np.float64],
    grid: NDArray[np.float64],
    n_sources: int,
    evaluate: Callable[[NDArray[np.complex128]], NDArray[np.float64]] | None,
    geometry: ArrayGeometry,
    wavelength_m: float,
) -> NDArray[np.float64]:
    """Return the ``n_sources`` highest local maxima of a spectrum on the circular grid, refined.

    Each is refined between its grid neighbours: to the continuous spectrum's maximum, or,
    without one, by a parabola through the logarithm of the three samples. Fewer peaks than
    sources repeat the highest, so the estimate cannot resolve; a flat spectrum has none and
    raises.
    """
    is_peak = (spectrum > np.roll(spectrum, 1)) & (spectrum >= np.roll(spectrum, -1))
    candidates = np.flatnonzero(is_peak)
    if not candidates.size:
        msg = "the spectrum is flat, with no peak: the snapshot holds no signal to locate."
        raise ValueError(msg)
    top = candidates[np.argsort(spectrum[candidates])[::-1][:n_sources]]
    top = np.concatenate([top, np.full(n_sources - top.size, top[0])]).astype(int)
    step = float(grid[1] - grid[0])
    found = []
    # A loop over a handful of peaks, each its own bounded search.
    for index in top:
        if evaluate is None:
            samples = spectrum[[index - 1, index, (index + 1) % grid.size]]
            low, middle, high = np.log(np.maximum(samples, 1e-300))
            curvature = low - 2.0 * middle + high
            offset = 0.5 * (low - high) / curvature if curvature < 0.0 else 0.0
            found.append(grid[index] + offset * step)
            continue

        def negative(u: float, spectrum_at: Callable[..., NDArray[np.float64]] = evaluate) -> float:
            return -float(spectrum_at(steering_u(geometry, u, wavelength_m))[0])

        best = minimize_scalar(
            negative,
            bounds=(grid[index] - step, grid[index] + step),
            method="bounded",
            options={"xatol": step * _REFINE_TOLERANCE},
        )
        found.append(float(best.x))
    # A half-wavelength array sees u and u + 2 alike: fold back into [-1, 1).
    result: NDArray[np.float64] = np.sort(np.mod(np.asarray(found) + 1.0, 2.0) - 1.0)
    return result
