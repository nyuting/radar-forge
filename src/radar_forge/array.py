r"""The receive array: its geometry, steering vectors, beamformer weights and covariances.

An :class:`ArrayGeometry` holds element positions in the **array frame**: x along the face (right,
looking out), y boresight, z up; :func:`enu_to_array_frame` is the one rotation from the receive
site's east, north, up frame into it. A direction is the unit vector ``[sin az cos el, cos az cos
el, sin el]``, azimuth from boresight towards +x; a linear array along x observes only its first
component, the direction cosine :math:`u`. The steering vector :math:`a_n = e^{+j2\pi
p_n\cdot\hat u/\lambda}`, unnormalised so :math:`\|a\|^2 = n_{rx}`, is written once, in
:func:`steering_vectors`: an element nearer the source has the shorter path, so with
:func:`~radar_forge.propagation.propagation_phase_rad`'s :math:`-2\pi R/\lambda` it leads in
phase. Weights are returned, never applied in passing; :func:`apply_weights` forms :math:`w^H x`.
A covariance is :math:`E[x x^H]` over snapshots ``(n_snapshots, n_rx)``. Two echoes in one
range-Doppler cell share every snapshot's range and Doppler response, so their covariance is rank
one; :func:`spatial_smoothing` and :func:`forward_backward_average` restore the rank from the
array's shift invariance, at the cost of aperture.

References
----------
.. [1] H. L. Van Trees, *Optimum Array Processing*, Wiley, 2002, ch. 2 (arrays and spatial
       filters).
.. [2] J. Capon, "High-resolution frequency-wavenumber spectrum analysis," *Proc. IEEE*,
       vol. 57, no. 8, pp. 1408-1418, 1969.
.. [3] T.-J. Shan, M. Wax and T. Kailath, "On spatial smoothing for direction-of-arrival
       estimation of coherent signals," *IEEE Trans. Acoust., Speech, Signal Process.*,
       vol. 33, no. 4, pp. 806-811, 1985.
.. [4] S. U. Pillai and B. H. Kwon, "Forward/backward spatial smoothing techniques for coherent
       signal identification," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 37, no. 1,
       pp. 8-15, 1989.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ArrayGeometry",
    "apply_weights",
    "beam_power",
    "conventional_weights",
    "direction_cosine",
    "directions_unit",
    "enu_to_array_frame",
    "forward_backward_average",
    "is_centro_symmetric",
    "mvdr_weights",
    "offset_direction_cosine",
    "rayleigh_width",
    "sample_covariance",
    "spatial_smoothing",
    "steering_vectors",
    "subarray",
    "uniform_linear_array",
]

# Positions within this fraction of the aperture of their mirror image count as symmetric.
_SYMMETRY_TOLERANCE = 1e-9


@dataclass(frozen=True, eq=False)
class ArrayGeometry:
    """Receive element positions in the array frame, ``(n_rx, 3)``, m, and where the array looks.

    The boresight is a compass bearing, clockwise from true north, and an elevation tilt.
    """

    element_positions_m: NDArray[np.float64]
    boresight_azimuth_deg: float = 0.0
    boresight_elevation_deg: float = 0.0

    def __post_init__(self) -> None:
        """Reject positions that are not ``(n_rx, 3)`` with at least one element."""
        positions = np.asarray(self.element_positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
            msg = f"element_positions_m must be (n_rx, 3), n_rx >= 1; got {positions.shape}."
            raise ValueError(msg)
        object.__setattr__(self, "element_positions_m", positions)

    @property
    def n_rx(self) -> int:
        """Number of receive channels."""
        return int(self.element_positions_m.shape[0])


def uniform_linear_array(
    n_rx: int,
    spacing_wavelengths: float,
    wavelength_m: float,
    *,
    boresight_azimuth_deg: float = 0.0,
    boresight_elevation_deg: float = 0.0,
) -> ArrayGeometry:
    r"""Return ``n_rx`` elements along x, :math:`x_n = (n - (n_{rx}-1)/2)\,d`: phase centre at 0.

    Centred, so its steering vector is conjugate centro-symmetric, :math:`J a^* = a`.
    """
    if n_rx < 1 or not spacing_wavelengths > 0.0 or not wavelength_m > 0.0:
        msg = "need n_rx >= 1, and spacing_wavelengths and wavelength_m strictly positive."
        raise ValueError(msg)
    positions_m = np.zeros((n_rx, 3))
    positions_m[:, 0] = (np.arange(n_rx) - (n_rx - 1) / 2.0) * spacing_wavelengths * wavelength_m
    return ArrayGeometry(positions_m, boresight_azimuth_deg, boresight_elevation_deg)


def subarray(geometry: ArrayGeometry, n_elements: int) -> ArrayGeometry:
    """Return the first ``n_elements``, moved onto their own phase centre.

    A ULA's subarrays are shifts of one another, so a smoothed covariance is that of this one
    subarray; centring it keeps its steering vector conjugate centro-symmetric.
    """
    if not 1 <= n_elements <= geometry.n_rx:
        msg = f"n_elements must lie in [1, {geometry.n_rx}]; got {n_elements}."
        raise ValueError(msg)
    positions_m = geometry.element_positions_m[:n_elements]
    return ArrayGeometry(
        positions_m - positions_m.mean(axis=0),
        geometry.boresight_azimuth_deg,
        geometry.boresight_elevation_deg,
    )


def is_centro_symmetric(geometry: ArrayGeometry) -> bool:
    """Return whether element ``n`` mirrors element ``n_rx - 1 - n`` through the origin."""
    positions_m = geometry.element_positions_m
    scale_m = max(float(np.max(np.abs(positions_m))), 1.0)
    mirrored_m = positions_m + positions_m[::-1]
    return bool(np.all(np.abs(mirrored_m) <= _SYMMETRY_TOLERANCE * scale_m))


def directions_unit(azimuth_rad: ArrayLike, elevation_rad: ArrayLike) -> NDArray[np.float64]:
    """Return unit vectors ``[sin az cos el, cos az cos el, sin el]``, ``(n_directions, 3)``.

    In the site's ENU frame for a compass azimuth; in the array frame for one from boresight.
    """
    azimuth, elevation = np.broadcast_arrays(
        np.atleast_1d(np.asarray(azimuth_rad, dtype=np.float64)),
        np.asarray(elevation_rad, dtype=np.float64),
    )
    result: NDArray[np.float64] = np.stack(
        [
            np.sin(azimuth) * np.cos(elevation),
            np.cos(azimuth) * np.cos(elevation),
            np.sin(elevation),
        ],
        axis=-1,
    )
    return result


def _array_axes(geometry: ArrayGeometry) -> NDArray[np.float64]:
    r"""Return the array's x, y and z axes in ENU, as rows.

    y is the boresight :math:`(\beta, \epsilon)`; x is horizontal, :math:`(\cos\beta,
    -\sin\beta, 0)`, to the right looking out; z is :math:`x \times y`, up when untilted.
    """
    bearing = np.radians(geometry.boresight_azimuth_deg)
    boresight = directions_unit(bearing, np.radians(geometry.boresight_elevation_deg))[0]
    right = np.array([np.cos(bearing), -np.sin(bearing), 0.0])
    return np.stack([right, boresight, np.cross(right, boresight)])


def enu_to_array_frame(geometry: ArrayGeometry, directions_enu: ArrayLike) -> NDArray[np.float64]:
    """Rotate vectors ``(n, 3)`` from the receive site's ENU frame into the array frame.

    With no tilt, a direction's azimuth from boresight is its bearing less the boresight's.
    """
    vectors = np.atleast_2d(np.asarray(directions_enu, dtype=np.float64))
    result: NDArray[np.float64] = vectors @ _array_axes(geometry).T
    return result


def steering_vectors(
    geometry: ArrayGeometry, directions: ArrayLike, wavelength_m: float
) -> NDArray[np.complex128]:
    r"""Return :math:`a_n(\hat u) = e^{+j2\pi p_n\cdot\hat u/\lambda}`, ``(n_rx, n_directions)``.

    ``directions`` are unit vectors in the array frame, ``(n_directions, 3)``. Unnormalised,
    :math:`\|a\|^2 = n_{rx}`; broadside, every element is 1 (Van Trees [1] ch. 2).

    Examples
    --------
    30 degrees right of boresight, the element at larger x leads by :math:`\pi/2`:

    >>> ula = uniform_linear_array(2, 0.5, 0.03)
    >>> np.angle(steering_vectors(ula, directions_unit(np.pi / 6, 0.0), 0.03)[:, 0]) / np.pi
    array([-0.25,  0.25])
    """
    unit = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    phase_rad = 2.0 * np.pi * (geometry.element_positions_m @ unit.T) / wavelength_m
    result: NDArray[np.complex128] = np.exp(1j * phase_rad)
    return result


def rayleigh_width(geometry: ArrayGeometry, wavelength_m: float) -> float:
    r"""Return a linear array's Rayleigh width in direction cosine, :math:`\lambda/(n_{rx} d)`.

    :math:`d` is the mean spacing along x, so :math:`n_{rx} d` is the aperture.
    """
    x_m = geometry.element_positions_m[:, 0]
    if geometry.n_rx < 2:
        msg = "a Rayleigh width needs at least two elements."
        raise ValueError(msg)
    spacing_m = (float(x_m.max()) - float(x_m.min())) / (geometry.n_rx - 1)
    return wavelength_m / (geometry.n_rx * spacing_m)


def direction_cosine(geometry: ArrayGeometry, aoa_rad: ArrayLike) -> NDArray[np.float64]:
    """Return the direction cosine along x of arrival angles ``(n, 2)`` (bearing, elevation)."""
    angles = np.atleast_2d(np.asarray(aoa_rad, dtype=np.float64))
    in_array = enu_to_array_frame(geometry, directions_unit(angles[:, 0], angles[:, 1]))
    result: NDArray[np.float64] = in_array[:, 0]
    return result


def offset_direction_cosine(
    geometry: ArrayGeometry, aoa_rad: ArrayLike, offset: float
) -> NDArray[np.float64]:
    """Return arrival angles ``(n, 2)`` moved by ``offset`` in direction cosine along x.

    The array-frame elevation is kept and the direction stays in front of the array; the
    result is a compass bearing in ``[0, 2 pi)`` and an elevation, as ``aoa_rad`` is.
    """
    angles = np.atleast_2d(np.asarray(aoa_rad, dtype=np.float64))
    in_array = enu_to_array_frame(geometry, directions_unit(angles[:, 0], angles[:, 1]))
    u_x = in_array[:, 0] + offset
    u_y_squared = 1.0 - u_x**2 - in_array[:, 2] ** 2
    if not np.all(u_y_squared >= 0.0):
        msg = f"an offset of {offset} in direction cosine leaves the visible region."
        raise ValueError(msg)
    moved = np.stack([u_x, np.sqrt(u_y_squared), in_array[:, 2]], axis=-1) @ _array_axes(geometry)
    bearing_rad = np.mod(np.arctan2(moved[:, 0], moved[:, 1]), 2.0 * np.pi)
    result: NDArray[np.float64] = np.stack(
        [bearing_rad, np.arcsin(np.clip(moved[:, 2], -1.0, 1.0))], axis=-1
    )
    return result


def conventional_weights(steering: ArrayLike) -> NDArray[np.complex128]:
    r"""Return the conventional (Bartlett) weights :math:`w = a/(a^H a)`, column by column.

    Distortionless, :math:`w^H a = 1`; on white noise of :math:`\sigma^2` per channel the
    output noise is :math:`\sigma^2/n_{rx}` (Van Trees [1] ch. 2).
    """
    a = np.asarray(steering, dtype=np.complex128)
    result: NDArray[np.complex128] = a / np.sum(np.abs(a) ** 2, axis=0)
    return result


def mvdr_weights(
    covariance: ArrayLike, steering: ArrayLike, *, diagonal_loading_w: float = 0.0
) -> NDArray[np.complex128]:
    r"""Return Capon's weights :math:`R^{-1}a/(a^H R^{-1} a)`, ``R`` plus ``diagonal_loading_w``.

    The least output power :math:`w^H R w` with :math:`w^H a = 1`: distortionless towards
    :math:`a`, minimum variance elsewhere (Capon [2] eqs. (18)-(20)). Returned, never applied.
    """
    r = np.asarray(covariance, dtype=np.complex128)
    a = np.asarray(steering, dtype=np.complex128)
    whitened = np.linalg.solve(r + diagonal_loading_w * np.eye(r.shape[0]), a)
    result: NDArray[np.complex128] = whitened / np.sum(a.conj() * whitened, axis=0)
    return result


def apply_weights(weights: ArrayLike, cube: ArrayLike, *, axis: int = -1) -> NDArray[np.complex128]:
    r"""Return :math:`y = w^H x` along the channel ``axis``.

    Weights ``(n_rx,)`` remove that axis; ``(n_rx, n_beams)`` put a beam axis last.
    """
    channels_last = np.moveaxis(np.asarray(cube, dtype=np.complex128), axis, -1)
    result: NDArray[np.complex128] = channels_last @ np.asarray(weights, dtype=np.complex128).conj()
    return result


def beam_power(weights: ArrayLike, covariance: ArrayLike) -> NDArray[np.float64]:
    r"""Return each beam's output power :math:`w^H R w`, W, ``(n_beams,)`` or a scalar."""
    w = np.asarray(weights, dtype=np.complex128)
    r = np.asarray(covariance, dtype=np.complex128)
    result: NDArray[np.float64] = np.real(np.sum(w.conj() * (r @ w), axis=0))
    return result


def sample_covariance(snapshots: ArrayLike) -> NDArray[np.complex128]:
    r"""Return :math:`\hat R = \frac{1}{K}\sum_k x_k x_k^H` of snapshots ``(n_snapshots, n_rx)``.

    Row ``k`` is snapshot :math:`x_k`, so :math:`\hat R = X^T X^* / K`: entry :math:`(i, j)` is
    the mean of :math:`x_i x_j^*`.
    """
    x = np.atleast_2d(np.asarray(snapshots, dtype=np.complex128))
    result: NDArray[np.complex128] = x.T @ x.conj() / x.shape[0]
    return result


def spatial_smoothing(covariance: ArrayLike, n_subarray_elements: int) -> NDArray[np.complex128]:
    r"""Return :math:`\frac{1}{K}\sum_{k=0}^{K-1} R[k:k+L,\,k:k+L]`, :math:`K = n_{rx} - L + 1`.

    Each subarray sees the sources with its own phase, so the average decorrelates a coherent
    pair once :math:`K \ge 2`, and resolves at the subarray's width :math:`\lambda/(L d)`
    (Shan, Wax and Kailath [3]).
    """
    r = np.asarray(covariance, dtype=np.complex128)
    n_rx, length = r.shape[0], n_subarray_elements
    if not 1 <= length <= n_rx:
        msg = f"n_subarray_elements must lie in [1, {n_rx}]; got {length}."
        raise ValueError(msg)
    rows = np.arange(n_rx - length + 1)[:, None] + np.arange(length)
    result: NDArray[np.complex128] = r[rows[:, :, None], rows[:, None, :]].mean(axis=0)
    return result


def forward_backward_average(
    covariance: ArrayLike, geometry: ArrayGeometry
) -> NDArray[np.complex128]:
    r"""Return :math:`\tfrac12(R + J R^* J)`, :math:`J` the exchange matrix.

    A centro-symmetric array's steering vector satisfies :math:`J a^* = a`, so the backward
    array sees the same sources at conjugated phases: the average decorrelates a pair unless
    their relative phase is 0 or :math:`\pi` (Pillai and Kwon [4]). ``geometry`` is the array
    the covariance belongs to, a :func:`subarray` after smoothing.
    """
    r = np.asarray(covariance, dtype=np.complex128)
    if r.shape != (geometry.n_rx, geometry.n_rx) or not is_centro_symmetric(geometry):
        msg = "forward-backward averaging needs the covariance of a centro-symmetric array."
        raise ValueError(msg)
    result: NDArray[np.complex128] = 0.5 * (r + r[::-1, ::-1].conj())
    return result
