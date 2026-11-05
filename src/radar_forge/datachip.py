r"""The data chip: a crop of a cube around one target, and the fine estimates made on it.

A :class:`DataChip` keeps its parent's axis order, a trailing channel axis whole, its offsets
into the parent and the parent's bin values, so every estimate leaves in parent coordinates.
Range never wraps and is clipped at the edge; Doppler is circular and wraps. On a slow-time
chip :func:`estimate_fine` alternates two one-dimensional problems: slow time, focused in
range, is a tone or a polynomial-phase signal; the in-band range spectrum, focused in
Doppler, is a tone whose frequency is the delay. :func:`estimate_tone` interpolates on DFT
coefficients [1]_-[5]_; ``"haf"`` [6]_ and ``"cpf"`` [8]_ fit the phase polynomial, ``"grft"``
refines it by maximum likelihood [9]_. :func:`cramer_rao_bounds` is the bound they approach.
The echo's slow-time phase is :math:`-2\pi R_b(t)/\lambda`, :math:`t = m/\mathrm{PRF}` from
pulse 0, so its polynomial :math:`c_1 m + c_2 m^2 + c_3 m^3` gives the rate, acceleration and
jerk, bistatic and positive receding.

References
----------
.. [1] D. C. Rife and R. R. Boorstyn, "Single-tone parameter estimation from discrete-time
       observations," *IEEE Trans. Information Theory*, vol. 20, no. 5, pp. 591-598, 1974.
.. [2] B. G. Quinn, "Estimating frequency by interpolation using Fourier coefficients," *IEEE
       Trans. Signal Processing*, vol. 42, no. 5, pp. 1264-1268, 1994.
.. [3] E. Jacobsen and P. Kootsookos, "Fast, accurate frequency estimators," *IEEE Signal
       Processing Magazine*, vol. 24, no. 3, pp. 123-125, 2007.
.. [4] Ç. Candan, "A method for fine resolution frequency estimation from three DFT samples,"
       *IEEE Signal Processing Letters*, vol. 18, no. 6, pp. 351-354, 2011.
.. [5] E. Aboutanios and B. Mulgrew, "Iterative frequency estimation by interpolation on
       Fourier coefficients," *IEEE Trans. Signal Processing*, vol. 53, no. 4, pp. 1237-1242,
       2005.
.. [6] S. Peleg and B. Porat, "Estimation and classification of polynomial-phase signals,"
       *IEEE Trans. Information Theory*, vol. 37, no. 2, pp. 422-430, 1991.
.. [7] S. Peleg and B. Porat, "The Cramer-Rao lower bound for signals with constant
       amplitude and polynomial phase," *IEEE Trans. Signal Processing*, vol. 39, no. 3,
       pp. 749-752, 1991.
.. [8] P. O'Shea, "A fast algorithm for estimating the parameters of a quadratic FM signal,"
       *IEEE Trans. Signal Processing*, vol. 52, no. 2, pp. 385-393, 2004.
.. [9] J. Xu, X.-G. Xia, S.-B. Peng, J. Yu, Y.-N. Peng and L.-C. Qian, "Radar maneuvering
       target motion estimation based on generalized Radon-Fourier transform," *IEEE Trans.
       Signal Processing*, vol. 60, no. 12, pp. 6190-6201, 2012.
.. [10] S. M. Kay, *Fundamentals of Statistical Signal Processing: Estimation Theory*,
        Prentice Hall, 1993.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize, minimize_scalar

from radar_forge.constants import SPEED_OF_LIGHT_MPS

__all__ = [
    "FINE_METHODS",
    "ChipProcessing",
    "CramerRaoBounds",
    "DataChip",
    "FineEstimate",
    "FineMethod",
    "cramer_rao_bounds",
    "crop_range_doppler",
    "crop_slow_time",
    "estimate_fine",
    "estimate_tone",
]

FineMethod = Literal[
    "aboutanios_mulgrew", "parabolic", "quinn", "jacobsen", "candan", "haf", "cpf", "grft"
]
"""The fine estimators: five tone interpolators, then three polynomial-phase methods."""

FINE_METHODS: tuple[FineMethod, ...] = get_args(FineMethod)
"""The fine estimators as a tuple."""

_TONE_METHODS = FINE_METHODS[:5]
# Range and slow time are each estimated twice, each time focused by the other's estimate.
_ALTERNATIONS = 2


@dataclass(frozen=True)
class ChipProcessing:
    """What the parent's processing did, as a chip needs it to refine an estimate.

    ``noise_power_w`` is the noise in one cell of the parent's ``data_rfst``; the fold factor,
    acceleration and jerk are what the migration correction applied (bistatic).
    """

    carrier_frequency_hz: float
    bandwidth_hz: float
    sample_rate_hz: float
    prf_hz: float
    noise_power_w: float
    keystone_applied: bool = False
    fold_factor: int = 0
    compensated_acceleration_mps2: float = 0.0
    compensated_jerk_mps3: float = 0.0


@dataclass(frozen=True)
class DataChip:
    """A crop ``(n_axis0, n_range[, n_rx])`` of a parent ``(n_axis0_parent, n_range_parent, ...)``.

    Axis 0 is slow time (``domain = "slow_time"``, every pulse) or shifted Doppler bins. Chip
    cell ``(i, j)`` is parent cell ``((i + axis0_offset) % parent_shape[0], j + range_offset)``,
    and its range and rate are the parent's bin values; on a slow-time chip
    ``rate_bin_centers_mps`` is the parent's whole Doppler axis.
    """

    data: NDArray[np.complex128]
    domain: Literal["slow_time", "doppler"]
    axis0_offset: int
    range_offset: int
    parent_shape: tuple[int, int]
    range_bin_centers_m: NDArray[np.float64]
    rate_bin_centers_mps: NDArray[np.float64]
    processing: ChipProcessing

    def parent_index(
        self, axis0_index: ArrayLike, column_index: ArrayLike
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return the parent cells of chip cells: integers, exactly."""
        axis0 = (np.asarray(axis0_index, dtype=np.int64) + self.axis0_offset) % self.parent_shape[0]
        return axis0, np.asarray(column_index, dtype=np.int64) + self.range_offset

    def range_at_m(self, fractional_bin: ArrayLike) -> NDArray[np.float64]:
        """Return the bistatic range at fractional chip range indices, by the uniform spacing."""
        spacing_m = self.range_bin_centers_m[1] - self.range_bin_centers_m[0]
        result: NDArray[np.float64] = (
            self.range_bin_centers_m[0] + np.asarray(fractional_bin, dtype=np.float64) * spacing_m
        )
        return result


@dataclass(frozen=True)
class CramerRaoBounds:
    """Standard deviations the Cramér-Rao bound allows, bistatic; nan for a term not fitted."""

    snr_linear: float
    bistatic_range_m: float
    bistatic_range_rate_mps: float
    bistatic_range_acceleration_mps2: float
    bistatic_range_jerk_mps3: float


@dataclass(frozen=True)
class FineEstimate:
    """A chip's estimates in parent coordinates, their bounds, and the chip-local intermediates.

    ``range_index`` is fractional on the chip; ``doppler_index`` fractional on the parent's
    shifted Doppler axis. The rate is unfolded by the chip's fold factor. A tone method fits
    constant rate, so its acceleration and jerk are nan.
    """

    method: FineMethod
    range_index: float
    doppler_index: float
    bistatic_range_m: float
    bistatic_range_rate_mps: float
    bistatic_range_acceleration_mps2: float
    bistatic_range_jerk_mps3: float
    snr_linear: float
    crb: CramerRaoBounds


def crop_slow_time(
    data_rfst: ArrayLike,
    centre_range_bin: int,
    *,
    range_bin_centers_m: ArrayLike,
    rate_bin_centers_mps: ArrayLike,
    processing: ChipProcessing,
    half_width_bins: int = 16,
) -> DataChip:
    """Crop every pulse of the range bins within ``half_width_bins`` of ``centre_range_bin``.

    A crop past either end of the range axis is clipped there; the chip is then shorter.
    """
    data = np.asarray(data_rfst, dtype=np.complex128)
    first, stop = _range_window(centre_range_bin, half_width_bins, data.shape[1])
    return DataChip(
        data[:, first:stop].copy(),
        "slow_time",
        0,
        first,
        (data.shape[0], data.shape[1]),
        np.asarray(range_bin_centers_m, dtype=np.float64)[first:stop].copy(),
        np.asarray(rate_bin_centers_mps, dtype=np.float64).copy(),
        processing,
    )


def crop_range_doppler(
    data_rv: ArrayLike,
    centre_doppler_bin: int,
    centre_range_bin: int,
    *,
    range_bin_centers_m: ArrayLike,
    rate_bin_centers_mps: ArrayLike,
    processing: ChipProcessing,
    half_width_bins: int = 16,
) -> DataChip:
    """Crop a shifted range-Doppler map about a cell: range clipped, Doppler taken round the circle.

    A chip across the fold carries the parent's rates as they are, jump included.
    """
    data = np.asarray(data_rv, dtype=np.complex128)
    first, stop = _range_window(centre_range_bin, half_width_bins, data.shape[1])
    n_doppler = data.shape[0]
    width = min(2 * half_width_bins + 1, n_doppler)
    rows = (centre_doppler_bin - half_width_bins + np.arange(width)) % n_doppler
    return DataChip(
        data[rows, first:stop],
        "doppler",
        int(rows[0]),
        first,
        (n_doppler, data.shape[1]),
        np.asarray(range_bin_centers_m, dtype=np.float64)[first:stop].copy(),
        np.asarray(rate_bin_centers_mps, dtype=np.float64)[rows],
        processing,
    )


def estimate_tone(
    samples: ArrayLike,
    *,
    method: FineMethod = "aboutanios_mulgrew",
    n_iterations: int = 2,
) -> float:
    r"""Return a complex tone's frequency in bins of its :math:`n`-point DFT, in ``[0, n)``.

    From the peak bin :math:`k` and its neighbours, the offset :math:`\delta` is
    ``"parabolic"``, the vertex through :math:`|X|`, biased by up to a quarter of a bin;
    ``"quinn"``, Quinn's first estimator from :math:`\mathrm{Re}(X_{k\pm1}/X_k)` [2]_ eqs. (8)-(9);
    ``"jacobsen"``, :math:`\mathrm{Re}[(X_{k-1} - X_{k+1})/(2X_k - X_{k-1} - X_{k+1})]` [3]_;
    ``"candan"``, that times :math:`\tan(\pi/n)/(\pi/n)` [4]_ eq. (9); or
    ``"aboutanios_mulgrew"``, ``n_iterations`` of
    :math:`\delta \mathrel{+}= \tfrac12\mathrm{Re}[(X_{+} + X_{-})/(X_{+} - X_{-})]` with
    :math:`X_\pm` the DTFT half a bin either side [5]_ §III. Only the last is exact on a
    noiseless tone; all assume a rectangular window [1]_.

    Examples
    --------
    >>> tone = np.exp(2j * np.pi * 10.3 * np.arange(64) / 64)
    >>> round(estimate_tone(tone), 6)
    10.3
    """
    if method not in _TONE_METHODS:
        msg = f"method must be one of {_TONE_METHODS}; got {method!r}."
        raise ValueError(msg)
    x = np.asarray(samples, dtype=np.complex128)
    n = x.size
    spectrum = np.fft.fft(x)
    k = int(np.argmax(np.abs(spectrum)))
    before, peak, after = spectrum[(k - 1) % n], spectrum[k], spectrum[(k + 1) % n]
    if method == "aboutanios_mulgrew":
        return _aboutanios_mulgrew(x, float(k), n_iterations) % n
    if method == "parabolic":
        low, middle, high = np.abs(before), np.abs(peak), np.abs(after)
        offset = 0.5 * (low - high) / (low - 2.0 * middle + high)
    elif method == "quinn":
        below = np.real(before / peak) / (1.0 - np.real(before / peak))
        above = -np.real(after / peak) / (1.0 - np.real(after / peak))
        offset = above if below > 0.0 and above > 0.0 else below
    else:
        offset = np.real((before - after) / (2.0 * peak - before - after))
        if method == "candan":
            offset *= np.tan(np.pi / n) / (np.pi / n)
    return float((k + offset) % n)


def cramer_rao_bounds(
    snr_linear: float,
    *,
    n_pulses: int,
    carrier_frequency_hz: float,
    bandwidth_hz: float,
    prf_hz: float,
    polynomial_order: int = 1,
) -> CramerRaoBounds:
    r"""Return the bound on bistatic range, rate, and to ``polynomial_order`` 2 or 3 more.

    The echo in the range spectrum is :math:`A e^{-j 2\pi (f_0 + f) R_b(t)/c}`, flat over the
    band, :math:`R_b(t) = R_0 + \dot R t + \ddot R t^2/2 + \dddot R t^3/6` from pulse 0, in
    complex white noise; amplitude and phase are unknown. With :math:`\partial\mu/\partial
    \theta_i = j h_i \mu` the Fisher information is :math:`2\rho\,\overline{h_i h_j}`, the bar a
    mean over pulses and the band (Kay [10]_ eq. (15.52); Peleg and Porat [7]_ eq. (3)),
    :math:`\rho` the integrated SNR. Without acceleration the rate's bound is the tone's,
    :math:`\mathrm{var}(\hat\omega) \ge 6/(\rho (N^2 - 1))` (Rife and Boorstyn [1]_ eq. (25)).
    """
    if polynomial_order not in (1, 2, 3) or not snr_linear > 0.0 or n_pulses < polynomial_order + 1:
        msg = "polynomial_order must be 1, 2 or 3, snr_linear positive, with enough pulses."
        raise ValueError(msg)
    time_s = np.arange(n_pulses, dtype=np.float64) / prf_hz
    # Each motion term's time factor: R0, its rate, acceleration and jerk.
    factors = np.stack(
        [time_s**p / np.prod(np.arange(1, p + 1)) for p in range(polynomial_order + 1)]
    )
    wavenumber = 2.0 * np.pi * carrier_frequency_hz / SPEED_OF_LIGHT_MPS
    band_spread = (2.0 * np.pi / SPEED_OF_LIGHT_MPS) ** 2 * bandwidth_hz**2 / 12.0
    size = polynomial_order + 2
    fisher = np.empty((size, size))
    fisher[0, 0] = 1.0
    fisher[0, 1:] = fisher[1:, 0] = -wavenumber * factors.mean(axis=1)
    fisher[1:, 1:] = (wavenumber**2 + band_spread) * (factors @ factors.T) / n_pulses
    fisher *= 2.0 * snr_linear
    scale = 1.0 / np.sqrt(np.diag(fisher))
    covariance = scale[:, None] * np.linalg.inv(scale[:, None] * fisher * scale) * scale
    standard = np.full(4, np.nan)
    standard[: polynomial_order + 1] = np.sqrt(np.diag(covariance)[1:])
    return CramerRaoBounds(float(snr_linear), *(float(s) for s in standard))


def estimate_fine(
    chip: DataChip,
    *,
    method: FineMethod = "aboutanios_mulgrew",
    polynomial_order: int = 2,
) -> FineEstimate:
    r"""Estimate a slow-time chip's range, rate and, for a phase method, acceleration and jerk.

    The chip's range spectrum, padded to four times its bins, is cut to the band. Slow time,
    the band summed with the delay's phase ramp undone, is a tone or a polynomial-phase
    signal; the range tone, the pulses summed with that phase undone, has the delay for its
    frequency. Each is estimated twice, each time focused by the other's latest estimate; a
    tone method does both by :func:`estimate_tone`. ``"haf"`` and ``"cpf"`` then fit
    ``polynomial_order`` 2 (acceleration) or 3 (jerk) to slow time, ``"grft"`` maximises the
    likelihood from the CPF fit, and the range is focused once more by that polynomial. The
    SNR is measured from the focused peak against ``processing.noise_power_w``.
    """
    if method not in FINE_METHODS or chip.domain != "slow_time" or chip.data.ndim != 2:
        msg = "estimate_fine needs a one-channel slow-time chip and a method in FINE_METHODS."
        raise ValueError(msg)
    if method not in _TONE_METHODS and polynomial_order not in (2, 3):
        msg = f"polynomial_order must be 2 or 3 for {method!r}; got {polynomial_order}."
        raise ValueError(msg)
    process = chip.processing
    n_pulses, n_range = chip.data.shape
    n_fft = int(2 ** np.ceil(np.log2(4 * n_range)))
    frequency_hz = np.fft.fftfreq(n_fft, 1.0 / process.sample_rate_hz)
    band = np.argsort(frequency_hz)
    band = band[np.abs(frequency_hz[band]) <= process.bandwidth_hz / 2.0]
    in_band = np.fft.fft(chip.data, n=n_fft, axis=1)[:, band]
    pulse = np.arange(n_pulses, dtype=np.float64)
    tone: FineMethod = method if method in _TONE_METHODS else "aboutanios_mulgrew"

    def focus_range(delay_bins: float) -> NDArray[np.complex128]:
        ramp = np.exp(2j * np.pi * frequency_hz[band] * delay_bins / process.sample_rate_hz)
        result: NDArray[np.complex128] = in_band @ ramp / n_fft
        return result

    def estimate_delay(coefficients: NDArray[np.float64], previous: float) -> float:
        demodulated = np.exp(-1j * np.polyval([*coefficients[::-1], 0.0], pulse)) @ in_band
        bins = estimate_tone(demodulated, method=tone)
        return _nearest_delay(bins, demodulated.size, n_fft, previous)

    # Start from the brightest cell of the chip's own Doppler-range map: coherent, so robust.
    peak_cell = np.argmax(np.abs(np.fft.fft(chip.data, axis=0)))
    delay_bins = float(np.unravel_index(int(peak_cell), chip.data.shape)[1])
    # Each pass focuses one axis with the other's latest estimate: a fixed two passes.
    for _ in range(_ALTERNATIONS):
        coefficients = _phase_polynomial(focus_range(delay_bins), tone, polynomial_order)
        delay_bins = estimate_delay(coefficients, delay_bins)
    slow_time = focus_range(delay_bins)
    if method not in _TONE_METHODS:
        coefficients = _phase_polynomial(slow_time, method, polynomial_order)
        delay_bins = estimate_delay(coefficients, delay_bins)
        slow_time = focus_range(delay_bins)
    focused = np.exp(-1j * np.polyval([*coefficients[::-1], 0.0], pulse)) @ slow_time
    snr = max(float(np.abs(focused) ** 2 / (n_pulses * process.noise_power_w)) - 1.0, 1e-12)
    return _in_parent_coordinates(chip, method, delay_bins, coefficients, snr, polynomial_order)


def _range_window(centre_bin: int, half_width_bins: int, n_range: int) -> tuple[int, int]:
    """Return the clipped ``[first, stop)`` range bins of a crop."""
    if half_width_bins < 1 or not 0 <= centre_bin < n_range:
        msg = f"need half_width_bins >= 1 and centre_bin in [0, {n_range}); got {centre_bin}."
        raise ValueError(msg)
    return max(0, centre_bin - half_width_bins), min(n_range, centre_bin + half_width_bins + 1)


def _nearest_delay(bins: float, n_band: int, n_fft: int, previous: float) -> float:
    r"""Return the delay, chip bins, of a range tone at ``bins``: :math:`-k M/K`, unwrapped."""
    delay = -bins * n_fft / n_band
    return float(delay + n_fft * np.round((previous - delay) / n_fft))


def _phase_polynomial(
    slow_time: NDArray[np.complex128], method: FineMethod, polynomial_order: int
) -> NDArray[np.float64]:
    r"""Return :math:`(c_1, c_2, c_3)`, rad per pulse to the power, of the slow-time phase."""
    n_pulses = slow_time.size
    pulse = np.arange(n_pulses, dtype=np.float64)
    higher = np.zeros(3)
    if method in _TONE_METHODS:
        cycles = estimate_tone(slow_time, method=method) / n_pulses
        return np.array([2.0 * np.pi * _signed(cycles), 0.0, 0.0])
    if method == "haf":
        higher = _haf(slow_time, polynomial_order)
    else:
        higher = _cpf(slow_time, polynomial_order)
    residual = slow_time * np.exp(-1j * (higher[1] * pulse**2 + higher[2] * pulse**3))
    linear = 2.0 * np.pi * _signed(estimate_tone(residual) / n_pulses)
    coefficients = np.array([linear, higher[1], higher[2]])
    if method == "grft":
        coefficients = _maximum_likelihood(slow_time, coefficients, polynomial_order)
    return coefficients


def _haf(slow_time: NDArray[np.complex128], polynomial_order: int) -> NDArray[np.float64]:
    r"""Return :math:`(0, c_2, c_3)` by the high-order ambiguity function, top order first.

    The order-3 product :math:`z_{m+2\tau} z^*_{m+\tau}{}^2 z_m` is a tone at
    :math:`6 c_3 \tau^2`; once :math:`c_3 m^3` is removed, :math:`z_{m+\tau} z^*_m` is one at
    :math:`2 c_2 \tau`, with :math:`\tau = N/3` and :math:`N/2` (Peleg and Porat [6]_ §III).
    """
    n_pulses = slow_time.size
    pulse = np.arange(n_pulses, dtype=np.float64)
    cubic = 0.0
    if polynomial_order == 3:
        lag = n_pulses // 3
        product = (
            slow_time[2 * lag :]
            * np.conj(slow_time[lag : n_pulses - lag]) ** 2
            * slow_time[: n_pulses - 2 * lag]
        )
        cubic = _tone_rad(product) / (6.0 * lag**2)
    work = slow_time * np.exp(-1j * cubic * pulse**3)
    lag = n_pulses // 2
    quadratic = _tone_rad(work[lag:] * np.conj(work[: n_pulses - lag])) / (2.0 * lag)
    return np.array([0.0, quadratic, cubic])


def _cpf(slow_time: NDArray[np.complex128], polynomial_order: int) -> NDArray[np.float64]:
    r"""Return :math:`(0, c_2, c_3)` from the cubic phase function's peaks.

    :math:`CP(n, \Omega) = \sum_m z_{n+m} z_{n-m} e^{-j\Omega m^2}` peaks at the phase's
    second derivative :math:`2c_2 + 6c_3 n` (O'Shea [8]_ eq. (3)), read at the centre and, for
    the cubic, a quarter of the dwell either side.
    """
    n_pulses = slow_time.size
    centre = n_pulses // 2
    if polynomial_order == 2:
        return np.array([0.0, _cpf_peak(slow_time, centre) / 2.0, 0.0])
    offset = n_pulses // 4
    early, late = (_cpf_peak(slow_time, centre + s) for s in (-offset, offset))
    cubic = (late - early) / (6.0 * 2 * offset)
    return np.array([0.0, (_cpf_peak(slow_time, centre) - 6.0 * cubic * centre) / 2.0, cubic])


def _cpf_peak(slow_time: NDArray[np.complex128], centre: int) -> float:
    r"""Return :math:`\Omega` maximising :math:`|CP(n, \Omega)|`: an FFT, then a line search.

    :math:`m^2` is an integer, so on the grid :math:`\Omega = 2\pi q/L` the CPF is the
    :math:`L`-point DFT of the products placed at :math:`u = m^2`. Only
    :math:`|\Omega| \le 4\pi/N` is searched: a phase whose frequency sweeps more than two PRFs
    over the dwell aliases in slow time, and beyond it lie only noise peaks.
    """
    n_pulses = slow_time.size
    half = min(centre, n_pulses - 1 - centre)
    lag = np.arange(half + 1)
    products = slow_time[centre + lag] * slow_time[centre - lag]
    n_fft = int(2 ** np.ceil(np.log2(2 * half**2 + 1)))
    placed = np.zeros(n_fft, dtype=np.complex128)
    placed[lag**2] = products
    magnitude = np.abs(np.fft.fft(placed))
    plausible = np.abs(np.fft.fftfreq(n_fft)) <= 2.0 / n_pulses
    best = int(np.argmax(np.where(plausible, magnitude, -1.0)))
    grid_step = 2.0 * np.pi / n_fft
    start = 2.0 * np.pi * _signed(best / n_fft)

    def negative_magnitude(rate: float) -> float:
        return -float(np.abs(products @ np.exp(-1j * rate * lag**2)))

    found = minimize_scalar(
        negative_magnitude,
        bounds=(start - grid_step, start + grid_step),
        method="bounded",
        options={"xatol": grid_step * 1e-9},
    )
    return float(found.x)


def _maximum_likelihood(
    slow_time: NDArray[np.complex128], start: NDArray[np.float64], polynomial_order: int
) -> NDArray[np.float64]:
    r"""Maximise :math:`|\sum_m z_m e^{-j(c_1 m + c_2 m^2 + c_3 m^3)}|` near ``start``.

    The local generalised Radon-Fourier transform: after the keystone the chip's trajectory is
    straight in range, so its sum along the trajectory is the range-focused sequence, and the
    likelihood is maximised over the phase alone (Xu et al. [9]_ §III). The search runs in
    :math:`c_p N^p`, the phase each term adds over the dwell.
    """
    n_pulses = slow_time.size
    pulse = np.arange(n_pulses, dtype=np.float64)
    n_terms = polynomial_order
    scale = float(n_pulses) ** np.arange(1, n_terms + 1)

    def negative_power(scaled: NDArray[np.float64]) -> float:
        terms = scaled / scale
        phase = sum(terms[p] * pulse ** (p + 1) for p in range(n_terms))
        return -float(np.abs(slow_time @ np.exp(-1j * phase)) ** 2) / n_pulses**2

    initial = start[:n_terms] * scale
    simplex = initial + np.vstack([np.zeros(n_terms), 0.05 * np.eye(n_terms)])
    found = minimize(
        negative_power,
        initial,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": 1e-9, "fatol": 1e-15, "maxiter": 4000},
    )
    result = np.zeros(3)
    result[:n_terms] = found.x / scale
    return result


def _in_parent_coordinates(
    chip: DataChip,
    method: FineMethod,
    delay_bins: float,
    coefficients: NDArray[np.float64],
    snr_linear: float,
    polynomial_order: int,
) -> FineEstimate:
    r"""Convert chip-local delay and phase coefficients to the parent's bistatic quantities.

    :math:`c_1 = -2\pi\dot R T/\lambda` folded, :math:`c_2 = -\pi\ddot R T^2/\lambda`,
    :math:`c_3 = -\pi\dddot R T^3/(3\lambda)`, :math:`T` the PRI; the fold factor adds back
    :math:`-F\lambda\,\mathrm{PRF}` and the compensation what it removed.
    """
    process = chip.processing
    wavelength_m = SPEED_OF_LIGHT_MPS / process.carrier_frequency_hz
    pri_s = 1.0 / process.prf_hz
    n_pulses = chip.data.shape[0]
    cycles = coefficients[0] / (2.0 * np.pi)
    rate_mps = -wavelength_m * cycles / pri_s - process.fold_factor * wavelength_m * process.prf_hz
    fitted = method not in _TONE_METHODS
    acceleration = process.compensated_acceleration_mps2 - coefficients[1] * wavelength_m / (
        np.pi * pri_s**2
    )
    jerk = process.compensated_jerk_mps3 - 3.0 * coefficients[2] * wavelength_m / (np.pi * pri_s**3)
    order = polynomial_order if fitted else 1
    crb = cramer_rao_bounds(
        snr_linear,
        n_pulses=n_pulses,
        carrier_frequency_hz=process.carrier_frequency_hz,
        bandwidth_hz=process.bandwidth_hz,
        prf_hz=process.prf_hz,
        polynomial_order=order,
    )
    return FineEstimate(
        method,
        delay_bins,
        float((cycles * n_pulses + n_pulses // 2) % n_pulses),
        float(chip.range_at_m(delay_bins)),
        float(rate_mps),
        float(acceleration) if fitted else float("nan"),
        float(jerk) if fitted and polynomial_order == 3 else float("nan"),
        snr_linear,
        crb,
    )


def _aboutanios_mulgrew(
    samples: NDArray[np.complex128], peak_bin: float, n_iterations: int
) -> float:
    """Return the frequency, DFT bins, after ``n_iterations`` of A&M's half-bin interpolation."""
    n = samples.size
    index = np.arange(n)
    estimate = peak_bin
    # Each iteration refines the last; two reach the noiseless tone to rounding [5].
    for _ in range(n_iterations):
        plus, minus = (
            samples @ np.exp(-2j * np.pi * index * (estimate + half) / n) for half in (0.5, -0.5)
        )
        estimate += 0.5 * float(np.real((plus + minus) / (plus - minus)))
    return estimate


def _tone_rad(samples: NDArray[np.complex128]) -> float:
    """Return a tone's frequency in rad per sample, in ``[-pi, pi)``, by A&M."""
    return 2.0 * np.pi * _signed(estimate_tone(samples) / samples.size)


def _signed(cycles: float) -> float:
    """Return cycles per sample wrapped into ``[-0.5, 0.5)``."""
    return float((cycles + 0.5) % 1.0 - 0.5)
