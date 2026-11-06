r"""Range migration: the keystone transform, Doppler-rate compensation and their searches.

A fast target walks across range bins within one coherent processing interval, and an
accelerating one spreads across Doppler bins. Every public function takes compressed pulses
``data_rfst``, ``(n_pulses, n_range[, n_rx])`` (slow time, bistatic range bin, channel), acts
on axes 0 and 1 only, and returns a frozen result holding a new ``data_rfst`` with its
intermediates; nothing is modified in place. The work is done in the range spectrum, the FFT
along range, unshifted, whose **signed** frequency :math:`f` makes the radio frequency
:math:`f_0 + f`. There a point echo reads :math:`|P(f)|^2 e^{-j 2\pi (f_0 + f) R_b(t)/c}`:
the :math:`f R_b` term walks in range, the :math:`f_0 R_b` term is the Doppler phase [1]_.
Slow time :math:`t = m/\mathrm{PRF}` is zero at pulse 0, the frame's truth time and the
keystone's reference. :math:`R_b` is bistatic and its rate positive receding, so
:math:`f_D = -\dot R_b/\lambda` (:func:`~radar_forge.propagation.doppler_shift_hz`).
:func:`pulse_compress` makes ``data_rfst`` from a cube, through its range gate, and
:func:`compensate_migration` applies a scenario's ``[migration]`` to it.

References
----------
.. [1] R. P. Perry, R. C. DiPietro and R. L. Fante, "SAR imaging of moving targets," *IEEE
       Trans. Aerospace and Electronic Systems*, vol. 35, no. 1, pp. 188-200, 1999.
.. [2] D. Zhu, Y. Li and Z. Zhu, "A keystone transform without interpolation for SAR ground
       moving-target imaging," *IEEE Geosci. Remote Sens. Lett.*, vol. 4, no. 1, pp. 18-22, 2007.
.. [3] P. Huang, G. Liao, Z. Yang, X.-G. Xia, J.-T. Ma and J. Ma, "Long-time coherent
       integration for weak maneuvering target detection and high-order motion parameter
       estimation based on keystone transform," *IEEE Trans. Signal Processing*, vol. 64,
       no. 15, pp. 4013-4026, 2016.
.. [4] D. Kirkland, "Imaging moving targets using the second-order keystone transform," *IET
       Radar, Sonar & Navigation*, vol. 5, no. 8, pp. 902-910, 2011.
.. [5] L. I. Bluestein, "A linear filtering approach to the computation of discrete Fourier
       transform," *IEEE Trans. Audio and Electroacoustics*, vol. 18, no. 4, pp. 451-455, 1970.
.. [6] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
       Pearson, 2010.
.. [7] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge._fourier import aboutanios_mulgrew, signed_cycles
from radar_forge.constants import SPEED_OF_LIGHT_MPS
from radar_forge.dsp import matched_filter
from radar_forge.radar import Radar
from radar_forge.signal import RangeGate, lfm_chirp

__all__ = [
    "KEYSTONE_METHODS",
    "CompensatedPulses",
    "DopplerRateCompensation",
    "FoldFactorSearch",
    "IntegratedPeak",
    "KeystoneMethod",
    "KeystoneResult",
    "RangeSpectrum",
    "compensate_migration",
    "doppler_rate_compensation",
    "fold_factor_search",
    "integrated_peak",
    "keystone",
    "pulse_compress",
    "range_spectrum",
    "second_order_keystone",
]

KeystoneMethod = Literal["czt", "sinc"]
"""How slow time is rescaled: band-limited (DFT) interpolation by chirp-z, or truncated sinc."""

KEYSTONE_METHODS: tuple[KeystoneMethod, ...] = get_args(KeystoneMethod)
"""The keystone methods as a tuple."""

# Half-length, in pulses, of the truncated-sinc baseline's Hann-tapered kernel.
_SINC_HALF_TAPS = 8
# Aboutanios-Mulgrew iterations that refine a hypothesis's Doppler peak.
_PEAK_ITERATIONS = 3
# Range bins either side of the brightest cell that score an acceleration hypothesis.
_STRIP_HALF_WIDTH = 2
# Range frequencies the chirp-z transform works on at once: 256 x L complex is 8 MiB at L = 2048.
_PLAN_BLOCK_ROWS = 256


@dataclass(frozen=True)
class RangeSpectrum:
    """The range spectrum ``(n_pulses, n_range_fft[, n_rx])`` and its signed frequency axis."""

    spectrum: NDArray[np.complex128]
    range_frequency_hz: NDArray[np.float64]
    n_range: int

    def data_rfst(self, spectrum: ArrayLike | None = None) -> NDArray[np.complex128]:
        """Return ``spectrum`` (default: this one) back in range, cut to ``n_range`` bins."""
        values = self.spectrum if spectrum is None else np.asarray(spectrum)
        result: NDArray[np.complex128] = np.fft.ifft(values, axis=1)[:, : self.n_range]
        return result


@dataclass(frozen=True)
class KeystoneResult:
    r"""A rescaled ``data_rfst``; ``slow_time_scale`` is :math:`t/\tau` per range frequency."""

    data_rfst: NDArray[np.complex128]
    range_frequency_hz: NDArray[np.float64]
    slow_time_scale: NDArray[np.float64]
    fold_factor: int
    method: KeystoneMethod


@dataclass(frozen=True)
class FoldFactorSearch:
    """The keystone's integrated peak at each fold factor, and the best one applied.

    ``doppler_cycles_per_pulse`` is the peak's unfolded Doppler at the carrier, in PRFs.
    """

    data_rfst: NDArray[np.complex128]
    fold_factors: NDArray[np.int64]
    integrated_peak_linear: NDArray[np.float64]
    fold_factor: int
    doppler_cycles_per_pulse: float


@dataclass(frozen=True)
class DopplerRateCompensation:
    """The filter's search curve ``(n_accelerations, n_jerks)``, and the best one applied."""

    data_rfst: NDArray[np.complex128]
    accelerations_mps2: NDArray[np.float64]
    jerks_mps3: NDArray[np.float64]
    integrated_peak_linear: NDArray[np.float64]
    acceleration_mps2: float
    jerk_mps3: float


@dataclass(frozen=True)
class IntegratedPeak:
    """A map's peak power off the bin grid, at a fractional unshifted Doppler bin and range bin."""

    power_linear: float
    doppler_bin: float
    range_bin: float


def range_spectrum(
    data_rfst: ArrayLike, *, sample_rate_hz: float, n_range_fft: int | None = None
) -> RangeSpectrum:
    r"""Return the FFT of ``data_rfst`` along range, unshifted, with its signed frequencies.

    ``n_range_fft`` defaults to the next power of two at least twice the range bins, so a walk
    of up to ``n_range`` bins cannot wrap. Bin :math:`k` is :math:`f = k f_s/M` below
    :math:`M/2` and negative above, ``fftfreq`` (Oppenheim [6]_ §8.5).
    """
    data = _cube(data_rfst)
    n_range = data.shape[1]
    n_fft = int(2 ** np.ceil(np.log2(2 * n_range))) if n_range_fft is None else n_range_fft
    if n_fft < n_range or not sample_rate_hz > 0.0:
        msg = f"n_range_fft ({n_fft}) must cover the {n_range} range bins; f_s must be positive."
        raise ValueError(msg)
    spectrum = np.fft.fft(data, n=n_fft, axis=1)
    frequency_hz = np.asarray(np.fft.fftfreq(n_fft, 1.0 / sample_rate_hz), dtype=np.float64)
    return RangeSpectrum(spectrum, frequency_hz, n_range)


def keystone(
    data_rfst: ArrayLike,
    *,
    carrier_frequency_hz: float,
    sample_rate_hz: float,
    fold_factor: int = 0,
    method: KeystoneMethod = "czt",
    n_range_fft: int | None = None,
) -> KeystoneResult:
    r"""Remove linear range walk for every velocity at once: :math:`t = \tau f_0/(f_0 + f)`.

    Substituting into :math:`(f_0 + f)\dot R_b t` leaves :math:`f_0 \dot R_b \tau`, the same at
    every range frequency, so each echo stays in its pulse-0 range bin (Perry [1]_). Each
    frequency's slow time is rescaled by band-limited (DFT) interpolation, a chirp-z transform
    (Zhu [2]_), or by the baseline, a Hann-tapered sinc of eight pulses each side. DFT
    interpolation treats the dwell as periodic: the low frequencies, stretched past the last
    pulse, read the first, so the closing pulses ring. Slow time sees only the alias of a
    Doppler beyond :math:`\pm\mathrm{PRF}/2`; the fold factor :math:`F`, the whole PRFs folded
    away, is put back by :math:`e^{+j 2\pi F m f_0/(f_0 + f)}` (Huang [3]_).
    """
    if method not in KEYSTONE_METHODS:
        msg = f"method must be one of {KEYSTONE_METHODS}; got {method!r}."
        raise ValueError(msg)
    spectrum = range_spectrum(data_rfst, sample_rate_hz=sample_rate_hz, n_range_fft=n_range_fft)
    scale = _keystone_scale(spectrum.range_frequency_hz, carrier_frequency_hz)
    rescaled = _rescale_slow_time(spectrum.spectrum, scale, method)
    if fold_factor:
        n_pulses = rescaled.shape[-1]
        rescaled *= _rows(_fold_phase(scale, n_pulses) ** fold_factor, rescaled.ndim)
    return KeystoneResult(
        _to_rfst(rescaled, spectrum.n_range),
        spectrum.range_frequency_hz,
        scale,
        int(fold_factor),
        method,
    )


# compensate_migration's ``keystone`` argument names the interpolation and hides the function.
_keystone = keystone


def second_order_keystone(
    data_rfst: ArrayLike,
    *,
    carrier_frequency_hz: float,
    sample_rate_hz: float,
    method: KeystoneMethod = "czt",
    n_range_fft: int | None = None,
) -> KeystoneResult:
    r"""Remove range curvature: :math:`t = \tau\sqrt{f_0/(f_0 + f)}`.

    Substituting into :math:`(f_0 + f)\ddot R_b t^2/2` leaves :math:`f_0 \ddot R_b \tau^2/2`,
    the same at every range frequency, so a target accelerating from rest at pulse 0 stays in
    one range bin (Kirkland [4]_). It rescales any linear walk too, so it suits a target
    whose walk is removed or negligible; the quadratic phase is left for
    :func:`doppler_rate_compensation`. Interpolation as :func:`keystone`.
    """
    spectrum = range_spectrum(data_rfst, sample_rate_hz=sample_rate_hz, n_range_fft=n_range_fft)
    scale = np.sqrt(_keystone_scale(spectrum.range_frequency_hz, carrier_frequency_hz))
    rescaled = _rescale_slow_time(spectrum.spectrum, scale, method)
    return KeystoneResult(
        _to_rfst(rescaled, spectrum.n_range), spectrum.range_frequency_hz, scale, 0, method
    )


def fold_factor_search(
    data_rfst: ArrayLike,
    *,
    carrier_frequency_hz: float,
    sample_rate_hz: float,
    bandwidth_hz: float,
    fold_factor_max: int,
    method: KeystoneMethod = "czt",
    n_range_fft: int | None = None,
) -> FoldFactorSearch:
    r"""Keystone at every fold factor in :math:`[-F_{max}, F_{max}]`; keep the brightest peak.

    The rescaling is done once and each :math:`F` only multiplies by its phase. A wrong
    :math:`F` leaves a walk of :math:`\Delta F\,\lambda\,\mathrm{PRF}` metres a second, so the
    integrated peak of the whole map is largest at the true one (Huang [3]_). Each
    hypothesis is scored on every :math:`\lfloor M/n_{range}\rfloor`-th range frequency: that
    undoes the zero padding, so the map's range circle is still the whole gate and nothing
    aliases. The Doppler at :math:`f_0 + f` is :math:`\nu (f_0 + f)/f_0` PRFs; when the band
    edges :math:`f = \pm B/2` put the peak's :math:`\nu` in two fold intervals, one :math:`F`
    cannot keystone it and the target is refused, a case [3]_ treats separately.
    """
    if fold_factor_max < 0:
        msg = f"fold_factor_max must be non-negative; got {fold_factor_max}."
        raise ValueError(msg)
    spectrum = range_spectrum(data_rfst, sample_rate_hz=sample_rate_hz, n_range_fft=n_range_fft)
    scale = _keystone_scale(spectrum.range_frequency_hz, carrier_frequency_hz)
    rescaled = _rescale_slow_time(spectrum.spectrum, scale, method)
    n_pulses, ndim = rescaled.shape[-1], rescaled.ndim
    every = max(1, scale.size // spectrum.n_range)
    step = _rows(_fold_phase(scale[::every], n_pulses), ndim)
    phase = np.conj(step) ** fold_factor_max
    fold_factors = np.arange(-fold_factor_max, fold_factor_max + 1)
    peaks, cycles = np.zeros(fold_factors.size), np.zeros(fold_factors.size)
    # One hypothesis at a time, each phase the last times one step: a whole map each.
    for index in range(fold_factors.size):
        peaks[index], cycles[index] = _scored_peak(rescaled[::every] * phase)
        phase = phase * step
    best = int(np.argmax(peaks))
    fold_factor = int(fold_factors[best])
    doppler_cycles = signed_cycles(cycles[best]) + fold_factor
    edges = doppler_cycles * (1.0 + np.array([-0.5, 0.5]) * bandwidth_hz / carrier_frequency_hz)
    if np.any(np.floor(edges + 0.5) != fold_factor):
        msg = (
            f"the target at {doppler_cycles:.4f} PRFs of Doppler spans fold factors "
            f"{np.floor(edges + 0.5).astype(int).tolist()} across the band: one fold factor "
            "cannot keystone it, a case this implementation does not treat (Huang [3])."
        )
        raise ValueError(msg)
    if fold_factor:
        rescaled *= _rows(_fold_phase(scale, n_pulses) ** fold_factor, ndim)
    chosen = _to_rfst(rescaled, spectrum.n_range)
    return FoldFactorSearch(chosen, fold_factors, peaks, fold_factor, float(doppler_cycles))


def doppler_rate_compensation(
    data_rfst: ArrayLike,
    *,
    carrier_frequency_hz: float,
    sample_rate_hz: float,
    prf_hz: float,
    accelerations_mps2: ArrayLike,
    jerks_mps3: ArrayLike | None = None,
    n_range_fft: int | None = None,
) -> DopplerRateCompensation:
    r"""Undo acceleration and jerk on a keystoned cube; search them, keep the brightest peak.

    After the keystone :math:`(f_0 + f) R_b(t)` holds :math:`\ddot R_b \tau^2 f_0^2/(2(f_0 + f))`
    and :math:`\dddot R_b \tau^3 f_0^3/(6(f_0 + f)^2)`, so for a bistatic acceleration
    :math:`a` and jerk :math:`j` the filter (Huang [3]_)

    .. math::

        H(f, \tau) = \exp\left(+j\frac{2\pi}{c}\left[\frac{f_0^2}{f_0 + f}\frac{a\tau^2}{2}
                     + \frac{f_0^3}{(f_0 + f)^2}\frac{j\tau^3}{6}\right]\right)

    cancels both. Every grid pair is scored by :func:`integrated_peak` on the five range bins
    about the input's brightest cell, which the keystone holds still, with :math:`H` at
    :math:`f = 0`: the rest of :math:`H` straightens a curvature of
    :math:`a\tau^2 f/(2 f_0)`, under a tenth of a bin here, and moves no Doppler. The best
    pair's whole :math:`H` is then applied. The curve is ``(n_accelerations, n_jerks)``; jerk
    defaults to zero. On an unkeystoned cube the filter only dechirps; the walk stays.
    """
    data = _cube(data_rfst)
    spectrum = range_spectrum(data, sample_rate_hz=sample_rate_hz, n_range_fft=n_range_fft)
    accelerations = np.atleast_1d(np.asarray(accelerations_mps2, dtype=np.float64))
    jerks = np.atleast_1d(np.asarray(0.0 if jerks_mps3 is None else jerks_mps3, np.float64))
    if accelerations.ndim != 1 or jerks.ndim != 1 or not accelerations.size:
        msg = "accelerations_mps2 and jerks_mps3 must be non-empty vectors."
        raise ValueError(msg)
    n_pulses, ndim = data.shape[0], data.ndim
    centre = int(integrated_peak(data, refine_range=False).range_bin)
    strip = data[:, max(0, centre - _STRIP_HALF_WIDTH) : centre + _STRIP_HALF_WIDTH + 1]
    carrier = np.array([0.0])
    quadratic, cubic = _rate_phase_rad(carrier, carrier_frequency_hz, prf_hz, n_pulses)
    peaks = np.zeros((accelerations.size, jerks.size))
    # One hypothesis at a time over the strip: a few thousand samples each.
    for (i, k), _ in np.ndenumerate(peaks):
        phase = np.exp(1j * (accelerations[i] * quadratic + jerks[k] * cubic))
        peaks[i, k] = integrated_peak(strip * _expand(phase, ndim), refine_range=False).power_linear
    i, k = np.unravel_index(int(np.argmax(peaks)), peaks.shape)
    quadratic, cubic = _rate_phase_rad(
        spectrum.range_frequency_hz, carrier_frequency_hz, prf_hz, n_pulses
    )
    best = _expand(np.exp(1j * (accelerations[i] * quadratic + jerks[k] * cubic)), ndim)
    return DopplerRateCompensation(
        spectrum.data_rfst(spectrum.spectrum * best),
        accelerations,
        jerks,
        peaks,
        float(accelerations[i]),
        float(jerks[k]),
    )


def integrated_peak(
    data_rfst: ArrayLike, *, refine_range: bool = True, oversample: int = 16
) -> IntegratedPeak:
    r"""Return the peak of :math:`|\mathrm{FFT}_m(\mathrm{data})|^2` off the bin grid.

    The strongest cell of the unshifted Doppler FFT, its Doppler refined on that range bin by
    Aboutanios-Mulgrew iterations, exact for a tone; with ``refine_range``, its range by
    band-limited interpolation of the Doppler-focused profile on a ``1/oversample``-bin grid.
    Off the grid, scalloping cannot pass for migration loss. Channels add in power.
    """
    data = _cube(data_rfst)
    n_pulses, n_range = data.shape[:2]
    magnitude = np.abs(_doppler_spectrum(data))
    power = magnitude if data.ndim == 2 else np.sum(magnitude**2, axis=tuple(range(2, data.ndim)))
    doppler_bin, range_bin = np.unravel_index(int(np.argmax(power)), power.shape)
    pulses = data[:, range_bin].reshape(n_pulses, -1)
    strongest = int(np.argmax(np.sum(np.abs(pulses) ** 2, axis=0)))
    cycles = aboutanios_mulgrew(pulses[:, strongest], float(doppler_bin), _PEAK_ITERATIONS)
    cycles /= n_pulses
    steering = np.exp(-2j * np.pi * cycles * np.arange(n_pulses))
    doppler = float(cycles * n_pulses % n_pulses)
    if not refine_range:
        power_linear = float(np.sum(np.abs(steering @ pulses) ** 2))
        return IntegratedPeak(power_linear, doppler, float(range_bin))
    profile = np.tensordot(steering, data, axes=(0, 0)).reshape(n_range, -1)
    n_fft = int(2 ** np.ceil(np.log2(2 * n_range)))
    offsets = range_bin + np.arange(-oversample, oversample + 1) / oversample
    basis = np.exp(2j * np.pi * np.outer(offsets, np.fft.fftfreq(n_fft)))
    fine = basis @ np.fft.fft(profile, n=n_fft, axis=0) / n_fft
    power_fine = np.sum(np.abs(fine) ** 2, axis=1)
    best = int(np.argmax(power_fine))
    return IntegratedPeak(float(power_fine[best]), doppler, float(offsets[best]))


def pulse_compress(
    cube: ArrayLike, radar: Radar, *, gate: RangeGate | None = None
) -> NDArray[np.complex128]:
    """Matched-filter a pulsed cube along fast time: ``data_rfst``, the same shape.

    Without a gate, or with one recording the whole PRI, the rows are one stream, since the
    receiver keeps listening: compressed as one and cut back into rows, an echo that runs into
    the next row compresses whole and every cell holds the same noise. A gated cube's rows are
    not contiguous, so each is compressed alone. Either way the reference's group delay,
    ``n_reference - 1``, is cut, so the gate's first sample lands at bin 0 (Richards [7]_ §4.6).
    """
    samples = np.asarray(cube, dtype=np.complex128)
    gate = RangeGate() if gate is None else gate
    if samples.shape[1] != gate.n_samples(radar):
        msg = (
            f"the cube's {samples.shape[1]} samples a row are not the {gate.n_samples(radar)} "
            "its range gate records; pass the gate it was synthesised through."
        )
        raise ValueError(msg)
    transmitter, sample_rate_hz = radar.transmitter, radar.receiver.sample_rate_hz
    reference = lfm_chirp(transmitter.bandwidth_hz, transmitter.chirp_duration_s, sample_rate_hz)
    start = reference.size - 1
    if gate.is_whole_pri(radar):
        stream = matched_filter(samples.reshape(-1), reference)
        return stream[start : start + samples.size].reshape(samples.shape)
    return matched_filter(samples, reference, axis=1)[:, start : start + samples.shape[1]]


@dataclass(frozen=True)
class CompensatedPulses:
    """``[migration]`` applied to ``data_rfst``: the result, and each step that ran or ``None``."""

    data_rfst: NDArray[np.complex128]
    keystone: KeystoneResult | None
    fold: FoldFactorSearch | None
    rate: DopplerRateCompensation | None

    @property
    def fold_factor(self) -> int:
        """The fold factor the keystone used: searched, or 0."""
        return 0 if self.fold is None else self.fold.fold_factor

    @property
    def acceleration_mps2(self) -> float:
        """The bistatic acceleration the Doppler-rate filter removed, m/s²: searched, or 0."""
        return 0.0 if self.rate is None else self.rate.acceleration_mps2


def compensate_migration(
    data_rfst: ArrayLike,
    radar: Radar,
    *,
    method: Literal["none", "keystone", "keystone_doppler_rate"] = "none",
    keystone: KeystoneMethod = "czt",
    fold_factor_max: int = 0,
    accelerations_mps2: ArrayLike = (),
) -> CompensatedPulses:
    """Apply a ``[migration]`` table's fields: keystone, its fold-factor search, Doppler rate.

    ``method`` ``"keystone"`` keystones, searching the fold factor when ``fold_factor_max`` is
    positive; ``"keystone_doppler_rate"`` then searches ``accelerations_mps2`` with the
    Doppler-rate filter. ``keystone`` names the interpolation.
    """
    data = np.asarray(data_rfst, dtype=np.complex128)
    if method == "none":
        return CompensatedPulses(data, None, None, None)
    transmitter, sample_rate_hz = radar.transmitter, radar.receiver.sample_rate_hz
    carrier_hz = transmitter.carrier_frequency_hz
    keystoned, fold = None, None
    if fold_factor_max > 0:
        fold = fold_factor_search(
            data,
            carrier_frequency_hz=carrier_hz,
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=transmitter.bandwidth_hz,
            fold_factor_max=fold_factor_max,
            method=keystone,
        )
        data = fold.data_rfst
    else:
        keystoned = _keystone(
            data, carrier_frequency_hz=carrier_hz, sample_rate_hz=sample_rate_hz, method=keystone
        )
        data = keystoned.data_rfst
    rate = None
    if method == "keystone_doppler_rate":
        rate = doppler_rate_compensation(
            data,
            carrier_frequency_hz=carrier_hz,
            sample_rate_hz=sample_rate_hz,
            prf_hz=transmitter.prf_hz,
            accelerations_mps2=accelerations_mps2,
        )
        data = rate.data_rfst
    return CompensatedPulses(data, keystoned, fold, rate)


def _cube(data_rfst: ArrayLike) -> NDArray[np.complex128]:
    """Return ``data_rfst`` as complex, refusing fewer than two axes or an empty one."""
    data = np.asarray(data_rfst, dtype=np.complex128)
    if data.ndim < 2 or min(data.shape[:2]) < 1:
        msg = f"data_rfst must be (n_pulses, n_range[, n_rx]); got shape {data.shape}."
        raise ValueError(msg)
    return data


def _expand(values: NDArray[Any], ndim: int) -> NDArray[Any]:
    """Append unit axes so a ``(n_pulses, n_range_fft)`` factor meets trailing channels."""
    return values.reshape(values.shape + (1,) * (ndim - values.ndim))


def _keystone_scale(
    range_frequency_hz: NDArray[np.float64], carrier_frequency_hz: float
) -> NDArray[np.float64]:
    r"""Return :math:`f_0/(f_0 + f)`, refusing a radio frequency at or below zero."""
    radio_hz = carrier_frequency_hz + range_frequency_hz
    if not np.all(radio_hz > 0.0):
        msg = "carrier_frequency_hz must exceed half the sample rate: f0 + f must stay positive."
        raise ValueError(msg)
    result: NDArray[np.float64] = carrier_frequency_hz / radio_hz
    return result


def _fold_phase(scale: NDArray[np.float64], n_pulses: int) -> NDArray[np.complex128]:
    r"""Return one fold factor's phase :math:`e^{+j 2\pi m s(f)}`, ``(n_range_fft, n_pulses)``."""
    pulse = np.arange(n_pulses, dtype=np.float64)
    result: NDArray[np.complex128] = np.exp(2j * np.pi * scale[:, None] * pulse)
    return result


def _to_rfst(by_frequency: NDArray[np.complex128], n_range: int) -> NDArray[np.complex128]:
    """Return a ``(n_range_fft, [n_rx,] n_pulses)`` spectrum as ``data_rfst``, cut to range."""
    profiles = np.fft.ifft(by_frequency, axis=0)[:n_range]
    return np.moveaxis(profiles, (0, -1), (1, 0))


def _scored_peak(by_frequency: NDArray[np.complex128]) -> tuple[float, float]:
    """Return a spectrum's map peak and Doppler, cycles per pulse, as :func:`integrated_peak`."""
    n_rows, n_pulses = by_frequency.shape[0], by_frequency.shape[-1]
    image = np.abs(np.fft.ifft(np.fft.fft(by_frequency, axis=-1), axis=0))
    power = image if image.ndim == 2 else np.sum(image**2, axis=tuple(range(1, image.ndim - 1)))
    range_bin, doppler_bin = np.unravel_index(int(np.argmax(power)), power.shape)
    steering = np.exp(2j * np.pi * np.arange(n_rows) * range_bin / n_rows) / n_rows
    pulses = np.tensordot(steering, by_frequency, axes=(0, 0)).reshape(-1, n_pulses)
    strongest = int(np.argmax(np.sum(np.abs(pulses) ** 2, axis=1)))
    cycles = aboutanios_mulgrew(pulses[strongest], float(doppler_bin), _PEAK_ITERATIONS) / n_pulses
    focused = pulses @ np.exp(-2j * np.pi * cycles * np.arange(n_pulses))
    return float(np.sum(np.abs(focused) ** 2)), float(cycles)


def _rate_phase_rad(
    range_frequency_hz: NDArray[np.float64],
    carrier_frequency_hz: float,
    prf_hz: float,
    n_pulses: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Return :math:`H`'s phase per unit acceleration and per unit jerk, ``(n_pulses, n_f)``."""
    tau_s = (np.arange(n_pulses, dtype=np.float64) / prf_hz)[:, None]
    scale = _keystone_scale(range_frequency_hz, carrier_frequency_hz)
    wavenumber_rad_per_m = 2.0 * np.pi * carrier_frequency_hz / SPEED_OF_LIGHT_MPS
    quadratic = wavenumber_rad_per_m * scale * tau_s**2 / 2.0
    cubic = wavenumber_rad_per_m * scale**2 * tau_s**3 / 6.0
    return quadratic, cubic


def _rescale_slow_time(
    spectrum: NDArray[np.complex128], scale: NDArray[np.float64], method: KeystoneMethod
) -> NDArray[np.complex128]:
    r"""Return each range frequency's slow time read at :math:`s(f)\,m`, ``(M, [n_rx,] N)``.

    ``"czt"`` is the band-limited interpolant of the :math:`N` pulses,
    :math:`\frac1N\sum_{k=-h}^{N-1-h} S[k] e^{j 2\pi k s m/N}`, :math:`h = \lfloor N/2\rfloor`:
    with :math:`k' = k + h` a chirp-z transform along :math:`W = e^{j 2\pi s/N}` of the slow-time
    DFT shifted by :math:`h`, times :math:`e^{-j 2\pi s h m/N}`, by Bluestein's convolution
    [5]_, every frequency in one batch. Slow time is moved last, where the FFTs are contiguous,
    and left there.
    """
    if method == "sinc":
        return np.moveaxis(_sinc_rescale(np.moveaxis(spectrum, 1, 0), scale), 1, -1)
    n_pulses = spectrum.shape[0]
    pre_chirp, kernel_fft, post_chirp = _chirp_z_plan(n_pulses, scale.tobytes())
    work = np.ascontiguousarray(np.moveaxis(spectrum, (1, 0), (0, -1)))
    half = n_pulses // 2
    # Multiplying by e^{j 2 pi h n / N} before the DFT shifts its output by h bins.
    work *= np.exp(2j * np.pi * half * np.arange(n_pulses) / n_pulses)
    n_fft = kernel_fft.shape[1]
    # A block of range frequencies at a time bounds the length-L work arrays; it is in place.
    for start in range(0, work.shape[0], _PLAN_BLOCK_ROWS):
        rows = slice(start, start + _PLAN_BLOCK_ROWS)
        shifted = np.fft.fft(work[rows], axis=-1)
        shifted *= _rows(pre_chirp[rows], work.ndim)
        convolved = np.fft.fft(shifted, n=n_fft, axis=-1)
        convolved *= _rows(kernel_fft[rows], work.ndim)
        work[rows] = np.fft.ifft(convolved, axis=-1)[..., :n_pulses]
        work[rows] *= _rows(post_chirp[rows], work.ndim)
    return work


def _rows(values: NDArray[np.complex128], ndim: int) -> NDArray[np.complex128]:
    """Reshape a ``(n_range_fft, n)`` factor to meet ``(n_range_fft, [n_rx,] n)`` work arrays."""
    return values.reshape(values.shape[:1] + (1,) * (ndim - 2) + values.shape[1:])


# One plan is kept, 16 M (2N + L) bytes: 128 MiB for 1024 pulses and 2048 range frequencies.
# Every frame of a scenario keystones on one grid, and building the plan is most of the cost;
# a second grid (another scenario, or the second-order keystone) replaces it.
@lru_cache(maxsize=1)
def _chirp_z_plan(
    n_pulses: int, scale_bytes: bytes
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    r"""Return Bluestein's chirps for :math:`X[m] = \sum_n x[n] W^{nm}`, one row per frequency.

    With :math:`nm = (n^2 + m^2 - (m - n)^2)/2`, :math:`X[m] = W^{m^2/2}\sum_n x[n] W^{n^2/2}
    W^{-(m-n)^2/2}`, a linear convolution done by FFTs of length :math:`L \ge 2N - 1` (Bluestein
    [5]_). The last chirp also carries the shift's :math:`e^{-j 2\pi s h m/N}` and the
    :math:`1/N`. The kernel is built a block of rows at a time, to bound its temporaries.
    """
    scale = np.frombuffer(scale_bytes, dtype=np.float64)[:, None]
    index = np.arange(n_pulses, dtype=np.float64)
    # The chirps are complex by construction; asarray states it where the stubs cannot.
    pre_chirp = np.asarray(np.exp(1j * np.pi * scale * index**2 / n_pulses), dtype=np.complex128)
    n_fft = int(2 ** np.ceil(np.log2(2 * n_pulses - 1)))
    lag = np.arange(n_fft, dtype=np.float64)
    lag_squared = np.where(lag < n_pulses, lag, lag - n_fft) ** 2
    kernel_fft: NDArray[np.complex128] = np.empty((scale.size, n_fft), dtype=np.complex128)
    # A block of rows at a time: the kernel's temporaries stay a block's size.
    for start in range(0, scale.size, _PLAN_BLOCK_ROWS):
        rows = slice(start, start + _PLAN_BLOCK_ROWS)
        kernel = np.exp(-1j * np.pi * scale[rows] * lag_squared / n_pulses)
        kernel[:, n_pulses : n_fft - n_pulses + 1] = 0.0
        kernel_fft[rows] = np.fft.fft(kernel, axis=1)
    half = n_pulses // 2
    shift = np.exp(-2j * np.pi * scale * half * index / n_pulses)
    post_chirp = np.asarray(pre_chirp * shift / n_pulses, dtype=np.complex128)
    return pre_chirp, kernel_fft, post_chirp


def _sinc_rescale(
    by_frequency: NDArray[np.complex128], scale: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Return a Hann-tapered sinc interpolation at ``scale * m``, zero beyond the dwell."""
    n_pulses = by_frequency.shape[1]
    position = scale[:, None] * np.arange(n_pulses, dtype=np.float64)
    base = np.floor(position).astype(np.int64)
    result = np.zeros_like(by_frequency)
    # One pass per kernel tap, a fixed sixteen; each pass is vectorised over the cube.
    for tap in range(1 - _SINC_HALF_TAPS, _SINC_HALF_TAPS + 1):
        index = base + tap
        offset = position - index
        weight = np.sinc(offset) * 0.5 * (1.0 + np.cos(np.pi * offset / _SINC_HALF_TAPS))
        weight = np.where((index >= 0) & (index < n_pulses), weight, 0.0)
        clipped = _expand(np.clip(index, 0, n_pulses - 1), by_frequency.ndim)
        taken = np.take_along_axis(by_frequency, clipped, axis=1)
        result += _expand(weight, by_frequency.ndim) * taken
    return result


def _doppler_spectrum(data: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return the unshifted FFT along pulses, ``(n_pulses, n_range[, n_rx])``."""
    contiguous = np.ascontiguousarray(np.moveaxis(data, 0, -1))
    spectrum: NDArray[np.complex128] = np.moveaxis(np.fft.fft(contiguous, axis=-1), -1, 0)
    return spectrum
