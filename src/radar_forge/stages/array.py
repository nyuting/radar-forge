r"""The ``doa`` stage: each detection's chip, its snapshot across channels, and the estimators.

Every frame is mapped per receive channel (:func:`~radar_forge.pipeline.frame_products`) and
detected on the boresight beam, as ``rd`` does. Each detection's 3-D map is cropped into a chip
(:func:`~radar_forge.datachip.crop_range_doppler`): its beam's peak cell across channels is the
snapshot, its cells the training set (scenario 006 D14). Range and rate are refined on a
slow-time chip of the beam (:func:`~radar_forge.datachip.estimate_fine`), in parent coordinates.
Each ``[doa]`` method runs on the target's detection and every other; the target's estimates are
scored against the truth, each error over the Cramér-Rao bound at the truth for that snapshot. On
noiseless frame 0 it checks the chain against the snapshot model and the coherent covariance's
rank. Writes ``doa.csv`` (``method = "truth"`` rows carry both aircraft) and ``doa.png``.

``[verify]``: ``direction_cosine_tolerance``, deterministic ML within it of both truths every
frame; ``resolved_fraction_min`` over ``resolved_methods``; ``baseline_resolved_fraction_max``
over ``baseline_methods``; ``<method>_rmse_crb_max``, RMSE over the bound's deviation (these
three judged from 50 scored frames on, the statistics their bounds were set for);
``rank_ratio_max``, :math:`\lambda_2/\lambda_1` of the noiseless chip's raw covariance;
``snapshot_model_tolerance``, the noiseless peak cell against :math:`g A(u) s`, per channel;
``parent_coordinates_tolerance``, the chip's peak cell, bins and estimate against the parent's.

References
----------
.. [1] P. Stoica and A. Nehorai, "MUSIC, maximum likelihood, and Cramer-Rao bound," *IEEE
       Trans. Acoust., Speech, Signal Process.*, vol. 37, no. 5, pp. 720-741, 1989.
"""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from radar_forge import evaluate, plots
from radar_forge.array import (
    ArrayGeometry,
    direction_cosine,
    directions_unit,
    enu_to_array_frame,
    forward_backward_average,
    rayleigh_width,
    sample_covariance,
    spatial_smoothing,
    subarray,
)
from radar_forge.datachip import (
    ChipProcessing,
    DataChip,
    FineEstimate,
    crop_range_doppler,
    crop_slow_time,
    estimate_fine,
)
from radar_forge.detection import Detection
from radar_forge.doa import (
    DOA_LABELS,
    DoaEstimate,
    crb_deterministic,
    estimate_doa,
    is_resolved,
    label_parts,
    steering_u,
)
from radar_forge.dsp import fold_velocity_mps
from radar_forge.pipeline import (
    RangeDopplerProduct,
    frame_plots,
    frame_products,
    frames,
    range_doppler_product,
    synthesise,
)
from radar_forge.radar import Radar, TargetGeometry
from radar_forge.scenario import scattering_phase_rad
from radar_forge.settings import DoaSettings

__all__ = ["doa"]

_KEYS = (
    "direction_cosine_tolerance",
    "resolved_fraction_min",
    "resolved_methods",
    "baseline_resolved_fraction_max",
    "baseline_methods",
    "rank_ratio_max",
    "snapshot_model_tolerance",
    "parent_coordinates_tolerance",
    *(f"{label}_rmse_crb_max" for label in DOA_LABELS),
)
# Methods that train on the chip's cells; the rest take its peak cell (D14).
_TRAINED = ("capon", "music", "root_music", "esprit")
# A detection is the target's when its peak cell is within this many bins of the truth's.
_TARGET_BINS = 1.5
# Resolved fractions and RMSE over the bound are judged on this many frames or more; the bounds
# were set on 200 trials (spec A8-A11, A14). Over n errors an efficient estimator's ratio is
# sqrt(chi2_n / n), past 1.3 with probability 0.12 at three frames and 2e-5 at fifty; a method
# resolving with P = 0.8 misses a 0.65 bound with probability 0.10 at three, 0.006 at fifty.
_MIN_SCORED_FRAMES = 50


@evaluate.stage("doa", *_KEYS)
def doa(ctx: evaluate.Context) -> evaluate.Context:
    """Every configured estimator on every detection's chip; scores, checks, doa.csv, doa.png."""
    scenario, verify = ctx.scenario, evaluate.thresholds(ctx, "doa")
    settings, array = scenario.doa, scenario.array
    if settings is None or array is None:
        msg = "the doa stage needs [receive_array] and [doa] tables."
        raise ValueError(msg)
    radar = scenario.radar
    second = scenario.second_aircraft
    separation = (second.separation_rayleigh if second else 1.0) * rayleigh_width(
        array, radar.wavelength_m
    )
    resolved: dict[str, list[bool]] = {label: [] for label in settings.methods}
    scaled: dict[str, list[float]] = {label: [] for label in settings.methods}
    worst: dict[str, float] = {"ml_error": 0.0, "parent": 0.0}
    n_targets_found = 0
    rows: list[dict[str, Any]] = []
    for frame in frames(scenario, n_frames=ctx.n_frames):
        product = frame_products(scenario, frame)[0]
        found = frame_plots([product], scenario.detection)
        truth_u = direction_cosine(array, frame.truth.aoa_rad)
        target = _target(found.detections, product, frame.truth)
        rows += _truth_rows(ctx, frame.index, frame.truth, truth_u, array)
        print(f"  frame {frame.index}: u_true = {np.round(truth_u, 6)}, target detection {target}")
        # Each detection's chip and estimators: a handful of detections a frame.
        for j, detection in enumerate(found.detections):
            chip, cell = _chip(product, detection, settings, radar, scenario.n_pulses[0])
            fine = _fine(product, int(chip.parent_index(*cell)[1]), radar, scenario.n_pulses[0])
            snapshot = chip.data[cell]
            training = chip.data.reshape(-1, array.n_rx)
            estimates = {
                label: _estimate(label, snapshot, training, array, settings, ctx)
                for label in settings.methods
            }
            rows += _estimate_rows(ctx, frame.index, j, chip, cell, fine, estimates, array)
            if j != target:
                continue
            n_targets_found += 1
            truths = np.sort(truth_u[[0, -1]])
            _ratios(ctx, training, array, settings, frame.index)
            bound = _bound_at_truth(snapshot, truths, array, ctx)
            for label, estimate in estimates.items():
                found_u = estimate.direction_cosine_x
                resolved[label].append(is_resolved(found_u, truths, separation))
                scaled[label] += list(((found_u - truths) / bound) ** 2)
                print(f"    {label}: u = {np.round(found_u, 6)}, resolved {resolved[label][-1]}")
            if "deterministic_ml" in estimates:
                error = np.abs(estimates["deterministic_ml"].direction_cosine_x - truths)
                worst["ml_error"] = max(worst["ml_error"], float(error.max()))
            first = next(iter(estimates.items()))
            worst["parent"] = max(worst["parent"], _parent_error(product, chip, cell, first, ctx))
            if frame.index == 0 and ctx.render:
                _figure(ctx, estimates, truths)
        # A frame whose target went undetected resolves nothing.
        for label in settings.methods:
            resolved[label] += [False] * (frame.index + 1 - len(resolved[label]))
    if n_targets_found < ctx.n_frames:
        # A frame without the target's detection has no estimate: no tolerance holds.
        worst["ml_error"] = worst["parent"] = math.inf
    worst["n_scored"] = n_targets_found
    _check(ctx, verify, resolved, scaled, worst)
    if {"rank_ratio_max", "snapshot_model_tolerance"} & set(verify):
        _noiseless(ctx, verify, array, settings)
    evaluate.write_csv(ctx.out_dir / "doa.csv", rows)
    return ctx


def _processing(radar: Radar, noise_power_w: float) -> ChipProcessing:
    """Return what a chip of an FMCW map needs of its parent: the radar and a cell's noise."""
    transmitter = radar.transmitter
    return ChipProcessing(
        carrier_frequency_hz=transmitter.carrier_frequency_hz,
        bandwidth_hz=transmitter.bandwidth_hz,
        sample_rate_hz=radar.receiver.sample_rate_hz,
        prf_hz=transmitter.prf_hz,
        noise_power_w=noise_power_w,
    )


def _cell_noise_w(radar: Radar, n_pulses: int) -> float:
    """Return one channel's noise in one map cell: per sample, times both unnormalised FFTs."""
    return radar.receiver.noise_power_per_sample_w * radar.n_samples_per_pri * n_pulses


def _chip(
    product: RangeDopplerProduct,
    detection: Detection,
    settings: DoaSettings,
    radar: Radar,
    n_pulses: int,
) -> tuple[DataChip, tuple[int, int]]:
    """Crop the 3-D map about a detection; return the chip and its beam's peak cell."""
    if product.channels is None:
        msg = "the doa stage needs a map per receive channel: a scenario with [receive_array]."
        raise ValueError(msg)
    rate_bin, range_bin = detection.peak_index
    chip = crop_range_doppler(
        product.channels,
        rate_bin,
        range_bin,
        range_bin_centers_m=product.range_axis_m,
        rate_bin_centers_mps=product.rate_axis_mps,
        processing=_processing(radar, _cell_noise_w(radar, n_pulses)),
        half_width_bins=settings.half_width_bins,
    )
    beam_w = np.abs(chip.data.mean(axis=-1)) ** 2
    row, column = np.unravel_index(int(np.argmax(beam_w)), beam_w.shape)
    return chip, (int(row), int(column))


def _fine(
    product: RangeDopplerProduct, centre_range_bin: int, radar: Radar, n_pulses: int
) -> FineEstimate:
    """Refine range and rate on a slow-time chip of the beam, in parent coordinates.

    The beam's range profiles are its map's inverse Doppler transform. An FMCW profile's range
    spectrum is the sweep's fast time, one-sided on :math:`[0, T)`, while ``estimate_fine``
    reads a pulse's band centred on zero; :math:`(-1)^k` on bin :math:`k` centres it on
    mid-sweep, so the chip reads as a pulsed one does. The range is the dwell's mean.
    """
    profiles = np.fft.ifft(np.fft.ifftshift(product.rd_map, axes=0), axis=0)
    centred = profiles * (-1.0) ** np.arange(profiles.shape[1])
    # The beam averages n_rx channels of a range FFT of n_samples: noise per profile cell.
    n_rx = 1 if product.channels is None else product.channels.shape[-1]
    noise_w = radar.receiver.noise_power_per_sample_w * radar.n_samples_per_pri / n_rx
    chip = crop_slow_time(
        centred,
        centre_range_bin,
        range_bin_centers_m=product.range_axis_m,
        rate_bin_centers_mps=product.rate_axis_mps,
        processing=_processing(radar, noise_w),
    )
    return estimate_fine(chip)


def _estimate(
    label: str,
    snapshot: NDArray[np.complex128],
    training: NDArray[np.complex128],
    array: ArrayGeometry,
    settings: DoaSettings,
    ctx: evaluate.Context,
) -> DoaEstimate:
    """Run one ``[doa]`` method on a chip: trained on its cells, or on its peak cell."""
    method, kind = label_parts(label)
    radar = ctx.scenario.radar
    iterations = {"iaa": settings.n_iaa_iterations, "spice": settings.n_spice_iterations}
    return estimate_doa(
        snapshot,
        array,
        settings.n_sources,
        wavelength_m=radar.wavelength_m,
        method=method,
        covariance=kind,
        training=training if method in _TRAINED else None,
        n_subarray_elements=settings.n_subarray_elements,
        n_grid=settings.n_grid,
        n_iterations=iterations.get(method),
        noise_power_w=_cell_noise_w(radar, ctx.scenario.n_pulses[0]),
    )


def _target(
    detections: tuple[Detection, ...], product: RangeDopplerProduct, truth: TargetGeometry
) -> int | None:
    """Return the detection whose peak cell lies nearest the first target's, within reach."""
    range_index, rate_index = product.index_of(
        truth.bistatic_range_m[0], truth.bistatic_range_rate_mps[0]
    )
    n_rate, n_range = product.rd_map.shape
    gaps = [
        max(
            abs((d.peak_index[0] - float(rate_index) + n_rate / 2) % n_rate - n_rate / 2),
            abs((d.peak_index[1] - float(range_index) + n_range / 2) % n_range - n_range / 2),
        )
        for d in detections
    ]
    if not gaps or min(gaps) > _TARGET_BINS:
        return None
    return int(np.argmin(gaps))


def _bound_at_truth(
    snapshot: NDArray[np.complex128],
    truths: NDArray[np.float64],
    array: ArrayGeometry,
    ctx: evaluate.Context,
) -> NDArray[np.float64]:
    """Return the bound's deviation at the truth, the amplitudes fitted to the snapshot there."""
    radar = ctx.scenario.radar
    a = steering_u(array, truths, radar.wavelength_m)
    amplitude, *_ = np.linalg.lstsq(a, snapshot, rcond=None)
    x_m = array.element_positions_m[:, 0]
    derivative = 2j * np.pi * x_m[:, None] / radar.wavelength_m * a
    noise_w = _cell_noise_w(radar, ctx.scenario.n_pulses[0])
    bound = crb_deterministic(a, derivative, amplitude, noise_w)
    for k, s in enumerate(amplitude):
        ctx.record(f"source_{k}_cell_snr_db", evaluate.to_db(abs(s) ** 2 / noise_w))
    result: NDArray[np.float64] = np.sqrt(np.diag(bound))
    return result


def _ratios(
    ctx: evaluate.Context,
    training: NDArray[np.complex128],
    array: ArrayGeometry,
    settings: DoaSettings,
    index: int,
) -> None:
    """Print, and on frame 0 record, lambda_2/lambda_1 of the raw, FB and FBSS covariances."""
    raw = sample_covariance(training)
    length = settings.n_subarray_elements
    for name, covariance in (
        ("raw", raw),
        ("fb", forward_backward_average(raw, array)),
        ("fbss", forward_backward_average(spatial_smoothing(raw, length), subarray(array, length))),
    ):
        values = np.linalg.eigvalsh(covariance)[::-1]
        ratio = float(values[1] / values[0])
        if index == 0:
            ctx.record(f"eigenvalue_ratio_{name}", ratio)
        else:
            print(f"    eigenvalue_ratio_{name} = {ratio:.6g}")


def _parent_error(
    product: RangeDopplerProduct,
    chip: DataChip,
    cell: tuple[int, int],
    estimate: tuple[str, DoaEstimate],
    ctx: evaluate.Context,
) -> float:
    """Return the largest relative gap between the chip's reading and the parent's own (A16).

    The chip's peak cell in parent coordinates against the parent's beam peak over the same
    cells; the bin values, the cells themselves; and the first method's estimate on the
    parent's own snapshot and cells.
    """
    settings, array = ctx.scenario.doa, ctx.scenario.array
    if product.channels is None or settings is None or array is None:
        return math.inf
    rate_bin, range_bin = (int(i) for i in chip.parent_index(*cell))
    rows, columns = np.broadcast_arrays(
        *chip.parent_index(
            np.arange(chip.data.shape[0])[:, None], np.arange(chip.data.shape[1])[None, :]
        )
    )
    beam_w = np.abs(product.rd_map[rows, columns]) ** 2
    peak = np.unravel_index(int(np.argmax(beam_w)), beam_w.shape)
    parent_cells = product.channels[rows, columns]
    label, found = estimate
    again = _estimate(
        label,
        product.channels[rate_bin, range_bin],
        parent_cells.reshape(-1, array.n_rx),
        array,
        settings,
        ctx,
    )
    return max(
        float(rows[peak] != rate_bin) + float(columns[peak] != range_bin),
        _relative(chip.range_bin_centers_m[cell[1]], product.range_axis_m[range_bin]),
        _relative(chip.rate_bin_centers_mps[cell[0]], product.rate_axis_mps[rate_bin]),
        _relative(chip.data, parent_cells),
        _relative(found.direction_cosine_x, again.direction_cosine_x),
    )


def _relative(value: Any, reference: Any) -> float:
    """Return the largest absolute difference over the largest reference magnitude."""
    scale = max(float(np.max(np.abs(reference))), 1e-300)
    return float(np.max(np.abs(np.asarray(value) - np.asarray(reference)))) / scale


def _truth_rows(
    ctx: evaluate.Context,
    index: int,
    truth: TargetGeometry,
    truth_u: NDArray[np.float64],
    array: ArrayGeometry,
) -> list[dict[str, Any]]:
    """Return doa.csv's truth rows: every aircraft's true direction, range, rate and cell SNR."""
    scenario = ctx.scenario
    radar, show = scenario.radar, plots.display_axes(scenario.radar)
    power_w = evaluate.target_power_w(radar, truth, scenario.rcs_m2)
    snr_db = 10.0 * np.log10(power_w / _cell_noise_w(radar, scenario.n_pulses[0]))
    snr_db += 10.0 * np.log10(radar.n_samples_per_chirp * scenario.n_pulses[0])
    in_array = enu_to_array_frame(array, directions_unit(truth.aoa_rad[:, 0], truth.aoa_rad[:, 1]))
    off_boresight_deg = np.degrees(np.arctan2(in_array[:, 0], in_array[:, 1]))
    rows = []
    # One row per aircraft, sorted by direction cosine as the estimates are.
    for source, k in enumerate(np.argsort(truth_u)):
        rows.append(
            {
                "frame": index,
                "time_s": scenario.frame_times_s[index],
                "detection_id": "",
                "method": "truth",
                "source_index": source,
                "direction_cosine_x": truth_u[k],
                "azimuth_boresight_deg": off_boresight_deg[k],
                "azimuth_deg": math.degrees(truth.aoa_rad[k, 0]),
                show.range_name: truth.bistatic_range_m[k] * show.scale,
                show.rate_name + "_folded": truth.bistatic_range_rate_mps[k] * show.scale,
                "range_bin": "",
                "doppler_bin": "",
                "snr_db": snr_db[k],
                "crb_std_direction_cosine": math.nan,
            }
        )
    return rows


def _estimate_rows(
    ctx: evaluate.Context,
    index: int,
    detection_id: int,
    chip: DataChip,
    cell: tuple[int, int],
    fine: FineEstimate,
    estimates: dict[str, DoaEstimate],
    array: ArrayGeometry,
) -> list[dict[str, Any]]:
    """Return doa.csv's rows for one detection: a row per method per source."""
    scenario = ctx.scenario
    radar, show = scenario.radar, plots.display_axes(scenario.radar)
    rate_bin, range_bin = (int(i) for i in chip.parent_index(*cell))
    noise_w = _cell_noise_w(radar, scenario.n_pulses[0])
    rate_mps = float(
        fold_velocity_mps(fine.bistatic_range_rate_mps, radar.unambiguous_bistatic_range_rate_mps)
    )
    rows = []
    # A row per method and source: a few dozen a detection.
    for label, estimate in estimates.items():
        for source, u in enumerate(estimate.direction_cosine_x):
            off_boresight_deg = math.degrees(math.asin(min(1.0, max(-1.0, float(u)))))
            rows.append(
                {
                    "frame": index,
                    "time_s": scenario.frame_times_s[index],
                    "detection_id": detection_id,
                    "method": label,
                    "source_index": source,
                    "direction_cosine_x": float(u),
                    "azimuth_boresight_deg": off_boresight_deg,
                    "azimuth_deg": (array.boresight_azimuth_deg + off_boresight_deg) % 360.0,
                    show.range_name: fine.bistatic_range_m * show.scale,
                    show.rate_name + "_folded": rate_mps * show.scale,
                    "range_bin": range_bin,
                    "doppler_bin": rate_bin,
                    "snr_db": evaluate.to_db(abs(estimate.amplitude[source]) ** 2 / noise_w),
                    "crb_std_direction_cosine": float(estimate.crb_std_direction_cosine[source]),
                }
            )
    return rows


def _check(
    ctx: evaluate.Context,
    verify: dict[str, Any],
    resolved: dict[str, list[bool]],
    scaled: dict[str, list[float]],
    worst: dict[str, float],
) -> None:
    """Record each method's resolved fraction and RMSE over the bound; check the thresholds."""
    fraction, rmse = {}, {}
    for label in resolved:
        fraction[label] = ctx.record(f"{label}_resolved_fraction", float(np.mean(resolved[label])))
        mean_square = float(np.mean(scaled[label])) if scaled[label] else math.nan
        rmse[label] = ctx.record(f"{label}_rmse_crb", math.sqrt(mean_square))
    ml_error = ctx.record("ml_direction_cosine_error_max", worst["ml_error"])
    parent = ctx.record("parent_coordinates_error", worst["parent"])
    if "direction_cosine_tolerance" in verify:
        tolerance = verify["direction_cosine_tolerance"]
        ctx.check("doa", "ml_direction_cosine_error_max", ml_error, 0.0, tolerance)
    if "parent_coordinates_tolerance" in verify:
        tolerance = verify["parent_coordinates_tolerance"]
        ctx.check("doa", "parent_coordinates_error", parent, 0.0, tolerance)
    if worst["n_scored"] < _MIN_SCORED_FRAMES:
        print(f"  under {_MIN_SCORED_FRAMES} scored frames: too few to judge fractions or RMSE")
        return
    for label in verify.get("resolved_methods", ()):
        low = verify.get("resolved_fraction_min", 1.0)
        ctx.check("doa", f"{label}_resolved_fraction", fraction.get(label, math.nan), low, 1.0)
    for label in verify.get("baseline_methods", ()):
        high = verify.get("baseline_resolved_fraction_max", 0.0)
        ctx.check("doa", f"{label}_resolved_fraction", fraction.get(label, math.nan), 0.0, high)
    for label in [label for label in DOA_LABELS if f"{label}_rmse_crb_max" in verify]:
        high = verify[f"{label}_rmse_crb_max"]
        ctx.check("doa", f"{label}_rmse_crb", rmse.get(label, math.nan), 0.0, high)


def _row(truth: TargetGeometry, k: int) -> TargetGeometry:
    """Return one truth row as a geometry of its own."""
    return TargetGeometry(**{f.name: getattr(truth, f.name)[k : k + 1] for f in fields(truth)})


def _noiseless(
    ctx: evaluate.Context, verify: dict[str, Any], array: ArrayGeometry, settings: DoaSettings
) -> None:
    """On noiseless frame 0: the peak cell against the snapshot model (A13), the rank (A7).

    The model's :math:`g s_k` is aircraft k's own one-channel map at the peak cell, its
    steering vector the truth's: nothing of the multi-channel chain enters the prediction.
    """
    scenario, truth = ctx.scenario, evaluate.truth(ctx)[0]
    radar, n_pulses, rcs_m2 = scenario.radar, scenario.n_pulses[0], scenario.rcs_m2
    phase_rad = scattering_phase_rad(scenario)[0]
    clean = synthesise(
        radar, truth, rcs_m2, n_pulses, rng=None, array=array, scattering_phase_rad=phase_rad
    )
    product = range_doppler_product(clean, radar)
    if product.channels is None:
        msg = "the noiseless frame has no channel axis."
        raise ValueError(msg)
    cell = np.unravel_index(int(np.argmax(np.abs(product.rd_map))), product.rd_map.shape)
    rate_bin, range_bin = int(cell[0]), int(cell[1])
    alone = [
        range_doppler_product(
            synthesise(
                radar,
                _row(truth, k),
                rcs_m2[k : k + 1],
                n_pulses,
                rng=None,
                scattering_phase_rad=phase_rad[k],
            ),
            radar,
        ).rd_map[rate_bin, range_bin]
        for k in range(len(rcs_m2))
    ]
    truth_u = direction_cosine(array, truth.aoa_rad)
    predicted = steering_u(array, truth_u, radar.wavelength_m) @ np.array(alone)
    snapshot = product.channels[rate_bin, range_bin]
    error = ctx.record(
        "snapshot_model_error", float(np.max(np.abs(snapshot - predicted) / np.abs(predicted)))
    )
    chip = crop_range_doppler(
        product.channels,
        rate_bin,
        range_bin,
        range_bin_centers_m=product.range_axis_m,
        rate_bin_centers_mps=product.rate_axis_mps,
        processing=_processing(radar, _cell_noise_w(radar, n_pulses)),
        half_width_bins=settings.half_width_bins,
    )
    values = np.linalg.eigvalsh(sample_covariance(chip.data.reshape(-1, array.n_rx)))[::-1]
    ratio = ctx.record("noiseless_eigenvalue_ratio_raw", float(values[1] / values[0]))
    if "snapshot_model_tolerance" in verify:
        ctx.check("doa", "snapshot_model_error", error, 0.0, verify["snapshot_model_tolerance"])
    if "rank_ratio_max" in verify:
        ctx.check("doa", "noiseless_eigenvalue_ratio_raw", ratio, 0.0, verify["rank_ratio_max"])


def _figure(
    ctx: evaluate.Context, estimates: dict[str, DoaEstimate], truths: NDArray[np.float64]
) -> None:
    """Draw frame 0's spectra of the target's detection, the estimates without one as markers."""
    spectra = [(label, e.spectrum) for label, e in estimates.items() if e.spectrum.size]
    grid = next((e.grid for e in estimates.values() if e.grid.size), np.zeros(0))
    markers = [(label, e.direction_cosine_x) for label, e in estimates.items() if not e.grid.size]
    figure = plots.render_doa_spectra(
        grid,
        spectra,
        estimates=markers,
        truth_direction_cosine=truths,
        title=f"{ctx.scenario.name}, frame 0, the target's detection",
    )
    plots.save_figure(figure, ctx.out_dir / "doa.png")
