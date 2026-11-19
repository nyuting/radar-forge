r"""The ``leakage`` stage: scenario 005's cancellation of a noise radar's leakage and near clutter.

Each scan's cancelled stream is set beside two maps of the same scan: uncancelled, and clean,
its leakage and clutter paths taken out (same code, noise and target). After ``rd`` the scans
and their cancellations are that stage's, kept on the context; alone, the stage makes its own.
It checks :math:`\mathrm{CA}` beside its ceiling, the floor rise and SINR loss against the clean
map, and the first target's CFAR detections with and without cancellation; it writes
``leakage.csv`` and ``cancellation.png`` (scan 0). ``[verify]``, checked per scan:
``cancellation_depth_db`` with ``cancellation_tolerance_db``, ``floor_rise_db`` with
``floor_rise_tolerance_db``, ``sinr_loss_db`` with ``sinr_loss_tolerance_db``,
``detections_per_scan``, ``uncancelled_detections_per_scan``; ``acceptance_scans`` keeps the
window's first scans, those the tolerances were derived for. The floor is the mean power of the
216 cells 2 to 4 Doppler bins and 3 to 20 lags from the target's cell, off its row and column,
where a white code's sinc range response and Doppler straddle bias it. A target off the lag
window has no cell: its floor and SINR are ``nan``, which fails their checks.

References
----------
.. [1] F. Colone, C. Palmarini, T. Martelli and E. Tilli, "Sliding extensive cancellation
       algorithm for disturbance removal in passive radar," *IEEE Trans. Aerosp. Electron.
       Syst.*, vol. 52, no. 3, pp. 1309-1326, 2016.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from radar_forge import clutter, evaluate, figures, pipeline
from radar_forge.dsp import fold_into_period
from radar_forge.pipeline import RangeDopplerProduct
from radar_forge.propagation import disturbance_paths
from radar_forge.radar import TargetGeometry
from radar_forge.scenario import Scenario
from radar_forge.signal import cubic_amplifier, stream_baseband
from radar_forge.units import db_to_power, power_to_db

__all__ = ["leakage"]

# The floor window: Doppler and lag offsets from the target's cell, each side (216 cells).
_FLOOR_DOPPLER_BINS = np.array([2, 3, 4])
_FLOOR_LAG_BINS = np.arange(3, 21)
# cancellation.png zooms on the target: Doppler rows and lags drawn either side of its cell.
_FIGURE_ROWS, _FIGURE_LAGS = 32, 64
# Its colour scale over the clean floor: the uncancelled floor, 41 dB, near the top.
_FIGURE_RANGE_DB = 45.0

_KEYS = (
    "cancellation_depth_db",
    "cancellation_tolerance_db",
    "floor_rise_db",
    "floor_rise_tolerance_db",
    "sinr_loss_db",
    "sinr_loss_tolerance_db",
    "detections_per_scan",
    "uncancelled_detections_per_scan",
    "acceptance_scans",
)


@evaluate.stage("leakage", *_KEYS)
def leakage(ctx: evaluate.Context) -> evaluate.Context:
    """Cancel each scan; CA, floor rise, SINR loss and target detections; leakage.csv."""
    scenario, verify = ctx.scenario, evaluate.thresholds(ctx, "leakage")
    radar, n_pulses, n_range_lags = scenario.radar, scenario.n_pulses[0], scenario.n_range_lags[0]
    if radar.transmitter.waveform != "noise":
        msg = "the leakage stage needs a noise burst: a stream beside the code it sent."
        raise ValueError(msg)
    disturbance = disturbance_paths(radar, scenario.leakage, scenario.clutter)
    distortion = scenario.transmitter_distortion
    amplifier = None if distortion is None else functools.partial(cubic_amplifier, a3=distortion.a3)
    noise_w = radar.receiver.noise_power_per_sample_w
    n_basis, n_fit = _fit_size(scenario, n_pulses * radar.n_samples_per_chirp)
    n_scans = min(ctx.n_scans, int(verify.get("acceptance_scans", ctx.n_scans)))
    rows: list[dict[str, Any]] = []
    # Scans in order: each is measured before the next is taken.
    for scan, result, cancelled in _cancelled_scans(ctx, n_scans):
        stream, code = scan.cubes[0].reshape(-1), scan.references[0].reshape(-1)
        echo, _ = stream_baseband(
            disturbance, radar, n_pulses, rng=None, reference=code, amplifier=amplifier
        )
        maps = {
            name: pipeline.range_doppler_product(
                samples.reshape(n_pulses, -1), radar, reference=code, n_range_lags=n_range_lags
            )
            for name, samples in (("uncancelled", stream), ("clean", stream - echo))
        }
        maps["cancelled"] = cancelled
        cell = _truth_cell(maps["clean"], scan.truth)
        floor_w = {name: _floor_w(product.rd_map, cell) for name, product in maps.items()}
        sinr_db = {
            name: math.nan
            if cell is None
            else float(power_to_db(np.abs(product.rd_map[cell]) ** 2 / floor_w[name]))
            for name, product in maps.items()
        }
        # rd has detected on the cancelled map already: its plots are reused when it ran.
        cancelled_plots = ctx.plots[scan.index] if len(ctx.plots) > scan.index else None
        found = {
            "cancelled": _target_detections(
                maps["cancelled"], scenario, scan.truth, cancelled_plots
            ),
            "uncancelled": _target_detections(maps["uncancelled"], scenario, scan.truth),
        }
        row: dict[str, Any] = {"scan": scan.index, "time_s": scan.time_s}
        # The ceiling on the scan's own disturbance: paths on one route add in phase.
        row["disturbance_inr_db"] = float(power_to_db(np.mean(np.abs(echo) ** 2) / noise_w))
        row["cancellation_depth_db"] = clutter.cancellation_depth_db(stream, result.residual)
        # Through dB and back, not the ratio itself: the golden bound was made from the
        # rounded value.
        row["cancellation_depth_bound_db"] = (
            clutter.expected_cancellation_depth_db(
                float(db_to_power(row["disturbance_inr_db"])), n_basis, n_fit
            )
            if n_basis
            else 0.0
        )
        row["floor_rise_db"] = float(power_to_db(floor_w["cancelled"] / floor_w["clean"]))
        row["uncancelled_floor_rise_db"] = float(
            power_to_db(floor_w["uncancelled"] / floor_w["clean"])
        )
        row["sinr_db"], row["clean_sinr_db"] = sinr_db["cancelled"], sinr_db["clean"]
        row["sinr_loss_db"] = sinr_db["clean"] - sinr_db["cancelled"]
        row["uncancelled_sinr_db"] = sinr_db["uncancelled"]
        row["target_detections"] = found["cancelled"]
        row["uncancelled_target_detections"] = found["uncancelled"]
        row["gram_condition_number_max"] = float(np.max(result.gram_condition_number, initial=0))
        for name, value in row.items():
            if name not in ("scan", "time_s"):
                ctx.record(f"{name}[{scan.index}]", value)
        _check(ctx, verify, row)
        rows.append(row)
        if scan.index == 0 and ctx.render and cell is not None:
            _draw(ctx, maps, cell, floor_w["clean"], scan.truth)
    evaluate.write_csv(ctx.out_dir / "leakage.csv", rows)
    return ctx


def _cancelled_scans(
    ctx: evaluate.Context, n_scans: int
) -> Iterator[tuple[pipeline.ScanIq, clutter.CancellationResult, RangeDopplerProduct]]:
    """Yield the first scans, their cancellations and maps: rd's, kept on the context, or new."""
    if len(ctx.cancelled) >= n_scans:
        yield from ctx.cancelled[:n_scans]
        return
    # One scan at a time: synthesised, cancelled, mapped as rd maps it, handed out.
    for scan in pipeline.scan_iq(ctx.scenario, n_scans=n_scans):
        code, stream = scan.references[0], scan.cubes[0]
        result = clutter.cancel(code, stream, ctx.scenario.cancellation)
        yield scan, result, pipeline.scan_products(ctx.scenario, scan, cancellation=result)[0]


def _fit_size(scenario: Scenario, n_samples: int) -> tuple[int, int]:
    """Return the canceller's basis size and the samples one fit sees (the CPI if uncancelled)."""
    settings = scenario.cancellation
    if settings is None or settings.method == "none":
        return 0, n_samples
    window = settings.n_samples_per_batch + 2 * settings.n_samples_extension
    n_fit = min(window, n_samples) if settings.n_samples_per_batch else n_samples
    return settings.n_taps * len(settings.odd_orders), n_fit


def _check(ctx: evaluate.Context, verify: dict[str, Any], row: dict[str, Any]) -> None:
    """Check one scan's row against the thresholds the scenario gives."""
    for key, tolerance_key in (
        ("cancellation_depth_db", "cancellation_tolerance_db"),
        ("floor_rise_db", "floor_rise_tolerance_db"),
        ("sinr_loss_db", "sinr_loss_tolerance_db"),
    ):
        if key in verify:
            expected, tolerance = verify[key], verify.get(tolerance_key, 0.0)
            ctx.check("leakage", key, row[key], expected - tolerance, expected + tolerance)
    for key, column in (
        ("detections_per_scan", "target_detections"),
        ("uncancelled_detections_per_scan", "uncancelled_target_detections"),
    ):
        if key in verify:
            ctx.check("leakage", column, row[column], verify[key], verify[key])


def _truth_cell(product: RangeDopplerProduct, truth: TargetGeometry) -> tuple[int, int] | None:
    """Return the first target's nearest map cell, ``(Doppler row, lag)``; ``None`` off the map.

    Within half a lag of the window's end the nearest lag would be past it: off the map too.
    """
    range_index, rate_index = product.index_of(
        truth.bistatic_range_m[0], truth.bistatic_range_rate_mps[0]
    )
    n_rate, n_lags = product.rd_map.shape
    if not np.isfinite(range_index) or np.rint(range_index) >= n_lags:
        return None
    return int(np.rint(rate_index)) % n_rate, int(np.rint(range_index))


def _floor_w(rd_map: NDArray[np.complex128], cell: tuple[int, int] | None) -> float:
    """Return the mean power of the floor window about ``cell``: Doppler wraps, lags do not."""
    if cell is None:
        return math.nan
    n_rate, n_lags = rd_map.shape
    doppler = np.concatenate([-_FLOOR_DOPPLER_BINS, _FLOOR_DOPPLER_BINS])
    lags = np.concatenate([-_FLOOR_LAG_BINS, _FLOOR_LAG_BINS])
    rows = (cell[0] + doppler) % n_rate
    columns = cell[1] + lags
    columns = columns[(columns >= 0) & (columns < n_lags)]
    return float(np.mean(np.abs(rd_map[np.ix_(rows, columns)]) ** 2))


def _target_detections(
    product: RangeDopplerProduct,
    scenario: Scenario,
    truth: TargetGeometry,
    found: pipeline.Plots | None = None,
) -> int:
    """Return how many CFAR clusters lie within one bin of the first target on both axes.

    ``found`` is the map's plots when another stage has made them; else CFAR runs here.
    """
    if found is None:
        found = pipeline.scan_plots([product], scenario.detection)
    if not found.detections:
        return 0
    range_index, rate_index = product.index_of(
        truth.bistatic_range_m[0], truth.bistatic_range_rate_mps[0]
    )
    cells = np.array([d.centroid_index for d in found.detections])
    n_rate = product.rd_map.shape[0]
    rate_gap = fold_into_period(cells[:, 0] - rate_index, n_rate, lower=-n_rate / 2)
    near = (np.abs(rate_gap) <= 1.0) & (np.abs(cells[:, 1] - range_index) <= 1.0)
    return int(np.sum(near))


def _draw(
    ctx: evaluate.Context,
    maps: dict[str, RangeDopplerProduct],
    cell: tuple[int, int],
    floor_w: float,
    truth: TargetGeometry,
) -> None:
    """Draw scan 0 about the target before and after cancellation: buried, then clear."""
    product = maps["clean"]
    n_rate, n_lags = product.rd_map.shape
    rows = slice(max(0, cell[0] - _FIGURE_ROWS), min(n_rate, cell[0] + _FIGURE_ROWS + 1))
    lags = slice(max(0, cell[1] - _FIGURE_LAGS), min(n_lags, cell[1] + _FIGURE_LAGS + 1))
    figure = figures.render_cancellation(
        maps["uncancelled"].rd_map[rows, lags],
        maps["cancelled"].rd_map[rows, lags],
        product.range_axis_m[lags],
        product.rate_axis_mps[rows],
        ctx.scenario.radar,
        floor_w=floor_w,
        truth_range_m=truth.bistatic_range_m,
        truth_rate_mps=truth.bistatic_range_rate_mps,
        title=f"{ctx.scenario.name}, scan 0",
        dynamic_range_db=_FIGURE_RANGE_DB,
    )
    figures.save_figure(figure, ctx.out_dir / "cancellation.png")
