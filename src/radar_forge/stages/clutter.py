r"""The ``leakage`` stage: scenario 005's cancellation of a noise radar's leakage and near clutter.

Each frame's stream is cancelled as ``[cancellation]`` says and set beside two maps of the same
frame: the stream uncancelled, and the clean stream, its leakage and clutter paths taken out
(the same code, noise and target). The stage prints and checks the cancellation
:math:`\mathrm{CA}` against its ceiling, the floor rise and the SINR loss against the clean
map, and the CFAR detections of the first target with and without cancellation; it writes
``leakage.csv`` and ``cancellation.png`` (frame 0). ``[verify]`` keys, each checked per frame:
``cancellation_depth_db`` with ``cancellation_tolerance_db``, ``floor_rise_db`` with
``floor_rise_tolerance_db``, ``sinr_loss_db`` with ``sinr_loss_tolerance_db``,
``detections_per_frame`` and ``uncancelled_detections_per_frame``. The floor is the mean power
of the 216 cells 2 to 4 Doppler bins and 3 to 20 lags from the target's cell, off its own row
and column, where a white code's sinc range response and Doppler straddle would bias it.

References
----------
.. [1] F. Colone, C. Palmarini, T. Martelli and E. Tilli, "Sliding extensive cancellation
       algorithm for disturbance removal in passive radar," *IEEE Trans. Aerosp. Electron.
       Syst.*, vol. 52, no. 3, pp. 1309-1326, 2016.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from numpy.typing import NDArray

from radar_forge import clutter, evaluate, pipeline, plots
from radar_forge.dsp import fold_into_period
from radar_forge.pipeline import RangeDopplerProduct
from radar_forge.radar import TargetGeometry
from radar_forge.scenario import Scenario
from radar_forge.signal import cubic_amplifier, stream_baseband

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
    "detections_per_frame",
    "uncancelled_detections_per_frame",
)


@evaluate.stage("leakage", *_KEYS)
def leakage(ctx: evaluate.Context) -> evaluate.Context:
    """Cancel each frame; CA, floor rise, SINR loss and target detections; leakage.csv."""
    scenario, verify = ctx.scenario, evaluate.thresholds(ctx, "leakage")
    radar, n_pulses, n_range_lags = scenario.radar, scenario.n_pulses[0], scenario.n_range_lags[0]
    if radar.transmitter.waveform != "noise":
        msg = "the leakage stage needs a noise burst: a stream beside the code it sent."
        raise ValueError(msg)
    disturbance = pipeline.disturbance_paths(scenario, radar)
    distortion = scenario.transmitter_distortion
    amplifier = None if distortion is None else functools.partial(cubic_amplifier, a3=distortion.a3)
    noise_w = radar.receiver.noise_power_per_sample_w
    inr_linear = float(np.sum(np.abs(disturbance.amplitude_linear) ** 2) / noise_w)
    n_basis, n_fit = _fit_size(scenario, n_pulses * radar.n_samples_per_chirp)
    ctx.record("disturbance_inr_db", evaluate.to_db(inr_linear))
    bound_db = ctx.record(
        "cancellation_depth_bound_db",
        clutter.expected_cancellation_depth_db(inr_linear, n_basis, n_fit),
    )
    rows: list[dict[str, Any]] = []
    # Frames in order: each is synthesised, cancelled and measured before the next.
    for frame in pipeline.frames(scenario, n_frames=ctx.n_frames):
        stream, code = frame.cubes[0].reshape(-1), frame.references[0].reshape(-1)
        echo, _ = stream_baseband(
            disturbance, radar, n_pulses, rng=None, reference=code, amplifier=amplifier
        )
        result = pipeline.cancel(code, stream, scenario.cancellation)
        maps = {
            name: pipeline.range_doppler_product(
                samples.reshape(n_pulses, -1), radar, reference=code, n_range_lags=n_range_lags
            )
            for name, samples in (
                ("uncancelled", stream),
                ("cancelled", result.residual),
                ("clean", stream - echo),
            )
        }
        cell = _truth_cell(maps["clean"], frame.truth)
        floor_w = {name: _floor_w(product.rd_map, cell) for name, product in maps.items()}
        sinr_db = {
            name: evaluate.to_db(np.abs(product.rd_map[cell]) ** 2 / floor_w[name])
            for name, product in maps.items()
        }
        found = {
            name: _target_detections(maps[name], scenario, frame.truth)
            for name in ("cancelled", "uncancelled")
        }
        row: dict[str, Any] = {"frame": frame.index, "time_s": frame.time_s}
        row["cancellation_depth_db"] = clutter.cancellation_depth_db(stream, result.residual)
        row["cancellation_depth_bound_db"] = bound_db
        row["floor_rise_db"] = evaluate.to_db(floor_w["cancelled"] / floor_w["clean"])
        row["uncancelled_floor_rise_db"] = evaluate.to_db(floor_w["uncancelled"] / floor_w["clean"])
        row["sinr_db"], row["clean_sinr_db"] = sinr_db["cancelled"], sinr_db["clean"]
        row["sinr_loss_db"] = sinr_db["clean"] - sinr_db["cancelled"]
        row["uncancelled_sinr_db"] = sinr_db["uncancelled"]
        row["target_detections"] = found["cancelled"]
        row["uncancelled_target_detections"] = found["uncancelled"]
        row["gram_condition_number_max"] = float(np.max(result.gram_condition_number, initial=0))
        for name, value in row.items():
            if name not in ("frame", "time_s", "cancellation_depth_bound_db"):
                ctx.record(f"{name}[{frame.index}]", value)
        _check(ctx, verify, row)
        rows.append(row)
        if frame.index == 0 and ctx.render:
            _draw(ctx, maps, cell, floor_w["clean"], frame.truth)
    evaluate.write_csv(ctx.out_dir / "leakage.csv", rows)
    return ctx


def _fit_size(scenario: Scenario, n_samples: int) -> tuple[int, int]:
    """Return the canceller's basis size and the samples one fit sees (the CPI if uncancelled)."""
    settings = scenario.cancellation
    if settings is None or settings.method == "none":
        return 0, n_samples
    window = settings.n_samples_per_batch + 2 * settings.n_samples_extension
    n_fit = min(window, n_samples) if settings.n_samples_per_batch else n_samples
    return settings.n_taps * len(settings.odd_orders), n_fit


def _check(ctx: evaluate.Context, verify: dict[str, Any], row: dict[str, Any]) -> None:
    """Check one frame's row against the thresholds the scenario gives."""
    for key, tolerance_key in (
        ("cancellation_depth_db", "cancellation_tolerance_db"),
        ("floor_rise_db", "floor_rise_tolerance_db"),
        ("sinr_loss_db", "sinr_loss_tolerance_db"),
    ):
        if key in verify:
            expected, tolerance = verify[key], verify.get(tolerance_key, 0.0)
            ctx.check("leakage", key, row[key], expected - tolerance, expected + tolerance)
    for key, column in (
        ("detections_per_frame", "target_detections"),
        ("uncancelled_detections_per_frame", "uncancelled_target_detections"),
    ):
        if key in verify:
            ctx.check("leakage", column, row[column], verify[key], verify[key])


def _truth_cell(product: RangeDopplerProduct, truth: TargetGeometry) -> tuple[int, int]:
    """Return the first target's nearest map cell, ``(Doppler row, lag)``."""
    range_index, rate_index = product.index_of(
        truth.bistatic_range_m[0], truth.bistatic_range_rate_mps[0]
    )
    n_rate = product.rd_map.shape[0]
    return int(np.rint(rate_index)) % n_rate, int(np.rint(range_index))


def _floor_w(rd_map: NDArray[np.complex128], cell: tuple[int, int]) -> float:
    """Return the mean power of the floor window about ``cell``: Doppler wraps, lags do not."""
    n_rate, n_lags = rd_map.shape
    doppler = np.concatenate([-_FLOOR_DOPPLER_BINS, _FLOOR_DOPPLER_BINS])
    lags = np.concatenate([-_FLOOR_LAG_BINS, _FLOOR_LAG_BINS])
    rows = (cell[0] + doppler) % n_rate
    columns = cell[1] + lags
    columns = columns[(columns >= 0) & (columns < n_lags)]
    return float(np.mean(np.abs(rd_map[np.ix_(rows, columns)]) ** 2))


def _target_detections(
    product: RangeDopplerProduct, scenario: Scenario, truth: TargetGeometry
) -> int:
    """Return how many CFAR clusters lie within one bin of the first target on both axes."""
    found = pipeline.frame_plots([product], scenario.detection)
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
    """Draw frame 0 about the target before and after cancellation: buried, then clear."""
    product = maps["clean"]
    n_rate, n_lags = product.rd_map.shape
    rows = slice(max(0, cell[0] - _FIGURE_ROWS), min(n_rate, cell[0] + _FIGURE_ROWS + 1))
    lags = slice(max(0, cell[1] - _FIGURE_LAGS), min(n_lags, cell[1] + _FIGURE_LAGS + 1))
    figure = plots.render_cancellation(
        maps["uncancelled"].rd_map[rows, lags],
        maps["cancelled"].rd_map[rows, lags],
        product.range_axis_m[lags],
        product.rate_axis_mps[rows],
        ctx.scenario.radar,
        floor_w=floor_w,
        truth_range_m=truth.bistatic_range_m,
        truth_rate_mps=truth.bistatic_range_rate_mps,
        title=f"{ctx.scenario.name}, frame 0",
        dynamic_range_db=_FIGURE_RANGE_DB,
    )
    plots.save_figure(figure, ctx.out_dir / "cancellation.png")
