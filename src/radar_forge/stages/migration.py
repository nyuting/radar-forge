r"""The ``keystone`` stage: range migration measured, corrected and estimated on frame 0.

On frame 0's noiseless cube (the generator's ``rng=None`` path) it measures the walk, the
loss of the uncorrected map, the keystone alone and the dechirp alone, then runs ``[migration]``
and measures its searches, residual walk and loss. A loss is the map's peak, off the bin grid,
over :math:`N^2` times pulse 0's compressed peak: 0 dB is ideal integration, :math:`10\log_{10}N`
above one pulse's SNR. On frame 0 with noise, the frame the ``rd`` stage mapped, it crops a
chip at the corrected map's peak and estimates range, rate and acceleration against their
Cramér-Rao bounds. Writes ``migration.csv`` and ``keystone.png``.

References
----------
.. [1] P. Huang, G. Liao, Z. Yang, X.-G. Xia, J.-T. Ma and J. Ma, "Long-time coherent
       integration for weak maneuvering target detection and high-order motion parameter
       estimation based on keystone transform," *IEEE Trans. Signal Processing*, vol. 64,
       no. 15, pp. 4013-4026, 2016.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from radar_forge import evaluate, plots
from radar_forge.datachip import ChipProcessing, FineMethod, crop_slow_time, estimate_fine
from radar_forge.migration import doppler_rate_compensation, integrated_peak, keystone
from radar_forge.pipeline import (
    CompensatedPulses,
    compensate_migration,
    frames,
    map_axes,
    pulse_compress,
    synthesise,
)
from radar_forge.radar import Radar

__all__ = ["keystone_stage"]

# The pulses either end of the dwell that DFT interpolation leaves ringing (spec A2).
_RINGING_PULSES = 1
# Range bins either side of the target that keystone.png shows.
_PLOT_HALF_WIDTH = 120


@evaluate.stage(
    "keystone",
    "walk_bins",
    "walk_tolerance_bins",
    "residual_walk_max_bins",
    "uncompensated_loss_db",
    "keystone_only_loss_db",
    "dechirp_only_loss_db",
    "loss_tolerance_db",
    "keystone_only_loss_tolerance_db",
    "compensated_loss_min_db",
    "fold_factor",
    "acceleration_tolerance_mps2",
    "rate_error_max_crb",
)
def keystone_stage(ctx: evaluate.Context) -> evaluate.Context:
    """Walk, losses, searches and fine estimates of the first target on frame 0.

    ``[verify]``: ``walk_bins`` within ``walk_tolerance_bins``; ``residual_walk_max_bins``;
    ``uncompensated_loss_db``, ``keystone_only_loss_db`` and ``dechirp_only_loss_db`` within
    ``loss_tolerance_db`` (the keystone alone within ``keystone_only_loss_tolerance_db``);
    ``compensated_loss_min_db``; the searched ``fold_factor``; the searched acceleration within
    ``acceleration_tolerance_mps2`` of the truth; the fine rate within ``rate_error_max_crb``
    of its bound.
    """
    scenario, verify = ctx.scenario, evaluate.thresholds(ctx, "keystone")
    radar, n_pulses, gate = scenario.radar, scenario.n_pulses[0], scenario.gates[0]
    truth = evaluate.truth(ctx)[0]
    rcs_m2 = [target.rcs_m2 for target in scenario.targets]
    start_m = gate.range_gate_start_m
    clean = pulse_compress(
        synthesise(radar, truth, rcs_m2, n_pulses, rng=None, gate=gate),
        radar,
        range_gate_start_m=start_m,
    )
    reference_w = integrated_peak(clean[:1]).power_linear * n_pulses**2
    row: dict[str, Any] = {"frame": 0}

    def loss(name: str, data: NDArray[np.complex128]) -> float:
        row[name] = ctx.record(
            name, evaluate.to_db(integrated_peak(data).power_linear / reference_w)
        )
        return float(row[name])

    track = _peak_track_bins(clean)
    slope = np.polyfit(np.arange(n_pulses), track, 1)[0]
    row["walk_bins"] = ctx.record("walk_bins", abs(slope) * n_pulses)
    loss("uncompensated_loss_db", clean)
    carrier = radar.transmitter.carrier_frequency_hz
    sample_rate_hz = radar.receiver.sample_rate_hz
    compensated = compensate_migration(clean, radar, scenario.migration)
    # The keystone at fold factor 0: the method's own first step, unless it searched folds.
    keystoned = compensated.keystone or keystone(
        clean, carrier_frequency_hz=carrier, sample_rate_hz=sample_rate_hz
    )
    loss("keystone_only_loss_db", keystoned.data_rfst)
    if scenario.migration.method == "keystone_doppler_rate":
        dechirped = doppler_rate_compensation(
            clean,
            carrier_frequency_hz=carrier,
            sample_rate_hz=sample_rate_hz,
            prf_hz=radar.transmitter.prf_hz,
            accelerations_mps2=truth.bistatic_range_acceleration_mps2[:1],
        )
        loss("dechirp_only_loss_db", dechirped.data_rfst)
    residual = _peak_track_bins(compensated.data_rfst)[_RINGING_PULSES:-_RINGING_PULSES]
    row["residual_walk_bins"] = ctx.record("residual_walk_bins", float(np.ptp(residual)))
    loss("compensated_loss_db", compensated.data_rfst)
    row.update(_searches(ctx, compensated, truth.bistatic_range_acceleration_mps2[0]))
    row.update(_estimates(ctx))
    _check(ctx, verify, row)
    evaluate.write_csv(ctx.out_dir / "migration.csv", [row])
    if ctx.render:
        centre = int(np.rint(track[0]))
        window = slice(max(0, centre - _PLOT_HALF_WIDTH), centre + _PLOT_HALF_WIDTH)
        range_axis_m, _ = map_axes(
            radar, n_pulses, range_gate_start_m=start_m, n_range_bins=clean.shape[1]
        )
        figure = plots.render_range_slow_time(
            [
                ("compressed", clean[:, window]),
                ("keystone", keystoned.data_rfst[:, window]),
                (scenario.migration.method, compensated.data_rfst[:, window]),
            ],
            range_axis_m[window],
            radar,
            title=f"{scenario.name}, frame 0, noiseless",
        )
        plots.save_figure(figure, ctx.out_dir / "keystone.png")
    return ctx


def _searches(
    ctx: evaluate.Context, compensated: CompensatedPulses, true_acceleration_mps2: float
) -> dict[str, float]:
    """Record the fold-factor and acceleration searches' curves and choices."""
    found: dict[str, float] = {}
    if compensated.fold is not None:
        curve_db = 10.0 * np.log10(compensated.fold.integrated_peak_linear)
        for factor, value_db in zip(compensated.fold.fold_factors, curve_db, strict=True):
            print(f"    fold factor {factor:+d}: integrated peak {value_db:.2f} dB")
        found["fold_factor"] = ctx.record("fold_factor", compensated.fold_factor)
    if compensated.rate is not None:
        curve_db = 10.0 * np.log10(compensated.rate.integrated_peak_linear[:, 0])
        for value_mps2, value_db in zip(compensated.rate.accelerations_mps2, curve_db, strict=True):
            print(f"    acceleration {value_mps2:+.1f} m/s^2: integrated peak {value_db:.3f} dB")
        found["acceleration_mps2"] = ctx.record("acceleration_mps2", compensated.acceleration_mps2)
        found["acceleration_error_mps2"] = ctx.record(
            "acceleration_error_mps2", compensated.acceleration_mps2 - true_acceleration_mps2
        )
    return found


def _estimates(ctx: evaluate.Context) -> dict[str, Any]:
    """Correct frame 0 with noise as configured, crop a chip at its peak, estimate on it."""
    scenario = ctx.scenario
    radar, n_pulses, gate = scenario.radar, scenario.n_pulses[0], scenario.gates[0]
    frame = next(frames(scenario, n_frames=1))
    noisy = pulse_compress(frame.cubes[0], radar, range_gate_start_m=gate.range_gate_start_m)
    compensated = compensate_migration(noisy, radar, scenario.migration)
    range_axis_m, rate_axis_mps = map_axes(
        radar, n_pulses, range_gate_start_m=gate.range_gate_start_m, n_range_bins=noisy.shape[1]
    )
    peak = integrated_peak(compensated.data_rfst, refine_range=False)
    chip = crop_slow_time(
        compensated.data_rfst,
        int(peak.range_bin),
        range_bin_centers_m=range_axis_m,
        rate_bin_centers_mps=rate_axis_mps,
        processing=_processing(radar, compensated, scenario.migration.method != "none"),
    )
    method: FineMethod = "grft" if compensated.rate is not None else "aboutanios_mulgrew"
    estimate = estimate_fine(chip, method=method)
    truth, crb = frame.truth, estimate.crb
    errors = {
        "range": (estimate.bistatic_range_m - truth.bistatic_range_m[0], crb.bistatic_range_m),
        "rate": (
            estimate.bistatic_range_rate_mps - truth.bistatic_range_rate_mps[0],
            crb.bistatic_range_rate_mps,
        ),
        "acceleration": (
            estimate.bistatic_range_acceleration_mps2 - truth.bistatic_range_acceleration_mps2[0],
            crb.bistatic_range_acceleration_mps2,
        ),
    }
    found: dict[str, Any] = {
        "estimator": method,
        "fine_fold_factor": compensated.fold_factor,
        "fine_snr_db": ctx.record("fine_snr_db", evaluate.to_db(estimate.snr_linear)),
        "fine_bistatic_range_m": ctx.record("fine_bistatic_range_m", estimate.bistatic_range_m),
        "fine_bistatic_range_rate_mps": ctx.record(
            "fine_bistatic_range_rate_mps", estimate.bistatic_range_rate_mps
        ),
    }
    if compensated.rate is not None:
        found["fine_acceleration_mps2"] = ctx.record(
            "fine_acceleration_mps2", estimate.bistatic_range_acceleration_mps2
        )
    for name, (error, bound) in errors.items():
        if np.isfinite(bound):
            found[f"{name}_crb"] = ctx.record(f"{name}_crb", bound)
            found[f"{name}_error_crb"] = ctx.record(f"{name}_error_crb", error / bound)
    return found


def _processing(radar: Radar, compensated: CompensatedPulses, keystoned: bool) -> ChipProcessing:
    """Return what a chip of ``compensated`` needs to know of its parent's processing."""
    transmitter, receiver = radar.transmitter, radar.receiver
    return ChipProcessing(
        carrier_frequency_hz=transmitter.carrier_frequency_hz,
        bandwidth_hz=transmitter.bandwidth_hz,
        sample_rate_hz=receiver.sample_rate_hz,
        prf_hz=transmitter.prf_hz,
        # The matched filter sums n_chirp unit-amplitude taps of per-sample noise.
        noise_power_w=receiver.noise_power_per_sample_w * radar.n_samples_per_chirp,
        keystone_applied=keystoned,
        fold_factor=compensated.fold_factor,
        compensated_acceleration_mps2=compensated.acceleration_mps2,
    )


def _peak_track_bins(data_rfst: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return each pulse's range peak, fractional bins: the vertex of a parabola through |x|."""
    magnitude = np.abs(data_rfst)
    peak = np.clip(np.argmax(magnitude, axis=1), 1, magnitude.shape[1] - 2)
    pulses = np.arange(magnitude.shape[0])
    below, centre, above = (magnitude[pulses, peak + k] for k in (-1, 0, 1))
    result: NDArray[np.float64] = peak + 0.5 * (below - above) / (below - 2.0 * centre + above)
    return result


def _check(ctx: evaluate.Context, verify: dict[str, Any], row: dict[str, Any]) -> None:
    """Check every threshold ``[verify]`` gives against what the stage measured."""
    tolerance_db = verify.get("loss_tolerance_db", 1.0)
    if "walk_bins" in verify:
        expected, within = verify["walk_bins"], verify.get("walk_tolerance_bins", 1.0)
        ctx.check("keystone", "walk_bins", row["walk_bins"], expected - within, expected + within)
    if "residual_walk_max_bins" in verify:
        high = verify["residual_walk_max_bins"]
        ctx.check("keystone", "residual_walk_bins", row["residual_walk_bins"], 0.0, high)
    for key in ("uncompensated_loss_db", "keystone_only_loss_db", "dechirp_only_loss_db"):
        if key in verify:
            within = tolerance_db
            if key == "keystone_only_loss_db":
                within = verify.get("keystone_only_loss_tolerance_db", tolerance_db)
            expected = verify[key]
            ctx.check("keystone", key, row[key], expected - within, expected + within)
    if "compensated_loss_min_db" in verify:
        low = verify["compensated_loss_min_db"]
        ctx.check("keystone", "compensated_loss_db", row["compensated_loss_db"], low, 0.0)
    if "fold_factor" in verify:
        expected = verify["fold_factor"]
        ctx.check("keystone", "fold_factor", row.get("fold_factor", 0), expected, expected)
    if "acceleration_tolerance_mps2" in verify:
        within = verify["acceleration_tolerance_mps2"]
        error = abs(row.get("acceleration_error_mps2", float("nan")))
        ctx.check("keystone", "acceleration_error_mps2", error, 0.0, within)
    if "rate_error_max_crb" in verify:
        error = abs(row.get("rate_error_crb", float("nan")))
        ctx.check("keystone", "rate_error_crb", error, 0.0, verify["rate_error_max_crb"])
