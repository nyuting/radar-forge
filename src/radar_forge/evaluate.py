r"""The stepper: a scenario run as named stages, each printing, checking and writing one step.

The stages live in :mod:`radar_forge.stages` and run in that package's import order;
:mod:`radar_forge.stages.core` holds ``geometry``, ``iq``, ``rd``, ``detect`` and ``track``.
Each prints its intermediate quantities and wall time, checks the thresholds the scenario's
``[verify]`` table gives, and writes its files under the run directory
(``spec/data-001-formats.md``). ``[verify]`` holds ``stages`` and thresholds:
``truth_in_map_fraction = [low, high]``, ``expected_snr_db`` with ``snr_tolerance_db``,
``noise_tolerance_db``, ``peak_tolerance_db``, ``truth_within_bins`` with
``detection_fraction``, ``pfa_tolerance_factor``, ``rate_tolerance_mps``, ``range_rmse_max_m``,
``range_rate_rmse_max_mps``, ``nees_fraction_min`` and ``confirmation_latency_max_scans``. A
stage not listed does not run, a threshold not given is not checked, and :func:`run` raises
:class:`VerificationError` naming every failure. CSVs show a monostatic radar's range and rate
halved (:func:`~radar_forge.plots.display_axes`); everything else is bistatic. The helpers
below are what a stage module shares.

:mod:`radar_forge.stages.migration` holds ``keystone`` (scenario 004): ``walk_bins`` with
``walk_tolerance_bins``, ``residual_walk_max_bins``, ``uncompensated_loss_db``,
``keystone_only_loss_db`` and ``dechirp_only_loss_db`` with ``loss_tolerance_db`` (the
keystone's own ``keystone_only_loss_tolerance_db``), ``compensated_loss_min_db``,
``fold_factor``, ``acceleration_tolerance_mps2`` and ``rate_error_max_crb``.

:mod:`radar_forge.stages.clutter` holds ``leakage`` (scenario 005): ``cancellation_depth_db``
with ``cancellation_tolerance_db``, ``floor_rise_db`` with ``floor_rise_tolerance_db``,
``sinr_loss_db`` with ``sinr_loss_tolerance_db``, ``detections_per_frame``,
``uncancelled_detections_per_frame`` and ``acceptance_frames``.

:mod:`radar_forge.stages.tracking` holds ``multitarget`` (scenario 011): ``gospa_mean_max_m``,
``swap_fraction_max``, ``mode_latency_max_scans``, ``nees_fraction_min``, ``pd_min`` and
``false_alarm_rate_max``; 011b's ``leaf_count_max``, ``pmbm_cardinality_error_max``,
``mht_resolved_lead_min`` and ``pmbm_resolved_lead_min``.

:mod:`radar_forge.stages.array` holds ``doa`` (scenario 006): ``direction_cosine_tolerance``,
``resolved_fraction_min`` with ``resolved_methods``, ``baseline_resolved_fraction_max`` with
``baseline_methods``, ``<method>_rmse_crb_max`` for each :data:`~radar_forge.doa.DOA_LABELS`
name, ``rank_ratio_max``, ``snapshot_model_tolerance`` and ``parent_coordinates_tolerance``.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.clutter import CancellationResult
from radar_forge.pipeline import Frame, Plots, RangeDopplerProduct, map_axes
from radar_forge.propagation import received_power_w
from radar_forge.radar import Radar, TargetGeometry
from radar_forge.scenario import Scenario, truth_geometry
from radar_forge.tracking.filters import TrackBank

__all__ = [
    "STAGES",
    "VERIFY_KEYS",
    "Check",
    "Context",
    "Report",
    "VerificationError",
    "cached_frames",
    "run",
    "spans",
    "stage",
    "target_power_w",
    "thresholds",
    "to_db",
    "truth",
    "write_csv",
]


@dataclass(frozen=True)
class Check:
    """One verified quantity: what was measured, the closed interval it needs, and the verdict."""

    stage: str
    name: str
    measured: float
    low: float
    high: float

    @property
    def passed(self) -> bool:
        """Whether the measured value lies in ``[low, high]``; nan never does."""
        return bool(self.low <= self.measured <= self.high)


@dataclass
class Report:
    """Every check of a run, every scalar a stage measured, and each stage's wall time."""

    scenario: str
    checks: list[Check] = field(default_factory=list)
    values: dict[str, float] = field(default_factory=dict)
    wall_time_s: dict[str, float] = field(default_factory=dict)

    @property
    def failures(self) -> list[Check]:
        """The checks that failed."""
        return [check for check in self.checks if not check.passed]


class VerificationError(Exception):
    """Raised by :func:`run` when any check fails; the message lists every failure."""

    def __init__(self, report: Report) -> None:
        """Keep the report, and name each failed check and the interval it needed."""
        self.report = report
        lines = [
            f"{c.stage}: {c.name} = {c.measured:.6g}, not in [{c.low}, {c.high}]"
            for c in report.failures
        ]
        super().__init__(f"{report.scenario}: {len(lines)} check(s) failed\n" + "\n".join(lines))


@dataclass
class Context:
    """What the stages share: the scenario, the run's settings, and each stage's results."""

    scenario: Scenario
    out_dir: Path
    n_frames: int
    save_iq: bool
    report: Report
    render: bool = True
    truth: list[TargetGeometry] = field(default_factory=list)
    plots: list[Plots] = field(default_factory=list)
    banks: list[tuple[TrackBank, NDArray[np.intp]]] = field(default_factory=list)
    stages: tuple[str, ...] = ()
    cancelled: list[tuple[Frame, CancellationResult, RangeDopplerProduct]] = field(
        default_factory=list
    )

    def record(self, name: str, value: float) -> float:
        """Print a measured scalar, keep it in the report, and return it."""
        self.report.values[name] = float(value)
        print(f"  {name} = {value:.6g}")
        return float(value)

    def check(self, stage_name: str, name: str, value: float, low: Any, high: Any) -> None:
        """Record a check that ``value`` lies in ``[low, high]``."""
        result = Check(stage_name, name, float(value), float(low), float(high))
        self.report.checks.append(result)
        verdict = "pass" if result.passed else "FAIL"
        print(f"  check {name} = {value:.6g} in [{low}, {high}]: {verdict}")


Stage = Callable[[Context], Context]
STAGES: dict[str, Stage] = {}
"""The registered stages, by name, in the order :func:`run` runs them."""
VERIFY_KEYS: dict[str, tuple[str, ...]] = {}
"""The ``[verify]`` keys each stage reads; a stage sees only these, and no other key loads."""


def stage(name: str, *keys: str) -> Callable[[Stage], Stage]:
    """Register a stage under ``name``, once, with the ``[verify]`` keys it reads."""

    def register(function: Stage) -> Stage:
        if name in STAGES:
            msg = f"a stage named {name!r} is already registered, by {STAGES[name].__module__}."
            raise ValueError(msg)
        STAGES[name], VERIFY_KEYS[name] = function, keys
        return function

    return register


def thresholds(ctx: Context, stage_name: str) -> dict[str, Any]:
    """Return the ``[verify]`` thresholds a stage registered, and only those."""
    return {k: v for k, v in ctx.scenario.verify.items() if k in VERIFY_KEYS[stage_name]}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dict rows under the first row's keys; a nan is written as an empty cell."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["frame"])
        writer.writeheader()
        writer.writerows(
            {k: "" if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()}
            for row in rows
        )


def to_db(value: ArrayLike) -> float:
    """Return a power ratio or a power in W as decibels."""
    return float(10.0 * np.log10(value))


def target_power_w(radar: Radar, truth: TargetGeometry, rcs_m2: ArrayLike) -> NDArray[np.float64]:
    """Return each target's received power by the radar equation, W."""
    return received_power_w(
        transmit_power_w=radar.transmitter.transmit_power_w,
        gain_tx_linear=10.0 ** (radar.transmitter.gain_tx_dbi / 10.0),
        gain_rx_linear=10.0 ** (radar.receiver.gain_rx_dbi / 10.0),
        wavelength_m=radar.wavelength_m,
        rcs_m2=rcs_m2,
        range_tx_m=truth.range_tx_m,
        range_rx_m=truth.range_rx_m,
    )


def spans(scenario: Scenario, burst: int) -> tuple[float, float, float]:
    """Return a burst's range span, rate span and range bin, m and m/s."""
    range_axis_m, rate_axis_mps = map_axes(
        scenario.radars[burst], scenario.n_pulses[burst], n_range_lags=scenario.n_range_lags[burst]
    )
    range_bin_m = float(range_axis_m[1] - range_axis_m[0])
    rate_span_mps = float(rate_axis_mps.size * (rate_axis_mps[0] - rate_axis_mps[1]))
    return range_axis_m.size * range_bin_m, rate_span_mps, range_bin_m


def cached_frames(ctx: Context) -> int:
    """Return how many noise frames ``rd`` keeps, cancelled, for ``leakage``: its acceptance frames.

    Each holds a stream, its code, the residual and the map, about 72 MB at 005's size; none is
    kept unless ``leakage`` runs.
    """
    if "leakage" not in ctx.stages:
        return 0
    return min(ctx.n_frames, int(ctx.scenario.verify.get("acceptance_frames", ctx.n_frames)))


def truth(ctx: Context) -> list[TargetGeometry]:
    """Return every frame's truth, from the geometry stage or computed now."""
    if not ctx.truth:
        ctx.truth = truth_geometry(ctx.scenario)[: ctx.n_frames]
    return ctx.truth


def run(
    scenario: Scenario,
    stages: Sequence[str] | None = None,
    *,
    n_frames: int | None = None,
    out_dir: Path | str,
    save_iq: bool = False,
    render: bool = True,
) -> Report:
    """Run the stages (default ``[verify].stages``); raise VerificationError naming each failure.

    Before any stage runs, a ``[verify]`` key no stage registered raises, so a misspelt
    threshold is never skipped in silence.
    """
    names = list(scenario.verify.get("stages", STAGES) if stages is None else stages)
    known = {"stages"}.union(*VERIFY_KEYS.values())
    unknown = sorted(set(names) - set(STAGES)) + sorted(set(scenario.verify) - known)
    if unknown:
        msg = f"[verify] names unknown stage(s) or key(s) {unknown}; known: {sorted(known)}."
        raise ValueError(msg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = scenario.n_frames if n_frames is None else min(n_frames, scenario.n_frames)
    ctx = Context(scenario, out, count, save_iq, Report(scenario.name), render, stages=tuple(names))
    print(f"{scenario.name}: {count} frame(s), {len(scenario.radars)} burst(s) -> {out}")
    for name, function in STAGES.items():
        if name in names:
            print(f"[{name}]")
            start_s = time.perf_counter()
            ctx = function(ctx)
            ctx.report.wall_time_s[name] = time.perf_counter() - start_s
            print(f"  wall time {ctx.report.wall_time_s[name]:.2f} s")
    if ctx.report.failures:
        raise VerificationError(ctx.report)
    return ctx.report


import radar_forge.stages  # noqa: E402, F401  -- the stages register themselves on import
