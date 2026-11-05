r"""Figures for reading a run: the range-Doppler map and the range-time history.

Matplotlib is the ``visualisation`` extra and is imported inside each drawing function, so
``import radar_forge`` needs no extra. :func:`display_axes` is the one place a monostatic
radar's bistatic range and rate are halved and relabelled range and radial velocity, for a
figure or a CSV column; a bistatic radar keeps :math:`R_T + R_R` and its rate. Every rate is
positive receding. Magnitudes are shown in decibels relative to the frame's own peak.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014.
.. [2] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.dsp import fold_velocity_mps
from radar_forge.radar import Radar

__all__ = [
    "DisplayAxes",
    "display_axes",
    "magnitude_db",
    "render_cancellation",
    "render_range_doppler",
    "render_range_time_history",
    "save_figure",
]

_EXTRA_HINT = (
    "matplotlib is required to draw figures but is not installed; it ships in the "
    "'visualisation' extra: uv sync --extra visualisation."
)


def _pyplot() -> ModuleType:
    """Import :mod:`matplotlib.pyplot` on the non-interactive backend, or name the extra."""
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(_EXTRA_HINT) from error
    pyplot: ModuleType = plt
    return pyplot


@dataclass(frozen=True)
class DisplayAxes:
    """How a radar's bistatic range and rate are shown: ``scale`` times, under these names."""

    scale: float
    range_name: str
    rate_name: str
    range_label: str
    rate_label: str


def display_axes(radar: Radar) -> DisplayAxes:
    """Return the display of a radar's axes: halved and renamed only when monostatic.

    Examples
    --------
    >>> from radar_forge.radar import Receiver, Site, Transmitter
    >>> tx = Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3)
    >>> axes = display_axes(Radar(tx, Receiver(1.0e6, 30.0, 3.0), Site(36.0, -78.9, 60.0)))
    >>> axes.scale, axes.range_name, axes.rate_name
    (0.5, 'range_m', 'range_rate_mps')
    """
    if radar.is_monostatic:
        return DisplayAxes(
            0.5,
            "range_m",
            "range_rate_mps",
            "Range (km)",
            "Radial velocity (m/s, positive receding)",
        )
    return DisplayAxes(
        1.0,
        "bistatic_range_m",
        "bistatic_range_rate_mps",
        "Bistatic range $R_T + R_R$ (km)",
        "Bistatic range rate (m/s, positive receding)",
    )


def magnitude_db(
    values: ArrayLike, *, reference_linear: float | None = None, floor_db: float = -120.0
) -> NDArray[np.float64]:
    r"""Return :math:`20\log_{10}(|x| / x_{ref})`, an amplitude in dB, clamped at ``floor_db``.

    The reference defaults to the largest magnitude, so the peak is 0 dB. An amplitude ratio,
    so the factor is 20: a power array passed here would halve every number on the scale.

    Examples
    --------
    >>> [round(float(x), 2) for x in magnitude_db([1.0, 0.5, 0.0])]
    [0.0, -6.02, -120.0]
    """
    magnitude = np.abs(np.asarray(values))
    if not np.all(np.isfinite(magnitude)):
        msg = "values must be finite; a nan or inf has no place on a decibel scale."
        raise ValueError(msg)
    peak = float(magnitude.max(initial=0.0))
    reference = (peak if peak > 0.0 else 1.0) if reference_linear is None else reference_linear
    if reference <= 0.0:
        msg = f"reference_linear must be strictly positive; got {reference_linear!r}."
        raise ValueError(msg)
    floor_linear = reference * 10.0 ** (floor_db / 20.0)
    result: NDArray[np.float64] = 20.0 * np.log10(np.maximum(magnitude, floor_linear) / reference)
    return result


def save_figure(figure: Any, path: Path | str, *, dpi: int = 100) -> None:
    """Write a figure and close it: a run draws one per frame, and pyplot keeps them all open."""
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    _pyplot().close(figure)


def render_range_doppler(
    rd_map: ArrayLike,
    range_axis_m: ArrayLike,
    rate_axis_mps: ArrayLike,
    radar: Radar,
    *,
    detection_range_m: ArrayLike = (),
    detection_rate_mps: ArrayLike = (),
    truth_range_m: ArrayLike = (),
    truth_rate_mps: ArrayLike = (),
    title: str | None = None,
    dynamic_range_db: float = 60.0,
) -> Any:
    """Draw a map ``(n_doppler_bins, n_range_bins)`` in dB with detections, truth and its fold.

    Axes, detections and truth are bistatic, as :func:`~radar_forge.pipeline.map_axes` gives
    them. Detections are circled; the unfolded truth is crossed, off the plot when it folds;
    the folded truth, where the peak must appear, is a dashed crosshair.
    """
    plt = _pyplot()
    values = np.asarray(rd_map)
    range_m = np.asarray(range_axis_m, dtype=np.float64)
    rate_mps = np.asarray(rate_axis_mps, dtype=np.float64)
    if values.shape != (rate_mps.size, range_m.size) or dynamic_range_db <= 0.0:
        msg = f"rd_map {values.shape} must be (rate, range) bins {(rate_mps.size, range_m.size)}."
        raise ValueError(msg)
    show = display_axes(radar)
    range_bin_m, rate_bin_mps = range_m[1] - range_m[0], abs(rate_mps[1] - rate_mps[0])
    km = show.scale * 1.0e-3
    extent = (
        (range_m[0] - range_bin_m / 2) * km,
        (range_m[-1] + range_bin_m / 2) * km,
        (rate_mps.min() - rate_bin_mps / 2) * show.scale,
        (rate_mps.max() + rate_bin_mps / 2) * show.scale,
    )
    figure, axes = plt.subplots(figsize=(8.0, 5.0))
    # Row 0 is the highest rate, so the map is drawn top down; imshow is a grid of equal cells.
    image = axes.imshow(
        magnitude_db(values, floor_db=-dynamic_range_db),
        aspect="auto",
        origin="upper",
        extent=extent,
        cmap="viridis",
        vmin=-dynamic_range_db,
        vmax=0.0,
        interpolation="nearest",
    )
    figure.colorbar(image, ax=axes, label="Magnitude relative to peak (dB)")
    axes.set_xlabel(show.range_label)
    axes.set_ylabel(show.rate_label)
    axes.scatter(
        np.asarray(detection_range_m) * km,
        np.asarray(detection_rate_mps) * show.scale,
        s=70.0,
        facecolors="none",
        edgecolors="tab:orange",
        label="detections",
    )
    truth_m = np.asarray(truth_range_m, dtype=np.float64)
    truth_mps = np.asarray(truth_rate_mps, dtype=np.float64)
    axes.plot(truth_m * km, truth_mps * show.scale, "x", color="red", ms=10, label="truth")
    folded_m = np.mod(truth_m, range_m.size * range_bin_m)
    folded_mps = fold_velocity_mps(truth_mps, rate_mps.size * rate_bin_mps)
    for range_value_m, rate_value_mps in zip(folded_m, folded_mps, strict=True):
        # A crosshair per target, so the peak it predicts can be read off either axis.
        axes.axvline(range_value_m * km, color="white", linestyle="--", linewidth=0.8)
        axes.axhline(rate_value_mps * show.scale, color="white", linestyle="--", linewidth=0.8)
    axes.set_xlim(extent[0], extent[1])
    axes.set_ylim(extent[2], extent[3])
    axes.legend(loc="upper right", fontsize="small", framealpha=0.85)
    if title is not None:
        axes.set_title(title)
    return figure


def _break_at_wraps(values: NDArray[np.float64], period: float | None) -> NDArray[np.float64]:
    """Return ``values`` with nan where consecutive samples jump over half a period."""
    if period is None or values.size < 2:
        return values
    jumps = np.abs(np.diff(values, axis=0)) > period / 2.0
    result = values.astype(np.float64, copy=True)
    result[1:][jumps] = np.nan
    return result


def render_range_time_history(
    truth_time_s: ArrayLike,
    truth_range_m: ArrayLike,
    truth_rate_mps: ArrayLike,
    radar: Radar,
    *,
    tracks: Sequence[tuple[str, ArrayLike, ArrayLike, ArrayLike]] = (),
    detection_time_s: ArrayLike = (),
    detection_range_m: ArrayLike = (),
    range_period_m: float | None = None,
    title: str | None = None,
) -> Any:
    """Draw range and rate against time: truth ``(n_frames, n_targets)``, detections, tracks.

    Each track is ``(label, time_s, range_m, rate_mps)``. With ``range_period_m`` every range
    is shown modulo it, the way the map sees it, and a line is broken where it wraps.
    """
    plt = _pyplot()
    show = display_axes(radar)
    km = show.scale * 1.0e-3
    period = None if range_period_m is None else float(range_period_m)

    def shown_m(range_value_m: ArrayLike) -> NDArray[np.float64]:
        value_m = np.asarray(range_value_m, dtype=np.float64)
        return value_m if period is None else np.mod(value_m, period)

    figure, (range_axes, rate_axes) = plt.subplots(
        2, 1, sharex=True, figsize=(8.0, 7.0), height_ratios=(2.0, 1.0)
    )
    time_s = np.asarray(truth_time_s, dtype=np.float64)
    truth_m = _break_at_wraps(shown_m(truth_range_m), period)
    range_axes.plot(time_s, truth_m * km, color="tab:green", linewidth=1.8, label="truth")
    rate_axes.plot(time_s, np.asarray(truth_rate_mps) * show.scale, color="tab:green")
    range_axes.scatter(
        np.asarray(detection_time_s),
        shown_m(detection_range_m) * km,
        s=9.0,
        color="tab:orange",
        alpha=0.6,
        label="detections",
    )
    # One line per track: a handful, each its own colour and label.
    for label, track_time_s, track_range_m, track_rate_mps in tracks:
        track_m = _break_at_wraps(shown_m(track_range_m), period)
        range_axes.plot(track_time_s, track_m * km, linestyle="--", label=label)
        rate_axes.plot(track_time_s, np.asarray(track_rate_mps) * show.scale, linestyle="--")
    if period is not None:
        range_axes.set_ylim(0.0, period * km)
    else:
        span_m = np.asarray(truth_range_m, dtype=np.float64)
        range_axes.set_ylim((span_m.min() - 2e3) * km, (span_m.max() + 2e3) * km)
    range_axes.set_ylabel(show.range_label + (" modulo the map" if period is not None else ""))
    rate_axes.set_ylabel(show.rate_label.replace(" (m/s", "\n(m/s"))
    rate_axes.set_xlabel("Time (s)")
    range_axes.legend(loc="upper right", fontsize="small", framealpha=0.85)
    if title is not None:
        range_axes.set_title(title)
    return figure


def render_cancellation(
    before: ArrayLike,
    after: ArrayLike,
    range_axis_m: ArrayLike,
    rate_axis_mps: ArrayLike,
    radar: Radar,
    *,
    floor_w: float,
    truth_range_m: ArrayLike = (),
    truth_rate_mps: ArrayLike = (),
    title: str | None = None,
    dynamic_range_db: float = 60.0,
) -> Any:
    """Draw a map before and after cancellation, side by side, on one scale over the noise.

    Both panels are power over ``floor_w``, the clean map's floor, from 0 to
    ``dynamic_range_db`` dB: what cancellation removed reads straight off the colour. Maps are
    ``(n_doppler_bins, n_range_bins)`` on bistatic axes, rate falling with row; truth is crossed.
    """
    plt = _pyplot()
    range_m = np.asarray(range_axis_m, dtype=np.float64)
    rate_mps = np.asarray(rate_axis_mps, dtype=np.float64)
    if not floor_w > 0.0 or dynamic_range_db <= 0.0:
        msg = "floor_w and dynamic_range_db must be strictly positive."
        raise ValueError(msg)
    show = display_axes(radar)
    km = show.scale * 1.0e-3
    range_bin_m, rate_bin_mps = range_m[1] - range_m[0], abs(rate_mps[1] - rate_mps[0])
    extent = (
        (range_m[0] - range_bin_m / 2) * km,
        (range_m[-1] + range_bin_m / 2) * km,
        (rate_mps.min() - rate_bin_mps / 2) * show.scale,
        (rate_mps.max() + rate_bin_mps / 2) * show.scale,
    )
    figure, panels = plt.subplots(1, 2, sharey=True, figsize=(11.0, 4.5), layout="constrained")
    image = None
    # Two panels, the same drawing: before cancellation, then after.
    for axes, values, label in zip(panels, (before, after), ("before", "after"), strict=True):
        power_db = 10.0 * np.log10(np.maximum(np.abs(np.asarray(values)) ** 2 / floor_w, 1e-30))
        image = axes.imshow(
            power_db,
            aspect="auto",
            origin="upper",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=dynamic_range_db,
            interpolation="nearest",
        )
        truth = zip(np.atleast_1d(truth_range_m), np.atleast_1d(truth_rate_mps), strict=True)
        # A crosshair per target, which leaves its own cell in view; one legend entry for all.
        for k, (range_value_m, rate_value_mps) in enumerate(truth):
            style = {"color": "white", "linestyle": "--", "linewidth": 0.8}
            axes.axvline(range_value_m * km, label="truth" if k == 0 else "_truth", **style)
            axes.axhline(rate_value_mps * show.scale, **style)
        axes.set_title(f"{label} cancellation")
        axes.set_xlabel(show.range_label)
        axes.set_xlim(extent[0], extent[1])
        axes.set_ylim(extent[2], extent[3])
    panels[0].set_ylabel(show.rate_label)
    if np.size(truth_range_m):
        panels[1].legend(loc="upper right", fontsize="small", framealpha=0.85)
    figure.colorbar(image, ax=panels, label="Power over the clean noise floor (dB)")
    if title is not None:
        figure.suptitle(title)
    return figure
