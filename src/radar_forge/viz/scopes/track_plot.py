"""The range-time track scope: truth, detections and the track, in one picture.

The second of the two pictures ``spec/scenario-003-tracking.md`` §1
asks for, and the one an intern can look at and immediately say is wrong. The
range-Doppler scope shows one frame; this shows the *history*, which is where a
tracker's characteristic failures live -- a track that lags through a turn, a
track that swaps onto a false alarm, a track that was deleted and re-initiated
under a new identity.

Range against time, not a plan view, and deliberately: a plan view is the ground plane,
and this radar measures range and range rate and nothing else (§4), so an
east-north plot would have to invent a bearing for every point it drew. Range
against time is what the radar actually knows, and the gap between the truth
line and the track line is the error, read straight off the y axis. The
function is named for what it draws.

References
----------
.. [1] ``spec/scenario-003-tracking.md``, §9 (outputs).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.viz.plotting import require_pyplot

__all__ = ["render_range_time_history"]


def render_range_time_history(
    truth_time_s: ArrayLike,
    truth_range_m: ArrayLike,
    *,
    detection_time_s: ArrayLike | None = None,
    detection_range_m: ArrayLike | None = None,
    track_time_s: ArrayLike | None = None,
    track_range_m: ArrayLike | None = None,
    track_sigma_range_m: ArrayLike | None = None,
    current_time_s: float | None = None,
    title: str | None = None,
    range_margin_m: float = 1_000.0,
    truth_range_rate_mps: ArrayLike | None = None,
    track_range_rate_mps: ArrayLike | None = None,
    range_period_m: float | None = None,
) -> Any:
    """Draw the truth, the detections and the track history against time.

    Parameters
    ----------
    truth_time_s, truth_range_m : array_like
        Shape ``(n_frames,)``. The true range at each frame, the reference
        everything else is read against.
    detection_time_s, detection_range_m : array_like, optional
        Shape ``(n_detections,)``. Every CFAR detection so far, associated or
        not. Drawn as small dots, so a frame's worth of false alarms reads as
        scatter around the truth line rather than as a competing track.
    track_time_s, track_range_m : array_like, optional
        Shape ``(n_track_frames,)``. The confirmed track's estimate.
    track_sigma_range_m : array_like, optional
        Shape ``(n_track_frames,)``. One standard deviation of the track's range
        estimate, drawn as a shaded band. A band that shrinks while the track
        drifts away from the truth is the picture of a filter that is confidently
        wrong, which is the most useful single thing this scope shows.
    current_time_s : float, optional
        Marks the frame being played, so a movie assembled from these has a
        moving cursor rather than a static plot that grows.
    title : str, optional
        Figure title.
    range_margin_m : float, optional
        How far above and below the truth and the track the y axis extends,
        default 1000 m. False alarms are scattered over the whole unambiguous
        range, so letting them set the scale would compress the truth and the
        track into a single line and hide the error this scope exists to show.
        Detections outside the limits are counted in a corner note rather than
        silently dropped; a marker that vanishes off-plot leaves the reader with
        a picture that looks cleaner than the data.
    truth_range_rate_mps : array_like, optional
        Shape ``(n_frames,)``. The true range rate, closing-positive. When
        given, a second panel below the first plots range rate against time,
        with ``track_range_rate_mps`` if that is given too.
    track_range_rate_mps : array_like, optional
        Shape ``(n_track_frames,)``. The track's range-rate estimate. Needs
        ``truth_range_rate_mps``.
    range_period_m : float, optional
        The range period when range folds: the radar sees range only modulo
        this, so the absolute range is unknown. Pass every range already
        wrapped into ``[0, range_period_m)``. The axis is then labelled as a
        modulo range and spans one period, and a line is broken where it wraps
        rather than drawn across the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure. The caller owns it and must close it;
        :func:`radar_forge.viz.plotting.save_figure` does both.

    Raises
    ------
    ValueError
        If the truth arrays disagree in shape, a paired optional argument is
        supplied without its partner, or ``track_range_rate_mps`` is given
        without ``truth_range_rate_mps``.

    Notes
    -----
    The y axis is kilometres and the x axis seconds, so the slope of either line
    is a range rate in km/s. That is a slightly awkward unit and it is the point:
    a reader who wants metres per second has to think about the scale, and a
    reader who does not is not going to be misled by a Doppler sign error.
    """
    plt = require_pyplot()

    time_s = np.asarray(truth_time_s, dtype=np.float64)
    range_m = np.asarray(truth_range_m, dtype=np.float64)
    if time_s.shape != range_m.shape or time_s.ndim != 1:
        msg = (
            f"truth_time_s and truth_range_m must be one-dimensional and the same "
            f"shape; got {time_s.shape} and {range_m.shape}."
        )
        raise ValueError(msg)

    _require_pair(detection_time_s, detection_range_m, "detection")
    _require_pair(track_time_s, track_range_m, "track")
    if track_range_rate_mps is not None and truth_range_rate_mps is None:
        msg = "track_range_rate_mps needs truth_range_rate_mps, which draws its panel."
        raise ValueError(msg)

    if truth_range_rate_mps is None:
        figure, axes = plt.subplots(figsize=(8.0, 5.0))
    else:
        figure, (axes, rate_axes) = plt.subplots(
            2, 1, sharex=True, figsize=(8.0, 7.0), height_ratios=(2.0, 1.0)
        )
    axes.plot(
        time_s,
        _break_at_wraps(range_m, range_period_m) * 1.0e-3,
        color="tab:green",
        linewidth=1.8,
        label="truth",
        zorder=2,
    )

    if detection_time_s is not None and detection_range_m is not None:
        axes.scatter(
            np.asarray(detection_time_s, dtype=np.float64),
            np.asarray(detection_range_m, dtype=np.float64) * 1.0e-3,
            s=9.0,
            color="tab:orange",
            alpha=0.6,
            label="detections",
            zorder=3,
        )

    if track_time_s is not None and track_range_m is not None:
        track_s = np.asarray(track_time_s, dtype=np.float64)
        track_m = np.asarray(track_range_m, dtype=np.float64)
        if track_sigma_range_m is not None:
            sigma_m = np.asarray(track_sigma_range_m, dtype=np.float64)
            axes.fill_between(
                track_s,
                (track_m - sigma_m) * 1.0e-3,
                (track_m + sigma_m) * 1.0e-3,
                color="tab:blue",
                alpha=0.2,
                linewidth=0.0,
                label="track +/- 1 sigma",
                zorder=1,
            )
        axes.plot(
            track_s,
            _break_at_wraps(track_m, range_period_m) * 1.0e-3,
            color="tab:blue",
            linewidth=1.4,
            linestyle="--",
            label="track",
            zorder=4,
        )

    if current_time_s is not None:
        axes.axvline(current_time_s, color="0.4", linewidth=0.8, linestyle=":", zorder=0)

    lower_m, upper_m = (
        (0.0, range_period_m)
        if range_period_m is not None
        else _range_limits_m(range_m, track_range_m, range_margin_m)
    )
    axes.set_ylim(lower_m * 1.0e-3, upper_m * 1.0e-3)
    if detection_range_m is not None:
        detections_m = np.asarray(detection_range_m, dtype=np.float64)
        n_off_scale = int(np.count_nonzero((detections_m < lower_m) | (detections_m > upper_m)))
        if n_off_scale:
            axes.text(
                0.99,
                0.01,
                f"{n_off_scale} detection(s) off-scale",
                transform=axes.transAxes,
                ha="right",
                va="bottom",
                fontsize="x-small",
                color="tab:orange",
            )

    axes.set_ylabel(
        "Range (km)"
        if range_period_m is None
        else f"Range modulo {range_period_m * 1.0e-3:.3f} km (km)"
    )
    axes.grid(visible=True, alpha=0.25)
    axes.legend(loc="best", framealpha=0.85, fontsize="small")
    if title is not None:
        axes.set_title(title)

    if truth_range_rate_mps is None:
        axes.set_xlabel("Time (s)")
        return figure

    rate_axes.plot(
        time_s,
        np.asarray(truth_range_rate_mps, dtype=np.float64),
        color="tab:green",
        linewidth=1.8,
        label="truth",
    )
    if track_time_s is not None and track_range_rate_mps is not None:
        rate_axes.plot(
            np.asarray(track_time_s, dtype=np.float64),
            np.asarray(track_range_rate_mps, dtype=np.float64),
            color="tab:blue",
            linewidth=1.4,
            linestyle="--",
            label="track",
        )
    if current_time_s is not None:
        rate_axes.axvline(current_time_s, color="0.4", linewidth=0.8, linestyle=":", zorder=0)
    rate_axes.set_xlabel("Time (s)")
    rate_axes.set_ylabel("Range rate, closing (m/s)")
    rate_axes.grid(visible=True, alpha=0.25)
    return figure


def _break_at_wraps(value: NDArray[np.float64], period: float | None) -> NDArray[np.float64]:
    """Return ``value`` with NaN where it wraps, so a line is not drawn across the plot.

    A jump of more than half a period between neighbours is a wrap, not motion.
    Matplotlib leaves a gap at a NaN, so setting the point after each jump to
    NaN breaks the line there and costs that one point.
    """
    if period is None or value.size < 2:
        return value
    broken = value.copy()
    broken[1:][np.abs(np.diff(value)) > period / 2.0] = np.nan
    return broken


def _range_limits_m(
    truth_range_m: NDArray[np.float64],
    track_range_m: ArrayLike | None,
    margin_m: float,
) -> tuple[float, float]:
    """Y limits that show the truth and the track, not the clutter."""
    values = [truth_range_m]
    if track_range_m is not None:
        track = np.asarray(track_range_m, dtype=np.float64)
        if track.size:
            values.append(track)
    combined = np.concatenate(values)
    return float(combined.min() - margin_m), float(combined.max() + margin_m)


def _require_pair(first: ArrayLike | None, second: ArrayLike | None, name: str) -> None:
    """Reject half a pair, rather than silently drawing nothing."""
    if (first is None) != (second is None):
        msg = f"{name}_time_s and {name}_range_m must be given together; got one without the other."
        raise ValueError(msg)
