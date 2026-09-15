"""The range-Doppler scope: the picture scenario 001 exists to produce.

The display is deliberately labelled from
:func:`radar_forge.core.dsp.range_bin_centers_m` and
:func:`~radar_forge.core.dsp.doppler_bin_centers_mps` rather than from axes
derived here. Those helpers already encode the asymmetry that catches people
out -- range is unshifted, so zero range is bin 0, while Doppler is
``fftshift``-ed, so zero velocity is the middle bin -- and they encode each
waveform's unambiguous limits. Re-deriving an axis is how a plot ends up
mirrored or offset by half a span while still looking entirely reasonable.

Two markers, and the distance between them is the lesson. The **unfolded**
truth is where the target really is; the **folded** truth is where a radar
with these ambiguities is obliged to report it. For S1 they separate in
velocity, for S2 in range, and for S3 they coincide because the dual-PRF pair
resolved the ambiguity.

The unfolded truth frequently falls outside the axes -- that is what folding
means -- so when it does it is drawn as an arrow pinned to the edge, labelled
with its real value. A marker that silently vanishes off-plot would leave the
reader with a picture that looks unambiguous and is not.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, S5.3 (range-Doppler ambiguity and its display).
.. [2] ``spec/scenario-001-singapore-xband.md``, S7.1 (the dsp seam) and S6.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from radar_forge.teaching.plotting import magnitude_db, require_pyplot

__all__ = ["render_range_doppler"]

_DEFAULT_DYNAMIC_RANGE_DB = 60.0


def render_range_doppler(
    rd_map: ArrayLike,
    range_axis_m: ArrayLike,
    velocity_axis_mps: ArrayLike,
    *,
    truth_range_m: float | None = None,
    truth_velocity_mps: float | None = None,
    folded_range_m: float | None = None,
    folded_velocity_mps: float | None = None,
    title: str | None = None,
    dynamic_range_db: float = _DEFAULT_DYNAMIC_RANGE_DB,
    bistatic: bool = False,
) -> Any:
    """Draw a range-Doppler map, optionally with the true target marked.

    Parameters
    ----------
    rd_map : array_like
        Complex range-Doppler map, shape ``(n_doppler_bins, n_range_bins)`` --
        the layout :func:`radar_forge.core.dsp.range_doppler_map` returns, with
        slow time on axis 0.
    range_axis_m : array_like
        Range bin centres in metres, shape ``(n_range_bins,)``. Pass the output
        of :func:`radar_forge.core.dsp.range_bin_centers_m`.
    velocity_axis_mps : array_like
        Velocity bin centres in metres/second, shape ``(n_doppler_bins,)``,
        from :func:`radar_forge.core.dsp.doppler_bin_centers_mps`. Already
        ``fftshift``-ed, zero in the middle, positive closing.
    truth_range_m, truth_velocity_mps : float, optional
        The true, **unfolded** target position. Drawn as a hollow circle if it
        falls inside the axes, and as an edge arrow labelled with its value if
        it does not -- which is the usual case for a folding waveform.
    folded_range_m, folded_velocity_mps : float, optional
        Where that truth is obliged to appear on *this* map, from
        :func:`radar_forge.core.ambiguity.fold_velocity_mps` and the range
        modulo. Drawn as a cross, and it should sit on the peak. Defaults to
        the unfolded values, which is right when nothing folds.
    title : str, optional
        Figure title.
    bistatic : bool, optional
        Label the axes for a bistatic pair: the range axis carries the bistatic
        mean range :math:`(R_t + R_r)/2` rather than a slant range, and the
        velocity axis the bisector range rate. Default ``False``. Only the
        labels change; the data is computed identically either way.
    dynamic_range_db : float, optional
        Decibels shown below the peak, default 60. Everything dimmer is
        clamped to the bottom of the colour scale.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure. The caller owns it and must close it;
        :func:`radar_forge.teaching.plotting.save_figure` does both.

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed; the message names the ``teaching``
        extra.
    ValueError
        If the axes do not match the map's shape, or if
        ``dynamic_range_db`` is not strictly positive.

    Notes
    -----
    The colour scale is dB relative to this frame's own peak, so successive
    frames are comparable in shape even as the target's received power changes
    by the fourth power of range across the track.
    """
    plt = require_pyplot()

    map_values = np.asarray(rd_map)
    range_m = np.asarray(range_axis_m, dtype=np.float64)
    velocity_mps = np.asarray(velocity_axis_mps, dtype=np.float64)

    if map_values.ndim != 2:
        msg = (
            "rd_map must be two-dimensional (n_doppler_bins, n_range_bins); "
            f"got {map_values.shape}."
        )
        raise ValueError(msg)
    if map_values.shape != (velocity_mps.size, range_m.size):
        msg = (
            f"axes do not match the map: rd_map is {map_values.shape} but the axes give "
            f"{(velocity_mps.size, range_m.size)}. Range is axis 1, Doppler is axis 0."
        )
        raise ValueError(msg)
    if dynamic_range_db <= 0.0:
        msg = f"dynamic_range_db must be strictly positive; got {dynamic_range_db!r}."
        raise ValueError(msg)

    map_db = magnitude_db(map_values, floor_db=-2.0 * dynamic_range_db)

    figure, axes = plt.subplots(figsize=(8.0, 5.0))
    mesh = axes.pcolormesh(
        range_m * 1.0e-3,
        velocity_mps,
        map_db,
        shading="nearest",
        cmap="viridis",
        vmin=-dynamic_range_db,
        vmax=0.0,
    )
    figure.colorbar(mesh, ax=axes, label="Magnitude relative to peak (dB)")

    # The only thing a bistatic map changes. The numbers on both axes are
    # produced by the same code either way -- see decision D6 of
    # spec/scenario-002-singapore-bistatic.md -- but they mean different
    # things, and a plot that did not say so would be read as a slant range.
    if bistatic:
        axes.set_xlabel("Bistatic mean range (km)")
        axes.set_ylabel("Bisector range rate (m/s, positive closing)")
    else:
        axes.set_xlabel("Range (km)")
        axes.set_ylabel("Radial velocity (m/s, positive closing)")
    range_limits_km = (range_m[0] * 1.0e-3, range_m[-1] * 1.0e-3)
    velocity_limits_mps = (velocity_mps[0], velocity_mps[-1])
    axes.set_xlim(*range_limits_km)
    axes.set_ylim(*velocity_limits_mps)

    if truth_range_m is not None and truth_velocity_mps is not None:
        _mark_truth(
            axes,
            truth_range_m,
            truth_velocity_mps,
            truth_range_m if folded_range_m is None else folded_range_m,
            truth_velocity_mps if folded_velocity_mps is None else folded_velocity_mps,
            range_limits_km,
            velocity_limits_mps,
        )
        axes.legend(loc="upper right", framealpha=0.85, fontsize="small")
    if title is not None:
        axes.set_title(title)

    return figure


def _mark_truth(
    axes: Any,
    truth_range_m: float,
    truth_velocity_mps: float,
    folded_range_m: float,
    folded_velocity_mps: float,
    range_limits_km: tuple[float, float],
    velocity_limits_mps: tuple[float, float],
) -> None:
    """Draw the folded and unfolded truth, pinning the latter to an edge if off-plot.

    Kept separate from :func:`render_range_doppler` because it is all display
    bookkeeping -- clamping, arrow placement, label text -- and none of it is
    about the radar.
    """
    truth_range_km = truth_range_m * 1.0e-3
    inside = (
        range_limits_km[0] <= truth_range_km <= range_limits_km[1]
        and velocity_limits_mps[0] <= truth_velocity_mps <= velocity_limits_mps[1]
    )

    axes.plot(
        folded_range_m * 1.0e-3,
        folded_velocity_mps,
        marker="x",
        markersize=10,
        markeredgecolor="red",
        markeredgewidth=1.8,
        linestyle="none",
        label="truth, folded onto this map",
    )

    if inside:
        axes.plot(
            truth_range_km,
            truth_velocity_mps,
            marker="o",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor="white",
            markeredgewidth=1.6,
            linestyle="none",
            label="truth (unfolded)",
        )
        return

    # Off the plot, which is what folding means. A triangle pinned to the edge
    # it left by, pointing that way: short, unambiguous, and it cannot overlap
    # the map the way a long arrow across it does. The real values are in the
    # title, so the marker only has to say "that way".
    clamped_range_km = float(np.clip(truth_range_km, *range_limits_km))
    clamped_velocity_mps = float(np.clip(truth_velocity_mps, *velocity_limits_mps))
    # The label names whichever axis it left by, because that is the axis that
    # folded: saying "+39.3 m/s" on a map whose velocity axis reaches 191 m/s
    # would point at the wrong lesson.
    if truth_velocity_mps > velocity_limits_mps[1]:
        edge_marker, off_scale = "^", f"{truth_velocity_mps:+.1f} m/s"
    elif truth_velocity_mps < velocity_limits_mps[0]:
        edge_marker, off_scale = "v", f"{truth_velocity_mps:+.1f} m/s"
    elif truth_range_km > range_limits_km[1]:
        edge_marker, off_scale = ">", f"{truth_range_km:.2f} km"
    else:
        edge_marker, off_scale = "<", f"{truth_range_km:.2f} km"

    axes.plot(
        clamped_range_km,
        clamped_velocity_mps,
        marker=edge_marker,
        markersize=12,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.8,
        linestyle="none",
        clip_on=False,
        label=f"truth (unfolded) off scale: {off_scale}",
    )
