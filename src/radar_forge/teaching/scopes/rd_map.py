"""The range-Doppler scope: the picture scenario 001 exists to produce.

The display is deliberately labelled from
:func:`radar_forge.core.dsp.range_bin_centers_m` and
:func:`~radar_forge.core.dsp.doppler_bin_centers_mps` rather than from axes
derived here. Those helpers already encode the asymmetry that catches people
out -- range is unshifted, so zero range is bin 0, while Doppler is
``fftshift``-ed, so zero velocity is the middle bin -- and they encode each
waveform's unambiguous limits. Re-deriving an axis is how a plot ends up
mirrored or offset by half a span while still looking entirely reasonable.

The truth marker is drawn at the **unfolded** truth, on a map that may be
folded. When the marker sits away from the peak the plot is not wrong: that
gap is the measurement the scenario is teaching. For S1 the marker walks off
the velocity axis while the peak stays put; for S2 it walks off in range.

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
    title: str | None = None,
    dynamic_range_db: float = _DEFAULT_DYNAMIC_RANGE_DB,
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
        The true, **unfolded** target position. Drawn as a marker if both are
        given. It is expected to sit away from the peak whenever the waveform
        folds; see the module docstring.
    title : str, optional
        Figure title.
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

    if truth_range_m is not None and truth_velocity_mps is not None:
        axes.plot(
            truth_range_m * 1.0e-3,
            truth_velocity_mps,
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=1.6,
            linestyle="none",
            label="truth (unfolded)",
        )
        axes.legend(loc="upper right", framealpha=0.8)

    axes.set_xlabel("Range (km)")
    axes.set_ylabel("Radial velocity (m/s, positive closing)")
    axes.set_xlim(range_m[0] * 1.0e-3, range_m[-1] * 1.0e-3)
    axes.set_ylim(velocity_mps[0], velocity_mps[-1])
    if title is not None:
        axes.set_title(title)

    return figure
