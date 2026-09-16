"""Tests for the range-time track scope.

A plot cannot be checked for looking right, so it is checked for the things a
wrong one would get wrong silently: which series is which, that kilometres are
kilometres, that an uncertainty band is drawn around the track and not the
truth, and that half a pair of arguments is refused rather than quietly
ignored.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.teaching.scopes.track_plot import render_range_time_history

pytest.importorskip("matplotlib")

N_FRAMES = 20


def truth():
    """A target opening steadily, so the truth line has an unmistakable slope."""
    time_s = np.arange(N_FRAMES, dtype=np.float64)
    range_m = 15_000.0 + 20.0 * time_s
    return time_s, range_m


def close(figure):
    import matplotlib.pyplot as plt

    plt.close(figure)


class TestRendering:
    """What the figure contains."""

    def test_truth_alone_renders(self):
        time_s, range_m = truth()
        figure = render_range_time_history(time_s, range_m)
        (axes,) = figure.axes
        assert len(axes.lines) >= 1
        close(figure)

    def test_the_y_axis_is_kilometres(self):
        """Ranges are metres in, kilometres out; a factor of 1000 is easy to lose."""
        time_s, range_m = truth()
        figure = render_range_time_history(time_s, range_m)
        (axes,) = figure.axes
        plotted = axes.lines[0].get_ydata()
        np.testing.assert_allclose(plotted, range_m * 1e-3, rtol=1e-12)
        assert "km" in axes.get_ylabel()
        close(figure)

    def test_the_x_axis_is_seconds(self):
        time_s, range_m = truth()
        figure = render_range_time_history(time_s, range_m)
        (axes,) = figure.axes
        np.testing.assert_allclose(axes.lines[0].get_xdata(), time_s, rtol=1e-12)
        assert "Time" in axes.get_xlabel()
        close(figure)

    def test_detections_are_drawn_as_a_scatter(self):
        time_s, range_m = truth()
        figure = render_range_time_history(
            time_s,
            range_m,
            detection_time_s=time_s,
            detection_range_m=range_m + 30.0,
        )
        (axes,) = figure.axes
        assert len(axes.collections) >= 1
        close(figure)

    def test_the_track_is_a_second_line(self):
        time_s, range_m = truth()
        figure = render_range_time_history(
            time_s, range_m, track_time_s=time_s, track_range_m=range_m - 10.0
        )
        (axes,) = figure.axes
        assert len(axes.lines) >= 2
        labels = [line.get_label() for line in axes.lines]
        assert "truth" in labels
        assert "track" in labels
        close(figure)

    def test_the_uncertainty_band_follows_the_track_not_the_truth(self):
        """A band drawn around the truth would hide exactly the error it exists
        to show: a filter whose variance shrinks while its estimate drifts."""
        time_s, range_m = truth()
        offset_m = 500.0
        figure = render_range_time_history(
            time_s,
            range_m,
            track_time_s=time_s,
            track_range_m=range_m - offset_m,
            track_sigma_range_m=np.full(N_FRAMES, 25.0),
        )
        (axes,) = figure.axes
        assert axes.collections, "no filled band was drawn"
        band = axes.collections[0].get_paths()[0].vertices[:, 1]
        # The band straddles the track, which is 500 m below the truth.
        assert band.min() == pytest.approx((range_m[0] - offset_m - 25.0) * 1e-3, rel=1e-6)
        close(figure)

    def test_the_current_frame_is_marked(self):
        time_s, range_m = truth()
        figure = render_range_time_history(time_s, range_m, current_time_s=7.0)
        (axes,) = figure.axes
        verticals = [line for line in axes.lines if line.get_linestyle() == ":"]
        assert verticals
        close(figure)

    def test_the_title_is_used(self):
        time_s, range_m = truth()
        figure = render_range_time_history(time_s, range_m, title="frame 3")
        assert figure.axes[0].get_title() == "frame 3"
        close(figure)


class TestValidation:
    """The documented failure modes."""

    def test_rejects_mismatched_truth_arrays(self):
        with pytest.raises(ValueError, match="same"):
            render_range_time_history(np.arange(5.0), np.arange(4.0))

    def test_rejects_a_two_dimensional_truth(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            render_range_time_history(np.zeros((2, 3)), np.zeros((2, 3)))

    @pytest.mark.parametrize("name", ["detection", "track"])
    def test_rejects_half_a_pair(self, name):
        """Silently drawing nothing is how a missing series goes unnoticed."""
        time_s, range_m = truth()
        with pytest.raises(ValueError, match=f"{name}_time_s"):
            render_range_time_history(time_s, range_m, **{f"{name}_time_s": time_s})


def test_off_scale_detections_are_counted_in_a_corner_note() -> None:
    """A detection outside the range window is annotated, not silently dropped.

    The scope scales to the truth, so a false alarm at an implausible range
    would otherwise vanish from the figure entirely and the reader would
    conclude the frame had no clutter in it.
    """
    import matplotlib

    matplotlib.use("Agg")
    truth_time_s = np.linspace(0.0, 10.0, 11)
    truth_range_m = np.full(truth_time_s.shape, 20_000.0)
    figure = render_range_time_history(
        truth_time_s=truth_time_s,
        truth_range_m=truth_range_m,
        detection_time_s=np.array([5.0]),
        detection_range_m=np.array([500_000.0]),
    )
    notes = [text.get_text() for text in figure.axes[0].texts]
    assert any("off-scale" in note for note in notes)
