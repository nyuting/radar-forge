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

from radar_forge.viz.scopes.track_plot import render_range_time_history

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


class TestRangeRatePanel:
    """The optional second panel: range rate against time."""

    def test_a_truth_range_rate_adds_a_panel_below_sharing_time(self):
        time_s, range_m = truth()
        figure = render_range_time_history(
            time_s,
            range_m,
            track_time_s=time_s,
            track_range_m=range_m,
            truth_range_rate_mps=np.full(N_FRAMES, -20.0),
            track_range_rate_mps=np.full(N_FRAMES, -19.0),
        )
        top, bottom = figure.axes
        assert bottom.get_shared_x_axes().joined(top, bottom)
        rates = sorted(float(line.get_ydata()[0]) for line in bottom.get_lines())
        assert rates == [-20.0, -19.0]
        close(figure)

    def test_without_a_truth_range_rate_there_is_one_panel(self):
        time_s, range_m = truth()
        figure = render_range_time_history(time_s, range_m)
        assert len(figure.axes) == 1
        close(figure)

    def test_rejects_a_track_range_rate_without_the_truth(self):
        time_s, range_m = truth()
        with pytest.raises(ValueError, match="truth_range_rate_mps"):
            render_range_time_history(time_s, range_m, track_range_rate_mps=np.zeros(N_FRAMES))


class TestFoldedRange:
    """A range period: one period on the axis, a modulo label, and lines broken at the wrap."""

    PERIOD_M = 6_000.0

    def test_the_axis_spans_one_period_and_says_it_is_modulo(self):
        time_s, range_m = truth()
        figure = render_range_time_history(
            time_s, range_m % self.PERIOD_M, range_period_m=self.PERIOD_M
        )
        (axes,) = figure.axes
        assert axes.get_ylim() == (0.0, 6.0)
        assert "modulo 6.000 km" in axes.get_ylabel()
        close(figure)

    @pytest.mark.parametrize(
        ("period_m", "gaps"), [(PERIOD_M, [False, False, True, False]), (None, [False] * 4)]
    )
    def test_a_line_is_broken_where_it_wraps(self, period_m, gaps):
        """The point after the wrap is left out, so no line crosses the plot.

        Without a period nothing is a wrap, and the line is drawn whole.
        """
        time_s = np.arange(4, dtype=np.float64)
        wrapped_m = np.array([5_900.0, 5_990.0, 80.0, 170.0])
        figure = render_range_time_history(time_s, wrapped_m, range_period_m=period_m)
        (truth_line,) = [line for line in figure.axes[0].get_lines() if line.get_label() == "truth"]
        np.testing.assert_array_equal(np.isnan(np.asarray(truth_line.get_ydata())), gaps)
        close(figure)
