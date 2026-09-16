"""Tests for the range-Doppler scope.

The scope is checked for the things a wrong plot would get wrong silently:
which axis is which, that the axes come from the dsp helpers rather than being
re-derived, and that a mismatched axis is refused rather than broadcast into
something plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.dsp import doppler_bin_centers_mps, range_bin_centers_m
from radar_forge.teaching.scopes.rd_map import render_range_doppler

pytest.importorskip("matplotlib")

N_DOPPLER_BINS = 32
N_RANGE_BINS = 64


def _axes() -> tuple[np.ndarray, np.ndarray]:
    """S1-like axes from the dsp helpers, per the scenario spec S7.1."""
    range_axis_m = range_bin_centers_m(N_RANGE_BINS, 2.0e6, 1.0e-3, 1.0e6)
    velocity_axis_mps = doppler_bin_centers_mps(N_DOPPLER_BINS, 1.0e-3, 0.030591)
    return range_axis_m, velocity_axis_mps


def _map_with_peak_at(doppler_bin: int, range_bin: int) -> np.ndarray:
    rd_map = np.full((N_DOPPLER_BINS, N_RANGE_BINS), 1.0e-6, dtype=np.complex128)
    rd_map[doppler_bin, range_bin] = 1.0
    return rd_map


class TestRenderRangeDoppler:
    def test_returns_a_figure(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(_map_with_peak_at(16, 20), range_axis_m, velocity_axis_mps)
        assert hasattr(figure, "savefig")

    def test_range_is_on_the_horizontal_axis_in_kilometres(self) -> None:
        """Getting this backwards produces a plot that still looks like a radar."""
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(_map_with_peak_at(16, 20), range_axis_m, velocity_axis_mps)
        axes = figure.axes[0]
        assert "Range" in axes.get_xlabel()
        np.testing.assert_allclose(
            axes.get_xlim(), (range_axis_m[0] * 1e-3, range_axis_m[-1] * 1e-3), rtol=1e-12
        )

    def test_velocity_is_on_the_vertical_axis_and_says_which_sign_closes(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(_map_with_peak_at(16, 20), range_axis_m, velocity_axis_mps)
        axes = figure.axes[0]
        assert "closing" in axes.get_ylabel()
        np.testing.assert_allclose(
            axes.get_ylim(), (velocity_axis_mps[0], velocity_axis_mps[-1]), rtol=1e-12
        )

    def test_an_on_scale_truth_is_drawn_where_it_was_asked_for(self) -> None:
        """Nothing folded, so both markers coincide at the true position."""
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(
            _map_with_peak_at(16, 20),
            range_axis_m,
            velocity_axis_mps,
            truth_range_m=10_000.0,
            truth_velocity_mps=3.0,
        )
        for line in figure.axes[0].get_lines():
            np.testing.assert_allclose(line.get_xdata(), [10.0], rtol=1e-12)
            np.testing.assert_allclose(line.get_ydata(), [3.0], rtol=1e-12)

    def test_the_folded_marker_goes_where_the_map_puts_the_target(self) -> None:
        """The cross must sit on the peak even when the truth is far away."""
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(
            _map_with_peak_at(16, 20),
            range_axis_m,
            velocity_axis_mps,
            truth_range_m=10_000.0,
            truth_velocity_mps=80.0,
            folded_range_m=10_000.0,
            folded_velocity_mps=-2.5,
        )
        (folded,) = [line for line in figure.axes[0].get_lines() if line.get_marker() == "x"]
        np.testing.assert_allclose(folded.get_xdata(), [10.0], rtol=1e-12)
        np.testing.assert_allclose(folded.get_ydata(), [-2.5], rtol=1e-12)

    def test_an_off_scale_truth_is_pinned_to_the_edge_it_left_by(self) -> None:
        """A marker that silently vanishes leaves the map looking unambiguous."""
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(
            _map_with_peak_at(16, 20),
            range_axis_m,
            velocity_axis_mps,
            truth_range_m=10_000.0,
            truth_velocity_mps=80.0,
            folded_velocity_mps=-2.5,
        )
        (edge,) = [line for line in figure.axes[0].get_lines() if line.get_marker() == "^"]
        np.testing.assert_allclose(edge.get_ydata(), [velocity_axis_mps[-1]], rtol=1e-12)
        np.testing.assert_allclose(edge.get_xdata(), [10.0], rtol=1e-12)
        assert "m/s" in edge.get_label()

    def test_an_off_scale_range_is_labelled_in_kilometres_not_velocity(self) -> None:
        """S2 folds in range; naming the velocity there points at the wrong lesson."""
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(
            _map_with_peak_at(16, 20),
            range_axis_m,
            velocity_axis_mps,
            truth_range_m=float(range_axis_m[-1]) * 3.0,
            truth_velocity_mps=1.0,
            folded_range_m=10_000.0,
        )
        (edge,) = [line for line in figure.axes[0].get_lines() if line.get_marker() == ">"]
        np.testing.assert_allclose(edge.get_xdata(), [range_axis_m[-1] * 1e-3], rtol=1e-12)
        assert "km" in edge.get_label()
        assert "m/s" not in edge.get_label()

    def test_a_target_below_the_velocity_axis_points_down(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(
            _map_with_peak_at(16, 20),
            range_axis_m,
            velocity_axis_mps,
            truth_range_m=10_000.0,
            truth_velocity_mps=-80.0,
        )
        assert any(line.get_marker() == "v" for line in figure.axes[0].get_lines())

    def test_no_marker_without_a_truth(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(_map_with_peak_at(16, 20), range_axis_m, velocity_axis_mps)
        assert figure.axes[0].get_lines() == []

    def test_rejects_axes_that_do_not_match_the_map(self) -> None:
        range_axis_m, velocity_axis_mps = _axes()
        with pytest.raises(ValueError, match="axes do not match"):
            render_range_doppler(_map_with_peak_at(16, 20), range_axis_m[:-1], velocity_axis_mps)

    def test_rejects_a_transposed_map(self) -> None:
        """Range on axis 1, Doppler on axis 0 -- the dsp layout, not a choice."""
        range_axis_m, velocity_axis_mps = _axes()
        with pytest.raises(ValueError, match="Range is axis 1"):
            render_range_doppler(_map_with_peak_at(16, 20).T, range_axis_m, velocity_axis_mps)

    def test_rejects_a_one_dimensional_map(self) -> None:
        range_axis_m, velocity_axis_mps = _axes()
        with pytest.raises(ValueError, match="two-dimensional"):
            render_range_doppler(np.zeros(10), range_axis_m, velocity_axis_mps)

    @pytest.mark.parametrize("bad_dynamic_range_db", [0.0, -10.0])
    def test_rejects_a_non_positive_dynamic_range(self, bad_dynamic_range_db: float) -> None:
        range_axis_m, velocity_axis_mps = _axes()
        with pytest.raises(ValueError, match="dynamic_range_db"):
            render_range_doppler(
                _map_with_peak_at(16, 20),
                range_axis_m,
                velocity_axis_mps,
                dynamic_range_db=bad_dynamic_range_db,
            )


class TestBistaticLabelling:
    """The one thing a bistatic map changes: what the axes are called."""

    @staticmethod
    def _figure(*, bistatic: bool) -> object:
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        return render_range_doppler(
            _map_with_peak_at(16, 20), range_axis_m, velocity_axis_mps, bistatic=bistatic
        )

    def test_the_default_labels_are_monostatic(self) -> None:
        axes = self._figure(bistatic=False).axes[0]
        assert axes.get_xlabel() == "Range (km)"
        assert "Radial velocity" in axes.get_ylabel()

    def test_the_bistatic_flag_renames_both_axes(self) -> None:
        """A bistatic range axis read as a slant range is read wrong."""
        axes = self._figure(bistatic=True).axes[0]
        assert axes.get_xlabel() == "Bistatic mean range (km)"
        assert "Bisector range rate" in axes.get_ylabel()

    def test_the_flag_changes_nothing_but_the_labels(self) -> None:
        """The data is identical either way; only its description differs. D6."""
        monostatic = self._figure(bistatic=False).axes[0]
        bistatic = self._figure(bistatic=True).axes[0]
        assert monostatic.get_xlim() == bistatic.get_xlim()
        assert monostatic.get_ylim() == bistatic.get_ylim()


def _close(figure) -> None:
    """Release a figure, so a long test module does not hold them all open."""
    import matplotlib.pyplot as plt

    plt.close(figure)


class TestTrackingOverlays:
    """The additive overlays scenario 003 draws on top of the map.

    Every one defaults to None, because scenario 001 and scenario 002 call this
    function without them and must keep producing the same figure.
    """

    def test_the_overlays_are_absent_by_default(self):
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(_map_with_peak_at(4, 8), range_axis_m, velocity_axis_mps)
        axes = figure.axes[0]
        labels = {line.get_label() for line in axes.lines}
        assert "track estimate" not in labels
        assert "validation gate" not in labels
        assert not axes.collections or all(
            collection.get_label() != "CFAR detections" for collection in axes.collections
        )
        _close(figure)

    def test_detections_are_drawn(self):
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(
            _map_with_peak_at(4, 8),
            range_axis_m,
            velocity_axis_mps,
            detections=[(float(range_axis_m[8]), float(velocity_axis_mps[4]))],
        )
        axes = figure.axes[0]
        assert any(collection.get_label() == "CFAR detections" for collection in axes.collections)
        _close(figure)

    def test_the_track_estimate_and_its_gate_are_drawn(self):
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(
            _map_with_peak_at(4, 8),
            range_axis_m,
            velocity_axis_mps,
            track_estimate=(float(range_axis_m[8]), float(velocity_axis_mps[4])),
            gate_extent=(200.0, 1.0),
        )
        labels = {line.get_label() for line in figure.axes[0].lines}
        assert "track estimate" in labels
        assert "validation gate" in labels
        _close(figure)

    def test_a_gate_without_an_estimate_is_ignored(self):
        """There is nothing to centre it on, so drawing it would be a lie."""
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(
            _map_with_peak_at(4, 8),
            range_axis_m,
            velocity_axis_mps,
            gate_extent=(200.0, 1.0),
        )
        labels = {line.get_label() for line in figure.axes[0].lines}
        assert "validation gate" not in labels
        _close(figure)


class TestRenderRangeDopplerEdgeCases:
    """Optional overlays and off-scale truths, which a run meets every frame."""

    def test_a_title_is_placed_on_the_axes(self) -> None:
        """A frame-numbered title is how a run's figures stay distinguishable."""
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(
            _map_with_peak_at(16, 20), range_axis_m, velocity_axis_mps, title="frame 007"
        )
        assert figure.axes[0].get_title() == "frame 007"

    def test_a_track_estimate_without_a_gate_is_still_drawn(self) -> None:
        """The gate is optional: a track that has one is drawn with it, one that
        has none is drawn without, and neither case may raise.

        A frame before the bootstrap completes has an estimate and no gate to
        report, so this is the ordinary early-run state, not a corner case.
        """
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(
            _map_with_peak_at(16, 20),
            range_axis_m,
            velocity_axis_mps,
            track_estimate=(float(range_axis_m[20]), float(velocity_axis_mps[16])),
        )
        assert figure.axes[0].legend_ is not None

    def test_a_truth_left_of_the_range_axis_is_pinned_to_that_edge(self) -> None:
        """Each off-scale direction gets its own marker, including short range.

        A target closer than the first range bin points left; the three other
        directions are covered above. The label reports the range in km, not
        the velocity, because range is the axis it left.
        """
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(
            _map_with_peak_at(16, 20),
            range_axis_m,
            velocity_axis_mps,
            truth_range_m=float(range_axis_m[0]) - 5_000.0,
            truth_velocity_mps=float(velocity_axis_mps[16]),
        )
        (edge,) = [line for line in figure.axes[0].get_lines() if line.get_marker() == "<"]
        np.testing.assert_allclose(
            edge.get_xdata(), [range_axis_m[0] * 1e-3], rtol=1e-12, atol=1e-12
        )
        assert "km" in edge.get_label()
        assert "m/s" not in edge.get_label()
