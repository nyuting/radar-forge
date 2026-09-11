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

    def test_the_truth_marker_is_drawn_where_it_was_asked_for(self) -> None:
        """Unfolded truth on a folded map: the marker may sit off the peak."""
        import matplotlib

        matplotlib.use("Agg")
        range_axis_m, velocity_axis_mps = _axes()
        figure = render_range_doppler(
            _map_with_peak_at(16, 20),
            range_axis_m,
            velocity_axis_mps,
            truth_range_m=10_000.0,
            truth_velocity_mps=80.0,
        )
        (line,) = figure.axes[0].get_lines()
        np.testing.assert_allclose(line.get_xdata(), [10.0], rtol=1e-12)
        np.testing.assert_allclose(line.get_ydata(), [80.0], rtol=1e-12)

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
