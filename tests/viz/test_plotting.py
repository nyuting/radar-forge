"""Tests for the viz plotting helpers.

matplotlib is an optional extra, so the drawing tests skip without it while
the decibel maths and the missing-extra error are checked unconditionally --
the second of those is only meaningful in an environment that does not have
the extra, which is exactly the environment the skip protects.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from radar_forge.viz.plotting import magnitude_db, require_pyplot


class TestMagnitudeDb:
    def test_halving_the_amplitude_costs_six_decibels(self) -> None:
        """An amplitude ratio, so the factor is 20 -- not 10.

        Uses the default reference, so it also catches a reference other than
        the peak: the mean, 0.75 here, would put the cell at -3.5 dB.
        """
        np.testing.assert_allclose(
            magnitude_db(np.array([1.0, 0.5]))[1], -6.020599913279624, rtol=1e-12
        )

    def test_a_fixed_reference_holds_the_scale(self) -> None:
        """What a run uses to keep successive frames on one colour scale."""
        np.testing.assert_allclose(
            magnitude_db(np.array([10.0]), reference_linear=1.0), 20.0, rtol=1e-12
        )

    def test_uses_complex_magnitude(self) -> None:
        np.testing.assert_allclose(
            magnitude_db(np.array([3.0 + 4.0j]), reference_linear=5.0), 0.0, atol=1e-15
        )

    def test_an_all_zero_array_is_all_floor(self) -> None:
        """Without the floor an empty cell is -inf and the colour scale dies.

        All zero also takes the no-peak branch, whose reference falls back to 1.
        """
        result = magnitude_db(np.zeros(4), floor_db=-90.0)
        np.testing.assert_array_equal(result, np.full(4, -90.0))

    def test_rejects_a_zero_reference(self) -> None:
        """Zero, not a negative: it catches a ``< 0`` guard that a negative would not."""
        with pytest.raises(ValueError, match="reference_linear"):
            magnitude_db(np.ones(3), reference_linear=0.0)


class TestRequirePyplot:
    def test_the_message_gives_a_command_to_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bare ModuleNotFoundError leaves the reader guessing which extra to install."""
        real_import: Callable[..., Any] = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("matplotlib"):
                msg = "No module named 'matplotlib'"
                raise ImportError(msg)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="uv sync --extra viz"):
            require_pyplot()


class TestSaveFigure:
    def test_writes_the_file_and_closes_the_figure(self, tmp_path) -> None:
        """Closing is the contract: a 16,500-frame run must not keep its figures.

        matplotlib holds every unclosed figure alive, so a scenario run that
        writes one figure per frame exhausts memory long before the end. The
        test asserts the file exists *and* that the figure has left
        matplotlib's registry, which is the half that silently regresses.
        """
        plt = pytest.importorskip("matplotlib.pyplot")
        import matplotlib

        matplotlib.use("Agg")
        from radar_forge.viz.plotting import save_figure

        figure = plt.figure()
        destination = tmp_path / "frame.png"
        save_figure(figure, destination)

        assert destination.is_file()
        assert destination.stat().st_size > 0
        assert not plt.fignum_exists(figure.number)
