"""Tests for the teaching plotting helpers.

matplotlib is an optional extra, so the drawing tests skip without it while
the decibel maths and the missing-extra error are checked unconditionally --
the second of those is only meaningful in an environment that does not have
the extra, which is exactly the environment the skip protects.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

from radar_forge.teaching.plotting import magnitude_db, require_pyplot


class TestMagnitudeDb:
    def test_the_peak_is_zero_decibels_by_default(self) -> None:
        values = np.array([0.25, 1.0, 0.5])
        np.testing.assert_allclose(magnitude_db(values)[1], 0.0, atol=1e-15)

    def test_halving_the_amplitude_costs_six_decibels(self) -> None:
        """An amplitude ratio, so the factor is 20 -- not 10."""
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

    def test_a_zero_cell_is_clamped_to_the_floor(self) -> None:
        """Without the floor an empty cell is -inf and the colour scale dies."""
        result = magnitude_db(np.array([1.0, 0.0]), floor_db=-90.0)
        assert result[1] == -90.0
        assert np.all(np.isfinite(result))

    def test_an_all_zero_array_is_all_floor(self) -> None:
        result = magnitude_db(np.zeros(4), floor_db=-90.0)
        np.testing.assert_array_equal(result, np.full(4, -90.0))

    def test_preserves_shape(self) -> None:
        assert magnitude_db(np.ones((3, 5))).shape == (3, 5)

    @pytest.mark.parametrize("bad_reference_linear", [0.0, -1.0])
    def test_rejects_a_non_positive_reference(self, bad_reference_linear: float) -> None:
        with pytest.raises(ValueError, match="reference_linear"):
            magnitude_db(np.ones(3), reference_linear=bad_reference_linear)


class TestRequirePyplot:
    def test_names_the_extra_when_matplotlib_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare ModuleNotFoundError leaves the reader guessing which extra."""
        real_import = builtins.__import__

        def fake_import(
            name: str,
            globals_: Any = None,
            locals_: Any = None,
            fromlist: Sequence[str] = (),
            level: int = 0,
        ) -> Any:
            if name.startswith("matplotlib"):
                msg = "No module named 'matplotlib'"
                raise ImportError(msg)
            return real_import(name, globals_, locals_, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="teaching"):
            require_pyplot()

    def test_the_message_gives_a_command_to_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import: Callable[..., Any] = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("matplotlib"):
                msg = "No module named 'matplotlib'"
                raise ImportError(msg)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="uv sync --extra teaching"):
            require_pyplot()

    def test_returns_pyplot_when_the_extra_is_present(self) -> None:
        pytest.importorskip("matplotlib")
        assert hasattr(require_pyplot(), "subplots")
