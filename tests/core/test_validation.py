"""Tests for the shared argument checks in ``radar_forge.core``."""

from __future__ import annotations

import pytest

from radar_forge.core._validation import _normalize_axis


class TestNormalizeAxis:
    def test_minus_one_is_the_last_axis(self) -> None:
        assert _normalize_axis(-1, 3) == 2

    def test_non_negative_axis_is_unchanged(self) -> None:
        assert _normalize_axis(0, 3) == 0
        assert _normalize_axis(2, 3) == 2

    @pytest.mark.parametrize("axis", [3, -4])
    def test_axis_outside_the_array_raises(self, axis: int) -> None:
        with pytest.raises(ValueError, match="out of bounds"):
            _normalize_axis(axis, 3)
