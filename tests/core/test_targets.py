"""Tests for point targets and cross-section unit conversion."""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.targets import PointTarget, dbsm_to_m2, m2_to_dbsm


class TestCrossSectionConversion:
    def test_twenty_dbsm_is_one_hundred_square_metres(self) -> None:
        """Catches a 20 log10 (amplitude) convention, a sign flip or a natural log.

        Those give 10, 0.01 and e^2 m^2. One decade point suffices: the
        conversion is a single power, so the round trip below covers the rest.
        """
        # rtol 1e-12: one power of ten in float64.
        np.testing.assert_allclose(dbsm_to_m2(20.0), 100.0, rtol=1e-12)

    def test_round_trips(self) -> None:
        rcs_dbsm = np.linspace(-30.0, 40.0, 71)
        np.testing.assert_allclose(m2_to_dbsm(dbsm_to_m2(rcs_dbsm)), rcs_dbsm, rtol=1e-12)

    def test_rejects_a_zero_area_in_decibels(self) -> None:
        """Zero is the boundary: a ``< 0`` guard would let it through to ``log10``."""
        with pytest.raises(ValueError, match="strictly positive"):
            m2_to_dbsm(0.0)


class TestPointTarget:
    def test_from_dbsm_matches_the_linear_constructor(self) -> None:
        np.testing.assert_allclose(PointTarget.from_dbsm(10.0).rcs_m2, 10.0, rtol=1e-12)

    def test_dbsm_property_round_trips(self) -> None:
        target = PointTarget.from_dbsm(13.0, name="light-aircraft")
        np.testing.assert_allclose(target.rcs_dbsm, 13.0, rtol=1e-12)
        assert target.name == "light-aircraft"

    def test_a_zero_cross_section_target_is_permitted_for_noise_only_runs(self) -> None:
        target = PointTarget(rcs_m2=0.0)
        assert target.rcs_m2 == 0.0
        assert target.rcs_dbsm == float("-inf")

    def test_rejects_a_negative_cross_section(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            PointTarget(rcs_m2=-1.0)

    def test_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            PointTarget(10.0).rcs_m2 = 20.0  # type: ignore[misc]  # frozen
