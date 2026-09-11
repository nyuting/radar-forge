"""Tests for point targets and cross-section unit conversion."""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.core.targets import PointTarget, dbsm_to_m2, m2_to_dbsm


class TestCrossSectionConversion:
    @pytest.mark.parametrize(
        ("rcs_dbsm", "expected_m2"),
        [(0.0, 1.0), (10.0, 10.0), (20.0, 100.0), (-10.0, 0.1), (30.0, 1000.0)],
        ids=["0dBsm", "10dBsm", "20dBsm", "-10dBsm", "30dBsm"],
    )
    def test_decade_values_are_exact_powers_of_ten(
        self, rcs_dbsm: float, expected_m2: float
    ) -> None:
        np.testing.assert_allclose(dbsm_to_m2(rcs_dbsm), expected_m2, rtol=1e-12)

    def test_round_trips(self) -> None:
        rcs_dbsm = np.linspace(-30.0, 40.0, 71)
        np.testing.assert_allclose(m2_to_dbsm(dbsm_to_m2(rcs_dbsm)), rcs_dbsm, rtol=1e-12)

    def test_three_db_is_a_factor_of_two(self) -> None:
        np.testing.assert_allclose(dbsm_to_m2(3.0103), 2.0, rtol=1e-5)

    def test_vectorises(self) -> None:
        assert dbsm_to_m2(np.zeros((3, 4))).shape == (3, 4)

    @pytest.mark.parametrize("bad_rcs_m2", [0.0, -1.0])
    def test_rejects_non_positive_area_in_decibels(self, bad_rcs_m2: float) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            m2_to_dbsm(bad_rcs_m2)


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

    def test_swerling_fluctuation_is_not_implemented(self) -> None:
        """Guards the scope decision in the module docstring.

        Swerling 1-4 are deliberately absent rather than stubbed. If someone
        adds them, this test should be deleted in the same commit — not left
        passing against a function whose body does nothing.
        """
        import radar_forge.core.targets as targets_module

        assert not hasattr(targets_module, "swerling_rcs_m2")
        assert set(targets_module.__all__) == {"PointTarget", "dbsm_to_m2", "m2_to_dbsm"}
