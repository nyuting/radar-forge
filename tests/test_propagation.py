"""Tests for propagation paths, the range equation, and point targets.

Ground truth is analytic: the transit time over the route, the Doppler sign of
the repo-wide convention (rate positive receding, Doppler positive closing),
the radar range equation in both forms, and the moments and distributions of
the Swerling models.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.propagation import (
    PointTarget,
    PropagationPaths,
    bistatic_range_rate_from_doppler_mps,
    dbsm_to_m2,
    doppler_shift_hz,
    line_of_sight_paths,
    m2_to_dbsm,
    path_amplitude_linear,
    propagation_delay_s,
    propagation_phase_rad,
    received_power_w,
    swerling_rcs_m2,
)
from radar_forge.radar import Radar, Receiver, Site, TargetGeometry, Transmitter

DUKE = Site(36.00250, -78.94100, 60.0)
RDU = Site(35.87750, -78.78750, 25.0)
S1_RADAR = Radar(
    Transmitter(9.8e9, 2.0e6, 100.0, 30.0, 1.0e-3, 1.0e3), Receiver(1.0e6, 30.0, 3.0), DUKE
)
S1_PAIR = Radar(S1_RADAR.transmitter, S1_RADAR.receiver, tx_site=RDU, rx_site=DUKE)
WAVELENGTH_M = S1_RADAR.wavelength_m

# A nominal 77 GHz automotive front end, with the receive gain 3 dB below the
# transmit gain so that a formula squaring one gain cannot pass.
NOMINAL = {
    "transmit_power_w": 0.01,
    "gain_tx_linear": 10 ** (15.0 / 10.0),
    "gain_rx_linear": 10 ** (12.0 / 10.0),
    "wavelength_m": 3.896e-3,
    "rcs_m2": 10 ** (5.0 / 10.0),
}


def _bistatic_geometry(range_tx_m: float, range_rx_m: float, rate_mps: float) -> TargetGeometry:
    """A one-target bistatic geometry with distinct departure and arrival angles."""
    return TargetGeometry(
        range_tx_m=np.array([range_tx_m]),
        range_rx_m=np.array([range_rx_m]),
        bistatic_range_m=np.array([range_tx_m + range_rx_m]),
        bistatic_range_rate_mps=np.array([rate_mps]),
        bistatic_angle_rad=np.array([1.0]),
        aod_rad=np.array([[0.5, 0.1]]),
        aoa_rad=np.array([[3.6, 0.2]]),
    )


class TestPathRelations:
    def test_delay_is_the_route_at_the_speed_of_light(self) -> None:
        np.testing.assert_allclose(propagation_delay_s(27.0e3), 27.0e3 / 299_792_458.0, rtol=1e-15)

    def test_one_wavelength_of_route_is_one_cycle_of_lag(self) -> None:
        np.testing.assert_allclose(
            propagation_phase_rad(3.0 * WAVELENGTH_M, WAVELENGTH_M), -6.0 * np.pi, rtol=1e-15
        )

    def test_a_receding_route_has_negative_doppler(self) -> None:
        """The one sign convention: f_D = -dR_b/dt / lambda, so lengthening is negative."""
        np.testing.assert_allclose(
            doppler_shift_hz([160.0, -160.0], WAVELENGTH_M),
            [-160.0 / WAVELENGTH_M, 160.0 / WAVELENGTH_M],
            rtol=1e-15,
        )

    def test_doppler_and_rate_are_inverses(self) -> None:
        rates_mps = np.array([-160.0, 0.0, 37.5, 900.0])
        # A divide then a multiply by the same wavelength: one ulp each way.
        np.testing.assert_allclose(
            bistatic_range_rate_from_doppler_mps(
                doppler_shift_hz(rates_mps, WAVELENGTH_M), WAVELENGTH_M
            ),
            rates_mps,
            rtol=1e-15,
        )

    def test_amplitude_squared_is_the_power_whatever_the_phase(self) -> None:
        power_w = np.array([1.0e-12, 4.0e-15, 0.0])
        phase_rad = np.array([0.3, -2.0e3, 1.0])
        amplitude = path_amplitude_linear(power_w, phase_rad)
        np.testing.assert_allclose(np.abs(amplitude) ** 2, power_w, rtol=1e-14)
        np.testing.assert_allclose(np.angle(amplitude[:1]), phase_rad[:1], rtol=1e-15)


class TestReceivedPower:
    def test_matches_the_bistatic_closed_form(self) -> None:
        """Catches lambda^1, one gain squared, or the ranges summed instead of multiplied."""
        range_tx_m, range_rx_m = 40.0, 55.0
        expected_w = (
            NOMINAL["transmit_power_w"]
            * NOMINAL["gain_tx_linear"]
            * NOMINAL["gain_rx_linear"]
            * NOMINAL["wavelength_m"] ** 2
            * NOMINAL["rcs_m2"]
        ) / ((4 * np.pi) ** 3 * range_tx_m**2 * range_rx_m**2)
        np.testing.assert_allclose(
            received_power_w(**NOMINAL, range_tx_m=range_tx_m, range_rx_m=range_rx_m),
            expected_w,
            rtol=1e-12,
        )

    def test_equal_ranges_are_the_monostatic_r4_form(self) -> None:
        """R_T = R_R = R reproduces Richards' P G G lambda^2 sigma / (4 pi)^3 R^4.

        rtol 1e-12, not equality: R^2 R^2 and R^4 round differently in the last bit.
        """
        range_m = np.array([42.0, 1.2e4])
        monostatic_w = (
            NOMINAL["transmit_power_w"]
            * NOMINAL["gain_tx_linear"]
            * NOMINAL["gain_rx_linear"]
            * NOMINAL["wavelength_m"] ** 2
            * NOMINAL["rcs_m2"]
        ) / ((4 * np.pi) ** 3 * range_m**4)
        np.testing.assert_allclose(
            received_power_w(**NOMINAL, range_tx_m=range_m, range_rx_m=range_m),
            monostatic_w,
            rtol=1e-12,
        )

    def test_doubling_a_monostatic_range_costs_twelve_db(self) -> None:
        power_w = received_power_w(**NOMINAL, range_tx_m=[50.0, 100.0], range_rx_m=[50.0, 100.0])
        np.testing.assert_allclose(power_w[0] / power_w[1], 16.0, rtol=1e-12)

    def test_power_follows_the_range_product_not_the_sum(self) -> None:
        """Swapped ranges and their geometric mean share R_T R_R, so share the power."""
        near_m, far_m = 1.0e3, 1.0e4
        mean_m = np.sqrt(near_m * far_m)
        power_w = received_power_w(
            **NOMINAL, range_tx_m=[near_m, far_m, mean_m], range_rx_m=[far_m, near_m, mean_m]
        )
        np.testing.assert_allclose(power_w[1:], power_w[0], rtol=1e-12)

    def test_loss_attenuates(self) -> None:
        lossless_w = received_power_w(**NOMINAL, range_tx_m=25.0, range_rx_m=25.0)
        lossy_w = received_power_w(**NOMINAL, range_tx_m=25.0, range_rx_m=25.0, loss_linear=10.0)
        np.testing.assert_allclose(lossy_w * 10.0, lossless_w, rtol=1e-12)

    def test_a_zero_cross_section_gives_zero_power_without_raising(self) -> None:
        """sigma = 0 is how a noise-only scene is built; the maths carries it."""
        assert received_power_w(**{**NOMINAL, "rcs_m2": 0.0}, range_tx_m=1e3, range_rx_m=1e3) == 0.0


class TestPropagationPaths:
    def _paths(self, n_paths: int = 3, **overrides: np.ndarray) -> PropagationPaths:
        fields = {
            "delay_s": np.full(n_paths, 1.0e-4),
            "doppler_hz": np.zeros(n_paths),
            "amplitude_linear": np.ones(n_paths, dtype=np.complex128),
            "aoa_rad": np.zeros((n_paths, 2)),
            "aod_rad": np.zeros((n_paths, 2)),
            "n_bounces": np.ones(n_paths, dtype=np.int_),
        }
        return PropagationPaths(**{**fields, **overrides})

    def test_rejects_mismatched_path_counts(self) -> None:
        with pytest.raises(ValueError, match="to match delay_s"):
            self._paths(doppler_hz=np.zeros(2))

    def test_rejects_angles_that_are_not_azimuth_elevation_pairs(self) -> None:
        with pytest.raises(ValueError, match=r"\(azimuth, elevation\)"):
            self._paths(aoa_rad=np.zeros((3, 3)))

    def test_rejects_a_doppler_rate_of_the_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="doppler_rate_hzps"):
            self._paths(doppler_rate_hzps=np.zeros(2))

    def test_doppler_rate_defaults_to_zero_and_is_stored_per_path(self) -> None:
        np.testing.assert_array_equal(self._paths().doppler_rate_hzps, np.zeros(3))
        np.testing.assert_array_equal(
            self._paths(doppler_rate_hzps=np.asarray(5.0)).doppler_rate_hzps, [5.0, 5.0, 5.0]
        )

    def test_delay_and_bistatic_range_are_inverses(self) -> None:
        ranges_m = np.array([20.0e3, 27.0e3, 63.4e3])
        paths = self._paths(delay_s=propagation_delay_s(ranges_m))
        # A divide then a multiply by the same constant: one ulp each way.
        np.testing.assert_allclose(paths.bistatic_range_m, ranges_m, rtol=1e-15)


class TestBistaticRangeAt:
    """How long each route is at slow time t, under stop-and-hop."""

    def test_unchanged_at_time_zero(self) -> None:
        geometry = TargetGeometry.monostatic([10.0e3, 15.0e3], [-80.0, 40.0])
        paths = line_of_sight_paths(S1_RADAR, geometry, 10.0)
        np.testing.assert_array_equal(
            paths.bistatic_range_at_m(0.0, WAVELENGTH_M), paths.bistatic_range_m
        )

    def test_shape_is_slow_time_by_paths(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, TargetGeometry.monostatic([1e4, 2e4, 3e4], 0.0), 10.0)
        assert paths.bistatic_range_at_m(np.arange(5) * 1e-3, WAVELENGTH_M).shape == (5, 3)

    def test_the_slope_is_the_bistatic_range_rate(self) -> None:
        """Catches the Doppler-to-rate sign: a receding route lengthens at +rate, unhalved."""
        paths = line_of_sight_paths(S1_PAIR, _bistatic_geometry(12.0e3, 15.0e3, 160.0), 10.0)
        slow_time_s = np.array([0.0, 0.25, 2.0])
        range_m = paths.bistatic_range_at_m(slow_time_s, WAVELENGTH_M)[:, 0]
        np.testing.assert_allclose(range_m, 27.0e3 + 160.0 * slow_time_s, rtol=1e-12)

    def test_a_doppler_rate_bends_the_route_quadratically(self) -> None:
        """R_b(t) = R_b(0) - lambda (f_D t + fdot t^2 / 2), the docstring's derivation."""
        base = line_of_sight_paths(S1_RADAR, TargetGeometry.monostatic(10.0e3, -40.0), 10.0)
        paths = PropagationPaths(
            base.delay_s,
            base.doppler_hz,
            base.amplitude_linear,
            base.aoa_rad,
            base.aod_rad,
            base.n_bounces,
            doppler_rate_hzps=np.array([300.0]),
        )
        slow_time_s = np.array([0.0, 0.5, 1.0, 3.0])
        expected_m = 20.0e3 - WAVELENGTH_M * (
            base.doppler_hz[0] * slow_time_s + 0.5 * 300.0 * slow_time_s**2
        )
        np.testing.assert_allclose(
            paths.bistatic_range_at_m(slow_time_s, WAVELENGTH_M)[:, 0], expected_m, rtol=1e-12
        )


class TestLineOfSightPaths:
    def test_delay_is_the_two_way_transit_time(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, TargetGeometry.monostatic(15_000.0, 0.0), 10.0)
        np.testing.assert_allclose(paths.delay_s, 2.0 * 15_000.0 / 299_792_458.0, rtol=1e-15)

    def test_a_receding_target_has_positive_rate_and_negative_doppler(self) -> None:
        """End to end from a velocity vector: away from the radar is +rate and -Doppler."""
        geometry = S1_RADAR.geometry(
            [[0.0, 10_000.0, 0.0]] * 2, [[0.0, 80.0, 0.0], [0.0, -80.0, 0.0]]
        )
        paths = line_of_sight_paths(S1_RADAR, geometry, 10.0)
        assert geometry.bistatic_range_rate_mps[0] > 0.0 > paths.doppler_hz[0]
        assert geometry.bistatic_range_rate_mps[1] < 0.0 < paths.doppler_hz[1]
        np.testing.assert_allclose(
            paths.doppler_hz, np.array([-160.0, 160.0]) / WAVELENGTH_M, rtol=1e-12
        )

    def test_amplitude_squared_is_the_received_power(self) -> None:
        """Keeps the cube on the scale of Receiver.noise_power_per_sample_w."""
        paths = line_of_sight_paths(S1_PAIR, _bistatic_geometry(12.0e3, 15.0e3, 0.0), 10.0)
        expected_w = received_power_w(
            transmit_power_w=100.0,
            gain_tx_linear=1e3,
            gain_rx_linear=1e3,
            wavelength_m=WAVELENGTH_M,
            rcs_m2=10.0,
            range_tx_m=12.0e3,
            range_rx_m=15.0e3,
        )
        np.testing.assert_allclose(np.abs(paths.amplitude_linear) ** 2, expected_w, rtol=1e-12)

    def test_supplied_gains_replace_the_boresight_gains(self) -> None:
        """A pattern can attenuate each path: 0.5 and 0.25 linear cost a factor eight."""
        geometry = TargetGeometry.monostatic(10_000.0, 0.0)
        boresight = line_of_sight_paths(S1_RADAR, geometry, 10.0)
        steered = line_of_sight_paths(
            S1_RADAR, geometry, 10.0, gain_tx_linear=500.0, gain_rx_linear=250.0
        )
        np.testing.assert_allclose(
            np.abs(boresight.amplitude_linear) ** 2 / np.abs(steered.amplitude_linear) ** 2,
            8.0,
            rtol=1e-12,
        )

    def test_a_zero_cross_section_target_returns_nothing(self) -> None:
        paths = line_of_sight_paths(S1_RADAR, TargetGeometry.monostatic(10_000.0, -80.0), 0.0)
        assert paths.amplitude_linear[0] == 0.0

    def test_each_path_is_its_own_target(self) -> None:
        """Catches broadcasting that reuses one target's range or rate for all of them."""
        ranges_m = np.array([9_000.0, 12_000.0, 15_000.0])
        rates_mps = np.array([-50.0, 20.0, 0.0])
        paths = line_of_sight_paths(S1_RADAR, TargetGeometry.monostatic(ranges_m, rates_mps), 10.0)
        np.testing.assert_allclose(paths.delay_s, 2.0 * ranges_m / SPEED_OF_LIGHT_MPS, rtol=1e-12)
        np.testing.assert_allclose(
            paths.doppler_hz, -2.0 * rates_mps / WAVELENGTH_M, rtol=1e-12, atol=0.0
        )
        np.testing.assert_array_equal(paths.n_bounces, [1, 1, 1])

    def test_rejects_a_negative_cross_section(self) -> None:
        """A negative sigma is a negative power, whose square root is NaN throughout the cube."""
        geometry = TargetGeometry.monostatic([10_000.0, 12_000.0], 0.0)
        with pytest.raises(ValueError, match="rcs_m2 must be non-negative"):
            line_of_sight_paths(S1_RADAR, geometry, [10.0, -1.0])

    def test_a_bistatic_path_uses_the_whole_route_with_no_factor_of_two(self) -> None:
        paths = line_of_sight_paths(S1_PAIR, _bistatic_geometry(12.0e3, 15.0e3, 160.0), 10.0)
        np.testing.assert_allclose(paths.delay_s, 27.0e3 / SPEED_OF_LIGHT_MPS, rtol=1e-15)
        np.testing.assert_allclose(paths.doppler_hz, -160.0 / WAVELENGTH_M, rtol=1e-15)

    def test_departure_and_arrival_angles_are_kept_apart(self) -> None:
        paths = line_of_sight_paths(S1_PAIR, _bistatic_geometry(12.0e3, 15.0e3, 0.0), 10.0)
        np.testing.assert_array_equal(paths.aod_rad, [[0.5, 0.1]])
        np.testing.assert_array_equal(paths.aoa_rad, [[3.6, 0.2]])


class TestCrossSection:
    def test_twenty_dbsm_is_one_hundred_square_metres(self) -> None:
        """Catches a 20 log10 convention, a sign flip or a natural log: 10, 0.01, e^2 m^2."""
        np.testing.assert_allclose(dbsm_to_m2(20.0), 100.0, rtol=1e-12)

    def test_round_trips(self) -> None:
        rcs_dbsm = np.linspace(-30.0, 40.0, 71)
        np.testing.assert_allclose(m2_to_dbsm(dbsm_to_m2(rcs_dbsm)), rcs_dbsm, rtol=1e-12)

    def test_rejects_a_zero_area_in_decibels(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            m2_to_dbsm(0.0)

    def test_point_target_round_trips_through_dbsm(self) -> None:
        target = PointTarget.from_dbsm(13.0, name="light-aircraft")
        np.testing.assert_allclose(target.rcs_dbsm, 13.0, rtol=1e-12)
        assert target.name == "light-aircraft"

    def test_a_zero_cross_section_target_is_permitted_for_noise_only_runs(self) -> None:
        assert PointTarget(rcs_m2=0.0).rcs_dbsm == float("-inf")

    def test_rejects_a_negative_cross_section(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            PointTarget(rcs_m2=-1.0)


MEAN_RCS_M2 = 10.0
N_DRAWS = 200_000


def _chi_square_interval(degrees_of_freedom: int) -> tuple[float, float]:
    """99.9 % interval of the sample mean of N_DRAWS sigma_bar chi2_k / k draws."""
    total_dof = degrees_of_freedom * N_DRAWS
    lower, upper = stats.chi2.ppf([0.0005, 0.9995], total_dof) / total_dof
    return MEAN_RCS_M2 * lower, MEAN_RCS_M2 * upper


class TestSwerling:
    """Monte Carlo against the models' exact distributions, with fixed seeds."""

    def test_case_zero_is_the_mean_exactly(self) -> None:
        draws = swerling_rcs_m2(MEAN_RCS_M2, 0, (4, 3), rng=np.random.default_rng(1))
        np.testing.assert_array_equal(draws, np.full((4, 3), MEAN_RCS_M2))

    @pytest.mark.parametrize(("case", "degrees_of_freedom"), [(1, 2), (2, 2), (3, 4), (4, 4)])
    def test_sample_mean_lies_in_its_chi_square_interval(
        self, case: int, degrees_of_freedom: int
    ) -> None:
        """The sum of the draws is sigma_bar / k times a chi-square with k N_DRAWS dof."""
        draws = swerling_rcs_m2(MEAN_RCS_M2, case, (N_DRAWS,), rng=np.random.default_rng(20260911))
        lower, upper = _chi_square_interval(degrees_of_freedom)
        assert lower < float(np.mean(draws)) < upper

    @pytest.mark.parametrize(("case", "degrees_of_freedom"), [(1, 2), (3, 4)])
    def test_draws_follow_the_right_chi_square(self, case: int, degrees_of_freedom: int) -> None:
        """KS test against sigma_bar chi2_k / k; the wrong k (2 for 4) fails by far."""
        draws = swerling_rcs_m2(MEAN_RCS_M2, case, (20_000,), rng=np.random.default_rng(20260911))
        scale = MEAN_RCS_M2 / degrees_of_freedom
        assert stats.kstest(draws, stats.chi2(degrees_of_freedom, scale=scale).cdf).pvalue > 1e-3
        wrong = 6 - degrees_of_freedom
        assert stats.kstest(draws, stats.chi2(wrong, scale=MEAN_RCS_M2 / wrong).cdf).pvalue < 1e-6

    def test_cases_three_and_four_fluctuate_half_as_much(self) -> None:
        """Variance sigma_bar^2 against sigma_bar^2 / 2: a ratio of 2.

        Each sample variance has a relative standard error of about 0.6 % at
        2e5 draws, so the ratio is within 5 % at over six sigma.
        """
        rng = np.random.default_rng(20260911)
        exponential = swerling_rcs_m2(MEAN_RCS_M2, 1, (N_DRAWS,), rng=rng)
        chi_square_4 = swerling_rcs_m2(MEAN_RCS_M2, 3, (N_DRAWS,), rng=rng)
        np.testing.assert_allclose(np.var(exponential) / np.var(chi_square_4), 2.0, rtol=0.05)

    def test_a_per_target_mean_is_a_column_against_targets_by_pulses(self) -> None:
        """Two targets, three pulses: an (n_targets, 1) mean scales each target's own row.

        The same seed draws the same chi-square variates, so each row must be its
        target's mean times the unit-mean draws; a mean paired with the pulse
        axis instead would scale columns.
        """
        mean_rcs_m2 = np.array([[1.0], [100.0]])
        draws = swerling_rcs_m2(mean_rcs_m2, 2, (2, 3), rng=np.random.default_rng(3))
        unit = swerling_rcs_m2(1.0, 2, (2, 3), rng=np.random.default_rng(3))
        assert draws.shape == (2, 3)
        # One multiply and one divide in a different order.
        np.testing.assert_allclose(draws, mean_rcs_m2 * unit, rtol=1e-15)

    def test_rejects_an_unknown_case(self) -> None:
        with pytest.raises(ValueError, match="Swerling case"):
            swerling_rcs_m2(MEAN_RCS_M2, 5, (3,), rng=np.random.default_rng(1))
