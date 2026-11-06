"""Tests for radar_forge.tracking.rfs: the PMBM tracker against the paper's update, by hand.

Ground truth is analytic: the single-Bernoulli case is the Kalman filter (spec 011b A10); a
first detection's existence, a missed Bernoulli's and two tracks' global weights are the
update of García-Fernández et al. (2018), (21)-(24), (30), (31) and (35), written out here
from the textbook Gaussian; the birth limit is the one every tracker shares.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from radar_forge.tracking.filters import TrackStatus
from radar_forge.tracking.initiation import DirectStateInitiator
from radar_forge.tracking.models import CartesianMotion, LinearMeasurement
from radar_forge.tracking.rfs import PmbmTracker
from radar_forge.tracking.tracker import Scan

P_G = 0.99
P_S = 0.995
# Process noise q = 2 sigma^2 tau = 1 m^2/s^3, measured position variance R = 4 m^2.
Q_M2PS3, R_M2 = 1.0, 4.0


def _pmbm(**settings: object) -> PmbmTracker:
    """PMBM on position along x over a 1000 m coverage."""
    motion = CartesianMotion("x", sigma_acceleration_mps2=1.0, acceleration_correlation_time_s=0.5)
    options: dict[str, object] = {
        "detection_probability": 0.9,
        "gate_probability": P_G,
        "coverage_volume": 1000.0,
        **settings,
    }
    return PmbmTracker(
        motion,
        LinearMeasurement(motion.state_layout, ("x_m",)),
        DirectStateInitiator([0.0, 0.0], np.diag([1.0, 50.0**2])),
        **options,  # type: ignore[arg-type]
    )


def _scan(time_s: float, positions_m: list[float], clutter: float = 0.01) -> Scan:
    z = np.reshape(positions_m, (-1, 1))
    return Scan(time_s, z, [[R_M2]], "x", np.full(len(z), clutter))


def _gaussian(x: float, variance: float) -> float:
    return math.exp(-0.5 * x**2 / variance) / math.sqrt(2 * math.pi * variance)


def test_one_target_seen_every_scan_with_no_clutter_or_birth_is_the_kalman_filter() -> None:
    """A10: one Bernoulli, Pd = 1, clutter 1e-300 and no birth after scan 0 is the KF.

    The KF is written out here: F = [[1, T], [0, 1]], Q = q [[T³/3, T²/2], [T²/2, T]], H =
    [1, 0]. rtol 1e-9: the same arithmetic, the filter's in the Joseph form.
    """
    tracker = _pmbm(detection_probability=1.0, birth_intensity_per_scan=0.0)
    mean, covariance = np.array([0.0, 20.0]), np.diag([100.0, 25.0])
    tracker.seed(mean[None], covariance[None], 0.0)
    rng = np.random.default_rng(20261106)
    period_s = 2.0
    transition = np.array([[1.0, period_s], [0.0, 1.0]])
    noise = Q_M2PS3 * np.array([[period_s**3 / 3, period_s**2 / 2], [period_s**2 / 2, period_s]])
    for scan in range(1, 31):
        time_s = scan * period_s
        z = 20.0 * time_s + rng.normal(0.0, math.sqrt(R_M2))
        bank = tracker.process(_scan(time_s, [z], clutter=1e-300))
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T + noise
        innovation_variance = covariance[0, 0] + R_M2
        gain = covariance[:, 0] / innovation_variance
        mean = mean + gain * (z - mean[0])
        covariance = covariance - np.outer(gain, covariance[0])
        assert bank.n_tracks == 1 and tracker.bernoulli_count == 1
        np.testing.assert_allclose(bank.mean[0], mean, rtol=1e-9)
        np.testing.assert_allclose(bank.covariance[0], covariance, rtol=1e-9)


def test_a_first_detection_opens_a_track_with_existence_e_over_clutter_plus_e() -> None:
    """(21)-(24): r = e/(lambda + e), e = P_D lambda_u, lambda_u the uniform fixed point.

    lambda_u = lambda_b / (1 - p_S (1 - P_D)) with lambda_b = 0.05 a scan over 1000 m.
    """
    tracker = _pmbm()
    tracker.process(_scan(0.0, [500.0], clutter=1e-4))
    undetected = 0.05 / 1000.0 / (1 - P_S * (1 - 0.9))
    e = 0.9 * undetected
    # rtol 1e-12: a few operations.
    np.testing.assert_allclose(tracker.local.existence, [e / (1e-4 + e)], rtol=1e-12)
    assert tracker.local.measurement_index.tolist() == [0]


def test_a_missed_bernoulli_keeps_r_times_its_miss_over_one_less_r_p_d_p_g() -> None:
    """(31), gated: r' = r (1 - P_D P_G) / (1 - r P_D P_G), r = p_S after the prediction."""
    tracker = _pmbm()
    tracker.seed([[500.0, 0.0]], [np.diag([4.0, 1.0])], 0.0)
    bank = tracker.process(_scan(1.0, []))
    r = P_S
    expected = r * (1 - 0.9 * P_G) / (1 - r * 0.9 * P_G)
    # rtol 1e-12: a few operations.
    np.testing.assert_allclose(tracker.local.existence, [expected], rtol=1e-12)
    assert bank.status.tolist() == [TrackStatus.COASTING]


def test_two_tracks_sharing_one_measurement_weigh_each_hypothesis_as_the_product() -> None:
    """(35): a global hypothesis weighs the product of its tracks' local weights.

    Two sure tracks at 0 m and 10 m, still, and one measurement at 4 m: either track took it
    (r P_D g, the other missed with 1 - r P_D P_G) or it is a new track or clutter (lambda,
    no birth, both missed). The weights are normalised in the cluster.
    """
    tracker = _pmbm(birth_intensity_per_scan=0.0, survival_probability=1.0)
    still = np.diag([4.0, 1e-6])
    tracker.seed([[0.0, 0.0], [10.0, 0.0]], [still, still], 0.0)
    tracker.process(_scan(1.0, [4.0], clutter=0.01))
    variance = 4.0 + 1e-6 + Q_M2PS3 / 3 + R_M2
    seen, missed = 0.9, 1 - 0.9 * P_G
    weights = np.array(
        [
            seen * _gaussian(4.0, variance) * missed,
            missed * seen * _gaussian(6.0, variance),
            0.01 * missed * missed,
        ]
    )
    # rtol 1e-9: a Cholesky-whitened likelihood against the closed form.
    np.testing.assert_allclose(
        np.sort(np.exp(tracker.hypotheses.log_weight)), np.sort(weights / weights.sum()), rtol=1e-9
    )


def test_no_track_opens_where_the_clutter_is_denser_than_the_birth_limit() -> None:
    """The birth intensity is zero where the clutter passes the limit every tracker shares."""
    tracker = _pmbm(max_birth_clutter_intensity=1e-3)
    tracker.process(_scan(0.0, [100.0], clutter=1e-2))
    assert tracker.bernoulli_count == 0
    tracker.process(_scan(1.0, [300.0], clutter=1e-4))
    assert tracker.bernoulli_count == 1


@pytest.mark.parametrize(
    "settings", [{"k_best": 0}, {"prune_existence": 0.6}, {"coverage_volume": 0.0}]
)
def test_settings_that_could_not_work_are_refused(settings: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _pmbm(**settings)
