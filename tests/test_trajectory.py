"""Tests for radar_forge.trajectory: closed-form arcs against their kinematics.

Ground truth is closed form: a turn's radius v/omega and a straight leg's kinematics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from radar_forge.trajectory import Arc, Leg


def test_a_straight_leg_follows_constant_acceleration_along_its_heading() -> None:
    """Heading 120 deg, 10 m/s, +2 m/s^2 for 30 s: 1200 m along track, 70 m/s."""
    arc = Arc(5.0, (100.0, 200.0), math.radians(120.0), 10.0, (Leg(30.0, 0.0, 2.0),))
    position_m, velocity_mps = arc.state(35.0)
    along = np.array([math.sin(math.radians(120.0)), math.cos(math.radians(120.0))])
    # rtol 1e-12: a few products.
    np.testing.assert_allclose(position_m, [100.0, 200.0] + 1200.0 * along, rtol=1e-12)
    np.testing.assert_allclose(velocity_mps, 70.0 * along, rtol=1e-12)


def test_a_turn_holds_its_radius_and_speed_and_after_the_legs_it_flies_straight() -> None:
    """3 deg/s at 72 m/s: every point 1375.1 m from the centre; then straight at 72 m/s."""
    rate = math.radians(3.0)
    arc = Arc(0.0, (0.0, 0.0), 0.0, 72.0, (Leg(60.0, rate),))
    position_m, velocity_mps = arc.state(np.linspace(0.0, 60.0, 61))
    centre = np.array([72.0 / rate, 0.0])
    np.testing.assert_allclose(np.linalg.norm(position_m - centre, axis=1), 72.0 / rate, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(velocity_mps, axis=1), 72.0, rtol=1e-12)
    later, later_velocity = arc.state(np.array([60.0, 70.0]))
    # atol 1e-9 m: the eastward step is zero, so only an absolute bound means anything there.
    np.testing.assert_allclose(later[1] - later[0], 10.0 * later_velocity[0], rtol=1e-12, atol=1e-9)


def test_a_target_does_not_exist_before_its_start() -> None:
    arc = Arc(100.0, (0.0, 0.0), 0.0, 50.0)
    position_m, _ = arc.state(np.array([99.0, 100.0]))
    assert np.isnan(position_m[0]).all() and np.isfinite(position_m[1]).all()


def test_a_leg_may_not_turn_and_accelerate_at_once() -> None:
    with pytest.raises(ValueError, match="leg"):
        Leg(10.0, 0.1, 1.0)


def test_a_leg_that_decelerates_past_a_stop_is_refused() -> None:
    """10 m/s braking at 2 m/s^2 for 10 s would end at -10 m/s, flying backwards."""
    with pytest.raises(ValueError, match="decelerates"):
        Arc(0.0, (0.0, 0.0), 0.0, 10.0, (Leg(10.0, 0.0, -2.0),))
    Arc(0.0, (0.0, 0.0), 0.0, 10.0, (Leg(5.0, 0.0, -2.0),))
