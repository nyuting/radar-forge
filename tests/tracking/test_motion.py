"""Tests for radar_forge.tracking.motion: state layouts and motion models.

Ground truth is analytic: the published continuous and discrete white-noise blocks
(Bar-Shalom, Li and Kirubarajan 2001, §6.2 and §6.3.2), exact dead reckoning, two
continuous half steps equal to one full step, and the coordinated turn's textbook radius. The
turn's analytic Jacobian is checked against a central difference at many random states.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from radar_forge.tracking.motion import (
    CartesianMotion,
    Coordinate,
    CoordinatedTurn,
    RadialConstantVelocity,
    StateLayout,
)

RANGE_PERIOD = StateLayout((Coordinate("range_m", "m", period=1000.0), Coordinate("v", "m/s")))


@pytest.mark.parametrize(
    ("name", "unit", "period"),
    [("", "m", None), ("x_m", "", None), ("x_m", "m", 0.0), ("x_m", "m", np.inf)],
)
def test_a_coordinate_rejects_an_empty_name_or_a_bad_period(
    name: str, unit: str, period: float | None
) -> None:
    with pytest.raises(ValueError, match="coordinate"):
        Coordinate(name, unit, period)


def test_a_layout_rejects_a_repeated_name() -> None:
    with pytest.raises(ValueError, match="unique"):
        StateLayout((Coordinate("x_m", "m"), Coordinate("x_m", "m")))


def test_indices_follow_the_order_asked_for_and_reject_an_unknown_name() -> None:
    assert RANGE_PERIOD.indices(("v", "range_m")) == (1, 0)
    with pytest.raises(ValueError, match="not in the layout"):
        RANGE_PERIOD.indices(("z_m",))


@pytest.mark.parametrize(
    ("range_m", "centred_m", "nonnegative_m"),
    [
        (700.0, -300.0, 700.0),
        (500.0, -500.0, 500.0),
        (-500.0, -500.0, 500.0),
        (1499.0, 499.0, 499.0),
    ],
)
def test_a_wrap_lands_in_its_half_open_interval(
    range_m: float, centred_m: float, nonnegative_m: float
) -> None:
    """Centred is [-p/2, p/2), nonnegative [0, p); a value one period on is the same value."""
    value = np.array([range_m, 7.0])
    np.testing.assert_array_equal(RANGE_PERIOD.wrap(value), [centred_m, 7.0])
    np.testing.assert_array_equal(
        RANGE_PERIOD.wrap(value, interval="nonnegative"), [nonnegative_m, 7.0]
    )


def test_a_tiny_negative_value_does_not_wrap_to_a_full_period() -> None:
    """np.mod(-1e-20, 1000) rounds to 1000, outside [0, 1000); the wrap folds it to 0."""
    assert RANGE_PERIOD.wrap([-1e-20, 0.0], interval="nonnegative")[0] == 0.0


def test_a_wrap_of_a_stack_wraps_every_row_and_rejects_an_unknown_interval() -> None:
    stack = np.array([[[700.0, 1.0], [-700.0, 2.0]], [[1200.0, 3.0], [0.0, 4.0]]])
    np.testing.assert_array_equal(RANGE_PERIOD.wrap(stack)[..., 0], [[-300.0, 300.0], [200.0, 0.0]])
    with pytest.raises(ValueError, match="interval"):
        RANGE_PERIOD.wrap(stack, interval="positive")  # type: ignore[arg-type]


def test_a_residual_across_due_north_takes_the_short_way_round() -> None:
    layout = StateLayout((Coordinate("azimuth_rad", "rad", period=2 * np.pi),))
    # rtol 1e-12: one subtraction and one wrap.
    np.testing.assert_allclose(layout.residual([0.01], [2 * np.pi - 0.01]), [0.02], rtol=1e-12)


def test_a_radial_transition_is_the_textbook_one_with_no_minus_sign() -> None:
    """Range rate is the range's derivative, positive receding: F = [[1, T], [0, 1]]."""
    motion = RadialConstantVelocity()
    np.testing.assert_array_equal(motion.jacobian(np.zeros(2), 2.5), [[1.0, 2.5], [0.0, 1.0]])
    np.testing.assert_array_equal(motion.transition(np.array([1000.0, 10.0]), 2.0), [1020.0, 10.0])
    names = motion.state_layout.names
    assert names == ("bistatic_range_m", "bistatic_range_rate_mps")


@pytest.mark.parametrize("order", [1, 2])
def test_dead_reckoning_follows_the_taylor_polynomial(order: int) -> None:
    """x + v T + a T²/2, exactly, on every axis."""
    motion = CartesianMotion("xy", order)  # type: ignore[arg-type]
    acceleration_mps2 = 2.0 if order == 2 else 0.0
    axis = [100.0, 10.0, acceleration_mps2][: order + 1]
    moved = motion.transition(np.array(axis + axis), 3.0)
    expected = [100.0 + 10.0 * 3.0 + acceleration_mps2 * 3.0**2 / 2, 10.0 + 3.0 * acceleration_mps2]
    np.testing.assert_allclose(moved[:2], expected, rtol=1e-15)
    np.testing.assert_array_equal(moved[: order + 1], moved[order + 1 :])


def test_the_continuous_noise_matches_the_published_blocks() -> None:
    """Bar-Shalom, Li and Kirubarajan (2001), §6.2: CV and CA blocks, q = 2 sigma² tau."""
    step_s, q = 1.5, 2 * 3.0**2 * 0.5
    cv = CartesianMotion("x", sigma_acceleration_mps2=3.0, acceleration_correlation_time_s=0.5)
    ca = CartesianMotion("x", 2, sigma_jerk_mps3=3.0, acceleration_correlation_time_s=0.5)
    t = step_s
    cv_block = q * np.array([[t**3 / 3, t**2 / 2], [t**2 / 2, t]])
    ca_block = q * np.array(
        [[t**5 / 20, t**4 / 8, t**3 / 6], [t**4 / 8, t**3 / 3, t**2 / 2], [t**3 / 6, t**2 / 2, t]]
    )
    # rtol 1e-14: a few products of exact coefficients.
    np.testing.assert_allclose(cv.process_noise(np.zeros(2), t), cv_block, rtol=1e-14)
    np.testing.assert_allclose(ca.process_noise(np.zeros(3), t), ca_block, rtol=1e-14)


@pytest.mark.parametrize(
    "motion",
    [
        CartesianMotion("x", process_noise="dwna", sigma_acceleration_mps2=3.0),
        RadialConstantVelocity(3.0, process_noise="dwna"),
    ],
)
def test_the_discrete_noise_matches_the_published_block(motion: Any) -> None:
    """Bar-Shalom, Li and Kirubarajan (2001), §6.3.2: sigma² Γ Γᵀ with Γ = (T²/2, T)."""
    t = 2.0
    expected = 9.0 * np.array([[t**4 / 4, t**3 / 2], [t**3 / 2, t**2]])
    np.testing.assert_allclose(motion.process_noise(np.zeros(2), t), expected, rtol=1e-14)


def test_the_defaults_give_a_noise_intensity_of_32() -> None:
    """sigma = 4 m/s², tau = 1 s: q = 2 · 16 · 1, the velocity variance added per second."""
    assert CartesianMotion("x").process_noise(np.zeros(2), 1.0)[1, 1] == 32.0


@pytest.mark.parametrize("order", [1, 2])
def test_two_continuous_half_steps_add_the_same_noise_as_one_full_step(order: int) -> None:
    """F(T/2) Q(T/2) F(T/2)ᵀ + Q(T/2) = Q(T): Q does not depend on how time is cut up."""
    motion = CartesianMotion("x", order)  # type: ignore[arg-type]
    state = np.zeros(order + 1)
    half_f, half_q = motion.jacobian(state, 0.7), motion.process_noise(state, 0.7)
    # rtol 1e-12: polynomial identities evaluated in float64.
    np.testing.assert_allclose(
        half_f @ half_q @ half_f.T + half_q, motion.process_noise(state, 1.4), rtol=1e-12
    )


def test_two_discrete_half_steps_add_less_velocity_noise_than_one_full_step() -> None:
    """DWNA adds sigma² T² per step: 2 · sigma² (T/2)² is half of sigma² T²."""
    motion = CartesianMotion("x", process_noise="dwna")
    half_f, half_q = motion.jacobian(np.zeros(2), 0.5), motion.process_noise(np.zeros(2), 0.5)
    split = half_f @ half_q @ half_f.T + half_q
    assert split[1, 1] == pytest.approx(motion.process_noise(np.zeros(2), 1.0)[1, 1] / 2)


def test_the_two_models_add_the_same_velocity_noise_when_tau_is_half_the_step() -> None:
    """2 sigma² tau T = sigma² T² exactly when tau = T/2 (module docstring)."""
    cwna = CartesianMotion("x", acceleration_correlation_time_s=0.75)
    dwna = CartesianMotion("x", process_noise="dwna")
    assert cwna.process_noise(np.zeros(2), 1.5)[1, 1] == pytest.approx(
        dwna.process_noise(np.zeros(2), 1.5)[1, 1], rel=1e-14
    )


def test_the_axes_are_interleaved_and_their_noise_independent() -> None:
    motion = CartesianMotion()
    assert motion.state_layout.names == ("x_m", "xdot_mps", "y_m", "ydot_mps", "z_m", "zdot_mps")
    noise = motion.process_noise(np.zeros(6), 1.0)
    assert np.all(noise[:2, 2:] == 0.0) and np.all(noise[2:4, 4:] == 0.0)


def test_a_batched_step_equals_one_state_at_a_time(rng: np.random.Generator) -> None:
    """Each row moves by its own dt, in one call, as the bank's rows do."""
    motion = CartesianMotion()
    states, steps_s = rng.normal(0.0, 100.0, (5, 6)), rng.uniform(0.1, 3.0, 5)
    one_by_one = [motion.transition(s, t) for s, t in zip(states, steps_s, strict=True)]
    noise = [motion.process_noise(s, t) for s, t in zip(states, steps_s, strict=True)]
    np.testing.assert_array_equal(motion.transition(states, steps_s), one_by_one)
    np.testing.assert_array_equal(motion.process_noise(states, steps_s), noise)


@pytest.mark.parametrize("motion", [CartesianMotion("xy", 2), RadialConstantVelocity()])
def test_every_motion_model_is_linear_as_the_taylor_prediction_assumes(
    motion: Any, rng: np.random.Generator
) -> None:
    """transition(x) is jacobian(x) @ x, so predicting the mean as F x needs one F."""
    state = rng.normal(0.0, 10.0, motion.state_layout.dimension)
    # rtol 1e-14: the same matrix applied two ways.
    np.testing.assert_allclose(
        motion.transition(state, 1.3), motion.jacobian(state, 1.3) @ state, rtol=1e-14
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"axes": ""},
        {"axes": "xw"},
        {"axes": "xx"},
        {"order": 3},
        {"sigma_acceleration_mps2": -1.0},
        {"sigma_acceleration_mps2": np.nan},
        {"acceleration_correlation_time_s": 0.0},
        {"process_noise": "singer"},
        {"order": 2, "process_noise": "dwna"},
    ],
)
def test_impossible_motion_settings_are_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"axes|process_noise|sigma"):
        CartesianMotion(**settings)


# --- CoordinatedTurn ---


def test_a_coordinated_turn_at_zero_rate_is_constant_velocity_exactly() -> None:
    """At omega = 0 the series branch gives CV's F and Q to the bit, on the shared axes."""
    turn, cv = CoordinatedTurn(0.5), CartesianMotion("xy", sigma_acceleration_mps2=0.5)
    state = np.array([1e4, 72.0, -2e3, -30.0, 0.0])
    np.testing.assert_array_equal(turn.jacobian(state, 4.8)[:4, :4], cv.jacobian(state[:4], 4.8))
    np.testing.assert_array_equal(
        turn.process_noise(state, 4.8)[:4, :4], cv.process_noise(state[:4], 4.8)
    )


def test_a_coordinated_turn_near_zero_rate_differs_from_cv_by_its_first_order_term() -> None:
    """At omega = 1e-9 F - F_CV is omega T²/2 across axes and omega T on velocity.

    Equality at rtol 1e-10 is out of reach for any entry that is zero in CV: the turn's
    first-order term is a real 1.2e-8 there. rtol 1e-10 holds against the series instead.
    """
    rate, step = 1e-9, 4.8
    turn = CoordinatedTurn(0.5)
    cv = CartesianMotion("xy", sigma_acceleration_mps2=0.5).jacobian(np.zeros(4), step)
    expected = cv.copy()
    expected[0, 3], expected[2, 1] = rate * step**2 / 2, -rate * step**2 / 2
    expected[1, 3], expected[3, 1] = rate * step, -rate * step
    got = turn.jacobian(np.array([0.0, 0.0, 0.0, 0.0, rate]), step)[:4, :4]
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)


def test_a_coordinated_turn_is_continuous_across_the_series_threshold() -> None:
    """F either side of |omega T| = 1e-6 agrees to rtol 1e-10: the series is not a step."""
    step = 4.8
    # A relative 1e-12 either side: entries proportional to the rate then move by 2e-12.
    below, above = (1e-6 / step) * (1 - 1e-12), (1e-6 / step) * (1 + 1e-12)
    turn = CoordinatedTurn(0.5)
    state = np.array([[1e4, 72.0, -2e3, -30.0, below], [1e4, 72.0, -2e3, -30.0, above]])
    jacobian = turn.jacobian(state, step)
    # rtol 1e-10 on F; the omega column is 1e-5 of the rest, so it is held to 1e-10 of F's scale.
    np.testing.assert_allclose(jacobian[0], jacobian[1], rtol=1e-10, atol=1e-10 * step)
    # The rate itself differs by 2e-12 between the two states; the moved position does not.
    moved = turn.transition(state, step)
    np.testing.assert_allclose(moved[0, :4], moved[1, :4], rtol=1e-12)


def test_a_positive_turn_rate_turns_right_onto_the_textbook_radius() -> None:
    """Heading north at 72 m/s, +3 deg/s for 30 s: heading east, at (R, R), R = v / omega."""
    rate = np.radians(3.0)
    radius = 72.0 / rate
    moved = CoordinatedTurn().transition(np.array([0.0, 0.0, 0.0, 72.0, rate]), 30.0)
    # rtol 1e-12: a handful of trigonometric evaluations.
    np.testing.assert_allclose(moved[[0, 2]], [radius, radius], rtol=1e-12)
    np.testing.assert_allclose(moved[[1, 3]], [72.0, 0.0], atol=1e-12)


def test_a_coordinated_turn_jacobian_matches_central_differences() -> None:
    """The analytic Jacobian, rate column included, against a central difference."""
    rng = np.random.default_rng(20261106)
    states = np.column_stack(
        [
            rng.normal(0, 1e4, 20),
            rng.normal(0, 100, 20),
            rng.normal(0, 1e4, 20),
            rng.normal(0, 100, 20),
            rng.normal(0, 0.05, 20),
        ]
    )
    turn = CoordinatedTurn()
    step = 1e-6 * np.maximum(np.abs(states), 1.0)
    # One central difference per state coordinate: five.
    numeric = np.stack(
        [
            (
                turn.transition(states + h[:, None] * e, 4.8)
                - turn.transition(states - h[:, None] * e, 4.8)
            )
            / (2 * h[:, None])
            for h, e in zip(step.T, np.eye(5), strict=True)
        ],
        axis=-1,
    )
    # rtol 1e-6: a central difference's truncation and cancellation, at a 1e-6 relative step.
    np.testing.assert_allclose(turn.jacobian(states, 4.8), numeric, rtol=1e-6, atol=1e-6)
