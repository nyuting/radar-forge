"""Tests for radar_forge.core.tracking.motion: CV/CA and range-only motion models.

Ground truth is analytic: the transition is a Taylor polynomial and the process noise is the
integrated white-noise matrix of Bar-Shalom, Li and Kirubarajan (2001), §6.2.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

from radar_forge.core.tracking.kalman import process_noise_dwna
from radar_forge.core.tracking.motion import CartesianMotion, RadialMotion

ORIGIN = (36.00250, -78.94100, 60.0)
# sigma = 1 with tau = 0.5 s gives q = 2 sigma² tau = 1 exactly, so Q is the published block.
UNIT_DENSITY = {"acceleration_correlation_time_s": 0.5}
AXES = [axes for n in (1, 2, 3) for axes in itertools.combinations("xyz", n)]
SEED = 20261005


@pytest.fixture
def rng() -> np.random.Generator:
    """One seeded generator, so a failure is reproducible."""
    return np.random.default_rng(SEED)


def _model(axes: tuple[str, ...], kind: str) -> CartesianMotion:
    return CartesianMotion(
        dict.fromkeys(axes, kind),
        origin_lla_deg_m=ORIGIN,
        sigma_acceleration_mps2=1.0,
        sigma_jerk_mps3=1.0,
    )


@pytest.mark.parametrize("axes", AXES)
@pytest.mark.parametrize("kind", ["CV", "CA"])
def test_position_follows_the_taylor_polynomial(axes: tuple[str, ...], kind: str) -> None:
    model = _model(axes, kind)
    names = model.state_layout.names
    state = np.arange(1, len(names) + 1, dtype=np.float64)
    predicted = model.transition(state, 2.0)
    for axis in axes:
        # A loop over at most three axes, each with its own named indices.
        p, v = names.index(f"{axis}_m"), names.index(f"{axis}dot_mps")
        acceleration = state[names.index(f"{axis}ddot_mps2")] if kind == "CA" else 0.0
        # rtol 1e-12: x + v T + a T²/2 is a handful of float64 operations.
        np.testing.assert_allclose(
            predicted[p], state[p] + 2 * state[v] + 2 * acceleration, rtol=1e-12
        )


@pytest.mark.parametrize("axes", AXES)
@pytest.mark.parametrize("kind", ["CV", "CA"])
def test_velocity_follows_the_taylor_polynomial(axes: tuple[str, ...], kind: str) -> None:
    model = _model(axes, kind)
    names = model.state_layout.names
    state = np.arange(1, len(names) + 1, dtype=np.float64)
    predicted = model.transition(state, 2.0)
    for axis in axes:
        # A loop over at most three axes, each with its own named indices.
        v = names.index(f"{axis}dot_mps")
        acceleration = state[names.index(f"{axis}ddot_mps2")] if kind == "CA" else 0.0
        # rtol 1e-12: v + a T is two float64 operations.
        np.testing.assert_allclose(predicted[v], state[v] + 2 * acceleration, rtol=1e-12)


@pytest.mark.parametrize("axes", AXES)
@pytest.mark.parametrize("kind", ["CV", "CA"])
def test_process_noise_is_positive_definite(axes: tuple[str, ...], kind: str) -> None:
    model = _model(axes, kind)
    state = np.zeros(model.state_layout.dimension)
    assert np.linalg.eigvalsh(model.process_noise(state, 2.0)).min() > 0


def test_a_permuted_order_permutes_the_transition() -> None:
    axes = {"x": "CA", "z": "CV"}
    model = CartesianMotion(axes, origin_lla_deg_m=ORIGIN)
    perm = [4, 0, 3, 2, 1]
    other = CartesianMotion(
        axes, origin_lla_deg_m=ORIGIN, order=tuple(model.state_layout.names[i] for i in perm)
    )
    # rtol 1e-12, atol 1e-14: the same few products, placed at other indices.
    np.testing.assert_allclose(
        other.matrices(0.3)[0], model.matrices(0.3)[0][np.ix_(perm, perm)], rtol=1e-12, atol=1e-14
    )


def test_a_permuted_order_permutes_the_process_noise() -> None:
    axes = {"x": "CA", "z": "CV"}
    model = CartesianMotion(axes, origin_lla_deg_m=ORIGIN)
    perm = [4, 0, 3, 2, 1]
    other = CartesianMotion(
        axes, origin_lla_deg_m=ORIGIN, order=tuple(model.state_layout.names[i] for i in perm)
    )
    # rtol 1e-12, atol 1e-14: the same few products, placed at other indices.
    np.testing.assert_allclose(
        other.matrices(0.3)[1], model.matrices(0.3)[1][np.ix_(perm, perm)], rtol=1e-12, atol=1e-14
    )


def test_a_cross_axis_covariance_is_carried_through_a_prediction() -> None:
    """Q has no cross-axis terms, but F P Fᵀ keeps the ones already in P."""
    axes = {"x": "CA", "z": "CV"}
    model = CartesianMotion(axes, origin_lla_deg_m=ORIGIN)
    perm = [4, 0, 3, 2, 1]
    other = CartesianMotion(
        axes, origin_lla_deg_m=ORIGIN, order=tuple(model.state_layout.names[i] for i in perm)
    )
    f, q = model.matrices(0.3)
    fp, qp = other.matrices(0.3)
    covariance = np.eye(5) + np.ones((5, 5))
    # rtol 1e-12: two 5 x 5 matrix products, summed in a different order.
    np.testing.assert_allclose(
        fp @ covariance[np.ix_(perm, perm)] @ fp.T + qp,
        (f @ covariance @ f.T + q)[np.ix_(perm, perm)],
        rtol=1e-12,
    )


def test_cv_noise_matches_the_published_block() -> None:
    """Bar-Shalom, Li and Kirubarajan (2001), §6.2: q [[T³/3, T²/2], [T²/2, T]]."""
    cv = CartesianMotion(
        {"x": "CV"}, origin_lla_deg_m=ORIGIN, sigma_acceleration_mps2=1.0, **UNIT_DENSITY
    )
    # rtol 1e-12: one power and one division per element.
    np.testing.assert_allclose(cv.matrices(1)[1], [[1 / 3, 1 / 2], [1 / 2, 1]], rtol=1e-12)


def test_ca_noise_matches_the_published_block() -> None:
    """Bar-Shalom, Li and Kirubarajan (2001), §6.2: the white-jerk (Wiener) block."""
    ca = CartesianMotion({"z": "CA"}, origin_lla_deg_m=ORIGIN, sigma_jerk_mps3=1.0, **UNIT_DENSITY)
    # rtol 1e-12: one power and one division per element.
    np.testing.assert_allclose(
        ca.matrices(1)[1],
        [[1 / 20, 1 / 8, 1 / 6], [1 / 8, 1 / 3, 1 / 2], [1 / 6, 1 / 2, 1]],
        rtol=1e-12,
    )


@pytest.mark.parametrize(("sigma_mps2", "tau_s"), [(4.0, 1.0), (1.5, 5.0), (0.0, 2.0)])
def test_the_noise_density_is_two_sigma_squared_tau(sigma_mps2: float, tau_s: float) -> None:
    """Module Notes: q = 2 sigma_a² tau, the white-noise limit of Singer's model."""
    cartesian = CartesianMotion(
        {"x": "CV"},
        origin_lla_deg_m=ORIGIN,
        sigma_acceleration_mps2=sigma_mps2,
        acceleration_correlation_time_s=tau_s,
    )
    radial = RadialMotion(sigma_mps2, tau_s)
    # rel 1e-12: three float64 multiplications.
    assert cartesian.noise_density["x"] == pytest.approx(2 * sigma_mps2**2 * tau_s, rel=1e-12)
    assert radial.noise_density_m2ps3 == pytest.approx(2 * sigma_mps2**2 * tau_s, rel=1e-12)


def test_the_defaults_give_q_of_32() -> None:
    """Module Notes: sigma_a = 4 m/s² and tau = 1 s, measured on scenario 001's truth."""
    # rel 1e-12: 2 * 16 * 1 is exact in float64.
    assert RadialMotion().noise_density_m2ps3 == pytest.approx(32.0, rel=1e-12)
    cartesian = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    assert cartesian.noise_density["x"] == pytest.approx(32.0, rel=1e-12)


@pytest.mark.parametrize("kind", ["CV", "CA"])
def test_two_short_steps_add_the_same_noise_as_one_long_one(kind: str) -> None:
    """Q(T1 + T2) = F(T2) Q(T1) F(T2)ᵀ + Q(T2): the reason to prefer it to a per-step model.

    Asynchronous sensors cut time into uneven steps, and the uncertainty must not depend on
    how it was cut.
    """
    model = CartesianMotion({"x": kind}, origin_lla_deg_m=ORIGIN)
    first_s, second_s = 0.3, 1.1
    f2, q2 = model.matrices(second_s)
    q1 = model.matrices(first_s)[1]
    # rtol 1e-12: a few products of polynomials in T, each exact to a few ulps.
    np.testing.assert_allclose(
        f2 @ q1 @ f2.T + q2, model.matrices(first_s + second_s)[1], rtol=1e-12
    )


def test_the_discrete_model_adds_less_noise_over_split_steps() -> None:
    """Module Notes: kalman.process_noise_dwna depends on how the time is cut up."""
    sigma_mps2 = 3.0
    f = np.array([[1.0, 1.0], [0.0, 1.0]])
    half = process_noise_dwna(1.0, sigma_mps2)
    split = f @ half @ f.T + half
    whole = process_noise_dwna(2.0, sigma_mps2)
    # Two 1 s steps add a velocity variance of 2 sigma² against 4 sigma² for one 2 s step.
    assert split[1, 1] < whole[1, 1]


def test_cv_matches_the_discrete_model_in_velocity_variance_when_tau_is_half_the_step() -> None:
    """The module's claim relating the two noise models: 2 sigma² tau T = sigma² T²."""
    step_s, sigma_accel_mps2 = 2.0, 3.0
    model = CartesianMotion(
        {"x": "CV"},
        origin_lla_deg_m=ORIGIN,
        sigma_acceleration_mps2=sigma_accel_mps2,
        acceleration_correlation_time_s=step_s / 2,
    )
    # rtol 1e-12: both are σ² T², a few float64 operations each.
    np.testing.assert_allclose(
        model.matrices(step_s)[1][1, 1],
        process_noise_dwna(step_s, sigma_accel_mps2)[1, 1],
        rtol=1e-12,
    )


def test_a_closing_radial_target_loses_range() -> None:
    radial = RadialMotion()
    # rtol 1e-12: 100 - 5 * 2 is exact in float64.
    np.testing.assert_allclose(radial.transition(np.array([100.0, 5.0]), 2), [90, 5], rtol=1e-12)


def test_radial_noise_is_the_cv_block_with_negative_cross_terms() -> None:
    radial = RadialMotion(1.0, 0.5)
    # rtol 1e-12: one power and one division per element.
    np.testing.assert_allclose(
        radial.process_noise(np.zeros(2), 1), [[1 / 3, -1 / 2], [-1 / 2, 1]], rtol=1e-12
    )


@pytest.mark.parametrize("kind", ["CV", "CA"])
def test_a_batched_cartesian_transition_equals_one_state_at_a_time(
    kind: str, rng: np.random.Generator
) -> None:
    model = _model(("x", "y", "z"), kind)
    states = rng.normal(0.0, 100.0, (13, model.state_layout.dimension))
    one_at_a_time = np.array([model.transition(row, 0.7) for row in states])
    # rtol 1e-14, atol 1e-12: the same products per element, but a matrix-matrix product may
    # sum them in a different order than a matrix-vector one, which costs a few ulps.
    np.testing.assert_allclose(model.transition(states, 0.7), one_at_a_time, rtol=1e-14, atol=1e-12)


def test_a_batched_radial_transition_equals_one_state_at_a_time(rng: np.random.Generator) -> None:
    model = RadialMotion()
    states = rng.normal(1000.0, 100.0, (5, 2))
    one_at_a_time = np.array([model.transition(row, 0.7) for row in states])
    # rtol 1e-15: the same two float64 operations per row, in the same order.
    np.testing.assert_allclose(model.transition(states, 0.7), one_at_a_time, rtol=1e-15)


def test_the_radial_transition_does_not_change_its_input() -> None:
    state = np.array([100.0, 5.0])
    RadialMotion().transition(state, 2.0)
    assert state.tolist() == [100.0, 5.0]


def test_a_transition_of_the_wrong_width_is_rejected() -> None:
    model = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    with pytest.raises(ValueError, match="shape"):
        model.transition(np.zeros((4, 3)), 1.0)


@pytest.mark.parametrize("axes", [{}, {"a": "CV"}, {"x": "bad"}])
def test_invalid_axes_are_rejected(axes: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="axes"):
        CartesianMotion(axes, origin_lla_deg_m=ORIGIN)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_acceleration_mps2": -1.0},
        {"sigma_acceleration_mps2": np.inf},
        {"sigma_jerk_mps3": -1.0},
        {"sigma_acceleration_mps2": {"x": -1.0}},
    ],
)
def test_a_negative_or_infinite_sigma_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="sigma"):
        CartesianMotion({"x": "CV", "y": "CA"}, origin_lla_deg_m=ORIGIN, **kwargs)


def test_a_sigma_mapping_must_name_exactly_its_axes() -> None:
    with pytest.raises(ValueError, match="exactly"):
        CartesianMotion(
            {"x": "CV", "y": "CV"},
            origin_lla_deg_m=ORIGIN,
            sigma_acceleration_mps2={"x": 1.0},
        )


@pytest.mark.parametrize("tau_s", [0.0, -1.0, np.inf])
def test_a_correlation_time_that_is_not_positive_and_finite_is_rejected(tau_s: float) -> None:
    with pytest.raises(ValueError, match="acceleration_correlation_time_s"):
        CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN, acceleration_correlation_time_s=tau_s)
    with pytest.raises(ValueError, match="acceleration_correlation_time_s"):
        RadialMotion(acceleration_correlation_time_s=tau_s)


def test_an_order_that_is_not_a_permutation_is_rejected() -> None:
    with pytest.raises(ValueError, match="order"):
        CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN, order=("x_m", "x_m"))


@pytest.mark.parametrize("sigma_mps2", [-1.0, np.nan])
def test_a_negative_or_missing_radial_sigma_is_rejected(sigma_mps2: float) -> None:
    with pytest.raises(ValueError, match="sigma_acceleration_mps2"):
        RadialMotion(sigma_mps2)


@pytest.mark.parametrize("dt_s", [-1.0, np.inf])
@pytest.mark.parametrize("call", ["matrices", "transition", "process_noise"])
def test_a_negative_or_infinite_cartesian_step_is_rejected(dt_s: float, call: str) -> None:
    model = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    method = getattr(model, call)
    args = (dt_s,) if call == "matrices" else (np.zeros(2), dt_s)
    with pytest.raises(ValueError, match="dt_s"):
        method(*args)


@pytest.mark.parametrize("dt_s", [-1.0, np.nan])
@pytest.mark.parametrize("call", ["transition", "process_noise"])
def test_a_negative_or_missing_radial_step_is_rejected(dt_s: float, call: str) -> None:
    with pytest.raises(ValueError, match="dt_s"):
        getattr(RadialMotion(), call)(np.zeros(2), dt_s)
