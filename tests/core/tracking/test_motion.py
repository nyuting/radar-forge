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
        acceleration_noise_density_m2ps3=2.0,
        jerk_noise_density_m2ps5=2.0,
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
    cv = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    # rtol 1e-12: one power and one division per element.
    np.testing.assert_allclose(cv.matrices(1)[1], [[1 / 3, 1 / 2], [1 / 2, 1]], rtol=1e-12)


def test_ca_noise_matches_the_published_block() -> None:
    """Bar-Shalom, Li and Kirubarajan (2001), §6.2: the white-jerk (Wiener) block."""
    ca = CartesianMotion({"z": "CA"}, origin_lla_deg_m=ORIGIN)
    # rtol 1e-12: one power and one division per element.
    np.testing.assert_allclose(
        ca.matrices(1)[1],
        [[1 / 20, 1 / 8, 1 / 6], [1 / 8, 1 / 3, 1 / 2], [1 / 6, 1 / 2, 1]],
        rtol=1e-12,
    )


def test_noise_scales_with_the_density() -> None:
    weak = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    strong = CartesianMotion(
        {"x": "CV"}, origin_lla_deg_m=ORIGIN, acceleration_noise_density_m2ps3=16.0
    )
    # rtol 1e-12: one extra multiplication per element.
    np.testing.assert_allclose(strong.matrices(0.5)[1], 16 * weak.matrices(0.5)[1], rtol=1e-12)


def test_cv_matches_the_discrete_model_in_velocity_variance_when_q_is_sigma_squared_t() -> None:
    """The module's claim relating the two noise conventions: q = sigma_a² T."""
    step_s, sigma_accel_mps2 = 2.0, 3.0
    model = CartesianMotion(
        {"x": "CV"},
        origin_lla_deg_m=ORIGIN,
        acceleration_noise_density_m2ps3=sigma_accel_mps2**2 * step_s,
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
    radial = RadialMotion()
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
        {"acceleration_noise_density_m2ps3": -1.0},
        {"acceleration_noise_density_m2ps3": np.inf},
        {"jerk_noise_density_m2ps5": -1.0},
        {"acceleration_noise_density_m2ps3": {"x": -1.0}},
    ],
)
def test_a_negative_or_infinite_density_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="densit"):
        CartesianMotion({"x": "CV", "y": "CA"}, origin_lla_deg_m=ORIGIN, **kwargs)


def test_a_density_mapping_must_name_exactly_its_axes() -> None:
    with pytest.raises(ValueError, match="exactly"):
        CartesianMotion(
            {"x": "CV", "y": "CV"},
            origin_lla_deg_m=ORIGIN,
            acceleration_noise_density_m2ps3={"x": 1.0},
        )


def test_an_order_that_is_not_a_permutation_is_rejected() -> None:
    with pytest.raises(ValueError, match="order"):
        CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN, order=("x_m", "x_m"))


@pytest.mark.parametrize("density", [-1.0, np.nan])
def test_a_negative_or_missing_radial_density_is_rejected(density: float) -> None:
    with pytest.raises(ValueError, match="acceleration_noise_density_m2ps3"):
        RadialMotion(density)


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
