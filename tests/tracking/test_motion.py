import itertools

import numpy as np
import pytest

from radar_forge.tracking import CartesianMotion, CoordinatedTurn, RadialMotion

ORIGIN = (36.00250, -78.94100, 60.0)
AXES = [axes for n in (1, 2, 3) for axes in itertools.combinations("xyz", n)]


@pytest.mark.parametrize("axes", AXES)
@pytest.mark.parametrize("kind", ["CV", "CA"])
def test_every_axis_subset_matches_analytic_polynomials(axes, kind):
    model = CartesianMotion(dict.fromkeys(axes, kind), origin_lla_deg_m=ORIGIN, noise_density=2.0)
    n = model.state_space.dimension
    state = np.arange(1, n + 1, dtype=np.float64)
    predicted = model.transition(state, 2.0)
    for axis in axes:
        names = model.state_space.names
        p, v = names.index(f"{axis}_m"), names.index(f"{axis}dot_mps")
        acceleration = state[names.index(f"{axis}ddot_mps2")] if kind == "CA" else 0
        # Polynomial propagation is a handful of float64 operations.
        np.testing.assert_allclose(
            predicted[p], state[p] + 2 * state[v] + 2 * acceleration, rtol=1e-12
        )
        np.testing.assert_allclose(predicted[v], state[v] + 2 * acceleration, rtol=1e-12)
    assert np.linalg.eigvalsh(model.process_noise(state, 2.0)).min() > 0


def test_mixed_axes_and_permutation_preserve_cross_covariance():
    axes = {"x": "CA", "z": "CV"}
    model = CartesianMotion(axes, origin_lla_deg_m=ORIGIN)
    perm = [4, 0, 3, 2, 1]
    other = CartesianMotion(
        axes, origin_lla_deg_m=ORIGIN, order=tuple(model.state_space.names[i] for i in perm)
    )
    f, q = model.matrices(0.3)
    fp, qp = other.matrices(0.3)
    # Matrix assembly must be invariant to a coordinate permutation.
    np.testing.assert_allclose(fp, f[np.ix_(perm, perm)], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(qp, q[np.ix_(perm, perm)], rtol=1e-12, atol=1e-14)
    covariance = np.eye(5) + np.ones((5, 5))
    np.testing.assert_allclose(
        fp @ covariance[np.ix_(perm, perm)] @ fp.T + qp,
        (f @ covariance @ f.T + q)[np.ix_(perm, perm)],
        rtol=1e-12,
    )


def test_continuous_noise_matches_published_cv_and_ca_blocks():
    # FilterPy Q_continuous_white_noise and Bar-Shalom: integrated white driving noise.
    cv = CartesianMotion({"x": "CV"}, origin_lla_deg_m=ORIGIN)
    ca = CartesianMotion({"z": "CA"}, origin_lla_deg_m=ORIGIN)
    np.testing.assert_allclose(cv.matrices(1)[1], [[1 / 3, 1 / 2], [1 / 2, 1]], rtol=1e-12)
    np.testing.assert_allclose(
        ca.matrices(1)[1],
        [[1 / 20, 1 / 8, 1 / 6], [1 / 8, 1 / 3, 1 / 2], [1 / 6, 1 / 2, 1]],
        rtol=1e-12,
    )
    radial = RadialMotion()
    np.testing.assert_allclose(radial.transition(np.array([100.0, 5.0]), 2), [90, 5], rtol=1e-12)
    np.testing.assert_allclose(
        radial.process_noise(np.zeros(2), 1), [[1 / 3, -1 / 2], [-1 / 2, 1]], rtol=1e-12
    )


def test_zero_turn_matches_cartesian_cv():
    turn = CoordinatedTurn(origin_lla_deg_m=ORIGIN)
    state = np.array([100.0, 5.0, 200.0, 6.0, 0.0])
    np.testing.assert_allclose(
        turn.transition(state, 2), [110, 5, 212, 6, 0], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("axes", [{}, {"a": "CV"}, {"x": "bad"}])
def test_invalid_axes_rejected(axes):
    with pytest.raises(ValueError):
        CartesianMotion(axes, origin_lla_deg_m=ORIGIN)
