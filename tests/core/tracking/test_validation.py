"""Tests for radar_forge.core.tracking._validation: the shared vector and covariance checks.

The module is private, but its helpers guard every array the tracker passes around, so they get
their own unit tests. Each case is analytic: a known Cholesky factor, a matrix with a known
negative eigenvalue, or a shape that is plainly wrong. The roundoff allowance is 1e-10 times the
largest absolute element (module docstring), and the cases sit far on either side of it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from radar_forge.core.tracking._validation import (
    as_covariance,
    as_points,
    as_vector,
    check_frame_and_origin,
    check_points,
    check_sigma_point_settings,
    cholesky_factor,
)

# --------------------------------------------------------------------------- #
# as_vector
# --------------------------------------------------------------------------- #


def test_as_vector_returns_a_float64_copy() -> None:
    original = np.array([1, 2, 3])
    result = as_vector(original, 3)
    assert result.dtype == np.float64
    assert not np.shares_memory(result, original)
    # Integers convert exactly, so equality is the right test.
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_as_vector_is_independent_of_the_callers_array() -> None:
    original = np.array([1.0, 2.0])
    result = as_vector(original, 2)
    original[0] = 99.0
    assert result[0] == 1.0


@pytest.mark.parametrize(
    ("value", "dimension"),
    [
        ([1.0, 2.0], 3),
        ([[1.0, 2.0]], 2),
        ([1.0, np.nan], 2),
        ([1.0, np.inf], 2),
        ([], 0),
    ],
    ids=["too-short", "two-dimensional", "nan", "infinite", "zero-dimension"],
)
def test_as_vector_rejects_a_bad_vector(value: Any, dimension: int) -> None:
    with pytest.raises(ValueError, match="must be finite with shape"):
        as_vector(value, dimension)


def test_as_vector_names_the_value_in_its_error() -> None:
    with pytest.raises(ValueError, match="weights"):
        as_vector([1.0], 2, "weights")


# --------------------------------------------------------------------------- #
# as_points
# --------------------------------------------------------------------------- #


def test_as_points_does_not_copy_a_float64_array() -> None:
    points = np.zeros((5, 2), dtype=np.float64)
    assert np.shares_memory(as_points(points, 5, 2), points)


def test_as_points_converts_a_list_to_float64() -> None:
    result = as_points([[1, 2], [3, 4]], 2, 2)
    assert result.dtype == np.float64
    # Integers convert exactly, so equality is the right test.
    np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize(
    "value",
    [np.zeros((3, 2)), np.zeros((2, 3)), np.zeros(4), np.array([[0.0, np.nan], [0.0, 0.0]])],
    ids=["too-many-rows", "too-many-columns", "one-dimensional", "nan"],
)
def test_as_points_rejects_a_bad_batch(value: Any) -> None:
    with pytest.raises(ValueError, match="must be finite with shape"):
        as_points(value, 2, 2)


# --------------------------------------------------------------------------- #
# as_covariance
# --------------------------------------------------------------------------- #


def test_as_covariance_returns_an_independent_copy() -> None:
    original = np.array([[4.0, 1.0], [1.0, 2.0]])
    result = as_covariance(original, 2)
    assert not np.shares_memory(result, original)
    # A symmetric matrix is unchanged by averaging it with its transpose, bit for bit.
    np.testing.assert_array_equal(result, original)


def test_as_covariance_makes_roundoff_asymmetry_exact() -> None:
    # An off-diagonal mismatch of 1e-13 is far below the allowance of 4e-10.
    result = as_covariance([[4.0, 1.0 + 1e-13], [1.0, 2.0]], 2)
    # Exactly symmetric: the copy is the average of the matrix and its transpose.
    np.testing.assert_array_equal(result, result.T)


@pytest.mark.parametrize(
    "value",
    [np.zeros((2, 2)), np.ones((2, 2)), np.diag([1.0, -1e-12])],
    ids=["all-zero", "rank-one", "negative-within-roundoff"],
)
def test_as_covariance_accepts_a_semidefinite_matrix(value: Any) -> None:
    """Singular matrices are valid covariances, and roundoff below zero is forgiven."""
    as_covariance(value, 2)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_as_covariance_judges_a_matrix_the_same_whatever_its_units(scale: float) -> None:
    """The allowance scales with the matrix (Notes), so only the shape of the spectrum counts."""
    as_covariance(scale * np.diag([1.0, -1e-12]), 2)
    with pytest.raises(ValueError, match="positive semidefinite"):
        as_covariance(scale * np.diag([1.0, -1e-6]), 2)


def test_as_covariance_rejects_a_negative_eigenvalue() -> None:
    # [[1, 2], [2, 1]] has eigenvalues 3 and -1.
    with pytest.raises(ValueError, match="positive semidefinite"):
        as_covariance([[1.0, 2.0], [2.0, 1.0]], 2)


def test_as_covariance_rejects_asymmetry_beyond_roundoff() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        as_covariance([[4.0, 1.0], [1.1, 2.0]], 2)


@pytest.mark.parametrize(
    ("value", "dimension"),
    [
        (np.eye(3), 2),
        (np.ones(2), 2),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), 2),
        (np.array([[np.inf]]), 1),
        (np.zeros((0, 0)), 0),
    ],
    ids=["wrong-size", "one-dimensional", "nan", "infinite", "zero-dimension"],
)
def test_as_covariance_rejects_a_bad_shape_or_value(value: Any, dimension: int) -> None:
    with pytest.raises(ValueError, match="must be finite with shape"):
        as_covariance(value, dimension)


# --------------------------------------------------------------------------- #
# cholesky_factor
# --------------------------------------------------------------------------- #


def test_cholesky_factor_of_a_diagonal_matrix_is_its_square_root() -> None:
    factor = cholesky_factor(np.diag([4.0, 9.0]))
    # Square roots of perfect squares are exact in float64.
    np.testing.assert_array_equal(factor, np.diag([2.0, 3.0]))


def test_cholesky_factor_matches_a_hand_computed_factor() -> None:
    """[[4, 2], [2, 3]] = L Lᵀ with L = [[2, 0], [1, √2]]: l11 = √4, l21 = 2/2, l22 = √(3 - 1)."""
    factor = cholesky_factor(np.array([[4.0, 2.0], [2.0, 3.0]]))
    # rtol 1e-12: three square roots and a division, so float64 roundoff only.
    np.testing.assert_allclose(factor, [[2.0, 0.0], [1.0, np.sqrt(2.0)]], rtol=1e-12, atol=0.0)


def test_cholesky_factor_retries_a_singular_matrix_within_roundoff() -> None:
    """A rank-one matrix fails a plain factorisation; the jitter retry recovers it (Notes)."""
    matrix = np.ones((2, 2))
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(matrix)
    factor = cholesky_factor(matrix)
    # atol 1e-10: the retry adds at most the allowance, 1e-10 times the largest element (1).
    np.testing.assert_allclose(factor @ factor.T, matrix, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize(
    "matrix",
    [np.zeros((2, 2)), np.diag([1.0, -1.0]), np.diag([1.0, -1e-6])],
    ids=["all-zero", "negative-definite-part", "negative-beyond-roundoff"],
)
def test_cholesky_factor_rejects_a_matrix_that_is_not_positive_definite(
    matrix: Any,
) -> None:
    with pytest.raises(ValueError, match="not positive definite"):
        cholesky_factor(matrix)


@pytest.mark.parametrize(
    "matrix",
    [np.ones((2, 3)), np.array([[1.0, np.nan], [np.nan, 1.0]]), np.array(1.0)],
    ids=["not-square", "nan", "scalar"],
)
def test_cholesky_factor_rejects_a_matrix_that_is_not_a_finite_square(matrix: Any) -> None:
    with pytest.raises(ValueError, match="must be finite with shape"):
        cholesky_factor(matrix)


def test_cholesky_factor_rejects_an_asymmetric_matrix() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        cholesky_factor(np.array([[4.0, 1.0], [0.0, 2.0]]))


# --------------------------------------------------------------------------- #
# check_points, check_frame_and_origin and check_sigma_point_settings
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape", [(3,), (5, 3)], ids=["one", "stack"])
def test_check_points_accepts_one_state_or_a_stack(shape: tuple[int, ...]) -> None:
    check_points(np.zeros(shape), 3, "state")


@pytest.mark.parametrize(
    "array",
    [np.zeros(2), np.zeros((2, 2, 3)), np.array([0.0, np.inf, 0.0])],
    ids=["wrong-length", "three-axes", "infinite"],
)
def test_check_points_rejects_a_wrong_shape_or_a_nonfinite_value(array: Any) -> None:
    with pytest.raises(ValueError, match="state must be finite"):
        check_points(array, 3, "state")


def test_check_frame_and_origin_returns_the_origin_as_floats() -> None:
    assert check_frame_and_origin("ENU", (36, -78, 60)) == (36.0, -78.0, 60.0)


def test_check_frame_and_origin_allows_no_origin() -> None:
    assert check_frame_and_origin("local", None) is None


def test_check_frame_and_origin_rejects_an_empty_frame() -> None:
    with pytest.raises(ValueError, match="frame"):
        check_frame_and_origin("", None)


@pytest.mark.parametrize(
    "origin",
    [(91.0, 0.0, 0.0), (0.0, -181.0, 0.0), (0.0, 0.0, np.nan), (0.0, 0.0)],
    ids=["lat", "lon", "nan", "two-values"],
)
def test_check_frame_and_origin_rejects_an_impossible_origin(origin: Any) -> None:
    with pytest.raises(ValueError, match="ENU origin"):
        check_frame_and_origin("ENU", origin)


def test_check_sigma_point_settings_accepts_the_defaults() -> None:
    check_sigma_point_settings(1.0, 2.0, 0.0, 6)


@pytest.mark.parametrize(
    ("alpha", "beta", "kappa"),
    [(0.0, 2.0, 0.0), (1.0, -1.0, 0.0), (1.0, 2.0, -6.0), (np.nan, 2.0, 0.0)],
    ids=["zero-alpha", "negative-beta", "n-plus-kappa-zero", "nan"],
)
def test_check_sigma_point_settings_rejects_an_impossible_setting(
    alpha: float, beta: float, kappa: float
) -> None:
    with pytest.raises(ValueError, match="alpha"):
        check_sigma_point_settings(alpha, beta, kappa, 6)
