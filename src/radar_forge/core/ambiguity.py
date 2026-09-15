r"""Resolving range and Doppler ambiguity by combining two repetition rates.

A single pulse repetition frequency buys either unambiguous range or
unambiguous Doppler, never both: their product is fixed at
:math:`c\lambda/8` by the carrier alone, so no choice of PRF escapes the trade.
Scenario 001 S1 and S2 each accept one half of that loss.

The way out is to measure twice at two different repetition rates. A target
folded at both rates produces a *pair* of false velocities, and if the two rates
are coprime multiples of a common base, only one true velocity is consistent
with both. This is the radar form of the Chinese remainder theorem, and it is
what scenario 001 S3 uses to recover an aircraft's real velocity from two FMCW
bursts neither of which could measure it alone.

The extension is not free. It costs dwell time, it fails when two targets are
present at the same range (the pairs become ambiguous between targets, the
classic "ghost" problem), and it degrades sharply once measurement noise
approaches half a Doppler bin. The first two are out of scope for a
single-target scenario; the third is why :func:`unfold_doppler_dual_prf` takes an
explicit tolerance and reports the residual rather than silently returning its
best guess.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §5.3.3 (multiple PRF ambiguity resolution).
.. [2] M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill,
       2001, §3.2 (multiple-PRF and the Chinese remainder approach).
.. [3] G. W. Stimson, *Introduction to Airborne Radar*, 2nd ed., SciTech, 1998,
       ch. 25 (PRF ratios, ghosts, and blind zones).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["fold_velocity_mps", "unfold_doppler_dual_prf"]


def fold_velocity_mps(
    velocity_mps: ArrayLike,
    unambiguous_velocity_mps: ArrayLike,
) -> NDArray[np.float64]:
    r"""Wrap a true radial velocity into a radar's unambiguous interval.

    The forward model of Doppler aliasing: what a radar with a half-interval of
    :math:`v_{ua}` would *report* for a target actually travelling at
    :math:`v`. Used to predict where a folded target should appear, and as the
    inverse against which :func:`unfold_doppler_dual_prf` is tested.

    Parameters
    ----------
    velocity_mps : array_like
        True radial velocity, metres/second, positive closing.
    unambiguous_velocity_mps : array_like
        Half-width of the unambiguous interval, metres/second — that is,
        :attr:`radar_forge.core.radar.Radar.unambiguous_velocity_mps`. Must be
        strictly positive.

    Returns
    -------
    numpy.ndarray
        The apparent velocity, in ``[-unambiguous_velocity_mps,
        +unambiguous_velocity_mps)``.

    Raises
    ------
    ValueError
        If any unambiguous velocity is non-positive.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isclose(fold_velocity_mps(3.0, 7.65), 3.0))  # already unambiguous
    True
    >>> bool(np.isclose(fold_velocity_mps(80.0, 7.65), 80.0 - 5 * 15.3))
    True
    """
    velocity_arr = np.asarray(velocity_mps, dtype=np.float64)
    half_interval = np.asarray(unambiguous_velocity_mps, dtype=np.float64)
    if np.any(half_interval <= 0.0):
        msg = "unambiguous_velocity_mps must be strictly positive."
        raise ValueError(msg)
    span = 2.0 * half_interval
    result: NDArray[np.float64] = np.mod(velocity_arr + half_interval, span) - half_interval
    return result


def unfold_doppler_dual_prf(
    folded_velocity_a_mps: ArrayLike,
    folded_velocity_b_mps: ArrayLike,
    unambiguous_velocity_a_mps: float,
    unambiguous_velocity_b_mps: float,
    *,
    max_velocity_mps: float,
    tolerance_mps: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Recover true radial velocity from two folded measurements.

    Enumerates the candidate velocities consistent with the first measurement,

    .. math:: v_k = v_a + 2 k\, v_{ua,a},
        \qquad |v_k| \le v_{\max},

    and selects the one whose prediction under the *second* radar's folding best
    matches :math:`v_b`. With coprime folding intervals exactly one candidate
    agrees, and the search is a small exact enumeration rather than a modular
    inverse — clearer to read, and it degrades gracefully under noise instead of
    jumping to a distant wrong answer.

    Parameters
    ----------
    folded_velocity_a_mps, folded_velocity_b_mps : array_like
        The apparent velocities reported by the two bursts, metres/second. Must
        broadcast against one another.
    unambiguous_velocity_a_mps, unambiguous_velocity_b_mps : float
        The two half-intervals, metres/second. Both strictly positive, and they
        must differ — two equal rates fold identically and resolve nothing.
    max_velocity_mps : float
        Largest true speed to consider, metres/second. This is a physical prior
        about the target, and it bounds the search: without it the problem has
        infinitely many solutions. Must exceed the smaller half-interval.
    tolerance_mps : float
        How closely the second measurement must agree, metres/second. A sensible
        value is about half a Doppler bin of the coarser burst. Must be positive.

    Returns
    -------
    velocity_mps : numpy.ndarray
        The recovered true radial velocity, positive closing. Entries whose best
        candidate misses by more than ``tolerance_mps`` are ``nan`` — an
        unresolved measurement is reported as unresolved rather than guessed.
    residual_mps : numpy.ndarray
        How far the chosen candidate missed the second measurement. Small values
        mean a confident match; this is the number to threshold on when the two
        bursts may be seeing different targets.

    Raises
    ------
    ValueError
        If either half-interval is non-positive, if they are equal, if
        ``max_velocity_mps`` does not exceed the smaller of them, or if
        ``tolerance_mps`` is non-positive.

    See Also
    --------
    fold_velocity_mps : The forward model this inverts.

    Notes
    -----
    The reachable span is set by the PRF ratio, and specifically by the least
    common multiple of the two folding spans. For half-intervals in the ratio
    :math:`p:q` in lowest terms the pair of readings repeats after
    :math:`q \cdot 2 v_{ua,a}`, so velocities up to :math:`q\, v_{ua,a}` are
    recoverable. Scenario 001 S3's 5:6 bursts give
    :math:`6 \times 38.24 \approx 229` m/s, comfortably outside the ±191.2 m/s
    the scenario asks for — the scenario's figure is a design target chosen to
    match S2, not this function's limit.

    Beyond the repeat the ambiguity returns, and a target outside
    ``max_velocity_mps`` is *not* simply missed: it can alias onto a candidate
    that is inside the bound and be returned with a near-zero residual. A
    target at 300 m/s on the S3 bursts comes back as -158.88 m/s and looks
    confident. That is why the bound is a required argument and not a generous
    default — it is a claim about the target, and a wrong claim produces a
    wrong answer rather than a missing one.

    Examples
    --------
    >>> import numpy as np
    >>> v_ua_a, v_ua_b = 38.24, 45.89
    >>> true_mps = 80.0
    >>> v, residual = unfold_doppler_dual_prf(
    ...     fold_velocity_mps(true_mps, v_ua_a),
    ...     fold_velocity_mps(true_mps, v_ua_b),
    ...     v_ua_a,
    ...     v_ua_b,
    ...     max_velocity_mps=191.0,
    ...     tolerance_mps=1.0,
    ... )
    >>> bool(np.isclose(v, true_mps, atol=1e-9))
    True
    """
    if unambiguous_velocity_a_mps <= 0.0 or unambiguous_velocity_b_mps <= 0.0:
        msg = "both unambiguous velocities must be strictly positive."
        raise ValueError(msg)
    if unambiguous_velocity_a_mps == unambiguous_velocity_b_mps:
        msg = (
            "the two unambiguous velocities must differ; equal repetition rates fold "
            "identically and resolve no ambiguity."
        )
        raise ValueError(msg)
    if tolerance_mps <= 0.0:
        msg = f"tolerance_mps must be strictly positive; got {tolerance_mps!r}."
        raise ValueError(msg)
    smaller_half_interval = min(unambiguous_velocity_a_mps, unambiguous_velocity_b_mps)
    if max_velocity_mps <= smaller_half_interval:
        msg = (
            f"max_velocity_mps ({max_velocity_mps!r}) must exceed the smaller half-interval "
            f"({smaller_half_interval!r}); below it there is no ambiguity to resolve."
        )
        raise ValueError(msg)

    velocity_a = np.asarray(folded_velocity_a_mps, dtype=np.float64)
    velocity_b = np.asarray(folded_velocity_b_mps, dtype=np.float64)
    velocity_a, velocity_b = np.broadcast_arrays(velocity_a, velocity_b)

    # Candidate folds of burst A that stay inside the physical speed bound.
    span_a = 2.0 * unambiguous_velocity_a_mps
    n_folds = int(np.ceil((max_velocity_mps + unambiguous_velocity_a_mps) / span_a))
    fold_index = np.arange(-n_folds, n_folds + 1, dtype=np.float64)

    # (..., n_candidates) by broadcasting the fold axis against the measurements.
    candidate_mps = velocity_a[..., None] + fold_index * span_a
    predicted_b_mps = fold_velocity_mps(candidate_mps, unambiguous_velocity_b_mps)

    # Compare on the circle: a candidate near +v_ua_b and a measurement near
    # -v_ua_b are neighbours, not opposites.
    residual_mps = np.abs(
        fold_velocity_mps(predicted_b_mps - velocity_b[..., None], unambiguous_velocity_b_mps)
    )
    residual_mps = np.where(np.abs(candidate_mps) <= max_velocity_mps, residual_mps, np.inf)

    best = np.argmin(residual_mps, axis=-1)
    best_residual_mps = np.take_along_axis(residual_mps, best[..., None], axis=-1)[..., 0]
    best_velocity_mps = np.take_along_axis(candidate_mps, best[..., None], axis=-1)[..., 0]

    resolved: NDArray[np.float64] = np.where(
        best_residual_mps <= tolerance_mps, best_velocity_mps, np.nan
    )
    return resolved, np.asarray(best_residual_mps, dtype=np.float64)
