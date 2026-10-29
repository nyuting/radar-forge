r"""Resolving range and Doppler ambiguity by combining two repetition rates.

A single pulse repetition frequency buys either unambiguous range or
unambiguous Doppler, never both: their product is fixed at
:math:`c\lambda/8` by the carrier alone, so no choice of PRF escapes the trade.
Scenario 001 S1 and S2 each accept one half of that loss.

The way out is to measure twice at two different repetition rates. A target
folded at both rates produces a *pair* of false readings, and if the two
folding periods are coprime multiples of a common base, only one true value is
consistent with both. This is the radar form of the Chinese remainder theorem,
and it is what scenario 001 S3 uses to recover an aircraft's real velocity from
two FMCW bursts neither of which could measure it alone.

Range and Doppler fold the same way, so the module solves the problem once,
for any period: :func:`fold_into_period` is the forward model and
:func:`unfold_dual_period` the inverse. Velocity folds into an interval
centred on zero and range into one that starts at zero, and that, the units
and the physical caveats are all that separate the two named wrappers,
:func:`unfold_doppler_dual_prf` and :func:`unfold_range_dual_prf`.

The extension is not free. It costs dwell time, it fails when two targets are
present at the same range (the pairs become ambiguous between targets, the
classic "ghost" problem), and it degrades sharply once measurement noise
approaches half a bin. The first two are out of scope for a single-target
scenario; the third is why :func:`unfold_dual_period` takes an explicit
tolerance and reports the residual rather than silently returning its best
guess.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §5.5.4 (ambiguity resolution).
.. [2] G. W. Stimson, *Introduction to Airborne Radar*, 2nd ed., SciTech, 1998,
       ch. 12 (resolving range ambiguities, eliminating ghosts), ch. 21
       (resolving Doppler ambiguities), ch. 25 (the crucial choice of PRF),
       ch. 27 (eliminating blind zones).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "fold_into_period",
    "fold_velocity_mps",
    "unfold_doppler_dual_prf",
    "unfold_dual_period",
    "unfold_range_dual_prf",
]


def fold_into_period(
    value: ArrayLike,
    period: ArrayLike,
    *,
    lower: ArrayLike,
) -> NDArray[np.float64]:
    r"""Wrap a true value into the half-open interval ``[lower, lower + period)``.

    The forward model of every kind of folding a radar does: a reading that
    repeats every ``period`` and is reported in one interval of that length.
    Doppler folds into an interval centred on zero, ``lower = -period / 2``;
    range folds into one that starts at zero, ``lower = 0``. Any unit will do,
    as long as all three arguments share it.

    Parameters
    ----------
    value : array_like
        The true value.
    period : array_like
        How often the reading repeats. Must be strictly positive.
    lower : array_like
        The bottom of the reporting interval, inclusive.

    Returns
    -------
    numpy.ndarray
        The reported value, in ``[lower, lower + period)``.

    Raises
    ------
    ValueError
        If any period is non-positive.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isclose(fold_into_period(70.0, 60.0, lower=0.0), 10.0))
    True
    >>> bool(np.isclose(fold_into_period(40.0, 60.0, lower=-30.0), -20.0))
    True
    """
    value_arr = np.asarray(value, dtype=np.float64)
    period_arr = np.asarray(period, dtype=np.float64)
    lower_arr = np.asarray(lower, dtype=np.float64)
    if np.any(period_arr <= 0.0):
        msg = "period must be strictly positive."
        raise ValueError(msg)
    result: NDArray[np.float64] = np.mod(value_arr - lower_arr, period_arr) + lower_arr
    return result


def unfold_dual_period(
    folded_a: ArrayLike,
    folded_b: ArrayLike,
    period_a: float,
    period_b: float,
    *,
    lower_bound: float,
    upper_bound: float,
    tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Recover a true value from two readings that fold at different periods.

    Enumerates the candidates consistent with the first reading,

    .. math:: x_k = a + k\, P_a,
        \qquad x_{\mathrm{lo}} \le x_k \le x_{\mathrm{hi}},

    and selects the one whose prediction under the *second* reading's folding
    best matches :math:`b`. With coprime periods exactly one candidate agrees,
    and the search is a small exact enumeration rather than a modular inverse:
    clearer to read, and it degrades gracefully under noise instead of jumping
    to a distant wrong answer.

    Nothing here knows whether the value is a range or a velocity. Any unit
    will do, as long as every argument shares it; the named wrappers
    :func:`unfold_range_dual_prf` and :func:`unfold_doppler_dual_prf` fix the
    unit and the interval.

    Parameters
    ----------
    folded_a, folded_b : array_like
        The two readings. Must broadcast against one another: a column against
        a row gives every pairing of the two.
    period_a, period_b : float
        How often each reading repeats. Both strictly positive.
    lower_bound, upper_bound : float
        The interval, both ends inclusive, the true value is known to lie in.
        A physical prior about the target, and it bounds the search: without it
        the problem has infinitely many solutions.
    tolerance : float
        How closely the second reading must agree. A sensible value is about
        half a bin of the coarser reading. Must be positive.

    Returns
    -------
    value : numpy.ndarray
        The recovered true value. Entries whose best candidate misses by more
        than ``tolerance`` are ``nan``: an unresolved measurement is reported as
        unresolved rather than guessed.
    residual : numpy.ndarray
        How far the chosen candidate missed the second reading, measured round
        its period. Small values mean a confident match; this is the number to
        threshold on when the two readings may be of different targets.

    Raises
    ------
    ValueError
        If either period or the tolerance is non-positive, if
        ``upper_bound`` is below ``lower_bound``, or if the periods are equal
        and the bounds are more than one period apart, so that two candidates
        would read the same on both.

    See Also
    --------
    fold_into_period : The forward model this inverts.

    Notes
    -----
    The reachable span is the least common multiple of the two periods. For
    periods in the ratio :math:`p:q` in lowest terms the pair of readings
    repeats after :math:`q P_a = p P_b`, so any interval of that length is
    resolvable. Scenario 001 S3's 5:6 bursts repeat after
    :math:`6 \times 2 \times 38.24 \approx 459` m/s in velocity and
    :math:`5 \times 60 = 6 \times 50 = 300` km in range.

    Beyond the repeat the ambiguity returns, and a target outside the bounds is
    *not* simply missed: it can alias onto a candidate that is inside them and
    be returned with a near-zero residual. That is why the bounds are required
    arguments and not generous defaults. They are a claim about the target,
    and a wrong claim produces a wrong answer rather than a missing one.

    Examples
    --------
    Periods of 6 and 5 repeat after 30, so a reading of 4 on the first and 2
    on the second is resolvable anywhere in ``[0, 30)``:

    >>> import numpy as np
    >>> true = 22.0
    >>> value, residual = unfold_dual_period(
    ...     fold_into_period(true, 6.0, lower=0.0),
    ...     fold_into_period(true, 5.0, lower=0.0),
    ...     6.0,
    ...     5.0,
    ...     lower_bound=0.0,
    ...     upper_bound=29.0,
    ...     tolerance=0.1,
    ... )
    >>> bool(np.isclose(value, true, atol=1e-9))
    True
    """
    if period_a <= 0.0 or period_b <= 0.0:
        msg = f"both periods must be strictly positive; got {period_a!r} and {period_b!r}."
        raise ValueError(msg)
    if tolerance <= 0.0:
        msg = f"tolerance must be strictly positive; got {tolerance!r}."
        raise ValueError(msg)
    if upper_bound < lower_bound:
        msg = f"upper_bound ({upper_bound!r}) is below lower_bound ({lower_bound!r})."
        raise ValueError(msg)
    if period_a == period_b and upper_bound - lower_bound > period_a:
        msg = (
            f"equal periods ({period_a!r}) cannot tell apart values more than one period "
            f"apart, and the bounds span {upper_bound - lower_bound!r}."
        )
        raise ValueError(msg)

    reading_a = np.asarray(folded_a, dtype=np.float64)
    reading_b = np.asarray(folded_b, dtype=np.float64)
    reading_a, reading_b = np.broadcast_arrays(reading_a, reading_b)

    # Bring reading A to the bottom fold of the bounds, so that the candidates
    # are k = 0, 1, ... up, the same count for every reading, wherever A's
    # radar chose to report it.
    base_a = fold_into_period(reading_a, period_a, lower=lower_bound)
    n_folds = int(np.ceil((upper_bound - lower_bound) / period_a))
    fold_index = np.arange(n_folds + 1, dtype=np.float64)

    # (..., n_candidates) by broadcasting the fold axis against the readings.
    candidate = base_a[..., None] + fold_index * period_a

    # Compare on the circle: a prediction just below a fold edge and a reading
    # just above it are neighbours, not a whole period apart. Folding the
    # difference into [-P/2, P/2) measures it the shorter way round.
    residual = np.abs(
        fold_into_period(candidate - reading_b[..., None], period_b, lower=-period_b / 2.0)
    )
    residual = np.where(candidate <= upper_bound, residual, np.inf)

    best = np.argmin(residual, axis=-1)
    best_residual = np.take_along_axis(residual, best[..., None], axis=-1)[..., 0]
    best_value = np.take_along_axis(candidate, best[..., None], axis=-1)[..., 0]

    resolved: NDArray[np.float64] = np.where(best_residual <= tolerance, best_value, np.nan)
    return resolved, np.asarray(best_residual, dtype=np.float64)


def fold_velocity_mps(
    velocity_mps: ArrayLike,
    unambiguous_velocity_mps: ArrayLike,
) -> NDArray[np.float64]:
    r"""Wrap a true radial velocity into a radar's unambiguous interval.

    The forward model of Doppler aliasing: what a radar with a half-interval of
    :math:`v_{ua}` would *report* for a target actually travelling at
    :math:`v`. Used to predict where a folded target should appear, and as the
    inverse against which :func:`unfold_doppler_dual_prf` is tested. It is
    :func:`fold_into_period` with period :math:`2 v_{ua}`, centred on zero.

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
    half_interval = np.asarray(unambiguous_velocity_mps, dtype=np.float64)
    if np.any(half_interval <= 0.0):
        msg = "unambiguous_velocity_mps must be strictly positive."
        raise ValueError(msg)
    return fold_into_period(velocity_mps, 2.0 * half_interval, lower=-half_interval)


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

    :func:`unfold_dual_period` with each burst's period :math:`2 v_{ua}` and
    the bounds :math:`\pm v_{\max}`: the candidates are

    .. math:: v_k = v_a + 2 k\, v_{ua,a},
        \qquad |v_k| \le v_{\max},

    and the one whose fold under the second burst best matches :math:`v_b` is
    kept.

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
    unfold_dual_period : The solver, and the span it reaches.

    Notes
    -----
    For half-intervals in the ratio :math:`p:q` in lowest terms, velocities up
    to :math:`q\, v_{ua,a}` are recoverable. Scenario 001 S3's 5:6 bursts give
    :math:`6 \times 38.24 = 5 \times 45.89 \approx 229` m/s.

    A target beyond that, or outside ``max_velocity_mps``, can alias onto a
    candidate inside the bound and come back confident: a target at 300 m/s on
    the S3 bursts comes back as -158.88 m/s. That is why the bound is a
    required argument and not a generous default.

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
    return unfold_dual_period(
        folded_velocity_a_mps,
        folded_velocity_b_mps,
        2.0 * unambiguous_velocity_a_mps,
        2.0 * unambiguous_velocity_b_mps,
        lower_bound=-max_velocity_mps,
        upper_bound=max_velocity_mps,
        tolerance=tolerance_mps,
    )


def unfold_range_dual_prf(
    folded_range_a_m: ArrayLike,
    folded_range_b_m: ArrayLike,
    span_a_m: float,
    span_b_m: float,
    *,
    max_range_m: float,
    tolerance_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Recover true range from two folded measurements.

    :func:`unfold_dual_period` with each burst's range span as its period and
    the bounds :math:`[0, R_{\max}]`: the candidates are

    .. math:: R_k = R_a + k\, S_a,
        \qquad 0 \le R_k \le R_{\max},

    and the one whose fold under the second burst best matches :math:`R_b` is
    kept. A burst's range axis is a circle of circumference :math:`S`, so a
    target at :math:`R` reads :math:`R \bmod S` [1]_.

    Parameters
    ----------
    folded_range_a_m, folded_range_b_m : array_like
        The ranges read off the two bursts' maps, metres. Must broadcast
        against one another: a column against a row gives every pairing.
    span_a_m, span_b_m : float
        The circumference of each burst's range axis, metres: the number of
        range bins times the bin width. Both strictly positive. They may be
        equal, but then nothing beyond one span can be resolved, and
        ``max_range_m`` must not exceed it.
    max_range_m : float
        Largest true range to consider, metres. A claim about the target, as
        ``max_velocity_mps`` is for :func:`unfold_doppler_dual_prf`, and with
        the same consequence when it is wrong.
    tolerance_m : float
        How closely the second measurement must agree, metres. A bin or two of
        the coarser map. Must be positive.

    Returns
    -------
    range_m : numpy.ndarray
        The recovered true range. ``nan`` where the best candidate misses by
        more than ``tolerance_m``.
    residual_m : numpy.ndarray
        How far the chosen candidate missed the second measurement, the shorter
        way round the second map's circle.

    Raises
    ------
    ValueError
        As :func:`unfold_dual_period`: a non-positive span, a non-positive
        tolerance, a negative ``max_range_m``, or equal spans with
        ``max_range_m`` beyond one span.

    See Also
    --------
    fold_into_period : The forward model, with ``lower = 0``.
    unfold_dual_period : The solver, and the span it reaches.

    Notes
    -----
    Scenario 001 S3's two FMCW maps span 60 and 50 km, so the pair repeats
    after 300 km. A real FMCW radar cannot use most of that: a sawtooth sweep
    needs the echo's delay shorter than the chirp, which caps range at
    :math:`cT/2`, below either map's span for S3. The simulator does not model
    that limit (``spec/scenario-003-tracking.md`` §14.11). The reach this
    function adds matters mainly for pulsed multi-PRF designs, where each
    span is :math:`c/2\,\mathrm{PRF}` and nothing else caps the range.

    References
    ----------
    .. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
           McGraw-Hill, 2014, §5.5.4 (range ambiguity and its resolution).

    Examples
    --------
    >>> import numpy as np
    >>> true_m = 120_000.0
    >>> range_m, residual_m = unfold_range_dual_prf(
    ...     fold_into_period(true_m, 60_000.0, lower=0.0),
    ...     fold_into_period(true_m, 50_000.0, lower=0.0),
    ...     60_000.0,
    ...     50_000.0,
    ...     max_range_m=300_000.0,
    ...     tolerance_m=50.0,
    ... )
    >>> bool(np.isclose(range_m, true_m))
    True
    """
    if max_range_m < 0.0:
        msg = f"max_range_m must be non-negative; got {max_range_m!r}."
        raise ValueError(msg)
    return unfold_dual_period(
        folded_range_a_m,
        folded_range_b_m,
        span_a_m,
        span_b_m,
        lower_bound=0.0,
        upper_bound=max_range_m,
        tolerance=tolerance_m,
    )
