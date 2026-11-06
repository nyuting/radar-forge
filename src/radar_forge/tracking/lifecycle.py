r"""When a track is confirmed or deleted: M-of-N with miss counts, or by existence (JIPDA).

:class:`LifecyclePolicy` holds every M-of-N setting, the re-acquisition window and
covariance-based deletion (tracker-001 §5.4); :class:`ExistencePolicy` confirms and deletes by
the probability that a track's target exists. :func:`m_of_n` and :func:`by_existence` move a
bank's statuses by them; :class:`~radar_forge.tracking.tracker.Tracker` applies them each scan.

References
----------
.. [1] Y. Bar-Shalom and X. R. Li, *Multitarget-Multisensor Tracking: Principles and
       Techniques*, YBS Publishing, 1995.
.. [2] D. Mušicki and R. Evans, "Joint integrated probabilistic data association: JIPDA,"
       *IEEE Trans. Aerosp. Electron. Syst.*, vol. 40, no. 3, pp. 1093-1099, 2004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from radar_forge.tracking.filters import TrackStatus

__all__ = ["ExistencePolicy", "LifecyclePolicy", "by_existence", "m_of_n"]


@dataclass(frozen=True)
class LifecyclePolicy:
    """When a track is confirmed (M hits in its first N scans) or deleted.

    ``n_reacquire_misses`` extra misses in a row are allowed a confirmed track only.
    ``reacquire_reset`` names coordinates whose variances return to the prior, and
    cross-covariances to zero, on the miss that opens re-acquisition (tracker-001 §5.4).
    ``max_position_sigma_m`` deletes a confirmed track once the square root of the largest
    eigenvalue of its ENU position covariance exceeds it after the update; None never does. A
    tentative track is spared: under PDA its covariance is wide until it has a velocity.

    References
    ----------
    Bar-Shalom and Li [1], §2.6.1.
    """

    n_confirm_hits: int = 3
    n_confirm_scans: int = 5
    n_delete_misses: int = 5
    n_reacquire_misses: int = 0
    max_coast_time_s: float = 30.0
    reacquire_reset: tuple[str, ...] = ()
    max_position_sigma_m: float | None = None

    def __post_init__(self) -> None:
        """Reject a policy that could not work."""
        counts = (self.n_confirm_hits, self.n_confirm_scans, self.n_delete_misses)
        # A bool is an int in Python, so the exact type is tested.
        if any(type(n) is not int for n in (*counts, self.n_reacquire_misses)) or not (
            1 <= self.n_confirm_hits <= self.n_confirm_scans
            and self.n_delete_misses >= 1
            and self.n_reacquire_misses >= 0
        ):
            msg = f"need int counts, 1 <= M <= N, n_delete_misses >= 1; got {counts}."
            raise ValueError(msg)
        names, sigma_m = self.reacquire_reset, self.max_position_sigma_m
        if not (math.isfinite(self.max_coast_time_s) and self.max_coast_time_s > 0) or not (
            isinstance(names, tuple) and len(set(names)) == len(names) and all(names)
        ):
            msg = f"need max_coast_time_s > 0 and unique reset names; got {names!r}."
            raise ValueError(msg)
        if sigma_m is not None and not (math.isfinite(sigma_m) and sigma_m > 0):
            msg = f"max_position_sigma_m must be None or positive and finite; got {sigma_m}."
            raise ValueError(msg)


@dataclass(frozen=True)
class ExistencePolicy:
    """Confirm and delete by the probability that a track's target exists (JIPDA).

    Each scan that sees a track multiplies its existence by ``p_survival`` before the
    association updates it; a track is confirmed at ``confirm`` or more and deleted below
    ``delete``. A new track starts at ``initial``. It replaces M-of-N, and needs an associator
    that weights.

    References
    ----------
    Mušicki and Evans [2] (the existence prediction and the confirm and delete thresholds).
    """

    p_survival: float = 0.995
    confirm: float = 0.95
    delete: float = 0.05
    initial: float = 0.2

    def __post_init__(self) -> None:
        """Need 0 < delete < initial and delete < confirm <= 1, survival in (0, 1]."""
        if not (
            0 < self.p_survival <= 1
            and 0 < self.delete < self.initial <= 1
            and self.delete < self.confirm <= 1
        ):
            msg = f"need 0 < delete < initial, confirm <= 1 and 0 < p_survival <= 1; got {self}."
            raise ValueError(msg)


def m_of_n(
    status: NDArray[np.int8],
    n_hits: NDArray[np.int32],
    n_scans: NDArray[np.int32],
    policy: LifecyclePolicy,
) -> NDArray[np.int8]:
    """Confirm a tentative row on its M-th hit; delete it once M is out of reach."""
    tentative = status == TrackStatus.TENTATIVE
    confirm = tentative & (n_hits >= policy.n_confirm_hits)
    reachable = n_hits + np.maximum(policy.n_confirm_scans - n_scans, 0)
    result = status.copy()
    result[confirm] = TrackStatus.CONFIRMED
    result[tentative & ~confirm & (reachable < policy.n_confirm_hits)] = TrackStatus.DELETED
    return result


def by_existence(
    status: NDArray[np.int8], existence: NDArray[np.float64], policy: ExistencePolicy
) -> NDArray[np.int8]:
    """Confirm a tentative row at the confirm threshold; delete any row below delete."""
    result = status.copy()
    result[(status == TrackStatus.TENTATIVE) & (existence >= policy.confirm)] = (
        TrackStatus.CONFIRMED
    )
    result[existence < policy.delete] = TrackStatus.DELETED
    return result
