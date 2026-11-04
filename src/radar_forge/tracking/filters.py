r"""The Gaussian filter over a bank of tracks: the KF, the EKF and the UKF in one flow.

A track is a row of a :class:`TrackBank`, a struct of arrays, and every step runs over every
row at once:

- :func:`predict`: :math:`\hat{x}^- = f(\hat{x})`, :math:`P^- = F P F^T + Q`.
- :func:`innovate`: every row against every measurement of a scan, the innovation
  :math:`\nu = z - \hat{z}` ``(n_tracks, n_meas, m)``, :math:`S = H P^- H^T + R`, the NIS and
  the log-likelihood, from one batched Cholesky factorisation and no loop over pairs.
- :func:`update`: the one correction core, for the rows given a measurement.

The three filters differ only in their :class:`Linearisation` [2]_, how a mean and covariance
go through a function: :class:`TaylorLinearisation` (the KF on linear models, the EKF
otherwise) or :class:`UnscentedTransform` (the UKF). Nothing here asks which one it holds; the
correction core asks only whether a measurement matrix H exists.

Validation happens once, in :meth:`TrackBank.append`; predict, innovate and update trust their
inputs. Scenario 011 plugs in without new files: an IMM mixes banks of mode rows, and JPDA
passes association weights to :func:`update`.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001, §5.2.4 (the Kalman filter), §5.2.6 (the innovations and
       the likelihood function), §5.4.2 (the NIS) and §10.3 (the extended Kalman filter).
.. [2] S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013, ch. 5
       (the extended and unscented Kalman filters) and ch. 6 (general Gaussian filtering).
.. [3] E. A. Wan and R. van der Merwe, "The unscented Kalman filter for nonlinear estimation,"
       *Proc. IEEE Adaptive Systems for Signal Processing, Communications, and Control
       Symposium*, 2000, pp. 153-158, §3 (the UKF and its weights).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import IntEnum
from functools import cache
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.tracking.measurement_models import MeasurementModel
from radar_forge.tracking.models import MotionModel, StateLayout

__all__ = [
    "Innovation",
    "Linearisation",
    "MeasurementMoments",
    "TaylorLinearisation",
    "TrackBank",
    "TrackSnapshot",
    "TrackStatus",
    "UnscentedTransform",
    "innovate",
    "predict",
    "update",
]

# The roundoff a covariance may carry, relative to its largest entry: products of a few small
# matrices carry about n float64 epsilons, 1e-15, and 1e-10 leaves a wide margin over that.
_ROUNDOFF_TOLERANCE = 1e-10


class TrackStatus(IntEnum):
    """Where a track is in its life, stored as int8 in :attr:`TrackBank.status`.

    ``TENTATIVE`` has not passed M-of-N; ``CONFIRMED`` has, and was hit this scan;
    ``COASTING`` is confirmed but missed this scan; ``DELETED`` is reported once, in the scan
    it dies, and then dropped.
    """

    TENTATIVE = 0
    CONFIRMED = 1
    COASTING = 2
    DELETED = 3


@dataclass(frozen=True, eq=False)
class TrackSnapshot:
    """A frozen copy of one track after a scan, for callers that want one object per track.

    Attributes
    ----------
    track_id : int
        Fixed for the track's life.
    status : TrackStatus
        After the scan.
    mean, covariance : numpy.ndarray
        Read-only, ``(n_state,)`` and ``(n_state, n_state)``.
    time_s : float
        Time of the estimate.
    nis : float or None
        NIS of this scan's update, or None without one.
    measurement_index : int or None
        The scan row the track took or was born from, or None.
    n_hits, n_misses : int
        Scans with a measurement since birth, and misses in a row.
    """

    track_id: int
    status: TrackStatus
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    time_s: float
    nis: float | None
    measurement_index: int | None
    n_hits: int
    n_misses: int


def _transpose(matrix: NDArray[Any]) -> NDArray[Any]:
    """Swap the last two axes of a stack of matrices."""
    return np.swapaxes(matrix, -1, -2)


def _symmetrise(matrix: NDArray[Any]) -> NDArray[np.float64]:
    """Return the symmetric part of a stack of matrices, removing asymmetric roundoff."""
    result: NDArray[np.float64] = (matrix + _transpose(matrix)) / 2
    return result


def _check_covariances(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    """Return a float64 copy of a stack of covariances, checked symmetric and semidefinite."""
    covariance = np.array(value, dtype=np.float64)
    if covariance.shape != shape or not np.all(np.isfinite(covariance)):
        msg = f"{name} must be finite with shape {shape}; got shape {covariance.shape}."
        raise ValueError(msg)
    allowance = _ROUNDOFF_TOLERANCE * np.max(np.abs(covariance), axis=(-2, -1), initial=0.0)
    asymmetry = np.max(np.abs(covariance - _transpose(covariance)), axis=(-2, -1), initial=0.0)
    if np.any(asymmetry > allowance):
        msg = f"{name} must be symmetric to within roundoff."
        raise ValueError(msg)
    covariance = _symmetrise(covariance)
    if np.any(np.linalg.eigvalsh(covariance)[..., 0] < -allowance):
        msg = f"{name} must be positive semidefinite; one has a negative eigenvalue."
        raise ValueError(msg)
    return covariance


def _column(value: ArrayLike, n_rows: int, dtype: type[np.generic], name: str) -> NDArray[Any]:
    """Return one column of new rows, checked to have one entry per row."""
    column = np.array(value, dtype=dtype)
    if column.shape != (n_rows,):
        msg = f"{name} must have shape ({n_rows},); got {column.shape}."
        raise ValueError(msg)
    return column


@dataclass(frozen=True, eq=False)
class TrackBank:
    """Every track as one row of a struct of arrays; treat it as immutable.

    Every function returns a new bank. Build one with :meth:`empty` and :meth:`append`, which
    validate; the constructor trusts its arguments.

    Attributes
    ----------
    layout : StateLayout
        Shared by every row.
    mean, covariance : numpy.ndarray
        ``(n_tracks, n_state)`` and ``(n_tracks, n_state, n_state)``.
    time_s, last_hit_time_s : numpy.ndarray
        Time of each estimate and of its last measurement, ``(n_tracks,)``.
    track_id : numpy.ndarray
        int64 identifiers.
    status : numpy.ndarray
        int8 :class:`TrackStatus` values.
    n_hits, n_misses, n_scans : numpy.ndarray
        int32 counts: scans with a measurement, misses in a row, scans taken part in; birth
        counts as a scan and a hit.
    nis : numpy.ndarray
        NIS of this scan's update, NaN without one.
    measurement_index : numpy.ndarray
        int64 scan row each track took or was born from this scan, -1 for none.

    Examples
    --------
    >>> from radar_forge.tracking.models import RadialConstantVelocity
    >>> bank = TrackBank.empty(RadialConstantVelocity().state_layout)
    >>> bank.append([[1000.0, 5.0]], [np.eye(2)], 0.0, [7], [0]).track_id
    array([7])
    """

    layout: StateLayout
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    time_s: NDArray[np.float64]
    track_id: NDArray[np.int64]
    status: NDArray[np.int8]
    n_hits: NDArray[np.int32]
    n_misses: NDArray[np.int32]
    n_scans: NDArray[np.int32]
    last_hit_time_s: NDArray[np.float64]
    nis: NDArray[np.float64]
    measurement_index: NDArray[np.int64]

    @classmethod
    def empty(cls, layout: StateLayout) -> TrackBank:
        """Return a bank with no rows in ``layout``."""
        n_state = layout.dimension
        columns = {f.name: np.zeros(0, dtype=_DTYPES[f.name]) for f in fields(cls)[3:]}
        return cls(layout, np.zeros((0, n_state)), np.zeros((0, n_state, n_state)), **columns)

    @property
    def n_tracks(self) -> int:
        """Number of rows."""
        return len(self.track_id)

    def append(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
        time_s: float,
        track_id: ArrayLike,
        status: ArrayLike,
        measurement_index: ArrayLike | None = None,
    ) -> TrackBank:
        """Return the bank with new rows, each born from one measurement or cue.

        This is where state enters the bank, so it is validated here, once.

        Parameters
        ----------
        mean : array_like
            ``(n_new, n_state)``.
        covariance : array_like
            ``(n_new, n_state, n_state)``, symmetric positive semidefinite.
        time_s : float
            Their time.
        track_id, status : array_like
            ``(n_new,)`` identifiers and :class:`TrackStatus` values.
        measurement_index : array_like or None, optional
            ``(n_new,)`` scan rows they came from; None for -1 (none).

        Returns
        -------
        TrackBank
            The old rows, then the new.

        Raises
        ------
        ValueError
            If a shape is wrong, a value is not finite, or a covariance is not symmetric
            positive semidefinite to within roundoff.
        """
        n_state = self.layout.dimension
        new_mean = np.array(mean, dtype=np.float64)
        if new_mean.ndim != 2 or new_mean.shape[1] != n_state or not np.all(np.isfinite(new_mean)):
            msg = f"mean must be finite with shape (n_new, {n_state}); got {new_mean.shape}."
            raise ValueError(msg)
        if not np.isfinite(time_s):
            msg = f"time_s must be finite; got {time_s}."
            raise ValueError(msg)
        n_new = len(new_mean)
        index = np.full(n_new, -1) if measurement_index is None else measurement_index
        new = {
            "mean": new_mean,
            "covariance": _check_covariances(covariance, (n_new, n_state, n_state), "covariance"),
            "time_s": np.full(n_new, time_s, dtype=np.float64),
            "track_id": _column(track_id, n_new, np.int64, "track_id"),
            "status": _column(status, n_new, np.int8, "status"),
            "n_hits": np.ones(n_new, dtype=np.int32),
            "n_misses": np.zeros(n_new, dtype=np.int32),
            "n_scans": np.ones(n_new, dtype=np.int32),
            "last_hit_time_s": np.full(n_new, time_s, dtype=np.float64),
            "nis": np.full(n_new, np.nan),
            "measurement_index": _column(index, n_new, np.int64, "measurement_index"),
        }
        columns = {name: np.concatenate([getattr(self, name), new[name]]) for name in new}
        return TrackBank(self.layout, **columns)

    def compact(self, keep: ArrayLike) -> TrackBank:
        """Return the rows where ``keep``, shape ``(n_tracks,)``, is True, in order."""
        rows = np.asarray(keep, dtype=bool)
        columns: dict[str, NDArray[Any]] = {
            f.name: getattr(self, f.name)[rows] for f in fields(self)[1:]
        }
        return TrackBank(self.layout, **columns)

    def snapshots(self) -> tuple[TrackSnapshot, ...]:
        """Return a frozen :class:`TrackSnapshot` of each row, in row order."""
        mean, covariance = self.mean.copy(), self.covariance.copy()
        mean.setflags(write=False)
        covariance.setflags(write=False)
        # One Python object per track, for callers such as the pipelines that want objects.
        return tuple(
            TrackSnapshot(
                int(self.track_id[i]),
                TrackStatus(int(self.status[i])),
                mean[i],
                covariance[i],
                float(self.time_s[i]),
                None if np.isnan(self.nis[i]) else float(self.nis[i]),
                None if self.measurement_index[i] < 0 else int(self.measurement_index[i]),
                int(self.n_hits[i]),
                int(self.n_misses[i]),
            )
            for i in range(self.n_tracks)
        )


_DTYPES = {
    "time_s": np.float64,
    "track_id": np.int64,
    "status": np.int8,
    "n_hits": np.int32,
    "n_misses": np.int32,
    "n_scans": np.int32,
    "last_hit_time_s": np.float64,
    "nis": np.float64,
    "measurement_index": np.int64,
}


@dataclass(frozen=True, eq=False)
class MeasurementMoments:
    r"""What a linearisation says about the measurement each row should produce.

    Attributes
    ----------
    predicted_measurement : numpy.ndarray
        :math:`\hat{z}`, ``(n_tracks, n_meas)``.
    cross_covariance : numpy.ndarray
        :math:`P_{xz}`, ``(n_tracks, n_state, n_meas)``.
    predicted_measurement_covariance : numpy.ndarray
        :math:`H P H^T`, or its sigma-point equivalent: S *without* R, which
        :func:`innovate` adds. ``(n_tracks, n_meas, n_meas)``.
    measurement_matrix : numpy.ndarray or None
        H or the Jacobian, ``(n_tracks, n_meas, n_state)``, when the linearisation has one.
    """

    predicted_measurement: NDArray[np.float64]
    cross_covariance: NDArray[np.float64]
    predicted_measurement_covariance: NDArray[np.float64]
    measurement_matrix: NDArray[np.float64] | None = None


class Linearisation(Protocol):
    """How a Gaussian filter carries every row's mean and covariance through a function."""

    def predict(
        self, motion: MotionModel, bank: TrackBank, dt_s: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return predicted means ``(n_tracks, n)`` and covariances ``(n_tracks, n, n)``."""
        ...

    def measure(self, model: MeasurementModel, bank: TrackBank) -> MeasurementMoments:
        """Return the moments of the measurement each row should produce."""
        ...


@dataclass(frozen=True)
class TaylorLinearisation:
    r"""First-order Taylor: the KF on linear models, the EKF otherwise.

    With F and H the Jacobians at the mean (F and H themselves on linear models) this is the
    textbook Kalman filter, line for line [1]_:

    .. math::

        \hat{x}^- = f(\hat{x}), \quad P^- = F P F^T + Q, \quad \hat{z} = h(\hat{x}^-), \quad
        P_{xz} = P^- H^T, \quad S - R = H P^- H^T.

    It hands H to the correction core, which then uses the Joseph form.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.2.4 and §10.3; see the module References.
    """

    def predict(
        self, motion: MotionModel, bank: TrackBank, dt_s: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Return :math:`f(\hat{x})` and :math:`F P F^T + Q` for every row."""
        transition = motion.jacobian(bank.mean, dt_s)
        noise = motion.process_noise(bank.mean, dt_s)
        mean = bank.layout.wrap(motion.transition(bank.mean, dt_s))
        return mean, _symmetrise(transition @ bank.covariance @ _transpose(transition) + noise)

    def measure(self, model: MeasurementModel, bank: TrackBank) -> MeasurementMoments:
        r"""Return :math:`h(\hat{x})`, :math:`P H^T`, :math:`H P H^T` and H for every row."""
        matrix = model.jacobian(bank.mean)
        cross = bank.covariance @ _transpose(matrix)
        return MeasurementMoments(model.predict(bank.mean), cross, matrix @ cross, matrix)


@cache
def _unscented_weights(
    alpha: float, beta: float, kappa: float, n_state: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return read-only mean and covariance weights, computed once per setting and size."""
    scale = alpha**2 * (n_state + kappa)
    if scale <= 0:
        msg = f"n_state + kappa must be positive; got {n_state + kappa}."
        raise ValueError(msg)
    mean_weights = np.full(2 * n_state + 1, 1 / (2 * scale))
    mean_weights[0] = 1 - n_state / scale
    covariance_weights = mean_weights.copy()
    covariance_weights[0] += 1 - alpha**2 + beta
    if covariance_weights[0] < 0:
        msg = (
            f"alpha={alpha}, beta={beta}, kappa={kappa} give the centre point a negative "
            f"covariance weight at n_state={n_state}; P - K S Kᵀ stays positive semidefinite "
            "only when every covariance weight is nonnegative. Use alpha nearer 1."
        )
        raise ValueError(msg)
    mean_weights.setflags(write=False)
    covariance_weights.setflags(write=False)
    return mean_weights, covariance_weights


@dataclass(frozen=True)
class UnscentedTransform:
    r"""The unscented transform, the linearisation of the UKF.

    It needs no derivatives [1]_. Each row gets :math:`2n + 1` sigma points,
    :math:`\chi_0 = \hat{x}` and :math:`\chi_{\pm i} = \hat{x} \pm \sqrt{c} L_i` with
    :math:`L_i` column i of the Cholesky factor of P and :math:`c = \alpha^2 (n + \kappa)`,
    every row at once as ``(n_tracks, 2n + 1, n)``. The moments are weighted sums over the
    pushed points, with :math:`W^{(m)}_0 = 1 - n/c`, :math:`W^{(c)}_0 = W^{(m)}_0 + 1 -
    \alpha^2 + \beta` and :math:`W_i = 1/(2c)`, cached per state size. Means are weighted
    offsets from the centre point, through the layout's ``residual`` and ``wrap``, so points
    straddling an azimuth wrap average the short way.

    **Every covariance weight must be nonnegative.** Without H the correction core uses
    :math:`P^- - K S K^T`, the Schur complement of S in the joint covariance, positive
    semidefinite only then; so :math:`W^{(c)}_0 < 0` is refused. The defaults give
    :math:`W^{(m)}_0 = 0`, :math:`W^{(c)}_0 = 2`, :math:`W_i = 1/(2n)`; the usual small
    :math:`\alpha` of [1]_ is refused on purpose.

    **On linear models the UKF is the KF.** The ± pairs cancel through a linear map, the
    centre point's deviation is zero so :math:`\beta` drops out, and :math:`\sum_i 2 \cdot
    \frac{1}{2c} c F L_i L_i^T F^T = F P F^T`: the moments are :class:`TaylorLinearisation`'s,
    and the two covariance forms agree at the optimal gain (:func:`update`).

    Parameters
    ----------
    alpha : float, default 1.0
        Spread of the points; positive.
    beta : float, default 2.0
        Prior-distribution term, 2 for a Gaussian; nonnegative.
    kappa : float, default 0.0
        Secondary spread; ``n + kappa`` must be positive.

    Raises
    ------
    ValueError
        If a setting is not finite, alpha is not positive or beta is negative.

    References
    ----------
    .. [1] Wan and van der Merwe (2000), §3; Särkkä (2013), ch. 5. See the module References.

    Examples
    --------
    >>> UnscentedTransform().weights(2)
    (array([0.  , 0.25, 0.25, 0.25, 0.25]), array([2.  , 0.25, 0.25, 0.25, 0.25]))
    """

    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        """Check the settings that do not depend on the state size."""
        finite = np.all(np.isfinite([self.alpha, self.beta, self.kappa]))
        if not (finite and self.alpha > 0 and self.beta >= 0):
            msg = f"need finite alpha > 0 and beta >= 0; got {self.alpha}, {self.beta}."
            raise ValueError(msg)

    def weights(self, n_state: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the read-only mean and covariance weights, each ``(2 n_state + 1,)``.

        Raises
        ------
        ValueError
            If ``n_state + kappa`` is not positive or the centre covariance weight is negative.
        """
        return _unscented_weights(self.alpha, self.beta, self.kappa, n_state)

    def sigma_points(self, bank: TrackBank) -> NDArray[np.float64]:
        """Return every row's sigma points, ``(n_tracks, 2 n_state + 1, n_state)``."""
        n_state = bank.layout.dimension
        # Row i of Lᵀ is column i of L, so each row of the offsets is one spread direction.
        root = _transpose(np.linalg.cholesky(bank.covariance))
        offsets = np.sqrt(self.alpha**2 * (n_state + self.kappa)) * root
        deviations = np.concatenate([np.zeros_like(offsets[:, :1]), offsets, -offsets], axis=1)
        return bank.layout.wrap(bank.mean[:, None, :] + deviations)

    def _moments(
        self, points: NDArray[np.float64], layout: StateLayout
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the weighted mean about the centre point, and each point's deviation."""
        mean_weights, _ = self.weights((points.shape[1] - 1) // 2)
        centre = points[:, :1, :]
        mean = layout.wrap(centre[:, 0] + mean_weights @ layout.residual(points, centre))
        return mean, layout.residual(points, mean[:, None, :])

    def predict(
        self, motion: MotionModel, bank: TrackBank, dt_s: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the weighted mean and covariance of the moved points, plus Q."""
        _, covariance_weights = self.weights(bank.layout.dimension)
        moved = motion.transition(self.sigma_points(bank), dt_s[:, None])
        mean, deviation = self._moments(moved, bank.layout)
        spread = np.einsum("k,tki,tkj->tij", covariance_weights, deviation, deviation)
        return mean, _symmetrise(spread + motion.process_noise(bank.mean, dt_s))

    def measure(self, model: MeasurementModel, bank: TrackBank) -> MeasurementMoments:
        r"""Return :math:`\hat{z}`, :math:`P_{xz}` and S without R from the pushed points."""
        _, covariance_weights = self.weights(bank.layout.dimension)
        points = self.sigma_points(bank)
        predicted, dz = self._moments(model.predict(points), model.measurement_layout)
        dx = bank.layout.residual(points, bank.mean[:, None, :])
        cross = np.einsum("k,tki,tkj->tij", covariance_weights, dx, dz)
        spread = np.einsum("k,tki,tkj->tij", covariance_weights, dz, dz)
        return MeasurementMoments(predicted, cross, _symmetrise(spread))


def predict(
    bank: TrackBank, motion: MotionModel, linearisation: Linearisation, time_s: float
) -> TrackBank:
    """Return every row predicted to ``time_s``, each by its own step.

    Parameters
    ----------
    bank : TrackBank
        No row later than ``time_s``; the caller checks.
    motion : MotionModel
        In the bank's layout.
    linearisation : Linearisation
        How the moments go through it.
    time_s : float
        The new time.

    Returns
    -------
    TrackBank
        New means, covariances and times.
    """
    dt_s = time_s - bank.time_s
    mean, covariance = linearisation.predict(motion, bank, dt_s)
    return replace(bank, mean=mean, covariance=covariance, time_s=np.full_like(dt_s, time_s))


@dataclass(frozen=True, eq=False)
class Innovation:
    r"""Every row scored against every measurement of a scan.

    Attributes
    ----------
    innovation : numpy.ndarray
        :math:`\nu = z - \hat{z}`, short way round, ``(n_tracks, n_meas, m)``.
    innovation_covariance : numpy.ndarray
        :math:`S = H P^- H^T + R`, ``(n_tracks, n_meas, m, m)``: each measurement has its R.
    nis : numpy.ndarray
        :math:`\nu^T S^{-1} \nu`, ``(n_tracks, n_meas)``.
    log_likelihood : numpy.ndarray
        :math:`\ln \mathcal{N}(\nu; 0, S)`, ``(n_tracks, n_meas)``.
    moments : MeasurementMoments
        What the linearisation returned.
    measurement_noise : numpy.ndarray
        R of each measurement, ``(n_meas, m, m)``.
    """

    innovation: NDArray[np.float64]
    innovation_covariance: NDArray[np.float64]
    nis: NDArray[np.float64]
    log_likelihood: NDArray[np.float64]
    moments: MeasurementMoments
    measurement_noise: NDArray[np.float64]


def innovate(
    bank: TrackBank,
    model: MeasurementModel,
    linearisation: Linearisation,
    measurements: ArrayLike,
    noise_covariance: ArrayLike,
) -> Innovation:
    r"""Score every row against every measurement: innovation, S, NIS and log-likelihood.

    All pairs are factorised in one call, :math:`L L^T = S`. The whitened innovation
    :math:`w = L^{-1} \nu` gives :math:`d^2 = w^T w` and :math:`\ln|S| = 2 \sum \ln L_{ii}`,
    so :math:`S^{-1}` is never formed [1]_. Cost O(n_tracks n_meas m³), no loop over pairs.

    Parameters
    ----------
    bank : TrackBank
        The predicted tracks.
    model : MeasurementModel
        Explains the measurements, in the bank's layout.
    linearisation : Linearisation
        How the measurement moments are formed.
    measurements : array_like
        z, ``(n_meas, m)``.
    noise_covariance : array_like
        R, ``(n_meas, m, m)``, or one ``(m, m)`` shared by every measurement.

    Returns
    -------
    Innovation
        Everything gating, assignment and :func:`update` need.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.2.6 and §5.4.2; see the module References.
    """
    measurements = np.asarray(measurements, dtype=np.float64)
    n_meas, size = measurements.shape
    # A shared R becomes a read-only view per measurement, so every pair indexes its R alike.
    # broadcast-exempt: a view, not a copy; the stack shape is what update() indexes.
    noise = np.broadcast_to(np.asarray(noise_covariance, dtype=np.float64), (n_meas, size, size))
    moments = linearisation.measure(model, bank)
    innovation = model.measurement_layout.residual(
        measurements[None, :, :], moments.predicted_measurement[:, None, :]
    )
    covariance = _symmetrise(moments.predicted_measurement_covariance[:, None] + noise[None])
    root = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(root, innovation[..., None])[..., 0]
    nis = np.sum(whitened**2, axis=-1)
    log_determinant = 2 * np.sum(np.log(np.diagonal(root, axis1=-2, axis2=-1)), axis=-1)
    log_likelihood = -0.5 * (size * np.log(2 * np.pi) + log_determinant + nis)
    return Innovation(innovation, covariance, nis, log_likelihood, moments, noise)


def update(
    bank: TrackBank,
    innovation: Innovation,
    assigned: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> TrackBank:
    r"""Correct each row with the measurement assigned to it: the one correction core.

    :math:`K = P_{xz} S^{-1}`, by solving :math:`S K^T = P_{xz}^T`, never inverting S, and
    :math:`\hat{x} = \hat{x}^- + K \nu`, wrapped by the *state* layout [1]_. The covariance:

    - with H (KF, EKF), the Joseph form :math:`(I - K H) P^- (I - K H)^T + K R K^T`, positive
      semidefinite for any K, both terms being :math:`A M A^T`;
    - without H (UKF), :math:`P^- - K S K^T`, the Schur complement of S in a joint covariance
      that is positive semidefinite when every sigma-point covariance weight is.

    At the optimal gain they agree: with :math:`S = H P^- H^T + R` and :math:`P_{xz} = P^-
    H^T` the Joseph form is :math:`P^- - K P_{xz}^T - P_{xz} K^T + K S K^T`, and each of the
    last three terms is :math:`K S K^T`. They differ only in roundoff; the symmetric part is
    taken last.

    Parameters
    ----------
    bank : TrackBank
        The predicted tracks ``innovation`` came from.
    innovation : Innovation
        From :func:`innovate`.
    assigned : array_like of int
        The measurement each row takes, ``(n_tracks,)``, -1 for none.
    weights : array_like or None, optional
        Reserved for PDA and JPDA association weights, ``(n_tracks, n_meas + 1)``, the last
        column for "no measurement" (scenario 011 D28).

    Returns
    -------
    TrackBank
        The assigned rows corrected, the rest unchanged.

    Raises
    ------
    NotImplementedError
        If ``weights`` is given.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.2.4; Särkkä (2013), ch. 6.
    """
    if weights is not None:
        msg = "association weights (PDA, JPDA) arrive with scenario 011 D28."
        raise NotImplementedError(msg)
    choice = np.asarray(assigned, dtype=np.intp)
    rows = np.flatnonzero(choice >= 0)
    cols = choice[rows]
    covariance = innovation.innovation_covariance[rows, cols]
    cross = innovation.moments.cross_covariance[rows]
    gain = _transpose(np.linalg.solve(covariance, _transpose(cross)))
    prior = bank.covariance[rows]
    matrix = innovation.moments.measurement_matrix
    if matrix is None:
        posterior = prior - gain @ covariance @ _transpose(gain)
    else:
        spread = np.eye(bank.layout.dimension) - gain @ matrix[rows]
        noise = innovation.measurement_noise[cols]
        posterior = spread @ prior @ _transpose(spread) + gain @ noise @ _transpose(gain)
    mean, full_covariance = bank.mean.copy(), bank.covariance.copy()
    step = (gain @ innovation.innovation[rows, cols][..., None])[..., 0]
    mean[rows] = bank.layout.wrap(bank.mean[rows] + step)
    full_covariance[rows] = _symmetrise(posterior)
    return replace(bank, mean=mean, covariance=full_covariance)
