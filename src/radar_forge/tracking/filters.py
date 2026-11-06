r"""The Gaussian filter over a bank of tracks: the KF, the EKF and the UKF in one flow.

A track is a row of a :class:`TrackBank`, a struct of arrays. :func:`predict` moves every row;
:func:`innovate` scores every row against every measurement of a scan, ``(n_tracks, n_meas)``
pairs from one batched Cholesky factorisation and no loop over pairs; :func:`update` is the one
correction core, and :func:`update_weighted` the same core over a probabilistic association
(PDA, JPDA). The filters differ only in their :class:`Linearisation` [2]:
:class:`TaylorLinearisation` (KF on linear models, EKF otherwise) or :class:`UnscentedTransform`
(UKF). Nothing asks which one it holds; the core only asks whether H exists. An IMM bank also
carries each mode's estimate and probability, which :mod:`~radar_forge.tracking.imm` moves
and corrects. Validation happens once, in :meth:`TrackBank.append`.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
.. [2] S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013.
.. [3] E. A. Wan and R. van der Merwe, "The unscented Kalman filter for nonlinear estimation,"
       *Proc. IEEE Adaptive Systems for Signal Processing, Communications, and Control
       Symposium*, 2000, pp. 153-158.
.. [4] Y. Bar-Shalom, F. Daum and J. Huang, "The probabilistic data association filter,"
       *IEEE Control Syst. Mag.*, vol. 29, no. 6, pp. 82-100, 2009.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import IntEnum
from functools import cache
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.tracking.models import (
    MeasurementModel,
    MotionModel,
    StateLayout,
    check_covariances,
)

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
    "update_weighted",
]


class TrackStatus(IntEnum):
    """A track's place in its life, stored as int8; ``DELETED`` is reported once, then dropped."""

    TENTATIVE = 0
    CONFIRMED = 1
    COASTING = 2
    DELETED = 3


@dataclass(frozen=True, eq=False)
class TrackSnapshot:
    """A frozen copy of one bank row; ``nis`` and ``measurement_index`` are None without one."""

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


def _column(value: ArrayLike, n_rows: int, dtype: type[np.generic], name: str) -> NDArray[Any]:
    """Return one column of new rows, checked to have one entry per row."""
    column = np.array(value, dtype=dtype)
    if column.shape != (n_rows,):
        msg = f"{name} must have shape ({n_rows},); got {column.shape}."
        raise ValueError(msg)
    return column


@dataclass(frozen=True, eq=False)
class TrackBank:
    """Every track as one row: ``mean`` ``(n_tracks, n_state)``, ``covariance`` stacked.

    Treat it as immutable; every function returns a new bank. :meth:`empty` and
    :meth:`append` validate, the constructor trusts. The counts are int32 (a birth is one scan
    and one hit), ``status`` int8 :class:`TrackStatus`, ``nis`` NaN and ``measurement_index``
    -1 for a row with no measurement this scan. ``existence`` is the probability the target is
    real (1 unless an existence policy runs). An IMM bank's ``mean`` and ``covariance`` are the
    mixture's, with each mode's in ``mode_mean`` ``(n_tracks, n_modes, n_state)``,
    ``mode_covariance`` and ``mode_probability`` ``(n_tracks, n_modes)``; ``n_modes`` is 0
    otherwise. Every column has the track as its first axis, so a row is always one track.

    References
    ----------
    Bar-Shalom [1], §5.2.1 (the state estimate and its covariance).
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
    existence: NDArray[np.float64]
    mode_probability: NDArray[np.float64]
    mode_mean: NDArray[np.float64]
    mode_covariance: NDArray[np.float64]

    @classmethod
    def empty(cls, layout: StateLayout, n_modes: int = 0) -> TrackBank:
        """Return a bank with no rows, holding ``n_modes`` IMM modes (0: one model)."""
        n = layout.dimension
        columns = {name: np.zeros(0, dtype=dtype) for name, dtype in _DTYPES.items()}
        return cls(
            layout,
            np.zeros((0, n)),
            np.zeros((0, n, n)),
            **columns,
            mode_probability=np.zeros((0, n_modes)),
            mode_mean=np.zeros((0, n_modes, n)),
            mode_covariance=np.zeros((0, n_modes, n, n)),
        )

    @property
    def n_tracks(self) -> int:
        """Number of rows."""
        return len(self.track_id)

    @property
    def n_modes(self) -> int:
        """Number of IMM modes; 0 for a one-model bank."""
        return int(self.mode_probability.shape[1])

    def append(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
        time_s: float,
        track_id: ArrayLike,
        status: ArrayLike,
        measurement_index: ArrayLike | None = None,
        *,
        existence: ArrayLike | None = None,
        mode_probability: ArrayLike | None = None,
    ) -> TrackBank:
        """Return the bank with new rows ``(n_new, n_state)``, validated: the API boundary.

        Every IMM mode starts at the row's mean and covariance, weighted by
        ``mode_probability`` ``(n_modes,)`` or ``(n_new, n_modes)``; ``existence`` defaults
        to 1.
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
            "covariance": check_covariances(covariance, (n_new, n_state, n_state), "covariance"),
            "time_s": np.full(n_new, time_s, dtype=np.float64),
            "track_id": _column(track_id, n_new, np.int64, "track_id"),
            "status": _column(status, n_new, np.int8, "status"),
            "n_hits": np.ones(n_new, dtype=np.int32),
            "n_misses": np.zeros(n_new, dtype=np.int32),
            "n_scans": np.ones(n_new, dtype=np.int32),
            "last_hit_time_s": np.full(n_new, time_s, dtype=np.float64),
            "nis": np.full(n_new, np.nan),
            "measurement_index": _column(index, n_new, np.int64, "measurement_index"),
            "existence": _column(
                np.ones(n_new) if existence is None else existence, n_new, np.float64, "existence"
            ),
        }
        if not np.all(np.isin(new["status"], list(TrackStatus))):
            msg = f"status must hold TrackStatus values; got {new['status'].tolist()}."
            raise ValueError(msg)
        if not np.all((new["existence"] > 0) & (new["existence"] <= 1)):
            msg = f"existence must lie in (0, 1]; got {new['existence'].tolist()}."
            raise ValueError(msg)
        n_modes = self.n_modes
        weights = np.zeros((n_new, n_modes))
        if n_modes:
            weights = weights + np.asarray(mode_probability, dtype=np.float64)
            if not (np.all(weights >= 0) and np.allclose(weights.sum(axis=1), 1.0)):
                msg = f"mode_probability must be ({n_modes},) weights summing to 1."
                raise ValueError(msg)
        new["mode_probability"] = weights
        new["mode_mean"] = np.zeros((n_new, n_modes, n_state)) + new_mean[:, None]
        new["mode_covariance"] = (
            np.zeros((n_new, n_modes, n_state, n_state)) + new["covariance"][:, None]
        )
        columns = {name: np.concatenate([getattr(self, name), new[name]]) for name in new}
        return TrackBank(self.layout, **columns)

    def compact(self, keep: ArrayLike) -> TrackBank:
        """Return the rows where ``keep``, ``(n_tracks,)`` bool, is True, in order."""
        return self.take(np.flatnonzero(np.asarray(keep, dtype=bool)))

    def take(self, rows: ArrayLike) -> TrackBank:
        """Return the rows at the indices ``rows``, in that order; a row may repeat."""
        index = np.asarray(rows, dtype=np.intp)
        columns: dict[str, NDArray[Any]] = {
            f.name: getattr(self, f.name)[index] for f in fields(self)[1:]
        }
        return TrackBank(self.layout, **columns)

    def join(self, *others: TrackBank) -> TrackBank:
        """Return this bank's rows followed by each other bank's, all of one layout."""
        columns: dict[str, NDArray[Any]] = {
            f.name: np.concatenate([getattr(b, f.name) for b in (self, *others)])
            for f in fields(self)[1:]
        }
        return TrackBank(self.layout, **columns)

    def snapshots(self) -> tuple[TrackSnapshot, ...]:
        """Return a frozen, read-only :class:`TrackSnapshot` of each row, in row order."""
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
    "existence": np.float64,
}


@dataclass(frozen=True, eq=False)
class MeasurementMoments:
    r"""What a linearisation says each row should measure.

    ``predicted_measurement`` :math:`\hat{z}` is ``(n_tracks, m)``, ``cross_covariance``
    :math:`P_{xz}` ``(n_tracks, n_state, m)``, ``predicted_measurement_covariance``
    :math:`H P H^T` ``(n_tracks, m, m)``, S without R; ``measurement_matrix`` is H
    ``(n_tracks, m, n_state)``, or None for a linearisation without one.
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

    With F and H the Jacobians at the mean this is the textbook filter line for line:
    :math:`\hat{x}^- = F \hat{x}`, :math:`P^- = F P F^T + Q`, :math:`P_{xz} = P^- H^T`,
    :math:`S - R = H P^- H^T`. The mean goes through the motion model, :math:`f(\hat{x})`,
    and the covariance through its Jacobian F; on a linear model f(x̂) is F x̂ to the bit. It
    hands H to the correction core, which then takes the Joseph form.

    References
    ----------
    Bar-Shalom [1], §5.2.4 (KF) and §10.3 (EKF).
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
    r"""The unscented transform, the UKF's linearisation; sigma points ``(n_tracks, 2n+1, n)``.

    :math:`\chi_{\pm i} = \hat{x} \pm \sqrt{c} L_i`, :math:`c = \alpha^2 (n + \kappa)`, with
    weights :math:`W^{(m)}_0 = 1 - n/c`, :math:`W^{(c)}_0 = W^{(m)}_0 + 1 - \alpha^2 + \beta`,
    :math:`W_i = 1/(2c)`, cached per size. Without H the core uses :math:`P^- - K S K^T`, the
    Schur complement of S, positive semidefinite only when every covariance weight is; so
    :math:`W^{(c)}_0 < 0` is refused, including the small :math:`\alpha` of [3] §3. Means are
    offsets from the centre point through the layout, so an azimuth wrap averages the short
    way. On linear models the ± pairs cancel and the centre deviation is zero, so the moments
    are the KF's for any accepted weights.

    >>> UnscentedTransform().weights(2)
    (array([0.  , 0.25, 0.25, 0.25, 0.25]), array([2.  , 0.25, 0.25, 0.25, 0.25]))

    References
    ----------
    Wan and van der Merwe [3], §3; Särkkä [2], ch. 5.
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
        """Return the read-only mean and covariance weights, each ``(2 n_state + 1,)``."""
        return _unscented_weights(self.alpha, self.beta, self.kappa, n_state)

    def sigma_points(self, bank: TrackBank) -> NDArray[np.float64]:
        """Return every row's sigma points, ``(n_tracks, 2 n_state + 1, n_state)``."""
        n_state = bank.layout.dimension
        # The weights refuse n + kappa <= 0, which would put every sigma point on the mean.
        self.weights(n_state)
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
    """Return every row predicted to ``time_s``, each by its own step; none may be later."""
    dt_s = time_s - bank.time_s
    mean, covariance = linearisation.predict(motion, bank, dt_s)
    return replace(bank, mean=mean, covariance=covariance, time_s=np.full_like(dt_s, time_s))


@dataclass(frozen=True, eq=False)
class Innovation:
    r"""Every row scored against every measurement of a scan.

    ``innovation`` :math:`\nu` is ``(n_tracks, n_meas, m)``, short way round;
    ``innovation_covariance`` S ``(n_tracks, n_meas, m, m)``, since each measurement has its
    own R; ``nis``, the normalised innovation squared :math:`\nu^T S^{-1} \nu`, and
    ``log_likelihood`` ``(n_tracks, n_meas)``; ``measurement_noise`` R ``(n_meas, m, m)``.

    References
    ----------
    Bar-Shalom [1], §5.2.4 (the innovation and its covariance) and §5.2.6 (likelihood).
    """

    innovation: NDArray[np.float64]
    innovation_covariance: NDArray[np.float64]
    nis: NDArray[np.float64]
    log_likelihood: NDArray[np.float64]
    moments: MeasurementMoments
    measurement_noise: NDArray[np.float64]

    def take(self, rows: ArrayLike) -> Innovation:
        """Return the scoring of the bank rows at the indices ``rows``, in that order."""
        index = np.asarray(rows, dtype=np.intp)
        m = self.moments
        moments = MeasurementMoments(
            m.predicted_measurement[index],
            m.cross_covariance[index],
            m.predicted_measurement_covariance[index],
            None if m.measurement_matrix is None else m.measurement_matrix[index],
        )
        return Innovation(
            self.innovation[index],
            self.innovation_covariance[index],
            self.nis[index],
            self.log_likelihood[index],
            moments,
            self.measurement_noise,
        )


def innovate(
    bank: TrackBank,
    model: MeasurementModel,
    linearisation: Linearisation,
    measurements: ArrayLike,
    noise_covariance: ArrayLike,
) -> Innovation:
    r"""Score every row against every measurement ``(n_meas, m)``, all pairs at once.

    One Cholesky factor per pair, :math:`L L^T = S`; the whitened :math:`w = L^{-1} \nu` gives
    :math:`d^2 = w^T w` and :math:`\ln|S| = 2 \sum \ln L_{ii}`, so :math:`S^{-1}` is never
    formed. ``noise_covariance`` is ``(n_meas, m, m)``, or one ``(m, m)`` for all. Cost
    O(n_tracks n_meas m³).

    References
    ----------
    Bar-Shalom [1], §5.2.6 (likelihood) and §5.4.2 (NIS).
    """
    measurements = np.asarray(measurements, dtype=np.float64)
    n_meas, size = measurements.shape
    # A shared R becomes a read-only view per measurement, so every pair indexes its R alike.
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
) -> TrackBank:
    r"""Correct each row with its measurement ``assigned`` ``(n_tracks,)``, -1 for none.

    :math:`K = P_{xz} S^{-1}` by solving, never inverting; the mean wraps by the *state*
    layout. With H (KF, EKF) the covariance is the Joseph form :math:`(I - K H) P^- (I -
    K H)^T + K R K^T`, positive semidefinite for any K; without H (UKF) it is :math:`P^- - K S
    K^T`. At the optimal gain the two are equal and differ only in roundoff.

    References
    ----------
    Bar-Shalom [1], §5.2.4; Särkkä [2], ch. 6.
    """
    choice = np.asarray(assigned, dtype=np.intp)
    rows = np.flatnonzero(choice >= 0)
    step, posterior = _correct(bank, innovation, rows, choice[rows])
    mean, full_covariance = bank.mean.copy(), bank.covariance.copy()
    mean[rows] = bank.layout.wrap(bank.mean[rows] + step)
    full_covariance[rows] = _symmetrise(posterior)
    return replace(bank, mean=mean, covariance=full_covariance)


def _correct(
    bank: TrackBank, innovation: Innovation, rows: NDArray[np.intp], cols: NDArray[np.intp]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the Kalman step (gain times innovation) and posterior covariance of each pair."""
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
    step = (gain @ innovation.innovation[rows, cols][..., None])[..., 0]
    return step, posterior


def update_weighted(bank: TrackBank, innovation: Innovation, weights: ArrayLike) -> TrackBank:
    r"""Correct each row with every measurement at once, by association weights.

    ``weights`` is ``(n_tracks, n_meas + 1)``, rows summing to one, the last column the
    probability that none is the target's. Each weighted pair is corrected by the core of
    :func:`update`, and the row becomes the moments of that mixture: :math:`\hat{x} = \hat{x}^-
    + \sum_j \beta_j K_j \nu_j`, :math:`P = \beta_0 P^- + \sum_j \beta_j (P_j + s_j s_j^T) -
    \bar{s} \bar{s}^T` with :math:`s_j = K_j \nu_j`. With one R for every measurement this is
    the PDA update, :math:`\beta_0 P^- + (1 - \beta_0) P^c + K (\sum_j \beta_j \nu_j \nu_j^T -
    \bar{\nu} \bar{\nu}^T) K^T`; with a one-hot row, the Kalman update.

    References
    ----------
    Bar-Shalom, Daum and Huang [4], the PDAF state and covariance update.
    """
    beta = np.asarray(weights, dtype=np.float64)
    rows, cols = np.nonzero(beta[:, :-1] > 0)
    step, posterior = _correct(bank, innovation, rows, cols)
    weight = beta[rows, cols]
    shift = np.zeros_like(bank.mean)
    np.add.at(shift, rows, weight[:, None] * step)
    outer = posterior + step[:, :, None] * step[:, None, :]
    covariance = beta[:, -1, None, None] * bank.covariance
    np.add.at(covariance, rows, weight[:, None, None] * outer)
    covariance -= shift[:, :, None] * shift[:, None, :]
    mean = bank.layout.wrap(bank.mean + shift)
    return replace(bank, mean=mean, covariance=_symmetrise(covariance))
