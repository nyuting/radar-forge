r"""The scan loop over a bank of tracks: its lifecycle rules, initiators and :class:`Tracker`.

:meth:`Tracker.process` takes one :class:`Scan` and runs, over every track and measurement at
once: predict; drop rows past ``max_coast_time_s``; innovate and gate through each model the
selector offers, a pair keeping the first it passes; assign, confirmed tracks first (the
default) or in one ``"joint"`` solve, which scenario 003 measured better (tracker-001 §7.3);
update, once per model offered; count hits and misses as integer arithmetic, confirming by
M-of-N [1] and, on the miss that opens re-acquisition, resetting ``reacquire_reset``; start a
row from each unclaimed measurement (in two-stage assignment, none inside a confirmed gate);
report deleted rows once and drop them. The bank is immutable and replaced only when a scan
completes, so an error leaves the tracker as it was. At M = 3, N = 5: hit, miss, hit, miss, hit
confirms on scan 5; hit, miss, miss, miss deletes on scan 4.

References
----------
.. [1] Y. Bar-Shalom and X. R. Li, *Multitarget-Multisensor Tracking: Principles and
       Techniques*, YBS Publishing, 1995.
.. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
.. [3] S. Bordonaro, P. Willett and Y. Bar-Shalom, "Decorrelated unbiased converted
       measurement Kalman filter," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 50, no. 2,
       pp. 1431-1444, 2014, as restated by W. Wang et al., arXiv:1412.5524, §III-B.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.tracking.association import (
    Associator,
    Cost,
    association_cost,
    chi_square_gate_threshold,
    gate,
)
from radar_forge.tracking.filters import (
    Innovation,
    Linearisation,
    TrackBank,
    TrackStatus,
    _check_covariances,
    innovate,
    predict,
    update,
)
from radar_forge.tracking.models import (
    LinearMeasurement,
    MarginalModel,
    MeasurementModel,
    MotionModel,
    RangeAzimuthElevationRangeRate,
)

__all__ = [
    "DebiasedSphericalInitiator",
    "DirectStateInitiator",
    "Initiator",
    "LifecyclePolicy",
    "ModelSelector",
    "Scan",
    "Tracker",
    "debiased_spherical_to_enu",
]

# Two times this close are one time: one scan time computed two ways differs by roundoff far
# under a microsecond, while a target at 300 m/s moves 0.3 mm in one.
_TIME_TOLERANCE_S = 1e-6

_CONFIRMED = (TrackStatus.CONFIRMED, TrackStatus.COASTING)

Rows = tuple[NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class LifecyclePolicy:
    """When a track is confirmed (M hits in its first N scans) or deleted.

    ``n_reacquire_scans`` extra misses are allowed a confirmed track only. ``reacquire_reset``
    names coordinates whose variances return to the prior, and cross-covariances to zero, on
    the miss that opens re-acquisition (tracker-001 §8.3).

    References
    ----------
    Bar-Shalom and Li [1], §2.6.1.
    """

    n_confirm_hits: int = 3
    n_confirm_scans: int = 5
    n_delete_misses: int = 5
    n_reacquire_scans: int = 0
    max_coast_time_s: float = 30.0
    reacquire_reset: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a policy that could not work."""
        counts = (self.n_confirm_hits, self.n_confirm_scans, self.n_delete_misses)
        # A bool is an int in Python, so the exact type is tested.
        if any(type(n) is not int for n in (*counts, self.n_reacquire_scans)) or not (
            1 <= self.n_confirm_hits <= self.n_confirm_scans
            and self.n_delete_misses >= 1
            and self.n_reacquire_scans >= 0
        ):
            msg = f"need int counts, 1 <= M <= N, n_delete_misses >= 1; got {counts}."
            raise ValueError(msg)
        names = self.reacquire_reset
        if not (math.isfinite(self.max_coast_time_s) and self.max_coast_time_s > 0) or not (
            isinstance(names, tuple) and len(set(names)) == len(names) and all(names)
        ):
            msg = f"need max_coast_time_s > 0 and unique reset names; got {names!r}."
            raise ValueError(msg)


@dataclass(frozen=True, eq=False)
class Scan:
    """Everything one sensor measured at one time: z ``(n_meas, m)`` and R, checked and frozen.

    ``noise_covariance`` may be one ``(m, m)`` for every measurement; it is stored as the
    ``(n_meas, m, m)`` stack. ``(0, m)`` measurements is a scan that saw nothing.
    """

    time_s: float
    measurements: NDArray[np.float64]
    noise_covariance: NDArray[np.float64]
    model_id: str

    def __post_init__(self) -> None:
        """Check, copy and freeze the measurements and their covariances."""
        measurements = np.array(self.measurements, dtype=np.float64)
        if not (math.isfinite(self.time_s) and self.model_id) or measurements.ndim != 2:
            msg = f"need a finite time, a model ID and z of shape (n_meas, m); got {self.time_s}."
            raise ValueError(msg)
        n_meas, size = measurements.shape
        if size < 1 or not np.all(np.isfinite(measurements)):
            msg = f"measurements must be finite with m >= 1; got shape {measurements.shape}."
            raise ValueError(msg)
        noise = np.asarray(self.noise_covariance, dtype=np.float64)
        if noise.shape == (size, size):
            noise = np.zeros((n_meas, size, size)) + noise
        noise = _check_covariances(noise, (n_meas, size, size), "noise_covariance")
        measurements.setflags(write=False)
        noise.setflags(write=False)
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "noise_covariance", noise)


class Initiator(Protocol):
    """Starts tracks from a prior ``(n_state,)``; the re-acquisition reset reads it too."""

    prior_mean: NDArray[np.float64]
    prior_covariance: NDArray[np.float64]

    def initiate(
        self,
        measurements: NDArray[np.float64],
        noise_covariance: NDArray[np.float64],
        model: MeasurementModel,
    ) -> Rows | None:
        """Return a mean and covariance per measurement, or None if ``model`` cannot seed."""
        ...


class _PriorInitiator:
    """A prior, checked once, and the rule for writing measured coordinates into it."""

    def __init__(self, prior_mean: ArrayLike, prior_covariance: ArrayLike) -> None:
        mean = np.array(prior_mean, dtype=np.float64)
        if mean.ndim != 1 or not np.all(np.isfinite(mean)):
            msg = f"prior_mean must be a finite vector; got shape {mean.shape}."
            raise ValueError(msg)
        self.prior_mean = mean
        shape = (len(mean), len(mean))
        self.prior_covariance = _check_covariances(prior_covariance, shape, "prior_covariance")

    def _seed(
        self, rows: Sequence[int], values: NDArray[np.float64], blocks: NDArray[np.float64]
    ) -> Rows:
        """Copy the prior per measurement, and replace the measured rows and their block."""
        index = np.asarray(rows, dtype=np.intp)
        # Replacing a block the prior ties to other coordinates would leave cross terms that
        # no longer fit, and the covariance could stop being positive semidefinite.
        if np.any(np.delete(self.prior_covariance[index], index, axis=1) != 0):
            msg = f"the prior ties the measured coordinates {index.tolist()} to the others."
            raise ValueError(msg)
        mean = np.zeros((len(values), len(self.prior_mean))) + self.prior_mean
        covariance = np.zeros((len(values), *self.prior_covariance.shape)) + self.prior_covariance
        mean[:, index] = values
        covariance[:, index[:, None], index[None, :]] = blocks
        return mean, covariance


class DirectStateInitiator(_PriorInitiator):
    """Start a track by writing a linear measurement's coordinates and their R into a prior.

    Unmeasured coordinates keep the prior, whose measured block must be uncorrelated with the
    rest. ``seed_names`` copies only those, such as range while a folded rate is untrusted.
    A model that is not a :class:`~radar_forge.tracking.models.LinearMeasurement` starts none.

    References
    ----------
    Bar-Shalom [2], §5.5.
    """

    def __init__(
        self,
        prior_mean: ArrayLike,
        prior_covariance: ArrayLike,
        *,
        seed_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(prior_mean, prior_covariance)
        self.seed_names = None if seed_names is None else tuple(seed_names)

    def initiate(
        self,
        measurements: NDArray[np.float64],
        noise_covariance: NDArray[np.float64],
        model: MeasurementModel,
    ) -> Rows | None:
        """Return the seeded rows, or None for a model that is not linear."""
        if not isinstance(model, LinearMeasurement):
            return None
        names = self.seed_names or model.measurement_layout.names
        rows = np.asarray(model.measurement_layout.indices(names))
        blocks = noise_covariance[:, rows[:, None], rows[None, :]]
        return self._seed(model.state_layout.indices(names), measurements[:, rows], blocks)


def _angle_moments(
    angle_rad: NDArray[np.float64], sigma_rad: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    """Return E[cos a], E[sin a], E[cos² a], E[sin² a], E[sin a cos a] for a Gaussian angle.

    E[cos a] = λ cos ā with λ = exp(-sigma²/2); the double angle's damping is λ⁴.
    """
    damping, double_damping = np.exp(-(sigma_rad**2) / 2), np.exp(-2 * sigma_rad**2)
    cos_2a = np.cos(2 * angle_rad)
    return (
        damping * np.cos(angle_rad),
        damping * np.sin(angle_rad),
        (1 + double_damping * cos_2a) / 2,
        (1 - double_damping * cos_2a) / 2,
        double_damping * np.sin(2 * angle_rad) / 2,
    )


def debiased_spherical_to_enu(
    range_m: ArrayLike,
    azimuth_rad: ArrayLike,
    elevation_rad: ArrayLike,
    sigma_range_m: ArrayLike,
    sigma_azimuth_rad: ArrayLike,
    sigma_elevation_rad: ArrayLike = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Return the ENU offset ``(..., 3)`` and covariance of the true position, without bias.

    The naive conversion is biased towards the radar: the mean of cos(θ + w) is λ cos θ with
    λ = exp(-sigma²/2). Given z, the true range and angles are Gaussians about the measured
    ones, so with p = r d, :math:`E[p] = r_m E[d]` and :math:`\operatorname{cov} p = (r_m^2 +
    \sigma_r^2) E[d d^T] - r_m^2 E[d] E[d]^T`: exact for Gaussian errors, and the posterior
    under a flat prior. Azimuth is clockwise from north; elevation 0 for a 2-D radar.

    >>> position_m, _ = debiased_spherical_to_enu(20_000.0, 0.0, 0.0, 10.0, 0.1)
    >>> round(float(position_m[1]), 1)
    19900.2

    References
    ----------
    Bordonaro, Willett and Bar-Shalom [3], restated as eqs. (8)-(14) of arXiv:1412.5524.
    """
    given = (range_m, azimuth_rad, elevation_rad, sigma_range_m, sigma_azimuth_rad)
    values = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in given),
        np.asarray(sigma_elevation_rad, dtype=np.float64),
    )
    radius_m, azimuth, elevation, sigma_r, sigma_az, sigma_el = values
    if not np.all(np.isfinite(values)) or np.any(radius_m <= 0) or np.any(np.stack(values[3:]) < 0):
        msg = "need finite values, positive ranges and nonnegative standard deviations."
        raise ValueError(msg)
    cos_az, sin_az, cos2_az, sin2_az, sincos_az = _angle_moments(azimuth, sigma_az)
    cos_el, sin_el, cos2_el, sin2_el, sincos_el = _angle_moments(elevation, sigma_el)
    direction = np.stack([cos_el * sin_az, cos_el * cos_az, sin_el], axis=-1)
    # E[d dᵀ]: azimuth and elevation are independent, so each entry factorises.
    rows = [
        [cos2_el * sin2_az, cos2_el * sincos_az, sincos_el * sin_az],
        [cos2_el * sincos_az, cos2_el * cos2_az, sincos_el * cos_az],
        [sincos_el * sin_az, sincos_el * cos_az, sin2_el],
    ]
    second_moment = np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)
    position_m = radius_m[..., None] * direction
    outer = position_m[..., :, None] * position_m[..., None, :]
    covariance_m2 = (radius_m**2 + sigma_r**2)[..., None, None] * second_moment - outer
    return position_m, (covariance_m2 + np.swapaxes(covariance_m2, -1, -2)) / 2


class DebiasedSphericalInitiator(_PriorInitiator):
    """Start a track at a spherical measurement's debiased position; the rest keeps the prior.

    Needs a :class:`~radar_forge.tracking.models.RangeAzimuthElevationRangeRate` model that
    reports range and azimuth, with independent errors; the height keeps the prior when no
    elevation is reported. Every later update uses the spherical model itself.

    References
    ----------
    Bordonaro, Willett and Bar-Shalom [3].
    """

    def initiate(
        self,
        measurements: NDArray[np.float64],
        noise_covariance: NDArray[np.float64],
        model: MeasurementModel,
    ) -> Rows | None:
        """Return the converted rows, or None for a model without range and azimuth."""
        if not isinstance(model, RangeAzimuthElevationRangeRate) or not (
            {"range_m", "azimuth_rad"} <= set(model.components)
        ):
            return None
        names = [n for n in ("range_m", "azimuth_rad", "elevation_rad") if n in model.components]
        rows = np.asarray(model.measurement_layout.indices(names))
        block = noise_covariance[:, rows[:, None], rows[None, :]]
        if np.any(block != block * np.eye(len(rows))):
            msg = f"the debiased conversion needs independent errors in {names}."
            raise ValueError(msg)
        value, sigma = measurements[:, rows], np.sqrt(np.diagonal(block, axis1=1, axis2=2))
        elevation = len(rows) == 3
        position_m, covariance_m2 = debiased_spherical_to_enu(
            value[:, 0],
            value[:, 1],
            value[:, 2] if elevation else 0.0,
            sigma[:, 0],
            sigma[:, 1],
            sigma[:, 2] if elevation else 0.0,
        )
        layout = model.state_layout
        axes = [a for a in "xyz" if f"{a}_m" in layout.names and (a != "z" or elevation)]
        enu = np.array(["xyz".index(a) for a in axes])
        return self._seed(
            layout.indices([f"{a}_m" for a in axes]),
            (position_m + model.sensor_enu_m)[:, enu],
            covariance_m2[:, enu[:, None], enu[None, :]],
        )


class ModelSelector(Protocol):
    """The models each track may use a scan through, best first (tracker-001 §8.3).

    Called once per scan; each model offered is the scan's own or a registered marginal of it.
    Scenario 003's offers range and rate to tracks that can unfold, then range alone to all.
    """

    def __call__(self, bank: TrackBank, model_id: str) -> Sequence[tuple[str, NDArray[np.bool_]]]:
        """Return ``(model_id, allowed)`` pairs, ``allowed`` of shape ``(n_tracks,)``."""
        ...


def _m_of_n(
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


def _births(n_new: int, policy: LifecyclePolicy) -> NDArray[np.int8]:
    """Return new rows' status: tentative, or confirmed at once under 1-of-N."""
    ones = np.ones(n_new, dtype=np.int32)
    return _m_of_n(np.zeros(n_new, dtype=np.int8), ones, ones, policy)


class Tracker:
    """Track many targets from scans, every track a row of one bank.

    Parameters
    ----------
    models : mapping of str to MeasurementModel
        Every model by ID, marginals included, all in the motion model's layout.
    linearisation : Linearisation
        ``TaylorLinearisation()`` (KF, EKF) or ``UnscentedTransform()`` (UKF).
    cost : {"nis", "negative_log_likelihood_ratio"}, default "nis"
        NIS cannot compare pairs of different dimensions, so a scan that mixes them raises.
    clutter_density : mapping of str to float or None, optional
        λ per model ID: required for every model by the likelihood ratio, refused otherwise.
    model_selector : ModelSelector or None, optional
        None offers the scan's own model to every track.
    """

    def __init__(
        self,
        models: Mapping[str, MeasurementModel],
        motion: MotionModel,
        linearisation: Linearisation,
        gate_probability: float,
        associator: Associator,
        initiator: Initiator,
        policy: LifecyclePolicy | None = None,
        cost: Cost = "nis",
        clutter_density: Mapping[str, float] | None = None,
        *,
        assignment: Literal["confirmed_first", "joint"] = "confirmed_first",
        model_selector: ModelSelector | None = None,
    ) -> None:
        layout = motion.state_layout
        densities = dict(clutter_density or {})
        needed = set(models) if cost == "negative_log_likelihood_ratio" else set()
        if not models or any(m.state_layout != layout for m in models.values()):
            msg = "every measurement model must share the motion model's state layout."
            raise ValueError(msg)
        if cost not in ("nis", "negative_log_likelihood_ratio") or assignment not in (
            "confirmed_first",
            "joint",
        ):
            msg = f"unknown cost {cost!r} or assignment {assignment!r}."
            raise ValueError(msg)
        if set(densities) != needed or not all(
            np.isfinite(d) and d > 0 for d in densities.values()
        ):
            msg = f"clutter_density needs one positive value for each of {sorted(needed)}."
            raise ValueError(msg)
        if initiator.prior_mean.shape != (layout.dimension,):
            msg = f"the initiator's prior must have {layout.dimension} coordinates."
            raise ValueError(msg)
        chi_square_gate_threshold(gate_probability, 1)
        self.policy = policy or LifecyclePolicy()
        self._reset = np.asarray(layout.indices(self.policy.reacquire_reset), dtype=np.intp)
        self.models = MappingProxyType(dict(models))
        self.clutter_density = MappingProxyType(densities)
        self.motion = motion
        self.linearisation = linearisation
        self.gate_probability = gate_probability
        self.associator = associator
        self.initiator = initiator
        self.cost = cost
        self.assignment = assignment
        self.model_selector = model_selector
        self.bank = TrackBank.empty(layout)
        self.time_s: float | None = None
        self._next_track_id = 1

    def seed(self, mean: ArrayLike, covariance: ArrayLike, time_s: float) -> NDArray[np.int64]:
        """Add tracks ``(n_new, n_state)`` from a cue and return their IDs; bistatic needs it."""
        if self.time_s is not None and time_s < self.time_s - _TIME_TOLERANCE_S:
            msg = f"cannot seed at {time_s} s, before the last scan at {self.time_s} s."
            raise ValueError(msg)
        n_new = len(np.asarray(mean))
        ids = np.arange(self._next_track_id, self._next_track_id + n_new, dtype=np.int64)
        self.bank = self.bank.append(mean, covariance, time_s, ids, _births(n_new, self.policy))
        self._next_track_id += n_new
        return ids

    def process(self, scan: Scan) -> TrackBank:
        """Run one scan; return survivors then births, this scan's deleted rows included once."""
        model = self._check(scan)
        time_s = max(scan.time_s, float(self.bank.time_s.max(initial=scan.time_s)))
        bank = predict(self.bank, self.motion, self.linearisation, time_s)
        no_index = np.full(bank.n_tracks, -1, dtype=np.int64)
        bank = replace(bank, nis=np.full(bank.n_tracks, np.nan), measurement_index=no_index)
        live = time_s - bank.last_hit_time_s <= self.policy.max_coast_time_s
        cost, chosen, innovations = self._score(bank, scan, live)
        assigned, blocked = self._assign(cost, chosen >= 0, bank.status)
        bank = self._update(bank, innovations, chosen, assigned)
        bank = self._count(bank, assigned >= 0, live, time_s)
        bank, next_track_id = self._start(bank, scan, model, assigned, blocked, time_s)
        self.bank = bank.compact(bank.status != TrackStatus.DELETED)
        self.time_s, self._next_track_id = time_s, next_track_id
        return bank

    def _check(self, scan: Scan) -> MeasurementModel:
        """Return the scan's model, or raise before anything changes."""
        model = self.models.get(scan.model_id)
        if model is None or scan.measurements.shape[1] != model.measurement_layout.dimension:
            msg = f"scan model {scan.model_id!r} is unknown, or its dimension does not match."
            raise ValueError(msg)
        last_scan_s = -math.inf if self.time_s is None else self.time_s
        latest_s = max(last_scan_s, float(self.bank.time_s.max(initial=-math.inf)))
        if scan.time_s < latest_s - _TIME_TOLERANCE_S:
            msg = f"out-of-sequence scan at {scan.time_s} s, after {latest_s} s."
            raise ValueError(msg)
        return model

    def _view(self, scan: Scan, model_id: str) -> Rows:
        """Return the scan as ``model_id`` sees it: itself, or a marginal's components."""
        if model_id == scan.model_id:
            return scan.measurements, scan.noise_covariance
        model = self.models.get(model_id)
        if not (isinstance(model, MarginalModel) and model.base is self.models[scan.model_id]):
            msg = f"the selector offered {model_id!r}: not {scan.model_id!r} or a marginal of it."
            raise ValueError(msg)
        rows = np.asarray(model.indices)
        return scan.measurements[:, rows], scan.noise_covariance[:, rows[:, None], rows[None, :]]

    def _score(
        self, bank: TrackBank, scan: Scan, live: NDArray[np.bool_]
    ) -> tuple[NDArray[np.float64], NDArray[np.intp], list[Innovation]]:
        """Gate every pair; return costs, each pair's model index (-1 none) and innovations."""
        everyone = [(scan.model_id, np.ones(bank.n_tracks, dtype=bool))]
        selector = self.model_selector
        offered = everyone if selector is None else selector(bank, scan.model_id)
        cost = np.full((bank.n_tracks, len(scan.measurements)), np.inf)
        chosen = np.full(cost.shape, -1, dtype=np.intp)
        innovations: list[Innovation] = []
        # A loop over the few models offered; each is batched over every pair.
        for k, (model_id, allowed) in enumerate(offered):
            measurements, noise = self._view(scan, model_id)
            innovation = innovate(
                bank, self.models[model_id], self.linearisation, measurements, noise
            )
            threshold = chi_square_gate_threshold(self.gate_probability, measurements.shape[1])
            passed = gate(innovation.nis, threshold) & (chosen < 0) & (live & allowed)[:, None]
            pair_cost = association_cost(innovation, self.cost, self.clutter_density.get(model_id))
            cost = np.where(passed, pair_cost, cost)
            chosen[passed] = k
            innovations.append(innovation)
        sizes = {innovations[k].innovation.shape[-1] for k in np.unique(chosen[chosen >= 0])}
        if self.cost == "nis" and len(sizes) > 1:
            msg = f"NIS cannot compare dimensions {sorted(sizes)}; use the likelihood ratio."
            raise ValueError(msg)
        return cost, chosen, innovations

    def _assign(
        self, cost: NDArray[np.float64], in_gate: NDArray[np.bool_], status: NDArray[np.int8]
    ) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
        """Return each row's measurement (-1 none), and the measurements no birth may use."""
        if self.assignment == "joint":
            return self.associator.assign(cost, in_gate)[0], np.zeros(cost.shape[1], dtype=bool)
        first = np.isin(status, _CONFIRMED)
        assigned = np.full(len(cost), -1, dtype=np.intp)
        assigned[first] = self.associator.assign(cost[first], in_gate[first])[0]
        free = np.ones(cost.shape[1], dtype=bool)
        free[assigned[assigned >= 0]] = False
        second = self.associator.assign(cost[~first][:, free], in_gate[~first][:, free])[0]
        assigned[np.flatnonzero(~first)[second >= 0]] = np.flatnonzero(free)[second[second >= 0]]
        return assigned, np.asarray(in_gate[first].any(axis=0), dtype=bool)

    def _update(
        self,
        bank: TrackBank,
        innovations: list[Innovation],
        chosen: NDArray[np.intp],
        assigned: NDArray[np.intp],
    ) -> TrackBank:
        """Correct the assigned rows through the model each passed, and record the NIS."""
        rows = np.flatnonzero(assigned >= 0)
        through = np.full(bank.n_tracks, -1, dtype=np.intp)
        through[rows] = chosen[rows, assigned[rows]]
        nis, index = bank.nis.copy(), bank.measurement_index.copy()
        # One batched correction per model offered: a few, each over its own rows.
        for k, innovation in enumerate(innovations):
            mine = np.flatnonzero(through == k)
            choice = np.full(bank.n_tracks, -1, dtype=np.intp)
            choice[mine] = assigned[mine]
            bank = update(bank, innovation, choice)
            nis[mine] = innovation.nis[mine, assigned[mine]]
        index[rows] = assigned[rows]
        return replace(bank, nis=nis, measurement_index=index)

    def _count(
        self, bank: TrackBank, hit: NDArray[np.bool_], live: NDArray[np.bool_], time_s: float
    ) -> TrackBank:
        """Count hits and misses, move statuses, and reset the rows opening re-acquisition."""
        policy = self.policy
        n_scans = (bank.n_scans + live).astype(np.int32)
        n_hits = (bank.n_hits + hit).astype(np.int32)
        n_misses = np.where(hit, 0, bank.n_misses + live).astype(np.int32)
        status = bank.status.copy()
        status[hit & (status == TrackStatus.COASTING)] = TrackStatus.CONFIRMED
        status[live & ~hit & (status == TrackStatus.CONFIRMED)] = TrackStatus.COASTING
        status = _m_of_n(status, n_hits, n_scans, policy)
        extra = np.where(np.isin(status, _CONFIRMED), policy.n_reacquire_scans, 0)
        status[~live | (n_misses >= policy.n_delete_misses + extra)] = TrackStatus.DELETED
        # The miss that opens re-acquisition, and only it: a later reset would undo what the
        # coast has learnt.
        reset = (status == TrackStatus.COASTING) & (n_misses == policy.n_delete_misses)
        covariance, kept = bank.covariance.copy(), self._reset
        block = covariance[reset]
        block[:, kept, :] = 0.0
        block[:, :, kept] = 0.0
        block[:, kept[:, None], kept[None, :]] = self.initiator.prior_covariance[np.ix_(kept, kept)]
        covariance[reset] = block
        return replace(
            bank,
            covariance=covariance,
            status=status,
            n_hits=n_hits,
            n_misses=n_misses,
            n_scans=n_scans,
            last_hit_time_s=np.where(hit, time_s, bank.last_hit_time_s),
        )

    def _start(
        self,
        bank: TrackBank,
        scan: Scan,
        model: MeasurementModel,
        assigned: NDArray[np.intp],
        blocked: NDArray[np.bool_],
        time_s: float,
    ) -> tuple[TrackBank, int]:
        """Append a tentative row per measurement no track took; return the next free ID."""
        claimed = np.zeros(len(scan.measurements), dtype=bool)
        claimed[assigned[assigned >= 0]] = True
        free = np.flatnonzero(~claimed & ~blocked)
        rows = self.initiator.initiate(scan.measurements[free], scan.noise_covariance[free], model)
        if rows is None or not free.size:
            return bank, self._next_track_id
        ids = np.arange(self._next_track_id, self._next_track_id + free.size, dtype=np.int64)
        status = _births(free.size, self.policy)
        return bank.append(*rows, time_s, ids, status, free), int(ids[-1]) + 1
