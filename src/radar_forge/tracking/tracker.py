r"""The scan loop over a bank of tracks: its lifecycle rules, initiators and :class:`Tracker`.

:meth:`Tracker.process` takes one :class:`Scan` and runs, over every track and measurement at
once: predict; drop rows past ``max_coast_time_s``; innovate and gate through each model the
selector offers, a pair keeping the first it passes; assign, confirmed tracks first (the
default) or in one ``"joint"`` solve, which scenario 003 measured better (tracker-001 §5.1);
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
.. [3] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking Systems*,
       Artech House, 1999.
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
    innovate,
    predict,
    update,
)
from radar_forge.tracking.initiation import Initiator
from radar_forge.tracking.models import (
    MarginalModel,
    MeasurementModel,
    MotionModel,
    check_covariances,
)

__all__ = [
    "LifecyclePolicy",
    "ModelSelector",
    "Scan",
    "Tracker",
]

# Two times this close are one time: one scan time computed two ways differs by roundoff far
# under a microsecond, while a target at 300 m/s moves 0.3 mm in one.
_TIME_TOLERANCE_S = 1e-6

_CONFIRMED = (TrackStatus.CONFIRMED, TrackStatus.COASTING)

Rows = tuple[NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class LifecyclePolicy:
    """When a track is confirmed (M hits in its first N scans) or deleted.

    ``n_reacquire_misses`` extra misses in a row are allowed a confirmed track only.
    ``reacquire_reset`` names coordinates whose variances return to the prior, and
    cross-covariances to zero, on the miss that opens re-acquisition (tracker-001 §5.4).

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
    ``(n_meas, m, m)`` stack, each positive definite. ``(0, m)`` measurements is a scan that
    saw nothing.

    References
    ----------
    Bar-Shalom [2], §4.3.1 (z = h(x) + w, w zero-mean with covariance R).
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
            shared = check_covariances(noise, (size, size), "noise_covariance")
            noise = np.zeros((n_meas, size, size)) + shared
        noise = check_covariances(noise, (n_meas, size, size), "noise_covariance")
        measurements.setflags(write=False)
        noise.setflags(write=False)
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "noise_covariance", noise)


class ModelSelector(Protocol):
    """The models each track may use a scan through, best first (tracker-001 §5.4).

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

    References
    ----------
    Bar-Shalom and Li [1], §2.6.1, and Blackman and Popoli [3] (M-of-N and assignment).
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
        extra = np.where(np.isin(status, _CONFIRMED), policy.n_reacquire_misses, 0)
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
