r"""The scan loop over a bank of tracks: its lifecycle rules and :class:`Tracker`.

:meth:`Tracker.process` takes one :class:`Scan` and runs, over every track the scan could see
(``coverage``; default all) and every measurement at once: predict, by one model or an IMM;
drop rows past ``max_coast_time_s``; innovate and gate through each model the selector offers,
a pair keeping the first it passes; associate, confirmed tracks first (the default) or jointly
(tracker-001 §5.1), by assignment or, for an associator that weights, by JPDA over confirmed
tracks and PDA over tentative ones; update; count hits and misses, confirming by M-of-N [1] or
by existence (JIPDA [4]) and, on the miss that opens re-acquisition, resetting
``reacquire_reset``; start a row from each measurement no track took or gated; report deleted
rows once and drop them. The bank is replaced only when a scan completes, so an error leaves
the tracker as it was. At M = 3, N = 5: hit, miss, hit, miss, hit confirms on scan 5; hit,
miss, miss, miss deletes on scan 4.

References
----------
.. [1] Y. Bar-Shalom and X. R. Li, *Multitarget-Multisensor Tracking: Principles and
       Techniques*, YBS Publishing, 1995.
.. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
.. [3] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking Systems*,
       Artech House, 1999.
.. [4] D. Mušicki and R. Evans, "Joint integrated probabilistic data association: JIPDA,"
       *IEEE Trans. Aerosp. Electron. Syst.*, vol. 40, no. 3, pp. 1093-1099, 2004.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.tracking.association import (
    AssociationWeights,
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
    update_weighted,
)
from radar_forge.tracking.imm import (
    InteractingMultipleModel,
    imm_innovate,
    imm_predict,
    imm_update,
)
from radar_forge.tracking.initiation import Initiator
from radar_forge.tracking.lifecycle import (
    ExistencePolicy,
    LifecyclePolicy,
    by_existence,
    m_of_n,
)
from radar_forge.tracking.models import (
    MarginalModel,
    MeasurementModel,
    MotionModel,
    check_covariances,
)

__all__ = [
    "Coverage",
    "ExistencePolicy",
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

# One model's scoring of a scan: the innovation that gates, and each IMM mode's (none without).
Scored = tuple[Innovation, tuple[Innovation, ...]]


@dataclass(frozen=True, eq=False)
class Scan:
    """Everything one sensor measured at one time: z ``(n_meas, m)`` and R, checked and frozen.

    ``noise_covariance`` may be one ``(m, m)`` for every measurement; it is stored as the
    ``(n_meas, m, m)`` stack, each positive definite. ``(0, m)`` measurements is a scan that
    saw nothing. ``clutter_intensity`` ``(n_meas,)`` is the false-measurement density at each
    measurement, per scan per unit of z: an associator that weights needs it.

    References
    ----------
    Bar-Shalom [2], §4.3.1 (z = h(x) + w, w zero-mean with covariance R).
    """

    time_s: float
    measurements: NDArray[np.float64]
    noise_covariance: NDArray[np.float64]
    model_id: str
    clutter_intensity: NDArray[np.float64] | None = None

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
        if self.clutter_intensity is not None:
            clutter = np.array(self.clutter_intensity, dtype=np.float64)
            if clutter.shape != (n_meas,) or not np.all(np.isfinite(clutter) & (clutter > 0)):
                msg = f"clutter_intensity must be ({n_meas},), positive and finite."
                raise ValueError(msg)
            clutter.setflags(write=False)
            object.__setattr__(self, "clutter_intensity", clutter)
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


class Coverage(Protocol):
    """Which tracks a scan could have seen, such as those in a rotating radar's sector.

    A track the scan could not see is neither predicted, gated nor counted: its row waits,
    at its own time, for the scan that covers it.
    """

    def __call__(self, bank: TrackBank, scan: Scan) -> NDArray[np.bool_]:
        """Return ``(n_tracks,)`` True where the scan covers the track."""
        ...


class Tracker:
    """Track many targets from scans, every track a row of one bank.

    Parameters
    ----------
    models : mapping of str to MeasurementModel
        Every model by ID, marginals included, all in the motion model's layout.
    motion : MotionModel or InteractingMultipleModel
        One motion model, or an IMM's modes, whose union layout the models share.
    linearisation : Linearisation
        ``TaylorLinearisation()`` (KF, EKF) or ``UnscentedTransform()`` (UKF).
    cost : {"nis", "negative_log_likelihood_ratio"}, default "nis"
        NIS cannot compare pairs of different dimensions, so a scan that mixes them raises.
    clutter_density : mapping of str to float or None, optional
        λ per model ID: required for every model by the likelihood ratio, refused otherwise.
    model_selector : ModelSelector or None, optional
        None offers the scan's own model to every track.
    existence : ExistencePolicy or None, optional
        Confirm and delete by existence instead of M-of-N; needs an associator that weights.
    max_birth_clutter_intensity : float or None, optional
        No track starts from a measurement whose ``Scan.clutter_intensity`` exceeds this.

    References
    ----------
    Bar-Shalom and Li [1], §2.6.1, and Blackman and Popoli [3] (M-of-N and assignment).
    """

    def __init__(
        self,
        models: Mapping[str, MeasurementModel],
        motion: MotionModel | InteractingMultipleModel,
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
        coverage: Coverage | None = None,
        existence: ExistencePolicy | None = None,
        max_birth_clutter_intensity: float | None = None,
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
        position = [f"{axis}_m" for axis in "xyz" if f"{axis}_m" in layout.names]
        self._position = np.asarray(layout.indices(position), dtype=np.intp)
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
        self.coverage = coverage
        self.existence = existence
        self.max_birth_clutter_intensity = max_birth_clutter_intensity
        self.imm = motion if isinstance(motion, InteractingMultipleModel) else None
        n_modes = 0 if self.imm is None else len(self.imm.motions)
        self.bank = TrackBank.empty(layout, n_modes)
        self.time_s: float | None = None
        self.last_association: AssociationWeights | None = None
        self._next_track_id = 1
        # Whether the associator weights: asked once, of an empty scan.
        empty = np.zeros((0, 0))
        self._weighs = (
            associator.weights(empty, empty.astype(bool), clutter_intensity=np.zeros(0)) is not None
        )

    def seed(self, mean: ArrayLike, covariance: ArrayLike, time_s: float) -> NDArray[np.int64]:
        """Add tracks ``(n_new, n_state)`` from a cue and return their IDs; bistatic needs it."""
        if self.time_s is not None and time_s < self.time_s - _TIME_TOLERANCE_S:
            msg = f"cannot seed at {time_s} s, before the last scan at {self.time_s} s."
            raise ValueError(msg)
        n_new = len(np.asarray(mean))
        ids = np.arange(self._next_track_id, self._next_track_id + n_new, dtype=np.int64)
        self.bank = self._append(self.bank, mean, covariance, time_s, ids, None)
        self._next_track_id += n_new
        return ids

    def process(self, scan: Scan) -> TrackBank:
        """Run one scan; return survivors then births, this scan's deleted rows included once."""
        model = self._check(scan)
        time_s = max(scan.time_s, float(self.bank.time_s.max(initial=scan.time_s)))
        seen = np.ones(self.bank.n_tracks, dtype=bool)
        if self.coverage is not None:
            seen = np.asarray(self.coverage(self.bank, scan), dtype=bool)
        bank = self.bank if seen.all() else self.bank.compact(seen)
        bank = self._predict(bank, time_s)
        no_index = np.full(bank.n_tracks, -1, dtype=np.int64)
        bank = replace(bank, nis=np.full(bank.n_tracks, np.nan), measurement_index=no_index)
        live = time_s - bank.last_hit_time_s <= self.policy.max_coast_time_s
        cost, chosen, innovations = self._score(bank, scan, live)
        weighed = self._weigh(bank, scan, chosen, innovations)
        if weighed is None:
            assigned, blocked = self._assign(cost, chosen >= 0, bank.status)
            bank = self._update(bank, innovations, chosen, assigned)
            hit = assigned >= 0
            used = blocked.copy()
            used[assigned[hit]] = True
        else:
            bank, hit, used = weighed
        bank = self._count(bank, hit, live, time_s)
        if not seen.all():
            bank = self._merge(seen, bank)
        bank, next_track_id = self._start(bank, scan, model, used, time_s)
        self.bank = bank.compact(bank.status != TrackStatus.DELETED)
        self.time_s, self._next_track_id = time_s, next_track_id
        return bank

    def _check(self, scan: Scan) -> MeasurementModel:
        """Return the scan's model, or raise before anything changes."""
        model = self.models.get(scan.model_id)
        if model is None or scan.measurements.shape[1] != model.measurement_layout.dimension:
            msg = f"scan model {scan.model_id!r} is unknown, or its dimension does not match."
            raise ValueError(msg)
        if self._weighs and scan.clutter_intensity is None:
            msg = (
                "the associator weights by clutter intensity, so the Scan needs "
                f"clutter_intensity, one per measurement: shape ({len(scan.measurements)},)."
            )
            raise ValueError(msg)
        last_scan_s = -math.inf if self.time_s is None else self.time_s
        latest_s = max(last_scan_s, float(self.bank.time_s.max(initial=-math.inf)))
        if scan.time_s < latest_s - _TIME_TOLERANCE_S:
            msg = f"out-of-sequence scan at {scan.time_s} s, after {latest_s} s."
            raise ValueError(msg)
        return model

    def _append(
        self,
        bank: TrackBank,
        mean: ArrayLike,
        covariance: ArrayLike,
        time_s: float,
        ids: NDArray[np.int64],
        measurement_index: ArrayLike | None,
    ) -> TrackBank:
        """Append new rows: tentative (or confirmed under 1-of-N), at the initial existence."""
        n_new = len(ids)
        ones = np.ones(n_new, dtype=np.int32)
        status = m_of_n(np.zeros(n_new, dtype=np.int8), ones, ones, self.policy)
        existence = None
        if self.existence is not None:
            existence = np.full(n_new, self.existence.initial)
            status = by_existence(np.zeros(n_new, dtype=np.int8), existence, self.existence)
        return bank.append(
            mean,
            covariance,
            time_s,
            ids,
            status,
            measurement_index,
            existence=existence,
            mode_probability=None if self.imm is None else self.imm.initial_probability,
        )

    def _predict(self, bank: TrackBank, time_s: float) -> TrackBank:
        """Predict every row to ``time_s``, by its one model or its IMM, and its existence."""
        if isinstance(self.motion, InteractingMultipleModel):
            bank = imm_predict(bank, self.motion, self.linearisation, time_s)
        else:
            bank = predict(bank, self.motion, self.linearisation, time_s)
        if self.existence is not None:
            bank = replace(bank, existence=bank.existence * self.existence.p_survival)
        return bank

    def _merge(self, seen: NDArray[np.bool_], part: TrackBank) -> TrackBank:
        """Return the bank with the seen rows replaced by ``part``; the rest took nothing."""
        columns: dict[str, Any] = {}
        for column in fields(TrackBank)[1:]:
            values = getattr(self.bank, column.name).copy()
            if column.name == "nis":
                values[:] = np.nan
            elif column.name == "measurement_index":
                values[:] = -1
            values[seen] = getattr(part, column.name)
            columns[column.name] = values
        return TrackBank(self.bank.layout, **columns)

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
    ) -> tuple[NDArray[np.float64], NDArray[np.intp], list[Scored]]:
        """Gate every pair; return costs, each pair's model index (-1 none) and innovations."""
        everyone = [(scan.model_id, np.ones(bank.n_tracks, dtype=bool))]
        selector = self.model_selector
        offered = everyone if selector is None else selector(bank, scan.model_id)
        cost = np.full((bank.n_tracks, len(scan.measurements)), np.inf)
        chosen = np.full(cost.shape, -1, dtype=np.intp)
        innovations: list[Scored] = []
        # A loop over the few models offered; each is batched over every pair.
        for k, (model_id, allowed) in enumerate(offered):
            measurements, noise = self._view(scan, model_id)
            arguments = (bank, self.models[model_id], self.linearisation, measurements, noise)
            modes: tuple[Innovation, ...] = ()
            if self.imm is None:
                innovation = innovate(*arguments)
            else:
                innovation, modes = imm_innovate(*arguments)
            threshold = chi_square_gate_threshold(self.gate_probability, measurements.shape[1])
            passed = gate(innovation.nis, threshold) & (chosen < 0) & (live & allowed)[:, None]
            pair_cost = association_cost(innovation, self.cost, self.clutter_density.get(model_id))
            cost = np.where(passed, pair_cost, cost)
            chosen[passed] = k
            innovations.append((innovation, modes))
        sizes = {innovations[k][0].innovation.shape[-1] for k in np.unique(chosen[chosen >= 0])}
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

    def _correct(self, bank: TrackBank, scored: Scored, weights: ArrayLike) -> TrackBank:
        """Correct every row by association weights, through its one model or its IMM."""
        innovation, modes = scored
        if self.imm is None:
            return update_weighted(bank, innovation, weights)
        return imm_update(bank, innovation, modes, weights)

    def _update(
        self,
        bank: TrackBank,
        innovations: list[Scored],
        chosen: NDArray[np.intp],
        assigned: NDArray[np.intp],
    ) -> TrackBank:
        """Correct the assigned rows through the model each passed, and record the NIS."""
        rows = np.flatnonzero(assigned >= 0)
        through = np.full(bank.n_tracks, -1, dtype=np.intp)
        through[rows] = chosen[rows, assigned[rows]]
        nis, index = bank.nis.copy(), bank.measurement_index.copy()
        # One batched correction per model offered: a few, each over its own rows.
        for k, (innovation, modes) in enumerate(innovations):
            mine = np.flatnonzero(through == k)
            choice = np.full(bank.n_tracks, -1, dtype=np.intp)
            choice[mine] = assigned[mine]
            if self.imm is None:
                bank = update(bank, innovation, choice)
            else:
                one_hot = np.zeros((bank.n_tracks, innovation.nis.shape[1] + 1))
                one_hot[np.arange(bank.n_tracks), choice] = 1.0
                bank = imm_update(bank, innovation, modes, one_hot)
            nis[mine] = innovation.nis[mine, assigned[mine]]
        index[rows] = assigned[rows]
        return replace(bank, nis=nis, measurement_index=index)

    def _weigh(
        self, bank: TrackBank, scan: Scan, chosen: NDArray[np.intp], innovations: list[Scored]
    ) -> tuple[TrackBank, NDArray[np.bool_], NDArray[np.bool_]] | None:
        """Update every row by association weights; None for an associator that only assigns.

        Confirmed rows are weighted jointly (JPDA, JIPDA), and in ``"confirmed_first"``
        assignment tentative rows each alone (PDA, IPDA) over the measurements outside every
        confirmed gate, so a new track's wide gate never joins a cluster. Returns the bank,
        whether each row took a measurement (weights over half: its hit), and the measurements
        any row gated.
        """
        in_gate = chosen >= 0
        n_meas = in_gate.shape[1]
        first = np.ones(bank.n_tracks, dtype=bool)
        if self.assignment == "confirmed_first":
            first = np.isin(bank.status, _CONFIRMED)
        prior = bank.existence if self.existence is not None else None
        clutter = scan.clutter_intensity
        log_likelihood = innovations[0][0].log_likelihood
        joint = self.associator.weights(
            log_likelihood[first],
            in_gate[first],
            clutter_intensity=clutter,
            existence=None if prior is None else prior[first],
        )
        if joint is None:
            if self.existence is not None:
                msg = "an existence policy needs an associator that returns weights."
                raise ValueError(msg)
            return None
        if len(innovations) != 1:
            msg = "probabilistic association needs one measurement model a scan."
            raise ValueError(msg)
        weights = np.zeros((bank.n_tracks, n_meas + 1))
        weights[:, -1] = 1.0
        weights[first] = joint.weights
        existence = bank.existence.copy()
        existence[first] = joint.existence
        free = np.asarray(~in_gate[first].any(axis=0), dtype=bool)
        in_gate = in_gate & (first[:, None] | free[None, :])
        second = ~first
        if second.any():
            alone = self.associator.weights(
                log_likelihood[second][:, free],
                in_gate[second][:, free],
                clutter_intensity=None if clutter is None else clutter[free],
                existence=None if prior is None else prior[second],
                joint=False,
            )
            if alone is not None:
                rows = np.flatnonzero(second)
                weights[rows] = 0.0
                weights[np.ix_(rows, np.flatnonzero(free))] = alone.weights[:, :-1]
                weights[rows, -1] = alone.weights[:, -1]
                existence[rows] = alone.existence
        self.last_association = joint
        bank = self._correct(bank, innovations[0], weights)
        # A track took a measurement, its hit, when one is likelier than none; that one is
        # recorded as the measurement it took.
        taken = np.asarray(weights[:, :-1].sum(axis=1) > 0.5, dtype=bool)
        index = np.full(bank.n_tracks, -1, dtype=np.int64)
        nis = np.full(bank.n_tracks, np.nan)
        if n_meas:
            likeliest = weights[:, :-1].argmax(axis=1)
            index[taken] = likeliest[taken]
            nis[taken] = innovations[0][0].nis[np.flatnonzero(taken), likeliest[taken]]
        if self.existence is not None:
            # Round-off can leave a sure target at 1 + eps, which the next scan would refuse.
            bank = replace(bank, existence=np.clip(existence, np.finfo(float).tiny, 1.0))
        bank = replace(bank, nis=nis, measurement_index=index)
        return bank, taken, np.asarray(in_gate.any(axis=0), dtype=bool)

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
        if self.existence is not None:
            status = by_existence(status, bank.existence, self.existence)
            status[~live] = TrackStatus.DELETED
        else:
            status = m_of_n(status, n_hits, n_scans, policy)
            extra = np.where(np.isin(status, _CONFIRMED), policy.n_reacquire_misses, 0)
            status[~live | (n_misses >= policy.n_delete_misses + extra)] = TrackStatus.DELETED
        if policy.max_position_sigma_m is not None and self._position.size:
            block = bank.covariance[:, self._position[:, None], self._position[None, :]]
            largest_m2 = np.linalg.eigvalsh(block)[:, -1] if len(block) else np.zeros(0)
            wide = (largest_m2 > policy.max_position_sigma_m**2) & np.isin(status, _CONFIRMED)
            status[wide] = TrackStatus.DELETED
        # The miss that opens re-acquisition, and only it: a later reset would undo what the
        # coast has learnt.
        reset = (status == TrackStatus.COASTING) & (n_misses == policy.n_delete_misses)
        covariance, modes = bank.covariance.copy(), bank.mode_covariance.copy()
        # An IMM predicts from its modes, so each mode is reset with the mixture.
        covariance[reset] = self._reset_block(covariance[reset])
        modes[reset] = self._reset_block(modes[reset])
        return replace(
            bank,
            covariance=covariance,
            mode_covariance=modes,
            status=status,
            n_hits=n_hits,
            n_misses=n_misses,
            n_scans=n_scans,
            last_hit_time_s=np.where(hit, time_s, bank.last_hit_time_s),
        )

    def _reset_block(self, matrices: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return covariances ``(..., n, n)`` with the reset coordinates back at the prior."""
        kept, result = self._reset, matrices.copy()
        result[..., kept, :] = 0.0
        result[..., :, kept] = 0.0
        result[..., kept[:, None], kept[None, :]] = self.initiator.prior_covariance[
            np.ix_(kept, kept)
        ]
        return result

    def _start(
        self,
        bank: TrackBank,
        scan: Scan,
        model: MeasurementModel,
        used: NDArray[np.bool_],
        time_s: float,
    ) -> tuple[TrackBank, int]:
        """Append a tentative row per measurement no track used; return the next free ID."""
        allowed = ~used
        limit = self.max_birth_clutter_intensity
        if limit is not None and scan.clutter_intensity is not None:
            allowed &= scan.clutter_intensity <= limit
        free = np.flatnonzero(allowed)
        rows = self.initiator.initiate(scan.measurements[free], scan.noise_covariance[free], model)
        if rows is None or not free.size:
            return bank, self._next_track_id
        ids = np.arange(self._next_track_id, self._next_track_id + free.size, dtype=np.int64)
        return self._append(bank, *rows, time_s, ids, free), int(ids[-1]) + 1
