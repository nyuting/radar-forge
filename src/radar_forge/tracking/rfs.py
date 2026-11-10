r"""Poisson multi-Bernoulli mixture (PMBM) tracking: the multi-target conjugate prior, carried.

The targets' density is a Poisson part, those never detected, at a uniform intensity over
coverage, times a mixture of multi-Bernoullis [1] (7), (9): tracks, each opened by a
measurement, whose single-target hypotheses are rows of one
:class:`~radar_forge.tracking.filters.TrackBank` with an existence probability r and a
Gaussian; a global hypothesis takes one per track and weighs their product (35). A scan
multiplies r by :math:`p_S` (§III-D); a prior hypothesis then has a misdetection child of
weight :math:`1 - r P_D P_G` (31) and a detection child of weight :math:`r P_D g` and r = 1
per gated measurement (30), and each measurement opens a track of weight
:math:`\lambda(z) + e(z)` and existence :math:`e/(\lambda + e)` (21)-(24). Murty's method ranks
each cluster's global hypotheses over the cost of §V-C3; weights and existences too small are
pruned (Algorithm 1). Williams [2] shows MHT and JIPDA approximate the same posterior, so
this is the reference their losses are measured against.

References
----------
.. [1] Á. F. García-Fernández, J. L. Williams, K. Granström and L. Svensson, "Poisson
       multi-Bernoulli mixture filter: direct derivation and implementation," *IEEE Trans.
       Aerosp. Electron. Syst.*, vol. 54, no. 4, pp. 1883-1901, 2018; arXiv:1703.04264.
.. [2] J. L. Williams, "Marginal multi-Bernoulli filters: RFS derivation of MHT, JIPDA, and
       association-based MeMBer," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 51, no. 3,
       pp. 1664-1687, 2015.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.tracking.association import chi_square_gate_threshold
from radar_forge.tracking.filters import (
    Linearisation,
    TaylorLinearisation,
    TrackBank,
    TrackStatus,
    innovate,
    predict,
    update,
)
from radar_forge.tracking.hypotheses import GlobalHypotheses, collapse, extend, split
from radar_forge.tracking.initiation import Initiator
from radar_forge.tracking.models import MeasurementModel, MotionModel
from radar_forge.tracking.tracker import Coverage, Scan

__all__ = ["PmbmTracker"]

# A new hypothesis's measurement code (hypotheses.Extension): -1 a miss, -2 not covered.
_MISS, _CARRIED = -1, -2


def _normalised(hypotheses: GlobalHypotheses) -> GlobalHypotheses:
    """Return the hypotheses with each cluster's weights summing to one, as logs."""
    labels, index = np.unique(hypotheses.cluster, return_inverse=True)
    total = np.full(labels.size, -np.inf)
    np.logaddexp.at(total, index, hypotheses.log_weight)
    return replace(hypotheses, log_weight=hypotheses.log_weight - total[index])


class PmbmTracker:
    r"""Track many targets by the PMBM filter; report each best hypothesis's likely Bernoullis.

    One motion model and one measurement model. ``coverage_volume`` is the measurement-space
    volume the birth intensity spreads over (m rad for range and azimuth); the undetected
    intensity is held at the fixed point of :math:`\lambda_u \leftarrow p_S (1 - P_D)
    \lambda_u + \lambda_b`, uniform, so every sector of a rotation sees it alike, and zero
    where ``Scan.clutter_intensity`` exceeds ``max_birth_clutter_intensity``, the birth limit
    every tracker shares. ``initiator`` places a first detection's density. Under
    ``"confirmed_first"`` only Bernoullis of existence 0.5 or more join clusters, as the
    tracker's confirmed tracks do (:func:`~radar_forge.tracking.hypotheses.extend`).
    :meth:`process` returns each track's marginal Bernoulli, moment-matched over its
    hypotheses ([2] §IV-A, (65)), where its existence passes 0.5, labelled by the
    measurement that opened the track. The estimators of [1] §VI report one global
    hypothesis's Bernoullis instead; the marginal is the posterior's own mean, the
    estimate JPDA's moment matching approximates.

    References
    ----------
    García-Fernández et al. [1], (21)-(24), (30), (31), (35), §III-D, §V-C3 and Algorithm 1;
    Williams [2], §IV-A, (65) (the marginal reported).
    """

    def __init__(
        self,
        motion: MotionModel,
        measurement: MeasurementModel,
        initiator: Initiator,
        *,
        detection_probability: float,
        gate_probability: float,
        coverage_volume: float,
        birth_intensity_per_scan: float = 0.05,
        survival_probability: float = 0.995,
        k_best: int = 100,
        prune_existence: float = 1e-5,
        prune_weight: float = 1e-4,
        max_birth_clutter_intensity: float | None = None,
        linearisation: Linearisation | None = None,
        coverage: Coverage | None = None,
        assignment: Literal["confirmed_first", "joint"] = "joint",
    ) -> None:
        p_d, p_s = detection_probability, survival_probability
        if not (0 < p_d <= 1 and 0 < gate_probability < 1 and 0 < p_s <= 1):
            msg = f"need 0 < P_D, p_S <= 1 and 0 < P_G < 1; got {p_d}, {p_s}, {gate_probability}."
            raise ValueError(msg)
        if not (
            coverage_volume > 0
            and birth_intensity_per_scan >= 0
            and type(k_best) is int
            and k_best >= 1
            and 0 < prune_existence < 0.5
            and 0 <= prune_weight < 1
        ):
            msg = (
                "need a positive coverage, births >= 0, k_best >= 1 and prune thresholds in range."
            )
            raise ValueError(msg)
        if measurement.state_layout != motion.state_layout:
            msg = "the measurement model must share the motion model's state layout."
            raise ValueError(msg)
        self.motion, self.measurement, self.initiator = motion, measurement, initiator
        self.linearisation = linearisation or TaylorLinearisation()
        self.coverage, self.assignment = coverage, assignment
        self.detection_probability, self.gate_probability = p_d, gate_probability
        self.survival_probability, self.k_best = p_s, k_best
        self.prune_existence, self.prune_weight = prune_existence, prune_weight
        self.max_birth_clutter_intensity = max_birth_clutter_intensity
        birth = birth_intensity_per_scan / coverage_volume
        self.undetected_intensity = birth / (1 - p_s * (1 - p_d))
        self.local = TrackBank.empty(motion.state_layout)
        self.hypotheses = GlobalHypotheses.empty()
        self.bank = self.local
        self.time_s: float | None = None
        self._next_track_id = 1

    @property
    def bernoulli_count(self) -> int:
        """Return how many single-target hypotheses (Bernoullis) the mixture holds."""
        return self.local.n_tracks

    def largest_cluster(self) -> int:
        """Return how many tracks the largest cluster holds."""
        if not self.local.n_tracks:
            return 0
        leaf_cluster = self.hypotheses.leaf_cluster(self.local.n_tracks)
        pairs = np.unique(np.column_stack([leaf_cluster, self.local.track_id]), axis=0)
        return int(np.unique(pairs[:, 0], return_counts=True)[1].max())

    def seed(self, mean: ArrayLike, covariance: ArrayLike, time_s: float) -> NDArray[np.int64]:
        """Add tracks ``(n_new, n_state)`` that surely exist, each its own cluster; return IDs."""
        n_new = len(np.asarray(mean))
        ids = np.arange(self._next_track_id, self._next_track_id + n_new, dtype=np.int64)
        status = np.full(n_new, TrackStatus.CONFIRMED)
        self.hypotheses = self.hypotheses.plant(self.local.n_tracks, n_new, 0.0)
        self.local = self.local.append(mean, covariance, time_s, ids, status)
        self._next_track_id += n_new
        self.bank = self._report()
        return ids

    def process(self, scan: Scan) -> TrackBank:
        """Run one scan: predict, update every global hypothesis, prune; return the report."""
        if scan.clutter_intensity is None:
            msg = "PMBM weighs by clutter intensity, so the Scan needs clutter_intensity."
            raise ValueError(msg)
        if scan.measurements.shape[1] != self.measurement.measurement_layout.dimension:
            msg = f"the scan's dimension does not match the measurement model's: {scan.model_id!r}."
            raise ValueError(msg)
        local = self.local
        time_s = max(scan.time_s, float(local.time_s.max(initial=scan.time_s)))
        seen = np.ones(local.n_tracks, dtype=bool)
        if self.coverage is not None:
            seen = np.asarray(self.coverage(local, scan), dtype=bool)
        seen = np.isin(local.track_id, local.track_id[seen])
        p_d, p_g = self.detection_probability, self.gate_probability
        bank = predict(local.compact(seen), self.motion, self.linearisation, time_s)
        existence = bank.existence * self.survival_probability
        z, noise = scan.measurements, scan.noise_covariance
        innovation = innovate(bank, self.measurement, self.linearisation, z, noise)
        in_gate = innovation.nis <= chi_square_gate_threshold(p_g, z.shape[1])
        clutter = scan.clutter_intensity
        limit = (
            np.inf if self.max_birth_clutter_intensity is None else self.max_birth_clutter_intensity
        )
        detected = np.where(clutter <= limit, self.undetected_intensity * p_d, 0.0)
        opened = detected / (clutter + detected)
        unassigned = np.log(clutter + detected)
        hit = np.log(existence * p_d)[:, None] + innovation.log_likelihood
        miss = np.log1p(-existence * p_d * p_g)
        cost = np.where(in_gate, -hit + unassigned[None, :] + miss[:, None], np.inf)
        step = extend(
            self.hypotheses,
            seen,
            cost,
            in_gate,
            non_assignment_cost=0.0,
            hit=hit,
            miss=miss,
            unassigned=unassigned,
            born=opened >= self.prune_existence,
            k_best=self.k_best,
            per_parent="weighted",
            linking=None if self.assignment == "joint" else existence >= 0.5,
            min_share=self.prune_weight,
        )
        parent, code = step.parent, step.measurement
        # Index -1, a new track's parent, reads an appended last entry.
        row = np.append(np.cumsum(seen) - 1, -1)
        prior = np.append(existence, 1.0)[row[parent]]
        missed = prior * (1 - p_d * p_g) / (1 - prior * p_d * p_g)
        stored = np.append(local.existence, 1.0)[parent]
        new_existence = np.where(code >= 0, 1.0, missed)
        new_existence = np.where(code == _CARRIED, stored, new_existence)
        new_existence = np.where(
            parent < 0, np.append(opened, 0.0)[np.where(parent < 0, code, -1)], new_existence
        )
        used, hypotheses = self._prune(step.hypotheses, new_existence)
        parent, code, new_existence = parent[used], code[used], new_existence[used]
        kept, new = code == _CARRIED, parent < 0
        grown = ~kept & ~new
        rows, taken = row[parent[grown]], code[grown]
        children = update(bank.take(rows), innovation.take(rows), taken)
        nis = np.column_stack([innovation.nis, np.full(bank.n_tracks, np.nan)])[rows, taken]
        children = replace(
            children,
            existence=new_existence[grown],
            measurement_index=taken.astype(np.int64),
            nis=nis,
        )
        free = code[new]
        ids = np.arange(self._next_track_id, self._next_track_id + free.size, dtype=np.int64)
        roots = bank.take(np.zeros(0, dtype=np.intp))
        start = self.initiator.initiate(z[free], noise[free], self.measurement)
        if start is not None and free.size:
            status = np.full(free.size, TrackStatus.TENTATIVE)
            roots = roots.append(*start, time_s, ids, status, free, existence=opened[free])
        carried = local.take(parent[kept])
        carried = replace(
            carried,
            nis=np.full(carried.n_tracks, np.nan),
            measurement_index=np.full(carried.n_tracks, -1, dtype=np.int64),
        )
        order = np.concatenate([np.flatnonzero(kept), np.flatnonzero(grown), np.flatnonzero(new)])
        self.local = carried.join(children, roots).take(np.argsort(order))
        self.hypotheses = _normalised(
            collapse(
                split(hypotheses, self.local.track_id, np.zeros(self.local.n_tracks)),
                self.k_best,
                "sum",
            )
        )
        self.time_s, self._next_track_id = time_s, self._next_track_id + free.size
        self.bank = self._report()
        return self.bank

    def _prune(
        self, hypotheses: GlobalHypotheses, existence: NDArray[np.float64]
    ) -> tuple[NDArray[np.bool_], GlobalHypotheses]:
        """Keep the heavy global hypotheses and their likely Bernoullis (Algorithm 1).

        Returns which new Bernoullis survive, and the hypotheses over the survivors.
        """
        weighted = _normalised(hypotheses)
        best = np.zeros(weighted.cluster.size, dtype=bool)
        best[weighted.best()] = True
        heavy = best | (weighted.log_weight >= math.log(max(self.prune_weight, 1e-300)))
        weighted = weighted.keep(heavy)
        likely = existence[weighted.leaf] >= self.prune_existence
        weighted = GlobalHypotheses(
            weighted.leaf[likely],
            weighted.hypothesis[likely],
            weighted.log_weight,
            weighted.cluster,
        )
        kept = collapse(weighted, self.k_best, "sum")
        used = kept.held(existence.size)
        return used, kept.drop(~used)

    def _report(self) -> TrackBank:
        """Return each track's marginal Bernoulli where its existence passes 0.5.

        A track's local hypotheses weigh their global hypotheses' weights; its existence is
        their sum times each one's r, and its density their mixture, moment-matched: the
        track-oriented projection of the posterior, [2] §IV-A, (65). A track is confirmed in
        the scans its likeliest hypothesis took a measurement, coasting otherwise.
        """
        local, hypotheses = self.local, self.hypotheses
        n_rows = local.n_tracks
        share = np.bincount(
            hypotheses.leaf,
            weights=np.exp(hypotheses.log_weight)[hypotheses.hypothesis],
            minlength=n_rows,
        )
        mass = share * local.existence
        ids, track = np.unique(local.track_id, return_inverse=True)
        track = track.reshape(-1)
        existence = np.bincount(track, weights=mass, minlength=ids.size)
        order = np.lexsort((-mass, track))
        likeliest = order[np.searchsorted(track[order], np.arange(ids.size))]
        mean = np.zeros((ids.size, local.layout.dimension))
        np.add.at(mean, track, mass[:, None] * local.mean)
        mean /= np.maximum(existence, np.finfo(float).tiny)[:, None]
        offset = local.mean - mean[track]
        spread = local.covariance + offset[:, :, None] * offset[:, None, :]
        covariance = np.zeros((ids.size, *local.covariance.shape[1:]))
        np.add.at(covariance, track, mass[:, None, None] * spread)
        covariance /= np.maximum(existence, np.finfo(float).tiny)[:, None, None]
        keep = existence > 0.5
        report = local.take(likeliest[keep])
        took = report.measurement_index >= 0
        return replace(
            report,
            mean=mean[keep],
            covariance=(covariance[keep] + np.swapaxes(covariance[keep], -1, -2)) / 2,
            existence=np.minimum(existence[keep], 1.0),
            status=np.where(took, TrackStatus.CONFIRMED, TrackStatus.COASTING).astype(np.int8),
        )
