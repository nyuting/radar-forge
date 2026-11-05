r"""The interacting multiple model (IMM) estimator over a bank of tracks.

An IMM bank (:class:`~radar_forge.tracking.filters.TrackBank` with ``n_modes`` > 0) holds each
track's estimate under every mode and the mode probabilities; its ``mean`` and ``covariance``
are the mixture's. :func:`imm_predict` mixes [1] and predicts each mode with the one Gaussian
filter; :func:`imm_innovate` scores the mixture, so any associator's cost is the IMM's;
:func:`imm_update` corrects each mode through
:func:`~radar_forge.tracking.filters.update_weighted`, hard or probabilistic association
alike. Modes of different layouts mix in the largest one [3].

References
----------
.. [1] H. A. P. Blom and Y. Bar-Shalom, "The interacting multiple model algorithm for systems
       with Markovian switching coefficients," *IEEE Trans. Autom. Control*, vol. 33, no. 8,
       pp. 780-783, 1988.
.. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking
       and Navigation*, Wiley, 2001.
.. [3] K. Granström, P. Willett and Y. Bar-Shalom, "Systematic approach to IMM mixing for
       unequal dimension states," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 51, no. 4,
       pp. 2975-2986, 2015.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm
from scipy.special import logsumexp

from radar_forge.tracking.filters import (
    Innovation,
    Linearisation,
    TrackBank,
    innovate,
    update_weighted,
)
from radar_forge.tracking.models import MeasurementModel, MotionModel, StateLayout

__all__ = [
    "InteractingMultipleModel",
    "imm_innovate",
    "imm_mix",
    "imm_predict",
    "imm_update",
]


def _symmetrise(matrix: NDArray[Any]) -> NDArray[np.float64]:
    """Return the symmetric part of a stack of matrices, removing asymmetric roundoff."""
    result: NDArray[np.float64] = (matrix + np.swapaxes(matrix, -1, -2)) / 2
    return result


# A mode probability is floored here before a logarithm or a division: the IMM's weights are
# ratios of probabilities, and one that underflows to zero would otherwise give 0/0.
_TINY_PROBABILITY = 1e-300


class _Held:
    """A mode's motion model lifted into the IMM's layout; the coordinates it lacks are held."""

    def __init__(self, motion: MotionModel, layout: StateLayout) -> None:
        self.state_layout = layout
        self._motion = motion
        self._own = np.asarray(layout.indices(motion.state_layout.names), dtype=np.intp)

    def transition(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return the states with the mode's coordinates moved on and the others held."""
        result = np.array(state, dtype=np.float64)
        result[..., self._own] = self._motion.transition(result[..., self._own], dt_s)
        return result

    def jacobian(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return the mode's Jacobian on its coordinates and the identity on the others."""
        own, n_state = self._own, self.state_layout.dimension
        inner = self._motion.jacobian(np.asarray(state)[..., own], dt_s)
        result = np.zeros((*inner.shape[:-2], n_state, n_state)) + np.eye(n_state)
        result[..., own[:, None], own[None, :]] = inner
        return result

    def process_noise(self, state: NDArray[np.float64], dt_s: ArrayLike) -> NDArray[np.float64]:
        """Return the mode's Q on its coordinates and zero on the held ones."""
        own, n_state = self._own, self.state_layout.dimension
        inner = self._motion.process_noise(np.asarray(state)[..., own], dt_s)
        result = np.zeros((*inner.shape[:-2], n_state, n_state))
        result[..., own[:, None], own[None, :]] = inner
        return result


@dataclass(frozen=True, eq=False)
class InteractingMultipleModel:
    r"""The modes of an IMM: a motion model each, their mean sojourn times, absent coordinates.

    The bank's layout is the mode layout that holds every other's coordinates (CV inside the
    coordinated turn). A mode lacking a coordinate carries it at zero mean and
    ``absent_sigma`` standard deviation, uncorrelated, reset on every mixing and held through
    its prediction: what it contributes when mixed into a mode that has it ([3] §III, eq.
    (10), a fixed q). Switching is a continuous-time Markov chain with generator
    :math:`\Lambda_{ii} = -1/\tau_i`, :math:`\Lambda_{ij} = 1/((M-1)\tau_i)`, so the switching
    matrix :math:`\Pi(T) = e^{\Lambda T}` fits any revisit. New tracks start at the chain's
    stationary probabilities, which are proportional to :math:`\tau_i`.

    References
    ----------
    Blom and Bar-Shalom [1]; Bar-Shalom [2], §11.6.6; Granström, Willett and Bar-Shalom [3].
    """

    motions: tuple[MotionModel, ...]
    sojourn_time_s: tuple[float, ...]
    absent_sigma: Mapping[str, float] = field(default_factory=dict)
    state_layout: StateLayout = field(init=False)
    generator_per_s: NDArray[np.float64] = field(init=False, repr=False)
    initial_probability: NDArray[np.float64] = field(init=False, repr=False)
    held: tuple[_Held, ...] = field(init=False, repr=False)
    absent: tuple[NDArray[np.intp], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Find the union layout, the lifted modes and the generator, or raise."""
        motions = tuple(self.motions)
        sojourn = np.asarray(self.sojourn_time_s, dtype=np.float64)
        if not motions or sojourn.shape != (len(motions),) or not np.all(sojourn > 0):
            msg = f"need a motion model and a sojourn time > 0 per mode; got {sojourn}."
            raise ValueError(msg)
        layouts = [m.state_layout for m in motions]
        union = max(layouts, key=lambda layout: layout.dimension)
        if any(set(layout.names) - set(union.names) for layout in layouts):
            msg = f"one mode's layout must hold every other's coordinates; got {layouts}."
            raise ValueError(msg)
        everything = np.arange(union.dimension)
        absent = tuple(
            np.setdiff1d(everything, union.indices(layout.names)).astype(np.intp)
            for layout in layouts
        )
        missing = {union.names[i] for a in absent for i in a} - set(self.absent_sigma)
        if missing or not all(s > 0 for s in self.absent_sigma.values()):
            msg = f"absent_sigma needs a value > 0 for each of {sorted(missing)}."
            raise ValueError(msg)
        n_modes = len(motions)
        generator = np.zeros((n_modes, n_modes))
        if n_modes > 1:
            generator += 1.0 / ((n_modes - 1) * sojourn[:, None])
            np.fill_diagonal(generator, -1.0 / sojourn)
        # The dataclass is frozen, so the derived values are stored through object.
        object.__setattr__(self, "motions", motions)
        object.__setattr__(self, "state_layout", union)
        object.__setattr__(self, "generator_per_s", generator)
        object.__setattr__(self, "initial_probability", sojourn / sojourn.sum())
        object.__setattr__(self, "held", tuple(_Held(m, union) for m in motions))
        object.__setattr__(self, "absent", absent)

    def switching_matrix(self, dt_s: ArrayLike) -> NDArray[np.float64]:
        r"""Return :math:`\Pi(T) = e^{\Lambda T}` for each step, ``(..., n_modes, n_modes)``."""
        step_s = np.asarray(dt_s, dtype=np.float64)[..., None, None]
        result: NDArray[np.float64] = expm(self.generator_per_s * step_s)
        return result


def _mode_bank(bank: TrackBank, mode: int) -> TrackBank:
    """Return the bank with one IMM mode's estimate as every row's estimate."""
    return replace(bank, mean=bank.mode_mean[:, mode], covariance=bank.mode_covariance[:, mode])


def _combine(
    layout: StateLayout,
    weight: NDArray[np.float64],
    mean: NDArray[np.float64],
    covariance: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the moments of Gaussian mixtures over axis 1, weights ``(n, k)``."""
    offset = layout.residual(mean, reference[:, None, :])
    centre = np.einsum("nk,nks->ns", weight, offset)
    deviation = offset - centre[:, None, :]
    spread = covariance + deviation[..., :, None] * deviation[..., None, :]
    return layout.wrap(reference + centre), _symmetrise(np.einsum("nk,nkst->nst", weight, spread))


def imm_mix(
    bank: TrackBank, switching: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Return each mode's mixed mean and covariance, and the predicted mode probabilities.

    With :math:`\Pi_{ij}` the probability of switching from mode i to j, ``switching``
    ``(n_tracks, M, M)``: :math:`\bar{c}_j = \sum_i \Pi_{ij} \mu_i`, :math:`\mu_{i|j} =
    \Pi_{ij} \mu_i / \bar{c}_j`, and mode j starts from the moments of :math:`\sum_i \mu_{i|j}
    \mathcal{N}(\hat{x}_i, P_i)`.

    References
    ----------
    Bar-Shalom [2], §11.6.6 (the mixing probabilities and the mixed initial conditions).
    """
    joint = bank.mode_probability[:, :, None] * switching
    predicted = joint.sum(axis=1)
    n_tracks, n_modes, n_state = bank.mode_mean.shape
    # A mode whose probability underflowed mixes from itself alone rather than divide 0 by 0.
    floor = _TINY_PROBABILITY * np.eye(n_modes)
    mixing = (joint + floor) / (predicted + _TINY_PROBABILITY)[:, None, :]
    rows = np.repeat(np.arange(n_tracks), n_modes)  # broadcast-exempt: one row per target mode
    mean, covariance = _combine(
        bank.layout,
        np.swapaxes(mixing, 1, 2).reshape(-1, n_modes),
        bank.mode_mean[rows],
        bank.mode_covariance[rows],
        bank.mean[rows],
    )
    return (
        mean.reshape(n_tracks, n_modes, n_state),
        covariance.reshape(n_tracks, n_modes, n_state, n_state),
        predicted,
    )


def imm_predict(
    bank: TrackBank, imm: InteractingMultipleModel, linearisation: Linearisation, time_s: float
) -> TrackBank:
    """Return every row mixed, predicted per mode to ``time_s``, and recombined.

    References
    ----------
    Bar-Shalom [2], §11.6.6.
    """
    dt_s = time_s - bank.time_s
    mean, covariance, predicted = imm_mix(bank, imm.switching_matrix(dt_s))
    # A loop over the few modes; each is one batched prediction over every row.
    for mode, (motion, absent) in enumerate(zip(imm.held, imm.absent, strict=True)):
        sigma = np.array([imm.absent_sigma[imm.state_layout.names[i]] for i in absent])
        mean[:, mode, absent] = 0.0
        covariance[:, mode, absent, :] = 0.0
        covariance[:, mode, :, absent] = 0.0
        covariance[:, mode, absent, absent] = sigma**2
        mixed = replace(bank, mean=mean[:, mode], covariance=covariance[:, mode])
        mean[:, mode], covariance[:, mode] = linearisation.predict(motion, mixed, dt_s)
    combined_mean, combined_covariance = _combine(
        bank.layout, predicted, mean, covariance, bank.mean
    )
    return replace(
        bank,
        mean=combined_mean,
        covariance=combined_covariance,
        time_s=np.full_like(dt_s, time_s),
        mode_mean=mean,
        mode_covariance=covariance,
        mode_probability=predicted,
    )


def imm_innovate(
    bank: TrackBank,
    model: MeasurementModel,
    linearisation: Linearisation,
    measurements: ArrayLike,
    noise_covariance: ArrayLike,
) -> tuple[Innovation, tuple[Innovation, ...]]:
    r"""Score every row against every measurement, per mode and for the mixture.

    The first result gates: the moments of the combined estimate, its ``log_likelihood``
    replaced by the mode-mixed :math:`\ln \sum_j \bar{c}_j \mathcal{N}(z; \hat{z}_j, S_j)`, so
    any associator's cost is the IMM's. The second holds each mode's own innovation.

    References
    ----------
    Bar-Shalom [2], §11.6.6 (the mode-matched likelihoods).
    """
    modes = tuple(
        innovate(_mode_bank(bank, k), model, linearisation, measurements, noise_covariance)
        for k in range(bank.n_modes)
    )
    combined = innovate(bank, model, linearisation, measurements, noise_covariance)
    log_weight = np.log(np.maximum(bank.mode_probability, _TINY_PROBABILITY))
    per_mode = np.stack([m.log_likelihood for m in modes], axis=1)
    mixed = logsumexp(per_mode + log_weight[:, :, None], axis=1)
    return replace(combined, log_likelihood=mixed), modes


def imm_update(
    bank: TrackBank,
    combined: Innovation,
    modes: Sequence[Innovation],
    weights: ArrayLike,
) -> TrackBank:
    r"""Correct every mode by association weights, update the mode probabilities, recombine.

    Given that measurement i is the target's, mode j's posterior is :math:`\bar{c}_j
    \mathcal{N}_j(z_i)/g(z_i)`, g the mixed likelihood; given none is, :math:`\bar{c}_j`. So
    :math:`\mu_j = \bar{c}_j (\beta_0 + \sum_i \beta_i \mathcal{N}_j(z_i)/g(z_i))`, and mode j
    is corrected by :func:`update_weighted` with the weights conditioned on j. A one-hot row
    is the textbook IMM update, :math:`\mu_j \propto \Lambda_j \bar{c}_j`.

    References
    ----------
    Bar-Shalom [2], §11.6.6 (the mode probability update and the combination).
    """
    beta = np.asarray(weights, dtype=np.float64)
    gated = beta[:, :-1] > 0
    prior = np.maximum(bank.mode_probability, _TINY_PROBABILITY)
    # N_j / g is at most 1 / c_j, so the exponent is bounded; ungated pairs are never formed.
    gap = np.stack([m.log_likelihood for m in modes], axis=1) - combined.log_likelihood[:, None]
    ratio = np.where(gated[:, None, :], np.exp(np.where(gated[:, None, :], gap, 0.0)), 0.0)
    joint = np.concatenate([beta[:, None, :-1] * ratio, beta[:, None, -1:] + 0 * ratio[..., :1]], 2)
    joint = joint * prior[:, :, None]
    posterior = joint.sum(axis=2)
    posterior = posterior / posterior.sum(axis=1, keepdims=True)
    conditional = joint / np.maximum(joint.sum(axis=2, keepdims=True), _TINY_PROBABILITY)
    mean, covariance = bank.mode_mean.copy(), bank.mode_covariance.copy()
    # A loop over the few modes; each is one batched correction over every row.
    for k, innovation in enumerate(modes):
        corrected = update_weighted(_mode_bank(bank, k), innovation, conditional[:, k])
        mean[:, k], covariance[:, k] = corrected.mean, corrected.covariance
    combined_mean, combined_covariance = _combine(
        bank.layout, posterior, mean, covariance, bank.mean
    )
    return replace(
        bank,
        mean=combined_mean,
        covariance=combined_covariance,
        mode_mean=mean,
        mode_covariance=covariance,
        mode_probability=posterior,
    )
