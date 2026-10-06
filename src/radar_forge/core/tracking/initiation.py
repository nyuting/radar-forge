"""Starting a new track from one measurement.

When a measurement matches no existing track, the tracker asks an *initiator*
for a new filter. Track initiation is kept apart from track management
(:mod:`radar_forge.core.tracking.lifecycle`), which decides when a track is
confirmed or deleted.

- :class:`TrackInitiator` is the interface every initiator follows.
- :class:`DirectStateInitiator` copies the measured coordinates straight into the
  new state, and takes everything else from a fixed prior: the starting guess
  used before any measurement.

Not implemented yet: an initiator that picks a different rule for each
measurement model, and one that never starts a track from a detection, for a
sensor whose tracks must all come from
:meth:`~radar_forge.core.tracking.tracker.Tracker.seed`.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications
       to Tracking and Navigation*, Wiley, 2001, §5.5.3 (initialising a tracking
       filter from its first measurement).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from radar_forge.core.tracking.coordinates import StateEstimate
from radar_forge.core.tracking.estimation import Estimator
from radar_forge.core.tracking.measurement_models import (
    CartesianPosition,
    Measurement,
    MeasurementModel,
)

__all__ = [
    "DirectStateInitiator",
    "TrackInitiator",
]


def _check_independent_blocks(covariance: NDArray[np.float64], indices: tuple[int, ...]) -> None:
    """Raise unless the measured and unmeasured parts of a covariance are uncorrelated.

    Parameters
    ----------
    covariance : numpy.ndarray
        Prior covariance, shape ``(n_state, n_state)``.
    indices : tuple of int
        Positions of the measured coordinates.

    Raises
    ------
    ValueError
        If any entry linking a measured coordinate to an unmeasured one is
        nonzero.
    """
    # Take the rows of the measured coordinates, then delete their columns. What is left is
    # the block that links each measured coordinate to every unmeasured one.
    if np.any(np.delete(covariance[list(indices)], list(indices), axis=1) != 0):
        msg = (
            "direct initiation requires independent observed/unobserved prior blocks: "
            f"the prior covariance links coordinates {indices} to the others."
        )
        raise ValueError(msg)


class TrackInitiator(Protocol):
    """The interface of an initiator: one measurement in, a new filter (or None) out."""

    def initiate(self, measurement: Measurement, model: MeasurementModel) -> Estimator | None:
        """Return a new filter started from ``measurement``, or None.

        Parameters
        ----------
        measurement : Measurement
            The measurement that matched no track.
        model : MeasurementModel
            The model that explains it.

        Returns
        -------
        Estimator or None
            A new filter at the measurement's time, independent of every other
            filter. None means this measurement cannot start a track, for example
            because it does not pin down enough of the state.
        """
        ...


class DirectStateInitiator:
    """Start a track by copying the measured coordinates into a prior.

    The new state starts as a copy of ``prior``. The coordinates the measurement
    reports are then replaced: their values by the measured value, and their
    block of the covariance by the measurement covariance R. Everything else,
    such as the velocity of a position-only sensor, keeps the prior's value.

    This only works for a :class:`CartesianPosition` model, which reports
    coordinates directly. It also needs the prior's measured and unmeasured
    coordinates to be independent (zero covariance between them). Otherwise,
    replacing one block would leave cross terms that no longer fit, and the
    covariance could stop being positive semidefinite.

    Parameters
    ----------
    estimator_factory : callable
        Builds a fresh filter from a :class:`StateEstimate`.
    prior : StateEstimate
        The starting guess: mean, shape ``(n_state,)``, and covariance, shape
        ``(n_state, n_state)``. Its timestamp is not used.
    measured_names : tuple of str or None, optional
        Keyword-only. The coordinates the measurements will report, such as
        ``("x_m", "y_m")``. If given, the independence of the prior is checked
        once, here, when the initiator is built. A model reporting other
        coordinates then gets None from :meth:`initiate`. If None (the
        default), the check runs on every call to :meth:`initiate`, during a
        scan. A bad prior is then found only when the first measurement tries
        to start a track.

    Raises
    ------
    ValueError
        If ``measured_names`` names a coordinate not in the prior's layout, or
        the prior links those coordinates to the others.

    Notes
    -----
    There is no initiator for
    :class:`~radar_forge.core.tracking.measurement_models.BistaticRangeDopplerModel`.
    One bistatic measurement gives a path length and a path rate. That pins
    the target to a surface, not to a point, so it cannot be copied into a
    state. This initiator returns None for it. So a tracker whose only model
    is bistatic never starts a track by itself. Its tracks must be added with
    :meth:`~radar_forge.core.tracking.tracker.Tracker.seed`.

    References
    ----------
    .. [1] Bar-Shalom, Li and Kirubarajan (2001), §5.5.3; see the module References.

    Examples
    --------
    >>> import numpy as np
    >>> from radar_forge.core.tracking.motion import CartesianMotion
    >>> from radar_forge.core.tracking.ukf import UKF
    >>> motion = CartesianMotion({"x": "CV"}, origin_lla_deg_m=(36.0, -78.9, 60.0))
    >>> layout = motion.state_layout
    >>> prior = StateEstimate(np.zeros(2), np.diag([1.0, 100.0]), 0.0, layout)
    >>> initiator = DirectStateInitiator(
    ...     lambda state: UKF(state, motion), prior, measured_names=("x_m",)
    ... )
    >>> z = Measurement(np.array([50.0]), np.array([[4.0]]), 2.0, "radar", "position")
    >>> ukf = initiator.initiate(z, CartesianPosition(layout, ("x_m",)))
    >>> ukf.state.mean, ukf.state.timestamp_s
    (array([50.,  0.]), 2.0)
    """

    def __init__(
        self,
        estimator_factory: Callable[[StateEstimate], Estimator],
        prior: StateEstimate,
        *,
        measured_names: tuple[str, ...] | None = None,
    ) -> None:
        measured = None if measured_names is None else prior.state_layout.indices(measured_names)
        if measured is not None:
            _check_independent_blocks(prior.covariance, measured)

        self.estimator_factory = estimator_factory
        self.prior = prior
        self._measured = measured

    def initiate(self, measurement: Measurement, model: MeasurementModel) -> Estimator | None:
        """Return a new filter whose measured coordinates come from ``measurement``.

        Parameters
        ----------
        measurement : Measurement
            The measurement that matched no track. Its value has shape
            ``(n_meas,)`` and its covariance ``(n_meas, n_meas)``.
        model : MeasurementModel
            The model that explains it.

        Returns
        -------
        Estimator or None
            A new filter at the measurement's time. None if ``model`` is not a
            :class:`CartesianPosition` on the prior's layout, or reports other
            coordinates than ``measured_names``.

        Raises
        ------
        ValueError
            Only when ``measured_names`` was not given: if the prior links the
            model's coordinates to the others.
        """
        if not isinstance(model, CartesianPosition) or model.state_layout != (
            self.prior.state_layout
        ):
            return None
        if self._measured is not None and model.indices != self._measured:
            return None
        if self._measured is None:
            _check_independent_blocks(self.prior.covariance, model.indices)

        indices = list(model.indices)
        mean = self.prior.mean.copy()
        covariance = self.prior.covariance.copy()
        mean[indices] = measurement.value
        covariance[np.ix_(indices, indices)] = measurement.covariance
        return self.estimator_factory(
            StateEstimate(mean, covariance, measurement.timestamp_s, model.state_layout)
        )
