"""Typed TOML configuration and composition for native tracking.

References
----------
.. [1] spec/tracker-001-integration.md; StoneSoup explicit component composition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any, TypeVar

import numpy as np

from radar_forge.core.detection_2d import DetectionConfig
from radar_forge.tracking import (
    UKF,
    CartesianMotion,
    CartesianPosition,
    ChiSquareGate,
    DirectStateInitiator,
    GlobalNearestNeighbour,
    LifecyclePolicy,
    MeasurementModel,
    MotionModel,
    NearestNeighbour,
    Sensor,
    StateEstimate,
    StateSpace,
    TrackerEngine,
    TrackInitiator,
    TrackManager,
)

__all__ = [
    "ModelRegistry",
    "TrackingConfig",
    "build_enu_tracker",
    "build_tracker",
    "parse_tracking_table",
]

_Config = TypeVar("_Config", DetectionConfig, "TrackingConfig")


@dataclass(frozen=True)
class TrackingConfig:
    """Shared estimator/lifecycle settings for the radial scenario adapter.

    Parameters
    ----------
    acceleration_noise_density_m2ps3 : float
        Continuous white acceleration density in m²/s³.
    alpha, beta, kappa : float
        Scaled unscented-transform parameters.
    gate_probability : float
        Chi-square validation probability.
    association : str
        GNN or NN; both use normalized innovation squared costs.
    initial_velocity_std_mps : float
        Unobserved initial closing velocity uncertainty.
    max_velocity_mps : float
        Dual-PRF physical search prior, independent of trajectory truth.
    unfolding_tolerance_bins : float
        Residual threshold in coarser Doppler bins.
    confirmation_hits, confirmation_window, deletion_misses, history_size : int
        Event-based lifecycle counts.
    max_coast_time_s : float
        Expire after a greater-than gap in seconds.

    References
    ----------
    .. [1] Tracker 001 defaults; local lifecycle policy and Julier UKF.
    """

    acceleration_noise_density_m2ps3: float = 1.0
    alpha: float = 0.5
    beta: float = 2.0
    kappa: float = 0.0
    gate_probability: float = 0.997
    association: str = "GNN"
    initial_velocity_std_mps: float = 100.0
    max_velocity_mps: float = 191.0
    unfolding_tolerance_bins: float = 2.0
    confirmation_hits: int = 3
    confirmation_window: int = 5
    deletion_misses: int = 5
    max_coast_time_s: float = 30.0
    history_size: int = 100

    # This function checks the stored filter, matching, and track-management settings before
    # building a tracker.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - None; raises an error for invalid numerical settings or management limits.
    def __post_init__(self) -> None:
        """Validate numerical settings before any stateful processing."""
        numeric = [
            self.acceleration_noise_density_m2ps3,
            self.alpha,
            self.beta,
            self.kappa,
            self.gate_probability,
            self.initial_velocity_std_mps,
            self.max_velocity_mps,
            self.unfolding_tolerance_bins,
            self.max_coast_time_s,
        ]
        if (
            not np.all(np.isfinite(numeric))
            or self.acceleration_noise_density_m2ps3 < 0
            or self.alpha <= 0
            or self.beta < 0
        ):
            raise ValueError("tracking noise and UKF parameters must be finite and valid")
        if self.association not in ("NN", "GNN") or not 0 < self.gate_probability < 1:
            raise ValueError("association must be NN/GNN and gate_probability in (0,1)")
        if (
            min(self.initial_velocity_std_mps, self.max_velocity_mps, self.unfolding_tolerance_bins)
            <= 0
        ):
            raise ValueError("velocity prior and unfolding bounds must be positive")
        self.lifecycle()

    # This function turns stored settings into the rules for confirming, remembering, and removing
    # tracks.
    #
    # Inputs:
    # - None.
    #
    # Outputs:
    # - LifecyclePolicy: Hit/miss thresholds, maximum gap in seconds, and history limit.
    def lifecycle(self) -> LifecyclePolicy:
        """Build the event policy; see TrackingConfig References."""
        return LifecyclePolicy(
            self.confirmation_hits,
            self.confirmation_window,
            self.deletion_misses,
            self.max_coast_time_s,
            self.history_size,
        )


# This function turns a decoded settings table into checked configuration for tracker
# construction.
#
# Inputs:
# - table (mapping of str to values): Settings already read from TOML.
# - config_type (type): TrackingConfig or DetectionConfig, selecting which settings are allowed.
#
# Outputs:
# - TrackingConfig or DetectionConfig: Validated settings with defaults for omitted fields;
#   unknown keys or wrong types raise an error.
def parse_tracking_table(table: Mapping[str, Any], config_type: type[_Config]) -> _Config:
    """Parse a strict flat TOML table without coercing strings or fractional counts.

    Parameters
    ----------
    table : mapping
        Decoded TOML values.
    config_type : type
        DetectionConfig or TrackingConfig.

    Returns
    -------
    DetectionConfig or TrackingConfig
        Validated typed settings; unknown fields raise ValueError.

    References
    ----------
    .. [1] Radar-Forge TOML configuration conventions.
    """
    if not isinstance(table, Mapping):
        raise ValueError("configuration must be a TOML table")
    defaults = config_type()
    allowed = {f.name for f in fields(defaults)}
    if set(table) - allowed:
        raise ValueError(f"unknown {config_type.__name__} keys: {sorted(set(table) - allowed)}")
    # TOML scalar kinds must be checked before dataclass construction.
    for name, value in table.items():
        default = getattr(defaults, name)
        valid = (
            type(value) is type(default)
            if not isinstance(default, float)
            else type(value) in (float, int)
        )
        if not valid:
            raise ValueError(f"{name} must have type {type(default).__name__}")
    return config_type(**dict(table))


# This function assembles a one-sensor tracker from compatible models, starting assumptions, and
# management settings.
#
# Inputs:
# - motion (MotionModel): Rule for predicting movement and added uncertainty.
# - observation (MeasurementModel): Rule for predicting what the sensor would read.
# - prior (StateEstimate): Starting values and uncertainty for quantities not supplied by a
#   measurement.
# - config (TrackingConfig or None): Filter, matching, and management settings; None uses
#   defaults.
# - initiator (TrackInitiator or None): Track-starting rule; None starts from directly measured
#   coordinates.
# - sensor_id (str): ID assigned to the registered sensor.
# - model_id (str): ID assigned to its measurement rule.
#
# Outputs:
# - TrackerEngine: Ready-to-use tracker with no active tracks; the prior is reserved for later
#   track starts.
def build_tracker(
    motion: MotionModel,
    observation: MeasurementModel,
    prior: StateEstimate,
    config: TrackingConfig | None = None,
    *,
    initiator: TrackInitiator | None = None,
    sensor_id: str = "sensor",
    model_id: str = "measurement",
) -> TrackerEngine:
    """Compose an estimator-independent engine for one registered measurement route.

    Parameters
    ----------
    motion, observation : model
        Compatible semantic models; arbitrary dimensions are supported.
    prior : StateEstimate
        Explicit unobserved-coordinate prior (n,), (n,n).
    config : TrackingConfig or None
        UKF, gate, association and lifecycle settings.
    initiator : TrackInitiator or None
        Custom birth rule; default DirectStateInitiator handles coordinate selections.
    sensor_id, model_id : str
        Stable registered identifiers.

    References
    ----------
    .. [1] StoneSoup component composition; local routed initiation and event semantics.
    """
    cfg = config or TrackingConfig()
    if motion.state_space != observation.state_space or prior.state_space != motion.state_space:
        raise ValueError("motion, observation and prior require the same StateSpace")

    # This function creates a fresh filter using the motion model and settings chosen by the
    # surrounding builder.
    #
    # Inputs:
    # - state (StateEstimate): Starting values, uncertainty, time, and coordinate definitions.
    #
    # Outputs:
    # - UKF: Independent filter with a copy of the supplied estimate.
    def factory(state: StateEstimate) -> UKF:
        return UKF(state, motion, cfg.alpha, cfg.beta, cfg.kappa)

    # Validate dimension-dependent sigma-point settings before first birth.
    factory(prior)
    birth = initiator or DirectStateInitiator(factory, prior)
    return TrackerEngine(
        {model_id: observation},
        {sensor_id: Sensor(sensor_id, (model_id,))},
        ChiSquareGate(cfg.gate_probability),
        GlobalNearestNeighbour() if cfg.association == "GNN" else NearestNeighbour(),
        TrackManager(birth, cfg.lifecycle()),
    )


@dataclass(frozen=True)
class ModelRegistry:
    """Explicit custom-model builders for application TOML composition.

    Parameters
    ----------
    motions, measurements : mapping
        Type IDs to builders. Options are decoded TOML dictionaries. Measurement
        builders receive the resolved StateSpace. Each builder validates its own
        options; custom initiation is explicitly passed to build_tracker.

    References
    ----------
    .. [1] Local model-builder registry; StoneSoup component composition.
    """

    motions: Mapping[str, Callable[[Mapping[str, Any]], MotionModel]]
    measurements: Mapping[str, Callable[[Mapping[str, Any], StateSpace], MeasurementModel]]

    # This function selects registered builders and checks that their motion and reading models
    # use matching coordinates.
    #
    # Inputs:
    # - motion_type (str): Registered name of the motion-model builder.
    # - motion_options (mapping of str to values): Settings passed to that builder.
    # - measurement_type (str): Registered name of the reading-model builder.
    # - measurement_options (mapping of str to values): Settings passed to the reading-model
    #   builder.
    #
    # Outputs:
    # - tuple of MotionModel and MeasurementModel: Compatible models ready for tracker
    #   construction.
    def build(
        self,
        motion_type: str,
        motion_options: Mapping[str, Any],
        measurement_type: str,
        measurement_options: Mapping[str, Any],
    ) -> tuple[MotionModel, MeasurementModel]:
        """Resolve named builders; reject unknown IDs and incompatible spaces.

        References
        ----------
        .. [1] ModelRegistry class references.
        """
        if motion_type not in self.motions or measurement_type not in self.measurements:
            raise ValueError("unregistered motion or measurement type")
        motion = self.motions[motion_type](motion_options)
        measurement = self.measurements[measurement_type](measurement_options, motion.state_space)
        if measurement.state_space != motion.state_space:
            raise ValueError("custom measurement builder returned an incompatible StateSpace")
        return motion, measurement


# This function builds an east/north/up tracker that starts from position readings and estimates
# velocity and optional acceleration.
#
# Inputs:
# - axes (mapping of str to str): x/y/z selects east/north/up; CV keeps velocity, CA keeps
#   acceleration constant.
# - origin_lla_deg_m (tuple of three floats): Shared origin: latitude/longitude in degrees, height
#   in metres.
# - order (tuple of str, or None): Tracked quantity order; None groups quantities by axis.
# - noise_density (float or mapping of str to float): Motion uncertainty added per axis; m²/s³ for
#   CV, m²/s⁵ for CA.
# - config (TrackingConfig or None): Filter and management settings, including starting velocity
#   uncertainty; None uses defaults.
#
# Outputs:
# - TrackerEngine: Empty position-reading tracker; starts new tracks with zero assumed velocity
#   and, where used, zero acceleration.
def build_enu_tracker(
    axes: Mapping[str, str],
    *,
    origin_lla_deg_m: tuple[float, float, float],
    order: tuple[str, ...] | None = None,
    noise_density: float | Mapping[str, float] = 1.0,
    config: TrackingConfig | None = None,
) -> TrackerEngine:
    """Build an ENU position-observation tracker for synthetic Cartesian examples.

    Parameters
    ----------
    axes, origin_lla_deg_m, order, noise_density : see CartesianMotion
        CV/CA selection and coordinate order; origin is explicit.
    config : TrackingConfig or None
        Estimator and lifecycle policy.

    Notes
    -----
    Initial position comes from observations. Velocity prior is zero with 100 m/s
    standard deviation (configurable); acceleration prior is zero with 10 m/s²
    standard deviation. Position observations alone do not constrain omitted axes.

    References
    ----------
    .. [1] StoneSoup CV/CA composition and measurement initialization.
    """
    cfg = config or TrackingConfig()
    motion = CartesianMotion(
        axes, origin_lla_deg_m=origin_lla_deg_m, order=order, noise_density=noise_density
    )
    observation = CartesianPosition(motion.state_space, tuple(f"{a}_m" for a in "xyz" if a in axes))
    variance = [
        cfg.initial_velocity_std_mps**2 if c.unit == "m/s" else 100.0 if c.unit == "m/s^2" else 1.0
        for c in motion.state_space.coordinates
    ]
    prior = StateEstimate(
        np.zeros(motion.state_space.dimension), np.diag(variance), 0, motion.state_space
    )
    return build_tracker(motion, observation, prior, cfg)
