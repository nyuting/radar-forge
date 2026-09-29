"""Measurement-only adapters for the three Duke range/Doppler variants.

References
----------
.. [1] spec/tracker-001-integration.md; Richards, 2014, dual-PRF ambiguity.
.. [2] Norfair detector/tracker separation, https://github.com/tryolabs/norfair.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from radar_forge.core.ambiguity import unfold_doppler_dual_prf
from radar_forge.core.detection_2d import DetectionConfig, cfar_2d
from radar_forge.core.radar import Radar
from radar_forge.pipelines.scenarios import RangeDopplerProduct
from radar_forge.pipelines.tracking_config import TrackingConfig, build_tracker
from radar_forge.tracking import (
    CartesianPosition,
    Measurement,
    MeasurementBatch,
    RadialMotion,
    StateEstimate,
    TrackSnapshot,
)
from radar_forge.tracking._numerics import FloatArray

__all__ = [
    "RadarDetection",
    "ScenarioTracker",
    "TrackingFrame",
    "detect_product",
]


@dataclass(frozen=True)
class RadarDetection:
    """One CFAR peak with SI observations and explicit ambiguity provenance.

    Attributes
    ----------
    timestamp_s : float
        Scan time in seconds.
    leg_index, doppler_bin, range_bin : int
        Source leg and map coordinates.
    range_m, radial_velocity_mps : float
        Measured bin-centre values; potentially folded.
    covariance : numpy.ndarray
        Quantization/floor covariance (2,2), in m², m²/s and m²/s².
    noise_power_linear : float
        Local CFAR noise in input map power units.
    status : str
        accepted, missing_pair, ambiguous_pair or unresolved_velocity.
    pair_id : int or None
        Frame-local dual-PRF pair identity.
    unfolded_velocity_mps, unfolding_residual_mps : float or None
        Unfolding result and residual; raw measured Doppler remains unchanged.

    References
    ----------
    .. [1] Tracker 001 measurement/provenance contract.
    """

    timestamp_s: float
    leg_index: int
    doppler_bin: int
    range_bin: int
    range_m: float
    radial_velocity_mps: float
    covariance: FloatArray
    noise_power_linear: float
    status: str = "accepted"
    pair_id: int | None = None
    unfolded_velocity_mps: float | None = None
    unfolding_residual_mps: float | None = None


@dataclass(frozen=True)
class TrackingFrame:
    """Measurement processing outputs for one timestamp, with no truth fields.

    References
    ----------
    .. [1] Tracker 001 detector/tracker boundary.
    """

    timestamp_s: float
    detections: tuple[RadarDetection, ...]
    snapshots: tuple[TrackSnapshot, ...]
    n_observations: int
    range_period_m: float | None


def detect_product(
    product: RangeDopplerProduct,
    timestamp_s: float,
    leg_index: int,
    config: DetectionConfig,
    *,
    wrap_range: bool,
) -> tuple[RadarDetection, ...]:
    """Convert a labelled map to CFAR peaks without reading simulation truth.

    Parameters
    ----------
    product : RangeDopplerProduct
        Map (n_doppler,n_range) and its physical bin axes.
    timestamp_s : float
        Scan time in seconds.
    leg_index : int
        Source leg identifier.
    config : DetectionConfig
        CFAR and uncertainty settings.
    wrap_range : bool
        True only for a periodic folded range axis.

    References
    ----------
    .. [1] Richards, 2014, CA-CFAR; uniform bin quantization variance Δ²/12.
    """
    shape = (len(product.velocity_axis_mps), len(product.range_axis_m))
    if product.rd_map.shape != shape or min(shape) < 2 or not np.isfinite(timestamp_s):
        raise ValueError(
            "map shape must match finite, uniformly increasing range and velocity axes"
        )
    differences = [np.diff(product.range_axis_m), np.diff(product.velocity_axis_mps)]
    if any(
        not np.all(np.isfinite(d)) or np.any(d <= 0) or not np.allclose(d, d[0])
        for d in differences
    ):
        raise ValueError("map axes must be finite, uniformly increasing")
    range_bin_m, velocity_bin_mps = float(differences[0][0]), float(differences[1][0])
    cov = np.diag(
        [
            max(range_bin_m**2 / 12, config.range_std_floor_m**2),
            max(velocity_bin_mps**2 / 12, config.velocity_std_floor_mps**2),
        ]
    )
    cov.setflags(write=False)
    result = cfar_2d(np.abs(product.rd_map) ** 2, config, wrap_range=wrap_range)
    return tuple(
        RadarDetection(
            timestamp_s,
            leg_index,
            int(row),
            int(col),
            float(product.range_axis_m[col]),
            float(product.velocity_axis_mps[row]),
            cov,
            float(result.noise_power_linear[row, col]),
        )
        for row, col in result.peak_indices
    )


class ScenarioTracker:
    """Compose detection, ambiguity handling and radial tracking without truth access.

    Parameters
    ----------
    legs : tuple of Radar
        One FMCW (S1), one pulsed (S2), or two FMCW (S3) sensor legs.
    detection : DetectionConfig or None
        CFAR settings.
    tracking : TrackingConfig or None
        Estimator, lifecycle and independent ambiguity search prior.

    Notes
    -----
    S2 internal range is a continuous arbitrary alias branch. Only range modulo
    the radar's ambiguity interval is observable. S3 pairing requires each peak
    to have exactly one range-compatible counterpart; multi-target ghosts are
    rejected, not selected using trajectory truth.

    References
    ----------
    .. [1] Richards, 2014, multiple-PRF ambiguity; Tracker 001 variant policy.
    """

    def __init__(
        self,
        legs: tuple[Radar, ...],
        detection: DetectionConfig | None = None,
        tracking: TrackingConfig | None = None,
    ) -> None:
        if len(legs) not in (1, 2) or (
            len(legs) == 2 and any(l.transmitter.waveform != "fmcw" for l in legs)
        ):
            raise ValueError("tracking supports one FMCW/pulsed leg or two FMCW legs")
        self.legs = legs
        self.detection = detection or DetectionConfig()
        self.config = tracking or TrackingConfig()
        self.variant = (
            "S3" if len(legs) == 2 else "S2" if legs[0].transmitter.waveform == "pulsed" else "S1"
        )
        self.range_period_m = legs[0].unambiguous_range_m if self.variant == "S2" else None
        self.motion = RadialMotion(self.config.acceleration_noise_density_m2ps3)
        names = ("range_m",) if self.variant == "S1" else self.motion.state_space.names
        periods = {"range_m": self.range_period_m} if self.range_period_m is not None else None
        self.observation = CartesianPosition(self.motion.state_space, names, periods)
        prior = StateEstimate(
            np.zeros(2, dtype=np.float64),
            np.diag([1.0, self.config.initial_velocity_std_mps**2]),
            0,
            self.motion.state_space,
        )
        self.engine = build_tracker(self.motion, self.observation, prior, self.config)
        if self.variant == "S3":
            a, b = (leg.unambiguous_velocity_mps for leg in legs)
            if a == b or self.config.max_velocity_mps <= min(a, b):
                raise ValueError(
                    "dual-PRF legs require different velocity intervals and a larger search bound"
                )

    def _measurement(
        self, detection: RadarDetection, velocity_mps: float | None = None
    ) -> Measurement:
        if self.variant == "S1":
            value, covariance = np.array([detection.range_m]), detection.covariance[:1, :1]
        else:
            value = np.array(
                [
                    detection.range_m,
                    detection.radial_velocity_mps if velocity_mps is None else velocity_mps,
                ]
            )
            covariance = detection.covariance
        return Measurement(value, covariance, detection.timestamp_s, "sensor", "measurement")

    def _pair(
        self, groups: list[tuple[RadarDetection, ...]], tolerance_mps: float
    ) -> tuple[list[RadarDetection], list[Measurement]]:
        a, b = groups
        records = [replace(d, status="missing_pair") for group in groups for d in group]
        measurements: list[Measurement] = []
        distance = np.abs(
            np.asarray([d.range_m for d in a])[:, None]
            - np.asarray([d.range_m for d in b])[None, :]
        )
        compatible = distance <= max(leg.range_resolution_m for leg in self.legs)
        degrees_a, degrees_b = compatible.sum(axis=1), compatible.sum(axis=0)
        # Pair graph decisions are discrete and preserve source-record identity.
        for i, j in np.argwhere(compatible):
            if degrees_a[i] != 1 or degrees_b[j] != 1:
                records[i] = replace(records[i], status="ambiguous_pair")
                records[len(a) + j] = replace(records[len(a) + j], status="ambiguous_pair")
                continue
            speed, residual = unfold_doppler_dual_prf(
                a[i].radial_velocity_mps,
                b[j].radial_velocity_mps,
                self.legs[0].unambiguous_velocity_mps,
                self.legs[1].unambiguous_velocity_mps,
                max_velocity_mps=self.config.max_velocity_mps,
                tolerance_mps=tolerance_mps,
            )
            valid = bool(np.isfinite(speed))
            updates = dict(
                status="accepted" if valid else "unresolved_velocity",
                pair_id=int(i),
                unfolded_velocity_mps=float(speed) if valid else None,
                unfolding_residual_mps=float(residual) if np.isfinite(residual) else None,
            )
            records[i] = replace(records[i], **updates)
            records[len(a) + j] = replace(records[len(a) + j], **updates)
            if valid:
                measurements.append(self._measurement(a[i], float(speed)))
        return records, measurements

    def process(
        self, timestamp_s: float, products: tuple[RangeDopplerProduct, ...]
    ) -> TrackingFrame:
        """Consume one frame's products and return detections and snapshots.

        Parameters
        ----------
        timestamp_s : float
            Nondecreasing event time in seconds.
        products : tuple of RangeDopplerProduct
            Exactly one labelled map per configured leg.

        References
        ----------
        .. [1] ScenarioTracker class references; truth is deliberately absent here.
        """
        if len(products) != len(self.legs):
            raise ValueError("one range-Doppler product is required per leg")
        if self.engine.last_timestamp_s is not None and timestamp_s < self.engine.last_timestamp_s:
            raise ValueError("out-of-sequence product frame")
        groups = [
            detect_product(p, timestamp_s, i, self.detection, wrap_range=self.variant == "S2")
            for i, p in enumerate(products)
        ]
        if self.variant == "S3":
            tolerance = self.config.unfolding_tolerance_bins * max(
                float(p.velocity_axis_mps[1] - p.velocity_axis_mps[0]) for p in products
            )
            records, observations = self._pair(groups, tolerance)
        else:
            records = list(groups[0])
            observations = [self._measurement(d) for d in records]
        snapshots = self.engine.process(
            MeasurementBatch(timestamp_s, "sensor", tuple(observations))
        )
        return TrackingFrame(
            timestamp_s, tuple(records), snapshots, len(observations), self.range_period_m
        )
