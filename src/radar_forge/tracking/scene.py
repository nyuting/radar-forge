r"""A plot-level scene: closed-form trajectories, the plots a surveillance radar makes of them.

Truth is a chain of closed-form :class:`Leg` s per :class:`Arc`: straight with constant
acceleration along track, or a turn at constant rate and speed, evaluated exactly at any time.
A rotating radar's plot model is three steps: :func:`resolve` merges echoes closer than its
resolution cell with probability :math:`P_u(d) = e^{-d^2/2}` [1]; :func:`detect` detects each
group with :math:`1 - \prod_i (1 - P_{D,i})`, an echo inside the MTI notch at a lower Pd;
:func:`measure` places one plot per detected group at its centroid plus noise. False plots are
Poisson in (range, azimuth) at an intensity per (m rad) per scan (:class:`ClutterComponent`),
plus :class:`FixedDiscretes` that report at fixed points. :class:`ClutterMap` estimates the
intensity from the plots. Headings and azimuths are clockwise from north, a turn rate positive
to the right; positions are ENU metres about the radar, ranges horizontal.

References
----------
.. [1] W. Koch and G. van Keuk, "Multiple hypothesis track maintenance with possibly unresolved
       measurements," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 33, no. 3, pp. 883-892, 1997.
.. [2] M. I. Skolnik (ed.), *Radar Handbook*, 3rd ed., McGraw-Hill, 2008.
.. [3] D. F. Crouse, "On implementing 2D rectangular assignment algorithms," *IEEE Trans.
       Aerosp. Electron. Syst.*, vol. 52, no. 4, pp. 1679-1696, 2016.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

__all__ = [
    "LEAST_INTENSITY_PER_M_RAD",
    "Arc",
    "ClutterComponent",
    "ClutterMap",
    "FixedDiscretes",
    "Leg",
    "clutter_intensity",
    "detect",
    "detection_probability",
    "measure",
    "poisson_clutter_plots",
    "polar",
    "resolve",
]


# The least false-plot intensity a point is given, plots per (m rad) per scan: a likelihood
# ratio divides by it, and a point outside every clutter region would otherwise get zero.
LEAST_INTENSITY_PER_M_RAD = 1e-9


def _heading(heading_rad: ArrayLike) -> NDArray[np.float64]:
    """Return the unit ENU vector of a heading clockwise from north, ``(..., 2)``."""
    angle = np.asarray(heading_rad, dtype=np.float64)
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1)


def polar(position_m: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the horizontal range and the azimuth in [0, 2π) of ENU positions ``(..., 2)``."""
    east_m, north_m = np.moveaxis(np.asarray(position_m, dtype=np.float64), -1, 0)
    return np.hypot(east_m, north_m), np.mod(np.arctan2(east_m, north_m), 2 * np.pi)


@dataclass(frozen=True)
class Leg:
    """One piece of a trajectory: straight (accelerating along track, or not) or a turn."""

    duration_s: float
    turn_rate_radps: float = 0.0
    acceleration_mps2: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a leg of no duration, or one that both turns and accelerates."""
        if not (math.isfinite(self.duration_s) and self.duration_s > 0) or (
            self.turn_rate_radps != 0 and self.acceleration_mps2 != 0
        ):
            msg = f"a leg needs a duration > 0 and must not both turn and accelerate; got {self}."
            raise ValueError(msg)


@dataclass(frozen=True, eq=False)
class Arc:
    r"""A closed-form trajectory: a start state, then its legs, then straight on for ever.

    Straight: :math:`p = p_0 + (v_0 \tau + a \tau^2/2) h(\psi_0)`; turning: :math:`\psi =
    \psi_0 + \omega \tau`, :math:`p = p_0 + (v/\omega)(\cos\psi_0 - \cos\psi, \sin\psi -
    \sin\psi_0)`, :math:`h(\psi) = (\sin\psi, \cos\psi)` east and north. Before
    ``start_time_s`` the target does not exist and every value is NaN.

    >>> arc = Arc(0.0, (0.0, 0.0), 0.0, 72.0, (Leg(30.0, math.radians(3.0)),))
    >>> position_m, velocity_mps = arc.state(30.0)
    >>> np.round(position_m, 3), np.round(np.abs(velocity_mps), 6)
    (array([1375.099, 1375.099]), array([72.,  0.]))
    """

    start_time_s: float
    position_m: tuple[float, float]
    heading_rad: float
    speed_mps: float
    legs: tuple[Leg, ...] = ()
    _starts: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Find each leg's start time, position, heading and speed, ``(n_legs + 1, 4)``."""
        values = [self.start_time_s, *self.position_m, self.heading_rad, self.speed_mps]
        if len(self.position_m) != 2 or not np.all(np.isfinite(values)) or self.speed_mps < 0:
            msg = f"an arc needs a finite start, a 2-D position and a speed >= 0; got {self}."
            raise ValueError(msg)
        rows = [[self.start_time_s, *self.position_m, self.heading_rad, self.speed_mps]]
        # A chain: each leg starts where the one before ended.
        for leg in self.legs:
            time_s, east, north, heading, speed = rows[-1]
            end = self._move(
                np.array([leg.duration_s]),
                np.array([[east, north, heading, speed]]),
                np.array([leg.turn_rate_radps]),
                np.array([leg.acceleration_mps2]),
            )
            rows.append([time_s + leg.duration_s, *end[0]])
        object.__setattr__(self, "_starts", np.array(rows))

    @staticmethod
    def _move(
        tau_s: NDArray[np.float64],
        start: NDArray[np.float64],
        turn_radps: NDArray[np.float64],
        acceleration_mps2: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return east, north, heading and speed ``(n, 4)`` after ``tau_s`` on each leg."""
        east, north, heading, speed = start.T
        turning = turn_radps != 0
        rate = np.where(turning, turn_radps, 1.0)
        along = speed * tau_s + acceleration_mps2 * tau_s**2 / 2
        new_heading = heading + turn_radps * tau_s
        straight = np.stack([east, north], -1) + along[:, None] * _heading(heading)
        arc = np.stack(
            [np.cos(heading) - np.cos(new_heading), np.sin(new_heading) - np.sin(heading)], -1
        )
        turned = np.stack([east, north], -1) + (speed / rate)[:, None] * arc
        position = np.where(turning[:, None], turned, straight)
        return np.column_stack([position, new_heading, speed + acceleration_mps2 * tau_s])

    def state(self, time_s: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return position and velocity at each time, ``(..., 2)`` each, NaN before the start."""
        times = np.asarray(time_s, dtype=np.float64)
        flat = times.reshape(-1)
        leg = np.clip(np.searchsorted(self._starts[:, 0], flat, side="right") - 1, 0, None)
        turn = np.array([g.turn_rate_radps for g in self.legs] + [0.0])[leg]
        acceleration = np.array([g.acceleration_mps2 for g in self.legs] + [0.0])[leg]
        moved = self._move(flat - self._starts[leg, 0], self._starts[leg, 1:], turn, acceleration)
        moved[flat < self.start_time_s] = np.nan
        velocity = moved[:, 3:4] * _heading(moved[:, 2])
        return moved[:, :2].reshape(*times.shape, 2), velocity.reshape(*times.shape, 2)


def detection_probability(
    range_rate_mps: ArrayLike,
    detection_probability_outside: ArrayLike,
    *,
    mti_notch_mps: float,
    notch_detection_probability: float,
) -> NDArray[np.float64]:
    """Return each echo's Pd: its own, or the notch's while ``|range_rate| < mti_notch_mps``.

    References
    ----------
    Skolnik [2], ch. 2 (moving-target indication and its blind speeds).
    """
    rate = np.abs(np.asarray(range_rate_mps, dtype=np.float64))
    outside = np.asarray(detection_probability_outside, dtype=np.float64)
    result: NDArray[np.float64] = np.where(
        rate < mti_notch_mps, notch_detection_probability, outside
    )
    return result


def resolve(
    range_m: ArrayLike,
    azimuth_rad: ArrayLike,
    *,
    alpha_range_m: float,
    alpha_azimuth_rad: float,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    r"""Return a group label per echo: pairs merge with :math:`P_u(d) = e^{-d^2/2}`, by union.

    :math:`d^2 = (\Delta r/\alpha_r)^2 + (\Delta az/\alpha_{az})^2`, azimuth the short way
    round. Each pair is drawn once, nearest first, and groups are the connected components of
    the unresolved pairs. Cost O(n²) in the echoes of one scan, a few dozen.

    References
    ----------
    Koch and van Keuk [1] (the Gaussian resolution model).
    """
    ranges = np.asarray(range_m, dtype=np.float64)
    azimuths = np.asarray(azimuth_rad, dtype=np.float64)
    first, second = np.triu_indices(ranges.size, k=1)
    gap_az = np.mod(azimuths[first] - azimuths[second] + np.pi, 2 * np.pi) - np.pi
    distance = np.hypot(
        (ranges[first] - ranges[second]) / alpha_range_m, gap_az / alpha_azimuth_rad
    )
    order = np.argsort(distance, kind="stable")
    merged = np.zeros(distance.size, dtype=bool)
    merged[order] = rng.random(distance.size) < np.exp(-(distance[order] ** 2) / 2)
    graph = coo_array(
        (np.ones(int(merged.sum())), (first[merged], second[merged])), shape=(ranges.size,) * 2
    )
    return np.asarray(connected_components(graph, directed=False)[1], dtype=np.intp)


def detect(
    group: ArrayLike, detection_probability_echo: ArrayLike, rng: np.random.Generator
) -> NDArray[np.bool_]:
    r"""Return, per group label, whether the group is detected: :math:`1 - \prod_i (1 - P_{D,i})`.

    References
    ----------
    Koch and van Keuk [1] (a group of unresolved echoes gives one detection).
    """
    labels = np.asarray(group, dtype=np.intp)
    missed = np.ones(labels.max(initial=-1) + 1)
    np.multiply.at(missed, labels, 1.0 - np.asarray(detection_probability_echo, dtype=np.float64))
    result: NDArray[np.bool_] = rng.random(missed.size) < 1.0 - missed
    return result


def measure(
    range_m: ArrayLike,
    azimuth_rad: ArrayLike,
    group: ArrayLike,
    *,
    sigma_range_m: float,
    sigma_azimuth_rad: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return one plot per group label: its echoes' centroid, plus Gaussian range and azimuth noise.

    References
    ----------
    Koch and van Keuk [1] (the unresolved group's measurement at its centroid).
    """
    labels = np.asarray(group, dtype=np.intp)
    azimuths = np.asarray(azimuth_rad, dtype=np.float64)
    count = np.bincount(labels).astype(np.float64)
    ranges = np.bincount(labels, weights=np.asarray(range_m, dtype=np.float64)) / count
    # Averaged as offsets from each group's first echo, so a group across north stays together.
    reference = azimuths[np.unique(labels, return_index=True)[1]]
    offset = np.mod(azimuths - reference[labels] + np.pi, 2 * np.pi) - np.pi
    centre = reference + np.bincount(labels, weights=offset) / count
    noise = rng.standard_normal((2, count.size))
    measured_az = np.mod(centre + sigma_azimuth_rad * noise[1], 2 * np.pi)
    return ranges + sigma_range_m * noise[0], measured_az


@dataclass(frozen=True)
class ClutterComponent:
    r"""A region of Poisson false plots at ``intensity_per_m_rad`` plots per (m rad) per scan.

    The region is the annulus ``[range_min_m, range_max_m)`` at every azimuth, or, with
    ``centre_enu_m``, a disc of ``radius_m`` drifting at ``drift_enu_mps`` (a rain cell),
    which must lie inside the annulus and not cover the radar. A disc's volume is
    :math:`\int dA / r = 2\pi (c - \sqrt{c^2 - R^2})`, c its centre's range: in (m rad), not
    m², because a CFAR holds the rate per resolution cell, uniform in range and azimuth.
    """

    intensity_per_m_rad: float
    range_min_m: float
    range_max_m: float
    centre_enu_m: tuple[float, float] | None = None
    radius_m: float = 0.0
    drift_enu_mps: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        """Check the intensity and the region."""
        if not (self.intensity_per_m_rad >= 0 and 0 <= self.range_min_m < self.range_max_m):
            msg = f"need intensity >= 0 and 0 <= range_min_m < range_max_m; got {self}."
            raise ValueError(msg)
        if self.centre_enu_m is not None and not self.radius_m > 0:
            msg = f"a disc needs radius_m > 0; got {self.radius_m}."
            raise ValueError(msg)

    def centre_m(self, time_s: float) -> NDArray[np.float64]:
        """Return the disc's centre at ``time_s``, ENU metres."""
        return np.add(self.centre_enu_m or (0.0, 0.0), np.multiply(self.drift_enu_mps, time_s))

    def volume_m_rad(self, time_s: float) -> float:
        """Return the region's volume in the measurement space, m rad."""
        if self.centre_enu_m is None:
            return 2 * np.pi * (self.range_max_m - self.range_min_m)
        centre_range_m = float(np.hypot(*self.centre_m(time_s)))
        inside = self.range_min_m <= centre_range_m - self.radius_m
        if not (inside and centre_range_m + self.radius_m <= self.range_max_m):
            msg = f"the disc at {centre_range_m:.0f} m must lie inside the annulus at {time_s} s."
            raise ValueError(msg)
        return 2 * np.pi * (centre_range_m - math.sqrt(centre_range_m**2 - self.radius_m**2))

    def contains(
        self, range_m: ArrayLike, azimuth_rad: ArrayLike, time_s: float
    ) -> NDArray[np.bool_]:
        """Return whether each (range, azimuth) lies in the region at ``time_s``."""
        ranges = np.asarray(range_m, dtype=np.float64)
        result = (ranges >= self.range_min_m) & (ranges < self.range_max_m)
        if self.centre_enu_m is not None:
            point = ranges[..., None] * _heading(azimuth_rad)
            result &= np.linalg.norm(point - self.centre_m(time_s), axis=-1) < self.radius_m
        return np.asarray(result, dtype=bool)


def poisson_clutter_plots(
    components: Sequence[ClutterComponent], time_s: float, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Return one scan's false plots: range, azimuth and the component each came from.

    Each component draws a Poisson count over a box in (range, azimuth) holding its region,
    uniform in the box, and keeps those inside: thinning a Poisson process leaves one, at the
    region's own volume.
    """
    parts: list[NDArray[np.float64]] = []
    # A loop over the few components, each drawn in one batch.
    for index, component in enumerate(components):
        low_r, high_r, low_az, span_az = (
            component.range_min_m,
            component.range_max_m,
            0.0,
            2 * np.pi,
        )
        if component.centre_enu_m is not None:
            centre_range_m, centre_az = polar(component.centre_m(time_s))
            low_r = float(centre_range_m) - component.radius_m
            high_r = float(centre_range_m) + component.radius_m
            half = math.asin(min(component.radius_m / float(centre_range_m), 1.0))
            low_az, span_az = float(centre_az) - half, 2 * half
        count = rng.poisson(component.intensity_per_m_rad * (high_r - low_r) * span_az)
        draw = rng.random((count, 2))
        ranges, azimuths = (
            low_r + draw[:, 0] * (high_r - low_r),
            np.mod(low_az + draw[:, 1] * span_az, 2 * np.pi),
        )
        kept = component.contains(ranges, azimuths, time_s)
        parts.append(np.stack([ranges[kept], azimuths[kept], np.full(int(kept.sum()), index)]))
    every = np.concatenate([np.zeros((3, 0)), *parts], axis=1)
    return every[0], every[1], every[2].astype(np.intp)


@dataclass(frozen=True, eq=False)
class FixedDiscretes:
    """Point clutter at fixed (range, azimuth), each reported with ``report_probability``."""

    range_m: NDArray[np.float64]
    azimuth_rad: NDArray[np.float64]
    report_probability: float

    @classmethod
    def uniform(
        cls,
        count: int,
        range_min_m: float,
        range_max_m: float,
        report_probability: float,
        seed: int,
    ) -> FixedDiscretes:
        """Return ``count`` discretes uniform in range and azimuth, drawn once from ``seed``."""
        draw = np.random.default_rng(seed).random((2, count))
        ranges = range_min_m + draw[0] * (range_max_m - range_min_m)
        return cls(ranges, 2 * np.pi * draw[1], report_probability)

    def plots(
        self, sigma_range_m: float, sigma_azimuth_rad: float, rng: np.random.Generator
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return this scan's reports: each discrete with ``report_probability``, plus noise."""
        reported = rng.random(self.range_m.size) < self.report_probability
        noise = rng.standard_normal((2, int(reported.sum())))
        range_m = self.range_m[reported] + sigma_range_m * noise[0]
        return range_m, np.mod(self.azimuth_rad[reported] + sigma_azimuth_rad * noise[1], 2 * np.pi)


def clutter_intensity(
    components: Sequence[ClutterComponent],
    range_m: ArrayLike,
    azimuth_rad: ArrayLike,
    time_s: float,
) -> NDArray[np.float64]:
    """Return the false-plot intensity at each point: the sum over the regions holding it."""
    total = np.zeros(np.shape(range_m))
    # A loop over the few components.
    for component in components:
        total += component.intensity_per_m_rad * component.contains(range_m, azimuth_rad, time_s)
    return np.maximum(total, LEAST_INTENSITY_PER_M_RAD)


class ClutterMap:
    """The false-plot intensity per (m rad) per scan, estimated per cell by forgetting average.

    Each scan adds its plots' counts: with f the forgetting factor, the cell's count sum and
    its scan sum both decay by f per scan, and their ratio over the cell's volume is the
    estimate, unbiased from the first scan. A cell never below ``(1 - f)`` of one plot over
    its volume, the least a forgetting average resolves, keeps every likelihood ratio finite.
    ``prior_per_m_rad`` counts as one scan already seen. The caller leaves out plots its
    confirmed tracks took. Azimuth cells wrap.

    References
    ----------
    Skolnik [2], ch. 2 (clutter maps in surveillance radars).
    """

    def __init__(
        self,
        cell_range_m: float = 1e3,
        cell_azimuth_rad: float = math.radians(5.0),
        forgetting: float = 0.9,
        *,
        range_max_m: float = 40e3,
        prior_per_m_rad: float = 0.0,
    ) -> None:
        if not (cell_range_m > 0 and 0 < cell_azimuth_rad <= 2 * np.pi and 0 <= forgetting < 1):
            msg = "need cells > 0 and 0 <= forgetting < 1."
            raise ValueError(msg)
        self.cell_range_m, self.cell_azimuth_rad, self.forgetting = (
            cell_range_m,
            cell_azimuth_rad,
            forgetting,
        )
        self.shape = (
            math.ceil(range_max_m / cell_range_m),
            math.ceil(2 * np.pi / cell_azimuth_rad),
        )
        self.counts = np.full(self.shape, prior_per_m_rad * cell_range_m * cell_azimuth_rad)
        self.n_scans = 1.0 if prior_per_m_rad > 0 else 0.0
        self.floor_per_m_rad = (1 - forgetting) / (cell_range_m * cell_azimuth_rad)

    def _cells(self, range_m: ArrayLike, azimuth_rad: ArrayLike) -> tuple[NDArray[np.intp], ...]:
        """Return each point's (range, azimuth) cell, clipped to the map."""
        rows = np.clip(
            (np.asarray(range_m) // self.cell_range_m).astype(np.intp), 0, self.shape[0] - 1
        )
        columns = (np.mod(azimuth_rad, 2 * np.pi) // self.cell_azimuth_rad).astype(np.intp)
        return rows, np.minimum(columns, self.shape[1] - 1)

    def update(self, range_m: ArrayLike, azimuth_rad: ArrayLike) -> None:
        """Fold one scan's false plots into the estimate."""
        counts = np.zeros(self.shape)
        np.add.at(counts, self._cells(range_m, azimuth_rad), 1.0)
        self.counts = self.forgetting * self.counts + counts
        self.n_scans = self.forgetting * self.n_scans + 1.0

    def intensity(self, range_m: ArrayLike, azimuth_rad: ArrayLike) -> NDArray[np.float64]:
        """Return the estimated intensity at each point, plots per (m rad) per scan."""
        volume = self.cell_range_m * self.cell_azimuth_rad
        estimate = self.counts[self._cells(range_m, azimuth_rad)] / max(self.n_scans, 1.0) / volume
        result: NDArray[np.float64] = np.maximum(estimate, self.floor_per_m_rad)
        return result
