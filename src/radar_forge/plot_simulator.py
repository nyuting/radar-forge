r"""A plot-level simulator: the plots a rotating surveillance radar makes of a scene, scan by scan.

:func:`plot_scans` plays a scenario with a ``[sensor]`` table (scenario 011) one scan at a time:
the beam crosses each target's :class:`~radar_forge.trajectory.Arc` at its own time, and a plot
model of three steps follows. :func:`resolve` merges echoes closer than the resolution cell
with probability :math:`P_u(d) = e^{-d^2/2}` [1]; :func:`detect` detects each group with
:math:`1 - \prod_i (1 - P_{D,i})`, an echo inside the MTI notch at a lower Pd; :func:`measure`
places one plot per detected group at its centroid plus noise. False plots are Poisson in
(range, azimuth) at a density per (m rad) per scan (:class:`ClutterComponent`), plus
:class:`FixedDiscretes` that report at fixed points; each plot carries the clutter density at
it (:func:`clutter_density`), the oracle a tracker may read. Azimuths are clockwise from north;
positions are ENU metres about the radar, ranges horizontal. Nothing here tracks, halves a
range or does I/O.

References
----------
.. [1] W. Koch and G. van Keuk, "Multiple hypothesis track maintenance with possibly unresolved
       measurements," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 33, no. 3, pp. 883-892, 1997.
.. [2] M. I. Skolnik (ed.), *Radar Handbook*, 3rd ed., McGraw-Hill, 2008.
.. [3] M. Abramowitz and I. A. Stegun (eds.), *Handbook of Mathematical Functions*, National
       Bureau of Standards, 1964.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from scipy.special import ellipe, ellipk

from radar_forge.coordinates import (
    east_north_to_range_azimuth_rad,
    range_azimuth_to_east_north_m,
)

if TYPE_CHECKING:
    # A type only: scenario.py builds this module's ClutterComponent and FixedDiscretes, so a
    # run-time import would be a cycle.
    from radar_forge.scenario import Scenario

__all__ = [
    "LEAST_DENSITY_PER_M_RAD",
    "ClutterComponent",
    "FixedDiscretes",
    "PlotScan",
    "clutter_density",
    "detect",
    "detection_probability",
    "measure",
    "plot_scans",
    "poisson_clutter_plots",
    "resolve",
    "sector_index",
]


# The least false-plot density a point is given, plots per (m rad) per scan: a likelihood
# ratio divides by it, and a point outside every clutter region would otherwise get zero.
LEAST_DENSITY_PER_M_RAD = 1e-9


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
    # Unguarded: a sum that rounds onto the edge gives +pi, where _periodic.fold gives -pi.
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
    # Unguarded: a sum that rounds onto the edge gives +pi, where _periodic.fold gives -pi.
    offset = np.mod(azimuths - reference[labels] + np.pi, 2 * np.pi) - np.pi
    centre = reference + np.bincount(labels, weights=offset) / count
    noise = rng.standard_normal((2, count.size))
    # Unguarded: a tiny negative angle rounds up to exactly 2 pi, where _periodic.fold gives 0.
    measured_az = np.mod(centre + sigma_azimuth_rad * noise[1], 2 * np.pi)
    return ranges + sigma_range_m * noise[0], measured_az


@dataclass(frozen=True)
class ClutterComponent:
    r"""A region of Poisson false plots at ``density_per_m_rad`` plots per (m rad) per scan.

    The region is the annulus ``[range_min_m, range_max_m)`` at every azimuth, or, with
    ``centre_enu_m``, a disc of ``radius_m`` drifting at ``drift_enu_mps`` (a rain cell),
    which must lie inside the annulus and not cover the radar. The density is per (m rad),
    not m², because a CFAR holds the rate per resolution cell, uniform in range and azimuth;
    so a disc's volume is :math:`\int dA / r = \int_{-\phi_0}^{\phi_0} 2\sqrt{R^2 - c^2
    \sin^2\phi}\, d\phi`, its chords along each azimuth, :math:`\sin\phi_0 = R/c`, which is
    :math:`4 (c E(m) - (c^2 - R^2) K(m) / c)` with :math:`m = R^2/c^2`, c the centre's range.

    References
    ----------
    Abramowitz and Stegun [3], §17.3 (the complete elliptic integrals K and E).
    """

    density_per_m_rad: float
    range_min_m: float
    range_max_m: float
    centre_enu_m: tuple[float, float] | None = None
    radius_m: float = 0.0
    drift_enu_mps: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        """Check the density and the region."""
        if not (self.density_per_m_rad >= 0 and 0 <= self.range_min_m < self.range_max_m):
            msg = f"need density >= 0 and 0 <= range_min_m < range_max_m; got {self}."
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
        m = (self.radius_m / centre_range_m) ** 2
        chord = (
            centre_range_m * ellipe(m)
            - (centre_range_m**2 - self.radius_m**2) * ellipk(m) / centre_range_m
        )
        return float(4 * chord)

    def contains(
        self, range_m: ArrayLike, azimuth_rad: ArrayLike, time_s: float
    ) -> NDArray[np.bool_]:
        """Return whether each (range, azimuth) lies in the region at ``time_s``."""
        ranges = np.asarray(range_m, dtype=np.float64)
        result = (ranges >= self.range_min_m) & (ranges < self.range_max_m)
        if self.centre_enu_m is not None:
            point = range_azimuth_to_east_north_m(ranges, azimuth_rad)
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
            centre_range_m, centre_az = east_north_to_range_azimuth_rad(component.centre_m(time_s))
            low_r = float(centre_range_m) - component.radius_m
            high_r = float(centre_range_m) + component.radius_m
            half = math.asin(min(component.radius_m / float(centre_range_m), 1.0))
            low_az, span_az = float(centre_az) - half, 2 * half
        count = rng.poisson(component.density_per_m_rad * (high_r - low_r) * span_az)
        draw = rng.random((count, 2))
        # Unguarded: a tiny negative angle rounds up to exactly 2 pi, where _periodic.fold gives 0.
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
        # Unguarded: a tiny negative angle rounds up to exactly 2 pi, where _periodic.fold gives 0.
        return range_m, np.mod(self.azimuth_rad[reported] + sigma_azimuth_rad * noise[1], 2 * np.pi)


def clutter_density(
    components: Sequence[ClutterComponent],
    range_m: ArrayLike,
    azimuth_rad: ArrayLike,
    time_s: float,
) -> NDArray[np.float64]:
    """Return the false-plot density at each point: the sum over the regions holding it."""
    total = np.zeros(np.shape(range_m))
    # A loop over the few components.
    for component in components:
        total += component.density_per_m_rad * component.contains(range_m, azimuth_rad, time_s)
    return np.maximum(total, LEAST_DENSITY_PER_M_RAD)


# The fixed-point iterations that find when the beam crosses a moving target: its azimuth moves
# under a degree a scan, so each iteration shrinks the time error a hundredfold.
_BEAM_ITERATIONS = 3


@dataclass(frozen=True, eq=False)
class PlotScan:
    """One scan of a plot-level scenario: its plots, and the truth behind them.

    Plots ``(n_plots,)``: the time the beam crossed each, horizontal range, azimuth, the
    sector it is batched in, its ``source`` (``target``, ``merged``, ``clutter``,
    ``discrete``) and ``source_ids`` (truth names joined by ``;``), and the clutter
    density there per (m rad), the oracle. Echoes ``(n_targets,)``: when the beam crossed
    each target, whether it was in coverage, in the MTI notch, and detected (alone or merged).
    The tracker never reads ``source`` or the echoes.
    """

    index: int
    start_time_s: float
    time_s: NDArray[np.float64]
    range_m: NDArray[np.float64]
    azimuth_rad: NDArray[np.float64]
    sector: NDArray[np.int64]
    source: tuple[str, ...]
    source_ids: tuple[str, ...]
    clutter_density_per_m_rad: NDArray[np.float64]
    echo_time_s: NDArray[np.float64]
    echo_in_coverage: NDArray[np.bool_]
    echo_in_notch: NDArray[np.bool_]
    echo_detected: NDArray[np.bool_]


def sector_index(azimuth_rad: ArrayLike, n_sectors: int) -> NDArray[np.int64]:
    """Return the sector each azimuth is batched in, ``[0, n_sectors)``; 2π itself in the last."""
    sector_rad = 2 * np.pi / n_sectors
    result: NDArray[np.int64] = np.minimum(
        np.asarray(azimuth_rad, dtype=np.float64) // sector_rad, n_sectors - 1
    ).astype(np.int64)
    return result


def _beam_crossing(
    scenario: Scenario, start_s: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return when the beam crosses each target in a scan, and its position and velocity."""
    period_s = scenario.scan_period_s
    time_s = np.full(len(scenario.arcs), start_s)
    # A short fixed-point iteration per scan: time from azimuth, azimuth from time.
    for _ in range(_BEAM_ITERATIONS):
        states = [target.arc.state(t) for target, t in zip(scenario.arcs, time_s, strict=True)]
        position_m = np.array([p for p, _ in states])
        # A target not yet flying is NaN; its azimuth counts as 0, the scan's start.
        azimuth = np.nan_to_num(east_north_to_range_azimuth_rad(position_m)[1])
        time_s = start_s + azimuth / (2 * np.pi) * period_s
    return time_s, position_m, np.array([v for _, v in states])


def plot_scans(
    scenario: Scenario, *, n_scans: int | None = None, rng: np.random.Generator | None = None
) -> Iterator[PlotScan]:
    """Play a plot-level scenario back one scan at a time, with the truth behind each plot.

    Each target is seen when the beam crosses it; echoes merge, are detected and measured by
    the plot model above; clutter and fixed discretes are added.
    One generator (default: the scenario's seed) draws everything, so a run replays.
    """
    sensor = scenario.sensor
    if sensor is None:
        msg = f"scenario {scenario.name!r} has no [sensor]: it is not plot-level."
        raise ValueError(msg)
    rng = np.random.default_rng(scenario.seed) if rng is None else rng
    names = np.array([target.name for target in scenario.arcs])
    pd = np.array(
        [
            getattr(sensor, f"detection_probability_{target.target_class}")
            for target in scenario.arcs
        ]
    )
    sigma_az = math.radians(sensor.sigma_azimuth_deg)
    count = scenario.n_scans if n_scans is None else min(n_scans, scenario.n_scans)
    # A loop over scans: each is drawn and handed out before the next.
    for index in range(count):
        start_s = scenario.start_time_s + index * scenario.scan_period_s
        echo_s, position_m, velocity_mps = _beam_crossing(scenario, start_s)
        ranges, azimuths = east_north_to_range_azimuth_rad(position_m)
        rate_mps = np.sum(position_m * velocity_mps, axis=-1) / ranges
        seen = np.isfinite(ranges) & (ranges >= sensor.range_min_m) & (ranges < sensor.range_max_m)
        notch = np.abs(rate_mps) < sensor.mti_notch_mps
        rows = np.flatnonzero(seen)
        group = resolve(
            ranges[rows],
            azimuths[rows],
            alpha_range_m=sensor.range_cell_m,
            alpha_azimuth_rad=math.radians(sensor.beamwidth_azimuth_deg),
            rng=rng,
        )
        echo_pd = detection_probability(
            rate_mps[rows],
            pd[rows],
            mti_notch_mps=sensor.mti_notch_mps,
            notch_detection_probability=sensor.detection_probability_notch,
        )
        found = detect(group, echo_pd, rng)
        plot_r, plot_az = measure(
            ranges[rows],
            azimuths[rows],
            group,
            sigma_range_m=sensor.sigma_range_m,
            sigma_azimuth_rad=sigma_az,
            rng=rng,
        )
        members = [names[rows][group == g] for g in range(found.size)]
        detected = np.zeros(len(names), dtype=bool)
        detected[rows] = found[group]
        clutter_r, clutter_az, _ = poisson_clutter_plots(
            scenario.clutter_plots.components, start_s, rng
        )
        discretes = scenario.clutter_plots.discretes or FixedDiscretes(
            np.zeros(0), np.zeros(0), 0.0
        )
        fixed_r, fixed_az = discretes.plots(sensor.sigma_range_m, sigma_az, rng)
        range_m = np.concatenate([plot_r[found], clutter_r, fixed_r])
        azimuth_rad = np.concatenate([plot_az[found], clutter_az, fixed_az])
        kinds = [("merged" if len(m) > 1 else "target", ";".join(m)) for m in members]
        source = [k for k, keep in zip(kinds, found, strict=True) if keep]
        source += [("clutter", "")] * clutter_r.size + [("discrete", "")] * fixed_r.size
        yield PlotScan(
            index=index,
            start_time_s=start_s,
            time_s=start_s + azimuth_rad / (2 * np.pi) * scenario.scan_period_s,
            range_m=range_m,
            azimuth_rad=azimuth_rad,
            sector=sector_index(azimuth_rad, sensor.n_sectors),
            source=tuple(s for s, _ in source),
            source_ids=tuple(i for _, i in source),
            clutter_density_per_m_rad=clutter_density(
                scenario.clutter_plots.components, range_m, azimuth_rad, start_s
            ),
            echo_time_s=echo_s,
            echo_in_coverage=seen,
            echo_in_notch=notch,
            echo_detected=detected,
        )
