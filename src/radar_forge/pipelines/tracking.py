r"""Turn range-Doppler maps into tracks: detect, unfold, associate, filter.

This module is the pipeline half of
``spec/scenario-003-tracking.md``. It owns everything between a
:class:`~radar_forge.pipelines.scenarios.RangeDopplerProduct` and a call to one
of the two trackers in :mod:`radar_forge.core.tracking`, chosen by
``TrackingConfig.estimator``:

* ``"kalman"``: :class:`~radar_forge.core.tracking.KalmanTracker`, a linear
  Kalman filter with track-aided Doppler unfolding. Scenario 003's own runs use
  it.
* ``"ukf"``: :class:`~radar_forge.core.tracking.Tracker`, an unscented Kalman
  filter that measures range modulo the span of the burst's range axis
  (:func:`~radar_forge.pipelines.scenarios.range_axis_m`). The runs of scenario
  001's three waveforms (S1, S2, S3) use it.

Between the map and the tracker this module does:

* CFAR detection and the conversion of fractional cell indices into metres and
  metres per second (§5.1), using the bin-centre helpers of
  :mod:`radar_forge.core.dsp` rather than a locally re-derived bin spacing;
* Doppler **unfolding**, either by pairing two bursts at coprime PRFs or, on
  the ``"kalman"`` path, with the bootstrap a brand-new track needs before it
  can unfold anything (§5.3);
* the **simulated angle measurement** the ENU state models need until an array
  exists (§6.3).

:mod:`radar_forge.core.tracking` knows about none of this, and that separation
is deliberate: the tracker's contract is "you give me measurements and a
model", and where a measurement came from is not its business.

Two departures from the specification as written, both recorded in its §14.2 and
§14.4 and both found by running it:

**The Doppler axis is circular and clustering is not.**
:func:`~radar_forge.core.detection.cluster_detections` labels with a
non-circular structure, so a target whose response straddles the
:math:`\pm v_{\text{unamb}}` wrap is returned as *two* clusters at opposite
ends of the axis, and a centroid over the pair is meaningless. In scenario
001's S1 at frame 0 the split is 58 cells at +7.493 m/s and 37 cells at
-7.511 m/s, against a true folded velocity of +7.502 m/s.
:func:`frame_detections` rolls the map so the array edge falls on the quietest
Doppler row before clustering, and maps the indices back afterwards. The clean
fix is a ``wrap_axes`` option on ``cluster_detections`` itself, which belongs to
that module's own workstream (§10.1), not to this one.

**A track's readiness to unfold is a batch statistic, not the filter's
covariance.** §5.3 gives the ten-frame crossing in terms of the standard
deviation of a least-squares range slope, and that is the estimator used here.
The filter's own rate variance cannot serve: discrete white-noise acceleration
adds :math:`\sigma_a^2 T^2` every frame, so at ``sigma_accel_mps2 = 2.0`` the
rate standard deviation floors near 4.1 m/s and never reaches the 2.55 m/s
threshold. See ``tests/core/test_tracking.py`` for the pinned floor.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §6.5 (CFAR), §7.1 and §7.2 (estimators and their
       accuracy: the CRLB, range and Doppler estimators).
.. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with
       Applications to Tracking and Navigation*, Wiley, 2001, §5.5
       (initialisation of state estimators).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Literal, cast, get_args, get_type_hints

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radar_forge.core.ambiguity import unfold_doppler_dual_prf
from radar_forge.core.detection import (
    CFAR_VARIANTS,
    CfarVariant,
    cfar_detect,
    cluster_detections,
)
from radar_forge.core.tracking import (
    CartesianPosition,
    Coordinate,
    KalmanFilter,
    KalmanTracker,
    LifecyclePolicy,
    RadialMotion,
    StateEstimate,
    StateLayout,
    Track,
    Tracker,
    TrackStatus,
    build_tracker,
    state_model_matrices,
)
from radar_forge.pipelines.scenarios import range_axis_m

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from radar_forge.core.radar import RadarLike
    from radar_forge.pipelines.scenarios import RangeDopplerProduct, Scenario

__all__ = [
    "DETECTION_STATUSES",
    "ESTIMATORS",
    "MEASUREMENT_MODEL_ID",
    "SENSOR_ID",
    "STATE_FIELDS",
    "UNFOLDING_MODES",
    "DetectionConfig",
    "DetectionStatus",
    "EstimatorKind",
    "FrameTracks",
    "Measurement",
    "MetricRow",
    "ScenarioTracker",
    "TrackRecord",
    "TrackingConfig",
    "UnfoldingMode",
    "bin_quantisation_sigmas",
    "configs_from_scenario",
    "dual_prf_detections",
    "dual_prf_measurements",
    "frame_detections",
    "minimum_unfold_history_frames",
    "primary_track_id",
    "range_layout",
    "range_slope_sigma_mps",
    "replace_measurement",
    "score_primary_track",
    "slope_velocity_mps",
    "unfold_velocity_mps",
]

UnfoldingMode = Literal["track_aided", "oracle", "none"]
UNFOLDING_MODES: tuple[UnfoldingMode, ...] = get_args(UnfoldingMode)

EstimatorKind = Literal["kalman", "ukf"]
"""Which tracker :class:`ScenarioTracker` runs. See the module docstring."""
ESTIMATORS: tuple[EstimatorKind, ...] = get_args(EstimatorKind)

DetectionStatus = Literal["accepted", "missing_pair", "ambiguous_pair", "unresolved_velocity"]
"""What became of one detection on its way to the tracker.

``"accepted"``
    Passed to the tracker. With one burst, every detection is accepted.
``"missing_pair"``
    Dual-PRF only: no detection in the other burst lies within the range
    tolerance, so the velocity cannot be unfolded.
``"ambiguous_pair"``
    Dual-PRF only: a detection in the other burst is within the range
    tolerance, but the two are not each other's nearest, so pairing them would
    be a guess, and it is not made.
``"unresolved_velocity"``
    Dual-PRF only: the pair is one-to-one, but no velocity in the search span
    agrees with both folded values.
"""
DETECTION_STATUSES: tuple[DetectionStatus, ...] = get_args(DetectionStatus)

SENSOR_ID = "rx0"
"""The receiver's sensor ID, in the UKF tracker and in ``detections.csv``.

``spec/data-001-formats.md`` §6.5 names today's single receiver ``"rx0"``.
"""

MEASUREMENT_MODEL_ID = "range_doppler"
"""The ID under which the UKF tracker registers its one measurement model."""

STATE_FIELDS: tuple[str, ...] = ("range_m", "range_rate_mps")
"""The state vector of the ``range_1d`` state model, in order.

Both trackers use it, and ``metadata.json`` records it as
``tracking.state_fields`` (``spec/data-001-formats.md`` §6.1).
"""


@dataclass(frozen=True)
class Measurement:
    r"""One CFAR detection, converted to physical units.

    Attributes
    ----------
    range_m : float
        Slant range at the cluster's power-weighted centroid, in metres.
    velocity_folded_mps : float
        Range rate as the range-Doppler map shows it, inside
        :math:`\pm v_{\text{unamb}}`, positive closing.
    velocity_unfolded_mps : float or None
        The true range rate, once a track has selected the fold. ``None`` while
        the measurement is still range-only, which is what the §5.3 bootstrap
        leaves a new track with.
    fold_index : int or None
        The integer :math:`k` such that
        ``velocity_unfolded = velocity_folded + k * fold_span``. ``None``
        alongside ``velocity_unfolded_mps``.
    peak_power_w, total_power_w : float
        Cluster power, in watts.
    n_cells : int
        Number of cells that crossed the threshold.
    range_index, velocity_index : float
        The fractional cell indices the physical values were interpolated from,
        kept for plotting a marker back onto the map.
    burst_index : int, default 0
        Which burst's map the detection was found in.
    status : DetectionStatus, default "accepted"
        What became of the detection; see :data:`DetectionStatus`.
    pair_id : int or None, default None
        Dual-PRF only: the same integer on the two detections that were paired,
        unique within the frame. ``None`` for a detection that was not paired.
    """

    range_m: float
    velocity_folded_mps: float
    peak_power_w: float
    total_power_w: float
    n_cells: int
    range_index: float
    velocity_index: float
    velocity_unfolded_mps: float | None = None
    fold_index: int | None = None
    burst_index: int = 0
    status: DetectionStatus = "accepted"
    pair_id: int | None = None


@dataclass(frozen=True)
class DetectionConfig:
    """The CFAR settings of scenario 003 §2.

    Attributes
    ----------
    pfa : float
        Design probability of false alarm per cell. At 1e-5 over the 245,760
        cells of S1's map that carry a complete reference window, roughly 2.46
        false alarms reach the associator every frame -- enough that gating and
        assignment are exercised continuously, while a spurious *confirmed*
        track stays a once-in-four-hundred-runs event.
    n_train, n_guard : int
        Training and guard cells per side.
    variant : str
        Which CFAR family, defaulting to cell averaging.
    merge_range_bins : float
        Detections within this many range bins of a stronger one are discarded
        as its sidelobes, default 1.0. See :func:`frame_detections`. Set to 0 to
        keep every cluster.
    """

    pfa: float = 1e-5
    n_train: int = 16
    n_guard: int = 4
    variant: CfarVariant = "ca"
    merge_range_bins: float = 1.0


@dataclass(frozen=True)
class TrackingConfig:
    r"""The filter, gate and track-management settings of scenario 003 §2.

    Two values depart from §2's table, both because the scenario was run and
    measured rather than assumed:

    ``sigma_accel_mps2`` is 5.0 rather than §6.2's 2.0. §6.2 picks 2.0 by
    judgement, for the lateral acceleration of a light aircraft in a circuit.
    But the target is synthesised from ``data/flight_coordinates.csv``, whose
    fixes are 2-3 s apart and carry ADS-B position noise, so the *simulated*
    target's range rate moves by 3.4 m/s from one frame to the next (standard
    deviation over the default window) and by as much as 11 m/s. The filter has
    to track the target as simulated, not as flown.

    **The comparison that chose 5.0 no longer separates the two.** §14.6 quoted
    98% of frames tracked at 5.0 against 93% at 2.0, and this docstring quoted
    95% against 93%; re-measured over the current 120-frame window, both
    settings hold a confirmed track in **117 of 120 frames under 2 track ids**,
    and 2.0 selects the fold slightly *better* -- 94% of associated frames
    against 90%. The figures were taken before §14.10 moved the window to 663 s
    and before §14.7 added re-acquisition, which holds a confirmed track through
    a mis-unfold and is what makes track retention insensitive to this setting
    here. 5.0 is kept as the shipped default because nothing measured argues for
    moving it, not because the quoted margin still exists; §14.6 records the
    re-measurement.

    ``sigma_azimuth_deg`` and ``sigma_elevation_deg`` belong to the ``enu_2d``
    and ``enu_3d`` state models of §6.3 and are ignored by ``range_1d``. Unlike
    the range and velocity sigmas they are not quantisation-limited, because
    this radar has no angle bin to quantise into: the angle is *synthesised*
    from truth by the pipeline, and 0.5° is §6.3's monopulse figure for the
    5.1° beam implied by 30 dBi.

    ``n_slope_frames`` is the trailing range window the fold-consistency monitor
    fits; see :meth:`ScenarioTracker.step`. Five frames measured better than
    eight or ten, because a longer fit averages down the range noise but lags a
    target whose velocity is genuinely moving.

    **The two estimators read some fields differently.** ``estimator`` picks
    the tracker (:data:`EstimatorKind`).

    - ``sigma_accel_mps2`` is the standard deviation of the target's
      acceleration for both, but the two process-noise models differ. The
      ``"kalman"`` path uses discrete white-noise acceleration, a constant
      acceleration of that size within each frame
      (:func:`~radar_forge.core.tracking.process_noise_dwna`). The ``"ukf"``
      path uses :class:`~radar_forge.core.tracking.RadialMotion`, which adds
      ``acceleration_correlation_time_s``: continuous white noise of density
      :math:`q = 2\sigma_a^2\tau`, the white-noise limit of Singer's model.
    - ``sigma_range_m`` and ``sigma_velocity_mps`` are read by the
      ``"kalman"`` path only. The ``"ukf"`` path computes its measurement
      noise from each frame's own map, one bin over :math:`\sqrt{12}`
      (:func:`bin_quantisation_sigmas`), so it needs no number fixed for one
      waveform.
    - ``v_max_mps`` bounds the target's speed for both. It sets the prior
      standard deviation of a new track's range rate. On the ``"ukf"`` path it
      also decides whether velocity is measured at all: a single burst whose
      unambiguous velocity is below ``v_max_mps`` may fold the Doppler, so the
      tracker measures range alone. With two bursts it is also the span the
      dual-PRF pair is asked to resolve over (:func:`dual_prf_detections`), so
      it must exceed the smaller burst's unambiguous velocity.
    - ``unfolding_mode`` must be ``"none"`` on the ``"ukf"`` path, which has no
      track-aided unfolding: it unfolds by dual PRF or not at all.
    - ``state_model`` must be ``"range_1d"`` on the ``"ukf"`` path.
    """

    sigma_range_m: float = 21.635652855125496
    sigma_velocity_mps: float = 0.017247813329811023
    sigma_accel_mps2: float = 5.0
    acceleration_correlation_time_s: float = 1.0
    sigma_azimuth_deg: float = 0.5
    sigma_elevation_deg: float = 0.5
    gate_probability: float = 0.99
    n_confirm_hits: int = 4
    n_confirm_frames: int = 5
    n_delete_misses: int = 3
    n_reacquire_frames: int = 5
    v_max_mps: float = 200.0
    unfolding_mode: UnfoldingMode = "track_aided"
    unfold_sigma_gate: float = 6.0
    n_slope_frames: int = 5
    state_model: str = "range_1d"
    estimator: EstimatorKind = "kalman"

    def unused_fields(self) -> tuple[str, ...]:
        """Return the names of the fields that :attr:`estimator`'s path does not read.

        A run records every setting, defaults included, so a reader needs to
        be told which of them had no effect. Otherwise a ``"ukf"`` run's
        ``sigma_range_m``, a value frozen for S1's bins, reads as the
        measurement noise it used.

        Returns
        -------
        tuple of str
            Field names, in declaration order. ``sigma_azimuth_deg`` and
            ``sigma_elevation_deg`` are always among them: only the ``enu_2d``
            and ``enu_3d`` state models would read them, and neither path
            builds those yet.
        """
        # Kept beside the fields, so that a field added to one path is added
        # here in the same change. ScenarioTracker is the only reader.
        unused = {"sigma_azimuth_deg", "sigma_elevation_deg"}
        if self.estimator == "ukf":
            unused |= {"sigma_range_m", "sigma_velocity_mps", "unfold_sigma_gate", "n_slope_frames"}
        else:
            unused |= {"acceleration_correlation_time_s"}
        return tuple(f.name for f in fields(self) if f.name in unused)


def _suppress_range_sidelobes(
    measurements: Sequence[Measurement],
    merge_range_bins: float,
    n_range_bins: int | None = None,
) -> list[Measurement]:
    """Keep the strongest detection in each range cell and discard the rest.

    ``measurements`` must arrive strongest first, which is the order
    ``cluster_detections`` returns. With ``n_range_bins``, the range axis is
    circular and distances are taken the short way round it.
    """
    if merge_range_bins <= 0.0:
        return list(measurements)

    def gap_bins(a: Measurement, b: Measurement) -> float:
        gap = abs(a.range_index - b.range_index)
        return gap if n_range_bins is None else min(gap, n_range_bins - gap)

    kept: list[Measurement] = []
    # Each detection is kept or dropped against the ones already kept, so the
    # order matters and the loop cannot be vectorised.
    for measurement in measurements:
        if all(gap_bins(measurement, other) > merge_range_bins for other in kept):
            kept.append(measurement)
    return kept


def _quietest_doppler_row(power_w: NDArray[np.float64]) -> int:
    """Index of the Doppler row carrying the least total power.

    The wrap has to be moved somewhere before clustering, and the quietest row
    is the place a real target is least likely to be sitting.
    """
    return int(np.argmin(power_w.sum(axis=1)))


def frame_detections(
    product: RangeDopplerProduct,
    config: DetectionConfig | None = None,
    *,
    wrap_range: bool = False,
) -> list[Measurement]:
    r"""Detect targets in one range-Doppler map and return them in physical units.

    Runs cell-averaging CFAR along the **range** axis, clusters the crossings,
    and interpolates each cluster's power-weighted centroid onto the product's
    own range and velocity axes. Interpolating onto those axes rather than
    deriving a bin spacing locally is what keeps the unshifted-range /
    fftshifted-Doppler asymmetry in one place; a tracker that reinvents the
    Doppler axis produces closing targets that open, and the picture still
    looks plausible.

    Parameters
    ----------
    product : RangeDopplerProduct
        The frame's map and its two axes. ``rd_map`` is complex, so the power
        map CFAR sees is :math:`|x|^2`.
    config : DetectionConfig, optional
        CFAR settings; the scenario's defaults if omitted.
    wrap_range : bool, optional
        Treat the range axis as circular, as the map's range axis is: a return
        beyond its last bin lands back near its first (see
        :func:`~radar_forge.pipelines.scenarios.range_axis_m`). Then the CFAR
        window wraps round the ends instead of leaving the end cells untested,
        a cluster may straddle the ends, and a range is
        ``range_axis_m[0] + range_index * bin`` for any index in
        ``[0, n_range_bins)``. Default False, which tests no cell within
        ``n_train + n_guard`` of either end.

    Returns
    -------
    list of Measurement
        One per cluster, strongest first, with folded velocity only. Unfolding
        is a later step and needs a track.

    Notes
    -----
    The map is rolled along the Doppler axis so that its array edge falls on the
    quietest Doppler row, clustered there, and the indices mapped back. Without
    this a target straddling the velocity wrap is split into two clusters at
    opposite ends of the axis and neither centroid is the target's velocity.
    The Doppler axis is circular; ``cluster_detections`` is not, and teaching it
    to be belongs to ``core/detection.py``'s workstream, not to this one.

    Detections within ``merge_range_bins`` of a stronger one are then discarded.
    Scenario 001 applies no Doppler taper, so a target at the 45-69 dB
    post-integration SNR of §4 puts its *sidelobes* tens of decibels above an
    11.4 dB threshold: measured over the first twelve frames, the target yields
    one cluster of 30-95 cells at its true bin plus one-cell satellites at the
    same range and 10-28 Doppler bins away, while genuine false alarms land
    16-750 range bins off. One point target cannot produce two returns at one
    range, so keeping only the strongest per range cell removes the sidelobes
    and leaves the false-alarm statistics of §3 alone.

    This is a **single-target** simplification, and it is the same gap §13.2 and
    §13.3 name: with two aircraft in one range cell it would discard a real
    detection, and the right answer there is a two-dimensional CFAR and a
    tapered Doppler response, not a wider merge.
    """
    settings = config if config is not None else DetectionConfig()
    power_w = np.abs(product.rd_map) ** 2
    n_doppler_bins, n_range_bins = power_w.shape
    # With a circular range axis, pad each end with the other end's cells, so
    # every cell has a whole reference window, then cut the padding off again.
    margin = settings.n_train + settings.n_guard if wrap_range else 0
    padded_w = np.pad(power_w, ((0, 0), (margin, margin)), mode="wrap")
    mask = cfar_detect(
        padded_w,
        pfa=settings.pfa,
        n_train=settings.n_train,
        n_guard=settings.n_guard,
        variant=settings.variant,
        axis=-1,
    )[:, margin : margin + n_range_bins]

    shift = _quietest_doppler_row(power_w)
    rolled_power_w = np.roll(power_w, -shift, axis=0)
    rolled_mask = np.roll(mask, -shift, axis=0)

    range_indices = np.arange(n_range_bins, dtype=np.float64)
    velocity_indices = np.arange(n_doppler_bins, dtype=np.float64)
    range_bin_m = float(product.range_axis_m[1] - product.range_axis_m[0])

    measurements: list[Measurement] = []
    clusters = cluster_detections(rolled_mask, rolled_power_w, wrap_axes=(1,) if wrap_range else ())
    for detection in clusters:
        velocity_index = float((detection.centroid_index[0] + shift) % n_doppler_bins)
        range_index = float(detection.centroid_index[1])
        # On a circular axis a centroid can lie past the last bin centre, where
        # np.interp would clamp; the uniform axis's own formula does not.
        range_m = (
            float(product.range_axis_m[0]) + range_index * range_bin_m
            if wrap_range
            else float(np.interp(range_index, range_indices, product.range_axis_m))
        )
        measurements.append(
            Measurement(
                range_m=range_m,
                velocity_folded_mps=float(
                    np.interp(velocity_index, velocity_indices, product.velocity_axis_mps)
                ),
                peak_power_w=detection.peak_power_w,
                total_power_w=detection.total_power_w,
                n_cells=detection.n_cells,
                range_index=range_index,
                velocity_index=velocity_index,
            )
        )
    return _suppress_range_sidelobes(
        measurements, settings.merge_range_bins, n_range_bins if wrap_range else None
    )


def _matches_type(expected: object, value: object) -> bool:
    """Whether a TOML value may fill a field annotated ``expected``.

    A float field takes an integer too, because ``pfa = 1`` means only one
    thing. A boolean is never a number here, although Python treats ``True``
    as the integer 1. Every other field of these configs holds a string. An
    ``expected`` of None, a key the dataclass does not define, is left for its
    constructor to reject.
    """
    is_number = isinstance(value, int | float) and not isinstance(value, bool)
    if expected is float:
        return is_number
    if expected is int:
        return is_number and isinstance(value, int)
    return expected is None or isinstance(value, str)


def _check_value_types(config_type: type[Any], table: Mapping[str, Any], name: str) -> None:
    """Raise ``TypeError`` for any TOML value whose type does not match its field.

    TOML tells integers, floats, strings and booleans apart, so nothing is
    coerced: ``n_train = 16.0`` and ``pfa = "1e-5"`` are rejected, not rounded
    or parsed. :func:`_matches_type` has the rule.
    """
    if wrong := {
        key: value
        for key, value in table.items()
        if not _matches_type(get_type_hints(config_type).get(key), value)
    }:
        hints = get_type_hints(config_type)
        details = "; ".join(
            f"{key} must be {getattr(hints[key], '__name__', 'str')}, "
            f"got {value!r} ({type(value).__name__})"
            for key, value in wrong.items()
        )
        msg = f"[{name}] has values of the wrong type: {details}."
        raise TypeError(msg)


def configs_from_scenario(scenario: Scenario) -> tuple[DetectionConfig, TrackingConfig]:
    """Build detection and tracking settings from a scenario's TOML tables.

    The one place the TOML spelling and the Python spelling are reconciled.
    They differ in two ways, both deliberate:

    * the TOML key is ``velocity_unfolding``, because that is what the
      specification calls it, but the conventions hook rejects a *parameter*
      whose leading token is a bare unit name -- ``velocity`` is one -- so the
      dataclass field is ``unfolding_mode``;
    * ``simulated_angles`` is a ``pipelines`` concern belonging to the ENU state
      models of §6.3 and is not part of the filter's settings, so it is dropped
      here and read from the scenario directly by whoever needs it.

    Parameters
    ----------
    scenario : Scenario
        A scenario loaded from TOML. Its ``detection`` and ``tracking`` tables
        may be ``None``, in which case the defaults are returned.

    Returns
    -------
    detection : DetectionConfig
    tracking : TrackingConfig

    Raises
    ------
    TypeError
        If a value's type does not match its field: a string where a number is
        expected, a float for a count, or a boolean for either. Also if a table
        carries a key the dataclass does not define, though ``_require_keys`` in
        :mod:`radar_forge.pipelines.scenarios` catches that first for any key
        outside the allowed set.
    """
    detection_table = dict(scenario.detection_table or {})
    tracking_table = dict(scenario.tracking_table or {})
    tracking_table.pop("simulated_angles", None)
    if "velocity_unfolding" in tracking_table:
        tracking_table["unfolding_mode"] = tracking_table.pop("velocity_unfolding")
    _check_value_types(DetectionConfig, detection_table, "detection")
    _check_value_types(TrackingConfig, tracking_table, "tracking")
    return DetectionConfig(**detection_table), TrackingConfig(**tracking_table)


def dual_prf_detections(
    products: Sequence[RangeDopplerProduct],
    fold_spans_mps: Sequence[float],
    *,
    config: DetectionConfig | None = None,
    max_velocity_mps: float,
    range_tolerance_m: float | None = None,
    wrap_range: bool = False,
) -> list[Measurement]:
    r"""Detect in a coprime pair of bursts, pair the detections and unfold their velocity.

    The dual-PRF alternative to §5.3. Two bursts at coprime pulse repetition
    frequencies fold the same true velocity differently, and the pair of folded
    values identifies it uniquely over a span far wider than either burst's
    own (:func:`~radar_forge.core.ambiguity.unfold_doppler_dual_prf`'s Notes
    give the span). The ambiguity is resolved in
    the **waveform**, so the tracker is handed a true range rate from the very
    first frame and needs no bootstrap, no fold selector and no consistency
    monitor.

    Two detections, one from each burst, are compatible when their ranges
    differ by at most ``range_tolerance_m``, the shorter way round when
    ``wrap_range`` is set. This assumes that the target's range is inside the
    span of both bursts' maps, so that both measure the same range: nothing
    here can check it, and :class:`ScenarioTracker`'s Notes say where it holds. They are paired when
    each is the other's nearest in range (mutual nearest neighbours), so no
    detection is used twice. A strong target's two detections pair even with a
    one-cell false alarm or sidelobe close by in one burst, and the false alarm
    is left out. Each pair is then resolved with
    :func:`radar_forge.core.ambiguity.unfold_doppler_dual_prf`. Every detection
    gets a status (:data:`DetectionStatus`):

    - a detection with no compatible detection in the other burst is
      ``"missing_pair"``;
    - a detection with compatible detections, none of which it is mutually
      nearest to, is ``"ambiguous_pair"``. Pairing it would be a guess;
    - a pair whose two folded velocities agree on no unfolded value within the
      tolerance is ``"unresolved_velocity"``;
    - otherwise both detections of the pair are ``"accepted"``.

    Two detections at exactly the same distance from a third make neither
    mutually nearest to it, so a tie is refused rather than broken.

    Parameters
    ----------
    products : sequence of RangeDopplerProduct
        Exactly two maps, one per burst, for the same frame, each of shape
        ``(n_doppler_bins, n_range_bins)``.
    fold_spans_mps : sequence of float
        Each burst's fold span, ``2 * burst.unambiguous_velocity_mps``.
    config : DetectionConfig, optional
        CFAR settings, shared by both bursts.
    max_velocity_mps : float
        The largest speed the target can have, in metres per second: the
        search bound passed to ``unfold_doppler_dual_prf``. A claim about the
        target, not about the pair, so it has no default. It should not exceed
        the pair's own unambiguous span, beyond which a velocity aliases onto a
        wrong one.
    range_tolerance_m : float, optional
        How close two detections must be in range to be called the same
        target. By default two range bins of the coarser map, as the velocity
        tolerance is two Doppler bins.
    wrap_range : bool, optional
        Passed to :func:`frame_detections` for both bursts. When set, range
        gaps are also measured round the circle of the smaller map's range
        axis, so a target at its wrap edge still pairs.

    Returns
    -------
    list of Measurement
        Every detection from both bursts, the first burst's first, each with
        ``burst_index`` and ``status`` set. The two detections of a pair share
        a ``pair_id`` and the unfolded velocity, whose ``fold_index`` is
        counted in the first burst's fold span.

    Raises
    ------
    ValueError
        If ``products`` and ``fold_spans_mps`` are not both of length two.

    References
    ----------
    .. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
           McGraw-Hill, 2014, §5.5.4 (resolving ambiguities with multiple PRFs).
    """
    if len(products) != 2 or len(fold_spans_mps) != 2:
        msg = (
            f"dual-PRF unfolding needs exactly two bursts; got {len(products)} "
            f"products and {len(fold_spans_mps)} fold spans."
        )
        raise ValueError(msg)

    first = frame_detections(products[0], config, wrap_range=wrap_range)
    second = [
        replace(m, burst_index=1)
        for m in frame_detections(products[1], config, wrap_range=wrap_range)
    ]
    span_a, span_b = (float(span) for span in fold_spans_mps)
    # One Doppler bin on the coarser burst, doubled: the tolerance the pair must
    # agree to before a candidate velocity is accepted.
    tolerance_mps = 2.0 * max(
        span_a / products[0].rd_map.shape[0], span_b / products[1].rd_map.shape[0]
    )
    if range_tolerance_m is None:
        # The same rule in range: two bins of the coarser map.
        range_tolerance_m = 2.0 * max(
            abs(float(product.range_axis_m[1] - product.range_axis_m[0])) for product in products
        )

    ranges_a_m = np.asarray([m.range_m for m in first], dtype=np.float64)
    ranges_b_m = np.asarray([m.range_m for m in second], dtype=np.float64)
    # gap_m[i, j]: range gap between first-burst detection i and second-burst
    # detection j. Shape (n_first, n_second).
    gap_m = ranges_a_m[:, None] - ranges_b_m[None, :]
    if wrap_range:
        # Each map's range axis is a circle, so the gap takes the shorter way
        # round, with the one wrap rule (StateLayout.wrap). The two circles may
        # differ (60 km and 50 km for S3), and the smaller is used: a target
        # inside both reads the same on both, except at the smaller circle's
        # edge, where one map shows it near zero and the other near the span.
        period_m = min(
            product.range_axis_m.size
            * abs(float(product.range_axis_m[1] - product.range_axis_m[0]))
            for product in products
        )
        layout = StateLayout((Coordinate("range_m", "m", period=period_m),), frame="radial")
        gap_m = layout.wrap(gap_m.reshape(-1, 1)).reshape(gap_m.shape)
    distance_m = np.abs(gap_m)
    compatible = distance_m <= range_tolerance_m
    # i and j are mutually nearest when j is strictly the nearest to i among
    # the second burst's detections, and i strictly the nearest to j among the
    # first's. Strictly: a tie makes neither the nearest.
    mutual = (
        compatible & _strictly_nearest(distance_m, axis=1) & _strictly_nearest(distance_m, axis=0)
    )
    paired_a = mutual.any(axis=1)
    paired_b = mutual.any(axis=0)

    # records holds the first burst's detections, then the second's, in one
    # flat list, so the second burst's detection j is records[len(first) + j].
    records = [replace(m, status="missing_pair") for m in (*first, *second)]
    unpaired = np.concatenate(
        [compatible.any(axis=1) & ~paired_a, compatible.any(axis=0) & ~paired_b]
    )
    for index in np.flatnonzero(unpaired):
        records[index] = replace(records[index], status="ambiguous_pair")

    # Each pair is unfolded on its own, and the outcome is discrete (accepted
    # or unresolved), so there is nothing to vectorise.
    for pair_id, (i, j) in enumerate(np.argwhere(mutual)):
        a, b = int(i), len(first) + int(j)
        velocity_mps, _ = unfold_doppler_dual_prf(
            first[i].velocity_folded_mps,
            second[j].velocity_folded_mps,
            span_a / 2.0,
            span_b / 2.0,
            max_velocity_mps=max_velocity_mps,
            tolerance_mps=tolerance_mps,
        )
        if np.isfinite(velocity_mps):
            fold_index = round((float(velocity_mps) - first[i].velocity_folded_mps) / span_a)
            for k in (a, b):
                records[k] = replace(
                    replace_measurement(records[k], float(velocity_mps), fold_index),
                    status="accepted",
                    pair_id=pair_id,
                )
        else:
            for k in (a, b):
                records[k] = replace(records[k], status="unresolved_velocity", pair_id=pair_id)
    return records


def _strictly_nearest(distance_m: NDArray[np.float64], axis: int) -> NDArray[np.bool_]:
    """Mark, along ``axis``, the one smallest entry of each line; a tie marks none.

    ``axis=1`` marks, in each row, the column nearest to that row. An empty
    line marks nothing.
    """
    if distance_m.shape[axis] == 0:
        return np.zeros(distance_m.shape, dtype=np.bool_)
    is_min = distance_m == distance_m.min(axis=axis, keepdims=True)
    return np.asarray(is_min & (is_min.sum(axis=axis, keepdims=True) == 1), dtype=np.bool_)


def dual_prf_measurements(
    products: Sequence[RangeDopplerProduct],
    fold_spans_mps: Sequence[float],
    *,
    config: DetectionConfig | None = None,
    max_velocity_mps: float,
    range_tolerance_m: float | None = None,
) -> list[Measurement]:
    """Detect in a coprime pair of bursts and return the measurements to track.

    The accepted detections of :func:`dual_prf_detections` from the first
    burst, one per resolved pair, carrying ``velocity_unfolded_mps``. A
    detection with no one-to-one partner is dropped: with nothing to pair
    against, its velocity cannot be resolved, and passing it on folded would be
    the §5.2 failure by another route.

    Parameters
    ----------
    products, fold_spans_mps, config, max_velocity_mps, range_tolerance_m
        As for :func:`dual_prf_detections`.

    Returns
    -------
    list of Measurement
        One per resolved pair, at the first burst's range.

    Raises
    ------
    ValueError
        If ``products`` and ``fold_spans_mps`` are not both of length two.
    """
    return [
        measurement
        for measurement in dual_prf_detections(
            products,
            fold_spans_mps,
            config=config,
            max_velocity_mps=max_velocity_mps,
            range_tolerance_m=range_tolerance_m,
        )
        if measurement.burst_index == 0 and measurement.status == "accepted"
    ]


def bin_quantisation_sigmas(product: RangeDopplerProduct) -> tuple[float, float]:
    r"""Return the range and velocity standard deviations of a detection at bin precision.

    A detection says only that the target is somewhere in a cell of width
    :math:`\Delta`. Taken as uniform across the cell, its position has
    variance :math:`\Delta^2/12`. At the high SNR of scenario 001 this is far
    larger than the thermal-noise accuracy (the CRLB) [1]_, so it is the
    measurement noise that matters.

    Parameters
    ----------
    product : RangeDopplerProduct
        The map and its two axes. Each axis must have at least two bins.

    Returns
    -------
    sigma_range_m : float
        Range bin width over :math:`\sqrt{12}`, in metres.
    sigma_velocity_mps : float
        Velocity bin width over :math:`\sqrt{12}`, in metres per second.

    Raises
    ------
    ValueError
        If either axis has fewer than two bins.

    References
    ----------
    .. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
           McGraw-Hill, 2014, §7.1 and §7.2 (estimators and their accuracy).
    """
    if product.range_axis_m.size < 2 or product.velocity_axis_mps.size < 2:
        msg = (
            "each axis needs at least two bins to have a bin width; got "
            f"{product.range_axis_m.size} range and {product.velocity_axis_mps.size} "
            "velocity bins."
        )
        raise ValueError(msg)
    range_bin_m = float(product.range_axis_m[1] - product.range_axis_m[0])
    velocity_bin_mps = float(product.velocity_axis_mps[1] - product.velocity_axis_mps[0])
    return abs(range_bin_m) / math.sqrt(12.0), abs(velocity_bin_mps) / math.sqrt(12.0)


def unfold_velocity_mps(
    velocity_folded_mps: float,
    velocity_predicted_mps: float,
    fold_span_mps: float,
) -> tuple[float, int]:
    r"""Resolve a folded range rate against a prediction, and return the fold index.

    The track's predicted range rate is unfolded, so it selects the fold:

    .. math::

        k = \operatorname{round}\!\left(
            \frac{\dot{r}_{\text{pred}} - \dot{r}_{\text{meas}}}{v_{\text{span}}}
        \right), \qquad
        \dot{r} = \dot{r}_{\text{meas}} + k\, v_{\text{span}}

    Parameters
    ----------
    velocity_folded_mps : float
        The measured range rate, inside one fold span, positive closing.
    velocity_predicted_mps : float
        The track's predicted range rate, already unfolded.
    fold_span_mps : float
        The width of one fold, :math:`2 v_{\text{unamb}}`.

    Returns
    -------
    velocity_unfolded_mps : float
        The resolved range rate.
    fold_index : int
        The integer :math:`k` chosen.

    Raises
    ------
    ValueError
        If ``fold_span_mps`` is not positive.

    Notes
    -----
    An unfolding error is not modelled by :math:`R` and must not be. One wrong
    fold displaces the measurement by a whole span against a velocity standard
    deviation of 0.017 m/s, which is a normalised innovation of order
    :math:`10^5` against a gate of 9.21. A mis-unfolded measurement is therefore
    rejected by the gate and counts as a *miss*, never as a corrupted update.

    Examples
    --------
    >>> velocity_mps, fold_index = unfold_velocity_mps(7.505, -7.79, 15.2955)
    >>> int(fold_index)
    -1
    >>> bool(abs(velocity_mps - (-7.79)) < 0.01)
    True
    """
    if fold_span_mps <= 0.0:
        msg = f"fold_span_mps must be a positive fold width; got {fold_span_mps}."
        raise ValueError(msg)
    fold_index = round((velocity_predicted_mps - velocity_folded_mps) / fold_span_mps)
    return velocity_folded_mps + fold_index * fold_span_mps, fold_index


def slope_velocity_mps(range_history_m: Sequence[float], frame_time_s: float) -> float | None:
    """Return the closing rate implied by a least-squares fit to recent ranges.

    Independent of Doppler, and therefore of any fold decision, which is the
    whole point: it is the only velocity estimate a track cannot talk itself
    into. Positive closing, so it is the negative of the fitted range slope.

    Parameters
    ----------
    range_history_m : sequence of float
        Recent associated ranges, oldest first, uniformly spaced in time.
    frame_time_s : float
        Interval between frames, in seconds.

    Returns
    -------
    float or None
        The closing rate in metres per second, or ``None`` with fewer than two
        ranges.
    """
    if len(range_history_m) < 2:
        return None
    ranges_m = np.asarray(range_history_m, dtype=np.float64)
    frames = np.arange(ranges_m.size, dtype=np.float64) * frame_time_s
    slope_mps = float(np.polyfit(frames, ranges_m, 1)[0])
    return -slope_mps


def range_slope_sigma_mps(
    n_frames: int,
    sigma_range_m: float,
    frame_time_s: float,
) -> float:
    r"""Return the standard deviation of a least-squares range slope over N frames.

    For ranges measured at a uniform interval :math:`T` with independent errors
    of standard deviation :math:`\sigma_R`, the ordinary-least-squares slope has

    .. math::

        \sigma_{\dot{R}} = \frac{\sigma_R}{T}
        \sqrt{\frac{12}{N(N^2 - 1)}}

    This is the estimator scenario 003 §5.3 uses to size its bootstrap, and it
    reproduces the values §14.4 quotes: 6.84 m/s at :math:`N = 5` and 3.34 m/s
    at :math:`N = 8`.

    Parameters
    ----------
    n_frames : int
        Number of range measurements in the fit, at least 2.
    sigma_range_m : float
        Range measurement standard deviation, in metres.
    frame_time_s : float
        Interval between frames, in seconds.

    Returns
    -------
    float
        The slope standard deviation, in metres per second.

    Raises
    ------
    ValueError
        If fewer than two frames, or a non-positive frame interval.

    Examples
    --------
    >>> bool(abs(range_slope_sigma_mps(8, 21.635652855125496, 1.0) - 3.338) < 1e-3)
    True
    """
    if n_frames < 2:
        msg = f"n_frames must be at least 2 to fit a slope; got {n_frames}."
        raise ValueError(msg)
    if frame_time_s <= 0.0:
        msg = f"frame_time_s must be positive, in seconds; got {frame_time_s}."
        raise ValueError(msg)
    return float(sigma_range_m / frame_time_s * np.sqrt(12.0 / (n_frames * (n_frames**2 - 1))))


def minimum_unfold_history_frames(
    sigma_range_m: float,
    fold_span_mps: float,
    frame_time_s: float,
    unfold_sigma_gate: float = 6.0,
) -> int:
    r"""Frames of range history a track needs before it may unfold its Doppler.

    Unfolding is safe once the predicted range rate is certain to well inside
    half a fold span, which scenario 003 §5.3 writes as

    .. math::

        \sigma_{\dot{R}} < v_{\text{span}} / \texttt{unfold\_sigma\_gate}

    Returns the smallest :math:`N` for which :func:`range_slope_sigma_mps`
    satisfies it. At the scenario's own numbers -- 21.64 m, 15.2955 m/s, 1 s and
    a gate of 6 -- that is **10 frames**, so a track spends its first ten
    seconds range-only and then switches to the two-dimensional measurement.

    Parameters
    ----------
    sigma_range_m : float
        Range measurement standard deviation, in metres.
    fold_span_mps : float
        Width of one Doppler fold, in metres per second.
    frame_time_s : float
        Interval between frames, in seconds.
    unfold_sigma_gate : float, optional
        How many slope standard deviations must fit inside one fold span,
        default 6.

    Returns
    -------
    int
        The frame count, at least 2.

    Raises
    ------
    ValueError
        If ``unfold_sigma_gate`` is not positive, or the threshold cannot be met
        within a thousand frames.

    Examples
    --------
    >>> minimum_unfold_history_frames(21.635652855125496, 15.2955, 1.0)
    10
    """
    if unfold_sigma_gate <= 0.0:
        msg = f"unfold_sigma_gate must be positive; got {unfold_sigma_gate}."
        raise ValueError(msg)
    threshold_mps = fold_span_mps / unfold_sigma_gate

    # A short scan rather than inverting the cubic: N is small, the expression is
    # monotonically decreasing in N, and the bound makes the failure explicit.
    for n_frames in range(2, 1000):
        if range_slope_sigma_mps(n_frames, sigma_range_m, frame_time_s) < threshold_mps:
            return n_frames
    msg = (
        f"a range slope of sigma {sigma_range_m} m at {frame_time_s} s never reaches "
        f"{threshold_mps} m/s within a thousand frames; the fold cannot be resolved "
        f"from range history alone at these parameters."
    )
    raise ValueError(msg)


@dataclass(frozen=True)
class TrackRecord:
    """One track as it stood at the end of one frame, from either tracker.

    The pipeline's own record of a track, so that the runner, the file
    writers and the plots read one shape whichever tracker ran. It holds
    copies, made when the frame ended. A live track and its filter change
    every frame, so keeping references to them would make every recorded frame
    show the *final* state of every track, and a plot or a CSV built from the
    history would be silently wrong in a way that still looks like a track.

    Attributes
    ----------
    track_id : int
        The track's identifier, fixed for its life.
    status : TrackStatus
        ``"tentative"``, ``"confirmed"`` or ``"coasting"``. A track deleted this
        frame has no record.
    state : numpy.ndarray
        Shape ``(2,)``: range in metres and range rate in metres per second,
        closing-positive, in :data:`STATE_FIELDS` order. When range folds,
        range is wrapped into ``[0, range period)``; see
        :attr:`ScenarioTracker.folding_layout`.
    covariance : numpy.ndarray
        Shape ``(2, 2)``, in the same order. Its units are m², m²/s and m²/s².
    measurement_dim : int
        Length of the measurement the track used this frame: 1 for range
        alone, 2 for range and range rate.
    n_hits : int
        Frames in which the track was updated with a measurement, counting its
        birth.
    n_misses : int
        Frames in a row without a measurement.
    nis : float or None
        The normalised innovation squared of the measurement the track took
        this frame. ``None`` if it missed, or was born this frame.
    detection_index : int or None, default None
        Index into :attr:`FrameTracks.measurements` of the detection the track
        took this frame, or was born from. ``None`` on a miss. A birth is not an
        association (:attr:`FrameTracks.associations`), but the track still
        began at a detection, and this says which.
    """

    track_id: int
    status: TrackStatus
    state: NDArray[np.float64]
    covariance: NDArray[np.float64]
    measurement_dim: int
    n_hits: int
    n_misses: int
    nis: float | None
    detection_index: int | None = None

    @property
    def is_confirmed(self) -> bool:
        """Whether the track has passed its M-of-N test: confirmed or coasting."""
        return self.status in ("confirmed", "coasting")


@dataclass(frozen=True)
class FrameTracks:
    """What the tracker did with one frame.

    Attributes
    ----------
    frame_index : int
        The frame's index within the run.
    time_s : float
        Frame time, in seconds.
    measurements : tuple of Measurement
        Every detection of the frame, from every burst, each with its
        ``status``. Only the ``"accepted"`` detections of the first burst
        reached the tracker. Each carries its unfolded velocity where one was
        available.
    associations : dict
        Maps ``track_id`` to the index into ``measurements`` of the detection
        it was updated with. A track born this frame is absent: a seed is not
        an association (data-001 §6.5), though its record's
        ``detection_index`` names it.
    tracks : tuple of TrackRecord
        Every track alive at the end of the frame.
    unfold_reference_id : int or None
        The track whose prediction selected the fold this frame, if any. Only
        the ``"kalman"`` path's track-aided unfolding sets it.
    """

    frame_index: int
    time_s: float
    measurements: tuple[Measurement, ...]
    associations: dict[int, int]
    tracks: tuple[TrackRecord, ...]
    unfold_reference_id: int | None = None


def range_layout(burst: RadarLike) -> StateLayout:
    """Return a one-coordinate layout, ``range_m``, with the period of a burst's map.

    The period is the span of the burst's range axis
    (:func:`~radar_forge.pipelines.scenarios.range_axis_m`): its number of bins
    times its bin width. A return beyond it lands back near zero on that map,
    so wrapping a true range with this layout puts it where the map shows it.

    Parameters
    ----------
    burst : Radar or BistaticRadar
        The burst whose map is meant.

    Returns
    -------
    StateLayout
        One coordinate, ``range_m`` in metres, with that period.

    References
    ----------
    .. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
           McGraw-Hill, 2014, §5.5.4 (range ambiguity and its resolution).
    """
    axis_m = range_axis_m(burst)
    range_period_m = axis_m.size * float(axis_m[1] - axis_m[0])
    return StateLayout((Coordinate("range_m", "m", period=range_period_m),), frame="radial")


@dataclass
class ScenarioTracker:
    """Detect, pair, unfold and track a scenario's frames, with either tracker.

    Holds the state that spans frames: the tracker itself and, on the
    ``"kalman"`` path, the per-track range history the §5.3 bootstrap needs.
    Feed it one :class:`~radar_forge.pipelines.scenarios.RangeDopplerProduct`
    per burst per frame, with :meth:`step`. :meth:`from_scenario` builds one
    from a scenario's TOML tables.

    Parameters
    ----------
    bursts : tuple of Radar or BistaticRadar
        One burst, or two FMCW bursts at different PRFs (dual PRF, as in
        scenario 001's S3), in the scenario's order.
    frame_time_s : float, optional
        Interval between frames, in seconds. This is the *frame* interval, not
        the CPI length.
    detection : DetectionConfig, optional
        CFAR settings.
    tracking : TrackingConfig, optional
        Which tracker, and its filter, gate and track-management settings.

    Attributes
    ----------
    bursts : tuple of Radar or BistaticRadar
        As passed in.
    detection : DetectionConfig
        The CFAR settings in use.
    tracking : TrackingConfig
        The tracking settings in use.
    tracker : KalmanTracker or Tracker
        The tracker that holds the tracks: a
        :class:`~radar_forge.core.tracking.KalmanTracker` when
        ``tracking.estimator`` is ``"kalman"``, a
        :class:`~radar_forge.core.tracking.Tracker` running a UKF when it is
        ``"ukf"``.
    folding_layout : StateLayout
        One coordinate, ``range_m``, with the period the measured range has:
        the span of the first burst's range axis
        (:func:`~radar_forge.pipelines.scenarios.range_axis_m`) on the
        ``"ukf"`` path, and none on the ``"kalman"`` path. The measurement
        model, the exported range (:meth:`StateLayout.wrap`) and the range
        error of a scored track (:meth:`StateLayout.residual`) all use it.
    min_unfold_frames : int or None
        ``"kalman"`` path only, ``None`` otherwise. Frames of associated range
        history a track needs before it may unfold its Doppler, from
        :func:`minimum_unfold_history_frames` at this tracker's own settings.
        Derived, never passed in: it is a consequence of ``sigma_range_m``,
        :attr:`fold_span_mps`, ``frame_time_s`` and ``unfold_sigma_gate``, and
        setting it independently of those would let a track unfold before its
        range slope can tell it which fold to take.
    frames : list of FrameTracks
        Every frame stepped so far, in order, appended by :meth:`step`.

    Raises
    ------
    ValueError
        If there are not one or two bursts; if two bursts share an unambiguous
        velocity, so their folds cannot be told apart, or ``tracking.v_max_mps``
        does not exceed the smaller of the two; if ``tracking.estimator``,
        ``tracking.unfolding_mode`` or ``detection.variant`` is not a known
        name; or if the ``"ukf"``
        estimator is asked for track-aided or oracle unfolding, or for a state
        model other than ``"range_1d"``.

    Notes
    -----
    **What is measured.** A burst folds range at the span of its range axis and
    Doppler at its unambiguous velocity, and the tracker should measure only
    what does not fold, or what has been unfolded.

    - Range: on the ``"ukf"`` path the radar only ever sees range modulo the
      span of the first burst's range axis, and the measurement model says
      exactly that. The span is ``Radar.unambiguous_range_m`` for a pulsed
      burst and twice it for an FMCW burst, whose deramped baseband is complex
      (:func:`~radar_forge.pipelines.scenarios.range_axis_m`, and
      ``spec/scenario-003-tracking.md`` §14.11). The filter's range state is
      left unwrapped. It sits on whichever fold the first detection gave it,
      and only its value modulo the period means anything, which is why the
      exported range is wrapped. Whether a target crosses the wrap depends on
      the scenario's window, so each run's figures are in
      ``spec/scenario-003-tracking.md`` §14.11, beside the window they hold
      for. The ``"kalman"`` path treats range as unambiguous.
    - Two bursts: their maps may span different ranges, and
      :func:`dual_prf_detections` pairs their detections on the assumption
      that the target is inside both spans, so that both measure the same
      range. The tracker cannot check that, since it does not know the truth.
      ``tests/pipelines/test_scenario_003.py`` checks it over each shipped
      scenario's window.
    - Range rate: with two bursts it is unfolded by
      :func:`dual_prf_detections` and measured. With one burst on the
      ``"ukf"`` path it is measured only if the burst's unambiguous velocity
      exceeds ``tracking.v_max_mps``, so that the Doppler cannot fold. On the
      ``"kalman"`` path it is unfolded by the track (§5.3).

    **Track-aided unfolding** (``"kalman"`` path) would have to be done per
    (track, measurement) pair rather than once per frame with more than one
    target: two targets at different speeds select different folds for the
    same detection. Scenario 003 has one target by construction, so a single
    reference track is used -- the unfoldable track with the longest history.
    Multi-target association is out of scope for this slice and is the
    subject of §13.3.

    References
    ----------
    .. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
           McGraw-Hill, 2014, §7.1 and §7.2 (range and Doppler estimators).
    .. [2] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with
           Applications to Tracking and Navigation*, Wiley, 2001, §5.5
           (initialisation of state estimators).
    """

    bursts: tuple[RadarLike, ...]
    frame_time_s: float = 1.0
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    tracker: KalmanTracker | Tracker = field(init=False)
    folding_layout: StateLayout = field(init=False)
    frames: list[FrameTracks] = field(default_factory=list, init=False)
    # The "ukf" path only: how many components each measurement has.
    _n_measured: int = field(default=2, init=False)
    # The "kalman" path only; they go with KalmanTracker (tracker-001 §13 step 6).
    min_unfold_frames: int | None = field(init=False)
    _range_history: dict[int, list[float]] = field(default_factory=dict, init=False)
    _unfold_frame: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Check the settings, then build the tracker they ask for."""
        self._check_settings()
        burst = self.bursts[0]
        if self.tracking.estimator == "ukf":
            self.folding_layout = range_layout(burst)
            # A single burst may fold the Doppler unless its unambiguous
            # velocity exceeds any speed the target can have.
            folds_doppler = (
                len(self.bursts) == 1 and burst.unambiguous_velocity_mps <= self.tracking.v_max_mps
            )
            self._n_measured = 1 if folds_doppler else 2
            self.min_unfold_frames = None
            self.tracker = self._build_ukf()
            return

        self.folding_layout = StateLayout((Coordinate("range_m", "m"),), frame="radial")
        self.tracker = self._build_kalman()
        self.min_unfold_frames = minimum_unfold_history_frames(
            self.tracking.sigma_range_m,
            self.fold_span_mps,
            self.frame_time_s,
            self.tracking.unfold_sigma_gate,
        )

    def _check_settings(self) -> None:
        """Raise ``ValueError`` for bursts or settings this class cannot run."""
        if len(self.bursts) not in (1, 2):
            msg = f"tracking needs one burst or a dual-PRF pair; got {len(self.bursts)} bursts."
            raise ValueError(msg)
        if (
            len(self.bursts) == 2
            and self.bursts[0].unambiguous_velocity_mps == self.bursts[1].unambiguous_velocity_mps
        ):
            msg = (
                "the two bursts of a dual-PRF pair must fold velocity differently; both have "
                f"an unambiguous velocity of {self.bursts[0].unambiguous_velocity_mps} m/s."
            )
            raise ValueError(msg)
        smallest_mps = min(burst.unambiguous_velocity_mps for burst in self.bursts)
        if len(self.bursts) == 2 and self.tracking.v_max_mps <= smallest_mps:
            msg = (
                "v_max_mps bounds the dual-PRF velocity search, so it must exceed the smaller "
                f"unambiguous velocity, {smallest_mps} m/s; got {self.tracking.v_max_mps} m/s."
            )
            raise ValueError(msg)
        if self.detection.variant not in CFAR_VARIANTS:
            msg = (
                f"detection variant must be one of {list(CFAR_VARIANTS)}; "
                f"got {self.detection.variant!r}."
            )
            raise ValueError(msg)
        if self.tracking.estimator not in ESTIMATORS:
            msg = f"estimator must be one of {list(ESTIMATORS)}; got {self.tracking.estimator!r}."
            raise ValueError(msg)
        if self.tracking.unfolding_mode not in UNFOLDING_MODES:
            msg = (
                f"unfolding_mode must be one of {list(UNFOLDING_MODES)}; "
                f"got {self.tracking.unfolding_mode!r}."
            )
            raise ValueError(msg)
        if self.tracking.estimator == "ukf" and self.tracking.unfolding_mode != "none":
            msg = (
                "the 'ukf' estimator has no track-aided unfolding: it unfolds by dual PRF or "
                f"not at all, so unfolding_mode must be 'none'; got "
                f"{self.tracking.unfolding_mode!r}."
            )
            raise ValueError(msg)
        if self.tracking.estimator == "ukf" and self.tracking.state_model != "range_1d":
            msg = (
                "the 'ukf' estimator runs the range_1d state model only; got "
                f"{self.tracking.state_model!r}."
            )
            raise ValueError(msg)

    def _build_ukf(self) -> Tracker:
        """Build the UKF tracker: radial motion, range (and range rate) measured."""
        motion = RadialMotion(
            self.tracking.sigma_accel_mps2, self.tracking.acceleration_correlation_time_s
        )
        range_period_m = self.folding_layout.coordinates[0].period
        observation = CartesianPosition(
            motion.state_layout,
            STATE_FIELDS[: self._n_measured],
            None if range_period_m is None else {"range_m": range_period_m},
        )
        # DirectStateInitiator overwrites every measured element of this prior,
        # and its time, with the first detection. So the range mean 0, its 1 m²
        # variance and the time 0 are placeholders that never reach a filter.
        # The range-rate variance is used only when range rate is not measured:
        # a new track then knows only that the target is no faster than
        # v_max_mps.
        prior = StateEstimate(
            np.zeros(2, dtype=np.float64),
            np.diag(np.asarray([1.0, self.tracking.v_max_mps**2], dtype=np.float64)),
            0.0,
            motion.state_layout,
        )
        policy = LifecyclePolicy(
            n_confirm_hits=self.tracking.n_confirm_hits,
            n_confirm_frames=self.tracking.n_confirm_frames,
            n_delete_misses=self.tracking.n_delete_misses,
            n_reacquire_frames=self.tracking.n_reacquire_frames,
        )
        return build_tracker(
            motion,
            observation,
            prior,
            policy=policy,
            gate_probability=self.tracking.gate_probability,
            sensor_id=SENSOR_ID,
            model_id=MEASUREMENT_MODEL_ID,
        )

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> ScenarioTracker:
        """Build the tracker a scenario's ``[detection]`` and ``[tracking]`` tables ask for.

        Parameters
        ----------
        scenario : Scenario
            A scenario loaded from TOML. Missing tables give the defaults.

        Returns
        -------
        ScenarioTracker
            With no frames stepped yet.

        Raises
        ------
        TypeError
            If a table value has the wrong type; see :func:`configs_from_scenario`.
        ValueError
            As for the class.
        """
        detection, tracking = configs_from_scenario(scenario)
        return cls(
            bursts=scenario.bursts,
            frame_time_s=1.0 / scenario.frame_rate_hz,
            detection=detection,
            tracking=tracking,
        )

    def step(
        self,
        products: Sequence[RangeDopplerProduct],
        *,
        frame_index: int,
        time_s: float,
        truth_velocity_mps: float | None = None,
    ) -> FrameTracks:
        """Detect, pair, unfold, associate and filter one frame.

        Parameters
        ----------
        products : sequence of RangeDopplerProduct
            One range-Doppler map per burst, in the same order as ``bursts``,
            each of shape ``(n_doppler_bins, n_range_bins)``.
        frame_index : int
            Index of this frame within the run.
        time_s : float
            Frame time, in seconds. Must not be earlier than the previous
            frame's.
        truth_velocity_mps : float, optional
            The true range rate, required only by the ``"oracle"`` unfolding
            mode.

        Returns
        -------
        FrameTracks
            Every detection with its status, which detection each track took,
            and the tracks after this frame's update.

        Raises
        ------
        ValueError
            If the number of products differs from the number of bursts, if
            ``time_s`` is earlier than the previous frame's, or if the
            ``"oracle"`` mode is selected and no truth velocity is given.
        """
        if len(products) != len(self.bursts):
            msg = (
                f"step needs one range-Doppler product per burst; got {len(products)} "
                f"products for {len(self.bursts)} bursts."
            )
            raise ValueError(msg)
        if self.frames and time_s < self.frames[-1].time_s:
            msg = (
                f"frames must arrive in time order; got time_s = {time_s} after "
                f"{self.frames[-1].time_s}."
            )
            raise ValueError(msg)

        # __post_init__ built the tracker tracking.estimator names, so that
        # field, not the tracker's class, says which path this is. The UKF path
        # measures range modulo the map's range axis, so it detects on that
        # axis as the circle it is. The Kalman path treats range as
        # unambiguous, as scenario 003 always has.
        is_ukf = self.tracking.estimator == "ukf"
        wrap_range = is_ukf
        if len(self.bursts) == 2:
            spans_mps = [2.0 * burst.unambiguous_velocity_mps for burst in self.bursts]
            detections = dual_prf_detections(
                products,
                spans_mps,
                config=self.detection,
                max_velocity_mps=self.tracking.v_max_mps,
                wrap_range=wrap_range,
            )
        else:
            detections = frame_detections(products[0], self.detection, wrap_range=wrap_range)

        reference_id: int | None = None
        if is_ukf:
            detections, associations, records = self._track_ukf(
                cast(Tracker, self.tracker), products[0], detections, time_s
            )
        else:
            detections, associations, records, reference_id = self._track_kalman(
                cast(KalmanTracker, self.tracker),
                detections,
                frame_index,
                time_s,
                truth_velocity_mps,
            )

        frame = FrameTracks(
            frame_index=frame_index,
            time_s=time_s,
            measurements=tuple(detections),
            associations=associations,
            tracks=records,
            unfold_reference_id=reference_id,
        )
        self.frames.append(frame)
        return frame

    def _track_ukf(
        self,
        tracker: Tracker,
        product: RangeDopplerProduct,
        detections: list[Measurement],
        time_s: float,
    ) -> tuple[list[Measurement], dict[int, int], tuple[TrackRecord, ...]]:
        """Run the UKF tracker over one frame's accepted detections."""
        if len(self.bursts) == 1 and self._n_measured == 2:
            # The burst's unambiguous velocity exceeds v_max_mps, so the
            # Doppler cannot fold: the folded velocity is the velocity, on fold 0.
            detections = [replace_measurement(m, m.velocity_folded_mps, 0) for m in detections]

        sigma_range_m, sigma_velocity_mps = bin_quantisation_sigmas(product)
        n = self._n_measured
        covariance = np.diag(
            np.asarray([sigma_range_m**2, sigma_velocity_mps**2], dtype=np.float64)
        )
        covariance = covariance[:n, :n]
        accepted = [
            index
            for index, measurement in enumerate(detections)
            if measurement.burst_index == 0 and measurement.status == "accepted"
        ]
        # When velocity is measured, every accepted detection must carry an
        # unfolded velocity: dual-PRF pairing or the fold-0 rule above set it.
        unfolded_mps = [detections[index].velocity_unfolded_mps for index in accepted]
        if n == 2 and None in unfolded_mps:
            msg = "an accepted detection reached the UKF without an unfolded velocity."
            raise RuntimeError(msg)
        values = [
            np.asarray(
                [detections[index].range_m, velocity_mps][:n],
                dtype=np.float64,
            )
            for index, velocity_mps in zip(accepted, unfolded_mps, strict=True)
        ]
        batch = tracker.sensors[SENSOR_ID].batch(
            time_s, [(MEASUREMENT_MODEL_ID, value, covariance) for value in values]
        )
        # A track deleted this scan reports once, as "deleted", and gets no
        # record (TrackRecord).
        snapshots = [
            snapshot for snapshot in tracker.process(batch) if snapshot.status != "deleted"
        ]
        # The batch held only the accepted detections, so a batch index maps
        # back to a detection through accepted.
        records = tuple(
            TrackRecord(
                track_id=snapshot.track_id,
                status=snapshot.status,
                state=np.concatenate(
                    [
                        self.folding_layout.wrap(snapshot.state[:1], interval="nonnegative"),
                        snapshot.state[1:],
                    ]
                ),
                covariance=np.array(snapshot.covariance, dtype=np.float64),
                measurement_dim=n,
                n_hits=snapshot.n_hits,
                n_misses=snapshot.n_misses,
                nis=snapshot.nis,
                detection_index=None
                if snapshot.measurement_index is None
                else accepted[snapshot.measurement_index],
            )
            for snapshot in snapshots
        )
        # An update has a NIS; a birth, which also took a detection, has none.
        associations = {
            record.track_id: record.detection_index
            for record in records
            if record.detection_index is not None and record.nis is not None
        }
        return detections, associations, records

    # ------------------------------------------------------------------ #
    # The "kalman" path only. Everything below goes with KalmanTracker
    # when scenario 003 moves onto Tracker (spec/tracker-001.md §13 step 6),
    # together with the else-branches of __post_init__ and step.
    # ------------------------------------------------------------------ #

    def _build_kalman(self) -> KalmanTracker:
        """Build the Kalman tracker: range_1d, with the §5.3 bootstrap's settings."""
        model = state_model_matrices(
            "range_1d",
            self.frame_time_s,
            sigma_accel_mps2=self.tracking.sigma_accel_mps2,
            sigma_range_m=self.tracking.sigma_range_m,
            sigma_velocity_mps=self.tracking.sigma_velocity_mps,
        )
        initial_covariance = np.diag(
            np.asarray(
                [self.tracking.sigma_range_m**2, self.tracking.v_max_mps**2],
                dtype=np.float64,
            )
        )
        return KalmanTracker(
            model=model,
            initial_covariance=initial_covariance,
            gate_probability=self.tracking.gate_probability,
            n_confirm_hits=self.tracking.n_confirm_hits,
            n_confirm_frames=self.tracking.n_confirm_frames,
            n_delete_misses=self.tracking.n_delete_misses,
            n_reacquire_frames=self.tracking.n_reacquire_frames,
        )

    @property
    def fold_span_mps(self) -> float:
        """Width of one Doppler fold of the first burst, ``2 * unambiguous_velocity_mps``."""
        return 2.0 * self.bursts[0].unambiguous_velocity_mps

    def is_unfoldable(self, track: Track[KalmanFilter]) -> bool:
        """Whether this track has enough range history to select a Doppler fold."""
        if self.min_unfold_frames is None:
            return False
        return len(self._range_history.get(track.track_id, ())) >= self.min_unfold_frames

    def unfold_frame_of(self, track_id: int) -> int | None:
        """Return the frame at which a track stepped from range-only to both components.

        The specification calls this the most informative single diagnostic the
        scenario produces, which is why it is recorded rather than recomputed.
        """
        return self._unfold_frame.get(track_id)

    def _track_kalman(
        self,
        tracker: KalmanTracker,
        detections: list[Measurement],
        frame_index: int,
        time_s: float,
        truth_velocity_mps: float | None,
    ) -> tuple[list[Measurement], dict[int, int], tuple[TrackRecord, ...], int | None]:
        """Run the Kalman tracker over one frame's detections."""
        if len(self.bursts) == 2:
            # The dual-PRF path: the waveform has resolved the ambiguity, so
            # there is no bootstrap to serve out and no fold to select. Every
            # track measures range and range rate from its first frame.
            accepted = [
                index
                for index, measurement in enumerate(detections)
                if measurement.burst_index == 0 and measurement.status == "accepted"
            ]
            associations, records = self._advance(
                tracker,
                [detections[index] for index in accepted],
                frame_index,
                time_s,
                unfolded=True,
            )
            # _advance indexes the accepted detections; map back to all of them.
            return (
                detections,
                {track_id: accepted[index] for track_id, index in associations.items()},
                tuple(
                    record
                    if record.detection_index is None
                    else replace(record, detection_index=accepted[record.detection_index])
                    for record in records
                ),
                None,
            )

        reference = self._reference_track(tracker)
        detections = self._unfold_all(detections, reference, truth_velocity_mps)
        associations, records = self._advance(tracker, detections, frame_index, time_s)
        return detections, associations, records, reference.track_id if reference else None

    def _reference_track(self, tracker: KalmanTracker) -> Track[KalmanFilter] | None:
        """Return the track whose prediction selects the fold for this frame."""
        candidates = [track for track in tracker.tracks if self.is_unfoldable(track)]
        if not candidates:
            return None
        return max(candidates, key=lambda track: len(self._range_history[track.track_id]))

    def _advance(
        self,
        tracker: KalmanTracker,
        measurements: list[Measurement],
        frame_index: int,
        time_s: float,
        *,
        unfolded: bool = False,
    ) -> tuple[dict[int, int], tuple[TrackRecord, ...]]:
        """Run the Kalman tracker over one frame's measurements.

        Returns the associations, as indices into ``measurements``, and a
        record of every track alive at the end of the frame.
        """
        live = [track for track in tracker.tracks if track.is_alive]
        if unfolded:
            # Nothing is ambiguous, so a new track may be seeded from both
            # measurement components and every track measures both from birth.
            tracker.n_initiation_rows = 2
            dims = [2] * len(live)
        else:
            dims = [2 if self.is_unfoldable(track) else 1 for track in live]
        for track, dim in zip(live, dims, strict=True):
            if dim == 2 and track.track_id not in self._unfold_frame:
                self._unfold_frame[track.track_id] = frame_index

        vectors = [
            np.asarray(
                [
                    measurement.range_m,
                    measurement.velocity_unfolded_mps
                    if measurement.velocity_unfolded_mps is not None
                    else measurement.velocity_folded_mps,
                ],
                dtype=np.float64,
            )
            for measurement in measurements
        ]
        known_ids = {track.track_id for track in tracker.tracks}
        result = tracker.step(vectors, time_s=time_s, measurement_dims=dims)
        # KalmanTracker seeds one track per unassociated measurement, in order,
        # and appends it, so the new IDs pair with result.unassociated in turn.
        new_ids = [track.track_id for track in result.tracks if track.track_id not in known_ids]
        detection_index = dict(result.associations)
        detection_index.update(zip(new_ids, result.unassociated, strict=True))

        # Range history drives the bootstrap, so it counts associations, not
        # frames: a coasting track learns nothing new about its range slope.
        for track_id, measurement_index in result.associations.items():
            self._range_history.setdefault(track_id, []).append(
                measurements[measurement_index].range_m
            )
        for track in result.tracks:
            self._range_history.setdefault(track.track_id, [])
        # A track held for re-acquisition keeps its range history: the ranges
        # are unambiguous and were never the reason it was lost, and the fold
        # monitor needs them the moment it comes back.
        alive_ids = {track.track_id for track in result.tracks}
        alive_ids.update(tracker.lost_track_ids)
        self._range_history = {
            track_id: history
            for track_id, history in self._range_history.items()
            if track_id in alive_ids
        }

        records = tuple(
            TrackRecord(
                track_id=track.track_id,
                status=track.status,
                state=np.array(track.estimator.estimate.state, dtype=np.float64),
                covariance=np.array(track.estimator.estimate.covariance, dtype=np.float64),
                measurement_dim=track.estimator.measurement_dim,
                n_hits=track.n_hits,
                n_misses=track.n_misses,
                nis=track.estimator.last_nis,
                detection_index=detection_index.get(track.track_id),
            )
            for track in result.tracks
        )
        return result.associations, records

    def _unfold_all(
        self,
        measurements: Sequence[Measurement],
        reference: Track[KalmanFilter] | None,
        truth_velocity_mps: float | None,
    ) -> list[Measurement]:
        """Attach an unfolded velocity to each measurement, where one is available."""
        mode = self.tracking.unfolding_mode
        if mode == "none":
            # Present so that the failure it causes can be demonstrated rather
            # than only described: the filter's motion model is unfolded, so a
            # folded measurement is a wrong measurement model, not a tuning
            # problem.
            return [
                replace_measurement(measurement, measurement.velocity_folded_mps, 0)
                for measurement in measurements
            ]

        if mode == "oracle":
            if truth_velocity_mps is None:
                msg = (
                    "the 'oracle' unfolding mode needs truth_velocity_mps. It exists for "
                    "tests and teaching, to isolate tracker error from unfolding error, "
                    "and must never be a scenario default."
                )
                raise ValueError(msg)
            predicted_mps: float | None = truth_velocity_mps
        else:
            predicted_mps = (
                float(reference.estimator.estimate.state[1]) if reference is not None else None
            )

        if predicted_mps is None:
            return list(measurements)

        # The fold-consistency monitor. The filter's own predicted rate is by far
        # the best selector while it is locked on: once it has a Doppler
        # measurement of standard deviation 0.017 m/s it knows the target's
        # velocity to a hundredth of a metre per second, and over the default
        # window only 3.4% of frames change velocity by more than half a fold
        # span -- the floor no selector can beat.
        #
        # Its failure is that it cannot recover. A measurement unfolded to the
        # wrong fold agrees perfectly with the wrong prediction that chose it, so
        # it passes the gate; and with a velocity variance five orders of
        # magnitude below the range variance, the accumulating range residual can
        # never pull the estimate back. The track then runs a whole fold span --
        # 15.3 m/s -- away from the target while reporting a normalised
        # innovation near zero, which is the most convincing way a tracker can be
        # wrong.
        #
        # So the selection is checked against the range history, which is
        # unambiguous and knows nothing about Doppler. A one-fold error shows
        # there as a 15.3 m/s disagreement against a slope uncertainty of about
        # 8 m/s, and when it does, the fold nearest the slope is taken instead.
        slope_mps = (
            slope_velocity_mps(
                self._range_history.get(reference.track_id, [])[-self.tracking.n_slope_frames :],
                self.frame_time_s,
            )
            if reference is not None
            else None
        )

        unfolded: list[Measurement] = []
        for measurement in measurements:
            velocity_mps, fold_index = unfold_velocity_mps(
                measurement.velocity_folded_mps, predicted_mps, self.fold_span_mps
            )
            if slope_mps is not None and abs(velocity_mps - slope_mps) > 0.5 * self.fold_span_mps:
                velocity_mps, fold_index = unfold_velocity_mps(
                    measurement.velocity_folded_mps, slope_mps, self.fold_span_mps
                )
            unfolded.append(replace_measurement(measurement, velocity_mps, fold_index))
        return unfolded


@dataclass(frozen=True)
class MetricRow:
    """One row of ``metrics.csv``: one metric over one scope.

    ``spec/data-001-formats.md`` §6.9 keeps metrics tidy and long, one row per
    metric per scope, so the unit goes in the metric's name.

    Attributes
    ----------
    metric : str
        The metric's name, with a unit suffix where it has a unit, such as
        ``range_rmse_m``. A count starts ``n_``.
    value : float
        The metric's value.
    frame_start, frame_end : int
        The inclusive window of frames the metric covers.
    track_id : int or None, default None
        The track the metric scores, or ``None`` for a run-level metric.
    target_id : str or None, default None
        The target the track is scored against, or ``None``.
    """

    metric: str
    value: float
    frame_start: int
    frame_end: int
    track_id: int | None = None
    target_id: str | None = None


def primary_track_id(frames: Sequence[FrameTracks]) -> int | None:
    """Return the ID of the track confirmed in the most of ``frames``.

    The one rule for which track stands for the target, shared by the scoring
    and every plot, so that a short-lived false track never stands in for it
    and the plots never show a different track from the one that is scored.
    Coasting counts as confirmed. A tie goes to the track confirmed first.

    This is a stand-in for a track-to-truth associator, which scenario 003's one
    target does not need. Stone Soup's ``TrackToTruth``, with SIAP or OSPA
    metrics, is the general form.

    Parameters
    ----------
    frames : sequence of FrameTracks
        The run's frames so far, in order.

    Returns
    -------
    int or None
        The track ID, or ``None`` if no track was ever confirmed.

    References
    ----------
    .. [1] ``spec/scenario-003-tracking.md`` §14.11 (the C4 metric and the
           primary track).
    """
    # A Python loop: each frame holds a handful of tracks, and counting them
    # into a dict keeps "first confirmed wins a tie" without sorting IDs.
    n_confirmed: dict[int, int] = {}
    for frame in frames:
        for track in frame.tracks:
            if track.is_confirmed:
                n_confirmed[track.track_id] = n_confirmed.get(track.track_id, 0) + 1
    if not n_confirmed:
        return None
    return max(n_confirmed, key=lambda track_id: n_confirmed[track_id])


def score_primary_track(
    frames: Sequence[FrameTracks],
    truth_range_m: ArrayLike,
    truth_range_rate_mps: ArrayLike,
    *,
    folding_layout: StateLayout,
    target_id: str = "0",
) -> list[MetricRow]:
    """Score a run's tracks against the truth of its one target.

    Scoring is kept apart from writing the files, so that the metrics can be
    recomputed from a run's ``tracks.csv`` and ``truth.csv`` alone, and a
    different track or a different metric needs no rerun.

    The **primary track** is :func:`primary_track_id`'s: the track confirmed
    in the most frames, the one the runner plots. Its metrics are taken over
    the frames in which it is confirmed or coasting:

    - ``n_confirmed_frames``: how many such frames there are;
    - ``confirmation_latency_s``: from the run's first frame to its first;
    - ``range_rmse_m`` and ``range_rate_rmse_mps``: root-mean-square error
      against the truth. Range errors take the shorter way round when range
      folds (``folding_layout.residual``);
    - ``mean_nis``: the mean normalised innovation squared over the frames in
      which it took a measurement. It should be near the measurement
      dimension when the filter's noise settings are right [1]_.

    Run-level metrics: ``n_tracked_frames``, the frames with any confirmed
    track; ``n_confirmed_tracks``, the distinct track IDs ever confirmed, which
    is 1 when the target is never lost; and ``n_detections_<status>``, one per
    :data:`DetectionStatus`.

    Parameters
    ----------
    frames : sequence of FrameTracks
        The run's frames, in order.
    truth_range_m, truth_range_rate_mps : array_like
        Shape ``(n_frames,)``. The target's true range and closing-positive
        range rate at each frame of ``frames``.
    folding_layout : StateLayout
        :attr:`ScenarioTracker.folding_layout`, which says whether range folds.
    target_id : str, optional
        The target's identifier, default ``"0"``.

    Returns
    -------
    list of MetricRow
        Every metric covers the run's whole window. With no confirmed track,
        only the run-level metrics are returned.

    Raises
    ------
    ValueError
        If ``frames`` is empty, or the truth arrays are not one value per frame.

    References
    ----------
    .. [1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with
           Applications to Tracking and Navigation*, Wiley, 2001, §5.4 (filter
           consistency: NIS and NEES).
    """
    range_m = np.asarray(truth_range_m, dtype=np.float64)
    range_rate_mps = np.asarray(truth_range_rate_mps, dtype=np.float64)
    if not frames or range_m.shape != (len(frames),) or range_rate_mps.shape != (len(frames),):
        msg = (
            f"need one truth value per frame for a nonempty run; got {len(frames)} frames, "
            f"truth shapes {range_m.shape} and {range_rate_mps.shape}."
        )
        raise ValueError(msg)

    start, end = frames[0].frame_index, frames[-1].frame_index
    statuses = [measurement.status for frame in frames for measurement in frame.measurements]
    confirmed_ids = {
        track.track_id for frame in frames for track in frame.tracks if track.is_confirmed
    }
    n_tracked = sum(any(track.is_confirmed for track in frame.tracks) for frame in frames)
    rows = [
        MetricRow("n_tracked_frames", float(n_tracked), start, end),
        MetricRow("n_confirmed_tracks", float(len(confirmed_ids)), start, end),
        *(
            MetricRow(f"n_detections_{status}", float(statuses.count(status)), start, end)
            for status in DETECTION_STATUSES
        ),
    ]
    primary_id = primary_track_id(frames)
    if primary_id is None:
        return rows

    # (frame position, record) for every frame in which the primary track is confirmed.
    confirmed = [
        (position, track)
        for position, frame in enumerate(frames)
        for track in frame.tracks
        if track.track_id == primary_id and track.is_confirmed
    ]
    positions = np.asarray([position for position, _ in confirmed], dtype=np.intp)
    states = np.asarray([track.state for _, track in confirmed], dtype=np.float64)
    range_error_m = folding_layout.residual(states[:, :1], range_m[positions, None])[:, 0]
    range_rate_error_mps = states[:, 1] - range_rate_mps[positions]
    nis = [track.nis for _, track in confirmed if track.nis is not None]
    latency_s = frames[int(positions[0])].time_s - frames[0].time_s
    scored = [
        ("n_confirmed_frames", float(len(confirmed))),
        ("confirmation_latency_s", latency_s),
        ("range_rmse_m", float(np.sqrt(np.mean(range_error_m**2)))),
        ("range_rate_rmse_mps", float(np.sqrt(np.mean(range_rate_error_mps**2)))),
        ("mean_nis", float(np.mean(nis)) if nis else math.nan),
    ]
    rows += [
        MetricRow(metric, value, start, end, primary_id, target_id) for metric, value in scored
    ]
    return rows


def replace_measurement(
    measurement: Measurement,
    velocity_unfolded_mps: float,
    fold_index: int,
) -> Measurement:
    """Return ``measurement`` carrying an unfolded velocity and its fold index.

    Parameters
    ----------
    measurement : Measurement
        The folded measurement; left unchanged, since :class:`Measurement` is
        frozen.
    velocity_unfolded_mps : float
        The resolved range rate, metres/second, positive closing.
    fold_index : int
        The integer :math:`k` that was chosen.

    Returns
    -------
    Measurement
        A copy carrying both, with every other field as it was.
    """
    return replace(
        measurement,
        velocity_unfolded_mps=velocity_unfolded_mps,
        fold_index=fold_index,
    )
