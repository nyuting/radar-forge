r"""Turn range-Doppler maps into tracks: detect, unfold, associate, filter.

This module is the pipeline half of
``spec/scenario-003-tracking.md``. It owns everything between a
:class:`~radar_forge.pipelines.scenarios.RangeDopplerProduct` and a call to
:class:`~radar_forge.core.tracking.TrackManager`:

* CFAR detection and the conversion of fractional cell indices into metres and
  metres per second (§5.1), using the bin-centre helpers of
  :mod:`radar_forge.core.dsp` rather than a locally re-derived bin spacing;
* Doppler **unfolding** and the bootstrap a brand-new track needs before it can
  unfold anything (§5.3);
* the **simulated angle measurement** the ENU state models need until an array
  exists (§6.3).

:mod:`radar_forge.core.tracking` knows about none of this, and that separation
is deliberate: the tracker's contract is "you give me measurements and a
model", and where a measurement came from is not its business.

Two departures from the specification as written, both recorded in its §5.3 and
§13 and both found by running it:

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
       McGraw-Hill, 2014, §6.5 (CFAR), §7.3 (measurement accuracy).
.. [2] S. S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking
       Systems*, Artech House, 1999, §4.3 (Doppler-aided tracking and ambiguity
       resolution).
.. [3] Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with
       Applications to Tracking and Navigation*, Wiley, 2001, §11.7 (track
       initiation).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
from numpy.typing import NDArray

from radar_forge.core.ambiguity import unfold_doppler_dual_prf
from radar_forge.core.detection import CfarVariant, cfar_detect, cluster_detections
from radar_forge.core.tracking import (
    Track,
    TrackManager,
    state_model_matrices,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from radar_forge.pipelines.scenarios import RangeDopplerProduct, Scenario

__all__ = [
    "UNFOLDING_MODES",
    "DetectionConfig",
    "FrameTracks",
    "Measurement",
    "ScenarioTracker",
    "TrackingConfig",
    "UnfoldingMode",
    "configs_from_scenario",
    "dual_prf_measurements",
    "frame_detections",
    "minimum_unfold_history_frames",
    "range_slope_sigma_mps",
    "replace_measurement",
    "slope_velocity_mps",
    "unfold_velocity_mps",
]

UnfoldingMode = Literal["track_aided", "oracle", "none"]
UNFOLDING_MODES: tuple[UnfoldingMode, ...] = get_args(UnfoldingMode)


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
    """The filter, gate and track-management settings of scenario 003 §2.

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
    """

    sigma_range_m: float = 21.635652855125496
    sigma_velocity_mps: float = 0.017247813329811023
    sigma_accel_mps2: float = 5.0
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


def _suppress_range_sidelobes(
    measurements: Sequence[Measurement],
    merge_range_bins: float,
) -> list[Measurement]:
    """Keep the strongest detection in each range cell and discard the rest.

    ``measurements`` must arrive strongest first, which is the order
    ``cluster_detections`` returns.
    """
    if merge_range_bins <= 0.0:
        return list(measurements)

    kept: list[Measurement] = []
    for measurement in measurements:
        if all(
            abs(measurement.range_index - other.range_index) > merge_range_bins for other in kept
        ):
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
    Scenario 001 applies no Doppler taper, so a target at the 51-65 dB
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
    n_doppler_bins = power_w.shape[0]

    mask = cfar_detect(
        power_w,
        pfa=settings.pfa,
        n_train=settings.n_train,
        n_guard=settings.n_guard,
        variant=settings.variant,
        axis=-1,
    )

    shift = _quietest_doppler_row(power_w)
    rolled_power_w = np.roll(power_w, -shift, axis=0)
    rolled_mask = np.roll(mask, -shift, axis=0)

    range_indices = np.arange(product.range_axis_m.size, dtype=np.float64)
    velocity_indices = np.arange(n_doppler_bins, dtype=np.float64)

    measurements: list[Measurement] = []
    for detection in cluster_detections(rolled_mask, rolled_power_w):
        velocity_index = float((detection.centroid_index[0] + shift) % n_doppler_bins)
        range_index = float(detection.centroid_index[1])
        measurements.append(
            Measurement(
                range_m=float(np.interp(range_index, range_indices, product.range_axis_m)),
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
    return _suppress_range_sidelobes(measurements, settings.merge_range_bins)


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
        If a table carries a key the dataclass does not define. ``_require_keys``
        in :mod:`radar_forge.pipelines.scenarios` catches this first for any key
        outside the allowed set.
    """
    detection = DetectionConfig(**(scenario.detection_table or {}))
    table = dict(scenario.tracking_table or {})
    table.pop("simulated_angles", None)
    if "velocity_unfolding" in table:
        table["unfolding_mode"] = table.pop("velocity_unfolding")
    return detection, TrackingConfig(**table)


def dual_prf_measurements(
    products: Sequence[RangeDopplerProduct],
    fold_spans_mps: Sequence[float],
    *,
    config: DetectionConfig | None = None,
    max_velocity_mps: float = 191.0,
    range_tolerance_m: float = 150.0,
) -> list[Measurement]:
    r"""Detect in a coprime pair of bursts and return velocity already unfolded.

    The dual-PRF alternative to §5.3. Two bursts at coprime pulse repetition
    frequencies fold the same true velocity differently, and the pair of folded
    values identifies it uniquely over a span far wider than either burst's
    own -- 191 m/s for scenario 001's 5:6 kHz pair. The ambiguity is resolved in
    the **waveform**, so the tracker is handed a true range rate from the very
    first frame and needs no bootstrap, no fold selector and no consistency
    monitor.

    Detections are paired across the two bursts by range, which is unambiguous
    in both, and each pair is resolved with
    :func:`radar_forge.core.ambiguity.unfold_doppler_dual_prf`.

    Parameters
    ----------
    products : sequence of RangeDopplerProduct
        Exactly two maps, one per burst, for the same frame.
    fold_spans_mps : sequence of float
        Each burst's fold span, ``2 * burst.unambiguous_velocity_mps``.
    config : DetectionConfig, optional
        CFAR settings, shared by both bursts.
    max_velocity_mps : float, optional
        The span the pair is asked to resolve over, default 191 m/s.
    range_tolerance_m : float, optional
        How close two detections must be in range to be called the same target,
        default 150 m, or about two of S3's range bins.

    Returns
    -------
    list of Measurement
        One per resolved pair, carrying ``velocity_unfolded_mps``. A detection
        in one burst with no partner in the other is dropped: with nothing to
        pair against, its velocity cannot be resolved, and passing it on folded
        would be the §5.2 failure by another route.

    Raises
    ------
    ValueError
        If ``products`` and ``fold_spans_mps`` are not both of length two.
    """
    if len(products) != 2 or len(fold_spans_mps) != 2:
        msg = (
            f"dual-PRF unfolding needs exactly two bursts; got {len(products)} "
            f"products and {len(fold_spans_mps)} fold spans."
        )
        raise ValueError(msg)

    first, second = (frame_detections(product, config) for product in products)
    span_a, span_b = (float(span) for span in fold_spans_mps)
    # One Doppler bin on the coarser burst, doubled: the tolerance the pair must
    # agree to before a candidate velocity is accepted.
    tolerance_mps = 2.0 * max(
        span_a / products[0].rd_map.shape[0], span_b / products[1].rd_map.shape[0]
    )

    resolved: list[Measurement] = []
    for measurement in first:
        partner = min(
            second,
            key=lambda other: abs(other.range_m - measurement.range_m),
            default=None,
        )
        if partner is None or abs(partner.range_m - measurement.range_m) > range_tolerance_m:
            continue

        velocity_mps, _ = unfold_doppler_dual_prf(
            measurement.velocity_folded_mps,
            partner.velocity_folded_mps,
            span_a / 2.0,
            span_b / 2.0,
            max_velocity_mps=max_velocity_mps,
            tolerance_mps=tolerance_mps,
        )
        if not np.isfinite(velocity_mps):
            continue
        resolved.append(
            replace_measurement(
                measurement,
                float(velocity_mps),
                round((float(velocity_mps) - measurement.velocity_folded_mps) / span_a),
            )
        )
    return resolved


def unfold_velocity_mps(
    velocity_folded_mps: float,
    velocity_predicted_mps: float,
    fold_span_mps: float,
) -> tuple[float, int]:
    r"""Resolve a folded range rate against a prediction, and return the fold index.

    The track's predicted range rate is unfolded, so it selects the fold [2]_:

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
    reproduces the values quoted there: 6.84 m/s at :math:`N = 5` and 3.34 m/s
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
class FrameTracks:
    """What the tracker did with one frame.

    Attributes
    ----------
    frame_index : int
        The frame's index within the run.
    time_s : float
        Frame time, in seconds.
    measurements : tuple of Measurement
        This frame's detections, carrying their unfolded velocity where a track
        was able to supply one.
    associations : dict
        Maps ``track_id`` to the index into ``measurements`` it took.
    tracks : tuple of Track
        Snapshots of every track alive at the end of the frame, copied so that
        this record keeps the state the tracks had *at this frame*.
    unfold_reference_id : int or None
        The track whose prediction selected the fold this frame, if any.
    """

    frame_index: int
    time_s: float
    measurements: tuple[Measurement, ...]
    associations: dict[int, int]
    tracks: tuple[Track, ...]
    unfold_reference_id: int | None = None


@dataclass
class ScenarioTracker:
    """Drive detection, unfolding and tracking over a scenario's frames.

    Holds the state that spans frames: the :class:`TrackManager` itself, and the
    per-track range history the §5.3 bootstrap needs. Feed it one
    :class:`~radar_forge.pipelines.scenarios.RangeDopplerProduct` per frame.

    Parameters
    ----------
    fold_span_mps : float
        Width of one Doppler fold, ``2 * burst.unambiguous_velocity_mps``.
    frame_time_s : float
        Interval between frames, in seconds. This is the *frame* interval, not
        the CPI length.
    detection : DetectionConfig, optional
        CFAR settings.
    tracking : TrackingConfig, optional
        Filter, gate and track-management settings.

    Attributes
    ----------
    min_unfold_frames : int
        Frames of associated range history a track needs before it may unfold
        its Doppler, from :func:`minimum_unfold_history_frames` at this
        tracker's own settings. Derived in ``__post_init__``, never passed in:
        it is a consequence of ``sigma_range_m``, ``fold_span_mps``,
        ``frame_time_s`` and ``unfold_sigma_gate``, and setting it
        independently of those would let a track unfold before its range slope
        can tell it which fold to take.
    manager : radar_forge.core.tracking.TrackManager
        The filter and track-management state, built from ``tracking``.
    frames : list of FrameTracks
        Every frame stepped so far, in order, appended by :meth:`step` and
        :meth:`step_unfolded`.

    Notes
    -----
    Unfolding is track-aided, and with more than one target it would have to be
    done per (track, measurement) pair rather than once per frame: two targets
    at different speeds select different folds for the same detection. Scenario
    003 has one target by construction, so a single reference track is used --
    the unfoldable track with the longest history. Multi-target association is
    out of scope for this slice and is the subject of §13.3.
    """

    fold_span_mps: float
    frame_time_s: float = 1.0
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    manager: TrackManager = field(init=False)
    min_unfold_frames: int = field(init=False)
    frames: list[FrameTracks] = field(default_factory=list, init=False)
    _range_history: dict[int, list[float]] = field(default_factory=dict, init=False)
    _unfold_frame: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Build the filter model and size the bootstrap."""
        if self.fold_span_mps <= 0.0:
            msg = f"fold_span_mps must be a positive fold width; got {self.fold_span_mps}."
            raise ValueError(msg)
        if self.tracking.unfolding_mode not in UNFOLDING_MODES:
            msg = (
                f"unfolding_mode must be one of {list(UNFOLDING_MODES)}; "
                f"got {self.tracking.unfolding_mode!r}."
            )
            raise ValueError(msg)

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
        self.manager = TrackManager(
            model=model,
            initial_covariance=initial_covariance,
            gate_probability=self.tracking.gate_probability,
            n_confirm_hits=self.tracking.n_confirm_hits,
            n_confirm_frames=self.tracking.n_confirm_frames,
            n_delete_misses=self.tracking.n_delete_misses,
            n_reacquire_frames=self.tracking.n_reacquire_frames,
        )
        self.min_unfold_frames = minimum_unfold_history_frames(
            self.tracking.sigma_range_m,
            self.fold_span_mps,
            self.frame_time_s,
            self.tracking.unfold_sigma_gate,
        )

    def is_unfoldable(self, track: Track) -> bool:
        """Whether this track has enough range history to select a Doppler fold."""
        return len(self._range_history.get(track.track_id, ())) >= self.min_unfold_frames

    def unfold_frame_of(self, track_id: int) -> int | None:
        """Return the frame at which a track stepped from range-only to both components.

        The specification calls this the most informative single diagnostic the
        scenario produces, which is why it is recorded rather than recomputed.
        """
        return self._unfold_frame.get(track_id)

    def _reference_track(self) -> Track | None:
        """Return the track whose prediction selects the fold for this frame."""
        candidates = [track for track in self.manager.tracks if self.is_unfoldable(track)]
        if not candidates:
            return None
        return max(candidates, key=lambda track: len(self._range_history[track.track_id]))

    def step(
        self,
        product: RangeDopplerProduct,
        *,
        frame_index: int,
        time_s: float,
        truth_velocity_mps: float | None = None,
    ) -> FrameTracks:
        """Detect, unfold, associate and filter one frame.

        Parameters
        ----------
        product : RangeDopplerProduct
            The frame's range-Doppler map and axes.
        frame_index : int
            Index of this frame within the run.
        time_s : float
            Frame time, in seconds.
        truth_velocity_mps : float, optional
            The true range rate, required only by the ``oracle`` unfolding mode.

        Returns
        -------
        FrameTracks
            This frame's measurements, associations and surviving tracks.

        Raises
        ------
        ValueError
            If the ``oracle`` mode is selected and no truth velocity is given.
        """
        measurements = frame_detections(product, self.detection)
        reference = self._reference_track()
        measurements = self._unfold_all(measurements, reference, truth_velocity_mps)
        return self._advance(measurements, frame_index, time_s, reference)

    def step_unfolded(
        self,
        measurements: Sequence[Measurement],
        *,
        frame_index: int,
        time_s: float,
    ) -> FrameTracks:
        """Advance one frame on measurements whose velocity is already resolved.

        The dual-PRF path. :func:`dual_prf_measurements` resolves the ambiguity
        in the waveform, so there is no bootstrap to serve out and no fold to
        select: every track measures range and range rate from its first frame.

        Parameters
        ----------
        measurements : sequence of Measurement
            This frame's detections, each carrying ``velocity_unfolded_mps``.
        frame_index : int
            Index of this frame within the run.
        time_s : float
            Frame time, in seconds.

        Returns
        -------
        FrameTracks
            This frame's measurements, associations and surviving tracks.
        """
        return self._advance(list(measurements), frame_index, time_s, None, unfolded=True)

    def _advance(
        self,
        measurements: list[Measurement],
        frame_index: int,
        time_s: float,
        reference: Track | None,
        *,
        unfolded: bool = False,
    ) -> FrameTracks:
        """Run the manager over one frame's measurements and record the result."""
        live = [track for track in self.manager.tracks if track.is_alive]
        if unfolded:
            # Nothing is ambiguous, so a new track may be seeded from both
            # measurement components and every track measures both from birth.
            self.manager.n_initiation_rows = 2
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
        result = self.manager.step(vectors, measurement_dims=dims)

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
        alive_ids.update(self.manager.lost_track_ids)
        self._range_history = {
            track_id: history
            for track_id, history in self._range_history.items()
            if track_id in alive_ids
        }

        frame = FrameTracks(
            frame_index=frame_index,
            time_s=time_s,
            measurements=tuple(measurements),
            associations=result.associations,
            # Snapshots, not the live Track objects. A Track is mutable by
            # design -- it is the thing that changes from frame to frame -- so
            # keeping references here would make every recorded frame show the
            # *final* state of every track, and a plot or a CSV built from the
            # history would be silently wrong in a way that still looks like a
            # track. KalmanState is frozen, so the copy is shallow and cheap.
            tracks=tuple(replace(track) for track in result.tracks),
            unfold_reference_id=reference.track_id if reference is not None else None,
        )
        self.frames.append(frame)
        return frame

    def _unfold_all(
        self,
        measurements: Sequence[Measurement],
        reference: Track | None,
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
            predicted_mps = float(reference.estimate.state[1]) if reference is not None else None

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
