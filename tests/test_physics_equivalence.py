"""TEMPORARY harness: the new flat physics modules reproduce the old ``core/`` cubes.

**The integration PR deletes this file**, together with the old ``core/``
modules it compares against, when it rewires ``pipelines/`` onto
``radar_forge.radar``, ``radar_forge.propagation`` and ``radar_forge.signal``.
Until then it is the evidence that the refactor moved nothing.

For scenario 001 (S1 FMCW, S2 pulsed, S3 dual-PRF FMCW) and scenario 002 (both
carriers), the old pipeline (``load_scenario``, ``iterate_frames``) produces the
first three frames. The same frames are rebuilt through the new
``Radar.geometry``, ``line_of_sight_paths`` and ``fmcw_deramp_baseband`` /
``pulsed_baseband``, from a generator seeded as the scenario seeds it and
consumed in the same order, frame by frame and burst by burst.

Two things the old pipeline does are reproduced on purpose:

- **The rate is a finite difference.** The old truth rate is a central
  difference of range over the frame grid. ``Radar.geometry`` computes
  ``v . u_T + v . u_R`` from a velocity instead, which differs by up to 0.6 m/s
  where the interpolated track turns, so the harness replaces it with
  ``np.gradient(bistatic_range_m, time_s)``. The old rate is closing-positive
  and half the route's (radial or bisector velocity), so the new one is
  compared with ``-2 * radial_velocity_mps``: the sign convention flips here.
- **Each bistatic range is measured in its own site's frame.** R_T comes from
  ``geometry(origin=tx_site)`` and R_R from ``geometry(origin=rx_site)``, as the
  old code measured them. One origin for both agrees to two ulp, which is
  asserted separately, but the carrier phase ``2 pi R_b / lambda`` turns one ulp
  of 2e4 m into 1e-9 rad, more than any 1e-12 cube tolerance allows.
"""

from __future__ import annotations

import itertools
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from radar_forge.core.radar import BistaticRadar
from radar_forge.core.radar import Radar as OldRadar
from radar_forge.pipelines.scenarios import Scenario, iterate_frames, load_scenario
from radar_forge.pipelines.trajectories import load_flight_csv, resample
from radar_forge.propagation import line_of_sight_paths
from radar_forge.radar import Radar, Receiver, Site, TargetGeometry, Transmitter, geodetic_to_enu_m
from radar_forge.signal import fmcw_deramp_baseband, pulsed_baseband

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
N_FRAMES = 3


def _new_radar(old: OldRadar | BistaticRadar) -> Radar:
    """The same transmitter, receiver and sites, as a new :class:`Radar`."""
    old_tx, old_rx = old.transmitter, old.receiver
    transmitter = Transmitter(
        old_tx.carrier_frequency_hz,
        old_tx.bandwidth_hz,
        old_tx.transmit_power_w,
        old_tx.gain_tx_dbi,
        old_tx.chirp_duration_s,
        old_tx.prf_hz,
        old_tx.waveform,
    )
    receiver = Receiver(old_rx.sample_rate_hz, old_rx.gain_rx_dbi, old_rx.noise_figure_db)
    if isinstance(old, BistaticRadar):
        tx_site = Site(
            old.transmitter_latitude_deg, old.transmitter_longitude_deg, old.transmitter_altitude_m
        )
        rx_site = Site(
            old.receiver_latitude_deg, old.receiver_longitude_deg, old.receiver_altitude_m
        )
        return Radar(transmitter, receiver, tx_site, rx_site)
    return Radar(transmitter, receiver, Site(old.latitude_deg, old.longitude_deg, old.altitude_m))


def _frame_grid_s(
    scenario: Scenario, track_time_s: NDArray[np.float64]
) -> tuple[NDArray[np.float64], int]:
    """The old pipeline's resampling grid: the window, padded a frame each side where it can be."""
    frame_times_s = scenario.frame_times_s
    frame_interval_s = 1.0 / scenario.frame_rate_hz
    pad_before = frame_times_s[0] - frame_interval_s >= track_time_s[0]
    pad_after = frame_times_s[-1] + frame_interval_s <= track_time_s[-1]
    grid_s = np.concatenate(
        [
            [frame_times_s[0] - frame_interval_s] if pad_before else [],
            frame_times_s,
            [frame_times_s[-1] + frame_interval_s] if pad_after else [],
        ]
    )
    return grid_s, int(pad_before)


def _new_geometry(scenario: Scenario, radar: Radar) -> tuple[TargetGeometry, int, TargetGeometry]:
    """The whole track's geometry through the new code, its first frame, and a one-origin copy."""
    trajectory = load_flight_csv(scenario.trajectory_path, altitude_m=scenario.target_altitude_m)
    grid_s, first_frame = _frame_grid_s(scenario, trajectory.time_s)
    sited = resample(trajectory, grid_s)
    rx_site = radar.tx_site if radar.rx_site is None else radar.rx_site

    def about(site: Site) -> TargetGeometry:
        position_enu_m = geodetic_to_enu_m(
            sited.latitude_deg,
            sited.longitude_deg,
            sited.altitude_m,
            site.latitude_deg,
            site.longitude_deg,
            site.altitude_m,
        )
        velocity_enu_mps = np.gradient(position_enu_m, grid_s, axis=0)
        return radar.geometry(position_enu_m, velocity_enu_mps, origin=site)

    one_origin = about(rx_site)
    from_tx = about(radar.tx_site)
    per_site = replace(
        one_origin,
        range_tx_m=from_tx.range_tx_m,
        bistatic_range_m=from_tx.range_tx_m + one_origin.range_rx_m,
        aod_rad=from_tx.aod_rad,
    )
    finite_difference = np.gradient(per_site.bistatic_range_m, grid_s)
    return replace(per_site, bistatic_range_rate_mps=finite_difference), first_frame, one_origin


def _at(geometry: TargetGeometry, index: int) -> TargetGeometry:
    """One frame of a track's geometry, as a one-target geometry."""
    return TargetGeometry(
        **{
            field.name: getattr(geometry, field.name)[index : index + 1]
            for field in fields(geometry)
        }
    )


@pytest.mark.parametrize(
    "toml_name",
    [
        "scenario_001_fmcw_low_prf.toml",
        "scenario_001_pulsed_medium_prf.toml",
        "scenario_001_fmcw_dual_prf.toml",
        "scenario_002_bistatic_xband.toml",
        "scenario_002_bistatic_sband.toml",
    ],
)
def test_new_physics_reproduces_the_old_frames(toml_name: str) -> None:
    scenario = load_scenario(SCENARIOS / toml_name)
    old_frames = list(itertools.islice(iterate_frames(scenario), N_FRAMES))
    assert len(old_frames) == N_FRAMES
    radars = [_new_radar(burst) for burst in scenario.bursts]
    geometry, first_frame, one_origin = _new_geometry(scenario, radars[0])
    # One origin for both ranges moves them by at most two ulp of 2e4 m.
    np.testing.assert_allclose(one_origin.range_tx_m, geometry.range_tx_m, rtol=1e-15, atol=0.0)

    rng = np.random.default_rng(scenario.seed)
    for old in old_frames:
        frame = _at(geometry, first_frame + old.index)
        if scenario.is_bistatic:
            assert frame.range_tx_m[0] == old.range_tx_m, "R_T, bit-identical"
            assert frame.range_rx_m[0] == old.range_rx_m, "R_R, bit-identical"
            assert old.bistatic_angle_deg is not None
            np.testing.assert_allclose(
                np.degrees(frame.bistatic_angle_rad[0]),
                old.bistatic_angle_deg,
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                np.degrees(frame.aod_rad[0]),
                [old.transmit_azimuth_deg, old.transmit_elevation_deg],
                rtol=0.0,
                atol=1e-12,
            )
        else:
            assert frame.range_tx_m[0] == old.range_m, "R, bit-identical"
            assert frame.bistatic_angle_rad[0] == 0.0, "monostatic beta is exactly 0"
        # The old rate is closing-positive and halved; the new one is receding-positive.
        assert frame.bistatic_range_rate_mps[0] == -2.0 * old.radial_velocity_mps, (
            "rate, bit-identical"
        )
        np.testing.assert_allclose(
            np.degrees(frame.aoa_rad[0]), [old.azimuth_deg, old.elevation_deg], rtol=0.0, atol=1e-12
        )

        for burst, (radar, n_pulses) in enumerate(zip(radars, scenario.n_pulses, strict=True)):
            paths = line_of_sight_paths(radar, frame, scenario.target.rcs_m2)
            generate = (
                pulsed_baseband if radar.transmitter.waveform == "pulsed" else fmcw_deramp_baseband
            )
            cube = generate(paths, radar, n_pulses, rng=rng)
            if scenario.is_bistatic:
                np.testing.assert_array_equal(
                    cube, old.iq[burst], err_msg="bistatic cube: bit-identical"
                )
            else:
                np.testing.assert_allclose(
                    cube,
                    old.iq[burst],
                    rtol=1e-12,
                    atol=0.0,
                    err_msg="monostatic cube: rtol 1e-12, not bit-identical, because the power "
                    "is now R_T^2 R_R^2 where it was R^4, which rounds differently",
                )
