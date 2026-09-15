"""Acceptance tests for scenario 001: the range-Doppler peak against the truth.

This is the criterion the whole vertical slice exists to satisfy, spec S9. For
each variant, and for every frame, the brightest range-Doppler cell must fall
where the truth labels say it must -- folded in Doppler for S1, folded in range
for S2, and unfolded for S3 once the dual-PRF pair has been combined.

The point is not that a peak exists. It is that the *specific* discrepancy
between the map and the truth is the one the waveform's ambiguities predict. A
sign error in the Doppler chain, or a missing group-delay trim in the pulsed
one, produces a map that looks entirely plausible and fails here.

Marked slow: it synthesises and processes real IQ cubes. A five-frame window
keeps `make check` quick while still running every frame of it.

The window at 2646 s is chosen because the target is closing at about 50 m/s
there, which is past *both* S3 bursts' unambiguous velocities and about 6.6 folds
of S1's, while the range of roughly 14 km is 2.3 unambiguous ranges of S2. One
window therefore exercises all three foldings; the shipped default window at
t = 0 does not fold S3 at all.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from radar_forge.core.ambiguity import fold_velocity_mps, unfold_doppler_dual_prf
from radar_forge.pipelines.scenarios import (
    Scenario,
    burst_range_doppler,
    iterate_frames,
    load_scenario,
    peak_range_velocity,
)

pytestmark = pytest.mark.slow

SCENARIOS_DIR = Path(__file__).parent.parent.parent / "scenarios"
S1_TOML = SCENARIOS_DIR / "scenario_001_fmcw_low_prf.toml"
S2_TOML = SCENARIOS_DIR / "scenario_001_pulsed_medium_prf.toml"
S3_TOML = SCENARIOS_DIR / "scenario_001_fmcw_dual_prf.toml"

FOLDING_WINDOW_START_S = 2646.0
N_FRAMES = 5


def _windowed(toml_path: Path) -> Scenario:
    """The shipped scenario, over the five-frame window that folds everything."""
    scenario = load_scenario(toml_path)
    return replace(
        scenario,
        start_time_s=FOLDING_WINDOW_START_S,
        duration_s=N_FRAMES / scenario.frame_rate_hz,
    )


def _velocity_bin_mps(burst, n_doppler_bins: int) -> float:
    """Width of one Doppler bin, m/s -- the resolution a peak can be located to."""
    return 2.0 * burst.unambiguous_velocity_mps / n_doppler_bins


def _circular_error(measured: float, expected: float, half_interval: float) -> float:
    """Distance on the folded axis, so a value near each edge is not 'far'."""
    return abs(float(fold_velocity_mps(measured - expected, half_interval)))


class TestS1DopplerFolds:
    """Range unambiguous to 37.5 km; velocity folds into +/-7.65 m/s."""

    def test_the_window_really_does_fold_the_doppler(self) -> None:
        """Guards the premise: a window that did not fold would prove nothing."""
        scenario = _windowed(S1_TOML)
        burst = scenario.bursts[0]
        for frame in iterate_frames(scenario):
            assert abs(frame.radial_velocity_mps) > 4.0 * burst.unambiguous_velocity_mps

    def test_the_peak_lands_at_the_true_range_and_the_folded_velocity(self) -> None:
        scenario = _windowed(S1_TOML)
        burst = scenario.bursts[0]
        for frame in iterate_frames(scenario):
            product = burst_range_doppler(frame.iq[0], burst)
            peak_range_m, peak_velocity_mps = peak_range_velocity(product)

            # Range does not fold here: within one bin of the true range.
            assert abs(peak_range_m - frame.range_m) < burst.range_resolution_m

            expected_mps = float(
                fold_velocity_mps(frame.radial_velocity_mps, burst.unambiguous_velocity_mps)
            )
            velocity_bin_mps = _velocity_bin_mps(burst, product.rd_map.shape[0])
            error_mps = _circular_error(
                peak_velocity_mps, expected_mps, burst.unambiguous_velocity_mps
            )
            assert error_mps <= velocity_bin_mps

    def test_the_peak_velocity_is_nothing_like_the_truth(self) -> None:
        """The lesson, asserted rather than assumed: the map is badly wrong."""
        scenario = _windowed(S1_TOML)
        burst = scenario.bursts[0]
        for frame in iterate_frames(scenario):
            _, peak_velocity_mps = peak_range_velocity(burst_range_doppler(frame.iq[0], burst))
            assert abs(peak_velocity_mps) <= burst.unambiguous_velocity_mps
            assert abs(peak_velocity_mps - frame.radial_velocity_mps) > 30.0


class TestS2RangeFolds:
    """Velocity unambiguous to +/-191 m/s; range folds into 5.996 km."""

    def test_the_window_really_does_fold_the_range(self) -> None:
        scenario = _windowed(S2_TOML)
        burst = scenario.bursts[0]
        for frame in iterate_frames(scenario):
            assert frame.range_m > 2.0 * burst.unambiguous_range_m

    def test_the_peak_lands_at_the_folded_range_and_the_true_velocity(self) -> None:
        scenario = _windowed(S2_TOML)
        burst = scenario.bursts[0]
        for frame in iterate_frames(scenario):
            product = burst_range_doppler(frame.iq[0], burst)
            peak_range_m, peak_velocity_mps = peak_range_velocity(product)

            expected_range_m = frame.range_m % burst.unambiguous_range_m
            # Compared on the folded axis: a target just inside the wrap and a
            # peak just outside it are neighbours, not opposites.
            range_error_m = _circular_error(
                peak_range_m, expected_range_m, 0.5 * burst.unambiguous_range_m
            )
            assert range_error_m < burst.range_resolution_m

            # Velocity does not fold here: it is the true one, to one bin.
            velocity_bin_mps = _velocity_bin_mps(burst, product.rd_map.shape[0])
            assert abs(peak_velocity_mps - frame.radial_velocity_mps) <= velocity_bin_mps

    def test_the_peak_range_is_nothing_like_the_truth(self) -> None:
        """S2 is the exact mirror of S1, and this is the half that goes wrong."""
        scenario = _windowed(S2_TOML)
        burst = scenario.bursts[0]
        for frame in iterate_frames(scenario):
            peak_range_m, _ = peak_range_velocity(burst_range_doppler(frame.iq[0], burst))
            assert peak_range_m < burst.unambiguous_range_m
            assert abs(peak_range_m - frame.range_m) > 5_000.0


class TestS3TheAmbiguityIsResolved:
    """Neither burst can measure the velocity; the coprime pair can."""

    def test_both_bursts_really_do_fold(self) -> None:
        """Otherwise the unfolding below would be tested against nothing."""
        scenario = _windowed(S3_TOML)
        for frame in iterate_frames(scenario):
            for burst in scenario.bursts:
                assert abs(frame.radial_velocity_mps) > burst.unambiguous_velocity_mps

    def test_each_burst_alone_reports_the_wrong_velocity(self) -> None:
        scenario = _windowed(S3_TOML)
        for frame in iterate_frames(scenario):
            for cube, burst in zip(frame.iq, scenario.bursts, strict=True):
                _, peak_velocity_mps = peak_range_velocity(burst_range_doppler(cube, burst))
                assert abs(peak_velocity_mps - frame.radial_velocity_mps) > 10.0

    def test_the_pair_recovers_the_true_unfolded_velocity(self) -> None:
        """The scenario's whole justification, frame by frame."""
        scenario = _windowed(S3_TOML)
        burst_a, burst_b = scenario.bursts
        for frame in iterate_frames(scenario):
            product_a = burst_range_doppler(frame.iq[0], burst_a)
            product_b = burst_range_doppler(frame.iq[1], burst_b)
            _, folded_a_mps = peak_range_velocity(product_a)
            _, folded_b_mps = peak_range_velocity(product_b)

            bin_a_mps = _velocity_bin_mps(burst_a, product_a.rd_map.shape[0])
            bin_b_mps = _velocity_bin_mps(burst_b, product_b.rd_map.shape[0])

            velocity_mps, residual_mps = unfold_doppler_dual_prf(
                folded_a_mps,
                folded_b_mps,
                burst_a.unambiguous_velocity_mps,
                burst_b.unambiguous_velocity_mps,
                max_velocity_mps=191.0,
                tolerance_mps=2.0 * max(bin_a_mps, bin_b_mps),
            )
            assert not np.isnan(velocity_mps), f"frame {frame.index} did not resolve"
            assert float(residual_mps) <= 2.0 * max(bin_a_mps, bin_b_mps)
            # Within one bin of the burst the candidate came from, which is burst A.
            assert abs(float(velocity_mps) - frame.radial_velocity_mps) <= bin_a_mps

    def test_both_bursts_agree_on_the_range(self) -> None:
        """Range is unambiguous on both bursts, so they must see the same target."""
        scenario = _windowed(S3_TOML)
        for frame in iterate_frames(scenario):
            ranges_m = [
                peak_range_velocity(burst_range_doppler(cube, burst))[0]
                for cube, burst in zip(frame.iq, scenario.bursts, strict=True)
            ]
            assert abs(ranges_m[0] - ranges_m[1]) < scenario.bursts[0].range_resolution_m
            assert abs(ranges_m[0] - frame.range_m) < scenario.bursts[0].range_resolution_m


class TestTheShippedDefaultWindowRuns:
    """The configuration as committed, not only the window the tests choose."""

    @pytest.mark.parametrize("toml_path", [S1_TOML, S2_TOML, S3_TOML], ids=["s1", "s2", "s3"])
    def test_the_first_frames_of_the_default_window_find_the_target(self, toml_path: Path) -> None:
        scenario = load_scenario(toml_path)
        short = replace(scenario, duration_s=2.0 / scenario.frame_rate_hz)
        for frame in iterate_frames(short):
            for cube, burst in zip(frame.iq, scenario.bursts, strict=True):
                product = burst_range_doppler(cube, burst)
                peak_range_m, _ = peak_range_velocity(product)
                expected_range_m = frame.range_m % burst.unambiguous_range_m
                range_error_m = _circular_error(
                    peak_range_m, expected_range_m, 0.5 * burst.unambiguous_range_m
                )
                assert range_error_m < burst.range_resolution_m
