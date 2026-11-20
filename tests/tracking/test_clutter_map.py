"""Tests for radar_forge.tracking.clutter_map: the clutter density learnt from the plots.

Ground truth is the uniform density the plots are drawn at, and the prior and floor the map
is built to hold.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from radar_forge.plot_simulator import ClutterComponent, poisson_clutter_plots
from radar_forge.tracking.clutter_map import ClutterMap

SEED = 20261106


def test_the_clutter_map_learns_a_uniform_density() -> None:
    """Uniform clutter at 0.02 per m rad for 60 scans: the map's mean estimate within 3 %.

    The estimate over 1440 cells averages ~ 1.7 plots a cell a scan; 3 % is over five standard
    errors of that mean.
    """
    rng = np.random.default_rng(SEED)
    layer = [ClutterComponent(0.02, 0.0, 20e3)]
    clutter_map = ClutterMap(range_max_m=20e3)
    # Sixty scans, each folded in turn.
    for _ in range(60):
        ranges, azimuths, _ = poisson_clutter_plots(layer, 0.0, rng)
        clutter_map.update(ranges, azimuths)
    centres_r, centres_az = np.meshgrid(
        np.arange(500.0, 20e3, 1e3), np.arange(72) * math.radians(5.0) + 0.01
    )
    estimate = clutter_map.density(centres_r.ravel(), centres_az.ravel())
    assert estimate.mean() == pytest.approx(0.02, rel=0.03)


def test_an_empty_clutter_map_holds_its_prior_then_its_floor() -> None:
    """Before any scan: the prior; a cell that never saw a plot later: the forgetting floor."""
    clutter_map = ClutterMap(prior_per_m_rad=0.02)
    assert clutter_map.density([5e3], [1.0])[0] == pytest.approx(0.02, rel=1e-12)
    for _ in range(200):
        clutter_map.update(np.zeros(0), np.zeros(0))
    assert clutter_map.density([5e3], [1.0])[0] == pytest.approx(
        clutter_map.floor_per_m_rad, rel=1e-12
    )
