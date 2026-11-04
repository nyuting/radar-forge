"""Shared fixtures for the batched tracking package's tests.

``LINEARISATIONS`` is the one list of filters. The drift guards and the consistency tests of
``spec/tracker-001.md`` §3 take the ``linearisation`` fixture, so a new linearisation joins
every guard by being added here once.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_forge.tracking.filters import Linearisation, TaylorLinearisation, UnscentedTransform

SEED = 20261104

LINEARISATIONS: dict[str, Linearisation] = {
    "taylor": TaylorLinearisation(),
    "unscented": UnscentedTransform(),
}


@pytest.fixture(scope="session", params=list(LINEARISATIONS.values()), ids=list(LINEARISATIONS))
def linearisation(request: pytest.FixtureRequest) -> Linearisation:
    """Each linearisation in turn: the KF/EKF, then the UKF."""
    value: Linearisation = request.param
    return value


@pytest.fixture
def rng() -> np.random.Generator:
    """One seeded generator per test, so a failure is reproducible."""
    return np.random.default_rng(SEED)
