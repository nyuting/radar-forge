"""Shared fixtures for the tracking tests.

``linearisation`` is the one list of every linearisation a
:class:`~radar_forge.core.tracking.estimation.GaussianFilter` can be built with. Each drift
guard of ``spec/tracker-001.md`` §4.5 (consistency, definiteness) takes it, so a new
linearisation joins every guard by being added here once.
"""

from __future__ import annotations

import pytest

from radar_forge.core.tracking.linearisation import (
    Linearisation,
    TaylorLinearisation,
    UnscentedLinearisation,
)

LINEARISATIONS: dict[str, Linearisation] = {
    "taylor": TaylorLinearisation(),
    "unscented": UnscentedLinearisation(),
}


@pytest.fixture(scope="session", params=list(LINEARISATIONS.values()), ids=list(LINEARISATIONS))
def linearisation(request: pytest.FixtureRequest) -> Linearisation:
    """Each linearisation in turn, so a test runs once per filter: the KF/EKF and the UKF."""
    value: Linearisation = request.param
    return value
