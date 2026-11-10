"""Suite-wide pytest configuration and the fixtures several test modules share.

The Hypothesis workaround below keeps collection fast on Windows. The fixtures build a seeded
generator, two small radars (monostatic and bistatic) and a scenario writer; a fixture builds,
it never asserts. A test directory with fixtures of its own keeps them in its own conftest.py.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from radar_forge.radar import Radar, Receiver, Site, Transmitter
from radar_forge.scenario import Scenario, load_scenario

if sys.platform == "win32":
    # At collection, Hypothesis scans every imported module for constants and
    # resolves the path of each one it cannot rule out as third-party. Its cheap
    # test for "third-party" looks for "/site-packages/", which never matches a
    # Windows path ("\site-packages\"), so every module gets a Path.resolve().
    # Where each filesystem call is slow, that was 12 s of every run. Handing it
    # forward slashes restores the cheap test; the modules it scans are the same.
    # If Hypothesis renames the function, this leaves it alone and the suite is
    # only slower.
    try:
        from hypothesis.internal.conjecture import providers as _providers
    except ImportError:  # pragma: no cover - hypothesis is a dev dependency
        _providers = None

    _original = getattr(_providers, "is_local_module_file", None)
    if _original is not None:

        def _is_local_module_file(path: str) -> bool:
            return bool(_original(path.replace("\\", "/")))

        _providers.is_local_module_file = _is_local_module_file

SEED = 20261105

# A short-range X-band FMCW radar small enough to run in a test: 5 MHz swept in 100 us at
# 10 kHz, deramped at 2 MHz, 64 chirps. Bistatic range bins of c/B = 59.96 m over a
# 11.99 km map; rate bins of lambda PRF / 64 = 4.68 m/s over +-149.9 m/s.
SMALL_FMCW = Transmitter(10.0e9, 5.0e6, 10.0, 20.0, 100.0e-6, 10.0e3)
SMALL_RECEIVER = Receiver(2.0e6, 20.0, 3.0)
N_PULSES = 64
RX_SITE = Site(36.0, -78.9, 100.0)
# A transmitter 1.5 km due east of the receiver at the same height.
TX_SITE = Site(36.0, -78.9 + 1500.0 / 90_163.0, 100.0)


@pytest.fixture
def rng() -> np.random.Generator:
    """One seeded generator per test, so a failure is reproducible."""
    return np.random.default_rng(SEED)


@pytest.fixture
def monostatic_radar() -> Radar:
    """The small FMCW radar, transmitter and receiver at one site."""
    return Radar(SMALL_FMCW, SMALL_RECEIVER, RX_SITE)


@pytest.fixture
def bistatic_radar() -> Radar:
    """The small FMCW radar with its transmitter 1.5 km east of its receiver."""
    return Radar(SMALL_FMCW, SMALL_RECEIVER, TX_SITE, RX_SITE)


def _site(table: str, site: Site) -> str:
    return (
        f"[{table}]\nlatitude_deg = {site.latitude_deg!r}\n"
        f"longitude_deg = {site.longitude_deg!r}\naltitude_m = {site.altitude_m!r}\n"
    )


def scene_toml(
    targets: Sequence[tuple[float, Sequence[float], Sequence[float]]],
    *,
    bistatic: bool = False,
    n_frames: int = 1,
    extra: str = "",
) -> str:
    """Return a scenario TOML of the small radar and constant-velocity targets.

    Each target is ``(rcs_dbsm, position_enu_m, velocity_enu_mps)`` about the receiver.
    """
    tx = SMALL_FMCW
    text = (
        f'[scenario]\nname = "small-scene"\nseed = {SEED}\n\n'
        f"[frames]\nstart_time_s = 0.0\nduration_s = {float(n_frames)}\nframe_rate_hz = 1.0\n\n"
        + _site("site.tx", TX_SITE if bistatic else RX_SITE)
        + (_site("site.rx", RX_SITE) if bistatic else "")
        + f"\n[[burst]]\ncarrier_frequency_hz = {tx.carrier_frequency_hz}\n"
        f"bandwidth_hz = {tx.bandwidth_hz}\ntransmit_power_w = {tx.transmit_power_w}\n"
        f"gain_tx_dbi = {tx.gain_tx_dbi}\nchirp_duration_s = {tx.chirp_duration_s}\n"
        f"prf_hz = {tx.prf_hz}\nsample_rate_hz = {SMALL_RECEIVER.sample_rate_hz}\n"
        f"n_pulses = {N_PULSES}\n\n"
        f"[receiver]\ngain_rx_dbi = {SMALL_RECEIVER.gain_rx_dbi}\n"
        f"noise_figure_db = {SMALL_RECEIVER.noise_figure_db}\n\n"
    )
    for k, (rcs_dbsm, position_m, velocity_mps) in enumerate(targets):
        text += (
            f'[[target]]\nname = "target-{k}"\nrcs_dbsm = {rcs_dbsm}\n'
            f"position_enu_m = {list(position_m)}\nvelocity_enu_mps = {list(velocity_mps)}\n\n"
        )
    return text + extra


@pytest.fixture
def write_scene(tmp_path: Path) -> Callable[..., Scenario]:
    """Return a function that writes :func:`scene_toml`'s scenario and loads it back."""

    def write(*args: object, **kwargs: object) -> Scenario:
        path = tmp_path / "scene.toml"
        path.write_text(scene_toml(*args, **kwargs), encoding="utf-8")  # type: ignore[arg-type]
        return load_scenario(path)

    return write


@pytest.fixture
def one_frame_scenario(write_scene: Callable[..., Scenario]) -> Scenario:
    """One frame of one 10 dBsm target 2 km north, closing at 30 m/s, monostatic."""
    return write_scene([(10.0, (0.0, 2000.0, 0.0), (0.0, -30.0, 0.0))])
