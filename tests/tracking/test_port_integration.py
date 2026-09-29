"""Check coexistence and multi-sensor updates through the relocated public API."""

import subprocess
import sys

import numpy as np

from radar_forge.pipelines.tracking_config import build_enu_tracker
from radar_forge.tracking import Sensor, TrackStatus


def test_imported_and_existing_apis_coexist_without_optional_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from pathlib import Path; "
            "import radar_forge.tracking as general; "
            "import radar_forge.core.tracking as original; "
            "import radar_forge.pipelines as pipelines; "
            "assert all(hasattr(general, name) for name in general.__all__); "
            "assert all(hasattr(pipelines, name) for name in pipelines.__all__); "
            "assert general.Track is not original.Track; "
            "assert pipelines.GeneralScenarioTracker is not pipelines.ScenarioTracker; "
            "assert pipelines.GeneralTrackingConfig is not pipelines.TrackingConfig; "
            "assert 'matplotlib' not in sys.modules; "
            "assert not any('DEPRECATED' in str(getattr(m, '__file__', '')) "
            "for n, m in sys.modules.items() if n.startswith('radar_forge')); "
            "assert Path(general.__file__).parent.name == 'tracking'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_two_sensors_update_both_crossing_tracks():
    engine = build_enu_tracker(
        {"x": "CV"}, origin_lla_deg_m=(36.00250, -78.94100, 60.0), noise_density=0.01
    )
    engine.sensors["second"] = Sensor("second", ("measurement",))
    identifiers = None
    # Alternate asynchronous scans; reverse measurement order on the second
    # sensor so association must follow predicted motion, not list positions.
    for scan in range(24):
        time_s = scan / 2
        sensor = engine.sensors["sensor" if scan % 2 == 0 else "second"]
        positions = [100 + 3 * time_s, 131 - 3 * time_s]
        if scan % 2:
            positions.reverse()
        snapshots = engine.process(
            sensor.batch(
                time_s,
                [("measurement", np.array([p]), np.array([[0.1]])) for p in positions],
            )
        )
        if identifiers is None:
            identifiers = [snapshot.track_id for snapshot in snapshots]
        assert [snapshot.track_id for snapshot in snapshots] == identifiers
    assert len(snapshots) == 2
    for snapshot in snapshots:
        assert snapshot.status == TrackStatus.CONFIRMED
        assert snapshot.source_sensor_ids == frozenset({"sensor", "second"})
    np.testing.assert_allclose([s.state[1] for s in snapshots], [3, -3], rtol=0, atol=0.1)
