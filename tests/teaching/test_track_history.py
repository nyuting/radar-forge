import csv

import pytest

from radar_forge.teaching.scopes.track_history import render_track_history


def test_modulo_range_plot_labels_unresolved_absolute_range(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    path = tmp_path / "tracks.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["time_s", "track_id", "range_m", "radial_velocity_mps", "range_interpretation"]
        )
        writer.writerow([0, "1", 5900, -20, "modulo_absolute_unresolved"])
        writer.writerow([1, "1", 5920, -20, "modulo_absolute_unresolved"])
    figure = render_track_history(path)
    assert figure.axes[0].get_ylabel() == "Modulo range (m)"
    assert "unresolved" in figure.axes[0].get_title()
    plt.close(figure)
