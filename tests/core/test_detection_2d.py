import numpy as np
import pytest

from radar_forge.core.detection_2d import DetectionConfig, cfar_2d


def test_isolated_peaks_and_wrapped_boundaries():
    power = np.ones((64, 64))
    power[0, 0] = 1000
    wrapped = cfar_2d(power, wrap_range=True)
    assert wrapped.peak_indices.tolist() == [[0, 0]]
    assert not cfar_2d(power).mask[0, 0]
    power[0, 32] = 2000
    assert [0, 32] in cfar_2d(power).peak_indices.tolist()


def test_tied_neighbour_peaks_are_deterministic():
    power = np.ones((64, 64))
    power[31:33, 31:33] = 1000
    assert cfar_2d(power).peak_indices.tolist() == [[31, 31]]
    assert cfar_2d(np.zeros((64, 64))).peak_indices.shape == (0, 2)


@pytest.mark.slow
def test_nominal_pfa_on_independent_exponential_noise():
    # Independent realizations avoid assuming overlapping CFAR windows are independent trials.
    # Under H0 the centre power is exponential and independent of its training ring.
    rng = np.random.default_rng(188)
    n_trials = 12000
    config = DetectionConfig(guard_cells=0, training_cells=1, pfa=0.05)
    hits = 0
    for _ in range(n_trials):
        hits += bool(cfar_2d(rng.exponential(size=(3, 3)), config).mask[1, 1])
    expected = n_trials * config.pfa
    # Five binomial standard deviations is a predetermined Monte Carlo acceptance interval.
    assert abs(hits - expected) < 5 * np.sqrt(expected * (1 - config.pfa))


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_invalid_power_rejected(value):
    with pytest.raises(ValueError):
        cfar_2d(np.full((64, 64), value))
