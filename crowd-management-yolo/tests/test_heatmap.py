import numpy as np
import pytest

from crowd_management.heatmap import DensityHeatmap


def test_single_head_peaks_near_one_at_its_position():
    hm = DensityHeatmap((100, 100), alpha=0.95, sigma=5)
    hm.update([(50, 50)])
    assert hm.heat[50, 50] == pytest.approx(1.0, abs=0.1)
    assert hm.heat[50, 50] == hm.heat.max()
    assert hm.heat[0, 0] < 1e-3


def test_heat_decays_without_detections():
    hm = DensityHeatmap((100, 100), alpha=0.9, sigma=5)
    hm.update([(50, 50)])
    before = hm.heat[50, 50]
    hm.update([])
    assert hm.heat[50, 50] == pytest.approx(0.9 * before, rel=1e-4)


def test_heat_accumulates_for_stationary_heads():
    hm = DensityHeatmap((100, 100), alpha=0.95, sigma=5)
    for _ in range(80):
        hm.update([(50, 50)])
    assert hm.heat[50, 50] > 10  # steady state approaches 1 / (1 - 0.95) = 20


def test_out_of_frame_points_are_ignored():
    hm = DensityHeatmap((50, 50))
    hm.update([(-5, 10), (10, 500), (999, 999)])
    assert hm.heat.max() == 0


def test_render_leaves_cold_areas_untouched():
    hm = DensityHeatmap((100, 100), sigma=4)
    hm.update([(50, 50)])
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    out = hm.render(frame, opacity=0.5)
    assert out.shape == frame.shape
    assert not np.array_equal(out[50, 50], frame[50, 50])  # hot spot is coloured
    assert np.array_equal(out[0, 0], frame[0, 0])  # far corner untouched


def test_render_of_empty_heatmap_returns_frame():
    hm = DensityHeatmap((20, 20))
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    assert hm.render(frame) is frame


def test_reset():
    hm = DensityHeatmap((50, 50))
    hm.update([(10, 10)])
    hm.reset()
    assert hm.heat.max() == 0
