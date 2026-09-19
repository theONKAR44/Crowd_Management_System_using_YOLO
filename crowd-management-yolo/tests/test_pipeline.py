import numpy as np

from crowd_management.alerts import AlertManager
from crowd_management.config import Config
from crowd_management.pipeline import CrowdMonitor
from crowd_management.tracker import HeadTracker
from crowd_management.zones import Status, ZoneManager
from tests.helpers import ScriptedDetector, make_head


def build(tmp_path, heads, capacity=10):
    cfg = Config(frame_width=320, default_capacity=capacity, log_dir=str(tmp_path))
    monitor = CrowdMonitor(
        ScriptedDetector([heads]),
        HeadTracker(n_init=3),
        ZoneManager(),
        AlertManager(cooldown=10, log_path=str(tmp_path / "alerts.jsonl")),
        cfg,
    )
    return monitor


def coloured_frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)  # 640x480, resized to 320x240


def grid(n):
    return [make_head(30 + 35 * (i % 8), 40 + 45 * (i // 8)) for i in range(n)]


def test_full_pipeline_counts_and_raises_danger_alert(tmp_path):
    monitor = build(tmp_path, grid(9), capacity=10)  # 9/10 = 90% -> DANGER
    frame = coloured_frame()
    results = [monitor.process_frame(frame, "test") for _ in range(5)]

    last = results[-1]
    assert last.frame.shape == (240, 320, 3)  # resized to the configured width
    assert last.total_heads == 9
    state = last.states["Full Frame"]
    assert (state.count, state.capacity, state.status) == (9, 10, Status.DANGER)

    all_alerts = [a for r in results for a in r.new_alerts]
    assert [a.level for a in all_alerts] == ["DANGER"]  # one alert, not one per frame
    assert monitor.heatmap is not None and monitor.heatmap.heat.max() > 0


def test_safe_crowd_raises_no_alerts(tmp_path):
    monitor = build(tmp_path, grid(3), capacity=10)
    frame = coloured_frame()
    results = [monitor.process_frame(frame) for _ in range(5)]
    assert results[-1].states["Full Frame"].status == Status.SAFE
    assert all(r.new_alerts == [] for r in results)


def test_warning_band(tmp_path):
    monitor = build(tmp_path, grid(6), capacity=10)  # 60% -> WARNING
    frame = coloured_frame()
    results = [monitor.process_frame(frame) for _ in range(5)]
    assert results[-1].states["Full Frame"].status == Status.WARNING


def test_heatmap_toggle_and_reset(tmp_path):
    monitor = build(tmp_path, grid(4))
    frame = coloured_frame()
    for _ in range(4):
        monitor.process_frame(frame)
    assert monitor.show_heatmap
    monitor.toggle_heatmap()
    assert not monitor.show_heatmap
    monitor.reset()
    assert monitor.heatmap.heat.max() == 0 and monitor.trails == {}


def test_trails_are_recorded_and_bounded(tmp_path):
    monitor = build(tmp_path, grid(2))
    frame = coloured_frame()
    for _ in range(50):
        monitor.process_frame(frame)
    assert monitor.trails
    assert all(len(t) <= monitor.config.trail_length for t in monitor.trails.values())


def test_works_on_tiny_and_grayscale_sized_inputs(tmp_path):
    monitor = build(tmp_path, [])
    tiny = np.zeros((48, 64, 3), dtype=np.uint8)
    result = monitor.process_frame(tiny)
    assert result.total_heads == 0 and result.frame.shape[1] == 320
