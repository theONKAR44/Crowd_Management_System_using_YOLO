import numpy as np
import pytest

from crowd_management.zones import Status, Zone, ZoneEditor, ZoneManager

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_point_in_polygon():
    zone = Zone("A", np.array(SQUARE))
    assert zone.contains((50, 50))
    assert zone.contains((0, 50))  # boundary counts as inside
    assert not zone.contains((150, 50))
    assert not zone.contains((-1, 50))


def test_zone_needs_three_vertices():
    with pytest.raises(ValueError):
        Zone("bad", np.array([(0, 0), (1, 1)]))


@pytest.mark.parametrize(
    "count, expected",
    [(0, Status.SAFE), (17, Status.SAFE), (18, Status.WARNING), (23, Status.WARNING),
     (24, Status.DANGER), (40, Status.DANGER)],
)
def test_status_thresholds_60_and_80_percent(count, expected):
    zone = Zone("A", np.array(SQUARE), capacity=30)  # T_warning = 18, T_danger = 24
    assert zone.status_for(count) == expected


def test_evaluate_counts_only_points_inside():
    zone = Zone("A", np.array(SQUARE), capacity=10)
    state = zone.evaluate([(10, 10), (20, 20), (500, 500)])
    assert state.count == 2
    assert state.occupancy == pytest.approx(0.2)
    assert state.density is None


def test_density_when_area_known():
    zone = Zone("A", np.array(SQUARE), capacity=10, area_m2=4.0)
    assert zone.evaluate([(10, 10), (20, 20)]).density == pytest.approx(0.5)


def test_default_zone_covers_frame():
    zm = ZoneManager()
    zm.ensure_default((480, 640, 3), capacity=20, warning=0.6, danger=0.8)
    assert len(zm.zones) == 1
    assert zm.zones[0].contains((320, 240))
    zm.ensure_default((480, 640, 3), 20, 0.6, 0.8)  # idempotent
    assert len(zm.zones) == 1


def test_save_and_load_roundtrip(tmp_path):
    zm = ZoneManager([Zone("Gate", np.array(SQUARE), capacity=42, danger_ratio=0.9, area_m2=12.5)])
    path = str(tmp_path / "zones.json")
    zm.save(path)
    loaded = ZoneManager.load(path)
    assert len(loaded.zones) == 1
    z = loaded.zones[0]
    assert (z.name, z.capacity, z.danger_ratio, z.area_m2) == ("Gate", 42, 0.9, 12.5)
    assert z.polygon.tolist() == [list(p) for p in SQUARE]


def test_load_missing_file_gives_empty_manager(tmp_path):
    assert ZoneManager.load(str(tmp_path / "nope.json")).zones == []


def test_editor_creates_zone_from_clicks():
    zm, editor = ZoneManager(), ZoneEditor()
    editor.start()
    for p in [(10, 10), (90, 10), (50, 80)]:
        editor.add_point(*p)
    zone = editor.finish(zm, 25, 0.6, 0.8)
    assert zone is not None and zone.name == "Zone 1" and zone.capacity == 25
    assert not editor.active


def test_editor_needs_three_points_and_supports_undo_cancel():
    zm, editor = ZoneManager(), ZoneEditor()
    editor.add_point(1, 1)  # ignored: editor not active
    assert editor.points == []
    editor.start()
    editor.add_point(1, 1)
    editor.add_point(2, 2)
    assert editor.finish(zm, 10, 0.6, 0.8) is None
    editor.undo()
    assert len(editor.points) == 1
    editor.cancel()
    assert not editor.active and zm.zones == []


def test_unique_zone_names():
    zm = ZoneManager([Zone("Zone 1", np.array(SQUARE))])
    assert zm.unique_name() == "Zone 2"
