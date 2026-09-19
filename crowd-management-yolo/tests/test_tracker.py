import cv2
import numpy as np

from crowd_management.tracker import HeadTracker, histogram_embedding
from tests.helpers import make_head

W, H = 320, 240
RED, BLUE = (0, 0, 220), (220, 0, 0)


def render(objects):
    """Grey frame with a coloured square for each (cx, cy, colour)."""
    frame = np.full((H, W, 3), 110, dtype=np.uint8)
    for cx, cy, colour in objects:
        cv2.rectangle(frame, (cx - 10, cy - 10), (cx + 10, cy + 10), colour, -1)
    return frame


def nearest_id(tracks, cx, cy):
    best = min(tracks, key=lambda t: (t.center[0] - cx) ** 2 + (t.center[1] - cy) ** 2)
    return best.track_id


def test_histogram_embedding_is_128d_unit_vector_and_colour_sensitive():
    frame = render([(60, 60, RED), (200, 60, BLUE)])
    red = histogram_embedding(frame, (50, 50, 20, 20))
    blue = histogram_embedding(frame, (190, 50, 20, 20))
    assert red.shape == (128,)
    assert np.linalg.norm(red) == np.float32(1.0) or abs(np.linalg.norm(red) - 1) < 1e-5
    assert float(red @ blue) < 0.2  # different colours -> very different histograms


def test_embedding_of_empty_crop_is_safe():
    frame = render([])
    emb = histogram_embedding(frame, (-50, -50, 10, 10))
    assert emb.shape == (128,) and np.isfinite(emb).all()


def test_track_confirmed_only_after_n_init_hits():
    tracker = HeadTracker(n_init=3)
    counts = []
    for _ in range(4):
        frame = render([(100, 100, RED)])
        counts.append(len(tracker.update(frame, [make_head(100, 100)])))
    assert counts == [0, 0, 1, 1]


def test_ids_stay_stable_for_two_crossing_heads():
    tracker = HeadTracker(n_init=3)
    seen = {"red": set(), "blue": set()}
    for i in range(40):
        rx, bx = 40 + 6 * i, 280 - 6 * i  # they pass each other around frame 20
        ry, by = 80, 140
        frame = render([(rx, ry, RED), (bx, by, BLUE)])
        tracks = tracker.update(frame, [make_head(rx, ry), make_head(bx, by)])
        if i >= 3:
            assert len(tracks) == 2
            seen["red"].add(nearest_id(tracks, rx, ry))
            seen["blue"].add(nearest_id(tracks, bx, by))
    assert len(seen["red"]) == 1 and len(seen["blue"]) == 1
    assert seen["red"] != seen["blue"]


def test_identity_survives_a_short_occlusion():
    tracker = HeadTracker(n_init=3, max_age=30)
    ids = set()
    for i in range(30):
        x = 60 + 3 * i
        visible = not (12 <= i < 18)  # detector misses the head for 6 frames
        frame = render([(x, 100, RED)])
        tracks = tracker.update(frame, [make_head(x, 100)] if visible else [])
        if visible and i >= 3:
            ids.update(t.track_id for t in tracks)
    assert len(ids) == 1


def test_reset_clears_tracks():
    tracker = HeadTracker(n_init=1)
    frame = render([(100, 100, RED)])
    tracker.update(frame, [make_head(100, 100)])
    tracker.reset()
    assert tracker.update(frame, []) == []
