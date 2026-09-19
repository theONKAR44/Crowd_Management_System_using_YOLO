import numpy as np

from crowd_management.detector import Detection


def make_row(cx, cy, w, h, class_id=0, score=0.9, objectness=0.9):
    """One raw YOLOv3 output row: [cx, cy, w, h, objectness, 80 class scores]."""
    row = np.zeros(85, dtype=np.float32)
    row[:5] = [cx, cy, w, h, objectness]
    row[5 + class_id] = score
    return row


def make_head(cx, cy, size=20, confidence=0.9):
    """A Detection whose head box is a size x size square centred on (cx, cy)."""
    x, y = int(cx - size // 2), int(cy - size // 2)
    return Detection(head_xywh=(x, y, size, size), person_xywh=(x, y, size, size * 3), confidence=confidence)


class ScriptedDetector:
    """Detector stub that replays a list of per-frame detections (the last one repeats)."""

    def __init__(self, per_frame):
        self.per_frame = per_frame
        self.calls = 0

    def detect(self, frame):
        idx = min(self.calls, len(self.per_frame) - 1)
        self.calls += 1
        return self.per_frame[idx]
