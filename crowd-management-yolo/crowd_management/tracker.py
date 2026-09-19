"""DeepSORT multi-object tracking of heads.

DeepSORT combines a Kalman filter (constant-velocity motion model), deep or
appearance descriptors for re-identification, Hungarian assignment and a
matching cascade with tentative / confirmed / deleted track states. This module
wraps the open-source `deep-sort-realtime` implementation and adds a
lightweight appearance option that needs no extra model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from .detector import Detection

EMBEDDING_BINS = (8, 4, 4)  # 8 * 4 * 4 = 128 dimensions, like the DeepSORT descriptor


@dataclass
class Track:
    track_id: int
    ltrb: Tuple[int, int, int, int]

    @property
    def center(self) -> Tuple[int, int]:
        l, t, r, b = self.ltrb
        return (l + r) // 2, (t + b) // 2


def histogram_embedding(frame: np.ndarray, ltwh: Sequence[int]) -> np.ndarray:
    """128-d HSV colour histogram of a head crop, L2-normalised.

    A dependency-free stand-in for the CNN appearance descriptor. It is weaker
    than a learned embedding but works out of the box.
    """
    x, y, w, h = [int(v) for v in ltwh]
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    dims = int(np.prod(EMBEDDING_BINS))
    if x2 <= x1 or y2 <= y1:
        return np.full(dims, 1.0 / np.sqrt(dims), dtype=np.float32)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(EMBEDDING_BINS), [0, 180, 0, 256, 0, 256])
    vec = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return np.full(dims, 1.0 / np.sqrt(dims), dtype=np.float32)
    return vec / norm


class HeadTracker:
    """Assigns persistent IDs to detected heads."""

    def __init__(
        self,
        appearance: str = "histogram",
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
        max_iou_distance: float = 0.7,
    ):
        if appearance not in ("histogram", "mobilenet"):
            raise ValueError("appearance must be 'histogram' or 'mobilenet'")
        self.appearance = appearance
        embedder = "mobilenet" if appearance == "mobilenet" else None
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            max_iou_distance=max_iou_distance,
            embedder=embedder,
        )

    def update(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        include_coasting: bool = False,
    ) -> List[Track]:
        """Feed one frame's detections and return the confirmed tracks.

        Tracks that were not matched this frame (for example, briefly occluded
        heads) keep their identity internally; they are only returned when
        `include_coasting` is True.
        """
        raw = [(list(d.head_xywh), d.confidence, "head") for d in detections]
        if self.appearance == "histogram":
            embeds = [histogram_embedding(frame, d.head_xywh) for d in detections]
            tracks = self._tracker.update_tracks(raw, embeds=embeds, frame=frame)
        else:
            tracks = self._tracker.update_tracks(raw, frame=frame)

        results: List[Track] = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            if t.time_since_update > 0 and not include_coasting:
                continue
            l, top, r, b = [int(round(v)) for v in t.to_ltrb()]
            results.append(Track(track_id=int(t.track_id), ltrb=(l, top, r, b)))
        return results

    def reset(self) -> None:
        self._tracker.delete_all_tracks()
