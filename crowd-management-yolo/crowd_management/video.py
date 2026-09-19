"""Video input: webcams, video files and network (RTSP/HTTP) streams."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def _parse_source(source: str):
    """A bare integer means a webcam index; anything else is a path or URL."""
    return int(source) if str(source).isdigit() else source


class VideoSource:
    """Thin wrapper around cv2.VideoCapture for a single source."""

    def __init__(self, source: str):
        self.source = str(source)
        self.capture = cv2.VideoCapture(_parse_source(self.source))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

    @property
    def is_live(self) -> bool:
        """True for webcams and network streams, False for files."""
        s = self.source
        return s.isdigit() or s.lower().startswith(("rtsp://", "http://", "https://", "rtmp://"))

    @property
    def fps(self) -> float:
        value = self.capture.get(cv2.CAP_PROP_FPS)
        return float(value) if value and value > 0 else 25.0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


class SourceManager:
    """Holds several sources and lets the operator switch between them."""

    def __init__(self, sources: List[str]):
        if not sources:
            raise ValueError("At least one video source is required")
        self.sources = list(sources)
        self.index = 0
        self.current = VideoSource(self.sources[0])

    @property
    def name(self) -> str:
        return self.current.source

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.current.read()

    def switch(self) -> bool:
        """Move to the next source. Returns False if there is nothing to switch to."""
        if len(self.sources) < 2:
            return False
        for step in range(1, len(self.sources) + 1):
            candidate = (self.index + step) % len(self.sources)
            try:
                new_source = VideoSource(self.sources[candidate])
            except RuntimeError:
                continue
            self.current.release()
            self.current = new_source
            self.index = candidate
            return True
        return False

    def release(self) -> None:
        self.current.release()
