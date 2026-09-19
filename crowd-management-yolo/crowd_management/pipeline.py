"""The real-time processing pipeline (Algorithm 2 of the report).

    frame -> preprocess -> YOLOv3 detect -> DeepSORT update -> zone counts
          -> heatmap update -> alert check -> draw results
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Protocol, Sequence

import numpy as np

from .alerts import Alert, AlertManager
from .config import Config
from .detector import Detection
from .heatmap import DensityHeatmap
from .preprocess import preprocess
from .tracker import HeadTracker, Track
from .visualize import draw_hud, draw_tracks, draw_zone_editor, draw_zones
from .zones import ZoneEditor, ZoneManager, ZoneState

RECENT_ALERT_SECONDS = 8.0


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> Sequence[Detection]: ...


@dataclass
class FrameResult:
    frame: np.ndarray  # annotated frame
    tracks: List[Track]
    states: Dict[str, ZoneState]
    new_alerts: List[Alert]
    fps: float
    total_heads: int = 0


class CrowdMonitor:
    """Runs the full pipeline one frame at a time."""

    def __init__(
        self,
        detector: Detector,
        tracker: HeadTracker,
        zones: ZoneManager,
        alerts: AlertManager,
        config: Config,
    ):
        self.detector = detector
        self.tracker = tracker
        self.zones = zones
        self.alerts = alerts
        self.config = config
        self.heatmap: Optional[DensityHeatmap] = None
        self.show_heatmap = config.heatmap
        self.trails: Dict[int, Deque] = {}
        self._last_seen: Dict[int, int] = {}
        self._frame_index = 0
        self._fps = 0.0
        self._last_tick: Optional[float] = None

    # ---- controls ----------------------------------------------------------
    def reset(self) -> None:
        self.tracker.reset()
        self.alerts.reset()
        self.trails.clear()
        self._last_seen.clear()
        if self.heatmap is not None:
            self.heatmap.reset()

    def toggle_heatmap(self) -> None:
        self.show_heatmap = not self.show_heatmap

    # ---- main entry point --------------------------------------------------
    def process_frame(
        self,
        raw_frame: np.ndarray,
        source_name: str = "",
        editor: Optional[ZoneEditor] = None,
        now: Optional[float] = None,
    ) -> FrameResult:
        cfg = self.config
        now = time.time() if now is None else now

        frame = preprocess(raw_frame, cfg.frame_width, cfg.blur, cfg.clahe)
        self.zones.ensure_default(frame.shape, cfg.default_capacity, cfg.warning_ratio, cfg.danger_ratio)

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(frame, detections)
        centers = [t.center for t in tracks]

        states = self.zones.evaluate(centers)
        self._update_trails(tracks)
        self._update_heatmap(frame.shape, centers)
        new_alerts = self.alerts.evaluate(states, now=now)

        # ---- render ----
        out = frame
        if self.heatmap is not None and self.show_heatmap:
            out = self.heatmap.render(out, cfg.heatmap_opacity)
        if out is frame:
            out = frame.copy()
        draw_zones(out, self.zones, states)
        draw_tracks(out, tracks, self.trails)
        if editor is not None:
            draw_zone_editor(out, editor)
        recent = [a for a in self.alerts.history if now - a.timestamp < RECENT_ALERT_SECONDS and a.level != "INFO"]
        title = "SINGLE ZONE CROWD MONITOR" if len(self.zones.zones) == 1 else "MULTI-ZONE CROWD MONITOR"
        fps = self._tick()
        draw_hud(out, title, source_name, len(tracks), fps, recent, self.show_heatmap)

        return FrameResult(out, tracks, states, new_alerts, fps, len(tracks))

    # ---- internals ---------------------------------------------------------
    def _tick(self) -> float:
        t = time.perf_counter()
        if self._last_tick is not None:
            dt = max(t - self._last_tick, 1e-6)
            inst = 1.0 / dt
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._last_tick = t
        return self._fps

    def _update_trails(self, tracks: Sequence[Track]) -> None:
        self._frame_index += 1
        for t in tracks:
            self.trails.setdefault(t.track_id, deque(maxlen=self.config.trail_length)).append(t.center)
            self._last_seen[t.track_id] = self._frame_index
        # Drop trails of tracks that vanished so memory stays bounded.
        stale = [k for k, seen in self._last_seen.items() if self._frame_index - seen > self.config.max_age]
        for track_id in stale:
            self.trails.pop(track_id, None)
            del self._last_seen[track_id]

    def _update_heatmap(self, shape, centers) -> None:
        if not self.config.heatmap:
            return
        if self.heatmap is None or self.heatmap.shape != tuple(shape[:2]):
            self.heatmap = DensityHeatmap(shape, self.config.heatmap_alpha, self.config.heatmap_sigma)
        self.heatmap.update(centers)
