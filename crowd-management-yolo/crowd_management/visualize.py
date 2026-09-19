"""Drawing helpers: tracks, trails, zones, HUD and alert banner."""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, Optional, Sequence

import cv2
import numpy as np

from .alerts import Alert
from .tracker import Track
from .zones import Status, ZoneEditor, ZoneManager, ZoneState

# BGR colours
GREEN = (0, 200, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
CYAN = (255, 200, 0)
STATUS_COLOR = {Status.SAFE: GREEN, Status.WARNING: ORANGE, Status.DANGER: RED}
LEVEL_COLOR = {"INFO": GREEN, "WARNING": ORANGE, "DANGER": RED}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(frame, text, org, scale=0.5, color=WHITE, thickness=1) -> None:
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_tracks(
    frame: np.ndarray,
    tracks: Sequence[Track],
    trails: Optional[Dict[int, deque]] = None,
    show_ids: bool = True,
) -> None:
    for t in tracks:
        if trails and t.track_id in trails and len(trails[t.track_id]) > 1:
            pts = np.array(trails[t.track_id], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, CYAN, 1, cv2.LINE_AA)
        l, top, r, b = t.ltrb
        cx, cy = t.center
        radius = max(6, (r - l) // 2)
        cv2.circle(frame, (cx, cy), radius, GREEN, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, WHITE, -1, cv2.LINE_AA)
        if show_ids:
            _text(frame, str(t.track_id), (cx + radius + 2, cy - radius), 0.4, WHITE)


def draw_zones(frame: np.ndarray, zones: ZoneManager, states: Dict[str, ZoneState]) -> None:
    for zone in zones.zones:
        state = states.get(zone.name)
        color = STATUS_COLOR[state.status] if state else GREEN
        cv2.polylines(frame, [zone.polygon.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        if state:
            label = f"{zone.name}: {state.count}/{state.capacity} ({round(100 * state.occupancy)}%) {state.status.value}"
            # Label sits at the bottom of the zone, clear of the HUD at the top
            # and the key help at the very bottom of the frame.
            x_min, _ = zone.polygon.min(axis=0)
            _, y_max = zone.polygon.max(axis=0)
            label_y = min(int(y_max) - 8, frame.shape[0] - 46)
            _text(frame, label, (int(x_min) + 4, label_y), 0.5, color, 1)
            if state.density is not None:
                _text(frame, f"{state.density:.2f} persons/m2", (int(x_min) + 4, label_y - 18), 0.45, color, 1)


def draw_zone_editor(frame: np.ndarray, editor: ZoneEditor) -> None:
    if not editor.active:
        return
    for p in editor.points:
        cv2.circle(frame, p, 4, CYAN, -1, cv2.LINE_AA)
    if len(editor.points) > 1:
        cv2.polylines(frame, [np.array(editor.points, dtype=np.int32).reshape(-1, 1, 2)], False, CYAN, 2, cv2.LINE_AA)
    _text(frame, "Drawing zone: click points | Enter = finish | Backspace = undo | Esc = cancel",
          (10, frame.shape[0] - 62), 0.45, CYAN)


def draw_hud(
    frame: np.ndarray,
    title: str,
    source: str,
    heads: int,
    fps: float,
    alerts: Iterable[Alert] = (),
    heatmap_on: bool = True,
) -> None:
    h, w = frame.shape[:2]
    _text(frame, title, (10, 24), 0.7, WHITE, 2)
    _text(frame, f"Source: {source}", (10, 46), 0.45, (200, 200, 255))
    _text(frame, f"Heads: {heads}   FPS: {fps:.1f}   Heatmap: {'ON' if heatmap_on else 'OFF'}", (10, 66), 0.5, WHITE)

    alerts = list(alerts)
    if alerts:
        worst = max(alerts, key=lambda a: ("INFO", "WARNING", "DANGER").index(a.level))
        color = LEVEL_COLOR[worst.level]
        banner = f"{worst.level}: {len(alerts)} alert(s)"
        (tw, th), _ = cv2.getTextSize(banner, FONT, 0.6, 2)
        cv2.rectangle(frame, (w - tw - 20, 6), (w - 6, th + 18), color, -1)
        cv2.putText(frame, banner, (w - tw - 13, th + 10), FONT, 0.6, WHITE, 2, cv2.LINE_AA)
        for i, a in enumerate(alerts[-3:]):
            stamp = time.strftime("%H:%M:%S", time.localtime(a.timestamp))
            _text(frame, f"[{stamp}] {a.message}", (10, 92 + 18 * i), 0.45, LEVEL_COLOR[a.level])

    _text(frame, "Q Quit   S Source   R Reset   C Capture   H Heatmap", (10, h - 26), 0.4, CYAN)
    _text(frame, "Z New zone   X Clear zones   + / - Capacity   [ ] Danger %", (10, h - 8), 0.4, CYAN)
