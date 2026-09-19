"""Three-level alerting with cooldown, history logging and export."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Dict, List, Optional

from .zones import Status, ZoneState


class Level(IntEnum):
    INFO = 0
    WARNING = 1
    DANGER = 2


_STATUS_TO_LEVEL = {Status.SAFE: Level.INFO, Status.WARNING: Level.WARNING, Status.DANGER: Level.DANGER}


@dataclass
class Alert:
    timestamp: float
    zone: str
    level: str
    count: int
    capacity: int
    message: str


class AlertManager:
    """Turns zone states into alerts.

    * Escalations (SAFE -> WARNING -> DANGER) alert immediately.
    * A condition that persists re-alerts only after `cooldown` seconds, which
      prevents alert fatigue.
    * Returning to SAFE emits a single INFO alert.
    * Every alert is kept in `history` and appended to a JSON-lines log file.
    """

    def __init__(
        self,
        cooldown: float = 10.0,
        log_path: Optional[str] = None,
        sound: bool = False,
    ):
        self.cooldown = cooldown
        self.log_path = log_path
        self.sound = sound
        self.history: List[Alert] = []
        self._last_level: Dict[str, Level] = {}
        self._last_time: Dict[str, float] = {}
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def reset(self) -> None:
        self._last_level.clear()
        self._last_time.clear()

    def evaluate(self, states: Dict[str, ZoneState], now: Optional[float] = None) -> List[Alert]:
        now = time.time() if now is None else now
        new_alerts: List[Alert] = []
        for name, state in states.items():
            level = _STATUS_TO_LEVEL[state.status]
            previous = self._last_level.get(name, Level.INFO)
            last_time = self._last_time.get(name, float("-inf"))
            alert: Optional[Alert] = None

            if level > previous:  # escalation: always alert
                alert = self._make(now, state, level)
            elif level == previous and level > Level.INFO and now - last_time >= self.cooldown:
                alert = self._make(now, state, level)  # persistent condition, cooldown elapsed
            elif level == Level.INFO and previous > Level.INFO:
                alert = self._make(now, state, level)  # back to normal

            self._last_level[name] = level
            if alert:
                self._last_time[name] = now
                self._record(alert)
                new_alerts.append(alert)
        return new_alerts

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _make(now: float, state: ZoneState, level: Level) -> Alert:
        pct = round(100 * state.occupancy)
        if level == Level.INFO:
            text = f"{state.name} back to normal ({state.count}/{state.capacity} heads, {pct}%)"
        else:
            text = f"{level.name}: {state.name} has {state.count}/{state.capacity} heads ({pct}% of capacity)"
        return Alert(now, state.name, level.name, state.count, state.capacity, text)

    def _record(self, alert: Alert) -> None:
        self.history.append(alert)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(alert)) + "\n")
        if self.sound and alert.level != Level.INFO.name:
            self._beep(alert.level)

    @staticmethod
    def _beep(level: str) -> None:
        try:
            import winsound  # Windows only

            winsound.Beep(1500 if level == Level.DANGER.name else 900, 300)
        except ImportError:
            sys.stdout.write("\a")
            sys.stdout.flush()

    def export_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "zone", "level", "count", "capacity", "message"])
            for a in self.history:
                writer.writerow([a.timestamp, a.zone, a.level, a.count, a.capacity, a.message])
