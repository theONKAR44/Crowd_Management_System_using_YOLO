"""Polygonal monitoring zones with capacity limits and threshold-based status."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


class Status(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    DANGER = "DANGER"


@dataclass
class ZoneState:
    """Result of evaluating one zone for one frame."""

    name: str
    count: int
    capacity: int
    status: Status
    density: Optional[float] = None  # persons per square metre, if the area is known

    @property
    def occupancy(self) -> float:
        return self.count / self.capacity if self.capacity else 0.0


@dataclass
class Zone:
    name: str
    polygon: np.ndarray  # (N, 2) int32 vertices in frame coordinates
    capacity: int = 30
    warning_ratio: float = 0.6
    danger_ratio: float = 0.8
    area_m2: Optional[float] = None  # real-world area, enables density in persons/m^2

    def __post_init__(self) -> None:
        self.polygon = np.asarray(self.polygon, dtype=np.int32).reshape(-1, 2)
        if len(self.polygon) < 3:
            raise ValueError("A zone needs at least 3 vertices")

    @property
    def t_warning(self) -> float:
        return self.warning_ratio * self.capacity

    @property
    def t_danger(self) -> float:
        return self.danger_ratio * self.capacity

    def contains(self, point: Point) -> bool:
        """Point-in-polygon test (points on the boundary count as inside)."""
        contour = self.polygon.astype(np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0

    def status_for(self, count: int) -> Status:
        """Safe below T_warning, Warning up to T_danger, Danger at or above T_danger."""
        if count < self.t_warning:
            return Status.SAFE
        if count < self.t_danger:
            return Status.WARNING
        return Status.DANGER

    def evaluate(self, points: Iterable[Point]) -> ZoneState:
        count = sum(1 for p in points if self.contains(p))
        density = count / self.area_m2 if self.area_m2 else None
        return ZoneState(self.name, count, self.capacity, self.status_for(count), density)

    def to_dict(self) -> dict:
        data = {
            "name": self.name,
            "polygon": self.polygon.tolist(),
            "capacity": self.capacity,
            "warning_ratio": self.warning_ratio,
            "danger_ratio": self.danger_ratio,
        }
        if self.area_m2 is not None:
            data["area_m2"] = self.area_m2
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        return cls(
            name=data["name"],
            polygon=np.array(data["polygon"]),
            capacity=int(data.get("capacity", 30)),
            warning_ratio=float(data.get("warning_ratio", 0.6)),
            danger_ratio=float(data.get("danger_ratio", 0.8)),
            area_m2=data.get("area_m2"),
        )


class ZoneManager:
    """Owns the list of zones and evaluates them against head positions."""

    def __init__(self, zones: Optional[Sequence[Zone]] = None):
        self.zones: List[Zone] = list(zones or [])

    def add(self, zone: Zone) -> None:
        self.zones.append(zone)

    def clear(self) -> None:
        self.zones.clear()

    def ensure_default(
        self, frame_shape: Tuple[int, ...], capacity: int, warning: float, danger: float
    ) -> None:
        """If no zones are defined, monitor the whole frame as a single zone."""
        if self.zones:
            return
        h, w = frame_shape[:2]
        margin = 5
        polygon = [(margin, margin), (w - margin, margin), (w - margin, h - margin), (margin, h - margin)]
        self.add(Zone("Full Frame", np.array(polygon), capacity, warning, danger))

    def evaluate(self, points: Sequence[Point]) -> Dict[str, ZoneState]:
        return {zone.name: zone.evaluate(points) for zone in self.zones}

    def unique_name(self, base: str = "Zone") -> str:
        existing = {z.name for z in self.zones}
        i = 1
        while f"{base} {i}" in existing:
            i += 1
        return f"{base} {i}"

    # ---- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"zones": [z.to_dict() for z in self.zones]}, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ZoneManager":
        if not os.path.isfile(path):
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls([Zone.from_dict(z) for z in data.get("zones", [])])


class ZoneEditor:
    """Collects mouse clicks to define a new polygon zone at runtime."""

    def __init__(self) -> None:
        self.active = False
        self.points: List[Point] = []

    def start(self) -> None:
        self.active = True
        self.points = []

    def add_point(self, x: int, y: int) -> None:
        if self.active:
            self.points.append((int(x), int(y)))

    def undo(self) -> None:
        if self.points:
            self.points.pop()

    def cancel(self) -> None:
        self.active = False
        self.points = []

    def finish(self, manager: ZoneManager, capacity: int, warning: float, danger: float) -> Optional[Zone]:
        """Create the zone if at least three points were clicked."""
        if not self.active or len(self.points) < 3:
            return None
        zone = Zone(manager.unique_name(), np.array(self.points), capacity, warning, danger)
        manager.add(zone)
        self.cancel()
        return zone
