"""Central configuration. Every field can be overridden from the command line."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import List


@dataclass
class Config:
    # --- Input ------------------------------------------------------------
    sources: List[str] = field(default_factory=lambda: ["0"])  # webcam index, file, or RTSP URL
    frame_width: int = 640  # frames are resized to this width (aspect ratio kept)

    # --- Pre-processing (all optional) -------------------------------------
    blur: bool = False  # Gaussian blur noise reduction
    clahe: bool = False  # CLAHE contrast enhancement for poor lighting

    # --- YOLOv3 detection --------------------------------------------------
    yolo_cfg: str = "models/yolov3.cfg"
    yolo_weights: str = "models/yolov3.weights"
    input_size: int = 416  # network input is input_size x input_size
    conf_threshold: float = 0.4
    nms_threshold: float = 0.4
    head_ratio: float = 0.3  # head = upper 30% of the person box
    use_cuda: bool = False

    # --- DeepSORT tracking -------------------------------------------------
    appearance: str = "histogram"  # "histogram" (no extra deps) or "mobilenet" (needs torch)
    max_age: int = 30  # frames a track survives without a match
    n_init: int = 3  # hits required to confirm a track
    max_cosine_distance: float = 0.4
    trail_length: int = 30

    # --- Zones and alerts --------------------------------------------------
    zones_file: str = "configs/zones.json"
    default_capacity: int = 30
    warning_ratio: float = 0.6  # WARNING at 60% of capacity
    danger_ratio: float = 0.8  # DANGER at 80% of capacity
    alert_cooldown: float = 10.0  # seconds between repeats of the same alert
    alert_sound: bool = False
    log_dir: str = "logs"

    # --- Heatmap -----------------------------------------------------------
    heatmap: bool = True
    heatmap_alpha: float = 0.95  # decay factor
    heatmap_sigma: float = 15.0  # Gaussian kernel width in pixels
    heatmap_opacity: float = 0.4

    @classmethod
    def field_names(cls) -> List[str]:
        return [f.name for f in fields(cls)]
