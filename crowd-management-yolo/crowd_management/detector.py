"""YOLOv3 person detection with head-region extraction.

Follows Algorithm 1 of the report: run YOLOv3 through OpenCV's DNN module, keep
COCO class 0 (person), take the upper 30% of each person box as the head
region, then apply Non-Maximum Suppression.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

PERSON_CLASS_ID = 0  # COCO class 0


@dataclass
class Detection:
    """One detected head. Boxes are (x, y, w, h) in pixels."""

    head_xywh: Tuple[int, int, int, int]
    person_xywh: Tuple[int, int, int, int]
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.head_xywh
        return x + w // 2, y + h // 2


def head_box_from_person(x: int, y: int, w: int, h: int, ratio: float = 0.3) -> Tuple[int, int, int, int]:
    """head_box = (x, y, x + w, y + floor(ratio * h)), returned as (x, y, w, h)."""
    head_h = max(1, int(math.floor(ratio * h)))
    return x, y, w, head_h


def parse_outputs(
    outputs: Sequence[np.ndarray],
    frame_w: int,
    frame_h: int,
    conf_threshold: float = 0.4,
    nms_threshold: float = 0.4,
    head_ratio: float = 0.3,
) -> List[Detection]:
    """Turn raw YOLOv3 layer outputs into head detections.

    Each output row is [cx, cy, w, h, objectness, class scores...] with
    coordinates relative to the frame size.
    """
    if len(outputs) == 0:
        return []
    rows = np.vstack([np.asarray(o).reshape(-1, np.asarray(o).shape[-1]) for o in outputs])
    if rows.size == 0:
        return []

    scores = rows[:, 5:]
    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]
    keep = (class_ids == PERSON_CLASS_ID) & (confidences > conf_threshold)
    if not keep.any():
        return []

    candidates: List[Detection] = []
    for row, conf in zip(rows[keep], confidences[keep]):
        cx, cy, bw, bh = row[0] * frame_w, row[1] * frame_h, row[2] * frame_w, row[3] * frame_h
        x = int(round(cx - bw / 2))
        y = int(round(cy - bh / 2))
        w, h = int(round(bw)), int(round(bh))
        # Clip the person box to the frame.
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        person = (x1, y1, x2 - x1, y2 - y1)
        head = head_box_from_person(*person, ratio=head_ratio)
        candidates.append(Detection(head_xywh=head, person_xywh=person, confidence=float(conf)))

    if not candidates:
        return []

    # NMS runs on the head boxes: they overlap far less than full-body boxes,
    # so people standing close together are not merged.
    boxes = [list(d.head_xywh) for d in candidates]
    scores_list = [d.confidence for d in candidates]
    indices = cv2.dnn.NMSBoxes(boxes, scores_list, conf_threshold, nms_threshold)
    indices = np.array(indices).flatten().astype(int) if len(indices) else np.array([], dtype=int)
    return [candidates[i] for i in indices]


class YoloV3HeadDetector:
    """Detects heads by running YOLOv3 (Darknet weights) via cv2.dnn."""

    def __init__(
        self,
        cfg_path: str = "models/yolov3.cfg",
        weights_path: str = "models/yolov3.weights",
        input_size: int = 416,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.4,
        head_ratio: float = 0.3,
        use_cuda: bool = False,
    ):
        for path in (cfg_path, weights_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing YOLOv3 file: {path}\n"
                    "Run `python scripts/download_weights.py` to fetch the model files."
                )
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.head_ratio = head_ratio

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        return parse_outputs(
            outputs, w, h, self.conf_threshold, self.nms_threshold, self.head_ratio
        )
