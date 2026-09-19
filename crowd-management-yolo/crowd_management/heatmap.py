"""Cumulative crowd-density heatmap.

Implements  H_{t+1}(x, y) = alpha * H_t(x, y) + sum_i G(x - x_i, y - y_i, sigma)
where G is a Gaussian blob centred on every head position. Because Gaussian
blurring is linear, blurring a frame of unit impulses (one per head) is
equivalent to adding one Gaussian kernel per head, and much faster.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


class DensityHeatmap:
    def __init__(self, shape: Tuple[int, int], alpha: float = 0.95, sigma: float = 15.0):
        self.alpha = alpha
        self.sigma = sigma
        self.heat = np.zeros(shape[:2], dtype=np.float32)
        # Scale so that a single head contributes a peak value of 1.0.
        self._peak_scale = 2.0 * math.pi * sigma * sigma

    @property
    def shape(self) -> Tuple[int, int]:
        return self.heat.shape

    def reset(self) -> None:
        self.heat[:] = 0

    def update(self, points: Iterable[Tuple[int, int]]) -> None:
        h, w = self.heat.shape
        impulses = np.zeros_like(self.heat)
        for x, y in points:
            if 0 <= x < w and 0 <= y < h:
                impulses[y, x] += 1.0
        blobs = cv2.GaussianBlur(impulses, (0, 0), self.sigma) * self._peak_scale
        self.heat = self.alpha * self.heat + blobs

    def render(self, frame: np.ndarray, opacity: float = 0.4, vmax: Optional[float] = None) -> np.ndarray:
        """Overlay the heatmap on a frame. Cold areas are left untouched."""
        peak = float(vmax) if vmax else float(self.heat.max())
        if peak < 1e-6:
            return frame
        norm = np.clip(self.heat / peak, 0.0, 1.0)
        coloured = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 1.0 - opacity, coloured, opacity, 0)
        mask = norm > 0.05
        out = frame.copy()
        out[mask] = blended[mask]
        return out
