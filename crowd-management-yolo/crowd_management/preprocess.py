"""Frame pre-processing: resizing, denoising and contrast enhancement."""

from __future__ import annotations

import cv2
import numpy as np


def preprocess(
    frame: np.ndarray,
    width: int = 640,
    blur: bool = False,
    clahe: bool = False,
) -> np.ndarray:
    """Standardise a BGR frame for the detector.

    Resizing keeps the aspect ratio and cuts computation. The BGR->RGB swap and
    0-1 normalisation happen inside cv2.dnn.blobFromImage in the detector.
    """
    h, w = frame.shape[:2]
    if width and w != width:
        scale = width / float(w)
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (width, max(1, int(round(h * scale)))), interpolation=interpolation)

    if blur:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

    if clahe:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        equaliser = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.merge((equaliser.apply(l_channel), a_channel, b_channel))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return frame
