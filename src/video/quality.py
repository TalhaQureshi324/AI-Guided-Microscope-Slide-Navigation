"""Lightweight per-frame image-quality measurements (spec section 10).

All measures are cheap enough to run on every raw video frame. They are
recorded for every frame (even skipped ones) so blur / exposure thresholds can
be re-tuned offline without re-decoding the video.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def frame_quality(frame_bgr: np.ndarray) -> Dict[str, float]:
    """Sharpness, brightness, contrast, saturation and exposure extremes.

    ``sharpness_lapvar`` is the variance of the Laplacian of the grayscale
    frame - the classic single-number focus measure. NOTE (calibrated on the
    first prototype video): brightfield microscopy footage has globally low
    Laplacian variance and the measure barely separates sharp from
    motion-blurred frames (stationary median ~15 vs moving median ~16 on
    video_dataset.mp4), so the blur gate built on it is deliberately a *gross*
    blur guard, not a fine focus gate.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return {
        "sharpness_lapvar": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": float(hsv[:, :, 1].mean()),
        "overexposed_pct": float((gray >= 250).mean() * 100.0),
        "underexposed_pct": float((gray <= 5).mean() * 100.0),
    }


def is_blurry(quality: Dict[str, float], blur_threshold: float) -> bool:
    """Gross motion-blur guard: True when sharpness falls below threshold."""
    return bool(quality["sharpness_lapvar"] < blur_threshold)
