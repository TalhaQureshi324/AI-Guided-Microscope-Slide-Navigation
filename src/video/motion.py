"""Microscope motion estimation between consecutive frames (spec section 7).

Method: **phase correlation** (`cv2.phaseCorrelate`) on downscaled grayscale
frames - the simplest reliable translational-registration method, robust to
uniform illumination changes and fast enough to run on every raw frame.

Sign convention (verified empirically): ``displacement(prev, cur)`` returns the
content displacement from *prev* to *cur* in full-resolution pixels, i.e. +dx
means the scene content moved to the right. Returned values are estimated on
``work_width``-wide frames and rescaled to full resolution.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class MotionEstimator:
    """Prepares small grayscale frames and measures frame-to-frame shift."""

    def __init__(self, work_width: int = 480) -> None:
        self.work_width = int(work_width)

    def prep(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Downscale + grayscale + float32 for phase correlation."""
        h, w = frame_bgr.shape[:2]
        scale = self.work_width / float(w)
        small = cv2.resize(
            frame_bgr,
            (self.work_width, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def displacement(
        self, prev_small: np.ndarray, cur_small: np.ndarray, full_width: int
    ) -> Tuple[float, float, float]:
        """Return (dx, dy, response) in FULL-RESOLUTION pixels.

        ``response`` (0..1) is the phase-correlation confidence; low values
        indicate unreliable estimates (e.g. heavy motion blur).
        """
        (sx, sy), response = cv2.phaseCorrelate(prev_small, cur_small)
        scale = full_width / float(self.work_width)
        return float(sx) * scale, float(sy) * scale, float(response)
