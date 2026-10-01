"""Fast screening stage (spec section 24): cheap pre-Cellpose field gate.

Computes low-resolution brightness/occupancy/texture/edge features and emits
one of three provisional labels: CLEARLY_THICK / CLEARLY_THIN / UNCERTAIN.

Honesty notes (spec sections 24, 25, 45):
  * The occupancy/edge gates below are PLACEHOLDER thresholds chosen to be
    deliberately conservative (most fields -> UNCERTAIN) until the paired
    data from this run (screen features + full Cellpose analysis per keyframe)
    lets us calibrate them against real counts/coverage.
  * In this prototype run the screen does NOT veto Cellpose
    (``fast_screen_decides_cellpose: false`` in the config) - we record its
    verdict and cost on every accepted field so its future value can be
    measured, not assumed.
"""

from __future__ import annotations

import time
from typing import Dict

import cv2
import numpy as np

CLEARLY_THICK = "CLEARLY_THICK"
CLEARLY_THIN = "CLEARLY_THIN"
UNCERTAIN = "UNCERTAIN"


class FastScreen:
    def __init__(
        self,
        work_width: int = 480,
        thick_occupancy_min: float = 0.62,
        thin_occupancy_max: float = 0.25,
        thick_edge_density_min: float = 0.35,
        thin_edge_density_max: float = 0.08,
    ) -> None:
        self.work_width = int(work_width)
        self.thick_occupancy_min = float(thick_occupancy_min)
        self.thin_occupancy_max = float(thin_occupancy_max)
        self.thick_edge_density_min = float(thick_edge_density_min)
        self.thin_edge_density_max = float(thin_edge_density_max)

    def evaluate(self, frame_bgr: np.ndarray) -> Dict:
        """Return screen features + provisional verdict + own runtime (ms)."""
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        scale = self.work_width / float(w)
        small = cv2.resize(
            frame_bgr, (self.work_width, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # RBCs are darker than the brightfield background: Otsu threshold,
        # foreground = darker side.
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        occupancy = float((blurred < otsu_thresh).mean())

        edges = cv2.Canny(blurred, 50, 150)
        edge_density = float(edges.mean() / 255.0)
        texture_std = float(blurred.std())
        brightness = float(gray.mean())

        thick = (
            occupancy >= self.thick_occupancy_min
            or edge_density >= self.thick_edge_density_min
        )
        thin = (
            occupancy <= self.thin_occupancy_max
            or edge_density <= self.thin_edge_density_max
        )
        if thick and not thin:
            verdict = CLEARLY_THICK
        elif thin and not thick:
            verdict = CLEARLY_THIN
        else:
            verdict = UNCERTAIN

        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "fast_screen_result": verdict,
            "screen_occupancy": occupancy,
            "screen_edge_density": edge_density,
            "screen_texture_std": texture_std,
            "screen_brightness": brightness,
            "screen_time_ms": ms,
        }
