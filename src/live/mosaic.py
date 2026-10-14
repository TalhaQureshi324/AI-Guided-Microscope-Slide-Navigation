"""Incremental RGB mosaic builder (Phase 9).

Maintains the stitched microscope image in SCAN/WORLD coordinates, separate
from the semantic monolayer map (semantic colors never enter this canvas).

  * INCREMENTAL: each accepted field is pasted once with a feathered,
    confidence-weighted blend - the canvas is never rebuilt from scratch;
    it only EXPANDS (dynamically, in any direction incl. negative) when a
    field lands outside the current bounds (spec §3/§7);
  * BLENDING (§4/§5): per-pixel weight = rectangular feather (edge ramp)
    x registration confidence. Overlaps become a weighted average, so RBC
    structures are not double-rendered, and weak registrations contribute
    less (registration quality gates how much blending is trusted);
  * COVERAGE MASK: pixels with accumulated weight > 0 are "observed";
    everything else is UNKNOWN/UNSCANNED and is never inferred (§6);
  * SOURCE FIELDS stay intact: placements (job id, pose, size, confidence,
    timestamp) are recorded and the original keyframes remain in the
    session - the mosaic is always reproducible (§2);
  * OVERVIEW: a cached downscaled copy for the GUI mini-map (§8); GUI zoom
    only changes rendering, coordinates stay scan/world (§6).

Exports (§9): mosaic.png, coverage_mask.png, placements.json per session.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class MosaicBuilder:
    """Thread-safe world-coordinate RGB mosaic with weighted blending."""

    def __init__(self, feather_fraction: float = 0.15,
                 overview_max_width: int = 560) -> None:
        self._lock = threading.Lock()
        self._feather_fraction = float(feather_fraction)
        self._overview_max = int(overview_max_width)
        self._origin: Optional[Tuple[float, float]] = None  # scan coords of px (0,0)
        self._canvas: Optional[np.ndarray] = None           # uint8 HxWx3
        self._wsum: Optional[np.ndarray] = None             # float32 HxW
        self._placements: List[Dict] = []
        self._dirty = True
        self._overview_cache: Optional[Tuple[np.ndarray, float, float, float]] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _feather_weight(w: int, h: int, frac: float) -> np.ndarray:
        """Rectangular edge-feather weight map in [~0, 1] (center = 1)."""
        fpx = max(4.0, w * frac)
        fpy = max(4.0, h * frac)
        rx = np.clip(np.minimum(np.arange(w) + 1.0, w - np.arange(w)) / fpx, 0.0, 1.0)
        ry = np.clip(np.minimum(np.arange(h) + 1.0, h - np.arange(h)) / fpy, 0.0, 1.0)
        return (np.minimum(rx[None, :], ry[:, None]) + 0.02).astype(np.float32)

    def _expand(self, tx0: float, ty0: float, tw: float, th: float) -> None:
        """Grow the canvas so the rect (tx0, ty0, tx0+tw, ty0+th) fits (§7)."""
        tx1, ty1 = tx0 + tw, ty0 + th
        ox, oy = self._origin
        ch, cw = self._canvas.shape[:2]
        nx0, ny0 = min(ox, tx0), min(oy, ty0)
        nx1, ny1 = max(ox + cw, tx1), max(oy + ch, ty1)
        ncw = int(math.ceil(nx1 - nx0))
        nch = int(math.ceil(ny1 - ny0))
        if ncw == cw and nch == ch and nx0 == ox and ny0 == oy:
            return
        nc = np.zeros((nch, ncw, 3), dtype=np.uint8)
        nw = np.zeros((nch, ncw), dtype=np.float32)
        oxc = int(math.floor(ox - nx0))
        oyc = int(math.floor(oy - ny0))
        nc[oyc:oyc + ch, oxc:oxc + cw] = self._canvas
        nw[oyc:oyc + ch, oxc:oxc + cw] = self._wsum
        self._canvas = nc
        self._wsum = nw
        self._origin = (float(nx0), float(ny0))

    # ------------------------------------------------------------------
    def place(self, frame_bgr: np.ndarray, gx: float, gy: float,
              confidence: float = 1.0, job_id: Optional[int] = None) -> Dict:
        """Blend one accepted field at its global pose (spec §1/§4).

        ``(gx, gy)`` is the field's registered CENTER in scan coordinates.
        Per-pixel blend weight = edge feather x registration confidence, so
        weak registrations contribute less and seams feather smoothly (§4/§5).
        """
        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        h, w = frame_bgr.shape[:2]
        conf = float(min(1.0, max(0.05, confidence)))
        with self._lock:
            if self._canvas is None:
                self._origin = (gx - w / 2.0, gy - h / 2.0)
                self._canvas = np.zeros((h, w, 3), dtype=np.uint8)
                self._wsum = np.zeros((h, w), dtype=np.float32)
            tx0, ty0 = gx - w / 2.0, gy - h / 2.0
            tx1, ty1 = tx0 + w, ty0 + h
            self._expand(tx0, ty0, tx1 - tx0, ty1 - ty0)

            px0 = int(math.floor(tx0 - self._origin[0]))
            py0 = int(math.floor(ty0 - self._origin[1]))
            # Destination slice derived from the SOURCE size - never from an
            # independent ceil/floor of the span (a fractional pose made
            # ceil(tx1)-floor(tx0) = w+1 and crashed the blend with a
            # (448,801,3) vs (448,800,3) broadcast error). _expand guarantees
            # the canvas covers [px0, px0+w] x [py0, py0+h].
            region = self._canvas[py0:py0 + h, px0:px0 + w]
            rw = self._wsum[py0:py0 + h, px0:px0 + w]
            assert region.shape[:2] == (h, w) and rw.shape[:2] == (h, w), \
                f"mosaic placement slice {region.shape} != frame {(h, w)}"
            tw = self._feather_weight(w, h, self._feather_fraction) * conf

            blended = (region.astype(np.float32) * rw[..., None]
                       + frame_bgr.astype(np.float32) * tw[..., None])
            self._canvas[py0:py0 + h, px0:px0 + w] = np.clip(
                blended / (rw[..., None] + tw[..., None] + 1e-6), 0, 255
            ).astype(np.uint8)
            self._wsum[py0:py0 + h, px0:px0 + w] = rw + tw

            placement = {
                "job_id": job_id,
                "x": round(float(gx), 1), "y": round(float(gy), 1),
                "w": w, "h": h, "confidence": round(conf, 3),
                "timestamp": time.time(),
            }
            self._placements.append(placement)
            self._dirty = True
            return placement

    # ------------------------------------------------------------------
    def coverage_mask(self) -> Optional[np.ndarray]:
        """Boolean observed-region mask (True = scanned pixels)."""
        with self._lock:
            if self._canvas is None:
                return None
            return self._wsum > 0.02

    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Scan-coordinate rect of the canvas (x0, y0, x1, y1)."""
        with self._lock:
            if self._canvas is None or self._origin is None:
                return None
            h, w = self._canvas.shape[:2]
            return (self._origin[0], self._origin[1],
                    self._origin[0] + w, self._origin[1] + h)

    def placements(self) -> List[Dict]:
        with self._lock:
            return list(self._placements)

    def canvas_snapshot(self) -> Optional[Tuple[np.ndarray, float, float]]:
        """Full-resolution copy of the canvas + scan origin (Phase 13
        relocalization target). Copy is taken under the lock."""
        with self._lock:
            if self._canvas is None or self._origin is None:
                return None
            return (self._canvas.copy(), self._origin[0], self._origin[1])

    def reset(self) -> None:
        with self._lock:
            self._canvas = None
            self._wsum = None
            self._origin = None
            self._placements.clear()
            self._overview_cache = None
            self._dirty = True

    # ------------------------------------------------------------------
    def overview(self, force: bool = False
                 ) -> Optional[Tuple[np.ndarray, float, float, float]]:
        """Cached downsampled overview: (rgb, origin_x, origin_y,
        scan_px_per_overview_px). Recomputed only when the canvas changed."""
        with self._lock:
            if self._canvas is None:
                return None
            if not self._dirty and self._overview_cache is not None and not force:
                cached = self._overview_cache
                return cached
            ch, cw = self._canvas.shape[:2]
            scale = min(1.0, self._overview_max / float(cw))
            small = cv2.resize(self._canvas,
                               (max(1, int(round(cw * scale))),
                                max(1, int(round(ch * scale)))),
                               interpolation=cv2.INTER_AREA)
            scan_per_ov = 1.0 / (small.shape[1] / float(cw)) if small.shape[1] else 1.0
            scan_per_ov = cw / float(small.shape[1])
            ox, oy = self._origin
            self._overview_cache = (small, ox, oy, scan_per_ov)
            self._dirty = False
            return self._overview_cache

    def mosaic_rect(self) -> Optional[Tuple[float, float, float, float]]:
        """Scan-rect of the current canvas (for GUI placement)."""
        b = self.bounds()
        return b

    # ------------------------------------------------------------------
    def export(self, out_dir: Path) -> Dict[str, str]:
        """Write mosaic.png + coverage_mask.png + placements.json (§9)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}
        with self._lock:
            if self._canvas is not None:
                p = out_dir / "mosaic.png"
                cv2.imwrite(str(p), cv2.cvtColor(self._canvas, cv2.COLOR_RGB2BGR))
                paths["mosaic"] = str(p)
            if self._wsum is not None:
                p = out_dir / "coverage_mask.png"
                cv2.imwrite(str(p), (self._wsum > 0.02).astype(np.uint8) * 255)
                paths["coverage_mask"] = str(p)
            p = out_dir / "placements.json"
            p.write_text(json.dumps(self._placements, indent=2), encoding="utf-8")
            paths["placements"] = str(p)
        return paths
