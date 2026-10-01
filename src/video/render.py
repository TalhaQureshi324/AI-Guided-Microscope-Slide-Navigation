"""Annotated-video rendering (spec sections 33-34).

Presentation mode (default): a compact, readable HUD panel - class colour
strip, monolayer scores, the key field measurements and per-stage timings -
plus a one-line status for skipped frames. Never draws thousands of instance
IDs into the video (those live in the per-keyframe debug overlays instead).
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

# visualization-only colours (spec: colours must never affect logic)
CLASS_COLOURS_BGR = {
    "TOO_THICK": (60, 60, 230),     # red
    "MONOLAYER": (80, 220, 80),     # green
    "TOO_THIN": (230, 160, 60),     # blue-ish (BGR)
    "UNCERTAIN": (60, 160, 255),    # amber
    "NO_DATA_YET": (160, 160, 160), # grey
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put(img, text, x, y, colour=(240, 240, 240), scale=0.52, thick=1):
    cv2.putText(img, text, (x, y), _FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), _FONT, scale, colour, thick, cv2.LINE_AA)


def draw_field_hud(
    frame_bgr: np.ndarray,
    info: Dict,
    smoothed_score: Optional[float] = None,
    smoothed_class: Optional[str] = None,
) -> np.ndarray:
    """HUD for an ACCEPTED keyframe (info carries field stats + timings)."""
    img = frame_bgr
    cls = info.get("raw_class") or "NO_DATA_YET"
    colour = CLASS_COLOURS_BGR.get(cls, CLASS_COLOURS_BGR["NO_DATA_YET"])

    # left colour strip: the at-a-glance class indicator
    cv2.rectangle(img, (0, 0), (10, img.shape[0]), colour, -1)

    # translucent panel
    panel_w, panel_h = 560, 190
    overlay = img.copy()
    cv2.rectangle(overlay, (16, 14), (16 + panel_w, 14 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    x, y = 28, 38
    _put(img, f"KEYFRAME {info.get('keyframe_index', '?')}  (frame {info.get('frame_index', '?')}, t={info.get('timestamp_sec', 0):.1f}s)", x, y, (200, 220, 255))
    y += 26
    line = f"REGION: {cls}"
    if smoothed_class and smoothed_class != cls:
        line += f"  (smoothed: {smoothed_class})"
    _put(img, line, x, y, colour, 0.62, 1)
    y += 26
    _put(img,
         f"Score: {info.get('raw_monolayer_score', float('nan')):.2f}"
         f"  | smoothed: {(smoothed_score if smoothed_score is not None else float('nan')):.2f}",
         x, y)
    y += 24
    _put(img,
         f"RBCs: {info.get('rbc_candidate_count', '-')} (valid {info.get('valid_region_rbc_count', '-')})"
         f"  density: {info.get('rbc_density', float('nan')):.0f}/Mpx",
         x, y)
    y += 24
    _put(img,
         f"coverage {info.get('coverage', float('nan')) * 100:.0f}%"
         f"  NN/diam {info.get('normalized_median_nn', float('nan')):.2f}"
         f"  contact {info.get('contact_ratio', float('nan')) * 100:.0f}%",
         x, y)
    y += 24
    _put(img,
         f"nbrs: 0:{info.get('pct_0_neighbors', float('nan')):.0f}%"
         f" 1:{info.get('pct_1_neighbor', float('nan')):.0f}%"
         f" 2:{info.get('pct_2_neighbors', float('nan')):.0f}%"
         f" 3+:{info.get('pct_3plus_neighbors', float('nan')):.0f}%"
         f"  deg {info.get('mean_graph_degree', float('nan')):.1f}",
         x, y)
    y += 24
    _put(img,
         f"max cluster {info.get('largest_cluster_fraction', float('nan')) * 100:.0f}%"
         f"  densityCV {info.get('density_cv', float('nan')):.2f}"
         f"  merges {info.get('merge_suspect_count', '-')}",
         x, y)
    y += 24
    _put(img,
         f"cellpose {info.get('cellpose_time_ms', 0):.0f}ms"
         f"  features {info.get('feature_time_ms', 0):.0f}ms"
         f"  total {info.get('total_time_ms', 0):.0f}ms"
         f"  screen:{info.get('fast_screen_result', '-')}",
         x, y, (170, 170, 170), 0.46)
    return img


def draw_skip_hud(
    frame_bgr: np.ndarray,
    frame_index: int,
    status: str,
    last_class: Optional[str],
    last_smoothed_score: Optional[float],
) -> np.ndarray:
    """Minimal one-line HUD for skipped (duplicate/blurry) raw frames."""
    img = frame_bgr
    cv2.rectangle(img, (0, 0), (10, img.shape[0]), (90, 90, 90), -1)
    text = f"frame {frame_index}  {status}"
    if last_class:
        score = f" {last_smoothed_score:.2f}" if last_smoothed_score is not None else ""
        text += f"   last field: {last_class}{score}"
    _put(img, text, 22, 34, (185, 185, 185), 0.5)
    return img
