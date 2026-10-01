"""Live viewport rendering (overlays, HUD, ROI, class highlight).

Reuses the project's established overlay STYLE - thin per-instance colour
boundaries, small centroid dots, orange merge-suspect emphasis, optional
instance IDs - by importing the exact helpers from ``src.visualization.overlays``
(the static diagnostic renderer), so the live GUI looks like the project's
existing outputs rather than the default filled Cellpose view.

Performance contract: the per-RESULT heavy layers (boundaries, dots, IDs) are
built ONCE by the analysis worker; the UI tick only composites cached layers
and draws a handful of text lines, keeping the live feed smooth while Cellpose
runs (spec §6/§17/§34).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.visualization.overlays import (
    _label_boundaries,
    _label_colours,
    COLOR_CENTROID,
    COLOR_MERGE,
    COLOR_BORDER,
)

CLASS_COLOURS_BGR = {
    "TOO_THICK": (60, 60, 230),
    "MONOLAYER": (80, 220, 80),
    "TOO_THIN": (230, 160, 60),
    "UNCERTAIN": (60, 160, 255),
    "STALE": (160, 160, 160),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def build_result_layers(labels: np.ndarray, rows: List[Dict],
                        debug_ids: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build (overlay_layer, id_layer) once per finished analysis.

    overlay_layer: thin boundaries + centroid dots + merge-suspect emphasis.
    id_layer: instance IDs for debug mode (None when ``debug_ids`` is False).
    """
    h, w = labels.shape[:2]
    layer = np.zeros((h, w, 3), dtype=np.uint8)
    max_label = int(labels.max()) if labels.size else 0
    merge_ids = {r["instance_id"] for r in rows if r.get("possible_merged_rbc")}

    if max_label > 0:
        lut = _label_colours(max_label)
        edges = _label_boundaries(labels)
        edge_labels = labels[edges]
        layer[edges] = lut[np.clip(edge_labels, 0, max_label)]

    # merge suspects: THICK orange outline + orange centroid marker (§17/§44)
    if merge_ids:
        mask = np.isin(labels, sorted(merge_ids)).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(layer, contours, -1, COLOR_MERGE, 3)

    id_layer = None
    if debug_ids:
        id_layer = np.zeros((h, w, 3), dtype=np.uint8)

    for row in rows:
        if not row.get("rbc_candidate"):
            continue
        x, y = int(round(row["centroid_x"])), int(round(row["centroid_y"]))
        if row.get("possible_merged_rbc"):
            cv2.circle(layer, (x, y), 3, COLOR_MERGE, -1)  # orange centre marker
        else:
            cv2.circle(layer, (x, y), 2, COLOR_CENTROID, -1)
        if id_layer is not None and row["area_px"] >= 400:
            colour = COLOR_MERGE if row.get("possible_merged_rbc") else COLOR_CENTROID
            cv2.putText(id_layer, str(row["instance_id"]), (x + 3, y - 3),
                        _FONT, 0.32, colour, 1, cv2.LINE_AA)
    return layer, id_layer


def draw_valid_roi(frame_bgr: np.ndarray, margin_fraction: float,
                   colour: Tuple[int, int, int], thickness: int = 2,
                   fill_alpha: float = 0.0) -> np.ndarray:
    """Central valid-analysis region rectangle (+ optional translucent fill)."""
    h, w = frame_bgr.shape[:2]
    mx, my = int(round(w * margin_fraction)), int(round(h * margin_fraction))
    if fill_alpha > 0:
        roi = frame_bgr[my:h - my, mx:w - mx]
        tint = np.full_like(roi, colour, dtype=np.uint8)
        frame_bgr[my:h - my, mx:w - mx] = cv2.addWeighted(roi, 1 - fill_alpha, tint, fill_alpha, 0)
    cv2.rectangle(frame_bgr, (mx, my), (w - mx, h - my), colour, thickness)
    return frame_bgr


def draw_class_border(frame_bgr: np.ndarray, colour: Tuple[int, int, int],
                      thickness: int = 8) -> np.ndarray:
    """Full-viewport border in the current class colour (monolayer highlight)."""
    h, w = frame_bgr.shape[:2]
    cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1), colour, thickness)
    return frame_bgr


def draw_live_hud(frame_bgr: np.ndarray, lines: List[Tuple[str, Tuple[int, int, int]]],
                  panel_width: int = 560) -> np.ndarray:
    """Compact top-left HUD lines over a translucent panel."""
    line_h = 24
    panel_h = 16 + line_h * len(lines)
    overlay_img = frame_bgr.copy()
    cv2.rectangle(overlay_img, (12, 12), (12 + panel_width, 12 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay_img, 0.5, frame_bgr, 0.5, 0, frame_bgr)
    y = 12 + line_h - 6
    for text, colour in lines:
        cv2.putText(frame_bgr, text, (24, y), _FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (24, y), _FONT, 0.5, colour, 1, cv2.LINE_AA)
        y += line_h
    return frame_bgr


def draw_stale_watermark(frame_bgr: np.ndarray) -> np.ndarray:
    """Corner tag marking the displayed detailed result as STALE (spec §16)."""
    h, w = frame_bgr.shape[:2]
    cv2.putText(frame_bgr, "RESULT STALE - microscope moved", (w - 560, h - 24),
                _FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, "RESULT STALE - microscope moved", (w - 560, h - 24),
                _FONT, 0.55, (90, 90, 240), 1, cv2.LINE_AA)
    return frame_bgr
