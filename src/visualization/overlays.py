"""Diagnostic overlay rendering.

Every dataset image gets an overlay containing: the original image, coloured
instance boundaries, centroid markers, instance IDs where readable, highlighted
suspicious objects (merge suspects / small artifacts / possible WBCs / border
cells) and a summary banner with counts and runtime. A second set of crops is
produced for merged-cell debugging.

All drawing uses OpenCV (BGR internally). Overlays are diagnostic outputs only
- the quantitative analysis always works on instance masks and centroids.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR colours for flag categories.
COLOR_MERGE = (0, 69, 255)       # orange - possible merged RBCs
COLOR_SMALL = (0, 0, 255)        # red    - possible small artifacts
COLOR_WBC = (0, 255, 255)        # yellow - possible WBCs
COLOR_BORDER = (255, 255, 0)     # cyan   - border-touching cells
COLOR_CENTROID = (255, 255, 255) # white  - candidate centroids
COLOR_PEAK = (0, 255, 0)         # green  - distance-transform peaks (debug)


def _label_colours(max_label: int) -> np.ndarray:
    """Deterministic per-instance colours via the golden-angle hue sequence."""
    n = max(max_label + 1, 2)
    hues = (np.arange(n, dtype=np.float32) * 137.508) % 180.0
    hsv = np.zeros((n, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = hues.astype(np.uint8)
    hsv[:, 0, 1] = 200
    hsv[:, 0, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(n, 3)


def _label_boundaries(labels: np.ndarray) -> np.ndarray:
    """Boolean edge map wherever adjacent pixels carry different labels."""
    edges = np.zeros(labels.shape, dtype=bool)
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edges[:, -1] |= labels[:, -1] != 0
    edges[-1, :] |= labels[-1, :] != 0
    return edges


def render_overlay(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    rows: List[Dict],
    title: str,
    inference_ms: float,
    show_instance_ids: bool = True,
    min_area_for_id: int = 400,
    banner: bool = True,
) -> np.ndarray:
    """Build the annotated overlay for one image (returns BGR array)."""
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    max_label = int(labels.max()) if labels.size else 0
    if max_label > 0:
        lut = _label_colours(max_label)
        edges = _label_boundaries(labels)
        edge_labels = labels[edges]
        canvas[edges] = lut[np.clip(edge_labels, 0, max_label)]

    id_font_scale = 0.32
    id_thickness = 1
    by_id = {r["instance_id"]: r for r in rows}

    # Flag-driven emphasis (drawn after the thin coloured boundaries).
    for row in rows:
        if row["possible_merged_rbc"]:
            mask = (labels == row["instance_id"]).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(canvas, contours, -1, COLOR_MERGE, 3)
        elif row["possible_wbc"]:
            mask = (labels == row["instance_id"]).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(canvas, contours, -1, COLOR_WBC, 2)
        elif row["possible_small_artifact"]:
            x, y = int(round(row["centroid_x"])), int(round(row["centroid_y"]))
            cv2.drawMarker(canvas, (x, y), COLOR_SMALL, cv2.MARKER_CROSS, 6, 2)

    # Centroids + instance IDs.
    for row in rows:
        x, y = int(round(row["centroid_x"])), int(round(row["centroid_y"]))
        if row["rbc_candidate"]:
            if row["touches_border"]:
                cv2.circle(canvas, (x, y), 4, COLOR_BORDER, 1)
            else:
                cv2.circle(canvas, (x, y), 2, COLOR_CENTROID, -1)
            if show_instance_ids and row["area_px"] >= min_area_for_id:
                cv2.putText(
                    canvas, str(row["instance_id"]), (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, (0, 0, 0), id_thickness + 1,
                )
                cv2.putText(
                    canvas, str(row["instance_id"]), (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, COLOR_CENTROID, id_thickness,
                )

    if banner:
        n_total = len(rows)
        n_cand = sum(1 for r in rows if r["rbc_candidate"])
        n_border = sum(1 for r in rows if r["rbc_candidate"] and r["touches_border"])
        n_merge = sum(1 for r in rows if r["possible_merged_rbc"])
        n_small = sum(1 for r in rows if r["possible_small_artifact"])
        n_wbc = sum(1 for r in rows if r["possible_wbc"])
        line1 = f"{title}"
        line2 = (
            f"instances={n_total} candidates={n_cand} border={n_border} "
            f"merge?={n_merge} small?={n_small} wbc?={n_wbc} "
            f"infer={inference_ms / 1000.0:.1f}s"
        )
        (tw1, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        (tw2, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        width = max(canvas.shape[1], max(tw1, tw2) + 20)
        strip = np.zeros((58, width, 3), dtype=np.uint8)
        cv2.putText(strip, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(strip, line2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        canvas = np.vstack([strip, canvas])

    return canvas


def render_merge_debug_crops(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    rows: List[Dict],
    peaks_by_id: Dict[int, np.ndarray],
    context_fraction: float = 0.35,
    max_crops: int = 12,
) -> List[Dict]:
    """Generate zoomed debug crops for suspicious merged masks.

    Each crop shows the original pixels, the suspect's contour in orange,
    neighbouring contours in grey, and distance-transform peaks as green
    crosses, plus a caption with the evidence (area ratio, solidity,
    elongation, peak count).
    """
    height, width = labels.shape[:2]
    by_id = {r["instance_id"]: r for r in rows}
    crops: List[Dict] = []

    suspects = [r for r in rows if r["possible_merged_rbc"]]
    suspects.sort(key=lambda r: -r["area_px"])  # biggest (most obvious) first
    for row in suspects[:max_crops]:
        lid = row["instance_id"]
        bw = row["bbox_max_x"] - row["bbox_min_x"]
        bh = row["bbox_max_y"] - row["bbox_min_y"]
        mx = int(max(10, context_fraction * bw))
        my = int(max(10, context_fraction * bh))
        x0 = max(0, row["bbox_min_x"] - mx)
        y0 = max(0, row["bbox_min_y"] - my)
        x1 = min(width, row["bbox_max_x"] + mx)
        y1 = min(height, row["bbox_max_y"] + my)

        crop = cv2.cvtColor(image_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
        local = labels[y0:y1, x0:x1]

        for other_id in np.unique(local):
            if other_id == 0 or other_id == lid:
                continue
            m = (local == other_id).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(crop, contours, -1, (120, 120, 120), 1)

        m = (local == lid).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, contours, -1, COLOR_MERGE, 2)

        for (pr, pc) in peaks_by_id.get(lid, []):
            px, py = int(pc) - x0, int(pr) - y0
            if 0 <= px < crop.shape[1] and 0 <= py < crop.shape[0]:
                cv2.drawMarker(crop, (px, py), COLOR_PEAK, cv2.MARKER_CROSS, 7, 2)

        caption = (
            f"id={lid} ratio={row['area_ratio_vs_median']:.2f} "
            f"solid={row['solidity']:.2f} elong={row['elongation']:.2f} "
            f"peaks={row['n_dt_peaks']}"
        )
        strip = np.zeros((26, crop.shape[1], 3), dtype=np.uint8)
        cv2.putText(strip, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        crops.append({"instance_id": lid, "image": np.vstack([strip, crop])})

    return crops
