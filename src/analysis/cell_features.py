"""Per-cell morphology measurements and conservative RBC candidate filtering.

Measurements are derived from the instance masks (not bounding boxes). Border
cells are flagged, never deleted: partial cells at the image edge have
incomplete geometry, so robust size statistics (median area / diameter) are
estimated from interior cells only.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
from skimage.measure import regionprops

logger = logging.getLogger(__name__)


def measure_instances(labels: np.ndarray) -> List[Dict]:
    """Extract morphology for every unique non-zero label.

    Uses ``skimage.measure.regionprops``, which handles non-consecutive labels,
    so Cellpose's original instance numbering is preserved. The count equals the
    number of returned properties, never ``labels.max()``.
    """
    height, width = labels.shape
    rows: List[Dict] = []

    for prop in regionprops(labels):
        area = float(prop.area)
        perimeter = float(prop.perimeter)
        circularity = (
            4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else float("nan")
        )
        min_row, min_col, max_row, max_col = prop.bbox
        cy, cx = prop.centroid
        equiv_diameter = float(prop.equivalent_diameter_area)

        rows.append(
            {
                "instance_id": int(prop.label),
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "area_px": int(round(area)),
                "equivalent_diameter_px": equiv_diameter,
                "equivalent_radius_px": equiv_diameter / 2.0,
                "perimeter_px": perimeter,
                "circularity": circularity,
                "solidity": float(prop.solidity),
                "eccentricity": float(prop.eccentricity),
                "axis_major_px": float(prop.axis_major_length),
                "axis_minor_px": float(prop.axis_minor_length),
                "elongation": (
                    float(prop.axis_major_length) / prop.axis_minor_length
                    if prop.axis_minor_length > 0
                    else float("inf")
                ),
                "bbox_min_x": int(min_col),
                "bbox_min_y": int(min_row),
                "bbox_max_x": int(max_col),
                "bbox_max_y": int(max_row),
                "touches_border": bool(
                    min_row == 0 or min_col == 0 or max_row == height or max_col == width
                ),
                # filled later by flagging / spatial stages
                "rbc_candidate": True,
                "possible_small_artifact": False,
                "possible_wbc": False,
                "possible_merged_rbc": False,
                "n_dt_peaks": 0,
                "area_ratio_vs_median": float("nan"),
            }
        )

    rows.sort(key=lambda r: r["instance_id"])
    return rows


def compute_reference_stats(rows: List[Dict], stats_scope: str = "interior") -> Dict:
    """Estimate the dominant RBC size from the robust area distribution.

    A first median is computed from the chosen scope (interior cells by
    default), extreme outliers relative to that median are dropped once, and
    the median is recomputed. This keeps merged giants / dust specks from
    dragging the reference while remaining fully data-driven - no fixed pixel
    thresholds that would break with a different magnification.
    """
    scope = [r for r in rows if not r["touches_border"]] if stats_scope == "interior" else rows
    if not scope:  # degenerate image where every cell touches the border
        scope = rows
    if not scope:
        return {
            "median_area_px": float("nan"),
            "median_diameter_px": float("nan"),
            "iqr_area_px": float("nan"),
            "n_stats_cells": 0,
            "stats_scope": stats_scope,
        }

    areas = np.array([r["area_px"] for r in scope], dtype=float)
    first_median = float(np.median(areas))
    core = areas[(areas >= 0.25 * first_median) & (areas <= 4.0 * first_median)]
    if core.size < max(3, areas.size // 10):
        core = areas  # too few survivors to trust the refinement pass

    return {
        "median_area_px": float(np.median(core)),
        "median_diameter_px": float(np.sqrt(4.0 * np.median(core) / math.pi)),
        "iqr_area_px": float(np.percentile(core, 75) - np.percentile(core, 25)),
        "n_stats_cells": int(core.size),
        "stats_scope": stats_scope,
    }


def flag_cells(
    rows: List[Dict],
    ref: Dict,
    small_area_factor: float = 0.35,
    wbc_area_factor: float = 3.5,
) -> List[Dict]:
    """Apply conservative RBC candidate filtering.

    Objects far below the dominant size are flagged ``possible_small_artifact``
    (platelets, dust, segmentation fragments); objects far above are flagged
    ``possible_wbc`` (WBCs or gross merges). Both are flags, NOT deletions:
    biologically unusual RBCs must survive inspection. ``rbc_candidate``
    excludes only clearly non-RBC objects; suspected merges stay candidates
    because they still contain RBC material.
    """
    median_area = ref.get("median_area_px", float("nan"))
    if not np.isfinite(median_area) or median_area <= 0:
        logger.warning("No valid reference median; skipping candidate flagging.")
        return rows

    for row in rows:
        ratio = row["area_px"] / median_area
        row["area_ratio_vs_median"] = ratio
        row["possible_small_artifact"] = bool(ratio < small_area_factor)
        row["possible_wbc"] = bool(ratio > wbc_area_factor)
        row["rbc_candidate"] = not (row["possible_small_artifact"] or row["possible_wbc"])
    return rows
