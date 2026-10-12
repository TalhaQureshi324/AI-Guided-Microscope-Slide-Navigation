"""Coverage and overlap controller (Phase 8).

Prevents human scanning errors by measuring, for any candidate capture, how
its footprint relates to everything already scanned:

    overlap with best single field  ->  primary operator guidance
    overlap with the full scanned union  ->  ALREADY SCANNED / duplicate
    new coverage the candidate would add  ->  the value of capturing it

All math uses Phase-7 registered poses in scan coordinates as axis-aligned
footprint rectangles (x0, y0, x1, y1) - direction-independent (works
horizontally, vertically and diagonally) and capture-order-independent.

States (spec Phase 8 §3):
    ALREADY_SCANNED   candidate >= duplicate_overlap_threshold of the union
    TOO_MUCH_OVERLAP  overlap beyond target_overlap_max
    CAPTURE_READY     overlap inside the target band
    GAP_RISK          overlap below target_overlap_min (or isolated)
    REGISTRATION UNCERTAIN is layered by the caller when the pose estimate
    itself is unreliable.

Move hints point in the direction of recent travel when the operator should
continue, and back when they have overshot into a gap. Thresholds are
configuration, never hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

ALREADY_SCANNED = "ALREADY_SCANNED"
TOO_MUCH_OVERLAP = "TOO_MUCH_OVERLAP"
CAPTURE_READY = "CAPTURE_READY"
GAP_RISK = "GAP_RISK"
FIRST_FIELD = "FIRST_FIELD"
REG_UNCERTAIN = "REGISTRATION UNCERTAIN"


@dataclass
class ExistingFootprint:
    rect: Tuple[float, float, float, float]   # x0, y0, x1, y1 in scan px
    job_id: int
    reg_confidence: float = 1.0
    human_label: Optional[str] = None


@dataclass
class CoverageReport:
    state: str
    overlap_prev_pct: float        # best single-field overlap / candidate area
    overlap_total_pct: float       # candidate area covered by the scanned union
    new_coverage_pct: float        # 1 - overlap_total
    duplicate_pct: float           # = overlap_total_pct (duplicate content)
    horizontal_overlap: float      # best x-extent overlap / candidate width
    vertical_overlap: float        # best y-extent overlap / candidate height
    gap_to_nearest_px: float       # empty edge distance to the nearest footprint
    best_job_id: Optional[int]
    move_hint: str = ""


def _rect(cand: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return cand


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def evaluate_coverage(
    candidate: Tuple[float, float, float, float],
    existing: List[ExistingFootprint],
    target_overlap_min: float = 0.20,
    target_overlap_max: float = 0.30,
    duplicate_overlap_threshold: float = 0.90,
    gap_risk_threshold: float = 0.35,
) -> CoverageReport:
    """Overlap of a candidate footprint vs the scanned union (spec §2, §8).

    Pure geometry: order-independent, works in any scan direction. The
    candidate is (x0, y0, x1, y1) in scan px.
    """
    cx0, cy0, cx1, cy1 = candidate
    c_area = max(1.0, (cx1 - cx0) * (cy1 - cy0))
    c_w, c_h = cx1 - cx0, cy1 - cy0

    if not existing:
        return CoverageReport(
            FIRST_FIELD, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, None,
            "first field - anchors the map")

    best_area = 0.0
    best_id: Optional[int] = None
    best_h = best_v = 0.0
    union_covered = np_masked = None  # computed via interval union below
    total_inter = 0.0
    # inclusion-exclusion is exponential; approximate the union with a
    # scanline over y-bands (footprints are few dozen at most)
    y_edges = sorted({cy0, cy1} | {f.rect[1] for f in existing}
                     | {f.rect[3] for f in existing})
    cover = np.zeros(max(1, len(y_edges) - 1), dtype=np.float64)
    gap_dist = float("inf")

    for f in existing:
        fx0, fy0, fx1, fy1 = f.rect
        ix = _overlap_1d(cx0, cx1, fx0, fx1)
        iy = _overlap_1d(cy0, cy1, fy0, fy1)
        inter = ix * iy
        if inter > best_area:
            best_area = inter
            best_id = f.job_id
            best_h = ix / max(1.0, c_w)
            best_v = iy / max(1.0, c_h)
        # nearest-edge distance for gap detection
        dx = max(fx0 - cx1, cx0 - fx1, 0.0)
        dy = max(fy0 - cy1, cy0 - fy1, 0.0)
        gap_dist = min(gap_dist, math_hypot(dx, dy))

    # union coverage via y-band scanline (exact for axis-aligned rects)
    band_h = [(y_edges[i + 1] - y_edges[i]) for i in range(len(y_edges) - 1)]
    for i in range(len(y_edges) - 1):
        y0, y1 = y_edges[i], y_edges[i + 1]
        ymid = (y0 + y1) / 2.0
        if not (cy0 <= ymid <= cy1):
            continue
        x_ivs = sorted((fx0, fx1) for f in existing
                       if f.rect[1] <= ymid <= f.rect[3])
        cov = _overlap_1d(cx0, cx1, cx0, cx0)  # 0
        merged: List[List[float]] = []
        for x0, x1 in x_ivs:
            iv = _intersect_1d(x0, x1, cx0, cx1)
            if iv is None:
                continue
            if merged and iv[0] <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], iv[1])
            else:
                merged.append([iv[0], iv[1]])
        band_cov = sum(m[1] - m[0] for m in merged)
        cover[i] = band_cov
        total_inter += band_cov * (y1 - y0)

    overlap_total = min(1.0, total_inter / c_area)
    overlap_prev = min(1.0, best_area / c_area)
    new_cov = 1.0 - overlap_total
    gap_frac = max(0.0, target_overlap_min - overlap_prev)

    if overlap_total >= duplicate_overlap_threshold:
        state = ALREADY_SCANNED
        hint = "ALREADY SCANNED - new content %.0f%%" % (new_cov * 100)
    elif overlap_prev > target_overlap_max or overlap_total > target_overlap_max:
        state = TOO_MUCH_OVERLAP
        hint = "TOO MUCH OVERLAP - MOVE FARTHER"
    elif overlap_prev < target_overlap_min:
        state = GAP_RISK
        shortfall = (target_overlap_min - overlap_prev) * c_w
        hint = "GAP RISK - MOVE BACK ~%.0f px for %.0f%% overlap" % (
            shortfall, target_overlap_min * 100)
    else:
        state = CAPTURE_READY
        hint = "CAPTURE READY - new content %.0f%%" % (new_cov * 100)

    return CoverageReport(
        state=state,
        overlap_prev_pct=round(overlap_prev, 4),
        overlap_total_pct=round(overlap_total, 4),
        new_coverage_pct=round(new_cov, 4),
        duplicate_pct=round(overlap_total, 4),
        horizontal_overlap=round(best_h, 4),
        vertical_overlap=round(best_v, 4),
        gap_to_nearest_px=round(min(gap_dist, 1e9), 1),
        best_job_id=best_id,
        move_hint=hint,
    )


def _intersect_1d(a0, a1, b0, b1):
    lo, hi = max(a0, b0), min(a1, b1)
    return [lo, hi] if hi > lo else None


def math_hypot(dx, dy):
    return (dx * dx + dy * dy) ** 0.5


def movement_hint(overlap_prev_pct: float, target_min: float,
                  target_max: float, dir_x: float = 0.0,
                  dir_y: float = 0.0) -> str:
    """Live arrow guidance while the slide moves (spec §7)."""
    if overlap_prev_pct > target_max:
        arrow = _arrow(dir_x, dir_y)
        return f"MOVE FARTHER {arrow}"
    if overlap_prev_pct < target_min:
        arrow = _arrow(-dir_x, -dir_y)
        return f"MOVE BACK {arrow}"
    return "HOLD - CAPTURE READY"


def _arrow(dx: float, dy: float) -> str:
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return ""
    import math as _m
    ang = _m.degrees(_m.atan2(-dy, dx))  # screen y grows downward
    table = [(0, "→"), (45, "↗"), (90, "↑"), (135, "↖"), (180, "←"),
             (-135, "↙"), (-90, "↓"), (-45, "↘")]
    best = min(table, key=lambda t: min(abs(ang - t[0]), 360 - abs(ang - t[0])))
    return best[1]
