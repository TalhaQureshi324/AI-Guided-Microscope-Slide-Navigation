#!/usr/bin/env python
"""Cross-magnification localization experiment (SIFT-based) — STANDALONE module.

FYP: AI-Guided Microscope Slide Navigation.

This module is a completely separate offline experiment. It does NOT import,
modify, or depend on any existing pipeline code (live GUI, Cellpose pipeline,
survey mapping, stitching). The only optional reuse is the Cellpose library
itself (used read-only as an independent sanity check, --no-cellpose to skip).

Experiments
-----------
  A  10x reference  ->  40x field              (locate 40x inside 10x)
  B  10x reference  ->  100x field, DIRECT     (allowed to fail — measured honestly)
  C  40x image      ->  100x field             (locate 100x inside 40x)
  +  Hierarchical composition  H_100_to_10 = H_40_to_10 @ H_100_to_40
  +  Direct vs hierarchical comparison (centroid/corner/IoU/area/scale)

Method per registration
-----------------------
  SIFT keypoints -> FLANN knn matching -> Lowe ratio test -> RANSAC homography
  -> geometric sanity checks (footprint size, convexity, aspect, perspective,
  mirror, spread of inliers) -> status: GOOD / WEAK / FAILED.
  Native-scale SIFT first, then scale-assisted SIFT over a candidate scale
  pyramid (child image temporarily downscaled; homography corrected back to
  full child coordinates). Controlled preprocessing strategies are tried in
  order and the strategy used is recorded for every result.

Outputs -> outputs/cross_mag_registration/
  10x_to_40x/  10x_to_100x_direct/  40x_to_100x/
  hierarchical_10x_to_100x/  comparison/

Run:
  python cross_magnification_register.py
      [--reference-10x "10x image.jpg"] [--view-40x "40x image.jpg"]
      [--view-100x "100x image.jpg"] [--out-dir outputs/cross_mag_registration]
      [--no-cellpose] [--self-test]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Experiment configuration (constants — deliberately explicit and bounded)
# --------------------------------------------------------------------------
PREP_STRATEGIES = ["gray", "clahe", "illum_norm", "gray_blur"]

SIFT_NFEATURES = 8000          # cap per image; keeps FLANN matching bounded
LOWE_RATIO = 0.75
RANSAC_PX = 4.0                # reprojection threshold, parent-image pixels
MIN_GOOD_MATCHES = 8           # below this an attempt is not even estimated
MIN_INLIERS = 6                # absolute floor to call anything usable
GOOD_INLIERS = 20
GOOD_INLIER_RATIO = 0.25
GOOD_REPROJ_PX = 4.0           # mean inlier reprojection error (parent px)
GOOD_SPREAD_FRAC = 0.08        # inlier convex-hull area fraction (both images)

# Geometric sanity bounds (section 25): a higher-mag field MUST occupy a
# smaller area of the parent; its projection must stay a convex quad with
# near-constant local scale and no mirror flip.
MIN_FOOTPRINT_FRAC = 0.0005    # degenerate if smaller
MAX_FOOTPRINT_FRAC = 0.85      # impossible if larger (higher mag => smaller FOV)
MAX_ASPECT_DISTORTION = 1.8    # footprint aspect vs child aspect
MAX_PERSPECTIVE_VARIATION = 2.5  # max/min local scale across child corners

# Nominal objective scale ratios (sanity reference ONLY — never hard-coded
# as an expectation; effective scale is measured from the transforms):
NOMINAL_A = 4.0                # 40 / 10
NOMINAL_B = 10.0               # 100 / 10
NOMINAL_C = 2.5                # 100 / 40

# Candidate downscale factors for scale-assisted SIFT (child px -> parent px).
# SIFT is scale invariant only across a limited octave range; a 10x gap exceeds
# it, so the child image is temporarily shrunk and the homography corrected.
SCALE_CANDIDATES_A = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
SCALE_CANDIDATES_B = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0]
SCALE_CANDIDATES_C = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Colors (BGR)
COL_GOOD = (60, 200, 60)       # green
COL_WEAK = (60, 160, 255)      # amber
COL_FAIL = (60, 60, 230)       # red
COL_A = (60, 220, 60)          # 40x footprint: green
COL_DIRECT = (60, 60, 255)     # direct 100x footprint: red
COL_HIER = (255, 60, 255)      # hierarchical 100x footprint: magenta
COL_TEXT = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

S = np.array  # alias for readability below

# OpenCV flag names differ across versions — resolve defensively
RICH_KP = getattr(cv2, "DrawMatchesFlags_DRAW_RICH_KEYPOINTS",
                  getattr(cv2, "DRAW_RICH_KEYPOINTS", 2))
NOT_SINGLE = getattr(cv2, "DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS",
                     getattr(cv2, "NOT_DRAW_SINGLE_POINTS", 1))


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def jnum(x):
    """JSON-safe conversion of numpy scalars/arrays."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [jnum(v) for v in x]
    if isinstance(x, dict):
        return {k: jnum(v) for k, v in x.items()}
    return x


def to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def xform(H, pts):
    """Apply 3x3 homography to an (N,2) point array (column-vector convention:
    p_parent ~ H @ p_child)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64))
    return out.reshape(-1, 2)


def child_corners(w: int, h: int) -> np.ndarray:
    """Image corners in TL, TR, BR, BL order."""
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)


def local_scales(H, p, eps: float = 2.0):
    """Local linear map of H at child point p: (sx, sy, signed area factor).
    signed area factor > 0 means orientation preserved; < 0 means mirrored."""
    p = np.asarray(p, np.float64)
    o = xform(H, [p])[0]
    bx = (xform(H, [p + np.array([eps, 0.0])])[0] - o) / eps
    by = (xform(H, [p + np.array([0.0, eps])])[0] - o) / eps
    sx = float(np.linalg.norm(bx))
    sy = float(np.linalg.norm(by))
    det = float(bx[0] * by[1] - bx[1] * by[0])
    return sx, sy, det


def quad_convex(quad: np.ndarray) -> bool:
    """A perspective image of a rectangle is always a convex quad."""
    signs = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cr)
    return all(v > 0 for v in signs) or all(v < 0 for v in signs)


def spread_frac(pts: np.ndarray, shape) -> float:
    """Convex-hull area fraction of points inside an image (spatial spread,
    section 24). Returns 0 for < 3 points."""
    if pts is None or len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts.astype(np.float32))
    area = float(cv2.contourArea(hull))
    return area / float(shape[0] * shape[1])


def hull_area_frac(pts: np.ndarray, denom_area: float) -> float:
    """Convex-hull area of points as a fraction of an arbitrary reference area
    (used for the parent side, where the natural denominator is the footprint,
    not the whole parent image)."""
    if pts is None or len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts.astype(np.float32))
    return float(cv2.contourArea(hull)) / max(float(denom_area), 1.0)


def effective_scale(H, child_shape) -> float:
    """Measured linear image scale (parent px per child px) at child centre."""
    sx, sy, _ = local_scales(H, (child_shape[1] / 2.0, child_shape[0] / 2.0))
    return math.sqrt(max(sx * sy, 1e-12))


def scale_similarity_matrix(s: float) -> np.ndarray:
    """Maps original child px -> downscaled child px for resize factor s
    (child_small = child / s)."""
    return np.array([[1.0 / s, 0, 0], [0, 1.0 / s, 0], [0, 0, 1.0]])


def quad_bbox(quad: np.ndarray, shape):
    x0 = int(max(0, np.floor(quad[:, 0].min())))
    y0 = int(max(0, np.floor(quad[:, 1].min())))
    x1 = int(min(shape[1] - 1, np.ceil(quad[:, 0].max())))
    y1 = int(min(shape[0] - 1, np.ceil(quad[:, 1].max())))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def quad_iou(q1: np.ndarray, q2: np.ndarray, shape) -> float:
    m1 = np.zeros(shape[:2], np.uint8)
    m2 = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m1, [np.round(q1).astype(np.int32)], 1)
    cv2.fillPoly(m2, [np.round(q2).astype(np.int32)], 1)
    union = np.logical_or(m1, m2).sum()
    inter = np.logical_and(m1, m2).sum()
    return float(inter / union) if union else 0.0


# --------------------------------------------------------------------------
# Controlled preprocessing (section 7)
# --------------------------------------------------------------------------
def preprocess(img_bgr: np.ndarray, strategy: str) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if strategy == "gray":
        return gray
    if strategy == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    if strategy == "illum_norm":
        f = gray.astype(np.float32)
        bg = cv2.GaussianBlur(f, (0, 0), sigmaX=40.0)
        norm = f / (bg + 1.0) * 128.0
        return np.clip(norm, 0.0, 255.0).astype(np.uint8)
    if strategy == "gray_blur":
        return cv2.GaussianBlur(gray, (3, 3), 0)
    raise ValueError(f"unknown preprocessing strategy: {strategy}")


# --------------------------------------------------------------------------
# SIFT detection + matching
# --------------------------------------------------------------------------
_SIFT = None


def sift_detect(gray: np.ndarray):
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    kp, des = _SIFT.detectAndCompute(gray, None)
    return kp, des


def knn_matches(des_parent, des_child, ratio: float):
    """knnMatch with query = CHILD descriptors so that m.queryIdx -> child kp
    and m.trainIdx -> parent kp; H therefore maps child -> parent."""
    if des_parent is None or des_child is None:
        return 0, []
    if len(des_parent) < 2 or len(des_child) < 2:
        return 0, []
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    raw = flann.knnMatch(des_child, des_parent, k=2)
    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return len(raw), good


def estimate_homography(kp_parent, kp_child, good):
    """Returns (H_child_to_parent, inlier_mask, error_reason)."""
    if len(good) < 4:
        return None, None, "too few good matches for homography (<4)"
    src = np.float32([kp_child[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_parent[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, RANSAC_PX,
                                     maxIters=20000, confidence=0.9999)
        method = "USAC_MAGSAC"
    except Exception:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX,
                                     maxIters=5000, confidence=0.999)
        method = "RANSAC"
    if H is None:
        return None, None, "homography estimation failed (degenerate matches)"
    return H, mask.ravel().astype(bool), method


def reproj_errors(H, kp_parent, kp_child, good):
    mapped = xform(H, [kp_child[m.queryIdx].pt for m in good])
    target = np.array([kp_parent[m.trainIdx].pt for m in good], np.float64)
    return np.linalg.norm(mapped - target, axis=1)


# --------------------------------------------------------------------------
# Geometric sanity checks (sections 23-25)
# --------------------------------------------------------------------------
def sanity_check(H, child_shape, parent_shape):
    """Returns (ok, quad_in_parent, reasons[])."""
    h, w = child_shape[:2]
    Ht, Wt = parent_shape[:2]
    quad = xform(H, child_corners(w, h))
    reasons = []
    if not np.all(np.isfinite(quad)):
        reasons.append("non-finite corner mapping")
        return False, quad, reasons
    _, _, det = local_scales(H, (w / 2.0, h / 2.0))
    if det <= 0:
        reasons.append("mirrored transform (negative orientation determinant)")
    if not quad_convex(quad):
        reasons.append("non-convex / self-intersecting footprint quad")
    area = float(cv2.contourArea(quad.astype(np.float32)))
    frac = area / float(Ht * Wt)
    if frac < MIN_FOOTPRINT_FRAC:
        reasons.append(f"degenerate footprint area ({frac:.2e} of reference)")
    if frac > MAX_FOOTPRINT_FRAC:
        reasons.append(
            f"footprint covers {frac * 100:.0f}% of reference "
            f"(impossible: higher magnification has a SMALLER field)")
    e_top = float(np.linalg.norm(quad[1] - quad[0]))
    e_bot = float(np.linalg.norm(quad[2] - quad[3]))
    e_lft = float(np.linalg.norm(quad[3] - quad[0]))
    e_rgt = float(np.linalg.norm(quad[2] - quad[1]))
    quad_ratio = ((e_top + e_bot) / 2.0) / max((e_lft + e_rgt) / 2.0, 1e-9)
    child_ratio = w / float(h)
    distort = max(quad_ratio / child_ratio, child_ratio / quad_ratio)
    if distort > MAX_ASPECT_DISTORTION:
        reasons.append(f"extreme aspect distortion ({distort:.2f}x)")
    scales = []
    for p in child_corners(w, h):
        sx, sy, _ = local_scales(H, p)
        scales.append(math.sqrt(max(sx * sy, 1e-12)))
    pv = max(scales) / max(min(scales), 1e-12)
    if pv > MAX_PERSPECTIVE_VARIATION:
        reasons.append(f"wild perspective variation ({pv:.2f}x across corners)")
    ctr = quad.mean(axis=0)
    if not (0 <= ctr[0] < Wt and 0 <= ctr[1] < Ht):
        reasons.append("footprint centre outside reference image")
    return (not reasons), quad, reasons


# --------------------------------------------------------------------------
# One registration attempt (one strategy x one scale)
# --------------------------------------------------------------------------
def evaluate_attempt(kp_parent, des_parent, prep_parent, parent_bgr,
                     kp_child, des_child, prep_child, child_view_bgr,
                     strategy, scale, parent_shape, child_shape):
    """Returns (record_dict, context_dict_or_None). record has no heavy data."""
    rec = {
        "strategy": strategy,
        "scale": "native" if scale is None else float(scale),
        "status": "FAILED",
        "num_keypoints_parent": len(kp_parent),
        "num_keypoints_child": len(kp_child),
        "raw_matches": 0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
    }
    raw_n, good = knn_matches(des_parent, des_child, LOWE_RATIO)
    rec["raw_matches"] = int(raw_n)
    rec["good_matches"] = len(good)
    if len(good) < MIN_GOOD_MATCHES:
        rec["fail_reason"] = f"too few good matches ({len(good)} < {MIN_GOOD_MATCHES})"
        return rec, None

    H, mask, err = estimate_homography(kp_parent, kp_child, good)
    if H is None:
        rec["fail_reason"] = err
        return rec, None
    inliers = int(mask.sum())
    rec["inliers"] = inliers
    rec["inlier_ratio"] = float(inliers) / len(good)
    # H from the keypoints maps child-VIEW pixels -> parent. With scale-assist
    # the view is the downscaled child, so compose back to ORIGINAL child px:
    #   p_parent = H_view @ S @ p_child  (S = child -> view coordinate scale)
    H_view = H
    H_full = H if scale is None else H @ scale_similarity_matrix(scale)
    errs = reproj_errors(H_view, kp_parent, kp_child, good)
    in_err = errs[mask]
    rec["reproj_mean_px"] = float(in_err.mean()) if in_err.size else float("inf")
    rec["reproj_rms_px"] = float(np.sqrt((in_err ** 2).mean())) if in_err.size else float("inf")
    rec["ransac_method"] = err  # method name comes back in err slot when ok

    child_pts = np.array([kp_child[m.queryIdx].pt for m in good], np.float64)
    parent_pts = np.array([kp_parent[m.trainIdx].pt for m in good], np.float64)

    ok, quad, reasons = sanity_check(H_full, child_shape, parent_shape)
    rec["sanity_ok"] = bool(ok)
    rec["sanity_reasons"] = reasons
    # Spatial spread (section 24): child side = fraction of the (view) image
    # covered by the inlier hull; parent side = fraction of the projected
    # footprint covered (NOT of the whole parent — a high-mag footprint is
    # legitimately tiny compared to the parent image).
    view_area = float(child_view_bgr.shape[0] * child_view_bgr.shape[1])
    foot_area = max(float(cv2.contourArea(quad.astype(np.float32))), 1.0)
    sp_c = hull_area_frac(child_pts[mask], view_area)
    sp_p = hull_area_frac(parent_pts[mask], foot_area)
    rec["inlier_spread_child"] = float(sp_c)
    rec["inlier_spread_parent"] = float(sp_p)
    if not ok:
        rec["status"] = "FAILED"
        rec["fail_reason"] = "geometric sanity rejected: " + "; ".join(reasons)
        return rec, None
    if inliers < MIN_INLIERS:
        rec["status"] = "FAILED"
        rec["fail_reason"] = f"too few RANSAC inliers ({inliers} < {MIN_INLIERS})"
        return rec, None

    rec["effective_scale"] = float(effective_scale(H_full, child_shape))
    rec["quad_in_parent"] = quad
    rec["footprint_area_px2"] = float(cv2.contourArea(quad.astype(np.float32)))
    rec["percent_parent_area"] = rec["footprint_area_px2"] / float(
        parent_shape[0] * parent_shape[1]) * 100.0
    bb = quad_bbox(quad, parent_shape)
    rec["bbox_in_parent"] = list(bb) if bb else None

    good_status = (inliers >= GOOD_INLIERS
                   and rec["inlier_ratio"] >= GOOD_INLIER_RATIO
                   and rec["reproj_mean_px"] <= GOOD_REPROJ_PX
                   and min(sp_c, sp_p) >= GOOD_SPREAD_FRAC)
    rec["status"] = "GOOD" if good_status else "WEAK"
    if rec["status"] == "WEAK":
        notes = []
        if inliers < GOOD_INLIERS:
            notes.append(f"inliers {inliers} < {GOOD_INLIERS}")
        if rec["inlier_ratio"] < GOOD_INLIER_RATIO:
            notes.append(f"inlier ratio {rec['inlier_ratio']:.2f} < {GOOD_INLIER_RATIO}")
        if rec["reproj_mean_px"] > GOOD_REPROJ_PX:
            notes.append(f"reproj {rec['reproj_mean_px']:.1f}px > {GOOD_REPROJ_PX}")
        if min(sp_c, sp_p) < GOOD_SPREAD_FRAC:
            notes.append(f"inlier spread {min(sp_c, sp_p):.2f} < {GOOD_SPREAD_FRAC}")
        rec["weak_reasons"] = notes

    ctx = {"H": H_full, "mask": mask, "good": good, "kp_parent": kp_parent,
           "kp_child": kp_child, "prep_parent": prep_parent,
           "prep_child": prep_child,
           "child_view": child_view_bgr, "child_pts": child_pts,
           "parent_pts": parent_pts}
    return rec, ctx


def _better(rec, best_rec) -> bool:
    if best_rec is None:
        return True
    key = (rec["inliers"], rec["inlier_ratio"], -rec["reproj_mean_px"])
    bkey = (best_rec["inliers"], best_rec["inlier_ratio"], -best_rec["reproj_mean_px"])
    return key > bkey


# --------------------------------------------------------------------------
# Full pairwise registration: native SIFT first, then scale-assisted (section 8)
# --------------------------------------------------------------------------
def register_pair(parent_bgr, child_bgr, tag: str, scale_candidates):
    t0 = time.perf_counter()
    parent_shape, child_shape = parent_bgr.shape, child_bgr.shape
    attempts, contexts = [], {}
    best_rec, best_ctx = None, None

    parent_cache = {}   # strategy -> (prep, kp, des)
    for strategy in PREP_STRATEGIES:
        prep = preprocess(parent_bgr, strategy)
        kp, des = sift_detect(prep)
        parent_cache[strategy] = (prep, kp, des)

    schedule = [("native", None)] + [("scaled", s) for s in scale_candidates]
    for kind, scale in schedule:
        for strategy in PREP_STRATEGIES:
            prep_p, kp_p, des_p = parent_cache[strategy]
            if kind == "native":
                child_view = child_bgr
                eff = None
            else:
                child_view = cv2.resize(child_bgr, None, fx=1.0 / scale,
                                        fy=1.0 / scale, interpolation=cv2.INTER_AREA)
                eff = scale
            prep_c = preprocess(child_view, strategy)
            kp_c, des_c = sift_detect(prep_c)
            rec, ctx = evaluate_attempt(
                kp_p, des_p, prep_p, parent_bgr, kp_c, des_c, prep_c,
                child_view, strategy, eff, parent_shape, child_shape)
            attempts.append(rec)
            log(f"  [{tag}] {strategy:>10s} "
                f"scale={'native' if eff is None else ('1/%g' % eff):>7s} "
                f"kps={rec['num_keypoints_parent']}/{rec['num_keypoints_child']} "
                f"good={rec['good_matches']:4d} -> {rec['status']}"
                + (f" (inliers={rec['inliers']}, ratio={rec['inlier_ratio']:.2f}, "
                   f"reproj={rec.get('reproj_mean_px', float('inf')):.1f}px)"
                   if rec['status'] in ('GOOD', 'WEAK') else
                   f" ({rec.get('fail_reason', '')[:90]})"))
            if ctx is not None:
                contexts[(strategy, rec["scale"])] = ctx
            if rec["status"] == "GOOD":
                best_rec, best_ctx = rec, ctx
                break
            if rec["status"] == "WEAK" and _better(rec, best_rec):
                best_rec, best_ctx = rec, ctx
        if best_rec is not None and best_rec["status"] == "GOOD":
            break

    summary = {
        "status": best_rec["status"] if best_rec else "FAILED",
        "best": best_rec if best_rec else None,
        "attempts": attempts,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    if best_ctx is not None:
        summary["ctx"] = best_ctx
    return summary


# --------------------------------------------------------------------------
# Drawing helpers (cv2 only — no matplotlib on this machine)
# --------------------------------------------------------------------------
def draw_styled_line(img, p1, p2, color, thickness, style="solid"):
    p1 = np.array(p1, float)
    p2 = np.array(p2, float)
    d = p2 - p1
    dist = float(np.linalg.norm(d))
    if dist < 1:
        return
    u = d / dist
    if style == "solid":
        cv2.line(img, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)),
                 color, thickness, cv2.LINE_AA)
        return
    if style == "dashed":
        on, off = 12, 8
    else:  # dashdot
        on, off = 14, 5
    t = 0.0
    while t < dist:
        seg = min(on, dist - t)
        a = p1 + u * t
        b = p1 + u * (t + seg)
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                 color, thickness, cv2.LINE_AA)
        t += seg + (off if style == "dashed" else off)
        if style == "dashdot" and t < dist:  # tiny dot between dashes
            seg2 = min(3, dist - t)
            a = p1 + u * t
            b = p1 + u * (t + seg2)
            cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                     color, thickness, cv2.LINE_AA)
            t += seg2 + off


def draw_quad(img, quad, color, thickness=3, style="solid"):
    pts = [tuple(np.round(p).astype(int)) for p in quad]
    for i in range(4):
        draw_styled_line(img, pts[i], pts[(i + 1) % 4], color, thickness, style)
    for i, p in enumerate(pts):  # corner ticks
        cv2.circle(img, p, thickness + 3, color, -1, cv2.LINE_AA)


def put_text_block(img, lines, org=(12, 12), scale=0.62, color=COL_TEXT,
                   bg=(0, 0, 0), thickness=1, line_h=None):
    line_h = line_h or int(26 * scale / 0.62)
    x, y = org
    sizes = [cv2.getTextSize(t, FONT, scale, thickness)[0] for t in lines]
    w = max(s[0] for s in sizes) + 2 * 6
    h = line_h * len(lines) + 2 * 4
    x0, y0 = max(0, x - 6), max(0, y - 4)
    x1, y1 = min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    yy = y
    for t in lines:
        cv2.putText(img, t, (x, yy + line_h - 10), FONT, scale, color, thickness, cv2.LINE_AA)
        yy += line_h
    return img


def draw_legend(img, entries, org, title=None):
    """entries: list of (color, style, thickness, label)."""
    scale, line_h, pad = 0.58, 26, 10
    widths = [cv2.getTextSize(e[3], FONT, scale, 1)[0][0] for e in entries]
    if title:
        widths.append(cv2.getTextSize(title, FONT, scale, 1)[0][0])
    w = max(widths) + 90 + 2 * pad
    h = line_h * (len(entries) + (1 if title else 0)) + 2 * pad
    x0, y0 = org
    x1 = min(img.shape[1] - 4, x0 + w)
    y1 = min(img.shape[0] - 4, y0 + h)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    yy = y0 + pad + 12
    if title:
        cv2.putText(img, title, (x0 + pad + 58, yy), FONT, scale, (30, 30, 30), 1, cv2.LINE_AA)
        yy += line_h
    for color, style, th, label in entries:
        pa = (x0 + pad + 8, yy - 5)
        pb = (x0 + pad + 58, yy - 5)
        draw_styled_line(img, pa, pb, color, th, style)
        cv2.putText(img, label, (x0 + pad + 66, yy), FONT, scale, (30, 30, 30), 1, cv2.LINE_AA)
        yy += line_h


def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    f = target_h / float(h)
    return cv2.resize(img, (int(w * f), target_h), interpolation=cv2.INTER_AREA)


def fit_width(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    f = max_w / float(w)
    return cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)


def label_bar(img, text, color=(30, 30, 30)):
    bar = np.full((34, img.shape[1], 3), color, np.uint8)
    cv2.putText(bar, text, (10, 24), FONT, 0.62, COL_TEXT, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


# --------------------------------------------------------------------------
# Per-experiment visual outputs
# --------------------------------------------------------------------------
def write_experiment_outputs(out_dir: Path, parent_bgr, child_bgr,
                             rec, ctx, title, nominal_scale):
    out_dir.mkdir(parents=True, exist_ok=True)
    status = rec["status"]
    col = {"GOOD": COL_GOOD, "WEAK": COL_WEAK}.get(status, COL_FAIL)

    if ctx is not None:
        # -- keypoints diagnostic ------------------------------------------
        kv_p = cv2.drawKeypoints(to_bgr(ctx["prep_parent"]), ctx["kp_parent"], None,
                                 flags=RICH_KP)
        kv_c = cv2.drawKeypoints(to_bgr(ctx["prep_child"]), ctx["kp_child"], None,
                                 flags=RICH_KP)
        th = min(kv_p.shape[0], kv_c.shape[0], 700)
        kv = np.hstack([resize_to_height(kv_p, th), resize_to_height(kv_c, th)])
        kv = fit_width(kv, 2400)
        kv = label_bar(kv, f"KEYPOINTS  parent: {rec['num_keypoints_parent']}   "
                           f"child: {rec['num_keypoints_child']}   "
                           f"(preprocessing: {rec['strategy']})")
        cv2.imwrite(str(out_dir / "keypoints.png"), kv)

        # -- matches / inliers ---------------------------------------------
        # matches connect queryIdx (CHILD) -> trainIdx (PARENT): child must be
        # the first (left) image, parent the second (right)
        mask_int = ctx["mask"].astype(int).ravel().tolist()
        common = dict(flags=NOT_SINGLE)
        mv = cv2.drawMatches(to_bgr(ctx["prep_child"]), ctx["kp_child"],
                             to_bgr(ctx["prep_parent"]), ctx["kp_parent"],
                             ctx["good"], None,
                             matchesMask=None, **common)
        mv = fit_width(mv, 2400)
        mv = label_bar(mv, f"GOOD MATCHES (Lowe ratio {LOWE_RATIO}): {len(ctx['good'])}"
                           f"   [left = high-mag view "
                           f"({'native' if rec['scale'] == 'native' else 'at 1/%g scale' % rec['scale']})"
                           f", right = reference]")
        cv2.imwrite(str(out_dir / "matches.png"), mv)
        iv = cv2.drawMatches(to_bgr(ctx["prep_child"]), ctx["kp_child"],
                             to_bgr(ctx["prep_parent"]), ctx["kp_parent"],
                             ctx["good"], None,
                             matchesMask=mask_int, **common)
        iv = fit_width(iv, 2400)
        iv = label_bar(iv, f"RANSAC INLIERS: {rec['inliers']} / {len(ctx['good'])} "
                           f"(ratio {rec['inlier_ratio']:.2f}, "
                           f"reproj {rec['reproj_mean_px']:.2f}px rms "
                           f"{rec['reproj_rms_px']:.2f}px)")
        cv2.imwrite(str(out_dir / "inliers.png"), iv)

    # -- reference location -------------------------------------------------
    loc = parent_bgr.copy()
    status_lines = [title,
                    f"status: {status}   preprocessing: {rec['strategy']}   "
                    f"scale-assist: {rec['scale']}",
                    f"good matches: {rec['good_matches']}   inliers: {rec['inliers']}   "
                    f"ratio: {rec['inlier_ratio']:.2f}"]
    if ctx is not None:
        quad = rec["quad_in_parent"]
        draw_quad(loc, quad, col, 4, "solid" if status == "GOOD" else "dashed")
        status_lines += [
            f"reproj mean/rms: {rec['reproj_mean_px']:.2f} / {rec['reproj_rms_px']:.2f} px",
            f"footprint: {rec['footprint_area_px2']:.0f} px2 = "
            f"{rec['percent_parent_area']:.2f}% of reference",
            f"effective scale (measured): {rec['effective_scale']:.2f} "
            f"(nominal {nominal_scale:g})",
            f"inlier spread child/parent: {rec['inlier_spread_child']:.2f} / "
            f"{rec['inlier_spread_parent']:.2f}",
        ]
        if status == "WEAK":
            status_lines.append("WEAK: " + "; ".join(rec.get("weak_reasons", [])))
    else:
        status_lines.append("REGISTRATION FAILED - no reliable transform")
        status_lines.append("reason: " + str(rec.get("fail_reason", "unknown")))
        if rec.get("sanity_reasons"):
            status_lines.append("sanity: " + "; ".join(rec["sanity_reasons"]))
    put_text_block(loc, status_lines)
    cv2.imwrite(str(out_dir / "reference_location.png"), loc)

    if ctx is not None:
        # -- crop + perspective-normalised warp (sections 11, 13, 16, 19) ---
        bb = quad_bbox(rec["quad_in_parent"], parent_bgr.shape)
        if bb:
            crop = parent_bgr[bb[1]:bb[3], bb[0]:bb[2]].copy()
            cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), col, 3)
            crop = label_bar(crop, f"REFERENCE CROP  bbox={bb}")
            cv2.imwrite(str(out_dir / "crop.png"), crop)
        Hc = np.linalg.inv(np.asarray(ctx["H"], np.float64))
        warp = cv2.warpPerspective(parent_bgr, Hc,
                                   (child_bgr.shape[1], child_bgr.shape[0]))
        sbs_h = 560
        wimg = resize_to_height(warp, sbs_h)
        cimg = resize_to_height(child_bgr, sbs_h)
        ww = max(wimg.shape[1], cimg.shape[1])
        wimg = cv2.copyMakeBorder(wimg, 0, 0, 0, ww - wimg.shape[1],
                                  cv2.BORDER_CONSTANT)
        cimg = cv2.copyMakeBorder(cimg, 0, 0, 0, ww - cimg.shape[1],
                                  cv2.BORDER_CONSTANT)
        sbs = np.hstack([wimg, np.full((sbs_h, 6, 3), 255, np.uint8), cimg])
        sbs = fit_width(sbs, 2400)
        sbs = label_bar(sbs, "PERSPECTIVE-NORMALISED REFERENCE REGION (from 10x warp)"
                             "   |   ACTUAL HIGH-MAG IMAGE")
        cv2.imwrite(str(out_dir / "warp.png"), warp)
        cv2.imwrite(str(out_dir / "side_by_side.png"), sbs)

    # -- result.json ---------------------------------------------------------
    result = {
        "experiment": title,
        "status": status,
        "preprocessing_strategy": rec["strategy"],
        "scale_assist": rec["scale"],
        "keypoints": {"parent": rec["num_keypoints_parent"],
                      "child": rec["num_keypoints_child"]},
        "raw_matches": rec["raw_matches"],
        "good_matches": rec["good_matches"],
        "inliers": rec["inliers"],
        "inlier_ratio": rec["inlier_ratio"],
        "reprojection_error_px": {"mean": rec.get("reproj_mean_px"),
                                  "rms": rec.get("reproj_rms_px")},
        "best_scale": rec["scale"],
        "effective_scale_measured": rec.get("effective_scale"),
        "effective_scale_nominal": nominal_scale,
        "transform_child_to_parent": jnum(ctx["H"]) if ctx else None,
        "polygon_in_parent": jnum(rec.get("quad_in_parent")),
        "footprint_area_px2": rec.get("footprint_area_px2"),
        "percent_parent_area": rec.get("percent_parent_area"),
        "bbox_in_parent": rec.get("bbox_in_parent"),
        "inlier_spread": {"child": rec.get("inlier_spread_child"),
                          "parent": rec.get("inlier_spread_parent")},
        "confidence": status,
        "fail_reason": rec.get("fail_reason"),
        "sanity_reasons": rec.get("sanity_reasons"),
        "weak_reasons": rec.get("weak_reasons"),
        "ransac_method": rec.get("ransac_method"),
        "runtime_seconds": None,
        "cellpose_validation": None,  # filled by caller
        "attempts": [],               # filled by caller
    }
    return result


# --------------------------------------------------------------------------
# Cellpose RBC validation (sections 26-27) — optional independent evidence
# --------------------------------------------------------------------------
_CP = {"model": None, "error": None}


def _cellpose_model():
    if _CP["error"]:
        raise RuntimeError(_CP["error"])
    if _CP["model"] is None:
        from cellpose import models
        import torch
        try:
            _CP["model"] = models.CellposeModel(
                gpu=torch.cuda.is_available(), pretrained_model="cpsam")
        except Exception as e:  # pragma: no cover
            _CP["error"] = f"{type(e).__name__}: {e}"
            raise
    return _CP["model"]


def segment_centroids(img_bgr):
    model = _cellpose_model()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masks, _flows, _styles = model.eval(rgb, channel_axis=-1, normalize=True)
    labels = np.asarray(masks)
    n = int(labels.max())
    if n == 0:
        return np.zeros((0, 2), np.float64), 0.0, 0
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=n + 1)[1:]
    ys, xs = np.nonzero(labels)
    sx = np.bincount(labels[ys, xs], weights=xs, minlength=n + 1)[1:]
    sy = np.bincount(labels[ys, xs], weights=ys, minlength=n + 1)[1:]
    valid = counts > 0
    cents = np.stack([sx[valid] / counts[valid], sy[valid] / counts[valid]], axis=1)
    radii = np.sqrt(counts[valid] / np.pi)
    return cents, float(np.median(radii)), int(valid.sum())


def cellpose_validate(parent_bgr, child_bgr, H, out_png: Path, eff_scale=None):
    try:
        cents_p, rad_p, n_p = segment_centroids(parent_bgr)
        cents_c, rad_c, n_c = segment_centroids(child_bgr)
    except Exception as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}
    if n_p < 5 or n_c < 5:
        return {"status": "insufficient_cells", "n_parent_rbc": n_p,
                "n_child_rbc": n_c}
    transformed = xform(H, cents_c)
    from scipy.spatial import cKDTree
    tree = cKDTree(cents_p)
    dist, _idx = tree.query(transformed, k=1)
    thr = max(0.75 * rad_p, 2.0)
    matched = dist <= thr
    res = {
        "status": "ok",
        "n_parent_rbc": n_p,
        "n_child_rbc": n_c,
        "median_parent_rbc_radius_px": rad_p,
        "match_threshold_px": float(thr),
        "n_matched": int(matched.sum()),
        "match_pct": float(matched.sum() / max(n_c, 1) * 100.0),
        "median_centroid_error_px": float(np.median(dist[matched])) if matched.any() else None,
    }
    if eff_scale:
        res["median_child_rbc_radius_in_parent_px"] = float(rad_c * eff_scale)

    vis = parent_bgr.copy()
    for p in cents_p:
        cv2.circle(vis, (int(p[0]), int(p[1])), 2, (80, 220, 80), -1)
    for i, p in enumerate(transformed):
        cv2.circle(vis, (int(p[0]), int(p[1])), 2, (60, 60, 255), -1)
        if matched[i]:
            q = cents_p[_idx[i]]
            cv2.line(vis, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                     (0, 255, 255), 1, cv2.LINE_AA)
    vis = fit_width(vis, 2000)
    put_text_block(vis, [
        "CELLPOSE RBC CENTROID VALIDATION (supporting evidence only)",
        f"parent RBCs: {n_p}   child RBCs: {n_c} (transformed into parent, red)",
        f"matched within {thr:.1f}px: {int(matched.sum())} "
        f"({res['match_pct']:.1f}%)   median error: "
        f"{res['median_centroid_error_px'] and round(res['median_centroid_error_px'], 1)}px",
    ])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), vis)
    return res


# --------------------------------------------------------------------------
# Hierarchical composition (sections 17-19)
# --------------------------------------------------------------------------
def compose_hierarchies(H_40_to_10, H_100_to_40, shape_10x, shape_40x, shape_100x):
    H = np.asarray(H_40_to_10, np.float64) @ np.asarray(H_100_to_40, np.float64)
    H = H / H[2, 2]
    w100, h100 = shape_100x[1], shape_100x[0]
    corners = child_corners(w100, h100)
    in40 = xform(H_100_to_40, corners)
    in10 = xform(H, corners)
    seq = xform(H_40_to_10, in40)
    verify = {
        "center_100x_in_40x": in40.mean(axis=0).tolist(),
        "center_in_40x_bounds": bool(
            0 <= in40[:, 0].mean() < shape_40x[1]
            and 0 <= in40[:, 1].mean() < shape_40x[0]),
        "center_100x_in_10x_hierarchical": in10.mean(axis=0).tolist(),
        "center_in_10x_bounds": bool(
            0 <= in10[:, 0].mean() < shape_10x[1]
            and 0 <= in10[:, 1].mean() < shape_10x[0]),
        "sequential_vs_composed_max_corner_err_px": float(
            np.linalg.norm(in10 - seq, axis=1).max()),
    }
    return H, in10, verify


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def image_meta(path: Path, mag: str):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"ERROR: cannot read image: {path}")
    h, w = img.shape[:2]
    return {
        "filename": path.name,
        "path": str(path),
        "magnification": mag,
        "width": w,
        "height": h,
        "channels": img.shape[2] if img.ndim == 3 else 1,
        "pixel_count": int(img.size // (img.shape[2] if img.ndim == 3 else 1)),
    }, img


def run_self_test() -> int:
    """Internal fast check of conventions used by the experiment."""
    ok = True

    def check(name, cond):
        nonlocal ok
        log(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    # 1. xform convention: p_parent ~ H p_child
    H = np.array([[0.5, 0, 10], [0, 0.5, 20], [0, 0, 1.0]])
    p = xform(H, [(2, 4)])[0]
    check("xform maps child (2,4) to parent (11,22)", np.allclose(p, [11, 22]))

    # 2. composition order: H_100_to_10 = H_40_to_10 @ H_100_to_40
    H_40_to_10 = np.array([[0.25, 0, 5], [0, 0.25, 7], [0, 0, 1.0]])
    H_100_to_40 = np.array([[0.4, 0, 1], [0, 0.4, 2], [0, 0, 1.0]])
    comp = H_40_to_10 @ H_100_to_40
    p100 = np.array([100.0, 60.0])
    step = xform(H_40_to_10, xform(H_100_to_40, [p100]))[0]
    direct = xform(comp, [p100])[0]
    check("composed == sequential mapping", np.allclose(step, direct))
    check("pure-scale composition 1/(4*2.5)=1/10",
          np.allclose(comp[0, 0], 0.25 * 0.4) and abs(comp[0, 2] - (5 + 0.25 * 1)) < 1e-12)

    # 3. scale-assist algebra: H_small @ S == direct homography
    s = 4.0
    H_small = np.array([[0.25, 0, 10], [0, 0.25, 10], [0, 0, 1.0]])
    H_full = H_small @ scale_similarity_matrix(s)
    check("scale-assist H_small@S maps original child px correctly",
          np.allclose(xform(H_full, [(40, 40)])[0], xform(H_small, [(10, 10)])[0]))

    # 4. sanity rejections
    big = np.array([[8.0, 0, -3000], [0, 8.0, -3000], [0, 0, 1.0]])  # blows up
    okq, quad, why = sanity_check(big, (1080, 1920, 3), (1080, 1920, 3))
    check("oversized footprint rejected", (not okq) and any("impossible" in w for w in why))
    mir = np.array([[-0.25, 0, 900], [0, 0.25, 100], [0, 0, 1.0]])
    okq, _, why = sanity_check(mir, (1080, 1920, 3), (1080, 1920, 3))
    check("mirrored transform rejected", (not okq) and any("mirror" in w for w in why))
    tiny = np.array([[0.01, 0, 900], [0, 0.01, 500], [0, 0, 1.0]])
    okq, _, why = sanity_check(tiny, (1080, 1920, 3), (1080, 1920, 3))
    check("degenerate footprint rejected", not okq)
    good = np.array([[0.25, 0, 500], [0, 0.25, 300], [0, 0, 1.0]])
    okq, _, why = sanity_check(good, (1080, 1920, 3), (1080, 1920, 3))
    check("sane similarity accepted", okq)

    # 5. IoU helper
    shape = (1000, 1000)
    q1 = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], float)
    q2 = np.array([[200, 100], [400, 100], [400, 300], [200, 300]], float)
    iou = quad_iou(q1, q2, shape)
    check(f"IoU of half-overlapping squares == 1/3 (got {iou:.3f})", abs(iou - 1 / 3) < 1e-6)

    log("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--reference-10x", default=str(here / "10x image.jpg"))
    ap.add_argument("--view-40x", default=str(here / "40x image.jpg"))
    ap.add_argument("--view-100x", default=str(here / "100x image.jpg"))
    ap.add_argument("--out-dir", default=str(here / "outputs" / "cross_mag_registration"))
    ap.add_argument("--no-cellpose", action="store_true",
                    help="skip the optional Cellpose RBC validation")
    ap.add_argument("--self-test", action="store_true",
                    help="run internal convention checks and exit")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    t_all = time.perf_counter()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    use_cellpose = not args.no_cellpose

    meta10, img10 = image_meta(Path(args.reference_10x), "10x")
    meta40, img40 = image_meta(Path(args.view_40x), "40x")
    meta100, img100 = image_meta(Path(args.view_100x), "100x")
    log("=== CROSS-MAGNIFICATION LOCALIZATION EXPERIMENT (SIFT) ===")
    log(f"reference 10x : {meta10['filename']}  {meta10['width']}x{meta10['height']}")
    log(f"view     40x : {meta40['filename']}  {meta40['width']}x{meta40['height']}")
    log(f"view    100x : {meta100['filename']}  {meta100['width']}x{meta100['height']}")
    log("NOTE: equal pixel dimensions do NOT mean equal physical field size.")
    log(f"preprocessing strategies: {PREP_STRATEGIES}; Lowe ratio {LOWE_RATIO}; "
        f"RANSAC px {RANSAC_PX}")
    log("")

    # ---------------- Experiment A: 10x -> 40x ----------------
    log("EXPERIMENT A: 10x reference -> 40x field")
    A = register_pair(img10, img40, "A", SCALE_CANDIDATES_A)
    dirA = out_root / "10x_to_40x"
    resA = {"experiment": "A: 10x -> 40x", "status": A["status"], "seconds": A["seconds"],
            "attempts": A["attempts"], "cellpose_validation": None}
    if A.get("ctx"):
        recA, ctxA = A["best"], A["ctx"]
        resA = write_experiment_outputs(dirA, img10, img40, recA, ctxA,
                                        "EXPERIMENT A: 40x FIELD LOCATED IN 10x REFERENCE",
                                        NOMINAL_A)
        resA["runtime_seconds"] = A["seconds"]
        resA["attempts"] = A["attempts"]
        if use_cellpose:
            log("  cellpose validation (10x <-> 40x)...")
            resA["cellpose_validation"] = cellpose_validate(
                img10, img40, ctxA["H"], dirA / "cell_validation.png",
                recA.get("effective_scale"))
            log(f"  -> {resA['cellpose_validation']}")
    else:
        resA["fail_reason"] = A["attempts"][-1].get("fail_reason") if A["attempts"] else "no attempts"
        dirA.mkdir(parents=True, exist_ok=True)
        banner = img10.copy()
        put_text_block(banner, ["EXPERIMENT A FAILED - no reliable 10x -> 40x transform",
                                str(resA["fail_reason"])], org=(12, 40), scale=0.9)
        cv2.imwrite(str(dirA / "reference_location.png"), banner)
    (dirA / "result.json").write_text(json.dumps(jnum(resA), indent=2))
    log(f"  => A: {resA['status']}  ({A['seconds']}s)")
    log("")

    # ---------------- Experiment B: 10x -> 100x DIRECT ----------------
    log("EXPERIMENT B: 10x reference -> 100x field (DIRECT — failure is a valid result)")
    B = register_pair(img10, img100, "B", SCALE_CANDIDATES_B)
    dirB = out_root / "10x_to_100x_direct"
    resB = {"experiment": "B: 10x -> 100x direct", "status": B["status"],
            "seconds": B["seconds"], "attempts": B["attempts"],
            "cellpose_validation": None}
    if B.get("ctx"):
        recB, ctxB = B["best"], B["ctx"]
        resB = write_experiment_outputs(dirB, img10, img100, recB, ctxB,
                                        "EXPERIMENT B: 100x FIELD DIRECTLY IN 10x REFERENCE",
                                        NOMINAL_B)
        resB["runtime_seconds"] = B["seconds"]
        resB["attempts"] = B["attempts"]
        if use_cellpose:
            log("  cellpose validation (10x <-> 100x)...")
            resB["cellpose_validation"] = cellpose_validate(
                img10, img100, ctxB["H"], dirB / "cell_validation.png",
                recB.get("effective_scale"))
            log(f"  -> {resB['cellpose_validation']}")
    else:
        fails = [a.get("fail_reason", "") for a in B["attempts"]]
        resB["fail_reason"] = ("DIRECT_10X_TO_100X = FAILED; no attempt passed "
                               "geometric verification. Reasons observed: "
                               + " | ".join(sorted(set(f for f in fails if f))[:6]))
        banner = img10.copy()
        put_text_block(banner, [
            "DIRECT 10x -> 100x REGISTRATION FAILED",
            "no scale/preprocessing attempt produced a geometrically valid",
            "homography — this is a measured scientific result, not an error.",
        ], org=(12, 40), scale=0.9)
        dirB.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dirB / "reference_location.png"), banner)
    (dirB / "result.json").write_text(json.dumps(jnum(resB), indent=2))
    log(f"  => B: {resB['status']}  ({B['seconds']}s)")
    log("")

    # ---------------- Experiment C: 40x -> 100x ----------------
    log("EXPERIMENT C: 40x image -> 100x field")
    C = register_pair(img40, img100, "C", SCALE_CANDIDATES_C)
    dirC = out_root / "40x_to_100x"
    resC = {"experiment": "C: 40x -> 100x", "status": C["status"],
            "seconds": C["seconds"], "attempts": C["attempts"],
            "cellpose_validation": None}
    if C.get("ctx"):
        recC, ctxC = C["best"], C["ctx"]
        resC = write_experiment_outputs(dirC, img40, img100, recC, ctxC,
                                        "EXPERIMENT C: 100x FIELD LOCATED IN 40x IMAGE",
                                        NOMINAL_C)
        resC["runtime_seconds"] = C["seconds"]
        resC["attempts"] = C["attempts"]
        if use_cellpose:
            log("  cellpose validation (40x <-> 100x)...")
            resC["cellpose_validation"] = cellpose_validate(
                img40, img100, ctxC["H"], dirC / "cell_validation.png",
                recC.get("effective_scale"))
            log(f"  -> {resC['cellpose_validation']}")
    else:
        fails = [a.get("fail_reason", "") for a in C["attempts"]]
        resC["fail_reason"] = "no attempt passed verification: " + " | ".join(
            sorted(set(f for f in fails if f))[:6])
    (dirC / "result.json").write_text(json.dumps(jnum(resC), indent=2))
    log(f"  => C: {resC['status']}  ({C['seconds']}s)")
    log("")

    # ---------------- Hierarchical composition ----------------
    log("HIERARCHICAL: 10x -> 40x -> 100x composition")
    dirH = out_root / "hierarchical_10x_to_100x"
    resH = {"experiment": "hierarchical 10x -> 40x -> 100x", "status": "FAILED",
            "cellpose_validation": None}
    H_40_to_10 = A["ctx"]["H"] if A.get("ctx") else None
    H_100_to_40 = C["ctx"]["H"] if C.get("ctx") else None
    if H_40_to_10 is None or H_100_to_40 is None:
        resH["fail_reason"] = (
            "composition impossible: A status "
            f"{resA['status']}, C status {resC['status']}")
        log(f"  => HIERARCHICAL: FAILED ({resH['fail_reason']})")
        dirH.mkdir(parents=True, exist_ok=True)
        banner = img10.copy()
        put_text_block(banner, ["HIERARCHICAL REGISTRATION NOT POSSIBLE",
                                resH["fail_reason"]], org=(12, 40), scale=0.9)
        cv2.imwrite(str(dirH / "reference_location.png"), banner)
    else:
        H_hier, quad_hier, verify = compose_hierarchies(
            H_40_to_10, H_100_to_40, img10.shape, img40.shape, img100.shape)
        log(f"  composition verification: {verify}")
        w100, h100 = img100.shape[1], img100.shape[0]
        area = float(cv2.contourArea(quad_hier.astype(np.float32)))
        recH = {
            "status": "GOOD" if verify["center_in_10x_bounds"] else "WEAK",
            "strategy": f"{resA['preprocessing_strategy']} + {resC['preprocessing_strategy']}",
            "scale": "composed",
            "quad_in_parent": quad_hier,
            "footprint_area_px2": area,
            "percent_parent_area": area / (img10.shape[0] * img10.shape[1]) * 100.0,
            "bbox_in_parent": quad_bbox(quad_hier, img10.shape),
            "effective_scale": effective_scale(H_hier, img100.shape),
            "inliers": None, "inlier_ratio": None, "reproj_mean_px": None,
        }
        bb = quad_bbox(quad_hier, img10.shape)
        loc = img10.copy()
        draw_quad(loc, quad_hier, COL_HIER, 4, "dashdot")
        put_text_block(loc, [
            "HIERARCHICAL 10x -> 40x -> 100x: 100x FIELD PROJECTED INTO 10x MAP",
            f"H_100_to_10 = H_40_to_10 @ H_100_to_40   (verified: "
            f"max corner err {verify['sequential_vs_composed_max_corner_err_px']:.2e} px)",
            f"footprint: {area:.0f} px2 = {recH['percent_parent_area']:.2f}% of 10x "
            f"(theoretical expectation ~1%: 1/100)",
            f"measured effective scale: {recH['effective_scale']:.2f} (nominal 10)",
            f"status: {recH['status']}",
        ])
        dirH.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dirH / "reference_location.png"), loc)
        if bb:
            crop = img10[bb[1]:bb[3], bb[0]:bb[2]].copy()
            cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), COL_HIER, 3)
            cv2.imwrite(str(dirH / "crop.png"), label_bar(
                crop, f"10x CROP FOR HIERARCHICAL 100x LOCATION bbox={bb}"))
        warp = cv2.warpPerspective(img10, np.linalg.inv(H_hier), (w100, h100))
        cv2.imwrite(str(dirH / "warp.png"), warp)
        sbs_h = 560
        wimg = resize_to_height(warp, sbs_h)
        cimg = resize_to_height(img100, sbs_h)
        ww = max(wimg.shape[1], cimg.shape[1])
        wimg = cv2.copyMakeBorder(wimg, 0, 0, 0, ww - wimg.shape[1], cv2.BORDER_CONSTANT)
        cimg = cv2.copyMakeBorder(cimg, 0, 0, 0, ww - cimg.shape[1], cv2.BORDER_CONSTANT)
        sbs = np.hstack([wimg, np.full((sbs_h, 6, 3), 255, np.uint8), cimg])
        cv2.imwrite(str(dirH / "side_by_side.png"), label_bar(
            fit_width(sbs, 2400),
            "10x REGION VIA HIERARCHICAL TRANSFORM   |   ACTUAL 100x IMAGE"))
        resH = {
            "experiment": "hierarchical 10x -> 40x -> 100x",
            "status": recH["status"],
            "built_on": {"A_status": resA["status"], "C_status": resC["status"]},
            "H_40_to_10": jnum(H_40_to_10),
            "H_100_to_40": jnum(H_100_to_40),
            "H_100_to_10_composed": jnum(H_hier),
            "composition_verification": verify,
            "polygon_in_10x": jnum(quad_hier),
            "footprint_area_px2": area,
            "percent_parent_area": recH["percent_parent_area"],
            "bbox_in_parent": recH["bbox_in_parent"],
            "effective_scale_measured": recH["effective_scale"],
            "confidence": recH["status"],
        }
    (dirH / "result.json").write_text(json.dumps(jnum(resH), indent=2))
    log("")

    # ---------------- Comparison: direct vs hierarchical ----------------
    log("COMPARISON: direct 10x->100x vs hierarchical 10x->40x->100x")
    dirK = out_root / "comparison"
    dirK.mkdir(parents=True, exist_ok=True)
    comparison = {"direct_status": resB["status"], "hierarchical_status": resH["status"]}
    quad_direct = resB.get("polygon_in_parent")
    quad_h = resH.get("polygon_in_10x")
    combined = img10.copy()
    legend_entries = [("EXPERIMENT A", COL_A, "solid", 4,
                       f"40x footprint ({resA['status']})")]
    draw_quad(combined, np.asarray(resA["polygon_in_parent"], np.float64),
              COL_A, 4, "solid") if resA.get("polygon_in_parent") is not None else None
    legend_entries.append(("HIERARCHICAL", COL_HIER, "dashdot", 3,
                           f"100x hierarchical ({resH['status']})"))
    if quad_h is not None:
        draw_quad(combined, np.asarray(quad_h, np.float64), COL_HIER, 3, "dashdot")
    if quad_direct is not None:
        quad_direct = np.asarray(quad_direct, np.float64)
        draw_quad(combined, quad_direct, COL_DIRECT, 3, "dashed")
        legend_entries.append(("DIRECT", COL_DIRECT, "dashed", 3,
                               f"100x direct ({resB['status']})"))
        comparison.update({
            "centroid_direct_px": np.asarray(quad_direct).mean(axis=0).tolist(),
            "centroid_hierarchical_px": np.asarray(quad_h).mean(axis=0).tolist(),
            "centroid_difference_px": float(np.linalg.norm(
                np.asarray(quad_direct).mean(axis=0) - np.asarray(quad_h).mean(axis=0))),
            "corner_mean_difference_px": float(np.linalg.norm(
                quad_direct - np.asarray(quad_h), axis=1).mean()),
            "corner_max_difference_px": float(np.linalg.norm(
                quad_direct - np.asarray(quad_h), axis=1).max()),
            "iou": quad_iou(quad_direct, np.asarray(quad_h), img10.shape),
            "area_direct_px2": resB.get("footprint_area_px2"),
            "area_hierarchical_px2": resH.get("footprint_area_px2"),
            "effective_scale_direct": resB.get("effective_scale_measured"),
            "effective_scale_hierarchical": resH.get("effective_scale_measured"),
        })
        # orientation difference (TL->TR edge)
        def edge_angle(q):
            v = np.asarray(q)[1] - np.asarray(q)[0]
            return math.degrees(math.atan2(v[1], v[0]))
        da = abs(edge_angle(quad_direct) - edge_angle(np.asarray(quad_h))) % 180.0
        comparison["orientation_difference_deg"] = float(min(da, 180.0 - da))
        # zoom panel for direct_vs_hierarchical png
        allq = np.vstack([quad_direct, np.asarray(quad_h)])
        x0, y0 = allq[:, 0].min(), allq[:, 1].min()
        x1, y1 = allq[:, 0].max(), allq[:, 1].max()
        mx, my = 0.35 * (x1 - x0), 0.35 * (y1 - y0)
        bbz = (int(max(0, x0 - mx)), int(max(0, y0 - my)),
               int(min(img10.shape[1], x1 + mx)), int(min(img10.shape[0], y1 + my)))
        zoom = img10[bbz[1]:bbz[3], bbz[0]:bbz[2]].copy()
        off = np.array([bbz[0], bbz[0]], np.float64)
        zd = quad_direct - np.array([bbz[0], bbz[1]])
        zh = np.asarray(quad_h) - np.array([bbz[0], bbz[1]])
        draw_quad(zoom, zd, COL_DIRECT, 3, "dashed")
        draw_quad(zoom, zh, COL_HIER, 3, "dashdot")
    else:
        put_text_block(combined, ["DIRECT 10x -> 100x REGISTRATION FAILED",
                                  "no direct 100x polygon exists to compare"], org=(12, 260),
                      scale=0.85, color=COL_FAIL)
        legend_entries.append(("DIRECT", COL_FAIL, "dashed", 3,
                               "100x direct: FAILED (no polygon)"))
    draw_legend(combined, [(c, s, t, l) for _, c, s, t, l in legend_entries],
                (img10.shape[1] - 560, 12),
                title=f"footprints in 10x   (A={resA['status']})")
    cv2.imwrite(str(dirK / "reference_10x_all_locations.png"), fit_width(combined, 2400))

    if quad_direct is not None and quad_h is not None:
        left = resize_to_height(fit_width(combined, 1100), 700)
        right = resize_to_height(fit_width(zoom, 1200), 700)
        # canvas sized to the actual panels; titles drawn AFTER sizing so the
        # font shrinks to fit each panel and nothing is clipped
        bar_h = 40
        dh = np.full((bar_h + max(left.shape[0], right.shape[0]),
                      left.shape[1] + right.shape[1] + 6, 3), 25, np.uint8)
        dh[bar_h:, :left.shape[1]] = left
        dh[bar_h:, left.shape[1] + 6:left.shape[1] + 6 + right.shape[1]] = right

        def panel_title(x, w, text):
            scale = 0.62
            tw = cv2.getTextSize(text, FONT, scale, 1)[0][0]
            if tw > w - 20:
                scale = max(0.35, scale * (w - 20) / tw)
            cv2.rectangle(dh, (x, 0), (x + w, bar_h), (30, 30, 30), -1)
            cv2.putText(dh, text, (x + 10, 28), FONT, scale, COL_TEXT, 1, cv2.LINE_AA)

        panel_title(0, left.shape[1],
                    "FULL 10x REFERENCE: direct (red dashed) vs hierarchical (magenta)")
        panel_title(left.shape[1] + 6, right.shape[1],
                    "ZOOM: direct (red dashed) vs hierarchical (magenta dashdot)")
        cv2.imwrite(str(dirK / "direct_vs_hierarchical_100x.png"), dh)
    else:
        dh = fit_width(label_bar(combined,
                                 "DIRECT 10x->100x REGISTRATION FAILED - hierarchical-only result"), 2400)
        cv2.imwrite(str(dirK / "direct_vs_hierarchical_100x.png"), dh)
    log(f"  => comparison: {json.dumps({k: v for k, v in comparison.items() if 'px2' not in k}, default=str)[:220]}")

    # ---------------- Summary + report ----------------
    summary = {
        "experiment_A_10_40": resA,
        "experiment_B_10_100_direct": resB,
        "experiment_C_40_100": resC,
        "hierarchical_10_40_100": resH,
        "direct_vs_hierarchical": comparison,
        "image_metadata": [meta10, meta40, meta100],
        "configuration": {
            "preprocessing_strategies": PREP_STRATEGIES,
            "lowe_ratio": LOWE_RATIO,
            "ransac_px": RANSAC_PX,
            "sift_nfeatures": SIFT_NFEATURES,
            "scale_candidates": {"A": SCALE_CANDIDATES_A, "B": SCALE_CANDIDATES_B,
                                 "C": SCALE_CANDIDATES_C},
            "thresholds": {"good_inliers": GOOD_INLIERS,
                           "good_inlier_ratio": GOOD_INLIER_RATIO,
                           "good_reproj_px": GOOD_REPROJ_PX,
                           "good_spread_frac": GOOD_SPREAD_FRAC,
                           "min_footprint_frac": MIN_FOOTPRINT_FRAC,
                           "max_footprint_frac": MAX_FOOTPRINT_FRAC},
            "cellpose_used": bool(use_cellpose),
        },
        "area_theory_check": {
            "note": "sanity reference only, never used to accept/reject",
            "nominal_area_fractions": {"40x_in_10x": 1 / 16, "100x_in_10x": 1 / 100,
                                       "100x_in_40x": 1 / 6.25},
        },
    }
    (dirK / "summary.json").write_text(json.dumps(jnum(summary), indent=2))

    write_report(dirK / "report.md", summary)
    log("")
    log(f"TOTAL: {time.perf_counter() - t_all:.1f}s   outputs -> {out_root}")
    return 0


def write_report(path: Path, s: dict) -> None:
    A, B, C, Hh = (s["experiment_A_10_40"], s["experiment_B_10_100_direct"],
                   s["experiment_C_40_100"], s["hierarchical_10_40_100"])
    cmp_ = s["direct_vs_hierarchical"]

    def cp_validation(res):
        cv_ = res.get("cellpose_validation")
        if not cv_:
            return "skipped"
        if cv_.get("status") == "ok":
            return (f"{cv_['n_matched']}/{cv_['n_child_rbc']} "
                    f"({cv_['match_pct']:.0f}%) err "
                    f"{cv_.get('median_centroid_error_px') and round(cv_['median_centroid_error_px'], 1)}px")
        return cv_.get("status", "n/a")

    def fmt(v, nd=2):
        return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))

    lines = [
        "# Cross-Magnification Registration Experiment — Report",
        "",
        f"Generated by `cross_magnification_register.py` (standalone module, SIFT).",
        "",
        "## Image metadata",
        "",
        "| file | mag | WxH | pixels |",
        "|---|---|---|---|",
    ]
    for m in s["image_metadata"]:
        lines.append(f"| {m['filename']} | {m['magnification']} | "
                     f"{m['width']}x{m['height']} | {m['pixel_count']:,} |")
    lines += [
        "",
        "Equal pixel dimensions do NOT imply equal physical field size — the "
        "objective sets the physical FOV; effective image scale is **measured** "
        "from the transforms, never assumed.",
        "",
        "## Main comparison table",
        "",
        "| Experiment | Status | Good Matches | Inliers | Inlier Ratio | "
        "Reproj Err (px) | Confidence | Cellpose RBC check |",
        "|---|---|---:|---:|---:|---:|---|---|",
        f"| 10x->40x | {A['status']} | {fmt(A.get('good_matches'), 0)} | "
        f"{fmt(A.get('inliers'), 0)} | {fmt(A.get('inlier_ratio'))} | "
        f"{fmt(A.get('reprojection_error_px', {}).get('mean') if A.get('reprojection_error_px') else None)} "
        f"| {A['confidence']} | {cp_validation(A)} |",
        f"| 10x->100x direct | {B['status']} | {fmt(B.get('good_matches'), 0)} | "
        f"{fmt(B.get('inliers'), 0)} | {fmt(B.get('inlier_ratio'))} | "
        f"{fmt(B.get('reprojection_error_px', {}).get('mean') if B.get('reprojection_error_px') else None)} "
        f"| {B.get('confidence', B['status'])} | {cp_validation(B)} |",
        f"| 40x->100x | {C['status']} | {fmt(C.get('good_matches'), 0)} | "
        f"{fmt(C.get('inliers'), 0)} | {fmt(C.get('inlier_ratio'))} | "
        f"{fmt(C.get('reprojection_error_px', {}).get('mean') if C.get('reprojection_error_px') else None)} "
        f"| {C['confidence']} | {cp_validation(C)} |",
        f"| 10x->40x->100x | {Hh['status']} | composed | composed | n/a | "
        f"validation | {Hh.get('confidence')} | n/a |",
        "",
        "## Effective scales (measured vs nominal)",
        "",
        "| path | measured | nominal | deviation |",
        "|---|---:|---:|---:|",
    ]
    for label, res, nom in [("10->40", A, 4.0), ("10->100 direct", B, 10.0),
                            ("40->100", C, 2.5), ("10->100 hierarchical", Hh, 10.0)]:
        es = res.get("effective_scale_measured")
        if es:
            mag = 1.0 / es  # measured linear magnification = 1 / image scale
            meas = f"{mag:.2f}x (scale {es:.3f})"
            dev = f"{(mag / nom - 1) * 100:+.1f}%"
        else:
            meas, dev = "-", "-"
        lines.append(f"| {label} | {meas} | {nom:g}x | {dev} |")
    lines += ["",
              "`measured magnification` = reciprocal of the measured image scale "
              "(parent px per child px); deviation is vs the nominal objective "
              "ratio and reflects camera/adapter optics, not error.",]

    lines += ["", "## Footprint areas vs theory (sanity only)", ""]
    th = s["area_theory_check"]["nominal_area_fractions"]
    for label, res, key, nom_key in [
            ("40x in 10x", A, "percent_parent_area", "40x_in_10x"),
            ("100x direct in 10x", B, "percent_parent_area", "100x_in_10x"),
            ("100x hierarchical in 10x", Hh, "percent_parent_area", "100x_in_10x")]:
        pa = res.get(key if res is not Hh else "percent_parent_area")
        lines.append(f"- {label}: {fmt(pa)}% of reference "
                     f"(theory ~{th[nom_key] * 100:.2f}%)")
    lines.append(f"- 100x in 40x (experiment C): "
                 f"{fmt(C.get('percent_parent_area'))}% of 40x (theory ~{th['100x_in_40x'] * 100:.2f}%)")

    lines += ["", "## Direct vs hierarchical (100x in 10x coordinates)", ""]
    if "iou" in cmp_:
        lines += [
            f"- centroid difference: **{cmp_['centroid_difference_px']:.1f} px**",
            f"- corner mean / max difference: {cmp_['corner_mean_difference_px']:.1f} / "
            f"{cmp_['corner_max_difference_px']:.1f} px",
            f"- polygon IoU: **{cmp_['iou']:.3f}**",
            f"- area direct / hierarchical: {cmp_['area_direct_px2']:.0f} / "
            f"{cmp_['area_hierarchical_px2']:.0f} px2",
            f"- orientation difference: {fmt(cmp_.get('orientation_difference_deg'))} deg",
            f"- scale direct vs hierarchical: {fmt(cmp_.get('effective_scale_direct'))} vs "
            f"{fmt(cmp_.get('effective_scale_hierarchical'))}",
        ]
    else:
        lines.append(f"- DIRECT 10x->100x registration FAILED "
                     f"({B.get('fail_reason', 'no valid transform')[:300]}).")
        lines.append("- Direct-vs-hierarchical quantities are undefined; the "
                     "hierarchical footprint stands alone.")
        lines.append(f"- hierarchical centroid in 10x: "
                     f"{fmt((Hh.get('composition_verification') or {}).get('center_100x_in_10x_hierarchical'))}")

    # Research answers (section 30)
    a_s, b_s, c_s, h_s = A["status"], B["status"], C["status"], Hh["status"]

    def yn(st, good=("GOOD",)):
        return f"**YES** ({st})" if st in good else (
            f"partly ({st})" if st == "WEAK" else f"**NO** ({st})")

    lines += [
        "",
        "## Research questions",
        "",
        f"1. Can SIFT reliably localize 40x directly inside 10x? -> {yn(a_s)}",
        f"2. Can SIFT reliably localize 100x directly inside 10x? -> {yn(b_s)}",
        f"3. Can 100x be reliably localized inside 40x? -> {yn(c_s)}",
        f"4. Does hierarchical composition 10x->40x->100x localize the 100x "
        f"field in the 10x map? -> {yn(h_s)}",
    ]
    if "iou" in cmp_:
        iou = cmp_["iou"]
        stable = ("DIRECT" if (B.get("inliers", 0) * B.get("inlier_ratio", 0)
                               >= (C.get("inliers", 0) or 0)) else "HIERARCHICAL")
        lines.append(f"5. Which is more stable/confident? -> agreement IoU {iou:.3f}; "
                     f"stronger evidence chain: {stable} "
                     "(higher inlier count x ratio); centroid disagreement "
                     f"{cmp_['centroid_difference_px']:.1f}px.")
    else:
        lines.append("5. Not applicable — direct path failed; hierarchical is the "
                     "only available 100x localization (its confidence: "
                     f"{h_s}).")
    lines += [
        "",
        "## Conclusion",
        "",
    ]
    concl = []
    if a_s in ("GOOD", "WEAK") and c_s in ("GOOD", "WEAK") and h_s in ("GOOD", "WEAK"):
        concl.append("Both sub-registrations (10x->40x, 40x->100x) succeeded, so the "
                     "hierarchical chain expresses the 100x field in 10x coordinates.")
    if b_s == "FAILED":
        concl.append("Direct 10x->100x registration FAILED — a single SIFT step across "
                     "a 10x magnification gap did not produce a geometrically valid "
                     "correspondence (measured, not forced).")
        if h_s in ("GOOD", "WEAK"):
            concl.append("HIERARCHICAL localization is therefore the preferred "
                         "architecture on this evidence: the scale gap is broken into "
                         "two manageable steps (4x, 2.5x).")
    elif "iou" in cmp_:
        iou = cmp_["iou"]
        if iou > 0.5:
            concl.append(f"Direct and hierarchical paths AGREE (IoU {iou:.2f}), "
                         "cross-validating each other.")
        else:
            concl.append(f"Direct and hierarchical paths DISAGREE (IoU {iou:.2f}) — "
                         "at least one transform is unreliable; compare reproj/inlier "
                         "evidence before trusting either.")
    if not concl:
        concl.append("Measured statuses did not support a strong claim; inspect "
                     "per-experiment evidence.")
    lines += [f"- {c}" for c in concl]
    lines += [
        "",
        "## Interpretation guardrail",
        "",
        "A failed direct 10x->100x step is NOT a project failure: it is evidence "
        "that a single-step registration across a 10x magnification difference is "
        "too difficult while hierarchical localization through an intermediate "
        "40x scale remains feasible — a meaningful FYP result. SIFT/RANSAC is the "
        "geometric method; Cellpose centroid checks are supporting evidence only.",
        "",
        "## Output artefacts",
        "",
        "```text",
        "10x_to_40x/  10x_to_100x_direct/  40x_to_100x/  hierarchical_10x_to_100x/",
        "comparison/reference_10x_all_locations.png  direct_vs_hierarchical_100x.png",
        "comparison/summary.json",
        "```",
        "",
        "Each experiment folder: keypoints.png, matches.png, inliers.png, "
        "reference_location.png, crop.png, warp.png, side_by_side.png, "
        "cell_validation.png (if enabled), result.json.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
