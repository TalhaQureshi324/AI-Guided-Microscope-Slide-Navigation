"""Per-field monolayer features, prototype score and classification
(spec sections 11, 13-21, 26-27).

Everything here operates on ONE accepted keyframe's Cellpose output.
Central valid analysis region (valid ROI): a cell contributes to
count/density/graph features when its *centroid* lies inside the ROI; mask
pixels are additionally cropped to the ROI for coverage so border cells
neither inflate nor destabilise the field statistics. Border cells are never
deleted - full-frame morphology survives in the rows.

ALL thresholds/weights/ranges below are EXPERIMENTAL PROTOTYPE values:
  * the count gate mirrors the static-baseline hypothesis (950-1050 RBC
    candidates per full frame, same microscope/camera/FOV) and is applied to
    the FULL-frame candidate count for direct comparability with the frozen
    static experiment - NOT a validated medical threshold;
  * the score-component ranges (density/coverage/spacing) are anchored to the
    measured ranges of the 10 static baseline images (coverage 0.34-0.39,
    normalized NN ~0.96-1.07, density ~450-540 cells/Mpx), which are assumed
    to be monolayer-like fields.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MONOLAYER = "MONOLAYER"
TOO_THICK = "TOO_THICK"
TOO_THIN = "TOO_THIN"
UNCERTAIN = "UNCERTAIN"


def valid_roi(shape: Tuple[int, int], margin_fraction: float) -> Tuple[int, int, int, int]:
    """Central ROI as (x0, y0, x1, y1) leaving ``margin_fraction`` per side."""
    h, w = shape
    mx = int(round(w * margin_fraction))
    my = int(round(h * margin_fraction))
    return mx, my, w - mx, h - my


def compute_field_features(
    rows: List[Dict],
    labels: np.ndarray,
    ref: Dict,
    valid_roi_margin: float = 0.10,
    contact_margin_fraction: float = 0.15,
    neighbor_distance_threshold: float = 1.5,
    large_cluster_min_cells: int = 30,
    uniformity_grid: int = 4,
    reference_shape: Optional[Tuple[int, int]] = None,
) -> Dict:
    """Compute all per-field features for one keyframe (schema of spec §35).

    ``reference_shape``: (width, height) of the calibration FOV (default
    1920x1080 - the static-baseline camera). Density is expressed per
    REFERENCE-FOV Mpx so that a smaller camera sampling the SAME optical
    field yields the same density (raw px/Mpx would inflate ~5.8x on an
    800x448 sensor and wrongly drag the score's density component to 0).
    Ratios (coverage, NN/diam, CV) are resolution-invariant already.
    """
    h, w = labels.shape
    x0, y0, x1, y1 = valid_roi(labels.shape, valid_roi_margin)
    roi_w, roi_h = x1 - x0, y1 - y0
    roi_area = float(roi_w * roi_h)

    # FOV-normalised ROI area: how many reference-frame Mpx this ROI covers.
    if reference_shape:
        ref_area = float(reference_shape[0] * reference_shape[1])
        roi_area_ref_mpx = roi_area * (ref_area / float(w * h)) / 1.0e6
    else:
        roi_area_ref_mpx = roi_area / 1.0e6

    candidates = [r for r in rows if r.get("rbc_candidate")]
    median_diameter = ref.get("median_diameter_px", float("nan"))

    features: Dict = {
        "cellpose_instance_count": int(len(rows)),
        "rbc_candidate_count": int(len(candidates)),
        "merge_suspect_count": int(sum(1 for r in rows if r.get("possible_merged_rbc"))),
        "median_rbc_diameter": float(median_diameter) if np.isfinite(median_diameter) else float("nan"),
        "valid_region_rbc_count": 0,
        "rbc_density": float("nan"),
        "coverage": float("nan"),
        "median_nn": float("nan"),
        "normalized_median_nn": float("nan"),
        "nn_p10_norm": float("nan"),
        "nn_p25_norm": float("nan"),
        "nn_p75_norm": float("nan"),
        "median_edge_gap": float("nan"),
        "normalized_edge_gap": float("nan"),
        "contact_ratio": float("nan"),
        "pct_0_neighbors": float("nan"),
        "pct_1_neighbor": float("nan"),
        "pct_2_neighbors": float("nan"),
        "pct_3plus_neighbors": float("nan"),
        "mean_graph_degree": float("nan"),
        "median_graph_degree": float("nan"),
        "n_clusters": float("nan"),
        "isolated_pct": float("nan"),
        "median_cluster_size": float("nan"),
        "largest_cluster_size": float("nan"),
        "largest_cluster_fraction": float("nan"),
        "pct_in_large_clusters": float("nan"),
        "density_mean_per_mpx": float("nan"),
        "density_cv": float("nan"),
        "density_max_min_ratio": float("nan"),
    }

    # coverage: mask pixels clipped to the ROI (border cells only count where
    # they actually overlap the valid region).
    labels_crop = labels[y0:y1, x0:x1]
    features["coverage"] = float((labels_crop > 0).mean())

    valid = [
        r for r in candidates
        if (x0 <= r["centroid_x"] < x1) and (y0 <= r["centroid_y"] < y1)
    ]
    n = len(valid)
    features["valid_region_rbc_count"] = n
    features["rbc_density"] = float(n / roi_area_ref_mpx) if roi_area_ref_mpx > 0 else float("nan")

    if n == 0 or not np.isfinite(median_diameter) or median_diameter <= 0:
        return features

    pts = np.array([[r["centroid_x"], r["centroid_y"]] for r in valid], dtype=float)
    radii = np.array([r["equivalent_radius_px"] for r in valid], dtype=float)
    tree = cKDTree(pts)

    # ---- nearest neighbours & edge gaps ---------------------------------
    if n >= 2:
        dist, idx = tree.query(pts, k=2)
        nn = dist[:, 1]
        gaps = nn - radii - radii[idx[:, 1]]
        features.update(
            median_nn=float(np.median(nn)),
            normalized_median_nn=float(np.median(nn) / median_diameter),
            nn_p10_norm=float(np.percentile(nn, 10) / median_diameter),
            nn_p25_norm=float(np.percentile(nn, 25) / median_diameter),
            nn_p75_norm=float(np.percentile(nn, 75) / median_diameter),
            median_edge_gap=float(np.median(gaps)),
            normalized_edge_gap=float(np.median(gaps) / median_diameter),
        )

    # ---- contact / crowding ratio ----------------------------------------
    # Same definition as the static baseline (src/analysis/spatial_features.py):
    # two cells are "in contact" when centre distance <= ri + rj + margin,
    # margin = contact_margin_fraction x median diameter.
    if n >= 2:
        margin = contact_margin_fraction * median_diameter
        search_r = float(2.0 * radii.max() + margin)
        in_contact = np.zeros(n, dtype=bool)
        contact_pairs = tree.query_pairs(search_r, output_type="ndarray")
        for i, j in contact_pairs:
            d = float(np.hypot(*(pts[i] - pts[j])))
            if d <= radii[i] + radii[j] + margin:
                in_contact[i] = True
                in_contact[j] = True
        features["contact_ratio"] = float(in_contact.mean())

    # ---- close-neighbour distribution (crowding) -------------------------
    close_r = neighbor_distance_threshold * median_diameter
    pairs = tree.query_pairs(close_r, output_type="ndarray")
    degrees = np.zeros(n, dtype=np.int32)
    if len(pairs):
        np.add.at(degrees, pairs[:, 0], 1)
        np.add.at(degrees, pairs[:, 1], 1)
    features.update(
        pct_0_neighbors=float((degrees == 0).mean() * 100.0),
        pct_1_neighbor=float((degrees == 1).mean() * 100.0),
        pct_2_neighbors=float((degrees == 2).mean() * 100.0),
        pct_3plus_neighbors=float((degrees >= 3).mean() * 100.0),
        mean_graph_degree=float(degrees.mean()),
        median_graph_degree=float(np.median(degrees)),
    )

    # ---- adjacency clusters ----------------------------------------------
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    ).tocsr()
    n_comp, comp = connected_components(graph, directed=False)
    sizes = np.bincount(comp, minlength=n)
    large = int((sizes >= large_cluster_min_cells).sum())
    features.update(
        n_clusters=int(n_comp),
        isolated_pct=float((sizes == 1).mean() * 100.0),
        median_cluster_size=float(np.median(sizes[sizes >= 2])) if (sizes >= 2).any() else 0.0,
        largest_cluster_size=int(sizes.max()),
        largest_cluster_fraction=float(sizes.max() / n),
        pct_in_large_clusters=float(sizes[sizes >= large_cluster_min_cells].sum() / n * 100.0),
    )

    # ---- spatial uniformity over the ROI (density CV on a grid) ----------
    local = np.histogram2d(
        pts[:, 1] - y0, pts[:, 0] - x0, bins=(uniformity_grid, uniformity_grid),
        range=[[0, roi_h], [0, roi_w]],
    )[0] / roi_area_ref_mpx
    mean_d = float(local.mean())
    features.update(
        density_mean_per_mpx=mean_d,
        density_cv=float(local.std() / mean_d) if mean_d > 0 else float("nan"),
        density_max_min_ratio=float(local.max() / local.min()) if local.min() > 0 else float("inf"),
    )
    return features


# --------------------------------------------------------------------- score
def _tri(x: float, lo: float, hi: float, wing: float) -> float:
    """Triangular suitability: 1 inside [lo, hi], linear falloff over ``wing``."""
    if not np.isfinite(x):
        return 0.0
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return float(max(0.0, 1.0 - (lo - x) / wing)) if wing > 0 else 0.0
    return float(max(0.0, 1.0 - (x - hi) / wing)) if wing > 0 else 0.0


def prototype_monolayer_score(features: Dict, score_cfg: Dict) -> Tuple[float, Dict]:
    """Experimental 0..1 monolayer suitability score (spec section 26).

    Interpretable weighted mix of suitability components; weights and ranges
    are configuration, to be calibrated later against manually labelled fields.
    Returns (score, per-component values) so the CSV can show WHY.
    """
    c = score_cfg
    comps = {
        "comp_density": _tri(
            features["rbc_density"],
            c["density_lo"], c["density_hi"], c["density_wing"],
        ),
        "comp_coverage": _tri(
            features["coverage"],
            c["coverage_lo"], c["coverage_hi"], c["coverage_wing"],
        ),
        "comp_spacing": _tri(
            features["normalized_median_nn"],
            c["spacing_lo"], c["spacing_hi"], c["spacing_wing"],
        ),
        "comp_low_crowding": (
            1.0 - min(1.0, max(0.0, features["contact_ratio"]))
            if np.isfinite(features["contact_ratio"]) else 0.0
        ),
        "comp_low_clustering": (
            1.0 - min(1.0, max(0.0, features["largest_cluster_fraction"]))
            if np.isfinite(features["largest_cluster_fraction"]) else 0.0
        ),
        "comp_uniformity": (
            1.0 - min(1.0, max(0.0, features["density_cv"] / c["uniformity_cv_ref"]))
            if np.isfinite(features["density_cv"]) else 0.0
        ),
    }
    weights = c["weights"]
    total_w = sum(weights[k] for k in comps)
    score = (
        sum(weights[k] * v for k, v in comps.items()) / total_w if total_w > 0 else 0.0
    )
    return float(min(1.0, max(0.0, score))), comps


# ---------------------------------------------------------------- classifier
def classify_field(
    features: Dict, score: float, cls_cfg: Dict,
    screen_occupancy: Optional[float] = None,
) -> Tuple[str, Dict]:
    """Prototype three-class decision + UNCERTAIN (spec sections 13, 27).

    Evidence-vote design: TOO_THICK / TOO_THIN require at least
    ``min_signals`` independent signals to agree, so a single feature (e.g.
    count alone) can never decide the class.

    Gross-occupancy override (2026-10-01, operator-approved): when the cheap
    pre-Cellpose screen reports the frame is mostly dark foreground
    (``screen_occupancy`` >= ``thick_occupancy_min``), the field is
    TOO_THICK outright. Rationale: extremely clumped networks defeat
    segmentation (few giant masks, low candidate count), which would
    otherwise read as bogus thin signals - a region that is mostly
    foreground is thick regardless of what the failed count says, and
    navigation only needs "move away" (spec section 23).
    """
    count = features["rbc_candidate_count"]  # full-frame, comparable to static baseline

    gross_min = cls_cfg.get("thick_occupancy_min")
    screen_gross = bool(
        gross_min is not None
        and screen_occupancy is not None
        and np.isfinite(screen_occupancy)
        and screen_occupancy >= gross_min
    )
    if screen_gross:
        return TOO_THICK, {
            "thick_signals": {"screen_occupancy_gross": True},
            "thin_signals": {},
            "thick_signal_count": 1,
            "thin_signal_count": 0,
            "screen_gross_thick": True,
        }
    thick_signals = {
        "count_above_gate": count > cls_cfg["prototype_count_max"],
        "coverage_high": features["coverage"] > cls_cfg["thick_coverage_min"],
        "degree_high": features["mean_graph_degree"] > cls_cfg["thick_degree_min"],
        "large_cluster_dominant":
            features["largest_cluster_fraction"] > cls_cfg["thick_cluster_frac_min"],
    }
    thin_signals = {
        "count_below_gate": count < cls_cfg["prototype_count_min"],
        "coverage_low": features["coverage"] < cls_cfg["thin_coverage_max"],
        "spacing_wide": features["normalized_median_nn"] > cls_cfg["thin_nn_norm_min"],
    }

    evidence = {
        "thick_signal_count": int(sum(thick_signals.values())),
        "thin_signal_count": int(sum(thin_signals.values())),
    }

    if evidence["thick_signal_count"] >= cls_cfg["min_signals"]:
        raw = TOO_THICK
    elif evidence["thin_signal_count"] >= cls_cfg["min_signals"]:
        raw = TOO_THIN
    elif (
        score >= cls_cfg["monolayer_score_min"]
        and cls_cfg["prototype_count_min"] <= count <= cls_cfg["prototype_count_max"]
    ):
        raw = MONOLAYER
    else:
        raw = UNCERTAIN

    return raw, {"thick_signals": thick_signals, "thin_signals": thin_signals, **evidence}
