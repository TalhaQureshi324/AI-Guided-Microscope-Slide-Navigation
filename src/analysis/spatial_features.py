"""Image-level spatial features for monolayer characterisation (Stage B).

These measurements describe HOW cells are distributed - coverage, density,
nearest-neighbour spacing, crowding/contact, adjacency-graph clustering and
grid uniformity. They are computed and stored for later calibration, but Phase 1
deliberately assigns NO monolayer thresholds to them (project prompt, section
27: monolayer detection must not be a naive count rule).

All distance-based features are normalised by the median RBC diameter so the
values remain meaningful if resolution or magnification changes.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def compute_spatial_features(
    rows: List[Dict],
    image_shape: Tuple[int, int],
    ref: Dict,
    contact_margin_fraction: float = 0.15,
    graph_proximity_norm: float = 1.5,
    uniformity_grids: Tuple[int, ...] = (3, 4),
) -> Dict:
    """Compute spatial/crowding features from candidate RBCs.

    Mutates each row dict with per-cell values (``nn_distance_px``,
    ``nn_gap_px``, ``in_contact``, ``cluster_id``, ``cluster_size``) so they can
    be exported in the cell-level CSV.
    """
    height, width = image_shape
    image_area_px = float(height * width)
    median_diameter = ref.get("median_diameter_px", float("nan"))

    features: Dict = {
        "rbc_coverage": float("nan"),
        "rbc_density_per_mpx": float("nan"),
        "nn_median_px": float("nan"),
        "nn_mean_px": float("nan"),
        "nn_std_px": float("nan"),
        "nn_p10_px": float("nan"),
        "nn_p25_px": float("nan"),
        "nn_p75_px": float("nan"),
        "nn_p90_px": float("nan"),
        "nn_median_normalized": float("nan"),
        "gap_median_px": float("nan"),
        "gap_median_normalized": float("nan"),
        "contact_ratio": float("nan"),
        "graph_isolated_pct": float("nan"),
        "graph_mean_degree": float("nan"),
        "graph_max_cluster": float("nan"),
        "graph_median_cluster": float("nan"),
        "graph_cluster_pct": float("nan"),
    }
    for g in uniformity_grids:
        features[f"density_cv_{g}x{g}"] = float("nan")

    # per-cell defaults (kept even when degenerate, for a stable CSV schema)
    for row in rows:
        row["nn_distance_px"] = float("nan")
        row["nn_gap_px"] = float("nan")
        row["in_contact"] = False
        row["cluster_id"] = -1
        row["cluster_size"] = 1

    candidates = [r for r in rows if r.get("rbc_candidate")]
    n = len(candidates)

    # ---- coverage & density ------------------------------------------------
    # Valid pixels: the full frame (no vignetting mask yet - revisit when a
    # calibration frame becomes available).
    features["rbc_coverage"] = float(sum(r["area_px"] for r in candidates) / image_area_px)
    features["rbc_density_per_mpx"] = float(n / (image_area_px / 1.0e6))

    if n == 0 or not np.isfinite(median_diameter) or median_diameter <= 0:
        logger.debug("Too few cells for spatial features (n=%d).", n)
        return features

    points = np.array([[r["centroid_x"], r["centroid_y"]] for r in candidates], dtype=float)
    radii = np.array([r["equivalent_radius_px"] for r in candidates], dtype=float)

    # ---- nearest-neighbour distances ----------------------------------------
    tree = cKDTree(points)
    if n >= 2:
        dist, idx = tree.query(points, k=2)  # k=2: column 0 is the point itself
        nn_dist = dist[:, 1]
        nn_idx = idx[:, 1]
        gaps = nn_dist - radii - radii[nn_idx]

        for row, d, g in zip(candidates, nn_dist, gaps):
            row["nn_distance_px"] = float(d)
            row["nn_gap_px"] = float(g)

        features.update(
            nn_median_px=float(np.median(nn_dist)),
            nn_mean_px=float(np.mean(nn_dist)),
            nn_std_px=float(np.std(nn_dist)),
            nn_p10_px=_percentile(nn_dist, 10),
            nn_p25_px=_percentile(nn_dist, 25),
            nn_p75_px=_percentile(nn_dist, 75),
            nn_p90_px=_percentile(nn_dist, 90),
            nn_median_normalized=float(np.median(nn_dist) / median_diameter),
            gap_median_px=float(np.median(gaps)),
            gap_median_normalized=float(np.median(gaps) / median_diameter),
        )

    # ---- contact / crowding ratio -------------------------------------------
    # Two cells are "in contact" when their centre distance is within a small
    # margin (a fraction of the median diameter) of the sum of their equivalent
    # radii - i.e. masks touch or nearly touch. This is a measurable crowding
    # proxy, NOT a claim about physical optical overlap.
    margin = contact_margin_fraction * median_diameter
    search_radius = float(2.0 * radii.max() + margin) if n >= 2 else 0.0
    in_contact = np.zeros(n, dtype=bool)
    if n >= 2 and search_radius > 0:
        for i, j in tree.query_pairs(search_radius, output_type="ndarray"):
            if (points[i] - points[j]).dot(points[i] - points[j]) == float("inf"):
                continue
            d = float(np.hypot(*(points[i] - points[j])))
            if d <= radii[i] + radii[j] + margin:
                in_contact[i] = True
                in_contact[j] = True
    features["contact_ratio"] = float(in_contact.sum() / n)
    for row, flag in zip(candidates, in_contact):
        row["in_contact"] = bool(flag)

    # ---- adjacency graph / clusters -----------------------------------------
    # Edge when centre distance < graph_proximity_norm x median diameter.
    graph_radius = graph_proximity_norm * median_diameter
    if n >= 2:
        pairs = tree.query_pairs(graph_radius, output_type="ndarray")
        if len(pairs):
            data = np.ones(len(pairs), dtype=np.int8)
            graph = coo_matrix(
                (data, (pairs[:, 0], pairs[:, 1])), shape=(n, n)
            ).tocsr()
        else:
            graph = coo_matrix((n, n), dtype=np.int8).tocsr()
        n_components, labels_comp = connected_components(graph, directed=False)

        cluster_sizes = np.bincount(labels_comp, minlength=n)
        for row, cid, csize in zip(candidates, labels_comp, cluster_sizes[labels_comp]):
            row["cluster_id"] = int(cid)
            row["cluster_size"] = int(csize)

        degrees = np.asarray(graph.sum(axis=1)).ravel()
        multi = cluster_sizes[cluster_sizes >= 2]
        features.update(
            graph_isolated_pct=float((degrees == 0).sum() / n * 100.0),
            graph_mean_degree=float(degrees.mean()),
            graph_max_cluster=float(cluster_sizes.max()),
            graph_median_cluster=float(np.median(multi)) if multi.size else 0.0,
            graph_cluster_pct=float(multi.sum() / n * 100.0),
        )
    else:
        features.update(
            graph_isolated_pct=100.0, graph_mean_degree=0.0,
            graph_max_cluster=1.0, graph_median_cluster=0.0, graph_cluster_pct=0.0,
        )
        if candidates:
            candidates[0]["cluster_id"] = 0
            candidates[0]["cluster_size"] = 1

    # ---- spatial uniformity (grid density CV) --------------------------------
    for g in uniformity_grids:
        counts = np.histogram2d(
            points[:, 1], points[:, 0],
            bins=(g, g),
            range=[[0, height], [0, width]],
        )[0]
        local_density = counts.ravel() / (counts.size / (image_area_px / 1.0e6))  # cells per Mpx
        mean_density = local_density.mean()
        features[f"density_cv_{g}x{g}"] = (
            float(local_density.std() / mean_density) if mean_density > 0 else float("nan")
        )

    return features
