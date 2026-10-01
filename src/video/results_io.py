"""Result writers: results.csv (every raw frame), video_frame_features.csv
(accepted keyframes, spec section 35 schema), results_summary.csv,
video_summary.json (spec section 36).

``results.csv`` is the master table requested for the project log: one row per
RAW video frame with every measured statistic (motion, quality, and - when the
frame was an accepted keyframe - all Cellpose/monolayer fields; empty for
skipped frames). ``video_frame_features.csv`` keeps the spec's exact
accepted-keyframes schema so external tooling can rely on it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

# Spec section 35 column order (accepted-keyframe schema), followed by
# measured extras. Keeping the spec order first keeps the file stable.
FRAME_FEATURE_COLUMNS = [
    "video_name", "frame_index", "timestamp_sec", "accepted_keyframe", "skip_reason",
    "dx", "dy", "cumulative_displacement", "sharpness", "brightness",
    "cellpose_used", "cellpose_instance_count", "rbc_candidate_count",
    "valid_region_rbc_count", "rbc_density", "coverage", "median_rbc_diameter",
    "median_nn", "normalized_median_nn", "median_edge_gap", "normalized_edge_gap",
    "contact_ratio", "pct_0_neighbors", "pct_1_neighbor", "pct_2_neighbors",
    "pct_3plus_neighbors", "mean_graph_degree", "largest_cluster_fraction",
    "density_cv", "merge_suspect_count",
    "raw_monolayer_score", "smoothed_monolayer_score", "raw_class", "smoothed_class",
    "fast_screen_result", "cellpose_time_ms", "feature_time_ms", "total_time_ms",
]

EXTRA_COLUMNS = [
    "keyframe_index", "contrast", "saturation", "overexposed_pct", "underexposed_pct",
    "motion_response", "nn_p10_norm", "nn_p25_norm", "nn_p75_norm",
    "median_graph_degree", "n_clusters", "isolated_pct", "median_cluster_size",
    "largest_cluster_size", "pct_in_large_clusters", "density_mean_per_mpx",
    "density_max_min_ratio", "screen_occupancy", "screen_edge_density",
    "screen_texture_std", "screen_brightness", "screen_time_ms",
    "cellpose_model_time_ms", "is_first_inference", "thick_signal_count",
    "thin_signal_count",
]

RESULTS_COLUMNS = FRAME_FEATURE_COLUMNS + EXTRA_COLUMNS

SUMMARY_COLUMNS = [
    "video_name", "duration_sec", "width", "height", "fps_avg", "raw_frames",
    "accepted_keyframes", "keyframes_with_cellpose", "keyframes_fast_only",
    "skipped_stationary", "skipped_rate_limit", "skipped_blur",
    "class_RAW_TOO_THICK", "class_RAW_MONOLAYER", "class_RAW_TOO_THIN",
    "class_RAW_UNCERTAIN", "class_SMOOTHED_TOO_THICK", "class_SMOOTHED_MONOLAYER",
    "class_SMOOTHED_TOO_THIN", "class_SMOOTHED_UNCERTAIN",
    "longest_monolayer_run_keyframes", "longest_monolayer_run_sec",
    "rbc_count_min", "rbc_count_median", "rbc_count_max",
    "raw_score_min", "raw_score_median", "raw_score_max",
    "cellpose_ms_mean", "cellpose_ms_median", "feature_ms_mean",
    "total_ms_per_keyframe_mean", "total_ms_per_keyframe_median",
    "decode_ms_total", "motion_quality_ms_per_frame_mean",
    "render_encode_ms_total", "total_pipeline_sec",
    "estimated_analysis_fps",
]


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}" if v == v else ""  # NaN -> empty
    return v


def write_results_csv(path: Path, rows: List[Dict]) -> None:
    """Master per-raw-frame table (results.csv)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(row.get(k)) for k in RESULTS_COLUMNS})


def write_frame_features_csv(path: Path, rows: List[Dict]) -> None:
    """Accepted-keyframes-only table with the spec section 35 schema."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=FRAME_FEATURE_COLUMNS + EXTRA_COLUMNS, extrasaction="ignore"
        )
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(row.get(k)) for k in FRAME_FEATURE_COLUMNS + EXTRA_COLUMNS})


def write_summary_csv(path: Path, summary: Dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerow({k: _fmt(summary.get(k)) for k in SUMMARY_COLUMNS})


def write_summary_json(path: Path, summary: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
