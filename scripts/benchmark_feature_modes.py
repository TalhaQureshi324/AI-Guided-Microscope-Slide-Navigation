"""Benchmark live vs detailed feature modes (spec §29-§30).

Questions answered with measurements (not assumptions):
  1. How much wall time does the DT merge-suspicion analysis add per field?
  2. Does skipping it change the raw classification / monolayer score?
     (Code inspection says no - score/classification never read merge flags -
     this script PROVES it on real data.)

Runs the full feature stage on the 78 saved masks from outputs/video_prototype_01
in both modes and reports timing + decision equality.

Usage:
    python scripts/benchmark_feature_modes.py [--masks outputs/video_prototype_01/masks]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.cell_features import (  # noqa: E402
    compute_reference_stats,
    flag_cells,
    measure_instances,
)
from src.postprocessing.merged_cell_splitter import analyze_merge_suspicion  # noqa: E402
from src.video.monolayer import classify_field, compute_field_features, prototype_monolayer_score  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--masks", type=Path,
                   default=REPO_ROOT / "outputs" / "video_prototype_01" / "masks")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "video.yaml")
    args = p.parse_args()

    vcfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    f_cfg, s_cfg, c_cfg = vcfg["field"], vcfg["monolayer_score"], vcfg["classification"]
    f_cfg = dict(f_cfg)
    f_cfg["uniformity_grid"] = int(f_cfg.get("uniformity_grid", 4))

    masks = sorted(args.masks.glob("*.png"))
    print(f"{len(masks)} masks -> benchmarking LIVE vs DETAILED feature modes ...")

    live_ms, detailed_ms, mismatches = [], [], 0
    per_field = []
    for m in masks:
        labels = cv2.imread(str(m), cv2.IMREAD_UNCHANGED).astype(np.int32)

        # shared prefix (identical in both modes)
        t0 = time.perf_counter()
        rows = measure_instances(labels)
        ref = compute_reference_stats(rows, stats_scope="interior")
        flag_cells(rows, ref, small_area_factor=0.35, wbc_area_factor=3.5)

        # LIVE mode: no DT merge analysis
        rows_live = [dict(r) for r in rows]
        f_live = compute_field_features(
            rows_live, labels, ref, valid_roi_margin=f_cfg["valid_roi_margin"],
            contact_margin_fraction=f_cfg["contact_margin_fraction"],
            neighbor_distance_threshold=f_cfg["neighbor_distance_threshold"],
            large_cluster_min_cells=f_cfg["large_cluster_min_cells"],
            uniformity_grid=f_cfg["uniformity_grid"])
        s_live, _ = prototype_monolayer_score(f_live, s_cfg)
        c_live, _ = classify_field(f_live, s_live, c_cfg)
        t_live = (time.perf_counter() - t0) * 1000.0

        # DETAILED mode: with DT merge analysis (same params as the frozen
        # video/static drivers: multi-signal defaults)
        t0 = time.perf_counter()
        analyze_merge_suspicion(labels, rows, ref,
                                area_factor_min=1.6, solidity_max=0.90,
                                elongation_min=1.5, min_signals=2)
        f_det = compute_field_features(
            rows, labels, ref, valid_roi_margin=f_cfg["valid_roi_margin"],
            contact_margin_fraction=f_cfg["contact_margin_fraction"],
            neighbor_distance_threshold=f_cfg["neighbor_distance_threshold"],
            large_cluster_min_cells=f_cfg["large_cluster_min_cells"],
            uniformity_grid=f_cfg["uniformity_grid"])
        s_det, _ = prototype_monolayer_score(f_det, s_cfg)
        c_det, _ = classify_field(f_det, s_det, c_cfg)
        t_det = (time.perf_counter() - t0) * 1000.0

        live_ms.append(t_live)
        detailed_ms.append(t_det)
        if abs(s_live - s_det) > 1e-9 or c_live != c_det:
            mismatches += 1
            per_field.append({"mask": m.name, "live": c_live, "detailed": c_det,
                              "s_live": round(s_live, 5), "s_det": round(s_det, 5)})

    report = {
        "fields": len(masks),
        "live_mode_ms_mean": round(statistics.mean(live_ms), 1),
        "live_mode_ms_median": round(statistics.median(live_ms), 1),
        "live_mode_ms_max": round(max(live_ms), 1),
        "detailed_mode_ms_mean": round(statistics.mean(detailed_ms), 1),
        "detailed_mode_ms_median": round(statistics.median(detailed_ms), 1),
        "detailed_mode_ms_max": round(max(detailed_ms), 1),
        "merge_dt_overhead_ms_mean": round(statistics.mean(detailed_ms) - statistics.mean(live_ms), 1),
        "decision_mismatches": mismatches,
        "mismatch_examples": per_field[:5],
    }
    print(json.dumps(report, indent=2))
    out = REPO_ROOT / "outputs" / "benchmarks" / "feature_modes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
