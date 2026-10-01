"""Controlled resolution benchmark for live Cellpose inference (spec §31).

Runs the SAME accepted fields at native 1920 px and reduced widths (1280, 960)
and compares: inference time, instance count, RBC candidates, monolayer score
and raw class. Reduced resolution is only worth adopting if decisions stay
equivalent - this script measures that, it does not assume it.

Usage:
    python scripts/benchmark_resolution.py [--fields 65:77] [--widths 1280,960]
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
from src.segmentation.cellpose_segmenter import CellposeSegmenter  # noqa: E402
from src.video.monolayer import classify_field, compute_field_features, prototype_monolayer_score  # noqa: E402


def analyse_frame(segmenter, rgb, width, vcfg):
    if width and width < rgb.shape[1]:
        scale = width / rgb.shape[1]
        small = cv2.resize(rgb, (width, int(round(rgb.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
        seg = segmenter.segment(small)
        labels = cv2.resize(seg.labels, (rgb.shape[1], rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(np.int32)
        inf_ms = seg.timings_ms["inference_ms"]
    else:
        seg = segmenter.segment(rgb)
        labels = seg.labels
        inf_ms = seg.timings_ms["inference_ms"]

    rows = measure_instances(labels)
    ref = compute_reference_stats(rows, stats_scope="interior")
    flag_cells(rows, ref, 0.35, 3.5)
    f = compute_field_features(rows, labels, ref,
                               valid_roi_margin=vcfg["field"]["valid_roi_margin"],
                               contact_margin_fraction=vcfg["field"]["contact_margin_fraction"],
                               neighbor_distance_threshold=vcfg["field"]["neighbor_distance_threshold"],
                               large_cluster_min_cells=vcfg["field"]["large_cluster_min_cells"],
                               uniformity_grid=int(vcfg["field"]["uniformity_grid"]))
    score, _ = prototype_monolayer_score(f, vcfg["monolayer_score"])
    cls, _ = classify_field(f, score, vcfg["classification"])
    return {
        "inf_ms": round(inf_ms, 0),
        "instances": len(rows),
        "rbc": f["rbc_candidate_count"],
        "coverage": round(f["coverage"], 3),
        "score": round(score, 3),
        "class": cls,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keyframes", type=Path,
                   default=REPO_ROOT / "outputs" / "video_prototype_01" / "keyframes")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "video.yaml")
    p.add_argument("--fields", type=str, default="65:77",
                   help="keyframe index range, e.g. 65:77")
    p.add_argument("--widths", type=str, default="1280,960")
    args = p.parse_args()

    lo, hi = (int(x) for x in args.fields.split(":"))
    widths = [None] + [int(w) for w in args.widths.split(",")]

    frames = sorted(args.keyframes.glob("keyframe_*.jpg"))
    selected = [f for f in frames if lo <= int(f.stem.split("_")[1]) < hi]
    print(f"{len(selected)} keyframes (KF{lo}..KF{hi - 1}) x widths {[w or 'native' for w in widths]}")

    vcfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    segmenter = CellposeSegmenter(model_name=vcfg["model"]["name"], gpu="auto",
                                  use_bfloat16=False)

    results = []
    for f in selected:
        bgr = cv2.imread(str(f))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        row = {"keyframe": f.stem}
        for w in widths:
            res = analyse_frame(segmenter, rgb, w, vcfg)
            row["native" if w is None else str(w)] = res
        results.append(row)
        print(f"{f.stem}: " + " | ".join(
            f"{'nat' if w is None else w}: {row['native' if w is None else str(w)]['rbc']}rbc "
            f"{row['native' if w is None else str(w)]['inf_ms']:.0f}ms {row['native' if w is None else str(w)]['class']}"
            for w in widths))

    # ---- aggregate ------------------------------------------------------
    agg = {}
    for w in widths:
        key = "native" if w is None else str(w)
        vals = [r[key] for r in results]
        native_classes = [r["native"]["class"] for r in results]
        agree = sum(1 for r, nc in zip(results, native_classes) if r[key]["class"] == nc)
        agg[key] = {
            "inf_ms_mean": round(statistics.mean(v["inf_ms"] for v in vals), 0),
            "rbc_mean": round(statistics.mean(v["rbc"] for v in vals), 1),
            "rbc_mean_abs_delta_vs_native": round(statistics.mean(
                abs(v["rbc"] - nv["rbc"]) for v, nv in zip(vals, results and [r["native"] for r in results])
            ), 1) if key != "native" else 0.0,
            "class_agreement_with_native": f"{agree}/{len(results)}",
            "score_mean": round(statistics.mean(v["score"] for v in vals), 3),
        }
    report = {"fields": len(results), "per_keyframe": results, "aggregate": agg}
    print(json.dumps(agg, indent=2))
    out = REPO_ROOT / "outputs" / "benchmarks" / "resolution_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
