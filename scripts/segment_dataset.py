"""Phase 1 pipeline: Cellpose segmentation + measurements over the dataset.

Runs the cpsam_v2 baseline (or any configured variant) over every microscope
image in ``Dataset/`` and produces, per experiment directory under ``outputs/``:

  overlays/            annotated diagnostic images (boundaries, centroids, IDs,
                       flags, banner with counts and runtime)
  masks/               original Cellpose integer label masks (16-bit PNG)
  debug_merged/        zoomed crops of suspicious merged masks with evidence
  cell_features.csv    one row per detected object (morphology + flags)
  image_features.csv   one row per image (counts + spatial features + timings)
  summary.json         aggregate results and performance statistics
  run_metadata.json    timestamp, git commit, library versions, device
  config_used.yaml     exact configuration for this experiment
  run.log              full debug log

Usage (from repository root):

    python scripts/segment_dataset.py
    python scripts/segment_dataset.py --experiment-name cpsam_v2_baseline
    python scripts/segment_dataset.py --limit 1 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.cell_features import (  # noqa: E402
    compute_reference_stats,
    flag_cells,
    measure_instances,
)
from src.analysis.spatial_features import compute_spatial_features  # noqa: E402
from src.postprocessing.merged_cell_splitter import analyze_merge_suspicion  # noqa: E402
from src.segmentation.cellpose_segmenter import CellposeSegmenter  # noqa: E402
from src.utils.image_io import (  # noqa: E402
    discover_images,
    load_image,
    save_mask_png,
    save_overlay_jpg,
)
from src.utils.runtime import Timer, collect_env_metadata, setup_logging  # noqa: E402
from src.visualization.overlays import (  # noqa: E402
    render_merge_debug_crops,
    render_overlay,
)

logger = logging.getLogger("segment_dataset")

CELL_COLUMNS = [
    "image", "instance_id", "centroid_x", "centroid_y", "area_px",
    "equivalent_diameter_px", "equivalent_radius_px", "perimeter_px",
    "circularity", "solidity", "eccentricity", "axis_major_px", "axis_minor_px",
    "elongation", "bbox_min_x", "bbox_min_y", "bbox_max_x", "bbox_max_y",
    "touches_border", "rbc_candidate", "possible_small_artifact",
    "possible_wbc", "possible_merged_rbc", "n_dt_peaks",
    "area_ratio_vs_median", "nn_distance_px", "nn_gap_px", "in_contact",
    "cluster_id", "cluster_size",
]

IMAGE_COLUMNS = [
    "image", "width", "height", "channels",
    "total_instances", "rbc_candidate_count", "interior_rbc_count",
    "border_rbc_count", "border_fraction",
    "rbc_coverage", "rbc_density_per_mpx",
    "median_area_px", "median_diameter_px", "iqr_area_px",
    "nn_median_px", "nn_mean_px", "nn_std_px", "nn_p10_px", "nn_p25_px",
    "nn_p75_px", "nn_p90_px", "nn_median_normalized",
    "gap_median_px", "gap_median_normalized",
    "contact_ratio", "graph_isolated_pct", "graph_mean_degree",
    "graph_max_cluster", "graph_median_cluster", "graph_cluster_pct",
    "density_cv_3x3", "density_cv_4x4",
    "n_merged_suspect", "n_small_artifact", "n_wbc",
    "cellprob_threshold", "flow_threshold", "min_size", "diameter_setting",
    "model_load_s", "load_ms", "inference_ms", "postprocess_ms",
    "features_ms", "overlay_ms", "total_ms",
    "manual_label",
]


def process_image(ctx: Dict, path: Path) -> Dict:
    """Run every stage for one image and return its image-level feature row."""
    seg: CellposeSegmenter = ctx["segmenter"]
    cfg = ctx["config"]
    timer = Timer()

    image_rgb = load_image(path)
    timer.lap("load_ms")
    height, width = image_rgb.shape[:2]

    result = seg.segment(image_rgb)
    labels = result.labels
    timer.lap("inference_ms")

    # ---- Stage B: measurements, reference stats, flagging -------------------
    rows = measure_instances(labels)
    ref = compute_reference_stats(rows, cfg["filtering"]["stats_scope"])
    rows = flag_cells(
        rows,
        ref,
        small_area_factor=cfg["filtering"]["small_area_factor"],
        wbc_area_factor=cfg["filtering"]["wbc_area_factor"],
    )
    rows, peaks_by_id = analyze_merge_suspicion(
        labels,
        rows,
        ref,
        area_factor_min=cfg["merge_detection"]["area_factor_min"],
        solidity_max=cfg["merge_detection"]["solidity_max"],
        elongation_min=cfg["merge_detection"]["elongation_min"],
        min_signals=cfg["merge_detection"]["min_signals"],
    )
    timer.lap("postprocess_ms")

    spatial = compute_spatial_features(
        rows,
        (height, width),
        ref,
        contact_margin_fraction=cfg["contact"]["margin_fraction"],
        graph_proximity_norm=cfg["contact"]["proximity_norm"],
        uniformity_grids=tuple(cfg["uniformity"]["grids"]),
    )
    timer.lap("features_ms")

    # ---- diagnostics ----------------------------------------------------------
    overlay = render_overlay(
        image_rgb,
        labels,
        rows,
        title=path.name,
        inference_ms=result.timings_ms["inference_ms"],
        show_instance_ids=cfg["visualization"]["show_instance_ids"],
        min_area_for_id=cfg["visualization"]["min_area_for_id"],
        banner=cfg["visualization"]["banner"],
    )
    stem = path.stem
    save_overlay_jpg(ctx["dirs"]["overlays"] / f"{stem}.jpg", overlay)
    save_mask_png(ctx["dirs"]["masks"] / f"{stem}.png", labels)

    for crop in render_merge_debug_crops(image_rgb, labels, rows, peaks_by_id):
        save_overlay_jpg(
            ctx["dirs"]["debug_merged"] / f"{stem}_id{crop['instance_id']}.jpg",
            crop["image"],
        )
    timer.lap("overlay_ms")
    timer.lap("total_ms")  # wall time across all stages (Timer spans process_image)

    # ---- collect rows ----------------------------------------------------------
    for row in rows:
        row["image"] = path.name
    ctx["cell_rows"].extend({k: row.get(k) for k in CELL_COLUMNS} for row in rows)

    candidates = [r for r in rows if r["rbc_candidate"]]
    interior = [r for r in candidates if not r["touches_border"]]
    border = [r for r in candidates if r["touches_border"]]
    n_cand = len(candidates)

    image_row = {
        "image": path.name,
        "width": width,
        "height": height,
        "channels": image_rgb.shape[2] if image_rgb.ndim == 3 else 1,
        "total_instances": result.n_instances,
        "rbc_candidate_count": n_cand,
        "interior_rbc_count": len(interior),
        "border_rbc_count": len(border),
        "border_fraction": (len(border) / n_cand) if n_cand else float("nan"),
        "median_area_px": ref["median_area_px"],
        "median_diameter_px": ref["median_diameter_px"],
        "iqr_area_px": ref["iqr_area_px"],
        "n_merged_suspect": sum(1 for r in rows if r["possible_merged_rbc"]),
        "n_small_artifact": sum(1 for r in rows if r["possible_small_artifact"]),
        "n_wbc": sum(1 for r in rows if r["possible_wbc"]),
        "cellprob_threshold": result.params["cellprob_threshold"],
        "flow_threshold": result.params["flow_threshold"],
        "min_size": result.params["min_size"],
        "diameter_setting": result.params.get("diameter"),
        "model_load_s": round(seg.model_load_s, 2),
        "manual_label": "",
    }
    image_row.update(spatial)
    image_row.update({k: round(v, 2) for k, v in timer.laps_ms.items()})
    return image_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="override the experiment name from the config")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="override the dataset directory from the config")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N images (smoke tests)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment_name = args.experiment_name or cfg["experiment"]["name"]
    dataset_dir = args.dataset or (REPO_ROOT / cfg["dataset"]["dir"])
    out_root = REPO_ROOT / cfg["outputs"]["dir"]
    exp_dir = out_root / experiment_name
    dirs = {
        "root": exp_dir,
        "overlays": exp_dir / "overlays",
        "masks": exp_dir / "masks",
        "debug_merged": exp_dir / "debug_merged",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    setup_logging(dirs["root"] / "run.log", verbose=args.verbose)

    images = discover_images(dataset_dir, tuple(cfg["dataset"]["extensions"]))
    if args.limit:
        images = images[: args.limit]

    # ---- reproducibility record -------------------------------------------------
    shutil.copyfile(args.config, dirs["root"] / "config_used.yaml")
    meta_extra = {
        "experiment": experiment_name,
        "model": cfg["model"]["name"],
        "cellpose_parameters": cfg["segmentation"],
        "use_bfloat16": cfg["model"]["use_bfloat16"],
        "image_count": len(images),
        "dataset_dir": str(dataset_dir),
    }
    meta = collect_env_metadata(REPO_ROOT, meta_extra)
    (dirs["root"] / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    print(f"Experiment : {experiment_name}")
    print(f"Device     : {meta['device']}")
    print(f"Model      : {cfg['model']['name']}")
    print(f"Images     : {len(images)}")

    # ---- model loaded ONCE before the loop ---------------------------------------
    try:
        segmenter = CellposeSegmenter(
            model_name=cfg["model"]["name"],
            gpu=cfg["model"]["gpu"],
            use_bfloat16=cfg["model"]["use_bfloat16"],
            cellprob_threshold=cfg["segmentation"]["cellprob_threshold"],
            flow_threshold=cfg["segmentation"]["flow_threshold"],
            min_size=cfg["segmentation"]["min_size"],
            diameter=cfg["segmentation"]["diameter"],
            normalize=cfg["segmentation"]["normalize"],
            augment=cfg["segmentation"]["augment"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialise Cellpose segmenter: %s", exc, exc_info=True)
        return 1

    ctx = {"config": cfg, "segmenter": segmenter, "dirs": dirs, "cell_rows": []}
    image_rows: List[Dict] = []
    failed: List[Dict] = []
    t_run = time.perf_counter()

    for i, path in enumerate(images, 1):
        try:
            row = process_image(ctx, path)
            image_rows.append(row)
            print(
                f"Image {i:2d}/{len(images)}: {path.name}\n"
                f"    Cellpose instances: {row['total_instances']}  "
                f"RBC candidates: {row['rbc_candidate_count']}  "
                f"(interior {row['interior_rbc_count']}, border {row['border_rbc_count']})  "
                f"merge-suspect {row['n_merged_suspect']}\n"
                f"    inference {row['inference_ms'] / 1000.0:.1f}s  "
                f"total {row['total_ms'] / 1000.0:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001 - keep processing remaining images
            logger.exception("Failed to process %s", path.name)
            failed.append({"image": path.name, "error": str(exc)})
            print(f"Image {i:2d}/{len(images)}: {path.name}  FAILED - {exc}")

    # ---- outputs -------------------------------------------------------------------
    pd.DataFrame(ctx["cell_rows"], columns=CELL_COLUMNS).to_csv(
        dirs["root"] / "cell_features.csv", index=False
    )
    image_frame = pd.DataFrame(image_rows, columns=IMAGE_COLUMNS)
    image_frame.to_csv(dirs["root"] / "image_features.csv", index=False)

    inference = image_frame["inference_ms"].dropna()
    totals = image_frame["total_ms"].dropna()
    summary = {
        "experiment": experiment_name,
        "device": meta["device"],
        "model": cfg["model"]["name"],
        "images_processed": len(image_rows),
        "successful": len(image_rows),
        "failed": failed,
        "counts": {
            "total_instances": int(image_frame["total_instances"].sum()),
            "rbc_candidates": int(image_frame["rbc_candidate_count"].sum()),
            "border_rbc": int(image_frame["border_rbc_count"].sum()),
            "merged_suspects": int(image_frame["n_merged_suspect"].sum()),
            "small_artifacts": int(image_frame["n_small_artifact"].sum()),
            "possible_wbc": int(image_frame["n_wbc"].sum()),
        },
        "performance": {
            "model_load_s": round(segmenter.model_load_s, 2),
            "inference_mean_ms": round(float(inference.mean()), 1) if len(inference) else None,
            "inference_median_ms": round(float(inference.median()), 1) if len(inference) else None,
            "inference_min_ms": round(float(inference.min()), 1) if len(inference) else None,
            "inference_max_ms": round(float(inference.max()), 1) if len(inference) else None,
            "total_mean_ms": round(float(totals.mean()), 1) if len(totals) else None,
            "approx_fps_inference": round(1000.0 / float(inference.median()), 3)
            if len(inference)
            else None,
            "approx_fps_end_to_end": round(1000.0 / float(totals.median()), 3)
            if len(totals)
            else None,
        },
        "wall_clock_s": round(time.perf_counter() - t_run, 1),
        "environment": meta,
    }
    (dirs["root"] / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    perf = summary["performance"]
    print("\n" + "=" * 64)
    print(f"Experiment : {experiment_name}")
    print(f"Device     : {meta['device']}")
    print(f"Images     : {len(images)} processed, {len(image_rows)} successful, {len(failed)} failed")
    if perf["inference_median_ms"] is not None:
        print(
            f"Inference  : mean {perf['inference_mean_ms'] / 1000.0:.1f}s | "
            f"median {perf['inference_median_ms'] / 1000.0:.1f}s | "
            f"~{perf['approx_fps_inference']} FPS (inference only)"
        )
        print(
            f"End-to-end : mean {perf['total_mean_ms'] / 1000.0:.1f}s/image | "
            f"~{perf['approx_fps_end_to_end']} FPS"
        )
    print(f"Results    : {dirs['root'].relative_to(REPO_ROOT)}")
    print("=" * 64)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
