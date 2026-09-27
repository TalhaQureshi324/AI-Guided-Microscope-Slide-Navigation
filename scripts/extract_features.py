"""Recompute cell/spatial features from saved masks (no Cellpose re-inference).

 lets feature logic and flagging parameters be iterated quickly on an
existing experiment without paying the inference cost again:

    python scripts/extract_features.py --experiment cpsam_v2_baseline

Reads ``outputs/<experiment>/masks/*.png`` and rewrites ``cell_features.csv``
and ``image_features.csv`` in the same directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.segment_dataset import CELL_COLUMNS, IMAGE_COLUMNS  # noqa: E402
from src.analysis.cell_features import (  # noqa: E402
    compute_reference_stats,
    flag_cells,
    measure_instances,
)
from src.analysis.spatial_features import compute_spatial_features  # noqa: E402
from src.postprocessing.merged_cell_splitter import analyze_merge_suspicion  # noqa: E402
from src.utils.runtime import setup_logging  # noqa: E402

import logging  # noqa: E402

logger = logging.getLogger("extract_features")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=str, default="cpsam_v2_baseline")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    args = parser.parse_args()

    exp_dir = REPO_ROOT / "outputs" / args.experiment
    masks_dir = exp_dir / "masks"
    if not masks_dir.is_dir():
        raise SystemExit(f"No masks directory found at {masks_dir} - run segment_dataset.py first.")

    setup_logging(exp_dir / "extract_features.log")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    cell_rows, image_rows = [], []
    for mask_path in sorted(masks_dir.glob("*.png")):
        labels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
        rows = measure_instances(labels)
        ref = compute_reference_stats(rows, cfg["filtering"]["stats_scope"])
        rows = flag_cells(
            rows, ref,
            small_area_factor=cfg["filtering"]["small_area_factor"],
            wbc_area_factor=cfg["filtering"]["wbc_area_factor"],
        )
        rows, _ = analyze_merge_suspicion(
            labels, rows, ref,
            area_factor_min=cfg["merge_detection"]["area_factor_min"],
            solidity_max=cfg["merge_detection"]["solidity_max"],
            elongation_min=cfg["merge_detection"]["elongation_min"],
            min_signals=cfg["merge_detection"]["min_signals"],
        )
        spatial = compute_spatial_features(
            rows, labels.shape[:2], ref,
            contact_margin_fraction=cfg["contact"]["margin_fraction"],
            graph_proximity_norm=cfg["contact"]["proximity_norm"],
            uniformity_grids=tuple(cfg["uniformity"]["grids"]),
        )

        for row in rows:
            row["image"] = mask_path.stem
        cell_rows.extend({k: row.get(k) for k in CELL_COLUMNS} for row in rows)

        candidates = [r for r in rows if r["rbc_candidate"]]
        image_rows.append({
            "image": mask_path.stem,
            "width": labels.shape[1],
            "height": labels.shape[0],
            "channels": 3,
            "total_instances": len(rows),
            "rbc_candidate_count": len(candidates),
            "interior_rbc_count": sum(1 for r in candidates if not r["touches_border"]),
            "border_rbc_count": sum(1 for r in candidates if r["touches_border"]),
            "median_area_px": ref["median_area_px"],
            "median_diameter_px": ref["median_diameter_px"],
            "iqr_area_px": ref["iqr_area_px"],
            "n_merged_suspect": sum(1 for r in rows if r["possible_merged_rbc"]),
            "n_small_artifact": sum(1 for r in rows if r["possible_small_artifact"]),
            "n_wbc": sum(1 for r in rows if r["possible_wbc"]),
            "manual_label": "",
            **spatial,
        })
        logger.info("%s: %d instances, %d candidates", mask_path.name, len(rows), len(candidates))

    pd.DataFrame(cell_rows, columns=CELL_COLUMNS).to_csv(exp_dir / "cell_features.csv", index=False)

    # preserve timing columns from the original run if present
    old = exp_dir / "image_features.csv"
    timing_cols = [
        "cellprob_threshold", "flow_threshold", "min_size", "diameter_setting",
        "model_load_s", "load_ms", "inference_ms", "postprocess_ms",
        "features_ms", "overlay_ms", "total_ms",
    ]
    if old.exists():
        prev = pd.read_csv(old).set_index("image")
        for row in image_rows:
            if row["image"] in prev.index:
                for col in timing_cols:
                    if col in prev.columns:
                        row[col] = prev.loc[row["image"], col]
    pd.DataFrame(image_rows, columns=IMAGE_COLUMNS).to_csv(exp_dir / "image_features.csv", index=False)

    logger.info("Features rewritten for %d images in %s", len(image_rows), exp_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
