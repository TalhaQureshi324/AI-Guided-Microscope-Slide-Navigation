"""Compare automated RBC counts against manual ground-truth counts (Phase 2).

Prerequisites:
  1. Run ``scripts/segment_dataset.py`` (produces ``outputs/<exp>/image_features.csv``).
  2. Fill in the ``manual_rbc_count`` column of ``annotations/manual_counts.csv``
     (same counting convention as the pipeline: RBC candidates including border
     cells). Optionally set ``manual_label`` to TOO_THICK / MONOLAYER /
     TOO_THIN / UNCERTAIN for the later monolayer calibration.

Then:

    python scripts/evaluate_counts.py --experiment cpsam_v2_baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import compare_counts  # noqa: E402
from src.utils.runtime import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=str, default="cpsam_v2_baseline")
    parser.add_argument(
        "--manual", type=Path, default=REPO_ROOT / "annotations" / "manual_counts.csv"
    )
    parser.add_argument("--predicted-column", type=str, default="rbc_candidate_count")
    args = parser.parse_args()

    setup_logging()
    image_csv = REPO_ROOT / "outputs" / args.experiment / "image_features.csv"
    result = compare_counts(
        image_csv, args.manual,
        predicted_column=args.predicted_column,
        output_csv=image_csv.parent / "count_comparison.csv",
    )

    cols = [
        "image", "rbc_candidate_count", "manual_rbc_count",
        "absolute_error", "percentage_error", "manual_label",
    ]
    cols = [c for c in cols if c in result.columns]
    done = result.dropna(subset=["manual_rbc_count"]) if "manual_rbc_count" in result.columns else result
    if len(done):
        print(done[cols].to_string(index=False))
        print(f"\nMAE : {done['absolute_error'].mean():.2f} RBCs")
        print(f"MPE : {done['percentage_error'].mean():.2f} %")
    else:
        print("No manual counts available yet - fill in annotations/manual_counts.csv first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
