"""Manual ground-truth count comparison.

Phase 2 tooling: the student manually counts RBCs per image (same convention
as the pipeline - RBC candidates, border cells included unless stated
otherwise) and records them in ``annotations/manual_counts.csv``. This module
joins the manual counts with the automated ``image_features.csv`` and reports
absolute / percentage count error plus dataset-level MAE and MPE.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

MANUAL_COUNT_COLUMN = "manual_rbc_count"
VALID_LABELS = {"TOO_THICK", "MONOLAYER", "TOO_THIN", "UNCERTAIN"}


def create_manual_template(image_names, path: Path) -> Path:
    """Write an empty manual-counts template listing every dataset image."""
    frame = pd.DataFrame(
        {
            "filename": list(image_names),
            MANUAL_COUNT_COLUMN: [None] * len(image_names),
            "manual_label": [None] * len(image_names),
            "notes": [None] * len(image_names),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Manual counts template written to %s", path)
    return path


def compare_counts(
    image_features_csv: Path,
    manual_counts_csv: Path,
    predicted_column: str = "rbc_candidate_count",
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Join predicted counts with manual counts and compute error metrics."""
    predicted = pd.read_csv(image_features_csv)
    manual = pd.read_csv(manual_counts_csv)

    manual[MANUAL_COUNT_COLUMN] = pd.to_numeric(
        manual[MANUAL_COUNT_COLUMN], errors="coerce"
    )
    merged = predicted.merge(
        manual[["filename", MANUAL_COUNT_COLUMN, "manual_label"]],
        on="filename",
        how="left",
    )
    done = merged.dropna(subset=[MANUAL_COUNT_COLUMN]).copy()
    if done.empty:
        logger.warning("No manual counts filled in yet - nothing to compare.")
        return merged

    done["absolute_error"] = (
        done[predicted_column] - done[MANUAL_COUNT_COLUMN]
    ).abs()
    done["percentage_error"] = (
        done["absolute_error"] / done[MANUAL_COUNT_COLUMN] * 100.0
    )

    mae = done["absolute_error"].mean()
    mpe = done["percentage_error"].mean()
    logger.info(
        "Count accuracy over %d images: MAE=%.2f RBCs, MPE=%.2f%%", len(done), mae, mpe
    )

    result = merged.join(done[["absolute_error", "percentage_error"]])
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        logger.info("Count comparison written to %s", output_csv)
    return result
