"""Detection of suspicious merged RBC masks (and a disabled watershed fallback).

Philosophy (project prompt, sections 13-16):
  * Cellpose instance segmentation is trusted first; we do NOT watershed
    everything - that would over-segment correctly detected RBCs.
  * A mask is flagged as a possible merge only when MULTIPLE signals agree
    (area ratio, solidity, elongation, distance-transform peaks). Area alone is
    never sufficient.
  * The marker-controlled watershed split exists here as an experimental,
    opt-in tool for later evaluation against ground truth. It is NOT part of
    the Phase 1 pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

logger = logging.getLogger(__name__)


def distance_transform_peaks(
    mask: np.ndarray, median_radius_px: float
) -> np.ndarray:
    """Robust local maxima of the smoothed Euclidean distance transform.

    A single healthy RBC mask yields one peak near its centre; two merged RBCs
    typically yield two well-separated peaks. The smoothing sigma and the
    ``min_distance`` between peaks both scale with the median RBC radius so the
    behaviour does not depend on image resolution.
    """
    if mask.sum() == 0:
        return np.empty((0, 2), dtype=int)

    dt = distance_transform_edt(mask)
    sigma = max(1.0, 0.25 * median_radius_px)
    dt_smooth = gaussian_filter(dt, sigma=sigma)

    min_distance = max(2, int(round(0.6 * median_radius_px)))
    threshold_abs = 0.45 * median_radius_px  # ignore shallow boundary noise
    coords = peak_local_max(
        dt_smooth,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        labels=mask,
        exclude_border=False,
    )
    return coords


def analyze_merge_suspicion(
    labels: np.ndarray,
    rows: List[Dict],
    ref: Dict,
    area_factor_min: float = 1.6,
    solidity_max: float = 0.90,
    elongation_min: float = 1.5,
    min_signals: int = 2,
) -> Tuple[List[Dict], Dict[int, np.ndarray]]:
    """Flag instances that may contain multiple merged RBCs.

    Returns the updated rows plus a map ``{instance_id: peak_coordinates}`` for
    the visual debugger. An instance is flagged only when its area is at least
    ``area_factor_min`` times the median AND at least ``min_signals``
    independent shape signals agree (low solidity, high elongation, >= 2
    distance-transform peaks).
    """
    median_area = ref.get("median_area_px", float("nan"))
    median_radius = ref.get("median_diameter_px", float("nan"))
    if not np.isfinite(median_area) or median_area <= 0:
        logger.warning("Merge analysis skipped: no valid size reference.")
        return rows, {}

    peaks_by_id: Dict[int, np.ndarray] = {}
    for row in rows:
        ratio = row["area_ratio_vs_median"]
        if not np.isfinite(ratio) or ratio < area_factor_min:
            row["possible_merged_rbc"] = False
            continue

        # Evaluate shape signals on the local mask.
        instance = labels == row["instance_id"]
        coords = distance_transform_peaks(instance, median_radius / 2.0)
        row["n_dt_peaks"] = int(len(coords))

        signals = 0
        if row["solidity"] < solidity_max:
            signals += 1
        if row["elongation"] > elongation_min:
            signals += 1
        if len(coords) >= 2:
            signals += 1

        row["possible_merged_rbc"] = bool(signals >= min_signals)
        if len(coords) >= 2:
            peaks_by_id[row["instance_id"]] = coords

    return rows, peaks_by_id


def experimental_watershed_split(
    mask: np.ndarray, median_radius_px: float
) -> Optional[np.ndarray]:
    """Marker-controlled watershed INSIDE one suspicious mask (experimental).

    Markers come from distance-transform peaks. Returns a label array of the
    split regions, or ``None`` when the split is rejected because the resulting
    sub-regions are not plausible RBCs (e.g. one RBC shattered into five tiny
    fragments). Acceptance criteria are deliberately strict and documented for
    the later calibration phase - this function is NOT called by the default
    pipeline (``postprocessing.enable_watershed_fallback: false``).
    """
    from skimage.morphology import label as cc_label

    peaks = distance_transform_peaks(mask, median_radius_px)
    if len(peaks) < 2:
        return None

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = i

    dt = distance_transform_edt(mask)
    split = watershed(-dt, markers, mask=mask)

    # Plausibility gate: every sub-region must be a reasonable fraction of a
    # single RBC area implied by the median radius.
    single_area = np.pi * median_radius_px ** 2
    sub_areas = np.bincount(split.ravel())[1:]
    sub_areas = sub_areas[sub_areas > 0]
    if len(sub_areas) < 2:
        return None
    if np.any(sub_areas < 0.35 * single_area) or np.any(sub_areas > 2.5 * single_area):
        logger.debug("Watershed split rejected: implausible sub-region areas %s", sub_areas)
        return None

    n_expected = np.clip(round(mask.sum() / single_area), 1, None)
    if abs(len(sub_areas) - n_expected) > 1:
        logger.debug(
            "Watershed split rejected: %d regions vs ~%d expected cells.",
            len(sub_areas), n_expected,
        )
        return None

    return split.astype(np.int32)
