"""Live session outputs (spec §35-§37).

One directory per session: outputs/live_session_<timestamp>/
    session_metadata.json    source/env/model/thresholds reproducibility record
    config_used.yaml         exact live configuration
    accepted_fields.csv      one row per finished detailed analysis
    raw_frames_light.csv     lightweight every-Nth-frame log
    labels.csv + labels/     human calibration labels (hotkeys M/T/N/U)
    keyframes/ overlays/ masks/   per-field artifacts (configurable)
    recorded.mp4             optional session recording
    runtime_summary.json     written on shutdown (latencies, busy %, counts)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import cv2

from src.utils.runtime import collect_env_metadata, setup_logging

ACCEPTED_COLUMNS = [
    "job_id", "priority", "manual_capture", "worker_id", "human_label",
    "frame_idx", "t_capture_sec", "request_reason", "stale", "stale_reason",
    "cum_x_px", "cum_y_px", "cum_total_px",
    "screen_result", "sharpness", "brightness",
    "queue_wait_ms", "cellpose_ms", "feature_ms", "merge_analysis_ms",
    "latency_ms", "feature_mode", "merge_analysis_ran", "first_inference",
    "cellpose_instances", "rbc_candidates", "valid_region_rbc", "border_cells",
    "merge_suspects", "rbc_density_per_mpx", "coverage",
    "median_rbc_diameter_px", "normalized_median_nn", "contact_ratio",
    "pct_0_neighbors", "pct_1_neighbor", "pct_2_neighbors", "pct_3plus_neighbors",
    "mean_graph_degree", "largest_cluster_fraction", "density_cv",
    "raw_monolayer_score", "raw_class", "smoothed_score", "smoothed_class",
]

RAW_COLUMNS = ["frame_idx", "t_sec", "disp_px", "sharpness", "brightness",
               "motion_state", "screen_result"]

LABEL_NAMES = {"m": "MONOLAYER", "t": "TOO_THICK", "n": "TOO_THIN", "u": "UNCERTAIN"}


class LiveSession:
    def __init__(self, root: Path, cfg: Dict, source_name: str, settings: Dict) -> None:
        self.dir = root
        self.cfg = cfg
        (self.dir / "keyframes").mkdir(parents=True, exist_ok=True)
        (self.dir / "overlays").mkdir(parents=True, exist_ok=True)
        if cfg.get("save_masks", False):
            (self.dir / "masks").mkdir(parents=True, exist_ok=True)
        (self.dir / "labels").mkdir(parents=True, exist_ok=True)

        setup_logging(self.dir / "run.log", verbose=False)
        (self.dir / "config_used.yaml").write_text(
            json.dumps(cfg, indent=2, default=str), encoding="utf-8"
        )
        meta = collect_env_metadata(self.dir, extra={
            "session": self.dir.name,
            "source": source_name,
            "source_settings": settings,
            "cellpose_model": cfg.get("model_name"),
            "cellpose_width": cfg.get("cellpose_width"),
            "cellpose_mode": cfg.get("cellpose_mode"),
            "feature_mode_default": cfg.get("feature_mode", "live"),
            "prototype_count_gate": [cfg.get("classification_cfg", {}).get("prototype_count_min"),
                                     cfg.get("classification_cfg", {}).get("prototype_count_max")],
            "note": "all thresholds experimental/prototype; see configs/live.yaml",
        })
        (self.dir / "session_metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        self.log: logging.Logger = logging.getLogger("live.session")

        self._acc_file = open(self.dir / "accepted_fields.csv", "w", newline="", encoding="utf-8")
        self._acc = csv.DictWriter(self._acc_file, fieldnames=ACCEPTED_COLUMNS)
        self._acc.writeheader()
        self._raw_file = open(self.dir / "raw_frames_light.csv", "w", newline="", encoding="utf-8")
        self._raw = csv.DictWriter(self._raw_file, fieldnames=RAW_COLUMNS)
        self._raw.writeheader()
        self._lab_file = open(self.dir / "labels.csv", "w", newline="", encoding="utf-8")
        self._lab = csv.DictWriter(self._lab_file, fieldnames=[
            "label", "job_id", "frame_idx", "t_capture", "predicted_class", "score",
            "rbc_candidates", "coverage", "file"])
        self._lab.writeheader()
        self._label_count = 0
        self.recording_path = lambda: str(self.dir / "recorded.mp4")

    # ------------------------------------------------------------------
    def log_accepted(self, j) -> None:
        """Log one completed analysis job (src/live/jobs.AnalysisJob)."""
        f = j.features or {}
        merge_ms = round(j.features_ms - (j.cellpose_ms or 0), 1) if (
            j.features_ms is not None and j.merge_analysis_ran) else 0.0
        self._acc.writerow({
            "job_id": j.job_id,
            "priority": j.priority,
            "manual_capture": j.priority in ("MANUAL", "FORCED"),
            "worker_id": j.worker_id,
            "human_label": j.human_label or "",
            "frame_idx": j.frame_idx,
            "t_capture_sec": round(j.t_capture, 3),
            "request_reason": j.reason,
            "stale": j.stale,
            "stale_reason": j.stale_reason,
            "cum_x_px": round(j.cumulative_x, 1),
            "cum_y_px": round(j.cumulative_y, 1),
            "cum_total_px": round(j.cumulative_total, 1),
            "screen_result": j.screen_result,
            "sharpness": round(j.sharpness, 2),
            "brightness": round(j.brightness, 2),
            "queue_wait_ms": round(j.queue_wait_ms, 1) if j.queue_wait_ms else "",
            "cellpose_ms": round(j.cellpose_ms, 1) if j.cellpose_ms else "",
            "feature_ms": round(j.features_ms, 1) if j.features_ms else "",
            "merge_analysis_ms": merge_ms,
            "latency_ms": round(j.latency_ms, 1),
            "feature_mode": j.feature_mode,
            "merge_analysis_ran": j.merge_analysis_ran,
            "first_inference": j.first_inference,
            "cellpose_instances": f.get("cellpose_instance_count"),
            "rbc_candidates": f.get("rbc_candidate_count"),
            "valid_region_rbc": f.get("valid_region_rbc_count"),
            "border_cells": sum(1 for x in (j.rows or [])
                                if x.get("rbc_candidate") and x.get("touches_border")),
            "merge_suspects": f.get("merge_suspect_count"),
            "rbc_density_per_mpx": (round(f["rbc_density"], 2)
                                    if f.get("rbc_density") == f.get("rbc_density") else ""),
            "coverage": round(f["coverage"], 4) if f.get("coverage") == f.get("coverage") else "",
            "median_rbc_diameter_px": round(f["median_rbc_diameter"], 2),
            "normalized_median_nn": round(f["normalized_median_nn"], 4),
            "contact_ratio": round(f["contact_ratio"], 4),
            "pct_0_neighbors": round(f["pct_0_neighbors"], 2),
            "pct_1_neighbor": round(f["pct_1_neighbor"], 2),
            "pct_2_neighbors": round(f["pct_2_neighbors"], 2),
            "pct_3plus_neighbors": round(f["pct_3plus_neighbors"], 2),
            "mean_graph_degree": round(f["mean_graph_degree"], 3),
            "largest_cluster_fraction": round(f["largest_cluster_fraction"], 4),
            "density_cv": round(f["density_cv"], 4),
            "raw_monolayer_score": round(j.score, 4) if j.score is not None else "",
            "raw_class": j.raw_class or "",
            "smoothed_score": (round(j.smoothed_score, 4)
                               if j.smoothed_score is not None else ""),
            "smoothed_class": j.smoothed_class or "",
        })
        self._acc_file.flush()
        if self.cfg.get("save_masks", False) and j.labels is not None:
            cv2.imwrite(str(self.dir / "masks" / f"job_{j.job_id:04d}.png"),
                        j.labels.astype(np.int32))

    def save_snapshot(self, frame, frame_idx: int) -> None:
        """SAVE SNAPSHOT button: store the raw frame without any analysis."""
        name = f"snapshot_f{frame_idx:06d}"
        cv2.imwrite(str(self.dir / "keyframes" / f"{name}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.log.info("snapshot saved: %s", name)

    def save_field_artifacts(self, j, frame=None) -> None:
        name = f"job_{j.job_id:04d}_f{j.frame_idx:06d}"
        frame = j.frame if frame is None else frame
        if frame is None:
            return
        cv2.imwrite(str(self.dir / "keyframes" / f"{name}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if j.overlay_layer is not None:
            comp = cv2.bitwise_or(frame, j.overlay_layer)
            if j.id_layer is not None:
                comp = cv2.bitwise_or(comp, j.id_layer)
            cv2.imwrite(str(self.dir / "overlays" / f"{name}.jpg"), comp,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])

    def log_raw(self, frame_idx, t, disp, sharpness, brightness,
                motion_state, screen_result) -> None:
        self._raw.writerow({
            "frame_idx": frame_idx, "t_sec": round(t, 3),
            "disp_px": round(disp, 2), "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2), "motion_state": motion_state,
            "screen_result": screen_result,
        })
        if frame_idx % 300 == 0:
            self._raw_file.flush()

    def save_label(self, label_key: str, frame, job=None) -> None:
        """Store a human label with the prediction kept SEPARATE (spec §30)."""
        label = LABEL_NAMES.get(label_key.lower())
        if label is None:
            return
        self._label_count += 1
        ident = f"job{job.job_id}" if job is not None else f"f{getattr(self, '_last_live_frame', 'x')}"
        name = f"label_{self._label_count:03d}_{label}_{ident}"
        cv2.imwrite(str(self.dir / "labels" / f"{name}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        self._lab.writerow({
            "label": label,
            "job_id": job.job_id if job is not None else "",
            "frame_idx": job.frame_idx if job is not None else "",
            "t_capture": round(job.t_capture, 3) if job is not None else "",
            "predicted_class": job.raw_class if job is not None else "",
            "score": round(job.score, 4) if (job is not None and job.score is not None) else "",
            "rbc_candidates": job.features.get("rbc_candidate_count") if job is not None else "",
            "coverage": job.features.get("coverage") if job is not None else "",
            "file": f"labels/{name}.jpg",
        })
        self._lab_file.flush()
        self.log.info("human label saved: %s (job %s, predicted %s)",
                      label, job.job_id if job is not None else "-",
                      job.raw_class if job is not None else "-")

    def write_summary(self, state, elapsed_sec: float, jobmgr=None) -> Dict:
        res = state.result
        summary = {
            "session": self.dir.name,
            "elapsed_sec": round(elapsed_sec, 1),
            "source": getattr(state, "session_settings", {}),
            "frames_captured": state.frame_seq + 1,
            "capture_fps_avg": round(state.capture_fps, 2),
            "jobs_total": jobmgr.stats()["total_jobs"] if jobmgr else None,
            "jobs_superseded_auto": jobmgr.superseded_auto if jobmgr else None,
            "jobs_refused_manual": jobmgr.refused_manual if jobmgr else None,
            "fields_analyzed": state.analysis_count,
            "cellpose_busy_sec": round(state.cellpose_busy_sec, 1),
            "cellpose_busy_fraction": round(state.cellpose_busy_sec / max(elapsed_sec, 1), 3),
            "last_latency_ms": round(res.latency_ms, 1) if res else None,
            "last_cellpose_ms": round(res.cellpose_ms, 1) if res else None,
            "last_features_ms": round(res.features_ms, 1) if res else None,
            "monolayer_history": [
                h for h in state.history if h["raw_class"] == "MONOLAYER"
            ],
            "stale_results": sum(1 for h in state.history if h["stale"]),
        }
        (self.dir / "runtime_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        self._acc_file.close()
        self._raw_file.close()
        self._lab_file.close()
        return summary
