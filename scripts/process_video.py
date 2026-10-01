"""Phase V2-V9 driver: motion-aware video analysis pipeline.

Flow (spec section 5):
  decode -> quality check -> motion estimation -> keyframe decision
  -> [accepted] fast screen -> Cellpose -> RBC measurements (reused static
  modules) -> monolayer features -> prototype score -> temporal smoothing
  -> annotated frame -> CSV/JSON outputs.

The static baseline (outputs/cpsam_v2_baseline) and external/cellpose are
never touched. Outputs land in outputs/<experiment_dir>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
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
from src.segmentation.cellpose_segmenter import CellposeSegmenter  # noqa: E402
from src.utils.runtime import Timer, collect_env_metadata, setup_logging  # noqa: E402
from src.utils.image_io import save_mask_png  # noqa: E402
from src.video.fast_screen import FastScreen  # noqa: E402
from src.video.keyframes import KeyframeSelector  # noqa: E402
from src.video.monolayer import (  # noqa: E402
    classify_field,
    compute_field_features,
    prototype_monolayer_score,
)
from src.video.quality import frame_quality  # noqa: E402
from src.video.render import draw_field_hud, draw_skip_hud  # noqa: E402
from src.video.results_io import (  # noqa: E402
    write_frame_features_csv,
    write_results_csv,
    write_summary_csv,
    write_summary_json,
)
from src.video.temporal import TemporalSmoother  # noqa: E402
from src.visualization.overlays import render_overlay  # noqa: E402

logger = logging.getLogger("process_video")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "video.yaml")
    p.add_argument("--video", type=Path, default=None, help="override config video path")
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--max-keyframes", type=int, default=None,
                   help="stop after N accepted keyframes (smoke tests)")
    p.add_argument("--no-annotated-video", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_path = Path(args.video or (REPO_ROOT / cfg["video"]["path"]))
    exp_name = args.experiment_name or cfg["outputs"]["experiment_dir"]
    out_dir = REPO_ROOT / cfg["outputs"]["dir"] / exp_name
    (out_dir / "keyframes").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "debug").mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "run.log", verbose=args.verbose)
    shutil.copy2(args.config, out_dir / "config_used.yaml")

    logger.info("=" * 70)
    logger.info("VIDEO PROTOTYPE RUN  experiment=%s", exp_name)
    logger.info("video=%s", video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video %s", video_path)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("meta: %dx%d @ %.2f fps, %d frames", width, height, fps, n_meta)

    writer = None
    if cfg["outputs"].get("write_annotated_video", True) and not args.no_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "annotated_video.mp4"), fourcc, fps, (width, height))

    # ---- model load + warm-up -------------------------------------------
    seg_cfg = cfg["segmentation"]
    t0 = time.perf_counter()
    segmenter = CellposeSegmenter(
        model_name=cfg["model"]["name"],
        gpu=cfg["model"].get("gpu", "auto"),
        use_bfloat16=cfg["model"].get("use_bfloat16", "auto"),
        cellprob_threshold=seg_cfg["cellprob_threshold"],
        flow_threshold=seg_cfg["flow_threshold"],
        min_size=seg_cfg["min_size"],
        diameter=seg_cfg.get("diameter"),
        normalize=seg_cfg.get("normalize", True),
        augment=seg_cfg.get("augment", False),
    )
    model_load_s = time.perf_counter() - t0
    logger.info("model '%s' loaded in %.1f s on %s (bfloat16=%s)",
                cfg["model"]["name"], model_load_s, segmenter.device_desc, segmenter.use_bfloat16)

    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        cuda_version = torch.version.cuda if torch.cuda.is_available() else None
        precision = "bfloat16" if segmenter.use_bfloat16 else "float32"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        gpu_name = cuda_version = precision = None

    # ---- stages ----------------------------------------------------------
    kf_cfg = cfg["keyframes"]
    selector = KeyframeSelector(
        motion_threshold_px=kf_cfg["motion_threshold_px"],
        min_interval_sec=kf_cfg["min_interval_sec"],
        heartbeat_sec=kf_cfg["heartbeat_sec"],
        blur_threshold=kf_cfg["blur_threshold"],
        fps=fps,
        motion_work_width=cfg["video"].get("motion_work_width", 480),
    )
    screen = FastScreen(
        work_width=cfg["fast_screen"].get("work_width", 480),
        **{k: v for k, v in cfg["fast_screen"].items()
           if k.startswith(("thick_", "thin_"))},
    ) if cfg["fast_screen"].get("enabled", True) else None
    smoother = TemporalSmoother(
        alpha=cfg["temporal"]["alpha"],
        window=cfg["temporal"]["window"],
        monolayer_score_min=cfg["classification"]["monolayer_score_min"],
    )

    stage_ms = {"decode": 0.0, "mq": 0.0, "mq_n": 0, "render": 0.0}
    cellpose_ms_list: list[float] = []
    feature_ms_list: list[float] = []
    total_kf_ms_list: list[float] = []
    first_inference_done = False
    warmup_done = False
    warmup_ms = None

    all_rows: list[dict] = []
    kf_rows: list[dict] = []
    kf_count = 0
    last_class = None
    last_smoothed_score = None
    pipeline_t0 = time.perf_counter()

    frame_idx = -1
    while True:
        t_all = time.perf_counter()
        t0 = time.perf_counter()
        ok = cap.grab()
        if not ok:
            break
        ok, frame_bgr = cap.retrieve()
        if not ok:
            break
        frame_idx += 1
        stage_ms["decode"] += (time.perf_counter() - t0) * 1000.0

        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # one-time GPU warm-up on a downscaled copy of the first decoded frame
        # (spec section 40: warm-up / first / steady-state reported separately)
        if not warmup_done and gpu_name:
            t0 = time.perf_counter()
            warm_frame = cv2.resize(frame_bgr, (960, 540), interpolation=cv2.INTER_AREA)
            segmenter.segment(cv2.cvtColor(warm_frame, cv2.COLOR_BGR2RGB))
            warmup_ms = (time.perf_counter() - t0) * 1000.0
            warmup_done = True
            try:
                import torch

                torch.cuda.reset_peak_memory_stats()
            except Exception:  # noqa: BLE001
                pass
            logger.info("GPU warm-up inference (960x540): %.0f ms", warmup_ms)

        # quality + motion + decision
        t0 = time.perf_counter()
        quality = frame_quality(frame_bgr)
        small = selector._motion.prep(frame_bgr)
        decision = selector.update(frame_idx, t_sec, quality, small, width)
        stage_ms["mq"] += (time.perf_counter() - t0) * 1000.0
        stage_ms["mq_n"] += 1

        row = {
            "video_name": video_path.name,
            "frame_index": frame_idx,
            "timestamp_sec": round(t_sec, 3),
            "accepted_keyframe": decision.accepted,
            "skip_reason": decision.status,  # full frame status for every row
            "dx": round(decision.dx, 2),
            "dy": round(decision.dy, 2),
            "cumulative_displacement": round(decision.cumulative_px, 2),
            "sharpness": round(quality["sharpness_lapvar"], 3),
            "brightness": round(quality["brightness"], 2),
            "contrast": round(quality["contrast"], 2),
            "saturation": round(quality["saturation"], 2),
            "overexposed_pct": round(quality["overexposed_pct"], 3),
            "underexposed_pct": round(quality["underexposed_pct"], 3),
            "motion_response": round(decision.response, 4),
        }
        all_rows.append(row)

        if not decision.accepted:
            t0 = time.perf_counter()
            if writer is not None:
                hud = draw_skip_hud(frame_bgr, frame_idx, decision.status,
                                    last_class, last_smoothed_score)
                writer.write(hud)
            stage_ms["render"] += (time.perf_counter() - t0) * 1000.0
            continue

        # ---------------- accepted keyframe -------------------------------
        kf_count += 1
        row["keyframe_index"] = kf_count
        row["fast_screen_result"] = ""
        row["cellpose_used"] = False
        row["is_first_inference"] = not first_inference_done

        if cfg["outputs"].get("save_keyframes", True):
            cv2.imwrite(str(out_dir / "keyframes" / f"keyframe_{kf_count:04d}_f{frame_idx:06d}.jpg"),
                        frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # fast screen (recorded, may or may not veto Cellpose)
        if screen is not None:
            screen_res = screen.evaluate(frame_bgr)
            row.update({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in screen_res.items()})
        run_cellpose = True
        if screen is not None and cfg["fast_screen"].get("decides_cellpose", False):
            from src.video.fast_screen import CLEARLY_THICK, CLEARLY_THIN

            run_cellpose = screen_res["fast_screen_result"] not in (CLEARLY_THICK, CLEARLY_THIN)

        labels = None
        if run_cellpose:
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            seg = segmenter.segment(image_rgb)
            labels = seg.labels
            row["cellpose_used"] = True
            row["cellpose_time_ms"] = round(seg.timings_ms["inference_ms"], 1)
            row["cellpose_model_time_ms"] = round(seg.timings_ms["inference_ms"], 1)
            cellpose_ms_list.append(seg.timings_ms["inference_ms"])
            if not first_inference_done:
                first_inference_done = True
                logger.info("first (post-warmup-implicit) inference: %.0f ms", seg.timings_ms["inference_ms"])

        total_before_features = time.perf_counter()
        if labels is not None:
            t0 = time.perf_counter()
            rows_cells = measure_instances(labels)
            ref = compute_reference_stats(rows_cells, stats_scope="interior")
            flag_cells(rows_cells, ref,
                       small_area_factor=0.35, wbc_area_factor=3.5)
            analyze_merge_suspicion(
                labels, rows_cells, ref,
                area_factor_min=1.6, solidity_max=0.90,
                elongation_min=1.5, min_signals=2,
            )
            f_cfg = cfg["field"]
            features = compute_field_features(
                rows_cells, labels, ref,
                valid_roi_margin=f_cfg["valid_roi_margin"],
                contact_margin_fraction=f_cfg["contact_margin_fraction"],
                neighbor_distance_threshold=f_cfg["neighbor_distance_threshold"],
                large_cluster_min_cells=f_cfg["large_cluster_min_cells"],
                uniformity_grid=f_cfg["uniformity_grid"],
            )
            score, comps = prototype_monolayer_score(features, cfg["monolayer_score"])
            raw_class, evidence = classify_field(features, score, cfg["classification"])
            smoothed_score, smoothed_class = smoother.update(score, raw_class)
            feature_ms = (time.perf_counter() - t0) * 1000.0
            feature_ms_list.append(feature_ms)

            row.update({
                "cellpose_instance_count": features["cellpose_instance_count"],
                "rbc_candidate_count": features["rbc_candidate_count"],
                "valid_region_rbc_count": features["valid_region_rbc_count"],
                "rbc_density": round(features["rbc_density"], 2),
                "coverage": round(features["coverage"], 4),
                "median_rbc_diameter": round(features["median_rbc_diameter"], 2),
                "median_nn": round(features["median_nn"], 2),
                "normalized_median_nn": round(features["normalized_median_nn"], 4),
                "nn_p10_norm": round(features["nn_p10_norm"], 4),
                "nn_p25_norm": round(features["nn_p25_norm"], 4),
                "nn_p75_norm": round(features["nn_p75_norm"], 4),
                "median_edge_gap": round(features["median_edge_gap"], 2),
                "normalized_edge_gap": round(features["normalized_edge_gap"], 4),
                "contact_ratio": round(features["contact_ratio"], 4),
                "pct_0_neighbors": round(features["pct_0_neighbors"], 2),
                "pct_1_neighbor": round(features["pct_1_neighbor"], 2),
                "pct_2_neighbors": round(features["pct_2_neighbors"], 2),
                "pct_3plus_neighbors": round(features["pct_3plus_neighbors"], 2),
                "mean_graph_degree": round(features["mean_graph_degree"], 3),
                "median_graph_degree": features["median_graph_degree"],
                "n_clusters": features["n_clusters"],
                "isolated_pct": round(features["isolated_pct"], 2),
                "median_cluster_size": features["median_cluster_size"],
                "largest_cluster_size": features["largest_cluster_size"],
                "largest_cluster_fraction": round(features["largest_cluster_fraction"], 4),
                "pct_in_large_clusters": round(features["pct_in_large_clusters"], 2),
                "density_mean_per_mpx": round(features["density_mean_per_mpx"], 1),
                "density_cv": round(features["density_cv"], 4),
                "density_max_min_ratio": round(features["density_max_min_ratio"], 2),
                "merge_suspect_count": features["merge_suspect_count"],
                "raw_monolayer_score": round(score, 4),
                "smoothed_monolayer_score": round(smoothed_score, 4),
                "raw_class": raw_class,
                "smoothed_class": smoothed_class,
                "thick_signal_count": evidence["thick_signal_count"],
                "thin_signal_count": evidence["thin_signal_count"],
                "feature_time_ms": round(feature_ms, 1),
            })
            row["score_components"] = comps  # not exported to CSV; debug only

            if cfg["outputs"].get("save_masks", True):
                save_mask_png(out_dir / "masks" / f"keyframe_{kf_count:04d}_f{frame_idx:06d}.png",
                              labels.astype(np.int32))
            if cfg["outputs"].get("save_debug_overlays", True):
                overlay = render_overlay(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), labels, rows_cells,
                    title=f"KF{kf_count} f{frame_idx} {raw_class} score={score:.2f}",
                    inference_ms=seg.timings_ms["inference_ms"] if run_cellpose else 0.0,
                    show_instance_ids=True,
                )
                cv2.imwrite(str(out_dir / "debug" / f"keyframe_{kf_count:04d}_f{frame_idx:06d}.jpg"),
                            overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])

            last_class = smoothed_class
            last_smoothed_score = smoothed_score

            if kf_count % 5 == 0 or kf_count == 1:
                logger.info(
                    "KF %4d | f%5d t=%6.1fs | %-9s raw %.2f smooth %.2f | RBC %4d cov %.2f nn %.2f"
                    " | cellpose %5.0fms feat %4.0fms",
                    kf_count, frame_idx, t_sec, raw_class, score, smoothed_score,
                    features["rbc_candidate_count"], features["coverage"],
                    features["normalized_median_nn"],
                    row.get("cellpose_time_ms", 0.0), feature_ms,
                )
        else:
            # fast-only field (screen vetoed Cellpose)
            row["raw_class"] = f"FAST_ONLY_{row.get('fast_screen_result', '')}"
            feature_ms = (time.perf_counter() - total_before_features) * 1000.0
            row["feature_time_ms"] = round(feature_ms, 1)

        row["total_time_ms"] = round((time.perf_counter() - t_all) * 1000.0, 1)
        total_kf_ms_list.append(row["total_time_ms"])
        kf_rows.append(row)

        t0 = time.perf_counter()
        if writer is not None:
            hud = draw_field_hud(frame_bgr, row, smoothed_score, smoothed_class)
            writer.write(hud)
        stage_ms["render"] += (time.perf_counter() - t0) * 1000.0

        if args.max_keyframes and kf_count >= args.max_keyframes:
            logger.info("--max-keyframes %d reached; stopping early.", args.max_keyframes)
            break

    cap.release()
    if writer is not None:
        writer.release()
    total_pipeline_sec = time.perf_counter() - pipeline_t0

    # ---- GPU memory -------------------------------------------------------
    try:
        import torch
        peak_gpu_mb = (torch.cuda.max_memory_allocated() / 1e6
                       if torch.cuda.is_available() else None)
    except Exception:  # noqa: BLE001
        peak_gpu_mb = None

    # ---- summary (explicit warm-up timing lives in the stage loop, spec §40)

    # ---- summary -----------------------------------------------------------
    def counts(rows, key):
        out = {}
        for r in rows:
            v = r.get(key)
            if v:
                out[v] = out.get(v, 0) + 1
        return out

    raw_classes = counts(kf_rows, "raw_class")
    smoothed_classes = counts(kf_rows, "smoothed_class")
    status_counts = counts(all_rows, "skip_reason")

    # longest continuous MONOLAYER run over accepted keyframes (smoothed)
    best_run = best_run_sec = run = run_sec = 0
    prev_t = None
    for r in kf_rows:
        if r.get("smoothed_class") == "MONOLAYER":
            run += 1
            if prev_t is not None:
                run_sec += r["timestamp_sec"] - prev_t
        else:
            best_run, best_run_sec = max(best_run, run), max(best_run_sec, run_sec)
            run, run_sec = 0, 0.0
        prev_t = r["timestamp_sec"]
    best_run, best_run_sec = max(best_run, run), max(best_run_sec, run_sec)

    rbc_counts = [r["rbc_candidate_count"] for r in kf_rows if "rbc_candidate_count" in r]
    raw_scores = [r["raw_monolayer_score"] for r in kf_rows if "raw_monolayer_score" in r]

    summary = {
        "experiment": exp_name,
        "video_name": video_path.name,
        "duration_sec": round(n_meta / fps, 1),
        "width": width, "height": height, "fps_avg": round(fps, 3),
        "raw_frames": len(all_rows),
        "accepted_keyframes": kf_count,
        "keyframes_with_cellpose": sum(1 for r in kf_rows if r.get("cellpose_used")),
        "keyframes_fast_only": sum(1 for r in kf_rows if not r.get("cellpose_used")),
        "skipped_stationary": status_counts.get("SKIPPED_STATIONARY", 0),
        "skipped_rate_limit": status_counts.get("SKIPPED_RATE_LIMIT", 0),
        "skipped_blur": status_counts.get("SKIPPED_BLUR", 0),
        "class_raw": raw_classes,
        "class_smoothed": smoothed_classes,
        "longest_monolayer_run_keyframes": best_run,
        "longest_monolayer_run_sec": round(best_run_sec, 1),
        "rbc_count_min": min(rbc_counts) if rbc_counts else None,
        "rbc_count_median": statistics.median(rbc_counts) if rbc_counts else None,
        "rbc_count_max": max(rbc_counts) if rbc_counts else None,
        "raw_score_min": round(min(raw_scores), 3) if raw_scores else None,
        "raw_score_median": round(statistics.median(raw_scores), 3) if raw_scores else None,
        "raw_score_max": round(max(raw_scores), 3) if raw_scores else None,
        "cellpose_ms_mean": round(statistics.mean(cellpose_ms_list), 1) if cellpose_ms_list else None,
        "cellpose_ms_median": round(statistics.median(cellpose_ms_list), 1) if cellpose_ms_list else None,
        "cellpose_ms_first": cellpose_ms_list[0] if cellpose_ms_list else None,
        "warmup_inference_ms": round(warmup_ms, 1) if warmup_ms else None,
        "feature_ms_mean": round(statistics.mean(feature_ms_list), 1) if feature_ms_list else None,
        "total_ms_per_keyframe_mean": round(statistics.mean(total_kf_ms_list), 1) if total_kf_ms_list else None,
        "total_ms_per_keyframe_median": round(statistics.median(total_kf_ms_list), 1) if total_kf_ms_list else None,
        "model_load_s": round(model_load_s, 2),
        "decode_ms_total": round(stage_ms["decode"], 1),
        "motion_quality_ms_per_frame_mean": round(stage_ms["mq"] / max(1, stage_ms["mq_n"]), 2),
        "render_encode_ms_total": round(stage_ms["render"], 1),
        "total_pipeline_sec": round(total_pipeline_sec, 1),
        "estimated_analysis_fps": round(kf_count / total_pipeline_sec, 3) if total_pipeline_sec else None,
        "device": segmenter.device_desc,
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "precision": precision,
        "peak_gpu_mem_mb": round(peak_gpu_mb, 1) if peak_gpu_mb else None,
        "config": cfg,
    }
    write_summary_json(out_dir / "video_summary.json", summary)
    write_summary_csv(out_dir / "results_summary.csv", summary)
    write_results_csv(out_dir / "results.csv", all_rows)
    write_frame_features_csv(out_dir / "video_frame_features.csv", kf_rows)

    meta = collect_env_metadata(REPO_ROOT, extra={
        "experiment": exp_name,
        "video": str(video_path),
        "model_load_s": round(model_load_s, 2),
        "gpu_name": gpu_name, "cuda_version": cuda_version, "precision": precision,
        "peak_gpu_mem_mb": peak_gpu_mb,
    })
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str),
                                               encoding="utf-8")

    logger.info("-" * 70)
    logger.info("DONE raw frames=%d keyframes=%d (cellpose %d) total %.1f min",
                len(all_rows), kf_count, summary["keyframes_with_cellpose"],
                total_pipeline_sec / 60.0)
    logger.info("class raw: %s", raw_classes)
    logger.info("class smoothed: %s", smoothed_classes)
    logger.info("outputs -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
