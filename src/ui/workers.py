"""Async workers: capture thread + configurable analysis pipeline (Phase G1-G8).

Pipeline (spec §7/§9 - GPU/CPU parallelism instead of N full model copies):

    CaptureWorker ──▶ JobManager (MANUAL > FORCED > AUTO, bounded) ──▶ GPUWorker(s)
                     (camera NEVER blocks; manual jobs never superseded)   │ Cellpose
                                                        MaskQueue ◀────────┘
                                                             │
                     GUI ◀── state/session ◀── FeatureWorker(s) (CPU pool)

  * CaptureWorker: reads the source, computes the CHEAP per-frame measurements
    (motion, sharpness, fast screen), fills AUTO jobs per the controller and
    MANUAL jobs on demand (CAPTURE & ANALYZE button / hotkey A). Manual
    requests are always honoured - even with Auto Analyze OFF (spec §3).
  * GPUWorker(s): ONE Cellpose model per worker (configurable count; benchmark
    before increasing - VRAM 11 GB), runs only inference, hands the mask on.
  * FeatureWorker(s): CPU pool - morphology, spatial features, merge suspects
    (manual/detailed jobs), score, classification; temporal smoothing is fed
    ONLY by fresh AUTO results (manual results never affect navigation state,
    spec §28). Results belong to their captured snapshot (spec §15).
  * CUDA OOM is caught per job: job FAILED with the error, session intact,
    no infinite retries (spec §35).
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np

from src.analysis.cell_features import (
    compute_reference_stats,
    flag_cells,
    measure_instances,
)
from src.live.controller import LiveFieldController
from src.live.jobs import (
    AnalysisJob,
    JobManager,
    CELLPOSE,
    COMPLETE,
    FAILED,
    FEATURES,
    QUEUED,
)
from src.postprocessing.merged_cell_splitter import analyze_merge_suspicion
from src.segmentation.cellpose_segmenter import CellposeSegmenter
from src.video.fast_screen import FastScreen
from src.video.monolayer import (
    classify_field,
    compute_field_features,
    prototype_monolayer_score,
)
from src.video.motion import MotionEstimator
from src.video.temporal import TemporalSmoother

from .rendering import build_result_layers
from .state import SharedState, CELLPOSE_IDLE, CELLPOSE_LOADING, CELLPOSE_PROCESSING

logger = logging.getLogger("live.capture")
logger_a = logging.getLogger("live.analysis")

_MOTION_WIDTH = 480


class CaptureWorker(threading.Thread):
    def __init__(self, source, state: SharedState, controller: LiveFieldController,
                 cfg: Dict, jobmgr: JobManager, session=None, scan_map=None) -> None:
        super().__init__(name="capture", daemon=True)
        self.source = source
        self.state = state
        self.controller = controller
        self.cfg = cfg
        self.jobmgr = jobmgr
        self.session = session
        self.scan_map = scan_map
        self.motion = MotionEstimator(_MOTION_WIDTH)
        self.screen = FastScreen(
            work_width=cfg.get("fast_screen_work_width", 480),
            **{k: v for k, v in cfg.items() if k.startswith(("thick_", "thin_"))},
        )
        self._prev_small: Optional[np.ndarray] = None
        self._fps_ema = 0.0
        self._last_read_t = 0.0
        self._recording_writer: Optional[cv2.VideoWriter] = None
        self.native_width = 1920
        self._threshold_scaled_for_width: Optional[int] = None
        self._reconnects = 0

    # ------------------------------------------------------------------
    def run(self) -> None:
        st = self.state
        detail = ""
        opened = False
        for attempt in range(1, 6):  # 5 tries x 3 s ~= 15 s of patience
            if st.quit_event.is_set():
                return
            if self.source.open():
                opened = True
                st.source_error = ""
                break
            detail = getattr(self.source, "last_error", "") or "could not open source"
            if attempt < 5:
                st.source_error = (
                    f"camera busy or starting up - retrying ({attempt}/4)..."
                )
                logger.warning("open attempt %d failed: %s", attempt, detail)
                time.sleep(3.0)
        if not opened:
            st.source_error = f"{self.source.name}: {detail}"
            logger.error(st.source_error)
            st.quit_event.set()
            return
        settings = self.source.actual_settings()
        logger.info("source opened: %s | %s", self.source.name, settings)
        st.lock.acquire()
        st.session_settings = settings  # type: ignore[attr-defined]
        st.lock.release()

        seq = 0
        try:
            while not st.quit_event.is_set():
                t0 = time.perf_counter()
                frame, t_src = self.source.read()
                if frame is None:
                    if getattr(self.source, "loop", False):
                        self.source.close()
                        if not self.source.open():
                            st.source_error = "source loop reopen failed"
                            break
                        continue
                    if getattr(self.source, "reconnect", False) and self._reconnects < 5:
                        self._reconnects += 1
                        logger.warning("camera returned no frame; re-opening (%d/5)",
                                       self._reconnects)
                        self.source.close()
                        time.sleep(1.0)
                        if self.source.open():
                            self._prev_small = None
                            continue
                    st.source_error = (
                        "camera stopped delivering frames "
                        f"({getattr(self.source, 'last_error', '') or 'device ended'})"
                    )
                    logger.info("capture stopping: %s", st.source_error)
                    break

                now = time.perf_counter()
                dt = now - self._last_read_t if self._last_read_t else 0.0
                self._last_read_t = now
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps_ema = inst if self._fps_ema == 0 else (
                        0.9 * self._fps_ema + 0.1 * inst
                    )

                seq += 1
                try:
                    self._process_frame(frame, t_src, seq)
                except Exception:  # noqa: BLE001 - a live feed must never hang
                    logger.exception("capture frame %d failed", seq)
                    st.last_error = "capture frame processing error (see run.log)"
                    break
                if (
                    self.cfg.get("max_analyses")
                    and st.analysis_count >= self.cfg["max_analyses"]
                ):
                    logger.info("max_analyses reached (%d); capture stopping", st.analysis_count)
                    break
        finally:
            self._stop_recording()
            self.source.close()
            logger.info("capture worker stopped (%d frames)", seq)

    def _process_frame(self, frame: np.ndarray, t_src: float, seq: int) -> None:
        st = self.state
        h, w = frame.shape[:2]
        self.native_width = w

        if self._threshold_scaled_for_width != w:
            base = float(self.cfg.get("motion_threshold_px", 240.0))
            self.controller.motion_threshold_px = base * (w / 1920.0)
            self._threshold_scaled_for_width = w
            logger.info("motion threshold auto-scaled: %.0f px at width %d",
                        self.controller.motion_threshold_px, w)

        small = self.motion.prep(frame)
        dx = dy = disp = 0.0
        if self._prev_small is not None:
            dx, dy, _resp = self.motion.displacement(self._prev_small, small, w)
            disp = float(math.hypot(dx, dy))
        self._prev_small = small

        sharp_small = float(
            cv2.Laplacian(small.astype(np.uint8), cv2.CV_64F).var()
        )
        scale_factor = (w / float(_MOTION_WIDTH)) ** 2
        sharpness = sharp_small * scale_factor
        brightness = float(small.mean())

        screen = self.screen.evaluate(frame)

        force = False
        manual = False
        with st.lock:
            if st.force_request:
                force = True
                st.force_request = False
            if st.manual_requests > 0:
                manual = True
                st.manual_requests -= 1

        decision = self.controller.update(
            seq, t_src, dx, dy, disp, sharpness, force=force
        )

        st.set_frame(frame, seq, self._fps_ema, {
            "motion_state": decision.motion_state,
            "disp_px": disp,
            "cum_x": self.controller.cumulative[0],
            "cum_y": self.controller.cumulative[1],
            "cum_total": self.controller.cumulative[2],
            "sharpness": sharpness,
            "brightness": brightness,
            "screen_result": screen["fast_screen_result"],
            "screen_occupancy": screen["screen_occupancy"],
        })

        if self.state.recording:
            self._ensure_recording(w, h)
            if self._recording_writer is not None:
                self._recording_writer.write(frame)

        if self.session is not None and seq % 30 == 0:
            self.session.log_raw(seq, t_src, disp, sharpness, brightness,
                                 decision.motion_state, screen["fast_screen_result"])

        with st.lock:
            want_snapshot = st.snapshot_requests > 0
            if want_snapshot:
                st.snapshot_requests -= 1
        if want_snapshot and self.session is not None:
            self.session.save_snapshot(frame, seq)
            logger.info("snapshot saved (frame %d)", seq)

        # ---- job submission (spec §2/§3/§8) ------------------------------
        cx, cy, ct = self.controller.cumulative
        off_x, off_y = (self.scan_map.current_offset()
                        if self.scan_map is not None else (0.0, 0.0))
        base_meta = dict(
            cumulative_x=cx, cumulative_y=cy, cumulative_total=ct,
            screen_result=screen["fast_screen_result"],
            screen_occupancy=screen["screen_occupancy"],
            sharpness=sharpness, brightness=brightness,
        )
        if manual:
            job = self.jobmgr.submit("MANUAL", frame, seq, t_src,
                                     reason="MANUAL_CAPTURE", **base_meta)
            if job is None:
                logger.warning("MANUAL capture REFUSED: manual queue full")
            else:
                logger.info("manual job #%d queued (f%d)", job.job_id, seq)
            return

        if force:
            job = self.jobmgr.submit(
                "FORCED", frame, seq, t_src, map_x=cx + off_x, map_y=cy + off_y,
                reason="FORCED_A", **base_meta)
            if job is not None:
                logger.info("forced job #%d queued (f%d)", job.job_id, seq)
            return

        auto_on = st.analysis_enabled and st.auto_analysis_enabled
        if decision.request_field and auto_on:
            hybrid = self.cfg.get("cellpose_mode", "benchmark") == "hybrid"
            if hybrid and screen["fast_screen_result"] != "UNCERTAIN":
                return  # Mode B: screening already decided - skip inference
            job = self.jobmgr.submit(
                "AUTO", frame, seq, t_src, map_x=cx + off_x, map_y=cy + off_y,
                reason=decision.request_reason, **base_meta)
            if job is not None:
                logger.debug("auto job #%d queued (f%d, %s)",
                             job.job_id, seq, decision.request_reason)

    # ------------------------------------------------------------------
    def _ensure_recording(self, w: int, h: int) -> None:
        if self._recording_writer is not None or self.session is None:
            return
        path = self.session.recording_path()
        self._recording_writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, self.cfg.get("camera_fps", 30.0)), (w, h),
        )
        logger.info("recording -> %s", path)

    def _stop_recording(self) -> None:
        if self._recording_writer is not None:
            self._recording_writer.release()
            self._recording_writer = None


class _MaskQueue:
    """Bounded hand-off between GPU workers and CPU feature workers."""

    def __init__(self, maxsize: int = 8):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, job: AnalysisJob) -> bool:
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            logger.error("mask queue full - job #%d dropped (should not happen)", job.job_id)
            self.jobmgr_mark_failed(job, "mask queue overflow")
            return False

    def get(self, timeout: float) -> Optional[AnalysisJob]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    @staticmethod
    def jobmgr_mark_failed(job: AnalysisJob, reason: str) -> None:
        job.status = FAILED
        job.error = reason


class GPUWorker(threading.Thread):
    """One resident Cellpose model; runs ONLY inference (spec §9)."""

    def __init__(self, worker_id: str, state: SharedState, cfg: Dict,
                 jobmgr: JobManager, maskq: _MaskQueue) -> None:
        super().__init__(name=worker_id, daemon=True)
        self.worker_id = worker_id
        self.state = state
        self.cfg = cfg
        self.jobmgr = jobmgr
        self.maskq = maskq

    def run(self) -> None:
        st = self.state
        st.cellpose_state = CELLPOSE_LOADING
        logger_a.info("[%s] loading Cellpose model '%s' ...",
                      self.worker_id, self.cfg.get("model_name", "cpsam_v2"))
        try:
            segmenter = CellposeSegmenter(
                model_name=self.cfg.get("model_name", "cpsam_v2"),
                gpu=self.cfg.get("gpu", "auto"),
                use_bfloat16=self.cfg.get("use_bfloat16", False),
                cellprob_threshold=self.cfg.get("cellprob_threshold", 0.0),
                flow_threshold=self.cfg.get("flow_threshold", 0.4),
                min_size=self.cfg.get("min_size", 15),
                diameter=self.cfg.get("diameter"),
                normalize=self.cfg.get("normalize", True),
                augment=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger_a.exception("[%s] model load failed", self.worker_id)
            st.last_error = f"Cellpose load failed: {exc}"
            st.cellpose_state = "ERROR"
            return

        logger_a.info("[%s] model ready on %s (%.1fs)",
                      self.worker_id, segmenter.device_desc, segmenter.model_load_s)
        st.cellpose_state = CELLPOSE_IDLE

        while not st.quit_event.is_set():
            job = self.jobmgr.take_next()
            if job is None:
                continue
            if not st.analysis_enabled:
                # master pause: leave QUEUED jobs queued (manual ones survive)
                self._requeue_paused(job)
                continue
            self._run_cellpose(segmenter, job)
        logger_a.info("[%s] stopped", self.worker_id)

    def _requeue_paused(self, job: AnalysisJob) -> None:
        # put it back at the FRONT of its own queue without losing order
        time.sleep(0.2)
        with self.jobmgr._lock:
            if job.priority == "AUTO":
                if self.jobmgr._auto_pending is None:
                    self.jobmgr._auto_pending = job
                else:
                    self.jobmgr._auto_pending = job  # replace - auto semantics
            else:
                self.jobmgr._manual[job.priority].appendleft(job)

    def _run_cellpose(self, segmenter: CellposeSegmenter, job: AnalysisJob) -> None:
        st = self.state
        with st.lock:
            st.cellpose_state = CELLPOSE_PROCESSING
            st.cellpose_field_idx = job.frame_idx
            st.cellpose_started_t = time.perf_counter()
        job.status = CELLPOSE
        job.worker_id = self.worker_id
        job.t_start = time.perf_counter()
        job.queue_wait_ms = (job.t_start - job.t_capture) * 1000.0

        rgb = cv2.cvtColor(job.frame, cv2.COLOR_BGR2RGB)
        target_w = self.cfg.get("cellpose_width") or job.frame.shape[1]
        try:
            if target_w < job.frame.shape[1]:
                scale = target_w / job.frame.shape[1]
                small = cv2.resize(rgb, (target_w, int(round(rgb.shape[0] * scale))),
                                   interpolation=cv2.INTER_AREA)
                seg = segmenter.segment(small)
                job.labels = cv2.resize(seg.labels, (job.frame.shape[1], job.frame.shape[0]),
                                        interpolation=cv2.INTER_NEAREST).astype(np.int32)
            else:
                seg = segmenter.segment(rgb)
                job.labels = np.asarray(seg.labels, dtype=np.int32)
        except Exception as exc:  # noqa: BLE001 - CUDA OOM etc. (spec §35)
            oom = "out of memory" in str(exc).lower()
            logger_a.error("[%s] job #%d inference failed: %s",
                           self.worker_id, job.job_id, exc)
            job.status = FAILED
            job.error = ("CUDA OOM" if oom else f"inference error: {exc}")
            job.t_end = time.perf_counter()
            self.jobmgr.mark(job, FAILED, self.worker_id)
            with st.lock:
                st.cellpose_state = CELLPOSE_IDLE
                st.last_error = f"job #{job.job_id} {job.error}"
            return

        job.cellpose_ms = seg.timings_ms["inference_ms"]
        job.first_inference = not st.first_inference_done
        st.first_inference_done = True
        job.status = FEATURES
        self.maskq.put(job)
        with st.lock:
            st.cellpose_state = CELLPOSE_IDLE
            st.cellpose_busy_sec += (time.perf_counter() - job.t_start)


class FeatureWorker(threading.Thread):
    """CPU-stage worker: features, merge suspects, score, artifacts (spec §9/§20)."""

    def __init__(self, worker_id: str, state: SharedState, cfg: Dict,
                 jobmgr: JobManager, maskq: _MaskQueue, session=None,
                 smoother: Optional[TemporalSmoother] = None,
                 scan_map=None, nav_machine=None) -> None:
        super().__init__(name=worker_id, daemon=True)
        self.worker_id = worker_id
        self.state = state
        self.cfg = cfg
        self.jobmgr = jobmgr
        self.maskq = maskq
        self.session = session
        self.scan_map = scan_map
        self.nav_machine = nav_machine  # shared Phase 2 state machine
        self.smoother = smoother or TemporalSmoother(
            alpha=cfg.get("temporal_alpha", 0.4),
            window=cfg.get("temporal_window", 5),
            monolayer_score_min=cfg.get("monolayer_score_min", 0.60),
        )

    def run(self) -> None:
        st = self.state
        while not st.quit_event.is_set():
            job = self.maskq.get(timeout=0.5)
            if job is None:
                continue
            try:
                self._run_features(job)
            except Exception as exc:  # noqa: BLE001 - one job failing never kills the pool
                logger_a.exception("features for job #%d failed", job.job_id)
                job.status = FAILED
                job.error = f"feature error: {exc}"
                job.t_end = time.perf_counter()
                self.jobmgr.mark(job, FAILED, self.worker_id)

    def _run_features(self, job: AnalysisJob) -> None:
        st = self.state
        job.status = FEATURES
        job.feature_mode = st.feature_mode
        job.worker_id = f"{job.worker_id}+{self.worker_id}"
        t0 = time.perf_counter()

        labels = job.labels
        rows = measure_instances(labels)
        ref = compute_reference_stats(rows, stats_scope="interior")
        flag_cells(rows, ref,
                   small_area_factor=self.cfg.get("small_area_factor", 0.35),
                   wbc_area_factor=self.cfg.get("wbc_area_factor", 3.5))

        # MANUAL/FORCED captures get the full detailed treatment incl. merge
        # suspects (spec §20/§22); AUTO stays on the fast live path unless
        # detailed mode is forced.
        detailed = job.priority in ("MANUAL", "FORCED") or st.feature_mode == "detailed"
        merge_ran = detailed or st.merge_analysis_live
        if merge_ran:
            analyze_merge_suspicion(
                labels, rows, ref,
                area_factor_min=self.cfg.get("merge_area_factor_min", 1.6),
                solidity_max=self.cfg.get("merge_solidity_max", 0.90),
                elongation_min=self.cfg.get("merge_elongation_min", 1.5),
                min_signals=self.cfg.get("merge_min_signals", 2),
            )
        else:
            for row in rows:
                row["possible_merged_rbc"] = False
                row["n_dt_peaks"] = 0

        f_cfg = self.cfg
        ref_frame = f_cfg.get("reference_frame") or None
        features = compute_field_features(
            rows, labels, ref,
            valid_roi_margin=f_cfg.get("valid_roi_margin", 0.10),
            contact_margin_fraction=f_cfg.get("contact_margin_fraction", 0.15),
            neighbor_distance_threshold=f_cfg.get("neighbor_distance_threshold", 1.5),
            large_cluster_min_cells=f_cfg.get("large_cluster_min_cells", 30),
            uniformity_grid=f_cfg.get("uniformity_grid", 4),
            reference_shape=tuple(ref_frame) if ref_frame else None,
        )
        features["screen_occupancy"] = float(job.screen_occupancy)
        score, comps = prototype_monolayer_score(features, self.cfg["monolayer_score_cfg"])
        raw_class, _evidence = classify_field(
            features, score, self.cfg["classification_cfg"],
            screen_occupancy=job.screen_occupancy,
        )
        features_ms = (time.perf_counter() - t0) * 1000.0

        # ---- freshness: only AUTO results drive the live field (spec §15/§16)
        stale, stale_reason = False, ""
        if job.priority == "AUTO":
            now = time.perf_counter()
            net_x, net_y = st.current_net()
            stale = job.is_stale(net_x, net_y, now,
                                 self.cfg.get("result_stale_displacement", 160.0),
                                 self.cfg.get("result_stale_seconds", 25.0))
            stale_reason = "moved" if stale else ""
            if stale:
                smoothed_score = smoothed_class = None  # never smoothed when stale
            else:
                smoothed_score, smoothed_class = self.smoother.update(score, raw_class)
        else:
            smoothed_score = smoothed_class = None  # manual: field-local result only

        overlay_layer, id_layer = build_result_layers(labels, rows, debug_ids=True)

        job.rows = rows
        job.features = features
        job.score = score
        job.score_components = comps
        job.raw_class = raw_class
        job.smoothed_score = smoothed_score
        job.smoothed_class = smoothed_class
        job.stale = stale
        job.stale_reason = stale_reason
        job.overlay_layer = overlay_layer
        job.id_layer = id_layer
        job.features_ms = features_ms
        job.t_end = time.perf_counter()
        self.jobmgr.mark(job, COMPLETE, self.worker_id)

        # ---- persistent scan map (Phase 1) + monolayer evidence layer (Phase 3)
        if self.scan_map is not None and job.score is not None:
            source_image = f"keyframes/job_{job.job_id:04d}_f{job.frame_idx:06d}.jpg"
            self.scan_map.add_field(job, source_image=source_image)
            ml_summary = self.scan_map.update_monolayer_layer(
                job,
                contradictions_to_demote=self.cfg.get("contradictions_to_demote", 2),
                overlap_min_fraction=self.cfg.get("overlap_min_fraction", 0.25),
                human_label_weight=self.cfg.get("human_label_weight", 2),
            )
            if any(ml_summary.values()):
                logger_a.info("monolayer layer: %s", ml_summary)
            if self.session is not None:
                self.session.save_scan_map(self.scan_map)

        if job.priority == "AUTO" and not job.stale:
            # navigation state machine: fresh AUTO results only (spec Phase 2)
            if self.nav_machine is not None:
                transitions_before = len(self.nav_machine.transitions)
                view = self.nav_machine.update(
                    smoothed_score if smoothed_score is not None else score,
                    t=job.t_capture, frame_idx=job.frame_idx,
                    x=job.cumulative_x, y=job.cumulative_y,
                )
                with st.lock:
                    st.nav_state = view["state"]
                    st.nav_score = view["score"]
                    st.nav_transitions = self.nav_machine.transitions_as_dicts()
                if len(self.nav_machine.transitions) > transitions_before:
                    tr = self.nav_machine.transitions[-1]
                    logger_a.info("NAV %s -> %s (score %.2f, f%d)",
                                  tr.from_state, tr.to_state, tr.score, job.frame_idx)
                    if self.session is not None:
                        self.session.log_nav_transition(tr)

        if job.priority == "AUTO":
            st.publish_auto_result(job)
        else:
            st.publish_manual_result(job)

        logger_a.info(
            "job #%d [%s] %s | %-9s score %.2f smooth %s | RBC %d cov %.2f "
            "merges %s | cellpose %.0fms feat %.0fms%s",
            job.job_id, job.priority, ("STALE" if stale else "fresh"),
            raw_class, score,
            f"{smoothed_score:.2f}" if smoothed_score is not None else "n/a",
            features["rbc_candidate_count"], features["coverage"],
            (f"{features['merge_suspect_count']}" if merge_ran else "n/a"),
            job.cellpose_ms or 0.0, features_ms,
            job.error or "",
        )
        if self.session is not None:
            self.session.log_accepted(job)
            self.session.save_field_artifacts(job, job.frame)
