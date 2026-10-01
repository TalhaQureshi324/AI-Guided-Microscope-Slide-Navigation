"""Async workers: capture thread + single Cellpose analysis worker.

Both are plain ``threading.Thread`` objects around a :class:`SharedState` -
deliberately Qt-free so the camera/UI never blocks on Cellpose and the whole
live pipeline can run headless (--selftest).

CAPTURE WORKER (spec §6, §7, §8):
  reads the source as fast as it produces frames, computes the CHEAP per-frame
  measurements (motion via phase correlation, sharpness/brightness on the
  480 px motion image, fast screen), updates the latest-frame slot, and fills
  the single pending-field slot when the controller says a NEW FIELD is ready.
  A new pending field REPLACES an older one (latest-frame strategy, spec §7/§43)
  - there is no queue and no backlog by construction.

ANALYSIS WORKER (spec §12-§16):
  loads the Cellpose model ONCE, keeps it resident on the GPU, and processes
  at most one field at a time: newest pending field -> optional downscale ->
  Cellpose -> features (live mode skips the expensive DT merge analysis) ->
  prototype score -> classification -> temporal smoothing (non-stale results
  only) -> overlay layers -> publish. Stale results are stored and logged but
  never smoothed or displayed as current (spec §15/§16).
"""

from __future__ import annotations

import logging
import math
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
from src.live.controller import FieldRequest, LiveFieldController
from src.postprocessing.merged_cell_splitter import analyze_merge_suspicion
from src.segmentation.cellpose_segmenter import CellposeSegmenter
from src.video.fast_screen import FastScreen
from src.video.monolayer import (
    classify_field,
    compute_field_features,
    prototype_monolayer_score,
)
from src.video.motion import MotionEstimator
from src.video.quality import frame_quality
from src.video.temporal import TemporalSmoother

from .rendering import build_result_layers
from .state import (
    AnalysisResult,
    SharedState,
    CELLPOSE_IDLE,
    CELLPOSE_LOADING,
    CELLPOSE_PROCESSING,
)

logger = logging.getLogger("live.capture")
logger_a = logging.getLogger("live.analysis")

_MOTION_WIDTH = 480


class CaptureWorker(threading.Thread):
    def __init__(self, source, state: SharedState, controller: LiveFieldController,
                 cfg: Dict, session=None) -> None:
        super().__init__(name="capture", daemon=True)
        self.source = source
        self.state = state
        self.controller = controller
        self.cfg = cfg
        self.session = session
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
        # Opening can legitimately fail on the first pass: the previous app
        # instance may still be releasing the device, or another camera app
        # may be closing. Retry with backoff instead of giving up - this
        # turns the most common camera failure into a self-healing pause.
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

        kf = self.cfg
        seq = 0
        try:
            while not st.quit_event.is_set():
                frame, t_src = self.source.read()
                if frame is None:
                    if getattr(self.source, "loop", False):
                        self.source.close()
                        if not self.source.open():
                            st.source_error = "source loop reopen failed"
                            break
                        continue
                    if getattr(self.source, "reconnect", False) and self._reconnects < 5:
                        # cameras hiccup (USB jitter, brief app collisions):
                        # re-open instead of ending the session
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

        # motion threshold scales with the actual captured width: the config
        # value is calibrated at 1920 px (~8 RBC diameters); USB cameras grant
        # different modes between runs (800x448 vs 1080p), so normalise.
        if self._threshold_scaled_for_width != w:
            base = float(self.cfg.get("motion_threshold_px", 240.0))
            self.controller.motion_threshold_px = base * (w / 1920.0)
            self._threshold_scaled_for_width = w
            logger.info("motion threshold auto-scaled: %.0f px at width %d",
                        self.controller.motion_threshold_px, w)

        # --- cheap per-frame measurements on the motion-work resolution ---
        small = self.motion.prep(frame)
        dx = dy = disp = 0.0
        if self._prev_small is not None:
            dx, dy, _resp = self.motion.displacement(self._prev_small, small, w)
            disp = float(math.hypot(dx, dy))
        self._prev_small = small

        # sharpness on the small gray, rescaled to native-equivalent Laplacian
        # variance (variance scales ~ area factor); the blur guard is a gross
        # gate only (calibrated note in src/video/quality.py). Cast float32 ->
        # uint8 first: OpenCV's Laplacian rejects float32+CV_64F combinations.
        sharp_small = float(
            cv2.Laplacian(small.astype(np.uint8), cv2.CV_64F).var()
        )
        scale_factor = (w / float(_MOTION_WIDTH)) ** 2
        sharpness = sharp_small * scale_factor
        brightness = float(small.mean())

        screen = self.screen.evaluate(frame)

        force = False
        with st.lock:
            if st.force_request:
                force = True
                st.force_request = False

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

        # --- field request (latest-frame slot) ----------------------------
        cx, cy, ct = self.controller.cumulative
        if decision.request_field and st.analysis_enabled:
            hybrid = self.cfg.get("cellpose_mode", "benchmark") == "hybrid"
            if hybrid and screen["fast_screen_result"] != "UNCERTAIN" and not force:
                # Mode B: screening already decides - skip expensive analysis
                return
            req = FieldRequest(
                frame_idx=seq, t_capture=t_src,
                cumulative_x=cx, cumulative_y=cy, cumulative_total=ct,
                reason=decision.request_reason,
                screen_result=screen["fast_screen_result"],
                sharpness=sharpness, brightness=brightness,
                frame=frame,
            )
            st.submit_field(req)
            logger.debug("field requested #%d f%d (%s)", st.requested_fields, seq,
                         decision.request_reason)

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


class AnalysisWorker(threading.Thread):
    def __init__(self, state: SharedState, cfg: Dict, session=None) -> None:
        super().__init__(name="analysis", daemon=True)
        self.state = state
        self.cfg = cfg
        self.session = session
        self.smoother = TemporalSmoother(
            alpha=cfg.get("temporal_alpha", 0.4),
            window=cfg.get("temporal_window", 5),
            monolayer_score_min=cfg.get("monolayer_score_min", 0.60),
        )

    # ------------------------------------------------------------------
    def run(self) -> None:
        st = self.state
        seg_cfg = self.cfg
        st.cellpose_state = CELLPOSE_LOADING
        logger_a.info("loading Cellpose model '%s' ...", seg_cfg.get("model_name", "cpsam_v2"))
        try:
            segmenter = CellposeSegmenter(
                model_name=seg_cfg.get("model_name", "cpsam_v2"),
                gpu=seg_cfg.get("gpu", "auto"),
                use_bfloat16=seg_cfg.get("use_bfloat16", False),
                cellprob_threshold=seg_cfg.get("cellprob_threshold", 0.0),
                flow_threshold=seg_cfg.get("flow_threshold", 0.4),
                min_size=seg_cfg.get("min_size", 15),
                diameter=seg_cfg.get("diameter"),
                normalize=seg_cfg.get("normalize", True),
                augment=False,
            )
        except Exception as exc:  # noqa: BLE001 - surface in UI, keep feed alive
            logger_a.exception("model load failed")
            st.last_error = f"Cellpose load failed: {exc}"
            st.cellpose_state = "ERROR"
            return

        logger_a.info("model ready on %s (%.1fs)",
                      segmenter.device_desc, segmenter.model_load_s)
        st.cellpose_state = CELLPOSE_IDLE

        while not st.quit_event.is_set():
            req = st.take_pending_field(timeout=0.4)
            if req is None:
                continue
            if not st.analysis_enabled:
                continue
            try:
                self._analyze_one(segmenter, req)
            except Exception as exc:  # noqa: BLE001 - skip the field, stay alive
                logger_a.exception("analysis of field f%d failed", req.frame_idx)
                with st.lock:
                    st.cellpose_state = CELLPOSE_IDLE
                    st.last_error = f"analysis error on f{req.frame_idx}: {exc}"

        logger_a.info("analysis worker stopped (%d fields, %.1f s busy)",
                      st.analysis_count, st.cellpose_busy_sec)

    # ------------------------------------------------------------------
    def _analyze_one(self, segmenter: CellposeSegmenter, req: FieldRequest) -> None:
        st = self.state
        frame = req.frame  # the EXACT field that was requested (spec §15)
        if frame is None:
            logger_a.warning("request f%d carries no frame; skipping", req.frame_idx)
            return

        st.lock.acquire()
        st.cellpose_state = CELLPOSE_PROCESSING
        st.cellpose_field_idx = req.frame_idx
        st.cellpose_started_t = time.perf_counter()
        st.lock.release()
        t_start = time.perf_counter()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # optional reduced-resolution inference (spec §31)
        target_w = self.cfg.get("cellpose_width") or frame.shape[1]
        if target_w < frame.shape[1]:
            scale = target_w / frame.shape[1]
            small = cv2.resize(rgb, (target_w, int(round(rgb.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA)
            seg = segmenter.segment(small)
            labels = cv2.resize(seg.labels, (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(np.int32)
        else:
            seg = segmenter.segment(rgb)
            labels = seg.labels
        cellpose_ms = seg.timings_ms["inference_ms"]
        first = not st.first_inference_done
        st.first_inference_done = True

        # --- features (live mode skips the expensive DT merge analysis) ---
        rows = measure_instances(labels)
        ref = compute_reference_stats(rows, stats_scope="interior")
        flag_cells(rows, ref,
                   small_area_factor=self.cfg.get("small_area_factor", 0.35),
                   wbc_area_factor=self.cfg.get("wbc_area_factor", 3.5))
        merge_ran = (st.feature_mode == "detailed") or st.merge_analysis_live
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
        score, comps = prototype_monolayer_score(features, self.cfg["monolayer_score_cfg"])
        raw_class, _evidence = classify_field(features, score, self.cfg["classification_cfg"])
        features_ms = (time.perf_counter() - t_start) * 1000.0 - cellpose_ms

        # --- freshness (spec §15/§16) --------------------------------------
        now = time.perf_counter()
        net_x, net_y = st.current_net()
        stale = req.is_stale(net_x, net_y, now,
                             self.cfg.get("result_stale_displacement", 160.0),
                             self.cfg.get("result_stale_seconds", 25.0))
        stale_reason = "moved" if stale else ""
        if not stale:
            age = now - req.t_capture
            if age > self.cfg.get("result_stale_seconds", 25.0):
                stale, stale_reason = True, "age"
        if stale:
            smoothed_score = smoothed_class = None  # never smoothed when stale
        else:
            smoothed_score, smoothed_class = self.smoother.update(score, raw_class)

        overlay_layer, id_layer = build_result_layers(labels, rows, debug_ids=True)

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        result = AnalysisResult(
            field=req, labels=labels, rows=rows, features=features,
            score=score, score_components=comps, raw_class=raw_class,
            smoothed_score=smoothed_score, smoothed_class=smoothed_class,
            stale=stale, stale_reason=stale_reason,
            cellpose_ms=cellpose_ms, features_ms=features_ms,
            latency_ms=latency_ms, t_start=t_start, t_end=time.perf_counter(),
            feature_mode=st.feature_mode, merge_analysis_ran=merge_ran,
            first_inference=first, overlay_layer=overlay_layer, id_layer=id_layer,
        )
        history_entry = {
            "t": req.t_capture, "frame_idx": req.frame_idx,
            "cum_x": req.cumulative_x, "cum_y": req.cumulative_y,
            "score": round(score, 4), "raw_class": raw_class, "stale": stale,
        }
        st.publish_result(result, history_entry)

        with st.lock:
            st.cellpose_busy_sec += latency_ms / 1000.0
            st.cellpose_state = CELLPOSE_IDLE

        logger_a.info(
            "field f%d %s | %-9s score %.2f smooth %s | RBC %d cov %.2f | "
            "cellpose %.0fms feat %.0fms | stale=%s",
            req.frame_idx, req.reason, raw_class, score,
            f"{smoothed_score:.2f}" if smoothed_score is not None else "n/a",
            features["rbc_candidate_count"], features["coverage"],
            cellpose_ms, features_ms, stale,
        )
        if self.session is not None:
            self.session.log_accepted(result)
            self.session.save_field_artifacts(result, frame)
