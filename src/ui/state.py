"""Thread-safe shared state between capture, analysis workers and the UI.

Design: plain ``threading.Thread`` workers around one locked state object; the
Qt main thread POLLS this state on timers. No cross-thread Qt signals, so the
workers stay Qt-free and testable headlessly (--selftest).

Job flow (Phase G2): capture submits jobs to the JobManager (manual queue +
latest-frame auto slot); workers publish results here:
  * ``result``      - newest fresh AUTO result (drives live overlay + banner);
  * ``last_manual`` - newest completed MANUAL/FORCED job (job panel only -
    manual results belong to their snapshot, never the live view, spec §15);
  * ``history``     - (position -> score/class) scan history for the future
    slide-map panel. Only fresh AUTO results feed the temporal smoother (§28).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.live.jobs import AnalysisJob, JobManager

CELLPOSE_LOADING = "LOADING_MODEL"
CELLPOSE_IDLE = "IDLE"
CELLPOSE_PROCESSING = "PROCESSING"


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    quit_event: threading.Event = field(default_factory=threading.Event)

    # capture-side (updated every frame by the capture worker)
    frame_bgr: Optional[np.ndarray] = None
    frame_seq: int = -1
    t_capture_wall: float = 0.0
    capture_fps: float = 0.0
    source_error: str = ""

    # motion / screening (per frame)
    motion_state: str = "MOVING"
    disp_px: float = 0.0
    cum_x: float = 0.0
    cum_y: float = 0.0
    cum_total: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0
    screen_result: str = ""
    screen_occupancy: float = 0.0

    # controller / job requests
    controller_ref: Optional[object] = None      # LiveFieldController
    jobmgr: Optional[JobManager] = None          # analysis job registry (Phase G2)
    force_request: bool = False                  # hotkey A -> FORCED job
    manual_requests: int = 0                     # CAPTURE & ANALYZE -> MANUAL jobs
    auto_analysis_enabled: bool = True           # Auto Analyze ON/OFF toggle (§3)
    snapshot_requests: int = 0                   # SAVE SNAPSHOT (no Cellpose)
    analysis_enabled: bool = True                # SPACE master pause (auto only)
    pending_superseded: int = 0
    requested_fields: int = 0

    # analysis worker pool
    cellpose_state: str = CELLPOSE_LOADING
    cellpose_field_idx: int = -1
    cellpose_started_t: float = 0.0
    cellpose_busy_sec: float = 0.0
    analysis_count: int = 0
    first_inference_done: bool = False
    feature_mode: str = "live"                   # live | detailed (hotkey L)
    merge_analysis_live: bool = False

    # newest fresh AUTO result (live overlay + banner); manual results live in
    # the JobManager registry and `last_manual`, never on the live view (§15)
    result: Optional[AnalysisJob] = None
    last_manual: Optional[AnalysisJob] = None

    # UI-side toggles
    outlines_on: bool = True
    debug_ids: bool = False
    recording: bool = False
    view_job_id: Optional[int] = None            # selected completed job (job viewer)

    # scan history: dicts(t, frame_idx, cum_x, cum_y, score, raw_class, stale)
    history: List[Dict] = field(default_factory=list)

    # navigation state machine (Phase 2): fresh AUTO results only
    nav_state: str = "OUTSIDE_MONOLAYER"
    nav_score: Optional[float] = None
    nav_transitions: List[Dict] = field(default_factory=list)

    # session bookkeeping
    session_dir: Optional[str] = None
    session_settings: Dict = field(default_factory=dict)
    source_label: str = "source"
    last_error: str = ""

    # ------------------------------------------------------------------
    def publish_auto_result(self, job: AnalysisJob) -> None:
        with self.lock:
            self.result = job
            self.analysis_count += 1
            self.history.append({
                "t": job.t_capture, "frame_idx": job.frame_idx,
                "cum_x": job.cumulative_x, "cum_y": job.cumulative_y,
                "score": round(job.score, 4) if job.score is not None else None,
                "raw_class": job.raw_class, "stale": job.stale,
            })

    def publish_manual_result(self, job: AnalysisJob) -> None:
        with self.lock:
            self.last_manual = job
            self.analysis_count += 1
            self.history.append({
                "t": job.t_capture, "frame_idx": job.frame_idx,
                "cum_x": job.cumulative_x, "cum_y": job.cumulative_y,
                "score": round(job.score, 4) if job.score is not None else None,
                "raw_class": job.raw_class, "stale": False,
                "manual": True, "job_id": job.job_id,
            })

    def snapshot_display(self) -> Dict:
        """Atomic-ish read of everything the UI needs per tick."""
        with self.lock:
            return {
                "frame": None if self.frame_bgr is None else self.frame_bgr,
                "frame_seq": self.frame_seq,
                "capture_fps": self.capture_fps,
                "motion_state": self.motion_state,
                "disp_px": self.disp_px,
                "cum_x": self.cum_x,
                "cum_y": self.cum_y,
                "screen_result": self.screen_result,
                "cellpose_state": self.cellpose_state,
                "cellpose_field_idx": self.cellpose_field_idx,
                "cellpose_started_t": self.cellpose_started_t,
                "result": self.result,
                "last_manual": self.last_manual,
                "outlines_on": self.outlines_on,
                "debug_ids": self.debug_ids,
                "analysis_enabled": self.analysis_enabled,
                "auto_analysis_enabled": self.auto_analysis_enabled,
                "feature_mode": self.feature_mode,
                "analysis_count": self.analysis_count,
                "view_job_id": self.view_job_id,
                "source_error": self.source_error,
            }

    def set_frame(self, frame: np.ndarray, seq: int, capture_fps: float,
                  motion: dict) -> None:
        with self.lock:
            self.frame_bgr = frame
            self.frame_seq = seq
            self.capture_fps = capture_fps
            self.t_capture_wall = time.perf_counter()
            self.motion_state = motion["motion_state"]
            self.disp_px = motion["disp_px"]
            self.cum_x = motion["cum_x"]
            self.cum_y = motion["cum_y"]
            self.cum_total = motion["cum_total"]
            self.sharpness = motion["sharpness"]
            self.brightness = motion["brightness"]
            self.screen_result = motion["screen_result"]
            self.screen_occupancy = motion.get("screen_occupancy", 0.0)

    def current_cum(self) -> float:
        with self.lock:
            return self.cum_total

    def current_net(self) -> tuple[float, float]:
        """Net (signed) displacement sums - the staleness metric."""
        with self.lock:
            return self.cum_x, self.cum_y
