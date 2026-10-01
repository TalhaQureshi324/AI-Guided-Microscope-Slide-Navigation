"""Thread-safe shared state between capture, analysis worker and the UI.

Design: plain ``threading.Thread`` workers + one locked state object; the Qt
main thread POLLS this state on timers. No cross-thread Qt signals, so the
workers stay Qt-free and testable headlessly (--selftest).

Everything the UI renders comes from here:
  * ``frame_bgr``          - newest camera frame (always current, spec §7);
  * ``motion/screen``      - per-frame cheap measurements;
  * ``cellpose_state``     - LOADING_MODEL / IDLE / PROCESSING (+ field id, age);
  * ``result``             - newest AnalysisResult (worker-built overlay layers
                             included), staleness evaluated at display time;
  * ``history``            - (position -> score/class) scan history for the
                             future slide-map panel (spec: preserve it now).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.live.controller import FieldRequest

CELLPOSE_LOADING = "LOADING_MODEL"
CELLPOSE_IDLE = "IDLE"
CELLPOSE_PROCESSING = "PROCESSING"


@dataclass
class AnalysisResult:
    """One finished detailed field analysis (spec §15: full freshness record)."""

    field: FieldRequest
    labels: np.ndarray                      # native-resolution int32 labels
    rows: List[Dict]                        # measured instances with flags
    features: Dict
    score: float
    score_components: Dict
    raw_class: str
    smoothed_score: Optional[float]
    smoothed_class: Optional[str]
    stale: bool                             # computed at completion time
    stale_reason: str
    cellpose_ms: float
    features_ms: float
    latency_ms: float
    t_start: float
    t_end: float
    feature_mode: str
    merge_analysis_ran: bool
    first_inference: bool
    overlay_layer: Optional[np.ndarray] = None   # BGR, black background
    id_layer: Optional[np.ndarray] = None        # BGR, instance IDs (debug)

    @property
    def result_age_sec(self) -> float:
        return time.perf_counter() - self.field.t_capture


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

    # controller / worker coordination
    controller_ref: Optional[object] = None      # LiveFieldController
    force_request: bool = False
    analysis_enabled: bool = True
    pending_superseded: int = 0
    requested_fields: int = 0

    # analysis worker
    cellpose_state: str = CELLPOSE_LOADING
    cellpose_field_idx: int = -1
    cellpose_started_t: float = 0.0
    cellpose_busy_sec: float = 0.0
    analysis_count: int = 0
    first_inference_done: bool = False
    feature_mode: str = "live"                   # live | detailed (hotkey L)
    merge_analysis_live: bool = False

    # newest finished result (None until the first analysis completes)
    result: Optional[AnalysisResult] = None

    # UI-side toggles
    outlines_on: bool = True
    debug_ids: bool = False
    recording: bool = False

    # scan history: dicts(t, frame_idx, cum_x, cum_y, score, raw_class, stale)
    history: List[Dict] = field(default_factory=list)

    # session bookkeeping
    session_dir: Optional[str] = None
    last_error: str = ""
    _pending: Optional[FieldRequest] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def submit_field(self, req: FieldRequest) -> None:
        """Latest-frame slot: a new request REPLACES any pending older one."""
        with self.lock:
            if self._pending is not None:
                self.pending_superseded += 1
            self._pending = req
            self.requested_fields += 1

    def take_pending_field(self, timeout: float = 0.5) -> Optional[FieldRequest]:
        """Worker side: block briefly for the newest pending field.

        A pending field is returned even when quit was requested - shutdown
        must not silently drop a field the worker can still analyse.
        """
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self.lock:
                if self._pending is not None:
                    req, self._pending = self._pending, None
                    return req
            if self.quit_event.is_set():
                return None
            self.quit_event.wait(0.05)
        return None

    def has_pending(self) -> bool:
        with self.lock:
            return self._pending is not None

    def publish_result(self, result: AnalysisResult, history_entry: Dict) -> None:
        with self.lock:
            self.result = result
            self.analysis_count += 1
            self.history.append(history_entry)

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
                "outlines_on": self.outlines_on,
                "debug_ids": self.debug_ids,
                "analysis_enabled": self.analysis_enabled,
                "feature_mode": self.feature_mode,
                "analysis_count": self.analysis_count,
                "pending_superseded": self.pending_superseded,
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
