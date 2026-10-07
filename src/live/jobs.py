"""Analysis job manager (Phase G2/G8): explicit jobs, priorities, bounded queues.

Replaces the single pending-slot for automatic fields while keeping its
latest-frame semantics:

  * MANUAL jobs (CAPTURE & ANALYZE button, or forced hotkey A) are the user's
    deliberate choices - they are QUEUED and NEVER silently discarded
    (spec §4/§12). Bounded by ``max_manual_queue``; when full the submit is
    REFUSED loudly, never silently dropped.
  * AUTO jobs follow the existing latest-frame policy: a newer auto request
    REPLACES an older pending one (old fields are irrelevant), counting
    supersessions as before.

Priority order when a GPU worker picks work (spec §23):
    MANUAL capture > FORCED (hotkey A) > AUTO candidate.

A job carries its exact captured frame (results belong to the captured
snapshot, spec §15) and every metadata field needed to reconstruct the run.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# job sources, in scheduling-priority order (spec §23)
PRIORITY_ORDER = {"MANUAL": 0, "FORCED": 1, "AUTO": 2}

# job lifecycle (spec §6)
QUEUED = "QUEUED"
CELLPOSE = "CELLPOSE"
FEATURES = "FEATURES"
COMPLETE = "COMPLETE"
FAILED = "FAILED"
CANCELLED = "CANCELLED"


@dataclass
class AnalysisJob:
    """One captured field travelling through the analysis pipeline."""

    job_id: int
    priority: str                      # MANUAL | FORCED | AUTO
    frame: np.ndarray                  # exact captured snapshot (results belong to it)
    frame_idx: int
    t_capture: float                   # session clock
    reason: str = ""
    cumulative_x: float = 0.0
    cumulative_y: float = 0.0
    cumulative_total: float = 0.0
    screen_result: str = ""
    screen_occupancy: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0
    status: str = QUEUED
    worker_id: Optional[str] = None
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    error: Optional[str] = None
    human_label: Optional[str] = None  # M/T/N/U - stored SEPARATELY from prediction (§30)
    feature_mode: str = "live"
    # results (filled by the CPU feature stage)
    labels: Optional[np.ndarray] = None
    rows: Optional[List[Dict]] = None
    features: Optional[Dict] = None
    score: Optional[float] = None
    score_components: Optional[Dict] = None
    raw_class: Optional[str] = None
    smoothed_score: Optional[float] = None
    smoothed_class: Optional[str] = None
    stale: bool = False
    stale_reason: str = ""
    overlay_layer: Optional[np.ndarray] = None
    id_layer: Optional[np.ndarray] = None
    cellpose_ms: Optional[float] = None
    features_ms: Optional[float] = None
    merge_analysis_ran: bool = False
    first_inference: bool = False
    queue_wait_ms: Optional[float] = None

    @property
    def runtime_sec(self) -> float:
        if self.t_start is None:
            return 0.0
        return (self.t_end or time.perf_counter()) - self.t_start

    @property
    def latency_ms(self) -> float:
        if self.t_start is None or self.t_end is None:
            return 0.0
        return (self.t_end - self.t_start) * 1000.0

    def is_stale(self, current_net_x: float, current_net_y: float, now: float,
                 stale_displacement: float, stale_seconds: float) -> bool:
        """Freshness for AUTO results vs the CURRENT microscope position.

        NET displacement (signed sums) - jitter averages out over the analysis
        latency while real stage movement accumulates. MANUAL jobs are never
        stale in this sense: their results belong to their own snapshot (§15).
        """
        moved = ((current_net_x - self.cumulative_x) ** 2 +
                 (current_net_y - self.cumulative_y) ** 2) ** 0.5 > stale_displacement
        old = (now - self.t_capture) > stale_seconds
        return bool(moved or old)


class JobManager:
    """Thread-safe bounded job queues + status registry."""

    def __init__(self, max_manual_queue: int = 20, completed_history: int = 40):
        self.max_manual_queue = int(max_manual_queue)
        self._manual: Dict[str, deque] = {
            "MANUAL": deque(), "FORCED": deque()
        }
        self._auto_pending: Optional[AnalysisJob] = None
        self._lock = threading.Lock()
        self._next_id = 1
        self._running = 0
        self.completed: deque = deque(maxlen=completed_history)
        self.all_jobs: Dict[int, AnalysisJob] = {}
        self.superseded_auto = 0
        self.refused_manual = 0

    # ------------------------------------------------------------------
    def submit(
        self,
        priority: str,
        frame: np.ndarray,
        frame_idx: int,
        t_capture: float,
        **meta,
    ) -> Optional[AnalysisJob]:
        """Enqueue a job. Returns the job, or None when refused (queue full)."""
        with self._lock:
            job = AnalysisJob(
                job_id=self._next_id, priority=priority, frame=frame,
                frame_idx=frame_idx, t_capture=t_capture, status=QUEUED, **meta,
            )
            if priority == "AUTO":
                if self._auto_pending is not None:
                    self.superseded_auto += 1
                self._auto_pending = job
            else:
                q = self._manual[priority]
                if len(q) >= self.max_manual_queue:
                    self.refused_manual += 1
                    return None  # REFUSED loudly - never silently discarded (§12)
                q.append(job)
            self.all_jobs[job.job_id] = job
            self._next_id += 1
            return job

    def take_next(self) -> Optional[AnalysisJob]:
        """Highest-priority waiting job (MANUAL > FORCED > AUTO latest)."""
        with self._lock:
            for prio in ("MANUAL", "FORCED"):
                if self._manual[prio]:
                    self._running += 1
                    return self._manual[prio].popleft()
            job, self._auto_pending = self._auto_pending, None
            if job is not None:
                self._running += 1
            return job

    def has_work(self) -> bool:
        with self._lock:
            return (self._auto_pending is not None
                    or any(self._manual[p] for p in self._manual)
                    or self._running > 0)

    def pending_count(self) -> int:
        """Jobs queued OR currently inside the pipeline (drain condition)."""
        with self._lock:
            queued = sum(len(q) for q in self._manual.values())
            return queued + (1 if self._auto_pending is not None else 0) + self._running

    # ------------------------------------------------------------------
    def mark(self, job: AnalysisJob, status: str, worker_id: Optional[str] = None) -> None:
        with self._lock:
            job.status = status
            if worker_id:
                job.worker_id = worker_id
            if status in (COMPLETE, FAILED, CANCELLED):
                self._running = max(0, self._running - 1)
            if status == COMPLETE:
                self.completed.append(job)

    def set_human_label(self, job_id: int, label: str) -> Optional[AnalysisJob]:
        with self._lock:
            job = self.all_jobs.get(job_id)
            if job is not None:
                job.human_label = label
            return job

    # ------------------------------------------------------------------
    def snapshot_jobs(self, max_rows: int = 12) -> List[Dict]:
        """GUI-safe view: newest jobs first (processing + queued + completed)."""
        with self._lock:
            jobs = sorted(self.all_jobs.values(), key=lambda j: -j.job_id)
            rows = []
            for j in jobs[:max_rows]:
                rows.append({
                    "job_id": j.job_id, "priority": j.priority, "status": j.status,
                    "raw_class": j.raw_class, "score": j.score,
                    "human_label": j.human_label, "runtime": j.runtime_sec,
                    "error": j.error,
                })
            return rows

    def get_completed(self, job_id: int) -> Optional[AnalysisJob]:
        with self._lock:
            job = self.all_jobs.get(job_id)
            return job if (job is not None and job.status == COMPLETE) else None

    def stats(self) -> Dict:
        with self._lock:
            waiting = sum(len(q) for q in self._manual.values())
            return {
                "waiting_manual": waiting,
                "waiting_auto": self._auto_pending is not None,
                "superseded_auto": self.superseded_auto,
                "refused_manual": self.refused_manual,
                "total_jobs": self._next_id - 1,
            }
