"""Capture-order spatial commit gate (2026-10-14 spatial repair, R0/R18).

Analysis MAY run in parallel and finish out of order - but the trusted
spatial state (registration chain -> global pose -> scan map -> mosaic ->
semantic layer) must be committed in CAPTURE order, never in
worker-completion order. Evidence from the 2026-10-14 live session: job #1
(captured first) reached the feature stage LAST and was registered against
job #2's state, stacking two different fields at (0, 0).

Usage (feature workers):

    gate.wait_turn(job.job_id, jobmgr)   # blocks until it's this job's turn
    try:
        ...register -> map -> mosaic...
    finally:
        gate.finish_turn(job.job_id)

A job's turn arrives when every LOWER job id is either committed here or in
a terminal state it can never leave (FAILED / CANCELLED / refused). A
timeout fallback guarantees the worker pool can never deadlock on a stuck
predecessor - the commit then proceeds out of order and is logged.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Set

# terminal job states that will never reach the spatial section
_TERMINAL_STATUSES = {"FAILED", "CANCELLED"}


class SpatialOrderGate:
    """Serialize spatial commits by job id (capture order)."""

    def __init__(self, timeout_sec: float = 15.0) -> None:
        self._cv = threading.Condition()
        self._committed: Set[int] = set()
        self._terminal: Set[int] = set()
        self._timeout = float(timeout_sec)
        self.out_of_order_commits = 0

    # ------------------------------------------------------------------
    def wait_turn(self, job_id: int, jobmgr=None) -> bool:
        """Block until all lower jobs are committed/terminal. Returns False
        when the timeout fallback fired (commit is out of order)."""
        start = time.perf_counter()
        timed_out = False
        with self._cv:
            while True:
                blockers = self._blockers(job_id, jobmgr)
                if not blockers:
                    break
                remaining = self._timeout - (time.perf_counter() - start)
                if remaining <= 0:
                    timed_out = True
                    self.out_of_order_commits += 1
                    break
                self._cv.wait(timeout=min(0.5, remaining))
        return not timed_out

    def finish_turn(self, job_id: int) -> None:
        with self._cv:
            self._committed.add(job_id)
            self._cv.notify_all()

    def mark_never(self, job_id: int) -> None:
        """Job will never commit spatially (feature-stage failure)."""
        with self._cv:
            self._terminal.add(job_id)
            self._cv.notify_all()

    def reset(self) -> None:
        with self._cv:
            self._committed.clear()
            self._terminal.clear()
            self.out_of_order_commits = 0

    # ------------------------------------------------------------------
    def _blockers(self, job_id: int, jobmgr) -> list:
        """Lower job ids that are neither committed nor terminal."""
        blockers = []
        if jobmgr is None:
            return blockers
        for other in jobmgr.all_jobs_snapshot():
            other_id = other.job_id
            if other_id >= job_id or other_id in self._committed \
                    or other_id in self._terminal:
                continue
            if other.status in _TERMINAL_STATUSES:
                self._terminal.add(other_id)
            elif other.status == "COMPLETE" and other_id not in self._committed:
                # COMPLETE but not yet committed: it is either being committed
                # right now (brief) or is next in line
                blockers.append(other_id)
            else:
                blockers.append(other_id)
        return blockers
