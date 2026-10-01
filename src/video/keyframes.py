"""Motion-aware keyframe selection (spec sections 6-9, 29).

A raw frame becomes an analyzed **keyframe** only when
  1. it is not grossly blurred, and
  2. the microscope has seen enough *new* content since the last accepted
     field (accumulated displacement >= ``motion_threshold_px``), and
  3. at least ``min_interval_sec`` passed since the last acceptance (a
     prototype-era compute-budget guard, so a fast continuous sweep cannot
     flood the expensive analysis stage), or
  4. ``heartbeat_sec`` elapsed regardless of movement (guarantees at least
     occasional analysis during long stationary periods and records
     illumination drift).

Accumulated displacement is the *path length* of per-frame phase-correlation
shifts, reset after every acceptance. Stationary periods therefore produce no
keyframes (except heartbeats) - hundreds of near-identical frames collapse to
at most one representative field per ``heartbeat_sec``.

Frame status vocabulary (stored in results.csv for EVERY raw frame):
  ACCEPTED_FIRST, ACCEPTED_NEW_FIELD, ACCEPTED_HEARTBEAT,
  SKIPPED_STATIONARY, SKIPPED_RATE_LIMIT, SKIPPED_BLUR
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .motion import MotionEstimator
from .quality import is_blurry

ACCEPTED_FIRST = "ACCEPTED_FIRST"
ACCEPTED_NEW_FIELD = "ACCEPTED_NEW_FIELD"
ACCEPTED_HEARTBEAT = "ACCEPTED_HEARTBEAT"
SKIPPED_STATIONARY = "SKIPPED_STATIONARY"
SKIPPED_RATE_LIMIT = "SKIPPED_RATE_LIMIT"
SKIPPED_BLUR = "SKIPPED_BLUR"


@dataclass
class KeyframeDecision:
    accepted: bool
    status: str
    dx: float
    dy: float
    displacement_px: float
    cumulative_px: float
    sec_since_last_accept: float
    response: float


class KeyframeSelector:
    """Stateful per-frame accept/skip decider."""

    def __init__(
        self,
        motion_threshold_px: float,
        min_interval_sec: float,
        heartbeat_sec: float,
        blur_threshold: float,
        fps: float,
        motion_work_width: int = 480,
    ) -> None:
        self.motion_threshold_px = float(motion_threshold_px)
        self.min_interval_sec = float(min_interval_sec)
        self.heartbeat_sec = float(heartbeat_sec)
        self.blur_threshold = float(blur_threshold)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._motion = MotionEstimator(motion_work_width)

        self._prev_small: Optional[np.ndarray] = None
        self._cumulative = 0.0
        self._last_accept_t: Optional[float] = None
        self._last_t: Optional[float] = None
        self.accepted_count = 0

    def update(
        self,
        frame_idx: int,
        t_sec: float,
        quality: dict,
        small_gray: np.ndarray,
        full_width: int,
    ) -> KeyframeDecision:
        """Feed one raw frame; returns the accept/skip decision.

        The motion reference is advanced on every frame (blurry or not) so a
        blur gap never produces a bogus huge jump across the skipped frames.
        """
        self._last_t = t_sec
        dx = dy = disp = 0.0
        response = 1.0
        if self._prev_small is not None:
            dx, dy, response = self._motion.displacement(
                self._prev_small, small_gray, full_width
            )
            disp = float(np.hypot(dx, dy))
            self._cumulative += disp
        self._prev_small = small_gray

        sec_since = (
            (t_sec - self._last_accept_t) if self._last_accept_t is not None else float("inf")
        )
        blurry = is_blurry(quality, self.blur_threshold)

        if blurry:
            return self._decide(False, SKIPPED_BLUR, dx, dy, disp, sec_since, response)
        if self._last_accept_t is None:
            return self._decide(True, ACCEPTED_FIRST, dx, dy, disp, sec_since, response)
        if sec_since >= self.heartbeat_sec:
            return self._decide(True, ACCEPTED_HEARTBEAT, dx, dy, disp, sec_since, response)
        if self._cumulative >= self.motion_threshold_px:
            if sec_since >= self.min_interval_sec:
                return self._decide(
                    True, ACCEPTED_NEW_FIELD, dx, dy, disp, sec_since, response
                )
            return self._decide(False, SKIPPED_RATE_LIMIT, dx, dy, disp, sec_since, response)
        return self._decide(False, SKIPPED_STATIONARY, dx, dy, disp, sec_since, response)

    def _decide(
        self,
        accepted: bool,
        status: str,
        dx: float,
        dy: float,
        disp: float,
        sec_since: float,
        response: float,
    ) -> KeyframeDecision:
        if accepted:
            self._cumulative = 0.0
            self._last_accept_t = self._last_t
            self.accepted_count += 1
        return KeyframeDecision(
            accepted=accepted,
            status=status,
            dx=dx,
            dy=dy,
            displacement_px=disp,
            cumulative_px=self._cumulative,
            sec_since_last_accept=sec_since if sec_since != float("inf") else -1.0,
            response=response,
        )
