"""Live perception controller (UI-independent, thread-agnostic).

Implements the live-camera phase logic (spec sections 8-16, 21, 43):
  * motion states: MOVING / STATIONARY (+ settle / blur guard);
  * field-change thinking: a new detailed analysis is requested when the
    accumulated displacement since the last analysis exceeds a threshold AND
    the view has settled (stopped moving) AND a sharp frame is available -
    never just because time passed (a rate limit + heartbeat remain as
    safety nets);
  * latest-frame policy helper: pending-field slot semantics live in the
    workers; this controller only decides WHEN a frame qualifies;
  * stale-result rule: a finished result must not describe the current field
    if the microscope moved more than ``result_stale_displacement`` px since
    the result's frame was captured, or the result is older than
    ``result_stale_seconds``.

All thresholds come from configs/live.yaml (nothing experimental hard-coded).
"""

from __future__ import annotations

from dataclasses import dataclass, field

MOVING = "MOVING"
STATIONARY = "STATIONARY"
SETTLING = "SETTLING"


@dataclass
class FrameDecision:
    """Per-frame outcome computed by :class:`LiveFieldController`."""

    motion_state: str = MOVING
    request_field: bool = False
    request_reason: str = ""
    cumulative_since_analysis: float = 0.0
    sharp_enough: bool = True
    settled_frames: int = 0


@dataclass
class FieldRequest:
    """A snapshot handed to the analysis worker (latest-frame slot).

    ``frame`` carries the exact captured image so the worker analyses the
    field that was requested - never whatever the camera shows seconds later
    (spec §15/§16). One slot, one frame: no backlog by construction.
    """

    frame_idx: int
    t_capture: float
    cumulative_x: float
    cumulative_y: float
    cumulative_total: float
    reason: str = ""
    superseded_count: int = field(default=0)
    screen_result: str = ""
    sharpness: float = 0.0
    brightness: float = 0.0
    frame: object = field(default=None, repr=False)

    def is_stale(self, current_net_x: float, current_net_y: float, now: float,
                 stale_displacement: float, stale_seconds: float) -> bool:
        """True when the microscope's NET position left this field's position.

        Net (signed dx/dy sums) is the right metric: registration jitter
        averages out over the analysis latency, while real stage movement
        accumulates. Path length would wrongly void results during
        stationary periods (jitter adds up as positive path length).
        """
        moved = ((current_net_x - self.cumulative_x) ** 2 +
                 (current_net_y - self.cumulative_y) ** 2) ** 0.5 > stale_displacement
        old = (now - self.t_capture) > stale_seconds
        return bool(moved or old)


class LiveFieldController:
    """Stateful per-frame decider for the live pipeline."""

    def __init__(
        self,
        moving_disp_thresh_px: float = 8.0,
        stationary_frames: int = 6,
        settle_frames: int = 4,
        motion_threshold_px: float = 240.0,
        min_interval_sec: float = 1.0,
        heartbeat_sec: float = 20.0,
        blur_guard: float = 5.0,
    ) -> None:
        self.moving_disp_thresh_px = float(moving_disp_thresh_px)
        self.stationary_frames = int(stationary_frames)
        self.settle_frames = int(settle_frames)
        self.motion_threshold_px = float(motion_threshold_px)
        self.min_interval_sec = float(min_interval_sec)
        self.heartbeat_sec = float(heartbeat_sec)
        self.blur_guard = float(blur_guard)

        self._still = 0
        self._cum_total = 0.0
        self._cum_x = 0.0
        self._cum_y = 0.0
        self._cum_since_analysis = 0.0
        self._last_analysis_t = None
        self.analysis_count = 0

    # ------------------------------------------------------------------
    def update(self, frame_idx: int, t_sec: float, dx: float, dy: float,
               disp: float, sharpness: float, force: bool = False) -> FrameDecision:
        """Feed one frame (from the capture loop); decide field request."""
        self._cum_total += disp
        self._cum_x += dx
        self._cum_y += dy
        self._cum_since_analysis += disp

        moving_now = disp > self.moving_disp_thresh_px
        if moving_now:
            self._still = 0
        else:
            self._still += 1

        if moving_now:
            state = MOVING
        elif self._still < self.stationary_frames:
            state = SETTLING  # motion just stopped - frames may still be smeared
        else:
            state = STATIONARY

        settled = self._still - self.stationary_frames  # frames past the stop point
        sharp = sharpness >= self.blur_guard
        since = (t_sec - self._last_analysis_t) if self._last_analysis_t is not None else float("inf")

        request = False
        reason = ""
        if state == STATIONARY and settled >= self.settle_frames and sharp:
            if force:
                request, reason = True, "MANUAL"
            elif self._last_analysis_t is None:
                request, reason = True, "FIRST_FIELD"
            elif self._cum_since_analysis >= self.motion_threshold_px and since >= self.min_interval_sec:
                request, reason = True, "NEW_FIELD"
            elif since >= self.heartbeat_sec:
                request, reason = True, "HEARTBEAT"

        if request:
            self._cum_since_analysis = 0.0
            self._last_analysis_t = t_sec
            self.analysis_count += 1

        return FrameDecision(
            motion_state=state,
            request_field=request,
            request_reason=reason,
            cumulative_since_analysis=self._cum_since_analysis,
            sharp_enough=sharp,
            settled_frames=max(0, settled),
        )

    # ------------------------------------------------------------------
    @property
    def cumulative(self) -> tuple[float, float, float]:
        """(cumulative_dx, cumulative_dy, cumulative_path_length) estimates."""
        return self._cum_x, self._cum_y, self._cum_total

    def reset_session(self) -> None:
        """Zero the cumulative displacement estimates (hotkey R)."""
        self._cum_total = self._cum_x = self._cum_y = 0.0
        self._cum_since_analysis = 0.0
