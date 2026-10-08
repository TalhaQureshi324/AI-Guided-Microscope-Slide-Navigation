"""Monolayer navigation state machine (Phase 2).

Describes whether the live scan is entering, inside, or leaving the
monolayer band - with HYSTERESIS so the GUI never flickers:

    enter threshold  (0.65)  >  exit threshold (0.45)
    + N consecutive compatible fields on each side

Input: ONLY fresh AUTO results and their existing smoothed Monolayer Score
(the multi-feature score - never cell count alone; manual captures and stale
results never move it, they describe their own snapshots).

States
------
OUTSIDE_MONOLAYER    default.
ENTERING_MONOLAYER   scores crossed ABOVE the enter threshold and are being
                     confirmed (``enter_consecutive_fields`` compatible fields).
IN_MONOLAYER         confirmed inside. Stays through the hysteresis band
                     (exit < score < enter) - that quiet zone is what
                     prevents flicker.
LEAVING_MONOLAYER    ``exit_consecutive_fields`` compatible scores fell to/below
                     the exit threshold. Recovers back to IN if the score
                     rises above the enter threshold again; otherwise falls
                     to OUTSIDE after the same consecutive count.

Every transition is recorded (timestamp, frame, scan position, score) so the
monolayer boundaries can later be reconstructed on the scan map (Phase 4+).
All thresholds are experimental prototype values, configurable in
configs/live.yaml - never medical constants.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

OUTSIDE = "OUTSIDE_MONOLAYER"
ENTERING = "ENTERING_MONOLAYER"
IN = "IN_MONOLAYER"
LEAVING = "LEAVING_MONOLAYER"


@dataclass
class NavTransition:
    t: float
    frame_idx: int
    x: float
    y: float
    from_state: str
    to_state: str
    score: float


class NavigationStateMachine:
    """Thread-safe hysteresis machine over the smoothed Monolayer Score."""

    def __init__(
        self,
        enter_score: float = 0.65,
        exit_score: float = 0.45,
        enter_consecutive_fields: int = 2,
        exit_consecutive_fields: int = 2,
    ) -> None:
        assert enter_score > exit_score, "hysteresis requires enter > exit"
        self.enter_score = float(enter_score)
        self.exit_score = float(exit_score)
        self.enter_n = max(1, int(enter_consecutive_fields))
        self.exit_n = max(1, int(exit_consecutive_fields))

        self.state: str = OUTSIDE
        self.transitions: List[NavTransition] = []
        self.last_score: Optional[float] = None
        self.good_streak = 0      # consecutive scores >= enter_score
        self.bad_streak = 0       # consecutive scores <= exit_score
        self.entering_count = 0   # compatible fields since ENTERING began
        self.leaving_count = 0    # bad fields since LEAVING began
        # RLock: update() holds the lock and calls view() (which re-acquires);
        # a plain Lock self-deadlocks on the first field (found by test).
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self.state = OUTSIDE
            self.transitions.clear()
            self.last_score = None
            self.good_streak = self.bad_streak = 0
            self.entering_count = self.leaving_count = 0

    def _transition(self, to_state: str, t: float, frame_idx: int,
                    x: float, y: float, score: float) -> None:
        self.transitions.append(NavTransition(
            t=t, frame_idx=frame_idx, x=x, y=y,
            from_state=self.state, to_state=to_state, score=score,
        ))
        self.state = to_state

    # ------------------------------------------------------------------
    def update(self, score: float, t: float = 0.0, frame_idx: int = -1,
               x: float = 0.0, y: float = 0.0) -> Dict:
        """Feed one fresh smoothed score; returns the machine's public view."""
        with self._lock:
            self.last_score = float(score)
            good = score >= self.enter_score
            bad = score <= self.exit_score
            self.good_streak = self.good_streak + 1 if good else 0
            self.bad_streak = self.bad_streak + 1 if bad else 0

            s = self.state
            if s == OUTSIDE:
                if self.good_streak >= self.enter_n:
                    self._transition(ENTERING, t, frame_idx, x, y, score)
                    self.entering_count = 0
            elif s == ENTERING:
                if bad:
                    self._transition(OUTSIDE, t, frame_idx, x, y, score)  # not confirmed
                elif good:
                    self.entering_count += 1
                    if self.entering_count + 1 >= self.enter_n:  # +1: the crossing field
                        self._transition(IN, t, frame_idx, x, y, score)
            elif s == IN:
                if bad:
                    self.leaving_count += 1
                    if self.leaving_count >= self.exit_n:
                        self._transition(LEAVING, t, frame_idx, x, y, score)
                else:
                    self.leaving_count = 0
            elif s == LEAVING:
                if good:
                    self.leaving_count = 0
                    self._transition(IN, t, frame_idx, x, y, score)  # recovered
                elif bad:
                    self.leaving_count += 1
                    if self.leaving_count >= self.exit_n:
                        self._transition(OUTSIDE, t, frame_idx, x, y, score)
                # mid-band keeps LEAVING (hysteresis: exit is "likely" until disproven)

            return self.view()

    def view(self) -> Dict:
        with self._lock:
            return {
                "state": self.state,
                "score": self.last_score,
                "good_streak": self.good_streak,
                "bad_streak": self.bad_streak,
                "transitions": len(self.transitions),
            }

    def transitions_as_dicts(self) -> List[Dict]:
        with self._lock:
            return [
                {
                    "t": tr.t, "frame_idx": tr.frame_idx,
                    "x": round(tr.x, 1), "y": round(tr.y, 1),
                    "from_state": tr.from_state, "to_state": tr.to_state,
                    "score": round(tr.score, 4),
                }
                for tr in self.transitions
            ]
