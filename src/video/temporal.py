"""Temporal smoothing across ACCEPTED keyframes (spec sections 28-29).

Smoothing deliberately operates on accepted keyframes only - meaningful,
non-redundant microscope fields - never on raw 30+ FPS frames, so a long
stationary period cannot dominate the vote.

Two independent mechanisms, both configurable:
  * exponential moving average of the raw Monolayer Score (``alpha``);
  * majority vote over the last ``window`` raw classifications
    (tie -> follow the EMA score's threshold mapping).

Both raw and smoothed values are always stored (spec: never hide the original).
"""

from __future__ import annotations

from collections import deque
from typing import Tuple

from .monolayer import MONOLAYER, UNCERTAIN


class TemporalSmoother:
    def __init__(self, alpha: float = 0.4, window: int = 5, monolayer_score_min: float = 0.60):
        self.alpha = float(alpha)
        self.window = int(window)
        self.monolayer_score_min = float(monolayer_score_min)
        self._ema: float | None = None
        self._recent: deque = deque(maxlen=self.window)

    def update(self, raw_score: float, raw_class: str) -> Tuple[float, str]:
        self._ema = raw_score if self._ema is None else (
            self.alpha * raw_score + (1.0 - self.alpha) * self._ema
        )
        self._recent.append(raw_class)

        # majority vote over the recent window
        counts: dict = {}
        for cls in self._recent:
            counts[cls] = counts.get(cls, 0) + 1
        best = max(counts.values())
        winners = [c for c, n in counts.items() if n == best]
        if len(winners) == 1:
            smoothed = winners[0]
        else:
            smoothed = MONOLAYER if self._ema >= self.monolayer_score_min else UNCERTAIN
        return float(self._ema), smoothed
