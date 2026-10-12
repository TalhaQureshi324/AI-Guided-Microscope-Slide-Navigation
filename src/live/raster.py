"""Raster / multi-sweep navigation assistant (Phase 12).

Guides manual scanning along a boustrophedon (lawnmower) pattern:

    Sweep 1  LEFT_TO_RIGHT   row 1
    Sweep 2  RIGHT_TO_LEFT   row 2   (after a vertical STEP DOWN)
    Sweep 3  LEFT_TO_RIGHT   row 3
    ...

Two phases per row:
  ROW_SWEEP   moving along the row; horizontal overlap with the previous
              field in THIS row must stay inside the target band;
  STEP_DOWN   the operator has turned: vertical overlap with the PREVIOUS
              ROW must come inside the vertical target band; when it does,
              a capture flips the direction and returns to ROW_SWEEP.

This is a HUMAN navigation assistant - motors are never controlled (spec
Phase 12). Every rule reads its thresholds from configuration. All state
lives here; the map/mosaic/semantic layers are never reset by a new sweep
(Phase 6 rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

L2R = "LEFT_TO_RIGHT"
R2L = "RIGHT_TO_LEFT"

ROW_SWEEP = "ROW_SWEEP"
STEP_DOWN = "STEP_DOWN"


@dataclass
class RasterConfig:
    target_overlap_min: float = 0.20     # horizontal, within a row
    target_overlap_max: float = 0.30
    vertical_target_min: float = 0.15    # vertical, when stepping down
    vertical_target_max: float = 0.30
    auto_alternate: bool = True          # next sweep flips direction


class RasterNavigator:
    """Stateful raster navigation assistant for manual scanning."""

    def __init__(self, cfg: Optional[Dict] = None) -> None:
        cfg = cfg or {}
        r = cfg.get("raster", {})
        self.cfg = r if r else {}
        self.target_min = float(self.cfg.get("target_overlap_min", 0.20))
        self.target_max = float(self.cfg.get("target_overlap_max", 0.30))
        self.v_min = float(self.cfg.get("vertical_target_min", 0.15))
        self.v_max = float(self.cfg.get("vertical_target_max", 0.30))
        self.auto_alternate = bool(self.cfg.get("auto_alternate", True))

        self.direction: str = L2R
        self.phase: str = ROW_SWEEP
        self.sweep_id: int = 0
        self.row: int = 0
        self.rows_completed: int = 0
        self._step_ready = False

    # ------------------------------------------------------------------
    def start_sweep(self, direction: Optional[str] = None) -> Dict:
        """Begin a new sweep/row. Direction alternates unless explicit."""
        if direction is None:
            if not self.auto_alternate:
                direction = self.direction
            else:
                # spec example: sweep 1 = LEFT_TO_RIGHT, sweep 2 = RIGHT_TO_LEFT
                # (direction is chosen for the row ABOUT to start)
                direction = L2R if (self.sweep_id + 1) % 2 == 1 else R2L
        self.direction = direction
        self.sweep_id += 1
        self.row = self.sweep_id
        self.phase = ROW_SWEEP
        self._step_ready = False
        return {"sweep_id": self.sweep_id, "row": self.row,
                "direction": self.direction}

    # ------------------------------------------------------------------
    def guidance(self, horizontal_overlap: float, vertical_overlap: float,
                 moving_vertical: bool) -> Dict:
        """Operator guidance for the current live frame (spec §3/§4/§7)."""
        if self.phase == ROW_SWEEP:
            if horizontal_overlap > self.target_max:
                hint = ("ROW OVERLAP TOO HIGH - MOVE "
                        + ("RIGHT" if self.direction == L2R else "LEFT"))
            elif horizontal_overlap < self.target_min:
                hint = ("GAP RISK - MOVE "
                        + ("LEFT" if self.direction == L2R else "RIGHT"))
            else:
                hint = "CAPTURE READY - new content %.0f%%" % (
                    (1.0 - horizontal_overlap) * 100)
            return {
                "phase": self.phase, "direction": self.direction,
                "hint": hint, "h_overlap": horizontal_overlap,
                "v_overlap": vertical_overlap,
                "progress": horizontal_overlap,
            }

        # STEP_DOWN: vertical overlap vs the previous row
        if vertical_overlap > self.v_max:
            hint = "ROW OVERLAP TOO HIGH - MOVE FARTHER DOWN"
        elif vertical_overlap < self.v_min:
            hint = "VERTICAL GAP RISK - MOVE BACK UP"
        else:
            hint = "CAPTURE READY (new row) - vertical overlap %.0f%%" % (
                vertical_overlap * 100)
        return {
            "phase": self.phase, "direction": self.direction,
            "hint": hint, "h_overlap": horizontal_overlap,
            "v_overlap": vertical_overlap, "progress": vertical_overlap,
        }

    # ------------------------------------------------------------------
    def note_vertical_step(self, vertical_overlap: float) -> bool:
        """After a capture in STEP_DOWN: True when the new row is placed
        inside the vertical target band -> flip direction, resume ROW."""
        if self.v_min <= vertical_overlap <= self.v_max:
            self.phase = ROW_SWEEP
            self.rows_completed += 1
            if self.auto_alternate:
                self.direction = R2L if self.direction == L2R else L2R
            return True
        return False
