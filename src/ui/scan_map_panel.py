"""Persistent scan-map panel (Phase 1).

Draws every analyzed field as its spatial FOOTPRINT rectangle on the
persistent scan-coordinate canvas - the map remembers all previously
analyzed fields even after the live camera moves away, for the whole
session until Reset Scan.

  * class colours: TOO_THICK red, MONOLAYER green, TOO_THIN blue,
    UNCERTAIN amber (visualization only, never logic);
  * monolayer footprints get a slightly stronger fill so the (future)
    connected band is already visible when several green fields neighbour;
  * human-labeled fields get a white corner tick;
  * the current live scan position is a cyan crosshair.

The panel is independent of the live camera viewport. Auto-fits its
transform to the data, so negative coordinates and any sweep direction
work. Zoom/scale independence and band-boundary extraction are later
phases; this widget only proves placement + persistence.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from src.live.scan_map import CLASS_COLORS, FieldFootprint


class ScanMapPanel(QWidget):
    """Canvas of analyzed-field footprints in scan coordinates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fields: List[FieldFootprint] = []
        self._current: Optional[Tuple[float, float]] = None
        self.setMinimumHeight(220)

    def set_data(self, fields: List[FieldFootprint],
                 current_pos: Optional[Tuple[float, float]]) -> None:
        self._fields = fields
        self._current = current_pos
        self.update()

    def paintEvent(self, ev):  # noqa: N802 (Qt naming)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(16, 16, 16))
        p.setPen(QPen(QColor(120, 120, 120), 1))
        p.drawText(6, 14, "SCAN MAP - analyzed field footprints "
                         "(persists until Reset Scan)")

        w, h = self.width() - 16, self.height() - 40
        if w <= 10 or h <= 10:
            p.end()
            return

        extra = [self._current] if self._current else []
        fields = self._fields
        if fields:
            xs0 = [f.x - f.w / 2 for f in fields]
            ys0 = [f.y - f.h / 2 for f in fields]
            xs1 = [f.x + f.w / 2 for f in fields]
            ys1 = [f.y + f.h / 2 for f in fields]
            for px, py in extra:
                xs0.append(px); ys0.append(py); xs1.append(px); ys1.append(py)
            min_x, min_y = min(xs0), min(ys0)
            span_x = max(1.0, max(xs1) - min_x)
            span_y = max(1.0, max(ys1) - min_y)
            scale = min(w / span_x, h / span_y)

            def to_px(x, y):
                return (8 + (x - min_x) * scale,
                        30 + (y - min_y) * scale)

            # footprints (rectangles, not dots - neighbouring fields overlap)
            for f in fields:
                colour = QColor(CLASS_COLORS.get(f.raw_class, "#888888"))
                alpha = 235 if f.raw_class == "MONOLAYER" else 130
                colour.setAlpha(alpha)
                x0, y0 = to_px(f.x - f.w / 2, f.y - f.h / 2)
                x1, y1 = to_px(f.x + f.w / 2, f.y + f.h / 2)
                p.fillRect(int(x0), int(y0),
                           max(2, int(x1 - x0)), max(2, int(y1 - y0)), colour)
                p.setPen(QPen(QColor(colour.red(), colour.green(), colour.blue(), 255),
                              2 if f.raw_class == "MONOLAYER" else 1))
                p.drawRect(int(x0), int(y0),
                           max(2, int(x1 - x0)), max(2, int(y1 - y0)))
                if f.human_label:  # human-labeled field: white corner tick
                    p.setPen(QPen(QColor(255, 255, 255), 2))
                    p.drawLine(int(x0) + 2, int(y0) + 6, int(x0) + 6, int(y0) + 2)
                p.setPen(QPen(QColor(200, 200, 200), 1))

        # current live scan position (crosshair)
        if self._current:
            if fields:
                cx0, cy0 = to_px(self._current[0], self._current[1])
            else:
                cx0, cy0 = 8 + w / 2, 30 + h / 2
            p.setPen(QPen(QColor(80, 200, 255), 2))
            p.drawLine(int(cx0) - 6, int(cy0), int(cx0) + 6, int(cy0))
            p.drawLine(int(cx0), int(cy0) - 6, int(cx0), int(cy0) + 6)

        if not fields:
            p.setPen(QPen(QColor(140, 140, 140), 1))
            p.drawText(8, self.height() // 2,
                       "analyze fields (CAPTURE or Auto) to build the survey map")
        p.end()
