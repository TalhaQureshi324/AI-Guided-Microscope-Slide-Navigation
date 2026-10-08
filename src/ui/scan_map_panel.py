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

from src.live.scan_map import CLASS_COLORS, FieldFootprint, point_in_any, union_rects


class ScanMapPanel(QWidget):
    """Canvas of analyzed-field footprints in scan coordinates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fields: List[FieldFootprint] = []
        self._mono_rects: List[Tuple[float, float, float, float]] = []
        self._boundary: List[List[Tuple[float, float]]] = []
        self._boundary_version: Optional[int] = None
        self._current: Optional[Tuple[float, float]] = None
        self.setMinimumHeight(220)

    def set_data(self, fields: List[FieldFootprint],
                 monolayer_rects: List[Tuple[float, float, float, float]],
                 current_pos: Optional[Tuple[float, float]],
                 boundary_version: Optional[int] = None,
                 boundary: Optional[List[List[Tuple[float, float]]]] = None) -> None:
        """monolayer_rects = ACTIVE monolayer footprints (x0,y0,x1,y1) in scan
        coordinates; boundary = outer contour(s) of the accumulated monolayer
        region in scan coordinates (Phase 4), tagged with its map version so
        it is only refreshed when the survey changed."""
        self._fields = fields
        self._mono_rects = list(monolayer_rects)
        if boundary is not None and boundary_version != self._boundary_version:
            self._boundary = boundary
            self._boundary_version = boundary_version
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
        mono = self._mono_rects
        if fields or mono:
            xs0 = [f.x - f.w / 2 for f in fields] + [r[0] for r in mono]
            ys0 = [f.y - f.h / 2 for f in fields] + [r[1] for r in mono]
            xs1 = [f.x + f.w / 2 for f in fields] + [r[2] for r in mono]
            ys1 = [f.y + f.h / 2 for f in fields] + [r[3] for r in mono]
            for contour in self._boundary:
                for pt in contour:
                    xs0.append(pt[0]); ys0.append(pt[1])
                    xs1.append(pt[0]); ys1.append(pt[1])
            for px, py in extra:
                xs0.append(px); ys0.append(py); xs1.append(px); ys1.append(py)
            min_x, min_y = min(xs0), min(ys0)
            span_x = max(1.0, max(xs1) - min_x)
            span_y = max(1.0, max(ys1) - min_y)
            scale = min(w / span_x, h / span_y)

            def to_px(x, y):
                return (8 + (x - min_x) * scale,
                        30 + (y - min_y) * scale)

            # ---- persistent monolayer layer: merged union, one boundary ----
            if mono:
                union = union_rects([(r[0], r[1], r[2], r[3]) for r in mono])
                fill = QColor("#2e7d32"); fill.setAlpha(120)
                for ux0, uy0, ux1, uy1 in union:
                    ux0p, uy0p = to_px(ux0, uy0)
                    ux1p, uy1p = to_px(ux1, uy1)
                    p.fillRect(int(ux0p), int(uy0p),
                               max(2, int(ux1p - ux0p)), max(2, int(uy1p - uy0p)), fill)
                # outer boundary only: sample just outside each candidate edge
                p.setPen(QPen(QColor(40, 167, 69), 2))
                eps = max(span_x, span_y) * 0.004
                for ux0, uy0, ux1, uy1 in union:
                    step_x = max(eps, (ux1 - ux0) / 24.0)
                    step_y = max(eps, (uy1 - uy0) / 24.0)
                    xs = [ux0 + i * step_x for i in range(25)] + [ux1]
                    ys = [uy0 + i * step_y for i in range(25)] + [uy1]
                    for x in xs:  # top/bottom edges
                        for (ex, ey) in ((x, uy0 - eps), (x, uy1 + eps)):
                            if not point_in_any(ex, ey, union):
                                a = to_px(ex, min(max(uy0, ey - eps), uy1))
                                b = to_px(ex, max(min(uy1, ey + eps), uy0))
                                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
                    for y in ys:  # left/right edges
                        for (ex, ey) in ((ux0 - eps, y), (ux1 + eps, y)):
                            if not point_in_any(ex, ey, union):
                                a = to_px(min(max(ux0, ex - eps), ux1), ey)
                                b = to_px(max(min(ux1, ex + eps), ux0), ey)
                                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))

            # ---- per-field footprints (thin outlines, human ticks) ----
            for f in fields:
                colour = QColor(CLASS_COLORS.get(f.raw_class, "#888888"))
                alpha = 90 if f.raw_class == "MONOLAYER" else 130
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

        # ---- Phase 4: continuous monolayer region boundary (green line) ----
        # drawn with the SAME to_px transform as the footprints (already
        # includes the boundary in the fit above)
        if self._boundary and fields:
            p.setPen(QPen(QColor(30, 200, 90), 3))
            for contour in self._boundary:
                pts_px = [to_px(x, y) for x, y in contour]
                if len(pts_px) >= 2:
                    for (ax, ay), (bx, by) in zip(pts_px, pts_px[1:] + pts_px[:1]):
                        p.drawLine(int(ax), int(ay), int(bx), int(by))

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
