"""Persistent scan-map panel with an explicit view transform (Phase 5).

ALL geometry (field footprints, the merged monolayer layer, the boundary
contour, the live position crosshair) is stored in SCAN/WORLD coordinates.
Rendering passes through ONE explicit transform:

    widget_px = center + (fit_px(scan) - center) * user_zoom + pan

so zoom and pan are pure rendering - the stored monolayer geometry never
touches screen pixels, and every zoom level (25%/50%/100%/200%...) shows the
exact same region, aligned (spec Phase 5). Wheel zooms, left-drag pans,
Reset View restores the default fit; Reset Scan (R) remains the only action
that deletes the survey.
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
        self._sweeps: List[dict] = []
        self._sweep_trails: List[List[Tuple[float, float]]] = []
        self._has_content = False

        # explicit view transform state (Phase 5)
        self._fit: Optional[Tuple[float, float, float]] = None  # (min_x, min_y, scale)
        self._user_zoom: float = 1.0
        self._pan: Tuple[float, float] = (0.0, 0.0)  # widget px
        self._drag_last: Optional[Tuple[float, float]] = None
        self.setMinimumHeight(220)

    # -------------------------------------------------------------- view
    def reset_view(self) -> None:
        """Reset zoom/pan to the default fit - scan data untouched."""
        self._user_zoom = 1.0
        self._pan = (0.0, 0.0)
        self.update()

    def zoom_by(self, factor: float) -> None:
        """Zoom about the widget centre; factor > 1 magnifies."""
        self._set_zoom(self._user_zoom * factor)

    def _set_zoom(self, new_zoom: float) -> None:
        new_zoom = max(0.25, min(8.0, new_zoom))
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        ratio = new_zoom / max(1e-9, self._user_zoom)
        px, py = self._pan
        # keep the centre point stable while zooming
        # widget = centre + (fit - centre)*zoom + pan  (pan applied AFTER zoom)
        # => a zoom by `ratio` about the centre simply scales the pan too
        self._pan = (self._pan[0] * ratio, self._pan[1] * ratio)
        self._user_zoom = new_zoom
        self.update()

    def to_widget_px(self, x: float, y: float) -> Tuple[float, float]:
        """The single explicit transform: scan coordinates -> widget pixels."""
        fx, fy = self._fit_to_widget(x, y)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        return (cx + (fx - cx) * self._user_zoom + self._pan[0],
                cy + (fy - cy) * self._user_zoom + self._pan[1])

    def _fit_to_widget(self, x: float, y: float) -> Tuple[float, float]:
        if self._fit is None:
            return (x, y)
        mcx, mcy, scale = self._fit
        return (self.width() / 2.0 + (x - mcx) * scale,
                self.height() / 2.0 + (y - mcy) * scale)

    # ------------------------------------------------------------ events
    def wheelEvent(self, ev):  # noqa: N802 (Qt naming)
        delta = ev.angleDelta().y()
        if delta:
            self._set_zoom(self._user_zoom * (1.2 if delta > 0 else 1 / 1.2))

    def mousePressEvent(self, ev):  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_last = (ev.position().x(), ev.position().y())

    def mouseMoveEvent(self, ev):  # noqa: N802
        last = self._drag_last
        if last is not None and ev.buttons() & Qt.MouseButton.LeftButton:
            dx = ev.position().x() - last[0]
            dy = ev.position().y() - last[1]
            self._pan = (self._pan[0] + dx, self._pan[1] + dy)
            self._drag_last = (ev.position().x(), ev.position().y())
            self.update()

    def mouseReleaseEvent(self, ev):  # noqa: N802
        self._drag_last = None

    # -------------------------------------------------------------- data
    def set_data(self, fields: List[FieldFootprint],
                 monolayer_rects: List[Tuple[float, float, float, float]],
                 current_pos: Optional[Tuple[float, float]],
                 boundary_version: Optional[int] = None,
                 boundary: Optional[List[List[Tuple[float, float]]]] = None,
                 sweeps: Optional[List[dict]] = None,
                 sweep_trails: Optional[List[List[Tuple[float, float]]]] = None) -> None:
        """monolayer_rects = ACTIVE monolayer footprints (x0,y0,x1,y1) in scan
        coordinates; boundary = outer contour(s) of the accumulated monolayer
        region (Phase 4); sweeps/trails = multi-sweep trajectories (Phase 6)."""
        self._fields = fields
        self._mono_rects = list(monolayer_rects)
        self._sweeps = sweeps or []
        self._sweep_trails = sweep_trails or []
        if boundary is not None and boundary_version != self._boundary_version:
            self._boundary = boundary
            self._boundary_version = boundary_version
        self._current = current_pos
        self._has_content = bool(fields or mono_rects or self._boundary or current_pos)
        self.update()

    # -------------------------------------------------------------- paint
    def paintEvent(self, ev):  # noqa: N802 (Qt naming)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(16, 16, 16))
        p.setPen(QPen(QColor(120, 120, 120), 1))
        p.drawText(6, 14, f"SCAN MAP - view {self._user_zoom * 100:.0f}% "
                          f"(wheel zoom, drag pan)")
        w, h = self.width(), self.height()  # full widget: fit is size-proportional

        mono = self._mono_rects
        fields = self._fields
        if not (self._has_content and w > 10 and h > 10):
            if not fields:
                p.setPen(QPen(QColor(140, 140, 140), 1))
                p.drawText(8, self.height() // 2,
                           "analyze fields (CAPTURE or Auto) to build the survey map")
            p.end()
            return

        # ---- fit transform over all content (fields + layer + boundary) ----
        xs0 = [f.x - f.w / 2 for f in fields] + [r[0] for r in mono]
        ys0 = [f.y - f.h / 2 for f in fields] + [r[1] for r in mono]
        xs1 = [f.x + f.w / 2 for f in fields] + [r[2] for r in mono]
        ys1 = [f.y + f.h / 2 for f in fields] + [r[3] for r in mono]
        for contour in self._boundary:
            for pt in contour:
                xs0.append(pt[0]); ys0.append(pt[1])
                xs1.append(pt[0]); ys1.append(pt[1])
        for trail in self._sweep_trails:
            for pt in trail:
                xs0.append(pt[0]); ys0.append(pt[1])
                xs1.append(pt[0]); ys1.append(pt[1])
        if self._current:
            xs0.append(self._current[0]); ys0.append(self._current[1])
            xs1.append(self._current[0]); ys1.append(self._current[1])
        min_x, min_y = min(xs0), min(ys0)
        span_x = max(1.0, max(xs1) - min_x)
        span_y = max(1.0, max(ys1) - min_y)
        scale = min(w / span_x, h / span_y)
        # CENTER-anchored fit: scan bbox centre -> widget centre, so the
        # zoom anchor (widget centre) is a pure scale of the fit (Phase 5)
        self._fit = ((min_x + max(xs1)) / 2.0, (min_y + max(ys1)) / 2.0, scale)

        def to_px(x, y):
            return self.to_widget_px(x, y)

        # ---- Phase 6: sweep trajectories (faint polylines + start markers) ----
        for si, trail in enumerate(self._sweep_trails, start=1):
            if len(trail) >= 2:
                p.setPen(QPen(QColor(120, 120, 150), 1, Qt.PenStyle.DashLine))
                pts_px = [to_px(x, y) for x, y in trail]
                for (ax, ay), (bx, by) in zip(pts_px, pts_px[1:]):
                    p.drawLine(int(ax), int(ay), int(bx), int(by))
        for sw in self._sweeps:
            sx, sy = to_px(sw["start_x"], sw["start_y"])
            p.setPen(QPen(QColor(200, 200, 255), 1))
            p.setBrush(QColor(60, 60, 110))
            p.drawEllipse(int(sx) - 7, int(sy) - 7, 14, 14)
            p.drawText(int(sx) - 3, int(sy) + 4, str(sw.get("sweep_id", "?")))

        # ---- persistent monolayer layer: merged union, one boundary ----
        if mono:
            union = union_rects([(r[0], r[1], r[2], r[3]) for r in mono])
            fill = QColor("#2e7d32"); fill.setAlpha(120)
            for ux0, uy0, ux1, uy1 in union:
                ux0p, uy0p = to_px(ux0, uy0)
                ux1p, uy1p = to_px(ux1, uy1)
                p.fillRect(int(ux0p), int(uy0p),
                           max(2, int(ux1p - ux0p)), max(2, int(uy1p - uy0p)), fill)
            p.setPen(QPen(QColor(40, 167, 69), 2))
            span = max(span_x, span_y)
            eps = span * 0.004
            for ux0, uy0, ux1, uy1 in union:
                step_x = max(eps, (ux1 - ux0) / 24.0)
                step_y = max(eps, (uy1 - uy0) / 24.0)
                xs = [ux0 + i * step_x for i in range(25)] + [ux1]
                ys = [uy0 + i * step_y for i in range(25)] + [uy1]
                for x in xs:
                    for (ex, ey) in ((x, uy0 - eps), (x, uy1 + eps)):
                        if not point_in_any(ex, ey, union):
                            a = to_px(ex, min(max(uy0, ey - eps), uy1))
                            b = to_px(ex, max(min(uy1, ey + eps), uy0))
                            p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
                for y in ys:
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
            p.setPen(QPen(QColor(colour.red(), colour.green(), colour.blue(), 255), 1))
            p.drawRect(int(x0), int(y0),
                       max(2, int(x1 - x0)), max(2, int(y1 - y0)))
            if f.human_label:
                p.setPen(QPen(QColor(255, 255, 255), 2))
                p.drawLine(int(x0) + 2, int(y0) + 6, int(x0) + 6, int(y0) + 2)
                p.setPen(QPen(QColor(200, 200, 200), 1))

        # ---- Phase 4 boundary: clear green line (same transform) ----
        if self._boundary:
            p.setPen(QPen(QColor(30, 200, 90), 3))
            for contour in self._boundary:
                pts_px = [to_px(x, y) for x, y in contour]
                if len(pts_px) >= 2:
                    for (ax, ay), (bx, by) in zip(pts_px, pts_px[1:] + pts_px[:1]):
                        p.drawLine(int(ax), int(ay), int(bx), int(by))

        # ---- current live scan position (crosshair) ----
        if self._current:
            cx0, cy0 = to_px(self._current[0], self._current[1])
            p.setPen(QPen(QColor(80, 200, 255), 2))
            p.drawLine(int(cx0) - 6, int(cy0), int(cx0) + 6, int(cy0))
            p.drawLine(int(cx0), int(cy0) - 6, int(cx0), int(cy0) + 6)
        p.end()
