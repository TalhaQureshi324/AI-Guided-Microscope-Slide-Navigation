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

import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
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
        self._holes: List[Tuple[float, float, float, float]] = []
        self._green_qimg = None
        self._green_rect_scan: Optional[tuple] = None
        self._conflicts: List[Tuple[float, float]] = []
        self._mosaic_qimg = None
        self._mosaic_rect_scan: Optional[tuple] = None
        self._pending_mosaic = None
        self._revisit: List[dict] = []
        self._revisit_target: int = -1
        self._fov_wh: Optional[Tuple[float, float]] = None
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
    def set_mosaic_layer(self, rgb: np.ndarray, origin_x: float, origin_y: float,
                         scan_per_px: float) -> None:
        """Phase 9: stitched RGB mosaic base layer in scan coordinates."""
        h, w = rgb.shape[:2]
        self._mosaic_qimg = QImage(rgb.data, w, h, 3 * w,
                                   QImage.Format.Format_RGB888).copy()
        self._mosaic_rect_scan = (origin_x, origin_y,
                                  origin_x + w * scan_per_px,
                                  origin_y + h * scan_per_px)

    def set_fov(self, fov_wh: Tuple[float, float]) -> None:
        self._fov_wh = fov_wh

    def set_semantic(self, green_mask: np.ndarray, origin_x: float,
                     origin_y: float, cell: float,
                     conflicts: List[Dict]) -> None:
        """Phase 11: semantic green mask + conflict cells (scan coords).
        green_mask[cell_y, cell_x] = True where monolayer is confirmed."""
        h, w = green_mask.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[green_mask] = (46, 204, 113)          # flat green semantic color
        self._green_qimg = QImage(rgb.data, w, h, 3 * w,
                                  QImage.Format.Format_RGB888).copy()
        self._green_rect_scan = (origin_x, origin_y,
                                 origin_x + w * cell, origin_y + h * cell)
        self._conflicts = list(conflicts)
        self.update()

    def set_holes(self, holes: List[Tuple[float, float, float, float]]) -> None:
        """Interior UNSCANNED gaps as (cx, cy, w, h) scan-coord tuples - Phase 10 §2."""
        self._holes = list(holes)
        self._hole_rects = [
            (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
            for (cx, cy, w, h) in holes
        ]
        self.update()

    def set_mosaic_overview(self, rgb: np.ndarray, origin_x: float,
                            origin_y: float, scan_per_px: float) -> None:
        self._pending_mosaic = (np.ascontiguousarray(rgb), origin_x, origin_y,
                                scan_per_px)

    def set_revisit(self, points: List[dict], target_idx: int = -1) -> None:
        """Phase 13: revisit candidates ({x_scan, y_scan, ...}) + the active
        target, drawn as numbered magenta markers (§6)."""
        self._revisit = list(points)
        self._revisit_target = int(target_idx)
        self.update()

    def set_boundary(self, version: Optional[int],
                     contours: Optional[List[List[Tuple[float, float]]]]) -> None:
        """Boundary update with version gating (Phase 11 semantic contours
        override the Phase 4 merged-layer contour when fresh)."""
        if contours is None or version is None:
            return
        if version != self._boundary_version:
            self._boundary = contours
            self._boundary_version = version
        self.update()

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
        if self._pending_mosaic is not None:
            rgb, ox, oy, sc = self._pending_mosaic
            h8, w8 = rgb.shape[:2]
            self._mosaic_qimg = QImage(rgb.data, w8, h8, 3 * w8,
                                       QImage.Format.Format_RGB888).copy()
            self._mosaic_rect_scan = (ox, oy, ox + w8 * sc, oy + h8 * sc)
            self._pending_mosaic = None
        if boundary is not None and boundary_version != self._boundary_version:
            self._boundary = boundary
            self._boundary_version = boundary_version
        self._current = current_pos
        self._has_content = bool(fields or monolayer_rects or self._boundary
                                 or current_pos
                                 or getattr(self, "_pending_mosaic", None)
                                 or self._mosaic_qimg is not None)
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
        mrect = getattr(self, "_mosaic_rect_scan", None)
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
        # The fit covers COMMITTED content only (fields, layers, boundary,
        # trails, revisit points). The live crosshair deliberately does NOT
        # participate: chasing it rescaled the whole map on every step, which
        # read as "slight movement moves me drastically" (2026-10-14 repair).
        # When the operator walks past the mapped territory the crosshair is
        # drawn as a clamped edge indicator instead.
        if not xs0 and self._current:
            xs0.append(self._current[0]); ys0.append(self._current[1])
            xs1.append(self._current[0]); ys1.append(self._current[1])
        for rp in self._revisit:
            xs0.append(rp["x_scan"]); ys0.append(rp["y_scan"])
            xs1.append(rp["x_scan"]); ys1.append(rp["y_scan"])
        min_x, min_y = min(xs0), min(ys0)
        span_x = max(1.0, max(xs1) - min_x)
        span_y = max(1.0, max(ys1) - min_y)
        scale = min(w / span_x, h / span_y)
        # CENTER-anchored fit: scan bbox centre -> widget centre, so the
        # zoom anchor (widget centre) is a pure scale of the fit (Phase 5)
        self._fit = ((min_x + max(xs1)) / 2.0, (min_y + max(ys1)) / 2.0, scale)

        def to_px(x, y):
            return self.to_widget_px(x, y)

        # ---- Phase 9: stitched RGB mosaic base layer (scan-coord placement) ----
        if getattr(self, "_mosaic_qimg", None) is not None and mrect:
            p0x, p0y = to_px(mrect[0], mrect[1])
            p1x, p1y = to_px(mrect[2], mrect[3])
            p.drawImage(QRectF(p0x, p0y, p1x - p0x, p1y - p0y), self._mosaic_qimg)

        # ---- Phase 11: semantic green mask (translucent, under everything) ----
        gq = getattr(self, "_green_qimg", None)
        grect = getattr(self, "_green_rect_scan", None)
        if gq is not None and grect:
            gp0x, gp0y = to_px(grect[0], grect[1])
            gp1x, gp1y = to_px(grect[2], grect[3])
            p.setOpacity(0.45)
            p.drawImage(QRectF(gp0x, gp0y, gp1x - gp0x, gp1y - gp0y), gq)
            p.setOpacity(1.0)

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

        # ---- Phase 10: UNSCANNED GAP holes (red translucent + hatch ticks) ----
        for hx0, hy0, hx1, hy1 in getattr(self, "_hole_rects", []) or []:
            ax0, ay0 = to_px(hx0, hy0)
            ax1, ay1 = to_px(hx1, hy1)
            fill = QColor("#b91c1c"); fill.setAlpha(56)
            p.fillRect(int(ax0), int(ay0),
                       max(2, int(ax1 - ax0)), max(2, int(ay1 - ay0)), fill)
            p.setPen(QPen(QColor(185, 28, 28), 1, Qt.PenStyle.DashLine))
            p.drawRect(int(ax0), int(ay0),
                       max(2, int(ax1 - ax0)), max(2, int(ay1 - ay0)))
            p.setPen(QPen(QColor(185, 28, 28), 1))
            gx0, gy0 = int(ax0) + 4, int(ay0) + 6
            p.drawText(int(gx0), int(gy0), "UNSCANNED GAP")

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

        # ---- Phase 11: CONFLICTING EVIDENCE markers (amber rings) ----
        for c in getattr(self, "_conflicts", []) or []:
            cx0, cy0 = to_px(c["x"], c["y"])
            p.setPen(QPen(QColor(250, 190, 40), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(int(cx0) - 5, int(cy0) - 5, 10, 10)

        # ---- Phase 13: revisit points (numbered magenta) + active target ----
        for ri, rp in enumerate(self._revisit):
            rx, ry = to_px(rp["x_scan"], rp["y_scan"])
            is_target = (ri == self._revisit_target)
            ring = QColor(220, 60, 220) if is_target else QColor(170, 80, 170)
            p.setPen(QPen(ring, 2 if is_target else 1))
            p.setBrush(QColor(16, 16, 16))
            r = 8 if is_target else 6
            p.drawEllipse(int(rx) - r, int(ry) - r, 2 * r, 2 * r)
            p.setPen(QPen(ring, 1))
            p.drawText(int(rx) - 3, int(ry) + 4, str(ri + 1))
            if is_target:
                p.setPen(QPen(QColor(220, 60, 220), 1, Qt.PenStyle.DashLine))
                p.drawLine(int(rx) - 14, int(ry), int(rx) + 14, int(ry))
                p.drawLine(int(rx), int(ry) - 14, int(rx), int(ry) + 14)

        # ---- current live scan position (crosshair; edge-indicator when the
        # stage has walked outside the mapped territory) ----
        if self._current:
            cx0, cy0 = to_px(self._current[0], self._current[1])
            m = 12
            off_view = (cx0 < m or cx0 > w - m or cy0 < m or cy0 > h - m)
            ix, iy = int(min(max(cx0, m), w - m)), int(min(max(cy0, m), h - m))
            if off_view:
                # dimmed crosshair pinned to the edge + tick pointing at the
                # true (off-screen) direction
                p.setPen(QPen(QColor(80, 200, 255, 150), 2))
                dx_real, dy_real = cx0 - ix, cy0 - iy
                n = (dx_real ** 2 + dy_real ** 2) ** 0.5 or 1.0
                p.drawLine(ix, iy, int(ix + dx_real / n * 11), int(iy + dy_real / n * 11))
                p.setPen(QPen(QColor(80, 200, 255, 150), 1))
                p.drawText(ix + 6, iy - 6, "LIVE")
            else:
                p.setPen(QPen(QColor(80, 200, 255), 2))
                p.drawLine(ix - 6, iy, ix + 6, iy)
                p.drawLine(ix, iy - 6, ix, iy + 6)
        if self._fov_wh and self._current:
            fw, fh = self._fov_wh
            fx0, fy0 = to_px(self._current[0] - fw / 2, self._current[1] - fh / 2)
            fx1, fy1 = to_px(self._current[0] + fw / 2, self._current[1] + fh / 2)
            if fx1 < 0 or fx0 > w or fy1 < 0 or fy0 > h:
                pass    # FOV entirely off-view: skip (fit no longer follows it)
            else:
                p.setPen(QPen(QColor(80, 200, 255), 1, Qt.PenStyle.DashLine))
                p.drawRect(int(fx0), int(fy0), int(fx1 - fx0), int(fy1 - fy0))
        p.end()
