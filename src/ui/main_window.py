"""FYP live-perception GUI (PyQt6).

Layout (mirrors the approved mock):
    +--------------------------------------------------+---------------+
    |  microscope viewport (live feed + overlays)      | LIVE ANALYSIS |
    |                                                  | panel         |
    +--------------------------------------------------+ scan history  |
    |  [✓ MONOLAYER DETECTED banner when stable]        | strip (future |
    +--------------------------------------------------+  slide map)   |
    | status: capture fps | motion | cellpose | last field | age       |
    +--------------------------------------------------+---------------+

The Qt thread NEVER runs Cellpose: a 33 ms timer composites the newest camera
frame with the cached result layers; a 250 ms timer refreshes the panel. All
cross-thread data flows through SharedState (poll, no signals).
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QVBoxLayout, QWidget,
)

from src.live.controller import LiveFieldController
from src.ui.rendering import (
    CLASS_COLOURS_BGR,
    draw_class_border,
    draw_live_hud,
    draw_stale_watermark,
    draw_valid_roi,
)
from src.ui.state import SharedState

_GREEN = (80, 220, 80)
_GREY = (150, 150, 150)
_QT_COLORS = {
    "TOO_THICK": "#d32f2f",
    "MONOLAYER": "#2e7d32",
    "TOO_THIN": "#1565c0",
    "UNCERTAIN": "#ef6c00",
    "STALE": "#9e9e9e",
}


def _np_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return img.copy()  # detach from the numpy buffer


class ScanStrip(QFrame):
    """Prototype scan-history strip: displacement -> score (future slide map)."""

    def set_history(self, history, cum_x, cum_y):
        self._history = history
        self._pos = (cum_x, cum_y)
        self.update()

    def paintEvent(self, ev):  # noqa: N802 (Qt naming)
        from PyQt6.QtGui import QPainter, QColor, QPen

        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        p.setPen(QPen(QColor(120, 120, 120), 1))
        p.drawText(6, 14, "scan history (cumulative dx, dy) -> score   [future slide map]")
        hist = getattr(self, "_history", [])
        if hist:
            xs = [h["cum_x"] for h in hist]
            ys = [h["cum_y"] for h in hist]
            min_x, max_x = min(xs), max(xs + [max(xs) + 1.0])
            min_y, max_y = min(ys), max(ys + [max(ys) + 1.0])
            w, hgt = self.width() - 20, self.height() - 34
            for pt in hist:
                fx = (pt["cum_x"] - min_x) / max(1.0, (max_x - min_x))
                fy = (pt["cum_y"] - min_y) / max(1.0, (max_y - min_y))
                colour = {"TOO_THICK": "#d32f2f", "MONOLAYER": "#2e7d32",
                          "TOO_THIN": "#1565c0", "UNCERTAIN": "#ef6c00"}.get(
                              pt["raw_class"], "#888888")
                alpha = 255 if not pt["stale"] else 90
                c = QColor(colour)
                c.setAlpha(alpha)  # PyQt6: QColor has no alpha= keyword
                p.setBrush(c)
                p.setPen(QPen(c, 1))
                p.drawEllipse(10 + int(fx * w), 24 + int(fy * hgt), 6, 6)
        p.end()


class MainWindow(QMainWindow):
    def __init__(self, state: SharedState, controller: LiveFieldController,
                 cfg: Dict, session=None, view_only: bool = False):
        super().__init__()
        self.state = state
        self.controller = controller
        self.cfg = cfg
        self.session = session
        self.view_only = view_only
        self.ui_fps = 0.0
        self._ui_frames = 0
        self._ui_fps_t = time.perf_counter()
        self._last_banner_state = False

        self.setWindowTitle("AI-Guided Microscope Navigation - live perception")
        self.resize(1600, 940)

        central = QWidget()
        root = QHBoxLayout(central)

        # ---- viewport + banner column -----------------------------------
        left = QVBoxLayout()
        self.viewport = QLabel("waiting for source ...")
        self.viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewport.setMinimumSize(960, 540)
        self.viewport.setStyleSheet("background:#0a0a0a; border:1px solid #333;")
        left.addWidget(self.viewport, stretch=1)

        self.banner = QLabel("")
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setVisible(False)
        self.banner.setStyleSheet(
            "background:#1b5e20; color:#ffffff; font-size:20px; font-weight:bold;"
            "padding:10px; border:2px solid #2e7d32;"
        )
        left.addWidget(self.banner)
        root.addLayout(left, stretch=1)

        # ---- analysis panel ----------------------------------------------
        panel = QWidget()
        panel.setFixedWidth(380)
        panel.setStyleSheet("background:#111111;")
        pv = QVBoxLayout(panel)
        title = QLabel("LIVE ANALYSIS")
        title.setStyleSheet("color:#ececec; font-size:15px; font-weight:bold; padding:4px;")
        pv.addWidget(title)
        self.status_top = QLabel("")
        self.status_top.setStyleSheet("color:#bdbdbd; font-size:12px; padding:2px;")
        pv.addWidget(self.status_top)
        self.status_top.setWordWrap(True)

        self.metrics: Dict[str, QLabel] = {}
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        row = 0
        for key, label in [
            ("cellpose_instances", "Cellpose instances"),
            ("rbc_candidates", "RBC candidates"),
            ("valid_region_rbc", "Valid-region RBCs"),
            ("border_cells", "Border cells"),
            ("merge_suspects", "Merge suspects"),
            ("rbc_density", "RBC density (/Mpx)"),
            ("coverage", "Coverage"),
            ("normalized_nn", "NN distance (norm)"),
            ("contact", "Crowding / contact"),
            ("neighbors", "Close neighbours"),
            ("degree", "Mean graph degree"),
            ("cluster", "Largest cluster"),
            ("uniformity", "Uniformity (density CV)"),
            ("score", "Monolayer Score (exp.)"),
            ("smoothed", "Smoothed score"),
            ("raw_class", "Raw class"),
            ("smoothed_class", "Smoothed class"),
            ("age", "Result age"),
            ("meta", "Field / mode"),
        ]:
            name = QLabel(label)
            name.setStyleSheet("color:#9e9e9e; font-size:12px;")
            val = QLabel("-")
            val.setStyleSheet("color:#ececec; font-size:13px; font-weight:bold;")
            val.setObjectName(f"val_{key}")
            self.metrics[key] = val
            grid.addWidget(name, row, 0)
            grid.addWidget(val, row, 1)
            row += 1
        pv.addLayout(grid)

        pv.addWidget(QLabel(""))
        strip_label = QLabel("SCAN HISTORY")
        strip_label.setStyleSheet("color:#9e9e9e; font-size:12px;")
        pv.addWidget(strip_label)
        self.scan_strip = ScanStrip()
        self.scan_strip.setFixedHeight(90)
        self.scan_strip.setStyleSheet("border:1px solid #333;")
        pv.addWidget(self.scan_strip)

        self.hint = QLabel(
            "keys: SPACE pause | A analyze now | D ids | O outlines | "
            "L live/detailed | S save frame | R reset | G record | "
            "M/T/N/U label | Q quit"
        )
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#757575; font-size:11px;")
        pv.addWidget(self.hint)
        pv.addStretch(1)
        root.addWidget(panel)

        self.setCentralWidget(central)
        self.statusBar().showMessage("starting ...")

        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_tick)
        self._render_timer.start(33)
        self._panel_timer = QTimer(self)
        self._panel_timer.timeout.connect(self._panel_tick)
        self._panel_timer.start(250)

    # ------------------------------------------------------------------
    def _render_tick(self) -> None:
        snap = self.state.snapshot_display()
        frame = snap["frame"]
        self._ui_frames += 1
        now = time.perf_counter()
        if now - self._ui_fps_t >= 1.0:
            self.ui_fps = self._ui_frames / (now - self._ui_fps_t)
            self._ui_frames = 0
            self._ui_fps_t = now

        if frame is None:
            if self.state.source_error:
                img = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.putText(img, f"source error: {self.state.source_error}", (40, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 240), 2)
                self.viewport.setPixmap(QPixmap.fromImage(_np_to_qimage(img)))
            return

        result = snap["result"]
        frame = frame.copy()

        # staleness evaluated against the CURRENT position (spec §16)
        stale = True
        result_age = None
        if result is not None:
            net_x, net_y = self.state.current_net()
            stale = result.field.is_stale(
                net_x, net_y, now,
                self.cfg.get("result_stale_displacement", 160.0),
                self.cfg.get("result_stale_seconds", 25.0),
            )
            result_age = now - result.field.t_capture
            if snap["outlines_on"] and result.overlay_layer is not None:
                frame = cv2.bitwise_or(frame, result.overlay_layer)
                if snap["debug_ids"] and result.id_layer is not None:
                    frame = cv2.bitwise_or(frame, result.id_layer)

        smoothed_class = result.smoothed_class if result else None
        monolayer_now = (
            result is not None and not stale
            and result.smoothed_class == "MONOLAYER"
        )

        # valid ROI + class highlight (visual only - never logic, spec §33)
        roi_colour = _GREEN if monolayer_now else _GREY
        frame = draw_valid_roi(
            frame, self.cfg.get("valid_roi_margin", 0.10), roi_colour,
            thickness=2 if not monolayer_now else 4,
            fill_alpha=0.12 if monolayer_now else 0.0,
        )
        if monolayer_now:
            frame = draw_class_border(frame, _GREEN, thickness=8)
        if result is not None and stale:
            frame = draw_stale_watermark(frame)

        # HUD lines
        cp = snap["cellpose_state"]
        if cp == "PROCESSING":
            cp_txt = f"PROCESSING field f{snap['cellpose_field_idx']} ... ({now - snap['cellpose_started_t']:.0f}s)"
        else:
            cp_txt = cp
        lines = [
            (f"capture {snap['capture_fps']:.1f} fps | ui {self.ui_fps:.0f} fps", _GREY),
            (f"motion: {snap['motion_state']} | disp {snap['disp_px']:.0f} px/frame", _GREY),
            (f"fast screen: {snap['screen_result'] or '-'}", _GREY),
            (f"cellpose: {cp_txt}", (90, 170, 250) if cp == "PROCESSING" else _GREY),
        ]
        if result is None:
            lines.append(("detailed analysis: waiting for first field ...", _GREY))
        elif stale:
            lines.append((f"last field ({result_age:.0f}s old): {result.raw_class} - STALE, awaiting analysis", (90, 90, 240)))
        else:
            cls_txt = result.raw_class
            if result.smoothed_class and result.smoothed_class != result.raw_class:
                cls_txt += f" (smoothed {result.smoothed_class})"
            lines.append((f"verified field: {cls_txt} | score {result.score:.2f} | age {result_age:.1f}s",
                          CLASS_COLOURS_BGR.get(result.raw_class, _GREY)))
        if not snap["analysis_enabled"]:
            lines.append(("ANALYSIS PAUSED", (60, 160, 255)))
        frame = draw_live_hud(frame, lines)

        self.viewport.setPixmap(QPixmap.fromImage(_np_to_qimage(frame)))

        # banner state machine (stable smoothed MONOLAYER, non-stale only)
        if monolayer_now != self._last_banner_state:
            self._last_banner_state = monolayer_now
            if monolayer_now and result is not None:
                conf = result.smoothed_score if result.smoothed_score is not None else result.score
                self.banner.setText(
                    f"✓ MONOLAYER DETECTED    score {conf:.2f}   (experimental prototype)"
                )
                self.banner.setVisible(True)
            else:
                self.banner.setVisible(False)

        self._latest = {"snap": snap, "stale": stale, "age": result_age,
                        "frame": frame}

    # ------------------------------------------------------------------
    def _panel_tick(self) -> None:
        snap = self.state.snapshot_display()
        result = snap["result"]
        now = time.perf_counter()

        if result is None:
            for v in self.metrics.values():
                v.setText("-")
            self.status_top.setText("no detailed result yet")
        else:
            net_x, net_y = self.state.current_net()
            stale = result.field.is_stale(
                net_x, net_y, now,
                self.cfg.get("result_stale_displacement", 160.0),
                self.cfg.get("result_stale_seconds", 25.0))
            f = result.features
            age = now - result.field.t_capture
            border = sum(1 for x in result.rows
                         if x.get("rbc_candidate") and x.get("touches_border"))
            colour = _QT_COLORS.get("STALE" if stale else result.raw_class, "#ececec")
            self._set("cellpose_instances", f.get("cellpose_instance_count"))
            self._set("rbc_candidates", f.get("rbc_candidate_count"))
            self._set("valid_region_rbc", f.get("valid_region_rbc_count"))
            self._set("border_cells", border)
            self._set("merge_suspects",
                      f.get("merge_suspect_count") if result.merge_analysis_ran else "n/a (live mode)")
            self._set("rbc_density", f"{f.get('rbc_density', float('nan')):.0f}")
            self._set("coverage", f"{f.get('coverage', float('nan')) * 100:.1f}%")
            self._set("normalized_nn", f"{f.get('normalized_median_nn', float('nan')):.2f}")
            self._set("contact", f"{f.get('contact_ratio', float('nan')) * 100:.0f}%")
            self._set("neighbors",
                      f"0:{f.get('pct_0_neighbors', 0):.0f}% 1:{f.get('pct_1_neighbor', 0):.0f}% "
                      f"2:{f.get('pct_2_neighbors', 0):.0f}% 3+: {f.get('pct_3plus_neighbors', 0):.0f}%")
            self._set("degree", f"{f.get('mean_graph_degree', float('nan')):.2f}")
            self._set("cluster", f"{f.get('largest_cluster_fraction', float('nan')) * 100:.0f}% of cells")
            self._set("uniformity", f"{f.get('density_cv', float('nan')):.2f}")
            self._set("score", f"{result.score:.3f}")
            self._set("smoothed",
                      "-" if result.smoothed_score is None else f"{result.smoothed_score:.3f}")
            cls_display = ("STALE " if stale else "") + result.raw_class
            self._set("raw_class", cls_display)
            self.metrics["raw_class"].setStyleSheet(
                f"color:{colour}; font-size:14px; font-weight:bold;")
            self._set("smoothed_class", result.smoothed_class or ("n/a (stale)" if stale else "-"))
            self._set("age", f"{age:.1f} s{'  [STALE]' if stale else ''}")
            self._set("meta",
                      f"f{result.field.frame_idx} | {result.feature_mode} mode | "
                      f"cellpose {result.cellpose_ms / 1000:.1f}s | features {result.features_ms / 1000:.1f}s")

        cp = snap["cellpose_state"]
        cp_txt = ("PROCESSING previous field ..." if cp == "PROCESSING"
                  else "loading model ..." if cp == "LOADING_MODEL" else cp)
        self.status_top.setText(
            f"Cellpose: {cp_txt}\n"
            f"analyses: {snap['analysis_count']} | superseded pending: "
            f"{snap['pending_superseded']} (latest-frame policy)\n"
            f"current field: LIVE - detailed result applies to its captured position"
        )
        self.scan_strip.set_history(list(self.state.history),
                                    self.state.current_cum(),
                                    self.state.cum_y)
        self.statusBar().showMessage(
            f"● {self.state_source} | capture {snap['capture_fps']:.1f} fps | "
            f"ui {self.ui_fps:.0f} fps | motion {snap['motion_state']} | "
            f"cellpose {cp} | analyses {snap['analysis_count']}"
        )

    def _set(self, key: str, value) -> None:
        self.metrics[key].setText(str(value))

    @property
    def state_source(self) -> str:
        return getattr(self.state, "source_label", "source")

    # ------------------------------------------------------------------
    def keyPressEvent(self, ev):  # noqa: N802 (Qt naming)
        key = ev.text().lower()
        st = self.state
        if key in ("q", "\x1b"):
            self.close()
        elif key == " ":
            with st.lock:
                st.analysis_enabled = not st.analysis_enabled
            self.statusBar().showMessage(
                "analysis PAUSED" if not st.analysis_enabled else "analysis resumed", 2000)
        elif key == "a":
            with st.lock:
                st.force_request = True
        elif key == "d":
            with st.lock:
                st.debug_ids = not st.debug_ids
        elif key == "o":
            with st.lock:
                st.outlines_on = not st.outlines_on
        elif key == "l":
            with st.lock:
                st.feature_mode = "detailed" if st.feature_mode == "live" else "live"
        elif key == "r":
            self.controller.reset_session()
            with st.lock:
                st.history.clear()
        elif key == "g":
            with st.lock:
                st.recording = not st.recording
        elif key in ("m", "t", "n", "u"):
            if self.session is not None and hasattr(self, "_latest"):
                self.session.save_label(key, self._latest["frame"],
                                        self._latest["snap"]["result"])
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev):  # noqa: N802 (Qt naming)
        self.state.quit_event.set()
        super().closeEvent(ev)
