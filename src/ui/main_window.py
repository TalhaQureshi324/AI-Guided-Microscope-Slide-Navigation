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
    QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QPushButton, QVBoxLayout,
    QWidget,
)

from src.live.controller import LiveFieldController
from src.ui.scan_map_panel import ScanMapPanel
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


class MainWindow(QMainWindow):
    def __init__(self, state: SharedState, controller: LiveFieldController,
                 cfg: Dict, session=None, view_only: bool = False, jobmgr=None,
                 scan_map=None, nav_machine=None, tracker=None, mosaic=None,
                 raster=None, exploration=None):
        super().__init__()
        self.state = state
        self.controller = controller
        self.cfg = cfg
        self.session = session
        self.jobmgr = jobmgr
        self.scan_map = scan_map
        self.nav_machine = nav_machine
        self.tracker = tracker
        self.mosaic = mosaic
        self.raster = raster
        self.exploration = exploration   # Phase 13 high-mag exploration
        self.view_only = view_only
        self._fov: Optional[Tuple[float, float]] = None
        self.ui_fps = 0.0
        self._ui_frames = 0
        self._ui_fps_t = time.perf_counter()
        self._selected_job_id: Optional[int] = None   # job viewer selection

        self.setWindowTitle("AI-Guided Microscope Navigation - live perception")
        self.resize(1600, 940)

        central = QWidget()
        root = QHBoxLayout(central)

        # ---- left column: controls + viewport + banner -------------------
        left = QVBoxLayout()
        controls = QHBoxLayout()
        self.btn_capture = QPushButton("  CAPTURE & ANALYZE  ")
        self.btn_capture.setStyleSheet(
            "background:#2e7d32; color:white; font-size:14px; font-weight:bold;"
            "padding:8px 18px; border:1px solid #1b5e20;"
        )
        self.btn_capture.setToolTip("Capture the current microscope field and "
                                    "analyze it in the background (camera stays live)")
        self.btn_capture.clicked.connect(self._on_capture)
        controls.addWidget(self.btn_capture)

        self.btn_auto = QPushButton("Auto Analyze: ON")
        self.btn_auto.setStyleSheet("padding:8px 12px;")
        self.btn_auto.setToolTip("Automatic field detection (motion-based). "
                                 "OFF = analyze only manual captures.")
        self.btn_auto.clicked.connect(self._on_toggle_auto)
        controls.addWidget(self.btn_auto)

        self.btn_live = QPushButton("● LIVE")
        self.btn_live.setStyleSheet("padding:8px 12px;")
        self.btn_live.setToolTip("Return the viewport to the live feed")
        self.btn_live.clicked.connect(self._on_view_live)
        controls.addWidget(self.btn_live)

        self.btn_snapshot = QPushButton("SAVE SNAPSHOT")
        self.btn_snapshot.setStyleSheet("padding:8px 12px;")
        self.btn_snapshot.setToolTip("Save the current frame WITHOUT Cellpose")
        self.btn_snapshot.clicked.connect(self._on_snapshot)
        controls.addWidget(self.btn_snapshot)

        self.btn_sweep = QPushButton("START NEW SWEEP")
        self.btn_sweep.setStyleSheet("padding:8px 12px;")
        self.btn_sweep.setToolTip(
            "Begin a new sweep trajectory on the SAME slide map. If a completed "
            "job is selected in the list, the new sweep is anchored to that "
            "field's location (position the microscope there first).")
        self.btn_sweep.clicked.connect(self._on_new_sweep)
        controls.addWidget(self.btn_sweep)
        controls.addStretch(1)
        left.addLayout(controls)

        self.viewport = QLabel("waiting for source ...")
        self.viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewport.setMinimumSize(960, 540)
        self.viewport.setStyleSheet("background:#0a0a0a; border:1px solid #333;")
        left.addWidget(self.viewport, stretch=1)

        # navigation state banner (Phase 2): driven ONLY by the hysteresis
        # state machine over fresh multi-feature scores - never by one frame
        self.banner = QLabel("OUTSIDE MONOLAYER")
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setStyleSheet(
            "background:#7f1d1d; color:#ffffff; font-size:18px; font-weight:bold;"
            "padding:8px; border:2px solid #b91c1c;"
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
        self.view_label = QLabel("VIEWING: LIVE")
        self.view_label.setStyleSheet(
            "color:#4fc3f7; font-size:13px; font-weight:bold; padding:2px 4px;")
        pv.addWidget(self.view_label)
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
        jobs_title = QLabel("ANALYSIS JOBS  (click a completed job to inspect it)")
        jobs_title.setStyleSheet("color:#9e9e9e; font-size:12px;")
        pv.addWidget(jobs_title)
        self.job_list = QListWidget()
        self.job_list.setFixedHeight(180)
        self.job_list.setStyleSheet(
            "background:#0d0d0d; color:#d6d6d6; font-size:12px; border:1px solid #333;")
        self.job_list.itemClicked.connect(self._on_job_selected)
        pv.addWidget(self.job_list)

        map_row = QHBoxLayout()
        strip_label = QLabel("SCAN MAP (spatial survey)")
        strip_label.setStyleSheet("color:#9e9e9e; font-size:12px;")
        map_row.addWidget(strip_label)
        map_row.addStretch(1)
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(28)
        self.btn_zoom_in.setToolTip("Zoom the map in (rendering only)")
        self.btn_zoom_in.clicked.connect(lambda: self.scan_map_panel.zoom_by(1.25))
        map_row.addWidget(self.btn_zoom_in)
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(28)
        self.btn_zoom_out.setToolTip("Zoom the map out (rendering only)")
        self.btn_zoom_out.clicked.connect(lambda: self.scan_map_panel.zoom_by(0.8))
        map_row.addWidget(self.btn_zoom_out)
        self.btn_fit_view = QPushButton("Reset View")
        self.btn_fit_view.setToolTip("Reset map zoom/pan - does NOT delete scan history")
        self.btn_fit_view.clicked.connect(lambda: self.scan_map_panel.reset_view())
        map_row.addWidget(self.btn_fit_view)
        pv.addLayout(map_row)
        self.scan_map_panel = ScanMapPanel()
        self.scan_map_panel.setStyleSheet("border:1px solid #333;")
        pv.addWidget(self.scan_map_panel)

        self.hint = QLabel(
            "keys: SPACE pause | A analyze now | D ids | O outlines | "
            "L live/detailed | S save frame | R reset | G record | "
            "M/T/N/U label | Q quit"
        )
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#757575; font-size:11px;")
        pv.addWidget(self.hint)

        # ---- Phase 13: high-mag exploration --------------------------------
        exp_title = QLabel("HIGH-MAG EXPLORATION (survey revisit)")
        exp_title.setStyleSheet("color:#ce93d8; font-size:12px; font-weight:bold;")
        pv.addWidget(exp_title)

        obj_row = QHBoxLayout()
        obj_row.addWidget(QLabel("objective"))
        self.exp_objective = QComboBox()
        if self.exploration is not None:
            self.exp_objective.addItems(sorted(self.exploration.objectives()))
            self.exp_objective.setCurrentText(
                self.exploration.active_objective)
            self.exp_objective.currentTextChanged.connect(
                self._on_objective_changed)
        obj_row.addWidget(self.exp_objective, stretch=1)
        pv.addLayout(obj_row)

        self.exp_cal = QLabel("calibration: -")
        self.exp_cal.setStyleSheet("color:#b39ddb; font-size:11px;")
        pv.addWidget(self.exp_cal)

        self.btn_finalize = QPushButton("FINALIZE SURVEY MAP")
        self.btn_finalize.setStyleSheet("padding:6px 10px;")
        self.btn_finalize.setToolTip(
            "Freeze the current low-mag survey as the reference: writes the "
            "RGB mosaic, coverage, monolayer mask, transforms and calibration "
            "to survey_final/ - revisits then work against this frozen map")
        self.btn_finalize.clicked.connect(self._on_finalize_survey)
        pv.addWidget(self.btn_finalize)

        self.exp_survey_status = QLabel("survey: NOT FINALIZED")
        self.exp_survey_status.setStyleSheet("color:#9e9e9e; font-size:11px;")
        self.exp_survey_status.setWordWrap(True)
        pv.addWidget(self.exp_survey_status)

        self.exp_nav = QLabel("navigation: -")
        self.exp_nav.setStyleSheet("color:#ececec; font-size:12px; font-weight:bold;")
        self.exp_nav.setWordWrap(True)
        pv.addWidget(self.exp_nav)

        rev_row = QHBoxLayout()
        self.exp_mode = QComboBox()
        self.exp_mode.addItems(["grid", "top_confidence", "centroid"])
        rev_row.addWidget(self.exp_mode)
        self.btn_generate = QPushButton("GENERATE")
        self.btn_generate.setStyleSheet("padding:6px 8px;")
        self.btn_generate.setToolTip(
            "Generate revisit points INSIDE the mapped monolayer (spec §6)")
        self.btn_generate.clicked.connect(self._on_generate_revisit)
        rev_row.addWidget(self.btn_generate)
        self.btn_goto = QPushButton("GO TO NEXT")
        self.btn_goto.setStyleSheet("padding:6px 8px;")
        self.btn_goto.setToolTip("Mark the next revisit point as the target")
        self.btn_goto.clicked.connect(self._on_goto_next)
        rev_row.addWidget(self.btn_goto)
        pv.addLayout(rev_row)

        self.btn_relocate = QPushButton("RELOCATE HERE (vs frozen mosaic)")
        self.btn_relocate.setStyleSheet("padding:6px 8px;")
        self.btn_relocate.setToolTip(
            "Recover the survey pose of the CURRENT field by registering it "
            "against the frozen mosaic - do not assume mechanical centering (§7)")
        self.btn_relocate.clicked.connect(self._on_relocate)
        pv.addWidget(self.btn_relocate)

        self.exp_points = QListWidget()
        self.exp_points.setFixedHeight(84)
        self.exp_points.setStyleSheet(
            "background:#0d0d0d; color:#e1bee7; font-size:11px; border:1px solid #333;")
        self.exp_points.itemClicked.connect(self._on_revisit_selected)
        pv.addWidget(self.exp_points)

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
    # manual capture workflow (Phase G1/G3/G9)
    def _on_capture(self):
        with self.state.lock:
            self.state.manual_requests += 1
        self.statusBar().showMessage("CAPTURE requested - job queued, camera stays live", 2000)

    def _on_toggle_auto(self):
        with self.state.lock:
            self.state.auto_analysis_enabled = not self.state.auto_analysis_enabled
            on = self.state.auto_analysis_enabled
        self.btn_auto.setText(f"Auto Analyze: {'ON' if on else 'OFF'}")
        self.statusBar().showMessage(
            "AUTO analysis enabled" if on else
            "AUTO analysis OFF - Cellpose runs only on CAPTURE & ANALYZE", 3000)

    def _on_view_live(self):
        self._selected_job_id = None
        with self.state.lock:
            self.state.view_job_id = None

    def _on_snapshot(self):
        with self.state.lock:
            self.state.snapshot_requests += 1
        self.statusBar().showMessage("snapshot will be saved (no Cellpose)", 2000)

    def _on_new_sweep(self):
        """Phase 6/12: begin a new sweep row, keeping the same slide map.

        Anchored to the selected completed job when one is chosen (the
        operator declares the microscope is at that field's location). The
        raster navigator alternates the row direction (boustrophedon)."""
        if self.scan_map is None:
            return
        raster_dir = None
        if self.raster is not None:
            raster_dir = self.raster.start_sweep()["direction"]
        job = self._selected_completed_job()
        anchor = job.job_id if job is not None else None
        net_x, net_y = self.state.current_net()
        sweep = self.scan_map.start_sweep(time.perf_counter(), net_x, net_y,
                                          anchor_job_id=anchor,
                                          row_direction=raster_dir)
        if sweep is not None:
            self.statusBar().showMessage(
                f"sweep #{sweep['sweep_id']} ({raster_dir}) started at map "
                f"({sweep['start_x']:.0f}, {sweep['start_y']:.0f})"
                + (f", anchored to job #{anchor}" if anchor else ""),
                4000)

    def _on_job_selected(self, item):
        job_id = item.data(Qt.ItemDataRole.UserRole)
        if job_id is not None:
            self._selected_job_id = job_id
            with self.state.lock:
                self.state.view_job_id = job_id

    def _selected_completed_job(self):
        if self._selected_job_id is None or self.jobmgr is None:
            return None
        return self.jobmgr.get_completed(self._selected_job_id)

    # ------------------------------------------------------------------
    # Phase 13: high-mag exploration handlers
    def _on_finalize_survey(self):
        if self.exploration is None or self.scan_map is None:
            return
        if not len(self.scan_map.fields()):
            self.statusBar().showMessage(
                "FINALIZE refused: survey map is empty - scan fields first", 4000)
            return
        from pathlib import Path
        out_dir = Path(self.state.session_dir or ".") / "survey_final"
        try:
            paths = self.exploration.finalize(
                self.scan_map, self.mosaic, out_dir,
                metadata={"session": self.state.session_dir,
                          "source": self.state.source_label})
        except ValueError as exc:
            self.statusBar().showMessage(f"FINALIZE failed: {exc}", 5000)
            return
        n = self.exploration.survey.monolayer_mask.sum() \
            if self.exploration.survey is not None else 0
        self.exp_survey_status.setText(
            f"survey: FROZEN -> {paths['survey_dir']} "
            f"({n} monolayer cells)")
        self.btn_finalize.setEnabled(False)
        self.statusBar().showMessage(
            "survey map FROZEN - switch objective and revisit the monolayer", 5000)

    def _on_objective_changed(self, name: str):
        if self.exploration is None:
            return
        self.exploration.set_active_objective(name)
        self._update_cal_label()

    def _update_cal_label(self):
        if self.exploration is None:
            return
        cal = self.exploration.active_cal()
        frame = self.state.frame_bgr
        fov = ""
        if frame is not None:
            fw, fh = frame.shape[1], frame.shape[0]
            fov = f" | FOV {cal.to_um(fw):.0f}x{cal.to_um(fh):.0f} um"
        self.exp_cal.setText(
            f"calibration: {cal.pixels_per_micron:.3f} px/um "
            f"({cal.microns_per_pixel:.3f} um/px, {cal.source}){fov}")

    def _on_generate_revisit(self):
        if self.exploration is None:
            return
        if self.exploration.survey is None:
            self.statusBar().showMessage(
                "revisit needs a FINALIZED survey map", 4000)
            return
        count = int(self.cfg.get("survey", {}).get("revisit_count", 6))
        pts = self.exploration.generate_revisit(self.exp_mode.currentText(),
                                                count)
        self.exp_points.clear()
        for i, p in enumerate(pts):
            self.exp_points.addItem(
                f"#{i + 1}  ({p['x_um']:.0f}, {p['y_um']:.0f}) um  "
                f"conf {p['conf']:.2f}")
        self.scan_map_panel.set_revisit(pts,
                                        0 if pts else -1)
        self.statusBar().showMessage(
            f"{len(pts)} revisit point(s) generated ({self.exp_mode.currentText()})",
            4000)

    def _on_goto_next(self):
        if self.exploration is None:
            return
        t = self.exploration.advance_target()
        if t is None:
            self.statusBar().showMessage("no revisit points - GENERATE first", 4000)
            return
        self.exp_points.setCurrentRow(self.exploration.target_idx)
        self.scan_map_panel.set_revisit(self.exploration.revisit,
                                        self.exploration.target_idx)
        self.statusBar().showMessage(
            f"target #{self.exploration.target_idx + 1} at "
            f"({t['x_um']:.0f}, {t['y_um']:.0f}) um - move the stage there",
            6000)

    def _on_revisit_selected(self, item):
        if self.exploration is None:
            return
        idx = self.exp_points.row(item)
        t = self.exploration.set_target(idx)
        if t is not None:
            self.scan_map_panel.set_revisit(self.exploration.revisit, idx)
            self.statusBar().showMessage(
                f"target #{idx + 1} at ({t['x_um']:.0f}, {t['y_um']:.0f}) um", 4000)

    def _on_relocate(self):
        if self.exploration is None:
            return
        if self.exploration.survey is None:
            self.statusBar().showMessage(
                "relocate needs a FINALIZED survey map", 4000)
            return
        with self.state.lock:
            frame = None if self.state.frame_bgr is None \
                else self.state.frame_bgr.copy()
        if frame is None:
            self.statusBar().showMessage("relocate: no live frame", 4000)
            return
        hit = self.exploration.relocate(frame)
        if hit is None:
            self.exp_nav.setText(
                "RELOCATE FAILED: field does not match the frozen mosaic "
                "(position unknown - see limitations)")
            self.statusBar().showMessage(
                "relocate FAILED - no pose assumed (honest failure, §8)", 5000)
        else:
            # relocate() already re-anchored the exploration chain on success
            self.statusBar().showMessage(
                f"relocated: pose ({hit['x_scan']:.0f}, {hit['y_scan']:.0f}) "
                f"scan px, response {hit['response']:.2f}", 6000)

    # ------------------------------------------------------------------
    def _render_tick(self) -> None:
        snap = self.state.snapshot_display()
        self._ui_frames += 1
        now = time.perf_counter()
        if now - self._ui_fps_t >= 1.0:
            self.ui_fps = self._ui_frames / (now - self._ui_fps_t)
            self._ui_frames = 0
            self._ui_fps_t = now

        # ---- job viewer: a selected completed job owns the viewport ------
        viewing_job = self._selected_completed_job()
        if viewing_job is not None and viewing_job.frame is not None:
            img = viewing_job.frame.copy()
            if snap["outlines_on"] and viewing_job.overlay_layer is not None:
                img = cv2.bitwise_or(img, viewing_job.overlay_layer)
                if snap["debug_ids"] and viewing_job.id_layer is not None:
                    img = cv2.bitwise_or(img, viewing_job.id_layer)
            colour = CLASS_COLOURS_BGR.get(viewing_job.raw_class, _GREY)
            frame = draw_class_border(img, colour, thickness=6)
            merges = (viewing_job.features or {}).get("merge_suspect_count", "?")
            occ = (viewing_job.features or {}).get("screen_occupancy")
            gross = (viewing_job.features or {}).get("screen_gross_thick", False)
            cls_txt = (f"{viewing_job.raw_class} (screen)" if gross
                       else str(viewing_job.raw_class))
            occ_txt = f"   screen occ {occ * 100:.0f}%" if occ is not None else ""
            reg = getattr(viewing_job, "reg_state", "")
            reg_txt = (f"{reg.split('_')[-1]} via {viewing_job.reg_method} "
                       f"conf {viewing_job.reg_confidence:.2f}") if reg else ""
            lines = [
                (f"JOB #{viewing_job.job_id}  [{viewing_job.priority}]", _GREY),
                (f"RESULT: {cls_txt}   score {viewing_job.score:.2f}",
                 CLASS_COLOURS_BGR.get(viewing_job.raw_class, _GREY)),
                (f"RBCs {(viewing_job.features or {}).get('rbc_candidate_count', '?')}"
                 f"   coverage {(viewing_job.features or {}).get('coverage', 0) * 100:.0f}%"
                 f"   merge suspects {merges}{occ_txt}", _GREY),
            ]
            if getattr(viewing_job, "drift_warning", False):
                lines.append(("!! POSE DRIFT WARNING - revisit disagrees with chained pose",
                              (60, 90, 250)))
            if reg:
                lines.append((f"registration: {reg_txt}"
                              f"   global ({viewing_job.map_x:.0f}, {viewing_job.map_y:.0f})",
                              _GREY))
            if reg == "REGISTRATION_FAILED":
                lines.append(("REGISTRATION FAILED - return to the previous known "
                              "field and capture again", (60, 60, 240)))
            lines.append((f"press ● LIVE to return to the camera feed", _GREY))
            frame = draw_live_hud(frame, lines)
            self.viewport.setPixmap(QPixmap.fromImage(_np_to_qimage(frame)))
            self._latest = {"snap": snap, "stale": False, "age": None, "frame": frame,
                            "job": viewing_job}
            return

        frame = snap["frame"]
        if frame is not None:
            self._fov = (frame.shape[1], frame.shape[0])
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
            stale = result.is_stale(
                net_x, net_y, now,
                self.cfg.get("result_stale_displacement", 160.0),
                self.cfg.get("result_stale_seconds", 25.0),
            )
            result_age = now - result.t_capture
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
        # ---- Phase 8/12: live movement guidance (coverage + raster) ----
        preview = getattr(self.state, "preview", {}) or {}
        if preview.get("raster_hint"):
            lines.append((f"RASTER: {preview['raster_hint']}", (170, 200, 250)))
        if preview.get("move_hint"):
            pv_state = preview.get("state", "")
            colour = ((80, 220, 130) if pv_state == "CAPTURE_READY"
                      else (250, 170, 60) if pv_state == "GAP_RISK"
                      else (90, 170, 250))
            lines.append((f"{preview['move_hint']}  |  scanned-overlap "
                          f"{preview.get('overlap_prev_pct', 0) * 100:.0f}%", colour))
        result_reg = snap.get("result")
        if result_reg is not None and getattr(result_reg, "reg_state", ""):
            reg_line = (f"registration: {result_reg.reg_state.split('_')[-1]} "
                        f"({result_reg.reg_method}, conf "
                        f"{result_reg.reg_confidence:.2f})")
            if getattr(result_reg, "drift_warning", False):
                reg_line += "  !! POSE DRIFT"
            lines.append((reg_line,
                          (80, 220, 130) if result_reg.reg_state == "REGISTRATION_GOOD"
                          else (90, 170, 250) if result_reg.reg_state == "REGISTRATION_WEAK"
                          else (90, 90, 240)))
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

        # ---- Phase 8: CAPTURE button guidance coloring ----
        prev = getattr(self.state, "preview", {}) or {}
        pv_state = prev.get("state", "")
        if pv_state:
            self._apply_capture_button_style(pv_state)

        self.viewport.setPixmap(QPixmap.fromImage(_np_to_qimage(frame)))

        self._latest = {"snap": snap, "stale": stale, "age": result_age,
                        "frame": frame}

    # ------------------------------------------------------------------
    def _panel_tick(self) -> None:
        snap = self.state.snapshot_display()
        now = time.perf_counter()

        # ---- explicit UI context (Phase 0): the panel shows exactly ONE
        # result object - the selected completed job when one is chosen,
        # otherwise the last valid live analysis. Never a mixture.
        viewing_job = self._selected_completed_job()
        selected_id = self._selected_job_id
        if viewing_job is not None:
            result = viewing_job
            context = f"VIEWING: JOB #{viewing_job.job_id}"
            stale = False            # a job's result belongs to its own snapshot
            age = now - viewing_job.t_capture
        elif snap["result"] is not None:
            result = snap["result"]
            context = "VIEWING: LIVE"
            net_x, net_y = self.state.current_net()
            stale = result.is_stale(
                net_x, net_y, now,
                self.cfg.get("result_stale_displacement", 160.0),
                self.cfg.get("result_stale_seconds", 25.0))
            age = now - result.t_capture
        else:
            result = None
            context = "VIEWING: LIVE"
            stale = False
            age = None

        if selected_id is not None and viewing_job is None:
            context += f"  (job #{selected_id} not finished - live shown)"
        self.view_label.setText(context)
        self.view_label.setStyleSheet(
            "color:#4fc3f7; font-size:13px; font-weight:bold; padding:2px 4px;"
            if viewing_job is None else
            "color:#ffb74d; font-size:13px; font-weight:bold; padding:2px 4px;")

        if result is None:
            for v in self.metrics.values():
                v.setText("-")
            self.status_top.setText("no detailed result yet")
        else:
            # defensive: a COMPLETE-marked job reaches the registry before its
            # result fields populate - never crash the tick on that window
            f = result.features or {}
            rows = result.rows or []
            border = sum(1 for x in rows
                         if x.get("rbc_candidate") and x.get("touches_border"))
            if result.raw_class is None or result.score is None:
                self._set("raw_class", "analyzing ...")
                self.metrics["raw_class"].setStyleSheet("color:#bdbdbd;")
                self._set("age", "-")
                self._set("meta",
                          f"job #{result.job_id} queued for registration/results")
            else:
                self._render_result_metrics(result, f, border, stale, age)

        cp = snap["cellpose_state"]
        cp_txt = ("PROCESSING previous field ..." if cp == "PROCESSING"
                  else "loading model ..." if cp == "LOADING_MODEL" else cp)

        # manual workflow status (Phase G1/G3)
        jobs_txt = ""
        if self.jobmgr is not None:
            stats = self.jobmgr.stats()
            jobs_txt = (f"jobs: {stats['total_jobs']} total | queue: "
                        f"manual {stats['waiting_manual']}, "
                        f"auto pending {1 if stats['waiting_auto'] else 0}")
            last_m = snap["last_manual"]
            if last_m is not None and last_m.raw_class:
                lbl = f" | human: {last_m.human_label}" if last_m.human_label else ""
                jobs_txt += (f"\nlatest capture: job #{last_m.job_id} "
                             f"{last_m.raw_class} score {last_m.score:.2f}{lbl}")

        if viewing_job is not None:
            scope = (f"all metrics belong to job #{viewing_job.job_id} "
                     f"(captured at frame {viewing_job.frame_idx}) - "
                     f"press \u25cf LIVE to return to the live analysis")
        elif stale:
            scope = ("live result is STALE - microscope has moved; "
                     "capture or wait for a new analysis")
        else:
            scope = "live result - metrics describe the current field"

        self.status_top.setText(
            f"{scope}\n"
            f"Cellpose: {cp_txt} | analyses: {snap['analysis_count']} | "
            f"superseded auto: {self.jobmgr.superseded_auto if self.jobmgr else 0}"
            + (f"\n{jobs_txt}" if jobs_txt else "")
        )
        self._update_nav_banner()
        self._apply_capture_button_style(
            (getattr(self.state, "preview", {}) or {}).get("state", ""))
        self._refresh_job_list()
        last_reg = snap.get("result")
        if last_reg is not None and getattr(last_reg, "reg_state", ""):
            reg_line = (f"registration: {last_reg.reg_state.split('_')[-1]} via "
                        f"{last_reg.reg_method} conf {last_reg.reg_confidence:.2f}"
                        + (" | !! POSE DRIFT" if getattr(last_reg, "drift_warning", False) else ""))
            self.status_top.setText(self.status_top.text() + "\n" + reg_line)

        # ---- Phase 13: exploration status + calibration readout ------------
        if self.exploration is not None:
            self._update_cal_label()
            exp = snap.get("exploration") or {}
            if self.exploration.survey is None:
                self.exp_nav.setText("navigation: survey not finalized")
            elif not exp:
                self.exp_nav.setText(
                    f"navigation: FROZEN survey ready ({self.exploration.survey.monolayer_mask.sum()} "
                    "monolayer cells) - capture a field at high mag")
            else:
                state_txt = exp.get("state", "-")
                colour = {"INSIDE MAPPED MONOLAYER": "#66bb6a",
                          "NEAR MONOLAYER EDGE": "#ffa726",
                          "OUTSIDE MAPPED MONOLAYER": "#ef5350"}.get(
                              state_txt, "#bdbdbd")
                line = f"navigation: {state_txt}"
                if exp.get("pose_um"):
                    line += (f"\npose ({exp['pose_um'][0]:.0f}, "
                             f"{exp['pose_um'][1]:.0f}) um")
                if exp.get("dist_um"):
                    line += f" | {exp['dist_um']:.0f} um from monolayer"
                self.exp_nav.setText(line)
                self.exp_nav.setStyleSheet(
                    f"color:{colour}; font-size:12px; font-weight:bold;")

        if self.scan_map is not None:
            if self.mosaic is not None:
                ov = self.mosaic.overview()
                if ov is not None:
                    rgb, mox, moy, msc = ov
                    self.scan_map_panel.set_mosaic_overview(rgb, mox, moy, msc)
            mono_rects = [(f.x - f.w / 2, f.y - f.h / 2,
                           f.x + f.w / 2, f.y + f.h / 2)
                          for f in self.scan_map.active_monolayer_footprints()]
            b_ver, boundary = self.scan_map.monolayer_boundary(
                cell_fraction=self.cfg.get("boundary_cell_fraction", 0.125),
                close_cells=int(self.cfg.get("boundary_close_cells", 2)),
                min_region_cells=int(self.cfg.get("boundary_min_region_cells", 4)),
            )
            # Phase 6: per-sweep trajectories (fields ordered by capture time)
            trails: Dict[int, list] = {}
            for fp in self.scan_map.fields():
                trails.setdefault(fp.sweep_id, []).append(
                    (fp.t_capture, fp.x, fp.y))
            sweep_trails = [
                [(x, y) for _, x, y in sorted(trail)] for trail in trails.values()
            ]
            frame_snap = snap.get("frame")
            if frame_snap is not None:
                self.scan_map_panel.set_fov((frame_snap.shape[1],
                                             frame_snap.shape[0]))
            cov = self.scan_map.coverage_summary(
                cell_fraction=float(self.cfg.get("boundary_cell_fraction", 0.125)),
                min_hole_cells=int(self.cfg.get("min_hole_cells", 2)))
            self.scan_map_panel.set_holes(
                [(hd["x"], hd["y"], hd["w"], hd["h"]) for hd in cov["holes"]])
            self._coverage_stats = cov
            # Phase 11: semantic green mask + boundary (cached per version)
            s_ver, sx0, sy0, s_cell, green, _conf_img, conflicts, s_cont =                 self.scan_map.semantic_grid(
                    cell_fraction=float(self.cfg.get("semantic_cell_fraction",
                                                     0.125)),
                    green_conf_thr=float(self.cfg.get("semantic_green_conf_thr",
                                                      0.5)),
                    min_votes=int(self.cfg.get("semantic_min_votes", 2)),
                    strong_single_thr=float(self.cfg.get(
                        "semantic_strong_single_thr", 0.75)),
                    monolayer_weight=float(self.cfg.get(
                        "semantic_monolayer_weight", 1.0)),
                    other_weight=float(self.cfg.get("semantic_other_weight",
                                                    1.0)))
            self.scan_map_panel.set_semantic(green if green is not None else
                                             np.zeros((2, 2), dtype=bool),
                                             sx0, sy0, s_cell, conflicts)
            self.scan_map_panel.set_boundary(
                s_ver, s_cont if s_cont else boundary)
            self.scan_map_panel.set_data(
                self.scan_map.fields(), mono_rects,
                self.state.current_net(),
                boundary_version=b_ver, boundary=boundary,
                sweeps=self.scan_map.sweeps(), sweep_trails=sweep_trails)
        cov_stats = getattr(self, "_coverage_stats", None)
        cov_line = ""
        if cov_stats:
            cov_line = (f" | coverage {cov_stats['coverage_pct'] * 100:.0f}%"
                        f" | unscanned gaps: {len(cov_stats['holes'])}")
        self.statusBar().showMessage(
            f"● {self.state_source} | capture {snap['capture_fps']:.1f} fps | "
            f"ui {self.ui_fps:.0f} fps | motion {snap['motion_state']} | "
            f"cellpose {cp} | analyses {snap['analysis_count']}" + cov_line
        )

    def _render_result_metrics(self, result, f: Dict, border: int,
                               stale: bool, age) -> None:
        """Sidebar metrics for one fully-analyzed result object."""
        colour = _QT_COLORS.get(result.raw_class, "#ececec")
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
                  f"job #{result.job_id} f{result.frame_idx} | {result.feature_mode} mode | "
                  f"cellpose {(result.cellpose_ms or 0) / 1000:.1f}s | "
                  f"features {(result.features_ms or 0) / 1000:.1f}s")

    BTN_STYLES = {
        "CAPTURE_READY": ("background:#2e7d32; color:white; font-size:14px; "
                          "font-weight:bold; padding:8px 18px; border:1px solid #1b5e20;"),
        "TOO_MUCH_OVERLAP": ("background:#92400e; color:white; font-size:14px; "
                             "font-weight:bold; padding:8px 18px; border:1px solid #f59e0b;"),
        "GAP_RISK": ("background:#7f1d1d; color:white; font-size:14px; "
                     "font-weight:bold; padding:8px 18px; border:1px solid #b91c1c;"),
        "ALREADY_SCANNED": ("background:#37474f; color:white; font-size:14px; "
                            "font-weight:bold; padding:8px 18px; border:1px solid #607d8b;"),
    }

    def _apply_capture_button_style(self, pv_state: str) -> None:
        style = self.BTN_STYLES.get(pv_state)
        if style:
            self.btn_capture.setStyleSheet(style)

    NAV_BANNER = {
        "OUTSIDE_MONOLAYER":   ("OUTSIDE MONOLAYER", "#7f1d1d", "#b91c1c"),
        "ENTERING_MONOLAYER":  ("ENTERING MONOLAYER ...", "#92400e", "#f59e0b"),
        "IN_MONOLAYER":        ("✓ CURRENTLY IN MONOLAYER", "#14532d", "#22c55e"),
        "LEAVING_MONOLAYER":   ("⚠ LEAVING MONOLAYER", "#9a3412", "#f97316"),
    }

    def _update_nav_banner(self) -> None:
        text, bg, border = self.NAV_BANNER.get(
            self.state.nav_state,
            ("OUTSIDE MONOLAYER", "#7f1d1d", "#b91c1c"))
        score = self.state.nav_score
        if score is not None:
            text += f"    score {score:.2f}   (experimental)"
        self.banner.setText(text)
        self.banner.setStyleSheet(
            f"background:{bg}; color:#ffffff; font-size:18px; font-weight:bold;"
            f"padding:8px; border:2px solid {border};"
        )

    def _set(self, key: str, value) -> None:
        self.metrics[key].setText(str(value))

    def _refresh_job_list(self) -> None:
        if self.jobmgr is None:
            return
        rows = self.jobmgr.snapshot_jobs(10)
        signature = "|".join(
            f"{r['job_id']}:{r['status']}:{r['raw_class']}:{r['human_label']}"
            for r in rows)
        if signature == getattr(self, "_job_sig", None):
            return
        self._job_sig = signature
        self.job_list.clear()
        for r in reversed(rows):  # newest first
            status = r["status"]
            if status == "COMPLETE":
                text = (f"#{r['job_id']:03d}  ✓ {r['raw_class']}   "
                        f"score {r['score']:.2f}"
                        + (f"  [human: {r['human_label']}]" if r["human_label"] else ""))
                colour = _QT_COLORS.get(r["raw_class"] or "", "#d6d6d6")
            elif status == "FAILED":
                text = f"#{r['job_id']:03d}  FAILED - {r['error']}"
                colour = "#d32f2f"
            else:
                text = f"#{r['job_id']:03d}  {status}  [{r['priority']}]"
                colour = "#9e9e9e"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, r["job_id"])
            item.setForeground(self._qt_colour(colour))
            self.job_list.addItem(item)

    @staticmethod
    def _qt_colour(hex_colour: str):
        from PyQt6.QtGui import QColor

        return QColor(hex_colour)

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
            if self.scan_map is not None:
                self.scan_map.reset()
            if self.nav_machine is not None:
                self.nav_machine.reset()
            if self.tracker is not None:
                self.tracker.reset()
            if self.mosaic is not None:
                self.mosaic.reset()
            with st.lock:
                st.nav_state = "OUTSIDE_MONOLAYER"
                st.nav_score = None
                st.nav_transitions.clear()
        elif key == "g":
            with st.lock:
                st.recording = not st.recording
        elif key in ("m", "t", "n", "u"):
            if self.session is None:
                return
            # label the SELECTED completed job when one is chosen (spec §29);
            # otherwise label the current live frame as before
            job = self._selected_completed_job()
            if job is not None:
                self.session.save_label(key, job.frame, job)
                if self.scan_map is not None:
                    self.scan_map.set_human_label(job.job_id, key.upper())
            elif hasattr(self, "_latest"):
                self.session.save_label(key, self._latest["frame"],
                                        self._latest["snap"]["result"])
            self.statusBar().showMessage(
                f"human label '{key.upper()}' saved", 2000)
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev):  # noqa: N802 (Qt naming)
        self.state.quit_event.set()
        super().closeEvent(ev)
