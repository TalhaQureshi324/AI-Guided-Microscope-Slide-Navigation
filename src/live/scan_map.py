"""Persistent scan-coordinate map (Phase 1).

Every analyzed field is stored as a spatial FOOTPRINT - an (x, y, w, h)
rectangle centred on the field's estimated scan position - plus its score,
class, confidence and human label. Rectangles (not dots) so neighbouring
analyzed areas can overlap spatially, which is what the later monolayer-band
reconstruction (Phases 4-6) needs.

Coordinate system: the existing cumulative image-registration displacement
(net summed dx, dy in full-resolution pixels), shared by all jobs. It is an
ESTIMATE; the architecture deliberately hides it behind this module so real
microscope stage coordinates can replace it later without touching the
mapping code or the GUI (just swap the position source).

The map persists for the whole session until the user presses Reset Scan
(R); it is also saved to <session>/scan_map.json after every added field so
a crash never loses the survey.
"""

from __future__ import annotations

import json
import math

import numpy as np
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CLASS_COLORS = {
    "TOO_THICK": "#d32f2f",
    "MONOLAYER": "#2e7d32",
    "TOO_THIN": "#1565c0",
    "UNCERTAIN": "#ef6c00",
}


@dataclass
class FieldFootprint:
    """One analyzed field's spatial footprint + result (spec Phase 1)."""

    job_id: int
    t_capture: float                     # session clock
    x: float                             # scan position (net cumulative dx)
    y: float                             # scan position (net cumulative dy)
    w: float                             # footprint width  (frame width in scan px)
    h: float                             # footprint height (frame height in scan px)
    score: float                         # Monolayer Score (confidence proxy 0..1)
    raw_class: str                       # TOO_THICK / MONOLAYER / TOO_THIN / UNCERTAIN
    priority: str = "AUTO"
    frame_idx: int = -1
    source_image: str = ""               # relative path of the saved keyframe
    human_label: Optional[str] = None    # M/T/N/U when the operator labeled it
    # persistent monolayer-layer evidence (Phase 3): accumulated over
    # overlapping later fields - the layer is conservative evidence, not the
    # last frame (spec: contradictions do not immediately erase)
    ml_agreements: int = 0               # overlapping later MONOLAYER fields
    ml_disagreements: int = 0            # overlapping later contradicting fields
    ml_demoted: bool = False             # removed from the active green layer
    sweep_id: int = 0                    # which sweep trajectory captured this
    reg_method: str = ""                 # FIRST_FIELD / PHASE_CORR / ORB_TRANSLATION / MAP_FIELD_SNAP
    reg_state: str = ""                  # REGISTRATION_GOOD / WEAK / FAILED
    reg_confidence: float = 0.0          # 0..1
    reg_dx: float = 0.0                  # local translation from the reference field
    reg_dy: float = 0.0
    reg_ref_job_id: int = -1             # reference field used for registration
    drift_warning: bool = False          # revisit disagreed with chained pose


class ScanMap:
    """Thread-safe registry of analyzed field footprints."""

    def __init__(self, cfg: Optional[Dict] = None) -> None:
        cfg = cfg or {}
        self.cfg = cfg                     # full config: nav enter/exit etc.
        navcfg = cfg.get("navigation", {})
        self.enter_votes = max(1, int(navcfg.get("enter_consecutive_fields", 2)))
        self._lock = threading.Lock()
        self._fields: List[FieldFootprint] = []
        self._by_job: Dict[int, FieldFootprint] = {}
        self._version = 0                 # bumps on every survey change
        self._boundary_cache: Dict[tuple, tuple] = {}
        self._coverage_cache: Dict[tuple, tuple] = {}
        self._semantic_cache: Dict[tuple, tuple] = {}
        # Phase 11: persistent per-cell hysteresis state (True = green)
        self._cell_green: Dict[Tuple[int, int], bool] = {}
        self._cell_geom: Optional[Tuple[float, float, float]] = None
        # Phase 6: multi-sweep trajectories in ONE map coordinate system
        self._sweeps: List[Dict] = []     # Sweep dicts (id, direction, start...)
        self._current_sweep: Optional[Dict] = None
        self._sweep_offset: Tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Phase 6: multi-sweep trajectories (same map, anchored new sweeps)
    # ------------------------------------------------------------------
    def start_sweep(self, t: float, current_x: float, current_y: float,
                    anchor_job_id: Optional[int] = None,
                    row_direction: Optional[str] = None) -> Optional[Dict]:
        """Begin a new sweep trajectory on the SAME map.

        The new sweep's starting position equals the current live position
        (no offset - the stage physically continued). If ``anchor_job_id`` is
        given, the operator declares "the microscope is now at that field's
        location": the sweep origin is offset so subsequent positions land
        correctly in the existing map. The monolayer layer is NEVER reset.
        """
        with self._lock:
            offset = (0.0, 0.0)
            if anchor_job_id is not None:
                fp = self._by_job.get(anchor_job_id)
                if fp is not None:
                    offset = (fp.x - current_x, fp.y - current_y)
            self._sweep_offset = offset
            sweep = {
                "sweep_id": len(self._sweeps) + 1,
                "row": len(self._sweeps) + 1,
                "row_direction": row_direction or "-",
                "started_at": t,
                "direction": "-",
                "start_x": round(current_x + offset[0], 1),
                "start_y": round(current_y + offset[1], 1),
                "anchor_job_id": anchor_job_id,
                "n_fields": 0,
            }
            self._sweeps.append(sweep)
            self._current_sweep = sweep
            self._version += 1
            return sweep

    def current_offset(self) -> Tuple[float, float]:
        with self._lock:
            return self._sweep_offset

    def sweeps(self) -> List[Dict]:
        with self._lock:
            return [dict(sw) for sw in self._sweeps]

    def _note_field_for_sweep(self, job) -> None:
        """Attach the field to the current sweep + auto-detect direction."""
        sweep = self._current_sweep
        if sweep is None:
            sweep = {"sweep_id": 0, "started_at": job.t_capture, "direction": "-",
                     "start_x": round(job.map_x, 1), "start_y": round(job.map_y, 1),
                     "anchor_job_id": None, "n_fields": 0}
            self._sweeps.append(sweep)
            self._current_sweep = sweep
        sweep["n_fields"] += 1
        mx = getattr(job, "map_x", job.cumulative_x)
        my = getattr(job, "map_y", job.cumulative_y)
        prev = sweep.get("_last")
        if prev is not None:
            dx, dy = mx - prev[0], my - prev[1]
            if math.hypot(dx, dy) > 10:  # ignore jitter
                sweep["direction"] = ("horizontal" if abs(dx) >= abs(dy)
                                      else "vertical")
        sweep["_last"] = (mx, my)

    # ------------------------------------------------------------------
    def add_field(self, job, source_image: str = "") -> Optional[FieldFootprint]:
        """Record a completed analysis job as a footprint.

        ``job`` is an AnalysisJob (has cumulative_x/y, labels, score, class).
        The footprint is the captured frame's rectangle centred on the field's
        net scan position - neighbouring fields genuinely overlap.
        """
        if job.labels is None or job.score is None:
            return None
        h, w = job.labels.shape[:2]
        self._note_field_for_sweep(job)
        current_sweep_id = (self._current_sweep or {}).get("sweep_id", 0)
        fp = FieldFootprint(
            job_id=job.job_id,
            t_capture=job.t_capture,
            x=float(getattr(job, "map_x", job.cumulative_x)),
            y=float(getattr(job, "map_y", job.cumulative_y)),
            w=float(w),
            h=float(h),
            score=float(job.score),
            raw_class=job.raw_class or "UNCERTAIN",
            priority=job.priority,
            frame_idx=job.frame_idx,
            source_image=source_image,
            human_label=job.human_label,
            sweep_id=current_sweep_id,
            reg_method=getattr(job, "reg_method", ""),
            reg_state=getattr(job, "reg_state", ""),
            reg_confidence=float(getattr(job, "reg_confidence", 0.0) or 0.0),
            reg_dx=float(getattr(job, "reg_dx", 0.0) or 0.0),
            reg_dy=float(getattr(job, "reg_dy", 0.0) or 0.0),
            reg_ref_job_id=int(getattr(job, "reg_ref_job_id", -1) or -1),
            drift_warning=bool(getattr(job, "drift_warning", False)),
        )
        with self._lock:
            self._fields.append(fp)
            self._by_job[fp.job_id] = fp
            self._version += 1
        return fp

    def set_human_label(self, job_id: int, label: str) -> bool:
        with self._lock:
            fp = self._by_job.get(job_id)
            if fp is None:
                return False
            fp.human_label = label
            return True

    def fields(self) -> List[FieldFootprint]:
        with self._lock:
            return list(self._fields)

    def active_fields(self) -> List[FieldFootprint]:
        with self._lock:
            return list(self._fields)

    def monolayer_fields(self) -> List[FieldFootprint]:
        with self._lock:
            return [f for f in self._fields if f.raw_class == "MONOLAYER"]

    # ------------------------------------------------------------------
    # Phase 3: persistent monolayer LAYER (accumulated evidence)
    # ------------------------------------------------------------------
    def update_monolayer_layer(self, job, contradictions_to_demote: int = 2,
                               overlap_min_fraction: float = 0.25,
                               human_label_weight: int = 2) -> Dict:
        """Fold one completed field into the monolayer evidence layer.

        * a MONOLAYER field re-affirms overlapping layer footprints and is
          added to the layer unless already ~fully covered by it;
        * a contradicting field adds disagreement to overlapping footprints;
          a footprint is demoted only after >= ``contradictions_to_demote``
          contradictions AND more contradictions than agreements
          (conservative: no single frame erases territory, spec Phase 3);
        * human labels weigh ``human_label_weight`` (operator evidence).
        The raw per-field footprints are never modified by this - only the
        evidence counters above.
        """
        if job.labels is None or job.raw_class is None:
            return {"added": False, "reason": "no result"}
        h, w = job.labels.shape[:2]
        mx = getattr(job, "map_x", job.cumulative_x)
        my = getattr(job, "map_y", job.cumulative_y)
        nx0, ny0 = mx - w / 2.0, my - h / 2.0
        nx1, ny1 = nx0 + w, ny0 + h
        is_mono = job.raw_class == "MONOLAYER"
        weight = human_label_weight if job.human_label else 1

        with self._lock:
            summary = {"added": False, "affirmed": 0, "contradicted": 0,
                       "demoted": 0, "promoted": 0}
            for fp in self._fields:
                if fp.raw_class != "MONOLAYER" or fp.job_id == job.job_id:
                    continue
                ox0, oy0 = max(nx0, fp.x - fp.w / 2), max(ny0, fp.y - fp.h / 2)
                ox1, oy1 = min(nx1, fp.x + fp.w / 2), min(ny1, fp.y + fp.h / 2)
                if ox1 <= ox0 or oy1 <= oy0:
                    continue  # no overlap
                inter = (ox1 - ox0) * (oy1 - oy0)
                smaller = min(w * h, fp.w * fp.h)
                if smaller > 0 and inter / smaller < overlap_min_fraction:
                    continue  # touching, not meaningfully overlapping

                changed = False
                if is_mono:
                    fp.ml_agreements += weight
                    summary["affirmed"] += 1
                    if fp.ml_demoted and (fp.ml_agreements > fp.ml_disagreements):
                        fp.ml_demoted = False  # evidence restored the area
                        summary["promoted"] += 1
                    changed = True
                else:
                    fp.ml_disagreements += weight
                    summary["contradicted"] += 1
                    if (not fp.ml_demoted
                            and fp.ml_disagreements >= contradictions_to_demote
                            and fp.ml_disagreements > fp.ml_agreements):
                        fp.ml_demoted = True  # conservative: repeated majority only
                        summary["demoted"] += 1
                        changed = True
            if any(summary.values()):
                self._version += 1

            if is_mono:
                covered = self._covered_fraction_locked(
                    (nx0, ny0, nx1, ny1),
                    [f for f in self._fields
                     if f.raw_class == "MONOLAYER" and not f.ml_demoted
                     and f.job_id != job.job_id])
                if covered < 0.8:  # mostly new territory -> join the layer
                    fp = self._by_job.get(job.job_id)
                    if fp is not None:
                        fp.ml_agreements = max(fp.ml_agreements, 1)
                        summary["added"] = True
            return summary

    def _covered_fraction_locked(self, rect, others) -> float:
        x0, y0, x1, y1 = rect
        area = max(1.0, (x1 - x0) * (y1 - y0))
        covered = 0.0
        for f in others:
            ox0, oy0 = max(x0, f.x - f.w / 2), max(y0, f.y - f.h / 2)
            ox1, oy1 = min(x1, f.x + f.w / 2), min(y1, f.y + f.h / 2)
            if ox1 > ox0 and oy1 > oy0:
                covered += (ox1 - ox0) * (oy1 - oy0)
        return min(1.0, covered / area)

    def active_monolayer_footprints(self) -> List[FieldFootprint]:
        """The persistent green layer: MONOLAYER footprints not demoted."""
        with self._lock:
            return [f for f in self._fields
                    if f.raw_class == "MONOLAYER" and not f.ml_demoted]

    # ------------------------------------------------------------------
    # Phase 10: coverage grid (SCANNED / LOW_CONFIDENCE / UNSCANNED) + holes
    # ------------------------------------------------------------------
    def coverage_grid(self, cell_fraction: float = 0.125,
                      low_conf_thr: float = 0.5, force: bool = False) -> tuple:
        """Rasterized observation grid over the union of registered fields.

        Returns (version, origin_x, origin_y, cell, grid, coverage_pct):
          grid[y, x] = 2 SCANNED (covered, best reg conf >= low_conf_thr)
                     = 1 LOW_CONFIDENCE (covered, best conf < threshold)
                     = 0 UNSCANNED
        Cached per survey version + params.
        """
        key = ("coverage", round(cell_fraction, 4), round(low_conf_thr, 2))
        cell = max(1.0, float(np.median(
            [f.w for f in self._fields])) * cell_fraction) if self._fields             else float(800 * cell_fraction)
        with self._lock:
            if not force:
                cached = self._coverage_cache.get(key)
                if cached is not None and cached[0] == self._version:
                    return cached
            fields = list(self._fields)
            version = self._version
        if not fields:
            with self._lock:
                self._coverage_cache[key] = (version, 0.0, 0.0, cell, None, 0.0)
            return version, 0.0, 0.0, cell, None, 0.0

        pad = 1
        x0 = min(f.x - f.w / 2 for f in fields) - pad * cell
        y0 = min(f.y - f.h / 2 for f in fields) - pad * cell
        x1 = max(f.x + f.w / 2 for f in fields) + pad * cell
        y1 = max(f.y + f.h / 2 for f in fields) + pad * cell
        gw = max(1, int(math.ceil((x1 - x0) / cell)))
        gh = max(1, int(math.ceil((y1 - y0) / cell)))
        best_conf = np.zeros((gh, gw), dtype=np.float64)
        for f in fields:
            conf = 1.0 if f.reg_state == "REGISTRATION_GOOD" else (
                float(f.reg_confidence) if f.reg_state == "REGISTRATION_WEAK" else 0.0)
            if conf <= 0:
                continue
            cx0 = max(0, int(math.floor((f.x - f.w / 2 - x0) / cell)))
            cy0 = max(0, int(math.floor((f.y - f.h / 2 - y0) / cell)))
            cx1 = min(gw, int(math.ceil((f.x + f.w / 2 - x0) / cell)))
            cy1 = min(gh, int(math.ceil((f.y + f.h / 2 - y0) / cell)))
            best_conf[cy0:cy1, cx0:cx1] = np.maximum(best_conf[cy0:cy1, cx0:cx1], conf)
        grid = np.zeros((gh, gw), dtype=np.uint8)
        grid[best_conf > 0] = 1
        grid[best_conf >= low_conf_thr] = 2
        # coverage over the survey bbox only: the 1-cell padded ring is
        # outside the scan and would otherwise dilute the percentage
        inner = grid[pad:gh - pad, pad:gw - pad]
        scanned_pct = (float((inner > 0).sum()) / float(inner.size)
                       if inner.size else 0.0)
        with self._lock:
            self._coverage_cache[key] = (version, x0, y0, cell, grid, scanned_pct)
        return version, x0, y0, cell, grid, scanned_pct

    def coverage_summary(self, cell_fraction: float = 0.125,
                         low_conf_thr: float = 0.5,
                         min_hole_cells: int = 2) -> Dict:
        """Coverage stats + interior UNSCANNED holes (spec Phase 10 §1/§2).

        Holes = unscanned grid components fully surrounded by scanned
        territory (components touching the padded outer border are outside
        the survey, not holes). Never filled or interpolated semantically.
        """
        version, x0, y0, cell, grid, scanned_pct = self.coverage_grid(
            cell_fraction, low_conf_thr)
        if grid is None:
            return {"version": version, "coverage_pct": 0.0, "holes": []}
        from scipy import ndimage
        unscanned = grid == 0
        # the padded ring is outside by construction; drop it from hole search
        inner = np.zeros_like(unscanned)
        inner[1:-1, 1:-1] = unscanned[1:-1, 1:-1]
        labelled, n = ndimage.label(inner)
        holes = []
        if n:
            sizes = ndimage.sum_labels(inner, labelled, range(1, n + 1))
            for i in range(1, n + 1):
                if sizes[i - 1] < min_hole_cells:
                    continue
                ys, xs = np.where(labelled == i)
                holes.append({
                    "area_cells": int(sizes[i - 1]),
                    "x": round(x0 + (float(xs.min()) + 0.5) * cell, 1),
                    "y": round(y0 + (float(ys.min()) + 0.5) * cell, 1),
                    "w": round(float(xs.max() - xs.min() + 1) * cell, 1),
                    "h": round(float(ys.max() - ys.min() + 1) * cell, 1),
                })
        holes.sort(key=lambda hd: -hd["area_cells"])
        total = float(grid.size)
        scanned = float((grid > 0).sum())
        return {"version": version, "coverage_pct": round(scanned / total, 4),
                "holes": holes, "cell": cell}

    def semantic_grid(self, cell_fraction: float = 0.125,
                      green_conf_thr: float = 0.5,
                      min_votes: int = 2,
                      monolayer_weight: float = 1.0,
                      other_weight: float = 1.0,
                      strong_single_thr: float = 0.75,
                      force: bool = False) -> tuple:
        """Phase 11: cell-level monolayer confidence map + green boundary.

        Every REGISTERED field contributes evidence to each grid cell it
        covers, weighted by registration confidence and observation type:
          mono_sum  += monolayer_weight * field.score   if class == MONOLAYER
          weight    += other_weight                     otherwise
        monolayer_confidence = mono_sum / weight  (0..1 per cell).

        GREEN mask = cells with >= min_votes observations AND
        monolayer_confidence >= green_conf_thr - one borderline field can
        never paint a huge green area (cell-level hysteresis), and evidence
        fusion is configurable (spec Phase 11 §2/§3).

        Conflicting cells (both green and non-green observations) are
        returned separately and flagged CONFLICTING EVIDENCE, never erased.

        Also returns the outer boundary contours of the connected green
        region (same extraction as Phase 4). Cached per survey version.
        Returns (version, ox, oy, cell, green, conf_img, conflicts, contours).
        """
        key = ("semantic", round(cell_fraction, 4), round(green_conf_thr, 2),
               min_votes, round(monolayer_weight, 2), round(other_weight, 2))
        with self._lock:
            if not force:
                cached = self._semantic_cache.get(key)
                if cached is not None and cached[0] == self._version:
                    return cached
            fields = list(self._fields)
            version = self._version
        if not fields:
            with self._lock:
                self._semantic_cache[key] = (version, 0.0, 0.0, 0.0, None, None, [], [])
            return version, 0.0, 0.0, 0.0, None, None, [], []

        import cv2
        from scipy import ndimage
        cell = max(1.0, float(np.median([f.w for f in fields])) * cell_fraction)
        pad = 1
        x0 = min(f.x - f.w / 2 for f in fields) - pad * cell
        y0 = min(f.y - f.h / 2 for f in fields) - pad * cell
        x1 = max(f.x + f.w / 2 for f in fields) + pad * cell
        y1 = max(f.y + f.h / 2 for f in fields) + pad * cell
        gw = max(1, int(math.ceil((x1 - x0) / cell)))
        gh = max(1, int(math.ceil((y1 - y0) / cell)))
        mono_sum = np.zeros((gh, gw), dtype=np.float64)
        weight = np.zeros((gh, gw), dtype=np.float64)
        votes = np.zeros((gh, gw), dtype=np.float64)
        mono_votes = np.zeros((gh, gw), dtype=np.float64)
        non_votes = np.zeros((gh, gw), dtype=np.float64)
        for f in fields:
            conf_w = max(0.1, min(1.0, f.reg_confidence)) if f.reg_confidence else 0.5
            fx0 = max(0, int(math.floor((f.x - f.w / 2 - x0) / cell)))
            fy0 = max(0, int(math.floor((f.y - f.h / 2 - y0) / cell)))
            fx1 = min(gw, int(math.ceil((f.x + f.w / 2 - x0) / cell)))
            fy1 = min(gh, int(math.ceil((f.y + f.h / 2 - y0) / cell)))
            if fx1 <= fx0 or fy1 <= fy0:
                continue
            is_mono = f.raw_class == "MONOLAYER"
            if is_mono:
                mono_sum[fy0:fy1, fx0:fx1] += monolayer_weight * conf_w * float(f.score)
                mono_votes[fy0:fy1, fx0:fx1] += monolayer_weight * conf_w
                weight[fy0:fy1, fx0:fx1] += monolayer_weight * conf_w
            else:
                weight[fy0:fy1, fx0:fx1] += other_weight * conf_w
                non_votes[fy0:fy1, fx0:fx1] += other_weight * conf_w
            votes[fy0:fy1, fx0:fx1] += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            conf_img = np.where(weight > 0, mono_sum / np.maximum(weight, 1e-9), 0.0)

        # ---- cell-level hysteresis (spec Phase 11 §3): mirrors the
        # navigation state machine - ENTER at enter_score with enough
        # support, LEAVE at the lower exit threshold; a strongly confirmed
        # single field may enter alone, one borderline field may not.
        enter_thr = float(self.cfg.get("navigation", {}).get(
            "enter_score", 0.65))
        exit_thr = float(self.cfg.get("navigation", {}).get(
            "exit_score", 0.45))
        strong_single = float(self.cfg.get("strong_single_thr", 0.75))

        geom = (x0, y0, cell)
        if self._cell_geom != geom:
            self._cell_green.clear()   # grid geometry changed: state void
            self._cell_geom = geom
        state_arr = np.zeros((gh, gw), dtype=bool)
        for (r, c), was_green in list(self._cell_green.items()):
            if 0 <= r < gh and 0 <= c < gw:
                state_arr[r, c] = was_green

        observed = votes > 0
        entered = observed & (conf_img >= max(enter_thr, strong_single))
        entered |= observed & (conf_img >= enter_thr) &             ((mono_votes >= self.enter_votes) | (non_votes == 0) &
             (conf_img >= strong_single))
        left = observed & (conf_img < exit_thr)
        state_arr = np.where(entered & ~left, True,
                             np.where(left, False, state_arr))
        self._cell_green.clear()
        for r, c in zip(*np.where(observed)):
            self._cell_green[(int(r), int(c))] = bool(state_arr[r, c])

        green = state_arr
        green = ndimage.binary_closing(green, structure=np.ones((3, 3), dtype=bool))
        conflict = observed & (conf_img < enter_thr) & (mono_votes > 0) &                    (non_votes > 0)

        # outer contours of the connected green region -> scan coordinates
        green_u8 = green.astype(np.uint8) * 255
        contours_scan: List[List[Tuple[float, float]]] = []
        found = cv2.findContours(green_u8, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        for cnt in found[0] if isinstance(found, tuple) else found:
            pts = [(x0 + (float(pt[0][0]) + 0.5) * cell,
                    y0 + (float(pt[0][1]) + 0.5) * cell) for pt in cnt]
            if len(pts) >= 3:
                contours_scan.append(pts)

        conflicts = []
        cys, cxs = np.where(conflict)
        for cy, cx in zip(cys, cxs):
            conflicts.append({"x": round(x0 + (cx + 0.5) * cell, 1),
                              "y": round(y0 + (cy + 0.5) * cell, 1),
                              "conf": round(float(conf_img[cy, cx]), 3)})

        with self._lock:
            out = (version, x0, y0, cell, green, conf_img, conflicts,
                   contours_scan)
            self._semantic_cache[key] = out
            return out

    def invalidate_caches(self) -> None:
        with self._lock:
            self._version += 1
            self._boundary_cache.clear()
            self._coverage_cache.clear()
            self._semantic_cache.clear()
            self._semantic_cache.clear()

    # ------------------------------------------------------------------
    # Phase 4: continuous monolayer region + outer boundary
    # ------------------------------------------------------------------
    def monolayer_boundary(self, cell_fraction: float = 0.125,
                           close_cells: int = 2,
                           min_region_cells: int = 4,
                           force: bool = False) -> tuple:
        """Outer contour(s) of the accumulated monolayer region.

        Pipeline: spatial confidence grid over the active monolayer layer
        (each cell accumulates the evidence confidence of covering fields)
        -> binary high-confidence mask -> modest cleanup (small-gap closing,
        tiny-island removal; no aggressive reshaping) -> cv2 outer contours
        converted back to scan coordinates.

        Cached on (version, params); recomputes only when the survey changes,
        so the boundary expands incrementally as scanning continues.
        Returns (version, contours) with contours = list of [(x, y), ...].
        """
        key = ("boundary", round(cell_fraction, 4), close_cells, min_region_cells)
        with self._lock:
            if not force:
                cached = self._boundary_cache.get(key)
                if cached is not None and cached[0] == self._version:
                    return cached
            fields = [f for f in self._fields
                      if f.raw_class == "MONOLAYER" and not f.ml_demoted]
            version = self._version
        if not fields:
            with self._lock:
                self._boundary_cache[key] = (version, [])
            return version, []

        import cv2
        from scipy import ndimage

        cell = max(1.0, float(np.median([f.w for f in fields])) * cell_fraction)
        min_x = min(f.x - f.w / 2 for f in fields) - cell
        min_y = min(f.y - f.h / 2 for f in fields) - cell
        max_x = max(f.x + f.w / 2 for f in fields) + cell
        max_y = max(f.y + f.h / 2 for f in fields) + cell
        gw = max(1, int(round((max_x - min_x) / cell)))
        gh = max(1, int(round((max_y - min_y) / cell)))

        conf = np.zeros((gh, gw), dtype=np.float64)
        weight = np.zeros((gh, gw), dtype=np.float64)
        for f in fields:
            f_conf = ((f.ml_agreements + f.score)
                      / (f.ml_agreements + f.ml_disagreements + 1.0))
            cx0 = int(max(0, (f.x - f.w / 2 - min_x) / cell))
            cy0 = int(max(0, (f.y - f.h / 2 - min_y) / cell))
            cx1 = int(min(gw, math.ceil((f.x + f.w / 2 - min_x) / cell)))
            cy1 = int(min(gh, math.ceil((f.y + f.h / 2 - min_y) / cell)))
            conf[cy0:cy1, cx0:cx1] += f_conf
            weight[cy0:cy1, cx0:cx1] += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            mono_conf = np.where(weight > 0, conf / np.maximum(weight, 1e-9), 0.0)

        mask = (weight > 0) & (mono_conf >= 0.5)
        # modest cleanup: small-gap closing + tiny-island removal ONLY
        if close_cells > 0:
            struct = np.ones((2 * close_cells + 1, 2 * close_cells + 1), dtype=bool)
            mask = ndimage.binary_closing(mask, structure=struct)
        labelled, n = ndimage.label(mask)
        if n:
            sizes = ndimage.sum_labels(mask, labelled, range(1, n + 1))
            keep = np.zeros_like(mask)
            for i in range(1, n + 1):
                if sizes[i - 1] >= min_region_cells:
                    keep |= labelled == i
            mask = keep

        contours_scan: List[List[Tuple[float, float]]] = []
        # findContours traces the outermost FOREGROUND pixels, which sit one
        # grid cell inside the true region edge; a 1-cell dilation puts the
        # traced ring back on the region boundary (documented discretization:
        # boundary fidelity is ~0.5 cell).
        mask_d = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))
        u8 = mask_d.astype(np.uint8) * 255
        found = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in found[0] if isinstance(found, tuple) else found:
            pts = [(min_x + (float(pt[0][0]) + 0.5) * cell,
                    min_y + (float(pt[0][1]) + 0.5) * cell) for pt in cnt]
            if len(pts) >= 3:
                contours_scan.append(pts)
        with self._lock:
            self._boundary_cache[key] = (version, contours_scan)
        return version, contours_scan

    def reset(self) -> None:
        """Reset Scan (hotkey R) clears the whole survey INCLUDING sweeps."""
        with self._lock:
            self._fields.clear()
            self._by_job.clear()
            self._sweeps.clear()
            self._current_sweep = None
            self._sweep_offset = (0.0, 0.0)
            self._version += 1
            self._boundary_cache.clear()
            self._coverage_cache.clear()
            self._semantic_cache.clear()
            self._semantic_cache.clear()

    # ------------------------------------------------------------------
    def bounds(self, extra_points: Optional[List[Tuple[float, float]]] = None
               ) -> Tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) over all footprints + extra points."""
        with self._lock:
            pts: List[Tuple[float, float]] = []
            for f in self._fields:
                pts.extend([
                    (f.x - f.w / 2.0, f.y - f.h / 2.0),
                    (f.x + f.w / 2.0, f.y + f.h / 2.0),
                ])
        for px, py in (extra_points or []):
            pts.append((px, py))
        if not pts:
            return (0.0, 0.0, 1.0, 1.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def save(self, path: Path) -> None:
        with self._lock:
            data = {
                "coordinate_system": "cumulative_image_registration (net dx, dy, px)",
                "saved_at": time.time(),
                "fields": [asdict(f) for f in self._fields],
                "sweeps": [dict(sw) for sw in self._sweeps],
                "sweep_offset": list(self._sweep_offset),
                "current_sweep_id": (self._current_sweep or {}).get("sweep_id"),
            }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ScanMap":
        m = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for d in data.get("fields", []):
            fp = FieldFootprint(**d)
            with m._lock:
                m._fields.append(fp)
                m._by_job[fp.job_id] = fp
        m._sweeps = list(data.get("sweeps", []))
        off = data.get("sweep_offset", [0.0, 0.0])
        m._sweep_offset = (off[0], off[1])
        cur = data.get("current_sweep_id")
        m._current_sweep = next((sw for sw in m._sweeps
                                 if sw.get("sweep_id") == cur), None)
        return m


# ---------------------------------------------------------------------- geometry
def union_rects(rects: List[Tuple[float, float, float, float]]
                ) -> List[Tuple[float, float, float, float]]:
    """Disjoint union of axis-aligned rects (x0, y0, x1, y1) via y-scanline.

    Overlapping monolayer footprints therefore merge into one visual region
    instead of dozens of separate rectangles (spec Phase 3).
    """
    if not rects:
        return []
    ys = sorted({r[1] for r in rects} | {r[3] for r in rects})
    out: List[Tuple[float, float, float, float]] = []
    for y0, y1 in zip(ys, ys[1:]):
        if y1 - y0 < 1e-9:
            continue
        ivs = sorted((r[0], r[2]) for r in rects if r[1] <= y0 and r[3] >= y1)
        merged: List[List[float]] = []
        for x0, x1 in ivs:
            if merged and x0 <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], x1)
            else:
                merged.append([x0, x1])
        for x0, x1 in merged:
            out.append((x0, y0, x1, y1))
    return out


def point_in_any(x: float, y: float,
                 rects: List[Tuple[float, float, float, float]]) -> bool:
    return any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for r in rects)
