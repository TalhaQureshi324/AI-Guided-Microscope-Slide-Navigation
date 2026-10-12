"""Finalized survey reference + high-magnification exploration (Phase 13).

Workflow (spec Phase 13): survey/map at LOW magnification -> FINALIZE SURVEY
MAP -> switch to a higher objective -> revisit only positions inside the
mapped monolayer ROI.

SurveyReference (§1)
--------------------
Immutable snapshot of everything Phases 0-12 accumulated on one slide: RGB
mosaic, coverage, monolayer mask + confidence grid, per-field placements and
transforms, calibration and session metadata. Stored as a self-contained
bundle (``survey_final/``) that survives process restarts. The frozen survey
is the reference frame; it is never modified by exploration.

Calibration (§2/§3)
-------------------
``pixels_per_micron`` per objective, from configuration. Physical position =
scan px x pixels_per_micron. Changing objectives changes microns-per-pixel,
but PHYSICAL slide coordinates stay valid - that is what makes the low-mag
monolayer map reusable at high magnification (§4): only the scale of a field
of view changes, never the map.

Monolayer-only navigation (§5)
------------------------------
containment() -> INSIDE / NEAR EDGE / OUTSIDE relative to the finalized
monolayer mask, with a guidance vector toward the nearest mapped monolayer
cell when outside.

Relocalization (§7)
-------------------
Scale-matches a live high-mag frame down to the reference pixel size and
template-matches it against the frozen mosaic. Best-effort by design: it
works only while the slide keeps its orientation and illumination stays
compatible. Failures are REPORTED clearly (§8) instead of guessed away.
Long-term answer: stage encoders, not image registration.

Revisit points (§6)
-------------------
generate_revisit_points(): centers / evenly spaced grid / top-confidence
cells inside the finalized monolayer region, plus user-selected points.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

INSIDE = "INSIDE MAPPED MONOLAYER"
NEAR_EDGE = "NEAR MONOLAYER EDGE"
OUTSIDE = "OUTSIDE MAPPED MONOLAYER"
UNKNOWN = "POSITION UNKNOWN (relocalization failed)"


# ---------------------------------------------------------------------------
# Calibration (spec §3)
# ---------------------------------------------------------------------------
class Calibration:
    """Pixels-per-micron for one objective/setup.

    Values are config-supplied; ``source`` records whether they were
    calibrated with a stage micrometer or estimated from the magnification
    ratio (absolute accuracy only matters for reported microns - the map
    math below uses scan px internally and ratios between objectives, which
    are exact even with estimated absolute scales).
    """

    def __init__(self, objective: str, pixels_per_micron: float,
                 source: str = "configured", calibrated_at: str = "") -> None:
        self.objective = str(objective)
        self.pixels_per_micron = max(1e-9, float(pixels_per_micron))
        self.source = source
        self.calibrated_at = calibrated_at

    @property
    def microns_per_pixel(self) -> float:
        return 1.0 / self.pixels_per_micron

    def to_um(self, px: float) -> float:
        return px * self.microns_per_pixel

    def to_px(self, um: float) -> float:
        return um * self.pixels_per_micron

    def as_dict(self) -> Dict:
        return {"objective": self.objective,
                "pixels_per_micron": self.pixels_per_micron,
                "microns_per_pixel": round(self.microns_per_pixel, 6),
                "source": self.source, "calibrated_at": self.calibrated_at}


def calibrations_from_cfg(cfg: Dict) -> Dict[str, Calibration]:
    out: Dict[str, Calibration] = {}
    for name, spec in (cfg.get("survey", {}).get("objectives") or {}).items():
        if isinstance(spec, dict):
            out[str(name)] = Calibration(
                str(name), spec.get("pixels_per_micron", 1.0),
                source=spec.get("source", "configured"))
        else:
            out[str(name)] = Calibration(str(name), spec)
    return out


# ---------------------------------------------------------------------------
# Finalized survey reference (spec §1)
# ---------------------------------------------------------------------------
class SurveyReference:
    """Immutable snapshot of a finalized low-magnification survey."""

    def __init__(self, monolayer_mask: np.ndarray,
                 monolayer_conf: Optional[np.ndarray],
                 coverage_grid: Optional[np.ndarray],
                 origin_x: float, origin_y: float, cell: float,
                 placements: List[Dict], calibration: Calibration,
                 metadata: Dict, mosaic_bgr: Optional[np.ndarray] = None,
                 mosaic_origin: Tuple[float, float] = (0.0, 0.0)) -> None:
        self.monolayer_mask = monolayer_mask      # bool grid (green cells)
        self.monolayer_conf = monolayer_conf      # float grid 0..1 (evidence)
        self.coverage_grid = coverage_grid        # uint8 0/1/2 or None
        self.origin_x = float(origin_x)           # scan px of cell (0,0) corner
        self.origin_y = float(origin_y)
        self.cell = float(cell)                   # scan px per grid cell
        self.placements = placements              # incl. transforms (§1)
        self.calibration = calibration            # reference objective
        self.metadata = dict(metadata or {})
        self.mosaic_bgr = mosaic_bgr              # for relocalization (§7)
        self.mosaic_origin = mosaic_origin
        self.created_at = time.time()

    # -- physical units ------------------------------------------------------
    @property
    def ppm(self) -> float:
        """Reference pixels-per-micron (exact ratios survive estimates)."""
        return self.calibration.pixels_per_micron

    def to_um(self, xy_scan: Tuple[float, float]) -> Tuple[float, float]:
        return (xy_scan[0] * self.ppm, xy_scan[1] * self.ppm)

    def bounds_um(self) -> Tuple[float, float, float, float]:
        h, w = self.monolayer_mask.shape[:2]
        x0, y0 = self.to_um((self.origin_x, self.origin_y))
        x1, y1 = self.to_um((self.origin_x + w * self.cell,
                             self.origin_y + h * self.cell))
        return (x0, y0, x1, y1)

    def field_of_view_um(self, objective: str, frame_w: int, frame_h: int,
                         cfg: Dict) -> Tuple[float, float]:
        """Physical FOV of one camera frame at ``objective`` (§2 readout)."""
        cal = calibrations_from_cfg(cfg).get(objective)
        if cal is None:
            return (0.0, 0.0)
        return (cal.to_um(frame_w), cal.to_um(frame_h))


# ---------------------------------------------------------------------------
# Finalize / load (spec §1)
# ---------------------------------------------------------------------------
def finalize_survey(scan_map, mosaic, cfg: Dict, out_dir: Path,
                    metadata: Optional[Dict] = None
                    ) -> Tuple[SurveyReference, Dict[str, str]]:
    """Package the live survey into ``out_dir`` (survey_final/) and return
    the immutable SurveyReference + export paths.

    Bundle contents (§1): RGB mosaic, pixel coverage mask, monolayer mask +
    confidence grid, per-field placements/transforms, calibration + session
    metadata (survey.json).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _ver, ox, oy, cell, green, conf_img, _conflicts, _contours = \
        scan_map.semantic_grid(
            cell_fraction=float(cfg.get("semantic_cell_fraction", 0.125)),
            green_conf_thr=float(cfg.get("semantic_green_conf_thr", 0.5)),
            min_votes=int(cfg.get("semantic_min_votes", 2)),
            monolayer_weight=float(cfg.get("semantic_monolayer_weight", 1.0)),
            other_weight=float(cfg.get("semantic_other_weight", 1.0)),
            force=True)
    if green is None:
        raise ValueError("survey map is empty - scan fields before FINALIZE")

    _cver, _cx0, _cy0, _ccell, cov_grid, _pct = scan_map.coverage_grid(
        cell_fraction=float(cfg.get("semantic_cell_fraction", 0.125)),
        force=True)

    ref_obj = str(cfg.get("survey", {}).get("reference_objective", "reference"))
    cals = calibrations_from_cfg(cfg)
    cal = cals.get(ref_obj) or Calibration(ref_obj, 1.0, source="default")

    placements = []
    for f in scan_map.fields():
        placements.append({
            "job_id": f.job_id, "frame_idx": f.frame_idx,
            "x": f.x, "y": f.y, "w": f.w, "h": f.h,
            "score": round(float(f.score), 4), "raw_class": f.raw_class,
            "sweep_id": f.sweep_id,
            "reg_method": f.reg_method, "reg_state": f.reg_state,
            "reg_confidence": round(float(f.reg_confidence or 0.0), 3),
            "reg_ref_job_id": getattr(f, "reg_ref_job_id", None),
            "human_label": f.human_label,
            "source_image": getattr(f, "source_image", ""),
        })

    mosaic_paths: Dict[str, str] = {}
    mosaic_bgr = None
    mosaic_origin = (ox, oy)
    if mosaic is not None:
        mosaic_paths = mosaic.export(out_dir)
        snap = mosaic.canvas_snapshot()
        if snap is not None:
            mosaic_bgr, mox, moy = snap
            mosaic_origin = (mox, moy)

    ref = SurveyReference(
        monolayer_mask=green.copy(),
        monolayer_conf=(conf_img.copy() if conf_img is not None else None),
        coverage_grid=(cov_grid.copy() if cov_grid is not None else None),
        origin_x=ox, origin_y=oy, cell=cell,
        placements=placements, calibration=cal,
        metadata=dict(metadata or {}),
        mosaic_bgr=(mosaic_bgr.copy() if mosaic_bgr is not None else None),
        mosaic_origin=mosaic_origin,
    )

    manifest = {
        "type": "finalized_survey",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calibration": cal.as_dict(),
        "reference_objective": ref_obj,
        "origin_scan_px": [round(ox, 2), round(oy, 2)],
        "cell_scan_px": round(cell, 3),
        "grid_shape": [int(green.shape[0]), int(green.shape[1])],
        "monolayer_cells": int(green.sum()),
        "n_placements": len(placements),
        "objective_scales": {name: c.pixels_per_micron
                             for name, c in cals.items()},
        "limitations": (
            "Image-only world coordinates: valid while the slide stays in "
            "place. Removing, rotating or re-clipping the slide invalidates "
            "the map-to-stage relationship; relocalization can recover "
            "translation only, never rotation, and only while appearance "
            "matches. Long-term solution: motorized stage encoders."),
        "metadata": ref.metadata,
    }
    (out_dir / "survey.json").write_text(json.dumps(manifest, indent=2),
                                         encoding="utf-8")
    cv2.imwrite(str(out_dir / "monolayer_mask.png"),
                green.astype(np.uint8) * 255)
    if conf_img is not None:
        np.save(str(out_dir / "monolayer_conf.npy"), conf_img)
    if cov_grid is not None:
        cv2.imwrite(str(out_dir / "coverage_grid.png"), cov_grid)
    (out_dir / "placements.json").write_text(
        json.dumps(placements, indent=2), encoding="utf-8")

    paths = {"survey_dir": str(out_dir),
             "manifest": str(out_dir / "survey.json"),
             "monolayer_mask": str(out_dir / "monolayer_mask.png")}
    paths.update(mosaic_paths)
    return ref, paths


def load_survey(survey_dir: Path) -> SurveyReference:
    """Rebuild a SurveyReference from a finalized bundle (restart-safe)."""
    survey_dir = Path(survey_dir)
    manifest = json.loads(
        (survey_dir / "survey.json").read_text(encoding="utf-8"))
    green_u8 = cv2.imread(str(survey_dir / "monolayer_mask.png"),
                          cv2.IMREAD_GRAYSCALE)
    if green_u8 is None:
        raise FileNotFoundError(f"monolayer_mask.png missing in {survey_dir}")
    green = green_u8 > 127
    conf = None
    conf_p = survey_dir / "monolayer_conf.npy"
    if conf_p.exists():
        conf = np.load(str(conf_p))
    cov = None
    cov_p = survey_dir / "coverage_grid.png"
    if cov_p.exists():
        cov = cv2.imread(str(cov_p), cv2.IMREAD_GRAYSCALE)
    placements: List[Dict] = []
    pl_p = survey_dir / "placements.json"
    if pl_p.exists():
        placements = json.loads(pl_p.read_text(encoding="utf-8"))
    mosaic_bgr = None
    mosaic_origin = tuple(manifest.get("origin_scan_px", (0.0, 0.0)))
    mosaic_p = survey_dir / "mosaic.png"
    if mosaic_p.exists():
        mosaic_bgr = cv2.imread(str(mosaic_p), cv2.IMREAD_COLOR)
        mo_p = survey_dir / "mosaic_origin.json"
        if mo_p.exists():
            o = json.loads(mo_p.read_text(encoding="utf-8"))
            mosaic_origin = (o["x"], o["y"])
    cal = manifest.get("calibration", {})
    return SurveyReference(
        monolayer_mask=green, monolayer_conf=conf, coverage_grid=cov,
        origin_x=manifest["origin_scan_px"][0],
        origin_y=manifest["origin_scan_px"][1],
        cell=manifest["cell_scan_px"],
        placements=placements,
        calibration=Calibration(
            cal.get("objective", manifest.get("reference_objective",
                                               "reference")),
            cal.get("pixels_per_micron", 1.0),
            source=cal.get("source", "loaded"),
            calibrated_at=cal.get("calibrated_at", "")),
        metadata=manifest.get("metadata", {}),
        mosaic_bgr=mosaic_bgr, mosaic_origin=mosaic_origin,
    )


# ---------------------------------------------------------------------------
# Monolayer-only navigation (spec §5)
# ---------------------------------------------------------------------------
def containment(pose_scan: Tuple[float, float], ref: SurveyReference,
                near_edge_um: float = 120.0) -> Dict:
    """Classify a scan-coordinate pose against the frozen monolayer mask.

    Returns {state, nearest_scan, nearest_um, dist_um, guidance_scan} where
    guidance_scan is the unit vector (scan px) toward the nearest mapped
    monolayer cell (None when inside).
    """
    px = (pose_scan[0] - ref.origin_x) / ref.cell - 0.5
    py = (pose_scan[1] - ref.origin_y) / ref.cell - 0.5
    h, w = ref.monolayer_mask.shape[:2]
    col, row = int(round(px)), int(round(py))
    if 0 <= row < h and 0 <= col < w and ref.monolayer_mask[row, col]:
        return {"state": INSIDE, "nearest_scan": None, "nearest_um": None,
                "dist_um": 0.0, "guidance_scan": None}
    greens = np.argwhere(ref.monolayer_mask)
    if greens.size == 0:
        return {"state": OUTSIDE, "nearest_scan": None, "nearest_um": None,
                "dist_um": float("inf"), "guidance_scan": None}
    d = np.hypot(greens[:, 0] - py, greens[:, 1] - px)
    j = int(np.argmin(d))
    nearest_scan = (ref.origin_x + (float(greens[j, 1]) + 0.5) * ref.cell,
                    ref.origin_y + (float(greens[j, 0]) + 0.5) * ref.cell)
    dist_um = math.hypot(nearest_scan[0] - pose_scan[0],
                         nearest_scan[1] - pose_scan[1]) * ref.ppm
    vec = (nearest_scan[0] - pose_scan[0], nearest_scan[1] - pose_scan[1])
    n = math.hypot(*vec) or 1.0
    return {"state": (NEAR_EDGE if dist_um <= near_edge_um else OUTSIDE),
            "nearest_scan": nearest_scan,
            "nearest_um": ref.to_um(nearest_scan),
            "dist_um": round(dist_um, 1),
            "guidance_scan": (vec[0] / n, vec[1] / n)}


# ---------------------------------------------------------------------------
# Revisit points (spec §6)
# ---------------------------------------------------------------------------
def generate_revisit_points(ref: SurveyReference, mode: str = "grid",
                            count: int = 6,
                            min_sep_cells: float = 3.0) -> List[Dict]:
    """Candidate revisit positions INSIDE the mapped monolayer (§6).

    grid           - evenly spaced lattice snapped to green cells
    top_confidence - highest-evidence green cells, spread by min separation
    centroid       - single densest point (mean of green cells)
    Returns scan-px + micron coordinates for guidance.
    """
    greens = np.argwhere(ref.monolayer_mask)
    if greens.size == 0:
        return []
    conf = ref.monolayer_conf
    if conf is None or conf.shape != ref.monolayer_mask.shape:
        conf = np.where(ref.monolayer_mask, 1.0, 0.0)

    def emit(cell_r: int, cell_c: int) -> Dict:
        x = ref.origin_x + (cell_c + 0.5) * ref.cell
        y = ref.origin_y + (cell_r + 0.5) * ref.cell
        return {"x_scan": round(x, 1), "y_scan": round(y, 1),
                "x_um": round(x * ref.ppm, 1), "y_um": round(y * ref.ppm, 1),
                "conf": round(float(conf[cell_r, cell_c]), 3)}

    picked: List[Tuple[int, int]] = []

    def far_enough(r: int, c: int) -> bool:
        return all(math.hypot(r - pr, c - pc) >= min_sep_cells
                   for pr, pc in picked)

    if mode == "centroid":
        cy, cx = float(greens[:, 0].mean()), float(greens[:, 1].mean())
        d = np.hypot(greens[:, 0] - cy, greens[:, 1] - cx)
        j = int(np.argmin(d))
        return [emit(int(greens[j, 0]), int(greens[j, 1]))]

    if mode == "top_confidence":
        ys, xs = np.where(ref.monolayer_mask)
        order = np.argsort(-conf[ref.monolayer_mask])
        for idx in order:
            r, c = int(ys[idx]), int(xs[idx])
            if far_enough(r, c):
                picked.append((r, c))
            if len(picked) >= count:
                break
    elif mode == "grid":
        r0, r1 = int(greens[:, 0].min()), int(greens[:, 0].max())
        c0, c1 = int(greens[:, 1].min()), int(greens[:, 1].max())
        ys_c, xs_c = greens[:, 0].mean(), greens[:, 1].mean()
        # densest green cell = ring center -> even coverage inward-out
        best = min(zip(greens[:, 0], greens[:, 1]),
                   key=lambda rc: math.hypot(rc[0] - ys_c, rc[1] - xs_c))
        side = max(1, int(math.ceil(math.sqrt(count))))
        step_r = max(1.0, (r1 - r0 + 1) / side)
        step_c = max(1.0, (c1 - c0 + 1) / side)
        targets = [(r0 + (i + 0.5) * step_r, c0 + (j + 0.5) * step_c)
                   for i in range(side) for j in range(side)]
        targets.sort(key=lambda t: math.hypot(t[0] - best[0], t[1] - best[1]))
        for tr, tc in targets:
            bd, bc = float("inf"), None
            for r, c in zip(greens[:, 0], greens[:, 1]):
                d = math.hypot(r - tr, c - tc)
                if d < bd:
                    bd, bc = d, (int(r), int(c))
            if bc is not None and far_enough(*bc):
                picked.append(bc)
            if len(picked) >= count:
                break
    else:
        raise ValueError(f"unknown revisit mode: {mode}")

    return [emit(r, c) for r, c in picked]


# ---------------------------------------------------------------------------
# Relocalization (spec §7)
# ---------------------------------------------------------------------------
class Relocalizer:
    """Recover the survey-coordinate pose of a live high-mag frame.

    The live frame is rescaled to the reference pixel size (calibration
    ratio) and template-matched against the frozen mosaic. Reports failure
    honestly - never invents a pose (§8).
    """

    def __init__(self, ref: SurveyReference, cfg: Dict) -> None:
        self.ref = ref
        self.match_width = int(cfg.get("survey", {})
                               .get("relocate_match_width", 160))
        self.min_response = float(cfg.get("survey", {})
                                  .get("relocate_min_response", 0.45))

    def relocate(self, frame_bgr: np.ndarray,
                 high_ppm: float) -> Optional[Dict]:
        """Return {x_scan, y_scan, response} or None (not found)."""
        if self.ref.mosaic_bgr is None or high_ppm <= 0:
            return None
        s = self.ref.ppm / float(high_ppm)   # high-mag px -> ref scan px
        h, w = frame_bgr.shape[:2]
        tw, th = max(8, int(round(w * s))), max(8, int(round(h * s)))
        tmpl = cv2.cvtColor(cv2.resize(frame_bgr, (tw, th),
                                       interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        # extra downscale of BOTH images for match speed; relative scale kept
        d = min(1.0, self.match_width / float(max(tw, 1)))
        if d < 1.0:
            tmpl = cv2.resize(tmpl, (max(8, int(tw * d)),
                                     max(8, int(th * d))),
                              interpolation=cv2.INTER_AREA)
        mos = cv2.cvtColor(self.ref.mosaic_bgr, cv2.COLOR_BGR2GRAY)
        if d < 1.0:
            mos = cv2.resize(mos, (max(tmpl.shape[1] + 1, int(mos.shape[1] * d)),
                                   max(tmpl.shape[0] + 1, int(mos.shape[0] * d))),
                             interpolation=cv2.INTER_AREA)
        if mos.shape[0] < tmpl.shape[0] or mos.shape[1] < tmpl.shape[1]:
            return None
        res = cv2.matchTemplate(mos, tmpl, cv2.TM_CCOEFF_NORMED)
        _minv, maxv, _minl, maxl = cv2.minMaxLoc(res)
        if maxv < self.min_response:
            return None
        ox, oy = self.ref.mosaic_origin
        x = ox + (maxl[0] + tmpl.shape[1] / 2.0) / d
        y = oy + (maxl[1] + tmpl.shape[0] / 2.0) / d
        return {"x_scan": round(x, 1), "y_scan": round(y, 1),
                "response": round(float(maxv), 3)}


# ---------------------------------------------------------------------------
# Exploration manager (spec §4/§5/§6/§7)
# ---------------------------------------------------------------------------
class ExplorationManager:
    """Owns the frozen survey + the high-mag exploration session state.

    Thread model: finalize/generate/GUI calls come from the UI thread;
    process_field() runs on CPU feature workers - every mutation is locked.
    """

    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.lock = threading.Lock()
        self.survey: Optional[SurveyReference] = None
        self.survey_dir: Optional[str] = None
        self.active_objective = str(cfg.get("survey", {})
                                    .get("reference_objective", "reference"))
        self.anchor_scan: Optional[Tuple[float, float]] = None
        self._chain = None                 # GlobalPoseTracker (lazy import)
        self._chain_origin: Optional[Tuple[float, float]] = None
        self.revisit: List[Dict] = []
        self.target_idx: int = -1
        self._reloc: Optional[Relocalizer] = None
        self._log_rows: List[Dict] = []

    # -- objectives / calibration ---------------------------------------------
    def objectives(self) -> Dict[str, Calibration]:
        return calibrations_from_cfg(self.cfg)

    def set_active_objective(self, name: str) -> None:
        with self.lock:
            if name in self.objectives():
                self.active_objective = str(name)

    def active_cal(self) -> Calibration:
        return (self.objectives().get(self.active_objective)
                or Calibration(self.active_objective, 1.0, "default"))

    def high_mag_active(self) -> bool:
        """True when the active objective differs from the survey reference."""
        with self.lock:
            if self.survey is None:
                return False
            return (abs(self.active_cal().pixels_per_micron - self.survey.ppm)
                    > 1e-9)

    # -- finalize / load (§1) ---------------------------------------------------
    def finalize(self, scan_map, mosaic, out_dir: Path,
                 metadata: Optional[Dict] = None) -> Dict[str, str]:
        ref, paths = finalize_survey(scan_map, mosaic, self.cfg, out_dir,
                                     metadata=metadata)
        with self.lock:
            self.survey = ref
            self.survey_dir = str(out_dir)
            self._reloc = Relocalizer(ref, self.cfg)
        return paths

    def load(self, survey_dir: Path) -> None:
        ref = load_survey(survey_dir)
        with self.lock:
            self.survey = ref
            self.survey_dir = str(survey_dir)
            self._reloc = Relocalizer(ref, self.cfg)

    # -- revisit points (§6) -----------------------------------------------------
    def generate_revisit(self, mode: str, count: int) -> List[Dict]:
        with self.lock:
            if self.survey is None:
                return []
            self.revisit = generate_revisit_points(
                self.survey, mode=mode, count=count,
                min_sep_cells=float(self.cfg.get("survey", {})
                                    .get("revisit_min_sep_cells", 3.0)))
            self.target_idx = 0 if self.revisit else -1
            return [dict(p) for p in self.revisit]

    def set_target(self, idx: int) -> Optional[Dict]:
        with self.lock:
            if 0 <= idx < len(self.revisit):
                self.target_idx = idx
                return dict(self.revisit[idx])
            return None

    def current_target(self) -> Optional[Dict]:
        with self.lock:
            if 0 <= self.target_idx < len(self.revisit):
                return dict(self.revisit[self.target_idx])
            return None

    def advance_target(self) -> Optional[Dict]:
        with self.lock:
            if not self.revisit:
                return None
            self.target_idx = (self.target_idx + 1) % len(self.revisit)
            return dict(self.revisit[self.target_idx])

    # -- anchors + relocalization (§7) ---------------------------------------------
    def _reset_chain(self, anchor_scan: Tuple[float, float]) -> None:
        from src.live.registration import GlobalPoseTracker
        self._chain = GlobalPoseTracker(self.cfg)
        self._chain_origin = None
        self.anchor_scan = anchor_scan

    def set_anchor(self, x_scan: float, y_scan: float,
                   reason: str = "declared") -> None:
        """Declare "the stage is now at this survey position" (§7 prior)."""
        with self.lock:
            self._reset_chain((float(x_scan), float(y_scan)))
            self._log_rows.append({"t": time.time(), "event": "anchor",
                                   "reason": reason, "x_scan": x_scan,
                                   "y_scan": y_scan})

    def relocate(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        with self.lock:
            reloc = self._reloc
        if reloc is None:
            return None
        hit = reloc.relocate(frame_bgr, self.active_cal().pixels_per_micron)
        if hit is not None:
            self.set_anchor(hit["x_scan"], hit["y_scan"],
                            reason=f"relocated (resp {hit['response']})")
        return hit

    # -- per-field processing (called from feature workers) --------------------------
    def should_handle(self) -> bool:
        with self.lock:
            return self.survey is not None and self.high_mag_active()

    def process_field(self, job, st) -> None:
        """High-mag path: chain-register within the current magnification,
        express the pose in SURVEY scan coordinates via the anchor + the
        calibration ratio (§4), classify containment (§5). Never touches the
        frozen survey map/mosaic (§1)."""
        with self.lock:
            survey = self.survey
            chain = self._chain
        if survey is None:
            return
        result = {"state": UNKNOWN, "dist_um": None, "guidance": None,
                  "pose_scan": None, "pose_um": None,
                  "objective": self.active_objective}
        reg = chain.register_field(job, job.frame) if chain is not None else None
        if reg is not None:
            job.reg_state = reg.state
            job.reg_method = reg.method
            job.reg_confidence = reg.confidence
            job.reg_dx, job.reg_dy = round(reg.dx, 1), round(reg.dy, 1)
            job.drift_warning = reg.drift_warning
            with self.lock:
                if self._chain_origin is None and \
                        reg.state != "REGISTRATION_FAILED":
                    self._chain_origin = (reg.global_x, reg.global_y)
                origin = self._chain_origin
                anchor = self.anchor_scan
                scale = survey.ppm / self.active_cal().pixels_per_micron
            if origin is not None and anchor is not None and \
                    reg.state != "REGISTRATION_FAILED":
                px = anchor[0] + (reg.global_x - origin[0]) * scale
                py = anchor[1] + (reg.global_y - origin[1]) * scale
                job.map_x, job.map_y = px, py
                cont = containment((px, py), survey,
                                   near_edge_um=float(self.cfg.get(
                                       "survey", {}).get("near_edge_um", 120)))
                result = {"state": cont["state"],
                          "dist_um": cont["dist_um"],
                          "guidance": cont["guidance_scan"],
                          "nearest_scan": cont["nearest_scan"],
                          "pose_scan": (round(px, 1), round(py, 1)),
                          "pose_um": survey.to_um((px, py)),
                          "objective": self.active_objective}
        job.features["exploration_state"] = result["state"]
        job.features["exploration_pose_um"] = (
            [round(result["pose_um"][0], 1), round(result["pose_um"][1], 1)]
            if result["pose_um"] else None)
        with st.lock:
            st.exploration = dict(result)
        self._log_rows.append({
            "t": time.time(), "event": "field", "job_id": job.job_id,
            "objective": result["objective"],
            "x_scan": result["pose_scan"][0] if result["pose_scan"] else "",
            "y_scan": result["pose_scan"][1] if result["pose_scan"] else "",
            "state": result["state"], "dist_um": result["dist_um"] or "",
            "class": job.raw_class or "",
            "score": round(job.score, 3) if job.score is not None else "",
            "reg_state": getattr(job, "reg_state", "") or "",
        })

    # -- log -----------------------------------------------------------------------
    def save_log(self, session_dir: Path) -> Optional[str]:
        with self.lock:
            rows = list(self._log_rows)
        if not rows:
            return None
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        p = session_dir / "exploration_log.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        return str(p)
