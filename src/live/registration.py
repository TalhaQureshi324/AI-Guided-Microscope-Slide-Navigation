"""Robust frame registration and global pose (Phase 7).

Answers: *where does this new captured field belong relative to everything
already scanned?*

Design (spec sections 1-10):
  * TRANSLATION-ONLY primary model via phase correlation on prepped
    (grayscale, downscaled, contrast-normalised, mean/std-scaled) frames -
    never on overlays, cell masks or GUI imagery;
  * registration CONFIDENCE with three states: REGISTRATION_GOOD /
    REGISTRATION_WEAK / REGISTRATION_FAILED - weak fields are placed but
    flagged, failed fields are never added to the trusted map;
  * FALLBACK via ORB keypoints + RANSAC-style robust translation when the
    primary fails (repeated circular RBC structures make phase correlation
    occasionally ambiguous);
  * REFERENCE SELECTION: not just the previous field - the tracker keeps a
    memory of recent good fields and can re-register against ANY of them
    (loop closure on revisits);
  * DRIFT MONITORING: when a new field registers against an old map field
    and disagrees with the chained prediction, a POSE_DRIFT_WARNING is
    recorded and the map position wins (revisits correct accumulated drift);
  * the first field anchors the session at global (0, 0).

Units: full-resolution reference pixels (documented internal scan unit).
GUI/screen coordinates never enter this module. Later, motorized stage
coordinates replace `register_field`'s position source without touching the
map, monolayer layer or GUI.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# registration states (spec §4)
REG_GOOD = "REGISTRATION_GOOD"
REG_WEAK = "REGISTRATION_WEAK"
REG_FAILED = "REGISTRATION_FAILED"

# methods
M_FIRST = "FIRST_FIELD"
M_PHASE = "PHASE_CORR"
M_ORB = "ORB_TRANSLATION"
M_MAP_SNAP = "MAP_FIELD_SNAP"

WORK_WIDTH = 480


@dataclass
class RegistrationResult:
    state: str                      # REG_GOOD / REG_WEAK / REG_FAILED
    method: str
    confidence: float               # 0..1
    dx: float                       # full-res px, translation from reference
    dy: float
    global_x: float = 0.0
    global_y: float = 0.0
    ref_job_id: Optional[int] = None
    drift_warning: bool = False
    detail: str = ""


def prep_frame(frame_bgr: np.ndarray, work_width: int = WORK_WIDTH) -> np.ndarray:
    """Registration representation of a RAW microscope frame (spec §3).

    Grayscale -> downscaled -> contrast-normalised (zero mean, unit std).
    Deliberately separate from Cellpose preprocessing and computed from the
    ORIGINAL frame - overlays/borders/masks never enter registration.
    """
    h, w = frame_bgr.shape[:2]
    scale = work_width / float(w)
    small = cv2.resize(frame_bgr, (work_width, max(1, int(round(h * scale)))),
                       interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = small.astype(np.float32)
    raw_std = float(gray.std())
    if raw_std > 1e-6:
        gray = (gray - gray.mean()) / raw_std
    else:
        gray -= gray.mean()   # featureless frame -> all zeros
    return gray, raw_std


def _phase_corr(ref: np.ndarray, cur: np.ndarray,
                scale: float) -> Tuple[float, float, float]:
    """Phase correlation translation ref->cur in FULL-RES px + response."""
    window = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    (sx, sy), response = cv2.phaseCorrelate(ref, cur, window)
    inv = 1.0 / scale if scale > 0 else 1.0
    return float(sx) * inv, float(sy) * inv, float(response)


def _orb_translation(ref: np.ndarray, cur: np.ndarray, scale: float,
                     cfg: Dict) -> Tuple[float, float, float, int, float]:
    """ORB keypoints + ratio-test matching + robust median translation.

    Returns (dx, dy, confidence, n_matches, inlier_ratio) in full-res px.
    Translation-only model: median of per-match offsets (RANSAC-like robust
    consensus for a 3-DOF model without fitting a full homography).
    """
    r_u8 = np.clip(ref, 0, 255).astype(np.uint8)
    c_u8 = np.clip(cur, 0, 255).astype(np.uint8)
    orb = cv2.ORB_create(nfeatures=2000, patchSize=31)
    kp_r, des_r = orb.detectAndCompute(r_u8, None)
    kp_c, des_c = orb.detectAndCompute(c_u8, None)
    if des_r is None or des_c is None or len(kp_r) < 8 or len(kp_c) < 8:
        return 0.0, 0.0, 0.0, 0, 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(des_r, des_c, k=2)
    good = [m for m, n in (pair for pair in raw if len(pair) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return 0.0, 0.0, 0.0, len(good), 0.0
    inv = 1.0 / scale if scale > 0 else 1.0
    dxs, dys = [], []
    for mch in good:
        dxs.append((kp_c[mch.trainIdx].pt[0] - kp_r[mch.queryIdx].pt[0]) * inv)
        dys.append((kp_c[mch.trainIdx].pt[1] - kp_r[mch.queryIdx].pt[1]) * inv)
    dxs_a = np.asarray(dxs)
    dys_a = np.asarray(dys)
    med_dx = float(np.median(dxs_a))
    med_dy = float(np.median(dys_a))
    inliers = int(((np.abs(dxs_a - med_dx) < 20.0) &
                   (np.abs(dys_a - med_dy) < 20.0)).sum())
    inlier_ratio = inliers / len(dxs_a)
    conf = min(1.0, inlier_ratio * min(1.0, len(good) / 40.0))
    return med_dx, med_dy, conf, len(good), inlier_ratio


@dataclass
class FieldReg:
    """Registration memory for one accepted field."""
    job_id: int
    pose: Tuple[float, float]
    prep: np.ndarray
    conf: float
    state: str
    method: str
    frame_idx: int
    t: float
    w: float
    h: float
    frame_diag: float


class GlobalPoseTracker:
    """Thread-safe global pose bookkeeper for the current slide session."""

    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg.get("registration", cfg)
        rcfg = self.cfg
        self.work_width = int(rcfg.get("work_width", WORK_WIDTH))
        self.phase_good = float(rcfg.get("phase_good", 0.22))
        self.phase_weak = float(rcfg.get("phase_weak", 0.12))
        self.orb_good_matches = int(rcfg.get("orb_good_matches", 15))
        self.orb_good_inlier = float(rcfg.get("orb_good_inlier", 0.5))
        self.orb_weak_matches = int(rcfg.get("orb_weak_matches", 8))
        self.orb_weak_inlier = float(rcfg.get("orb_weak_inlier", 0.35))
        self.max_jump_frac = float(rcfg.get("max_jump_frac", 1.6))
        self.drift_warn_px = float(rcfg.get("drift_warn_px", 300.0))
        self.search_radius_frac = float(rcfg.get("search_radius_frac", 1.3))
        self.max_cache = int(rcfg.get("max_prep_cache", 30))

        self._lock = threading.Lock()
        self._fields: Dict[int, FieldReg] = {}
        self._order: List[int] = []          # job_ids in acceptance order
        self._last_good_id: Optional[int] = None
        self.preferred_ref: Optional[int] = None  # START NEW SWEEP anchor
        self.version = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._fields.clear()
            self._order.clear()
            self._last_good_id = None
            self.preferred_ref = None
            self.version += 1

    # ------------------------------------------------------------------
    def register_field(self, job, frame_bgr: np.ndarray) -> RegistrationResult:
        """Compute the global pose for one analyzed job (spec §1-§7)."""
        prep, cur_raw_std = prep_frame(frame_bgr, self.work_width)
        diag = math.hypot(prep.shape[1], prep.shape[0])

        texture_min = float(self.cfg.get("texture_min_std", 3.0))
        ncc_min = float(self.cfg.get("overlap_ncc_min", 0.25))

        with self._lock:
            ref = self._pick_reference(job, prep)
            if ref is None:
                if cur_raw_std < texture_min:
                    return RegistrationResult(
                        REG_FAILED, "FEATURELESS", 0.0, 0.0, 0.0, 0.0, 0.0, None,
                        False, "featureless first field (std < "
                               f"{texture_min:g}) - nothing to anchor")
                rec = FieldReg(job.job_id, (0.0, 0.0), prep, 1.0, REG_GOOD,
                               M_FIRST, job.frame_idx, job.t_capture,
                               float(job.frame.shape[1]), float(job.frame.shape[0]),
                               diag)
                self._store(rec, good=True)
                return RegistrationResult(
                    REG_GOOD, M_FIRST, 1.0, 0.0, 0.0, 0.0, 0.0, None,
                    False, "first field anchors the session at (0,0)")
            if cur_raw_std < texture_min:
                return RegistrationResult(
                    REG_FAILED, "FEATURELESS", 0.0, 0.0, 0.0, 0.0, 0.0,
                    ref.job_id, False, "featureless current frame")

        # primary: phase correlation vs the chosen reference.
        # KNOWN LIMIT: phase correlation WRAPS for shifts beyond half the prep
        # frame (aliases large moves to small wrong ones with high response).
        # Therefore ORB translation runs as VERIFICATION on every field: when
        # ORB evidence is strong it wins (wrap-free); agreement between the
        # two raises confidence. Pure-phase answers stand only when ORB is
        # unavailable (texture-poor scenes) - and then large shifts are
        # treated as WEAK at best, never silently trusted.
        prep_to_full = job.frame.shape[1] / float(prep.shape[1])
        pdx, pdy, resp = _phase_corr(ref.prep, prep, 1.0)   # prep-scale px
        pdx *= prep_to_full; pdy *= prep_to_full            # -> full-res px
        pconf = min(1.0, max(0.0, resp))
        p_good = pconf >= self.phase_good
        p_wrapped = math.hypot(pdx, pdy) > 0.45 * job.frame.shape[1]

        odx, ody, oconf, nmatch, inl = _orb_translation(
            ref.prep, prep, 1.0, self.cfg)                  # prep-scale px
        odx *= prep_to_full; ody *= prep_to_full            # -> full-res px
        orb_good = (nmatch >= self.orb_good_matches
                    and inl >= self.orb_good_inlier and oconf >= 0.3)
        orb_weak = (nmatch >= self.orb_weak_matches
                    and inl >= self.orb_weak_inlier)

        if orb_good:
            dx, dy, conf, method = odx, ody, max(oconf, pconf if p_good and
                                                 abs(odx - pdx) < 25 and
                                                 abs(ody - pdy) < 25 else oconf), M_ORB
            state = REG_GOOD
            if p_good and abs(odx - pdx) < 25 and abs(ody - pdy) < 25:
                method = M_ORB  # cross-verified
        elif p_good and not p_wrapped and not orb_good and nmatch > 0:
            # phase confident, ORB found some but not enough good matches:
            # trust phase only for sub-wrap shifts
            dx, dy, conf, method, state = pdx, pdy, pconf, M_PHASE, REG_GOOD
        elif orb_weak:
            dx, dy, conf, method = odx, ody, max(oconf, 0.2), M_ORB
            state = REG_WEAK if not p_wrapped else REG_WEAK
        elif p_good:
            dx, dy, conf, method = pdx, pdy, min(pconf, 0.45), M_PHASE
            state = REG_WEAK
        elif pconf >= self.phase_weak and not p_wrapped:
            dx, dy, conf, method = pdx, pdy, pconf, M_PHASE, REG_WEAK
        else:
            dx, dy, conf, method = pdx, pdy, 0.0, M_PHASE
            state = REG_FAILED

        # ---- overlap-consistency verification (spec §4) -------------------
        # phase-corr WRAPS at large shifts (aliases to a wrong small offset
        # with high response), and featureless frames "correlate" perfectly;
        # the actual NCC over the implied overlap exposes both.
        ncc = overlap_ncc(ref.prep, prep, dx / prep_to_full, dy / prep_to_full)
        if ncc < ncc_min:
            state = REG_FAILED
            conf = min(conf, 0.05)

        jump = math.hypot(dx, dy)
        if jump > self.max_jump_frac * ref.frame_diag:
            conf = 0.0
            state = REG_FAILED

        # candidate pose from the reference chain
        gx, gy = ref.pose[0] + dx, ref.pose[1] + dy
        drift = False

        # loop closure: try to snap against a nearby EXISTING map field (§6/§7)
        snap = self._try_snap(prep, gx, gy, ref.frame_diag, exclude=ref.job_id)
        if snap is not None:
            snap_field, sdx, sdy, sconf, sstate = snap
            sdx *= prep_to_full; sdy *= prep_to_full
            if sstate in (REG_GOOD, REG_WEAK):
                snapped = (snap_field.pose[0] + sdx, snap_field.pose[1] + sdy)
                drift = math.hypot(snapped[0] - gx, snapped[1] - gy) > self.drift_warn_px
                if sconf >= conf or drift:
                    gx, gy = snapped
                    conf = max(conf, sconf)
                    method = M_MAP_SNAP + ("+" + method if method != M_MAP_SNAP else "")
                    state = sstate

        with self._lock:
            rec = FieldReg(job.job_id, (gx, gy), prep, conf, state, method,
                           job.frame_idx, job.t_capture,
                           float(job.frame.shape[1]), float(job.frame.shape[0]), diag)
            good = state == REG_GOOD
            self._store(rec, good=good)
        return RegistrationResult(state, method, conf, dx, dy, gx, gy,
                                  ref.job_id, drift, "")

    # ------------------------------------------------------------------
    def _pick_reference(self, job, prep: np.ndarray) -> Optional[FieldReg]:
        """Reference selection (spec §6): preferred anchor > last good >
        most recent. Only REGISTRATION_GOOD fields serve as references."""
        if self.preferred_ref is not None and self.preferred_ref in self._fields:
            ref = self._fields[self.preferred_ref]
            if ref.state == REG_GOOD:
                self.preferred_ref = None  # consumed: applies to the first field only
                return ref
        if self._last_good_id is not None and self._last_good_id in self._fields:
            return self._fields[self._last_good_id]
        if self._order:
            for jid in reversed(self._order):
                if self._fields[jid].state == REG_GOOD:
                    return self._fields[jid]
        return None

    def _store(self, rec: FieldReg, good: bool) -> None:
        self._fields[rec.job_id] = rec
        if rec.job_id not in self._order:
            self._order.append(rec.job_id)
        if good:
            self._last_good_id = rec.job_id
        if len(self._order) > self.max_cache:
            drop = self._order.pop(0)
            self._fields.pop(drop, None)
            if self._last_good_id == drop:
                self._last_good_id = self._order[-1] if self._order else None
        self.version += 1

    def _try_snap(self, prep: np.ndarray, gx: float, gy: float,
                  diag: float, exclude: int):
        """Loop closure: register against existing fields near the predicted
        pose (spec §6). Returns (field, dx, dy, conf, state) or None."""
        with self._lock:
            candidates = [self._fields[j] for j in self._order
                          if j != exclude and self._fields[j].state == REG_GOOD]
        if not candidates:
            return None
        search_r = self.search_radius_frac * diag
        best = None
        for cand in candidates[-8:]:  # recent fields first
            dist = math.hypot(cand.pose[0] - gx, cand.pose[1] - gy)
            if dist > search_r:
                continue
            dx, dy, resp = _phase_corr(cand.prep, prep, self.work_width / cand.prep.shape[1])
            conf = min(1.0, max(0.0, resp))
            state = (REG_GOOD if conf >= self.phase_good else
                     REG_WEAK if conf >= self.phase_weak else REG_FAILED)
            if state == REG_FAILED:
                continue
            if best is None or conf > best[3]:
                best = (cand, dx, dy, conf, state)
        return best

    def preview_coverage(self, frame_bgr: np.ndarray, sweep_off_x: float,
                         sweep_off_y: float, scan_map=None) -> Dict:
        """Read-only live overlap preview for operator guidance (spec §7).

        Registers the current live frame against the last GOOD field (phase
        correlation only - no commitment) and evaluates footprint coverage
        vs the scanned union. Returns a plain dict for the GUI:
        state / overlap fractions / move hint / confidence.
        """
        try:
            prep, raw_std = prep_frame(frame_bgr, self.work_width)
        except Exception:  # noqa: BLE001
            return {"state": "REGISTRATION UNCERTAIN"}
        with self._lock:
            ref = None
            if self._last_good_id is not None and self._last_good_id in self._fields:
                ref = self._fields[self._last_good_id]
        if ref is None:
            return {"state": "FIRST_FIELD", "overlap_prev_pct": 0.0,
                    "overlap_total_pct": 0.0, "new_coverage_pct": 1.0}
        if raw_std < float(self.cfg.get("registration", self).get(
                "texture_min_std", 3.0)):
            return {"state": "REGISTRATION UNCERTAIN", "overlap_prev_pct": 0.0}

        scale_full = 1.0 / (self.work_width / float(frame_bgr.shape[1]))
        dx, dy, resp = _phase_corr(ref.prep, prep, 1.0)
        dx *= scale_full; dy *= scale_full
        conf = min(1.0, max(0.0, resp))
        cand_pose = (ref.pose[0] + dx, ref.pose[1] + dy)

        from src.live.coverage import ExistingFootprint, evaluate_coverage
        h, w = frame_bgr.shape[:2]
        cand_rect = (cand_pose[0] - w / 2, cand_pose[1] - h / 2,
                     cand_pose[0] + w / 2, cand_pose[1] + h / 2)
        existing = [
            ExistingFootprint((f.x - f.w / 2, f.y - f.h / 2,
                               f.x + f.w / 2, f.y + f.h / 2),
                              f.job_id, f.reg_confidence, f.human_label)
            for f in scan_map.active_fields() if f.job_id != -1
        ] if scan_map is not None else []
        rep = evaluate_coverage(
            cand_rect, existing,
            target_overlap_min=float(self.cfg.get("target_overlap_min", 0.20)),
            target_overlap_max=float(self.cfg.get("target_overlap_max", 0.30)),
            duplicate_overlap_threshold=float(self.cfg.get(
                "duplicate_overlap_threshold", 0.90)),
            gap_risk_threshold=float(self.cfg.get("gap_risk_threshold", 0.35)),
        )
        out = {
            "state": rep.state,
            "overlap_prev_pct": rep.overlap_prev_pct,
            "overlap_total_pct": rep.overlap_total_pct,
            "new_coverage_pct": rep.new_coverage_pct,
            "duplicate_pct": rep.duplicate_pct,
            "best_job_id": rep.best_job_id,
            "best_reg_conf": (existing[0].reg_confidence if existing else 1.0),
            "reg_conf": conf,
            "dir_x": dx, "dir_y": dy,
        }
        from src.live.coverage import movement_hint
        out["move_hint"] = movement_hint(
            rep.overlap_prev_pct,
            float(self.cfg.get("target_overlap_min", 0.20)),
            float(self.cfg.get("target_overlap_max", 0.30)), dx, dy)
        return out

    # ------------------------------------------------------------------
    def poses(self) -> Dict[int, Tuple[float, float]]:
        with self._lock:
            return {jid: rec.pose for jid, rec in self._fields.items()}

    def field_count(self) -> int:
        with self._lock:
            return len(self._fields)


def overlap_ncc(ref: np.ndarray, cur: np.ndarray,
                dx: float, dy: float) -> float:
    """Normalized cross-correlation over the overlap implied by shift (dx, dy).

    Returns 0.0 when the overlap is too small to judge or when either crop is
    featureless. This is the 'overlap consistency' evidence from spec §4: a
    WRONG candidate shift (e.g. phase-correlation wraparound, or a confident
    answer on a featureless frame) yields low NCC over its claimed overlap
    even when the correlation RESPONSE looked high.
    """
    h, w = ref.shape[:2]
    # cur(x, y) = ref(x - dx, y - dy): the corresponding pair is
    # ref(x, y) <-> cur(x - dx, y - dy), valid for
    # x in [max(0, dx), min(w, w + dx)] (equivalently mirrored in -dx).
    x0 = max(0.0, -dx); x1 = min(float(w), w - dx)
    y0 = max(0.0, -dy); y1 = min(float(h), h - dy)
    if x1 - x0 < 24 or y1 - y0 < 24:
        return 0.0
    rx0, rx1 = int(round(x0)), int(round(x1))
    ry0, ry1 = int(round(y0)), int(round(y1))
    cx0, cx1 = int(round(x0 + dx)), int(round(x1 + dx))
    cy0, cy1 = int(round(y0 + dy)), int(round(y1 + dy))
    a = ref[ry0:ry1, rx0:rx1]
    b = cur[cy0:cy1, cx0:cx1]
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if den < 1e-6:
        return 0.0
    return float((a * b).sum() / den)
