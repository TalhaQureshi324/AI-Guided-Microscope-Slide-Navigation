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


class ScanMap:
    """Thread-safe registry of analyzed field footprints."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fields: List[FieldFootprint] = []
        self._by_job: Dict[int, FieldFootprint] = {}

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
        fp = FieldFootprint(
            job_id=job.job_id,
            t_capture=job.t_capture,
            x=float(job.cumulative_x),
            y=float(job.cumulative_y),
            w=float(w),
            h=float(h),
            score=float(job.score),
            raw_class=job.raw_class or "UNCERTAIN",
            priority=job.priority,
            frame_idx=job.frame_idx,
            source_image=source_image,
            human_label=job.human_label,
        )
        with self._lock:
            self._fields.append(fp)
            self._by_job[fp.job_id] = fp
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
        nx0 = job.cumulative_x - w / 2.0
        ny0 = job.cumulative_y - h / 2.0
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

                if is_mono:
                    fp.ml_agreements += weight
                    summary["affirmed"] += 1
                    if fp.ml_demoted and (fp.ml_agreements > fp.ml_disagreements):
                        fp.ml_demoted = False  # evidence restored the area
                        summary["promoted"] += 1
                else:
                    fp.ml_disagreements += weight
                    summary["contradicted"] += 1
                    if (not fp.ml_demoted
                            and fp.ml_disagreements >= contradictions_to_demote
                            and fp.ml_disagreements > fp.ml_agreements):
                        fp.ml_demoted = True  # conservative: repeated majority only
                        summary["demoted"] += 1

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

    def reset(self) -> None:
        """Reset Scan (hotkey R) clears the whole survey."""
        with self._lock:
            self._fields.clear()
            self._by_job.clear()

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
