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
