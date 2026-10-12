"""Phase 13 tests: survey finalization, calibration, containment,
revisit points, relocalization, exploration pose transform.

Run:  python scripts/test_phase13.py
Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import math
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.live.scan_map import ScanMap            # noqa: E402
from src.live.survey import (                    # noqa: E402
    INSIDE, NEAR_EDGE, OUTSIDE, ExplorationManager, Relocalizer,
    SurveyReference, Calibration, containment, generate_revisit_points,
    load_survey,
)

CFG = {
    "survey": {
        "reference_objective": "10x",
        "objectives": {"10x": {"pixels_per_micron": 1.0, "source": "estimated"},
                       "40x": {"pixels_per_micron": 4.0, "source": "estimated"}},
        "near_edge_um": 120.0,
        "revisit_count": 6,
        "revisit_min_sep_cells": 2.0,
        "relocate_match_width": 160,
        "relocate_min_response": 0.45,
    },
    "semantic_cell_fraction": 0.125,
    "semantic_green_conf_thr": 0.5,
    "semantic_min_votes": 2,
    "semantic_monolayer_weight": 1.0,
    "semantic_other_weight": 1.0,
    "navigation": {"enter_score": 0.65, "exit_score": 0.45},
    "strong_single_thr": 0.75,
    "enter_votes": 2,
}


def fake_job(job_id, x, y, w, h, score, raw_class):
    return SimpleNamespace(
        job_id=job_id, frame_idx=job_id, t_capture=time.perf_counter(),
        cumulative_x=x, cumulative_y=y, map_x=x, map_y=y,
        labels=np.zeros((h, w), dtype=np.uint16), score=score,
        raw_class=raw_class, priority="MANUAL", human_label=None,
        reg_method="phase", reg_state="REGISTRATION_GOOD",
        reg_confidence=0.9, reg_dx=0.0, reg_dy=0.0, reg_ref_job_id=-1,
        drift_warning=False,
    )


def build_survey_map():
    sm = ScanMap(CFG)
    jid = 1
    # 3x3 cluster of MONOLAYER fields centred at (400, 300), spaced 180 px
    for row in range(3):
        for col in range(3):
            sm.add_field(fake_job(jid, 400 + col * 180, 300 + row * 180,
                                  400, 224, 0.85, "MONOLAYER"))
            jid += 1
    # non-monolayer far away
    sm.add_field(fake_job(jid, 1600, 1200, 400, 224, 0.40, "TOO_THIN"))
    return sm


def test_finalize_and_load(tmp: Path):
    sm = build_survey_map()
    exp = ExplorationManager(CFG)
    paths = exp.finalize(sm, mosaic=None, out_dir=tmp / "survey_final",
                         metadata={"session": "test"})
    for key in ("survey_dir", "manifest", "monolayer_mask"):
        assert key in paths, f"missing export path {key}"
    assert (tmp / "survey_final" / "survey.json").exists()
    assert (tmp / "survey_final" / "placements.json").exists()
    assert (tmp / "survey_final" / "monolayer_conf.npy").exists()

    ref = exp.survey
    assert ref is not None
    n_green = int(ref.monolayer_mask.sum())
    assert n_green > 0, "monolayer mask must be non-empty after finalize"
    assert ref.monolayer_mask.shape == ref.monolayer_conf.shape
    assert len(ref.placements) == 10, "all fields must be recorded"
    assert ref.calibration.objective == "10x"
    assert abs(ref.calibration.pixels_per_micron - 1.0) < 1e-9
    # physical bounds in microns = scan px (ppm 1.0)
    x0, y0, x1, y1 = ref.bounds_um()
    assert x1 > x0 > 0 and y1 > y0 > 0

    # restart-safety: load the bundle from disk
    ref2 = load_survey(tmp / "survey_final")
    assert ref2.monolayer_mask.shape == ref.monolayer_mask.shape
    assert int(ref2.monolayer_mask.sum()) == n_green
    assert abs(ref2.origin_x - ref.origin_x) < 1e-6
    assert abs(ref2.cell - ref.cell) < 1e-6
    assert len(ref2.placements) == len(ref.placements)
    assert abs(ref2.calibration.pixels_per_micron - 1.0) < 1e-9
    print(f"  finalize/load OK ({n_green} green cells, "
          f"grid {ref.monolayer_mask.shape}, {len(ref.placements)} placements)")
    return ref2


def test_containment(ref: SurveyReference):
    greens = np.argwhere(ref.monolayer_mask)
    r0, c0 = greens[greens[:, 0].argmin()]     # topmost green cell
    inside_xy = (ref.origin_x + (c0 + 0.5) * ref.cell,
                 ref.origin_y + (r0 + 0.5) * ref.cell)
    res = containment(inside_xy, ref, near_edge_um=120.0)
    assert res["state"] == INSIDE, f"expected INSIDE, got {res['state']}"

    # 2.2 cells ABOVE the top edge -> outside the mask, ~110 um from it
    near_xy = (inside_xy[0], inside_xy[1] - 2.2 * ref.cell)
    res = containment(near_xy, ref, near_edge_um=120.0)
    # cell = 0.125 * 400 = 50 px -> 2.2 cells = 110 um -> NEAR EDGE
    assert res["state"] == NEAR_EDGE, \
        f"expected NEAR EDGE, got {res['state']} at {res['dist_um']} um"
    assert res["dist_um"] <= 120.0

    far_xy = (near_xy[0], near_xy[1] - 8 * ref.cell)
    res = containment(far_xy, ref, near_edge_um=120.0)
    assert res["state"] == OUTSIDE, f"expected OUTSIDE, got {res['state']}"
    g = res["guidance_scan"]
    assert g is not None and abs(math.hypot(*g) - 1.0) < 1e-6, \
        "guidance must be a unit vector"
    # guidance must point TOWARD the nearest monolayer cell (downward here)
    to_nearest = (res["nearest_scan"][0] - far_xy[0],
                  res["nearest_scan"][1] - far_xy[1])
    n = math.hypot(*to_nearest)
    dot = g[0] * to_nearest[0] / n + g[1] * to_nearest[1] / n
    assert dot > 0.999, "guidance vector must point toward the monolayer"
    assert g[1] > 0, "nearest monolayer is below - guidance y must be positive"
    # micron reporting consistent with ppm
    assert abs(res["dist_um"] - n * ref.ppm) < 0.5
    print(f"  containment OK (inside/near {res['dist_um']}um/outside + guidance)")


def test_revisit_points(ref: SurveyReference):
    greens = np.argwhere(ref.monolayer_mask)
    for mode in ("grid", "top_confidence", "centroid"):
        pts = generate_revisit_points(ref, mode=mode, count=6,
                                      min_sep_cells=2.0)
        assert pts, f"{mode}: no points generated"
        if mode == "centroid":
            assert len(pts) == 1
        assert len(pts) <= 6
        for p in pts:
            col = int((p["x_scan"] - ref.origin_x) / ref.cell)
            row = int((p["y_scan"] - ref.origin_y) / ref.cell)
            assert 0 <= row < ref.monolayer_mask.shape[0] and \
                0 <= col < ref.monolayer_mask.shape[1], \
                f"{mode}: point outside grid"
            assert ref.monolayer_mask[row, col], \
                f"{mode}: revisit point NOT on a monolayer cell"
            assert abs(p["x_um"] - p["x_scan"] * ref.ppm) < 0.2
        # pairwise separation
        for i in range(len(pts)):
            for k in range(i + 1, len(pts)):
                d_cells = math.hypot(
                    (pts[i]["x_scan"] - pts[k]["x_scan"]) / ref.cell,
                    (pts[i]["y_scan"] - pts[k]["y_scan"]) / ref.cell)
                assert d_cells >= 2.0 - 1e-6, \
                    f"{mode}: points too close ({d_cells:.2f} cells)"
        print(f"  revisit[{mode}] OK ({len(pts)} pts, all on monolayer cells)")


def test_relocalizer():
    rng = np.random.default_rng(7)
    mosaic = (rng.random((240, 360, 3)) * 255).astype(np.uint8)
    # add structure so NCC is meaningful (random noise alone already matches)
    mosaic = cv2_gauss(mosaic)
    ref = SurveyReference(
        monolayer_mask=np.zeros((8, 8), dtype=bool), monolayer_conf=None,
        coverage_grid=None, origin_x=-20.0, origin_y=-20.0, cell=50.0,
        placements=[], calibration=Calibration("10x", 1.0), metadata={},
        mosaic_bgr=mosaic, mosaic_origin=(0.0, 0.0))
    reloc = Relocalizer(ref, CFG)

    # "high-mag frame" = 4x upscaled crop at known survey position
    cx, cy, fw, fh = 180.0, 90.0, 80.0, 60.0      # crop center + size in ref px
    x0, y0 = int(cx - fw / 2), int(cy - fh / 2)
    crop = mosaic[y0:y0 + int(fh), x0:x0 + int(fw)]
    frame = cv2_resize4x(crop)                     # 320x240 at 40x (ppm 4.0)

    hit = reloc.relocate(frame, high_ppm=4.0)
    assert hit is not None, "relocalization must find the crop"
    dx = hit["x_scan"] - cx
    dy = hit["y_scan"] - cy
    assert math.hypot(dx, dy) <= 4.0, \
        f"pose error too large: ({dx:.1f}, {dy:.1f}) px"
    assert hit["response"] >= 0.45

    # featureless frame -> honest failure (§8)
    blank = np.full((240, 320, 3), 90, dtype=np.uint8)
    assert reloc.relocate(blank, high_ppm=4.0) is None, \
        "featureless field must NOT yield a pose"
    print(f"  relocalizer OK (err ({dx:.1f},{dy:.1f}) px, resp "
          f"{hit['response']:.2f}; blank rejected)")


def cv2_gauss(img):
    import cv2
    return cv2.GaussianBlur(img, (0, 0), 2.0)


def cv2_resize4x(img):
    import cv2
    return cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4),
                      interpolation=cv2.INTER_CUBIC)


def test_exploration_transform():
    """§4: high-mag chain delta scaled by the calibration ratio into survey
    coordinates + §5 containment + frozen map untouched."""
    sm = build_survey_map()
    exp = ExplorationManager(CFG)
    tmp = Path(tempfile.mkdtemp(prefix="ph13_transform_"))
    exp.finalize(sm, mosaic=None, out_dir=tmp / "survey_final")
    exp.set_active_objective("40x")
    assert exp.high_mag_active(), "40x must activate exploration mode"
    exp.set_anchor(400.0, 300.0, reason="test")

    class StubChain:
        def __init__(self, gx, gy, dx=80.0, dy=-40.0):
            self.gx, self.gy, self.dx, self.dy = gx, gy, dx, dy

        def register_field(self, job, frame):
            # pretend the chain measured (dx, dy) high-mag px from its origin
            return SimpleNamespace(global_x=self.gx + self.dx,
                                   global_y=self.gy + self.dy,
                                   state="REGISTRATION_GOOD", method="phase",
                                   confidence=0.9, dx=self.dx, dy=self.dy,
                                   ref_job_id=1, drift_warning=False)

    exp._chain_origin = (1000.0, 2000.0)   # chain's first-field pose
    exp._chain = StubChain(1000.0, 2000.0)

    st = SimpleNamespace(lock=threading.Lock(), exploration={})
    job = fake_job(99, 0, 0, 100, 100, 0.8, "MONOLAYER")
    job.frame = np.zeros((100, 100, 3), dtype=np.uint8)
    job.features = {}
    n_fields_before = len(sm.fields())
    exp.process_field(job, st)

    scale = 1.0 / 4.0                       # ref_ppm / high_ppm
    expect_x = 400.0 + 80.0 * scale         # 420
    expect_y = 300.0 - 40.0 * scale         # 290
    assert abs(job.map_x - expect_x) < 1e-6 and abs(job.map_y - expect_y) < 1e-6, \
        f"transform wrong: got ({job.map_x}, {job.map_y})"
    assert st.exploration["state"] == INSIDE, \
        f"(420, 290) is inside the monolayer cluster, got {st.exploration['state']}"
    assert job.features["exploration_state"] == INSIDE
    assert len(sm.fields()) == n_fields_before, "frozen map must gain no fields"
    assert exp.survey.monolayer_mask.sum() > 0, "frozen mask must be untouched"

    # far outside pose -> OUTSIDE with guidance
    # chain delta +4000 high-mag px * 0.25 = +1000 scan px -> pose (1400, 300),
    # far right of the monolayer cluster (green union ends near x=960)
    exp.set_anchor(400.0, 300.0, reason="test-far")
    exp._chain = StubChain(1000.0, 2000.0, dx=4000.0, dy=0.0)
    exp._chain_origin = (1000.0, 2000.0)
    job2 = fake_job(100, 0, 0, 100, 100, 0.8, "MONOLAYER")
    job2.frame = np.zeros((100, 100, 3), dtype=np.uint8)
    job2.features = {}
    exp.process_field(job2, st)
    assert st.exploration["state"] == OUTSIDE
    assert st.exploration["guidance"] is not None
    print(f"  exploration transform OK (pose ({job.map_x:.0f}, {job.map_y:.0f}) "
          f"INSIDE; far pose OUTSIDE with guidance)")

    # relocalization failure path: no mosaic in survey -> relocate returns None
    hit = exp.relocate(np.zeros((100, 100, 3), dtype=np.uint8))
    assert hit is None, "no mosaic bundle -> relocate must refuse (no guess)"


def test_worker_accepts_exploration():
    from src.ui.workers import FeatureWorker
    from src.ui.state import SharedState
    st = SharedState()
    w = FeatureWorker("cpuT", st, CFG, jobmgr=None, maskq=None,
                      exploration=ExplorationManager(CFG))
    assert w.exploration is not None
    print("  FeatureWorker exploration wiring OK")


def main() -> int:
    t0 = time.perf_counter()
    print("[phase13] survey finalize / load ...")
    with tempfile.TemporaryDirectory(prefix="ph13_") as td:
        ref = test_finalize_and_load(Path(td))
        print("[phase13] containment ...")
        test_containment(ref)
        print("[phase13] revisit points ...")
        test_revisit_points(ref)
    print("[phase13] relocalizer ...")
    test_relocalizer()
    print("[phase13] exploration transform ...")
    test_exploration_transform()
    print("[phase13] worker wiring ...")
    test_worker_accepts_exploration()
    print(f"[phase13] ALL PASSED in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
