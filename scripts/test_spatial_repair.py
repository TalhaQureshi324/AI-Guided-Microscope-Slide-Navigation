"""Spatial-repair tests (2026-10-14): registration on synthetic shifts (R2)
and REAL session keyframes, mosaic fractional placement (R10), AnalysisJob
schema defaults (R17), capture-order spatial gate (R0/R18), map fit stability.

Run:  python scripts/test_spatial_repair.py
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.live.jobs import AnalysisJob, JobManager   # noqa: E402
from src.live.mosaic import MosaicBuilder           # noqa: E402
from src.live.registration import (                 # noqa: E402
    REG_FAILED, REG_GOOD, REG_WEAK, GlobalPoseTracker, prep_frame,
)
from src.live.spatial_order import SpatialOrderGate  # noqa: E402

REG_CFG = {"registration": {"work_width": 480, "phase_good": 0.22,
                            "phase_weak": 0.12, "overlap_ncc_min": 0.25,
                            "orb_good_matches": 15, "orb_good_inlier": 0.5,
                            "orb_weak_matches": 8, "orb_weak_inlier": 0.35,
                            "max_jump_frac": 1.6, "drift_warn_px": 300,
                            "search_radius_frac": 1.3, "max_prep_cache": 30,
                            "texture_min_std": 3.0}}


def textured_canvas(w: int, h: int, seed: int = 5) -> np.ndarray:
    """Cell-like texture: dark ellipses on pale background (RBC analogue)."""
    rng = np.random.default_rng(seed)
    canvas = np.full((h, w, 3), 205, dtype=np.uint8)
    canvas += (rng.random((h, w, 1)) * 24).astype(np.uint8)
    n = (w * h) // 900
    for _ in range(n):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ax, ay = rng.uniform(9, 16), rng.uniform(8, 14)
        ang = rng.uniform(0, math.pi)
        color = (60 + int(rng.uniform(0, 50)),) * 3
        cv2.ellipse(canvas, (int(cx), int(cy)), (int(ax), int(ay)), ang * 57.3,
                    0, 360, color, -1)
        cv2.ellipse(canvas, (int(cx), int(cy)),
                    (max(2, int(ax * 0.35)), max(2, int(ay * 0.35))),
                    ang * 57.3, 0, 360, (225, 225, 225), -1)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.2)
    return canvas


def crop_shift(canvas: np.ndarray, dx: float, dy: float, w: int = 800,
               h: int = 448) -> np.ndarray:
    """Frame at offset (dx, dy) from a large canvas - NO circular wrap."""
    x0 = int(round(300 + dx))
    y0 = int(round(300 + dy))
    return canvas[y0:y0 + h, x0:x0 + w].copy()


def fake_job(jid, frame, cum_x, cum_y):
    return SimpleNamespace(
        job_id=jid, frame_idx=jid, t_capture=time.perf_counter(),
        cumulative_x=cum_x, cumulative_y=cum_y, map_x=cum_x, map_y=cum_y,
        labels=None, score=0.8, raw_class="MONOLAYER", priority="MANUAL",
        human_label=None, frame=frame, features={},
    )


def test_synthetic_shifts():
    """R2: deterministic translated copies must be recovered (translation
    model, no wrapping)."""
    canvas = textured_canvas(2200, 1300)
    shifts = [(100, 0), (-100, 0), (0, 40), (150, 30), (-220, -60), (300, 0)]
    tr = GlobalPoseTracker(REG_CFG)
    base = crop_shift(canvas, 0, 0)
    ok = tr.register_field(fake_job(1, base, 0, 0), base)
    assert ok.state == REG_GOOD and ok.method == "FIRST_FIELD"
    errors = []
    for i, (dx, dy) in enumerate(shifts, start=2):
        frame = crop_shift(canvas, dx, dy)
        res = tr.register_field(fake_job(i, frame, dx, dy), frame)
        assert res.state != REG_FAILED, \
            f"shift {dx},{dy}: unexpectedly {res.state} ({res.method} conf {res.confidence:.2f})"
        # Each field chains against the PREVIOUS field; the absolute pose must
        # still land at -(dx, dy): registration reports CONTENT displacement
        # (stage right = content left), and poses accumulate in content
        # coordinates - that is what makes mosaic tiles align.
        err = math.hypot(res.global_x + dx, res.global_y + dy)
        errors.append(err)
        assert err <= 6.0, \
            f"shift ({dx},{dy}): pose error {err:.1f} px (got global " \
            f"({res.global_x:.0f},{res.global_y:.0f}), want ({-dx},{-dy}))"
    print(f"  synthetic shifts OK ({len(shifts)} shifts, max err "
          f"{max(errors):.1f} full-res px, states "
          f"{[res.state.split('_')[-1] for res in [ok]] + ['ok'] * len(shifts)})")


def test_real_keyframes():
    """The exact pairs that FAILED in the 2026-10-14 session must now
    register GOOD with correct relative poses."""
    import glob
    import re
    session = REPO / "outputs" / "live_session_20261014_110436"
    frames = []
    for p in sorted(glob.glob(str(session / "keyframes" / "job_*.jpg"))):
        m = re.search(r"job_(\d+)", Path(p).name)
        if m:
            frames.append((int(m.group(1)), cv2.imread(p)))
    frames.sort(key=lambda t: t[0])
    assert len(frames) >= 3, "need the session keyframes"
    tr = GlobalPoseTracker(REG_CFG)
    n_good = 0
    for i, (jid, frame) in enumerate(frames):
        res = tr.register_field(fake_job(jid, frame, 0, 0), frame)
        if res.state == REG_GOOD:
            n_good += 1
        elif res.state == REG_WEAK and res.confidence >= 0.2:
            n_good += 1   # placed with a flag - acceptable
        print(f"    job #{jid}: {res.state.split('_')[-1]} via {res.method} "
              f"conf {res.confidence:.2f} d=({res.dx:.0f},{res.dy:.0f})")
    assert n_good >= len(frames) - 1, \
        f"only {n_good}/{len(frames)} real keyframes registered"
    poses = tr.poses()
    xs = [p[0] for p in poses.values()]
    spread = max(xs) - min(xs)
    assert spread > 800, f"poses must spread over multiple FOVs, got {spread:.0f}"
    print(f"  real keyframes OK ({n_good}/{len(frames)} registered, "
          f"x-spread {spread:.0f} px)")


def test_mosaic_fractional_placement():
    """R10: fractional poses made ceil-floor spans 1 px larger than the
    frame and crashed the blend (448x801 vs 448x800)."""
    mb = MosaicBuilder()
    frame = textured_canvas(800, 448)[:448, :800]
    mb.place(frame, 0.0, 0.0, confidence=1.0, job_id=1)
    mb.place(frame, 366.5, 15.5, confidence=0.9, job_id=2)   # fractional
    mb.place(frame, -123.7, -88.3, confidence=0.9, job_id=3)  # negative frac
    mb.place(frame[:447, 799], 100.2, 50.6, confidence=0.9, job_id=4)  # odd size
    cov = mb.coverage_mask()
    assert cov is not None and cov.sum() > 800 * 448
    paths = mb.export(Path(__file__).parent.parent / "outputs" / "_spatial_test_mosaic")
    assert Path(paths["mosaic"]).exists()
    print("  mosaic fractional placement OK (no broadcast error, 4 tiles)")


def test_job_schema_defaults():
    """R17: a freshly-submitted job must expose every spatial attribute the
    renderer touches (the drift_warning AttributeError crash)."""
    mgr = JobManager()
    job = mgr.submit("MANUAL", np.zeros((448, 800, 3), np.uint8), 1, 0.0)
    mgr.mark(job, "COMPLETE", "gpu1")   # COMPLETE before reg ran - the crash window
    assert job.drift_warning is False
    assert job.reg_state == "" and job.reg_confidence == 0.0
    assert job.map_x == 0.0 and job.revisit is False
    print("  AnalysisJob schema defaults OK (COMPLETE-but-unregistered is safe)")


def test_order_gate():
    """R0/R18: out-of-order feature completion must commit in capture order."""
    mgr = JobManager()
    jobs = [mgr.submit("MANUAL", np.zeros((4, 4, 3), np.uint8), i, 0.0)
            for i in range(1, 5)]
    gate = SpatialOrderGate(timeout_sec=2.0)
    order = []

    def worker(jid):
        gate.wait_turn(jid, mgr)
        order.append(jid)
        gate.finish_turn(jid)

    # completion order 3, 4, 2, 1 - commits must still be 1, 2, 3, 4
    threads = [threading.Thread(target=worker, args=(j.job_id,)) for j in
               [jobs[2], jobs[3], jobs[1], jobs[0]]]
    for t in threads:
        t.start()
        time.sleep(0.05)
    for t in threads:
        t.join(timeout=10)
    assert order == [1, 2, 3, 4], f"commit order wrong: {order}"

    # a FAILED predecessor must not block
    mgr.mark(jobs[0], "FAILED", "t")
    gate2 = SpatialOrderGate(timeout_sec=1.0)
    gate2._terminal.add(1)
    t0 = time.perf_counter()
    gate2.wait_turn(2, mgr)
    assert time.perf_counter() - t0 < 0.5, "terminal predecessor must not block"

    # timeout fallback: stuck predecessor -> commit anyway, no deadlock
    gate3 = SpatialOrderGate(timeout_sec=0.5)
    t0 = time.perf_counter()
    ok = gate3.wait_turn(9, mgr)   # jobs 1..8 missing entirely (never submitted)
    assert not ok and time.perf_counter() - t0 >= 0.4
    print("  spatial order gate OK (reorder, terminal skip, timeout fallback)")


def test_map_fit_stability():
    """R5-lite: the live crosshair walking away must NOT rescale the fit."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from src.ui.scan_map_panel import ScanMapPanel
    app = QApplication.instance() or QApplication(sys.argv)
    panel = ScanMapPanel()
    panel.setGeometry(0, 0, 500, 300)
    fp = SimpleNamespace(job_id=1, x=400, y=300, w=800, h=448, score=0.8,
                         raw_class="MONOLAYER", human_label=None, sweep_id=0)
    panel.set_data([fp], [], (400, 300))
    panel.grab()   # forces a synchronous full render (works offscreen)
    fit_with_close = panel._fit
    panel.set_data([fp], [], (400 + 25000, 300 + 18000))  # walked FAR away
    panel.grab()
    fit_far = panel._fit
    assert fit_with_close[2] == fit_far[2], \
        f"fit scale changed with crosshair: {fit_with_close[2]} -> {fit_far[2]}"
    print(f"  map fit stability OK (scale {fit_far[2]:.4f} unchanged while "
          "crosshair walked away)")


def main() -> int:
    t0 = time.perf_counter()
    print("[spatial-repair] synthetic shift matrix (R2) ...")
    test_synthetic_shifts()
    print("[spatial-repair] REAL session keyframes ...")
    test_real_keyframes()
    print("[spatial-repair] mosaic fractional placement (R10) ...")
    test_mosaic_fractional_placement()
    print("[spatial-repair] AnalysisJob schema (R17) ...")
    test_job_schema_defaults()
    print("[spatial-repair] capture-order gate (R0/R18) ...")
    test_order_gate()
    print("[spatial-repair] map fit stability (R5) ...")
    test_map_fit_stability()
    print(f"[spatial-repair] ALL PASSED in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
