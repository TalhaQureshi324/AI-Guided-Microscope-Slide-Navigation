#!/usr/bin/env python
"""Unit tests for the cross-magnification experiment's geometric core.

Tests (run with the cellpose_env python):
  1. Homography point-mapping convention (p_parent ~ H @ p_child).
  2. Composition order H_100_to_10 = H_40_to_10 @ H_100_to_40 on known points.
  3. Scale-assist algebra: H_small @ S maps original child pixels correctly.
  4. Sanity checker rejects mirrored / oversized / degenerate transforms.
  5. SIFT END-TO-END recovery of a known homography on a synthetic image pair
     (integration of the exact register_pair machinery used by the experiment).
  6. Polygon IoU helper on a known geometry.

Run:  python test_composition.py
Exit code 0 = all passed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_magnification_register as cmr  # noqa: E402

RESULTS = []


def check(name, cond):
    RESULTS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)
    return bool(cond)


def synthetic_texture(w=1400, h=1000, seed=7):
    """Deterministic rich texture (blobs + lines + checker) for SIFT."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 210, np.uint8)
    for _ in range(260):
        x, y = rng.integers(0, w), rng.integers(0, h)
        r = int(rng.integers(4, 26))
        col = tuple(int(c) for c in rng.integers(0, 200, 3))
        cv2.circle(img, (int(x), int(y)), r, col, -1, cv2.LINE_AA)
    for _ in range(90):
        p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p0, p1, (60, 60, 60), 2, cv2.LINE_AA)
    for yy in range(0, h, 120):
        for xx in range(0, w, 120):
            cv2.rectangle(img, (xx, yy), (xx + 55, yy + 55), (10, 10, 10), 3)
    return img


def test_convention():
    H = np.array([[0.5, 0, 10], [0, 0.5, 20], [0, 0, 1.0]])
    p = cmr.xform(H, [(2, 4)])[0]
    check("convention: H maps child(2,4) -> parent(11,22)", np.allclose(p, [11, 22]))


def test_composition():
    H_40_to_10 = np.array([[0.25, 0, 5], [0, 0.25, 7], [0, 0, 1.0]])
    H_100_to_40 = np.array([[0.4, 0, 1], [0, 0.4, 2], [0, 0, 1.0]])
    comp = H_40_to_10 @ H_100_to_40
    pts = rng_pts(50, 2000, seed=3)
    seq = cmr.xform(H_40_to_10, cmr.xform(H_100_to_40, pts))
    dire = cmr.xform(comp, pts)
    check("composition: sequential == composed (50 random points)",
          np.allclose(seq, dire, atol=1e-9))
    check("composition: pure scales 0.25 * 0.4 == 0.1 (1/10)",
          abs(comp[0, 0] - 0.1) < 1e-12)
    check("composition: translation composes as t1 + A1*t2",
          abs(comp[0, 2] - (5 + 0.25 * 1)) < 1e-12 and abs(comp[1, 2] - (7 + 0.25 * 2)) < 1e-12)


def test_scale_assist():
    s = 4.0
    H_small = np.array([[0.25, 0, 10], [0, 0.25, 10], [0, 0, 1.0]])
    H_full = H_small @ cmr.scale_similarity_matrix(s)
    a = cmr.xform(H_full, [(40, 40)])[0]
    b = cmr.xform(H_small, [(10, 10)])[0]
    check("scale-assist: H_small @ S == mapping through the resized image",
          np.allclose(a, b))


def test_sanity_rejections():
    shape_c, shape_p = (1080, 1920, 3), (1080, 1920, 3)
    big = np.array([[8.0, 0, -3000], [0, 8.0, -3000], [0, 0, 1.0]])
    ok, _, why = cmr.sanity_check(big, shape_c, shape_p)
    check("sanity: oversized footprint rejected", (not ok) and any("impossible" in w for w in why))

    mir = np.array([[-0.25, 0, 900], [0, 0.25, 100], [0, 0, 1.0]])
    ok, _, why = cmr.sanity_check(mir, shape_c, shape_p)
    check("sanity: mirrored transform rejected", (not ok) and any("mirror" in w for w in why))

    tiny = np.array([[0.01, 0, 900], [0, 0.01, 500], [0, 0, 1.0]])
    ok, _, why = cmr.sanity_check(tiny, shape_c, shape_p)
    check("sanity: degenerate footprint rejected", not ok)

    # extreme aspect: stretch x by 5x relative to y
    skew = np.array([[0.6, 0, 300], [0, 0.12, 400], [0, 0, 1.0]])
    ok, _, why = cmr.sanity_check(skew, shape_c, shape_p)
    check("sanity: extreme aspect distortion rejected",
          (not ok) and any("aspect" in w for w in why))

    good = np.array([[0.25, 0, 500], [0, 0.25, 300], [0, 0, 1.0]])
    ok, _, why = cmr.sanity_check(good, shape_c, shape_p)
    check("sanity: sane similarity transform accepted", ok and not why)


def rng_pts(n, k, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, k, (n, 2))


def test_iou():
    shape = (1000, 1000)
    q1 = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], float)
    q2 = np.array([[200, 100], [400, 100], [400, 300], [200, 300]], float)
    # IoU is rasterized, so shared boundary pixels land in both masks;
    # allow the small rasterization tolerance
    iou = cmr.quad_iou(q1, q2, shape)
    check(f"IoU half-overlap ~= 1/3 (got {iou:.4f})", abs(iou - 1 / 3) < 0.02)
    check("IoU identical quads == 1",
          abs(cmr.quad_iou(q1, q1, shape) - 1.0) < 1e-6)
    q3 = q2 + np.array([900, 0])
    check("IoU disjoint quads == 0", cmr.quad_iou(q1, q3, shape) == 0.0)


def test_sift_end_to_end():
    """Integration: build child = warp of parent by inv(H_true), then run the
    EXPERIMENT's own register_pair machinery and confirm it recovers H_true
    and a sane footprint (scale-assisted, as the effective gap here is 4x)."""
    parent = synthetic_texture()
    h, w = parent.shape[:2]
    th = 15 * math.pi / 180
    R = np.array([[math.cos(th), -math.sin(th), 0],
                  [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
    H_true = np.array([[0.25, 0, 480], [0, 0.25, 350], [0, 0, 1.0]]) @ R
    H_true /= H_true[2, 2]
    child = cv2.warpPerspective(parent, np.linalg.inv(H_true), (700, 560),
                                flags=cv2.INTER_LINEAR)
    res = cmr.register_pair(parent, child, "TEST", [3.0, 3.5, 4.0, 4.5, 5.0])
    rec = res.get("best")
    check(f"end-to-end: registration recovered (status {res['status']})",
          rec is not None)
    if rec is None:
        return
    H = np.asarray(res["ctx"]["H"], np.float64)
    pts = rng_pts(200, 600, seed=5)
    err = np.linalg.norm(cmr.xform(H, pts) - cmr.xform(H_true, pts), axis=1)
    # linear scale = norm of a column of the linear part (not Frobenius!)
    lin = float(np.linalg.norm(H_true[:2, 0]))
    ok = (res["status"] in ("GOOD", "WEAK")
          and err.mean() < 5.0  # px at parent scale
          and abs(rec["effective_scale"] - lin) < 0.15 * lin)
    check(f"end-to-end: recovered H matches ground truth "
          f"(status {res['status']}, mean dev {err.mean():.2f}px, measured scale "
          f"{rec['effective_scale']:.3f} vs true {lin:.3f})", ok)


def main() -> int:
    print("=== composition / convention / sanity tests ===", flush=True)
    test_convention()
    test_composition()
    test_scale_assist()
    test_sanity_rejections()
    test_iou()
    print("=== SIFT end-to-end integration test (synthetic) ===", flush=True)
    test_sift_end_to_end()
    n_ok = sum(RESULTS)
    print(f"=== {n_ok}/{len(RESULTS)} tests passed "
          f"({'ALL PASSED' if n_ok == len(RESULTS) else 'FAILURES PRESENT'}) ===",
          flush=True)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
