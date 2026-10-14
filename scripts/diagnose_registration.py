"""Diagnose registration on REAL keyframes from a live session.

For each consecutive keyframe pair:
  * phase-correlation proposal + response (what the live tracker used)
  * ORB proposal + match stats
  * brute-force ground truth: exhaustive overlap-NCC search (coarse grid at
    quarter prep res, refined at full prep res)
  * NCC evaluated at each proposal

Run: python scripts/diagnose_registration.py <session_dir>
"""

from __future__ import annotations

import glob
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.live.registration import _orb_translation, _phase_corr, overlap_ncc, prep_frame


def load_keyframes(session_dir: Path):
    out = []
    for p in sorted(glob.glob(str(session_dir / "keyframes" / "job_*.jpg"))):
        m = re.search(r"job_(\d+)", Path(p).name)
        if m:
            out.append((int(m.group(1)), cv2.imread(p)))
    out.sort(key=lambda t: t[0])
    return out


def brute_force(ref: np.ndarray, cur: np.ndarray, coarse_step: int = 2):
    """Exhaustive overlap-NCC translation search. Returns (dx, dy, ncc) in
    PREP px, convention: cur(x) = ref(x - dx)."""
    h, w = ref.shape[:2]
    small_r = cv2.resize(ref, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    small_c = cv2.resize(cur, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    sh, sw = small_r.shape
    best = (0.0, 0.0, -2.0)
    for dy in range(-(sh - 24), sh - 24 + 1, coarse_step):
        for dx in range(-(sw - 24), sw - 24 + 1, coarse_step):
            n = overlap_ncc(small_r, small_c, float(dx), float(dy))
            if n > best[2]:
                best = (float(dx) * 2, float(dy) * 2, n)
    # refine +-2*coarse_step at full prep res
    bx, by, bn = best
    for dy in range(int(by) - 4, int(by) + 5):
        for dx in range(int(bx) - 4, int(bx) + 5):
            n = overlap_ncc(ref, cur, float(dx), float(dy))
            if n > bn:
                bx, by, bn = float(dx), float(dy), n
    return bx, by, bn


def main() -> int:
    session_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        REPO / "outputs" / "live_session_20261014_110436"
    frames = load_keyframes(session_dir)
    print(f"{len(frames)} keyframes loaded from {session_dir}")
    if len(frames) < 2:
        print("need >= 2 keyframes")
        return 1

    # ---- empirical phaseCorrelate sign check ------------------------------
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur((rng.random((269, 480)) * 255).astype(np.uint8),
                            (0, 0), 3).astype(np.float32)
    shifted = np.roll(base, 30, axis=1)   # content moved RIGHT by 30
    sx, sy, resp = _phase_corr(base, shifted, 1.0)
    print(f"\n[sign check] roll(base, +30px x): phaseCorrelate -> "
          f"({sx:+.1f}, {sy:+.1f}) resp {resp:.2f}  "
          f"(>0 means phaseCorr reports +dx for content moved right)")

    print(f"\n{'pair':>10} | {'phase prop':>14} {'resp':>5} | "
          f"{'ORB dx,dy':>12} {'match':>5} {'inl':>4} | "
          f"{'TRUTH dx,dy':>14} {'NCC':>5} | {'NCC@phase':>9} {'NCC@orb':>8}")
    for (jid_a, fa), (jid_b, fb) in zip(frames, frames[1:]):
        ra, std_a = prep_frame(fa)
        rb, std_b = prep_frame(fb)
        pdx, pdy, resp = _phase_corr(ra, rb, 1.0)
        odx, ody, oconf, nmatch, inl = _orb_translation(ra, rb, 1.0, {})
        tx, ty, tn = brute_force(ra, rb)
        ncc_at_phase = overlap_ncc(ra, rb, pdx, pdy)
        ncc_at_orb = overlap_ncc(ra, rb, odx, ody)
        print(f"#{jid_a:>3}->#{jid_b:<3} | ({pdx:+7.1f},{pdy:+6.1f}) {resp:5.2f} | "
              f"({odx:+5.1f},{ody:+5.1f}) {nmatch:5d} {inl:4.2f} | "
              f"({tx:+7.1f},{ty:+6.1f}) {tn:5.2f} | {ncc_at_phase:9.3f} "
              f"{ncc_at_orb:8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
