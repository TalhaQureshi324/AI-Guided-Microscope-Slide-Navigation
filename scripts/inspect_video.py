"""Phase V1 - microscope video inspection (no expensive analysis).

Reports video metadata and characterises movement / quality over a uniform
sample of frames:
  * sharpness (variance of Laplacian), brightness, contrast, saturation,
    over/under-exposure percentage per sampled frame;
  * inter-sample displacement via phase correlation (downscaled grayscale)
    to expose stationary periods, speed changes and motion blur.

Outputs (outputs/video_inspection/):
  inspection_report.json   - aggregate statistics and thresholds guidance
  frame_samples.csv        - one row per sampled frame
  motion_samples.csv       - displacement between consecutive samples

This script never writes into Dataset/ and never touches the static baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.runtime import Timer, setup_logging  # noqa: E402

logger = logging.getLogger("inspect_video")

MOTION_WORK_WIDTH = 480  # downscale width for phase correlation work


def frame_quality(frame_bgr: np.ndarray) -> dict:
    """Lightweight per-frame quality measurements."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    contrast = float(gray.std())
    saturation = float(hsv[:, :, 1].mean())
    over = float((gray >= 250).mean() * 100.0)
    under = float((gray <= 5).mean() * 100.0)
    return {
        "sharpness_lapvar": sharpness,
        "brightness": bright,
        "contrast": contrast,
        "saturation": saturation,
        "overexposed_pct": over,
        "underexposed_pct": under,
    }


def to_motion_gray(frame_bgr: np.ndarray, width: int = MOTION_WORK_WIDTH) -> np.ndarray:
    scale = width / frame_bgr.shape[1]
    small = cv2.resize(frame_bgr, (width, int(round(frame_bgr.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=REPO_ROOT / "Dataset" / "video_dataset.mp4")
    parser.add_argument("--sample-every", type=int, default=10,
                        help="sample every Nth frame for quality/motion statistics")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "video_inspection")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.out / "inspection.log", verbose=args.verbose)
    args.out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        logger.error("Could not open video: %s", args.video)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    logger.info("video=%s  %dx%d @ %.2f fps (avg), %d frames, codec=%s",
                args.video.name, width, height, fps, n_meta, codec)

    timer = Timer()
    frame_rows: list[dict] = []
    motion_rows: list[dict] = []
    prev_small: np.ndarray | None = None

    idx = 0
    n_decoded = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % args.sample_every != 0:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        n_decoded += 1

        q = frame_quality(frame)
        row = {"frame_index": idx, "timestamp_sec": round(t_sec, 3), **q}
        frame_rows.append(row)

        small = to_motion_gray(frame)
        if prev_small is not None:
            (shift, response) = cv2.phaseCorrelate(prev_small, small)
            scale = width / MOTION_WORK_WIDTH
            dx_full = float(shift[0]) * scale
            dy_full = float(shift[1]) * scale
            motion_rows.append({
                "sample_index": len(motion_rows),
                "frame_index": idx,
                "dx_px": round(dx_full, 2),
                "dy_px": round(dy_full, 2),
                "displacement_px": round(math.hypot(dx_full, dy_full), 2),
                "response": round(float(response), 4),
            })
        prev_small = small
        idx += 1

    cap.release()
    decode_ms = timer.lap("decode_and_analyze")

    # ------------------------------------------------------------------ stats
    def col(name: str) -> np.ndarray:
        return np.array([r[name] for r in frame_rows], dtype=float)

    sharp = col("sharpness_lapvar")
    bright = col("brightness")
    disp = np.array([m["displacement_px"] for m in motion_rows], dtype=float)
    per_sample_step = disp / args.sample_every  # approx per-frame displacement

    qtiles = lambda a: {  # noqa: E731
        "min": round(float(np.min(a)), 3), "p5": round(float(np.percentile(a, 5)), 3),
        "p25": round(float(np.percentile(a, 25)), 3), "median": round(float(np.median(a)), 3),
        "p75": round(float(np.percentile(a, 75)), 3), "p95": round(float(np.percentile(a, 95)), 3),
        "max": round(float(np.max(a)), 3),
    }

    # stationary / fast sample classification (displacement per sample gap)
    still = float((disp < 20).mean() * 100.0)          # < 20 px between samples
    slow = float(((disp >= 20) & (disp < 100)).mean() * 100.0)
    fast = float((disp >= 100).mean() * 100.0)

    # longest stationary run (consecutive samples with < 20 px gap)
    runs, run = [], 0
    for d in disp:
        if d < 20:
            run += 1
        else:
            if run:
                runs.append(run)
            run = 0
    if run:
        runs.append(run)
    longest_still_sec = (max(runs) * args.sample_every / fps) if runs else 0.0

    report = {
        "video": {
            "path": str(args.video),
            "width": width, "height": height,
            "fps_avg": round(fps, 3),
            "frames_meta": n_meta,
            "frames_decoded_end": idx,
            "duration_sec": round(n_meta / fps, 1) if fps else None,
            "codec": codec,
            "samples_analyzed": n_decoded,
            "sample_every": args.sample_every,
        },
        "sharpness_lapvar_quantiles": qtiles(sharp),
        "brightness_quantiles": qtiles(bright),
        "contrast_quantiles": qtiles(col("contrast")),
        "saturation_quantiles": qtiles(col("saturation")),
        "overexposed_pct_quantiles": qtiles(col("overexposed_pct")),
        "underexposed_pct_quantiles": qtiles(col("underexposed_pct")),
        "motion_per_sample_gap_px": qtiles(disp),
        "motion_per_frame_px_median": round(float(np.median(per_sample_step)), 3),
        "motion_pct_of_samples": {"still_lt20px": round(still, 1),
                                  "slow_20_100px": round(slow, 1),
                                  "fast_ge100px": round(fast, 1)},
        "longest_stationary_sec": round(longest_still_sec, 1),
        "illumination_range": {
            "brightness_spread": round(float(bright.max() - bright.min()), 1),
            "note": "large spread (>~30) suggests exposure/illumination drift",
        },
        "timing": {"decode_analyze_total_ms": round(decode_ms, 1),
                   "per_sample_ms": round(decode_ms / max(1, n_decoded), 1)},
    }

    with open(args.out / "inspection_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for name, rows in (("frame_samples.csv", frame_rows), ("motion_samples.csv", motion_rows)):
        if rows:
            with open(args.out / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    logger.info("samples=%d  sharpness median=%.0f (p5=%.0f, p95=%.0f)",
                n_decoded, np.median(sharp), np.percentile(sharp, 5), np.percentile(sharp, 95))
    logger.info("brightness median=%.1f (min=%.1f, max=%.1f)",
                np.median(bright), bright.min(), bright.max())
    logger.info("motion per sample gap: median=%.0f px, p75=%.0f px, p95=%.0f px",
                np.median(disp), np.percentile(disp, 75), np.percentile(disp, 95))
    logger.info("sample mix: still %.0f%% | slow %.0f%% | fast %.0f%% | longest still %.1f s",
                still, slow, fast, longest_still_sec)
    logger.info("report -> %s", args.out / "inspection_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
