"""Worker-concurrency benchmark (Phase G7, spec §8/§38): 1-4 GPU workers.

Submits the same 4 representative microscope fields as MANUAL jobs to the
job manager and measures, for each GPU worker count:
  * time to first result;
  * total time for all 4;
  * per-job latency (queue wait + Cellpose + features);
  * peak VRAM and any CUDA OOM/errors.

Each configuration loads fresh (models are heavy); Cellpose parameters are
IDENTICAL across configurations - parallelism must not change results (§39).

Usage:
    python scripts/benchmark_workers.py --workers 1,2,3,4
    python scripts/benchmark_workers.py --workers 1,2 --width 960
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.live.controller import LiveFieldController  # noqa: E402
from src.live.jobs import JobManager  # noqa: E402
from src.ui.state import SharedState  # noqa: E402
from src.ui.workers import CaptureWorker, FeatureWorker, GPUWorker, _MaskQueue  # noqa: E402


def build_fields(n: int, width: int) -> list:
    """Representative dense fields from the video prototype keyframes."""
    kf_dir = REPO_ROOT / "outputs" / "video_prototype_01" / "keyframes"
    frames = sorted(kf_dir.glob("keyframe_*.jpg"))
    picked = frames[-n:] if len(frames) >= n else (frames * ((n // len(frames)) + 1))[:n]
    out = []
    for f in picked:
        img = cv2.imread(str(f))
        if width and img.shape[1] > width:
            s = width / img.shape[1]
            img = cv2.resize(img, (width, int(round(img.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        out.append((f.stem, img))
    return out


def run_config(gpu_workers: int, fields, cfg: dict, controller: LiveFieldController):
    from src.live.jobs import JobManager
    from src.ui.state import SharedState
    from src.ui.workers import CaptureWorker, FeatureWorker, GPUWorker, _MaskQueue

    state = SharedState()
    jobmgr = JobManager(max_manual_queue=20)
    maskq = _MaskQueue(maxsize=8)

    class _NoCapture:  # benchmark feeds jobs directly; no camera needed
        reconnect = False
        name = "benchmark"

        def open(self):
            return True

        def read(self):
            time.sleep(0.05)
            return None, time.perf_counter()

        def close(self):
            pass

        def actual_settings(self):
            return {"mode": "benchmark"}

    capture = CaptureWorker(_NoCapture(), state, controller, cfg, jobmgr, session=None)
    gpu = [GPUWorker(f"gpu{i + 1}", state, cfg, jobmgr, maskq)
           for i in range(gpu_workers)]
    cpu = [FeatureWorker(f"cpu{i + 1}", state, cfg, jobmgr, maskq)
           for i in range(int(cfg.get("cpu_feature_workers", 2)))]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    capture.start()
    for w in gpu + cpu:
        w.start()
    time.sleep(0.3)

    submit_t = time.perf_counter()
    first_result_t = None
    n_results = 0
    for stem, img in fields:
        jobmgr.submit("MANUAL", img, frame_idx=-1, t_capture=time.perf_counter(),
                      reason="BENCHMARK")
    while time.perf_counter() - submit_t < 600:
        if n_results == 0 and state.analysis_count >= 1:
            first_result_t = time.perf_counter() - submit_t
            n_results = 1
        if state.analysis_count >= len(fields):
            break
        time.sleep(0.05)
    total_t = time.perf_counter() - submit_t
    state.quit_event.set()
    for w in gpu + cpu + [capture]:
        w.join(timeout=10)

    vram_mb = (torch.cuda.max_memory_allocated() / 1e6
               if torch.cuda.is_available() else None)
    jobs = [jobmgr.all_jobs[jid] for jid in sorted(jobmgr.all_jobs)][-len(fields):]
    latencies = [j.latency_ms for j in jobs if j.latency_ms]
    failed = [j for j in jobs if j.status == "FAILED"]
    return {
        "gpu_workers": gpu_workers,
        "fields": len(fields),
        "first_result_s": round(first_result_t, 2) if first_result_t else None,
        "total_s": round(total_t, 2),
        "throughput_jobs_per_s": round(len(jobs) / max(total_t, 0.01), 3),
        "avg_latency_s": round(statistics.mean(latencies) / 1000.0, 2) if latencies else None,
        "max_latency_s": round(max(latencies) / 1000.0, 2) if latencies else None,
        "peak_vram_mb": round(vram_mb, 0) if vram_mb else None,
        "failed": len(failed),
        "errors": [j.error for j in failed][:3],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "live.yaml")
    p.add_argument("--workers", type=str, default="1,2,3,4")
    p.add_argument("--width", type=int, default=960,
                   help="Cellpose inference width (960 = live configuration)")
    p.add_argument("--fields", type=int, default=4)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    analysis = cfg.get("analysis", {})
    cfg.setdefault("gpu_workers", analysis.get("gpu_workers", 1))
    cfg.setdefault("cpu_feature_workers", analysis.get("cpu_feature_workers", 2))
    cfg["cellpose_width"] = args.width

    fields = build_fields(args.fields, args.width)
    print(f"{len(fields)} fields @ {args.width}px inference | "
          f"CPU feature workers: {cfg['cpu_feature_workers']}")
    print("GPU workers | first result | total (4 jobs) | avg latency | VRAM | failed")
    print("-" * 78)

    results = []
    for gw in (int(x) for x in args.workers.split(",")):
        cfg["gpu_workers"] = gw
        controller = LiveFieldController()
        try:
            r = run_config(gw, fields, cfg, controller)
        except torch.cuda.OutOfMemoryError:
            r = {"gpu_workers": gw, "fields": len(fields), "first_result_s": None,
                 "total_s": None, "throughput_jobs_per_s": None,
                 "avg_latency_s": None, "max_latency_s": None,
                 "peak_vram_mb": None, "failed": len(fields),
                 "errors": ["CUDA OOM during setup/inference"]}
        results.append(r)
        print(f"    {gw}       | {r['first_result_s']} s        | "
              f"{r['total_s']} s         | {r['avg_latency_s']} s      | "
              f"{r['peak_vram_mb']} MB | {r['failed']}")
        time.sleep(3.0)  # let the GPU settle between configurations

    out = REPO_ROOT / "outputs" / "benchmarks" / "worker_pool.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")
    best = min((r for r in results if r["total_s"]), key=lambda r: r["total_s"],
               default=None)
    if best:
        print(f"RECOMMENDED for the GTX 1080 Ti: {best['gpu_workers']} GPU worker(s) "
              f"(total {best['total_s']} s, {best['peak_vram_mb']} MB VRAM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
