"""FYP live-perception application (spec sections 4-46).

Usage:
    python scripts/live_app.py                          # camera 0, GUI
    python scripts/live_app.py --source camera:1        # another camera index
    python scripts/live_app.py --source video:Dataset/video_dataset.mp4
    python scripts/live_app.py --view-only              # camera check, NO Cellpose
    python scripts/live_app.py --cellpose-mode hybrid   # screen-gated Cellpose
    python scripts/live_app.py --feature-mode detailed  # full merge diagnostics
    python scripts/live_app.py --record                 # record session MP4
    python scripts/live_app.py --selftest               # headless pipeline test

Phase order honoured: camera->live GUI works first (--view-only); manual
analysis via hotkey A; automatic motion-aware analysis afterwards. Nothing in
external/cellpose is touched; the static/video pipelines remain untouched.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.capture.sources import make_source  # noqa: E402
from src.live.controller import LiveFieldController  # noqa: E402
from src.live.jobs import JobManager  # noqa: E402
from src.live.scan_map import ScanMap  # noqa: E402
from src.live.session import LiveSession  # noqa: E402
from src.ui.state import SharedState  # noqa: E402
from src.ui.workers import CaptureWorker, FeatureWorker, GPUWorker, _MaskQueue  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "live.yaml")
    p.add_argument("--source", type=str, default=None,
                   help="camera:<idx|url> | video:<path> | image:<path>")
    p.add_argument("--backend", choices=["auto", "msmf", "dshow"], default=None,
                   help="camera backend override (auto = first that streams)")
    p.add_argument("--width", type=int, default=None,
                   help="camera width; 0 = native/default mode (recommended first)")
    p.add_argument("--height", type=int, default=None,
                   help="camera height; 0 = native/default mode")
    p.add_argument("--view-only", action="store_true",
                   help="live feed + motion + screening only (NO Cellpose)")
    p.add_argument("--cellpose-mode", choices=["benchmark", "hybrid"], default=None)
    p.add_argument("--feature-mode", choices=["live", "detailed"], default=None)
    p.add_argument("--cellpose-width", type=int, default=None,
                   help="reduced-resolution inference (spec §31 benchmark)")
    p.add_argument("--record", action="store_true", help="record session to MP4")
    p.add_argument("--gpu-workers", type=int, default=None,
                   help="concurrent Cellpose workers (benchmark before raising)")
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--max-analyses", type=int, default=None)
    p.add_argument("--selftest", action="store_true",
                   help="headless run: capture+analysis, no GUI, writes session")
    p.add_argument("--selftest-seconds", type=float, default=None,
                   help="selftest time limit (default: until source ends)")
    p.add_argument("--session-name", type=str, default=None)
    return p.parse_args()


def build_pool(state, cfg, jobmgr: JobManager, session=None, scan_map=None):
    """Configurable worker pool: N GPU Cellpose workers + M CPU feature
    workers over one job manager (spec §7/§10 - benchmark before raising)."""
    maskq = _MaskQueue(maxsize=8)
    gpu = [
        GPUWorker(f"gpu{i + 1}", state, cfg, jobmgr, maskq)
        for i in range(int(cfg.get("gpu_workers", 1)))
    ]
    cpu = [
        FeatureWorker(f"cpu{i + 1}", state, cfg, jobmgr, maskq, session=session,
                      scan_map=scan_map)
        for i in range(int(cfg.get("cpu_feature_workers", 2)))
    ]
    return maskq, gpu, cpu


def build_cfg(args) -> dict:
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    # flatten the nested analysis block for the worker pool
    analysis = cfg.get("analysis", {})
    cfg.setdefault("gpu_workers", analysis.get("gpu_workers", 1))
    cfg.setdefault("cpu_feature_workers", analysis.get("cpu_feature_workers", 2))
    if args.source:
        cfg["source"] = args.source
    if args.cellpose_mode:
        cfg["cellpose_mode"] = args.cellpose_mode
    if args.feature_mode:
        cfg["feature_mode"] = args.feature_mode
    if args.cellpose_width:
        cfg["cellpose_width"] = args.cellpose_width
    if args.backend:
        cfg["camera_backend"] = args.backend
    if args.width is not None:
        cfg["camera_width"] = args.width
    if args.height is not None:
        cfg["camera_height"] = args.height
    if args.save_masks:
        cfg["save_masks"] = True
    if args.gpu_workers:
        cfg["gpu_workers"] = args.gpu_workers
    cfg["record_enabled"] = bool(args.record)
    cfg["max_analyses"] = args.max_analyses
    return cfg


def main() -> int:
    # faster GIL rotation: the CPU feature stage is Python-heavy (regionprops)
    # and would otherwise starve the capture thread between C calls
    sys.setswitchinterval(0.002)
    args = parse_args()
    cfg = build_cfg(args)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = args.session_name or f"live_session_{ts}"
    session_dir = REPO_ROOT / cfg.get("outputs_root", "outputs") / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    if args.selftest:
        return _run_selftest(state=None, controller=None, session_dir=session_dir,
                             args=args, cfg=cfg, source=None)

    state = SharedState()
    controller = LiveFieldController(
        moving_disp_thresh_px=cfg["moving_disp_thresh_px"],
        stationary_frames=cfg["stationary_frames"],
        settle_frames=cfg["settle_frames"],
        motion_threshold_px=cfg["motion_threshold_px"],
        min_interval_sec=cfg["min_interval_sec"],
        heartbeat_sec=cfg["heartbeat_sec"],
        blur_guard=cfg["blur_guard"],
    )
    source = make_source(cfg["source"], cfg)

    session = None
    scan_map = None
    if args.view_only:
        # capture worker only (spec §5: prove the feed before Cellpose)
        capture = CaptureWorker(source, state, controller, cfg, session=None)
        capture.start()
    else:
        jobmgr = JobManager(max_manual_queue=cfg.get("max_manual_queue", 20))
        state.jobmgr = jobmgr
        scan_map = ScanMap()
        session = LiveSession(session_dir, cfg, cfg["source"], settings={})
        state.session_dir = str(session_dir)
        state.source_label = source.name
        capture = CaptureWorker(source, state, controller, cfg, jobmgr, session=session)
        _maskq, gpu_workers, cpu_workers = build_pool(state, cfg, jobmgr, session,
                                                      scan_map=scan_map)
        if cfg.get("record_enabled"):
            state.recording = True
        capture.start()
        for w in gpu_workers + cpu_workers:
            w.start()

    # ---- GUI mode -----------------------------------------------------
    from PyQt6.QtWidgets import QApplication

    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)

    # PyQt6 only PRINTS unhandled exceptions and keeps running - a crashed
    # window would leave a headless process holding the camera (observed).
    # Any unhandled exception now quits the app cleanly, releasing the camera.
    def _quit_on_crash(exc_type, exc, tb):
        sys.__excepthook__(exc_type, exc, tb)
        state.quit_event.set()
        app.quit()

    sys.excepthook = _quit_on_crash

    win = MainWindow(state, controller, cfg, session=session,
                     view_only=args.view_only, jobmgr=jobmgr, scan_map=scan_map)
    if args.view_only:
        win.statusBar().showMessage("VIEW-ONLY mode: no Cellpose (camera check)")
    win.show()
    rc = app.exec()
    if scan_map is not None and session is not None:
        session.save_scan_map(scan_map)
    _shutdown(state, session_dir, session)
    return rc


def _run_selftest(state, controller, session_dir, args, cfg, source):
    """Headless validation: same workers, no Qt (spec §46 evidence)."""
    print(f"[selftest] session: {session_dir}")
    print(f"[selftest] source: {cfg['source']} | cellpose_mode={cfg.get('cellpose_mode')} "
          f"| feature_mode={cfg.get('feature_mode')} | width={cfg.get('cellpose_width')}")
    state = SharedState()
    controller = LiveFieldController(
        moving_disp_thresh_px=cfg["moving_disp_thresh_px"],
        stationary_frames=cfg["stationary_frames"],
        settle_frames=cfg["settle_frames"],
        motion_threshold_px=cfg["motion_threshold_px"],
        min_interval_sec=cfg["min_interval_sec"],
        heartbeat_sec=cfg["heartbeat_sec"],
        blur_guard=cfg["blur_guard"],
    )
    source = make_source(cfg["source"], cfg)
    jobmgr = JobManager(max_manual_queue=cfg.get("max_manual_queue", 20))
    state.jobmgr = jobmgr
    scan_map = ScanMap()
    session = LiveSession(session_dir, cfg, cfg["source"], settings={})
    state.session_dir = str(session_dir)
    state.source_label = source.name
    capture = CaptureWorker(source, state, controller, cfg, jobmgr, session=session)
    _maskq, gpu_workers, cpu_workers = build_pool(state, cfg, jobmgr, session,
                                                  scan_map=scan_map)
    t0 = time.perf_counter()
    capture.start()
    for w in gpu_workers + cpu_workers:
        w.start()

    last_print = 0.0
    while not state.quit_event.is_set():
        time.sleep(0.5)
        elapsed = time.perf_counter() - t0
        if args.selftest_seconds and elapsed > args.selftest_seconds:
            print("[selftest] time limit reached")
            break
        if elapsed - last_print >= 10:
            last_print = elapsed
            r = state.result
            last = (f"{r.raw_class} score={r.score:.2f} f{r.frame_idx} "
                    f"stale={r.stale}") if r else "none yet"
            print(f"[{elapsed:6.1f}s] frames={state.frame_seq + 1} "
                  f"cap={state.capture_fps:.1f}fps motion={state.motion_state} "
                  f"cp={state.cellpose_state} analyses={state.analysis_count} "
                  f"superseded={state.pending_superseded} | last: {last}")
        if state.source_error == "source ended":
            break
        if state.last_error:
            print(f"[selftest] FATAL: {state.last_error}")
            break

    # drain: let the worker finish the in-flight field AND any pending one
    # (a short video can end while the model is still loading - the pending
    # field must still be analysed, spec §43: never lose the newest field)
    print("[selftest] capture done; draining analysis worker ...")
    t_drain = time.perf_counter()
    while time.perf_counter() - t_drain < 180:
        if jobmgr.pending_count() > 0:
            time.sleep(0.5)
            continue
        if state.cellpose_state == "LOADING_MODEL" and jobmgr.stats()["total_jobs"] > 0:
            time.sleep(0.5)
            continue
        break
    state.quit_event.set()
    capture.join(timeout=5)
    for w in gpu_workers + cpu_workers:
        w.join(timeout=10)
    elapsed = time.perf_counter() - t0
    session.save_scan_map(scan_map)
    summary = session.write_summary(state, elapsed, jobmgr=jobmgr)
    summary["scan_map_fields"] = len(scan_map.fields())

    r = state.result
    print("=" * 70)
    print(f"[selftest] DONE {elapsed:.1f}s | frames={summary['frames_captured']} "
          f"| capture avg {summary['capture_fps_avg']} fps")
    print(f"[selftest] jobs={summary['jobs_total']} "
          f"analyzed={summary['fields_analyzed']} "
          f"superseded_auto={summary['jobs_superseded_auto']} "
          f"refused_manual={summary['jobs_refused_manual']} (latest-frame policy)")
    print(f"[selftest] cellpose busy {summary['cellpose_busy_sec']}s "
          f"({summary['cellpose_busy_fraction'] * 100:.0f}% of session) "
          f"| last latency {summary['last_latency_ms']}ms")
    classes = {}
    for h in state.history:
        classes[h["raw_class"]] = classes.get(h["raw_class"], 0) + 1
    print(f"[selftest] classes: {classes} | stale: {summary['stale_results']}")
    if any(h["raw_class"] == "MONOLAYER" and not h["stale"] for h in state.history):
        print("[selftest] MONOLAYER candidate detected during session ✓")
    print(f"[selftest] outputs -> {session_dir}")
    return 0


def _shutdown(state, session_dir, session) -> None:
    state.quit_event.set()
    time.sleep(1.0)
    if session is not None:
        session.write_summary(state, time.perf_counter())
    print(f"session -> {session_dir}")


if __name__ == "__main__":
    sys.exit(main())
