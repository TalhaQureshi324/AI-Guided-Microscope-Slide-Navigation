"""Enumerate / probe OpenCV camera access on Windows (camera input hardening).

Every camera index is tested INDEPENDENTLY with BOTH cv2.CAP_MSMF and
cv2.CAP_DSHOW. For each combination we report separately:
  * whether VideoCapture.isOpened() succeeds;
  * whether the first REAL read() succeeds.

A valid frame is the only success criterion - a reported FPS of 0.0 is legal
and never counts as failure. Two distinct failures are distinguished:
  * device did not open at all        (driver/backend/privacy problem)
  * device opened but read() failed   (device busy / streaming problem)

Resolution probe (optional): opens the camera at its NATIVE/default mode
first, then - only after native frames flow - negotiates a ladder of practical
resolutions and reports what the driver actually grants:
    python scripts/list_cameras.py --probe-resolutions 0:msmf

Usage:
    python scripts/list_cameras.py                  # all indices x both backends
    python scripts/list_cameras.py --max 8
    python scripts/list_cameras.py --backends msmf  # only MSMF
    python scripts/list_cameras.py --probe-resolutions 0:msmf
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

BACKENDS = {
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
}
RESOLUTION_LADDER = [(640, 480), (800, 448), (1280, 720), (1920, 1080)]


def try_open(index: int, backend_name: str, read_patience_s: float = 3.0) -> dict:
    """Open one index on one backend; report open vs first-frame separately."""
    r = {
        "index": index, "backend": backend_name.upper(),
        "opened": False, "first_frame": False,
        "resolution": None, "fps_reported": None, "shape": None, "error": None,
    }
    cap = cv2.VideoCapture(index, BACKENDS[backend_name])
    if not cap.isOpened():
        cap.release()
        # MSMF sometimes needs a second attempt right after another app
        # released the device - one retry, then give up.
        time.sleep(0.4)
        cap = cv2.VideoCapture(index, BACKENDS[backend_name])
        if not cap.isOpened():
            cap.release()
            r["error"] = "isOpened() = False (device did not open)"
            return r
    r["opened"] = True

    # first REAL frame; patience loop - some devices start streaming late
    deadline = time.perf_counter() + read_patience_s
    ok, frame = False, None
    while time.perf_counter() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.05)

    if ok and frame is not None:
        r["first_frame"] = True
        r["shape"] = list(frame.shape)
        r["resolution"] = f"{frame.shape[1]}x{frame.shape[0]}"
        r["fps_reported"] = round(float(cap.get(cv2.CAP_PROP_FPS)), 2)
    else:
        r["error"] = "opened but read() never returned a frame (device busy or not streaming)"
    cap.release()
    return r


def probe_resolutions(index: int, backend_name: str) -> None:
    """Native-first resolution negotiation report for one working camera."""
    flag = BACKENDS[backend_name]

    def open_and_verify() -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(index, flag)
        if not cap.isOpened():
            cap.release()
            return None
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            time.sleep(0.05)
        cap.release()
        return None

    print(f"\nResolution probe: camera {index} / {backend_name.upper()}")
    print("  1) NATIVE/default mode first (no resolution forced):")
    cap = open_and_verify()
    if cap is None:
        print("     native open/read FAILED - cannot probe resolutions")
        return
    native = cap.read()[1]
    print(f"     native grants {native.shape[1]}x{native.shape[0]} and streams OK")
    cap.release()

    for w, h in RESOLUTION_LADDER:
        if (w, h) == (native.shape[1], native.shape[0]):
            print(f"     {w}x{h}: identical to native - skipped")
            continue
        cap = cv2.VideoCapture(index, flag)
        ok_open = cap.isOpened()
        if ok_open:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            deadline = time.perf_counter() + 3.0
            got, frame = False, None
            while time.perf_counter() < deadline:
                got, frame = cap.read()
                if got and frame is not None:
                    break
                time.sleep(0.05)
            if got and frame is not None:
                gw, gh = frame.shape[1], frame.shape[0]
                verdict = "OK" if (gw, gh) == (w, h) else f"granted {gw}x{gh} instead"
                print(f"     {w}x{h}: {verdict} - streaming OK")
            else:
                print(f"     {w}x{h}: opened but no frames")
        else:
            print(f"     {w}x{h}: failed to open")
        cap.release()
        time.sleep(0.3)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max", type=int, default=6, help="probe indices 0..max-1")
    p.add_argument("--backends", type=str, default="msmf,dshow",
                   help="comma list from: msmf, dshow")
    p.add_argument("--probe-resolutions", type=str, default=None,
                   help="INDEX:BACKEND - native-first resolution ladder probe")
    args = p.parse_args()

    if args.probe_resolutions:
        idx, _, be = args.probe_resolutions.partition(":")
        if be not in BACKENDS:
            print(f"unknown backend {be!r} (use msmf / dshow)")
            return 1
        probe_resolutions(int(idx), be)
        return 0

    order = [b.strip().lower() for b in args.backends.split(",") if b.strip()]
    print(f"Probing camera indices 0..{args.max - 1} x backends {order}")
    print("(success = first real read(); reported FPS 0.0 is NOT a failure)\n")

    working = []
    for idx in range(args.max):
        for be in order:
            r = try_open(idx, be)
            status = ("WORKING" if r["first_frame"]
                      else "OPENED-NO-FRAMES" if r["opened"]
                      else "did not open")
            print(f"  camera {idx} / {r['backend']:<5} : {status}")
            print(f"      isOpened={r['opened']}  firstFrame={r['first_frame']}"
                  f"  resolution={r['resolution']}"
                  f"  fpsReported={r['fps_reported']}"
                  + (f"  ({r['error']})" if r["error"] else ""))
            if r["first_frame"]:
                working.append((idx, be, r["resolution"]))

    print("\nSummary:")
    if working:
        for idx, be, res in working:
            print(f"  camera {idx} + {be.upper()} WORKING at {res}"
                  f"  ->  python scripts\\live_app.py --source camera:{idx} --backend {be}")
        print("\nIf multiple work, prefer MSMF for modern UVC microscopes; "
              "confirm with --probe-resolutions INDEX:BACKEND before forcing "
              "any resolution (native mode first).")
    else:
        print("  no working camera/backend combination right now.")
        print("  Recovery: close Windows Camera/Zoom/Teams, check Task Manager "
              "for WindowsCamera.exe, unplug/replug the camera, then re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
