"""Frame sources: live camera, video file, image file - one interface.

Source spec strings used by scripts/live_app.py:
    camera:<index_or_url>   e.g. camera:0  or camera:http://.../mjpeg
    video:<path>            e.g. video:Dataset/video_dataset.mp4
    image:<path>            e.g. image:Dataset/WIN_20260924_15_46_35_Pro.jpg

LiveCameraSource requests width/height/fps and MJPG when supported but always
reports the ACTUAL values the device granted (spec §37). ``read()`` returns
(frame_bgr or None, timestamp_sec); timestamps are a monotonic session clock
for cameras and the file clock for videos, so downstream timing logic is
identical for both.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameSource:
    """Common interface. ``read()`` -> (frame_bgr|None, t_sec)."""

    name = "source"
    last_error = ""

    def open(self) -> bool:
        raise NotImplementedError

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def actual_settings(self) -> dict:
        return {}


class LiveCameraSource(FrameSource):
    """OpenCV capture from a device index or stream URL (spec §4, hardened).

    Connection rules (camera input layer hardening):
      * NATIVE-FIRST: ``width=0``/``height=0`` (the default) opens the camera
        in its default mode WITHOUT forcing any resolution - forcing an
        unsupported mode (e.g. 1920x1080 on an 800x448 sensor) yields a
        "0x0, no frames" device. Request explicit sizes only after
        ``scripts/list_cameras.py --probe-resolutions`` shows them working.
      * BACKEND: ``auto`` tries MSMF then DSHOW and keeps the first backend
        that actually returns a frame; ``msmf``/``dshow`` force one.
      * A camera that OPENS but never streams is rejected with a precise
        diagnostic ("Camera 0 / DSHOW opened but returned no frames") -
        typically another app (Windows Camera) holds the device.
    """

    reconnect = True  # transient failures are worth an automatic re-open

    def __init__(
        self,
        source,
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        backend: str = "auto",
        buffer_size: int = 1,
    ) -> None:
        self.source = int(source) if str(source).isdigit() else str(source)
        self.requested = {"width": width, "height": height, "fps": fps}
        self.backend = backend
        self.buffer_size = buffer_size
        self.cap: Optional[cv2.VideoCapture] = None
        self.name = f"camera:{self.source}"
        self.last_error = ""
        self._backend_name = "auto"

    def open(self) -> bool:
        if self.backend == "dshow":
            backends = [("dshow", cv2.CAP_DSHOW)]
        elif self.backend == "msmf":
            backends = [("msmf", cv2.CAP_MSMF)]
        else:  # auto: first backend that actually streams wins
            backends = [("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW)]

        errors = []
        for be_name, be in backends:
            cap = cv2.VideoCapture(self.source, be)
            if not cap.isOpened():
                cap.release()
                errors.append(f"{be_name.upper()}: device did not open")
                continue
            # native-first: only force properties when explicitly requested
            if self.requested["width"] > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested["width"])
            if self.requested["height"] > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested["height"])
            if self.requested["fps"] > 0:
                cap.set(cv2.CAP_PROP_FPS, self.requested["fps"])
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, self.buffer_size))
            if self._streams(cap):
                self.cap = cap
                self._backend_name = be_name
                return True
            errors.append(
                f"Camera {self.source} / {be_name.upper()} opened but returned no frames"
            )
            cap.release()
            time.sleep(0.4)

        self.last_error = " | ".join(errors) + (
            ". If Windows Camera is open, close it fully (check Task Manager "
            "for WindowsCamera.exe) and retry."
        )
        return False

    def _streams(self, cap: cv2.VideoCapture, timeout_s: float = 4.0) -> bool:
        """True when the device actually delivers a frame within timeout."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            if cap.grab():
                return True
            time.sleep(0.05)
        return False

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (frame, t_session).

        ``t_session`` is ALWAYS ``time.perf_counter()`` - the single monotonic
        session clock used by the controller, staleness and result ages.
        VideoFileSource keeps the file clock internally for replay pacing
        only; mixing the two clocks broke staleness (file seconds vs hours
        since boot).
        """
        if self.cap is None:
            return None, time.perf_counter()
        ok, frame = self.cap.read()
        return (frame if ok else None), time.perf_counter()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def actual_settings(self) -> dict:
        if self.cap is None:
            return {"requested": self.requested}
        return {
            "requested": self.requested,
            "actual": {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(float(self.cap.get(cv2.CAP_PROP_FPS)), 2),
                "backend": self._backend_name,
            },
        }


class VideoFileSource(FrameSource):
    """MP4/AVI replay at (approximately) native frame rate - regression mode."""

    def __init__(self, path: str, realtime: bool = True, speed: float = 1.0,
                 loop: bool = False) -> None:
        self.path = str(path)
        self.realtime = bool(realtime)
        self.speed = max(0.1, float(speed))
        self.loop = bool(loop)
        self.cap: Optional[cv2.VideoCapture] = None
        self.name = f"video:{Path(self.path).name}"
        self._fps = 30.0
        self._start = 0.0
        self._idx = 0

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            return False
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self._fps = fps if fps > 0 else 30.0
        self._start = time.perf_counter()
        return True

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        if self.cap is None:
            return None, time.perf_counter()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return None, time.perf_counter()
            t_file = self._idx / self._fps
            self._idx += 1
            if self.realtime:
                target = self._start + t_file / self.speed
                now = time.perf_counter()
                if now < target:
                    time.sleep(target - now)
            # single session clock for the whole pipeline (see FrameSource.read)
            return frame, time.perf_counter()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def actual_settings(self) -> dict:
        return {"path": self.path, "fps_avg": round(self._fps, 3), "mode": "replay"}


class ImageFileSource(FrameSource):
    """Single still image, re-served at ~2 Hz (sanity testing without camera)."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.frame: Optional[np.ndarray] = None
        self.name = f"image:{Path(self.path).name}"

    def open(self) -> bool:
        self.frame = cv2.imread(self.path)
        return self.frame is not None

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        time.sleep(0.5)
        return self.frame, time.perf_counter()

    def close(self) -> None:
        self.frame = None

    def actual_settings(self) -> dict:
        return {"path": self.path, "mode": "still"}


def make_source(spec: str, live_cfg: dict) -> FrameSource:
    """Build a FrameSource from a 'kind:value[?key=val]' spec string.

    Camera spec examples: ``camera:0``, ``camera:0?backend=msmf``,
    ``camera:0?backend=dshow``. CLI/config overrides still win.
    """
    kind, _, rest = spec.partition(":")
    kind = kind.lower()
    value, _, query = rest.partition("?")
    qparams = dict(p.split("=", 1) for p in query.split("&") if "=" in p) if query else {}

    if kind == "camera":
        backend = qparams.get("backend", live_cfg.get("camera_backend", "auto"))
        return LiveCameraSource(
            value or 0,
            width=live_cfg.get("camera_width", 0),
            height=live_cfg.get("camera_height", 0),
            fps=live_cfg.get("camera_fps", 0.0),
            backend=backend,
            buffer_size=live_cfg.get("capture_buffer_size", 1),
        )
    if kind == "video":
        return VideoFileSource(
            value,
            realtime=live_cfg.get("video_realtime", True),
            loop=live_cfg.get("video_loop", False),
        )
    if kind == "image":
        return ImageFileSource(value)
    raise ValueError(f"Unknown source spec: {spec!r} (use camera:|video:|image:)")
