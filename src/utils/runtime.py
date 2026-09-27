"""Runtime helpers: timing, environment metadata, logging setup."""

from __future__ import annotations

import datetime as _dt
import logging
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional


class Timer:
    """Simple wall-clock stopwatch used for per-stage performance measurement."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._laps: Dict[str, float] = {}

    def lap(self, name: str) -> float:
        """Record elapsed seconds since start under *name* and return milliseconds."""
        ms = (time.perf_counter() - self._t0) * 1000.0
        self._laps[name] = ms
        return ms

    @property
    def laps_ms(self) -> Dict[str, float]:
        return dict(self._laps)


@contextmanager
def timed(name: str, sink: Optional[Dict[str, float]] = None, level: int = logging.DEBUG):
    """Context manager that logs (and optionally stores) the duration of a block."""
    log = logging.getLogger(__name__)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - t0) * 1000.0
        if sink is not None:
            sink[name] = ms
        log.log(level, "%s took %.1f ms", name, ms)


def git_commit(repo_dir: Path) -> Optional[str]:
    """Return the short git commit hash, or None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - git is entirely optional metadata
        pass
    return None


def collect_env_metadata(repo_dir: Path, extra: Optional[Dict] = None) -> Dict:
    """Collect the reproducibility record required for every experiment run."""
    meta: Dict = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(repo_dir),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        meta.update(
            torch=torch.__version__,
            cuda_available=cuda_available,
            cuda_version=torch.version.cuda if cuda_available else None,
            cudnn_version=torch.backends.cudnn.version() if cuda_available else None,
            device=("CUDA - " + torch.cuda.get_device_name(0)) if cuda_available else "CPU",
        )
    except Exception as exc:  # noqa: BLE001 - report rather than crash
        meta.update(torch=None, cuda_available=False, device="CPU", torch_import_error=str(exc))

    try:
        from importlib.metadata import version

        meta["cellpose_version"] = version("cellpose")
    except Exception:  # noqa: BLE001
        meta["cellpose_version"] = None

    if extra:
        meta.update(extra)
    return meta


def describe_device() -> str:
    """One-line human-readable device description, e.g. 'CUDA - NVIDIA ...' or 'CPU'."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"CUDA - {torch.cuda.get_device_name(0)}"
        return "CPU (CUDA not available)"
    except Exception:  # noqa: BLE001
        return "CPU (torch unavailable)"


def setup_logging(log_file: Optional[Path] = None, verbose: bool = False) -> None:
    """Configure root logging: concise console output, full detail to file."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(file_handler)
