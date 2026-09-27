"""Reusable Cellpose segmentation wrapper (Stage A of the pipeline).

Design rules enforced here (see project prompt):
  * The model is created **once** and kept resident; ``segment()`` may be called
    per frame / per image without reloading weights (real-time requirement).
  * Device selection is automatic: CUDA when available, clean CPU fallback
    otherwise - the application must never crash because CUDA is missing.
  * ``use_bfloat16`` is configurable so reduced-precision problems on specific
    GPUs can be tested with float32 without code changes.
  * The integer label image returned by Cellpose is passed through unchanged
    (background = 0, instances = 1..N, non-consecutive labels allowed).

Modern Cellpose 4 / Cellpose-SAM API only: ``models.CellposeModel(...)``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Outcome of segmenting a single image."""

    labels: np.ndarray                 # int32 label image (Cellpose original)
    n_instances: int                   # number of unique non-zero labels
    timings_ms: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


class CellposeSegmenter:
    """Loads a Cellpose model once and segments images repeatedly.

    Parameters
    ----------
    model_name:
        Cellpose model identifier or checkpoint path. Default ``cpsam_v2``.
    gpu:
        ``"auto"`` (default) uses CUDA when available; ``True``/``False``
        force a device. A forced GPU request gracefully falls back to CPU.
    use_bfloat16:
        ``"auto"`` (recommended) uses bfloat16 on CUDA GPUs and float32 on
        CPU - bfloat16 has no native CPU support on most processors and its
        software emulation measured >4x slower than float32 on the project
        laptop (i5-8250U). ``True``/``False`` force a precision explicitly
        (e.g. to test float32 on a GPU that misbehaves with reduced precision).
    cellprob_threshold, flow_threshold, min_size, diameter, normalize, augment:
        Standard Cellpose ``eval`` parameters, kept configurable for
        reproducible calibration experiments. Defaults are Cellpose defaults.
    """

    def __init__(
        self,
        model_name: str = "cpsam_v2",
        gpu: Any = "auto",
        use_bfloat16: Any = "auto",
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        min_size: int = 15,
        diameter: Optional[float] = None,
        normalize: Any = True,
        augment: bool = False,
    ) -> None:
        import torch
        from cellpose import models

        # --- device selection -------------------------------------------------
        cuda_available = bool(torch.cuda.is_available())
        if gpu == "auto":
            use_gpu = cuda_available
        else:
            use_gpu = bool(gpu) and cuda_available
            if bool(gpu) and not cuda_available:
                logger.warning("GPU requested but CUDA is unavailable - falling back to CPU.")

        self.device_desc = (
            f"CUDA - {torch.cuda.get_device_name(0)}" if use_gpu else "CPU (CUDA not available)"
        )
        self.model_name = model_name
        # Portable precision: bfloat16 where the hardware supports it natively
        # (CUDA), float32 everywhere else. Explicit True/False override this.
        if use_bfloat16 == "auto" or use_bfloat16 is None:
            use_bfloat16 = use_gpu
        self.use_bfloat16 = bool(use_bfloat16)
        self.params: Dict[str, Any] = {
            "cellprob_threshold": cellprob_threshold,
            "flow_threshold": flow_threshold,
            "min_size": min_size,
            "diameter": diameter,
            "normalize": normalize,
            "augment": augment,
        }

        # --- one-time model load ---------------------------------------------
        load_kwargs: Dict[str, Any] = {
            "gpu": use_gpu,
            "pretrained_model": model_name,
            "use_bfloat16": self.use_bfloat16,
        }

        t0 = time.perf_counter()
        logger.info("Loading Cellpose model '%s' (device: %s)...", model_name, self.device_desc)
        self.model = models.CellposeModel(**load_kwargs)
        self.model_load_s = time.perf_counter() - t0
        logger.info("Model loaded in %.1f s", self.model_load_s)

    # ---------------------------------------------------------------------
    def segment(self, image_rgb: np.ndarray, **overrides) -> SegmentationResult:
        """Run instance segmentation on one RGB image.

        ``overrides`` may temporarily override any eval parameter (used by the
        calibration experiments); the segmenter itself is reused.
        """
        if image_rgb.ndim not in (2, 3):
            raise ValueError(f"Expected a 2D grayscale or 3D RGB image, got shape {image_rgb.shape}.")

        params = dict(self.params)
        params.update(overrides)

        eval_kwargs: Dict[str, Any] = {
            "normalize": params["normalize"],
            "cellprob_threshold": params["cellprob_threshold"],
            "flow_threshold": params["flow_threshold"],
            "min_size": params["min_size"],
            "augment": params["augment"],
        }
        if params.get("diameter") is not None:
            eval_kwargs["diameter"] = params["diameter"]
        if image_rgb.ndim == 3:
            # Cellpose 4 expects channels-last; an RGB image needs no conversion.
            eval_kwargs["channel_axis"] = -1

        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        masks, _flows, _styles = self.model.eval(image_rgb, **eval_kwargs)
        timings["inference_ms"] = (time.perf_counter() - t0) * 1000.0

        labels = np.asarray(masks, dtype=np.int32)
        n_instances = int(len(np.unique(labels))) - (1 if (labels == 0).any() else 0)

        return SegmentationResult(
            labels=labels,
            n_instances=n_instances,
            timings_ms=timings,
            params=params,
        )
