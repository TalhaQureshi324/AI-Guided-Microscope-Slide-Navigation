"""Image discovery and I/O helpers.

The original microscope images in ``Dataset/`` are never modified. Outputs are
always written to a separate experiment directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageOps
from natsort import natsorted

logger = logging.getLogger(__name__)

# Pillow safety: microscope images can occasionally be very large.
Image.MAX_IMAGE_PIXELS = None


def discover_images(dataset_dir: Path, extensions=("jpg", "jpeg", "png", "tif", "tiff", "bmp")) -> List[Path]:
    """Return a deterministic, naturally sorted list of image paths under *dataset_dir*.

    Non-image files and sub-directories are ignored. Sorting guarantees that
    repeated runs process images in the same order (reproducibility).
    """
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    exts = {f".{e.lower().lstrip('.')}" for e in extensions}
    images = [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    images = natsorted(images, key=lambda p: p.name)
    logger.info("Discovered %d image(s) in %s", len(images), dataset_dir)
    return images


def load_image(path: Path) -> np.ndarray:
    """Load an image as an RGB uint8 array (EXIF-rotation applied).

    Returns a 3-channel (H, W, 3) array regardless of the on-disk mode so the
    downstream pipeline sees a consistent input format.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)  # honour camera rotation metadata
        rgb = im.convert("RGB")
    return np.asarray(rgb, dtype=np.uint8)


def save_mask_png(path: Path, labels: np.ndarray) -> None:
    """Save an integer instance-label image as a lossless 16-bit PNG.

    Labels are preserved exactly (background = 0, instances = 1..N, gaps
    allowed), which keeps Cellpose's original mask recoverable for later
    re-analysis without re-running inference.
    """
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), labels.astype(np.uint16))
    if not ok:
        raise IOError(f"Failed to write label mask to {path}")


def save_overlay_jpg(path: Path, image_bgr: np.ndarray, quality: int = 92) -> None:
    """Save a BGR overlay image as JPG (diagnostic output only)."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise IOError(f"Failed to write overlay to {path}")
