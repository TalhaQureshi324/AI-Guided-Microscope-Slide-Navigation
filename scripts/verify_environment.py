"""Environment verification for the AI-Guided Microscope Slide Navigation project.

Checks repository structure, dataset inventory, Python/PyTorch/CUDA versions,
Cellpose installation and (optionally) that the segmentation model loads.
Run from the repository root:

    python scripts/verify_environment.py            # full check incl. model load
    python scripts/verify_environment.py --no-load  # skip model load (fast)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.image_io import discover_images, load_image  # noqa: E402
from src.utils.runtime import describe_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-load", action="store_true", help="skip Cellpose model load test")
    args = parser.parse_args()

    print("=" * 64)
    print("FYP environment verification")
    print("=" * 64)

    # --- repository structure -------------------------------------------------
    print("\n[Repository]")
    for rel in ("Dataset", "external/cellpose", "src", "scripts", "configs"):
        target = REPO_ROOT / rel
        state = "OK" if target.exists() else "MISSING"
        print(f"  {rel:<22s} {state}")

    # --- dataset inventory ------------------------------------------------------
    print("\n[Dataset]")
    try:
        images = discover_images(REPO_ROOT / "Dataset")
        print(f"  {len(images)} image(s) discovered")
        for i, path in enumerate(images, 1):
            img = load_image(path)
            h, w, c = img.shape
            size_kb = path.stat().st_size / 1024
            print(f"  {i:2d}. {path.name}  {w}x{h}  {c}ch  {size_kb:.0f} KB")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED to inventory dataset: {exc}")
        return 1

    # --- python / torch / cuda --------------------------------------------------
    print("\n[Python & PyTorch]")
    try:
        import torch

        print(f"  python      : {sys.version.split()[0]} ({sys.executable})")
        print(f"  torch       : {torch.__version__}")
        print(f"  cuda avail. : {torch.cuda.is_available()}")
        print(f"  device      : {describe_device()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED to import torch: {exc}")
        return 1

    # --- cellpose ---------------------------------------------------------------
    print("\n[Cellpose]")
    try:
        from importlib.metadata import version

        from cellpose import models

        print(f"  version     : {version('cellpose')}")
        print(f"  module path : {Path(models.__file__).parent}")
        print(f"  models      : {models.MODEL_NAMES}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED to import cellpose: {exc}")
        return 1

    # --- model weights ------------------------------------------------------------
    print("\n[Model weights]")
    model_dir = Path.home() / ".cellpose" / "models"
    for name in ("cpsam_v2", "cpsam"):
        p = model_dir / name
        if p.exists():
            print(f"  {name:<10s} present ({p.stat().st_size / 1e6:.0f} MB)")
        else:
            print(f"  {name:<10s} not downloaded yet (auto-download on first use)")

    # --- optional model load --------------------------------------------------------
    if not args.no_load:
        print("\n[Model load test]")
        try:
            from src.segmentation.cellpose_segmenter import CellposeSegmenter

            seg = CellposeSegmenter(model_name="cpsam_v2")
            print(f"  OK - cpsam_v2 loaded in {seg.model_load_s:.1f} s on {seg.device_desc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED to load cpsam_v2: {exc}")
            return 1

    print("\nEnvironment verification finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
