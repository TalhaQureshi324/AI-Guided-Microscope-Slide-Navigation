# AI-Guided Microscope Slide Navigation

AI-assisted navigation and segmentation of microscope slide images, built on the
official [Cellpose](https://github.com/MouseLand/cellpose) segmentation engine
(integrated as a git submodule at `external/cellpose`, cellpose v4.2.1.1 / Cellpose-SAM).

## Project objective

Develop an intelligent microscope navigation system that automatically locates the
**monolayer region of a peripheral blood smear**. At every microscope field the
system will: segment RBCs with Cellpose → count individual (touching/overlapping)
RBC instances → measure spatial characteristics → classify the field as
`TOO_THICK` / `MONOLAYER` / `TOO_THIN` → navigate the stage accordingly.

The project is built in validated layers (currently in **Phase 1: prove reliable
Cellpose-based RBC instance counting on the current dataset** — no monolayer
thresholds are defined yet, and no hardware navigation is implemented yet).

## Quickstart (fresh machine)

```bash
# 1. Clone WITH the Cellpose submodule (external/cellpose must not be empty)
git clone --recurse-submodules https://github.com/TalhaQureshi324/AI-Guided-Microscope-Slide-Navigation.git
cd AI-Guided-Microscope-Slide-Navigation

# 2. Environment (see "Environment setup" below for details)
conda create --name cellpose_env python=3.12 -y
conda activate cellpose_env
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # GPU PCs
# (CPU-only laptops: plain `python -m pip install torch torchvision` is fine)
python -m pip install -e "external/cellpose[gui]"
python -m pip install scikit-image pandas pyyaml opencv-python natsort

# 3. Verify everything (auto-downloads cpsam_v2 weights ~1.15 GB on first run)
python scripts/verify_environment.py

# 4. Run the Phase 1 baseline over the dataset
python scripts/segment_dataset.py
```

> **Pulling updates on an existing clone:** git submodules are not fetched by
> `git pull` automatically. If `external/cellpose` is empty after pulling, run:
> `git submodule update --init`

Device selection is automatic on every run — the startup log prints
`CUDA - <GPU name>` on GPU machines and `CPU` otherwise, and the numerical
precision adapts likewise (`use_bfloat16: auto` → bfloat16 on CUDA, float32 on
CPU). No configuration changes are needed when moving between machines.

## Repository structure

```
.
├── Dataset/            # sample microscope slide images (Windows Camera captures)
├── external/cellpose/  # official Cellpose repository (git submodule, unmodified)
├── src/                # application code (Stage A-D separation)
│   ├── segmentation/   #   Stage A: cellpose_segmenter.py (model loaded once)
│   ├── analysis/       #   Stage B: cell_features.py, spatial_features.py
│   ├── postprocessing/ #   merged_cell_splitter.py (multi-signal merge flags;
│   │                   #   watershed fallback experimental + disabled)
│   ├── visualization/  #   overlays.py (diagnostic overlays & debug crops)
│   ├── evaluation/     #   metrics.py (manual-count comparison)
│   └── utils/          #   image_io.py, runtime.py
├── scripts/
│   ├── verify_environment.py  # repo/env/cellpose sanity check
│   ├── segment_dataset.py     # Phase 1 pipeline (main entry point)
│   ├── extract_features.py    # recompute features from saved masks (no inference)
│   └── evaluate_counts.py     # Phase 2: compare counts to manual ground truth
├── configs/default.yaml       # all tunable parameters live here
├── annotations/manual_counts.csv  # manual RBC counts + field labels (ground truth)
├── outputs/            # experiment results (one directory per configuration)
└── README.md
```

## Environment setup (Conda)

Requires an existing Conda installation (Miniconda / Miniforge / Anaconda).

```bash
# 1. Create and activate the dedicated environment
conda create --name cellpose_env python=3.12 -y
conda activate cellpose_env

# 2. Clone the repository with its submodule
git clone --recurse-submodules https://github.com/TalhaQureshi324/AI-Guided-Microscope-Slide-Navigation.git
cd AI-Guided-Microscope-Slide-Navigation

# 3. CUDA-enabled PyTorch (NVIDIA GPU). CPU-only machines can skip this line.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install Cellpose from the integrated submodule (editable) with GUI extras
python -m pip install -e "external/cellpose[gui]"

# 5. Sanity check
python -c "import cellpose; print('Cellpose version:', cellpose.__version__)"
```

> Cellpose auto-downloads its pretrained model weights to `~/.cellpose/models/`
> on first run; those files are intentionally not tracked in git.

Additional analysis dependencies used by our own code:

```bash
python -m pip install scikit-image pandas pyyaml opencv-python natsort
```

## Verify GPU acceleration

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Reference hardware used for this setup: NVIDIA GeForce GTX 1080 Ti (11 GB).
CPU-only laptops work too — the pipeline automatically falls back to CPU
(inference is simply slower; see `outputs/<experiment>/summary.json` for
measured runtimes). Device selection is fully automatic (`gpu: auto` in
`configs/default.yaml`) and the chosen device is printed at startup, so the
same code runs unmodified on a GPU desktop at university and a CPU laptop at
home. Numerical precision is likewise portable (`use_bfloat16: auto`):
bfloat16 on CUDA GPUs, float32 on CPUs, where bfloat16 emulation measured
>4x slower (i5-8250U: >55 min/image in bfloat16 vs ~13.6 min in float32).

## Verify the environment

```bash
python scripts/verify_environment.py            # includes a cpsam_v2 model load test
python scripts/verify_environment.py --no-load  # fast check without model load
```

## Run the segmentation pipeline

```bash
python scripts/segment_dataset.py                                # default config (cpsam_v2 baseline)
python scripts/segment_dataset.py --limit 1 --verbose            # smoke test on one image
python scripts/segment_dataset.py --experiment-name <name>       # new experiment directory
```

The Cellpose model is created **once** and reused for all images. Device
selection is automatic (`CUDA - <gpu>` or `CPU`) and printed at start. All
parameters come from `configs/default.yaml`; each experiment stores the exact
config it used.

## How outputs are organized

```
outputs/<experiment>/
├── overlays/           # annotated image per input: coloured boundaries,
│                       # centroids, IDs, flagged objects, summary banner
├── masks/              # original Cellpose integer label masks (16-bit PNG)
├── debug_merged/       # zoomed crops of merge-suspect masks with evidence
├── cell_features.csv   # one row per detected object (morphology + flags + NN)
├── image_features.csv  # one row per image (counts, spatial features, timings)
├── summary.json        # aggregate counts + performance statistics
├── run_metadata.json   # timestamp, git commit, versions, device, parameters
├── config_used.yaml    # snapshot of the configuration for this run
└── run.log             # full debug log
```

Original images in `Dataset/` are never modified.

## Entering manual counts (Phase 2 ground truth)

1. Open `outputs/<experiment>/overlays/*.jpg` and the original images.
2. Count RBCs manually per image (same convention as the pipeline: RBC
   candidates **including** border-touching cells).
3. Enter the numbers in `annotations/manual_counts.csv`
   (`manual_rbc_count` column); optionally set `manual_label` to
   `TOO_THICK`, `MONOLAYER`, `TOO_THIN` or `UNCERTAIN`.
4. Run the comparison:

```bash
python scripts/evaluate_counts.py --experiment cpsam_v2_baseline
```

This produces `outputs/<experiment>/count_comparison.csv` with per-image
absolute/percentage error and dataset-level MAE / MPE.

## Current project phase

**Phase 1 — Cellpose segmentation foundation (in progress).**
`cpsam_v2` baseline over the current 10-image dataset with instance counting,
per-cell measurements, conservative flagging (small artifacts / possible WBC /
possible merges / border cells) and full spatial-feature extraction
(coverage, density, NN spacing, contact ratio, adjacency graph, uniformity).
Watershed splitting is **not** applied (merged masks are only flagged for
inspection); monolayer thresholds are deliberately **not** defined yet.

Next phases: manual count validation → controlled parameter calibration →
prototype monolayer calibration → more data → optional Cellpose fine-tuning →
real-time operation → microscope navigation.

## Scientific workflow rules

* Every algorithm/parameter change must state the problem, the change, and the
  measured effect on the whole dataset (evidence-driven development).
* `external/cellpose/` is third-party source: import it, never modify it.
* No universal clinical thresholds are claimed from the current small dataset.
