# AI-Guided Microscope Slide Navigation

AI-assisted navigation and segmentation of microscope slide images, built on the
official [Cellpose](https://github.com/MouseLand/cellpose) segmentation engine
(integrated as a git submodule at `external/cellpose`, cellpose v4.2.1.1 / Cellpose-SAM).

## Repository structure

```
.
├── Dataset/            # sample microscope slide images (Windows Camera captures)
├── external/cellpose/  # official Cellpose repository (git submodule)
├── .gitignore
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

## Verify GPU acceleration

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Reference hardware used for this setup: NVIDIA GeForce GTX 1080 Ti (11 GB).
