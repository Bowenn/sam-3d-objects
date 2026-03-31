# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM 3D Objects is a Meta Research foundation model that reconstructs full 3D shape geometry, texture, and layout from a single image. It outputs multiple 3D representations: Gaussian splats, meshes, radiance fields, and octrees.

## Environment Setup

Requires Linux 64-bit with NVIDIA GPU (32GB+ VRAM, tested on A100/H100/H200), CUDA 12.1, Python 3.12.

```bash
mamba env create -f environments/default.yml
mamba activate sam3d-objects
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'          # separate step due to pytorch3d dependency ordering
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
./patching/hydra                  # patches Hydra 1.3.2
```

Model checkpoints require HuggingFace access approval at `facebook/sam-3d-objects`, then:
```bash
hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects
mv checkpoints/hf-download/checkpoints checkpoints/hf
rm -rf checkpoints/hf-download
```

## Commands

```bash
# Run tests
pytest

# Format code
black .
usort format .

# Lint
flake8

# Quick inference demo
python demo.py
```

No CI/CD pipelines exist in the repo. No test files currently exist.

## Architecture

### Inference Pipeline (entry point)

`notebook/inference.py` defines the public `Inference` class. `demo.py` is the simplest usage example. The `Inference` class wraps `InferencePipelinePointMap` and enforces Hydra config safety via whitelist/blacklist filters on `_target_` fields.

The pipeline path: `notebook/inference.py` -> `sam3d_objects/pipeline/inference_pipeline_pointmap.py` -> `sam3d_objects/pipeline/inference_pipeline.py`

### Generation Pipeline Stages

1. Extract DINO vision embeddings + pointmaps from input image
2. Estimate monocular depth via MoGe
3. Generate sparse 3D structure via flow matching (`sparse_structure_flow`)
4. Encode/decode structured latent representations (`structured_latent_vae`, `structured_latent_flow`)
5. Decode to output representations: Gaussian splats, meshes (FlexiCubes), radiance fields, octrees
6. Infer object pose (rotation, translation, scale) and optionally post-optimize layout

### Key Modules

- **`sam3d_objects/model/backbone/tdfy_dit/`** - Core Diffusion Transformer architecture
  - `models/` - Sparse VAE, flow matching models, latent models
  - `modules/sparse/` - Custom sparse convolution/attention (spconv-based, supports shift-window, serialized, and full attention modes)
  - `representations/` - Output format implementations (Gaussian, Mesh, Octree, RadianceField)
  - `renderers/` - Gaussian splatting and octree renderers (supports pytorch3d and nvdiffrast backends)
- **`sam3d_objects/pipeline/`** - Inference orchestration, depth estimation, layout post-optimization
- **`sam3d_objects/config/`** - Hydra config utilities with safety checks
- **`sam3d_objects/data/dataset/tdfy/`** - Image/mask preprocessing and 3D transforms

### Configuration

Uses Hydra (patched 1.3.2) + OmegaConf. Pipeline config loaded from `checkpoints/hf/pipeline.yaml`. The `_target_` pattern is used throughout for instantiation.

### Environment Variables

- `LIDRA_SKIP_INIT` - Skip heavy initialization in `sam3d_objects/__init__.py` (set automatically by `notebook/inference.py`)
- `ATTN_BACKEND` - Auto-set to `flash_attn` on A100/H100
- `SPARSE_ATTN_BACKEND` - Controls sparse attention backend selection

### Rendering

Default rendering engine is `pytorch3d` (forced in `notebook/inference.py`). `nvdiffrast` is an alternative but disabled in the public API. `gsplat` is used for video rendering.

## Code Style

- Copyright header: `# Copyright (c) Meta Platforms, Inc. and affiliates.`
- Uses Python type hints
