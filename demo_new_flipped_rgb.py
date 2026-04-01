# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Same as demo_new_flipped.py but saves the PLY with true RGB vertex colors
(0–255) instead of raw SH coefficients, so any standard PLY viewer shows
correct colors.
"""
import sys
import torch
import numpy as np
from copy import deepcopy
from plyfile import PlyData, PlyElement

sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

from pytorch3d.transforms import quaternion_to_matrix, quaternion_multiply, quaternion_invert
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

# SH band-0 coefficient: 1 / (2 * sqrt(pi))
C0 = 0.28209479177387814


def save_ply_rgb(gs, path):
    """Save a Gaussian as a PLY with uint8 red/green/blue vertex colors."""
    xyz = gs.get_xyz.detach().cpu().numpy()

    # Convert SH DC coefficients → linear RGB [0, 1] → uint8
    # _features_dc can be (N, 1, 3) or (N, 3, 1) depending on the decoder
    f_dc = gs._features_dc.detach().reshape(-1, 3).cpu()  # (N, 3)
    rgb = (f_dc * C0 + 0.5).clamp(0, 1)
    rgb = (rgb * 255).to(torch.uint8).numpy()

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["red"] = rgb[:, 0]
    elements["green"] = rgb[:, 1]
    elements["blue"] = rgb[:, 2]

    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


# ── 1. Load model ──────────────────────────────────────────────────────────────
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# ── 2. Load image and mask ─────────────────────────────────────────────────────
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# ── 3. Run inference ───────────────────────────────────────────────────────────
output = inference(image, mask, seed=42)

# ── 4. Transform gaussian local → camera space ────────────────────────────────
gs = deepcopy(output["gs"])

rotation    = output["rotation"].cuda().float()
translation = output["translation"].cuda().float()
scale       = output["scale"].cuda().float()

R_l2c = quaternion_to_matrix(rotation)
l2c = compose_transform(scale=scale, rotation=R_l2c, translation=translation)

points_cam = l2c.transform_points(gs.get_xyz.unsqueeze(0)).squeeze(0)
gs.from_xyz(points_cam)
gs.from_rotation(
    quaternion_multiply(quaternion_invert(rotation), gs.get_rotation)
)
adjusted_scale = gs.get_scaling * scale
gs.mininum_kernel_size *= scale[0, 0].item()
adjusted_scale = torch.maximum(
    adjusted_scale,
    torch.tensor(gs.mininum_kernel_size * 1.1, device=adjusted_scale.device),
)
gs.from_scaling(adjusted_scale)

# ── 5. Flip for PLY viewers (180° around Y) ───────────────────────────────────
flip = torch.tensor([-1.0, 1.0, -1.0], device="cuda")
gs.from_xyz(gs.get_xyz * flip)

flip_quat = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device="cuda")
gs.from_rotation(quaternion_multiply(flip_quat, gs.get_rotation))

# ── 6. Save ────────────────────────────────────────────────────────────────────
# Standard 3DGS format (SH coefficients) – for SuperSplat / splatting viewers
gs.save_ply("splat_original_view_flipped.ply")
print("Saved 3DGS PLY  → splat_original_view_flipped.ply")

# RGB vertex-color format – for MeshLab / Blender / any generic viewer
save_ply_rgb(gs, "splat_original_view_flipped_rgb.ply")
print("Saved RGB PLY   → splat_original_view_flipped_rgb.ply")
