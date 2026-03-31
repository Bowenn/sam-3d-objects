# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Same as demo_new.py but flips the .ply so that PLY viewers (which look along -Z)
show the object from the original camera angle.

PyTorch3D camera space has +Z forward; most viewers default to -Z forward.
A 180° rotation around Y (negate X and Z) fixes this.
"""
import sys
import torch
from copy import deepcopy
from PIL import Image as PILImage

sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

from pytorch3d.transforms import quaternion_to_matrix, quaternion_multiply, quaternion_invert
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
from sam3d_objects.model.backbone.tdfy_dit.utils.render_utils import render_frames

# ── 1. Load model ──────────────────────────────────────────────────────────────
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# ── 2. Load image and mask ─────────────────────────────────────────────────────
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# ── 3. Compute camera intrinsics via the depth model (MoGe) ───────────────────
image_rgba = inference.merge_mask_to_rgba(image, mask)
with torch.no_grad():
    pointmap_dict = inference._pipeline.compute_pointmap(image_rgba)
intrinsics = pointmap_dict["intrinsics"].cuda().float()  # (3, 3)

# ── 4. Run full inference ──────────────────────────────────────────────────────
output = inference(image, mask, seed=42)

# ── 5. Transform gaussian from local → camera space ───────────────────────────
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
gs.from_scaling(adjusted_scale)
gs.mininum_kernel_size *= scale[0, 0].item()

# ── 6. Flip for PLY viewers ───────────────────────────────────────────────────
# 180° rotation around Y axis: negate X and Z so the default -Z view matches
# the original camera's +Z forward direction.
flip = torch.tensor([-1.0, 1.0, -1.0], device="cuda")
gs.from_xyz(gs.get_xyz * flip)

flip_quat = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device="cuda")  # 180° around Y
gs.from_rotation(quaternion_multiply(flip_quat, gs.get_rotation))

gs.save_ply("splat_original_view_flipped.ply")
print("Saved to splat_original_view_flipped.ply")
