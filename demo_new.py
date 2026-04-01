# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Renders the reconstructed 3D object from the same camera perspective as the input image.

The standard demo.py exports a .ply in the object's local coordinate frame, which
has an arbitrary orientation.  This script uses the predicted camera pose (rotation,
translation, scale) and the estimated intrinsics to render the gaussian splat from
the original viewpoint.
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
# The Inference wrapper doesn't expose intrinsics, so we call the pipeline's
# depth estimation directly.  The intrinsics matrix is 3x3 (normalised, with
# cx=cy=0.5) and encodes the camera's focal length.
image_rgba = inference.merge_mask_to_rgba(image, mask)
with torch.no_grad():
    pointmap_dict = inference._pipeline.compute_pointmap(image_rgba)
intrinsics = pointmap_dict["intrinsics"].cuda().float()  # (3, 3)

# ── 4. Run full inference ──────────────────────────────────────────────────────
output = inference(image, mask, seed=42)

# Print what the output contains so you can explore further:
print("Output keys:", list(output.keys()))
# Pose keys of interest:
#   rotation    – (1, 4) quaternion, local-to-camera (l2c)
#   translation – (1, 3) translation, l2c
#   scale       – (1, 3) scale, l2c
# 3D representations:
#   gs / gaussian – Gaussian splat (in local object space)
#   mesh          – extracted mesh
#   glb           – GLB scene (mesh + texture)
# Scene context:
#   pointmap        – (H, W, 3) 3D pointmap from the camera's perspective
#   pointmap_colors – (H, W, 3) corresponding RGB colours

# ── 5. Transform gaussian from local → camera space ───────────────────────────
# The gaussian splat lives in the object's local frame.  The model also predicts
# a local-to-camera (l2c) pose: rotation (quaternion), translation, and scale.
# Applying this places the object where the camera originally saw it.
gs = deepcopy(output["gs"])

rotation    = output["rotation"].cuda().float()     # (1, 4)
translation = output["translation"].cuda().float()  # (1, 3)
scale       = output["scale"].cuda().float()        # (1, 3)

R_l2c = quaternion_to_matrix(rotation)              # (1, 3, 3)
l2c = compose_transform(scale=scale, rotation=R_l2c, translation=translation)

# Positions
points_cam = l2c.transform_points(gs.get_xyz.unsqueeze(0)).squeeze(0)
gs.from_xyz(points_cam)

# Rotations (apply the inverse of l2c rotation to each gaussian's orientation)
gs.from_rotation(
    quaternion_multiply(quaternion_invert(rotation), gs.get_rotation)
)

# Scales – must update mininum_kernel_size BEFORE from_scaling so the
# internal round-trip (from_scaling → get_scaling) stays consistent.
adjusted_scale = gs.get_scaling * scale
gs.mininum_kernel_size *= scale[0, 0].item()
adjusted_scale = torch.maximum(
    adjusted_scale,
    torch.tensor(gs.mininum_kernel_size * 1.1, device=adjusted_scale.device),
)
gs.from_scaling(adjusted_scale)

# Save the posed gaussian (in PyTorch3D camera space)
gs.save_ply("splat_original_view.ply")
print("Saved posed gaussian to splat_original_view.ply")

# ── 6. Set up camera for original-viewpoint rendering ─────────────────────────
# After the l2c transform the gaussian lives in the camera's coordinate system,
# so the camera sits at the origin → identity extrinsics.
extrinsics = torch.eye(4, dtype=torch.float32, device="cuda")

# Near / far from the depth range of the transformed points
z_vals = points_cam[:, 2]
active_mask = (gs.get_opacity.squeeze() > 0.5) & (z_vals > 0)
z_active = z_vals[active_mask]
near = max(0.01, z_active.min().item() * 0.8)
far  = z_active.max().item() * 1.5

print(f"Intrinsics  fx={intrinsics[0,0]:.4f}  fy={intrinsics[1,1]:.4f}")
print(f"Depth range near={near:.4f}  far={far:.4f}")

# ── 7. Render ──────────────────────────────────────────────────────────────────
resolution = 512
result = render_frames(
    gs,
    [extrinsics],
    [intrinsics],
    {
        "resolution": resolution,
        "bg_color": (1, 1, 1),
        "backend": "gsplat",
        "near": near,
        "far": far,
    },
    verbose=False,
)

rendered = result["color"][0]  # (H, W, 3) uint8
PILImage.fromarray(rendered).save("render_original_view.png")
print("Saved to render_original_view.png")
