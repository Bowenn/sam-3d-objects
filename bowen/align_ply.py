import torch
import numpy as np
from plyfile import PlyData, PlyElement

def save_aligned_ply(output, filename="aligned_model.ply"):
    """
    Extracts Gaussian points from the inference output, applies the 
    predicted camera transform and Meta's alignment fix, then saves to PLY.
    """
    # 1. Get raw points from the Gaussian object
    # The output['gaussian'] is typically a list of GaussianModel objects
    gaussians = output["gaussian"][0]
    xyz = gaussians.get_xyz # [N, 3]
    
    # 2. Get the predicted Pose (Extrinsics)
    rotation = output["rotation"]    # [1, 4] quaternion or [1, 3, 3] matrix
    translation = output["translation"] # [1, 3]
    scale = output["scale"]          # [1, 3] or scalar
    
    # 3. Apply Camera Transform (Object Space -> Camera Space)
    # If rotation is a quaternion, convert it to a matrix first
    if rotation.shape[-1] == 4:
        from pytorch3d.transforms import quaternion_to_matrix
        R = quaternion_to_matrix(rotation)
    else:
        R = rotation
        
    # Transform: x' = (x * scale) @ R.T + translation
    xyz_aligned = (xyz * scale) @ R.transpose(-1, -2) + translation

    # 4. Apply Meta's "Fix Alignment" Matrix 
    # This aligns the internal 3D space with standard image-plane coordinates
    fix_matrix = torch.tensor([
        [-1,  0,  0],
        [ 0,  0,  1],
        [ 0,  1,  0]
    ], device=xyz.device, dtype=xyz.dtype)
    
    xyz_final = xyz_aligned @ fix_matrix.T
    
    # 5. Extract colors (features_dc)
    # Convert Spherical Harmonics (SH) degree 0 to RGB
    SH_C0 = 0.28209479177387814
    colors = torch.clamp(0.5 + SH_C0 * gaussians.get_features[:, 0, :], 0.0, 1.0)
    colors = (colors.cpu().detach().numpy() * 255).astype(np.uint8)
    
    # 6. Prepare and Save PLY
    points = xyz_final.cpu().detach().numpy()
    vertex_data = [
        (points[i, 0], points[i, 1], points[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
        for i in range(len(points))
    ]
    
    el = PlyElement.describe(
        np.array(vertex_data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 
        'vertex'
    )
    PlyData([el]).write(filename)
    print(f"Successfully saved aligned model to {filename}")

# Usage:
# save_aligned_ply(output, "my_object_aligned.ply")