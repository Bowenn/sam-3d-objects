import torch

def save_inference_metadata(output, folder_path="output_data"):
    import os
    os.makedirs(folder_path, exist_ok=True)
    
    # We strip the heavy Gaussian object and save just the alignment math
    metadata = {
        "rotation": output["rotation"],
        "translation": output["translation"],
        "scale": output["scale"],
        "seed": 42 # Optional: save the seed used
    }
    
    torch.save(metadata, os.path.join(folder_path, "metadata.pt"))
    print(f"Metadata saved to {folder_path}/metadata.pt")

    # Save the raw Gaussian model state
    # torch.save(output["gaussian"][0].state_dict(), "output_data/gaussian_model.pth")

# Usage:
# save_inference_metadata(output)


def load_inference_metadata(folder_path="output_data"):
    metadata = torch.load(os.path.join(folder_path, "metadata.pt"))
    print(f"Metadata loaded from {folder_path}/metadata.pt")

    # 2. Re-create a Gaussian object (you'll need your pipeline config)
    # gaussians = GaussianModel(config) 
    # gaussians.load_state_dict(torch.load("output_data/gaussian_model.pth"))

    # 3. Now you can use the alignment script from before
    # xyz_final = (gaussians.get_xyz * meta["scale"]) @ meta["rotation"].T + meta["translation"]
    return metadata