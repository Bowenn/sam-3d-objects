# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask
# from align_ply import save_aligned_ply
from save_output import save_inference_metadata

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# From the example in the README:
output = inference(image, mask, seed=42)

# save_aligned_ply(output, filename="aligned_model.ply")

# The 'output' object typically contains the camera/layout information
# Look for keys like 'pose', 'R' (rotation), or 'T' (translation)
print(output.keys())

# export gaussian splat
save_inference_metadata(output, folder_path="output_data")
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")
