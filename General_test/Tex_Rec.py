from PIL import Image
from lang_sam import LangSAM
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import sys
import os
import subprocess as sp
from threading import Thread, Timer
import torch
import open3d as o3d
from helper import (save_overlayed_image_with_black_background, save_binary_masks,
                    print_bounding_boxes, print_detected_phrases, print_logits,
                    rename_files_in_directory, print_gpu_memory_every_sec,
                    count_assets_in_directory, get_file_extension_from_directory)
from bin_helper import save_point_cloud

import hydra
from omegaconf import DictConfig, OmegaConf
# Add the directory containing vggsfm to the system path
sys.path.append(os.path.abspath("/workspace/data/BscArbeit/Reconstruction_from_image/vggsfm/"))

from vggsfm.runners.runner import VGGSfMRunner
from vggsfm.datasets.demo_loader import DemoLoader
from vggsfm.utils.utils import seed_all_random_engines

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# Global configuration variable
cfg: DictConfig = None  # Initialize cfg as a global variable

@hydra.main(config_path="/workspace/data/BscArbeit/Reconstruction_from_image/vggsfm/cfgs/", config_name="demo")
def demo_fn(config: DictConfig):
    global cfg
    cfg = config  # Assign the passed config to the global variable

    # Print configuration
    print("Model Config:", OmegaConf.to_yaml(cfg))
    
    # Configure CUDA settings
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set seed for reproducibility
    seed_all_random_engines(cfg.seed)

    # Initialize VGGSfM Runner
    vggsfm_runner = VGGSfMRunner(cfg)

    # Load Data
    test_dataset = DemoLoader(
        SCENE_DIR=cfg.SCENE_DIR,
        img_size=cfg.img_size,
        normalize_cameras=False,
        load_gt=cfg.load_gt,
    )

    sequence_list = test_dataset.sequence_list
    seq_name = sequence_list[0]  # Run on one Scene

    # Load the data for the selected sequence
    batch, image_paths = test_dataset.get_data(
        sequence_name=seq_name, return_path=True
    )

    output_dir = batch["scene_dir"]  # which is also cfg.SCENE_DIR for DemoLoader

    images = batch["image"]
    masks = batch["masks"] if batch["masks"] is not None else None
    crop_params = batch["crop_params"] if batch["crop_params"] is not None else None

    # Cache the original images for visualization, so that we don't need to re-load many times
    original_images = batch["original_images"]

    # Run VGGSfM
    predictions = vggsfm_runner.run(
        images,
        masks=masks,
        original_images=original_images,
        image_paths=image_paths,
        crop_params=crop_params,
        seq_name=seq_name,
        output_dir=output_dir,
    )

    print("Finished Successfully")
    del vggsfm_runner
    torch.cuda.empty_cache()  # Clear GPU memory   
    print("Model deleted and GPU memory cleared.")

    save_point_cloud(output_points3D_path)
    return True

# Initialize assets path and output paths
assets_path = "/workspace/data/data_reconstruction/less_cat/"
inputs_path = os.path.join(assets_path, "Inputs/")
output_images_path = os.path.join(assets_path, "Outputs/images/")
output_masks_path = os.path.join(assets_path, "Outputs/masks/")
output_points3D_path = os.path.join(assets_path, "Outputs/sparse/points3D.bin")

# Automatically get file extension from the directory
assets_data_type = get_file_extension_from_directory(inputs_path)

# Count assets
assets_amount = count_assets_in_directory(inputs_path, assets_data_type)
print('assets_amount:', assets_amount)

# Rename the files in the assets directory if they are not already named in the correct format
rename_files_in_directory(inputs_path, assets_data_type)

# Load the LangSAM model and set the text prompt
model = LangSAM()
text_prompt = "cat"

# Iterate through images
for i in range(assets_amount):
    image_path = inputs_path + f"{str(i).zfill(3)}" + assets_data_type
    print('image_path:', image_path)
    image_pil = Image.open(image_path).convert("RGB")
    
    height, width = np.shape(image_pil)[0:2]
    print('height:', height, 'width:', width)
    min_val, max_val = np.min(image_pil), np.max(image_pil)
    print('min:', min_val, 'max:', max_val)
    print(f"Processing image {i} with the '{text_prompt}' prompt...")
    
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    torch.cuda.empty_cache()
    
    if len(masks) == 0:
        print(f"No objects of the '{text_prompt}' prompt detected in the image.")
    else:
        # Convert masks to numpy arrays
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
        save_overlayed_image_with_black_background(image_pil, masks_np, f"{str(i).zfill(3)}" + assets_data_type, output_images_path)
        save_binary_masks(masks_np, f"{str(i).zfill(3)}" + assets_data_type, output_masks_path)

        # Print the bounding boxes, phrases, and logits
        print_bounding_boxes(boxes)
        print_detected_phrases(phrases)
        print_logits(logits)
        print('image:', i)

del model
torch.cuda.empty_cache()  # Clear GPU memory
print("Model deleted and GPU memory cleared.")

# Call the demo function
with torch.no_grad():
    demo_fn()
