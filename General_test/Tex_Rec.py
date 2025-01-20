from PIL import Image
from lang_sam import LangSAM
import matplotlib.pyplot as plt
import cv2 as cv
import pymeshlab
import numpy as np
import sys
import os
import subprocess as sp
import time
from threading import Thread, Timer
import torch
import open3d as o3d
from timm.models import register_model
from helper import (save_overlayed_image_with_black_background, save_binary_masks,
                    print_bounding_boxes, print_detected_phrases, print_logits,
                    rename_files_in_directory, print_gpu_memory_every_sec,
                    count_assets_in_directory, get_file_extension_from_directory,
                    save_as_png, save_overlayed_image, upscale_image, remove_black_faces)
from bin_helper import save_point_cloud

import hydra
from omegaconf import DictConfig, OmegaConf
# Add the directory containing vggsfm to the system path
sys.path.append(os.path.abspath("/workspace/data/BscArbeit/Reconstruction_from_image/vggsfm/"))

from vggsfm.runners.runner import VGGSfMRunner
from vggsfm.datasets.demo_loader import DemoLoader
from vggsfm.utils.utils import seed_all_random_engines

import os

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

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

    save_point_cloud(output_points3D_path, output_points3D_path)
    return True

# Separate method to load the cfg like demo_fn does without calling it
@hydra.main(config_path="/workspace/data/BscArbeit/Reconstruction_from_image/vggsfm/cfgs/", config_name="demo")
def load_cfg(config: DictConfig):
    global cfg
    cfg = config

load_cfg()  # Load the cfg and print SCENE_DIR
# Initialize assets path and output paths
assets_path = cfg.WORK_DIR
inputs_path = os.path.join(assets_path, "Inputs/")
output_images_path = os.path.join(assets_path, "Outputs/images/")
output_masks_path = os.path.join(assets_path, "Outputs/masks/")
output_points3D_path = os.path.join(assets_path, "Outputs/sparse/points3D.bin")
output_sugar = os.path.join(assets_path, "Outputs/sugar/")
output_sugar_images_path = assets_path + "Outputs/sugar/images/"
output_sugar_masks_path = assets_path + "Outputs/sugar/masks/"


# Automatically get file extension from the directory
assets_data_type = get_file_extension_from_directory(inputs_path)

# Count assets
assets_amount = count_assets_in_directory(inputs_path, assets_data_type)
print('assets_amount:', assets_amount)

# Rename the files in the assets directory if they are not already named in the correct format
rename_files_in_directory(inputs_path, assets_data_type)

# Measure total start time
total_start_time = time.time()

### Start measuring how much time this script takes from here
seg_start_time = time.time()

# Load the LangSAM model and set the text prompt
model = LangSAM()
text_prompt = cfg.TEXT_PROMPT

# Iterate through images
for i in range(assets_amount):
    image_path = inputs_path + f"{str(i).zfill(3)}" + assets_data_type
    print('image_path:', image_path)
    # upscale_image(image_path)
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
        if any(logit < cfg.CONFIDENCE for logit in logits):
            print(f"Skipping image {i} because at least one logit is below the threshold.")
            continue  
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
        save_as_png(image_pil, f"{str(i).zfill(3)}" + assets_data_type, output_images_path)
        save_binary_masks(masks_np, f"{str(i).zfill(3)}" + assets_data_type, output_masks_path, cfg.MASK_PADDING)
        save_overlayed_image(image_pil, masks_np, f"{str(i).zfill(3)}" + assets_data_type, output_sugar_images_path)
        save_binary_masks(masks_np, f"{str(i).zfill(3)}" + assets_data_type, output_sugar_masks_path, 0)

        # Print the bounding boxes, phrases, and logits
        print_bounding_boxes(boxes)
        print_detected_phrases(phrases)
        print_logits(logits)
        print('image:', i)

del model
torch.cuda.empty_cache()  # Clear GPU memory
print("Model deleted and GPU memory cleared.")

# Segmentation time stamp
seg_end_time = time.time()
seg_time = seg_end_time - seg_start_time
print("Segmentation completed.")

### Start measuring with new time stamp from here
sparse_start_time = time.time()

# Load the config separately and print SCENE_DIR before calling demo_fn
with torch.no_grad():
    demo_fn()  # Then call demo_fn

# Sparse 3D Reconstruction time stamp
sparse_end_time = time.time()
sparse_time = sparse_end_time - sparse_start_time
print("Sparse 3D Reconstruction completed.")

# Store the current working directory to return to it later
original_dir = os.getcwd()

# Change to the directory where train_full_pipeline.py is located
os.chdir('/workspace/data/BscArbeit/Mesh_generation/SuGaR/')

### Start measuring with new time stamp from here
mesh_start_time = time.time()

# Run train_full_pipeline.py with the specified arguments
os.system(
    f"python train_full_pipeline.py \
       -s {output_sugar} \
        -r sdf \
        --high_poly True \
        --export_ply True"
)

# Mesh Reconstruction time stamp
mesh_end_time = time.time()
mesh_time = mesh_end_time - mesh_start_time
print("Mesh Reconstruction completed.")
# Return to the original directory
os.chdir(original_dir)

# Create a MeshSet object
ms = pymeshlab.MeshSet()
mesh_path = output_sugar + 'sugarmesh_3Dgs7000_sdfestim02_sdfnorm02_level03_decim1000000.ply'
# Load the mesh (make sure the mesh has color data)
ms.load_new_mesh(mesh_path)
# Apply the 'Invert Faces Orientation' filter
ms.apply_filter('meshing_invert_face_orientation', forceflip=True, onlyselected=False)

# Check if face color data exists, otherwise transfer vertex colors to faces
try:
    # Try accessing the face color matrix
    face_color_matrix = ms.current_mesh().face_color_matrix()
    print("Face color matrix found.")
except pymeshlab.pmeshlab.MissingComponentException:
    print("Face colors missing, transferring vertex colors to faces.")
    ms.apply_filter('compute_color_transfer_vertex_to_face')
    ms.apply_filter('compute_color_from_texture_per_vertex') 

# Check dimensions of the face color matrix
print(np.shape(ms.current_mesh().face_color_matrix()))
# Save the modified mesh to a new file
ms.save_current_mesh(output_sugar + 'temporary.ply')
# Apply the selection filter based on black faces (RGB values close to [0,0,0])
temporary_mesh_path = output_sugar + 'temporary.ply'
remove_black_faces(temporary_mesh_path, temporary_mesh_path)
# Create a MeshSet object
ms = pymeshlab.MeshSet()
# Load the mesh (make sure the mesh has color data) 
ms.load_new_mesh(temporary_mesh_path)
#this works but should be done at the end
ms.apply_filter('meshing_remove_connected_component_by_face_number', mincomponentsize= 200) 
meshname = cfg.WORK_DIR.rstrip('/').split('/')[-1] + '.ply'
ms.save_current_mesh(output_sugar + meshname)
# Print all times
print("-" * 50)
print("Artifcats removed")
print("-" * 50)



# Total time stamp
total_end_time = time.time()
total_time = total_end_time - total_start_time

# Print all times
print("-" * 50)
print(
    f"Segmentation/Sparse 3D Reconstruction/Mesh Reconstruction/Total: "
    f"{seg_time:.2f} seconds ({seg_time / 60:.2f} minutes)/"
    f"{sparse_time:.2f} seconds ({sparse_time / 60:.2f} minutes)/"
    f"{mesh_time:.2f} seconds ({mesh_time / 60:.2f} minutes)/"
    f"{total_time:.2f} seconds ({total_time / 60:.2f} minutes)"
)
print("-" * 50)


