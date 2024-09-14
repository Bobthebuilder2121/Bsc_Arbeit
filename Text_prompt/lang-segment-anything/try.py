from PIL import Image
from lang_sam import LangSAM
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
import subprocess as sp
from threading import Thread , Timer
import sched, time
import torch

def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values


def print_gpu_memory_every_sec():
    """
        This function calls itself every 5 secs and print the gpu_memory.
    """
    Timer(.1, print_gpu_memory_every_sec).start()
    print(get_gpu_memory())

def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)

# Display the original image and masks side by side
def display_image_with_masks(image, masks):
    num_masks = len(masks)

    fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, mask_np in enumerate(masks):
        axes[i+1].imshow(mask_np, cmap='gray')
        axes[i+1].set_title(f"Mask {i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

# Save the image with the masks overlayed
def save_overlayed_image(image, masks, filename, save_dir):
    cv2_image = np.array(image)
    height, width = np.shape(cv2_image)[0:2]
    image_of_masks = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(len(masks)):
        mask = masks[i].astype(np.uint8) * 255  # Convert boolean mask to uint8 and scale to 255
        masked_image = cv.bitwise_and(cv2_image, cv2_image, mask=mask)
        image_of_masks = cv.add(image_of_masks, masked_image)

    os.makedirs(save_dir, exist_ok=True)
    image_of_masks = Image.fromarray(image_of_masks)
    full_path = os.path.join(save_dir, filename)
    image_of_masks.save(full_path)

# Display the image with the masks overlayed
def display_image_of_masks(image, masks):
    cv2_image = np.array(image)
    height, width = np.shape(cv2_image)[0:2]
    image_of_masks = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(len(masks)):
        mask = masks[i].astype(np.uint8) * 255  # Convert boolean mask to uint8 and scale to 255
        masked_image = cv.bitwise_and(cv2_image, cv2_image, mask=mask)
        image_of_masks = cv.add(image_of_masks, masked_image)#

    plt.imshow(image_of_masks)
    plt.show()

# Display the image with bounding boxes and confidence scores
def display_image_with_boxes(image, boxes, logits):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis('off')

    for box, logit in zip(boxes, logits):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(logit.item(), 2)  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(x_min, y_min, f"Confidence: {confidence_score}", fontsize=8, color='red', verticalalignment='top')

    plt.show()

def print_bounding_boxes(boxes):
    print("Bounding Boxes:")
    for i, box in enumerate(boxes):
        print(f"Box {i+1}: {box}")

def print_detected_phrases(phrases):
    print("\nDetected Phrases:")
    for i, phrase in enumerate(phrases):
        print(f"Phrase {i+1}: {phrase}")

def print_logits(logits):
    print("\nConfidence:")
    for i, logit in enumerate(logits):
        print(f"Logit {i+1}: {logit}")

# Renaming function
def rename_files_in_directory(directory_path, assets_data_type):
    for root_dir, sub_dirs, files in os.walk(directory_path):
        for i, filename in enumerate(files):
            if filename.endswith(assets_data_type):
                old_file_path = os.path.join(root_dir, filename)
                new_file_path = os.path.join(root_dir, f"{str(i).zfill(3)}{assets_data_type}")
                print('old_file_path:', old_file_path)
                print('new_file_path:', new_file_path)
                os.rename(old_file_path, new_file_path)



# Count the number of files in the assets directory
assets_path = "./Orange_can/"
inputs_path = assets_path + "Inputs/"
outputs_path = assets_path + "Outputs/"
assets_amount = 0
assets_data_type = ".jpeg"
for root_dir, cur_dir, files in os.walk(inputs_path):
    assets_amount += len(files)
print('file count:', assets_amount)

# Rename the files in the assets directory if they are not already named in the correct format
#rename_files_in_directory(inputs_path, assets_data_type)

#print_gpu_memory_every_sec()
# Load the LangSAM model and set the text prompt
model = LangSAM()
text_prompt = "orange can"
#print_gpu_memory_every_sec()
for i in range(assets_amount):
    image_path =  inputs_path + f"{str(i).zfill(3)}" + assets_data_type
    print('image_path:', image_path)
    image_pil = Image.open(image_path).convert("RGB")
    
    height, width = np.shape(image_pil)[0:2]
    print('height:', height, 'width:', width)
    min, max = np.min(image_pil), np.max(image_pil)
    print('min:', min, 'max:', max)
    print(f"Processing image {i} with the '{text_prompt}' prompt...")
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

    if len(masks) == 0:
        print(f"No objects of the '{text_prompt}' prompt detected in the image.")
    else:
        # Convert masks to numpy arrays
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
        save_overlayed_image(image_pil, masks_np, f"{str(i).zfill(3)}" + assets_data_type, outputs_path)

        # Print the bounding boxes, phrases, and logits
        print_bounding_boxes(boxes)
        print_detected_phrases(phrases)
        print_logits(logits)
        print('image:', i)
