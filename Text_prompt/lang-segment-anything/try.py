from PIL import Image
from lang_sam import LangSAM
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os


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

# Count the number of files in the assets directory
assets_path = "./assets2"
assets_amount = 0
for root_dir, cur_dir, files in os.walk(assets_path):
    assets_amount += len(files)
print('file count:', assets_amount)

# Load the LangSAM model and set the text prompt
model = LangSAM()
text_prompt = "phone"

for i in range(assets_amount):
    image_path = f"./assets2/{str(i).zfill(3)}.jpeg"
    image_pil = Image.open(image_path).convert("RGB")
    text_prompt = "phone"
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    if len(masks) == 0:
        print(f"No objects of the '{text_prompt}' prompt detected in the image.")
    else:
        # Convert masks to numpy arrays
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
        save_overlayed_image(image_pil, masks_np, f"{str(i).zfill(3)}.jpeg", "./assets2/overlayed_output")

        # Print the bounding boxes, phrases, and logits
        print_bounding_boxes(boxes)
        print_detected_phrases(phrases)
        print_logits(logits)
        print('image:', i)
