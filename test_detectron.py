import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import os

def setup_predictor():
    """
    Set up a Detectron2 configuration for Mask R-CNN with stricter thresholds
    to reduce duplicate detections.
    """
    cfg = get_cfg()
    # Use Mask R-CNN model
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Increase score threshold so only high-confidence detections remain
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Make NMS more aggressive (lower threshold => merges/suppresses duplicates more)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

    # Load pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    predictor = DefaultPredictor(cfg)
    return predictor

def compute_iou(boxA, boxB):
    """
    Compute the Intersection-over-Union (IoU) of two bounding boxes.
    Each box is [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if (boxAArea + boxBArea - interArea) == 0:
        return 0.0

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def filter_duplicate_detections(boxes, scores, masks, iou_threshold=0.7):
    """
    Filter out duplicate detections with high IoU.
    Keep the detection with the higher confidence score.
    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2]
        scores: (N,) array of confidence scores
        masks: (N, H, W) array of boolean masks
        iou_threshold: float, IoU above which two boxes are considered duplicates
    Returns:
        filtered_boxes, filtered_scores, filtered_masks
    """
    indices = list(range(len(boxes)))
    suppressed = set()

    # Sort indices by descending score, so we keep higher-score boxes first
    indices.sort(key=lambda i: scores[i], reverse=True)
    keep = []

    for i in indices:
        if i in suppressed:
            continue
        keep.append(i)
        for j in indices:
            if j <= i or j in suppressed:
                continue
            iou = compute_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                # Suppress the lower-score detection
                suppressed.add(j)

    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]
    filtered_masks = masks[keep]
    return filtered_boxes, filtered_scores, filtered_masks

def crop_and_pad(image, bbox, mask, target_size=(128, 64)):
    """
    Crop the image using the bbox and mask, then resize and pad to target_size.
    
    Args:
        image (np.array): Original image.
        bbox (list or array): [x1, y1, x2, y2] coordinates.
        mask (np.array): Boolean mask (or 0/1) with the same size as image.
        target_size (tuple): (height, width) for the final image.
    
    Returns:
        final_img (np.array): The cropped, resized, and padded image.
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Crop the region of interest from the image and corresponding mask
    cropped_img = image[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2]

    # Apply mask to remove background (optional: here we keep only person pixels)
    cropped_img[cropped_mask == 0] = 0

    # Resize cropped image to fit within target size while keeping aspect ratio
    target_h, target_w = target_size
    h, w = cropped_img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(cropped_img, (new_w, new_h))

    # Pad the resized image to exactly match target_size
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    final_img = cv2.copyMakeBorder(
        resized_img, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return final_img

def crop_without_resize(image, bbox, mask):
    """
    Crop the image using the given bbox and apply the mask without resizing or padding.
    
    Args:
        image (np.array): The original image.
        bbox (list or array): [x1, y1, x2, y2] coordinates for cropping.
        mask (np.array): A boolean or 0/1 mask of the same size as the image.
        
    Returns:
        np.array: The cropped image with the mask applied.
    """
    # Convert bbox coordinates to integers
    x1, y1, x2, y2 = map(int, bbox)
    
    # Crop the image and mask based on bbox
    cropped_img = image[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2]
    
    # Apply the mask: setting pixels where mask is 0 to black
    cropped_img[cropped_mask == 0] = 0
    
    return cropped_img


def draw_segmentation_overlay(image, bbox, mask, color=(0, 255, 0), alpha=0.5):
    """
    Draw the segmentation mask and bounding box on a copy of the input image.
    
    Args:
        image (np.array): Original image.
        bbox (list or array): [x1, y1, x2, y2] coordinates.
        mask (np.array): Boolean mask (or 0/1) with the same size as image.
        color (tuple): BGR color for the overlay and bounding box.
        alpha (float): Transparency factor for the overlay.
        
    Returns:
        overlay_img (np.array): The image with overlay drawn.
    """
    overlay_img = image.copy()
    x1, y1, x2, y2 = map(int, bbox)
    
    # Create a colored overlay for the mask
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask == 1] = color
    
    # Overlay the mask onto the image with transparency
    cv2.addWeighted(colored_mask, alpha, overlay_img, 1 - alpha, 0, overlay_img)
    
    # Draw the bounding box
    cv2.rectangle(overlay_img, (x1, y1), (x2, y2), color, thickness=2)
    
    return overlay_img

def segment_and_save(image_path, output_folder="segmented_persons", target_size=(128, 64)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image:", image_path)
        return

    predictor = setup_predictor()
    outputs = predictor(image)
    instances = outputs["instances"]

    # Filter for person class (COCO class id 0 corresponds to "person")
    person_indices = (instances.pred_classes == 0).nonzero().flatten()
    person_boxes = instances.pred_boxes.tensor[person_indices].cpu().numpy()
    person_scores = instances.scores[person_indices].cpu().numpy()
    person_masks = instances.pred_masks[person_indices].cpu().numpy()

    print(f"Found {len(person_boxes)} person detections before duplicate filtering.")

    # ---- 1) Apply custom duplicate filtering (if needed) ----
    person_boxes, person_scores, person_masks = filter_duplicate_detections(
        person_boxes, person_scores, person_masks, iou_threshold=0.7
    )

    print(f"Found {len(person_boxes)} person detections after duplicate filtering.")

    for idx, (bbox, mask, score) in enumerate(zip(person_boxes, person_masks, person_scores)):
        # Save the cropped and padded person image
        cropped_padded = crop_without_resize(image, bbox, mask)
        crop_output_path = os.path.join(output_folder, f"person_{idx}.jpg")
        cv2.imwrite(crop_output_path, cropped_padded)
        print(f"Saved cropped and padded image: {crop_output_path} that has confidence score of {score}")
        
        # Create an overlay image that draws the segmentation mask and bounding box
        overlay_img = draw_segmentation_overlay(image, bbox, mask, color=(0, 255, 0), alpha=0.5)
        overlay_output_path = os.path.join(output_folder, f"person_overlay_1705_{idx}.jpg")
        cv2.imwrite(overlay_output_path, overlay_img)
        print(f"Saved overlay image: {overlay_output_path}")

if __name__ == "__main__":
    # Example usage: segment an image and store cropped, padded, and overlay images.
    image_path = "/home/azureuser/workspace/Genfied/utils/frame_1705.jpg"  # Path to your input image
    segment_and_save(image_path)
