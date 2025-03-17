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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
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
                suppressed.add(j)
    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]
    filtered_masks = masks[keep]
    return filtered_boxes, filtered_scores, filtered_masks

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
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = image[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2]
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
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask == 1] = color
    cv2.addWeighted(colored_mask, alpha, overlay_img, 1 - alpha, 0, overlay_img)
    cv2.rectangle(overlay_img, (x1, y1), (x2, y2), color, thickness=2)
    return overlay_img

def process_video(video_path, output_folder="video_output", frame_interval=5):
    """
    Process a video file, performing segmentation on every Nth frame,
    and save the cropped images and overlay images for all detected persons.
    
    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Folder where output images will be saved.
        frame_interval (int): Process every Nth frame.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create sub-folders for cropped and overlay images
    crop_folder = os.path.join(output_folder, "crops")
    overlay_folder = os.path.join(output_folder, "overlays")
    os.makedirs(crop_folder, exist_ok=True)
    os.makedirs(overlay_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return
    
    predictor = setup_predictor()
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 'frame_interval' frame
        if frame_idx % frame_interval == 0:
            outputs = predictor(frame)
            instances = outputs["instances"]
            # Filter for person detections (COCO class id 0 corresponds to "person")
            person_indices = (instances.pred_classes == 0).nonzero().flatten()
            if person_indices.numel() == 0:
                frame_idx += 1
                continue
            person_boxes = instances.pred_boxes.tensor[person_indices].cpu().numpy()
            person_scores = instances.scores[person_indices].cpu().numpy()
            person_masks = instances.pred_masks[person_indices].cpu().numpy()

            print(f"Frame {frame_idx}: Found {len(person_boxes)} person detections before duplicate filtering.")

            # Filter duplicate detections if necessary
            person_boxes, person_scores, person_masks = filter_duplicate_detections(
                person_boxes, person_scores, person_masks, iou_threshold=0.7
            )

            print(f"Frame {frame_idx}: {len(person_boxes)} detections after duplicate filtering.")

            for idx, (bbox, mask) in enumerate(zip(person_boxes, person_masks)):
                crop_img = crop_without_resize(frame, bbox, mask)
                crop_output_path = os.path.join(crop_folder, f"frame_{frame_idx}_person_{idx}.jpg")
                cv2.imwrite(crop_output_path, crop_img)
                
                overlay_img = draw_segmentation_overlay(frame, bbox, mask, color=(0, 255, 0), alpha=0.5)
                overlay_output_path = os.path.join(overlay_folder, f"frame_{frame_idx}_person_{idx}.jpg")
                cv2.imwrite(overlay_output_path, overlay_img)
                
                saved_count += 1
                print(f"Frame {frame_idx}: Saved person {idx} (Total saved: {saved_count}).")
        
        frame_idx += 1

    cap.release()
    print("Processing complete.")

if __name__ == "__main__":
    # Example usage: process a video file and save segmented images every 5th frame
    video_path = "/home/azureuser/workspace/Genfied/input_videos/video1long_cropped_lady.mp4"  # Replace with your video file path
    process_video(video_path, output_folder="video_output", frame_interval=5)
