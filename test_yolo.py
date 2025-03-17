import cv2
import numpy as np
import os
from ultralytics import YOLO

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

def check_aspect_ratio(bbox, min_aspect=0.2, max_aspect=0.8):
    """
    Check if the aspect ratio is reasonable for a person.
    Args:
        bbox: [x1, y1, x2, y2]
        min_aspect: Minimum acceptable width/height ratio
        max_aspect: Maximum acceptable width/height ratio
    Returns:
        bool: True if aspect ratio is within range
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if height == 0:  # Avoid division by zero
        return False
    aspect_ratio = width / height
    return min_aspect <= aspect_ratio <= max_aspect

def check_area_ratio(bbox, frame_shape, max_area_ratio=0.2):
    """
    Check if the area is reasonable for a person.
    Args:
        bbox: [x1, y1, x2, y2]
        frame_shape: (height, width) of the frame
        max_area_ratio: Maximum acceptable box area to frame area ratio
    Returns:
        bool: True if area ratio is within range
    """
    frame_height, frame_width = frame_shape[:2]
    frame_area = frame_height * frame_width
    
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]
    box_area = box_width * box_height
    
    area_ratio = box_area / frame_area
    return area_ratio <= max_area_ratio

def filter_duplicate_detections(boxes, scores, classes, image_shape, 
                               iou_threshold=0.5, containment_threshold=0.8, 
                               min_aspect=0.2, max_aspect=0.8, 
                               max_area_ratio=0.2):
    """
    Advanced filtering for duplicate and suspicious detections.
    
    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2]
        scores: (N,) array of confidence scores
        classes: (N,) array of class ids
        image_shape: (height, width) of the image
        iou_threshold: IoU above which two boxes are considered duplicates
        containment_threshold: Containment ratio above which the lower-score box is removed
        min_aspect, max_aspect: Valid aspect ratio range for person detections
        max_area_ratio: Maximum box area to frame area ratio
        
    Returns:
        filtered_boxes, filtered_scores, filtered_classes
    """
    if len(boxes) == 0:
        return boxes, scores, classes
    
    # Step 1: Filter out boxes with suspicious aspect ratios or areas
    valid_indices = []
    for i, box in enumerate(boxes):
        if classes[i] == 0 and check_aspect_ratio(box, min_aspect, max_aspect) and \
           check_area_ratio(box, image_shape, max_area_ratio):
            valid_indices.append(i)
    
    if not valid_indices:
        return np.array([]), np.array([]), np.array([])
    
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    classes = classes[valid_indices]
    
    # Step 2: Two-stage NMS (first filter containment, then IoU)
    indices = list(range(len(boxes)))
    suppressed = set()
    
    # Sort indices by descending score
    indices.sort(key=lambda i: scores[i], reverse=True)
    
    # First pass: Check for containment (one box mostly inside another)
    for i in indices:
        if i in suppressed:
            continue
            
        box_i = boxes[i]
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        
        for j in indices:
            if j <= i or j in suppressed:
                continue
                
            box_j = boxes[j]
            area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
            
            # Calculate intersection
            xx1 = max(box_i[0], box_j[0])
            yy1 = max(box_i[1], box_j[1])
            xx2 = min(box_i[2], box_j[2])
            yy2 = min(box_i[3], box_j[3])
            
            inter_w = max(0, xx2 - xx1)
            inter_h = max(0, yy2 - yy1)
            inter_area = inter_w * inter_h
            
            # Calculate containment ratio (how much of box_j is inside box_i)
            containment_ratio = inter_area / area_j
            
            # If box_j is mostly contained in box_i, suppress box_j
            if containment_ratio > containment_threshold:
                suppressed.add(j)
    
    # Second pass: Standard IoU-based NMS
    keep = []
    for i in indices:
        if i in suppressed:
            continue
            
        keep.append(i)
        box_i = boxes[i]
        
        for j in indices:
            if j <= i or j in suppressed:
                continue
                
            box_j = boxes[j]
            iou = compute_iou(box_i, box_j)
            
            if iou > iou_threshold:
                suppressed.add(j)
    
    # Create final filtered arrays
    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]
    filtered_classes = classes[keep]
    
    return filtered_boxes, filtered_scores, filtered_classes

def draw_box_overlay(image, bbox, score, class_name, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box with class name and confidence score.
    
    Args:
        image (np.array): Original image
        bbox (list or array): [x1, y1, x2, y2] coordinates
        score (float): Confidence score
        class_name (str): Class name
        color (tuple): BGR color for the box
        thickness (int): Line thickness
        
    Returns:
        image with drawn box
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare text
    text = f"{class_name}: {score:.2f}"
    font_scale = 0.5
    font_thickness = 1
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    
    # Draw text background
    cv2.rectangle(
        image, 
        (x1, y1 - text_height - 10), 
        (x1 + text_width + 5, y1), 
        color, 
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        font_thickness,
        cv2.LINE_AA,
    )
    
    return image

def yolo_detect_and_save(image_path, output_folder="yolo_detections", 
                         conf_threshold=0.3, iou_threshold=0.5, containment_threshold=0.8):
    """
    Detect persons using YOLOv8, apply advanced filtering, and save visualizations.
    
    Args:
        image_path (str): Path to input image
        output_folder (str): Folder to save outputs
        conf_threshold (float): Confidence threshold for YOLO
        iou_threshold (float): IoU threshold for NMS
        containment_threshold (float): Containment threshold for filtering
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load YOLO model
    yolo_model = YOLO("yolov8l.pt", verbose=False)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Run YOLO detection
    results = yolo_model(image, conf=conf_threshold)
    
    # Process detection results
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        # Filter for person class (class 0 in COCO)
        person_indices = np.where(classes == 0)[0]
        person_boxes = boxes[person_indices]
        person_scores = scores[person_indices]
        person_classes = classes[person_indices]
        
        all_boxes.append(person_boxes)
        all_scores.append(person_scores)
        all_classes.append(person_classes)
    
    # Combine all detections
    if all_boxes:
        boxes = np.vstack(all_boxes)
        scores = np.hstack(all_scores)
        classes = np.hstack(all_classes)
    else:
        boxes = np.array([])
        scores = np.array([])
        classes = np.array([])
    
    print(f"Found {len(boxes)} person detections before filtering")
    
    # Save original detections visualization
    orig_output_img = image.copy()
    for box, score in zip(boxes, scores):
        orig_output_img = draw_box_overlay(orig_output_img, box, score, "person", (255, 0, 0))
    
    orig_output_path = os.path.join(output_folder, f"original_detections.jpg")
    cv2.imwrite(orig_output_path, orig_output_img)
    print(f"Saved original detections: {orig_output_path}")
    
    # Apply advanced filtering
    filtered_boxes, filtered_scores, filtered_classes = filter_duplicate_detections(
        boxes, scores, classes, image.shape,
        iou_threshold=iou_threshold,
        containment_threshold=containment_threshold,
        min_aspect=0.2,
        max_aspect=0.8,
        max_area_ratio=0.2
    )
    
    print(f"Found {len(filtered_boxes)} person detections after filtering")
    
    # Create visualization with filtered boxes
    filtered_output_img = image.copy()
    for box, score in zip(filtered_boxes, filtered_scores):
        filtered_output_img = draw_box_overlay(filtered_output_img, box, score, "person", (0, 255, 0))
    
    filtered_output_path = os.path.join(output_folder, f"filtered_detections.jpg")
    cv2.imwrite(filtered_output_path, filtered_output_img)
    print(f"Saved filtered detections: {filtered_output_path}")
    
    # Save individual crops
    for idx, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
        x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2]
        
        # Skip empty crops
        if cropped.size == 0:
            continue
            
        crop_output_path = os.path.join(output_folder, f"person_{idx}.jpg")
        cv2.imwrite(crop_output_path, cropped)
        print(f"Saved cropped image: {crop_output_path}")

if __name__ == "__main__":
    # Example usage
    image_path = "utils/frame_1705.jpg"  # Path to your input image
    
    # Standard settings
    yolo_detect_and_save(
        image_path, 
        output_folder="yolo_standard",
        conf_threshold=0.3, 
        iou_threshold=0.6,
        containment_threshold=0.6
    )
    
    # Stricter settings
    yolo_detect_and_save(
        image_path, 
        output_folder="yolo_strict",
        conf_threshold=0.5, 
        iou_threshold=0.4,
        containment_threshold=0.7
    )
