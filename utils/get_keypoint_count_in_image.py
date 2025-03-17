import cv2
import numpy as np
import mediapipe as mp

# --- Set up Detectron2 Predictor ---
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

def setup_predictor():
    cfg = get_cfg()
    # Use a Mask R-CNN model from the Detectron2 model zoo.
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # Increase score threshold so only high-confidence detections remain.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    # Make NMS more aggressive.
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    # Load pre-trained weights.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor

predictor = setup_predictor()

# --- Set up MediaPipe Pose Detector ---
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=0,   # Lower complexity for faster processing.
    enable_segmentation=False
)

def count_keypoints(image, bbox, min_confidence=0.5):
    """
    Counts the number of keypoints with visibility above min_confidence in the given bbox.
    
    Args:
        image (np.ndarray): The full image (BGR).
        bbox (list/tuple): Bounding box in the form [x1, y1, x2, y2].
        min_confidence (float): Threshold for keypoint visibility.
    
    Returns:
        int: Number of keypoints detected with sufficient visibility.
    """
    x1, y1, x2, y2 = map(int, bbox)
    crop = image[y1:y2, x1:x2]
    # Convert from BGR to RGB.
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(crop_rgb)
    
    if results.pose_landmarks is None:
        return 0
    # Count keypoints with visibility above threshold.
    count = sum(1 for lm in results.pose_landmarks.landmark if lm.visibility >= min_confidence)
    return count

# --- Main Script ---
# Load an example image.
image_path = "frame_2975.jpg"  # Replace with your image file path.
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Could not load image.")

# Run segmentation using Detectron2 to get person detections.
outputs = predictor(image)
instances = outputs["instances"]

# Filter detections to only persons (typically class 0 in COCO).
person_indices = (instances.pred_classes == 0).nonzero().flatten()
if len(person_indices) == 0:
    print("No persons detected in the image.")
else:
    # Get bounding boxes (format: [x1, y1, x2, y2]) for persons.
    person_boxes = instances.pred_boxes.tensor[person_indices].cpu().numpy()
    
    # Loop through each detected bounding box.
    for bbox in person_boxes:
        keypoint_count = count_keypoints(image, bbox)
        print(f"BBox {bbox} -> {keypoint_count} keypoints detected.")
        
        # Draw the bounding box.
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put the keypoint count text above the bounding box.
        text = f"{keypoint_count} keypoints"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save the annotated image to disk.
output_path = "annotated_example.jpg"  # Change the filename as needed.
cv2.imwrite(output_path, image)
print(f"Annotated image saved as {output_path}")

# Optionally, display the image.
cv2.imshow("Image with Keypoint Counts", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
