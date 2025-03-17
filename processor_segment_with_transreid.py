import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
transreid_path = os.path.join(current_dir, "TransReID")
if transreid_path not in sys.path:
    sys.path.append(transreid_path)
from TransReID.config import cfg
from TransReID.model import make_model
from TransReID.datasets.transforms import build_transforms
from TransReID.processor import extract_features
import torch
import json
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from yolov4_deepsort.deep_sort import preprocessing, nn_matching
from yolov4_deepsort.deep_sort.detection import Detection
from yolov4_deepsort.deep_sort.tracker import Tracker
from yolov4_deepsort.tools import generate_detections as gdet
from yolov4_deepsort.deep_sort.track import TrackState

from ultralytics import YOLO
from status_checker import improved_human_status
from PIL import Image
cfg.merge_from_file("/home/azureuser/workspace/Genfied/TransReID/configs/Market/vit_transreid.yml")  # Use the MSMT17 config
cfg.TEST.WEIGHT = "/home/azureuser/workspace/Genfied/TransReID/models/vit_base_msmt.pth"  # Fine-tuned model weights
cfg.freeze()
import mediapipe as mp

# === 2️⃣ Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MSMT17 has 4,101 IDs so we pass num_class=4101
model = make_model(cfg, num_class = 1041, camera_num=0, view_num=0).to(device)
model.load_param(cfg.TEST.WEIGHT)
model.eval()

transform = build_transforms(cfg, is_train=False)

# --- Detectron2 segmentation code ---
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import config
import logging
import os

# Configure logger at the top of your module (or in a separate config module)
LOG_FILENAME = "tracker.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from azure.storage.blob import BlobServiceClient, ContentSettings
blob_service_client = BlobServiceClient.from_connection_string(config.AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(config.AZURE_CONTAINER_NAME)
logger.info(f"Connected to Azure container: {config.AZURE_CONTAINER_NAME}")

# Initialize MediaPipe Pose once (for efficiency)
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, 
                             model_complexity=0,   # Lower complexity for real-time use
                             enable_segmentation=False)
MIN_FRAMES_FOR_CONFIRMATION = 3
def confirm_human(image, bbox, min_keypoints=3, min_confidence=0.5):
    """
    Checks if the region defined by bbox contains sufficient human keypoints.
    
    Args:
        image (np.ndarray): The full image (BGR format).
        bbox (list/tuple): Bounding box [x1, y1, x2, y2].
        min_keypoints (int): Minimum number of keypoints required to confirm a human.
        min_confidence (float): Minimum confidence/visibility required for each keypoint.
    
    Returns:
        bool: True if the region is confirmed as human, False otherwise.
    """
    # Crop the region corresponding to the bounding box.
    x1, y1, x2, y2 = map(int, bbox)
    crop = image[y1:y2, x1:x2]
    
    # Convert BGR image to RGB as required by MediaPipe.
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    # Process the cropped image to obtain pose landmarks.
    results = pose_detector.process(crop_rgb)
    
    # Check if landmarks were detected.
    if results.pose_landmarks is None:
        return False

    # Count landmarks (keypoints) with visibility above threshold.
    count = sum(1 for lm in results.pose_landmarks.landmark if lm.visibility >= min_confidence)
    
    # Return True if enough keypoints are detected.
    return count >= min_keypoints

def upload_video_to_azure(video_filepath: str, blob_name: str) -> str:
    """
    Uploads a video file to Azure Blob Storage.

    Args:
        video_filepath (str): The local path to the video file.
        blob_name (str): The name to assign to the blob in Azure.

    Returns:
        str: The URL of the uploaded blob if successful, otherwise None.
    """
    try:
        with open(video_filepath, "rb") as f:
            file_bytes = f.read()
        logger.info(f"Uploading video from {video_filepath} as blob '{blob_name}'")
        # Upload the video file; overwrite if it already exists
        container_client.upload_blob(
            name=blob_name,
            data=file_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type="video/mp4")
        )
        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{blob_name}"
        logger.info(f"Upload successful. Blob URL: {blob_url}")
        return blob_url
    except Exception as e:
        logger.error("Error uploading video", exc_info=True)
        return None

def setup_predictor():
    """
    Set up a Detectron2 configuration for Mask R-CNN with stricter thresholds
    to reduce duplicate detections.
    """
    cfg = get_cfg()
    # Use Mask R-CNN model from the model zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # Increase score threshold so only high-confidence detections remain
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # Make NMS more aggressive to merge/suppress duplicates
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
    return interArea / float(boxAArea + boxBArea - interArea)

def filter_duplicate_detections(boxes, scores, masks, frame_copy, iou_threshold=0.7):
    """
    Filter out duplicate detections with high IoU.
    Keep the detection with the higher confidence score.
    Additionally, draw the suppressed (duplicate) boxes in red on frame_copy.
    
    Args:
        boxes (np.ndarray): Array of bounding boxes.
        scores (np.ndarray): Array of confidence scores.
        masks (np.ndarray): Array of masks.
        frame_copy (np.ndarray): The image frame on which to draw discarded boxes.
        iou_threshold (float): IoU threshold for considering two boxes as duplicates.
        
    Returns:
        filtered_boxes, filtered_scores, filtered_masks: The filtered detection outputs.
    """
    indices = list(range(len(boxes)))
    suppressed = set()
    indices.sort(key=lambda i: scores[i], reverse=True)
    keep = []
    
    for i in indices:
        if i in suppressed:
            # Draw suppressed detection in red (if not already drawn in the inner loop)
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            continue
        
        keep.append(i)
        for j in indices:
            if j <= i or j in suppressed:
                continue
            iou = compute_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                suppressed.add(j)
                # Draw the suppressed box in red
                x1, y1, x2, y2 = map(int, boxes[j])
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                logger.info(f"suppressed box {x1}, {y1}, {x2}, {y2}. Had an IoU of {iou}")
    
    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]
    filtered_masks = masks[keep]
    return filtered_boxes, filtered_scores, filtered_masks


def crop_without_resize(image, bbox, mask):
    """
    Crop the image using the given bbox and apply the mask without resizing or padding.
    """
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = image[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2]
    # Apply mask: set pixels where mask==0 to black.
    cropped_img[cropped_mask == 0] = 0
    return cropped_img


def has_left_store_percent(bbox, line_y, threshold=0.20):
    """
    Determines if more than a given percentage of the bounding box is above a horizontal line.

    In image coordinates (y increases downward):
      - If the entire bbox is above the line (y2 < line_y), fraction = 1.0.
      - If the entire bbox is below the line (y1 >= line_y), fraction = 0.0.
      - Otherwise, fraction = (line_y - y1) / (y2 - y1).

    Args:
        bbox (list or tuple): The bounding box [x1, y1, x2, y2].
        line_y (int): The y-coordinate of the horizontal line.
        threshold (float): The fraction threshold (default 0.30 for 30%).

    Returns:
        (bool, float): A tuple where the first element is True if the fraction above the line 
                       is greater than the threshold, and the second element is the fraction.
    """
    y1 = bbox[1]
    y2 = bbox[3]
    height = y2 - y1
    if height <= 0:
        return False, 0.0

    # If the entire box is above the line.
    if y2 < line_y:
        fraction_above = 1.0
    # If the entire box is below the line.
    elif y1 >= line_y:
        fraction_above = 0.0
    else:
        fraction_above = (line_y - y1) / height

    return fraction_above > threshold, fraction_above

def is_entering_store_percent(bbox, line_y, threshold=0.70):
    """
    Determines if more than a given percentage of the bounding box is below a horizontal line.

    In image coordinates (y increases downward):
      - If the entire bbox is below the line (y1 >= line_y), fraction = 1.0.
      - If the entire bbox is above the line (y2 < line_y), fraction = 0.0.
      - Otherwise, fraction = (y2 - line_y) / (y2 - y1).

    Args:
        bbox (list or tuple): The bounding box [x1, y1, x2, y2].
        line_y (int): The y-coordinate of the horizontal line marking the store entrance.
        threshold (float): The fraction threshold (default 0.70 means 70%).

    Returns:
        (bool, float): A tuple where the first element is True if the fraction below the line
                       is greater than the threshold (i.e. the person is inside the store), 
                       and the second element is the calculated fraction.
    """
    y1 = bbox[1]
    y2 = bbox[3]
    height = y2 - y1
    if height <= 0:
        return False, 0.0

    # If the entire bbox has entered
    if y1 >= line_y:
        fraction_below = 1.0
        return False,fraction_below
    # If the entire bbox has not entered
    elif y2 <= line_y:
        fraction_below = 0.0
        return False,fraction_below
    else:
        fraction_below = (y2 - line_y) / height

    return fraction_below > threshold, fraction_below




class Segmentation_DeepSort:
    def __init__(self, info_flag=True):
        # Initialize YOLOv8 model
        self.seg_predictor = setup_predictor()
        # Initialize DeepSORT
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 0.8
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)
        
        # Flag for detailed information
        self.info_flag = info_flag

        self.person_status = {}
        self.recent_entries = {"entries": [], "classified": {}}
        self.previous_positions = {}  # Stores previous coordinates for direction checking
        self.counted_track_ids = set() 
        self.singles = 0
        self.couples = 0
        self.groups = 0    
        self.false_positive_blacklist = []
    
    def is_inside_roi(self,x, y, roi_x1, roi_y1, roi_x2, roi_y2):
        return roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2

    def is_blacklisted(self, bbox, tolerance=10):
        """
        Checks if the given bbox is similar to one in the blacklist.
        bbox format: [x1, y1, x2, y2]
        """
        for bb in self.false_positive_blacklist:
            if (abs(bbox[0] - bb[0]) <= tolerance and
                abs(bbox[1] - bb[1]) <= tolerance and
                abs(bbox[2] - bb[2]) <= tolerance and
                abs(bbox[3] - bb[3]) <= tolerance):
                return True
        return False

    def calculate_overlap(self, bbox, green_box):
        """
        Calculate the overlap area between a detection bounding box and the green box.
        Args:
            bbox: [x1, y1, x2, y2] coordinates of the detection bounding box.
            green_box: [x1, y1, x2, y2] coordinates of the green box.
        Returns:
            overlap_area:xArea of the intersection between the two boxes.
            bbox_area: Area of the detection bounding box.
        """
        # Calculate intersection coordinates
        x1_inter = max(bbox[0], green_box[0])
        y1_inter = max(bbox[1], green_box[1])
        x2_inter = min(bbox[2], green_box[2])
        y2_inter = min(bbox[3], green_box[3])

        # Calculate intersection area
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        overlap_area = inter_width * inter_height

        # Calculate detection bounding box area
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        return overlap_area, bbox_area


    def process(self, video_path: str, output_path: str = 'output_video.mp4', desired_fps = 1, images_folder="images/cam1"):
        """
        Process the video frame by frame and return detections for each frame.
        Also saves the processed video with detections.
        """
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise ValueError("Unable to open video file")
        
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        # Get video properties for output
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(vid.get(cv2.CAP_PROP_FPS))
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(round(original_fps / desired_fps))

        logger.info(f"\nVideo Properties:")
        logger.info(f"Original Resolution: {width}x{height}")
        logger.info(f"Total Frames: {total_frames}")
        logger.info(f"FPS: {original_fps}\n")
        logger.info(f"Detection/Tracking will run every {frame_interval} frame(s).\n")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))
        # logger.info(f"Processing every {frame_interval} frames to achieve {desired_fps} fps sampling rate.")

        frame_num = 0
        all_frames_data = []
        start_time = datetime.utcnow()  # Start time for the video

        # Store the last computed overlay information so that we can re-use it on in-between frames.
        last_active_tracks = []
        last_entity_coordinates = []
        last_frame_overlay = None  # You could store a copy of the last frame with overlays

        roi_x1, roi_y1, roi_x2, roi_y2 = (768, 270, 1152, 800)  # Entrance Region

        while True:
            ret, frame = vid.read()
            if not ret:
                break
            frame_copy = frame.copy()
            if frame_num % frame_interval == 0:
                frame_start_time = time.time()
                logger.info(f"processing frame {frame_num}")
                frame_num_text = f"Frame {frame_num}"

                current_height, current_width = frame.shape[:2]
                # logger.info(f"\nProcessing Frame #{frame_num}")
                # logger.info(f"Current Frame Resolution: {current_width}x{current_height}")

                # Ensure consistent resolution
                if current_width != width or current_height != height:
                    frame = cv2.resize(frame, (width, height))
                    logger.info(f"Frame resized to match original resolution: {width}x{height}")

                # Calculate the timestamp based on frame number and FPS
                current_time_video = start_time + timedelta(seconds=(frame_num / original_fps))
                current_timestamp = current_time_video.timestamp() 

                # Convert BGR to RGB for YOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Define the green box dimensions (10% smaller on all sides)
                frame_height, frame_width, _ = frame.shape
                shrink_percentage = 0.10
                shrink_percentage_top = 0.10
                x1_box = 0  # Align with the left edge of the frame
                y1_box = int(frame_height * shrink_percentage_top)  # Variable top edge
                x2_box = frame_width  # Align with the right edge of the frame
                y2_box = frame_height  # Align with the bottom edge of the frame
                line_y = y1_box
                # Draw the green box on the frame
                cv2.rectangle(frame_copy, (x1_box, y1_box), (x2_box, y2_box), (0, 255, 0), 2)

                # Run YOLO inference on the full frame
                # yolo_results = self.yolo_model(frame_rgb, stream=True)
                outputs = self.seg_predictor(frame)
                instances = outputs["instances"]
                person_indices = (instances.pred_classes == 0).nonzero().flatten()
                if len(person_indices) == 0:
                    detections = []
                else:
                    person_boxes = instances.pred_boxes.tensor[person_indices].cpu().numpy()
                    person_scores = instances.scores[person_indices].cpu().numpy()
                    person_masks = instances.pred_masks[person_indices].cpu().numpy()

                    person_boxes, person_scores, person_masks = filter_duplicate_detections(
                        person_boxes, person_scores, person_masks, frame_copy, iou_threshold=0.9
                    )
                    detections = []
                    person_count = 0
                    for i, bbox in enumerate(person_boxes):

                        if self.is_blacklisted(bbox):
                            continue
                        score = person_scores[i]
                        mask = person_masks[i]
                        cv2.rectangle(frame_copy, (int(bbox[0]), int(bbox[1])),
                                      (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                        person_count += 1
                        green_box = [x1_box, y1_box, x2_box, y2_box]
                        overlap_area, bbox_area = self.calculate_overlap(bbox, green_box)

                        if overlap_area / bbox_area > 0.7:
                            # Convert bbox to TLWH for DeepSORT
                            tlwh_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                            detections.append((tlwh_bbox, score, "person", mask))
                    logger.info(f"Number of persons detected in frame {frame_num}: {person_count}")

                # Get features
                body_features = []
                for det in detections:
                    tlwh_bbox, confidence, class_name, mask = det
                    x, y, w, h = map(int, tlwh_bbox)
                    crop = frame_rgb[y:y+h, x:x+w]
                    if crop.size == 0:
                        body_features.append(np.zeros((1,)))  # You may want to set the correct shape
                        continue
                    crop_masked = crop_without_resize(frame, [x, y, x+w, y+h], mask)
                    crop_pil = Image.fromarray(cv2.cvtColor(crop_masked, cv2.COLOR_BGR2RGB))
                    crop_tensor = transform(crop_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = extract_features(model, crop_tensor).cpu().numpy()
                        feat = feat.reshape(-1)  # or np.squeeze(feat) if that gives you the correct shape
                    body_features.append(feat)

                # Create Detection objects
                detections_objs = [Detection(bbox, score, class_name, feature) 
                            for (bbox, score, class_name, _), feature in zip(detections, body_features)]
                
                logger.info(f"")

                # Process detections
                boxes_np = np.array([d.tlwh for d in detections_objs])
                scores_np = np.array([d.confidence for d in detections_objs])
                classes_np = np.array([d.class_name for d in detections_objs])
                indices = preprocessing.non_max_suppression(boxes_np, classes_np, self.nms_max_overlap, scores_np)
                detections_objs = [detections_objs[i] for i in indices] #check boxes that remain

                # Update tracker
                self.tracker.predict()
                self.tracker.update(detections_objs)

                # Initialize color map for visualization
                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                active_tracks = []
                entity_coordinates = []

                # logger.info frame header for detailed info
                # if self.info_flag:
                    # logger.info(f"\nFrame {frame_num} Tracking Details:")

                for track in self.tracker.tracks:
                    if track.state == TrackState.Tentative:
                        bbox = track.to_tlbr()
                        if not confirm_human(frame, bbox):
                            self.false_positive_blacklist.append(bbox)
                            track.mark_missed()
                        else:
                            if not getattr(track, "has_entered", False):
                                has_entered, fraction = is_entering_store_percent(bbox, line_y)
                                if has_entered:
                                    track.has_entered = True
                                    logger.info(f"Track {track.track_id} has entered the store tentative_check (bbox y1: {bbox[1]} is above line {line_y}).")

                        
                    
                    if track.time_since_update > 1:
                        bbox = track.to_tlbr()
                        has_left, fraction = has_left_store_percent(bbox, line_y)
                        has_entered, fraction = is_entering_store_percent(bbox, line_y)
                        if has_left:
                            if track.mean[5] < 0:
                                self.tracker.mark_track_as_left(track)
                                logger.info(f"y1 is {line_y}")
                                logger.info(f"fraction is is {fraction}")
                                logger.info(f"Mean is {track.mean}")
                                logger.info(f"Track {track.track_id} has left the store (bbox y1: {bbox[1]} is above line {line_y}).")
                        if has_entered:
                            if track.mean[5] > 0:
                                logger.info(f"fraction is is {fraction}")
                                logger.info(f"Mean is {track.mean}")
                                logger.info(f"Track {track.track_id} has entered the store (bbox y1: {bbox[1]} is above line {line_y}).")
                    

                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    bbox = track.to_tlbr()
                    top_left = (bbox[0], bbox[1])
                    top_right = (bbox[2], bbox[1])
                    bottom_left = (bbox[0], bbox[3])
                    bottom_right = (bbox[2], bbox[3])
                    bbox_center_x = int(round((bbox[0] + bbox[2]) / 2))
                    bbox_center_y = int(round((bbox[1] + bbox[3]) / 2))
                    bbox_y_80 = int(round(bbox[1] + 0.8 * (bbox[3] - bbox[1])))

                    classification, is_final = improved_human_status(track, current_timestamp, self.recent_entries, self.previous_positions, roi_y1)#, entrance_direction='top')
                    if classification:
                        if "_" in classification:
                            classification_type, group_id_str = classification.split("_", 1)
                            group_id = int(group_id_str)
                            logger.info(f"classification_type for {track.track_id} is {classification_type} and id is {group_id}")
                        else:
                            classification_type = classification
                            group_id = None
                            logger.info(f"classification_type for {track.track_id} is {classification_type} and id is {group_id}")
                    else:
                        classification_type = None
                        group_id = None
                    # logger.info(f"Classification result for Track ID {track.track_id}:")
                    # logger.info(f"  Type: {classification_type}, Group ID: {group_id}")
                    # logger.info(f"  Is it finalized: {is_final}")
                    # Update counters if classification is finalized
                    if is_final and track.track_id not in self.counted_track_ids:
                    # if classification and track.track_id not in self.recent_entries["classified"]:
                        if classification_type == "alone":
                            self.singles += 1
                        elif classification_type == "couple":
                            self.couples += 1
                        elif classification_type == "group":
                            self.groups += 1
                        self.counted_track_ids.add(track.track_id)
                    
                    if track.track_id not in self.person_status:
                        self.person_status[track.track_id] = {"status": "", "group_id": ""}
                    if is_final:
                        logger.info(f"finalizing status of {track.track_id} to {classification_type}")
                        self.person_status[track.track_id]["status"] = classification_type if classification_type else ""
                        self.person_status[track.track_id]["group_id"] = str(group_id) if group_id is not None else ""
                    else:
                        # For non-finalized tracks, leave the status and group_id as empty strings.
                        self.person_status[track.track_id]["status"] = ""
                        self.person_status[track.track_id]["group_id"] = ""

                # entity_coordinates.append({
                #     "person_id": str(track.track_id),
                #     "x_coord": bbox_center_x,
                #     "y_coord": bbox_center_y,
                #     "status": classification if classification else "undetermined"
                # })

                # active_tracks.append(track.track_id)

                # if (roi_x1 <= bbox_center_x <= roi_x2 and roi_y1 <= bbox_center_y <= roi_y2):
                    # if any(self.is_inside_roi(x, y, roi_x1, roi_y1, roi_x2, roi_y2) for x, y in [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3])]):
                        # logger.info detailed tracking information
                        # if self.info_flag:
                            # logger.info(f"Tracker ID: {track.track_id}")
                            # logger.info(f"Class: {track.get_class()}")
                            # logger.info(f"BBox Coords (xmin, ymin, xmax, ymax): ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])})")
                            # logger.info(f"Center Coordinates (x, y): ({bbox_center_x}, {bbox_center_y})")
                            # logger.info(f"80y Offset Coordinates (x, y): ({bbox_center_x}, {bbox_y_80})")

                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]

                    # Draw bounding box
                    cv2.rectangle(frame_copy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

                    # Prepare text for display (ID and coordinates)
                    id_text = f"{track.get_class()}-{track.track_id}"
                    status_text = f"Status: {classification}" if classification else "Status: undetermined"
                    coord_text = f"({bbox_center_x}, {bbox_y_80})"

                    # Calculate text sizes for positioning
                    (id_width, id_height), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                    (status_width, status_height), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                    id_text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
                    coord_text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]

                    # Draw background rectangles and texts
                    cv2.rectangle(frame_copy, 
                                (int(bbox[0]), int(bbox[1]-35)), 
                                (int(bbox[0])+id_width, int(bbox[1]-5)), 
                                color, -1)
                    cv2.rectangle(frame_copy, 
                                (int(bbox[0]), int(bbox[1]-70)), 
                                (int(bbox[0])+status_width, int(bbox[1]-40)), 
                                color, -1)

                    cv2.putText(frame_copy, id_text,
                                (int(bbox[0]), int(bbox[1]-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                    cv2.putText(frame_copy, status_text,
                                (int(bbox[0]), int(bbox[1]-45)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

                    # Get text size (width, height) and baseline
                    (text_width, text_height), baseline = cv2.getTextSize(frame_num_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                    # Define a margin from the top-right corner
                    margin = 10
                    # Calculate the coordinates for the text (top-right corner)
                    x = frame.shape[1] - text_width - margin
                    y = text_height + margin  # y-coordinate is measured from the top
                    cv2.putText(frame_copy, frame_num_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                    active_tracks.append(track.track_id)
                    if bbox_center_x is not None and bbox_center_y is not None:
                        entity_coordinates.append({
                            "person_id": str(track.track_id),
                            "x_coord": bbox_center_x,
                            "y_coord": bbox_y_80,
                            "status": self.person_status.get(track.track_id, {}).get("status", "undetermined"),
                            "group_id": self.person_status.get(track.track_id, {}).get("group_id", "")
                        })
                # entity_coordinates.append({
                #     "person_id": str(track.track_id),
                #     "x_coord": bbox_center_x,
                #     "y_coord": bbox_y_80,
                #     "status": classification if classification else "undetermined"
                # })
                cv2.rectangle(frame_copy, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)  # Red for entrance ROI
            # Calculate FPS
                last_active_tracks = active_tracks
                last_entity_coordinates = entity_coordinates
                processing_fps = 1.0 / (time.time() - frame_start_time)
                # logger.info(f"Processing FPS for key frame: {processing_fps:.2f}")
            # calculated_fps = 1.0 / (time.time() - frame_start_time)
            # fps = 1.0 / (time.time() - frame_start_time)
            # logger.info(f"FPS: {fps:.2f}")

            # logger.info total number of people in the frame
                # if self.info_flag:
                #     logger.info(f"\nTotal people in frame: {len(active_tracks)}")
                #     if active_tracks:
                #         logger.info(f"Active tracker IDs: {active_tracks}")

                frame_data = {
                    "camera_id": 1,
                    "frame_number": frame_num,
                    "resolution": f"{width}x{height}",
                    "fps": round(processing_fps, 2),
                    "image_url": "",
                    "is_organised": True,
                    "no_of_people": len(active_tracks),
                    "current_time": current_time_video.isoformat(),
                    "ENTITY_COORDINATES": entity_coordinates
                }

                all_frames_data.append(frame_data)
            else:
                self.tracker.predict()
                for track in self.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    color = (0, 255, 255)  # For example, a different color for predicted boxes
                    cv2.rectangle(frame_copy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame_copy, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)

            # Write the frame to output video
            out.write(frame_copy)
            frame_num += 1

        # Release everything
        vid.release()
        out.release()
        self.tracker.save_global_database()

        # Save tracking results to JSON
        videofile_name = "video1_test_overlap"
        output_json_file = f"tracking_results_{videofile_name}_{desired_fps}fps.json"
        output_file = os.path.join(os.getcwd(), output_json_file)
        with open(output_file, "w") as f:
            json.dump(all_frames_data, f, indent=4)

        # return all_frames_data
        return {
            "singles": self.singles,
            "couples": self.couples,
            "groups": self.groups,
            "all_frames_data": all_frames_data
        }


def process_video(file_path: str, output_path: str = 'output_video.mp4', info_flag=True, desired_fps = 1, images_folder="images/cam1"):
    """
    Process the video and extract required information.
    Args:
        file_path: Path to input video
        output_path: Path to save processed video
        info_flag: Whether to logger.info detailed information
    """
    processor = Segmentation_DeepSort(info_flag=info_flag)
    results = processor.process(file_path, output_path, desired_fps, images_folder)
    singles, couples, groups = results["singles"], results["couples"], results["groups"]
    logger.info(f"singles count {singles}, couples count: {couples}, groups: {groups}")
    return results

if __name__ == "__main__":
    videofile_name = "video1_test_overlap"
    video_path = f"/home/azureuser/workspace/Genfied/input_videos/{videofile_name}.mp4"
    output_path = f"/home/azureuser/workspace/Genfied/input_videos/result_{videofile_name}_Transreid_lowconf_030325_wooverlap_logic.mp4"
    desired_fps = 25
    results = process_video(video_path, output_path, info_flag=True, desired_fps = desired_fps, images_folder="images/video2long")
    output_json_file = f"tracking_results_{videofile_name}_{desired_fps}fps.json"
    logger.info(f"Processing complete. Results saved in {output_json_file}")
    logger.info(f"Processed video saved as '{output_path}'")

# After processing is complete, upload the video to Azure
    blob_name = os.path.basename(output_path)
    uploaded_url = upload_video_to_azure(output_path, blob_name)
    if uploaded_url:
        logger.info(f"Video successfully uploaded to Azure: {uploaded_url}")
    else:
        logger.error("Failed to upload video to Azure.")