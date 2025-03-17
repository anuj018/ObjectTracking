import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from deepface.commons import package_utils

tf_major = package_utils.get_tf_major_version()
tf_minor = package_utils.get_tf_minor_version()

if tf_major == 2 and tf_minor >= 16:
    try:
        import tf_keras
        print(f"tf_keras is already available - {tf_keras.__version__}")
    except ImportError as err:
        raise ValueError(
            f"You have tensorflow {tf.__version__} and this requires the tf-keras package. "
            "Please run `pip install tf-keras` or downgrade your TensorFlow version."
        ) from err

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
from ultralytics import YOLO
from status_checker import improved_human_status
import tempfile

# Import DeepFace for face detection/embedding extraction
from deepface import DeepFace

# You can later adjust these weights when integrating the combined cost
WEIGHT_BODY = 0.8
WEIGHT_FACE = 0.2
FACIAL_SIMILARITY_THRESHOLD = 0.2  # example threshold for face similarity

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = FATAL, only show critical errors
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return None
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return None
    return np.dot(vec1, vec2) / (norm1 * norm2)


def compute_iou(boxA, boxB):
    """
    Compute the Intersection-over-Union (IoU) of two bounding boxes.
    Boxes are [x1, y1, x2, y2].
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
    print("xA, yA, xB, yB, interW, interH, interArea is: ")
    print(xA, yA, xB, yB, interW, interH, interArea, boxAArea, boxBArea)
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    print(f"IoU is {iou}")
    return iou


class YOLOv8_DeepSort:
    def __init__(self, info_flag=True):
        # Initialize YOLOv8 model
        self.yolo_model = YOLO("yolov8l.pt", verbose=False)  # using yolov8l.pt
        
        # Initialize DeepSORT
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 0.8
        self.model_filename = 'yolov4_deepsort/model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
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

    def is_inside_roi(self, x, y, roi_x1, roi_y1, roi_x2, roi_y2):
        return roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2

    def calculate_overlap(self, bbox, green_box):
        """
        Calculate the overlap area between a detection bounding box and the green box.
        Args:
            bbox: [x1, y1, x2, y2] coordinates of the detection bounding box.
            green_box: [x1, y1, x2, y2] coordinates of the green box.
        Returns:
            overlap_area: Area of the intersection between the two boxes.
            bbox_area: Area of the detection bounding box.
        """
        x1_inter = max(bbox[0], green_box[0])
        y1_inter = max(bbox[1], green_box[1])
        x2_inter = min(bbox[2], green_box[2])
        y2_inter = min(bbox[3], green_box[3])
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        overlap_area = inter_width * inter_height
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return overlap_area, bbox_area

    def extract_face_embedding(self, frame, bbox, frame_num):
        """
        Extract a facial embedding for a given person detection.
        Args:
            frame: The original BGR frame.
            bbox: [x1, y1, x2, y2] coordinates of the person detection.
        Returns:
            face_embedding: A feature vector (list or numpy array) or None if no face detected.
        """
        try:
            # Crop the person region using the original bbox coordinates
            x1, y1, x2, y2 = list(map(int, bbox))
            person_region = frame[y1:y2, x1:x2]
            print(f"person region size is {person_region.shape}")
            print(f"input image is of type {type(person_region)}")
            cv2.imwrite(f"faces/person_region_{(frame_num)}.jpg", person_region)
            # Optional: you might check if the person_region is large enough for face detection
            
            # Use DeepFace.represent to both detect the face (via retinaface) and extract features.
            # enforce_detection=True will raise an exception if no face is detected.
            embedding_results = DeepFace.represent(
                img_path=person_region,
                # model_name="Facenet",         # or use "ArcFace" if preferred
                detector_backend="retinaface",  # using a robust deep learning–based face detector
                enforce_detection=True
            )
            print(f"embedding_results: {embedding_results}")
            # DeepFace.represent returns a list of dicts; we take the first one if available.
            if isinstance(embedding_results, list) and len(embedding_results) > 0:
                return embedding_results[0]["embedding"]
            else:
                return None
        except Exception as e:
            # If no face is detected or another error occurs, return None.
            if self.info_flag:
                print(f"Face extraction error: {e}")
            return None
    def extract_face_embedding_v2(self, frame, bbox, frame_num, person_index):
        """
        Extract a facial embedding using the "extract then represent" approach:
        1. Crop the person region using bbox.
        2. Run DeepFace.extract_faces on the cropped region.
        3. If a face is detected, crop that face region and run DeepFace.represent on it.
        4. Return the embedding.
        
        Args:
            frame: Original BGR frame.
            bbox: [x1, y1, x2, y2] coordinates of the person detection.
            frame_num: Frame number (for logging or saving, if needed).
        
        Returns:
            face_embedding: The extracted face embedding or None if no face is found.
        """
        try:
            # Crop the person region from the frame
            x1, y1, x2, y2 = list(map(int, bbox))
            person_region = frame[y1:y2, x1:x2]
            if self.info_flag:
                print(f"Frame {frame_num}: Person region size: {person_region.shape}")
            # Optionally, save the person region for debugging
            cv2.imwrite(f"faces/person_region_{frame_num}_{person_index}.jpg", person_region)
            
            # Use DeepFace.extract_faces on the cropped person region
            faces = DeepFace.extract_faces(person_region, detector_backend="mtcnn", enforce_detection=False)
            if self.info_flag:
                print(f"Frame {frame_num}: Number of faces detected in person region: {len(faces)}")
            if not faces:
                return None
            
            # Use the first detected face
            face_data = faces[0]
            facial_area = face_data.get("facial_area", {})
            fx = int(facial_area.get("x", 0))
            fy = int(facial_area.get("y", 0))
            fw = int(facial_area.get("w", 0))
            fh = int(facial_area.get("h", 0))
            # Crop the detected face from the person_region
            face_crop = person_region[fy:fy+fh, fx:fx+fw]
            if self.info_flag:
                print(f"Frame {frame_num}: Detected face crop size: {face_crop.shape}")
            # Now run DeepFace.represent on the face crop
            face_filename = os.path.join("faces", f"face_crop_{frame_num}.jpg")
            cv2.imwrite(face_filename, face_crop)
            representation = DeepFace.represent(
                img_path=face_crop,
                model_name="Facenet",
                # detector_backend="retinaface",
                enforce_detection=False
            )
            if self.info_flag:
                print(f"Frame {frame_num}: Representation result: {representation}")
            if representation and isinstance(representation, list) and len(representation) > 0:
                return representation[0]["embedding"]
            else:
                return None
        except Exception as e:
            if self.info_flag:
                print(f"Frame {frame_num}: Face extraction error: {e}")
            return None
        
    def extract_face_embedding_from_frame(frame, person_bbox, overlap_threshold=0.5):
        """
        Extract face embedding by processing the entire frame and selecting only the faces
        that have sufficient overlap with the given person bounding box.
        
        Args:
            frame (numpy.ndarray): The entire BGR frame.
            person_bbox (list): [x1, y1, x2, y2] coordinates for the person detection.
            overlap_threshold (float): The minimum fraction of overlap required.
            
        Returns:
            face_embedding (list or numpy.ndarray): The face embedding if a face is found with sufficient overlap,
                                                     else None.
        """
        try:
            # Run face extraction on the entire frame.
            faces = DeepFace.extract_faces(frame, detector_backend='retinaface', enforce_detection=False)
        except Exception as e:
            print(f"Error during face extraction: {e}")
            return None
    
        if not faces:
            print("No faces detected")
            return None
    
        # Function to calculate intersection area
        def calculate_overlap(box1, box2):
            x1_inter = max(box1[0], box2[0])
            y1_inter = max(box1[1], box2[1])
            x2_inter = min(box1[2], box2[2])
            y2_inter = min(box1[3], box2[3])
            inter_width = max(0, x2_inter - x1_inter)
            inter_height = max(0, y2_inter - y1_inter)
            return inter_width * inter_height
    
        # Convert person_bbox to [x1, y1, x2, y2]
        person_box = list(map(int, person_bbox))
        person_area = (person_box[2] - person_box[0]) * (person_box[3] - person_box[1])
    
        # Iterate over detected faces to find one that overlaps enough
        for face in faces:
            face_area_dict = face.get("facial_area", {})
            # Get face bounding box
            fx = face_area_dict.get("x")
            fy = face_area_dict.get("y")
            fw = face_area_dict.get("w")
            fh = face_area_dict.get("h")
            if fx is None or fy is None or fw is None or fh is None:
                continue
            face_box_coords = [fx, fy, fx + fw, fy + fh]
            overlap_area = calculate_overlap(person_box, face_box_coords)
            overlap_ratio = overlap_area / person_area
    
            if overlap_ratio >= overlap_threshold:
                # Use this face; DeepFace.extract_faces typically returns embeddings as part of the output.
                # If not, you can run DeepFace.represent on the cropped face region.
                if "embedding" in face:
                    return face["embedding"]
                else:
                    # Alternatively, extract embedding from the face crop:
                    face_img = frame[fy:fy+fh, fx:fx+fw]
                    try:
                        rep = DeepFace.represent(
                            img_path=face_img,
                            model_name="Facenet",
                            detector_backend="retinaface",
                            enforce_detection=False
                        )
                        if rep and isinstance(rep, list):
                            return rep[0]["embedding"]
                    except Exception as e:
                        print(f"Error in representing face: {e}")
                        continue
        return None
    
    def get_face_embedding_for_person(self, frame, person_bbox, overlap_threshold=0.5):
        """
        Run DeepFace.represent on the entire frame to detect faces and extract embeddings.
        Then, select the face whose bounding box (from the "facial_area" key) overlaps the person_bbox
        best. If the best IoU is above overlap_threshold, return its embedding; otherwise, return None.

        Args:
            frame (numpy.ndarray): The entire BGR frame.
            person_bbox (list): [x1, y1, x2, y2] coordinates of the person detection.
            overlap_threshold (float): Minimum IoU required.

        Returns:
            embedding (list or numpy.ndarray): The selected face embedding or None.
        """
        try:
            # Run represent on the entire frame. This will return a list of dictionaries.
            print(f"frame type is {type(frame)}")
                # Save the frame to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_filepath = tmp.name
                cv2.imwrite(temp_filepath, frame)
            reps = DeepFace.represent(
                img_path=temp_filepath,
                # model_name="Facemetcl",
                detector_backend="retinaface",
                enforce_detection=False
            )
        except Exception as e:
            print(f"Error in represent on full frame: {e}")
            return None

        if not reps or not isinstance(reps, list):
            print("No face representations returned.")
            return None

        # Ensure person_bbox is in integer format.
        person_box = list(map(int, person_bbox))
        print(f"NUMBER OF REPRESENTATIONS ARE {len(reps)}")
        best_iou = 0.0
        best_embedding = None
        for rep in reps:
            facial_area = rep.get("facial_area", {})
            print(f"Facial area is {facial_area}")
            # Make sure we have all needed coordinates.
            if not all(k in facial_area for k in ["x", "y", "w", "h"]):
                continue
            # Build the face bounding box.
            face_box = [
                int(facial_area["x"]),
                int(facial_area["y"]),
                int(facial_area["x"] + facial_area["w"]),
                int(facial_area["y"] + facial_area["h"])
            ]
            iou = compute_iou(person_box, face_box)
            if iou > best_iou:
                best_iou = iou
                best_embedding = rep.get("embedding", None)

        if best_iou >= overlap_threshold and best_embedding is not None:
            print("Found best embedding")
            return best_embedding
        else:
            print("Cant find a decent overlap")
            return None
    

    def process(self, video_path: str, output_path: str = 'output_video.mp4', desired_fps=1, images_folder="images/cam1"):
        """
        Process the video frame by frame and return detections for each frame.
        Also saves the processed video with detections.
        """
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise ValueError("Unable to open video file")
        
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(vid.get(cv2.CAP_PROP_FPS))
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(round(original_fps / desired_fps))

        print(f"\nVideo Properties:")
        print(f"Original Resolution: {width}x{height}")
        print(f"Total Frames: {total_frames}")
        print(f"FPS: {original_fps}\n")
        print(f"Detection/Tracking will run every {frame_interval} frame(s).\n")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))
        frame_num = 0
        all_frames_data = []
        start_time = datetime.utcnow()

        # Variables to hold last overlay information (if needed)
        last_active_tracks = []
        last_entity_coordinates = []
        last_frame_overlay = None

        roi_x1, roi_y1, roi_x2, roi_y2 = (768, 270, 1152, 800)  # Entrance ROI
        frame_number = 0
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                frame_start_time = time.time()
                
                current_height, current_width = frame.shape[:2]
                if current_width != width or current_height != height:
                    frame = cv2.resize(frame, (width, height))
                    if self.info_flag:
                        print(f"Frame resized to {width}x{height}")

                current_time_video = start_time + timedelta(seconds=(frame_num / original_fps))
                current_timestamp = current_time_video.timestamp()

                # Convert frame to RGB for YOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                print(f"frame shape is {frame.shape}")
                # Draw a green box (10% inset) on the frame
                frame_height, frame_width, _ = frame.shape
                shrink_percentage = 0.10
                x1_box = int(frame_width * shrink_percentage)
                y1_box = int(frame_height * shrink_percentage)
                x2_box = int(frame_width * (1 - shrink_percentage))
                y2_box = int(frame_height * (1 - shrink_percentage))
                # cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), (0, 255, 0), 2)

                # Run YOLO inference on the frame
                yolo_results = self.yolo_model(frame_rgb, stream=True)
                # print(f"YOLO RESULTS: {yolo_results}")
                # Instead of only storing (tlwh_bbox, confidence, class_name), we now also store face_embedding.
                detections_data = []  # Each element: (tlwh_bbox, confidence, class_name, face_embedding)
                person_count = 0
                for result in yolo_results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    print(f"number of ids: {len(class_ids)}")
                    for i, class_id in enumerate(class_ids):
                        class_name = self.yolo_model.names[class_id]
                        if class_name == "person" and scores[i] >= 0.40:
                            person_count += 1

                    print(f"Number of persons detected in this frame {frame_num}: {person_count}")
                    for i, class_id in enumerate(class_ids):
                        class_name = self.yolo_model.names[class_id]
                        if class_name == "person" and scores[i] >= 0.40:
                            bbox = boxes[i]  # original [x1, y1, x2, y2]
                            print(f"box is {bbox}")
                            green_box = [x1_box, y1_box, x2_box, y2_box]
                            overlap_area, bbox_area = self.calculate_overlap(bbox, green_box)
                            if overlap_area / bbox_area > 0.5:
                                # Convert bbox to TLWH for DeepSORT
                                tlwh_bbox = [
                                    bbox[0],
                                    bbox[1],
                                    bbox[2] - bbox[0],
                                    bbox[3] - bbox[1]
                                ]
                                confidence = scores[i]
                                # --- FACE EXTRACTION STEP ---
                                # Extract face features from the person detection region.
                                # face_embedding = self.extract_face_embedding(frame, bbox, frame_num)
                                # face_embedding = extract_face_embedding_from_frame(frame, bbox, iou_threshold=0.5)
                                # face_embedding = self.get_face_embedding_for_person(frame, bbox, overlap_threshold=0.5)
                                face_embedding = self.extract_face_embedding_v2(frame, bbox, frame_num, i)
                                # You might log or flag if face_embedding is None.
                                if self.info_flag:
                                    if face_embedding is None:
                                        print(f"No reliable face features for detection at frame {frame_num}")
                                    else:
                                        print(f"*******Extracted face features for detection at frame {frame_num}********")
                                # Store all information in detections_data
                                detections_data.append((tlwh_bbox, confidence, class_name, face_embedding))

                # Get body features using your pre-trained encoder (for re-ID)
                # Note: the encoder is applied on the full frame using the TLWH bboxes.
                body_features = self.encoder(frame_rgb, [d[0] for d in detections_data])
                
                # Create Detection objects and attach the face_embedding attribute.
                detections = []
                for (bbox, score, class_name, face_embedding), body_feature in zip(detections_data, body_features):
                    det = Detection(bbox, score, class_name, body_feature)
                    det.face_embedding = face_embedding  # Augment the detection with facial features
                    detections.append(det)

                # Process detections with non-max suppression and update the tracker
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                classes = np.array([d.class_name for d in detections])
                indices = preprocessing.non_max_suppression(boxes, classes, self.nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                self.tracker.predict()
                self.tracker.update(detections)

                # Visualization and status updates
                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
                active_tracks = []
                entity_coordinates = []

                for track in self.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    bbox = track.to_tlbr()
                    bbox_center_x = int(round((bbox[0] + bbox[2]) / 2))
                    bbox_center_y = int(round((bbox[1] + bbox[3]) / 2))
                    bbox_y_80 = int(round(bbox[1] + 0.8 * (bbox[3] - bbox[1])))

                    # Run your improved human status classification
                    classification, is_final = improved_human_status(
                        track, current_timestamp, self.recent_entries, self.previous_positions, roi_y1
                    )
                    if classification:
                        if "_" in classification:
                            classification_type, group_id_str = classification.split("_", 1)
                            group_id = int(group_id_str)
                        else:
                            classification_type = classification
                            group_id = None
                    else:
                        classification_type = None
                        group_id = None

                    # Update counters if classification is finalized
                    if is_final and track.track_id not in self.counted_track_ids:
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
                        self.person_status[track.track_id]["status"] = classification_type if classification_type else ""
                        self.person_status[track.track_id]["group_id"] = str(group_id) if group_id is not None else ""
                    else:
                        self.person_status[track.track_id]["status"] = ""
                        self.person_status[track.track_id]["group_id"] = ""

                    # if any(self.is_inside_roi(x, y, roi_x1, roi_y1, roi_x2, roi_y2)
                    #        for x, y in [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3])]):
                    color = colors[int(track.track_id) % len(colors)]
                    color = [int(i * 255) for i in color]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

                    id_text = f"{track.get_class()}-{track.track_id}"
                    status_text = f"Status: {classification}" if classification else "Status: undetermined"
                    (id_width, _), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                    (status_width, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)

                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-35)), (int(bbox[0])+id_width, int(bbox[1]-5)), color, -1)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-70)), (int(bbox[0])+status_width, int(bbox[1]-40)), color, -1)
                    cv2.putText(frame, id_text, (int(bbox[0]), int(bbox[1]-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                    cv2.putText(frame, status_text, (int(bbox[0]), int(bbox[1]-45)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

                    active_tracks.append(track.track_id)
                    entity_coordinates.append({
                        "person_id": str(track.track_id),
                        "x_coord": bbox_center_x,
                        "y_coord": bbox_y_80,
                        "status": self.person_status.get(track.track_id, {}).get("status", "undetermined"),
                        "group_id": self.person_status.get(track.track_id, {}).get("group_id", "")
                    })

                # Draw the entrance ROI (red rectangle)
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)

                processing_fps = 1.0 / (time.time() - frame_start_time)
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
                    color = (0, 255, 255)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)

            out.write(frame)
            frame_num += 1

        vid.release()
        out.release()
        self.tracker.save_global_database()

        videofile_name = "cropped_video_multiple_humans"
        output_json_file = f"tracking_results_{videofile_name}_{desired_fps}fps.json"
        output_file = os.path.join(os.getcwd(), output_json_file)
        with open(output_file, "w") as f:
            json.dump(all_frames_data, f, indent=4)

        return {
            "singles": self.singles,
            "couples": self.couples,
            "groups": self.groups,
            "all_frames_data": all_frames_data
        }


def process_video(file_path: str, output_path: str = 'output_video.mp4', info_flag=True, desired_fps=1, images_folder="images/cam1"):
    """
    Process the video and extract required information.
    """
    model = YOLOv8_DeepSort(info_flag=info_flag)
    results = model.process(file_path, output_path, desired_fps, images_folder)
    singles, couples, groups = results["singles"], results["couples"], results["groups"]
    print(f"singles count {singles}, couples count: {couples}, groups: {groups}")
    return results


if __name__ == "__main__":
    videofile_name = "cropped_video_multiple_humans"
    video_path = f"/home/azureuser/workspace/Genfied/input_videos/{videofile_name}.mp4"
    output_path = f"/home/azureuser/workspace/Genfied/input_videos/result_{videofile_name}.mp4"
    desired_fps = 25
    results = process_video(video_path, output_path, info_flag=True, desired_fps=desired_fps, images_folder="images/video2long")
    output_json_file = f"tracking_results_{videofile_name}_{desired_fps}fps.json"
    print(f"Processing complete. Results saved in {output_json_file}")
    print(f"Processed video saved as '{output_path}'")
