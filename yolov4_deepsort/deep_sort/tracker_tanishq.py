# tracker.py
import numpy as np
import os
import pickle
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track_tanishq import Track
from .track_tanishq import TrackState
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

class Tracker:
    """
    This is the multi-target tracker.
    keeping the max_age forces the features to be checked with the features in the global database as quickly as possible. 
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=3, n_init=5, matching_threshold=0.5):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.matching_threshold = matching_threshold  # Set the matching threshold
        # logger.info(f"MATCHING THRESHOLD IS {self.matching_threshold}")
        # logger.info(f"METRIC MATCHING THRESHOLD IS {self.metric.matching_threshold}")
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

        # Load or create the global database
        self.global_database_file = "global_database_anuj.pkl"
        if os.path.exists(self.global_database_file):
            with open(self.global_database_file, "rb") as f:
                self.global_database = pickle.load(f)
            # logger.info(f"Loaded global database from {self.global_database_file}.")
        else:
            self.global_database = {}
            logger.info(f"Created new global database.")
        self.retired_ids = set()

    def mark_track_as_left(self, track):
        """Mark a track as left and retire its ID."""
        # Remove from global database if present
        if track.track_id in self.global_database:
            del self.global_database[track.track_id]
        # Add to retired IDs so it is not reused
        self.retired_ids.add(track.track_id)
        # Mark the track as deleted so it will be removed from active tracking
        track.state = TrackState.Deleted
        logger.info(f"Track {track.track_id} retired and removed from global database.")

    def save_global_database(self):
        """Save the global database to a file."""
        with open(self.global_database_file, "wb") as f:
            pickle.dump(self.global_database, f)
        logger.info(f"Saved global database to {self.global_database_file}.")

    def _is_completely_inside_with_gap(self, detection_tlbr, green_box, min_gap=5):
        """
        Check if detection is completely inside green box with a minimum gap.
        Args:
            detection_tlbr: [x1, y1, x2, y2] of detection box
            green_box: [x1, y1, x2, y2] of green box
            min_gap: minimum pixel gap required between boxes
        Returns:
            bool: True if detection is inside with minimum gap
        """
        return (detection_tlbr[0] >= green_box[0] + min_gap and
                detection_tlbr[1] >= green_box[1] + min_gap and
                detection_tlbr[2] <= green_box[2] - min_gap and
                detection_tlbr[3] <= green_box[3] - min_gap)

    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management."""
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set for matched detections.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx], self.global_database)
            logger.info(f"Matched detection to track_id {self.tracks[track_idx].track_id}.")

        # Mark unmatched tracks as missed.
        for track_idx in unmatched_tracks:
            # logger.info(f"Marking track_id {self.tracks[track_idx].track_id} as missed.")
            self.tracks[track_idx].mark_missed()

        # Handle unmatched detections.
        for detection_idx in unmatched_detections:
            # logger.info("Checking global database for a match.")
            detection = detections[detection_idx]
            bbox = detection.to_tlbr()
            entering, fraction = is_entering_store_percent(bbox, line_y = 108, threshold=0.70) #there is a slim change that a person leaving the store might detect this function if the detection of the person leaving the store is unmatched and >= 0.7 of the box is below the green line
            if entering:
                # mean, covariance = self.kf.initiate(detection.to_xyah())
                # if mean[5] > 0:
                logger.info(f"Detection with bbox {bbox} is entering the store (fraction below line: {fraction}).")
                # Do not check the global database; force a new ID.
                matched_track_id = None
            else:
                # For detections that are not entering (e.g. those that might be re‑identified as someone who left),
                # check the global database.
                # matched_track_id = self._match_with_global_database(detection.feature)
                matched_track_id = self._match_with_global_database(detection.feature, bbox)
            # matched_track_id = self._match_with_global_database(detection.fea ture)

            if matched_track_id is not None:
                # Re-identify the track using the database.
                mean, covariance = self.kf.initiate(detection.to_xyah())
                class_name = detection.get_class()
                self.tracks.append(Track(
                    mean, covariance, matched_track_id, self.n_init, self.max_age,
                    detection.feature, class_name))

                # Append features from the database to the new track's features list.
                self.tracks[-1].features = self.global_database[matched_track_id]["features"]
                logger.info(f"Re-identified track_id {matched_track_id} from global database with {len(self.tracks[-1].features)} features.")
            else:
                # Assign a new track_id.
                mean, covariance = self.kf.initiate(detection.to_xyah())
                class_name = detection.get_class()
                self.tracks.append(Track(
                    mean, covariance, self._next_id, self.n_init, self.max_age,
                    detection.feature, class_name))
                self._next_id += 1
                # logger.info(f"Assigned new track_id {self._next_id - 1}.")

        # Remove deleted tracks.
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # logger.info IDs in the database and number of features per ID
        # logger.info("Global database contents:")
        # for track_id, data in self.global_database.items():
            # logger.info(f"Track ID: {track_id}, Features: {len(data['features'])}, Class: {data['class_name']}")

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    # def update(self, detections, green_box=None):
    #     """Perform measurement update and track management."""
    #     matches, unmatched_tracks, unmatched_detections = self._match(detections)

    #     print(f"In tracker.py: {detections}")

    #     # Update track set for matched detections
    #     for track_idx, detection_idx in matches:
    #         self.tracks[track_idx].update(self.kf, detections[detection_idx], self.global_database)
    #         print(f"Matched detection to track_id {self.tracks[track_idx].track_id}.")

    #     # Mark unmatched tracks as missed
    #     for track_idx in unmatched_tracks:
    #         print(f"Marking track_id {self.tracks[track_idx].track_id} as missed.")
    #         self.tracks[track_idx].mark_missed()

    #     # Handle unmatched detections
    #     for detection_idx in unmatched_detections:
    #         detection = detections[detection_idx]
            
    #         # Check if detection is completely inside green box with gap
    #         is_inside_box = False
    #         if green_box is not None:
    #             detection_tlbr = detection.to_tlbr()
    #             is_inside_box = self._is_completely_inside_with_gap(detection_tlbr, green_box)
    #             if is_inside_box:
    #                 print("Detection is completely inside green box with gap")

    #         # Match with database, passing the is_inside_box flag
    #         matched_track_id = self._match_with_global_database(detection.feature, is_inside_box)

    #         if matched_track_id is not None:
    #             # Re-identify the track using the database
    #             mean, covariance = self.kf.initiate(detection.to_xyah())
    #             class_name = detection.get_class()
    #             self.tracks.append(Track(
    #                 mean, covariance, matched_track_id, self.n_init, self.max_age,
    #                 detection.feature, class_name))

    #             # Append features from the database
    #             self.tracks[-1].features = self.global_database[matched_track_id]["features"]
    #             print(f"Re-identified track_id {matched_track_id} from database with {len(self.tracks[-1].features)} features.")
    #         else:
    #             # Assign new track_id
    #             mean, covariance = self.kf.initiate(detection.to_xyah())
    #             class_name = detection.get_class()
    #             self.tracks.append(Track(
    #                 mean, covariance, self._next_id, self.n_init, self.max_age,
    #                 detection.feature, class_name))
    #             self._next_id += 1
    #             print(f"Assigned new track_id {self._next_id - 1}.")

    #     # Remove deleted tracks
    #     self.tracks = [t for t in self.tracks if not t.is_deleted()]

    #     # Print database contents
    #     print("Global database contents:")
    #     for track_id, data in self.global_database.items():
    #         print(f"Track ID: {track_id}, Features: {len(data['features'])}, Class: {data['class_name']}")

    #     # Update distance metric
    #     active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
    #     features, targets = [], []
    #     for track in self.tracks:
    #         if not track.is_confirmed():
    #             continue
    #         features += track.features
    #         targets += [track.track_id for _ in track.features]
    #         track.features = []
    #     self.metric.partial_fit(
    #         np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        """Match detections to tracks."""
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            logger.info(f"Cost matrix is {cost_matrix}")
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # logger.info("MAT")
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        # Combine matches and unmatched lists
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        unmatched_detections = list(set(unmatched_detections))

        return matches, unmatched_tracks, unmatched_detections

    # def _match_with_global_database(self, detection_feature, is_inside_box=False):
    #     """
    #     Match a detection with tracks in the global database.
    #     Returns the track_id with the smallest distance, ignoring threshold if inside box.
    #     """
    #     min_distance = float('inf')
    #     matched_track_id = None
        
    #     # Get currently active track IDs
    #     active_track_ids = [track.track_id for track in self.tracks if track.is_confirmed()]

    #     # If database is empty and detection is inside box, return None to get new ID
    #     if not self.global_database:
    #         print("Database empty, will assign new ID")
    #         return None

    #     for track_id, data in self.global_database.items():
    #         if track_id in active_track_ids:
    #             print(f"Skipping track_id {track_id} because it is currently active.")
    #             continue

    #         valid_distances = []

    #         for db_feature in data["features"]:
    #             detection_feature = np.asarray(detection_feature).flatten()
    #             db_feature = np.asarray(db_feature).flatten()

    #             dot_product = np.dot(detection_feature, db_feature)
    #             norm_detection = np.linalg.norm(detection_feature)
    #             norm_db = np.linalg.norm(db_feature)
    #             cosine_similarity = dot_product / (norm_detection * norm_db)
    #             cosine_distance = 1 - cosine_similarity

    #             # For inside box case, consider all distances
    #             # For normal case, only consider distances < 0.3
    #             if is_inside_box or cosine_distance < 0.4:
    #                 valid_distances.append(cosine_distance)

    #         if valid_distances:
    #             smallest_distance_for_track = min(valid_distances)
    #             print(f"Track ID {track_id} has distances: {valid_distances}. Smallest: {smallest_distance_for_track}")

    #             if smallest_distance_for_track < min_distance:
    #                 min_distance = smallest_distance_for_track
    #                 matched_track_id = track_id
    #         else:
    #             print(f"No valid distances found for track_id {track_id}")

    #     if matched_track_id is not None:
    #         print(f"Matched track_id {matched_track_id} with distance {min_distance}")
    #     else:
    #         print("No match found in database. Will assign new ID.")

    #     return matched_track_id

    def _match_with_global_database(self, detection_feature, detection_bbox, center_bbox = (0, 270, 1152, 800)):
        """
        Match a detection with tracks in the global database.
        Returns the track_id with the smallest distance if it's below the threshold (0.1).
        Only checks tracks that are not currently active in the scene.
        """
        def is_inside_store_center(bbox, center_bbox):
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            return (center_bbox[0] <= cx <= center_bbox[2]) and (center_bbox[1] <= cy <= center_bbox[3])

        min_distance = float('inf')  # Initialize with a large value
        matched_track_id = None  # Track ID of the best match
        default_threshold = 0.4
        increased_threshold = 0.5
        if is_inside_store_center(detection_bbox, center_bbox):
            threshold = increased_threshold
            logger.info(f"Detection bbox {detection_bbox} is in the center; using increased threshold {threshold}.")
        else:
            threshold = default_threshold
            logger.info(f"Detection bbox {detection_bbox} is not in the center; using default threshold {threshold}.")

        # store_center_x1, store_center_y1, store_center_x2, store_center_y2 = (0, 270, 1152, 800)

        # Get list of track IDs that are both confirmed AND currently visible in the scene
        visible_confirmed_track_ids = [
            track.track_id for track in self.tracks 
            if track.is_confirmed() and track.time_since_update == 0
        ]
        # Get the list of currently active track IDs
        # active_track_ids = [track.track_id for track in self.tracks if track.is_confirmed()]

        # Iterate through all track_ids in the global database
        for track_id, data in self.global_database.items():
            # Skip if the track_id is currently active
            if track_id in visible_confirmed_track_ids:
                logger.info(f"Skipping track_id {track_id} because it is confirmed and currently visible in scene")
                continue

            # Initialize a list to store distances less than 0.1 for this track_id
            valid_distances = []

            # Iterate through all features for this track_id
            for db_feature in data["features"]:
                # Ensure the features are in the correct format (1D arrays)
                detection_feature = np.asarray(detection_feature).flatten()
                db_feature = np.asarray(db_feature).flatten()

                # Calculate cosine distance between detection feature and database feature
                dot_product = np.dot(detection_feature, db_feature)
                norm_detection = np.linalg.norm(detection_feature)
                norm_db = np.linalg.norm(db_feature)
                cosine_similarity = dot_product / (norm_detection * norm_db)
                cosine_distance = 1 - cosine_similarity

                # Only consider distances less than 0.3
                if cosine_distance < threshold:
                    valid_distances.append(cosine_distance)

            # If there are valid distances for this track_id, find the smallest one
            if valid_distances:
                smallest_distance_for_track = min(valid_distances)
                logger.info(f"Track ID {track_id} has valid distances: {valid_distances}. Smallest: {smallest_distance_for_track}")

                # Check if this track_id has the smallest overall distance
                if smallest_distance_for_track < min_distance:
                    min_distance = smallest_distance_for_track
                    matched_track_id = track_id
            else:
                logger.info(f"Smallest distance with {track_id} is more than 0.5, so we are skipping it.")

        if matched_track_id is not None:
            logger.info(f"Matched track_id {matched_track_id} with distance {min_distance}.")
        else:
            logger.info("No match found in global database. Assigning new track_id.")

        return matched_track_id


# import numpy as np
# import os
# import pickle
# from . import kalman_filter
# from . import linear_assignment
# from . import iou_matching
# from .track import Track

# class Tracker:
#     """
#     This is the multi-target tracker.
#     """

#     def __init__(self, metric, max_iou_distance=0.7, max_age=60, n_init=3, matching_threshold=0.5):
#         self.metric = metric
#         self.max_iou_distance = max_iou_distance
#         self.max_age = max_age
#         self.n_init = n_init
#         self.matching_threshold = matching_threshold  # Set the matching threshold

#         self.kf = kalman_filter.KalmanFilter()
#         self.tracks = []
#         self._next_id = 1

#         # Load or create the global database
#         self.global_database_file = "global_database.pkl"
#         if os.path.exists(self.global_database_file):
#             with open(self.global_database_file, "rb") as f:
#                 self.global_database = pickle.load(f)
#             print(f"Loaded global database from {self.global_database_file}.")
#         else:
#             self.global_database = {}
#             print(f"Created new global database.")

#     def save_global_database(self):
#         """Save the global database to a file."""
#         with open(self.global_database_file, "wb") as f:
#             pickle.dump(self.global_database, f)
#         print(f"Saved global database to {self.global_database_file}.")

#     def predict(self):
#         """Propagate track state distributions one time step forward."""
#         for track in self.tracks:
#             track.predict(self.kf)

#     def _is_completely_inside_with_gap(self, detection_tlbr, green_box, min_gap=5):
#         """
#         Check if detection is completely inside green box with a minimum gap.
#         Args:
#             detection_tlbr: [x1, y1, x2, y2] of detection box
#             green_box: [x1, y1, x2, y2] of green box
#             min_gap: minimum pixel gap required between boxes
#         Returns:
#             bool: True if detection is inside with minimum gap
#         """
#         return (detection_tlbr[0] >= green_box[0] + min_gap and
#                 detection_tlbr[1] >= green_box[1] + min_gap and
#                 detection_tlbr[2] <= green_box[2] - min_gap and
#                 detection_tlbr[3] <= green_box[3] - min_gap)

#     def _match_with_global_database(self, detection_feature, is_inside_box=False):
#         """
#         Match a detection with tracks in the global database.
#         Returns the track_id with the smallest distance, ignoring threshold if inside box.
#         """
#         min_distance = float('inf')
#         matched_track_id = None
        
#         # Get currently active track IDs
#         active_track_ids = [track.track_id for track in self.tracks if track.is_confirmed()]

#         # If database is empty and detection is inside box, return None to get new ID
#         if not self.global_database:
#             print("Database empty, will assign new ID")
#             return None

#         for track_id, data in self.global_database.items():
#             if track_id in active_track_ids:
#                 print(f"Skipping track_id {track_id} because it is currently active.")
#                 continue

#             valid_distances = []

#             for db_feature in data["features"]:
#                 detection_feature = np.asarray(detection_feature).flatten()
#                 db_feature = np.asarray(db_feature).flatten()

#                 dot_product = np.dot(detection_feature, db_feature)
#                 norm_detection = np.linalg.norm(detection_feature)
#                 norm_db = np.linalg.norm(db_feature)
#                 cosine_similarity = dot_product / (norm_detection * norm_db)
#                 cosine_distance = 1 - cosine_similarity

#                 # For inside box case, consider all distances
#                 # For normal case, only consider distances < 0.3
#                 if is_inside_box or cosine_distance < 0.4:
#                     valid_distances.append(cosine_distance)

#             if valid_distances:
#                 smallest_distance_for_track = min(valid_distances)
#                 print(f"Track ID {track_id} has distances: {valid_distances}. Smallest: {smallest_distance_for_track}")

#                 if smallest_distance_for_track < min_distance:
#                     min_distance = smallest_distance_for_track
#                     matched_track_id = track_id
#             else:
#                 print(f"No valid distances found for track_id {track_id}")

#         if matched_track_id is not None:
#             print(f"Matched track_id {matched_track_id} with distance {min_distance}")
#         else:
#             print("No match found in database. Will assign new ID.")

#         return matched_track_id

#     def update(self, detections, green_box=None):
#         """Perform measurement update and track management."""
#         matches, unmatched_tracks, unmatched_detections = self._match(detections)

#         print(f"In tracker.py: {detections}")

#         # Update track set for matched detections
#         for track_idx, detection_idx in matches:
#             self.tracks[track_idx].update(self.kf, detections[detection_idx], self.global_database)
#             print(f"Matched detection to track_id {self.tracks[track_idx].track_id}.")

#         # Mark unmatched tracks as missed
#         for track_idx in unmatched_tracks:
#             print(f"Marking track_id {self.tracks[track_idx].track_id} as missed.")
#             self.tracks[track_idx].mark_missed()

#         # Handle unmatched detections
#         for detection_idx in unmatched_detections:
#             detection = detections[detection_idx]
            
#             # Check if detection is completely inside green box with gap
#             is_inside_box = False
#             if green_box is not None:
#                 detection_tlbr = detection.to_tlbr()
#                 is_inside_box = self._is_completely_inside_with_gap(detection_tlbr, green_box)
#                 if is_inside_box:
#                     print("Detection is completely inside green box with gap")

#             # Match with database, passing the is_inside_box flag
#             matched_track_id = self._match_with_global_database(detection.feature, is_inside_box)

#             if matched_track_id is not None:
#                 # Re-identify the track using the database
#                 mean, covariance = self.kf.initiate(detection.to_xyah())
#                 class_name = detection.get_class()
#                 self.tracks.append(Track(
#                     mean, covariance, matched_track_id, self.n_init, self.max_age,
#                     detection.feature, class_name))

#                 # Append features from the database
#                 self.tracks[-1].features = self.global_database[matched_track_id]["features"]
#                 print(f"Re-identified track_id {matched_track_id} from database with {len(self.tracks[-1].features)} features.")
#             else:
#                 # Assign new track_id
#                 mean, covariance = self.kf.initiate(detection.to_xyah())
#                 class_name = detection.get_class()
#                 self.tracks.append(Track(
#                     mean, covariance, self._next_id, self.n_init, self.max_age,
#                     detection.feature, class_name))
#                 self._next_id += 1
#                 print(f"Assigned new track_id {self._next_id - 1}.")

#         # Remove deleted tracks
#         self.tracks = [t for t in self.tracks if not t.is_deleted()]

#         # Print database contents
#         print("Global database contents:")
#         for track_id, data in self.global_database.items():
#             print(f"Track ID: {track_id}, Features: {len(data['features'])}, Class: {data['class_name']}")

#         # Update distance metric
#         active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
#         features, targets = [], []
#         for track in self.tracks:
#             if not track.is_confirmed():
#                 continue
#             features += track.features
#             targets += [track.track_id for _ in track.features]
#             track.features = []
#         self.metric.partial_fit(
#             np.asarray(features), np.asarray(targets), active_targets)

#     def _match(self, detections):
#         """Match detections to tracks."""
#         def gated_metric(tracks, dets, track_indices, detection_indices):
#             features = np.array([dets[i].feature for i in detection_indices])
#             targets = np.array([tracks[i].track_id for i in track_indices])
#             cost_matrix = self.metric.distance(features, targets)
#             cost_matrix = linear_assignment.gate_cost_matrix(
#                 self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)
#             return cost_matrix

#         # Split track set into confirmed and unconfirmed tracks
#         confirmed_tracks = [
#             i for i, t in enumerate(self.tracks) if t.is_confirmed()]
#         unconfirmed_tracks = [
#             i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

#         # Associate confirmed tracks using appearance features
#         matches_a, unmatched_tracks_a, unmatched_detections = \
#             linear_assignment.matching_cascade(
#                 gated_metric, self.metric.matching_threshold, self.max_age,
#                 self.tracks, detections, confirmed_tracks)

#         # Associate remaining tracks together with unconfirmed tracks using IOU
#         iou_track_candidates = unconfirmed_tracks + [
#             k for k in unmatched_tracks_a if
#             self.tracks[k].time_since_update == 1]
#         unmatched_tracks_a = [
#             k for k in unmatched_tracks_a if
#             self.tracks[k].time_since_update != 1]
#         matches_b, unmatched_tracks_b, unmatched_detections = \
#             linear_assignment.min_cost_matching(
#                 iou_matching.iou_cost, self.max_iou_distance, self.tracks,
#                 detections, iou_track_candidates, unmatched_detections)

#         matches = matches_a + matches_b
#         unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
#         unmatched_detections = list(set(unmatched_detections))

#         return matches, unmatched_tracks, unmatched_detections