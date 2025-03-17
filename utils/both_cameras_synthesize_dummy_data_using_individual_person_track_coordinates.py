import csv
import json
import os
import random
from datetime import datetime, timedelta

# ============================================
# 1. Load trajectories from CSV files for both cameras.
# ============================================

# For Camera 1:
trajectories_cam1 = {}
with open('person_movement.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pid = row['person_id']
        coord = (row['x'], row['y'])  # keep as strings to mirror the JSON sample
        trajectories_cam1.setdefault(pid, []).append(coord)

# For Camera 2:
trajectories_cam2 = {}
with open('person_movement_cam2.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pid = row['person_id']
        coord = (row['x'], row['y'])
        trajectories_cam2.setdefault(pid, []).append(coord)

# ============================================
# 2. Define per–hour time–spent distributions for each camera.
# ============================================
# Distribution for Camera 1 (in seconds)
distribution_cam1 = {
    "2": 720,
    "1": 510,
    "4": 506,
    "14": 139,
    "29": 137,
    "20": 281,
    "23": 56,
    "12": 174,
    "18": 118,
    "6": 1,
    "27": 171,
    "9": 23,
    "19": 78,
    "30": 41,
    "15": 17,
    "16": 2,
    "21": 3,
    "22": 35,
    "25": 9,
    "26": 1
}

# Distribution for Camera 2 (in seconds)
distribution_cam2 = {
    "12": 77,
    "22": "660",
    "27": 227,
    "29": 227,
    "1": 540,
    "14": 329,
    "2": 498,
    "3": 10,
    "20": 1153,
    "4": 510,
    "18": 1277,
    "5": 1,
    "7": 1,
    "19": 36,
    "9": 255,
    "10": 150,
    "13": 313,
    "16": 1,
    "17": 5,
    "21": 94,
    "24": 56,
    "30": 252
}

# ============================================
# 3. Determine overlapping persons (those that appear in both distributions).
# ============================================
overlapping_set = set(distribution_cam1.keys()).intersection(set(distribution_cam2.keys()))
# overlapping_set = {"1", "2", "4", "9", "12", "14", "16", "18", "19", "20", "21", "22", "27", "29", "30"}
print("Overlapping person IDs:", overlapping_set)

# ============================================
# 4. Synthetic generation parameters.
# ============================================
duplicate_probability = 0.1  # Chance to include an extra instance (duplicate) in camera 1

# For camera 1, we use time-dependent inclusion probability:
#   10-2 PM: 0.7, 2-6 PM: 0.3, 6-10 PM: 0.7.
# (Camera 2 tracks will be generated automatically for overlapping persons.)
# ============================================

# ============================================
# 5. Set time range and output folder.
# ============================================
start_date = datetime(2025, 1, 1)
days = 7
hours_per_day = 12  # from 10:00 to 21:59 (i.e. 10AM to 10PM)
start_hour = 10     # starting at 10:00 AM

base_folder = "bothcamerassyntheticdata"
os.makedirs(base_folder, exist_ok=True)

# ============================================
# 6. Global person ID counter (unique across both cameras and all files)
# ============================================
global_person_id_counter = 1

# ============================================
# 7. Generate synthetic JSON files for both cameras.
# ============================================
for day in range(days):
    current_date = start_date + timedelta(days=day)
    day_folder = os.path.join(base_folder, current_date.strftime('%Y-%m-%d'))
    os.makedirs(day_folder, exist_ok=True)
    
    for hour in range(hours_per_day):
        current_hour = start_hour + hour
        base_datetime = datetime(current_date.year, current_date.month, current_date.day, current_hour, 0, 0)
        
        # Set inclusion probability for camera 1 based on current hour.
        if 10 <= current_hour < 14:
            include_probability = 0.7
        elif 14 <= current_hour < 18:
            include_probability = 0.3
        elif 18 <= current_hour < 22:
            include_probability = 0.7
        else:
            include_probability = 0.7
        
        # Create unique video IDs for each camera.
        video_id_cam1 = f"VID-{current_date.strftime('%Y%m%d')}_{current_hour:02d}_CAM1"
        video_id_cam2 = f"VID-{current_date.strftime('%Y%m%d')}_{current_hour:02d}_CAM2"
        
        # -------------------------------------
        # (A) Generate active tracks for Camera 1.
        # -------------------------------------
        active_tracks_cam1 = []
        for pid, duration in distribution_cam1.items():
            # Check if there is enough trajectory data for camera 1.
            if pid not in trajectories_cam1 or len(trajectories_cam1[pid]) < duration:
                continue
            
            # Decide (with time-dependent probability) whether to include this track.
            if random.random() < include_probability:
                max_start = 3600 - duration
                start_second = random.randint(0, max_start)
                active_tracks_cam1.append({
                    "orig_person_id": pid,
                    "duration": duration,
                    "start_second": start_second,
                    "trajectory": trajectories_cam1[pid][:duration]
                })
                
                # Optionally, add a duplicate track.
                if random.random() < duplicate_probability:
                    dup_start_second = random.randint(0, max_start)
                    active_tracks_cam1.append({
                        "orig_person_id": pid,
                        "duration": duration,
                        "start_second": dup_start_second,
                        "trajectory": trajectories_cam1[pid][:duration]
                    })
        
        # Assign a new unique global integer ID to each camera 1 track.
        for track in active_tracks_cam1:
            track["unique_id"] = global_person_id_counter
            global_person_id_counter += 1
        
        # -------------------------------------
        # (B) Generate corresponding active tracks for Camera 2.
        # Only include tracks for overlapping persons that are available in cam2.
        # For each qualifying track from cam1, generate a corresponding cam2 track.
        # -------------------------------------
        active_tracks_cam2 = []
        for track in active_tracks_cam1:
            pid = track["orig_person_id"]
            if pid in overlapping_set and pid in trajectories_cam2 and len(trajectories_cam2[pid]) >= distribution_cam2[pid]:
                duration_cam2 = distribution_cam2[pid]
                max_start_cam2 = 3600 - duration_cam2
                start_second_cam2 = random.randint(0, max_start_cam2)
                # Create a cam2 track using the same unique_id as in cam1.
                active_tracks_cam2.append({
                    "unique_id": track["unique_id"],
                    "orig_person_id": pid,
                    "duration": duration_cam2,
                    "start_second": start_second_cam2,
                    "trajectory": trajectories_cam2[pid][:duration_cam2]
                })
        
        # -------------------------------------
        # (C) Build detection entries for each camera (one per second for 3600 seconds).
        # -------------------------------------
        def build_detections(active_tracks, base_dt, cam_id):
            detections = []
            for sec in range(3600):
                current_time = base_dt + timedelta(seconds=sec)
                next_time = current_time + timedelta(seconds=1)
                from_datetime = current_time.strftime('%Y-%m-%dT%H:%M:%S') + ".602Z"
                to_datetime = next_time.strftime('%Y-%m-%dT%H:%M:%S') + ".602Z"
                
                persons_in_frame = []
                for track in active_tracks:
                    if track["start_second"] <= sec < track["start_second"] + track["duration"]:
                        offset = sec - track["start_second"]
                        x, y = track["trajectory"][offset]
                        persons_in_frame.append({
                            "id": track["unique_id"],
                            "person_id": track["unique_id"],
                            "coords": {"x": x, "y": y}
                        })
                
                if persons_in_frame:
                    detection_x = persons_in_frame[0]["coords"]["x"]
                    detection_y = persons_in_frame[0]["coords"]["y"]
                else:
                    detection_x, detection_y = "0", "0"
                
                detection_entry = {
                    "camera_id": cam_id,
                    "image_url": "",
                    "is_organised": True,
                    "no_of_people": len(persons_in_frame),
                    "from_datetime": from_datetime,
                    "to_datetime": to_datetime,
                    "visitor_type": "Solo",
                    "x_coord": detection_x,
                    "y_coord": detection_y,
                    "persons": persons_in_frame
                }
                detections.append(detection_entry)
            return detections
        
        detections_cam1 = build_detections(active_tracks_cam1, base_datetime, cam_id=1)
        detections_cam2 = build_detections(active_tracks_cam2, base_datetime, cam_id=2)
        
        # -------------------------------------
        # (D) Build final JSON structures and write files.
        # -------------------------------------
        output_data_cam1 = {
            "video_id": video_id_cam1,
            "detections": detections_cam1
        }
        output_data_cam2 = {
            "video_id": video_id_cam2,
            "detections": detections_cam2
        }
        
        # File names: e.g. "2025-01-01_10_cam1.json" and "2025-01-01_10_cam2.json"
        filename_cam1 = f"{current_date.strftime('%Y-%m-%d')}_{current_hour:02d}_cam1.json"
        filename_cam2 = f"{current_date.strftime('%Y-%m-%d')}_{current_hour:02d}_cam2.json"
        
        file_path_cam1 = os.path.join(day_folder, filename_cam1)
        file_path_cam2 = os.path.join(day_folder, filename_cam2)
        
        with open(file_path_cam1, 'w') as outfile1:
            json.dump(output_data_cam1, outfile1, indent=4)
        with open(file_path_cam2, 'w') as outfile2:
            json.dump(output_data_cam2, outfile2, indent=4)
        
        print(f"Generated files: {file_path_cam1} (video_id: {video_id_cam1}) and {file_path_cam2} (video_id: {video_id_cam2})")
