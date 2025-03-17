# import csv
# import json
# import os
# import random
# from datetime import datetime, timedelta

# # -------------------------------
# # 1. Load the CSV trajectories.
# # -------------------------------
# # Assumes CSV headers: person_id,frame,x,y,movement_x,movement_y
# # Each person's trajectory is stored as a list of (x, y) coordinates.
# trajectories = {}
# with open('person_movement.csv', newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         pid = row['person_id']
#         # Keep coordinates as strings to mirror JSON sample
#         coord = (row['x'], row['y'])
#         trajectories.setdefault(pid, []).append(coord)

# # (Optional: sort trajectories by frame if not already sorted)
# # for pid in trajectories:
# #     trajectories[pid].sort(key=lambda r: int(r['frame']))

# # -------------------------------
# # 2. Define the distribution for time spent (in seconds) per hour.
# # -------------------------------
# # These values represent the continuous seconds a person is in the frame in an hour.
# distribution = {
#     "2": 720,
#     "1": 510,
#     "4": 506,
#     "14": 139,
#     "29": 137,
#     "20": 281,
#     "23": 56,
#     "12": 174,
#     "18": 118,
#     "6": 1,
#     "27": 171,
#     "9": 23,
#     "19": 78,
#     "30": 41,
#     "15": 17,
#     "16": 2,
#     "21": 3,
#     "22": 35,
#     "25": 9,
#     "26": 1
# }

# # -------------------------------
# # 3. Synthetic generation parameters.
# # -------------------------------
# include_probability = 0.7    # Chance that a person’s track is included in a given hour.
# duplicate_probability = 0.1  # Chance to add a duplicate track (with a new person_id) in the same hour.

# # -------------------------------
# # 4. Set the time range and output folder.
# # -------------------------------
# # We generate synthetic logs for 7 days (Jan 1 to Jan 7) for 12 hours each day (from 10AM to 10PM).
# start_date = datetime(2025, 1, 1)
# days = 7
# hours_per_day = 12
# start_hour = 10  # starting at 10:00 AM

# # Base folder to store synthetic JSON files.
# base_folder = "syntheticdata"

# # Ensure the base folder exists.
# os.makedirs(base_folder, exist_ok=True)

# # -------------------------------
# # 5. Generate synthetic JSON files.
# # -------------------------------
# for day in range(days):
#     current_date = start_date + timedelta(days=day)
#     # Create a subfolder for the current day (e.g., "syntheticdata/2025-01-01")
#     day_folder = os.path.join(base_folder, current_date.strftime('%Y-%m-%d'))
#     os.makedirs(day_folder, exist_ok=True)
    
#     for hour in range(hours_per_day):
#         # Calculate the base datetime for the current hour.
#         current_hour = start_hour + hour
#         base_datetime = datetime(current_date.year, current_date.month, current_date.day, current_hour, 0, 0)
        
#         # List to hold the active tracks for this hour.
#         # Each active track is a dictionary with:
#         #   "person_id": identifier,
#         #   "duration": seconds (from distribution),
#         #   "start_second": offset in seconds when the track starts,
#         #   "trajectory": list of (x, y) coordinates for each second.
#         active_tracks = []
#         for pid, duration in distribution.items():
#             # Only use the track if the CSV trajectory has at least 'duration' frames.
#             if pid not in trajectories or len(trajectories[pid]) < duration:
#                 continue

#             # Decide randomly whether to include this person's track this hour.
#             if random.random() < include_probability:
#                 max_start = 3600 - duration  # ensure block fits within the hour
#                 start_second = random.randint(0, max_start)
#                 active_tracks.append({
#                     "person_id": pid,
#                     "duration": duration,
#                     "start_second": start_second,
#                     "trajectory": trajectories[pid][:duration]  # use the first 'duration' frames
#                 })
                
#                 # Optionally, add a duplicate track (with a different id).
#                 if random.random() < duplicate_probability:
#                     dup_pid = f"{pid}_dup"
#                     dup_start_second = random.randint(0, max_start)
#                     active_tracks.append({
#                         "person_id": dup_pid,
#                         "duration": duration,
#                         "start_second": dup_start_second,
#                         "trajectory": trajectories[pid][:duration]
#                     })
        
#         # Build detection entries for each second in the hour.
#         detections = []
#         for sec in range(3600):
#             # Build the timestamps with the fixed milliseconds (.602Z).
#             current_time = base_datetime + timedelta(seconds=sec)
#             next_time = current_time + timedelta(seconds=1)
#             from_datetime = current_time.strftime('%Y-%m-%dT%H:%M:%S') + ".602Z"
#             to_datetime = next_time.strftime('%Y-%m-%dT%H:%M:%S') + ".602Z"
            
#             # Determine which tracks are active at this second.
#             persons_in_frame = []
#             for track in active_tracks:
#                 # A track is active if the current second is within its continuous block.
#                 if track["start_second"] <= sec < track["start_second"] + track["duration"]:
#                     offset = sec - track["start_second"]
#                     x, y = track["trajectory"][offset]
#                     persons_in_frame.append({
#                         "id": track["person_id"],
#                         "person_id": track["person_id"],
#                         "coords": {
#                             "x": x,
#                             "y": y
#                         }
#                     })
            
#             # Set detection-level coordinates:
#             # If there is at least one person, use the coordinates of the first person.
#             if persons_in_frame:
#                 detection_x = persons_in_frame[0]["coords"]["x"]
#                 detection_y = persons_in_frame[0]["coords"]["y"]
#             else:
#                 detection_x = "0"
#                 detection_y = "0"
            
#             detection_entry = {
#                 "camera_id": 1,
#                 "image_url": "",
#                 "is_organised": True,
#                 "no_of_people": len(persons_in_frame),
#                 "from_datetime": from_datetime,
#                 "to_datetime": to_datetime,
#                 "visitor_type": "Solo",
#                 "x_coord": detection_x,
#                 "y_coord": detection_y,
#                 "persons": persons_in_frame
#             }
#             detections.append(detection_entry)
        
#         # Build the final JSON structure.
#         output_data = {"detections": detections}
        
#         # Name the file based on the date and hour (e.g., "2025-01-01_10.json")
#         filename = f"{current_date.strftime('%Y-%m-%d')}_{current_hour:02d}.json"
#         file_path = os.path.join(day_folder, filename)
#         with open(file_path, 'w') as outfile:
#             json.dump(output_data, outfile, indent=4)
        
#         print(f"Generated file: {file_path}")
import csv
import json
import os
import random
from datetime import datetime, timedelta

# -------------------------------
# 1. Load the CSV trajectories.
# -------------------------------
# Assumes CSV headers: person_id,frame,x,y,movement_x,movement_y
# Each person's trajectory is stored as a list of (x, y) coordinate pairs (kept as strings).
trajectories = {}
with open('person_movement.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pid = row['person_id']
        coord = (row['x'], row['y'])
        trajectories.setdefault(pid, []).append(coord)

# -------------------------------
# 2. Define the distribution for time spent (in seconds) per hour.
# -------------------------------
# These values represent the continuous number of seconds a person appears in the frame.
distribution = {
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

# -------------------------------
# 3. Synthetic generation parameters.
# -------------------------------
# duplicate_probability remains constant.
duplicate_probability = 0.1  # 10% chance to include an extra instance of a track

# -------------------------------
# 4. Set the time range and output folder.
# -------------------------------
# We generate synthetic logs for 7 days (Jan 1 to Jan 7) for a 12-hour period each day (from 10AM to 10PM).
start_date = datetime(2025, 1, 1)
days = 7
hours_per_day = 12
start_hour = 10  # starting at 10:00 AM

# Base folder to store synthetic JSON files.
base_folder = "syntheticdata"
os.makedirs(base_folder, exist_ok=True)

# -------------------------------
# 5. Global person ID counter
# -------------------------------
global_person_id_counter = 1

# -------------------------------
# 6. Generate synthetic JSON files.
# -------------------------------
for day in range(days):
    current_date = start_date + timedelta(days=day)
    # Create a subfolder for the current day (e.g., "syntheticdata/2025-01-01")
    day_folder = os.path.join(base_folder, current_date.strftime('%Y-%m-%d'))
    os.makedirs(day_folder, exist_ok=True)
    
    for hour in range(hours_per_day):
        current_hour = start_hour + hour
        
        # Set include_probability based on the current hour:
        #   - 10-2 PM (10,11,12,13): high probability (0.7)
        #   - 2-6 PM (14,15,16,17): lower probability (0.3)
        #   - 6-10 PM (18,19,20,21): high probability (0.7)
        if 10 <= current_hour < 14:
            include_probability = 0.7
        elif 14 <= current_hour < 18:
            include_probability = 0.3
        elif 18 <= current_hour < 22:
            include_probability = 0.7
        else:
            include_probability = 0.7  # fallback
        
        # Base datetime for the current hour.
        base_datetime = datetime(current_date.year, current_date.month, current_date.day, current_hour, 0, 0)
        
        # Create a unique video ID for this hour's video.
        unique_video_id = f"VID-{current_date.strftime('%Y%m%d')}_{current_hour:02d}"
        
        # -------------------------------
        # Build active tracks for this hour.
        # -------------------------------
        # Each active track will later be assigned a new unique integer ID that is unique across all JSON files.
        active_tracks = []
        for pid, duration in distribution.items():
            # Only include if there is sufficient trajectory data.
            if pid not in trajectories or len(trajectories[pid]) < duration:
                continue

            # Decide (with time-dependent probability) whether to include this track.
            if random.random() < include_probability:
                max_start = 3600 - duration  # ensure block fits in the hour
                start_second = random.randint(0, max_start)
                active_tracks.append({
                    "orig_person_id": pid,  # original id from CSV/distribution (for reference)
                    "duration": duration,
                    "start_second": start_second,
                    "trajectory": trajectories[pid][:duration]
                })
                
                # Optionally add a duplicate track based on duplicate_probability.
                if random.random() < duplicate_probability:
                    dup_start_second = random.randint(0, max_start)
                    active_tracks.append({
                        "orig_person_id": pid,
                        "duration": duration,
                        "start_second": dup_start_second,
                        "trajectory": trajectories[pid][:duration]
                    })
        
        # Now, assign each track a new unique integer ID using the global counter.
        for track in active_tracks:
            track["unique_id"] = global_person_id_counter
            global_person_id_counter += 1
        
        # -------------------------------
        # Build detection entries (one per second for 3600 seconds).
        # -------------------------------
        detections = []
        for sec in range(3600):
            # Build timestamps with fixed milliseconds (.602Z).
            current_time = base_datetime + timedelta(seconds=sec)
            next_time = current_time + timedelta(seconds=1)
            from_datetime = current_time.strftime('%Y-%m-%dT%H:%M:%S') + ".602Z"
            to_datetime = next_time.strftime('%Y-%m-%dT%H:%M:%S') + ".602Z"
            
            # Determine which tracks are active at this second.
            persons_in_frame = []
            for track in active_tracks:
                if track["start_second"] <= sec < track["start_second"] + track["duration"]:
                    offset = sec - track["start_second"]
                    x, y = track["trajectory"][offset]
                    persons_in_frame.append({
                        "id": track["unique_id"],       # global unique integer ID
                        "person_id": track["unique_id"],  # same here
                        "coords": {
                            "x": x,
                            "y": y
                        }
                    })
            
            # Set detection-level coordinates:
            # If there is at least one person, use the coordinates of the first person.
            if persons_in_frame:
                detection_x = persons_in_frame[0]["coords"]["x"]
                detection_y = persons_in_frame[0]["coords"]["y"]
            else:
                detection_x = "0"
                detection_y = "0"
            
            detection_entry = {
                "camera_id": 1,
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
        
        # Build the final JSON structure, including the video ID.
        output_data = {
            "video_id": unique_video_id,
            "detections": detections
        }
        
        # Save the JSON file in the day folder (e.g., "2025-01-01_10.json").
        filename = f"{current_date.strftime('%Y-%m-%d')}_{current_hour:02d}.json"
        file_path = os.path.join(day_folder, filename)
        with open(file_path, 'w') as outfile:
            json.dump(output_data, outfile, indent=4)
        
        print(f"Generated file: {file_path} with video_id: {unique_video_id}")

