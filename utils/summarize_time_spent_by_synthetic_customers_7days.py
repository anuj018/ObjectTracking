import os
import csv
import json
from collections import defaultdict

# Directories
syntheticdata_dir = "syntheticdata"       # Folder where JSON files (day-wise) are stored
tracks_dir = "syntheticdatatracks"          # Folder to store the intermediate CSV files
os.makedirs(tracks_dir, exist_ok=True)

# File to log the time spent summary (this will have per‑hour info for all 7 days)
summary_filepath = os.path.join(tracks_dir, "time_spent_summary.txt")

with open(summary_filepath, "w") as summary_file:
    # Loop over day folders (e.g., "2025-01-01", "2025-01-02", etc.)
    for day_folder in sorted(os.listdir(syntheticdata_dir)):
        day_path = os.path.join(syntheticdata_dir, day_folder)
        if not os.path.isdir(day_path):
            continue  # skip if not a directory
        
        summary_file.write(f"Day: {day_folder}\n")
        
        # Process each JSON file (each representing one hour) in the day folder
        for json_filename in sorted(os.listdir(day_path)):
            if not json_filename.endswith(".json"):
                continue  # skip non‑JSON files
            
            json_file_path = os.path.join(day_path, json_filename)
            with open(json_file_path, "r") as jf:
                detections_data = json.load(jf)
            
            # -------------------------------
            # Part 1: Process detections to create movement data
            # -------------------------------
            movement_data = []  # List to store each person's movement track in this hour
            
            # Enumerate over each detection entry (each second)
            for frame_number, detection in enumerate(detections_data["detections"], start=1):
                for person in detection["persons"]:
                    person_id = person["person_id"]
                    # Convert coordinate strings to integers
                    x = int(person["coords"]["x"])
                    y = int(person["coords"]["y"])
                    
                    # Look for an existing record for this person
                    existing_person = next((entry for entry in movement_data if entry["person_id"] == person_id), None)
                    
                    if existing_person:
                        # For movement calculation, use the last recorded coordinates
                        prev_x, prev_y = existing_person["coords"]
                        movement_x = abs(x - prev_x)
                        movement_y = abs(y - prev_y)
                        existing_person["movement"].append({
                            "frame": frame_number,
                            "x": x,
                            "y": y,
                            "movement_x": movement_x,
                            "movement_y": movement_y
                        })
                        # Update stored coordinates to the current ones
                        existing_person["coords"] = (x, y)
                    else:
                        movement_data.append({
                            "person_id": person_id,
                            "coords": (x, y),
                            "movement": [{
                                "frame": frame_number,
                                "x": x,
                                "y": y,
                                "movement_x": 0,   # No movement in the first frame
                                "movement_y": 0
                            }]
                        })
            
            # -------------------------------
            # Save the movement data as an intermediate CSV file
            # -------------------------------
            csv_filename = json_filename.replace(".json", ".csv")
            csv_filepath = os.path.join(tracks_dir, csv_filename)
            
            with open(csv_filepath, "w", newline="") as csvfile:
                fieldnames = ["person_id", "frame", "x", "y", "movement_x", "movement_y"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for person in movement_data:
                    for movement in person["movement"]:
                        writer.writerow({
                            "person_id": person["person_id"],
                            "frame": movement["frame"],
                            "x": movement["x"],
                            "y": movement["y"],
                            "movement_x": movement["movement_x"],
                            "movement_y": movement["movement_y"]
                        })
            
            # -------------------------------
            # Part 2: Compute time spent by each person from the CSV file
            # -------------------------------
            time_spent = defaultdict(int)
            with open(csv_filepath, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Each row represents 1 second
                    person_id = row["person_id"]
                    time_spent[person_id] += 1
            
            # Log the time spent info for this JSON (i.e. for this hour) into the summary file
            summary_file.write(f"  File: {json_filename}\n")
            for person_id, secs in time_spent.items():
                summary_file.write(f"    Person {person_id} spent {secs} seconds.\n")
            summary_file.write("\n")
            
            # Optional: also print to console for verification
            print(f"Processed {json_filename} from {day_folder}.")
            for person_id, secs in time_spent.items():
                print(f"  Person {person_id} spent {secs} seconds.")
            print("--------------------------------------------------\n")

print(f"Time spent summary written to {summary_filepath}")
