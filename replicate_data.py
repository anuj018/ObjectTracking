import os
import json
import copy
import random
from datetime import datetime, timedelta

# ----------------------------
# Configuration
# ----------------------------
TEMPLATE_FILE = "/home/azureuser/workspace/Genfied/tracking_results_video1long_25fps_processed.json"  # Template JSON file with a top-level "detections" key
START_DATE_STR = "2025-01-01"      # Starting date (adjust as needed)
START_DATE = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
NUM_DAYS = 30                    # Generate files for 30 days
START_HOUR = 10                  # Files from 10 AM
END_HOUR = 22                    # Up to (but not including) 22, i.e. 10AM to 9PM slots (12 files)

# ----------------------------
# Helper Functions
# ----------------------------
def format_hour_label(hour):
    """
    Converts a 24-hour integer (e.g., 10, 11, ... 21) to a label like "10AM", "11AM", ..., "10PM".
    """
    if hour < 12:
        return f"{hour}AM"
    elif hour == 12:
        return "12PM"
    else:
        return f"{hour-12}PM"

def adjust_coordinates(detection):
    """
    Adjusts the coordinates in a detection entry by applying a random offset (between -10 and 10).

    - For detections with a non-empty "persons" array:
        * Each person in the list gets an independent random offset.
        * The detection's top-level "x_coord" and "y_coord" are updated to match the first person's new coordinates.
    - For detections with no persons:
        * If the top-level coordinates are already "0", "0", no offset is applied.
        * Otherwise, a random offset is applied.
    """
    det = copy.deepcopy(detection)
    if det.get("persons") and len(det["persons"]) > 0:
        for idx, person in enumerate(det["persons"]):
            orig_x = int(person["coords"]["x"])
            orig_y = int(person["coords"]["y"])
            offset_x = random.randint(-10, 10)
            offset_y = random.randint(-10, 10)
            new_x = orig_x + offset_x
            new_y = orig_y + offset_y
            person["coords"]["x"] = str(new_x)
            person["coords"]["y"] = str(new_y)
            # Update the top-level coordinates using the first person's new values.
            if idx == 0:
                det["x_coord"] = str(new_x)
                det["y_coord"] = str(new_y)
    else:
        # No persons present.
        orig_x = int(det["x_coord"])
        orig_y = int(det["y_coord"])
        # If coordinates are already 0,0 then assume no offset is needed.
        if orig_x == 0 and orig_y == 0:
            pass
        else:
            offset_x = random.randint(-10, 10)
            offset_y = random.randint(-10, 10)
            new_x = orig_x + offset_x
            new_y = orig_y + offset_y
            det["x_coord"] = str(new_x)
            det["y_coord"] = str(new_y)
    return det

def adjust_timestamps(detection, delta):
    """
    Adjusts the detection's timestamps by adding the specified delta.
    The delta is computed as:
        desired_file_start - template_first_detection_from_datetime
    so that the first detection in each file gets re-based to the desired start time (e.g., 10AM).
    The "to_datetime" is set to be exactly one second after "from_datetime".
    """
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
    orig_from_dt = datetime.strptime(detection["from_datetime"], fmt)
    new_from_dt = orig_from_dt + delta
    new_to_dt = new_from_dt + timedelta(seconds=1)
    detection["from_datetime"] = new_from_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    detection["to_datetime"] = new_to_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return detection

# ----------------------------
# Main Processing Loop
# ----------------------------
def generate_synthetic_files():
    # Load the template JSON file.
    with open(TEMPLATE_FILE, "r") as f:
        template_data = json.load(f)

    if "detections" not in template_data:
        raise ValueError("Template JSON must have a top-level 'detections' key.")

    # Define the datetime format.
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
    # Get the base timestamp from the template (first detection's from_datetime).
    template_base_from = datetime.strptime(template_data["detections"][0]["from_datetime"], fmt)

    # Process each day.
    for day in range(NUM_DAYS):
        current_date = START_DATE + timedelta(days=day)
        date_folder = current_date.strftime("%Y-%m-%d")
        folder_path = os.path.join("duplicate_data", date_folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Process each hour (from 10 AM to 9 PM).
        for hour in range(START_HOUR, END_HOUR):
            # For this file, we want the first detection to have a timestamp equal to current_date at [hour]:00:00.000.
            desired_file_start = datetime(current_date.year, current_date.month, current_date.day, hour, 0, 0, 0)
            # Compute the delta needed to re-base the template's timestamp to the desired start.
            delta = desired_file_start - template_base_from

            hour_label = format_hour_label(hour)
            file_name = f"{current_date.strftime('%Y-%m-%d')}_{hour_label}.json"
            file_path = os.path.join("duplicate_data", date_folder, file_name)
            new_detections = []

            # Process each detection entry from the template.
            for detection in template_data["detections"]:
                new_det = copy.deepcopy(detection)
                new_det = adjust_coordinates(new_det)
                new_det = adjust_timestamps(new_det, delta)
                new_detections.append(new_det)

            new_data = {"detections": new_detections}

            # Write the updated data to the JSON file.
            with open(file_path, "w") as out_file:
                json.dump(new_data, out_file, indent=2)
            print(f"Generated file: {file_path}")

if __name__ == "__main__":
    generate_synthetic_files()
