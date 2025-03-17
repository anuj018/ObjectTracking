import json
import requests
from datetime import datetime

# Load JSON data from a file
with open("demo_video_snippet.json", "r") as file:
    data = json.load(file)

# API endpoint
url = "http://dev-api.xperie.nz:8000/detection"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

# Extracting first and last timestamp
from_datetime = data[0]["current_time"]
to_datetime = data[-1]["current_time"]

# Collect unique person IDs and all coordinate entries
person_entries = []
person_ids = set()
all_x_coords = []
all_y_coords = []

for entry in data:
    for person in entry["ENTITY_COORDINATES"]:
        person_entries.append({
            "id": person["person_id"],  # ID and person_id are the same
            "person_id": person["person_id"],
            "coords": {
                "x": str(person["x_coord"]),
                "y": str(person["y_coord"])
            }
        })
        person_ids.add(person["person_id"])
        all_x_coords.append(str(person["x_coord"]))
        all_y_coords.append(str(person["y_coord"]))

# Construct final payload
formatted_data = {
    "camera_id": data[0]["camera_id"],
    "image_url": "",  # Fixed empty string
    "is_organised": data[0]["is_organised"],
    "no_of_people": len(person_ids),
    "from_datetime": from_datetime,
    "to_datetime": to_datetime,
    "visitor_type": "solo",
    "x_coord": ",".join(all_x_coords),  # Concatenating all x coordinates
    "y_coord": ",".join(all_y_coords),  # Concatenating all y coordinates
    "persons": person_entries
}

# Send PUT request
response = requests.put(url, headers=headers, json=formatted_data)

# Print response
print(f"Response: {response.status_code} - {response.text}")
