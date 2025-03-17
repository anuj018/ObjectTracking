import json
from datetime import datetime

def transform_tracking_data(input_file, output_file, frame_interval):
    """
    Reads a JSON file (an array of frame-level detection data) and produces a transformed JSON 
    structured to send to an endpoint. Every `frame_interval`‑th frame is considered to be one detection 
    record. For each such record:
      - from_datetime is taken from the current frame’s current_time,
      - to_datetime is taken from the current frame + frame_interval’s current_time (or the current frame if not available),
      - x_coord and y_coord come from the first detected person (if any) in that frame,
      - persons is an array of person dictionaries reformatted as required.
    
    The output JSON is structured as:
    
    {
      "detections": [
         { ... detection dictionary ... },
         ...
      ]
    }
    """
    # Read the input JSON (assumed to be an array of frame entries)
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    detections = []
    n = len(data)
    
    # Iterate over the input data in steps of frame_interval
    for i in range(0, n, frame_interval):
        current_entry = data[i]
        # Look ahead frame_interval entries; if not available, use the current entry's time
        if i + frame_interval < n:
            next_entry = data[i + frame_interval]
        else:
            next_entry = current_entry

        # Format the datetimes: convert from ISO format and output with millisecond precision + "Z"
        from_dt = datetime.fromisoformat(current_entry["current_time"]).isoformat(timespec='milliseconds') + "Z"
        to_dt = datetime.fromisoformat(next_entry["current_time"]).isoformat(timespec='milliseconds') + "Z"
        
        # Get the x_coord and y_coord from the first detected person (if any); otherwise default to "0"
        if current_entry["ENTITY_COORDINATES"]:
            x_coord = str(current_entry["ENTITY_COORDINATES"][0]["x_coord"])
            y_coord = str(current_entry["ENTITY_COORDINATES"][0]["y_coord"])
        else:
            x_coord = "0"
            y_coord = "0"
        
        # Build the persons list (if any)
        persons = []
        for person in current_entry["ENTITY_COORDINATES"]:
            person_entry = {
                "id": person["person_id"],
                "person_id": person["person_id"],
                "coords": {
                    "x": str(person["x_coord"]),
                    "y": str(person["y_coord"])
                }
            }
            persons.append(person_entry)
        
        # Create the detection record using fixed and derived values.
        detection = {
            "camera_id": 2,                           # fixed to 1 as specified
            "image_url": current_entry.get("image_url", ""),  # empty string if not provided
            "is_organised": current_entry.get("is_organised", True),
            "no_of_people": current_entry.get("no_of_people", 0),
            "from_datetime": from_dt,
            "to_datetime": to_dt,
            "visitor_type": "Solo",                   # fixed to "Solo" for all entries
            "x_coord": x_coord,
            "y_coord": y_coord,
            "persons": persons
        }
        
        detections.append(detection)
    
    # Wrap the detections list in a top-level key and write to the output JSON file.
    output_data = {"detections": detections}
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Transformed data saved to {output_file}")

if __name__ == "__main__":
    # Parameters (adjust the input file, output file, and frame_interval as needed)
    input_file = "tracking_results_video2long_25fps.json"   # Replace with your input file
    output_file = "tracking_results_video2long_25fps_processed.json"
    
    # For example, for a 25fps video if you want 1 detection per second, set frame_interval = 25.
    # If frame_interval = 1, then every frame is considered.
    frame_interval = 25
    
    transform_tracking_data(input_file, output_file, frame_interval)
