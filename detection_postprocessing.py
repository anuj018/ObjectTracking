import json
import requests
from typing import Dict, Any
from datetime import datetime

def process_video_entries(file_path: str) -> Dict[str, Any]:
    """
    Processes video entries from a JSON file and formats the data for API submission.

    Args:
        file_path (str): The path to the JSON file containing video entries.

    Returns:
        Dict[str, Any]: The formatted data dictionary ready for API submission.
    
    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
        json.JSONDecodeError: If the JSON file contains invalid JSON.
        KeyError: If expected keys are missing in the JSON data.
        IndexError: If the JSON data list is empty.
    """
    try:
        # Load JSON data from the file
        with open(file_path, "r") as file:
            data = json.load(file)
        
        if not isinstance(data, list) or not data:
            raise ValueError("JSON data must be a non-empty list of entries.")
        
        # Extract first and last timestamp
        from_datetime = data[0]["current_time"]
        to_datetime = data[-1]["current_time"]
        
        # Initialize containers for processing
        person_entries = []
        person_ids = set()
        all_x_coords = []
        all_y_coords = []
        
        for entry in data:
            # Ensure 'ENTITY_COORDINATES' exists and is a list
            entity_coords = entry.get("ENTITY_COORDINATES", [])
            if not isinstance(entity_coords, list):
                logger.warning(f"'ENTITY_COORDINATES' is not a list in entry: {entry}")
                continue  # Skip to the next entry
            
            for person in entity_coords:
                # Extract necessary fields with default fallbacks
                person_id = person.get("person_id")
                x_coord = person.get("x_coord")
                y_coord = person.get("y_coord")
                
                if person_id is None or x_coord is None or y_coord is None:
                    logger.warning(f"Missing fields in person entry: {person}")
                    continue  # Skip incomplete person entries
                
                person_entries.append({
                    "id": person_id,  # Assuming 'id' and 'person_id' are the same
                    "person_id": person_id,
                    "coords": {
                        "x": str(x_coord),
                        "y": str(y_coord)
                    }
                })
                person_ids.add(person_id)
                all_x_coords.append(str(x_coord))
                all_y_coords.append(str(y_coord))
        
        # Construct final payload
        formatted_data = {
            "camera_id": data[0].get("camera_id", ""),  # Provide a default if missing
            "image_url": "",  # Fixed empty string as per original script
            "is_organised": data[0].get("is_organised", False),  # Default to False if missing
            "no_of_people": len(person_ids),
            "from_datetime": from_datetime,
            "to_datetime": to_datetime,
            "visitor_type": "solo",
            "x_coord": ",".join(all_x_coords),  # Concatenating all x coordinates
            "y_coord": ",".join(all_y_coords),  # Concatenating all y coordinates
            "persons": person_entries
        }
        
        return formatted_data
    
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")
        raise
    except json.JSONDecodeError as json_error:
        logger.error(f"Invalid JSON format: {json_error}")
        raise
    except KeyError as key_error:
        logger.error(f"Missing expected key: {key_error}")
        raise
    except IndexError as index_error:
        logger.error(f"Data list is empty: {index_error}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
