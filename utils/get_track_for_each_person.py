import csv
import json

# Load the JSON data from a file
with open('file2.json', 'r') as file:
    detections_data = json.load(file)

# Store the tracked movement
movement_data = []

# Track each person's movement
for frame_number, detection in enumerate(detections_data['detections'], start=1):
    for person in detection['persons']:
        person_id = person['person_id']
        x = int(person['coords']['x'])
        y = int(person['coords']['y'])
        
        # Check if the person already exists in the movement_data
        existing_person = next((entry for entry in movement_data if entry['person_id'] == person_id), None)
        
        if existing_person:
            # Calculate movement based on previous position
            prev_x, prev_y = existing_person['coords']
            movement_x = abs(x - prev_x)
            movement_y = abs(y - prev_y)
            existing_person['movement'].append({
                'frame': frame_number,
                'x': x,
                'y': y,
                'movement_x': movement_x,
                'movement_y': movement_y
            })
        else:
            # If new person, initialize their movement tracking
            movement_data.append({
                'person_id': person_id,
                'coords': (x, y),
                'movement': [{
                    'frame': frame_number,
                    'x': x,
                    'y': y,
                    'movement_x': 0,  # No movement for first frame
                    'movement_y': 0
                }]
            })

# Save movement data to a CSV
with open('person_movement_cam2.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['person_id', 'frame', 'x', 'y', 'movement_x', 'movement_y'])
    writer.writeheader()
    
    for person in movement_data:
        for movement in person['movement']:
            writer.writerow({
                'person_id': person['person_id'],
                'frame': movement['frame'],
                'x': movement['x'],
                'y': movement['y'],
                'movement_x': movement['movement_x'],
                'movement_y': movement['movement_y']
            })

# Print the movement data for verification
print(json.dumps(movement_data, indent=4))
