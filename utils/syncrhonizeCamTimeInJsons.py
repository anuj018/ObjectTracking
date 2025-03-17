import json

# Load file1.json and file2.json
with open('tracking_results_video1long_25fps_processed.json', 'r') as f1:
    data1 = json.load(f1)

with open('tracking_results_video2long_25fps_processed.json', 'r') as f2:
    data2 = json.load(f2)

# Extract the detections lists from each file
detections1 = data1.get('detections', [])
detections2 = data2.get('detections', [])

# Copy the time fields for the 3604 entries from file1 to file2.
# (Assuming detections1 has exactly 3604 entries.)
for i in range(len(detections1)):
    if i < len(detections2):
        # Update the corresponding detection in file2 with time fields from file1.
        detections2[i]['from_datetime'] = detections1[i].get('from_datetime')
        detections2[i]['to_datetime'] = detections1[i].get('to_datetime')
    else:
        # This branch should not execute since file2 has more entries.
        print(f"Warning: No corresponding detection in file2 for index {i}")

# Save the updated file2 back to disk
with open('file2.json', 'w') as f2:
    json.dump(data2, f2, indent=2)

print("Updated file2 with time fields from file1 for 3604 entries.")
