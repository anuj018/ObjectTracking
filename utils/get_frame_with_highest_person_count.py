import csv
from collections import defaultdict

# Dictionary to store the count of people per frame.
frame_counts = defaultdict(int)

# Open and read the CSV file.
with open("person_movement.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        frame = row['frame']
        # Increase the count for the current frame.
        frame_counts[frame] += 1

# Sort the frames by count in descending order.
sorted_frames = sorted(frame_counts.items(), key=lambda item: item[1], reverse=True)

# Get the frame with the highest count.
if sorted_frames:
    highest_frame, highest_count = sorted_frames[0]
    print(f"Frame with the highest number of people: Frame {highest_frame} with {highest_count} people")
else:
    print("No data found in the CSV.")

# List the top 15 frames with the highest number of people.
print("\nTop 15 frames with the highest number of people:")
for frame, count in sorted_frames[:15]:
    print(f"Frame {frame}: {count} people")
