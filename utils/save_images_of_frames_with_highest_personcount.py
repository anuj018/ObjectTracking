import os
import csv
import cv2

# Path to the video file.
video_path = "/home/azureuser/workspace/Genfied/input_videos/video1long.mp4"

# Output folder for frames with drawn bounding boxes.
output_folder = "framesWithHighestPersonCountCam1"
os.makedirs(output_folder, exist_ok=True)

# List of frames (as integers) with the highest person count (top 15).
frames_of_interest = [124, 638, 1132, 1762, 1766, 1767, 1775, 2381, 378, 611, 613, 636, 637, 658, 661]

# Build a dictionary mapping each frame number to a list of (x, y) coordinates.
bbox_dict = {}
csv_filename = "person_movement.csv"

with open(csv_filename, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            frame_num = int(row["frame"])
        except ValueError:
            continue  # skip rows with invalid frame numbers

        # Only store coordinates for frames of interest.
        if frame_num in frames_of_interest:
            try:
                x = int(float(row["x"]))  # in case x is a string number (or float string)
                y = int(float(row["y"]))
            except ValueError:
                continue
            bbox_dict.setdefault(frame_num, []).append((x, y))

# Open the video file using OpenCV.
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file at {video_path}")
    exit(1)

# Define fixed bounding box dimensions.
box_width = 50
box_height = 50
half_width = box_width // 2
half_height = box_height // 2

# Process each frame of interest.
for frame_num in frames_of_interest:
    # Set the video position to the desired frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: Could not read frame {frame_num}")
        continue

    # Check if we have any bounding boxes for this frame.
    if frame_num in bbox_dict:
        for (x, y) in bbox_dict[frame_num]:
            # Assume (x, y) is the center; calculate top-left and bottom-right.
            top_left = (x - half_width, y - half_height)
            bottom_right = (x + half_width, y + half_height)
            # Draw a green bounding box with thickness 2.
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            # Optionally, put the person ID (or a label) near the bounding box.
            cv2.putText(frame, f"{x},{y}", (top_left[0], top_left[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save the modified frame as an image.
    output_file = os.path.join(output_folder, f"frame_{frame_num}.jpg")
    cv2.imwrite(output_file, frame)
    print(f"Saved frame {frame_num} with bounding boxes to {output_file}")

cap.release()
