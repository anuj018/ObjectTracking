import cv2

# Specify the video file path and the frame number you want to extract
video_path = "../input_videos/video1long_cropped_lady.mp4"  # Replace with your video file path
frame_to_extract = 707      # Frame number to extract
output_image = "frame_707.jpg" # Output image file name

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit(1)

# Set the current position of the video to the desired frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_extract)

# Read the frame
ret, frame = cap.read()
if ret:
    cv2.imwrite(output_image, frame)
    print(f"Frame {frame_to_extract} saved as {output_image}")
else:
    print(f"Error: Could not read frame {frame_to_extract}")

# Release the video capture object
cap.release()
