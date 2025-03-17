import cv2
import os

def extract_frame(video_path, frame_number, output_path):
    """
    Extracts a specific frame from a video and saves it as an image.

    :param video_path: Path to the input video file.
    :param frame_number: The frame number to extract (1-based indexing).
    :param output_path: Path to save the extracted frame image.
    :return: None
    """
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Capture the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'.")
        return

    # Total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {total_frames}")

    if frame_number < 1 or frame_number > total_frames:
        print(f"Error: Frame number {frame_number} is out of range.")
        cap.release()
        return

    # Set the position of the next frame to read
    # OpenCV uses 0-based indexing for frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    # Read the frame
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Cannot read frame {frame_number}.")
    else:
        # Save the frame as an image
        cv2.imwrite(output_path, frame)
        print(f"Frame {frame_number} saved successfully at '{output_path}'.")

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Define the path to the video file
    video_file = "/home/azureuser/workspace/Genfied/input_videos/02e3c90e-aa1e-428b-accf-a0befb9fffab.mp4"

    # Define the frame number to extract
    desired_frame_number = 150

    # Define the output image path
    # For example, saving in 'output_images' directory
    output_image_path = "/home/azureuser/workspace/Genfied/output_images/frame_150.jpg"

    # Extract and save the frame
    extract_frame(video_file, desired_frame_number, output_image_path)
