import cv2
import os
from deepface import DeepFace

def process_video(video_path, output_path, detector_backend="retinaface"):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Use DeepFace to extract faces from the full frame
        try:
            faces = DeepFace.extract_faces(frame, detector_backend=detector_backend, enforce_detection=False)
        except Exception as e:
            print(f"Error in face detection at frame {frame_count}: {e}")
            faces = []

        # For each detected face, draw a bounding box and check embedding
        print(f"number of faces detected are {len(faces)}")
        for idx, face in enumerate(faces):
            facial_area = face.get("facial_area", {})
            x = facial_area.get("x", 0)
            y = facial_area.get("y", 0)
            w = facial_area.get("w", 0)
            h = facial_area.get("h", 0)
            
            # Draw rectangle with green color and thickness 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Crop the detected face region from the frame
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                print(f"Empty face region at frame {frame_count}, face index {idx}")
                continue

            # Use DeepFace.represent to extract the embedding from the face region
            try:
                # Here we use enforce_detection=False to avoid errors if detection fails again
                representation = DeepFace.represent(face_img, model_name='Facenet', detector_backend=detector_backend, enforce_detection=False)
                if representation and isinstance(representation, list):
                    embedding = representation[0].get('embedding')
                    print(f"Frame {frame_count}, Face {idx}: Embedding length = {len(embedding) if embedding else 'None'}")
                else:
                    print(f"Frame {frame_count}, Face {idx}: No representation returned.")
            except Exception as e:
                print(f"Error in represent at frame {frame_count}, face {idx}: {e}")

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print("Processing complete. Output saved to:", output_path)

if __name__ == "__main__":
    videofile_name = "cropped_video_employeeAndBaldie"
    video_path = f"/home/azureuser/workspace/Genfied/input_videos/{videofile_name}.mp4"
    output_path = f"/home/azureuser/workspace/Genfied/input_videos/result_{videofile_name}_faces.mp4"
    process_video(video_path, output_path)
