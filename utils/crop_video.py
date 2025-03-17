import ffmpeg

input_video = "/home/azureuser/workspace/Genfied/input_videos/video.mp4"
output_video = "/home/azureuser/workspace/Genfied/input_videos/video3_cropped_version.mp4"

start_time = "00:00:45"  # Start time (HH:MM:SS)
duration = "00:01:20"    # Duration (HH:MM:SS) (from 6:45 to 7:00)

# Run ffmpeg command
(
    ffmpeg
    .input(input_video, ss=start_time, t=duration)
    .output(output_video, codec="copy")  # Copying codec for faster processing
    .run(overwrite_output=True)
)

print(f"Video cropped successfully and saved as {output_video}")
