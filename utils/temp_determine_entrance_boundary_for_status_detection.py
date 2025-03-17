import cv2
import numpy as np

# Image path
image_path = "/home/azureuser/workspace/Genfied/output_images/frame_150.jpg"

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found at the specified path.")
    exit()

# Image dimensions
image_height, image_width, _ = image.shape

# Define percentage ranges and increments
x_start_percent, x_end_percent = 40, 70
y_start_percent, y_end_percent = 10, 60
increment = 5

# Convert percentages to pixel values
def percent_to_pixels(percent, axis):
    if axis == "x":
        return int(image_width * percent / 100)
    elif axis == "y":
        return int(image_height * percent / 100)
    else:
        raise ValueError("Invalid axis. Use 'x' or 'y'.")

# Iterate over the X and Y ranges
for x_percent in range(x_start_percent, x_end_percent + 1, increment):
    for y_percent in range(y_start_percent, y_end_percent + 1, increment):
        # Calculate rectangle coordinates
        x1 = percent_to_pixels(x_percent, "x")
        y1 = percent_to_pixels(y_percent, "y")
        x2 = percent_to_pixels(x_percent + increment, "x")
        y2 = percent_to_pixels(y_percent + increment, "y")

        # Draw the rectangle on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle with thickness 2


# Save the image with rectangles (optional)
output_path = "/home/azureuser/workspace/Genfied/output_images/frame_150_with_rectangles.jpg"
cv2.imwrite(output_path, image)
print(f"Image with rectangles saved to {output_path}")