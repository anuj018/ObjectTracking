import cv2

# Define image path
image_path = "/home/azureuser/workspace/Genfied/output_images/frame_150.jpg"

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Get image dimensions
height, width, _ = image.shape

# Define rectangle coordinates
x_min = int(0.40 * width)
x_max = int(0.60 * width)
# y_min = int(0.30 * height)
y_min = 270
y_max = int(0.70 * height)

# Draw rectangle (BGR: Blue, Green, Red)
color = (0, 255, 0)  # Green rectangle
thickness = 2
cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
print(x_min, y_min, x_max, y_max)

# Save the output image
output_path = "/home/azureuser/workspace/Genfied/output_images/frame_150_rectangle.jpg"
cv2.imwrite(output_path, image)

print(f"Rectangle drawn and saved at {output_path}")
