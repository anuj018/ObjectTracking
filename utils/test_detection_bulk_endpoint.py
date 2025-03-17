import json
import requests

# Define the endpoint URL
endpoint = "http://genfied-api.xperie.nz:8000/detection/bulk"

# Load the JSON data from file
with open("tracking_results_video1long_25fps_processed.json", "r") as f:
    payload = json.load(f)

# Send the PUT request
response = requests.put(endpoint, json=payload)

# Check the response status
if response.status_code in (200, 201):
    print("Data sent successfully!")
else:
    print(f"Failed to send data. Status code: {response.status_code}")
    print("Response:", response.text)


# import os
# import json
# import requests

# # Define the endpoint URL

# endpoint = "http://genfied-api.xperie.nz:8000/detection/bulk"

# # Define the base folder where your synthetic data is stored.
# base_folder = "duplicate_data"

# # Iterate over all folders and files inside the base_folder.
# for root, dirs, files in os.walk(base_folder):
#     for file in files:
#         if file.endswith(".json"):
#             file_path = os.path.join(root, file)
#             print(f"Pushing data from: {file_path}")
#             try:
#                 # Load the JSON data from file.
#                 with open(file_path, "r") as f:
#                     payload = json.load(f)
#             except Exception as e:
#                 print(f"Failed to load JSON from {file_path}: {e}")
#                 continue

#             # Send the PUT request.
#             try:
#                 response = requests.put(endpoint, json=payload)
#             except Exception as e:
#                 print(f"Request error for {file_path}: {e}")
#                 continue

#             # Check the response status.
#             if response.status_code in (200, 201):
#                 print(f"Data from {file_path} sent successfully!")
#             else:
#                 print(f"Failed to send data from {file_path}. Status code: {response.status_code}")
#                 print("Response:", response.text)
