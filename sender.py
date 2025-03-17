# sender.py

import asyncio
import logging
import sys
from typing import Dict, Any

import aiohttp
from detection_postprocessing import process_video_entries
from config import config

# Configure Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),        # Log to stdout
        logging.FileHandler("sender.log")         # Log to a file named sender.log
    ]
)

logger = logging.getLogger(__name__)

# Configuration Variables
DETECTION_ENDPOINT = config.get("detection_endpoint")

if not DETECTION_ENDPOINT:
    logger.error("Detection endpoint not found in configuration.")
    sys.exit(1)


async def send_detection_data(detections_jsonfile_path: str):
    """
    Processes the raw JSON file and sends the formatted data to the detection API endpoint.
    
    Args:
        detections_jsonfile_path (str): The path to the raw JSON file containing detections.
    
    Raises:
        SystemExit: If sending data fails after retries.
    """
    try:
        # Process the raw JSON file to get formatted data
        formatted_data: Dict[str, Any] = process_video_entries(detections_jsonfile_path)
        logger.info(f"Processed data from '{detections_jsonfile_path}' successfully.")
        
    except Exception as e:
        logger.error(f"Failed to process video entries: {e}", exc_info=True)
        sys.exit(1)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.put(DETECTION_ENDPOINT, json=formatted_data) as response:
                if response.status == 200:
                    logger.info("Data sent successfully.")
                    response_text = await response.text()
                    logger.info(f"Response Text: {response_text}")
                else:
                    logger.error(f"Failed to send data. Status: {response.status}")
                    response_text = await response.text()
                    logger.error(f"Response Text: {response_text}")
                    # Optionally, implement retry logic or other error handling here
    except aiohttp.ClientError as client_error:
        logger.error(f"HTTP Client error occurred: {client_error}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while sending data: {e}", exc_info=True)
        sys.exit(1)


async def main():
    """
    The main entry point for the script.
    """
    # Define the path to your raw JSON file
    detections_jsonfile_path = "demo_video_snippet.json"  # Update this path as needed
    
    # Optionally, you can make the JSON file path configurable via environment variables or command-line arguments
    
    await send_detection_data(detections_jsonfile_path)


if __name__ == "__main__":
    """
    When the script is run directly, execute the main coroutine.
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script interrupted by user.")
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}", exc_info=True)
        sys.exit(1)
