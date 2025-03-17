# azure_client.py

from azure.storage.blob import BlobServiceClient, ContainerClient
from config import config
import logging
import sys

# Configure Logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to stdout
        logging.FileHandler("azure_client.log")  # Log to a file
    ]
)

logger = logging.getLogger(__name__)

# Configuration Variables
AZURE_CONNECTION_STRING = config.get("azure_connection_string")
AZURE_CONTAINER_NAME = config.get("azure_container_name")

if not AZURE_CONNECTION_STRING or not AZURE_CONTAINER_NAME:
    logger.error("Azure connection string or container name not found in configuration.")
    sys.exit(1)

try:
    # Initialize Blob Service Client
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client: ContainerClient = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
    logger.info(f"Successfully connected to Azure Blob container: '{AZURE_CONTAINER_NAME}'.")
except Exception as e:
    logger.error(f"Failed to connect to Azure Blob storage: {e}")
    sys.exit(1)


def list_blobs():
    """
    Lists all blobs in the specified Azure Blob container.
    """
    try:
        blobs = container_client.list_blobs()
        blob_list = [blob.name for blob in blobs]
        if blob_list:
            logger.info(f"Blobs in container '{AZURE_CONTAINER_NAME}':")
            for blob_name in blob_list:
                logger.info(f" - {blob_name}")
        else:
            logger.info(f"No blobs found in container '{AZURE_CONTAINER_NAME}'.")
    except Exception as e:
        logger.error(f"Error listing blobs in container '{AZURE_CONTAINER_NAME}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    """
    When run as a script, this module will attempt to connect to the specified Azure Blob container
    and list all blobs present in it.
    """
    logger.info("Starting Azure Blob storage connection verification.")
    list_blobs()
    logger.info("Azure Blob storage verification completed successfully.")
