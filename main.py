import asyncio
from fastapi import FastAPI, Depends
from datetime import datetime
from database import SessionLocal, ProcessedVideo
from azure_client import container_client
from processor import process_video
from detection_postprocessing import process_video_entries
from sender import send_detection_data
import tempfile
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
import logging


# Configure Logging
logging.basicConfig(
    level=logging.INFO,  # Set the base logging level
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()          # Also log to stdout
    ]
)

# Create a logger for this module
logger = logging.getLogger(__name__)

# Initialize ProcessPoolExecutor once to reuse across tasks
executor = ProcessPoolExecutor()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def monitor_blob_storage():
    logger.info("Started monitoring blob storage.")
    loop = asyncio.get_running_loop()
    while True:
        db = SessionLocal()
        try:
            blobs = container_client.list_blobs()
            blob_list = [blob.name for blob in blobs]
            logger.debug(f"Retrieved {len(blob_list)} blobs from container.")
            for blob_name in blob_list:
                processed = db.query(ProcessedVideo).filter(ProcessedVideo.blob_name == blob_name).first()
                if not processed:
                    logger.info(f"New video found: {blob_name}")

                    # Download the blob asynchronously
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                            logger.debug(f"Downloading blob: {blob_name}")
                            # download_stream = await loop.run_in_executor(
                            #     None, 
                            #     container_client.download_blob(blob).readall
                            # )
                            download_stream = await loop.run_in_executor(
                                None, 
                                container_client.download_blob(blob_name).readall
                            )

                            tmp_file.write(download_stream)
                            temp_file_path = tmp_file.name
                            logger.debug(f"Downloaded blob to temporary file: {temp_file_path}")
                    except Exception as download_error:
                        logger.error(f"Failed to download blob {blob_name}: {download_error}", exc_info=True)
                        continue  # Skip to the next blob

                    # Process the video in a separate process
                    try:
                        logger.info(f"Processing video: {temp_file_path}")
                        detections_jsonfile_path = await loop.run_in_executor(
                            executor, 
                            process_video, 
                            temp_file_path
                        )
                        logger.info(f"Processing complete for video: {temp_file_path}")
                    except Exception as process_error:
                        logger.error(f"Error processing video {temp_file_path}: {process_error}", exc_info=True)
                        os.remove(temp_file_path)
                        continue  # Skip to the next blob

                    # Send the detection data
                    try:
                        logger.debug("Sending detection data.")
                        await send_detection_data(detections_jsonfile_path)
                        logger.info("Detection data sent successfully.")
                    except Exception as send_error:
                        logger.error(f"Failed to send detection data for {blob_name}: {send_error}", exc_info=True)

                    # Mark the blob as processed in the database
                    try:
                        new_record = ProcessedVideo(
                            blob_name=blob_name,
                            processed_at=datetime.utcnow()
                        )
                        db.add(new_record)
                        db.commit()
                        logger.info(f"Marked blob as processed: {blob_name}")
                    except Exception as db_error:
                        logger.error(f"Database error while processing {blob_name}: {db_error}", exc_info=True)

                    # Remove the temporary file
                    try:
                        os.remove(temp_file_path)
                        logger.debug(f"Removed temporary file: {temp_file_path}")
                    except Exception as remove_error:
                        logger.warning(f"Could not remove temporary file {temp_file_path}: {remove_error}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during blob monitoring: {e}", exc_info=True)
        finally:
            db.close()
            logger.debug("Database session closed.")

        logger.debug("Sleeping for 30 seconds before next check.")
        await asyncio.sleep(30)  # Check every 30 seconds

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Starting blob storage monitor task.")
    monitor_task = asyncio.create_task(monitor_blob_storage())
    try:
        yield
    finally:
        logger.info("Application shutdown: Cancelling blob storage monitor task.")
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            logger.info("Blob storage monitor task cancelled successfully.")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    logger.info("Received request for root endpoint.")
    return {"message": "Blob Monitor is running."}
