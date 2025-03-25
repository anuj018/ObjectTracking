import asyncio
from confluent_kafka import Consumer, KafkaException, KafkaError
import json
import base64
import io
from PIL import Image
import numpy as np
import uuid
import os
import logging
from datetime import datetime
import cv2
from concurrent.futures import ProcessPoolExecutor

# Import your custom modules - adjust as needed
from processor_segment_with_transreid import Segmentation_DeepSort
from sender import send_detection_data

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize process pool
executor = ProcessPoolExecutor()

# Results buffer
frame_results = {}
BATCH_SIZE = 10

# Kafka configuration
kafka_config = {
    'bootstrap.servers': os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka-broker:9092'),
    'group.id': os.environ.get('KAFKA_GROUP_ID', 'image-processor-group'),
    'auto.offset.reset': 'latest'
}

# Define your process_image function
def process_image(image_data, store_id=None, camera_id=None, frame_id=0, info_flag=True):
    """
    Process a single image frame and extract required information.
    
    Args:
        image_data: Image data as bytes or numpy array
        store_id: Store identifier (optional)
        camera_id: Camera identifier (optional)
        frame_id: Frame identifier (optional)
        info_flag: Whether to log detailed information
    
    Returns:
        Dictionary containing detection results and metadata
    """
    try:
        # Initialize the processor
        processor = Segmentation_DeepSort(info_flag=info_flag)
        
        # Convert bytes to image if needed
        if isinstance(image_data, bytes):
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            # Convert PIL to numpy array (BGR for OpenCV)
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            # Assume it's already a numpy array
            frame = image_data
        
        # Process the frame using the class method
        result = processor.process_single_frame(frame, store_id, camera_id, frame_id)
        
        # Save the tracker state
        processor.tracker.save_global_database()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in process_image: {e}", exc_info=True)
        return {
            "error": str(e),
            "camera_id": camera_id if camera_id else "unknown",
            "store_id": store_id if store_id else "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "failed"
        }

async def process_frame_data(frame_data):
    """Process a frame received from Kafka"""
    try:
        # Extract data from Kafka message
        store_id = frame_data.get('store_id', 'unknown')
        camera_id = frame_data.get('camera_id', 'unknown')
        frame_id = frame_data.get('frame_id', 0)
        timestamp = frame_data.get('timestamp', datetime.utcnow().isoformat())
        
        # Decode image
        image_bytes = base64.b64decode(frame_data['image_data'])
        
        # Process image using our function
        loop = asyncio.get_running_loop()
        detection_results = await loop.run_in_executor(
            executor,
            lambda: process_image(
                image_bytes, 
                store_id=store_id,
                camera_id=camera_id,
                frame_id=frame_id
            )
        )
        
        # Check for errors
        if 'error' in detection_results:
            logger.error(f"Error in image processing: {detection_results['error']}")
            return
        
        # Extract key result metrics
        singles = detection_results.get('singles', 0)
        couples = detection_results.get('couples', 0)
        groups = detection_results.get('groups', 0)
        
        logger.info(f"Processing results - Store: {store_id}, Camera: {camera_id}, " +
                   f"Singles: {singles}, Couples: {couples}, Groups: {groups}")
        
        # Create result record for batch sending
        result = {
            "store_id": store_id,
            "camera_id": camera_id,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "processed_timestamp": datetime.utcnow().isoformat(),
            "entity_coordinates": detection_results.get('entity_coordinates', []),
            "singles": singles,
            "couples": couples,
            "groups": groups,
            "total_people": detection_results.get('no_of_people', 0)
        }
        
        # # Add to buffer
        # if store_id not in frame_results:
        #     frame_results[store_id] = {}
        # if camera_id not in frame_results[store_id]:
        #     frame_results[store_id][camera_id] = []
            
        # frame_results[store_id][camera_id].append(result)
        success = await send_detection_data([result])
        if success:
            logger.info(f"Successfully sent frame {frame_id} for store {store_id}, camera {camera_id}")
        else:
            logger.error(f"Failed to send frame {frame_id} for store {store_id}, camera {camera_id}")

        # # Check if batch should be sent
        # if len(frame_results[store_id][camera_id]) >= BATCH_SIZE:
        #     await send_batch(store_id, camera_id)
            
    except Exception as e:
        logger.error(f"Error processing frame: {e}", exc_info=True)

async def send_batch(store_id, camera_id):
    """Send batch of results to final destination"""
    if store_id not in frame_results or camera_id not in frame_results[store_id]:
        return
        
    batch = frame_results[store_id][camera_id]
    if not batch:
        return
        
    logger.info(f"Sending batch of {len(batch)} frames from store {store_id}, camera {camera_id}")
    
    # Write batch to temp file
    batch_id = str(uuid.uuid4())
    batch_file = f"batch_{store_id}_{camera_id}_{batch_id}.json"
    
    with open(batch_file, 'w') as f:
        json.dump(batch, f)
    
    try:
        # Send batch to destination
        await send_detection_data(batch_file)
        logger.info(f"Successfully sent batch {batch_id}")
        
        # Clear the sent frames
        frame_results[store_id][camera_id] = []
    except Exception as e:
        logger.error(f"Failed to send batch {batch_id}: {e}", exc_info=True)
    finally:
        # Clean up
        if os.path.exists(batch_file):
            os.remove(batch_file)

async def kafka_consumer_task():
    """Task to consume messages from Kafka"""
    # Add regex pattern matching for topics
    consumer_config = kafka_config.copy()
    consumer_config['auto.offset.reset'] = 'latest'
    
    consumer = Consumer(consumer_config)
    
    # List potential topics instead of using regex
    topics = [
        'store-001-camera-001',
        'store-001-camera-002',
        # Add all your expected topics here
    ]
    
    consumer.subscribe(topics)
    logger.info(f"Subscribed to Kafka topics: {topics}")
    
    try:
        while True:
            msg = consumer.poll(1.0)
            
            if msg is None:
                continue
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    logger.info(f'Reached end of partition: {msg.topic()} [{msg.partition()}]')
                elif msg.error().code() == KafkaError._TRANSPORT:
                    logger.error(f'Transport error: {msg.error()}')
                    break
                else:
                    raise KafkaException(msg.error())
                
            try:
                # Get topic details (for logging)
                topic = msg.topic()
                
                # Parse the message
                frame_data = json.loads(msg.value().decode('utf-8'))
                logger.debug(f"Received message from topic {topic}")
                
                # Process the frame asynchronously
                asyncio.create_task(process_frame_data(frame_data))
                
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                
    except KeyboardInterrupt:
        logger.info("Shutting down Kafka consumer")
    finally:
        # Close down consumer
        consumer.close()

async def main():
    """Main entry point for the consumer service"""
    logger.info("Starting Kafka consumer service")
    
    # Start the Kafka consumer
    consumer_task = asyncio.create_task(kafka_consumer_task())
    
    try:
        # Run indefinitely
        await consumer_task
    except asyncio.CancelledError:
        # Handle shutdown
        logger.info("Consumer task cancelled, shutting down")
        
        # # Send any remaining batches
        # for store_id in frame_results:
        #     for camera_id in frame_results[store_id]:
        #         if frame_results[store_id][camera_id]:
        #             logger.info(f"Sending remaining batch for store {store_id}, camera {camera_id}")
        #             await send_batch(store_id, camera_id)

if __name__ == "__main__":
    asyncio.run(main())