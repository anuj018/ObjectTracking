import asyncio
import time
import logging
import threading
from collections import deque, defaultdict
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FrameBufferStats:
    """Tracks statistics about the frame buffer performance"""
    def __init__(self):
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.processing_times = deque(maxlen=1000)  # Last 1000 processing times
        self.queue_times = deque(maxlen=1000)       # Last 1000 queue times
        self.buffer_sizes = deque(maxlen=1000)      # Historical buffer sizes
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def update_received(self):
        with self.lock:
            self.frames_received += 1
    
    def update_processed(self, processing_time, queue_time):
        with self.lock:
            self.frames_processed += 1
            self.processing_times.append(processing_time)
            self.queue_times.append(queue_time)
    
    def update_dropped(self):
        with self.lock:
            self.frames_dropped += 1
    
    def update_buffer_size(self, size):
        with self.lock:
            self.buffer_sizes.append(size)
    
    def get_stats(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Calculate rates
            fps_in = self.frames_received / elapsed if elapsed > 0 else 0
            fps_out = self.frames_processed / elapsed if elapsed > 0 else 0
            drop_rate = self.frames_dropped / max(1, self.frames_received) * 100
            
            # Calculate averages
            avg_processing = np.mean(self.processing_times) if self.processing_times else 0
            avg_queue = np.mean(self.queue_times) if self.queue_times else 0
            avg_buffer = np.mean(self.buffer_sizes) if self.buffer_sizes else 0
            
            # Reset counters
            self.frames_received = 0
            self.frames_processed = 0
            self.frames_dropped = 0
            self.last_update = now
            
            return {
                "fps_in": fps_in,
                "fps_out": fps_out, 
                "drop_rate": drop_rate,
                "avg_processing_time": avg_processing,
                "avg_queue_time": avg_queue,
                "avg_buffer_size": avg_buffer,
                "max_buffer_size": max(self.buffer_sizes) if self.buffer_sizes else 0,
                "processing_time_p95": np.percentile(self.processing_times, 95) if len(self.processing_times) >= 20 else None,
                "queue_time_p95": np.percentile(self.queue_times, 95) if len(self.queue_times) >= 20 else None
            }


class RobustFrameBuffer:
    """
    A robust frame buffer system that handles:
    - Priority-based processing
    - Frame dropping when overloaded
    - Timeouts for stale frames
    - Auto-adjustment based on system performance
    - Statistics collection
    """
    def __init__(self, 
                 max_size_per_camera=3000,        # Maximum frames per camera
                 max_total_size=30000,            # Maximum total frames
                 timeout_seconds=100.0,           # Maximum time a frame can stay in buffer
                 drop_strategy='oldest',        # Strategy for dropping frames: 'oldest', 'newest', 'smart'
                 auto_adjust=False,              # Auto-adjust buffer sizes based on performance
                 camera_priorities=None):       # Dict of camera_id -> priority (1-10, higher is more important)
        # Initialize buffer as dict of deques (per camera)
        self.buffers = defaultdict(lambda: deque(maxlen=max_size_per_camera))
        self.timestamps = defaultdict(dict)     # Store enter/exit timestamps
        self.frame_metadata = defaultdict(dict) # Store frame metadata
        
        # Buffer configuration
        self.max_size_per_camera = max_size_per_camera
        self.max_total_size = max_total_size
        self.timeout_seconds = timeout_seconds
        self.drop_strategy = drop_strategy
        self.auto_adjust = auto_adjust
        
        # Camera priorities (default: all equal at 5)
        self.camera_priorities = camera_priorities or {}
        self.default_priority = 5
        
        # Performance tracking
        self.stats = FrameBufferStats()
        self.last_stats_time = time.time()
        self.stats_interval = 30  # Log stats every 30 seconds
        
        # Locking
        self.buffer_lock = asyncio.Lock()
        
        # Status
        self.active = True
        
        # Start monitor task
        self.monitor_task = None
        
        logger.info(f"Initialized RobustFrameBuffer with max_size_per_camera={max_size_per_camera}, " 
                    f"max_total_size={max_total_size}, drop_strategy='{drop_strategy}'")
    
    async def start_monitors(self):
        """Start monitoring tasks"""
        self.monitor_task = asyncio.create_task(self._monitor_buffer())
        logger.info("Started buffer monitoring tasks")
    
    async def _monitor_buffer(self):
        """Monitor task to handle timeouts and auto-adjustment"""
        while self.active:
            try:
                # Wait a bit before checking
                await asyncio.sleep(1.0)
                
                await self._clean_stale_frames()
                
                # Log performance stats periodically
                now = time.time()
                if now - self.last_stats_time > self.stats_interval:
                    self._log_performance_stats()
                    self.last_stats_time = now
                    
                    # Adjust buffer size if needed
                    if self.auto_adjust:
                        await self._auto_adjust_buffer_sizes()
                
            except Exception as e:
                logger.error(f"Error in buffer monitor: {e}", exc_info=True)
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        stats = self.stats.get_stats()
        logger.info(f"Buffer stats: FPS in={stats['fps_in']:.1f}, out={stats['fps_out']:.1f}, "
                    f"drop={stats['drop_rate']:.1f}%, avg_queue={stats['avg_queue_time']*1000:.1f}ms, "
                    f"avg_buffer={stats['avg_buffer_size']:.1f} frames")
                    
        # Log per-camera buffer sizes
        camera_sizes = {camera_id: len(buffer) for camera_id, buffer in self.buffers.items()}
        if camera_sizes:
            logger.info(f"Camera buffer sizes: {camera_sizes}")
    
    async def _auto_adjust_buffer_sizes(self):
        """Automatically adjust buffer sizes based on performance"""
        stats = self.stats.get_stats()
        
        # If we're dropping frames (more than 5%) and queue times are high, 
        # increase buffer size to allow more bursting
        if stats['drop_rate'] > 5.0 and stats['avg_queue_time'] > 0.5:
            old_size = self.max_size_per_camera
            # Increase by 20% with a cap
            self.max_size_per_camera = min(int(self.max_size_per_camera * 1.2), 300)
            self.max_total_size = min(int(self.max_total_size * 1.2), 1500)
            
            if old_size != self.max_size_per_camera:
                logger.info(f"Auto-increased buffer sizes: per_camera={old_size}->{self.max_size_per_camera}, "
                           f"total={self.max_total_size}")
                
                # Update max lengths on existing buffers
                async with self.buffer_lock:
                    for camera_id, buffer in self.buffers.items():
                        buffer_list = list(buffer)
                        self.buffers[camera_id] = deque(buffer_list, maxlen=self.max_size_per_camera)
        
        # If we're processing frames quickly with low drop rate, decrease buffer
        # to reduce memory usage and latency
        elif stats['drop_rate'] < 1.0 and stats['avg_queue_time'] < 0.1 and stats['avg_buffer_size'] < self.max_size_per_camera * 0.3:
            old_size = self.max_size_per_camera
            # Decrease by 10% with a floor
            self.max_size_per_camera = max(int(self.max_size_per_camera * 0.9), 30)
            self.max_total_size = max(int(self.max_total_size * 0.9), 150)
            
            if old_size != self.max_size_per_camera:
                logger.info(f"Auto-decreased buffer sizes: per_camera={old_size}->{self.max_size_per_camera}, "
                           f"total={self.max_total_size}")
                
                # Update max lengths on existing buffers (will trim if needed)
                async with self.buffer_lock:
                    for camera_id, buffer in self.buffers.items():
                        buffer_list = list(buffer)
                        # Keep the newest frames if we need to trim
                        if len(buffer_list) > self.max_size_per_camera:
                            buffer_list = buffer_list[-self.max_size_per_camera:]
                        self.buffers[camera_id] = deque(buffer_list, maxlen=self.max_size_per_camera)
    
    async def _clean_stale_frames(self):
        """Remove frames that have been in the buffer too long"""
        now = time.time()
        stale_threshold = now - self.timeout_seconds
        
        async with self.buffer_lock:
            for camera_id, buffer in list(self.buffers.items()):
                # Skip empty buffers
                if not buffer:
                    continue
                
                # Check for stale frames
                while buffer and self.timestamps[camera_id].get(id(buffer[0]), now) < stale_threshold:
                    # Remove oldest frame
                    frame, metadata = buffer.popleft()
                    frame_id = id(frame)
                    
                    # Clean up associated data
                    if frame_id in self.timestamps[camera_id]:
                        del self.timestamps[camera_id][frame_id]
                    if frame_id in self.frame_metadata[camera_id]:
                        del self.frame_metadata[camera_id][frame_id]
                    
                    # Update stats
                    self.stats.update_dropped()
                    logger.debug(f"Dropped stale frame from camera {camera_id}, age={now - self.timestamps[camera_id].get(frame_id, now):.2f}s")
    
    async def add_frame(self, frame, metadata):
        """
        Add a frame to the buffer.
        
        Args:
            frame: The image frame
            metadata: Dict with at least 'camera_id' key
        
        Returns:
            bool: True if added, False if dropped
        """
        camera_id = metadata.get('camera_id', 'unknown')
        
        # Update received count in stats
        self.stats.update_received()
        
        async with self.buffer_lock:
            # Check if we need to drop frames
            total_frames = sum(len(buffer) for buffer in self.buffers.values())
            
            # If total buffer exceeds max and this camera's buffer is not empty
            if total_frames >= self.max_total_size and self.buffers[camera_id]:
                # Implement frame dropping strategy
                await self._drop_frames(camera_id, 1)
            
            # Store frame and metadata
            self.buffers[camera_id].append((frame, metadata))
            frame_id = id(frame)
            self.timestamps[camera_id][frame_id] = time.time()
            self.frame_metadata[camera_id][frame_id] = metadata
            
            # Update buffer size stats
            self.stats.update_buffer_size(total_frames + 1)
            
            return True
    
    async def _drop_frames(self, current_camera_id, count=1):
        """
        Drop frames based on strategy.
        
        Args:
            current_camera_id: The camera ID that's trying to add a frame
            count: Number of frames to drop
        """
        if self.drop_strategy == 'newest':
            # Drop newest from current camera
            for _ in range(min(count, len(self.buffers[current_camera_id]))):
                if self.buffers[current_camera_id]:
                    frame, metadata = self.buffers[current_camera_id].pop()  # Remove newest
                    frame_id = id(frame)
                    
                    # Clean up associated data
                    if frame_id in self.timestamps[current_camera_id]:
                        del self.timestamps[current_camera_id][frame_id]
                    if frame_id in self.frame_metadata[current_camera_id]:
                        del self.frame_metadata[current_camera_id][frame_id]
                    
                    # Update stats
                    self.stats.update_dropped()
            
        elif self.drop_strategy == 'smart':
            # Find lowest priority camera with the most frames
            lowest_priority = self.camera_priorities.get(current_camera_id, self.default_priority)
            target_camera = current_camera_id
            
            for camera_id, buffer in self.buffers.items():
                camera_priority = self.camera_priorities.get(camera_id, self.default_priority)
                
                # If this camera has lower priority and more frames
                if (camera_priority < lowest_priority or 
                    (camera_priority == lowest_priority and len(buffer) > len(self.buffers[target_camera]))):
                    lowest_priority = camera_priority
                    target_camera = camera_id
            
            # Drop oldest frame from target camera
            for _ in range(min(count, len(self.buffers[target_camera]))):
                if self.buffers[target_camera]:
                    frame, metadata = self.buffers[target_camera].popleft()  # Remove oldest
                    frame_id = id(frame)
                    
                    # Clean up associated data
                    if frame_id in self.timestamps[target_camera]:
                        del self.timestamps[target_camera][frame_id]
                    if frame_id in self.frame_metadata[target_camera]:
                        del self.frame_metadata[target_camera][frame_id]
                    
                    # Update stats
                    self.stats.update_dropped()
                    logger.debug(f"Smart dropped frame from camera {target_camera} (priority {lowest_priority})")
        
        else:  # Default: 'oldest'
            # Find the oldest frame across all cameras
            oldest_time = float('inf')
            oldest_camera = None
            
            for camera_id, timestamp_dict in self.timestamps.items():
                if not self.buffers[camera_id]:
                    continue
                
                # Get timestamp of oldest frame in this camera's buffer
                oldest_frame = self.buffers[camera_id][0]
                oldest_frame_id = id(oldest_frame[0])
                
                if oldest_frame_id in timestamp_dict:
                    frame_time = timestamp_dict[oldest_frame_id]
                    if frame_time < oldest_time:
                        oldest_time = frame_time
                        oldest_camera = camera_id
            
            # Drop oldest frames
            if oldest_camera:
                for _ in range(min(count, len(self.buffers[oldest_camera]))):
                    if self.buffers[oldest_camera]:
                        frame, metadata = self.buffers[oldest_camera].popleft()  # Remove oldest
                        frame_id = id(frame)
                        
                        # Clean up associated data
                        if frame_id in self.timestamps[oldest_camera]:
                            del self.timestamps[oldest_camera][frame_id]
                        if frame_id in self.frame_metadata[oldest_camera]:
                            del self.frame_metadata[oldest_camera][frame_id]
                        
                        # Update stats
                        self.stats.update_dropped()
                        logger.debug(f"Dropped oldest frame from camera {oldest_camera}")
    
    async def get_next_batch(self, max_batch_size, strategy='fair'):
        """
        Get the next batch of frames to process.
        
        Args:
            max_batch_size: Maximum number of frames to return
            strategy: Strategy for selecting frames:
                      'fair' - Round-robin across cameras
                      'priority' - Higher priority cameras first
                      'oldest' - Oldest frames first
        
        Returns:
            list: List of (frame, metadata) tuples
        """
        async with self.buffer_lock:
            # Special case: if buffer is empty, return empty batch
            total_frames = sum(len(buffer) for buffer in self.buffers.values())
            if total_frames == 0:
                return []
            
            batch = []
            cameras_with_frames = [camera_id for camera_id, buffer in self.buffers.items() if buffer]
            
            if strategy == 'priority':
                # Sort cameras by priority (highest first)
                cameras_with_frames.sort(
                    key=lambda camera_id: self.camera_priorities.get(camera_id, self.default_priority),
                    reverse=True
                )
                
                # Take frames from highest priority cameras first
                remaining = max_batch_size
                while cameras_with_frames and remaining > 0:
                    camera_id = cameras_with_frames[0]
                    
                    # Take as many frames as possible from this camera
                    frames_to_take = min(remaining, len(self.buffers[camera_id]))
                    for _ in range(frames_to_take):
                        frame_tuple = self.buffers[camera_id].popleft()
                        batch.append(frame_tuple)
                        
                        # Track processing time
                        frame_id = id(frame_tuple[0])
                        enter_time = self.timestamps[camera_id].get(frame_id, time.time())
                        queue_time = time.time() - enter_time
                        self.stats.queue_times.append(queue_time)
                        
                        # Clean up
                        if frame_id in self.timestamps[camera_id]:
                            del self.timestamps[camera_id][frame_id]
                        if frame_id in self.frame_metadata[camera_id]:
                            del self.frame_metadata[camera_id][frame_id]
                    
                    remaining -= frames_to_take
                    
                    # If this camera is empty now, remove it from the list
                    if not self.buffers[camera_id]:
                        cameras_with_frames.pop(0)
                    
            elif strategy == 'oldest':
                # Take oldest frames first
                frame_entries = []
                
                # Collect all frames with their timestamps
                for camera_id in cameras_with_frames:
                    for i, frame_tuple in enumerate(self.buffers[camera_id]):
                        frame_id = id(frame_tuple[0])
                        timestamp = self.timestamps[camera_id].get(frame_id, time.time())
                        frame_entries.append((camera_id, i, timestamp))
                
                # Sort by timestamp
                frame_entries.sort(key=lambda entry: entry[2])
                
                # Take up to max_batch_size
                taken = 0
                
                # Need to track which frames we've taken from each camera
                # to adjust indices after removal
                taken_from_camera = defaultdict(list)
                
                for camera_id, idx, _ in frame_entries[:max_batch_size]:
                    # Adjust index based on frames already taken from this camera
                    adjusted_idx = idx - len([i for i in taken_from_camera[camera_id] if i < idx])
                    
                    # Take the frame
                    frame_tuple = self.buffers[camera_id][adjusted_idx]
                    self.buffers[camera_id].remove(frame_tuple)  # Remove specific frame
                    batch.append(frame_tuple)
                    
                    # Update tracking
                    taken_from_camera[camera_id].append(idx)
                    
                    # Track processing time
                    frame_id = id(frame_tuple[0])
                    enter_time = self.timestamps[camera_id].get(frame_id, time.time())
                    queue_time = time.time() - enter_time
                    self.stats.queue_times.append(queue_time)
                    
                    # Clean up
                    if frame_id in self.timestamps[camera_id]:
                        del self.timestamps[camera_id][frame_id]
                    if frame_id in self.frame_metadata[camera_id]:
                        del self.frame_metadata[camera_id][frame_id]
                    
                    taken += 1
                    if taken >= max_batch_size:
                        break
                
            else:  # Default: 'fair' (round-robin)
                # Take frames in round-robin fashion to be fair to all cameras
                frames_per_camera = max(1, max_batch_size // len(cameras_with_frames))
                
                for camera_id in cameras_with_frames:
                    # Take up to frames_per_camera from this camera
                    frames_to_take = min(frames_per_camera, len(self.buffers[camera_id]))
                    
                    for _ in range(frames_to_take):
                        if not self.buffers[camera_id]:
                            break
                            
                        frame_tuple = self.buffers[camera_id].popleft()
                        batch.append(frame_tuple)
                        
                        # Track processing time
                        frame_id = id(frame_tuple[0])
                        enter_time = self.timestamps[camera_id].get(frame_id, time.time())
                        queue_time = time.time() - enter_time
                        self.stats.queue_times.append(queue_time)
                        
                        # Clean up
                        if frame_id in self.timestamps[camera_id]:
                            del self.timestamps[camera_id][frame_id]
                        if frame_id in self.frame_metadata[camera_id]:
                            del self.frame_metadata[camera_id][frame_id]
                        
                        if len(batch) >= max_batch_size:
                            break
                    
                    if len(batch) >= max_batch_size:
                        break
            
            # Update frame count
            for _ in batch:
                self.stats.frames_processed += 1
            
            return batch
    
    def get_buffer_status(self):
        """Get the current status of the buffer"""
        total_frames = sum(len(buffer) for buffer in self.buffers.values())
        per_camera = {camera_id: len(buffer) for camera_id, buffer in self.buffers.items()}
        
        return {
            "total_frames": total_frames,
            "per_camera": per_camera,
            "max_size_per_camera": self.max_size_per_camera,
            "max_total_size": self.max_total_size,
            "utilization_percent": (total_frames / self.max_total_size * 100) if self.max_total_size > 0 else 0
        }
    
    async def stop(self):
        """Stop the buffer monitor tasks"""
        self.active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Frame buffer stopped")