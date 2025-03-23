import os
import json
import time
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class ResultMonitor:
    """Monitor processing results to validate system performance"""
    
    def __init__(self, results_dir="./results", visualization_dir="./visualizations"):
        self.results_dir = results_dir
        self.visualization_dir = visualization_dir
        
        # Create directories if they don't exist
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.processing_times = defaultdict(list)
        self.detection_counts = defaultdict(list)
        self.classification_stats = defaultdict(lambda: defaultdict(int))
    
    def visualize_results(self, result_data, original_frame=None):
        """Create a visualization of tracking results"""
        if original_frame is None:
            # Create a blank canvas
            height, width = 1080, 1920
            image = Image.new('RGB', (width, height), color=(0, 0, 0))
        else:
            # Use the original frame
            if isinstance(original_frame, np.ndarray):
                # Convert OpenCV BGR to PIL RGB
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(original_frame)
            else:
                image = original_frame
        
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw metadata
        camera_id = result_data.get("camera_id", "unknown")
        store_id = result_data.get("store_id", "unknown")
        timestamp = result_data.get("timestamp", "unknown")
        frame_number = result_data.get("frame_number", 0)
        
        draw.text((10, 10), f"Store: {store_id}, Camera: {camera_id}", fill=(255, 255, 255), font=font)
        draw.text((10, 40), f"Frame: {frame_number}, Time: {timestamp}", fill=(255, 255, 255), font=font)
        draw.text((10, 70), f"People: {result_data.get('no_of_people', 0)}", fill=(255, 255, 255), font=font)
        
        # Draw classification counts
        singles = result_data.get("singles", 0)
        couples = result_data.get("couples", 0)
        groups = result_data.get("groups", 0)
        
        draw.text((10, 100), f"Singles: {singles}", fill=(0, 255, 0), font=font)
        draw.text((10, 130), f"Couples: {couples}", fill=(0, 255, 255), font=font)
        draw.text((10, 160), f"Groups: {groups}", fill=(255, 255, 0), font=font)
        
        # Draw entity coordinates
        entity_coordinates = result_data.get("entity_coordinates", [])
        
        # Map for colors based on status
        status_colors = {
            "alone": (0, 255, 0),      # Green
            "couple": (0, 255, 255),   # Cyan
            "group": (255, 255, 0),    # Yellow
            "undetermined": (150, 150, 150)  # Gray
        }
        
        for entity in entity_coordinates:
            person_id = entity.get("person_id", "?")
            x = entity.get("x_coord", 0)
            y = entity.get("y_coord", 0)
            status = entity.get("status", "undetermined")
            group_id = entity.get("group_id", "")
            
            # Get color based on status
            color = status_colors.get(status, (255, 255, 255))
            
            # Draw a dot at the position
            dot_radius = 5
            draw.ellipse(
                [(x-dot_radius, y-dot_radius), (x+dot_radius, y+dot_radius)],
                fill=color
            )
            
            # Draw ID and status
            text = f"ID: {person_id}"
            if status:
                text += f"\nStatus: {status}"
            if group_id:
                text += f"\nGroup: {group_id}"
                
            draw.text((x+10, y), text, fill=color, font=font)
        
        # Save the visualization
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{store_id}_{camera_id}_{frame_number}_{timestamp_str}.jpg"
        filepath = os.path.join(self.visualization_dir, filename)
        image.save(filepath)
        
        return filepath
    
    def process_result(self, result_data):
        """Process a single result message"""
        try:
            # Extract key information
            camera_id = result_data.get("camera_id", "unknown")
            store_id = result_data.get("store_id", "unknown")
            timestamp = result_data.get("timestamp", "")
            processed_timestamp = result_data.get("processed_timestamp", "")
            
            # Calculate processing time if both timestamps are available
            if timestamp and processed_timestamp:
                try:
                    start_time = datetime.fromisoformat(timestamp)
                    end_time = datetime.fromisoformat(processed_timestamp)
                    processing_time = (end_time - start_time).total_seconds()
                    self.processing_times[f"{store_id}_{camera_id}"].append(processing_time)
                except ValueError:
                    # Skip if timestamp parsing fails
                    pass
            
            # Record detection count
            people_count = result_data.get("no_of_people", 0)
            self.detection_counts[f"{store_id}_{camera_id}"].append(people_count)
            
            # Record classification statistics
            singles = result_data.get("singles", 0)
            couples = result_data.get("couples", 0)
            groups = result_data.get("groups", 0)
            
            self.classification_stats[f"{store_id}_{camera_id}"]["singles"] = singles
            self.classification_stats[f"{store_id}_{camera_id}"]["couples"] = couples
            self.classification_stats[f"{store_id}_{camera_id}"]["groups"] = groups
            
            # Visualize result
            viz_path = self.visualize_results(result_data)
            print(f"Visualization saved to: {viz_path}")
            
            # Save raw result
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{store_id}_{camera_id}_{timestamp_str}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)
                
            print(f"Processed result from store {store_id}, camera {camera_id}")
            print(f"  People: {people_count}, Singles: {singles}, Couples: {couples}, Groups: {groups}")
            
            if processing_time:
                print(f"  Processing time: {processing_time:.3f} seconds")
            
            return True
            
        except Exception as e:
            print(f"Error processing result: {e}")
            return False
    
    def generate_performance_report(self):
        """Generate performance report with metrics and visualizations"""
        if not self.processing_times and not self.detection_counts:
            print("No data available for performance report")
            return
        
        # Create report directory
        report_dir = os.path.join(self.results_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Processing time statistics
        if self.processing_times:
            plt.figure(figsize=(12, 6))
            for camera, times in self.processing_times.items():
                if times:
                    plt.plot(times, label=f"Camera {camera}")
            
            plt.xlabel("Frame Number")
            plt.ylabel("Processing Time (seconds)")
            plt.title("Processing Time per Frame")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(report_dir, "processing_times.png"))
            
            # Calculate statistics
            with open(os.path.join(report_dir, "processing_stats.txt"), 'w') as f:
                f.write("Processing Time Statistics:\n")
                f.write("===========================\n\n")
                
                for camera, times in self.processing_times.items():
                    if times:
                        f.write(f"Camera {camera}:\n")
                        f.write(f"  Average: {sum(times)/len(times):.3f} seconds\n")
                        f.write(f"  Min: {min(times):.3f} seconds\n")
                        f.write(f"  Max: {max(times):.3f} seconds\n")
                        f.write(f"  Total frames: {len(times)}\n\n")
        
        # Detection count statistics
        if self.detection_counts:
            plt.figure(figsize=(12, 6))
            for camera, counts in self.detection_counts.items():
                if counts:
                    plt.plot(counts, label=f"Camera {camera}")
            
            plt.xlabel("Frame Number")
            plt.ylabel("People Count")
            plt.title("People Detected per Frame")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(report_dir, "detection_counts.png"))
        
        # Classification statistics
        if self.classification_stats:
            cameras = list(self.classification_stats.keys())
            categories = ["singles", "couples", "groups"]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(cameras))
            width = 0.25
            
            for i, category in enumerate(categories):
                values = [self.classification_stats[camera][category] for camera in cameras]
                ax.bar(x + i*width, values, width, label=category.capitalize())
            
            ax.set_ylabel('Count')
            ax.set_title('Classification Results by Camera')
            ax.set_xticks(x + width)
            ax.set_xticklabels(cameras)
            ax.legend()
            
            plt.savefig(os.path.join(report_dir, "classification_stats.png"))
        
        print(f"Performance report generated in {report_dir}")

class MockResultGenerator:
    """Generate mock results for testing the monitor"""
    
    @staticmethod
    def generate_mock_result(store_id="001", camera_id="001", frame_id=0):
        """Generate a mock result message"""
        # Create timestamp slightly in the past
        timestamp = (datetime.now()).isoformat()
        processed_timestamp = datetime.now().isoformat()
        
        # Random number of people
        num_people = np.random.randint(0, 10)
        
        # Random classification counts
        singles = np.random.randint(0, num_people + 1)
        couples = np.random.randint(0, (num_people - singles) // 2 + 1)
        groups = np.random.randint(0, (num_people - singles - couples*2) // 3 + 1)
        
        # Generate entity coordinates
        entity_coordinates = []
        statuses = ["alone", "couple", "group", "undetermined"]
        for i in range(num_people):
            # Random position on screen
            x = np.random.randint(100, 1820)
            y = np.random.randint(100, 980)
            
            # Random status
            status_idx = np.random.randint(0, len(statuses))
            status = statuses[status_idx]
            
            # Group ID based on status
            group_id = ""
            if status == "couple":
                group_id = str(np.random.randint(1, 100))
            elif status == "group":
                group_id = str(np.random.randint(1, 100))
            
            entity_coordinates.append({
                "person_id": str(np.random.randint(1, 100)),
                "x_coord": x,
                "y_coord": y,
                "status": status,
                "group_id": group_id
            })
        
        # Create result message
        result = {
            "camera_id": camera_id,
            "store_id": store_id,
            "timestamp": timestamp,
            "processed_timestamp": processed_timestamp,
            "frame_number": frame_id,
            "resolution": "1920x1080",
            "is_organised": True,
            "no_of_people": num_people,
            "entity_coordinates": entity_coordinates,
            "singles": singles,
            "couples": couples,
            "groups": groups,
            "total_people": num_people
        }
        
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Result Monitoring Tool')
    parser.add_argument('--mode', choices=['monitor', 'mock'], default='monitor', 
                      help='Monitor mode: process real results or generate mocks')
    parser.add_argument('--results-dir', default='./results', 
                      help='Directory to save results')
    parser.add_argument('--viz-dir', default='./visualizations', 
                      help='Directory to save visualizations')
    parser.add_argument('--mock-count', type=int, default=10, 
                      help='Number of mock results to generate')
    parser.add_argument('--store-id', default="001", 
                      help='Store ID for mock results')
    parser.add_argument('--num-cameras', type=int, default=2, 
                      help='Number of cameras for mock results')
    
    args = parser.parse_args()
    
    monitor = ResultMonitor(args.results_dir, args.viz_dir)
    
    if args.mode == 'mock':
        print(f"Generating {args.mock_count} mock results...")
        for frame_id in range(args.mock_count):
            for camera_idx in range(args.num_cameras):
                camera_id = f"{camera_idx+1:03d}"
                mock_result = MockResultGenerator.generate_mock_result(
                    args.store_id, camera_id, frame_id
                )
                monitor.process_result(mock_result)
                time.sleep(0.1)  # Small delay between results
        
        monitor.generate_performance_report()
    else:
        print("Monitoring mode not implemented in this test script.")
        print("In a real system, this would listen to result messages.")