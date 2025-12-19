"""
Real-time Head Counter for RTSP Camera Streams
Optimized for live camera feeds with zone-based counting
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import json
import os
from collections import defaultdict
from scipy.spatial import distance as dist
import time


class CentroidTracker:
    """Track objects across frames using centroids and bounding boxes"""
    
    def __init__(self, max_disappeared=50, max_distance=80):
        self.next_object_id = 0
        self.objects = {}  # Centroids
        self.bboxes = {}  # Bounding boxes for better tracking
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        
    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
            
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        input_bboxes = []
        for (i, (x1, y1, x2, y2)) in enumerate(detections):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
            input_bboxes.append((x1, y1, x2, y2))
            
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, input_bboxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                if D[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
                
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                    
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col])
                
        return self.objects


class RTSPHeadCounter:
    """Real-time head counter for RTSP streams"""
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.35):
        """Initialize the head counter"""
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Check if GPU is available
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✓ Model loaded successfully on {self.device.upper()}")
        if self.device == 'cpu':
            print("  ⚠ Running on CPU - consider using GPU for better performance")
        
    def count_rtsp_stream(self, rtsp_url, upper_zone=None, lower_zone=None, 
                         box_shrink=0.4, save_output=None, show=True):
        """
        Count heads from RTSP stream
        
        Args:
            rtsp_url: RTSP stream URL
            upper_zone: [x1, y1, x2, y2] coordinates for upper zone
            lower_zone: [x1, y1, x2, y2] coordinates for lower zone
            box_shrink: Shrink detection boxes by this factor (0.0-1.0)
            save_output: Path to save output video (optional)
            show: Show live preview window
        """
        
        # Open RTSP stream with optimized settings
        print(f"\nConnecting to RTSP stream: {rtsp_url}")
        
        # Use FFMPEG backend with optimized parameters + error suppression
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;2048000|fflags;nobuffer|flags;low_delay"
        os.environ["OPENCV_LOG_LEVEL"] = "ERROR"  # Suppress H.264 decoding warnings
        
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # Set stream properties for better stability
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer to reduce lag and corrupted frames
        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        
        if not cap.isOpened():
            raise ValueError(f"Could not connect to RTSP stream: {rtsp_url}")
            
        # Get stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        
        print(f"✓ Stream connected successfully")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        
        # Load or set default zones
        if upper_zone is None or lower_zone is None:
            config_file = 'head_counter_config.json'
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    upper_zone = config.get('upper_zone', [0, 0, width-2, height//3])
                    lower_zone = config.get('lower_zone', [1, height//3+3, width-6, height-5])
                    print(f"  Loaded zones from: {config_file}")
            else:
                upper_zone = [0, 0, width-2, height//3]
                lower_zone = [1, height//3+3, width-6, height-5]
                print(f"  Using default zones")
        
        print(f"  Upper zone (IN): {upper_zone}")
        print(f"  Lower zone (OUT): {lower_zone}")
        
        # Initialize tracker and counters with improved settings
        tracker = CentroidTracker(max_disappeared=50, max_distance=80)
        in_count = 0
        out_count = 0
        tracked_states = {}
        counted_ids = set()
        
        # Video writer if saving
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
            print(f"  Saving output to: {save_output}")
        
        print(f"\n📺 LIVE RTSP STREAM")
        print(f"  Press 'Q' to quit, 'P' to pause")
        
        frame_count = 0
        process_count = 0
        start_time = time.time()
        paused = False
        last_detections = []
        last_objects = {}
        last_valid_frame = None  # Backup frame for stream errors
        reconnect_attempts = 0
        max_reconnects = 10
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    
                    if not ret or frame is None:
                        print(f"\n⚠ Stream disconnected (attempt {reconnect_attempts + 1}/{max_reconnects})")
                        
                        # Use last valid frame to keep display alive
                        if last_valid_frame is not None:
                            frame = last_valid_frame
                        
                        # Try to reconnect
                        print("Attempting to reconnect...")
                        cap.release()
                        time.sleep(1)
                        
                        reconnect_attempts += 1
                        if reconnect_attempts >= max_reconnects:
                            print("Max reconnection attempts reached. Exiting.")
                            break
                        
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;2048000|fflags;nobuffer|flags;low_delay"
                        os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
                        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FPS, 15)
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                        continue
                    
                    # Valid frame received
                    reconnect_attempts = 0
                    
                    # Flip frame horizontally
                    frame = cv2.flip(frame, 1)
                    
                    last_valid_frame = frame.copy()
                    frame_count += 1
                    process_count += 1
                    
                    # Process every frame with GPU
                    try:
                        # Run detection with GPU-optimized settings
                        results = self.model(frame, conf=self.conf_threshold, verbose=False, 
                                           imgsz=640, device=self.device, half=True)  # Use FP16 for speed
                        
                        # Extract detections
                        detections = []
                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                
                                # Shrink bounding box
                                if box_shrink > 0:
                                    w = x2 - x1
                                    h = y2 - y1
                                    shrink_w = w * box_shrink / 2
                                    shrink_h = h * box_shrink / 2
                                    x1, y1 = x1 + shrink_w, y1 + shrink_h
                                    x2, y2 = x2 - shrink_w, y2 - shrink_h
                                
                                detections.append([x1, y1, x2, y2])
                        
                        # Update tracker
                        objects = tracker.update(detections)
                        
                        # Cache results for error recovery
                        last_detections = detections
                        last_objects = objects
                        
                    except Exception as e:
                        # Skip processing on error, use cached results
                        print(f"Frame processing error (using cached): {str(e)[:50]}")
                        detections = last_detections
                        objects = last_objects
                    
                    # Draw detections (lighter weight)
                    for (x1, y1, x2, y2) in detections:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                    (0, 255, 0), 1)  # Thinner lines
                    
                    # Check zone transitions
                    for object_id, centroid in objects.items():
                        cx, cy = centroid
                        
                        # Initialize tracking state
                        if object_id not in tracked_states:
                            tracked_states[object_id] = None
                        
                        # Determine current zone
                        in_upper = (upper_zone[0] <= cx <= upper_zone[2] and 
                                   upper_zone[1] <= cy <= upper_zone[3])
                        in_lower = (lower_zone[0] <= cx <= lower_zone[2] and 
                                   lower_zone[1] <= cy <= lower_zone[3])
                        
                        current_zone = 'upper' if in_upper else ('lower' if in_lower else None)
                        previous_zone = tracked_states[object_id]
                        
                        # Detect crossing
                        if object_id not in counted_ids and current_zone and previous_zone:
                            if previous_zone == 'lower' and current_zone == 'upper':
                                in_count += 1
                                counted_ids.add(object_id)
                                print(f"  IN: ID {object_id} (lower → upper)")
                            elif previous_zone == 'upper' and current_zone == 'lower':
                                out_count += 1
                                counted_ids.add(object_id)
                                print(f"  OUT: ID {object_id} (upper → lower)")
                        
                        tracked_states[object_id] = current_zone
                        
                        # Draw centroid and ID
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, f"ID:{object_id}", (cx - 10, cy - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                    # Draw zones
                    cv2.rectangle(frame, (upper_zone[0], upper_zone[1]), 
                                (upper_zone[2], upper_zone[3]), (255, 0, 0), 2)
                    cv2.rectangle(frame, (lower_zone[0], lower_zone[1]), 
                                (lower_zone[2], lower_zone[3]), (0, 0, 255), 2)
                    
                    # Draw counts with better visibility
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    process_fps = process_count / elapsed if elapsed > 0 else 0
                    
                    # Simplified overlay (less processing)
                    cv2.rectangle(frame, (10, 10), (250, 190), (0, 0, 0), -1)
                    
                    cv2.putText(frame, f"IN: {in_count}", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(frame, f"OUT: {out_count}", (20, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.putText(frame, f"HEADS: {len(objects)}", (20, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 155),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Process: {process_fps:.1f}", (20, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Write frame
                    if writer:
                        writer.write(frame)
                
                # Show frame
                if show:
                    # Resize for display if too small
                    display_frame = frame
                    if width < 640:
                        scale = 640 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_frame = cv2.resize(frame, (new_width, new_height))
                    
                    cv2.imshow('RTSP Head Counter', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        print("\n⏹ Stopped by user")
                        break
                    elif key == ord('p') or key == ord('P'):
                        paused = not paused
                        print(f"{'⏸ Paused' if paused else '▶ Resumed'}")
                else:
                    # Small delay to prevent CPU overload when not showing
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n⏹ Stopped by user (Ctrl+C)")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print(f"\n{'='*50}")
            print(f"FINAL STATISTICS")
            print(f"{'='*50}")
            print(f"Total IN:  {in_count}")
            print(f"Total OUT: {out_count}")
            print(f"Net Count: {in_count - out_count}")
            print(f"Frames Processed: {frame_count}")
            print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")
            print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='RTSP Head Counter with Zone-based Tracking')
    parser.add_argument('--rtsp', required=True, help='RTSP stream URL')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--box-shrink', type=float, default=0.4, 
                       help='Shrink detection boxes (0.0-1.0)')
    parser.add_argument('--output', help='Save output video to file')
    parser.add_argument('--show', action='store_true', default=True,
                       help='Show live preview (default: True)')
    parser.add_argument('--no-show', action='store_true', help='Disable live preview')
    
    args = parser.parse_args()
    
    # Initialize counter
    counter = RTSPHeadCounter(model_path=args.model, conf_threshold=args.conf)
    
    # Run counting
    show = args.show and not args.no_show
    counter.count_rtsp_stream(
        rtsp_url=args.rtsp,
        box_shrink=args.box_shrink,
        save_output=args.output,
        show=show
    )


if __name__ == "__main__":
    main()
