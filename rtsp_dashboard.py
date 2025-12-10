"""
Modern Dashboard for RTSP Head Counter
Displays real-time statistics with visual banners
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from collections import defaultdict
from scipy.spatial import distance as dist
import time
import threading


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


class RTSPStreamProcessor:
    """Process RTSP stream with head counting"""
    
    def __init__(self, rtsp_url, model_path='yolov8s.pt', conf_threshold=0.25, box_shrink=0.4):
        self.rtsp_url = rtsp_url
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.box_shrink = box_shrink
        
        # Counters
        self.in_count = 0
        self.out_count = 0
        self.pool_count = 0
        self.current_heads = 0
        
        # Tracking with optimized parameters for higher resolution
        self.tracker = CentroidTracker(max_disappeared=60, max_distance=100)
        self.tracked_states = {}
        self.counted_ids = set()
        
        # Stream state
        self.cap = None
        self.frame = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Stats
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Load zones
        self.load_zones()
        
    def load_zones(self):
        """Load zone configuration"""
        config_file = 'head_counter_config.json'
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.upper_zone = config.get('upper_zone', [0, 0, 640, 120])
                self.lower_zone = config.get('lower_zone', [0, 120, 640, 288])
        else:
            self.upper_zone = [0, 0, 640, 120]
            self.lower_zone = [0, 120, 640, 288]
    
    def connect_stream(self):
        """Connect to RTSP stream with error recovery"""
        # Suppress H.264 decoding errors
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;2048000|fflags;nobuffer|flags;low_delay"
        os.environ["OPENCV_LOG_LEVEL"] = "ERROR"  # Suppress verbose errors
        
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Optimized settings for stability
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer to reduce lag
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability
        
        return self.cap.isOpened()
    
    def process_frame(self, frame):
        """Process a single frame with error handling"""
        try:
            # Run detection with optimized GPU settings
            results = self.model(frame, conf=self.conf_threshold, verbose=False, 
                               imgsz=1024, device='cuda', half=True, 
                               stream_buffer=True, max_det=50)
            
            # Extract detections
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Shrink bounding box
                    if self.box_shrink > 0:
                        w = x2 - x1
                        h = y2 - y1
                        shrink_w = w * self.box_shrink / 2
                        shrink_h = h * self.box_shrink / 2
                        x1, y1 = x1 + shrink_w, y1 + shrink_h
                        x2, y2 = x2 - shrink_w, y2 - shrink_h
                    
                    detections.append([x1, y1, x2, y2])
                    
                    # Draw detection
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 255, 0), 2)
            
            # Update tracker
            objects = self.tracker.update(detections)
            self.current_heads = len(objects)
            
            # Check zone transitions
            for object_id, centroid in objects.items():
                cx, cy = centroid
                
                # Initialize tracking state
                if object_id not in self.tracked_states:
                    self.tracked_states[object_id] = None
                
                # Determine current zone
                in_upper = (self.upper_zone[0] <= cx <= self.upper_zone[2] and 
                           self.upper_zone[1] <= cy <= self.upper_zone[3])
                in_lower = (self.lower_zone[0] <= cx <= self.lower_zone[2] and 
                           self.lower_zone[1] <= cy <= self.lower_zone[3])
                
                current_zone = 'upper' if in_upper else ('lower' if in_lower else None)
                previous_zone = self.tracked_states[object_id]
                
                # Detect crossing (adjusted for vertical flip)
                if object_id not in self.counted_ids and current_zone and previous_zone:
                    if previous_zone == 'upper' and current_zone == 'lower':
                        self.in_count += 1
                        self.counted_ids.add(object_id)
                    elif previous_zone == 'lower' and current_zone == 'upper':
                        self.out_count += 1
                        self.counted_ids.add(object_id)
                
                self.tracked_states[object_id] = current_zone
                
                # Draw centroid and ID
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{object_id}", (cx - 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw zones
            cv2.rectangle(frame, (self.upper_zone[0], self.upper_zone[1]), 
                        (self.upper_zone[2], self.upper_zone[3]), (255, 0, 0), 2)
            cv2.rectangle(frame, (self.lower_zone[0], self.lower_zone[1]), 
                        (self.lower_zone[2], self.lower_zone[3]), (0, 0, 255), 2)
            
            # Calculate pool count
            self.pool_count = self.in_count - self.out_count
            
            return frame
            
        except Exception as e:
            # Return original frame if processing fails
            print(f"Frame processing error (skipping): {str(e)[:50]}")
            return frame
    
    def start(self):
        """Start processing stream"""
        self.is_running = True
        thread = threading.Thread(target=self._process_loop, daemon=True)
        thread.start()
    
    def _process_loop(self):
        """Main processing loop with robust error handling"""
        reconnect_attempts = 0
        max_reconnects = 10
        last_valid_frame = None
        
        if not self.connect_stream():
            print("Failed to connect to RTSP stream")
            return
        
        print(f"✓ Connected to RTSP stream")
        
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                print(f"Stream disconnected (attempt {reconnect_attempts + 1}/{max_reconnects})")
                
                # Use last valid frame while reconnecting
                if last_valid_frame is not None:
                    with self.lock:
                        self.frame = last_valid_frame
                
                self.cap.release()
                time.sleep(1)  # Shorter wait
                
                reconnect_attempts += 1
                if reconnect_attempts >= max_reconnects:
                    print("Max reconnect attempts reached")
                    break
                
                if not self.connect_stream():
                    continue
                else:
                    reconnect_attempts = 0  # Reset on successful connect
                continue
            
            # Valid frame received
            reconnect_attempts = 0
            
            # Flip frame vertically
            frame = cv2.flip(frame, 0)
            self.frame_count += 1
            last_valid_frame = frame.copy()  # Keep backup
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Calculate FPS
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            # Store frame
            with self.lock:
                self.frame = processed_frame
    
    def get_frame(self):
        """Get current frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_stats(self):
        """Get current statistics"""
        return {
            'in_count': self.in_count,
            'out_count': self.out_count,
            'pool_count': self.pool_count,
            'current_heads': self.current_heads,
            'fps': round(self.fps, 1),
            'timestamp': time.time()
        }
    
    def stop(self):
        """Stop processing"""
        self.is_running = False
        if self.cap:
            self.cap.release()


# Flask application
app = Flask(__name__)

# Global processor
processor = None


def generate_frames():
    """Generate video frames for streaming"""
    global processor
    
    while True:
        if processor is None:
            time.sleep(0.1)
            continue
        
        frame = processor.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('rtsp_dashboard.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    """Get current statistics"""
    global processor
    if processor is None:
        return jsonify({
            'in_count': 0,
            'out_count': 0,
            'pool_count': 0,
            'current_heads': 0,
            'fps': 0,
            'timestamp': time.time()
        })
    return jsonify(processor.get_stats())


@app.route('/reset', methods=['POST'])
def reset_counts():
    """Reset all counters"""
    global processor
    if processor:
        processor.in_count = 0
        processor.out_count = 0
        processor.pool_count = 0
        processor.counted_ids.clear()
        return jsonify({'success': True, 'message': 'Counters reset successfully'})
    return jsonify({'success': False, 'error': 'No active processor'})


@app.route('/health')
def health():
    """Health check endpoint"""
    global processor
    return jsonify({
        'status': 'running',
        'processor_active': processor is not None and processor.is_running,
        'timestamp': time.time()
    })


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RTSP Head Counter Dashboard')
    parser.add_argument('--rtsp', required=True, help='RTSP stream URL')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--box-shrink', type=float, default=0.4, help='Box shrink factor')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    
    args = parser.parse_args()
    
    # Initialize processor
    global processor
    processor = RTSPStreamProcessor(
        rtsp_url=args.rtsp,
        model_path=args.model,
        conf_threshold=args.conf,
        box_shrink=args.box_shrink
    )
    
    # Start processing
    processor.start()
    
    # Run Flask app
    print(f"\n{'='*60}")
    print(f"🎯 RTSP Head Counter Dashboard")
    print(f"{'='*60}")
    print(f"📺 Stream: {args.rtsp}")
    print(f"🌐 Dashboard: http://{args.host}:{args.port}")
    print(f"{'='*60}\n")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
