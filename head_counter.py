"""
Head Counter for Top-View Camera
Uses YOLOv8 pre-trained model to detect and count people (heads) from overhead camera angle
Includes ROI line crossing for IN/OUT counting
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import time
from scipy.spatial import distance as dist


class CentroidTracker:
    """Simple centroid-based object tracker for counting line crossings"""
    
    def __init__(self, max_disappeared=50):
        """
        Initialize the centroid tracker
        
        Args:
            max_disappeared: Maximum frames an object can be missing before being deregistered
        """
        self.next_object_id = 0
        self.objects = {}  # ID -> centroid
        self.disappeared = {}  # ID -> frame count
        self.max_disappeared = max_disappeared
        
        # Track crossing state for each object
        self.crossed = {}  # ID -> {'crossed': bool, 'direction': 'in'/'out', 'start_side': 'top'/'bottom'}
    
    def register(self, centroid):
        """Register a new object with a unique ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.crossed[self.next_object_id] = {'crossed': False, 'direction': None, 'start_side': None}
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.crossed:
            del self.crossed[object_id]
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of bounding boxes [(x1, y1, x2, y2), ...]
            
        Returns:
            Dictionary of object_id -> centroid
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # Calculate centroids from bounding boxes
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(detections):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
        
        # If no objects tracked yet, register all
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calculate distance between existing and new centroids
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Find minimum distance matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Update object position
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle disappeared objects
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])
        
        return self.objects


class HeadCounter:
    """Head detector and counter using YOLOv8 pre-trained model"""
    
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the head counter
        
        Args:
            model_name: YOLOv8 model name (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        print(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # COCO class ID for 'person' is 0
        self.person_class_id = 0
        
        print(f"✓ Model loaded successfully")
        print(f"  Confidence threshold: {conf_threshold}")
        print(f"  IOU threshold: {iou_threshold}")
        print(f"  Detecting class: person (for head counting)")
    
    def count_video(self, video_path, output_path=None, show=False, 
                    roi_config_file='head_counter_config.json', 
                    skip_frames=1, resize_width=None):
        """
        Count heads in video with ROI line crossing
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            show: Whether to display real-time preview
            roi_config_file: Path to ROI configuration JSON file
            skip_frames: Process every N frames (1 = process all frames)
            resize_width: Resize frame width for faster processing
            
        Returns:
            Dictionary with counting statistics
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate resize dimensions
        resize_height = height
        scale_factor = 1.0
        if resize_width and resize_width < width:
            resize_height = int(height * (resize_width / width))
            scale_factor = resize_width / width
            print(f"  Resizing from {width}x{height} to {resize_width}x{resize_height}")
        
        # Load ROI configuration
        roi_line = None
        if roi_config_file and Path(roi_config_file).exists():
            with open(roi_config_file, 'r') as f:
                roi_config = json.load(f)
                config_type = roi_config.get('type', 'custom')
                
                if config_type == 'horizontal':
                    y_pos = roi_config.get('y')
                    roi_line = {'y': int(y_pos * scale_factor)}
                    print(f"  Loaded horizontal ROI line at Y={y_pos}")
                elif config_type == 'vertical':
                    x_pos = roi_config.get('x')
                    roi_line = {'x': int(x_pos * scale_factor)}
                    print(f"  Loaded vertical ROI line at X={x_pos}")
                elif config_type == 'custom':
                    line_points = roi_config.get('line_points')
                    if line_points:
                        p1 = (int(line_points[0][0] * scale_factor), int(line_points[0][1] * scale_factor))
                        p2 = (int(line_points[1][0] * scale_factor), int(line_points[1][1] * scale_factor))
                        roi_line = {'line_points': [p1, p2]}
                        print(f"  Loaded custom ROI line")
        
        # Default ROI line if not configured
        if roi_line is None:
            roi_line = {'y': resize_height // 2 if resize_width else height // 2}
            print(f"  Using default horizontal ROI line at center")
        
        # Determine line type
        if 'line_points' in roi_line:
            is_custom_line = True
            is_horizontal = False
            line_pos = None
            line_p1, line_p2 = roi_line['line_points']
        else:
            is_custom_line = False
            is_horizontal = 'y' in roi_line
            line_pos = roi_line.get('y') if is_horizontal else roi_line.get('x')
            line_p1, line_p2 = None, None
        
        print(f"\nProcessing video: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        if show:
            print(f"\n📺 LIVE PREVIEW MODE")
            print(f"  Press 'Q' to quit")
            print(f"  Press 'P' to pause/resume")
        
        # Initialize video writer
        writer = None
        output_size = (resize_width, resize_height) if resize_width else (width, height)
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, output_size)
        
        # Initialize tracker
        tracker = CentroidTracker(max_disappeared=30)
        in_count = 0
        out_count = 0
        counted_ids = set()
        
        frame_count = 0
        processed_count = 0
        paused = False
        start_time = time.time()
        
        # Statistics
        frame_results = []
        current_head_count = 0
        max_head_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for speed
            if frame_count % skip_frames != 0:
                continue
            
            processed_count += 1
            
            # Resize if needed
            if resize_width and resize_width < width:
                frame_resized = cv2.resize(frame, (resize_width, resize_height))
            else:
                frame_resized = frame
            
            # Run YOLOv8 detection
            results = self.model(
                frame_resized,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[self.person_class_id],  # Only detect 'person' class
                verbose=False
            )[0]
            
            # Collect detections for tracking
            detections_for_tracking = []
            current_head_count = 0
            
            for box in results.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                detections_for_tracking.append(bbox)
                current_head_count += 1
            
            # Update max head count
            if current_head_count > max_head_count:
                max_head_count = current_head_count
            
            # Update tracker
            objects = tracker.update(detections_for_tracking)
            
            # Check line crossings
            for object_id, centroid in objects.items():
                cx, cy = centroid
                
                # Initialize tracking data
                if object_id not in tracker.crossed:
                    tracker.crossed[object_id] = {'crossed': False, 'direction': None, 'start_side': None}
                
                # Determine side
                if is_custom_line:
                    v1 = (line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])
                    v2 = (cx - line_p1[0], cy - line_p1[1])
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    current_side = 'left' if cross > 0 else 'right'
                elif is_horizontal:
                    current_side = 'top' if cy < line_pos else 'bottom'
                else:
                    current_side = 'left' if cx < line_pos else 'right'
                
                # Set initial side
                if tracker.crossed[object_id]['start_side'] is None:
                    tracker.crossed[object_id]['start_side'] = current_side
                
                # Detect crossing
                if object_id not in counted_ids:
                    start_side = tracker.crossed[object_id]['start_side']
                    
                    if is_custom_line or not is_horizontal:
                        if start_side == 'left' and current_side == 'right':
                            out_count += 1
                            counted_ids.add(object_id)
                        elif start_side == 'right' and current_side == 'left':
                            in_count += 1
                            counted_ids.add(object_id)
                    else:
                        if start_side == 'top' and current_side == 'bottom':
                            out_count += 1
                            counted_ids.add(object_id)
                        elif start_side == 'bottom' and current_side == 'top':
                            in_count += 1
                            counted_ids.add(object_id)
            
            # Store frame results
            frame_results.append({
                'frame': frame_count,
                'head_count': current_head_count,
                'in_count': in_count,
                'out_count': out_count
            })
            
            # Annotate frame
            annotated = results.plot()
            
            # Draw ROI line
            if is_custom_line:
                cv2.line(annotated, line_p1, line_p2, (0, 255, 255), 3)
                cv2.circle(annotated, line_p1, 8, (0, 255, 0), -1)
                cv2.circle(annotated, line_p2, 8, (0, 0, 255), -1)
                cv2.putText(annotated, "IN", (line_p1[0] - 30, line_p1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated, "OUT", (line_p2[0] + 10, line_p2[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif is_horizontal:
                cv2.line(annotated, (0, line_pos), (annotated.shape[1], line_pos), (0, 255, 255), 3)
            else:
                cv2.line(annotated, (line_pos, 0), (line_pos, annotated.shape[0]), (0, 255, 255), 3)
            
            # Draw tracked objects
            for object_id, centroid in objects.items():
                cx, cy = centroid
                cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(annotated, f"ID:{object_id}", (cx - 20, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add statistics overlay
            cv2.rectangle(annotated, (10, 10), (750, 110), (0, 0, 0), -1)
            text1 = f"Frame {frame_count}/{total_frames}"
            text2 = f"Current Heads: {current_head_count} | Max: {max_head_count}"
            text3 = f"IN: {in_count} | OUT: {out_count} | Net: {in_count - out_count}"
            
            cv2.putText(annotated, text1, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, text2, (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated, text3, (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame
            if writer:
                writer.write(annotated)
            
            # Show frame
            if show:
                if paused:
                    cv2.rectangle(annotated, (10, 120), (250, 150), (0, 0, 0), -1)
                    cv2.putText(annotated, "PAUSED - Press 'P'", (20, 140),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow('Head Counter', annotated)
                
                wait_time = 0 if paused else 1
                key = cv2.waitKey(wait_time) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n⏹️  Stopped by user")
                    break
                elif key == ord('p') or key == ord('P'):
                    paused = not paused
            
            # Progress update
            if processed_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps_proc = processed_count / elapsed
                print(f"  Progress: {progress:.1f}% | FPS: {fps_proc:.1f} | Heads: {current_head_count}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        avg_fps = processed_count / elapsed_time
        avg_heads = sum([r['head_count'] for r in frame_results]) / len(frame_results) if frame_results else 0
        
        result = {
            'video': str(video_path),
            'total_frames': frame_count,
            'processed_frames': processed_count,
            'processing_time': elapsed_time,
            'avg_fps': avg_fps,
            'line_crossing': {
                'in_count': in_count,
                'out_count': out_count,
                'net_count': in_count - out_count,
                'total_crossings': in_count + out_count
            },
            'head_statistics': {
                'max_heads': max_head_count,
                'avg_heads': avg_heads
            },
            'roi_line': roi_line
        }
        
        print(f"\n✓ Video processing complete")
        print(f"  LINE CROSSING COUNTS:")
        print(f"    IN:  {in_count}")
        print(f"    OUT: {out_count}")
        print(f"    Net: {in_count - out_count}")
        print(f"  HEAD STATISTICS:")
        print(f"    Maximum heads detected: {max_head_count}")
        print(f"    Average heads per frame: {avg_heads:.1f}")
        print(f"  Processing FPS: {avg_fps:.2f}")
        
        if output_path:
            print(f"  Output saved to: {output_path}")
        
        return result


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Head Counter for Top-View Camera')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                        help='YOLOv8 model (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--roi-config', type=str, default='head_counter_config.json',
                        help='Path to ROI config file')
    parser.add_argument('--show', action='store_true', help='Show live preview')
    parser.add_argument('--skip-frames', type=int, default=1, 
                        help='Process every N frames (1=all frames)')
    parser.add_argument('--resize', type=int, help='Resize width for faster processing')
    parser.add_argument('--save-json', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize counter
    counter = HeadCounter(model_name=args.model, conf_threshold=args.conf)
    
    # Process video
    results = counter.count_video(
        video_path=args.video,
        output_path=args.output,
        show=args.show,
        roi_config_file=args.roi_config,
        skip_frames=args.skip_frames,
        resize_width=args.resize
    )
    
    # Save JSON results
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.save_json}")


if __name__ == "__main__":
    main()
