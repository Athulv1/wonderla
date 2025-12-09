"""
Head Counter with Zone-Based Counting
Uses two detection zones (upper and lower boxes) for IN/OUT counting
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
    """Improved object tracker with IoU and centroid tracking"""
    
    def __init__(self, max_disappeared=80, max_distance=100):
        self.next_object_id = 0
        self.objects = {}  # ID -> centroid
        self.bboxes = {}   # ID -> bbox
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance  # Maximum distance for matching
        
        # Track which zone each object was last seen in
        self.zone_history = {}  # ID -> list of zones ['upper', 'lower', ...]
        self.counted = set()  # IDs that have been counted
    
    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.zone_history[self.next_object_id] = []
        self.next_object_id += 1
    
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        if object_id in self.zone_history:
            del self.zone_history[object_id]
    
    def compute_iou(self, boxA, boxB):
        """Compute IoU between two bounding boxes"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
        return iou
    
    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # Calculate centroids and store bboxes
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
            object_bboxes = list(self.bboxes.values())
            
            # Calculate both distance and IoU
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Calculate IoU matrix
            iou_matrix = np.zeros((len(object_bboxes), len(input_bboxes)))
            for i, obj_bbox in enumerate(object_bboxes):
                for j, det_bbox in enumerate(input_bboxes):
                    iou_matrix[i, j] = self.compute_iou(obj_bbox, det_bbox)
            
            # Combine distance and IoU for better matching
            # Lower distance and higher IoU = better match
            # Normalize and combine (IoU weighted higher)
            D_norm = D / (D.max() + 1e-5)
            combined_score = D_norm - (iou_matrix * 1.5)  # IoU weighted 1.5x for better separation
            
            rows = combined_score.min(axis=1).argsort()
            cols = combined_score.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Only match if distance is reasonable OR IoU is good
                # Lower IoU threshold to prevent merging separate close heads
                if D[row, col] > self.max_distance and iou_matrix[row, col] < 0.25:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col])
        
        return self.objects
    
    def update_zone(self, object_id, zone):
        """Update the zone history for an object"""
        if object_id not in self.zone_history:
            self.zone_history[object_id] = []
        
        # Only add if it's a different zone than the last one
        if len(self.zone_history[object_id]) == 0 or self.zone_history[object_id][-1] != zone:
            self.zone_history[object_id].append(zone)


class HeadCounterZones:
    """Head counter using zone-based detection (upper and lower boxes)"""
    
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.20, iou_threshold=0.50, min_box_area=400, box_shrink=0.3):
        print(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = 0
        self.min_box_area = min_box_area  # Minimum bounding box area to filter tiny detections
        self.box_shrink = box_shrink  # Shrink bounding boxes by this ratio (0.3 = 30% smaller)
        
        print(f"✓ Model loaded successfully")
        print(f"  Mode: Zone-based counting (upper/lower boxes)")
        print(f"  Bounding box shrink: {int(box_shrink * 100)}%")
    
    def count_video(self, video_path, output_path=None, show=False, 
                    roi_config_file='head_counter_config.json', 
                    skip_frames=1, resize_width=None):
        
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
        
        # Load zones configuration
        upper_zone = None
        lower_zone = None
        
        if roi_config_file and Path(roi_config_file).exists():
            with open(roi_config_file, 'r') as f:
                roi_config = json.load(f)
                
                if 'upper_zone' in roi_config and 'lower_zone' in roi_config:
                    upper_zone = roi_config['upper_zone']
                    lower_zone = roi_config['lower_zone']
                    # Scale zones if resizing
                    if scale_factor != 1.0:
                        upper_zone = [int(c * scale_factor) for c in upper_zone]
                        lower_zone = [int(c * scale_factor) for c in lower_zone]
                    print(f"  Loaded upper zone: {upper_zone}")
                    print(f"  Loaded lower zone: {lower_zone}")
                elif 'y' in roi_config:
                    # Convert horizontal line to zones
                    y_line = int(roi_config['y'] * scale_factor) if scale_factor != 1.0 else roi_config['y']
                    zone_height = 100  # Height of each zone
                    w = resize_width if resize_width else width
                    h = resize_height if resize_height else height
                    
                    upper_zone = [0, max(0, y_line - zone_height), w, y_line]
                    lower_zone = [0, y_line, w, min(h, y_line + zone_height)]
                    print(f"  Created zones from horizontal line at y={y_line}")
        
        # Default zones if not configured - 50% split
        if upper_zone is None or lower_zone is None:
            h = resize_height if resize_height else height
            w = resize_width if resize_width else width
            mid_y = h // 2
            
            # Upper 50% of frame
            upper_zone = [0, 0, w, mid_y]
            # Lower 50% of frame
            lower_zone = [0, mid_y, w, h]
            print(f"  Using default 50/50 zones at Y={mid_y}")
        
        print(f"\nProcessing video: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Upper zone (IN): [{upper_zone[0]}, {upper_zone[1]}, {upper_zone[2]}, {upper_zone[3]}]")
        print(f"  Lower zone (OUT): [{lower_zone[0]}, {lower_zone[1]}, {lower_zone[2]}, {lower_zone[3]}]")
        
        if show:
            print(f"\n📺 LIVE PREVIEW MODE")
            print(f"  Press 'Q' to quit, 'P' to pause")
        
        # Initialize video writer
        writer = None
        output_size = (resize_width, resize_height) if resize_width else (width, height)
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, output_size)
        
        # Initialize tracker with improved parameters
        tracker = CentroidTracker(max_disappeared=100, max_distance=200)
        in_count = 0
        out_count = 0
        
        frame_count = 0
        processed_count = 0
        paused = False
        start_time = time.time()
        
        current_head_count = 0
        max_head_count = 0
        frame_results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
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
                classes=[self.person_class_id],
                verbose=False
            )[0]
            
            # Collect detections for tracking with size filtering
            detections_for_tracking = []
            current_head_count = 0
            
            for box in results.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                
                # Calculate box area
                box_width = bbox[2] - bbox[0]
                box_height = bbox[3] - bbox[1]
                box_area = box_width * box_height
                
                # Filter out very small detections (noise/partial detections)
                if box_area >= self.min_box_area:
                    # Shrink bounding box to reduce overlap
                    # Calculate center point
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    
                    # Reduce width and height by shrink ratio
                    new_width = box_width * (1 - self.box_shrink)
                    new_height = box_height * (1 - self.box_shrink)
                    
                    # Create smaller box around center
                    shrunken_bbox = [
                        cx - new_width / 2,  # x1
                        cy - new_height / 2,  # y1
                        cx + new_width / 2,  # x2
                        cy + new_height / 2   # y2
                    ]
                    
                    detections_for_tracking.append(shrunken_bbox)
                    current_head_count += 1
            
            if current_head_count > max_head_count:
                max_head_count = current_head_count
            
            # Update tracker
            objects = tracker.update(detections_for_tracking)
            
            # Check which zone each object is in
            for object_id, centroid in objects.items():
                cx, cy = centroid
                
                # Check if centroid is in upper zone
                if (upper_zone[0] <= cx <= upper_zone[2] and 
                    upper_zone[1] <= cy <= upper_zone[3]):
                    tracker.update_zone(object_id, 'upper')
                
                # Check if centroid is in lower zone
                elif (lower_zone[0] <= cx <= lower_zone[2] and 
                      lower_zone[1] <= cy <= lower_zone[3]):
                    tracker.update_zone(object_id, 'lower')
                
                # Check for counting events
                if object_id not in tracker.counted and len(tracker.zone_history[object_id]) >= 2:
                    zones = tracker.zone_history[object_id]
                    
                    # Upper to Lower = IN (entering from top)
                    if 'upper' in zones and 'lower' in zones:
                        if zones.index('upper') < zones.index('lower'):
                            in_count += 1
                            tracker.counted.add(object_id)
                            print(f"  IN: ID {object_id} (upper → lower)")
                    
                    # Lower to Upper = OUT (exiting to top)
                    if 'lower' in zones and 'upper' in zones:
                        if zones.index('lower') < zones.index('upper'):
                            out_count += 1
                            tracker.counted.add(object_id)
                            print(f"  OUT: ID {object_id} (lower → upper)")
            
            frame_results.append({
                'frame': frame_count,
                'head_count': current_head_count,
                'in_count': in_count,
                'out_count': out_count
            })
            
            # Annotate frame
            annotated = results.plot()
            
            # Draw zones
            cv2.rectangle(annotated, 
                         (upper_zone[0], upper_zone[1]), 
                         (upper_zone[2], upper_zone[3]), 
                         (0, 255, 0), 3)
            cv2.putText(annotated, "UPPER ZONE", 
                       (upper_zone[0] + 10, upper_zone[1] + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.rectangle(annotated, 
                         (lower_zone[0], lower_zone[1]), 
                         (lower_zone[2], lower_zone[3]), 
                         (0, 0, 255), 3)
            cv2.putText(annotated, "LOWER ZONE", 
                       (lower_zone[0] + 10, lower_zone[1] + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Draw tracked objects
            for object_id, centroid in objects.items():
                cx, cy = centroid
                cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(annotated, f"ID:{object_id}", (cx - 20, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
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
            
            if writer:
                writer.write(annotated)
            
            if show:
                # Resize for display if too large
                display_frame = annotated.copy()
                if annotated.shape[1] > 1280:
                    display_h = int(annotated.shape[0] * (1280 / annotated.shape[1]))
                    display_frame = cv2.resize(annotated, (1280, display_h))
                
                if paused:
                    cv2.rectangle(display_frame, (10, 120), (250, 150), (0, 0, 0), -1)
                    cv2.putText(display_frame, "PAUSED - Press 'P'", (20, 140),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow('Head Counter - Zones', display_frame)
                
                wait_time = 0 if paused else 1
                key = cv2.waitKey(wait_time) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n⏹️  Stopped by user")
                    break
                elif key == ord('p') or key == ord('P'):
                    paused = not paused
            
            if processed_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps_proc = processed_count / elapsed
                print(f"  Progress: {progress:.1f}% | FPS: {fps_proc:.1f} | Heads: {current_head_count} | IN: {in_count} | OUT: {out_count}")
        
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
            'zone_counting': {
                'in_count': in_count,
                'out_count': out_count,
                'net_count': in_count - out_count,
                'total_crossings': in_count + out_count
            },
            'head_statistics': {
                'max_heads': max_head_count,
                'avg_heads': avg_heads
            },
            'zones': {
                'upper': upper_zone,
                'lower': lower_zone
            }
        }
        
        print(f"\n✓ Video processing complete")
        print(f"  ZONE CROSSING COUNTS:")
        print(f"    IN:  {in_count} (upper → lower)")
        print(f"    OUT: {out_count} (lower → upper)")
        print(f"    Net: {in_count - out_count}")
        print(f"  HEAD STATISTICS:")
        print(f"    Maximum heads: {max_head_count}")
        print(f"    Average heads: {avg_heads:.1f}")
        print(f"  Processing FPS: {avg_fps:.2f}")
        
        if output_path:
            print(f"  Output saved to: {output_path}")
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Head Counter with Zone-Based Counting')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.20, help='Confidence threshold')
    parser.add_argument('--min-size', type=int, default=400, help='Minimum detection box area (pixels)')
    parser.add_argument('--box-shrink', type=float, default=0.3, help='Shrink bounding boxes by ratio (0.3 = 30%% smaller)')
    parser.add_argument('--roi-config', type=str, default='head_counter_config.json',
                        help='Path to ROI config file')
    parser.add_argument('--show', action='store_true', help='Show live preview')
    parser.add_argument('--skip-frames', type=int, default=1, help='Process every N frames')
    parser.add_argument('--resize', type=int, help='Resize width')
    parser.add_argument('--save-json', type=str, help='Save results to JSON')
    
    args = parser.parse_args()
    
    counter = HeadCounterZones(model_name=args.model, conf_threshold=args.conf, min_box_area=args.min_size, box_shrink=args.box_shrink)
    
    results = counter.count_video(
        video_path=args.video,
        output_path=args.output,
        show=args.show,
        roi_config_file=args.roi_config,
        skip_frames=args.skip_frames,
        resize_width=args.resize
    )
    
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.save_json}")


if __name__ == "__main__":
    main()
