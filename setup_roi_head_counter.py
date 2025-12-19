"""
Interactive ROI Line Setup for Head Counter
Draw counting lines on your video to set up IN/OUT counting zones
"""

import cv2
import json
import sys
from pathlib import Path


class ROISetup:
    """Interactive ROI line drawing tool"""
    
    def __init__(self, video_path):
        """
        Initialize ROI setup
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Read first frame
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        self.original_frame = self.frame.copy()
        self.height, self.width = self.frame.shape[:2]
        
        # ROI line variables
        self.drawing = False
        self.line_type = None  # 'horizontal', 'vertical', or 'custom'
        self.line_pos = None
        self.line_points = []  # For custom line: [(x1, y1), (x2, y2)]
        
        print("\n" + "="*70)
        print("🎯 HEAD COUNTER - ROI LINE SETUP")
        print("="*70)
        print(f"Video: {video_path}")
        print(f"Resolution: {self.width}x{self.height}")
        print("="*70)
        print("\nCHOOSE LINE TYPE:")
        print("  [H] - Horizontal line (for vertical movement)")
        print("  [V] - Vertical line (for horizontal movement)")
        print("  [C] - Custom line (any angle)")
        print("\nCONTROLS:")
        print("  Click and drag to draw the line")
        print("  [R] - Reset line")
        print("  [S] - Save configuration")
        print("  [Q] - Quit without saving")
        print("="*70 + "\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing ROI line"""
        
        if self.line_type == 'horizontal':
            # Horizontal line - only Y coordinate matters
            if event == cv2.EVENT_MOUSEMOVE:
                self.frame = self.original_frame.copy()
                cv2.line(self.frame, (0, y), (self.width, y), (0, 255, 255), 2)
                cv2.putText(self.frame, f"Y={y}", (10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            elif event == cv2.EVENT_LBUTTONDOWN:
                self.line_pos = y
                self.frame = self.original_frame.copy()
                cv2.line(self.frame, (0, y), (self.width, y), (0, 255, 0), 3)
                cv2.putText(self.frame, f"Y={y} ✓", (10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"✓ Horizontal line set at Y={y}")
        
        elif self.line_type == 'vertical':
            # Vertical line - only X coordinate matters
            if event == cv2.EVENT_MOUSEMOVE:
                self.frame = self.original_frame.copy()
                cv2.line(self.frame, (x, 0), (x, self.height), (0, 255, 255), 2)
                cv2.putText(self.frame, f"X={x}", (x + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            elif event == cv2.EVENT_LBUTTONDOWN:
                self.line_pos = x
                self.frame = self.original_frame.copy()
                cv2.line(self.frame, (x, 0), (x, self.height), (0, 255, 0), 3)
                cv2.putText(self.frame, f"X={x} ✓", (x + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"✓ Vertical line set at X={x}")
        
        elif self.line_type == 'custom':
            # Custom line - two points
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.line_points = [(x, y)]
                self.frame = self.original_frame.copy()
                cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
            
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                temp_frame = self.original_frame.copy()
                cv2.circle(temp_frame, self.line_points[0], 5, (0, 255, 0), -1)
                cv2.line(temp_frame, self.line_points[0], (x, y), (0, 255, 255), 2)
                cv2.circle(temp_frame, (x, y), 5, (0, 0, 255), -1)
                self.frame = temp_frame
            
            elif event == cv2.EVENT_LBUTTONUP and self.drawing:
                self.drawing = False
                self.line_points.append((x, y))
                self.frame = self.original_frame.copy()
                cv2.circle(self.frame, self.line_points[0], 8, (0, 255, 0), -1)
                cv2.circle(self.frame, self.line_points[1], 8, (0, 0, 255), -1)
                cv2.line(self.frame, self.line_points[0], self.line_points[1], (0, 255, 0), 3)
                
                # Add labels
                cv2.putText(self.frame, "IN", 
                           (self.line_points[0][0] - 30, self.line_points[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(self.frame, "OUT", 
                           (self.line_points[1][0] + 10, self.line_points[1][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                print(f"✓ Custom line set from {self.line_points[0]} to {self.line_points[1]}")
                print(f"  Green circle = IN side, Red circle = OUT side")
    
    def run(self):
        """Run the interactive ROI setup"""
        
        cv2.namedWindow('ROI Setup - Head Counter')
        cv2.setMouseCallback('ROI Setup - Head Counter', self.mouse_callback)
        
        # Add instructions overlay
        self.add_instructions()
        
        while True:
            cv2.imshow('ROI Setup - Head Counter', self.frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('h') or key == ord('H'):
                self.line_type = 'horizontal'
                self.line_pos = None
                self.line_points = []
                self.frame = self.original_frame.copy()
                self.add_instructions()
                print("\n📏 Mode: HORIZONTAL LINE")
                print("   Click to set the Y position")
            
            elif key == ord('v') or key == ord('V'):
                self.line_type = 'vertical'
                self.line_pos = None
                self.line_points = []
                self.frame = self.original_frame.copy()
                self.add_instructions()
                print("\n📏 Mode: VERTICAL LINE")
                print("   Click to set the X position")
            
            elif key == ord('c') or key == ord('C'):
                self.line_type = 'custom'
                self.line_pos = None
                self.line_points = []
                self.frame = self.original_frame.copy()
                self.add_instructions()
                print("\n📏 Mode: CUSTOM LINE")
                print("   Click and drag to draw the line")
                print("   Start point = IN side (green)")
                print("   End point = OUT side (red)")
            
            elif key == ord('r') or key == ord('R'):
                self.line_pos = None
                self.line_points = []
                self.frame = self.original_frame.copy()
                self.add_instructions()
                print("\n🔄 Line reset")
            
            elif key == ord('s') or key == ord('S'):
                if self.save_config():
                    print("\n✓ Configuration saved!")
                    print("  You can now run head_counter.py with this config")
                    break
                else:
                    print("\n⚠️  Please draw a line first!")
            
            elif key == ord('q') or key == ord('Q'):
                print("\n❌ Cancelled - configuration not saved")
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def add_instructions(self):
        """Add instruction overlay to frame"""
        overlay = self.frame.copy()
        
        # Semi-transparent black background
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        self.frame = cv2.addWeighted(overlay, 0.7, self.frame, 0.3, 0)
        
        # Instructions text
        y_pos = 35
        cv2.putText(self.frame, "CONTROLS:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25
        cv2.putText(self.frame, "[H] Horizontal Line", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_pos += 25
        cv2.putText(self.frame, "[V] Vertical Line", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_pos += 25
        cv2.putText(self.frame, "[C] Custom Line", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_pos += 25
        cv2.putText(self.frame, "[R] Reset", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        y_pos += 25
        cv2.putText(self.frame, "[S] Save & Exit", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += 25
        cv2.putText(self.frame, "[Q] Quit", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    def save_config(self, config_path='head_counter_config.json'):
        """Save ROI configuration to JSON file"""
        
        if self.line_type == 'horizontal' and self.line_pos is not None:
            config = {
                'type': 'horizontal',
                'y': self.line_pos,
                'description': f'Horizontal line at Y={self.line_pos}'
            }
        
        elif self.line_type == 'vertical' and self.line_pos is not None:
            config = {
                'type': 'vertical',
                'x': self.line_pos,
                'description': f'Vertical line at X={self.line_pos}'
            }
        
        elif self.line_type == 'custom' and len(self.line_points) == 2:
            config = {
                'type': 'custom',
                'line_points': self.line_points,
                'description': f'Custom line from {self.line_points[0]} to {self.line_points[1]}'
            }
        
        else:
            return False
        
        # Add metadata
        config['video_resolution'] = {
            'width': self.width,
            'height': self.height
        }
        config['video_path'] = str(self.video_path)
        
        # Save to file
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n📁 Configuration saved to: {config_path}")
        print(f"   Type: {config['type']}")
        print(f"   {config['description']}")
        
        return True


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup ROI Line for Head Counter')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='head_counter_config.json',
                        help='Output config file path')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        print(f"❌ Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Run ROI setup
    try:
        setup = ROISetup(args.video)
        setup.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
