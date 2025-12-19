"""
Interactive Zone Setup for Head Counter
Draw upper and lower detection zones for zone-based counting
"""

import cv2
import json
import sys
from pathlib import Path


class ZoneSetup:
    """Interactive zone drawing tool"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        # Resize for display if too large
        self.height, self.width = self.frame.shape[:2]
        self.display_width = 1280
        if self.width > self.display_width:
            self.display_height = int(self.height * (self.display_width / self.width))
            self.scale = self.width / self.display_width
            self.frame = cv2.resize(self.frame, (self.display_width, self.display_height))
        else:
            self.display_width = self.width
            self.display_height = self.height
            self.scale = 1.0
        
        self.original_frame = self.frame.copy()
        
        # Zone variables
        self.drawing = False
        self.current_zone = None  # 'upper' or 'lower'
        self.upper_zone = None  # [x1, y1, x2, y2]
        self.lower_zone = None
        self.start_point = None
        
        print("\n" + "="*70)
        print("🎯 HEAD COUNTER - ZONE SETUP (Upper/Lower Boxes)")
        print("="*70)
        print(f"Video: {video_path}")
        print(f"Original Resolution: {self.width}x{self.height}")
        print(f"Display Resolution: {self.display_width}x{self.display_height}")
        print("="*70)
        print("\nINSTRUCTIONS:")
        print("  1. Press [U] to draw UPPER zone (IN detection)")
        print("  2. Click and drag to draw the upper box")
        print("  3. Press [L] to draw LOWER zone (OUT detection)")
        print("  4. Click and drag to draw the lower box")
        print("\nCONTROLS:")
        print("  [U] - Draw Upper Zone (green)")
        print("  [L] - Draw Lower Zone (red)")
        print("  [R] - Reset zones")
        print("  [S] - Save configuration")
        print("  [Q] - Quit without saving")
        print("="*70)
        print("\nLOGIC:")
        print("  Person detected in UPPER first, then LOWER = IN count")
        print("  Person detected in LOWER first, then UPPER = OUT count")
        print("="*70 + "\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing zones"""
        
        if self.current_zone is None:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            temp_frame = self.original_frame.copy()
            
            # Draw existing zones (scaled for display)
            if self.upper_zone:
                x1_disp = int(self.upper_zone[0] / self.scale)
                y1_disp = int(self.upper_zone[1] / self.scale)
                x2_disp = int(self.upper_zone[2] / self.scale)
                y2_disp = int(self.upper_zone[3] / self.scale)
                cv2.rectangle(temp_frame, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 255, 0), 2)
                cv2.putText(temp_frame, "UPPER (IN)", (x1_disp + 10, y1_disp + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.lower_zone:
                x1_disp = int(self.lower_zone[0] / self.scale)
                y1_disp = int(self.lower_zone[1] / self.scale)
                x2_disp = int(self.lower_zone[2] / self.scale)
                y2_disp = int(self.lower_zone[3] / self.scale)
                cv2.rectangle(temp_frame, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 0, 255), 2)
                cv2.putText(temp_frame, "LOWER (OUT)", (x1_disp + 10, y1_disp + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw current zone being drawn
            color = (0, 255, 0) if self.current_zone == 'upper' else (0, 0, 255)
            cv2.rectangle(temp_frame, self.start_point, (x, y), color, 2)
            
            self.frame = temp_frame
        
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            
            # Save the zone
            x1 = min(self.start_point[0], x)
            y1 = min(self.start_point[1], y)
            x2 = max(self.start_point[0], x)
            y2 = max(self.start_point[1], y)
            
            # Scale coordinates back to original resolution
            x1_orig = int(x1 * self.scale)
            y1_orig = int(y1 * self.scale)
            x2_orig = int(x2 * self.scale)
            y2_orig = int(y2 * self.scale)
            
            if self.current_zone == 'upper':
                self.upper_zone = [x1_orig, y1_orig, x2_orig, y2_orig]
                print(f"✓ Upper zone set: [{x1_orig}, {y1_orig}, {x2_orig}, {y2_orig}]")
            else:
                self.lower_zone = [x1_orig, y1_orig, x2_orig, y2_orig]
                print(f"✓ Lower zone set: [{x1_orig}, {y1_orig}, {x2_orig}, {y2_orig}]")
            
            # Redraw frame with both zones (scaled for display)
            self.frame = self.original_frame.copy()
            
            if self.upper_zone:
                # Scale zones for display
                x1_disp = int(self.upper_zone[0] / self.scale)
                y1_disp = int(self.upper_zone[1] / self.scale)
                x2_disp = int(self.upper_zone[2] / self.scale)
                y2_disp = int(self.upper_zone[3] / self.scale)
                cv2.rectangle(self.frame, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 255, 0), 2)
                cv2.putText(self.frame, "UPPER (IN)", (x1_disp + 10, y1_disp + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.lower_zone:
                x1_disp = int(self.lower_zone[0] / self.scale)
                y1_disp = int(self.lower_zone[1] / self.scale)
                x2_disp = int(self.lower_zone[2] / self.scale)
                y2_disp = int(self.lower_zone[3] / self.scale)
                cv2.rectangle(self.frame, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 0, 255), 2)
                cv2.putText(self.frame, "LOWER (OUT)", (x1_disp + 10, y1_disp + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            self.add_instructions()
            self.current_zone = None
    
    def run(self):
        """Run the interactive zone setup"""
        
        cv2.namedWindow('Zone Setup - Head Counter')
        cv2.setMouseCallback('Zone Setup - Head Counter', self.mouse_callback)
        
        self.add_instructions()
        
        while True:
            cv2.imshow('Zone Setup - Head Counter', self.frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('u') or key == ord('U'):
                self.current_zone = 'upper'
                print("\n📦 Mode: UPPER ZONE (green)")
                print("   Click and drag to draw the upper detection box")
            
            elif key == ord('l') or key == ord('L'):
                self.current_zone = 'lower'
                print("\n📦 Mode: LOWER ZONE (red)")
                print("   Click and drag to draw the lower detection box")
            
            elif key == ord('r') or key == ord('R'):
                self.upper_zone = None
                self.lower_zone = None
                self.current_zone = None
                self.frame = self.original_frame.copy()
                self.add_instructions()
                print("\n🔄 Zones reset")
            
            elif key == ord('s') or key == ord('S'):
                if self.save_config():
                    print("\n✓ Configuration saved!")
                    print("  Run: head_counter_zones.py with this config")
                    break
                else:
                    print("\n⚠️  Please draw both upper and lower zones first!")
            
            elif key == ord('q') or key == ord('Q'):
                print("\n❌ Cancelled - configuration not saved")
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def add_instructions(self):
        """Add instruction overlay to frame"""
        overlay = self.frame.copy()
        
        cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
        self.frame = cv2.addWeighted(overlay, 0.7, self.frame, 0.3, 0)
        
        y_pos = 35
        cv2.putText(self.frame, "ZONE SETUP:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(self.frame, "[U] Draw Upper Zone", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += 25
        cv2.putText(self.frame, "[L] Draw Lower Zone", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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
        """Save zone configuration to JSON file"""
        
        if self.upper_zone is None or self.lower_zone is None:
            return False
        
        config = {
            'type': 'zones',
            'upper_zone': self.upper_zone,
            'lower_zone': self.lower_zone,
            'description': 'Zone-based counting: upper → lower = OUT, lower → upper = IN',
            'video_resolution': {
                'width': self.width,
                'height': self.height
            },
            'video_path': str(self.video_path)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n📁 Configuration saved to: {config_path}")
        print(f"   Upper zone: {self.upper_zone}")
        print(f"   Lower zone: {self.lower_zone}")
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Zones for Head Counter')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='head_counter_config.json',
                        help='Output config file')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"❌ Error: Video file not found: {args.video}")
        sys.exit(1)
    
    try:
        setup = ZoneSetup(args.video)
        setup.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
