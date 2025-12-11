"""
Interactive Two-Line Setup for Head Counter
Draw separate IN and OUT counting lines
"""

import cv2
import json
import sys
import argparse
from pathlib import Path


class TwoLineSetup:
    """Interactive two-line drawing tool"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        # Flip frame vertically to match runtime
        self.frame = cv2.flip(self.frame, 0)
        
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
        
        # Line variables
        self.in_line_y = None  # Y position for IN line
        self.out_line_y = None  # Y position for OUT line
        self.current_mode = 'in'  # 'in' or 'out'
        self.dragging = False
        self.drag_line = None  # Which line is being dragged
        
        print("\n" + "="*70)
        print("🎯 HEAD COUNTER - TWO-LINE SETUP (IN/OUT Lines)")
        print("="*70)
        print(f"Video: {video_path}")
        print(f"Original Resolution: {self.width}x{self.height}")
        print(f"Display Resolution: {self.display_width}x{self.display_height}")
        print("="*70)
        print("\nINSTRUCTIONS:")
        print("  1. Press [I] to set IN counting line (green)")
        print("  2. Click and drag to position the line")
        print("  3. Press [O] to set OUT counting line (red)")
        print("  4. Click and drag to position the line")
        print("\nCONTROLS:")
        print("  [I] - Set IN line mode (green)")
        print("  [O] - Set OUT line mode (red)")
        print("  [R] - Reset both lines")
        print("  [S] - Save configuration")
        print("  [Q] - Quit without saving")
        print("="*70)
        print("\nLOGIC:")
        print("  Person crosses IN line = IN count increases")
        print("  Person crosses OUT line = OUT count increases")
        print("="*70)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for line positioning"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near existing lines to drag them
            if self.in_line_y is not None and abs(y - self.in_line_y) < 10:
                self.dragging = True
                self.drag_line = 'in'
            elif self.out_line_y is not None and abs(y - self.out_line_y) < 10:
                self.dragging = True
                self.drag_line = 'out'
            else:
                # Create new line in current mode
                self.dragging = True
                self.drag_line = self.current_mode
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.drag_line:
                # Update line position
                if self.drag_line == 'in':
                    self.in_line_y = y
                elif self.drag_line == 'out':
                    self.out_line_y = y
                self.update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_line = None
    
    def update_display(self):
        """Update the display with current lines"""
        display_frame = self.original_frame.copy()
        
        # Draw IN line (green)
        if self.in_line_y is not None:
            cv2.line(display_frame, (0, self.in_line_y), 
                    (self.display_width, self.in_line_y), (0, 255, 0), 3)
            cv2.putText(display_frame, "IN LINE", (10, self.in_line_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw OUT line (red)
        if self.out_line_y is not None:
            cv2.line(display_frame, (0, self.out_line_y), 
                    (self.display_width, self.out_line_y), (0, 0, 255), 3)
            cv2.putText(display_frame, "OUT LINE", (10, self.out_line_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw mode indicator
        mode_text = f"Mode: {'IN (Green)' if self.current_mode == 'in' else 'OUT (Red)'}"
        cv2.putText(display_frame, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Two-Line Setup', display_frame)
    
    def run(self):
        """Run the interactive setup"""
        cv2.namedWindow('Two-Line Setup')
        cv2.setMouseCallback('Two-Line Setup', self.mouse_callback)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('i') or key == ord('I'):
                self.current_mode = 'in'
                print("\n📦 Mode: IN LINE (green)")
                print("   Click and drag to position the IN counting line")
                self.update_display()
                
            elif key == ord('o') or key == ord('O'):
                self.current_mode = 'out'
                print("\n📦 Mode: OUT LINE (red)")
                print("   Click and drag to position the OUT counting line")
                self.update_display()
                
            elif key == ord('r') or key == ord('R'):
                self.in_line_y = None
                self.out_line_y = None
                print("\n🔄 Lines reset")
                self.update_display()
                
            elif key == ord('s') or key == ord('S'):
                if self.in_line_y is None or self.out_line_y is None:
                    print("\n❌ ERROR: Both IN and OUT lines must be set before saving!")
                    continue
                    
                self.save_configuration()
                break
                
            elif key == ord('q') or key == ord('Q'):
                print("\n❌ Quit without saving")
                break
        
        cv2.destroyAllWindows()
        self.cap.release()
    
    def save_configuration(self):
        """Save line configuration to JSON"""
        # Scale lines back to original resolution
        in_line_scaled = int(self.in_line_y * self.scale)
        out_line_scaled = int(self.out_line_y * self.scale)
        
        config = {
            'type': 'two_lines',
            'in_line_y': in_line_scaled,
            'out_line_y': out_line_scaled,
            'description': 'Two-line counting: Cross IN line = IN count, Cross OUT line = OUT count',
            'video_resolution': {
                'width': int(self.width * self.scale),
                'height': int(self.height * self.scale)
            },
            'video_path': str(self.video_path)
        }
        
        output_path = Path('head_counter_config.json')
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\n" + "="*70)
        print("✅ CONFIGURATION SAVED")
        print("="*70)
        print(f"Output file: {output_path}")
        print(f"IN Line Y: {in_line_scaled}")
        print(f"OUT Line Y: {out_line_scaled}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Setup two counting lines for head counter')
    parser.add_argument('--video', required=True, help='Video file or frame image path')
    parser.add_argument('--output', default='head_counter_config.json', help='Output config file')
    
    args = parser.parse_args()
    
    try:
        setup = TwoLineSetup(args.video)
        setup.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
