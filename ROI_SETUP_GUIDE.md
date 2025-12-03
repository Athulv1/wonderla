# 🎯 ROI Line Setup Guide for IN/OUT Counting

## Quick Start

### Step 1: Set Your ROI Line

Run the interactive configuration tool:
```bash
python3 set_roi_line.py
```

You'll be prompted to choose:

#### Option 1: Horizontal Line
- **Use case:** Counting people crossing a horizontal boundary
- **Direction:** Top → Bottom = **OUT** | Bottom → Top = **IN**
- **Example:** Y=540 (middle of 1920x1080 video)

#### Option 2: Vertical Line (Recommended for your case)
- **Use case:** Counting people crossing a vertical boundary  
- **Direction:** Left → Right = **OUT** | Right → Left = **IN**
- **Example:** X=960 (middle of 1920x1080 video) or X=1100 (right side)

#### Option 3: Custom Diagonal Line
- **Use case:** Angled or diagonal boundaries
- **Setup:** Click two points
  - Point 1 side → Point 2 side = **OUT**
  - Point 2 side → Point 1 side = **IN**
- **Example:** (200, 100) to (1800, 900)

### Step 2: Run Inference with ROI

```bash
python3 inference.py \
    --model weights/best.pt \
    --source 00119.mp4 \
    --output output_with_roi.mp4 \
    --roi-config roi_config.json \
    --conf 0.25 \
    --save-json results.json
```

## Understanding IN/OUT Direction

### Horizontal Line Example
```
   ╔════════════════════════╗
   ║                        ║  ← IN side (Top)
   ║         ↓ OUT          ║
   ╠════════════════════════╣  ← ROI LINE (Y=500)
   ║         ↑ IN           ║
   ║                        ║  ← OUT side (Bottom)
   ╚════════════════════════╝
```

### Vertical Line Example
```
   ╔═══════╦════════════╗
   ║       ║            ║
   ║   IN  ║  OUT       ║
   ║   ←   ║   →        ║
   ║       ║            ║
   ╚═══════╩════════════╝
          ROI LINE (X=800)
```

## Manual ROI Configuration

### Quick Commands (without setup tool)

#### Horizontal line at center:
```bash
python3 inference.py --model weights/best.pt --source video.mp4 \
    --output output.mp4 --roi-y 540
```

#### Vertical line at X=1100:
```bash
python3 inference.py --model weights/best.pt --source video.mp4 \
    --output output.mp4 --roi-x 1100
```

#### No tracking (just detection):
```bash
python3 inference.py --model weights/best.pt --source video.mp4 \
    --output output.mp4 --no-tracking
```

## Tips for Setting ROI Line

1. **Watch your video first** - Understand the flow direction
2. **Start simple** - Use horizontal or vertical line before custom
3. **Test and adjust** - Run inference, check the output video, adjust position
4. **Consider scale** - The tool automatically scales for resized processing

## Output Information

The video will show:
- **Yellow line** - ROI boundary
- **Green dots with IDs** - Tracked objects
- **Top overlay** - Real-time IN/OUT counts
- **Point markers** (custom line only):
  - Green circle = IN side (Point 1)
  - Red circle = OUT side (Point 2)

## Common Issues

### Wrong Direction?
- Swap the ROI line position or flip the points
- For horizontal: Lower Y = more OUT, Higher Y = more IN
- For vertical: Lower X = more IN, Higher X = more OUT

### Missing Counts?
- Adjust `--conf` threshold (try 0.15-0.35)
- Check if line position intersects the path
- Ensure objects fully cross the line

### Too Many False Counts?
- Use custom line to avoid edge areas
- Increase confidence threshold
- Adjust tracking parameters in code

## Example Results

From your test run:
```
IN Count:  8   (People entering from right/bottom)
OUT Count: 22  (People exiting to left/top)
Total:     30 crossings
```

## Advanced: Edit roi_config.json Directly

```json
{
  "type": "vertical",
  "x": 1100,
  "description": "Vertical line at X=1100"
}
```

Or for custom line:
```json
{
  "type": "custom",
  "line_points": [[200, 100], [1800, 900]],
  "description": "Diagonal line from top-left to bottom-right"
}
```

---

**Need help?** The line drawn in the output video shows exactly where counting happens!
