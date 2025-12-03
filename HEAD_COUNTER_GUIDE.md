# Head Counter for Top-View Camera 🎯

Count people (heads) from overhead/top-view camera footage with line-crossing detection.

## 🌟 Features

- **YOLOv8 Person Detection**: Uses pre-trained model optimized for detecting people from any angle
- **Top-View Optimized**: Works great with overhead cameras where you see heads, not full bodies
- **Interactive ROI Editor**: Draw custom counting lines on your video
- **IN/OUT Counting**: Track people crossing the line in both directions
- **Real-time Statistics**: Shows current head count, max count, and crossing statistics
- **Live Preview**: Watch detection in real-time while processing

## 📋 Requirements

```bash
pip install ultralytics opencv-python scipy numpy
```

## 🚀 Quick Start

### Option 1: Use the Batch Script (Easiest)

1. Double-click `run_head_counter.bat`
2. Follow the on-screen instructions

### Option 2: Manual Steps

#### Step 1: Setup ROI Line (First Time Only)

Draw the counting line on your video:

```bash
python setup_roi_head_counter.py --video "your_video.mp4"
```

**Controls:**
- Press `H` - Horizontal line (for people moving up/down)
- Press `V` - Vertical line (for people moving left/right)
- Press `C` - Custom line (any angle)
- Click to place the line
- Press `S` - Save and exit
- Press `Q` - Quit without saving

This creates `head_counter_config.json` with your ROI line.

#### Step 2: Run Head Counter

Process your video with the configured ROI line:

```bash
python head_counter.py --video "your_video.mp4" --output "output.mp4" --show
```

**Arguments:**
- `--video`: Input video path (required)
- `--output`: Output video path (optional)
- `--show`: Show live preview while processing
- `--model`: YOLOv8 model (default: yolov8n.pt)
  - `yolov8n.pt` - Fastest, lower accuracy
  - `yolov8s.pt` - Balanced
  - `yolov8m.pt` - More accurate, slower
- `--conf`: Confidence threshold (default: 0.25)
- `--roi-config`: ROI config file (default: head_counter_config.json)
- `--skip-frames`: Process every N frames (default: 1)
- `--resize`: Resize width for faster processing
- `--save-json`: Save results to JSON file

## 📊 Example Usage

### Basic Usage
```bash
python head_counter.py --video wavepool.mp4 --show
```

### High Quality Output
```bash
python head_counter.py --video wavepool.mp4 --output counted.mp4 --model yolov8m.pt --conf 0.3
```

### Fast Processing
```bash
python head_counter.py --video wavepool.mp4 --skip-frames 2 --resize 1280
```

### Save Statistics
```bash
python head_counter.py --video wavepool.mp4 --save-json results.json
```

## 🎮 Live Preview Controls

When using `--show`:
- Press `P` - Pause/Resume
- Press `Q` - Quit early

## 📈 Output Statistics

The counter provides:

### Line Crossing Counts
- **IN Count**: People crossing line in one direction
- **OUT Count**: People crossing line in opposite direction
- **Net Count**: IN - OUT (current occupancy)
- **Total Crossings**: Total number of line crossings

### Head Statistics
- **Current Heads**: Number of heads detected in current frame
- **Max Heads**: Maximum number of heads detected in any frame
- **Average Heads**: Average heads per frame

## 📁 Output Files

### Video Output
Annotated video with:
- Bounding boxes around detected heads
- Tracking IDs for each person
- ROI counting line
- Real-time statistics overlay

### JSON Output
Detailed statistics including:
- Frame-by-frame head counts
- Line crossing events
- Total statistics
- Processing metadata

## ⚙️ Configuration File

`head_counter_config.json` format:

**Horizontal Line:**
```json
{
  "type": "horizontal",
  "y": 540,
  "description": "Horizontal line at Y=540"
}
```

**Vertical Line:**
```json
{
  "type": "vertical",
  "x": 960,
  "description": "Vertical line at X=960"
}
```

**Custom Line:**
```json
{
  "type": "custom",
  "line_points": [[100, 200], [800, 600]],
  "description": "Custom diagonal line"
}
```

## 🎯 Tips for Best Results

### Camera Setup
- ✅ Top-view (overhead) angle works best
- ✅ Good lighting conditions
- ✅ Stable camera (not moving)
- ✅ Clear view of counting area

### Configuration
- Start with `--conf 0.25` and adjust if needed
- Lower confidence (0.15-0.2) for distant/small heads
- Higher confidence (0.3-0.5) to reduce false positives
- Use `yolov8n.pt` for speed, `yolov8m.pt` for accuracy

### ROI Line Placement
- Place line where people clearly cross
- Avoid edges where people might be cut off
- For wave pools, place line at entry/exit points
- Use custom line for diagonal paths

### Performance
- Use `--skip-frames 2` to process faster (every 2nd frame)
- Use `--resize 1280` to reduce processing load
- Remove `--show` for background processing

## 🔍 Troubleshooting

**No heads detected:**
- Lower confidence threshold: `--conf 0.2`
- Try different model: `--model yolov8s.pt`
- Check if video quality is good enough

**Too many false detections:**
- Increase confidence: `--conf 0.35`
- Use larger model: `--model yolov8m.pt`

**Processing too slow:**
- Use `--skip-frames 2` or higher
- Use `--resize 1280` or `--resize 960`
- Use smaller model: `--model yolov8n.pt`
- Remove `--show` flag

**Line crossings not counting:**
- Check ROI line placement using setup tool
- Ensure line crosses the path people take
- Verify people are fully crossing the line

## 📞 Example for Wave Pool Camera

```bash
# Step 1: Setup ROI line
python setup_roi_head_counter.py --video "WAVEPOOL TEST CAMERA_2025-12-01_03-55-00-2025-12-01_04-05-59.mp4"

# Step 2: Run counter with live preview
python head_counter.py --video "WAVEPOOL TEST CAMERA_2025-12-01_03-55-00-2025-12-01_04-05-59.mp4" --output "wavepool_counted.mp4" --show --conf 0.3 --save-json "wavepool_stats.json"
```

## 🎓 How It Works

1. **Detection**: YOLOv8 detects people in each frame
2. **Tracking**: Centroid tracker assigns unique IDs to each person
3. **Line Crossing**: Monitors when tracked objects cross the ROI line
4. **Counting**: Increments IN/OUT counters based on crossing direction
5. **Visualization**: Draws bounding boxes, IDs, and statistics

## 📝 License

Uses YOLOv8 from Ultralytics (AGPL-3.0 license)
