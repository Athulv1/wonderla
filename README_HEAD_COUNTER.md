# Head Counter for Top-View Cameras 🎯

AI-powered people counting system for overhead/top-view camera footage using YOLOv8 and zone-based tracking.

## Features

- ✅ **YOLOv8 Person Detection** - Pre-trained model for detecting people from any angle
- ✅ **Top-View Optimized** - Works with overhead cameras (sees heads, not full bodies)
- ✅ **Zone-Based Counting** - Two detection zones (upper/lower) for accurate IN/OUT tracking
- ✅ **Improved Tracking** - IoU + centroid tracking for stable ID assignment
- ✅ **Interactive ROI Editor** - Visual tool to draw custom counting zones
- ✅ **Real-time Statistics** - Live head count, max count, and crossing counts
- ✅ **Live Preview** - Watch detection while processing

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/Athulv1/wonderla.git
cd wonderla/live_detection_app
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r head_counter_requirements.txt
```

## Quick Start

### Step 1: Setup Detection Zones

Draw upper and lower detection zones:

```bash
python setup_zones.py --video "your_video.mp4"
```

**Controls:**
- `U` - Draw upper zone (green)
- `L` - Draw lower zone (red)
- Click and drag to draw
- `S` - Save
- `R` - Reset

### Step 2: Run Counter

```bash
python head_counter_zones.py --video "your_video.mp4" --output "counted.mp4" --show --conf 0.3
```

## Usage

### Basic
```bash
python head_counter_zones.py --video video.mp4 --show
```

### High Accuracy
```bash
python head_counter_zones.py --video video.mp4 --model yolov8m.pt --conf 0.3 --output result.mp4
```

### Fast Processing
```bash
python head_counter_zones.py --video video.mp4 --skip-frames 2 --resize 1280
```

### Save Statistics
```bash
python head_counter_zones.py --video video.mp4 --save-json stats.json
```

## Counting Logic

- **Upper → Lower** = **IN** (entering)
- **Lower → Upper** = **OUT** (exiting)
- **Net** = IN - OUT

## Configuration

Default: 50/50 vertical split. Customize with `setup_zones.py`.

## Live Preview

- `P` - Pause/Resume
- `Q` - Quit

## Tips

- Use `--conf 0.2` for distant heads
- Use `--conf 0.35` to reduce false positives
- Use `yolov8n.pt` for speed, `yolov8m.pt` for accuracy
- Use `--skip-frames 2` for faster processing

## Files

- `head_counter_zones.py` - Main counter
- `setup_zones.py` - Zone editor
- `head_counter_config.json` - Configuration
- `head_counter_requirements.txt` - Dependencies

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- SciPy

## License

YOLOv8 from Ultralytics (AGPL-3.0)
