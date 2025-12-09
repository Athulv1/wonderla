# RTSP Head Counter Dashboard Guide

## 🎯 Enhanced Features

The dashboard now includes advanced monitoring and control features:

### 📊 Enhanced Statistics Display

1. **🔽 Total Entered Banner (Green)**
   - Live count animation
   - Trend indicator (shows recent changes)
   - Visual comparison bar
   - Click to copy value

2. **🔼 Total Exited Banner (Orange)**
   - Real-time exit tracking
   - Trend visualization
   - Percentage comparison
   - Click to copy value

3. **🏊 Current Pool Count (Large Purple Banner)**
   - Animated pulse background
   - NET count (IN - OUT)
   - Four sub-metrics:
     - **Active Heads**: Currently detected
     - **Processing FPS**: Performance monitor
     - **Occupancy Rate**: % of capacity (default: 50 max)
     - **Peak Today**: Highest count recorded

### ⚡ Real-time Features

- **Live Clock**: Date and time display in header
- **Auto-update**: Stats refresh every 500ms
- **Smooth Animations**: Number counting, color pulses
- **Trend Indicators**: Show recent increases/decreases
- **Capacity Alerts**: Warning when pool reaches 90% capacity

### 🎮 Interactive Controls

**Control Panel (Bottom Right):**
- **🖥️ Fullscreen**: Enter/exit fullscreen mode
- **🔄 Reset Counts**: Clear all counters (with confirmation)

**Keyboard Shortcuts:**
- `R` - Reload dashboard
- `F` - Toggle fullscreen

**Click Features:**
- Click any stat value to copy to clipboard
- Hover effects on all cards
- Shimmer animation on hover

### 🚨 Alert System

- **Capacity Warning**: Appears when pool count exceeds 90% of max capacity
- **Auto-dismiss**: Alerts hide after 5 seconds
- **Visual feedback**: Color-coded notifications

## Running the Dashboard

### Command Line:
```powershell
.\venv\Scripts\python.exe rtsp_dashboard.py --rtsp "rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=1&subtype=1" --box-shrink 0.4
```

### Access URLs:
- **Local:** http://127.0.0.1:5000
- **Network:** http://192.168.1.6:5000

### Parameters:
- `--rtsp`: RTSP stream URL (required)
- `--model`: YOLO model path (default: yolov8n.pt)
- `--conf`: Confidence threshold (default: 0.25)
- `--box-shrink`: Box shrink factor (default: 0.4)
- `--host`: Host address (default: 0.0.0.0)
- `--port`: Port number (default: 5000)

## API Endpoints

The dashboard provides several REST API endpoints:

### GET `/stats`
Returns current statistics:
```json
{
  "in_count": 25,
  "out_count": 18,
  "pool_count": 7,
  "current_heads": 4,
  "fps": 12.5,
  "timestamp": 1702134567.89
}
```

### POST `/reset`
Resets all counters to zero. Returns:
```json
{
  "success": true,
  "message": "Counters reset successfully"
}
```

### GET `/health`
Health check endpoint:
```json
{
  "status": "running",
  "processor_active": true,
  "timestamp": 1702134567.89
}
```

## Configuration

### Adjust Pool Capacity
Edit the `MAX_CAPACITY` variable in the dashboard HTML:
```javascript
const MAX_CAPACITY = 50; // Change to your pool's capacity
```

### Zone Configuration
Modify `head_counter_config.json` for your camera resolution:
```json
{
  "upper_zone": [0, 0, 352, 96],
  "lower_zone": [0, 96, 352, 288]
}
```

## Design Elements

### Color Scheme
- Background: Dark gradient (#0a0e27 → #1a1a2e → #16213e)
- Cards: Dark transparent with blur
- Green accent: #11998e → #38ef7d (IN)
- Orange accent: #ee0979 → #ff6a00 (OUT)
- Purple accent: #667eea → #764ba2 (Pool)

### Typography
- Font: Inter (Google Fonts)
- Weights: 400, 500, 600, 700
- Responsive sizing

### Animations
- Gradient shift (15s loop)
- Pulse animation (4s loop)
- Count-up animation (0.8s)
- Hover lift effect
- Fade-in on load

## Tips

1. **Adjust Zones:** Modify `head_counter_config.json` for your camera resolution
2. **Performance:** Lower `--conf` threshold if missing detections
3. **Stability:** TCP transport is already enabled for best performance
4. **Mobile:** Dashboard is fully responsive

## Troubleshooting

### Stream not connecting?
- Check RTSP URL format
- Verify camera credentials
- Ensure network connectivity

### Counts not accurate?
- Adjust box-shrink factor
- Configure zones for your resolution
- Check confidence threshold

### Low FPS?
- Model already optimized with imgsz=320
- No frame skipping (maintains tracking)
- Check network bandwidth

## Files

- `rtsp_dashboard.py`: Backend Flask server
- `templates/rtsp_dashboard.html`: Frontend UI
- `head_counter_config.json`: Zone configuration
- `yolov8n.pt`: YOLO model weights

---

**Status:** ✅ Running
**Design:** 🎨 Modern Dark Theme
**Updates:** ⚡ Real-time (500ms)
