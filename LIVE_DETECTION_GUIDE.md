# 📺 Live Detection Mode Guide

## Live Detection with Real-time Preview

Your inference script now supports **LIVE DETECTION** mode where you can see the detection happening in real-time while the video processes in the background!

---

## 🚀 Quick Start - Live Detection

### Option 1: Live Preview (Default for Videos)
```bash
python3 inference.py \
    --model weights/best.pt \
    --source 00119.mp4 \
    --output detected_output.mp4 \
    --roi-config roi_config.json
```

**What happens:**
- ✅ Live window opens showing real-time detection
- ✅ Video saves to `detected_output.mp4` in background
- ✅ You see IN/OUT counts updating live
- ✅ ROI line and tracking dots visible in real-time

### Option 2: Background Only (No Preview)
```bash
python3 inference.py \
    --model weights/best.pt \
    --source 00119.mp4 \
    --output detected_output.mp4 \
    --roi-config roi_config.json \
    --no-show
```

**What happens:**
- ⏩ Processes faster (no display overhead)
- 💾 Saves video to file
- 📊 Prints progress in terminal
- ✅ Good for batch processing

---

## 🎮 Live Preview Controls

When the live preview window is open:

| Key | Action |
|-----|--------|
| **Q** | Quit/Stop processing early |
| **P** | Pause/Resume detection |

### Pause Example
```
Press 'P' → Detection pauses
               ⏸️  PAUSED message appears
               Review current frame

Press 'P' again → Detection resumes
                  ▶️  RESUMED
                  Processing continues
```

---

## 📋 Complete Examples

### 1. Live Detection with ROI Line Counting
```bash
# Step 1: Set up your ROI line
python3 set_roi_line.py

# Step 2: Run live detection
python3 inference.py \
    --model weights/best.pt \
    --source 00119.mp4 \
    --output live_detected.mp4 \
    --roi-config roi_config.json \
    --conf 0.25
```

**You'll see:**
- 🎯 Yellow ROI line
- 🟢 Green tracking dots with IDs
- 📊 Real-time IN/OUT counts
- 🎬 Frame-by-frame detection

### 2. Live Detection with Custom Settings
```bash
python3 inference.py \
    --model weights/best.pt \
    --source your_video.mp4 \
    --output result.mp4 \
    --roi-y 540 \
    --conf 0.3 \
    --save-json results.json
```

### 3. Process Multiple Videos (Background Mode)
```bash
# No live preview - faster processing
for video in *.mp4; do
    python3 inference.py \
        --model weights/best.pt \
        --source "$video" \
        --output "detected_$video" \
        --roi-config roi_config.json \
        --no-show
done
```

---

## 💡 Performance Tips

### For Live Preview
- **Slower Processing:** Live display adds ~5-10% overhead
- **Better for:** Development, testing, demonstrations
- **Use when:** You want to see results immediately

### For Background Mode (--no-show)
- **Faster Processing:** ~15-20 FPS vs 14-15 FPS with preview
- **Better for:** Batch processing, production runs
- **Use when:** Processing many videos overnight

---

## 🎯 Common Use Cases

### Use Case 1: Testing ROI Line Position
```bash
# Live preview to check if line is in right position
python3 inference.py \
    --model weights/best.pt \
    --source 00119.mp4 \
    --output test.mp4 \
    --roi-y 600

# If wrong, press Q to stop, adjust, and re-run
```

### Use Case 2: Real-time Monitoring
```bash
# Watch live as people cross the ROI line
python3 inference.py \
    --model weights/best.pt \
    --source security_footage.mp4 \
    --output monitored.mp4 \
    --roi-config roi_config.json

# Press P to pause when you see something interesting
# Press Q when done reviewing
```

### Use Case 3: Production Processing
```bash
# Process without preview for maximum speed
python3 inference.py \
    --model weights/best.pt \
    --source batch_video.mp4 \
    --output processed.mp4 \
    --roi-config roi_config.json \
    --no-show \
    --save-json results.json
```

---

## 📊 What You'll See in Live Mode

```
┌─────────────────────────────────────────┐
│ Frame 1234/15889 | IN: 14 | OUT: 239   │ ← Count overlay
├─────────────────────────────────────────┤
│                                         │
│         🟢 ID:1    🟢 ID:3              │ ← Tracked objects
│                                         │
│    ═══════════════════════════         │ ← ROI Line (Yellow)
│                                         │
│              🟢 ID:2                    │
│                                         │
└─────────────────────────────────────────┘

Terminal Output:
Progress: 45.3% | Processed 3600 frames | FPS: 14.8
```

---

## ⚠️ Troubleshooting Live Preview

### Problem: Window doesn't show (WSL/SSH)
```bash
# You're on WSL or remote server - use background mode
python3 inference.py ... --no-show
```

### Problem: Window shows but is laggy
```bash
# Increase process_every_n_frames for smoother preview
# Edit inference.py and change default from 2 to 3
```

### Problem: Can't interact with window
```bash
# Click on the window first to give it focus
# Then use Q/P keys
```

---

## 🎬 Current Setup Results

From your last run with **OUT object tracking**:
- **IN Count:** 14
- **OUT Count:** 239
- **Total Crossings:** 253
- **Processing Speed:** ~15 FPS

Now with live preview, you can **watch these counts increment in real-time**! 🎉

---

## 📝 Summary

| Mode | Command Flag | Speed | Use Case |
|------|-------------|-------|----------|
| **Live Preview** | (default) | Normal | Development, Testing |
| **Explicit Live** | `--show` | Normal | Force live display |
| **Background** | `--no-show` | Fast | Production, Batch |

**Remember:** 
- Live preview is **ON by default** for videos
- Use `--no-show` for faster background processing
- Press **Q** to stop, **P** to pause/resume

Enjoy your live detection! 🚀
