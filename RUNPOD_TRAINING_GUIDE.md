# YOLOv11 Training on RunPod RTX 5090 - Complete Guide

## 📋 Prerequisites
- RunPod account with sufficient credits
- Dataset ready (OUT_MOBILE.v1i.yolov11 folder)
- Basic knowledge of terminal/command line

---

## 🚀 Step-by-Step Setup Guide

### Step 1: Rent RTX 5090 on RunPod

1. **Login to RunPod**
   - Go to https://www.runpod.io/
   - Sign in to your account

2. **Deploy GPU Instance**
   - Click "Deploy" → "GPU Instances"
   - Filter by GPU type: Select **RTX 5090**
   - Choose template: **PyTorch 2.x** or **RunPod Pytorch** (latest)
   - Disk size: **50 GB** minimum (100 GB recommended)
   - Click "Deploy On-Demand" or "Deploy Spot" (cheaper but can be interrupted)

3. **Wait for Pod to Start**
   - Your pod will initialize in 1-2 minutes
   - Status will change to "Running"

4. **Connect to Pod**
   - Click "Connect" button
   - Choose one of these options:
     - **Jupyter Lab** (easiest for beginners)
     - **SSH** (for command line users)
     - **Web Terminal** (built-in browser terminal)

---

### Step 2: Upload Dataset to RunPod

#### Option A: Using Jupyter Lab (Recommended for Beginners)

1. Open Jupyter Lab from RunPod interface
2. In Jupyter, use the upload button to upload files:
   - Upload `OUT_MOBILE.v1i.yolov11.zip` (zip your dataset first)
   - Or upload the entire folder structure

3. Extract if zipped:
   ```bash
   unzip OUT_MOBILE.v1i.yolov11.zip
   ```

#### Option B: Using SCP (Fast for Large Datasets)

1. Get SSH credentials from RunPod pod details
2. From your local machine:
   ```bash
   scp -r "/home/rasheeque/VS CODE FOLDER/MARUTI/OUT_MOBILE.v1i.yolov11" \
       root@<pod-ip>:/workspace/
   ```

#### Option C: Using Google Drive/Dropbox

1. Upload dataset to cloud storage
2. In RunPod terminal:
   ```bash
   # For Google Drive (using gdown)
   pip install gdown
   gdown --folder <your-google-drive-link>
   
   # Or use wget for direct links
   wget <your-dataset-link>
   unzip dataset.zip
   ```

---

### Step 3: Upload Training Scripts

Upload these files to RunPod's `/workspace/` directory:
- `train_yolov11.py`
- `requirements.txt`
- `setup_runpod.sh`
- `inference.py`

**Using Jupyter Lab:**
1. Navigate to `/workspace/`
2. Use upload button to upload each file

**Using SCP:**
```bash
scp train_yolov11.py root@<pod-ip>:/workspace/
scp requirements.txt root@<pod-ip>:/workspace/
scp setup_runpod.sh root@<pod-ip>:/workspace/
scp inference.py root@<pod-ip>:/workspace/
```

---

### Step 4: Run Setup Script

Open a terminal in RunPod (Jupyter Lab → File → New → Terminal) and run:

```bash
cd /workspace
chmod +x setup_runpod.sh
bash setup_runpod.sh
```

This will:
- Install all dependencies
- Verify GPU is working
- Download YOLOv11 base model
- Takes about 5-10 minutes

**Expected Output:**
```
✅ Setup completed successfully!
GPU: NVIDIA GeForce RTX 5090
CUDA available: True
```

---

### Step 5: Verify Dataset Structure

Make sure your dataset is structured correctly:

```bash
cd /workspace
ls -la OUT_MOBILE.v1i.yolov11/
```

**Expected structure:**
```
OUT_MOBILE.v1i.yolov11/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

**Verify data.yaml paths:**
```bash
cat OUT_MOBILE.v1i.yolov11/data.yaml
```

The paths should be relative. If they start with `../`, they're correct.

---

### Step 6: Start Training

```bash
cd /workspace
python train_yolov11.py
```

**What happens during training:**

1. **Environment Check** - Verifies GPU, CUDA, dataset
2. **Model Loading** - Downloads YOLOv11 nano weights (~6 MB)
3. **Training Starts** - You'll see:
   ```
   Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     1/300      8.5G      1.234      0.856      1.123         45        512
     2/300      8.5G      1.156      0.789      1.089         42        512
   ```

**Training Progress:**
- **Epochs**: 300 total (can be stopped early if you want)
- **Time**: ~2-4 hours on RTX 5090 (depends on batch size)
- **GPU Usage**: Should see ~80-95% GPU utilization
- **Checkpoints**: Saved every 10 epochs

**Monitor Progress:**
- Losses should decrease over time
- mAP should increase
- Best model automatically saved

---

### Step 7: Monitor Training

#### Option A: Watch Terminal Output
- Training progress shows in real-time
- Loss values, mAP, learning rate displayed each epoch

#### Option B: TensorBoard (Advanced)
```bash
# In a new terminal
tensorboard --logdir mobile_out_detection/train --host 0.0.0.0 --port 6006
```
Then access via RunPod's port forwarding

#### Option C: Check Results Folder
```bash
# View training plots
ls mobile_out_detection/train/
```

Files created:
- `weights/best.pt` - Best model (use this for inference!)
- `weights/last.pt` - Latest checkpoint
- `results.png` - Training curves
- `confusion_matrix.png` - Classification performance
- `PR_curve.png` - Precision-Recall curve

---

### Step 8: Training Complete - Download Model

Once training finishes:

```bash
cd /workspace/mobile_out_detection/train/weights/
ls -lh best.pt
```

**Download best.pt to your local machine:**

#### Using Jupyter Lab:
1. Navigate to `mobile_out_detection/train/weights/`
2. Right-click `best.pt` → Download

#### Using SCP:
```bash
# Run on your local machine
scp root@<pod-ip>:/workspace/mobile_out_detection/train/weights/best.pt ./
```

---

### Step 9: Test Your Model

Test on validation images:

```bash
# Single image
python inference.py \
    --model mobile_out_detection/train/weights/best.pt \
    --source OUT_MOBILE.v1i.yolov11/valid/images/example.jpg \
    --output result.jpg \
    --show

# Entire validation folder
python inference.py \
    --model mobile_out_detection/train/weights/best.pt \
    --source OUT_MOBILE.v1i.yolov11/valid/images/ \
    --output results/ \
    --save-json results.json

# Video processing
python inference.py \
    --model mobile_out_detection/train/weights/best.pt \
    --source video.mp4 \
    --output output_video.mp4 \
    --conf 0.25
```

**Inference Arguments:**
- `--model`: Path to trained model (best.pt)
- `--source`: Image, video, or directory
- `--output`: Where to save results
- `--conf`: Confidence threshold (0.1-0.9, default: 0.25)
- `--show`: Display results in window
- `--save-json`: Save detection data to JSON

---

### Step 10: Optimize Training (Optional)

If you want to adjust training for better results:

Edit `train_yolov11.py` before training:

```python
# For faster training (less accurate)
MODEL_SIZE = "yolo11n.pt"  # nano - fastest
EPOCHS = 150
BATCH_SIZE = 64

# For better accuracy (slower)
MODEL_SIZE = "yolo11m.pt"  # medium
EPOCHS = 500
BATCH_SIZE = 32

# For best accuracy (slowest)
MODEL_SIZE = "yolo11x.pt"  # extra large
EPOCHS = 800
BATCH_SIZE = 16
```

**Model Sizes:**
- `yolo11n` - Nano: Fastest, smallest, 2.6M params
- `yolo11s` - Small: Balanced, 9.4M params
- `yolo11m` - Medium: Good accuracy, 20.1M params
- `yolo11l` - Large: Better accuracy, 25.3M params
- `yolo11x` - XLarge: Best accuracy, 56.9M params

---

## 💰 Cost Optimization Tips

1. **Use Spot Instances**: 50-70% cheaper than on-demand
2. **Stop Pod When Not Training**: Pay only for training time
3. **Download Results First**: Before stopping pod
4. **Use Smaller Models**: Start with yolo11n, upgrade if needed

**Estimated Costs (RTX 5090):**
- On-Demand: ~$1.50-2.50/hour
- Spot Instance: ~$0.50-1.00/hour
- Training Time: 2-4 hours
- **Total Cost**: $2-10 per training run

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `train_yolov11.py`:
```python
BATCH_SIZE = 32  # or 16
```

### Issue: "Dataset not found"
**Solution:** Check data.yaml paths:
```bash
cat OUT_MOBILE.v1i.yolov11/data.yaml
```
Ensure paths are correct relative paths.

### Issue: "Model not improving"
**Solutions:**
- Increase epochs: `EPOCHS = 500`
- Adjust learning rate: `LEARNING_RATE = 0.0005`
- Enable augmentation: `AUGMENT = True`

### Issue: "Too slow"
**Solution:** Verify GPU is being used:
```python
python -c "import torch; print(torch.cuda.is_available())"
```
Should print `True`.

### Issue: "Class imbalance warnings"
**Already handled!** The script uses class weights to handle the MOBILE (3.5%) vs OUT (96.5%) imbalance.

---

## 📊 Understanding Results

After training, check these metrics in `results.png`:

- **mAP50**: Mean Average Precision at 50% IOU (target: >0.70)
- **mAP50-95**: Stricter metric (target: >0.50)
- **Precision**: How many detections are correct (target: >0.80)
- **Recall**: How many objects are found (target: >0.75)

**Good model indicators:**
- mAP50 > 0.70
- Low and stable loss values
- High precision and recall

---

## 🎯 Next Steps After Training

1. **Test on real images/videos** using `inference.py`
2. **Fine-tune** if results aren't satisfactory
3. **Export model** for deployment:
   ```python
   from ultralytics import YOLO
   model = YOLO('best.pt')
   model.export(format='onnx')  # For production deployment
   ```
4. **Deploy** to your application

---

## 📞 Need Help?

Common commands cheatsheet:

```bash
# Check GPU usage while training
watch -n 1 nvidia-smi

# Stop training gracefully
# Press Ctrl+C once (it will save checkpoint)

# Resume training from checkpoint
# Edit train_yolov11.py: resume=True

# View training logs
tail -f mobile_out_detection/train/results.txt

# Compress results for download
tar -czf training_results.tar.gz mobile_out_detection/
```

---

## ✅ Quick Start Commands Summary

```bash
# 1. Setup environment
cd /workspace
bash setup_runpod.sh

# 2. Start training
python train_yolov11.py

# 3. Test model after training
python inference.py \
    --model mobile_out_detection/train/weights/best.pt \
    --source OUT_MOBILE.v1i.yolov11/valid/images/ \
    --output results/ \
    --conf 0.25

# 4. Download results
# Use Jupyter interface or SCP to download best.pt
```

---

**Good luck with your training! 🚀**
