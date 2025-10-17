# YOLOv12 RF Signal Detection Setup - COMPLETE

**Date**: 2025-10-16
**Status**: Ready for Training (pending label annotation)

---

## Summary

Successfully set up end-to-end pipeline for RF signal detection using YOLOv12:

1. Data extraction from USRP captures
2. Spectrogram generation with Georgia Tech normalization
3. YOLO dataset preparation
4. YOLOv12 training pipeline

---

## What Was Accomplished

### 1. Data Setup (Step 1)
- Copied 12GB of real RF data to `data/raw/`
  - epoch_23.sc16 (5.6GB) + headers
  - test4_2412.sc16 (5.6GB) + headers
  - test4_2412.fc32 (382MB) + headers

### 2. Dependencies (Step 2)
- Installed all Python packages from requirements.txt
- Installed YOLOv12 from GitHub: `sunsmarterjie/yolov12`
- Installed supervision for annotations

### 3. Spectrogram Generation (Steps 3-4)
- Created `generate_first_spec.py` - generates visualization spectrograms
- Created `batch_process.py` - fast batch processing (~332 spectrograms/sec)
- **Generated 1000 spectrograms** in data/spectrograms/epoch23/
  - Size: 256 x 7811 x 3 (RGB)
  - Format: PNG
  - Total size: 31.3 MB
  - Processing time: ~3 seconds

### 4. YOLO Dataset Preparation
- Created YOLO directory structure:
  ```
  data/yolo/
    ├── data.yaml (dataset configuration)
    ├── train/
    │   ├── images/ (800 images)
    │   └── labels/ (800 label files - empty placeholders)
    └── val/
        ├── images/ (200 images)
        └── labels/ (200 label files - empty placeholders)
  ```

### 5. YOLOv12 Training Pipeline
Created complete training workflow:
- `train_yolov12.py` - Main training script
- `val_yolov12.py` - Validation script
- `detect_yolov12.py` - Inference script

---

## Project Structure

```
idc/
├── data/
│   ├── raw/                    # Original USRP captures
│   ├── spectrograms/
│   │   └── epoch23/           # 1000 generated spectrograms
│   └── yolo/
│       ├── data.yaml          # YOLO dataset config
│       ├── train/
│       │   ├── images/        # 800 training images
│       │   └── labels/        # 800 label files (to be annotated)
│       └── val/
│           ├── images/        # 200 validation images
│           └── labels/        # 200 label files (to be annotated)
│
├── results/
│   └── first_real_spectrogram.png  # Visualization
│
├── Scripts:
│   ├── test_header.py             # Parse USRP headers
│   ├── generate_first_spec.py     # Generate single spectrogram
│   ├── batch_process.py           # Batch process spectrograms
│   ├── prepare_yolo_dataset.py    # Prepare YOLO format
│   ├── train_yolov12.py          # Train YOLOv12
│   ├── val_yolov12.py            # Validate model
│   └── detect_yolov12.py         # Run inference
│
└── Configuration:
    ├── requirements.txt           # Python dependencies
    ├── QUICK_START_GUIDE.md      # Original guide
    └── YOLOV12_SETUP_COMPLETE.md # This file
```

---

## Next Steps

### CRITICAL: Annotate Your Data

Before training can begin, you must annotate the spectrograms with bounding boxes:

1. **Choose an annotation tool:**
   - **Roboflow** (recommended): https://roboflow.com
     - Web-based, easy to use
     - Auto-exports to YOLO format
     - Has pre-trained models for assistance

   - **LabelImg**: https://github.com/heartexlabs/labelImg
     - Desktop app
     - Exports directly to YOLO format

   - **CVAT**: https://cvat.ai
     - Web-based, powerful
     - Good for team collaboration

2. **Annotation Guidelines:**
   - **Bluetooth signals**: Narrow vertical streaks (1-2 MHz wide, ~625 μs duration)
   - **WiFi signals**: Wide horizontal bands (20 MHz wide)
   - **Noise**: Background areas

3. **YOLO Label Format:**
   Each .txt file contains one line per object:
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
   All values normalized to [0, 1]

   Example (Bluetooth detection at center):
   ```
   0 0.5 0.5 0.05 0.2
   ```

### Training Workflow

Once labels are ready:

```bash
# 1. Train YOLOv12 model
python train_yolov12.py

# 2. Validate performance
python val_yolov12.py

# 3. Run inference on new data
python detect_yolov12.py
```

### Training Configuration

Current settings in `train_yolov12.py`:
- **Model**: YOLOv12s (small)
- **Epochs**: 100
- **Batch size**: 16
- **Image size**: 640x640
- **Optimizer**: AdamW
- **Learning rate**: 0.01
- **Early stopping**: 20 epochs patience

Adjust based on:
- GPU memory (reduce batch size if OOM errors)
- Dataset size (increase epochs for more data)
- Performance requirements (use yolov12m or yolov12l for better accuracy)

---

## Performance Metrics to Track

After training, monitor:
- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: mAP across IoU thresholds 0.5-0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

Target metrics (estimate):
- mAP50: > 0.7 (good)
- mAP50-95: > 0.5 (good)
- Precision: > 0.8 (minimize false alarms)
- Recall: > 0.7 (catch most signals)

---

## Alternative: Transfer Learning

If you have limited annotated data, consider transfer learning:

1. Find pre-trained YOLOv12 weights
2. Modify `train_yolov12.py`:
   ```python
   model = YOLO('yolov12s.pt')  # Load pre-trained
   model.train(
       data='data/yolo/data.yaml',
       epochs=50,  # Fewer epochs needed
       freeze=10,  # Freeze first 10 layers
       ...
   )
   ```

---

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution**: Reduce batch size in `train_yolov12.py`:
```python
batch=8,  # or smaller
```

### Issue: Training too slow
**Solution**:
- Use smaller model: `yolov12n.yaml` (nano)
- Reduce image size: `imgsz=320`
- Use fewer workers: `workers=4`

### Issue: Poor detection performance
**Solutions**:
- Annotate more data (aim for 500+ per class)
- Use data augmentation (already enabled in YOLO)
- Try larger model: `yolov12m.yaml` or `yolov12l.yaml`
- Adjust confidence threshold in inference

---

## Files Created

| File | Purpose | Status |
|------|---------|--------|
| test_header.py | Parse USRP metadata | Complete |
| generate_first_spec.py | Generate single spectrogram | Complete |
| batch_process.py | Batch spectrogram generation | Complete |
| prepare_yolo_dataset.py | YOLO dataset preparation | Complete |
| train_yolov12.py | YOLOv12 training | Ready to run |
| val_yolov12.py | Model validation | Ready to run |
| detect_yolov12.py | Inference | Ready to run |
| data/yolo/data.yaml | YOLO config | Complete |

---

## Estimated Timeline

Assuming you're annotating manually:

1. **Annotation** (1-2 days):
   - 1000 images × 1-2 min/image = 16-33 hours
   - Can be done in batches

2. **Training** (2-4 hours):
   - 100 epochs @ 1-2 min/epoch
   - Depends on GPU

3. **Validation & Tuning** (1-2 hours):
   - Test different thresholds
   - Analyze failure cases

**Total**: 2-3 days to working model

---

## Additional Resources

- YOLOv12 GitHub: https://github.com/sunsmarterjie/yolov12
- Ultralytics Docs: https://docs.ultralytics.com
- Roboflow Tutorials: https://roboflow.com/learn
- Georgia Tech RF Dataset Paper: (see project references)

---

## Success Criteria

You'll know the system is working when:
1. ✓ Spectrograms generated from real RF data
2. ⏳ Images annotated with bounding boxes (NEXT STEP)
3. ⏳ YOLOv12 trains without errors
4. ⏳ Validation mAP50 > 0.7
5. ⏳ Inference detects Bluetooth signals in new spectrograms

---

**Status**: System is ready for annotation and training!
**Next Action**: Start annotating spectrograms in `data/yolo/train/images/`

