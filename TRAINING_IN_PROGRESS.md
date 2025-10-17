# YOLOv12 Training In Progress

**Started**: 2025-10-17 03:37 UTC
**Status**: Running (Epoch 1/100)

---

## Summary

Successfully generated auto-labels and started YOLOv12 training!

### Auto-Labeling Results

**Script**: `auto_label_spectrograms.py`
**Method**: Signal processing-based detection
- Column energy analysis for Bluetooth (narrow vertical streaks)
- Row energy analysis for WiFi (wide horizontal bands)

**Results**:
- **Training set**: 800 images, 687 labels generated (0.9 avg/image)
- **Validation set**: 200 images, 182 labels generated (0.9 avg/image)
- **Debug images**: 30 samples saved to `results/auto_labels_debug/`
  - Green boxes = Bluetooth signals
  - Blue boxes = WiFi signals

### Dataset Statistics

**Training**:
- Total images: 800
- Images with labels: 605 (75.6%)
- Background images: 195 (24.4%)

**Validation**:
- Total images: 200
- Images with labels: 150 (75%)
- Background images: 50 (25%)

This distribution is healthy - about 75% have objects, 25% are backgrounds.

---

## YOLOv12 Model Configuration

**Architecture**: YOLOv12s (small)
- **Parameters**: 9,097,625 (9.1M)
- **Layers**: 497
- **GFLOPs**: 19.6

**Training Configuration**:
- **Device**: CPU (Snapdragon X 12-core)
- **Epochs**: 100
- **Batch size**: 16
- **Image size**: 640x640
- **Optimizer**: AdamW
- **Learning rate**: 0.01
- **Early stopping**: 20 epochs patience

**Classes**:
- Class 0: Bluetooth
- Class 1: WiFi
- Class 2: Noise (not detected by auto-labeling)

---

## Training Progress

Training started successfully. Monitoring:
- box_loss: Bounding box regression loss
- cls_loss: Classification loss
- dfl_loss: Distribution Focal Loss

Checkpoints saved every 10 epochs to: `runs/train/yolov12_rf_detection/weights/`

---

## Expected Timeline

**On CPU** (Snapdragon X 12-core):
- Estimated: ~2-5 minutes/epoch
- Total time: ~3-8 hours for 100 epochs
- Early stopping may reduce this

**Note**: Training on CPU is slow but functional. For faster training, use a CUDA-enabled GPU.

---

## Monitoring Training

Check progress:
```bash
# View latest output
tail -f runs/train/yolov12_rf_detection/results.txt

# Or check logs directory
ls -lh runs/train/yolov12_rf_detection/
```

---

## Next Steps

### 1. Monitor Training (now)
Wait for training to complete. Watch for:
- Decreasing losses
- Improving mAP scores on validation set
- Early stopping if no improvement

### 2. Evaluate Results (after training)
```bash
python val_yolov12.py
```

Expected metrics for first iteration:
- mAP50: 0.3-0.5 (decent for auto-labels)
- mAP50-95: 0.2-0.3
- Precision: 0.4-0.6
- Recall: 0.3-0.5

### 3. Review Predictions
```bash
python detect_yolov12.py
```

Check `runs/detect/yolov12_rf_detection/` for annotated images.

### 4. Iterative Improvement

**Option A: Refine Auto-Labels**
1. Review debug images in `results/auto_labels_debug/`
2. Adjust detection thresholds in `auto_label_spectrograms.py`
3. Re-run auto-labeling and training

**Option B: Manual Annotation**
1. Load images into Roboflow/LabelImg
2. Correct auto-generated labels
3. Add missing detections
4. Re-train with corrected labels

**Option C: Active Learning**
1. Use trained model to predict on unlabeled data
2. Manually correct high-confidence errors
3. Add corrected samples to training set
4. Re-train

---

## Auto-Labeling Algorithm Details

### Bluetooth Detection
- **Feature**: Narrow vertical streaks
- **Method**: Column-wise energy peak detection
- **Parameters**:
  - Min distance between peaks: 5 pixels
  - Energy prominence: 0.15
  - Width range: 1-15 pixels (~ 1-3 MHz)
  - Max frequency height: 30% of spectrum

### WiFi Detection
- **Feature**: Wide horizontal bands
- **Method**: Row-wise energy threshold + connected components
- **Parameters**:
  - Energy threshold: mean + 0.5 * std
  - Min height: 10 rows
  - Time width: 90% of spectrogram
  - Height range: 10-60% of spectrum

### Limitations
- May miss weak signals
- May create false positives on noise spikes
- Fixed thresholds may not work for all spectrograms
- No detection of class 2 (noise) - only background

**Improvement**: Manually review and correct labels for best results.

---

## Files Created

| File | Purpose |
|------|---------|
| auto_label_spectrograms.py | Auto-generate YOLO labels |
| results/auto_labels_debug/*.png | Visualization of detections |
| data/yolo/train/labels/*.txt | Training labels (YOLO format) |
| data/yolo/val/labels/*.txt | Validation labels (YOLO format) |

---

## Troubleshooting

### Issue: Training too slow on CPU
**Solution**: Reduce batch size or use GPU
```python
# In train_yolov12.py
batch=8,  # Reduce from 16
epochs=50,  # Reduce epochs for testing
```

### Issue: Poor detection performance
**Possible causes**:
1. Auto-labels are imperfect
2. Need more training data
3. Model too simple for task

**Solutions**:
1. Manually review and correct labels
2. Generate more spectrograms from remaining data
3. Try larger model (yolov12m)

### Issue: OOM (Out of Memory)
**Solution**: Reduce batch size
```python
batch=4,  # or even smaller
```

---

## Success Metrics

After training completes, success is achieved if:
- mAP50 > 0.5 (good for auto-labeled data)
- Visual inspection shows correct detections
- False positive rate < 20%
- False negative rate < 30%

If metrics are poor, iterate with manual annotation!

---

**Status**: Training running in background
**Command**: `python train_yolov12.py`
**Output**: `runs/train/yolov12_rf_detection/`

Training will automatically save best weights and stop early if no improvement for 20 epochs.
