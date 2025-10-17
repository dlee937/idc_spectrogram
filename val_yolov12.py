"""
YOLOv12 Validation Script
Evaluate trained model on validation set
"""
from ultralytics import YOLO

print("=" * 60)
print("YOLOv12 RF Signal Detection Validation")
print("=" * 60)

# Load trained model
model_path = 'runs/train/yolov12_rf_detection/weights/best.pt'
print(f"\nLoading model from: {model_path}")
model = YOLO(model_path)

# Validate the model
print("\nRunning validation...")
metrics = model.val(
    data='data/yolo/data.yaml',
    imgsz=640,
    batch=16,
    conf=0.25,  # confidence threshold
    iou=0.6,    # IoU threshold for NMS
    plots=True,  # save validation plots
)

# Print results
print("\n" + "=" * 60)
print("Validation Results")
print("=" * 60)
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
