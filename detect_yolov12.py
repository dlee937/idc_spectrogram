"""
YOLOv12 Inference Script
Run detection on new spectrogram images
"""
from ultralytics import YOLO
from pathlib import Path

print("=" * 60)
print("YOLOv12 RF Signal Detection Inference")
print("=" * 60)

# Load trained model
model_path = 'runs/train/yolov12_rf_detection/weights/best.pt'
print(f"\nLoading model from: {model_path}")
model = YOLO(model_path)

# Run inference on a directory of images
source = 'data/yolo/val/images'  # or path to new spectrograms
print(f"Running inference on: {source}")

results = model.predict(
    source=source,
    imgsz=640,
    conf=0.25,      # confidence threshold
    iou=0.6,        # IoU threshold for NMS
    save=True,      # save annotated images
    save_txt=True,  # save detection results as .txt
    save_conf=True, # save confidence scores
    project='runs/detect',
    name='yolov12_rf_detection',
    exist_ok=True,
)

print("\n" + "=" * 60)
print("Inference Complete!")
print("=" * 60)
print(f"\nResults saved to: runs/detect/yolov12_rf_detection")
print("\nDetection Summary:")
for i, result in enumerate(results[:10]):  # Show first 10 results
    img_path = Path(result.path).name
    num_detections = len(result.boxes)
    print(f"  {img_path}: {num_detections} objects detected")
