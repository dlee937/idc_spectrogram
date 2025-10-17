"""
YOLOv12 Training Script for RF Signal Detection
Train YOLOv12 on spectrogram images to detect Bluetooth, WiFi, and noise patterns
"""
from ultralytics import YOLO
import torch

print("=" * 60)
print("YOLOv12 RF Signal Detection Training")
print("=" * 60)

# Check CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Initialize YOLOv12 model
# Model sizes: yolov12n (nano), yolov12s (small), yolov12m (medium), yolov12l (large), yolov12x (extra-large)
print("\nInitializing YOLOv12s model...")
model = YOLO('yolov12s.yaml')  # Start from scratch with YOLOv12-small architecture

# Alternatively, you can load pre-trained weights if available:
# model = YOLO('yolov12s.pt')

print("\nModel architecture loaded successfully!")
print("\nTraining Configuration:")
print(f"  - Dataset: data/yolo/data.yaml")
print(f"  - Epochs: 100")
print(f"  - Batch size: 16")
print(f"  - Image size: 640x640")
print(f"  - Device: {device}")

# Train the model
print("\n" + "=" * 60)
print("Starting Training...")
print("=" * 60)

results = model.train(
    data='data/yolo/data.yaml',  # path to data.yaml
    epochs=100,                   # number of training epochs
    imgsz=640,                    # input image size
    batch=16,                     # batch size (adjust based on GPU memory)
    device=device,                # device to use (cuda or cpu)
    workers=8,                    # number of dataloader workers
    patience=20,                  # early stopping patience
    save=True,                    # save checkpoints
    save_period=10,               # save checkpoint every N epochs
    project='runs/train',         # project directory
    name='yolov12_rf_detection',  # experiment name
    exist_ok=True,                # overwrite existing experiment
    pretrained=False,             # use pretrained weights
    optimizer='AdamW',            # optimizer (SGD, Adam, AdamW, etc.)
    lr0=0.01,                     # initial learning rate
    momentum=0.937,               # SGD momentum/Adam beta1
    weight_decay=0.0005,          # optimizer weight decay
    warmup_epochs=3.0,            # warmup epochs
    warmup_momentum=0.8,          # warmup initial momentum
    warmup_bias_lr=0.1,           # warmup initial bias lr
    box=7.5,                      # box loss gain
    cls=0.5,                      # cls loss gain
    dfl=1.5,                      # dfl loss gain
    plots=True,                   # save plots during training
    verbose=True,                 # verbose output
)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"\nBest model saved to: {results.save_dir}")
print("\nNext steps:")
print("  1. Annotate your spectrograms with actual labels")
print("  2. Re-run training with annotated data")
print("  3. Evaluate model performance: python val_yolov12.py")
print("  4. Run inference: python detect_yolov12.py")
