"""
Prepare YOLO dataset from generated spectrograms
This script copies spectrograms to YOLO format and creates dummy labels for initial testing
"""
import shutil
from pathlib import Path

# Source and destination paths
src_dir = Path('data/spectrograms/epoch23')
train_dir = Path('data/yolo/train/images')
val_dir = Path('data/yolo/val/images')
train_labels = Path('data/yolo/train/labels')
val_labels = Path('data/yolo/val/labels')

# Get all spectrogram images
spec_files = sorted(src_dir.glob('*.png'))
print(f"Found {len(spec_files)} spectrogram images")

# Split into train/val (80/20)
split_idx = int(len(spec_files) * 0.8)
train_files = spec_files[:split_idx]
val_files = spec_files[split_idx:]

print(f"Train: {len(train_files)} images")
print(f"Val: {len(val_files)} images")

# Copy training files
print("\nCopying training images...")
for img_file in train_files:
    dest = train_dir / img_file.name
    shutil.copy(img_file, dest)

    # Create dummy label file (empty for now - you'll need to annotate these)
    label_file = train_labels / (img_file.stem + '.txt')
    label_file.touch()

# Copy validation files
print("Copying validation images...")
for img_file in val_files:
    dest = val_dir / img_file.name
    shutil.copy(img_file, dest)

    # Create dummy label file
    label_file = val_labels / (img_file.stem + '.txt')
    label_file.touch()

print(f"\nDataset prepared successfully!")
print(f"Train images: {len(list(train_dir.glob('*.png')))}")
print(f"Val images: {len(list(val_dir.glob('*.png')))}")
print(f"\nNOTE: Label files are empty placeholders.")
print("You'll need to annotate the spectrograms using tools like:")
print("  - Roboflow (https://roboflow.com)")
print("  - LabelImg (https://github.com/heartexlabs/labelImg)")
print("  - CVAT (https://cvat.ai)")
