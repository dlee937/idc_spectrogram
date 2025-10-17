"""
Automatic Label Generation for RF Spectrograms
Generates initial YOLO format labels based on signal processing techniques
"""
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from scipy.signal import find_peaks

def detect_signals_in_spectrogram(img_path, debug=False):
    """
    Detect green regions (signals) in viridis spectrogram
    Returns YOLO format labels: class x_center y_center width height
    """
    # Read image
    img = cv2.imread(str(img_path))
    height, width = img.shape[:2]

    # Convert BGR to HSV for better green detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for green/yellow-green colors in viridis colormap
    # Viridis green range: H: 40-85, S: 100-255, V: 100-255
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Create mask for green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    labels = []

    # Process each contour
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out very small regions (noise)
        min_area = 50  # minimum pixels
        if w * h < min_area:
            continue

        # Calculate center and normalize to [0, 1]
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        box_width = w / width
        box_height = h / height

        # Ensure minimum box size
        box_width = max(box_width, 0.01)
        box_height = max(box_height, 0.01)

        # Classify based on characteristics
        # Narrow vertical = Bluetooth (class 0)
        # Wide horizontal = WiFi (class 1)
        aspect_ratio = w / h if h > 0 else 0

        if aspect_ratio < 0.5:  # Tall and narrow
            cls = 0  # Bluetooth
        else:
            cls = 1  # WiFi/other signals

        labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Debug visualization
    if debug:
        debug_img = img.copy()
        for label in labels:
            parts = label.split()
            cls = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])

            # Convert to pixel coordinates
            x1 = int((x_c - w/2) * width)
            y1 = int((y_c - h/2) * height)
            x2 = int((x_c + w/2) * width)
            y2 = int((y_c + h/2) * height)

            color = (0, 255, 0) if cls == 0 else (255, 0, 0)  # Green=BT, Blue=WiFi
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)

        return labels, debug_img

    return labels

# Process test images from test4_2412 spectrograms
source_dir = Path('data/spectrograms/test4_2412')
train_img_dir = Path('data/yolo/train/images')
train_label_dir = Path('data/yolo/train/labels')
val_img_dir = Path('data/yolo/val/images')
val_label_dir = Path('data/yolo/val/labels')

debug_dir = Path('results/auto_labels_debug')
debug_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Automatic Label Generation for RF Spectrograms")
print("=" * 60)
print(f"\nSource: {source_dir}")

# Get all source images
all_images = sorted(source_dir.glob('*.png'))
print(f"Found {len(all_images)} spectrograms")

# Split 80/20 for train/val
split_idx = int(len(all_images) * 0.8)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# Process training images
print(f"\nProcessing {len(train_images)} training images...")
total_train_labels = 0

for i, img_path in enumerate(tqdm(train_images, desc="Train")):
    # Generate labels
    if i < 20:  # Save debug images for first 20
        labels, debug_img = detect_signals_in_spectrogram(img_path, debug=True)
        cv2.imwrite(str(debug_dir / f"train_{img_path.stem}_debug.png"), debug_img)
    else:
        labels = detect_signals_in_spectrogram(img_path, debug=False)

    # Write label file
    label_path = train_label_dir / (img_path.stem + '.txt')
    with open(label_path, 'w') as f:
        f.write('\n'.join(labels))

    total_train_labels += len(labels)

# Process validation images
print(f"\nProcessing {len(val_images)} validation images...")
total_val_labels = 0

for i, img_path in enumerate(tqdm(val_images, desc="Val")):
    # Generate labels
    if i < 10:  # Save debug images for first 10
        labels, debug_img = detect_signals_in_spectrogram(img_path, debug=True)
        cv2.imwrite(str(debug_dir / f"val_{img_path.stem}_debug.png"), debug_img)
    else:
        labels = detect_signals_in_spectrogram(img_path, debug=False)

    # Write label file
    label_path = val_label_dir / (img_path.stem + '.txt')
    with open(label_path, 'w') as f:
        f.write('\n'.join(labels))

    total_val_labels += len(labels)

print("\n" + "=" * 60)
print("Label Generation Complete!")
print("=" * 60)
print(f"\nTraining set:")
print(f"  Images: {len(train_images)}")
print(f"  Total labels: {total_train_labels}")
print(f"  Avg labels/image: {total_train_labels/len(train_images):.1f}")

print(f"\nValidation set:")
print(f"  Images: {len(val_images)}")
print(f"  Total labels: {total_val_labels}")
print(f"  Avg labels/image: {total_val_labels/len(val_images):.1f}")

print(f"\nDebug images saved to: {debug_dir}")
print("  - Green boxes = Bluetooth")
print("  - Blue boxes = WiFi")

print("\n" + "=" * 60)
print("Next Steps:")
print("=" * 60)
print("1. Review debug images in results/auto_labels_debug/")
print("2. Manually correct labels in Roboflow/LabelImg if needed")
print("3. Run training: python train_yolov12.py")
print("\nNote: Auto-generated labels are a starting point.")
print("Manual review recommended for best results!")
