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
    Detect Bluetooth and WiFi signals in a spectrogram image
    Returns YOLO format labels: class x_center y_center width height
    """
    # Read image
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Normalize
    gray_norm = gray.astype(np.float32) / 255.0

    labels = []

    # --- Bluetooth Detection ---
    # Bluetooth appears as narrow vertical streaks (high energy, short time bursts)
    # Strategy: Look for vertical columns with high energy

    # Calculate column-wise energy
    col_energy = np.mean(gray_norm, axis=0)

    # Find peaks in column energy (potential Bluetooth bursts)
    # Bluetooth is typically 1-2 MHz wide, which is ~5-10 pixels at 256 freq bins over 20 MHz
    min_distance = 5  # Minimum separation between bursts
    prominence = 0.15  # Minimum prominence above background

    peaks, properties = find_peaks(col_energy,
                                    distance=min_distance,
                                    prominence=prominence,
                                    width=(1, 15))  # Width in pixels (1-15 pixels ~ 1-3 MHz)

    # For each peak, create a bounding box
    for peak_idx, peak_pos in enumerate(peaks):
        # Get width from peak properties
        left_base = int(properties['left_bases'][peak_idx])
        right_base = int(properties['right_bases'][peak_idx])

        # Temporal extent
        time_width = right_base - left_base
        time_center = (left_base + right_base) / 2

        # Find vertical extent (frequency range with high energy)
        col_slice = gray_norm[:, peak_pos]

        # Find rows with energy above threshold
        energy_threshold = np.mean(col_slice) + 0.3 * np.std(col_slice)
        active_rows = np.where(col_slice > energy_threshold)[0]

        if len(active_rows) > 0:
            freq_min = active_rows.min()
            freq_max = active_rows.max()
            freq_center = (freq_min + freq_max) / 2
            freq_height = freq_max - freq_min

            # Filter: Bluetooth is typically narrow in frequency
            if freq_height < height * 0.3:  # Less than 30% of freq range
                # Convert to YOLO format (normalized)
                x_center = time_center / width
                y_center = freq_center / height
                box_width = max(time_width / width, 0.01)  # Minimum width
                box_height = max(freq_height / height, 0.05)  # Minimum height

                # Class 0 = Bluetooth
                labels.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # --- WiFi Detection ---
    # WiFi appears as wide horizontal bands (20 MHz channels)
    # Strategy: Look for horizontal rows with sustained high energy

    # Calculate row-wise energy
    row_energy = np.mean(gray_norm, axis=1)

    # Find sustained high-energy regions
    # WiFi occupies ~20 MHz = ~256 pixels @ 20 MHz = full bandwidth
    # Look for horizontal bands

    # Smooth row energy to find bands
    row_energy_smooth = ndimage.gaussian_filter1d(row_energy, sigma=5)

    # Find regions above threshold
    wifi_threshold = np.mean(row_energy_smooth) + 0.5 * np.std(row_energy_smooth)
    wifi_regions = row_energy_smooth > wifi_threshold

    # Find connected components (bands)
    labeled, num_features = ndimage.label(wifi_regions)

    for region_id in range(1, num_features + 1):
        region_mask = labeled == region_id
        region_rows = np.where(region_mask)[0]

        if len(region_rows) > 10:  # Minimum height for WiFi band
            freq_min = region_rows.min()
            freq_max = region_rows.max()
            freq_center = (freq_min + freq_max) / 2
            freq_height = freq_max - freq_min

            # WiFi typically spans most of the time
            # Use full width for now
            time_center = 0.5
            time_width = 0.9  # Most of the spectrogram

            # Convert to YOLO format
            x_center = time_center
            y_center = freq_center / height
            box_width = time_width
            box_height = freq_height / height

            # Filter: WiFi should be relatively wide
            if box_height > 0.1 and box_height < 0.6:
                # Class 1 = WiFi
                labels.append(f"1 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

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
