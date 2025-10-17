#!/bin/bash
# Prepare YOLO dataset from generated spectrograms
# Creates train/val split and placeholder labels

echo "=========================================="
echo "Preparing YOLO Dataset from Spectrograms"
echo "=========================================="

# Create directories
mkdir -p data/yolo/train/images
mkdir -p data/yolo/train/labels
mkdir -p data/yolo/val/images
mkdir -p data/yolo/val/labels

# Count available spectrograms
TOTAL_SPECS=$(ls data/spectrograms/test4_2412/*.png 2>/dev/null | wc -l)
echo "Found $TOTAL_SPECS spectrograms"

if [ $TOTAL_SPECS -eq 0 ]; then
    echo "ERROR: No spectrograms found in data/spectrograms/test4_2412/"
    echo "Please run batch_process.py first to generate spectrograms"
    exit 1
fi

# Calculate split (80% train, 20% val)
TRAIN_COUNT=$((TOTAL_SPECS * 80 / 100))
VAL_COUNT=$((TOTAL_SPECS - TRAIN_COUNT))

echo "Splitting dataset:"
echo "  Training: $TRAIN_COUNT images"
echo "  Validation: $VAL_COUNT images"

# Copy training images
echo ""
echo "Copying training images..."
ls data/spectrograms/test4_2412/*.png | head -n $TRAIN_COUNT | while read img; do
    cp "$img" data/yolo/train/images/
    # Create empty label file (you'll need to annotate these later)
    basename=$(basename "$img" .png)
    touch "data/yolo/train/labels/${basename}.txt"
done

# Copy validation images
echo "Copying validation images..."
ls data/spectrograms/test4_2412/*.png | tail -n $VAL_COUNT | while read img; do
    cp "$img" data/yolo/val/images/
    # Create empty label file
    basename=$(basename "$img" .png)
    touch "data/yolo/val/labels/${basename}.txt"
done

# Verify
TRAIN_IMAGES=$(ls data/yolo/train/images/*.png 2>/dev/null | wc -l)
VAL_IMAGES=$(ls data/yolo/val/images/*.png 2>/dev/null | wc -l)

echo ""
echo "=========================================="
echo "Dataset Preparation Complete!"
echo "=========================================="
echo "Training images: $TRAIN_IMAGES"
echo "Validation images: $VAL_IMAGES"
echo ""
echo "IMPORTANT: Label files created but empty!"
echo ""
echo "Next steps:"
echo "  1. Annotate images (create .txt label files)"
echo "     - Use tools like labelImg or roboflow"
echo "     - Or use auto_label_spectrograms.py for automatic labeling"
echo "  2. Run: ./setup_yolo_pace.sh (install dependencies)"
echo "  3. Submit training: sbatch train_yolo_pace.sh"
echo ""
echo "To use auto-labeling:"
echo "  python auto_label_spectrograms.py"
