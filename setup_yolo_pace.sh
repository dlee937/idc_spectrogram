#!/bin/bash
# Setup YOLO training environment on PACE ICE
# Run this before training YOLO models

echo "=========================================="
echo "Setting up YOLO Training Environment"
echo "=========================================="

# Load modules
echo "Loading modules..."
module load anaconda3
module load cuda/11.8  # Adjust version as needed

# Activate environment
echo "Activating conda environment..."
source activate rf_signal

# Install YOLO and deep learning dependencies
echo ""
echo "Installing PyTorch and YOLO dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics  # YOLOv8/v12
pip install tensorboard

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || echo "CUDA not available"
python -c "import ultralytics; print(f'Ultralytics version: {ultralytics.__version__}')"

# Create data.yaml with correct paths for PACE
echo ""
echo "Updating data.yaml for PACE paths..."
cat > data/yolo/data.yaml << EOF
# YOLOv12 Dataset Configuration for RF Signal Detection
# Bluetooth signal detection in spectrograms

# Dataset paths (relative to this file or absolute)
path: $HOME/idc_spectrogram/data/yolo  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')

# Classes
names:
  0: bluetooth
  1: wifi
  2: noise

# Number of classes
nc: 3
EOF

# Check if images exist
echo ""
echo "Checking dataset..."
TRAIN_IMAGES=$(ls data/yolo/train/images/*.png 2>/dev/null | wc -l)
VAL_IMAGES=$(ls data/yolo/val/images/*.png 2>/dev/null | wc -l)

echo "Training images: $TRAIN_IMAGES"
echo "Validation images: $VAL_IMAGES"

if [ $TRAIN_IMAGES -eq 0 ]; then
    echo ""
    echo "WARNING: No training images found!"
    echo "You need to either:"
    echo "  1. Transfer YOLO dataset from local machine"
    echo "  2. Copy generated spectrograms to data/yolo/train/images/"
    echo ""
    echo "To copy spectrograms:"
    echo "  cp data/spectrograms/test4_2412/*.png data/yolo/train/images/"
    echo ""
    echo "Note: You'll need to create label files (.txt) for each image"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Ensure you have training images and labels"
echo "  2. Submit training job: sbatch train_yolo_pace.sh"
echo "  3. Monitor: squeue -u \$USER"
echo "  4. Check logs: tail -f logs/yolo_train_*.out"
