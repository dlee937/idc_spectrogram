#!/bin/bash
#SBATCH -J yolo_rf_train             # Job name
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 8                         # Number of CPU cores
#SBATCH --mem=64GB                   # Memory per node
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH -t 16:00:00                  # Time limit (16 hours)
#SBATCH -p coc-gpu                   # Partition (COC GPU partition)
#SBATCH -q coc-ice                   # QoS (required for coc-gpu partition)
#SBATCH -o logs/yolo_train_%j.out    # Output file
#SBATCH -e logs/yolo_train_%j.err    # Error file
#SBATCH --mail-type=END,FAIL         # Email notifications
#SBATCH --mail-user=dlee937@gatech.edu

# Print job info
echo "=========================================="
echo "YOLO RF Signal Detection Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load modules
echo "Loading modules..."
module load anaconda3
module load cuda/11.8  # Adjust CUDA version as needed

# Activate environment
echo "Activating conda environment..."
source activate rf_signal

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Go to project directory
cd $HOME/idc_spectrogram

# Create necessary directories
mkdir -p logs
mkdir -p runs/train

# Print configuration
echo "=========================================="
echo "Training Configuration:"
echo "=========================================="
echo "Dataset: data/yolo/data.yaml"
echo "Model: YOLOv12s"
echo "Epochs: 100"
echo "Batch size: 16"
echo "Image size: 640"
echo "Device: GPU"
echo ""

# Run training
echo "=========================================="
echo "Starting YOLO Training..."
echo "=========================================="
python train_yolov12.py

# Training complete
echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved in: runs/train/yolov12_rf_detection"
echo ""
echo "Next steps:"
echo "  1. Check training metrics: tensorboard --logdir runs/train"
echo "  2. Validate model: python val_yolov12.py"
echo "  3. Run inference: python detect_yolov12.py"
