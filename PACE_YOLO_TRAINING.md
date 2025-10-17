# YOLO Training on PACE ICE

Complete guide for training YOLO models on PACE ICE supercomputer.

## Prerequisites

1. Access to PACE ICE cluster
2. GPU allocation/quota
3. Spectrograms generated from test4 data
4. Labeled dataset (or auto-labeling capability)

## Quick Start

### 1. Prepare YOLO Dataset

```bash
cd ~/idc_spectrogram

# Copy spectrograms to YOLO directories and create train/val split
chmod +x prepare_yolo_pace.sh
./prepare_yolo_pace.sh
```

This will:
- Create train/val split (80/20)
- Copy spectrograms to appropriate directories
- Create placeholder label files

### 2. Create Labels

You have two options:

**Option A: Auto-labeling (recommended for initial testing)**
```bash
module load anaconda3
source activate rf_signal
python auto_label_spectrograms.py
```

**Option B: Manual annotation**
- Download images to local machine
- Annotate using labelImg, Roboflow, or CVAT
- Upload labels back to PACE

### 3. Setup YOLO Environment

```bash
chmod +x setup_yolo_pace.sh
./setup_yolo_pace.sh
```

This will:
- Install PyTorch with CUDA support
- Install Ultralytics YOLO
- Update data.yaml with correct paths
- Verify GPU availability

### 4. Submit Training Job

```bash
# Check GPU availability
sinfo -p inferno

# Submit job
sbatch train_yolo_pace.sh

# Check job status
squeue -u $USER

# Monitor training output
tail -f logs/yolo_train_*.out
```

## Job Configuration

The SLURM job (`train_yolo_pace.sh`) is configured for:
- **GPU**: 1x GPU (adjust with `--gres=gpu:N`)
- **CPU**: 8 cores
- **Memory**: 64GB
- **Time**: 24 hours
- **Queue**: inferno (GPU queue)

### Adjusting Resources

Edit `train_yolo_pace.sh`:

```bash
#SBATCH --gres=gpu:2          # Use 2 GPUs
#SBATCH --mem=128GB           # Use more memory
#SBATCH -t 48:00:00           # Extend time to 48 hours
```

## Training Configuration

### Default Settings (in `train_yolov12.py`)

- **Model**: YOLOv12s (small)
- **Epochs**: 100
- **Batch size**: 16
- **Image size**: 640x640
- **Optimizer**: AdamW
- **Learning rate**: 0.01

### Customizing Training

Edit `train_yolov12.py` to adjust:

```python
results = model.train(
    epochs=200,           # More epochs
    batch=32,             # Larger batch (if GPU memory allows)
    imgsz=1024,          # Larger images (better for small objects)
    device=[0, 1],       # Multi-GPU training
)
```

## Monitoring Training

### Real-time Monitoring

```bash
# Watch job queue
watch -n 10 squeue -u $USER

# Follow training log
tail -f logs/yolo_train_*.out

# Check GPU usage
ssh <node-name>
nvidia-smi -l 5
```

### TensorBoard (for detailed metrics)

```bash
# On PACE (in a separate session)
module load anaconda3
source activate rf_signal
tensorboard --logdir runs/train --port 6006

# Then port forward from local machine:
# ssh -L 6006:localhost:6006 dlee937@login-ice.pace.gatech.edu
# Access at: http://localhost:6006
```

## After Training

### 1. Check Results

```bash
ls -lh runs/train/yolov12_rf_detection/

# Important files:
# - weights/best.pt       # Best model weights
# - weights/last.pt       # Last checkpoint
# - results.png           # Training curves
# - confusion_matrix.png  # Confusion matrix
```

### 2. Validate Model

```bash
python val_yolov12.py
```

### 3. Run Inference

```bash
python detect_yolov12.py
```

### 4. Download Results to Local Machine

```bash
# From local machine
scp -r dlee937@login-ice.pace.gatech.edu:~/idc_spectrogram/runs/train/yolov12_rf_detection ./results/
```

## Troubleshooting

### Job Doesn't Start

```bash
# Check job status
squeue -u $USER

# Check job details
scontrol show job <job_id>

# Common issues:
# - No GPU available: wait or reduce GPU request
# - Quota exceeded: check storage quota with 'quota'
```

### Out of Memory Error

Reduce batch size in `train_yolov12.py`:
```python
batch=8,  # or batch=4 for very limited GPU memory
```

### CUDA Not Available

```bash
# Check CUDA module
module list

# Load correct CUDA version
module load cuda/11.8

# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Slow Training

- Use multiple GPUs: `#SBATCH --gres=gpu:2`
- Increase batch size if GPU memory allows
- Use mixed precision training (fp16)
- Check if using GPU queue: `#SBATCH -q inferno`

## Dataset Specifications

### Expected Structure

```
data/yolo/
├── data.yaml              # Dataset configuration
├── train/
│   ├── images/           # Training images (.png)
│   └── labels/           # Training labels (.txt)
└── val/
    ├── images/           # Validation images (.png)
    └── labels/           # Validation labels (.txt)
```

### Label Format (YOLO)

Each `.txt` file contains one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values normalized to [0, 1]:
- `class_id`: 0=bluetooth, 1=wifi, 2=noise
- `x_center`, `y_center`: center of bounding box
- `width`, `height`: dimensions of bounding box

Example:
```
0 0.5 0.3 0.1 0.2
1 0.7 0.6 0.15 0.25
```

## Performance Tips

1. **Start with small dataset**: Test with 100-200 images first
2. **Use pretrained weights**: If available for YOLOv12
3. **Monitor validation loss**: Stop if overfitting
4. **Adjust learning rate**: If loss plateaus early
5. **Use data augmentation**: Already configured in training script

## Resource Limits

Check your PACE allocation:
```bash
pace-quota
sinfo -p inferno  # Check GPU availability
```

Typical GPU training times:
- 100 epochs, 800 images, 1 GPU: ~2-4 hours
- 200 epochs, 2000 images, 1 GPU: ~8-12 hours

## Questions?

- PACE documentation: https://pace.gatech.edu/
- PACE support: pace-support@gatech.edu
- YOLO documentation: https://docs.ultralytics.com/
