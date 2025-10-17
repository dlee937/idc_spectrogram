# RF Signal Detection Project

A modular pipeline for processing RF spectrograms and detecting Bluetooth signals using YOLO object detection.

## Project Overview

This project processes Verdis spectrograms with channel/temporal slicing using sliding windows to detect Bluetooth signals and prepare datasets for YOLO training.

### Data Specifications

- **Center Frequency**: 2.437 GHz (Wi-Fi Channel 6)
- **Bandwidth**: 20 MHz (captures 2.427-2.447 GHz)
- **Time Duration**: 410 μs per spectrogram frame
- **Image Format**: 256×256 RGB (pre-processed with colormap)
- **Sample Rate**: 20 MS/s (20 million samples/second)
- **File Formats**: `.sc16`, `.sc32` (complex signed integers)

## Directory Structure

```
idc/
├── data/
│   ├── raw/                    # Original .sc16/.sc32 files
│   ├── spectrograms/          # Generated 256×256 RGB images
│   ├── sliced/                # Temporal/channel sliced spectrograms
│   └── annotations/           # YOLO format labels (.txt files)
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_spectrogram_generation.ipynb
│   ├── 03_sliding_window_analysis.ipynb
│   ├── 04_signal_detection.ipynb
│   └── 05_yolo_dataset_prep.ipynb
│
├── src/
│   ├── io_utils.py           # IQ data loading functions
│   ├── spectrogram.py        # Spectrogram generation
│   ├── preprocessing.py      # Signal preprocessing
│   ├── slicing.py           # Channel/temporal slicing
│   └── visualization.py     # Plotting utilities
│
├── models/                   # Trained model checkpoints
├── results/                  # Detection results, metrics
├── config/
│   └── config.yaml          # Hyperparameters, paths
│
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your `.sc16` or `.sc32` IQ data files in `data/raw/`

## Quick Start

Follow the notebooks in order:

### 1. Data Preprocessing (`01_data_preprocessing.ipynb`)
- Load IQ binary files
- Inspect file properties
- Visualize I/Q channels
- Validate signal quality

### 2. Spectrogram Generation (`02_spectrogram_generation.ipynb`)
- Convert IQ data to spectrograms
- Apply Georgia Tech normalization
- Batch process multiple files
- Save as PNG images

### 3. Sliding Window Analysis (`03_sliding_window_analysis.ipynb`)
- Implement temporal sliding windows
- Detect Bluetooth signal cutoffs
- Visualize detections
- Analyze frequency continuity

### 4. Signal Detection (`04_signal_detection.ipynb`)
- Semi-automated signal detection
- Generate YOLO bounding boxes
- Create preliminary annotations
- Manual verification workflow

### 5. YOLO Dataset Prep (`05_yolo_dataset_prep.ipynb`)
- Create train/val/test splits
- Organize YOLO directory structure
- Train YOLOv8 model
- Evaluate and run inference

## Configuration

All parameters are centralized in `config/config.yaml`:

- RF parameters (frequency, bandwidth, sample rate)
- Spectrogram settings (FFT size, overlap, colormap)
- Detection thresholds
- YOLO training hyperparameters
- Dataset split ratios
- Signal classes

## Signal Classes

The model detects 4 signal types:

0. **Bluetooth** - Narrow vertical streaks, frequency hopping
1. **Wi-Fi** - Wider bandwidth, continuous presence
2. **Zigbee** - Similar to Bluetooth but different hopping pattern
3. **Drone** - Custom RF signatures

## Usage Examples

### Load IQ Data

```python
from src.io_utils import load_iq_data, get_file_info

# Get file info
info = get_file_info('data/raw/capture.sc16', dtype='sc16')
print(f"Duration: {info['duration_ms']:.2f} ms")

# Load IQ samples
iq_data = load_iq_data('data/raw/capture.sc16', dtype='sc16')
```

### Generate Spectrogram

```python
from src.spectrogram import generate_spectrogram

spec_img, t, f = generate_spectrogram(
    iq_samples,
    fs=20e6,
    colormap_name='viridis'
)
```

### Detect Signals

```python
from src.slicing import sliding_window_bluetooth_detection

detections = sliding_window_bluetooth_detection(
    spectrogram_dir='data/spectrograms',
    window_size=5,
    stride=1
)
```

## Performance Targets

### Phase 1 (Data Processing)
- Successfully load all .sc16/.sc32 files
- Generate spectrograms matching Verdis quality
- Identify >80% of Bluetooth cutoff events

### Phase 2 (Dataset Creation)
- Annotate 500+ spectrograms
- Achieve <5% annotation error rate
- Create balanced train/val/test splits

### Phase 3 (Model Training)
- mAP@0.5 >0.60 on validation set
- Bluetooth detection precision >0.70
- Inference speed >20 FPS on GPU

## Annotation Tools

For manual annotation verification:

1. **LabelImg**: Simple, beginner-friendly
```bash
pip install labelImg
labelImg data/spectrograms data/annotations
```

2. **Roboflow**: Web-based, team collaboration
3. **CVAT**: Advanced, supports video annotation

## References

1. Georgia Tech Paper: "A Near Real-Time System for ISM Band Packet Detection and Localization Using Object Detection" (2023)
2. YOLOv8 Documentation: https://docs.ultralytics.com/
3. IQ Data Processing: PySDR - https://pysdr.org/
4. SciPy Signal Processing: https://docs.scipy.org/doc/scipy/reference/signal.html

## Troubleshooting

### Memory Issues with Large Files
Use chunked loading:
```python
from src.io_utils import load_iq_chunked

for chunk in load_iq_chunked('large_file.sc16', chunk_size=10_000_000):
    # Process chunk
    pass
```

### Imbalanced Dataset
Adjust class weights in `config/config.yaml`:
```yaml
classes:
  class_weights: [1.0, 0.5, 1.5, 2.0]
```

## License

This project is for research and educational purposes.

## Contact

For questions or issues, please refer to the project documentation or open an issue.

---

**Last Updated**: 2025-10-16
**Status**: Ready for implementation
