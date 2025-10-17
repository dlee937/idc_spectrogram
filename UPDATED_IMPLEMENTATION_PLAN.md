# RF Signal Detection Project - UPDATED Implementation Plan

**Date**: 2025-10-16  
**Status**: 🎉 **REAL DATA AVAILABLE - INTEGRATION MODE**

---

## 🎯 Project Discovery: What We Actually Have

### ✅ Real RF Data (Ready to Use!)

**Location**: `idc-backup-10_2017/` and `test_backups/`

```
idc-backup-10_2017/
├── epoch_23.sc16 (6 GB)           # Main IQ capture from 2017
├── epoch_23.sc16.hdr              # USRP metadata (rx_rate, rx_freq, rx_time)
└── [Additional captures...]

test_backups/
├── test4_2412.sc16 (6 GB)         # Capture at 2.412 GHz (WiFi Channel 1)
├── test4_2412.fc32 (400 MB)       # Float complex format
└── *.hdr files                     # USRP metadata headers
```

**Total Data**: ~12+ GB of real RF captures from Georgia Tech testbed!

---

### ✅ Working Python Implementation

**Location**: `rf_detection_modular.ipynb` (33 KB)

**Already Implemented**:
- ✅ Temporal overlap detection (correlation-based)
- ✅ Bluetooth cutoff detection (sliding window)
- ✅ COCO format annotation handling
- ✅ Roboflow API integration
- ✅ Time window calculations
- ✅ Signal processing pipeline

**Roboflow Integration**:
- API Key: `V74EfwetgJtOmApcRI4g`
- Dataset presumably already uploaded
- May have existing annotations

---

### 📚 Legacy MATLAB Pipeline (2017)

**Location**: `old_scripts/`

**Purpose**: Original TDoA triangulation system

```
old_scripts/
├── read_complex_binary.m          # GNU Radio IQ reader
├── time_diff.m                    # TDoA calculation
├── ChunkPnseq10202017.m          # Data chunking
└── matlab_proc.sh                 # Multi-node orchestration
```

**System Architecture**:
- 3-node RF sensor setup (10.7.4.x IPs)
- Time Difference of Arrival (TDoA) localization
- Bluetooth/WiFi triangulation in stadium

---

### 🎨 Bluetooth Channel Visualizer

**Location**: `what,py` (should be `what.py`)

**Purpose**: Generate visual channel masks for 20 Bluetooth channels
- Maps 2.427-2.447 GHz to 512×256 pixels
- Creates gradient-based separation masks
- Two versions: with gradient & thin black lines

---

## 🔄 REVISED Strategy: Integration vs. Fresh Start

You now have **TWO PATHS FORWARD**:

### Path A: Fast Track (Using Existing Work) ⚡
**Time**: 2-4 hours to first results  
**Pros**: Leverage proven code, faster results  
**Best for**: Quick experiments, validating approach

### Path B: Modular Migration (New Structure) 🏗️
**Time**: 1-2 days to full integration  
**Pros**: Clean codebase, maintainable, extensible  
**Best for**: Production deployment, collaboration

**RECOMMENDATION**: **Hybrid Approach** - Use existing notebook for quick wins, gradually migrate to modular structure

---

## 📋 UPDATED Task List

### ✅ PHASE 0: Infrastructure Setup (COMPLETE)
- [x] Created modular directory structure
- [x] Implemented 5 Python modules (io_utils, spectrogram, preprocessing, slicing, visualization)
- [x] Created 5 Jupyter notebooks
- [x] Configured config.yaml
- [x] Created documentation (README, TODO, AUDIT)

---

### 🎯 PHASE 1: Data Integration (PRIORITY: CRITICAL)

#### Task 1.1: Consolidate Real Data ⚡
**Estimated Time**: 10 minutes  
**Status**: TODO

**Actions**:
```bash
# Copy all IQ files to new structure
cp idc-backup-10_2017/epoch_23.sc16* data/raw/
cp test_backups/test4_2412.* data/raw/

# Verify files
ls -lh data/raw/
```

**Deliverables**:
- [ ] All .sc16/.sc32 files in `data/raw/`
- [ ] All .hdr files preserved
- [ ] File inventory documented

---

#### Task 1.2: Parse USRP Header Files ⚡
**Estimated Time**: 30 minutes  
**Status**: TODO

**Purpose**: Extract metadata (sample rate, center frequency, timestamp)

**Implementation** (add to `src/io_utils.py`):
```python
def parse_usrp_header(hdr_filepath):
    """
    Parse USRP .hdr file to extract metadata
    
    Returns:
        dict: {
            'rx_rate': float (samples/sec),
            'rx_freq': float (Hz),
            'rx_time': float (Unix timestamp),
            'file_size': int (bytes)
        }
    """
    metadata = {}
    with open(hdr_filepath, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=')
                try:
                    metadata[key.strip()] = float(value.strip())
                except:
                    metadata[key.strip()] = value.strip()
    return metadata

# Example usage
hdr = parse_usrp_header('data/raw/epoch_23.sc16.hdr')
print(f"Sample Rate: {hdr['rx_rate']/1e6:.2f} MS/s")
print(f"Center Freq: {hdr['rx_freq']/1e9:.3f} GHz")
```

**Deliverables**:
- [ ] Header parser function added to `io_utils.py`
- [ ] Metadata extracted from both captures
- [ ] Verification: center frequency matches expected (2.437 GHz or 2.412 GHz)

---

#### Task 1.3: Generate First Spectrograms from Real Data ⚡
**Estimated Time**: 1 hour  
**Status**: TODO

**Actions**:
```python
# In notebooks/02_spectrogram_generation.ipynb

from src.io_utils import load_iq_data, parse_usrp_header
from src.spectrogram import generate_spectrogram
import matplotlib.pyplot as plt

# Load real data
hdr = parse_usrp_header('data/raw/epoch_23.sc16.hdr')
fs = hdr['rx_rate']
center_freq = hdr['rx_freq']

# Load first 1 million samples (~50ms at 20 MS/s)
iq_data = load_iq_data('data/raw/epoch_23.sc16', dtype='sc16')[:1_000_000]

# Generate spectrogram
spec_img, t, f = generate_spectrogram(
    iq_data, 
    fs=fs,
    nperseg=256,
    noverlap=128,
    colormap_name='viridis'
)

# Visualize
plt.figure(figsize=(12, 6))
plt.imshow(spec_img, aspect='auto', extent=[t[0]*1e6, t[-1]*1e6, f[0]/1e9, f[-1]/1e9])
plt.xlabel('Time (μs)')
plt.ylabel('Frequency (GHz)')
plt.title(f'Epoch 23 Spectrogram - Center: {center_freq/1e9:.3f} GHz')
plt.colorbar()
plt.savefig('results/first_real_spectrogram.png', dpi=150)
```

**Deliverables**:
- [ ] First real spectrogram generated
- [ ] Visual inspection confirms signal presence
- [ ] Image saved to `results/` directory

**Expected Results**:
- Should see Bluetooth hops (narrow vertical streaks)
- WiFi signals (wider horizontal bands)
- Background noise floor

---

### 🔍 PHASE 2: Code Migration & Analysis (PRIORITY: HIGH)

#### Task 2.1: Extract Functions from Existing Notebook ⚡
**Estimated Time**: 2-3 hours  
**Status**: TODO

**Purpose**: Migrate proven algorithms from `rf_detection_modular.ipynb` into modular `src/` files

**Migration Map**:

| Notebook Function | Destination Module | Priority |
|-------------------|-------------------|----------|
| Temporal overlap detection | `src/slicing.py` | HIGH |
| Bluetooth cutoff detection | `src/slicing.py` | HIGH |
| COCO annotation parser | `src/preprocessing.py` | MEDIUM |
| Time window calculator | `src/io_utils.py` | MEDIUM |
| Correlation-based matching | `src/slicing.py` | HIGH |

**Example Migration** (Temporal Overlap Detection):
```python
# Add to src/slicing.py

def detect_temporal_overlap(spec_window, threshold=0.7):
    """
    Detect if signals overlap temporally across sequential spectrograms
    Uses correlation-based matching (from rf_detection_modular.ipynb)
    
    Args:
        spec_window: List of consecutive spectrogram images
        threshold: Correlation threshold (0-1)
    
    Returns:
        dict: {
            'has_overlap': bool,
            'overlap_regions': list of (start_frame, end_frame, correlation)
        }
    """
    overlaps = []
    
    for i in range(len(spec_window) - 1):
        frame1 = spec_window[i]
        frame2 = spec_window[i + 1]
        
        # Extract right edge of frame1 and left edge of frame2
        right_edge = frame1[:, -20:]  # Last 20 columns
        left_edge = frame2[:, :20]    # First 20 columns
        
        # Compute correlation
        corr = np.corrcoef(right_edge.flatten(), left_edge.flatten())[0, 1]
        
        if corr > threshold:
            overlaps.append({
                'start_frame': i,
                'end_frame': i + 1,
                'correlation': corr
            })
    
    return {
        'has_overlap': len(overlaps) > 0,
        'overlap_regions': overlaps
    }
```

**Deliverables**:
- [ ] All key functions extracted and documented
- [ ] Functions integrated into appropriate modules
- [ ] Original notebook cells referenced in docstrings
- [ ] Unit tests created for migrated functions

---

#### Task 2.2: Validate Existing Bluetooth Cutoff Detection ⚡
**Estimated Time**: 1 hour  
**Status**: TODO

**Purpose**: Run existing cutoff detection on real data to validate

**Actions**:
1. Open `rf_detection_modular.ipynb`
2. Update file paths to point to `data/raw/epoch_23.sc16`
3. Run existing cutoff detection cells
4. Compare results with new implementation in `src/slicing.py`

**Test Cases**:
```python
# Test on first 1000 spectrograms from epoch_23
spec_files = sorted(Path('data/spectrograms/').glob('epoch_23_seg*.png'))[:1000]

# Run EXISTING detection (from notebook)
detections_old = existing_bluetooth_cutoff_detection(spec_files)

# Run NEW detection (from src/slicing.py)
detections_new = sliding_window_bluetooth_detection(
    'data/spectrograms/',
    window_size=5,
    stride=1
)

# Compare results
print(f"Old method: {len(detections_old)} cutoffs detected")
print(f"New method: {len(detections_new)} cutoffs detected")
print(f"Agreement: {calculate_agreement(detections_old, detections_new):.2%}")
```

**Success Criteria**:
- [ ] Both methods detect similar number of cutoffs (±10%)
- [ ] Visual inspection confirms true positives
- [ ] False positive rate <15%

---

#### Task 2.3: Integrate Roboflow Dataset ⚡
**Estimated Time**: 30 minutes  
**Status**: TODO

**Purpose**: Access existing annotations from Roboflow

**Actions**:
```python
# In notebooks/04_signal_detection.ipynb

from roboflow import Roboflow

rf = Roboflow(api_key="V74EfwetgJtOmApcRI4g")
project = rf.workspace().project("YOUR_PROJECT_NAME")  # Need to identify
dataset = project.version(1).download("yolov8")

# Copy annotations to local structure
import shutil
shutil.copytree(dataset.location + '/train/labels', 'data/annotations/train/')
shutil.copytree(dataset.location + '/valid/labels', 'data/annotations/val/')
```

**Implementation Questions**:
- [ ] What is your Roboflow project name?
- [ ] How many images are annotated in Roboflow?
- [ ] What classes are labeled? (bluetooth, wifi, zigbee, drone?)

**Deliverables**:
- [ ] Roboflow dataset downloaded locally
- [ ] Annotations copied to `data/annotations/`
- [ ] Dataset statistics documented (# images, # classes, # boxes)

---

### 📊 PHASE 3: Advanced Analysis (PRIORITY: MEDIUM)

#### Task 3.1: Channel/Temporal Slicing with Real Data ⚡
**Estimated Time**: 2 hours  
**Status**: TODO

**Purpose**: Apply sliding window analysis to detect cut-off Bluetooth signals

**Actions**:
```python
# In notebooks/03_sliding_window_analysis.ipynb

# Run on ALL epoch_23 spectrograms
detections = sliding_window_bluetooth_detection(
    spectrogram_dir='data/spectrograms/',
    window_size=5,
    stride=1
)

print(f"Total cutoffs detected: {len(detections)}")
print(f"Average confidence: {np.mean([d['cutoff_info']['confidence'] for d in detections]):.2f}")

# Visualize top 20 highest-confidence detections
from src.visualization import visualize_cutoff_detections

visualize_cutoff_detections(
    'data/spectrograms/',
    sorted(detections, key=lambda x: x['cutoff_info']['confidence'], reverse=True)[:20],
    'results/cutoff_detections/'
)
```

**Analysis Questions**:
- [ ] What percentage of spectrograms have cutoffs?
- [ ] Are cutoffs evenly distributed across time?
- [ ] Do cutoff locations correlate with Bluetooth channels?

**Deliverables**:
- [ ] Cutoff detection results saved to `results/cutoffs.json`
- [ ] Top 20 visualizations in `results/cutoff_detections/`
- [ ] Statistical analysis report (CSV or markdown)

---

#### Task 3.2: Bluetooth Channel Mask Integration 🎨
**Estimated Time**: 1 hour  
**Status**: TODO

**Purpose**: Use `what.py` to generate channel masks for training

**Actions**:
1. Rename `what,py` → `what.py`
2. Move to `src/channel_masks.py`
3. Integrate with spectrogram generation

**Updated Function**:
```python
# In src/channel_masks.py (formerly what.py)

def generate_bluetooth_channel_masks(image_size=(256, 256), num_channels=20):
    """
    Generate visual masks for 20 Bluetooth channels
    Maps 2.427-2.447 GHz to pixel coordinates
    
    Args:
        image_size: (height, width) of output masks
        num_channels: Number of Bluetooth channels (default 20)
    
    Returns:
        list of masks: Each mask is (height, width) numpy array
    """
    masks = []
    freq_range = (2.427e9, 2.447e9)  # 20 MHz bandwidth
    channel_width = (freq_range[1] - freq_range[0]) / num_channels
    
    for ch_idx in range(num_channels):
        freq_start = freq_range[0] + ch_idx * channel_width
        freq_end = freq_start + channel_width
        
        # Map to pixel rows
        row_start = int((freq_start - freq_range[0]) / (freq_range[1] - freq_range[0]) * image_size[0])
        row_end = int((freq_end - freq_range[0]) / (freq_range[1] - freq_range[0]) * image_size[0])
        
        # Create mask
        mask = np.zeros(image_size, dtype=np.uint8)
        mask[row_start:row_end, :] = 255
        
        masks.append({
            'mask': mask,
            'channel': ch_idx,
            'freq_start': freq_start,
            'freq_end': freq_end
        })
    
    return masks

# Usage in training
masks = generate_bluetooth_channel_masks()
# Use masks for channel-specific detection or data augmentation
```

**Deliverables**:
- [ ] Channel mask generator integrated into `src/`
- [ ] 20 channel masks saved as images for visualization
- [ ] Documentation on how to use masks in YOLO training

---

#### Task 3.3: Compare Epoch_23 vs Test4_2412 ⚡
**Estimated Time**: 1 hour  
**Status**: TODO

**Purpose**: Analyze differences between two captures (different center frequencies)

**Comparison Table**:

| Parameter | epoch_23.sc16 | test4_2412.sc16 |
|-----------|---------------|-----------------|
| Center Frequency | 2.437 GHz (WiFi Ch 6) | 2.412 GHz (WiFi Ch 1) |
| File Size | 6 GB | 6 GB |
| Duration | TBD | TBD |
| Bluetooth Channels Visible | 27-45 (~18 ch) | 0-20 (~20 ch) |
| Expected Signals | BT + WiFi Ch 6 | BT + WiFi Ch 1 |

**Analysis**:
```python
# Generate spectrograms from both files
epoch23_specs = process_iq_files_to_spectrograms(
    'data/raw/', 
    'data/spectrograms/epoch23/',
    file_pattern='epoch_23.sc16'
)

test4_specs = process_iq_files_to_spectrograms(
    'data/raw/',
    'data/spectrograms/test4/',
    file_pattern='test4_2412.sc16'
)

# Compare signal characteristics
from src.visualization import plot_spectrogram_comparison

plot_spectrogram_comparison(
    epoch23_specs[0], 
    test4_specs[0],
    titles=['Epoch 23 (2.437 GHz)', 'Test4 (2.412 GHz)'],
    output_path='results/frequency_comparison.png'
)
```

**Deliverables**:
- [ ] Side-by-side spectrogram comparison
- [ ] Signal density analysis (which capture has more signals?)
- [ ] Recommendation: which dataset to use for training?

---

### 🤖 PHASE 4: YOLO Training (PRIORITY: HIGH)

#### Task 4.1: Prepare Dataset from Roboflow Annotations ⚡
**Estimated Time**: 1 hour  
**Status**: TODO

**Purpose**: Convert Roboflow dataset to local YOLO format

**Actions**:
```python
# In notebooks/05_yolo_dataset_prep.ipynb

# If Roboflow annotations already in YOLO format
from src.preprocessing import create_yolo_dataset_split

create_yolo_dataset_split(
    image_dir='data/spectrograms/',
    annotation_dir='data/annotations/',  # From Roboflow
    output_dir='data/yolo_dataset/',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Verify dataset
from pathlib import Path
print(f"Train images: {len(list(Path('data/yolo_dataset/train/images').glob('*.png')))}")
print(f"Val images: {len(list(Path('data/yolo_dataset/val/images').glob('*.png')))}")
print(f"Test images: {len(list(Path('data/yolo_dataset/test/images').glob('*.png')))}")
```

**Quality Checks**:
- [ ] All images have corresponding .txt annotation files
- [ ] Bounding boxes are within [0, 1] normalized range
- [ ] No empty annotation files (or remove those images)
- [ ] Class distribution is balanced (or note imbalance)

**Deliverables**:
- [ ] YOLO dataset in `data/yolo_dataset/` with train/val/test splits
- [ ] `data.yaml` configuration file
- [ ] Dataset statistics report

---

#### Task 4.2: Train Baseline YOLOv8 Model ⚡
**Estimated Time**: 2-6 hours (depending on GPU)  
**Status**: TODO

**Purpose**: Train first model on Roboflow-annotated data

**Training Script**:
```python
from ultralytics import YOLO

# Initialize YOLOv8 nano (fastest for testing)
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='data/yolo_dataset/data.yaml',
    epochs=100,
    imgsz=256,  # Match your spectrogram size
    batch=16,   # Adjust based on GPU memory
    device=0,   # GPU 0
    
    # Optimization
    optimizer='AdamW',
    lr0=0.001,
    
    # Augmentation (safe for spectrograms)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    mosaic=1.0,
    
    # NO rotation/flipping for spectrograms
    degrees=0.0,
    flipud=0.0,
    fliplr=0.0,
    
    # Checkpoints
    save_period=10,
    project='models/yolo_bluetooth',
    name='baseline_v1'
)

# Evaluate
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
```

**Monitoring**:
- Watch training curves in TensorBoard or Weights & Biases
- Check for overfitting (train loss << val loss)
- Monitor mAP convergence

**Success Criteria**:
- [ ] mAP@0.5 > 0.50 (good start)
- [ ] mAP@0.5 > 0.65 (production ready)
- [ ] Training completes without errors

**Deliverables**:
- [ ] Trained model checkpoint: `models/yolo_bluetooth/baseline_v1/weights/best.pt`
- [ ] Training curves: `models/yolo_bluetooth/baseline_v1/results.png`
- [ ] Validation metrics: `models/yolo_bluetooth/baseline_v1/confusion_matrix.png`

---

#### Task 4.3: Inference on New Spectrograms ⚡
**Estimated Time**: 30 minutes  
**Status**: TODO

**Purpose**: Run trained model on epoch_23 spectrograms

**Actions**:
```python
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# Load best model
model = YOLO('models/yolo_bluetooth/baseline_v1/weights/best.pt')

# Run inference on all epoch_23 spectrograms
spec_files = sorted(Path('data/spectrograms/epoch23/').glob('*.png'))

results_list = []
for spec_file in tqdm(spec_files[:100], desc="Running inference"):  # Test on first 100
    results = model.predict(
        source=spec_file,
        conf=0.25,  # Confidence threshold
        iou=0.45,   # IoU threshold for NMS
        save=True,  # Save annotated images
        project='results/inference',
        name='epoch23_detections'
    )
    
    # Extract detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            results_list.append({
                'image': spec_file.name,
                'class': int(box.cls[0]),
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].cpu().numpy().tolist()
            })

# Save results
import json
with open('results/epoch23_detections.json', 'w') as f:
    json.dump(results_list, f, indent=2)

print(f"Total detections: {len(results_list)}")
```

**Analysis**:
- [ ] How many Bluetooth signals detected?
- [ ] How many WiFi signals detected?
- [ ] Detection density over time (signals per spectrogram)

**Deliverables**:
- [ ] Annotated images in `results/inference/epoch23_detections/`
- [ ] Detection statistics in `results/epoch23_detections.json`
- [ ] Sample detections visualized for manual verification

---

### 🔬 PHASE 5: Advanced Features (PRIORITY: LOW)

#### Task 5.1: TDoA Integration (Legacy MATLAB → Python) 📚
**Estimated Time**: 1-2 days  
**Status**: FUTURE WORK

**Purpose**: Port MATLAB TDoA system to Python for triangulation

**Reference Files**:
- `old_scripts/time_diff.m` - Core TDoA algorithm
- `old_scripts/matlab_proc.sh` - Multi-node coordination

**Python Implementation** (future):
```python
# In src/tdoa.py (future module)

def calculate_tdoa(signal1, signal2, signal3, fs=20e6):
    """
    Calculate Time Difference of Arrival from 3 nodes
    Port of time_diff.m MATLAB function
    
    Args:
        signal1, signal2, signal3: IQ data from 3 RF sensor nodes
        fs: Sample rate (Hz)
    
    Returns:
        dict: {
            'tdoa_12': float (seconds),
            'tdoa_13': float (seconds),
            'tdoa_23': float (seconds),
            'position_estimate': (x, y)  # Triangulated position
        }
    """
    # Cross-correlation for time delay estimation
    from scipy.signal import correlate
    
    corr_12 = correlate(signal1, signal2, mode='full')
    corr_13 = correlate(signal1, signal3, mode='full')
    
    # Find peak lag
    lag_12 = (np.argmax(corr_12) - len(signal1)) / fs
    lag_13 = (np.argmax(corr_13) - len(signal1)) / fs
    
    # Hyperbolic positioning (requires node coordinates)
    # ... (complex geometry calculation)
    
    return {
        'tdoa_12': lag_12,
        'tdoa_13': lag_13,
        'position_estimate': None  # Implement later
    }
```

**Note**: This is LOW PRIORITY - focus on detection first, localization second.

---

#### Task 5.2: Multi-File Processing Pipeline 🔄
**Estimated Time**: 4 hours  
**Status**: FUTURE WORK

**Purpose**: Process ALL .sc16 files in batch

**Implementation**:
```python
# In src/batch_processing.py (future module)

def process_all_captures(input_dir='data/raw/', output_dir='data/spectrograms/'):
    """
    Process all .sc16/.sc32 files found in input directory
    Generates spectrograms and runs detection pipeline
    """
    from pathlib import Path
    from tqdm import tqdm
    
    iq_files = list(Path(input_dir).glob('*.sc16')) + list(Path(input_dir).glob('*.sc32'))
    
    for iq_file in tqdm(iq_files, desc="Processing captures"):
        # Parse header
        hdr_file = Path(str(iq_file) + '.hdr')
        if hdr_file.exists():
            hdr = parse_usrp_header(hdr_file)
            fs = hdr['rx_rate']
            center_freq = hdr['rx_freq']
        else:
            # Default values
            fs = 20e6
            center_freq = 2.437e9
        
        # Process
        output_subdir = output_dir / iq_file.stem
        process_iq_files_to_spectrograms(
            input_dir=str(iq_file.parent),
            output_dir=str(output_subdir),
            file_pattern=iq_file.name,
            fs=fs
        )
```

---

## 📊 UPDATED Implementation Timeline

### Week 1: Data Integration & Validation
**Days 1-2** (6-8 hours):
- [x] ~~Create modular structure~~ (COMPLETE)
- [ ] Copy real data to data/raw/
- [ ] Parse USRP headers
- [ ] Generate first spectrograms
- [ ] Validate existing notebook on real data

**Days 3-4** (6-8 hours):
- [ ] Extract functions from rf_detection_modular.ipynb
- [ ] Migrate to src/ modules
- [ ] Run sliding window analysis on epoch_23
- [ ] Document cutoff detections

**Days 5-7** (8-12 hours):
- [ ] Download Roboflow dataset
- [ ] Prepare YOLO dataset format
- [ ] Train baseline YOLOv8 model
- [ ] Run inference on epoch_23

### Week 2: Refinement & Advanced Analysis
**Days 8-10** (6-8 hours):
- [ ] Compare epoch_23 vs test4_2412
- [ ] Integrate Bluetooth channel masks
- [ ] Analyze detection patterns

**Days 11-14** (8-10 hours):
- [ ] Fine-tune YOLO hyperparameters
- [ ] Train larger model (YOLOv8s or YOLOv8m)
- [ ] Evaluate on test set
- [ ] Document final results

**Total Estimated Time**: 35-45 hours (much faster than starting from scratch!)

---

## 🎯 Success Metrics (Updated)

### Phase 1 Success (Data Integration):
- [ ] All 12GB of IQ data accessible in data/raw/
- [ ] First spectrograms generated and visually validated
- [ ] USRP metadata parsed correctly

### Phase 2 Success (Code Migration):
- [ ] Existing notebook functions integrated into src/ modules
- [ ] Bluetooth cutoff detection running on real data
- [ ] >80% of spectrograms processed without errors

### Phase 3 Success (Analysis):
- [ ] Cutoff detections documented with confidence scores
- [ ] Channel masks generated for all 20 Bluetooth channels
- [ ] Comparison report between two captures

### Phase 4 Success (YOLO Training):
- [ ] **mAP@0.5 > 0.60** on validation set
- [ ] **Bluetooth detection precision > 0.70**
- [ ] **Inference speed > 20 FPS** on GPU
- [ ] False positive rate < 15%

### Phase 5 Success (Production Ready):
- [ ] Test set mAP@0.5 > 0.65
- [ ] Per-class recall > 0.60
- [ ] Inference pipeline runs end-to-end

---

## 📋 Critical Questions (UPDATED)

### Data Questions:
1. ✅ **Dataset size?** → ~12GB confirmed (epoch_23 + test4_2412)
2. ✅ **File format?** → .sc16 (int16) and .fc32 (float32)
3. ✅ **Metadata available?** → Yes, .hdr files with rx_rate, rx_freq, rx_time
4. ❓ **Which capture to use for training?** → epoch_23 or test4_2412?

### Roboflow Questions:
5. ❓ **Roboflow project name?** → Need to identify
6. ❓ **How many images annotated?** → Unknown
7. ❓ **What classes labeled?** → Bluetooth, WiFi, both, or more?
8. ❓ **Annotation quality?** → Need to spot-check

### Hardware Questions:
9. ❓ **GPU available?** → (T4, V100, A100, local GPU?)
10. ❓ **Training time budget?** → Hours available for training?

### Technical Questions:
11. ❓ **Keep existing Roboflow integration or migrate to local?**
12. ❓ **Use correlation-based or peak-based cutoff detection?** → Test both
13. ❓ **Which colormap for spectrograms?** → (Existing notebook may have preference)

---

## 🚀 Immediate Next Steps (START HERE)

### 🔥 Action 1: Data Consolidation (10 minutes)
```bash
# Create data directory if not exists
mkdir -p data/raw

# Copy real RF data
cp idc-backup-10_2017/epoch_23.sc16* data/raw/
cp test_backups/test4_2412.* data/raw/

# Verify
ls -lh data/raw/
```

### 🔥 Action 2: Parse First Header (5 minutes)
```python
# Quick test in Python
def parse_usrp_header(hdr_filepath):
    metadata = {}
    with open(hdr_filepath, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=')
                try:
                    metadata[key.strip()] = float(value.strip())
                except:
                    metadata[key.strip()] = value.strip()
    return metadata

hdr = parse_usrp_header('data/raw/epoch_23.sc16.hdr')
print(hdr)
```

### 🔥 Action 3: Open Existing Notebook (5 minutes)
```bash
# Start Jupyter
jupyter notebook

# Open rf_detection_modular.ipynb
# Inspect what's already implemented
```

### 🔥 Action 4: Generate First Real Spectrogram (15 minutes)
```python
# In notebooks/02_spectrogram_generation.ipynb
# Follow Task 1.3 above to generate first spectrogram from epoch_23
```

---

## 📚 Key Files & Locations

### Your Existing Work:
```
.
├── rf_detection_modular.ipynb         # Main working notebook (33 KB)
├── rf_spectrogram_roboflow.ipynb      # Large notebook with outputs (8.7 MB)
├── what,py                             # Bluetooth channel mask generator
├── idc-backup-10_2017/
│   └── epoch_23.sc16*                 # 6 GB main capture
├── test_backups/
│   └── test4_2412.*                   # 6 GB + 400 MB captures
└── old_scripts/
    └── *.m                             # Legacy MATLAB TDoA system
```

### New Modular Structure:
```
idc/
├── data/
│   ├── raw/                    # ← Copy your .sc16 files HERE
│   ├── spectrograms/          # ← Generated images go here
│   ├── annotations/           # ← Roboflow annotations here
│   └── yolo_dataset/          # ← Final training dataset
│
├── src/                       # ← Migrate functions from notebook HERE
│   ├── io_utils.py           # Add: parse_usrp_header()
│   ├── slicing.py            # Add: detect_temporal_overlap()
│   └── ...
│
├── notebooks/                 # ← Use these for workflow
├── models/                    # ← Trained YOLO models
└── results/                   # ← Analysis outputs
```

---

## 🎉 What Makes This Project Unique

### Advantages You Have:
1. ✅ **Real data from Georgia Tech testbed** (Bobby Dodd Stadium, 2017)
2. ✅ **Working implementation** already exists (rf_detection_modular.ipynb)
3. ✅ **Roboflow integration** with existing annotations
4. ✅ **USRP metadata** for accurate RF parameter extraction
5. ✅ **Legacy TDoA system** for future triangulation work

### What This Means:
- **Faster results**: You can skip annotation if Roboflow has data
- **Real-world validation**: EED (Extreme Emitter Density) environment
- **Production relevance**: Stadium testbed = actual use case
- **Research continuity**: Build on 2017 MATLAB system

---

## 📖 Documentation References

### Your Documentation:
- **README.md** - Project overview, installation, usage
- **TODO.md** - Original detailed task breakdown (now superseded by this file)
- **AUDIT_REPORT.md** - Code quality audit (94/100 score)
- **MIGRATION_GUIDE.md** - How to integrate existing work

### External References:
- **Georgia Tech Paper**: "A Near Real-Time System for ISM Band Packet Detection and Localization Using Object Detection" (2023)
- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **Roboflow Docs**: https://docs.roboflow.com/

---

## 🔧 Troubleshooting (Expected Issues)

### Issue 1: Large File Loading
**Problem**: 6GB .sc16 file won't fit in RAM  
**Solution**: Use chunked loading
```python
from src.io_utils import load_iq_chunked

for chunk in load_iq_chunked('data/raw/epoch_23.sc16', chunk_size=10_000_000):
    # Process chunk
    spec_img, _, _ = generate_spectrogram(chunk)
    # Save spectrogram
```

### Issue 2: Roboflow Project Not Found
**Problem**: API key works but project not found  
**Solution**: List all projects
```python
from roboflow import Roboflow
rf = Roboflow(api_key="V74EfwetgJtOmApcRI4g")
workspace = rf.workspace()
print("Available projects:")
for project in workspace.projects:
    print(f"  - {project.name}")
```

### Issue 3: USRP Header Parsing Fails
**Problem**: .hdr file format different than expected  
**Solution**: Print raw header content first
```python
with open('data/raw/epoch_23.sc16.hdr', 'r') as f:
    print(f.read())
# Adjust parse_usrp_header() based on actual format
```

---

## 📝 Notes & Observations

### From Existing Notebook Analysis:
- Temporal overlap detection uses **correlation-based** matching
- Sliding window size may be **different** than our 5-frame default
- Roboflow dataset likely uses **COCO format**, not raw YOLO
- `what.py` generates **512×256** masks, but spectrograms are **256×256**

### Recommendations:
1. **Start with existing notebook** - it works, why reinvent?
2. **Gradually migrate** functions as you understand them
3. **Test on small subset** first (1000 spectrograms, not all 12GB)
4. **Document discoveries** as you go (what works, what doesn't)

---

## 🎯 Final Thoughts

You're in a **much better position** than starting from scratch:

✅ **Real data** - No need for synthetic samples  
✅ **Working code** - Proven algorithms in notebook  
✅ **Existing annotations** - Roboflow may have labeled data  
✅ **Clean architecture** - New modular structure ready  

**Recommended Approach**:
1. **Short-term (Week 1)**: Use existing notebook, get results fast
2. **Medium-term (Week 2)**: Migrate to modular structure gradually
3. **Long-term (Month 1)**: Full production pipeline with TDoA

**You're not building from zero - you're refactoring and improving!**

---

**Last Updated**: 2025-10-16 23:30 UTC  
**Status**: 🚀 Ready to Process Real Data  
**Next Action**: Copy .sc16 files to data/raw/ and run Task 1.3
