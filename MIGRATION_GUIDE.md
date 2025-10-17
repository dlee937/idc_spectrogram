# Migration Guide: Integrating Existing Work with New Structure

## Overview
This guide explains how to integrate your existing notebooks, data, and scripts into the new modular structure.

---

## 🗂️ Current Assets

### Existing Data Files
```
idc-backup-10_2017/
├── epoch_23.sc16 (6 GB)          → Move to: data/raw/
├── epoch_23.sc16.hdr             → Move to: data/raw/
└── epoch_23.sc16.sh              → Keep as reference

test_backups/
├── test4_2412.sc16 (6 GB)        → Move to: data/raw/
├── test4_2412.sc16.hdr           → Move to: data/raw/
├── test4_2412.fc32 (400 MB)      → Move to: data/raw/
└── test4_2412.fc32.hdr           → Move to: data/raw/
```

### Existing Code Assets
```
rf_detection_modular.ipynb        → Extract useful functions to src/
rf_spectrogram_roboflow.ipynb     → Reference for Roboflow integration
what,py                           → Add to src/ as channel_mask.py
old_scripts/*.m                   → Keep as reference documentation
```

---

## 📋 Step-by-Step Migration

### Step 1: Move Data Files (IMPORTANT: Copy, don't move originals yet)
```bash
# Create backup first!
cp idc-backup-10_2017/epoch_23.sc16* data/raw/
cp test_backups/test4_2412.* data/raw/

# Verify copies
ls -lh data/raw/
```

### Step 2: Extract Useful Functions from rf_detection_modular.ipynb

**Functions to Extract**:

#### A. Add to `src/io_utils.py`:
```python
def parse_usrp_header(header_file: str) -> dict:
    """Parse USRP .hdr file to extract metadata"""
    # Extract rx_rate, rx_time, rx_freq, etc.
    pass
```

#### B. Add to `src/preprocessing.py`:
```python
def detect_temporal_overlap(spectrogram_dir: Path, n_workers: int = 8) -> float:
    """
    Detect temporal overlap using correlation analysis
    (from rf_detection_modular.ipynb cell 11)
    """
    pass
```

#### C. Add to `src/slicing.py`:
```python
def check_bluetooth_cutoff(images: List[Path], annotations_dict: Dict,
                          time_windows: List[TimeWindow]) -> Dict:
    """
    Check for signal cutoffs at boundaries
    (from rf_detection_modular.ipynb cell 13)
    """
    pass
```

#### D. Add to `src/visualization.py`:
```python
def visualize_cutoff_detection(detection: Dict, output_path: Optional[Path] = None):
    """
    Visualize cutoff detections side-by-side
    (from rf_detection_modular.ipynb cell 21)
    """
    pass
```

### Step 3: Rename and Integrate what,py
```bash
# Rename and move
mv "what,py" src/channel_mask.py

# Then edit src/channel_mask.py to add as function:
def generate_bluetooth_channel_mask(height=512, width=256,
                                   freq_range=(2427, 2447),
                                   gradient=True) -> np.ndarray:
    """Generate visual mask for Bluetooth channels"""
    # (existing code from what,py)
    pass
```

### Step 4: Create Roboflow Integration Notebook
**New file**: `notebooks/06_roboflow_integration.ipynb`

Content should include:
- Roboflow API setup (use existing API key)
- Dataset download
- COCO format conversion to YOLO
- Integration with existing annotations

### Step 5: Update config.yaml with Real Data Parameters

Based on the `.hdr` files, update:
```yaml
data:
  # From epoch_23.sc16.hdr
  sample_rate: 20e6  # Check actual rx_rate from header
  center_frequency: 2.437e9  # Check actual rx_freq

  # File mappings
  raw_files:
    - epoch_23.sc16
    - test4_2412.sc16
    - test4_2412.fc32
```

---

## 🔧 Enhanced src/ Modules

### Create: `src/roboflow_utils.py`
Extract Roboflow-specific code from rf_detection_modular.ipynb:
```python
"""
Roboflow Dataset Integration
"""

from roboflow import Roboflow
from pathlib import Path
from typing import Dict, Tuple
import json

def download_roboflow_dataset(api_key: str,
                              workspace: str,
                              project: str,
                              version: int) -> Tuple[Path, Dict]:
    """Download dataset and return path with COCO annotations"""
    pass

def convert_coco_to_yolo(coco_data: Dict, output_dir: Path):
    """Convert COCO format annotations to YOLO format"""
    pass
```

### Create: `src/time_analysis.py`
Extract temporal analysis from rf_detection_modular.ipynb:
```python
"""
Temporal Analysis and Time Window Calculations
"""

from dataclasses import dataclass
from typing import List

@dataclass
class TimeWindow:
    """Represents a time window for a spectrogram frame"""
    frame_idx: int
    start_us: float
    end_us: float
    duration_us: float

    def overlaps_with(self, other: 'TimeWindow') -> bool:
        pass

    def overlap_percent(self, other: 'TimeWindow') -> float:
        pass

def calculate_time_windows(n_images: int, overlap_pct: float,
                          base_duration_us: float = 410) -> List[TimeWindow]:
    """Calculate absolute time windows for each spectrogram frame"""
    pass
```

### Create: `src/annotation_utils.py`
```python
"""
Annotation Handling (COCO, YOLO, Roboflow)
"""

from dataclasses import dataclass
from typing import Dict

@dataclass
class BoundingBox:
    """Represents a bounding box annotation"""
    x: float
    y: float
    width: float
    height: float
    category_id: int
    category_name: str
    confidence: float = 1.0

    def to_yolo_format(self, img_width: int, img_height: int) -> str:
        """Convert to YOLO format"""
        pass

    def to_coco_format(self) -> Dict:
        """Convert to COCO format"""
        pass
```

---

## 🚀 Recommended Workflow

### Phase 1: Data Migration & Validation (1-2 hours)
1. Copy data files to `data/raw/`
2. Parse `.hdr` files to extract metadata
3. Run `notebooks/01_data_preprocessing.ipynb` to load and validate
4. Verify sample rate, center frequency match expectations

### Phase 2: Code Integration (2-3 hours)
1. Create new modules (`roboflow_utils.py`, `time_analysis.py`, `annotation_utils.py`)
2. Extract functions from `rf_detection_modular.ipynb`
3. Integrate `what,py` as `channel_mask.py`
4. Update `__init__.py` with new imports

### Phase 3: Enhanced Notebooks (2-3 hours)
1. Update `notebooks/02_spectrogram_generation.ipynb` to use real .sc16 files
2. Create `notebooks/06_roboflow_integration.ipynb`
3. Add temporal overlap detection to `notebooks/03_sliding_window_analysis.ipynb`
4. Test full pipeline end-to-end

### Phase 4: MATLAB Reference Documentation (1 hour)
1. Document legacy MATLAB pipeline in README
2. Note differences between old (TDoA) and new (YOLO detection) approaches
3. Keep MATLAB scripts as reference for signal processing equations

---

## 📝 Updated Project Structure

After migration:
```
idc/
├── data/
│   ├── raw/
│   │   ├── epoch_23.sc16          # ← Real data!
│   │   ├── epoch_23.sc16.hdr
│   │   ├── test4_2412.sc16        # ← Real data!
│   │   ├── test4_2412.sc16.hdr
│   │   └── test4_2412.fc32
│   ├── spectrograms/              # Generated from .sc16
│   ├── sliced/
│   └── annotations/
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_spectrogram_generation.ipynb
│   ├── 03_sliding_window_analysis.ipynb
│   ├── 04_signal_detection.ipynb
│   ├── 05_yolo_dataset_prep.ipynb
│   └── 06_roboflow_integration.ipynb  # ← NEW
│
├── src/
│   ├── __init__.py
│   ├── io_utils.py                    # + parse_usrp_header()
│   ├── spectrogram.py
│   ├── preprocessing.py               # + detect_temporal_overlap()
│   ├── slicing.py                     # + check_bluetooth_cutoff()
│   ├── visualization.py               # + visualize_cutoff_detection()
│   ├── channel_mask.py                # ← NEW (from what,py)
│   ├── roboflow_utils.py              # ← NEW
│   ├── time_analysis.py               # ← NEW
│   └── annotation_utils.py            # ← NEW
│
├── reference/                         # ← NEW
│   ├── matlab_legacy/
│   │   └── (moved from old_scripts/)
│   ├── notebooks_original/
│   │   ├── rf_detection_modular.ipynb
│   │   └── rf_spectrogram_roboflow.ipynb
│   └── LEGACY_NOTES.md
│
└── backups/                           # ← Rename existing
    ├── idc-backup-10_2017/
    └── test_backups/
```

---

## ⚠️ Important Notes

### Data Safety
- **NEVER delete original files** in `idc-backup-10_2017/` or `test_backups/`
- Always work with copies in `data/raw/`
- Keep `.hdr` files with corresponding `.sc16` files

### Roboflow API Key
- Existing key: `V74EfwetgJtOmApcRI4g`
- Store in environment variable for security
- Add to `.gitignore` if using version control

### MATLAB Scripts
- Keep as reference - contain valuable signal processing knowledge
- Document their purpose in LEGACY_NOTES.md
- May need them for comparison/validation

### File Format Notes
- `.sc16` = signed complex int16 (your src/io_utils.py already handles this!)
- `.fc32` = float complex 32-bit (need to add support)
- `.hdr` = USRP metadata in binary format (need parser)

---

## 🎯 Quick Start Commands

```bash
# 1. Backup originals
cp -r idc-backup-10_2017 backups/
cp -r test_backups backups/

# 2. Copy data to working directory
cp idc-backup-10_2017/epoch_23.sc16* data/raw/
cp test_backups/test4_2412.* data/raw/

# 3. Rename what,py
mv "what,py" src/channel_mask.py

# 4. Create reference directory
mkdir -p reference/{matlab_legacy,notebooks_original}
cp old_scripts/* reference/matlab_legacy/
cp rf_*.ipynb reference/notebooks_original/

# 5. Install dependencies
pip install -r requirements.txt

# 6. Start with notebook 01
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

---

## ✅ Validation Checklist

- [ ] Data files copied to `data/raw/`
- [ ] `.hdr` files parsed successfully
- [ ] Sample rate and frequency match expectations
- [ ] Spectrograms generate correctly from .sc16 files
- [ ] Temporal overlap detection works on generated spectrograms
- [ ] Bluetooth cutoff detection identifies boundary signals
- [ ] Roboflow integration downloads dataset
- [ ] YOLO format conversion works
- [ ] End-to-end pipeline runs without errors

---

**Next Step**: Run the Quick Start Commands above, then proceed with Phase 1!
