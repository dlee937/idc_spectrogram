# 🚀 Quick Start Guide: Process Your Real Data TODAY

**Goal**: Generate first spectrograms from your 12GB of real RF data in the next 2 hours

---

## ⚡ Fast Track: 30 Minutes to First Results

### Step 1: Copy Your Real Data (2 minutes)

```bash
# Create destination directory
mkdir -p data/raw

# Copy both captures with headers
cp idc-backup-10_2017/epoch_23.sc16* data/raw/
cp test_backups/test4_2412.* data/raw/

# Verify (should see ~12GB)
ls -lh data/raw/
```

**Expected Output**:
```
-rw-r--r--  1 user  staff   6.0G  Oct 16  2017 epoch_23.sc16
-rw-r--r--  1 user  staff   1.2K  Oct 16  2017 epoch_23.sc16.hdr
-rw-r--r--  1 user  staff   6.0G  Oct 16  2017 test4_2412.sc16
-rw-r--r--  1 user  staff   1.2K  Oct 16  2017 test4_2412.sc16.hdr
-rw-r--r--  1 user  staff   400M  Oct 16  2017 test4_2412.fc32
```

---

### Step 2: Install Dependencies (5 minutes)

```bash
# Install Python packages
pip install numpy scipy matplotlib opencv-python tqdm pyyaml ultralytics Pillow

# Or use requirements.txt
pip install -r requirements.txt
```

---

### Step 3: Parse USRP Metadata (3 minutes)

**Create quick test script** (`test_header.py`):

```python
def parse_usrp_header(hdr_filepath):
    """Parse USRP .hdr file for metadata"""
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

# Test on epoch_23
print("=" * 50)
print("EPOCH_23 METADATA:")
print("=" * 50)
hdr = parse_usrp_header('data/raw/epoch_23.sc16.hdr')
for key, value in hdr.items():
    if 'freq' in key.lower():
        print(f"{key}: {value/1e9:.3f} GHz")
    elif 'rate' in key.lower():
        print(f"{key}: {value/1e6:.2f} MS/s")
    else:
        print(f"{key}: {value}")

print("\n" + "=" * 50)
print("TEST4_2412 METADATA:")
print("=" * 50)
hdr2 = parse_usrp_header('data/raw/test4_2412.sc16.hdr')
for key, value in hdr2.items():
    if 'freq' in key.lower():
        print(f"{key}: {value/1e9:.3f} GHz")
    elif 'rate' in key.lower():
        print(f"{key}: {value/1e6:.2f} MS/s")
    else:
        print(f"{key}: {value}")
```

**Run it**:
```bash
python test_header.py
```

**Expected Output** (example):
```
==================================================
EPOCH_23 METADATA:
==================================================
rx_rate: 20.00 MS/s
rx_freq: 2.437 GHz
rx_time: 1508176234.123456
file_size: 6442450944

==================================================
TEST4_2412 METADATA:
==================================================
rx_rate: 20.00 MS/s
rx_freq: 2.412 GHz
rx_time: 1508180000.000000
file_size: 6442450944
```

---

### Step 4: Generate First Spectrogram (20 minutes)

**Create quick spectrogram script** (`generate_first_spec.py`):

```python
import numpy as np
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_iq_data_quick(filepath, num_samples=1_000_000):
    """Load first N samples from .sc16 file"""
    raw = np.fromfile(filepath, dtype=np.int16, count=num_samples*2)
    i_channel = raw[::2].astype(np.float32)
    q_channel = raw[1::2].astype(np.float32)
    samples = (i_channel + 1j * q_channel) / 32768.0
    samples = samples - np.mean(samples)  # Remove DC
    return samples

def generate_spectrogram_quick(iq_samples, fs=20e6):
    """Generate spectrogram with Georgia Tech normalization"""
    # STFT
    f, t, Sxx = signal.spectrogram(
        iq_samples,
        fs=fs,
        window='hamming',
        nperseg=256,
        noverlap=128,
        return_onesided=False,
        mode='magnitude'
    )
    
    # Center DC
    Sxx_shifted = fftshift(Sxx, axes=0)
    
    # Georgia Tech normalization to mean=0.5
    Sxx_min = np.min(Sxx_shifted)
    Sxx_adjusted = Sxx_shifted - Sxx_min
    current_mean = np.mean(Sxx_adjusted)
    max_val = np.max(Sxx_adjusted)
    
    if current_mean > 0 and max_val > 0:
        target_mean = 0.5
        n = np.log(current_mean / max_val) / np.log(target_mean)
        for _ in range(3):  # 3 iterations
            Sxx_norm = (Sxx_adjusted / max_val) ** n
            Sxx_norm = Sxx_norm / np.max(Sxx_norm)
            current_mean = np.mean(Sxx_norm)
            n = np.log(current_mean) / np.log(target_mean)
    else:
        Sxx_norm = Sxx_adjusted / (max_val + 1e-10)
    
    # Convert to 8-bit RGB
    Sxx_uint8 = (Sxx_norm * 255).astype(np.uint8)
    colormap = cm.get_cmap('viridis')
    spec_rgb = colormap(Sxx_uint8 / 255.0)[:, :, :3]
    spec_rgb = (spec_rgb * 255).astype(np.uint8)
    
    return spec_rgb, t, fftshift(f)

# Process epoch_23
print("Loading IQ data from epoch_23.sc16...")
iq_data = load_iq_data_quick('data/raw/epoch_23.sc16', num_samples=1_000_000)
print(f"Loaded {len(iq_data):,} samples")

print("Generating spectrogram...")
spec_img, t, f = generate_spectrogram_quick(iq_data, fs=20e6)
print(f"Spectrogram shape: {spec_img.shape}")

# Visualize
plt.figure(figsize=(14, 8))

# Main spectrogram
plt.subplot(2, 1, 1)
plt.imshow(spec_img, aspect='auto', 
           extent=[t[0]*1e6, t[-1]*1e6, (f[0]+2.437e9)/1e9, (f[-1]+2.437e9)/1e9])
plt.xlabel('Time (μs)')
plt.ylabel('Frequency (GHz)')
plt.title('Epoch 23 Spectrogram - First 1M Samples (50ms @ 20 MS/s)')
plt.colorbar(label='Intensity')
plt.grid(True, alpha=0.3)

# Zoom to Bluetooth band (2.402 - 2.480 GHz)
plt.subplot(2, 1, 2)
plt.imshow(spec_img, aspect='auto',
           extent=[t[0]*1e6, t[-1]*1e6, (f[0]+2.437e9)/1e9, (f[-1]+2.437e9)/1e9])
plt.xlabel('Time (μs)')
plt.ylabel('Frequency (GHz)')
plt.title('Zoomed: Bluetooth Band (2.402 - 2.480 GHz)')
plt.ylim(2.402, 2.480)
plt.colorbar(label='Intensity')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/first_real_spectrogram.png', dpi=150, bbox_inches='tight')
print("\n✅ Spectrogram saved to: results/first_real_spectrogram.png")

# Save raw spectrogram image
import cv2
cv2.imwrite('data/spectrograms/epoch_23_seg0000.png', 
            cv2.cvtColor(spec_img, cv2.COLOR_RGB2BGR))
print("✅ Raw spectrogram saved to: data/spectrograms/epoch_23_seg0000.png")

plt.show()
```

**Run it**:
```bash
# Create results directory
mkdir -p results
mkdir -p data/spectrograms

# Generate spectrogram
python generate_first_spec.py
```

**What You'll See**:
- Top plot: Full 20 MHz bandwidth around 2.437 GHz
- Bottom plot: Zoomed to Bluetooth band (2.402-2.480 GHz)
- Should see narrow vertical streaks (Bluetooth hops)
- Should see wider bands (WiFi Channel 6)

---

## 🔍 What to Look For in Your First Spectrogram

### Bluetooth Signals:
- **Appearance**: Narrow vertical streaks (1-2 MHz wide)
- **Duration**: Short bursts (~625 μs)
- **Pattern**: Random frequency hopping across band
- **Color**: Bright yellow/white in viridis colormap

### WiFi Signals:
- **Appearance**: Wide horizontal bands (20 MHz)
- **Duration**: Longer continuous transmissions
- **Pattern**: Fixed at 2.437 GHz (Channel 6)
- **Color**: Green/yellow, less intense than Bluetooth

### Noise Floor:
- **Appearance**: Dark blue/purple background
- **Pattern**: Even across frequency and time
- **Intensity**: Low (normalized to ~0.1-0.3)

---

## 📊 Quick Validation Checklist

After running `generate_first_spec.py`:

- [ ] **File created**: `results/first_real_spectrogram.png` exists
- [ ] **Visual inspection**: Can see vertical streaks (Bluetooth)
- [ ] **Frequency range**: Y-axis shows 2.427 - 2.447 GHz
- [ ] **Time range**: X-axis shows 0 - 50,000 μs (50 ms)
- [ ] **Colormap**: Viridis (blue to yellow)
- [ ] **No errors**: Script ran without exceptions

---

## 🎯 Next: Batch Process 1000 Spectrograms (1 hour)

**Create batch processing script** (`batch_process.py`):

```python
import numpy as np
from scipy import signal
from scipy.fft import fftshift
import matplotlib.cm as cm
import cv2
from pathlib import Path
from tqdm import tqdm

def load_iq_chunked(filepath, chunk_size=8200):
    """Load IQ file in chunks (410 μs segments at 20 MS/s)"""
    with open(filepath, 'rb') as f:
        while True:
            raw = np.fromfile(f, dtype=np.int16, count=chunk_size*2)
            if raw.size == 0:
                break
            i_channel = raw[::2].astype(np.float32)
            q_channel = raw[1::2].astype(np.float32)
            samples = (i_channel + 1j * q_channel) / 32768.0
            samples = samples - np.mean(samples)
            yield samples

def generate_spectrogram_fast(iq_samples, fs=20e6):
    """Fast spectrogram generation (no plotting)"""
    f, t, Sxx = signal.spectrogram(
        iq_samples, fs=fs, window='hamming',
        nperseg=256, noverlap=128,
        return_onesided=False, mode='magnitude'
    )
    Sxx_shifted = fftshift(Sxx, axes=0)
    
    # Quick normalization (1 iteration instead of 3)
    Sxx_min = np.min(Sxx_shifted)
    Sxx_adjusted = Sxx_shifted - Sxx_min
    max_val = np.max(Sxx_adjusted)
    Sxx_norm = Sxx_adjusted / (max_val + 1e-10)
    
    Sxx_uint8 = (Sxx_norm * 255).astype(np.uint8)
    colormap = cm.get_cmap('viridis')
    spec_rgb = colormap(Sxx_uint8 / 255.0)[:, :, :3]
    spec_rgb = (spec_rgb * 255).astype(np.uint8)
    
    return spec_rgb

# Process first 1000 segments from epoch_23
output_dir = Path('data/spectrograms/epoch23/')
output_dir.mkdir(parents=True, exist_ok=True)

print("Processing epoch_23.sc16...")
print("Generating 1000 spectrograms (410 μs each)...")

seg_idx = 0
for chunk in tqdm(load_iq_chunked('data/raw/epoch_23.sc16', chunk_size=8200), 
                  total=1000, desc="Processing"):
    if seg_idx >= 1000:
        break
    
    spec_img = generate_spectrogram_fast(chunk)
    output_file = output_dir / f"epoch_23_seg{seg_idx:04d}.png"
    cv2.imwrite(str(output_file), cv2.cvtColor(spec_img, cv2.COLOR_RGB2BGR))
    
    seg_idx += 1

print(f"\n✅ Generated {seg_idx} spectrograms in {output_dir}")
print(f"Total size: {sum(f.stat().st_size for f in output_dir.glob('*.png')) / 1024 / 1024:.1f} MB")
```

**Run it**:
```bash
python batch_process.py
```

**Expected**:
- Processing time: ~10-15 minutes for 1000 spectrograms
- Output: 1000 PNG files (~100-200 KB each)
- Total size: ~100-200 MB

---

## 🤖 Optional: Run Existing Notebook (15 minutes)

If you want to see your **existing implementation** in action:

```bash
# Start Jupyter
jupyter notebook

# Open rf_detection_modular.ipynb
# Update file paths to point to data/raw/epoch_23.sc16
# Run all cells
```

**What to check**:
- Does temporal overlap detection work on your spectrograms?
- Does Bluetooth cutoff detection identify edge signals?
- What's the detection rate?

---

## 📋 Today's Success Checklist

By end of today, you should have:

- [x] ✅ Copied 12GB of real data to `data/raw/`
- [x] ✅ Parsed USRP metadata from .hdr files
- [x] ✅ Generated first spectrogram from epoch_23
- [x] ✅ Visually validated Bluetooth signals present
- [x] ✅ Batch processed 1000 spectrograms
- [ ] 🎯 (Optional) Tested existing notebook on new spectrograms
- [ ] 🎯 (Optional) Compared epoch_23 vs test4_2412

---

## 🚀 Tomorrow's Goals

Once you have spectrograms:

### Morning (2-3 hours):
- [ ] Run sliding window analysis (notebook 03)
- [ ] Detect Bluetooth cutoffs
- [ ] Visualize top 20 detections

### Afternoon (3-4 hours):
- [ ] Download Roboflow dataset (if available)
- [ ] Prepare YOLO training data
- [ ] Start training baseline YOLOv8 model

### Evening (1-2 hours):
- [ ] Monitor training progress
- [ ] Run inference on test spectrograms
- [ ] Document initial results

---

## 🎉 Why This Approach Works

### Fast Results:
- **30 minutes** to first spectrogram
- **1 hour** to 1000 spectrograms
- **2 hours** to sliding window analysis

### Real Data:
- Not synthetic or simulated
- Actual Georgia Tech testbed captures
- EED (Extreme Emitter Density) environment

### Proven Algorithms:
- Your existing notebook already works
- Just need to run it on new spectrograms
- No debugging required

---

## 🐛 Common Issues & Fixes

### Issue: "FileNotFoundError: epoch_23.sc16.hdr"
**Cause**: Header file not copied  
**Fix**: 
```bash
cp idc-backup-10_2017/epoch_23.sc16.hdr data/raw/
```

### Issue: "MemoryError: Cannot allocate array"
**Cause**: Trying to load entire 6GB file at once  
**Fix**: Use chunked loading (already in `batch_process.py`)

### Issue: "No Bluetooth signals visible in spectrogram"
**Possible causes**:
1. Wrong center frequency (should be 2.437 GHz or 2.412 GHz)
2. Wrong sample rate (should be 20 MS/s)
3. Data corruption

**Debug**:
```python
# Check metadata
hdr = parse_usrp_header('data/raw/epoch_23.sc16.hdr')
print(f"Center freq: {hdr['rx_freq']/1e9:.3f} GHz")  # Should be 2.437
print(f"Sample rate: {hdr['rx_rate']/1e6:.1f} MS/s")  # Should be 20.0
```

### Issue: "Spectrogram looks wrong (all one color)"
**Cause**: Normalization failure  
**Fix**: Check that IQ data has variation
```python
iq_data = load_iq_data_quick('data/raw/epoch_23.sc16', 10000)
print(f"Mean: {np.mean(np.abs(iq_data)):.3f}")  # Should be ~0.1-0.3
print(f"Std: {np.std(np.abs(iq_data)):.3f}")    # Should be ~0.2-0.4
print(f"Max: {np.max(np.abs(iq_data)):.3f}")    # Should be ~0.8-1.0
```

---

## 📖 Additional Resources

### If You Get Stuck:
1. **Check existing notebook**: `rf_detection_modular.ipynb` has working code
2. **Review documentation**: README.md, UPDATED_IMPLEMENTATION_PLAN.md
3. **Inspect .hdr files**: They contain valuable metadata

### For Deep Dives:
- **Georgia Tech Paper**: See project references
- **USRP Documentation**: https://files.ettus.com/manual/
- **GNU Radio IQ Format**: https://wiki.gnuradio.org/index.php/File_Formats

---

## 🎯 Key Takeaways

1. **You have real data** - This is huge! No need for synthetic samples.
2. **Start simple** - Generate one spectrogram, then batch process.
3. **Use existing code** - Your notebook already works, leverage it.
4. **Iterate quickly** - Test on 1000 spectrograms before processing all 12GB.
5. **Document findings** - Take notes on what works and what doesn't.

---

**Time Investment Today**: 2 hours  
**Payoff**: Real spectrograms from actual RF captures  
**Next Milestone**: 1000 spectrograms ready for YOLO training

**Let's get started! 🚀**

---

**Created**: 2025-10-16  
**Status**: Ready to execute  
**First Command**: `cp idc-backup-10_2017/epoch_23.sc16* data/raw/`
