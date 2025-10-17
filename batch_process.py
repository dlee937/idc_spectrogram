import numpy as np
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm

def load_iq_chunked(filepath, chunk_size=200000):
    """Load IQ file in chunks (10ms segments at 20 MS/s)"""
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
    colormap = plt.get_cmap('viridis')
    spec_rgb = colormap(Sxx_uint8 / 255.0)[:, :, :3]
    spec_rgb = (spec_rgb * 255).astype(np.uint8)

    return spec_rgb

# Process segments from test4_2412 with longer time slices
output_dir = Path('data/spectrograms/test4_2412/')
output_dir.mkdir(parents=True, exist_ok=True)

print("Processing test4_2412.sc16...")
print("Generating spectrograms (10ms segments each)...")

seg_idx = 0
for chunk in tqdm(load_iq_chunked('data/raw/test4_2412.sc16', chunk_size=200000),
                  total=1000, desc="Processing"):
    if seg_idx >= 1000:
        break

    spec_img = generate_spectrogram_fast(chunk)
    output_file = output_dir / f"test4_2412_seg{seg_idx:04d}.png"
    cv2.imwrite(str(output_file), cv2.cvtColor(spec_img, cv2.COLOR_RGB2BGR))

    seg_idx += 1

print(f"\nGenerated {seg_idx} spectrograms in {output_dir}")
total_size = sum(f.stat().st_size for f in output_dir.glob('*.png')) / 1024 / 1024
print(f"Total size: {total_size:.1f} MB")
