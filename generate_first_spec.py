import numpy as np
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

def load_iq_data_quick(filepath, num_samples=10_000_000):
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
    colormap = plt.get_cmap('viridis')
    spec_rgb = colormap(Sxx_uint8 / 255.0)[:, :, :3]
    spec_rgb = (spec_rgb * 255).astype(np.uint8)

    return spec_rgb, t, fftshift(f)

# Process test4_2412 with longer time slices
print("Loading IQ data from test4_2412.sc16...")
iq_data = load_iq_data_quick('data/raw/test4_2412.sc16', num_samples=10_000_000)
print(f"Loaded {len(iq_data):,} samples")

print("Generating spectrogram...")
spec_img, t, f = generate_spectrogram_quick(iq_data, fs=20e6)
print(f"Spectrogram shape: {spec_img.shape}")

# Visualize
plt.figure(figsize=(14, 8))

# Main spectrogram
plt.subplot(2, 1, 1)
plt.imshow(spec_img, aspect='auto',
           extent=[t[0]*1e3, t[-1]*1e3, (f[0]+2.412e9)/1e9, (f[-1]+2.412e9)/1e9])
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (GHz)')
plt.title('Test4_2412 Spectrogram - 10M Samples (500ms @ 20 MS/s)')
plt.colorbar(label='Intensity')
plt.grid(True, alpha=0.3)

# Zoom to Bluetooth band (2.402 - 2.480 GHz)
plt.subplot(2, 1, 2)
plt.imshow(spec_img, aspect='auto',
           extent=[t[0]*1e3, t[-1]*1e3, (f[0]+2.412e9)/1e9, (f[-1]+2.412e9)/1e9])
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (GHz)')
plt.title('Zoomed: Bluetooth Band (2.402 - 2.480 GHz)')
plt.ylim(2.402, 2.480)
plt.colorbar(label='Intensity')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/first_real_spectrogram.png', dpi=150, bbox_inches='tight')
print("\nSpectrogram saved to: results/first_real_spectrogram.png")

# Save raw spectrogram image
cv2.imwrite('data/spectrograms/test4_2412_seg0000.png',
            cv2.cvtColor(spec_img, cv2.COLOR_RGB2BGR))
print("Raw spectrogram saved to: data/spectrograms/test4_2412_seg0000.png")

# plt.show()
