"""
Spectrogram Generation
Based on Georgia Tech preprocessing algorithm
"""

import numpy as np
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, Optional


def generate_spectrogram(iq_samples: np.ndarray,
                        fs: float = 20e6,
                        nperseg: int = 256,
                        noverlap: int = 128,
                        normalize: bool = True,
                        apply_colormap: bool = True,
                        colormap_name: str = 'viridis',
                        center_freq: float = 2.437e9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 256×256 spectrogram matching Verdis preprocessing

    Args:
        iq_samples: Complex IQ data array
        fs: Sample rate (20 MHz)
        nperseg: FFT window size (256 for 256×256 output)
        noverlap: Overlap samples (128 = 50% overlap)
        normalize: Apply power normalization to move mean to 0.5
        apply_colormap: Convert to RGB with colormap
        colormap_name: Matplotlib colormap name ('viridis', 'plasma', 'turbo', 'jet')
        center_freq: Center frequency in Hz (default 2.437 GHz)

    Returns:
        Tuple of (spectrogram_image, time_array, frequency_array)
        - spectrogram_image: 256×256 RGB numpy array (uint8)
        - time_array: Time axis values
        - frequency_array: Frequency axis values in Hz
    """
    # Compute STFT spectrogram
    f, t, Sxx = signal.spectrogram(
        iq_samples,
        fs=fs,
        window='hamming',
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        mode='magnitude'
    )

    # Shift zero frequency to center
    Sxx_shifted = fftshift(Sxx, axes=0)

    # --- Georgia Tech Normalization Procedure ---
    if normalize:
        Sxx_norm = georgia_tech_normalize(Sxx_shifted)
    else:
        Sxx_norm = Sxx_shifted / np.max(Sxx_shifted)

    # Convert to 8-bit
    Sxx_uint8 = (Sxx_norm * 255).astype(np.uint8)

    # Apply colormap
    if apply_colormap:
        colormap = cm.get_cmap(colormap_name)
        spectrogram_rgb = colormap(Sxx_uint8 / 255.0)[:, :, :3]  # Drop alpha channel
        spectrogram_rgb = (spectrogram_rgb * 255).astype(np.uint8)
    else:
        spectrogram_rgb = np.stack([Sxx_uint8] * 3, axis=-1)

    # Convert frequency to RF
    f_rf = fftshift(f) + center_freq

    return spectrogram_rgb, t, f_rf


def georgia_tech_normalize(Sxx: np.ndarray, target_mean: float = 0.5,
                          iterations: int = 3) -> np.ndarray:
    """
    Apply Georgia Tech paper normalization procedure
    Normalizes spectrogram to [0, 1] with mean ≈ 0.5

    Args:
        Sxx: Spectrogram magnitude array
        target_mean: Target mean value (default 0.5)
        iterations: Number of normalization iterations (default 3)

    Returns:
        Normalized spectrogram in [0, 1] range
    """
    # Normalize to [0, 1] with mean ≈ 0.5
    Sxx_min = np.min(Sxx)
    Sxx_adjusted = Sxx - Sxx_min

    # Apply power function to adjust mean to 0.5
    # n = log(μ/M) / log(μ_target)
    current_mean = np.mean(Sxx_adjusted)
    max_val = np.max(Sxx_adjusted)

    if current_mean > 0 and max_val > 0:
        n = np.log(current_mean / max_val) / np.log(target_mean)

        # Iterate normalization (paper uses 3 iterations)
        for _ in range(iterations):
            Sxx_norm = (Sxx_adjusted / max_val) ** n
            Sxx_norm = Sxx_norm / np.max(Sxx_norm)  # Renormalize

            current_mean = np.mean(Sxx_norm)
            n = np.log(current_mean) / np.log(target_mean)
    else:
        Sxx_norm = Sxx_adjusted / (max_val + 1e-10)

    return Sxx_norm


def resize_spectrogram(spectrogram: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Resize spectrogram to target dimensions

    Args:
        spectrogram: Input spectrogram array
        target_size: Target (height, width)

    Returns:
        Resized spectrogram
    """
    import cv2
    return cv2.resize(spectrogram, (target_size[1], target_size[0]),
                     interpolation=cv2.INTER_LINEAR)
