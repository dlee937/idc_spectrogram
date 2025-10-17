"""
Channel and Temporal Slicing Utilities
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
from scipy.signal import find_peaks


def frequency_channel_slice(spectrogram: np.ndarray,
                            num_channels: int = 4,
                            center_freq: float = 2.437e9,
                            bandwidth: float = 20e6) -> List[dict]:
    """
    Slice spectrogram into frequency sub-bands

    Args:
        spectrogram: 256×256 RGB spectrogram
        num_channels: Number of frequency divisions (default 4 = 5 MHz each)
        center_freq: Center frequency in Hz
        bandwidth: Total bandwidth in Hz

    Returns:
        List of channel slices with metadata
    """
    height = spectrogram.shape[0]
    channel_height = height // num_channels
    channel_bandwidth = bandwidth / num_channels

    slices = []
    for i in range(num_channels):
        start_row = i * channel_height
        end_row = (i + 1) * channel_height if i < num_channels - 1 else height

        channel_slice = spectrogram[start_row:end_row, :, :]

        # Calculate frequency range for this channel
        freq_start = center_freq - bandwidth/2 + i * channel_bandwidth
        freq_end = freq_start + channel_bandwidth

        slices.append({
            'slice': channel_slice,
            'channel_id': i,
            'freq_range_hz': (freq_start, freq_end),
            'freq_range_ghz': (freq_start/1e9, freq_end/1e9),
            'row_range': (start_row, end_row)
        })

    return slices


def sliding_window_bluetooth_detection(spectrogram_dir: Union[str, Path],
                                       window_size: int = 5,
                                       stride: int = 1) -> List[dict]:
    """
    Analyze sequential spectrograms to detect cut-off Bluetooth signals

    Args:
        spectrogram_dir: Directory with sequential spectrograms
        window_size: Number of consecutive frames to analyze
        stride: Step size between windows

    Returns:
        List of cutoff detections
    """
    spec_files = sorted(Path(spectrogram_dir).glob("*.png"))

    # Parse frame numbers from filenames
    def get_frame_number(filepath):
        stem = filepath.stem
        seg_num = int(stem.split('_seg')[-1])
        return seg_num

    spec_files_sorted = sorted(spec_files, key=get_frame_number)

    cutoff_detections = []

    for i in range(0, len(spec_files_sorted) - window_size + 1, stride):
        window_files = spec_files_sorted[i:i + window_size]

        # Load spectrograms in window
        window_specs = []
        for spec_file in window_files:
            img = cv2.imread(str(spec_file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            window_specs.append(img_rgb)

        # Check for Bluetooth signals at boundaries
        cutoff_info = check_bluetooth_cutoff(window_specs)

        if cutoff_info['has_cutoff']:
            cutoff_detections.append({
                'start_frame': get_frame_number(window_files[0]),
                'end_frame': get_frame_number(window_files[-1]),
                'cutoff_info': cutoff_info
            })

    return cutoff_detections


def check_bluetooth_cutoff(spectrogram_window: List[np.ndarray]) -> dict:
    """
    Detect if Bluetooth signal is cut off at boundaries

    Bluetooth characteristics:
    - Narrow bandwidth (~5-10 pixels in 256×256 spectrogram)
    - Vertical streak patterns
    - High intensity in narrow frequency band

    Args:
        spectrogram_window: List of consecutive spectrogram images

    Returns:
        dict with cutoff detection results
    """
    # Convert to grayscale for easier analysis
    window_gray = [cv2.cvtColor(spec, cv2.COLOR_RGB2GRAY) for spec in spectrogram_window]

    # Analyze first and last frames for edge signals
    first_frame = window_gray[0]
    last_frame = window_gray[-1]

    # Check right edge of first frame
    right_edge = first_frame[:, -10:]  # Last 10 columns
    right_edge_intensity = np.mean(right_edge, axis=1)

    # Check left edge of last frame
    left_edge = last_frame[:, :10]  # First 10 columns
    left_edge_intensity = np.mean(left_edge, axis=1)

    # Detect narrow high-intensity regions (Bluetooth signature)
    def find_narrow_peaks(intensity_profile, min_height=150, max_width=15):
        peaks, properties = find_peaks(
            intensity_profile,
            height=min_height,
            width=(2, max_width)
        )
        return peaks, properties

    right_peaks, right_props = find_narrow_peaks(right_edge_intensity)
    left_peaks, left_props = find_narrow_peaks(left_edge_intensity)

    # Check for potential cutoff
    has_right_cutoff = len(right_peaks) > 0
    has_left_cutoff = len(left_peaks) > 0

    # Check frequency continuity across frames
    frequency_match = False
    if has_right_cutoff and has_left_cutoff:
        # Compare peak locations (should be at similar frequencies)
        freq_diff = np.min(np.abs(right_peaks[:, None] - left_peaks[None, :]))
        frequency_match = freq_diff < 20  # Within 20 pixels (frequency tolerance)

    return {
        'has_cutoff': has_right_cutoff and has_left_cutoff and frequency_match,
        'right_edge_peaks': right_peaks.tolist() if hasattr(right_peaks, 'tolist') else [],
        'left_edge_peaks': left_peaks.tolist() if hasattr(left_peaks, 'tolist') else [],
        'frequency_match': frequency_match,
        'confidence': calculate_confidence(right_props, left_props)
    }


def calculate_confidence(right_props: dict, left_props: dict) -> float:
    """Calculate confidence score for Bluetooth detection"""
    # Based on peak prominence, width, and intensity
    if len(right_props.get('prominences', [])) == 0 or len(left_props.get('prominences', [])) == 0:
        return 0.0

    right_score = np.mean(right_props['prominences']) * np.mean(right_props['peak_heights'])
    left_score = np.mean(left_props['prominences']) * np.mean(left_props['peak_heights'])

    # Normalize to [0, 1]
    confidence = np.tanh((right_score + left_score) / 10000)
    return float(confidence)
