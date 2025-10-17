"""
Signal Preprocessing Utilities
Batch processing and temporal segmentation
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Union, List, Optional
from .io_utils import load_iq_data
from .spectrogram import generate_spectrogram


def process_iq_files_to_spectrograms(input_dir: Union[str, Path],
                                     output_dir: Union[str, Path],
                                     file_pattern: str = '*.sc16',
                                     segment_duration: float = 410e-6,
                                     fs: float = 20e6,
                                     dtype: str = 'sc16',
                                     colormap: str = 'viridis') -> None:
    """
    Process all IQ files in directory to spectrograms

    Args:
        input_dir: Directory with .sc16/.sc32 files
        output_dir: Where to save spectrogram images
        file_pattern: Glob pattern for files
        segment_duration: Duration per frame in seconds (410 μs from paper)
        fs: Sample rate in Hz (20 MHz)
        dtype: 'sc16' or 'sc32'
        colormap: Matplotlib colormap name
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    iq_files = sorted(input_path.glob(file_pattern))
    print(f"Found {len(iq_files)} IQ files")

    samples_per_segment = int(segment_duration * fs)  # ~8200 samples
    print(f"Samples per segment: {samples_per_segment}")

    total_spectrograms = 0

    for iq_file in tqdm(iq_files, desc="Processing IQ files"):
        # Load IQ data
        iq_data = load_iq_data(iq_file, dtype=dtype)

        # Segment into frames
        num_segments = len(iq_data) // samples_per_segment

        for seg_idx in range(num_segments):
            start_idx = seg_idx * samples_per_segment
            end_idx = start_idx + samples_per_segment
            segment = iq_data[start_idx:end_idx]

            # Generate spectrogram
            spec_img, _, _ = generate_spectrogram(segment, fs=fs, colormap_name=colormap)

            # Save as PNG
            output_filename = f"{iq_file.stem}_seg{seg_idx:04d}.png"
            output_filepath = output_path / output_filename

            cv2.imwrite(str(output_filepath), cv2.cvtColor(spec_img, cv2.COLOR_RGB2BGR))
            total_spectrograms += 1

    print(f"Generated {total_spectrograms} spectrograms saved to {output_dir}")


def temporal_slice_recording(iq_data: np.ndarray,
                             fs: float = 20e6,
                             slice_duration: float = 1.0,
                             overlap: float = 0.5) -> List[dict]:
    """
    Slice long IQ recording into temporal segments

    Args:
        iq_data: Full IQ recording
        fs: Sample rate
        slice_duration: Duration of each slice in seconds
        overlap: Overlap fraction (0.0 to 1.0)

    Returns:
        List of temporal slices with metadata
    """
    samples_per_slice = int(slice_duration * fs)
    hop_size = int(samples_per_slice * (1 - overlap))

    slices = []
    for start_idx in range(0, len(iq_data) - samples_per_slice + 1, hop_size):
        end_idx = start_idx + samples_per_slice

        time_slice = iq_data[start_idx:end_idx]

        slices.append({
            'data': time_slice,
            'start_time': start_idx / fs,
            'end_time': end_idx / fs,
            'sample_range': (start_idx, end_idx)
        })

    return slices


def filter_low_energy_segments(spectrogram: np.ndarray,
                               energy_threshold: float = 0.1) -> bool:
    """
    Check if spectrogram has sufficient energy (signals present)

    Args:
        spectrogram: Spectrogram array
        energy_threshold: Minimum mean intensity (0-1 range)

    Returns:
        True if segment has sufficient energy, False otherwise
    """
    # Convert to grayscale if RGB
    if len(spectrogram.shape) == 3:
        gray = cv2.cvtColor(spectrogram, cv2.COLOR_RGB2GRAY)
    else:
        gray = spectrogram

    # Normalize to [0, 1]
    gray_norm = gray.astype(np.float32) / 255.0

    # Calculate mean energy
    mean_energy = np.mean(gray_norm)

    return mean_energy > energy_threshold
