"""
RF Signal Detection Source Package
"""

from .io_utils import load_iq_data, load_iq_chunked, get_file_info
from .spectrogram import generate_spectrogram, georgia_tech_normalize, resize_spectrogram
from .preprocessing import (
    process_iq_files_to_spectrograms,
    temporal_slice_recording,
    filter_low_energy_segments
)
from .slicing import (
    frequency_channel_slice,
    sliding_window_bluetooth_detection,
    check_bluetooth_cutoff,
    calculate_confidence
)
from .visualization import (
    visualize_cutoff_detections,
    plot_spectrogram_with_detections,
    plot_iq_data
)

__all__ = [
    # IO utilities
    'load_iq_data',
    'load_iq_chunked',
    'get_file_info',
    # Spectrogram
    'generate_spectrogram',
    'georgia_tech_normalize',
    'resize_spectrogram',
    # Preprocessing
    'process_iq_files_to_spectrograms',
    'temporal_slice_recording',
    'filter_low_energy_segments',
    # Slicing
    'frequency_channel_slice',
    'sliding_window_bluetooth_detection',
    'check_bluetooth_cutoff',
    'calculate_confidence',
    # Visualization
    'visualize_cutoff_detections',
    'plot_spectrogram_with_detections',
    'plot_iq_data',
]
