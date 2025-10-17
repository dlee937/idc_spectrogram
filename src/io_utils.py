"""
IQ Data Loading Utilities
Handles loading and preprocessing of .sc16 and .sc32 binary files
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple


def load_iq_data(filepath: Union[str, Path], dtype: str = 'sc16') -> np.ndarray:
    """
    Load IQ data from .sc16 or .sc32 files

    Args:
        filepath: Path to binary IQ file
        dtype: 'sc16' or 'sc32'

    Returns:
        Complex numpy array (I + jQ)
    """
    if dtype == 'sc16':
        raw = np.fromfile(filepath, dtype=np.int16)
    elif dtype == 'sc32':
        raw = np.fromfile(filepath, dtype=np.int32)
    else:
        raise ValueError("dtype must be 'sc16' or 'sc32'")

    # Deinterleave I and Q channels
    i_channel = raw[::2].astype(np.float32)
    q_channel = raw[1::2].astype(np.float32)

    # Normalize to [-1, 1] range
    if dtype == 'sc16':
        samples = (i_channel + 1j * q_channel) / 32768.0
    else:  # sc32
        samples = (i_channel + 1j * q_channel) / 2147483648.0

    # Remove DC offset
    samples = samples - np.mean(samples)

    return samples


def load_iq_chunked(filepath: Union[str, Path], chunk_size: int = 10_000_000,
                   dtype: str = 'sc16'):
    """
    Load IQ data in chunks for memory efficiency

    Args:
        filepath: Path to binary IQ file
        chunk_size: Number of samples per chunk
        dtype: 'sc16' or 'sc32'

    Yields:
        Complex numpy arrays (chunks of I + jQ)
    """
    dt = np.int16 if dtype == 'sc16' else np.int32
    scale = 32768.0 if dtype == 'sc16' else 2147483648.0

    with open(filepath, 'rb') as f:
        while True:
            chunk = np.fromfile(f, dtype=dt, count=chunk_size * 2)
            if chunk.size == 0:
                break

            i_channel = chunk[::2].astype(np.float32)
            q_channel = chunk[1::2].astype(np.float32)
            samples = (i_channel + 1j * q_channel) / scale
            samples = samples - np.mean(samples)

            yield samples


def get_file_info(filepath: Union[str, Path], dtype: str = 'sc16') -> dict:
    """
    Get information about an IQ file

    Args:
        filepath: Path to binary IQ file
        dtype: 'sc16' or 'sc32'

    Returns:
        Dictionary with file information
    """
    path = Path(filepath)
    file_size = path.stat().st_size

    bytes_per_sample = 4 if dtype == 'sc16' else 8  # 2 bytes/channel * 2 channels
    num_samples = file_size // bytes_per_sample

    # Assuming 20 MHz sample rate
    duration_seconds = num_samples / 20e6

    return {
        'filename': path.name,
        'file_size_bytes': file_size,
        'file_size_mb': file_size / (1024 * 1024),
        'num_samples': num_samples,
        'duration_seconds': duration_seconds,
        'duration_ms': duration_seconds * 1000,
        'dtype': dtype
    }
