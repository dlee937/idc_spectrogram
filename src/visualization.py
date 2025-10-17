"""
Visualization Utilities
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Union, Optional


def visualize_cutoff_detections(spectrogram_dir: Union[str, Path],
                                detections: List[dict],
                                output_dir: Union[str, Path]) -> None:
    """
    Create visualizations showing detected cutoffs

    Args:
        spectrogram_dir: Directory with spectrograms
        detections: List of detection dictionaries
        output_dir: Where to save visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for detection in detections:
        start_frame = detection['start_frame']
        end_frame = detection['end_frame']

        # Load relevant spectrograms
        spec_files = sorted(Path(spectrogram_dir).glob(f"*_seg{start_frame:04d}.png"))

        if len(spec_files) == 0:
            continue

        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Load first and last frame
        first_img = cv2.imread(str(spec_files[0]))
        first_img_rgb = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)

        # Find last frame file
        last_files = sorted(Path(spectrogram_dir).glob(f"*_seg{end_frame:04d}.png"))
        if len(last_files) == 0:
            continue

        last_img = cv2.imread(str(last_files[0]))
        last_img_rgb = cv2.cvtColor(last_img, cv2.COLOR_BGR2RGB)

        axes[0].imshow(first_img_rgb)
        axes[0].set_title(f"Frame {start_frame} (right edge)")
        axes[0].axvline(x=246, color='red', linestyle='--', linewidth=2, label='Edge')

        axes[1].imshow(last_img_rgb)
        axes[1].set_title(f"Frame {end_frame} (left edge)")
        axes[1].axvline(x=10, color='red', linestyle='--', linewidth=2, label='Edge')

        # Mark detected peaks
        cutoff_info = detection['cutoff_info']
        for peak in cutoff_info.get('right_edge_peaks', []):
            axes[0].scatter(250, peak, color='red', s=100, marker='x')
        for peak in cutoff_info.get('left_edge_peaks', []):
            axes[1].scatter(5, peak, color='red', s=100, marker='x')

        confidence = cutoff_info.get('confidence', 0.0)
        plt.suptitle(f"Bluetooth Cutoff Detection (Confidence: {confidence:.2f})")
        plt.tight_layout()

        output_file = output_path / f"cutoff_detection_{start_frame}_{end_frame}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()


def plot_spectrogram_with_detections(spectrogram: np.ndarray,
                                     bboxes: List[dict],
                                     title: str = "Spectrogram with Detections",
                                     save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot spectrogram with bounding boxes overlaid

    Args:
        spectrogram: RGB spectrogram image
        bboxes: List of bounding box dictionaries with 'bbox' and 'class_id'
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(spectrogram)

    height, width = spectrogram.shape[:2]

    # Define class colors
    class_colors = {
        0: 'red',      # bluetooth
        1: 'green',    # wifi
        2: 'blue',     # zigbee
        3: 'yellow'    # drone
    }

    class_names = {
        0: 'bluetooth',
        1: 'wifi',
        2: 'zigbee',
        3: 'drone'
    }

    for bbox_dict in bboxes:
        class_id = bbox_dict['class_id']
        x_center, y_center, w, h = bbox_dict['bbox']

        # Convert from normalized YOLO format to pixel coordinates
        x_center_px = x_center * width
        y_center_px = y_center * height
        w_px = w * width
        h_px = h * height

        x1 = x_center_px - w_px / 2
        y1 = y_center_px - h_px / 2

        color = class_colors.get(class_id, 'white')

        # Draw rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), w_px, h_px,
                        linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Add label
        label = class_names.get(class_id, f'class_{class_id}')
        ax.text(x1, y1 - 5, label, color=color, fontsize=10,
               bbox=dict(facecolor='black', alpha=0.5))

    ax.set_title(title)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_iq_data(iq_samples: np.ndarray,
                fs: float = 20e6,
                max_samples: int = 10000,
                title: str = "IQ Data") -> None:
    """
    Plot I and Q channels

    Args:
        iq_samples: Complex IQ samples
        fs: Sample rate
        max_samples: Maximum number of samples to plot
        title: Plot title
    """
    samples_to_plot = min(len(iq_samples), max_samples)
    time = np.arange(samples_to_plot) / fs * 1e6  # Convert to microseconds

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # I channel
    axes[0].plot(time, iq_samples[:samples_to_plot].real)
    axes[0].set_ylabel('I (In-phase)')
    axes[0].set_title(f"{title} - I Channel")
    axes[0].grid(True)

    # Q channel
    axes[1].plot(time, iq_samples[:samples_to_plot].imag)
    axes[1].set_ylabel('Q (Quadrature)')
    axes[1].set_xlabel('Time (μs)')
    axes[1].set_title(f"{title} - Q Channel")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
