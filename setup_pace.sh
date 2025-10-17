#!/bin/bash
# Setup script for PACE ICE
# Run this after cloning the repository

echo "Setting up idc_spectrogram on PACE ICE..."

# Load modules
echo "Loading anaconda module..."
module load anaconda3

# Create conda environment
echo "Creating conda environment..."
conda create -n rf_signal python=3.9 -y

# Activate environment
echo "Activating environment..."
source activate rf_signal

# Install dependencies
echo "Installing Python packages..."
pip install numpy scipy matplotlib opencv-python tqdm pyyaml

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/spectrograms
mkdir -p data/spectrograms/test4_2412
mkdir -p results
mkdir -p logs

# Check if test4 data exists
echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Transfer your data files:"
echo "   rsync -avz --progress local/path/test4_2412.sc16* username@login-ice.pace.gatech.edu:~/idc_spectrogram/data/raw/"
echo ""
echo "2. Run batch processing:"
echo "   Interactive: conda activate rf_signal && python batch_process.py"
echo "   SLURM job: sbatch run_batch_slurm.sh"
echo ""
echo "3. Monitor SLURM job:"
echo "   squeue -u \$USER"
echo "   tail -f logs/batch_*.out"
