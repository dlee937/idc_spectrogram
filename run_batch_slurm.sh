#!/bin/bash
#SBATCH -J spectrogram_batch        # Job name
#SBATCH -N 1                        # Number of nodes
#SBATCH -n 8                        # Number of cores
#SBATCH --mem=32GB                  # Memory per node
#SBATCH -t 02:00:00                 # Time limit (2 hours)
#SBATCH -o logs/batch_%j.out        # Output file
#SBATCH -e logs/batch_%j.err        # Error file
#SBATCH --mail-type=END,FAIL        # Email notifications
#SBATCH --mail-user=your_email@gatech.edu

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"

# Load modules
module load anaconda3

# Activate environment
source activate rf_signal

# Go to project directory
cd $HOME/idc_spectrogram

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the batch processing
python batch_process.py

echo "Finished at: $(date)"
