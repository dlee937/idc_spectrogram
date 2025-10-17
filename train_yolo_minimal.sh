#!/bin/bash
#SBATCH -J yolo_test
#SBATCH -p coc-gpu
#SBATCH -q coc-ice
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o test_%j.out

echo "Test job running"
hostname
nvidia-smi
