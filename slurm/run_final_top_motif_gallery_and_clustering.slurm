#!/bin/bash
#SBATCH --job-name=final_visual_gallery
#SBATCH --partition=research
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --output=logs/final_visual_gallery_%j.out
#SBATCH --error=logs/final_visual_gallery_%j.err

set -euo pipefail

cd ~/Final_master_thesis
mkdir -p logs
source .thesis-env/bin/activate

hostname
date
pwd
which python

python scripts/run_final_top_motif_gallery_and_clustering.py
