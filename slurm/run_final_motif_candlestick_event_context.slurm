#!/bin/bash
#SBATCH --job-name=final_visual_candlestick
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --output=logs/final_visual_candlestick_%j.out
#SBATCH --error=logs/final_visual_candlestick_%j.err

set -euo pipefail

cd ~/Final_master_thesis
mkdir -p logs
source .thesis-env/bin/activate

hostname
date
pwd
which python

python scripts/run_final_motif_candlestick_event_context.py
