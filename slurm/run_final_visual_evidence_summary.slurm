#!/bin/bash
#SBATCH --job-name=final_visual_summary
#SBATCH --partition=research
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --output=logs/final_visual_summary_%j.out
#SBATCH --error=logs/final_visual_summary_%j.err

set -euo pipefail

cd ~/Final_master_thesis
mkdir -p logs
source .thesis-env/bin/activate

hostname
date
pwd
which python

python scripts/run_final_visual_evidence_summary.py
