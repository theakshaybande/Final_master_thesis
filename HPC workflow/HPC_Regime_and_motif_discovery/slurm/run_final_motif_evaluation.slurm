#!/bin/bash
#SBATCH --job-name=final_motif_evaluation
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=reports/final_motif_evaluation/logs/final_motif_evaluation_%j.out
#SBATCH --error=reports/final_motif_evaluation/logs/final_motif_evaluation_%j.err

set -euo pipefail

cd "$HOME/Final_master_thesis/HPC workflow/HPC_Regime_and_motif_discovery"
mkdir -p reports/final_motif_evaluation/logs

source "$HOME/Final_master_thesis/.thesis-env/bin/activate"

python scripts/run_final_motif_evaluation.py \
  --config run_configs/final_motif_evaluation.yaml \
  --verbose
