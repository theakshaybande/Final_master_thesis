#!/bin/bash
#SBATCH --job-name=final_uni_mp_story
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/final_univariate_visual_story_%j.out
#SBATCH --error=logs/final_univariate_visual_story_%j.err

set -euo pipefail

cd "$HOME/Final_master_thesis"
source .thesis-env/bin/activate

export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs reports/final_story/executed_notebooks

echo "Job started at: $(date)"
echo "Node: $(hostname)"
python --version

jupyter nbconvert \
  --to notebook \
  --execute "notebooks/final_story/01_univariate_matrix_profile_visual_story.ipynb" \
  --output-dir "reports/final_story/executed_notebooks" \
  --output "executed_01_univariate_matrix_profile_visual_story.ipynb" \
  --ExecutePreprocessor.timeout=-1

echo "Job finished at: $(date)"
