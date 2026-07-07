#!/usr/bin/env bash
#SBATCH --job-name=locomotif_low_tiny
#SBATCH --partition=research
## If the research partition is unavailable, change the line above to:
## #SBATCH --partition=education
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/locomotif_low_tiny_%j.out
#SBATCH --error=logs/locomotif_low_tiny_%j.err

set -euo pipefail

PROJECT_ROOT="${HOME}/Final_master_thesis"
cd "${PROJECT_ROOT}"
mkdir -p logs

echo "Job: locomotif_low_tiny"
echo "Hostname: $(hostname)"
echo "Start date: $(date)"
echo "PWD: $(pwd)"

source .thesis-env/bin/activate

echo "Python: $(which python)"
python --version
python -c "import importlib.util; print('locomotif import:', importlib.util.find_spec('locomotif')); print('stumpy import:', importlib.util.find_spec('stumpy'))"

python scripts/run_locomotif_controlled_slice_comparison.py \
  --mode full \
  --slice low_vol \
  --feature close_z \
  --max-points 500 \
  --rho 0.65 \
  --lmin 12 \
  --lmax 48 \
  --nb 3 \
  --timeout-seconds 1200 \
  --run-mp yes \
  --run-locomotif yes \
  --hpc yes

echo "Completion date: $(date)"
