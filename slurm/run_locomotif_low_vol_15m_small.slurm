#!/usr/bin/env bash
#SBATCH --job-name=locomotif_low_small
#SBATCH --partition=research
## If the research partition is unavailable, change the line above to:
## #SBATCH --partition=education
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/locomotif_low_small_%j.out
#SBATCH --error=logs/locomotif_low_small_%j.err

set -euo pipefail

PROJECT_ROOT="${HOME}/Final_master_thesis"
cd "${PROJECT_ROOT}"
mkdir -p logs

echo "Job: locomotif_low_small"
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
  --max-points 1000 \
  --rho 0.65 \
  --lmin 24 \
  --lmax 96 \
  --nb 5 \
  --timeout-seconds 1800 \
  --run-mp yes \
  --run-locomotif yes \
  --hpc yes

echo "Completion date: $(date)"
