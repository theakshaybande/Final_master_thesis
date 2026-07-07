#!/usr/bin/env bash
#SBATCH --job-name=locomotif_micro
#SBATCH --partition=research
## If the research partition is unavailable, change the line above to:
## #SBATCH --partition=education
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/locomotif_micro_%j.out
#SBATCH --error=logs/locomotif_micro_%j.err

set -euo pipefail

PROJECT_ROOT="${HOME}/Final_master_thesis"
cd "${PROJECT_ROOT}"
mkdir -p logs

echo "Job: locomotif_micro"
echo "Hostname: $(hostname)"
echo "Start date: $(date)"
echo "PWD: $(pwd)"

source .thesis-env/bin/activate

echo "Python: $(which python)"
python --version
python -c "import importlib.util; print('locomotif import:', importlib.util.find_spec('locomotif')); print('stumpy import:', importlib.util.find_spec('stumpy'))"

python scripts/run_locomotif_controlled_slice_comparison.py \
  --mode micro \
  --slice agnostic_1h \
  --feature close_z \
  --max-points 300 \
  --rho 0.65 \
  --lmin 12 \
  --lmax 48 \
  --nb 3 \
  --timeout-seconds 900 \
  --run-mp yes \
  --run-locomotif yes \
  --hpc yes

echo "Completion date: $(date)"
