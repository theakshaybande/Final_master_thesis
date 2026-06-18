#!/bin/bash
#SBATCH --job-name=thesis_mp_15m_1h
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/thesis_mp_15m_1h_%j.out
#SBATCH --error=logs/thesis_mp_15m_1h_%j.err

set -euo pipefail

cd "$HOME/Final_master_thesis"

source .thesis-env/bin/activate

export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python:"
which python
python --version

echo "Checking environment..."
python - <<'PY'
import pandas, numpy, pyarrow, sklearn, hmmlearn, stumpy
print("Matrix Profile environment ok.")
PY

NOTEBOOK="HPC workflow/HPC_Regime_and_motif_discovery/notebooks/03_matrix_profile_motif_discovery.ipynb"
OUTPUT_DIR="HPC workflow/HPC_Regime_and_motif_discovery/reports/executed_notebooks"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "Running notebook:"
echo "$NOTEBOOK"

jupyter nbconvert \
  --to notebook \
  --execute "$NOTEBOOK" \
  --output-dir "$OUTPUT_DIR" \
  --output "executed_03_matrix_profile_motif_discovery_crypto_15m_1h.ipynb" \
  --ExecutePreprocessor.timeout=-1

echo "Job finished at: $(date)"
