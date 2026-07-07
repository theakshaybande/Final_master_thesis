#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# Recommended order:
# 1. Run micro first.
# 2. If micro succeeds, run tiny high/low.
# 3. If tiny succeeds, run small high/low.
# 4. Do not submit all at once until micro succeeds.

sbatch slurm/run_locomotif_micro_1h.slurm

# sbatch slurm/run_locomotif_high_vol_15m_tiny.slurm
# sbatch slurm/run_locomotif_low_vol_15m_tiny.slurm
# sbatch slurm/run_locomotif_high_vol_15m_small.slurm
# sbatch slurm/run_locomotif_low_vol_15m_small.slurm
