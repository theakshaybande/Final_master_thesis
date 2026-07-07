# HPC Run Instructions: Controlled LoCoMotif

Local preparation:

```powershell
git add scripts notebooks slurm reports/locomotif_controlled_slice_comparison
git commit -m "Add HPC SLURM controlled LoCoMotif jobs"
git push
```

On HPC:

```bash
cd ~/Final_master_thesis
git pull
source .thesis-env/bin/activate
python -m py_compile scripts/run_locomotif_controlled_slice_comparison.py
mkdir -p logs
sbatch slurm/run_locomotif_micro_1h.slurm
squeue -u $USER
tail -f logs/locomotif_micro_*.out
tail -f logs/locomotif_micro_*.err
```

Recommended job order:

1. Submit `slurm/run_locomotif_micro_1h.slurm`.
2. If micro succeeds, submit `slurm/run_locomotif_high_vol_15m_tiny.slurm` and `slurm/run_locomotif_low_vol_15m_tiny.slurm`.
3. If tiny succeeds, submit `slurm/run_locomotif_high_vol_15m_small.slurm` and `slurm/run_locomotif_low_vol_15m_small.slurm`.
4. Do not submit all jobs at once until the micro job succeeds.

After completion:

```bash
ls -R reports/locomotif_controlled_slice_comparison
cat reports/locomotif_controlled_slice_comparison/CONTROLLED_LOCOMOTIF_RUN_REPORT.md
python scripts/collect_locomotif_hpc_results.py
cat reports/locomotif_controlled_slice_comparison/HPC_RESULTS_SUMMARY.md
```

Commit outputs:

```bash
git add reports/locomotif_controlled_slice_comparison logs
git commit -m "Add HPC controlled LoCoMotif outputs"
git push
```

Local pull:

```powershell
git pull
```

Local safety note:

`scripts/run_locomotif_controlled_slice_comparison.py` refuses to run LoCoMotif on local Windows by default. Use SLURM with `--hpc yes` for real LoCoMotif experiments. `--allow-local` exists only for deliberate tiny smoke tests and should not be used for the controlled experiment runs.
