# Matrix Profile rerun: BTCUSDT/ETHUSDT at 15m/1h

Previous Matrix Profile job 5178 timed out due to the SLURM time limit.

The full all-asset/all-frequency Matrix Profile run was too large for one 16h job. This rerun intentionally restricts Matrix Profile execution to BTCUSDT and ETHUSDT at 15m and 1h.

The rerun still preserves the thesis design:

- Regime-agnostic Matrix Profile motif discovery
- Quantile-conditioned Matrix Profile motif discovery
- HMM-conditioned Matrix Profile motif discovery
- Univariate Matrix Profile
- Multivariate Matrix Profile

Partial timeout figures from job 5178 were preserved in `results/figures_partial_timeout_5178`.

The original full config was backed up as `hpc_full_run_BACKUP_before_mp_split.yaml`.
