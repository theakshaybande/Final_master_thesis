# HPC Regime and Motif Discovery Workflow

This workflow runs the thesis pipeline for regime-conditioned multivariate motif discovery in financial time series. It separates regime detection from motif mining so that Matrix Profile and LoCoMotif consume the same saved Quantile and HMM regime labels.

## Folder Structure

```text
HPC_Regime_and_motif_discovery/
  notebooks/
    01_quantile_regime_detection.ipynb
    02_hmm_regime_detection.ipynb
    03_matrix_profile_motif_discovery.ipynb
    04_locomotif_motif_discovery.ipynb
  results/
    regimes/quantile/
    regimes/hmm/
    motifs/matrix_profile/
    motifs/locomotif/
    comparisons/
    figures/
    tables/
    logs/
    configs/
  src/
  run_configs/
```

## What Each Notebook Does

1. `01_quantile_regime_detection.ipynb` creates transparent volatility regimes from rolling volatility quantiles.
2. `02_hmm_regime_detection.ipynb` fits Gaussian HMM regimes when `hmmlearn` is installed, then compares them with Quantile labels.
3. `03_matrix_profile_motif_discovery.ipynb` runs regime-agnostic and regime-conditioned Matrix Profile motif discovery. Conditioned runs use continuous same-regime segments and do not concatenate disconnected regime periods.
4. `04_locomotif_motif_discovery.ipynb` runs the real LoCoMotif API when installed. If `dtai-locomotif` is unavailable or fails, the notebook saves explicit failure rows and does not create substitute motif detections.

## Local Run

Local mode is a smoke test only. On Windows, `EXECUTION_MODE = "auto"` resolves to local mode and uses `run_configs/local_smoke_test.yaml`.

```powershell
cd "C:\Users\learn\OneDrive\Desktop\Final Masters Thesis\HPC workflow\HPC_Regime_and_motif_discovery"
jupyter lab
```

Run the notebooks in numeric order. Local outputs include the `_LOCAL_SMOKE_TEST` suffix.

## HPC Run

Sync the project folder to HPC storage, install dependencies, then run from the workflow directory. On Linux/HPC, `EXECUTION_MODE = "auto"` resolves to HPC mode unless manually overridden.

```bash
cd ~/FinalMastersThesis/HPC\ workflow/HPC_Regime_and_motif_discovery
jupyter nbconvert --to notebook --execute notebooks/01_quantile_regime_detection.ipynb --output executed_01_quantile_regime_detection.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_hmm_regime_detection.ipynb --output executed_02_hmm_regime_detection.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_matrix_profile_motif_discovery.ipynb --output executed_03_matrix_profile_motif_discovery.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_locomotif_motif_discovery.ipynb --output executed_04_locomotif_motif_discovery.ipynb
```

Optional long-run patterns:

```bash
tmux new -s thesis_motifs
nohup jupyter nbconvert --to notebook --execute notebooks/03_matrix_profile_motif_discovery.ipynb --output executed_03_matrix_profile_motif_discovery.ipynb > results/logs/nbconvert_03.out 2>&1 &
```

## Expected Outputs

Regime labels, summaries, transition matrices, motif results, runtime tables, evaluation tables, figures, logs, config snapshots, and environment snapshots are written under `results/`.

The final comparison tables are produced by Notebook 04 under `results/comparisons/`, including:

- `mp_vs_locomotif_summary.csv`
- `agnostic_vs_conditioned_summary.csv`
- `regime_method_comparison_summary.csv`
- `runtime_comparison.csv`
- `thesis_key_results_table.csv`

## Debugging

Check `results/logs/` first. Every notebook saves a stage log and a failure table. Failures are recorded per asset, frequency, window, regime, or parameter setting so one failed run does not stop the whole batch.

Common blockers:

- Missing `hmmlearn`: install `hmmlearn`; Notebook 02 records an explicit failure otherwise.
- Missing `stumpy`: install `stumpy`; Notebook 03 records Matrix Profile failures otherwise.
- Missing `dtai-locomotif`: install the real LoCoMotif package; Notebook 04 will not fake results.
- No feature files found: confirm `final_dataset/features/**/**/*_features_*.parquet` exists after syncing to HPC.

## Thesis Mapping

The workflow supports the thesis questions by comparing regime-agnostic versus regime-conditioned motif discovery, Quantile versus HMM regime definitions, Matrix Profile versus LoCoMotif methods, cross-regime recurrence, time-split robustness, sensitivity to windows/parameters, and runtime scalability under nonstationarity.

