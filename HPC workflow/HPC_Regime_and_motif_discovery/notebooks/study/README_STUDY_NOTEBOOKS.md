# Study Notebook Suite

These notebooks are for thesis illustration and result interpretation only. They read existing result files and create study figures/tables under `HPC workflow/HPC_Regime_and_motif_discovery/reports/study_notebooks`. They do not rerun regime detection, Matrix Profile, or LoCoMotif, and they do not modify original result files.

## Analysis-only safety

The notebooks never import or call STUMPY, STUMP/MSTUMP, HMM fitting, LoCoMotif search, or any other expensive experiment algorithm. They only use `pandas`, `numpy`, `matplotlib`, `pathlib`, `json`, and the local `study_helpers.py` module to read saved files and produce derived visualizations/tables.

## Notebooks

1. `01_regime_detection_visual_study.ipynb`
   - Loads quantile and HMM regime outputs.
   - Builds coverage tables, regime count charts, close/regime overlays, volatility-by-regime charts, transition heatmaps, and duration summaries.

2. `02_regime_comparison_quantile_vs_hmm.ipynb`
   - Loads HMM model selection, posterior confidence, quantile/HMM comparison, confusion, and persistence files.
   - Creates ARI/NMI charts, confusion heatmaps, confidence plots, and self-transition comparisons when the files are populated.

3. `03_matrix_profile_motif_visual_study.ipynb`
   - Loads Matrix Profile motif results, evaluation, runtime, and profile curves.
   - Creates file inventories, motif overview tables, distance threshold tables, top motif tables, motif overlays, timelines, overlap tables, evaluation summaries, and CPU/GPU runtime audits.

4. `04_locomotif_visual_study.ipynb`
   - Loads LoCoMotif result, evaluation, runtime, and failure files.
   - Creates scope tables, interval length plots, top motif-set tables, interval timelines, runtime/failure audits, and a LoCoMotif-vs-Matrix-Profile comparison.

## Outputs

The notebooks create these folders if needed:

- `reports/study_notebooks/figures/regime`
- `reports/study_notebooks/figures/regime_comparison`
- `reports/study_notebooks/figures/matrix_profile`
- `reports/study_notebooks/figures/locomotif`
- `reports/study_notebooks/tables`
- `reports/study_notebooks/html_exports`

Figures are saved as PNG and, where possible, PDF. Important tables are saved as CSV.

## Run Commands

Run from the repository root:

```bash
cd ~/Final_master_thesis
source .thesis-env/bin/activate
jupyter nbconvert --to notebook --execute \
  "HPC workflow/HPC_Regime_and_motif_discovery/notebooks/study/01_regime_detection_visual_study.ipynb" \
  --output "executed_01_regime_detection_visual_study.ipynb" \
  --output-dir "HPC workflow/HPC_Regime_and_motif_discovery/reports/study_notebooks/html_exports" \
  --ExecutePreprocessor.timeout=-1
```

```bash
jupyter nbconvert --to notebook --execute \
  "HPC workflow/HPC_Regime_and_motif_discovery/notebooks/study/02_regime_comparison_quantile_vs_hmm.ipynb" \
  --output "executed_02_regime_comparison_quantile_vs_hmm.ipynb" \
  --output-dir "HPC workflow/HPC_Regime_and_motif_discovery/reports/study_notebooks/html_exports" \
  --ExecutePreprocessor.timeout=-1
```

```bash
jupyter nbconvert --to notebook --execute \
  "HPC workflow/HPC_Regime_and_motif_discovery/notebooks/study/03_matrix_profile_motif_visual_study.ipynb" \
  --output "executed_03_matrix_profile_motif_visual_study.ipynb" \
  --output-dir "HPC workflow/HPC_Regime_and_motif_discovery/reports/study_notebooks/html_exports" \
  --ExecutePreprocessor.timeout=-1
```

```bash
jupyter nbconvert --to notebook --execute \
  "HPC workflow/HPC_Regime_and_motif_discovery/notebooks/study/04_locomotif_visual_study.ipynb" \
  --output "executed_04_locomotif_visual_study.ipynb" \
  --output-dir "HPC workflow/HPC_Regime_and_motif_discovery/reports/study_notebooks/html_exports" \
  --ExecutePreprocessor.timeout=-1
```

## Notes

- The notebooks resolve exact expected result filenames first, then fall back to suffixed files such as local smoke-test outputs if those are the only files present.
- Every numerical table or figure is derived from existing parquet result files or thesis-scope feature files.
- If a file or column is missing, the relevant cell prints that fact and continues.
- The quantile regime caveat is included in the regime notebooks: quantile outputs store `rolling_volatility_60` as the actual volatility column across quantile method identifiers, so they are interpreted as 60-period rolling-volatility regimes with different regime-count granularities.
