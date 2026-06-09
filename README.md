# Regime-Conditioned Multivariate Motif Discovery In Financial Time Series

This repository supports the master thesis "Regime-Conditioned Multivariate Motif Discovery In Financial Time Series: A Reproducible Empirical Benchmark Under Nonstationarity". The project benchmarks motif discovery methods on financial time series under changing market regimes, comparing regime-agnostic discovery with regime-conditioned workflows that segment observations by volatility and latent state structure before motif search.

## Main Methods

- Matrix Profile for fixed-length motif discovery and nearest-neighbor structure in time series.
- LoCoMotif for variable-length and locally constrained motif discovery.
- Volatility quantile regimes based on rolling volatility and empirical quantile thresholds.
- HMM regimes using Hidden Markov Models fitted to engineered financial features.
- Regime-agnostic vs regime-conditioned motif discovery to evaluate whether conditioning improves interpretability, stability, and empirical performance under nonstationarity.

## Suggested Folder Structure

```text
src/          Reusable Python package code
scripts/      Command-line utilities and pipeline entry points
notebooks/    Research notebooks and exploratory analysis
configs/      Experiment and pipeline configuration files
docs/         Thesis notes, methodology, and reproducibility documentation
references/   Bibliography and citation files
```

Raw data, downloaded market data, generated reports, model artifacts, caches, and heavy result folders are excluded from Git. Sync data separately through HPC storage, cloud storage, or another controlled data-management workflow.

## Basic Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name final-master-thesis
```
