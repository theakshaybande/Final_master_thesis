# LoCoMotif Initial Study Notebooks

This folder contains the first controlled motif-discovery notebooks for the thesis:

**Regime-Conditioned Multivariate Motif Discovery in Financial Time Series: A Reproducible Empirical Benchmark Under Nonstationarity**

## Notebooks

`01_locomotif_basics_with_visual_intuition.ipynb` introduces the conceptual difference between fixed-length Matrix Profile motifs and variable-length, time-warped LoCoMotif-style motif sets. It uses synthetic examples first, then prepares a small BTCUSDT 1-hour feature sample for financial intuition.

`02_matrix_profile_vs_locomotif_initial_comparison.ipynb` runs a controlled initial comparison pattern on BTCUSDT and ETHUSDT 1-hour feature data. It executes Matrix Profile baselines when `stumpy` is available and prepares a robust LoCoMotif adapter/placeholder when the LoCoMotif package is not installed.

## Thesis Connection

These notebooks are not the final benchmark. They establish a reproducible discussion workflow before the full regime-conditioned experiments:

- start with a small, interpretable sample;
- keep feature preparation identical across methods;
- compare fixed-length and variable-length motif logic;
- save figures, tables, and configuration files for supervisor review.

## Next Steps

The next thesis notebooks should add volatility-quantile and HMM regime labels, then rerun Matrix Profile and LoCoMotif inside full-sample, low-volatility, medium-volatility, high-volatility, and HMM-state subsets.
