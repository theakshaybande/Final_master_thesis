# Final Dataset README

## Purpose

This dataset package supports the master's thesis "Regime-Conditioned Multivariate Motif Discovery in Financial Time Series: A Reproducible Empirical Benchmark Under Nonstationarity."

The pipeline downloads, processes, resamples, validates, and documents the data required for Matrix Profile, LoCoMotif, Hidden Markov Model regime detection, volatility-quantile regime detection, and external regime validation.

## Dataset Scope

The core empirical dataset is BTCUSDT and ETHUSDT 1-minute Binance spot OHLCV data from 2020-01-01 through 2025-12-31. These assets provide continuous, liquid, high-frequency crypto markets for motif discovery under nonstationarity.

The benchmark also includes lower-frequency market data:

| Asset class | Instruments | Frequency | Period | Source |
|---|---|---:|---:|---|
| Crypto core | BTCUSDT, ETHUSDT | 1-minute | 2020-2025 | Binance public klines |
| Crypto derived | BTCUSDT, ETHUSDT | 5m, 15m, 1h, 1d | Derived | Local OHLCV aggregation |
| FX | EURUSD, GBPUSD | daily | 2015-2025 | yfinance |
| Equity indices | S&P 500, NASDAQ 100, DAX | daily | 2010-2025 | yfinance |
| Volatility | VIX | daily | 2010-2025 | yfinance |

## Why BTCUSDT and ETHUSDT 1-Minute Are Core

BTCUSDT and ETHUSDT are the main assets for high-frequency Matrix Profile and LoCoMotif experiments because they are liquid, widely traded, and available from Binance public historical klines at 1-minute frequency. This gives enough observations to evaluate motif stability across different market regimes without expanding the dataset into too many assets before the benchmark is working.

## Why Resampled Frequencies Are Created

The pipeline derives 5-minute, 15-minute, 1-hour, and daily crypto bars from the 1-minute source data. These frequencies make it possible to compare motif behavior across temporal resolutions, reduce microstructure noise for regime detection, and support sensitivity checks without downloading separate datasets.

## Why FX, Equity Indices, and VIX Use Lower Frequency

FX pairs, equity indices, and VIX are used for external regime validation and cross-market comparison. Daily data is sufficient for this role because these datasets are not the main intraday motif discovery target. Keeping them lower frequency reduces storage and compute cost while preserving broad market context.

## Expected Data Volume

Crypto 1-minute data dominates the storage and runtime. BTCUSDT and ETHUSDT together should produce roughly 5.2 million rows for five non-leap years, with additional rows because 2020 and 2024 are leap years. Parquet compression should keep the processed files materially smaller than CSV.

Daily FX, equity index, and VIX files are small by comparison and should be quick to download, process, and validate.

## How to Run

Install requirements:

```powershell
pip install -r final_dataset\requirements_dataset.txt
```

Run the full pipeline:

```powershell
python final_dataset\scripts\run_all_dataset_pipeline.py
```

Run individual steps:

```powershell
python final_dataset\scripts\01_download_crypto_binance.py
python final_dataset\scripts\02_resample_crypto.py
python final_dataset\scripts\03_download_market_data_yfinance.py
python final_dataset\scripts\04_build_features.py
python final_dataset\scripts\05_validate_dataset.py
```

## Folder Structure

```text
final_dataset/
  raw/
    crypto/binance_1m/BTCUSDT/
    crypto/binance_1m/ETHUSDT/
    fx/
    equity_indices/
    volatility/
  processed/
    crypto/1m/
    crypto/5m/
    crypto/15m/
    crypto/1h/
    crypto/1d/
    fx/
    equity_indices/
    volatility/
  features/
    crypto/
    fx/
    equity_indices/
    volatility/
  metadata/
    dataset_inventory.csv
    download_log.csv
    missing_data_report.csv
    README.md
    04_regime_detection_data_requirements.md
  scripts/
```

## Reproducibility Notes

All processed and feature datasets are saved as Parquet using `pyarrow`. Timestamps are converted to UTC where possible. Downloads are skipped when output files already exist unless `OVERWRITE = True` is set inside the relevant script.

The optional robustness crypto asset is disabled by default with `DOWNLOAD_OPTIONAL_CRYPTO = False` in `01_download_crypto_binance.py`.
