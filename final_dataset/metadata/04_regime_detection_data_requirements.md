# Regime Detection Data Requirements

## 1. Purpose of this document

Regime detection is required before comparing motif stability across low, medium, and high-volatility regimes. The thesis uses two regime methods:

- Volatility-quantile segmentation
- Hidden Markov Model based regimes

The regime-detection dataset should be large enough to capture different market conditions but not so large that the thesis becomes computationally overloaded.

## 2. Core data principle

High-frequency data is needed mainly for crypto motif discovery.
Lower-frequency data is enough for cross-market regime validation.

"The benchmark should be data-rich enough to capture regime changes, but not unnecessarily large. The goal is not to collect every possible asset, but to build a reproducible and computationally feasible dataset that supports regime-conditioned motif analysis."

## 3. Recommended dataset scope

| Asset class | Instruments | Frequency | Period | Purpose | Priority |
|---|---|---:|---:|---|---|
| Crypto core | BTCUSDT, ETHUSDT | 1-minute | 2020 to 2025 | Main Matrix Profile, LoCoMotif, and regime-conditioned motif discovery | Must-have |
| Crypto derived | BTCUSDT, ETHUSDT | 5-minute, 15-minute, 1-hour, daily | Derived from 1-minute data | Cleaner regime detection and frequency sensitivity | Must-have |
| FX | EURUSD, GBPUSD | 1-hour or daily | 2015 to 2025 | Cross-asset regime comparison | Strongly recommended |
| Equity indices | S&P 500, NASDAQ 100, DAX | Daily | 2010 to 2025 | Broader market regime comparison | Strongly recommended |
| Volatility index | VIX | Daily | 2010 to 2025 | External validation for high-volatility regimes | Must-have for equity validation |
| Optional crypto | SOLUSDT or BNBUSDT | 1-minute or 5-minute | 2021 to 2025 | Robustness check | Optional |

## 4. Estimated data volume

| Period | Rows per asset | Rows for BTC + ETH | Approx Parquet size | Approx CSV size |
|---|---:|---:|---:|---:|
| 1 year | 525,600 rows per asset | 1,051,200 rows for BTC + ETH | 40-120 MB parquet | 150-300 MB CSV |
| 3 years | 1,576,800 rows per asset | 3,153,600 rows for BTC + ETH | 120-350 MB parquet | 450-900 MB CSV |
| 5 years | 2,628,000 rows per asset | 5,256,000 rows for BTC + ETH | 200-600 MB parquet | 750 MB-1.5 GB CSV |

Crypto 1-minute data dominates storage and computation. Daily FX, equity index, and VIX data are very small in comparison.

## 5. Dataset coverage timeline

```text
2010 ───────────────────────────────────────────── 2025
         Equity indices daily: S&P 500, NASDAQ 100, DAX
         VIX daily

2015 ─────────────────────────────── 2025
         FX daily/hourly: EURUSD, GBPUSD

2020 ─────────────────── 2025
         Crypto 1-minute: BTCUSDT, ETHUSDT
```

This timeline supports both intraday motif discovery and longer-horizon regime validation.

## 6. Regime detection feature requirements

### 6.1 Volatility-quantile regimes

This method needs:

- log returns
- rolling realized volatility
- volatility quantiles
- low, medium, high regime labels

One year is the minimum for crypto, but 3-5 years is better.

### 6.2 HMM regimes

HMM regimes should use engineered features, not raw close prices alone.

Recommended features:

- log_return
- abs_log_return
- rolling_volatility
- hl_range
- volume_zscore
- optional cross_asset_return or market_return

Daily HMM works best with 5+ years.
Hourly HMM works well with 2-5 years.
1-minute HMM can be noisy and should usually be smoothed or resampled.

## 7. Practical thesis recommendation

Use BTCUSDT and ETHUSDT 1-minute data from 2020-2025 as the core dataset.
Derive 5-minute, 15-minute, 1-hour, and daily bars from this data.
Use FX, equity index, and VIX data at daily or hourly frequency for regime validation.
Do not download too many assets before the regime pipeline is working.

## 8. Final locked dataset plan

| Priority | Dataset | Frequency | Period | Status |
|---|---|---:|---:|---|
| 1 | BTCUSDT | 1-minute | 2020-2025 | Required |
| 2 | ETHUSDT | 1-minute | 2020-2025 | Required |
| 3 | BTCUSDT, ETHUSDT resampled | 5m, 15m, 1h, daily | Derived | Required |
| 4 | EURUSD, GBPUSD | 1h or daily | 2015-2025 | Recommended |
| 5 | S&P 500, NASDAQ 100, DAX | daily | 2010-2025 | Recommended |
| 6 | VIX | daily | 2010-2025 | Required for validation |
| 7 | SOLUSDT or BNBUSDT | 1m or 5m | 2021-2025 | Optional |

## 9. Thesis-ready summary paragraph

The dataset design separates computationally intensive intraday motif discovery from lightweight cross-market regime validation. High-frequency 1-minute OHLCV data is retained for BTCUSDT and ETHUSDT, where Matrix Profile and LoCoMotif experiments are performed. Lower-frequency FX, equity index, and VIX data are used to evaluate whether detected regimes align with broader market conditions. This keeps the benchmark computationally feasible while still covering multiple asset classes and supporting regime-conditioned motif stability analysis.
