# Final Results Consolidation for ChatGPT

Generated: 2026-06-25T10:29:41.919199+00:00

## Direct answer to the research question

The results support the presence of recurring subsequence shapes in the studied cryptocurrency series, and show that regime-conditioned discovery produces identifiable motifs within volatility states, while agnostic discovery provides a global recurrence baseline. Differences in row counts across modes and algorithms describe the configured search process, not statistical significance, causality, or trading profitability. LoCoMotif is reported as a controlled subset experiment rather than a full-scale benchmark.

## Key numbers

| metric | value | method | source_file | note |
| --- | --- | --- | --- | --- |
| quantile total rows | 73,097,388 | Quantile | results/regimes/quantile/quantile_regime_labels.parquet |  |
| HMM total rows | 8,121,932 | HMM | results/regimes/hmm/hmm_regime_labels.parquet |  |
| quantile failure rows | 0 | Quantile | results/logs/01_quantile_failures.parquet |  |
| HMM failure rows | 0 | HMM | results/logs/02_hmm_failures.parquet |  |
| Matrix Profile motif rows | 81,987 | Matrix Profile | results/motifs/matrix_profile/matrix_profile_motif_results.parquet |  |
| Matrix Profile evaluation rows | 1,384 | Matrix Profile | results/motifs/matrix_profile/matrix_profile_evaluation.parquet |  |
| Matrix Profile runtime rows | 368 | Matrix Profile | results/motifs/matrix_profile/matrix_profile_runtime.parquet |  |
| Matrix Profile figure count | 84 | Matrix Profile | results/figures |  |
| Matrix Profile agnostic rows | 720 | Matrix Profile | results/motifs/matrix_profile/matrix_profile_motif_results.parquet |  |
| Matrix Profile conditioned rows | 81,267 | Matrix Profile | results/motifs/matrix_profile/matrix_profile_motif_results.parquet |  |
| Matrix Profile GPU true rows | 0 | Matrix Profile | results/motifs/matrix_profile/matrix_profile_motif_results.parquet |  |
| Matrix Profile GPU false rows | 81,987 | Matrix Profile | results/motifs/matrix_profile/matrix_profile_motif_results.parquet |  |
| LoCoMotif motif rows | 18,147 | LoCoMotif | results/motifs/locomotif/locomotif_motif_results.parquet |  |
| LoCoMotif evaluation rows | 38 | LoCoMotif | results/motifs/locomotif/locomotif_evaluation.parquet |  |
| LoCoMotif runtime rows | 662 | LoCoMotif | results/motifs/locomotif/locomotif_runtime.parquet |  |
| LoCoMotif failure rows | not available | LoCoMotif | results/motifs/locomotif/04_locomotif_failures.parquet |  |
| LoCoMotif figure count | 6 | LoCoMotif | results/figures |  |
| selected final figure count | 18 | All | results/figures |  |

## Quantile regime results

- Total label rows: **73,097,388**
- Assets: BTCUSDT, DAX, ETHUSDT, EURUSD, GBPUSD, NASDAQ100, SP500, VIX
- Frequencies: 15m, 1d, 1h, 1m, 5m
- Timestamp range: 2010-01-04T00:00:00+00:00 to 2025-12-31T23:59:00+00:00
- Regime methods: quantile_2_rolling_240, quantile_2_rolling_30, quantile_2_rolling_60, quantile_3_rolling_240, quantile_3_rolling_30, quantile_3_rolling_60, quantile_4_rolling_240, quantile_4_rolling_30, quantile_4_rolling_60
- Regime labels: extreme_vol, high_vol, low_vol, medium_vol
- Failure rows: **0**

### Rows by asset

| asset | rows |
| --- | --- |
| BTCUSDT | 36,450,342 |
| ETHUSDT | 36,450,333 |
| DAX | 36,531 |
| NASDAQ100 | 36,216 |
| SP500 | 36,216 |
| VIX | 36,216 |
| EURUSD | 25,767 |
| GBPUSD | 25,767 |

### Rows by frequency

| frequency | rows |
| --- | --- |
| 1m | 56,774,781 |
| 5m | 11,355,012 |
| 15m | 3,785,040 |
| 1h | 946,386 |
| 1d | 236,169 |

### Rows by regime method

| regime_method | rows |
| --- | --- |
| quantile_2_rolling_30 | 8,121,932 |
| quantile_3_rolling_30 | 8,121,932 |
| quantile_4_rolling_30 | 8,121,932 |
| quantile_2_rolling_60 | 8,121,932 |
| quantile_3_rolling_60 | 8,121,932 |
| quantile_4_rolling_60 | 8,121,932 |
| quantile_2_rolling_240 | 8,121,932 |
| quantile_3_rolling_240 | 8,121,932 |
| quantile_4_rolling_240 | 8,121,932 |

### Rows by regime label

| regime_label | rows |
| --- | --- |
| low_vol | 26,396,274 |
| high_vol | 26,396,262 |
| medium_vol | 14,213,349 |
| extreme_vol | 6,091,503 |

### Thesis-scope file status

| asset | frequency | exists | rows | path |
| --- | --- | --- | --- | --- |
| BTCUSDT | 15m | True | 1,892,520 | results/regimes/quantile/BTCUSDT_15m_quantile_regimes.parquet |
| BTCUSDT | 1h | True | 473,193 | results/regimes/quantile/BTCUSDT_1h_quantile_regimes.parquet |
| ETHUSDT | 15m | True | 1,892,520 | results/regimes/quantile/ETHUSDT_15m_quantile_regimes.parquet |
| ETHUSDT | 1h | True | 473,193 | results/regimes/quantile/ETHUSDT_1h_quantile_regimes.parquet |

### Supporting table shapes

| table | shape |
| --- | --- |
| summary | 432 rows x 12 columns |
| transition_matrix | 1,203 rows x 7 columns |
| failures | 0 rows x 8 columns |

### Volatility-column audit

The quantile regime outputs store `rolling_volatility_60` as the actual volatility column across all quantile method identifiers. Therefore, quantile regimes should be interpreted as 60-period rolling-volatility regimes with different regime-count granularities rather than as separate 30/60/240 volatility-horizon experiments.

## HMM regime results

- Total label rows: **8,121,932**
- Assets: BTCUSDT, DAX, ETHUSDT, EURUSD, GBPUSD, NASDAQ100, SP500, VIX
- Frequencies: 15m, 1d, 1h, 1m, 5m
- Timestamp range: 2010-01-04T00:00:00+00:00 to 2025-12-31T23:59:00+00:00
- Regime methods: hmm_2_states, hmm_4_states
- Regime labels: extreme_vol, high_vol, low_vol, medium_vol
- Failure rows: **0**

### Rows by asset

| asset | rows |
| --- | --- |
| BTCUSDT | 4,050,038 |
| ETHUSDT | 4,050,037 |
| DAX | 4,059 |
| NASDAQ100 | 4,024 |
| SP500 | 4,024 |
| VIX | 4,024 |
| EURUSD | 2,863 |
| GBPUSD | 2,863 |

### Rows by frequency

| frequency | rows |
| --- | --- |
| 1m | 6,308,309 |
| 5m | 1,261,668 |
| 15m | 420,560 |
| 1h | 105,154 |
| 1d | 26,241 |

### Rows by regime method

| regime_method | rows |
| --- | --- |
| hmm_4_states | 7,491,098 |
| hmm_2_states | 630,834 |

### Rows by regime label

| regime_label | rows |
| --- | --- |
| low_vol | 3,120,058 |
| medium_vol | 2,940,630 |
| high_vol | 1,801,920 |
| extreme_vol | 259,324 |

### Thesis-scope file status

| asset | frequency | exists | rows | path |
| --- | --- | --- | --- | --- |
| BTCUSDT | 15m | True | 210,280 | results/regimes/hmm/BTCUSDT_15m_hmm_regimes.parquet |
| BTCUSDT | 1h | True | 52,577 | results/regimes/hmm/BTCUSDT_1h_hmm_regimes.parquet |
| ETHUSDT | 15m | True | 210,280 | results/regimes/hmm/ETHUSDT_15m_hmm_regimes.parquet |
| ETHUSDT | 1h | True | 52,577 | results/regimes/hmm/ETHUSDT_1h_hmm_regimes.parquet |

### Supporting table shapes

| table | shape |
| --- | --- |
| summary | 62 rows x 12 columns |
| transition_matrix | 205 rows x 7 columns |
| model_selection | 48 rows x 10 columns |
| persistence_metrics | 62 rows x 7 columns |
| feature_diagnostics | 93 rows x 9 columns |
| quantile_comparison | 144 rows x 8 columns |
| confusion_table | 1,548 rows x 7 columns |
| failures | 0 rows x 8 columns |

## Matrix Profile motif discovery

- Total motif rows: **81,987**
- Assets: BTCUSDT, ETHUSDT
- Frequencies: 15m, 1h
- Evaluation table: 1,384 rows x 16 columns
- Evaluation numeric metrics: number_of_motifs, recurrence_count, mean_motif_distance_or_score, median_motif_distance, time_split_stability, cross_regime_overlap, runtime_seconds, window_length
- Runtime table: 368 rows x 7 columns
- Profile table: 0 rows x not available columns; **empty**

Matrix Profile successfully discovered motifs when motif result rows are present. Agnostic mode identifies global recurring subsequences, whereas conditioned mode repeats motif search within regimes, methods, and eligible segments. Consequently, a larger conditioned row count partly reflects repeated search scope. Raw motif count is not statistical significance, and shape similarity is not evidence of trading profitability.

### Rows by asset, frequency, and mode

| asset | frequency | mode | rows |
| --- | --- | --- | --- |
| ETHUSDT | 15m | conditioned | 34,071 |
| BTCUSDT | 15m | conditioned | 30,633 |
| BTCUSDT | 1h | conditioned | 8,913 |
| ETHUSDT | 1h | conditioned | 7,650 |
| BTCUSDT | 1h | agnostic | 200 |
| ETHUSDT | 1h | agnostic | 200 |
| BTCUSDT | 15m | agnostic | 160 |
| ETHUSDT | 15m | agnostic | 160 |

### Rows by regime method

| regime_method | rows |
| --- | --- |
| quantile_2_rolling_240 | 15,618 |
| quantile_2_rolling_30 | 15,618 |
| quantile_2_rolling_60 | 15,618 |
| quantile_3_rolling_240 | 6,885 |
| quantile_3_rolling_30 | 6,885 |
| quantile_3_rolling_60 | 6,885 |
| quantile_4_rolling_240 | 4,559 |
| quantile_4_rolling_30 | 4,559 |
| quantile_4_rolling_60 | 4,559 |
| none | 720 |
| hmm_4_states | 81 |

### Rows by regime label

| regime_label | rows |
| --- | --- |
| low_vol | 42,609 |
| high_vol | 31,644 |
| extreme_vol | 7,014 |
| all | 720 |

### Rows by frequency and window length

| frequency | window_length | rows |
| --- | --- | --- |
| 15m | 32 | 27,531 |
| 15m | 64 | 20,004 |
| 15m | 128 | 11,840 |
| 1h | 24 | 5,840 |
| 15m | 256 | 5,649 |
| 1h | 48 | 5,009 |
| 1h | 72 | 3,803 |
| 1h | 168 | 1,682 |
| 1h | 336 | 629 |

### Rows by feature set

| feature_set | rows |
| --- | --- |
| log_return | 20,629 |
| close | 20,615 |
| rolling_volatility_60 | 20,486 |
| log_return,abs_log_return,rolling_volatility_60,rolling_volatility_30,hl_range,volume_zscore,quote_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,spread_proxy,open,high,low,close,volume,pct_return,rolling_volatility_240,absolute_return,squared_return,rolling_vol,range | 20,257 |

### Rows by profile type

| profile_type | rows |
| --- | --- |
| univariate | 61,730 |
| multivariate | 20,257 |

### Rows by asset, frequency, and regime label

| asset | frequency | regime_label | rows |
| --- | --- | --- | --- |
| ETHUSDT | 15m | low_vol | 17,694 |
| BTCUSDT | 15m | low_vol | 15,423 |
| ETHUSDT | 15m | high_vol | 13,650 |
| BTCUSDT | 15m | high_vol | 12,357 |
| BTCUSDT | 1h | low_vol | 5,307 |
| ETHUSDT | 1h | low_vol | 4,185 |
| BTCUSDT | 15m | extreme_vol | 2,853 |
| BTCUSDT | 1h | high_vol | 2,832 |
| ETHUSDT | 1h | high_vol | 2,805 |
| ETHUSDT | 15m | extreme_vol | 2,727 |
| BTCUSDT | 1h | extreme_vol | 774 |
| ETHUSDT | 1h | extreme_vol | 660 |
| BTCUSDT | 1h | all | 200 |
| ETHUSDT | 1h | all | 200 |
| BTCUSDT | 15m | all | 160 |
| ETHUSDT | 15m | all | 160 |

### Rows by asset, frequency, and window length

| asset | frequency | window_length | rows |
| --- | --- | --- | --- |
| ETHUSDT | 15m | 32 | 14,468 |
| BTCUSDT | 15m | 32 | 13,063 |
| ETHUSDT | 15m | 64 | 10,499 |
| BTCUSDT | 15m | 64 | 9,505 |
| ETHUSDT | 15m | 128 | 6,232 |
| BTCUSDT | 15m | 128 | 5,608 |
| BTCUSDT | 1h | 24 | 3,040 |
| ETHUSDT | 15m | 256 | 3,032 |
| ETHUSDT | 1h | 24 | 2,800 |
| BTCUSDT | 1h | 48 | 2,680 |
| BTCUSDT | 15m | 256 | 2,617 |
| ETHUSDT | 1h | 48 | 2,329 |
| BTCUSDT | 1h | 72 | 2,077 |
| ETHUSDT | 1h | 72 | 1,726 |
| BTCUSDT | 1h | 168 | 955 |
| ETHUSDT | 1h | 168 | 727 |
| BTCUSDT | 1h | 336 | 361 |
| ETHUSDT | 1h | 336 | 268 |

### Rows by asset, frequency, and feature set

| asset | frequency | feature_set | rows |
| --- | --- | --- | --- |
| ETHUSDT | 15m | close | 8,643 |
| ETHUSDT | 15m | log_return | 8,618 |
| ETHUSDT | 15m | rolling_volatility_60 | 8,577 |
| ETHUSDT | 15m | log_return,abs_log_return,rolling_volatility_60,rolling_volatility_30,hl_range,volume_zscore,quote_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,spread_proxy,open,high,low,close,volume,pct_return,rolling_volatility_240,absolute_return,squared_return,rolling_vol,range | 8,393 |
| BTCUSDT | 15m | log_return | 7,750 |
| BTCUSDT | 15m | close | 7,723 |
| BTCUSDT | 15m | rolling_volatility_60 | 7,708 |
| BTCUSDT | 15m | log_return,abs_log_return,rolling_volatility_60,rolling_volatility_30,hl_range,volume_zscore,quote_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,spread_proxy,open,high,low,close,volume,pct_return,rolling_volatility_240,absolute_return,squared_return,rolling_vol,range | 7,612 |
| BTCUSDT | 1h | close | 2,288 |
| BTCUSDT | 1h | log_return | 2,288 |
| BTCUSDT | 1h | log_return,abs_log_return,rolling_volatility_60,rolling_volatility_30,hl_range,volume_zscore,quote_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,spread_proxy,open,high,low,close,volume,pct_return,rolling_volatility_240,absolute_return,squared_return,rolling_vol,range | 2,282 |
| BTCUSDT | 1h | rolling_volatility_60 | 2,255 |
| ETHUSDT | 1h | log_return | 1,973 |
| ETHUSDT | 1h | log_return,abs_log_return,rolling_volatility_60,rolling_volatility_30,hl_range,volume_zscore,quote_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,spread_proxy,open,high,low,close,volume,pct_return,rolling_volatility_240,absolute_return,squared_return,rolling_vol,range | 1,970 |
| ETHUSDT | 1h | close | 1,961 |
| ETHUSDT | 1h | rolling_volatility_60 | 1,946 |

### GPU audit

| used_gpu | rows |
| --- | --- |
| False | 81,987 |

### Evaluation grouped summaries

| asset | frequency | mode | number_of_motifs_count | number_of_motifs_mean | number_of_motifs_median | number_of_motifs_min | number_of_motifs_max | recurrence_count_count | recurrence_count_mean | recurrence_count_median | recurrence_count_min | recurrence_count_max | mean_motif_distance_or_score_count | mean_motif_distance_or_score_mean | mean_motif_distance_or_score_median | mean_motif_distance_or_score_min | mean_motif_distance_or_score_max | median_motif_distance_count | median_motif_distance_mean | median_motif_distance_median | median_motif_distance_min | median_motif_distance_max | time_split_stability_count | time_split_stability_mean | time_split_stability_median | time_split_stability_min | time_split_stability_max | cross_regime_overlap_count | cross_regime_overlap_mean | cross_regime_overlap_median | cross_regime_overlap_min | cross_regime_overlap_max | runtime_seconds_count | runtime_seconds_mean | runtime_seconds_median | runtime_seconds_min | runtime_seconds_max | window_length_count | window_length_mean | window_length_median | window_length_min | window_length_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 15m | agnostic | 16 | 10 | 10 | 10 | 10 | 16 | 20 | 20 | 20 | 20 | 16 | 3.1466 | 1.6057 | 0.1267 | 12.3027 | 16 | 3.1964 | 1.6106 | 0.1283 | 12.6423 | 16 | 0.6079 | 0.6667 | 0.2500 | 1 | 16 | 0.3563 | 0.3000 | 0 | 0.6000 | 16 | 21,026.8724 | 78.7733 | 71.7669 | 85,043.8745 | 16 | 120 | 96 | 32 | 256 |
| BTCUSDT | 15m | conditioned | 288 | 106.3646 | 73.5000 | 10 | 369 | 288 | 212.7292 | 147 | 20 | 738 | 288 | 7.8997 | 7.0938 | 1.1818 | 19.6937 | 288 | 7.7914 | 6.9935 | 0.9784 | 19.7630 | 288 | 0.4499 | 0.3958 | 0.0847 | 1 | 288 | 0 | 0 | 0 | 0 | 288 | 22.0391 | 1.3512 | 0.1175 | 290.3519 | 288 | 120 | 96 | 32 | 256 |
| BTCUSDT | 1h | agnostic | 20 | 10 | 10 | 10 | 10 | 20 | 20 | 20 | 20 | 20 | 20 | 4.3930 | 2.4934 | 0.1337 | 19.3145 | 20 | 4.5078 | 2.4567 | 0.1376 | 20.0177 | 20 | 0.6655 | 0.6667 | 0.2500 | 1 | 20 | 0.3650 | 0.4000 | 0.1000 | 0.6000 | 20 | 1,276.2124 | 7.6124 | 6.6093 | 5,177.5753 | 20 | 129.6000 | 72 | 24 | 336 |
| BTCUSDT | 1h | conditioned | 360 | 24.7583 | 20 | 1 | 90 | 360 | 49.5167 | 40 | 2 | 180 | 360 | 8.1172 | 6.4869 | 0.6904 | 23.5359 | 360 | 8.0685 | 6.4710 | 0.6955 | 23.5359 | 360 | 0.6933 | 0.7033 | 0 | 1 | 360 | 0 | 0 | 0 | 0 | 360 | 3.3990 | 0.2563 | 0.0060 | 41.4260 | 360 | 129.6000 | 72 | 24 | 336 |
| ETHUSDT | 15m | agnostic | 16 | 10 | 10 | 10 | 10 | 16 | 20 | 20 | 20 | 20 | 16 | 3.1362 | 1.6449 | 0.1164 | 12.4496 | 16 | 3.2160 | 1.6647 | 0.1161 | 13.0193 | 16 | 0.6014 | 0.6667 | 0.1111 | 1 | 16 | 0.3250 | 0.3500 | 0.1000 | 0.6000 | 16 | 20,021.6208 | 71.7170 | 66.1298 | 81,329.2132 | 16 | 120 | 96 | 32 | 256 |
| ETHUSDT | 15m | conditioned | 304 | 112.0757 | 76 | 1 | 366 | 304 | 224.1513 | 152 | 2 | 732 | 304 | 7.9150 | 7.0777 | 1.3295 | 19.9868 | 304 | 7.7799 | 7.0759 | 1.1232 | 19.9868 | 304 | 0.3724 | 0.3628 | 0 | 1 | 304 | 0 | 0 | 0 | 0 | 304 | 20.1281 | 1.2739 | 0.0059 | 221.0123 | 304 | 120 | 96 | 32 | 256 |
| ETHUSDT | 1h | agnostic | 20 | 10 | 10 | 10 | 10 | 20 | 20 | 20 | 20 | 20 | 20 | 4.3375 | 2.5552 | 0.1569 | 19.2805 | 20 | 4.4032 | 2.5991 | 0.1537 | 19.8347 | 20 | 0.5827 | 0.6667 | 0.2500 | 1 | 20 | 0.4000 | 0.4000 | 0.1000 | 0.6000 | 20 | 1,218.4067 | 7.0035 | 6.5053 | 4,891.3060 | 20 | 129.6000 | 72 | 24 | 336 |
| ETHUSDT | 1h | conditioned | 360 | 21.2500 | 17 | 1 | 70 | 360 | 42.5000 | 34 | 2 | 140 | 360 | 8.2715 | 6.5152 | 0.8605 | 23.4786 | 360 | 8.1515 | 6.5289 | 0.7656 | 23.4786 | 360 | 0.5533 | 0.5000 | 0 | 1 | 360 | 0 | 0 | 0 | 0 | 360 | 2.2733 | 0.1839 | 0.0061 | 28.9680 | 360 | 129.6000 | 72 | 24 | 336 |

### Runtime summaries

| asset | frequency | mode | profile_type | count | mean | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 15m | agnostic | multivariate | 4 | 83,838.9496 | 84,062.0903 | 82,187.7431 | 85,043.8745 |
| BTCUSDT | 15m | agnostic | univariate | 4 | 268.5400 | 232.4129 | 229.8061 | 379.5282 |
| BTCUSDT | 15m | conditioned | multivariate | 36 | 168.8865 | 111.0762 | 21.3631 | 459.0056 |
| BTCUSDT | 15m | conditioned | univariate | 36 | 7.4261 | 5.5888 | 1.1748 | 20.4065 |
| BTCUSDT | 1h | agnostic | multivariate | 5 | 5,082.4051 | 5,058.4668 | 4,985.3273 | 5,177.5753 |
| BTCUSDT | 1h | agnostic | univariate | 5 | 22.4446 | 22.5827 | 20.5172 | 24.3037 |
| BTCUSDT | 1h | conditioned | multivariate | 45 | 25.6591 | 17.8638 | 1.1354 | 75.3007 |
| BTCUSDT | 1h | conditioned | univariate | 45 | 1.5330 | 1.2596 | 0.0752 | 5.5357 |
| ETHUSDT | 15m | agnostic | multivariate | 4 | 79,875.6643 | 80,211.0795 | 77,751.2851 | 81,329.2132 |
| ETHUSDT | 15m | agnostic | univariate | 4 | 210.8189 | 210.3075 | 202.7792 | 219.8815 |
| ETHUSDT | 15m | conditioned | multivariate | 40 | 145.6284 | 112.4791 | 0.1658 | 416.0848 |
| ETHUSDT | 15m | conditioned | univariate | 40 | 7.3455 | 6.1321 | 0.0179 | 21.8976 |
| ETHUSDT | 1h | agnostic | multivariate | 5 | 4,852.1487 | 4,852.4434 | 4,825.4746 | 4,891.3060 |
| ETHUSDT | 1h | agnostic | univariate | 5 | 21.4781 | 20.5123 | 19.8349 | 23.5222 |
| ETHUSDT | 1h | conditioned | multivariate | 45 | 17.0461 | 11.5709 | 0.5154 | 57.0349 |
| ETHUSDT | 1h | conditioned | univariate | 45 | 1.1406 | 0.9097 | 0.0544 | 3.6473 |

**Runtime caution:** Some runtime values are unusually large relative to the median. Confirm whether these values are cumulative job time rather than per-experiment runtime before quoting them.

## LoCoMotif motif discovery

- Total motif interval rows: **18,147**
- Assets: BTCUSDT
- Frequencies: 15m
- Evaluation table: 38 rows x 18 columns
- Evaluation rows: **38**
- Evaluation metrics: number_of_motifs, recurrence_count, mean_motif_length, median_motif_length, time_split_stability, cross_regime_overlap, runtime_seconds, l_min
- Runtime rows: **662**
- Failure table: missing or unreadable
- Recorded failure rows: **not available**.

LoCoMotif is reported as a controlled subset experiment rather than a full-scale benchmark.

### Rows by asset, frequency, and mode

| asset | frequency | mode | rows |
| --- | --- | --- | --- |
| BTCUSDT | 15m | conditioned | 18,051 |
| BTCUSDT | 15m | agnostic | 96 |

### Rows by regime method

| regime_method | rows |
| --- | --- |
| quantile_2_rolling_240 | 3,428 |
| quantile_2_rolling_30 | 3,428 |
| quantile_2_rolling_60 | 3,428 |
| quantile_3_rolling_240 | 1,657 |
| quantile_3_rolling_30 | 1,657 |
| quantile_3_rolling_60 | 1,657 |
| quantile_4_rolling_240 | 932 |
| quantile_4_rolling_30 | 932 |
| quantile_4_rolling_60 | 932 |
| none | 96 |

### Rows by regime label

| regime_label | rows |
| --- | --- |
| low_vol | 9,057 |
| high_vol | 7,335 |
| extreme_vol | 1,659 |
| all | 96 |

### Rows by feature set

| feature_set | rows |
| --- | --- |
| log_return,abs_log_return,rolling_volatility_60,rolling_volatility_30,hl_range,volume_zscore,quote_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,spread_proxy,open,high,low,close,volume,pct_return,rolling_volatility_240,absolute_return,squared_return,rolling_vol,range | 18,147 |

### Motif-result status

| status | rows |
| --- | --- |
| success | 18,147 |

### Evaluation grouped summaries

| asset | frequency | mode | number_of_motifs_count | number_of_motifs_mean | number_of_motifs_median | number_of_motifs_min | number_of_motifs_max | recurrence_count_count | recurrence_count_mean | recurrence_count_median | recurrence_count_min | recurrence_count_max | mean_motif_length_count | mean_motif_length_mean | mean_motif_length_median | mean_motif_length_min | mean_motif_length_max | median_motif_length_count | median_motif_length_mean | median_motif_length_median | median_motif_length_min | median_motif_length_max | time_split_stability_count | time_split_stability_mean | time_split_stability_median | time_split_stability_min | time_split_stability_max | cross_regime_overlap_count | cross_regime_overlap_mean | cross_regime_overlap_median | cross_regime_overlap_min | cross_regime_overlap_max | runtime_seconds_count | runtime_seconds_mean | runtime_seconds_median | runtime_seconds_min | runtime_seconds_max | l_min_count | l_min_mean | l_min_median | l_min_min | l_min_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 15m | agnostic | 2 | 4.5000 | 4.5000 | 4 | 5 | 2 | 48 | 48 | 25 | 71 | 2 | 44.9975 | 44.9975 | 22.1549 | 67.8400 | 2 | 41 | 41 | 23 | 59 | 2 | 0.8523 | 0.8523 | 0.7857 | 0.9189 | 0 | not available | not available | not available | not available | 2 | 60.8386 | 60.8386 | 8.2292 | 113.4480 | 2 | 36 | 36 | 24 | 48 |
| BTCUSDT | 15m | conditioned | 36 | 5 | 5 | 5 | 5 | 36 | 501.4167 | 436 | 122 | 1,277 | 36 | 38.6496 | 37.6491 | 23.4358 | 57.8555 | 36 | 36.7083 | 36 | 24 | 52 | 36 | 0.4575 | 0.4015 | 0.1302 | 0.8209 | 0 | not available | not available | not available | not available | 36 | 79.4403 | 73.0252 | 17.8220 | 188.1748 | 36 | 36 | 36 | 24 | 48 |

### Runtime status

| status | rows |
| --- | --- |
| success | 662 |

### Runtime summaries

| asset | frequency | mode | status | count | mean | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 15m | agnostic | success | 2 | 0.9635 | 0.9635 | 0.3292 | 1.5979 |
| BTCUSDT | 15m | conditioned | success | 660 | 0.1080 | 0.0522 | 0.0191 | 0.4081 |

## Matrix Profile vs LoCoMotif

| aspect | matrix_profile | locomotif |
| --- | --- | --- |
| scope completed | 81,987 motif rows | 18,147 motif interval rows; controlled subset |
| assets/frequencies | BTCUSDT, ETHUSDT; 15m, 1h | BTCUSDT; 15m |
| result scale | Pairwise recurring subsequence candidates across configured searches | Motif interval instances grouped into motif sets |
| agnostic support | Yes | Yes |
| conditioned support | Yes | Yes |
| regime labels used | all, extreme_vol, high_vol, low_vol | all, extreme_vol, high_vol, low_vol |
| runtime characteristics | Recorded; grouped runtime statistics available | Recorded; grouped runtime statistics available |
| failure status | No dedicated failure file requested; inspect runtime/status fields | not available recorded failure rows |
| thesis interpretation | Strong baseline for global and regime-conditioned shape recurrence | Flexible-length motif evidence; interpret within completed scope |
| limitations | Raw motif counts are search-dependent and do not imply significance or profitability | Coverage may be capped/subset; counts are not directly comparable with Matrix Profile |

## Recommended figures

| category | selected_filename | recommendation_reason | recommended_use |
| --- | --- | --- | --- |
| locomotif | 04_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_l24_72_intervals.png | conditioned example, high-volatility example | Presentation and thesis evidence |
| locomotif | 04_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_l48_168_intervals.png | conditioned example, high-volatility example | Presentation and thesis evidence |
| locomotif | 04_BTCUSDT_15m_conditioned_quantile_2_rolling_240_low_vol_l24_72_intervals.png | conditioned example, low-volatility example | Presentation and thesis evidence |
| locomotif | 04_BTCUSDT_15m_conditioned_quantile_2_rolling_240_low_vol_l48_168_intervals.png | conditioned example, low-volatility example | Presentation and thesis evidence |
| locomotif | 04_BTCUSDT_15m_agnostic_none_all_l24_72_intervals.png | agnostic example | Presentation and thesis evidence |
| locomotif | 04_BTCUSDT_15m_agnostic_none_all_l48_168_intervals.png | agnostic example | Presentation and thesis evidence |
| matrix_profile | 03_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_close_w128_overlay.png | motif overlay, conditioned example, high-volatility example | Presentation and thesis evidence |
| matrix_profile | 03_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_close_w256_overlay.png | motif overlay, conditioned example, high-volatility example | Presentation and thesis evidence |
| matrix_profile | 03_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_close_w32_overlay.png | motif overlay, conditioned example, high-volatility example | Presentation and thesis evidence |
| matrix_profile | 03_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_close_w64_overlay.png | motif overlay, conditioned example, high-volatility example | Presentation and thesis evidence |
| matrix_profile | 03_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_log_return_w128_overlay.png | motif overlay, conditioned example, high-volatility example | Presentation and thesis evidence |
| matrix_profile | 03_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_log_return_w256_overlay.png | motif overlay, conditioned example, high-volatility example | Presentation and thesis evidence |
| matrix_profile | 03_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_log_return_w32_overlay.png | motif overlay, conditioned example, high-volatility example | Presentation and thesis evidence |
| matrix_profile | 03_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_log_return_w64_overlay.png | motif overlay, conditioned example, high-volatility example | Presentation and thesis evidence |
| matrix_profile | 03_BTCUSDT_15m_conditioned_quantile_2_rolling_240_low_vol_close_w128_profile.png | matrix profile, conditioned example, low-volatility example | Presentation and thesis evidence |
| matrix_profile | 03_BTCUSDT_15m_agnostic_none_all_close_w128_overlay.png | motif overlay, agnostic example | Presentation and thesis evidence |
| regime | 01_BTCUSDT_15m_quantile_2_rolling_240_close_by_regime.png | regime timeline | Presentation and thesis evidence |
| regime | 01_BTCUSDT_15m_quantile_2_rolling_240_transition_heatmap.png | transition evidence | Presentation and thesis evidence |

## File inventory

| method | file_role | path | exists | size_mb | rows | cols | columns | thesis_use | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Quantile | labels | results/regimes/quantile/quantile_regime_labels.parquet | True | 1,019.8830 | 73,097,388 | 11 | timestamp, asset, frequency, regime_method, regime_label, regime_code, rolling_window, n_regimes, volatility_column, volatility_value, thresholds | Primary quantile regime counts and coverage |  |
| Quantile | summary | results/regimes/quantile/quantile_regime_summary.parquet | True | 0.0160 | 432 | 12 | asset, frequency, regime_method, regime_label, observations, share, mean_return, std_return, mean_rolling_vol, median_rolling_vol, min_timestamp, max_timestamp | Regime descriptive statistics |  |
| Quantile | transition matrix | results/regimes/quantile/quantile_transition_matrix.parquet | True | 0.0120 | 1,203 | 7 | asset, frequency, regime_method, from_regime, to_regime, count, probability | Regime persistence and transition evidence |  |
| Quantile | scope file | results/regimes/quantile/BTCUSDT_15m_quantile_regimes.parquet | True | 26.9170 | 1,892,520 | 11 | timestamp, asset, frequency, regime_method, regime_label, regime_code, rolling_window, n_regimes, volatility_column, volatility_value, thresholds | BTCUSDT 15m scope confirmation |  |
| Quantile | scope file | results/regimes/quantile/BTCUSDT_1h_quantile_regimes.parquet | True | 2.6140 | 473,193 | 11 | timestamp, asset, frequency, regime_method, regime_label, regime_code, rolling_window, n_regimes, volatility_column, volatility_value, thresholds | BTCUSDT 1h scope confirmation |  |
| Quantile | scope file | results/regimes/quantile/ETHUSDT_15m_quantile_regimes.parquet | True | 26.9190 | 1,892,520 | 11 | timestamp, asset, frequency, regime_method, regime_label, regime_code, rolling_window, n_regimes, volatility_column, volatility_value, thresholds | ETHUSDT 15m scope confirmation |  |
| Quantile | scope file | results/regimes/quantile/ETHUSDT_1h_quantile_regimes.parquet | True | 2.6150 | 473,193 | 11 | timestamp, asset, frequency, regime_method, regime_label, regime_code, rolling_window, n_regimes, volatility_column, volatility_value, thresholds | ETHUSDT 1h scope confirmation |  |
| Quantile | failure log | results/logs/01_quantile_failures.parquet | True | 0.0040 | 0 | 8 | stage, asset, frequency, context, error_type, error_message, traceback, status | Completeness and failure audit |  |
| HMM | labels | results/regimes/hmm/hmm_regime_labels.parquet | True | 336.3390 | 8,121,932 | 13 | timestamp, asset, frequency, regime_method, regime_label, raw_state, regime_confidence, selected_n_states, feature_set, posterior_state_0, posterior_state_1, posterior_state_2, posterior_state_3 | Primary HMM regime counts and coverage |  |
| HMM | summary | results/regimes/hmm/hmm_regime_summary.parquet | True | 0.0110 | 62 | 12 | asset, frequency, regime_method, regime_label, observations, share, mean_return, std_return, mean_rolling_vol, median_rolling_vol, min_timestamp, max_timestamp | HMM regime descriptive statistics |  |
| HMM | transition matrix | results/regimes/hmm/hmm_transition_matrix.parquet | True | 0.0070 | 205 | 7 | asset, frequency, regime_method, from_regime, to_regime, count, probability | HMM persistence and transition evidence |  |
| HMM | model selection | results/regimes/hmm/hmm_model_selection.parquet | True | 0.0080 | 48 | 10 | asset, frequency, n_states, status, log_likelihood, aic, bic, runtime_seconds, error, selected | Model-state selection evidence |  |
| HMM | persistence metrics | results/regimes/hmm/hmm_persistence_metrics.parquet | True | 0.0060 | 62 | 7 | asset, frequency, regime_method, raw_state, regime_label, self_transition_probability, expected_duration_observations | Expected state duration and persistence |  |
| HMM | feature diagnostics | results/regimes/hmm/hmm_feature_diagnostics.parquet | True | 0.0080 | 93 | 9 | asset, frequency, feature, missing_fraction, unique_values, kept, center, scale, scaler | Input quality audit |  |
| HMM | quantile comparison | results/regimes/hmm/hmm_quantile_comparison.parquet | True | 0.0060 | 144 | 8 | asset, frequency, hmm_method, quantile_method, observations, adjusted_rand_index, normalized_mutual_information, sklearn_available | Cross-method regime validation |  |
| HMM | confusion table | results/regimes/hmm/hmm_quantile_confusion_table.parquet | True | 0.0090 | 1,548 | 7 | asset, frequency, regime_method_hmm, regime_method_quantile, regime_label_hmm, regime_label_quantile, count | Cross-method label agreement |  |
| HMM | scope file | results/regimes/hmm/BTCUSDT_15m_hmm_regimes.parquet | True | 8.6340 | 210,280 | 13 | timestamp, asset, frequency, regime_method, regime_label, raw_state, regime_confidence, selected_n_states, feature_set, posterior_state_0, posterior_state_1, posterior_state_2, posterior_state_3 | BTCUSDT 15m scope confirmation |  |
| HMM | scope file | results/regimes/hmm/BTCUSDT_1h_hmm_regimes.parquet | True | 2.8730 | 52,577 | 13 | timestamp, asset, frequency, regime_method, regime_label, raw_state, regime_confidence, selected_n_states, feature_set, posterior_state_0, posterior_state_1, posterior_state_2, posterior_state_3 | BTCUSDT 1h scope confirmation |  |
| HMM | scope file | results/regimes/hmm/ETHUSDT_15m_hmm_regimes.parquet | True | 8.6530 | 210,280 | 13 | timestamp, asset, frequency, regime_method, regime_label, raw_state, regime_confidence, selected_n_states, feature_set, posterior_state_0, posterior_state_1, posterior_state_2, posterior_state_3 | ETHUSDT 15m scope confirmation |  |
| HMM | scope file | results/regimes/hmm/ETHUSDT_1h_hmm_regimes.parquet | True | 2.3340 | 52,577 | 13 | timestamp, asset, frequency, regime_method, regime_label, raw_state, regime_confidence, selected_n_states, feature_set, posterior_state_0, posterior_state_1, posterior_state_2, posterior_state_3 | ETHUSDT 1h scope confirmation |  |
| HMM | failure log | results/logs/02_hmm_failures.parquet | True | 0.0040 | 0 | 8 | stage, asset, frequency, context, error_type, error_message, traceback, status | Completeness and failure audit |  |
| Matrix Profile | motif results | results/motifs/matrix_profile/matrix_profile_motif_results.parquet | True | 1.7800 | 81,987 | 33 | asset, frequency, mode, regime_method, regime_label, segment_id, feature_set, profile_type, method, window_length, runtime_seconds, n_observations, used_gpu, motif_rank, motif_start_1, motif_start_2, motif_distance, motif_timestamp_1, motif_timestamp_2, motif_end_timestamp_1, motif_end_timestamp_2, exclusion_zone, status, mstump_dimension_row, quantile_regime_1, quantile_regime_2, quantile_regime_method_used, quantile_cross_regime_pair, cross_regime_pair, hmm_regime_1, hmm_regime_2, hmm_regime_method_used, hmm_cross_regime_pair | Primary Matrix Profile motif evidence |  |
| Matrix Profile | evaluation | results/motifs/matrix_profile/matrix_profile_evaluation.parquet | True | 0.0350 | 1,384 | 16 | asset, frequency, method, mode, regime_method, regime_label, window_length, feature_set, number_of_motifs, mean_motif_distance_or_score, median_motif_distance, recurrence_count, runtime_seconds, time_split_stability, cross_regime_overlap, notes | Motif recurrence, stability, and quality metrics |  |
| Matrix Profile | runtime | results/motifs/matrix_profile/matrix_profile_runtime.parquet | True | 0.0080 | 368 | 7 | asset, frequency, mode, regime_method, window_length, profile_type, runtime_seconds | Computational performance |  |
| Matrix Profile | profiles | results/motifs/matrix_profile/matrix_profile_profiles.parquet | True | 0.0010 | 0 | not available |  | Profile availability audit only | Metadata inspected without loading the full profile table |
| LoCoMotif | motif results | results/motifs/locomotif/locomotif_motif_results.parquet | True | 0.2220 | 18,147 | 27 | asset, frequency, mode, regime_method, regime_label, segment_id, feature_set, method, motif_set_rank, motif_instance_id, role, motif_start, motif_end, motif_length, motif_start_timestamp, motif_end_timestamp, motif_score, motif_set_size, l_min, l_max, rho, nb, overlap, warping, runtime_seconds, n_observations, status | Primary LoCoMotif interval evidence |  |
| LoCoMotif | evaluation | results/motifs/locomotif/locomotif_evaluation.parquet | True | 0.0120 | 38 | 18 | asset, frequency, method, mode, regime_method, regime_label, l_min, l_max, rho, number_of_motifs, mean_motif_distance_or_score, recurrence_count, mean_motif_length, median_motif_length, runtime_seconds, time_split_stability, cross_regime_overlap, notes | Motif recurrence, length, and stability metrics |  |
| LoCoMotif | runtime | results/motifs/locomotif/locomotif_runtime.parquet | True | 0.0180 | 662 | 18 | module, raw_motif_sets_count, runtime_seconds, rows, asset, frequency, mode, regime_method, regime_label, segment_id, feature_set, l_min, l_max, rho, nb, overlap, warping, status | Computational performance and status |  |
| LoCoMotif | failure log | results/motifs/locomotif/04_locomotif_failures.parquet | False | not available | not available | not available |  | Completeness and failure audit | expected file missing |

## Caveats and missing inputs

- Missing: `results/motifs/locomotif/04_locomotif_failures.parquet`

- Motif discovery demonstrates recurring shape similarity; it does not establish profitability, causality, or predictive power.
- Raw row counts depend on configured windows, feature sets, regimes, segments, and algorithm output structure.
- Algorithm row counts are not directly comparable unless the search spaces and output units are harmonized.
