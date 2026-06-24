# Notebook 01 Study Notes: Quantile Volatility Regime Detection

## Scope and evidence status

This report explains the implementation and available results of:

`notebooks/01_quantile_regime_detection.ipynb`

The repository contains the source notebook, source modules, feature data, a full-run HPC configuration, and a completed local smoke-test output set. The following expected HPC artifacts were **not found**:

- `notebooks/executed_01_quantile_regime_detection.ipynb`
- `results/configs/01_quantile_regime_detection_config_snapshot.json`
- unsuffixed full-run quantile label, summary, transition, figure, and log files

The verified numerical results in this report therefore come from the files ending in `_LOCAL_SMOKE_TEST`. The HPC configuration is documented as the intended full-run design, not as a completed result.

Inspection date: 2026-06-24.

## 1. Conceptual explanation

### Returns

A financial return measures the change in an asset price from one observation to the next. The feature files contain `log_return`, which is normally computed as

\[
r_t=\log(P_t)-\log(P_{t-1}),
\]

where \(P_t\) is the closing price at time \(t\). Returns are preferable to raw prices for volatility analysis because their scale is more comparable through time and they directly measure relative price movement.

### Rolling volatility

Rolling volatility is the standard deviation of recent returns over a fixed lookback window:

\[
\sigma_{t,w}=\operatorname{sd}(r_{t-w+1},\ldots,r_t).
\]

For example, `rolling_volatility_60` is the standard deviation of returns over a 60-observation window. At 1h frequency this corresponds approximately to the previous 60 hours; at 15m frequency it corresponds to the previous 15 hours.

The repository feature files contain `rolling_volatility_30`, `rolling_volatility_60`, and `rolling_volatility_240`.

### Quantile thresholds and bins

For \(K\) regimes, the implementation computes \(K-1\) empirical quantiles of the selected volatility series:

\[
q_j=Q_{\sigma}(j/K), \qquad j=1,\ldots,K-1.
\]

These thresholds divide the observed volatility distribution into approximately equal-sized bins. With three regimes, the thresholds are the one-third and two-thirds quantiles. A volatility observation below the first threshold is assigned to the lowest regime, an observation between thresholds to the middle regime, and an observation above the second threshold to the highest regime.

The labels are semantic:

| Regime count | Ordered labels |
| --- | --- |
| 2 | `low_vol`, `high_vol` |
| 3 | `low_vol`, `medium_vol`, `high_vol` |
| 4 | `low_vol`, `medium_vol`, `high_vol`, `extreme_vol` |

The labels describe **relative volatility within the fitted sample**. They do not imply an absolute economic definition of calm or crisis.

### Why this is useful before motif discovery

Motif discovery searches for recurring subsequences. Financial patterns may behave differently in calm and stressed markets, so pooling all observations can hide regime-specific recurrence or make common low-volatility behavior dominate the result. Quantile labels provide an interpretable conditioning variable:

- regime-agnostic analysis searches the full time series;
- regime-conditioned analysis searches only continuous periods with the same label;
- comparing the two tests whether motifs are stable across market conditions or specific to a volatility state.

## 2. How the actual code works

### Notebook structure

Notebook 01 has seven cells:

1. title and purpose;
2. execution-mode explanation;
3. environment, paths, configuration, logging, and snapshot setup;
4. feature-discovery explanation;
5. feature-file discovery and inventory saving;
6. processing explanation;
7. the complete asset/frequency processing loop and output saving.

The notebook delegates most calculations to modules under `src/`.

### Configuration loading and execution mode

The notebook begins with:

```python
EXECUTION_MODE = "auto"
STAGE_NAME = "01_quantile_regime_detection"
```

`find_workflow_root()` searches the current directory and its parents for `src/hpc_config.py`, with the local Windows path as a fallback. The notebook then:

```python
MODE = detect_execution_mode(EXECUTION_MODE, PROJECT_ROOT)
CONFIG = load_workflow_config(MODE, WORKFLOW_ROOT)
PATHS = build_workflow_paths(WORKFLOW_ROOT, PROJECT_ROOT)
SUFFIX = output_suffix(CONFIG)
```

On Windows, `auto` resolves to `local`; on Linux or a host with HPC markers such as SLURM variables, it resolves to `hpc`. Local mode loads `run_configs/local_smoke_test.yaml` and adds `_LOCAL_SMOKE_TEST` to output names. HPC mode loads `run_configs/hpc_full_run.yaml` and uses no suffix.

### Output directories, logger, and config snapshot

`ensure_workflow_dirs(PATHS)` creates the standard result directories. `setup_stage_logger()` writes a stage log to `results/logs/`. `save_config_snapshot()` writes the merged configuration and environment details, including Python version, packages, memory, GPU detection, and resolved paths.

Verified local snapshot:

`results/configs/01_quantile_regime_detection_config_snapshot_LOCAL_SMOKE_TEST.json`

The expected unsuffixed HPC snapshot was not found.

### Data discovery and filtering

`discover_feature_files()` searches:

```text
final_dataset/features/**/*_features_*.parquet
```

It parses the asset and frequency from each filename, then filters by:

- `allowed_assets`;
- `allowed_frequencies`;
- local-only frequency overrides;
- local maximum asset/file counts;
- HPC `max_files`, if configured.

The discovered inventory is saved as both CSV and Parquet.

The full-run configuration selects BTCUSDT and ETHUSDT at 15m and 1h. All four corresponding feature files exist:

| Asset | Frequency | Rows | Date range |
| --- | ---: | ---: | --- |
| BTCUSDT | 15m | 210,280 | 2020-01-01 00:00 to 2025-12-31 23:45 UTC |
| BTCUSDT | 1h | 52,577 | 2020-01-01 00:00 to 2025-12-31 23:00 UTC |
| ETHUSDT | 15m | 210,280 | 2020-01-01 00:00 to 2025-12-31 23:45 UTC |
| ETHUSDT | 1h | 52,577 | 2020-01-01 00:00 to 2025-12-31 23:00 UTC |

The completed local run selected only the two 1h files and retained the first 1,500 rows between 2021-01-01 and 2021-03-31. Both processed datasets therefore end at 2021-03-04 12:00 UTC.

### Feature-file loading and timestamps

`load_feature_file()` reads each Parquet file and searches for a timestamp column in this order:

```python
["timestamp", "datetime", "date", "open_time", "time"]
```

Numeric timestamps are interpreted using their magnitude as nanoseconds, microseconds, milliseconds, or seconds. Other values are parsed directly. All timestamps are converted to UTC. Rows with invalid timestamps are removed, then data are sorted and duplicate timestamps are dropped.

### Core return and volatility features

`ensure_core_features()` computes missing core features:

- `log_return` from log closing prices;
- `pct_return`;
- absolute and squared returns;
- `rolling_vol`;
- `rolling_volatility_<rolling_window>`;
- high-low range and spread proxies.

It does not overwrite feature columns that already exist.

### Volatility-column selection

The actual selection rule is:

```python
vol_col = choose_volatility_column(
    prepared,
    config.get("quantile", {}).get("volatility_columns"),
)
```

The merged default preference order is:

1. `rolling_volatility_60`
2. `rolling_volatility_30`
3. `rolling_volatility_240`
4. `rolling_vol`
5. `realized_vol`

#### Important implementation finding

The selected column is not explicitly matched to the loop's `rolling_window`. All four configured full-run feature files contain `rolling_volatility_60`, so the current implementation will select that column first for every requested window. Consequently, a method named `quantile_3_rolling_30` or `quantile_4_rolling_240` would still be calculated from `rolling_volatility_60` unless the preference logic is changed or the 60-period column is absent.

This is a material reproducibility issue. The method name records the requested window, while the `volatility_column` field records the series actually used. Thesis analysis should treat `volatility_column` as authoritative and should not claim that all 30/60/240 variants were computed from their matching volatility columns under the current code.

### Missing-value handling and minimum observations

The selected volatility series is converted to numeric, infinite values become missing, and missing values are filled forward and backward:

```python
values = values.ffill().bfill()
```

The method requires at least:

```python
max(30, n_regimes * 10)
```

non-missing volatility observations. This is a quantile-stage check. The `hmm.min_rows` parameter belongs to Notebook 02 and is not used by Notebook 01.

### Threshold calculation

For each configured regime count:

```python
quantile_points = [i / n_regimes for i in range(1, n_regimes)]
thresholds = values.quantile(quantile_points).to_numpy(dtype=float)
```

Thresholds are calculated separately for each asset, frequency, and method run. They are in-sample empirical quantiles of the entire processed series.

### Label assignment

Labels are assigned using:

```python
codes = np.searchsorted(
    thresholds,
    values.to_numpy(dtype=float),
    side="right",
)
```

`side="right"` means a value exactly equal to a threshold enters the higher bin. Codes are converted to semantic labels in increasing volatility order.

The method identifier is:

```python
regime_method = f"quantile_{n_regimes}_rolling_{rolling_window}"
```

Each label row contains:

- timestamp;
- asset and frequency;
- `regime_method`;
- `regime_label` and integer `regime_code`;
- requested `rolling_window`;
- `n_regimes`;
- actual `volatility_column`;
- actual `volatility_value`;
- comma-separated thresholds.

### Summary and transition tables

`summarize_regimes()` groups rows by label and calculates:

- observations and sample share;
- mean and standard deviation of `log_return`;
- mean and median selected volatility;
- first and last timestamp at which the label appears.

`transition_table()` counts adjacent label pairs and divides each count by the total transitions leaving the source label. These are empirical one-step transition frequencies. They are descriptive and are not an estimated Markov-switching model.

### Per-asset and consolidated output tables

For each asset/frequency, the notebook saves a label table. It then concatenates all processed assets into:

- `quantile_regime_labels...`
- `quantile_regime_summary...`
- `quantile_transition_matrix...`
- `01_quantile_dataset_summary...`

Every table is attempted in CSV and Parquet format.

### Figure creation

For each asset/frequency/method, the notebook creates up to four figures:

1. closing price colored by regime;
2. selected volatility colored by regime;
3. regime distribution;
4. transition heatmap.

Figure generation stops when `max_figures_per_notebook` is reached. Time-series plots are capped by `max_points_per_figure`.

The available local run created eight figures: four for each asset.

### Failure handling and logs

Each asset/frequency is processed in a `try/except` block. A failure is logged and appended to a structured failure table without stopping other datasets. The verified local failure CSV contains only its header, so zero Notebook 01 failures were recorded.

The stage log records successful processing of BTCUSDT 1h and ETHUSDT 1h on 2026-06-09.

## 3. Exact parameters

| Parameter | Full HPC value | Verified local value | Meaning and importance |
| --- | --- | --- | --- |
| Allowed assets | BTCUSDT, ETHUSDT | unrestricted before local caps; first two discovered assets were BTCUSDT, ETHUSDT | Defines the markets analyzed. |
| Allowed frequencies | 15m, 1h | local override: 1h | Controls temporal resolution and the economic duration represented by each window. |
| Full input files implied by filters | 4 | 2 | Asset-frequency datasets entering Notebook 01. |
| Local maximum assets | inherited 2 | 2 | Restricts smoke-test scope. |
| Local maximum files | inherited 2 | 2 | Restricts smoke-test scope. |
| Local date range | not applicable | 2021-01-01 to 2021-03-31 | Limits smoke-test data. |
| Local maximum rows | not applicable | 1,500 | The row cap is reached before the configured end date. |
| Rolling windows | 30, 60, 240 | 60 | Requested lookback variants. See the implementation mismatch described above. |
| Regime counts | 2, 3, 4 | 3 | Requested levels of regime granularity. |
| Default rolling window | 60 | 60 | Used for preparation and plotting outside each quantile-method calculation. |
| Default regime count | 3 | 3 | Used by downstream local mode to select a quantile method. |
| Volatility preference | 60, 30, 240, `rolling_vol`, `realized_vol` | same merged default | Determines the actual volatility column; currently causes 60 to dominate. |
| Quantile minimum observations | `max(30, 10 × regimes)` | 30 for three regimes | Prevents quantiles from being estimated from very small samples. |
| HMM minimum rows | 500 | 250 | Not used by Notebook 01; relevant only to Notebook 02. |
| Maximum points per figure | 10,000 | 3,000 | Limits plotting cost, not regime estimation. |
| Maximum figures | 200 | 24 | Caps generated figures. |
| Output suffix | none | `_LOCAL_SMOKE_TEST` | Separates smoke-test outputs from HPC outputs. |
| Seed | 20260609 | 20260609 | Stored for workflow reproducibility; quantile assignment itself is deterministic. |

## 4. Why use 2, 3, and 4 regimes?

Two regimes provide the simplest calm-versus-stress partition. This binary specification is comparatively stable, produces larger state samples, and is useful as a baseline. Its limitation is that ordinary and extreme high-volatility periods are pooled.

Three regimes provide a low/medium/high structure. This is often the most interpretable compromise because it separates normal intermediate conditions from both tails without excessive fragmentation.

Four regimes add an `extreme_vol` state. This can isolate tail stress that would otherwise be mixed with moderately high volatility, but each regime receives fewer observations and the label sequence can split into more and shorter runs.

This trade-off matters directly for motif discovery. Matrix Profile and LoCoMotif require enough observations to form candidate subsequences. The HPC configuration requires continuous conditioned segments of at least 512 observations for both methods. Increasing the regime count generally:

- increases state granularity;
- reduces observations per state;
- increases the number of regime boundaries;
- shortens continuous same-regime segments;
- reduces the number of segments long enough for motif mining.

Fewer regimes therefore support more stable and longer samples, while more regimes support more specific market-state interpretation. Testing 2, 3, and 4 regimes is a sensitivity analysis over this bias-granularity trade-off.

## 5. Actual output inventory

The complete machine-readable inventory is:

`reports/01_quantile_regime_detection_output_inventory.csv`

The primary verified Notebook 01 output set consists of:

| Output group | Files | Verified content |
| --- | ---: | --- |
| Config snapshots | 1 local; HPC snapshot absent | Merged configuration and environment |
| Input inventory | CSV + Parquet | Two 1h crypto feature files |
| Dataset summary | CSV + Parquet | Two rows |
| Per-asset labels | 2 CSV + 2 Parquet | 1,500 labels per asset |
| Consolidated labels | CSV + Parquet | 3,000 rows |
| Regime summary | CSV + Parquet | Six rows |
| Transition table | CSV + Parquet | Fifteen rows |
| Figures | 8 PNG files | Four per asset |
| Stage log | 1 log | Two successful datasets |
| Failure table | CSV + Parquet | Zero failures |

The CSV inventory also records the downstream Matrix Profile files that consume the labels.

## 6. What the verified quantile outputs show

### Dataset coverage

The local outputs cover:

- assets: BTCUSDT and ETHUSDT;
- frequency: 1h only;
- period: 2021-01-01 00:00 through 2021-03-04 12:00 UTC;
- rows: 1,500 per asset;
- method: `quantile_3_rolling_60`;
- actual volatility column: `rolling_volatility_60`.

No verified full-run outputs for 15m, 2 regimes, 4 regimes, 30 windows, or 240 windows were found.

### Thresholds

| Asset | Lower threshold | Upper threshold | Interpretation |
| --- | ---: | ---: | --- |
| BTCUSDT | 0.00921575899003 | 0.0135375541062 | Low below the first threshold; medium between; high at or above the second |
| ETHUSDT | 0.01127940116 | 0.0153860734094 | Same rule |

ETHUSDT has higher thresholds than BTCUSDT in this local sample, indicating that its 60-hour return volatility distribution was shifted upward.

### Regime distributions and moments

Because empirical terciles were used and each dataset contains 1,500 observations, every asset has exactly 500 observations in each regime.

| Asset | Regime | Rows | Mean return | Return standard deviation | Mean rolling volatility | Median rolling volatility |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | low | 500 | 0.000824 | 0.007841 | 0.007773 | 0.007855 |
| BTCUSDT | medium | 500 | -0.000569 | 0.012113 | 0.011224 | 0.011261 |
| BTCUSDT | high | 500 | 0.000814 | 0.016415 | 0.016760 | 0.015776 |
| ETHUSDT | low | 500 | 0.000923 | 0.009686 | 0.008924 | 0.008714 |
| ETHUSDT | medium | 500 | 0.000369 | 0.013768 | 0.013318 | 0.013321 |
| ETHUSDT | high | 500 | 0.000211 | 0.019718 | 0.019883 | 0.019168 |

The ordering of realized return dispersion and mean rolling volatility is consistent with the semantic labels. Mean returns do not increase monotonically with volatility, which is expected: the method classifies the magnitude of variation, not return direction.

### Transition behavior

The largest transition probability for every state is remaining in the same state:

| Asset | Low → low | Medium → medium | High → high |
| --- | ---: | ---: | ---: |
| BTCUSDT | 0.966 | 0.948 | 0.978 |
| ETHUSDT | 0.966 | 0.942 | 0.974 |

This persistence is consistent with volatility clustering. Most cross-state movements occur between adjacent categories. BTCUSDT has one direct low-to-high transition; no direct high-to-low transition appears in the saved table. ETHUSDT has no direct low/high jump in either direction. These observations describe this limited sample and should not be generalized to the full 2020–2025 data without an HPC run.

### Example regime periods and fragmentation

Contiguous same-label periods were reconstructed from the saved labels:

| Asset | Regime | Number of runs | Median run length | Maximum run length | Runs ≥96 | Runs ≥512 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | low | 17 | 5 | 228 | 2 | 0 |
| BTCUSDT | medium | 27 | 5 | 108 | 1 | 0 |
| BTCUSDT | high | 11 | 53 | 100 | 1 | 0 |
| ETHUSDT | low | 17 | 6 | 276 | 1 | 0 |
| ETHUSDT | medium | 30 | 7 | 104 | 1 | 0 |
| ETHUSDT | high | 13 | 12 | 180 | 2 | 0 |

Examples of the longest runs include:

- BTCUSDT low volatility: 2021-02-13 01:00 to 2021-02-22 12:00 UTC, 228 observations;
- BTCUSDT medium volatility: 2021-01-15 10:00 to 2021-01-19 21:00 UTC, 108 observations;
- ETHUSDT low volatility: 2021-02-11 00:00 to 2021-02-22 12:00 UTC, 276 observations;
- ETHUSDT high volatility: 2021-01-08 00:00 to 2021-01-15 11:00 UTC, 180 observations.

No local segment reaches the HPC minimum of 512 observations. The local Matrix Profile configuration instead uses a minimum of 96, which allows a small number of conditioned runs to be analyzed.

### Figure interpretation

The close-by-regime plots show volatility states distributed across both rising and falling price periods. This is correct because the regime is based on return dispersion rather than price level or direction. The transition heatmaps are strongly diagonal, visually confirming the high one-step persistence reported in the transition table.

## 7. Use in Notebook 03 Matrix Profile

Notebook 03 loads saved labels with:

```python
quantile_labels = load_regime_labels(
    WORKFLOW_ROOT, "quantile", SUFFIX
)
```

For each asset and frequency, it first runs an agnostic analysis on the complete series:

```python
run_mp_slice(
    df, asset, frequency, windows,
    "agnostic", "none", "all", "full_series"
)
```

It then loops over available `regime_method` values, merges labels by asset, frequency, and timestamp, and calls `continuous_segments()`. This function assigns a `segment_id` only to uninterrupted same-regime runs meeting `min_segment_length`. Disconnected periods with the same label are not concatenated.

The key fields mean:

- `regime_method`: the exact quantile specification, such as `quantile_3_rolling_60`;
- `regime_label`: the state attached to the continuous slice;
- `segment_id`: a unique identifier such as `low_vol_0001`;
- `mode`: `agnostic` or `conditioned`.

Each qualifying segment is passed independently to univariate and multivariate Matrix Profile calculations. The resulting motif rows retain the method, label, and segment identifiers. Notebook 03 also annotates motif endpoints with their quantile regimes and a `quantile_cross_regime_pair` flag. This supports comparison of:

- motif recurrence in the full series;
- motif recurrence inside a single volatility state;
- motif pairs whose occurrences fall in different states.

### Verified downstream local results

The available Matrix Profile result table contains 137 motif rows:

- 48 agnostic rows;
- 89 conditioned rows using `quantile_3_rolling_60`.

Conditioned rows by asset and label:

| Asset | Low | Medium | High | Total |
| --- | ---: | ---: | ---: | ---: |
| BTCUSDT | 30 | 8 | 5 | 43 |
| ETHUSDT | 20 | 4 | 22 | 46 |
| Total | 50 | 12 | 27 | 89 |

These are result-table rows, not necessarily 89 statistically independent motif families. Counts depend on channels, windows, profile type, qualifying segments, and `top_k`.

The proposed full-run counts of 15,618, 6,885, and 4,559 rows for various 2/3/4-regime methods were **not included** because no current result file contains those methods or verifies those values.

## 8. Limitations

1. **Relative rather than absolute states.** A `high_vol` observation is high relative to the sample used to estimate thresholds. The same numeric volatility can receive a different label in another asset, period, or frequency.

2. **In-sample, data-dependent thresholds.** Thresholds are estimated from the full processed sample. For predictive or real-time analysis, this creates look-ahead information because future observations influence earlier labels. A causal extension would estimate thresholds using an expanding or rolling historical calibration window.

3. **Forced partitioning.** Quantiles create bins even if the volatility distribution changes smoothly and has no distinct state boundaries.

4. **Boundary noise.** Small fluctuations around a threshold can cause rapid label changes. The implementation applies no hysteresis, smoothing, or minimum-duration rule during label creation.

5. **Window sensitivity.** Volatility and labels depend on the lookback length. Short windows react faster but are noisier; long windows are smoother but respond more slowly.

6. **Current window-selection mismatch.** Requested 30/60/240 method names do not guarantee matching volatility columns. With the current feature files, `rolling_volatility_60` is preferred for every method.

7. **Missing-value filling.** Forward and backward filling can extend nearby volatility estimates into missing regions. Backward filling is non-causal at the start of a series.

8. **Fragmentation.** More regimes and noisy boundaries shorten continuous segments, reducing the number of subsequences available to Matrix Profile and LoCoMotif.

9. **No transition model.** Transition probabilities are descriptive counts. Unlike a Hidden Markov Model, this approach does not estimate latent states, transition dynamics, or state uncertainty.

10. **Equal counts do not imply equal durations or economic importance.** Quantile bins target similar observation counts, while continuous run lengths can differ substantially.

11. **Local results are not thesis-scale results.** The saved outputs cover only 3,000 hourly observations across two assets and one method. They validate the pipeline but do not establish full-run conclusions.

Despite these limitations, the method is transparent, deterministic, easy to audit, computationally inexpensive, and reproducible when the sample and selected volatility column are fixed.

## 9. Candidate citations

No matching bibliography entries or DOI records were found in the inspected repository text files. DOI fields below are therefore marked for verification before final submission.

### Why each reference is relevant

- Mandelbrot (1963) is a foundational discussion of heavy-tailed speculative price variation.
- Engle (1982) formalizes time-varying conditional variance through ARCH.
- Cont (2001) surveys stylized facts including volatility clustering and heavy tails.
- Hamilton (1989) provides a foundational probabilistic regime-switching framework.
- Ang and Timmermann (2012) review regime changes in financial markets.
- Yeh et al. (2016) introduce the Matrix Profile framework used downstream for motif discovery.

### BibTeX-style entries

```bibtex
@article{mandelbrot1963variation,
  author  = {Mandelbrot, Benoit},
  title   = {The Variation of Certain Speculative Prices},
  journal = {The Journal of Business},
  year    = {1963},
  volume  = {36},
  number  = {4},
  pages   = {394--419},
  doi     = {check before final submission}
}

@article{engle1982arch,
  author  = {Engle, Robert F.},
  title   = {Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation},
  journal = {Econometrica},
  year    = {1982},
  volume  = {50},
  number  = {4},
  pages   = {987--1007},
  doi     = {check before final submission}
}

@article{hamilton1989regime,
  author  = {Hamilton, James D.},
  title   = {A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle},
  journal = {Econometrica},
  year    = {1989},
  volume  = {57},
  number  = {2},
  pages   = {357--384},
  doi     = {check before final submission}
}

@article{cont2001stylized,
  author  = {Cont, Rama},
  title   = {Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues},
  journal = {Quantitative Finance},
  year    = {2001},
  volume  = {1},
  number  = {2},
  pages   = {223--236},
  doi     = {check before final submission}
}

@article{ang2012regime,
  author  = {Ang, Andrew and Timmermann, Allan},
  title   = {Regime Changes and Financial Markets},
  journal = {The Review of Financial Studies},
  year    = {2012},
  volume  = {25},
  number  = {4},
  pages   = {1019--1042},
  doi     = {check before final submission}
}

@inproceedings{yeh2016matrixprofile,
  author    = {Yeh, Chin-Chia Michael and Zhu, Yan and Ulanova, Liudmila and Begum, Nurjahan and Ding, Yifei and Dau, Hoang Anh and Silva, Diego and Mueen, Abdullah and Keogh, Eamonn},
  title     = {Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifying View That Includes Motifs, Discords and Shapelets},
  booktitle = {2016 IEEE 16th International Conference on Data Mining},
  year      = {2016},
  pages     = {1317--1322},
  doi       = {check before final submission}
}
```

## 10. Quantile-Based Volatility Regime Detection

Quantile-based volatility regimes were employed as a transparent baseline for identifying market conditions before motif discovery. Financial return series exhibit time-varying volatility and volatility clustering, implying that recurring temporal patterns may differ between relatively calm and stressed periods. For each asset-frequency series, volatility was represented by a rolling standard deviation of log returns. Empirical quantile thresholds of the selected volatility series were then used to assign each timestamp to an ordered regime. Two-regime specifications distinguish low- from high-volatility conditions, three-regime specifications introduce an intermediate state, and four-regime specifications isolate an additional extreme-volatility tail. This range provides a sensitivity analysis between parsimonious, stable partitions and more granular but potentially fragmented state definitions.

Rolling windows of 30, 60, and 240 observations were configured to represent short-, intermediate-, and longer-horizon volatility conditions. Shorter windows are intended to react more quickly to changes in market variability, whereas longer windows provide smoother state estimates. The resulting method identifiers combine the regime count and requested rolling window, for example `quantile_3_rolling_60`. Each observation is stored with its timestamp, regime label, regime code, selected volatility value, and estimated quantile thresholds, thereby preserving an auditable mapping between the input series and the assigned state.

The saved labels support regime-conditioned motif discovery in subsequent Matrix Profile and LoCoMotif stages. Regime-agnostic analyses operate on the complete series, whereas conditioned analyses operate on uninterrupted same-regime segments that satisfy a minimum-length requirement. This design avoids concatenating temporally disconnected periods and permits direct comparison of motif recurrence across volatility states. The regime count is therefore not only a descriptive choice: it determines the number and length of continuous segments available to downstream algorithms. Coarser partitions provide larger samples and longer segments, while finer partitions may identify more specific stress conditions at the cost of reduced motif sample size.

The current implementation requires one qualification before this text is used as a final methodological claim. Although the configuration requests 30-, 60-, and 240-observation variants, the volatility-selection function prioritizes `rolling_volatility_60` whenever it is present. The current feature files contain this column, so the code must be corrected or the selected `volatility_column` must be verified before asserting that all three method identifiers represent distinct matching lookback series.

### Results Interpretation

The available local smoke-test outputs confirm that the pipeline produces timestamp-aligned regime labels, balanced quantile groups, descriptive regime summaries, persistent transition matrices, figures, and downstream conditioned motif results. For BTCUSDT and ETHUSDT at 1h frequency, the three-regime method assigned 500 of 1,500 observations to each state. Return dispersion and average rolling volatility increased from low to medium to high regimes, while one-step self-transition probabilities ranged from approximately 0.942 to 0.978. These results show that the quantile procedure generated economically ordered and temporally persistent labels in the test sample.

The same labels were used in the available Matrix Profile output, which contains 48 agnostic motif rows and 89 rows conditioned on `quantile_3_rolling_60`. The conditioned results demonstrate that motif extraction can be restricted to continuous volatility states and compared with the full-series baseline. However, these counts are smoke-test diagnostics rather than final thesis evidence. Full interpretation requires verified unsuffixed HPC outputs across both frequencies, all requested regime counts, and correctly matched rolling-volatility columns.

## Recommended next analysis steps

1. Change volatility selection so that `rolling_window=30`, `60`, or `240` explicitly selects `rolling_volatility_<window>`.
2. Add an assertion that the requested window equals the suffix of `volatility_column`.
3. Execute Notebook 01 in HPC mode and preserve `executed_01_quantile_regime_detection.ipynb`.
4. Recompute segment-length distributions for every asset, frequency, window, and regime count before launching expensive motif runs.
5. Verify that enough segments exceed the configured 512-observation minimum; otherwise justify a frequency-specific minimum.
6. Consider an out-of-sample or expanding-window threshold variant to avoid full-sample look-ahead.
7. Compare quantile labels with HMM labels only after both methods are available for the same timestamps and scope.

## Source checklist

| Source | Status |
| --- | --- |
| Source Notebook 01 | found and parsed |
| Executed Notebook 01 | not found |
| HPC full-run YAML | found |
| Local smoke-test YAML | found |
| Unsuffixed HPC config snapshot | not found |
| Local config snapshot | found |
| Quantile label CSV/Parquet | found |
| Quantile summary CSV/Parquet | found |
| Quantile transition CSV/Parquet | found |
| Notebook 01 figures | eight found |
| Notebook 01 log | found |
| Notebook 01 failure table | found; zero failures |
| Full-run feature files | four found |
| Matrix Profile source notebook | found and parsed |
| Matrix Profile local result table | found and summarized |
| Full 2/3/4-regime Matrix Profile counts supplied in the request | not verifiable from current files |

## Generated report artifacts

- `reports/01_quantile_regime_detection_study_notes.md`
- `reports/01_quantile_regime_detection_output_inventory.csv`
- `reports/01_quantile_regime_detection_summary_tables.csv`
