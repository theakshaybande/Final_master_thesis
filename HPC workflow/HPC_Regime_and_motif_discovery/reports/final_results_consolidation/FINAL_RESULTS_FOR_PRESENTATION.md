# Final Results for Presentation

Generated: 2026-06-25T10:29:41.919199+00:00

## Slide 1: Research question

- Do recurring cryptocurrency price/volatility patterns differ across market regimes?
- Compare regime-agnostic discovery with volatility-regime-conditioned discovery.
- Use Quantile and HMM labels for regime context.
- Compare fixed-length Matrix Profile motifs with flexible-length LoCoMotif intervals.

**Recommended figure:** Use a concise study-design diagram if one is available.

**Speaker note:** The analysis asks whether conditioning motif discovery on market state changes the recurring structures that are observed. The empirical claims concern recurrence and regime context, not trading profitability.

## Slide 2: Pipeline

- Load precomputed feature and regime result files.
- Validate Quantile and HMM coverage, labels, transitions, and failures.
- Run agnostic and conditioned motif discovery.
- Evaluate recurrence, stability, overlap, runtime, and failure status.
- Consolidate tables and representative figures without rerunning notebooks.

**Recommended figure:** Use a pipeline diagram from the thesis, if available.

**Speaker note:** All numbers in this presentation come from saved result files. The consolidation step is read-only for original results and preserves missing-file evidence.

## Slide 3: Regime detection validation

- Quantile label rows: 73,097,388.
- HMM label rows: 8,121,932.
- Quantile failures: 0; HMM failures: 0.
- Quantile labels: extreme_vol, high_vol, low_vol, medium_vol.
- The quantile regime outputs store `rolling_volatility_60` as the actual volatility column across all quantile method identifiers. Therefore, quantile regimes should be interpreted as 60-period rolling-volatility regimes with different regime-count granularities rather than as separate 30/60/240 volatility-horizon experiments.

**Recommended figure:** `selected_final_figures/01_BTCUSDT_15m_quantile_2_rolling_240_close_by_regime.png`

**Speaker note:** Regime validation establishes the conditioning variable used by the motif experiments. Missing files or failures should be reported as scope limitations rather than inferred away.

## Slide 4: Matrix Profile setup

- Assets/frequencies: BTCUSDT, ETHUSDT; 15m, 1h.
- Agnostic mode searches the full eligible series.
- Conditioned mode repeats search inside regime-specific segments.
- Configured window lengths, feature sets, and profile types define the search space.
- GPU audit rows: true=0, false=81,987.

**Recommended figure:** `selected_final_figures/03_BTCUSDT_15m_conditioned_quantile_2_rolling_240_low_vol_close_w128_profile.png`

**Speaker note:** Matrix Profile measures subsequence shape similarity at configured fixed windows. Conditioned searches create more search units, so raw counts require normalization before comparative interpretation.

## Slide 5: Matrix Profile key results

- Total motif rows: 81,987.
- Agnostic rows: 720.
- Conditioned rows: 81,267.
- Evaluation table: 1,384 rows x 16 columns.
- Recurring shapes were identified, but motif count is not a significance test.

**Recommended figure:** `selected_final_figures/03_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_close_w128_overlay.png`

**Speaker note:** The central result is successful recurrence discovery in global and, where present, regime-conditioned searches. Count differences partly reflect repeated searches over regimes, methods, and segments.

## Slide 6: Matrix Profile figure evidence

- Overlay plots show paired subsequences in the original context.
- Profile plots show low-distance candidate locations.
- Use both agnostic and conditioned examples.
- Include low-, high-, or extreme-volatility examples when available.
- Interpret visual similarity jointly with evaluation metrics.

**Recommended figure:** `selected_final_figures/03_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_close_w128_overlay.png`

**Speaker note:** Representative figures should demonstrate what the algorithm calls similar without implying economic value. The selected folder contains a compact set chosen for mode, regime, and plot-type diversity.

## Slide 7: LoCoMotif setup

- Assets/frequencies: BTCUSDT; 15m.
- LoCoMotif searches flexible-length motif intervals.
- Agnostic and conditioned support is reported from completed outputs.
- Runtime rows: 662.
- LoCoMotif is reported as a controlled subset experiment rather than a full-scale benchmark.

**Recommended figure:** `selected_final_figures/04_BTCUSDT_15m_agnostic_none_all_l24_72_intervals.png`

**Speaker note:** LoCoMotif provides a complementary representation because interval lengths can vary. Its completed scope must be stated explicitly, especially if computation was capped.

## Slide 8: LoCoMotif key results

- Motif interval rows: 18,147.
- Evaluation rows: 38.
- Failure rows: not available.
- Regime labels represented: all, extreme_vol, high_vol, low_vol.
- Use recurrence, motif length, and stability metrics where populated.

**Recommended figure:** `selected_final_figures/04_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_l24_72_intervals.png`

**Speaker note:** LoCoMotif results should be interpreted within their completed computational scope. Empty outputs and failure logs are substantive audit evidence, not values to replace with assumptions.

## Slide 9: Matrix Profile vs LoCoMotif

- Matrix Profile: fixed-window shape recurrence and direct profile diagnostics.
- LoCoMotif: flexible-length interval motif sets.
- Output units differ, so raw motif counts are not directly comparable.
- Runtime comparisons require matched datasets, modes, and search settings.
- The methods provide complementary rather than interchangeable evidence.

**Recommended figure:** `selected_final_figures/04_BTCUSDT_15m_conditioned_quantile_2_rolling_240_high_vol_l24_72_intervals.png`

**Speaker note:** The comparison should focus on methodological behavior, completed scope, and qualitative evidence. A numerical winner cannot be inferred from unmatched raw counts.

## Slide 10: Main findings and limitations

- The results support the presence of recurring subsequence shapes in the studied cryptocurrency series, and show that regime-conditioned discovery produces identifiable motifs within volatility states, while agnostic discovery provides a global recurrence baseline. Differences in row counts across modes and algorithms describe the configured search process, not statistical significance, causality, or trading profitability. LoCoMotif is reported as a controlled subset experiment rather than a full-scale benchmark.
- Conditioned count inflation partly reflects repeated regime-specific searches.
- Shape recurrence does not establish statistical significance or profitability.
- Quantile interpretation must follow the actual stored volatility column.
- Missing files, failures, and capped experiments limit generalization.

**Recommended figure:** `selected_final_figures/01_BTCUSDT_15m_quantile_2_rolling_240_transition_heatmap.png`

**Speaker note:** The defensible conclusion is that recurring shapes can be characterized globally and within regimes when completed outputs support them. Predictive or economic claims require separate out-of-sample testing.
