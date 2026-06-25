# LaTeX-ready thesis text

\section{Regime Detection Validation}

The quantile procedure produced 73,097,388 labelled observations, while the hidden Markov model procedure produced 8,121,932 labelled observations. The corresponding failure logs contained 0 and 0 rows, respectively. The observed quantile labels were extreme\_vol, high\_vol, low\_vol, medium\_vol, and the observed HMM labels were extreme\_vol, high\_vol, low\_vol, medium\_vol. Missing or empty outputs are treated as limitations of the completed computational scope rather than imputed results.

The quantile regime outputs store `rolling\_volatility\_60` as the actual volatility column across all quantile method identifiers. Therefore, quantile regimes should be interpreted as 60-period rolling-volatility regimes with different regime-count granularities rather than as separate 30/60/240 volatility-horizon experiments.

\section{Matrix Profile Motif Discovery Results}

The Matrix Profile output contained 81,987 motif-result rows. Of these, 720 rows were associated with agnostic searches and 81,267 rows were associated with regime-conditioned searches. Agnostic discovery identifies recurring subsequences over the full eligible series, whereas conditioned discovery repeats the search within eligible regime-specific subsets. The resulting motifs constitute evidence of shape recurrence under the configured windows and features; they do not by themselves establish statistical significance, predictive ability, or economic profitability.

\section{LoCoMotif Motif Discovery Results}

The LoCoMotif output contained 18,147 motif interval rows and 38 evaluation rows. The failure log contained not available rows. LoCoMotif is reported as a controlled subset experiment rather than a full-scale benchmark. LoCoMotif provides complementary evidence because it represents recurring structures as flexible-length intervals rather than fixed-window motif pairs.

\section{Comparison of Agnostic and Regime-Conditioned Discovery}

The agnostic configuration provides a global recurrence baseline, while regime-conditioned configurations evaluate recurring shapes within market states. A larger conditioned output count can arise mechanically because motif search is repeated over multiple regimes, regime methods, and contiguous segments. Therefore, raw row counts should not be interpreted as direct evidence that one mode is statistically superior. Comparisons should instead consider matched search settings, recurrence metrics, stability, cross-regime overlap, and representative motif plots.

\section{Computational Performance}

Runtime measurements were summarized by the available asset, frequency, mode, profile type, and status fields. Runtime comparisons are valid only when the underlying data volume and search configuration are comparable. Unusually large values should be checked for cumulative job-time accounting, initialization overhead, or retries before being quoted as per-experiment execution time.

\section{Summary of Empirical Findings}

The results support the presence of recurring subsequence shapes in the studied cryptocurrency series, and show that regime-conditioned discovery produces identifiable motifs within volatility states, while agnostic discovery provides a global recurrence baseline. Differences in row counts across modes and algorithms describe the configured search process, not statistical significance, causality, or trading profitability. LoCoMotif is reported as a controlled subset experiment rather than a full-scale benchmark.

\section{Limitations}

The study is limited by the completed computational scope, any missing result files, recorded failures, and potentially capped LoCoMotif experiments. Motif counts depend on the number of assets, frequencies, windows, features, regimes, and segments searched. Matrix Profile pair rows and LoCoMotif interval rows are different output units and are not directly comparable. Finally, recurring shape similarity does not establish causality, statistical significance, out-of-sample predictability, or trading profitability.
