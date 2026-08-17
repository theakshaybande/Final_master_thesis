# Tasks 12–14: MP vs LoCoMotif Evaluation Summary

## Task 12 — Controlled comparison

Final controlled cases:

- BTCUSDT 1h regime-agnostic
- BTCUSDT 15m high-volatility
- BTCUSDT 15m low-volatility
- common feature: `close_z`

Matrix Profile returns fixed-length nearest-neighbour motif pairs.
LoCoMotif returns variable-length time-warped motif sets.

Key controlled results:

| Case | Method | Motif count | Occurrences | Length/window | Quality |
|---|---|---:|---:|---|---|
| BTCUSDT 1h agnostic | Matrix Profile | 1 | 2 | m=24 | distance=1.097757 |
| BTCUSDT 1h agnostic | LoCoMotif | 3 | 18 | 12–48 | rho=0.65 |
| BTCUSDT 15m high-vol | Matrix Profile | 1 | 2 | m=32 | distance=1.546767 |
| BTCUSDT 15m high-vol | LoCoMotif | 3 | 27 | 12–48 | rho=0.65 |
| BTCUSDT 15m low-vol | Matrix Profile | 1 | 2 | m=32 | distance=1.451609 |
| BTCUSDT 15m low-vol | LoCoMotif | 3 | 28 | 12–48 | rho=0.65 |

Runtime values are stored in `final_comparison_cases.csv`.

Important caveat:
the first STUMPY calls around 19–21 seconds likely include Numba JIT compilation.
They should not be interpreted as steady-state Matrix Profile runtime.

## Task 13 — Interpretation

Both methods discover recurring structure in the same controlled BTCUSDT slices.

Matrix Profile identifies the closest fixed-length nearest-neighbour pair.

LoCoMotif identifies broader motif sets containing multiple related occurrences,
allowing variable motif duration and local time warping.

Therefore motif counts are not directly comparable across the two methods.

The methods are complementary:
- Matrix Profile provides precise fixed-length similarity.
- LoCoMotif captures broader recurrent motif-set structure.

## Task 14 — Evaluation framework

The thesis uses a multi-dimensional motif evaluation framework.

Intrinsic metrics:
- nearest-neighbour distance
- recurrence / occurrence count
- motif-set count
- mean and median motif length
- temporal coverage
- overlap / redundancy

Robustness metrics:
- time-split stability
- cross-regime recurrence / overlap
- parameter sensitivity
- window-length sensitivity
- regime-specification sensitivity

Computational metrics:
- runtime
- scaling behaviour
- observations processed

Ground-truth metrics such as precision, recall, F1, and PROM-style evaluation
are not primary metrics because financial datasets lack labelled motif ground truth.

Next downstream financial evaluation:
- forward returns
- directional hit rate
- future realized volatility
- MFE
- MAE
- directional consistency
- comparison against unconditional/random timestamps
