from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from feature_selection import choose_volatility_column, ensure_core_features
from regime_utils import semantic_volatility_labels, summarize_regimes, transition_table


def create_quantile_regime_frame(
    df: pd.DataFrame,
    asset: str,
    frequency: str,
    n_regimes: int,
    rolling_window: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepared = ensure_core_features(df, rolling_window=rolling_window)
    vol_col = choose_volatility_column(prepared, config.get("quantile", {}).get("volatility_columns"))
    if vol_col is None:
        raise ValueError("No volatility column is available for quantile regime detection.")

    values = pd.to_numeric(prepared[vol_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.ffill().bfill()
    if values.notna().sum() < max(30, n_regimes * 10):
        raise ValueError(f"Not enough non-missing volatility observations for {asset} {frequency}.")

    quantile_points = [i / n_regimes for i in range(1, n_regimes)]
    thresholds = values.quantile(quantile_points).to_numpy(dtype=float)
    labels = semantic_volatility_labels(n_regimes)
    codes = np.searchsorted(thresholds, values.to_numpy(dtype=float), side="right")
    regime_labels = pd.Series([labels[int(code)] for code in codes], index=prepared.index)
    regime_method = f"quantile_{n_regimes}_rolling_{rolling_window}"

    label_df = pd.DataFrame(
        {
            "timestamp": prepared["timestamp"],
            "asset": asset,
            "frequency": frequency,
            "regime_method": regime_method,
            "regime_label": regime_labels,
            "regime_code": codes.astype(int),
            "rolling_window": int(rolling_window),
            "n_regimes": int(n_regimes),
            "volatility_column": vol_col,
            "volatility_value": values,
        }
    )
    label_df["thresholds"] = ",".join(f"{threshold:.12g}" for threshold in thresholds)

    working = prepared.copy()
    working["regime_label"] = regime_labels
    summary = summarize_regimes(working, "regime_label", asset, frequency, regime_method, vol_col=vol_col)
    transitions = transition_table(working, "regime_label", asset, frequency, regime_method)
    return label_df, summary, transitions


def run_quantile_regime_detection(
    df: pd.DataFrame,
    asset: str,
    frequency: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q_cfg = config.get("quantile", {})
    all_labels = []
    all_summaries = []
    all_transitions = []
    for rolling_window in q_cfg.get("rolling_windows", [60]):
        for n_regimes in q_cfg.get("regime_counts", [3]):
            labels, summary, transitions = create_quantile_regime_frame(
                df,
                asset=asset,
                frequency=frequency,
                n_regimes=int(n_regimes),
                rolling_window=int(rolling_window),
                config=config,
            )
            all_labels.append(labels)
            all_summaries.append(summary)
            all_transitions.append(transitions)
    return (
        pd.concat(all_labels, ignore_index=True) if all_labels else pd.DataFrame(),
        pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame(),
        pd.concat(all_transitions, ignore_index=True) if all_transitions else pd.DataFrame(),
    )

