from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feature_selection import choose_volatility_column


def semantic_volatility_labels(n_regimes: int) -> list[str]:
    if n_regimes == 2:
        return ["low_vol", "high_vol"]
    if n_regimes == 3:
        return ["low_vol", "medium_vol", "high_vol"]
    if n_regimes == 4:
        return ["low_vol", "medium_vol", "high_vol", "extreme_vol"]
    return [f"regime_{i}" for i in range(n_regimes)]


def summarize_regimes(
    df: pd.DataFrame,
    label_col: str,
    asset: str,
    frequency: str,
    regime_method: str,
    vol_col: str | None = None,
) -> pd.DataFrame:
    if df.empty or label_col not in df.columns:
        return pd.DataFrame()
    vol_col = vol_col if vol_col in df.columns else choose_volatility_column(df)
    return_col = "log_return" if "log_return" in df.columns else None
    rows: list[dict[str, Any]] = []
    total = max(len(df), 1)
    for regime_label, group in df.groupby(label_col, dropna=False):
        row = {
            "asset": asset,
            "frequency": frequency,
            "regime_method": regime_method,
            "regime_label": regime_label,
            "observations": int(len(group)),
            "share": float(len(group) / total),
            "mean_return": float(group[return_col].mean()) if return_col else np.nan,
            "std_return": float(group[return_col].std()) if return_col else np.nan,
            "mean_rolling_vol": float(group[vol_col].mean()) if vol_col else np.nan,
            "median_rolling_vol": float(group[vol_col].median()) if vol_col else np.nan,
            "min_timestamp": group["timestamp"].min(),
            "max_timestamp": group["timestamp"].max(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def transition_table(
    df: pd.DataFrame,
    label_col: str,
    asset: str,
    frequency: str,
    regime_method: str,
) -> pd.DataFrame:
    if df.empty or label_col not in df.columns:
        return pd.DataFrame()
    labels = df[label_col].astype(str).reset_index(drop=True)
    if len(labels) < 2:
        return pd.DataFrame()
    transitions = pd.DataFrame({"from_regime": labels.iloc[:-1].values, "to_regime": labels.iloc[1:].values})
    counts = transitions.value_counts(["from_regime", "to_regime"]).reset_index(name="count")
    totals = counts.groupby("from_regime")["count"].transform("sum")
    counts["probability"] = counts["count"] / totals
    counts.insert(0, "regime_method", regime_method)
    counts.insert(0, "frequency", frequency)
    counts.insert(0, "asset", asset)
    return counts


def continuous_segments(
    labels: pd.Series,
    min_length: int,
    timestamps: pd.Series | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if labels.empty:
        return pd.DataFrame(rows)

    current_label = labels.iloc[0]
    start = 0
    segment_counter = 0
    for position in range(1, len(labels) + 1):
        is_end = position == len(labels)
        if is_end or labels.iloc[position] != current_label:
            end = position
            length = end - start
            if pd.notna(current_label) and length >= int(min_length):
                segment_counter += 1
                row = {
                    "segment_id": f"{str(current_label)}_{segment_counter:04d}",
                    "regime_label": current_label,
                    "start_pos": int(start),
                    "end_pos_exclusive": int(end),
                    "n_observations": int(length),
                }
                if timestamps is not None and len(timestamps) >= end:
                    row["start_timestamp"] = timestamps.iloc[start]
                    row["end_timestamp"] = timestamps.iloc[end - 1]
                rows.append(row)
            if not is_end:
                current_label = labels.iloc[position]
                start = position
    return pd.DataFrame(rows)


def load_regime_labels(workflow_root: str | Path, source: str, suffix: str = "") -> pd.DataFrame:
    workflow_root = Path(workflow_root)
    if source == "quantile":
        base = workflow_root / "results" / "regimes" / "quantile" / f"quantile_regime_labels{suffix}.parquet"
        fallback = base.with_suffix(".csv")
    elif source == "hmm":
        base = workflow_root / "results" / "regimes" / "hmm" / f"hmm_regime_labels{suffix}.parquet"
        fallback = base.with_suffix(".csv")
    else:
        raise ValueError(f"Unknown regime label source: {source}")
    if base.exists():
        return pd.read_parquet(base)
    if fallback.exists():
        return pd.read_csv(fallback, parse_dates=["timestamp"])
    return pd.DataFrame()


def merge_regime_labels(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    asset: str,
    frequency: str,
    regime_method: str,
) -> pd.DataFrame:
    if labels.empty:
        return df.copy()
    subset = labels[
        (labels["asset"].astype(str) == str(asset))
        & (labels["frequency"].astype(str) == str(frequency))
        & (labels["regime_method"].astype(str) == str(regime_method))
    ].copy()
    if subset.empty:
        return df.copy()
    subset["timestamp"] = pd.to_datetime(subset["timestamp"], utc=True, errors="coerce")
    columns = ["timestamp", "regime_label"]
    if "regime_confidence" in subset.columns:
        columns.append("regime_confidence")
    merged = df.merge(subset[columns], on="timestamp", how="left")
    return merged


def default_regime_methods(labels: pd.DataFrame, config: dict[str, Any], source: str) -> list[str]:
    if labels.empty or "regime_method" not in labels.columns:
        return []
    methods = sorted(labels["regime_method"].dropna().astype(str).unique())
    if config.get("active_mode") == "local":
        if source == "quantile":
            count = config.get("quantile", {}).get("default_regime_count", 3)
            methods = [m for m in methods if f"quantile_{count}_" in m] or methods[:1]
        else:
            methods = methods[:1]
    return methods


def compare_regime_partitions(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    if left.empty or right.empty:
        return pd.DataFrame()
    merged = left.merge(
        right,
        on=["asset", "frequency", "timestamp"],
        how="inner",
        suffixes=(f"_{left_name}", f"_{right_name}"),
    )
    if merged.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        sklearn_available = True
    except Exception:
        adjusted_rand_score = None
        normalized_mutual_info_score = None
        sklearn_available = False

    for keys, group in merged.groupby(
        [
            "asset",
            "frequency",
            f"regime_method_{left_name}",
            f"regime_method_{right_name}",
        ]
    ):
        asset, frequency, left_method, right_method = keys
        left_labels = group[f"regime_label_{left_name}"].astype(str)
        right_labels = group[f"regime_label_{right_name}"].astype(str)
        row = {
            "asset": asset,
            "frequency": frequency,
            f"{left_name}_method": left_method,
            f"{right_name}_method": right_method,
            "observations": int(len(group)),
            "adjusted_rand_index": np.nan,
            "normalized_mutual_information": np.nan,
            "sklearn_available": sklearn_available,
        }
        if sklearn_available:
            row["adjusted_rand_index"] = float(adjusted_rand_score(left_labels, right_labels))
            row["normalized_mutual_information"] = float(normalized_mutual_info_score(left_labels, right_labels))
        rows.append(row)
    return pd.DataFrame(rows)


def timestamp_regime_lookup(labels: pd.DataFrame, asset: str, frequency: str, regime_method: str) -> dict[pd.Timestamp, str]:
    if labels.empty:
        return {}
    subset = labels[
        (labels["asset"].astype(str) == str(asset))
        & (labels["frequency"].astype(str) == str(frequency))
        & (labels["regime_method"].astype(str) == str(regime_method))
    ].copy()
    if subset.empty:
        return {}
    subset["timestamp"] = pd.to_datetime(subset["timestamp"], utc=True, errors="coerce")
    return dict(zip(subset["timestamp"], subset["regime_label"].astype(str)))
