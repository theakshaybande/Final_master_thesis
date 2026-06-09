from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


NON_MOTIF_COLUMNS = {
    "timestamp",
    "datetime",
    "date",
    "time",
    "open_time",
    "close_time",
    "asset",
    "frequency",
    "market",
    "symbol",
}

FEATURE_KEYWORDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote",
    "trade",
    "taker",
    "return",
    "vol",
    "range",
    "spread",
    "zscore",
]


def ensure_core_features(df: pd.DataFrame, rolling_window: int = 60) -> pd.DataFrame:
    df = df.copy()
    if "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce")
        if "log_return" not in df.columns:
            df["log_return"] = np.log(close.replace(0, np.nan)).diff()
        if "pct_return" not in df.columns:
            df["pct_return"] = close.pct_change()

    if "log_return" in df.columns:
        log_return = pd.to_numeric(df["log_return"], errors="coerce")
        if "absolute_return" not in df.columns:
            df["absolute_return"] = log_return.abs()
        if "abs_log_return" not in df.columns:
            df["abs_log_return"] = log_return.abs()
        if "squared_return" not in df.columns:
            df["squared_return"] = log_return.pow(2)
        if "rolling_vol" not in df.columns:
            df["rolling_vol"] = log_return.rolling(rolling_window, min_periods=max(5, rolling_window // 5)).std()
        rolling_name = f"rolling_volatility_{rolling_window}"
        if rolling_name not in df.columns:
            df[rolling_name] = log_return.rolling(rolling_window, min_periods=max(5, rolling_window // 5)).std()

    if {"high", "low"}.issubset(df.columns):
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        if "range" not in df.columns:
            df["range"] = high - low
        if "hl_range" not in df.columns:
            denominator = pd.to_numeric(df.get("close", high), errors="coerce").replace(0, np.nan)
            df["hl_range"] = (high - low) / denominator
        if "spread_proxy" not in df.columns:
            denominator = pd.to_numeric(df.get("close", high), errors="coerce").replace(0, np.nan)
            df["spread_proxy"] = (high - low) / denominator

    return df


def choose_volatility_column(df: pd.DataFrame, preferred: list[str] | None = None) -> str | None:
    preferred = preferred or [
        "rolling_volatility_60",
        "rolling_volatility_30",
        "rolling_volatility_240",
        "rolling_vol",
        "realized_vol",
    ]
    for column in preferred:
        if column in df.columns:
            return column
    candidates = [column for column in df.columns if "vol" in str(column).lower()]
    if not candidates:
        return None

    def score(column: str) -> tuple[int, str]:
        digits = "".join(ch for ch in column if ch.isdigit())
        if digits:
            return (abs(int(digits) - 60), column)
        return (10_000, column)

    return sorted(candidates, key=score)[0]


def _resolve_feature(df: pd.DataFrame, feature: str) -> str | None:
    if feature in df.columns:
        return feature
    aliases = {
        "absolute_return": ["absolute_return", "abs_log_return"],
        "rolling_vol": ["rolling_vol", "rolling_volatility_60", "rolling_volatility_30", "realized_vol"],
        "range": ["range", "hl_range", "spread_proxy"],
    }
    for candidate in aliases.get(feature, []):
        if candidate in df.columns:
            return candidate
    if feature == "rolling_volatility_60":
        return choose_volatility_column(df)
    return None


def candidate_numeric_features(df: pd.DataFrame, preferred_features: list[str] | None = None) -> list[str]:
    selected: list[str] = []
    preferred_features = preferred_features or []
    for feature in preferred_features:
        resolved = _resolve_feature(df, feature)
        if resolved and resolved not in selected:
            selected.append(resolved)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        low = str(column).lower()
        if low in NON_MOTIF_COLUMNS:
            continue
        if any(keyword in low for keyword in FEATURE_KEYWORDS) and column not in selected:
            selected.append(column)
    return selected


def clean_feature_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    max_nan_fraction: float = 0.40,
    min_non_constant_values: int = 3,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    frame = df[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    diagnostics = []
    kept: list[str] = []
    for column in frame.columns:
        missing_fraction = float(frame[column].isna().mean())
        non_constant = int(frame[column].nunique(dropna=True))
        keep = missing_fraction <= max_nan_fraction and non_constant >= min_non_constant_values
        diagnostics.append(
            {
                "feature": column,
                "missing_fraction": missing_fraction,
                "unique_values": non_constant,
                "kept": keep,
            }
        )
        if keep:
            kept.append(column)
    if not kept:
        raise ValueError("No usable numeric motif features after missing-value and variance checks.")
    cleaned = frame[kept].ffill().bfill().fillna(0.0)
    return cleaned, kept, pd.DataFrame(diagnostics)


def scale_feature_frame(frame: pd.DataFrame, scaler: str = "robust") -> tuple[pd.DataFrame, pd.DataFrame]:
    values = frame.astype(float)
    if scaler == "zscore":
        center = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0).replace(0, np.nan)
        method = "zscore"
    else:
        center = values.median(axis=0)
        q75 = values.quantile(0.75)
        q25 = values.quantile(0.25)
        scale = (q75 - q25).replace(0, np.nan)
        fallback = values.std(axis=0, ddof=0).replace(0, np.nan)
        scale = scale.fillna(fallback)
        method = "robust"

    scaled = ((values - center) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    stats = pd.DataFrame({"feature": values.columns, "center": center.values, "scale": scale.fillna(1.0).values})
    stats["scaler"] = method
    return scaled, stats


def select_motif_feature_matrix(
    df: pd.DataFrame,
    config: dict[str, Any],
    preferred_features: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[str], pd.DataFrame]:
    feature_cfg = config.get("feature_selection", {})
    rolling_window = int(config.get("quantile", {}).get("default_rolling_window", 60))
    prepared = ensure_core_features(df, rolling_window=rolling_window)
    preferred = preferred_features or feature_cfg.get("preferred_features", [])
    candidates = candidate_numeric_features(prepared, preferred)
    cleaned, kept, diagnostics = clean_feature_frame(
        prepared,
        candidates,
        max_nan_fraction=float(feature_cfg.get("max_nan_fraction", 0.40)),
        min_non_constant_values=int(feature_cfg.get("min_non_constant_values", 3)),
    )
    max_features = feature_cfg.get("local_max_features") if config.get("active_mode") == "local" else feature_cfg.get("max_features")
    if max_features:
        kept = kept[: int(max_features)]
        cleaned = cleaned[kept]
        diagnostics.loc[~diagnostics["feature"].isin(kept), "kept"] = False
    scaled, stats = scale_feature_frame(cleaned, scaler=str(feature_cfg.get("scaler", "robust")))
    diagnostics = diagnostics.merge(stats, on="feature", how="left")
    return scaled.to_numpy(dtype=np.float64), scaled, kept, diagnostics


def select_hmm_feature_matrix(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, list[str], pd.DataFrame]:
    feature_cfg = config.get("feature_selection", {})
    prepared = ensure_core_features(df, rolling_window=int(config.get("quantile", {}).get("default_rolling_window", 60)))
    preferred = feature_cfg.get("hmm_features", [])
    columns = []
    for feature in preferred:
        resolved = _resolve_feature(prepared, feature)
        if resolved and resolved not in columns:
            columns.append(resolved)
    if not columns:
        columns = candidate_numeric_features(prepared, preferred)
    cleaned, kept, diagnostics = clean_feature_frame(
        prepared,
        columns,
        max_nan_fraction=float(feature_cfg.get("max_nan_fraction", 0.40)),
        min_non_constant_values=int(feature_cfg.get("min_non_constant_values", 3)),
    )
    max_features = feature_cfg.get("local_max_features") if config.get("active_mode") == "local" else feature_cfg.get("max_features")
    if max_features:
        kept = kept[: int(max_features)]
        cleaned = cleaned[kept]
        diagnostics.loc[~diagnostics["feature"].isin(kept), "kept"] = False
    scaled, stats = scale_feature_frame(cleaned, scaler=str(feature_cfg.get("scaler", "robust")))
    diagnostics = diagnostics.merge(stats, on="feature", how="left")
    return scaled.to_numpy(dtype=np.float64), scaled, kept, diagnostics
