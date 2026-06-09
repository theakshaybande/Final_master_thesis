from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


FEATURE_FILE_RE = re.compile(
    r"^(?P<asset>.+)_(?P<frequency>\d+[mhd]|daily)_features(?:_.+)?\.parquet$",
    re.IGNORECASE,
)


def parse_feature_filename(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    match = FEATURE_FILE_RE.match(path.name)
    if match:
        asset = match.group("asset")
        frequency = match.group("frequency").lower()
    else:
        parts = path.stem.split("_")
        asset = parts[0] if parts else path.stem
        frequency = next((part.lower() for part in parts if re.fullmatch(r"\d+[mhd]", part.lower())), "unknown")
    return {
        "path": path,
        "asset": asset,
        "frequency": frequency,
        "market": path.parent.name,
        "filename": path.name,
    }


def discover_feature_files(project_root: str | Path, config: dict[str, Any]) -> pd.DataFrame:
    project_root = Path(project_root)
    data_cfg = config.get("data", {})
    feature_root = project_root / data_cfg.get("feature_root", "final_dataset/features")
    feature_glob = data_cfg.get("feature_glob", "**/*_features_*.parquet")
    files = sorted(feature_root.glob(feature_glob))
    rows = [parse_feature_filename(path) for path in files if path.is_file()]
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        return pd.DataFrame(columns=["path", "asset", "frequency", "market", "filename"])

    allowed_assets = data_cfg.get("allowed_assets")
    allowed_frequencies = data_cfg.get("allowed_frequencies")
    if config.get("active_mode") == "local":
        allowed_frequencies = data_cfg.get("local_allowed_frequencies") or allowed_frequencies

    if allowed_assets:
        inventory = inventory[inventory["asset"].isin(set(map(str, allowed_assets)))]
    if allowed_frequencies:
        freq_set = {str(freq).lower() for freq in allowed_frequencies}
        inventory = inventory[inventory["frequency"].str.lower().isin(freq_set)]

    inventory = inventory.sort_values(["market", "asset", "frequency", "filename"]).reset_index(drop=True)

    if config.get("active_mode") == "local":
        max_assets = data_cfg.get("local_max_assets")
        if max_assets:
            keep_assets = list(inventory["asset"].drop_duplicates().head(int(max_assets)))
            inventory = inventory[inventory["asset"].isin(keep_assets)]
        max_files = data_cfg.get("local_max_files")
    else:
        max_files = data_cfg.get("max_files")

    if max_files:
        inventory = inventory.head(int(max_files))

    return inventory.reset_index(drop=True)


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        median_value = numeric.dropna().median()
        if median_value > 1e17:
            return pd.to_datetime(numeric, unit="ns", utc=True, errors="coerce")
        if median_value > 1e14:
            return pd.to_datetime(numeric, unit="us", utc=True, errors="coerce")
        if median_value > 1e11:
            return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        if median_value > 1e9:
            return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def load_feature_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path).copy()
    timestamp_candidates = ["timestamp", "datetime", "date", "open_time", "time"]
    timestamp_col = next((column for column in timestamp_candidates if column in df.columns), None)
    if timestamp_col is not None:
        df["timestamp"] = _parse_timestamp_series(df[timestamp_col])
    elif isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df.index, utc=True, errors="coerce")
    else:
        raise ValueError(f"No timestamp column found in {path}")

    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )
    return df


def apply_mode_limits(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if config.get("active_mode") != "local":
        return df.reset_index(drop=True)

    data_cfg = config.get("data", {})
    start_date = data_cfg.get("local_start_date")
    end_date = data_cfg.get("local_end_date")
    limited = df.copy()
    if start_date:
        limited = limited[limited["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        limited = limited[limited["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]
    max_rows = data_cfg.get("local_max_rows")
    if max_rows:
        limited = limited.head(int(max_rows))
    return limited.reset_index(drop=True)


def describe_feature_frame(meta: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    return {
        "asset": meta.get("asset"),
        "frequency": meta.get("frequency"),
        "market": meta.get("market"),
        "path": str(meta.get("path")),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_columns": int(len(numeric_columns)),
        "min_timestamp": df["timestamp"].min().isoformat() if len(df) else None,
        "max_timestamp": df["timestamp"].max().isoformat() if len(df) else None,
        "column_names": ", ".join(map(str, df.columns)),
    }

