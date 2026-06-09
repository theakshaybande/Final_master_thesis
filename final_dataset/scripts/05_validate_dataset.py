"""Validate expected parquet datasets and write metadata summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
OVERWRITE = True
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

METADATA_DIR = BASE_DIR / "metadata"
INVENTORY_PATH = METADATA_DIR / "dataset_inventory.csv"
MISSING_REPORT_PATH = METADATA_DIR / "missing_data_report.csv"


def expected_files() -> list[dict]:
    """Return the locked expected dataset outputs."""
    files: list[dict] = []
    for symbol in SYMBOLS:
        files.append(
            {
                "asset_class": "crypto",
                "symbol": symbol,
                "frequency": "1m",
                "kind": "processed",
                "path": BASE_DIR / "processed" / "crypto" / "1m" / f"{symbol}_1m_2020_2025.parquet",
                "expected_freq": "1min",
            }
        )
        for frequency, pandas_freq in [("5m", "5min"), ("15m", "15min"), ("1h", "1h"), ("1d", "1D")]:
            files.append(
                {
                    "asset_class": "crypto",
                    "symbol": symbol,
                    "frequency": frequency,
                    "kind": "processed",
                    "path": BASE_DIR / "processed" / "crypto" / frequency / f"{symbol}_{frequency}_2020_2025.parquet",
                    "expected_freq": pandas_freq,
                }
            )
        for frequency in ["1m", "5m", "15m", "1h", "1d"]:
            files.append(
                {
                    "asset_class": "crypto",
                    "symbol": symbol,
                    "frequency": frequency,
                    "kind": "features",
                    "path": BASE_DIR / "features" / "crypto" / f"{symbol}_{frequency}_features_2020_2025.parquet",
                    "expected_freq": None,
                }
            )

    for symbol in ["EURUSD", "GBPUSD"]:
        files.extend(
            [
                {
                    "asset_class": "fx",
                    "symbol": symbol,
                    "frequency": "1d",
                    "kind": "processed",
                    "path": BASE_DIR / "processed" / "fx" / f"{symbol}_1d_2015_2025.parquet",
                    "expected_freq": None,
                },
                {
                    "asset_class": "fx",
                    "symbol": symbol,
                    "frequency": "1d",
                    "kind": "features",
                    "path": BASE_DIR / "features" / "fx" / f"{symbol}_1d_features_2015_2025.parquet",
                    "expected_freq": None,
                },
            ]
        )

    for symbol in ["SP500", "NASDAQ100", "DAX"]:
        files.extend(
            [
                {
                    "asset_class": "equity_indices",
                    "symbol": symbol,
                    "frequency": "1d",
                    "kind": "processed",
                    "path": BASE_DIR / "processed" / "equity_indices" / f"{symbol}_1d_2010_2025.parquet",
                    "expected_freq": None,
                },
                {
                    "asset_class": "equity_indices",
                    "symbol": symbol,
                    "frequency": "1d",
                    "kind": "features",
                    "path": BASE_DIR / "features" / "equity_indices" / f"{symbol}_1d_features_2010_2025.parquet",
                    "expected_freq": None,
                },
            ]
        )

    files.extend(
        [
            {
                "asset_class": "volatility",
                "symbol": "VIX",
                "frequency": "1d",
                "kind": "processed",
                "path": BASE_DIR / "processed" / "volatility" / "VIX_1d_2010_2025.parquet",
                "expected_freq": None,
            },
            {
                "asset_class": "volatility",
                "symbol": "VIX",
                "frequency": "1d",
                "kind": "features",
                "path": BASE_DIR / "features" / "volatility" / "VIX_1d_features_2010_2025.parquet",
                "expected_freq": None,
            },
        ]
    )
    return files


def count_missing_timestamps(frame: pd.DataFrame, expected_freq: str | None) -> int | None:
    """Count missing timestamps for fixed-frequency crypto files."""
    if expected_freq is None or frame.empty:
        return None
    timestamps = pd.to_datetime(frame["timestamp"], utc=True).sort_values().drop_duplicates()
    full_index = pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq=expected_freq)
    return int(len(full_index.difference(pd.DatetimeIndex(timestamps))))


def validate_one(item: dict) -> tuple[dict, list[dict]]:
    """Validate one expected parquet file."""
    path = item["path"]
    inventory = {
        "asset_class": item["asset_class"],
        "symbol": item["symbol"],
        "frequency": item["frequency"],
        "kind": item["kind"],
        "path": str(path),
        "exists": path.exists(),
        "rows": 0,
        "start_timestamp": "",
        "end_timestamp": "",
        "duplicate_timestamps": "",
        "missing_timestamps": "",
        "memory_size_mb": "",
        "file_size_mb": round(path.stat().st_size / 1_000_000, 3) if path.exists() else "",
        "columns": "",
        "status": "missing",
    }
    missing_rows: list[dict] = []

    if not path.exists():
        missing_rows.append({**inventory, "column": "__file__", "missing_values": "file_missing"})
        return inventory, missing_rows

    try:
        frame = pd.read_parquet(path)
        if "timestamp" not in frame.columns:
            raise ValueError("Missing timestamp column")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        duplicate_count = int(frame["timestamp"].duplicated().sum())
        missing_count = count_missing_timestamps(frame, item["expected_freq"])
        inventory.update(
            {
                "rows": len(frame),
                "start_timestamp": frame["timestamp"].min().isoformat() if len(frame) else "",
                "end_timestamp": frame["timestamp"].max().isoformat() if len(frame) else "",
                "duplicate_timestamps": duplicate_count,
                "missing_timestamps": "" if missing_count is None else missing_count,
                "memory_size_mb": round(frame.memory_usage(deep=True).sum() / 1_000_000, 3),
                "columns": "|".join(frame.columns),
                "status": "ok" if duplicate_count == 0 else "warning",
            }
        )
        for column, value in frame.isna().sum().items():
            missing_rows.append(
                {
                    "asset_class": item["asset_class"],
                    "symbol": item["symbol"],
                    "frequency": item["frequency"],
                    "kind": item["kind"],
                    "path": str(path),
                    "column": column,
                    "missing_values": int(value),
                    "missing_pct": round(float(value) / len(frame), 6) if len(frame) else "",
                }
            )
    except Exception as exc:
        inventory["status"] = "error"
        missing_rows.append({**inventory, "column": "__error__", "missing_values": str(exc)})

    return inventory, missing_rows


def main() -> None:
    """Run validation and write CSV metadata outputs."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    inventories = []
    missing_reports = []

    for item in expected_files():
        inventory, missing_rows = validate_one(item)
        inventories.append(inventory)
        missing_reports.extend(missing_rows)
        print(f"[{inventory['status']}] {inventory['path']}")

    pd.DataFrame(inventories).to_csv(INVENTORY_PATH, index=False)
    pd.DataFrame(missing_reports).to_csv(MISSING_REPORT_PATH, index=False)
    print(f"[saved] {INVENTORY_PATH}")
    print(f"[saved] {MISSING_REPORT_PATH}")


if __name__ == "__main__":
    main()
