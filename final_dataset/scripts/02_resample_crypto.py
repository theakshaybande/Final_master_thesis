"""Resample processed Binance 1-minute crypto bars to coarser OHLCV frequencies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parents[1]
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
OVERWRITE = False
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

INPUT_DIR = BASE_DIR / "processed" / "crypto" / "1m"
OUTPUT_ROOT = BASE_DIR / "processed" / "crypto"
RESAMPLE_RULES = {
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "1d": "1D",
}


def load_1m(symbol: str) -> pd.DataFrame:
    """Load one processed 1-minute parquet file."""
    path = INPUT_DIR / f"{symbol}_1m_{START_DATE[:4]}_{END_DATE[:4]}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing required 1-minute file: {path}")
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if "close_time" in frame.columns:
        frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
    return frame.sort_values("timestamp").drop_duplicates("timestamp")


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Apply thesis OHLCV aggregation rules."""
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "number_of_trades": "sum",
        "taker_buy_base_volume": "sum",
        "taker_buy_quote_volume": "sum",
    }
    if "close_time" in frame.columns:
        aggregations["close_time"] = "max"

    indexed = frame.set_index("timestamp")
    resampled = indexed.resample(rule, label="left", closed="left").agg(aggregations)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()
    if "number_of_trades" in resampled.columns:
        resampled["number_of_trades"] = resampled["number_of_trades"].round().astype("Int64")
    return resampled


def output_matches_source(output_path: Path, source: pd.DataFrame) -> bool:
    """Return true when an existing resampled file reaches the same final source timestamp."""
    try:
        if not output_path.exists():
            return False
        existing = pd.read_parquet(output_path, columns=["timestamp"])
        if existing.empty or source.empty:
            return False
        existing_end = pd.to_datetime(existing["timestamp"], utc=True, errors="coerce").max()
        source_end = pd.to_datetime(source["timestamp"], utc=True, errors="coerce").max()
        return existing_end.date() >= source_end.date()
    except Exception:
        return False


def main() -> None:
    """Run all configured crypto resampling jobs."""
    for symbol in SYMBOLS:
        try:
            source = load_1m(symbol)
        except Exception as exc:
            print(f"[error] {symbol}: {exc}")
            continue

        for label, rule in tqdm(RESAMPLE_RULES.items(), desc=f"Resampling {symbol}", unit="freq"):
            output_dir = OUTPUT_ROOT / label
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{symbol}_{label}_{START_DATE[:4]}_{END_DATE[:4]}.parquet"
            if output_path.exists() and not OVERWRITE and output_matches_source(output_path, source):
                print(f"[skip] {output_path} exists")
                continue
            if output_path.exists() and not OVERWRITE:
                print(f"[repair] {output_path} does not match source coverage; rebuilding")
            try:
                result = resample_ohlcv(source, rule)
                result.to_parquet(output_path, engine="pyarrow", index=False)
                print(f"[saved] {output_path} ({len(result):,} rows)")
            except Exception as exc:
                print(f"[error] {symbol} {label}: {exc}")


if __name__ == "__main__":
    main()
