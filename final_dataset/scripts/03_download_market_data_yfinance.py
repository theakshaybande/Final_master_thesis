"""Download FX, equity index, and VIX daily data from yfinance."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parents[1]
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
OVERWRITE = False
SYMBOLS = ["EURUSD=X", "GBPUSD=X", "^GSPC", "^NDX", "^GDAXI", "^VIX"]

METADATA_DIR = BASE_DIR / "metadata"
DOWNLOAD_LOG = METADATA_DIR / "download_log.csv"

MARKETS = [
    {
        "ticker": "EURUSD=X",
        "name": "EURUSD",
        "asset_class": "fx",
        "start": "2015-01-01",
        "end": END_DATE,
        "frequency": "1d",
    },
    {
        "ticker": "GBPUSD=X",
        "name": "GBPUSD",
        "asset_class": "fx",
        "start": "2015-01-01",
        "end": END_DATE,
        "frequency": "1d",
    },
    {
        "ticker": "^GSPC",
        "name": "SP500",
        "asset_class": "equity_indices",
        "start": "2010-01-01",
        "end": END_DATE,
        "frequency": "1d",
    },
    {
        "ticker": "^NDX",
        "name": "NASDAQ100",
        "asset_class": "equity_indices",
        "start": "2010-01-01",
        "end": END_DATE,
        "frequency": "1d",
    },
    {
        "ticker": "^GDAXI",
        "name": "DAX",
        "asset_class": "equity_indices",
        "start": "2010-01-01",
        "end": END_DATE,
        "frequency": "1d",
    },
    {
        "ticker": "^VIX",
        "name": "VIX",
        "asset_class": "volatility",
        "start": "2010-01-01",
        "end": END_DATE,
        "frequency": "1d",
    },
]


def append_download_log(item: dict, path: Path | None, status: str, message: str, rows: int | None = None) -> None:
    """Append one yfinance download log row."""
    header = [
        "timestamp_utc",
        "script",
        "asset_class",
        "symbol",
        "frequency",
        "source",
        "period",
        "path",
        "status",
        "message",
        "rows",
    ]
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "asset_class": item["asset_class"],
        "symbol": item["name"],
        "frequency": item["frequency"],
        "source": f"yfinance:{item['ticker']}",
        "period": f"{item['start']}_{item['end']}",
        "path": str(path) if path else "",
        "status": status,
        "message": message,
        "rows": rows if rows is not None else "",
    }
    write_header = not DOWNLOAD_LOG.exists()
    with DOWNLOAD_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def normalize_yfinance(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to lowercase thesis-friendly columns."""
    if frame.empty:
        return frame
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] for column in frame.columns]
    frame = frame.reset_index()
    rename = {
        "Date": "timestamp",
        "Datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    frame = frame.rename(columns=rename)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    keep = [column for column in ["timestamp", "open", "high", "low", "close", "adj_close", "volume"] if column in frame.columns]
    frame = frame[keep].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for column in [column for column in frame.columns if column != "timestamp"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def end_exclusive(end_date: str) -> str:
    """yfinance end dates are exclusive."""
    return (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def download_item(item: dict) -> None:
    """Download, normalize, and save one configured market dataset."""
    processed_dir = BASE_DIR / "processed" / item["asset_class"]
    raw_dir = BASE_DIR / "raw" / item["asset_class"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / f"{item['name']}_{item['frequency']}_{item['start'][:4]}_{item['end'][:4]}.parquet"
    raw_path = raw_dir / f"{item['name']}_{item['frequency']}_{item['start'][:4]}_{item['end'][:4]}_raw.parquet"
    if output_path.exists() and not OVERWRITE:
        print(f"[skip] {output_path} exists")
        append_download_log(item, output_path, "skipped", "Processed parquet exists")
        return

    print(f"[download] {item['name']} from yfinance ticker {item['ticker']}")
    raw = yf.download(
        item["ticker"],
        start=item["start"],
        end=end_exclusive(item["end"]),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    normalized = normalize_yfinance(raw)
    if normalized.empty:
        raise ValueError(f"yfinance returned no rows for {item['ticker']}")

    raw_normalized = normalize_yfinance(raw)
    raw_normalized.to_parquet(raw_path, engine="pyarrow", index=False)
    normalized.to_parquet(output_path, engine="pyarrow", index=False)
    append_download_log(item, output_path, "downloaded", "Saved yfinance processed parquet", len(normalized))
    print(f"[saved] {output_path} ({len(normalized):,} rows)")


def main() -> None:
    """Run all yfinance market downloads."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    for item in MARKETS:
        try:
            download_item(item)
        except Exception as exc:
            print(f"[error] {item['name']}: {exc}")
            append_download_log(item, None, "error", str(exc))


if __name__ == "__main__":
    main()
