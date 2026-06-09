"""Download Binance spot monthly 1-minute klines and build processed crypto parquet files."""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


BASE_DIR = Path(__file__).resolve().parents[1]
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
OVERWRITE = False
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

DOWNLOAD_OPTIONAL_CRYPTO = False
OPTIONAL_SYMBOLS = ["SOLUSDT"]
OPTIONAL_START_DATE = "2021-01-01"

FREQUENCY = "1m"
BINANCE_URL = (
    "https://data.binance.vision/data/spot/monthly/klines/"
    "{symbol}/1m/{symbol}-1m-{year}-{month:02d}.zip"
)

RAW_DIR = BASE_DIR / "raw" / "crypto" / "binance_1m"
PROCESSED_DIR = BASE_DIR / "processed" / "crypto" / "1m"
METADATA_DIR = BASE_DIR / "metadata"
DOWNLOAD_LOG = METADATA_DIR / "download_log.csv"

BINANCE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

OUTPUT_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def ensure_dirs() -> None:
    """Create required output directories."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in active_symbols():
        (RAW_DIR / symbol).mkdir(parents=True, exist_ok=True)


def active_symbols() -> list[str]:
    """Return required symbols plus optional robustness symbols when enabled."""
    symbols = list(SYMBOLS)
    if DOWNLOAD_OPTIONAL_CRYPTO:
        symbols.extend(OPTIONAL_SYMBOLS)
    return symbols


def make_session() -> requests.Session:
    """Create a requests session with conservative retry behavior."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def month_starts(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Return month starts covering the inclusive date window."""
    start = pd.Timestamp(start_date, tz="UTC").to_period("M").to_timestamp()
    end = pd.Timestamp(end_date, tz="UTC").to_period("M").to_timestamp()
    return pd.date_range(start=start, end=end, freq="MS")


def end_exclusive(end_date: str) -> pd.Timestamp:
    """Convert an inclusive date string into an exclusive UTC upper bound."""
    return pd.Timestamp(end_date, tz="UTC").normalize() + pd.Timedelta(days=1)


def append_download_log(
    symbol: str,
    period: str,
    path: Path | None,
    status: str,
    message: str,
    rows: int | None = None,
) -> None:
    """Append a lightweight CSV log row."""
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
        "asset_class": "crypto",
        "symbol": symbol,
        "frequency": FREQUENCY,
        "source": "Binance public data monthly spot klines",
        "period": period,
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


def parse_zip_payload(payload: bytes, symbol: str, period: str) -> pd.DataFrame:
    """Parse one Binance monthly ZIP payload into a typed DataFrame."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found in Binance ZIP for {symbol} {period}.")
        with archive.open(csv_names[0]) as csv_file:
            frame = pd.read_csv(csv_file, header=None)

    if str(frame.iloc[0, 0]).lower() in {"open_time", "timestamp"}:
        frame = frame.iloc[1:].reset_index(drop=True)

    if frame.shape[1] < len(BINANCE_COLUMNS) - 1:
        raise ValueError(f"Unexpected Binance column count {frame.shape[1]} for {symbol} {period}.")

    frame = frame.iloc[:, : len(BINANCE_COLUMNS)].copy()
    frame.columns = BINANCE_COLUMNS[: frame.shape[1]]

    if "ignore" not in frame.columns:
        frame["ignore"] = pd.NA

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    integer_columns = ["open_time", "close_time", "number_of_trades"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in integer_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")

    frame["timestamp"] = epoch_to_utc(frame["open_time"])
    frame["close_time"] = epoch_to_utc(frame["close_time"])
    frame["number_of_trades"] = frame["number_of_trades"].astype("Int64")

    frame = frame[OUTPUT_COLUMNS].sort_values("timestamp").drop_duplicates("timestamp")
    return frame


def epoch_to_utc(values: pd.Series) -> pd.Series:
    """Convert Binance epoch values to UTC, handling millisecond and microsecond files."""
    numeric = pd.to_numeric(values, errors="coerce")
    median_abs = numeric.dropna().abs().median()
    if pd.isna(median_abs):
        unit = "ms"
    elif median_abs >= 1e17:
        unit = "ns"
    elif median_abs >= 1e14:
        unit = "us"
    else:
        unit = "ms"
    return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")


def raw_month_is_valid(path: Path, month_start: pd.Timestamp) -> bool:
    """Check that an existing raw monthly parquet has plausible coverage for its filename."""
    try:
        frame = pd.read_parquet(path, columns=["timestamp"])
        if frame.empty:
            return False
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if timestamps.isna().all():
            return False
        expected_start = pd.Timestamp(month_start).tz_localize("UTC")
        expected_end = expected_start + pd.offsets.MonthBegin(1)
        actual_min = timestamps.min()
        actual_max = timestamps.max()
        return actual_min >= expected_start and actual_max < expected_end
    except Exception:
        return False


def download_month(session: requests.Session, symbol: str, month_start: pd.Timestamp) -> Path | None:
    """Download and save one monthly parquet file if needed."""
    period = f"{month_start.year}-{month_start.month:02d}"
    raw_path = RAW_DIR / symbol / f"{symbol}_1m_{period}.parquet"
    if raw_path.exists() and not OVERWRITE:
        if raw_month_is_valid(raw_path, month_start):
            print(f"[skip] {symbol} {period}: raw parquet exists")
            append_download_log(symbol, period, raw_path, "skipped", "Raw parquet already exists")
            return raw_path
        print(f"[repair] {symbol} {period}: existing raw parquet has invalid timestamp coverage; redownloading")

    url = BINANCE_URL.format(symbol=symbol, year=month_start.year, month=month_start.month)
    print(f"[download] {symbol} {period}: {url}")
    response = session.get(url, timeout=90)
    if response.status_code == 404:
        print(f"[missing] {symbol} {period}: Binance file not found")
        append_download_log(symbol, period, None, "missing", "HTTP 404 from Binance")
        return None
    response.raise_for_status()

    frame = parse_zip_payload(response.content, symbol, period)
    frame.to_parquet(raw_path, engine="pyarrow", index=False)
    append_download_log(symbol, period, raw_path, "downloaded", "Saved monthly raw parquet", len(frame))
    print(f"[saved] {raw_path} ({len(frame):,} rows)")
    return raw_path


def build_processed_symbol(symbol: str, start_date: str, end_date: str) -> Path:
    """Combine monthly raw parquet files into one processed symbol parquet."""
    output_path = PROCESSED_DIR / f"{symbol}_1m_{start_date[:4]}_{end_date[:4]}.parquet"
    if output_path.exists() and not OVERWRITE and processed_file_is_current(output_path, start_date, end_date):
        print(f"[skip] {symbol}: processed 1m parquet exists")
        append_download_log(symbol, f"{start_date}_{end_date}", output_path, "skipped", "Processed parquet exists")
        return output_path
    if output_path.exists() and not OVERWRITE:
        print(f"[repair] {symbol}: processed 1m parquet does not cover requested period; rebuilding")

    paths = sorted((RAW_DIR / symbol).glob(f"{symbol}_1m_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No raw monthly parquet files found for {symbol} in {RAW_DIR / symbol}.")

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = end_exclusive(end_date)
    frames = []
    for path in tqdm(paths, desc=f"Combining {symbol}", unit="file"):
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        mask = (frame["timestamp"] >= start_ts) & (frame["timestamp"] < end_ts)
        if mask.any():
            frames.append(frame.loc[mask])

    if not frames:
        raise ValueError(f"No rows for {symbol} inside {start_date} to {end_date}.")

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, engine="pyarrow", index=False)
    append_download_log(
        symbol,
        f"{start_date}_{end_date}",
        output_path,
        "processed",
        "Combined monthly raw files",
        len(combined),
    )
    print(f"[saved] {output_path} ({len(combined):,} rows)")
    return output_path


def processed_file_is_current(path: Path, start_date: str, end_date: str) -> bool:
    """Check that a processed file covers the requested inclusive date window."""
    try:
        frame = pd.read_parquet(path, columns=["timestamp"])
        if frame.empty:
            return False
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        requested_start = pd.Timestamp(start_date, tz="UTC")
        requested_end_last_minute = pd.Timestamp(end_date, tz="UTC").normalize() + pd.Timedelta(hours=23, minutes=59)
        return timestamps.min() <= requested_start and timestamps.max() >= requested_end_last_minute
    except Exception:
        return False


def main() -> None:
    """Run Binance crypto download and processed-file construction."""
    ensure_dirs()
    session = make_session()
    for symbol in active_symbols():
        symbol_start = OPTIONAL_START_DATE if symbol in OPTIONAL_SYMBOLS else START_DATE
        months = month_starts(symbol_start, END_DATE)
        for month_start in tqdm(months, desc=f"Downloading {symbol}", unit="month"):
            try:
                download_month(session, symbol, month_start)
            except Exception as exc:
                period = f"{month_start.year}-{month_start.month:02d}"
                print(f"[error] {symbol} {period}: {exc}")
                append_download_log(symbol, period, None, "error", str(exc))
        try:
            build_processed_symbol(symbol, symbol_start, END_DATE)
        except Exception as exc:
            print(f"[error] {symbol}: failed to build processed parquet: {exc}")
            append_download_log(symbol, f"{symbol_start}_{END_DATE}", None, "error", str(exc))


if __name__ == "__main__":
    main()
